//! FN3-26 (#72) step 1: persistent weights on the physical device.
//!
//! # What this proves
//!
//! `Session::with_runtime` asks the backend to keep every immutable weight on
//! the device once (`Backend::prepare_persistent`). After that, a weight bind is
//! a handle lookup: no host bytes move. The claim is not read off a trace — the
//! staging path *rejects* a prepared tensor
//! (`VulkanBackend::check_staged_input`), so a kernel that forgets the resident
//! path fails instead of quietly re-uploading 28 MB.
//!
//! The test therefore asserts, per phase:
//!
//! | Counter | prepare | prefill | decode |
//! |---|---|---|---|
//! | `persistent_tensors` | 48 | unchanged | unchanged |
//! | `persistent_prepare_bytes` | every weight byte | **0** | **0** |
//! | `persistent_hits[phase]` | — | **> 0** | **> 0** |
//! | `late_persistent_uploads` | **0** | **0** | **0** |
//!
//! and that the run still produces device execution evidence and the fixture
//! parity the end-to-end gate already checks (this binary only adds the
//! persistence counters, so a failure here is a residency failure).
//!
//! ```bash
//! FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
//! FN3_VK_REQUIRE_PHYSICAL=1 \
//! FASTNN_FN3_ONNX_MODEL=/path/to/model.onnx \
//!   cargo test --features vulkan --test fn3_vulkan_resident -- --nocapture --test-threads=1
//! ```

#![cfg(feature = "vulkan")]

use fastnn::fn3::backend::vulkan::resident::ResidencyPhase;
use fastnn::fn3::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
use fastnn::fn3::device::DeviceRequest;
use fastnn::fn3::Session;

#[path = "fn3_e2e/harness.rs"]
mod harness;

use harness::*;

fn require_physical() -> bool {
    std::env::var("FN3_VK_REQUIRE_PHYSICAL").as_deref() == Ok("1")
}

fn pinned_uuid() -> Option<String> {
    let value = std::env::var("FN3_VK_DEVICE_UUID").ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_ascii_lowercase())
}

fn physical_adapter() -> Option<u32> {
    let pinned = pinned_uuid();
    adapters()
        .iter()
        .find(|adapter| {
            adapter.has_compute_queue()
                && !adapter.is_software()
                && pinned
                    .as_deref()
                    .map(|uuid| adapter.uuid_string().eq_ignore_ascii_case(uuid))
                    .unwrap_or(true)
        })
        .map(|adapter| adapter.index)
}

fn open_physical() -> Option<std::sync::Arc<VulkanBackend>> {
    let Some(index) = physical_adapter() else {
        if require_physical() {
            panic!(
                "FN3_VK_REQUIRE_PHYSICAL=1 but no physical Vulkan device with a compute queue is \
                 present"
            );
        }
        eprintln!("[fn3vk][skip] no physical Vulkan device; the residency gate needs one");
        return None;
    };
    let backend = VulkanBackend::open_with(
        &DeviceRequest::Vulkan { index: Some(index) },
        VulkanConfig::validated(),
    )
    .unwrap_or_else(|error| {
        if require_physical() {
            panic!("FN3_VK_REQUIRE_PHYSICAL=1 but adapter {index} failed to open: {error}");
        }
        panic!("opening adapter {index} failed: {error}");
    });
    assert!(
        !backend.is_software(),
        "the residency gate needs a real GPU"
    );
    Some(backend)
}

/// The persistent tensors `prepare_persistent` will keep: every contiguous,
/// non-empty weight (a zero-byte or strided immutable tensor is not bindable as
/// one buffer and deliberately keeps the staged path).
fn expected_persistent(weights: &fastnn::fn3::ModelWeights) -> (u64, u64) {
    let mut count = 0u64;
    let mut bytes = 0u64;
    for index in 0..weights.len() {
        let Some(tensor) = weights.get(index) else {
            continue;
        };
        if !tensor.is_contiguous() {
            continue;
        }
        let len = tensor.meta().byte_len().expect("weight byte length") as u64;
        if len == 0 {
            continue;
        }
        count += 1;
        bytes += len;
    }
    (count, bytes)
}

#[test]
fn persistent_weights_upload_once_and_are_rebound_by_handle() {
    let Some(backend) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let device = backend.device_id();
    let (model, _envelope) = prepared(&import, device, FIXTURE_CONTEXT);
    let (expected_tensors, expected_bytes) = expected_persistent(model.weights());
    let tokens = token_ids();

    // ---- prepare ---------------------------------------------------------
    backend.reset_residency_stats();
    backend.set_phase(ResidencyPhase::Prepare);
    let mut session = Session::with_runtime(&model, &import, &*backend).expect("session");
    let prepare = backend.residency_stats();
    assert_eq!(
        prepare.persistent_tensors, expected_tensors,
        "every contiguous weight must be device-resident after prepare"
    );
    assert_eq!(
        prepare.persistent_prepare_bytes, expected_bytes,
        "prepare must move exactly the weight bytes"
    );
    assert_eq!(prepare.late_persistent_uploads, 0);
    println!(
        "[fn3-26][prepare] persistent_tensors={} persistent_bytes={} prepare_upload_bytes={}",
        prepare.persistent_tensors, prepare.persistent_bytes, prepare.persistent_prepare_bytes
    );

    // ---- a second session over the same model uploads nothing -----------
    backend.reset_residency_stats();
    backend.set_phase(ResidencyPhase::Prepare);
    let mut second = Session::with_runtime(&model, &import, &*backend).expect("second session");
    let re_prepare = backend.residency_stats();
    assert_eq!(
        re_prepare.persistent_prepare_bytes, 0,
        "a second session over one model must not re-upload a weight"
    );
    assert_eq!(re_prepare.late_persistent_uploads, 0);

    // ---- prefill ---------------------------------------------------------
    backend.reset_residency_stats();
    let submissions_before = backend.queue_submissions();
    let logits = session
        .prefill(&tokens[..PROMPT])
        .expect("prefill with persistent weights");
    let prefill_submissions = backend.queue_submissions() - submissions_before;
    let prefill = backend.residency_stats();
    assert_eq!(
        prefill.persistent_prepare_bytes, 0,
        "prefill uploaded a persistent tensor"
    );
    assert_eq!(
        prefill.late_persistent_uploads, 0,
        "a prepared weight reached the staging path during prefill"
    );
    assert!(
        prefill.persistent_hits[ResidencyPhase::Prefill.index()] > 0,
        "prefill bound no weight from the persistent table"
    );
    assert!(
        prefill.staged_upload_bytes[ResidencyPhase::Prefill.index()] > 0,
        "the graph inputs (tokens/positions/mask) still travel host -> device"
    );
    assert_eq!(logits.shape().dims()[2], VOCAB as u64);
    let summary = backend.residency().summary();
    assert_eq!(summary.input_readbacks, 0);
    assert_eq!(summary.proven_device_executions, summary.executions);
    println!(
        "[fn3-26][prefill] queue_submissions={prefill_submissions} persistent_hits={} \
         staged_uploads={} staged_upload_bytes={} late={} executions={} ledger_submissions={} \
         upload_bytes={} readback_bytes={}",
        prefill.persistent_hits[ResidencyPhase::Prefill.index()],
        prefill.staged_uploads[ResidencyPhase::Prefill.index()],
        prefill.staged_upload_bytes[ResidencyPhase::Prefill.index()],
        prefill.late_persistent_uploads,
        summary.executions,
        summary.submissions,
        summary.upload_bytes,
        summary.readback_bytes,
    );

    // ---- decode ----------------------------------------------------------
    backend.residency().clear();
    backend.reset_residency_stats();
    let submissions_before = backend.queue_submissions();
    let mut decoded = Vec::new();
    for token in [640u32, 11, 612] {
        let logits = session
            .decode(&[token])
            .expect("decode with persistent weights");
        decoded.push(logits);
    }
    let decode_submissions = backend.queue_submissions() - submissions_before;
    let decode = backend.residency_stats();
    assert_eq!(
        decode.persistent_prepare_bytes, 0,
        "decode uploaded a persistent tensor"
    );
    assert_eq!(
        decode.late_persistent_uploads, 0,
        "a prepared weight reached the staging path during decode"
    );
    assert!(
        decode.persistent_hits[ResidencyPhase::Decode.index()] > 0,
        "decode bound no weight from the persistent table"
    );
    assert_eq!(session.cache_len(), (PROMPT + 3) as u64);
    assert_eq!(session.step(), 3);
    let summary = backend.residency().summary();
    assert_eq!(summary.input_readbacks, 0);
    assert_eq!(summary.proven_device_executions, summary.executions);
    println!(
        "[fn3-26][decode] queue_submissions={decode_submissions} persistent_hits={} \
         staged_uploads={} staged_upload_bytes={} late={} executions={} ledger_submissions={} \
         upload_bytes={} readback_bytes={}",
        decode.persistent_hits[ResidencyPhase::Decode.index()],
        decode.staged_uploads[ResidencyPhase::Decode.index()],
        decode.staged_upload_bytes[ResidencyPhase::Decode.index()],
        decode.late_persistent_uploads,
        summary.executions,
        summary.submissions,
        summary.upload_bytes,
        summary.readback_bytes,
    );

    // ---- the second session produces the same bytes ----------------------
    let second_logits = second
        .prefill(&tokens[..PROMPT])
        .expect("second prefill with persistent weights");
    let first = logits.as_f32_slice().expect("host logits");
    let again = second_logits.as_f32_slice().expect("host logits");
    assert_eq!(first, again, "persistent weights changed the result");

    let (errors, warnings) = fastnn::fn3::backend::vulkan::instance::validation_counts();
    assert_eq!((errors, warnings), (0, 0), "validation counts");
    println!("[fn3-26][validation] errors {errors} warnings {warnings}");
}
