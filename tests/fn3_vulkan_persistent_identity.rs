//! Persistent-weight identity across model lifetimes (release blocker 2).
//!
//! # What this proves
//!
//! The Vulkan backend is shared and outlives any one model. A persistent table
//! key must therefore identify *bytes*, not a host address: after a model is
//! dropped its address can be recycled by the allocator, and a table that keyed
//! by that address would silently serve the old model's weights to the new one.
//!
//! The fix gives every [`Storage`] a process-global monotonic generation
//! ([`Storage::generation`]) and releases a table entry with its last storage
//! handle. These tests pin four consequences:
//!
//! 1. two successive models with different bytes and identical geometry each
//!    run with their own weights (B matches B's CPU reference and differs from
//!    A; B's prepare uploads B's bytes, not zero);
//! 2. a second session over the *same* loaded artifact uploads nothing and
//!    produces byte-identical logits;
//! 3. repeated create/drop cycles leave the table empty, with the release
//!    counter accounting for every entry;
//! 4. a forced same-size create/drop loop cannot resolve the previous
//!    allocation's device copy.
//!
//! ```bash
//! FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
//! FN3_VK_REQUIRE_PHYSICAL=1 \
//! FASTNN_FN3_ONNX_MODEL=/path/to/model.onnx \
//!   cargo test --features vulkan --test fn3_vulkan_persistent_identity \
//!     -- --nocapture --test-threads=1
//! ```
//!
//! [`Storage`]: fastnn::inference::Storage
//! [`Storage::generation`]: fastnn::inference::Storage::generation

#![cfg(feature = "vulkan")]

use fastnn::inference::backend::vulkan::resident::ResidencyPhase;
use fastnn::inference::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
use fastnn::inference::device::DeviceRequest;
use fastnn::inference::import::ModelImport;
use fastnn::inference::model::{ModelWeights, Planner, PreparedModel, Session};
use fastnn::inference::{
    Backend, CpuBackend, DType, DeviceId, Shape, Storage, Tensor, TensorMeta, TensorView,
};

#[path = "fn3_e2e/harness.rs"]
mod harness;

use harness::*;

/// Scale applied to model B's learned parameters (`*.weight` / `*.bias`).
///
/// Chosen so B has the same shapes and byte lengths as A (the allocator sees
/// identical allocation sizes) but different bytes and a clearly different
/// result. Graph constants are copied unchanged; scaling the LayerNorm
/// exponent would change the operator, not the model.
const B_WEIGHT_SCALE: f32 = 0.5;

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
        eprintln!("[fn3vk][skip] no physical Vulkan device; the identity gate needs one");
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
    assert!(!backend.is_software(), "the identity gate needs a real GPU");
    Some(backend)
}

/// The persistent tensors `prepare_persistent` will keep: every contiguous,
/// non-empty weight. Mirrors `tests/fn3_vulkan_resident.rs`.
fn expected_persistent(weights: &ModelWeights) -> (u64, u64) {
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

/// Retag every weight for `device` without copying bytes.
fn retag(weights: &ModelWeights, device: DeviceId) -> ModelWeights {
    let mut named = Vec::with_capacity(weights.len());
    for index in 0..weights.len() {
        let tensor = weights.get(index).expect("weight index");
        let meta = TensorMeta::contiguous(tensor.shape().clone(), tensor.dtype(), device)
            .expect("weight metadata");
        named.push((
            weights.name(index).expect("weight name").to_string(),
            Tensor::from_storage(tensor.storage().clone(), meta).expect("re-tagged weight"),
        ));
    }
    ModelWeights::with_names(named).expect("weights share one device")
}

/// A fresh copy of every weight with the model parameters scaled.
///
/// Only tensors named `*.weight` / `*.bias` are scaled. The importer also
/// lifts graph constants into the weight table (a causal-mask fill value, the
/// exponent and epsilon of LayerNorm), and scaling those would change the
/// operator semantics rather than the learned parameters. Non-`f32` entries
/// keep their bytes. Every tensor owns a new allocation, so dropping the
/// returned set releases model B's bytes and nothing else.
fn perturbed(weights: &ModelWeights) -> ModelWeights {
    let mut named = Vec::with_capacity(weights.len());
    for index in 0..weights.len() {
        let tensor = weights.get(index).expect("weight index");
        let name = weights.name(index).expect("weight name").to_string();
        let parameter = name.ends_with(".weight") || name.ends_with(".bias");
        let copy = if tensor.dtype() == DType::F32 && parameter {
            let mut values = tensor
                .as_f32_slice()
                .expect("host f32 weight is contiguous")
                .to_vec();
            for value in &mut values {
                *value *= B_WEIGHT_SCALE;
            }
            Tensor::from_f32(tensor.shape().clone(), &values).expect("scaled f32 weight")
        } else {
            let meta =
                TensorMeta::contiguous(tensor.shape().clone(), tensor.dtype(), DeviceId::Cpu)
                    .expect("weight metadata");
            let storage =
                Storage::from_bytes(tensor.storage().as_bytes()).expect("weight byte copy");
            Tensor::from_storage(storage, meta).expect("copied weight")
        };
        named.push((name, copy));
    }
    ModelWeights::with_names(named).expect("weights share one device")
}

/// Build the pinned model on `device` from `weights`.
///
/// The plan depends on the import geometry, not on the weight values, so two
/// models with identical shapes but different bytes share this path.
fn build_model(import: &ModelImport, weights: ModelWeights, device: DeviceId) -> PreparedModel {
    let cpu = CpuBackend::new();
    let planner = Planner::new(import, &cpu);
    let envelope = harness::envelope(&planner, FIXTURE_CONTEXT);
    let config = import.geometry().kv_cache_config(FIXTURE_CONTEXT, 1);
    let plan = planner
        .plan(&envelope, harness::dims(4, 0))
        .expect("prefill plan")
        .execution()
        .clone();
    PreparedModel::new(plan, weights, device, envelope, Some(config)).expect("prepared model")
}

fn prefill(session: &mut Session<'_>, tokens: &[u32]) -> Vec<f32> {
    session
        .prefill(&tokens[..PROMPT])
        .expect("prefill")
        .as_f32_slice()
        .expect("host logits")
        .to_vec()
}

fn assert_validation_clean() {
    let (errors, warnings) = fastnn::inference::backend::vulkan::instance::validation_counts();
    assert_eq!(
        (errors, warnings),
        (0, 0),
        "validation errors/warnings: {errors}/{warnings}"
    );
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "logit length");
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Model A, then model B with the same geometry and different bytes.
///
/// The two models are built from the same import (identical shapes and byte
/// lengths), so B's host allocations are the sizes A just released. The
/// identities recorded from A must not recur on B; if the table keyed by an
/// address instead, B would hit A's entries and produce A's logits.
#[test]
fn a_reused_backend_does_not_serve_a_dropped_models_weights() {
    let Some(backend) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let device = backend.device_id();
    let tokens = token_ids();

    // ---- model A: the pinned weights ------------------------------------
    let weights_a = import.to_model_weights().expect("A host weights");
    let identities_a: Vec<u64> = (0..weights_a.len())
        .map(|index| {
            weights_a
                .get(index)
                .expect("A weight")
                .storage()
                .generation()
        })
        .collect();
    let model_a = build_model(&import, retag(&weights_a, device), device);
    drop(weights_a);
    let (count_a, bytes_a) = expected_persistent(model_a.weights());

    backend.reset_residency_stats();
    backend.set_phase(ResidencyPhase::Prepare);
    let mut session_a = Session::with_runtime(&model_a, &import, &*backend).expect("session A");
    let prepare_a = backend.residency_stats();
    assert_eq!(
        prepare_a.persistent_prepare_bytes, bytes_a,
        "model A must upload its own weights"
    );
    assert_eq!(prepare_a.persistent_tensors, count_a);
    let logits_a = prefill(&mut session_a, &tokens);

    // ---- drop A completely: its bytes, its sessions and its table entries ----
    drop(session_a);
    drop(model_a);
    assert_eq!(
        backend.persistent_tensor_count(),
        0,
        "dropping model A must release its persistent entries"
    );
    assert_eq!(
        backend.persistent_live_bytes(),
        0,
        "dropping model A must release its device bytes"
    );

    // ---- model B: identical geometry, different bytes --------------------
    let source_b = import.to_model_weights().expect("B source weights");
    let weights_b = perturbed(&source_b);
    drop(source_b);
    let identities_b: Vec<u64> = (0..weights_b.len())
        .map(|index| {
            weights_b
                .get(index)
                .expect("B weight")
                .storage()
                .generation()
        })
        .collect();
    assert!(
        identities_a
            .iter()
            .all(|identity| !identities_b.contains(identity)),
        "a storage generation was reused across models: the table key would alias"
    );

    // B's CPU reference, built from duplicate bytes with the same scale.
    let cpu_weights = perturbed(&import.to_model_weights().expect("B source weights"));
    let cpu_model = build_model(&import, cpu_weights, DeviceId::Cpu);
    let cpu_backend = CpuBackend::new();
    let mut cpu_session =
        Session::with_runtime(&cpu_model, &import, &cpu_backend).expect("CPU session B");
    let reference_b = prefill(&mut cpu_session, &tokens);
    drop(cpu_session);
    drop(cpu_model);

    let model_b = build_model(&import, retag(&weights_b, device), device);
    drop(weights_b);
    let (count_b, bytes_b) = expected_persistent(model_b.weights());

    backend.reset_residency_stats();
    backend.set_phase(ResidencyPhase::Prepare);
    let mut session_b = Session::with_runtime(&model_b, &import, &*backend).expect("session B");
    let prepare_b = backend.residency_stats();
    assert_eq!(
        prepare_b.persistent_prepare_bytes, bytes_b,
        "model B must upload its own weights, not resolve model A's entries"
    );
    assert_eq!(
        prepare_b.persistent_tensors, count_b,
        "a model B weight missed the persistent table"
    );
    let logits_b = prefill(&mut session_b, &tokens);

    // B is B: numerically the same as its own CPU reference, decisively
    // different from A.
    harness::compare_logits("model B / Vulkan vs model B / CPU", &logits_b, &reference_b);
    let b_vs_a = max_abs(&logits_b, &logits_a);
    assert!(
        b_vs_a > 1e-2,
        "model B produced model A's logits (max_abs {b_vs_a:e}); the persistent \
         table served a dropped model's weights"
    );
    println!(
        "[persistent-identity][successive] A prepare {bytes_a} B / {count_a} tensors; \
         B prepare {bytes_b} B / {count_b} tensors; B_vs_CPU max_abs {}; B_vs_A max_abs {b_vs_a:e}",
        max_abs(&logits_b, &reference_b),
    );

    drop(session_b);
    drop(model_b);
    assert_eq!(
        backend.persistent_tensor_count(),
        0,
        "dropping model B must release its persistent entries"
    );
    assert_eq!(
        backend.persistent_release_count(),
        count_a + count_b,
        "every entry released with its owner must be counted once"
    );
    assert_validation_clean();
}

/// Two sessions over one loaded artifact: the second uploads nothing and
/// produces byte-identical logits.
#[test]
fn a_second_session_over_the_same_artifact_uploads_nothing() {
    let Some(backend) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let device = backend.device_id();
    let tokens = token_ids();

    let weights = import.to_model_weights().expect("host weights");
    let model = build_model(&import, retag(&weights, device), device);
    drop(weights);
    let (count, bytes) = expected_persistent(model.weights());

    backend.reset_residency_stats();
    backend.set_phase(ResidencyPhase::Prepare);
    let mut first = Session::with_runtime(&model, &import, &*backend).expect("first session");
    let prepare = backend.residency_stats();
    assert_eq!(prepare.persistent_prepare_bytes, bytes);
    assert_eq!(prepare.persistent_tensors, count);
    let first_logits = prefill(&mut first, &tokens);

    backend.reset_residency_stats();
    backend.set_phase(ResidencyPhase::Prepare);
    let mut second = Session::with_runtime(&model, &import, &*backend).expect("second session");
    let reprepare = backend.residency_stats();
    assert_eq!(
        reprepare.persistent_prepare_bytes, 0,
        "a second session over one loaded model re-uploaded a weight"
    );
    assert_eq!(
        reprepare.persistent_tensors, 0,
        "a second session over one loaded model inserted a duplicate entry"
    );
    let second_logits = prefill(&mut second, &tokens);
    assert_eq!(
        first_logits, second_logits,
        "a cache hit changed the result; the weights were not the model's"
    );
    println!(
        "[persistent-identity][reuse] prepare {bytes} B then 0 B; table {} entries; \
         logits byte-identical over {} values",
        backend.persistent_tensor_count(),
        first_logits.len()
    );
    assert_eq!(backend.persistent_tensor_count(), count as usize);

    drop(first);
    drop(second);
    assert_eq!(backend.persistent_tensor_count(), count as usize);
    assert_validation_clean();
}

/// Repeated model create/prepare/drop cycles cannot grow the table.
#[test]
fn dropping_models_releases_persistent_entries() {
    let Some(backend) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let device = backend.device_id();

    let baseline = backend.persistent_release_count();
    let cycles = 4u64;
    let mut expected = (0u64, 0u64);
    for cycle in 0..cycles {
        let weights = import.to_model_weights().expect("host weights");
        let model = build_model(&import, retag(&weights, device), device);
        drop(weights);
        let per_cycle = expected_persistent(model.weights());
        if cycle == 0 {
            expected = per_cycle;
        } else {
            assert_eq!(per_cycle, expected, "cycle {cycle} geometry changed");
        }

        let session = Session::with_runtime(&model, &import, &*backend).expect("session in cycle");
        assert_eq!(
            backend.persistent_tensor_count(),
            expected.0 as usize,
            "cycle {cycle}: prepared table size"
        );
        assert_eq!(
            backend.persistent_live_bytes(),
            expected.1,
            "cycle {cycle}: live persistent bytes"
        );

        drop(session);
        drop(model);
        assert_eq!(
            backend.persistent_tensor_count(),
            0,
            "cycle {cycle}: entries outlived their model"
        );
        assert_eq!(
            backend.persistent_live_bytes(),
            0,
            "cycle {cycle}: device bytes outlived their model"
        );
    }
    let releases = backend.persistent_release_count() - baseline;
    assert_eq!(
        releases,
        expected.0 * cycles,
        "the release counter must account for every entry"
    );
    assert_eq!(
        backend.persistent_tensor_count(),
        0,
        "the table must be empty after the last drop, not merely bounded"
    );
    println!(
        "[persistent-identity][cleanup] {cycles} cycles, {releases} entries released, \
         table {} entries / {} B after the last drop",
        backend.persistent_tensor_count(),
        backend.persistent_live_bytes()
    );
    assert_validation_clean();
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Read the device buffer `resident_input` resolves for `tensor`.
fn resident_bytes(backend: &VulkanBackend, tensor: &Tensor, len: usize) -> Vec<u8> {
    let view = TensorView::new(tensor.storage(), tensor.meta().clone()).expect("tensor view");
    let buffer = backend
        .resident_input(&view)
        .expect("resident input")
        .expect("a persistent buffer");
    let mut bytes = vec![0u8; len];
    backend.download(&buffer, &mut bytes).expect("download");
    bytes
}

/// The forcing case for an address key: allocate, prepare, drop, then allocate
/// the *same size* again in a loop.
///
/// A host allocator hands the freed block back quickly, which is the alias an
/// address key cannot see. A generation key must upload the replacement every
/// time and resolve the replacement's own buffer. The loop is what makes the
/// reuse likely; the assertions do not depend on it happening.
#[test]
fn a_dropped_host_block_cannot_lend_its_device_copy() {
    let Some(backend) = open_physical() else {
        return;
    };
    let shape = Shape::from_dims([2048u64]);
    let a_values: Vec<f32> = (0..2048).map(|index| 1.0 + index as f32 * 0.25).collect();
    let b_values: Vec<f32> = (0..2048).map(|index| -3.0 - index as f32 * 0.5).collect();
    let bytes = 2048 * 4;

    let a = Tensor::from_f32(shape.clone(), &a_values).expect("A tensor");
    backend.prepare_persistent(&[&a]).expect("A prepare");
    assert_eq!(backend.persistent_tensor_count(), 1);
    assert_eq!(
        resident_bytes(&backend, &a, bytes),
        f32_bytes(&a_values),
        "A's device copy is A's bytes"
    );
    drop(a);
    assert_eq!(
        backend.persistent_tensor_count(),
        0,
        "the dropped tensor's entry must be released"
    );

    for attempt in 0..64 {
        let b = Tensor::from_f32(shape.clone(), &b_values).expect("B tensor");
        backend.reset_residency_stats();
        backend.set_phase(ResidencyPhase::Prepare);
        backend.prepare_persistent(&[&b]).expect("B prepare");
        let stats = backend.residency_stats();
        assert_eq!(
            stats.persistent_prepare_bytes, bytes as u64,
            "attempt {attempt}: B resolved a dropped allocation's entry instead of uploading"
        );
        assert_eq!(stats.persistent_tensors, 1);
        assert_eq!(
            resident_bytes(&backend, &b, bytes),
            f32_bytes(&b_values),
            "attempt {attempt}: B resolved a dropped allocation's bytes"
        );
        drop(b);
        assert_eq!(
            backend.persistent_tensor_count(),
            0,
            "attempt {attempt}: entry outlived its tensor"
        );
    }
    println!(
        "[persistent-identity][forced] 64 same-size create/drop cycles: every prepare uploaded \
         {bytes} B and resolved its own bytes"
    );
    assert_validation_clean();
}
