//! FN3-27 (#73): repeated reset on the physical device, without injection.
//!
//! This is the **hardware** half of the repeated-reset evidence. It compiles
//! and runs with only `--features vulkan`, so no fault-injection code is
//! present in the binary; every failure path is the real one. The injected
//! reset matrix (which also arms a failure and recovers) lives in
//! `tests/fn3_vulkan_faults.rs` and is labelled separately.
//!
//! For eight rounds it prefills the pinned 13-token prompt, decodes to extents
//! 14..17 with the resident path (the default), checks the committed logits and
//! KV fixtures, then resets. After the round that warmed every plan it asserts
//! that the extents, KV bytes, plan cache and allocator counters return to the
//! same baseline on every reset.
//!
//! ```bash
//! flock /tmp/opencode/fn3-gpu.lock \
//! FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
//! FN3_VK_REQUIRE_PHYSICAL=1 \
//! FASTNN_FN3_ONNX_MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx \
//!   cargo test --features vulkan --test fn3_vulkan_reset -- --nocapture --test-threads=1
//! ```

#![cfg(feature = "vulkan")]

use std::sync::Mutex;

use fastnn::fn3::backend::vulkan::instance::validation_counts;
use fastnn::fn3::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
use fastnn::fn3::device::DeviceRequest;
use fastnn::fn3::model::{SessionOptions, SessionPhase};
use fastnn::fn3::Session;

#[path = "fn3_e2e/harness.rs"]
mod harness;

use harness::*;

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

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
                 present (pinned uuid {:?})",
                pinned_uuid()
            );
        }
        eprintln!("[fn3-27][reset][skip] no physical Vulkan device; the reset gate needs one");
        return None;
    };
    let backend = VulkanBackend::open_with(
        &DeviceRequest::Vulkan { index: Some(index) },
        VulkanConfig::validated(),
    )
    .unwrap_or_else(|error| panic!("opening adapter {index} failed: {error}"));
    assert!(!backend.is_software(), "the reset gate needs a real GPU");
    assert!(
        backend.validation_enabled(),
        "the reset gate runs with validation enabled in code"
    );
    eprintln!(
        "[fn3-27][reset][device] name=\"{}\" uuid={} validation={}",
        backend.adapter().name,
        backend.adapter().uuid_string(),
        backend.validation_enabled()
    );
    Some(backend)
}

#[test]
fn repeated_reset_on_the_physical_device_is_stable_and_preserves_parity() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();
    let (model, _envelope) = prepared(&import, device, FIXTURE_CONTEXT);
    let tokens = token_ids();
    let mut session =
        Session::with_runtime_options(&model, &import, &*backend, SessionOptions::default())
            .expect("session");
    assert!(
        session.caches()[0].is_device_resident(),
        "the hardware reset gate must run the resident path"
    );
    let kv_bytes = session.kv_bytes();
    let expected_kv_bytes = LAYERS as u64 * 2 * KV_HEADS * FIXTURE_CONTEXT * HEAD_DIM * 4;
    assert_eq!(kv_bytes, expected_kv_bytes);

    let mut plan_baseline = None;
    let mut memory_baseline = None;
    for round in 0..8usize {
        let logits = session.prefill(&tokens[..PROMPT]).expect("prefill");
        compare_logits(
            &format!("prefill.p13.last_logits [reset round {round}]"),
            logits.as_f32_slice().unwrap(),
            &fixture("prefill.p13.last_logits"),
        );
        for layer in 0..LAYERS {
            compare_kv(
                &format!("prefill.p13.present.{layer}.key"),
                "reset",
                &session.caches()[layer].live_keys().unwrap(),
                PROMPT as u64,
            );
            compare_kv(
                &format!("prefill.p13.present.{layer}.value"),
                "reset",
                &session.caches()[layer].live_values().unwrap(),
                PROMPT as u64,
            );
        }
        for step in 0..4usize {
            let logits = session.decode(&[tokens[PROMPT + step]]).expect("decode");
            if step == 0 {
                compare_logits(
                    &format!("decode.step1.logits [reset round {round}]"),
                    logits.as_f32_slice().unwrap(),
                    &fixture("decode.step1.logits"),
                );
            } else if let Some(reference) = summary(&format!("decode.step{}.logits", step + 1)) {
                compare_summary(
                    &format!("decode.step{} [reset round {round}]", step + 1),
                    logits.as_f32_slice().unwrap(),
                    &reference,
                );
            }
            assert_eq!(session.cache_len(), 14 + step as u64);
        }
        assert_eq!(session.cache_len(), 17);
        assert_eq!(session.step(), 4);
        assert_eq!(session.tokens_consumed(), 17);
        assert_eq!(session.phase(), SessionPhase::Ready);
        for layer in 0..LAYERS {
            compare_kv(
                &format!("decode.step4.present.{layer}.key"),
                "reset",
                &session.caches()[layer].live_keys().unwrap(),
                FIXTURE_CONTEXT,
            );
            compare_kv(
                &format!("decode.step4.present.{layer}.value"),
                "reset",
                &session.caches()[layer].live_values().unwrap(),
                FIXTURE_CONTEXT,
            );
        }

        session.reset();
        assert_eq!(session.cache_len(), 0, "round {round}: extent after reset");
        assert_eq!(session.step(), 0, "round {round}: step after reset");
        assert_eq!(
            session.tokens_consumed(),
            0,
            "round {round}: tokens after reset"
        );
        assert_eq!(
            session.phase(),
            SessionPhase::Empty,
            "round {round}: phase after reset"
        );
        for (layer, cache) in session.caches().iter().enumerate() {
            assert_eq!(
                cache.live_len(),
                0,
                "round {round}: layer {layer} extent after reset"
            );
        }
        assert_eq!(
            session.kv_bytes(),
            kv_bytes,
            "round {round}: KV bytes moved"
        );

        let plans = (session.plan_cache_len(), session.plan_evictions());
        let memory = backend.memory_stats();
        let memory = (
            memory.live_bytes,
            memory.live_device_local_bytes,
            memory.peak_live_bytes,
        );
        match (plan_baseline, memory_baseline) {
            (None, None) => {
                plan_baseline = Some(plans);
                memory_baseline = Some(memory);
                println!(
                    "[fn3-27][reset] baseline after round 0: plans {plans:?}, live_bytes {}, \
                     live_device_local {}, peak_live {}",
                    memory.0, memory.1, memory.2
                );
            }
            (Some(plans0), Some(memory0)) => {
                assert_eq!(
                    plans, plans0,
                    "round {round}: the plan cache is not stable across resets"
                );
                assert_eq!(
                    (memory.0, memory.1),
                    (memory0.0, memory0.1),
                    "round {round}: live device memory did not return to the reset baseline"
                );
                assert!(
                    memory.2 >= memory0.2,
                    "round {round}: the peak allocator counter went backwards"
                );
            }
            _ => unreachable!(),
        }
        // `recover` on a healthy, freshly reset session is a no-op boundary.
        session.recover().expect("recover on a reset session");
    }

    let plans = (session.plan_cache_len(), session.plan_evictions());
    assert_eq!(plans, plan_baseline.unwrap());
    assert_eq!(plans.1, 0, "a repeated reset evicted a plan");
    println!(
        "[fn3-27][reset] hardware reset gate: 8 rounds at extents 14..17 on the resident path, \
         parity fixtures checked every round, extents 0 / KV {kv_bytes} B / plans {plans:?} stable"
    );
    let (errors, warnings) = validation_counts();
    assert_eq!(
        (errors, warnings),
        (0, 0),
        "absolute validation counts must be zero"
    );
    assert_eq!(
        validation_counts(),
        counts_before,
        "the reset gate changed the validation counters"
    );
    println!("[fn3-27][reset][validation] absolute errors {errors} warnings {warnings}, delta 0/0");
}
