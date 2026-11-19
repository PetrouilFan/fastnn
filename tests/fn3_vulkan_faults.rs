//! FN3-27 (#73) runtime hardening: the injected-failure matrix.
//!
//! # What this binary is, and is not
//!
//! Every failure here is **injected deterministically** at a named platform
//! seam (`vulkan-fault-injection`) or simulated with a synthetic adapter. The
//! output is prefixed `[fn3-27][injected]` and must never be quoted as
//! physical-device evidence. The physical evidence is the unmodified
//! `fn3_e2e`, `fn3_vulkan_resident` and repeated-reset runs recorded in
//! `docs/reports/fn3-27-runtime-hardening.md`.
//!
//! The matrix still runs *on* the GTX 1080 Ti (under `flock` and with
//! validation on) because the contract is about the real backend's state, not
//! a mock's.
//!
//! # The contract each case asserts
//!
//! | Failure | Class | Session after | Retry |
//! |---|---|---|---|
//! | allocation before submission | recoverable | unchanged | same logits |
//! | readback (logits download) | recoverable | unchanged | same logits |
//! | KV past-prefix copy | recoverable | unchanged | same logits |
//! | KV append mid multi-layer commit | recoverable | every extent rolled back | same logits, no double append |
//! | `vkQueueSubmit` (device lost) | terminal | unchanged, poisoned | typed terminal error |
//! | fence wait (device lost) | terminal | unchanged, poisoned | typed terminal error |
//!
//! ```bash
//! flock /tmp/opencode/fn3-measure.lock env \
//! FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
//! FN3_VK_REQUIRE_PHYSICAL=1 \
//! FASTNN_FN3_ONNX_MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx \
//!   cargo test --features vulkan,vulkan-fault-injection \
//!   --test fn3_vulkan_faults -- --nocapture --test-threads=1
//! ```

#![cfg(all(feature = "vulkan", feature = "vulkan-fault-injection"))]

use std::sync::Mutex;

use fastnn::inference::backend::vulkan::instance::validation_counts;
use fastnn::inference::backend::vulkan::{
    adapters, BufferDesc, FaultPoint, VulkanBackend, VulkanConfig,
};
use fastnn::inference::device::DeviceRequest;
use fastnn::inference::model::{SessionOptions, SessionPhase};
use fastnn::inference::{Fn3Error, Session};

#[path = "fn3_e2e/harness.rs"]
mod harness;

use harness::*;

/// Valid token ids beyond the 17 committed ones, so the matrix can reach
/// extent 19 without inventing a tokeniser.
const EXTRA_TOKEN: u32 = 50_256;
const EXTRA_TOKEN_2: u32 = 50_255;

/// Serialises physical-GPU use and the validation-count deltas.
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
        eprintln!(
            "[fn3-27][injected][skip] no physical Vulkan device; the injected matrix needs one"
        );
        return None;
    };
    let backend = VulkanBackend::open_with(
        &DeviceRequest::Vulkan { index: Some(index) },
        VulkanConfig::validated(),
    )
    .unwrap_or_else(|error| panic!("opening adapter {index} failed: {error}"));
    assert!(!backend.is_software(), "the matrix needs a real GPU");
    eprintln!(
        "[fn3-27][injected][device] name=\"{}\" uuid={} validation={}",
        backend.adapter().name,
        backend.adapter().uuid_string(),
        backend.validation_enabled()
    );
    Some(backend)
}

#[test]
fn injected_allocation_oom_reports_buffer_memory_requirements() {
    let _lock = gpu_lock();
    let counts_before = validation_counts();
    let Some(backend) = open_physical() else {
        return;
    };
    let desc = BufferDesc::device_storage(1025, "oom-byte-reporting");
    let baseline = backend.alloc_buffer(&desc).expect("baseline buffer");
    // SAFETY: the live buffer belongs to this device and stays alive through
    // the read-only requirements query, which takes no host pointer.
    let requirements = unsafe {
        backend
            .vulkan_device()
            .ash()
            .get_buffer_memory_requirements(baseline.raw())
    };
    drop(baseline);

    backend.faults().arm(FaultPoint::Allocation, 0);
    let error = backend.alloc_buffer(&desc).expect_err("injected OOM");
    assert_eq!(backend.faults().fired_at(), Some(FaultPoint::Allocation));
    match error {
        Fn3Error::Allocation { bytes, detail } => {
            assert_eq!(bytes, requirements.size);
            assert!(detail.contains("ERROR_OUT_OF_DEVICE_MEMORY"), "{detail}");
            assert!(detail.contains("injected allocation failure"), "{detail}");
        }
        other => panic!("expected a sized allocation failure, found {other:?}"),
    }
    drop(backend);
    assert_eq!(validation_counts().0, counts_before.0, "validation errors");
}

/// The observable session state a failure must not change.
#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    cache_len: u64,
    step: u64,
    tokens: u64,
    phase: &'static str,
    kv_bytes: u64,
    per_layer_extents: Vec<u64>,
}

fn snapshot(session: &Session<'_>) -> Snapshot {
    Snapshot {
        cache_len: session.cache_len(),
        step: session.step(),
        tokens: session.tokens_consumed(),
        phase: session.phase().name(),
        kv_bytes: session.kv_bytes(),
        per_layer_extents: session
            .caches()
            .iter()
            .map(|cache| cache.live_len())
            .collect(),
    }
}

/// The semantic KV content at one committed extent: the live prefix, not the
/// capacity buffers, per layer.
struct KvSnapshot {
    extents: Vec<u64>,
    layers: Vec<(Vec<f32>, Vec<f32>)>,
}

fn kv_snapshot(session: &Session<'_>) -> KvSnapshot {
    KvSnapshot {
        extents: session
            .caches()
            .iter()
            .map(|cache| cache.live_len())
            .collect(),
        layers: session
            .caches()
            .iter()
            .map(|cache| {
                (
                    cache
                        .live_keys()
                        .expect("live keys")
                        .as_f32_slice()
                        .expect("host key bytes")
                        .to_vec(),
                    cache
                        .live_values()
                        .expect("live values")
                        .as_f32_slice()
                        .expect("host value bytes")
                        .to_vec(),
                )
            })
            .collect(),
    }
}

/// Compare the semantic KV content (the live prefix) with a recorded extent.
fn assert_kv_equals(subject: &Session<'_>, expected: &KvSnapshot, label: &str) {
    assert_eq!(
        subject.caches().len(),
        expected.layers.len(),
        "{label}: layers"
    );
    assert_eq!(
        subject
            .caches()
            .iter()
            .map(|cache| cache.live_len())
            .collect::<Vec<_>>(),
        expected.extents,
        "{label}: live extents"
    );
    for (layer, (cache, (expected_keys, expected_values))) in subject
        .caches()
        .iter()
        .zip(expected.layers.iter())
        .enumerate()
    {
        let actual_keys = cache.live_keys().expect("subject keys");
        assert_f32_slice(
            actual_keys.as_f32_slice().expect("subject key bytes"),
            expected_keys,
            &format!("{label}: layer {layer} keys"),
        );
        let actual_values = cache.live_values().expect("subject values");
        assert_f32_slice(
            actual_values.as_f32_slice().expect("subject value bytes"),
            expected_values,
            &format!("{label}: layer {layer} values"),
        );
    }
}

/// Compare two f32 slices without dumping 3600 values into a panic message.
fn assert_f32_slice(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length {} vs {}",
        actual.len(),
        expected.len()
    );
    if let Some((index, (actual, expected))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected)
    {
        panic!("{label}: element {index} is {actual} vs {expected}");
    }
}

fn expect_device_unavailable(error: &Fn3Error, device_name: &str, what: &str) {
    assert!(
        matches!(error, Fn3Error::DeviceUnavailable(_)),
        "{what}: expected DeviceUnavailable, found {error:?}"
    );
    let text = error.to_string();
    assert!(
        text.contains(device_name),
        "{what}: the error must name the adapter `{device_name}`: {text}"
    );
    assert!(
        text.contains("no longer usable") || text.contains("unusable"),
        "{what}: the error must say the device is unusable: {text}"
    );
    println!("[fn3-27][injected] {what}: {text}");
}

// ---------------------------------------------------------------------------
// 1. Recoverable failures: allocation, readback, KV prefix and KV commit
// ---------------------------------------------------------------------------

#[test]
fn recoverable_failures_leave_the_session_intact_and_never_double_append() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();
    // 13-token prompt plus six decode steps: every appended extent from 14
    // to 19 is exercised, so the commit rollback has room after a mid-layer
    // failure.
    let (model, _envelope) = prepared(&import, device, 19);
    let tokens = token_ids();
    // Four committed decode tokens (indices 13..17) plus two more inside the
    // vocabulary, so every appended extent from 14 to 19 is exercised.
    let decode_tokens = [
        tokens[13],
        tokens[14],
        tokens[15],
        tokens[16],
        EXTRA_TOKEN,
        EXTRA_TOKEN_2,
    ];

    // The reference is a clean session on the same backend and model. It
    // grounds the expected logits and KV; the first decode is also checked
    // against the committed `decode.step1` fixture so this is the pinned
    // parity chain, not an ad-hoc sequence.
    let mut reference =
        Session::with_runtime_options(&model, &import, &*backend, SessionOptions::default())
            .expect("reference session");
    let reference_prefill = reference
        .prefill(&tokens[..PROMPT])
        .expect("reference prefill");
    let mut expected = vec![reference_prefill.as_f32_slice().unwrap().to_vec()];
    let mut reference_kv = vec![kv_snapshot(&reference)];
    for token in decode_tokens {
        let logits = reference.decode(&[token]).expect("reference decode");
        expected.push(logits.as_f32_slice().unwrap().to_vec());
        reference_kv.push(kv_snapshot(&reference));
    }
    compare_logits(
        "decode.step1 [fn3-27 injected matrix reference]",
        &expected[1],
        &fixture("decode.step1.logits"),
    );

    let mut session =
        Session::with_runtime_options(&model, &import, &*backend, SessionOptions::default())
            .expect("subject session");
    assert!(
        session.caches()[0].is_device_resident(),
        "the matrix must run the resident path"
    );
    let prefill = session.prefill(&tokens[..PROMPT]).expect("subject prefill");
    assert_eq!(prefill.as_f32_slice().unwrap(), expected[0].as_slice());
    let first = session
        .decode(&[decode_tokens[0]])
        .expect("subject decode 14");
    assert_eq!(first.as_f32_slice().unwrap(), expected[1].as_slice());
    let baseline = snapshot(&session);
    assert_eq!(baseline.cache_len, 14);
    assert_eq!(baseline.step, 1);
    println!("[fn3-27][injected] baseline: {baseline:?}");

    // ---- 1a. allocation failure before any submission -------------------
    let submissions_before = backend.queue_submissions();
    backend.faults().arm(FaultPoint::Allocation, 0);
    let error = session
        .decode(&[decode_tokens[1]])
        .expect_err("the injected allocation must fail the step");
    assert!(
        matches!(error, Fn3Error::Allocation { .. }),
        "expected Allocation, found {error:?}"
    );
    assert_eq!(backend.faults().fired_at(), Some(FaultPoint::Allocation));
    let submissions_after = backend.queue_submissions();
    assert_eq!(
        snapshot(&session),
        baseline,
        "an allocation failure changed session state"
    );
    assert_eq!(
        submissions_after,
        submissions_before,
        "an allocation failure before submission submitted work ({} submissions)",
        submissions_after - submissions_before
    );
    assert_kv_equals(&session, &reference_kv[1], "after allocation failure");
    println!("[fn3-27][injected] allocation: {error}");
    session.recover().expect("recover after allocation failure");
    let logits = session
        .decode(&[decode_tokens[1]])
        .expect("retry after allocation failure");
    assert_eq!(logits.as_f32_slice().unwrap(), expected[2].as_slice());
    assert_kv_equals(&session, &reference_kv[2], "after allocation retry");

    // ---- 1b. readback (logits download) failure -------------------------
    let before = snapshot(&session);
    backend.faults().arm(FaultPoint::Readback, 0);
    let error = session
        .decode(&[decode_tokens[2]])
        .expect_err("the injected readback must fail the step");
    assert!(
        matches!(error, Fn3Error::Allocation { .. }),
        "expected Allocation for a readback, found {error:?}"
    );
    assert_eq!(backend.faults().fired_at(), Some(FaultPoint::Readback));
    assert_eq!(
        snapshot(&session),
        before,
        "a readback failure changed session state (the KV commit must not have run)"
    );
    assert_kv_equals(&session, &reference_kv[2], "after readback failure");
    println!("[fn3-27][injected] readback: {error}");
    session.recover().expect("recover after readback failure");
    let logits = session
        .decode(&[decode_tokens[2]])
        .expect("retry after readback failure");
    assert_eq!(logits.as_f32_slice().unwrap(), expected[3].as_slice());

    // ---- 1c. failure while recording the plan batch ---------------------
    // Fail the 41st dispatch, so the batch already holds a substantial prefix
    // of the 144-instruction plan. `end_step` must discard it rather than
    // submit that prefix; the state and the submission counter prove nothing
    // ran.
    let before = snapshot(&session);
    let submissions_before = backend.queue_submissions();
    backend.faults().arm(FaultPoint::Record, 40);
    let error = session
        .decode(&[decode_tokens[3]])
        .expect_err("the injected recording failure must fail the step");
    assert!(
        matches!(error, Fn3Error::InvalidShape { .. }),
        "expected a runtime error for a rejected recording, found {error:?}"
    );
    assert_eq!(backend.faults().fired_at(), Some(FaultPoint::Record));
    // The four KV past-prefix copies happen before the plan batch opens, so
    // they are the only submissions a discarded batch may leave behind. A
    // submitted partial plan would add one more.
    assert_eq!(
        backend.queue_submissions() - submissions_before,
        (LAYERS * 2) as u64,
        "a failed step batch was submitted instead of discarded"
    );
    assert_eq!(
        snapshot(&session),
        before,
        "a recording failure changed session state"
    );
    assert_kv_equals(&session, &reference_kv[3], "after recording failure");
    println!("[fn3-27][injected] record: {error}");
    session.recover().expect("recover after recording failure");
    let logits = session
        .decode(&[decode_tokens[3]])
        .expect("retry after recording failure");
    assert_eq!(logits.as_f32_slice().unwrap(), expected[4].as_slice());
    assert_kv_equals(&session, &reference_kv[4], "after recording retry");

    // ---- 1d. failure in the KV past-prefix refresh (before the plan) ----
    // A decode step copies four operand prefixes (two layers x keys/values)
    // before the plan batch. Failing the third leaves the cache untouched.
    let before = snapshot(&session);
    let copies_before = backend.faults().calls(FaultPoint::ResidentCopy);
    backend.faults().arm(FaultPoint::ResidentCopy, 2);
    let error = session
        .decode(&[decode_tokens[4]])
        .expect_err("the injected prefix copy must fail the step");
    assert!(
        matches!(error, Fn3Error::Allocation { .. }),
        "expected Allocation for a resident copy, found {error:?}"
    );
    assert_eq!(
        backend.faults().calls(FaultPoint::ResidentCopy) - copies_before,
        3,
        "the third copy was the armed one"
    );
    assert_eq!(
        snapshot(&session),
        before,
        "a prefix-copy failure changed session state"
    );
    assert_kv_equals(&session, &reference_kv[4], "after prefix-copy failure");
    println!("[fn3-27][injected] kv-prefix: {error}");
    session.recover().expect("recover after prefix failure");
    let logits = session
        .decode(&[decode_tokens[4]])
        .expect("retry after prefix failure");
    assert_eq!(logits.as_f32_slice().unwrap(), expected[5].as_slice());

    // ---- 1e. failure between the layers of the KV commit ----------------
    // At extent 18 a decode copies four prefixes (events 0-3), then appends
    // layer 0 keys/values (4, 5), then layer 1 keys (6), then layer 1 values
    // (7). Failing event 7 leaves layer 0 fully appended and layer 1's keys
    // appended: the exact half-commit the rollback must undo.
    let before = snapshot(&session);
    assert_eq!(before.cache_len, 18);
    let copies_before = backend.faults().calls(FaultPoint::ResidentCopy);
    backend.faults().arm(FaultPoint::ResidentCopy, 7);
    let error = session
        .decode(&[decode_tokens[5]])
        .expect_err("the injected KV append must fail the step");
    assert!(
        matches!(error, Fn3Error::Allocation { .. }),
        "expected Allocation for a KV append, found {error:?}"
    );
    assert_eq!(
        backend.faults().calls(FaultPoint::ResidentCopy) - copies_before,
        8,
        "the eighth copy was the armed layer-1 value append"
    );
    let after = snapshot(&session);
    assert_eq!(
        after, before,
        "a half-committed KV append must roll every layer back"
    );
    assert_eq!(
        after.per_layer_extents,
        vec![18, 18],
        "the rollback must leave both layers at the committed extent"
    );
    // The bytes past the restored extent are unobservable; the semantic KV is
    // exactly the reference's 17-token prefix.
    assert_kv_equals(&session, &reference_kv[5], "after half-commit rollback");
    println!("[fn3-27][injected] kv-half-commit: {error}");
    session.recover().expect("recover after half-commit");
    let logits = session
        .decode(&[decode_tokens[5]])
        .expect("retry after half-commit");
    assert_eq!(logits.as_f32_slice().unwrap(), expected[6].as_slice());
    assert_eq!(session.cache_len(), 19);
    assert_eq!(session.caches()[0].live_len(), 19);
    assert_eq!(session.caches()[1].live_len(), 19);
    assert_kv_equals(&session, &reference_kv[6], "after half-commit retry");

    // The retry appended each token exactly once: the semantic KV is the
    // reference's, and the extents are 18, not 19.
    println!(
        "[fn3-27][injected] recoverable matrix: allocation, readback, recording, prefix-copy and \
         half-commit failures all left the session at the committed extent and the retry matched \
         the clean reference byte for byte (extent 19, no double append)"
    );

    let (errors, warnings) = validation_counts();
    assert_eq!(
        (errors, warnings),
        (0, 0),
        "injected recoverable failures must not raise validation messages"
    );
    assert_eq!(validation_counts(), counts_before, "validation delta");
    println!(
        "[fn3-27][injected][validation] absolute errors {errors} warnings {warnings}, delta 0/0"
    );
}

// ---------------------------------------------------------------------------
// 2. Repeated reset at multiple extents, interleaved with a failure
// ---------------------------------------------------------------------------

#[test]
fn repeated_reset_returns_extents_kv_plans_and_memory_to_a_stable_baseline() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();
    let (model, _envelope) = prepared(&import, device, 18);
    let tokens = token_ids();
    // Four committed decode tokens (indices 13..17) plus two more inside the
    // vocabulary, so every appended extent from 14 to 19 is exercised.
    // The reset rounds stop at extent 18, matching the prepared window.
    let decode_tokens = [tokens[13], tokens[14], tokens[15], tokens[16], EXTRA_TOKEN];
    let mut session =
        Session::with_runtime_options(&model, &import, &*backend, SessionOptions::default())
            .expect("session");
    let kv_bytes = session.kv_bytes();
    assert!(kv_bytes > 0, "the session must own a KV cache");

    let mut reference =
        Session::with_runtime_options(&model, &import, &*backend, SessionOptions::default())
            .expect("reference session");
    let reference_prefill = reference
        .prefill(&tokens[..PROMPT])
        .expect("reference prefill");
    let mut expected = vec![reference_prefill.as_f32_slice().unwrap().to_vec()];
    for token in decode_tokens {
        expected.push(
            reference
                .decode(&[token])
                .expect("reference decode")
                .as_f32_slice()
                .unwrap()
                .to_vec(),
        );
    }

    // Warm up every plan the loop can use (prefill 13 plus decodes at past
    // 13..17) and the resident KV past-prefix buffers, then reset. The
    // baseline below is a steady state, not a cold-start one.
    session.prefill(&tokens[..PROMPT]).expect("warmup prefill");
    for token in decode_tokens {
        session.decode(&[token]).expect("warmup decode");
    }
    session.reset();
    let plan_baseline = (session.plan_cache_len(), session.plan_evictions());
    let memory = backend.memory_stats();
    let memory_baseline = (memory.live_bytes, memory.live_device_local_bytes);
    println!(
        "[fn3-27][injected] reset baseline: plans {plan_baseline:?}, live_bytes {}, \
         live_device_local {}",
        memory.live_bytes, memory.live_device_local_bytes
    );

    for round in 0..12usize {
        let decodes = round % 5; // extents 13, then 14..=18
        let prefill = session.prefill(&tokens[..PROMPT]).expect("round prefill");
        assert_eq!(prefill.as_f32_slice().unwrap(), expected[0].as_slice());
        for step in 0..decodes {
            let logits = session
                .decode(&[decode_tokens[step]])
                .expect("round decode");
            assert_eq!(
                logits.as_f32_slice().unwrap(),
                expected[step + 1].as_slice()
            );
            assert_eq!(session.cache_len(), 13 + step as u64 + 1);
        }
        // Round 3 fails its last decode (or its only decode) and recovers.
        if round == 3 {
            let before = snapshot(&session);
            backend.faults().arm(FaultPoint::Allocation, 0);
            let error = session
                .decode(&[decode_tokens[decodes]])
                .expect_err("the interleaved failure must fire");
            assert!(matches!(error, Fn3Error::Allocation { .. }), "{error}");
            assert_eq!(snapshot(&session), before, "round 3 state changed");
            session.recover().expect("round 3 recovery");
            let logits = session
                .decode(&[decode_tokens[decodes]])
                .expect("round 3 retry");
            assert_eq!(
                logits.as_f32_slice().unwrap(),
                expected[decodes + 1].as_slice()
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
        session.recover().expect("recover on a reset session");

        let plans = (session.plan_cache_len(), session.plan_evictions());
        let memory = backend.memory_stats();
        assert_eq!(
            plans, plan_baseline,
            "round {round}: the plan cache is not stable across resets"
        );
        assert_eq!(
            (memory.live_bytes, memory.live_device_local_bytes),
            memory_baseline,
            "round {round}: device memory did not return to the reset baseline"
        );
    }
    let plans = (session.plan_cache_len(), session.plan_evictions());
    assert_eq!(
        plans, plan_baseline,
        "the final plan cache differs from the baseline"
    );
    assert_eq!(plans.1, 0, "a repeated reset evicted a plan");
    println!(
        "[fn3-27][injected] repeated reset: 12 rounds at extents 13..=18, one injected failure \
         and recovery in round 3; extents 0, KV {} B, plans {plans:?} and device memory stable",
        session.kv_bytes()
    );

    let (errors, warnings) = validation_counts();
    assert_eq!(
        (errors, warnings),
        (0, 0),
        "repeated reset must not raise validation messages"
    );
    assert_eq!(validation_counts(), counts_before, "validation delta");
}

// ---------------------------------------------------------------------------
// 3. Terminal device loss at submit and at the fence
// ---------------------------------------------------------------------------

/// Reopen the same physical device and prove the process and the model are
/// usable after the terminal test dropped the poisoned backend.
fn prove_device_recreation(tokens: &[u32], expected_prefill: &[f32], counts_before: (u64, u64)) {
    let Some(fresh) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let device = fresh.device_id();
    let (model, _envelope) = prepared(&import, device, FIXTURE_CONTEXT);
    let mut session =
        Session::with_runtime_options(&model, &import, &*fresh, SessionOptions::default())
            .expect("session on the recreated device");
    let logits = session
        .prefill(&tokens[..4])
        .expect("prefill after recreation");
    assert_eq!(
        logits.as_f32_slice().unwrap(),
        expected_prefill,
        "the recreated device must produce the same prompt logits"
    );
    drop(session);
    drop(fresh);
    let (errors, warnings) = validation_counts();
    assert_eq!(
        (errors, warnings),
        counts_before,
        "recreation validation delta"
    );
    println!(
        "[fn3-27][injected] recreation: a fresh backend after the terminal failure ran a real \
         prefill; validation {errors}/{warnings}"
    );
}

fn terminal_case(point: FaultPoint) -> (u64, u64) {
    let _guard = gpu_lock();
    let counts_before = validation_counts();
    let Some(backend) = open_physical() else {
        return counts_before;
    };
    let Some(import) = pinned() else {
        return counts_before;
    };
    let device = backend.device_id();
    let device_name = backend.adapter().name.clone();
    let (model, _envelope) = prepared(&import, device, FIXTURE_CONTEXT);
    let tokens = token_ids();
    let mut session =
        Session::with_runtime_options(&model, &import, &*backend, SessionOptions::default())
            .expect("session");
    let prefill = session.prefill(&tokens[..4]).expect("prefill");
    let prefill_bytes = prefill.as_f32_slice().unwrap().to_vec();
    let baseline = snapshot(&session);
    assert_eq!(baseline.cache_len, 4);
    assert_eq!(baseline.phase, "ready");

    let submissions_before = backend.queue_submissions();
    backend.faults().arm(point, 0);
    let error = session
        .decode(&[tokens[4]])
        .expect_err("the injected terminal failure must fail the step");
    expect_device_unavailable(&error, &device_name, point.name());
    assert_eq!(backend.faults().fired_at(), Some(point));
    assert_eq!(
        snapshot(&session),
        baseline,
        "{}: the terminal failure changed session state",
        point.name()
    );
    if point == FaultPoint::Submit {
        assert_eq!(
            backend.queue_submissions(),
            submissions_before,
            "the injected submit failure reached vkQueueSubmit"
        );
    } else {
        assert!(
            backend.queue_submissions() > submissions_before,
            "the fence failure must happen after the real submission"
        );
    }

    // Recovery is not possible in place; the same typed error comes back from
    // every later call, and no partial plan is executed.
    let error = session
        .recover()
        .expect_err("a poisoned device cannot recover");
    expect_device_unavailable(&error, &device_name, "recover after terminal loss");
    let error = session
        .decode(&[tokens[5]])
        .expect_err("a poisoned device must refuse the next step");
    expect_device_unavailable(&error, &device_name, "step after terminal loss");
    assert_eq!(
        snapshot(&session),
        baseline,
        "refused calls on a poisoned device changed session state"
    );

    // Local session state can still be reset; the device cannot. A fresh
    // prefill then reaches the terminal check rather than the phase guard.
    session.reset();
    assert_eq!(session.phase(), SessionPhase::Empty);
    assert_eq!(session.cache_len(), 0);
    let error = session
        .prefill(&tokens[..1])
        .expect_err("a poisoned device must refuse a fresh prefill");
    expect_device_unavailable(&error, &device_name, "prefill after terminal loss");
    assert_eq!(session.cache_len(), 0, "the refused prefill appended KV");
    drop(session);
    drop(backend);

    let (errors, warnings) = validation_counts();
    assert_eq!(
        (errors, warnings),
        (0, 0),
        "{}: terminal failure raised validation messages",
        point.name()
    );
    assert_eq!(validation_counts(), counts_before, "validation delta");

    prove_device_recreation(&tokens, &prefill_bytes, counts_before);
    counts_before
}

#[test]
fn device_loss_at_submit_is_terminal_and_named() {
    terminal_case(FaultPoint::Submit);
}

#[test]
fn device_loss_at_the_fence_is_terminal_and_named() {
    terminal_case(FaultPoint::Fence);
}
