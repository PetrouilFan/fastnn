//! FN3-16 (#62): the pinned model, end to end, on CPU and Vulkan.
//!
//! This is the authoritative alpha gate. It loads the committed tiny-GPT-2
//! ONNX graph, prepares it for one device, prefills 13 tokens, runs cached
//! decode steps and compares the logits and every appended KV tensor with the
//! FN3-03 ORT/PyTorch fixtures under the documented tolerance. `fn3_session`
//! proves the session contract in isolation; this suite proves the same
//! production path with the real model at two context lengths, and — with
//! `--features vulkan` — proves every instruction dispatched on the named
//! physical device.
//!
//! ```bash
//! FASTNN_FN3_ONNX_MODEL=/path/to/model.onnx \
//!   CARGO_BUILD_JOBS=12 cargo test --test fn3_e2e -- --nocapture --test-threads=1
//! FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
//!   CARGO_BUILD_JOBS=12 cargo test --features vulkan --test fn3_e2e -- \
//!   --nocapture --test-threads=1
//! ```
//!
//! `FN3_E2E_CONTEXT` sets the long-chain context length (default 32).
//!
//! The model is not committed (no declared licence). Without it every
//! pinned-model test prints a reason and passes, so an offline checkout is
//! honest rather than falsely green.

// A test crate root resolves `mod x;` next to itself, so the shared harness
// and the Vulkan half live in the sibling directory and are reached by path.
#[path = "fn3_e2e/harness.rs"]
mod harness;

#[cfg(feature = "vulkan")]
#[path = "fn3_e2e/gpu.rs"]
mod gpu;

use fastnn::fn3::shape::ShapeEnvelope;
use fastnn::fn3::{
    CpuBackend, DeviceId, Fn3Error, PhysicalDeviceId, PreparedModel, Session, Tensor,
};

use harness::*;

/// Prepare the pinned model on CPU at the fixture context.
fn prepared_cpu(import: &fastnn::fn3::import::ModelImport) -> (PreparedModel, ShapeEnvelope) {
    prepared(import, DeviceId::Cpu, FIXTURE_CONTEXT)
}

// ---------------------------------------------------------------------------
// 1. Short prefills
// ---------------------------------------------------------------------------

/// `prefill.p1` and `prefill.p4` logits and KV, the two shortest committed
/// points. They catch a KV layout regression that a 13-token prompt might hide
/// behind the attention window.
#[test]
fn cpu_short_prefills_match_the_committed_fixtures() {
    let Some(import) = pinned() else {
        return;
    };
    let (model, envelope) = prepared_cpu(&import);
    let backend = CpuBackend::new();
    let tokens = token_ids();
    for seq in [1u64, 4] {
        check_prefill(&backend, &import, &model, &envelope, &tokens, seq);
    }
}

// ---------------------------------------------------------------------------
// 2. Prefill 13 + four cached decode steps
// ---------------------------------------------------------------------------

/// The headline claim: 13-token prefill, then four cached decode steps whose
/// live extents are exactly 14, 15, 16 and 17, with every logits and KV
/// fixture checked and the cache allocation constant throughout.
#[test]
fn cpu_prefill_and_four_cached_decode_steps_match_the_fixtures() {
    let Some(import) = pinned() else {
        return;
    };
    let (model, envelope) = prepared_cpu(&import);
    let backend = CpuBackend::new();
    let tokens = token_ids();
    // The two short prefills, then the chain (which itself checks the 13-token
    // prefill against `prefill.p13.last_logits` and all 16 KV tensors).
    check_prefill(&backend, &import, &model, &envelope, &tokens, 1);
    check_prefill(&backend, &import, &model, &envelope, &tokens, 4);
    let chain = run_chain(
        &backend,
        &import,
        &model,
        &envelope,
        &tokens,
        FIXTURE_CONTEXT,
    );
    let extents: Vec<u64> = chain.steps.iter().map(|step| step.extent).collect();
    assert_eq!(extents, vec![14, 15, 16, 17]);
    assert_eq!(chain.prefill.len(), VOCAB);
    assert_eq!(chain.kv_bytes, 34_816, "the pinned cache is 2 x 17,408 B");
    assert_eq!(chain.plan_misses, 5, "one plan per (seq, past)");
    assert_eq!(chain.plan_hits, 0);
    assert_eq!(chain.plan_evictions, 0);
    assert!(chain.slot_reuses > 0, "the slot pool must be reused");
    println!(
        "[cpu] gate: prefill 13 + decode 14..17 matched every fixture, kv_bytes {} constant, \
         slot allocations {} reuses {}",
        chain.kv_bytes, chain.slot_allocations, chain.slot_reuses
    );
}

// ---------------------------------------------------------------------------
// 3. A longer chain, bounded
// ---------------------------------------------------------------------------

/// Prefill 13, decode to a configurable context (default 32) twice over the
/// same prepared model. The two chains must be identical step by step and the
/// KV allocation must never grow: the cache is allocated once, the live extent
/// is the only thing that moves.
#[test]
fn cpu_long_chain_stays_bounded_and_is_deterministic() {
    let Some(import) = pinned() else {
        return;
    };
    let context = long_context();
    assert!(
        context > FIXTURE_CONTEXT,
        "FN3_E2E_CONTEXT={context} must exceed the {FIXTURE_CONTEXT}-token fixture chain"
    );
    let (model, envelope) = prepared(&import, DeviceId::Cpu, context);
    let backend = CpuBackend::new();
    let tokens = token_ids();

    let first = run_chain(&backend, &import, &model, &envelope, &tokens, context);
    let second = run_chain(&backend, &import, &model, &envelope, &tokens, context);

    assert_eq!(
        first.prefill, second.prefill,
        "two sessions must produce identical prompt logits"
    );
    assert_eq!(first.steps.len(), second.steps.len());
    let expected: Vec<u64> = (14..=context).collect();
    let actual: Vec<u64> = first.steps.iter().map(|step| step.extent).collect();
    assert_eq!(actual, expected, "every extent from 14 to {context}");
    for (index, (a, b)) in first.steps.iter().zip(&second.steps).enumerate() {
        assert_eq!(
            a.logits,
            b.logits,
            "decode step {} at extent {} differs between two sessions",
            index + 1,
            a.extent
        );
        assert_eq!(a.census.total, b.census.total);
        assert_eq!(a.census.with_bytes, b.census.with_bytes);
    }
    assert_eq!(first.kv_bytes, second.kv_bytes);
    assert_eq!(
        first.kv_bytes,
        LAYERS as u64 * 2 * KV_HEADS * context * HEAD_DIM * 4,
        "the KV cache is sized by capacity alone, not by the live extent"
    );
    assert_eq!(first.plan_evictions, 0);
    assert!(first.plan_cache_len <= 64);
    println!(
        "[cpu] long chain to {context}: extents 14..={context}, kv_bytes {} constant, \
         plan cache {} of 64, two sessions byte-identical",
        first.kv_bytes, first.plan_cache_len
    );
}

// ---------------------------------------------------------------------------
// 4. Session isolation and capacity
// ---------------------------------------------------------------------------

/// Two sessions over one `PreparedModel` are independent and identical, and a
/// decode at capacity is refused without touching a byte.
#[test]
fn cpu_sessions_are_isolated_and_capacity_does_not_corrupt_state() {
    let Some(import) = pinned() else {
        return;
    };
    let (model, _) = prepared_cpu(&import);
    let backend = CpuBackend::new();
    let outcome =
        check_isolation_and_capacity(&backend, &import, &model, &token_ids(), FIXTURE_CONTEXT);
    assert_eq!(outcome.first_prefill, outcome.second_prefill);
    assert_eq!(outcome.first_steps.len(), 4);
    assert_eq!(outcome.second_steps.len(), 2);
    assert_eq!(outcome.first_steps[0], outcome.second_steps[0]);
}

// ---------------------------------------------------------------------------
// 5. No silent CPU fallback
// ---------------------------------------------------------------------------

/// A model prepared for a Vulkan device is never stepped by the CPU backend,
/// and a tensor cannot be conjured onto a device with no allocator.
///
/// This is the device half of the no-silent-fallback rule. The operation half
/// (`UnsupportedOp` instead of an approximation) is pinned by
/// `tests/fn3_contracts.rs`; the explicit-request failure is pinned by the
/// Vulkan module below, because it needs a driver probe.
#[test]
fn a_model_prepared_for_vulkan_is_never_executed_by_the_cpu_backend() {
    let Some(import) = pinned() else {
        return;
    };
    let vulkan = DeviceId::Vulkan(PhysicalDeviceId::new(0));
    let (model, _) = prepared(&import, vulkan, FIXTURE_CONTEXT);
    let backend = CpuBackend::new();

    // Construction is refused before any cache is allocated or plan is built.
    let error = Session::with_runtime(&model, &import, &backend)
        .expect_err("the CPU backend must not accept a Vulkan model");
    match &error {
        Fn3Error::DeviceMismatch {
            expected, actual, ..
        } => {
            assert_eq!(*expected, vulkan);
            assert_eq!(*actual, DeviceId::Cpu);
        }
        other => panic!("expected DeviceMismatch, found {other:?}"),
    }
    // And so is the model's own fixed plan.
    let error = model
        .execute(&backend, &mut [])
        .expect_err("the fixed plan must not run on the CPU");
    assert!(matches!(error, Fn3Error::DeviceMismatch { .. }), "{error}");

    // Host storage tagged for a Vulkan device without a backend is not the
    // same thing as a Vulkan tensor: `Tensor::zeros` reports the device, it
    // does not quietly allocate host memory for it.
    let error = Tensor::zeros(
        fastnn::fn3::shape::Shape::from_dims([1, 4]),
        fastnn::fn3::DType::F32,
        vulkan,
    )
    .expect_err("there is no Vulkan allocator without a backend");
    assert!(
        matches!(error, Fn3Error::DeviceUnavailable(_)),
        "expected DeviceUnavailable, found {error}"
    );
    println!(
        "no-fallback: a Vulkan-prepared model was refused by cpu (DeviceMismatch); no tensor \
         was allocated on {vulkan}"
    );
}
