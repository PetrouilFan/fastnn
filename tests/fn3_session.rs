//! FN3-12 (#58): session-owned KV state, prefill and cached decode.
//!
//! # What these tests prove
//!
//! 1. **Prefill matches the committed fixtures.** `prefill.p1.logits`,
//!    `prefill.p4.last_logits` and `prefill.p13.last_logits` are compared with
//!    the FN3-03 tiny-logits tolerance, and the K/V the session appends is
//!    compared with `prefill.p{1,4,8,13}.present.*`.
//! 2. **Cached decode matches four steps.** After a 13-token prefill the session
//!    runs `decode.step{1..4}` with live extents exactly 14, 15, 16, 17, and
//!    each step's logits and appended K/V are compared with the fixtures.
//!    Step 1's logits are stored; steps 2–4 are summary-only, so their recorded
//!    argmax/first/last/min/max are checked and the cached logits are also
//!    compared against an independent **full recompute** (a fresh session
//!    prefilling the whole sequence).
//! 3. **The plan is cached.** A repeated round (reset, prefill, decode) is
//!    served from `PlanCache` — the hit/miss counters are printed and asserted.
//! 4. **Bounds and atomicity.** Appending past capacity is a typed error that
//!    changes no extent and no counter, and a backend that fails in the middle
//!    of a plan leaves the session exactly where it was.
//! 5. **Sessions are isolated.** Two sessions over one model have distinct KV
//!    buffers, extents and counters, and produce identical results.
//! 6. **Every instruction goes through the session's backend.** A counting
//!    backend sees one `execute_op` per plan instruction per step, and a model
//!    prepared for Vulkan is refused by a CPU backend rather than computed on
//!    the host. With `--features vulkan` the pinned model is additionally run
//!    on the named physical device and every instruction must carry a residency
//!    record proving device execution.
//!
//! # Running
//!
//! ```bash
//! CARGO_BUILD_JOBS=4 cargo test --test fn3_session -- --nocapture --test-threads=1
//! FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
//!   CARGO_BUILD_JOBS=4 cargo test --features vulkan --test fn3_session -- \
//!   --nocapture --test-threads=1
//! ```
//!
//! The pinned graph is not committed (no declared licence). Set
//! `FASTNN_FN3_ONNX_MODEL=/path/to/model.onnx`, or place it at
//! `tools/parity/.cache/models/tiny-gpt2/onnx/model.onnx`; otherwise every
//! pinned-model test prints a reason and passes.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use fastnn::fn3::import::{import_onnx_model, ModelImport};
use fastnn::fn3::model::{ModelWeights, Planner, PreparedModel, Session, SessionPhase};
use fastnn::fn3::shape::ShapeEnvelope;
use fastnn::fn3::{
    Backend, Capabilities, CpuBackend, DType, DeviceId, Fn3Error, LiveDims, OpId, OpParams,
    Support, Tensor, TensorMeta, TensorMut, TensorView,
};

// ---------------------------------------------------------------------------
// Pinned model and fixtures
// ---------------------------------------------------------------------------

/// Bytes of the pinned graph (`fixtures/parity/tiny-gpt2/model.lock.json`).
const PINNED_BYTES: usize = 27_897_974;

/// FNV-1a 64-bit fingerprint of the pinned file, so a different graph at the
/// configured path is a failure rather than a silently different test.
const PINNED_FNV1A: u64 = 0xc294_d3ab_837f_b1be;

/// Vocabulary of the pinned model.
const VOCAB: usize = 50_257;

/// Layers, KV heads and per-head width of the pinned model.
const LAYERS: usize = 2;
const KV_HEADS: u64 = 2;
const HEAD_DIM: u64 = 64;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn pinned() -> Option<ModelImport> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("FASTNN_FN3_ONNX_MODEL") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tools/parity/.cache/models/tiny-gpt2/onnx/model.onnx"),
    );
    for path in candidates {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        assert_eq!(
            bytes.len(),
            PINNED_BYTES,
            "the file at {} is not the pinned graph",
            path.display()
        );
        assert_eq!(
            fnv1a(&bytes),
            PINNED_FNV1A,
            "the file at {} is not the pinned graph",
            path.display()
        );
        return Some(
            import_onnx_model(&bytes)
                .unwrap_or_else(|error| panic!("the pinned model failed to import:\n{error}")),
        );
    }
    println!(
        "SKIP: the pinned 27,897,974-byte ONNX model is not present. Set \
         FASTNN_FN3_ONNX_MODEL or run `python3 tools/parity/fetch_models.py --suite tiny-gpt2`."
    );
    None
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/tiny-gpt2")
}

/// One stored `f32` fixture payload.
fn fixture(name: &str) -> Vec<f32> {
    let path = fixture_root().join("tensors").join(format!("{name}.f32"));
    let bytes =
        std::fs::read(&path).unwrap_or_else(|error| panic!("fixture {}: {error}", path.display()));
    let (chunks, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty(), "fixture is not f32-aligned");
    chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// The fixture manifest, parsed once per test that needs a summary-only tensor.
fn manifest() -> &'static serde_json::Value {
    use std::sync::OnceLock;
    static MANIFEST: OnceLock<serde_json::Value> = OnceLock::new();
    MANIFEST.get_or_init(|| {
        let path = fixture_root().join("manifest.json");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("manifest {}: {error}", path.display()));
        serde_json::from_str(&text).unwrap_or_else(|error| panic!("manifest is not JSON: {error}"))
    })
}

/// A summary-only tensor's recorded fields.
struct Summary {
    argmax: usize,
    first: Vec<f32>,
    last: Vec<f32>,
    min: f32,
    max: f32,
}

fn summary(name: &str) -> Option<Summary> {
    let tensors = manifest().get("tensors")?.as_array()?;
    let entry = tensors
        .iter()
        .find(|entry| entry.get("name").and_then(|value| value.as_str()) == Some(name))?;
    summary_of(entry)
}

/// A summary-only tensor's recorded row summary (`slices.items[index]`).
///
/// `full.p{14..17}.logits` are stored as per-position digests and summaries, so
/// the last row of an independent full recompute can be checked against the
/// reference even though the payload is not committed.
fn slice_summary(name: &str, index: usize) -> Option<Summary> {
    let tensors = manifest().get("tensors")?.as_array()?;
    let entry = tensors
        .iter()
        .find(|entry| entry.get("name").and_then(|value| value.as_str()) == Some(name))?;
    let item = entry
        .get("slices")?
        .get("items")?
        .as_array()?
        .iter()
        .find(|item| item.get("index").and_then(|value| value.as_u64()) == Some(index as u64))?;
    summary_of(item)
}

fn summary_of(entry: &serde_json::Value) -> Option<Summary> {
    let summary = entry.get("summary")?;
    let floats = |key: &str| -> Vec<f32> {
        summary
            .get(key)
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_f64().expect("float summary entry") as f32)
                    .collect()
            })
            .unwrap_or_default()
    };
    Some(Summary {
        argmax: summary.get("argmax_flat")?.as_u64()? as usize,
        first: floats("first"),
        last: floats("last"),
        min: summary.get("min")?.as_f64()? as f32,
        max: summary.get("max")?.as_f64()? as f32,
    })
}

/// The 17-token sequence the fixtures use.
fn token_ids() -> Vec<u32> {
    vec![
        464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257, 640, 11, 612,
        373,
    ]
}

// ---------------------------------------------------------------------------
// Numerics
// ---------------------------------------------------------------------------

const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-4;
const MEAN_ABS: f64 = 2.5e-6;

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 {
                (index, value)
            } else {
                best
            }
        })
        .0
}

/// The gap between the top two values, or `None` for a single element.
fn gap(values: &[f32]) -> Option<f32> {
    let mut top = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &value in values {
        if value > top.0 {
            top = (value, top.0);
        } else if value > top.1 {
            top.1 = value;
        }
    }
    Some(top.0 - top.1)
}

/// Compare against a stored payload with the tiny-logits tolerance.
///
/// Every element must be finite, inside `atol + rtol * |reference|`, the mean
/// absolute error must stay under 2.5e-6, and the argmax must agree whenever
/// the reference's top-two gap exceeds `2 * atol`.
fn compare_logits(label: &str, actual: &[f32], reference: &[f32]) -> (f32, f32) {
    assert_eq!(actual.len(), reference.len(), "{label}: length");
    let mut max_abs = 0.0f32;
    let mut worst = (0usize, 0.0f32, 0.0f32);
    let mut total_abs = 0.0f64;
    for (index, (&got, &want)) in actual.iter().zip(reference).enumerate() {
        assert!(got.is_finite(), "{label}: element {index} is {got}");
        let difference = (got - want).abs();
        if difference > max_abs {
            max_abs = difference;
            worst = (index, got, want);
        }
        total_abs += f64::from(difference);
        let bound = ATOL + RTOL * want.abs();
        assert!(
            difference <= bound,
            "{label}: element {index} is {got} vs {want} ({difference} > {bound})"
        );
    }
    let mean_abs = (total_abs / reference.len() as f64) as f32;
    assert!(
        f64::from(mean_abs) <= MEAN_ABS,
        "{label}: mean_abs {mean_abs:e} above the documented 2.5e-6 systematic-bias bound"
    );
    if let Some(reference_gap) = gap(reference) {
        if reference_gap > 2.0 * ATOL {
            assert_eq!(
                argmax(actual),
                argmax(reference),
                "{label}: argmax differs with a decisive reference gap of {reference_gap:e}"
            );
        }
    }
    println!(
        "{label}: {} logits, max_abs {max_abs:e} (element {}: {:.7} vs {:.7}), mean_abs {mean_abs:e}",
        reference.len(),
        worst.0,
        worst.1,
        worst.2
    );
    (max_abs, mean_abs)
}

/// Compare against a summary-only reference.
///
/// The full payload is not committed, so the manifest's argmax, first/last
/// eight values and min/max are checked. The argmax rule uses *our* top-two gap:
/// if our two best logits differ by more than `2 * atol` then no value within
/// `atol` of ours can reorder them, so the reference must agree.
fn compare_summary(label: &str, actual: &[f32], reference: &Summary) {
    for (index, &value) in actual.iter().enumerate() {
        assert!(value.is_finite(), "{label}: element {index} is {value}");
    }
    let mut worst = 0.0f32;
    for (position, (&got, &want)) in actual.iter().zip(&reference.first).enumerate() {
        let difference = (got - want).abs();
        worst = worst.max(difference);
        assert!(
            difference <= ATOL + RTOL * want.abs(),
            "{label}: first[{position}] is {got} vs {want}"
        );
    }
    for (position, (&got, &want)) in actual
        .iter()
        .rev()
        .zip(reference.last.iter().rev())
        .enumerate()
    {
        let difference = (got - want).abs();
        worst = worst.max(difference);
        assert!(
            difference <= ATOL + RTOL * want.abs(),
            "{label}: last[{position}] is {got} vs {want}"
        );
    }
    let (min, max) = actual
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &value| {
            (low.min(value), high.max(value))
        });
    assert!(
        (min - reference.min).abs() <= ATOL + RTOL * reference.min.abs(),
        "{label}: min {min} vs {}",
        reference.min
    );
    assert!(
        (max - reference.max).abs() <= ATOL + RTOL * reference.max.abs(),
        "{label}: max {max} vs {}",
        reference.max
    );
    if let Some(our_gap) = gap(actual) {
        if our_gap > 2.0 * ATOL {
            assert_eq!(
                argmax(actual),
                reference.argmax,
                "{label}: argmax differs with a decisive gap of {our_gap:e}"
            );
        }
    }
    println!(
        "{label}: summary-only, first/last/max_abs {worst:e}, argmax {} (recorded {})",
        argmax(actual),
        reference.argmax
    );
}

/// Compare a KV tensor against a stored fixture.
fn compare_kv(label: &str, actual: &Tensor, expected_tokens: u64) -> (f32, f32) {
    assert_eq!(
        actual.shape().dims(),
        &[1, KV_HEADS, expected_tokens, HEAD_DIM],
        "{label}: shape"
    );
    let reference = fixture(label);
    compare_logits(label, actual.as_f32_slice().expect("host KV"), &reference)
}

// ---------------------------------------------------------------------------
// Model plumbing
// ---------------------------------------------------------------------------

/// The envelope these tests use: batch 1, at most 13 new tokens, a cache of at
/// most 17 tokens, at most 18 total.
const CAPACITY: u64 = 17;

fn envelope(planner: &Planner<'_>) -> ShapeEnvelope {
    planner
        .envelope(1, 13, CAPACITY, 18)
        .expect("the pinned geometry fixes the head symbols")
}

fn dims(seq: u64, past: u64) -> LiveDims {
    LiveDims::new(1, seq, past, 2, KV_HEADS, HEAD_DIM).expect("total fits")
}

/// Prepare the pinned model on `device` for the envelope above.
fn prepared(import: &ModelImport, device: DeviceId) -> (PreparedModel, ShapeEnvelope) {
    prepared_with(import, device, 13, CAPACITY, 18, CAPACITY)
}

/// Prepare the pinned model for an explicit dynamic window.
///
/// `max_seq`/`max_past`/`max_total` are the declared envelope and `capacity`
/// the KV cache capacity. The cached-vs-full comparison needs a second model
/// whose window admits a full 14–17 token prefill.
fn prepared_with(
    import: &ModelImport,
    device: DeviceId,
    max_seq: u64,
    max_past: u64,
    max_total: u64,
    capacity: u64,
) -> (PreparedModel, ShapeEnvelope) {
    let backend = CpuBackend::new();
    let planner = Planner::new(import, &backend);
    let envelope = planner
        .envelope(1, max_seq, max_past, max_total)
        .expect("the window is consistent");
    let config = import.geometry().kv_cache_config(capacity, 1);
    let weights = import.to_model_weights().expect("host weights");
    let weights = retag_weights(&weights, device);
    // The model's own fixed plan is the prefill at seq = 4; a session with a
    // runtime specialises its own plans and never executes this one.
    let plan = planner
        .plan(&envelope, dims(4, 0))
        .expect("prefill plan")
        .execution()
        .clone();
    let model =
        PreparedModel::new(plan, weights, device, envelope, Some(config)).expect("prepared model");
    (model, envelope)
}

/// Re-tag every weight for `device` without copying bytes.
fn retag_weights(weights: &ModelWeights, device: DeviceId) -> ModelWeights {
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

fn cpu_session<'a>(
    model: &'a PreparedModel,
    import: &'a ModelImport,
    backend: &'a CpuBackend,
) -> Session<'a> {
    Session::with_runtime(model, import, backend).expect("session")
}

/// Number of instructions a plan of shape `(seq, past)` contains.
fn instruction_count(import: &ModelImport, envelope: &ShapeEnvelope, seq: u64, past: u64) -> usize {
    let backend = CpuBackend::new();
    Planner::new(import, &backend)
        .plan(envelope, dims(seq, past))
        .expect("plan")
        .instructions()
        .len()
}

// ---------------------------------------------------------------------------
// 1. Prefill
// ---------------------------------------------------------------------------

#[test]
fn prefill_matches_the_committed_fixtures_and_populates_the_cache() {
    let Some(import) = pinned() else {
        return;
    };
    let (model, _) = prepared(&import, DeviceId::Cpu);
    let backend = CpuBackend::new();
    let tokens = token_ids();

    // seq = 1: one logits row, one cached token.
    let mut session = cpu_session(&model, &import, &backend);
    let logits = session.prefill(&tokens[..1]).expect("prefill p1");
    assert_eq!(logits.shape().dims(), &[1, 1, VOCAB as u64]);
    assert_eq!(session.cache_len(), 1);
    assert_eq!(session.phase(), SessionPhase::Ready);
    assert_eq!(session.step(), 0, "prefill does not count as a decode step");
    assert_eq!(session.tokens_consumed(), 1);
    compare_logits(
        "prefill.p1",
        logits.as_f32_slice().unwrap(),
        &fixture("prefill.p1.logits"),
    );
    for layer in 0..LAYERS {
        compare_kv(
            &format!("prefill.p1.present.{layer}.key"),
            &session.caches()[layer].live_keys().unwrap(),
            1,
        );
        compare_kv(
            &format!("prefill.p1.present.{layer}.value"),
            &session.caches()[layer].live_values().unwrap(),
            1,
        );
    }
    assert_eq!(session.plan_misses(), 1);
    assert_eq!(session.plan_cache_len(), 1);
    println!(
        "prefill p1: cache_len {} tokens {} plan {} slots {} vm bytes {}",
        session.cache_len(),
        session.tokens_consumed(),
        session.last_plan().unwrap(),
        session.slot_allocations(),
        session.kv_bytes()
    );

    // seq = 4, 8, 13 against the stored last-position logits and K/V.
    for seq in [4u64, 8, 13] {
        let mut session = cpu_session(&model, &import, &backend);
        let logits = session
            .prefill(&tokens[..seq as usize])
            .unwrap_or_else(|error| panic!("prefill seq={seq}: {error}"));
        assert_eq!(logits.shape().dims(), &[1, 1, VOCAB as u64]);
        assert_eq!(session.cache_len(), seq);
        assert_eq!(session.last_plan_dims().unwrap().total, seq);
        let stored = match seq {
            4 => "prefill.p4.last_logits",
            8 => "prefill.p8.last_logits",
            _ => "prefill.p13.last_logits",
        };
        if fixture_root()
            .join("tensors")
            .join(format!("{stored}.f32"))
            .exists()
        {
            compare_logits(stored, logits.as_f32_slice().unwrap(), &fixture(stored));
        }
        for layer in 0..LAYERS {
            compare_kv(
                &format!("prefill.p{seq}.present.{layer}.key"),
                &session.caches()[layer].live_keys().unwrap(),
                seq,
            );
            compare_kv(
                &format!("prefill.p{seq}.present.{layer}.value"),
                &session.caches()[layer].live_values().unwrap(),
                seq,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Cached decode
// ---------------------------------------------------------------------------

#[test]
fn cached_decode_matches_four_committed_steps() {
    let Some(import) = pinned() else {
        return;
    };
    let (model, _) = prepared(&import, DeviceId::Cpu);
    let backend = CpuBackend::new();
    let tokens = token_ids();
    let expected_extents = [14u64, 15, 16, 17];

    let mut session = cpu_session(&model, &import, &backend);
    let prompt = session.prefill(&tokens[..13]).expect("prefill p13");
    compare_logits(
        "prefill.p13.last (again)",
        prompt.as_f32_slice().unwrap(),
        &fixture("prefill.p13.last_logits"),
    );

    let mut cached_logits = Vec::new();
    for (index, expected_extent) in expected_extents.iter().enumerate() {
        let step = index + 1;
        let token = tokens[13 + index];
        let logits = session
            .decode(&[token])
            .unwrap_or_else(|error| panic!("decode step {step}: {error}"));
        assert_eq!(logits.shape().dims(), &[1, 1, VOCAB as u64]);
        assert_eq!(
            session.cache_len(),
            *expected_extent,
            "step {step}: the live extent must be exactly {expected_extent}"
        );
        assert_eq!(session.step(), step as u64);
        assert_eq!(session.tokens_consumed(), 13 + step as u64);
        let dims = session.last_plan_dims().expect("dims");
        assert_eq!(
            (dims.seq, dims.past, dims.total),
            (1, 13 + index as u64, *expected_extent)
        );

        if step == 1 {
            compare_logits(
                "decode.step1",
                logits.as_f32_slice().unwrap(),
                &fixture("decode.step1.logits"),
            );
        }
        if let Some(reference) = summary(&format!("decode.step{step}.logits")) {
            compare_summary(
                &format!("decode.step{step} (summary)"),
                logits.as_f32_slice().unwrap(),
                &reference,
            );
        }
        for layer in 0..LAYERS {
            compare_kv(
                &format!("decode.step{step}.present.{layer}.key"),
                &session.caches()[layer].live_keys().unwrap(),
                *expected_extent,
            );
            compare_kv(
                &format!("decode.step{step}.present.{layer}.value"),
                &session.caches()[layer].live_values().unwrap(),
                *expected_extent,
            );
        }
        cached_logits.push(logits.as_f32_slice().unwrap().to_vec());
    }

    // Cached-vs-full: prefill the whole sequence in a fresh session and compare
    // its last-position logits with the cached decode of the same position.
    //
    // The fixtures record this comparison against ORT's own full recompute at
    // 5.36e-7 / 5.96e-7 / 5.96e-7 / 4.77e-7. Our two paths differ only in how
    // the K/V entered the attention (appended cache vs recomputed prefix), so
    // the number below is the cached-vs-full figure for this build.
    let (full_model, _) = prepared_with(&import, DeviceId::Cpu, 17, 0, 17, CAPACITY);
    for (index, cached) in cached_logits.iter().enumerate() {
        let length = 14 + index;
        let mut full = cpu_session(&full_model, &import, &backend);
        let logits = full
            .prefill(&tokens[..length])
            .unwrap_or_else(|error| panic!("full recompute p{length}: {error}"));
        let (max_abs, mean_abs) = compare_logits(
            &format!("cached-vs-full.p{length}"),
            cached,
            logits.as_f32_slice().unwrap(),
        );
        if let Some(reference) = slice_summary(&format!("full.p{length}.logits"), length - 1) {
            compare_summary(
                &format!("full.p{length}.logits[{}] (summary)", length - 1),
                logits.as_f32_slice().unwrap(),
                &reference,
            );
        }
        println!(
            "cached-vs-full p{length}: max_abs {max_abs:e}, mean_abs {mean_abs:e} \
             (fixture floor for this step: {})",
            match index {
                0 => "5.36e-7",
                1 => "5.96e-7",
                2 => "5.96e-7",
                _ => "4.77e-7",
            }
        );
        assert!(
            max_abs <= 1e-5,
            "cached and full recompute disagree by {max_abs:e} at p{length}"
        );
    }

    // One plan per distinct (seq, past) and no evictions.
    assert_eq!(session.plan_hits(), 0);
    assert_eq!(
        session.plan_misses(),
        5,
        "prefill 13, then past 13, 14, 15, 16"
    );
    assert_eq!(session.plan_evictions(), 0);
    assert_eq!(session.plan_cache_len(), 5);
    println!(
        "plans: hits {} misses {} evictions {} cached {}",
        session.plan_hits(),
        session.plan_misses(),
        session.plan_evictions(),
        session.plan_cache_len()
    );
    println!(
        "slots: allocations {} reuses {} bytes {}",
        session.slot_allocations(),
        session.slot_reuses(),
        session.slot_bytes_allocated()
    );

    // A second round over the same shapes is served from the cache: no rebuild
    // per step, proven by the hit counters.
    session.reset();
    assert_eq!(session.cache_len(), 0);
    assert_eq!(session.step(), 0);
    assert_eq!(session.tokens_consumed(), 0);
    session.prefill(&tokens[..13]).expect("prefill round 2");
    for token in &tokens[13..17] {
        session.decode(&[*token]).expect("decode round 2");
    }
    assert_eq!(session.plan_misses(), 5, "round 2 rebuilt nothing");
    assert_eq!(session.plan_hits(), 5, "round 2 was served from the cache");
    assert_eq!(session.cache_len(), 17);
    println!(
        "round 2: plan hits {} misses {} (no recompilation per step)",
        session.plan_hits(),
        session.plan_misses()
    );
}

// ---------------------------------------------------------------------------
// 3. Bounds and atomic failure
// ---------------------------------------------------------------------------

/// A backend that delegates to the CPU reference until it is told to fail.
///
/// It exists to make "a failed step changes nothing" testable at the one point
/// where a step can fail after the plan was built: mid-plan execution.
#[derive(Debug)]
struct FailingBackend {
    inner: CpuBackend,
    calls: AtomicU64,
    fail_at: Mutex<Option<u64>>,
    fail_op: Mutex<Option<OpId>>,
}

impl FailingBackend {
    fn new() -> FailingBackend {
        FailingBackend {
            inner: CpuBackend::new(),
            calls: AtomicU64::new(0),
            fail_at: Mutex::new(None),
            fail_op: Mutex::new(None),
        }
    }

    fn arm(&self, fail_at: u64) {
        *self.fail_at.lock().unwrap() = Some(fail_at);
    }

    fn disarm(&self) {
        *self.fail_at.lock().unwrap() = None;
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Backend for FailingBackend {
    fn name(&self) -> &'static str {
        "cpu-failing-test"
    }

    fn device(&self) -> DeviceId {
        self.inner.device()
    }

    fn capabilities(&self) -> &Capabilities {
        self.inner.capabilities()
    }

    fn supports(&self, op: OpId) -> Support {
        self.inner.supports(op)
    }

    fn execute_op(
        &self,
        op: OpId,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
        params: &OpParams,
    ) -> Result<(), Fn3Error> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let fail_at = *self.fail_at.lock().unwrap();
        if fail_at == Some(call) {
            *self.fail_op.lock().unwrap() = Some(op);
            return Err(Fn3Error::UnsupportedOp {
                op,
                backend: "cpu-failing-test",
                reason: "injected failure (FN3-12 atomicity test)".to_string(),
            });
        }
        self.inner.execute_op(op, inputs, outputs, params)
    }
}

#[test]
fn capacity_and_mid_plan_failures_change_nothing() {
    let Some(import) = pinned() else {
        return;
    };
    let (model, _) = prepared(&import, DeviceId::Cpu);
    let backend = CpuBackend::new();
    let tokens = token_ids();

    // Capacity: 13 prompt + 4 decode steps fills 17 exactly; the next one is a
    // typed error and nothing moves.
    let mut session = cpu_session(&model, &import, &backend);
    session.prefill(&tokens[..13]).unwrap();
    for (index, token) in tokens[13..17].iter().enumerate() {
        session.decode(&[*token]).unwrap();
        assert_eq!(session.cache_len(), 14 + index as u64);
    }
    assert_eq!(session.cache_len(), CAPACITY);
    let before = (
        session.cache_len(),
        session.step(),
        session.tokens_consumed(),
        session.phase(),
        session.kv_bytes(),
    );
    let error = session.decode(&[373]).unwrap_err();
    assert!(
        matches!(
            error,
            Fn3Error::CapacityExceeded {
                what: "KV cache live extent",
                requested: 18,
                capacity: 17
            }
        ),
        "{error}"
    );
    assert_eq!(
        (
            session.cache_len(),
            session.step(),
            session.tokens_consumed(),
            session.phase(),
            session.kv_bytes(),
        ),
        before,
        "a capacity failure changes no extent, counter, phase or byte"
    );
    println!(
        "capacity: refusing the 18th token leaves cache_len {} and step {}",
        session.cache_len(),
        session.step()
    );

    // A prefill longer than the envelope's `seq` bound is refused by the
    // planner after the capacity check passes (14 <= 17).
    let mut fresh = cpu_session(&model, &import, &backend);
    let error = fresh.prefill(&tokens[..14]).unwrap_err();
    assert!(
        matches!(error, Fn3Error::DimOutOfEnvelope { .. }),
        "{error}"
    );
    assert_eq!(fresh.cache_len(), 0);
    assert_eq!(fresh.step(), 0);
    assert_eq!(fresh.tokens_consumed(), 0);
    assert_eq!(fresh.phase(), SessionPhase::Empty);
    assert_eq!(fresh.plan_cache_len(), 0);

    // Mid-plan execution failure: the backend rejects one instruction of a
    // prefill. The KV cache, the counters and the phase stay empty.
    let failing = FailingBackend::new();
    let mut session = Session::with_runtime(&model, &import, &failing).expect("session");
    failing.arm(37);
    let error = session.prefill(&tokens[..4]).unwrap_err();
    match &error {
        Fn3Error::UnsupportedOp { reason, .. } => {
            assert!(reason.contains("injected failure"), "{reason}");
        }
        other => panic!("expected the injected failure, found {other:?}"),
    }
    assert_eq!(session.cache_len(), 0);
    assert_eq!(session.step(), 0);
    assert_eq!(session.tokens_consumed(), 0);
    assert_eq!(session.phase(), SessionPhase::Empty);
    let executed = failing.calls();
    assert_eq!(
        executed, 38,
        "the failing instruction and the ones before it ran"
    );
    println!(
        "mid-plan failure at instruction 37 ({}): {} of {} instructions ran, session untouched",
        failing
            .fail_op
            .lock()
            .unwrap()
            .map(OpId::name)
            .unwrap_or("?"),
        executed,
        instruction_count(&import, &model.envelope(), 4, 0)
    );

    // The same backend, disarmed, runs the identical prefill: the failure was
    // the injected one, not the model.
    failing.disarm();
    let failures_before = failing.calls();
    session
        .prefill(&tokens[..4])
        .expect("prefill after the injected failure");
    assert_eq!(session.cache_len(), 4);
    assert_eq!(
        failing.calls() - failures_before,
        instruction_count(&import, &model.envelope(), 4, 0) as u64
    );

    // And a mid-decode failure leaves a Ready session exactly where it was.
    let before = (
        session.cache_len(),
        session.step(),
        session.tokens_consumed(),
    );
    failing.arm(failing.calls() + 5);
    let error = session.decode(&[tokens[4]]).unwrap_err();
    assert!(matches!(error, Fn3Error::UnsupportedOp { .. }), "{error}");
    assert_eq!(
        (
            session.cache_len(),
            session.step(),
            session.tokens_consumed()
        ),
        before,
        "a failed decode changes no extent and no counter"
    );
    assert_eq!(session.phase(), SessionPhase::Ready);
    failing.disarm();
    session
        .decode(&[tokens[4]])
        .expect("decode after the injected failure");
    assert_eq!(session.step(), 1);
    assert_eq!(session.cache_len(), 5);
}

#[test]
fn a_session_without_a_runtime_says_so() {
    let Some(import) = pinned() else {
        return;
    };
    let (model, _) = prepared(&import, DeviceId::Cpu);
    let mut session = Session::new(&model).expect("session");
    for (label, error) in [
        ("prefill", session.prefill(&[464]).unwrap_err()),
        ("decode", session.decode(&[464]).unwrap_err()),
    ] {
        assert!(
            matches!(error, Fn3Error::SessionState { .. }),
            "{label}: {error}"
        );
    }
    // `decode` on an empty session reports the phase, `prefill` reports the
    // missing runtime; neither mutates anything.
    assert_eq!(session.cache_len(), 0);
    assert_eq!(session.step(), 0);
    assert_eq!(session.phase(), SessionPhase::Empty);
    assert_eq!(session.caches().len(), LAYERS);
    assert_eq!(session.caches()[0].capacity(), CAPACITY);
    assert_eq!(
        session.caches()[0].keys().shape().dims(),
        &[1, KV_HEADS, CAPACITY, HEAD_DIM]
    );
    assert_eq!(session.kv_bytes(), 17 * 2 * 64 * 4 * 2 * LAYERS as u64);
    println!(
        "session without a runtime: {} layers, {} KV bytes, key shape {}",
        session.caches().len(),
        session.kv_bytes(),
        session.caches()[0].keys().shape()
    );
}

// ---------------------------------------------------------------------------
// 4. Isolation
// ---------------------------------------------------------------------------

#[test]
fn two_sessions_share_weights_and_nothing_else() {
    let Some(import) = pinned() else {
        return;
    };
    let (model, _) = prepared(&import, DeviceId::Cpu);
    let backend = CpuBackend::new();
    let tokens = token_ids();

    let mut first = cpu_session(&model, &import, &backend);
    let mut second = cpu_session(&model, &import, &backend);
    // Distinct allocations, byte for byte.
    for layer in 0..LAYERS {
        assert_ne!(
            first.caches()[layer].keys().storage().as_bytes().as_ptr(),
            second.caches()[layer].keys().storage().as_bytes().as_ptr()
        );
        assert_ne!(
            first.caches()[layer].values().storage().as_bytes().as_ptr(),
            second.caches()[layer]
                .values()
                .storage()
                .as_bytes()
                .as_ptr()
        );
    }
    // Weights are shared: a model clone shares one Arc per weight, which is the
    // mechanism every session relies on instead of copying 27.9 MB.
    let copy = model.clone();
    assert!(std::sync::Arc::ptr_eq(
        model.weights().handle(0).unwrap(),
        copy.weights().handle(0).unwrap()
    ));

    let first_logits = first.prefill(&tokens[..4]).unwrap();
    let second_logits = second.prefill(&tokens[..13]).unwrap();
    assert_eq!(first.cache_len(), 4);
    assert_eq!(second.cache_len(), 13);
    assert_eq!(first.tokens_consumed(), 4);
    assert_eq!(first.step(), 0);
    assert_eq!(first.plan_misses(), 1);
    assert_eq!(second.plan_misses(), 1);

    // The short session's cache is untouched by the long one and still matches
    // its own fixture.
    for layer in 0..LAYERS {
        compare_kv(
            &format!("prefill.p4.present.{layer}.key"),
            &first.caches()[layer].live_keys().unwrap(),
            4,
        );
    }
    // The second session's decode does not move the first session's extent.
    second.decode(&[tokens[13]]).unwrap();
    assert_eq!(first.cache_len(), 4);
    assert_eq!(first.step(), 0);
    assert_eq!(second.cache_len(), 14);
    assert_eq!(second.step(), 1);

    // Same state, same result: a third session prefilling the same 4 tokens
    // reproduces the first byte for byte.
    let mut third = cpu_session(&model, &import, &backend);
    let third_logits = third.prefill(&tokens[..4]).unwrap();
    assert_eq!(
        first_logits.as_f32_slice().unwrap(),
        third_logits.as_f32_slice().unwrap(),
        "identical sessions must produce identical logits"
    );
    // Plan identity is a function of the model and the shapes, not the session.
    assert_eq!(third.last_plan(), first.last_plan());
    println!(
        "isolation: first (4 tokens, step {}) and second (14 tokens, step {}) kept separate; \
         both built plan {:?}",
        first.step(),
        second.step(),
        third.last_plan().unwrap()
    );
    let _ = second_logits;
}

// ---------------------------------------------------------------------------
// 5. Device execution: every instruction goes through the session's backend
// ---------------------------------------------------------------------------

/// A backend that delegates to the CPU reference and counts every instruction.
#[derive(Debug)]
struct CountingBackend {
    inner: CpuBackend,
    calls: AtomicU64,
    foreign_calls: AtomicU64,
}

impl CountingBackend {
    fn new() -> CountingBackend {
        CountingBackend {
            inner: CpuBackend::new(),
            calls: AtomicU64::new(0),
            foreign_calls: AtomicU64::new(0),
        }
    }
}

impl Backend for CountingBackend {
    fn name(&self) -> &'static str {
        "cpu-counting-test"
    }

    fn device(&self) -> DeviceId {
        self.inner.device()
    }

    fn capabilities(&self) -> &Capabilities {
        self.inner.capabilities()
    }

    fn supports(&self, op: OpId) -> Support {
        self.inner.supports(op)
    }

    fn execute_op(
        &self,
        op: OpId,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
        params: &OpParams,
    ) -> Result<(), Fn3Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        for view in inputs {
            if view.device() != self.device() {
                self.foreign_calls.fetch_add(1, Ordering::SeqCst);
            }
        }
        for output in outputs.iter() {
            if output.device() != self.device() {
                self.foreign_calls.fetch_add(1, Ordering::SeqCst);
            }
        }
        self.inner.execute_op(op, inputs, outputs, params)
    }
}

#[test]
fn every_instruction_is_dispatched_through_the_sessions_backend() {
    let Some(import) = pinned() else {
        return;
    };
    let (model, envelope) = prepared(&import, DeviceId::Cpu);
    let backend = CountingBackend::new();
    let tokens = token_ids();
    let mut session = Session::with_runtime(&model, &import, &backend).expect("session");
    session.prefill(&tokens[..4]).expect("prefill p4");
    session.decode(&[tokens[4]]).expect("decode step 1");

    let expected =
        instruction_count(&import, &envelope, 4, 0) + instruction_count(&import, &envelope, 1, 4);
    assert_eq!(
        backend.calls.load(Ordering::SeqCst),
        expected as u64,
        "one execute_op per plan instruction per step"
    );
    assert_eq!(
        backend.foreign_calls.load(Ordering::SeqCst),
        0,
        "no operand may be tagged for another device"
    );
    println!(
        "cpu dispatch: {} execute_op calls for 2 steps ({} instructions), 0 foreign operands",
        backend.calls.load(Ordering::SeqCst),
        expected
    );
}

#[test]
fn a_model_prepared_for_vulkan_is_never_stepped_by_the_cpu_backend() {
    let Some(import) = pinned() else {
        return;
    };
    let vulkan = DeviceId::Vulkan(fastnn::fn3::PhysicalDeviceId::new(0));
    let (model, _) = prepared(&import, vulkan);
    assert_eq!(model.device(), vulkan);
    let backend = CpuBackend::new();
    let error = Session::with_runtime(&model, &import, &backend)
        .expect_err("a Vulkan model must not accept a CPU backend");
    assert!(
        matches!(
            error,
            Fn3Error::DeviceMismatch {
                expected: DeviceId::Vulkan(_),
                actual: DeviceId::Cpu,
                ..
            }
        ),
        "{error}"
    );
    println!("device rule: {error}");
}

// ---------------------------------------------------------------------------
// 6. Masked rows stay finite and uniform
// ---------------------------------------------------------------------------

#[test]
fn the_mask_the_session_runs_is_finite_and_an_all_masked_row_is_uniform() {
    let Some(import) = pinned() else {
        return;
    };
    let backend = CpuBackend::new();
    let planner = Planner::new(&import, &backend);
    let envelope = envelope(&planner);

    // Every plan the session can build records finite mask constants; this is
    // the static replacement for "a NaN showed up in step 3".
    for (seq, past) in [(1u64, 0u64), (4, 0), (13, 0), (1, 13), (1, 16)] {
        let plan = planner.plan(&envelope, dims(seq, past)).expect("plan");
        for entry in plan.mask_constants() {
            assert!(
                entry.value.is_finite(),
                "seq={seq} past={past}: {} at instruction {} is {}",
                entry.role.name(),
                entry.instruction,
                entry.value
            );
        }
    }

    // An all-masked attention row: feed a fully zero `attention_mask` and every
    // causal row is masked. The graph's finite -FLT_MAX makes that row uniform
    // (not NaN), which is the parity contract in `fixtures/parity/README.md`.
    let plan = planner.plan(&envelope, dims(4, 0)).expect("prefill plan");
    let weights = import.to_model_weights().expect("weights");
    let mut slots = plan.materialise(&weights, DeviceId::Cpu).expect("slots");
    let tokens = [464u32, 2068, 7586, 21831];
    let ids: Vec<i64> = tokens.iter().map(|&token| i64::from(token)).collect();
    let positions: Vec<i64> = (0..4).collect();
    let zeros = vec![0i64; 4];
    for (name, values) in [
        ("input_ids", &ids),
        ("position_ids", &positions),
        ("attention_mask", &zeros),
    ] {
        let input = plan.input(name).expect(name);
        assert_eq!(input.dtype(), DType::I64);
        slots[input.slot() as usize] =
            Some(Tensor::from_i64(input.shape().clone(), values).expect(name));
    }
    for layer in 0..LAYERS {
        for half in ["key", "value"] {
            let name = format!("past_key_values.{layer}.{half}");
            let input = plan.input(&name).expect(&name);
            slots[input.slot() as usize] = Some(
                Tensor::zeros(input.shape().clone(), input.dtype(), DeviceId::Cpu).expect(&name),
            );
        }
    }
    plan.execute(&backend, &mut slots).expect("all-masked plan");

    // The first softmax output has one row per (batch, head, query); row 0 of
    // every head is fully masked by the causal geometry when the mask is zero.
    let softmax = plan
        .instructions()
        .iter()
        .find(|instruction| instruction.op == OpId::Softmax)
        .expect("the pinned graph softmaxes attention scores");
    let slot = plan.slot(softmax.outputs[0]).expect("slot");
    let tensor = slots[softmax.outputs[0] as usize]
        .as_ref()
        .expect("softmax output");
    let values = tensor.as_f32_slice().expect("softmax is f32");
    let width = *tensor.shape().dims().last().expect("rank >= 1") as usize;
    assert!(width > 0);
    let row = &values[..width];
    let expected = 1.0 / width as f32;
    for (index, &value) in row.iter().enumerate() {
        assert!(
            value.is_finite(),
            "all-masked row element {index} is {value}"
        );
        assert!(
            (value - expected).abs() <= 1e-6,
            "all-masked row element {index} is {value}, expected the uniform {expected}"
        );
    }
    // And the model's own logits stay finite.
    let logits = plan.output("logits").expect("logits");
    let logits = slots[logits.slot() as usize]
        .as_ref()
        .expect("logits tensor");
    for (index, &value) in logits.as_f32_slice().expect("f32").iter().enumerate() {
        assert!(value.is_finite(), "logits element {index} is {value}");
    }
    println!(
        "masked rows: all-masked softmax row in slot `{}` is uniform 1/{width}, logits finite",
        slot.name()
    );
}

// ---------------------------------------------------------------------------
// 7. Physical GPU: residency proves device execution
// ---------------------------------------------------------------------------

#[cfg(feature = "vulkan")]
mod gpu {
    use super::*;
    use fastnn::fn3::backend::vulkan::instance::validation_counts;
    use fastnn::fn3::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
    use fastnn::fn3::device::DeviceRequest;

    /// Serialises physical-GPU use and validation-count assertions.
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

    /// The pinned physical adapter, selected by UUID when one is set.
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
                     present{}",
                    match pinned_uuid() {
                        Some(uuid) => format!(" (pinned uuid {uuid})"),
                        None => String::new(),
                    }
                );
            }
            eprintln!(
                "[fn3vk][skip] no physical Vulkan device with a compute queue is present; this \
                 test proves device residency and must not run on a software ICD"
            );
            return None;
        };
        let backend = match VulkanBackend::open_with(
            &DeviceRequest::Vulkan { index: Some(index) },
            VulkanConfig::validated(),
        ) {
            Ok(backend) => backend,
            Err(error) => {
                if require_physical() {
                    panic!("FN3_VK_REQUIRE_PHYSICAL=1 but adapter {index} could not be opened: {error}");
                }
                eprintln!("[fn3vk][skip] could not open adapter {index}: {error}");
                return None;
            }
        };
        let adapter = backend.adapter();
        eprintln!(
            "[fn3vk][test] device=\"{}\" uuid={} api={} driver={} validation={}",
            adapter.name,
            adapter.uuid_string(),
            fastnn::fn3::backend::vulkan::device_info::format_version(adapter.api_version),
            adapter.driver_version_string(),
            backend.validation_enabled()
        );
        assert!(
            !backend.is_software(),
            "the residency test needs a real GPU"
        );
        Some(backend)
    }

    #[test]
    fn vulkan_residency_proves_every_decode_instruction_ran_on_the_device() {
        let _guard = gpu_lock();
        let Some(backend) = open_physical() else {
            return;
        };
        let Some(import) = pinned() else {
            return;
        };
        let counts_before = validation_counts();
        let device = backend.device_id();
        let (model, envelope) = prepared(&import, device);
        let tokens = token_ids();
        // The production path: the session's own backend plans and runs the
        // prompt. No test adapter, no seeded prefix — this is the same
        // `Session::with_runtime` call #62, #63 and #64 make.
        let mut session =
            Session::with_runtime(&model, &import, &*backend).expect("session on the GPU");
        let prompt = session.prefill(&tokens[..13]).expect("gpu prefill p13");
        assert_eq!(prompt.device(), device);
        compare_logits(
            "gpu prefill.p13.last",
            prompt.as_f32_slice().unwrap(),
            &fixture("prefill.p13.last_logits"),
        );
        for layer in 0..LAYERS {
            compare_kv(
                &format!("prefill.p13.present.{layer}.key"),
                &session.caches()[layer].live_keys().unwrap(),
                13,
            );
            compare_kv(
                &format!("prefill.p13.present.{layer}.value"),
                &session.caches()[layer].live_values().unwrap(),
                13,
            );
        }
        assert_eq!(session.cache_len(), 13);
        assert_eq!(session.step(), 0);
        assert!(
            !backend.residency().is_empty(),
            "the prefill must have dispatched on the device"
        );
        println!(
            "gpu prefill: {} residency records for the 13-token prompt",
            backend.residency().len()
        );
        backend.residency().clear();

        let mut records = 0usize;
        for (index, expected_extent) in [14u64, 15, 16, 17].iter().enumerate() {
            let step = index + 1;
            let logits = session
                .decode(&[tokens[13 + index]])
                .unwrap_or_else(|error| panic!("gpu decode step {step}: {error}"));
            assert_eq!(logits.device(), device);
            assert_eq!(session.cache_len(), *expected_extent);
            assert_eq!(session.step(), step as u64);
            if step == 1 {
                compare_logits(
                    "gpu decode.step1",
                    logits.as_f32_slice().unwrap(),
                    &fixture("decode.step1.logits"),
                );
            }
            if let Some(reference) = summary(&format!("decode.step{step}.logits")) {
                compare_summary(
                    &format!("gpu decode.step{step} (summary)"),
                    logits.as_f32_slice().unwrap(),
                    &reference,
                );
            }
            for layer in 0..LAYERS {
                compare_kv(
                    &format!("decode.step{step}.present.{layer}.key"),
                    &session.caches()[layer].live_keys().unwrap(),
                    *expected_extent,
                );
                compare_kv(
                    &format!("decode.step{step}.present.{layer}.value"),
                    &session.caches()[layer].live_values().unwrap(),
                    *expected_extent,
                );
            }
            records += instruction_count(&import, &envelope, 1, 13 + index as u64);
        }

        // Residency: one record per executed instruction, all on this device.
        let ledger = backend.residency();
        assert_eq!(
            ledger.len(),
            records,
            "every instruction of every decode step must be recorded"
        );
        let mut device_local_bindings = 0usize;
        let mut bindings = 0usize;
        for record in ledger.records() {
            assert_eq!(record.backend, "vulkan");
            assert_eq!(record.device, device);
            assert!(
                record.submissions >= 1,
                "record {} submitted nothing",
                record.sequence
            );
            assert!(
                record.workgroups.iter().all(|count| *count > 0),
                "record {} dispatched no workgroup",
                record.sequence
            );
            assert!(
                record.bindings.iter().any(|binding| binding.device_local),
                "record {} has no device-local binding",
                record.sequence
            );
            assert!(
                !record.inputs_read_back,
                "record {} read an input back to the host",
                record.sequence
            );
            assert!(
                record.host_staged,
                "record {} must state that alpha storage is host-staged",
                record.sequence
            );
            assert!(record.proves_device_execution());
            bindings += record.bindings.len();
            device_local_bindings += record
                .bindings
                .iter()
                .filter(|binding| binding.device_local)
                .count();
        }
        assert_eq!(
            device_local_bindings, bindings,
            "every binding is device-local"
        );
        let summary = ledger.summary();
        assert_eq!(summary.input_readbacks, 0);
        assert_eq!(summary.proven_device_executions, records as u64);
        assert_eq!(summary.executions, records as u64);
        assert!(summary.submissions >= records as u64);
        println!(
            "gpu residency: executions {} submissions {} proven {} input_readbacks {} upload {} \
             readback {} device_local_bindings {}/{} host_staged true",
            summary.executions,
            summary.submissions,
            summary.proven_device_executions,
            summary.input_readbacks,
            summary.upload_bytes,
            summary.readback_bytes,
            device_local_bindings,
            bindings
        );
        let (errors, warnings) = validation_counts();
        println!("gpu validation: errors {errors} warnings {warnings} (absolute), delta 0/0");
        println!(
            "gpu session: device {} cache_len {} steps {} plan hits {} misses {}",
            device,
            session.cache_len(),
            session.step(),
            session.plan_hits(),
            session.plan_misses()
        );
        assert_eq!(
            validation_counts(),
            counts_before,
            "the validation layer reported an error or warning; see the [fn3vk][validation] lines"
        );
    }
}
