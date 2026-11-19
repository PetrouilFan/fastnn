//! Shared plumbing for the FN3-16 (#62) end-to-end gate.
//!
//! Everything here is mechanism: the pinned model, the committed fixtures, the
//! documented tolerances, model preparation for one device, and the two chains
//! (short prefills and one long prefill/decode run) the CPU and Vulkan tests
//! both execute. The assertions that make the gate a gate live in those tests.
//!
//! # Why a SHA-256 in a test
//!
//! The pinned graph is not committed (upstream declares no licence), so the
//! only thing that separates "the model this evidence is about" from "a model
//! at that path" is a digest. `fnv1a` is cheap and catches most corruption; the
//! SHA-256 is the value the handoff quotes, so the gate verifies that exact
//! value instead of trusting the quotation.

// The Vulkan half of the gate is compiled only with the `vulkan` feature, so a
// default build legitimately leaves its helpers unused. With the feature on,
// dead code is a real warning.
#![cfg_attr(not(feature = "vulkan"), allow(dead_code))]

use std::collections::BTreeMap;
use std::path::PathBuf;

use fastnn::fn3::import::{import_onnx_model, ModelImport};
use fastnn::fn3::model::{ModelWeights, Planner, PreparedModel, Session, SessionPhase};
use fastnn::fn3::shape::ShapeEnvelope;
use fastnn::fn3::{Backend, DeviceId, LiveDims, OpId, Tensor, TensorMeta};

// ---------------------------------------------------------------------------
// The pinned model
// ---------------------------------------------------------------------------

/// Bytes of the pinned graph (`fixtures/parity/tiny-gpt2/model.lock.json`).
pub const PINNED_BYTES: usize = 27_897_974;

/// FNV-1a 64-bit fingerprint of the pinned file.
pub const PINNED_FNV1A: u64 = 0xc294_d3ab_837f_b1be;

/// SHA-256 of the pinned file, as quoted in the FN3-16 handoff.
pub const PINNED_SHA256: &str = "e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1";

/// Vocabulary of the pinned model.
pub const VOCAB: usize = 50_257;

/// Layers, KV heads and per-head width of the pinned model.
pub const LAYERS: usize = 2;
pub const KV_HEADS: u64 = 2;
pub const HEAD_DIM: u64 = 64;

/// Prompt length the fixtures use for the cached-decode chain.
pub const PROMPT: usize = 13;

/// Context length of the committed fixture chain (`decode.step4` ends at 17).
pub const FIXTURE_CONTEXT: u64 = 17;

/// Longest context the long-chain test runs unless `FN3_E2E_CONTEXT` says
/// otherwise. 32 keeps the run bounded while crossing two plan-cache decades
/// and 19 distinct decode shapes.
pub const DEFAULT_LONG_CONTEXT: u64 = 32;

/// The context the long-chain test uses.
pub fn long_context() -> u64 {
    match std::env::var("FN3_E2E_CONTEXT") {
        Ok(value) => value.trim().parse().unwrap_or_else(|error| {
            panic!("FN3_E2E_CONTEXT=`{value}` is not a context length: {error}")
        }),
        Err(_) => DEFAULT_LONG_CONTEXT,
    }
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// SHA-256 of `bytes` as lowercase hex.
pub fn sha256(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = bytes.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut schedule = [0u32; 64];
    for chunk in message.as_chunks::<64>().0 {
        for (index, word) in chunk.as_chunks::<4>().0.iter().enumerate() {
            schedule[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
    let mut hex = String::with_capacity(64);
    for word in state {
        hex.push_str(&format!("{word:08x}"));
    }
    hex
}

/// Load and verify the pinned graph, or print why it is skipped.
pub fn pinned() -> Option<ModelImport> {
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
        let digest = sha256(&bytes);
        assert_eq!(
            digest,
            PINNED_SHA256,
            "the file at {} has the right length but not the pinned bytes",
            path.display()
        );
        println!(
            "model: {} bytes, fnv1a {:#018x}, sha256 {}",
            PINNED_BYTES, PINNED_FNV1A, digest
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

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

pub fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/tiny-gpt2")
}

/// One stored `f32` fixture payload.
pub fn fixture(name: &str) -> Vec<f32> {
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

/// `true` when a payload is committed (some logits are summary-only).
pub fn has_fixture(name: &str) -> bool {
    fixture_root()
        .join("tensors")
        .join(format!("{name}.f32"))
        .exists()
}

/// The fixture manifest, parsed once.
pub fn manifest() -> &'static serde_json::Value {
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
pub struct Summary {
    pub argmax: usize,
    pub first: Vec<f32>,
    pub last: Vec<f32>,
    pub min: f32,
    pub max: f32,
}

pub fn summary(name: &str) -> Option<Summary> {
    let tensors = manifest().get("tensors")?.as_array()?;
    let entry = tensors
        .iter()
        .find(|entry| entry.get("name").and_then(|value| value.as_str()) == Some(name))?;
    summary_of(entry)
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
pub fn token_ids() -> Vec<u32> {
    vec![
        464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257, 640, 11, 612,
        373,
    ]
}

// ---------------------------------------------------------------------------
// Numerics
// ---------------------------------------------------------------------------

pub const ATOL: f32 = 1e-5;
pub const RTOL: f32 = 1e-4;
pub const MEAN_ABS: f64 = 2.5e-6;

pub fn argmax(values: &[f32]) -> usize {
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
pub fn compare_logits(label: &str, actual: &[f32], reference: &[f32]) {
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
        "{label}: {} values, max_abs {max_abs:e} (element {}: {:.7} vs {:.7}), mean_abs {mean_abs:e}",
        reference.len(),
        worst.0,
        worst.1,
        worst.2
    );
}

/// Compare against a summary-only reference.
pub fn compare_summary(label: &str, actual: &[f32], reference: &Summary) {
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

/// Compare a live KV buffer against a stored fixture.
///
/// `fixture_name` is the committed payload name; `tag` is only for the printed
/// line (`cpu` / `vulkan`), so the two runs can be read apart.
pub fn compare_kv(fixture_name: &str, tag: &str, actual: &Tensor, expected_tokens: u64) {
    assert_eq!(
        actual.shape().dims(),
        &[1, KV_HEADS, expected_tokens, HEAD_DIM],
        "{fixture_name} [{tag}]: shape"
    );
    compare_logits(
        &format!("{fixture_name} [{tag}]"),
        actual.as_f32_slice().expect("host KV"),
        &fixture(fixture_name),
    );
}

// ---------------------------------------------------------------------------
// Model preparation
// ---------------------------------------------------------------------------

/// The declared window for a cache of `context` tokens with a 13-token prompt.
///
/// `max_past` leaves room for one more token below `max_total`, exactly as the
/// session tests declare it; the cache capacity is the binding bound.
pub fn envelope(planner: &Planner<'_>, context: u64) -> ShapeEnvelope {
    planner
        .envelope(1, PROMPT as u64, context, context + 1)
        .unwrap_or_else(|error| panic!("the {context}-token window is inconsistent: {error}"))
}

pub fn dims(seq: u64, past: u64) -> LiveDims {
    LiveDims::new(1, seq, past, 2, KV_HEADS, HEAD_DIM).expect("total fits")
}

/// Prepare the pinned model on `device` for a cache of `context` tokens.
pub fn prepared(
    import: &ModelImport,
    device: DeviceId,
    context: u64,
) -> (PreparedModel, ShapeEnvelope) {
    let backend = fastnn::fn3::CpuBackend::new();
    let planner = Planner::new(import, &backend);
    let envelope = envelope(&planner, context);
    let config = import.geometry().kv_cache_config(context, 1);
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

/// How a plan's instructions divide into device work and structural no-ops.
///
/// A plan can contain an instruction whose every output slot is zero-element.
/// The pinned prefill has exactly one: the `ConstantOfShape` that fills the
/// "future" half of a `[seq, total]` causal mask at `total == seq`. There is no
/// byte to write, so the kernel correctly returns before allocating or
/// dispatching; it is not an execution and must not claim to be one. `ops`
/// counts only the byte-producing instructions, which is what the residency
/// ledger must match one-for-one.
#[derive(Debug, Clone, Default)]
pub struct Census {
    /// Instructions in the plan.
    pub total: usize,
    /// Instructions with at least one byte of output (one dispatch each).
    pub with_bytes: usize,
    /// Byte-producing instructions per op.
    pub ops: BTreeMap<OpId, usize>,
    /// Zero-element instructions per op.
    pub zero_byte_ops: BTreeMap<OpId, usize>,
}

impl Census {
    /// Zero-element instructions in the plan.
    pub fn zero_byte(&self) -> usize {
        self.total.saturating_sub(self.with_bytes)
    }

    /// Sum of two censuses, for a run of several steps.
    pub fn merge(&self, other: &Census) -> Census {
        let mut merged = Census {
            total: self.total + other.total,
            with_bytes: self.with_bytes + other.with_bytes,
            ops: self.ops.clone(),
            zero_byte_ops: self.zero_byte_ops.clone(),
        };
        for (op, count) in &other.ops {
            *merged.ops.entry(*op).or_default() += count;
        }
        for (op, count) in &other.zero_byte_ops {
            *merged.zero_byte_ops.entry(*op).or_default() += count;
        }
        merged
    }
}

/// Census of a plan of shape `(seq, past)` on `backend`.
pub fn census(
    import: &ModelImport,
    backend: &dyn Backend,
    envelope: &ShapeEnvelope,
    seq: u64,
    past: u64,
) -> Census {
    let plan = Planner::new(import, backend)
        .plan(envelope, dims(seq, past))
        .unwrap_or_else(|error| panic!("plan ({seq}, {past}): {error}"));
    let mut census = Census {
        total: plan.instructions().len(),
        ..Census::default()
    };
    for instruction in plan.instructions() {
        let bytes = instruction
            .outputs
            .iter()
            .map(|slot| {
                plan.slot(*slot)
                    .expect("instruction output slot")
                    .byte_len()
                    .unwrap_or(0)
            })
            .sum::<u64>();
        if bytes > 0 {
            census.with_bytes += 1;
            *census.ops.entry(instruction.op).or_default() += 1;
        } else {
            *census.zero_byte_ops.entry(instruction.op).or_default() += 1;
        }
    }
    census
}

// ---------------------------------------------------------------------------
// The chains
// ---------------------------------------------------------------------------

/// What one prefill produced, for the caller's own assertions.
pub struct PrefillOutcome {
    pub census: Census,
}

/// Run one prefill of `seq` tokens and check it against the committed fixture.
///
/// `prefill.p1`, `prefill.p4.last` and `prefill.p13.last` are the stored
/// payloads; the KV of every layer is checked for all three lengths.
pub fn check_prefill(
    backend: &dyn Backend,
    import: &ModelImport,
    model: &PreparedModel,
    envelope: &ShapeEnvelope,
    tokens: &[u32],
    seq: u64,
) -> PrefillOutcome {
    let mut session = Session::with_runtime(model, import, backend).expect("session");
    let logits = session
        .prefill(&tokens[..seq as usize])
        .unwrap_or_else(|error| panic!("prefill seq={seq}: {error}"));
    assert_eq!(logits.shape().dims(), &[1, 1, VOCAB as u64]);
    assert_eq!(logits.device(), backend.device());
    assert_eq!(session.cache_len(), seq);
    assert_eq!(session.step(), 0, "prefill does not count as a decode step");
    assert_eq!(session.tokens_consumed(), seq);
    assert_eq!(session.phase(), SessionPhase::Ready);
    let stored = match seq {
        1 => "prefill.p1.logits",
        4 => "prefill.p4.last_logits",
        13 => "prefill.p13.last_logits",
        other => panic!("no committed logits fixture for a {other}-token prefill"),
    };
    compare_logits(
        &format!("{stored} [{}]", backend.name()),
        logits.as_f32_slice().unwrap(),
        &fixture(stored),
    );
    for layer in 0..LAYERS {
        compare_kv(
            &format!("prefill.p{seq}.present.{layer}.key"),
            backend.name(),
            &session.caches()[layer].live_keys().unwrap(),
            seq,
        );
        compare_kv(
            &format!("prefill.p{seq}.present.{layer}.value"),
            backend.name(),
            &session.caches()[layer].live_values().unwrap(),
            seq,
        );
    }
    let census = census(import, backend, envelope, seq, 0);
    println!(
        "[{}] prefill p{seq}: cache_len {} kv_bytes {} plan misses {} instructions {} ({} with \
         bytes, {} zero-element)",
        backend.name(),
        session.cache_len(),
        session.kv_bytes(),
        session.plan_misses(),
        census.total,
        census.with_bytes,
        census.zero_byte()
    );
    PrefillOutcome { census }
}

/// One executed step of the cached-decode chain.
pub struct DecodeStep {
    pub extent: u64,
    pub logits: Vec<f32>,
    pub census: Census,
}

/// What a full chain produced.
pub struct ChainOutcome {
    pub prefill: Vec<f32>,
    pub prefill_census: Census,
    pub steps: Vec<DecodeStep>,
    pub kv_bytes: u64,
    pub plan_hits: u64,
    pub plan_misses: u64,
    pub plan_evictions: u64,
    pub plan_cache_len: usize,
    pub slot_allocations: u64,
    pub slot_reuses: u64,
}

/// Prefill 13 tokens, then decode one token at a time to `context`.
///
/// The first four decode tokens are the fixture sequence, so steps 1–4 are
/// compared with `decode.step{1..4}` (logits and every KV tensor). Beyond that
/// the chain feeds its own argmax, which keeps the run self-contained; those
/// steps have no committed reference, so they assert finiteness, an exact
/// extent, a constant KV allocation and the capacity bound.
///
/// `check_prefill_fixture` decides whether the 13-token prefill is compared
/// with `prefill.p13.last_logits` and its KV (the fixture-context run) or is
/// only checked structurally (a longer context uses the same prompt, so it is
/// the same fixture and is checked once by the caller).
pub fn run_chain(
    backend: &dyn Backend,
    import: &ModelImport,
    model: &PreparedModel,
    envelope: &ShapeEnvelope,
    tokens: &[u32],
    context: u64,
) -> ChainOutcome {
    assert!(
        context >= FIXTURE_CONTEXT,
        "context {context} is below the fixture chain"
    );
    let mut session = Session::with_runtime(model, import, backend).expect("session");
    let prompt = session
        .prefill(&tokens[..PROMPT])
        .unwrap_or_else(|error| panic!("prefill {PROMPT}: {error}"));
    assert_eq!(session.cache_len(), PROMPT as u64);
    // The 13-token prefill is the same fixture chain the main run uses.
    compare_logits(
        &format!("prefill.p13.last_logits [{}]", backend.name()),
        prompt.as_f32_slice().unwrap(),
        &fixture("prefill.p13.last_logits"),
    );
    for layer in 0..LAYERS {
        compare_kv(
            &format!("prefill.p13.present.{layer}.key"),
            backend.name(),
            &session.caches()[layer].live_keys().unwrap(),
            PROMPT as u64,
        );
        compare_kv(
            &format!("prefill.p13.present.{layer}.value"),
            backend.name(),
            &session.caches()[layer].live_values().unwrap(),
            PROMPT as u64,
        );
    }
    let kv_bytes = session.kv_bytes();
    let mut chain = ChainOutcome {
        prefill: prompt.as_f32_slice().unwrap().to_vec(),
        prefill_census: census(import, backend, envelope, PROMPT as u64, 0),
        steps: Vec::new(),
        kv_bytes,
        plan_hits: 0,
        plan_misses: 0,
        plan_evictions: 0,
        plan_cache_len: 0,
        slot_allocations: 0,
        slot_reuses: 0,
    };

    let mut next = 0usize;
    let mut previous = chain.prefill.clone();
    while session.cache_len() < context {
        let step = chain.steps.len() + 1;
        let past = session.cache_len();
        let token = if next < tokens.len() - PROMPT {
            tokens[PROMPT + next]
        } else {
            argmax(&previous) as u32
        };
        next += 1;
        let logits = session
            .decode(&[token])
            .unwrap_or_else(|error| panic!("decode step {step} at past {past}: {error}"));
        assert_eq!(logits.shape().dims(), &[1, 1, VOCAB as u64]);
        assert_eq!(logits.device(), backend.device());
        assert_eq!(session.cache_len(), past + 1, "step {step}: live extent");
        assert_eq!(session.step(), step as u64);
        assert_eq!(session.tokens_consumed(), PROMPT as u64 + step as u64);
        assert_eq!(
            session
                .last_plan_dims()
                .map(|dims| (dims.seq, dims.past, dims.total)),
            Some((1, past, past + 1))
        );
        assert_eq!(
            session.kv_bytes(),
            kv_bytes,
            "step {step}: the KV cache is allocated once and never grows"
        );
        let dims = session.last_plan_dims().unwrap();
        assert!(
            dims.total <= context,
            "step {step}: total {} left the {context}-token window",
            dims.total
        );
        let values = logits.as_f32_slice().unwrap();
        for (index, &value) in values.iter().enumerate() {
            assert!(value.is_finite(), "step {step}: logit {index} is {value}");
        }
        let stored = format!("decode.step{step}.logits");
        if step <= 4 && has_fixture(&stored) {
            compare_logits(
                &format!("{stored} [{}]", backend.name()),
                values,
                &fixture(&stored),
            );
        }
        if let Some(reference) = summary(&format!("decode.step{step}.logits")) {
            compare_summary(
                &format!("decode.step{step} (summary) [{}]", backend.name()),
                values,
                &reference,
            );
        }
        if step <= 4 {
            for layer in 0..LAYERS {
                let extent = past + 1;
                compare_kv(
                    &format!("decode.step{step}.present.{layer}.key"),
                    backend.name(),
                    &session.caches()[layer].live_keys().unwrap(),
                    extent,
                );
                compare_kv(
                    &format!("decode.step{step}.present.{layer}.value"),
                    backend.name(),
                    &session.caches()[layer].live_values().unwrap(),
                    extent,
                );
            }
        }
        let census = census(import, backend, envelope, 1, past);
        println!(
            "[{}] decode step {step} at past {past}: extent {} kv_bytes {} instructions {} ({} \
             with bytes)",
            backend.name(),
            session.cache_len(),
            session.kv_bytes(),
            census.total,
            census.with_bytes
        );
        chain.steps.push(DecodeStep {
            extent: past + 1,
            logits: values.to_vec(),
            census,
        });
        previous = values.to_vec();
    }

    // Capacity is a hard bound: the cache is exactly `context` tokens and the
    // allocation did not grow across the chain.
    assert_eq!(session.cache_len(), context);
    assert_eq!(session.kv_bytes(), kv_bytes);
    for cache in session.caches() {
        assert_eq!(cache.capacity(), context);
        assert_eq!(cache.live_len(), context);
        assert_eq!(cache.allocated_bytes(), kv_bytes / LAYERS as u64);
    }
    // The step after capacity is refused before any mutation, at the exact
    // point the chain stops. This costs no device work.
    let before = (
        session.cache_len(),
        session.step(),
        session.tokens_consumed(),
        session.phase(),
        kv_fingerprint(&session),
    );
    let error = session
        .decode(&[tokens[PROMPT]])
        .expect_err("the chain ends exactly at capacity");
    assert!(
        matches!(
            error,
            fastnn::fn3::Fn3Error::CapacityExceeded {
                what: "KV cache live extent",
                ..
            }
        ),
        "{error}"
    );
    let after = (
        session.cache_len(),
        session.step(),
        session.tokens_consumed(),
        session.phase(),
        kv_fingerprint(&session),
    );
    assert_eq!(before, after, "a refused step changed session state");
    println!(
        "[{}] capacity {}: the next decode was refused with `{error}`, extent/step/tokens/phase \
         and every KV byte unchanged",
        backend.name(),
        context
    );
    chain.plan_hits = session.plan_hits();
    chain.plan_misses = session.plan_misses();
    chain.plan_evictions = session.plan_evictions();
    chain.plan_cache_len = session.plan_cache_len();
    chain.slot_allocations = session.slot_allocations();
    chain.slot_reuses = session.slot_reuses();
    assert_eq!(
        chain.plan_misses as usize,
        1 + (context - PROMPT as u64) as usize,
        "one plan per distinct (seq, past): the prefill and every decode position"
    );
    assert_eq!(
        chain.plan_evictions, 0,
        "the plan cache is bounded and large enough"
    );
    assert!(chain.plan_cache_len <= 64);
    println!(
        "[{}] chain to context {context}: extents {:?}, kv_bytes {} (constant), plan hits {} \
         misses {} evictions {} cached {}",
        backend.name(),
        chain
            .steps
            .iter()
            .map(|step| step.extent)
            .collect::<Vec<_>>(),
        chain.kv_bytes,
        chain.plan_hits,
        chain.plan_misses,
        chain.plan_evictions,
        chain.plan_cache_len
    );
    chain
}

/// Hash of every live KV byte, for "the failure changed nothing" assertions.
pub fn kv_fingerprint(session: &Session<'_>) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for cache in session.caches() {
        for tensor in [cache.live_keys().unwrap(), cache.live_values().unwrap()] {
            let values = tensor.as_f32_slice().expect("host KV");
            for value in values {
                for byte in value.to_bits().to_le_bytes() {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
    }
    hash
}

/// Two sessions over one prepared model, plus the capacity failure.
pub struct IsolationOutcome {
    pub first_prefill: Vec<f32>,
    pub second_prefill: Vec<f32>,
    pub first_steps: Vec<Vec<f32>>,
    pub second_steps: Vec<Vec<f32>>,
}

/// Prove sessions are isolated, identical and bounded at capacity.
///
/// - Two sessions over the *same* `PreparedModel` prefill the same prompt and
///   must produce byte-identical logits.
/// - They take different numbers of decode steps without seeing each other.
/// - A third session reproduces the first one exactly.
/// - At capacity, one more decode is `CapacityExceeded` and leaves the extent,
///   the counters, the phase and every cache byte unchanged.
pub fn check_isolation_and_capacity(
    backend: &dyn Backend,
    import: &ModelImport,
    model: &PreparedModel,
    tokens: &[u32],
    context: u64,
) -> IsolationOutcome {
    let mut first = Session::with_runtime(model, import, backend).expect("first session");
    let mut second = Session::with_runtime(model, import, backend).expect("second session");
    assert!(
        !std::ptr::eq(first.caches()[0].keys(), second.caches()[0].keys()),
        "sessions must not share a KV buffer"
    );
    let first_prefill = first.prefill(&tokens[..PROMPT]).expect("first prefill");
    let second_prefill = second.prefill(&tokens[..PROMPT]).expect("second prefill");
    assert_eq!(
        first_prefill.as_f32_slice().unwrap(),
        second_prefill.as_f32_slice().unwrap(),
        "two sessions over one model must produce identical logits"
    );
    assert_eq!(first.cache_len(), second.cache_len());
    assert_eq!(first.step(), 0);
    assert_eq!(second.step(), 0);

    // Different histories, independent counters.
    assert_eq!(
        context, FIXTURE_CONTEXT,
        "this helper walks the fixture token sequence, so it runs at the fixture context"
    );
    let mut first_steps = Vec::new();
    while first.cache_len() < context {
        let step = first.step() as usize;
        let logits = first
            .decode(&[tokens[PROMPT + step]])
            .unwrap_or_else(|error| panic!("first decode {step}: {error}"));
        first_steps.push(logits.as_f32_slice().unwrap().to_vec());
    }
    let mut second_steps = Vec::new();
    for step in 0..2 {
        let logits = second
            .decode(&[tokens[PROMPT + step]])
            .unwrap_or_else(|error| panic!("second decode {step}: {error}"));
        second_steps.push(logits.as_f32_slice().unwrap().to_vec());
    }
    for (index, (a, b)) in first_steps.iter().zip(&second_steps).enumerate() {
        assert_eq!(a, b, "decode step {} differs between sessions", index + 1);
    }
    assert_eq!(first.step(), context - PROMPT as u64);
    assert_eq!(second.step(), 2);
    assert_eq!(second.cache_len(), PROMPT as u64 + 2);

    // A third session reproduces the first one's prompt exactly.
    let mut third = Session::with_runtime(model, import, backend).expect("third session");
    let third_prefill = third.prefill(&tokens[..PROMPT]).expect("third prefill");
    assert_eq!(
        third_prefill.as_f32_slice().unwrap(),
        first_prefill.as_f32_slice().unwrap()
    );

    // Capacity: the next decode is refused before any mutation.
    let before = (
        first.cache_len(),
        first.step(),
        first.tokens_consumed(),
        first.phase(),
        kv_fingerprint(&first),
    );
    let error = first
        .decode(&[tokens[PROMPT]])
        .expect_err("the cache is at capacity");
    assert!(
        matches!(
            error,
            fastnn::fn3::Fn3Error::CapacityExceeded {
                what: "KV cache live extent",
                ..
            }
        ),
        "{error}"
    );
    let after = (
        first.cache_len(),
        first.step(),
        first.tokens_consumed(),
        first.phase(),
        kv_fingerprint(&first),
    );
    assert_eq!(before, after, "a refused step changed session state");
    println!(
        "[{}] isolation: two sessions identical, extents {}, {}; capacity {} refused, state and \
         KV bytes unchanged",
        backend.name(),
        first.cache_len(),
        second.cache_len(),
        context
    );
    IsolationOutcome {
        first_prefill: first_prefill.as_f32_slice().unwrap().to_vec(),
        second_prefill: second_prefill.as_f32_slice().unwrap().to_vec(),
        first_steps,
        second_steps,
    }
}
