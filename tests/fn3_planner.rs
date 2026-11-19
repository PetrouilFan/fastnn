//! FN3-10 (#56): bounded shape specialisation and execution plans.
//!
//! # What these tests prove
//!
//! 1. **The pinned graph specialises.** `onnx-community/tiny-random-gpt2-ONNX`
//!    (~27.9 MB, SHA-256 `e18370ba…141a1`) plans for prefill `seq = 1, 4, 8, 13`
//!    at `past = 0` and for decode `seq = 1` at every `past` in `0..18`, with
//!    every slot holding a concrete shape and no symbolic axis left anywhere.
//! 2. **Shape arithmetic is folded, not executed.** The four `i64` `Add`/`Sub`
//!    nodes the exporter emits (`onnx_index` 17, 21, 27, 44) are folded into
//!    static `I64` constants, and no instruction in any plan carries an integer
//!    operand to an arithmetic op.
//! 3. **The bound is real.** [`PlanBound`] is derived by planning the worst
//!    corners of the envelope, and every prefill and decode plan above is
//!    admitted by it; the observed peak byte figure is reported.
//! 4. **Plans are deterministic.** The same model, envelope and dims produce an
//!    identical instruction list and the same [`PlanId`].
//! 5. **Rejection is typed.** A `total` that is not `past + seq`, a sequence
//!    outside the envelope, and a backend that offers only `Partial` support
//!    are all plan-build errors naming the offending thing.
//! 6. **The mask constant is finite.** Every plan records the float constants
//!    it uses to mask attention, and every one of them is finite — the pinned
//!    `-FLT_MAX`, never `-inf`.
//! 7. **The plan actually runs.** Where the pinned ONNX file is present, the
//!    prefill plan for `seq = 1` and `seq = 4` executes on [`CpuBackend`] and
//!    the logits match the committed `fixtures/parity/tiny-gpt2` reference
//!    within the FN3-03 tolerances.
//!
//! # Running the pinned-model tests
//!
//! The graph is not committed (no declared licence). Verify or fetch it with
//!
//! ```text
//! python3 tools/parity/fetch_models.py --suite tiny-gpt2 --verify-only
//! ```
//!
//! and point the tests at it with `FASTNN_FN3_ONNX_MODEL=/path/to/model.onnx`.
//! Without the variable the tests look in
//! `tools/parity/.cache/models/tiny-gpt2/onnx/model.onnx` and, when neither
//! exists, print a reason and pass so the CI lane stays honest.

use std::path::PathBuf;

use fastnn::inference::import::{import_onnx_model, ModelImport};
use fastnn::inference::model::{Plan, Planner, SlotRole};
use fastnn::inference::shape::DimBound;
use fastnn::inference::{
    Backend, CpuBackend, DType, Fn3Error, LiveDims, OpId, OpParams, ShapeEnvelope, Support, Symbol,
};

// ---------------------------------------------------------------------------
// Pinned artifact plumbing (same convention as tests/fn3_import.rs)
// ---------------------------------------------------------------------------

/// Byte length of the pinned graph (`fixtures/parity/tiny-gpt2/model.lock.json`).
const PINNED_BYTES: usize = 27_897_974;

/// FNV-1a 64-bit fingerprint of the pinned file, so a different graph at the
/// configured path is a failure rather than a silently different test.
const PINNED_FNV1A: u64 = 0xc294_d3ab_837f_b1be;

/// Vocabulary of the pinned model; the logits fixtures are `50257` wide.
const VOCAB: usize = 50_257;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn pinned_bytes() -> Option<Vec<u8>> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("FASTNN_FN3_ONNX_MODEL") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tools/parity/.cache/models/tiny-gpt2/onnx/model.onnx"),
    );
    for path in candidates {
        match std::fs::read(&path) {
            Ok(bytes) => return Some(bytes),
            Err(_) => continue,
        }
    }
    println!(
        "SKIP: the pinned 27,897,974-byte ONNX model is not present. Set \
         FASTNN_FN3_ONNX_MODEL or run `python3 tools/parity/fetch_models.py --suite tiny-gpt2`."
    );
    None
}

fn pinned() -> Option<ModelImport> {
    let bytes = pinned_bytes()?;
    assert_eq!(
        bytes.len(),
        PINNED_BYTES,
        "the file at the configured path is not the pinned graph"
    );
    assert_eq!(
        fnv1a(&bytes),
        PINNED_FNV1A,
        "the file at the configured path is not the pinned graph"
    );
    Some(import_onnx_model(&bytes).unwrap_or_else(|error| {
        panic!("the pinned model failed to import:\n{error}");
    }))
}

/// The pinned model's geometry: 2 layers, hidden 128, 2 query/kv heads, 64.
fn dims(seq: u64, past: u64) -> LiveDims {
    LiveDims::new(1, seq, past, 2, 2, 64).expect("total fits")
}

fn decoder_planner<'a>(import: &'a ModelImport, backend: &'a dyn Backend) -> Planner<'a> {
    Planner::new(import, backend)
}

/// The envelope these tests use: batch 1, at most 13 new tokens, a cache of at
/// most 17 tokens, so at most 18 total.
fn envelope(planner: &Planner<'_>) -> ShapeEnvelope {
    planner
        .envelope(1, 13, 17, 18)
        .expect("the pinned geometry fixes the head symbols")
}

fn band(plan: &Plan, seq: u64, past: u64) {
    assert_eq!(plan.dims().seq, seq);
    assert_eq!(plan.dims().past, past);
    assert_eq!(plan.dims().total, past + seq);
}

// ---------------------------------------------------------------------------
// 1. Specialisation of the pinned graph
// ---------------------------------------------------------------------------

#[test]
fn pinned_prefill_and_decode_plans_are_concrete_and_bounded() {
    let Some(import) = pinned() else {
        return;
    };
    let backend = CpuBackend::new();
    let planner = decoder_planner(&import, &backend);
    let envelope = envelope(&planner);
    let bound = planner
        .bound(&envelope, 1)
        .expect("the envelope has a feasible corner");

    println!(
        "bound: {bound}\n  corners: {:?}\n  slots {} instructions {} peak {} bytes \
         allocation {} bytes weights {} bytes",
        bound.corners(),
        bound.max_value_slots(),
        bound.max_instructions(),
        bound.max_peak_bytes(),
        bound.max_allocation_bytes(),
        bound.max_weight_bytes()
    );

    let mut shapes: Vec<(u64, u64)> = vec![(1, 0), (4, 0), (8, 0), (13, 0)];
    for past in 0..18 {
        shapes.push((1, past));
    }

    let mut instruction_counts = Vec::new();
    let mut peaks = Vec::new();
    for (seq, past) in shapes {
        let plan = planner
            .plan(&envelope, dims(seq, past))
            .unwrap_or_else(|error| panic!("prefill/decode plan seq={seq} past={past}: {error}"));
        band(&plan, seq, past);

        // Every slot has a concrete shape. `Shape` cannot carry a symbol at
        // all, so the real assertion is that the table is complete and every
        // scalar count is checked, plus the specific shapes a decoder needs.
        assert_eq!(plan.slots().len(), plan.value_count() as usize);
        for slot in plan.slots() {
            slot.shape().numel().unwrap_or_else(|error| {
                panic!("slot `{}` shape is not allocatable: {error}", slot.name())
            });
            assert!(
                slot.shape().rank() <= 8,
                "slot `{}` has rank {} above the backend limit",
                slot.name(),
                slot.shape().rank()
            );
        }
        // Spot-check the shapes that carry the dynamic symbols.
        let logits = plan.output("logits").expect("logits is a graph output");
        assert_eq!(logits.dtype(), DType::F32);
        assert_eq!(
            logits.shape().dims(),
            &[1, seq, VOCAB as u64],
            "logits for seq={seq} past={past}"
        );
        for layer in 0..2 {
            for half in ["key", "value"] {
                let present = plan
                    .output(&format!("present.{layer}.{half}"))
                    .unwrap_or_else(|| panic!("present.{layer}.{half}"));
                assert_eq!(present.shape().dims(), &[1, 2, past + seq, 64]);
            }
        }
        // The flatten feeding each `Gemm` is the value the *importer* could not
        // resolve, and the whole point of specialisation is that the planner
        // can: `[batch * seq, hidden]` with a concrete first extent.
        let flattened = plan
            .slots()
            .iter()
            .find(|slot| slot.name() == "/transformer/h.0/attn/c_attn/Reshape_output_0")
            .expect("the c_attn projection reshape has a slot");
        assert_eq!(
            flattened.shape().dims(),
            &[seq, 128],
            "the unresolved `-1` must resolve to batch * seq"
        );
        // No instruction may read or write an integer with an arithmetic op.
        for instruction in plan.instructions() {
            if matches!(
                instruction.op,
                OpId::Add
                    | OpId::Sub
                    | OpId::Mul
                    | OpId::Div
                    | OpId::Pow
                    | OpId::MatMul
                    | OpId::Softmax
                    | OpId::ReduceMean
            ) {
                for slot in instruction.inputs.iter().chain(instruction.outputs.iter()) {
                    let dtype = plan.slot(*slot).expect("slot in range").dtype();
                    assert_eq!(
                        dtype,
                        DType::F32,
                        "op {} in the plan for seq={seq} past={past} would execute on {dtype}",
                        instruction.op
                    );
                }
            }
        }
        bound.admits(&plan).unwrap_or_else(|error| {
            panic!("plan seq={seq} past={past} escaped the declared bound: {error}")
        });
        instruction_counts.push(plan.instructions().len());
        peaks.push(plan.peak_bytes());
        println!(
            "seq={seq:2} past={past:2} total={:2}: {} instructions, {} slots, peak {} bytes, \
             allocated {} bytes, weights {} bytes, id {}",
            past + seq,
            plan.instructions().len(),
            plan.value_count(),
            plan.peak_bytes(),
            plan.allocation_bytes(),
            plan.weight_bytes(),
            plan.identity()
        );
    }
    // The instruction structure is a property of the graph, not of the shape:
    // a folding or pruning decision that changed with `seq` would make the
    // declared bound unsound.
    let first = instruction_counts[0];
    assert!(
        instruction_counts.iter().all(|count| *count == first),
        "the instruction count changed with the shape: {instruction_counts:?}"
    );
    assert_eq!(
        first, 144,
        "the pinned graph's specialised instruction count"
    );
    println!("instruction count is {first} for every planned shape");
}

// ---------------------------------------------------------------------------
// 2. Integer shape arithmetic is folded into constants
// ---------------------------------------------------------------------------

#[test]
fn pinned_integer_shape_arithmetic_is_folded_not_executed() {
    let Some(import) = pinned() else {
        return;
    };
    let backend = CpuBackend::new();
    let planner = decoder_planner(&import, &backend);
    let envelope = envelope(&planner);
    let plan = planner.plan(&envelope, dims(4, 9)).expect("prefill plan");

    // The four nodes `tests/fn3_import.rs::integer_arithmetic_is_preserved_rather_than_cast`
    // pins are exactly the shape arithmetic, and each must be folded here.
    for (onnx_index, op) in [
        (17, OpId::Add),
        (21, OpId::Sub),
        (27, OpId::Add),
        (44, OpId::Add),
    ] {
        let folded = plan
            .folded_node(onnx_index)
            .unwrap_or_else(|| panic!("ONNX node {onnx_index} ({op}) was not folded"));
        assert_eq!(folded.op, op, "ONNX node {onnx_index}");
    }
    // ... and each of them is absent from the instruction list. A folded node
    // produces constants, so no plan instruction can carry its ONNX index; the
    // structural check is that every integer arithmetic value is a constant
    // slot (asserted below) and that no arithmetic instruction sees an integer.
    assert!(
        plan.folded().len() >= 4,
        "the four pinned nodes plus the rest of the shape algebra must fold"
    );
    let mut weights = 0usize;
    let mut constants = 0usize;
    let mut inputs = 0usize;
    let mut outputs = 0usize;
    for slot in plan.slots() {
        match slot.role() {
            SlotRole::Weight { .. } => weights += 1,
            SlotRole::Constant { .. } => constants += 1,
            SlotRole::GraphInput => inputs += 1,
            SlotRole::InstructionOutput { .. } => outputs += 1,
        }
    }
    println!(
        "pinned graph: {} imported nodes -> {} instructions + {} folded nodes; \
         {} slots ({} weights, {} folded constants, {} graph inputs, {} instruction outputs); \
         {} instruction operands, {} weight bytes, {} peak bytes",
        import.graph().nodes().len(),
        plan.instructions().len(),
        plan.folded().len(),
        plan.value_count(),
        weights,
        constants,
        inputs,
        outputs,
        plan.instructions()
            .iter()
            .map(|instruction| instruction.inputs.len() + instruction.outputs.len())
            .sum::<usize>(),
        plan.weight_bytes(),
        plan.peak_bytes()
    );
    // Every imported node is accounted for exactly once: an instruction or a
    // fold, never both and never neither (a `Split` still emits one).
    assert_eq!(
        plan.instructions().len() + plan.folded().len(),
        import.graph().nodes().len(),
        "every imported node must be either executed or folded"
    );

    // The folded values are the real extents, not placeholders. Dead
    // constants are pruned (that is why `Add_1`'s own slot is absent: its only
    // consumer folded too), so the assertions use live constants whose value
    // *depends* on the arithmetic.
    let constant_of = |name: &str| -> Vec<i64> {
        let slot = plan
            .slots()
            .iter()
            .find(|slot| slot.name() == name)
            .unwrap_or_else(|| panic!("value `{name}` has a slot"));
        let SlotRole::Constant { index } = slot.role() else {
            panic!(
                "value `{name}` must be a folded constant, found {:?}",
                slot.role()
            );
        };
        plan.constants()[*index as usize].data().as_i64()
    };
    // `Concat_4` is `[Unsqueeze(seq), Unsqueeze(total - seq)]`, so its two
    // elements are the fold of the `Add` (total) and the `Sub` (past).
    assert_eq!(
        constant_of("/transformer/Concat_4_output_0"),
        vec![4, 9],
        "the mask's past extent is total - seq, folded from the pinned Add/Sub pair"
    );
    // `Concat_2` is two copies of `Unsqueeze(seq)`: the causal mask shape.
    assert_eq!(constant_of("/transformer/Concat_2_output_0"), vec![4, 4]);
    // `Concat_5` is `[batch, seq, 384]`: the reshape target of the fused QKV
    // projection, which the importer could not resolve.
    assert_eq!(
        constant_of("/transformer/h.0/attn/c_attn/Concat_1_output_0"),
        vec![1, 4, 384]
    );
    // `Cast_2` is `Less(Range(0, seq), Reshape(Range(0, seq) + 1, [-1, 1]))`,
    // i.e. the whole folded `Range`/`Add`/`Reshape`/`Less` chain. It is live
    // because the float `Where` at ONNX node 59 consumes it.
    let causal = constant_of("/transformer/Cast_2_output_0");
    assert_eq!(causal.len(), 16);
    let expected: Vec<i64> = (0..4)
        .flat_map(|row| (0..4).map(move |column| i64::from(column <= row)))
        .collect();
    assert_eq!(
        causal, expected,
        "the folded causal index range must be row <= column"
    );

    // Every remaining integer instruction is index plumbing, never arithmetic.
    for instruction in plan.instructions() {
        for slot in instruction.inputs.iter().chain(instruction.outputs.iter()) {
            let slot = plan.slot(*slot).expect("slot in range");
            if slot.dtype() == DType::F32 {
                continue;
            }
            assert!(
                !matches!(
                    instruction.op,
                    OpId::Add
                        | OpId::Sub
                        | OpId::Mul
                        | OpId::Div
                        | OpId::Pow
                        | OpId::Neg
                        | OpId::MatMul
                        | OpId::Sqrt
                        | OpId::Tanh
                        | OpId::Softmax
                        | OpId::ReduceMean
                ),
                "op {} would execute on {} slot `{}`",
                instruction.op,
                slot.dtype(),
                slot.name()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Determinism and plan identity
// ---------------------------------------------------------------------------

#[test]
fn pinned_plans_are_deterministic_with_a_stable_identity() {
    let Some(import) = pinned() else {
        return;
    };
    let backend = CpuBackend::new();
    let planner = decoder_planner(&import, &backend);
    let envelope = envelope(&planner);

    for (seq, past) in [(1, 0), (13, 0), (1, 17)] {
        let first = planner.plan(&envelope, dims(seq, past)).expect("plan");
        let second = planner.plan(&envelope, dims(seq, past)).expect("plan");
        assert_eq!(first, second, "seq={seq} past={past} is not deterministic");
        assert_eq!(
            first.identity(),
            second.identity(),
            "seq={seq} past={past} identity is not stable"
        );
        assert_eq!(first.identity().as_bytes().len(), 16);
        assert_eq!(first.identity().to_hex().len(), 32);
    }
    // The identity covers the concrete environment, so every planned
    // environment has its own identity.
    let identities: Vec<_> = [(1, 0), (2, 0), (13, 0), (1, 1), (1, 17)]
        .into_iter()
        .map(|(seq, past)| {
            planner
                .plan(&envelope, dims(seq, past))
                .expect("plan")
                .identity()
        })
        .collect();
    for (left, right) in
        (0..identities.len()).flat_map(|i| (i + 1..identities.len()).map(move |j| (i, j)))
    {
        assert_ne!(
            identities[left], identities[right],
            "an identity that ignores the shapes cannot key a cache"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Typed rejection
// ---------------------------------------------------------------------------

#[test]
fn plan_rejects_a_total_that_is_not_past_plus_seq() {
    let Some(import) = pinned() else {
        return;
    };
    let backend = CpuBackend::new();
    let planner = decoder_planner(&import, &backend);
    let envelope = envelope(&planner);
    let mut bad = dims(4, 0);
    bad.total = 5;
    let error = planner.plan(&envelope, bad).unwrap_err();
    let text = error.to_string();
    assert!(
        text.contains("total (5) must equal past (0) + seq (4) = 4"),
        "{text}"
    );
    assert!(matches!(error, Fn3Error::InvalidShape { .. }), "{error}");
}

#[test]
fn plan_rejects_a_sequence_outside_the_envelope() {
    let Some(import) = pinned() else {
        return;
    };
    let backend = CpuBackend::new();
    let planner = decoder_planner(&import, &backend);
    let envelope = envelope(&planner);
    let error = planner.plan(&envelope, dims(14, 0)).unwrap_err();
    assert!(
        matches!(
            error,
            Fn3Error::DimOutOfEnvelope {
                symbol: "seq",
                requested: 14,
                min: 1,
                max: 13
            }
        ),
        "{error}"
    );
    // A decode step past the cache capacity is the same class of error.
    let error = planner.plan(&envelope, dims(1, 18)).unwrap_err();
    assert!(
        matches!(error, Fn3Error::DimOutOfEnvelope { symbol: "past", .. }),
        "{error}"
    );
}

/// A backend that is honest about `Softmax` being only partially implemented.
///
/// The planner must refuse to build a plan for it: a plan is a promise for one
/// concrete shape, so "partially supported" is not a promise it can keep.
#[derive(Debug)]
struct PartialSoftmax {
    inner: CpuBackend,
    op: OpId,
}

impl Backend for PartialSoftmax {
    fn name(&self) -> &'static str {
        "partial-test"
    }

    fn device(&self) -> fastnn::inference::DeviceId {
        self.inner.device()
    }

    fn capabilities(&self) -> &fastnn::inference::Capabilities {
        self.inner.capabilities()
    }

    fn supports(&self, op: OpId) -> Support {
        if op == self.op {
            Support::partial("test backend: only one axis, keys <= 4 (owner: nobody, test only)")
        } else {
            self.inner.supports(op)
        }
    }

    fn execute_op(
        &self,
        op: OpId,
        inputs: &[fastnn::inference::TensorView<'_>],
        outputs: &mut [fastnn::inference::TensorMut<'_>],
        params: &OpParams,
    ) -> Result<(), Fn3Error> {
        self.inner.execute_op(op, inputs, outputs, params)
    }
}

/// A backend that cannot run `MatMul` at all.
#[derive(Debug)]
struct NoMatMul {
    inner: CpuBackend,
    op: OpId,
}

impl Backend for NoMatMul {
    fn name(&self) -> &'static str {
        "no-matmul-test"
    }

    fn device(&self) -> fastnn::inference::DeviceId {
        self.inner.device()
    }

    fn capabilities(&self) -> &fastnn::inference::Capabilities {
        self.inner.capabilities()
    }

    fn supports(&self, op: OpId) -> Support {
        if op == self.op {
            Support::unsupported("test backend: no matmul kernel (owner: test only)")
        } else {
            self.inner.supports(op)
        }
    }

    fn execute_op(
        &self,
        op: OpId,
        inputs: &[fastnn::inference::TensorView<'_>],
        outputs: &mut [fastnn::inference::TensorMut<'_>],
        params: &OpParams,
    ) -> Result<(), Fn3Error> {
        self.inner.execute_op(op, inputs, outputs, params)
    }
}

#[test]
fn plan_rejects_an_op_without_full_support() {
    let Some(import) = pinned() else {
        return;
    };
    let inner = CpuBackend::new();
    let planner = Planner::new(&import, &inner);
    let envelope = envelope(&planner);

    let partial = PartialSoftmax {
        inner: CpuBackend::new(),
        op: OpId::Softmax,
    };
    let planner = Planner::new(&import, &partial);
    let error = planner
        .plan(&envelope, dims(4, 0))
        .expect_err("a partially supported op must not become a plan");
    match &error {
        Fn3Error::UnsupportedOp {
            op,
            backend,
            reason,
        } => {
            assert_eq!(*op, OpId::Softmax);
            assert_eq!(*backend, "partial-test");
            assert!(reason.contains("only partially supported"), "{reason}");
            assert!(reason.contains("at plan instruction"), "{reason}");
        }
        other => panic!("expected UnsupportedOp, found {other:?}"),
    }

    let absent = NoMatMul {
        inner: CpuBackend::new(),
        op: OpId::MatMul,
    };
    let planner = Planner::new(&import, &absent);
    let error = planner
        .plan(&envelope, dims(4, 0))
        .expect_err("an unsupported op must not become a plan");
    match &error {
        Fn3Error::UnsupportedOp { op, backend, .. } => {
            assert_eq!(*op, OpId::MatMul);
            assert_eq!(*backend, "no-matmul-test");
        }
        other => panic!("expected UnsupportedOp, found {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 5. The mask constant is finite (-FLT_MAX, never -inf)
// ---------------------------------------------------------------------------

#[test]
fn pinned_mask_constants_are_finite_and_are_flt_max() {
    let Some(import) = pinned() else {
        return;
    };
    let backend = CpuBackend::new();
    let planner = decoder_planner(&import, &backend);
    let envelope = envelope(&planner);

    for (seq, past) in [(1, 0), (4, 0), (1, 17), (13, 0)] {
        let plan = planner.plan(&envelope, dims(seq, past)).expect("plan");
        let mask = plan.mask_constants();
        assert!(
            !mask.is_empty(),
            "the pinned graph masks attention, so a plan must record the constant"
        );
        for entry in mask {
            assert!(
                entry.value.is_finite(),
                "{} at instruction {} (op {}) is {}; the parity contract requires a finite \
                 -FLT_MAX so an all-masked softmax row stays uniform",
                entry.role.name(),
                entry.instruction,
                entry.op,
                entry.value
            );
            assert!(
                entry.origin.contains("weight"),
                "a mask constant must name where it came from: {}",
                entry.origin
            );
        }
        let extremes: Vec<f32> = mask
            .iter()
            .filter(|entry| entry.value < 0.0)
            .map(|entry| entry.value)
            .collect();
        assert!(
            !extremes.is_empty(),
            "seq={seq} past={past}: the -FLT_MAX branch must be recorded"
        );
        for value in extremes {
            assert_eq!(
                value,
                -f32::MAX,
                "the masked branch must be exactly the finite -FLT_MAX"
            );
        }
        // A fill and a Where branch both ride on -FLT_MAX in the pinned export.
        assert!(
            mask.iter()
                .any(|entry| entry.role == fastnn::inference::model::MaskRole::Fill),
            "ConstantOfShape's fill value must be checked"
        );
        assert!(
            mask.iter()
                .any(|entry| entry.role == fastnn::inference::model::MaskRole::Selected),
            "Where's selected branch must be checked"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. The plan runs on the CPU reference and matches the fixtures
// ---------------------------------------------------------------------------

/// Load one stored `f32` fixture payload.
fn fixture(root: &std::path::Path, name: &str) -> Vec<f32> {
    let path = root.join("tensors").join(format!("{name}.f32"));
    let bytes =
        std::fs::read(&path).unwrap_or_else(|error| panic!("fixture {}: {error}", path.display()));
    let (chunks, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty(), "fixture is not f32-aligned");
    chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// Compare `actual` to `reference` with the FN3-03 tiny-logits tolerance.
fn compare_logits(label: &str, actual: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(actual.len(), reference.len(), "{label}: length");
    let atol = 1e-5f32;
    let rtol = 1e-4f32;
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
        let bound = atol + rtol * want.abs();
        assert!(
            difference <= bound,
            "{label}: element {index} is {got} vs {want} ({difference} > {bound})"
        );
    }
    let mean_abs = (total_abs / reference.len() as f64) as f32;
    assert!(
        mean_abs <= 2.5e-6,
        "{label}: mean_abs {mean_abs} above the documented 2.5e-6 systematic-bias bound"
    );
    // argmax is only a failure when the reference top-two gap is decisive; for
    // the tiny model the fixtures pin an unambiguous token, so compare it.
    let argmax = |values: &[f32]| {
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
    };
    assert_eq!(argmax(actual), argmax(reference), "{label}: argmax differs");
    println!(
        "{label}: {} logits, max_abs {max_abs:e} (element {}: {:.7} vs {:.7}), mean_abs {mean_abs:e}",
        reference.len(),
        worst.0,
        worst.1,
        worst.2
    );
    max_abs
}

fn token_ids() -> Vec<u32> {
    vec![
        464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257, 640, 11, 612,
        373,
    ]
}

/// Run one prefill plan and return the full logits for every prompt position.
fn run_prefill(import: &ModelImport, seq: u64) -> Vec<f32> {
    use fastnn::inference::{DeviceId, Tensor};

    let backend = CpuBackend::new();
    let planner = Planner::new(import, &backend);
    let envelope = envelope(&planner);
    let dims = dims(seq, 0);
    let plan = planner.plan(&envelope, dims).expect("prefill plan");
    let weights = import.to_model_weights().expect("host weights");
    let mut slots = plan
        .materialise(&weights, DeviceId::Cpu)
        .expect("allocate the plan's slots");

    let tokens = token_ids();
    let ids: Vec<i64> = tokens[..seq as usize]
        .iter()
        .map(|&token| i64::from(token))
        .collect();
    let put = |slots: &mut Vec<Option<Tensor>>, name: &str, tensor: Tensor| {
        let slot = plan
            .input(name)
            .unwrap_or_else(|| panic!("graph input `{name}`"))
            .slot() as usize;
        slots[slot] = Some(tensor);
    };
    put(
        &mut slots,
        "input_ids",
        Tensor::from_i64(plan.input("input_ids").unwrap().shape().clone(), &ids).unwrap(),
    );
    let positions: Vec<i64> = (0..seq as i64).collect();
    put(
        &mut slots,
        "position_ids",
        Tensor::from_i64(
            plan.input("position_ids").unwrap().shape().clone(),
            &positions,
        )
        .unwrap(),
    );
    // GPT-2's mask is all ones; the graph turns it into "not masked".
    let mask = vec![1i64; seq as usize];
    put(
        &mut slots,
        "attention_mask",
        Tensor::from_i64(plan.input("attention_mask").unwrap().shape().clone(), &mask).unwrap(),
    );
    for layer in 0..2 {
        for half in ["key", "value"] {
            let name = format!("past_key_values.{layer}.{half}");
            let input = plan.input(&name).unwrap_or_else(|| panic!("{name}"));
            // past = 0, so the cache input is a zero-element tensor.
            put(
                &mut slots,
                &name,
                Tensor::zeros(input.shape().clone(), input.dtype(), DeviceId::Cpu).unwrap(),
            );
        }
    }

    plan.execute(&backend, &mut slots)
        .unwrap_or_else(|error| panic!("prefill seq={seq} failed to execute: {error}"));

    let logits = plan.output("logits").expect("logits output");
    let tensor = slots[logits.slot() as usize]
        .as_ref()
        .expect("the plan writes logits");
    assert_eq!(tensor.shape().dims(), &[1, seq, VOCAB as u64]);
    tensor.as_f32_slice().expect("logits are f32").to_vec()
}

#[test]
fn pinned_prefill_executes_and_matches_the_committed_fixtures() {
    let Some(import) = pinned() else {
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/tiny-gpt2");
    if !root.join("tensors/prefill.p1.logits.f32").exists() {
        println!(
            "SKIP: {} is not present; cannot compare logits",
            root.display()
        );
        return;
    }

    // prefill.p1.logits is the full [1, 1, 50257] output.
    let logits = run_prefill(&import, 1);
    let reference = fixture(&root, "prefill.p1.logits");
    assert_eq!(reference.len(), VOCAB);
    compare_logits("prefill.p1", &logits[..VOCAB], &reference);

    // prefill.p4.last_logits is the final position of a four-token prefill.
    let logits = run_prefill(&import, 4);
    let reference = fixture(&root, "prefill.p4.last_logits");
    assert_eq!(reference.len(), VOCAB);
    let last = 3 * VOCAB;
    compare_logits("prefill.p4.last", &logits[last..last + VOCAB], &reference);
}

// ---------------------------------------------------------------------------
// 7. Shape-resolution sanity: the importer's 24 unresolved values
// ---------------------------------------------------------------------------

#[test]
fn pinned_importer_unresolved_values_all_resolve_in_the_plan() {
    let Some(import) = pinned() else {
        return;
    };
    let graph = import.graph();
    // A value is "unresolved" when the importer could not name one of its
    // axes: `Dim::Unknown` with a reason. Symbolic axes (`Dim::Symbol`) are
    // resolved, not unresolved — that is the point of specialisation.
    let unresolved: Vec<u32> = graph
        .values()
        .iter()
        .enumerate()
        .filter(|(_, value)| {
            value
                .shape
                .dims()
                .iter()
                .any(|dim| matches!(dim, fastnn::inference::import::Dim::Unknown(_)))
        })
        .map(|(index, _)| index as u32)
        .collect();
    assert_eq!(
        unresolved.len(),
        24,
        "the pinned import resolves all but the 24 flatten/Gemm values"
    );
    let backend = CpuBackend::new();
    let planner = decoder_planner(&import, &backend);
    let envelope = envelope(&planner);
    let plan = planner.plan(&envelope, dims(13, 0)).expect("prefill plan");
    for index in unresolved {
        let value = graph.value(fastnn::inference::import::ValueId::new(index));
        let slot = plan
            .slots()
            .iter()
            .find(|slot| slot.name() == value.name)
            .unwrap_or_else(|| panic!("unresolved value `{}` has no slot", value.name));
        assert!(
            slot.shape().numel().is_ok(),
            "value `{}` was not resolved into a concrete shape",
            value.name
        );
    }
}

// ---------------------------------------------------------------------------
// 8. Repeated calls reuse the plan
// ---------------------------------------------------------------------------

#[test]
fn plan_cache_reuses_the_plan_and_refuses_a_foreign_envelope() {
    use fastnn::inference::model::PlanCache;
    use std::sync::Arc;

    let Some(import) = pinned() else {
        return;
    };
    let backend = CpuBackend::new();
    let planner = decoder_planner(&import, &backend);
    let envelope = envelope(&planner);
    let mut cache = PlanCache::new(envelope, backend.name(), 4).expect("cache");

    let first = cache.plan(&planner, &envelope, dims(4, 0)).expect("plan");
    let again = cache.plan(&planner, &envelope, dims(4, 0)).expect("plan");
    assert!(
        Arc::ptr_eq(&first, &again),
        "a repeated request must reuse the same plan allocation"
    );
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.len(), 1);

    // A different shape is a different key.
    let decode = cache.plan(&planner, &envelope, dims(1, 4)).expect("plan");
    assert!(!Arc::ptr_eq(&first, &decode));
    assert_eq!(cache.misses(), 2);
    assert_eq!(cache.len(), 2);

    // The key is the shape, and the shape is inside the envelope, so the
    // envelope check runs before the cache is consulted.
    let error = cache
        .plan(&planner, &envelope, dims(14, 0))
        .expect_err("out-of-envelope through the cache");
    assert!(
        matches!(error, Fn3Error::DimOutOfEnvelope { .. }),
        "{error}"
    );

    // A plan is only valid for the bounds it was built for: a cache for one
    // envelope must not serve another.
    let other = envelope.with(Symbol::Total, DimBound::new(1, 20).unwrap());
    let error = cache
        .plan(&planner, &other, dims(1, 4))
        .expect_err("foreign envelope");
    assert!(
        error
            .to_string()
            .contains("not the one this cache is keyed on"),
        "{error}"
    );

    // Explicit invalidation.
    cache.clear();
    assert!(cache.is_empty());
    let rebuilt = cache.plan(&planner, &envelope, dims(4, 0)).expect("plan");
    assert!(!Arc::ptr_eq(&first, &rebuilt), "clear() must invalidate");
    assert_eq!(
        rebuilt.identity(),
        first.identity(),
        "content is still the same"
    );
}

// ---------------------------------------------------------------------------
// 9. The envelope helper itself
// ---------------------------------------------------------------------------

#[test]
fn envelope_pins_the_head_symbols_and_rejects_impossible_bounds() {
    let Some(import) = pinned() else {
        return;
    };
    let backend = CpuBackend::new();
    let planner = Planner::new(&import, &backend);
    let envelope = envelope(&planner);
    assert_eq!(envelope.get(Symbol::NHeads), DimBound::exact(2).unwrap());
    assert_eq!(envelope.get(Symbol::NKvHeads), DimBound::exact(2).unwrap());
    assert_eq!(envelope.get(Symbol::HeadDim), DimBound::exact(64).unwrap());
    envelope.validate().unwrap();

    assert!(planner.envelope(1, 13, 17, 12).is_err(), "seq > total");
    assert!(
        planner.envelope(1, 13, 18, 18).is_err(),
        "no room for a token"
    );
    assert!(
        planner.envelope(1, 13, 17, 0).is_err(),
        "total must be positive"
    );

    // Head counts are part of the weights, so a plan cannot change them.
    let error = planner
        .plan(&envelope, LiveDims::new(1, 4, 0, 4, 2, 64).unwrap())
        .unwrap_err();
    assert!(error.to_string().contains("n_heads"), "{error}");
}
