//! FN3-07 (#53): conformance tests for the portable CPU reference backend.
//!
//! # What these tests prove
//!
//! 1. **Fixture parity.** Every tensor in `fixtures/parity/ops/manifest.json`
//!    is loaded, the corresponding kernel is executed through the real
//!    [`Backend::execute_op`] path, and the result is compared with the
//!    manifest's tolerance table using the predicates in
//!    [`fn3_fixtures`] (elementwise bound, non-finite agreement, summary
//!    statistics, the argmax rule).
//! 2. **Real geometry.** Prefill `[1, 13, 128]`, decode `[1, 1, 128]`,
//!    attention `[1, 2, 17, 64]` and the `[50257, 128]` embedding/lm-head
//!    shapes are executed and checked against an `f64` reference written in this
//!    file (a layout self-check; the fixtures, not this file, are the parity
//!    evidence).
//! 3. **Honest support.** Every [`OpId`] the backend reports as `Full` is
//!    executed at least once by [`every_supported_op_executes`], and
//!    `MaskedAttention` stays `Unsupported` with the owning issue named.
//! 4. **Error paths.** Wrong dtype, rank, arity, device, shape envelope,
//!    unsupported shapes, zero-size tensors, and a failed call that must leave
//!    the output buffer untouched.
//!
//! Run with `--nocapture` to see the measured `max_abs`/`mean_abs` for every
//! fixture, which is what the hand-off report quotes.

mod fn3_fixtures;

use fn3_fixtures::{Actual, Agreement, Suite};

use fastnn::inference::{
    Backend, CpuBackend, DType, DeviceId, ExecutionPlan, Fn3Error, Instruction, OpId, OpParams,
    Shape, Strides, Tensor, TensorMeta, TensorMut, TensorView,
};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn cpu() -> DeviceId {
    DeviceId::Cpu
}

fn backend() -> CpuBackend {
    CpuBackend::new()
}

/// Metadata for `dims`, working around a defect in the view contract.
///
/// `Shape::default_strides` gives an interior or trailing extent-1 axis the
/// stride of the next axis, which `Strides::check_no_overlap` then rejects
/// (for example `[2, 1]`, `[128, 1]`, `[1, 3, 1]` and `[2, 1, 4, 4]` all fail).
/// Those shapes are ordinary in a transformer graph -- `[batch, 1, seq, seq]`
/// is the canonical attention mask -- so the harness gives each extent-1 axis
/// the stride `numel`, which is never advanced, keeps the row-major layout
/// (including `is_contiguous` and `as_slice`), and satisfies the validator.
///
/// This is a *workaround*; the hand-off report asks the core owner to exempt
/// extent-1 axes in `Strides::check_no_overlap`.
fn meta_for(dims: &[u64], dtype: DType) -> TensorMeta {
    let shape = Shape::from_dims(dims.iter().copied());
    let numel = shape.numel().unwrap();
    let default = shape.default_strides().unwrap();
    let mut strides = default.as_slice().to_vec();
    let mut patched = false;
    if numel > 0 {
        for (axis, &extent) in shape.dims().iter().enumerate() {
            if extent == 1 && strides[axis] != numel as i64 {
                strides[axis] = numel as i64;
                patched = true;
            }
        }
    }
    if !patched {
        return TensorMeta::contiguous(shape, dtype, cpu()).unwrap();
    }
    TensorMeta::strided(shape, Strides::from_elements(strides), 0, dtype, cpu()).unwrap()
}

fn tensor_from_bytes(dims: &[u64], dtype: DType, bytes: Vec<u8>) -> Tensor {
    let meta = meta_for(dims, dtype);
    assert_eq!(
        bytes.len(),
        meta.required_bytes().unwrap(),
        "test tensor bytes"
    );
    Tensor::from_storage(
        fastnn::inference::Storage::from_bytes(&bytes).unwrap(),
        meta,
    )
    .unwrap()
}

fn f32_tensor(dims: &[u64], values: &[f32]) -> Tensor {
    let meta = meta_for(dims, DType::F32);
    assert_eq!(
        values.len() as u64,
        meta.numel().unwrap(),
        "test tensor elements"
    );
    tensor_from_bytes(
        dims,
        DType::F32,
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
    )
}

fn i64_tensor(dims: &[u64], values: &[i64]) -> Tensor {
    tensor_from_bytes(
        dims,
        DType::I64,
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
    )
}

fn mask_tensor(dims: &[u64], values: &[bool]) -> Tensor {
    tensor_from_bytes(
        dims,
        DType::Bool,
        values.iter().map(|&value| u8::from(value)).collect(),
    )
}

fn zeros_with(dims: &[u64], dtype: DType) -> Tensor {
    let meta = meta_for(dims, dtype);
    let bytes = meta.required_bytes().unwrap();
    Tensor::from_storage(fastnn::inference::Storage::zeroed(bytes).unwrap(), meta).unwrap()
}

fn zeros_f32(dims: &[u64]) -> Tensor {
    zeros_with(dims, DType::F32)
}

fn zeros_i64(dims: &[u64]) -> Tensor {
    zeros_with(dims, DType::I64)
}

fn zeros_mask(dims: &[u64]) -> Tensor {
    zeros_with(dims, DType::Bool)
}

/// Execute `op` with `inputs` into `out` through the public backend path.
fn run_into(
    op: OpId,
    inputs: &[&Tensor],
    out: &mut Tensor,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let views: Vec<TensorView<'_>> = inputs
        .iter()
        .map(|tensor| tensor.as_view().unwrap())
        .collect();
    let mut out_view = out.as_mut_view().unwrap();
    backend().execute_op(op, &views, std::slice::from_mut(&mut out_view), params)
}

/// Execute `op` with pre-built views (used for strided inputs that are not
/// standalone tensors).
fn run_views(
    op: OpId,
    inputs: &[TensorView<'_>],
    out: &mut Tensor,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let mut out_view = out.as_mut_view().unwrap();
    backend().execute_op(op, inputs, std::slice::from_mut(&mut out_view), params)
}

/// Execute `op` into several pre-allocated outputs (`Split`).
fn run_multi(
    op: OpId,
    inputs: &[&Tensor],
    outputs: &mut [&mut Tensor],
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let views: Vec<TensorView<'_>> = inputs
        .iter()
        .map(|tensor| tensor.as_view().unwrap())
        .collect();
    let mut out_views: Vec<TensorMut<'_>> = outputs
        .iter_mut()
        .map(|tensor| tensor.as_mut_view().unwrap())
        .collect();
    backend().execute_op(op, &views, &mut out_views, params)
}

/// Compare an `f32` result with a stored fixture tensor and report the
/// measurement.
fn compare_f32(suite: &Suite, name: &str, actual: &[f32], shape: &[u64]) -> Agreement {
    let reference = suite.tensor(name);
    let agreement = reference
        .check(Actual::F32(actual), shape)
        .unwrap_or_else(|error| panic!("fixture {name}: {error}"));
    reference
        .check_summary(Actual::F32(actual), shape)
        .unwrap_or_else(|error| panic!("fixture {name} summary: {error}"));
    reference
        .check_slices(Actual::F32(actual), shape)
        .unwrap_or_else(|error| panic!("fixture {name} slices: {error}"));
    println!(
        "[fixture] {name:34} shape {shape:?} {agreement}  tolerance {}",
        reference.tolerance()
    );
    agreement
}

/// Deterministic pseudo-random `f32` values in `[low, high)`.
fn pseudo_values(count: usize, seed: u64, low: f32, high: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 33) as f32) / ((1u64 << 31) as f32);
            low + (high - low) * unit
        })
        .collect()
}

/// Assert `actual` is within `atol + rtol*|reference|` of an `f64` reference and
/// return the measured maximum absolute error.
fn assert_close(label: &str, actual: &[f32], reference: &[f64], atol: f64, rtol: f64) -> f64 {
    assert_eq!(actual.len(), reference.len(), "{label}: length");
    let mut max_abs = 0.0f64;
    let mut mean_abs = 0.0f64;
    for (position, (&actual, &reference)) in actual.iter().zip(reference.iter()).enumerate() {
        let difference = (f64::from(actual) - reference).abs();
        max_abs = max_abs.max(difference);
        mean_abs += difference;
        assert!(
            difference <= atol + rtol * reference.abs(),
            "{label}: element {position} is {actual}, reference {reference}, |diff|={difference:e}"
        );
    }
    mean_abs /= actual.len().max(1) as f64;
    println!("[self-check] {label:46} max_abs={max_abs:e} mean_abs={mean_abs:e} (bound {atol:e}+{rtol:e}|ref|)");
    max_abs
}

/// Naive `f64` `[M,K] @ [K,N]`.
fn reference_matmul_2d(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut accumulator = 0.0f64;
            for step in 0..k {
                accumulator += f64::from(a[row * k + step]) * f64::from(b[step * n + column]);
            }
            out[row * n + column] = accumulator;
        }
    }
    out
}

/// A strided transpose view of the last two axes of a contiguous 4-D tensor
/// (`[.., M, N] -> [.., N, M]`), used for `Q @ K^T`.
fn transpose_last_two(tensor: &Tensor) -> TensorMeta {
    let dims = tensor.shape().dims().to_vec();
    let rank = dims.len();
    let strides = tensor.meta().strides().as_slice().to_vec();
    let mut new_dims = dims.clone();
    new_dims.swap(rank - 2, rank - 1);
    let mut new_strides = strides.clone();
    new_strides.swap(rank - 2, rank - 1);
    TensorMeta::strided(
        Shape::from_dims(new_dims),
        Strides::from_elements(new_strides),
        tensor.meta().offset(),
        DType::F32,
        cpu(),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// 1. Fixture parity: embedding, layer norm, matmul, gelu, softmax, attention
// ---------------------------------------------------------------------------

#[test]
fn fixture_embedding_gather() {
    let suite = Suite::load("ops");
    let table = suite.tensor("embedding.table").f32_values();
    let indices = suite.tensor("embedding.indices").i64_values();
    let table = f32_tensor(&[17, 7], &table);
    let indices = i64_tensor(&[2, 5], &indices);
    let mut out = zeros_f32(&[2, 5, 7]);
    run_into(
        OpId::Gather,
        &[&table, &indices],
        &mut out,
        &OpParams::default(),
    )
    .unwrap();
    let values = out.as_f32_slice().unwrap().to_vec();
    // Embedding lookups are exact: the fixture tolerance is atol = rtol = 0.
    compare_f32(&suite, "embedding.out", &values, &[2, 5, 7]);
}

#[test]
fn fixture_layernorm() {
    let suite = Suite::load("ops");
    let input = suite.tensor("layernorm.input").f32_values();
    let weight = suite.tensor("layernorm.weight").f32_values();
    let bias = suite.tensor("layernorm.bias").f32_values();
    let input = f32_tensor(&[2, 3, 5], &input);
    let weight = f32_tensor(&[5], &weight);
    let bias = f32_tensor(&[5], &bias);
    let mut out = zeros_f32(&[2, 3, 5]);
    run_into(
        OpId::LayerNorm,
        &[&input, &weight, &bias],
        &mut out,
        &OpParams::default().with_axes([-1]).with_epsilon(1e-5),
    )
    .unwrap();
    let values = out.as_f32_slice().unwrap().to_vec();
    compare_f32(&suite, "layernorm.out", &values, &[2, 3, 5]);
}

#[test]
fn fixture_layernorm_model_shape() {
    let suite = Suite::load("ops");
    let input = suite.tensor("layernorm.model_shape.input").f32_values();
    let weight = suite.tensor("layernorm.model_shape.weight").f32_values();
    let bias = suite.tensor("layernorm.model_shape.bias").f32_values();
    let input = f32_tensor(&[1, 4, 128], &input);
    let weight = f32_tensor(&[128], &weight);
    let bias = f32_tensor(&[128], &bias);
    let mut out = zeros_f32(&[1, 4, 128]);
    run_into(
        OpId::LayerNorm,
        &[&input, &weight, &bias],
        &mut out,
        &OpParams::default().with_axes([-1]).with_epsilon(1e-5),
    )
    .unwrap();
    let values = out.as_f32_slice().unwrap().to_vec();
    compare_f32(&suite, "layernorm.model_shape.out", &values, &[1, 4, 128]);
}

#[test]
fn fixture_matmul_2d_k3() {
    let suite = Suite::load("ops");
    let a = f32_tensor(&[2, 3], &suite.tensor("matmul.2d_k3.a").f32_values());
    let b = f32_tensor(&[3, 4], &suite.tensor("matmul.2d_k3.b").f32_values());
    let mut out = zeros_f32(&[2, 4]);
    run_into(OpId::MatMul, &[&a, &b], &mut out, &OpParams::default()).unwrap();
    let values = out.as_f32_slice().unwrap().to_vec();
    compare_f32(&suite, "matmul.2d_k3.out", &values, &[2, 4]);
}

#[test]
fn fixture_matmul_gemm_k13_bias() {
    let suite = Suite::load("ops");
    let a = f32_tensor(
        &[7, 13],
        &suite.tensor("matmul.gemm_k13_bias.a").f32_values(),
    );
    let w = f32_tensor(
        &[5, 13],
        &suite.tensor("matmul.gemm_k13_bias.w").f32_values(),
    );
    let bias = f32_tensor(
        &[5],
        &suite.tensor("matmul.gemm_k13_bias.bias").f32_values(),
    );

    // Gemm(transB=1) is `MatMul(a, w^T) + bias`: the transpose is a strided view.
    let transposed = TensorMeta::strided(
        Shape::from_dims([13, 5]),
        Strides::from_elements(vec![1, 13]),
        0,
        DType::F32,
        cpu(),
    )
    .unwrap();
    let w_view = TensorView::new(w.storage(), transposed).unwrap();
    let a_view = a.as_view().unwrap();
    let mut product = zeros_f32(&[7, 5]);
    run_views(
        OpId::MatMul,
        &[a_view, w_view],
        &mut product,
        &OpParams::default(),
    )
    .unwrap();

    let mut out = zeros_f32(&[7, 5]);
    run_into(
        OpId::Add,
        &[&product, &bias],
        &mut out,
        &OpParams::default(),
    )
    .unwrap();
    let values = out.as_f32_slice().unwrap().to_vec();
    compare_f32(&suite, "matmul.gemm_k13_bias.out", &values, &[7, 5]);
}

#[test]
fn fixture_matmul_batched_4d() {
    let suite = Suite::load("ops");
    let a = f32_tensor(
        &[2, 3, 5, 7],
        &suite.tensor("matmul.batched_4d.a").f32_values(),
    );
    let b = f32_tensor(
        &[2, 3, 7, 11],
        &suite.tensor("matmul.batched_4d.b").f32_values(),
    );
    let mut out = zeros_f32(&[2, 3, 5, 11]);
    run_into(OpId::MatMul, &[&a, &b], &mut out, &OpParams::default()).unwrap();
    let values = out.as_f32_slice().unwrap().to_vec();
    compare_f32(&suite, "matmul.batched_4d.out", &values, &[2, 3, 5, 11]);
}

#[test]
fn fixture_matmul_model_proj() {
    let suite = Suite::load("ops");
    // Rank-3 by rank-2: the batch axes come from the left operand only.
    let a = f32_tensor(
        &[1, 13, 128],
        &suite.tensor("matmul.model_proj.a").f32_values(),
    );
    let b = f32_tensor(&[128, 8], &suite.tensor("matmul.model_proj.b").f32_values());
    let mut out = zeros_f32(&[1, 13, 8]);
    run_into(OpId::MatMul, &[&a, &b], &mut out, &OpParams::default()).unwrap();
    let values = out.as_f32_slice().unwrap().to_vec();
    compare_f32(&suite, "matmul.model_proj.out", &values, &[1, 13, 8]);
}

#[test]
fn fixture_matmul_tail_k65() {
    let suite = Suite::load("ops");
    let a = f32_tensor(&[3, 65], &suite.tensor("matmul.tail_k65.a").f32_values());
    let b = f32_tensor(&[65, 7], &suite.tensor("matmul.tail_k65.b").f32_values());
    let mut out = zeros_f32(&[3, 7]);
    run_into(OpId::MatMul, &[&a, &b], &mut out, &OpParams::default()).unwrap();
    let values = out.as_f32_slice().unwrap().to_vec();
    compare_f32(&suite, "matmul.tail_k65.out", &values, &[3, 7]);
}

#[test]
fn fixture_gelu_erf_form() {
    let suite = Suite::load("ops");
    let input = f32_tensor(&[69], &suite.tensor("gelu.input").f32_values());
    let mut out = zeros_f32(&[69]);
    run_into(OpId::Gelu, &[&input], &mut out, &OpParams::default()).unwrap();
    let values = out.as_f32_slice().unwrap().to_vec();
    compare_f32(&suite, "gelu.erf.out", &values, &[69]);

    // The same result must be reachable from the primitive composition
    // `0.5*x*(1+erf(x/sqrt(2)))`, so `Erf` and `Gelu` cannot drift apart.
    let mut erf_out = zeros_f32(&[69]);
    run_into(OpId::Erf, &[&input], &mut erf_out, &OpParams::default()).unwrap();
    let scaled_input = f32_tensor(
        &[69],
        &input
            .as_f32_slice()
            .unwrap()
            .iter()
            .map(|value| value * std::f32::consts::FRAC_1_SQRT_2)
            .collect::<Vec<f32>>(),
    );
    let mut erf_scaled = zeros_f32(&[69]);
    run_into(
        OpId::Erf,
        &[&scaled_input],
        &mut erf_scaled,
        &OpParams::default(),
    )
    .unwrap();
    let one = f32_tensor(&[], &[1.0]);
    let half = f32_tensor(&[], &[0.5]);
    let mut plus_one = zeros_f32(&[69]);
    run_into(
        OpId::Add,
        &[&erf_scaled, &one],
        &mut plus_one,
        &OpParams::default(),
    )
    .unwrap();
    let mut halved_input = zeros_f32(&[69]);
    run_into(
        OpId::Mul,
        &[&input, &half],
        &mut halved_input,
        &OpParams::default(),
    )
    .unwrap();
    let mut composed = zeros_f32(&[69]);
    run_into(
        OpId::Mul,
        &[&halved_input, &plus_one],
        &mut composed,
        &OpParams::default(),
    )
    .unwrap();
    // Not bit-identical: `Gelu` evaluates the whole expression in `f64`, while
    // the decomposition rounds `x/sqrt(2)` to `f32` before feeding `Erf`. The
    // two agree to float32 noise, which is the point of the check.
    let composed_values = composed.as_f32_slice().unwrap();
    for (position, (&composed, &gelu)) in composed_values.iter().zip(values.iter()).enumerate() {
        assert!(
            (composed - gelu).abs() <= 2e-7 + 1e-6 * gelu.abs(),
            "Gelu({}) = {gelu} but its erf decomposition is {composed}",
            input.as_f32_slice().unwrap()[position]
        );
    }
}

#[test]
fn fixture_gelu_tanh_form_as_the_model_decomposes_it() {
    let suite = Suite::load("ops");
    let input_values = suite.tensor("gelu.input").f32_values();
    let input = f32_tensor(&[69], &input_values);

    // Exactly the node chain of `/transformer/h.0/mlp/act`: the constants are
    // the graph's own f32 initializers.
    let three = f32_tensor(&[], &[3.0]);
    let c0 = f32_tensor(&[], &[0.044715]);
    let c1 = f32_tensor(&[], &[0.797_884_6]);
    let one = f32_tensor(&[], &[1.0]);
    let half = f32_tensor(&[], &[0.5]);

    let mut cube = zeros_f32(&[69]);
    run_into(
        OpId::Pow,
        &[&input, &three],
        &mut cube,
        &OpParams::default(),
    )
    .unwrap();
    let mut scaled_cube = zeros_f32(&[69]);
    run_into(
        OpId::Mul,
        &[&cube, &c0],
        &mut scaled_cube,
        &OpParams::default(),
    )
    .unwrap();
    let mut inner = zeros_f32(&[69]);
    run_into(
        OpId::Add,
        &[&input, &scaled_cube],
        &mut inner,
        &OpParams::default(),
    )
    .unwrap();
    let mut argument = zeros_f32(&[69]);
    run_into(
        OpId::Mul,
        &[&inner, &c1],
        &mut argument,
        &OpParams::default(),
    )
    .unwrap();
    let mut activation = zeros_f32(&[69]);
    run_into(
        OpId::Tanh,
        &[&argument],
        &mut activation,
        &OpParams::default(),
    )
    .unwrap();
    let mut shifted = zeros_f32(&[69]);
    run_into(
        OpId::Add,
        &[&activation, &one],
        &mut shifted,
        &OpParams::default(),
    )
    .unwrap();
    let mut halved = zeros_f32(&[69]);
    run_into(
        OpId::Mul,
        &[&input, &half],
        &mut halved,
        &OpParams::default(),
    )
    .unwrap();
    let mut out = zeros_f32(&[69]);
    run_into(
        OpId::Mul,
        &[&halved, &shifted],
        &mut out,
        &OpParams::default(),
    )
    .unwrap();

    let values = out.as_f32_slice().unwrap().to_vec();
    compare_f32(&suite, "gelu.tanh.out", &values, &[69]);
    // The model uses the tanh form; assert the extreme values are what the
    // decomposition must produce (0 for large negative, x for large positive).
    let last = input_values.len() - 1;
    assert_eq!(values[last], 1.0e4);
    assert_eq!(values[last - 1], 10.0);
    assert_eq!(values[last - 2], 0.0);
}

#[test]
fn fixture_softmax_unmasked() {
    let suite = Suite::load("ops");
    let input = f32_tensor(
        &[2, 3, 4, 5],
        &suite.tensor("softmax.unmasked.input").f32_values(),
    );
    let mut out = zeros_f32(&[2, 3, 4, 5]);
    run_into(
        OpId::Softmax,
        &[&input],
        &mut out,
        &OpParams::default().with_axes([-1]),
    )
    .unwrap();
    let values = out.as_f32_slice().unwrap().to_vec();
    compare_f32(&suite, "softmax.unmasked.out", &values, &[2, 3, 4, 5]);
}

/// Add the fixture's additive mask and run `Softmax`, the way the graph does.
fn masked_softmax(suite: &Suite, case: &str, dims: &[u64]) -> Vec<f32> {
    let input = f32_tensor(
        dims,
        &suite.tensor(&format!("softmax.{case}.input")).f32_values(),
    );
    let mask = f32_tensor(
        dims,
        &suite.tensor(&format!("softmax.{case}.mask")).f32_values(),
    );
    let mut scores = zeros_f32(dims);
    run_into(
        OpId::Add,
        &[&input, &mask],
        &mut scores,
        &OpParams::default(),
    )
    .unwrap();
    let mut out = zeros_f32(dims);
    run_into(
        OpId::Softmax,
        &[&scores],
        &mut out,
        &OpParams::default().with_axes([-1]),
    )
    .unwrap();
    out.as_f32_slice().unwrap().to_vec()
}

#[test]
fn fixture_softmax_causal_mask() {
    let suite = Suite::load("ops");
    let values = masked_softmax(&suite, "causal", &[1, 2, 4, 4]);
    compare_f32(&suite, "softmax.causal.out", &values, &[1, 2, 4, 4]);
}

#[test]
fn fixture_softmax_padded_mask() {
    let suite = Suite::load("ops");
    let values = masked_softmax(&suite, "padded", &[1, 1, 3, 5]);
    compare_f32(&suite, "softmax.padded.out", &values, &[1, 1, 3, 5]);
}

#[test]
fn fixture_softmax_all_masked_row_is_uniform() {
    let suite = Suite::load("ops");
    let values = masked_softmax(&suite, "all_masked", &[1, 1, 2, 4]);
    compare_f32(&suite, "softmax.all_masked.out", &values, &[1, 1, 2, 4]);
    // The conformance requirement: finite, exactly uniform probabilities.
    let second_row = &values[4..8];
    assert_eq!(
        second_row,
        &[0.25f32, 0.25, 0.25, 0.25],
        "a fully masked row must be uniform"
    );
    assert!(values.iter().all(|value| value.is_finite()));
}

#[test]
fn fixture_masked_attention_decomposition() {
    let suite = Suite::load("ops");
    let q = f32_tensor(&[2, 2, 4, 5], &suite.tensor("attention.q").f32_values());
    let k = f32_tensor(&[2, 2, 4, 5], &suite.tensor("attention.k").f32_values());
    let v = f32_tensor(&[2, 2, 4, 5], &suite.tensor("attention.v").f32_values());
    // `[2, 1, 4, 4]`: the size-1 axis broadcasts over the two heads.
    let mask = f32_tensor(&[2, 1, 4, 4], &suite.tensor("attention.mask").f32_values());

    // scores = Q @ K^T
    let scores_meta = transpose_last_two(&k);
    let k_transposed = TensorView::new(k.storage(), scores_meta).unwrap();
    let q_view = q.as_view().unwrap();
    let mut scores = zeros_f32(&[2, 2, 4, 4]);
    run_views(
        OpId::MatMul,
        &[q_view, k_transposed],
        &mut scores,
        &OpParams::default(),
    )
    .unwrap();

    // scores / sqrt(head_dim); the reference divides in float64, so divide here
    // too rather than multiplying by a reciprocal.
    let scale = f32_tensor(&[], &[(5.0f32).sqrt()]);
    let mut scaled = zeros_f32(&[2, 2, 4, 4]);
    run_into(
        OpId::Div,
        &[&scores, &scale],
        &mut scaled,
        &OpParams::default(),
    )
    .unwrap();

    // + additive mask ([2,1,4,4] broadcasts over heads)
    let mut masked = zeros_f32(&[2, 2, 4, 4]);
    run_into(
        OpId::Add,
        &[&scaled, &mask],
        &mut masked,
        &OpParams::default(),
    )
    .unwrap();

    // softmax over keys
    let mut probs = zeros_f32(&[2, 2, 4, 4]);
    run_into(
        OpId::Softmax,
        &[&masked],
        &mut probs,
        &OpParams::default().with_axes([-1]),
    )
    .unwrap();

    // out = probs @ V
    let mut out = zeros_f32(&[2, 2, 4, 5]);
    run_into(OpId::MatMul, &[&probs, &v], &mut out, &OpParams::default()).unwrap();

    let probability_values = probs.as_f32_slice().unwrap().to_vec();
    compare_f32(
        &suite,
        "attention.probs",
        &probability_values,
        &[2, 2, 4, 4],
    );
    let out_values = out.as_f32_slice().unwrap().to_vec();
    compare_f32(&suite, "attention.out", &out_values, &[2, 2, 4, 5]);

    // Batch 1, every head, query row 2 is fully masked by the fixture.
    for head in 0..2 {
        let batch = 1;
        let heads = 2;
        let base = (batch * heads + head) * 4 * 4 + 2 * 4;
        assert_eq!(
            &probability_values[base..base + 4],
            &[0.25f32, 0.25, 0.25, 0.25],
            "all-masked query row must be uniform for head {head}"
        );
    }
    assert!(out_values.iter().all(|value| value.is_finite()));
}

// ---------------------------------------------------------------------------
// 2. Real prefill/decode geometry
// ---------------------------------------------------------------------------

#[test]
fn real_shape_prefill_projection() {
    // The pinned model's c_attn projection: [seq, 128] @ [128, 384].
    let sequence = 13usize;
    let hidden = 128usize;
    let projected = 384usize;
    let a = f32_tensor(
        &[1, sequence as u64, hidden as u64],
        &pseudo_values(sequence * hidden, 0x51_0001, -2.0, 2.0),
    );
    let b = f32_tensor(
        &[hidden as u64, projected as u64],
        &pseudo_values(hidden * projected, 0x51_0002, -1.0, 1.0),
    );
    let mut out = zeros_f32(&[1, sequence as u64, projected as u64]);
    run_into(OpId::MatMul, &[&a, &b], &mut out, &OpParams::default()).unwrap();
    let reference = reference_matmul_2d(
        a.as_f32_slice().unwrap(),
        sequence,
        hidden,
        b.as_f32_slice().unwrap(),
        projected,
    );
    // K = 128: the fixture policy scales atol by sqrt(K/64).
    let atol = 2.0e-5 * (hidden as f64 / 64.0).sqrt();
    assert_close(
        "prefill [13,128] @ [128,384]",
        out.as_f32_slice().unwrap(),
        &reference,
        atol,
        1.0e-5,
    );
}

#[test]
fn real_shape_decode_logits_projection() {
    // Decode step: one token against the full 50,257-row vocabulary
    // (`lm_head`, `[1,1,128] @ [128,50257]`). This is the largest single op the
    // model executes and the reason the kernel must not materialise a copy of
    // either operand.
    let hidden = 128usize;
    let vocabulary = 50_257usize;
    let a_values = pseudo_values(hidden, 0x52_0001, -1.0, 1.0);
    let b_values = pseudo_values(hidden * vocabulary, 0x52_0002, -1.0, 1.0);
    let a = f32_tensor(&[1, 1, hidden as u64], &a_values);
    let b = f32_tensor(&[hidden as u64, vocabulary as u64], &b_values);
    let mut out = zeros_f32(&[1, 1, vocabulary as u64]);
    run_into(OpId::MatMul, &[&a, &b], &mut out, &OpParams::default()).unwrap();
    let actual = out.as_f32_slice().unwrap();
    assert_eq!(actual.len(), vocabulary);

    // Check every column: the f64 reference is one dot of length 128 per column,
    // so this stays cheap while still covering the whole vocabulary.
    let reference = reference_matmul_2d(&a_values, 1, hidden, &b_values, vocabulary);
    let atol = 1.0e-5 * (hidden as f64 / 64.0).sqrt();
    assert_close(
        "decode [1,128] @ [128,50257]",
        actual,
        &reference,
        atol,
        1.0e-4,
    );
}

#[test]
fn real_shape_attention_geometry_with_mask() {
    // [1, 2 heads, 17 keys, 64 head_dim], a sequence length that is not a power
    // of two, with a causal additive mask.
    let batch = 1usize;
    let heads = 2usize;
    let sequence = 17usize;
    let head_dim = 64usize;
    let q = f32_tensor(
        &[batch as u64, heads as u64, sequence as u64, head_dim as u64],
        &pseudo_values(batch * heads * sequence * head_dim, 0x53_0001, -1.0, 1.0),
    );
    let k = f32_tensor(
        &[batch as u64, heads as u64, sequence as u64, head_dim as u64],
        &pseudo_values(batch * heads * sequence * head_dim, 0x53_0002, -1.0, 1.0),
    );
    // Causal mask: -FLT_MAX above the diagonal, 0 elsewhere.
    let mut mask_values = vec![0.0f32; heads * sequence * sequence];
    for head in 0..heads {
        for row in 0..sequence {
            for column in 0..sequence {
                if column > row {
                    mask_values[(head * sequence + row) * sequence + column] = -3.402_823_5e38;
                }
            }
        }
    }
    let mask = f32_tensor(
        &[batch as u64, heads as u64, sequence as u64, sequence as u64],
        &mask_values,
    );

    let k_transposed = TensorView::new(k.storage(), transpose_last_two(&k)).unwrap();
    let q_view = q.as_view().unwrap();
    let mut scores = zeros_f32(&[1, 2, 17, 17]);
    run_views(
        OpId::MatMul,
        &[q_view, k_transposed],
        &mut scores,
        &OpParams::default(),
    )
    .unwrap();
    let scale = f32_tensor(&[], &[(head_dim as f32).sqrt()]);
    let mut scaled = zeros_f32(&[1, 2, 17, 17]);
    run_into(
        OpId::Div,
        &[&scores, &scale],
        &mut scaled,
        &OpParams::default(),
    )
    .unwrap();
    let mut masked = zeros_f32(&[1, 2, 17, 17]);
    run_into(
        OpId::Add,
        &[&scaled, &mask],
        &mut masked,
        &OpParams::default(),
    )
    .unwrap();
    let mut probs = zeros_f32(&[1, 2, 17, 17]);
    run_into(
        OpId::Softmax,
        &[&masked],
        &mut probs,
        &OpParams::default().with_axes([-1]),
    )
    .unwrap();

    let probability_values = probs.as_f32_slice().unwrap();
    assert!(probability_values.iter().all(|value| value.is_finite()));
    for head in 0..heads {
        for row in 0..sequence {
            let base = (head * sequence + row) * sequence;
            let row_values = &probability_values[base..base + sequence];
            let total: f32 = row_values.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "head {head} row {row} sums to {total}"
            );
            for (column, &value) in row_values.iter().enumerate() {
                if column > row {
                    assert_eq!(
                        value, 0.0,
                        "masked entry head {head} row {row} col {column}"
                    );
                } else {
                    assert!(
                        value > 0.0,
                        "visible entry head {head} row {row} col {column}"
                    );
                }
            }
        }
    }
}

#[test]
fn real_shape_embedding_table_is_gathered_exactly() {
    // The pinned `wte` table: 50,257 rows of 128.
    let rows = 50_257usize;
    let hidden = 128usize;
    let table_values = pseudo_values(rows * hidden, 0x54_0001, -0.05, 0.05);
    let table = f32_tensor(&[rows as u64, hidden as u64], &table_values);
    // Duplicate indices, the last row, and a negative index (-1 == the last row).
    let indices = i64_tensor(&[4], &[0, rows as i64 - 1, 7, -1]);
    let mut out = zeros_f32(&[4, hidden as u64]);
    run_into(
        OpId::Gather,
        &[&table, &indices],
        &mut out,
        &OpParams::default(),
    )
    .unwrap();
    let gathered = out.as_f32_slice().unwrap();
    for (slot, &row) in [0usize, rows - 1, 7, rows - 1].iter().enumerate() {
        let expected = &table_values[row * hidden..(row + 1) * hidden];
        assert_eq!(
            &gathered[slot * hidden..(slot + 1) * hidden],
            expected,
            "row {row} was not gathered exactly"
        );
    }
}

#[test]
fn real_shape_batched_attention_scores() {
    // [1,2,17,64] @ [1,2,64,17] with the right operand as a transposed view.
    let q = f32_tensor(
        &[1, 2, 17, 64],
        &pseudo_values(2 * 17 * 64, 0x55_0001, -1.0, 1.0),
    );
    let k = f32_tensor(
        &[1, 2, 17, 64],
        &pseudo_values(2 * 17 * 64, 0x55_0002, -1.0, 1.0),
    );
    let k_transposed = TensorView::new(k.storage(), transpose_last_two(&k)).unwrap();
    let q_view = q.as_view().unwrap();
    let mut out = zeros_f32(&[1, 2, 17, 17]);
    run_views(
        OpId::MatMul,
        &[q_view, k_transposed],
        &mut out,
        &OpParams::default(),
    )
    .unwrap();
    // Reference for `Q @ K^T`: contract over the head dimension.
    let q_values = q.as_f32_slice().unwrap();
    let k_values = k.as_f32_slice().unwrap();
    let mut reference = Vec::with_capacity(2 * 17 * 17);
    for head in 0..2usize {
        for row in 0..17usize {
            for column in 0..17usize {
                let mut accumulator = 0.0f64;
                for step in 0..64usize {
                    accumulator += f64::from(q_values[(head * 17 + row) * 64 + step])
                        * f64::from(k_values[(head * 17 + column) * 64 + step]);
                }
                reference.push(accumulator);
            }
        }
    }
    assert_close(
        "batched [1,2,17,64] @ [1,2,64,17]",
        out.as_f32_slice().unwrap(),
        &reference,
        2.0e-5,
        1.0e-4,
    );
}

#[test]
fn batched_matmul_broadcasts_batch_axes() {
    // The attention projections and the KV-cache concat produce batch axes that
    // broadcast: `[2,1,3,4] @ [1,5,4,6] -> [2,5,3,6]`.
    let a_values = pseudo_values(2 * 3 * 4, 0x56_0001, -1.0, 1.0);
    let b_values = pseudo_values(5 * 4 * 6, 0x56_0002, -1.0, 1.0);
    let a = f32_tensor(&[2, 1, 3, 4], &a_values);
    let b = f32_tensor(&[1, 5, 4, 6], &b_values);
    let mut out = zeros_f32(&[2, 5, 3, 6]);
    run_into(OpId::MatMul, &[&a, &b], &mut out, &OpParams::default()).unwrap();

    let mut reference = Vec::with_capacity(2 * 5 * 3 * 6);
    for left in 0..2usize {
        for right in 0..5usize {
            let block = reference_matmul_2d(
                &a_values[left * 3 * 4..(left + 1) * 3 * 4],
                3,
                4,
                &b_values[right * 4 * 6..(right + 1) * 4 * 6],
                6,
            );
            reference.extend(block);
        }
    }
    assert_close(
        "broadcast batch [2,1,3,4] @ [1,5,4,6]",
        out.as_f32_slice().unwrap(),
        &reference,
        2.0e-5,
        1.0e-5,
    );
}

#[test]
fn awkward_tails_and_shapes() {
    // K values that are not multiples of any vector width, degenerate M/N and a
    // strided right-hand side.
    for (m, k, n) in [
        (1usize, 1usize, 1usize),
        (2, 2, 3),
        (5, 17, 3),
        (3, 65, 7),
        (1, 129, 2),
        (4, 128, 1),
    ] {
        let a_values = pseudo_values(m * k, 0x60_0001 + k as u64, -2.0, 2.0);
        let b_values = pseudo_values(k * n, 0x60_0002 + m as u64, -2.0, 2.0);
        let a = f32_tensor(&[m as u64, k as u64], &a_values);
        let b = f32_tensor(&[k as u64, n as u64], &b_values);
        let mut out = zeros_f32(&[m as u64, n as u64]);
        run_into(OpId::MatMul, &[&a, &b], &mut out, &OpParams::default()).unwrap();
        let reference = reference_matmul_2d(&a_values, m, k, &b_values, n);
        let atol = 2.0e-5 * (k as f64 / 64.0).max(1.0).sqrt();
        assert_close(
            &format!("tail M={m} K={k} N={n}"),
            out.as_f32_slice().unwrap(),
            &reference,
            atol,
            1.0e-5,
        );
    }

    // A transposed right-hand side with an odd K.
    let k = 65usize;
    let n = 5usize;
    let a_values = pseudo_values(3 * k, 0x61_0001, -1.0, 1.0);
    let stored = pseudo_values(n * k, 0x61_0002, -1.0, 1.0);
    let a = f32_tensor(&[3, k as u64], &a_values);
    let storage = f32_tensor(&[n as u64, k as u64], &stored);
    let transposed = TensorMeta::strided(
        Shape::from_dims([k as u64, n as u64]),
        Strides::from_elements(vec![1, k as i64]),
        0,
        DType::F32,
        cpu(),
    )
    .unwrap();
    let b_view = TensorView::new(storage.storage(), transposed).unwrap();
    let mut out = zeros_f32(&[3, n as u64]);
    run_views(
        OpId::MatMul,
        &[a.as_view().unwrap(), b_view],
        &mut out,
        &OpParams::default(),
    )
    .unwrap();
    let mut reference = Vec::with_capacity(3 * n);
    for row in 0..3 {
        for column in 0..n {
            let mut accumulator = 0.0f64;
            for step in 0..k {
                accumulator +=
                    f64::from(a_values[row * k + step]) * f64::from(stored[column * k + step]);
            }
            reference.push(accumulator);
        }
    }
    assert_close(
        "transposed rhs K=65",
        out.as_f32_slice().unwrap(),
        &reference,
        2.0e-5,
        1.0e-5,
    );
}

#[test]
fn awkward_softmax_widths() {
    // S = 1, 2, 3, 5, 17: no width is a power of two and the widest is 17.
    for width in [1usize, 2, 3, 5, 17] {
        let values = pseudo_values(2 * width, 0x62_0001 + width as u64, -4.0, 4.0);
        let input = f32_tensor(&[2, width as u64], &values);
        let mut out = zeros_f32(&[2, width as u64]);
        run_into(
            OpId::Softmax,
            &[&input],
            &mut out,
            &OpParams::default().with_axes([-1]),
        )
        .unwrap();
        let probabilities = out.as_f32_slice().unwrap();
        for row in 0..2 {
            let slice = &probabilities[row * width..(row + 1) * width];
            let total: f32 = slice.iter().sum();
            assert!((total - 1.0).abs() < 1e-6, "width {width} sums to {total}");
            // Reference: exp(x - max) / sum, in f64.
            let row_values = &values[row * width..(row + 1) * width];
            let max = row_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f64 = row_values
                .iter()
                .map(|&value| (f64::from(value) - f64::from(max)).exp())
                .sum();
            for (position, &value) in row_values.iter().enumerate() {
                let expected = (f64::from(value) - f64::from(max)).exp() / denominator;
                assert!(
                    (f64::from(slice[position]) - expected).abs() < 2e-6,
                    "width {width} row {row} column {position}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Every supported op executes
// ---------------------------------------------------------------------------

#[test]
fn structural_ops_match_a_naive_reference() {
    // Data movement has no arithmetic, so every one of these must be *exact*.
    // The reference here is a straightforward index computation over the same
    // source values, independent of the kernel's metadata path.
    let dims = [3u64, 4, 5];
    let values = pseudo_values(60, 0x70_0001, -1.0, 1.0);
    let tensor = f32_tensor(&dims, &values);

    // Transpose perm [2, 0, 1] -> [5, 3, 4].
    let mut transposed = zeros_f32(&[5, 3, 4]);
    run_into(
        OpId::Transpose,
        &[&tensor],
        &mut transposed,
        &OpParams::default().with_axes([2, 0, 1]),
    )
    .unwrap();
    let mut reference = Vec::with_capacity(60);
    for axis2 in 0..5usize {
        for axis0 in 0..3usize {
            for axis1 in 0..4usize {
                reference.push(values[(axis0 * 4 + axis1) * 5 + axis2]);
            }
        }
    }
    assert_eq!(transposed.as_f32_slice().unwrap(), reference.as_slice());

    // Slice axis 1 with a negative step: rows 3, 2, 1, 0.
    let starts = i64_tensor(&[1], &[3]);
    let ends = i64_tensor(&[1], &[-5]); // -5 + 4 clamps to -1: the open end
    let axes = i64_tensor(&[1], &[1]);
    let steps = i64_tensor(&[1], &[-1]);
    let mut reversed = zeros_f32(&[3, 4, 5]);
    run_into(
        OpId::Slice,
        &[&tensor, &starts, &ends, &axes, &steps],
        &mut reversed,
        &OpParams::default(),
    )
    .unwrap();
    let mut reference = Vec::with_capacity(60);
    for outer in 0..3usize {
        for row in (0..4usize).rev() {
            for inner in 0..5usize {
                reference.push(values[(outer * 4 + row) * 5 + inner]);
            }
        }
    }
    assert_eq!(reversed.as_f32_slice().unwrap(), reference.as_slice());

    // Slice a single interior row and compare with the source. When the axes
    // input is omitted, ONNX slices axes `0..n`; this call names axis 1.
    let starts = i64_tensor(&[1], &[2]);
    let ends = i64_tensor(&[1], &[3]);
    let axes = i64_tensor(&[1], &[1]);
    let mut row = zeros_f32(&[3, 1, 5]);
    run_into(
        OpId::Slice,
        &[&tensor, &starts, &ends, &axes],
        &mut row,
        &OpParams::default(),
    )
    .unwrap();
    for outer in 0..3usize {
        assert_eq!(
            &row.as_f32_slice().unwrap()[outer * 5..(outer + 1) * 5],
            &values[(outer * 4 + 2) * 5..(outer * 4 + 3) * 5]
        );
    }

    // Split into three along axis 2 and Concatenate back: the round trip must be
    // bit-identical to the source.
    let mut first = zeros_f32(&[3, 4, 2]);
    let mut second = zeros_f32(&[3, 4, 2]);
    let mut third = zeros_f32(&[3, 4, 1]);
    let sizes = i64_tensor(&[3], &[2, 2, 1]);
    run_multi(
        OpId::Split,
        &[&tensor, &sizes],
        &mut [&mut first, &mut second, &mut third],
        &OpParams::default().with_axes([2]),
    )
    .unwrap();
    let mut rejoined = zeros_f32(&[3, 4, 5]);
    run_into(
        OpId::Concat,
        &[&first, &second, &third],
        &mut rejoined,
        &OpParams::default().with_axes([2]),
    )
    .unwrap();
    assert_eq!(rejoined.as_f32_slice().unwrap(), values.as_slice());

    // Expand [3, 1, 5] to [3, 4, 5] by broadcast.
    let mut shrunk = Vec::with_capacity(15);
    for outer in 0..3usize {
        for inner in 0..5usize {
            shrunk.push(values[outer * 4 * 5 + inner]);
        }
    }
    let column = f32_tensor(&[3, 1, 5], &shrunk);
    let target = i64_tensor(&[3], &[3, 4, 5]);
    let mut expanded = zeros_f32(&[3, 4, 5]);
    run_into(
        OpId::Expand,
        &[&column, &target],
        &mut expanded,
        &OpParams::default(),
    )
    .unwrap();
    let mut reference_4d = Vec::with_capacity(60);
    for outer in 0..3usize {
        for _row in 0..4usize {
            for inner in 0..5usize {
                reference_4d.push(shrunk[outer * 5 + inner]);
            }
        }
    }
    assert_eq!(expanded.as_f32_slice().unwrap(), reference_4d.as_slice());

    // Reshape -> Squeeze -> Unsqueeze -> Reshape back to the source shape.
    let flat_shape = i64_tensor(&[2], &[60, 1]);
    let mut flat = zeros_f32(&[60, 1]);
    run_into(
        OpId::Reshape,
        &[&tensor, &flat_shape],
        &mut flat,
        &OpParams::default(),
    )
    .unwrap();
    let mut flat_vector = zeros_f32(&[60]);
    run_into(
        OpId::Squeeze,
        &[&flat],
        &mut flat_vector,
        &OpParams::default().with_axes([1]),
    )
    .unwrap();
    let mut expanded_2d = zeros_f32(&[60, 1]);
    run_into(
        OpId::Unsqueeze,
        &[&flat_vector],
        &mut expanded_2d,
        &OpParams::default().with_axes([1]),
    )
    .unwrap();
    let original_shape = i64_tensor(&[3], &[3, 4, 5]);
    let mut restored = zeros_f32(&[3, 4, 5]);
    run_into(
        OpId::Reshape,
        &[&expanded_2d, &original_shape],
        &mut restored,
        &OpParams::default(),
    )
    .unwrap();
    assert_eq!(restored.as_f32_slice().unwrap(), values.as_slice());
}

#[test]
fn every_supported_op_executes() {
    let suite = Suite::load("ops");
    let backend = backend();
    let mut executed = Vec::new();
    let params = OpParams::default();

    // Gather
    let data = f32_tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let indices = i64_tensor(&[2], &[2, 0]);
    let mut out = zeros_f32(&[2, 2]);
    run_into(OpId::Gather, &[&data, &indices], &mut out, &params).unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[5.0, 6.0, 1.0, 2.0]);
    executed.push(OpId::Gather);

    // MatMul
    let lhs = f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = f32_tensor(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let mut out = zeros_f32(&[2, 2]);
    run_into(OpId::MatMul, &[&lhs, &rhs], &mut out, &params).unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    executed.push(OpId::MatMul);

    // Binary elementwise
    let lhs = f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let rhs = f32_tensor(&[2], &[10.0, 20.0]);
    for (op, expected) in [
        (OpId::Add, [11.0f32, 22.0, 13.0, 24.0]),
        (OpId::Sub, [-9.0, -18.0, -7.0, -16.0]),
        (OpId::Mul, [10.0, 40.0, 30.0, 80.0]),
    ] {
        let mut out = zeros_f32(&[2, 2]);
        run_into(op, &[&lhs, &rhs], &mut out, &params).unwrap();
        assert_eq!(out.as_f32_slice().unwrap(), &expected, "{op}");
        executed.push(op);
    }
    let mut out = zeros_f32(&[2, 2]);
    run_into(OpId::Div, &[&lhs, &rhs], &mut out, &params).unwrap();
    assert_eq!(
        out.as_f32_slice().unwrap(),
        &[0.1, 0.1, 0.3, 0.2],
        "division"
    );
    executed.push(OpId::Div);
    let exponent = f32_tensor(&[2], &[2.0, 3.0]);
    let mut out = zeros_f32(&[2, 2]);
    run_into(OpId::Pow, &[&lhs, &exponent], &mut out, &params).unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 8.0, 9.0, 64.0]);
    executed.push(OpId::Pow);

    // Unary elementwise
    let input = f32_tensor(&[4], &[0.0, 0.25, 1.0, 4.0]);
    let cases: [(OpId, [f32; 4]); 8] = [
        (OpId::Neg, [-0.0, -0.25, -1.0, -4.0]),
        (OpId::Sqrt, [0.0, 0.5, 1.0, 2.0]),
        (OpId::Erf, [0.0, 0.276_326_4, 0.842_700_8, 1.0]),
        (OpId::Tanh, [0.0, 0.244_918_66, 0.761_594_2, 0.999_329_3]),
        (OpId::Gelu, [0.0, 0.149_676_58, 0.841_344_8, 3.999_873_4]),
        (OpId::Relu, [0.0, 0.25, 1.0, 4.0]),
        (OpId::Sin, [0.0, 0.247_403_96, 0.841_470_96, -0.756_802_5]),
        (OpId::Cos, [1.0, 0.968_912_4, 0.540_302_3, -0.653_643_6]),
    ];
    for (op, expected) in cases {
        let mut out = zeros_f32(&[4]);
        run_into(op, &[&input], &mut out, &params).unwrap();
        let values = out.as_f32_slice().unwrap();
        for (position, (&actual, &expected)) in values.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "{op} element {position}: {actual} vs {expected}"
            );
        }
        executed.push(op);
    }

    // Softmax / LayerNorm / reductions
    let input = f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 4.0, 4.0]);
    let mut out = zeros_f32(&[2, 3]);
    run_into(
        OpId::Softmax,
        &[&input],
        &mut out,
        &OpParams::default().with_axes([-1]),
    )
    .unwrap();
    executed.push(OpId::Softmax);

    let weight = f32_tensor(&[3], &[1.0, 1.0, 1.0]);
    let bias = f32_tensor(&[3], &[0.0, 0.0, 0.0]);
    let mut out = zeros_f32(&[2, 3]);
    run_into(
        OpId::LayerNorm,
        &[&input, &weight, &bias],
        &mut out,
        &OpParams::default().with_axes([-1]).with_epsilon(1e-5),
    )
    .unwrap();
    executed.push(OpId::LayerNorm);

    let mut out = zeros_f32(&[2, 1]);
    run_into(
        OpId::ReduceSum,
        &[&input],
        &mut out,
        &OpParams::default().with_axes([1]).with_keep_dims(true),
    )
    .unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[6.0, 12.0]);
    executed.push(OpId::ReduceSum);

    let mut out = zeros_f32(&[2, 1]);
    run_into(
        OpId::ReduceMean,
        &[&input],
        &mut out,
        &OpParams::default().with_axes([1]).with_keep_dims(true),
    )
    .unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[2.0, 4.0]);
    executed.push(OpId::ReduceMean);

    let mut out = zeros_i64(&[2]);
    run_into(
        OpId::ArgMax,
        &[&input],
        &mut out,
        &OpParams::default().with_axes([1]),
    )
    .unwrap();
    assert_eq!(
        out.as_i64_slice().unwrap(),
        &[2, 0],
        "ArgMax keeps the first maximum"
    );
    executed.push(OpId::ArgMax);

    // Data movement
    let matrix = f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut out = zeros_f32(&[3, 2]);
    run_into(
        OpId::Transpose,
        &[&matrix],
        &mut out,
        &OpParams::default().with_axes([1, 0]),
    )
    .unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    executed.push(OpId::Transpose);

    let shape_tensor = i64_tensor(&[2], &[3, 2]);
    let mut out = zeros_f32(&[3, 2]);
    run_into(OpId::Reshape, &[&matrix, &shape_tensor], &mut out, &params).unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    executed.push(OpId::Reshape);

    let mut out = zeros_f32(&[6, 1]);
    run_into(
        OpId::Flatten,
        &[&matrix],
        &mut out,
        &OpParams::default().with_axes([2]),
    )
    .unwrap();
    assert_eq!(out.shape().dims(), &[6, 1]);
    executed.push(OpId::Flatten);

    let squeezed_input = f32_tensor(&[1, 3, 1], &[1.0, 2.0, 3.0]);
    let mut out = zeros_f32(&[3]);
    run_into(
        OpId::Squeeze,
        &[&squeezed_input],
        &mut out,
        &OpParams::default().with_axes([0, 2]),
    )
    .unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0]);
    executed.push(OpId::Squeeze);

    let vector = f32_tensor(&[3], &[1.0, 2.0, 3.0]);
    let mut out = zeros_f32(&[1, 3, 1]);
    run_into(
        OpId::Unsqueeze,
        &[&vector],
        &mut out,
        &OpParams::default().with_axes([0, 2]),
    )
    .unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0]);
    executed.push(OpId::Unsqueeze);

    let top = f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let bottom = f32_tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
    let mut out = zeros_f32(&[4, 2]);
    run_into(
        OpId::Concat,
        &[&top, &bottom],
        &mut out,
        &OpParams::default().with_axes([0]),
    )
    .unwrap();
    assert_eq!(
        out.as_f32_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
    executed.push(OpId::Concat);

    let five = f32_tensor(&[5], &[0.0, 10.0, 20.0, 30.0, 40.0]);
    let starts = i64_tensor(&[1], &[1]);
    let ends = i64_tensor(&[1], &[4]);
    let mut out = zeros_f32(&[3]);
    run_into(OpId::Slice, &[&five, &starts, &ends], &mut out, &params).unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[10.0, 20.0, 30.0]);
    executed.push(OpId::Slice);

    let six = f32_tensor(&[6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut first = zeros_f32(&[2]);
    let mut second = zeros_f32(&[2]);
    let mut third = zeros_f32(&[2]);
    run_multi(
        OpId::Split,
        &[&six],
        &mut [&mut first, &mut second, &mut third],
        &OpParams::default().with_axes([0]),
    )
    .unwrap();
    assert_eq!(first.as_f32_slice().unwrap(), &[0.0, 1.0]);
    assert_eq!(third.as_f32_slice().unwrap(), &[4.0, 5.0]);
    executed.push(OpId::Split);

    let mut out = zeros_i64(&[2]);
    run_into(OpId::Shape, &[&matrix], &mut out, &params).unwrap();
    assert_eq!(out.as_i64_slice().unwrap(), &[2, 3]);
    executed.push(OpId::Shape);

    let shape_input = i64_tensor(&[2], &[2, 2]);
    let fill = f32_tensor(&[], &[2.5]);
    let mut out = zeros_f32(&[2, 2]);
    run_into(
        OpId::ConstantOfShape,
        &[&shape_input, &fill],
        &mut out,
        &params,
    )
    .unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[2.5; 4]);
    executed.push(OpId::ConstantOfShape);

    let expandable = f32_tensor(&[1, 3], &[1.0, 2.0, 3.0]);
    let target = i64_tensor(&[2], &[2, 3]);
    let mut out = zeros_f32(&[2, 3]);
    run_into(OpId::Expand, &[&expandable, &target], &mut out, &params).unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    executed.push(OpId::Expand);

    let start = i64_tensor(&[], &[0]);
    let limit = i64_tensor(&[], &[5]);
    let delta = i64_tensor(&[], &[2]);
    let mut out = zeros_i64(&[3]);
    run_into(OpId::Range, &[&start, &limit, &delta], &mut out, &params).unwrap();
    assert_eq!(out.as_i64_slice().unwrap(), &[0, 2, 4]);
    executed.push(OpId::Range);

    let floats = f32_tensor(&[3], &[1.9, -1.9, 3.0]);
    let mut out = zeros_i64(&[3]);
    run_into(OpId::Cast, &[&floats], &mut out, &params).unwrap();
    assert_eq!(out.as_i64_slice().unwrap(), &[1, -1, 3]);
    executed.push(OpId::Cast);

    // Comparisons and selection
    let left = f32_tensor(&[3], &[1.0, 2.0, 3.0]);
    let right = f32_tensor(&[3], &[1.0, 3.0, 2.0]);
    for op in [OpId::Equal, OpId::Less, OpId::Greater] {
        let mut out = zeros_mask(&[3]);
        run_into(op, &[&left, &right], &mut out, &params).unwrap();
        executed.push(op);
    }
    let condition = mask_tensor(&[3], &[true, false, true]);
    let yes = f32_tensor(&[3], &[10.0, 20.0, 30.0]);
    let no = f32_tensor(&[3], &[-1.0, -2.0, -3.0]);
    let mut out = zeros_f32(&[3]);
    run_into(OpId::Where, &[&condition, &yes, &no], &mut out, &params).unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[10.0, -2.0, 30.0]);
    executed.push(OpId::Where);

    let square = f32_tensor(
        &[3, 3],
        &(0..9).map(|value| value as f32).collect::<Vec<_>>(),
    );
    let mut out = zeros_f32(&[3, 3]);
    run_into(
        OpId::Trilu,
        &[&square],
        &mut out,
        &OpParams::default().with_axes([1]),
    )
    .unwrap();
    assert_eq!(
        out.as_f32_slice().unwrap(),
        &[0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 0.0, 8.0],
        "upper triangle keeps column >= row"
    );
    executed.push(OpId::Trilu);

    // ScatterND
    let target = f32_tensor(&[3, 2], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let scatter_indices = i64_tensor(&[1, 1], &[1]);
    let updates = f32_tensor(&[1, 2], &[9.0, 9.0]);
    let mut out = zeros_f32(&[3, 2]);
    run_into(
        OpId::ScatterND,
        &[&target, &scatter_indices, &updates],
        &mut out,
        &params,
    )
    .unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[0.0, 0.0, 9.0, 9.0, 0.0, 0.0]);
    executed.push(OpId::ScatterND);

    // Every op id reported as Full must appear above.
    for op in OpId::ALL {
        let support = backend.supports(op);
        if support.is_full() {
            assert!(
                executed.contains(&op),
                "op {op} reports Full but no test in this file executes it"
            );
        } else {
            assert!(
                !support.is_supported() && support.reason().is_some(),
                "op {op} must be Unsupported with a reason, found {support}"
            );
        }
    }
    assert_eq!(
        executed.len(),
        39,
        "the implemented op count changed; update support::IMPLEMENTED and this list together"
    );
    // The fixture suite is loaded here too, so a missing checkout fails this
    // binary loudly rather than silently skipping parity.
    assert_eq!(suite.names().len(), 47);
}

// ---------------------------------------------------------------------------
// 4. Error paths
// ---------------------------------------------------------------------------

#[test]
fn adversarial_wrong_dtype_is_rejected() {
    let integers = i64_tensor(&[2], &[1, 2]);
    let mut out = zeros_i64(&[2]);
    let error = run_into(
        OpId::Add,
        &[&integers, &integers],
        &mut out,
        &OpParams::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::DTypeMismatch {
            op: Some(OpId::Add),
            expected: DType::F32,
            actual: DType::I64
        }
    ));
    assert!(error.to_string().contains("op add"), "{error}");

    // `Cast` is the only conversion; `Where` refuses to mix dtypes.
    let condition = mask_tensor(&[2], &[true, false]);
    let floats = f32_tensor(&[2], &[1.0, 2.0]);
    let mut out = zeros_f32(&[2]);
    let error = run_into(
        OpId::Where,
        &[&condition, &floats, &integers],
        &mut out,
        &OpParams::default(),
    )
    .unwrap_err();
    assert!(matches!(error, Fn3Error::DTypeMismatch { .. }), "{error}");
}

#[test]
fn adversarial_rank_and_shape_errors_are_reported() {
    // ArgMax on a scalar has no axis.
    let scalar = f32_tensor(&[], &[1.0]);
    let mut out = zeros_i64(&[1]);
    let error = run_into(OpId::ArgMax, &[&scalar], &mut out, &OpParams::default()).unwrap_err();
    assert!(matches!(error, Fn3Error::RankMismatch { .. }), "{error}");

    // A rank-9 operand exceeds the backend's declared max_shape_rank.
    let wide = f32_tensor(&[1, 1, 1, 1, 1, 1, 1, 1, 1], &[1.0]);
    let mut out = zeros_f32(&[1, 1, 1, 1, 1, 1, 1, 1, 1]);
    let error = run_into(OpId::Neg, &[&wide], &mut out, &OpParams::default()).unwrap_err();
    assert!(matches!(error, Fn3Error::InvalidShape { .. }), "{error}");
    assert!(error.to_string().contains("rank 9"), "{error}");

    // A rank-1 MatMul operand is outside the documented contract and is named.
    let vector = f32_tensor(&[3], &[1.0, 2.0, 3.0]);
    let matrix = f32_tensor(&[3, 2], &[1.0; 6]);
    let mut out = zeros_f32(&[2]);
    let error = run_into(
        OpId::MatMul,
        &[&vector, &matrix],
        &mut out,
        &OpParams::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::UnsupportedOp {
            op: OpId::MatMul,
            backend: "cpu",
            ..
        }
    ));
    assert!(error.to_string().contains("FN3-07"), "{error}");

    // Output geometry is checked, not trusted.
    let a = f32_tensor(&[2, 3], &[1.0; 6]);
    let b = f32_tensor(&[3, 4], &[1.0; 12]);
    let mut out = zeros_f32(&[2, 3]);
    let error = run_into(OpId::MatMul, &[&a, &b], &mut out, &OpParams::default()).unwrap_err();
    assert!(matches!(error, Fn3Error::ShapeMismatch { .. }), "{error}");

    // Attributes are rejected, not ignored.
    let lhs = f32_tensor(&[2], &[1.0, 2.0]);
    let mut out = zeros_f32(&[2]);
    let error = run_into(
        OpId::Add,
        &[&lhs, &lhs],
        &mut out,
        &OpParams::default().with_axes([0]),
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("accepts no attributes"),
        "{error}"
    );

    // LayerNorm with a non-trailing axis is outside the documented contract.
    let x = f32_tensor(&[2, 3], &[1.0; 6]);
    let weight = f32_tensor(&[3], &[1.0; 3]);
    let bias = f32_tensor(&[3], &[0.0; 3]);
    let mut out = zeros_f32(&[2, 3]);
    let error = run_into(
        OpId::LayerNorm,
        &[&x, &weight, &bias],
        &mut out,
        &OpParams::default().with_axes([0]),
    )
    .unwrap_err();
    assert!(matches!(error, Fn3Error::UnsupportedOp { .. }), "{error}");
}

#[test]
fn adversarial_arity_mismatch_is_rejected() {
    let input = f32_tensor(&[2], &[1.0, 2.0]);
    let mut out = zeros_f32(&[2]);
    let error = run_into(OpId::Add, &[&input], &mut out, &OpParams::default()).unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::ArityMismatch {
            op: OpId::Add,
            expected_inputs: 2,
            actual_inputs: 1,
            expected_outputs: 1,
            actual_outputs: 1
        }
    ));

    let error = run_into(OpId::Neg, &[&input, &input], &mut out, &OpParams::default()).unwrap_err();
    assert!(matches!(error, Fn3Error::ArityMismatch { .. }), "{error}");

    // Split refuses an empty output list.
    let error = run_multi(
        OpId::Split,
        &[&input],
        &mut [],
        &OpParams::default().with_axes([0]),
    )
    .unwrap_err();
    assert!(matches!(error, Fn3Error::ArityMismatch { .. }), "{error}");
}

#[test]
fn adversarial_device_mismatch_is_rejected() {
    use fastnn::inference::storage::Storage;
    let foreign_storage =
        Storage::from_bytes(&[1.0f32.to_ne_bytes(), 2.0f32.to_ne_bytes()].concat()).unwrap();
    let meta = TensorMeta::contiguous(
        Shape::from_dims([2]),
        DType::F32,
        DeviceId::Vulkan(fastnn::inference::PhysicalDeviceId::new(0)),
    )
    .unwrap();
    let foreign = Tensor::from_storage(foreign_storage, meta).unwrap();
    let local = f32_tensor(&[2], &[1.0, 2.0]);
    let mut out = zeros_f32(&[2]);
    let error = run_into(
        OpId::Add,
        &[&local, &foreign],
        &mut out,
        &OpParams::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::DeviceMismatch {
            op: Some(OpId::Add),
            expected: DeviceId::Cpu,
            actual: DeviceId::Vulkan(_)
        }
    ));
    assert!(error.to_string().contains("vulkan(0)"), "{error}");
}

#[test]
fn adversarial_unsupported_op_is_reported() {
    let input = f32_tensor(&[2, 4], &[0.0; 8]);
    let mut out = zeros_f32(&[2, 4]);
    let error = run_into(
        OpId::MaskedAttention,
        &[&input],
        &mut out,
        &OpParams::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::UnsupportedOp {
            op: OpId::MaskedAttention,
            backend: "cpu",
            ..
        }
    ));
    let text = error.to_string();
    assert!(text.contains("#61"), "{text}");
    assert_eq!(out.as_f32_slice().unwrap(), &[0.0; 8]);
}

#[test]
fn zero_size_tensors_are_legal_and_execute() {
    // Add where one extent is zero.
    let lhs = zeros_f32(&[0, 3]);
    let rhs = f32_tensor(&[3], &[1.0, 2.0, 3.0]);
    let mut out = zeros_f32(&[0, 3]);
    run_into(OpId::Add, &[&lhs, &rhs], &mut out, &OpParams::default()).unwrap();
    assert_eq!(out.numel().unwrap(), 0);

    // MatMul with K = 0 is the empty product: +0.0 for every element.
    let a = zeros_f32(&[2, 0]);
    let b = zeros_f32(&[0, 3]);
    let mut out = zeros_f32(&[2, 3]);
    run_into(OpId::MatMul, &[&a, &b], &mut out, &OpParams::default()).unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[0.0; 6]);

    // Softmax with a zero-width axis writes nothing.
    let input = zeros_f32(&[2, 0]);
    let mut out = zeros_f32(&[2, 0]);
    run_into(
        OpId::Softmax,
        &[&input],
        &mut out,
        &OpParams::default().with_axes([-1]),
    )
    .unwrap();

    // Concat with a zero-length piece.
    let empty = zeros_f32(&[0, 2]);
    let filled = f32_tensor(&[1, 2], &[1.0, 2.0]);
    let mut out = zeros_f32(&[1, 2]);
    run_into(
        OpId::Concat,
        &[&empty, &filled],
        &mut out,
        &OpParams::default().with_axes([0]),
    )
    .unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 2.0]);

    // ReduceMean over an empty axis pins +0.0 rather than NaN.
    let input = zeros_f32(&[2, 0]);
    let mut out = zeros_f32(&[2, 1]);
    run_into(
        OpId::ReduceMean,
        &[&input],
        &mut out,
        &OpParams::default().with_axes([1]).with_keep_dims(true),
    )
    .unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[0.0, 0.0]);
}

#[test]
fn failed_execution_leaves_outputs_untouched() {
    // A sentinel value in every output slot: any write before the failure is
    // visible.
    let sentinel = 1234.5f32;
    let mut out = f32_tensor(&[2, 2], &[sentinel; 4]);
    let inputs_ok = [
        f32_tensor(&[2, 2], &[1.0; 4]),
        f32_tensor(&[2, 2], &[1.0; 4]),
    ];

    // 1. Wrong dtype.
    let integers = i64_tensor(&[2, 2], &[1; 4]);
    let error = run_into(
        OpId::Add,
        &[&integers, &integers],
        &mut out,
        &OpParams::default(),
    )
    .unwrap_err();
    assert_eq!(error.op(), Some(OpId::Add));
    assert_eq!(out.as_f32_slice().unwrap(), &[sentinel; 4]);

    // 2. Wrong output shape.
    let error = run_into(
        OpId::Add,
        &[&inputs_ok[0], &inputs_ok[1]],
        &mut out,
        &OpParams::default().with_axes([0]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("accepts no attributes"));
    assert_eq!(out.as_f32_slice().unwrap(), &[sentinel; 4]);

    // 3. Unsupported op.
    let error = run_into(
        OpId::MaskedAttention,
        &[&inputs_ok[0]],
        &mut out,
        &OpParams::default(),
    )
    .unwrap_err();
    assert!(matches!(error, Fn3Error::UnsupportedOp { .. }));
    assert_eq!(out.as_f32_slice().unwrap(), &[sentinel; 4]);

    // 4. Arity.
    let error = run_into(OpId::Add, &[&inputs_ok[0]], &mut out, &OpParams::default()).unwrap_err();
    assert!(matches!(error, Fn3Error::ArityMismatch { .. }));
    assert_eq!(out.as_f32_slice().unwrap(), &[sentinel; 4]);

    // 5. A cross-device operand.
    use fastnn::inference::storage::Storage;
    let foreign_storage =
        Storage::from_bytes(&[1.0f32.to_ne_bytes(), 2.0f32.to_ne_bytes()].concat()).unwrap();
    let foreign = Tensor::from_storage(
        foreign_storage,
        TensorMeta::contiguous(
            Shape::from_dims([2]),
            DType::F32,
            DeviceId::Vulkan(fastnn::inference::PhysicalDeviceId::new(0)),
        )
        .unwrap(),
    )
    .unwrap();
    let error = run_into(
        OpId::Add,
        &[&inputs_ok[0], &foreign],
        &mut out,
        &OpParams::default(),
    )
    .unwrap_err();
    assert!(matches!(error, Fn3Error::DeviceMismatch { .. }));
    assert_eq!(out.as_f32_slice().unwrap(), &[sentinel; 4]);

    // A passing call does write, which proves the sentinel is not simply never
    // touched by the harness.
    run_into(
        OpId::Add,
        &[&inputs_ok[0], &inputs_ok[1]],
        &mut out,
        &OpParams::default(),
    )
    .unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[2.0; 4]);
}

#[test]
fn output_aliasing_an_input_is_rejected() {
    // Plan level: an instruction may not read and write one slot.
    let plan = ExecutionPlan::new(vec![Instruction::new(OpId::Add, vec![0, 1], vec![0])], 2);
    let error = plan.unwrap_err();
    assert!(error.to_string().contains("reads and writes"), "{error}");

    // Backend level: a writable view cannot be created while any other handle
    // to the same allocation exists, so an output cannot alias an input.
    let mut tensor = f32_tensor(&[2, 2], &[1.0; 4]);
    // A second `Storage` handle is what a live view or a shared weight looks
    // like at runtime; the borrow checker already prevents aliasing through the
    // safe API, and this is the runtime rule behind it.
    let shared = tensor.storage().clone();
    let error = tensor.as_mut_view().unwrap_err();
    assert!(matches!(error, Fn3Error::StorageShared { .. }), "{error}");
    drop(shared);
    tensor.as_mut_view().unwrap();
}

// ---------------------------------------------------------------------------
// 5. Support table / fixture policy cross-checks
// ---------------------------------------------------------------------------

#[test]
fn support_table_matches_the_documented_contract() {
    let cpu = backend();
    assert_eq!(cpu.name(), "cpu");
    assert_eq!(cpu.device(), DeviceId::Cpu);
    // MaskedAttention is the reserved fused id; the twelve CNN ops are imported
    // but have no kernel yet. Every other id must be Full.
    let pending = [
        OpId::Conv,
        OpId::MaxPool,
        OpId::AveragePool,
        OpId::GlobalAveragePool,
        OpId::BatchNorm,
        OpId::Resize,
        OpId::Pad,
        OpId::Clip,
        OpId::Sigmoid,
        OpId::Maximum,
        OpId::Minimum,
        OpId::TopK,
    ];
    for op in OpId::ALL {
        let support = cpu.supports(op);
        assert!(
            !matches!(support, fastnn::inference::Support::Partial { .. }),
            "{op} must not hide a gap behind Partial"
        );
        if op == OpId::MaskedAttention {
            assert!(!support.is_supported());
            assert!(support.reason().unwrap().contains("#61"));
        } else if pending.contains(&op) {
            assert!(!support.is_supported(), "{op} reports {support}");
            assert!(support.reason().unwrap().contains("#102"), "{op}");
        } else {
            assert!(support.is_full(), "{op} reports {support}");
        }
    }

    // The pinned graph's op inventory: every distinct ONNX type must map to a
    // Full op id. `Gemm` maps to MatMul + Add and is not an id of its own.
    let pinned_graph_surface = [
        OpId::Gather, // Gather
        OpId::MatMul, // MatMul and Gemm
        OpId::Add,    // Add, including the Gemm bias
        OpId::Sub,
        OpId::Mul,
        OpId::Div,
        OpId::Pow,
        OpId::Sqrt,
        OpId::Tanh,
        OpId::Reshape,
        OpId::Squeeze,
        OpId::Unsqueeze,
        OpId::Concat,
        OpId::Split,
        OpId::Slice,
        OpId::Transpose,
        OpId::Shape,
        OpId::Range,
        OpId::ConstantOfShape,
        OpId::Cast,
        OpId::Equal,
        OpId::Less,
        OpId::Where,
        OpId::Expand,
        OpId::Softmax,
        OpId::ReduceMean,
    ];
    for op in pinned_graph_surface {
        assert!(
            cpu.supports(op).is_full(),
            "the pinned GPT-2 graph needs {op}, which reports {}",
            cpu.supports(op)
        );
    }
}

#[test]
fn fixture_policy_mean_abs_bound_predicate() {
    // The whole-model policy states `mean_abs <= 2.5e-6` on the *error*; the
    // operator suite states no bound, which is why the loader only applies one
    // when a tensor manifest provides it.
    let model = Suite::load("tiny-gpt2");
    let bound = model
        .mean_abs_policy()
        .expect("tiny-gpt2 must state a mean_abs policy");
    assert!((bound - 2.5e-6).abs() < 1e-12, "policy bound {bound:e}");

    // The reference agrees with itself: zero error.
    let reference = model.tensor("prefill.p1.logits");
    let values = reference.f32_values();
    let agreement = reference
        .check(Actual::F32(&values), &reference.shape())
        .unwrap();
    assert_eq!(agreement.mean_abs, 0.0);
    assert!(!fn3_fixtures::mean_abs_bound_violated(
        &values, &values, bound
    ));

    // A systematic 1e-5 bias exceeds the 2.5e-6 bound even though each element
    // is a plausible logit.
    let biased: Vec<f32> = values.iter().map(|value| value + 1e-5).collect();
    assert!(fn3_fixtures::mean_abs_bound_violated(
        &biased, &values, bound
    ));
    println!(
        "[policy] mean_abs error for a +1e-5 bias = {:e} > {bound:e}",
        fn3_fixtures::mean_abs_error(&biased, &values)
    );
}

#[test]
fn fixture_manifests_are_self_consistent() {
    for name in ["ops", "tiny-gpt2"] {
        let suite = Suite::load(name);
        let bytes = suite.verify_stored_payloads();
        println!(
            "[fixture] suite {name}: {} tensors, {bytes} stored bytes",
            suite.names().len()
        );
        assert!(!suite.names().is_empty());
        // Every stored tensor's payload digest is verified by `bytes()`.
        for tensor in suite.tensors() {
            let shape = tensor.shape();
            let elements = tensor.element_count();
            let expected = shape.iter().product::<u64>() as usize;
            assert_eq!(elements, expected, "{} element count", tensor.name());
            if tensor.stored() {
                let payload = tensor.bytes().len();
                assert_eq!(
                    payload,
                    expected * tensor.dtype().size_bytes(),
                    "{} payload length",
                    tensor.name()
                );
            }
        }
    }
}
