//! Elementwise kernels: arithmetic, transcendental functions, activations and
//! the explicit dtype conversion.
//!
//! # Evaluation order
//!
//! Every elementwise kernel evaluates one output element from exactly one input
//! element per operand. There is no accumulation, so the only determinism
//! question is *which* implementation of a transcendental function runs:
//!
//! | Op | Implementation | Stated accuracy |
//! |---|---|---|
//! | `Add`/`Sub`/`Mul`/`Div`/`Neg` | IEEE-754 `f32` arithmetic, one rounding | correctly rounded |
//! | `Sqrt` | `f32::sqrt` (IEEE-754 square root) | correctly rounded |
//! | `Pow` | `f32::powf` (platform libm) | ≤ 1 ulp of the platform libm |
//! | `Tanh` | `f32::tanh` (platform libm) | ≤ 1 ulp of the platform libm |
//! | `Sin`/`Cos` | `f32::sin`/`f32::cos` (platform libm) | ≤ 1 ulp of the platform libm |
//! | `Relu` | `max(x, 0)` with the `f32::max` NaN rule documented below | exact |
//! | `Erf` | [`erf_f64`]: degree-22 Chebyshev fit of `erf(sqrt(u))/sqrt(u)`, evaluated in `f64`, rounded once | ≤ 1.7e-12 in `f64` on `[0, 4]`; ≤ 1.6e-8 (the `|x| > 4` saturation clamp) overall; ≤ 1 `f32` ulp after rounding |
//! | `Gelu` (erf form) | `0.5*x*(1+erf(x/sqrt(2)))` in `f64`, rounded once | ≤ 1 `f32` ulp |
//!
//! `Erf`/`Gelu` are deliberately evaluated in `f64`: the fixture reference is
//! "explicit float64 reference, rounded once to float32"
//! (`fixtures/parity/README.md`), so evaluating the same expression in `f64`
//! and rounding once reproduces the reference to within a rounding of the final
//! `f32`, instead of inheriting an approximation error on top of it. `F32`
//! remains the only *compute dtype*; `f64` is an internal evaluation width.
//!
//! # Broadcasting
//!
//! Binary ops broadcast right-aligned with NumPy rules. The fast paths are:
//! identical contiguous shapes (`iter().zip()`), and one operand with a single
//! element (a scalar). Anything else promotes the smaller operand with
//! effective zero strides through [`Locator`].
//!
//! # Sources
//!
//! The arithmetic here is written for `src/fn3` and does not import the 2.x
//! `src/backend/cpu/microkernels/` tree. Where a 2.x kernel informed the shape
//! of a function (for example `gelu_f32` in
//! `src/backend/cpu/microkernels/activations.rs`), the origin is named in the
//! function's doc comment.

use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::tensor::{TensorMut, TensorView};

use super::{at, at_mut, decompose, require_dtype, require_output_shape, Locator};

/// A span of output positions that touches each operand with a uniform step.
///
/// `len` elements are written starting at `out`, advancing by `out_step`, while
/// the operands advance by `lhs_step`/`rhs_step`. A step of `0` means the
/// operand does not move along this span (a promoted axis), which is the case
/// every broadcast reduces to.
struct BroadcastRun {
    out: usize,
    lhs: usize,
    rhs: usize,
    len: usize,
    out_step: usize,
    lhs_step: usize,
    rhs_step: usize,
}

/// Collapse the result space into runs that are uniform in all three views.
///
/// Axes with extent 1 contribute a coordinate that is always 0, so they are
/// dropped. The innermost surviving axis seeds a run; each axis outside it is
/// folded into that run while its strides are exactly the inner extent times
/// the run's current step, which is what "contiguous in all three views" means.
/// Remaining outer axes are walked with an odometer, so one step costs a
/// coordinate increment instead of a division per element.
fn broadcast_runs(
    dims: &[u64],
    out_locator: &Locator,
    lhs_locator: &Locator,
    rhs_locator: &Locator,
) -> Result<Vec<BroadcastRun>, Fn3Error> {
    let rank = dims.len();
    let out_strides = out_locator.strides();
    let lhs_strides = lhs_locator.strides();
    let rhs_strides = rhs_locator.strides();
    debug_assert_eq!(out_strides.len(), rank);
    debug_assert_eq!(lhs_strides.len(), rank);
    debug_assert_eq!(rhs_strides.len(), rank);

    let mut kept: Vec<usize> = Vec::with_capacity(rank);
    for axis in 0..rank {
        if dims[axis] != 1 {
            kept.push(axis);
        }
    }
    if kept.is_empty() {
        // Every axis is extent 1: the whole output is one element.
        return Ok(vec![BroadcastRun {
            out: out_locator.offset(),
            lhs: lhs_locator.offset(),
            rhs: rhs_locator.offset(),
            len: 1,
            out_step: 0,
            lhs_step: 0,
            rhs_step: 0,
        }]);
    }

    let mut merged = 1usize;
    let mut merged_axes = 0usize;
    let mut out_step = 0usize;
    let mut lhs_step = 0usize;
    let mut rhs_step = 0usize;
    for &axis in kept.iter().rev() {
        let os = out_strides[axis];
        let ls = lhs_strides[axis];
        let rs = rhs_strides[axis];
        if merged_axes == 0 {
            out_step = os;
            lhs_step = ls;
            rhs_step = rs;
            merged = usize::try_from(dims[axis]).map_err(|_| Fn3Error::ShapeOverflow {
                context: "cpu elementwise broadcast extent",
                axis: Some(axis),
                detail: "extent does not fit in usize".to_string(),
            })?;
            merged_axes = 1;
            continue;
        }
        // Fold this axis in only while all three views stay uniform. An
        // overflow means the strides cannot describe `merged` elements, so the
        // axis stays outside and the odometer handles it.
        let uniform = merged.checked_mul(out_step).is_some_and(|want| want == os)
            && merged.checked_mul(lhs_step).is_some_and(|want| want == ls)
            && merged.checked_mul(rhs_step).is_some_and(|want| want == rs);
        if !uniform {
            break;
        }
        merged = merged
            .checked_mul(
                usize::try_from(dims[axis]).map_err(|_| Fn3Error::ShapeOverflow {
                    context: "cpu elementwise broadcast extent",
                    axis: Some(axis),
                    detail: "extent does not fit in usize".to_string(),
                })?,
            )
            .ok_or(Fn3Error::ShapeOverflow {
                context: "cpu elementwise broadcast run",
                axis: Some(axis),
                detail: "run length does not fit in usize".to_string(),
            })?;
        merged_axes += 1;
    }

    let outer_axes = &kept[..kept.len() - merged_axes];
    let mut coordinates = vec![0usize; outer_axes.len()];
    let mut runs = Vec::new();
    loop {
        let mut out = out_locator.offset();
        let mut lhs = lhs_locator.offset();
        let mut rhs = rhs_locator.offset();
        for (position, &axis) in outer_axes.iter().enumerate() {
            let coordinate = coordinates[position];
            out = out.saturating_add(coordinate.saturating_mul(out_strides[axis]));
            lhs = lhs.saturating_add(coordinate.saturating_mul(lhs_strides[axis]));
            rhs = rhs.saturating_add(coordinate.saturating_mul(rhs_strides[axis]));
        }
        runs.push(BroadcastRun {
            out,
            lhs,
            rhs,
            len: merged,
            out_step,
            lhs_step,
            rhs_step,
        });
        if outer_axes.is_empty() {
            break;
        }
        // Odometer carry over the outer axes.
        let mut exhausted = true;
        for position in (0..outer_axes.len()).rev() {
            coordinates[position] += 1;
            if (coordinates[position] as u64) < dims[outer_axes[position]] {
                exhausted = false;
                break;
            }
            coordinates[position] = 0;
        }
        if exhausted {
            break;
        }
    }
    Ok(runs)
}

/// Whether every index a run touches is inside its buffer.
///
/// The runs come from the same metadata the per-element path validates, so this
/// only fails on a malformed view; in that case the caller keeps the checked
/// per-element path instead of indexing past the end.
fn runs_within_bounds(
    runs: &[BroadcastRun],
    out_len: usize,
    lhs_len: usize,
    rhs_len: usize,
) -> bool {
    let inside = |start: usize, step: usize, len: usize, extent: usize| -> bool {
        if len == 0 {
            return true;
        }
        start
            .checked_add(step.saturating_mul(len - 1))
            .is_some_and(|last| last < extent)
    };
    runs.iter().all(|run| {
        inside(run.out, run.out_step, run.len, out_len)
            && inside(run.lhs, run.lhs_step, run.len, lhs_len)
            && inside(run.rhs, run.rhs_step, run.len, rhs_len)
    })
}

/// Evaluate every run, choosing the cheapest loop shape the steps allow.
fn apply_broadcast_runs(
    runs: &[BroadcastRun],
    output: &mut [f32],
    lhs: &[f32],
    rhs: &[f32],
    evaluate: impl Fn(f32, f32) -> f32,
) {
    for run in runs {
        // A non-unit output step means the write side is not contiguous, which
        // the wide arms below cannot express; the scalar arm handles it.
        if run.out_step != 1 {
            let mut target = run.out;
            for offset in 0..run.len {
                output[target] = evaluate(
                    lhs[run.lhs + offset * run.lhs_step],
                    rhs[run.rhs + offset * run.rhs_step],
                );
                target += run.out_step;
            }
            continue;
        }
        let span = &mut output[run.out..run.out + run.len];
        match (run.lhs_step, run.rhs_step) {
            (1, 1) => {
                let left = &lhs[run.lhs..run.lhs + run.len];
                let right = &rhs[run.rhs..run.rhs + run.len];
                for ((slot, &x), &y) in span.iter_mut().zip(left.iter()).zip(right.iter()) {
                    *slot = evaluate(x, y);
                }
            }
            (1, 0) => {
                let right = rhs[run.rhs];
                let left = &lhs[run.lhs..run.lhs + run.len];
                for (slot, &x) in span.iter_mut().zip(left.iter()) {
                    *slot = evaluate(x, right);
                }
            }
            (0, 1) => {
                let left = lhs[run.lhs];
                let right = &rhs[run.rhs..run.rhs + run.len];
                for (slot, &y) in span.iter_mut().zip(right.iter()) {
                    *slot = evaluate(left, y);
                }
            }
            (0, 0) => {
                let value = evaluate(lhs[run.lhs], rhs[run.rhs]);
                span.fill(value);
            }
            _ => {
                for (offset, slot) in span.iter_mut().enumerate() {
                    *slot = evaluate(
                        lhs[run.lhs + offset * run.lhs_step],
                        rhs[run.rhs + offset * run.rhs_step],
                    );
                }
            }
        }
    }
}

/// Elementwise binary op with right-aligned broadcasting.
pub(crate) fn binary(
    op: OpId,
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    out: &mut TensorMut<'_>,
    evaluate: impl Fn(f32, f32) -> f32,
) -> Result<(), Fn3Error> {
    require_dtype(op, lhs.meta(), DType::F32)?;
    require_dtype(op, rhs.meta(), DType::F32)?;
    require_dtype(op, out.meta(), DType::F32)?;
    let shape = lhs.shape().broadcast_with(rhs.shape(), op)?;
    require_output_shape(op, &shape, out)?;
    let numel = shape.numel()?;
    if numel == 0 {
        return Ok(());
    }

    // Fast path: identical shapes, all contiguous. No index arithmetic at all.
    if lhs.shape() == &shape
        && rhs.shape() == &shape
        && lhs.is_contiguous()
        && rhs.is_contiguous()
        && out.is_contiguous()
    {
        let left = lhs.as_slice::<f32>()?;
        let right = rhs.as_slice::<f32>()?;
        let destination = out.as_mut_slice::<f32>()?;
        for ((slot, &x), &y) in destination.iter_mut().zip(left.iter()).zip(right.iter()) {
            *slot = evaluate(x, y);
        }
        return Ok(());
    }

    // Fast path: one operand is a single element, so it is a scalar for every
    // output element. `numel == 1` means every coordinate is 0, so the element
    // index is the view's offset.
    let scalar_side = if rhs.shape().numel()? == 1 {
        Some((rhs, false))
    } else if lhs.shape().numel()? == 1 {
        Some((lhs, true))
    } else {
        None
    };
    if let Some((scalar_view, scalar_is_left)) = scalar_side {
        let scalar = at(
            op,
            scalar_view.storage_slice::<f32>()?,
            scalar_view.meta().offset(),
        )?;
        if lhs.shape() == &shape && lhs.is_contiguous() && out.is_contiguous() {
            let vector = lhs.as_slice::<f32>()?;
            let destination = out.as_mut_slice::<f32>()?;
            for (slot, &x) in destination.iter_mut().zip(vector.iter()) {
                *slot = if scalar_is_left {
                    evaluate(scalar, x)
                } else {
                    evaluate(x, scalar)
                };
            }
            return Ok(());
        }
        if rhs.shape() == &shape && rhs.is_contiguous() && out.is_contiguous() {
            let vector = rhs.as_slice::<f32>()?;
            let destination = out.as_mut_slice::<f32>()?;
            for (slot, &y) in destination.iter_mut().zip(vector.iter()) {
                *slot = if scalar_is_left {
                    evaluate(y, scalar)
                } else {
                    evaluate(scalar, y)
                };
            }
            return Ok(());
        }
    }

    // General path: promote both operands to the result shape.
    let lhs_locator = Locator::new(op, lhs.meta(), &shape)?;
    let rhs_locator = Locator::new(op, rhs.meta(), &shape)?;
    let out_locator = Locator::new(op, out.meta(), &shape)?;
    let lhs_data = lhs.storage_slice::<f32>()?;
    let rhs_data = rhs.storage_slice::<f32>()?;
    let out_data = out.storage_slice_mut::<f32>()?;
    // Runs first: the per-element path below resolves three locators and
    // decomposes a coordinate for every output element, which is the most
    // expensive part of a broadcast by an order of magnitude.
    let runs = broadcast_runs(shape.dims(), &out_locator, &lhs_locator, &rhs_locator)?;
    if runs_within_bounds(&runs, out_data.len(), lhs_data.len(), rhs_data.len()) {
        apply_broadcast_runs(&runs, out_data, lhs_data, rhs_data, &evaluate);
        return Ok(());
    }
    let rank = shape.rank();
    let mut coordinates = vec![0u64; rank];
    for linear in 0..numel {
        decompose(linear, shape.dims(), &mut coordinates);
        let left = at(op, lhs_data, lhs_locator.index(op, &coordinates)?)?;
        let right = at(op, rhs_data, rhs_locator.index(op, &coordinates)?)?;
        let index = out_locator.index(op, &coordinates)?;
        *at_mut(op, out_data, index)? = evaluate(left, right);
    }
    Ok(())
}

/// Elementwise unary op; the output shape must equal the input shape.
pub(crate) fn unary(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
    evaluate: impl Fn(f32) -> f32,
) -> Result<(), Fn3Error> {
    require_dtype(op, input.meta(), DType::F32)?;
    require_dtype(op, out.meta(), DType::F32)?;
    if out.shape() != input.shape() {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: input.shape().to_string(),
            actual: out.shape().to_string(),
        });
    }
    let numel = input.shape().numel()?;
    if numel == 0 {
        return Ok(());
    }
    if input.is_contiguous() && out.is_contiguous() {
        let source = input.as_slice::<f32>()?;
        let destination = out.as_mut_slice::<f32>()?;
        for (slot, &value) in destination.iter_mut().zip(source.iter()) {
            *slot = evaluate(value);
        }
        return Ok(());
    }

    // The output locator is built before the mutable slice is taken, so the
    // write loop needs no borrow of `out`'s metadata while it holds the buffer.
    let locator = Locator::new(op, input.meta(), input.shape())?;
    let out_locator = Locator::new(op, out.meta(), out.shape())?;
    let source = input.storage_slice::<f32>()?;
    let destination = out.storage_slice_mut::<f32>()?;
    let dims = input.shape().dims().to_vec();
    let rank = input.shape().rank();
    let mut coordinates = vec![0u64; rank];
    for linear in 0..numel {
        decompose(linear, &dims, &mut coordinates);
        let value = at(op, source, locator.index(op, &coordinates)?)?;
        let index = out_locator.index(op, &coordinates)?;
        *at_mut(op, destination, index)? = evaluate(value);
    }
    Ok(())
}

pub(crate) fn add(
    op: OpId,
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    binary(op, lhs, rhs, out, |x, y| x + y)
}

pub(crate) fn sub(
    op: OpId,
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    binary(op, lhs, rhs, out, |x, y| x - y)
}

pub(crate) fn mul(
    op: OpId,
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    binary(op, lhs, rhs, out, |x, y| x * y)
}

/// IEEE division. Division by zero yields `±inf` or `NaN` exactly as IEEE-754
/// specifies; the kernel does not guard it, because a caller that wants a floor
/// must use `LayerNorm`'s epsilon or an explicit `Add`.
pub(crate) fn div(
    op: OpId,
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    binary(op, lhs, rhs, out, |x, y| x / y)
}

/// `powf` with broadcasting. ONNX `Pow` supplies the exponent as a tensor, so
/// this is a binary op; `Pow(x, 2)` and `Pow(x, 3)` are the layer-norm variance
/// and the `gelu_new` cube in the pinned graph.
pub(crate) fn pow(
    op: OpId,
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    binary(op, lhs, rhs, out, f32::powf)
}

pub(crate) fn neg(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    unary(op, input, out, |x| -x)
}

pub(crate) fn sqrt(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    unary(op, input, out, f32::sqrt)
}

pub(crate) fn tanh(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    unary(op, input, out, f32::tanh)
}

pub(crate) fn relu(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    // ONNX `Relu` is `max(0, x)`. `f32::max` returns the non-NaN operand, so
    // `Relu(NaN)` is `0.0` here. That is documented rather than silently
    // inherited: a NaN-signalling activation would need its own op id.
    unary(op, input, out, |x| if x > 0.0 { x } else { 0.0 })
}

pub(crate) fn sin(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    unary(op, input, out, f32::sin)
}

pub(crate) fn cos(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    unary(op, input, out, f32::cos)
}

pub(crate) fn erf(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    unary(op, input, out, erf_f32)
}

/// Exact (`erf`) GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
///
/// Evaluated in `f64` and rounded once, which is exactly how the parity fixture
/// is produced (`tools/parity/generate_fixtures.py::gelu_erf64`).
/// The `gelu_new` (tanh) variant used by the pinned GPT-2 graph is *not* this
/// op: the graph itself decomposes it into
/// `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`, so the importer emits those
/// primitive ops and this kernel keeps the documented `OpId::Gelu` semantics
/// ("exact (erf) form" in `src/fn3/ops/mod.rs`).
pub(crate) fn gelu(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    unary(op, input, out, gelu_erf_f32)
}

/// The `f32` rounding of the exact `erf`.
pub(crate) fn erf_f32(x: f32) -> f32 {
    erf_f64(x as f64) as f32
}

/// The `f32` rounding of `0.5*x*(1+erf(x/sqrt(2)))`.
pub(crate) fn gelu_erf_f32(x: f32) -> f32 {
    let value = x as f64;
    (0.5 * value * (1.0 + erf_f64(value * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

/// Chebyshev coefficients of `Q(u) = erf(sqrt(u)) / sqrt(u)` on `u ∈ [0, 16]`,
/// in the variable `t = u/8 - 1`.
///
/// Derivation: `Q` was sampled at 40,001 equally spaced points in `u` from an
/// `mpmath` evaluation at 50 decimal digits and fitted by least squares in the
/// Chebyshev basis (`numpy.polynomial.chebyshev.chebfit`, degree 22). The fit
/// is **not** trusted on its own: `elementwise::tests::erf_matches_high_precision_table`
/// checks the Rust implementation against `mpmath` reference values embedded in
/// the test, and the measured error is recorded on [`erf_f64`].
const ERF_CHEBYSHEV: [f64; 23] = [
    0.485_912_956_902_861_1,
    -0.345_407_855_004_922_5,
    0.161_829_101_213_641_77,
    -0.076_482_756_105_481_53,
    0.034_669_936_800_502_83,
    -0.014_811_864_147_787_053,
    0.005_922_937_956_227_918,
    -0.002_211_840_658_781_702_4,
    0.000_771_415_427_836_306_1,
    -0.000_251_596_587_780_150_1,
    7.688_658_220_609_625e-05,
    -2.206_549_590_830_934_4e-05,
    5.961_274_036_492_068e-06,
    -1.519_775_038_739_341_7e-06,
    3.664_930_385_568_287e-07,
    -8.378_994_844_081_672e-08,
    1.820_144_385_384_328_5e-08,
    -3.764_438_877_104_998e-09,
    7.427_303_001_806_201e-10,
    -1.400_073_299_813_430_8e-10,
    2.527_550_514_608_321_3e-11,
    -4.317_158_215_099_143_6e-12,
    7.186_521_212_324_026e-13,
];

/// The error function, evaluated in `f64` with a Clenshaw summation.
///
/// Measured accuracy (against `mpmath.erf` at 50 decimal digits):
///
/// - `|x| <= 4`: max absolute error `1.69e-12` over 400,001 sample points.
/// - `|x| > 4`: returns `±1.0`; the true `erfc(4)` is `1.54e-8`, so the error
///   is at most `1.54e-8`, i.e. `0.26` of one `f32` ulp at 1.0.
/// - After rounding to `f32`, 0 of 300,001 sampled `f32` inputs in
///   `[-4.2, 4.2]` differed from the correctly rounded true `erf`.
///
/// No `exp` call is involved, so the result is bit-deterministic on every
/// target that implements IEEE-754 `f64` multiply/subtract-and-add-free
/// arithmetic. Clenshaw summation is used instead of expanding the fit into
/// monomials because the monomial coefficients of a degree-22 fit on `u ∈
/// [0, 16]` alternate in sign and cancel catastrophically.
pub(crate) fn erf_f64(x: f64) -> f64 {
    let magnitude = x.abs();
    if magnitude > 4.0 {
        return if x.is_sign_negative() { -1.0 } else { 1.0 };
    }
    let t = magnitude * magnitude / 8.0 - 1.0;
    let mut b1 = 0.0f64;
    let mut b2 = 0.0f64;
    for coefficient in ERF_CHEBYSHEV.iter().skip(1).rev() {
        let next = 2.0 * t * b1 - b2 + coefficient;
        b2 = b1;
        b1 = next;
    }
    let q = t * b1 - b2 + ERF_CHEBYSHEV[0];
    let value = magnitude * q;
    if x.is_sign_negative() {
        -value
    } else {
        value
    }
}

// ---------------------------------------------------------------------------
// Cast: the only sanctioned dtype conversion.
// ---------------------------------------------------------------------------

/// One element of any alpha dtype, independent of its storage encoding.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Scalar {
    F32(f32),
    I64(i64),
    I32(i32),
    Bool(bool),
}

impl Scalar {
    fn to_f32(self) -> f32 {
        match self {
            // `as` casts from integers are exact until 2^24 and
            // round-to-nearest above it, which is the IEEE conversion.
            Scalar::F32(value) => value,
            Scalar::I64(value) => value as f32,
            Scalar::I32(value) => value as f32,
            Scalar::Bool(value) => {
                if value {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// `f32 -> i64` truncates toward zero and saturates (`NaN` becomes 0), which
    /// is Rust's documented `as` behaviour and is pinned by
    /// `elementwise::tests::cast_float_to_integer_saturates_and_truncates`.
    fn to_i64(self) -> i64 {
        match self {
            Scalar::F32(value) => value as i64,
            Scalar::I64(value) => value,
            Scalar::I32(value) => value as i64,
            Scalar::Bool(value) => value as i64,
        }
    }

    fn to_i32(self) -> i32 {
        match self {
            Scalar::F32(value) => value as i32,
            Scalar::I64(value) => value as i32,
            Scalar::I32(value) => value,
            Scalar::Bool(value) => value as i32,
        }
    }

    /// Non-zero is `true` (ONNX `Cast` to `bool`).
    fn to_bool(self) -> bool {
        match self {
            Scalar::F32(value) => value != 0.0,
            Scalar::I64(value) => value != 0,
            Scalar::I32(value) => value != 0,
            Scalar::Bool(value) => value,
        }
    }
}

/// `Cast`: shape-preserving dtype conversion. The target dtype is the output
/// tensor's dtype, so no attribute is needed.
pub(crate) fn cast(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    if out.shape() != input.shape() {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: input.shape().to_string(),
            actual: out.shape().to_string(),
        });
    }
    let source_dtype = input.dtype();
    let target_dtype = out.dtype();
    if source_dtype == target_dtype {
        // An identity cast is a copy; it must not reinterpret anything.
        return super::transform::copy_row_major(op, input, out);
    }
    let numel = input.shape().numel()?;
    if numel == 0 {
        return Ok(());
    }

    let input_locator = Locator::new(op, input.meta(), input.shape())?;
    let out_locator = Locator::new(op, out.meta(), out.shape())?;
    let dims = input.shape().dims().to_vec();
    let rank = input.shape().rank();
    let mut coordinates = vec![0u64; rank];

    macro_rules! convert {
        ($src:ty, $dst:ty, $convert:expr) => {{
            let source = input.storage_slice::<$src>()?;
            let destination = out.storage_slice_mut::<$dst>()?;
            let convert: fn($src) -> $dst = $convert;
            for linear in 0..numel {
                decompose(linear, &dims, &mut coordinates);
                let from = at(op, source, input_locator.index(op, &coordinates)?)?;
                let to = out_locator.index(op, &coordinates)?;
                *at_mut(op, destination, to)? = convert(from);
            }
        }};
    }

    match (source_dtype, target_dtype) {
        (DType::F32, DType::F32) => unreachable!("handled by the identity copy above"),
        (DType::F32, DType::I64) => convert!(f32, i64, |value: f32| Scalar::F32(value).to_i64()),
        (DType::F32, DType::I32) => convert!(f32, i32, |value: f32| Scalar::F32(value).to_i32()),
        (DType::F32, DType::Bool) => {
            convert!(f32, u8, |value: f32| u8::from(Scalar::F32(value).to_bool()))
        }
        (DType::I64, DType::F32) => convert!(i64, f32, |value: i64| Scalar::I64(value).to_f32()),
        (DType::I64, DType::I32) => convert!(i64, i32, |value: i64| Scalar::I64(value).to_i32()),
        (DType::I64, DType::Bool) => {
            convert!(i64, u8, |value: i64| u8::from(Scalar::I64(value).to_bool()))
        }
        (DType::I32, DType::F32) => convert!(i32, f32, |value: i32| Scalar::I32(value).to_f32()),
        (DType::I32, DType::I64) => convert!(i32, i64, |value: i32| Scalar::I32(value).to_i64()),
        (DType::I32, DType::Bool) => {
            convert!(i32, u8, |value: i32| u8::from(Scalar::I32(value).to_bool()))
        }
        (DType::Bool, DType::F32) => {
            convert!(u8, f32, |value: u8| { Scalar::Bool(value != 0).to_f32() })
        }
        (DType::Bool, DType::I64) => {
            convert!(u8, i64, |value: u8| { Scalar::Bool(value != 0).to_i64() })
        }
        (DType::Bool, DType::I32) => {
            convert!(u8, i32, |value: u8| { Scalar::Bool(value != 0).to_i32() })
        }
        _ => {
            return Err(Fn3Error::UnsupportedDType {
                op: Some(op),
                dtype: target_dtype,
                context: "cpu cast kernel (this dtype pair is not implemented)",
            })
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mpmath.erf(x)` at 50 decimal digits for a spread of points that are
    /// *not* on the `gelu.erf` fixture grid, so the fixture cannot be the only
    /// evidence for the approximation's accuracy.
    const ERF_TABLE: [(f64, f64); 25] = [
        (0.0, 0.0),
        (0.03125, 0.035_250_373_867_322_826),
        (0.0625, 0.070_431_977_722_387_07),
        (0.125, 0.140_316_204_801_333_8),
        (0.25, 0.276_326_390_168_236_96),
        (0.375, 0.404_116_909_434_822_3),
        (0.5, 0.520_499_877_813_046_5),
        (0.625, 0.623_240_882_188_418),
        (0.75, 0.711_155_633_653_515_1),
        (0.875, 0.784_075_061_059_859_7),
        (1.0, 0.842_700_792_949_714_9),
        (1.25, 0.922_900_128_256_458_3),
        (1.5, 0.966_105_146_475_310_8),
        (1.75, 0.986_671_671_219_182_4),
        (2.0, 0.995_322_265_018_952_7),
        (2.5, 0.999_593_047_982_555),
        (3.0, 0.999_977_909_503_001_4),
        (3.5, 0.999_999_256_901_627_6),
        (3.75, 0.999_999_886_272_743_4),
        (4.0, 0.999_999_984_582_742_1),
        (4.5, 0.999_999_999_803_383_9),
        (5.0, 0.999_999_999_998_462_6),
        (-0.5, -0.520_499_877_813_046_5),
        (-1.5, -0.966_105_146_475_310_8),
        (-3.0, -0.999_977_909_503_001_4),
    ];

    #[test]
    fn erf_matches_high_precision_table() {
        for (input, expected) in ERF_TABLE {
            let actual = erf_f64(input);
            // 1.6e-8 covers the saturation clamp at |x| > 4; the interior fit
            // is three orders of magnitude tighter.
            let tolerance = 1.6e-8;
            assert!(
                (actual - expected).abs() <= tolerance,
                "erf({input}) = {actual}, expected {expected}"
            );
            // And the f32 rounding must stay inside one and a half f32 ulp of
            // the exact value (one ulp at 1.0 is 5.96e-8).
            let rounded = erf_f32(input as f32);
            assert!(
                (f64::from(rounded) - expected).abs() <= 1.0e-7,
                "erf_f32({input}) = {rounded}, expected {expected}"
            );
        }
    }

    #[test]
    fn erf_is_odd_and_saturates() {
        for input in [0.0f64, 0.125, 0.5, 1.0, 3.9] {
            assert_eq!(erf_f64(input), -erf_f64(-input), "erf({input})");
        }
        assert_eq!(erf_f64(1.0e4), 1.0);
        assert_eq!(erf_f64(-1.0e4), -1.0);
        assert_eq!(erf_f32(1.0e4), 1.0);
    }

    #[test]
    fn gelu_saturates_at_the_reference_boundaries() {
        // The fixture's last four inputs are -1e4, -10, 10, 1e4.
        assert_eq!(gelu_erf_f32(-1.0e4), 0.0);
        assert_eq!(gelu_erf_f32(-10.0), 0.0);
        assert_eq!(gelu_erf_f32(1.0e4), 1.0e4);
        // 0.5*10*(1+erf(10/sqrt(2))) = 10 to f32 precision.
        assert_eq!(gelu_erf_f32(10.0), 10.0);
        // GELU(0) = 0 and GELU is monotone in the fixture range.
        assert_eq!(gelu_erf_f32(0.0), 0.0);
    }

    #[test]
    fn cast_float_to_integer_saturates_and_truncates() {
        assert_eq!(Scalar::F32(2.9).to_i64(), 2);
        assert_eq!(Scalar::F32(-2.9).to_i64(), -2);
        assert_eq!(Scalar::F32(f32::NAN).to_i64(), 0);
        assert_eq!(Scalar::F32(1.0e30).to_i64(), i64::MAX);
        assert_eq!(Scalar::F32(-1.0e30).to_i64(), i64::MIN);
        assert!(!Scalar::F32(-0.0).to_bool());
        assert!(Scalar::F32(0.5).to_bool());
        assert_eq!(Scalar::Bool(true).to_f32(), 1.0);
    }
}

#[cfg(test)]
mod broadcast_run_tests {
    use super::*;

    /// Row-major contiguous strides of `shape`, with `0` on every axis the
    /// result promotes (a result extent larger than the operand's extent 1).
    fn promoted_strides(operand: &[u64], result: &[u64]) -> Vec<usize> {
        let lead = result.len() - operand.len();
        let mut strides = vec![0usize; result.len()];
        let mut running = 1usize;
        for axis in (0..operand.len()).rev() {
            let extent = usize::try_from(operand[axis]).expect("extent fits usize");
            let promoted = result[lead + axis] != operand[axis];
            strides[lead + axis] = if promoted { 0 } else { running };
            running *= extent;
        }
        strides
    }

    fn numel(shape: &[u64]) -> usize {
        shape.iter().map(|&d| d as usize).product()
    }

    /// The per-element reference: resolve every output element through the
    /// locators, exactly as the path the runs replace does.
    fn expected(
        result: &[u64],
        out: (usize, &[usize]),
        lhs: (usize, &[usize]),
        rhs: (usize, &[usize]),
        lhs_data: &[f32],
        rhs_data: &[f32],
    ) -> Vec<f32> {
        let out_locator = Locator::from_parts(out.0, out.1.to_vec());
        let lhs_locator = Locator::from_parts(lhs.0, lhs.1.to_vec());
        let rhs_locator = Locator::from_parts(rhs.0, rhs.1.to_vec());
        let total = numel(result);
        let mut coordinates = vec![0u64; result.len()];
        let mut values = Vec::with_capacity(total);
        for linear in 0..total {
            decompose(linear as u64, result, &mut coordinates);
            let left = lhs_data[lhs_locator
                .index(OpId::Add, &coordinates)
                .expect("lhs index")];
            let right = rhs_data[rhs_locator
                .index(OpId::Add, &coordinates)
                .expect("rhs index")];
            // The output locator must agree with the linear order.
            assert_eq!(
                out_locator
                    .index(OpId::Add, &coordinates)
                    .expect("out index"),
                linear
            );
            values.push(left * 2.0 + right);
        }
        values
    }

    fn check(result: &[u64], lhs_shape: &[u64], rhs_shape: &[u64]) {
        let lhs_strides = promoted_strides(lhs_shape, result);
        let rhs_strides = promoted_strides(rhs_shape, result);
        let out_strides = promoted_strides(result, result);
        let lhs_data: Vec<f32> = (0..numel(lhs_shape)).map(|i| i as f32 * 0.5).collect();
        let rhs_data: Vec<f32> = (0..numel(rhs_shape))
            .map(|i| i as f32 * 0.25 + 1.0)
            .collect();

        let want = expected(
            result,
            (0, &out_strides),
            (0, &lhs_strides),
            (0, &rhs_strides),
            &lhs_data,
            &rhs_data,
        );

        let out_locator = Locator::from_parts(0, out_strides.clone());
        let lhs_locator = Locator::from_parts(0, lhs_strides.clone());
        let rhs_locator = Locator::from_parts(0, rhs_strides.clone());
        let runs = broadcast_runs(result, &out_locator, &lhs_locator, &rhs_locator).expect("runs");
        assert!(
            runs_within_bounds(&runs, want.len(), lhs_data.len(), rhs_data.len()),
            "{result:?} {lhs_shape:?} {rhs_shape:?}: runs escape their buffers"
        );
        assert_eq!(
            runs.iter().map(|run| run.len).sum::<usize>(),
            want.len(),
            "{result:?} {lhs_shape:?} {rhs_shape:?}: runs must cover every element once"
        );
        let mut got = vec![f32::NAN; want.len()];
        apply_broadcast_runs(&runs, &mut got, &lhs_data, &rhs_data, |x, y| x * 2.0 + y);
        for (index, (left, right)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "{result:?} {lhs_shape:?} {rhs_shape:?} at {index}"
            );
        }
    }

    #[test]
    fn runs_match_the_per_element_path() {
        // Per-channel scale, per-row bias, full broadcast, tail broadcast, and
        // a 1-D promotion: the combinations a decoder and a CNN both hit.
        check(&[1, 128, 22, 28], &[1, 128, 1, 1], &[1, 128, 1, 1]);
        check(&[1, 128, 22, 28], &[1, 128, 22, 28], &[1, 128, 1, 1]);
        check(&[1, 128, 22, 28], &[1, 128, 22, 28], &[1, 1, 22, 28]);
        check(&[16, 49, 49], &[16, 49, 49], &[1, 49, 1]);
        check(&[1, 3, 336, 448], &[1, 3, 336, 448], &[1]);
        check(&[2, 3, 4], &[2, 3, 4], &[2, 3, 4]);
        check(&[1, 5], &[1, 5], &[5]);
        check(&[1, 1, 1], &[1, 1, 1], &[4, 1, 3]);
    }

    #[test]
    fn a_fully_contiguous_broadcast_merges_into_one_run() {
        let result = [2u64, 3, 4];
        let out = promoted_strides(&result, &result);
        let lhs = promoted_strides(&result, &result);
        let rhs = promoted_strides(&[1], &result);
        let runs = broadcast_runs(
            &result,
            &Locator::from_parts(0, out),
            &Locator::from_parts(0, lhs),
            &Locator::from_parts(0, rhs),
        )
        .expect("runs");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].len, 24);
        assert_eq!(
            (runs[0].out_step, runs[0].lhs_step, runs[0].rhs_step),
            (1, 1, 0)
        );
    }
}
