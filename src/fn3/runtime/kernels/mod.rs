//! Scalar NCHW kernels for the twelve CNN op ids, plus the dispatch table.
//!
//! # What lives here
//!
//! The alpha CPU reference in `crate::inference::backend::cpu::kernels` predates
//! the v3 runtime and covers the forty ops the decoder profile needs. The twelve
//! CNN ids (`Conv` 41 … `TopK` 52) are executed here, by scalar loops over
//! `f32` NCHW operands, and these modules are the *only* implementation of them:
//!
//! | Module | Ops |
//! |---|---|
//! | `conv` | `Conv` (grouped, dilated, padded, 1/2/3 spatial axes) |
//! | `pool` | `MaxPool`, `AveragePool`, `GlobalAveragePool` |
//! | `norm` | `BatchNorm` (inference form) |
//! | `resize` | `Resize` (nearest and linear, constant scales or sizes) |
//! | `pad` | `Pad` (constant/reflect/edge, negative pads crop) |
//! | `act` | `Clip`, `Sigmoid`, and `Maximum`/`Minimum` with broadcasting |
//! | `topk` | `TopK` (values and `i64` indices along one axis) |
//! | `data` | `Split`, `Slice` — delegated to the existing CPU kernels |
//!
//! This module deliberately contains no `Backend` implementation and no
//! `Support` query: it is a table of functions. Both the v3 executor
//! ([`crate::inference::runtime::Session`]) and
//! `CpuBackend::execute_op` call
//! `dispatch`, so a `Support::Full` verdict and the kernel behind it cannot
//! drift apart.
//!
//! # Numerics
//!
//! Every kernel is a plain scalar loop with a documented evaluation order:
//!
//! - compute dtype is `f32`; `i64`/`i32`/`bool` appear only as data (TopK
//!   indices, Slice bounds);
//! - `Conv` accumulates over input channels and then kernel positions, both in
//!   ascending row-major order, and adds the bias once at the end;
//! - a reduction never reorders its terms for a different thread count, so two
//!   runs of the same plan are bitwise identical whatever [`Threading`] the
//!   session was opened with;
//! - a failed kernel writes nothing it did not first validate: arity, dtype,
//!   shape, attributes and the output geometry are checked before the first
//!   write.
//!
//! [`Threading`]: crate::inference::runtime::Threading
//!
//! # Parallelism
//!
//! A kernel splits its output into whole rows and hands them to the session's
//! own pool only when the output has at least `MIN_PARALLEL_ELEMENTS` (8 Ki)
//! elements and at least `MIN_PARALLEL_ELEMENTS_PER_WORKER` elements for every
//! worker — below that, waking a worker costs more than the arithmetic. Rows are
//! independent and each row's arithmetic is fixed, so the split cannot change a
//! result. The split is always over the *outer* axes (`N*C` rows), so a row's
//! inner loop is untouched.
//!
//! # Shapes
//!
//! Every operand is contiguous NCHW/NCDHW: the arena gives each slot one
//! contiguous range and the plan never inserts a strided view. The kernels still
//! read the shape from the view's metadata rather than assuming it, and reject a
//! rank or extent the contract does not cover.

use crate::inference::backend::OpParams;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::runtime::SessionThreads;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

pub(crate) mod act;
pub(crate) mod conv;
pub(crate) mod data;
pub(crate) mod fused;
pub(crate) mod int;
pub(crate) mod movement;
pub(crate) mod norm;
pub(crate) mod pad;
pub(crate) mod pool;
pub(crate) mod resize;
pub(crate) mod softmax;
pub(crate) mod topk;

/// Smallest output, in elements, worth handing to the session pool.
///
/// The phase-3b floor was a flat 64 KiB (16 Ki `f32` elements), chosen as four
/// cache lines per worker on a sixteen-core host. At eight workers that is 2 Ki
/// elements per hand-off — below the point a Rayon wake-up is amortised — while
/// the pointwise glue segments the row split targets are 8–50 Ki elements, so
/// they stayed serial. The floor is now counted in elements and scaled by the
/// session's worker count: a hand-off needs at least
/// [`MIN_PARALLEL_ELEMENTS_PER_WORKER`] elements per worker and never fewer
/// than [`MIN_PARALLEL_ELEMENTS`] in total (below ~8 Ki elements the pool
/// costs more than the arithmetic).
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) const MIN_PARALLEL_ELEMENTS: usize = 8 * 1024;

/// Elements per worker a parallel row loop must have before it uses the pool.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) const MIN_PARALLEL_ELEMENTS_PER_WORKER: usize = 1024;

/// `true` for the twelve op ids these kernels own.
pub(crate) fn is_cnn_op(op: OpId) -> bool {
    matches!(
        op,
        OpId::Conv
            | OpId::MaxPool
            | OpId::AveragePool
            | OpId::GlobalAveragePool
            | OpId::BatchNorm
            | OpId::Resize
            | OpId::Pad
            | OpId::Clip
            | OpId::Sigmoid
            | OpId::Maximum
            | OpId::Minimum
            | OpId::TopK
            | OpId::Concat
            | OpId::Softmax
            | OpId::Reshape
            | OpId::Split
            | OpId::Slice
    )
}

/// Execute one CNN op.
///
/// `threads` is the session's own pool. `None` means "run on the calling
/// thread", which is what [`crate::inference::backend::cpu::CpuBackend`] passes:
/// a bare backend has no session and must not borrow one.
pub(crate) fn dispatch(
    op: OpId,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    match op {
        OpId::Conv => conv::run(inputs, outputs, params, threads),
        OpId::MaxPool | OpId::AveragePool | OpId::GlobalAveragePool => {
            pool::run(op, inputs, outputs, params, threads)
        }
        OpId::BatchNorm => norm::run(inputs, outputs, params),
        OpId::Resize => resize::run(inputs, outputs, params, threads),
        OpId::Pad => pad::run(inputs, outputs, params, threads),
        OpId::Clip | OpId::Sigmoid | OpId::Maximum | OpId::Minimum => {
            act::run(op, inputs, outputs, params, threads)
        }
        OpId::TopK => topk::run(inputs, outputs, params),
        OpId::Split | OpId::Slice | OpId::Reshape => data::run(op, inputs, outputs, params),
        OpId::Concat => {
            let Some(output) = outputs.first_mut() else {
                return Err(Fn3Error::ArityMismatch {
                    op,
                    expected_inputs: 1,
                    actual_inputs: inputs.len(),
                    expected_outputs: 1,
                    actual_outputs: outputs.len(),
                });
            };
            movement::concat(op, inputs, output, params, threads)
        }
        OpId::Softmax => {
            let (Some(input), Some(output)) = (inputs.first(), outputs.first_mut()) else {
                return Err(Fn3Error::ArityMismatch {
                    op,
                    expected_inputs: 1,
                    actual_inputs: inputs.len(),
                    expected_outputs: 1,
                    actual_outputs: outputs.len(),
                });
            };
            softmax::run(op, input, output, params, threads)
        }
        other => Err(Fn3Error::UnsupportedOp {
            op: other,
            backend: "fn3-runtime-kernels",
            reason: "not an op id this table owns; the legacy table in backend::cpu owns the rest"
                .to_string(),
        }),
    }
}

/// Check an exact input and output count, naming the op.
pub(crate) fn arity(
    op: OpId,
    inputs: usize,
    outputs: usize,
    expected_inputs: usize,
    expected_outputs: usize,
) -> Result<(), Fn3Error> {
    if inputs == expected_inputs && outputs == expected_outputs {
        Ok(())
    } else {
        Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs,
            expected_outputs,
            actual_inputs: inputs,
            actual_outputs: outputs,
        })
    }
}

/// Check that the input count is one of `allowed` and the output count is exact.
pub(crate) fn flexible_arity(
    op: OpId,
    inputs: usize,
    outputs: usize,
    allowed_inputs: &[usize],
    expected_outputs: usize,
) -> Result<(), Fn3Error> {
    if allowed_inputs.contains(&inputs) && outputs == expected_outputs {
        Ok(())
    } else {
        Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs: allowed_inputs.last().copied().unwrap_or(0),
            expected_outputs,
            actual_inputs: inputs,
            actual_outputs: outputs,
        })
    }
}

/// Require `dtype` on a view, naming the op.
pub(crate) fn require_dtype(op: OpId, view: &TensorView<'_>, dtype: DType) -> Result<(), Fn3Error> {
    view.meta().require_dtype(Some(op), dtype)
}

/// An input's elements as `f32`.
pub(crate) fn input_f32<'a>(op: OpId, view: &'a TensorView<'_>) -> Result<&'a [f32], Fn3Error> {
    require_dtype(op, view, DType::F32)?;
    view.as_slice::<f32>()
}

/// An output's elements as `f32`, mutably.
pub(crate) fn output_f32<'a>(
    op: OpId,
    view: &'a mut TensorMut<'_>,
) -> Result<&'a mut [f32], Fn3Error> {
    view.meta().require_dtype(Some(op), DType::F32)?;
    view.as_mut_slice::<f32>()
}

/// The concrete extents of `shape` as `usize`, checked.
pub(crate) fn extents(shape: &Shape, what: &'static str) -> Result<Vec<usize>, Fn3Error> {
    shape
        .dims()
        .iter()
        .map(|&extent| {
            usize::try_from(extent).map_err(|_| Fn3Error::ShapeOverflow {
                context: what,
                axis: None,
                detail: format!("extent {extent} does not fit in usize"),
            })
        })
        .collect()
}

/// `product(values)`, checked.
pub(crate) fn product(values: &[usize], what: &'static str) -> Result<usize, Fn3Error> {
    let mut result = 1usize;
    for (axis, &value) in values.iter().enumerate() {
        result = result
            .checked_mul(value)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: what,
                axis: Some(axis),
                detail: format!("{values:?} exceeds usize::MAX elements"),
            })?;
    }
    Ok(result)
}

/// Require `out` to hold exactly `expected` `f32` elements.
pub(crate) fn check_output_len(
    op: OpId,
    out: &[f32],
    expected: usize,
    shape: &Shape,
) -> Result<(), Fn3Error> {
    if out.len() == expected {
        Ok(())
    } else {
        Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{expected} elements for {shape}"),
            actual: format!("{} elements", out.len()),
        })
    }
}

/// Require an operand to hold exactly `elements` `f32` values.
pub(crate) fn check_input_len(
    op: OpId,
    values: &[f32],
    elements: usize,
    shape: &Shape,
    role: &'static str,
) -> Result<(), Fn3Error> {
    if values.len() == elements {
        Ok(())
    } else {
        Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{elements} elements for the {role} {shape}"),
            actual: format!("{} elements", values.len()),
        })
    }
}

/// Run `body` once per output row, in parallel when the output is large enough.
///
/// `row_len` is the number of `f32` elements per row; `out.len()` must be a
/// multiple of it. Row `index` is the outer (`N*C`-style) index, so a kernel can
/// derive the batch/channel pair from it.
///
/// The serial path is the same loop with the pool skipped, so the arithmetic and
/// its order are identical either way; only *where* it runs changes.
pub(crate) fn for_each_row<F>(
    threads: Option<&SessionThreads>,
    out: &mut [f32],
    row_len: usize,
    body: F,
) -> Result<(), Fn3Error>
where
    F: Fn(usize, &mut [f32]) -> Result<(), Fn3Error> + Sync + Send,
{
    if out.is_empty() {
        return Ok(());
    }
    if row_len == 0 || !out.len().is_multiple_of(row_len) {
        return Err(Fn3Error::InvalidShape {
            context: "runtime kernel row split",
            detail: format!(
                "the output has {} elements, which is not a whole number of {row_len}-element rows",
                out.len()
            ),
        });
    }
    #[cfg(feature = "parallel")]
    {
        if let Some(threads) = threads {
            let workers = threads.worker_count();
            let floor =
                MIN_PARALLEL_ELEMENTS.max(workers.saturating_mul(MIN_PARALLEL_ELEMENTS_PER_WORKER));
            // One task is one whole row: `par_chunks_mut(row_len)` never splits
            // a row, and the kernel's inner loop and arithmetic order are the
            // serial ones.
            if workers > 1 && out.len() >= floor && out.len() / row_len >= 2 {
                use rayon::prelude::*;
                return threads.install(|| {
                    out.par_chunks_mut(row_len)
                        .enumerate()
                        .try_for_each(|(index, row)| body(index, row))
                });
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    let _ = threads;
    for (index, row) in out.chunks_mut(row_len).enumerate() {
        body(index, row)?;
    }
    Ok(())
}

pub(crate) use crate::inference::backend::window::{resolve_pads, window_axes, window_output};

/// Require the planned output extent to be the extent this kernel computes.
pub(crate) fn require_extent(
    op: OpId,
    what: &'static str,
    axis: usize,
    computed: usize,
    planned: usize,
) -> Result<(), Fn3Error> {
    if computed == planned {
        Ok(())
    } else {
        Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{what} {axis} to be the planned extent {planned}"),
            actual: format!("the kernel's rule gives {computed}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigmoid(values: &mut [f32]) {
        for value in values.iter_mut() {
            *value = 1.0 / (1.0 + (-*value).exp());
        }
    }

    #[test]
    fn the_row_loop_is_bit_identical_around_the_parallel_floor() {
        // 160 rows of 80 values is 12 800 elements (50 KiB): below the old
        // flat 64 KiB floor, above the new 8 Ki-element one. The pooled path
        // must produce the same bits as the serial one.
        let row_len = 80usize;
        let values: Vec<f32> = (0..12_800)
            .map(|index| ((index as f32) * 0.001).sin() * 3.0 - 1.5)
            .collect();
        let mut serial = values.clone();
        for_each_row(None, &mut serial, row_len, |_, row| {
            sigmoid(row);
            Ok(())
        })
        .unwrap();
        #[cfg(feature = "parallel")]
        {
            let threads =
                SessionThreads::new(crate::inference::runtime::Threading::Fixed(4)).unwrap();
            let mut pooled = values.clone();
            for_each_row(Some(&threads), &mut pooled, row_len, |_, row| {
                sigmoid(row);
                Ok(())
            })
            .unwrap();
            assert_eq!(
                pooled
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                serial
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "the pooled row loop changed a bit"
            );
        }
        // And the same loop below every floor still runs (serially) correctly.
        let mut small = vec![0.5f32; 4096];
        for_each_row(None, &mut small, 4, |_, row| {
            sigmoid(row);
            Ok(())
        })
        .unwrap();
        let expected = 1.0 / (1.0 + (-0.5f32).exp());
        assert!(small.iter().all(|value| *value == expected));
    }

    #[test]
    fn product_and_extents_are_checked() {
        assert_eq!(product(&[2, 3, 4], "test"), Ok(24));
        assert!(product(&[usize::MAX, 2], "test").is_err());
        assert_eq!(
            extents(&Shape::from_dims([2, 3]), "test").unwrap(),
            vec![2, 3]
        );
    }
}
