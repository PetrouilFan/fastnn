//! Matrix product: rank-2 and batched, with NumPy batch broadcasting.
//!
//! # Contract
//!
//! `MatMul(lhs, rhs) -> out` follows ONNX/NumPy `matmul` for operands of rank
//! `>= 2`:
//!
//! ```text
//! lhs: [B..., M, K]     rhs: [B'..., K, N]     out: [B'', ..., M, N]
//! ```
//!
//! where `B''` is the right-aligned broadcast of `B...` and `B'...`. Both
//! operands are addressed through their metadata, so a transposed right-hand
//! side (`K @ V^T`, the attention score product) or a strided view needs no
//! materialisation.
//!
//! Rank-1 operands (ONNX vector-matrix products) are **not** implemented: the
//! pinned GPT-2 graph only uses rank-2 and rank-4 `MatMul` plus one rank-3 by
//! rank-2 projection, and a silent vector promotion would change the output
//! rank. They are rejected with `UnsupportedOp`.
//!
//! # Accumulation order (the reference every faster kernel is measured against)
//!
//! Each output element is computed with **one `f32` accumulator, adding the
//! products in ascending `k`**, starting from `+0.0`:
//!
//! ```text
//! out[b, m, n] = ((0 + lhs[b, m, 0] * rhs[b, 0, n]) + lhs[b, m, 1] * rhs[b, 1, n]) + ...
//! ```
//!
//! There is no pairwise/tree summation, no `f64` accumulator and no fused
//! multiply-add, so the result is a pure function of the operands on every
//! target. An empty product (`K == 0`) is `+0.0`. A blocked, SIMD or threaded
//! implementation (FN3-42 / #71) may change this order only by publishing a new
//! measured tolerance against `fixtures/parity/README.md`; it may not silently
//! differ.
//!
//! # Performance seam
//!
//! [`matmul`] is the correctness reference. [`dot_rows`] is the single
//! inner-product loop, isolated so #71 can replace it with
//! `blocked_row_matmul`-style blocking and SIMD (the 2.x
//! `src/backend/cpu/microkernels/misc.rs` has a `simd_dot_product` worth
//! copying) without touching layout, validation or broadcasting.

use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;
use crate::fn3::shape::Shape;
use crate::fn3::tensor::{TensorMut, TensorView};

use super::{at_mut, require_dtype, require_output_shape, Locator};

/// Matrix product over `f32`.
pub(crate) fn matmul(
    op: OpId,
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    require_dtype(op, lhs.meta(), DType::F32)?;
    require_dtype(op, rhs.meta(), DType::F32)?;
    require_dtype(op, out.meta(), DType::F32)?;

    let lhs_rank = lhs.shape().rank();
    let rhs_rank = rhs.shape().rank();
    for (view, rank) in [(lhs, lhs_rank), (rhs, rhs_rank)] {
        if rank < 2 {
            return Err(Fn3Error::UnsupportedOp {
                op,
                backend: crate::fn3::backend::cpu::CPU_BACKEND_NAME,
                reason: format!(
                    "the fn3 matmul contract covers rank >= 2 with the matrix axes last; \
                     found rank {rank} ({}). Rank-1 vector products are owned by FN3-07 (#53) \
                     and are rejected rather than promoted to a different output rank",
                    view.shape()
                ),
            });
        }
    }

    let lhs_dims = lhs.shape().dims();
    let rhs_dims = rhs.shape().dims();
    let rows = lhs_dims[lhs_rank - 2];
    let inner = lhs_dims[lhs_rank - 1];
    let rhs_inner = rhs_dims[rhs_rank - 2];
    let columns = rhs_dims[rhs_rank - 1];
    if inner != rhs_inner {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("rhs contraction of {inner} (lhs {})", lhs.shape()),
            actual: format!("rhs {} contracts {rhs_inner}", rhs.shape()),
        });
    }

    let lhs_batch = Shape::from(&lhs_dims[..lhs_rank - 2]);
    let rhs_batch = Shape::from(&rhs_dims[..rhs_rank - 2]);
    let batch = lhs_batch.broadcast_with(&rhs_batch, op)?;

    let mut expected_dims = batch.dims().to_vec();
    expected_dims.push(rows);
    expected_dims.push(columns);
    let expected = Shape::from_dims(expected_dims);
    require_output_shape(op, &expected, out)?;

    let out_numel = expected.numel()?;
    if out_numel == 0 {
        return Ok(());
    }
    let batch_numel = batch.numel()?;

    // Effective batch strides, right-aligned and zero where broadcast.
    let lhs_batch_strides = batch_strides(lhs, &batch)?;
    let rhs_batch_strides = batch_strides(rhs, &batch)?;
    let lhs_row_step = stride_usize(op, lhs, lhs_rank - 1)?;
    let lhs_column_step = stride_usize(op, lhs, lhs_rank - 2)?;
    let rhs_row_step = stride_usize(op, rhs, rhs_rank - 2)?;
    let rhs_column_step = stride_usize(op, rhs, rhs_rank - 1)?;

    let out_locator = Locator::new(op, out.meta(), &expected)?;
    let lhs_data = lhs.storage_slice::<f32>()?;
    let rhs_data = rhs.storage_slice::<f32>()?;
    let out_data = out.storage_slice_mut::<f32>()?;

    let rank = batch.rank();
    let mut batch_coordinates = vec![0u64; rank];
    let mut out_coordinates = vec![0u64; expected.rank()];

    for batch_linear in 0..batch_numel {
        if rank > 0 {
            super::decompose(batch_linear, batch.dims(), &mut batch_coordinates);
        }
        // Bases of the two batch slabs. `rows`/`columns` and the batch extents
        // were all validated by `broadcast_with` and metadata construction, so
        // these sums stay inside the storage span; every access below is still
        // checked through `get`.
        let mut lhs_base = lhs.meta().offset() as u64;
        let mut rhs_base = rhs.meta().offset() as u64;
        for axis in 0..rank {
            let coordinate = batch_coordinates[axis];
            lhs_base = lhs_base
                .checked_add(
                    coordinate
                        .checked_mul(lhs_batch_strides[axis] as u64)
                        .ok_or_else(|| Fn3Error::ShapeOverflow {
                            context: "cpu matmul lhs batch base",
                            axis: Some(axis),
                            detail: format!(
                                "coordinate {coordinate} times batch stride {} overflows",
                                lhs_batch_strides[axis]
                            ),
                        })?,
                )
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "cpu matmul lhs batch base",
                    axis: Some(axis),
                    detail: "element index exceeds u64::MAX".to_string(),
                })?;
            rhs_base = rhs_base
                .checked_add(
                    coordinate
                        .checked_mul(rhs_batch_strides[axis] as u64)
                        .ok_or_else(|| Fn3Error::ShapeOverflow {
                            context: "cpu matmul rhs batch base",
                            axis: Some(axis),
                            detail: format!(
                                "coordinate {coordinate} times batch stride {} overflows",
                                rhs_batch_strides[axis]
                            ),
                        })?,
                )
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "cpu matmul rhs batch base",
                    axis: Some(axis),
                    detail: "element index exceeds u64::MAX".to_string(),
                })?;
            out_coordinates[axis] = coordinate;
        }
        // Widening: the batch base is an element index inside an allocation that
        // already fits in `usize` (see the element-count note in `kernels`).
        let lhs_base = lhs_base as usize;
        let rhs_base = rhs_base as usize;

        for row in 0..rows {
            let row_offset =
                usize::try_from(row.checked_mul(lhs_column_step as u64).ok_or_else(|| {
                    Fn3Error::ShapeOverflow {
                        context: "cpu matmul lhs row",
                        axis: None,
                        detail: format!("row {row} times stride {lhs_column_step} overflows"),
                    }
                })?)
                .map_err(|_| Fn3Error::ShapeOverflow {
                    context: "cpu matmul lhs row",
                    axis: None,
                    detail: "row offset does not fit in usize".to_string(),
                })?;
            let row_base =
                lhs_base
                    .checked_add(row_offset)
                    .ok_or_else(|| Fn3Error::ShapeOverflow {
                        context: "cpu matmul lhs row",
                        axis: None,
                        detail: "element index exceeds usize::MAX".to_string(),
                    })?;
            out_coordinates[expected.rank() - 2] = row;
            for column in 0..columns {
                let column_offset =
                    usize::try_from(column.checked_mul(rhs_column_step as u64).ok_or_else(
                        || Fn3Error::ShapeOverflow {
                            context: "cpu matmul rhs column",
                            axis: None,
                            detail: format!(
                                "column {column} times stride {rhs_column_step} overflows"
                            ),
                        },
                    )?)
                    .map_err(|_| Fn3Error::ShapeOverflow {
                        context: "cpu matmul rhs column",
                        axis: None,
                        detail: "column offset does not fit in usize".to_string(),
                    })?;
                let column_base =
                    rhs_base
                        .checked_add(column_offset)
                        .ok_or_else(|| Fn3Error::ShapeOverflow {
                            context: "cpu matmul rhs column",
                            axis: None,
                            detail: "element index exceeds usize::MAX".to_string(),
                        })?;
                let accumulator = dot_rows(
                    op,
                    lhs_data,
                    row_base,
                    lhs_row_step,
                    rhs_data,
                    column_base,
                    rhs_row_step,
                    inner,
                )?;
                out_coordinates[expected.rank() - 1] = column;
                let index = out_locator.index(op, &out_coordinates)?;
                *at_mut(op, out_data, index)? = accumulator;
            }
        }
    }
    Ok(())
}

/// One output element: `sum over k` of `lhs[k] * rhs[k]`, ascending `k`, one
/// `f32` accumulator.
///
/// `lhs_start`/`rhs_start` are element indices; `lhs_step`/`rhs_step` are the
/// element strides between consecutive `k`. Both bounds are checked once as a
/// slice range, and the `step_by` iterators cannot then run past the end, so
/// this function contains no unchecked indexing.
#[allow(clippy::too_many_arguments)]
fn dot_rows(
    op: OpId,
    lhs: &[f32],
    lhs_start: usize,
    lhs_step: usize,
    rhs: &[f32],
    rhs_start: usize,
    rhs_step: usize,
    inner: u64,
) -> Result<f32, Fn3Error> {
    if inner == 0 {
        // The empty product is +0.0, matching `sum([])`.
        return Ok(0.0);
    }
    if lhs_step == 0 || rhs_step == 0 {
        return Err(Fn3Error::InvalidStride {
            axis: 0,
            stride: 0,
            detail: format!(
                "op {op}: the contraction axis has a zero stride; a zero-stride contraction \
                 is not a matrix product (input is an explicitly broadcast view)"
            ),
        });
    }
    let inner = usize::try_from(inner).map_err(|_| Fn3Error::ShapeOverflow {
        context: "cpu matmul contraction length",
        axis: None,
        detail: "contraction length does not fit in usize".to_string(),
    })?;
    let last = inner - 1;
    let lhs_end = lhs_start
        .checked_add(
            last.checked_mul(lhs_step)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "cpu matmul lhs row span",
                    axis: None,
                    detail: format!("{last} times stride {lhs_step} overflows"),
                })?,
        )
        .ok_or_else(|| Fn3Error::ShapeOverflow {
            context: "cpu matmul lhs row span",
            axis: None,
            detail: "row span exceeds usize::MAX".to_string(),
        })?;
    let rhs_end = rhs_start
        .checked_add(
            last.checked_mul(rhs_step)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "cpu matmul rhs column span",
                    axis: None,
                    detail: format!("{last} times stride {rhs_step} overflows"),
                })?,
        )
        .ok_or_else(|| Fn3Error::ShapeOverflow {
            context: "cpu matmul rhs column span",
            axis: None,
            detail: "column span exceeds usize::MAX".to_string(),
        })?;
    let lhs_row = lhs
        .get(lhs_start..=lhs_end)
        .ok_or_else(|| super::kernel_bounds_error(op, lhs_end, lhs.len()))?;
    let rhs_column = rhs
        .get(rhs_start..=rhs_end)
        .ok_or_else(|| super::kernel_bounds_error(op, rhs_end, rhs.len()))?;
    Ok(lhs_row
        .iter()
        .step_by(lhs_step)
        .zip(rhs_column.iter().step_by(rhs_step))
        .fold(0.0f32, |accumulator, (&left, &right)| {
            accumulator + left * right
        }))
}

/// Effective strides of `view`'s batch axes for the broadcast batch `batch`.
fn batch_strides(view: &TensorView<'_>, batch: &Shape) -> Result<Vec<usize>, Fn3Error> {
    let view_rank = view.shape().rank();
    let batch_axes = view_rank - 2;
    let lead = batch.rank() - batch_axes;
    let mut strides = vec![0usize; batch.rank()];
    for axis in lead..batch.rank() {
        let operand_axis = axis - lead;
        let extent = view.shape().dims()[operand_axis];
        let stride = view.strides().as_slice()[operand_axis];
        let stride = usize::try_from(stride).map_err(|_| Fn3Error::InvalidStride {
            axis: operand_axis,
            stride,
            detail: "negative strides are not addressable in the alpha view contract".to_string(),
        })?;
        strides[axis] = if extent == 1 && batch.dims()[axis] != 1 {
            0
        } else {
            stride
        };
    }
    Ok(strides)
}

/// A matrix-axis stride as `usize`.
///
/// A zero stride is legal here (it means "the same row/column every time",
/// which is what a broadcast view of that axis says); the contraction step is
/// rejected separately inside [`dot_rows`], where a zero step would make
/// `step_by` panic.
fn stride_usize(op: OpId, view: &TensorView<'_>, axis: usize) -> Result<usize, Fn3Error> {
    let stride = view.strides().as_slice()[axis];
    if stride < 0 {
        return Err(Fn3Error::InvalidStride {
            axis,
            stride,
            detail: format!(
                "op {op}: a matrix axis of {} has negative stride {stride}",
                view.shape()
            ),
        });
    }
    usize::try_from(stride).map_err(|_| Fn3Error::InvalidStride {
        axis,
        stride,
        detail: format!("op {op}: stride does not fit in usize"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fn3::device::DeviceId;
    use crate::fn3::tensor::{Tensor, TensorMeta};

    fn matmul_via_backend(lhs: &Tensor, rhs: &Tensor, out_shape: [u64; 2]) -> Vec<f32> {
        let mut out =
            Tensor::zeros(Shape::from_dims(out_shape), DType::F32, DeviceId::Cpu).unwrap();
        let lhs_view = lhs.as_view().unwrap();
        let rhs_view = rhs.as_view().unwrap();
        let mut out_view = out.as_mut_view().unwrap();
        matmul(OpId::MatMul, &lhs_view, &rhs_view, &mut out_view).unwrap();
        out.as_f32_slice().unwrap().to_vec()
    }

    #[test]
    fn rank_two_product_accumulates_ascending_k() {
        let lhs =
            Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs =
            Tensor::from_f32(Shape::from_dims([3, 2]), &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        assert_eq!(
            matmul_via_backend(&lhs, &rhs, [2, 2]),
            vec![58.0, 64.0, 139.0, 154.0]
        );
    }

    #[test]
    fn transposed_view_is_honoured_without_a_copy() {
        // rhs logically [3, 2] but stored as [2, 3] with strides [1, 3].
        let storage =
            Tensor::from_f32(Shape::from_dims([2, 3]), &[7.0, 9.0, 11.0, 8.0, 10.0, 12.0]).unwrap();
        let meta = TensorMeta::strided(
            Shape::from_dims([3, 2]),
            crate::fn3::shape::Strides::from_elements(vec![1, 3]),
            0,
            DType::F32,
            DeviceId::Cpu,
        )
        .unwrap();
        let rhs = Tensor::from_storage(storage.storage().clone(), meta).unwrap();
        let lhs =
            Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            matmul_via_backend(&lhs, &rhs, [2, 2]),
            vec![58.0, 64.0, 139.0, 154.0]
        );
    }

    #[test]
    fn empty_contraction_is_zero() {
        let lhs = Tensor::zeros(Shape::from_dims([2, 0]), DType::F32, DeviceId::Cpu).unwrap();
        let rhs = Tensor::zeros(Shape::from_dims([0, 3]), DType::F32, DeviceId::Cpu).unwrap();
        assert_eq!(matmul_via_backend(&lhs, &rhs, [2, 3]), vec![0.0; 6]);
    }
}
