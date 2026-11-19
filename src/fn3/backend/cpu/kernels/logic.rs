//! Comparison, selection and masking kernels: `Equal`, `Less`, `Greater`,
//! `Where`, `Trilu` and `ScatterND`.
//!
//! # Semantics
//!
//! - `Equal`/`Less`/`Greater` broadcast right-aligned like the arithmetic ops
//!   and produce `Bool` (one byte per element, exactly `0` or `1`). The
//!   comparison is performed in the operand's own dtype, so an `I64` compare
//!   never loses precision through `f32`.
//! - `Where(cond, x, y)` selects whole elements from `x` or `y`; it never
//!   interprets them, so it works for every dtype and cannot convert one.
//! - `Trilu` keeps one triangle of the last two axes and zeroes the other.
//!   ONNX carries `upper` as an attribute and `k` as an optional input;
//!   [`OpParams`] has no boolean field, so `upper` is read from `axes[0]`
//!   (`0` = lower, anything else = upper, which is the ONNX default) and
//!   documented here as an fn3 encoding the importer must set explicitly.
//! - `ScatterND` copies `data`, then writes each update block at its index
//!   tuple in ascending tuple order; a duplicated tuple therefore ends with the
//!   highest tuple winning, which is deterministic and documented rather than
//!   "unspecified".
//!
//! NaN comparisons are IEEE: `Equal(NaN, NaN)` is `false`, and `Less`/`Greater`
//! are `false` for any NaN operand.

use crate::inference::backend::OpParams;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

use super::{
    at, at_mut, checked_product, copy_element, decompose, read_i64_scalar, require_dtype,
    require_output_shape, zero_element, Locator,
};

/// Which comparison a [`compare`] call performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Comparison {
    Equal,
    Less,
    Greater,
}

/// `Equal`/`Less`/`Greater` with broadcasting, producing `Bool`.
pub(crate) fn compare(
    op: OpId,
    comparison: Comparison,
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: lhs.dtype(),
            actual: rhs.dtype(),
        });
    }
    require_dtype(op, out.meta(), DType::Bool)?;
    let dtype = lhs.dtype();
    if comparison != Comparison::Equal && dtype == DType::Bool {
        return Err(Fn3Error::UnsupportedDType {
            op: Some(op),
            dtype,
            context: "cpu comparison kernel (less/greater need an ordered dtype)",
        });
    }
    let shape = lhs.shape().broadcast_with(rhs.shape(), op)?;
    require_output_shape(op, &shape, out)?;
    let elements = shape.numel()?;
    if elements == 0 {
        return Ok(());
    }

    let lhs_locator = Locator::new(op, lhs.meta(), &shape)?;
    let rhs_locator = Locator::new(op, rhs.meta(), &shape)?;
    let out_locator = Locator::new(op, out.meta(), &shape)?;
    let destination = out.storage_slice_mut::<u8>()?;
    let dims = shape.dims().to_vec();
    let mut coordinates = vec![0u64; shape.rank()];

    macro_rules! run {
        ($ty:ty, $predicate:expr) => {{
            let predicate: fn($ty, $ty) -> bool = $predicate;
            let left = lhs.storage_slice::<$ty>()?;
            let right = rhs.storage_slice::<$ty>()?;
            for linear in 0..elements {
                decompose(linear, &dims, &mut coordinates);
                let x = at(op, left, lhs_locator.index(op, &coordinates)?)?;
                let y = at(op, right, rhs_locator.index(op, &coordinates)?)?;
                let verdict = predicate(x, y);
                let to = out_locator.index(op, &coordinates)?;
                *at_mut(op, destination, to)? = u8::from(verdict);
            }
        }};
    }

    match (comparison, dtype) {
        (Comparison::Equal, DType::F32) => run!(f32, |x: f32, y: f32| x == y),
        (Comparison::Less, DType::F32) => run!(f32, |x: f32, y: f32| x < y),
        (Comparison::Greater, DType::F32) => run!(f32, |x: f32, y: f32| x > y),
        (Comparison::Equal, DType::I64) => run!(i64, |x: i64, y: i64| x == y),
        (Comparison::Less, DType::I64) => run!(i64, |x: i64, y: i64| x < y),
        (Comparison::Greater, DType::I64) => run!(i64, |x: i64, y: i64| x > y),
        (Comparison::Equal, DType::I32) => run!(i32, |x: i32, y: i32| x == y),
        (Comparison::Less, DType::I32) => run!(i32, |x: i32, y: i32| x < y),
        (Comparison::Greater, DType::I32) => run!(i32, |x: i32, y: i32| x > y),
        (Comparison::Equal, DType::Bool) => run!(u8, |x: u8, y: u8| (x != 0) == (y != 0)),
        (_, other) => {
            return Err(Fn3Error::UnsupportedDType {
                op: Some(op),
                dtype: other,
                context: "cpu comparison kernel",
            })
        }
    }
    Ok(())
}

/// `Where(condition, x, y)` with three-way broadcasting.
pub(crate) fn where_select(
    op: OpId,
    condition: &TensorView<'_>,
    x: &TensorView<'_>,
    y: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    require_dtype(op, condition.meta(), DType::Bool)?;
    if x.dtype() != y.dtype() {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: x.dtype(),
            actual: y.dtype(),
        });
    }
    if out.dtype() != x.dtype() {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: x.dtype(),
            actual: out.dtype(),
        });
    }
    let partial = condition.shape().broadcast_with(x.shape(), op)?;
    let shape = partial.broadcast_with(y.shape(), op)?;
    require_output_shape(op, &shape, out)?;
    let elements = shape.numel()?;
    if elements == 0 {
        return Ok(());
    }

    let element_bytes = x.dtype().size_bytes();
    let condition_locator = Locator::new(op, condition.meta(), &shape)?;
    let x_locator = Locator::new(op, x.meta(), &shape)?;
    let y_locator = Locator::new(op, y.meta(), &shape)?;
    let out_locator = Locator::new(op, out.meta(), &shape)?;
    let mask = condition.storage_slice::<u8>()?;
    let x_bytes = x.storage_bytes();
    let y_bytes = y.storage_bytes();
    let destination = out.storage_slice_mut::<u8>()?;
    let dims = shape.dims().to_vec();
    let mut coordinates = vec![0u64; shape.rank()];
    for linear in 0..elements {
        decompose(linear, &dims, &mut coordinates);
        let selector = at(op, mask, condition_locator.index(op, &coordinates)?)?;
        let source_index = if selector != 0 {
            x_locator.index(op, &coordinates)?
        } else {
            y_locator.index(op, &coordinates)?
        };
        let source = if selector != 0 { x_bytes } else { y_bytes };
        let to = out_locator.index(op, &coordinates)?;
        copy_element(op, source, source_index, destination, to, element_bytes)?;
    }
    Ok(())
}

/// `Trilu`: keep the upper (`axes[0] != 0`, the default) or lower triangle of
/// the last two axes, relative to the diagonal offset `k` (optional `I64`
/// scalar input, default 0).
pub(crate) fn trilu(
    op: OpId,
    input: &TensorView<'_>,
    offset: Option<&TensorView<'_>>,
    out: &mut TensorMut<'_>,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    if out.dtype() != input.dtype() {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: input.dtype(),
            actual: out.dtype(),
        });
    }
    let rank = input.shape().rank();
    if rank < 2 {
        return Err(Fn3Error::RankMismatch {
            op: Some(op),
            expected: 2,
            actual: rank,
        });
    }
    let upper = match params.axes.as_slice() {
        [] => true,
        [flag] => *flag != 0,
        axes => {
            return Err(Fn3Error::InvalidShape {
                context: "trilu upper flag",
                detail: format!(
                    "op {op} reads axes[0] as the ONNX upper flag, but the plan supplies {axes:?}"
                ),
            })
        }
    };
    let k = match offset {
        Some(view) => read_i64_scalar(op, view, "trilu diagonal offset")?,
        None => 0,
    };
    require_output_shape(op, input.shape(), out)?;
    let elements = input.shape().numel()?;
    if elements == 0 {
        return Ok(());
    }

    let element_bytes = input.dtype().size_bytes();
    let out_locator = Locator::new(op, out.meta(), input.shape())?;
    let source = input.storage_bytes();
    let destination = out.storage_slice_mut::<u8>()?;
    let dims = input.shape().dims().to_vec();
    let mut coordinates = vec![0u64; rank];
    for linear in 0..elements {
        decompose(linear, &dims, &mut coordinates);
        let row = coordinates[rank - 2];
        let column = coordinates[rank - 1];
        let difference = i128::from(column) - i128::from(row);
        let keep = if upper {
            difference >= i128::from(k)
        } else {
            difference <= i128::from(k)
        };
        let to = out_locator.index(op, &coordinates)?;
        if keep {
            let from = input
                .meta()
                .checked_index(&coordinates)
                .map_err(|error| error.with_op(op))?;
            copy_element(op, source, from, destination, to, element_bytes)?;
        } else {
            zero_element(op, destination, to, element_bytes)?;
        }
    }
    Ok(())
}

/// `ScatterND(data, indices, updates)`.
///
/// `indices` is `I64` with shape `[..., k]`, `data` has rank `r >= k`, and
/// `updates` has shape `indices.shape[..-1] ++ data.shape[k..]`. Every index is
/// range-checked (negative indices count from the end) before anything is
/// written.
pub(crate) fn scatter_nd(
    op: OpId,
    data: &TensorView<'_>,
    indices: &TensorView<'_>,
    updates: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Result<(), Fn3Error> {
    if updates.dtype() != data.dtype() {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: data.dtype(),
            actual: updates.dtype(),
        });
    }
    if out.dtype() != data.dtype() {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: data.dtype(),
            actual: out.dtype(),
        });
    }
    let data_rank = data.shape().rank();
    let indices_rank = indices.shape().rank();
    if indices_rank == 0 {
        return Err(Fn3Error::RankMismatch {
            op: Some(op),
            expected: 1,
            actual: 0,
        });
    }
    let depth = indices.shape().dims()[indices_rank - 1];
    let depth_usize = usize::try_from(depth).map_err(|_| Fn3Error::ShapeOverflow {
        context: "cpu scatter_nd depth",
        axis: Some(indices_rank - 1),
        detail: "index depth does not fit in usize".to_string(),
    })?;
    if depth_usize > data_rank {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("an index depth of at most {data_rank} for {}", data.shape()),
            actual: format!("{depth}"),
        });
    }
    let mut expected_updates = indices.shape().dims()[..indices_rank - 1].to_vec();
    expected_updates.extend_from_slice(&data.shape().dims()[depth_usize..]);
    let expected_updates = Shape::from_dims(expected_updates);
    if updates.shape() != &expected_updates {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: expected_updates.to_string(),
            actual: updates.shape().to_string(),
        });
    }
    require_output_shape(op, data.shape(), out)?;

    // Start from an exact copy of `data`, then write the updates in ascending
    // index-tuple order.
    super::transform::copy_row_major(op, data, out)?;

    let tuples = checked_product(
        &indices.shape().dims()[..indices_rank - 1],
        "cpu scatter_nd tuples",
    )?;
    if tuples == 0 {
        return Ok(());
    }
    let block = checked_product(&data.shape().dims()[depth_usize..], "cpu scatter_nd block")?;
    let index_values = super::transform::index_values(op, indices)?;
    let element_bytes = data.dtype().size_bytes();
    let update_bytes = updates.storage_bytes();
    let out_locator = Locator::new(op, out.meta(), data.shape())?;
    let destination = out.storage_slice_mut::<u8>()?;
    let mut tuple_coordinates = vec![0u64; indices_rank - 1];
    let mut data_coordinates = vec![0u64; data_rank];
    let mut update_coordinates = vec![0u64; updates.shape().rank()];
    let block_dims = data.shape().dims()[depth_usize..].to_vec();

    for tuple in 0..tuples {
        if !tuple_coordinates.is_empty() {
            decompose(
                tuple,
                &indices.shape().dims()[..indices_rank - 1],
                &mut tuple_coordinates,
            );
        }
        let mut resolved = Vec::with_capacity(depth_usize);
        for position in 0..depth_usize {
            let flat = tuple * depth + position as u64;
            let raw = at(
                op,
                &index_values,
                usize::try_from(flat).map_err(|_| Fn3Error::ShapeOverflow {
                    context: "cpu scatter_nd index position",
                    axis: None,
                    detail: "index position does not fit in usize".to_string(),
                })?,
            )?;
            let extent = data.shape().dims()[position];
            let extent_i64 = i64::try_from(extent).map_err(|_| Fn3Error::ShapeOverflow {
                context: "cpu scatter_nd extent",
                axis: Some(position),
                detail: "extent does not fit in i64".to_string(),
            })?;
            let resolved_value = if raw < 0 {
                raw.saturating_add(extent_i64)
            } else {
                raw
            };
            if resolved_value < 0 || resolved_value >= extent_i64 {
                return Err(Fn3Error::OutOfBounds {
                    what: "scatter_nd index",
                    index: raw as u64,
                    len: extent,
                });
            }
            resolved.push(resolved_value as u64);
        }

        for offset in 0..block {
            if !block_dims.is_empty() {
                decompose(offset, &block_dims, &mut data_coordinates[depth_usize..]);
            }
            data_coordinates[..depth_usize].copy_from_slice(&resolved);
            update_coordinates[..indices_rank - 1].copy_from_slice(&tuple_coordinates);
            if !block_dims.is_empty() {
                decompose(
                    offset,
                    &block_dims,
                    &mut update_coordinates[indices_rank - 1..],
                );
            }
            let from = updates
                .meta()
                .checked_index(&update_coordinates)
                .map_err(|error| error.with_op(op))?;
            let to = out_locator.index(op, &data_coordinates)?;
            copy_element(op, update_bytes, from, destination, to, element_bytes)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_enum_is_total() {
        assert_ne!(Comparison::Equal, Comparison::Less);
        assert_ne!(Comparison::Less, Comparison::Greater);
    }
}
