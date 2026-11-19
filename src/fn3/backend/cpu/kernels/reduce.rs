//! Reductions: `ReduceMean`, `ReduceSum` and `ArgMax`.
//!
//! # Accumulation order
//!
//! Reductions walk the **input** in ascending row-major order and add each
//! element to its output cell as it is visited. For a reduction over a
//! contiguous trailing axis that is the same as ascending axis index; for a
//! reduction over an interior axis it is still a single, documented order (the
//! one a faster implementation must reproduce or publish a tolerance against).
//!
//! `ReduceMean` divides the completed sum by the number of reduced elements;
//! it does not scale each element first. An empty reduction (one of the reduced
//! extents is 0) yields `+0.0` for both `ReduceSum` and `ReduceMean`; `0/0` is
//! undefined, and the alpha contract pins `+0.0` so a downstream comparison
//! never has to reason about a NaN that carries no information.
//!
//! # ArgMax tie rule
//!
//! `ArgMax` keeps the **first** occurrence of the maximum: a candidate replaces
//! the running best only when it compares strictly greater. NaN compares false,
//! so a NaN in the first position wins and a later NaN never does; if every
//! value is NaN the index is 0. `keep_dims` defaults to `false` here, which is
//! the [`OpParams`] default; the ONNX default is `true`, so the importer (#54)
//! must set it explicitly.

use crate::inference::backend::OpParams;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

use super::{at, at_mut, decompose, normalize_axis, require_dtype, require_output_shape, Locator};

/// Resolve and validate the reduced axes for `op`.
///
/// Empty `axes` means "every axis". The result is sorted and de-duplicated so
/// the output shape and the accumulation order do not depend on attribute
/// order.
pub(crate) fn resolve_axes(
    op: OpId,
    params: &OpParams,
    rank: usize,
) -> Result<Vec<usize>, Fn3Error> {
    if params.axes.is_empty() {
        return Ok((0..rank).collect());
    }
    let mut axes = Vec::with_capacity(params.axes.len());
    for &axis in &params.axes {
        let axis = normalize_axis(op, axis, rank, "reduce axes")?;
        if axes.contains(&axis) {
            return Err(Fn3Error::InvalidShape {
                context: "reduce axes",
                detail: format!("op {op}: axis {axis} is listed twice"),
            });
        }
        axes.push(axis);
    }
    axes.sort_unstable();
    Ok(axes)
}

/// Shape of a reduction over `axes`.
pub(crate) fn reduced_shape(input: &Shape, axes: &[usize], keep_dims: bool) -> Shape {
    let mut dims = Vec::with_capacity(input.rank());
    for (axis, &extent) in input.dims().iter().enumerate() {
        if axes.contains(&axis) {
            if keep_dims {
                dims.push(1);
            }
        } else {
            dims.push(extent);
        }
    }
    Shape::from_dims(dims)
}

/// Contiguous strides of `shape` (used to map output coordinates to a linear
/// slot in the accumulation buffer).
fn contiguous_strides(shape: &Shape) -> Vec<u64> {
    let mut strides = vec![1u64; shape.rank()];
    let mut running = 1u64;
    for axis in (0..shape.rank()).rev() {
        strides[axis] = running;
        running = running.saturating_mul(shape.dims()[axis]);
    }
    strides
}

/// `ReduceSum` / `ReduceMean` over `f32`.
pub(crate) fn reduce(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
    params: &OpParams,
    mean: bool,
) -> Result<(), Fn3Error> {
    require_dtype(op, input.meta(), DType::F32)?;
    require_dtype(op, out.meta(), DType::F32)?;
    let rank = input.shape().rank();
    let axes = resolve_axes(op, params, rank)?;
    let expected = reduced_shape(input.shape(), &axes, params.keep_dims);
    require_output_shape(op, &expected, out)?;

    let out_numel = expected.numel()?;
    let mut accumulator = vec![
        0.0f32;
        usize::try_from(out_numel).map_err(|_| {
            Fn3Error::ShapeOverflow {
                context: "cpu reduce output",
                axis: None,
                detail: "output element count does not fit in usize".to_string(),
            }
        })?
    ];
    let mut count: u64 = 1;
    for &axis in &axes {
        count = count
            .checked_mul(input.shape().dims()[axis])
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "cpu reduce count",
                axis: Some(axis),
                detail: "reduced element count exceeds u64::MAX".to_string(),
            })?;
    }
    if count == 0 {
        // Empty reduction: the accumulator stays +0.0 and is written below.
        // `0/0` is undefined; the alpha contract pins `+0.0` (module docs).
    }

    if out_numel > 0 && count > 0 {
        let out_strides = contiguous_strides(&expected);
        let locator = Locator::new(op, input.meta(), input.shape())?;
        let source = input.storage_slice::<f32>()?;
        let elements = input.shape().numel()?;
        let dims = input.shape().dims().to_vec();
        let mut coordinates = vec![0u64; rank];
        for linear in 0..elements {
            decompose(linear, &dims, &mut coordinates);
            let value = at(op, source, locator.index(op, &coordinates)?)?;
            let mut position = 0usize;
            let mut slot = 0u64;
            for axis in 0..rank {
                if axes.contains(&axis) {
                    if params.keep_dims {
                        position += 1;
                    }
                    continue;
                }
                slot = slot
                    .checked_add(
                        coordinates[axis]
                            .checked_mul(out_strides[position])
                            .ok_or_else(|| Fn3Error::ShapeOverflow {
                                context: "cpu reduce slot",
                                axis: Some(axis),
                                detail: "output slot exceeds u64::MAX".to_string(),
                            })?,
                    )
                    .ok_or_else(|| Fn3Error::ShapeOverflow {
                        context: "cpu reduce slot",
                        axis: Some(axis),
                        detail: "output slot exceeds u64::MAX".to_string(),
                    })?;
                position += 1;
            }
            let slot = usize::try_from(slot).map_err(|_| Fn3Error::ShapeOverflow {
                context: "cpu reduce slot",
                axis: None,
                detail: "output slot does not fit in usize".to_string(),
            })?;
            let cell = at_mut(op, &mut accumulator, slot)?;
            *cell += value;
        }
    }

    let divisor = if mean && count > 0 { count as f32 } else { 1.0 };
    let out_locator = Locator::new(op, out.meta(), &expected)?;
    let destination = out.storage_slice_mut::<f32>()?;
    let mut coordinates = vec![0u64; expected.rank()];
    for linear in 0..out_numel {
        decompose(linear, expected.dims(), &mut coordinates);
        let slot = usize::try_from(linear).map_err(|_| Fn3Error::ShapeOverflow {
            context: "cpu reduce output slot",
            axis: None,
            detail: "output slot does not fit in usize".to_string(),
        })?;
        let value = at(op, &accumulator, slot)? / divisor;
        let index = out_locator.index(op, &coordinates)?;
        *at_mut(op, destination, index)? = value;
    }
    Ok(())
}

/// `ArgMax` along one axis, producing `I64`.
pub(crate) fn argmax(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let dtype = input.dtype();
    if !matches!(dtype, DType::F32 | DType::I64 | DType::I32) {
        return Err(Fn3Error::UnsupportedDType {
            op: Some(op),
            dtype,
            context: "cpu argmax kernel (f32, i64 and i32 inputs)",
        });
    }
    require_dtype(op, out.meta(), DType::I64)?;
    let rank = input.shape().rank();
    if rank == 0 {
        return Err(Fn3Error::RankMismatch {
            op: Some(op),
            expected: 1,
            actual: 0,
        });
    }
    let axes = match params.axes.as_slice() {
        [] => vec![normalize_axis(op, 0, rank, "argmax axis")?],
        [axis] => vec![normalize_axis(op, *axis, rank, "argmax axis")?],
        axes => {
            return Err(Fn3Error::InvalidShape {
                context: "argmax axis",
                detail: format!("op {op} takes one axis, but the plan supplies {axes:?}"),
            })
        }
    };
    let expected = reduced_shape(input.shape(), &axes, params.keep_dims);
    require_output_shape(op, &expected, out)?;

    let out_numel = expected.numel()?;
    if out_numel == 0 {
        return Ok(());
    }
    let out_strides = contiguous_strides(&expected);
    let dims = input.shape().dims().to_vec();
    let elements = input.shape().numel()?;
    let axis = axes[0];
    let mut best_index = vec![0i64; out_numel as usize];
    let mut seen = vec![false; out_numel as usize];
    let mut coordinates = vec![0u64; rank];
    let mut out_coordinates = vec![0u64; expected.rank()];

    // The running best is kept in the input's own dtype, not in `f32`: a
    // `f32` staging value would lose the low bits of an `i64` above 2^53 and
    // could then report the wrong index.
    macro_rules! scan {
        ($source:ty) => {{
            let mut best_value: Vec<$source> = vec![<$source>::default(); out_numel as usize];
            let source = input.storage_slice::<$source>()?;
            let locator = Locator::new(op, input.meta(), input.shape())?;
            for linear in 0..elements {
                decompose(linear, &dims, &mut coordinates);
                let value = at(op, source, locator.index(op, &coordinates)?)?;
                let mut position = 0usize;
                let mut slot = 0u64;
                for candidate in 0..rank {
                    if candidate == axis {
                        if params.keep_dims {
                            position += 1;
                        }
                        continue;
                    }
                    out_coordinates[position] = coordinates[candidate];
                    slot = slot
                        .checked_add(
                            coordinates[candidate]
                                .checked_mul(out_strides[position])
                                .ok_or_else(|| Fn3Error::ShapeOverflow {
                                    context: "cpu argmax slot",
                                    axis: Some(candidate),
                                    detail: "output slot exceeds u64::MAX".to_string(),
                                })?,
                        )
                        .ok_or_else(|| Fn3Error::ShapeOverflow {
                            context: "cpu argmax slot",
                            axis: Some(candidate),
                            detail: "output slot exceeds u64::MAX".to_string(),
                        })?;
                    position += 1;
                }
                let slot = usize::try_from(slot).map_err(|_| Fn3Error::ShapeOverflow {
                    context: "cpu argmax slot",
                    axis: None,
                    detail: "output slot does not fit in usize".to_string(),
                })?;
                // First occurrence wins: only a strictly greater value replaces
                // the running best, and a NaN therefore never does.
                let first = !seen[slot];
                if first || value > best_value[slot] {
                    best_value[slot] = value;
                    best_index[slot] =
                        i64::try_from(coordinates[axis]).map_err(|_| Fn3Error::ShapeOverflow {
                            context: "cpu argmax index",
                            axis: Some(axis),
                            detail: "index does not fit in i64".to_string(),
                        })?;
                }
                seen[slot] = true;
            }
        }};
    }

    match dtype {
        DType::F32 => scan!(f32),
        DType::I64 => scan!(i64),
        DType::I32 => scan!(i32),
        other => {
            return Err(Fn3Error::UnsupportedDType {
                op: Some(op),
                dtype: other,
                context: "cpu argmax kernel (f32, i64 and i32 inputs)",
            })
        }
    }

    let out_locator = Locator::new(op, out.meta(), &expected)?;
    let destination = out.storage_slice_mut::<i64>()?;
    let mut coordinates = vec![0u64; expected.rank()];
    for linear in 0..out_numel {
        decompose(linear, expected.dims(), &mut coordinates);
        let slot = usize::try_from(linear).map_err(|_| Fn3Error::ShapeOverflow {
            context: "cpu argmax output slot",
            axis: None,
            detail: "output slot does not fit in usize".to_string(),
        })?;
        let index = out_locator.index(op, &coordinates)?;
        *at_mut(op, destination, index)? = at(op, &best_index, slot)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_shapes_keep_or_drop() {
        let input = Shape::from_dims([2, 3, 4]);
        assert_eq!(reduced_shape(&input, &[2], false), Shape::from_dims([2, 3]));
        assert_eq!(
            reduced_shape(&input, &[1, 2], true),
            Shape::from_dims([2, 1, 1])
        );
        assert_eq!(
            reduced_shape(&input, &[0, 1, 2], false),
            Shape::from_dims([])
        );
    }

    #[test]
    fn empty_axes_means_every_axis() {
        let params = OpParams::default();
        assert_eq!(
            resolve_axes(OpId::ReduceSum, &params, 3).unwrap(),
            vec![0, 1, 2]
        );
        let params = OpParams::default().with_axes([-1, 0]);
        assert_eq!(
            resolve_axes(OpId::ReduceSum, &params, 3).unwrap(),
            vec![0, 2]
        );
        assert!(resolve_axes(OpId::ReduceSum, &OpParams::default().with_axes([1, 1]), 3).is_err());
    }
}
