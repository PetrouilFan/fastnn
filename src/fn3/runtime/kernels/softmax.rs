//! `Softmax`, with the same arithmetic order as the legacy kernel and the
//! session's pool over independent groups.
//!
//! A softmax group is one index of the softmax axis for every combination of
//! the axes outside it. Groups never share elements, so they can be split
//! across workers and the result stays bit-identical: each group's maximum,
//! exponent sum and division are computed in ascending axis order, exactly as
//! the reference does.
//!
//! # The two layouts it optimizes
//!
//! - **Axis last** (`inner == 1`): a group is a contiguous run of `width`
//!   values, so rows are handed to [`for_each_row`](super::for_each_row).
//! - **Axis in the middle** (`outer == 1`, the detector-head DFL shape
//!   `[1, 64, 8400]` with axis 1): a group is a strided column, and the output
//!   is split into column ranges. Each worker gets its range as one
//!   `&mut [f32]` per position along the axis, which is a safe, disjoint view
//!   of the same buffer — no aliasing and no copy.
//!
//! Any other shape (a batch of mid-axis softmaxes) runs the serial loop: it is
//! rare, and the safe column split above does not generalize to it without
//! rebuilding the views per outer index.

use crate::inference::backend::OpParams;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::runtime::SessionThreads;
use crate::inference::tensor::{TensorMut, TensorView};

/// Execute `Softmax(input, axis)`.
pub(crate) fn run(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    if input.dtype() != crate::inference::dtype::DType::F32
        || out.dtype() != crate::inference::dtype::DType::F32
    {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: crate::inference::dtype::DType::F32,
            actual: input.dtype(),
        });
    }
    input.meta().require_contiguous("softmax input")?;
    out.meta().require_contiguous("softmax output")?;
    if input.shape().dims() != out.shape().dims() {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("the input shape {:?}", input.shape()),
            actual: format!("{:?}", out.shape()),
        });
    }
    let rank = input.shape().rank();
    let axis = match params.axes.as_slice() {
        [axis] => normalize(*axis, rank)?,
        [] => rank.saturating_sub(1),
        _ => {
            return Err(Fn3Error::InvalidShape {
                context: "softmax axis",
                detail: format!("expected one axis, got {:?}", params.axes),
            })
        }
    };
    let dims = input.shape().dims();
    let outer = product(&dims[..axis])?;
    let width = usize::try_from(dims[axis]).map_err(|_| Fn3Error::ShapeOverflow {
        context: "softmax width",
        axis: Some(axis),
        detail: "extent does not fit in usize".to_string(),
    })?;
    let inner = product(&dims[axis + 1..])?;
    if outer == 0 || width == 0 || inner == 0 {
        return Ok(());
    }
    let source = input.storage_slice::<f32>()?;
    let destination = out.storage_slice_mut::<f32>()?;
    if source.len() < outer * width * inner || destination.len() < outer * width * inner {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{} elements", outer * width * inner),
            actual: format!("{} input, {} output", source.len(), destination.len()),
        });
    }

    let workers = threads.map(SessionThreads::worker_count).unwrap_or(1);
    let elements = outer * width * inner;

    if inner == 1 {
        // Contiguous groups: one row per group, in ascending order.
        let pool = if workers > 1 { threads } else { None };
        return super::for_each_row(pool, destination, width, |row, slice| {
            let group = &source[row * width..(row + 1) * width];
            softmax_row(group, slice);
            Ok(())
        });
    }

    if outer == 1 && workers > 1 && destination.len() >= super::MIN_PARALLEL_ELEMENTS {
        // Mid-axis: split the inner axis into column ranges. Worker `w` owns
        // `[w*chunk, min((w+1)*chunk, inner))` and touches only indices whose
        // inner coordinate is in that range, so the ranges are disjoint and
        // every group is computed exactly as the serial loop computes it.
        let chunks = workers.min(inner);
        let chunk = inner.div_ceil(chunks);
        let threads = threads.expect("a worker count implies a pool");
        let base = destination.as_mut_ptr() as usize;
        // SAFETY: `destination` is a live, contiguous `outer(==1) * width *
        // inner` f32 buffer owned by this call, and `base` is its start. Each
        // worker reconstructs the whole slice and writes only the indices
        // `p * inner + index` for `index` in its own disjoint column range, so
        // no two workers write the same element and none reads what another
        // writes. `source` is a different buffer (the executor hands a segment
        // distinct input and output windows), so it is read concurrently
        // without aliasing. The pointer outlives the pool call because
        // `destination` is borrowed for the whole function.
        threads.run_blocks(chunks, |worker| {
            let destination =
                unsafe { std::slice::from_raw_parts_mut(base as *mut f32, width * inner) };
            let start = worker * chunk;
            let end = (start + chunk).min(inner);
            for index in start..end {
                let mut maximum = f32::NEG_INFINITY;
                for position in 0..width {
                    let value = source[position * inner + index];
                    if value > maximum {
                        maximum = value;
                    }
                }
                let mut total = 0.0f32;
                for position in 0..width {
                    let at = position * inner + index;
                    let exponent = (source[at] - maximum).exp();
                    destination[at] = exponent;
                    total += exponent;
                }
                for position in 0..width {
                    let at = position * inner + index;
                    destination[at] /= total;
                }
            }
            Ok(())
        })?;
        return Ok(());
    }

    // Serial generic: outer groups, each with strided columns.
    let plane = width * inner;
    for outer_index in 0..outer {
        let base = outer_index * plane;
        for index in 0..inner {
            let mut maximum = f32::NEG_INFINITY;
            for position in 0..width {
                let value = source[base + position * inner + index];
                if value > maximum {
                    maximum = value;
                }
            }
            let mut total = 0.0f32;
            for position in 0..width {
                let at = base + position * inner + index;
                let exponent = (source[at] - maximum).exp();
                destination[at] = exponent;
                total += exponent;
            }
            for position in 0..width {
                let at = base + position * inner + index;
                destination[at] /= total;
            }
        }
    }
    let _ = elements;
    Ok(())
}

fn softmax_row(source: &[f32], destination: &mut [f32]) {
    let mut maximum = f32::NEG_INFINITY;
    for &value in source {
        if value > maximum {
            maximum = value;
        }
    }
    let mut total = 0.0f32;
    for (slot, &value) in destination.iter_mut().zip(source.iter()) {
        let exponent = (value - maximum).exp();
        *slot = exponent;
        total += exponent;
    }
    for slot in destination.iter_mut() {
        *slot /= total;
    }
}

fn normalize(axis: i64, rank: usize) -> Result<usize, Fn3Error> {
    let rank_i64 = i64::try_from(rank).map_err(|_| Fn3Error::ShapeOverflow {
        context: "softmax rank",
        axis: None,
        detail: "rank does not fit in i64".to_string(),
    })?;
    let normalized = if axis < 0 { axis + rank_i64 } else { axis };
    if normalized < 0 || normalized >= rank_i64 {
        return Err(Fn3Error::InvalidShape {
            context: "softmax axis",
            detail: format!("axis {axis} is outside a rank-{rank} value"),
        });
    }
    Ok(normalized as usize)
}

fn product(extents: &[u64]) -> Result<usize, Fn3Error> {
    let mut total: usize = 1;
    for (position, &extent) in extents.iter().enumerate() {
        let extent = usize::try_from(extent).map_err(|_| Fn3Error::ShapeOverflow {
            context: "softmax extents",
            axis: Some(position),
            detail: format!("extent {extent} does not fit in usize"),
        })?;
        total = total.checked_mul(extent).ok_or(Fn3Error::ShapeOverflow {
            context: "softmax extents",
            axis: Some(position),
            detail: format!("{total} x {extent}"),
        })?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::device::DeviceId;
    use crate::inference::dtype::DType;
    use crate::inference::shape::Shape;
    use crate::inference::tensor::TensorMeta;

    fn meta(dims: &[u64]) -> TensorMeta {
        TensorMeta::contiguous(
            Shape::from_dims(dims.iter().copied()),
            DType::F32,
            DeviceId::Cpu,
        )
        .expect("shape is valid")
    }

    #[test]
    fn the_mid_axis_pooled_path_matches_the_serial_loop() {
        let dims = [1u64, 4, 1000];
        let values: Vec<f32> = (0..4000).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut serial = vec![0.0f32; values.len()];
        let mut pooled = vec![0.0f32; values.len()];
        // A build without `parallel` has no pool: the same comparison then
        // runs one worker, which is still the documented serial order.
        let workers = if cfg!(feature = "parallel") { 4 } else { 1 };
        for (target, threads) in [
            (&mut serial, None),
            (
                &mut pooled,
                Some(
                    SessionThreads::new(crate::inference::runtime::Threading::Fixed(workers))
                        .unwrap(),
                ),
            ),
        ] {
            let view =
                TensorView::from_host_bytes(bytemuck::cast_slice(&values), meta(&dims)).unwrap();
            let mut mut_view =
                TensorMut::from_host_bytes(bytemuck::cast_slice_mut(target), meta(&dims)).unwrap();
            run(
                OpId::Softmax,
                &view,
                &mut mut_view,
                &OpParams::new().with_axes([1]),
                threads.as_ref(),
            )
            .unwrap();
        }
        assert_eq!(
            serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            pooled.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
