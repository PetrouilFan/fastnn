//! `Concat`: byte-level data movement along one axis.
//!
//! Concatenation is a copy, not arithmetic: the element type never enters the
//! loop and the result is bit-identical however the destination is distributed.
//! For every outer index and every input, the input's contribution to the
//! output is **one contiguous run** (`axis_len * inner` elements), so the
//! kernel builds a piece list and copies runs, not elements. Parallelism
//! splits the *destination byte range*, so each worker owns a disjoint slice of
//! the output and copies the intersections of the runs with it.
//!
//! # Why it is here
//!
//! The legacy kernel walked one element at a time through a locator and ran on
//! one thread. `Concat` is the largest scalar segment of the pinned detectors
//! (17 calls per forward), and this kernel is both faster serially and able to
//! use the session's pool.

use crate::inference::backend::OpParams;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::runtime::SessionThreads;
use crate::inference::tensor::{TensorMut, TensorView};

/// Execute `Concat(inputs..., axis)`.
pub(crate) fn concat(
    op: OpId,
    inputs: &[TensorView<'_>],
    out: &mut TensorMut<'_>,
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let first = inputs.first().ok_or_else(|| Fn3Error::InvalidShape {
        context: "concat inputs",
        detail: format!("op {op} needs at least one input"),
    })?;
    let rank = first.shape().rank();
    if rank == 0 {
        return Err(Fn3Error::InvalidShape {
            context: "concat",
            detail: "a rank-0 value has no axis to concatenate".to_string(),
        });
    }
    let axis = match params.axes.as_slice() {
        [axis] => normalize(*axis, rank)?,
        [] => 0,
        _ => {
            return Err(Fn3Error::InvalidShape {
                context: "concat axis",
                detail: format!("expected one axis, got {:?}", params.axes),
            })
        }
    };
    for view in inputs {
        if view.shape().rank() != rank {
            return Err(Fn3Error::RankMismatch {
                op: Some(op),
                expected: rank,
                actual: view.shape().rank(),
            });
        }
        if view.dtype() != first.dtype() {
            return Err(Fn3Error::DTypeMismatch {
                op: Some(op),
                expected: first.dtype(),
                actual: view.dtype(),
            });
        }
        for (position, (&extent, &reference)) in view
            .shape()
            .dims()
            .iter()
            .zip(first.shape().dims())
            .enumerate()
        {
            if position != axis && extent != reference {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!(
                        "axis {position} to be {reference} on every input, as on {:?}",
                        first.shape()
                    ),
                    actual: format!("{extent} on an input"),
                });
            }
        }
        view.meta().require_contiguous("concat input")?;
    }
    out.meta().require_contiguous("concat output")?;
    if out.dtype() != first.dtype() {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: first.dtype(),
            actual: out.dtype(),
        });
    }

    let dims = first.shape().dims();
    let mut expected = dims.to_vec();
    expected[axis] = 0;
    for view in inputs {
        let Some(total) = expected[axis].checked_add(view.shape().dims()[axis]) else {
            return Err(Fn3Error::ShapeOverflow {
                context: "concat output extent",
                axis: Some(axis),
                detail: format!("{:?} plus {}", expected[axis], view.shape().dims()[axis]),
            });
        };
        expected[axis] = total;
    }
    if out.shape().dims() != expected.as_slice() {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("the concatenated shape {expected:?}"),
            actual: format!("{:?}", out.shape()),
        });
    }

    let element = first.dtype().size_bytes();
    let inner = product(&dims[axis + 1..], "concat inner")?;
    let outer = product(&dims[..axis], "concat outer")?;
    let out_axis =
        usize::try_from(out.shape().dims()[axis]).map_err(|_| Fn3Error::ShapeOverflow {
            context: "concat output axis",
            axis: Some(axis),
            detail: "extent does not fit in usize".to_string(),
        })?;
    let row_bytes = out_axis
        .checked_mul(inner)
        .and_then(|elements| elements.checked_mul(element))
        .ok_or(Fn3Error::ShapeOverflow {
            context: "concat row bytes",
            axis: Some(axis),
            detail: "row length overflows usize".to_string(),
        })?;
    if row_bytes == 0 || outer == 0 {
        return Ok(());
    }
    let total_bytes = row_bytes
        .checked_mul(outer)
        .ok_or(Fn3Error::ShapeOverflow {
            context: "concat total bytes",
            axis: Some(axis),
            detail: "output overflows usize".to_string(),
        })?;

    let mut sources = Vec::with_capacity(inputs.len());
    let mut axis_lens = Vec::with_capacity(inputs.len());
    for view in inputs {
        sources.push(view.host_bytes()?);
        axis_lens.push(usize::try_from(view.shape().dims()[axis]).map_err(|_| {
            Fn3Error::ShapeOverflow {
                context: "concat input axis",
                axis: Some(axis),
                detail: "extent does not fit in usize".to_string(),
            }
        })?);
    }
    let destination = out.host_bytes_mut()?;
    if destination.len() < total_bytes {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{total_bytes} output bytes"),
            actual: format!("{}", destination.len()),
        });
    }

    // One piece per (outer index, input): a contiguous source run and its
    // contiguous destination run. Runs inside one outer block are adjacent, so
    // the destination offsets are cumulative along the axis.
    struct Piece {
        source: usize,
        from: usize,
        to: usize,
        len: usize,
    }
    let mut pieces = Vec::with_capacity(outer * inputs.len());
    for outer_index in 0..outer {
        let mut axis_offset = 0usize;
        for (source, &axis_len) in axis_lens.iter().enumerate() {
            let run = axis_len * inner * element;
            pieces.push(Piece {
                source,
                from: outer_index * axis_len * inner * element,
                to: outer_index * row_bytes + axis_offset,
                len: run,
            });
            axis_offset += run;
        }
    }

    let workers = threads.map(SessionThreads::worker_count).unwrap_or(1);
    let elements = inner * out_axis * outer;
    let use_pool = workers > 1
        && elements >= super::MIN_PARALLEL_ELEMENTS
        && elements / workers >= super::MIN_PARALLEL_ELEMENTS_PER_WORKER;

    if use_pool {
        let threads = threads.expect("a worker count implies a pool");
        let chunks = workers.min(total_bytes.max(1));
        let chunk = total_bytes.div_ceil(chunks);
        let base = destination.as_mut_ptr() as usize;
        // SAFETY: `destination` is a live, contiguous `total_bytes` buffer
        // owned by this call; `base` is its start. Worker `w` writes exactly
        // `[w*chunk, min((w+1)*chunk, total_bytes))` — disjoint ranges — and
        // each piece it copies lies inside that range by the intersection
        // test. `sources` are separate buffers (the executor hands a segment
        // distinct input and output windows) and are read-only here, so no
        // worker aliases another's writes. The pointer outlives the pool call
        // because `destination` is borrowed for the whole function.
        threads.run_blocks(chunks, |worker| {
            let start = worker * chunk;
            let end = (start + chunk).min(total_bytes);
            if start >= end {
                return Ok(());
            }
            let destination =
                unsafe { std::slice::from_raw_parts_mut(base as *mut u8, total_bytes) };
            for piece in &pieces {
                let piece_end = piece.to + piece.len;
                let from = piece.to.max(start);
                let to = piece_end.min(end);
                if from >= to {
                    continue;
                }
                let offset = from - piece.to;
                destination[from..to].copy_from_slice(
                    &sources[piece.source][piece.from + offset..piece.from + offset + (to - from)],
                );
            }
            Ok(())
        })?;
        return Ok(());
    }

    for piece in &pieces {
        destination[piece.to..piece.to + piece.len]
            .copy_from_slice(&sources[piece.source][piece.from..piece.from + piece.len]);
    }
    Ok(())
}

fn normalize(axis: i64, rank: usize) -> Result<usize, Fn3Error> {
    let rank_i64 = i64::try_from(rank).map_err(|_| Fn3Error::ShapeOverflow {
        context: "concat rank",
        axis: None,
        detail: "rank does not fit in i64".to_string(),
    })?;
    let normalized = if axis < 0 { axis + rank_i64 } else { axis };
    if normalized < 0 || normalized >= rank_i64 {
        return Err(Fn3Error::InvalidShape {
            context: "concat axis",
            detail: format!("axis {axis} is outside a rank-{rank} value"),
        });
    }
    Ok(normalized as usize)
}

fn product(extents: &[u64], what: &'static str) -> Result<usize, Fn3Error> {
    let mut total: usize = 1;
    for (position, &extent) in extents.iter().enumerate() {
        let extent = usize::try_from(extent).map_err(|_| Fn3Error::ShapeOverflow {
            context: what,
            axis: Some(position),
            detail: format!("extent {extent} does not fit in usize"),
        })?;
        total = total.checked_mul(extent).ok_or(Fn3Error::ShapeOverflow {
            context: what,
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
    fn concat_copies_runs_along_the_axis() {
        // Two [1, 2, 3] inputs on axis 1 -> [1, 4, 3]; each plane keeps order.
        let left: Vec<f32> = (0..6).map(|value| value as f32).collect();
        let right: Vec<f32> = (10..16).map(|value| value as f32).collect();
        let mut out = vec![0.0f32; 12];
        let mut out_mut =
            TensorMut::from_host_bytes(bytemuck::cast_slice_mut(&mut out), meta(&[1, 4, 3]))
                .unwrap();
        let left_view =
            TensorView::from_host_bytes(bytemuck::cast_slice(&left), meta(&[1, 2, 3])).unwrap();
        let right_view =
            TensorView::from_host_bytes(bytemuck::cast_slice(&right), meta(&[1, 2, 3])).unwrap();
        let params = OpParams::new().with_axes([1]);
        concat(
            OpId::Concat,
            &[left_view, right_view],
            &mut out_mut,
            &params,
            None,
        )
        .unwrap();
        assert_eq!(
            out,
            vec![0., 1., 2., 3., 4., 5., 10., 11., 12., 13., 14., 15.]
        );
    }
}
