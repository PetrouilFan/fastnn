//! `TopK`: the largest (or smallest) `k` values along one axis, and their
//! indices.
//!
//! # Order and ties
//!
//! The values are emitted in `largest` order: descending for `largest = true`,
//! ascending otherwise. That is a superset of what ONNX requires — with
//! `sorted = false` any order is legal, and emitting the sorted order makes the
//! kernel's output deterministic without a second code path.
//!
//! A tie is broken by the **lower index**, which is what a stable descending
//! selection gives and what the reference exporter's consumers expect. The
//! selection is a `k`-pass scan (`O(n * k)`), never a comparison sort, so the tie
//! rule is explicit rather than inherited from a sort implementation. A `NaN`
//! never displaces a non-`NaN`, because every comparison with `NaN` is false; a
//! line made only of `NaN`s returns them in index order.
//!
//! # Shape
//!
//! `k` comes from `OpParams::topk_k` (the importer folds the constant `k`
//! operand) or, when the node still carries the operand, from that `i64` scalar.
//! The values output has the input's shape with `axis` replaced by `k`; the
//! indices output is `i64` with the same shape.
//!
//! The selection is serial. TopK is not on the CNN path, and a parallel
//! selection would need a merge step whose tie rule would have to be proved to
//! match this one; a scalar reference that is obviously right is worth more here
//! than the microseconds.

use crate::inference::backend::OpParams;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::runtime::kernels::{
    extents, flexible_arity, input_f32, product, require_dtype,
};
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

/// Execute `TopK` from views.
pub(crate) fn run(
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let op = OpId::TopK;
    flexible_arity(op, inputs.len(), outputs.len(), &[1, 2], 2)?;
    let x = input_f32(op, &inputs[0])?;
    let x_shape = inputs[0].shape().clone();
    let k = match inputs.get(1) {
        Some(view) => {
            require_dtype(op, view, DType::I64)?;
            let values = view.as_slice::<i64>()?;
            match values {
                [k] => *k,
                other => {
                    return Err(Fn3Error::InvalidShape {
                        context: "runtime top_k",
                        detail: format!(
                            "op top_k needs a scalar i64 k input, found {} values",
                            other.len()
                        ),
                    })
                }
            }
        }
        None => params.topk_k,
    };
    let values_shape = outputs[0].shape().clone();
    let indices_shape = outputs[1].shape().clone();
    {
        let values = outputs[0].meta();
        values.require_dtype(Some(op), DType::F32)?;
    }
    {
        let indices = outputs[1].meta();
        indices.require_dtype(Some(op), DType::I64)?;
    }
    let (values_output, indices_output) = outputs.split_at_mut(1);
    let values = values_output[0].as_mut_slice::<f32>()?;
    let indices = indices_output[0].as_mut_slice::<i64>()?;
    topk_f32(
        x,
        &x_shape,
        values,
        &values_shape,
        indices,
        &indices_shape,
        params,
        k,
    )
}

/// The kernel: contiguous operands, explicit shapes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn topk_f32(
    x: &[f32],
    x_shape: &Shape,
    values: &mut [f32],
    values_shape: &Shape,
    indices: &mut [i64],
    indices_shape: &Shape,
    params: &OpParams,
    k: i64,
) -> Result<(), Fn3Error> {
    let op = OpId::TopK;
    let rank = x_shape.rank();
    if rank == 0 {
        return Err(Fn3Error::InvalidShape {
            context: "runtime top_k",
            detail: "op top_k needs a rank-1 or higher operand to name an axis".to_string(),
        });
    }
    let axis = normalize(params.topk_axis, rank)?;
    if k < 0 {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: "fn3-runtime-kernels",
            reason: "TopK needs a constant k; pass it in OpParams::topk_k or as an i64 input"
                .to_string(),
        });
    }
    let dims = extents(x_shape, "runtime top_k input")?;
    let n = dims[axis];
    let k = usize::try_from(k).map_err(|_| Fn3Error::ShapeOverflow {
        context: "runtime top_k",
        axis: Some(axis),
        detail: format!("k = {k} does not fit in usize"),
    })?;
    if k > n {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("k <= the axis {axis} extent {n}"),
            actual: format!("k = {k}"),
        });
    }
    let mut expected = dims.clone();
    expected[axis] = k;
    let out_shape = Shape::from_dims(expected.iter().map(|&extent| extent as u64));
    if values_shape != &out_shape || indices_shape != &out_shape {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("two outputs of shape {out_shape}"),
            actual: format!("values {values_shape}, indices {indices_shape}"),
        });
    }
    let elements = product(&dims, "runtime top_k input")?;
    if x.len() != elements {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{elements} input elements for {x_shape}"),
            actual: format!("{} elements", x.len()),
        });
    }
    let outer = product(&dims[..axis], "runtime top_k outer")?;
    let inner = product(&dims[axis + 1..], "runtime top_k inner")?;
    let out_elements = outer * k * inner;
    if values.len() != out_elements || indices.len() != out_elements {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{out_elements} output elements for {out_shape}"),
            actual: format!("{} values, {} indices", values.len(), indices.len()),
        });
    }
    if out_elements == 0 {
        return Ok(());
    }

    let largest = params.largest;
    let mut used = vec![false; n];
    for line in 0..outer * inner {
        let o = line / inner.max(1);
        let i = line % inner.max(1);
        for flag in used.iter_mut() {
            *flag = false;
        }
        for slot in 0..k {
            let mut best: Option<(usize, f32)> = None;
            for a in 0..n {
                if used[a] {
                    continue;
                }
                let candidate = x[o * n * inner + a * inner + i];
                let better = match best {
                    None => true,
                    // A NaN is unordered against everything, so a plain `>`
                    // would let a selected NaN block every later number. Keep
                    // numbers ahead of NaNs explicitly; among two numbers the
                    // comparison decides, and the first index wins a tie.
                    Some((_, current)) if current.is_nan() && !candidate.is_nan() => true,
                    Some((_, current)) if !current.is_nan() && candidate.is_nan() => false,
                    Some((_, current)) => {
                        if largest {
                            candidate > current
                        } else {
                            candidate < current
                        }
                    }
                };
                if better {
                    best = Some((a, candidate));
                }
            }
            let (a, value) = best.ok_or_else(|| Fn3Error::InvalidShape {
                context: "runtime top_k",
                detail: format!(
                    "internal: k = {k} passed validation but a line has only {n} values"
                ),
            })?;
            used[a] = true;
            values[o * k * inner + slot * inner + i] = value;
            indices[o * k * inner + slot * inner + i] = a as i64;
        }
    }
    Ok(())
}

/// Normalise a signed axis, rejecting out-of-range values.
fn normalize(axis: i64, rank: usize) -> Result<usize, Fn3Error> {
    let resolved = if axis < 0 { axis + rank as i64 } else { axis };
    if resolved < 0 || resolved >= rank as i64 {
        return Err(Fn3Error::InvalidShape {
            context: "runtime top_k",
            detail: format!(
                "op top_k axis {axis} is outside a rank-{rank} operand; legal axes are -{rank}..={}",
                rank.saturating_sub(1)
            ),
        });
    }
    Ok(resolved as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topk(x: &[f32], dims: &[u64], axis: i64, k: i64, largest: bool) -> (Vec<f32>, Vec<i64>) {
        let shape = Shape::from_dims(dims.iter().copied());
        let mut out_dims = dims.to_vec();
        let resolved = if axis < 0 {
            (dims.len() as i64 + axis) as usize
        } else {
            axis as usize
        };
        out_dims[resolved] = k as u64;
        let out_shape = Shape::from_dims(out_dims.iter().copied());
        let len: u64 = out_dims.iter().product();
        let mut values = vec![0.0f32; len as usize];
        let mut indices = vec![0i64; len as usize];
        let params = OpParams::new().with_topk(axis, k, largest, true);
        topk_f32(
            x,
            &shape,
            &mut values,
            &out_shape,
            &mut indices,
            &out_shape,
            &params,
            k,
        )
        .unwrap();
        (values, indices)
    }

    #[test]
    fn largest_two_of_a_row_are_sorted_and_tie_on_the_lower_index() {
        let (values, indices) = topk(&[1.0, 5.0, 3.0, 5.0], &[4], 0, 2, true);
        assert_eq!(values, [5.0, 5.0]);
        assert_eq!(indices, [1, 3], "a tie keeps the lower index first");
    }

    #[test]
    fn smallest_two_and_the_axis_is_a_middle_one() {
        // [2, 3] with axis 1, k = 1, smallest: rows [3, 1, 2] and [0, 9, 4].
        let x = [3.0, 1.0, 2.0, 0.0, 9.0, 4.0];
        let (values, indices) = topk(&x, &[2, 3], 1, 1, false);
        assert_eq!(values, [1.0, 0.0]);
        assert_eq!(indices, [1, 0]);
        let (values, indices) = topk(&x, &[2, 3], -1, 2, true);
        assert_eq!(values, [3.0, 2.0, 9.0, 4.0]);
        assert_eq!(indices, [0, 2, 1, 2]);
    }

    #[test]
    fn a_nan_never_displaces_a_number() {
        let (values, indices) = topk(&[f32::NAN, 2.0, 1.0], &[3], 0, 2, true);
        assert_eq!(values, [2.0, 1.0]);
        assert_eq!(indices, [1, 2]);
    }

    #[test]
    fn k_above_the_extent_is_rejected() {
        let shape = Shape::from_dims([3]);
        let out_shape = Shape::from_dims([4]);
        let mut values = [0.0f32; 4];
        let mut indices = [0i64; 4];
        let params = OpParams::new().with_topk(0, 4, true, true);
        let error = topk_f32(
            &[1.0, 2.0, 3.0],
            &shape,
            &mut values,
            &out_shape,
            &mut indices,
            &out_shape,
            &params,
            4,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("k <= the axis 0 extent 3"),
            "{error}"
        );
    }
}
