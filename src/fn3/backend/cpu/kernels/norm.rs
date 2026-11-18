//! Layer normalization over the trailing axis.
//!
//! # Contract
//!
//! ```text
//! mean     = sum(x[a]) / n                     (ascending a, f32)
//! variance = sum((x[a] - mean)^2) / n          (ascending a, f32)
//! out[a]   = (x[a] - mean) / sqrt(variance + eps) * weight[a] + bias[a]
//! ```
//!
//! This is the exact expression the parity fixture evaluates in `float64`
//! (`tools/parity/generate_fixtures.py::layernorm64`) and the exact expression
//! the pinned GPT-2 graph decomposes into
//! `ReduceMean → Sub → Pow(2) → ReduceMean → Add(eps) → Sqrt → Div → Mul(w) → Add(b)`,
//! which is why the kernel *divides* by `sqrt(variance + eps)` instead of
//! multiplying by a reciprocal: the two differ in the last bits and the
//! decomposition is the reference.
//!
//! # Scope
//!
//! The normalized axes are the trailing axis (`axes` empty, or naming the last
//! axis). `weight` and `bias` are rank-1 `[hidden]`. ONNX also allows
//! normalizing a trailing *block* of axes with multi-dimensional
//! `weight`/`bias`; that form is not used by the pinned graph and is rejected
//! with `UnsupportedOp` naming FN3-07 rather than silently normalizing the wrong
//! extent.
//!
//! # Epsilon
//!
//! `OpParams::epsilon` is required to be finite and non-negative; the importer
//! must supply the model's `1e-5`. A zero epsilon is legal (it is the
//! `OpParams` default) and simply removes the numerical floor.

use crate::fn3::backend::OpParams;
use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;
use crate::fn3::tensor::{TensorMut, TensorView};

use super::{
    at, at_mut, checked_product, decompose, normalize_axis, require_dtype, require_output_shape,
    Locator,
};

/// Read a rank-1 `f32` operand of extent `extent` into a vector.
fn read_vector(
    op: OpId,
    view: &TensorView<'_>,
    extent: u64,
    what: &'static str,
) -> Result<Vec<f32>, Fn3Error> {
    require_dtype(op, view.meta(), DType::F32)?;
    if view.shape().rank() != 1 || view.shape().dims()[0] != extent {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("[{extent}] for {what}"),
            actual: view.shape().to_string(),
        });
    }
    let source = view.storage_slice::<f32>()?;
    let mut values = Vec::with_capacity(extent as usize);
    for axis in 0..extent {
        let index = view
            .meta()
            .checked_index(&[axis])
            .map_err(|error| error.with_op(op))?;
        values.push(at(op, source, index)?);
    }
    Ok(values)
}

/// Layer normalization over the trailing axis.
pub(crate) fn layer_norm(
    op: OpId,
    input: &TensorView<'_>,
    weight: &TensorView<'_>,
    bias: &TensorView<'_>,
    out: &mut TensorMut<'_>,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    require_dtype(op, input.meta(), DType::F32)?;
    require_dtype(op, out.meta(), DType::F32)?;
    let rank = input.shape().rank();
    if rank == 0 {
        return Err(Fn3Error::RankMismatch {
            op: Some(op),
            expected: 1,
            actual: 0,
        });
    }
    if !params.epsilon.is_finite() || params.epsilon < 0.0 {
        return Err(Fn3Error::InvalidShape {
            context: "layer_norm epsilon",
            detail: format!(
                "op {op}: epsilon {} must be finite and non-negative",
                params.epsilon
            ),
        });
    }
    if let [axis] = params.axes.as_slice() {
        let axis = normalize_axis(op, *axis, rank, "layer_norm axes")?;
        if axis != rank - 1 {
            return Err(Fn3Error::UnsupportedOp {
                op,
                backend: crate::fn3::backend::cpu::CPU_BACKEND_NAME,
                reason: format!(
                    "the fn3 LayerNorm contract normalizes the trailing axis only; axis {axis} \
                     of {} was requested (owner: FN3-07 / #53)",
                    input.shape()
                ),
            });
        }
    } else if params.axes.len() > 1 {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: crate::fn3::backend::cpu::CPU_BACKEND_NAME,
            reason: format!(
                "the fn3 LayerNorm contract normalizes the trailing axis only; axes {:?} were \
                 requested (owner: FN3-07 / #53)",
                params.axes
            ),
        });
    }
    require_output_shape(op, input.shape(), out)?;

    let hidden = input.shape().dims()[rank - 1];
    let weight_values = read_vector(op, weight, hidden, "layer_norm weight")?;
    let bias_values = read_vector(op, bias, hidden, "layer_norm bias")?;
    let rows = checked_product(&input.shape().dims()[..rank - 1], "cpu layer_norm rows")?;
    if rows == 0 || hidden == 0 {
        return Ok(());
    }

    let locator = Locator::new(op, input.meta(), input.shape())?;
    let out_locator = Locator::new(op, out.meta(), input.shape())?;
    let source = input.storage_slice::<f32>()?;
    let destination = out.storage_slice_mut::<f32>()?;
    let leading_dims = input.shape().dims()[..rank - 1].to_vec();
    let mut coordinates = vec![0u64; rank];
    let hidden_f32 = hidden as f32;

    for row in 0..rows {
        for coordinate in coordinates.iter_mut().take(rank - 1) {
            *coordinate = 0;
        }
        if !leading_dims.is_empty() {
            decompose(row, &leading_dims, &mut coordinates[..rank - 1]);
        }

        let mut sum = 0.0f32;
        for a in 0..hidden {
            coordinates[rank - 1] = a;
            sum += at(op, source, locator.index(op, &coordinates)?)?;
        }
        let mean = sum / hidden_f32;

        let mut variance_sum = 0.0f32;
        for a in 0..hidden {
            coordinates[rank - 1] = a;
            let value = at(op, source, locator.index(op, &coordinates)?)?;
            let centred = value - mean;
            variance_sum += centred * centred;
        }
        let variance = variance_sum / hidden_f32;
        let deviation = (variance + params.epsilon).sqrt();

        for a in 0..hidden {
            coordinates[rank - 1] = a;
            let index = locator.index(op, &coordinates)?;
            let value = at(op, source, index)?;
            let position = usize::try_from(a).map_err(|_| Fn3Error::ShapeOverflow {
                context: "cpu layer_norm position",
                axis: Some(rank - 1),
                detail: "hidden position does not fit in usize".to_string(),
            })?;
            let scale = at(op, &weight_values, position)?;
            let shift = at(op, &bias_values, position)?;
            let normalised = (value - mean) / deviation;
            let out_index = out_locator.index(op, &coordinates)?;
            *at_mut(op, destination, out_index)? = normalised * scale + shift;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epsilon_must_be_usable() {
        let input = crate::fn3::tensor::Tensor::from_f32(
            crate::fn3::shape::Shape::from_dims([1, 2]),
            &[1.0, 2.0],
        )
        .unwrap();
        let weight = crate::fn3::tensor::Tensor::from_f32(
            crate::fn3::shape::Shape::from_dims([2]),
            &[1.0, 1.0],
        )
        .unwrap();
        let bias = crate::fn3::tensor::Tensor::from_f32(
            crate::fn3::shape::Shape::from_dims([2]),
            &[0.0, 0.0],
        )
        .unwrap();
        let mut out = crate::fn3::tensor::Tensor::zeros(
            crate::fn3::shape::Shape::from_dims([1, 2]),
            DType::F32,
            crate::fn3::device::DeviceId::Cpu,
        )
        .unwrap();
        let input_view = input.as_view().unwrap();
        let weight_view = weight.as_view().unwrap();
        let bias_view = bias.as_view().unwrap();
        let mut out_view = out.as_mut_view().unwrap();
        let error = layer_norm(
            OpId::LayerNorm,
            &input_view,
            &weight_view,
            &bias_view,
            &mut out_view,
            &OpParams::default().with_epsilon(f32::NAN),
        )
        .unwrap_err();
        assert!(error.to_string().contains("epsilon"), "{error}");
    }
}
