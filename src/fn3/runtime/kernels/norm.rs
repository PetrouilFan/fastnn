//! Inference-mode `BatchNorm`.
//!
//! ONNX's `BatchNormalization` in inference form normalises every channel with
//! the four 1-D statistics the model carries:
//!
//! ```text
//! y[n, c, ...] = (x[n, c, ...] - mean[c]) / sqrt(var[c] + epsilon) * scale[c] + B[c]
//! ```
//!
//! The kernel evaluates `1 / sqrt(var[c] + epsilon)` **once per channel row**, so
//! every element of a row is scaled by the same `f32` value and the result
//! cannot depend on the order the rows ran in. `epsilon` must be finite and
//! non-negative, which is the same contract the shape pass enforces; a NaN or a
//! negative epsilon would make the square root produce a NaN for every element,
//! so it is rejected rather than propagated.

use crate::inference::backend::OpParams;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::runtime::kernels::{
    arity, check_input_len, check_output_len, extents, for_each_row, input_f32, output_f32,
    product, require_dtype,
};
use crate::inference::runtime::SessionThreads;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

/// Execute `BatchNorm(X, scale, B, mean, var)` from views.
pub(crate) fn run(
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let op = OpId::BatchNorm;
    arity(op, inputs.len(), outputs.len(), 5, 1)?;
    let x = input_f32(op, &inputs[0])?;
    let scale = input_f32(op, &inputs[1])?;
    let bias = input_f32(op, &inputs[2])?;
    let mean = input_f32(op, &inputs[3])?;
    let variance = input_f32(op, &inputs[4])?;
    let x_shape = inputs[0].shape().clone();
    for (position, name) in ["scale", "B", "mean", "var"].iter().enumerate() {
        let view = &inputs[position + 1];
        require_dtype(op, view, crate::inference::dtype::DType::F32)?;
        if view.shape().rank() != 1 {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(op),
                expected: format!("a rank-1 {name}"),
                actual: format!("{}", view.shape()),
            });
        }
    }
    let out_shape = outputs[0].shape().clone();
    let out = output_f32(op, &mut outputs[0])?;
    batch_norm_f32(
        x,
        &x_shape,
        scale,
        bias,
        mean,
        variance,
        out,
        &out_shape,
        params.epsilon,
        None,
    )
}

/// The kernel: contiguous operands, explicit shapes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_norm_f32(
    x: &[f32],
    x_shape: &Shape,
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    variance: &[f32],
    out: &mut [f32],
    out_shape: &Shape,
    epsilon: f32,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let op = OpId::BatchNorm;
    let rank = x_shape.rank();
    if rank < 2 {
        return Err(Fn3Error::InvalidShape {
            context: "runtime batch_norm",
            detail: format!(
                "op {op} needs an [N, C, ...] operand of rank 2 or more, found {x_shape}"
            ),
        });
    }
    if out_shape != x_shape {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("the input shape {x_shape}"),
            actual: format!("{out_shape}"),
        });
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(Fn3Error::InvalidShape {
            context: "runtime batch_norm",
            detail: format!("op {op} epsilon {epsilon} must be finite and non-negative"),
        });
    }
    let dims = extents(x_shape, "runtime batch_norm input")?;
    let (batch, channels) = (dims[0], dims[1]);
    for (values, name) in [
        (scale, "scale"),
        (bias, "B"),
        (mean, "mean"),
        (variance, "var"),
    ] {
        if values.len() != channels {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(op),
                expected: format!("{channels} {name} values for {channels} channels"),
                actual: format!("{}", values.len()),
            });
        }
    }
    let elements = product(&dims, "runtime batch_norm input")?;
    check_input_len(op, x, elements, x_shape, "input")?;
    check_output_len(op, out, elements, out_shape)?;
    if elements == 0 {
        return Ok(());
    }
    let row_len = elements / (batch * channels);
    let channels_for_row = channels;

    for_each_row(threads, out, row_len, |row, values| {
        let channel = row % channels_for_row;
        let inverse = 1.0f32 / (variance[channel] + epsilon).sqrt();
        let scale = scale[channel] * inverse;
        let shift = bias[channel] - mean[channel] * scale;
        let source = &x[row * row_len..row * row_len + row_len];
        for (slot, &value) in values.iter_mut().zip(source.iter()) {
            *slot = value * scale + shift;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_normalised_value_is_scale_times_centred_value_plus_shift() {
        // Two channels of two values each. Channel 0: mean 1, var 1, scale 2,
        // B 0.5 -> x=3 gives (3-1)/1*2+0.5 = 4.5. Channel 1: mean 4, var 4,
        // scale 1, B -1 -> x=8 gives (8-4)/2*1-1 = 1.
        let x = [3.0f32, -1.0, 8.0, 0.0];
        let mut out = [0.0f32; 4];
        batch_norm_f32(
            &x,
            &Shape::from_dims([1, 2, 2]),
            &[2.0, 1.0],
            &[0.5, -1.0],
            &[1.0, 4.0],
            &[1.0, 4.0],
            &mut out,
            &Shape::from_dims([1, 2, 2]),
            0.0,
            None,
        )
        .unwrap();
        assert!((out[0] - 4.5).abs() < 1e-6, "{out:?}");
        assert!((out[1] - (-3.5)).abs() < 1e-6, "{out:?}");
        assert!((out[2] - 1.0).abs() < 1e-6, "{out:?}");
        assert!((out[3] - (-3.0)).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn epsilon_enters_the_variance_not_the_mean() {
        // var = 0, epsilon = 1: 1/sqrt(1) = 1, so the output is x - mean.
        let x = [5.0f32, 7.0];
        let mut out = [0.0f32; 2];
        batch_norm_f32(
            &x,
            &Shape::from_dims([2, 1]),
            &[1.0],
            &[0.0],
            &[4.0],
            &[0.0],
            &mut out,
            &Shape::from_dims([2, 1]),
            1.0,
            None,
        )
        .unwrap();
        assert_eq!(out, [1.0, 3.0]);
    }

    #[test]
    fn a_negative_epsilon_and_a_wrong_bias_length_are_rejected() {
        let x = [1.0f32, 2.0];
        let mut out = [0.0f32; 2];
        let error = batch_norm_f32(
            &x,
            &Shape::from_dims([1, 2]),
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &mut out,
            &Shape::from_dims([1, 2]),
            -1.0,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("epsilon"), "{error}");

        let error = batch_norm_f32(
            &x,
            &Shape::from_dims([1, 2]),
            &[1.0, 1.0],
            &[0.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &mut out,
            &Shape::from_dims([1, 2]),
            0.0,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("B values"), "{error}");
    }
}
