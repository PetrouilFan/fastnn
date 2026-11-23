//! The scalar reference for a fused segment.
//!
//! # Why this exists
//!
//! [`SegmentOp::Fused`] is a plan decision, not a promise about a provider. The
//! portable scalar engine implements every fused form, so:
//!
//! - a build without `cpu-onednn` still runs a fused plan (there is no feature
//!   test in the planner, and a fused segment is ordinary plan data);
//! - a provider that declines the epilogue leaves the segment scalar rather
//!   than forcing the planner to re-expand it;
//! - the fused and unfused scalar results are **bit-identical**, because this
//!   module evaluates exactly the same expressions in exactly the same order as
//!   the unfused kernels it replaces.
//!
//! # Arithmetic order
//!
//! ```text
//! anchor(inputs)      conv: ascending channel then kernel order, bias last
//! out += inputs[sum]  one element at a time, in place
//! out = activation(out)
//! ```
//!
//! The unfused chain `conv -> add -> relu` computes the same three steps in the
//! same order, so the results agree bit for bit; `conv -> sigmoid -> mul` equals
//! `x * sigmoid(x)` evaluated per element. `runtime::fuse`'s tests and
//! `tests/fn3_onednn.rs` pin that equality.

use crate::inference::backend::{FusedOp, OpParams};
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::runtime::kernels;
use crate::inference::runtime::SessionThreads;
use crate::inference::tensor::{TensorMut, TensorView};

/// Execute one fused segment.
pub(crate) fn run(
    fused: &FusedOp,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    match fused.anchor {
        OpId::Conv => run_conv(fused, inputs, outputs, params, threads),
        other => Err(Fn3Error::UnsupportedOp {
            op: other,
            backend: "fn3-runtime-kernels",
            reason: format!(
                "the scalar fused kernel implements a Conv anchor only, not {other} (chain {})",
                fused.name()
            ),
        }),
    }
}

fn run_conv(
    fused: &FusedOp,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let anchor_inputs = fused.sum.unwrap_or(inputs.len()).min(inputs.len());
    if outputs.len() != 1 {
        return Err(Fn3Error::ArityMismatch {
            op: OpId::Conv,
            expected_inputs: anchor_inputs,
            actual_inputs: inputs.len(),
            expected_outputs: 1,
            actual_outputs: outputs.len(),
        });
    }
    let residual = fused
        .sum
        .map(|position| {
            inputs.get(position).ok_or_else(|| Fn3Error::ArityMismatch {
                op: OpId::Add,
                expected_inputs: position + 1,
                actual_inputs: inputs.len(),
                expected_outputs: 1,
                actual_outputs: 1,
            })
        })
        .transpose()?;

    // The anchor writes the result, exactly as the unfused convolution does.
    kernels::conv::run(&inputs[..anchor_inputs], outputs, params, threads)?;

    let out_shape = outputs[0].meta().shape().clone();
    let elements = out_shape.numel_usize()?;
    let row_len = usize::try_from(*out_shape.dims().last().unwrap_or(&1))
        .unwrap_or(1)
        .max(1);

    if let Some(residual) = residual {
        let source = kernels::input_f32(OpId::Add, residual)?;
        if source.len() != elements {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(OpId::Add),
                expected: format!("{elements} elements, the fused result"),
                actual: format!("{} elements", source.len()),
            });
        }
        let out = kernels::output_f32(OpId::Add, &mut outputs[0])?;
        kernels::for_each_row(threads, out, row_len, |row, values| {
            let start = row * row_len;
            for (offset, slot) in values.iter_mut().enumerate() {
                *slot += source[start + offset];
            }
            Ok(())
        })?;
    }

    if let Some(activation) = fused.activation {
        let out = kernels::output_f32(activation.op(), &mut outputs[0])?;
        kernels::for_each_row(threads, out, row_len, |_, values| {
            for value in values.iter_mut() {
                *value = activation.apply(*value);
            }
            Ok(())
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::backend::{Activation, OpParams};
    use crate::inference::dtype::DType;
    use crate::inference::shape::Shape;
    use crate::inference::tensor::Tensor;

    fn conv_params() -> OpParams {
        OpParams::new()
            .with_kernel_shape([1, 1])
            .with_strides([1, 1])
            .with_dilations([1, 1])
            .with_group(1)
    }

    /// A `[1, 2, 2, 2]` input through a per-channel 1x1 convolution.
    fn conv_inputs() -> (Tensor, Tensor, Tensor) {
        let x = Tensor::from_f32(
            Shape::from_dims([1, 2, 2, 2]),
            &[1.0, -2.0, 0.5, -0.25, 0.0, 3.0, -1.5, 2.0],
        )
        .unwrap();
        let w = Tensor::from_f32(Shape::from_dims([2, 2, 1, 1]), &[2.0, -0.5, 0.25, 0.5]).unwrap();
        let b = Tensor::from_f32(Shape::from_dims([2]), &[0.5, -0.25]).unwrap();
        (x, w, b)
    }

    fn fused(output: &mut Tensor, activation: Option<Activation>, residual: Option<&Tensor>) {
        let (x, w, b) = conv_inputs();
        let views: Vec<TensorView<'_>> = match residual {
            Some(residual) => vec![
                x.as_view().unwrap(),
                w.as_view().unwrap(),
                b.as_view().unwrap(),
                residual.as_view().unwrap(),
            ],
            None => vec![
                x.as_view().unwrap(),
                w.as_view().unwrap(),
                b.as_view().unwrap(),
            ],
        };
        let fused = FusedOp {
            anchor: OpId::Conv,
            activation,
            sum: residual.map(|_| 3),
            chain: vec![OpId::Conv],
        };
        let mut out_view = output.as_mut_view().unwrap();
        run(
            &fused,
            &views,
            std::slice::from_mut(&mut out_view),
            &conv_params(),
            None,
        )
        .unwrap();
    }

    fn zeroed() -> Tensor {
        Tensor::zeros(
            Shape::from_dims([1, 2, 2, 2]),
            DType::F32,
            crate::inference::device::DeviceId::Cpu,
        )
        .unwrap()
    }

    #[test]
    fn a_fused_relu_equals_the_unfused_relu() {
        let mut fused_out = zeroed();
        fused(&mut fused_out, Some(Activation::Relu), None);
        // The same two kernels the unfused `Conv` then `Relu` segments use.
        let (x, w, b) = conv_inputs();
        let mut conv_out = zeroed();
        {
            let views = [
                x.as_view().unwrap(),
                w.as_view().unwrap(),
                b.as_view().unwrap(),
            ];
            let mut out = conv_out.as_mut_view().unwrap();
            kernels::conv::run(&views, std::slice::from_mut(&mut out), &conv_params(), None)
                .unwrap();
        }
        let mut expected = zeroed();
        {
            let input = conv_out.as_view().unwrap();
            let mut out = expected.as_mut_view().unwrap();
            // `Relu` is one of the legacy ids, so the unfused plan runs it
            // through the backend's table with a budget of one.
            crate::inference::backend::cpu::dispatch_kernels(
                OpId::Relu,
                std::slice::from_ref(&input),
                std::slice::from_mut(&mut out),
                &OpParams::new(),
                1,
            )
            .unwrap();
        }
        assert_eq!(
            fused_out.as_f32_slice().unwrap(),
            expected.as_f32_slice().unwrap(),
            "the fused relu must be bit-identical to conv then relu"
        );
    }

    #[test]
    fn a_fused_swish_equals_sigmoid_then_multiply() {
        let mut fused_out = zeroed();
        fused(&mut fused_out, Some(Activation::Swish), None);
        let (x, w, b) = conv_inputs();
        let mut conv = zeroed();
        {
            let views = [
                x.as_view().unwrap(),
                w.as_view().unwrap(),
                b.as_view().unwrap(),
            ];
            let mut out = conv.as_mut_view().unwrap();
            kernels::conv::run(&views, std::slice::from_mut(&mut out), &conv_params(), None)
                .unwrap();
        }
        let mut sigmoid = zeroed();
        {
            let view = conv.as_view().unwrap();
            let mut out = sigmoid.as_mut_view().unwrap();
            kernels::act::run(
                OpId::Sigmoid,
                std::slice::from_ref(&view),
                std::slice::from_mut(&mut out),
                &OpParams::new(),
                None,
            )
            .unwrap();
        }
        let expected: Vec<f32> = conv
            .as_f32_slice()
            .unwrap()
            .iter()
            .zip(sigmoid.as_f32_slice().unwrap().iter())
            .map(|(&a, &b)| a * b)
            .collect();
        assert_eq!(fused_out.as_f32_slice().unwrap(), expected.as_slice());
    }

    #[test]
    fn a_fused_sum_and_activation_equals_the_three_unfused_steps() {
        let residual = Tensor::from_f32(
            Shape::from_dims([1, 2, 2, 2]),
            &[0.25, 0.5, -1.0, 2.0, -0.5, 1.5, 0.0, -2.5],
        )
        .unwrap();
        let mut fused_out = zeroed();
        fused(&mut fused_out, Some(Activation::Sigmoid), Some(&residual));
        let (x, w, b) = conv_inputs();
        let mut conv = zeroed();
        {
            let views = [
                x.as_view().unwrap(),
                w.as_view().unwrap(),
                b.as_view().unwrap(),
            ];
            let mut out = conv.as_mut_view().unwrap();
            kernels::conv::run(&views, std::slice::from_mut(&mut out), &conv_params(), None)
                .unwrap();
        }
        let expected: Vec<f32> = conv
            .as_f32_slice()
            .unwrap()
            .iter()
            .zip(residual.as_f32_slice().unwrap().iter())
            .map(|(&a, &b)| 1.0 / (1.0 + (-(a + b)).exp()))
            .collect();
        assert_eq!(fused_out.as_f32_slice().unwrap(), expected.as_slice());
    }
}
