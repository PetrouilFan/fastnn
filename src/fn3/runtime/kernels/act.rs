//! `Clip`, `Sigmoid`, and `Maximum`/`Minimum` with broadcasting.
//!
//! # `Clip` and `Sigmoid`
//!
//! Both are pointwise, so a row split is arbitrary and exact: every element is
//! computed by the same expression in the same `f32` order whatever thread ran
//! it. `Clip` evaluates `min(max(x, low), high)` from the folded `OpParams`
//! bounds, which is ONNX's `Clip`; a missing bound leaves that side alone. The
//! two bounds are validated (`low <= high`, both finite) before the first write,
//! because `f32::clamp` would panic on a reversed interval.
//!
//! # `Maximum` and `Minimum`
//!
//! These delegate to the existing elementwise binary kernel
//! ([`crate::inference::backend::cpu::kernels::elementwise::binary`]) with a
//! `max`/`min` closure. Right-aligned NumPy broadcasting is subtle enough
//! (effective zero strides, run builders, scalar fast paths) that a second
//! implementation would be a second chance to get it wrong; the shared kernel is
//! the one the rest of the crate is tested against.

use crate::inference::backend::OpParams;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::runtime::kernels::{
    arity, check_input_len, check_output_len, extents, for_each_row, input_f32, output_f32, product,
};
use crate::inference::runtime::SessionThreads;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

/// Execute one of the four activation op ids.
pub(crate) fn run(
    op: OpId,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    match op {
        OpId::Clip => {
            if inputs.is_empty() || inputs.len() > 3 {
                return Err(Fn3Error::ArityMismatch {
                    op,
                    expected_inputs: 1,
                    actual_inputs: inputs.len(),
                    expected_outputs: 1,
                    actual_outputs: outputs.len(),
                });
            }
            arity(op, 1, outputs.len(), 1, 1)?;
            let x = input_f32(op, &inputs[0])?;
            let x_shape = inputs[0].shape().clone();
            let out_shape = outputs[0].shape().clone();
            let out = output_f32(op, &mut outputs[0])?;
            clip_f32(x, &x_shape, out, &out_shape, params, threads)
        }
        OpId::Sigmoid => {
            arity(op, inputs.len(), outputs.len(), 1, 1)?;
            let x = input_f32(op, &inputs[0])?;
            let x_shape = inputs[0].shape().clone();
            let out_shape = outputs[0].shape().clone();
            let out = output_f32(op, &mut outputs[0])?;
            sigmoid_f32(x, &x_shape, out, &out_shape, threads)
        }
        OpId::Maximum | OpId::Minimum => {
            arity(op, inputs.len(), outputs.len(), 2, 1)?;
            fn maximum(x: f32, y: f32) -> f32 {
                x.max(y)
            }
            fn minimum(x: f32, y: f32) -> f32 {
                x.min(y)
            }
            let evaluate: fn(f32, f32) -> f32 = if op == OpId::Maximum {
                maximum
            } else {
                minimum
            };
            crate::inference::backend::cpu::kernels::elementwise::binary(
                op,
                &inputs[0],
                &inputs[1],
                &mut outputs[0],
                evaluate,
            )
        }
        other => Err(Fn3Error::UnsupportedOp {
            op: other,
            backend: "fn3-runtime-kernels",
            reason: "not a Clip/Sigmoid/Maximum/Minimum id".to_string(),
        }),
    }
}

/// `clip(x, low, high)` with either bound optional.
pub(crate) fn clip_f32(
    x: &[f32],
    x_shape: &Shape,
    out: &mut [f32],
    out_shape: &Shape,
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let op = OpId::Clip;
    if out_shape != x_shape {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("the input shape {x_shape}"),
            actual: format!("{out_shape}"),
        });
    }
    for bound in [params.clip_min, params.clip_max].into_iter().flatten() {
        if !bound.is_finite() {
            return Err(Fn3Error::InvalidShape {
                context: "runtime clip",
                detail: format!("op clip bound {bound} is not finite"),
            });
        }
    }
    if let (Some(low), Some(high)) = (params.clip_min, params.clip_max) {
        if low > high {
            return Err(Fn3Error::InvalidShape {
                context: "runtime clip",
                detail: format!("op clip min {low} is above max {high}"),
            });
        }
    }
    let dims = extents(x_shape, "runtime clip input")?;
    let elements = product(&dims, "runtime clip input")?;
    check_input_len(op, x, elements, x_shape, "input")?;
    check_output_len(op, out, elements, out_shape)?;
    let row_len = *dims.last().unwrap_or(&1);
    let low = params.clip_min;
    let high = params.clip_max;
    for_each_row(threads, out, row_len, |row, values| {
        let source = &x[row * row_len..row * row_len + row_len];
        for (slot, &value) in values.iter_mut().zip(source.iter()) {
            let mut value = value;
            if let Some(low) = low {
                value = value.max(low);
            }
            if let Some(high) = high {
                value = value.min(high);
            }
            *slot = value;
        }
        Ok(())
    })
}

/// `1 / (1 + exp(-x))`.
pub(crate) fn sigmoid_f32(
    x: &[f32],
    x_shape: &Shape,
    out: &mut [f32],
    out_shape: &Shape,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let op = OpId::Sigmoid;
    if out_shape != x_shape {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("the input shape {x_shape}"),
            actual: format!("{out_shape}"),
        });
    }
    let dims = extents(x_shape, "runtime sigmoid input")?;
    let elements = product(&dims, "runtime sigmoid input")?;
    check_input_len(op, x, elements, x_shape, "input")?;
    check_output_len(op, out, elements, out_shape)?;
    let row_len = *dims.last().unwrap_or(&1);
    for_each_row(threads, out, row_len, |row, values| {
        let source = &x[row * row_len..row * row_len + row_len];
        for (slot, &value) in values.iter_mut().zip(source.iter()) {
            // For a large negative input `exp(-x)` is `+inf` and the result is
            // `0.0`: the limit, not a NaN, and no overflow trap in `f32`.
            *slot = 1.0 / (1.0 + (-value).exp());
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::shape::Shape;
    use crate::inference::tensor::Tensor;

    fn tensor(values: &[f32], dims: &[u64]) -> Tensor {
        Tensor::from_f32(Shape::from_dims(dims.iter().copied()), values).unwrap()
    }

    fn run_act(
        op: OpId,
        inputs: &[&Tensor],
        out: &mut Tensor,
        params: &OpParams,
    ) -> Result<(), Fn3Error> {
        let views: Vec<TensorView<'_>> = inputs.iter().map(|t| t.as_view().unwrap()).collect();
        let mut out_view = out.as_mut_view().unwrap();
        run(
            op,
            &views,
            std::slice::from_mut(&mut out_view),
            params,
            None,
        )
    }

    fn zeros(dims: &[u64]) -> Tensor {
        Tensor::zeros(
            Shape::from_dims(dims.iter().copied()),
            crate::inference::dtype::DType::F32,
            crate::inference::device::DeviceId::Cpu,
        )
        .unwrap()
    }

    #[test]
    fn clip_uses_the_folded_bounds() {
        let x = tensor(&[-2.0, 0.0, 0.5, 3.0], &[4]);
        let mut out = zeros(&[4]);
        let params = OpParams::new().with_clip(Some(0.0), Some(1.0));
        run_act(OpId::Clip, &[&x], &mut out, &params).unwrap();
        assert_eq!(out.as_f32_slice().unwrap(), &[0.0, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn clip_with_one_bound_leaves_the_other_side_alone() {
        let x = tensor(&[-2.0, 0.0, 0.5, 3.0], &[4]);
        let mut out = zeros(&[4]);
        run_act(
            OpId::Clip,
            &[&x],
            &mut out,
            &OpParams::new().with_clip(Some(0.0), None),
        )
        .unwrap();
        assert_eq!(out.as_f32_slice().unwrap(), &[0.0, 0.0, 0.5, 3.0]);
    }

    #[test]
    fn sigmoid_matches_the_closed_form_at_the_extremes() {
        let x = tensor(&[0.0, 40.0, -40.0, 1.0], &[4]);
        let mut out = zeros(&[4]);
        run_act(OpId::Sigmoid, &[&x], &mut out, &OpParams::new()).unwrap();
        let out = out.as_f32_slice().unwrap();
        assert_eq!(out[0], 0.5);
        assert_eq!(out[1], 1.0);
        assert!(
            out[2] < 1e-15 && out[2] >= 0.0,
            "sigmoid(-40) is the underflow tail of exp, not a negative: {out:?}"
        );
        assert!((out[3] - 0.7310586).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn maximum_and_minimum_broadcast_like_numpy() {
        let x = tensor(&[1.0, 5.0, 2.0, 8.0], &[2, 2]);
        let y = tensor(&[4.0, 4.0], &[2]);
        let mut out = zeros(&[2, 2]);
        run_act(OpId::Maximum, &[&x, &y], &mut out, &OpParams::new()).unwrap();
        assert_eq!(out.as_f32_slice().unwrap(), &[4.0, 5.0, 4.0, 8.0]);

        let mut out = zeros(&[2, 2]);
        run_act(OpId::Minimum, &[&x, &y], &mut out, &OpParams::new()).unwrap();
        assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 4.0, 2.0, 4.0]);
    }

    #[test]
    fn a_reversed_clip_interval_is_rejected_before_writing() {
        let x = tensor(&[1.0], &[1]);
        let mut out = zeros(&[1]);
        let params = OpParams::new().with_clip(Some(2.0), Some(1.0));
        let error = run_act(OpId::Clip, &[&x], &mut out, &params).unwrap_err();
        assert!(error.to_string().contains("above max"), "{error}");
        assert_eq!(out.as_f32_slice().unwrap(), &[0.0]);
    }
}
