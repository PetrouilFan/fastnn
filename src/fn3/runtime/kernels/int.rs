//! Integer arithmetic on `i64` values.
//!
//! # Why this exists
//!
//! ONNX exporters compute shape arithmetic in `i64`: a YOLO head derives an
//! anchor count from `Shape` → `Gather` → `Add` → `Div` → `Mul`, and a decoder
//! derives positions and a causal mask range the same way. The importer
//! deliberately preserves those nodes (`import/mod.rs`: *`i64` arithmetic is not
//! rejected*), so a plan that executes every node needs an `i64` kernel for them.
//!
//! The alpha elementwise kernel
//! ([`crate::inference::backend::cpu::kernels::elementwise`]) is `f32`-only by
//! contract and stays that way: it is the numerical reference the models are
//! measured against, and adding an integer branch to it would put two evaluation
//! orders behind one name. This module is the integer sibling, called only when
//! an operand is `i64`.
//!
//! # Semantics
//!
//! - `Add`/`Sub`/`Mul`/`Neg` wrap on overflow, which is what the removed 2.x
//!   planner did and what a debug build of a plain `+` would panic on;
//! - `Div` truncates toward zero (C and ONNX integer division) and a zero divisor
//!   is an error naming the op, never a panic or a saturated value;
//! - right-aligned NumPy broadcasting, exactly like the `f32` binary kernel, so a
//!   scalar shape value can meet a `[4]` vector.
//!
//! # Shapes
//!
//! The output shape is the broadcast of the two input shapes, checked against the
//! plan's output. The kernel is serial: these tensors hold a handful of elements
//! (a rank, a stride, a count), so a parallel split would cost more than the work.

use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::runtime::kernels::{arity, require_dtype};
use crate::inference::tensor::{TensorMut, TensorView};

/// Execute an integer elementwise op from views.
pub(crate) fn run(
    op: OpId,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
) -> Result<(), Fn3Error> {
    match op {
        OpId::Neg => {
            arity(op, inputs.len(), outputs.len(), 1, 1)?;
            require_dtype(op, &inputs[0], DType::I64)?;
            outputs[0].meta().require_dtype(Some(op), DType::I64)?;
            if inputs[0].shape() != outputs[0].shape() {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("the input shape {}", inputs[0].shape()),
                    actual: format!("{}", outputs[0].shape()),
                });
            }
            let source = inputs[0].as_slice::<i64>()?;
            let destination = outputs[0].as_mut_slice::<i64>()?;
            for (slot, &value) in destination.iter_mut().zip(source.iter()) {
                *slot = value.wrapping_neg();
            }
            Ok(())
        }
        OpId::Add | OpId::Sub | OpId::Mul | OpId::Div => {
            arity(op, inputs.len(), outputs.len(), 2, 1)?;
            require_dtype(op, &inputs[0], DType::I64)?;
            require_dtype(op, &inputs[1], DType::I64)?;
            outputs[0].meta().require_dtype(Some(op), DType::I64)?;
            let shape = inputs[0].shape().broadcast_with(inputs[1].shape(), op)?;
            if outputs[0].shape() != &shape {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("the broadcast shape {shape}"),
                    actual: format!("{}", outputs[0].shape()),
                });
            }
            let lhs = inputs[0].as_slice::<i64>()?;
            let rhs = inputs[1].as_slice::<i64>()?;
            let destination = outputs[0].as_mut_slice::<i64>()?;
            binary(
                op,
                lhs,
                inputs[0].shape(),
                rhs,
                inputs[1].shape(),
                &shape,
                destination,
            )
        }
        other => Err(Fn3Error::UnsupportedOp {
            op: other,
            backend: "fn3-runtime-kernels",
            reason: "integer arithmetic is implemented for add, sub, mul, div and neg only"
                .to_string(),
        }),
    }
}

/// Right-aligned broadcast over `i64` operands.
fn binary(
    op: OpId,
    lhs: &[i64],
    lhs_shape: &crate::inference::shape::Shape,
    rhs: &[i64],
    rhs_shape: &crate::inference::shape::Shape,
    result: &crate::inference::shape::Shape,
    destination: &mut [i64],
) -> Result<(), Fn3Error> {
    let rank = result.rank();
    let lhs_lead = rank - lhs_shape.rank();
    let rhs_lead = rank - rhs_shape.rank();
    let lhs_strides = broadcast_strides(lhs_shape, rank, lhs_lead)?;
    let rhs_strides = broadcast_strides(rhs_shape, rank, rhs_lead)?;
    let dims = result.dims();
    let elements = result.numel_usize()?;
    if destination.len() != elements {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{elements} output elements for {result}"),
            actual: format!("{}", destination.len()),
        });
    }
    let mut coordinates = vec![0u64; rank];
    for (linear, slot) in destination.iter_mut().enumerate() {
        let mut remainder = linear;
        for axis in (0..rank).rev() {
            let extent = dims[axis].max(1) as usize;
            coordinates[axis] = (remainder % extent) as u64;
            remainder /= extent;
        }
        let mut left = 0usize;
        let mut right = 0usize;
        for axis in 0..rank {
            left += coordinates[axis] as usize * lhs_strides[axis];
            right += coordinates[axis] as usize * rhs_strides[axis];
        }
        let left = *lhs.get(left).ok_or_else(|| bounds(op, left, lhs.len()))?;
        let right = *rhs.get(right).ok_or_else(|| bounds(op, right, rhs.len()))?;
        *slot = match op {
            OpId::Add => left.wrapping_add(right),
            OpId::Sub => left.wrapping_sub(right),
            OpId::Mul => left.wrapping_mul(right),
            OpId::Div => {
                if right == 0 {
                    return Err(Fn3Error::InvalidShape {
                        context: "runtime integer div",
                        detail: format!("op {op}: integer division by zero"),
                    });
                }
                left.wrapping_div(right)
            }
            other => {
                return Err(Fn3Error::UnsupportedOp {
                    op: other,
                    backend: "fn3-runtime-kernels",
                    reason: "not an integer arithmetic op".to_string(),
                })
            }
        };
    }
    Ok(())
}

/// Strides for an operand promoted to `rank` axes, `0` on promoted axes.
fn broadcast_strides(
    shape: &crate::inference::shape::Shape,
    rank: usize,
    lead: usize,
) -> Result<Vec<usize>, Fn3Error> {
    let mut strides = vec![0usize; rank];
    let dims = shape.dims();
    let mut stride = 1usize;
    for axis in (0..dims.len()).rev() {
        strides[lead + axis] = if dims[axis] == 1 { 0 } else { stride };
        stride = stride.saturating_mul(dims[axis] as usize);
    }
    Ok(strides)
}

fn bounds(op: OpId, index: usize, len: usize) -> Fn3Error {
    Fn3Error::InvalidShape {
        context: "runtime integer elementwise",
        detail: format!("op {op}: element index {index} is outside the {len}-element operand"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::device::DeviceId;
    use crate::inference::shape::Shape;
    use crate::inference::tensor::Tensor;

    fn i64_tensor(values: &[i64], dims: &[u64]) -> Tensor {
        Tensor::from_i64(Shape::from_dims(dims.iter().copied()), values).unwrap()
    }

    fn run_op(op: OpId, inputs: &[&Tensor], out: &mut Tensor) {
        let views: Vec<TensorView<'_>> = inputs.iter().map(|t| t.as_view().unwrap()).collect();
        let mut out_view = out.as_mut_view().unwrap();
        run(op, &views, std::slice::from_mut(&mut out_view)).unwrap();
    }

    #[test]
    fn add_truncating_div_and_wrapping_mul_keep_integer_semantics() {
        let a = i64_tensor(&[7, -7, 9], &[3]);
        let b = i64_tensor(&[2, 2, 3], &[3]);
        let mut out = Tensor::zeros(Shape::from_dims([3]), DType::I64, DeviceId::Cpu).unwrap();
        run_op(OpId::Add, &[&a, &b], &mut out);
        assert_eq!(out.as_i64_slice().unwrap(), &[9, -5, 12]);
        run_op(OpId::Div, &[&a, &b], &mut out);
        assert_eq!(
            out.as_i64_slice().unwrap(),
            &[3, -3, 3],
            "integer division truncates toward zero"
        );
        run_op(OpId::Mul, &[&a, &b], &mut out);
        assert_eq!(out.as_i64_slice().unwrap(), &[14, -14, 27]);
    }

    #[test]
    fn a_scalar_broadcasts_and_wraps() {
        let a = i64_tensor(&[i64::MAX, 1], &[2]);
        let one = i64_tensor(&[1], &[1]);
        let mut out = Tensor::zeros(Shape::from_dims([2]), DType::I64, DeviceId::Cpu).unwrap();
        run_op(OpId::Add, &[&a, &one], &mut out);
        assert_eq!(out.as_i64_slice().unwrap(), &[i64::MIN, 2]);
    }

    #[test]
    fn a_zero_divisor_is_an_error_not_a_panic() {
        let a = i64_tensor(&[1], &[1]);
        let zero = i64_tensor(&[0], &[1]);
        let mut out = Tensor::zeros(Shape::from_dims([1]), DType::I64, DeviceId::Cpu).unwrap();
        let views: Vec<TensorView<'_>> = [&a, &zero].iter().map(|t| t.as_view().unwrap()).collect();
        let mut out_view = out.as_mut_view().unwrap();
        let error = run(OpId::Div, &views, std::slice::from_mut(&mut out_view)).unwrap_err();
        assert!(error.to_string().contains("division by zero"), "{error}");
    }
}
