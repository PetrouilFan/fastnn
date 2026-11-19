//! Numerically stable softmax over one axis.
//!
//! # Contract
//!
//! ```text
//! shifted[a] = x[a] - max(x)            (max over the softmax axis, f32)
//! out[a]     = exp(shifted[a]) / sum(exp(shifted))   (ascending a, f32)
//! ```
//!
//! One `f32` accumulator for the sum, in ascending axis index, and a final
//! division per element (not a reciprocal multiply). This is the order a faster
//! kernel must reproduce or publish a tolerance against.
//!
//! # The all-masked row is a conformance requirement
//!
//! The pinned GPT-2 graph masks with the **finite** value `-FLT_MAX`
//! (`-3.4028234663852886e38`), not `-inf`. Max subtraction makes every shifted
//! value `0`, so a fully masked row is uniform:
//!
//! ```text
//! softmax([-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX]) = [0.25, 0.25, 0.25, 0.25]
//! ```
//!
//! This is pinned by `fixtures/parity/ops` (`softmax.all_masked`,
//! `attention.mask`); a NaN here is a conformance failure, not a tolerance
//! question. `-FLT_MAX - max` can overflow to `-inf`, which is fine: `exp(-inf)`
//! is exactly `0`. It never produces `NaN` because the maximum itself is
//! finite.
//!
//! # Broadcasting
//!
//! None. The input and output shapes must match; the additive mask is an
//! explicit `Add` before the `Softmax`, exactly as the graph does it.

use crate::inference::backend::OpParams;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::tensor::{TensorMut, TensorView};

use super::{
    at, at_mut, checked_product, decompose, require_dtype, require_output_shape, single_axis,
    Locator,
};

/// Softmax over one axis (default `-1`).
pub(crate) fn softmax(
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    require_dtype(op, input.meta(), DType::F32)?;
    require_dtype(op, out.meta(), DType::F32)?;
    let rank = input.shape().rank();
    let axis = single_axis(op, params, rank, -1, "softmax axis")?;
    require_output_shape(op, input.shape(), out)?;

    let outer = checked_product(&input.shape().dims()[..axis], "cpu softmax outer")?;
    let width = input.shape().dims()[axis];
    let inner = checked_product(&input.shape().dims()[axis + 1..], "cpu softmax inner")?;
    if outer == 0 || width == 0 || inner == 0 {
        return Ok(());
    }
    let width_usize = usize::try_from(width).map_err(|_| Fn3Error::ShapeOverflow {
        context: "cpu softmax width",
        axis: Some(axis),
        detail: "softmax width does not fit in usize".to_string(),
    })?;

    let locator = Locator::new(op, input.meta(), input.shape())?;
    let out_locator = Locator::new(op, out.meta(), input.shape())?;
    let source = input.storage_slice::<f32>()?;
    let destination = out.storage_slice_mut::<f32>()?;
    let leading_dims = input.shape().dims()[..axis].to_vec();
    let trailing_dims = input.shape().dims()[axis + 1..].to_vec();
    let mut coordinates = vec![0u64; rank];
    let mut scratch = vec![0.0f32; width_usize];

    for outer_index in 0..outer {
        for row in 0..inner {
            for coordinate in coordinates.iter_mut() {
                *coordinate = 0;
            }
            if !leading_dims.is_empty() {
                decompose(outer_index, &leading_dims, &mut coordinates[..axis]);
            }
            if !trailing_dims.is_empty() {
                decompose(row, &trailing_dims, &mut coordinates[axis + 1..]);
            }

            // Maximum over the axis.
            let mut maximum = f32::NEG_INFINITY;
            for position in 0..width {
                coordinates[axis] = position;
                let value = at(op, source, locator.index(op, &coordinates)?)?;
                if value > maximum {
                    maximum = value;
                }
            }
            // Exp and ascending sum.
            let mut total = 0.0f32;
            for position in 0..width {
                coordinates[axis] = position;
                let value = at(op, source, locator.index(op, &coordinates)?)?;
                let exponent = (value - maximum).exp();
                *at_mut(op, &mut scratch, position as usize)? = exponent;
                total += exponent;
            }
            // Divide. `total` is >= 1 for a non-empty row because the maximum
            // element contributes exp(0) = 1, so this cannot divide by zero.
            for position in 0..width {
                coordinates[axis] = position;
                let exponent = at(op, &scratch, position as usize)?;
                let out_index = out_locator.index(op, &coordinates)?;
                *at_mut(op, destination, out_index)? = exponent / total;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::device::DeviceId;
    use crate::inference::shape::Shape;
    use crate::inference::tensor::Tensor;

    fn run(input: &[f32], shape: [u64; 2]) -> Vec<f32> {
        let input = Tensor::from_f32(Shape::from_dims(shape), input).unwrap();
        let mut out = Tensor::zeros(Shape::from_dims(shape), DType::F32, DeviceId::Cpu).unwrap();
        let input_view = input.as_view().unwrap();
        let mut out_view = out.as_mut_view().unwrap();
        softmax(
            OpId::Softmax,
            &input_view,
            &mut out_view,
            &OpParams::default().with_axes([-1]),
        )
        .unwrap();
        out.as_f32_slice().unwrap().to_vec()
    }

    #[test]
    fn all_masked_rows_are_uniform_not_nan() {
        let masked = run(
            &[
                -3.402_823_5e38,
                -3.402_823_5e38,
                -3.402_823_5e38,
                -3.402_823_5e38,
            ],
            [1, 4],
        );
        assert_eq!(masked, vec![0.25, 0.25, 0.25, 0.25]);
        assert!(masked.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn masked_entries_vanish_beside_a_finite_maximum() {
        let row = run(&[1.0, -3.402_823_5e38, 2.0, 3.0], [1, 4]);
        // exp(-FLT_MAX - 3.0) underflows to 0; the remaining three values are a
        // normal softmax over [1, 2, 3].
        let denominator = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp();
        assert_eq!(row[1], 0.0);
        assert!((row[0] - 1.0f32.exp() / denominator).abs() < 1e-6);
        assert!((row[3] - 3.0f32.exp() / denominator).abs() < 1e-6);
    }
}
