//! `Pad`: constant, reflect and edge padding, with negative pads cropping.
//!
//! # Mapping
//!
//! ONNX supplies `rank` begin extents followed by `rank` end extents. Output
//! extent `O = I + begin + end`; a negative pair crops, and the shape pass has
//! already rejected a pair that would make an extent negative. Output coordinate
//! `o` reads input coordinate `s = o - begin`:
//!
//! | Mode | `s` outside `[0, I)` |
//! |---|---|
//! | `constant` | the fill value `pad_value` |
//! | `edge` | `clamp(s, 0, I - 1)` |
//! | `reflect` | mirror without repeating the edge: `-s`, then `2(I - 1) - s`, repeated until in range |
//! | `wrap` | rejected — there is no kernel |
//!
//! `reflect` repeats the mirror for a pad wider than the axis, which ONNX allows
//! for `mode = reflect` and which a single reflection would get wrong.
//!
//! # Fill values
//!
//! Only `constant` mode may carry a non-zero fill value; the shape pass enforces
//! that, and the kernel checks it again because `CpuBackend::execute_op` can be
//! called without a shape pass.

use crate::inference::backend::{OpParams, PadMode};
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::runtime::kernels::{
    check_input_len, check_output_len, extents, for_each_row, input_f32, output_f32, product,
    require_extent,
};
use crate::inference::runtime::SessionThreads;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

/// Execute `Pad` from views.
pub(crate) fn run(
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let op = OpId::Pad;
    if inputs.is_empty() || inputs.len() > 3 || outputs.len() != 1 {
        return Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs: 1,
            actual_inputs: inputs.len(),
            expected_outputs: 1,
            actual_outputs: outputs.len(),
        });
    }
    let x = input_f32(op, &inputs[0])?;
    let x_shape = inputs[0].shape().clone();
    let out_shape = outputs[0].shape().clone();
    let out = output_f32(op, &mut outputs[0])?;
    pad_f32(x, &x_shape, out, &out_shape, params, threads)
}

/// The kernel: contiguous operands, explicit shapes.
pub(crate) fn pad_f32(
    x: &[f32],
    x_shape: &Shape,
    out: &mut [f32],
    out_shape: &Shape,
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let op = OpId::Pad;
    let rank = x_shape.rank();
    if out_shape.rank() != rank {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("a rank-{rank} output like {x_shape}"),
            actual: format!("{out_shape}"),
        });
    }
    if params.pad_mode == PadMode::Wrap {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: "fn3-runtime-kernels",
            reason: "op pad mode wrap has no kernel; constant, reflect and edge do".to_string(),
        });
    }
    if params.pad_mode != PadMode::Constant && params.pad_value != 0.0 {
        return Err(Fn3Error::InvalidShape {
            context: "runtime pad",
            detail: format!(
                "op pad in {} mode cannot use the fill value {}",
                params.pad_mode.name(),
                params.pad_value
            ),
        });
    }
    if params.pads.len() != 2 * rank {
        return Err(Fn3Error::InvalidShape {
            context: "runtime pad",
            detail: format!(
                "op pad has {} pad extents for a rank-{rank} operand; ONNX supplies {rank} begin \
                 extents followed by {rank} end extents",
                params.pads.len()
            ),
        });
    }
    let input = extents(x_shape, "runtime pad input")?;
    let planned = extents(out_shape, "runtime pad output")?;
    let mut begins = vec![0i64; rank];
    for axis in 0..rank {
        let begin = params.pads[axis];
        let end = params.pads[rank + axis];
        begins[axis] = begin;
        let computed = i128::from(input[axis] as i64) + i128::from(begin) + i128::from(end);
        if computed < 0 {
            return Err(Fn3Error::InvalidShape {
                context: "runtime pad",
                detail: format!(
                    "op pad on axis {axis} of extent {} with pads {begin} + {end} would have \
                     {computed} elements",
                    input[axis]
                ),
            });
        }
        require_extent(op, "pad output", axis, computed as usize, planned[axis])?;
    }
    let elements = product(&input, "runtime pad input")?;
    check_input_len(op, x, elements, x_shape, "input")?;
    let out_elements = product(&planned, "runtime pad output")?;
    check_output_len(op, out, out_elements, out_shape)?;
    if out_elements == 0 {
        return Ok(());
    }

    let row_len = planned[rank - 1];
    let mut input_strides = vec![1usize; rank];
    for axis in (0..rank - 1).rev() {
        input_strides[axis] = input_strides[axis + 1] * input[axis + 1];
    }
    let begins = &begins;
    let input = &input;
    let planned = &planned;
    let input_strides = &input_strides;
    let mode = params.pad_mode;
    let fill = params.pad_value;

    // One row is one last-axis line, which is exactly what `for_each_row`
    // hands out: the last coordinate is the column index and never carries.
    for_each_row(threads, out, row_len, |row, values| {
        let mut coordinates = vec![0usize; rank];
        let mut remainder = row;
        for axis in (0..rank - 1).rev() {
            coordinates[axis] = remainder % planned[axis].max(1);
            remainder /= planned[axis].max(1);
        }
        for (column, value) in values.iter_mut().enumerate() {
            coordinates[rank - 1] = column;
            let mut source = 0usize;
            let mut inside = true;
            for axis in 0..rank {
                let coordinate = coordinates[axis] as i64 - begins[axis];
                let mapped = match mode {
                    PadMode::Constant => {
                        if coordinate < 0 || coordinate >= input[axis] as i64 {
                            inside = false;
                            0
                        } else {
                            coordinate as usize
                        }
                    }
                    PadMode::Edge => {
                        coordinate.clamp(0, input[axis].saturating_sub(1) as i64) as usize
                    }
                    PadMode::Reflect => reflect(coordinate, input[axis]),
                    PadMode::Wrap => unreachable!("wrap was rejected above"),
                };
                source += mapped * input_strides[axis];
            }
            *value = if inside { x[source] } else { fill };
        }
        Ok(())
    })
}

/// Mirror `coordinate` into `[0, extent)` without repeating the edge sample.
fn reflect(mut coordinate: i64, extent: usize) -> usize {
    if extent <= 1 {
        return 0;
    }
    let extent = extent as i64;
    loop {
        if coordinate < 0 {
            coordinate = -coordinate;
        } else if coordinate >= extent {
            coordinate = 2 * (extent - 1) - coordinate;
        } else {
            return coordinate as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pad(x: &[f32], x_dims: &[u64], out_dims: &[u64], params: &OpParams) -> Vec<f32> {
        let out_len: u64 = out_dims.iter().product();
        let mut out = vec![0.0f32; out_len as usize];
        pad_f32(
            x,
            &Shape::from_dims(x_dims.iter().copied()),
            &mut out,
            &Shape::from_dims(out_dims.iter().copied()),
            params,
            None,
        )
        .unwrap();
        out
    }

    #[test]
    fn constant_padding_fills_with_the_value() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let params = OpParams::new()
            .with_pad(PadMode::Constant, -1.0)
            .with_pads([0, 0, 1, 1, 0, 0, 1, 1]);
        let out = pad(&x, &[1, 1, 2, 2], &[1, 1, 4, 4], &params);
        assert_eq!(
            out,
            [
                -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 2.0, -1.0, -1.0, 3.0, 4.0, -1.0, -1.0, -1.0,
                -1.0, -1.0
            ]
        );
    }

    #[test]
    fn edge_padding_repeats_the_border() {
        let x = [1.0f32, 2.0];
        let params = OpParams::new()
            .with_pad(PadMode::Edge, 0.0)
            .with_pads([1, 1]);
        // 1-D: one element on each side.
        let out = pad(&x, &[2], &[4], &params);
        assert_eq!(out, [1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn reflect_mirrors_without_repeating_the_edge() {
        let x = [1.0f32, 2.0, 3.0];
        let params = OpParams::new()
            .with_pad(PadMode::Reflect, 0.0)
            .with_pads([1, 1]);
        let out = pad(&x, &[3], &[5], &params);
        assert_eq!(out, [2.0, 1.0, 2.0, 3.0, 2.0]);
        // A pad wider than the axis reflects twice: index -3 -> 3 -> 1.
        let params = OpParams::new()
            .with_pad(PadMode::Reflect, 0.0)
            .with_pads([3, 0]);
        let out = pad(&x, &[3], &[6], &params);
        assert_eq!(out, [2.0, 3.0, 2.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn negative_pads_crop() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let params = OpParams::new()
            .with_pad(PadMode::Constant, 0.0)
            .with_pads([-1, 0, 0, 1]);
        let out = pad(&x, &[3, 2], &[2, 3], &params);
        assert_eq!(out, [3.0, 4.0, 0.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn wrap_and_a_negative_extent_are_reported() {
        let x = [0.0f32];
        let mut out = [0.0f32; 1];
        let params = OpParams::new()
            .with_pad(PadMode::Wrap, 0.0)
            .with_pads([0, 0]);
        let error = pad_f32(
            &x,
            &Shape::from_dims([1]),
            &mut out,
            &Shape::from_dims([1]),
            &params,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("wrap"), "{error}");

        let params = OpParams::new()
            .with_pad(PadMode::Constant, 0.0)
            .with_pads([-2, 0]);
        let error = pad_f32(
            &x,
            &Shape::from_dims([1]),
            &mut out,
            &Shape::from_dims([1]),
            &params,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("elements"), "{error}");
    }
}
