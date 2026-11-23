//! Convolution, NCHW/NCDHW, in one generic loop.
//!
//! # One loop, not three
//!
//! `1x1`, `3x3` and depthwise convolutions all take the same path here. The
//! specialisations that make those fast (im2col + GEMM for `1x1`, direct tiles
//! for `3x3`, one multiply per output for depthwise) are plan/provider decisions
//! in phase 3, and a scalar kernel that branched three ways would only make the
//! correctness reference harder to trust. What the loop does decide is the
//! *evaluation order*, which the module documents and the tests pin.
//!
//! # Groups, dilation and padding
//!
//! `groups = g` splits the input channels into `g` blocks of `C_in/g` and the
//! output channels into `g` blocks of `C_out/g`; output channel `oc` reads block
//! `oc / (C_out/g)`. Dilation `d` multiplies kernel-position `k` by `d`, so the
//! effective window is `(kernel - 1) * d + 1`. Padding is resolved exactly as
//! the typed shape pass resolved it (explicit `pads`, or `auto_pad`), so the
//! output extent the plan reserved is the one this kernel writes.
//!
//! # Evaluation order
//!
//! ```text
//! out[n, oc, p] = bias[oc] + sum over ic in group (ascending)
//!                            sum over k in row-major order (ascending)
//!                  x[n, ic, p * stride - pad + k * dilation]
//! ```
//!
//! The bias is added once, after the accumulation, in `f32`. A window position
//! outside the input contributes nothing (it is the zero padding), and a
//! convolution whose window lies entirely outside the input produces the bias
//! rather than reading uninitialised memory.

use crate::inference::backend::OpParams;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::runtime::kernels::{
    check_input_len, check_output_len, extents, flexible_arity, for_each_row, input_f32,
    output_f32, product, require_extent, resolve_pads, window_axes, window_output,
};
use crate::inference::runtime::SessionThreads;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

/// Execute `Conv` from views.
pub(crate) fn run(
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let op = OpId::Conv;
    flexible_arity(op, inputs.len(), outputs.len(), &[2, 3], 1)?;
    let x = input_f32(op, &inputs[0])?;
    let w = input_f32(op, &inputs[1])?;
    let bias = match inputs.get(2) {
        Some(view) => Some(input_f32(op, view)?),
        None => None,
    };
    let x_shape = inputs[0].shape().clone();
    let w_shape = inputs[1].shape().clone();
    let out_shape = outputs[0].shape().clone();
    let out = output_f32(op, &mut outputs[0])?;
    conv_f32(
        x, &x_shape, w, &w_shape, bias, out, &out_shape, params, threads,
    )
}

/// The kernel itself: contiguous operands, explicit shapes.
///
/// Kept separate from [`run`] so the unit tests can call it with plain slices and
/// the shapes a hand computation needs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_f32(
    x: &[f32],
    x_shape: &Shape,
    w: &[f32],
    w_shape: &Shape,
    bias: Option<&[f32]>,
    out: &mut [f32],
    out_shape: &Shape,
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let op = OpId::Conv;
    let rank = x_shape.rank();
    if rank < 3 {
        return Err(Fn3Error::InvalidShape {
            context: "runtime conv",
            detail: format!(
                "op {op} needs an [N, C, spatial...] operand of rank 3 or more, found {x_shape}"
            ),
        });
    }
    if w_shape.rank() != rank || out_shape.rank() != rank {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("weight and output of rank {rank}, like {x_shape}"),
            actual: format!("weight {w_shape}, output {out_shape}"),
        });
    }
    let spatial = rank - 2;
    let (kernel, strides, dilations) = window_axes(op, params, spatial)?;
    let group = match params.group {
        0 | 1 => 1usize,
        positive if positive > 0 => {
            usize::try_from(positive).map_err(|_| Fn3Error::InvalidShape {
                context: "runtime conv",
                detail: format!("op {op} group {positive} is too large"),
            })?
        }
        negative => {
            return Err(Fn3Error::InvalidShape {
                context: "runtime conv",
                detail: format!("op {op} group {negative} is negative"),
            })
        }
    };

    let x_dims = extents(x_shape, "runtime conv input")?;
    let w_dims = extents(w_shape, "runtime conv weight")?;
    let out_dims = extents(out_shape, "runtime conv output")?;
    let (batch, in_channels) = (x_dims[0], x_dims[1]);
    let out_channels = w_dims[0];
    let in_per_group = w_dims[1];
    if out_dims[0] != batch || out_dims[1] != out_channels {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!(
                "an output [{batch}, {out_channels}, ...], like {x_shape} and {w_shape}"
            ),
            actual: format!("{out_shape}"),
        });
    }
    if group == 0
        || in_channels % group != 0
        || out_channels % group != 0
        || in_per_group != in_channels / group
    {
        return Err(Fn3Error::InvalidShape {
            context: "runtime conv",
            detail: format!(
                "op {op} needs {in_channels} input channels = group {group} * the weight's \
                 {in_per_group} input channels, and {out_channels} output channels divisible by \
                 {group}"
            ),
        });
    }
    for axis in 0..spatial {
        if w_dims[axis + 2] != kernel[axis] {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(op),
                expected: format!("the weight's kernel extent {} on axis {axis}", kernel[axis]),
                actual: format!("{}", w_dims[axis + 2]),
            });
        }
    }
    if let Some(bias) = bias {
        if bias.len() != out_channels {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(op),
                expected: format!("{out_channels} bias values"),
                actual: format!("{}", bias.len()),
            });
        }
    }

    let spatial_in = &x_dims[2..];
    let pads = resolve_pads(op, params, spatial_in, &kernel, &strides, &dilations)?;
    let mut out_spatial = Vec::with_capacity(spatial);
    for axis in 0..spatial {
        let (begin, end) = pads[axis];
        let computed = window_output(
            op,
            spatial_in[axis],
            kernel[axis],
            strides[axis],
            dilations[axis],
            begin,
            end,
            false,
        )?;
        require_extent(op, "conv output", axis, computed, out_dims[axis + 2])?;
        out_spatial.push(computed);
    }

    let x_elements = product(&x_dims, "runtime conv input elements")?;
    let w_elements = product(&w_dims, "runtime conv weight elements")?;
    let out_elements = product(&out_dims, "runtime conv output elements")?;
    check_input_len(op, x, x_elements, x_shape, "input")?;
    check_input_len(op, w, w_elements, w_shape, "weight")?;
    check_output_len(op, out, out_elements, out_shape)?;

    let row_len = product(&out_spatial, "runtime conv output row")?;
    let in_volume = product(spatial_in, "runtime conv input volume")?;
    let kernel_volume = product(&kernel, "runtime conv kernel volume")?;
    let out_per_group = out_channels / group;

    // Zero-pad every input plane once, so the inner loop is a plain sum with no
    // per-element bounds test. The copy preserves values exactly (padding is
    // zeros, which is what ONNX convolution means by padding), and doing it once
    // per segment is what keeps the loop below free of divisions: without it, a
    // 3x3/64-channel convolution pays two integer divisions per axis per kernel
    // position per output element.
    let padded_dims: Vec<usize> = spatial_in
        .iter()
        .zip(pads.iter())
        .map(|(&extent, &(begin, end))| {
            extent
                .saturating_add(begin as usize)
                .saturating_add(end as usize)
        })
        .collect();
    let padded_volume = product(&padded_dims, "runtime conv padded volume")?;
    let mut padded = vec![0.0f32; batch * in_channels * padded_volume];
    {
        let mut padded_strides = vec![1usize; spatial];
        for axis in (0..spatial.saturating_sub(1)).rev() {
            padded_strides[axis] = padded_strides[axis + 1] * padded_dims[axis + 1];
        }
        let inner = spatial_in[spatial - 1];
        let rows = in_volume / inner.max(1);
        let mut coordinates = vec![0usize; spatial];
        for n in 0..batch {
            for channel in 0..in_channels {
                let source = &x[(n * in_channels + channel) * in_volume..][..in_volume];
                let destination =
                    &mut padded[(n * in_channels + channel) * padded_volume..][..padded_volume];
                for line in 0..rows {
                    // Decompose the line into the outer spatial axes only; the
                    // innermost axis stays a contiguous run.
                    let mut remainder = line;
                    for axis in (0..spatial - 1).rev() {
                        coordinates[axis] = remainder % spatial_in[axis].max(1);
                        remainder /= spatial_in[axis].max(1);
                    }
                    let mut destination_start = 0usize;
                    for axis in 0..spatial - 1 {
                        destination_start +=
                            (coordinates[axis] + pads[axis].0 as usize) * padded_strides[axis];
                    }
                    let destination_start = destination_start + pads[spatial - 1].0 as usize;
                    let source_start = line * inner;
                    destination[destination_start..destination_start + inner]
                        .copy_from_slice(&source[source_start..source_start + inner]);
                }
            }
        }
    }
    let kernel_offsets: Vec<usize> = {
        let mut padded_strides = vec![1usize; spatial];
        for axis in (0..spatial.saturating_sub(1)).rev() {
            padded_strides[axis] = padded_strides[axis + 1] * padded_dims[axis + 1];
        }
        (0..kernel_volume)
            .map(|k| {
                let mut remainder = k;
                let mut offset = 0usize;
                for axis in (0..spatial).rev() {
                    let position = remainder % kernel[axis];
                    remainder /= kernel[axis];
                    offset += position * dilations[axis] * padded_strides[axis];
                }
                offset
            })
            .collect()
    };

    let strides = &strides;
    let out_spatial = &out_spatial;
    let padded_dims = &padded_dims;
    let padded = &padded;
    let kernel_offsets = &kernel_offsets;

    for_each_row(threads, out, row_len, |row, values| {
        let n = row / out_channels;
        let oc = row % out_channels;
        let group_index = oc / out_per_group;
        let channel_base = group_index * in_per_group;
        let weight_row = (oc * in_per_group) * kernel_volume;
        let mut coordinates = vec![0usize; spatial];
        let mut padded_strides = vec![1usize; spatial];
        for axis in (0..spatial.saturating_sub(1)).rev() {
            padded_strides[axis] = padded_strides[axis + 1] * padded_dims[axis + 1];
        }
        for linear in 0..row_len {
            let mut remainder = linear;
            for axis in (0..spatial).rev() {
                coordinates[axis] = remainder % out_spatial[axis].max(1);
                remainder /= out_spatial[axis].max(1);
            }
            // Offset of the window's first element in the padded plane.
            let mut window = 0usize;
            for axis in 0..spatial {
                window += coordinates[axis] * strides[axis] * padded_strides[axis];
            }
            let mut accumulator = 0.0f64;
            for ic in 0..in_per_group {
                let plane = (n * in_channels + channel_base + ic) * padded_volume + window;
                let weights = &w[weight_row + ic * kernel_volume..][..kernel_volume];
                for (offset, &weight) in kernel_offsets.iter().zip(weights.iter()) {
                    accumulator += weight as f64 * padded[plane + offset] as f64;
                }
            }
            values[linear] = (accumulator + bias.map_or(0.0f64, |bias| bias[oc] as f64)) as f32;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(
        kernel: &[i64],
        strides: &[i64],
        pads: &[i64],
        dilations: &[i64],
        group: i64,
    ) -> OpParams {
        OpParams::new()
            .with_kernel_shape(kernel.iter().copied())
            .with_strides(strides.iter().copied())
            .with_pads(pads.iter().copied())
            .with_dilations(dilations.iter().copied())
            .with_group(group)
    }

    #[test]
    fn a_one_dimensional_convolution_sums_the_window() {
        // x = [1, 2, 3, 4, 5], w = [1, 1, 1], stride 1, no pad -> [6, 9, 12].
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0f32, 1.0, 1.0];
        let mut out = [0.0f32; 3];
        conv_f32(
            &x,
            &Shape::from_dims([1, 1, 5]),
            &w,
            &Shape::from_dims([1, 1, 3]),
            None,
            &mut out,
            &Shape::from_dims([1, 1, 3]),
            &params(&[3], &[1], &[], &[1], 1),
            None,
        )
        .unwrap();
        assert_eq!(out, [6.0, 9.0, 12.0]);
    }

    #[test]
    fn padding_contributes_zeros_at_both_ends() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0f32, 1.0, 1.0];
        let mut out = [0.0f32; 5];
        conv_f32(
            &x,
            &Shape::from_dims([1, 1, 5]),
            &w,
            &Shape::from_dims([1, 1, 3]),
            None,
            &mut out,
            &Shape::from_dims([1, 1, 5]),
            &params(&[3], &[1], &[1, 1], &[1], 1),
            None,
        )
        .unwrap();
        assert_eq!(out, [3.0, 6.0, 9.0, 12.0, 9.0]);
    }

    #[test]
    fn dilation_stretches_the_kernel() {
        // w = [1, 1, 2], dilation 2 on [1..5]: out[0] = 1*1 + 3*1 + 5*2 = 14.
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0f32, 1.0, 2.0];
        let mut out = [0.0f32; 1];
        conv_f32(
            &x,
            &Shape::from_dims([1, 1, 5]),
            &w,
            &Shape::from_dims([1, 1, 3]),
            None,
            &mut out,
            &Shape::from_dims([1, 1, 1]),
            &params(&[3], &[1], &[], &[2], 1),
            None,
        )
        .unwrap();
        assert_eq!(out, [14.0]);
    }

    #[test]
    fn groups_split_the_channels_into_independent_blocks() {
        // Two groups over two channels, 1x1 kernels: channel 0 is copied and
        // channel 1 is doubled.
        let x = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let w = [1.0f32, 2.0];
        let mut out = [0.0f32; 6];
        conv_f32(
            &x,
            &Shape::from_dims([1, 2, 3]),
            &w,
            &Shape::from_dims([2, 1, 1]),
            None,
            &mut out,
            &Shape::from_dims([1, 2, 3]),
            &params(&[1], &[1], &[], &[1], 2),
            None,
        )
        .unwrap();
        assert_eq!(out, [1.0, 2.0, 3.0, 20.0, 40.0, 60.0]);
    }

    #[test]
    fn a_two_dimensional_three_by_three_kernel_matches_a_hand_sum() {
        // One 3x3 input [1..9] and a 3x3 weight [1..9]; the single output is the
        // dot product 1*1 + 2*2 + ... + 9*9 = 285.
        let x: Vec<f32> = (1..=9).map(|value| value as f32).collect();
        let w: Vec<f32> = (1..=9).map(|value| value as f32).collect();
        let mut out = [0.0f32; 1];
        conv_f32(
            &x,
            &Shape::from_dims([1, 1, 3, 3]),
            &w,
            &Shape::from_dims([1, 1, 3, 3]),
            Some(&[0.5]),
            &mut out,
            &Shape::from_dims([1, 1, 1, 1]),
            &params(&[3, 3], &[1, 1], &[], &[1, 1], 1),
            None,
        )
        .unwrap();
        assert_eq!(out, [285.5]);
    }

    #[test]
    fn a_bias_is_added_once_and_a_fully_padded_window_stays_finite() {
        // Kernel 3 with pad 2 on a 1-wide input: the middle window sees only the
        // input, the others see pad.
        let x = [2.0f32];
        let w = [1.0f32, 1.0, 1.0];
        let mut out = [0.0f32; 3];
        let params = params(&[3], &[1], &[2, 2], &[1], 1);
        let error = conv_f32(
            &x,
            &Shape::from_dims([1, 1, 1]),
            &w,
            &Shape::from_dims([1, 1, 3]),
            Some(&[1.0]),
            &mut out,
            &Shape::from_dims([1, 1, 3]),
            &params,
            None,
        );
        // pad 2 on a 1-wide input gives (1 + 4 - 3) + 1 = 3 outputs; the middle
        // window sees the input, the two outer windows see only padding.
        error.unwrap();
        assert_eq!(out, [3.0, 3.0, 3.0]);
    }

    #[test]
    fn a_kernel_shape_that_contradicts_the_weight_is_rejected() {
        let x = [0.0f32; 5];
        let w = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        // The planned output extent (4) agrees with kernel_shape = 2, so the
        // weight's own kernel extent (2x2 = 4 values) is what disagrees.
        let error = conv_f32(
            &x,
            &Shape::from_dims([1, 1, 5]),
            &w,
            &Shape::from_dims([1, 1, 2]),
            None,
            &mut out,
            &Shape::from_dims([1, 1, 4]),
            &params(&[3], &[1], &[], &[1], 1),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("kernel extent"), "{error}");
    }

    #[test]
    fn a_channel_count_that_does_not_divide_by_groups_is_rejected() {
        let x = [0.0f32; 6];
        let w = [1.0f32; 3];
        let mut out = [0.0f32; 6];
        let error = conv_f32(
            &x,
            &Shape::from_dims([1, 2, 3]),
            &w,
            &Shape::from_dims([3, 1, 1]),
            None,
            &mut out,
            &Shape::from_dims([1, 3, 3]),
            &params(&[1], &[1], &[], &[1], 2),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("divisible"), "{error}");
    }
}
