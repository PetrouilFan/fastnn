//! Pooling: `MaxPool`, `AveragePool`, `GlobalAveragePool`.
//!
//! # The window rule
//!
//! For a spatial axis, output coordinate `o` and kernel position `k` address
//! input coordinate `o * stride - pad_begin + k * dilation`, exactly as
//! [`crate::inference::runtime::kernels::window_output`] assumes when it checks
//! the planned extent. A position outside `[0, extent)` is padding:
//!
//! - `MaxPool` ignores it; a window with no valid position yields
//!   `f32::NEG_INFINITY` (the identity of a maximum), never a read outside the
//!   input;
//! - `AveragePool` sums only valid positions. Its denominator counts valid
//!   positions when `count_include_pad` is false, and every position inside the
//!   *padded* extent `[-pad_begin, extent + pad_end)` when it is true, which is
//!   what ONNX's `count_include_pad` means. A window with an empty denominator
//!   produces `0.0` rather than a division by zero.
//!
//! # `ceil_mode`
//!
//! `ceil_mode` rounds a window extent up instead of down and then drops a last
//! window that starts inside the padding. The extent arithmetic is the shape
//! pass's, so the kernel writes exactly the planned geometry.
//!
//! # `GlobalAveragePool`
//!
//! The mean over every spatial axis, producing `[N, C, 1, ...]`. It takes no
//! attributes, and the plan's output extent is `[N, C, 1, ...]`; the kernel
//! checks both.

use crate::inference::backend::OpParams;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::runtime::kernels::{
    arity, check_input_len, check_output_len, extents, for_each_row, input_f32, output_f32,
    product, require_extent, resolve_pads, window_axes, window_output,
};
use crate::inference::runtime::SessionThreads;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

/// Execute `MaxPool`, `AveragePool` or `GlobalAveragePool` from views.
pub(crate) fn run(
    op: OpId,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    arity(op, inputs.len(), outputs.len(), 1, 1)?;
    let x = input_f32(op, &inputs[0])?;
    let x_shape = inputs[0].shape().clone();
    let out_shape = outputs[0].shape().clone();
    let out = output_f32(op, &mut outputs[0])?;
    if op == OpId::GlobalAveragePool {
        return global_average_f32(x, &x_shape, out, &out_shape, threads);
    }
    window_pool_f32(op, x, &x_shape, out, &out_shape, params, threads)
}

/// `[N, C, spatial...]` -> `[N, C, 1, ...]`, the mean over the spatial axes.
fn global_average_f32(
    x: &[f32],
    x_shape: &Shape,
    out: &mut [f32],
    out_shape: &Shape,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let op = OpId::GlobalAveragePool;
    let rank = x_shape.rank();
    if rank < 3 {
        return Err(Fn3Error::InvalidShape {
            context: "runtime global_average_pool",
            detail: format!(
                "op {op} needs an [N, C, spatial...] operand of rank 3 or more, found {x_shape}"
            ),
        });
    }
    let dims = extents(x_shape, "runtime global_average_pool input")?;
    let out_dims = extents(out_shape, "runtime global_average_pool output")?;
    if out_dims.len() != rank {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("a rank-{rank} output like {x_shape}"),
            actual: format!("{out_shape}"),
        });
    }
    for axis in 2..rank {
        require_extent(op, "global_average_pool output", axis, 1, out_dims[axis])?;
    }
    let (batch, channels) = (dims[0], dims[1]);
    let spatial = product(&dims[2..], "runtime global_average_pool spatial")?;
    let elements = product(&dims, "runtime global_average_pool input")?;
    check_input_len(op, x, elements, x_shape, "input")?;
    check_output_len(op, out, batch * channels, out_shape)?;
    if spatial == 0 {
        return Err(Fn3Error::InvalidShape {
            context: "runtime global_average_pool",
            detail: format!("op {op} needs a non-empty spatial extent, found {x_shape}"),
        });
    }

    for_each_row(threads, out, 1, |row, values| {
        let plane = row * spatial;
        let mut sum = 0.0f32;
        for &value in &x[plane..plane + spatial] {
            sum += value;
        }
        values[0] = sum / spatial as f32;
        Ok(())
    })
}

/// Windowed max or average pooling.
fn window_pool_f32(
    op: OpId,
    x: &[f32],
    x_shape: &Shape,
    out: &mut [f32],
    out_shape: &Shape,
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let rank = x_shape.rank();
    if rank < 3 {
        return Err(Fn3Error::InvalidShape {
            context: "runtime pooling",
            detail: format!(
                "op {op} needs an [N, C, spatial...] operand of rank 3 or more, found {x_shape}"
            ),
        });
    }
    if out_shape.rank() != rank {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("a rank-{rank} output like {x_shape}"),
            actual: format!("{out_shape}"),
        });
    }
    let spatial = rank - 2;
    let (kernel, strides, dilations) = window_axes(op, params, spatial)?;
    let dims = extents(x_shape, "runtime pooling input")?;
    let out_dims = extents(out_shape, "runtime pooling output")?;
    let (batch, channels) = (dims[0], dims[1]);
    if out_dims[0] != batch || out_dims[1] != channels {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("an output [{batch}, {channels}, ...] like {x_shape}"),
            actual: format!("{out_shape}"),
        });
    }
    let spatial_in = &dims[2..];
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
            params.ceil_mode,
        )?;
        require_extent(op, "pool output", axis, computed, out_dims[axis + 2])?;
        out_spatial.push(computed);
    }
    let elements = product(&dims, "runtime pooling input")?;
    check_input_len(op, x, elements, x_shape, "input")?;
    let row_len = product(&out_spatial, "runtime pooling output row")?;
    check_output_len(op, out, batch * channels * row_len, out_shape)?;

    let is_max = op == OpId::MaxPool;
    let kernel_volume = product(&kernel, "runtime pooling kernel volume")?;
    let in_row_len = product(spatial_in, "runtime pooling input row")?;

    for_each_row(threads, out, row_len, |row, values| {
        let n = row / channels;
        let c = row % channels;
        let plane = (n * channels + c) * in_row_len;
        let mut out_coordinates = vec![0usize; spatial];
        let mut in_coordinates = vec![0usize; spatial];
        for linear in 0..row_len {
            let mut remainder = linear;
            for axis in (0..spatial).rev() {
                out_coordinates[axis] = remainder % out_spatial[axis].max(1);
                remainder /= out_spatial[axis].max(1);
            }
            let mut accumulator = if is_max { f32::NEG_INFINITY } else { 0.0 };
            let mut denominator = 0usize;
            for k in 0..kernel_volume {
                let mut k_remainder = k;
                let mut valid = true;
                let mut inside_padded = true;
                for axis in (0..spatial).rev() {
                    let position = out_coordinates[axis] as i64 * strides[axis] as i64
                        - pads[axis].0
                        + (k_remainder % kernel[axis]) as i64 * dilations[axis] as i64;
                    k_remainder /= kernel[axis];
                    if position < -pads[axis].0
                        || position >= spatial_in[axis] as i64 + pads[axis].1
                    {
                        inside_padded = false;
                    }
                    if position < 0 || position >= spatial_in[axis] as i64 {
                        valid = false;
                    } else {
                        in_coordinates[axis] = position as usize;
                    }
                }
                if is_max {
                    if !valid {
                        continue;
                    }
                    let mut index = 0usize;
                    for axis in 0..spatial {
                        index = index * spatial_in[axis] + in_coordinates[axis];
                    }
                    accumulator = accumulator.max(x[plane + index]);
                } else {
                    if params.count_include_pad {
                        if inside_padded {
                            denominator += 1;
                        }
                    } else if valid {
                        denominator += 1;
                    }
                    if valid {
                        let mut index = 0usize;
                        for axis in 0..spatial {
                            index = index * spatial_in[axis] + in_coordinates[axis];
                        }
                        accumulator += x[plane + index];
                    }
                }
            }
            values[linear] = if is_max {
                accumulator
            } else if denominator == 0 {
                0.0
            } else {
                accumulator / denominator as f32
            };
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_pool(
        op: OpId,
        x: &[f32],
        x_dims: &[u64],
        out: &mut [f32],
        out_dims: &[u64],
        params: &OpParams,
    ) {
        let x_shape = Shape::from_dims(x_dims.iter().copied());
        let out_shape = Shape::from_dims(out_dims.iter().copied());
        if op == OpId::GlobalAveragePool {
            global_average_f32(x, &x_shape, out, &out_shape, None).unwrap();
        } else {
            window_pool_f32(op, x, &x_shape, out, &out_shape, params, None).unwrap();
        }
    }

    #[test]
    fn max_pool_takes_the_window_maximum() {
        let x: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let mut out = [0.0f32; 4];
        run_pool(
            OpId::MaxPool,
            &x,
            &[1, 1, 4, 4],
            &mut out,
            &[1, 1, 2, 2],
            &OpParams::new()
                .with_kernel_shape([2, 2])
                .with_strides([2, 2]),
        );
        assert_eq!(out, [6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn max_pool_ignores_the_padding_and_keeps_the_edge_value() {
        // 2x2 input, 2x2 window, stride 1, pad 1 -> 3x3. A window that is mostly
        // padding still reports the one real value it sees, and a window that is
        // entirely padding reports -inf rather than reading out of bounds.
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 9];
        run_pool(
            OpId::MaxPool,
            &x,
            &[1, 1, 2, 2],
            &mut out,
            &[1, 1, 3, 3],
            &OpParams::new()
                .with_kernel_shape([2, 2])
                .with_strides([1, 1])
                .with_pads([1, 1, 1, 1]),
        );
        assert_eq!(out, [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 4.0, 4.0]);
    }

    #[test]
    fn average_pool_counts_valid_cells_unless_padding_is_included() {
        // 3x3 input 1..9, 2x2 window, stride 2, pad 1 -> 2x2 output.
        let x: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let base = OpParams::new()
            .with_kernel_shape([2, 2])
            .with_strides([2, 2])
            .with_pads([1, 1, 1, 1]);
        let mut out = [0.0f32; 4];
        run_pool(
            OpId::AveragePool,
            &x,
            &[1, 1, 3, 3],
            &mut out,
            &[1, 1, 2, 2],
            &base.clone(),
        );
        assert_eq!(out, [1.0, 2.5, 5.5, 7.0]);

        let mut out = [0.0f32; 4];
        run_pool(
            OpId::AveragePool,
            &x,
            &[1, 1, 3, 3],
            &mut out,
            &[1, 1, 2, 2],
            &base.with_count_include_pad(true),
        );
        assert_eq!(out, [0.25, 1.25, 2.75, 7.0]);
    }

    #[test]
    fn ceil_mode_adds_the_trailing_partial_window() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 3];
        run_pool(
            OpId::MaxPool,
            &x,
            &[1, 1, 5],
            &mut out,
            &[1, 1, 3],
            &OpParams::new()
                .with_kernel_shape([2])
                .with_strides([2])
                .with_ceil_mode(true),
        );
        assert_eq!(out, [2.0, 4.0, 5.0]);
    }

    #[test]
    fn global_average_pool_collapses_every_spatial_axis() {
        let x = [1.0f32, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
        let mut out = [0.0f32; 2];
        run_pool(
            OpId::GlobalAveragePool,
            &x,
            &[1, 2, 2, 2],
            &mut out,
            &[1, 2, 1, 1],
            &OpParams::new(),
        );
        assert_eq!(out, [4.0, 5.0]);
    }
}
