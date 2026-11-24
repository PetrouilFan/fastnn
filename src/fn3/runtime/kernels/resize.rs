//! `Resize`: nearest and linear resampling of a 4-D NCHW tensor.
//!
//! # Coordinate transformation
//!
//! For an output axis of extent `O` over an input axis of extent `I` with scale
//! `s = O / I` (or the constant scale the model supplied), output index `i` maps
//! to the source coordinate `x(i)`:
//!
//! | `coordinate_transformation_mode` | `x(i)` |
//! |---|---|
//! | `half_pixel` (default) | `(i + 0.5) / s - 0.5` |
//! | `pytorch_half_pixel` | `(i + 0.5) / s - 0.5` when `O > 1`, else `0` |
//! | `align_corners` | `i * (I - 1) / (O - 1)` when `O > 1`, else `0` |
//! | `asymmetric` | `i / s` |
//!
//! # Nearest
//!
//! `nearest_mode` turns `x` into an index: `round_prefer_floor` is
//! `ceil(x - 0.5)` (a tie goes down), `round_prefer_ceil` is `floor(x + 0.5)`, and
//! `floor`/`ceil` are themselves. The index is then clamped to `[0, I - 1]`, which
//! is ONNX's rule for a coordinate outside the input.
//!
//! # Linear
//!
//! Two taps per spatial axis: `x0 = floor(x)`, `x1 = x0 + 1`, weights `1 - frac`
//! and `frac`, both indices clamped. The 2-D result is the tensor product of the
//! two 1-D interpolations, evaluated in the order `(y0, x0), (y0, x1), (y1, x0),
//! (y1, x1)` so the sum is deterministic. A row partner outside the input
//! contributes the clamped edge value, which is what ONNX specifies.
//!
//! # Scales or sizes
//!
//! The importer folds the constant scales/sizes operand into `OpParams`; the
//! kernel accepts either (not both), checks the planned output extent against the
//! rule, and rejects `cubic` and `tf_crop_and_resize`, which have no kernel.
//!
//! # Cost
//!
//! The width axis mapping does not depend on the row, so it is resolved once
//! per forward into a table: a run-length form for nearest (`output columns
//! start..end read input column source`) and a two-tap form for linear. The row
//! loop then writes each output row from that table, and rows are the unit of
//! parallelism. The value of an output element is unchanged: the tables hold
//! exactly the indices and weights the direct form computes, and the sums are
//! evaluated in the same order.

use crate::inference::backend::{CoordinateTransformationMode, NearestMode, OpParams, ResizeMode};
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::runtime::kernels::{
    check_input_len, check_output_len, extents, for_each_row, input_f32, output_f32, product,
    require_extent,
};
use crate::inference::runtime::SessionThreads;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

/// Execute `Resize` from views.
pub(crate) fn run(
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let op = OpId::Resize;
    if inputs.is_empty() || inputs.len() > 4 || outputs.len() != 1 {
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
    resize_f32(x, &x_shape, out, &out_shape, params, threads)
}

/// The kernel: contiguous NCHW operands, explicit shapes.
pub(crate) fn resize_f32(
    x: &[f32],
    x_shape: &Shape,
    out: &mut [f32],
    out_shape: &Shape,
    params: &OpParams,
    threads: Option<&SessionThreads>,
) -> Result<(), Fn3Error> {
    let op = OpId::Resize;
    if x_shape.rank() != 4 || out_shape.rank() != 4 {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: "fn3-runtime-kernels",
            reason: format!(
                "the Resize kernel implements the 4-D [N, C, H, W] form; found {x_shape} -> \
                 {out_shape}"
            ),
        });
    }
    if params.resize_mode == ResizeMode::Cubic {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: "fn3-runtime-kernels",
            reason: "cubic Resize has no kernel in this revision; nearest and linear do"
                .to_string(),
        });
    }
    if params.coordinate_transformation_mode == CoordinateTransformationMode::TfCropAndResize {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: "fn3-runtime-kernels",
            reason: "tf_crop_and_resize is an ROI crop, which this runtime does not represent"
                .to_string(),
        });
    }
    let input = extents(x_shape, "runtime resize input")?;
    let planned = extents(out_shape, "runtime resize output")?;
    let mut scales = [0f32; 4];
    if !params.resize_sizes.is_empty() && !params.resize_scales.is_empty() {
        return Err(Fn3Error::InvalidShape {
            context: "runtime resize",
            detail: "op resize carries both scales and sizes".to_string(),
        });
    }
    if !params.resize_sizes.is_empty() {
        if params.resize_sizes.len() != 4 {
            return Err(Fn3Error::InvalidShape {
                context: "runtime resize",
                detail: format!(
                    "op resize has {} sizes for a rank-4 operand",
                    params.resize_sizes.len()
                ),
            });
        }
        for axis in 0..4 {
            let size = usize::try_from(params.resize_sizes[axis])
                .ok()
                .filter(|&size| size > 0)
                .ok_or_else(|| Fn3Error::InvalidShape {
                    context: "runtime resize",
                    detail: format!(
                        "op resize output extent {} on axis {axis} must be positive",
                        params.resize_sizes[axis]
                    ),
                })?;
            require_extent(op, "resize output", axis, size, planned[axis])?;
            scales[axis] = size as f32 / input[axis] as f32;
        }
    } else if !params.resize_scales.is_empty() {
        if params.resize_scales.len() != 4 {
            return Err(Fn3Error::InvalidShape {
                context: "runtime resize",
                detail: format!(
                    "op resize has {} scales for a rank-4 operand",
                    params.resize_scales.len()
                ),
            });
        }
        for axis in 0..4 {
            let scale = params.resize_scales[axis];
            if !scale.is_finite() || scale <= 0.0 {
                return Err(Fn3Error::InvalidShape {
                    context: "runtime resize",
                    detail: format!(
                        "op resize scale {scale} on axis {axis} must be finite and positive"
                    ),
                });
            }
            let extent = (input[axis] as f64 * f64::from(scale)).floor();
            if extent > usize::MAX as f64 {
                return Err(Fn3Error::ShapeOverflow {
                    context: "runtime resize extent",
                    axis: Some(axis),
                    detail: format!("{extent} does not fit in usize"),
                });
            }
            require_extent(op, "resize output", axis, extent as usize, planned[axis])?;
            scales[axis] = scale;
        }
    } else {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: "fn3-runtime-kernels",
            reason: "Resize needs constant scales or constant sizes in OpParams; the importer \
                     folds them there"
                .to_string(),
        });
    }
    for axis in 0..2 {
        require_extent(op, "resize output", axis, input[axis], planned[axis])?;
    }

    let elements = product(&input, "runtime resize input")?;
    check_input_len(op, x, elements, x_shape, "input")?;
    let out_elements = product(&planned, "runtime resize output")?;
    check_output_len(op, out, out_elements, out_shape)?;

    let (in_h, in_w) = (input[2], input[3]);
    let (out_h, out_w) = (planned[2], planned[3]);
    let mode = params.resize_mode;
    let transform = params.coordinate_transformation_mode;
    let nearest_mode = params.nearest_mode;

    // The column tables depend on the width axis alone, so they are built once
    // per forward. Building them inside the row loop made every output element
    // pay for a coordinate transform, which is most of the 1.51 ms the two
    // YOLOv8n upsamples cost at one thread.
    //
    // One row is one (n, c, y) line: [out_w] elements.
    match mode {
        ResizeMode::Nearest => {
            let runs = nearest_runs(transform, nearest_mode, in_w, out_w, scales[3]);
            for_each_row(threads, out, out_w, |row, values| {
                let plane_in = (row / out_h) * in_h * in_w;
                let y = row % out_h;
                let source_y = coordinate(transform, y, in_h, out_h, scales[2]);
                let sy = nearest(nearest_mode, source_y, in_h);
                let line = &x[plane_in + sy * in_w..plane_in + sy * in_w + in_w];
                // Walk the run cursor rather than building one sub-slice per
                // run: a 64x40x40 upsample is 20 runs of two per row, and the
                // per-run slice setup was most of the kernel's cost.
                let Some(first) = runs.first() else {
                    return Ok(());
                };
                let mut run = 0usize;
                let mut end = first.1;
                let mut value = line[first.2];
                for (index, slot) in values.iter_mut().enumerate() {
                    if index == end {
                        run += 1;
                        let next = runs[run];
                        end = next.1;
                        value = line[next.2];
                    }
                    *slot = value;
                }
                Ok(())
            })
        }
        ResizeMode::Linear => {
            let columns = linear_columns(transform, in_w, out_w, scales[3]);
            for_each_row(threads, out, out_w, |row, values| {
                let plane_in = (row / out_h) * in_h * in_w;
                let y = row % out_h;
                let source_y = coordinate(transform, y, in_h, out_h, scales[2]);
                let y0 = source_y.floor();
                let wy = source_y - y0;
                let sy0 = clamp_index(y0, in_h);
                let sy1 = clamp_index(y0 + 1.0, in_h);
                let top = &x[plane_in + sy0 * in_w..plane_in + sy0 * in_w + in_w];
                let bottom = &x[plane_in + sy1 * in_w..plane_in + sy1 * in_w + in_w];
                for (value, &(sx0, sx1, w0, w1)) in values.iter_mut().zip(&columns) {
                    let upper = top[sx0] * w0 + top[sx1] * w1;
                    let lower = bottom[sx0] * w0 + bottom[sx1] * w1;
                    *value = upper * (1.0 - wy) + lower * wy;
                }
                Ok(())
            })
        }
        ResizeMode::Cubic => unreachable!("cubic was rejected above"),
    }
}

/// The run-length form of the nearest mapping on the width axis.
///
/// Each entry is `(start, end, source_column)`: output columns `start..end` all
/// read input column `source_column`. The mapping is monotone (a coordinate
/// transform that is non-decreasing plus a clamp), so consecutive equal columns
/// collapse into one run and a nearest upsample becomes a handful of slice
/// fills instead of one gather per output element.
fn nearest_runs(
    transform: CoordinateTransformationMode,
    mode: NearestMode,
    in_w: usize,
    out_w: usize,
    scale: f32,
) -> Vec<(usize, usize, usize)> {
    let mut runs: Vec<(usize, usize, usize)> = Vec::with_capacity(out_w);
    for index in 0..out_w {
        let sx = nearest(mode, coordinate(transform, index, in_w, out_w, scale), in_w);
        match runs.last_mut() {
            Some(last) if last.2 == sx => last.1 = index + 1,
            _ => runs.push((index, index + 1, sx)),
        }
    }
    runs
}

/// The two-tap form of the linear mapping on the width axis.
///
/// Each entry is `(left, right, left_weight, right_weight)` for one output
/// column: the clamped input columns and the weights the row loop applies. The
/// weights are the same `1.0 - frac` and `frac` the direct form computes, so a
/// value is bit-identical to computing them per element.
fn linear_columns(
    transform: CoordinateTransformationMode,
    in_w: usize,
    out_w: usize,
    scale: f32,
) -> Vec<(usize, usize, f32, f32)> {
    let mut columns: Vec<(usize, usize, f32, f32)> = Vec::with_capacity(out_w);
    for index in 0..out_w {
        let source = coordinate(transform, index, in_w, out_w, scale);
        let left = source.floor();
        let weight = source - left;
        columns.push((
            clamp_index(left, in_w),
            clamp_index(left + 1.0, in_w),
            1.0 - weight,
            weight,
        ));
    }
    columns
}

/// Source coordinate of output index `index` on one axis.
fn coordinate(
    mode: CoordinateTransformationMode,
    index: usize,
    input: usize,
    output: usize,
    scale: f32,
) -> f32 {
    let index = index as f32;
    match mode {
        CoordinateTransformationMode::HalfPixel => (index + 0.5) / scale - 0.5,
        CoordinateTransformationMode::PytorchHalfPixel => {
            if output > 1 {
                (index + 0.5) / scale - 0.5
            } else {
                0.0
            }
        }
        CoordinateTransformationMode::AlignCorners => {
            if output > 1 {
                index * (input - 1) as f32 / (output - 1) as f32
            } else {
                0.0
            }
        }
        CoordinateTransformationMode::Asymmetric => index / scale,
        // Rejected by `resize_f32` before a coordinate is computed.
        CoordinateTransformationMode::TfCropAndResize => 0.0,
    }
}

/// The nearest input index of `coordinate`, clamped to the input extent.
fn nearest(mode: NearestMode, coordinate: f32, input: usize) -> usize {
    let rounded = match mode {
        NearestMode::RoundPreferFloor => (coordinate - 0.5).ceil(),
        NearestMode::RoundPreferCeil => (coordinate + 0.5).floor(),
        NearestMode::Floor => coordinate.floor(),
        NearestMode::Ceil => coordinate.ceil(),
    };
    clamp_index(rounded, input)
}

/// Clamp a coordinate to `[0, extent - 1]`.
fn clamp_index(coordinate: f32, extent: usize) -> usize {
    if coordinate <= 0.0 {
        0
    } else if coordinate >= (extent - 1) as f32 {
        extent - 1
    } else {
        coordinate as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resize(x: &[f32], x_dims: &[u64], out_dims: &[u64], params: &OpParams) -> Vec<f32> {
        let out_len: u64 = out_dims.iter().product();
        let mut out = vec![0.0f32; out_len as usize];
        resize_f32(
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
    fn asymmetric_nearest_doubles_each_pixel() {
        // [[1, 2], [3, 4]] -> [[1, 1, 2, 2], [1, 1, 2, 2], ...] with scale 2.
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let params = OpParams::new()
            .with_resize(
                ResizeMode::Nearest,
                CoordinateTransformationMode::Asymmetric,
                NearestMode::Floor,
            )
            .with_resize_scales([1.0, 1.0, 2.0, 2.0]);
        let out = resize(&x, &[1, 1, 2, 2], &[1, 1, 4, 4], &params);
        assert_eq!(
            out,
            [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0]
        );
    }

    #[test]
    fn half_pixel_linear_averages_neighbours_evenly() {
        // 2x2 -> 4x4 half_pixel, scale 2. The corners keep their value, an edge
        // midpoint is the mean of two neighbours, and the centre is (1+2+3+4)/4.
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let params = OpParams::new()
            .with_resize(
                ResizeMode::Linear,
                CoordinateTransformationMode::HalfPixel,
                NearestMode::RoundPreferFloor,
            )
            .with_resize_sizes([1, 1, 4, 4]);
        let out = resize(&x, &[1, 1, 2, 2], &[1, 1, 4, 4], &params);
        assert_eq!(
            out,
            [
                1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75,
                4.0
            ]
        );
    }

    #[test]
    fn align_corners_keeps_both_ends() {
        // 1-D along W: [1, 2] -> 3 with align_corners gives [1, 1.5, 2].
        let x = [1.0f32, 2.0];
        let params = OpParams::new()
            .with_resize(
                ResizeMode::Linear,
                CoordinateTransformationMode::AlignCorners,
                NearestMode::RoundPreferFloor,
            )
            .with_resize_sizes([1, 1, 1, 3]);
        let out = resize(&x, &[1, 1, 1, 2], &[1, 1, 1, 3], &params);
        assert_eq!(out, [1.0, 1.5, 2.0]);
    }

    #[test]
    fn round_prefer_floor_and_ceil_differ_exactly_on_ties() {
        // scale 2, half_pixel: x(1) = 0.25 -> both round to 0; x(2) = 0.75 ->
        // both round to 1; x(3) = 1.25 -> both round to 1. Use a case with x = 0.5:
        // 1-D input of 2, output of 4, asymmetric, so x(1) = 0.5.
        let x = [10.0f32, 20.0];
        let floor = OpParams::new()
            .with_resize(
                ResizeMode::Nearest,
                CoordinateTransformationMode::Asymmetric,
                NearestMode::RoundPreferFloor,
            )
            .with_resize_sizes([1, 1, 1, 4]);
        let out = resize(&x, &[1, 1, 1, 2], &[1, 1, 1, 4], &floor);
        assert_eq!(out, [10.0, 10.0, 20.0, 20.0]);
        let ceil = floor.with_resize(
            ResizeMode::Nearest,
            CoordinateTransformationMode::Asymmetric,
            NearestMode::RoundPreferCeil,
        );
        let out = resize(&x, &[1, 1, 1, 2], &[1, 1, 1, 4], &ceil);
        assert_eq!(out, [10.0, 20.0, 20.0, 20.0]);
    }

    #[test]
    fn cubic_is_reported_not_approximated() {
        let x = [0.0f32];
        let mut out = [0.0f32; 1];
        let params = OpParams::new()
            .with_resize(
                ResizeMode::Cubic,
                CoordinateTransformationMode::HalfPixel,
                NearestMode::RoundPreferFloor,
            )
            .with_resize_scales([1.0, 1.0, 1.0, 1.0]);
        let error = resize_f32(
            &x,
            &Shape::from_dims([1, 1, 1, 1]),
            &mut out,
            &Shape::from_dims([1, 1, 1, 1]),
            &params,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("cubic"), "{error}");
    }
}
