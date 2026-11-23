//! `Reshape`, `Split` and `Slice`: the existing CPU kernels, called with folded
//! constants.
//!
//! All three read their constant operands (`shape`, `split`, `starts`, `ends`,
//! `axes`, `steps`) as tensors in ONNX, and the typed IR also folds them into
//! [`OpParams`] so shape inference can read them without a tensor payload. This
//! module is therefore only an adapter:
//!
//! - when the node still carries the constant operand, the view is passed
//!   straight to [`transform::split`] / [`transform::slice`];
//! - when the IR folded it, a temporary `i64` slice over the parameter vector is
//!   bound as a host-bytes view and the *same* kernel runs, so there is one
//!   implementation of slicing and splitting in the crate.
//!
//! An `i64` vector is 8-byte aligned, so binding it as a view is a checked cast,
//! never a copy. The temporary exists for the duration of one segment and is
//! only built for a model whose exporter folded the operand (YOLOv8's equal
//! `Split` sizes take the attribute path, which needs no temporary at all).

use crate::inference::backend::cpu::kernels::transform;
use crate::inference::backend::OpParams;
use crate::inference::device::DeviceId;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMeta, TensorMut, TensorView};

/// Execute `Split` or `Slice` from views.
pub(crate) fn run(
    op: OpId,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Result<(), Fn3Error> {
    match op {
        OpId::Reshape => reshape(inputs, outputs, params),
        OpId::Split => split(inputs, outputs, params),
        OpId::Slice => slice(inputs, outputs, params),
        other => Err(Fn3Error::UnsupportedOp {
            op: other,
            backend: "fn3-runtime-kernels",
            reason: "not a Split or Slice id".to_string(),
        }),
    }
}

fn reshape(
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let op = OpId::Reshape;
    if inputs.is_empty() || inputs.len() > 2 || outputs.len() != 1 {
        return Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs: 2,
            actual_inputs: inputs.len(),
            expected_outputs: 1,
            actual_outputs: outputs.len(),
        });
    }
    if inputs.len() == 2 && params.axes.is_empty() {
        return transform::reshape(op, &inputs[0], &inputs[1], &mut outputs[0]);
    }
    // Folded form: the target shape is in `OpParams::axes`, including a `-1`
    // extent that the kernel resolves from the input's element count.
    let target = params.axes.clone();
    if target.is_empty() {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: "fn3-runtime-kernels",
            reason: "Reshape needs a target shape: either the i64 operand or \
                     OpParams::axes"
                .to_string(),
        });
    }
    let meta = i64_vector_meta(target.len())?;
    let view = TensorView::from_host_bytes(bytemuck::cast_slice(&target), meta)?;
    transform::reshape(op, &inputs[0], &view, &mut outputs[0])
}

fn split(
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let op = OpId::Split;
    if inputs.is_empty() || inputs.len() > 2 || outputs.is_empty() {
        return Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs: 1,
            actual_inputs: inputs.len(),
            expected_outputs: 1,
            actual_outputs: outputs.len(),
        });
    }
    if let Some(sizes) = inputs.get(1) {
        return transform::split(op, &inputs[0], Some(sizes), outputs, params);
    }
    if params.split_sizes.is_empty() {
        return transform::split(op, &inputs[0], None, outputs, params);
    }
    let sizes = params.split_sizes.clone();
    if sizes.len() != outputs.len() {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!(
                "{} split sizes for {} outputs",
                outputs.len(),
                outputs.len()
            ),
            actual: format!("{} sizes", sizes.len()),
        });
    }
    let meta = i64_vector_meta(sizes.len())?;
    let view = TensorView::from_host_bytes(bytemuck::cast_slice(&sizes), meta)?;
    transform::split(op, &inputs[0], Some(&view), outputs, params)
}

fn slice(
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let op = OpId::Slice;
    if inputs.is_empty() || inputs.len() > 5 || outputs.len() != 1 {
        return Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs: 1,
            actual_inputs: inputs.len(),
            expected_outputs: 1,
            actual_outputs: outputs.len(),
        });
    }
    if inputs.len() >= 3 {
        return transform::slice(
            op,
            &inputs[0],
            &inputs[1],
            &inputs[2],
            inputs.get(3),
            inputs.get(4),
            &mut outputs[0],
        );
    }
    // Folded form: starts/ends are required (the shape pass reports their
    // absence), axes and steps are optional and default exactly as ONNX says.
    let starts = params.slice_starts.clone();
    let ends = params.slice_ends.clone();
    let axes = params.axes.clone();
    let steps = params.slice_steps.clone();
    if starts.is_empty() || ends.is_empty() {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: "fn3-runtime-kernels",
            reason: "Slice needs constant starts and ends; the importer folds them into \
                     OpParams::slice_starts and OpParams::slice_ends"
                .to_string(),
        });
    }
    let starts_meta = i64_vector_meta(starts.len())?;
    let ends_meta = i64_vector_meta(ends.len())?;
    let starts_view = TensorView::from_host_bytes(bytemuck::cast_slice(&starts), starts_meta)?;
    let ends_view = TensorView::from_host_bytes(bytemuck::cast_slice(&ends), ends_meta)?;
    let axes_view = if axes.is_empty() {
        None
    } else {
        let meta = i64_vector_meta(axes.len())?;
        Some(TensorView::from_host_bytes(
            bytemuck::cast_slice(&axes),
            meta,
        )?)
    };
    let steps_view = if steps.is_empty() {
        None
    } else {
        let meta = i64_vector_meta(steps.len())?;
        Some(TensorView::from_host_bytes(
            bytemuck::cast_slice(&steps),
            meta,
        )?)
    };
    transform::slice(
        op,
        &inputs[0],
        &starts_view,
        &ends_view,
        axes_view.as_ref(),
        steps_view.as_ref(),
        &mut outputs[0],
    )
}

/// Contiguous metadata for a temporary `[len]` `i64` vector.
fn i64_vector_meta(len: usize) -> Result<TensorMeta, Fn3Error> {
    let extents = u64::try_from(len).map_err(|_| Fn3Error::ShapeOverflow {
        context: "runtime Slice/Split constants",
        axis: None,
        detail: format!("{len} constants does not fit in u64"),
    })?;
    TensorMeta::contiguous(Shape::from_dims([extents]), DType::I64, DeviceId::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::device::DeviceId;
    use crate::inference::tensor::Tensor;

    fn tensor(values: &[f32], dims: &[u64]) -> Tensor {
        Tensor::from_f32(Shape::from_dims(dims.iter().copied()), values).unwrap()
    }

    fn zeros(dims: &[u64]) -> Tensor {
        Tensor::zeros(
            Shape::from_dims(dims.iter().copied()),
            DType::F32,
            DeviceId::Cpu,
        )
        .unwrap()
    }

    #[test]
    fn a_folded_split_uses_the_parameter_sizes() {
        // [2, 3] split into [1, 2] along the trailing axis via OpParams only.
        let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mut first = zeros(&[2, 1]);
        let mut second = zeros(&[2, 2]);
        let view = x.as_view().unwrap();
        let mut outputs = [first.as_mut_view().unwrap(), second.as_mut_view().unwrap()];
        let params = OpParams::new().with_axes([1]).with_split_sizes([1, 2]);
        run(
            OpId::Split,
            std::slice::from_ref(&view),
            &mut outputs,
            &params,
        )
        .unwrap();
        drop(outputs);
        assert_eq!(first.as_f32_slice().unwrap(), &[1.0, 4.0]);
        assert_eq!(second.as_f32_slice().unwrap(), &[2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn a_folded_slice_uses_the_parameter_bounds() {
        let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mut out = zeros(&[2, 2]);
        let view = x.as_view().unwrap();
        let mut out_view = out.as_mut_view().unwrap();
        let params = OpParams::new().with_axes([1]).with_slice([1], [3], [1]);
        run(
            OpId::Slice,
            std::slice::from_ref(&view),
            std::slice::from_mut(&mut out_view),
            &params,
        )
        .unwrap();
        drop(out_view);
        assert_eq!(out.as_f32_slice().unwrap(), &[2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn a_slice_with_a_step_reverses_and_strides() {
        let x = tensor(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]);
        let mut out = zeros(&[2]);
        let view = x.as_view().unwrap();
        let mut out_view = out.as_mut_view().unwrap();
        let params = OpParams::new().with_slice([3], [0], [-2]);
        run(
            OpId::Slice,
            std::slice::from_ref(&view),
            std::slice::from_mut(&mut out_view),
            &params,
        )
        .unwrap();
        drop(out_view);
        assert_eq!(out.as_f32_slice().unwrap(), &[3.0, 1.0]);
    }
}
