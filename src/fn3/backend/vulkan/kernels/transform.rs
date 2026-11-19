//! GPU kernels owned by FN3-13 (#59): shape and data movement.
//!
//! # What is registered here
//!
//! | op | kernel | support |
//! |---|---|---|
//! | `Transpose` | `transform.permute` | full |
//! | `Reshape` `Flatten` `Squeeze` `Unsqueeze` | `transform.permute` (identity mode) | full |
//! | `Slice` `Split` `Concat` | `transform.block_copy` | full |
//! | `Expand` | `transform.expand` | full |
//! | `Shape` | `transform.shape` | full |
//! | `ConstantOfShape` | `transform.fill` | full |
//! | `Range` | `transform.range.f32` / `transform.range.i64` | full |
//!
//! `Gather` (the embedding lookup) lives in `embedding.rs`, which owns the
//! table-lookup kernel.
//!
//! # Host-side shape work vs. device-side data work
//!
//! The issue asks for the boundary to be stated, so: the *shape* of every
//! output is resolved on the host, from the output tensor the planner already
//! allocated and from the small integer tensors ONNX uses as shape inputs
//! (`Reshape` shape, `Squeeze`/`Unsqueeze` axes, `Slice` starts/ends/steps,
//! `Split` sizes, `Expand` target, `ConstantOfShape` shape, `Range` scalars).
//! Those payloads are metadata: validating them is exactly the CPU reference's
//! job and re-deriving them on the device would duplicate the rules without
//! making a single output element more GPU-computed. Every output *element* is
//! produced by a dispatch, and `Shape`'s i64 output is written by a dispatch
//! too, so no op is a silent host computation.
//!
//! Element widths moved by the generic kernels are 4 and 8 bytes. A one-byte
//! (`Bool`) tensor is rejected here and named as a #59 gap: core Vulkan cannot
//! write a single byte without the `storageBuffer8BitAccess` feature, which the
//! FN3-11 platform layer deliberately does not enable. `Bool` producers
//! (`Equal`/`Less`/`Greater`, `Cast` to `Bool`) live in `pointwise.rs` and use a
//! four-elements-per-invocation packing.
//!
//! # Attributes
//!
//! `Transpose` reads its permutation from `OpParams::axes` (empty means the
//! ONNX default of reversing the axes). `Squeeze`/`Unsqueeze` accept the axes
//! either as a parameter or as a second input, exactly one of the two.
//! `Flatten`, `Split` and `Concat` take one axis from `OpParams::axes`.

use crate::fn3::backend::check_arity;
use crate::fn3::backend::vulkan::context::BindingRole;
use crate::fn3::backend::vulkan::kernels::pointwise::{
    broadcast_dims, element_count, normalize_axis, output_dims, output_window_bytes,
    require_contiguous_output, require_contiguous_view, require_device, require_no_attributes,
    require_only_axes, require_output_dtype, require_output_shape, require_shader_rank,
    single_axis, view_window_bytes, workgroups, Job, Push, BACKEND, WORKGROUP,
};
use crate::fn3::backend::vulkan::kernels::{grid, BindingSpec, KernelSet, KernelSpec, OpKernel};
use crate::fn3::backend::vulkan::VulkanBackend;
use crate::fn3::backend::{CapabilityFeatures, OpParams, Support};
use crate::fn3::dtype::DType;
use crate::fn3::error::{Fn3Error, Fn3Result};
use crate::fn3::ops::OpId;
use crate::fn3::shape::Shape;
use crate::fn3::tensor::{TensorMut, TensorView};

/// Axis permutation and identity layout copy.
pub const PERMUTE: KernelSpec = KernelSpec {
    name: "transform.permute",
    spirv: include_bytes!("../shaders/transform_permute.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "input"),
        BindingSpec::storage(1, "out"),
    ],
    push_constant_bytes: 112,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32, DType::I32, DType::I64],
    shape: "one contiguous 4- or 8-byte tensor, permuted (Transpose) or copied row-major \
            (Reshape/Flatten/Squeeze/Unsqueeze) into a contiguous output of equal element count",
    grid: grid::elementwise,
};

/// Single-axis block move for `Slice`, `Split` and `Concat`.
pub const BLOCK_COPY: KernelSpec = KernelSpec {
    name: "transform.block_copy",
    spirv: include_bytes!("../shaders/transform_block_copy.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "input"),
        BindingSpec::storage(1, "out"),
    ],
    push_constant_bytes: 36,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32, DType::I32, DType::I64],
    shape: "one contiguous 4- or 8-byte block [outer, axis, inner], moved to a contiguous \
            destination with an axis start/step and an axis offset into the destination",
    grid: grid::elementwise,
};

/// `Expand`: broadcast to a target shape.
pub const EXPAND: KernelSpec = KernelSpec {
    name: "transform.expand",
    spirv: include_bytes!("../shaders/transform_expand.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "input"),
        BindingSpec::storage(1, "out"),
    ],
    push_constant_bytes: 76,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32, DType::I32, DType::I64],
    shape: "one contiguous 4- or 8-byte tensor broadcast right-aligned to a contiguous output \
            shape",
    grid: grid::elementwise,
};

/// `ConstantOfShape`: fill from a scalar parameter.
pub const FILL: KernelSpec = KernelSpec {
    name: "transform.fill",
    spirv: include_bytes!("../shaders/transform_fill.comp.spv"),
    entry_point: "main",
    bindings: &[BindingSpec::storage(0, "out")],
    push_constant_bytes: 16,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32, DType::I32, DType::I64, DType::Bool],
    shape: "a contiguous output of any alpha dtype filled with the single-element value input",
    grid: grid::elementwise,
};

/// `Shape`: write the extents as an i64 tensor.
pub const SHAPE: KernelSpec = KernelSpec {
    name: "transform.shape",
    spirv: include_bytes!("../shaders/transform_shape.comp.spv"),
    entry_point: "main",
    bindings: &[BindingSpec::storage(0, "out")],
    push_constant_bytes: 68,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::I64],
    shape: "a rank-1 contiguous i64 output of length rank",
    grid: grid::elementwise,
};

/// f32 `Range`.
pub const RANGE_F32: KernelSpec = KernelSpec {
    name: "transform.range.f32",
    spirv: include_bytes!("../shaders/transform_range_f32.comp.spv"),
    entry_point: "main",
    bindings: &[BindingSpec::storage(0, "out")],
    push_constant_bytes: 12,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32],
    shape: "a contiguous rank-1 f32 output; value i is start + i * delta",
    grid: grid::elementwise,
};

/// i64 `Range`.
pub const RANGE_I64: KernelSpec = KernelSpec {
    name: "transform.range.i64",
    spirv: include_bytes!("../shaders/transform_range_i64.comp.spv"),
    entry_point: "main",
    bindings: &[BindingSpec::storage(0, "out")],
    push_constant_bytes: 20,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::I64],
    shape: "a contiguous rank-1 i64 output; value i is start + i * delta",
    grid: grid::elementwise,
};

// ---------------------------------------------------------------------------
// Small integer payload readers
// ---------------------------------------------------------------------------

/// The dtype set the generic byte-moving kernels accept.
pub(super) const MOVE_DTYPES: &[DType] = &[DType::F32, DType::I32, DType::I64];

/// The subset gap the layout kernels name in their `Partial` verdict.
pub(super) const WIDTH_GAP: &str =
    "4- and 8-byte elements (F32/I32/I64); a Bool tensor is rejected with UnsupportedOp naming \
     #59";

/// Concrete admission for the layout kernels: every operand is 4 or 8 bytes.
///
/// This is the part of the subset that the plan can decide: the element width
/// is the only operand property the permute, identity-copy and block-copy
/// kernels restrict by dtype. Shape-only restrictions are resolved by the
/// planner before the plan exists, and value-dependent ones (how many axes a
/// `Slice` really names) remain the launch function's job.
pub(super) fn admits_movable(
    _shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
) -> std::result::Result<(), String> {
    match dtypes
        .iter()
        .find(|dtype| !matches!(dtype.size_bytes(), 4 | 8))
    {
        Some(dtype) => Err(format!(
            "{} element(s) are {} byte(s), not 4 or 8",
            dtype,
            dtype.size_bytes()
        )),
        None => Ok(()),
    }
}

/// Concrete admission for `Concat`: moveable widths and at most [`MAX_PIECES`]
/// inputs (one output slot follows the inputs).
pub(super) fn admits_concat(
    shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
) -> std::result::Result<(), String> {
    admits_movable(shapes, dtypes, params)?;
    let inputs = shapes.len().saturating_sub(1);
    if inputs > MAX_PIECES {
        return Err(format!(
            "{inputs} inputs in one concatenation; the kernel records at most {MAX_PIECES}"
        ));
    }
    Ok(())
}

pub(super) fn element_bytes(op: OpId, dtype: DType) -> Fn3Result<u32> {
    let bytes = dtype.size_bytes();
    if bytes == 4 || bytes == 8 {
        return Ok(bytes as u32);
    }
    Err(Fn3Error::UnsupportedOp {
        op,
        backend: BACKEND,
        reason: format!(
            "the FN3-13 (#59) layout kernels move 4- and 8-byte elements; {dtype} is 1 byte and \
             would need `storageBuffer8BitAccess`, which the platform layer does not enable"
        ),
    })
}

/// Read an integer tensor (contiguous `I64` or `I32`) as `i64` values.
pub(super) fn read_i64_values(
    op: OpId,
    view: &TensorView<'_>,
    what: &'static str,
) -> Fn3Result<Vec<i64>> {
    require_contiguous_view(op, view, what)?;
    match view.dtype() {
        DType::I64 => Ok(view.as_i64_slice()?.to_vec()),
        DType::I32 => Ok(view.as_i32_slice()?.iter().map(|&v| i64::from(v)).collect()),
        other => Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: DType::I64,
            actual: other,
        }),
    }
}

/// Read a single-element integer tensor as `i64`.
fn read_i64_scalar(op: OpId, view: &TensorView<'_>, what: &'static str) -> Fn3Result<i64> {
    let values = read_i64_values(op, view, what)?;
    match values.as_slice() {
        [value] => Ok(*value),
        _ => Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{what} to have exactly one element"),
            actual: format!("{} has {} elements", view.shape(), values.len()),
        }),
    }
}

/// Resolve ONNX `Reshape`'s target (`allowzero = 0`), exactly as the CPU
/// reference does.
fn resolve_reshape_shape(op: OpId, input: &Shape, target: &[i64]) -> Fn3Result<Shape> {
    let mut dims = Vec::with_capacity(target.len());
    let mut inferred: Option<usize> = None;
    let mut known: u64 = 1;
    for (position, &value) in target.iter().enumerate() {
        let extent = match value {
            -1 => {
                if inferred.is_some() {
                    return Err(Fn3Error::InvalidShape {
                        context: "vulkan reshape target",
                        detail: format!(
                            "op {op}: {target:?} infers more than one dimension with -1"
                        ),
                    });
                }
                inferred = Some(position);
                dims.push(0);
                continue;
            }
            0 => input
                .dims()
                .get(position)
                .copied()
                .ok_or_else(|| Fn3Error::InvalidShape {
                    context: "vulkan reshape target",
                    detail: format!(
                        "op {op}: {target:?} copies dimension {position} with 0, but the input \
                         {input} has rank {}",
                        input.rank()
                    ),
                })?,
            positive => u64::try_from(positive).map_err(|_| Fn3Error::InvalidShape {
                context: "vulkan reshape target",
                detail: format!("op {op}: {target:?} has a negative dimension other than -1"),
            })?,
        };
        dims.push(extent);
        known = known.checked_mul(extent).ok_or(Fn3Error::ShapeOverflow {
            context: "vulkan reshape target",
            axis: Some(position),
            detail: "known extents exceed u64::MAX".to_string(),
        })?;
    }
    if let Some(position) = inferred {
        let elements = input.numel()?;
        if known == 0 {
            if elements != 0 {
                return Err(Fn3Error::InvalidShape {
                    context: "vulkan reshape target",
                    detail: format!(
                        "op {op}: {target:?} leaves the product of known extents at 0, so the \
                         -1 dimension cannot be inferred from {elements} elements"
                    ),
                });
            }
            dims[position] = 0;
        } else {
            if elements % known != 0 {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("{elements} elements divisible by {known}"),
                    actual: format!("{target:?}"),
                });
            }
            dims[position] = elements / known;
        }
    }
    Ok(Shape::from_dims(dims))
}

/// The axes of a `Squeeze`/`Unsqueeze`, from `params` or the input tensor.
fn axes_from_params_or_input(
    op: OpId,
    params: &OpParams,
    axes_input: Option<&TensorView<'_>>,
) -> Fn3Result<Vec<i64>> {
    match (params.axes.as_slice(), axes_input) {
        ([], None) => Ok(Vec::new()),
        (axes, None) => Ok(axes.to_vec()),
        ([], Some(view)) => read_i64_values(op, view, "axes input"),
        (axes, Some(_)) => Err(Fn3Error::InvalidShape {
            context: "vulkan axes attribute",
            detail: format!(
                "op {op} was given both axes={axes:?} and an axes input tensor; supply exactly one"
            ),
        }),
    }
}

/// Python/ONNX slice bounds for one axis.
#[derive(Debug, Clone, Copy)]
struct AxisSlice {
    start: i64,
    step: i64,
    count: u64,
}

/// Resolve `start`, `end` and `step` for one axis with Python's clamping rules;
/// the same rules as the CPU reference (which is the oracle this mirrors).
fn resolve_axis_slice(
    op: OpId,
    dimension: u64,
    start: i64,
    end: i64,
    step: i64,
) -> Fn3Result<AxisSlice> {
    if step == 0 {
        return Err(Fn3Error::InvalidShape {
            context: "vulkan slice step",
            detail: format!("op {op}: a slice step of 0 is undefined"),
        });
    }
    let dim = i128::from(dimension);
    let step_wide = i128::from(step);
    let (start, count) = if step > 0 {
        let mut resolved_start = i128::from(start);
        let mut resolved_end = i128::from(end);
        if resolved_start < 0 {
            resolved_start += dim;
        }
        if resolved_end < 0 {
            resolved_end += dim;
        }
        let resolved_start = resolved_start.clamp(0, dim);
        let resolved_end = resolved_end.clamp(0, dim);
        let count = if resolved_end > resolved_start {
            (resolved_end - resolved_start + step_wide - 1) / step_wide
        } else {
            0
        };
        (resolved_start, count)
    } else {
        let mut resolved_start = i128::from(start);
        let mut resolved_end = i128::from(end);
        if resolved_start < 0 {
            resolved_start += dim;
        }
        if resolved_end < 0 {
            resolved_end += dim;
        }
        let upper = (dim - 1).max(0);
        let resolved_start = resolved_start.clamp(0, upper);
        let resolved_end = resolved_end.clamp(-1, upper);
        let count = if resolved_start > resolved_end {
            (resolved_start - resolved_end + (-step_wide) - 1) / (-step_wide)
        } else {
            0
        };
        (resolved_start, count)
    };
    let count = u64::try_from(count).map_err(|_| Fn3Error::ShapeOverflow {
        context: "vulkan slice extent",
        axis: None,
        detail: "slice element count does not fit in u64".to_string(),
    })?;
    let start = i64::try_from(start).map_err(|_| Fn3Error::ShapeOverflow {
        context: "vulkan slice start",
        axis: None,
        detail: "resolved start does not fit in i64".to_string(),
    })?;
    Ok(AxisSlice { start, step, count })
}

/// A product of extents, checked.
pub(super) fn checked_product(dims: &[u64]) -> Fn3Result<u64> {
    let mut product = 1u64;
    for (axis, &extent) in dims.iter().enumerate() {
        product = product.checked_mul(extent).ok_or(Fn3Error::ShapeOverflow {
            context: "vulkan kernel extent product",
            axis: Some(axis),
            detail: "extents exceed u64::MAX".to_string(),
        })?;
    }
    Ok(product)
}

/// Read `count` extents from a shape tensor, rejecting negatives.
fn shape_values(op: OpId, view: &TensorView<'_>, what: &'static str) -> Fn3Result<Shape> {
    let values = read_i64_values(op, view, what)?;
    let mut dims = Vec::with_capacity(values.len());
    for (axis, value) in values.into_iter().enumerate() {
        dims.push(u64::try_from(value).map_err(|_| Fn3Error::InvalidShape {
            context: "vulkan kernel shape input",
            detail: format!("op {op}: raw extent {value} on axis {axis} is negative"),
        })?);
    }
    Ok(Shape::from_dims(dims))
}

// ---------------------------------------------------------------------------
// Layout copies
// ---------------------------------------------------------------------------

/// One identity or permuted layout copy.
fn launch_permute(
    backend: &VulkanBackend,
    op: OpId,
    input: &TensorView<'_>,
    out: &mut TensorMut<'_>,
    permutation: Option<Vec<usize>>,
    elements: u64,
) -> Fn3Result<()> {
    let rank = input.shape().rank();
    let width = element_bytes(op, input.dtype())?;
    let count = element_count(op, elements)?;
    let mut perm = [0u32; 8];
    if let Some(permutation) = &permutation {
        for (position, &axis) in permutation.iter().enumerate() {
            perm[position] = u32::try_from(axis).map_err(|_| Fn3Error::InvalidShape {
                context: "vulkan permutation",
                detail: format!("axis {axis} does not fit u32"),
            })?;
        }
    }
    let push = Push::new()
        .u32(count)
        .u32(width)
        .u32(u32::from(permutation.is_some()))
        .u32(rank as u32)
        .dims(perm)
        .dims(output_dims(op, out.shape())?)
        .dims(output_dims(op, input.shape())?)
        .bytes();
    let workgroups = workgroups(&PERMUTE, elements, out.shape().dims())?;
    let mut bytes = vec![0u8; out.meta().byte_len()?];
    let mut job = Job::new(backend, op, &PERMUTE)?;
    let source = job.input(view_window_bytes(input)?, "fn3.transform.in")?;
    let destination = job.output(out.meta().byte_len()? as u64, "fn3.transform.out")?;
    job.step(
        push,
        workgroups,
        vec![
            (0, source, BindingRole::Input),
            (1, destination, BindingRole::Output),
        ],
    );
    job.run(&mut [&mut bytes])?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}

/// `Transpose`: the ONNX default reverses the axes when `axes` is empty.
pub fn launch_transpose(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Transpose;
    check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
    require_only_axes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let input = &inputs[0];
    let out = &mut outputs[0];
    require_contiguous_view(op, input, "input")?;
    require_contiguous_output(op, out, "output")?;
    let rank = input.shape().rank();
    require_shader_rank(op, rank)?;
    let permutation: Vec<usize> = if params.axes.is_empty() {
        (0..rank).rev().collect()
    } else {
        params
            .axes
            .iter()
            .map(|&axis| normalize_axis(op, axis, rank, "transpose permutation"))
            .collect::<Fn3Result<Vec<_>>>()?
    };
    Shape::validate_permutation(&permutation, rank).map_err(|error| error.with_op(op))?;
    let expected = input.shape().permuted(&permutation)?;
    require_output_shape(op, out, &expected)?;
    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    launch_permute(backend, op, input, out, Some(permutation), elements)
}

/// `Reshape(data, shape)`: a row-major identity copy into the validated target.
pub fn launch_reshape(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Reshape;
    check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
    require_no_attributes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let input = &inputs[0];
    let out = &mut outputs[0];
    require_contiguous_view(op, input, "input")?;
    require_contiguous_output(op, out, "output")?;
    require_shader_rank(op, input.shape().rank())?;
    require_shader_rank(op, out.shape().rank())?;
    let target = read_i64_values(op, &inputs[1], "reshape shape input")?;
    let expected = resolve_reshape_shape(op, input.shape(), &target)?;
    require_output_shape(op, out, &expected)?;
    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    launch_permute(backend, op, input, out, None, elements)
}

/// `Flatten`: collapse to `[prod(dims[..axis]), prod(dims[axis..])]`.
pub fn launch_flatten(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Flatten;
    check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
    require_only_axes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let input = &inputs[0];
    let out = &mut outputs[0];
    require_contiguous_view(op, input, "input")?;
    require_contiguous_output(op, out, "output")?;
    let rank = input.shape().rank();
    let axis = if params.axes.is_empty() {
        1.min(rank)
    } else {
        match params.axes.as_slice() {
            [axis] => {
                let resolved = if *axis < 0 {
                    (*axis + rank as i64 + 1).max(0)
                } else {
                    *axis
                };
                usize::try_from(resolved)
                    .ok()
                    .filter(|&value| value <= rank)
                    .ok_or_else(|| Fn3Error::InvalidShape {
                        context: "vulkan flatten axis",
                        detail: format!("op {op}: axis {axis} is outside 0..={rank}"),
                    })?
            }
            axes => {
                return Err(Fn3Error::InvalidShape {
                    context: "vulkan flatten axis",
                    detail: format!("op {op} takes one axis, but the plan supplies {axes:?}"),
                })
            }
        }
    };
    let leading = checked_product(&input.shape().dims()[..axis])?;
    let trailing = checked_product(&input.shape().dims()[axis..])?;
    let expected = Shape::from_dims([leading, trailing]);
    require_output_shape(op, out, &expected)?;
    require_shader_rank(op, out.shape().rank())?;
    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    launch_permute(backend, op, input, out, None, elements)
}

/// `Squeeze`: remove extent-1 axes (all of them when no axes are given).
pub fn launch_squeeze(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Squeeze;
    if !(1..=2).contains(&inputs.len()) || outputs.len() != 1 {
        return Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs: 1,
            actual_inputs: inputs.len(),
            expected_outputs: 1,
            actual_outputs: outputs.len(),
        });
    }
    require_only_axes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let input = &inputs[0];
    let out = &mut outputs[0];
    require_contiguous_view(op, input, "input")?;
    require_contiguous_output(op, out, "output")?;
    let rank = input.shape().rank();
    require_shader_rank(op, rank)?;
    let axes = axes_from_params_or_input(op, params, inputs.get(1))?;
    let mut removed = vec![false; rank];
    if axes.is_empty() {
        for axis in 0..rank {
            removed[axis] = input.shape().dims()[axis] == 1;
        }
    } else {
        for &axis in &axes {
            let axis = normalize_axis(op, axis, rank, "squeeze axis")?;
            if removed[axis] {
                return Err(Fn3Error::InvalidShape {
                    context: "vulkan squeeze axis",
                    detail: format!("op {op}: axis {axis} is listed twice"),
                });
            }
            if input.shape().dims()[axis] != 1 {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("extent 1 on axis {axis} of {}", input.shape()),
                    actual: format!("{}", input.shape().dims()[axis]),
                });
            }
            removed[axis] = true;
        }
    }
    let dims: Vec<u64> = input
        .shape()
        .dims()
        .iter()
        .enumerate()
        .filter(|(axis, _)| !removed[*axis])
        .map(|(_, &extent)| extent)
        .collect();
    let expected = Shape::from_dims(dims);
    require_output_shape(op, out, &expected)?;
    require_shader_rank(op, out.shape().rank())?;
    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    launch_permute(backend, op, input, out, None, elements)
}

/// `Unsqueeze`: insert extent-1 axes at positions named in the *output* rank.
pub fn launch_unsqueeze(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Unsqueeze;
    if !(1..=2).contains(&inputs.len()) || outputs.len() != 1 {
        return Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs: 1,
            actual_inputs: inputs.len(),
            expected_outputs: 1,
            actual_outputs: outputs.len(),
        });
    }
    require_only_axes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let input = &inputs[0];
    let out = &mut outputs[0];
    require_contiguous_view(op, input, "input")?;
    require_contiguous_output(op, out, "output")?;
    let axes = axes_from_params_or_input(op, params, inputs.get(1))?;
    if axes.is_empty() {
        return Err(Fn3Error::InvalidShape {
            context: "vulkan unsqueeze axes",
            detail: format!("op {op} needs at least one axis to insert"),
        });
    }
    let out_rank = input.shape().rank() + axes.len();
    require_shader_rank(op, out_rank)?;
    let mut inserted = vec![false; out_rank];
    for &axis in &axes {
        let axis = normalize_axis(op, axis, out_rank, "unsqueeze axis")?;
        if inserted[axis] {
            return Err(Fn3Error::InvalidShape {
                context: "vulkan unsqueeze axis",
                detail: format!("op {op}: axis {axis} is listed twice"),
            });
        }
        inserted[axis] = true;
    }
    let mut dims = Vec::with_capacity(out_rank);
    let mut source = input.shape().dims().iter();
    for axis in 0..out_rank {
        if inserted[axis] {
            dims.push(1);
        } else {
            let extent = source
                .next()
                .copied()
                .ok_or_else(|| Fn3Error::InvalidShape {
                    context: "vulkan unsqueeze axes",
                    detail: format!(
                        "op {op}: more inserted axes than the output rank allows for {}",
                        input.shape()
                    ),
                })?;
            dims.push(extent);
        }
    }
    let expected = Shape::from_dims(dims);
    require_output_shape(op, out, &expected)?;
    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    launch_permute(backend, op, input, out, None, elements)
}

// ---------------------------------------------------------------------------
// Slice / Split / Concat
// ---------------------------------------------------------------------------

/// Push constants for one block move.
#[allow(clippy::too_many_arguments)]
fn block_push(
    op: OpId,
    elements: u32,
    outer: u32,
    src_axis: u32,
    inner: u32,
    src_start: i64,
    src_step: i64,
    dst_axis: u32,
    dst_axis_offset: u32,
    width: u32,
) -> Fn3Result<Vec<u8>> {
    let start = i32::try_from(src_start).map_err(|_| Fn3Error::UnsupportedOp {
        op,
        backend: BACKEND,
        reason: format!("resolved axis start {src_start} does not fit the 32-bit shader field"),
    })?;
    let step = i32::try_from(src_step).map_err(|_| Fn3Error::UnsupportedOp {
        op,
        backend: BACKEND,
        reason: format!("axis step {src_step} does not fit the 32-bit shader field"),
    })?;
    Ok(Push::new()
        .u32(elements)
        .u32(outer)
        .u32(src_axis)
        .u32(inner)
        .i32(start)
        .i32(step)
        .u32(dst_axis)
        .u32(dst_axis_offset)
        .u32(width)
        .bytes())
}

/// `Slice(data, starts, ends, axes?, steps?)`: one sliced axis per call.
pub fn launch_slice(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Slice;
    if !(2..=5).contains(&inputs.len()) || outputs.len() != 1 {
        return Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs: 5,
            actual_inputs: inputs.len(),
            expected_outputs: 1,
            actual_outputs: outputs.len(),
        });
    }
    require_no_attributes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let input = &inputs[0];
    let out = &mut outputs[0];
    let dtype = input.dtype();
    require_contiguous_view(op, input, "input")?;
    require_contiguous_output(op, out, "output")?;
    let rank = input.shape().rank();
    require_shader_rank(op, rank)?;
    let width = element_bytes(op, dtype)?;

    let starts = read_i64_values(op, &inputs[1], "slice starts")?;
    let ends = read_i64_values(op, &inputs[2], "slice ends")?;
    if starts.len() != ends.len() {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{} ends to match the starts", starts.len()),
            actual: format!("{} ends", ends.len()),
        });
    }
    let count = starts.len();
    let axes = match inputs.get(3) {
        Some(view) => read_i64_values(op, view, "slice axes")?,
        None => (0..count as i64).collect(),
    };
    let steps = match inputs.get(4) {
        Some(view) => read_i64_values(op, view, "slice steps")?,
        None => vec![1i64; count],
    };
    if axes.len() != count || steps.len() != count {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{count} axes and {count} steps to match the starts"),
            actual: format!("{} axes and {} steps", axes.len(), steps.len()),
        });
    }

    let mut plan: Vec<AxisSlice> = input
        .shape()
        .dims()
        .iter()
        .map(|&extent| AxisSlice {
            start: 0,
            step: 1,
            count: extent,
        })
        .collect();
    let mut seen = vec![false; rank];
    let mut changed: Vec<usize> = Vec::new();
    for position in 0..count {
        let axis = normalize_axis(op, axes[position], rank, "slice axis")?;
        if seen[axis] {
            return Err(Fn3Error::InvalidShape {
                context: "vulkan slice axis",
                detail: format!("op {op}: axis {axis} is sliced twice"),
            });
        }
        seen[axis] = true;
        let slice = resolve_axis_slice(
            op,
            input.shape().dims()[axis],
            starts[position],
            ends[position],
            steps[position],
        )?;
        if slice.start != 0 || slice.step != 1 || slice.count != input.shape().dims()[axis] {
            changed.push(axis);
        }
        plan[axis] = slice;
    }
    if changed.len() > 1 {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: BACKEND,
            reason: format!(
                "the FN3-13 (#59) Vulkan slice kernel moves one axis per dispatch, but this plan \
                 changes axes {changed:?}; the pinned graph slices one axis"
            ),
        });
    }
    let expected = Shape::from_dims(plan.iter().map(|axis| axis.count));
    require_output_shape(op, out, &expected)?;
    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    let axis = changed.first().copied().unwrap_or(0);
    let dims = input.shape().dims();
    let outer = checked_product(&dims[..axis])?;
    let inner = checked_product(&dims[axis + 1..])?;
    let slice = plan[axis];
    let push = block_push(
        op,
        element_count(op, elements)?,
        u32::try_from(outer).map_err(|_| Fn3Error::ShapeOverflow {
            context: "vulkan slice outer extent",
            axis: Some(axis),
            detail: "outer extent does not fit u32".to_string(),
        })?,
        u32::try_from(dims[axis]).map_err(|_| Fn3Error::ShapeOverflow {
            context: "vulkan slice axis extent",
            axis: Some(axis),
            detail: "axis extent does not fit u32".to_string(),
        })?,
        u32::try_from(inner).map_err(|_| Fn3Error::ShapeOverflow {
            context: "vulkan slice inner extent",
            axis: Some(axis),
            detail: "inner extent does not fit u32".to_string(),
        })?,
        slice.start,
        slice.step,
        element_count(op, slice.count)?,
        0,
        width,
    )?;
    let workgroups = workgroups(&BLOCK_COPY, elements, out.shape().dims())?;
    let mut bytes = vec![0u8; out.meta().byte_len()?];
    let mut job = Job::new(backend, op, &BLOCK_COPY)?;
    let source = job.input(view_window_bytes(input)?, "fn3.slice.in")?;
    let destination = job.output(out.meta().byte_len()? as u64, "fn3.slice.out")?;
    job.step(
        push,
        workgroups,
        vec![
            (0, source, BindingRole::Input),
            (1, destination, BindingRole::Output),
        ],
    );
    job.run(&mut [&mut bytes])?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}

/// The maximum number of inputs or outputs one `Split`/`Concat` recording
/// carries, bounded by the platform layer's transient descriptor pool.
const MAX_PIECES: usize = 16;

/// `Split`: slice along one axis into one output per piece.
pub fn launch_split(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Split;
    if !(1..=2).contains(&inputs.len()) || outputs.is_empty() {
        return Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs: 2,
            actual_inputs: inputs.len(),
            expected_outputs: 1,
            actual_outputs: outputs.len(),
        });
    }
    require_only_axes(op, params)?;
    let device = backend.device_id();
    require_device(op, device, inputs, &[])?;
    for out in outputs.iter() {
        out.meta().require_device(Some(op), device)?;
    }
    let input = &inputs[0];
    let dtype = input.dtype();
    require_contiguous_view(op, input, "input")?;
    let rank = input.shape().rank();
    require_shader_rank(op, rank)?;
    let width = element_bytes(op, dtype)?;
    let axis = single_axis(op, params, rank, 0, "split axis")?;
    let dimension = input.shape().dims()[axis];
    let count = outputs.len();
    if count > MAX_PIECES {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: BACKEND,
            reason: format!(
                "the FN3-13 (#59) Vulkan split kernel records at most {MAX_PIECES} pieces per \
                 submission, and this request has {count}"
            ),
        });
    }
    let sizes: Vec<u64> = match inputs.get(1) {
        Some(view) => {
            let values = read_i64_values(op, view, "split sizes input")?;
            if values.len() != count {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("{count} split sizes to match the outputs"),
                    actual: format!("{} sizes", values.len()),
                });
            }
            values
                .into_iter()
                .map(|value| {
                    u64::try_from(value).map_err(|_| Fn3Error::InvalidShape {
                        context: "vulkan split sizes",
                        detail: format!("op {op}: split size {value} is negative"),
                    })
                })
                .collect::<Fn3Result<Vec<_>>>()?
        }
        None => {
            if !dimension.is_multiple_of(count as u64) {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("axis {axis} of {} divisible by {count}", input.shape()),
                    actual: format!("extent {dimension}"),
                });
            }
            vec![dimension / count as u64; count]
        }
    };
    let total: u64 = sizes.iter().sum();
    if total != dimension {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("split sizes summing to {dimension} (axis {axis})"),
            actual: format!("{sizes:?} sum to {total}"),
        });
    }
    let dims = input.shape().dims();
    let outer = checked_product(&dims[..axis])?;
    let inner = checked_product(&dims[axis + 1..])?;
    let outer_u32 = u32::try_from(outer).map_err(|_| Fn3Error::ShapeOverflow {
        context: "vulkan split outer extent",
        axis: Some(axis),
        detail: "outer extent does not fit u32".to_string(),
    })?;
    let axis_u32 = u32::try_from(dims[axis]).map_err(|_| Fn3Error::ShapeOverflow {
        context: "vulkan split axis extent",
        axis: Some(axis),
        detail: "axis extent does not fit u32".to_string(),
    })?;
    let inner_u32 = u32::try_from(inner).map_err(|_| Fn3Error::ShapeOverflow {
        context: "vulkan split inner extent",
        axis: Some(axis),
        detail: "inner extent does not fit u32".to_string(),
    })?;

    let mut expected_dims = dims.to_vec();
    let mut offset = 0u64;
    type PlannedSplit = Option<(Vec<u8>, [u32; 3], usize)>;
    let mut plans: Vec<PlannedSplit> = Vec::with_capacity(count);
    for (position, out) in outputs.iter_mut().enumerate() {
        if out.dtype() != dtype {
            return Err(Fn3Error::DTypeMismatch {
                op: Some(op),
                expected: dtype,
                actual: out.dtype(),
            });
        }
        require_contiguous_output(op, out, "split output")?;
        expected_dims[axis] = sizes[position];
        let expected = Shape::from_dims(expected_dims.clone());
        require_output_shape(op, out, &expected)?;
        let byte_len = out.meta().byte_len()?;
        if byte_len == 0 {
            plans.push(None);
            offset += sizes[position];
            continue;
        }
        let push = block_push(
            op,
            element_count(op, expected.numel()?)?,
            outer_u32,
            axis_u32,
            inner_u32,
            i64::try_from(offset).map_err(|_| Fn3Error::ShapeOverflow {
                context: "vulkan split offset",
                axis: Some(axis),
                detail: "offset does not fit i64".to_string(),
            })?,
            1,
            element_count(op, sizes[position])?,
            0,
            width,
        )?;
        plans.push(Some((
            push,
            workgroups(&BLOCK_COPY, expected.numel()?, expected.dims())?,
            byte_len,
        )));
        offset += sizes[position];
    }
    if plans.iter().all(Option::is_none) {
        return Ok(());
    }
    let mut bytes: Vec<Vec<u8>> = plans
        .iter()
        .map(|plan| vec![0u8; plan.as_ref().map(|(_, _, bytes)| *bytes).unwrap_or(0)])
        .collect();
    let mut job = Job::new(backend, op, &BLOCK_COPY)?;
    let source = job.input(view_window_bytes(input)?, "fn3.split.in")?;
    let mut destinations = Vec::with_capacity(count);
    for (position, plan) in plans.iter().enumerate() {
        match plan {
            Some((_, _, byte_len)) => {
                let index = job.output(*byte_len as u64, "fn3.split.out")?;
                destinations.push(Some((position, index)));
            }
            None => destinations.push(None),
        }
    }
    for (position, slot) in destinations.iter().enumerate() {
        let Some((_, index)) = slot else { continue };
        let Some((push, workgroups, _)) = plans[position].as_ref() else {
            continue;
        };
        job.step(
            push.clone(),
            *workgroups,
            vec![
                (0, source, BindingRole::Input),
                (1, *index, BindingRole::Output),
            ],
        );
    }
    let mut active: Vec<&mut [u8]> = bytes
        .iter_mut()
        .enumerate()
        .filter(|(position, _)| plans[*position].is_some())
        .map(|(_, bytes)| bytes.as_mut_slice())
        .collect();
    job.run(&mut active)?;
    for (out, bytes) in outputs.iter_mut().zip(bytes.iter()) {
        if !bytes.is_empty() {
            output_window_bytes(out)?.copy_from_slice(bytes);
        }
    }
    Ok(())
}

/// `Concat`: join one axis of several inputs.
pub fn launch_concat(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Concat;
    if inputs.is_empty() || outputs.len() != 1 {
        return Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs: 1,
            actual_inputs: inputs.len(),
            expected_outputs: 1,
            actual_outputs: outputs.len(),
        });
    }
    require_only_axes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let out = &mut outputs[0];
    let first = &inputs[0];
    let dtype = first.dtype();
    let rank = first.shape().rank();
    require_shader_rank(op, rank)?;
    let width = element_bytes(op, dtype)?;
    let axis = single_axis(op, params, rank, 0, "concat axis")?;
    let count = inputs.len();
    if count > MAX_PIECES {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: BACKEND,
            reason: format!(
                "the FN3-13 (#59) Vulkan concat kernel records at most {MAX_PIECES} inputs per \
                 submission, and this request has {count}"
            ),
        });
    }
    let mut expected_dims = first.shape().dims().to_vec();
    expected_dims[axis] = 0;
    for view in inputs {
        require_contiguous_view(op, view, "concat input")?;
        if view.dtype() != dtype {
            return Err(Fn3Error::DTypeMismatch {
                op: Some(op),
                expected: dtype,
                actual: view.dtype(),
            });
        }
        if view.shape().rank() != rank {
            return Err(Fn3Error::RankMismatch {
                op: Some(op),
                expected: rank,
                actual: view.shape().rank(),
            });
        }
        for (position, (&extent, &reference)) in view
            .shape()
            .dims()
            .iter()
            .zip(first.shape().dims())
            .enumerate()
        {
            if position != axis && extent != reference {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!("every axis but {axis} to match {}", first.shape()),
                    actual: format!("{} differs on axis {position}", view.shape()),
                });
            }
        }
        expected_dims[axis] = expected_dims[axis]
            .checked_add(view.shape().dims()[axis])
            .ok_or(Fn3Error::ShapeOverflow {
                context: "vulkan concat extent",
                axis: Some(axis),
                detail: "concatenated extent exceeds u64::MAX".to_string(),
            })?;
    }
    let expected = Shape::from_dims(expected_dims);
    require_output_shape(op, out, &expected)?;
    require_contiguous_output(op, out, "output")?;
    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    let dims = first.shape().dims();
    let outer = checked_product(&dims[..axis])?;
    let inner = checked_product(&dims[axis + 1..])?;
    let outer_u32 = u32::try_from(outer).map_err(|_| Fn3Error::ShapeOverflow {
        context: "vulkan concat outer extent",
        axis: Some(axis),
        detail: "outer extent does not fit u32".to_string(),
    })?;
    let inner_u32 = u32::try_from(inner).map_err(|_| Fn3Error::ShapeOverflow {
        context: "vulkan concat inner extent",
        axis: Some(axis),
        detail: "inner extent does not fit u32".to_string(),
    })?;
    let dst_axis = u32::try_from(expected.dims()[axis]).map_err(|_| Fn3Error::ShapeOverflow {
        context: "vulkan concat destination extent",
        axis: Some(axis),
        detail: "concatenated extent does not fit u32".to_string(),
    })?;

    let mut bytes = vec![0u8; out.meta().byte_len()?];
    let mut job = Job::new(backend, op, &BLOCK_COPY)?;
    let destination = job.output(out.meta().byte_len()? as u64, "fn3.concat.out")?;
    let mut offset = 0u64;
    for view in inputs {
        let piece = view.shape().numel()?;
        if piece != 0 {
            // A zero-element input (the empty `past` of a prefill) contributes
            // no bytes and cannot be bound: skip it instead of allocating an
            // unaddressable buffer. The destination extent is unchanged, and a
            // concatenation of only empty inputs returned above.
            let source = job.input(view_window_bytes(view)?, "fn3.concat.in")?;
            let push = block_push(
                op,
                element_count(op, piece)?,
                outer_u32,
                u32::try_from(view.shape().dims()[axis]).map_err(|_| Fn3Error::ShapeOverflow {
                    context: "vulkan concat source extent",
                    axis: Some(axis),
                    detail: "source extent does not fit u32".to_string(),
                })?,
                inner_u32,
                0,
                1,
                dst_axis,
                element_count(op, offset)?,
                width,
            )?;
            job.step(
                push,
                workgroups(&BLOCK_COPY, piece, view.shape().dims())?,
                vec![
                    (0, source, BindingRole::Input),
                    (1, destination, BindingRole::Output),
                ],
            );
        }
        offset += view.shape().dims()[axis];
    }
    job.run(&mut [&mut bytes])?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}

// ---------------------------------------------------------------------------
// Expand
// ---------------------------------------------------------------------------

/// `Expand`: broadcast to the target shape.
pub fn launch_expand(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Expand;
    check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
    require_no_attributes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let input = &inputs[0];
    let out = &mut outputs[0];
    require_contiguous_view(op, input, "input")?;
    require_contiguous_output(op, out, "output")?;
    require_shader_rank(op, input.shape().rank())?;
    let target = shape_values(op, &inputs[1], "expand shape input")?;
    let expected = input.shape().broadcast_with(&target, op)?;
    require_output_shape(op, out, &expected)?;
    let rank = expected.rank();
    require_shader_rank(op, rank)?;
    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    let width = element_bytes(op, input.dtype())?;
    let push = Push::new()
        .u32(element_count(op, elements)?)
        .u32(width)
        .u32(rank as u32)
        .dims(output_dims(op, &expected)?)
        .dims(broadcast_dims(op, input.shape(), rank)?)
        .bytes();
    let workgroups = workgroups(&EXPAND, elements, expected.dims())?;
    let mut bytes = vec![0u8; out.meta().byte_len()?];
    let mut job = Job::new(backend, op, &EXPAND)?;
    let source = job.input(view_window_bytes(input)?, "fn3.expand.in")?;
    let destination = job.output(out.meta().byte_len()? as u64, "fn3.expand.out")?;
    job.step(
        push,
        workgroups,
        vec![
            (0, source, BindingRole::Input),
            (1, destination, BindingRole::Output),
        ],
    );
    job.run(&mut [&mut bytes])?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}

// ---------------------------------------------------------------------------
// Shape / ConstantOfShape / Range
// ---------------------------------------------------------------------------

/// `Shape`: write the input extents as an i64 tensor.
pub fn launch_shape(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Shape;
    check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
    require_no_attributes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let input = &inputs[0];
    let out = &mut outputs[0];
    require_output_dtype(op, out, DType::I64)?;
    require_contiguous_output(op, out, "output")?;
    let rank = input.shape().rank();
    require_shader_rank(op, rank)?;
    let expected = Shape::from_dims([rank as u64]);
    require_output_shape(op, out, &expected)?;
    if rank == 0 {
        return Ok(());
    }
    let mut low = [0u32; 8];
    let mut high = [0u32; 8];
    for (axis, &extent) in input.shape().dims().iter().enumerate() {
        low[axis] = (extent & 0xffff_ffff) as u32;
        high[axis] = (extent >> 32) as u32;
    }
    let push = Push::new().u32(rank as u32).dims(low).dims(high).bytes();
    let workgroups = workgroups(&SHAPE, rank as u64, expected.dims())?;
    let mut bytes = vec![0u8; out.meta().byte_len()?];
    let mut job = Job::new(backend, op, &SHAPE)?;
    let destination = job.output(out.meta().byte_len()? as u64, "fn3.shape.out")?;
    job.step(
        push,
        workgroups,
        vec![(0, destination, BindingRole::Output)],
    );
    job.run(&mut [&mut bytes])?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}

/// `ConstantOfShape(shape, value)`.
pub fn launch_constant_of_shape(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::ConstantOfShape;
    check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
    require_no_attributes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let value = &inputs[1];
    let out = &mut outputs[0];
    if value.dtype() != out.dtype() {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: out.dtype(),
            actual: value.dtype(),
        });
    }
    if value.shape().numel()? != 1 {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: "a single-element fill value".to_string(),
            actual: value.shape().to_string(),
        });
    }
    require_contiguous_view(op, value, "fill value")?;
    require_contiguous_output(op, out, "output")?;
    let expected = shape_values(op, &inputs[0], "constant_of_shape shape input")?;
    require_output_shape(op, out, &expected)?;
    require_shader_rank(op, expected.rank())?;
    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    let dtype = out.dtype();
    let width = dtype.size_bytes() as u32;
    let raw = view_window_bytes(value)?;
    let (low, high) = match raw {
        [byte] => (u32::from(*byte), 0u32),
        [a, b, c, d] => (u32::from_le_bytes([*a, *b, *c, *d]), 0u32),
        [a, b, c, d, e, f, g, h] => (
            u32::from_le_bytes([*a, *b, *c, *d]),
            u32::from_le_bytes([*e, *f, *g, *h]),
        ),
        other => {
            return Err(Fn3Error::StorageCast {
                element: "fill value",
                byte_len: other.len(),
                detail: format!(
                    "op {op}: expected 1, 4 or 8 bytes for a {dtype} fill value, found {}",
                    other.len()
                ),
            })
        }
    };
    let push = Push::new()
        .u32(element_count(op, elements)?)
        .u32(width)
        .u32(low)
        .u32(high)
        .bytes();
    let threads = if width == 1 {
        elements.div_ceil(4)
    } else {
        elements
    };
    let workgroups = workgroups(&FILL, threads, expected.dims())?;
    let mut bytes = vec![0u8; out.meta().byte_len()?];
    let mut job = Job::new(backend, op, &FILL)?;
    let destination = job.output(out.meta().byte_len()? as u64, "fn3.fill.out")?;
    job.step(
        push,
        workgroups,
        vec![(0, destination, BindingRole::Output)],
    );
    job.run(&mut [&mut bytes])?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}

/// `Range(start, limit, delta)`.
pub fn launch_range(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Range;
    check_arity(op, inputs.len(), outputs.len(), 3, 1)?;
    require_no_attributes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let out = &mut outputs[0];
    for input in inputs {
        if input.dtype() != out.dtype() {
            return Err(Fn3Error::DTypeMismatch {
                op: Some(op),
                expected: out.dtype(),
                actual: input.dtype(),
            });
        }
        require_contiguous_view(op, input, "range scalar")?;
        if input.shape().numel()? != 1 {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(op),
                expected: "scalar start/limit/delta".to_string(),
                actual: input.shape().to_string(),
            });
        }
    }
    require_contiguous_output(op, out, "output")?;
    let dtype = out.dtype();
    let (count, push, spec) = match dtype {
        DType::I64 => {
            let start = read_i64_scalar(op, &inputs[0], "range start")?;
            let limit = read_i64_scalar(op, &inputs[1], "range limit")?;
            let delta = read_i64_scalar(op, &inputs[2], "range delta")?;
            if delta == 0 {
                return Err(Fn3Error::InvalidShape {
                    context: "vulkan range delta",
                    detail: format!("op {op}: a range delta of 0 is undefined"),
                });
            }
            let span = i128::from(limit) - i128::from(start);
            let step = i128::from(delta);
            let magnitude = if (step > 0 && span > 0) || (step < 0 && span < 0) {
                let quotient = (span / step).abs();
                if span % step != 0 {
                    quotient + 1
                } else {
                    quotient
                }
            } else {
                0
            };
            let count = u64::try_from(magnitude).map_err(|_| Fn3Error::ShapeOverflow {
                context: "vulkan range extent",
                axis: None,
                detail: "range element count does not fit u64".to_string(),
            })?;
            if count > 0 {
                let last = i128::from(start) + i128::from(count - 1) * step;
                i64::try_from(last).map_err(|_| Fn3Error::UnsupportedOp {
                    op,
                    backend: BACKEND,
                    reason: format!(
                        "the FN3-13 (#59) Vulkan i64 range kernel cannot represent the last \
                         value {last} of this sequence in i64"
                    ),
                })?;
            }
            let (start_low, start_high) = split_i64(start);
            let (delta_low, delta_high) = split_i64(delta);
            (
                count,
                Push::new()
                    .u32(element_count(op, count)?)
                    .u32(start_low)
                    .u32(start_high)
                    .u32(delta_low)
                    .u32(delta_high)
                    .bytes(),
                &RANGE_I64,
            )
        }
        DType::F32 => {
            let start = f32_at(op, &inputs[0], "range start")?;
            let limit = f32_at(op, &inputs[1], "range limit")?;
            let delta = f32_at(op, &inputs[2], "range delta")?;
            if delta == 0.0 || !delta.is_finite() {
                return Err(Fn3Error::InvalidShape {
                    context: "vulkan range delta",
                    detail: format!("op {op}: a range delta of {delta} is not usable"),
                });
            }
            let span = f64::from(limit) - f64::from(start);
            let step = f64::from(delta);
            let ratio = span / step;
            let count = if ratio > 0.0 {
                let ceil = ratio.ceil();
                if ceil.is_finite() && ceil >= 0.0 {
                    ceil as u64
                } else {
                    0
                }
            } else {
                0
            };
            (
                count,
                Push::new()
                    .u32(element_count(op, count)?)
                    .f32(start)
                    .f32(delta)
                    .bytes(),
                &RANGE_F32,
            )
        }
        other => {
            return Err(Fn3Error::UnsupportedDType {
                op: Some(op),
                dtype: other,
                context: "vulkan range kernel (i64 and f32 only)",
            })
        }
    };
    let expected = Shape::from_dims([count]);
    require_output_shape(op, out, &expected)?;
    if count == 0 {
        return Ok(());
    }
    let workgroups = workgroups(spec, count, expected.dims())?;
    let mut bytes = vec![0u8; out.meta().byte_len()?];
    let mut job = Job::new(backend, op, spec)?;
    let destination = job.output(out.meta().byte_len()? as u64, "fn3.range.out")?;
    job.step(
        push,
        workgroups,
        vec![(0, destination, BindingRole::Output)],
    );
    job.run(&mut [&mut bytes])?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}

/// Split an `i64` into `(low, high)` words.
fn split_i64(value: i64) -> (u32, u32) {
    let bits = value as u64;
    ((bits & 0xffff_ffff) as u32, (bits >> 32) as u32)
}

/// Read the single element of an f32 tensor.
fn f32_at(op: OpId, view: &TensorView<'_>, what: &'static str) -> Fn3Result<f32> {
    let values = view.as_f32_slice()?;
    match values {
        [value] => Ok(*value),
        _ => Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!("{what} to have exactly one element"),
            actual: format!("{} has {} elements", view.shape(), values.len()),
        }),
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register this module's kernels.
pub fn register(set: &mut KernelSet) {
    // Shape-only ops copy row-major; they share the permute shader's identity
    // mode because the extents differ but the element order does not.
    //
    // The verdict is `Partial` where the kernel's subset is narrower than the
    // op's documented contract, and the reason names the missing case and its
    // owner: a `Bool` (1-byte) element cannot be written by a workgroup-wide
    // dispatch without `storageBuffer8BitAccess`, which the platform layer does
    // not enable. `admits_movable` answers the same question for the plan's
    // concrete element types; it never widens the registration.
    macro_rules! layout_op {
        ($op:expr, $launch:expr, $kernel:expr, $support:expr, $admits:expr) => {
            set.register(OpKernel {
                op: $op,
                support: $support,
                features: CapabilityFeatures::default(),
                dtypes: MOVE_DTYPES,
                admits: Some($admits),
                kernel: $kernel,
                launch: $launch,
            });
        };
    }
    for (op, launch) in [
        (OpId::Transpose, launch_transpose as super::LaunchFn),
        (OpId::Reshape, launch_reshape),
        (OpId::Flatten, launch_flatten),
        (OpId::Squeeze, launch_squeeze),
        (OpId::Unsqueeze, launch_unsqueeze),
        (OpId::Expand, launch_expand),
    ] {
        layout_op!(
            op,
            launch,
            &PERMUTE,
            Support::partial(WIDTH_GAP),
            admits_movable
        );
    }
    layout_op!(
        OpId::Slice,
        launch_slice,
        &BLOCK_COPY,
        Support::partial(
            "one sliced axis per call (ONNX starts/ends/axes/steps, negative steps              included); 4- and 8-byte elements. A multi-axis slice and a Bool tensor are              rejected with UnsupportedOp naming #59"
        ),
        admits_movable
    );
    layout_op!(
        OpId::Split,
        launch_split,
        &BLOCK_COPY,
        Support::partial(
            "one axis, up to 16 outputs per call, 4- and 8-byte elements. More than 16 outputs              and a Bool tensor are rejected with UnsupportedOp naming #59"
        ),
        admits_movable
    );
    layout_op!(
        OpId::Concat,
        launch_concat,
        &BLOCK_COPY,
        Support::partial(
            "one axis, up to 16 inputs per call, 4- and 8-byte elements. More than 16 inputs and              a Bool tensor are rejected with UnsupportedOp naming #59"
        ),
        admits_concat
    );
    set.register(OpKernel {
        op: OpId::Shape,
        support: Support::Full,
        features: CapabilityFeatures::default(),
        dtypes: &[DType::I64],
        admits: None,
        kernel: &SHAPE,
        launch: launch_shape,
    });
    set.register(OpKernel {
        op: OpId::ConstantOfShape,
        support: Support::Full,
        features: CapabilityFeatures::default(),
        dtypes: &[DType::F32, DType::I32, DType::I64, DType::Bool],
        admits: None,
        kernel: &FILL,
        launch: launch_constant_of_shape,
    });
    // `Range` selects its pipeline from the output dtype, so the registration
    // names the f32 spec; `launch_range` resolves the i64 module itself.
    set.register(OpKernel {
        op: OpId::Range,
        support: Support::Full,
        features: CapabilityFeatures::default(),
        dtypes: &[DType::F32, DType::I64],
        admits: None,
        kernel: &RANGE_F32,
        launch: launch_range,
    });
}
