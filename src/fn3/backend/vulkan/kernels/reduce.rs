//! GPU kernels owned by FN3-15 (#61): `ReduceMean`, `ReduceSum` and `ArgMax`.
//!
//! # Contract
//!
//! The reduction rules are the CPU reference's (`backend/cpu/kernels/reduce.rs`):
//!
//! * Empty `OpParams::axes` means "every axis"; axes are resolved against the
//!   rank, sorted and de-duplicated, and `keep_dims` decides whether a reduced
//!   axis survives as extent 1.
//! * Each output element is accumulated by scanning the reduced coordinates in
//!   ascending row-major order — the order the CPU reference visits them, since
//!   it scans the input linearly — into one `f32` accumulator.
//! * `ReduceMean` divides the completed sum by the reduced element count; it
//!   never scales each element first. An empty reduction writes `+0.0`.
//! * `ArgMax` keeps the first occurrence of the maximum: a candidate replaces
//!   the running best only when it compares strictly greater, so a tie keeps
//!   the lowest index and a NaN never wins.
//!
//! One invocation owns one output element. Shapes ride in a storage-buffer
//! parameter block (plain scalar std430, the layout `ParamWriter` writes), so
//! no push-constant limit constrains the rank; the rank limit is the fn3
//! contract's 8.
//!
//! # Scope
//!
//! `ReduceMean`/`ReduceSum` are `Full`: any axis set, either `keep_dims`,
//! contiguous `f32`, which is exactly the fn3 reduction contract.
//!
//! `ArgMax` is `Partial`: `f32` and `i32` payloads are compared natively and the
//! `I64` index is written as two little-endian words. `i64` *inputs* are
//! rejected by name — comparing them needs the `shaderInt64` device feature,
//! which the FN3-11 (#57) platform layer does not enable.

use super::super::VulkanBackend;
use super::io;
use super::norm::{
    execute_staged, normalize_axis, require_shape, unsupported, BufferSource, ParamWriter,
};
use super::{grid, BindingSpec, KernelSet, KernelSpec, OpKernel};
use crate::inference::backend::{check_arity, CapabilityFeatures, OpParams, Support};
use crate::inference::dtype::DType;
use crate::inference::error::{Fn3Error, Fn3Result};
use crate::inference::ops::OpId;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

/// Owning issue named in this family's rejection messages.
const KERNEL_OWNER: &str = "FN3-15 (#61)";

/// The largest tensor rank the alpha contract accepts (matches the core).
const MAX_RANK: usize = 8;

/// Byte size of the `reduce.sum_mean.f32` parameter block: six scalars
/// (`rank`, `out_numel`, `reduced_count`, `mean`, `divisor`, `trailing`) followed
/// by three `uint[8]` arrays. The arrays are **fixed at 8 entries**, so a rank-3
/// request still uploads all 8 slots; the shader indexes only `0..rank`.
/// Shortening the block would make every array member after it read the wrong
/// bytes.
pub const REDUCE_PARAMS_BYTES: usize = 6 * 4 + 3 * MAX_RANK * 4;

/// Byte size of the `reduce.argmax.f32_i32` parameter block: four scalars
/// followed by two `uint[8]` arrays.
pub const ARGMAX_PARAMS_BYTES: usize = 4 * 4 + 2 * MAX_RANK * 4;

/// Pad a rank-sized value list to the fixed 8-entry parameter slot.
fn padded(values: &[u32]) -> [u32; MAX_RANK] {
    let mut slot = [0u32; MAX_RANK];
    slot[..values.len()].copy_from_slice(values);
    slot
}

/// `ReduceSum` / `ReduceMean` (`shaders/fn3_reduce_sum_mean.comp`).
pub const REDUCE_SUM_MEAN: KernelSpec = KernelSpec {
    name: "reduce.sum_mean.f32",
    spirv: include_bytes!("../shaders/fn3_reduce_sum_mean.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "input"),
        BindingSpec::storage(1, "out"),
        BindingSpec::storage(2, "params"),
    ],
    push_constant_bytes: 0,
    workgroup_size: [64, 1, 1],
    dtypes: &[DType::F32],
    shape: "contiguous f32 reduce over an arbitrary axis set, either keep_dims, rank <= 8",
    grid: grid::elementwise,
};

/// `ArgMax` along one axis (`shaders/fn3_argmax.comp`).
pub const ARGMAX: KernelSpec = KernelSpec {
    name: "reduce.argmax.f32_i32",
    spirv: include_bytes!("../shaders/fn3_argmax.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "input"),
        BindingSpec::storage(1, "out"),
        BindingSpec::storage(2, "params"),
    ],
    push_constant_bytes: 0,
    workgroup_size: [64, 1, 1],
    dtypes: &[DType::F32, DType::I32],
    shape: "contiguous f32 or i32 input, one axis, either keep_dims, I64 output, rank <= 8",
    grid: grid::elementwise,
};

/// Register the FN3-15 reduction kernels.
pub fn register(set: &mut KernelSet) {
    set.register(OpKernel {
        op: OpId::ReduceSum,
        support: Support::Full,
        features: CapabilityFeatures::default(),
        dtypes: &[DType::F32],
        admits: None,
        kernel: &REDUCE_SUM_MEAN,
        launch: launch_reduce,
    });
    set.register(OpKernel {
        op: OpId::ReduceMean,
        support: Support::Full,
        features: CapabilityFeatures::default(),
        dtypes: &[DType::F32],
        admits: None,
        kernel: &REDUCE_SUM_MEAN,
        launch: launch_reduce_mean,
    });
    set.register(OpKernel {
        op: OpId::ArgMax,
        support: Support::partial(
            "f32 and i32 inputs along one axis with either keep_dims, I64 output. i64 inputs \
             need the shaderInt64 device feature, which the FN3-11 (#57) platform layer does \
             not enable, so they are rejected by name (owner: FN3-15 / #61)",
        ),
        features: CapabilityFeatures::default(),
        dtypes: &[DType::F32, DType::I32],
        admits: Some(admits_argmax),
        kernel: &ARGMAX,
        launch: launch_argmax,
    });
}

/// Concrete admission for `ArgMax`: a 4-byte f32/i32 input, one axis, i64 out.
fn admits_argmax(
    _shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
) -> std::result::Result<(), String> {
    let mut operands = dtypes.iter();
    let Some(input) = operands.next() else {
        return Err("no operands were supplied".to_string());
    };
    let Some(output) = operands.next() else {
        return Err("argmax writes an i64 index but no output was supplied".to_string());
    };
    if !matches!(input, DType::F32 | DType::I32) {
        return Err(format!(
            "{input} input; this kernel compares f32 and i32 (i64 needs shaderInt64)"
        ));
    }
    if *output != DType::I64 {
        return Err(format!("{output} output; argmax writes an i64 index"));
    }
    if params.axes.len() > 1 {
        return Err(format!(
            "{} axes were supplied; this kernel reduces one axis",
            params.axes.len()
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shape helpers (mirrors of the CPU reference rules)
// ---------------------------------------------------------------------------

/// Resolve, sort and de-duplicate the reduced axes.
///
/// Empty `axes` means "every axis".
fn resolve_axes(op: OpId, params: &OpParams, rank: usize) -> Fn3Result<Vec<usize>> {
    if params.axes.is_empty() {
        return Ok((0..rank).collect());
    }
    let mut axes = Vec::with_capacity(params.axes.len());
    for &axis in &params.axes {
        let axis = normalize_axis(op, axis, rank, "vulkan reduce axes")?;
        if axes.contains(&axis) {
            return Err(Fn3Error::InvalidShape {
                context: "vulkan reduce axes",
                detail: format!("op {op}: axis {axis} is listed twice"),
            });
        }
        axes.push(axis);
    }
    axes.sort_unstable();
    Ok(axes)
}

/// Shape of a reduction over `axes`.
fn reduced_shape(input: &Shape, axes: &[usize], keep_dims: bool) -> Shape {
    let mut dims = Vec::with_capacity(input.rank());
    for (axis, &extent) in input.dims().iter().enumerate() {
        if axes.contains(&axis) {
            if keep_dims {
                dims.push(1);
            }
        } else {
            dims.push(extent);
        }
    }
    Shape::from_dims(dims)
}

/// `true` when `axes` is exactly the trailing run `rank - axes.len() .. rank`.
///
/// `axes` is sorted and de-duplicated by [`resolve_axes`]. A contiguous input
/// then stores each output's reduced elements in one ascending block, which is
/// the condition the shader's trailing-loop fast path relies on.
fn trailing_axes(axes: &[usize], rank: usize) -> bool {
    let Some(first) = rank.checked_sub(axes.len()) else {
        return false;
    };
    axes.iter()
        .enumerate()
        .all(|(offset, &axis)| axis == first + offset)
}

/// Contiguous row-major strides, as `u32`, rejecting element counts above 2^32.
fn strides_u32(op: OpId, dims: &[u64], context: &'static str) -> Fn3Result<Vec<u32>> {
    let mut strides = vec![0u32; dims.len()];
    let mut running: u64 = 1;
    for axis in (0..dims.len()).rev() {
        strides[axis] = u32::try_from(running).map_err(|_| {
            unsupported(
                op,
                format!(
                    "{context}: the stride at axis {axis} is {running}, above the 32-bit index \
                     range of the parameter block; split the tensor"
                ),
            )
        })?;
        running = running
            .checked_mul(dims[axis])
            .ok_or(Fn3Error::ShapeOverflow {
                context,
                axis: Some(axis),
                detail: "extent product exceeds u64".to_string(),
            })?;
    }
    Ok(strides)
}

fn u32_extent(op: OpId, extent: u64, axis: usize, context: &'static str) -> Fn3Result<u32> {
    u32::try_from(extent).map_err(|_| Fn3Error::ShapeOverflow {
        context,
        axis: Some(axis),
        detail: format!("op {op}: extent {extent} does not fit in u32"),
    })
}

fn check_rank(op: OpId, rank: usize) -> Fn3Result<()> {
    if rank > MAX_RANK {
        return Err(unsupported(
            op,
            format!("rank {rank} exceeds the fn3 contract limit of {MAX_RANK}"),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ReduceSum / ReduceMean
// ---------------------------------------------------------------------------

/// Execute `ReduceSum` through the platform layer.
pub fn launch_reduce(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_reduce_impl(OpId::ReduceSum, backend, inputs, outputs, params, false)
}

/// Execute `ReduceMean` through the same kernel with the mean divisor.
pub fn launch_reduce_mean(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_reduce_impl(OpId::ReduceMean, backend, inputs, outputs, params, true)
}

fn launch_reduce_impl(
    op: OpId,
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
    mean: bool,
) -> Fn3Result<()> {
    check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
    let input = &inputs[0];
    let out = &mut outputs[0];
    let rank = input.shape().rank();
    check_rank(op, rank)?;
    let axes = resolve_axes(op, params, rank)?;
    let expected = reduced_shape(input.shape(), &axes, params.keep_dims);
    require_shape(op, &expected, out.shape())?;

    let out_numel = expected.numel()?;
    if out_numel == 0 {
        return Ok(());
    }

    let input_operand = io::input(KERNEL_OWNER, backend, op, "reduce.input", input, DType::F32)?;

    let dims: Vec<u64> = input.shape().dims().to_vec();
    let strides = strides_u32(op, &dims, "vulkan reduce strides")?;
    let in_dims: Vec<u32> = dims
        .iter()
        .enumerate()
        .map(|(axis, &extent)| u32_extent(op, extent, axis, "vulkan reduce extent"))
        .collect::<Fn3Result<_>>()?;
    let mut reduced = vec![0u32; rank];
    for &axis in &axes {
        reduced[axis] = 1;
    }

    let mut reduced_count: u64 = 1;
    for &axis in &axes {
        reduced_count = reduced_count
            .checked_mul(dims[axis])
            .ok_or(Fn3Error::ShapeOverflow {
                context: "vulkan reduce count",
                axis: Some(axis),
                detail: "reduced element count exceeds u64".to_string(),
            })?;
    }
    let reduced_count_u32 = if reduced_count == 0 {
        0
    } else {
        u32::try_from(reduced_count).map_err(|_| {
            unsupported(
                op,
                format!(
                    "the reduction covers {reduced_count} elements, above the 32-bit loop \
                     counter of the parameter block; split the reduction"
                ),
            )
        })?
    };
    let out_numel_u32 = u32::try_from(out_numel).map_err(|_| {
        unsupported(
            op,
            format!("{out_numel} output elements exceed the 32-bit dispatch coordinate"),
        )
    })?;
    let divisor = if mean && reduced_count > 0 {
        reduced_count as f32
    } else {
        1.0
    };
    let trailing = trailing_axes(&axes, rank) && input.is_contiguous();

    let mut block = ParamWriter::new();
    block
        .u32(u32::try_from(rank).expect("rank <= 8"))
        .u32(out_numel_u32)
        .u32(reduced_count_u32)
        .u32(u32::from(mean))
        .f32(divisor)
        .u32(u32::from(trailing))
        .u32_slice(&padded(&in_dims))
        .u32_slice(&padded(&reduced))
        .u32_slice(&padded(&strides));
    let params_bytes = block.finish();
    debug_assert_eq!(params_bytes.len(), REDUCE_PARAMS_BYTES);

    let workgroups = REDUCE_SUM_MEAN
        .workgroups(out_numel, expected.dims())
        .map_err(|error| error.to_fn3())?;
    let out_operand = io::output(KERNEL_OWNER, backend, op, "reduce.out", out, DType::F32)?;
    let mut sources = [
        BufferSource::input("reduce.input", input_operand),
        BufferSource::output("reduce.out", out_operand),
        BufferSource::Upload {
            name: "reduce.params",
            bytes: &params_bytes,
        },
    ];
    execute_staged(op, backend, &REDUCE_SUM_MEAN, &mut sources, &[], workgroups)
}

// ---------------------------------------------------------------------------
// ArgMax
// ---------------------------------------------------------------------------

/// Execute `ArgMax` through the platform layer.
pub fn launch_argmax(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::ArgMax;
    check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
    let input = &inputs[0];
    let out = &mut outputs[0];
    let dtype = input.dtype();
    let mode = match dtype {
        DType::F32 => 0u32,
        DType::I32 => 1u32,
        other => {
            return Err(unsupported(
                op,
                format!(
                    "this kernel compares f32 and i32 payloads; a {other} input would need \
                     shaderInt64 (or a wide comparison), which FN3-11 (#57) does not enable"
                ),
            ))
        }
    };
    out.meta().require_dtype(Some(op), DType::I64)?;
    let rank = input.shape().rank();
    if rank == 0 {
        return Err(Fn3Error::RankMismatch {
            op: Some(op),
            expected: 1,
            actual: 0,
        });
    }
    check_rank(op, rank)?;
    let axis = match params.axes.as_slice() {
        [] => 0usize,
        [axis] => normalize_axis(op, *axis, rank, "vulkan argmax axis")?,
        axes => {
            return Err(Fn3Error::InvalidShape {
                context: "vulkan argmax axis",
                detail: format!("op {op} takes one axis, but the plan supplies {axes:?}"),
            })
        }
    };
    let expected = reduced_shape(input.shape(), &[axis], params.keep_dims);
    require_shape(op, &expected, out.shape())?;

    let out_numel = expected.numel()?;
    if out_numel == 0 {
        return Ok(());
    }
    if input.numel()? == 0 {
        return Err(unsupported(
            op,
            format!(
                "an empty reduction (input {}, axis {axis}) has no maximum to report",
                input.shape()
            ),
        ));
    }

    if !input.is_contiguous() {
        return Err(unsupported(
            op,
            format!(
                "argmax input is a strided view ({}); the FN3-15 kernels require contiguous \
                 operands",
                input.strides()
            ),
        ));
    }
    let dims: Vec<u64> = input.shape().dims().to_vec();
    let strides = strides_u32(op, &dims, "vulkan argmax strides")?;
    let in_dims: Vec<u32> = dims
        .iter()
        .enumerate()
        .map(|(position, &extent)| u32_extent(op, extent, position, "vulkan argmax extent"))
        .collect::<Fn3Result<_>>()?;
    let out_numel_u32 = u32::try_from(out_numel).map_err(|_| {
        unsupported(
            op,
            format!("{out_numel} output elements exceed the 32-bit dispatch coordinate"),
        )
    })?;

    let mut block = ParamWriter::new();
    block
        .u32(u32::try_from(rank).expect("rank <= 8"))
        .u32(out_numel_u32)
        .u32(u32::try_from(axis).expect("axis < rank <= 8"))
        .u32(mode)
        .u32_slice(&padded(&in_dims))
        .u32_slice(&padded(&strides));
    let params_bytes = block.finish();
    debug_assert_eq!(params_bytes.len(), ARGMAX_PARAMS_BYTES);

    // The input travels as raw 32-bit words: `f32` bits or native `i32`, which
    // the shader reinterprets according to `mode`.
    let input_operand = match dtype {
        DType::F32 => io::input(KERNEL_OWNER, backend, op, "argmax.input", input, DType::F32)?,
        DType::I32 => io::input(KERNEL_OWNER, backend, op, "argmax.input", input, DType::I32)?,
        other => {
            return Err(unsupported(
                op,
                format!("argmax does not accept a {other} input on Vulkan"),
            ))
        }
    };
    let out_operand = io::output(KERNEL_OWNER, backend, op, "argmax.out", out, DType::I64)?;

    let workgroups = ARGMAX
        .workgroups(out_numel, expected.dims())
        .map_err(|error| error.to_fn3())?;
    let mut sources = [
        BufferSource::input("argmax.input", input_operand),
        BufferSource::output("argmax.out", out_operand),
        BufferSource::Upload {
            name: "argmax.params",
            bytes: &params_bytes,
        },
    ];
    execute_staged(op, backend, &ARGMAX, &mut sources, &[], workgroups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_match_the_committed_shader_metadata() {
        for spec in [&REDUCE_SUM_MEAN, &ARGMAX] {
            assert_eq!(spec.workgroup_size, [64, 1, 1]);
            assert_eq!(spec.push_constant_bytes, 0);
            assert_eq!(spec.bindings.len(), 3);
            assert!(spec.validate().is_ok());
        }
        assert_eq!(REDUCE_SUM_MEAN.bindings[2].name, "params");
        assert!(ARGMAX.dtypes.contains(&DType::I32));
        assert!(REDUCE_SUM_MEAN.spirv.len().is_multiple_of(4));
        assert!(!REDUCE_SUM_MEAN.spirv.is_empty());
        assert!(!ARGMAX.spirv.is_empty());
    }

    #[test]
    fn reduced_shape_matches_the_cpu_rule() {
        let input = Shape::from_dims([2, 3, 4]);
        assert_eq!(reduced_shape(&input, &[2], false), Shape::from_dims([2, 3]));
        assert_eq!(
            reduced_shape(&input, &[1, 2], true),
            Shape::from_dims([2, 1, 1])
        );
        assert_eq!(
            reduced_shape(&input, &[0, 1, 2], false),
            Shape::from_dims([])
        );
    }

    #[test]
    fn empty_axes_means_every_axis() {
        let params = OpParams::default();
        assert_eq!(
            resolve_axes(OpId::ReduceSum, &params, 3).unwrap(),
            vec![0, 1, 2]
        );
        let params = OpParams::default().with_axes([-1, 0]);
        assert_eq!(
            resolve_axes(OpId::ReduceSum, &params, 3).unwrap(),
            vec![0, 2]
        );
        assert!(resolve_axes(OpId::ReduceSum, &OpParams::default().with_axes([1, 1]), 3).is_err());
    }

    #[test]
    fn strides_are_row_major_and_32_bit_checked() {
        assert_eq!(
            strides_u32(OpId::ReduceSum, &[2, 3, 4], "test").unwrap(),
            vec![12, 4, 1]
        );
    }

    #[test]
    fn parameter_blocks_are_the_declared_shader_size() {
        // These sizes are the std430 member offsets of the committed SPIR-V:
        // 0/4/8/12/16/20 for six scalars, arrays at 24/56/88 (reduce) and
        // 0/4/8/12 plus arrays at 16/48 (argmax).
        assert_eq!(REDUCE_PARAMS_BYTES, 120);
        assert_eq!(ARGMAX_PARAMS_BYTES, 80);

        let mut block = ParamWriter::new();
        block
            .u32(3)
            .u32(6)
            .u32(5)
            .u32(1)
            .f32(5.0)
            .u32(1)
            .u32_slice(&padded(&[2, 3, 5]))
            .u32_slice(&padded(&[0, 0, 1]))
            .u32_slice(&padded(&[15, 5, 1]));
        let bytes = block.finish();
        assert_eq!(bytes.len(), REDUCE_PARAMS_BYTES);
        // The first strides element must land at byte 88, not at 48: the
        // arrays are fixed-width, so a short rank cannot shift them.
        assert_eq!(&bytes[88..92], &15u32.to_le_bytes());
        // Padding entries are zero.
        assert_eq!(&bytes[36..40], &0u32.to_le_bytes());
        assert_eq!(&bytes[100..104], &0u32.to_le_bytes());
    }

    #[test]
    fn trailing_axes_recognises_only_the_final_run() {
        assert!(trailing_axes(&[], 0));
        assert!(trailing_axes(&[0, 1, 2], 3));
        assert!(trailing_axes(&[2], 3));
        assert!(trailing_axes(&[1, 2], 3));
        assert!(!trailing_axes(&[0], 3));
        assert!(!trailing_axes(&[0, 2], 3));
        assert!(!trailing_axes(&[1], 3));
        assert!(!trailing_axes(&[0, 1, 2], 2));
    }
}
