//! GPU kernel owned by FN3-13 (#59): `Gather`, which is the embedding lookup.
//!
//! # Why `Gather` lives here
//!
//! There is deliberately no `Embedding` op id
//! (`src/fn3/ops/mod.rs`): a GPT-2 token embedding is `Gather` with
//! `axis = 0` over an `F32` weight table and `I64` indices. The same op also
//! indexes the `I64` shape tensors the graph builds, so the kernel is written
//! for any axis while its primary use is the `[vocab, hidden]` table lookup.
//!
//! # Contract
//!
//! `out = data[..., index[..], ...]` with the indexed slices along `axis`
//! (default 0), exactly ONNX `Gather` and exactly the CPU reference:
//!
//! - `data` may be `F32`, `I32` or `I64` (4- or 8-byte elements);
//! - `indices` may be `I64` or `I32`, of any rank;
//! - a negative index counts from the end (`-1` is the last slice), and an
//!   index outside `[-dim, dim)` is [`Fn3Error::OutOfBounds`];
//! - the output shape is `data[..axis] ++ indices.dims() ++ data[axis+1..]`.
//!
//! Bounds are checked on the host before the dispatch. That is a *validation*
//! read of an input the backend has to upload anyway, not a device readback:
//! `ExecutionRecord::inputs_read_back` stays false, and the arithmetic that
//! produces every output byte runs on the device. Defining out-of-range
//! behaviour this way is the acceptance criterion "bad indices have defined
//! behavior" — a shader cannot return `Fn3Error`.
//!
//! A `Bool` table is rejected: a one-byte element cannot be written atomically
//! without `storageBuffer8BitAccess`, which the platform layer does not enable.
//! The pinned graph gathers `f32` tables and `i64` shape vectors only.

use crate::fn3::backend::check_arity;
use crate::fn3::backend::vulkan::context::BindingRole;
use crate::fn3::backend::vulkan::kernels::pointwise::{
    element_count, output_window_bytes, require_contiguous_output, require_contiguous_view,
    require_device, require_only_axes, require_output_shape, require_shader_rank, single_axis,
    view_window_bytes, workgroups, Job, Push, WORKGROUP,
};
use crate::fn3::backend::vulkan::kernels::transform::{checked_product, read_i64_values};
use crate::fn3::backend::vulkan::kernels::{grid, BindingSpec, KernelSet, KernelSpec, OpKernel};
use crate::fn3::backend::vulkan::VulkanBackend;
use crate::fn3::backend::{CapabilityFeatures, OpParams, Support};
use crate::fn3::dtype::DType;
use crate::fn3::error::{Fn3Error, Fn3Result};
use crate::fn3::ops::OpId;
use crate::fn3::shape::Shape;
use crate::fn3::tensor::{TensorMut, TensorView};

/// Table lookup over any axis.
pub const GATHER: KernelSpec = KernelSpec {
    name: "embedding.gather",
    spirv: include_bytes!("../shaders/embedding_gather.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "table"),
        BindingSpec::storage(1, "indices"),
        BindingSpec::storage(2, "out"),
    ],
    push_constant_bytes: 28,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &DType::ALL,
    shape:
        "one contiguous 1-, 4- or 8-byte data tensor, an i64 or i32 index tensor of any rank, and \
            a contiguous output of data[..axis] ++ indices ++ data[axis+1..]; indices resolved \
            from [-dim, dim) and validated on the host",
    grid: grid::elementwise,
};

/// Register the embedding kernel.
pub fn register(set: &mut KernelSet) {
    set.register(OpKernel {
        op: OpId::Gather,
        // Any axis, any index rank and every alpha element width: a one-byte
        // `Bool` table is packed four elements per invocation, so the whole
        // documented contract is implemented.
        support: Support::Full,
        features: CapabilityFeatures::default(),
        dtypes: &DType::ALL,
        kernel: &GATHER,
        launch: launch_gather,
    });
}

/// `Gather`: embedding / shape-table lookup.
pub fn launch_gather(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Gather;
    check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
    require_only_axes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let data = &inputs[0];
    let indices = &inputs[1];
    let out = &mut outputs[0];
    let index_dtype = indices.dtype();
    if index_dtype != DType::I64 && index_dtype != DType::I32 {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: DType::I64,
            actual: index_dtype,
        });
    }
    require_contiguous_view(op, data, "table")?;
    require_contiguous_view(op, indices, "indices")?;
    require_contiguous_output(op, out, "output")?;
    let rank = data.shape().rank();
    require_shader_rank(op, rank)?;
    let axis = single_axis(op, params, rank, 0, "gather axis")?;
    let width = u32::try_from(data.dtype().size_bytes()).map_err(|_| Fn3Error::InvalidShape {
        context: "vulkan gather element width",
        detail: "dtype size does not fit u32".to_string(),
    })?;

    let data_dims = data.shape().dims();
    let mut expected_dims = data_dims[..axis].to_vec();
    expected_dims.extend_from_slice(indices.shape().dims());
    expected_dims.extend_from_slice(&data_dims[axis + 1..]);
    let expected = Shape::from_dims(expected_dims);
    require_output_shape(op, out, &expected)?;
    require_shader_rank(op, expected.rank())?;

    let dimension = data_dims[axis];
    let index_values = read_i64_values(op, indices, "gather indices")?;
    let dimension_i64 = i64::try_from(dimension).map_err(|_| Fn3Error::ShapeOverflow {
        context: "vulkan gather axis extent",
        axis: Some(axis),
        detail: format!("extent {dimension} does not fit in i64"),
    })?;
    if dimension >= 1 << 31 {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: "vulkan",
            reason: format!(
                "the FN3-13 (#59) gather kernel resolves a negative index with 32-bit \
                 arithmetic, so the axis extent {dimension} must stay below 2^31"
            ),
        });
    }
    for (position, &raw) in index_values.iter().enumerate() {
        let resolved = if raw < 0 {
            raw.saturating_add(dimension_i64)
        } else {
            raw
        };
        if resolved < 0 || resolved >= dimension_i64 {
            return Err(Fn3Error::OutOfBounds {
                what: "gather index",
                index: raw as u64,
                len: dimension,
            });
        }
        let _ = position;
    }

    let elements = expected.numel()?;
    if elements == 0 {
        return Ok(());
    }
    let outer = checked_product(&data_dims[..axis])?;
    let inner = checked_product(&data_dims[axis + 1..])?;
    let push = Push::new()
        .u32(element_count(op, elements)?)
        .u32(u32::try_from(outer).map_err(|_| Fn3Error::ShapeOverflow {
            context: "vulkan gather outer extent",
            axis: Some(axis),
            detail: "outer extent does not fit u32".to_string(),
        })?)
        .u32(element_count(op, indices.shape().numel()?)?)
        .u32(u32::try_from(inner).map_err(|_| Fn3Error::ShapeOverflow {
            context: "vulkan gather inner extent",
            axis: Some(axis),
            detail: "inner extent does not fit u32".to_string(),
        })?)
        .u32(
            u32::try_from(dimension).map_err(|_| Fn3Error::ShapeOverflow {
                context: "vulkan gather axis extent",
                axis: Some(axis),
                detail: "axis extent does not fit u32".to_string(),
            })?,
        )
        .u32(index_dtype.size_bytes() as u32)
        .u32(width)
        .bytes();
    // A one-byte element packs four gathered bytes per invocation.
    let threads = if width == 1 {
        elements.div_ceil(4)
    } else {
        elements
    };
    let workgroups = workgroups(&GATHER, threads, expected.dims())?;
    let mut bytes = vec![0u8; out.meta().byte_len()?];
    let mut job = Job::new(backend, op, &GATHER)?;
    let table = job.input(view_window_bytes(data)?, "fn3.gather.table")?;
    let index_buffer = job.input(view_window_bytes(indices)?, "fn3.gather.indices")?;
    let destination = job.output(out.meta().byte_len()? as u64, "fn3.gather.out")?;
    job.step(
        push,
        workgroups,
        vec![
            (0, table, BindingRole::Input),
            (1, index_buffer, BindingRole::Input),
            (2, destination, BindingRole::Output),
        ],
    );
    job.run(&mut [&mut bytes])?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}
