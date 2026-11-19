//! The FN3-11 smoke kernel: contiguous, same-shape `f32` `Add`.
//!
//! This is the working end-to-end proof the platform layer needs, not the final
//! pointwise implementation. It deliberately supports exactly:
//!
//! - `OpId::Add`, two inputs, one output;
//! - `DType::F32`;
//! - identical, contiguous, non-broadcast shapes;
//! - any element count that fits one storage buffer, including a tail that is
//!   not a multiple of the workgroup size (the shader bounds-checks it).
//!
//! Everything else is rejected with `UnsupportedOp` naming FN3-13 (#59), which
//! owns the real pointwise family. Registering #59's kernel for `Add` replaces
//! this one ([`super::KernelSet::register`] keeps the later registration).
//!
//! The launch path is the reference for #59/#60/#61: allocate device-local
//! buffers, lease host-visible staging, write inputs, record
//! `upload → barrier → dispatch → barrier → readback` as **one** submission,
//! wait once, read the output and append a residency record.

use crate::inference::backend::check_arity;
use crate::inference::backend::vulkan::context::{
    BindingArg, BindingRole, BufferDesc, KernelInvocation,
};
use crate::inference::backend::vulkan::kernels::{
    grid, BindingSpec, KernelSet, KernelSpec, OpKernel,
};
use crate::inference::backend::vulkan::residency::ExecutionRecord;
use crate::inference::backend::vulkan::VulkanBackend;
use crate::inference::backend::Backend;
use crate::inference::backend::CapabilityFeatures;
use crate::inference::backend::OpParams;
use crate::inference::backend::Support;
use crate::inference::dtype::DType;
use crate::inference::error::{Fn3Error, Fn3Result};
use crate::inference::ops::OpId;
use crate::inference::tensor::{TensorMut, TensorView};

/// Declared kernel: `out[i] = a[i] + b[i]`.
pub const SMOKE_ADD: KernelSpec = KernelSpec {
    name: "smoke.add.f32",
    spirv: include_bytes!("../shaders/smoke_add.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "lhs"),
        BindingSpec::storage(1, "rhs"),
        BindingSpec::storage(2, "out"),
    ],
    push_constant_bytes: 4,
    workgroup_size: [64, 1, 1],
    dtypes: &[DType::F32],
    shape: "two contiguous same-shape f32 inputs, one contiguous f32 output of that shape",
    grid: grid::elementwise,
};

/// The workgroup size the shader was compiled with.
pub const SMOKE_WORKGROUP: u32 = 64;

/// Register the smoke kernel for `Add`.
pub fn register(set: &mut KernelSet) {
    set.register(OpKernel {
        op: OpId::Add,
        support: Support::partial(
            "f32 same-shape contiguous add only (the FN3-11 smoke kernel); right-aligned \
             broadcasting, strided views and the remaining elementwise ops are owned by \
             FN3-13 (#59)",
        ),
        features: CapabilityFeatures::default(),
        dtypes: &[DType::F32],
        admits: None,
        kernel: &SMOKE_ADD,
        launch: launch_add,
    });
}

/// Execute `Add` through the platform layer.
pub fn launch_add(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    check_arity(OpId::Add, inputs.len(), outputs.len(), 2, 1)?;
    if !params.axes.is_empty() || params.keep_dims || params.epsilon != 0.0 {
        return Err(Fn3Error::InvalidShape {
            context: "vulkan smoke add attributes",
            detail: format!(
                "op add accepts no attributes, but the plan supplies axes={:?}, keep_dims={}, \
                 epsilon={}",
                params.axes, params.keep_dims, params.epsilon
            ),
        });
    }
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    let out = &mut outputs[0];
    for view in [lhs, rhs] {
        view.meta().require_dtype(Some(OpId::Add), DType::F32)?;
    }
    out.meta().require_dtype(Some(OpId::Add), DType::F32)?;
    if lhs.shape() != rhs.shape() || lhs.shape() != out.shape() {
        return Err(Fn3Error::UnsupportedOp {
            op: OpId::Add,
            backend: "vulkan",
            reason: format!(
                "the FN3-11 smoke kernel implements same-shape add only: lhs {}, rhs {}, out \
                 {}. Right-aligned broadcasting is owned by FN3-13 (#59)",
                lhs.meta(),
                rhs.meta(),
                out.meta()
            ),
        });
    }
    for (name, view) in [("lhs", lhs), ("rhs", rhs)] {
        if !view.is_contiguous() {
            return Err(Fn3Error::UnsupportedOp {
                op: OpId::Add,
                backend: "vulkan",
                reason: format!(
                    "the FN3-11 smoke kernel requires contiguous operands: {name} is a strided \
                     view ({}). Strided pointwise kernels are owned by FN3-13 (#59)",
                    view.meta()
                ),
            });
        }
    }
    if !out.is_contiguous() {
        return Err(Fn3Error::UnsupportedOp {
            op: OpId::Add,
            backend: "vulkan",
            reason: format!(
                "the FN3-11 smoke kernel requires a contiguous output, found {}. Strided writes \
                 are owned by FN3-13 (#59)",
                out.meta()
            ),
        });
    }

    let numel = lhs.numel()?;
    if numel == 0 {
        return Ok(());
    }
    let numel_u32 = u32::try_from(numel).map_err(|_| Fn3Error::UnsupportedOp {
        op: OpId::Add,
        backend: "vulkan",
        reason: format!(
            "{numel} elements do not fit the smoke kernel's 32-bit element index; chunked \
             dispatch is owned by FN3-13 (#59)"
        ),
    })?;
    let bytes = numel.checked_mul(4).ok_or(Fn3Error::ShapeOverflow {
        context: "vulkan smoke add",
        axis: None,
        detail: format!("{numel} f32 elements"),
    })?;
    backend.capabilities().check_buffer(bytes)?;

    let kernel = backend.kernel(&SMOKE_ADD)?;
    let lhs_host = lhs.as_f32_slice()?;
    let rhs_host = rhs.as_f32_slice()?;

    let lhs_buffer = backend.alloc_buffer(&BufferDesc::device_storage(bytes, "add.lhs"))?;
    let rhs_buffer = backend.alloc_buffer(&BufferDesc::device_storage(bytes, "add.rhs"))?;
    let out_buffer = backend.alloc_buffer(&BufferDesc::device_storage(bytes, "add.out"))?;

    let staging = backend.staging();
    let lhs_staging = staging.acquire(bytes).map_err(|error| error.to_fn3())?;
    let rhs_staging = staging.acquire(bytes).map_err(|error| error.to_fn3())?;
    let readback = staging.acquire(bytes).map_err(|error| error.to_fn3())?;
    lhs_staging
        .write(bytemuck::cast_slice(lhs_host))
        .map_err(|error| error.to_fn3())?;
    rhs_staging
        .write(bytemuck::cast_slice(rhs_host))
        .map_err(|error| error.to_fn3())?;

    let bindings = [
        BindingArg::whole(0, BindingRole::Input, &lhs_buffer),
        BindingArg::whole(1, BindingRole::Input, &rhs_buffer),
        BindingArg::whole(2, BindingRole::Output, &out_buffer),
    ];
    let push = numel_u32.to_le_bytes();
    let workgroups = kernel
        .spec()
        .workgroups(numel, out.shape().dims())
        .map_err(|error| error.to_fn3())?;

    let mut submitter = backend.checkout_submitter()?;
    let mut dispatch = None;
    let submission = {
        let lhs_staging_buffer = lhs_staging.buffer();
        let rhs_staging_buffer = rhs_staging.buffer();
        let readback_buffer = readback.buffer();
        submitter
            .record(|recorder| {
                recorder.copy_buffer(lhs_staging_buffer, 0, &lhs_buffer, 0, bytes)?;
                recorder.copy_buffer(rhs_staging_buffer, 0, &rhs_buffer, 0, bytes)?;
                for buffer in [&lhs_buffer, &rhs_buffer, &out_buffer] {
                    recorder.barrier(
                        buffer,
                        ash::vk::PipelineStageFlags::TRANSFER,
                        ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                        ash::vk::AccessFlags::TRANSFER_WRITE,
                        ash::vk::AccessFlags::SHADER_READ,
                    );
                }
                let info = recorder.dispatch(&KernelInvocation {
                    kernel: &kernel,
                    bindings: &bindings,
                    push_constants: &push,
                    workgroups,
                })?;
                recorder.barrier(
                    &out_buffer,
                    ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                    ash::vk::PipelineStageFlags::TRANSFER,
                    ash::vk::AccessFlags::SHADER_WRITE,
                    ash::vk::AccessFlags::TRANSFER_READ,
                );
                recorder.copy_buffer(&out_buffer, 0, readback_buffer, 0, bytes)?;
                dispatch = Some(info);
                Ok(())
            })
            .map_err(|error| error.to_fn3())?
    };
    let report = submission.wait().map_err(|error| error.to_fn3())?;
    drop(submitter);

    let mut output_bytes = vec![0u8; bytes as usize];
    readback
        .read_into(&mut output_bytes)
        .map_err(|error| error.to_fn3())?;
    {
        let out_slice = out.as_mut_slice::<f32>()?;
        let out_bytes = bytemuck::cast_slice_mut::<f32, u8>(out_slice);
        if out_bytes.len() != output_bytes.len() {
            return Err(Fn3Error::CapacityExceeded {
                what: "vulkan smoke add output copy",
                requested: output_bytes.len() as u64,
                capacity: out_bytes.len() as u64,
            });
        }
        out_bytes.copy_from_slice(&output_bytes);
    }

    let dispatch = dispatch.ok_or_else(|| Fn3Error::InvalidShape {
        context: "vulkan smoke add",
        detail: "the recorder produced no dispatch information".to_string(),
    })?;
    let mut record = ExecutionRecord::new(0, OpId::Add, SMOKE_ADD.name, backend.device_id());
    record.submissions = report.submissions;
    record.workgroups = dispatch.workgroups;
    record.bindings = dispatch.bindings;
    record.upload_bytes = 2 * bytes;
    record.readback_bytes = bytes;
    record.inputs_read_back = false;
    record.host_staged = true;
    record.device_local_bytes = 3 * bytes;
    record.host_ns = report.host_ns;
    record.gpu_ns = report.gpu_ns;
    record.gpu_ticks = report.timestamps;
    backend.residency().record(record);

    Ok(())
}

/// The smoke kernel's declared spec, for diagnostics.
pub fn spec() -> &'static KernelSpec {
    &SMOKE_ADD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_matches_the_committed_shader_metadata() {
        assert_eq!(SMOKE_ADD.workgroup_size, [SMOKE_WORKGROUP, 1, 1]);
        assert_eq!(SMOKE_ADD.push_constant_bytes, 4);
        assert_eq!(SMOKE_ADD.bindings.len(), 3);
        assert!(SMOKE_ADD.validate().is_ok());
        assert_eq!(
            SMOKE_ADD.workgroups(65, &[65]).expect("grid computes"),
            [2, 1, 1]
        );
    }
}
