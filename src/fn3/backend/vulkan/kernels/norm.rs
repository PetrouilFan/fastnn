//! GPU kernels owned by FN3-15 (#61): LayerNorm and Softmax.
//!
//! # Contract
//!
//! Both kernels reproduce the CPU reference arithmetic of
//! `backend/cpu/kernels/{norm,softmax}.rs`, which is the numerical oracle:
//!
//! ```text
//! LayerNorm:  mean = sum(x[a])/n;  var = sum((x[a]-mean)^2)/n       (biased)
//!             out[a] = (x[a]-mean)/sqrt(var+eps)*w[a] + b[a]
//!
//! Softmax:    out[a] = exp(x[a]-max(x)) / sum(exp(x-max))           (ascending a)
//! ```
//!
//! One invocation owns one row and accumulates ascending with a single `f32`
//! accumulator, so the summation order is the reference order rather than a
//! tree order. That costs intra-row parallelism — the row count is the parallel
//! axis — and buys bit-comparable results for the small row counts the pinned
//! graph produces (`hidden` 128, key extent ≤ 17 in the fixtures).
//!
//! Max subtraction in Softmax is a conformance requirement, not an
//! optimisation: the pinned mask fills with the finite value `-FLT_MAX`, and an
//! all-masked row must be exactly uniform (`1/width`), never NaN. See
//! `shaders/fn3_softmax_last.comp`.
//!
//! # Scope
//!
//! | op | supported | rejected by name |
//! |---|---|---|
//! | `LayerNorm` | `f32`, trailing axis only, rank-1 `weight`/`bias`, contiguous | multi-axis norm, strided views |
//! | `Softmax` | `f32`, trailing axis, contiguous | any other axis, strided views |
//!
//! `Support::Full` for `LayerNorm` is honest because the fn3 LayerNorm *contract*
//! normalizes the trailing axis only; `Softmax` reports `Partial` because the
//! CPU reference accepts any axis and this kernel does not.
//!
//! # Shared launch plumbing
//!
//! [`execute_staged`] is the one place the FN3-15 kernels allocate device
//! buffers, lease host-visible staging, record `upload → barrier → dispatch →
//! barrier → readback` as a single submission, wait once, read back and append a
//! residency record. `reduce.rs` and `attention.rs` use it too, so every kernel
//! in this family produces the same evidence fields.

use ash::vk;

use super::super::context::{BindingArg, BindingRole, Buffer, BufferDesc, KernelInvocation};
use super::super::error::Error;
use super::super::residency::ExecutionRecord;
use super::super::staging::StagingLease;
use super::super::VulkanBackend;
use super::{BindingKind, BindingSpec, KernelSet, KernelSpec, OpKernel};
use crate::fn3::backend::vulkan::VULKAN_BACKEND_NAME;
use crate::fn3::backend::{check_arity, Backend, CapabilityFeatures, OpParams, Support};
use crate::fn3::dtype::DType;
use crate::fn3::error::{Fn3Error, Fn3Result};
use crate::fn3::ops::OpId;
use crate::fn3::shape::Shape;
use crate::fn3::tensor::{TensorMut, TensorView};

/// LayerNorm over the trailing axis (`shaders/fn3_layernorm.comp`).
pub const LAYER_NORM: KernelSpec = KernelSpec {
    name: "norm.layernorm.f32",
    spirv: include_bytes!("../shaders/fn3_layernorm.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "input"),
        BindingSpec::storage(1, "weight"),
        BindingSpec::storage(2, "bias"),
        BindingSpec::storage(3, "out"),
    ],
    push_constant_bytes: 12,
    workgroup_size: [64, 1, 1],
    dtypes: &[DType::F32],
    shape: "contiguous f32 [rows, hidden] with rank-1 weight and bias of extent hidden",
    grid: super::grid::elementwise,
};

/// Softmax over the trailing axis (`shaders/fn3_softmax_last.comp`).
pub const SOFTMAX_LAST: KernelSpec = KernelSpec {
    name: "norm.softmax.last.f32",
    spirv: include_bytes!("../shaders/fn3_softmax_last.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "input"),
        BindingSpec::storage(1, "out"),
    ],
    push_constant_bytes: 8,
    workgroup_size: [64, 1, 1],
    dtypes: &[DType::F32],
    shape: "contiguous f32, rows x width, one invocation per row, ascending accumulation",
    grid: super::grid::elementwise,
};

// ---------------------------------------------------------------------------
// Shared launch plumbing (used by `reduce.rs` and `attention.rs` as well)
// ---------------------------------------------------------------------------

/// One host-visible source for a declared storage-buffer binding.
pub(super) enum BufferSource<'a> {
    /// Host bytes copied to a device buffer before the dispatch.
    Upload {
        /// Diagnostic buffer name.
        name: &'static str,
        /// Bytes to upload; length must be a multiple of 4.
        bytes: &'a [u8],
    },
    /// A device buffer the kernel writes and the host reads after the wait.
    Readback {
        /// Diagnostic buffer name.
        name: &'static str,
        /// Destination for the readback; its length is the byte count.
        bytes: &'a mut [u8],
    },
    /// A device buffer the kernel owns and the host never touches.
    Scratch {
        /// Diagnostic buffer name.
        name: &'static str,
        /// Byte size; must be a non-zero multiple of 4.
        size: u64,
    },
}

impl BufferSource<'_> {
    fn name(&self) -> &'static str {
        match self {
            BufferSource::Upload { name, .. }
            | BufferSource::Readback { name, .. }
            | BufferSource::Scratch { name, .. } => name,
        }
    }

    fn size(&self) -> u64 {
        match self {
            BufferSource::Upload { bytes, .. } => bytes.len() as u64,
            BufferSource::Readback { bytes, .. } => bytes.len() as u64,
            BufferSource::Scratch { size, .. } => *size,
        }
    }

    fn role(&self) -> BindingRole {
        match self {
            BufferSource::Upload { .. } => BindingRole::Input,
            BufferSource::Readback { .. } => BindingRole::Output,
            BufferSource::Scratch { .. } => BindingRole::Scratch,
        }
    }
}

/// Record and wait for one dispatch over device-resident buffers.
///
/// `sources` must have exactly one entry per declared binding, in declaration
/// order. Every input is uploaded, every [`BufferSource::Readback`] is filled
/// from the device, and one [`ExecutionRecord`] is appended with the submission,
/// dispatch geometry, bindings, transfer sizes and device timestamps.
pub(super) fn execute_staged(
    op: OpId,
    backend: &VulkanBackend,
    spec: &'static KernelSpec,
    sources: &mut [BufferSource<'_>],
    push_constants: &[u8],
    workgroups: [u32; 3],
) -> Fn3Result<()> {
    if sources.len() != spec.bindings.len() {
        return Err(Error::internal(format!(
            "{}: {} buffer source(s) for {} declared binding(s)",
            spec.name,
            sources.len(),
            spec.bindings.len()
        ))
        .to_fn3());
    }
    if push_constants.len() != spec.push_constant_bytes as usize {
        return Err(Error::internal(format!(
            "{}: {} push-constant bytes supplied but the spec declares {}",
            spec.name,
            push_constants.len(),
            spec.push_constant_bytes
        ))
        .to_fn3());
    }
    for (index, source) in sources.iter().enumerate() {
        let declared = &spec.bindings[index];
        if declared.kind != BindingKind::StorageBuffer {
            return Err(Error::internal(format!(
                "{}: binding {} ({}) is a {}; the FN3-15 launch helper only binds storage \
                 buffers",
                spec.name,
                declared.binding,
                declared.name,
                declared.kind.name()
            ))
            .to_fn3());
        }
        let size = source.size();
        if size == 0 {
            return Err(Error::internal(format!(
                "{}: binding {} ({}) is a zero-byte buffer; handle zero-element tensors before \
                 allocation",
                spec.name,
                declared.binding,
                source.name()
            ))
            .to_fn3());
        }
        if !size.is_multiple_of(4) {
            return Err(Error::internal(format!(
                "{}: binding {} ({}) is {size} bytes, which vkCmdCopyBuffer cannot move",
                spec.name,
                declared.binding,
                source.name()
            ))
            .to_fn3());
        }
        backend.capabilities().check_buffer(size)?;
    }

    let kernel = backend.kernel(spec)?;
    let mut buffers: Vec<Buffer> = Vec::with_capacity(sources.len());
    for source in sources.iter() {
        buffers
            .push(backend.alloc_buffer(&BufferDesc::device_storage(source.size(), source.name()))?);
    }

    let staging = backend.staging();
    let mut leases: Vec<Option<StagingLease<'_>>> = Vec::with_capacity(sources.len());
    for source in sources.iter() {
        let lease = match source {
            BufferSource::Scratch { .. } => None,
            _ => Some(
                staging
                    .acquire(source.size())
                    .map_err(|error| error.to_fn3())?,
            ),
        };
        leases.push(lease);
    }
    for (index, source) in sources.iter().enumerate() {
        if let BufferSource::Upload { bytes, .. } = source {
            leases[index]
                .as_ref()
                .expect("every upload source leased staging")
                .write(bytes)
                .map_err(|error| error.to_fn3())?;
        }
    }

    let bindings: Vec<BindingArg<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(index, buffer)| {
            BindingArg::whole(spec.bindings[index].binding, sources[index].role(), buffer)
        })
        .collect();

    let mut submitter = backend.checkout_submitter()?;
    let mut dispatch = None;
    let (submission_report, dispatch_info) = {
        let report = submitter
            .record(|recorder| {
                for (index, source) in sources.iter().enumerate() {
                    if matches!(source, BufferSource::Upload { .. }) {
                        recorder.copy_buffer(
                            leases[index].as_ref().expect("upload lease").buffer(),
                            0,
                            &buffers[index],
                            0,
                            source.size(),
                        )?;
                        recorder.barrier(
                            &buffers[index],
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::AccessFlags::TRANSFER_WRITE,
                            vk::AccessFlags::SHADER_READ,
                        );
                    }
                }
                dispatch = Some(recorder.dispatch(&KernelInvocation {
                    kernel: &kernel,
                    bindings: &bindings,
                    push_constants,
                    workgroups,
                })?);
                for (index, source) in sources.iter().enumerate() {
                    if matches!(source, BufferSource::Readback { .. }) {
                        recorder.barrier(
                            &buffers[index],
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::AccessFlags::SHADER_WRITE,
                            vk::AccessFlags::TRANSFER_READ,
                        );
                        recorder.copy_buffer(
                            &buffers[index],
                            0,
                            leases[index].as_ref().expect("readback lease").buffer(),
                            0,
                            source.size(),
                        )?;
                    }
                }
                Ok(())
            })
            .map_err(|error| error.to_fn3())?;
        let report = report.wait().map_err(|error| error.to_fn3())?;
        let info = dispatch.ok_or_else(|| Fn3Error::InvalidShape {
            context: "vulkan kernel dispatch",
            detail: format!(
                "{}: the recorder produced no dispatch information",
                spec.name
            ),
        })?;
        (report, info)
    };
    drop(submitter);

    for (index, source) in sources.iter_mut().enumerate() {
        if let BufferSource::Readback { bytes, .. } = source {
            leases[index]
                .as_ref()
                .expect("readback lease")
                .read_into(bytes)
                .map_err(|error| error.to_fn3())?;
        }
    }

    let mut upload_bytes = 0u64;
    let mut readback_bytes = 0u64;
    let mut device_local_bytes = 0u64;
    for source in sources.iter() {
        device_local_bytes = device_local_bytes.saturating_add(source.size());
        match source {
            BufferSource::Upload { .. } => {
                upload_bytes = upload_bytes.saturating_add(source.size());
            }
            BufferSource::Readback { .. } => {
                readback_bytes = readback_bytes.saturating_add(source.size());
            }
            BufferSource::Scratch { .. } => {}
        }
    }

    let mut record = ExecutionRecord::new(0, op, spec.name, backend.device_id());
    record.submissions = submission_report.submissions;
    record.workgroups = dispatch_info.workgroups;
    record.bindings = dispatch_info.bindings;
    record.upload_bytes = upload_bytes;
    record.readback_bytes = readback_bytes;
    // No input is ever copied back to the host: inputs travel host → device and
    // only declared outputs travel device → host.
    record.inputs_read_back = false;
    record.host_staged = true;
    record.device_local_bytes = device_local_bytes;
    record.host_ns = submission_report.host_ns;
    record.gpu_ns = submission_report.gpu_ns;
    record.gpu_ticks = submission_report.timestamps;
    backend.record_execution(record);

    Ok(())
}

/// Build the little-endian parameter bytes the kernels read as `uint`/`float`.
///
/// Member order matters: the shaders read the block as a flat sequence of
/// 4-byte scalars (and `uint[8]` arrays with a stride of 4 under std430), so a
/// writer call per member in declaration order is the ABI.
#[derive(Default)]
pub(super) struct ParamWriter {
    bytes: Vec<u8>,
}

impl ParamWriter {
    pub(super) fn new() -> ParamWriter {
        ParamWriter::default()
    }

    pub(super) fn u32(&mut self, value: u32) -> &mut ParamWriter {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub(super) fn f32(&mut self, value: f32) -> &mut ParamWriter {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub(super) fn u32_slice(&mut self, values: &[u32]) -> &mut ParamWriter {
        for value in values {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }
        self
    }

    pub(super) fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// An `UnsupportedOp` naming this issue, as the honest-rejection path requires.
pub(super) fn unsupported(op: OpId, reason: impl Into<String>) -> Fn3Error {
    Fn3Error::UnsupportedOp {
        op,
        backend: VULKAN_BACKEND_NAME,
        reason: format!("{} (owner: FN3-15 / #61)", reason.into()),
    }
}

/// Resolve a possibly-negative axis against `rank`.
pub(super) fn normalize_axis(
    op: OpId,
    axis: i64,
    rank: usize,
    context: &'static str,
) -> Fn3Result<usize> {
    let rank = i64::try_from(rank).map_err(|_| Fn3Error::ShapeOverflow {
        context,
        axis: None,
        detail: "tensor rank does not fit in i64".to_string(),
    })?;
    let resolved = if axis < 0 { axis + rank } else { axis };
    if resolved < 0 || resolved >= rank {
        return Err(Fn3Error::InvalidShape {
            context,
            detail: format!("op {op}: axis {axis} is out of range for rank {rank}"),
        });
    }
    Ok(resolved as usize)
}

/// `require_dtype` plus the contiguous-view precondition of every FN3-15 kernel.
pub(super) fn contiguous_f32<'a>(
    op: OpId,
    view: &TensorView<'a>,
    what: &'static str,
) -> Fn3Result<&'a [f32]> {
    view.meta().require_dtype(Some(op), DType::F32)?;
    if !view.is_contiguous() {
        return Err(unsupported(
            op,
            format!(
                "{what} is a strided view ({}), and the FN3-15 kernels require contiguous \
                 operands; materialise it with `to_contiguous` first",
                view.strides()
            ),
        ));
    }
    view.as_f32_slice()
}

/// Mutable twin of [`contiguous_f32`], returning raw little-endian bytes.
pub(super) fn contiguous_f32_bytes_mut<'a>(
    op: OpId,
    view: &'a mut TensorMut<'_>,
    what: &'static str,
) -> Fn3Result<&'a mut [u8]> {
    view.meta().require_dtype(Some(op), DType::F32)?;
    if !view.is_contiguous() {
        return Err(unsupported(
            op,
            format!(
                "{what} is a strided view ({}), and the FN3-15 kernels require contiguous \
                 outputs",
                view.strides()
            ),
        ));
    }
    let values = view.as_mut_slice::<f32>()?;
    Ok(bytemuck::cast_slice_mut::<f32, u8>(values))
}

/// Require `actual` to equal `expected` and report both when it does not.
pub(super) fn require_shape(op: OpId, expected: &Shape, actual: &Shape) -> Fn3Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: expected.to_string(),
            actual: actual.to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

/// Register the FN3-15 normalization kernels.
pub fn register(set: &mut KernelSet) {
    set.register(OpKernel {
        op: OpId::LayerNorm,
        // The fn3 LayerNorm contract is "normalize the trailing axis with
        // rank-1 weight/bias", which this kernel implements in full. Requests
        // outside it (multi-axis normalization) are rejected by name.
        support: Support::Full,
        features: CapabilityFeatures::default(),
        dtypes: &[DType::F32],
        kernel: &LAYER_NORM,
        launch: launch_layer_norm,
    });
    set.register(OpKernel {
        op: OpId::Softmax,
        support: Support::partial(
            "f32 softmax over the trailing axis of a contiguous operand. The CPU reference \
             also accepts an arbitrary axis; that case is rejected by name (owner: FN3-15 / \
             #61)",
        ),
        features: CapabilityFeatures::default(),
        dtypes: &[DType::F32],
        kernel: &SOFTMAX_LAST,
        launch: launch_softmax,
    });
}

/// Execute `LayerNorm` through the platform layer.
pub fn launch_layer_norm(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::LayerNorm;
    check_arity(op, inputs.len(), outputs.len(), 3, 1)?;
    let input = &inputs[0];
    let weight = &inputs[1];
    let bias = &inputs[2];
    let out = &mut outputs[0];

    let rank = input.shape().rank();
    if rank == 0 {
        return Err(Fn3Error::RankMismatch {
            op: Some(op),
            expected: 1,
            actual: 0,
        });
    }
    if !params.epsilon.is_finite() || params.epsilon < 0.0 {
        return Err(Fn3Error::InvalidShape {
            context: "vulkan layer_norm epsilon",
            detail: format!(
                "op {op}: epsilon {} must be finite and non-negative",
                params.epsilon
            ),
        });
    }
    match params.axes.as_slice() {
        [] => {}
        [axis] => {
            let axis = normalize_axis(op, *axis, rank, "vulkan layer_norm axes")?;
            if axis != rank - 1 {
                return Err(unsupported(
                    op,
                    format!(
                        "the fn3 LayerNorm contract normalizes the trailing axis only; axis \
                         {axis} of {} was requested",
                        input.shape()
                    ),
                ));
            }
        }
        axes => {
            return Err(unsupported(
                op,
                format!(
                    "the fn3 LayerNorm contract normalizes the trailing axis only; axes {axes:?} \
                     were requested"
                ),
            ))
        }
    }
    require_shape(op, input.shape(), out.shape())?;

    let hidden = input.shape().dims()[rank - 1];
    for (view, what) in [(weight, "weight"), (bias, "bias")] {
        if view.shape().rank() != 1 || view.shape().dims()[0] != hidden {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(op),
                expected: format!("[{hidden}] for layer_norm {what}"),
                actual: view.shape().to_string(),
            });
        }
    }

    let input_values = contiguous_f32(op, input, "layer_norm input")?;
    let weight_values = contiguous_f32(op, weight, "layer_norm weight")?;
    let bias_values = contiguous_f32(op, bias, "layer_norm bias")?;
    let out_dims: Vec<u64> = out.shape().dims().to_vec();
    let out_bytes = contiguous_f32_bytes_mut(op, out, "layer_norm output")?;

    let rows = input
        .shape()
        .dims()
        .iter()
        .take(rank - 1)
        .try_fold(1u64, |accumulator, &extent| accumulator.checked_mul(extent))
        .ok_or(Fn3Error::ShapeOverflow {
            context: "vulkan layer_norm rows",
            axis: None,
            detail: format!("leading extents of {} exceed u64", input.shape()),
        })?;
    if rows == 0 || hidden == 0 {
        return Ok(());
    }
    let rows = u32::try_from(rows).map_err(|_| {
        unsupported(
            op,
            format!(
                "{rows} rows do not fit the 32-bit dispatch coordinate; split the tensor \
                 (the row-chunked kernel belongs to #61's follow-up)"
            ),
        )
    })?;
    let hidden_u32 = u32::try_from(hidden).map_err(|_| Fn3Error::ShapeOverflow {
        context: "vulkan layer_norm hidden",
        axis: Some(rank - 1),
        detail: "hidden does not fit in u32".to_string(),
    })?;

    let mut push = ParamWriter::new();
    push.u32(rows).u32(hidden_u32).f32(params.epsilon);
    let workgroups = LAYER_NORM
        .workgroups(u64::from(rows), &out_dims)
        .map_err(|error| error.to_fn3())?;

    let mut sources = [
        BufferSource::Upload {
            name: "layernorm.input",
            bytes: bytemuck::cast_slice(input_values),
        },
        BufferSource::Upload {
            name: "layernorm.weight",
            bytes: bytemuck::cast_slice(weight_values),
        },
        BufferSource::Upload {
            name: "layernorm.bias",
            bytes: bytemuck::cast_slice(bias_values),
        },
        BufferSource::Readback {
            name: "layernorm.out",
            bytes: out_bytes,
        },
    ];
    execute_staged(
        op,
        backend,
        &LAYER_NORM,
        &mut sources,
        &push.finish(),
        workgroups,
    )
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// Execute `Softmax` over the trailing axis through the platform layer.
pub fn launch_softmax(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Softmax;
    check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
    let input = &inputs[0];
    let out = &mut outputs[0];

    let rank = input.shape().rank();
    if rank == 0 {
        return Err(Fn3Error::RankMismatch {
            op: Some(op),
            expected: 1,
            actual: 0,
        });
    }
    let axis = match params.axes.as_slice() {
        [] => rank - 1,
        [axis] => normalize_axis(op, *axis, rank, "vulkan softmax axis")?,
        axes => {
            return Err(Fn3Error::InvalidShape {
                context: "vulkan softmax axis",
                detail: format!("op {op} takes one axis, but the plan supplies {axes:?}"),
            })
        }
    };
    if axis != rank - 1 {
        return Err(unsupported(
            op,
            format!(
                "this kernel implements the trailing axis only (the fn3 softmax contract and \
                 the pinned graph); axis {axis} of {} was requested",
                input.shape()
            ),
        ));
    }
    require_shape(op, input.shape(), out.shape())?;

    let input_values = contiguous_f32(op, input, "softmax input")?;
    let out_dims: Vec<u64> = out.shape().dims().to_vec();
    let out_bytes = contiguous_f32_bytes_mut(op, out, "softmax output")?;

    let rows = input
        .shape()
        .dims()
        .iter()
        .take(rank - 1)
        .try_fold(1u64, |accumulator, &extent| accumulator.checked_mul(extent))
        .ok_or(Fn3Error::ShapeOverflow {
            context: "vulkan softmax rows",
            axis: None,
            detail: format!("leading extents of {} exceed u64", input.shape()),
        })?;
    let width = input.shape().dims()[rank - 1];
    if rows == 0 || width == 0 {
        return Ok(());
    }
    let rows = u32::try_from(rows).map_err(|_| {
        unsupported(
            op,
            format!("{rows} rows do not fit the 32-bit dispatch coordinate; split the tensor"),
        )
    })?;
    let width = u32::try_from(width).map_err(|_| Fn3Error::ShapeOverflow {
        context: "vulkan softmax width",
        axis: Some(rank - 1),
        detail: "width does not fit in u32".to_string(),
    })?;

    let mut push = ParamWriter::new();
    push.u32(rows).u32(width);
    let workgroups = SOFTMAX_LAST
        .workgroups(u64::from(rows), &out_dims)
        .map_err(|error| error.to_fn3())?;

    let mut sources = [
        BufferSource::Upload {
            name: "softmax.input",
            bytes: bytemuck::cast_slice(input_values),
        },
        BufferSource::Readback {
            name: "softmax.out",
            bytes: out_bytes,
        },
    ];
    execute_staged(
        op,
        backend,
        &SOFTMAX_LAST,
        &mut sources,
        &push.finish(),
        workgroups,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_match_the_committed_shader_metadata() {
        assert_eq!(LAYER_NORM.workgroup_size, [64, 1, 1]);
        assert_eq!(LAYER_NORM.push_constant_bytes, 12);
        assert_eq!(LAYER_NORM.bindings.len(), 4);
        assert!(LAYER_NORM.validate().is_ok());
        assert_eq!(
            LAYER_NORM.workgroups(65, &[65, 5]).expect("grid"),
            [2, 1, 1]
        );

        assert_eq!(SOFTMAX_LAST.workgroup_size, [64, 1, 1]);
        assert_eq!(SOFTMAX_LAST.push_constant_bytes, 8);
        assert_eq!(SOFTMAX_LAST.bindings.len(), 2);
        assert!(SOFTMAX_LAST.validate().is_ok());
    }

    #[test]
    fn param_writer_is_a_flat_little_endian_scalar_stream() {
        let mut writer = ParamWriter::new();
        writer.u32(1).u32(2).f32(0.5).u32_slice(&[3, 4]);
        let bytes = writer.finish();
        assert_eq!(bytes.len(), 20);
        assert_eq!(&bytes[..4], &1u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &2u32.to_le_bytes());
        assert_eq!(&bytes[8..12], &0.5f32.to_le_bytes());
        assert_eq!(&bytes[12..16], &3u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &4u32.to_le_bytes());
    }

    #[test]
    fn negative_axes_resolve_and_out_of_range_axes_are_rejected() {
        assert_eq!(normalize_axis(OpId::LayerNorm, -1, 3, "test").unwrap(), 2);
        assert_eq!(normalize_axis(OpId::LayerNorm, 0, 3, "test").unwrap(), 0);
        assert!(normalize_axis(OpId::LayerNorm, 3, 3, "test").is_err());
        assert!(normalize_axis(OpId::LayerNorm, -4, 3, "test").is_err());
    }
}
