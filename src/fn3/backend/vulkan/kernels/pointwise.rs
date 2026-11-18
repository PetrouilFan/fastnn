//! GPU kernels owned by FN3-13 (#59): pointwise arithmetic, comparison,
//! selection and explicit dtype conversion.
//!
//! # What is registered here
//!
//! | op | kernel | support |
//! |---|---|---|
//! | `Add` `Sub` `Mul` `Div` `Pow` | `pointwise.binary.f32` | full |
//! | `Neg` `Sqrt` `Tanh` `Relu` `Sin` `Cos` `Erf` `Gelu` | `pointwise.unary.f32` | full |
//! | `Where` | `pointwise.where` | full |
//! | `Equal` `Less` `Greater` | `pointwise.compare` | full |
//! | `Cast` | `cast.to.{f32,i32,i64,bool}` | full |
//!
//! `Add` is registered here with right-aligned broadcasting and
//! `features.broadcasting = true`, which replaces the FN3-11 smoke kernel's
//! same-shape-only `Partial` registration ([`super::KernelSet::register`] keeps
//! the later registration).
//!
//! # Where the numerics come from
//!
//! The CPU reference (`src/fn3/backend/cpu/kernels/`) is the contract:
//!
//! - `Add`/`Sub`/`Mul`/`Div` are one IEEE-754 `f32` operation per element, so
//!   they are bit-identical to the reference.
//! - `Pow` reproduces C `pow` including the negative-base case (`x^2`, `x^3`),
//!   which GLSL's `pow` leaves undefined. Without that branch the pinned graph's
//!   `gelu_new` cube would be NaN for negative activations.
//! - `Sqrt`, `Sin`, `Cos` and `Tanh` use the GPU's GLSL.std.450
//!   implementations, which are within a few `f32` ulp of the platform libm the
//!   reference calls.
//! - `Erf` uses Abramowitz & Stegun 7.1.26 (|error| <= 1.5e-7 absolute) and
//!   `Gelu` is the exact `0.5*x*(1+erf(x/sqrt(2)))` form. The parity fixture
//!   bound for GELU is `atol 2e-6 + rtol 1e-6`, which this meets with a factor
//!   of about ten to spare.
//! - `Cast` implements all sixteen dtype pairs with the reference's rules
//!   (truncate-and-saturate for float to integer, non-zero for `Bool`); only
//!   `i64 <-> f32` differs above 2^24, by at most one `f32` ulp.
//!
//! # Layout preconditions
//!
//! The alpha Vulkan capability table says `supports_strided_views = false`, so
//! every kernel here requires contiguous row-major windows and rejects a
//! strided view with `UnsupportedOp` naming #59. Broadcasting is a *shape* rule
//! and is fully implemented; it is not a layout fallback.
//!
//! # Shared launch plumbing
//!
//! [`Job`] is the single-submission executor every FN3-13 kernel uses: it
//! allocates device buffers, uploads through host-visible staging, records
//! `copy -> barrier -> dispatch+ -> barrier -> copy` as one submission, waits
//! once, reads the outputs and appends one [`ExecutionRecord`]. It lives here
//! because the pointwise family is the lowest-level one and `transform` /
//! `embedding` are its siblings in the same module; `kernels/mod.rs` is the
//! shared seam and is not ours to grow.

use std::sync::Arc;

use ash::vk;

use crate::fn3::backend::check_arity;
use crate::fn3::backend::vulkan::context::{
    BindingArg, BindingRole, Buffer, BufferDesc, KernelInvocation,
};
use crate::fn3::backend::vulkan::kernels::{grid, BindingSpec, KernelSet, KernelSpec, OpKernel};
use crate::fn3::backend::vulkan::pipeline::Kernel;
use crate::fn3::backend::vulkan::residency::ExecutionRecord;
use crate::fn3::backend::vulkan::staging::StagingLease;
use crate::fn3::backend::vulkan::submission::DispatchInfo;
use crate::fn3::backend::vulkan::VulkanBackend;
use crate::fn3::backend::{CapabilityFeatures, OpParams, Support};
use crate::fn3::dtype::DType;
use crate::fn3::error::{Fn3Error, Fn3Result};
use crate::fn3::ops::OpId;
use crate::fn3::shape::Shape;
use crate::fn3::tensor::{TensorMut, TensorView};

/// Backend name used in every rejection.
pub(super) const BACKEND: &str = "vulkan";

/// Workgroup size of every FN3-13 shader (matches `layout(local_size_x)`).
pub(super) const WORKGROUP: u32 = 64;

/// The largest rank the shaders can decompose (one `uint` per axis).
pub(super) const MAX_RANK: usize = 8;

// ---------------------------------------------------------------------------
// Kernel declarations
// ---------------------------------------------------------------------------

/// Right-aligned broadcast binary arithmetic on `f32`.
pub const BINARY: KernelSpec = KernelSpec {
    name: "pointwise.binary.f32",
    spirv: include_bytes!("../shaders/pointwise_binary.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "lhs"),
        BindingSpec::storage(1, "rhs"),
        BindingSpec::storage(2, "out"),
    ],
    push_constant_bytes: 108,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32],
    shape: "contiguous f32 operands of rank <= 8 with right-aligned broadcasting; contiguous f32 \
            output of the broadcast shape",
    grid: grid::elementwise,
};

/// Elementwise unary `f32` functions.
pub const UNARY: KernelSpec = KernelSpec {
    name: "pointwise.unary.f32",
    spirv: include_bytes!("../shaders/pointwise_unary.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "input"),
        BindingSpec::storage(1, "out"),
    ],
    push_constant_bytes: 8,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32],
    shape: "one contiguous f32 operand and a contiguous f32 output of the same shape",
    grid: grid::elementwise,
};

/// `Where(cond, x, y)` with three-way broadcasting.
pub const WHERE: KernelSpec = KernelSpec {
    name: "pointwise.where",
    spirv: include_bytes!("../shaders/pointwise_where.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "cond"),
        BindingSpec::storage(1, "x"),
        BindingSpec::storage(2, "y"),
        BindingSpec::storage(3, "out"),
    ],
    push_constant_bytes: 140,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32, DType::I32, DType::I64],
    shape: "a Bool condition and two 4- or 8-byte operands of one dtype, all right-aligned \
            broadcast to the contiguous output shape",
    grid: grid::elementwise,
};

/// `Equal`/`Less`/`Greater`, producing `Bool`.
pub const COMPARE: KernelSpec = KernelSpec {
    name: "pointwise.compare",
    spirv: include_bytes!("../shaders/pointwise_compare.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "lhs"),
        BindingSpec::storage(1, "rhs"),
        BindingSpec::storage(2, "out"),
    ],
    push_constant_bytes: 112,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32, DType::I32, DType::I64, DType::Bool],
    shape: "two operands of one dtype broadcast to a Bool output; four verdicts per invocation \
            because a Bool element is one byte",
    grid: grid::elementwise,
};

/// `Cast` into `f32`.
pub const CAST_TO_F32: KernelSpec = KernelSpec {
    name: "cast.to.f32",
    spirv: include_bytes!("../shaders/cast_to_f32.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "input"),
        BindingSpec::storage(1, "out"),
    ],
    push_constant_bytes: 8,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32, DType::I32, DType::I64, DType::Bool],
    shape: "one contiguous operand of any alpha dtype, cast to a contiguous f32 output of the \
            same shape",
    grid: grid::elementwise,
};

/// `Cast` into `i32`.
pub const CAST_TO_I32: KernelSpec = KernelSpec {
    name: "cast.to.i32",
    spirv: include_bytes!("../shaders/cast_to_i32.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "input"),
        BindingSpec::storage(1, "out"),
    ],
    push_constant_bytes: 8,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32, DType::I32, DType::I64, DType::Bool],
    shape: "one contiguous operand of any alpha dtype, cast to a contiguous i32 output of the \
            same shape",
    grid: grid::elementwise,
};

/// `Cast` into `i64`.
pub const CAST_TO_I64: KernelSpec = KernelSpec {
    name: "cast.to.i64",
    spirv: include_bytes!("../shaders/cast_to_i64.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "input"),
        BindingSpec::storage(1, "out"),
    ],
    push_constant_bytes: 8,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32, DType::I32, DType::I64, DType::Bool],
    shape: "one contiguous operand of any alpha dtype, cast to a contiguous i64 output of the \
            same shape",
    grid: grid::elementwise,
};

/// `Cast` into `Bool`.
pub const CAST_TO_BOOL: KernelSpec = KernelSpec {
    name: "cast.to.bool",
    spirv: include_bytes!("../shaders/cast_to_bool.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "input"),
        BindingSpec::storage(1, "out"),
    ],
    push_constant_bytes: 8,
    workgroup_size: [WORKGROUP, 1, 1],
    dtypes: &[DType::F32, DType::I32, DType::I64, DType::Bool],
    shape: "one contiguous operand of any alpha dtype, cast to a contiguous Bool output (four \
            packed verdicts per invocation)",
    grid: grid::elementwise,
};

// ---------------------------------------------------------------------------
// Push-constant builder
// ---------------------------------------------------------------------------

/// Little-endian push-constant bytes, built field by field.
///
/// Every shader in this family uses only 4-byte scalar members, so the byte
/// order is exactly the field order; the SPIR-V reflection check at
/// registration proves the declared size matches the shader block.
pub(super) struct Push(Vec<u8>);

impl Push {
    pub(super) fn new() -> Push {
        Push(Vec::new())
    }

    pub(super) fn u32(mut self, value: u32) -> Push {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub(super) fn i32(mut self, value: i32) -> Push {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub(super) fn f32(mut self, value: f32) -> Push {
        self.0.extend_from_slice(&value.to_bits().to_le_bytes());
        self
    }

    pub(super) fn dims(mut self, dims: [u32; MAX_RANK]) -> Push {
        for extent in dims {
            self.0.extend_from_slice(&extent.to_le_bytes());
        }
        self
    }

    pub(super) fn bytes(self) -> Vec<u8> {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Shared launch plumbing
// ---------------------------------------------------------------------------

/// One dispatch inside a [`Job`]: push constants, geometry and bindings.
struct Step {
    push: Vec<u8>,
    workgroups: [u32; 3],
    bindings: Vec<(u32, usize, BindingRole)>,
}

/// A single-submission execution of one op.
///
/// Buffers are allocated from the backend's suballocator, inputs travel through
/// host-visible staging, and the whole op is one fence-waited submission with
/// one [`ExecutionRecord`]. Dropping the `Job` frees every device buffer after
/// the wait, so an op never leaks device memory (`memory_stats().live_bytes`
/// returns to its previous value).
pub(super) struct Job<'a> {
    backend: &'a VulkanBackend,
    op: OpId,
    spec: &'static KernelSpec,
    kernel: Arc<Kernel>,
    buffers: Vec<Buffer>,
    uploads: Vec<(usize, Vec<u8>)>,
    readbacks: Vec<(usize, usize)>,
    steps: Vec<Step>,
}

impl<'a> Job<'a> {
    /// Resolve the compiled kernel for `spec`.
    pub(super) fn new(
        backend: &'a VulkanBackend,
        op: OpId,
        spec: &'static KernelSpec,
    ) -> Fn3Result<Job<'a>> {
        let kernel = backend.kernel(spec)?;
        Ok(Job {
            backend,
            op,
            spec,
            kernel,
            buffers: Vec::new(),
            uploads: Vec::new(),
            readbacks: Vec::new(),
            steps: Vec::new(),
        })
    }

    /// Allocate one device input buffer and stage `bytes` for upload.
    pub(super) fn input(&mut self, bytes: &[u8], name: &'static str) -> Fn3Result<usize> {
        let size = padded_bytes(bytes.len() as u64);
        let buffer = self
            .backend
            .alloc_buffer(&BufferDesc::device_storage(size, name))?;
        let mut padded = Vec::with_capacity(size as usize);
        padded.extend_from_slice(bytes);
        padded.resize(size as usize, 0);
        let index = self.buffers.len();
        self.buffers.push(buffer);
        self.uploads.push((index, padded));
        Ok(index)
    }

    /// Allocate one device output buffer of `bytes` bytes (rounded to 4).
    pub(super) fn output(&mut self, bytes: u64, name: &'static str) -> Fn3Result<usize> {
        let size = padded_bytes(bytes);
        let buffer = self
            .backend
            .alloc_buffer(&BufferDesc::device_storage(size, name))?;
        let index = self.buffers.len();
        self.buffers.push(buffer);
        self.readbacks.push((index, bytes as usize));
        Ok(index)
    }

    /// Record one dispatch. Bindings are `(binding, buffer index, role)`.
    pub(super) fn step(
        &mut self,
        push: Vec<u8>,
        workgroups: [u32; 3],
        bindings: Vec<(u32, usize, BindingRole)>,
    ) {
        self.steps.push(Step {
            push,
            workgroups,
            bindings,
        });
    }

    /// Run every recorded step as one submission and copy the outputs back.
    ///
    /// `outputs` is indexed in the order [`Job::output`] was called.
    pub(super) fn run(&self, outputs: &mut [&mut [u8]]) -> Fn3Result<()> {
        if self.steps.is_empty() {
            return Ok(());
        }
        if outputs.len() != self.readbacks.len() {
            return Err(Fn3Error::InvalidShape {
                context: "vulkan kernel output count",
                detail: format!(
                    "op {} has {} readback slot(s) but {} host destination(s)",
                    self.op,
                    self.readbacks.len(),
                    outputs.len()
                ),
            });
        }
        let staging = self.backend.staging();
        // A zero-element operand owns a logical zero-byte buffer (see
        // `context::Buffer`); there is nothing to transfer for it, and
        // `vkCmdCopyBuffer` cannot move zero bytes. Skipping the lease keeps
        // the empty case out of the transfer path entirely.
        let mut upload_leases: Vec<Option<StagingLease<'_>>> =
            Vec::with_capacity(self.uploads.len());
        for (_, bytes) in &self.uploads {
            if bytes.is_empty() {
                upload_leases.push(None);
                continue;
            }
            let lease = staging
                .acquire(bytes.len() as u64)
                .map_err(|e| e.to_fn3())?;
            lease.write(bytes).map_err(|e| e.to_fn3())?;
            upload_leases.push(Some(lease));
        }
        let mut readback_leases: Vec<Option<StagingLease<'_>>> =
            Vec::with_capacity(self.readbacks.len());
        for (index, _) in &self.readbacks {
            let bytes = self.buffers[*index].size();
            if bytes == 0 {
                readback_leases.push(None);
                continue;
            }
            readback_leases.push(Some(staging.acquire(bytes).map_err(|e| e.to_fn3())?));
        }

        let mut submitter = self.backend.checkout_submitter()?;
        let mut first: Option<DispatchInfo> = None;
        let report = {
            submitter
                .record(|recorder| {
                    for ((index, _), lease) in self.uploads.iter().zip(upload_leases.iter()) {
                        let Some(lease) = lease else {
                            continue;
                        };
                        let buffer = &self.buffers[*index];
                        recorder.copy_buffer(lease.buffer(), 0, buffer, 0, buffer.size())?;
                    }
                    for ((index, _), lease) in self.uploads.iter().zip(upload_leases.iter()) {
                        if lease.is_none() {
                            continue;
                        }
                        recorder.barrier(
                            &self.buffers[*index],
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::AccessFlags::TRANSFER_WRITE,
                            vk::AccessFlags::SHADER_READ,
                        );
                    }
                    for (position, step) in self.steps.iter().enumerate() {
                        let arguments: Vec<BindingArg<'_>> = step
                            .bindings
                            .iter()
                            .map(|(binding, index, role)| {
                                let buffer = &self.buffers[*index];
                                BindingArg::range(*binding, *role, buffer, 0, buffer.size())
                            })
                            .collect();
                        let info = recorder.dispatch(&KernelInvocation {
                            kernel: &self.kernel,
                            bindings: &arguments,
                            push_constants: &step.push,
                            workgroups: step.workgroups,
                        })?;
                        if first.is_none() {
                            first = Some(info);
                        }
                        if position + 1 < self.steps.len() {
                            // Order this dispatch's writes before the next
                            // dispatch's writes to the same output buffer: the
                            // regions are disjoint, but an explicit
                            // compute-to-compute barrier keeps the command
                            // stream free of an implicit write-after-write
                            // dependency the validation layer would report.
                            for (binding, index, role) in &step.bindings {
                                if *role == BindingRole::Output {
                                    let _ = binding;
                                    recorder.barrier(
                                        &self.buffers[*index],
                                        vk::PipelineStageFlags::COMPUTE_SHADER,
                                        vk::PipelineStageFlags::COMPUTE_SHADER,
                                        vk::AccessFlags::SHADER_WRITE,
                                        vk::AccessFlags::SHADER_WRITE,
                                    );
                                }
                            }
                        }
                    }
                    for ((index, _), lease) in self.readbacks.iter().zip(readback_leases.iter()) {
                        let Some(lease) = lease else {
                            continue;
                        };
                        let buffer = &self.buffers[*index];
                        recorder.barrier(
                            buffer,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::AccessFlags::SHADER_WRITE,
                            vk::AccessFlags::TRANSFER_READ,
                        );
                        recorder.copy_buffer(buffer, 0, lease.buffer(), 0, buffer.size())?;
                    }
                    Ok(())
                })
                .map_err(|error| error.to_fn3())?
                .wait()
                .map_err(|error| error.to_fn3())?
        };
        drop(submitter);

        for (((index, host_len), lease), destination) in self
            .readbacks
            .iter()
            .zip(readback_leases.iter())
            .zip(outputs.iter_mut())
        {
            if destination.len() != *host_len {
                return Err(Fn3Error::CapacityExceeded {
                    what: "vulkan kernel readback destination",
                    requested: *host_len as u64,
                    capacity: destination.len() as u64,
                });
            }
            // A zero-element output has no bytes to read and no lease.
            let Some(lease) = lease else {
                debug_assert_eq!(*host_len, 0);
                continue;
            };
            let device_len = self.buffers[*index].size() as usize;
            let mut bytes = vec![0u8; device_len];
            lease.read_into(&mut bytes).map_err(|e| e.to_fn3())?;
            destination.copy_from_slice(&bytes[..*host_len]);
        }

        let dispatch = first.ok_or_else(|| Fn3Error::InvalidShape {
            context: "vulkan kernel submission",
            detail: format!("op {} recorded no dispatch information", self.op),
        })?;
        let mut record = ExecutionRecord::new(0, self.op, self.spec.name, self.backend.device_id());
        record.submissions = report.submissions;
        record.workgroups = dispatch.workgroups;
        record.bindings = dispatch.bindings;
        record.upload_bytes = self
            .uploads
            .iter()
            .map(|(_, bytes)| bytes.len() as u64)
            .sum();
        record.readback_bytes = self
            .readbacks
            .iter()
            .map(|(index, _)| self.buffers[*index].size())
            .sum();
        record.inputs_read_back = false;
        record.host_staged = true;
        record.device_local_bytes = self.buffers.iter().map(Buffer::size).sum();
        record.host_ns = report.host_ns;
        record.gpu_ns = report.gpu_ns;
        record.gpu_ticks = report.timestamps;
        self.backend.residency().record(record);
        Ok(())
    }
}

/// Derive a dispatch grid, translating the platform error into `Fn3Error`.
pub(super) fn workgroups(spec: &KernelSpec, elements: u64, dims: &[u64]) -> Fn3Result<[u32; 3]> {
    spec.workgroups(elements, dims)
        .map_err(|error| error.to_fn3())
}

/// Round a byte count up to the 4-byte multiple `vkCmdCopyBuffer` requires.
pub(super) fn padded_bytes(bytes: u64) -> u64 {
    if bytes == 0 {
        0
    } else {
        bytes.div_ceil(4) * 4
    }
}

/// The contiguous window of a view, as raw bytes.
pub(super) fn view_window_bytes<'a>(view: &TensorView<'a>) -> Fn3Result<&'a [u8]> {
    let start = view.meta().byte_offset()?;
    let len = view.meta().byte_len()?;
    let bytes = view.storage_bytes();
    let end = start.checked_add(len).ok_or(Fn3Error::ShapeOverflow {
        context: "vulkan kernel input window",
        axis: None,
        detail: format!("byte offset {start} plus length {len} exceeds usize::MAX"),
    })?;
    bytes.get(start..end).ok_or(Fn3Error::OutOfBounds {
        what: "vulkan kernel input window",
        index: end as u64,
        len: bytes.len() as u64,
    })
}

/// The contiguous window of an output view, as mutable raw bytes.
pub(super) fn output_window_bytes<'a>(out: &'a mut TensorMut<'_>) -> Fn3Result<&'a mut [u8]> {
    let start = out.meta().byte_offset()?;
    let len = out.meta().byte_len()?;
    let end = start.checked_add(len).ok_or(Fn3Error::ShapeOverflow {
        context: "vulkan kernel output window",
        axis: None,
        detail: format!("byte offset {start} plus length {len} exceeds usize::MAX"),
    })?;
    let bytes = out.storage_slice_mut::<u8>()?;
    if end > bytes.len() {
        return Err(Fn3Error::OutOfBounds {
            what: "vulkan kernel output window",
            index: end as u64,
            len: bytes.len() as u64,
        });
    }
    Ok(&mut bytes[start..end])
}

/// Reject any attribute: the op takes none.
pub(super) fn require_no_attributes(op: OpId, params: &OpParams) -> Fn3Result<()> {
    if params.axes.is_empty() && !params.keep_dims && params.epsilon == 0.0 {
        return Ok(());
    }
    Err(Fn3Error::InvalidShape {
        context: "vulkan kernel attributes",
        detail: format!(
            "op {op} accepts no attributes, but the plan supplies axes={:?}, keep_dims={}, \
             epsilon={}",
            params.axes, params.keep_dims, params.epsilon
        ),
    })
}

/// Reject every attribute except `axes`.
pub(super) fn require_only_axes(op: OpId, params: &OpParams) -> Fn3Result<()> {
    if !params.keep_dims && params.epsilon == 0.0 {
        return Ok(());
    }
    Err(Fn3Error::InvalidShape {
        context: "vulkan kernel attributes",
        detail: format!(
            "op {op} accepts axes only, but the plan supplies keep_dims={}, epsilon={}",
            params.keep_dims, params.epsilon
        ),
    })
}

/// Resolve a negative axis against `rank`.
pub(super) fn normalize_axis(
    op: OpId,
    axis: i64,
    rank: usize,
    what: &'static str,
) -> Fn3Result<usize> {
    let resolved = if axis < 0 { axis + rank as i64 } else { axis };
    usize::try_from(resolved)
        .ok()
        .filter(|&value| value < rank)
        .ok_or_else(|| Fn3Error::InvalidShape {
            context: what,
            detail: format!("op {op}: axis {axis} is outside rank {rank}"),
        })
}

/// The single axis named by `params.axes`, or `default`.
pub(super) fn single_axis(
    op: OpId,
    params: &OpParams,
    rank: usize,
    default: i64,
    what: &'static str,
) -> Fn3Result<usize> {
    let axis = match params.axes.as_slice() {
        [] => default,
        [axis] => *axis,
        axes => {
            return Err(Fn3Error::InvalidShape {
                context: what,
                detail: format!("op {op} takes one axis, but the plan supplies {axes:?}"),
            })
        }
    };
    normalize_axis(op, axis, rank, what)
}

/// Reject a rank the shaders cannot decompose, naming FN3-13.
pub(super) fn require_shader_rank(op: OpId, rank: usize) -> Fn3Result<()> {
    if rank <= MAX_RANK {
        return Ok(());
    }
    Err(Fn3Error::UnsupportedOp {
        op,
        backend: BACKEND,
        reason: format!(
            "the FN3-13 (#59) Vulkan kernels decompose at most {MAX_RANK} axes, and this request \
             has rank {rank}"
        ),
    })
}

/// Reject a strided view, naming FN3-13.
pub(super) fn require_contiguous_view(
    op: OpId,
    view: &TensorView<'_>,
    what: &'static str,
) -> Fn3Result<()> {
    if view.is_contiguous() {
        return Ok(());
    }
    Err(Fn3Error::UnsupportedOp {
        op,
        backend: BACKEND,
        reason: format!(
            "the FN3-13 (#59) Vulkan kernels execute contiguous row-major windows only and \
             `supports_strided_views` is false for this backend; the {what} is {} with strides {}. \
             Materialise it with `TensorView::to_contiguous` before executing",
            view.meta(),
            view.strides()
        ),
    })
}

/// Reject a strided output, naming FN3-13.
pub(super) fn require_contiguous_output(
    op: OpId,
    out: &TensorMut<'_>,
    what: &'static str,
) -> Fn3Result<()> {
    if out.is_contiguous() {
        return Ok(());
    }
    Err(Fn3Error::UnsupportedOp {
        op,
        backend: BACKEND,
        reason: format!(
            "the FN3-13 (#59) Vulkan kernels write contiguous row-major windows only; the {what} \
             is {} with strides {}",
            out.meta(),
            out.strides()
        ),
    })
}

/// Require one of `allowed` dtypes, naming the op.
pub(super) fn require_dtype_in(
    op: OpId,
    view: &TensorView<'_>,
    allowed: &[DType],
    what: &'static str,
) -> Fn3Result<DType> {
    let dtype = view.dtype();
    if allowed.contains(&dtype) {
        return Ok(dtype);
    }
    Err(Fn3Error::UnsupportedDType {
        op: Some(op),
        dtype,
        context: what,
    })
}

/// Require the output dtype, naming the op.
pub(super) fn require_output_dtype(
    op: OpId,
    out: &TensorMut<'_>,
    expected: DType,
) -> Fn3Result<()> {
    out.meta().require_dtype(Some(op), expected)
}

/// Require the output shape to be exactly `expected`.
pub(super) fn require_output_shape(
    op: OpId,
    out: &TensorMut<'_>,
    expected: &Shape,
) -> Fn3Result<()> {
    if out.shape() == expected {
        return Ok(());
    }
    Err(Fn3Error::ShapeMismatch {
        op: Some(op),
        expected: expected.to_string(),
        actual: out.shape().to_string(),
    })
}

/// Check every operand and the output are on the backend's device.
pub(super) fn require_device(
    op: OpId,
    device: crate::fn3::device::DeviceId,
    inputs: &[TensorView<'_>],
    outputs: &[TensorMut<'_>],
) -> Fn3Result<()> {
    for view in inputs {
        view.meta().require_device(Some(op), device)?;
    }
    for out in outputs {
        out.meta().require_device(Some(op), device)?;
    }
    Ok(())
}

/// The extents of `shape`, left-aligned, with `0` in unused slots.
pub(super) fn output_dims(op: OpId, shape: &Shape) -> Fn3Result<[u32; MAX_RANK]> {
    let mut dims = [0u32; MAX_RANK];
    for (axis, &extent) in shape.dims().iter().enumerate() {
        dims[axis] = u32::try_from(extent).map_err(|_| Fn3Error::UnsupportedOp {
            op,
            backend: BACKEND,
            reason: format!(
                "axis {axis} of {shape} has extent {extent}, which does not fit the 32-bit \
                 per-axis push constant the FN3-13 (#59) shaders use"
            ),
        })?;
    }
    Ok(dims)
}

/// The extents of `shape` right-aligned inside an `out_rank` window, padded
/// with `1` so a broadcast axis contributes a zero coordinate.
pub(super) fn broadcast_dims(
    op: OpId,
    shape: &Shape,
    out_rank: usize,
) -> Fn3Result<[u32; MAX_RANK]> {
    let mut dims = [1u32; MAX_RANK];
    let base = out_rank.saturating_sub(shape.rank());
    for (axis, &extent) in shape.dims().iter().enumerate() {
        dims[base + axis] = u32::try_from(extent).map_err(|_| Fn3Error::UnsupportedOp {
            op,
            backend: BACKEND,
            reason: format!(
                "axis {axis} of {shape} has extent {extent}, which does not fit the 32-bit \
                 per-axis push constant the FN3-13 (#59) shaders use"
            ),
        })?;
    }
    Ok(dims)
}

/// An element count that fits a 32-bit element index.
pub(super) fn element_count(op: OpId, elements: u64) -> Fn3Result<u32> {
    u32::try_from(elements).map_err(|_| Fn3Error::UnsupportedOp {
        op,
        backend: BACKEND,
        reason: format!(
            "{elements} elements do not fit the FN3-13 (#59) kernels' 32-bit element index; \
             chunked dispatch is not implemented"
        ),
    })
}

/// Dispatch one kernel over `inputs` (in binding order) into one output.
pub(super) fn dispatch_unary_shape(
    backend: &VulkanBackend,
    op: OpId,
    spec: &'static KernelSpec,
    push: Vec<u8>,
    workgroups: [u32; 3],
    inputs: &[&[u8]],
    output: &mut [u8],
    names: &[&'static str],
) -> Fn3Result<()> {
    let mut job = Job::new(backend, op, spec)?;
    let mut bindings = Vec::with_capacity(inputs.len() + 1);
    for (position, bytes) in inputs.iter().enumerate() {
        let name = names
            .get(position)
            .copied()
            .unwrap_or("fn3.pointwise.input");
        let index = job.input(bytes, name)?;
        bindings.push((position as u32, index, BindingRole::Input));
    }
    let out_index = job.output(output.len() as u64, "fn3.pointwise.out")?;
    bindings.push((inputs.len() as u32, out_index, BindingRole::Output));
    job.step(push, workgroups, bindings);
    job.run(&mut [output])
}

// ---------------------------------------------------------------------------
// Binary arithmetic
// ---------------------------------------------------------------------------

fn launch_binary(
    backend: &VulkanBackend,
    op: OpId,
    opcode: u32,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
    require_no_attributes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    let out = &mut outputs[0];
    for (name, view) in [("lhs", lhs), ("rhs", rhs)] {
        require_dtype_in(op, view, &[DType::F32], "vulkan pointwise binary operand")?;
        require_contiguous_view(op, view, name)?;
    }
    require_output_dtype(op, out, DType::F32)?;
    require_contiguous_output(op, out, "output")?;
    let shape = lhs.shape().broadcast_with(rhs.shape(), op)?;
    require_output_shape(op, out, &shape)?;
    let rank = shape.rank();
    require_shader_rank(op, rank)?;
    require_shader_rank(op, lhs.shape().rank())?;
    require_shader_rank(op, rhs.shape().rank())?;
    let elements = shape.numel()?;
    if elements == 0 {
        return Ok(());
    }
    let count = element_count(op, elements)?;
    let push = Push::new()
        .u32(count)
        .u32(opcode)
        .u32(rank as u32)
        .dims(output_dims(op, &shape)?)
        .dims(broadcast_dims(op, lhs.shape(), rank)?)
        .dims(broadcast_dims(op, rhs.shape(), rank)?)
        .bytes();
    let workgroups = workgroups(&BINARY, elements, shape.dims())?;
    let mut bytes = vec![0u8; out.meta().byte_len()?];
    dispatch_unary_shape(
        backend,
        op,
        &BINARY,
        push,
        workgroups,
        &[view_window_bytes(lhs)?, view_window_bytes(rhs)?],
        &mut bytes,
        &["fn3.pointwise.lhs", "fn3.pointwise.rhs"],
    )?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}

/// `Add` with right-aligned broadcasting.
pub fn launch_add(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_binary(backend, OpId::Add, 0, inputs, outputs, params)
}

/// `Sub` with right-aligned broadcasting.
pub fn launch_sub(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_binary(backend, OpId::Sub, 1, inputs, outputs, params)
}

/// `Mul` with right-aligned broadcasting.
pub fn launch_mul(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_binary(backend, OpId::Mul, 2, inputs, outputs, params)
}

/// `Div` with right-aligned broadcasting.
pub fn launch_div(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_binary(backend, OpId::Div, 3, inputs, outputs, params)
}

/// `Pow` with right-aligned broadcasting and C `pow` negative-base semantics.
pub fn launch_pow(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_binary(backend, OpId::Pow, 4, inputs, outputs, params)
}

// ---------------------------------------------------------------------------
// Unary functions
// ---------------------------------------------------------------------------

fn launch_unary(
    backend: &VulkanBackend,
    op: OpId,
    opcode: u32,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
    require_no_attributes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let input = &inputs[0];
    let out = &mut outputs[0];
    require_dtype_in(op, input, &[DType::F32], "vulkan pointwise unary operand")?;
    require_contiguous_view(op, input, "input")?;
    require_output_dtype(op, out, DType::F32)?;
    require_contiguous_output(op, out, "output")?;
    require_output_shape(op, out, input.shape())?;
    require_shader_rank(op, input.shape().rank())?;
    let elements = input.shape().numel()?;
    if elements == 0 {
        return Ok(());
    }
    let count = element_count(op, elements)?;
    let push = Push::new().u32(count).u32(opcode).bytes();
    let workgroups = workgroups(&UNARY, elements, input.shape().dims())?;
    let mut bytes = vec![0u8; out.meta().byte_len()?];
    dispatch_unary_shape(
        backend,
        op,
        &UNARY,
        push,
        workgroups,
        &[view_window_bytes(input)?],
        &mut bytes,
        &["fn3.pointwise.in"],
    )?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}

/// `Neg`.
pub fn launch_neg(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_unary(backend, OpId::Neg, 0, inputs, outputs, params)
}

/// `Sqrt`.
pub fn launch_sqrt(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_unary(backend, OpId::Sqrt, 1, inputs, outputs, params)
}

/// `Tanh`.
pub fn launch_tanh(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_unary(backend, OpId::Tanh, 2, inputs, outputs, params)
}

/// `Relu`.
pub fn launch_relu(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_unary(backend, OpId::Relu, 3, inputs, outputs, params)
}

/// `Sin`.
pub fn launch_sin(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_unary(backend, OpId::Sin, 4, inputs, outputs, params)
}

/// `Cos`.
pub fn launch_cos(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_unary(backend, OpId::Cos, 5, inputs, outputs, params)
}

/// `Erf`.
pub fn launch_erf(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_unary(backend, OpId::Erf, 6, inputs, outputs, params)
}

/// `Gelu` in its exact (erf) form.
pub fn launch_gelu(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_unary(backend, OpId::Gelu, 7, inputs, outputs, params)
}

// ---------------------------------------------------------------------------
// Where
// ---------------------------------------------------------------------------

/// `Where(cond, x, y)` with three-way broadcasting.
pub fn launch_where(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Where;
    check_arity(op, inputs.len(), outputs.len(), 3, 1)?;
    require_no_attributes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let cond = &inputs[0];
    let x = &inputs[1];
    let y = &inputs[2];
    let out = &mut outputs[0];
    let allowed: &[DType] = &[DType::F32, DType::I32, DType::I64];
    require_dtype_in(op, x, allowed, "vulkan where value operand")?;
    require_dtype_in(op, y, allowed, "vulkan where value operand")?;
    if cond.dtype() != DType::Bool {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: DType::Bool,
            actual: cond.dtype(),
        });
    }
    if x.dtype() != y.dtype() {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: x.dtype(),
            actual: y.dtype(),
        });
    }
    require_output_dtype(op, out, x.dtype())?;
    for (view, name) in [(cond, "condition"), (x, "x"), (y, "y")] {
        require_contiguous_view(op, view, name)?;
        require_shader_rank(op, view.shape().rank())?;
    }
    require_contiguous_output(op, out, "output")?;
    let shape = cond
        .shape()
        .broadcast_with(x.shape(), op)?
        .broadcast_with(y.shape(), op)?;
    require_output_shape(op, out, &shape)?;
    let rank = shape.rank();
    require_shader_rank(op, rank)?;
    let elements = shape.numel()?;
    if elements == 0 {
        return Ok(());
    }
    let count = element_count(op, elements)?;
    let elem_bytes = u32::try_from(x.dtype().size_bytes()).map_err(|_| Fn3Error::InvalidShape {
        context: "vulkan where element width",
        detail: "dtype size does not fit u32".to_string(),
    })?;
    let push = Push::new()
        .u32(count)
        .u32(rank as u32)
        .u32(elem_bytes)
        .dims(output_dims(op, &shape)?)
        .dims(broadcast_dims(op, x.shape(), rank)?)
        .dims(broadcast_dims(op, y.shape(), rank)?)
        .dims(broadcast_dims(op, cond.shape(), rank)?)
        .bytes();
    let workgroups = workgroups(&WHERE, elements, shape.dims())?;
    let mut bytes = vec![0u8; out.meta().byte_len()?];
    dispatch_unary_shape(
        backend,
        op,
        &WHERE,
        push,
        workgroups,
        &[
            view_window_bytes(cond)?,
            view_window_bytes(x)?,
            view_window_bytes(y)?,
        ],
        &mut bytes,
        &["fn3.where.cond", "fn3.where.x", "fn3.where.y"],
    )?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

fn launch_compare(
    backend: &VulkanBackend,
    op: OpId,
    comparison: u32,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
    require_no_attributes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    let out = &mut outputs[0];
    let allowed: &[DType] = if comparison == 0 {
        &[DType::F32, DType::I32, DType::I64, DType::Bool]
    } else {
        &[DType::F32, DType::I32, DType::I64]
    };
    let dtype = require_dtype_in(op, lhs, allowed, "vulkan compare operand")?;
    if rhs.dtype() != dtype {
        return Err(Fn3Error::DTypeMismatch {
            op: Some(op),
            expected: dtype,
            actual: rhs.dtype(),
        });
    }
    require_output_dtype(op, out, DType::Bool)?;
    for (view, name) in [(lhs, "lhs"), (rhs, "rhs")] {
        require_contiguous_view(op, view, name)?;
        require_shader_rank(op, view.shape().rank())?;
    }
    require_contiguous_output(op, out, "output")?;
    let shape = lhs.shape().broadcast_with(rhs.shape(), op)?;
    require_output_shape(op, out, &shape)?;
    let rank = shape.rank();
    require_shader_rank(op, rank)?;
    let elements = shape.numel()?;
    if elements == 0 {
        return Ok(());
    }
    let count = element_count(op, elements)?;
    let kind = match dtype {
        DType::F32 => 0u32,
        DType::I32 => 1,
        DType::I64 => 2,
        _ => 3,
    };
    let push = Push::new()
        .u32(count)
        .u32(rank as u32)
        .u32(comparison)
        .u32(kind)
        .dims(output_dims(op, &shape)?)
        .dims(broadcast_dims(op, lhs.shape(), rank)?)
        .dims(broadcast_dims(op, rhs.shape(), rank)?)
        .bytes();
    // Four comparisons per invocation pack one Bool word; the grid must cover
    // `ceil(n / 4)` threads, not `n`.
    let threads = elements.div_ceil(4);
    let workgroups = workgroups(&COMPARE, threads, shape.dims())?;
    let mut bytes = vec![0u8; out.meta().byte_len()?];
    dispatch_unary_shape(
        backend,
        op,
        &COMPARE,
        push,
        workgroups,
        &[view_window_bytes(lhs)?, view_window_bytes(rhs)?],
        &mut bytes,
        &["fn3.compare.lhs", "fn3.compare.rhs"],
    )?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}

/// `Equal`, producing `Bool`.
pub fn launch_equal(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_compare(backend, OpId::Equal, 0, inputs, outputs, params)
}

/// `Less`, producing `Bool`.
pub fn launch_less(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_compare(backend, OpId::Less, 1, inputs, outputs, params)
}

/// `Greater`, producing `Bool`.
pub fn launch_greater(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    launch_compare(backend, OpId::Greater, 2, inputs, outputs, params)
}

// ---------------------------------------------------------------------------
// Cast
// ---------------------------------------------------------------------------

/// `Cast`: the target dtype is the output tensor's dtype, so no attribute is
/// needed.
pub fn launch_cast(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::Cast;
    check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
    require_no_attributes(op, params)?;
    require_device(op, backend.device_id(), inputs, outputs)?;
    let input = &inputs[0];
    let out = &mut outputs[0];
    let allowed: &[DType] = &[DType::F32, DType::I32, DType::I64, DType::Bool];
    let source = require_dtype_in(op, input, allowed, "vulkan cast source")?;
    require_contiguous_view(op, input, "input")?;
    require_output_shape(op, out, input.shape())?;
    require_contiguous_output(op, out, "output")?;
    require_shader_rank(op, input.shape().rank())?;
    let target = out.dtype();
    if !allowed.contains(&target) {
        return Err(Fn3Error::UnsupportedDType {
            op: Some(op),
            dtype: target,
            context: "vulkan cast destination",
        });
    }
    let elements = input.shape().numel()?;
    if elements == 0 {
        return Ok(());
    }
    let count = element_count(op, elements)?;
    let src_kind = match source {
        DType::F32 => 0u32,
        DType::I64 => 1,
        DType::I32 => 2,
        _ => 3,
    };
    let (spec, threads) = match target {
        DType::F32 => (&CAST_TO_F32, elements),
        DType::I32 => (&CAST_TO_I32, elements),
        DType::I64 => (&CAST_TO_I64, elements),
        _ => (&CAST_TO_BOOL, elements.div_ceil(4)),
    };
    let push = Push::new().u32(count).u32(src_kind).bytes();
    let workgroups = workgroups(spec, threads, input.shape().dims())?;
    let mut bytes = vec![0u8; out.meta().byte_len()?];
    dispatch_unary_shape(
        backend,
        op,
        spec,
        push,
        workgroups,
        &[view_window_bytes(input)?],
        &mut bytes,
        &["fn3.cast.in"],
    )?;
    output_window_bytes(out)?.copy_from_slice(&bytes);
    Ok(())
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register this module's kernels.
///
/// Called from [`super::KernelSet::alpha`]. Replacement is by op id, so the
/// registrations below supersede the FN3-11 smoke kernel for `Add`.
pub fn register(set: &mut KernelSet) {
    for (op, launch) in [
        (OpId::Add, launch_add as super::LaunchFn),
        (OpId::Sub, launch_sub),
        (OpId::Mul, launch_mul),
        (OpId::Div, launch_div),
        (OpId::Pow, launch_pow),
    ] {
        set.register(OpKernel {
            op,
            support: Support::Full,
            features: CapabilityFeatures {
                broadcasting: true,
                ..CapabilityFeatures::default()
            },
            dtypes: &[DType::F32],
            admits: None,
            kernel: &BINARY,
            launch,
        });
    }
    for (op, launch) in [
        (OpId::Neg, launch_neg as super::LaunchFn),
        (OpId::Sqrt, launch_sqrt),
        (OpId::Tanh, launch_tanh),
        (OpId::Relu, launch_relu),
        (OpId::Sin, launch_sin),
        (OpId::Cos, launch_cos),
        (OpId::Erf, launch_erf),
        (OpId::Gelu, launch_gelu),
    ] {
        set.register(OpKernel {
            op,
            support: Support::Full,
            features: CapabilityFeatures::default(),
            dtypes: &[DType::F32],
            admits: None,
            kernel: &UNARY,
            launch,
        });
    }
    set.register(OpKernel {
        op: OpId::Where,
        // `Where` selects whole elements and never converts one, so the kernel
        // is dtype-agnostic for 4- and 8-byte values. A `Bool` x or y would be
        // a 1-byte element, which core Vulkan cannot write atomically here.
        support: Support::partial(
            "F32/I32/I64 values selected element-wise with a Bool condition and three-way \
             broadcasting; a Bool value operand is rejected with UnsupportedOp naming #59",
        ),
        features: CapabilityFeatures::default(),
        dtypes: &[DType::F32, DType::I32, DType::I64],
        admits: Some(admits_where),
        kernel: &WHERE,
        launch: launch_where,
    });
    for (op, launch) in [
        (OpId::Equal, launch_equal as super::LaunchFn),
        (OpId::Less, launch_less),
        (OpId::Greater, launch_greater),
    ] {
        set.register(OpKernel {
            op,
            support: Support::Full,
            features: CapabilityFeatures::default(),
            dtypes: &[DType::F32, DType::I32, DType::I64],
            admits: None,
            kernel: &COMPARE,
            launch,
        });
    }
    set.register(OpKernel {
        op: OpId::Cast,
        support: Support::Full,
        features: CapabilityFeatures::default(),
        dtypes: &[DType::F32, DType::I32, DType::I64, DType::Bool],
        admits: None,
        // The four destination-typed shaders share one registration; `launch`
        // selects the compiled pipeline from the output dtype.
        kernel: &CAST_TO_F32,
        launch: launch_cast,
    });
}

/// Concrete admission for `Where`: a Bool condition and a 4- or 8-byte value.
///
/// The condition is necessarily `Bool` (the kernel rejects anything else) and
/// the value operands are the two the `Partial` reason names. The element type
/// of a value operand is the only plan-visible part of the subset.
fn admits_where(
    _shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
) -> std::result::Result<(), String> {
    let [condition, x, y, output, ..] = dtypes else {
        return Err(format!(
            "{} operands were supplied; where takes 4",
            dtypes.len()
        ));
    };
    if *condition != DType::Bool {
        return Err(format!(
            "{condition} condition; where needs a Bool condition"
        ));
    }
    if !matches!(x, DType::F32 | DType::I32 | DType::I64) {
        return Err(format!("{x} value operand is a 1-byte element"));
    }
    if y != x {
        return Err(format!("{y} value operand does not match {x}"));
    }
    if output != x {
        return Err(format!("{output} output does not match {x}"));
    }
    Ok(())
}
