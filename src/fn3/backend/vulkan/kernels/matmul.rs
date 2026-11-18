//! GPU matrix product: rank `>= 2`, batched, with left-aligned batch
//! broadcasting (FN3-14 / #60).
//!
//! # The contract is the CPU reference
//!
//! [`crate::fn3::backend::cpu::kernels::matmul`] defines the semantics this
//! kernel implements; it is not free to invent a different one:
//!
//! ```text
//! lhs: [B..., M, K]     rhs: [B'..., K, N]     out: [B'', ..., M, N]
//! ```
//!
//! - both operands have rank `>= 2` with the matrix axes last; rank-1 operands
//!   are rejected (the CPU reference does the same, because promoting a vector
//!   would change the output rank);
//! - the batch axes of the two operands broadcast right-aligned, so rank-3 by
//!   rank-2 (`[1, 13, 128] @ [128, 384]`, the prefill projection) and
//!   `[2, 1, 3, 4] @ [1, 5, 4, 6]` are ordinary cases;
//! - each output element is **one `f32` accumulator over ascending `k`**,
//!   starting at `+0.0`. A K tail is zero-padded in shared memory, and
//!   `x + 0.0 == x` exactly, so padding cannot perturb the sum. No tree
//!   reduction, no `f64` accumulator;
//! - operands are addressed through their metadata: element offset, per-axis
//!   strides, and zero strides where a batch axis broadcasts. A transposed
//!   right-hand side (`Gemm(transB=1)` is `MatMul(a, w^T)`), a strided window or
//!   an explicitly broadcast view therefore needs no host-side copy. A zero
//!   stride on the *contraction* axis is not a matrix product and is rejected,
//!   exactly as the CPU reference rejects it.
//!
//! # Why a tiled shared-memory kernel
//!
//! The decode projection `[1, 128] @ [128, 50257]` reads 25 MB of weights to
//! produce 50,257 outputs, so it is bandwidth-bound; the prefill projection
//! `[13, 128] @ [128, 384]` is launch-bound. A 16x16 output tile with a 16-deep
//! shared-memory stage gives each workgroup 16x reuse of every loaded operand
//! element without needing uniform/storage auxiliaries, and keeps the
//! accumulation order identical to the reference.
//!
//! One dispatch covers the whole op: `gl_WorkGroupID.z` selects the batch
//! slice, `x` the output-column tile and `y` the output-row tile. Bucketing the
//! batch into `z` means the residency record's `workgroups` triple describes
//! the entire op, not the last of several launches.
//!
//! # Not implemented, and rejected rather than approximated
//!
//! - rank < 2, other dtypes, a non-contiguous output, a read-only broadcast
//!   output, a zero-stride contraction, more than six batch axes, and any
//!   element span that does not fit the shader's 32-bit index: each returns
//!   [`Fn3Error::UnsupportedOp`] or the matching contract error, naming
//!   FN3-14 (#60).
//! - #71/#72 (simd/threaded CPU microkernels, kernel variants and fence ring)
//!   own faster variants; a wider-tile or vectorised shader must publish its
//!   own measured agreement rather than replace this one silently. This module
//!   deliberately ships one path only, so there is no second implementation to
//!   drift.

use super::super::context::{BindingArg, BindingRole, BufferDesc, KernelInvocation};
use super::super::error::{Error as VulkanError, Result as VulkanResult};
use super::super::residency::ExecutionRecord;
use super::super::VulkanBackend;
use super::{BindingSpec, GridInput, KernelSet, KernelSpec, OpKernel};
use crate::fn3::backend::vulkan::VULKAN_BACKEND_NAME;
use crate::fn3::backend::{check_arity, Backend, CapabilityFeatures, OpParams, Support};
use crate::fn3::dtype::DType;
use crate::fn3::error::{Fn3Error, Fn3Result};
use crate::fn3::ops::OpId;
use crate::fn3::shape::Shape;
use crate::fn3::tensor::{TensorMut, TensorView};

/// Side of one output tile: a workgroup computes a `TILE x TILE` square.
pub const TILE: u32 = 16;

/// Batch axes the push-constant block can describe. The alpha rank limit is 8,
/// so at most six axes remain after the two matrix axes.
pub const MAX_BATCH_AXES: usize = 6;

/// Words in the push-constant block; must match `MATMUL_PUSH_WORDS` in the
/// shader and the reflected SPIR-V block.
const PUSH_CONSTANT_WORDS: usize = 28;

/// Declared kernel: `out[b..., m, n] = sum_k lhs[b..., m, k] * rhs[b..., k, n]`.
pub const MATMUL_TILED: KernelSpec = KernelSpec {
    name: "matmul.tiled.f32",
    spirv: include_bytes!("../shaders/matmul_tiled.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "lhs"),
        BindingSpec::storage(1, "rhs"),
        BindingSpec::storage(2, "out"),
    ],
    push_constant_bytes: (PUSH_CONSTANT_WORDS * 4) as u32,
    workgroup_size: [TILE, TILE, 1],
    dtypes: &[DType::F32],
    shape: "f32 matmul, both operands rank >= 2 with the matrix axes last, right-aligned batch \
            broadcasting, strided or broadcast read-only operands, contiguous writable output, \
            K tailed and zero-padded in shared memory",
    grid: grid_matmul,
};

/// Register the tiled matmul for [`OpId::MatMul`].
///
/// Called from [`super::KernelSet::alpha`]. Replacement is by op id, so this
/// registration supersedes the FN3-11 smoke kernel if one ever claims `MatMul`.
pub fn register(set: &mut KernelSet) {
    set.register(OpKernel {
        op: OpId::MatMul,
        // The whole documented contract is implemented: rank >= 2, batch
        // broadcast, any K/M/N including tails, strided and broadcast operands.
        // The dtype table is F32 because that is the only alpha compute dtype
        // (DType::I64/I32/Bool are indices, masks and auxiliaries), and a
        // non-f32 request is rejected by `require_dtype` rather than cast.
        support: Support::Full,
        // The four capability flags describe elementwise broadcasting, the
        // fused attention op, quantization and weight residency. None of them
        // describes matmul's batch broadcasting (which is part of the op's own
        // contract, not a capability the core might otherwise decompose), so an
        // honest answer leaves every flag false. FN3-13 (#59) owns the
        // elementwise `broadcasting` flag.
        features: CapabilityFeatures::default(),
        dtypes: &[DType::F32],
        kernel: &MATMUL_TILED,
        launch: launch_matmul,
    });
}

/// Dispatch geometry: column tiles on `x`, row tiles on `y`, batch on `z`.
///
/// `output_dims` is the *output* shape `[B..., M, N]`, so the batch count is the
/// product of every leading extent. The launch function calls this through
/// [`KernelSpec::workgroups`], which keeps the recorded geometry and the
/// declaration in one place.
fn grid_matmul(input: &GridInput<'_>) -> VulkanResult<[u32; 3]> {
    let dims = input.output_dims;
    let rank = dims.len();
    if rank < 2 {
        return Err(VulkanError::internal(format!(
            "matmul grid: output rank {rank} has no matrix axes"
        )));
    }
    let columns = dims[rank - 1];
    let rows = dims[rank - 2];
    let mut batch = 1u64;
    for extent in &dims[..rank - 2] {
        batch = batch.saturating_mul(*extent);
    }
    let tile = u64::from(input.workgroup_size[0].max(1));
    let tile_rows = u64::from(input.workgroup_size[1].max(1));
    let axis = |value: u64, what: &str| {
        u32::try_from(value).map_err(|_| {
            VulkanError::internal(format!(
                "matmul grid: {value} {what} tiles exceed the 32-bit dispatch limit"
            ))
        })
    };
    Ok([
        axis(columns.div_ceil(tile).max(1), "column")?,
        axis(rows.div_ceil(tile_rows).max(1), "row")?,
        axis(batch.max(1), "batch")?,
    ])
}

/// The 112-byte push-constant block, in the order the shader declares it.
///
/// Every field is an element count or an element stride, never a byte count:
/// the shader indexes `float a[]`-style storage arrays. The three `*_base`
/// fields are the offset of the first addressed element inside the uploaded
/// window; this kernel uploads exactly the view's addressed window, so they are
/// always zero today, but the shader keeps them so a later device-resident path
/// can bind one allocation and offset into it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PushConstants {
    m: u32,
    n: u32,
    k: u32,
    a_base: u32,
    b_base: u32,
    out_base: u32,
    a_row_stride: u32,
    a_k_stride: u32,
    b_k_stride: u32,
    b_col_stride: u32,
    batch_dims: [u32; MAX_BATCH_AXES],
    a_batch_strides: [u32; MAX_BATCH_AXES],
    b_batch_strides: [u32; MAX_BATCH_AXES],
}

impl PushConstants {
    /// Pack the block in declaration order without relying on struct layout.
    fn words(&self) -> [u32; PUSH_CONSTANT_WORDS] {
        let mut words = [0u32; PUSH_CONSTANT_WORDS];
        words[0] = self.m;
        words[1] = self.n;
        words[2] = self.k;
        words[3] = self.a_base;
        words[4] = self.b_base;
        words[5] = self.out_base;
        words[6] = self.a_row_stride;
        words[7] = self.a_k_stride;
        words[8] = self.b_k_stride;
        words[9] = self.b_col_stride;
        words[10..16].copy_from_slice(&self.batch_dims);
        words[16..22].copy_from_slice(&self.a_batch_strides);
        words[22..28].copy_from_slice(&self.b_batch_strides);
        words
    }
}

/// One operand's addressing facts, extracted from a [`TensorView`].
#[derive(Debug, Clone, Copy)]
struct OperandLayout {
    /// Elements the shader may read inside the uploaded window.
    window_elements: u64,
    /// Element stride between consecutive rows (lhs) or contraction steps
    /// (rhs).
    row_stride: u32,
    /// Element stride between consecutive contraction steps (lhs) or columns
    /// (rhs).
    inner_stride: u32,
    /// Effective, right-aligned batch strides; zero where the axis broadcasts.
    batch_strides: [u32; MAX_BATCH_AXES],
}

/// Execute `MatMul` through the platform layer.
pub fn launch_matmul(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    check_arity(OpId::MatMul, inputs.len(), outputs.len(), 2, 1)?;
    require_no_attributes(params)?;

    let lhs = &inputs[0];
    let rhs = &inputs[1];
    let out = &mut outputs[0];
    for view in [lhs, rhs] {
        view.meta().require_dtype(Some(OpId::MatMul), DType::F32)?;
    }
    out.meta().require_dtype(Some(OpId::MatMul), DType::F32)?;

    // ---- shapes: identical rules and messages to the CPU reference ----------
    let lhs_rank = lhs.shape().rank();
    let rhs_rank = rhs.shape().rank();
    for (name, view, rank) in [("lhs", lhs, lhs_rank), ("rhs", rhs, rhs_rank)] {
        if rank < 2 {
            return Err(unsupported(format!(
                "{name} has rank {rank} ({}), but the fn3 matmul contract covers rank >= 2 \
                 with the matrix axes last. Rank-1 vector products are rejected rather than \
                 promoted to a different output rank, matching the CPU reference",
                view.shape()
            )));
        }
    }
    let lhs_dims = lhs.shape().dims();
    let rhs_dims = rhs.shape().dims();
    let rows = lhs_dims[lhs_rank - 2];
    let inner = lhs_dims[lhs_rank - 1];
    let rhs_inner = rhs_dims[rhs_rank - 2];
    let columns = rhs_dims[rhs_rank - 1];
    if inner != rhs_inner {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(OpId::MatMul),
            expected: format!("rhs contraction of {inner} (lhs {})", lhs.shape()),
            actual: format!("rhs {} contracts {rhs_inner}", rhs.shape()),
        });
    }

    let lhs_batch = Shape::from(&lhs_dims[..lhs_rank - 2]);
    let rhs_batch = Shape::from(&rhs_dims[..rhs_rank - 2]);
    let batch = lhs_batch.broadcast_with(&rhs_batch, OpId::MatMul)?;
    if batch.rank() > MAX_BATCH_AXES {
        return Err(unsupported(format!(
            "the broadcast batch has {} axes ({}), above the {MAX_BATCH_AXES} the kernel \
             decomposes; the alpha rank limit is 8 so this cannot come from a legal plan, \
             and the kernel refuses rather than dropping leading axes",
            batch.rank(),
            batch
        )));
    }

    let mut expected_dims = batch.dims().to_vec();
    expected_dims.push(rows);
    expected_dims.push(columns);
    let expected = Shape::from_dims(expected_dims);
    if out.shape() != &expected {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(OpId::MatMul),
            expected: expected.to_string(),
            actual: out.shape().to_string(),
        });
    }

    let out_numel = expected.numel()?;
    if out_numel == 0 {
        // Nothing to write: every extent that matters is zero. The empty
        // product case (K == 0 with a non-empty output) is the shader's, below.
        return Ok(());
    }

    // ---- the output must be a contiguous writable window --------------------
    if out.meta().is_broadcast() {
        return Err(unsupported(format!(
            "the output {} is an explicitly broadcast (read-only) view; a matrix product \
             needs a writable output",
            out.meta()
        )));
    }
    if !out.is_contiguous() {
        return Err(unsupported(format!(
            "the output {} is a strided view ({}); the kernel writes one contiguous row-major \
             window. Pass a contiguous output and copy into a strided destination on the plan \
             side",
            out.meta(),
            out.strides()
        )));
    }
    let rows_u32 = check_u32("output rows M", rows)?;
    let columns_u32 = check_u32("output columns N", columns)?;
    let inner_u32 = check_u32("contraction length K", inner)?;
    let batch_count = batch.numel()?;
    let batch_count_u32 = check_u32("batch slice count", batch_count)?;
    check_u32("output element count", out_numel)?;

    // ---- operand addressing -------------------------------------------------
    // The contraction axis is the last axis of the left operand (`[M, K]`)
    // and the second-to-last of the right operand (`[K, N]`).
    let lhs_layout = operand_layout("lhs", lhs, &batch, lhs_rank, lhs_rank - 1)?;
    let rhs_layout = operand_layout("rhs", rhs, &batch, rhs_rank, rhs_rank - 2)?;

    let mut batch_dims = [1u32; MAX_BATCH_AXES];
    for (slot, extent) in batch.dims().iter().enumerate() {
        batch_dims[MAX_BATCH_AXES - batch.rank() + slot] =
            check_u32("broadcast batch extent", *extent)?;
    }

    let lhs_bytes =
        lhs_layout
            .window_elements
            .checked_mul(4)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "vulkan matmul lhs upload",
                axis: None,
                detail: format!("{} elements of f32", lhs_layout.window_elements),
            })?;
    let rhs_bytes =
        rhs_layout
            .window_elements
            .checked_mul(4)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "vulkan matmul rhs upload",
                axis: None,
                detail: format!("{} elements of f32", rhs_layout.window_elements),
            })?;
    let out_bytes = out_numel.checked_mul(4).ok_or(Fn3Error::ShapeOverflow {
        context: "vulkan matmul readback",
        axis: None,
        detail: format!("{out_numel} elements of f32"),
    })?;
    for bytes in [lhs_bytes, rhs_bytes, out_bytes] {
        backend.capabilities().check_buffer(bytes)?;
    }

    // ---- descriptors, staging and one submission ----------------------------
    let push = PushConstants {
        m: rows_u32,
        n: columns_u32,
        k: inner_u32,
        a_base: 0,
        b_base: 0,
        out_base: 0,
        a_row_stride: lhs_layout.row_stride,
        a_k_stride: lhs_layout.inner_stride,
        b_k_stride: rhs_layout.row_stride,
        b_col_stride: rhs_layout.inner_stride,
        batch_dims,
        a_batch_strides: lhs_layout.batch_strides,
        b_batch_strides: rhs_layout.batch_strides,
    };
    let push_words = push.words();
    let push_bytes: &[u8] = bytemuck::cast_slice(&push_words);

    let kernel = backend.kernel(&MATMUL_TILED)?;
    let workgroups = kernel
        .spec()
        .workgroups(out_numel, out.shape().dims())
        .map_err(|error| error.to_fn3())?;
    let workgroup_limits = backend
        .vulkan_device()
        .limits()
        .max_compute_work_group_count;
    for axis in 0..3 {
        if workgroups[axis] > workgroup_limits[axis] {
            return Err(unsupported(format!(
                "dispatch needs {} workgroups on axis {axis} but \"{}\" allows {}: batch \
                 {batch_count_u32}, M {rows_u32}, N {columns_u32} with {TILE}x{TILE} tiles. \
                 Split the batch or the vocabulary axis",
                workgroups[axis],
                backend.adapter().name,
                workgroup_limits[axis]
            )));
        }
    }

    // A zero-element operand (K == 0) has no bytes to upload, but a descriptor
    // still needs a non-empty range, so it binds a scratch element the shader
    // never reads: the k loop does not execute and every output is +0.0.
    let lhs_buffer =
        backend.alloc_buffer(&BufferDesc::device_storage(lhs_bytes.max(4), "matmul.lhs"))?;
    let rhs_buffer =
        backend.alloc_buffer(&BufferDesc::device_storage(rhs_bytes.max(4), "matmul.rhs"))?;
    let out_buffer =
        backend.alloc_buffer(&BufferDesc::device_storage(out_bytes.max(4), "matmul.out"))?;

    let staging = backend.staging();
    let lhs_staging = staging.acquire(lhs_bytes.max(4)).map_err(|e| e.to_fn3())?;
    let rhs_staging = staging.acquire(rhs_bytes.max(4)).map_err(|e| e.to_fn3())?;
    let readback = staging.acquire(out_bytes).map_err(|e| e.to_fn3())?;
    if lhs_bytes > 0 {
        lhs_staging
            .write(operand_window_bytes(
                "lhs",
                lhs,
                lhs_layout.window_elements,
            )?)
            .map_err(|e| e.to_fn3())?;
    }
    if rhs_bytes > 0 {
        rhs_staging
            .write(operand_window_bytes(
                "rhs",
                rhs,
                rhs_layout.window_elements,
            )?)
            .map_err(|e| e.to_fn3())?;
    }

    let bindings = [
        BindingArg::whole(0, BindingRole::Input, &lhs_buffer),
        BindingArg::whole(1, BindingRole::Input, &rhs_buffer),
        BindingArg::whole(2, BindingRole::Output, &out_buffer),
    ];

    let mut submitter = backend.checkout_submitter()?;
    let mut dispatch = None;
    let submission = {
        let lhs_staging_buffer = lhs_staging.buffer();
        let rhs_staging_buffer = rhs_staging.buffer();
        let readback_buffer = readback.buffer();
        submitter
            .record(|recorder| {
                if lhs_bytes > 0 {
                    recorder.copy_buffer(lhs_staging_buffer, 0, &lhs_buffer, 0, lhs_bytes)?;
                }
                if rhs_bytes > 0 {
                    recorder.copy_buffer(rhs_staging_buffer, 0, &rhs_buffer, 0, rhs_bytes)?;
                }
                for buffer in [&lhs_buffer, &rhs_buffer] {
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
                    push_constants: push_bytes,
                    workgroups,
                })?;
                recorder.barrier(
                    &out_buffer,
                    ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                    ash::vk::PipelineStageFlags::TRANSFER,
                    ash::vk::AccessFlags::SHADER_WRITE,
                    ash::vk::AccessFlags::TRANSFER_READ,
                );
                recorder.copy_buffer(&out_buffer, 0, readback_buffer, 0, out_bytes)?;
                dispatch = Some(info);
                Ok(())
            })
            .map_err(|error| error.to_fn3())?
    };
    let report = submission.wait().map_err(|error| error.to_fn3())?;
    drop(submitter);

    let mut output_bytes = vec![0u8; out_bytes as usize];
    readback
        .read_into(&mut output_bytes)
        .map_err(|error| error.to_fn3())?;
    {
        let out_slice = out.as_mut_slice::<f32>()?;
        let out_slice_bytes = bytemuck::cast_slice_mut::<f32, u8>(out_slice);
        if out_slice_bytes.len() != output_bytes.len() {
            return Err(Fn3Error::CapacityExceeded {
                what: "vulkan matmul output copy",
                requested: output_bytes.len() as u64,
                capacity: out_slice_bytes.len() as u64,
            });
        }
        out_slice_bytes.copy_from_slice(&output_bytes);
    }

    let dispatch = dispatch.ok_or_else(|| Fn3Error::InvalidShape {
        context: "vulkan matmul",
        detail: "the recorder produced no dispatch information".to_string(),
    })?;
    let mut record = ExecutionRecord::new(0, OpId::MatMul, MATMUL_TILED.name, backend.device_id());
    record.submissions = report.submissions;
    record.workgroups = dispatch.workgroups;
    record.bindings = dispatch.bindings;
    record.upload_bytes = lhs_bytes + rhs_bytes;
    record.readback_bytes = out_bytes;
    record.inputs_read_back = false;
    record.host_staged = true;
    record.device_local_bytes = lhs_buffer.size() + rhs_buffer.size() + out_buffer.size();
    record.host_ns = report.host_ns;
    record.gpu_ns = report.gpu_ns;
    record.gpu_ticks = report.timestamps;
    backend.record_execution(record);

    Ok(())
}

/// Reject attributes `MatMul` does not take instead of ignoring them.
fn require_no_attributes(params: &OpParams) -> Fn3Result<()> {
    if !params.axes.is_empty() || params.keep_dims || params.epsilon != 0.0 {
        return Err(Fn3Error::InvalidShape {
            context: "vulkan matmul attributes",
            detail: format!(
                "op matmul accepts no attributes, but the plan supplies axes={:?}, \
                 keep_dims={}, epsilon={}",
                params.axes, params.keep_dims, params.epsilon
            ),
        });
    }
    Ok(())
}

/// A named `UnsupportedOp` that always says who owns the Vulkan matmul.
fn unsupported(reason: impl Into<String>) -> Fn3Error {
    Fn3Error::UnsupportedOp {
        op: OpId::MatMul,
        backend: VULKAN_BACKEND_NAME,
        reason: format!("{} (owned by FN3-14, #60)", reason.into()),
    }
}

/// Convert an element count or stride to the shader's 32-bit space.
fn check_u32(what: &'static str, value: u64) -> Fn3Result<u32> {
    u32::try_from(value).map_err(|_| {
        unsupported(format!(
            "{what} is {value}, above the 4294967295 elements the kernel can address with a \
             32-bit index"
        ))
    })
}

/// A matrix-axis or batch stride as the shader's `u32`, rejecting negatives.
fn stride_u32(what: &'static str, axis: usize, stride: i64) -> Fn3Result<u32> {
    let unsigned = u32::try_from(stride).map_err(|_| Fn3Error::InvalidStride {
        axis,
        stride,
        detail: format!(
            "vulkan matmul {what}: stride {stride} is negative or above 4294967295 elements"
        ),
    })?;
    Ok(unsigned)
}

/// Extract the addressing facts of one operand.
///
/// `batch` is the right-aligned broadcast shape, so an operand axis whose
/// extent is 1 but whose batch extent is larger has an effective stride of 0:
/// every index along it addresses the same element.
fn operand_layout(
    name: &'static str,
    view: &TensorView<'_>,
    batch: &Shape,
    rank: usize,
    contraction_axis: usize,
) -> Fn3Result<OperandLayout> {
    let strides = view.strides().as_slice();
    let row_stride = stride_u32(name, rank - 2, strides[rank - 2])?;
    let inner_stride = stride_u32(name, rank - 1, strides[rank - 1])?;
    if strides[contraction_axis] == 0 {
        return Err(Fn3Error::InvalidStride {
            axis: contraction_axis,
            stride: 0,
            detail: format!(
                "vulkan matmul {name} ({}) contracts over axis {contraction_axis} with a zero \
                 stride; a zero-stride contraction is an explicitly broadcast view, not a \
                 matrix product. The CPU reference rejects it the same way",
                view.shape()
            ),
        });
    }

    let mut batch_strides = [0u32; MAX_BATCH_AXES];
    let batch_axes = rank - 2;
    let lead = batch
        .rank()
        .checked_sub(batch_axes)
        .ok_or_else(|| Fn3Error::InvalidShape {
            context: "vulkan matmul batch alignment",
            detail: format!(
                "{name} has {batch_axes} batch axes but the broadcast batch {batch} has {}",
                batch.rank()
            ),
        })?;
    for axis in lead..batch.rank() {
        let operand_axis = axis - lead;
        let extent = view.shape().dims()[operand_axis];
        let stride = stride_u32(name, operand_axis, strides[operand_axis])?;
        let effective = if extent == 1 && batch.dims()[axis] != 1 {
            0
        } else {
            stride
        };
        batch_strides[MAX_BATCH_AXES - batch.rank() + axis] = effective;
    }

    let offset = view.meta().offset() as u64;
    let required = view.meta().required_elements()?;
    let window_elements = required
        .checked_sub(offset)
        .ok_or_else(|| Fn3Error::InvalidShape {
            context: "vulkan matmul operand window",
            detail: format!("{name} required elements {required} are below its offset {offset}"),
        })?;
    // The shader's index arithmetic is `u32`; the device's
    // `maxStorageBufferRange` already caps a legal buffer below that, so this
    // only guards metadata that no buffer can satisfy.
    check_u32(
        match name {
            "lhs" => "lhs addressed span",
            _ => "rhs addressed span",
        },
        window_elements,
    )?;
    Ok(OperandLayout {
        window_elements,
        row_stride,
        inner_stride,
        batch_strides,
    })
}

/// The addressed window of `view` as little-endian `f32` bytes.
///
/// The window starts at the view's element offset and covers its stride span,
/// which is exactly what the shader can address. Uploading only the window
/// keeps a sliced view from transferring its whole parent allocation.
fn operand_window_bytes<'a>(
    name: &'static str,
    view: &TensorView<'a>,
    window_elements: u64,
) -> Fn3Result<&'a [u8]> {
    let start = view.meta().offset();
    let window = usize::try_from(window_elements).map_err(|_| Fn3Error::ShapeOverflow {
        context: "vulkan matmul operand window",
        axis: None,
        detail: format!("{name}: {window_elements} elements do not fit in usize"),
    })?;
    let end = start.checked_add(window).ok_or(Fn3Error::ShapeOverflow {
        context: "vulkan matmul operand window",
        axis: None,
        detail: format!("{name}: element {start} plus window {window}"),
    })?;
    let storage = view.storage_slice::<f32>()?;
    let slice = storage
        .get(start..end)
        .ok_or_else(|| Fn3Error::OutOfBounds {
            what: "vulkan matmul operand window (storage elements)",
            index: end as u64,
            len: storage.len() as u64,
        })?;
    Ok(bytemuck::cast_slice(slice))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_matches_the_committed_shader_metadata() {
        assert_eq!(MATMUL_TILED.workgroup_size, [16, 16, 1]);
        assert_eq!(
            MATMUL_TILED.push_constant_bytes,
            (PUSH_CONSTANT_WORDS * 4) as u32
        );
        assert_eq!(MATMUL_TILED.bindings.len(), 3);
        assert_eq!(MATMUL_TILED.entry_point, "main");
        assert!(MATMUL_TILED.validate().is_ok());
    }

    #[test]
    fn grid_puts_the_batch_on_z_and_rounds_tiles_up() {
        let input = GridInput {
            element_count: 0,
            output_dims: &[2, 3, 5, 11],
            workgroup_size: [TILE, TILE, 1],
        };
        // 11 columns -> one 16-wide tile, 5 rows -> one 16-tall tile, 6 batches.
        assert_eq!(grid_matmul(&input).unwrap(), [1, 1, 6]);

        let input = GridInput {
            output_dims: &[17, 32],
            ..input
        };
        assert_eq!(grid_matmul(&input).unwrap(), [2, 2, 1]);

        let input = GridInput {
            // The decode projection: 50,257 columns, one row, one batch slice.
            output_dims: &[1, 50257],
            ..input
        };
        assert_eq!(grid_matmul(&input).unwrap(), [3142, 1, 1]);

        let input = GridInput {
            // The same vocabulary with 128 rows: 50,257 columns still need
            // 3,142 tiles, and 128 rows need eight.
            output_dims: &[128, 50257],
            ..input
        };
        assert_eq!(grid_matmul(&input).unwrap(), [3142, 8, 1]);

        let scalar = GridInput {
            output_dims: &[3],
            ..input
        };
        assert!(grid_matmul(&scalar).is_err(), "rank 1 has no matrix axes");
    }

    #[test]
    fn push_constants_pack_in_declaration_order() {
        let push = PushConstants {
            m: 1,
            n: 2,
            k: 3,
            a_base: 4,
            b_base: 5,
            out_base: 6,
            a_row_stride: 7,
            a_k_stride: 8,
            b_k_stride: 9,
            b_col_stride: 10,
            batch_dims: [11, 12, 13, 14, 15, 16],
            a_batch_strides: [17, 18, 19, 20, 21, 22],
            b_batch_strides: [23, 24, 25, 26, 27, 28],
        };
        assert_eq!(
            push.words().to_vec(),
            (1u32..=28).collect::<Vec<u32>>(),
            "every word is in the same order the shader declares"
        );
    }

    #[test]
    fn unsupported_errors_name_the_owner() {
        let error = unsupported("test reason");
        let text = error.to_string();
        assert!(text.contains("#60"), "{text}");
        assert!(text.contains("vulkan"), "{text}");
    }
}
