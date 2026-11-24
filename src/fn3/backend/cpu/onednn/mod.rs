//! The optional oneDNN CPU engine.
//!
//! # What this engine is
//!
//! A [`SegmentProvider`] that claims `f32` 2-D NCHW convolution and 2-D
//! matmul, and runs them on oneDNN 3.14.1 (one `dnnl_engine`, one sequential
//! stream, and one prepared primitive per output-channel block). Everything
//! else stays on the scalar kernels; the planner records the split in
//! [`Plan::fallbacks`](crate::inference::runtime::Plan::fallbacks).
//!
//! # Why pooling is not claimed
//!
//! The engine hands oneDNN the arena's plain NCHW descriptor, because that is
//! how the plan stores activations and the engine does not reorder a pooling
//! operand. On AVX2 the x64 JIT pooling kernel does not accept that descriptor:
//! `jit_uni_pool_kernel.cpp` leaves `ncsp_fmt_tag` unset unless the ISA is
//! AVX-512, so oneDNN answers with `ref_pooling`. Measured on the SPPF pools of
//! YOLOv8n (three 128x20x20 5x5 windows), that reference implementation cost
//! 0.98 ms per forward, against 3.27 ms for the scalar kernel of the time and
//! 0.45 ms after the scalar kernel grew a separable 2-D form. A claim whose
//! best case here is the provider's slowest implementation is not a claim worth
//! making from a pure function of the segment, which is all the planner can
//! offer (it runs before an engine exists). Pooling therefore stays scalar; a
//! host whose oneDNN JITs plain NCHW pooling would have to re-earn the claim
//! with a measurement.
//!
//! # oneDNN is sequential here, on purpose
//!
//! The build is pinned to `ONEDNN_CPU_RUNTIME=SEQ` (`.cargo/config.toml` and
//! `tools/cmake/onednn-seq.cmake`). A threaded oneDNN would spawn its own team
//! inside every call, on top of the session's pool, and the two would
//! oversubscribe the machine. The engine reads
//! `dnnl_version()->cpu_runtime` and, if a foreign build reports a threaded
//! runtime, still produces correct output but runs its blocks serially instead
//! of nesting teams.
//!
//! # Threading: blocks, not oneDNN
//!
//! A segment is split into contiguous **output-channel blocks** at plan time,
//! from the session's worker count:
//!
//! ```text
//! plain conv:      channels 0..8 | 8..16 | 16..24 | ...        (one submemory each)
//! grouped conv:    whole groups only; a boundary inside a group would
//!                  need a weight submemory oneDNN cannot express
//! depthwise conv:  one channel (one group) per block
//! matmul:          result columns 0..n | n..2n | ...
//! ```
//!
//! The block count is `min(workers, units / min_units)`, where `min_units`
//! enforces at least [`MIN_BLOCK_CHANNELS`] channels and
//! [`MIN_BLOCK_ELEMENTS`] output elements per block; a problem below either
//! floor stays one block. The split is deterministic because the only host
//! input is the session's worker count, which the plan/build arguments record.
//!
//! Each block describes its slice with `dnnl_memory_desc_create_submemory`:
//!
//! - **weights**: a submemory of the ONNX payload, `[o0, 0, 0, 0]`, i.e. the
//!   element offset `o0 * (C/group) * kh * kw`. It is read **once**, at session
//!   open, by the prepack reorder.
//! - **destination**: a submemory of the output slot, `[0, o0, 0, 0]`, i.e. the
//!   element offset `o0 * H_out * W_out` per batch row.
//! - **source**: the full input slot for a plain convolution (no halo, no
//!   spatial split), or a channel submemory `[0, g0 * C/group, 0, 0]` for a
//!   grouped block (groups change which input channels are read; there is still
//!   no spatial halo).
//!
//! Blocks are independent: each owns its primitive, its stream and its
//! scratchpad, and writes a disjoint destination range. The executor runs them
//! through [`SessionThreads`](crate::inference::runtime::SessionThreads) when
//! oneDNN is sequential.
//!
//! # What is prepacked, and when
//!
//! At **session construction**, once per segment and block:
//!
//! 1. the weight payload is copied into a 64-byte-aligned staging buffer (the
//!    importer promises no alignment);
//! 2. the primitive descriptor is created with the weight descriptor `any`, so
//!    oneDNN chooses the layout its kernel wants;
//! 3. the descriptor oneDNN chose is queried and **cloned**;
//! 4. the block's weights are reordered into an engine-owned buffer of exactly
//!    that descriptor, and the staging copy is dropped.
//!
//! The result is one session-owned buffer per block, reused unchanged by every
//! forward. No forward calls a reorder, and no forward touches the ONNX weight
//! payload. `PreparedSegment`/`Block` are a `Vec` indexed by plan position, so
//! the "cache" is deterministic and has no hash iteration order.
//!
//! The bias (when present) is copied per block into an aligned buffer in the
//! same step; it is small and constant, which the planner guarantees by
//! claiming only segments whose weight and bias slots are read-only.
//!
//! # Buffers and lifetimes
//!
//! oneDNN borrows a host pointer only for the duration of
//! `dnnl_primitive_execute`. A block's memories are created with
//! `DNNL_MEMORY_NONE`; immediately before each execute the engine installs the
//! pointers for that call:
//!
//! - source and destination are the executor's live arena windows
//!   (`&[u8]`/`&mut [u8]`, disjoint by the plan);
//! - weights and bias are the block's own prepacked buffers;
//! - the scratchpad is the block's own buffer.
//!
//! Nothing caches an arena pointer between forwards, so the arena may move in
//! address space (it does not, but nothing depends on it) and there is no
//! long-lived `&mut` provenance for a later write to invalidate. The arena and
//! the prepacked buffers outlive the `dnnl_primitive_execute` call by
//! construction: the engine is a field of the session and the buffers are
//! fields of the engine.
//!
//! # Scratchpads
//!
//! Primitives are created with `dnnl_scratchpad_mode_user`. oneDNN's default
//! library scratchpad is shared per primitive and, with
//! `DNNL_ENABLE_CONCURRENT_EXEC=OFF`, tied to the creating thread; a block
//! executes on whichever worker the pool assigns, so the engine supplies the
//! scratchpad instead. The buffers are session-wide: one slot per worker, each
//! the size of the largest scratchpad any block needs, instead of one buffer
//! per block (which cost 27-29 MB on the YOLO graphs). Different concurrent
//! blocks still use different slots, the documented condition for executing
//! them concurrently.

pub(crate) mod ffi;
pub(crate) mod geometry;

use crate::inference::backend::{
    Activation, Claim, FusedOp, OpParams, SegmentEngine, SegmentOperand, SegmentProvider,
};
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::shape::Shape;
use crate::inference::storage::AlignedBuffer;
use std::sync::{Arc, Mutex};

use self::ffi::{
    context, Desc, EngineHandle, ExecArg, MemoryHandle, PostOpsHandle, PrimitiveAttrHandle,
    PrimitiveDescHandle, PrimitiveHandle, StreamHandle, Version, ALG_ELTWISE_CLIP,
    ALG_ELTWISE_GELU_ERF, ALG_ELTWISE_LOGISTIC, ALG_ELTWISE_RELU, ALG_ELTWISE_SWISH,
    ALG_ELTWISE_TANH, ARG_BIAS, ARG_DST, ARG_SCRATCHPAD, ARG_SRC, ARG_WEIGHTS, FORMAT_TAG_A,
    FORMAT_TAG_ABCD, FORMAT_TAG_ABCDE, FORMAT_TAG_ANY, FORMAT_TAG_NHWC, QUERY_WEIGHTS_MD,
};
use self::geometry::{Conv, ConvSplitAxis, MatMul};

/// Provider name in errors.
const BACKEND: &str = "oneDNN";

/// The ISA oneDNN executes on and whether it has a JIT int8 convolution there.
///
/// Returns `(isa code, isa name, int8 available)`. Used by `prepare` to refuse
/// `precision = "w8a8"` with a measured reason instead of writing an artifact
/// no engine can run.
pub(crate) fn int8_execution_support() -> (i32, &'static str, bool) {
    let isa = ffi::effective_cpu_isa();
    (
        isa,
        ffi::cpu_isa_name(isa),
        ffi::int8_kernels_available(isa),
    )
}

/// The planner-facing capability: it owns no oneDNN object.
///
/// A plan is built before a session (and therefore before an engine) exists, so
/// the claim must be a pure function of the segment. [`OneDnnEngine`] is built
/// afterwards and prepares exactly the segments this claimed.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OneDnnCapability;

impl SegmentProvider for OneDnnCapability {
    fn engine(&self) -> SegmentEngine {
        SegmentEngine::OneDnn
    }

    fn claim(
        &self,
        op: OpId,
        inputs: &[SegmentOperand],
        outputs: &[SegmentOperand],
        params: &OpParams,
    ) -> Claim {
        if !is_claimed_op(op) {
            return Claim::NotMine;
        }
        match claimed_geometry(op, inputs, outputs, params) {
            Ok(()) => Claim::Accepted,
            Err(error) => Claim::Declined(error.to_string()),
        }
    }

    fn claim_fused(
        &self,
        fused: &FusedOp,
        inputs: &[SegmentOperand],
        outputs: &[SegmentOperand],
        params: &OpParams,
    ) -> Claim {
        // The fused vocabulary is a convolution with a sum and/or one
        // activation: exactly the oneDNN post-op pair. Any other anchor is
        // outside it, not a fallback.
        if fused.anchor != OpId::Conv {
            return Claim::NotMine;
        }
        match claimed_fused_conv(fused, inputs, outputs, params) {
            Ok(()) => Claim::Accepted,
            Err(error) => Claim::Declined(error.to_string()),
        }
    }
}

/// The claim for a fused convolution: the plain convolution's geometry plus the
/// epilogue's own rules.
fn claimed_fused_conv(
    fused: &FusedOp,
    inputs: &[SegmentOperand],
    outputs: &[SegmentOperand],
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let op = OpId::Conv;
    // The residual operand is the last input; the convolution's own operands
    // are the ones the unfused segment would have had.
    let anchor_inputs = match fused.sum {
        Some(position) => {
            if position + 1 != inputs.len() {
                return Err(unsupported(
                    op,
                    format!(
                        "the fused residual is operand {position}, but the segment has {} inputs",
                        inputs.len()
                    ),
                ));
            }
            &inputs[..position]
        }
        None => inputs,
    };
    claimed_geometry(op, anchor_inputs, outputs, params)?;
    if let Some(position) = fused.sum {
        let sum = &inputs[position];
        if sum.constant {
            return Err(unsupported(
                op,
                "the residual operand is constant; the engine pre-fills the destination from a \
                 live arena window",
            ));
        }
        require_f32(op, &[sum])?;
        if sum.shape != outputs[0].shape {
            return Err(unsupported(
                op,
                format!(
                    "the residual operand is {} but the result is {}; a oneDNN sum post-op has no \
                     broadcast",
                    sum.shape, outputs[0].shape
                ),
            ));
        }
    }
    if fused.activation.is_some_and(|activation| {
        matches!(activation, Activation::Clip { low, high } if !low.is_finite() || !high.is_finite())
    }) {
        return Err(unsupported(
            op,
            "a clip bound is not finite; oneDNN eltwise_clip needs both bounds",
        ));
    }
    Ok(())
}

fn unsupported(op: OpId, reason: impl Into<String>) -> Fn3Error {
    Fn3Error::UnsupportedOp {
        op,
        backend: BACKEND,
        reason: reason.into(),
    }
}

fn shape_of(operand: &SegmentOperand) -> &Shape {
    &operand.shape
}

fn require_f32(op: OpId, operands: &[&SegmentOperand]) -> Result<(), Fn3Error> {
    for (index, operand) in operands.iter().enumerate() {
        if operand.dtype != DType::F32 {
            return Err(Fn3Error::UnsupportedDType {
                op: Some(op),
                dtype: operand.dtype,
                context: if index == 0 {
                    "the oneDNN CPU engine computes f32 only"
                } else {
                    "the oneDNN CPU engine computes f32 only on this operand"
                },
            });
        }
    }
    Ok(())
}

/// `true` for the ops this engine implements at all.
fn is_claimed_op(op: OpId) -> bool {
    matches!(op, OpId::Conv | OpId::MatMul)
}

/// The claim: pure `(op, operands, params) -> yes/reason`.
fn claimed_geometry(
    op: OpId,
    inputs: &[SegmentOperand],
    outputs: &[SegmentOperand],
    params: &OpParams,
) -> Result<(), Fn3Error> {
    match op {
        OpId::Conv => {
            if outputs.len() != 1 || !(2..=3).contains(&inputs.len()) {
                return Err(unsupported(
                    op,
                    format!(
                        "expected 2 or 3 inputs and one output, found {} and {}",
                        inputs.len(),
                        outputs.len()
                    ),
                ));
            }
            require_f32(op, &[&inputs[0], &inputs[1], &outputs[0]])?;
            // The weight and bias must be read-only payloads: they are
            // prepacked at session open, so a value computed by an earlier
            // segment would not exist yet.
            if !inputs[1].constant {
                return Err(unsupported(
                    op,
                    "the weight is produced at run time; the oneDNN engine prepacks constant \
                     weights at session open",
                ));
            }
            let conv = Conv::new(
                shape_of(&inputs[0]),
                shape_of(&inputs[1]),
                shape_of(&outputs[0]),
                params,
            )?;
            if let Some(bias) = inputs.get(2) {
                require_f32(op, &[bias])?;
                if !bias.constant {
                    return Err(unsupported(
                        op,
                        "the bias is produced at run time; the oneDNN engine prepacks constant \
                         biases at session open",
                    ));
                }
                let dims = bias.shape.dims();
                if dims.len() != 1 || dims[0] != conv.out_channels as u64 {
                    return Err(unsupported(
                        op,
                        format!(
                            "the bias is {:?}, but a 1-D [{}] bias is required",
                            dims, conv.out_channels
                        ),
                    ));
                }
            }
            Ok(())
        }
        OpId::MatMul => {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(unsupported(
                    op,
                    format!(
                        "expected 2 inputs and one output, found {} and {}",
                        inputs.len(),
                        outputs.len()
                    ),
                ));
            }
            require_f32(op, &[&inputs[0], &inputs[1], &outputs[0]])?;
            if !inputs[1].constant {
                return Err(unsupported(
                    op,
                    "the right operand is produced at run time; the oneDNN engine prepacks \
                     constant weights at session open",
                ));
            }
            MatMul::new(
                shape_of(&inputs[0]),
                shape_of(&inputs[1]),
                shape_of(&outputs[0]),
            )
            .map(|_| ())
        }
        other => Err(unsupported(
            other,
            "the oneDNN engine implements Conv and 2-D MatMul",
        )),
    }
}

/// One segment, described by the executor for preparation.
///
/// Shapes and payloads rather than `Plan` slots: this module sits under
/// `backend` and must not import the runtime.
pub(crate) struct SegmentSpec<'a> {
    /// The op to prepare.
    pub op: OpId,
    /// Input extents, in operand order.
    pub inputs: Vec<Shape>,
    /// Input element types, in operand order.
    pub input_dtypes: Vec<DType>,
    /// Whether each input is read-only model data, in operand order.
    pub input_constants: Vec<bool>,
    /// Output extents, in operand order.
    pub outputs: Vec<Shape>,
    /// Output element types, in operand order.
    pub output_dtypes: Vec<DType>,
    /// The op's attributes.
    pub params: &'a OpParams,
    /// Weight payload for a constant `inputs[1]`, in the ONNX layout.
    pub weights: Option<&'a [u8]>,
    /// Bias payload for a constant `inputs[2]`, in the ONNX layout.
    pub bias: Option<&'a [u8]>,
    /// The session's worker count, which fixes the block split.
    pub workers: usize,
    /// The recognised chain this segment folds, when it is a fused segment.
    pub fused: Option<&'a FusedOp>,
    /// Plan slot indices of the inputs, in operand order.
    pub input_slots: Vec<usize>,
    /// Plan slot indices of the outputs, in operand order.
    pub output_slots: Vec<usize>,
    /// The executor's layout decision: every consumer of the output slot is a
    /// oneDNN convolution, so the result may stay in the channels-last layout
    /// the engine chose. The engine still requires a row split (the only block
    /// shape whose channels-last destination range is one contiguous run) and
    /// falls back to the plain layout when rows cannot fill the pool.
    pub dst_blocked: bool,
}

impl SegmentSpec<'_> {
    fn operands(&self) -> Vec<SegmentOperand> {
        self.inputs
            .iter()
            .cloned()
            .zip(self.input_dtypes.iter().copied())
            .zip(self.input_constants.iter().copied())
            .map(|((shape, dtype), constant)| SegmentOperand {
                shape,
                dtype,
                constant,
            })
            .collect()
    }

    fn results(&self) -> Vec<SegmentOperand> {
        self.outputs
            .iter()
            .cloned()
            .zip(self.output_dtypes.iter().copied())
            .map(|(shape, dtype)| SegmentOperand {
                shape,
                dtype,
                constant: false,
            })
            .collect()
    }
}

/// One prepared block: a primitive, the streams/buffers it needs, and the
/// argument list rebuilt from live pointers on every call.
#[derive(Debug)]
struct Block {
    /// The compiled kernel.
    primitive: PrimitiveHandle,
    /// This block's own stream; a block runs on one pool worker at a time.
    stream: StreamHandle,
    /// One memory object per argument, created without a buffer.
    memories: Vec<MemoryHandle>,
    /// The `(argument, memory)` list handed to `dnnl_primitive_execute`.
    args: Vec<ExecArg>,
    /// Where each argument's pointer comes from at run time.
    sources: Vec<ArgSource>,
    /// The block's prepacked weights, when the op has weights.
    ///
    /// Shared by the row blocks of one convolution: they use the same weight
    /// descriptor and read it immutably, so one prepacked copy serves them all
    /// (and stays in cache across the blocks).
    weights: Option<Arc<AlignedBuffer>>,
    /// The block's bias slice, when the op has a constant bias.
    bias: Option<AlignedBuffer>,
    /// Bytes of user-managed scratchpad the primitive needs; zero when it
    /// needs none. The buffer itself is leased from the session-wide pool.
    scratchpad_bytes: usize,
    /// Dense source/destination staging for a row block. `None` otherwise.
    staging: Option<RowStaging>,
    /// The fused activation, applied by the engine after the primitive.
    epilogue: Option<FusedEpilogue>,
    /// The channels-last buffer this block reads, when its source is blocked.
    blocked_source: Option<Arc<AlignedBuffer>>,
    /// The channels-last buffer this block writes, when its destination is
    /// blocked. Shared with the other row blocks of the segment and with the
    /// engine's layout table.
    blocked_destination: Option<Arc<AlignedBuffer>>,
}

/// Dense staging for one row block.
///
/// # Why it exists
///
/// A row submemory of the parent slot keeps the parent's strides, so its
/// descriptor no longer matches `nchw` (`C * H * W` is `C * in_rows * W`
/// only when the block spans the whole height). Measured on oneDNN 3.14.1, a
/// `[1, 16, 21, 160]` source submemory of `[1, 16, 160, 160]` runs a 3x3
/// convolution in ~100 ms per block, against ~4.8 ms for the whole dense
/// tensor: the mismatched strides take the primitive off its JIT path. The
/// block therefore copies its input rows into a dense buffer, runs the
/// primitive on dense descriptors, and copies the dense output rows back.
/// That is one extra input read plus one output write per block — still O(1)
/// input traffic, not the O(blocks) the channel split pays.
#[derive(Debug)]
struct RowStaging {
    /// Byte length of the dense source rows this block copies in; zero when
    /// the source is already dense (a channels-last block reads its buffer).
    source_bytes: usize,
    /// Byte length of the dense destination rows; zero when the destination is
    /// blocked.
    destination_bytes: usize,
    /// Byte length of the parent input slot (what the executor hands us).
    source_len: usize,
    /// Byte length of the parent output slot.
    destination_len: usize,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    in_rows: usize,
    out_rows: usize,
    in_row_start: usize,
    out_row_start: usize,
    in_h: usize,
    out_h: usize,
    in_row_bytes: usize,
    out_row_bytes: usize,
}

impl RowStaging {
    /// Copy this block's input rows into `destination_buffer`, a pooled slot.
    fn copy_in(&self, address: usize, destination_buffer: &AlignedBuffer) -> Result<(), Fn3Error> {
        if self.source_bytes == 0 {
            return Ok(());
        }
        if address == 0 {
            return Err(segment_error(
                "a staged source has a null address".to_string(),
            ));
        }
        // SAFETY: `address` is the executor's live source window for this
        // call. The plan sized that slot to `source_len` bytes (batch, channels,
        // in_h, in_w), which is the length used here; the window outlives the
        // copy. The destination is the staging slot this block leased for the
        // call, and one worker runs the block at a time.
        let source = unsafe { std::slice::from_raw_parts(address as *const u8, self.source_len) };
        let destination = unsafe {
            std::slice::from_raw_parts_mut(
                destination_buffer.as_ptr() as *mut u8,
                destination_buffer.len(),
            )
        };
        let run = self.in_rows * self.in_row_bytes;
        for batch in 0..self.batch {
            for channel in 0..self.in_channels {
                let from = ((batch * self.in_channels + channel) * self.in_h + self.in_row_start)
                    * self.in_row_bytes;
                let to = (batch * self.in_channels + channel) * run;
                destination[to..to + run].copy_from_slice(&source[from..from + run]);
            }
        }
        Ok(())
    }

    /// Copy this block's output rows back out of `source_buffer`, a pooled slot.
    fn copy_out(&self, address: usize, source_buffer: &AlignedBuffer) -> Result<(), Fn3Error> {
        if self.destination_bytes == 0 {
            return Ok(());
        }
        if address == 0 {
            return Err(segment_error(
                "a staged destination has a null address".to_string(),
            ));
        }
        // SAFETY: `address` is the executor's live destination window for this
        // call, `destination_len` bytes (batch, channels, out_h, out_w), and it
        // is disjoint from the source by the plan. The staging buffer is the
        // slot this block leased; one worker runs the block at a time.
        let destination =
            unsafe { std::slice::from_raw_parts_mut(address as *mut u8, self.destination_len) };
        let source =
            unsafe { std::slice::from_raw_parts(source_buffer.as_ptr(), source_buffer.len()) };
        let run = self.out_rows * self.out_row_bytes;
        for batch in 0..self.batch {
            for channel in 0..self.out_channels {
                let from = (batch * self.out_channels + channel) * run;
                let to = ((batch * self.out_channels + channel) * self.out_h + self.out_row_start)
                    * self.out_row_bytes;
                destination[to..to + run].copy_from_slice(&source[from..from + run]);
            }
        }
        Ok(())
    }
}

/// Row-staging buffers shared by every block of the session's engine.
///
/// # Why it exists
///
/// A row block stages dense source and destination rows while it runs (see
/// [`RowStaging`]). Those buffers used to be owned by the block, allocated at
/// session open and retained for the session's life, so staging memory grew
/// with the *block count*: eight workers on YOLOv8n allocated 14,110,720 B.
/// At most one staged block per worker runs at a time, so one slot per worker
/// is enough, and each slot only has to be as large as the largest block. The
/// pool makes staging O(workers x block) instead of O(blocks), while every
/// block still gets a dense buffer for its own geometry.
#[derive(Debug)]
struct StagingPool {
    slots: Vec<StagingSlot>,
    /// Indices of free slots. Locking costs a few nanoseconds per block,
    /// against the millisecond-scale primitive the block runs.
    free: Mutex<Vec<usize>>,
}

#[derive(Debug)]
struct StagingSlot {
    /// Dense NCHW input rows; `None` when no staged block copies input rows
    /// (every source is already blocked).
    source: Option<AlignedBuffer>,
    /// Dense NCHW output rows; `None` when every destination is blocked.
    destination: Option<AlignedBuffer>,
    /// User scratchpad for the block running on this slot; `None` when no
    /// prepared block needs one.
    scratchpad: Option<AlignedBuffer>,
}

/// One leased staging slot, returned to the pool on drop.
struct StagingLease<'a> {
    pool: &'a StagingPool,
    index: usize,
}

impl<'a> StagingLease<'a> {
    fn slot(&self) -> &'a StagingSlot {
        &self.pool.slots[self.index]
    }
}

impl Drop for StagingLease<'_> {
    fn drop(&mut self) {
        let mut free = self
            .pool
            .free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        free.push(self.index);
    }
}

impl StagingPool {
    fn new(
        slots: usize,
        source_bytes: usize,
        destination_bytes: usize,
        scratchpad_bytes: usize,
    ) -> Result<Self, Fn3Error> {
        assert!(slots > 0, "a pool with no slot cannot serve a block");
        let mut buffers = Vec::with_capacity(slots);
        for _ in 0..slots {
            let source = if source_bytes == 0 {
                None
            } else {
                Some(AlignedBuffer::zeroed(source_bytes)?)
            };
            let destination = if destination_bytes == 0 {
                None
            } else {
                Some(AlignedBuffer::zeroed(destination_bytes)?)
            };
            let scratchpad = if scratchpad_bytes == 0 {
                None
            } else {
                Some(AlignedBuffer::zeroed(scratchpad_bytes)?)
            };
            buffers.push(StagingSlot {
                source,
                destination,
                scratchpad,
            });
        }
        Ok(StagingPool {
            slots: buffers,
            free: Mutex::new((0..slots).rev().collect()),
        })
    }

    fn acquire(&self) -> StagingLease<'_> {
        let mut free = self
            .free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let index = free
            .pop()
            .expect("the pool has one slot per concurrently running staged block");
        StagingLease { pool: self, index }
    }

    fn staging_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|slot| {
                slot.source.as_ref().map_or(0, AlignedBuffer::len)
                    + slot.destination.as_ref().map_or(0, AlignedBuffer::len)
            })
            .sum()
    }

    fn scratchpad_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|slot| slot.scratchpad.as_ref().map_or(0, AlignedBuffer::len))
            .sum()
    }
}

/// Where a scalar epilogue writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EpilogueTarget {
    /// The block's output window (a channel range or the whole tensor).
    Output,
    /// The block's dense NCHW destination staging.
    StagingDestination,
    /// The block's dense channels-last destination range.
    BlockedDestination,
}

/// The fused activation, evaluated by the engine over the block's destination.
///
/// # Why not a oneDNN eltwise post-op
///
/// Measured on this host (oneDNN 3.14.1, AVX2, one `[1,32,80,80]` 3x3
/// convolution with 204 800 outputs): a `relu` post-op costs 1.3 ns/element, a
/// `clip` 6 ns, a `logistic` 9 ns and a `swish` 9.8 ns, against 0.4–2.7 ns for
/// this engine's scalar loop over the same buffer — oneDNN evaluates post-ops
/// on a reference path here, and `swish` as a post-op made a forward 3.4x
/// slower than the two scalar segments it replaced.
///
/// # Why a standalone eltwise primitive is different
///
/// A standalone `dnnl_eltwise_forward` over 16.8 M f32 elements (the number of
/// SiLU elements in a YOLO11n forward) measured **10.4 ms** for `swish` and
/// 9.8 ms for `logistic` against **45.9 ms** for the scalar loop, so the
/// transcendental activations use the primitive. `relu` and `clip` stay scalar:
/// oneDNN's memory-bound eltwise is slower than the scalar loop there. The
/// scalar path is also kept as the fallback if a primitive cannot be created.
#[derive(Debug)]
struct FusedEpilogue {
    activation: Activation,
    /// Contiguous runs of `f32` the activation applies to.
    runs: usize,
    /// Elements per run.
    run_len: usize,
    /// Elements between consecutive runs.
    stride: usize,
    /// Byte offset of the block's first run from the target base (a channel
    /// block of a plain destination starts at its first output channel).
    offset_bytes: usize,
    target: EpilogueTarget,
    /// The standalone eltwise primitive, for the activations it accelerates.
    onednn: Option<OneDnnActivation>,
}

/// An in-place oneDNN eltwise primitive bound to the block's destination.
#[derive(Debug)]
struct OneDnnActivation {
    primitive: PrimitiveHandle,
    stream: StreamHandle,
    src: MemoryHandle,
    dst: MemoryHandle,
}

impl FusedEpilogue {
    fn new(
        activation: Activation,
        runs: usize,
        run_len: usize,
        stride: usize,
        offset_bytes: usize,
        target: EpilogueTarget,
        engine: &EngineHandle,
    ) -> Self {
        let onednn = if runs > 0 && run_len > 0 && uses_onednn_eltwise(activation) {
            build_eltwise(activation, run_len, engine).ok()
        } else {
            None
        };
        Self {
            activation,
            runs,
            run_len,
            stride,
            offset_bytes,
            target,
            onednn,
        }
    }

    /// Apply over `address`, the base of the block's destination.
    ///
    /// # Safety
    ///
    /// `address` must be the base of a live, writable region of at least
    /// `(runs - 1) * stride + run_len` `f32` elements, and no other worker may
    /// write the same range at the same time. A block's destination is either
    /// its own output submemory (a channel range of the slot, or the whole
    /// slot), its own staging buffer, or its own row range of the shared
    /// channels-last buffer; the plan proved the slot windows disjoint and the
    /// engine gives every block a distinct row range, so the three hold.
    unsafe fn apply(&self, address: usize) -> Result<(), Fn3Error> {
        if address == 0 || self.runs == 0 || self.run_len == 0 {
            return Err(segment_error(
                "a fused epilogue has no destination".to_string(),
            ));
        }
        let base = address
            .checked_add(self.offset_bytes)
            .ok_or_else(|| segment_error("a fused epilogue offset overflowed".to_string()))?;
        if let Some(onednn) = &self.onednn {
            let width = std::mem::size_of::<f32>();
            for run in 0..self.runs {
                let start = run
                    .checked_mul(self.stride)
                    .and_then(|elements| elements.checked_mul(width))
                    .and_then(|bytes| base.checked_add(bytes))
                    .ok_or_else(|| {
                        segment_error("a fused epilogue run offset overflowed".to_string())
                    })?;
                // SAFETY: the caller's promise holds for every run; the region
                // is this block's own destination, and the primitive reads and
                // writes exactly `run_len` f32 starting at `start`.
                unsafe {
                    onednn.src.set_data_handle(start as *mut u8)?;
                    onednn.dst.set_data_handle(start as *mut u8)?;
                }
                onednn.primitive.execute(
                    &onednn.stream,
                    &[
                        ExecArg {
                            arg: ARG_SRC,
                            memory: onednn.src.raw(),
                        },
                        ExecArg {
                            arg: ARG_DST,
                            memory: onednn.dst.raw(),
                        },
                    ],
                )?;
                onednn.stream.wait()?;
            }
            return Ok(());
        }
        let length = (self.runs - 1) * self.stride + self.run_len;
        // SAFETY: the caller's promise above; `length` is the descriptor's own
        // logical extent, which the engine sized the buffer to.
        let values = unsafe { std::slice::from_raw_parts_mut(base as *mut f32, length) };
        let activation = self.activation;
        for run in 0..self.runs {
            let start = run * self.stride;
            for value in &mut values[start..start + self.run_len] {
                *value = activation.apply(*value);
            }
        }
        Ok(())
    }
}

/// Transcendental activations, where the standalone oneDNN eltwise beats the
/// scalar `exp` loop; `relu`/`clip` stay scalar because they are cheaper there.
fn uses_onednn_eltwise(activation: Activation) -> bool {
    matches!(
        activation,
        Activation::Sigmoid | Activation::Tanh | Activation::Gelu | Activation::Swish
    )
}

/// Build the in-place eltwise primitive for one run of `run_len` elements.
fn build_eltwise(
    activation: Activation,
    run_len: usize,
    engine: &EngineHandle,
) -> Result<OneDnnActivation, Fn3Error> {
    let dims = [run_len as i64];
    let desc = Desc::with_tag(&dims, FORMAT_TAG_A)?;
    let attr = PrimitiveAttrHandle::library_default()?;
    let (alg, alpha, beta) = eltwise(activation);
    let pd = PrimitiveDescHandle::eltwise(engine, alg, &desc, alpha, beta, &attr)?;
    let primitive = PrimitiveHandle::new(&pd)?;
    let stream = StreamHandle::new(engine)?;
    let src = MemoryHandle::new(&desc, engine)?;
    let dst = MemoryHandle::new(&desc, engine)?;
    Ok(OneDnnActivation {
        primitive,
        stream,
        src,
        dst,
    })
}

/// Where a block argument's data handle comes from at execute time.
#[derive(Debug, Clone, Copy)]
enum ArgSource {
    /// The executor's input window at this operand index.
    Input(usize),
    /// The executor's output window at this operand index.
    Output(usize),
    /// This block's dense source staging (a row block).
    StagingSource,
    /// This block's dense destination staging (a row block).
    StagingDestination,
    /// The block's source, in the engine's channels-last layout, at a byte
    /// offset inside `Block::blocked_source`.
    BlockedSource(usize),
    /// The block's destination, in the engine's channels-last layout, at a byte
    /// offset inside `Block::blocked_destination`.
    BlockedDestination(usize),
    /// The block's prepacked weights.
    Weights,
    /// The block's bias slice.
    Bias,
    /// The block's scratchpad.
    Scratchpad,
}

impl Block {
    fn run(
        &self,
        inputs: &[usize],
        outputs: &[usize],
        pool: Option<&StagingPool>,
    ) -> Result<(), Fn3Error> {
        // One slot per concurrently running block: a lease held for the whole
        // call, returned when the block finishes (including on error). A block
        // leases one when it needs staging rows, a scratchpad, or both.
        let needs_buffers = self.staging.is_some() || self.scratchpad_bytes > 0;
        let lease = match (needs_buffers, pool) {
            (false, _) => None,
            (true, Some(pool)) => Some(pool.acquire()),
            (true, None) => {
                return Err(segment_error(
                    "a block needs session buffers but has no pool".to_string(),
                ))
            }
        };
        let slot = lease.as_ref().map(StagingLease::slot);
        if let (Some(staging), Some(slot)) = (&self.staging, slot) {
            if let Some(source_buffer) = slot.source.as_ref() {
                let source = *inputs
                    .first()
                    .ok_or_else(|| segment_error("missing input window 0".to_string()))?;
                staging.copy_in(source, source_buffer)?;
            }
        }
        for (memory, source) in self.memories.iter().zip(self.sources.iter()) {
            let address = match *source {
                ArgSource::Input(index) => *inputs
                    .get(index)
                    .ok_or_else(|| segment_error(format!("missing input window {index}")))?,
                ArgSource::Output(index) => *outputs
                    .get(index)
                    .ok_or_else(|| segment_error(format!("missing output window {index}")))?,
                ArgSource::StagingSource => slot
                    .and_then(|slot| slot.source.as_ref())
                    .map(|buffer| buffer.as_ptr() as usize)
                    .ok_or_else(|| segment_error("a block has no source staging".to_string()))?,
                ArgSource::StagingDestination => slot
                    .and_then(|slot| slot.destination.as_ref())
                    .map(|buffer| buffer.as_ptr() as usize)
                    .ok_or_else(|| {
                        segment_error("a block has no destination staging".to_string())
                    })?,
                ArgSource::Weights => buffer_address(self.weights.as_deref())?,
                ArgSource::Bias => buffer_address(self.bias.as_ref())?,
                ArgSource::Scratchpad => {
                    let scratchpad =
                        slot.and_then(|slot| slot.scratchpad.as_ref())
                            .ok_or_else(|| {
                                segment_error("a block has no scratchpad slot".to_string())
                            })?;
                    if scratchpad.len() < self.scratchpad_bytes {
                        return Err(segment_error(format!(
                            "scratchpad slot is {} B but the primitive needs {} B",
                            scratchpad.len(),
                            self.scratchpad_bytes
                        )));
                    }
                    scratchpad.as_ptr() as usize
                }
                ArgSource::BlockedSource(offset) => buffer_address(self.blocked_source.as_deref())?
                    .checked_add(offset)
                    .ok_or_else(|| segment_error("a blocked source offset overflowed".into()))?,
                ArgSource::BlockedDestination(offset) => {
                    buffer_address(self.blocked_destination.as_deref())?
                        .checked_add(offset)
                        .ok_or_else(|| {
                            segment_error("a blocked destination offset overflowed".into())
                        })?
                }
            };
            if address == 0 {
                return Err(segment_error(
                    "an argument has a null data handle".to_string(),
                ));
            }
            // SAFETY: the address is either a live arena window the executor
            // holds for this call (`Input`/`Output`), the start of a buffer
            // owned by this block (`Weights`/`Bias`/`Scratchpad`/staging), or
            // the staging buffer's pointer. All outlive the
            // `dnnl_primitive_execute` call below; the descriptor they are
            // installed on requires at most `desc.size()` bytes, which the
            // prepared shape provides. oneDNN borrows the pointer only for the
            // duration of the execute call.
            unsafe { memory.set_data_handle(address as *mut u8) }?;
            debug_assert!(
                memory
                    .data_handle()
                    .is_ok_and(|handle| handle as usize == address),
                "oneDNN did not keep the data handle this call installed"
            );
        }
        self.primitive.execute(&self.stream, &self.args)?;
        // The sequential CPU stream completes inside `execute`; the wait is a
        // no-op today and the fence that keeps a future threaded runtime
        // correct.
        self.stream.wait()?;
        if let Some(epilogue) = &self.epilogue {
            let address = match epilogue.target {
                EpilogueTarget::Output => *outputs
                    .first()
                    .ok_or_else(|| segment_error("missing output window 0".to_string()))?,
                EpilogueTarget::StagingDestination => slot
                    .and_then(|slot| slot.destination.as_ref())
                    .map(|buffer| buffer.as_ptr() as usize)
                    .ok_or_else(|| {
                        segment_error("a fused epilogue has no destination staging".to_string())
                    })?,
                EpilogueTarget::BlockedDestination => {
                    buffer_address(self.blocked_destination.as_deref())?
                }
            };
            // SAFETY: see `FusedEpilogue::apply`; each target is a region this
            // block alone writes.
            unsafe { epilogue.apply(address) }?;
        }
        if let (Some(staging), Some(slot)) = (&self.staging, slot) {
            if let Some(destination_buffer) = slot.destination.as_ref() {
                let destination = *outputs
                    .first()
                    .ok_or_else(|| segment_error("missing output window 0".to_string()))?;
                staging.copy_out(destination, destination_buffer)?;
            }
        }
        Ok(())
    }
}

fn buffer_address(buffer: Option<&AlignedBuffer>) -> Result<usize, Fn3Error> {
    buffer
        .map(|buffer| buffer.as_ptr() as usize)
        .ok_or_else(|| Fn3Error::ProviderFailure {
            provider: BACKEND,
            call: "dnnl_memory_set_data_handle",
            status: ffi::STATUS_SUCCESS,
            detail: "a prepared block is missing a buffer it needs".to_string(),
        })
}

fn segment_error(detail: String) -> Fn3Error {
    Fn3Error::ProviderFailure {
        provider: BACKEND,
        call: "dnnl_primitive_execute",
        status: ffi::STATUS_SUCCESS,
        detail,
    }
}

/// The byte length of a dense array with these extents, checked.
fn staging_bytes(extents: &[usize], element_bytes: usize) -> Result<usize, Fn3Error> {
    let mut bytes = element_bytes;
    for (&extent, axis) in extents.iter().zip(0..) {
        bytes = bytes.checked_mul(extent).ok_or(Fn3Error::ShapeOverflow {
            context: "oneDNN row staging",
            axis: Some(axis),
            detail: format!("{extents:?} exceeds usize::MAX bytes"),
        })?;
    }
    Ok(bytes)
}

/// How a prepared convolution divided its output, for the session report.
///
/// Recorded at preparation time from `(shape, params, workers)` and never
/// re-decided by a forward. `ranges` are `(start, count)` in the split
/// dimension: output channels for [`ConvSplitAxis::Channels`], output rows for
/// [`ConvSplitAxis::Rows`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ConvSplitSummary {
    pub axis: ConvSplitAxis,
    pub ranges: Vec<(usize, usize)>,
    /// Input rows read across all blocks (a traffic proxy; equals blocks ×
    /// input height for a channel split).
    pub source_rows: usize,
    /// The split dimension's total extent in the parent output.
    pub out_extent: usize,
}

/// One layout conversion a segment runs before its blocks.
///
/// The source is the plan slot (NCHW, the graph's contract) and the destination
/// is an engine-owned channels-last buffer the blocks then read. It runs once
/// per forward on the executor's thread, before any block starts.
#[derive(Debug)]
struct SegmentReorder {
    primitive: PrimitiveHandle,
    stream: StreamHandle,
    /// The NCHW side, which always names the plan slot.
    plain: MemoryHandle,
    /// The channels-last side, which always names the engine buffer.
    blocked: MemoryHandle,
    /// Kept alive for the memories above and shared with every block.
    buffer: Arc<AlignedBuffer>,
    /// `true` for NCHW -> channels-last, `false` for the other direction.
    to_blocked: bool,
}

impl SegmentReorder {
    fn run(&self, slot_address: usize) -> Result<(), Fn3Error> {
        if slot_address == 0 {
            return Err(segment_error(
                "a layout reorder has no plan-slot window".to_string(),
            ));
        }
        // SAFETY: `slot_address` is the executor's live window for the plan
        // slot; the memory object borrows it only for this execute call. The
        // buffer is this segment's own engine allocation, kept alive by
        // `buffer` and sized to the blocked descriptor.
        unsafe { self.plain.set_data_handle(slot_address as *mut u8) }?;
        // SAFETY: as above for the engine-owned channels-last buffer.
        unsafe {
            self.blocked
                .set_data_handle(self.buffer.as_ptr() as *mut u8)
        }?;
        let args = if self.to_blocked {
            [
                ExecArg {
                    arg: ARG_SRC,
                    memory: self.plain.raw(),
                },
                ExecArg {
                    arg: ARG_DST,
                    memory: self.blocked.raw(),
                },
            ]
        } else {
            [
                ExecArg {
                    arg: ARG_SRC,
                    memory: self.blocked.raw(),
                },
                ExecArg {
                    arg: ARG_DST,
                    memory: self.plain.raw(),
                },
            ]
        };
        self.primitive.execute(&self.stream, &args)?;
        self.stream.wait()
    }
}

/// A segment's prepared blocks.
#[derive(Debug)]
pub(crate) struct PreparedSegment {
    /// `"conv"`, `"pool"` or `"matmul"`, for diagnostics.
    kind: &'static str,
    /// The convolution split, when this is a convolution segment.
    split: Option<ConvSplitSummary>,
    blocks: Vec<Block>,
    /// A channels-last conversion of the segment's source, when the plan slot
    /// still holds NCHW (the head of a channels-last chain).
    prelude: Option<Arc<SegmentReorder>>,
    /// `true` when the result stays in the engine's channels-last layout.
    blocked: bool,
    /// Shared staging slots for this segment's row blocks, `None` when it has
    /// none. `Arc` because the pool is session-wide: one allocation serves
    /// every staged segment.
    staging_pool: Option<Arc<StagingPool>>,
}

impl PreparedSegment {
    /// The number of blocks the planner split this segment into.
    pub(crate) fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Run the segment's layout prelude, if it has one.
    ///
    /// Called once per forward on the executor's thread, before any block. The
    /// conversion is counted in the plan's reorder report.
    pub(crate) fn run_prelude(&self, inputs: &[usize]) -> Result<(), Fn3Error> {
        let Some(prelude) = self.prelude.as_ref() else {
            return Ok(());
        };
        let source = *inputs
            .first()
            .ok_or_else(|| segment_error("a reordered segment has no source window".to_string()))?;
        prelude.run(source)
    }

    /// Execute one block.
    ///
    /// `inputs` and `outputs` are the addresses of the segment's arena windows
    /// (zero for a weight operand, which the block does not read from the
    /// window). Each block reads only inputs[0] and writes only outputs[0] for
    /// the three claimed op families; `ArgSource` decides which.
    pub(crate) fn run_block(
        &self,
        index: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) -> Result<(), Fn3Error> {
        let block = self.blocks.get(index).ok_or(Fn3Error::OutOfBounds {
            what: "oneDNN block index",
            index: index as u64,
            len: self.blocks.len() as u64,
        })?;
        block.run(inputs, outputs, self.staging_pool.as_deref())
    }
}

/// One layout conversion, as the plan report records it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReorderRecord {
    /// Plan position of the segment that runs the conversion.
    pub segment: usize,
    /// Plan slot the conversion reads or writes.
    pub slot: usize,
    /// `true` for NCHW -> channels-last, `false` for the other direction.
    pub to_blocked: bool,
    /// Bytes moved per forward.
    pub bytes: usize,
}

/// The session-owned oneDNN engine.
#[derive(Debug)]
pub(crate) struct OneDnnEngine {
    /// The one CPU engine every primitive is created on.
    engine: EngineHandle,
    /// The stream used for prepack reorders at session open.
    prep_stream: StreamHandle,
    /// The library version actually linked.
    version: Version,
    /// `prepared[i]` is the prepared segment for plan segment `i`.
    prepared: Vec<Option<PreparedSegment>>,
    /// Bytes of engine-owned prepacked weights and biases.
    prepacked_bytes: usize,
    /// Bytes of dense source/destination staging owned by row blocks.
    staging_bytes: usize,
    /// Bytes of user scratchpad pooled across blocks.
    scratchpad_bytes: usize,
    /// Prepared segments per kind, for the session report.
    conv_segments: usize,
    matmul_segments: usize,
    /// Weight blocks whose layout oneDNN did not choose as plain on the ONNX
    /// payload, i.e. blocks that needed a real reorder (not just a copy).
    reordered_blocks: usize,
    /// Segments prepared with a fused epilogue (a `sum` and/or an activation).
    fused_segments: usize,
    /// Channels-last buffers keyed by plan slot: the value whose bytes they
    /// stand for, in the engine's layout. A buffer exists only while one
    /// segment produces and another consumes the blocked form, so a slot the
    /// session reads or writes stays NCHW.
    layout: Vec<Option<Arc<AlignedBuffer>>>,
    /// Layout conversions inside segments, in plan order.
    reorders: Vec<ReorderRecord>,
    /// Bytes those conversions move per forward.
    reorder_bytes: usize,
    /// Bytes of engine-owned channels-last buffers.
    layout_bytes: usize,
    /// Segments whose result stays channels-last.
    blocked_segments: usize,
}

// SAFETY: `OneDnnEngine` is `Send` because a session may be moved between
// threads (one forward at a time; `Session` is deliberately `!Sync`). Its
// fields are oneDNN handles in the sequential runtime, which have no thread
// affinity, plus `Vec`/`AlignedBuffer` data. `PreparedSegment` is additionally
// `Sync` because the executor shares `&PreparedSegment` across its pool while
// executing disjoint blocks; see `Block`'s impls.
unsafe impl Send for OneDnnEngine {}

// SAFETY: a `Block` is shared by reference across Rayon tasks. Each block owns
// its primitive, stream, memories, scratchpad and row-staging buffers, all of
// which are distinct from every other block's; `run` only writes through
// `&self` to oneDNN objects that are not shared, to arena windows the plan
// proved disjoint, and to its own staging buffers (which no other block names).
// A block's prepacked weights may be an `Arc` shared with the other row blocks
// of the same convolution, but `AlignedBuffer` exposes only immutable views and
// oneDNN only reads a weight argument. The raw pointers inside the wrapper
// types are never mutated by `&self` Rust code.
unsafe impl Sync for Block {}
unsafe impl Send for Block {}

impl OneDnnEngine {
    /// Create the engine, its prep stream and its (empty) prepared table.
    pub(crate) fn new(segments: usize) -> Result<OneDnnEngine, Fn3Error> {
        let version = ffi::runtime_version()?;
        if version.major != 3 {
            return Err(Fn3Error::ProviderFailure {
                provider: BACKEND,
                call: "dnnl_version",
                status: ffi::STATUS_SUCCESS,
                detail: format!(
                    "the linked oneDNN is {}, but this engine's hand-written ABI is for 3.x",
                    version.number()
                ),
            });
        }
        let engine = EngineHandle::cpu()?;
        let prep_stream = StreamHandle::new(&engine)?;
        Ok(OneDnnEngine {
            engine,
            prep_stream,
            version,
            prepared: (0..segments).map(|_| None).collect(),
            prepacked_bytes: 0,
            staging_bytes: 0,
            scratchpad_bytes: 0,
            conv_segments: 0,
            matmul_segments: 0,
            reordered_blocks: 0,
            fused_segments: 0,
            layout: Vec::new(),
            reorders: Vec::new(),
            reorder_bytes: 0,
            layout_bytes: 0,
            blocked_segments: 0,
        })
    }

    /// The linked oneDNN version.
    pub(crate) fn version(&self) -> &Version {
        &self.version
    }

    /// Bytes of prepacked weights and biases this session owns.
    pub(crate) fn prepacked_bytes(&self) -> usize {
        self.prepacked_bytes
    }

    /// Bytes of dense staging owned by row blocks.
    pub(crate) fn staging_bytes(&self) -> usize {
        self.staging_bytes
    }

    /// Bytes of user scratchpad pooled across blocks.
    pub(crate) fn scratchpad_bytes(&self) -> usize {
        self.scratchpad_bytes
    }

    /// Prepared segments by kind: `(conv, matmul)`.
    pub(crate) fn segment_kinds(&self) -> (usize, usize) {
        (self.conv_segments, self.matmul_segments)
    }

    /// Weight blocks that needed a reorder rather than a plain copy.
    pub(crate) fn reordered_blocks(&self) -> usize {
        self.reordered_blocks
    }

    /// Segments prepared with a fused `sum` and/or activation epilogue.
    pub(crate) fn fused_segments(&self) -> usize {
        self.fused_segments
    }

    /// Segments whose result stays in the engine's channels-last layout.
    pub(crate) fn blocked_segments(&self) -> usize {
        self.blocked_segments
    }

    /// Bytes of engine-owned channels-last buffers.
    pub(crate) fn layout_bytes(&self) -> usize {
        self.layout_bytes
    }

    /// Layout conversions inside segments, in plan order.
    pub(crate) fn reorders(&self) -> &[ReorderRecord] {
        &self.reorders
    }

    /// Bytes those conversions move per forward.
    pub(crate) fn reorder_bytes(&self) -> usize {
        self.reorder_bytes
    }

    /// Install a channels-last buffer for a plan slot, growing the table.
    fn set_layout_slot(&mut self, slot: usize, buffer: Arc<AlignedBuffer>) {
        if self.layout.len() <= slot {
            self.layout.resize(slot + 1, None);
        }
        self.layout[slot] = Some(buffer);
    }

    /// `true` when blocks may run concurrently.
    ///
    /// Only a sequential oneDNN makes "the session's pool is the only
    /// parallelism" true. A build that reports OMP/TBB/threadpool executes its
    /// blocks serially and lets the library use its own threads; the results
    /// are identical, the structure is not.
    pub(crate) fn blocks_run_in_parallel(&self) -> bool {
        self.version.is_sequential()
    }

    /// Prepare one segment and store it at `index`.
    ///
    /// The segment must have been claimed for [`SegmentEngine::OneDnn`] by
    /// [`OneDnnCapability`]; preparation re-runs the same geometry checks, so a
    /// claim/prepare disagreement is a loud error rather than a mis-run.
    pub(crate) fn prepare_segment(
        &mut self,
        index: usize,
        spec: &SegmentSpec<'_>,
    ) -> Result<(), Fn3Error> {
        let inputs = spec.operands();
        let outputs = spec.results();
        let claim = match spec.fused {
            Some(fused) => OneDnnCapability.claim_fused(fused, &inputs, &outputs, spec.params),
            None => OneDnnCapability.claim(spec.op, &inputs, &outputs, spec.params),
        };
        match claim {
            Claim::Accepted => {}
            Claim::NotMine => {
                return Err(unsupported(
                    spec.op,
                    "preparation was asked for an op the engine does not implement",
                ))
            }
            Claim::Declined(reason) => {
                return Err(Fn3Error::ProviderFailure {
                    provider: BACKEND,
                    call: "prepare_segment",
                    status: ffi::STATUS_SUCCESS,
                    detail: format!(
                        "segment {index} was planned for oneDNN, but preparation declined it: \
                         {reason}"
                    ),
                })
            }
        }
        let prepared = match spec.op {
            OpId::Conv => self.prepare_conv(index, spec)?,
            OpId::MatMul => self.prepare_matmul(spec)?,
            other => {
                return Err(unsupported(
                    other,
                    "the oneDNN engine implements Conv and 2-D MatMul",
                ))
            }
        };
        match prepared.kind {
            "conv" => self.conv_segments += 1,
            _ => self.matmul_segments += 1,
        }
        if prepared.blocked {
            self.blocked_segments += 1;
        }
        if spec.fused.is_some() {
            self.fused_segments += 1;
        }
        let len = self.prepared.len();
        let slot = self.prepared.get_mut(index).ok_or(Fn3Error::OutOfBounds {
            what: "oneDNN prepared segment index",
            index: index as u64,
            len: len as u64,
        })?;
        *slot = Some(prepared);
        Ok(())
    }

    /// The prepared segment at a plan position, if one was built.
    pub(crate) fn prepared(&self, index: usize) -> Option<&PreparedSegment> {
        self.prepared.get(index).and_then(Option::as_ref)
    }

    /// Build the shared session buffers once every segment is prepared.
    ///
    /// The pool has one slot per worker, and never more than the largest
    /// block count in one segment (segments execute one at a time). Each slot
    /// holds the largest staged source/destination rows and the largest
    /// scratchpad any block needs, so this memory is O(workers x block) rather
    /// than O(blocks): 391-672 blocks owned 27-29 MB of scratchpad on the YOLO
    /// graphs before this pool existed (#115 follow-up).
    pub(crate) fn finish_buffers(&mut self, workers: usize) -> Result<(), Fn3Error> {
        let mut max_blocks = 0usize;
        let mut max_source = 0usize;
        let mut max_destination = 0usize;
        let mut max_scratchpad = 0usize;
        for prepared in self.prepared.iter().flatten() {
            max_blocks = max_blocks.max(prepared.blocks.len());
            for block in &prepared.blocks {
                if let Some(staging) = block.staging.as_ref() {
                    max_source = max_source.max(staging.source_bytes);
                    max_destination = max_destination.max(staging.destination_bytes);
                }
                max_scratchpad = max_scratchpad.max(block.scratchpad_bytes);
            }
        }
        if max_blocks == 0 || (max_source == 0 && max_destination == 0 && max_scratchpad == 0) {
            self.staging_bytes = 0;
            self.scratchpad_bytes = 0;
            return Ok(());
        }
        let slots = workers.max(1).min(max_blocks);
        let pool = Arc::new(StagingPool::new(
            slots,
            max_source,
            max_destination,
            max_scratchpad,
        )?);
        self.staging_bytes = pool.staging_bytes();
        self.scratchpad_bytes = pool.scratchpad_bytes();
        for prepared in self.prepared.iter_mut().flatten() {
            if prepared
                .blocks
                .iter()
                .any(|block| block.staging.is_some() || block.scratchpad_bytes > 0)
            {
                prepared.staging_pool = Some(Arc::clone(&pool));
            }
        }
        Ok(())
    }

    /// The convolution split recorded for plan segment `index`, if it is a
    /// prepared convolution.
    pub(crate) fn prepared_split(&self, index: usize) -> Option<&ConvSplitSummary> {
        self.prepared(index)
            .and_then(|segment| segment.split.as_ref())
    }

    fn prepare_conv(
        &mut self,
        index: usize,
        spec: &SegmentSpec<'_>,
    ) -> Result<PreparedSegment, Fn3Error> {
        let conv = Conv::new(
            &spec.inputs[0],
            &spec.inputs[1],
            &spec.outputs[0],
            spec.params,
        )?;
        let weights = spec.weights.ok_or_else(|| {
            unsupported(OpId::Conv, "preparation needs the constant weight payload")
        })?;
        let weight_parent_dims: Vec<i64> = if conv.group == 1 {
            vec![
                conv.out_channels as i64,
                conv.in_channels as i64,
                conv.kernel[0] as i64,
                conv.kernel[1] as i64,
            ]
        } else {
            vec![
                conv.group as i64,
                (conv.out_channels / conv.group) as i64,
                conv.in_per_group as i64,
                conv.kernel[0] as i64,
                conv.kernel[1] as i64,
            ]
        };
        let weight_tag = if conv.group == 1 {
            FORMAT_TAG_ABCD
        } else {
            FORMAT_TAG_ABCDE
        };
        let weight_parent = Desc::with_tag(&weight_parent_dims, weight_tag)?;
        // One aligned copy of the whole payload, so every submemory read (and
        // every reorder) sees a 64-byte-aligned base. Dropped after the blocks
        // exist; the prepacked buffers are what a forward uses.
        let expected_weight_bytes = conv.out_channels
            * conv.in_per_group
            * conv.kernel[0]
            * conv.kernel[1]
            * std::mem::size_of::<f32>();
        if weights.len() != expected_weight_bytes {
            return Err(Fn3Error::ProviderFailure {
                provider: BACKEND,
                call: "prepare Conv",
                status: ffi::STATUS_SUCCESS,
                detail: format!(
                    "weight payload is {} bytes, expected {expected_weight_bytes} for {}x{}x{}x{}",
                    weights.len(),
                    conv.out_channels,
                    conv.in_per_group,
                    conv.kernel[0],
                    conv.kernel[1]
                ),
            });
        }
        let staged_weights = AlignedBuffer::from_bytes(weights)?;

        let src_parent_dims = vec![
            conv.batch as i64,
            conv.in_channels as i64,
            conv.in_hw[0] as i64,
            conv.in_hw[1] as i64,
        ];
        let src_parent = Desc::with_tag(&src_parent_dims, FORMAT_TAG_ABCD)?;
        let dst_parent_dims = vec![
            conv.batch as i64,
            conv.out_channels as i64,
            conv.out_hw[0] as i64,
            conv.out_hw[1] as i64,
        ];
        let dst_parent = Desc::with_tag(&dst_parent_dims, FORMAT_TAG_ABCD)?;

        let bias_values = spec
            .bias
            .map(|bytes| {
                if bytes.len() != conv.out_channels * std::mem::size_of::<f32>() {
                    Err(Fn3Error::ProviderFailure {
                        provider: BACKEND,
                        call: "prepare Conv",
                        status: ffi::STATUS_SUCCESS,
                        detail: format!(
                            "bias payload is {} bytes, expected {} for {} channels",
                            bytes.len(),
                            conv.out_channels * 4,
                            conv.out_channels
                        ),
                    })
                } else {
                    Ok(bytes)
                }
            })
            .transpose()?;

        // ------------------------------------------------------------------
        // Layout decisions for this segment.
        //
        // The source is channels-last when the producing segment installed a
        // layout buffer for this slot. The destination is channels-last only
        // when the executor proved every consumer is a oneDNN convolution,
        // which also means the result must be split by rows: a row range of a
        // channels-last tensor is one contiguous run, a channel range is not,
        // and oneDNN's channel-submemory form took its JIT path off the table
        // (measured: 233 ms for a 4-block 32x80x80 3x3 convolution, against
        // 1.1 ms for the same four blocks as dense channels-last buffers).
        // One dense block is a valid row range: the policy is live at one
        // worker, and that is worth 42.2 -> 36.6 ms on YOLOv8n there.
        // ------------------------------------------------------------------
        let src_slot = spec.input_slots.first().copied();
        let dst_slot = spec.output_slots.first().copied();
        let src_layout = src_slot
            .and_then(|slot| self.layout.get(slot))
            .and_then(Option::as_ref)
            .cloned();
        if src_layout.is_some() && conv.group != 1 {
            return Err(unsupported(
                OpId::Conv,
                "a grouped convolution cannot read a channels-last source: its per-group \
                 channel slice is a strided range of that layout",
            ));
        }
        let sum = spec.fused.is_some_and(FusedOp::has_sum);
        let plain_split = || {
            if sum {
                conv.channel_split(spec.workers)
            } else {
                conv.split(spec.workers)
            }
        };
        let mut split = plain_split();
        let mut dst_layout: Option<Arc<AlignedBuffer>> = None;
        if spec.dst_blocked && !sum && conv.batch == 1 {
            let rows = conv.row_split(spec.workers);
            let channels = conv.channel_split(spec.workers);
            if !rows.blocks.is_empty() && rows.blocks.len() >= channels.blocks.len() {
                split = rows;
                let bytes = conv.batch
                    * conv.out_channels
                    * conv.out_hw[0]
                    * conv.out_hw[1]
                    * std::mem::size_of::<f32>();
                let buffer = Arc::new(AlignedBuffer::zeroed(bytes)?);
                self.layout_bytes += bytes;
                let slot = dst_slot.ok_or_else(|| {
                    unsupported(OpId::Conv, "a channels-last destination needs a plan slot")
                })?;
                self.set_layout_slot(slot, Arc::clone(&buffer));
                dst_layout = Some(buffer);
            }
        }
        let row_split = split.axis == ConvSplitAxis::Rows;
        let summary = ConvSplitSummary {
            axis: split.axis,
            ranges: split
                .blocks
                .iter()
                .map(|block| {
                    if row_split {
                        (block.out_row_start, block.out_rows)
                    } else {
                        (block.out_start, block.out_channels)
                    }
                })
                .collect(),
            source_rows: split.blocks.iter().map(|block| block.in_rows).sum(),
            out_extent: if row_split {
                conv.out_hw[0]
            } else {
                conv.out_channels
            },
        };
        let blocks = split.blocks;

        // --------------------------------------------------------------
        // A segment never mixes layout families: oneDNN's kernels are fast on
        // `nchw -> nchw` and on `channels-last -> channels-last`, and the mixed
        // pair (`channels-last -> nchw`, measured on this host) made a 1-block
        // 64x40x40 3x3 convolution take 50 ms instead of 0.5 ms — the library
        // falls off its JIT path. So an edge between two families carries one
        // explicit conversion, run once before the blocks:
        //
        // - plain source, channels-last result: NCHW -> channels-last of the
        //   whole input slot (the head of a chain);
        // - channels-last source, plain result: channels-last -> NCHW of the
        //   whole source buffer (the tail of a chain, or a consumer that cannot
        //   take the blocked form).
        //
        // Both are counted in the plan report with their bytes.
        // --------------------------------------------------------------
        let bytes = conv.batch
            * conv.in_channels
            * conv.in_hw[0]
            * conv.in_hw[1]
            * std::mem::size_of::<f32>();
        let slot = src_slot
            .ok_or_else(|| unsupported(OpId::Conv, "a layout conversion needs a plan slot"))?;
        let (preload, blocked_source) = if let Some(buffer) = &src_layout {
            if dst_layout.is_some() {
                // Interior edge: the producer's channels-last buffer is read
                // directly.
                (None, Some(Arc::clone(buffer)))
            } else {
                // Tail edge: convert the producer's buffer back into the plan
                // slot, which the plain blocks then read.
                let plain = Desc::with_tag(&src_parent_dims, FORMAT_TAG_ABCD)?;
                let blocked = Desc::with_tag(&src_parent_dims, FORMAT_TAG_NHWC)?;
                let attr = PrimitiveAttrHandle::library_default()?;
                let pd = PrimitiveDescHandle::reorder(&self.engine, &blocked, &plain, &attr)?;
                let primitive = PrimitiveHandle::new(&pd)?;
                let single = Arc::new(SegmentReorder {
                    primitive,
                    stream: StreamHandle::new(&self.engine)?,
                    plain: MemoryHandle::new(&plain, &self.engine)?,
                    blocked: MemoryHandle::new(&blocked, &self.engine)?,
                    buffer: Arc::clone(buffer),
                    to_blocked: false,
                });
                self.reorder_bytes += bytes;
                self.reorders.push(ReorderRecord {
                    segment: index,
                    slot,
                    to_blocked: false,
                    bytes,
                });
                (Some(single), None)
            }
        } else if dst_layout.is_some() && row_split {
            // Head edge: convert the plan slot once, then read dense
            // channels-last rows. Eight per-block NCHW staging copies would
            // re-read the halo eight times.
            let buffer = Arc::new(AlignedBuffer::zeroed(bytes)?);
            let plain = Desc::with_tag(&src_parent_dims, FORMAT_TAG_ABCD)?;
            let blocked = Desc::with_tag(&src_parent_dims, FORMAT_TAG_NHWC)?;
            let attr = PrimitiveAttrHandle::library_default()?;
            let pd = PrimitiveDescHandle::reorder(&self.engine, &plain, &blocked, &attr)?;
            let primitive = PrimitiveHandle::new(&pd)?;
            self.layout_bytes += bytes;
            self.reorder_bytes += bytes;
            self.reorders.push(ReorderRecord {
                segment: index,
                slot,
                to_blocked: true,
                bytes,
            });
            let single = Arc::new(SegmentReorder {
                primitive,
                stream: StreamHandle::new(&self.engine)?,
                plain: MemoryHandle::new(&plain, &self.engine)?,
                blocked: MemoryHandle::new(&blocked, &self.engine)?,
                buffer: Arc::clone(&buffer),
                to_blocked: true,
            });
            (Some(single), Some(buffer))
        } else {
            (None, None)
        };
        let single_block = blocks.len() == 1;
        // Row blocks of one convolution use the same weights; prepack once and
        // share the buffer when oneDNN chose the same layout for every block.
        let mut shared_weights: Option<(Desc, Arc<AlignedBuffer>)> = None;
        let mut prepared = Vec::with_capacity(blocks.len());
        for block in blocks {
            let block_dims: Vec<i64> = if conv.group == 1 {
                vec![
                    block.out_channels as i64,
                    conv.in_channels as i64,
                    conv.kernel[0] as i64,
                    conv.kernel[1] as i64,
                ]
            } else {
                vec![
                    block.groups as i64,
                    (conv.out_channels / conv.group) as i64,
                    conv.in_per_group as i64,
                    conv.kernel[0] as i64,
                    conv.kernel[1] as i64,
                ]
            };
            let block_offsets: Vec<i64> = if conv.group == 1 {
                vec![block.out_start as i64, 0, 0, 0]
            } else {
                vec![block.group_start as i64, 0, 0, 0, 0]
            };
            // The submemory oneDNN reads the block's ONNX weights through.
            // Its element offset is `o0 * (C/group) * kh * kw` (plain) or
            // `g0 * (O/group) * (C/group) * kh * kw` (grouped); both are the
            // contiguous range the parent payload stores for the block.
            let weight_source = Desc::submemory(&weight_parent, &block_dims, &block_offsets)?;
            // What oneDNN may choose for the weights.
            let weight_any = Desc::with_tag(&block_dims, FORMAT_TAG_ANY)?;

            let src = if row_split {
                let dims = vec![
                    conv.batch as i64,
                    conv.in_channels as i64,
                    block.in_rows as i64,
                    conv.in_hw[1] as i64,
                ];
                if blocked_source.is_some() {
                    // Dense channels-last rows: `[N, in_rows, W, C]`.
                    Desc::with_tag(&dims, FORMAT_TAG_NHWC)?
                } else {
                    // Dense NCHW staging rows; a row submemory of the parent
                    // would keep strides oneDNN cannot JIT (see `RowStaging`).
                    Desc::with_tag(&dims, FORMAT_TAG_ABCD)?
                }
            } else if single_block || conv.group == 1 {
                if blocked_source.is_some() {
                    Desc::with_tag(&src_parent_dims, FORMAT_TAG_NHWC)?
                } else {
                    // The full input: no spatial split, no halo.
                    Desc::with_tag(&src_parent_dims, FORMAT_TAG_ABCD)?
                }
            } else {
                let dims = vec![
                    conv.batch as i64,
                    block.in_channels as i64,
                    conv.in_hw[0] as i64,
                    conv.in_hw[1] as i64,
                ];
                let offsets = vec![0, block.in_start as i64, 0, 0];
                Desc::submemory(&src_parent, &dims, &offsets)?
            };
            let dst = if row_split {
                let dims = vec![
                    conv.batch as i64,
                    conv.out_channels as i64,
                    block.out_rows as i64,
                    conv.out_hw[1] as i64,
                ];
                if dst_layout.is_some() {
                    Desc::with_tag(&dims, FORMAT_TAG_NHWC)?
                } else {
                    Desc::with_tag(&dims, FORMAT_TAG_ABCD)?
                }
            } else if single_block {
                Desc::with_tag(&dst_parent_dims, FORMAT_TAG_ABCD)?
            } else {
                let dims = vec![
                    conv.batch as i64,
                    block.out_channels as i64,
                    conv.out_hw[0] as i64,
                    conv.out_hw[1] as i64,
                ];
                let offsets = vec![0, block.out_start as i64, 0, 0];
                Desc::submemory(&dst_parent, &dims, &offsets)?
            };

            let bias_desc = bias_values
                .map(|_| Desc::with_tag(&[block.out_channels as i64], FORMAT_TAG_A))
                .transpose()?;
            let bias_buffer = bias_values
                .map(|bytes| {
                    let start = block.out_start * 4;
                    AlignedBuffer::from_bytes(&bytes[start..start + block.out_channels * 4])
                })
                .transpose()?;
            if let Some(buffer) = &bias_buffer {
                self.prepacked_bytes += buffer.len();
            }

            let attr = PrimitiveAttrHandle::user_scratchpad()?;
            if let Some(fused) = spec.fused {
                // oneDNN applies post-ops in append order, and its direct
                // convolution requires `sum` first: `dst = sum(dst_old,
                // conv(dst))` then the elementwise epilogue.
                let mut post_ops = PostOpsHandle::new()?;
                if fused.has_sum() {
                    // `scale = 1`, `zero_point = 0`, `data_type = undef` (the
                    // destination's own type).
                    post_ops.append_sum(1.0, 0, 0)?;
                }
                if EPILOGUE_POST_OP {
                    if let Some(activation) = fused.activation {
                        let (alg, alpha, beta) = eltwise(activation);
                        post_ops.append_eltwise(alg, alpha, beta)?;
                    }
                }
                attr.set_post_ops(&post_ops)?;
            }
            let strides: Vec<i64> = conv.strides.iter().map(|&value| value as i64).collect();
            // oneDNN's `dilates` array is "dilation minus one" (0 means no
            // dilation), unlike ONNX's dilation of 1.
            let dilates: Vec<i64> = conv
                .dilations
                .iter()
                .map(|&value| value as i64 - 1)
                .collect();
            let (pad_top, pad_bottom) = if row_split && !single_block {
                (block.pad_top as i64, block.pad_bottom as i64)
            } else {
                (conv.pads[0].0, conv.pads[0].1)
            };
            let padding_l = vec![pad_top, conv.pads[1].0];
            let padding_r = vec![pad_bottom, conv.pads[1].1];
            let pd =
                PrimitiveDescHandle::convolution(
                    &self.engine,
                    ffi::ALG_CONVOLUTION_DIRECT,
                    &src,
                    &weight_any,
                    bias_desc.as_ref(),
                    &dst,
                    &strides,
                    &dilates,
                    &padding_l,
                    &padding_r,
                    &attr,
                )
                .map_err(|error| {
                    context(error, || {
                        format!(
                        "conv segment: {}x{}, {} of {} output channels, groups {}, {}, source {}, \
                         destination {}",
                        conv.in_hw[0],
                        conv.in_hw[1],
                        block.out_channels,
                        conv.out_channels,
                        conv.group,
                        if row_split {
                            format!(
                                "rows {}..{} of {} (input rows {}..{})",
                                block.out_row_start,
                                block.out_row_start + block.out_rows,
                                conv.out_hw[0],
                                block.in_row_start,
                                block.in_row_start + block.in_rows
                            )
                        } else {
                            format!("channels from {}", block.out_start)
                        },
                        if blocked_source.is_some() { "channels-last" } else { "nchw" },
                        if dst_layout.is_some() { "channels-last" } else { "nchw" },
                    )
                    })
                })?;
            let expected_weights = pd.query_md(QUERY_WEIGHTS_MD, 0)?;
            let weight_plain = Desc::with_tag(&block_dims, weight_tag)?;
            let (prepacked, reformatted) = match shared_weights.as_ref() {
                Some((descriptor, buffer)) if row_split && descriptor.equal(&expected_weights) => {
                    (Arc::clone(buffer), false)
                }
                _ => {
                    let (buffer, reformatted) = self.repack(
                        &staged_weights,
                        &weight_source,
                        &expected_weights,
                        &weight_plain,
                    )?;
                    let buffer = Arc::new(buffer);
                    if row_split {
                        shared_weights = Some((expected_weights.duplicate()?, Arc::clone(&buffer)));
                    }
                    (buffer, reformatted)
                }
            };
            if reformatted {
                self.reordered_blocks += 1;
            }
            let scratchpad = prepare_scratchpad(&pd)?;
            let primitive = PrimitiveHandle::new(&pd)?;

            let staging = if !row_split {
                None
            } else {
                let source_bytes = if blocked_source.is_none() {
                    staging_bytes(
                        &[conv.batch, conv.in_channels, block.in_rows, conv.in_hw[1]],
                        std::mem::size_of::<f32>(),
                    )?
                } else {
                    0
                };
                let destination_bytes = if dst_layout.is_none() {
                    staging_bytes(
                        &[
                            conv.batch,
                            conv.out_channels,
                            block.out_rows,
                            conv.out_hw[1],
                        ],
                        std::mem::size_of::<f32>(),
                    )?
                } else {
                    0
                };
                if source_bytes + destination_bytes == 0 {
                    None
                } else {
                    Some(RowStaging {
                        source_bytes,
                        destination_bytes,
                        source_len: staging_bytes(
                            &[conv.batch, conv.in_channels, conv.in_hw[0], conv.in_hw[1]],
                            std::mem::size_of::<f32>(),
                        )?,
                        destination_len: staging_bytes(
                            &[
                                conv.batch,
                                conv.out_channels,
                                conv.out_hw[0],
                                conv.out_hw[1],
                            ],
                            std::mem::size_of::<f32>(),
                        )?,
                        batch: conv.batch,
                        in_channels: conv.in_channels,
                        out_channels: conv.out_channels,
                        in_rows: block.in_rows,
                        out_rows: block.out_rows,
                        in_row_start: block.in_row_start,
                        out_row_start: block.out_row_start,
                        in_h: conv.in_hw[0],
                        out_h: conv.out_hw[0],
                        in_row_bytes: conv.in_hw[1] * std::mem::size_of::<f32>(),
                        out_row_bytes: conv.out_hw[1] * std::mem::size_of::<f32>(),
                    })
                }
            };

            // The fused activation runs over this block's destination after the
            // primitive. The destination is either the dense staging buffer (a
            // row block with an NCHW result), this block's row range of the
            // channels-last buffer, or the block's channel range of the plan
            // slot.
            let epilogue = spec
                .fused
                .and_then(|fused| fused.activation)
                .map(|activation| {
                    if dst_layout.is_some() {
                        let run_len = block.out_rows * conv.out_hw[1] * conv.out_channels;
                        FusedEpilogue::new(
                            activation,
                            conv.batch,
                            run_len,
                            run_len,
                            block.out_row_start
                                * conv.out_hw[1]
                                * conv.out_channels
                                * std::mem::size_of::<f32>(),
                            EpilogueTarget::BlockedDestination,
                            &self.engine,
                        )
                    } else if row_split {
                        let run_len = block.out_rows * conv.out_hw[1] * conv.out_channels;
                        FusedEpilogue::new(
                            activation,
                            conv.batch,
                            run_len,
                            run_len,
                            0,
                            EpilogueTarget::StagingDestination,
                            &self.engine,
                        )
                    } else {
                        let run_len = block.out_channels * conv.out_hw[0] * conv.out_hw[1];
                        FusedEpilogue::new(
                            activation,
                            conv.batch,
                            run_len,
                            conv.out_channels * conv.out_hw[0] * conv.out_hw[1],
                            block.out_start
                                * conv.out_hw[0]
                                * conv.out_hw[1]
                                * std::mem::size_of::<f32>(),
                            EpilogueTarget::Output,
                            &self.engine,
                        )
                    }
                });
            let blocked_destination_bytes = block.out_row_start
                * conv.out_hw[1]
                * conv.out_channels
                * std::mem::size_of::<f32>();
            // A channels-last source is one dense buffer; a row block starts at
            // its first halo row, a channel block at the tensor's start.
            let blocked_src_arg = ArgSource::BlockedSource(
                block.in_row_start * conv.in_hw[1] * conv.in_channels * std::mem::size_of::<f32>(),
            );
            let blocked_dst_arg = ArgSource::BlockedDestination(blocked_destination_bytes);
            let mut memories = Vec::with_capacity(5);
            let mut args = Vec::with_capacity(5);
            let mut sources = Vec::with_capacity(5);
            push_arg(
                &mut memories,
                &mut args,
                &mut sources,
                &src,
                &self.engine,
                ARG_SRC,
                if blocked_source.is_some() {
                    blocked_src_arg
                } else if row_split {
                    ArgSource::StagingSource
                } else {
                    ArgSource::Input(0)
                },
            )?;
            push_arg(
                &mut memories,
                &mut args,
                &mut sources,
                &expected_weights,
                &self.engine,
                ARG_WEIGHTS,
                ArgSource::Weights,
            )?;
            if let Some(desc) = &bias_desc {
                push_arg(
                    &mut memories,
                    &mut args,
                    &mut sources,
                    desc,
                    &self.engine,
                    ARG_BIAS,
                    ArgSource::Bias,
                )?;
            }
            push_arg(
                &mut memories,
                &mut args,
                &mut sources,
                &dst,
                &self.engine,
                ARG_DST,
                if dst_layout.is_some() {
                    blocked_dst_arg
                } else if row_split {
                    ArgSource::StagingDestination
                } else {
                    ArgSource::Output(0)
                },
            )?;
            if let Some(desc) = scratchpad.as_ref().map(|(desc, _)| desc) {
                push_arg(
                    &mut memories,
                    &mut args,
                    &mut sources,
                    desc,
                    &self.engine,
                    ARG_SCRATCHPAD,
                    ArgSource::Scratchpad,
                )?;
            }
            prepared.push(Block {
                primitive,
                stream: StreamHandle::new(&self.engine)?,
                memories,
                args,
                sources,
                weights: Some(prepacked),
                bias: bias_buffer,
                scratchpad_bytes: scratchpad.map(|(_, bytes)| bytes).unwrap_or(0),
                staging,
                epilogue,
                blocked_source: blocked_source.clone(),
                blocked_destination: dst_layout.clone(),
            });
        }
        Ok(PreparedSegment {
            kind: "conv",
            split: Some(summary),
            blocks: prepared,
            prelude: preload,
            blocked: dst_layout.is_some(),
            staging_pool: None,
        })
    }

    fn prepare_matmul(&mut self, spec: &SegmentSpec<'_>) -> Result<PreparedSegment, Fn3Error> {
        let matmul = MatMul::new(&spec.inputs[0], &spec.inputs[1], &spec.outputs[0])?;
        let weights = spec.weights.ok_or_else(|| {
            unsupported(OpId::MatMul, "preparation needs the constant right operand")
        })?;
        let weight_parent_dims = vec![matmul.k as i64, matmul.n as i64];
        let weight_parent = Desc::with_tag(&weight_parent_dims, ffi::FORMAT_TAG_AB)?;
        let expected_weight_bytes = matmul.k * matmul.n * std::mem::size_of::<f32>();
        if weights.len() != expected_weight_bytes {
            return Err(Fn3Error::ProviderFailure {
                provider: BACKEND,
                call: "prepare MatMul",
                status: ffi::STATUS_SUCCESS,
                detail: format!(
                    "right operand is {} bytes, expected {expected_weight_bytes} for {}x{}",
                    weights.len(),
                    matmul.k,
                    matmul.n
                ),
            });
        }
        let staged_weights = AlignedBuffer::from_bytes(weights)?;
        let src_dims = vec![matmul.m as i64, matmul.k as i64];
        let src = Desc::with_tag(&src_dims, ffi::FORMAT_TAG_AB)?;
        let dst_parent_dims = vec![matmul.m as i64, matmul.n as i64];
        let dst_parent = Desc::with_tag(&dst_parent_dims, ffi::FORMAT_TAG_AB)?;
        let blocks = matmul.blocks(spec.workers);
        let single_block = blocks.len() == 1;
        let mut prepared = Vec::with_capacity(blocks.len());
        for (column_start, columns) in blocks {
            let weight_dims = vec![matmul.k as i64, columns as i64];
            let weight_offsets = vec![0i64, column_start as i64];
            let weight_source = Desc::submemory(&weight_parent, &weight_dims, &weight_offsets)?;
            let weight_any = Desc::with_tag(&weight_dims, ffi::FORMAT_TAG_AB)?;
            let dst = if single_block {
                Desc::with_tag(&dst_parent_dims, ffi::FORMAT_TAG_AB)?
            } else {
                let dims = vec![matmul.m as i64, columns as i64];
                let offsets = vec![0i64, column_start as i64];
                Desc::submemory(&dst_parent, &dims, &offsets)?
            };
            let attr = PrimitiveAttrHandle::user_scratchpad()?;
            let pd =
                PrimitiveDescHandle::matmul(&self.engine, &src, &weight_any, None, &dst, &attr)
                    .map_err(|error| {
                        context(error, || {
                            format!(
                                "matmul segment: {}x{} times {}x{}, columns {}..{}",
                                matmul.m,
                                matmul.k,
                                matmul.k,
                                matmul.n,
                                column_start,
                                column_start + columns
                            )
                        })
                    })?;
            let expected_weights = pd.query_md(QUERY_WEIGHTS_MD, 0)?;
            let weight_plain = Desc::with_tag(&weight_dims, ffi::FORMAT_TAG_AB)?;
            let (prepacked, reformatted) = self.repack(
                &staged_weights,
                &weight_source,
                &expected_weights,
                &weight_plain,
            )?;
            if reformatted {
                self.reordered_blocks += 1;
            }
            let scratchpad = prepare_scratchpad(&pd)?;
            let primitive = PrimitiveHandle::new(&pd)?;
            let mut memories = Vec::with_capacity(4);
            let mut args = Vec::with_capacity(4);
            let mut sources = Vec::with_capacity(4);
            push_arg(
                &mut memories,
                &mut args,
                &mut sources,
                &src,
                &self.engine,
                ARG_SRC,
                ArgSource::Input(0),
            )?;
            push_arg(
                &mut memories,
                &mut args,
                &mut sources,
                &expected_weights,
                &self.engine,
                ARG_WEIGHTS,
                ArgSource::Weights,
            )?;
            push_arg(
                &mut memories,
                &mut args,
                &mut sources,
                &dst,
                &self.engine,
                ARG_DST,
                ArgSource::Output(0),
            )?;
            if let Some(desc) = scratchpad.as_ref().map(|(desc, _)| desc) {
                push_arg(
                    &mut memories,
                    &mut args,
                    &mut sources,
                    desc,
                    &self.engine,
                    ARG_SCRATCHPAD,
                    ArgSource::Scratchpad,
                )?;
            }
            prepared.push(Block {
                primitive,
                stream: StreamHandle::new(&self.engine)?,
                memories,
                args,
                sources,
                weights: Some(Arc::new(prepacked)),
                bias: None,
                scratchpad_bytes: scratchpad.map(|(_, bytes)| bytes).unwrap_or(0),
                staging: None,
                epilogue: None,
                blocked_source: None,
                blocked_destination: None,
            });
        }
        Ok(PreparedSegment {
            kind: "matmul",
            split: None,
            blocks: prepared,
            prelude: None,
            blocked: false,
            staging_pool: None,
        })
    }

    /// Reorder one block's weights into an engine-owned buffer.
    ///
    /// `staged` is the aligned copy of the whole payload, `source` the block's
    /// submemory of it, and `expected` the descriptor the primitive wants. The
    /// reorder runs once here, on the prep stream, and never again.
    fn repack(
        &mut self,
        staged: &AlignedBuffer,
        source: &Desc,
        expected: &Desc,
        plain: &Desc,
    ) -> Result<(AlignedBuffer, bool), Fn3Error> {
        let reformatted = !expected.equal(plain);
        let bytes = expected.size();
        let destination = AlignedBuffer::zeroed(bytes)?;
        // Account for the engine-owned copy here, where it is created: a
        // prepacked buffer shared by several row blocks is one allocation and
        // must be counted once.
        self.prepacked_bytes += bytes;
        // A reorder has no scratchpad and runs on this thread; the library's
        // default scratchpad policy is the right one here.
        let attr = PrimitiveAttrHandle::library_default()?;
        let pd = PrimitiveDescHandle::reorder(&self.engine, source, expected, &attr)?;
        let primitive = PrimitiveHandle::new(&pd)?;
        let src_memory = MemoryHandle::new(source, &self.engine)?;
        let dst_memory = MemoryHandle::new(expected, &self.engine)?;
        // SAFETY: both handles are 64-byte-aligned buffers owned by the caller
        // and this function; each is at least as large as the descriptor
        // requires (`staged` holds the whole payload the submemory addresses,
        // `destination` was allocated with `expected.size()`). Both outlive the
        // execute and wait calls below.
        unsafe { src_memory.set_data_handle(staged.as_ptr() as *mut u8) }?;
        // SAFETY: as above for `destination`.
        unsafe { dst_memory.set_data_handle(destination.as_ptr() as *mut u8) }?;
        let args = [
            ExecArg {
                arg: ARG_SRC,
                memory: src_memory.raw(),
            },
            ExecArg {
                arg: ARG_DST,
                memory: dst_memory.raw(),
            },
        ];
        primitive.execute(&self.prep_stream, &args)?;
        self.prep_stream.wait()?;
        Ok((destination, reformatted))
    }
}

/// Evaluate a fused activation as a oneDNN eltwise post-op instead of the
/// engine's own scalar epilogue.
///
/// Measured and rejected on this host: see [`FusedEpilogue`] for the numbers.
/// The switch is kept, rather than the code deleted, so the comparison can be
/// reproduced on another CPU or another oneDNN build where post-ops do reach
/// the JIT injector.
pub(crate) const EPILOGUE_POST_OP: bool = false;

/// The oneDNN eltwise triple for a fused activation.
fn eltwise(activation: Activation) -> (i32, f32, f32) {
    match activation {
        // `alpha = 0` is the plain `max(0, x)`.
        Activation::Relu => (ALG_ELTWISE_RELU, 0.0, 0.0),
        Activation::Clip { low, high } => (ALG_ELTWISE_CLIP, low, high),
        Activation::Sigmoid => (ALG_ELTWISE_LOGISTIC, 0.0, 0.0),
        Activation::Tanh => (ALG_ELTWISE_TANH, 0.0, 0.0),
        Activation::Gelu => (ALG_ELTWISE_GELU_ERF, 0.0, 0.0),
        // oneDNN's swish is `x / (1 + exp(-alpha * x))`; alpha = 1 is SiLU.
        Activation::Swish => (ALG_ELTWISE_SWISH, 1.0, 0.0),
    }
}

/// The scratchpad descriptor and its byte size, when the primitive needs one.
///
/// The buffer is not allocated here: the engine pools one slot per worker and
/// leases it at run time, because a per-block buffer cost 27-29 MB on the YOLO
/// graphs (391-672 blocks) and the blocks of one segment run at most
/// `workers`-at-a-time.
fn prepare_scratchpad(pd: &PrimitiveDescHandle) -> Result<Option<(Desc, usize)>, Fn3Error> {
    match pd.scratchpad()? {
        Some(desc) => {
            let bytes = desc.size();
            Ok(Some((desc, bytes)))
        }
        None => Ok(None),
    }
}

fn push_arg(
    memories: &mut Vec<MemoryHandle>,
    args: &mut Vec<ExecArg>,
    sources: &mut Vec<ArgSource>,
    desc: &Desc,
    engine: &EngineHandle,
    arg: i32,
    source: ArgSource,
) -> Result<(), Fn3Error> {
    let memory = MemoryHandle::new(desc, engine)?;
    args.push(ExecArg {
        arg,
        memory: memory.raw(),
    });
    memories.push(memory);
    sources.push(source);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn operand(dims: &[u64], constant: bool) -> SegmentOperand {
        SegmentOperand {
            shape: Shape::from_dims(dims.iter().copied()),
            dtype: DType::F32,
            constant,
        }
    }

    #[test]
    fn the_staging_pool_reuses_a_slot_after_the_block_finishes() {
        let pool = StagingPool::new(2, 128, 64, 0).unwrap();
        assert_eq!(pool.staging_bytes(), 2 * (128 + 64));
        let first = pool.acquire();
        let second = pool.acquire();
        // Both slots are leased; the two leases address different buffers.
        assert_ne!(
            first.slot().source.as_ref().unwrap().as_ptr(),
            second.slot().source.as_ref().unwrap().as_ptr()
        );
        let first_index = first.index;
        drop(first);
        let third = pool.acquire();
        assert_eq!(third.index, first_index, "the freed slot is reused");
        drop(second);
        drop(third);
        assert_eq!(
            pool.staging_bytes(),
            384,
            "a lease never changes the pool size"
        );
    }

    #[test]
    fn a_pool_with_no_destination_only_allocates_sources() {
        let pool = StagingPool::new(1, 0, 64, 0).unwrap();
        assert!(pool.slots[0].source.is_none());
        assert_eq!(pool.staging_bytes(), 64);
    }

    #[test]
    fn the_scratchpad_pool_serves_one_slot_per_worker() {
        let pool = StagingPool::new(2, 0, 0, 4096).unwrap();
        assert_eq!(pool.staging_bytes(), 0);
        assert_eq!(pool.scratchpad_bytes(), 2 * 4096);
        let first = pool.acquire();
        let second = pool.acquire();
        assert_ne!(
            first.slot().scratchpad.as_ref().unwrap().as_ptr(),
            second.slot().scratchpad.as_ref().unwrap().as_ptr(),
            "concurrent blocks must not share one scratchpad"
        );
        let first_index = first.index;
        drop(first);
        let third = pool.acquire();
        assert_eq!(third.index, first_index, "the freed slot is reused");
    }

    #[test]
    fn the_claim_names_what_it_cannot_take() {
        let params = OpParams::new()
            .with_kernel_shape([3, 3])
            .with_strides([1, 1])
            .with_pads([1, 1, 1, 1]);
        let inputs = [
            operand(&[1, 8, 16, 16], false),
            operand(&[8, 8, 3, 3], true),
        ];
        let outputs = [operand(&[1, 8, 16, 16], false)];
        assert!(claimed_geometry(OpId::Conv, &inputs, &outputs, &params).is_ok());
        // A dynamic weight is rejected by name.
        let mut dynamic = inputs.clone();
        dynamic[1] = operand(&[8, 8, 3, 3], false);
        let error = claimed_geometry(OpId::Conv, &dynamic, &outputs, &params).unwrap_err();
        assert!(error.to_string().contains("prepacks constant"), "{error}");
        // An op outside the vocabulary is NotMine, not a fallback report.
        assert_eq!(
            OneDnnCapability.claim(OpId::Relu, &inputs[..1], &outputs, &params),
            Claim::NotMine
        );
    }

    #[test]
    fn the_capability_reports_the_onednn_engine() {
        let capability = OneDnnCapability;
        assert_eq!(capability.engine(), SegmentEngine::OneDnn);
    }

    #[test]
    fn the_linked_library_is_sequential() {
        let engine = OneDnnEngine::new(0).unwrap();
        let version = engine.version();
        println!(
            "[fn3onednn][measure] oneDNN {} (hash {:?}), cpu runtime {}, parallel blocks {}",
            version.number(),
            version.hash,
            version.cpu_runtime_name(),
            engine.blocks_run_in_parallel()
        );
        assert_eq!(version.major, 3, "the hand-written ABI is for oneDNN 3.x");
        assert!(
            version.is_sequential(),
            "the oneDNN build is {}; this engine requires the sequential CPU runtime so the \
             session's pool is the only parallelism (see tools/cmake/onednn-seq.cmake)",
            version.cpu_runtime_name()
        );
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(values).to_vec()
    }

    fn run_one(spec: &SegmentSpec<'_>, input: &[f32], outputs: usize) -> Vec<f32> {
        let mut engine = OneDnnEngine::new(1).unwrap();
        engine.prepare_segment(0, spec).unwrap();
        engine.finish_buffers(1).unwrap();
        let prepared = engine.prepared(0).unwrap();
        assert_eq!(prepared.block_count(), 1);
        let input_buffer = AlignedBuffer::from_bytes(bytemuck::cast_slice(input)).unwrap();
        let output_buffer = AlignedBuffer::zeroed(outputs * std::mem::size_of::<f32>()).unwrap();
        prepared
            .run_block(
                0,
                &[input_buffer.as_ptr() as usize],
                &[output_buffer.as_ptr() as usize],
            )
            .unwrap();
        bytemuck::cast_slice::<u8, f32>(output_buffer.as_bytes()).to_vec()
    }

    fn close(got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len());
        for (index, (&got, &expected)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - expected).abs() <= 1e-6 + 1e-6 * expected.abs(),
                "element {index}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn a_grouped_dilated_strided_convolution_with_bias_runs_on_onednn() {
        // Two groups, one output channel each, a 2x2 kernel dilated to 3x3 on a
        // 3x3 input: exactly one output element per group.
        let params = OpParams::new()
            .with_kernel_shape([2, 2])
            .with_dilations([2, 2])
            .with_group(2);
        let weights = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
        let bias = f32_bytes(&[0.5, -0.5]);
        let input: Vec<f32> = (1..=18).map(|value| value as f32).collect();
        let spec = SegmentSpec {
            op: OpId::Conv,
            inputs: vec![
                Shape::from_dims([1, 2, 3, 3]),
                Shape::from_dims([2, 1, 2, 2]),
                Shape::from_dims([2]),
            ],
            input_dtypes: vec![DType::F32, DType::F32, DType::F32],
            input_constants: vec![false, true, true],
            outputs: vec![Shape::from_dims([1, 2, 1, 1])],
            output_dtypes: vec![DType::F32],
            params: &params,
            weights: Some(&weights),
            bias: Some(&bias),
            workers: 1,
            fused: None,
            input_slots: (0..3).collect(),
            output_slots: vec![3],
            dst_blocked: false,
        };
        let got = run_one(&spec, &input, 2);
        // Group 0 sees 1, 3, 7, 9 -> 64; group 1 sees 10, 12, 16, 18 -> 154.
        close(&got, &[64.5, 153.5]);
    }

    #[test]
    fn a_two_dimensional_matmul_runs_on_onednn() {
        let params = OpParams::new();
        let weights = f32_bytes(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let spec = SegmentSpec {
            op: OpId::MatMul,
            inputs: vec![Shape::from_dims([2, 3]), Shape::from_dims([3, 2])],
            input_dtypes: vec![DType::F32, DType::F32],
            input_constants: vec![false, true],
            outputs: vec![Shape::from_dims([2, 2])],
            output_dtypes: vec![DType::F32],
            params: &params,
            weights: Some(&weights),
            bias: None,
            workers: 1,
            fused: None,
            input_slots: (0..3).collect(),
            output_slots: vec![3],
            dst_blocked: false,
        };
        let got = run_one(&spec, &input, 4);
        close(&got, &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn a_fused_claim_names_what_it_cannot_take() {
        let params = OpParams::new()
            .with_kernel_shape([1, 1])
            .with_strides([1, 1])
            .with_dilations([1, 1]);
        let inputs = [
            operand(&[1, 8, 16, 16], false),
            operand(&[8, 8, 1, 1], true),
            operand(&[1, 8, 16, 16], false),
        ];
        let outputs = [operand(&[1, 8, 16, 16], false)];
        let fused = FusedOp {
            anchor: OpId::Conv,
            activation: Some(Activation::Relu),
            sum: Some(2),
            chain: vec![OpId::Conv, OpId::Add, OpId::Relu],
        };
        assert!(claimed_fused_conv(&fused, &inputs, &outputs, &params).is_ok());
        // A residual that is not the last operand is a planner bug, reported.
        let wrong = FusedOp {
            sum: Some(1),
            ..fused.clone()
        };
        assert!(claimed_fused_conv(&wrong, &inputs, &outputs, &params).is_err());
        // A broadcast residual is not expressible as a oneDNN sum.
        let mut broadcast = inputs.clone();
        broadcast[2] = operand(&[1, 1, 16, 16], false);
        let error = claimed_fused_conv(&fused, &broadcast, &outputs, &params).unwrap_err();
        assert!(error.to_string().contains("broadcast"), "{error}");
        // A non-conv anchor is outside the vocabulary, not a fallback.
        let relu = FusedOp {
            anchor: OpId::Relu,
            activation: Some(Activation::Relu),
            sum: None,
            chain: vec![OpId::Relu, OpId::Relu],
        };
        assert_eq!(
            OneDnnCapability.claim_fused(&relu, &inputs[..1], &outputs, &params),
            Claim::NotMine
        );
    }

    /// A fused `conv + sum + relu` is one primitive: the destination is
    /// pre-filled with the residual (oneDNN's sum post-op reads it back), then
    /// the convolution adds into it and the engine applies the activation.
    #[test]
    fn a_fused_sum_and_activation_matches_the_three_step_chain() {
        let input: Vec<f32> = (0..(4 * 4 * 4))
            .map(|index| (index as f32) * 0.125 - 2.0)
            .collect();
        let residual: Vec<f32> = (0..(2 * 4 * 4))
            .map(|index| ((index as f32) * 0.25).cos())
            .collect();
        let weights = f32_bytes(&[1.0, -0.5, 0.25, 2.0, 0.75, -1.5, 0.5, 0.25]);
        let params = OpParams::new()
            .with_kernel_shape([1, 1])
            .with_strides([1, 1])
            .with_dilations([1, 1])
            .with_group(1);
        let fused = FusedOp {
            anchor: OpId::Conv,
            activation: Some(Activation::Relu),
            sum: Some(2),
            chain: vec![OpId::Conv, OpId::Add, OpId::Relu],
        };
        let spec = SegmentSpec {
            op: OpId::Conv,
            inputs: vec![
                Shape::from_dims([1, 4, 4, 4]),
                Shape::from_dims([2, 4, 1, 1]),
                Shape::from_dims([1, 2, 4, 4]),
            ],
            input_dtypes: vec![DType::F32, DType::F32, DType::F32],
            input_constants: vec![false, true, false],
            outputs: vec![Shape::from_dims([1, 2, 4, 4])],
            output_dtypes: vec![DType::F32],
            params: &params,
            weights: Some(&weights),
            bias: None,
            workers: 1,
            fused: Some(&fused),
            input_slots: vec![0, 1, 2],
            output_slots: vec![3],
            dst_blocked: false,
        };
        let mut engine = OneDnnEngine::new(1).unwrap();
        engine.prepare_segment(0, &spec).unwrap();
        let prepared = engine.prepared(0).unwrap();
        assert_eq!(engine.fused_segments(), 1);
        let input_buffer = AlignedBuffer::from_bytes(bytemuck::cast_slice(&input)).unwrap();
        // The executor pre-fills the result slot with the residual; the same
        // buffer is then the sum post-op's accumulator.
        let output_buffer = AlignedBuffer::from_bytes(bytemuck::cast_slice(&residual)).unwrap();
        prepared
            .run_block(
                0,
                &[
                    input_buffer.as_ptr() as usize,
                    0,
                    residual.as_ptr() as usize,
                ],
                &[output_buffer.as_ptr() as usize],
            )
            .unwrap();
        // The unfused chain: 1x1 convolution, add the residual, relu.
        let mut expected = vec![0.0f32; residual.len()];
        for channel in 0..2 {
            for row in 0..4 {
                for column in 0..4 {
                    let mut sum = 0.0f32;
                    for inner in 0..4 {
                        let x = input[(inner * 4 + row) * 4 + column];
                        sum += x * f32::from_le_bytes([
                            weights[(channel * 4 + inner) * 4],
                            weights[(channel * 4 + inner) * 4 + 1],
                            weights[(channel * 4 + inner) * 4 + 2],
                            weights[(channel * 4 + inner) * 4 + 3],
                        ]);
                    }
                    let position = (channel * 4 + row) * 4 + column;
                    let value = sum + residual[position];
                    expected[position] = if value > 0.0 { value } else { 0.0 };
                }
            }
        }
        let got = bytemuck::cast_slice::<u8, f32>(output_buffer.as_bytes()).to_vec();
        close(&got, &expected);
    }

    #[test]
    fn a_memory_object_starts_empty_and_takes_the_installed_handle() {
        let engine = EngineHandle::cpu().unwrap();
        let desc = Desc::with_tag(&[4], FORMAT_TAG_A).unwrap();
        let memory = MemoryHandle::new(&desc, &engine).unwrap();
        assert!(
            memory.data_handle().unwrap().is_null(),
            "a block must not cache a data handle between forwards"
        );
        let buffer = AlignedBuffer::zeroed(16).unwrap();
        // SAFETY: `buffer` is a 64-byte-aligned, 16-byte allocation, exactly the
        // descriptor's `4 * f32`, and it outlives the memory object.
        unsafe { memory.set_data_handle(buffer.as_ptr() as *mut u8) }.unwrap();
        assert_eq!(
            memory.data_handle().unwrap() as usize,
            buffer.as_ptr() as usize
        );
    }

    /// The standalone eltwise is a different kernel from the post-op path; it
    /// must agree with the scalar expression the fused chain uses.
    #[test]
    fn the_standalone_eltwise_agrees_with_the_scalar_activation() {
        let engine = EngineHandle::cpu().unwrap();
        let values: Vec<f32> = (0..801)
            .map(|index| (index as f32 - 400.0) * 0.05)
            .collect();
        for activation in [
            Activation::Swish,
            Activation::Sigmoid,
            Activation::Tanh,
            Activation::Gelu,
        ] {
            let mut got = values.clone();
            let epilogue = FusedEpilogue::new(
                activation,
                1,
                values.len(),
                values.len(),
                0,
                EpilogueTarget::Output,
                &engine,
            );
            assert!(
                epilogue.onednn.is_some(),
                "{activation} should use the standalone eltwise"
            );
            // SAFETY: `got` is a live, writable buffer of `values.len()` f32 and
            // nothing else writes it; the epilogue reads and writes exactly
            // that range.
            unsafe { epilogue.apply(got.as_mut_ptr() as usize) }.unwrap();
            for (index, (input, actual)) in values.iter().zip(got.iter()).enumerate() {
                let expected = activation.apply(*input);
                let tolerance = 2e-5 + 2e-5 * expected.abs();
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{activation} at {index} (input {input}): oneDNN {actual}, scalar {expected}"
                );
            }
        }
    }
}
