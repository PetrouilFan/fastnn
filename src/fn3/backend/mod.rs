//! Backend contract: capabilities, support reporting, and op dispatch.
//!
//! # Support is a promise, not a hint
//!
//! [`Backend::supports`] answers exactly one question: *can this backend run
//! this op at all?* The answer is three-valued:
//!
//! | [`Support`] | Meaning | Caller may run it? |
//! |---|---|---|
//! | `Full` | every documented dtype/shape combination is implemented | yes |
//! | `Partial { reason }` | a named subset is implemented; other shapes are rejected by the op | yes, for the subset |
//! | `Unsupported { reason }` | nothing is implemented | no |
//!
//! `Partial` never authorises an approximation. A backend that cannot compute a
//! request returns [`Fn3Error::UnsupportedOp`] naming the op and the issue that
//! owns the missing case. The 2.x behaviour of "unknown operation returns its
//! first input" is explicitly forbidden here; see
//! `docs/roadmap/fn3-architecture.md`.
//!
//! # Two questions, not one
//!
//! A plan is built for one concrete shape environment, so op-level support is
//! necessary but not sufficient. [`Backend::admits`] adds the concrete operands
//! (shapes and element types) and defaults to [`Backend::supports`]. A `Partial`
//! verdict whose subset is defined in terms of element width, axis or operand
//! count can therefore answer for the plan's actual tensors without widening
//! the op-level claim. An `Unsupported` verdict stays unsupported.
//!
//! # Extension seam for Vulkan
//!
//! FN3-11 (#57) adds the `vulkan` feature, declares the module below, and
//! implements [`Backend`] for it. The core contract deliberately does not
//! declare the feature: an empty feature would advertise a device that cannot
//! allocate or execute.
//!
//! ```ignore
//! #[cfg(feature = "vulkan")]
//! pub mod vulkan;
//! ```

use std::fmt;

use crate::inference::device::DeviceId;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

pub mod cpu;

#[cfg(feature = "vulkan")]
pub mod vulkan;

/// What a backend can execute.
///
/// A backend reports these once at construction; plans validate against them
/// before allocating or dispatching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// Dtypes the backend can move and compute on.
    pub dtypes: &'static [DType],
    /// Largest single buffer the backend will allocate, in bytes.
    pub max_buffer_bytes: u64,
    /// Largest supported tensor rank.
    pub max_shape_rank: u8,
    /// Workgroup geometry limit `[x, y, z]`. The CPU backend reports `[1, 1,
    /// 1]` because it has no workgroup model; its parallelism is
    /// [`Capabilities::max_threads`].
    pub max_workgroup_size: [u32; 3],
    /// Worker threads (CPU) or invocations per workgroup (GPU).
    pub max_threads: u32,
    /// `true` if the backend executes explicitly strided views without
    /// requiring a contiguous copy.
    pub supports_strided_views: bool,
    /// Optional feature flags.
    pub features: CapabilityFeatures,
}

/// Optional backend features. Each flag is a promise about a specific op or
/// layout that the core may otherwise have to decompose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CapabilityFeatures {
    /// Right-aligned broadcasting is implemented for elementwise ops.
    pub broadcasting: bool,
    /// The fused `MaskedAttention` op is implemented (FN3-15 / #61).
    pub masked_attention: bool,
    /// Quantized dtypes/ops are implemented (post-alpha).
    pub quantized: bool,
    /// The backend keeps weights resident between steps.
    pub persistent_weights: bool,
}

impl Capabilities {
    /// `true` if `dtype` is listed.
    pub fn supports_dtype(&self, dtype: DType) -> bool {
        self.dtypes.contains(&dtype)
    }

    /// Require `dtype` to be listed, naming the backend in the error context.
    pub fn check_dtype(&self, op: Option<OpId>, dtype: DType) -> Result<(), Fn3Error> {
        if self.supports_dtype(dtype) {
            Ok(())
        } else {
            Err(Fn3Error::UnsupportedDType {
                op,
                dtype,
                context: "this backend's capability table",
            })
        }
    }

    /// Require a buffer to fit the backend's per-buffer limit.
    pub fn check_buffer(&self, bytes: u64) -> Result<(), Fn3Error> {
        if bytes <= self.max_buffer_bytes {
            Ok(())
        } else {
            Err(Fn3Error::CapacityExceeded {
                what: "backend max_buffer_bytes",
                requested: bytes,
                capacity: self.max_buffer_bytes,
            })
        }
    }

    /// Require a rank to fit the backend's rank limit.
    pub fn check_rank(&self, rank: usize) -> Result<(), Fn3Error> {
        if rank <= self.max_shape_rank as usize {
            Ok(())
        } else {
            Err(Fn3Error::InvalidShape {
                context: "backend max_shape_rank",
                detail: format!(
                    "rank {rank} exceeds the backend limit of {}",
                    self.max_shape_rank
                ),
            })
        }
    }
}

/// Three-valued answer to "can this backend run this op?".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Support {
    /// Every documented shape/dtype combination is implemented.
    Full,
    /// A named subset is implemented. Requests outside the subset must be
    /// rejected by the op with [`Fn3Error::UnsupportedOp`].
    Partial {
        /// What is supported, and what is not.
        reason: String,
    },
    /// Nothing is implemented.
    Unsupported {
        /// Why, and which issue owns the implementation.
        reason: String,
    },
}

impl Support {
    /// Fully supported.
    pub fn full() -> Support {
        Support::Full
    }

    /// Partially supported; `reason` names the implemented and missing subsets.
    pub fn partial(reason: impl Into<String>) -> Support {
        Support::Partial {
            reason: reason.into(),
        }
    }

    /// Not supported; `reason` names the owner of the missing work.
    pub fn unsupported(reason: impl Into<String>) -> Support {
        Support::Unsupported {
            reason: reason.into(),
        }
    }

    /// `true` only for [`Support::Full`].
    pub fn is_full(&self) -> bool {
        matches!(self, Support::Full)
    }

    /// `true` for `Full` and `Partial`.
    pub fn is_supported(&self) -> bool {
        !matches!(self, Support::Unsupported { .. })
    }

    /// The explanation for a non-`Full` answer.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Support::Full => None,
            Support::Partial { reason } | Support::Unsupported { reason } => Some(reason),
        }
    }

    /// Reject `Unsupported`, allow `Full` and `Partial`.
    ///
    /// This is the check [`Backend::execute_op`] implementations run before
    /// looking at shapes. A `Partial` answer still obliges the op to reject the
    /// shapes it does not implement.
    pub fn require_supported(&self, op: OpId, backend: &'static str) -> Result<(), Fn3Error> {
        match self {
            Support::Full | Support::Partial { .. } => Ok(()),
            Support::Unsupported { reason } => Err(Fn3Error::UnsupportedOp {
                op,
                backend,
                reason: reason.clone(),
            }),
        }
    }

    /// Require `Full`. Use this in plan preflight when a partially supported op
    /// would silently select a different execution path.
    pub fn require_full(&self, op: OpId, backend: &'static str) -> Result<(), Fn3Error> {
        match self {
            Support::Full => Ok(()),
            Support::Partial { reason } => Err(Fn3Error::UnsupportedOp {
                op,
                backend,
                reason: format!("only partially supported: {reason}"),
            }),
            Support::Unsupported { reason } => Err(Fn3Error::UnsupportedOp {
                op,
                backend,
                reason: reason.clone(),
            }),
        }
    }
}

impl fmt::Display for Support {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Support::Full => f.write_str("full"),
            Support::Partial { reason } => write!(f, "partial ({reason})"),
            Support::Unsupported { reason } => write!(f, "unsupported ({reason})"),
        }
    }
}

/// Attributes of one op invocation.
///
/// This is the alpha attribute carrier. It only holds the attributes the alpha
/// CPU reference needs; FN3-10 (#56) replaces it with op-specific typed
/// payloads when the execution-plan format is defined. Adding a field here is
/// cheap, but an op must reject a field it does not understand rather than
/// ignore it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OpParams {
    /// Axes for transpose/reduce/softmax/concat/split/unsqueeze. Negative
    /// values count from the end of the rank.
    pub axes: Vec<i64>,
    /// Keep reduced axes as extent 1.
    pub keep_dims: bool,
    /// Numerical floor for normalization layers.
    pub epsilon: f32,
}

impl OpParams {
    /// Empty attributes.
    pub fn new() -> OpParams {
        OpParams::default()
    }

    /// Builder: set axes.
    pub fn with_axes(mut self, axes: impl IntoIterator<Item = i64>) -> OpParams {
        self.axes = axes.into_iter().collect();
        self
    }

    /// Builder: set `keep_dims`.
    pub fn with_keep_dims(mut self, keep_dims: bool) -> OpParams {
        self.keep_dims = keep_dims;
        self
    }

    /// Builder: set `epsilon`.
    pub fn with_epsilon(mut self, epsilon: f32) -> OpParams {
        self.epsilon = epsilon;
        self
    }
}

/// An execution backend bound to one device.
///
/// Implementations must be thread-safe (`Send + Sync`): a `PreparedModel`
/// shares one backend handle between sessions and every session may run on its
/// own thread. Per-step mutable state belongs to the session, never to the
/// backend.
pub trait Backend: Send + Sync + fmt::Debug {
    /// Concrete Vulkan backend, for session-owned packed projections.
    #[cfg(feature = "vulkan")]
    #[doc(hidden)]
    fn as_vulkan(&self) -> Option<&vulkan::VulkanBackend> {
        None
    }

    /// Stable backend name used in diagnostics (`"cpu"`, `"vulkan"`).
    fn name(&self) -> &'static str;

    /// The device this backend executes on.
    fn device(&self) -> DeviceId;

    /// Capabilities, fixed for the lifetime of the backend.
    fn capabilities(&self) -> &Capabilities;

    /// Whether `op` is implemented, partially implemented or unsupported.
    fn supports(&self, op: OpId) -> Support;

    /// Whether this backend can run **this concrete instruction**.
    ///
    /// [`Backend::supports`] answers the op-level question: *is there a kernel
    /// for this op at all?* A plan is a stronger promise — it is built for one
    /// concrete `(seq, past)` environment — so plan preflight asks here as
    /// well. `shapes` and `dtypes` are the instruction's operands in plan
    /// order (inputs first, then outputs) and have the same length; `params`
    /// are its encoded attributes.
    ///
    /// The element types are part of the question because [`Shape`] carries
    /// extents only. A kernel that documents "4- and 8-byte elements" is
    /// `Partial` at the op level but can answer for a concrete `f32` operand,
    /// and the pinned decoder graph is exactly that case: its structural ops
    /// are `Partial`, its index tensors are `i64`, and neither fact is
    /// expressible without the concrete operands.
    ///
    /// The default body returns [`Backend::supports`], so a backend with
    /// nothing concrete to add keeps its previous verdicts exactly. An
    /// implementation may return [`Support::Full`] where `supports` returns
    /// [`Support::Partial`] only when every concrete operand lies inside the
    /// subset that reason names; it must never widen [`Support::Unsupported`].
    /// Returning `Partial` for a concrete case is always honest, and the op
    /// still rejects anything it cannot execute.
    fn admits(&self, op: OpId, shapes: &[Shape], dtypes: &[DType], params: &OpParams) -> Support {
        let _ = (shapes, dtypes, params);
        self.supports(op)
    }

    /// Execute one op into already-allocated outputs.
    ///
    /// Contract for implementations:
    ///
    /// 1. Call [`Support::require_supported`] before dispatching.
    /// 2. Validate that every input and output is on [`Backend::device`] and
    ///    report [`Fn3Error::DeviceMismatch`] otherwise.
    /// 3. Validate dtypes, ranks and shapes and report the offending values.
    /// 4. Never approximate an unsupported request.
    ///
    /// The implementation must write every output element it is responsible
    /// for, or return an error. Outputs are pre-allocated by the caller, so a
    /// failed call leaves them in an unspecified but valid state; plan-level
    /// atomicity is the caller's responsibility (see
    /// [`crate::inference::model::Session`]).
    fn execute_op(
        &self,
        op: OpId,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
        params: &OpParams,
    ) -> Result<(), Fn3Error>;

    /// `supports(op)` plus a rejection with the backend name filled in.
    fn preflight(&self, op: OpId) -> Result<(), Fn3Error> {
        self.supports(op).require_supported(op, self.name())
    }

    // -----------------------------------------------------------------------
    // Device residency seam (FN3-26 / #72)
    // -----------------------------------------------------------------------
    //
    // These four hooks are additive: every default body is the host-staged
    // behaviour, so a backend that does not implement device memory is
    // unchanged. They exist so the session layer can allocate device-resident
    // tensors, keep immutable model data on the device, and bracket a step with
    // one submission *without* naming Vulkan.

    /// `true` when the backend can allocate and compute on device-resident
    /// storage ([`crate::inference::storage::Storage::from_resident`]).
    ///
    /// Reporting `true` is a promise that
    /// [`Backend::allocate_resident`] succeeds for every dtype and shape this
    /// backend claims in [`Capabilities`], and that its kernels bind such a
    /// tensor without a host copy. A backend that answers `false` must be given
    /// host tensors; the caller keeps the staged path.
    fn supports_resident_storage(&self) -> bool {
        false
    }

    /// Allocate a zero-initialised device-resident tensor on [`Backend::device`].
    ///
    /// The returned tensor has device storage: no host bytes, an explicit
    /// download path ([`crate::inference::tensor::Tensor::to_host`]), and metadata
    /// identical to [`crate::inference::tensor::Tensor::zeros`]. The default body
    /// reports [`Fn3Error::NotImplemented`] naming this backend, so a caller
    /// that checked [`Backend::supports_resident_storage`] never reaches it.
    fn allocate_resident(
        &self,
        shape: &Shape,
        dtype: DType,
    ) -> Result<crate::inference::tensor::Tensor, Fn3Error> {
        let _ = (shape, dtype);
        Err(Fn3Error::NotImplemented {
            what: "device-resident allocation on this backend (it has no device allocator); \
                   allocate host tensors and use the staged path",
            owner: "FN3-26 (#72)",
        })
    }

    /// Upload immutable tensors once and bind them by handle thereafter.
    ///
    /// The default body is a no-op: a host-staged backend re-reads the bytes on
    /// every dispatch, which is exactly the behaviour it already had. A backend
    /// that keeps persistent device copies must be idempotent here (the same
    /// tensor may be presented by several sessions) and must not modify the
    /// tensor: the host storage stays valid for the CPU/fallback path. A copy
    /// is keyed by [`crate::inference::Storage::generation`] and released when
    /// the last handle to that storage drops; it is never keyed by a host
    /// address, which the allocator may reuse for a different model.
    fn prepare_persistent(
        &self,
        tensors: &[&crate::inference::tensor::Tensor],
    ) -> Result<(), Fn3Error> {
        let _ = tensors;
        Ok(())
    }

    /// Open a submission scope for one plan execution.
    ///
    /// A backend that batches records every dispatch between `begin_step` and
    /// [`Backend::end_step`] into one submission. The default bodies are
    /// no-ops, so per-op backends (the CPU reference) are unaffected. Scopes do
    /// not nest within one thread; `label` is a diagnostic ("prefill", "decode")
    /// and `batch` is the session's decision: it is `true` only when the whole
    /// step's operands and results stay on the device, because a deferred
    /// submission cannot satisfy a host read.
    fn begin_step(&self, label: &'static str, batch: bool) -> Result<(), Fn3Error> {
        let _ = (label, batch);
        Ok(())
    }

    /// Write host bytes into a device-resident storage at `byte_offset`.
    ///
    /// This is the session-boundary input path: token ids, positions and the
    /// attention mask are computed on the host every step. The default body
    /// reports [`Fn3Error::NotImplemented`] — a backend with no device memory
    /// is given host tensors and never calls it.
    fn write_resident(
        &self,
        dst: &crate::inference::storage::Storage,
        byte_offset: u64,
        bytes: &[u8],
    ) -> Result<(), Fn3Error> {
        let _ = (dst, byte_offset, bytes);
        Err(Fn3Error::NotImplemented {
            what: "writing a device-resident tensor on this backend (it has no device allocator)",
            owner: "FN3-26 (#72)",
        })
    }

    /// Copy `regions` of `(dst_offset, src_offset, len)` bytes between two
    /// device-resident storages.
    ///
    /// One call records one transfer for the whole region list, so a strided
    /// layout (the KV cache's `[batch, heads, capacity, head_dim]` live prefix)
    /// costs one submission, not one per row. The default body reports
    /// [`Fn3Error::NotImplemented`]; a host-staged session never calls it.
    fn copy_resident_regions(
        &self,
        dst: &crate::inference::storage::Storage,
        src: &crate::inference::storage::Storage,
        regions: &[(u64, u64, u64)],
    ) -> Result<(), Fn3Error> {
        let _ = (dst, src, regions);
        Err(Fn3Error::NotImplemented {
            what: "device-to-device copies on this backend (it has no device allocator)",
            owner: "FN3-26 (#72)",
        })
    }

    /// Record a device → host readback for the residency counters.
    ///
    /// The default body is a no-op: a host backend has no counters and no
    /// device traffic to attribute.
    fn note_readback(&self, bytes: u64) {
        let _ = bytes;
    }

    /// Close the scope opened by [`Backend::begin_step`] and submit it once.
    ///
    /// Returns the number of queue submissions this call made (0 for a backend
    /// that submits per op). On failure the scope is still closed and every
    /// resource it owns is released: a step whose recording failed is
    /// discarded, never submitted as a partial plan.
    fn end_step(&self) -> Result<u32, Fn3Error> {
        Ok(0)
    }

    /// A **terminal** backend failure, when the backend can no longer be used.
    ///
    /// The default body is `None`: a host backend has no device state that can
    /// be permanently lost, so every failure it reports is recoverable.
    ///
    /// A backend that returns an error here is poisoned: a session must not
    /// execute another step, because the state of an in-flight fence or
    /// command buffer is unknown. The caller's only recovery is to drop the
    /// backend and open a new device. [`crate::inference::model::Session::recover`]
    /// returns this error instead of pretending the session can continue.
    fn terminal_error(&self) -> Option<Fn3Error> {
        None
    }
}

/// Check an op's input and output count.
pub fn check_arity(
    op: OpId,
    inputs: usize,
    outputs: usize,
    expected_inputs: usize,
    expected_outputs: usize,
) -> Result<(), Fn3Error> {
    if inputs == expected_inputs && outputs == expected_outputs {
        Ok(())
    } else {
        Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs,
            actual_inputs: inputs,
            expected_outputs,
            actual_outputs: outputs,
        })
    }
}
