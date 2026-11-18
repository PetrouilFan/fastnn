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

use crate::fn3::device::DeviceId;
use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;
use crate::fn3::tensor::{TensorMut, TensorView};

pub mod cpu;

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
    /// Stable backend name used in diagnostics (`"cpu"`, `"vulkan"`).
    fn name(&self) -> &'static str;

    /// The device this backend executes on.
    fn device(&self) -> DeviceId;

    /// Capabilities, fixed for the lifetime of the backend.
    fn capabilities(&self) -> &Capabilities;

    /// Whether `op` is implemented, partially implemented or unsupported.
    fn supports(&self, op: OpId) -> Support;

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
    /// [`crate::fn3::model::Session`]).
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
