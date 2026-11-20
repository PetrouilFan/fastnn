//! The single error type of the 3.0 inference core.
//!
//! # Actionability rule
//!
//! Every variant carries enough context to answer three questions without a
//! debugger: *which* operation or reader failed, *what* it expected, and *what*
//! it found. Messages name the op (`op add: ...`), quote shapes as `[2, 3]`,
//! name devices as `cpu` / `vulkan(0)`, and state the bound that a dynamic
//! dimension violated.
//!
//! # One type, no panics
//!
//! Contract code never panics on caller input: overflow, out-of-bounds,
//! unsupported ops and unimplemented skeleton entry points all return
//! `Fn3Error`. `Fn3Error` is `Send + Sync` so it can cross thread boundaries.

use std::fmt;

use crate::inference::device::{DeviceId, DeviceResolveError};
use crate::inference::dtype::{DType, UnknownDTypeCode};
use crate::inference::ops::OpId;

/// Convenience alias used throughout the 3.0 core.
pub type Fn3Result<T> = Result<T, Fn3Error>;

/// A contract violation or execution failure in the 3.0 inference core.
///
/// # Rust source compatibility
///
/// Out-of-memory errors with no known request size now use
/// [`Fn3Error::AllocationSizeUnknown`] instead of `Allocation { bytes: 0, .. }`.
/// Exhaustive matches must handle the new variant. The fields of
/// [`Fn3Error::Allocation`] are unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fn3Error {
    /// A checked element-count, byte-length or stride computation overflowed.
    ShapeOverflow {
        /// The function or operation whose arithmetic overflowed.
        context: &'static str,
        /// Axis index, when the overflow happened on one axis.
        axis: Option<usize>,
        /// What was multiplied by what.
        detail: String,
    },
    /// A shape is structurally invalid (rank, permutation, envelope, ...).
    InvalidShape {
        /// The validating function.
        context: &'static str,
        /// What was wrong, including the offending value.
        detail: String,
    },
    /// An operation received a tensor of the wrong rank.
    RankMismatch {
        /// The operation, when one is in scope.
        op: Option<OpId>,
        /// Required rank.
        expected: usize,
        /// Actual rank.
        actual: usize,
    },
    /// An operation received a tensor whose shape is incompatible.
    ///
    /// Shapes are rendered (`"[2, 3]"`) rather than stored so call sites can
    /// describe the *rule* that was violated as well as the operand.
    ShapeMismatch {
        /// The operation, when one is in scope.
        op: Option<OpId>,
        /// The required shape or shape rule.
        expected: String,
        /// The offending shape.
        actual: String,
    },
    /// An operation received a tensor of the wrong dtype.
    DTypeMismatch {
        /// The operation, when one is in scope.
        op: Option<OpId>,
        /// Required dtype.
        expected: DType,
        /// Actual dtype.
        actual: DType,
    },
    /// The dtype is known but not supported for this operation.
    ///
    /// This is the "reject, do not coerce" path: an `I64` operand handed to an
    /// `f32` kernel produces this error instead of an implicit cast.
    UnsupportedDType {
        /// The operation, when one is in scope.
        op: Option<OpId>,
        /// The rejected dtype.
        dtype: DType,
        /// Kernel or reader that rejected it.
        context: &'static str,
    },
    /// A serialized payload carried a dtype code this build does not define.
    UnknownDTypeCode(UnknownDTypeCode),
    /// A view declared a stride that the alpha view contract rejects.
    InvalidStride {
        /// Axis whose stride is invalid.
        axis: usize,
        /// The rejected stride, in elements.
        stride: i64,
        /// Why it was rejected.
        detail: String,
    },
    /// An access needs a contiguous view, but the view is legitimately strided.
    ///
    /// This is not an invalid view: it is a valid transpose or slice that must
    /// be materialised (for example with
    /// [`crate::inference::tensor::Tensor::to_contiguous`]) before this particular
    /// access. The detail names the view and its strides.
    NonContiguous {
        /// The operation that requires a contiguous view.
        context: &'static str,
        /// The rejected view and its strides.
        detail: String,
    },
    /// Operands of one operation are on different devices.
    DeviceMismatch {
        /// The operation, when one is in scope.
        op: Option<OpId>,
        /// The device required by the operation or the first operand.
        expected: DeviceId,
        /// The device actually found.
        actual: DeviceId,
    },
    /// An explicitly requested device cannot be used by this build.
    DeviceUnavailable(DeviceResolveError),
    /// The backend does not implement this operation.
    ///
    /// Backends must return this instead of approximating (for example
    /// returning the first input). See `docs/roadmap/fn3-architecture.md`.
    UnsupportedOp {
        /// The rejected operation.
        op: OpId,
        /// Backend that rejected it.
        backend: &'static str,
        /// Why, and which issue implements it.
        reason: String,
    },
    /// An operation was called with the wrong number of inputs or outputs.
    ArityMismatch {
        /// The operation.
        op: OpId,
        /// Required input count.
        expected_inputs: usize,
        /// Actual input count.
        actual_inputs: usize,
        /// Required output count.
        expected_outputs: usize,
        /// Actual output count.
        actual_outputs: usize,
    },
    /// A dynamic dimension request falls outside the declared envelope.
    DimOutOfEnvelope {
        /// Symbolic axis name (`"seq"`, `"total"`, ...).
        symbol: &'static str,
        /// Requested extent.
        requested: u64,
        /// Inclusive lower bound.
        min: u64,
        /// Inclusive upper bound.
        max: u64,
    },
    /// A bounded resource would be exceeded.
    CapacityExceeded {
        /// What is bounded (`"kv cache"`, `"plan value slots"`, ...).
        what: &'static str,
        /// Requested amount.
        requested: u64,
        /// Allowed amount.
        capacity: u64,
    },
    /// An index or range is outside the addressed object.
    OutOfBounds {
        /// What was addressed.
        what: &'static str,
        /// The offending index.
        index: u64,
        /// The length of the addressed object.
        len: u64,
    },
    /// A byte buffer cannot be viewed as the requested element type.
    StorageCast {
        /// Element type requested.
        element: &'static str,
        /// Buffer length in bytes.
        byte_len: usize,
        /// Alignment/size/length detail.
        detail: String,
    },
    /// Mutation was requested on storage that another handle can still read.
    StorageShared {
        /// Buffer length in bytes.
        bytes: usize,
        /// Number of live handles.
        handles: usize,
    },
    /// A byte window or sub-range is outside its storage or misaligned for its
    /// dtype.
    ///
    /// Carried by [`crate::inference::storage::Storage::window`] and the constructors
    /// that consume a window. The window is always checked with
    /// `offset.checked_add(len)`, so a wrapping request is reported here rather
    /// than silently becoming an in-bounds request.
    StorageWindow {
        /// The constructor or method being validated.
        what: &'static str,
        /// Requested byte offset.
        offset: u64,
        /// Requested byte length.
        len: u64,
        /// Length of the storage that was addressed.
        storage_len: u64,
        /// Whether the failure was bounds or alignment, with the values.
        detail: String,
    },
    /// An allocation request of a known size failed.
    Allocation {
        /// Byte count requested by the failing operation, not bytes successfully
        /// allocated. This can differ from the logical tensor size (for example,
        /// a Vulkan chunk allocation requests the full chunk). Zero is a known
        /// zero-byte request, never a sentinel for an unknown size.
        bytes: u64,
        /// Allocator detail.
        detail: String,
    },
    /// An allocation failed without a known request size.
    AllocationSizeUnknown {
        /// Allocator detail.
        detail: String,
    },
    /// A serialized payload is structurally invalid.
    InvalidFormat {
        /// Format name (`"fn3 weights"`, ...).
        format: &'static str,
        /// What is wrong.
        detail: String,
    },
    /// A session was used in a phase that forbids the call, or an invariant of
    /// the session state machine was violated.
    SessionState {
        /// The rejected call.
        action: &'static str,
        /// Current phase name.
        phase: &'static str,
        /// What the caller must do instead.
        detail: String,
    },
    /// A step recording failed and the backend discarded it without
    /// submitting a prefix of the plan.
    ///
    /// The operation that failed reports its own error; this variant reports
    /// the discard when the step is closed separately, so `Ok(0)` cannot be
    /// mistaken for a step that ran and submitted nothing.
    StepDiscarded {
        /// What was discarded and what the caller must do instead.
        detail: String,
    },
    /// A signature-level skeleton entry point reached an unimplemented stage.
    ///
    /// This is a hard error, never a panic and never a silent approximation. It
    /// names the issue that owns the implementation.
    NotImplemented {
        /// The unimplemented capability.
        what: &'static str,
        /// Owning issue (`"FN3-07 (#53)"`).
        owner: &'static str,
    },
}

impl Fn3Error {
    /// Attach an operation to a variant that carries `op: Option<OpId>`.
    ///
    /// No-op for variants that do not name an op.
    pub fn with_op(self, op: OpId) -> Self {
        match self {
            Fn3Error::RankMismatch {
                op: None,
                expected,
                actual,
            } => Fn3Error::RankMismatch {
                op: Some(op),
                expected,
                actual,
            },
            Fn3Error::ShapeMismatch {
                op: None,
                expected,
                actual,
            } => Fn3Error::ShapeMismatch {
                op: Some(op),
                expected,
                actual,
            },
            Fn3Error::DTypeMismatch {
                op: None,
                expected,
                actual,
            } => Fn3Error::DTypeMismatch {
                op: Some(op),
                expected,
                actual,
            },
            Fn3Error::UnsupportedDType {
                op: None,
                dtype,
                context,
            } => Fn3Error::UnsupportedDType {
                op: Some(op),
                dtype,
                context,
            },
            other => other,
        }
    }

    /// The operation this error is attributed to, if any.
    pub fn op(&self) -> Option<OpId> {
        match self {
            Fn3Error::RankMismatch { op, .. }
            | Fn3Error::ShapeMismatch { op, .. }
            | Fn3Error::DTypeMismatch { op, .. }
            | Fn3Error::UnsupportedDType { op, .. }
            | Fn3Error::DeviceMismatch { op, .. } => *op,
            Fn3Error::UnsupportedOp { op, .. } | Fn3Error::ArityMismatch { op, .. } => Some(*op),
            _ => None,
        }
    }
}

fn op_prefix(op: Option<OpId>) -> String {
    match op {
        Some(op) => format!("op {op}: "),
        None => String::new(),
    }
}

impl fmt::Display for Fn3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fn3Error::ShapeOverflow {
                context,
                axis,
                detail,
            } => {
                write!(f, "shape arithmetic overflow in {context}")?;
                if let Some(axis) = axis {
                    write!(f, " (axis {axis})")?;
                }
                write!(f, ": {detail}")
            }
            Fn3Error::InvalidShape { context, detail } => {
                write!(f, "invalid shape in {context}: {detail}")
            }
            Fn3Error::RankMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "{}expected rank {expected}, found rank {actual}",
                op_prefix(*op)
            ),
            Fn3Error::ShapeMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "{}expected shape {expected}, found {actual}",
                op_prefix(*op)
            ),
            Fn3Error::DTypeMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "{}expected dtype {expected}, found {actual}",
                op_prefix(*op)
            ),
            Fn3Error::UnsupportedDType { op, dtype, context } => write!(
                f,
                "{}dtype {dtype} is not supported by {context}; the alpha compute dtype is f32",
                op_prefix(*op)
            ),
            Fn3Error::UnknownDTypeCode(error) => write!(f, "{error}"),
            Fn3Error::InvalidStride {
                axis,
                stride,
                detail,
            } => write!(f, "invalid stride on axis {axis}: {stride} ({detail})"),
            Fn3Error::NonContiguous { context, detail } => {
                write!(f, "{context} requires a contiguous view: {detail}")
            }
            Fn3Error::DeviceMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "{}expected device {expected}, found {actual}",
                op_prefix(*op)
            ),
            Fn3Error::DeviceUnavailable(error) => write!(f, "device unavailable: {error}"),
            Fn3Error::UnsupportedOp {
                op,
                backend,
                reason,
            } => write!(
                f,
                "op {op} ({}) is unsupported by backend {backend}: {reason}",
                op.description()
            ),
            Fn3Error::ArityMismatch {
                op,
                expected_inputs,
                actual_inputs,
                expected_outputs,
                actual_outputs,
            } => write!(
                f,
                "op {op}: expected {expected_inputs} inputs and {expected_outputs} outputs, \
                 found {actual_inputs} inputs and {actual_outputs} outputs"
            ),
            Fn3Error::DimOutOfEnvelope {
                symbol,
                requested,
                min,
                max,
            } => write!(
                f,
                "dimension {symbol} = {requested} is outside the declared envelope \
                 [{min}, {max}] (inclusive)"
            ),
            Fn3Error::CapacityExceeded {
                what,
                requested,
                capacity,
            } => write!(
                f,
                "capacity exceeded for {what}: requested {requested}, capacity {capacity}"
            ),
            Fn3Error::OutOfBounds { what, index, len } => {
                write!(f, "{what}: index {index} is out of bounds for length {len}")
            }
            Fn3Error::StorageCast {
                element,
                byte_len,
                detail,
            } => write!(f, "cannot view {byte_len} bytes as {element}: {detail}"),
            Fn3Error::StorageShared { bytes, handles } => write!(
                f,
                "cannot mutate {bytes}-byte storage: it is shared by {handles} handles"
            ),
            Fn3Error::StorageWindow {
                what,
                offset,
                len,
                storage_len,
                detail,
            } => write!(
                f,
                "{what}: byte window at offset {offset} with length {len} is invalid for \
                 {storage_len} bytes of storage: {detail}"
            ),
            Fn3Error::Allocation { bytes, detail } => {
                write!(f, "failed to allocate {bytes} bytes: {detail}")
            }
            Fn3Error::AllocationSizeUnknown { detail } => {
                write!(f, "allocation failed (request size unknown): {detail}")
            }
            Fn3Error::InvalidFormat { format, detail } => {
                write!(f, "invalid {format}: {detail}")
            }
            Fn3Error::SessionState {
                action,
                phase,
                detail,
            } => write!(
                f,
                "session state error during {action} (phase {phase}): {detail}"
            ),
            Fn3Error::StepDiscarded { detail } => {
                write!(f, "step recording discarded: {detail}")
            }
            Fn3Error::NotImplemented { what, owner } => write!(
                f,
                "{what} is not implemented in FastNN 3.0 (owner: {owner})"
            ),
        }
    }
}

impl std::error::Error for Fn3Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Fn3Error::UnknownDTypeCode(error) => Some(error),
            Fn3Error::DeviceUnavailable(error) => Some(error),
            _ => None,
        }
    }
}

impl From<UnknownDTypeCode> for Fn3Error {
    fn from(error: UnknownDTypeCode) -> Self {
        Fn3Error::UnknownDTypeCode(error)
    }
}

impl From<DeviceResolveError> for Fn3Error {
    fn from(error: DeviceResolveError) -> Self {
        Fn3Error::DeviceUnavailable(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::device::DeviceRequest;

    #[test]
    fn messages_name_op_and_values() {
        let error = Fn3Error::ShapeMismatch {
            op: Some(OpId::MatMul),
            expected: "[2, 3]".to_string(),
            actual: "[4, 3]".to_string(),
        };
        let text = error.to_string();
        assert!(text.contains("op matmul"), "{text}");
        assert!(text.contains("[2, 3]"), "{text}");
        assert!(text.contains("[4, 3]"), "{text}");
    }

    #[test]
    fn device_errors_delegate_and_keep_source() {
        let resolve = DeviceRequest::Vulkan { index: Some(1000) }
            .resolve(&crate::inference::device::available_devices())
            .unwrap_err();
        let error = Fn3Error::from(resolve);
        let text = error.to_string();
        assert!(text.contains("vulkan(index=1000)"), "{text}");
        // cpu is always listed first; adapters follow only when this build can
        // see them, so a fixed one-element list would be a snapshot that breaks
        // the moment the vulkan feature is enabled.
        assert!(text.contains("available devices: [cpu"), "{text}");
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn with_op_only_fills_missing_ops() {
        let error = Fn3Error::ShapeOverflow {
            context: "test",
            axis: None,
            detail: "x".to_string(),
        }
        .with_op(OpId::Add);
        assert_eq!(error.op(), None);
        assert!(matches!(error, Fn3Error::ShapeOverflow { .. }));
    }
}
