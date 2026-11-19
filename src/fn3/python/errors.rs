//! Python exception hierarchy for the 3.0 inference surface.
//!
//! # Why a hierarchy rather than one class
//!
//! A Python caller acts differently on each of the failures this surface can
//! report, so each one is its own class:
//!
//! | Class | Raised for | Attributes worth reading |
//! |---|---|---|
//! | [`Fn3Error`] | base of everything below | `message` |
//! | [`Fn3ModelFileError`] | a model file is missing or unreadable | `path` |
//! | [`Fn3FormatError`] | the bytes are not a valid artifact | `kind` |
//! | [`Fn3ImportError`] | an ONNX file could not be imported | `problems` |
//! | [`Fn3ModelExistsError`] | the destination exists and differs | `path` |
//! | [`Fn3DeviceError`] | an explicit device request cannot be honoured | `requested`, `available` |
//! | [`Fn3EnvelopeError`] | a request falls outside the declared window | `symbol`, `requested`, `min`, `max` |
//! | [`Fn3CapacityError`] | a bounded resource would be exceeded | `what`, `requested`, `capacity` |
//! | [`Fn3SessionError`] | the session phase forbids the call | `action`, `phase` |
//! | [`Fn3TokenError`] | a token id is not a valid id for this model | `token`, `vocab_size` |
//! | [`Fn3ShapeError`] | shapes, strides, ranks, storage | — |
//! | [`Fn3TypeError`] | a dtype the kernel does not compute on | — |
//! | [`Fn3UnsupportedError`] | the backend has no kernel for an op | `op`, `backend` |
//! | [`Fn3ExecutionError`] | anything else, with the core's message intact | — |
//!
//! Every message is the core's own message, so the two surfaces say the same
//! thing; the attributes exist so a caller can branch without parsing prose.
//!
//! # No bare strings, no panics
//!
//! A `PyErr::new_err("...")` is used only where the failure is a Python API
//! misuse that no Rust error describes (a wrong argument type, an unparsable
//! device string). Everything that comes out of the Rust core goes through the
//! mapping below, and the mapping itself cannot panic: PyO3 raises the class and
//! the attribute is best-effort, with the value already in the message.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::inference::device::DeviceResolveError;
use crate::inference::error::Fn3Error as CoreError;
use crate::inference::import::{ImportError, ImportProblemKind};
use crate::inference::model::format::{FormatError, FormatErrorKind};

create_exception!(
    fastnn,
    Fn3Error,
    PyException,
    "Base exception for the FastNN 3.0 inference surface."
);
create_exception!(
    fastnn,
    Fn3ModelError,
    Fn3Error,
    "Base exception for prepared-model files."
);
create_exception!(
    fastnn,
    Fn3ModelFileError,
    Fn3ModelError,
    "A model file is missing, unreadable or unwritable."
);
create_exception!(
    fastnn,
    Fn3FormatError,
    Fn3ModelError,
    "The bytes are not a valid FastNN 3 prepared-model artifact."
);
create_exception!(
    fastnn,
    Fn3ImportError,
    Fn3ModelError,
    "An ONNX model could not be imported."
);
create_exception!(
    fastnn,
    Fn3ModelExistsError,
    Fn3ModelFileError,
    "The destination exists and describes a different model; pass overwrite=True to replace it."
);
create_exception!(
    fastnn,
    Fn3DeviceError,
    Fn3Error,
    "An explicit device request cannot be honoured by this build or this machine."
);
create_exception!(
    fastnn,
    Fn3EnvelopeError,
    Fn3Error,
    "A dimension is outside the model artifact's declared window."
);
create_exception!(
    fastnn,
    Fn3CapacityError,
    Fn3Error,
    "A bounded resource would be exceeded (KV cache, position table, ...)."
);
create_exception!(
    fastnn,
    Fn3SessionError,
    Fn3Error,
    "The session phase forbids this call, or the call needs a different constructor."
);
create_exception!(
    fastnn,
    Fn3TokenError,
    Fn3Error,
    "A token id is not a valid id for this model."
);
create_exception!(
    fastnn,
    Fn3ShapeError,
    Fn3Error,
    "A shape, stride, rank, index or storage request is invalid."
);
create_exception!(
    fastnn,
    Fn3TypeError,
    Fn3Error,
    "A tensor dtype the operation does not compute on."
);
create_exception!(
    fastnn,
    Fn3UnsupportedError,
    Fn3Error,
    "The backend or this build does not implement the requested operation."
);
create_exception!(
    fastnn,
    Fn3ExecutionError,
    Fn3Error,
    "The core rejected the request; the message is the core's own."
);

/// Register this module's exceptions on the `inference` submodule.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("Fn3Error", py.get_type::<Fn3Error>())?;
    module.add("Fn3ModelError", py.get_type::<Fn3ModelError>())?;
    module.add("Fn3ModelFileError", py.get_type::<Fn3ModelFileError>())?;
    module.add("Fn3FormatError", py.get_type::<Fn3FormatError>())?;
    module.add("Fn3ImportError", py.get_type::<Fn3ImportError>())?;
    module.add("Fn3ModelExistsError", py.get_type::<Fn3ModelExistsError>())?;
    module.add("Fn3DeviceError", py.get_type::<Fn3DeviceError>())?;
    module.add("Fn3EnvelopeError", py.get_type::<Fn3EnvelopeError>())?;
    module.add("Fn3CapacityError", py.get_type::<Fn3CapacityError>())?;
    module.add("Fn3SessionError", py.get_type::<Fn3SessionError>())?;
    module.add("Fn3TokenError", py.get_type::<Fn3TokenError>())?;
    module.add("Fn3ShapeError", py.get_type::<Fn3ShapeError>())?;
    module.add("Fn3TypeError", py.get_type::<Fn3TypeError>())?;
    module.add("Fn3UnsupportedError", py.get_type::<Fn3UnsupportedError>())?;
    module.add("Fn3ExecutionError", py.get_type::<Fn3ExecutionError>())?;
    Ok(())
}

/// Machine-readable name of a format failure, exposed as `Fn3FormatError.kind`.
///
/// The variant list is the format module's; a new variant must be added here,
/// which is what keeps the Python `kind` vocabulary honest.
pub(crate) fn format_kind(kind: &FormatErrorKind) -> &'static str {
    match kind {
        FormatErrorKind::Io { .. } => "io",
        FormatErrorKind::BadMagic { .. } => "bad_magic",
        FormatErrorKind::UnsupportedVersion { .. } => "unsupported_version",
        FormatErrorKind::UnsupportedDtypeEncoding { .. } => "unsupported_dtype_encoding",
        FormatErrorKind::ReservedField { .. } => "reserved_field",
        FormatErrorKind::Truncated { .. } => "truncated",
        FormatErrorKind::LengthMismatch { .. } => "length_mismatch",
        FormatErrorKind::ChecksumMismatch { .. } => "checksum_mismatch",
        FormatErrorKind::LimitExceeded { .. } => "limit_exceeded",
        FormatErrorKind::Malformed { .. } => "malformed",
        FormatErrorKind::Inconsistent { .. } => "inconsistent",
        FormatErrorKind::UnknownDType(_) => "unknown_dtype",
        FormatErrorKind::UnknownOpCode { .. } => "unknown_op_code",
        FormatErrorKind::Contract { .. } => "contract",
    }
}

/// Build a typed exception, then attach attributes best-effort.
///
/// The message always carries every value, so a failed `setattr` (which cannot
/// happen for a class we define and a plain Python value, but is not worth an
/// `unwrap` at the FFI boundary) loses convenience, never information.
fn typed<T: pyo3::type_object::PyTypeInfo>(
    py: Python<'_>,
    message: impl Into<String>,
    set: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<()>,
) -> PyErr {
    let error = PyErr::new::<T, _>((message.into(),));
    let _ = set(error.value(py).as_any());
    error
}

/// The values of a device-resolution failure.
fn device_attributes(error: &DeviceResolveError) -> (String, Vec<String>, String) {
    (
        error.requested().to_string(),
        error.available().to_vec(),
        error.reason().to_string(),
    )
}

/// Map a device-resolution failure.
pub(crate) fn from_device(py: Python<'_>, error: DeviceResolveError) -> PyErr {
    let (requested, available, reason) = device_attributes(&error);
    let message = error.to_string();
    typed::<Fn3DeviceError>(py, message, |value| {
        value.setattr("requested", requested)?;
        value.setattr("available", available)?;
        value.setattr("reason", reason)
    })
}

/// A device request this binding could not even parse.
pub(crate) fn typed_device(
    py: Python<'_>,
    message: String,
    requested: String,
    available: Vec<String>,
    reason: String,
) -> PyErr {
    typed::<Fn3DeviceError>(py, message, |value| {
        value.setattr("requested", requested)?;
        value.setattr("available", available)?;
        value.setattr("reason", reason)
    })
}

/// A token id that is not an id this model can consume.
pub(crate) fn token_error(py: Python<'_>, message: String, token: i64, vocab_size: u32) -> PyErr {
    typed::<Fn3TokenError>(py, message, |value| {
        value.setattr("token", token)?;
        value.setattr("vocab_size", vocab_size)
    })
}

/// A model file could not be read or written.
pub(crate) fn file_error(py: Python<'_>, message: String, path: String) -> PyErr {
    typed::<Fn3ModelFileError>(py, message, |value| {
        value.setattr("path", path)?;
        value.setattr("kind", "io")
    })
}

/// A model-level failure that is neither a file error nor a format error.
pub(crate) fn model_error(py: Python<'_>, message: String) -> PyErr {
    typed::<Fn3ModelError>(py, message, |value| value.setattr("kind", "model"))
}

/// The destination exists and describes a different model.
pub(crate) fn model_exists(
    py: Python<'_>,
    message: String,
    path: String,
    differences: Vec<String>,
) -> PyErr {
    typed::<Fn3ModelExistsError>(py, message, |value| {
        value.setattr("path", path)?;
        value.setattr("kind", "exists")?;
        value.setattr("differences", differences)
    })
}

/// A bounded resource would be exceeded before anything was done.
pub(crate) fn capacity(
    py: Python<'_>,
    message: String,
    what: &'static str,
    requested: u64,
    capacity: u64,
) -> PyErr {
    typed::<Fn3CapacityError>(py, message, |value| {
        value.setattr("what", what)?;
        value.setattr("requested", requested)?;
        value.setattr("capacity", capacity)
    })
}

/// Map an artifact read/write failure.
pub(crate) fn from_format(py: Python<'_>, error: FormatError) -> PyErr {
    let kind = format_kind(error.kind());
    let version_mismatch = error.is_version_mismatch();
    let corruption = error.is_corruption();
    let message = error.to_string();
    if let FormatErrorKind::Io { path, .. } = error.kind() {
        let path = path.clone();
        return typed::<Fn3ModelFileError>(py, message, |value| {
            value.setattr("path", path)?;
            value.setattr("kind", kind)
        });
    }
    typed::<Fn3FormatError>(py, message, |value| {
        value.setattr("kind", kind)?;
        value.setattr("version_mismatch", version_mismatch)?;
        value.setattr("corruption", corruption)
    })
}

/// Map an ONNX import failure, keeping every problem the importer accumulated.
pub(crate) fn from_import(py: Python<'_>, error: ImportError) -> PyErr {
    let problems: Vec<String> = error.problems().iter().map(ToString::to_string).collect();
    let count = problems.len();
    let message = error.to_string();
    let missing = error
        .problems()
        .iter()
        .find(|problem| matches!(problem.kind, ImportProblemKind::Io { .. }))
        .map(|problem| problem.to_string());
    if let Some(path_message) = missing {
        return typed::<Fn3ModelFileError>(py, path_message, |value| {
            value.setattr("kind", "io")?;
            value.setattr("problems", problems)
        });
    }
    typed::<Fn3ImportError>(py, message, |value| {
        value.setattr("problems", problems)?;
        value.setattr("problem_count", count)
    })
}

/// Map a core error, choosing the class a Python caller would branch on.
pub(crate) fn from_core(py: Python<'_>, error: CoreError) -> PyErr {
    match &error {
        CoreError::DeviceUnavailable(resolve) => from_device(py, resolve.clone()),
        CoreError::DeviceMismatch {
            op,
            expected,
            actual,
        } => {
            let message = error.to_string();
            let requested = expected.diagnostic();
            let actual = actual.diagnostic();
            let op = op.map(|op| op.to_string());
            let available: Vec<String> = crate::inference::device::available_devices()
                .iter()
                .map(crate::inference::device::DeviceInfo::label)
                .collect();
            typed::<Fn3DeviceError>(py, message, |value| {
                value.setattr("requested", requested)?;
                value.setattr("actual", actual)?;
                value.setattr("available", available)?;
                value.setattr("op", op)
            })
        }
        CoreError::DimOutOfEnvelope {
            symbol,
            requested,
            min,
            max,
        } => {
            let message = error.to_string();
            let (symbol, requested, min, max) = (*symbol, *requested, *min, *max);
            typed::<Fn3EnvelopeError>(py, message, |value| {
                value.setattr("symbol", symbol)?;
                value.setattr("requested", requested)?;
                value.setattr("min", min)?;
                value.setattr("max", max)
            })
        }
        CoreError::CapacityExceeded {
            what,
            requested,
            capacity,
        } => {
            let message = error.to_string();
            let (what, requested, capacity) = (*what, *requested, *capacity);
            typed::<Fn3CapacityError>(py, message, |value| {
                value.setattr("what", what)?;
                value.setattr("requested", requested)?;
                value.setattr("capacity", capacity)
            })
        }
        CoreError::SessionState {
            action,
            phase,
            detail,
        } => {
            let message = error.to_string();
            let (action, phase, detail) = (*action, *phase, detail.clone());
            typed::<Fn3SessionError>(py, message, |value| {
                value.setattr("action", action)?;
                value.setattr("phase", phase)?;
                value.setattr("detail", detail)
            })
        }
        CoreError::UnsupportedOp {
            op,
            backend,
            reason,
        } => {
            let message = error.to_string();
            let (op, backend, reason) = (op.to_string(), *backend, reason.clone());
            typed::<Fn3UnsupportedError>(py, message, |value| {
                value.setattr("op", op)?;
                value.setattr("backend", backend)?;
                value.setattr("reason", reason)
            })
        }
        CoreError::NotImplemented { what, owner } => {
            let message = error.to_string();
            let (what, owner) = (*what, *owner);
            typed::<Fn3UnsupportedError>(py, message, |value| {
                value.setattr("what", what)?;
                value.setattr("owner", owner)
            })
        }
        CoreError::DTypeMismatch { .. }
        | CoreError::UnsupportedDType { .. }
        | CoreError::UnknownDTypeCode(_)
        | CoreError::StorageCast { .. } => typed::<Fn3TypeError>(py, error.to_string(), |_| Ok(())),
        CoreError::ShapeOverflow { .. }
        | CoreError::InvalidShape { .. }
        | CoreError::RankMismatch { .. }
        | CoreError::ShapeMismatch { .. }
        | CoreError::InvalidStride { .. }
        | CoreError::NonContiguous { .. }
        | CoreError::OutOfBounds { .. }
        | CoreError::StorageShared { .. }
        | CoreError::StorageWindow { .. } => {
            typed::<Fn3ShapeError>(py, error.to_string(), |_| Ok(()))
        }
        CoreError::Allocation { .. }
        | CoreError::InvalidFormat { .. }
        | CoreError::ArityMismatch { .. } => {
            typed::<Fn3ExecutionError>(py, error.to_string(), |_| Ok(()))
        }
    }
}

/// Register the public exception names on the Python package's dynamism.
///
/// Kept for symmetry with the other registration helpers: the Python facade
/// re-exports these classes so `except fastnn.inference.Fn3DeviceError` works.
pub(crate) const NAMES: &[&str] = &[
    "Fn3Error",
    "Fn3ModelError",
    "Fn3ModelFileError",
    "Fn3FormatError",
    "Fn3ImportError",
    "Fn3ModelExistsError",
    "Fn3DeviceError",
    "Fn3EnvelopeError",
    "Fn3CapacityError",
    "Fn3SessionError",
    "Fn3TokenError",
    "Fn3ShapeError",
    "Fn3TypeError",
    "Fn3UnsupportedError",
    "Fn3ExecutionError",
];
