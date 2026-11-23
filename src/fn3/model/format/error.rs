//! Typed errors for the versioned prepared-model format (version 2).
//!
//! # Why a format-local error type
//!
//! [`crate::inference::error::Fn3Error`] is the core's single error type, but its
//! `InvalidFormat` variant carries one flat string. A reader of an untrusted
//! file has to distinguish cases that a caller acts on differently: *this file is
//! from the future* (upgrade the reader), *this file is the old format*
//! (re-prepare it), *this file is corrupt* (re-download it), *this file is
//! hostile* (refuse it) and *this file is internally inconsistent* (a writer
//! bug). Hence one small, exhaustive enum here, converted to
//! `Fn3Error::InvalidFormat` at the boundary.
//!
//! Every variant names what was being read, and every structural error also
//! carries the absolute byte offset within the file.

use std::fmt;

use crate::inference::dtype::UnknownDTypeCode;
use crate::inference::error::Fn3Error;

/// A prepared-model artifact could not be written, read or validated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatError {
    kind: FormatErrorKind,
}

/// What went wrong, with the values that make the message actionable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatErrorKind {
    /// A file could not be read or written.
    Io {
        /// Path as the caller spelled it.
        path: String,
        /// Operating-system detail.
        detail: String,
    },
    /// The first eight bytes are not an artifact magic.
    ///
    /// This is also what a 2.x `.fnnc` file produces: that format is raw
    /// `bincode` with no magic, so it can never be mistaken for an artifact.
    BadMagic {
        /// The eight bytes that were found.
        found: [u8; 8],
    },
    /// A version-1 artifact (`FNN3PREP`, or `FNN3ART2` version 1) was opened.
    ///
    /// Version 1 stored a decoder-shaped graph and only three `OpParams`
    /// fields. It is not read by this build and there is no compatibility
    /// parser: the source ONNX must be prepared again.
    ObsoleteFormat {
        /// The magic the file carries.
        magic: [u8; 8],
        /// The version field the file declares (1 for a `FNN3PREP` file).
        version: u16,
    },
    /// The file declares a format version this build does not read.
    UnsupportedVersion {
        /// Version in the file header.
        found: u16,
        /// The only version this build reads.
        supported: u16,
    },
    /// The file declares a dtype encoding this build does not read.
    UnsupportedDtypeEncoding {
        /// Encoding id in the file header.
        found: u32,
        /// The only encoding this build reads.
        supported: u32,
    },
    /// A header field that this version requires to be zero is not.
    ReservedField {
        /// Field name.
        field: &'static str,
        /// The value that was found, widened for display.
        value: u64,
        /// Absolute byte offset of the field.
        offset: u64,
    },
    /// The input ended before the declared structure did.
    Truncated {
        /// What was being read.
        what: String,
        /// Bytes required to finish it.
        needed: u64,
        /// Bytes actually remaining.
        available: u64,
        /// Absolute byte offset where the read started.
        offset: u64,
    },
    /// Two length fields that must agree do not.
    LengthMismatch {
        /// Which length relationship failed.
        what: &'static str,
        /// The length the file declares.
        declared: u64,
        /// The length that is actually present.
        actual: u64,
        /// Absolute byte offset of the field that declared it.
        offset: u64,
    },
    /// A section's SHA-256 does not match the hash the file stores.
    HashMismatch {
        /// Section name.
        section: String,
        /// Absolute byte offset of the section.
        offset: u64,
        /// Hash stored in the `HASHES` section.
        stored: [u8; 32],
        /// Hash the section bytes actually produce.
        computed: [u8; 32],
    },
    /// A count or declared size exceeds this build's documented policy cap.
    ///
    /// Reported *before* the object is allocated, so a hostile count costs one
    /// comparison, not memory.
    LimitExceeded {
        /// What is capped.
        what: String,
        /// The count or size the file declares.
        requested: u64,
        /// The cap this build enforces.
        limit: u64,
    },
    /// The bytes are structurally wrong.
    ///
    /// The section name and the absolute file offset are the two facts a user
    /// needs to locate the bad byte.
    MalformedAt {
        /// Section or structure being read.
        section: String,
        /// Absolute byte offset within the file.
        offset: u64,
        /// The offending value and the rule it broke.
        detail: String,
    },
    /// The bytes decode but describe an impossible model.
    ///
    /// The distinction from [`FormatErrorKind::MalformedAt`] matters: a corrupt
    /// file breaks the framing, a *writer bug* produces a well-framed file whose
    /// graph, weight table and plans disagree.
    Inconsistent {
        /// What was being checked.
        what: String,
        /// Which two facts disagree.
        detail: String,
    },
    /// A tensor carried a dtype code this build does not define.
    UnknownDType(UnknownDTypeCode),
    /// A node or plan carried an op code this build does not define.
    UnknownOpCode {
        /// The rejected code, as it appeared in the file.
        code: u16,
    },
    /// The quantization metadata describes a precision this build cannot execute.
    UnsupportedPrecision {
        /// The precision code found.
        found: u8,
        /// The only precision version 2 executes.
        supported: &'static str,
    },
    /// A core contract rejected the decoded object.
    Contract {
        /// What was being built.
        what: &'static str,
        /// The core error, kept whole so its detail is not flattened.
        error: Fn3Error,
    },
}

impl FormatError {
    /// The machine-readable cause.
    pub fn kind(&self) -> &FormatErrorKind {
        &self.kind
    }

    /// Construct from a kind. Used inside the crate's codec.
    pub(super) fn new(kind: FormatErrorKind) -> FormatError {
        FormatError { kind }
    }

    /// The input ended early, naming the absolute file offset.
    pub(super) fn truncated_at(
        what: impl Into<String>,
        offset: u64,
        needed: u64,
        available: u64,
    ) -> FormatError {
        FormatError::new(FormatErrorKind::Truncated {
            what: what.into(),
            needed,
            available,
            offset,
        })
    }

    /// A structural failure at a known file offset.
    pub(super) fn malformed_at(
        section: impl Into<String>,
        offset: u64,
        detail: impl Into<String>,
    ) -> FormatError {
        FormatError::new(FormatErrorKind::MalformedAt {
            section: section.into(),
            offset,
            detail: detail.into(),
        })
    }

    /// The bytes decode but describe an impossible model.
    pub(super) fn inconsistent(what: impl Into<String>, detail: impl Into<String>) -> FormatError {
        FormatError::new(FormatErrorKind::Inconsistent {
            what: what.into(),
            detail: detail.into(),
        })
    }

    /// A count or size exceeded a policy cap.
    pub(super) fn limit(what: impl Into<String>, requested: u64, limit: u64) -> FormatError {
        FormatError::new(FormatErrorKind::LimitExceeded {
            what: what.into(),
            requested,
            limit,
        })
    }

    /// A core contract rejected the decoded object.
    pub(super) fn contract(what: &'static str, error: Fn3Error) -> FormatError {
        FormatError::new(FormatErrorKind::Contract { what, error })
    }

    /// A file could not be read.
    pub(super) fn io(path: impl Into<String>, detail: impl Into<String>) -> FormatError {
        FormatError::new(FormatErrorKind::Io {
            path: path.into(),
            detail: detail.into(),
        })
    }

    /// A length relationship failed at `offset`.
    pub(super) fn length_mismatch(
        what: &'static str,
        declared: u64,
        actual: u64,
        offset: u64,
    ) -> FormatError {
        FormatError::new(FormatErrorKind::LengthMismatch {
            what,
            declared,
            actual,
            offset,
        })
    }

    /// `true` when the file was produced by a different format version.
    ///
    /// A caller that can re-serialize (a CLI, the Python facade) uses this to
    /// print "regenerate the artifact" instead of "the file is corrupt".
    pub fn is_version_mismatch(&self) -> bool {
        matches!(
            self.kind,
            FormatErrorKind::UnsupportedVersion { .. }
                | FormatErrorKind::UnsupportedDtypeEncoding { .. }
                | FormatErrorKind::ObsoleteFormat { .. }
        )
    }

    /// `true` when the bytes are corrupt rather than inconsistent.
    pub fn is_corruption(&self) -> bool {
        matches!(
            self.kind,
            FormatErrorKind::Truncated { .. }
                | FormatErrorKind::HashMismatch { .. }
                | FormatErrorKind::BadMagic { .. }
                | FormatErrorKind::ObsoleteFormat { .. }
                | FormatErrorKind::LengthMismatch { .. }
                | FormatErrorKind::ReservedField { .. }
                | FormatErrorKind::MalformedAt { .. }
                | FormatErrorKind::LimitExceeded { .. }
        )
    }

    /// The section this error belongs to, when it names one.
    pub fn section(&self) -> Option<&str> {
        match &self.kind {
            FormatErrorKind::MalformedAt { section, .. }
            | FormatErrorKind::HashMismatch { section, .. } => Some(section),
            _ => None,
        }
    }

    /// The absolute file offset this error names, when it names one.
    pub fn offset(&self) -> Option<u64> {
        match &self.kind {
            FormatErrorKind::Truncated { offset, .. }
            | FormatErrorKind::LengthMismatch { offset, .. }
            | FormatErrorKind::ReservedField { offset, .. }
            | FormatErrorKind::HashMismatch { offset, .. }
            | FormatErrorKind::MalformedAt { offset, .. } => Some(*offset),
            _ => None,
        }
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            FormatErrorKind::Io { path, detail } => {
                write!(f, "cannot access `{path}`: {detail}")
            }
            FormatErrorKind::BadMagic { found } => write!(
                f,
                "the first eight bytes are {}; a prepared-model artifact starts with FNN3ART2",
                render_magic(*found)
            ),
            FormatErrorKind::ObsoleteFormat { magic, version } => write!(
                f,
                "this is a version-{version} artifact (magic {}); format version 2 replaced it \
                 and has no compatibility reader. Re-prepare the source ONNX with this build \
                 (`fastnn-infer prepare --onnx <model.onnx> --out <model.fnn3>`)",
                render_magic(*magic)
            ),
            FormatErrorKind::UnsupportedVersion { found, supported } => write!(
                f,
                "this file is prepared-model format version {found}, but this build reads \
                 version {supported} only; re-prepare it with this build"
            ),
            FormatErrorKind::UnsupportedDtypeEncoding { found, supported } => write!(
                f,
                "this file uses dtype encoding {found}; this build reads encoding {supported} only"
            ),
            FormatErrorKind::ReservedField {
                field,
                value,
                offset,
            } => write!(
                f,
                "reserved field `{field}` at byte {offset} is {value:#x}, expected 0"
            ),
            FormatErrorKind::Truncated {
                what,
                needed,
                available,
                offset,
            } => write!(
                f,
                "truncated {what} at byte {offset}: {needed} bytes are required but only \
                 {available} are available"
            ),
            FormatErrorKind::LengthMismatch {
                what,
                declared,
                actual,
                offset,
            } => write!(
                f,
                "{what} at byte {offset} declares {declared}, but {actual} is present"
            ),
            FormatErrorKind::HashMismatch {
                section,
                offset,
                stored,
                computed,
            } => write!(
                f,
                "the {section} section at byte {offset} does not match its SHA-256: stored {}, \
                 computed {}",
                hex(stored),
                hex(computed)
            ),
            FormatErrorKind::LimitExceeded {
                what,
                requested,
                limit,
            } => write!(
                f,
                "{what} declares {requested}, above this build's limit of {limit}"
            ),
            FormatErrorKind::MalformedAt {
                section,
                offset,
                detail,
            } => write!(f, "malformed {section} at byte {offset}: {detail}"),
            FormatErrorKind::Inconsistent { what, detail } => {
                write!(f, "inconsistent {what}: {detail}")
            }
            FormatErrorKind::UnknownDType(error) => write!(f, "{error}"),
            FormatErrorKind::UnknownOpCode { code } => {
                write!(f, "unknown op code {code} in the node or plan table")
            }
            FormatErrorKind::UnsupportedPrecision { found, supported } => write!(
                f,
                "the artifact declares precision code {found}; this build executes {supported} only"
            ),
            FormatErrorKind::Contract { what, error } => {
                write!(f, "{what} failed the core contract: {error}")
            }
        }
    }
}

impl std::error::Error for FormatError {}

impl From<FormatError> for Fn3Error {
    fn from(error: FormatError) -> Fn3Error {
        Fn3Error::InvalidFormat {
            format: "prepared-model artifact",
            detail: error.to_string(),
        }
    }
}

fn render_magic(magic: [u8; 8]) -> String {
    let mut text = String::with_capacity(10);
    text.push('`');
    for byte in magic {
        if byte.is_ascii_graphic() {
            text.push(byte as char);
        } else {
            text.push('.');
        }
    }
    text.push('`');
    text
}

fn hex(bytes: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in bytes {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}
