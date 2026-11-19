//! Typed errors for the versioned prepared-model format.
//!
//! # Why a format-local error type
//!
//! [`crate::inference::error::Fn3Error`] is the core's single error type, but it is
//! owned by FN3-02 (#48) and its `InvalidFormat` variant carries one flat
//! string. A reader of an untrusted file has to distinguish cases that a caller
//! acts on differently: *this file is from the future* (upgrade the reader),
//! *this file is corrupt* (re-download it), *this file is hostile* (refuse it)
//! and *this file is internally inconsistent* (a writer bug). Hence one small,
//! exhaustive enum here, converted to `Fn3Error::InvalidFormat` at the boundary.
//!
//! Every variant names what was being read and the values involved. Nothing is
//! ever reported as "parse error".

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
    /// The first eight bytes are not the prepared-model magic.
    ///
    /// This is also what a 2.x `.fnnc` file produces: that format is raw
    /// `bincode` with no magic, so it can never be mistaken for a 3.0 artifact.
    BadMagic {
        /// The eight bytes that were found.
        found: [u8; 8],
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
    },
    /// The input ended before the declared structure did.
    Truncated {
        /// What was being read.
        what: String,
        /// Bytes required to finish it.
        needed: u64,
        /// Bytes actually remaining.
        available: u64,
    },
    /// Two length fields that must agree do not.
    LengthMismatch {
        /// Which length relationship failed.
        what: &'static str,
        /// The length the file declares.
        declared: u64,
        /// The length that is actually present.
        actual: u64,
    },
    /// The payload checksum does not match the payload.
    ChecksumMismatch {
        /// CRC-32 stored in the header.
        stored: u32,
        /// CRC-32 the payload actually has.
        computed: u32,
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
    /// The bytes are structurally wrong (bad tag, non-UTF-8 name, ...).
    Malformed {
        /// What was being read.
        what: String,
        /// The offending value and the rule it broke.
        detail: String,
    },
    /// The bytes decode but describe an impossible model.
    ///
    /// The distinction from [`FormatErrorKind::Malformed`] matters: a corrupt
    /// file breaks the framing, a *writer bug* produces a well-framed file whose
    /// geometry, weights and value table disagree.
    Inconsistent {
        /// What was being checked.
        what: String,
        /// Which two facts disagree.
        detail: String,
    },
    /// A tensor carried a dtype code this build does not define.
    UnknownDType(UnknownDTypeCode),
    /// A node carried an op code this build does not define.
    UnknownOpCode {
        /// The rejected code, as it appeared in the file.
        code: u16,
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

    /// The input ended early.
    pub(super) fn truncated(what: impl Into<String>, needed: u64, available: u64) -> FormatError {
        FormatError::new(FormatErrorKind::Truncated {
            what: what.into(),
            needed,
            available,
        })
    }

    /// The bytes are structurally wrong.
    pub(super) fn malformed(what: impl Into<String>, detail: impl Into<String>) -> FormatError {
        FormatError::new(FormatErrorKind::Malformed {
            what: what.into(),
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

    /// `true` when the file was produced by a different format version.
    ///
    /// A caller that can re-serialize (a CLI, the Python facade) uses this to
    /// print "regenerate the artifact" instead of "the file is corrupt".
    pub fn is_version_mismatch(&self) -> bool {
        matches!(
            self.kind,
            FormatErrorKind::UnsupportedVersion { .. }
                | FormatErrorKind::UnsupportedDtypeEncoding { .. }
        )
    }

    /// `true` when the bytes are corrupt rather than inconsistent.
    pub fn is_corruption(&self) -> bool {
        matches!(
            self.kind,
            FormatErrorKind::Truncated { .. }
                | FormatErrorKind::ChecksumMismatch { .. }
                | FormatErrorKind::BadMagic { .. }
                | FormatErrorKind::LengthMismatch { .. }
                | FormatErrorKind::ReservedField { .. }
                | FormatErrorKind::Malformed { .. }
        )
    }
}

/// Render bytes as lowercase hex pairs, for a magic-mismatch message.
fn hex(bytes: &[u8]) -> String {
    let mut rendered = String::with_capacity(bytes.len() * 3);
    for (position, byte) in bytes.iter().enumerate() {
        if position > 0 {
            rendered.push(' ');
        }
        rendered.push_str(&format!("{byte:02x}"));
    }
    rendered
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            FormatErrorKind::Io { path, detail } => write!(f, "{path}: {detail}"),
            FormatErrorKind::BadMagic { found } => write!(
                f,
                "not a FastNN 3 prepared model: the file starts with {}, the magic is {}",
                hex(found),
                hex(&super::MAGIC)
            ),
            FormatErrorKind::UnsupportedVersion { found, supported } => write!(
                f,
                "this file is prepared-model format version {found}, but this build reads \
                 version {supported} only"
            ),
            FormatErrorKind::UnsupportedDtypeEncoding { found, supported } => write!(
                f,
                "this file uses dtype encoding {found}, but this build reads encoding \
                 {supported} only"
            ),
            FormatErrorKind::ReservedField { field, value } => write!(
                f,
                "reserved header field `{field}` is {value}, but this build requires 0"
            ),
            FormatErrorKind::Truncated {
                what,
                needed,
                available,
            } => write!(
                f,
                "truncated {what}: {needed} bytes are required, {available} are available"
            ),
            FormatErrorKind::LengthMismatch {
                what,
                declared,
                actual,
            } => write!(
                f,
                "{what}: the file declares {declared}, but {actual} is present"
            ),
            FormatErrorKind::ChecksumMismatch { stored, computed } => write!(
                f,
                "payload checksum mismatch: the header stores crc32 {stored:#010x}, the \
                 payload hashes to {computed:#010x}; the file is corrupt"
            ),
            FormatErrorKind::LimitExceeded {
                what,
                requested,
                limit,
            } => write!(
                f,
                "{what} declares {requested}, above this build's limit of {limit}"
            ),
            FormatErrorKind::Malformed { what, detail } => {
                write!(f, "malformed {what}: {detail}")
            }
            FormatErrorKind::Inconsistent { what, detail } => {
                write!(f, "inconsistent {what}: {detail}")
            }
            FormatErrorKind::UnknownDType(error) => write!(f, "{error}"),
            FormatErrorKind::UnknownOpCode { code } => {
                write!(f, "unknown op code {code}")?;
                if crate::inference::ops::OpId::is_reserved(*code) {
                    write!(
                        f,
                        " (it falls in a reserved range this build does not define)"
                    )?;
                }
                write!(f, "; the file was written by a build with a larger op set")
            }
            FormatErrorKind::Contract { what, error } => write!(f, "{what}: {error}"),
        }
    }
}

impl std::error::Error for FormatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            FormatErrorKind::UnknownDType(error) => Some(error),
            FormatErrorKind::Contract { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl From<UnknownDTypeCode> for FormatError {
    fn from(error: UnknownDTypeCode) -> FormatError {
        FormatError::new(FormatErrorKind::UnknownDType(error))
    }
}

impl From<FormatError> for Fn3Error {
    fn from(error: FormatError) -> Fn3Error {
        Fn3Error::InvalidFormat {
            format: "fn3 prepared model",
            detail: error.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_errors_state_both_versions_and_are_classified() {
        let error = FormatError::new(FormatErrorKind::UnsupportedVersion {
            found: 2,
            supported: 1,
        });
        let text = error.to_string();
        assert!(text.contains("version 2"), "{text}");
        assert!(text.contains("version 1 only"), "{text}");
        assert!(error.is_version_mismatch());
        assert!(!error.is_corruption());
    }

    #[test]
    fn magic_errors_render_hex_and_are_corruption() {
        let error = FormatError::new(FormatErrorKind::BadMagic {
            found: *b"FNN2PREP",
        });
        let text = error.to_string();
        assert!(text.contains("46 4e 4e 32"), "{text}");
        assert!(text.contains("46 4e 4e 33 50 52 45 50"), "{text}");
        assert!(error.is_corruption());
    }

    #[test]
    fn checksum_errors_name_both_values() {
        let error = FormatError::new(FormatErrorKind::ChecksumMismatch {
            stored: 0x1234_5678,
            computed: 0x9abc_def0,
        });
        let text = error.to_string();
        assert!(text.contains("0x12345678"), "{text}");
        assert!(text.contains("0x9abcdef0"), "{text}");
        assert!(error.is_corruption());
    }

    #[test]
    fn reserved_op_codes_say_so() {
        let error = FormatError::new(FormatErrorKind::UnknownOpCode { code: 0x0101 });
        assert!(error.to_string().contains("reserved range"), "{error}");
        let plain = FormatError::new(FormatErrorKind::UnknownOpCode { code: 5000 });
        assert!(!plain.to_string().contains("reserved range"), "{plain}");
    }

    #[test]
    fn core_errors_keep_their_detail_and_source() {
        let error = FormatError::contract(
            "weight table",
            Fn3Error::OutOfBounds {
                what: "weight",
                index: 3,
                len: 2,
            },
        );
        assert!(error.to_string().contains("weight table"), "{error}");
        assert!(std::error::Error::source(&error).is_some());
        let converted: Fn3Error = error.into();
        assert!(
            converted.to_string().contains("fn3 prepared model"),
            "{converted}"
        );
    }
}
