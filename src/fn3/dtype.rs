//! Element types for the FastNN 3.0 inference core.
//!
//! The alpha dtype set is deliberately tiny: one compute dtype (`F32`), one
//! index/shape dtype (`I64`), one auxiliary integer dtype (`I32`) and one mask
//! dtype (`Bool`). Quantized formats (`U4`, `I8`, `F16`, ...) are represented
//! nowhere in the alpha type system; they get their own variants in a later
//! milestone and their own reserved `OpId` range (see [`crate::fn3::ops`]).
//!
//! # Rejection, not coercion
//!
//! Nothing in this module converts one dtype into another. A request for a
//! dtype that this build does not know about is reported as
//! [`UnknownDTypeCode`], carrying the call-site context, and an operation that
//! receives a known-but-unsupported dtype is reported by its backend as
//! `Fn3Error::UnsupportedDType`. Silent promotion (for example `I32` -> `F32`
//! compute) is a contract violation.

use std::fmt;

/// Element type of a tensor buffer.
///
/// The discriminants are the stable on-disk / on-wire codes used by typed
/// weight sections. They must never be renumbered; add new codes instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum DType {
    /// IEEE-754 binary32. The only alpha compute dtype.
    F32 = 0,
    /// Signed 64-bit integer. Indices, shape tensors, `ArgMax` results.
    I64 = 1,
    /// Signed 32-bit integer. Auxiliary integer payloads.
    I32 = 2,
    /// Boolean mask, stored as one byte per element with value `0` or `1`.
    Bool = 3,
}

impl DType {
    /// Every dtype this build understands, in stable code order.
    pub const ALL: [DType; 4] = [DType::F32, DType::I64, DType::I32, DType::Bool];

    /// Size of one element in bytes.
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::Bool => 1,
        }
    }

    /// Stable short name, used in diagnostics and in plan/weight metadata.
    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::Bool => "bool",
        }
    }

    /// Stable numeric code used by serialized formats.
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// `true` for dtypes that alpha compute kernels accept as arithmetic
    /// operands. Only `F32` qualifies: `I32`/`I64`/`Bool` are metadata dtypes
    /// and kernels must reject them with a contextual error instead of casting.
    pub const fn is_compute(self) -> bool {
        matches!(self, DType::F32)
    }

    /// `true` for dtypes used as indices or shape values.
    pub const fn is_index(self) -> bool {
        matches!(self, DType::I64)
    }

    /// `true` for integral dtypes (`I32`, `I64`).
    pub const fn is_integer(self) -> bool {
        matches!(self, DType::I32 | DType::I64)
    }

    /// `true` for the boolean mask dtype.
    pub const fn is_mask(self) -> bool {
        matches!(self, DType::Bool)
    }

    /// Decode a stable numeric code.
    ///
    /// `context` names the reader (for example `"typed weight header"`) so the
    /// resulting error tells a user where the bad byte came from.
    pub fn from_code(code: u8, context: &'static str) -> Result<DType, UnknownDTypeCode> {
        match code {
            0 => Ok(DType::F32),
            1 => Ok(DType::I64),
            2 => Ok(DType::I32),
            3 => Ok(DType::Bool),
            other => Err(UnknownDTypeCode {
                code: other,
                context,
            }),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A numeric dtype code that this build of the 3.0 core does not define.
///
/// Kept separate from [`crate::fn3::error::Fn3Error`] so that `dtype` stays a
/// dependency-free leaf; `Fn3Error` converts it with [`From`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownDTypeCode {
    /// The rejected code.
    pub code: u8,
    /// Where the code was read from.
    pub context: &'static str,
}

impl fmt::Display for UnknownDTypeCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown dtype code {} in {}; this build supports f32=0, i64=1, i32=2, bool=3",
            self.code, self.context
        )
    }
}

impl std::error::Error for UnknownDTypeCode {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip_and_are_stable() {
        assert_eq!(DType::F32.code(), 0);
        assert_eq!(DType::I64.code(), 1);
        assert_eq!(DType::I32.code(), 2);
        assert_eq!(DType::Bool.code(), 3);
        for dtype in DType::ALL {
            assert_eq!(DType::from_code(dtype.code(), "test"), Ok(dtype));
        }
    }

    #[test]
    fn classification_is_exclusive_for_compute_dtypes() {
        assert!(DType::F32.is_compute());
        assert!(!DType::I64.is_compute());
        assert!(DType::I64.is_index());
        assert!(DType::Bool.is_mask());
        assert_eq!(DType::I64.size_bytes(), 8);
    }

    #[test]
    fn unknown_code_reports_context() {
        let error = DType::from_code(9, "typed weight header").unwrap_err();
        let text = error.to_string();
        assert!(text.contains('9'), "{text}");
        assert!(text.contains("typed weight header"), "{text}");
    }
}
