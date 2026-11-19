//! Typed failures for `fn3-quant-format-v1`.
//!
//! Every malformed metadata case in the format has a distinct variant. The
//! [`QuantError::kind`] strings match the independent Python reference in
//! `tools/quant/fn3_quant.py`, so a fixture can assert the same failure from
//! both implementations without matching prose.
//!
//! The parser never panics on malformed input and never reads outside the
//! supplied slice: all offsets are bounds-checked before a slice is formed.

use std::fmt;

/// A rejected `fn3-quant-format-v1` object or an invalid decode request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantError {
    /// The blob is shorter than the fixed 64-byte header.
    TruncatedHeader { available: usize },
    /// The first four bytes are not `FN3Q`.
    BadMagic { found: [u8; 4] },
    /// `format_version` is not 1.
    UnsupportedVersion { found: u16 },
    /// `header_bytes` is not 64.
    BadHeaderSize { found: u16 },
    /// `flags` is not zero in v1.
    UnsupportedFlags { found: u32 },
    /// `tensor_count` is zero or exceeds the v1 limit.
    BadTensorCount { found: u32 },
    /// `descriptor_bytes` is not `tensor_count * 64`.
    BadDescriptorBytes { found: u32 },
    /// `descriptor_offset` is not 64.
    BadDescriptorOffset { found: u64 },
    /// `group_offset` is not the end of the descriptor array.
    BadGroupOffset { found: u64 },
    /// `group_dir_bytes` is not a multiple of 8.
    BadGroupDirBytes { found: u32 },
    /// `payload_offset` is not the 16-byte-aligned end of the group directory.
    BadPayloadOffset { found: u64 },
    /// The file length is not exactly `payload_offset + payload_bytes`.
    TrailingOrTruncatedBytes { file: usize, payload_end: u64 },
    /// The metadata checksum does not match.
    MetaChecksum { expected: u32, actual: u32 },
    /// The payload checksum does not match.
    PayloadChecksum { expected: u32, actual: u32 },
    /// A tensor name is empty, non-ASCII, unterminated, or has trailing bytes.
    BadName { tensor: usize, reason: &'static str },
    /// Two tensors in one blob use the same name.
    DuplicateName { name: String },
    /// A tensor has zero rows or zero reduction length.
    ZeroExtent { tensor: String, field: &'static str },
    /// `qbits` is not 8; v1 has no 4-bit payload.
    UnsupportedQBits { tensor: String, found: u16 },
    /// `group_size` is not 64; v1 fixes it to 64.
    UnsupportedGroupSize { tensor: String, found: u16 },
    /// `groups_per_row` is not `ceil(k / group_size)`.
    BadGroupCount {
        tensor: String,
        found: u32,
        expected: u32,
    },
    /// `group_dir_index` is not the cumulative offset of this tensor's groups.
    BadGroupIndex {
        tensor: String,
        found: u32,
        expected: u32,
    },
    /// `row_stride_bytes` is not `align_up(k, 16)`.
    BadRowStride {
        tensor: String,
        found: u32,
        expected: u32,
    },
    /// A tensor payload does not start on a 16-byte boundary.
    MisalignedOffset {
        tensor: String,
        offset: u64,
        alignment: u64,
    },
    /// A tensor payload is aligned but not at the canonical cursor.
    BadDataOffset {
        tensor: String,
        found: u64,
        expected: u64,
    },
    /// A tensor's data, scales or zero points run past the payload section.
    TruncatedPayload { tensor: String },
    /// Alignment padding between tensors is non-zero.
    NonZeroPadding { tensor: String, offset: u64 },
    /// A row's post-`k` alignment padding is non-zero.
    NonZeroTailPadding { tensor: String, offset: u64 },
    /// A group does not start exactly where the previous group ended.
    GroupGapOrOverlap {
        tensor: String,
        group: u32,
        found: u32,
        expected: u32,
    },
    /// A group length is zero, longer than 64, or not the tail length.
    BadGroupLength {
        tensor: String,
        group: u32,
        found: u32,
        expected: u32,
    },
    /// The groups do not cover exactly `0..k`.
    IncompleteGroups { tensor: String, end: u32, k: u32 },
    /// A scale is zero, negative, infinite or NaN.
    BadScale {
        tensor: String,
        row: u32,
        group: u32,
        bits: u16,
    },
    /// A zero-point is outside `[-128, 127]` for 8-bit codes.
    ZeroPointOutOfRange {
        tensor: String,
        row: u32,
        group: u32,
        value: i32,
    },
    /// The group directory has entries no descriptor claims.
    UnusedGroupEntries { count: u32 },
    /// The canonical payload cursor does not end at the payload boundary.
    PayloadTail { expected: u64, found: u64 },
    /// `dequantize_into` was given a buffer of the wrong length.
    OutputTooSmall { expected: usize, found: usize },
}

impl QuantError {
    /// Stable machine-readable kind, shared with the Python reference.
    pub fn kind(&self) -> &'static str {
        match self {
            QuantError::TruncatedHeader { .. } => "truncated-header",
            QuantError::BadMagic { .. } => "bad-magic",
            QuantError::UnsupportedVersion { .. } => "unsupported-version",
            QuantError::BadHeaderSize { .. } => "bad-header-size",
            QuantError::UnsupportedFlags { .. } => "unsupported-flags",
            QuantError::BadTensorCount { .. } => "bad-tensor-count",
            QuantError::BadDescriptorBytes { .. } => "bad-descriptor-bytes",
            QuantError::BadDescriptorOffset { .. } => "bad-descriptor-offset",
            QuantError::BadGroupOffset { .. } => "bad-group-offset",
            QuantError::BadGroupDirBytes { .. } => "bad-group-dir-bytes",
            QuantError::BadPayloadOffset { .. } => "bad-payload-offset",
            QuantError::TrailingOrTruncatedBytes { .. } => "trailing-or-truncated-bytes",
            QuantError::MetaChecksum { .. } => "meta-checksum",
            QuantError::PayloadChecksum { .. } => "payload-checksum",
            QuantError::BadName { .. } => "bad-name",
            QuantError::DuplicateName { .. } => "duplicate-name",
            QuantError::ZeroExtent { .. } => "zero-extent",
            QuantError::UnsupportedQBits { .. } => "unsupported-qbits",
            QuantError::UnsupportedGroupSize { .. } => "unsupported-group-size",
            QuantError::BadGroupCount { .. } => "bad-group-count",
            QuantError::BadGroupIndex { .. } => "bad-group-index",
            QuantError::BadRowStride { .. } => "bad-row-stride",
            QuantError::MisalignedOffset { .. } => "misaligned-offset",
            QuantError::BadDataOffset { .. } => "bad-data-offset",
            QuantError::TruncatedPayload { .. } => "truncated-payload",
            QuantError::NonZeroPadding { .. } => "nonzero-padding",
            QuantError::NonZeroTailPadding { .. } => "nonzero-tail-padding",
            QuantError::GroupGapOrOverlap { .. } => "group-gap-or-overlap",
            QuantError::BadGroupLength { .. } => "bad-group-length",
            QuantError::IncompleteGroups { .. } => "incomplete-groups",
            QuantError::BadScale { .. } => "bad-scale",
            QuantError::ZeroPointOutOfRange { .. } => "zero-point-out-of-range",
            QuantError::UnusedGroupEntries { .. } => "unused-group-entries",
            QuantError::PayloadTail { .. } => "payload-tail",
            QuantError::OutputTooSmall { .. } => "output-too-small",
        }
    }
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantError::TruncatedHeader { available } => write!(
                f,
                "quant: blob is {available} bytes, shorter than the 64-byte header"
            ),
            QuantError::BadMagic { found } => {
                write!(f, "quant: magic {found:?} is not FN3Q")
            }
            QuantError::UnsupportedVersion { found } => {
                write!(f, "quant: unsupported format version {found}")
            }
            QuantError::BadHeaderSize { found } => {
                write!(f, "quant: header_bytes is {found}, expected 64")
            }
            QuantError::UnsupportedFlags { found } => {
                write!(f, "quant: unsupported flags {found:#x}")
            }
            QuantError::BadTensorCount { found } => {
                write!(f, "quant: tensor_count {found} is outside 1..=4096")
            }
            QuantError::BadDescriptorBytes { found } => {
                write!(f, "quant: descriptor_bytes is {found}")
            }
            QuantError::BadDescriptorOffset { found } => {
                write!(f, "quant: descriptor_offset is {found}, expected 64")
            }
            QuantError::BadGroupOffset { found } => {
                write!(f, "quant: group_offset is {found}")
            }
            QuantError::BadGroupDirBytes { found } => {
                write!(f, "quant: group_dir_bytes {found} is not a multiple of 8")
            }
            QuantError::BadPayloadOffset { found } => {
                write!(f, "quant: payload_offset is {found}")
            }
            QuantError::TrailingOrTruncatedBytes { file, payload_end } => write!(
                f,
                "quant: file is {file} bytes, payload ends at {payload_end}"
            ),
            QuantError::MetaChecksum { expected, actual } => write!(
                f,
                "quant: metadata checksum {actual:#010x} != {expected:#010x}"
            ),
            QuantError::PayloadChecksum { expected, actual } => write!(
                f,
                "quant: payload checksum {actual:#010x} != {expected:#010x}"
            ),
            QuantError::BadName { tensor, reason } => {
                write!(f, "quant: tensor {tensor} name: {reason}")
            }
            QuantError::DuplicateName { name } => {
                write!(f, "quant: duplicate tensor name {name:?}")
            }
            QuantError::ZeroExtent { tensor, field } => {
                write!(f, "quant: tensor {tensor:?} has zero {field}")
            }
            QuantError::UnsupportedQBits { tensor, found } => write!(
                f,
                "quant: tensor {tensor:?} uses {found} bits, v1 requires 8"
            ),
            QuantError::UnsupportedGroupSize { tensor, found } => write!(
                f,
                "quant: tensor {tensor:?} uses group size {found}, v1 requires 64"
            ),
            QuantError::BadGroupCount {
                tensor,
                found,
                expected,
            } => write!(
                f,
                "quant: tensor {tensor:?} declares {found} groups, expected {expected}"
            ),
            QuantError::BadGroupIndex {
                tensor,
                found,
                expected,
            } => write!(
                f,
                "quant: tensor {tensor:?} group directory starts at {found}, expected {expected}"
            ),
            QuantError::BadRowStride {
                tensor,
                found,
                expected,
            } => write!(
                f,
                "quant: tensor {tensor:?} row stride {found} != {expected}"
            ),
            QuantError::MisalignedOffset {
                tensor,
                offset,
                alignment,
            } => write!(
                f,
                "quant: tensor {tensor:?} data offset {offset} is not {alignment}-byte aligned"
            ),
            QuantError::BadDataOffset {
                tensor,
                found,
                expected,
            } => write!(
                f,
                "quant: tensor {tensor:?} data offset {found} != {expected}"
            ),
            QuantError::TruncatedPayload { tensor } => {
                write!(f, "quant: tensor {tensor:?} payload runs past the section")
            }
            QuantError::NonZeroPadding { tensor, offset } => write!(
                f,
                "quant: tensor {tensor:?} has non-zero padding at byte {offset}"
            ),
            QuantError::NonZeroTailPadding { tensor, offset } => write!(
                f,
                "quant: tensor {tensor:?} has non-zero row padding at byte {offset}"
            ),
            QuantError::GroupGapOrOverlap {
                tensor,
                group,
                found,
                expected,
            } => write!(
                f,
                "quant: tensor {tensor:?} group {group} starts at {found}, expected {expected}"
            ),
            QuantError::BadGroupLength {
                tensor,
                group,
                found,
                expected,
            } => write!(
                f,
                "quant: tensor {tensor:?} group {group} length {found}, expected {expected}"
            ),
            QuantError::IncompleteGroups { tensor, end, k } => write!(
                f,
                "quant: tensor {tensor:?} groups end at {end}, k is {k}"
            ),
            QuantError::BadScale {
                tensor,
                row,
                group,
                bits,
            } => write!(
                f,
                "quant: tensor {tensor:?} scale [{row}, {group}] is not finite positive (bits {bits:#06x})"
            ),
            QuantError::ZeroPointOutOfRange {
                tensor,
                row,
                group,
                value,
            } => write!(
                f,
                "quant: tensor {tensor:?} zero-point [{row}, {group}] is {value}"
            ),
            QuantError::UnusedGroupEntries { count } => {
                write!(f, "quant: {count} group directory entries are unused")
            }
            QuantError::PayloadTail { expected, found } => write!(
                f,
                "quant: payload cursor ends at {found}, expected {expected}"
            ),
            QuantError::OutputTooSmall { expected, found } => write!(
                f,
                "quant: output buffer has {found} elements, needs {expected}"
            ),
        }
    }
}

impl std::error::Error for QuantError {}
