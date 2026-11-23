//! The versioned, memory-mappable prepared-model artifact (format version 2).
//!
//! # The file, in full
//!
//! ```text
//! header 128 bytes, 64-byte aligned
//!   0..8    b"FNN3ART2"                 magic
//!   8..10   u16 format version          = 2
//!  10..12   u16 header length           = 128
//!  12..16   u32 section count           = 5
//!  16..20   u32 target profile count    1..=8
//!  20..24   u32 flags                   = 0
//!  24..28   u32 dtype encoding          = 1  (one DType::code byte per tensor)
//!  28..32   u32 reserved                = 0
//!  32..64   source ONNX SHA-256
//!  64..72   u64 file length
//!  72..80   u64 payload length           file length - 128
//!  80..88   u64 directory offset         = 128
//!  88..96   u64 directory length         = 128
//!  96..104  u64 weights section offset
//! 104..112  u64 weights section length
//! 112..120  u64 weight payload offset   (absolute)
//! 120..128  u64 weight payload length
//!
//! directory 128 bytes at offset 128
//!   u32 section count (5)
//!   u32 reserved (0)
//!   5 × { u32 tag, u32 reserved, u64 absolute offset, u64 length }
//!
//! sections, each starting on a 64-byte boundary and zero-padded to one:
//!   1 GRAPH    typed IR, provenance and interface names
//!   2 PLANS    one full execution plan per target profile
//!   3 WEIGHTS  one contiguous 64-byte-aligned payload plus its table
//!   4 QUANT    per-tensor scale/zero-point/axis metadata (empty for F32)
//!   5 HASHES   SHA-256 of the header and of GRAPH, PLANS, WEIGHTS, QUANT
//! ```
//!
//! All integers are little-endian. All strings are a `u32` byte length followed
//! by UTF-8 bytes. Floats are their `f32` bit pattern. Booleans are `u8` 0/1.
//! Enumerations are `u8` codes with a documented meaning; an unknown code is an
//! error, never a default.
//!
//! # Why mmap
//!
//! The `WEIGHTS` section is one payload whose every entry starts 64 bytes into a
//! page-aligned mapping. A reader hands each weight out as a `&[u8]` (and, for
//! F32, a `&[f32]`) pointing straight into the mapping. Two sessions opened on
//! one artifact therefore share exactly the same pages, and loading a 200 MB
//! model does not allocate 200 MB of process memory. `memmap2` is the only
//! dependency this format adds, and it is not optional: the core artifact reader
//! owns the mapping.
//!
//! The file is opened read-only. The format has no executable content, no
//! pointers, no offsets into process memory and no code. A weight range is cast
//! to `&[f32]` only after the table proves its length is a whole number of `f32`
//! elements and its start is 64-byte aligned.
//!
//! # Versioning policy
//!
//! Version 1 (`FNN3PREP`) stored a decoder-shaped graph and only three
//! `OpParams` fields. Version 2 replaces it; there is no compatibility reader.
//! A version-1 file is [`FormatErrorKind::ObsoleteFormat`] and the message tells
//! the caller to re-prepare the source ONNX. A version above 2 is
//! [`FormatErrorKind::UnsupportedVersion`]. A change that adds a section,
//! reorders a field, changes a meaning or assigns a new code **must** bump the
//! version.
//!
//! # Determinism
//!
//! Every table is in declaration order, no `HashMap` reaches the byte stream, and
//! padding is zeros. `to_bytes` twice is byte-identical, and
//! `from_bytes` → `to_bytes` is byte-identical for a decoded artifact.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::inference::digest::Sha256;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::import::ImportedGraph;
use crate::inference::ir::Graph;
use crate::inference::model::WeightSet;
use crate::inference::runtime::plan::{
    EngineCounts, LayoutStrategy, Plan, Slot, SlotId, SlotSource, SLOT_ALIGNMENT,
};
use crate::inference::runtime::PlanOptions;

use super::error::{FormatError, FormatErrorKind};
use super::graph::{check_interface, decode_graph, encode_graph, ArtifactMetadata, GraphParts};
use super::plan::{decode_plans, encode_plans, ArtifactPlan, Precision, MAX_TARGETS};
use super::weights::{decode_weight_table, decode_weights, encode_weights, ArtifactWeight};
use super::wire::{Reader, Writer};

/// The eight magic bytes of a version-2 prepared-model artifact.
pub const MAGIC: [u8; 8] = *b"FNN3ART2";

/// The magic of the replaced version-1 artifact, kept only to name it.
pub const LEGACY_MAGIC: [u8; 8] = *b"FNN3PREP";

/// The only format version this build writes and reads.
pub const FORMAT_VERSION: u16 = 2;

/// Size of the fixed header, in bytes.
pub const HEADER_BYTES: usize = 128;

/// Size of the section directory.
pub const DIRECTORY_BYTES: usize = 128;

/// Number of sections (`GRAPH`, `PLANS`, `WEIGHTS`, `QUANT`, `HASHES`).
pub const SECTION_COUNT: usize = 5;

/// Dtype encoding id: one byte per tensor, carrying [`DType::code`].
pub const DTYPE_ENCODING: u32 = 1;

/// Largest artifact payload this build accepts, in bytes.
///
/// Eight GiB is far above the largest model in scope (498 MB) and far below
/// "the process dies before reporting an error".
pub const MAX_PAYLOAD_BYTES: u64 = 8 * (1 << 30);

/// Largest value table this build accepts.
pub const MAX_VALUES: u64 = 1 << 20;

/// Largest op list this build accepts.
pub const MAX_NODES: u64 = 1 << 20;

/// Largest weight table this build accepts.
pub const MAX_WEIGHTS: u64 = 1 << 20;

/// Largest tensor rank this build accepts anywhere in the artifact.
pub const MAX_RANK: u64 = 64;

/// Largest `OpParams` vector this build accepts.
pub const MAX_PARAM_AXES: u64 = 1024;

/// Largest section `inspect` is willing to read into memory.
///
/// `inspect` reads `GRAPH`, `PLANS` and `QUANT` (never the weight payload), so
/// this bounds a metadata allocation, not a model load.
pub const MAX_INSPECT_SECTION_BYTES: u64 = 256 * (1 << 20);

/// Target name for the default CPU profile (`onednn` where this build has it).
pub const TARGET_CPU: &str = "cpu";

/// Target name for the scalar, unfused reference profile.
pub const TARGET_CPU_REFERENCE: &str = "cpu-reference";

/// Target name that expands to every profile this build can produce.
pub const TARGET_ALL: &str = "all";

/// The target names a user may pass to `prepare`.
pub const TARGETS: [&str; 3] = [TARGET_CPU, TARGET_CPU_REFERENCE, TARGET_ALL];

/// Resolve a target request into the profiles to write, in deterministic order.
pub fn resolve_targets(name: &str) -> Result<Vec<&'static str>, FormatError> {
    match name {
        TARGET_CPU => Ok(vec![TARGET_CPU]),
        TARGET_CPU_REFERENCE => Ok(vec![TARGET_CPU_REFERENCE]),
        TARGET_ALL => Ok(vec![TARGET_CPU, TARGET_CPU_REFERENCE]),
        other => Err(FormatError::inconsistent(
            "target request",
            format!(
                "target `{other}` is not defined; this build knows {}",
                TARGETS.join(", ")
            ),
        )),
    }
}

/// Build the plan for one target name.
fn plan_for_target(
    imported: &ImportedGraph,
    target: &str,
    precision: Precision,
) -> Result<Plan, FormatError> {
    if precision != Precision::F32 {
        return Err(FormatError::new(FormatErrorKind::UnsupportedPrecision {
            found: precision.code(),
            supported: "f32 (code 0)",
        }));
    }
    let plan = match target {
        TARGET_CPU => {
            #[cfg(feature = "cpu-onednn")]
            {
                use crate::inference::backend::cpu::onednn::OneDnnCapability;
                Plan::from_imported_with_options(imported, &OneDnnCapability, PlanOptions::new())
            }
            #[cfg(not(feature = "cpu-onednn"))]
            {
                use crate::inference::backend::ScalarOnly;
                Plan::from_imported_with_options(imported, &ScalarOnly, PlanOptions::new())
            }
        }
        TARGET_CPU_REFERENCE => {
            use crate::inference::backend::ScalarOnly;
            Plan::from_imported_with_options(imported, &ScalarOnly, PlanOptions::unfused())
        }
        other => {
            return Err(FormatError::malformed_at(
                "target",
                0,
                format!("target `{other}` has no plan builder"),
            ))
        }
    };
    plan.map_err(|error| FormatError::contract("target plan", error))
}

// ---------------------------------------------------------------------------
// Header and sections
// ---------------------------------------------------------------------------

mod tag {
    pub(super) const HEADER: u32 = 0;
    pub(super) const GRAPH: u32 = 1;
    pub(super) const PLANS: u32 = 2;
    pub(super) const WEIGHTS: u32 = 3;
    pub(super) const QUANT: u32 = 4;
    pub(super) const HASHES: u32 = 5;

    /// Sections in write order.
    pub(super) const ALL: [u32; super::SECTION_COUNT] = [GRAPH, PLANS, WEIGHTS, QUANT, HASHES];

    pub(super) fn name(tag: u32) -> &'static str {
        match tag {
            HEADER => "HEADER",
            GRAPH => "GRAPH",
            PLANS => "PLANS",
            WEIGHTS => "WEIGHTS",
            QUANT => "QUANT",
            HASHES => "HASHES",
            _ => "unknown",
        }
    }
}

/// The validated fixed header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactHeader {
    /// Format version.
    pub version: u16,
    /// Number of sections (always 5).
    pub section_count: u32,
    /// Number of target plans.
    pub target_count: u32,
    /// Reserved flags (always 0).
    pub flags: u32,
    /// Dtype encoding id (always 1).
    pub dtype_encoding: u32,
    /// SHA-256 of the source ONNX file.
    pub source_onnx_sha256: [u8; 32],
    /// Total file length.
    pub file_bytes: u64,
    /// `file_bytes - HEADER_BYTES`.
    pub payload_bytes: u64,
    /// Directory offset (always 128).
    pub directory_offset: u64,
    /// Directory length (always 128).
    pub directory_bytes: u64,
    /// Absolute offset of the `WEIGHTS` section.
    pub weights_offset: u64,
    /// Length of the `WEIGHTS` section.
    pub weights_length: u64,
    /// Absolute offset of the weight payload.
    pub weight_payload_offset: u64,
    /// Length of the weight payload.
    pub weight_payload_bytes: u64,
}

impl ArtifactHeader {
    /// Parse and validate the fixed header.
    ///
    /// `file_bytes` is the actual file length, so a header cannot claim a size
    /// the file does not have.
    pub fn parse(bytes: &[u8], file_bytes: u64) -> Result<ArtifactHeader, FormatError> {
        if bytes.len() < HEADER_BYTES {
            return Err(FormatError::truncated_at(
                "artifact header",
                0,
                HEADER_BYTES as u64,
                bytes.len() as u64,
            ));
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[..8]);
        if magic == LEGACY_MAGIC {
            let version = u16::from_le_bytes([bytes[8], bytes[9]]);
            return Err(FormatError::new(FormatErrorKind::ObsoleteFormat {
                magic,
                version,
            }));
        }
        if magic != MAGIC {
            return Err(FormatError::new(FormatErrorKind::BadMagic { found: magic }));
        }
        let mut reader = Reader::new(&bytes[..HEADER_BYTES]);
        reader.take(8, "magic")?;
        let version = reader.u16("format version")?;
        if version != FORMAT_VERSION {
            if version < FORMAT_VERSION {
                return Err(FormatError::new(FormatErrorKind::ObsoleteFormat {
                    magic,
                    version,
                }));
            }
            return Err(FormatError::new(FormatErrorKind::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            }));
        }
        let header_length = u64::from(reader.u16("header length")?);
        if header_length != HEADER_BYTES as u64 {
            return Err(FormatError::length_mismatch(
                "header length",
                header_length,
                HEADER_BYTES as u64,
                10,
            ));
        }
        let section_count = reader.u32("section count")?;
        if section_count != SECTION_COUNT as u32 {
            return Err(FormatError::malformed_at(
                "HEADER",
                12,
                format!(
                    "version {FORMAT_VERSION} has {SECTION_COUNT} sections, the file declares \
                     {section_count}"
                ),
            ));
        }
        let target_count = reader.u32("target profile count")?;
        if target_count == 0 || u64::from(target_count) > MAX_TARGETS {
            return Err(FormatError::limit(
                "target profile count",
                u64::from(target_count),
                MAX_TARGETS,
            ));
        }
        let flags = reader.u32("flags")?;
        if flags != 0 {
            return Err(FormatError::new(FormatErrorKind::ReservedField {
                field: "flags",
                value: u64::from(flags),
                offset: 20,
            }));
        }
        let dtype_encoding = reader.u32("dtype encoding")?;
        if dtype_encoding != DTYPE_ENCODING {
            return Err(FormatError::new(
                FormatErrorKind::UnsupportedDtypeEncoding {
                    found: dtype_encoding,
                    supported: DTYPE_ENCODING,
                },
            ));
        }
        reader.zero_u32("reserved header word")?;
        let mut source_onnx_sha256 = [0u8; 32];
        source_onnx_sha256.copy_from_slice(reader.take(32, "source ONNX SHA-256")?);
        let declared_file_bytes = reader.u64("file length")?;
        let payload_bytes = reader.u64("payload length")?;
        let directory_offset = reader.u64("directory offset")?;
        let directory_bytes = reader.u64("directory length")?;
        let weights_offset = reader.u64("weights offset")?;
        let weights_length = reader.u64("weights length")?;
        let weight_payload_offset = reader.u64("weight payload offset")?;
        let weight_payload_bytes = reader.u64("weight payload length")?;

        if declared_file_bytes != file_bytes {
            return Err(FormatError::length_mismatch(
                "file length",
                declared_file_bytes,
                file_bytes,
                64,
            ));
        }
        let expected_payload = file_bytes.saturating_sub(HEADER_BYTES as u64);
        if payload_bytes != expected_payload {
            return Err(FormatError::length_mismatch(
                "payload length",
                payload_bytes,
                expected_payload,
                72,
            ));
        }
        if payload_bytes > MAX_PAYLOAD_BYTES {
            return Err(FormatError::limit(
                "payload length",
                payload_bytes,
                MAX_PAYLOAD_BYTES,
            ));
        }
        if directory_offset != HEADER_BYTES as u64 {
            return Err(FormatError::length_mismatch(
                "directory offset",
                directory_offset,
                HEADER_BYTES as u64,
                80,
            ));
        }
        if directory_bytes != DIRECTORY_BYTES as u64 {
            return Err(FormatError::length_mismatch(
                "directory length",
                directory_bytes,
                DIRECTORY_BYTES as u64,
                88,
            ));
        }
        if weights_offset % 64 != 0 || weights_length % 64 != 0 {
            return Err(FormatError::malformed_at(
                "HEADER",
                96,
                format!(
                    "the WEIGHTS section at {weights_offset} (length {weights_length}) is not \
                     64-byte aligned"
                ),
            ));
        }
        let weights_end = weights_offset.checked_add(weights_length).ok_or_else(|| {
            FormatError::malformed_at("HEADER", 96, "the WEIGHTS range overflows")
        })?;
        if weights_end > file_bytes {
            return Err(FormatError::malformed_at(
                "HEADER",
                96,
                format!(
                    "the WEIGHTS section ends at byte {weights_end}, past the {file_bytes}-byte \
                     file"
                ),
            ));
        }
        if weight_payload_offset < weights_offset
            || weight_payload_offset % 64 != 0
            || weight_payload_offset
                .checked_add(weight_payload_bytes)
                .is_none_or(|end| end > weights_end)
        {
            return Err(FormatError::malformed_at(
                "HEADER",
                112,
                format!(
                    "the weight payload range {weight_payload_offset}..{} is not inside the \
                     WEIGHTS section {weights_offset}..{weights_end}",
                    weight_payload_offset.saturating_add(weight_payload_bytes)
                ),
            ));
        }
        Ok(ArtifactHeader {
            version,
            section_count,
            target_count,
            flags,
            dtype_encoding,
            source_onnx_sha256,
            file_bytes,
            payload_bytes,
            directory_offset,
            directory_bytes,
            weights_offset,
            weights_length,
            weight_payload_offset,
            weight_payload_bytes,
        })
    }

    /// The source ONNX digest as lowercase hexadecimal.
    pub fn source_sha256_hex(&self) -> String {
        hex(&self.source_onnx_sha256)
    }
}

/// One section's absolute position and declared length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionRange {
    /// Section tag (1..=5).
    pub tag: u32,
    /// Absolute file offset.
    pub offset: u64,
    /// Encoded length, excluding padding.
    pub length: u64,
}

impl SectionRange {
    /// Human name (`GRAPH`).
    pub fn name(&self) -> &'static str {
        tag::name(self.tag)
    }
}

/// One section's stored SHA-256.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionHash {
    /// Section name (`HEADER`, `GRAPH`, `PLANS`, `WEIGHTS`, `QUANT`).
    pub name: &'static str,
    /// Stored digest.
    pub sha256: [u8; 32],
}

impl SectionHash {
    /// Lowercase hexadecimal.
    pub fn hex(&self) -> String {
        hex(&self.sha256)
    }
}

fn align64(value: u64) -> Result<u64, FormatError> {
    value
        .checked_add(63)
        .map(|value| value & !63)
        .ok_or_else(|| FormatError::limit("64-byte alignment", value, u64::MAX))
}

/// Parse the fixed directory.
pub fn parse_directory(
    directory: &[u8],
    header: &ArtifactHeader,
) -> Result<[SectionRange; SECTION_COUNT], FormatError> {
    if directory.len() < DIRECTORY_BYTES {
        return Err(FormatError::truncated_at(
            "artifact directory",
            header.directory_offset,
            DIRECTORY_BYTES as u64,
            directory.len() as u64,
        ));
    }
    let mut reader = Reader::at(&directory[..DIRECTORY_BYTES], header.directory_offset);
    let count = reader.count("section directory", SECTION_COUNT as u64, 24)?;
    if count != SECTION_COUNT {
        return Err(FormatError::malformed_at(
            "HEADER",
            header.directory_offset,
            format!("expected {SECTION_COUNT} section entries, found {count}"),
        ));
    }
    reader.zero_u32("directory reserved word")?;

    let mut ranges = [SectionRange {
        tag: 0,
        offset: 0,
        length: 0,
    }; SECTION_COUNT];
    let mut cursor = header.directory_offset + DIRECTORY_BYTES as u64;
    for (position, expected) in tag::ALL.into_iter().enumerate() {
        let entry_offset = header.directory_offset + 8 + position as u64 * 24;
        let found = reader.u32("section tag")?;
        reader.zero_u32("section tag reserved word")?;
        let offset = reader.u64("section offset")?;
        let length = reader.u64("section length")?;
        if found != expected {
            return Err(FormatError::malformed_at(
                "HEADER",
                entry_offset,
                format!(
                    "section {position} is tagged {found} ({}), expected {expected} ({})",
                    tag::name(found),
                    tag::name(expected)
                ),
            ));
        }
        if offset != cursor {
            return Err(FormatError::malformed_at(
                "HEADER",
                entry_offset + 8,
                format!(
                    "section {position} ({}) starts at {offset}, expected {cursor}",
                    tag::name(found)
                ),
            ));
        }
        if offset % 64 != 0 {
            return Err(FormatError::malformed_at(
                "HEADER",
                entry_offset + 8,
                format!("section {} is not 64-byte aligned", tag::name(found)),
            ));
        }
        let end = offset.checked_add(length).ok_or_else(|| {
            FormatError::malformed_at(
                "HEADER",
                entry_offset + 16,
                format!("section {} length overflows", tag::name(found)),
            )
        })?;
        if end > header.file_bytes {
            return Err(FormatError::truncated_at(
                format!("{} section", tag::name(found)),
                offset,
                length,
                header.file_bytes.saturating_sub(offset),
            ));
        }
        if length == 0 {
            return Err(FormatError::malformed_at(
                "HEADER",
                entry_offset + 16,
                format!("section {} has length 0", tag::name(found)),
            ));
        }
        cursor = align64(end)?;
        ranges[position] = SectionRange {
            tag: found,
            offset,
            length,
        };
    }
    if cursor != header.file_bytes {
        return Err(FormatError::length_mismatch(
            "file length",
            header.file_bytes,
            cursor,
            header.directory_offset,
        ));
    }
    let weights = ranges[2];
    if weights.offset != header.weights_offset || weights.length != header.weights_length {
        return Err(FormatError::inconsistent(
            "section directory",
            format!(
                "the directory places WEIGHTS at {}..{} but the header says {}..{}",
                weights.offset,
                weights.offset + weights.length,
                header.weights_offset,
                header.weights_length
            ),
        ));
    }
    Ok(ranges)
}

/// Require the bytes between sections to be zero padding.
///
/// The writer pads every section to 64 bytes; a reader that accepted a nonzero
/// padding byte would make the encoding non-canonical and a single-byte flip in
/// the gap invisible to every section hash.
pub(super) fn check_section_padding(
    bytes: &[u8],
    ranges: &[SectionRange; SECTION_COUNT],
) -> Result<(), FormatError> {
    for range in ranges {
        let end = range.offset as usize + range.length as usize;
        let padded = align64((range.offset + range.length) as u64)? as usize;
        let limit = padded.min(bytes.len());
        if let Some(position) = bytes[end..limit].iter().position(|&byte| byte != 0) {
            return Err(FormatError::malformed_at(
                range.name(),
                end as u64 + position as u64,
                format!(
                    "inter-section padding byte is {:#04x}, expected 0",
                    bytes[end + position]
                ),
            ));
        }
    }
    Ok(())
}

fn read_hashes(section: &[u8], base: u64) -> Result<[SectionHash; 5], FormatError> {
    let mut reader = Reader::at(section, base);
    let count = reader.count("hash table", 5, 40)?;
    if count != 5 {
        return Err(FormatError::malformed_at(
            "HASHES",
            base,
            format!("expected 5 hashes (HEADER, GRAPH, PLANS, WEIGHTS, QUANT), found {count}"),
        ));
    }
    reader.zero_u32("hash table reserved word")?;
    let mut hashes = [SectionHash {
        name: "unknown",
        sha256: [0u8; 32],
    }; 5];
    for position in 0..5 {
        let entry_offset = base + 8 + position as u64 * 40;
        let tag = reader.u32("hash section tag")?;
        let expected = position as u32;
        if tag != expected {
            return Err(FormatError::malformed_at(
                "HASHES",
                entry_offset,
                format!("hash entry {position} is tagged {tag}, expected {expected}"),
            ));
        }
        reader.zero_u32("hash entry reserved word")?;
        let mut digest = [0u8; 32];
        digest.copy_from_slice(reader.take(32, "section SHA-256")?);
        hashes[position] = SectionHash {
            name: tag::name(tag),
            sha256: digest,
        };
    }
    reader.finish("HASHES section")?;
    Ok(hashes)
}

fn hash_slice(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finish()
}

/// Hash one file range with a fixed 1 MiB buffer.
///
/// The buffer is reused for every chunk, so validating a 200 MB weight section
/// does not grow process memory by the section size. It also deliberately uses
/// `File` reads rather than the mapping, so a load does not fault the weight
/// pages into the process just to check them.
fn hash_file_range(
    file: &mut std::fs::File,
    offset: u64,
    length: u64,
) -> std::io::Result<[u8; 32]> {
    use std::io::{Read, Seek, SeekFrom};
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    file.seek(SeekFrom::Start(offset))?;
    let mut remaining = length;
    while remaining > 0 {
        let take = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..take])?;
        hasher.update(&buffer[..take]);
        remaining -= take as u64;
    }
    Ok(hasher.finish())
}

fn verify_hashes_slice(
    bytes: &[u8],
    ranges: &[SectionRange; SECTION_COUNT],
    hashes: &[SectionHash; 5],
) -> Result<(), FormatError> {
    let mut pairs: Vec<(&'static str, &[u8])> = Vec::with_capacity(5);
    pairs.push(("HEADER", &bytes[..HEADER_BYTES]));
    for range in ranges.iter().take(4) {
        let start = range.offset as usize;
        let end = start + range.length as usize;
        pairs.push((range.name(), &bytes[start..end]));
    }
    for (name, data) in pairs {
        let stored = hashes
            .iter()
            .find(|hash| hash.name == name)
            .ok_or_else(|| {
                FormatError::inconsistent("hash table", format!("no hash for {name}"))
            })?;
        let computed = hash_slice(data);
        if computed != stored.sha256 {
            return Err(FormatError::new(FormatErrorKind::HashMismatch {
                section: name.to_string(),
                offset: if name == "HEADER" {
                    0
                } else {
                    ranges
                        .iter()
                        .find(|range| range.name() == name)
                        .map(|range| range.offset)
                        .unwrap_or(0)
                },
                stored: stored.sha256,
                computed,
            }));
        }
    }
    Ok(())
}

fn verify_hashes_file(
    file: &mut std::fs::File,
    ranges: &[SectionRange; SECTION_COUNT],
    hashes: &[SectionHash; 5],
) -> Result<(), FormatError> {
    let mut pairs: Vec<(&'static str, u64, u64)> = Vec::with_capacity(5);
    pairs.push(("HEADER", 0, HEADER_BYTES as u64));
    for range in ranges.iter().take(4) {
        pairs.push((range.name(), range.offset, range.length));
    }
    for (name, offset, length) in pairs {
        let stored = hashes
            .iter()
            .find(|hash| hash.name == name)
            .ok_or_else(|| {
                FormatError::inconsistent("hash table", format!("no hash for {name}"))
            })?;
        let computed = hash_file_range(file, offset, length)
            .map_err(|error| FormatError::io("<artifact>", error.to_string()))?;
        if computed != stored.sha256 {
            return Err(FormatError::new(FormatErrorKind::HashMismatch {
                section: name.to_string(),
                offset: if name == "HEADER" {
                    0
                } else {
                    ranges
                        .iter()
                        .find(|range| range.name() == name)
                        .map(|range| range.offset)
                        .unwrap_or(0)
                },
                stored: stored.sha256,
                computed,
            }));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Quantization metadata
// ---------------------------------------------------------------------------

/// One tensor's quantization metadata, reserved for a future precision.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantTensor {
    /// Graph value or weight id the entry describes.
    pub tensor: u32,
    /// Scale.
    pub scale: f32,
    /// Zero point.
    pub zero_point: i64,
    /// Axis for per-channel metadata, `None` for per-tensor.
    pub axis: Option<i64>,
}

/// The `QUANT` section: empty but well formed for `F32`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantMetadata {
    precision: Precision,
    tensors: Vec<QuantTensor>,
}

impl QuantMetadata {
    /// Empty metadata for an F32 artifact.
    pub fn empty_f32() -> QuantMetadata {
        QuantMetadata {
            precision: Precision::F32,
            tensors: Vec::new(),
        }
    }

    /// Declared execution precision.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Per-tensor entries (empty for F32).
    pub fn tensors(&self) -> &[QuantTensor] {
        &self.tensors
    }
}

const QUANT_HEADER_BYTES: u64 = 8;
const QUANT_ENTRY_BYTES: u64 = 28;

fn encode_quant(quant: &QuantMetadata) -> Result<Vec<u8>, FormatError> {
    if quant.precision != Precision::F32 {
        return Err(FormatError::new(FormatErrorKind::UnsupportedPrecision {
            found: quant.precision.code(),
            supported: "f32 (code 0)",
        }));
    }
    if !quant.tensors.is_empty() {
        return Err(FormatError::inconsistent(
            "QUANT section",
            "an F32 artifact carries no quantization entries",
        ));
    }
    let mut section = Writer::with_capacity(QUANT_HEADER_BYTES as usize);
    section.u8(quant.precision.code());
    section.u8(0);
    section.u16(0);
    section.u32(0);
    Ok(section.into_bytes())
}

fn decode_quant(section: &[u8], base: u64) -> Result<QuantMetadata, FormatError> {
    let mut reader = Reader::at(section, base);
    let code = reader.u8("quantization precision")?;
    let precision = Precision::from_code(code).ok_or_else(|| {
        FormatError::new(FormatErrorKind::UnsupportedPrecision {
            found: code,
            supported: "f32 (code 0)",
        })
    })?;
    reader.u8("quantization reserved byte")?;
    let reserved = reader.u16("quantization reserved word")?;
    if reserved != 0 {
        return Err(FormatError::new(FormatErrorKind::ReservedField {
            field: "quantization reserved word",
            value: u64::from(reserved),
            offset: base + 2,
        }));
    }
    let count = reader.count("quantization tensor table", MAX_VALUES, QUANT_ENTRY_BYTES)?;
    if precision == Precision::F32 && count != 0 {
        return Err(FormatError::inconsistent(
            "QUANT section",
            format!("an F32 artifact declares {count} quantized tensors"),
        ));
    }
    let mut tensors = Vec::with_capacity(count);
    for index in 0..count {
        let tensor = reader.u32(format!("quant tensor {index}"))?;
        let scale = reader.f32(format!("quant tensor {index} scale"))?;
        let zero_point = reader.i64(format!("quant tensor {index} zero point"))?;
        let axis = reader.i64(format!("quant tensor {index} axis"))?;
        tensors.push(QuantTensor {
            tensor,
            scale,
            zero_point,
            axis: if axis == i64::MIN { None } else { Some(axis) },
        });
    }
    reader.finish("QUANT section")?;
    Ok(QuantMetadata { precision, tensors })
}

// ---------------------------------------------------------------------------
// The loaded artifact
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Backing {
    Mapped(Arc<memmap2::Mmap>),
    Owned(Arc<[u8]>),
}

impl Backing {
    fn as_slice(&self) -> &[u8] {
        match self {
            Backing::Mapped(map) => map.as_ref(),
            Backing::Owned(bytes) => bytes.as_ref(),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

/// A validated artifact whose weights point into a read-only mapping.
///
/// `Artifact` is `Send + Sync`. Cloning shares the one mapping (the `Arc<Mmap>`)
/// and copies the graph and plan tables, never the weight payload; using
/// `Artifact` behind an `Arc` is the usual deployment path. It owns the mapping
/// for as long as any clone lives, which is what lets a session keep weight
/// slices without copying them.
#[derive(Debug, Clone)]
pub struct Artifact {
    backing: Backing,
    header: ArtifactHeader,
    metadata: ArtifactMetadata,
    graph: Graph,
    inputs: Vec<String>,
    outputs: Vec<String>,
    plans: Vec<ArtifactPlan>,
    weights: Vec<ArtifactWeight>,
    weight_payload_offset: u64,
    quant: QuantMetadata,
    hashes: [SectionHash; 5],
    ranges: [SectionRange; SECTION_COUNT],
}

impl Artifact {
    /// Map an artifact file read-only and validate every section.
    ///
    /// The weight payload is hashed from the file in fixed-size reads *before*
    /// the mapping is used, so corruption is reported before a single weight
    /// page is faulted in.
    pub fn open(path: impl AsRef<Path>) -> Result<Artifact, FormatError> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path)
            .map_err(|error| FormatError::io(path.display().to_string(), error.to_string()))?;
        let file_bytes = file
            .metadata()
            .map_err(|error| FormatError::io(path.display().to_string(), error.to_string()))?
            .len();
        let mut header_bytes = [0u8; HEADER_BYTES];
        read_range(&mut file, 0, &mut header_bytes, path)?;
        let header = ArtifactHeader::parse(&header_bytes, file_bytes)?;
        let mut directory = [0u8; DIRECTORY_BYTES];
        read_range(&mut file, header.directory_offset, &mut directory, path)?;
        let ranges = parse_directory(&directory, &header)?;

        let hashes_range = ranges[4];
        let mut hashes_bytes = vec![0u8; hashes_range.length as usize];
        read_range(&mut file, hashes_range.offset, &mut hashes_bytes, path)?;
        let hashes = read_hashes(&hashes_bytes, hashes_range.offset)?;
        verify_hashes_file(&mut file, &ranges, &hashes)?;

        // SAFETY: the file is opened read-only (`File::open`), the mapping is
        // created with `MmapOptions::map` (not `map_mut`), and `Artifact` keeps
        // the `Arc<Mmap>` alive for at least as long as this value and every
        // clone. The mapping's length is the file length the header validated.
        // No code in this module writes through the pointer or constructs a
        // `&mut [u8]` from it. `memmap2` guarantees the mapping is page-aligned;
        // the format's 64-byte offsets preserve alignment for every dtype.
        let mapping = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|error| FormatError::io(path.display().to_string(), error.to_string()))?
        };
        if mapping.len() as u64 != file_bytes {
            return Err(FormatError::length_mismatch(
                "mapped file length",
                file_bytes,
                mapping.len() as u64,
                0,
            ));
        }
        Artifact::decode(Backing::Mapped(Arc::new(mapping)), header, ranges, hashes)
    }

    /// Decode an artifact from bytes already in memory.
    ///
    /// This is the test and embedding seam. `open` is the deployment path and
    /// the only one that provides zero-copy sharing between sessions.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Artifact, FormatError> {
        let file_bytes = bytes.len() as u64;
        let header = ArtifactHeader::parse(&bytes, file_bytes)?;
        let directory = bytes
            .get(
                header.directory_offset as usize
                    ..(header.directory_offset + DIRECTORY_BYTES as u64) as usize,
            )
            .ok_or_else(|| {
                FormatError::truncated_at(
                    "artifact directory",
                    header.directory_offset,
                    DIRECTORY_BYTES as u64,
                    file_bytes,
                )
            })?;
        let ranges = parse_directory(directory, &header)?;
        let hash_range = ranges[4];
        let hash_bytes = section_slice(&bytes, hash_range)?;
        let hashes = read_hashes(hash_bytes, hash_range.offset)?;
        verify_hashes_slice(&bytes, &ranges, &hashes)?;
        Artifact::decode(
            Backing::Owned(Arc::from(bytes.into_boxed_slice())),
            header,
            ranges,
            hashes,
        )
    }

    fn decode(
        backing: Backing,
        header: ArtifactHeader,
        ranges: [SectionRange; SECTION_COUNT],
        hashes: [SectionHash; 5],
    ) -> Result<Artifact, FormatError> {
        if backing.len() as u64 != header.file_bytes {
            return Err(FormatError::length_mismatch(
                "artifact length",
                header.file_bytes,
                backing.len() as u64,
                0,
            ));
        }
        check_section_padding(backing.as_slice(), &ranges)?;
        let graph_bytes = section_slice(backing.as_slice(), ranges[0])?;
        let parts = decode_graph(graph_bytes, ranges[0].offset)?;
        let weights_bytes = section_slice(backing.as_slice(), ranges[2])?;
        let weight_section = decode_weights(weights_bytes, ranges[2].offset)?;
        let quant_bytes = section_slice(backing.as_slice(), ranges[3])?;
        let quant = decode_quant(quant_bytes, ranges[3].offset)?;
        let plans_bytes = section_slice(backing.as_slice(), ranges[1])?;
        let plans = decode_plans(
            plans_bytes,
            ranges[1].offset,
            &parts.graph,
            &parts.inputs,
            &parts.outputs,
            &weight_section.entries,
        )?;
        if plans.len() as u32 != header.target_count {
            return Err(FormatError::inconsistent(
                "PLANS section",
                format!(
                    "the header declares {} target profiles but the section carries {}",
                    header.target_count,
                    plans.len()
                ),
            ));
        }
        let section = &ranges[2];
        let expected_payload = section.offset + weight_section.payload_offset;
        if expected_payload != header.weight_payload_offset
            || weight_section.payload_bytes != header.weight_payload_bytes
        {
            return Err(FormatError::inconsistent(
                "WEIGHTS section",
                format!(
                    "the table places the payload at {expected_payload} ({} bytes) but the header \
                     says {} ({} bytes)",
                    weight_section.payload_bytes,
                    header.weight_payload_offset,
                    header.weight_payload_bytes
                ),
            ));
        }
        Ok(Artifact {
            backing,
            header,
            metadata: parts.metadata,
            graph: parts.graph,
            inputs: parts.inputs,
            outputs: parts.outputs,
            plans,
            weights: weight_section.entries,
            weight_payload_offset: header.weight_payload_offset,
            quant,
            hashes,
            ranges,
        })
    }

    /// The validated header.
    pub fn header(&self) -> &ArtifactHeader {
        &self.header
    }

    /// Graph provenance (source digest is in the header).
    pub fn metadata(&self) -> &ArtifactMetadata {
        &self.metadata
    }

    /// The typed graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Graph input names, in declaration order.
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    /// Graph output names, in declaration order.
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }

    /// The default plan: the first target profile in the file.
    pub fn default_plan(&self) -> &Plan {
        self.plans[0].plan()
    }

    /// The default target name.
    pub fn default_target(&self) -> &str {
        self.plans[0].target()
    }

    /// Every target profile.
    pub fn plans(&self) -> &[ArtifactPlan] {
        &self.plans
    }

    /// The plan for an exact target name.
    pub fn plan_for(&self, target: &str) -> Option<&Plan> {
        self.plans
            .iter()
            .find(|plan| plan.target() == target)
            .map(ArtifactPlan::plan)
    }

    /// The weight table.
    pub fn weights(&self) -> &[ArtifactWeight] {
        &self.weights
    }

    /// Total weight payload bytes.
    pub fn weight_bytes(&self) -> u64 {
        self.weights.iter().map(|weight| weight.len()).sum()
    }

    /// The weight payload's absolute file range.
    pub fn weight_payload_range(&self) -> std::ops::Range<u64> {
        self.weight_payload_offset..self.weight_payload_offset + self.header.weight_payload_bytes
    }

    /// One weight's payload, borrowed from the mapping with no copy.
    pub fn weight_bytes_at(&self, id: crate::inference::model::WeightId) -> Option<&[u8]> {
        let entry = self.weights.get(id.index())?;
        let start = usize::try_from(entry.file_range().start).ok()?;
        let end = usize::try_from(entry.file_range().end).ok()?;
        self.backing.as_slice().get(start..end)
    }

    /// One `f32` weight, borrowed from the mapping with no copy.
    ///
    /// Fails for a non-`f32` weight rather than reinterpreting its bytes.
    pub fn weight_f32(&self, id: crate::inference::model::WeightId) -> Result<&[f32], Fn3Error> {
        let entry = self.weights.get(id.index()).ok_or(Fn3Error::OutOfBounds {
            what: "artifact weight id",
            index: id.index() as u64,
            len: self.weights.len() as u64,
        })?;
        if entry.dtype() != DType::F32 {
            return Err(Fn3Error::UnsupportedDType {
                op: None,
                dtype: entry.dtype(),
                context: "artifact weight_f32",
            });
        }
        let bytes = self.weight_bytes_at(id).ok_or(Fn3Error::OutOfBounds {
            what: "artifact weight bytes",
            index: id.index() as u64,
            len: self.weights.len() as u64,
        })?;
        // SAFETY: the table proves `bytes.len()` is a whole number of `f32`
        // elements (`DType::F32` is four bytes and the decoder rejects a
        // non-multiple). The mapping is page-aligned and the payload start is
        // 64-byte aligned, so the pointer is at least 4-byte aligned. The slice
        // borrows `self`, and the mapping is owned by `self.backing` for at
        // least as long as the returned reference; the mapping is read-only, so
        // no `&mut` alias can exist. `bytemuck` re-checks the length and
        // alignment.
        let slice: &[f32] =
            bytemuck::try_cast_slice(bytes).map_err(|error| Fn3Error::StorageCast {
                element: "f32",
                byte_len: bytes.len(),
                detail: format!("artifact weight {}: {error:?}", id.index()),
            })?;
        Ok(slice)
    }

    /// `true` when `[ptr, ptr + len)` lies inside the mapping.
    ///
    /// This is the evidence a sharing test uses: two sessions' weight pointers
    /// must be equal *and* inside the one mapping.
    pub fn contains_address(&self, ptr: *const u8, len: usize) -> bool {
        let base = self.backing.as_slice().as_ptr() as usize;
        let end = match base.checked_add(self.backing.len()) {
            Some(end) => end,
            None => return false,
        };
        let start = ptr as usize;
        start >= base && start.checked_add(len).is_some_and(|end_ptr| end_ptr <= end)
    }

    /// Every graph value that a plan names as a weight slot, with its weight id.
    pub fn weight_slots(&self) -> Vec<(SlotId, crate::inference::model::WeightId)> {
        let mut slots = Vec::new();
        for (index, slot) in self.default_plan().slots.iter().enumerate() {
            if let SlotSource::Weight(id) = slot.source {
                slots.push((SlotId::new(index as u32), id));
            }
        }
        slots
    }

    /// The stored section hashes.
    pub fn hashes(&self) -> &[SectionHash; 5] {
        &self.hashes
    }

    /// The section ranges.
    pub fn sections(&self) -> &[SectionRange; SECTION_COUNT] {
        &self.ranges
    }

    /// Quantization metadata.
    pub fn quant(&self) -> &QuantMetadata {
        &self.quant
    }

    /// The target plans as compact reports.
    pub fn plan_summaries(&self) -> Vec<PlanSummary> {
        self.plans
            .iter()
            .map(|artifact_plan| PlanSummary::of(artifact_plan.target(), artifact_plan))
            .collect()
    }

    /// Bytes the mapping occupies; the backing bytes, not resident pages.
    pub fn mapped_bytes(&self) -> u64 {
        self.backing.len() as u64
    }
}

/// One target plan, summarised for reports and the CLI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanSummary {
    /// Target name.
    pub target: String,
    /// Execution precision.
    pub precision: Precision,
    /// Segments in the plan.
    pub segments: usize,
    /// Recognised fusions.
    pub fusions: usize,
    /// Layout conversions.
    pub reorders: usize,
    /// Bytes those conversions move per forward.
    pub reorder_bytes: u64,
    /// The arena allocation.
    pub arena_bytes: u64,
    /// Largest live payload.
    pub peak_bytes: u64,
    /// Engine split.
    pub engines: EngineCounts,
    /// Layout policy.
    pub layout: LayoutStrategy,
    /// `segments + sum(removed)`.
    pub unfused_segments: usize,
}

impl PlanSummary {
    fn of(target: &str, artifact_plan: &ArtifactPlan) -> PlanSummary {
        let plan = artifact_plan.plan();
        PlanSummary {
            target: target.to_string(),
            precision: artifact_plan.precision(),
            segments: plan.segments.len(),
            fusions: plan.fusions.len(),
            reorders: plan.reorders.len(),
            reorder_bytes: plan.reorder_bytes as u64,
            arena_bytes: plan.arena_bytes as u64,
            peak_bytes: plan.peak_bytes as u64,
            engines: plan.engine_counts(),
            layout: plan.layout,
            unfused_segments: plan.unfused_segment_count(),
        }
    }
}

fn section_slice(bytes: &[u8], range: SectionRange) -> Result<&[u8], FormatError> {
    let start = usize::try_from(range.offset)
        .map_err(|_| FormatError::limit("section offset", range.offset, usize::MAX as u64))?;
    let end = usize::try_from(range.offset.saturating_add(range.length)).map_err(|_| {
        FormatError::limit(
            "section end",
            range.offset + range.length,
            usize::MAX as u64,
        )
    })?;
    bytes.get(start..end).ok_or_else(|| {
        FormatError::truncated_at(
            format!("{} section", range.name()),
            range.offset,
            range.length,
            bytes.len().saturating_sub(start) as u64,
        )
    })
}

fn read_range(
    file: &mut std::fs::File,
    offset: u64,
    buffer: &mut [u8],
    path: &Path,
) -> Result<(), FormatError> {
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))
        .and_then(|_| file.read_exact(buffer))
        .map_err(|error| {
            FormatError::io(
                path.display().to_string(),
                format!("reading {} bytes at byte {offset}: {error}", buffer.len()),
            )
        })
}

// ---------------------------------------------------------------------------
// The writer
// ---------------------------------------------------------------------------

/// A prepared artifact before it is written.
///
/// Built from a typed import by [`PreparedArtifact::from_imported`], or from
/// already-owned parts by [`PreparedArtifact::from_parts`]. It owns the graph,
/// the weight bytes and the plans; `to_bytes` produces the canonical file.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedArtifact {
    metadata: ArtifactMetadata,
    source_onnx_sha256: [u8; 32],
    graph: Graph,
    inputs: Vec<String>,
    outputs: Vec<String>,
    weights: WeightSet,
    plans: Vec<ArtifactPlan>,
}

impl PreparedArtifact {
    /// Import-derived constructor.
    ///
    /// `targets` is a list of target names (see [`resolve_targets`]); the first
    /// plan is the artifact's default. Every target is planned once, here, and
    /// the plans are serialized as-is.
    pub fn from_imported(
        imported: &ImportedGraph,
        targets: &[&str],
        precision: Precision,
        source_onnx_sha256: [u8; 32],
    ) -> Result<PreparedArtifact, FormatError> {
        if targets.is_empty() {
            return Err(FormatError::inconsistent(
                "target plan",
                "at least one target is required",
            ));
        }
        let graph = imported.graph().clone();
        let mut input_names = Vec::with_capacity(imported.inputs().len());
        for &id in imported.inputs() {
            input_names.push(value_name(&graph, id, "graph input")?);
        }
        let mut output_names = Vec::with_capacity(imported.outputs().len());
        for &id in imported.outputs() {
            output_names.push(value_name(&graph, id, "graph output")?);
        }
        let mut plans = Vec::with_capacity(targets.len());
        for target in targets {
            let plan = plan_for_target(imported, target, precision)?;
            plans.push(ArtifactPlan::new(*target, precision, plan));
        }
        let prepared = PreparedArtifact {
            metadata: ArtifactMetadata::from_import(imported),
            source_onnx_sha256,
            graph,
            inputs: input_names,
            outputs: output_names,
            weights: imported.weights().clone(),
            plans,
        };
        prepared.validate()?;
        Ok(prepared)
    }

    /// Build from already-owned parts.
    ///
    /// This is the seam for a caller that already has a typed graph, a weight
    /// table and plans (the synthetic large-model test uses it).
    pub fn from_parts(
        metadata: ArtifactMetadata,
        source_onnx_sha256: [u8; 32],
        graph: Graph,
        inputs: Vec<String>,
        outputs: Vec<String>,
        weights: WeightSet,
        plans: Vec<ArtifactPlan>,
    ) -> Result<PreparedArtifact, FormatError> {
        let prepared = PreparedArtifact {
            metadata,
            source_onnx_sha256,
            graph,
            inputs,
            outputs,
            weights,
            plans,
        };
        prepared.validate()?;
        Ok(prepared)
    }

    /// Provenance.
    pub fn metadata(&self) -> &ArtifactMetadata {
        &self.metadata
    }

    /// Source ONNX digest.
    pub fn source_onnx_sha256(&self) -> [u8; 32] {
        self.source_onnx_sha256
    }

    /// The graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The weight table.
    pub fn weights(&self) -> &WeightSet {
        &self.weights
    }

    /// The target plans.
    pub fn plans(&self) -> &[ArtifactPlan] {
        &self.plans
    }

    /// Weight payload bytes the artifact will carry (including alignment padding).
    pub fn weight_bytes(&self) -> u64 {
        let table = 24 + 24 * self.weights.len() as u64;
        let offset = (table + 63) & !63;
        let mut payload = 0u64;
        for weight in self.weights.iter() {
            payload = (payload + weight.byte_len() as u64).div_ceil(64) * 64;
        }
        offset + payload
    }

    /// Check every invariant the writer and reader rely on.
    pub fn validate(&self) -> Result<(), FormatError> {
        if self.graph.value_count() == 0 {
            return Err(FormatError::inconsistent(
                "graph",
                "a graph needs at least one value",
            ));
        }
        check_interface(&self.graph, &self.inputs, &self.outputs)?;
        if self.plans.is_empty() {
            return Err(FormatError::inconsistent(
                "target plan",
                "an artifact needs at least one target plan",
            ));
        }
        if self.plans.len() as u64 > MAX_TARGETS {
            return Err(FormatError::limit(
                "target profile count",
                self.plans.len() as u64,
                MAX_TARGETS,
            ));
        }
        for (position, plan) in self.plans.iter().enumerate() {
            if plan.target().is_empty() {
                return Err(FormatError::inconsistent(
                    "target plan",
                    format!("profile {position} has an empty target name"),
                ));
            }
            if self.plans[..position]
                .iter()
                .any(|other| other.target() == plan.target())
            {
                return Err(FormatError::inconsistent(
                    "target plan",
                    format!("target `{}` appears twice", plan.target()),
                ));
            }
            if plan.precision() != Precision::F32 {
                return Err(FormatError::new(FormatErrorKind::UnsupportedPrecision {
                    found: plan.precision().code(),
                    supported: "f32 (code 0)",
                }));
            }
            self.validate_plan(plan.plan(), plan.target())?;
        }
        // Every weight the graph names must have matching dtype and length.
        for value in self.graph.values() {
            let Some(name) = value.name.as_deref() else {
                continue;
            };
            let Some(id) = self.weights.id_of(name) else {
                continue;
            };
            let weight = self.weights.get(id).ok_or_else(|| {
                FormatError::inconsistent("weight table", format!("missing weight `{name}`"))
            })?;
            let bytes = graph_bytes(value)?;
            if weight.dtype() != value.ty.dtype || weight.byte_len() as u64 != bytes {
                return Err(FormatError::inconsistent(
                    "weight table",
                    format!(
                        "weight `{name}` is {} {} bytes but the graph value is {} {bytes} bytes",
                        weight.dtype(),
                        weight.byte_len(),
                        value.ty.dtype
                    ),
                ));
            }
        }
        Ok(())
    }

    fn validate_plan(&self, plan: &Plan, target: &str) -> Result<(), FormatError> {
        if plan.slots.len() != self.graph.value_count() {
            return Err(FormatError::inconsistent(
                "plan slot table",
                format!(
                    "target `{target}` has {} slots but the graph has {} values",
                    plan.slots.len(),
                    self.graph.value_count()
                ),
            ));
        }
        if plan.peak_bytes > plan.arena_bytes {
            return Err(FormatError::inconsistent(
                "plan memory",
                format!("target `{target}` has peak above arena"),
            ));
        }
        for (index, slot) in plan.slots.iter().enumerate() {
            let value = &self.graph.values()[index];
            if slot.value != value.id {
                return Err(FormatError::inconsistent(
                    "plan slot table",
                    format!(
                        "slot {index} names {} but the graph value is {}",
                        slot.value, value.id
                    ),
                ));
            }
            let bytes = graph_bytes(value)?;
            if slot.bytes as u64 != bytes {
                return Err(FormatError::inconsistent(
                    "plan slot table",
                    format!(
                        "slot {index} is {} bytes but its graph value needs {bytes}",
                        slot.bytes
                    ),
                ));
            }
            match slot.source {
                SlotSource::Arena => {
                    if slot.offset % SLOT_ALIGNMENT != 0
                        || slot.offset.saturating_add(slot.bytes) > plan.arena_bytes
                    {
                        return Err(FormatError::inconsistent(
                            "plan slot table",
                            format!(
                                "arena slot {index} at {}..{} is outside the {}-byte arena",
                                slot.offset,
                                slot.offset.saturating_add(slot.bytes),
                                plan.arena_bytes
                            ),
                        ));
                    }
                }
                SlotSource::Weight(id) => {
                    if slot.offset != 0 {
                        return Err(FormatError::inconsistent(
                            "plan slot table",
                            format!("weight slot {index} has a non-zero arena offset"),
                        ));
                    }
                    let weight = self.weights.get(id).ok_or_else(|| {
                        FormatError::inconsistent(
                            "plan slot table",
                            format!(
                                "slot {index} names weight {}, which does not exist",
                                id.index()
                            ),
                        )
                    })?;
                    if weight.dtype() != value.ty.dtype
                        || weight.byte_len() as u64 != slot.bytes as u64
                    {
                        return Err(FormatError::inconsistent(
                            "plan slot table",
                            format!(
                                "slot {index} names weight {}, which is {} {} bytes, but the slot \
                                 is {} {} bytes",
                                id.index(),
                                weight.dtype(),
                                weight.byte_len(),
                                value.ty.dtype,
                                slot.bytes
                            ),
                        ));
                    }
                }
            }
        }
        check_slot_names(&plan.slots, &plan.inputs, &self.inputs, "input", target)?;
        check_slot_names(&plan.slots, &plan.outputs, &self.outputs, "output", target)?;
        for (index, segment) in plan.segments.iter().enumerate() {
            for &id in segment.inputs.iter().chain(segment.outputs.iter()) {
                if id.index() >= plan.slots.len() {
                    return Err(FormatError::inconsistent(
                        "plan segment table",
                        format!(
                            "segment {index} names slot {}, but the plan has {} slots",
                            id.index(),
                            plan.slots.len()
                        ),
                    ));
                }
            }
            if segment.outputs.is_empty() {
                return Err(FormatError::inconsistent(
                    "plan segment table",
                    format!("segment {index} has no outputs"),
                ));
            }
            if let Some(fused) = segment.fused() {
                if fused.chain.len() < 2 || fused.chain[0] != fused.anchor {
                    return Err(FormatError::inconsistent(
                        "fusion",
                        format!("segment {index} has an invalid fused chain"),
                    ));
                }
                if let Some(position) = fused.sum {
                    if position >= segment.inputs.len() {
                        return Err(FormatError::inconsistent(
                            "fusion",
                            format!(
                                "segment {index} names residual operand {position} but has {} \
                                 inputs",
                                segment.inputs.len()
                            ),
                        ));
                    }
                }
            }
        }
        for fusion in &plan.fusions {
            let segment = plan.segments.get(fusion.segment).ok_or_else(|| {
                FormatError::inconsistent(
                    "fusion table",
                    format!("fusion names segment {}", fusion.segment),
                )
            })?;
            if segment.fused() != Some(&fusion.fused) {
                return Err(FormatError::inconsistent(
                    "fusion table",
                    format!("fusion does not match segment {}", fusion.segment),
                ));
            }
        }
        let mut reorder_bytes = 0usize;
        for reorder in &plan.reorders {
            if reorder.segment >= plan.segments.len() || reorder.slot.index() >= plan.slots.len() {
                return Err(FormatError::inconsistent(
                    "reorder table",
                    "a reorder names a segment or slot outside the plan",
                ));
            }
            reorder_bytes = reorder_bytes.checked_add(reorder.bytes).ok_or_else(|| {
                FormatError::inconsistent("reorder table", "the byte sum overflows usize")
            })?;
        }
        if reorder_bytes != plan.reorder_bytes {
            return Err(FormatError::inconsistent(
                "reorder table",
                "reorder_bytes does not match the list",
            ));
        }
        for fallback in &plan.fallbacks {
            let segment = plan.segments.get(fallback.segment).ok_or_else(|| {
                FormatError::inconsistent(
                    "fallback table",
                    format!("fallback names segment {}", fallback.segment),
                )
            })?;
            if segment.op_id() != fallback.op {
                return Err(FormatError::inconsistent(
                    "fallback table",
                    "a fallback names an op its segment does not run",
                ));
            }
        }
        Ok(())
    }

    /// Encode the canonical byte string.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatError> {
        self.validate()?;
        let graph_bytes = encode_graph(&GraphParts {
            metadata: self.metadata.clone(),
            graph: self.graph.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
        })?;
        let plans_bytes = encode_plans(&self.plans)?;
        let weights_bytes = encode_weights(&self.weights)?;
        let quant_bytes = encode_quant(&QuantMetadata::empty_f32())?;
        let hashes_length = 8 + 5 * 40u64;

        let sections: [&[u8]; 4] = [&graph_bytes, &plans_bytes, &weights_bytes, &quant_bytes];
        let mut offsets = [0u64; SECTION_COUNT];
        let mut lengths = [0u64; SECTION_COUNT];
        let mut cursor = (HEADER_BYTES + DIRECTORY_BYTES) as u64;
        for index in 0..SECTION_COUNT {
            offsets[index] = cursor;
            let length = if index == 4 {
                hashes_length
            } else {
                sections[index].len() as u64
            };
            lengths[index] = length;
            cursor = align64(cursor + length)?;
        }
        let file_bytes = cursor;
        if file_bytes > HEADER_BYTES as u64 + MAX_PAYLOAD_BYTES {
            return Err(FormatError::limit(
                "artifact length",
                file_bytes,
                HEADER_BYTES as u64 + MAX_PAYLOAD_BYTES,
            ));
        }
        let weights = decode_weights(&weights_bytes, 0)?;

        let header = encode_header(&ArtifactHeader {
            version: FORMAT_VERSION,
            section_count: SECTION_COUNT as u32,
            target_count: self.plans.len() as u32,
            flags: 0,
            dtype_encoding: DTYPE_ENCODING,
            source_onnx_sha256: self.source_onnx_sha256,
            file_bytes,
            payload_bytes: file_bytes - HEADER_BYTES as u64,
            directory_offset: HEADER_BYTES as u64,
            directory_bytes: DIRECTORY_BYTES as u64,
            weights_offset: offsets[2],
            weights_length: lengths[2],
            weight_payload_offset: offsets[2] + weights.payload_offset,
            weight_payload_bytes: weights.payload_bytes,
        });

        let mut hashes = Writer::with_capacity(hashes_length as usize);
        hashes.u32(5);
        hashes.u32(0);
        for (tag, digest) in [
            (tag::HEADER, hash_slice(&header)),
            (tag::GRAPH, hash_slice(&graph_bytes)),
            (tag::PLANS, hash_slice(&plans_bytes)),
            (tag::WEIGHTS, hash_slice(&weights_bytes)),
            (tag::QUANT, hash_slice(&quant_bytes)),
        ] {
            hashes.u32(tag);
            hashes.u32(0);
            hashes.bytes(&digest);
        }
        let hashes_bytes = hashes.into_bytes();
        debug_assert_eq!(hashes_bytes.len() as u64, hashes_length);

        let mut directory = Writer::with_capacity(DIRECTORY_BYTES);
        directory.u32(SECTION_COUNT as u32);
        directory.u32(0);
        for index in 0..SECTION_COUNT {
            directory.u32(tag::ALL[index]);
            directory.u32(0);
            directory.u64(offsets[index]);
            directory.u64(lengths[index]);
        }
        let directory = directory.into_bytes();
        debug_assert_eq!(directory.len(), DIRECTORY_BYTES);

        let file_len = usize::try_from(file_bytes)
            .map_err(|_| FormatError::limit("artifact length", file_bytes, usize::MAX as u64))?;
        let mut file = vec![0u8; file_len];
        file[..HEADER_BYTES].copy_from_slice(&header);
        file[HEADER_BYTES..HEADER_BYTES + DIRECTORY_BYTES].copy_from_slice(&directory);
        for index in 0..SECTION_COUNT {
            let bytes: &[u8] = if index == 4 {
                &hashes_bytes
            } else {
                sections[index]
            };
            let start = offsets[index] as usize;
            file[start..start + bytes.len()].copy_from_slice(bytes);
        }
        Ok(file)
    }

    /// Write the canonical file, replacing any existing one.
    pub fn write_file(&self, path: impl AsRef<Path>) -> Result<(), FormatError> {
        let path = path.as_ref();
        let bytes = self.to_bytes()?;
        std::fs::write(path, &bytes)
            .map_err(|error| FormatError::io(path.display().to_string(), error.to_string()))
    }
}

fn encode_header(header: &ArtifactHeader) -> Vec<u8> {
    let mut bytes = Writer::with_capacity(HEADER_BYTES);
    bytes.bytes(&MAGIC);
    bytes.u16(header.version);
    bytes.u16(HEADER_BYTES as u16);
    bytes.u32(header.section_count);
    bytes.u32(header.target_count);
    bytes.u32(header.flags);
    bytes.u32(header.dtype_encoding);
    bytes.u32(0);
    bytes.bytes(&header.source_onnx_sha256);
    bytes.u64(header.file_bytes);
    bytes.u64(header.payload_bytes);
    bytes.u64(header.directory_offset);
    bytes.u64(header.directory_bytes);
    bytes.u64(header.weights_offset);
    bytes.u64(header.weights_length);
    bytes.u64(header.weight_payload_offset);
    bytes.u64(header.weight_payload_bytes);
    let bytes = bytes.into_bytes();
    debug_assert_eq!(bytes.len(), HEADER_BYTES);
    bytes
}

fn value_name(
    graph: &Graph,
    id: crate::inference::ir::ValueId,
    what: &str,
) -> Result<String, FormatError> {
    graph
        .value(id)
        .ok_or_else(|| {
            FormatError::inconsistent(what, format!("{what} handle {id} is out of range"))
        })?
        .name
        .clone()
        .ok_or_else(|| FormatError::inconsistent(what, format!("{what} {id} has no name")))
}

fn graph_bytes(value: &crate::inference::ir::Value) -> Result<u64, FormatError> {
    let shape = value.ty.static_shape().ok_or_else(|| {
        FormatError::inconsistent(
            "graph value",
            format!(
                "{} is {}; an artifact needs static extents",
                value.id, value.ty
            ),
        )
    })?;
    let elements = shape
        .numel()
        .map_err(|error| FormatError::contract("graph value bytes", error))?;
    elements
        .checked_mul(value.ty.dtype.size_bytes() as u64)
        .ok_or_else(|| {
            FormatError::inconsistent(
                "graph value",
                format!("{} overflows its byte count", value.id),
            )
        })
}

fn check_slot_names(
    slots: &[Slot],
    ids: &[SlotId],
    names: &[String],
    kind: &str,
    target: &str,
) -> Result<(), FormatError> {
    if ids.len() != names.len() {
        return Err(FormatError::inconsistent(
            "plan interface",
            format!(
                "target `{target}` has {} {kind} slots but the graph declares {}",
                ids.len(),
                names.len()
            ),
        ));
    }
    for (position, (id, name)) in ids.iter().zip(names).enumerate() {
        if slots[id.index()].name.as_deref() != Some(name.as_str()) {
            return Err(FormatError::inconsistent(
                "plan interface",
                format!(
                    "target `{target}` {kind} {position} names `{name}` but the slot is {:?}",
                    slots[id.index()].name
                ),
            ));
        }
    }
    Ok(())
}

fn hex(bytes: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in bytes {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

// ---------------------------------------------------------------------------
// inspect
// ---------------------------------------------------------------------------

/// One section, as `inspect` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionInfo {
    /// `HEADER`, `GRAPH`, `PLANS`, `WEIGHTS` or `QUANT`.
    pub name: &'static str,
    /// Absolute file offset.
    pub offset: u64,
    /// Encoded length.
    pub length: u64,
    /// Stored SHA-256.
    pub sha256: [u8; 32],
    /// `true` when `inspect` re-read and matched the hash.
    pub verified: bool,
}

impl SectionInfo {
    /// Lowercase hexadecimal digest.
    pub fn sha256_hex(&self) -> String {
        hex(&self.sha256)
    }
}

/// What `inspect` learned without mapping or executing the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ArtifactInspection {
    /// Path inspected.
    pub path: PathBuf,
    /// File length.
    pub file_bytes: u64,
    /// Format version.
    pub format_version: u16,
    /// Target profiles.
    pub target_count: u32,
    /// Source ONNX digest.
    pub source_onnx_sha256: [u8; 32],
    /// Graph provenance (same metadata a prepared artifact carries).
    pub metadata: ArtifactMetadata,
    /// Nodes in the typed graph.
    pub nodes: usize,
    /// Values in the typed graph.
    pub values: usize,
    /// Weights in the table.
    pub weights: usize,
    /// Graph inputs.
    pub inputs: usize,
    /// Graph outputs.
    pub outputs: usize,
    /// Sum of the weight payloads.
    pub weight_bytes: u64,
    /// Compact per-target plans.
    pub plans: Vec<PlanSummary>,
    /// Quantization metadata.
    pub quant: QuantMetadata,
    /// Section inventory.
    pub sections: Vec<SectionInfo>,
    /// Bytes `inspect` actually read.
    pub bytes_read: u64,
    /// Always zero: `inspect` never maps the file.
    pub mapped_bytes: u64,
}

/// Inspect an artifact file without mapping it.
///
/// `GRAPH`, `PLANS`, `QUANT` and `HASHES` are read into memory; the weight
/// table is read from the first 64-aligned prefix of `WEIGHTS`; the weight
/// payload is never read and no `Mmap` is constructed. `WEIGHTS`' hash is
/// reported but not marked verified, because re-reading a 200 MB payload is not
/// what "inspect before trusting" should cost.
pub fn inspect_file(path: impl AsRef<Path>) -> Result<ArtifactInspection, FormatError> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path)
        .map_err(|error| FormatError::io(path.display().to_string(), error.to_string()))?;
    let file_bytes = file
        .metadata()
        .map_err(|error| FormatError::io(path.display().to_string(), error.to_string()))?
        .len();
    let mut bytes_read = 0u64;
    let mut header_bytes = [0u8; HEADER_BYTES];
    read_range(&mut file, 0, &mut header_bytes, path)?;
    bytes_read += HEADER_BYTES as u64;
    let header = ArtifactHeader::parse(&header_bytes, file_bytes)?;
    let mut directory = [0u8; DIRECTORY_BYTES];
    read_range(&mut file, header.directory_offset, &mut directory, path)?;
    bytes_read += DIRECTORY_BYTES as u64;
    let ranges = parse_directory(&directory, &header)?;
    let mut hashes_bytes = vec![0u8; ranges[4].length as usize];
    read_range(&mut file, ranges[4].offset, &mut hashes_bytes, path)?;
    bytes_read += ranges[4].length;
    let hashes = read_hashes(&hashes_bytes, ranges[4].offset)?;

    let mut graph_bytes = read_metadata_section(&mut file, ranges[0], path, &mut bytes_read)?;
    graph_bytes.shrink_to_fit();
    let parts = decode_graph(&graph_bytes, ranges[0].offset)?;
    let plans_bytes = read_metadata_section(&mut file, ranges[1], path, &mut bytes_read)?;
    let quant_bytes = read_metadata_section(&mut file, ranges[3], path, &mut bytes_read)?;
    let quant = decode_quant(&quant_bytes, ranges[3].offset)?;

    // Read only the table prefix of WEIGHTS, never the payload.
    let mut table_header = [0u8; 24];
    read_range(&mut file, ranges[2].offset, &mut table_header, path)?;
    bytes_read += 24;
    let count = u32::from_le_bytes([
        table_header[0],
        table_header[1],
        table_header[2],
        table_header[3],
    ]) as u64;
    if count > MAX_WEIGHTS {
        return Err(FormatError::limit("weight table", count, MAX_WEIGHTS));
    }
    let table_len = align64(24 + 24 * count)?;
    if table_len > ranges[2].length {
        return Err(FormatError::truncated_at(
            "WEIGHTS table",
            ranges[2].offset,
            table_len,
            ranges[2].length,
        ));
    }
    let mut table = vec![0u8; table_len as usize];
    read_range(&mut file, ranges[2].offset, &mut table, path)?;
    bytes_read += table_len;
    let weight_section = decode_weight_table(&table, ranges[2].length, ranges[2].offset)?;
    let plans = decode_plans(
        &plans_bytes,
        ranges[1].offset,
        &parts.graph,
        &parts.inputs,
        &parts.outputs,
        &weight_section.entries,
    )?;
    if plans.len() as u32 != header.target_count {
        return Err(FormatError::inconsistent(
            "PLANS section",
            format!(
                "the header declares {} target profiles but the section carries {}",
                header.target_count,
                plans.len()
            ),
        ));
    }

    let mut verified = [true; 5];
    // The weight payload was deliberately not read.
    verified[3] = false;
    let mut section_info = Vec::with_capacity(6);
    for (position, name) in ["HEADER", "GRAPH", "PLANS", "WEIGHTS", "QUANT"]
        .into_iter()
        .enumerate()
    {
        let (offset, length, data): (u64, u64, Option<&[u8]>) = match position {
            0 => (0, HEADER_BYTES as u64, Some(&header_bytes)),
            1 => (ranges[0].offset, ranges[0].length, Some(&graph_bytes)),
            2 => (ranges[1].offset, ranges[1].length, Some(&plans_bytes)),
            3 => (ranges[2].offset, ranges[2].length, None),
            4 => (ranges[3].offset, ranges[3].length, Some(&quant_bytes)),
            _ => unreachable!(),
        };
        let stored = hashes[position].sha256;
        let stored_ok = match data {
            Some(data) => hash_slice(data) == stored,
            None => true,
        };
        if !stored_ok {
            return Err(FormatError::new(FormatErrorKind::HashMismatch {
                section: name.to_string(),
                offset,
                stored,
                computed: hash_slice(data.expect("matched above")),
            }));
        }
        section_info.push(SectionInfo {
            name,
            offset,
            length,
            sha256: stored,
            verified: verified[position],
        });
    }

    Ok(ArtifactInspection {
        path: path.to_path_buf(),
        file_bytes,
        format_version: header.version,
        target_count: header.target_count,
        source_onnx_sha256: header.source_onnx_sha256,
        metadata: parts.metadata.clone(),
        nodes: parts.graph.node_count(),
        values: parts.graph.value_count(),
        weights: weight_section.entries.len(),
        inputs: parts.inputs.len(),
        outputs: parts.outputs.len(),
        weight_bytes: weight_section.entries.iter().map(|entry| entry.len()).sum(),
        plans: plans
            .iter()
            .map(|plan| PlanSummary::of(plan.target(), plan))
            .collect(),
        quant,
        sections: section_info,
        bytes_read,
        mapped_bytes: 0,
    })
}

fn read_metadata_section(
    file: &mut std::fs::File,
    range: SectionRange,
    path: &Path,
    bytes_read: &mut u64,
) -> Result<Vec<u8>, FormatError> {
    if range.length > MAX_INSPECT_SECTION_BYTES {
        return Err(FormatError::limit(
            format!("{} section (inspect)", range.name()),
            range.length,
            MAX_INSPECT_SECTION_BYTES,
        ));
    }
    let mut bytes = vec![0u8; range.length as usize];
    read_range(file, range.offset, &mut bytes, path)?;
    *bytes_read += range.length;
    Ok(bytes)
}
