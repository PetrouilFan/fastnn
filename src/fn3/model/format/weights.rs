//! The `WEIGHTS` section: one contiguous, 64-byte-aligned payload.
//!
//! # Layout
//!
//! ```text
//! 0..4    u32 weight count
//! 4..8    u32 reserved (0)
//! 8..16   u64 payload offset from the section start (64-byte aligned)
//! 16..24  u64 payload length in bytes
//! 24..    count × { u64 offset, u64 len, u32 dtype, u32 reserved }
//! ...     zero padding to a 64-byte boundary
//! ...     payload; each weight starts on a 64-byte boundary
//! ```
//!
//! The offsets are relative to the payload start. Because the section itself
//! starts on a 64-byte boundary of a page-aligned mapping, every weight's first
//! byte is 64-byte aligned in memory for every dtype. That is what makes a
//! zero-copy [`&[f32]`] possible: the payload is not re-aligned, repacked or
//! copied when a session opens.
//!
//! # Why one payload and not one blob per weight
//!
//! One mapping, one page-cache footprint, one file range to hash. The table is
//! metadata; the payload is exactly the bytes the model carried, padded between
//! entries so each starts aligned.
//!
//! # Reader rules
//!
//! The count is checked against [`MAX_WEIGHTS`] and the remaining table bytes;
//! every offset and length is checked against the payload length; the payload
//! offset is required to equal `align64(table end)`, every entry offset to equal
//! `align64(previous end)`, and the last end to equal the payload length. Those
//! equalities make the encoding canonical: one model, one byte string.

use crate::inference::dtype::DType;
use crate::inference::model::WeightSet;

use super::error::{FormatError, FormatErrorKind};
use super::wire::{Reader, Writer};
use super::{MAX_PAYLOAD_BYTES, MAX_WEIGHTS};

/// One weight's metadata in a loaded artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactWeight {
    dtype: DType,
    bytes: u64,
    file_offset: u64,
}

impl ArtifactWeight {
    /// Element type.
    pub fn dtype(self) -> DType {
        self.dtype
    }

    /// Payload length in bytes.
    pub fn len(self) -> u64 {
        self.bytes
    }

    /// `true` for an empty weight (a zero-extent tensor).
    pub fn is_empty(self) -> bool {
        self.bytes == 0
    }

    /// Absolute byte range of this weight in the artifact file.
    pub fn file_range(self) -> std::ops::Range<u64> {
        self.file_offset..self.file_offset + self.bytes
    }
}

/// The decoded weight table plus the payload's location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WeightSection {
    pub(super) entries: Vec<ArtifactWeight>,
    pub(super) payload_offset: u64,
    pub(super) payload_bytes: u64,
}

const HEADER_BYTES: u64 = 24;
const ENTRY_BYTES: u64 = 24;

fn align64(value: u64) -> Result<u64, FormatError> {
    value
        .checked_add(63)
        .map(|value| value & !63)
        .ok_or_else(|| FormatError::limit("64-byte alignment", value, u64::MAX))
}

/// Encode the weight table and the one contiguous payload.
pub(super) fn encode_weights(weights: &WeightSet) -> Result<Vec<u8>, FormatError> {
    let count = u64::try_from(weights.len())
        .map_err(|_| FormatError::limit("weight count", weights.len() as u64, u32::MAX as u64))?;
    if count > MAX_WEIGHTS {
        return Err(FormatError::limit("weight count", count, MAX_WEIGHTS));
    }
    let table_bytes = HEADER_BYTES
        .checked_add(count.saturating_mul(ENTRY_BYTES))
        .ok_or_else(|| FormatError::limit("weight table", u64::MAX, MAX_PAYLOAD_BYTES))?;
    let payload_offset = align64(table_bytes)?;

    let mut payload = Writer::with_capacity(weights.total_bytes() as usize);
    for weight in weights.iter() {
        payload.pad_to_64();
        if payload.len() as u64 % 64 != 0 {
            return Err(FormatError::inconsistent(
                "weight payload",
                "an entry did not start on a 64-byte boundary",
            ));
        }
        payload.bytes(weight.bytes());
    }
    payload.pad_to_64();
    let payload = payload.into_bytes();
    let payload_bytes = payload.len() as u64;
    if payload_offset.saturating_add(payload_bytes) > MAX_PAYLOAD_BYTES {
        return Err(FormatError::limit(
            "weight payload bytes",
            payload_offset + payload_bytes,
            MAX_PAYLOAD_BYTES,
        ));
    }

    let mut section =
        Writer::with_capacity(usize::try_from(payload_offset + payload_bytes).unwrap_or(0));
    section.u32(count as u32);
    section.u32(0);
    section.u64(payload_offset);
    section.u64(payload_bytes);

    let mut offset = 0u64;
    for weight in weights.iter() {
        section.u64(offset);
        section.u64(weight.byte_len() as u64);
        section.u32(u32::from(weight.dtype().code()));
        section.u32(0);
        offset = align64(offset + weight.byte_len() as u64)?;
    }
    section.pad_to_64();
    debug_assert_eq!(section.len() as u64, payload_offset);
    section.bytes(&payload);
    Ok(section.into_bytes())
}

/// Decode a complete `WEIGHTS` section from the mapping.
pub(super) fn decode_weights(section: &[u8], base: u64) -> Result<WeightSection, FormatError> {
    let section_len = section.len() as u64;
    if section_len < HEADER_BYTES {
        return Err(FormatError::truncated_at(
            "WEIGHTS header",
            base,
            HEADER_BYTES,
            section_len,
        ));
    }
    // Read the four header words without asking the count for allocation
    // headroom: the table prefix is sliced below and validated there.
    let count = u32::from_le_bytes([section[0], section[1], section[2], section[3]]) as u64;
    if count > MAX_WEIGHTS {
        return Err(FormatError::limit("weight table", count, MAX_WEIGHTS));
    }
    let mut header_reader = Reader::at(&section[..HEADER_BYTES as usize], base);
    header_reader.u32("weight count")?;
    header_reader.zero_u32("weight table reserved word")?;
    let payload_offset = header_reader.u64("weight payload offset")?;
    let payload_bytes = header_reader.u64("weight payload bytes")?;
    let table_bytes = HEADER_BYTES.saturating_add(count.saturating_mul(ENTRY_BYTES));
    let expected_offset = align64(table_bytes)?;
    if payload_offset != expected_offset {
        return Err(FormatError::malformed_at(
            "WEIGHTS",
            base + 8,
            format!(
                "payload offset is {payload_offset}, expected {expected_offset} for {count} \
                 entries"
            ),
        ));
    }
    if payload_offset > section_len {
        return Err(FormatError::truncated_at(
            "WEIGHTS payload",
            base + 8,
            payload_offset,
            section_len,
        ));
    }
    if payload_offset.saturating_add(payload_bytes) != section_len {
        return Err(FormatError::length_mismatch(
            "WEIGHTS payload",
            payload_offset.saturating_add(payload_bytes),
            section_len,
            base + 16,
        ));
    }
    if payload_bytes > MAX_PAYLOAD_BYTES {
        return Err(FormatError::limit(
            "weight payload bytes",
            payload_bytes,
            MAX_PAYLOAD_BYTES,
        ));
    }
    let table_end = payload_offset as usize;
    decode_weight_table(&section[..table_end], section_len, base)
}

/// Decode only the table part of a `WEIGHTS` section.
///
/// `table` is the section prefix containing the header, the entries and the
/// padding; `section_len` is the declared length of the whole section. `inspect`
/// uses this with a prefix read from the file, so the weight payload is never
/// read or mapped.
pub(super) fn decode_weight_table(
    table: &[u8],
    section_len: u64,
    base: u64,
) -> Result<WeightSection, FormatError> {
    if table.len() as u64 > section_len {
        return Err(FormatError::truncated_at(
            "WEIGHTS table",
            base,
            table.len() as u64,
            section_len,
        ));
    }
    let mut reader = Reader::at(table, base);
    let count = reader.count("weight table", MAX_WEIGHTS, ENTRY_BYTES)? as u64;
    reader.zero_u32("weight table reserved word")?;
    let payload_offset = reader.u64("weight payload offset")?;
    let payload_bytes = reader.u64("weight payload bytes")?;
    let table_bytes = HEADER_BYTES.saturating_add(count.saturating_mul(ENTRY_BYTES));
    let expected_offset = align64(table_bytes)?;
    if payload_offset != expected_offset {
        return Err(FormatError::malformed_at(
            "WEIGHTS",
            base + 8,
            format!(
                "payload offset is {payload_offset}, expected {expected_offset} for {count} \
                 entries"
            ),
        ));
    }
    if payload_offset > section_len || payload_offset.saturating_add(payload_bytes) != section_len {
        return Err(FormatError::length_mismatch(
            "WEIGHTS payload",
            payload_offset.saturating_add(payload_bytes),
            section_len,
            base + 16,
        ));
    }
    if payload_bytes > MAX_PAYLOAD_BYTES {
        return Err(FormatError::limit(
            "weight payload bytes",
            payload_bytes,
            MAX_PAYLOAD_BYTES,
        ));
    }
    if (table.len() as u64) < table_bytes {
        return Err(FormatError::truncated_at(
            "WEIGHTS table",
            base,
            table_bytes,
            table.len() as u64,
        ));
    }

    let mut entries = Vec::with_capacity(count as usize);
    let mut expected_entry_offset = 0u64;
    let mut index = 0u64;
    while index < count {
        let entry_offset = base + HEADER_BYTES + index * ENTRY_BYTES;
        let offset = reader.u64("weight offset")?;
        let len = reader.u64("weight length")?;
        let dtype_code = reader.u32("weight dtype")?;
        let dtype = u8::try_from(dtype_code)
            .map_err(|_| {
                FormatError::malformed_at(
                    "WEIGHTS",
                    entry_offset + 16,
                    format!("weight dtype code {dtype_code} does not fit in a byte"),
                )
            })
            .and_then(|code| {
                DType::from_code(code, "WEIGHTS entry")
                    .map_err(|error| FormatError::new(FormatErrorKind::UnknownDType(error)))
            })?;
        let reserved = reader.u32("weight entry reserved word")?;
        if reserved != 0 {
            return Err(FormatError::new(FormatErrorKind::ReservedField {
                field: "weight entry reserved word",
                value: u64::from(reserved),
                offset: entry_offset + 20,
            }));
        }
        if offset != expected_entry_offset {
            return Err(FormatError::malformed_at(
                "WEIGHTS",
                entry_offset,
                format!(
                    "weight {index} starts at payload byte {offset}, expected \
                     {expected_entry_offset}"
                ),
            ));
        }
        if offset % 64 != 0 {
            return Err(FormatError::malformed_at(
                "WEIGHTS",
                entry_offset,
                format!("weight {index} starts at byte {offset}, which is not 64-byte aligned"),
            ));
        }
        if !len.is_multiple_of(dtype.size_bytes() as u64) {
            return Err(FormatError::malformed_at(
                "WEIGHTS",
                entry_offset + 8,
                format!("weight {index} is {len} bytes, not a whole number of {dtype} elements"),
            ));
        }
        let end = offset.checked_add(len).ok_or_else(|| {
            FormatError::malformed_at(
                "WEIGHTS",
                entry_offset,
                format!("weight {index} length overflows: {offset} + {len}"),
            )
        })?;
        if end > payload_bytes {
            return Err(FormatError::malformed_at(
                "WEIGHTS",
                entry_offset,
                format!(
                    "weight {index} ends at payload byte {end}, past the {payload_bytes}-byte \
                     payload"
                ),
            ));
        }
        entries.push(ArtifactWeight {
            dtype,
            bytes: len,
            file_offset: base + payload_offset + offset,
        });
        expected_entry_offset = align64(end)?;
        index += 1;
    }
    if expected_entry_offset != payload_bytes {
        return Err(FormatError::malformed_at(
            "WEIGHTS",
            base + HEADER_BYTES,
            format!(
                "the last weight ends at {expected_entry_offset} in the payload, but the payload \
                 declares {payload_bytes} bytes"
            ),
        ));
    }
    if (table.len() as u64) > payload_offset {
        return Err(FormatError::malformed_at(
            "WEIGHTS",
            base + payload_offset,
            "the table prefix contains bytes past the declared payload offset",
        ));
    }
    // Padding is required to be zero so the encoding is canonical.
    reader.pad_to_64("WEIGHTS table padding")?;
    reader.finish("WEIGHTS table")?;
    Ok(WeightSection {
        entries,
        payload_offset,
        payload_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::model::{WeightSource, WeightTensor};
    use crate::inference::shape::Shape;

    #[test]
    fn the_payload_is_aligned_and_canonical() {
        let weights = WeightSet::new(vec![
            WeightTensor::new(
                "a",
                DType::F32,
                Shape::from_dims([2, 3]),
                vec![0u8; 24],
                WeightSource::Initializer,
            )
            .unwrap(),
            WeightTensor::new(
                "b",
                DType::I64,
                Shape::from_dims([1]),
                vec![0u8; 8],
                WeightSource::Initializer,
            )
            .unwrap(),
        ])
        .unwrap();
        let section = encode_weights(&weights).unwrap();
        let decoded = decode_weights(&section, 4096).unwrap();
        assert_eq!(decoded.entries.len(), 2);
        assert_eq!(decoded.entries[0].len(), 24);
        assert_eq!(decoded.payload_offset, 128);
        assert_eq!(decoded.entries[0].file_range().start, 4096 + 128 + 0);
        assert_eq!(decoded.entries[1].file_range().start, 4096 + 128 + 64);
        assert_eq!(decoded.payload_bytes, 128);
        assert_eq!(section.len(), 128 + 128);
    }

    #[test]
    fn a_payload_offset_that_disagrees_with_the_table_is_rejected() {
        let weights = WeightSet::new(vec![WeightTensor::new(
            "a",
            DType::F32,
            Shape::from_dims([1]),
            vec![0u8; 4],
            WeightSource::Initializer,
        )
        .unwrap()])
        .unwrap();
        let mut section = encode_weights(&weights).unwrap();
        section[8] ^= 0x40;
        let error = decode_weights(&section, 0).unwrap_err();
        assert!(error.to_string().contains("payload offset"), "{error}");
    }
}
