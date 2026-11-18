//! Bounds-checked little-endian wire helpers and the payload checksum.
//!
//! Everything here is deliberately small and dependency-free: the format is
//! hand-framed, so the reader owns its own cursor instead of delegating to a
//! serialization crate. Two rules are enforced by construction:
//!
//! 1. **No allocation without a validated length.** [`Reader::take`] checks the
//!    length against the remaining input *before* slicing, and every count is
//!    validated by [`Reader::count`] against both a policy cap and the number of
//!    bytes that could possibly encode it. A hostile count therefore costs one
//!    multiplication and one comparison, never memory.
//! 2. **No unchecked arithmetic.** Every offset is added with `checked_add`, so
//!    a 32-bit target reports [`FormatErrorKind::Truncated`] instead of wrapping.
//!
//! The checksum is CRC-32 (IEEE 802.3, reflected, polynomial `0xEDB88320`,
//! initial value `0xFFFF_FFFF`, final XOR `0xFFFF_FFFF`, check value
//! `0xCBF43926` for `"123456789"`). CRC-32 — not a hash — because the job is to
//! detect accidental corruption (a truncated copy, a flipped bit on a network
//! share), and a standard polynomial with a published check value is auditable
//! without trusting this file. It is explicitly *not* a security boundary: an
//! attacker who controls the bytes controls the checksum too, which is why the
//! reader validates structure independently of the checksum.

use super::error::{FormatError, FormatErrorKind};

/// Largest name or string this format accepts, in bytes.
///
/// ONNX value names in a real decoder are well under 100 bytes; the cap keeps a
/// declared length from being a memory request.
pub(super) const MAX_STRING_BYTES: u64 = 4096;

/// Write side: a byte vector with little-endian field helpers.
pub(super) struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    /// An empty writer with room for `capacity` bytes.
    ///
    /// `capacity` always comes from data this process already holds, never from
    /// an untrusted count.
    pub(super) fn with_capacity(capacity: usize) -> Writer {
        Writer {
            bytes: Vec::with_capacity(capacity),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.bytes.len()
    }

    pub(super) fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    pub(super) fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub(super) fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub(super) fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Copy a byte blob, used for weight payloads and finished sections.
    pub(super) fn bytes(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    pub(super) fn i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Store an `f32` by its bit pattern.
    ///
    /// Bit patterns, not decimal text: a rounded or reformatted float would make
    /// two equal models serialize differently, and `NaN` payloads must survive
    /// unchanged.
    pub(super) fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    /// A length-prefixed UTF-8 string.
    pub(super) fn string(&mut self, value: &str) -> Result<(), FormatError> {
        let bytes = value.as_bytes();
        let len = bytes.len() as u64;
        if len > MAX_STRING_BYTES {
            return Err(FormatError::limit("string length", len, MAX_STRING_BYTES));
        }
        self.u32(len as u32);
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    /// Zero-pad to the next multiple of eight.
    pub(super) fn pad_to_8(&mut self) {
        while !self.bytes.len().is_multiple_of(8) {
            self.bytes.push(0);
        }
    }

    pub(super) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Read side: a cursor over one byte range with checked reads.
pub(super) struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Reader<'a> {
        Reader { bytes, pos: 0 }
    }

    /// Bytes consumed so far, i.e. the position that alignment is measured from.
    pub(super) fn pos(&self) -> usize {
        self.pos
    }

    pub(super) fn remaining(&self) -> u64 {
        (self.bytes.len() - self.pos) as u64
    }

    /// Take exactly `len` bytes, or report how many were missing.
    pub(super) fn take(
        &mut self,
        len: u64,
        what: impl Into<String>,
    ) -> Result<&'a [u8], FormatError> {
        let remaining = self.remaining();
        if len > remaining {
            return Err(FormatError::truncated(what, len, remaining));
        }
        let len = len as usize;
        let start = self.pos;
        let end = start
            .checked_add(len)
            .ok_or_else(|| FormatError::truncated(what, u64::MAX, remaining))?;
        // `end <= bytes.len()` follows from `len <= remaining`.
        self.pos = end;
        Ok(&self.bytes[start..end])
    }

    pub(super) fn u8(&mut self, what: impl Into<String>) -> Result<u8, FormatError> {
        Ok(self.take(1, what)?[0])
    }

    pub(super) fn u16(&mut self, what: impl Into<String>) -> Result<u16, FormatError> {
        let bytes = self.take(2, what)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub(super) fn u32(&mut self, what: impl Into<String>) -> Result<u32, FormatError> {
        let bytes = self.take(4, what)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(super) fn u64(&mut self, what: impl Into<String>) -> Result<u64, FormatError> {
        let bytes = self.take(8, what)?;
        let mut field = [0u8; 8];
        field.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(field))
    }

    pub(super) fn i64(&mut self, what: impl Into<String>) -> Result<i64, FormatError> {
        Ok(self.u64(what)? as i64)
    }

    pub(super) fn f32(&mut self, what: impl Into<String>) -> Result<f32, FormatError> {
        Ok(f32::from_bits(self.u32(what)?))
    }

    /// A length-prefixed UTF-8 string.
    ///
    /// The length is validated against the policy cap and the remaining bytes
    /// before it is used, and UTF-8 is validated before the `String` is
    /// allocated.
    pub(super) fn string(&mut self, what: impl Into<String>) -> Result<String, FormatError> {
        let what = what.into();
        let len = self.u32(what.clone())? as u64;
        if len > MAX_STRING_BYTES {
            return Err(FormatError::limit(
                format!("{what} length"),
                len,
                MAX_STRING_BYTES,
            ));
        }
        let bytes = self.take(len, what.clone())?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|error| FormatError::malformed(what, format!("not valid UTF-8: {error}")))
    }

    /// Read a `u32` entry count and prove the entries can fit before allocating.
    ///
    /// `limit` is the policy cap this build enforces; `min_entry_bytes` is the
    /// smallest possible encoded size of one entry. A count that would need more
    /// bytes than remain is rejected here, so `Vec::with_capacity(count)` can
    /// never be driven by an untrusted number.
    pub(super) fn count(
        &mut self,
        what: impl Into<String>,
        limit: u64,
        min_entry_bytes: u64,
    ) -> Result<usize, FormatError> {
        let what = what.into();
        let count = self.u32(what.clone())? as u64;
        if count > limit {
            return Err(FormatError::limit(what, count, limit));
        }
        let needed = count.saturating_mul(min_entry_bytes);
        let remaining = self.remaining();
        if needed > remaining {
            return Err(FormatError::truncated(what, needed, remaining));
        }
        Ok(count as usize)
    }

    /// Skip to the next multiple of eight, requiring the padding to be zero.
    ///
    /// Requiring zeros makes the encoding canonical: there is exactly one valid
    /// byte string for a given model, so a re-write is byte-identical.
    pub(super) fn pad_to_8(&mut self, what: impl Into<String>) -> Result<(), FormatError> {
        let padding = (8 - (self.pos % 8)) % 8;
        if padding == 0 {
            return Ok(());
        }
        let what = what.into();
        let bytes = self.take(padding as u64, what.clone())?;
        if let Some(position) = bytes.iter().position(|&byte| byte != 0) {
            return Err(FormatError::malformed(
                what,
                format!(
                    "alignment padding byte {position} is {:#04x}, expected 0",
                    bytes[position]
                ),
            ));
        }
        Ok(())
    }

    /// Require that the whole slice was consumed.
    pub(super) fn finish(&self, what: impl Into<String>) -> Result<(), FormatError> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(FormatError::malformed(
                what,
                format!("{} trailing bytes after the last field", self.remaining()),
            ))
        }
    }

    /// A reserved field that must be zero.
    pub(super) fn zero_u32(&mut self, field: &'static str) -> Result<(), FormatError> {
        let value = self.u32(field)?;
        if value == 0 {
            Ok(())
        } else {
            Err(FormatError::new(FormatErrorKind::ReservedField {
                field,
                value: u64::from(value),
            }))
        }
    }
}

/// CRC-32 table for the reflected IEEE polynomial `0xEDB88320`.
fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for (index, slot) in table.iter_mut().enumerate() {
        let mut value = index as u32;
        for _ in 0..8 {
            value = if value & 1 == 1 {
                (value >> 1) ^ 0xEDB8_8320
            } else {
                value >> 1
            };
        }
        *slot = value;
    }
    table
}

/// CRC-32 (IEEE 802.3) of `bytes`.
pub(super) fn crc32(bytes: &[u8]) -> u32 {
    let table = crc_table();
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        let index = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[index];
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_published_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn crc32_detects_every_single_byte_change() {
        let reference = b"the quick brown fox jumps over the lazy dog";
        let baseline = crc32(reference);
        for position in 0..reference.len() {
            let mut mutated = reference.to_vec();
            mutated[position] ^= 0x01;
            assert_ne!(
                crc32(&mutated),
                baseline,
                "byte {position} was not detected"
            );
        }
    }

    #[test]
    fn a_short_read_reports_both_sizes() {
        let mut reader = Reader::new(&[1, 2, 3]);
        let error = reader.u64("payload length").unwrap_err();
        let text = error.to_string();
        assert!(text.contains("8 bytes are required"), "{text}");
        assert!(text.contains("3 are available"), "{text}");
    }

    #[test]
    fn a_hostile_count_is_rejected_before_it_is_allocated() {
        let mut reader = Reader::new(&[0xFF, 0xFF, 0xFF, 0xFF]);
        let error = reader.count("value table", 1 << 20, 5).unwrap_err();
        assert!(matches!(
            error.kind(),
            FormatErrorKind::LimitExceeded { .. }
        ));
        // A count inside the policy cap but larger than the file is a truncation.
        let mut small = Reader::new(&[0x10, 0x00, 0x00, 0x00, 0, 0]);
        let error = small.count("value table", 1 << 20, 5).unwrap_err();
        assert!(matches!(error.kind(), FormatErrorKind::Truncated { .. }));
    }

    #[test]
    fn strings_are_capped_and_validated() {
        let mut over = vec![0xFF, 0xFF, 0x00, 0x00];
        over.extend(std::iter::repeat_n(0x41, 64));
        let mut reader = Reader::new(&over);
        assert!(matches!(
            reader.string("name").unwrap_err().kind(),
            FormatErrorKind::LimitExceeded { .. }
        ));

        let mut invalid = vec![2, 0, 0, 0, 0xC3, 0x28];
        let mut reader = Reader::new(&invalid);
        assert!(matches!(
            reader.string("name").unwrap_err().kind(),
            FormatErrorKind::Malformed { .. }
        ));
        invalid.clear();
    }

    #[test]
    fn padding_must_be_zero_and_alignment_is_exact() {
        let mut writer = Writer::with_capacity(0);
        writer.u8(1);
        writer.pad_to_8();
        assert_eq!(writer.len(), 8);
        writer.u32(7);
        writer.u64(9);
        assert_eq!(writer.len(), 20);
        writer.pad_to_8();
        let bytes = writer.into_bytes();
        assert_eq!(bytes.len(), 24);

        let mut reader = Reader::new(&bytes);
        assert_eq!(reader.u8("tag").unwrap(), 1);
        reader.pad_to_8("padding").unwrap();
        assert_eq!(reader.u32("seven").unwrap(), 7);
        assert_eq!(reader.u64("nine").unwrap(), 9);
        reader.pad_to_8("padding").unwrap();
        reader.finish("section").unwrap();

        let mut dirty = bytes.clone();
        dirty[1] = 0xFF;
        let mut reader = Reader::new(&dirty);
        reader.u8("tag").unwrap();
        assert!(matches!(
            reader.pad_to_8("padding").unwrap_err().kind(),
            FormatErrorKind::Malformed { .. }
        ));
    }

    #[test]
    fn f32_round_trips_its_exact_bits_including_nan() {
        let mut writer = Writer::with_capacity(8);
        writer.f32(0.1);
        writer.f32(f32::from_bits(0x7FC0_0001));
        let bytes = writer.into_bytes();
        let mut reader = Reader::new(&bytes);
        assert_eq!(reader.f32("a").unwrap().to_bits(), 0.1f32.to_bits());
        assert_eq!(reader.f32("b").unwrap().to_bits(), 0x7FC0_0001);
    }

    #[test]
    fn trailing_bytes_are_reported() {
        let mut reader = Reader::new(&[1, 2, 3]);
        reader.u8("one").unwrap();
        assert!(matches!(
            reader.finish("section").unwrap_err().kind(),
            FormatErrorKind::Malformed { .. }
        ));
    }
}
