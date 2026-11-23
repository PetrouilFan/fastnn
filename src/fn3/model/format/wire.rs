//! Bounds-checked little-endian wire helpers.
//!
//! Everything here is deliberately small and dependency-free: the format is
//! hand-framed, so the reader owns its own cursor instead of delegating to a
//! serialization crate. Three rules are enforced by construction:
//!
//! 1. **No allocation without a validated length.** [`Reader::take`] checks the
//!    length against the remaining input *before* slicing, and every count is
//!    validated by [`Reader::count`] against both a policy cap and the number of
//!    bytes that could possibly encode it. A hostile count therefore costs one
//!    multiplication and one comparison, never memory.
//! 2. **No unchecked arithmetic.** Every offset is added with `checked_add`, so
//!    a 32-bit target reports [`FormatErrorKind::Truncated`] instead of wrapping.
//! 3. **Every diagnostic carries an absolute file offset.** A [`Reader`] is
//!    created with the file offset of its first byte, so a section decoded from
//!    an mmap can name the byte the user has to look at.
//!
//! The payload is protected by SHA-256 in the `HASHES` section, not by a CRC.
//! This module does not compute it: the artifact layer streams the ranges it
//! hashes so a mapped weight section is never faulted in just to validate it.

use super::error::{FormatError, FormatErrorKind};

/// Largest name or string this format accepts, in bytes.
///
/// ONNX value names in a real model are well under 100 bytes; the cap keeps a
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

    /// Zero-pad to the next multiple of 64.
    ///
    /// Every section and every weight payload is 64-byte aligned so a mapped
    /// weight range is aligned for every dtype the artifact can carry.
    pub(super) fn pad_to_64(&mut self) {
        while !self.bytes.len().is_multiple_of(64) {
            self.bytes.push(0);
        }
    }

    pub(super) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Read side: a cursor over one byte range with checked reads.
///
/// `base` is the absolute file offset of `bytes[0]`; it is carried only so error
/// messages can name a file position. It never changes an arithmetic result.
pub(super) struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    base: u64,
}

impl<'a> Reader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Reader<'a> {
        Reader::at(bytes, 0)
    }

    /// A reader whose first byte is `base` in the file.
    pub(super) fn at(bytes: &'a [u8], base: u64) -> Reader<'a> {
        Reader {
            bytes,
            pos: 0,
            base,
        }
    }

    /// Absolute file offset of the next byte.
    pub(super) fn pos_abs(&self) -> u64 {
        self.base.saturating_add(self.pos as u64)
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
            return Err(FormatError::truncated_at(
                what,
                self.pos_abs(),
                len,
                remaining,
            ));
        }
        let len = len as usize;
        let start = self.pos;
        let end = start
            .checked_add(len)
            .ok_or_else(|| FormatError::truncated_at(what, self.pos_abs(), u64::MAX, remaining))?;
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
        let offset = self.pos_abs();
        let bytes = self.take(len, what.clone())?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|error| {
                FormatError::malformed_at(
                    what,
                    offset,
                    format!("the bytes are not valid UTF-8: {error}"),
                )
            })
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
        let offset = self.pos_abs();
        let count = self.u32(what.clone())? as u64;
        if count > limit {
            return Err(FormatError::limit(what, count, limit));
        }
        let needed = count.saturating_mul(min_entry_bytes);
        let remaining = self.remaining();
        if needed > remaining {
            return Err(FormatError::truncated_at(what, offset, needed, remaining));
        }
        Ok(count as usize)
    }

    /// Skip to the next multiple of 64, requiring the padding to be zero.
    ///
    /// Requiring zeros makes the encoding canonical: there is exactly one valid
    /// byte string for a given model, so a re-write is byte-identical.
    pub(super) fn pad_to_64(&mut self, what: impl Into<String>) -> Result<(), FormatError> {
        let padding = (64 - (self.pos % 64)) % 64;
        if padding == 0 {
            return Ok(());
        }
        let what = what.into();
        let offset = self.pos_abs();
        let bytes = self.take(padding as u64, what.clone())?;
        if let Some(position) = bytes.iter().position(|&byte| byte != 0) {
            return Err(FormatError::malformed_at(
                what,
                offset + position as u64,
                format!(
                    "alignment padding byte is {:#04x}, expected 0",
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
            Err(FormatError::malformed_at(
                what,
                self.pos_abs(),
                format!("{} trailing bytes after the last field", self.remaining()),
            ))
        }
    }

    /// A reserved field that must be zero.
    pub(super) fn zero_u32(&mut self, field: &'static str) -> Result<(), FormatError> {
        let offset = self.pos_abs();
        let value = self.u32(field)?;
        if value == 0 {
            Ok(())
        } else {
            Err(FormatError::new(FormatErrorKind::ReservedField {
                field,
                value: u64::from(value),
                offset,
            }))
        }
    }

    /// An enum discriminant this build defines.
    pub(super) fn enum_code(
        &mut self,
        what: impl Into<String>,
        limit: u8,
    ) -> Result<u8, FormatError> {
        let what = what.into();
        let offset = self.pos_abs();
        let code = self.u8(what.clone())?;
        if code > limit {
            return Err(FormatError::malformed_at(
                what,
                offset,
                format!("code {code} is not defined; known codes are 0..={limit}"),
            ));
        }
        Ok(code)
    }

    /// A boolean encoded as one byte, 0 or 1.
    pub(super) fn boolean(&mut self, what: impl Into<String>) -> Result<bool, FormatError> {
        let what = what.into();
        let offset = self.pos_abs();
        match self.u8(what.clone())? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(FormatError::malformed_at(
                what,
                offset,
                format!("boolean is {other}, expected 0 or 1"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_read_reports_the_absolute_offset_and_both_sizes() {
        let mut reader = Reader::at(&[1, 2, 3], 4096);
        let error = reader.u64("payload length").unwrap_err();
        let text = error.to_string();
        assert!(text.contains("byte 4096"), "{text}");
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

        let invalid = vec![2, 0, 0, 0, 0xC3, 0x28];
        let mut reader = Reader::new(&invalid);
        assert!(matches!(
            reader.string("name").unwrap_err().kind(),
            FormatErrorKind::MalformedAt { .. }
        ));
    }

    #[test]
    fn padding_must_be_zero_and_alignment_is_exact() {
        let mut writer = Writer::with_capacity(0);
        writer.u8(1);
        writer.pad_to_64();
        assert_eq!(writer.len(), 64);
        writer.u32(7);
        writer.u64(9);
        assert_eq!(writer.len(), 76);
        writer.pad_to_64();
        let bytes = writer.into_bytes();
        assert_eq!(bytes.len(), 128);

        let mut reader = Reader::new(&bytes);
        assert_eq!(reader.u8("tag").unwrap(), 1);
        reader.pad_to_64("padding").unwrap();
        assert_eq!(reader.u32("seven").unwrap(), 7);
        assert_eq!(reader.u64("nine").unwrap(), 9);
        reader.pad_to_64("padding").unwrap();
        reader.finish("section").unwrap();

        let mut dirty = bytes.clone();
        dirty[1] = 0xFF;
        let mut reader = Reader::new(&dirty);
        reader.u8("tag").unwrap();
        assert!(matches!(
            reader.pad_to_64("padding").unwrap_err().kind(),
            FormatErrorKind::MalformedAt { .. }
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
    fn trailing_bytes_are_reported_at_their_offset() {
        let mut reader = Reader::new(&[1, 2, 3]);
        reader.u8("one").unwrap();
        let error = reader.finish("section").unwrap_err();
        assert!(matches!(error.kind(), FormatErrorKind::MalformedAt { .. }));
        assert!(error.to_string().contains("byte 1"), "{error}");
    }
}
