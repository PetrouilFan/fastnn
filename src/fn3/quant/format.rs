//! The `fn3-quant-format-v1` container: parse, validate, decode.
//!
//! # Layout
//!
//! All integers are little-endian. The blob is:
//!
//! ```text
//! 0               64-byte header
//! 64              tensor_count * 64-byte descriptors
//! ...             group directory, 8 bytes per group
//! align_up(..,16) payload
//! ```
//!
//! Header (64 bytes):
//!
//! ```text
//!  0  magic           4  "FN3Q"
//!  4  version         2  1
//!  6  header_bytes    2  64
//!  8  flags           4  0
//! 12  tensor_count    4  1..=4096
//! 16  descriptor_bytes4  tensor_count * 64
//! 20  group_dir_bytes 4  total groups * 8
//! 24  descriptor_offset 8 64
//! 32  group_offset    8  descriptor_offset + descriptor_bytes
//! 40  payload_offset  8  align_up(group_offset + group_dir_bytes, 16)
//! 48  payload_bytes   8
//! 56  payload_crc32   4  IEEE CRC-32 of the payload section
//! 60  meta_crc32      4  IEEE CRC-32 of bytes [0, payload_offset) with this
//!                        field zeroed
//! ```
//!
//! Descriptor (64 bytes):
//!
//! ```text
//!  0  name            32 NUL-padded ASCII, unique
//! 32  n_rows          4
//! 36  k               4
//! 40  qbits           2  8
//! 42  group_size      2  64
//! 44  groups_per_row  4  ceil(k / 64)
//! 48  group_dir_index 4
//! 52  row_stride_bytes4  align_up(k, 16)
//! 56  data_offset     8  absolute, 16-byte aligned
//! ```
//!
//! A tensor's payload is `n_rows * row_stride_bytes` code bytes, then
//! `n_rows * groups_per_row` binary16 scales, then the same number of int16
//! zero-points. Rows are zero-padded after `k`. Codes are signed INT8; the
//! dequantized value is `(code - zero_point) * scale`.
//!
//! The canonical layout is enforced, not merely described: descriptor order,
//! group entries, data offsets and padding must all be exactly what the encoder
//! in `tools/quant/fn3_quant.py` writes. A valid blob therefore has one byte
//! representation, which is what makes the committed fixtures meaningful.

use std::collections::BTreeSet;

use half::f16;

use super::error::QuantError;

/// Stable identifier for this byte layout.
pub const FORMAT_ID: &str = "fn3-quant-format-v1";
/// Container version.
pub const VERSION: u16 = 1;
/// Fixed header size.
pub const HEADER_BYTES: usize = 64;
/// Fixed descriptor size.
pub const DESCRIPTOR_BYTES: usize = 64;
/// Group directory entry size.
pub const GROUP_ENTRY_BYTES: usize = 8;
/// Quantization group size along K.
pub const GROUP_SIZE: usize = 64;
/// Weight bits per element.
pub const QUBITS: u16 = 8;
/// Payload alignment for every tensor's code bytes.
pub const DATA_ALIGNMENT: u64 = 16;
/// Largest accepted tensor count.
pub const MAX_TENSORS: u32 = 4096;
/// Maximum tensor name length in bytes, including the NUL when shorter.
pub const NAME_BYTES: usize = 32;

const MAGIC: [u8; 4] = *b"FN3Q";

/// One validated tensor view inside a [`QuantBlob`].
#[derive(Debug, Clone)]
pub struct QuantTensor<'a> {
    name: String,
    n_rows: usize,
    k: usize,
    qbits: u16,
    group_size: u16,
    groups_per_row: usize,
    row_stride: usize,
    data_offset: usize,
    data: &'a [u8],
    scales: &'a [u8],
    zeros: &'a [u8],
    groups: Vec<(u32, u32)>,
}

impl<'a> QuantTensor<'a> {
    /// Unique tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Number of output rows (N).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    /// Reduction length (K).
    pub fn k(&self) -> usize {
        self.k
    }
    /// Bits per stored weight; always 8 in v1.
    pub fn qbits(&self) -> u16 {
        self.qbits
    }
    /// Elements per quantization group; always 64 in v1.
    pub fn group_size(&self) -> u16 {
        self.group_size
    }
    /// Number of groups in each row.
    pub fn groups_per_row(&self) -> usize {
        self.groups_per_row
    }
    /// Packed bytes per row, including alignment padding.
    pub fn row_stride_bytes(&self) -> usize {
        self.row_stride
    }
    /// Absolute offset of the code bytes in the blob.
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }
    /// Raw signed code bytes, `n_rows * row_stride_bytes` long.
    pub fn codes(&self) -> &'a [u8] {
        self.data
    }
    /// Raw binary16 scales, `n_rows * groups_per_row * 2` bytes.
    pub fn scales(&self) -> &'a [u8] {
        self.scales
    }
    /// Raw int16 zero-points, `n_rows * groups_per_row * 2` bytes.
    pub fn zero_points(&self) -> &'a [u8] {
        self.zeros
    }
    /// A group's `(start, end)` columns within a row.
    pub fn group_range(&self, group: usize) -> (usize, usize) {
        let (start, length) = self.groups[group];
        (start as usize, start as usize + length as usize)
    }
    /// Binary16 scale for `(row, group)`.
    pub fn scale(&self, row: usize, group: usize) -> f32 {
        let index = (row * self.groups_per_row + group) * 2;
        f16::from_bits(u16::from_le_bytes([
            self.scales[index],
            self.scales[index + 1],
        ]))
        .to_f32()
    }
    /// Integer zero-point for `(row, group)`.
    pub fn zero_point(&self, row: usize, group: usize) -> i16 {
        let index = (row * self.groups_per_row + group) * 2;
        i16::from_le_bytes([self.zeros[index], self.zeros[index + 1]])
    }
    /// Signed code at `(row, col)`.
    pub fn code(&self, row: usize, col: usize) -> i8 {
        self.data[row * self.row_stride + col] as i8
    }
    /// Dequantized value at `(row, col)`, or `None` outside the tensor.
    pub fn value(&self, row: usize, col: usize) -> Option<f32> {
        if row >= self.n_rows || col >= self.k {
            return None;
        }
        let group = col / GROUP_SIZE;
        let scale = self.scale(row, group);
        let zero = i32::from(self.zero_point(row, group));
        Some((i32::from(self.code(row, col)) - zero) as f32 * scale)
    }
    /// Dequantize into a row-major `[n_rows, k]` buffer.
    ///
    /// The arithmetic is normative: `(code - zero_point)` in `i32`, then one
    /// f32 multiply by the binary16 scale widened to f32.
    pub fn dequantize_into(&self, output: &mut [f32]) -> Result<(), QuantError> {
        let expected = self.n_rows * self.k;
        if output.len() < expected {
            return Err(QuantError::OutputTooSmall {
                expected,
                found: output.len(),
            });
        }
        for row in 0..self.n_rows {
            let base = row * self.k;
            for group in 0..self.groups_per_row {
                let (start, end) = self.group_range(group);
                let scale = self.scale(row, group);
                let zero = i32::from(self.zero_point(row, group));
                for col in start..end {
                    let code = i32::from(self.code(row, col));
                    output[base + col] = (code - zero) as f32 * scale;
                }
            }
        }
        Ok(())
    }
    /// Allocate and dequantize a row-major `[n_rows, k]` tensor.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut values = vec![0.0f32; self.n_rows * self.k];
        self.dequantize_into(&mut values)
            .expect("the buffer is sized from the tensor");
        values
    }
}

/// A parsed and validated `fn3-quant-format-v1` blob.
#[derive(Debug, Clone)]
pub struct QuantBlob<'a> {
    bytes: &'a [u8],
    payload_offset: usize,
    payload_bytes: usize,
    tensors: Vec<QuantTensor<'a>>,
}

impl<'a> QuantBlob<'a> {
    /// Parse and fully validate a blob.
    ///
    /// Every structural rule is checked before any value is returned, including
    /// both checksums, every scale, every zero-point and all padding bytes.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, QuantError> {
        if bytes.len() < HEADER_BYTES {
            return Err(QuantError::TruncatedHeader {
                available: bytes.len(),
            });
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != MAGIC {
            return Err(QuantError::BadMagic { found: magic });
        }
        let version = read_u16(bytes, 4);
        if version != VERSION {
            return Err(QuantError::UnsupportedVersion { found: version });
        }
        let header_bytes = read_u16(bytes, 6);
        if header_bytes as usize != HEADER_BYTES {
            return Err(QuantError::BadHeaderSize {
                found: header_bytes,
            });
        }
        let flags = read_u32(bytes, 8);
        if flags != 0 {
            return Err(QuantError::UnsupportedFlags { found: flags });
        }
        let tensor_count = read_u32(bytes, 12);
        if tensor_count == 0 || tensor_count > MAX_TENSORS {
            return Err(QuantError::BadTensorCount {
                found: tensor_count,
            });
        }
        let descriptor_bytes = read_u32(bytes, 16);
        if descriptor_bytes != tensor_count * DESCRIPTOR_BYTES as u32 {
            return Err(QuantError::BadDescriptorBytes {
                found: descriptor_bytes,
            });
        }
        let group_dir_bytes = read_u32(bytes, 20);
        let descriptor_offset = read_u64(bytes, 24);
        if descriptor_offset != HEADER_BYTES as u64 {
            return Err(QuantError::BadDescriptorOffset {
                found: descriptor_offset,
            });
        }
        let group_offset = read_u64(bytes, 32);
        if group_offset != descriptor_offset + u64::from(descriptor_bytes) {
            return Err(QuantError::BadGroupOffset {
                found: group_offset,
            });
        }
        if !group_dir_bytes.is_multiple_of(GROUP_ENTRY_BYTES as u32) {
            return Err(QuantError::BadGroupDirBytes {
                found: group_dir_bytes,
            });
        }
        let payload_offset = read_u64(bytes, 40);
        let canonical_payload = align_up(group_offset + u64::from(group_dir_bytes), DATA_ALIGNMENT);
        if payload_offset != canonical_payload {
            return Err(QuantError::BadPayloadOffset {
                found: payload_offset,
            });
        }
        let payload_bytes = read_u64(bytes, 48);
        let Some(payload_end) = payload_offset.checked_add(payload_bytes) else {
            return Err(QuantError::BadPayloadOffset {
                found: payload_offset,
            });
        };
        if payload_end != bytes.len() as u64 {
            return Err(QuantError::TrailingOrTruncatedBytes {
                file: bytes.len(),
                payload_end,
            });
        }

        let stored_meta_crc = read_u32(bytes, 60);
        let mut meta = Crc32::new();
        meta.update(&bytes[..60]);
        meta.update(&[0u8; 4]);
        meta.update(&bytes[64..payload_offset as usize]);
        let actual_meta_crc = meta.finish();
        if actual_meta_crc != stored_meta_crc {
            return Err(QuantError::MetaChecksum {
                expected: stored_meta_crc,
                actual: actual_meta_crc,
            });
        }
        let stored_payload_crc = read_u32(bytes, 56);
        let actual_payload_crc = crc32(&bytes[payload_offset as usize..]);
        if actual_payload_crc != stored_payload_crc {
            return Err(QuantError::PayloadChecksum {
                expected: stored_payload_crc,
                actual: actual_payload_crc,
            });
        }

        let group_count = group_dir_bytes as usize / GROUP_ENTRY_BYTES;
        let group_table_start = group_offset as usize;
        let mut group_table = Vec::with_capacity(group_count);
        for index in 0..group_count {
            let base = group_table_start + index * GROUP_ENTRY_BYTES;
            group_table.push((read_u32(bytes, base), read_u32(bytes, base + 4)));
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut names = BTreeSet::new();
        let mut expected_group_index = 0usize;
        let mut expected_data_offset = payload_offset;
        for index in 0..tensor_count as usize {
            let base = descriptor_offset as usize + index * DESCRIPTOR_BYTES;
            let raw_name = &bytes[base..base + NAME_BYTES];
            let nul = raw_name
                .iter()
                .position(|&byte| byte == 0)
                .ok_or(QuantError::BadName {
                    tensor: index,
                    reason: "not NUL-terminated",
                })?;
            let name_bytes = &raw_name[..nul];
            if name_bytes.is_empty() {
                return Err(QuantError::BadName {
                    tensor: index,
                    reason: "empty",
                });
            }
            if !name_bytes.iter().all(|&byte| (0x20..=0x7e).contains(&byte)) {
                return Err(QuantError::BadName {
                    tensor: index,
                    reason: "non-ASCII",
                });
            }
            if raw_name[nul + 1..].iter().any(|&byte| byte != 0) {
                return Err(QuantError::BadName {
                    tensor: index,
                    reason: "bytes after the NUL",
                });
            }
            let name = String::from_utf8(name_bytes.to_vec()).map_err(|_| QuantError::BadName {
                tensor: index,
                reason: "non-ASCII",
            })?;
            if !names.insert(name.clone()) {
                return Err(QuantError::DuplicateName { name });
            }

            let n_rows = read_u32(bytes, base + 32) as usize;
            let k = read_u32(bytes, base + 36) as usize;
            let qbits = read_u16(bytes, base + 40);
            let group_size = read_u16(bytes, base + 42);
            let groups_per_row = read_u32(bytes, base + 44) as usize;
            let group_index = read_u32(bytes, base + 48) as usize;
            let row_stride = read_u32(bytes, base + 52) as usize;
            let data_offset = read_u64(bytes, base + 56);

            if n_rows == 0 {
                return Err(QuantError::ZeroExtent {
                    tensor: name,
                    field: "n_rows",
                });
            }
            if k == 0 {
                return Err(QuantError::ZeroExtent {
                    tensor: name,
                    field: "k",
                });
            }
            if qbits != QUBITS {
                return Err(QuantError::UnsupportedQBits {
                    tensor: name,
                    found: qbits,
                });
            }
            if group_size as usize != GROUP_SIZE {
                return Err(QuantError::UnsupportedGroupSize {
                    tensor: name,
                    found: group_size,
                });
            }
            let expected_groups = k.div_ceil(GROUP_SIZE);
            if groups_per_row != expected_groups {
                return Err(QuantError::BadGroupCount {
                    tensor: name,
                    found: groups_per_row as u32,
                    expected: expected_groups as u32,
                });
            }
            if group_index != expected_group_index {
                return Err(QuantError::BadGroupIndex {
                    tensor: name,
                    found: group_index as u32,
                    expected: expected_group_index as u32,
                });
            }
            expected_group_index += groups_per_row;
            let expected_stride = align_up(k as u64, DATA_ALIGNMENT) as usize;
            if row_stride != expected_stride {
                return Err(QuantError::BadRowStride {
                    tensor: name,
                    found: row_stride as u32,
                    expected: expected_stride as u32,
                });
            }
            if !data_offset.is_multiple_of(DATA_ALIGNMENT) {
                return Err(QuantError::MisalignedOffset {
                    tensor: name,
                    offset: data_offset,
                    alignment: DATA_ALIGNMENT,
                });
            }
            if data_offset != expected_data_offset {
                return Err(QuantError::BadDataOffset {
                    tensor: name,
                    found: data_offset,
                    expected: expected_data_offset,
                });
            }

            let data_bytes =
                n_rows
                    .checked_mul(row_stride)
                    .ok_or_else(|| QuantError::TruncatedPayload {
                        tensor: name.clone(),
                    })?;
            let scale_bytes = n_rows
                .checked_mul(groups_per_row)
                .and_then(|value| value.checked_mul(2))
                .ok_or_else(|| QuantError::TruncatedPayload {
                    tensor: name.clone(),
                })?;
            let end = data_offset as usize + data_bytes + 2 * scale_bytes;
            if end > payload_end as usize {
                return Err(QuantError::TruncatedPayload { tensor: name });
            }
            let expected_end = align_up(end as u64, DATA_ALIGNMENT) as usize;
            if expected_end > bytes.len() {
                return Err(QuantError::TruncatedPayload { tensor: name });
            }
            if bytes[end..expected_end].iter().any(|&byte| byte != 0) {
                return Err(QuantError::NonZeroPadding {
                    tensor: name,
                    offset: end as u64,
                });
            }
            expected_data_offset = expected_end as u64;

            let mut groups = Vec::with_capacity(groups_per_row);
            let mut previous_end = 0u32;
            for group in 0..groups_per_row {
                if group_index + group >= group_table.len() {
                    return Err(QuantError::TruncatedPayload {
                        tensor: name.clone(),
                    });
                }
                let (start, length) = group_table[group_index + group];
                if start != previous_end {
                    return Err(QuantError::GroupGapOrOverlap {
                        tensor: name,
                        group: group as u32,
                        found: start,
                        expected: previous_end,
                    });
                }
                let expected_length = (k - start as usize).min(GROUP_SIZE) as u32;
                if length != expected_length {
                    return Err(QuantError::BadGroupLength {
                        tensor: name,
                        group: group as u32,
                        found: length,
                        expected: expected_length,
                    });
                }
                previous_end = start + length;
                groups.push((start, length));
            }
            if previous_end as usize != k {
                return Err(QuantError::IncompleteGroups {
                    tensor: name,
                    end: previous_end,
                    k: k as u32,
                });
            }

            let data_start = data_offset as usize;
            let data = &bytes[data_start..data_start + data_bytes];
            for row in 0..n_rows {
                let row_base = row * row_stride;
                if data[row_base + k..row_base + row_stride]
                    .iter()
                    .any(|&byte| byte != 0)
                {
                    return Err(QuantError::NonZeroTailPadding {
                        tensor: name,
                        offset: (data_start + row_base + k) as u64,
                    });
                }
            }
            let scale_start = data_start + data_bytes;
            let scales = &bytes[scale_start..scale_start + scale_bytes];
            let zero_start = scale_start + scale_bytes;
            let zeros = &bytes[zero_start..zero_start + scale_bytes];

            for row in 0..n_rows {
                for group in 0..groups_per_row {
                    let at = (row * groups_per_row + group) * 2;
                    let bits = u16::from_le_bytes([scales[at], scales[at + 1]]);
                    let scale = f16::from_bits(bits).to_f32();
                    if !scale.is_finite() || scale <= 0.0 {
                        return Err(QuantError::BadScale {
                            tensor: name,
                            row: row as u32,
                            group: group as u32,
                            bits,
                        });
                    }
                    let zero = i16::from_le_bytes([zeros[at], zeros[at + 1]]);
                    if !(-128..=127).contains(&zero) {
                        return Err(QuantError::ZeroPointOutOfRange {
                            tensor: name,
                            row: row as u32,
                            group: group as u32,
                            value: i32::from(zero),
                        });
                    }
                }
            }

            tensors.push(QuantTensor {
                name,
                n_rows,
                k,
                qbits,
                group_size,
                groups_per_row,
                row_stride,
                data_offset: data_start,
                data,
                scales,
                zeros,
                groups,
            });
        }

        if expected_group_index != group_table.len() {
            return Err(QuantError::UnusedGroupEntries {
                count: (group_table.len() - expected_group_index) as u32,
            });
        }
        if expected_data_offset != payload_end {
            return Err(QuantError::PayloadTail {
                expected: expected_data_offset,
                found: payload_end,
            });
        }

        Ok(QuantBlob {
            bytes,
            payload_offset: payload_offset as usize,
            payload_bytes: payload_bytes as usize,
            tensors,
        })
    }

    /// The validated tensors, in descriptor order.
    pub fn tensors(&self) -> &[QuantTensor<'a>] {
        &self.tensors
    }
    /// Look a tensor up by name.
    pub fn tensor(&self, name: &str) -> Option<&QuantTensor<'a>> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
    /// The original bytes.
    pub fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }
    /// Absolute start of the payload section.
    pub fn payload_offset(&self) -> usize {
        self.payload_offset
    }
    /// Length of the payload section.
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

const fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

/// IEEE CRC-32 (the zlib/PNG polynomial, reflected).
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = Crc32::new();
    crc.update(bytes);
    crc.finish()
}

struct Crc32(u32);

impl Crc32 {
    fn new() -> Self {
        Crc32(0xffff_ffff)
    }
    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            let index = ((self.0 ^ u32::from(byte)) & 0xff) as usize;
            self.0 = CRC_TABLE[index] ^ (self.0 >> 8);
        }
    }
    fn finish(self) -> u32 {
        !self.0
    }
}

const fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 != 0 {
                0xedb8_8320 ^ (value >> 1)
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

static CRC_TABLE: [u32; 256] = crc_table();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_ieee_check_value() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32(b""), 0);
    }
}
