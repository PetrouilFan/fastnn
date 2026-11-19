//! `fn3-quant-format-v1`: the locked weight-quantization format.
//!
//! # What v1 is
//!
//! Signed INT8 weight codes in groups of 64 along the reduction axis, each
//! group with an IEEE binary16 scale and a signed int16 integer zero-point:
//!
//! ```text
//! w[r, c] = (code[r, c] - zero_point[r, c / 64]) * scale[r, c / 64]
//! ```
//!
//! Activations stay FP32. There are no Q/DQ nodes and no dequantized weight
//! buffer: the projection kernel consumes the INT8 code and the per-group
//! metadata directly and applies the scale after a group's dot product.
//!
//! The byte layout is specified and enforced in [`format`]:
//!
//! * one 64-byte header, `tensor_count` 64-byte descriptors, an 8-byte-per-group
//!   directory, then a 16-byte-aligned payload;
//! * rows are `align_up(k, 16)` bytes with zero padding after `k`;
//! * per tensor the payload is `codes | scales | zero_points`;
//! * two IEEE CRC-32 checksums cover the metadata and the payload.
//!
//! v1 has **no 4-bit payload and no nibble interleaving**. The evaluated
//! grouped-W4 candidates and their measured accuracy are recorded in
//! `docs/reports/fn3-22-quant-format.md`; 4-bit is a future format, not a
//! configuration of this one. `qbits` exists in the descriptor so the reader
//! can name the rejected width (`UnsupportedQBits`) rather than misread it.
//!
//! # Independent reference
//!
//! `tools/quant/fn3_quant.py` is a from-scratch encoder/decoder that shares no
//! code with this module. `fixtures/quant/` holds hand-checkable blobs that both
//! implementations decode to identical f32 values, plus one malformed blob per
//! typed error. The error [`QuantError::kind`] strings are shared verbatim with
//! the Python reference.
//!
//! # Contract for #69 / #70
//!
//! * The decoder returns a borrowed, validated view; it does not copy weights.
//! * `QuantTensor::dequantize_into` is the normative scalar dequantization.
//! * A kernel may fuse the same arithmetic, but must compute
//!   `sum((code - zero_point) * activation)` per group in f32 and multiply by
//!   the binary16 scale once per group. Integer codes never leave their packed
//!   form and no FP32 weight buffer is materialized.
//! * Group size, bit width, scale type and zero-point range are fixed. A blob
//!   that differs is rejected, not silently accepted.

pub mod error;
pub mod format;

pub use error::QuantError;
pub use format::{
    crc32, QuantBlob, QuantTensor, DATA_ALIGNMENT, DESCRIPTOR_BYTES, FORMAT_ID, GROUP_ENTRY_BYTES,
    GROUP_SIZE, HEADER_BYTES, MAX_TENSORS, NAME_BYTES, QUBITS, VERSION,
};
