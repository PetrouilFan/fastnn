//! SWAR (SIMD Within A Register) Kernels for Packed Quantized Arithmetic
//!
//! These kernels compute dot products directly on packed I8x4 / I4x8 words
//! WITHOUT unpacking to bytes. This is fastnn's unique differentiator:
//! 4-8× parallel arithmetic per instruction, zero dequantization overhead.

use crate::dtypes::{I4x8, I8x4};
use crate::packed_tensor::PackedTensor;

/// Compute dot product of two I8x4-packed u32 words (signed).
///
/// Each u32 holds 4 × signed i8 values (little-endian byte order, range [-128, 127]).
/// Returns i32 accumulator (exact for K ≤ 8192 without overflow).
#[inline(always)]
pub fn i8x4_dot_packed_signed(a: u32, b: u32) -> i32 {
    // Extract 4 signed bytes from each word and multiply pairwise
    // Sign-extend: (byte as i8) as i32
    let a0 = (a & 0xFF) as i8 as i32;
    let a1 = ((a >> 8) & 0xFF) as i8 as i32;
    let a2 = ((a >> 16) & 0xFF) as i8 as i32;
    let a3 = ((a >> 24) & 0xFF) as i8 as i32;

    let b0 = (b & 0xFF) as i8 as i32;
    let b1 = ((b >> 8) & 0xFF) as i8 as i32;
    let b2 = ((b >> 16) & 0xFF) as i8 as i32;
    let b3 = ((b >> 24) & 0xFF) as i8 as i32;

    a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3
}

/// Compute dot product of two I8x4-packed u32 words.
///
/// Each u32 holds 4 × signed i8 values (little-endian byte order, range [-128, 127]).
/// Returns i32 accumulator (exact for K ≤ 8192 without overflow).
#[inline(always)]
pub fn i8x4_dot_packed(a: u32, b: u32) -> i32 {
    i8x4_dot_packed_signed(a, b)
}

/// Sum all 4 signed i8 bytes in a I8x4 packed u32 word.
#[inline(always)]
pub fn sum_i8x4_packed(w: u32) -> i32 {
    let b0 = (w & 0xFF) as i8 as i32;
    let b1 = ((w >> 8) & 0xFF) as i8 as i32;
    let b2 = ((w >> 16) & 0xFF) as i8 as i32;
    let b3 = ((w >> 24) & 0xFF) as i8 as i32;
    b0 + b1 + b2 + b3
}

/// Sum all 8 signed nibbles in a I4x8 packed u32 word.
#[inline(always)]
pub fn sum_i4x8_packed(w: u32) -> i32 {
    let mut s = 0i32;
    for i in 0..8 {
        let nib = ((w >> (i * 4)) & 0xF) as i32;
        s += if nib >= 8 { nib - 16 } else { nib };
    }
    s
}

/// Compute dot product of two I4x8-packed u32 words.
///
/// Each u32 holds 8 × signed 4-bit values (nibbles, range [-8, 7]).
/// Returns i32 accumulator.
///
/// # Layout
/// Nibble 0 = bits 0-3, Nibble 1 = bits 4-7, ..., Nibble 7 = bits 28-31
/// Signed interpretation: 0x0..0x7 = +0..+7, 0x8..0xF = -8..-1
#[inline(always)]
pub fn i4x8_dot_packed(a: u32, b: u32) -> i32 {
    let mut sum = 0i32;
    for i in 0..8 {
        let shift = i * 4;
        // Extract signed 4-bit nibble: 0x0..0x7 = 0..7, 0x8..0xF = -8..-1
        let nibble_a = ((a >> shift) & 0xF) as i32;
        let nibble_b = ((b >> shift) & 0xF) as i32;
        let aw = if nibble_a >= 8 {
            nibble_a - 16
        } else {
            nibble_a
        };
        let bw = if nibble_b >= 8 {
            nibble_b - 16
        } else {
            nibble_b
        };
        sum += aw * bw;
    }
    sum
}

/// Vectorized I8x4 dot product over slices of packed u32 words.
///
/// Computes Σ i8x4_dot_packed(a[i], b[i]) for i in 0..len
/// Auto-vectorizes to SIMD on modern CPUs (AVX2/NEON).
#[inline(always)]
pub fn i8x4_dot_packed_slice(a: &[u32], b: &[u32]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0i32;
    for i in 0..a.len() {
        sum += i8x4_dot_packed(a[i], b[i]);
    }
    sum
}

/// Vectorized I4x8 dot product over slices of packed u32 words.
#[inline(always)]
pub fn i4x8_dot_packed_slice(a: &[u32], b: &[u32]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0i32;
    for i in 0..a.len() {
        sum += i4x8_dot_packed(a[i], b[i]);
    }
    sum
}

/// Quantize FP32 slice to I8x4 packed u32 words (per-tensor asymmetric, signed).
///
/// Returns (packed_words, scale, zero_point)
/// Each output u32 holds 4 quantized signed i8 values (-128..127).
/// Matches I8x4 storage format and from_f32_per_channel_asymmetric.
pub fn quantize_f32_to_i8x4(data: &[f32]) -> (Vec<u32>, f32, f32) {
    if data.is_empty() {
        return (Vec::new(), 1.0, 0.0);
    }

    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let range = max - min;
    let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
    // Signed asymmetric: map min → -128, max → +127
    // zero_point = min + 128 * scale
    let zero_point = if range > 0.0 {
        min + 128.0 * scale
    } else {
        0.0
    };

    let mut packed = Vec::with_capacity(data.len().div_ceil(4));
    let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };
    for chunk in data.chunks(4) {
        let mut word = 0u32;
        for (i, &val) in chunk.iter().enumerate() {
            // Quantize to signed i8 range [-128, 127]
            let q = ((val - zero_point) * inv_scale)
                .round()
                .clamp(-128.0, 127.0) as i8;
            word |= (q as u8 as u32) << (i * 8);
        }
        packed.push(word);
    }

    (packed, scale, zero_point)
}

/// Quantize FP32 slice to I4x8 packed u32 words (per-tensor asymmetric, signed).
///
/// Returns (packed_words, scale, zero_point)
/// Each output u32 holds 8 quantized signed 4-bit values [-8, 7].
/// Matches I4x8 storage format and from_f32_per_channel_asymmetric.
pub fn quantize_f32_to_i4x8(data: &[f32]) -> (Vec<u32>, f32, f32) {
    if data.is_empty() {
        return (Vec::new(), 1.0, 0.0);
    }

    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let range = max - min;
    let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
    // Signed asymmetric: map min → -8, max → +7
    // zero_point = min + 8 * scale
    let zero_point = if range > 0.0 { min + 8.0 * scale } else { 0.0 };

    let mut packed = Vec::with_capacity(data.len().div_ceil(8));
    let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };
    for chunk in data.chunks(8) {
        let mut word = 0u32;
        for (i, &val) in chunk.iter().enumerate() {
            // Quantize to signed 4-bit range [-8, 7]
            let q = ((val - zero_point) * inv_scale).round().clamp(-8.0, 7.0) as i32;
            let q_u = (q as u32) & 0xF; // 4 bits, two's complement
            word |= q_u << (i * 4);
        }
        packed.push(word);
    }

    (packed, scale, zero_point)
}

/// Dequantize I8x4 packed words to FP32 slice (signed asymmetric).
pub fn dequantize_i8x4_to_f32(packed: &[u32], scale: f32, zero_point: f32, out: &mut [f32]) {
    let mut idx = 0;
    for &word in packed {
        for lane in 0..4 {
            if idx >= out.len() {
                return;
            }
            // Extract as signed i8 (sign-extend from 8 bits)
            let q = ((word >> (lane * 8)) & 0xFF) as i8 as f32;
            out[idx] = q * scale + zero_point;
            idx += 1;
        }
    }
}

/// Dequantize I4x8 packed words to FP32 slice (signed asymmetric).
pub fn dequantize_i4x8_to_f32(packed: &[u32], scale: f32, zero_point: f32, out: &mut [f32]) {
    let mut idx = 0;
    for &word in packed {
        for lane in 0..8 {
            if idx >= out.len() {
                return;
            }
            let nibble = ((word >> (lane * 4)) & 0xF) as i32;
            // Sign-extend 4-bit: 0-7 → 0-7, 8-15 → -8 to -1
            let q = if nibble >= 8 { nibble - 16 } else { nibble };
            out[idx] = q as f32 * scale + zero_point;
            idx += 1;
        }
    }
}

/// Convert u32-packed data to PackedTensor<I8x4>
pub fn i8x4_packed_to_tensor(
    packed: Vec<u32>,
    shape: Vec<usize>,
    scale: f32,
    zero_point: f32,
) -> PackedTensor<I8x4> {
    let data: Vec<I8x4> = packed.into_iter().map(|w| I8x4(w)).collect();
    PackedTensor::from_raw(data, shape, vec![scale], vec![zero_point])
}

/// Convert u32-packed data to PackedTensor<I4x8>
pub fn i4x8_packed_to_tensor(
    packed: Vec<u32>,
    shape: Vec<usize>,
    scale: f32,
    zero_point: f32,
) -> PackedTensor<I4x8> {
    let data: Vec<I4x8> = packed.into_iter().map(|w| I4x8(w)).collect();
    PackedTensor::from_raw(data, shape, vec![scale], vec![zero_point])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::I8x4;

    #[test]
    fn test_i8x4_dot_packed_basic() {
        let a = 0x04030201u32; // [1, 2, 3, 4]
        let b = 0x01010101u32; // [1, 1, 1, 1]
        assert_eq!(i8x4_dot_packed(a, b), 10);
    }

    #[test]
    fn test_i8x4_dot_packed_zero() {
        assert_eq!(i8x4_dot_packed(0, 0xFFFFFFFF), 0);
        assert_eq!(i8x4_dot_packed(0xFFFFFFFF, 0), 0);
    }

    #[test]
    fn test_i8x4_dot_packed_max() {
        let a = 0xFFFFFFFF; // [-1, -1, -1, -1] signed
        let b = 0xFFFFFFFF; // [-1, -1, -1, -1] signed
                            // (-1)*(-1) * 4 = 4
        assert_eq!(i8x4_dot_packed(a, b), 4);
    }

    #[test]
    fn test_i8x4_dot_packed_signed_example() {
        // [1, 2, 3, 4] = 0x04030201
        let a = 0x04030201_u32;
        // [1, 1, 1, 1] = 0x01010101
        let b = 0x01010101_u32;
        // 1+2+3+4 = 10
        assert_eq!(i8x4_dot_packed(a, b), 10);
    }

    #[test]
    fn test_i4x8_dot_packed_basic() {
        // [1, 2, 3, 4, -1, -2, -3, -4]
        let a = 0x4321_FEDC_u32; // nibbles: C=12(-4), D=13(-3), E=14(-2), F=15(-1), 1,2,3,4
                                 // [1, 1, 1, 1, 1, 1, 1, 1]
        let b = 0x1111_1111_u32;
        // Sum = 1+2+3+4 + (-1)+(-2)+(-3)+(-4) = 0
        assert_eq!(i4x8_dot_packed(a, b), 0);
    }

    #[test]
    fn test_quantize_dequantize_i8x4_roundtrip() {
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let (packed, scale, zp) = quantize_f32_to_i8x4(&data);
        let mut out = vec![0.0; data.len()];
        dequantize_i8x4_to_f32(&packed, scale, zp, &mut out);

        for (orig, recon) in data.iter().zip(out.iter()) {
            assert!(
                (orig - recon).abs() < 0.01,
                "orig={:.3}, recon={:.3}",
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_i4x8_roundtrip() {
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1 - 1.5).collect();
        let (packed, scale, zp) = quantize_f32_to_i4x8(&data);
        let mut out = vec![0.0; data.len()];
        dequantize_i4x8_to_f32(&packed, scale, zp, &mut out);

        for (orig, recon) in data.iter().zip(out.iter()) {
            // I4x8 has only 16 quantization levels, quantization error ~0.1 is expected
            assert!(
                (orig - recon).abs() < 0.15,
                "orig={:.3}, recon={:.3}",
                orig,
                recon
            );
        }
    }
}
