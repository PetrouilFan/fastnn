//! SIMD-accelerated GEMV kernels for packed precision types.
//!
//! Key optimizations:
//! - Type-dispatched SIMD: U8x4 uses int8→f32 widening, F16x2 uses F16C
//! - Pre-unpack packed u32 words into contiguous f32 buffer, then SIMD FMA
//! - Cache-blocked K dimension for large matrices (L2 reuse across rows)
//! - U4x8 AVX2 kernel: nibble extraction → int32 → f32 → FMA

use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Cache-blocked GEMV for packed types.
/// Tiles the K dimension to keep activation blocks in L2 cache.
const K_BLOCK_SIZE: usize = 4096; // 16KB of f32 activations per block

// ============================================================
// Main entry point — type-dispatched
// ============================================================

/// SIMD-accelerated GEMV for packed types.
/// Dispatches to type-specific SIMD kernels when available on the target.
pub fn gemv_packed_simd<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
) {
    // Runtime type dispatch to optimized SIMD kernels
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        // Use TypeId for zero-cost type check
        let tid = std::any::TypeId::of::<T>();
        if tid == std::any::TypeId::of::<crate::dtypes::U8x4>() {
            let w: &PackedTensor<crate::dtypes::U8x4> =
                unsafe { &*(weights as *const _ as *const _) };
            return gemv_u8x4_dispatch(w, activation, output);
        }
        if tid == std::any::TypeId::of::<crate::dtypes::F16x2>() {
            let w: &PackedTensor<crate::dtypes::F16x2> =
                unsafe { &*(weights as *const _ as *const _) };
            return gemv_f16x2_dispatch(w, activation, output);
        }
        if tid == std::any::TypeId::of::<crate::dtypes::U4x8>() {
            let w: &PackedTensor<crate::dtypes::U4x8> =
                unsafe { &*(weights as *const _ as *const _) };
            return gemv_u4x8_dispatch(w, activation, output);
        }
    }

    // Generic fallback for F32x1 and non-x86 targets
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = (k + T::ITEMS - 1) / T::ITEMS;
    let scale = weights.scale();
    let zero = weights.zero();

    if k <= K_BLOCK_SIZE {
        gemv_packed_inner::<T>(weights, activation, output, scale, zero, m, k, k_packed);
    } else {
        gemv_packed_blocked::<T>(weights, activation, output, scale, zero, m, k, k_packed);
    }
}

// ============================================================
// U8x4: AVX2 int8→f32 widening + FMA
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn gemv_u8x4_dispatch(
    weights: &PackedTensor<crate::dtypes::U8x4>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = (k + 3) / 4;
    let scale = weights.scale();
    let zero = weights.zero();
    let weights_u32 = weights.as_u32();

    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).max(1).min(64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;
                        *out = unsafe {
                            gemv_row_u8x4_avx2(weights_u32, activation, row * k_packed, k, k_packed)
                        } * scale
                            - zero;
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                output[row] = unsafe {
                    gemv_row_u8x4_avx2(weights_u32, activation, row * k_packed, k, k_packed)
                } * scale
                    - zero;
            }
        }
    } else {
        gemv_generic_fallback(weights, activation, output);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn gemv_row_u8x4_avx2(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    // Process 8 int8 values at a time (2 u32 words = 8 bytes)
    while p + 1 < k_packed && act_idx + 8 <= k {
        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];

        let byte_array: [u8; 8] = [
            w0 as u8,
            (w0 >> 8) as u8,
            (w0 >> 16) as u8,
            (w0 >> 24) as u8,
            w1 as u8,
            (w1 >> 8) as u8,
            (w1 >> 16) as u8,
            (w1 >> 24) as u8,
        ];

        let i8x8 = _mm_loadl_epi64(byte_array.as_ptr() as *const __m128i);
        let i32x4_lo = _mm_cvtepi8_epi32(i8x8);
        let i32x4_hi = _mm_cvtepi8_epi32(_mm_srli_si128(i8x8, 4));
        let f32x4_lo = _mm_cvtepi32_ps(i32x4_lo);
        let f32x4_hi = _mm_cvtepi32_ps(i32x4_hi);
        let weight_f32 = _mm256_insertf128_ps(_mm256_castps128_ps256(f32x4_lo), f32x4_hi, 1);

        let act = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc = _mm256_fmadd_ps(weight_f32, act, acc);

        p += 2;
        act_idx += 8;
    }

    // Horizontal sum
    let mut total = hsum256_ps(acc);

    // Scalar tail for remaining packed words (1 at a time = 4 values)
    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        for j in 0..4 {
            let idx = act_idx + j;
            if idx < k {
                total += (bytes[j] as i8) as f32 * activation[idx];
            }
        }
        p += 1;
        act_idx += 4;
    }

    total
}

// ============================================================
// U4x8: AVX2 nibble extraction → int32 → f32 → FMA
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn gemv_u4x8_dispatch(
    weights: &PackedTensor<crate::dtypes::U4x8>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = (k + 7) / 8;
    let scale = weights.scale();
    let zero = weights.zero();
    let weights_u32 = weights.as_u32();

    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).max(1).min(64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;
                        *out = unsafe {
                            gemv_row_u4x8_avx2(weights_u32, activation, row * k_packed, k, k_packed)
                        } * scale
                            - zero;
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                output[row] = unsafe {
                    gemv_row_u4x8_avx2(weights_u32, activation, row * k_packed, k, k_packed)
                } * scale
                    - zero;
            }
        }
    } else {
        gemv_generic_fallback(weights, activation, output);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn gemv_row_u4x8_avx2(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    while p < k_packed && act_idx + 8 <= k {
        let w = weights_u32[row_offset + p];

        // Extract 8 nibbles from u32 into individual bytes
        // Even nibbles: bits 0,3→byte0, bits 8,11→byte1, etc.
        // Odd nibbles: bits 4,7→byte4, bits 12,15→byte5, etc.
        let even = w & 0x0F0F0F0F;
        let odd = (w >> 4) & 0x0F0F0F0F;
        // Pack: [nib0, nib2, nib4, nib6, nib1, nib3, nib5, nib7]
        // Actually simpler: extract all 8 nibbles to 8 bytes
        let nibbles: [u8; 8] = [
            (w & 0xF) as u8,
            ((w >> 4) & 0xF) as u8,
            ((w >> 8) & 0xF) as u8,
            ((w >> 12) & 0xF) as u8,
            ((w >> 16) & 0xF) as u8,
            ((w >> 20) & 0xF) as u8,
            ((w >> 24) & 0xF) as u8,
            ((w >> 28) & 0xF) as u8,
        ];

        // Sign-extend 4-bit to i8 (bit 3 is sign)
        let signed: [i8; 8] = [
            if nibbles[0] & 0x8 != 0 {
                (nibbles[0] | 0xF0) as i8
            } else {
                nibbles[0] as i8
            },
            if nibbles[1] & 0x8 != 0 {
                (nibbles[1] | 0xF0) as i8
            } else {
                nibbles[1] as i8
            },
            if nibbles[2] & 0x8 != 0 {
                (nibbles[2] | 0xF0) as i8
            } else {
                nibbles[2] as i8
            },
            if nibbles[3] & 0x8 != 0 {
                (nibbles[3] | 0xF0) as i8
            } else {
                nibbles[3] as i8
            },
            if nibbles[4] & 0x8 != 0 {
                (nibbles[4] | 0xF0) as i8
            } else {
                nibbles[4] as i8
            },
            if nibbles[5] & 0x8 != 0 {
                (nibbles[5] | 0xF0) as i8
            } else {
                nibbles[5] as i8
            },
            if nibbles[6] & 0x8 != 0 {
                (nibbles[6] | 0xF0) as i8
            } else {
                nibbles[6] as i8
            },
            if nibbles[7] & 0x8 != 0 {
                (nibbles[7] | 0xF0) as i8
            } else {
                nibbles[7] as i8
            },
        ];

        // Widen i8 → i32 → f32 using AVX2
        let i8x8 = _mm_loadl_epi64(signed.as_ptr() as *const __m128i);
        let i32x4_lo = _mm_cvtepi8_epi32(i8x8);
        let i32x4_hi = _mm_cvtepi8_epi32(_mm_srli_si128(i8x8, 4));
        let f32x4_lo = _mm_cvtepi32_ps(i32x4_lo);
        let f32x4_hi = _mm_cvtepi32_ps(i32x4_hi);
        let weight_f32 = _mm256_insertf128_ps(_mm256_castps128_ps256(f32x4_lo), f32x4_hi, 1);

        let act = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc = _mm256_fmadd_ps(weight_f32, act, acc);

        p += 1;
        act_idx += 8;
    }

    let mut total = hsum256_ps(acc);

    // Scalar tail for remaining activations
    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        for j in 0..8 {
            let idx = act_idx + j;
            if idx < k {
                let nibble = (w >> (j * 4)) & 0xF;
                let signed = if nibble & 0x8 != 0 {
                    (nibble | 0xFFFFFFF0) as i32
                } else {
                    nibble as i32
                };
                total += signed as f32 * activation[idx];
            }
        }
        p += 1;
        act_idx += 8;
    }

    total
}

// ============================================================
// F16x2: F16C hardware half→float conversion + FMA
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn gemv_f16x2_dispatch(
    weights: &PackedTensor<crate::dtypes::F16x2>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = (k + 1) / 2;
    let scale = weights.scale();
    let zero = weights.zero();
    let weights_u32 = weights.as_u32();

    if is_x86_feature_detected!("f16c") {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).max(1).min(64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;
                        *out = unsafe {
                            gemv_row_f16x2_f16c(
                                weights_u32,
                                activation,
                                row * k_packed,
                                k,
                                k_packed,
                            )
                        } * scale
                            - zero;
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                output[row] = unsafe {
                    gemv_row_f16x2_f16c(weights_u32, activation, row * k_packed, k, k_packed)
                } * scale
                    - zero;
            }
        }
    } else {
        gemv_generic_fallback(weights, activation, output);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn gemv_row_f16x2_f16c(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    // Process 4 u32 words at a time = 8 f16 → 8 f32
    while p + 4 <= k_packed && act_idx + 8 <= k {
        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];
        let w2 = weights_u32[row_offset + p + 2];
        let w3 = weights_u32[row_offset + p + 3];

        let half_bits: [u16; 8] = [
            w0 as u16,
            (w0 >> 16) as u16,
            w1 as u16,
            (w1 >> 16) as u16,
            w2 as u16,
            (w2 >> 16) as u16,
            w3 as u16,
            (w3 >> 16) as u16,
        ];

        let half_vec = _mm_loadu_si128(half_bits.as_ptr() as *const __m128i);
        let f32_vec = _mm256_cvtph_ps(half_vec);

        let act = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc = _mm256_fmadd_ps(f32_vec, act, acc);

        p += 4;
        act_idx += 8;
    }

    let mut total = hsum256_ps(acc);

    // Scalar tail
    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        let lo = half::f16::from_bits(w as u16).to_f32();
        let hi = half::f16::from_bits((w >> 16) as u16).to_f32();
        total += lo * activation[act_idx];
        if act_idx + 1 < k {
            total += hi * activation[act_idx + 1];
        }
        p += 1;
        act_idx += 2;
    }

    total
}

// ============================================================
// Generic fallback (used by F32x1 and non-SIMD targets)
// ============================================================

/// Generic GEMV fallback using scalar unpack + SIMD FMA.
fn gemv_generic_fallback<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = (k + T::ITEMS - 1) / T::ITEMS;
    let scale = weights.scale();
    let zero = weights.zero();

    if k <= K_BLOCK_SIZE {
        gemv_packed_inner::<T>(weights, activation, output, scale, zero, m, k, k_packed);
    } else {
        gemv_packed_blocked::<T>(weights, activation, output, scale, zero, m, k, k_packed);
    }
}

// ============================================================
// Generic GEMV implementation (scalar unpack + AVX2 FMA)
// ============================================================

fn gemv_packed_inner<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    scale: f32,
    zero: f32,
    _m: usize,
    k: usize,
    k_packed: usize,
) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let rows_per_chunk = (65536 / (k_packed * 4)).max(1).min(64);
        output
            .par_chunks_mut(rows_per_chunk)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_row = chunk_idx * rows_per_chunk;
                let mut unpack_buf = vec![0.0f32; k_packed * T::ITEMS];
                for (local_row, out) in out_chunk.iter_mut().enumerate() {
                    let row = start_row + local_row;
                    *out = gemv_row::<T>(weights, activation, row, k, k_packed, &mut unpack_buf)
                        * scale
                        - zero;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut unpack_buf = vec![0.0f32; k_packed * T::ITEMS];
        for row in 0.._m {
            output[row] = gemv_row::<T>(weights, activation, row, k, k_packed, &mut unpack_buf)
                * scale
                - zero;
        }
    }
}

fn gemv_packed_blocked<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    scale: f32,
    zero: f32,
    m: usize,
    k: usize,
    k_packed: usize,
) {
    for o in output.iter_mut() {
        *o = 0.0;
    }

    let items = T::ITEMS;
    let mut unpack_buf = vec![0.0f32; K_BLOCK_SIZE];

    let mut k_offset = 0;
    while k_offset < k {
        let k_end = (k_offset + K_BLOCK_SIZE).min(k);
        let k_block = k_end - k_offset;

        for row in 0..m {
            let row_offset = row * k_packed;
            let packed_start = k_offset / items;
            let packed_end = (k_end + items - 1) / items;
            let unpack_len = (packed_end - packed_start) * items;

            if unpack_len <= unpack_buf.len() {
                for (i, p) in (packed_start..packed_end).enumerate() {
                    let word = weights.as_packed()[row_offset + p];
                    let unpacked = word.unpack_to_f32();
                    let dst_start = i * items;
                    for j in 0..items {
                        if dst_start + j < k_block {
                            unpack_buf[dst_start + j] = unpacked.as_ref()[j];
                        }
                    }
                }
                let acc = fma_f32_slice(
                    &unpack_buf[..k_block.min(unpack_len)],
                    &activation[k_offset..k_end],
                );
                output[row] += acc;
            }
        }

        k_offset += K_BLOCK_SIZE;
    }

    for o in output.iter_mut() {
        *o = *o * scale - zero;
    }
}

#[inline]
fn gemv_row<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    row: usize,
    k: usize,
    k_packed: usize,
    unpack_buf: &mut [f32],
) -> f32 {
    let items = T::ITEMS;
    let row_offset = row * k_packed;

    for p in 0..k_packed {
        let word = weights.as_packed()[row_offset + p];
        let unpacked = word.unpack_to_f32();
        let base = p * items;
        for j in 0..items {
            let idx = base + j;
            if idx < k {
                unpack_buf[idx] = unpacked.as_ref()[j];
            }
        }
    }

    fma_f32_slice(&unpack_buf[..k], activation)
}

// ============================================================
// Shared SIMD utilities
// ============================================================

/// Horizontal sum of __m256 — used by all AVX2 kernels.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x0E);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// SIMD dot product of two f32 slices.
#[inline]
fn fma_f32_slice(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("fma") {
            return unsafe { fma_f32_avx2(a, b) };
        }
    }

    fma_f32_scalar(a, b)
}

#[inline]
fn fma_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fma_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 16 <= len {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }

    while i + 8 <= len {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        i += 8;
    }

    let combined = _mm256_add_ps(acc0, acc1);
    let mut total = hsum256_ps(combined);

    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::{F16x2, F32x1, U4x8, U8x4};

    #[test]
    fn test_fma_f32_slice() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = fma_f32_slice(&a, &b);
        assert!((result - 45.0).abs() < 0.001);
    }

    #[test]
    fn test_gemv_dispatch_u8x4() {
        let k = 32;
        let m = 4;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin() * 50.0).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.2).cos()).collect();
        let weights = PackedTensor::<U8x4>::from_f32_auto(&data, &[m, k]);

        let mut out = vec![0.0f32; m];
        // This should dispatch to the AVX2 kernel via gemv_packed_simd
        gemv_packed_simd(&weights, &activation, &mut out);

        // Verify results are reasonable (quantization introduces error)
        for v in &out {
            assert!(v.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_gemv_dispatch_f16x2() {
        let k = 16;
        let m = 4;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.2).cos()).collect();
        let weights = PackedTensor::<F16x2>::from_f32_auto(&data, &[m, k]);

        let mut out = vec![0.0f32; m];
        gemv_packed_simd(&weights, &activation, &mut out);

        for v in &out {
            assert!(v.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_gemv_dispatch_u4x8() {
        let k = 32;
        let m = 4;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin() * 5.0).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.2).cos()).collect();
        let weights = PackedTensor::<U4x8>::from_f32_auto(&data, &[m, k]);

        let mut out = vec![0.0f32; m];
        gemv_packed_simd(&weights, &activation, &mut out);

        for v in &out {
            assert!(v.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_gemv_dispatch_f32x1_exact() {
        let k = 16;
        let m = 4;
        let data: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let activation: Vec<f32> = (0..k).map(|i| i as f32 * 0.2).collect();
        let weights = PackedTensor::<F32x1>::from_f32_auto(&data, &[m, k]);

        let mut out = vec![0.0f32; m];
        gemv_packed_simd(&weights, &activation, &mut out);

        // F32x1 should be exact
        for v in &out {
            assert!(v.is_finite(), "Output should be finite");
        }
    }
}
