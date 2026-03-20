//! SIMD-accelerated GEMV kernels for packed precision types.
//!
//! Key optimizations:
//! - Pre-unpack packed u32 words into contiguous f32 buffer, then SIMD FMA
//! - Eliminates per-element branches in inner loop → LLVM can auto-vectorize
//! - F16C intrinsics for hardware half→float conversion (1 cycle vs ~10 software)
//! - Cache-blocked K dimension for large matrices (L2 reuse across rows)

use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Cache-blocked GEMV for packed types.
/// Tiles the K dimension to keep activation blocks in L2 cache,
/// improving reuse across output rows.
const K_BLOCK_SIZE: usize = 4096; // 16KB of f32 activations per block

/// SIMD-accelerated GEMV for packed types.
/// Pre-unpacks weights, then uses auto-vectorized f32 FMA.
pub fn gemv_packed_simd<T: PackedWord>(
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
        // Small K: no blocking needed
        gemv_packed_inner::<T>(weights, activation, output, scale, zero, m, k, k_packed);
    } else {
        // Large K: cache-block along K dimension
        gemv_packed_blocked::<T>(weights, activation, output, scale, zero, m, k, k_packed);
    }
}

/// Non-blocked GEMV: process full K per row.
fn gemv_packed_inner<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    scale: f32,
    zero: f32,
    m: usize,
    k: usize,
    k_packed: usize,
) {
    // Thread-local unpack buffer (reused across rows per thread)
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // Use chunk-based parallelism for better cache locality
        // Each chunk processes multiple consecutive rows, keeping activations in L1
        let rows_per_chunk = (65536 / (k_packed * 4)).max(1).min(64); // ~64KB weight chunk
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
        for row in 0..m {
            output[row] = gemv_row::<T>(weights, activation, row, k, k_packed, &mut unpack_buf)
                * scale
                - zero;
        }
    }
}

/// Cache-blocked GEMV: tile K dimension for L2 reuse.
/// Accumulates partial results across K blocks.
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
    // Zero output accumulators
    for o in output.iter_mut() {
        *o = 0.0;
    }

    let items = T::ITEMS;
    let mut unpack_buf = vec![0.0f32; K_BLOCK_SIZE];

    // Process K in blocks
    let mut k_offset = 0;
    while k_offset < k {
        let k_end = (k_offset + K_BLOCK_SIZE).min(k);
        let k_block = k_end - k_offset;

        // Process each row, accumulating partial dot products for this K block
        for row in 0..m {
            let mut acc: f32 = 0.0;
            let row_offset = row * k_packed;

            // Unpack this row's weights for the current K block
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

                // SIMD FMA on the unpacked block
                acc = fma_f32_slice(
                    &unpack_buf[..k_block.min(unpack_len)],
                    &activation[k_offset..k_end],
                );
            }

            output[row] += acc;
        }

        k_offset += K_BLOCK_SIZE;
    }

    // Apply scale and zero
    for o in output.iter_mut() {
        *o = *o * scale - zero;
    }
}

/// Compute a single GEMV row: unpack weights, dot product with activation.
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

    // Phase 1: Unpack all weights for this row into contiguous f32 buffer
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

    // Phase 2: SIMD FMA on contiguous f32 slices
    fma_f32_slice(&unpack_buf[..k], activation)
}

/// SIMD dot product of two f32 slices.
/// LLVM auto-vectorizes this to AVX2/NEON when compiled with target-cpu=native.
#[inline]
fn fma_f32_slice(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        // Use AVX2 intrinsics for explicit SIMD control
        if is_x86_feature_detected!("fma") {
            return unsafe { fma_f32_avx2(a, b) };
        }
    }

    // Scalar fallback (LLVM may still auto-vectorize)
    fma_f32_scalar(a, b)
}

/// Scalar dot product fallback.
#[inline]
fn fma_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// AVX2 + FMA dot product — explicit SIMD.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fma_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;

    // Process 16 elements at a time (2x unroll for ILP)
    while i + 16 <= len {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }

    // Process 8 elements at a time
    while i + 8 <= len {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        i += 8;
    }

    // Combine two accumulators
    let combined = _mm256_add_ps(acc0, acc1);

    // Horizontal sum of __m256
    let hi128 = _mm256_extractf128_ps(combined, 1);
    let lo128 = _mm256_castps256_ps128(combined);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x0E);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Scalar tail
    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

// ============================================================
// Phase 1 continued: AVX2 int8→f32 direct unpack+dot for U8x4
// ============================================================

/// Optimized GEMV for U8x4: use AVX2 to widen int8→f32 directly,
/// then FMA — avoids the generic unpack_to_f32 loop entirely.
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
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    // Process 8 int8 values at a time (2 u32 words = 8 bytes)
    while p + 1 < k_packed && act_idx + 8 <= k {
        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];

        // Pack u32 words into an i32 register as 8 bytes
        let bytes = _mm_set_epi32(
            (w1 >> 24) as i32,
            (w1 >> 16) as i32,
            (w1 >> 8) as i32,
            w1 as i32,
        );

        // Bytes are stored in little-endian: byte0 at lowest position
        // We need to load bytes 0-7 as individual i8 values
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

        // Load 8 bytes as __m128i, then widen int8→int32
        let i8x8 = _mm_loadl_epi64(byte_array.as_ptr() as *const __m128i);
        let i32x4_lo = _mm_cvtepi8_epi32(i8x8);
        let i32x4_hi = _mm_cvtepi8_epi32(_mm_srli_si128(i8x8, 4));

        // Convert int32→f32
        let f32x4_lo = _mm_cvtepi32_ps(i32x4_lo);
        let f32x4_hi = _mm_cvtepi32_ps(i32x4_hi);
        let weight_f32 = _mm256_insertf128_ps(_mm256_castps128_ps256(f32x4_lo), f32x4_hi, 1);

        // Load 8 activations
        let act = _mm256_loadu_ps(activation.as_ptr().add(act_idx));

        // FMA
        acc0 = _mm256_fmadd_ps(weight_f32, act, acc0);

        p += 2;
        act_idx += 8;
    }

    // Combine accumulators
    let hi128 = _mm256_extractf128_ps(acc0, 1);
    let lo128 = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x0E);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

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
// Phase 1 continued: F16C for F16x2 — hardware half→float
// ============================================================

/// Optimized GEMV for F16x2: use F16C intrinsics for hardware f16→f32 conversion.
/// Each u32 contains 2 f16 values. Process 4 u32 words (8 f16) at a time.
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

    // Process 4 u32 words at a time = 8 f16 values → 8 f32 values
    while p + 4 <= k_packed && act_idx + 8 <= k {
        // Load 4 u32 words containing 8 packed f16 values
        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];
        let w2 = weights_u32[row_offset + p + 2];
        let w3 = weights_u32[row_offset + p + 3];

        // Pack f16 bits into an __m128i register (8 × u16)
        let f16_bits = _mm_set_epi32(
            (w3 as i32) | ((w3 >> 16) as i32) << 16,
            0,
            0,
            0, // placeholder
        );

        // Actually, easier: build the u16 array directly
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

        // Load 8 f16 bits as __m128i, convert to f32 with F16C
        let half_vec = _mm_loadu_si128(half_bits.as_ptr() as *const __m128i);
        let f32_vec = _mm256_cvtph_ps(half_vec);

        // Load 8 activations and FMA
        let act = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc = _mm256_fmadd_ps(f32_vec, act, acc);

        p += 4;
        act_idx += 8;
    }

    // Horizontal sum
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x0E);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

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

/// High-level dispatch: use best available SIMD for packed GEMV row.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn gemv_packed_optimized_u8x4(
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

    let has_avx2_fma = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");

    if has_avx2_fma {
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
        // Fallback to generic
        super::cpu::gemv_cpu(weights, activation, output);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn gemv_packed_optimized_f16x2(
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

    let has_f16c = is_x86_feature_detected!("f16c");

    if has_f16c {
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
        super::cpu::gemv_cpu(weights, activation, output);
    }
}

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
    fn test_gemv_simd_u8x4_matches_scalar() {
        let k = 32;
        let m = 4;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin() * 50.0).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.2).cos()).collect();
        let weights = PackedTensor::<U8x4>::from_f32_auto(&data, &[m, k]);

        let mut out_generic = vec![0.0f32; m];
        super::super::cpu::gemv_cpu(&weights, &activation, &mut out_generic);

        let mut out_simd = vec![0.0f32; m];
        gemv_packed_simd(&weights, &activation, &mut out_simd);

        for i in 0..m {
            let tol = 1.0; // Quantization introduces error
            assert!(
                (out_generic[i] - out_simd[i]).abs() <= tol,
                "Row {}: generic={}, simd={}, diff={}",
                i,
                out_generic[i],
                out_simd[i],
                (out_generic[i] - out_simd[i]).abs()
            );
        }
    }

    #[test]
    fn test_gemv_simd_f16x2_matches_scalar() {
        let k = 16;
        let m = 4;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.2).cos()).collect();
        let weights = PackedTensor::<F16x2>::from_f32_auto(&data, &[m, k]);

        let mut out_generic = vec![0.0f32; m];
        super::super::cpu::gemv_cpu(&weights, &activation, &mut out_generic);

        let mut out_simd = vec![0.0f32; m];
        gemv_packed_simd(&weights, &activation, &mut out_simd);

        for i in 0..m {
            let tol = 0.05; // f16 precision
            assert!(
                (out_generic[i] - out_simd[i]).abs() <= tol,
                "Row {}: generic={}, simd={}",
                i,
                out_generic[i],
                out_simd[i]
            );
        }
    }

    #[test]
    fn test_gemv_simd_f32x1_exact() {
        let k = 16;
        let m = 4;
        let data: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let activation: Vec<f32> = (0..k).map(|i| i as f32 * 0.2).collect();
        let weights = PackedTensor::<F32x1>::from_f32_auto(&data, &[m, k]);

        let mut out_generic = vec![0.0f32; m];
        super::super::cpu::gemv_cpu(&weights, &activation, &mut out_generic);

        let mut out_simd = vec![0.0f32; m];
        gemv_packed_simd(&weights, &activation, &mut out_simd);

        for i in 0..m {
            assert!(
                (out_generic[i] - out_simd[i]).abs() < 0.001,
                "Row {}: generic={}, simd={}",
                i,
                out_generic[i],
                out_simd[i]
            );
        }
    }
}
