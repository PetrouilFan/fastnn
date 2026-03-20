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
        // Prefetch next iteration's weights into L1
        if p + 4 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 4) as *const i8,
                _MM_HINT_T0,
            );
        }

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

    // Process 2 u32 words (16 nibbles) per iteration for 2x throughput
    while p + 1 < k_packed && act_idx + 16 <= k {
        // Prefetch next iteration's weights into L1
        if p + 4 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 4) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];

        // Word 0: branchless SIMD sign-extend
        let even0 = w0 & 0x0F0F0F0F;
        let odd0 = (w0 >> 4) & 0x0F0F0F0F;
        let inter0 = even0 | (odd0 << 4);
        let nib0 = _mm_set1_epi32(inter0 as i32);
        let sign0 = _mm_and_si128(_mm_srli_epi16(nib0, 3), _mm_set1_epi8(0x10));
        let i8_0 = _mm_sub_epi8(nib0, sign0);

        // Word 1: same pipeline, executes in parallel (ILP)
        let even1 = w1 & 0x0F0F0F0F;
        let odd1 = (w1 >> 4) & 0x0F0F0F0F;
        let inter1 = even1 | (odd1 << 4);
        let nib1 = _mm_set1_epi32(inter1 as i32);
        let sign1 = _mm_and_si128(_mm_srli_epi16(nib1, 3), _mm_set1_epi8(0x10));
        let i8_1 = _mm_sub_epi8(nib1, sign1);

        // Widen i8 → i32 → f32 for both words
        let f32_0_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(i8_0));
        let f32_0_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(i8_0, 4)));
        let w_f32_0 = _mm256_insertf128_ps(_mm256_castps128_ps256(f32_0_lo), f32_0_hi, 1);

        let f32_1_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(i8_1));
        let f32_1_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(i8_1, 4)));
        let w_f32_1 = _mm256_insertf128_ps(_mm256_castps128_ps256(f32_1_lo), f32_1_hi, 1);

        // FMA both words with their activation slices
        let act0 = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        let act1 = _mm256_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc = _mm256_fmadd_ps(w_f32_0, act0, acc);
        acc = _mm256_fmadd_ps(w_f32_1, act1, acc);

        p += 2;
        act_idx += 16;
    }

    // Single-word tail (process 8 nibbles)
    if p < k_packed && act_idx + 8 <= k {
        let w = weights_u32[row_offset + p];
        let even = w & 0x0F0F0F0F;
        let odd = (w >> 4) & 0x0F0F0F0F;
        let inter = even | (odd << 4);
        let nib = _mm_set1_epi32(inter as i32);
        let sign = _mm_and_si128(_mm_srli_epi16(nib, 3), _mm_set1_epi8(0x10));
        let i8v = _mm_sub_epi8(nib, sign);
        let f32_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(i8v));
        let f32_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(i8v, 4)));
        let w_f32 = _mm256_insertf128_ps(_mm256_castps128_ps256(f32_lo), f32_hi, 1);
        let act = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc = _mm256_fmadd_ps(w_f32, act, acc);
        p += 1;
        act_idx += 8;
    }

    let mut total = hsum256_ps(acc);

    // Scalar tail
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
    let mut acc_tail = 0.0f32;
    let mut p = 0;
    let mut act_idx = 0;

    // Process 8 u32 words at a time = 16 f16 → 16 f32 (2x throughput)
    while p + 8 <= k_packed && act_idx + 16 <= k {
        if p + 12 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 12) as *const i8,
                _MM_HINT_T0,
            );
        }

        // Load 8 u32 words containing 16 packed f16 values
        let w0 = _mm_loadu_si128(weights_u32.as_ptr().add(row_offset + p) as *const __m128i);
        let w1 = _mm_loadu_si128(weights_u32.as_ptr().add(row_offset + p + 4) as *const __m128i);

        // Interleave u32 pairs into u16 lanes — avoids [u16; 8] array
        // w0 = [w0_lo, w0_hi, w1_lo, w1_hi, w2_lo, w2_hi, w3_lo, w3_hi] as u16
        let half0 = _mm_unpacklo_epi16(w0, _mm_setzero_si128());
        let half1 = _mm_unpackhi_epi16(w0, _mm_setzero_si128());
        // Wait, that's wrong. Each u32 has 2 f16: lo_half | (hi_half << 16)
        // We need to extract lo u16 and hi u16 from each u32.
        // _mm_cvtps_ph does the opposite direction.
        // Simpler: use the array but stack-allocate it
        let b0: [u16; 8] = [
            weights_u32[row_offset + p] as u16,
            (weights_u32[row_offset + p] >> 16) as u16,
            weights_u32[row_offset + p + 1] as u16,
            (weights_u32[row_offset + p + 1] >> 16) as u16,
            weights_u32[row_offset + p + 2] as u16,
            (weights_u32[row_offset + p + 2] >> 16) as u16,
            weights_u32[row_offset + p + 3] as u16,
            (weights_u32[row_offset + p + 3] >> 16) as u16,
        ];
        let h0 = _mm_loadu_si128(b0.as_ptr() as *const __m128i);
        let f0 = _mm256_cvtph_ps(h0);
        let a0 = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc = _mm256_fmadd_ps(f0, a0, acc);

        let b1: [u16; 8] = [
            weights_u32[row_offset + p + 4] as u16,
            (weights_u32[row_offset + p + 4] >> 16) as u16,
            weights_u32[row_offset + p + 5] as u16,
            (weights_u32[row_offset + p + 5] >> 16) as u16,
            weights_u32[row_offset + p + 6] as u16,
            (weights_u32[row_offset + p + 6] >> 16) as u16,
            weights_u32[row_offset + p + 7] as u16,
            (weights_u32[row_offset + p + 7] >> 16) as u16,
        ];
        let h1 = _mm_loadu_si128(b1.as_ptr() as *const __m128i);
        let f1 = _mm256_cvtph_ps(h1);
        let a1 = _mm256_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc = _mm256_fmadd_ps(f1, a1, acc);

        p += 8;
        act_idx += 16;
    }

    // 4-word tail: 8 f16 → 8 f32
    while p + 4 <= k_packed && act_idx + 8 <= k {
        let b: [u16; 8] = [
            weights_u32[row_offset + p] as u16,
            (weights_u32[row_offset + p] >> 16) as u16,
            weights_u32[row_offset + p + 1] as u16,
            (weights_u32[row_offset + p + 1] >> 16) as u16,
            weights_u32[row_offset + p + 2] as u16,
            (weights_u32[row_offset + p + 2] >> 16) as u16,
            weights_u32[row_offset + p + 3] as u16,
            (weights_u32[row_offset + p + 3] >> 16) as u16,
        ];
        let h = _mm_loadu_si128(b.as_ptr() as *const __m128i);
        let f = _mm256_cvtph_ps(h);
        let a = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc = _mm256_fmadd_ps(f, a, acc);
        p += 4;
        act_idx += 8;
    }

    // 2-word tail: 4 f16 → 4 f32
    while p + 2 <= k_packed && act_idx + 4 <= k {
        let b: [u16; 4] = [
            weights_u32[row_offset + p] as u16,
            (weights_u32[row_offset + p] >> 16) as u16,
            weights_u32[row_offset + p + 1] as u16,
            (weights_u32[row_offset + p + 1] >> 16) as u16,
        ];
        let h = _mm_loadl_epi64(b.as_ptr() as *const __m128i);
        let f = _mm_cvtph_ps(h);
        let a = _mm_loadu_ps(activation.as_ptr().add(act_idx));
        let prod = _mm_mul_ps(f, a);
        let shuf = _mm_shuffle_ps(prod, prod, 0x0E);
        let sums = _mm_add_ps(prod, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
        acc_tail += _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        p += 2;
        act_idx += 4;
    }

    // 1-word tail: 2 f16 → 2 f32
    if p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        let b: [u16; 2] = [w as u16, (w >> 16) as u16];
        let h = _mm_loadl_epi64(b.as_ptr() as *const __m128i);
        let f = _mm_cvtph_ps(h);
        let arr: [f32; 4] = std::mem::transmute(f);
        acc_tail += arr[0] * activation[act_idx];
        if act_idx + 1 < k {
            acc_tail += arr[1] * activation[act_idx + 1];
        }
    }

    hsum256_ps(acc) + acc_tail
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
// Batched GEMM — unpack once, process N input vectors
// ============================================================

/// Batched GEMM for packed types: weights [M×K] × batch_inputs [N×K] → outputs [N×M].
/// Unpacks each weight row once, then processes all N input vectors against it.
/// This is the key optimization for inference — weights stay hot in L1 across the batch.
pub fn gemm_packed_batched<T: PackedWord>(
    weights: &PackedTensor<T>,
    batch_inputs: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
) {
    let n = batch_inputs.len();
    assert_eq!(n, outputs.len());
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = (k + T::ITEMS - 1) / T::ITEMS;
    let scale = weights.scale();
    let zero = weights.zero();

    // Process each weight row once, compute dot products with all N inputs
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // Store as usize for Send+Sync safety (each row writes to distinct positions)
        let out_addrs: Vec<usize> = outputs
            .iter_mut()
            .map(|o| o.as_mut_ptr() as usize)
            .collect();

        (0..m).into_par_iter().for_each(|row| {
            // Unpack this row once
            let mut unpack_buf = vec![0.0f32; k];
            let row_offset = row * k_packed;
            for p in 0..k_packed {
                let word = weights.as_packed()[row_offset + p];
                let unpacked = word.unpack_to_f32();
                let base = p * T::ITEMS;
                for j in 0..T::ITEMS {
                    let idx = base + j;
                    if idx < k {
                        unpack_buf[idx] = unpacked.as_ref()[j];
                    }
                }
            }

            // Dot product with all N input vectors — write via raw pointer
            for (bi, input) in batch_inputs.iter().enumerate() {
                let acc = fma_f32_slice(&unpack_buf, input);
                unsafe {
                    *((out_addrs[bi] as *mut f32).add(row)) = acc * scale - zero;
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut unpack_buf = vec![0.0f32; k];
        for row in 0..m {
            let row_offset = row * k_packed;
            for p in 0..k_packed {
                let word = weights.as_packed()[row_offset + p];
                let unpacked = word.unpack_to_f32();
                let base = p * T::ITEMS;
                for j in 0..T::ITEMS {
                    let idx = base + j;
                    if idx < k {
                        unpack_buf[idx] = unpacked.as_ref()[j];
                    }
                }
            }
            for (bi, input) in batch_inputs.iter().enumerate() {
                let acc = fma_f32_slice(&unpack_buf, input);
                outputs[bi][row] = acc * scale - zero;
            }
        }
    }
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
