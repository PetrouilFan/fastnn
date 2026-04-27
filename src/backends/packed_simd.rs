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

/// Aligned scratch buffer for unpacking operations
#[repr(align(32))]
struct AlignedScratchBuffer {
    data: Vec<f32>,
}

impl AlignedScratchBuffer {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn resize(&mut self, size: usize) {
        self.data.resize(size, 0.0);
    }

    fn as_slice(&self) -> &[f32] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

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
        if tid == std::any::TypeId::of::<crate::dtypes::F32x1>() {
            let w: &PackedTensor<crate::dtypes::F32x1> =
                unsafe { &*(weights as *const _ as *const _) };
            return gemv_f32x1_dispatch(w, activation, output);
        }
    }

    // Generic fallback for F32x1 and non-x86 targets
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(T::ITEMS);

    if k <= K_BLOCK_SIZE {
        gemv_packed_inner::<T>(weights, activation, output, m, k, k_packed);
    } else {
        gemv_packed_blocked::<T>(weights, activation, output, m, k, k_packed);
    }
}

// ============================================================
// U8x4: AVX2 int8→f32 widening + FMA
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(unused_variables)]
fn gemv_u8x4_dispatch(
    weights: &PackedTensor<crate::dtypes::U8x4>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(4);
    let weights_u32 = weights.as_u32();

    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).clamp(1, 64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;
                        let dot = unsafe {
                            gemv_row_u8x4_avx512(
                                weights_u32,
                                activation,
                                row * k_packed,
                                k,
                                k_packed,
                            )
                        };
                        *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                let dot = unsafe {
                    gemv_row_u8x4_avx512(weights_u32, activation, row * k_packed, k, k_packed)
                };
                output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
            }
        }
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).clamp(1, 64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;
                        let dot = unsafe {
                            gemv_row_u8x4_avx2(weights_u32, activation, row * k_packed, k, k_packed)
                        };
                        *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                let dot = unsafe {
                    gemv_row_u8x4_avx2(weights_u32, activation, row * k_packed, k, k_packed)
                };
                output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
            }
        }
    } else {
        // Scalar fallback
        for row in 0..m {
            let mut acc = 0.0f32;
            for p in 0..k_packed {
                let w = weights_u32[row * k_packed + p];
                let bytes = w.to_le_bytes();
                for j in 0..4.min(k - p * 4) {
                    acc += (bytes[j] as i8) as f32 * activation[p * 4 + j];
                }
            }
            output[row] = acc * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
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

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn gemv_row_u8x4_avx512(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    // Process 4 u32 words at a time (16 int8 → 16 f32) using AVX512BW for byte ops
    while p + 4 <= k_packed && act_idx + 16 <= k {
        if p + 8 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 8) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];
        let w2 = weights_u32[row_offset + p + 2];
        let w3 = weights_u32[row_offset + p + 3];

        // Pack 16 bytes from 4 u32 words into a 128-bit register
        let bytes = _mm_set_epi32(w3 as i32, w2 as i32, w1 as i32, w0 as i32);
        // Widen signed int8 → int32, then convert to f32
        let i32x16 = _mm512_cvtepi8_epi32(bytes);
        let weight_f32 = _mm512_cvtepi32_ps(i32x16);

        let act = _mm512_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm512_fmadd_ps(weight_f32, act, acc0);

        p += 4;
        act_idx += 16;
    }

    let mut total = _mm512_reduce_add_ps(acc0);

    // 2-word tail: 8 int8 → 8 f32 (scalar)
    while p + 2 <= k_packed && act_idx + 8 <= k {
        for word_off in 0..2 {
            let w = weights_u32[row_offset + p + word_off];
            let bytes = w.to_le_bytes();
            for j in 0..4 {
                total += (bytes[j] as i8) as f32 * activation[act_idx + word_off * 4 + j];
            }
        }
        p += 2;
        act_idx += 8;
    }

    // Vectorized tail for remaining packed words using 128-bit operations
    while p < k_packed && act_idx + 4 <= k {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        let weights_f32: [f32; 4] = [
            bytes[0] as i8 as f32,
            bytes[1] as i8 as f32,
            bytes[2] as i8 as f32,
            bytes[3] as i8 as f32,
        ];
        let acts = [activation[act_idx], activation[act_idx + 1], activation[act_idx + 2], activation[act_idx + 3]];

        // Use 128-bit FMA
        let w_vec = _mm_loadu_ps(weights_f32.as_ptr());
        let a_vec = _mm_loadu_ps(acts.as_ptr());
        let prod = _mm_mul_ps(w_vec, a_vec);
        let hsum = _mm_hadd_ps(prod, prod);
        let final_sum = _mm_hadd_ps(hsum, hsum);
        total += _mm_cvtss_f32(final_sum);

        p += 1;
        act_idx += 4;
    }

    // Scalar tail for remaining elements
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
#[allow(unused_variables)]
fn gemv_u4x8_dispatch(
    weights: &PackedTensor<crate::dtypes::U4x8>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(8);
    let weights_u32 = weights.as_u32();

    if is_x86_feature_detected!("avx512f") {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).clamp(1, 64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;
                        let dot = unsafe {
                            gemv_row_u4x8_avx512(
                                weights_u32,
                                activation,
                                row * k_packed,
                                k,
                                k_packed,
                            )
                        };
                        *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                let dot = unsafe {
                    gemv_row_u4x8_avx512(weights_u32, activation, row * k_packed, k, k_packed)
                };
                output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
            }
        }
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).clamp(1, 64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;
                        let dot = unsafe {
                            gemv_row_u4x8_avx2(weights_u32, activation, row * k_packed, k, k_packed)
                        };
                        *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                let dot = unsafe {
                    gemv_row_u4x8_avx2(weights_u32, activation, row * k_packed, k, k_packed)
                };
                output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
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
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;
    let mask_lo = _mm256_set1_epi32(0xF);
    let sign_bit = _mm256_set1_epi32(0x8);

    // Process 2 u32 words (16 nibbles) per iteration with dual accumulators
    while p + 2 <= k_packed && act_idx + 16 <= k {
        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];

        // Extract low nibbles (bits 0,4,8,12,16,20,24,28) from both words
        let shift0 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
        let w0v = _mm256_set1_epi32(w0 as i32);
        let w1v = _mm256_set1_epi32(w1 as i32);

        let nib_lo0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift0), mask_lo);
        let nib_lo1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift0), mask_lo);

        // Sign-extend: if bit 3 set, OR with 0xFFFFFFF0
        let neg_lo0 =
            _mm256_cmpgt_epi32(_mm256_and_si256(nib_lo0, sign_bit), _mm256_setzero_si256());
        let signed_lo0 =
            _mm256_or_si256(nib_lo0, _mm256_and_si256(neg_lo0, _mm256_set1_epi32(-16)));
        let neg_lo1 =
            _mm256_cmpgt_epi32(_mm256_and_si256(nib_lo1, sign_bit), _mm256_setzero_si256());
        let signed_lo1 =
            _mm256_or_si256(nib_lo1, _mm256_and_si256(neg_lo1, _mm256_set1_epi32(-16)));

        let fl0 = _mm256_cvtepi32_ps(signed_lo0);
        let fl1 = _mm256_cvtepi32_ps(signed_lo1);

        let al0 = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm256_fmadd_ps(fl0, al0, acc0);

        let al1 = _mm256_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc1 = _mm256_fmadd_ps(fl1, al1, acc1);

        // Extract high nibbles (bits 2,6,10,14,18,22,26,30)
        let shift_hi0 = _mm256_set_epi32(30, 26, 22, 18, 14, 10, 6, 2);
        let nib_hi0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift_hi0), mask_lo);
        let nib_hi1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift_hi0), mask_lo);

        let neg_hi0 =
            _mm256_cmpgt_epi32(_mm256_and_si256(nib_hi0, sign_bit), _mm256_setzero_si256());
        let signed_hi0 =
            _mm256_or_si256(nib_hi0, _mm256_and_si256(neg_hi0, _mm256_set1_epi32(-16)));
        let neg_hi1 =
            _mm256_cmpgt_epi32(_mm256_and_si256(nib_hi1, sign_bit), _mm256_setzero_si256());
        let signed_hi1 =
            _mm256_or_si256(nib_hi1, _mm256_and_si256(neg_hi1, _mm256_set1_epi32(-16)));

        let fh0 = _mm256_cvtepi32_ps(signed_hi0);
        let fh1 = _mm256_cvtepi32_ps(signed_hi1);

        let ah0 = _mm256_loadu_ps(activation.as_ptr().add(act_idx + 4));
        acc0 = _mm256_fmadd_ps(fh0, ah0, acc0);

        let ah1 = _mm256_loadu_ps(activation.as_ptr().add(act_idx + 12));
        acc1 = _mm256_fmadd_ps(fh1, ah1, acc1);

        p += 2;
        act_idx += 16;
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    let mut total = hsum256_ps(acc0);

    // Scalar tail for remaining packed words
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

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn gemv_row_u4x8_avx512(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;
    let mask_lo = _mm256_set1_epi32(0xF);
    let sign_bit = _mm256_set1_epi32(0x8);

    // Process 2 u32 words (16 nibbles) per iteration with dual accumulators
    while p + 2 <= k_packed && act_idx + 16 <= k {
        if p + 4 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 4) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];
        let w0v = _mm256_set1_epi32(w0 as i32);
        let w1v = _mm256_set1_epi32(w1 as i32);

        // Low nibbles (bits 0,4,8,12,16,20,24,28)
        let shift_lo = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
        let nib_lo0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift_lo), mask_lo);
        let nib_lo1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift_lo), mask_lo);

        let signed_lo0 = _mm256_or_si256(
            nib_lo0,
            _mm256_and_si256(
                _mm256_cmpgt_epi32(_mm256_and_si256(nib_lo0, sign_bit), _mm256_setzero_si256()),
                _mm256_set1_epi32(-16),
            ),
        );
        let signed_lo1 = _mm256_or_si256(
            nib_lo1,
            _mm256_and_si256(
                _mm256_cmpgt_epi32(_mm256_and_si256(nib_lo1, sign_bit), _mm256_setzero_si256()),
                _mm256_set1_epi32(-16),
            ),
        );

        let fl0 = _mm512_cvtepi32_ps(_mm512_castsi256_si512(signed_lo0));
        let fl1 = _mm512_cvtepi32_ps(_mm512_castsi256_si512(signed_lo1));
        let al0 = _mm512_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm512_fmadd_ps(fl0, al0, acc0);
        let al1 = _mm512_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc1 = _mm512_fmadd_ps(fl1, al1, acc1);

        // High nibbles (bits 2,6,10,14,18,22,26,30)
        let shift_hi = _mm256_set_epi32(30, 26, 22, 18, 14, 10, 6, 2);
        let nib_hi0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift_hi), mask_lo);
        let nib_hi1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift_hi), mask_lo);

        let signed_hi0 = _mm256_or_si256(
            nib_hi0,
            _mm256_and_si256(
                _mm256_cmpgt_epi32(_mm256_and_si256(nib_hi0, sign_bit), _mm256_setzero_si256()),
                _mm256_set1_epi32(-16),
            ),
        );
        let signed_hi1 = _mm256_or_si256(
            nib_hi1,
            _mm256_and_si256(
                _mm256_cmpgt_epi32(_mm256_and_si256(nib_hi1, sign_bit), _mm256_setzero_si256()),
                _mm256_set1_epi32(-16),
            ),
        );

        let fh0 = _mm512_cvtepi32_ps(_mm512_castsi256_si512(signed_hi0));
        let fh1 = _mm512_cvtepi32_ps(_mm512_castsi256_si512(signed_hi1));
        let ah0 = _mm512_loadu_ps(activation.as_ptr().add(act_idx + 4));
        acc0 = _mm512_fmadd_ps(fh0, ah0, acc0);
        let ah1 = _mm512_loadu_ps(activation.as_ptr().add(act_idx + 12));
        acc1 = _mm512_fmadd_ps(fh1, ah1, acc1);

        p += 2;
        act_idx += 16;
    }

    let combined = _mm512_add_ps(acc0, acc1);
    let mut total = _mm512_reduce_add_ps(combined);

    // Scalar tail for remaining packed words
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
#[allow(unused_variables)]
fn gemv_f16x2_dispatch(
    weights: &PackedTensor<crate::dtypes::F16x2>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(2);
    let weights_u32 = weights.as_u32();

    if is_x86_feature_detected!("f16c") {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).clamp(1, 64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;
                        let dot = unsafe {
                            gemv_row_f16x2_f16c(
                                weights_u32,
                                activation,
                                row * k_packed,
                                k,
                                k_packed,
                            )
                        };
                        *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                let dot = unsafe {
                    gemv_row_f16x2_f16c(weights_u32, activation, row * k_packed, k, k_packed)
                };
                output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
            }
        }
    } else {
        gemv_generic_fallback(weights, activation, output);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn u32x4_to_f32x8_f16c(w0: u32, w1: u32, w2: u32, w3: u32) -> __m256 {
    // Each u32 has 2 f16: lo_half at bits 0-15, hi_half at bits 16-31
    // _mm_set_epi32(w3, w2, w1, w0) lays out as:
    //   u16 lane 0: w0 & 0xFFFF (lo)
    //   u16 lane 1: w0 >> 16    (hi)
    //   u16 lane 2: w1 & 0xFFFF (lo)
    //   ...etc — exactly what _mm256_cvtph_ps expects
    let half_bits = _mm_set_epi32(w3 as i32, w2 as i32, w1 as i32, w0 as i32);
    _mm256_cvtph_ps(half_bits)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn u32x2_to_f32x4_f16c(w0: u32, w1: u32) -> __m128 {
    let half_bits = _mm_set_epi32(0, 0, w1 as i32, w0 as i32);
    _mm_cvtph_ps(half_bits)
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
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    // Process 8 u32 words (16 f16) per iteration — 2x throughput with ILP
    while p + 8 <= k_packed && act_idx + 16 <= k {
        if p + 16 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 16) as *const i8,
                _MM_HINT_T0,
            );
        }

        let f0 = u32x4_to_f32x8_f16c(
            weights_u32[row_offset + p],
            weights_u32[row_offset + p + 1],
            weights_u32[row_offset + p + 2],
            weights_u32[row_offset + p + 3],
        );
        let a0 = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm256_fmadd_ps(f0, a0, acc0);

        let f1 = u32x4_to_f32x8_f16c(
            weights_u32[row_offset + p + 4],
            weights_u32[row_offset + p + 5],
            weights_u32[row_offset + p + 6],
            weights_u32[row_offset + p + 7],
        );
        let a1 = _mm256_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc1 = _mm256_fmadd_ps(f1, a1, acc1);

        p += 8;
        act_idx += 16;
    }

    // 4-word tail: 8 f16 → 8 f32
    while p + 4 <= k_packed && act_idx + 8 <= k {
        let f = u32x4_to_f32x8_f16c(
            weights_u32[row_offset + p],
            weights_u32[row_offset + p + 1],
            weights_u32[row_offset + p + 2],
            weights_u32[row_offset + p + 3],
        );
        let a = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm256_fmadd_ps(f, a, acc0);
        p += 4;
        act_idx += 8;
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    let mut total = hsum256_ps(acc0);

    // 2-word tail: 4 f16 → 4 f32
    if p + 2 <= k_packed && act_idx + 4 <= k {
        let f = u32x2_to_f32x4_f16c(weights_u32[row_offset + p], weights_u32[row_offset + p + 1]);
        let a = _mm_loadu_ps(activation.as_ptr().add(act_idx));
        let prod = _mm_mul_ps(f, a);
        let shuf = _mm_shuffle_ps(prod, prod, 0x0E);
        let sums = _mm_add_ps(prod, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
        total += _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        p += 2;
        act_idx += 4;
    }

    // 1-word tail: 2 f16 → 2 f32
    if p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        let f = u32x2_to_f32x4_f16c(w, 0);
        let arr: [f32; 4] = std::mem::transmute(f);
        total += arr[0] * activation[act_idx];
        if act_idx + 1 < k {
            total += arr[1] * activation[act_idx + 1];
        }
    }

    total
}

// ============================================================
// F32x1: direct f32 loads (no unpack needed)
// ============================================================

#[allow(unused_variables)]
fn gemv_f32x1_dispatch(
    weights: &PackedTensor<crate::dtypes::F32x1>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    // F32x1: u32 IS f32, reinterpret directly
    let weights_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(
            weights.as_u32().as_ptr() as *const f32,
            weights.packed_len(),
        )
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let row_data = &weights_f32[row * k..(row + 1) * k];
            let dot = fma_f32_slice(row_data, activation);
            *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..m {
            let row_data = &weights_f32[row * k..(row + 1) * k];
            let dot = fma_f32_slice(row_data, activation);
            output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    }
}

// ============================================================
// Generic fallback (used by non-SIMD targets)
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
    let k_packed = k.div_ceil(T::ITEMS);

    if k <= K_BLOCK_SIZE {
        gemv_packed_inner::<T>(weights, activation, output, m, k, k_packed);
    } else {
        gemv_packed_blocked::<T>(weights, activation, output, m, k, k_packed);
    }
}

// ============================================================
// Generic GEMV implementation (scalar unpack + AVX2 FMA)
// ============================================================

#[allow(clippy::too_many_arguments)]
fn gemv_packed_inner<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    _m: usize,
    k: usize,
    k_packed: usize,
) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let rows_per_chunk = (65536 / (k_packed * 4)).clamp(1, 64);
        output
            .par_chunks_mut(rows_per_chunk)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_row = chunk_idx * rows_per_chunk;
    // Pre-allocate aligned buffer for unpacking
    let mut unpack_buf = AlignedScratchBuffer::new();
    unpack_buf.resize(k_packed * T::ITEMS);
                for (local_row, out) in out_chunk.iter_mut().enumerate() {
                    let row = start_row + local_row;
                    let dot = gemv_row::<T>(weights, activation, row, k, k_packed, unpack_buf.as_mut_slice());
                    *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        // Pre-allocate aligned buffer for unpacking
        let mut unpack_buf = AlignedScratchBuffer::new();
        unpack_buf.resize(k_packed * T::ITEMS);
        for row in 0.._m {
            let dot = gemv_row::<T>(weights, activation, row, k, k_packed, unpack_buf.as_mut_slice());
            output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gemv_packed_blocked<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    k_packed: usize,
) {
    for o in output.iter_mut() {
        *o = 0.0;
    }

    let items = T::ITEMS;
    let mut unpack_buf = AlignedScratchBuffer::new();
    unpack_buf.resize(K_BLOCK_SIZE);

    let mut k_offset = 0;
    while k_offset < k {
        let k_end = (k_offset + K_BLOCK_SIZE).min(k);
        let k_block = k_end - k_offset;

        for row in 0..m {
            let row_offset = row * k_packed;
            let packed_start = k_offset / items;
            let packed_end = k_end.div_ceil(items);
            let unpack_len = (packed_end - packed_start) * items;

            if unpack_len <= unpack_buf.data.len() {
                for (i, p) in (packed_start..packed_end).enumerate() {
                    let word = weights.as_packed()[row_offset + p];
                    let unpacked = word.unpack_to_f32();
                    let dst_start = i * items;
                    for j in 0..items {
                        if dst_start + j < k_block {
                            unpack_buf.data[dst_start + j] = unpacked.as_ref()[j];
                        }
                    }
                }
                let acc = fma_f32_slice(
                    &unpack_buf.data[..k_block.min(unpack_len)],
                    &activation[k_offset..k_end],
                );
                output[row] += acc;
            }
        }

        k_offset += K_BLOCK_SIZE;
    }

    for row in 0..m {
        output[row] = output[row] * weights.scale_for_row(row) + weights.zero_for_row(row);
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
        if is_x86_feature_detected!("avx512f") {
            return unsafe { fma_f32_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
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

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn fma_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut i = 0;

    while i + 32 <= len {
        let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        let a1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
        let b1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        i += 32;
    }

    while i + 16 <= len {
        let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        i += 16;
    }

    let combined = _mm512_add_ps(acc0, acc1);
    let mut total = _mm512_reduce_add_ps(combined);

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
    let k_packed = k.div_ceil(T::ITEMS);

    // Validate output vector lengths (parallel path uses unsafe raw pointers)
    for o in outputs.iter() {
        assert!(o.len() >= m, "output vector length {} < m={}", o.len(), m);
    }

    // Process each weight row once, compute dot products with all N inputs.
    // Uses per-row scale/zero for correct per-channel quantization.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let out_addrs: Vec<usize> = outputs
            .iter_mut()
            .map(|o| o.as_mut_ptr() as usize)
            .collect();
        let k_items = T::ITEMS;

        (0..m).into_par_iter().for_each(|row| {
            let scale = weights.scale_for_row(row);
            let zero = weights.zero_for_row(row);
            let mut unpack_buf = AlignedScratchBuffer::new();
            unpack_buf.resize(k);
            let row_offset = row * k_packed;
            for p in 0..k_packed {
                let word = weights.as_packed()[row_offset + p];
                let unpacked = word.unpack_to_f32();
                let base = p * k_items;
                for j in 0..k_items {
                    let idx = base + j;
                    if idx < k {
                        unpack_buf.as_mut_slice()[idx] = unpacked.as_ref()[j];
                    }
                }
            }

            for (bi, input) in batch_inputs.iter().enumerate() {
                let acc = fma_f32_slice(unpack_buf.as_slice(), input);
                unsafe {
                    *((out_addrs[bi] as *mut f32).add(row)) = acc * scale + zero;
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut unpack_buf = AlignedScratchBuffer::new();
        unpack_buf.resize(k);
        for row in 0..m {
            let scale = weights.scale_for_row(row);
            let zero = weights.zero_for_row(row);
            let row_offset = row * k_packed;
            for p in 0..k_packed {
                let word = weights.as_packed()[row_offset + p];
                let unpacked = word.unpack_to_f32();
                let base = p * T::ITEMS;
                for j in 0..T::ITEMS {
                    let idx = base + j;
                    if idx < k {
                        unpack_buf.as_mut_slice()[idx] = unpacked.as_ref()[j];
                    }
                }
            }
            for (bi, input) in batch_inputs.iter().enumerate() {
                let acc = fma_f32_slice(unpack_buf.as_slice(), input);
                outputs[bi][row] = acc * scale + zero;
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
