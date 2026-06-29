//! CPU GEMM/GEMV microkernels — extracted from microkernels.rs

#![allow(dead_code)]

use crate::dtypes::{F16x2, F32x1, I4x8, I8x4, PackedWord};
use crate::packed_tensor::PackedTensor;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

use super::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ============================================================
// GEMM constants
// ============================================================

/// Cache-blocked GEMV for packed types.
/// Tiles the K dimension to keep activation blocks in L2 cache.
const K_BLOCK_SIZE: usize = 8192;

// ============================================================
// Type-dispatched SIMD GEMV entry point
// ============================================================

#[inline(always)]
pub fn gemv_packed_simd<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    match (T::BIT_WIDTH, T::IS_FLOAT) {
        (8, false) => {
            // SAFETY: PackedTensor<T> has the same in-memory representation for all T
            // (a packed data slice + shape + scales + zeros). The bit-width match
            // guarantees the reinterpret cast is valid.
            let w = unsafe { &*(weights as *const _ as *const PackedTensor<I8x4>) };
            return gemv_i8x4_dispatch(w, activation, output);
        }
        (4, false) => {
            // SAFETY: Same reasoning as I8x4 case — memory layout is identical across
            // PackedWord types, and the match on BIT_WIDTH guarantees correctness.
            let w = unsafe { &*(weights as *const _ as *const PackedTensor<I4x8>) };
            return gemv_i4x8_dispatch(w, activation, output);
        }
        (16, true) => {
            // SAFETY: Same as above — F16x2 has the same packed layout as other
            // PackedWord types and the dispatch ensures bit-width compatibility.
            let w = unsafe { &*(weights as *const _ as *const PackedTensor<F16x2>) };
            return gemv_f16x2_dispatch(w, activation, output);
        }
        (32, true) => {
            // SAFETY: Same as above — F32x1 has the same packed layout as other
            // PackedWord types and the dispatch ensures bit-width compatibility.
            let w = unsafe { &*(weights as *const _ as *const PackedTensor<F32x1>) };
            return gemv_f32x1_dispatch(w, activation, output);
        }
        _ => {}
    }

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
// I8x4: AVX2 int8→f32 widening + FMA
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(unused_variables)]
fn gemv_i8x4_dispatch(weights: &PackedTensor<I8x4>, activation: &[f32], output: &mut [f32]) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(4);
    let weights_u32 = weights.as_u32();
    let act_sum = affine_sum_term(activation, k, 1.0);

    if has_avx512() {
        for row in 0..m {
            let dot = unsafe {
                gemv_row_i8x4_avx512(weights_u32, activation, row * k_packed, k, k_packed)
            };
            output[row] = apply_affine_dot(
                dot,
                weights.scale_for_row(row),
                weights.zero_for_row(row),
                act_sum,
            );
        }
    } else if has_avx2() {
        for row in 0..m {
            let dot =
                unsafe { gemv_row_i8x4_avx2(weights_u32, activation, row * k_packed, k, k_packed) };
            output[row] = apply_affine_dot(
                dot,
                weights.scale_for_row(row),
                weights.zero_for_row(row),
                act_sum,
            );
        }
    } else {
        for row in 0..m {
            let row_offset = row * k_packed;
            let dot =
                unsafe { gemv_row_i8x4_scalar(weights_u32, activation, row_offset, k, k_packed) };
            output[row] = apply_affine_dot(
                dot,
                weights.scale_for_row(row),
                weights.zero_for_row(row),
                act_sum,
            );
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
// SAFETY: Caller must ensure `weights_u32` and `activation` are valid slices,
// `row_offset` is in-bounds for `weights_u32`, and k/k_packed are correct.
unsafe fn gemv_row_i8x4_scalar(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    // NOTE: I8x4 values are stored as SIGNED i8 (range -128..127), not unsigned u8.
    // The cast (bytes[j] as i8) as f32 correctly interprets the packed data.
    let mut total = 0.0f32;
    for p in 0..k_packed {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        for j in 0..4 {
            let idx = p * 4 + j;
            if idx < k {
                total += (bytes[j] as i8) as f32 * activation[idx];
            }
        }
    }
    total
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
// SAFETY: Same as gemv_row_i8x4_scalar — caller ensures valid slices, correct
// offsets, and AVX2+FMA availability.
unsafe fn gemv_row_i8x4_avx2(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    while p + 1 < k_packed && act_idx + 8 <= k {
        if p + 4 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 4) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];

        let word_pair = _mm_set_epi32(0, 0, w1 as i32, w0 as i32);
        let weight_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(word_pair));

        let act = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc = _mm256_fmadd_ps(weight_f32, act, acc);

        p += 2;
        act_idx += 8;
    }

    let mut total = hsum256_ps(acc);

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
// SAFETY: Same as gemv_row_i8x4_scalar — caller ensures valid slices, correct
// offsets, and AVX512F+AVX512BW availability.
unsafe fn gemv_row_i8x4_avx512(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

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

        let bytes = _mm_set_epi32(w3 as i32, w2 as i32, w1 as i32, w0 as i32);
        let i32x16 = _mm512_cvtepi8_epi32(bytes);
        let weight_f32 = _mm512_cvtepi32_ps(i32x16);

        let act = _mm512_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm512_fmadd_ps(weight_f32, act, acc0);

        p += 4;
        act_idx += 16;
    }

    let mut total = _mm512_reduce_add_ps(acc0);

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

    while p < k_packed && act_idx + 4 <= k {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        let weights_f32: [f32; 4] = [
            bytes[0] as i8 as f32,
            bytes[1] as i8 as f32,
            bytes[2] as i8 as f32,
            bytes[3] as i8 as f32,
        ];
        let acts = [
            activation[act_idx],
            activation[act_idx + 1],
            activation[act_idx + 2],
            activation[act_idx + 3],
        ];

        let w_vec = _mm_loadu_ps(weights_f32.as_ptr());
        let a_vec = _mm_loadu_ps(acts.as_ptr());
        let prod = _mm_mul_ps(w_vec, a_vec);
        let hsum = _mm_hadd_ps(prod, prod);
        let final_sum = _mm_hadd_ps(hsum, hsum);
        total += _mm_cvtss_f32(final_sum);

        p += 1;
        act_idx += 4;
    }

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
// I4x8: AVX2 nibble extraction → int32 → f32 → FMA
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(unused_variables)]
fn gemv_i4x8_dispatch(weights: &PackedTensor<I4x8>, activation: &[f32], output: &mut [f32]) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(8);
    let weights_u32 = weights.as_u32();
    let act_sum = affine_sum_term(activation, k, 1.0);

    if has_avx512() {
        for row in 0..m {
            let dot = unsafe {
                gemv_row_i4x8_avx512(weights_u32, activation, row * k_packed, k, k_packed)
            };
            output[row] = apply_affine_dot(
                dot,
                weights.scale_for_row(row),
                weights.zero_for_row(row),
                act_sum,
            );
        }
    } else if has_avx2() {
        for row in 0..m {
            let dot =
                unsafe { gemv_row_i4x8_avx2(weights_u32, activation, row * k_packed, k, k_packed) };
            output[row] = apply_affine_dot(
                dot,
                weights.scale_for_row(row),
                weights.zero_for_row(row),
                act_sum,
            );
        }
    } else {
        gemv_generic_fallback(weights, activation, output);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
// SAFETY: Same as gemv_row_i8x4_avx2 — caller ensures valid slices, correct
// offsets, and AVX2+FMA availability.
unsafe fn gemv_row_i4x8_avx2(
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
    let sign_ext = _mm256_set1_epi32(8);

    while p + 2 <= k_packed && act_idx + 16 <= k {
        if p + 4 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 4) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];

        let shift0 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
        let w0v = _mm256_set1_epi32(w0 as i32);
        let w1v = _mm256_set1_epi32(w1 as i32);

        let nib_lo0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift0), mask_lo);
        let nib_lo1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift0), mask_lo);

        let signed_lo0 = _mm256_sub_epi32(_mm256_xor_si256(nib_lo0, sign_ext), sign_ext);
        let signed_lo1 = _mm256_sub_epi32(_mm256_xor_si256(nib_lo1, sign_ext), sign_ext);

        let fl0 = _mm256_cvtepi32_ps(signed_lo0);
        let fl1 = _mm256_cvtepi32_ps(signed_lo1);

        let al0 = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm256_fmadd_ps(fl0, al0, acc0);

        let al1 = _mm256_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc1 = _mm256_fmadd_ps(fl1, al1, acc1);

        p += 2;
        act_idx += 16;
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    let mut total = hsum256_ps(acc0);

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
// SAFETY: Same as gemv_row_i8x4_avx512 — caller ensures valid slices, correct
// offsets, and AVX512F availability.
unsafe fn gemv_row_i4x8_avx512(
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
    let sign_ext = _mm256_set1_epi32(8);

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

        let shift_lo = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
        let nib_lo0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift_lo), mask_lo);
        let nib_lo1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift_lo), mask_lo);

        let signed_lo0 = _mm256_sub_epi32(_mm256_xor_si256(nib_lo0, sign_ext), sign_ext);
        let signed_lo1 = _mm256_sub_epi32(_mm256_xor_si256(nib_lo1, sign_ext), sign_ext);

        let fl0 = _mm512_cvtepi32_ps(_mm512_castsi256_si512(signed_lo0));
        let fl1 = _mm512_cvtepi32_ps(_mm512_castsi256_si512(signed_lo1));
        let al0 = _mm512_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm512_fmadd_ps(fl0, al0, acc0);
        let al1 = _mm512_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc1 = _mm512_fmadd_ps(fl1, al1, acc1);

        p += 2;
        act_idx += 16;
    }

    let combined = _mm512_add_ps(acc0, acc1);
    let mut total = _mm512_reduce_add_ps(combined);

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
fn gemv_f16x2_dispatch(weights: &PackedTensor<F16x2>, activation: &[f32], output: &mut [f32]) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(2);
    let weights_u32 = weights.as_u32();
    let act_sum = affine_sum_term(activation, k, 1.0);

    if has_f16c() {
        for row in 0..m {
            let dot = unsafe {
                gemv_row_f16x2_f16c(weights_u32, activation, row * k_packed, k, k_packed)
            };
            output[row] = apply_affine_dot(
                dot,
                weights.scale_for_row(row),
                weights.zero_for_row(row),
                act_sum,
            );
        }
    } else {
        gemv_generic_fallback(weights, activation, output);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
// SAFETY: The input u32 values must be valid F16x2 bit patterns; AVX2+FMA+F16C
// availability is guaranteed by the caller.
unsafe fn u32x4_to_f32x8_f16c(w0: u32, w1: u32, w2: u32, w3: u32) -> __m256 {
    let half_bits = _mm_set_epi32(w3 as i32, w2 as i32, w1 as i32, w0 as i32);
    _mm256_cvtph_ps(half_bits)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
// SAFETY: Same as u32x4_to_f32x8_f16c — input u32 values must be valid F16x2
// bit patterns; AVX2+FMA+F16C is available.
unsafe fn u32x2_to_f32x4_f16c(w0: u32, w1: u32) -> __m128 {
    let half_bits = _mm_set_epi32(0, 0, w1 as i32, w0 as i32);
    _mm_cvtph_ps(half_bits)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
// SAFETY: Same as gemv_row_i8x4_avx2 — caller ensures valid slices, correct
// offsets, and AVX2+FMA+F16C availability.
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
fn gemv_f32x1_dispatch(weights: &PackedTensor<F32x1>, activation: &[f32], output: &mut [f32]) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    // SAFETY: `weights.as_u32()` returns a valid u32 slice; reinterpreting as
    // f32 is sound because F32x1 stores raw f32 bit patterns in u32 words.
    let weights_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(
            weights.as_u32().as_ptr() as *const f32,
            weights.packed_len(),
        )
    };
    let act_sum = affine_sum_term(activation, k, 1.0);

    for row in 0..m {
        let row_data = &weights_f32[row * k..(row + 1) * k];
        let dot = fma_f32_slice(row_data, activation);
        output[row] = apply_affine_dot(
            dot,
            weights.scale_for_row(row),
            weights.zero_for_row(row),
            act_sum,
        );
    }
}

// ============================================================
// Generic fallback
// ============================================================

#[inline(always)]
pub(crate) fn gemv_generic_fallback<T: PackedWord>(
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

#[inline(always)]
pub(crate) fn gemv_packed_inner<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    _m: usize,
    k: usize,
    k_packed: usize,
) {
    let act_sum = affine_sum_term(activation, k, 1.0);
    with_scratch(k_packed * T::ITEMS, |unpack_buf| {
        for row in 0.._m {
            let dot = gemv_row::<T>(weights, activation, row, k, k_packed, unpack_buf);
            output[row] = apply_affine_dot(
                dot,
                weights.scale_for_row(row),
                weights.zero_for_row(row),
                act_sum,
            );
        }
    });
}

#[inline(always)]
pub(crate) fn gemv_packed_blocked<T: PackedWord>(
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
    let act_sum = affine_sum_term(activation, k, 1.0);

    with_scratch(K_BLOCK_SIZE, |unpack_buf| {
        let mut k_offset = 0;
        while k_offset < k {
            let k_end = (k_offset + K_BLOCK_SIZE).min(k);
            let k_block = k_end - k_offset;

            for row in 0..m {
                let row_offset = row * k_packed;
                let packed_start = k_offset / items;
                let packed_end = k_end.div_ceil(items);
                let unpack_len = (packed_end - packed_start) * items;

                if unpack_len <= K_BLOCK_SIZE {
                    for p in packed_start..packed_end {
                        let word = weights.as_packed()[row_offset + p];
                        let unpacked = word.unpack_to_f32();
                        let base = p * items;
                        for j in 0..items {
                            let idx = base + j;
                            if idx >= k_offset && idx < k_end {
                                unpack_buf[idx - k_offset] = unpacked.as_ref()[j];
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

        for row in 0..m {
            output[row] = apply_affine_dot(
                output[row],
                weights.scale_for_row(row),
                weights.zero_for_row(row),
                act_sum,
            );
        }
    });
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
// BLIS-style tiled GEMV (cache-blocked K, register-blocked M)
// ============================================================

thread_local! {
    static BLAS_SCRATCH: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

fn with_blas_scratch<R>(f: impl FnOnce(&mut [f32]) -> R) -> R {
    BLAS_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.resize(KC * MR, 0.0);
        f(&mut buf)
    })
}

const MR: usize = 4;
const KC: usize = 8192;

#[inline(always)]
pub fn gemv_packed_tiled<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(T::ITEMS);
    let act_sum = affine_sum_term(activation, k, 1.0);

    for o in output.iter_mut() {
        *o = 0.0;
    }

    with_blas_scratch(|row_bufs| {
        let mut k_offset = 0;
        while k_offset < k {
            let k_end = (k_offset + KC).min(k);
            let k_block = k_end - k_offset;

            let mut row = 0;
            while row + MR <= m {
                micro_kernel::<T>(
                    weights,
                    &activation[k_offset..k_end],
                    &mut output[row..row + MR],
                    row,
                    k_offset,
                    k_end,
                    k_packed,
                    k_block,
                    row_bufs,
                );
                row += MR;
            }

            while row < m {
                let mut acc = 0.0f32;
                let row_offset = row * k_packed;
                let packed_start = k_offset / T::ITEMS;
                let packed_end = k_end.div_ceil(T::ITEMS);
                for p in packed_start..packed_end {
                    let word = weights.as_packed()[row_offset + p];
                    let unpacked = word.unpack_to_f32();
                    let base = p * T::ITEMS;
                    for j in 0..T::ITEMS {
                        let idx = base + j;
                        if idx >= k_offset && idx < k_end {
                            acc += unpacked.as_ref()[j] * activation[idx];
                        }
                    }
                }
                output[row] += acc;
                row += 1;
            }

            k_offset += KC;
        }

        for row in 0..m {
            output[row] = apply_affine_dot(
                output[row],
                weights.scale_for_row(row),
                weights.zero_for_row(row),
                act_sum,
            );
        }
    });
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn micro_kernel<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    start_row: usize,
    k_start: usize,
    k_end: usize,
    k_packed: usize,
    k_block: usize,
    row_bufs: &mut [f32],
) {
    debug_assert_eq!(output.len(), MR);
    debug_assert!(row_bufs.len() >= KC * MR);

    for r in 0..MR {
        let row_start = r * KC;
        if k_block < KC {
            row_bufs[row_start + k_block..row_start + KC].fill(0.0);
        }
        let row = start_row + r;
        let row_offset = row * k_packed;
        let packed_start = k_start / T::ITEMS;
        let packed_end = k_end.div_ceil(T::ITEMS);

        for p in packed_start..packed_end {
            let word = weights.as_packed()[row_offset + p];
            let unpacked = word.unpack_to_f32();
            let base = p * T::ITEMS;
            for j in 0..T::ITEMS {
                let idx = base + j;
                if idx >= k_start && idx < k_end {
                    row_bufs[row_start + (idx - k_start)] = unpacked.as_ref()[j];
                }
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature check guarantees AVX2+FMA availability; all slices
            // are valid and correctly sized (row_bufs is KC*MR, activation is k_block).
            unsafe {
                micro_kernel_avx2(row_bufs, activation, output, k_block);
            }
            return;
        }
    }

    for r in 0..MR {
        let row_start = r * KC;
        let mut acc = 0.0f32;
        for kk in 0..k_block {
            acc += row_bufs[row_start + kk] * activation[kk];
        }
        output[r] += acc;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
// SAFETY: Caller must ensure `row_bufs` is at least KC * MR elements, `activation`
// is at least `k` elements, and `output` is at least MR elements. AVX2+FMA is
// guaranteed by the target_feature attribute.
unsafe fn micro_kernel_avx2(row_bufs: &[f32], activation: &[f32], output: &mut [f32], k: usize) {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut kk = 0;

    while kk + 8 <= k {
        if kk + 32 < k {
            _mm_prefetch(activation.as_ptr().add(kk + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(row_bufs.as_ptr().add(kk + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(
                row_bufs.as_ptr().add(KC + kk + 32) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                row_bufs.as_ptr().add(2 * KC + kk + 32) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                row_bufs.as_ptr().add(3 * KC + kk + 32) as *const i8,
                _MM_HINT_T0,
            );
        }

        let act = _mm256_loadu_ps(activation.as_ptr().add(kk));

        let row0 = _mm256_loadu_ps(row_bufs.as_ptr().add(kk));
        acc0 = _mm256_fmadd_ps(row0, act, acc0);

        let row1 = _mm256_loadu_ps(row_bufs.as_ptr().add(KC + kk));
        acc1 = _mm256_fmadd_ps(row1, act, acc1);

        let row2 = _mm256_loadu_ps(row_bufs.as_ptr().add(2 * KC + kk));
        acc2 = _mm256_fmadd_ps(row2, act, acc2);

        let row3 = _mm256_loadu_ps(row_bufs.as_ptr().add(3 * KC + kk));
        acc3 = _mm256_fmadd_ps(row3, act, acc3);

        kk += 8;
    }

    let mut tail0 = 0.0f32;
    let mut tail1 = 0.0f32;
    let mut tail2 = 0.0f32;
    let mut tail3 = 0.0f32;
    while kk < k {
        let act = *activation.as_ptr().add(kk);
        tail0 += *row_bufs.as_ptr().add(kk) * act;
        tail1 += *row_bufs.as_ptr().add(KC + kk) * act;
        tail2 += *row_bufs.as_ptr().add(2 * KC + kk) * act;
        tail3 += *row_bufs.as_ptr().add(3 * KC + kk) * act;
        kk += 1;
    }

    *output.as_mut_ptr().add(0) += hsum256_ps(acc0) + tail0;
    *output.as_mut_ptr().add(1) += hsum256_ps(acc1) + tail1;
    *output.as_mut_ptr().add(2) += hsum256_ps(acc2) + tail2;
    *output.as_mut_ptr().add(3) += hsum256_ps(acc3) + tail3;
}

// ============================================================
// SIMD unpack helpers
// ============================================================

// ============================================================
// Top-level CPU GEMV dispatcher
// ============================================================

const TILED_K_THRESHOLD: usize = 4096;

#[inline(always)]
pub fn gemv_cpu<T: PackedWord>(weights: &PackedTensor<T>, activation: &[f32], output: &mut [f32]) {
    let k = weights.shape()[1];
    if k > TILED_K_THRESHOLD {
        gemv_packed_tiled(weights, activation, output);
    } else {
        gemv_packed_simd(weights, activation, output);
    }
}

// ============================================================
// Top-level CPU GEMM (batch of GEMV calls)
// ============================================================

#[inline(always)]
pub fn gemm_cpu<T: PackedWord>(
    weights: &PackedTensor<T>,
    batch_inputs: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
) {
    assert_eq!(batch_inputs.len(), outputs.len());
    gemm_packed_batched(weights, batch_inputs, outputs);
}

#[inline(always)]
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

    for o in outputs.iter() {
        assert!(o.len() >= m, "output vector length {} < m={}", o.len(), m);
    }

    with_scratch(k, |unpack_buf| {
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
                        unpack_buf[idx] = unpacked.as_ref()[j];
                    }
                }
            }
            for (bi, input) in batch_inputs.iter().enumerate() {
                let acc = fma_f32_slice(unpack_buf, input);
                let input_sum = affine_sum_term(input, k, zero);
                outputs[bi][row] = apply_affine_dot(acc, scale, zero, input_sum);
            }
        }
    });
}

// ============================================================
// Flat-buffer GEMM (avoids Vec<Vec> allocation storm)
// ============================================================

#[inline(always)]
pub fn gemm_cpu_flat<T: PackedWord>(
    weights: &PackedTensor<T>,
    batch_inputs: &[f32],
    outputs: &mut [f32],
    num_pixels: usize,
    col_w: usize,
    oc: usize,
) {
    gemm_batch_packed_simd(weights, batch_inputs, outputs, num_pixels, col_w, oc);
}

// ============================================================
// I8 × I8x4 quantized GEMM  (scalar, no SIMD)
// ============================================================

/// Flat-buffer I8 × I8x4 quantized GEMM.
///
/// Activation payload format: [scale_f32][zp_f32][i8_data...]
/// (8-byte header followed by `m × k` signed i8 elements).
///
/// Weights are `PackedTensor<I8x4>` with shape [N, K].
/// Output is `[m × n]` f32.
///
/// Computes the affine dot product exactly for the stored quantized values:
///   weight_i = qW_i * scale_w + zero_w
///   activation_i = qA_i * scale_a + zp_a
///   output = Σ_i weight_i * activation_i
///
/// The zp_a == 0 path keeps the qW*qA term as an i32 dot product and adds
/// the zero_w contribution as `zero_w * scale_a * ΣqA`.
#[inline(always)]
pub fn gemm_cpu_flat_i8_i8x4(
    weights: &PackedTensor<I8x4>,
    activation_payload: &[u8],
    outputs: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let shape = weights.shape();
    let oc = shape[0];
    let k_in = shape[1];
    assert_eq!(
        oc, n,
        "gemm_cpu_flat_i8_i8x4: weight output channels (oc={oc}) must equal n={n}"
    );
    assert_eq!(
        k_in, k,
        "gemm_cpu_flat_i8_i8x4: weight input channels (k_in={k_in}) must equal k={k}"
    );

    let (activation_affine, act_i8) = parse_i8_activation_payload(activation_payload);

    // Non-zero activation zero-point requires lane-by-lane dequantization.
    if activation_affine.zero != 0.0 {
        let k_packed = k.div_ceil(I8x4::ITEMS);
        // M-outer loop: activation data for each pixel is reused across all output channels
        for bi in 0..m {
            let act_base = bi * k;
            for row in 0..oc {
                let scale_w = weights.scale_for_row(row);
                let zero_w = weights.zero_for_row(row);
                let row_offset = row * k_packed;
                let mut acc_f32 = 0.0f32;
                let mut activation_sum = 0.0f32;
                for kk in 0..k {
                    let a = activation_affine.dequantize(act_i8[act_base + kk] as i8);
                    let word = weights.as_packed()[row_offset + kk / 4].0;
                    let bytes = word.to_le_bytes();
                    let q_w = bytes[kk % 4] as i8 as f32;
                    acc_f32 += a * q_w;
                    activation_sum += a;
                }
                outputs[bi * n + row] = apply_affine_dot(acc_f32, scale_w, zero_w, activation_sum);
            }
        }
        return;
    }

    // Fast path: zero activation offset keeps the qW*qA term in integer space,
    // then reuses the shared Σactivation helper for the affine zero_w term.
    let k_packed = k.div_ceil(I8x4::ITEMS);

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if crate::backend::cpu::microkernels::has_avx2() {
        // SAFETY: AVX2 feature check passed; all slices are valid and correctly
        // sized (weights, act_i8, outputs), and the integer→float reinterpretations
        // use correct lane counts for the packed formats.
        unsafe {
            for bi in 0..m {
                let act_base = bi * k;
                for row in 0..oc {
                    let scale_w = weights.scale_for_row(row);
                    let zero_w = weights.zero_for_row(row);
                    let row_offset = row * k_packed;
                    let mut acc = _mm256_setzero_ps();
                    let mut act_sum_vec = _mm256_setzero_ps();
                    let mut p = 0usize;

                    while p + 1 < k_packed && p * 4 + 8 <= k {
                        let w0 = weights.as_packed()[row_offset + p].0;
                        let w1 = weights.as_packed()[row_offset + p + 1].0;
                        let word_pair = _mm_set_epi32(0, 0, w1 as i32, w0 as i32);
                        let qw_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(word_pair));

                        let act_ptr = act_i8.as_ptr().add(act_base + p * 4);
                        let act_128 = _mm_loadl_epi64(act_ptr as *const __m128i);
                        let act_lo4 = _mm256_cvtepi8_epi32(act_128);
                        let act_hi4 = _mm256_cvtepi8_epi32(_mm_bsrli_si128(act_128, 4));
                        let act_i32 =
                            _mm256_insertf128_si256(act_lo4, _mm256_castsi256_si128(act_hi4), 1);
                        let act_f32 = _mm256_cvtepi32_ps(act_i32);

                        acc = _mm256_fmadd_ps(qw_f32, act_f32, acc);
                        act_sum_vec = _mm256_add_ps(act_sum_vec, act_f32);
                        p += 2;
                    }

                    let mut acc_tail = 0.0f32;
                    let mut act_sum_tail = 0.0f32;
                    while p < k_packed {
                        let word = weights.as_packed()[row_offset + p].0;
                        let bytes = word.to_le_bytes();
                        for lane in 0..4 {
                            let idx = p * 4 + lane;
                            if idx < k {
                                let q_w = bytes[lane] as i8 as i32;
                                let q_a = act_i8[act_base + idx] as i8 as i32;
                                acc_tail += q_w as f32 * q_a as f32;
                                act_sum_tail += q_a as f32;
                            }
                        }
                        p += 1;
                    }

                    let raw_acc = hsum256_ps(acc) + acc_tail;
                    let raw_q_sum = hsum256_ps(act_sum_vec) + act_sum_tail;
                    let dot = raw_acc * activation_affine.scale;
                    let input_sum =
                        raw_q_sum * activation_affine.scale + (k as f32) * activation_affine.zero;
                    outputs[bi * n + row] = apply_affine_dot(dot, scale_w, zero_w, input_sum);
                }
            }
        }
        return;
    }

    for bi in 0..m {
        let act_base = bi * k;
        for row in 0..oc {
            let scale_w = weights.scale_for_row(row);
            let zero_w = weights.zero_for_row(row);
            let row_offset = row * k_packed;

            let mut acc: i32 = 0;
            let mut q_activation_sum: i32 = 0;

            for p in 0..k_packed {
                let word = weights.as_packed()[row_offset + p].0;
                let bytes = word.to_le_bytes();
                for lane in 0..4 {
                    let idx = p * 4 + lane;
                    if idx < k {
                        let q_w = bytes[lane] as i8 as i32;
                        let q_a = act_i8[act_base + idx] as i8 as i32;
                        acc += q_w * q_a;
                        q_activation_sum += q_a;
                    }
                }
            }
            outputs[bi * n + row] = apply_affine_dot(
                (acc as f32) * activation_affine.scale,
                scale_w,
                zero_w,
                activation_affine.sum_from_q_sum(q_activation_sum, k),
            );
        }
    }
}

/// Flat-buffer I8 × I4x8 quantized GEMM.
///
/// Semantics mirror `gemm_cpu_flat_i8_i8x4`, but each packed weight word stores
/// eight signed 4-bit qW lanes (`[-8, 7]`) before the affine `qW * scale_w + zero_w`
/// dequantization is applied.
#[inline(always)]
pub fn gemm_cpu_flat_i8_i4x8(
    weights: &PackedTensor<I4x8>,
    activation_payload: &[u8],
    outputs: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let shape = weights.shape();
    let oc = shape[0];
    let k_in = shape[1];
    assert_eq!(
        oc, n,
        "gemm_cpu_flat_i8_i4x8: weight output channels (oc={oc}) must equal n={n}"
    );
    assert_eq!(
        k_in, k,
        "gemm_cpu_flat_i8_i4x8: weight input channels (k_in={k_in}) must equal k={k}"
    );

    let (activation_affine, act_i8) = parse_i8_activation_payload(activation_payload);
    let k_packed = k.div_ceil(I4x8::ITEMS);

    if activation_affine.zero != 0.0 {
        // M-outer loop: activation data reused across all output channels
        for bi in 0..m {
            let act_base = bi * k;
            for row in 0..oc {
                let scale_w = weights.scale_for_row(row);
                let zero_w = weights.zero_for_row(row);
                let row_offset = row * k_packed;
                let mut acc_f32 = 0.0f32;
                let mut activation_sum = 0.0f32;
                for p in 0..k_packed {
                    let word = weights.as_packed()[row_offset + p].0;
                    for lane in 0..I4x8::ITEMS {
                        let idx = p * I4x8::ITEMS + lane;
                        if idx < k {
                            let nibble = (word >> (lane * 4)) & 0xF;
                            let signed = if nibble & 0x8 != 0 {
                                (nibble | 0xFFFFFFF0) as i32
                            } else {
                                nibble as i32
                            };
                            let a = activation_affine.dequantize(act_i8[act_base + idx] as i8);
                            acc_f32 += signed as f32 * a;
                            activation_sum += a;
                        }
                    }
                }
                outputs[bi * n + row] = apply_affine_dot(acc_f32, scale_w, zero_w, activation_sum);
            }
        }
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if crate::backend::cpu::microkernels::has_avx2() {
        // SAFETY: AVX2 feature check passed; all slices are valid and correctly
        // sized, and the I4x8 nibble extraction follows the packed layout.
        unsafe {
            for bi in 0..m {
                let act_base = bi * k;
                for row in 0..oc {
                    let scale_w = weights.scale_for_row(row);
                    let zero_w = weights.zero_for_row(row);
                    let row_offset = row * k_packed;
                    let mut acc0 = _mm256_setzero_ps();
                    let mut acc1 = _mm256_setzero_ps();
                    let mut act_sum_vec = _mm256_setzero_ps();
                    let mut p = 0usize;
                    let mask_lo = _mm256_set1_epi32(0xF);
                    let sign_ext = _mm256_set1_epi32(8);
                    let shift0 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);

                    while p + 1 < k_packed && p * 8 + 16 <= k {
                        let w0 = weights.as_packed()[row_offset + p].0;
                        let w1 = weights.as_packed()[row_offset + p + 1].0;
                        let w0v = _mm256_set1_epi32(w0 as i32);
                        let w1v = _mm256_set1_epi32(w1 as i32);
                        let nib_lo0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift0), mask_lo);
                        let nib_lo1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift0), mask_lo);
                        let signed_lo0 =
                            _mm256_sub_epi32(_mm256_xor_si256(nib_lo0, sign_ext), sign_ext);
                        let signed_lo1 =
                            _mm256_sub_epi32(_mm256_xor_si256(nib_lo1, sign_ext), sign_ext);
                        let fl0 = _mm256_cvtepi32_ps(signed_lo0);
                        let fl1 = _mm256_cvtepi32_ps(signed_lo1);

                        let act_ptr = act_i8.as_ptr().add(act_base + p * 8);
                        let act0_128 = _mm_loadl_epi64(act_ptr as *const __m128i);
                        let act0_lo4 = _mm256_cvtepi8_epi32(act0_128);
                        let act0_hi4 = _mm256_cvtepi8_epi32(_mm_bsrli_si128(act0_128, 4));
                        let act_i32_0 =
                            _mm256_insertf128_si256(act0_lo4, _mm256_castsi256_si128(act0_hi4), 1);
                        let al0 = _mm256_cvtepi32_ps(act_i32_0);

                        let act1_128 = _mm_loadl_epi64(act_ptr.add(8) as *const __m128i);
                        let act1_lo4 = _mm256_cvtepi8_epi32(act1_128);
                        let act1_hi4 = _mm256_cvtepi8_epi32(_mm_bsrli_si128(act1_128, 4));
                        let act_i32_1 =
                            _mm256_insertf128_si256(act1_lo4, _mm256_castsi256_si128(act1_hi4), 1);
                        let al1 = _mm256_cvtepi32_ps(act_i32_1);

                        acc0 = _mm256_fmadd_ps(fl0, al0, acc0);
                        acc1 = _mm256_fmadd_ps(fl1, al1, acc1);

                        act_sum_vec = _mm256_add_ps(act_sum_vec, al0);
                        act_sum_vec = _mm256_add_ps(act_sum_vec, al1);

                        p += 2;
                    }

                    let mut acc_tail = 0.0f32;
                    let mut act_sum_tail = 0.0f32;
                    while p < k_packed {
                        let word = weights.as_packed()[row_offset + p].0;
                        for lane in 0..I4x8::ITEMS {
                            let idx = p * I4x8::ITEMS + lane;
                            if idx < k {
                                let nibble = (word >> (lane * 4)) & 0xF;
                                let signed = if nibble & 0x8 != 0 {
                                    (nibble | 0xFFFFFFF0) as i32
                                } else {
                                    nibble as i32
                                };
                                let q_a = act_i8[act_base + idx] as i8 as i32;
                                acc_tail += signed as f32 * (q_a as f32);
                                act_sum_tail += q_a as f32;
                            }
                        }
                        p += 1;
                    }

                    let raw_acc = hsum256_ps(_mm256_add_ps(acc0, acc1)) + acc_tail;
                    let raw_q_sum = hsum256_ps(act_sum_vec) + act_sum_tail;
                    let dot = raw_acc * activation_affine.scale;
                    let input_sum =
                        raw_q_sum * activation_affine.scale + (k as f32) * activation_affine.zero;
                    outputs[bi * n + row] = apply_affine_dot(dot, scale_w, zero_w, input_sum);
                }
            }
        }
        return;
    }

    for bi in 0..m {
        let act_base = bi * k;
        for row in 0..oc {
            let scale_w = weights.scale_for_row(row);
            let zero_w = weights.zero_for_row(row);
            let row_offset = row * k_packed;
            let mut acc = 0.0f32;
            let mut q_activation_sum: i32 = 0;
            for p in 0..k_packed {
                let word = weights.as_packed()[row_offset + p].0;
                for lane in 0..I4x8::ITEMS {
                    let idx = p * I4x8::ITEMS + lane;
                    if idx < k {
                        let nibble = (word >> (lane * 4)) & 0xF;
                        let signed = if nibble & 0x8 != 0 {
                            (nibble | 0xFFFFFFF0) as i32
                        } else {
                            nibble as i32
                        };
                        let q_a = act_i8[act_base + idx] as i8 as i32;
                        acc += signed as f32 * (q_a as f32);
                        q_activation_sum += q_a;
                    }
                }
            }
            outputs[bi * n + row] = apply_affine_dot(
                acc * activation_affine.scale,
                scale_w,
                zero_w,
                activation_affine.sum_from_q_sum(q_activation_sum, k),
            );
        }
    }
}

// ============================================================
// Batch GEMM — K-tiled for cache reuse across batch rows
// ============================================================

#[inline(always)]
pub fn gemm_batch_packed_simd<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    _n: usize,
    k: usize,
    m: usize,
) {
    let k_packed = k.div_ceil(T::ITEMS);

    for o in output.iter_mut() {
        *o = 0.0;
    }

    let mut k_start = 0;
    while k_start < k {
        let k_end = (k_start + K_BLOCK_SIZE).min(k);
        let k_len = k_end - k_start;
        let packed_start = k_start / T::ITEMS;
        let packed_end = k_end.div_ceil(T::ITEMS);

        with_scratch(k_len, |unpack_buf| {
            for row in 0..m {
                let row_offset = row * k_packed;

                for p in packed_start..packed_end {
                    let word = weights.as_packed()[row_offset + p];
                    let unpacked = word.unpack_to_f32();
                    let base = p * T::ITEMS;
                    for j in 0..T::ITEMS {
                        let idx = base + j;
                        if idx >= k_start && idx < k_end {
                            unpack_buf[idx - k_start] = unpacked.as_ref()[j];
                        }
                    }
                }

                #[cfg(feature = "parallel")]
                {
                    output
                        .par_chunks_mut(m)
                        .enumerate()
                        .for_each(|(bi, chunk)| {
                            let act_slice = &activation[bi * k + k_start..bi * k + k_end];
                            let acc = fma_f32_slice(unpack_buf, act_slice);
                            chunk[row] += acc;
                        });
                }
                #[cfg(not(feature = "parallel"))]
                for bi in 0..n {
                    let act_slice = &activation[bi * k + k_start..bi * k + k_end];
                    let acc = fma_f32_slice(unpack_buf, act_slice);
                    output[bi * m + row] += acc;
                }
            }
        });

        k_start += K_BLOCK_SIZE;
    }

    #[cfg(feature = "parallel")]
    {
        let k_usize = k;
        output
            .par_chunks_mut(m)
            .enumerate()
            .for_each(|(bi, chunk)| {
                let input = &activation[bi * k_usize..(bi + 1) * k_usize];
                let input_sum = affine_sum_term(input, k_usize, 1.0);
                for row in 0..m {
                    let scale = weights.scale_for_row(row);
                    let zero = weights.zero_for_row(row);
                    chunk[row] = apply_affine_dot(chunk[row], scale, zero, input_sum);
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for bi in 0..n {
            let input = &activation[bi * k..(bi + 1) * k];
            let input_sum = affine_sum_term(input, k, 1.0);
            for row in 0..m {
                let scale = weights.scale_for_row(row);
                let zero = weights.zero_for_row(row);
                output[bi * m + row] =
                    apply_affine_dot(output[bi * m + row], scale, zero, input_sum);
            }
        }
    }
}
