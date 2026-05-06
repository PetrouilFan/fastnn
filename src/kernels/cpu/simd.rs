//! CPU simd kernels.

#![allow(unused_imports)]
#![allow(clippy::missing_safety_doc)]

use crate::autograd::{AutogradMeta, Edge, Node};
use crate::dispatcher::{register, DispatchKey, KernelFn};
use crate::iterator::TensorIterator;
use crate::kernels::blas::{
    matmul_blas, matmul_blas_into, matmul_blas_with_transpose,
    matmul_blas_with_transpose_into, MIN_BLAS_SIZE,
};
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use std::sync::Arc;
use super::*;

use wide::f32x4;
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
use wide::f32x8;

#[inline]
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn from_slice_unaligned_f32x8(slice: &[f32]) -> f32x8 {
    let arr: [f32; 8] = slice.try_into().unwrap();
    f32x8::new(arr)
}

#[inline]
    #[allow(dead_code)]
pub fn from_slice_unaligned_f32x4(slice: &[f32]) -> f32x4 {
    let arr: [f32; 4] = slice.try_into().unwrap();
    f32x4::new(arr)
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub fn relu_simd(input: &[f32], output: &mut [f32]) {
    let zero = f32x4::ZERO;
    let (chunks, remainder) = input.as_chunks::<4>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x4::from(*in_chunk);
        let result = v.max(zero);
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.max(0.0);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn fast_exp_avx2(x: __m256) -> __m256 {
    // Cephes-style fast exp approximation accurate to ~1 ULP for f32
    // Algorithm: exp(x) = 2^(x/ln2) = 2^floor(x/ln2) * 2^frac(x/ln2)
    // Then approximate 2^frac using a degree-5 polynomial.
    let ln2_rcp = _mm256_set1_ps(std::f32::consts::LOG2_E); // 1/ln(2)
    let ln2_hi = _mm256_set1_ps(0.693_359_4_f32);
    let ln2_lo = _mm256_set1_ps(-2.121_944_4e-4_f32);
    let half = _mm256_set1_ps(0.5_f32);
    let one = _mm256_set1_ps(1.0_f32);
    let clamp_hi = _mm256_set1_ps(88.376_26_f32); // max before f32 overflow
    let clamp_lo = _mm256_set1_ps(-88.376_26_f32);

    // Polynomial coefficients for 2^x on [0,1]: from Cephes
    let p0 = _mm256_set1_ps(1.987_569_3e-4_f32);
    let p1 = _mm256_set1_ps(1.398_199e-3_f32);
    let p2 = _mm256_set1_ps(8.333_452e-3_f32);
    let p3 = _mm256_set1_ps(4.166_579_5e-2_f32);
    let p4 = _mm256_set1_ps(1.666_666_7e-1_f32);
    let p5 = _mm256_set1_ps(5.0e-1_f32);

    let x = _mm256_min_ps(_mm256_max_ps(x, clamp_lo), clamp_hi);
    // n = round(x / ln2)
    let t = _mm256_fmadd_ps(x, ln2_rcp, half);
    let n = _mm256_floor_ps(t);
    // x = x - n*ln2 (range reduction)
    let x = _mm256_fnmadd_ps(n, ln2_hi, x);
    let x = _mm256_fnmadd_ps(n, ln2_lo, x);
    // Polynomial evaluation: p(x) = 1 + x*(p5 + x*(p4 + x*(p3 + x*(p2 + x*(p1 + x*p0)))))
    let r = p0;
    let r = _mm256_fmadd_ps(r, x, p1);
    let r = _mm256_fmadd_ps(r, x, p2);
    let r = _mm256_fmadd_ps(r, x, p3);
    let r = _mm256_fmadd_ps(r, x, p4);
    let r = _mm256_fmadd_ps(r, x, p5);
    let r = _mm256_fmadd_ps(r, x, one);
    // Scale by 2^n using integer exponent trick
    let n_int = _mm256_cvtps_epi32(n);
    let bias = _mm256_set1_epi32(127);
    let n_biased = _mm256_add_epi32(n_int, bias);
    let n_shifted = _mm256_slli_epi32(n_biased, 23);
    let scale = _mm256_castsi256_ps(n_shifted);
    _mm256_mul_ps(r, scale)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn fast_exp_avx512(x: __m512) -> __m512 {
    // Process 16 floats as 2x 8 floats with AVX2
    let x_lo = _mm512_castps512_ps256(x);
    let x_hi = _mm512_extractf32x8_ps(x, 1);

    let exp_lo = fast_exp_avx2(x_lo);
    let exp_hi = fast_exp_avx2(x_hi);

    let result_lo = _mm512_castps256_ps512(exp_lo);
    let result_hi = exp_hi;

    _mm512_insertf32x8(result_lo, result_hi, 1)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn fast_log_avx2(x: __m256) -> __m256 {
    // ln(x) = log2(x) * ln(2)
    // log2(x) = (exponent - 127) + log2(mantissa)
    // log2(mantissa) approximated with polynomial on [1,2]
    let one = _mm256_set1_ps(1.0f32);
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
    let clamp_hi = _mm256_set1_ps(1e30f32);
    let clamp_lo = _mm256_set1_ps(1e-30f32);

    // Clamp to avoid issues with denormals
    let x = _mm256_min_ps(_mm256_max_ps(x, clamp_lo), clamp_hi);

    // Extract exponent and mantissa
    let x_i = _mm256_castps_si256(x);
    let exp_i = _mm256_srli_epi32(x_i, 23);
    let exp_f = _mm256_cvtepi32_ps(exp_i);
    let exp = _mm256_sub_ps(exp_f, _mm256_set1_ps(127.0f32));

    // Mask mantissa bits
    let mantissa_i = _mm256_and_si256(x_i, _mm256_set1_epi32(0x007fffff));
    let mantissa_i = _mm256_or_si256(mantissa_i, _mm256_set1_epi32(0x3f800000));
    let mantissa = _mm256_castsi256_ps(mantissa_i);

    // Polynomial approximation for log2(mantissa) on [1,2]
    // log2(y) ≈ ((y-1)/(y+1)) * (a0 + a2*(y-1)^2 + ...)
    let y = mantissa;
    let y_minus_1 = _mm256_sub_ps(y, one);
    let y_plus_1 = _mm256_add_ps(y, one);
    let z = _mm256_div_ps(y_minus_1, y_plus_1);
    let z2 = _mm256_mul_ps(z, z);

    let a0 = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let a1 = _mm256_set1_ps(0.721205f32);
    let a2 = _mm256_set1_ps(0.480898f32);
    let a3 = _mm256_set1_ps(0.252011f32);
    let a4 = _mm256_set1_ps(0.152576f32);

    let log2_mantissa = _mm256_fmadd_ps(z2, a4, a3);
    let log2_mantissa = _mm256_fmadd_ps(z2, log2_mantissa, a2);
    let log2_mantissa = _mm256_fmadd_ps(z2, log2_mantissa, a1);
    let log2_mantissa = _mm256_fmadd_ps(z2, log2_mantissa, a0);
    let log2_mantissa = _mm256_mul_ps(z, log2_mantissa);

    let log2_x = _mm256_add_ps(exp, log2_mantissa);
    _mm256_mul_ps(log2_x, ln2)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn fast_log_avx512(x: __m512) -> __m512 {
    // Use fast_log_avx2 on each 256-bit half
    let x_lo = _mm512_castps512_ps256(x);
    let x_hi = _mm512_extractf32x8_ps(x, 1);

    let log_lo = fast_log_avx2(x_lo);
    let log_hi = fast_log_avx2(x_hi);

    let result_lo = _mm512_castps256_ps512(log_lo);
    let result_hi = log_hi;

    _mm512_insertf32x8(result_lo, result_hi, 1)
}

/// Horizontal sum of __m256 — used by softmax and other AVX2 kernels.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x0E);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn add_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let a1 = _mm256_loadu_ps((a_usize + (i + 8) * 4) as *const f32);
        let b0 = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let b1 = _mm256_loadu_ps((b_usize + (i + 8) * 4) as *const f32);
        let r0 = _mm256_add_ps(a0, b0);
        let r1 = _mm256_add_ps(a1, b1);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, r0);
        _mm256_storeu_ps((out_usize + (i + 8) * 4) as *mut f32, r1);
        i += 16;
    }
    // Vectorize tail with 128-bit vectors for better performance
    while i + 4 <= end {
        let a_vec = _mm_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm_loadu_ps((b_usize + i * 4) as *const f32);
        let result = _mm_add_ps(a_vec, b_vec);
        _mm_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 4;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) + *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
pub unsafe fn add_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = vld1q_f32((a_usize + i * 4) as *const f32);
        let a1 = vld1q_f32((a_usize + (i + 4) * 4) as *const f32);
        let a2 = vld1q_f32((a_usize + (i + 8) * 4) as *const f32);
        let a3 = vld1q_f32((a_usize + (i + 12) * 4) as *const f32);

        let b0 = vld1q_f32((b_usize + i * 4) as *const f32);
        let b1 = vld1q_f32((b_usize + (i + 4) * 4) as *const f32);
        let b2 = vld1q_f32((b_usize + (i + 8) * 4) as *const f32);
        let b3 = vld1q_f32((b_usize + (i + 12) * 4) as *const f32);

        let r0 = vaddq_f32(a0, b0);
        let r1 = vaddq_f32(a1, b1);
        let r2 = vaddq_f32(a2, b2);
        let r3 = vaddq_f32(a3, b3);

        vst1q_f32((out_usize + i * 4) as *mut f32, r0);
        vst1q_f32((out_usize + (i + 4) * 4) as *mut f32, r1);
        vst1q_f32((out_usize + (i + 8) * 4) as *mut f32, r2);
        vst1q_f32((out_usize + (i + 12) * 4) as *mut f32, r3);

        i += 16;
    }
    while i + 4 <= end {
        let a_vec = vld1q_f32((a_usize + i * 4) as *const f32);
        let b_vec = vld1q_f32((b_usize + i * 4) as *const f32);
        let result = vaddq_f32(a_vec, b_vec);
        vst1q_f32((out_usize + i * 4) as *mut f32, result);
        i += 4;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) + *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn add_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a_vec = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm512_loadu_ps((b_usize + i * 4) as *const f32);
        let result = _mm512_add_ps(a_vec, b_vec);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Use masked operations for tails instead of falling back to AVX2
    let remaining = end - i;
    if remaining > 0 {
        let mask = (1u16 << remaining) - 1;
        let a_vec = _mm512_maskz_loadu_ps(mask, (a_usize + i * 4) as *const f32);
        let b_vec = _mm512_maskz_loadu_ps(mask, (b_usize + i * 4) as *const f32);
        let result = _mm512_add_ps(a_vec, b_vec);
        _mm512_mask_storeu_ps((out_usize + i * 4) as *mut f32, mask, result);
    }
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn add_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    for i in start..end {
        unsafe {
            *((out_usize + i * 4) as *mut f32) =
                *((a_usize + i * 4) as *const f32) + *((b_usize + i * 4) as *const f32);
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn mul_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let a1 = _mm256_loadu_ps((a_usize + (i + 8) * 4) as *const f32);
        let b0 = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let b1 = _mm256_loadu_ps((b_usize + (i + 8) * 4) as *const f32);
        let r0 = _mm256_mul_ps(a0, b0);
        let r1 = _mm256_mul_ps(a1, b1);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, r0);
        _mm256_storeu_ps((out_usize + (i + 8) * 4) as *mut f32, r1);
        i += 16;
    }
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let result = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) * *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn mul_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = vld1q_f32((a_usize + i * 4) as *const f32);
        let a1 = vld1q_f32((a_usize + (i + 4) * 4) as *const f32);
        let a2 = vld1q_f32((a_usize + (i + 8) * 4) as *const f32);
        let a3 = vld1q_f32((a_usize + (i + 12) * 4) as *const f32);

        let b0 = vld1q_f32((b_usize + i * 4) as *const f32);
        let b1 = vld1q_f32((b_usize + (i + 4) * 4) as *const f32);
        let b2 = vld1q_f32((b_usize + (i + 8) * 4) as *const f32);
        let b3 = vld1q_f32((b_usize + (i + 12) * 4) as *const f32);

        let r0 = vmulq_f32(a0, b0);
        let r1 = vmulq_f32(a1, b1);
        let r2 = vmulq_f32(a2, b2);
        let r3 = vmulq_f32(a3, b3);

        vst1q_f32((out_usize + i * 4) as *mut f32, r0);
        vst1q_f32((out_usize + (i + 4) * 4) as *mut f32, r1);
        vst1q_f32((out_usize + (i + 8) * 4) as *mut f32, r2);
        vst1q_f32((out_usize + (i + 12) * 4) as *mut f32, r3);

        i += 16;
    }
    while i + 4 <= end {
        let a_vec = vld1q_f32((a_usize + i * 4) as *const f32);
        let b_vec = vld1q_f32((b_usize + i * 4) as *const f32);
        let result = vmulq_f32(a_vec, b_vec);
        vst1q_f32((out_usize + i * 4) as *mut f32, result);
        i += 4;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) * *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn mul_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a_vec = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm512_loadu_ps((b_usize + i * 4) as *const f32);
        let result = _mm512_mul_ps(a_vec, b_vec);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, _mm256_mul_ps(a_vec, b_vec));
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) * *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn mul_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    for i in start..end {
        unsafe {
            *((out_usize + i * 4) as *mut f32) =
                *((a_usize + i * 4) as *const f32) * *((b_usize + i * 4) as *const f32);
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn relu_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let zero = _mm256_set1_ps(0.0f32);
    let mut i = start;
    while i + 16 <= end {
        let v0 = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let v1 = _mm256_loadu_ps((a_usize + (i + 8) * 4) as *const f32);
        let r0 = _mm256_max_ps(v0, zero);
        let r1 = _mm256_max_ps(v1, zero);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, r0);
        _mm256_storeu_ps((out_usize + (i + 8) * 4) as *mut f32, r1);
        i += 16;
    }
    while i + 8 <= end {
        let v = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = _mm256_max_ps(v, zero);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let val = *((a_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = val.max(0.0);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn relu_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let zero = vdupq_n_f32(0.0);
    let mut i = start;
    while i + 16 <= end {
        let v0 = vld1q_f32((a_usize + i * 4) as *const f32);
        let v1 = vld1q_f32((a_usize + (i + 4) * 4) as *const f32);
        let v2 = vld1q_f32((a_usize + (i + 8) * 4) as *const f32);
        let v3 = vld1q_f32((a_usize + (i + 12) * 4) as *const f32);

        let r0 = vmaxq_f32(v0, zero);
        let r1 = vmaxq_f32(v1, zero);
        let r2 = vmaxq_f32(v2, zero);
        let r3 = vmaxq_f32(v3, zero);

        vst1q_f32((out_usize + i * 4) as *mut f32, r0);
        vst1q_f32((out_usize + (i + 4) * 4) as *mut f32, r1);
        vst1q_f32((out_usize + (i + 8) * 4) as *mut f32, r2);
        vst1q_f32((out_usize + (i + 12) * 4) as *mut f32, r3);

        i += 16;
    }
    while i + 4 <= end {
        let v = vld1q_f32((a_usize + i * 4) as *const f32);
        let result = vmaxq_f32(v, zero);
        vst1q_f32((out_usize + i * 4) as *mut f32, result);
        i += 4;
    }
    while i < end {
        let val = *((a_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = val.max(0.0);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn relu_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let zero = _mm512_set1_ps(0.0f32);
    let mut i = start;
    while i + 16 <= end {
        let v = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let result = _mm512_max_ps(v, zero);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let v = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = _mm256_max_ps(v, _mm256_set1_ps(0.0f32));
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let val = *((a_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = val.max(0.0);
        i += 1;
    }
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn relu_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    for i in start..end {
        unsafe {
            let val = *((a_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = val.max(0.0);
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn div_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let a1 = _mm256_loadu_ps((a_usize + (i + 8) * 4) as *const f32);
        let b0 = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let b1 = _mm256_loadu_ps((b_usize + (i + 8) * 4) as *const f32);
        let r0 = _mm256_div_ps(a0, b0);
        let r1 = _mm256_div_ps(a1, b1);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, r0);
        _mm256_storeu_ps((out_usize + (i + 8) * 4) as *mut f32, r1);
        i += 16;
    }
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let result = _mm256_div_ps(a_vec, b_vec);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) / *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn div_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = vld1q_f32((a_usize + i * 4) as *const f32);
        let a1 = vld1q_f32((a_usize + (i + 4) * 4) as *const f32);
        let a2 = vld1q_f32((a_usize + (i + 8) * 4) as *const f32);
        let a3 = vld1q_f32((a_usize + (i + 12) * 4) as *const f32);

        let b0 = vld1q_f32((b_usize + i * 4) as *const f32);
        let b1 = vld1q_f32((b_usize + (i + 4) * 4) as *const f32);
        let b2 = vld1q_f32((b_usize + (i + 8) * 4) as *const f32);
        let b3 = vld1q_f32((b_usize + (i + 12) * 4) as *const f32);

        let r0 = vdivq_f32(a0, b0);
        let r1 = vdivq_f32(a1, b1);
        let r2 = vdivq_f32(a2, b2);
        let r3 = vdivq_f32(a3, b3);

        vst1q_f32((out_usize + i * 4) as *mut f32, r0);
        vst1q_f32((out_usize + (i + 4) * 4) as *mut f32, r1);
        vst1q_f32((out_usize + (i + 8) * 4) as *mut f32, r2);
        vst1q_f32((out_usize + (i + 12) * 4) as *mut f32, r3);

        i += 16;
    }
    while i + 4 <= end {
        let a_vec = vld1q_f32((a_usize + i * 4) as *const f32);
        let b_vec = vld1q_f32((b_usize + i * 4) as *const f32);
        let result = vdivq_f32(a_vec, b_vec);
        vst1q_f32((out_usize + i * 4) as *mut f32, result);
        i += 4;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) / *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn div_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a_vec = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm512_loadu_ps((b_usize + i * 4) as *const f32);
        let result = _mm512_div_ps(a_vec, b_vec);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, _mm256_div_ps(a_vec, b_vec));
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) / *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn div_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    for i in start..end {
        unsafe {
            *((out_usize + i * 4) as *mut f32) =
                *((a_usize + i * 4) as *const f32) / *((b_usize + i * 4) as *const f32);
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn neg_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let a1 = _mm256_loadu_ps((a_usize + (i + 8) * 4) as *const f32);
        let r0 = _mm256_xor_ps(a0, _mm256_set1_ps(-0.0f32));
        let r1 = _mm256_xor_ps(a1, _mm256_set1_ps(-0.0f32));
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, r0);
        _mm256_storeu_ps((out_usize + (i + 8) * 4) as *mut f32, r1);
        i += 16;
    }
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = _mm256_xor_ps(a_vec, _mm256_set1_ps(-0.0f32));
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = -*((a_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn neg_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = vld1q_f32((a_usize + i * 4) as *const f32);
        let a1 = vld1q_f32((a_usize + (i + 4) * 4) as *const f32);
        let a2 = vld1q_f32((a_usize + (i + 8) * 4) as *const f32);
        let a3 = vld1q_f32((a_usize + (i + 12) * 4) as *const f32);

        // vnegq_f32 flips the sign bit directly
        let r0 = vnegq_f32(a0);
        let r1 = vnegq_f32(a1);
        let r2 = vnegq_f32(a2);
        let r3 = vnegq_f32(a3);

        vst1q_f32((out_usize + i * 4) as *mut f32, r0);
        vst1q_f32((out_usize + (i + 4) * 4) as *mut f32, r1);
        vst1q_f32((out_usize + (i + 8) * 4) as *mut f32, r2);
        vst1q_f32((out_usize + (i + 12) * 4) as *mut f32, r3);

        i += 16;
    }
    while i + 4 <= end {
        let a_vec = vld1q_f32((a_usize + i * 4) as *const f32);
        let result = vnegq_f32(a_vec);
        vst1q_f32((out_usize + i * 4) as *mut f32, result);
        i += 4;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = -*((a_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn neg_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a_vec = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let result = _mm512_xor_ps(a_vec, _mm512_set1_ps(-0.0f32));
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = _mm256_xor_ps(a_vec, _mm256_set1_ps(-0.0f32));
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = -*((a_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn neg_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    for i in start..end {
        unsafe {
            *((out_usize + i * 4) as *mut f32) = -*((a_usize + i * 4) as *const f32);
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn abs_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let sign_bit = _mm256_set1_ps(-0.0f32);
    let mut i = start;
    while i + 16 <= end {
        let a0 = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let a1 = _mm256_loadu_ps((a_usize + (i + 8) * 4) as *const f32);
        let r0 = _mm256_andnot_ps(sign_bit, a0);
        let r1 = _mm256_andnot_ps(sign_bit, a1);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, r0);
        _mm256_storeu_ps((out_usize + (i + 8) * 4) as *mut f32, r1);
        i += 16;
    }
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = _mm256_andnot_ps(sign_bit, a_vec);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).abs();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
pub unsafe fn abs_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = vld1q_f32((a_usize + i * 4) as *const f32);
        let a1 = vld1q_f32((a_usize + (i + 4) * 4) as *const f32);
        let a2 = vld1q_f32((a_usize + (i + 8) * 4) as *const f32);
        let a3 = vld1q_f32((a_usize + (i + 12) * 4) as *const f32);

        let r0 = vabsq_f32(a0);
        let r1 = vabsq_f32(a1);
        let r2 = vabsq_f32(a2);
        let r3 = vabsq_f32(a3);

        vst1q_f32((out_usize + i * 4) as *mut f32, r0);
        vst1q_f32((out_usize + (i + 4) * 4) as *mut f32, r1);
        vst1q_f32((out_usize + (i + 8) * 4) as *mut f32, r2);
        vst1q_f32((out_usize + (i + 12) * 4) as *mut f32, r3);

        i += 16;
    }
    while i + 4 <= end {
        let a_vec = vld1q_f32((a_usize + i * 4) as *const f32);
        let result = vabsq_f32(a_vec);
        vst1q_f32((out_usize + i * 4) as *mut f32, result);
        i += 4;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).abs();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn abs_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let sign_bit = _mm512_set1_ps(-0.0f32);
    let mut i = start;
    while i + 16 <= end {
        let a_vec = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let result = _mm512_andnot_ps(sign_bit, a_vec);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let sign_bit_256 = _mm256_set1_ps(-0.0f32);
        let result = _mm256_andnot_ps(sign_bit_256, a_vec);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).abs();
        i += 1;
    }
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn abs_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    for i in start..end {
        unsafe {
            *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).abs();
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn sub_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let a1 = _mm256_loadu_ps((a_usize + (i + 8) * 4) as *const f32);
        let b0 = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let b1 = _mm256_loadu_ps((b_usize + (i + 8) * 4) as *const f32);
        let r0 = _mm256_sub_ps(a0, b0);
        let r1 = _mm256_sub_ps(a1, b1);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, r0);
        _mm256_storeu_ps((out_usize + (i + 8) * 4) as *mut f32, r1);
        i += 16;
    }
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let result = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) - *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn sub_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = vld1q_f32((a_usize + i * 4) as *const f32);
        let a1 = vld1q_f32((a_usize + (i + 4) * 4) as *const f32);
        let a2 = vld1q_f32((a_usize + (i + 8) * 4) as *const f32);
        let a3 = vld1q_f32((a_usize + (i + 12) * 4) as *const f32);

        let b0 = vld1q_f32((b_usize + i * 4) as *const f32);
        let b1 = vld1q_f32((b_usize + (i + 4) * 4) as *const f32);
        let b2 = vld1q_f32((b_usize + (i + 8) * 4) as *const f32);
        let b3 = vld1q_f32((b_usize + (i + 12) * 4) as *const f32);

        let r0 = vsubq_f32(a0, b0);
        let r1 = vsubq_f32(a1, b1);
        let r2 = vsubq_f32(a2, b2);
        let r3 = vsubq_f32(a3, b3);

        vst1q_f32((out_usize + i * 4) as *mut f32, r0);
        vst1q_f32((out_usize + (i + 4) * 4) as *mut f32, r1);
        vst1q_f32((out_usize + (i + 8) * 4) as *mut f32, r2);
        vst1q_f32((out_usize + (i + 12) * 4) as *mut f32, r3);

        i += 16;
    }
    while i + 4 <= end {
        let a_vec = vld1q_f32((a_usize + i * 4) as *const f32);
        let b_vec = vld1q_f32((b_usize + i * 4) as *const f32);
        let result = vsubq_f32(a_vec, b_vec);
        vst1q_f32((out_usize + i * 4) as *mut f32, result);
        i += 4;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) - *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn sub_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a_vec = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm512_loadu_ps((b_usize + i * 4) as *const f32);
        let result = _mm512_sub_ps(a_vec, b_vec);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, _mm256_sub_ps(a_vec, b_vec));
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) - *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn sub_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    for i in start..end {
        unsafe {
            *((out_usize + i * 4) as *mut f32) =
                *((a_usize + i * 4) as *const f32) - *((b_usize + i * 4) as *const f32);
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn fused_add_relu_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let zero = _mm256_set1_ps(0.0f32);
    let mut i = start;
    while i + 16 <= end {
        let a0 = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let a1 = _mm256_loadu_ps((a_usize + (i + 8) * 4) as *const f32);
        let b0 = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let b1 = _mm256_loadu_ps((b_usize + (i + 8) * 4) as *const f32);
        let sum0 = _mm256_add_ps(a0, b0);
        let sum1 = _mm256_add_ps(a1, b1);
        let relu0 = _mm256_max_ps(sum0, zero);
        let relu1 = _mm256_max_ps(sum1, zero);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, relu0);
        _mm256_storeu_ps((out_usize + (i + 8) * 4) as *mut f32, relu1);
        i += 16;
    }
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let sum = _mm256_add_ps(a_vec, b_vec);
        let relu = _mm256_max_ps(sum, zero);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, relu);
        i += 8;
    }
    while i < end {
        let val = *((a_usize + i * 4) as *const f32) + *((b_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = val.max(0.0);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_add_relu_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let zero = _mm512_set1_ps(0.0f32);
    let mut i = start;
    while i + 16 <= end {
        let a_vec = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm512_loadu_ps((b_usize + i * 4) as *const f32);
        let sum = _mm512_add_ps(a_vec, b_vec);
        let relu = _mm512_max_ps(sum, zero);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, relu);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let sum = _mm256_add_ps(a_vec, b_vec);
        let relu = _mm256_max_ps(sum, _mm256_set1_ps(0.0f32));
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, relu);
        i += 8;
    }
    while i < end {
        let val = *((a_usize + i * 4) as *const f32) + *((b_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = val.max(0.0);
        i += 1;
    }
}

#[cfg(feature = "parallel")]
pub fn fused_add_relu_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    for i in start..end {
        unsafe {
            let val = *((a_usize + i * 4) as *const f32) + *((b_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = val.max(0.0);
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn fused_add_relu_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    let zero = vdupq_n_f32(0.0);
    while i + 16 <= end {
        let a0 = vld1q_f32((a_usize + i * 4) as *const f32);
        let a1 = vld1q_f32((a_usize + (i + 4) * 4) as *const f32);
        let a2 = vld1q_f32((a_usize + (i + 8) * 4) as *const f32);
        let a3 = vld1q_f32((a_usize + (i + 12) * 4) as *const f32);

        let b0 = vld1q_f32((b_usize + i * 4) as *const f32);
        let b1 = vld1q_f32((b_usize + (i + 4) * 4) as *const f32);
        let b2 = vld1q_f32((b_usize + (i + 8) * 4) as *const f32);
        let b3 = vld1q_f32((b_usize + (i + 12) * 4) as *const f32);

        let sum0 = vaddq_f32(a0, b0);
        let sum1 = vaddq_f32(a1, b1);
        let sum2 = vaddq_f32(a2, b2);
        let sum3 = vaddq_f32(a3, b3);

        let relu0 = vmaxq_f32(sum0, zero);
        let relu1 = vmaxq_f32(sum1, zero);
        let relu2 = vmaxq_f32(sum2, zero);
        let relu3 = vmaxq_f32(sum3, zero);

        vst1q_f32((out_usize + i * 4) as *mut f32, relu0);
        vst1q_f32((out_usize + (i + 4) * 4) as *mut f32, relu1);
        vst1q_f32((out_usize + (i + 8) * 4) as *mut f32, relu2);
        vst1q_f32((out_usize + (i + 12) * 4) as *mut f32, relu3);

        i += 16;
    }
    while i + 4 <= end {
        let a_vec = vld1q_f32((a_usize + i * 4) as *const f32);
        let b_vec = vld1q_f32((b_usize + i * 4) as *const f32);
        let sum = vaddq_f32(a_vec, b_vec);
        let result = vmaxq_f32(sum, zero);
        vst1q_f32((out_usize + i * 4) as *mut f32, result);
        i += 4;
    }
    while i < end {
        let val = *((a_usize + i * 4) as *const f32) + *((b_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = val.max(0.0);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_mul_add_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    c_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let a1 = _mm256_loadu_ps((a_usize + (i + 8) * 4) as *const f32);
        let b0 = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let b1 = _mm256_loadu_ps((b_usize + (i + 8) * 4) as *const f32);
        let c0 = _mm256_loadu_ps((c_usize + i * 4) as *const f32);
        let c1 = _mm256_loadu_ps((c_usize + (i + 8) * 4) as *const f32);
        // a * b + c using FMA
        let r0 = _mm256_fmadd_ps(a0, b0, c0);
        let r1 = _mm256_fmadd_ps(a1, b1, c1);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, r0);
        _mm256_storeu_ps((out_usize + (i + 8) * 4) as *mut f32, r1);
        i += 16;
    }
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let c_vec = _mm256_loadu_ps((c_usize + i * 4) as *const f32);
        let result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let a_val = *((a_usize + i * 4) as *const f32);
        let b_val = *((b_usize + i * 4) as *const f32);
        let c_val = *((c_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = a_val * b_val + c_val;
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_mul_add_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    c_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a_vec = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm512_loadu_ps((b_usize + i * 4) as *const f32);
        let c_vec = _mm512_loadu_ps((c_usize + i * 4) as *const f32);
        // a * b + c using FMA
        let result = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let c_vec = _mm256_loadu_ps((c_usize + i * 4) as *const f32);
        let result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let a_val = *((a_usize + i * 4) as *const f32);
        let b_val = *((b_usize + i * 4) as *const f32);
        let c_val = *((c_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = a_val * b_val + c_val;
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn fused_mul_add_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    c_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = vld1q_f32((a_usize + i * 4) as *const f32);
        let a1 = vld1q_f32((a_usize + (i + 4) * 4) as *const f32);
        let a2 = vld1q_f32((a_usize + (i + 8) * 4) as *const f32);
        let a3 = vld1q_f32((a_usize + (i + 12) * 4) as *const f32);

        let b0 = vld1q_f32((b_usize + i * 4) as *const f32);
        let b1 = vld1q_f32((b_usize + (i + 4) * 4) as *const f32);
        let b2 = vld1q_f32((b_usize + (i + 8) * 4) as *const f32);
        let b3 = vld1q_f32((b_usize + (i + 12) * 4) as *const f32);

        let c0 = vld1q_f32((c_usize + i * 4) as *const f32);
        let c1 = vld1q_f32((c_usize + (i + 4) * 4) as *const f32);
        let c2 = vld1q_f32((c_usize + (i + 8) * 4) as *const f32);
        let c3 = vld1q_f32((c_usize + (i + 12) * 4) as *const f32);

        // vfmaq_f32 computes: c + a * b
        let r0 = vfmaq_f32(c0, a0, b0);
        let r1 = vfmaq_f32(c1, a1, b1);
        let r2 = vfmaq_f32(c2, a2, b2);
        let r3 = vfmaq_f32(c3, a3, b3);

        vst1q_f32((out_usize + i * 4) as *mut f32, r0);
        vst1q_f32((out_usize + (i + 4) * 4) as *mut f32, r1);
        vst1q_f32((out_usize + (i + 8) * 4) as *mut f32, r2);
        vst1q_f32((out_usize + (i + 12) * 4) as *mut f32, r3);

        i += 16;
    }
    while i + 4 <= end {
        let a_vec = vld1q_f32((a_usize + i * 4) as *const f32);
        let b_vec = vld1q_f32((b_usize + i * 4) as *const f32);
        let c_vec = vld1q_f32((c_usize + i * 4) as *const f32);
        let result = vfmaq_f32(c_vec, a_vec, b_vec);
        vst1q_f32((out_usize + i * 4) as *mut f32, result);
        i += 4;
    }
    while i < end {
        let a_val = *((a_usize + i * 4) as *const f32);
        let b_val = *((b_usize + i * 4) as *const f32);
        let c_val = *((c_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = a_val * b_val + c_val;
        i += 1;
    }
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn fused_mul_add_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    c_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    for i in start..end {
        unsafe {
            let a_val = *((a_usize + i * 4) as *const f32);
            let b_val = *((b_usize + i * 4) as *const f32);
            let c_val = *((c_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = a_val * b_val + c_val;
        }
    }
}

#[cfg(all(feature = "parallel", feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sigmoid_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let one = _mm256_set1_ps(1.0f32);

    let mut i = start;
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        // sigmoid(x) = 1 / (1 + exp(-x))
        let neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f32));
        let exp_neg_x = fast_exp_avx2(neg_x);
        let denom = _mm256_add_ps(one, exp_neg_x);
        // Use Newton-Raphson reciprocal approximation for better accuracy
        let rcp = _mm256_rcp_ps(denom);
        let result = _mm256_mul_ps(
            _mm256_mul_ps(rcp, _mm256_fnmadd_ps(rcp, denom, _mm256_set1_ps(2.0f32))),
            one,
        );
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }

    while i < end {
        let x = *((a_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = 1.0 / (1.0 + (-x).exp());
        i += 1;
    }
}

#[cfg(all(feature = "parallel", feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn sigmoid_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let one = _mm512_set1_ps(1.0f32);

    let mut i = start;
    while i + 16 <= end {
        let x = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        // sigmoid(x) = 1 / (1 + exp(-x))
        let neg_x = _mm512_xor_ps(x, _mm512_set1_ps(-0.0f32));
        let exp_neg_x = fast_exp_avx512(neg_x);
        let denom = _mm512_add_ps(one, exp_neg_x);
        let result = _mm512_div_ps(one, denom);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f32));
        let exp_neg_x = fast_exp_avx2(neg_x);
        let denom = _mm256_add_ps(_mm256_set1_ps(1.0f32), exp_neg_x);
        let result = _mm256_div_ps(_mm256_set1_ps(1.0f32), denom);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let x = *((a_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = 1.0 / (1.0 + (-x).exp());
        i += 1;
    }
}

#[cfg(feature = "parallel")]
pub fn sigmoid_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    for i in start..end {
        unsafe {
            let x = *((a_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = 1.0 / (1.0 + (-x).exp());
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn sigmoid_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let one = f32x4::new([1.0; 4]);

    let a_ptr = a_usize as *const f32;
    let out_ptr = out_usize as *mut f32;

    let a_ptr_arr = a_ptr as *const [f32; 4];
    let out_ptr_arr = out_ptr as *mut [f32; 4];

    let mut i = start;
    while i + 4 <= end {
        let x = f32x4::from(unsafe { *a_ptr_arr.add(i / 4) });
        let neg_x = -x;
        let exp_neg_x = neg_x.exp();
        let result = one / (one + exp_neg_x);
        unsafe {
            *out_ptr_arr.add(i / 4) = result.into();
        }
        i += 4;
    }
    while i < end {
        let x = *a_ptr.add(i);
        *out_ptr.add(i) = 1.0 / (1.0 + (-x).exp());
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn tanh_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let clamp_lo = _mm256_set1_ps(-10.0f32);
    let clamp_hi = _mm256_set1_ps(10.0f32);
    let one = _mm256_set1_ps(1.0f32);
    let two = _mm256_set1_ps(2.0f32);

    let mut i = start;
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        // Clamp values to prevent overflow in exp
        let x_clamped = _mm256_min_ps(_mm256_max_ps(x, clamp_lo), clamp_hi);
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let two_x = _mm256_mul_ps(two, x_clamped);
        let exp_2x = fast_exp_avx2(two_x);
        let result = _mm256_div_ps(_mm256_sub_ps(exp_2x, one), _mm256_add_ps(exp_2x, one));
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }

    while i < end {
        let x = *((a_usize + i * 4) as *const f32);
        let exp_2x = (2.0 * x).exp();
        *((out_usize + i * 4) as *mut f32) = (exp_2x - 1.0) / (exp_2x + 1.0);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn tanh_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let clamp_lo = _mm512_set1_ps(-10.0f32);
    let clamp_hi = _mm512_set1_ps(10.0f32);
    let one = _mm512_set1_ps(1.0f32);
    let two = _mm512_set1_ps(2.0f32);

    let mut i = start;
    while i + 16 <= end {
        let x = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        // Clamp values to prevent overflow in exp
        let x_clamped = _mm512_min_ps(_mm512_max_ps(x, clamp_lo), clamp_hi);
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let two_x = _mm512_mul_ps(two, x_clamped);
        let exp_2x = fast_exp_avx512(two_x);
        let result = _mm512_div_ps(_mm512_sub_ps(exp_2x, one), _mm512_add_ps(exp_2x, one));
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let clamp_lo_256 = _mm256_set1_ps(-10.0f32);
        let clamp_hi_256 = _mm256_set1_ps(10.0f32);
        let x_clamped = _mm256_min_ps(_mm256_max_ps(x, clamp_lo_256), clamp_hi_256);
        let two_x = _mm256_mul_ps(_mm256_set1_ps(2.0f32), x_clamped);
        let exp_2x = fast_exp_avx2(two_x);
        let result = _mm256_div_ps(
            _mm256_sub_ps(exp_2x, _mm256_set1_ps(1.0f32)),
            _mm256_add_ps(exp_2x, _mm256_set1_ps(1.0f32)),
        );
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let x = *((a_usize + i * 4) as *const f32);
        let exp_2x = (2.0 * x).exp();
        *((out_usize + i * 4) as *mut f32) = (exp_2x - 1.0) / (exp_2x + 1.0);
        i += 1;
    }
}

#[cfg(feature = "parallel")]
pub fn tanh_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    for i in start..end {
        unsafe {
            let x = *((a_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = x.tanh();
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn tanh_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let two = f32x4::new([2.0; 4]);
    let one = f32x4::new([1.0; 4]);
    let clamp_lo = f32x4::new([-10.0; 4]);
    let clamp_hi = f32x4::new([10.0; 4]);

    let a_ptr = a_usize as *const f32;
    let out_ptr = out_usize as *mut f32;
    let a_ptr_arr = a_ptr as *const [f32; 4];
    let out_ptr_arr = out_ptr as *mut [f32; 4];

    let mut i = start;
    while i + 4 <= end {
        let x = f32x4::from(unsafe { *a_ptr_arr.add(i / 4) });

        // Clamp values to prevent overflow in exp
        let x_clamped = x.max(clamp_lo).min(clamp_hi);

        // Compute tanh using formula: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let exp_2x = (two * x_clamped).exp();
        let result = (exp_2x - one) / (exp_2x + one);

        unsafe {
            *out_ptr_arr.add(i / 4) = result.into();
        }
        i += 4;
    }
    while i < end {
        let x = *a_ptr.add(i);
        let exp_2x = (2.0 * x).exp();
        *out_ptr.add(i) = (exp_2x - 1.0) / (exp_2x + 1.0);
        i += 1;
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
pub fn relu_simd(input: &[f32], output: &mut [f32]) {
    let zero = f32x8::ZERO;
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x8::from(*in_chunk);
        let result = v.max(zero);
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.max(0.0);
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
pub fn exp_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x8::from(*in_chunk);
        let result = v.exp();
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.exp();
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
pub fn log_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x8::from(*in_chunk);
        let result = v.ln();
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.ln();
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
pub fn sqrt_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x8::from(*in_chunk);
        let result = v.sqrt();
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.sqrt();
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
pub fn gelu_simd(input: &[f32], output: &mut [f32]) {
    let sqrt_2_over_pi = f32x8::new([0.797_884_6; 8]);
    let coeff = f32x8::new([0.044715; 8]);
    let half = f32x8::new([0.5; 8]);
    let one = f32x8::new([1.0; 8]);
    let two = f32x8::new([2.0; 8]);

    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let x = f32x8::from(*in_chunk);
        let x3 = x * x * x;
        let y = sqrt_2_over_pi * (x + coeff * x3);
        // Compute tanh(y) stably to avoid inf/inf -> NaN
        // tanh(z) = sign(z) * (1 - 2 / (exp(2|z|) + 1))
        let abs_y = y.abs();
        let exp_2abs_y = (abs_y + abs_y).exp(); // 2.0 * abs_y
        let tanh_val = one - (two / (exp_2abs_y + one));
        // Safe sign calculation avoiding div by zero (y / abs_y with epsilon)
        let sign = y / (abs_y + 1e-9);
        let t = tanh_val * sign;
        let result = half * x * (one + t);
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        let x = *in_val;
        let x3 = x * x * x;
        let t = (0.797_884_6 * (x + 0.044715 * x3)).tanh();
        *out_val = 0.5 * x * (1.0 + t);
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
pub fn silu_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();
    let one = f32x8::new([1.0; 8]);

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let x = f32x8::from(*in_chunk);
        let exp_neg_x = (-x).exp();
        let result = x / (one + exp_neg_x);
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        let x = *in_val;
        *out_val = x / (1.0 + (-x).exp());
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub fn exp_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<4>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x4::from(*in_chunk);
        let result = v.exp();
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.exp();
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub fn log_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<4>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x4::from(*in_chunk);
        let result = v.ln();
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.ln();
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub fn sqrt_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<4>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x4::from(*in_chunk);
        let result = v.sqrt();
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.sqrt();
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub fn gelu_simd(input: &[f32], output: &mut [f32]) {
    let sqrt_2_over_pi = f32x4::new([0.797_884_6, 0.797_884_6, 0.797_884_6, 0.797_884_6]);
    let coeff = f32x4::new([0.044715f32, 0.044715f32, 0.044715f32, 0.044715f32]);
    let half = f32x4::new([0.5f32, 0.5f32, 0.5f32, 0.5f32]);
    let one = f32x4::new([1.0f32, 1.0f32, 1.0f32, 1.0f32]);

    let (chunks, remainder) = input.as_chunks::<4>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let x = f32x4::from(*in_chunk);
        let x3 = x * x * x;
        let y = sqrt_2_over_pi * (x + coeff * x3);
        // Using exp to compute tanh
        let exp_y = y.exp();
        let t = (exp_y - one) / (exp_y + one);
        let result = half * x * (one + t);
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        let x = *in_val;
        let x3 = x * x * x;
        let t = (0.797_884_6 * (x + 0.044715 * x3)).tanh();
        *out_val = 0.5 * x * (1.0 + t);
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub fn tanh_simd(input: &[f32], output: &mut [f32]) {
    let two = f32x4::new([2.0; 4]);
    let one = f32x4::new([1.0; 4]);

    let (chunks, remainder) = input.as_chunks::<4>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x4::from(*in_chunk);
        let abs_v = v.abs();
        let exp_2x = (two * abs_v).exp();
        let result = (exp_2x - one) / (exp_2x + one);
        let sign = v.sign_bit();
        let result = result.blend(-result, sign);
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        let x = *in_val;
        let abs_x = x.abs();
        let exp_2x = (2.0 * abs_x).exp();
        let mut result = (exp_2x - 1.0) / (exp_2x + 1.0);
        if x < 0.0 {
            result = -result;
        }
        *out_val = result;
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
pub fn tanh_simd(input: &[f32], output: &mut [f32]) {
    // tanh(x) = (e^2x - 1) / (e^2x + 1)
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x8::from(*in_chunk);
        let v_clamped = v.max(f32x8::new([-10.0; 8])).min(f32x8::new([10.0; 8]));
        let exp_2x = (f32x8::new([2.0; 8]) * v_clamped).exp();
        let result = (exp_2x - f32x8::new([1.0; 8])) / (exp_2x + f32x8::new([1.0; 8]));
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        let x_clamped = in_val.clamp(-10.0, 10.0);
        let exp_2x = (2.0 * x_clamped).exp();
        *out_val = (exp_2x - 1.0) / (exp_2x + 1.0);
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub fn sigmoid_simd(input: &[f32], output: &mut [f32]) {
    let one = f32x4::new([1.0; 4]);

    let (chunks, remainder) = input.as_chunks::<4>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let x = f32x4::from(*in_chunk);
        let neg_x = -x;
        let exp_neg_x = neg_x.exp();
        let result = one / (one + exp_neg_x);
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        let x = *in_val;
        *out_val = 1.0 / (1.0 + (-x).exp());
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(dead_code)]
pub fn sigmoid_simd_x86(input: &[f32], output: &mut [f32]) {
    unsafe {
        let (chunks, remainder) = input.as_chunks::<8>();
        let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

        for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
            let x = _mm256_loadu_ps(in_chunk.as_ptr());
            // Compute sigmoid using 1/(1+exp(-x))
            let x_f32 = std::mem::transmute::<__m256, [f32; 8]>(x);
            let mut results = [0.0f32; 8];
            for j in 0..8 {
                let val = x_f32[j];
                if val > 9.0 {
                    results[j] = 1.0;
                } else if val < -9.0 {
                    results[j] = 0.0;
                } else {
                    results[j] = 1.0 / (1.0 + (-val).exp());
                }
            }
            let result = std::mem::transmute::<[f32; 8], __m256>(results);
            _mm256_storeu_ps(out_chunk.as_mut_ptr(), result);
        }

        for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
            let val = *in_val;
            if val > 9.0 {
                *out_val = 1.0;
            } else if val < -9.0 {
                *out_val = 0.0;
            } else {
                *out_val = 1.0 / (1.0 + (-val).exp());
            }
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = fast_exp_avx2(x);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).exp();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn exp_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let x = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let result = fast_exp_avx512(x);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = fast_exp_avx2(x);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).exp();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn exp_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let a_ptr = a_usize as *const f32;
    let out_ptr = out_usize as *mut f32;
    let a_ptr_arr = a_ptr as *const [f32; 4];
    let out_ptr_arr = out_ptr as *mut [f32; 4];

    let mut i = start;
    while i + 4 <= end {
        let x = f32x4::from(unsafe { *a_ptr_arr.add(i / 4) });
        let result = x.exp();
        unsafe {
            *out_ptr_arr.add(i / 4) = result.into();
        }
        i += 4;
    }
    while i < end {
        *out_ptr.add(i) = (*a_ptr.add(i)).exp();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn log_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = fast_log_avx2(x);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).ln();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn log_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let x = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let result = fast_log_avx512(x);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = fast_log_avx2(x);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).ln();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn log_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let a_ptr = a_usize as *const f32;
    let out_ptr = out_usize as *mut f32;
    let a_ptr_arr = a_ptr as *const [f32; 4];
    let out_ptr_arr = out_ptr as *mut [f32; 4];

    let mut i = start;
    while i + 4 <= end {
        let x = f32x4::from(unsafe { *a_ptr_arr.add(i / 4) });
        let result = x.ln();
        unsafe {
            *out_ptr_arr.add(i / 4) = result.into();
        }
        i += 4;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).ln();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn sqrt_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = _mm256_sqrt_ps(x);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).sqrt();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn sqrt_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let x = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let result = _mm512_sqrt_ps(x);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = _mm256_sqrt_ps(x);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).sqrt();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn sqrt_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let a_ptr = a_usize as *const f32;
    let out_ptr = out_usize as *mut f32;
    let a_ptr_arr = a_ptr as *const [f32; 4];
    let out_ptr_arr = out_ptr as *mut [f32; 4];

    let mut i = start;
    while i + 4 <= end {
        let x = f32x4::from(unsafe { *a_ptr_arr.add(i / 4) });
        let result = x.sqrt();
        unsafe {
            *out_ptr_arr.add(i / 4) = result.into();
        }
        i += 4;
    }
    while i < end {
        *out_ptr.add(i) = (*a_ptr.add(i)).sqrt();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gelu_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let sqrt_2_over_pi = _mm256_set1_ps(0.7978846f32);
    let coeff = _mm256_set1_ps(0.044715f32);
    let half = _mm256_set1_ps(0.5f32);
    let one = _mm256_set1_ps(1.0f32);
    let two = _mm256_set1_ps(2.0f32);

    let mut i = start;
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);

        // Compute x^3
        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x, x2);

        // Compute inner = sqrt_2_over_pi * (x + coeff * x^3)
        let inner = _mm256_fmadd_ps(coeff, x3, x);
        let inner = _mm256_mul_ps(sqrt_2_over_pi, inner);

        // Compute tanh(inner) using: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let exp_2x = fast_exp_avx2(_mm256_mul_ps(two, inner));
        let tanh = _mm256_div_ps(_mm256_sub_ps(exp_2x, one), _mm256_add_ps(exp_2x, one));

        // Compute gelu = 0.5 * x * (1 + tanh)
        let result = _mm256_mul_ps(half, x);
        let result = _mm256_mul_ps(result, _mm256_add_ps(one, tanh));

        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let x = *((a_usize + i * 4) as *const f32);
        let x3 = x * x * x;
        let t = (0.7978846 * (x + 0.044715 * x3)).tanh();
        *((out_usize + i * 4) as *mut f32) = 0.5 * x * (1.0 + t);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn gelu_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let sqrt_2_over_pi = _mm512_set1_ps(0.7978846f32);
    let coeff = _mm512_set1_ps(0.044715f32);
    let half = _mm512_set1_ps(0.5f32);
    let one = _mm512_set1_ps(1.0f32);
    let two = _mm512_set1_ps(2.0f32);

    let mut i = start;
    while i + 16 <= end {
        let x = _mm512_loadu_ps((a_usize + i * 4) as *const f32);

        // Compute x^3
        let x2 = _mm512_mul_ps(x, x);
        let x3 = _mm512_mul_ps(x, x2);

        // Compute inner = sqrt_2_over_pi * (x + coeff * x^3)
        let inner = _mm512_fmadd_ps(coeff, x3, x);
        let inner = _mm512_mul_ps(sqrt_2_over_pi, inner);

        // Compute tanh(inner) using: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let exp_2x = fast_exp_avx512(_mm512_mul_ps(two, inner));
        let tanh = _mm512_div_ps(_mm512_sub_ps(exp_2x, one), _mm512_add_ps(exp_2x, one));

        // Compute gelu = 0.5 * x * (1 + tanh)
        let result = _mm512_mul_ps(half, x);
        let result = _mm512_mul_ps(result, _mm512_add_ps(one, tanh));

        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let sqrt_2_over_pi_256 = _mm256_set1_ps(0.7978846f32);
        let coeff_256 = _mm256_set1_ps(0.044715f32);
        let half_256 = _mm256_set1_ps(0.5f32);
        let one_256 = _mm256_set1_ps(1.0f32);
        let two_256 = _mm256_set1_ps(2.0f32);

        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x, x2);
        let inner = _mm256_fmadd_ps(coeff_256, x3, x);
        let inner = _mm256_mul_ps(sqrt_2_over_pi_256, inner);
        let exp_2x = fast_exp_avx2(_mm256_mul_ps(two_256, inner));
        let tanh = _mm256_div_ps(
            _mm256_sub_ps(exp_2x, one_256),
            _mm256_add_ps(exp_2x, one_256),
        );
        let result = _mm256_mul_ps(half_256, x);
        let result = _mm256_mul_ps(result, _mm256_add_ps(one_256, tanh));

        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let x = *((a_usize + i * 4) as *const f32);
        let x3 = x * x * x;
        let t = (0.7978846 * (x + 0.044715 * x3)).tanh();
        *((out_usize + i * 4) as *mut f32) = 0.5 * x * (1.0 + t);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
pub unsafe fn gelu_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let sqrt_2_over_pi = f32x4::new([0.7978846; 4]);
    let coeff = f32x4::new([0.044715; 4]);
    let half = f32x4::new([0.5; 4]);
    let one = f32x4::new([1.0; 4]);
    let two = f32x4::new([2.0; 4]);

    let a_ptr = a_usize as *const f32;
    let out_ptr = out_usize as *mut f32;
    let a_ptr_arr = a_ptr as *const [f32; 4];
    let out_ptr_arr = out_ptr as *mut [f32; 4];

    let mut i = start;
    while i + 4 <= end {
        let x = f32x4::from(unsafe { *a_ptr_arr.add(i / 4) });

        // Compute x^3
        let x2 = x * x;
        let x3 = x * x2;

        // Compute inner = sqrt_2_over_pi * (x + coeff * x^3)
        let inner = sqrt_2_over_pi * (x + coeff * x3);

        // Compute tanh(inner) using: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let exp_2x = (two * inner).exp();
        let tanh = (exp_2x - one) / (exp_2x + one);

        // Compute gelu = 0.5 * x * (1 + tanh)
        let result = half * x * (one + tanh);

        unsafe {
            *out_ptr_arr.add(i / 4) = result.into();
        }
        i += 4;
    }
    while i < end {
        let x = *a_ptr.add(i);
        let x3 = x * x * x;
        let t = (0.7978846 * (x + 0.044715 * x3)).tanh();
        *out_ptr.add(i) = 0.5 * x * (1.0 + t);
        i += 1;
    }
}

    #[test]
    pub fn test_parallel_matmul_fallback_3d_batched() {
        // Exercise the parallel_matmul fallback path with 3D batched tensors
        // that are both contiguous but below BLAS threshold (so BLAS is skipped).
        let batch: usize = 4;
        let m: usize = 16;
        let k: usize = 16;
        let n: usize = 16;

        // Build A [batch, m, k] and B [batch, k, n] with simple values
        let a_data: Vec<f32> = (0..batch * m * k)
            .map(|i| ((i % 100) as f32) * 0.01)
            .collect();
        let b_data: Vec<f32> = (0..batch * k * n)
            .map(|i| ((i % 100) as f32) * 0.02 - 1.0)
            .collect();

        let a = Tensor::from_vec(a_data.clone(), vec![batch as i64, m as i64, k as i64]);
        let b = Tensor::from_vec(b_data.clone(), vec![batch as i64, k as i64, n as i64]);

        let result = a.matmul(&b);
        let result_data = result.as_f32_slice();

        // Verify against reference scalar matmul for each batch
        for bat in 0..batch {
            let a_batch = &a_data[bat * m * k..(bat + 1) * m * k];
            let b_batch = &b_data[bat * k * n..(bat + 1) * k * n];
            let expected = reference_matmul(a_batch, b_batch, m, k, n);
            let result_batch = &result_data[bat * m * n..(bat + 1) * m * n];

            for (idx, (got, exp)) in result_batch.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-4,
                    "batch={}, idx={}: got={}, expected={}",
                    bat,
                    idx,
                    got,
                    exp
                );
            }
        }
    }

    #[test]
    pub fn test_parallel_matmul_fallback_2d_small() {
        // Exercise the parallel_matmul fallback path with 2D matrices
        // below BLAS threshold
        let m: usize = 8;
        let k: usize = 12;
        let n: usize = 10;

        let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.1).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.1).cos()).collect();

        let a = Tensor::from_vec(a_data.clone(), vec![m as i64, k as i64]);
        let b = Tensor::from_vec(b_data.clone(), vec![k as i64, n as i64]);

        let result = a.matmul(&b);
        let result_data = result.as_f32_slice();

        let expected = reference_matmul(&a_data, &b_data, m, k, n);

        for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "idx={}: got={}, expected={}",
                idx,
                got,
                exp
            );
        }
    }


#[cfg(test)]
/// Reference scalar matmul for a single batch: C = A @ B
/// A is [m, k], B is [k, n], C is [m, n]
pub fn reference_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

#[cfg(test)]
/// Reference scalar matmul with explicit strides
pub fn reference_matmul_strided(
    a: &[f32],
    b: &[f32],
    _batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch_stride: usize,
    a_s0: usize,
    a_s1: usize,
    b_batch_stride: usize,
    b_s0: usize,
    b_s1: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * a_s0 + kk * a_s1] * b[kk * b_s0 + j * b_s1];
            }
            out[i * n + j] = sum;
        }
    }
    out
}
