#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::wildcard_in_or_patterns
)]
#![allow(unused_imports)]

use crate::autograd::{AutogradMeta, Edge, Node};
use crate::dispatcher::{register, DispatchKey, KernelFn};
use crate::iterator::TensorIterator;
use crate::kernels::blas::{matmul_blas, matmul_blas_with_transpose, MIN_BLAS_SIZE};
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use std::sync::Arc;
use std::sync::OnceLock;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use std::arch::aarch64::*;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use wide::f32x4;

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
use wide::f32x8;

// Memory-bound elementwise ops: 64KB working set
const CHUNK_MEMBOUND: usize = 1024 * 16; // 16K f32 = 64KB (L1 cache fit)

// Compute-bound transcendental ops: smaller for better load balancing
const CHUNK_TRANSCENDENTAL: usize = 1024 * 4; // 4K f32 = 16KB

// Matrix ops: larger chunks for better BLAS-level locality
// SIMD-only threshold (non-parallel)
// This constant is used to determine when to use SIMD operations
#[allow(dead_code)]
const SIMD_THRESHOLD: usize = 256;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
fn relu_simd(input: &[f32], output: &mut [f32]) {
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

// Runtime SIMD level detection
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[derive(Clone, Copy, PartialEq)]
enum SimdLevel {
    Scalar,
    Avx2,
    Avx512,
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
    }
    SimdLevel::Scalar
}

// Fast exp approximation using Cephes-style algorithm
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fast_exp_avx2(x: __m256) -> __m256 {
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

// Fast exp approximation for AVX512
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn fast_exp_avx512(x: __m512) -> __m512 {
    // Process 16 floats as 2x 8 floats with AVX2
    let x_lo = _mm512_castps512_ps256(x);
    let x_hi = _mm512_extractf32x8_ps(x, 1);

    let exp_lo = fast_exp_avx2(x_lo);
    let exp_hi = fast_exp_avx2(x_hi);

    let result_lo = _mm512_castps256_ps512(exp_lo);
    let result_hi = exp_hi;

    _mm512_insertf32x8(result_lo, result_hi, 1)
}

// Fast log approximation using integer exponent extraction
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fast_log_avx2(x: __m256) -> __m256 {
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

// Fast log approximation for AVX512
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn fast_log_avx512(x: __m512) -> __m512 {
    // Use fast_log_avx2 on each 256-bit half
    let x_lo = _mm512_castps512_ps256(x);
    let x_hi = _mm512_extractf32x8_ps(x, 1);

    let log_lo = fast_log_avx2(x_lo);
    let log_hi = fast_log_avx2(x_hi);

    let result_lo = _mm512_castps256_ps512(log_lo);
    let result_hi = log_hi;

    _mm512_insertf32x8(result_lo, result_hi, 1)
}

// Parallel SIMD kernels - AVX2 version
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_parallel_avx2(
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
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let result = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) + *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

// Parallel SIMD kernels - NEON version
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn add_parallel_neon(
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

// Parallel SIMD kernels - AVX512 version
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn add_parallel_avx512(
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
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, _mm256_add_ps(a_vec, b_vec));
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) =
            *((a_usize + i * 4) as *const f32) + *((b_usize + i * 4) as *const f32);
        i += 1;
    }
}

// Parallel scalar fallback
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn add_parallel_scalar(
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

// Mul parallel AVX2 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mul_parallel_avx2(
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

// Mul parallel NEON kernel
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn mul_parallel_neon(
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

// Mul parallel AVX512 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn mul_parallel_avx512(
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

// Parallel scalar fallback for mul
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn mul_parallel_scalar(
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

// Relu parallel AVX2 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_parallel_avx2(
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

// Relu parallel NEON kernel
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn relu_parallel_neon(
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

// Relu parallel AVX512 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn relu_parallel_avx512(
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

// Parallel scalar fallback for relu
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn relu_parallel_scalar(
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

// Div parallel AVX2 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn div_parallel_avx2(
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

// Div parallel NEON kernel
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn div_parallel_neon(
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

// Div parallel AVX512 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn div_parallel_avx512(
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

// Parallel scalar fallback for div
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn div_parallel_scalar(
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

// Neg parallel AVX2 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn neg_parallel_avx2(
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

// Neg parallel NEON kernel
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn neg_parallel_neon(
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

// Neg parallel AVX512 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn neg_parallel_avx512(
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

// Parallel scalar fallback for neg
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn neg_parallel_scalar(
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

// Abs parallel AVX2 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn abs_parallel_avx2(
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

// Abs parallel NEON kernel
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn abs_parallel_neon(
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

// Abs parallel AVX512 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn abs_parallel_avx512(
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

// Parallel scalar fallback for abs
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn abs_parallel_scalar(
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

// Sub parallel AVX2 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sub_parallel_avx2(
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

// Sub parallel NEON kernel
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn sub_parallel_neon(
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

// Sub parallel AVX512 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn sub_parallel_avx512(
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

// Parallel scalar fallback for sub
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn sub_parallel_scalar(
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

// Fused add+relu parallel AVX2 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn fused_add_relu_parallel_avx2(
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

// Fused add+relu parallel AVX512 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn fused_add_relu_parallel_avx512(
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

// Parallel scalar fallback for fused_add_relu
#[cfg(feature = "parallel")]
fn fused_add_relu_parallel_scalar(
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

// Fused add+relu parallel NEON kernel
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn fused_add_relu_parallel_neon(
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

// Fused mul+add parallel AVX2 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_mul_add_parallel_avx2(
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

// Fused mul+add parallel AVX512 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn fused_mul_add_parallel_avx512(
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

// Fused mul+add parallel NEON kernel
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn fused_mul_add_parallel_neon(
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

// Parallel scalar fallback for fused_mul_add
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn fused_mul_add_parallel_scalar(
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

// Parallel sigmoid AVX2 kernel using exp approximation
#[cfg(all(feature = "parallel", feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sigmoid_parallel_avx2(
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
        // Compute sigmoid using 1/(1+exp(-x))
        // Process elements one by one for now - Section 3 will vectorize this
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
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }

    while i < end {
        let x = *((a_usize + i * 4) as *const f32);
        if x > 9.0 {
            *((out_usize + i * 4) as *mut f32) = 1.0;
        } else if x < -9.0 {
            *((out_usize + i * 4) as *mut f32) = 0.0;
        } else {
            *((out_usize + i * 4) as *mut f32) = 1.0 / (1.0 + (-x).exp());
        }
        i += 1;
    }
}

// Sigmoid parallel AVX512 kernel
#[cfg(all(feature = "parallel", feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn sigmoid_parallel_avx512(
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
        // Compute sigmoid using 1/(1+exp(-x))
        // Process elements one by one for now - Section 3 will vectorize this
        let x_f32 = std::mem::transmute::<__m512, [f32; 16]>(x);
        let mut results = [0.0f32; 16];
        for j in 0..16 {
            let val = x_f32[j];
            if val > 9.0 {
                results[j] = 1.0;
            } else if val < -9.0 {
                results[j] = 0.0;
            } else {
                results[j] = 1.0 / (1.0 + (-val).exp());
            }
        }
        let result = std::mem::transmute::<[f32; 16], __m512>(results);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
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
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let x = *((a_usize + i * 4) as *const f32);
        if x > 9.0 {
            *((out_usize + i * 4) as *mut f32) = 1.0;
        } else if x < -9.0 {
            *((out_usize + i * 4) as *mut f32) = 0.0;
        } else {
            *((out_usize + i * 4) as *mut f32) = 1.0 / (1.0 + (-x).exp());
        }
        i += 1;
    }
}

// Parallel scalar fallback for sigmoid
#[cfg(feature = "parallel")]
fn sigmoid_parallel_scalar(
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

// Parallel sigmoid NEON kernel using wide::f32x4
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn sigmoid_parallel_neon(
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

// Parallel tanh AVX2 kernel using exact computation
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn tanh_parallel_avx2(
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

    let mut i = start;
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);

        // Clamp values to prevent overflow in exp
        let x_clamped = _mm256_min_ps(_mm256_max_ps(x, clamp_lo), clamp_hi);

        // Compute tanh using formula: tanh(x) = (e^2x - 1) / (e^2x + 1)
        // Process elements one by one for now - Section 3 will vectorize this
        let x_f32 = std::mem::transmute::<__m256, [f32; 8]>(x_clamped);
        let mut results = [0.0f32; 8];
        for j in 0..8 {
            let val = x_f32[j];
            let exp_2x = (2.0 * val).exp();
            results[j] = (exp_2x - 1.0) / (exp_2x + 1.0);
        }
        let result = std::mem::transmute::<[f32; 8], __m256>(results);
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

// Tanh parallel AVX512 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn tanh_parallel_avx512(
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

    let mut i = start;
    while i + 16 <= end {
        let x = _mm512_loadu_ps((a_usize + i * 4) as *const f32);

        // Clamp values to prevent overflow in exp
        let x_clamped = _mm512_min_ps(_mm512_max_ps(x, clamp_lo), clamp_hi);

        // Compute tanh using formula: tanh(x) = (e^2x - 1) / (e^2x + 1)
        // Process elements one by one for now - Section 3 will vectorize this
        let x_f32 = std::mem::transmute::<__m512, [f32; 16]>(x_clamped);
        let mut results = [0.0f32; 16];
        for j in 0..16 {
            let val = x_f32[j];
            let exp_2x = (2.0 * val).exp();
            results[j] = (exp_2x - 1.0) / (exp_2x + 1.0);
        }
        let result = std::mem::transmute::<[f32; 16], __m512>(results);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2 for remaining 8, then scalar for remaining < 8
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let clamp_lo_256 = _mm256_set1_ps(-10.0f32);
        let clamp_hi_256 = _mm256_set1_ps(10.0f32);
        let x_clamped = _mm256_min_ps(_mm256_max_ps(x, clamp_lo_256), clamp_hi_256);
        let x_f32 = std::mem::transmute::<__m256, [f32; 8]>(x_clamped);
        let mut results = [0.0f32; 8];
        for j in 0..8 {
            let val = x_f32[j];
            let exp_2x = (2.0 * val).exp();
            results[j] = (exp_2x - 1.0) / (exp_2x + 1.0);
        }
        let result = std::mem::transmute::<[f32; 8], __m256>(results);
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

// Parallel scalar fallback for tanh
#[cfg(feature = "parallel")]
fn tanh_parallel_scalar(
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

// Parallel tanh NEON kernel using wide::f32x4
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn tanh_parallel_neon(
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
fn relu_simd(input: &[f32], output: &mut [f32]) {
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

// Exp kernel using wide library for SIMD
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
fn exp_simd(input: &[f32], output: &mut [f32]) {
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

// Log kernel using wide library for SIMD
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
fn log_simd(input: &[f32], output: &mut [f32]) {
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

// Sqrt kernel using wide library for SIMD
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
fn sqrt_simd(input: &[f32], output: &mut [f32]) {
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

// GELU kernel using wide library for SIMD
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
fn gelu_simd(input: &[f32], output: &mut [f32]) {
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

// SiLU kernel using wide library for SIMD
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
fn silu_simd(input: &[f32], output: &mut [f32]) {
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

// Exp kernel using wide library for SIMD on ARM
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
fn exp_simd(input: &[f32], output: &mut [f32]) {
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

// Log kernel using wide library for SIMD on ARM
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
fn log_simd(input: &[f32], output: &mut [f32]) {
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

// Sqrt kernel using wide library for SIMD on ARM
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
fn sqrt_simd(input: &[f32], output: &mut [f32]) {
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
fn gelu_simd(input: &[f32], output: &mut [f32]) {
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
fn tanh_simd(input: &[f32], output: &mut [f32]) {
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
fn tanh_simd(input: &[f32], output: &mut [f32]) {
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
fn sigmoid_simd(input: &[f32], output: &mut [f32]) {
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

// Sigmoid SIMD for x86_64 using exp approximation
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(dead_code)]
fn sigmoid_simd_x86(input: &[f32], output: &mut [f32]) {
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

#[allow(dead_code)]
fn create_output(tensor: &Tensor, shape: Vec<i64>) -> Tensor {
    let sizes: smallvec::SmallVec<[i64; 8]> = shape.into();
    let numel: i64 = sizes.iter().product();
    let nbytes = (numel * tensor.dtype().size() as i64) as usize;
    let storage = Arc::new(Storage::new_cpu(tensor.dtype(), nbytes));
    Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        sizes,
        tensor.dtype(),
    ))
}

#[inline]
#[allow(dead_code)]
fn broadcast_shapes_simple(a: &[i64], b: &[i64]) -> Vec<i64> {
    let ndim = std::cmp::max(a.len(), b.len());
    let mut result = vec![1i64; ndim];

    let offset_a = ndim - a.len();
    let offset_b = ndim - b.len();

    for i in 0..ndim {
        let a_val = if i < offset_a { 1 } else { a[i - offset_a] };
        let b_val = if i < offset_b { 1 } else { b[i - offset_b] };
        result[i] = a_val.max(b_val);
    }
    result
}

// Section 7: Optimized broadcast index decomposition
// Precomputes multipliers to map output index to input index without loops
#[inline]
fn broadcast_index_decomposition(
    idx: usize,
    out_shape: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
    storage_offset: usize,
) -> usize {
    let ndim = out_shape.len();
    // Precompute multipliers: multipliers[i] = product(out_shape[i+1..])
    let mut multipliers = vec![0usize; ndim];
    let mut mult = 1usize;
    for i in (0..ndim).rev() {
        multipliers[i] = mult;
        mult *= out_shape[i];
    }

    // Map output index to 1D input index
    let mut input_idx = 0usize;
    for i in 0..ndim {
        let dim_idx = (idx / multipliers[i]) % out_shape[i];
        if i < input_shape.len() && input_shape[i] != 1 {
            input_idx += dim_idx * input_strides[i];
        }
    }
    input_idx + storage_offset
}

fn add_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.inner.sizes.as_slice();
    let b_shape = b.inner.sizes.as_slice();
    let a_contig = a.is_contiguous();
    let b_contig = b.is_contiguous();
    let a_numel = a.inner.numel() as usize;

    if a_contig && b_contig && a_shape == b_shape && a_numel > 2048 {
        let output_shape = a_shape.to_vec();
        let numel = a_numel;

        let mut output = Tensor::empty(output_shape, a.dtype(), a.device());

        let a_ptr = a.data_ptr_f32();
        let b_ptr = b.data_ptr_f32();

        let output_inner = Arc::make_mut(&mut output.inner);
        let output_storage = Arc::make_mut(&mut output_inner.storage);
        let Storage::Cpu(cpu_storage) = output_storage else {
            panic!("Expected CPU storage");
        };
        let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            add_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            add_parallel_avx2(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        add_parallel_scalar(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        add_parallel_neon(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    add_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm512_loadu_ps(b_ptr.add(i));
                        let result = _mm512_add_ps(a_vec, b_vec);
                        _mm512_storeu_ps(out_ptr.add(i), result);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_add_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = _mm256_loadu_ps(a_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let b0 = _mm256_loadu_ps(b_ptr.add(i));
                        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                        let r0 = _mm256_add_ps(a0, b0);
                        let r1 = _mm256_add_ps(a1, b1);
                        _mm256_storeu_ps(out_ptr.add(i), r0);
                        _mm256_storeu_ps(out_ptr.add(i + 8), r1);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_add_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) + *b_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));

                        let b0 = vld1q_f32(b_ptr.add(i));
                        let b1 = vld1q_f32(b_ptr.add(i + 4));
                        let b2 = vld1q_f32(b_ptr.add(i + 8));
                        let b3 = vld1q_f32(b_ptr.add(i + 12));

                        let r0 = vaddq_f32(a0, b0);
                        let r1 = vaddq_f32(a1, b1);
                        let r2 = vaddq_f32(a2, b2);
                        let r3 = vaddq_f32(a3, b3);

                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);

                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let b_vec = vld1q_f32(b_ptr.add(i));
                        let result = vaddq_f32(a_vec, b_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                        i += 1;
                    }
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) + *b_ptr.add(idx);
                    }
                }
            }
        }
        return vec![output];
    }

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();
    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    // Convert to usize for the helper function
    let out_shape_usize: Vec<usize> = out_shape.iter().map(|&x| x as usize).collect();
    let a_shape_usize: Vec<usize> = a_shape.iter().map(|&x| x as usize).collect();
    let b_shape_usize: Vec<usize> = b_shape.iter().map(|&x| x as usize).collect();
    let a_strides_usize: Vec<usize> = a_strides.iter().map(|&x| x as usize).collect();
    let b_strides_usize: Vec<usize> = b_strides.iter().map(|&x| x as usize).collect();

    for idx in 0..numel {
        let a_idx = broadcast_index_decomposition(
            idx,
            &out_shape_usize,
            &a_shape_usize,
            &a_strides_usize,
            a_storage_offset,
        );
        let b_idx = broadcast_index_decomposition(
            idx,
            &out_shape_usize,
            &b_shape_usize,
            &b_strides_usize,
            b_storage_offset,
        );

        unsafe {
            let a_val = *a_ptr.add(a_idx);
            let b_val = *b_ptr.add(b_idx);
            *out_ptr.add(idx) = a_val + b_val;
        }
    }

    vec![output]
}

fn sub_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.inner.sizes.as_slice();
    let b_shape = b.inner.sizes.as_slice();
    let a_contig = a.is_contiguous();
    let b_contig = b.is_contiguous();
    let a_numel = a.inner.numel() as usize;

    if a_contig && b_contig && a_shape == b_shape && a_numel > 2048 {
        let output_shape = a_shape.to_vec();
        let numel = a_numel;

        let mut output = Tensor::empty(output_shape, a.dtype(), a.device());

        let a_ptr = a.data_ptr_f32();
        let b_ptr = b.data_ptr_f32();

        let output_inner = Arc::make_mut(&mut output.inner);
        let output_storage = Arc::make_mut(&mut output_inner.storage);
        let Storage::Cpu(cpu_storage) = output_storage else {
            panic!("Expected CPU storage");
        };
        let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            sub_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            sub_parallel_avx2(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        sub_parallel_scalar(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    sub_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm512_loadu_ps(b_ptr.add(i));
                        let result = _mm512_sub_ps(a_vec, b_vec);
                        _mm512_storeu_ps(out_ptr.add(i), result);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_sub_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = _mm256_loadu_ps(a_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let b0 = _mm256_loadu_ps(b_ptr.add(i));
                        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                        let r0 = _mm256_sub_ps(a0, b0);
                        let r1 = _mm256_sub_ps(a1, b1);
                        _mm256_storeu_ps(out_ptr.add(i), r0);
                        _mm256_storeu_ps(out_ptr.add(i + 8), r1);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_sub_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) - *b_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));

                        let b0 = vld1q_f32(b_ptr.add(i));
                        let b1 = vld1q_f32(b_ptr.add(i + 4));
                        let b2 = vld1q_f32(b_ptr.add(i + 8));
                        let b3 = vld1q_f32(b_ptr.add(i + 12));

                        let r0 = vsubq_f32(a0, b0);
                        let r1 = vsubq_f32(a1, b1);
                        let r2 = vsubq_f32(a2, b2);
                        let r3 = vsubq_f32(a3, b3);

                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);

                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let b_vec = vld1q_f32(b_ptr.add(i));
                        let result = vsubq_f32(a_vec, b_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i);
                        i += 1;
                    }
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) - *b_ptr.add(idx);
                    }
                }
            }
        }
        return vec![output];
    }

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();
    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    // Convert to usize for the helper function
    let out_shape_usize: Vec<usize> = out_shape.iter().map(|&x| x as usize).collect();
    let a_shape_usize: Vec<usize> = a_shape.iter().map(|&x| x as usize).collect();
    let b_shape_usize: Vec<usize> = b_shape.iter().map(|&x| x as usize).collect();
    let a_strides_usize: Vec<usize> = a_strides.iter().map(|&x| x as usize).collect();
    let b_strides_usize: Vec<usize> = b_strides.iter().map(|&x| x as usize).collect();

    // Broadcast loop should always run when shapes differ or tensors are non-contiguous
    for idx in 0..numel {
        let a_idx = broadcast_index_decomposition(
            idx,
            &out_shape_usize,
            &a_shape_usize,
            &a_strides_usize,
            a_storage_offset,
        );
        let b_idx = broadcast_index_decomposition(
            idx,
            &out_shape_usize,
            &b_shape_usize,
            &b_strides_usize,
            b_storage_offset,
        );

        unsafe {
            let a_val = *a_ptr.add(a_idx);
            let b_val = *b_ptr.add(b_idx);
            *out_ptr.add(idx) = a_val - b_val;
        }
    }

    vec![output]
}

fn mul_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.inner.sizes.as_slice();
    let b_shape = b.inner.sizes.as_slice();
    let a_contig = a.is_contiguous();
    let b_contig = b.is_contiguous();
    let a_numel = a.inner.numel() as usize;

    if a_contig && b_contig && a_shape == b_shape && a_numel > 2048 {
        let output_shape = a_shape.to_vec();
        let numel = a_numel;

        let mut output = Tensor::empty(output_shape, a.dtype(), a.device());

        let a_ptr = a.data_ptr_f32();
        let b_ptr = b.data_ptr_f32();

        let output_inner = Arc::make_mut(&mut output.inner);
        let output_storage = Arc::make_mut(&mut output_inner.storage);
        let Storage::Cpu(cpu_storage) = output_storage else {
            panic!("Expected CPU storage");
        };
        let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            mul_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            mul_parallel_avx2(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        mul_parallel_scalar(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        mul_parallel_neon(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    mul_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm512_loadu_ps(b_ptr.add(i));
                        let result = _mm512_mul_ps(a_vec, b_vec);
                        _mm512_storeu_ps(out_ptr.add(i), result);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_mul_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = _mm256_loadu_ps(a_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let b0 = _mm256_loadu_ps(b_ptr.add(i));
                        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                        let r0 = _mm256_mul_ps(a0, b0);
                        let r1 = _mm256_mul_ps(a1, b1);
                        _mm256_storeu_ps(out_ptr.add(i), r0);
                        _mm256_storeu_ps(out_ptr.add(i + 8), r1);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_mul_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) * *b_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));

                        let b0 = vld1q_f32(b_ptr.add(i));
                        let b1 = vld1q_f32(b_ptr.add(i + 4));
                        let b2 = vld1q_f32(b_ptr.add(i + 8));
                        let b3 = vld1q_f32(b_ptr.add(i + 12));

                        let r0 = vmulq_f32(a0, b0);
                        let r1 = vmulq_f32(a1, b1);
                        let r2 = vmulq_f32(a2, b2);
                        let r3 = vmulq_f32(a3, b3);

                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);

                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let b_vec = vld1q_f32(b_ptr.add(i));
                        let result = vmulq_f32(a_vec, b_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                        i += 1;
                    }
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) * *b_ptr.add(idx);
                    }
                }
            }
        }
        return vec![output];
    }

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    // Convert to usize for the helper function
    let out_shape_usize: Vec<usize> = out_shape.iter().map(|&x| x as usize).collect();
    let a_shape_usize: Vec<usize> = a_shape.iter().map(|&x| x as usize).collect();
    let b_shape_usize: Vec<usize> = b_shape.iter().map(|&x| x as usize).collect();
    let a_strides_usize: Vec<usize> = a_strides.iter().map(|&x| x as usize).collect();
    let b_strides_usize: Vec<usize> = b_strides.iter().map(|&x| x as usize).collect();

    for idx in 0..numel {
        let a_idx = broadcast_index_decomposition(
            idx,
            &out_shape_usize,
            &a_shape_usize,
            &a_strides_usize,
            a_storage_offset,
        );
        let b_idx = broadcast_index_decomposition(
            idx,
            &out_shape_usize,
            &b_shape_usize,
            &b_strides_usize,
            b_storage_offset,
        );

        unsafe {
            let a_val = *a_ptr.add(a_idx);
            let b_val = *b_ptr.add(b_idx);
            *out_ptr.add(idx) = a_val * b_val;
        }
    }

    vec![output]
}

fn div_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    // Convert to usize for the helper function
    let out_shape_usize: Vec<usize> = out_shape.iter().map(|&x| x as usize).collect();
    let a_shape_usize: Vec<usize> = a_shape.iter().map(|&x| x as usize).collect();
    let b_shape_usize: Vec<usize> = b_shape.iter().map(|&x| x as usize).collect();
    let a_strides_usize: Vec<usize> = a_strides.iter().map(|&x| x as usize).collect();
    let b_strides_usize: Vec<usize> = b_strides.iter().map(|&x| x as usize).collect();

    // Check if broadcasting is needed - only use parallel path when shapes are equal
    let needs_broadcast = a_shape != b_shape;
    let use_parallel = a.is_contiguous() && b.is_contiguous() && numel > 2048 && !needs_broadcast;

    if use_parallel {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            div_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            div_parallel_avx2(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        div_parallel_scalar(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        div_parallel_neon(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    div_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm512_loadu_ps(b_ptr.add(i));
                        let result = _mm512_div_ps(a_vec, b_vec);
                        _mm512_storeu_ps(out_ptr.add(i), result);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_div_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) / *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = _mm256_loadu_ps(a_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let b0 = _mm256_loadu_ps(b_ptr.add(i));
                        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                        let r0 = _mm256_div_ps(a0, b0);
                        let r1 = _mm256_div_ps(a1, b1);
                        _mm256_storeu_ps(out_ptr.add(i), r0);
                        _mm256_storeu_ps(out_ptr.add(i + 8), r1);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_div_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) / *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) / *b_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));

                        let b0 = vld1q_f32(b_ptr.add(i));
                        let b1 = vld1q_f32(b_ptr.add(i + 4));
                        let b2 = vld1q_f32(b_ptr.add(i + 8));
                        let b3 = vld1q_f32(b_ptr.add(i + 12));

                        let r0 = vdivq_f32(a0, b0);
                        let r1 = vdivq_f32(a1, b1);
                        let r2 = vdivq_f32(a2, b2);
                        let r3 = vdivq_f32(a3, b3);

                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);

                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let b_vec = vld1q_f32(b_ptr.add(i));
                        let result = vdivq_f32(a_vec, b_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) / *b_ptr.add(i);
                        i += 1;
                    }
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) / *b_ptr.add(idx);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            let a_idx = broadcast_index_decomposition(
                idx,
                &out_shape_usize,
                &a_shape_usize,
                &a_strides_usize,
                a_storage_offset,
            );
            let b_idx = broadcast_index_decomposition(
                idx,
                &out_shape_usize,
                &b_shape_usize,
                &b_strides_usize,
                b_storage_offset,
            );

            unsafe {
                let a_val = *a_ptr.add(a_idx);
                let b_val = *b_ptr.add(b_idx);
                *out_ptr.add(idx) = a_val / b_val;
            }
        }
    }

    vec![output]
}

fn neg_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            neg_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            neg_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        neg_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    neg_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let result = _mm512_xor_ps(a_vec, _mm512_set1_ps(-0.0f32));
                        _mm512_storeu_ps(out_ptr.add(i), result);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let result = _mm256_xor_ps(a_vec, _mm256_set1_ps(-0.0f32));
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = -*a_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = _mm256_loadu_ps(a_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let r0 = _mm256_xor_ps(a0, _mm256_set1_ps(-0.0f32));
                        let r1 = _mm256_xor_ps(a1, _mm256_set1_ps(-0.0f32));
                        _mm256_storeu_ps(out_ptr.add(i), r0);
                        _mm256_storeu_ps(out_ptr.add(i + 8), r1);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let result = _mm256_xor_ps(a_vec, _mm256_set1_ps(-0.0f32));
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = -*a_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = -*a_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));

                        let r0 = vnegq_f32(a0);
                        let r1 = vnegq_f32(a1);
                        let r2 = vnegq_f32(a2);
                        let r3 = vnegq_f32(a3);

                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);

                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let result = vnegq_f32(a_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = -*a_ptr.add(i);
                        i += 1;
                    }
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = -*a_ptr.add(idx);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = -val;
            }
        }
    }

    vec![output]
}

fn abs_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            abs_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            abs_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        abs_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        abs_parallel_neon(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    abs_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));

                        let r0 = vabsq_f32(a0);
                        let r1 = vabsq_f32(a1);
                        let r2 = vabsq_f32(a2);
                        let r3 = vabsq_f32(a3);

                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);

                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let result = vabsq_f32(a_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = (*a_ptr.add(i)).abs();
                        i += 1;
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).abs();
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.abs();
            }
        }
    }

    vec![output]
}

// Parallel exp AVX2 kernel using fast_exp_avx2
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_parallel_avx2(
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

// Parallel exp AVX512 kernel using fast_exp_avx512
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn exp_parallel_avx512(
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

// Parallel exp NEON kernel using wide::f32x4
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn exp_parallel_neon(
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

// Parallel log AVX2 kernel using fast_log_avx2
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn log_parallel_avx2(
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

// Parallel log AVX512 kernel using fast_log_avx512
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn log_parallel_avx512(
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

// Parallel log NEON kernel using wide::f32x4
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn log_parallel_neon(
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

fn exp_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            exp_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            exp_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        let start = chunk_idx * chunk_size;
                        let end = std::cmp::min(start + chunk_size, numel);

                        for i in start..end {
                            unsafe {
                                *((out_usize + i * 4) as *mut f32) =
                                    (*((a_usize + i * 4) as *const f32)).exp();
                            }
                        }
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        exp_parallel_neon(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = std::cmp::min(start + chunk_size, numel);

                    for i in start..end {
                        unsafe {
                            *((out_usize + i * 4) as *mut f32) =
                                (*((a_usize + i * 4) as *const f32)).exp();
                        }
                    }
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                exp_simd(a_slice, out_slice);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                exp_simd(a_slice, out_slice);
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).exp();
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.exp();
            }
        }
    }

    vec![output]
}

fn log_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            log_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            log_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        let start = chunk_idx * chunk_size;
                        let end = std::cmp::min(start + chunk_size, numel);

                        for i in start..end {
                            unsafe {
                                *((out_usize + i * 4) as *mut f32) =
                                    (*((a_usize + i * 4) as *const f32)).ln();
                            }
                        }
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        log_parallel_neon(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = std::cmp::min(start + chunk_size, numel);

                    for i in start..end {
                        unsafe {
                            *((out_usize + i * 4) as *mut f32) =
                                (*((a_usize + i * 4) as *const f32)).ln();
                        }
                    }
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                log_simd(a_slice, out_slice);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                log_simd(a_slice, out_slice);
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).ln();
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.ln();
            }
        }
    }

    vec![output]
}

// Parallel sqrt AVX2 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sqrt_parallel_avx2(
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

// Parallel sqrt AVX512 kernel
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn sqrt_parallel_avx512(
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

// Parallel sqrt NEON kernel using wide::f32x4
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn sqrt_parallel_neon(
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

fn sqrt_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            sqrt_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            sqrt_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        let start = chunk_idx * chunk_size;
                        let end = std::cmp::min(start + chunk_size, numel);

                        for i in start..end {
                            unsafe {
                                *((out_usize + i * 4) as *mut f32) =
                                    (*((a_usize + i * 4) as *const f32)).sqrt();
                            }
                        }
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        sqrt_parallel_neon(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = std::cmp::min(start + chunk_size, numel);

                    for i in start..end {
                        unsafe {
                            *((out_usize + i * 4) as *mut f32) =
                                (*((a_usize + i * 4) as *const f32)).sqrt();
                        }
                    }
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                sqrt_simd(a_slice, out_slice);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                sqrt_simd(a_slice, out_slice);
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).sqrt();
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.sqrt();
            }
        }
    }

    vec![output]
}

fn relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            relu_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            relu_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        relu_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        relu_parallel_neon(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    relu_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 | SimdLevel::Avx2 => {
                    let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                    relu_simd(a_slice, out_slice);
                }
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            let val = *a_ptr.add(idx);
                            *out_ptr.add(idx) = val.max(0.0);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                let zero = f32x4::ZERO;
                let (chunks, remainder) = a_slice.as_chunks::<4>();
                let (out_chunks, out_remainder) = out_slice.as_chunks_mut::<4>();
                for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
                    let v = f32x4::from(*in_chunk);
                    let result = v.max(zero);
                    *out_chunk = result.into();
                }

                for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
                    *out_val = in_val.max(0.0);
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        let val = *a_ptr.add(idx);
                        *out_ptr.add(idx) = val.max(0.0);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.max(0.0);
            }
        }
    }

    vec![output]
}

#[inline]
fn fused_add_relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.shape();
    let _b_shape = b.shape();
    let output_shape = a_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && b.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            fused_add_relu_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            fused_add_relu_parallel_avx2(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        fused_add_relu_parallel_scalar(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        fused_add_relu_parallel_neon(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    fused_add_relu_parallel_scalar(
                        chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                    );
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let zero = f32x4::ZERO;
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

                let (a_chunks, a_rem) = a_slice.as_chunks::<4>();
                let (b_chunks, b_rem) = b_slice.as_chunks::<4>();
                let (out_chunks, out_rem) = out_slice.as_chunks_mut::<4>();

                for ((a_chunk, b_chunk), out_chunk) in a_chunks
                    .iter()
                    .zip(b_chunks.iter())
                    .zip(out_chunks.iter_mut())
                {
                    let a_vec = f32x4::from(*a_chunk);
                    let b_vec = f32x4::from(*b_chunk);
                    let sum = a_vec + b_vec;
                    let result = sum.max(zero);
                    *out_chunk = result.into();
                }
                for ((a_val, b_val), out_val) in
                    a_rem.iter().zip(b_rem.iter()).zip(out_rem.iter_mut())
                {
                    *out_val = (*a_val + *b_val).max(0.0);
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let zero = _mm512_set1_ps(0.0f32);
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm512_loadu_ps(b_ptr.add(i));
                        let sum = _mm512_add_ps(a_vec, b_vec);
                        let relu = _mm512_max_ps(sum, zero);
                        _mm512_storeu_ps(out_ptr.add(i), relu);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let sum = _mm256_add_ps(a_vec, b_vec);
                        let relu = _mm256_max_ps(sum, _mm256_set1_ps(0.0f32));
                        _mm256_storeu_ps(out_ptr.add(i), relu);
                        i += 8;
                    }
                    while i < numel {
                        let val = *a_ptr.add(i) + *b_ptr.add(i);
                        *out_ptr.add(i) = val.max(0.0);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let zero = _mm256_set1_ps(0.0f32);
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = _mm256_loadu_ps(a_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let b0 = _mm256_loadu_ps(b_ptr.add(i));
                        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                        let sum0 = _mm256_add_ps(a0, b0);
                        let sum1 = _mm256_add_ps(a1, b1);
                        let relu0 = _mm256_max_ps(sum0, zero);
                        let relu1 = _mm256_max_ps(sum1, zero);
                        _mm256_storeu_ps(out_ptr.add(i), relu0);
                        _mm256_storeu_ps(out_ptr.add(i + 8), relu1);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let sum = _mm256_add_ps(a_vec, b_vec);
                        let relu = _mm256_max_ps(sum, zero);
                        _mm256_storeu_ps(out_ptr.add(i), relu);
                        i += 8;
                    }
                    while i < numel {
                        let val = *a_ptr.add(i) + *b_ptr.add(i);
                        *out_ptr.add(i) = val.max(0.0);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            let val = *a_ptr.add(idx) + *b_ptr.add(idx);
                            *out_ptr.add(idx) = val.max(0.0);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

                let (a_chunks, a_rem) = a_slice.as_chunks::<4>();
                let (b_chunks, b_rem) = b_slice.as_chunks::<4>();
                let (out_chunks, out_rem) = out_slice.as_chunks_mut::<4>();

                let zero = f32x4::ZERO;
                for ((a_chunk, b_chunk), out_chunk) in a_chunks
                    .iter()
                    .zip(b_chunks.iter())
                    .zip(out_chunks.iter_mut())
                {
                    let a_vec = f32x4::from(*a_chunk);
                    let b_vec = f32x4::from(*b_chunk);
                    let sum = a_vec + b_vec;
                    let result = sum.max(zero);
                    *out_chunk = result.into();
                }

                for ((a_val, b_val), out_val) in
                    a_rem.iter().zip(b_rem.iter()).zip(out_rem.iter_mut())
                {
                    *out_val = (*a_val + *b_val).max(0.0);
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        let val = *a_ptr.add(idx) + *b_ptr.add(idx);
                        *out_ptr.add(idx) = val.max(0.0);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx) + *b_ptr.add(idx);
                *out_ptr.add(idx) = val.max(0.0);
            }
        }
    }

    vec![output]
}

#[inline]
fn fused_mul_add_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];
    let c = args[2];

    let a_shape = a.shape();
    let output_shape = a_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;
    let c_ptr = c.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && b.is_contiguous() && c.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let c_usize = c_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            fused_mul_add_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, c_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            fused_mul_add_parallel_avx2(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, c_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        fused_mul_add_parallel_scalar(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, c_usize, out_usize,
                        );
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        fused_mul_add_parallel_neon(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, c_usize, out_usize,
                        );
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    fused_mul_add_parallel_scalar(
                        chunk_idx, chunk_size, numel, a_usize, b_usize, c_usize, out_usize,
                    );
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm512_loadu_ps(b_ptr.add(i));
                        let c_vec = _mm512_loadu_ps(c_ptr.add(i));
                        let result = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                        _mm512_storeu_ps(out_ptr.add(i), result);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let c_vec = _mm256_loadu_ps(c_ptr.add(i));
                        let result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i) + *c_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let mut i = 0usize;
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let c_vec = _mm256_loadu_ps(c_ptr.add(i));
                        let result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i) + *c_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) * *b_ptr.add(idx) + *c_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));
                        let b0 = vld1q_f32(b_ptr.add(i));
                        let b1 = vld1q_f32(b_ptr.add(i + 4));
                        let b2 = vld1q_f32(b_ptr.add(i + 8));
                        let b3 = vld1q_f32(b_ptr.add(i + 12));
                        let c0 = vld1q_f32(c_ptr.add(i));
                        let c1 = vld1q_f32(c_ptr.add(i + 4));
                        let c2 = vld1q_f32(c_ptr.add(i + 8));
                        let c3 = vld1q_f32(c_ptr.add(i + 12));
                        let r0 = vfmaq_f32(c0, a0, b0);
                        let r1 = vfmaq_f32(c1, a1, b1);
                        let r2 = vfmaq_f32(c2, a2, b2);
                        let r3 = vfmaq_f32(c3, a3, b3);
                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);
                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let b_vec = vld1q_f32(b_ptr.add(i));
                        let c_vec = vld1q_f32(c_ptr.add(i));
                        let result = vfmaq_f32(c_vec, a_vec, b_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i) + *c_ptr.add(i);
                        i += 1;
                    }
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) * *b_ptr.add(idx) + *c_ptr.add(idx);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let a_val = *a_ptr.add(idx);
                let b_val = *b_ptr.add(idx);
                let c_val = *c_ptr.add(idx);
                *out_ptr.add(idx) = a_val * b_val + c_val;
            }
        }
    }

    vec![output]
}

// Parallel GELU AVX2 kernel using fast_exp_avx2 for tanh
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_parallel_avx2(
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

// Parallel GELU AVX512 kernel using fast_exp_avx512 for tanh
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn gelu_parallel_avx512(
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

// Parallel GELU NEON kernel using wide::f32x4
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn gelu_parallel_neon(
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

fn gelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            gelu_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            gelu_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        let start = chunk_idx * chunk_size;
                        let end = std::cmp::min(start + chunk_size, numel);

                        for i in start..end {
                            unsafe {
                                let x = *((a_usize + i * 4) as *const f32);
                                let x3 = x * x * x;
                                let t = (0.797_884_6 * (x + 0.044715 * x3)).tanh();
                                *((out_usize + i * 4) as *mut f32) = 0.5 * x * (1.0 + t);
                            }
                        }
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        gelu_parallel_neon(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = std::cmp::min(start + chunk_size, numel);

                    for i in start..end {
                        unsafe {
                            let x = *((a_usize + i * 4) as *const f32);
                            let x3 = x * x * x;
                            let t = (0.797_884_6 * (x + 0.044715 * x3)).tanh();
                            *((out_usize + i * 4) as *mut f32) = 0.5 * x * (1.0 + t);
                        }
                    }
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                gelu_simd(a_slice, out_slice);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                gelu_simd(a_slice, out_slice);
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        let x = *a_ptr.add(idx);
                        let x3 = x * x * x;
                        let t = (0.797_884_6 * (x + 0.044715 * x3)).tanh();
                        *out_ptr.add(idx) = 0.5 * x * (1.0 + t);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                let x3 = x * x * x;
                let t = (0.797_884_6 * (x + 0.044715 * x3)).tanh();
                *out_ptr.add(idx) = 0.5 * x * (1.0 + t);
            }
        }
    }

    vec![output]
}

fn sigmoid_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            // Use runtime feature detection for x86_64
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            sigmoid_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            sigmoid_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        sigmoid_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        sigmoid_parallel_neon(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    sigmoid_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                sigmoid_simd(a_slice, out_slice);
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

                // Use runtime detection to choose implementation
                if is_x86_feature_detected!("avx2") {
                    sigmoid_simd_x86(a_slice, out_slice);
                } else {
                    for idx in 0..numel {
                        unsafe {
                            let x = *a_ptr.add(idx);
                            *out_ptr.add(idx) = 1.0 / (1.0 + (-x).exp());
                        }
                    }
                }
            }
            #[cfg(not(all(
                feature = "simd",
                any(target_arch = "aarch64", target_arch = "x86_64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        let x = *a_ptr.add(idx);
                        *out_ptr.add(idx) = 1.0 / (1.0 + (-x).exp());
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                *out_ptr.add(idx) = 1.0 / (1.0 + (-x).exp());
            }
        }
    }

    vec![output]
}

fn tanh_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            // Use runtime feature detection for x86_64
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            tanh_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            tanh_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        tanh_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        tanh_parallel_neon(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    tanh_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                tanh_simd(a_slice, out_slice);
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

                // Use runtime detection to choose implementation
                if is_x86_feature_detected!("avx2") {
                    tanh_simd(a_slice, out_slice);
                } else {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = (*a_ptr.add(idx)).tanh();
                        }
                    }
                }
            }
            #[cfg(not(all(
                feature = "simd",
                any(target_arch = "aarch64", target_arch = "x86_64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).tanh();
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                *out_ptr.add(idx) = x.tanh();
            }
        }
    }

    vec![output]
}

fn silu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            // SiLU is transcendental, use scalar for parallel path
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = std::cmp::min(start + chunk_size, numel);

                for i in start..end {
                    unsafe {
                        let x = *((a_usize + i * 4) as *const f32);
                        *((out_usize + i * 4) as *mut f32) = x / (1.0 + (-x).exp());
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(feature = "simd")]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                silu_simd(a_slice, out_slice);
            }
            #[cfg(not(feature = "simd"))]
            {
                for idx in 0..numel {
                    unsafe {
                        let x = *a_ptr.add(idx);
                        *out_ptr.add(idx) = x / (1.0 + (-x).exp());
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                *out_ptr.add(idx) = x / (1.0 + (-x).exp());
            }
        }
    }

    vec![output]
}

fn matmul_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    use std::io::Write;
    let mut debug_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/matmul_debug.log")
        .unwrap();
    writeln!(debug_file, "DEBUG: Entered matmul_kernel").unwrap();

    let a = args[0];
    let b = args[1];

    let a_shape = a.shape();
    let b_shape = b.shape();
    writeln!(
        debug_file,
        "DEBUG: a_shape={:?}, b_shape={:?}",
        a_shape, b_shape
    )
    .unwrap();

    if a_shape.len() < 2 || b_shape.len() < 2 {
        panic!("matmul: both tensors must have at least 2 dimensions");
    }

    let m = a_shape[a_shape.len() - 2] as i32;
    let k = a_shape[a_shape.len() - 1] as i32;
    let n = b_shape[b_shape.len() - 1] as i32;

    // Detect transposed matrices by checking strides
    // A contiguous matrix [rows, cols] has strides [cols, 1]
    // A transposed matrix (from [rows, cols]) has strides [1, rows]
    let a_strides = a.strides();
    let b_strides = b.strides();
    let a_is_transposed =
        a_strides[a.ndim() - 2] == 1 && a_strides[a.ndim() - 1] >= a_shape[a_shape.len() - 2];
    let b_is_transposed =
        b_strides[b.ndim() - 2] == 1 && b_strides[b.ndim() - 1] >= b_shape[b_shape.len() - 2];

    // Debug the transposition detection
    eprintln!(
        "DEBUG: b_strides[0]={}, b_strides[1]={}, b_shape[0]={}, check={}",
        b_strides[0],
        b_strides[1],
        b_shape[0],
        b_strides[0] == 1 && b_strides[1] >= b_shape[0]
    );

    // Debug output
    eprintln!(
        "DEBUG matmul: a_shape={:?}, b_shape={:?}, b_strides={:?}, b_is_transposed={}",
        a_shape, b_shape, b_strides, b_is_transposed
    );

    // For matmul: A[m, k] @ B[k, n] = C[m, n]
    // When B is transposed (shape [n, k] representing original [k, n]):
    // The transposed view has shape [n, k] where n is the original outer dim
    // and k is the original inner dim
    if b_is_transposed {
        // B is transposed: shape [n, k] represents original matrix [k, n]
        // For matmul A[m,k] @ B[k,n], we need B's inner dim (k) to match A's inner dim (k)
        // In the transposed view, B's inner dim is b_shape[-1] (which is k)
        let b_inner_dim = b_shape[b_shape.len() - 1] as i32; // This is k
        if b_inner_dim != k {
            panic!(
                "matmul: transposed B dimensions incompatible: A[{}, {}] @ B.T[{}, {}]",
                m,
                k,
                b_shape[b_shape.len() - 2],
                b_shape[b_shape.len() - 1]
            );
        }
    } else {
        // Standard case: B is not transposed, shape [k, n]
        if b_shape[b_shape.len() - 2] as i32 != k {
            panic!("matmul: {} != {}", b_shape[b_shape.len() - 2], k);
        }
    }

    // Use custom tiled matmul
    let batch_a = if a_shape.len() > 2 {
        a_shape[..a_shape.len() - 2].iter().product::<i64>() as usize
    } else {
        1
    };
    let batch_b = if b_shape.len() > 2 {
        b_shape[..b_shape.len() - 2].iter().product::<i64>() as usize
    } else {
        1
    };
    let batch = batch_a.max(batch_b);

    let mut output_shape: Vec<i64> = vec![];
    if a_shape.len() > 2 {
        for i in 0..a_shape.len() - 2 {
            // If b has matching dimensions, use max (broadcasting)
            // If b is 2D (no batch dims), just use a's batch dims
            if b_shape.len() > 2 && i < b_shape.len() - 2 {
                output_shape.push(a_shape[i].max(b_shape[i]));
            } else {
                output_shape.push(a_shape[i]);
            }
        }
    }
    output_shape.push(m as i64);
    output_shape.push(n as i64);

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    let a_rows = a_shape[a_shape.len() - 2] as usize;
    let a_cols = a_shape[a_shape.len() - 1] as usize;
    let b_cols = b_shape[b_shape.len() - 1] as usize;

    let a_strides = a.strides();
    let b_strides = b.strides();

    let a_stride_0 = a_strides[a.ndim() - 2];
    let a_stride_1 = a_strides[a.ndim() - 1];
    let b_stride_0 = b_strides[b.ndim() - 2];
    let b_stride_1 = b_strides[b.ndim() - 1];

    let a_batch_stride = if a.ndim() > 2 { a_strides[0] } else { 0 };
    let b_batch_stride = if b.ndim() > 2 { b_strides[0] } else { 0 };
    // Detect transposed matrices by checking strides
    // For row-major contiguous matrix [rows, cols], stride_0 = cols, stride_1 = 1
    // For transposed [rows, cols] stored as [cols, rows], stride_0 = 1, stride_1 = rows
    let a_is_transposed = a_stride_0 == 1 && a_stride_1 == a_rows as i64;
    let b_is_transposed = b_stride_0 == 1 && b_stride_1 == k as i64;

    // For BLAS, we need contiguous matrices or simple 2D transposition
    // A transposed matrix has stride_0 = 1 and stride_1 = original_rows
    let a_valid_for_blas = a.is_contiguous() || a_is_transposed;
    let b_valid_for_blas = b.is_contiguous() || b_is_transposed;

    let use_blas = batch == 1
        && m as usize >= MIN_BLAS_SIZE
        && n as usize >= MIN_BLAS_SIZE
        && k as usize >= MIN_BLAS_SIZE
        && a_valid_for_blas
        && b_valid_for_blas;

    if use_blas {
        // For BLAS, we can handle transposed matrices by passing the transpose flag
        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, a_rows * a_cols) };
        let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, k as usize * b_cols) };
        let result = matmul_blas_with_transpose(
            a_slice,
            b_slice,
            m as usize,
            k as usize,
            n as usize,
            a_is_transposed,
            b_is_transposed,
        );
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, m as usize * n as usize) };
        out_slice.copy_from_slice(&result);
    } else {
        #[cfg(feature = "parallel")]
        {
            if batch > 1 || m as usize * n as usize > 10000 {
                parallel_matmul(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    batch,
                    m,
                    n,
                    k,
                    a_batch_stride,
                    a_stride_0,
                    a_stride_1,
                    b_batch_stride,
                    b_stride_0,
                    b_stride_1,
                );
            } else if m as usize <= 64 && n as usize <= 64 && k as usize <= 64 {
                small_matrix_matmul(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    batch,
                    m,
                    n,
                    k,
                    a_batch_stride,
                    a_stride_0,
                    a_stride_1,
                    b_batch_stride,
                    b_stride_0,
                    b_stride_1,
                );
            } else {
                single_threaded_matmul(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    batch,
                    m,
                    n,
                    k,
                    a_batch_stride,
                    a_stride_0,
                    a_stride_1,
                    b_batch_stride,
                    b_stride_0,
                    b_stride_1,
                );
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            if m as usize <= 64 && n as usize <= 64 && k as usize <= 64 {
                small_matrix_matmul(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    batch,
                    m,
                    n,
                    k,
                    a_batch_stride,
                    a_stride_0,
                    a_stride_1,
                    b_batch_stride,
                    b_stride_0,
                    b_stride_1,
                );
            } else {
                single_threaded_matmul(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    batch,
                    m,
                    n,
                    k,
                    a_batch_stride,
                    a_stride_0,
                    a_stride_1,
                    b_batch_stride,
                    b_stride_0,
                    b_stride_1,
                );
            }
        }
    }

    vec![output]
}

#[cfg(feature = "parallel")]
#[inline]
fn parallel_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    batch: usize,
    m: i32,
    n: i32,
    k: i32,
    a_batch_stride: i64,
    a_stride_0: i64,
    a_stride_1: i64,
    b_batch_stride: i64,
    b_stride_0: i64,
    b_stride_1: i64,
) {
    let a_usize = a_ptr as usize;
    let b_usize = b_ptr as usize;
    let out_usize = out_ptr as usize;
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;

    // Parallelize at the row level (batch * m rows)
    let total_rows = batch * m_usize;

    (0..total_rows).into_par_iter().for_each(|row_idx| {
        let bat = row_idx / m_usize;
        let i = row_idx % m_usize;

        let bat_a_offset = bat * a_batch_stride as usize;
        let bat_b_offset = bat * b_batch_stride as usize;

        for j in 0..n_usize {
            let mut sum = 0.0f32;
            let mut kk = 0;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("fma") {
                    unsafe {
                        while kk + 8 <= k_usize {
                            let a0 = _mm256_loadu_ps(
                                (a_usize
                                    + (bat_a_offset
                                        + i * a_stride_0 as usize
                                        + (kk) * a_stride_1 as usize)
                                        * 4) as *const f32,
                            );
                            let a1 = _mm256_loadu_ps(
                                (a_usize
                                    + (bat_a_offset
                                        + i * a_stride_0 as usize
                                        + (kk + 8) * a_stride_1 as usize)
                                        * 4) as *const f32,
                            );

                            let b0 = _mm256_loadu_ps(
                                (b_usize
                                    + (bat_b_offset
                                        + (kk) * b_stride_0 as usize
                                        + j * b_stride_1 as usize)
                                        * 4) as *const f32,
                            );
                            let b1 = _mm256_loadu_ps(
                                (b_usize
                                    + (bat_b_offset
                                        + (kk + 8) * b_stride_0 as usize
                                        + j * b_stride_1 as usize)
                                        * 4) as *const f32,
                            );

                            let acc0 = _mm256_mul_ps(a0, b0);
                            let acc1 = _mm256_mul_ps(a1, b1);
                            let acc = _mm256_add_ps(acc0, acc1);

                            let acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                            _mm256_storeu_ps(acc_arr.as_ptr() as *mut f32, acc);
                            sum += acc_arr.assume_init().iter().sum::<f32>();

                            #[cfg(feature = "prefetch")]
                            {
                                _mm_prefetch(
                                    (b_usize
                                        + (bat_b_offset
                                            + (kk + 16) * b_stride_0 as usize
                                            + j * b_stride_1 as usize)
                                            * 4) as *const i8,
                                    _MM_HINT_T0,
                                );
                                _mm_prefetch(
                                    (a_usize
                                        + (bat_a_offset
                                            + i * a_stride_0 as usize
                                            + (kk + 16) * a_stride_1 as usize)
                                            * 4) as *const i8,
                                    _MM_HINT_T0,
                                );
                            }

                            kk += 16;
                        }
                    }
                }
            }

            while kk + 4 <= k_usize {
                unsafe {
                    let a_base = bat_a_offset + i * a_stride_0 as usize;
                    let a_val =
                        *((a_usize + (a_base + kk * a_stride_1 as usize) * 4) as *const f32);
                    let a_val1 =
                        *((a_usize + (a_base + (kk + 1) * a_stride_1 as usize) * 4) as *const f32);
                    let a_val2 =
                        *((a_usize + (a_base + (kk + 2) * a_stride_1 as usize) * 4) as *const f32);
                    let a_val3 =
                        *((a_usize + (a_base + (kk + 3) * a_stride_1 as usize) * 4) as *const f32);

                    let b_base = bat_b_offset + j * b_stride_1 as usize;
                    let b_val =
                        *((b_usize + (b_base + kk * b_stride_0 as usize) * 4) as *const f32);
                    let b_val1 =
                        *((b_usize + (b_base + (kk + 1) * b_stride_0 as usize) * 4) as *const f32);
                    let b_val2 =
                        *((b_usize + (b_base + (kk + 2) * b_stride_0 as usize) * 4) as *const f32);
                    let b_val3 =
                        *((b_usize + (b_base + (kk + 3) * b_stride_0 as usize) * 4) as *const f32);

                    sum += a_val * b_val + a_val1 * b_val1 + a_val2 * b_val2 + a_val3 * b_val3;
                }
                kk += 4;
            }

            while kk < k_usize {
                unsafe {
                    let a_val = *((a_usize
                        + (bat_a_offset + i * a_stride_0 as usize + kk * a_stride_1 as usize) * 4)
                        as *const f32);
                    let b_val = *((b_usize
                        + (bat_b_offset + kk * b_stride_0 as usize + j * b_stride_1 as usize) * 4)
                        as *const f32);
                    sum += a_val * b_val;
                }
                kk += 1;
            }

            unsafe {
                let out_idx = bat * m_usize * n_usize + i * n_usize + j;
                *((out_usize + out_idx * 4) as *mut f32) = sum;
            };
        }
    });
}

#[inline]
fn small_matrix_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    batch: usize,
    m: i32,
    n: i32,
    k: i32,
    a_batch_stride: i64,
    a_stride_0: i64,
    a_stride_1: i64,
    b_batch_stride: i64,
    b_stride_0: i64,
    b_stride_1: i64,
) {
    for bat in 0..batch {
        let bat_a_offset = bat * a_batch_stride as usize;
        let bat_b_offset = bat * b_batch_stride as usize;

        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut sum = 0.0f32;

                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("fma") {
                        unsafe {
                            let mut acc0 = _mm256_setzero_ps();
                            let mut acc1 = _mm256_setzero_ps();
                            let mut kk = 0usize;

                            while kk + 8 <= k as usize {
                                let a0 = _mm256_loadu_ps(a_ptr.add(
                                    bat_a_offset
                                        + i * a_stride_0 as usize
                                        + kk * a_stride_1 as usize,
                                ));
                                let a1 = _mm256_loadu_ps(a_ptr.add(
                                    bat_a_offset
                                        + i * a_stride_0 as usize
                                        + (kk + 8) * a_stride_1 as usize,
                                ));

                                let b0 = _mm256_loadu_ps(b_ptr.add(
                                    bat_b_offset
                                        + kk * b_stride_0 as usize
                                        + j * b_stride_1 as usize,
                                ));
                                let b1 = _mm256_loadu_ps(b_ptr.add(
                                    bat_b_offset
                                        + (kk + 8) * b_stride_0 as usize
                                        + j * b_stride_1 as usize,
                                ));

                                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                                acc1 = _mm256_fmadd_ps(a1, b1, acc1);

                                kk += 16;
                            }

                            let acc = _mm256_add_ps(acc0, acc1);
                            let acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                            _mm256_storeu_ps(acc_arr.as_ptr() as *mut f32, acc);
                            sum += acc_arr.assume_init().iter().sum::<f32>();

                            while kk < k as usize {
                                let a_val = *a_ptr.add(
                                    bat_a_offset
                                        + i * a_stride_0 as usize
                                        + kk * a_stride_1 as usize,
                                );
                                let b_val = *b_ptr.add(
                                    bat_b_offset
                                        + kk * b_stride_0 as usize
                                        + j * b_stride_1 as usize,
                                );
                                sum += a_val * b_val;
                                kk += 1;
                            }
                        }
                    } else {
                        let mut sum0 = 0.0f32;
                        let mut sum1 = 0.0f32;
                        let mut sum2 = 0.0f32;
                        let mut sum3 = 0.0f32;

                        let mut kk = 0usize;
                        let k_limit = k as usize;

                        while kk + 4 <= k_limit {
                            let a0 = unsafe {
                                *a_ptr.add(
                                    bat_a_offset
                                        + i * a_stride_0 as usize
                                        + kk * a_stride_1 as usize,
                                )
                            };
                            let a1 = unsafe {
                                *a_ptr.add(
                                    bat_a_offset
                                        + i * a_stride_0 as usize
                                        + (kk + 1) * a_stride_1 as usize,
                                )
                            };
                            let a2 = unsafe {
                                *a_ptr.add(
                                    bat_a_offset
                                        + i * a_stride_0 as usize
                                        + (kk + 2) * a_stride_1 as usize,
                                )
                            };
                            let a3 = unsafe {
                                *a_ptr.add(
                                    bat_a_offset
                                        + i * a_stride_0 as usize
                                        + (kk + 3) * a_stride_1 as usize,
                                )
                            };

                            let b_base = bat_b_offset + j * b_stride_1 as usize;
                            let b0 = unsafe { *b_ptr.add(b_base + kk * b_stride_0 as usize) };
                            let b1 = unsafe { *b_ptr.add(b_base + (kk + 1) * b_stride_0 as usize) };
                            let b2 = unsafe { *b_ptr.add(b_base + (kk + 2) * b_stride_0 as usize) };
                            let b3 = unsafe { *b_ptr.add(b_base + (kk + 3) * b_stride_0 as usize) };

                            sum0 += a0 * b0;
                            sum1 += a1 * b1;
                            sum2 += a2 * b2;
                            sum3 += a3 * b3;
                            kk += 4;
                        }

                        sum = sum0 + sum1 + sum2 + sum3;

                        while kk < k_limit {
                            let a_val = unsafe {
                                *a_ptr.add(
                                    bat_a_offset
                                        + i * a_stride_0 as usize
                                        + kk * a_stride_1 as usize,
                                )
                            };
                            let b_val = unsafe {
                                *b_ptr.add(
                                    bat_b_offset
                                        + kk * b_stride_0 as usize
                                        + j * b_stride_1 as usize,
                                )
                            };
                            sum += a_val * b_val;
                            kk += 1;
                        }
                    }
                }

                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    let mut sum0 = 0.0f32;
                    let mut sum1 = 0.0f32;
                    let mut sum2 = 0.0f32;
                    let mut sum3 = 0.0f32;

                    let mut kk = 0usize;
                    let k_limit = k as usize;

                    while kk + 4 <= k_limit {
                        let a0 = unsafe {
                            *a_ptr.add(
                                bat_a_offset + i * a_stride_0 as usize + kk * a_stride_1 as usize,
                            )
                        };
                        let a1 = unsafe {
                            *a_ptr.add(
                                bat_a_offset
                                    + i * a_stride_0 as usize
                                    + (kk + 1) * a_stride_1 as usize,
                            )
                        };
                        let a2 = unsafe {
                            *a_ptr.add(
                                bat_a_offset
                                    + i * a_stride_0 as usize
                                    + (kk + 2) * a_stride_1 as usize,
                            )
                        };
                        let a3 = unsafe {
                            *a_ptr.add(
                                bat_a_offset
                                    + i * a_stride_0 as usize
                                    + (kk + 3) * a_stride_1 as usize,
                            )
                        };

                        let b_base = bat_b_offset + j * b_stride_1 as usize;
                        let b0 = unsafe { *b_ptr.add(b_base + kk * b_stride_0 as usize) };
                        let b1 = unsafe { *b_ptr.add(b_base + (kk + 1) * b_stride_0 as usize) };
                        let b2 = unsafe { *b_ptr.add(b_base + (kk + 2) * b_stride_0 as usize) };
                        let b3 = unsafe { *b_ptr.add(b_base + (kk + 3) * b_stride_0 as usize) };

                        sum0 += a0 * b0;
                        sum1 += a1 * b1;
                        sum2 += a2 * b2;
                        sum3 += a3 * b3;
                        kk += 4;
                    }

                    sum = sum0 + sum1 + sum2 + sum3;

                    while kk < k_limit {
                        let a_val = unsafe {
                            *a_ptr.add(
                                bat_a_offset + i * a_stride_0 as usize + kk * a_stride_1 as usize,
                            )
                        };
                        let b_val = unsafe {
                            *b_ptr.add(
                                bat_b_offset + kk * b_stride_0 as usize + j * b_stride_1 as usize,
                            )
                        };
                        sum += a_val * b_val;
                        kk += 1;
                    }
                }

                let out_idx = bat * m as usize * n as usize + i * n as usize + j;
                unsafe { *out_ptr.add(out_idx) = sum };
            }
        }
    }
}

#[inline]
fn single_threaded_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    batch: usize,
    m: i32,
    n: i32,
    k: i32,
    a_batch_stride: i64,
    a_stride_0: i64,
    a_stride_1: i64,
    b_batch_stride: i64,
    b_stride_0: i64,
    b_stride_1: i64,
) {
    const TILE_SIZE: usize = 64;

    for bat in 0..batch {
        let bat_a_offset = bat * a_batch_stride as usize;
        let bat_b_offset = bat * b_batch_stride as usize;

        for i_tile in (0..m as usize).step_by(TILE_SIZE) {
            for j_tile in (0..n as usize).step_by(TILE_SIZE) {
                for k_tile in (0..k as usize).step_by(TILE_SIZE) {
                    let i_max = (i_tile + TILE_SIZE).min(m as usize);
                    let j_max = (j_tile + TILE_SIZE).min(n as usize);
                    let k_max = (k_tile + TILE_SIZE).min(k as usize);

                    for i in i_tile..i_max {
                        for j in j_tile..j_max {
                            let mut sum = 0.0f32;

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                            {
                                if is_x86_feature_detected!("fma") {
                                    unsafe {
                                        let mut acc0 = _mm256_setzero_ps();
                                        let mut acc1 = _mm256_setzero_ps();
                                        let mut kk = k_tile;

                                        while kk + 16 <= k_max {
                                            let a0 = _mm256_loadu_ps(a_ptr.add(
                                                bat_a_offset
                                                    + i * a_stride_0 as usize
                                                    + kk * a_stride_1 as usize,
                                            ));
                                            let a1 = _mm256_loadu_ps(a_ptr.add(
                                                bat_a_offset
                                                    + i * a_stride_0 as usize
                                                    + (kk + 8) * a_stride_1 as usize,
                                            ));

                                            let b0 = _mm256_loadu_ps(b_ptr.add(
                                                bat_b_offset
                                                    + kk * b_stride_0 as usize
                                                    + j * b_stride_1 as usize,
                                            ));
                                            let b1 = _mm256_loadu_ps(b_ptr.add(
                                                bat_b_offset
                                                    + (kk + 8) * b_stride_0 as usize
                                                    + j * b_stride_1 as usize,
                                            ));

                                            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                                            acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                                            kk += 16;
                                        }

                                        // Handle remaining 8 elements with 8-wide loop
                                        while kk + 8 <= k_max {
                                            let a0 = _mm256_loadu_ps(a_ptr.add(
                                                bat_a_offset
                                                    + i * a_stride_0 as usize
                                                    + kk * a_stride_1 as usize,
                                            ));

                                            let b0 = _mm256_loadu_ps(b_ptr.add(
                                                bat_b_offset
                                                    + kk * b_stride_0 as usize
                                                    + j * b_stride_1 as usize,
                                            ));

                                            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                                            kk += 8;
                                        }

                                        let acc = _mm256_add_ps(acc0, acc1);
                                        let acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                                        _mm256_storeu_ps(acc_arr.as_ptr() as *mut f32, acc);
                                        sum += acc_arr.assume_init().iter().sum::<f32>();

                                        // Scalar tail for remaining < 8 elements
                                        while kk < k_max {
                                            let a_val = *a_ptr.add(
                                                bat_a_offset
                                                    + i * a_stride_0 as usize
                                                    + kk * a_stride_1 as usize,
                                            );
                                            let b_val = *b_ptr.add(
                                                bat_b_offset
                                                    + kk * b_stride_0 as usize
                                                    + j * b_stride_1 as usize,
                                            );
                                            sum += a_val * b_val;
                                            kk += 1;
                                        }
                                        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                                        {
                                            for idx in 0..numel {
                                                unsafe {
                                                    *out_ptr.add(idx) =
                                                        *a_ptr.add(idx) / *b_ptr.add(idx);
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    let mut kk = k_tile;
                                    while kk + 4 <= k_max {
                                        let a0 = unsafe {
                                            *a_ptr.add(
                                                bat_a_offset
                                                    + i * a_stride_0 as usize
                                                    + kk * a_stride_1 as usize,
                                            )
                                        };
                                        let a1 = unsafe {
                                            *a_ptr.add(
                                                bat_a_offset
                                                    + i * a_stride_0 as usize
                                                    + (kk + 1) * a_stride_1 as usize,
                                            )
                                        };
                                        let a2 = unsafe {
                                            *a_ptr.add(
                                                bat_a_offset
                                                    + i * a_stride_0 as usize
                                                    + (kk + 2) * a_stride_1 as usize,
                                            )
                                        };
                                        let a3 = unsafe {
                                            *a_ptr.add(
                                                bat_a_offset
                                                    + i * a_stride_0 as usize
                                                    + (kk + 3) * a_stride_1 as usize,
                                            )
                                        };

                                        let b0 = unsafe {
                                            *b_ptr.add(
                                                bat_b_offset
                                                    + kk * b_stride_0 as usize
                                                    + j * b_stride_1 as usize,
                                            )
                                        };
                                        let b1 = unsafe {
                                            *b_ptr.add(
                                                bat_b_offset
                                                    + (kk + 1) * b_stride_0 as usize
                                                    + j * b_stride_1 as usize,
                                            )
                                        };
                                        let b2 = unsafe {
                                            *b_ptr.add(
                                                bat_b_offset
                                                    + (kk + 2) * b_stride_0 as usize
                                                    + j * b_stride_1 as usize,
                                            )
                                        };
                                        let b3 = unsafe {
                                            *b_ptr.add(
                                                bat_b_offset
                                                    + (kk + 3) * b_stride_0 as usize
                                                    + j * b_stride_1 as usize,
                                            )
                                        };

                                        sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
                                        kk += 4;
                                    }

                                    while kk < k_max {
                                        let a_val = unsafe {
                                            *a_ptr.add(
                                                bat_a_offset
                                                    + i * a_stride_0 as usize
                                                    + kk * a_stride_1 as usize,
                                            )
                                        };
                                        let b_val = unsafe {
                                            *b_ptr.add(
                                                bat_b_offset
                                                    + kk * b_stride_0 as usize
                                                    + j * b_stride_1 as usize,
                                            )
                                        };
                                        sum += a_val * b_val;
                                        kk += 1;
                                    }
                                }
                            }

                            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                            {
                                let mut kk = k_tile;
                                while kk + 4 <= k_max {
                                    let a0 = unsafe {
                                        *a_ptr.add(
                                            bat_a_offset
                                                + i * a_stride_0 as usize
                                                + kk * a_stride_1 as usize,
                                        )
                                    };
                                    let a1 = unsafe {
                                        *a_ptr.add(
                                            bat_a_offset
                                                + i * a_stride_0 as usize
                                                + (kk + 1) * a_stride_1 as usize,
                                        )
                                    };
                                    let a2 = unsafe {
                                        *a_ptr.add(
                                            bat_a_offset
                                                + i * a_stride_0 as usize
                                                + (kk + 2) * a_stride_1 as usize,
                                        )
                                    };
                                    let a3 = unsafe {
                                        *a_ptr.add(
                                            bat_a_offset
                                                + i * a_stride_0 as usize
                                                + (kk + 3) * a_stride_1 as usize,
                                        )
                                    };

                                    let b0 = unsafe {
                                        *b_ptr.add(
                                            bat_b_offset
                                                + kk * b_stride_0 as usize
                                                + j * b_stride_1 as usize,
                                        )
                                    };
                                    let b1 = unsafe {
                                        *b_ptr.add(
                                            bat_b_offset
                                                + (kk + 1) * b_stride_0 as usize
                                                + j * b_stride_1 as usize,
                                        )
                                    };
                                    let b2 = unsafe {
                                        *b_ptr.add(
                                            bat_b_offset
                                                + (kk + 2) * b_stride_0 as usize
                                                + j * b_stride_1 as usize,
                                        )
                                    };
                                    let b3 = unsafe {
                                        *b_ptr.add(
                                            bat_b_offset
                                                + (kk + 3) * b_stride_0 as usize
                                                + j * b_stride_1 as usize,
                                        )
                                    };

                                    sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
                                    kk += 4;
                                }

                                while kk < k_max {
                                    let a_val = unsafe {
                                        *a_ptr.add(
                                            bat_a_offset
                                                + i * a_stride_0 as usize
                                                + kk * a_stride_1 as usize,
                                        )
                                    };
                                    let b_val = unsafe {
                                        *b_ptr.add(
                                            bat_b_offset
                                                + kk * b_stride_0 as usize
                                                + j * b_stride_1 as usize,
                                        )
                                    };
                                    sum += a_val * b_val;
                                    kk += 1;
                                }
                            }

                            let out_idx = bat * m as usize * n as usize + i * n as usize + j;
                            unsafe { *out_ptr.add(out_idx) = sum };
                        }
                    }
                }
            }
        }
    }
}

fn linear_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 { Some(args[2]) } else { None };

    let x_shape = x.shape();
    let w_shape = w.shape();

    let batch_size: i64 = if x_shape.len() > 1 {
        x_shape[..x_shape.len() - 1].iter().product()
    } else {
        1
    };
    let in_features = x_shape[x_shape.len() - 1];
    let out_features = w_shape[0];

    let x_flat = x.reshape(vec![batch_size, in_features]);

    // w has shape [out_features, in_features]
    // We need to compute x_flat @ w.T (where w.T has shape [in_features, out_features])
    // The matmul kernel will detect that w is transposed by checking its strides
    // and use the appropriate BLAS transpose flag
    let w_t = w.transpose(0, 1);

    let mut result = (x_flat.matmul(&w_t)).reshape(
        x_shape[..x_shape.len() - 1]
            .iter()
            .copied()
            .chain(std::iter::once(out_features))
            .collect(),
    );

    if let Some(b) = bias {
        result = result.add(b);
    }

    vec![result]
}

fn fused_linear_relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 { Some(args[2]) } else { None };

    let x_shape = x.shape();
    let w_shape = w.shape();

    let batch_size: i64 = if x_shape.len() > 1 {
        x_shape[..x_shape.len() - 1].iter().product()
    } else {
        1
    };
    let in_features = x_shape[x_shape.len() - 1];
    let out_features = w_shape[0];

    let x_ptr = x.data_ptr_f32();
    let w_ptr = w.data_ptr_f32();
    debug_assert!(
        w.is_contiguous(),
        "fused_linear_relu: weight tensor must be contiguous"
    );

    let output_shape: Vec<i64> = if x_shape.len() > 1 {
        let mut s = x_shape[..x_shape.len() - 1].to_vec();
        s.push(out_features);
        s
    } else {
        vec![out_features]
    };

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    let batch_size = batch_size as usize;
    let in_features = in_features as usize;
    let out_features = out_features as usize;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let total = batch_size * out_features;

        let x_usize = x_ptr as usize;
        let w_usize = w_ptr as usize;
        let out_usize = out_ptr as usize;

        (0..total).into_par_iter().for_each(|idx| {
            let batch_idx = idx / out_features;
            let out_idx = idx % out_features;

            let mut sum = 0.0f32;
            for k in 0..in_features {
                let x_offset = batch_idx * in_features + k;
                let w_offset = out_idx * in_features + k;
                let x_val = unsafe { *((x_usize + x_offset * 4) as *const f32) };
                let w_val = unsafe { *((w_usize + w_offset * 4) as *const f32) };
                sum += x_val * w_val;
            }

            if let Some(b) = bias {
                let b_ptr = b.data_ptr_f32();
                sum += unsafe { *b_ptr.add(out_idx) };
            }

            unsafe {
                *((out_usize + idx * 4) as *mut f32) = sum.max(0.0);
            };
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for batch_idx in 0..batch_size {
            for out_idx in 0..out_features {
                let mut sum = 0.0f32;
                for k in 0..in_features {
                    let x_offset = batch_idx * in_features + k;
                    let w_offset = out_idx * in_features + k;
                    let x_val = unsafe { *x_ptr.add(x_offset) };
                    let w_val = unsafe { *w_ptr.add(w_offset) };
                    sum += x_val * w_val;
                }

                if let Some(b) = bias {
                    let b_ptr = b.data_ptr_f32();
                    sum += unsafe { *b_ptr.add(out_idx) };
                }

                unsafe {
                    *out_ptr.add(batch_idx * out_features + out_idx) = sum.max(0.0);
                };
            }
        }
    }

    vec![output]
}

fn fused_linear_silu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 { Some(args[2]) } else { None };

    let x_shape = x.shape();
    let w_shape = w.shape();

    let batch_size: i64 = if x_shape.len() > 1 {
        x_shape[..x_shape.len() - 1].iter().product()
    } else {
        1
    };
    let in_features = x_shape[x_shape.len() - 1];
    let out_features = w_shape[0];

    let x_ptr = x.data_ptr_f32();
    let w_ptr = w.data_ptr_f32();
    debug_assert!(
        w.is_contiguous(),
        "fused_linear_silu: weight tensor must be contiguous"
    );

    let output_shape: Vec<i64> = if x_shape.len() > 1 {
        let mut s = x_shape[..x_shape.len() - 1].to_vec();
        s.push(out_features);
        s
    } else {
        vec![out_features]
    };

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    let batch_size = batch_size as usize;
    let in_features = in_features as usize;
    let out_features = out_features as usize;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let total = batch_size * out_features;

        let x_usize = x_ptr as usize;
        let w_usize = w_ptr as usize;
        let out_usize = out_ptr as usize;

        (0..total).into_par_iter().for_each(|idx| {
            let batch_idx = idx / out_features;
            let out_idx = idx % out_features;

            let mut sum = 0.0f32;
            for k in 0..in_features {
                let x_offset = batch_idx * in_features + k;
                let w_offset = out_idx * in_features + k;
                let x_val = unsafe { *((x_usize + x_offset * 4) as *const f32) };
                let w_val = unsafe { *((w_usize + w_offset * 4) as *const f32) };
                sum += x_val * w_val;
            }

            if let Some(b) = bias {
                let b_ptr = b.data_ptr_f32();
                sum += unsafe { *b_ptr.add(out_idx) };
            }

            // SiLU: x / (1.0 + (-x).exp())
            let silu_val = sum / (1.0 + (-sum).exp());
            unsafe {
                *((out_usize + idx * 4) as *mut f32) = silu_val;
            };
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for batch_idx in 0..batch_size {
            for out_idx in 0..out_features {
                let mut sum = 0.0f32;
                for k in 0..in_features {
                    let x_offset = batch_idx * in_features + k;
                    let w_offset = out_idx * in_features + k;
                    let x_val = unsafe { *x_ptr.add(x_offset) };
                    let w_val = unsafe { *w_ptr.add(w_offset) };
                    sum += x_val * w_val;
                }

                if let Some(b) = bias {
                    let b_ptr = b.data_ptr_f32();
                    sum += unsafe { *b_ptr.add(out_idx) };
                }

                // SiLU: x / (1.0 + (-x).exp())
                let silu_val = sum / (1.0 + (-sum).exp());
                unsafe {
                    *out_ptr.add(batch_idx * out_features + out_idx) = silu_val;
                };
            }
        }
    }

    vec![output]
}

fn fused_linear_gelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 { Some(args[2]) } else { None };

    let x_shape = x.shape();
    let w_shape = w.shape();

    let batch_size: i64 = if x_shape.len() > 1 {
        x_shape[..x_shape.len() - 1].iter().product()
    } else {
        1
    };
    let in_features = x_shape[x_shape.len() - 1];
    let out_features = w_shape[0];

    let x_ptr = x.data_ptr_f32();
    let w_ptr = w.data_ptr_f32();
    debug_assert!(
        w.is_contiguous(),
        "fused_linear_gelu: weight tensor must be contiguous"
    );

    let output_shape: Vec<i64> = if x_shape.len() > 1 {
        let mut s = x_shape[..x_shape.len() - 1].to_vec();
        s.push(out_features);
        s
    } else {
        vec![out_features]
    };

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    let batch_size = batch_size as usize;
    let in_features = in_features as usize;
    let out_features = out_features as usize;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let coeff = 0.044715f32;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let total = batch_size * out_features;

        let x_usize = x_ptr as usize;
        let w_usize = w_ptr as usize;
        let out_usize = out_ptr as usize;

        (0..total).into_par_iter().for_each(|idx| {
            let batch_idx = idx / out_features;
            let out_idx = idx % out_features;

            let mut sum = 0.0f32;
            for k in 0..in_features {
                let x_offset = batch_idx * in_features + k;
                let w_offset = out_idx * in_features + k;
                let x_val = unsafe { *((x_usize + x_offset * 4) as *const f32) };
                let w_val = unsafe { *((w_usize + w_offset * 4) as *const f32) };
                sum += x_val * w_val;
            }

            if let Some(b) = bias {
                let b_ptr = b.data_ptr_f32();
                sum += unsafe { *b_ptr.add(out_idx) };
            }

            let x3 = sum * sum * sum;
            let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
            let gelu = 0.5 * sum * (1.0 + t);

            unsafe {
                *((out_usize + idx * 4) as *mut f32) = gelu;
            };
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for batch_idx in 0..batch_size {
            for out_idx in 0..out_features {
                let mut sum = 0.0f32;
                for k in 0..in_features {
                    let x_offset = batch_idx * in_features + k;
                    let w_offset = out_idx * in_features + k;
                    let x_val = unsafe { *x_ptr.add(x_offset) };
                    let w_val = unsafe { *w_ptr.add(w_offset) };
                    sum += x_val * w_val;
                }

                if let Some(b) = bias {
                    let b_ptr = b.data_ptr_f32();
                    sum += unsafe { *b_ptr.add(out_idx) };
                }

                let x3 = sum * sum * sum;
                let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                let gelu = 0.5 * sum * (1.0 + t);

                unsafe {
                    *out_ptr.add(batch_idx * out_features + out_idx) = gelu;
                };
            }
        }
    }

    vec![output]
}

fn sum_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let a_shape = a.shape();
        let ndim = a_shape.len() as i32;
        // Handle negative dimensions
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();

    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.clone();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    if output_shape.is_empty() {
        output_shape = vec![1];
    }

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let a_ptr = a.data_ptr() as *const f32;
    let a_storage_offset = a.inner.storage_offset as usize;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    let dim_size = a_shape[dim] as usize;
    let mut strides_before = 1i64;
    let mut strides_after = 1i64;

    for i in 0..dim {
        strides_before *= a_shape[i];
    }
    for i in (dim + 1)..ndim {
        strides_after *= a_shape[i];
    }

    let total_blocks = (strides_before * strides_after) as usize;
    let a_numel = a.numel() as usize;

    #[cfg(feature = "parallel")]
    {
        if total_blocks > 256 && dim_size > 32 {
            let a_slice: &[f32] = unsafe { std::slice::from_raw_parts(a_ptr, a_numel) };
            let results: Vec<f32> = (0..total_blocks)
                .into_par_iter()
                .map(|block| {
                    let block_before = block / (strides_after as usize);
                    let block_after = block % (strides_after as usize);

                    let mut sum_val = 0.0f32;
                    for d in 0..dim_size {
                        let linear_idx =
                            (block_before * dim_size + d) * strides_after as usize + block_after;
                        let idx = a_storage_offset + linear_idx;
                        if idx < a_numel {
                            sum_val += a_slice[idx];
                        }
                    }
                    sum_val
                })
                .collect();

            for (i, &sum_val) in results.iter().enumerate() {
                unsafe {
                    *out_ptr.add(i) = sum_val;
                }
            }
            return vec![output];
        }
    }

    for block in 0..total_blocks {
        let block_before = block / (strides_after as usize);
        let block_after = block % (strides_after as usize);

        let mut sum_val = 0.0f32;
        for d in 0..dim_size {
            let linear_idx = (block_before * dim_size + d) * strides_after as usize + block_after;
            let idx = a_storage_offset + linear_idx;
            if idx < a_numel {
                unsafe {
                    sum_val += *a_ptr.add(idx);
                }
            }
        }

        unsafe {
            *out_ptr.add(block) = sum_val;
        }
    }

    vec![output]
}

fn min_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let a_shape = a.shape();
        let ndim = a_shape.len() as i32;
        // Handle negative dimensions
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.clone();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    if output_shape.is_empty() {
        output_shape = vec![1];
    }

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let a_ptr = a.data_ptr() as *const f32;
    let a_storage_offset = a.inner.storage_offset as usize;
    let a_numel = a.numel() as usize;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    let dim_size = a_shape[dim] as usize;
    let mut strides_before = 1i64;
    let mut strides_after = 1i64;

    for i in 0..dim {
        strides_before *= a_shape[i];
    }
    for i in (dim + 1)..ndim {
        strides_after *= a_shape[i];
    }

    let total_blocks = (strides_before * strides_after) as usize;

    #[cfg(feature = "parallel")]
    {
        if total_blocks > 256 && dim_size > 32 {
            let a_slice: &[f32] = unsafe { std::slice::from_raw_parts(a_ptr, a_numel) };
            let results: Vec<f32> = (0..total_blocks)
                .into_par_iter()
                .map(|block| {
                    let block_before = block / (strides_after as usize);
                    let block_after = block % (strides_after as usize);

                    let mut min_val = f32::MAX;
                    for d in 0..dim_size {
                        let linear_idx =
                            (block_before * dim_size + d) * strides_after as usize + block_after;
                        let idx = a_storage_offset + linear_idx;
                        if idx < a_numel {
                            min_val = min_val.min(a_slice[idx]);
                        }
                    }
                    min_val
                })
                .collect();

            for (i, &min_val) in results.iter().enumerate() {
                unsafe {
                    *out_ptr.add(i) = min_val;
                }
            }
            return vec![output];
        }
    }

    for block in 0..total_blocks {
        let block_before = block / (strides_after as usize);
        let block_after = block % (strides_after as usize);

        let mut min_val = f32::MAX;
        for d in 0..dim_size {
            let linear_idx = (block_before * dim_size + d) * strides_after as usize + block_after;
            let idx = a_storage_offset + linear_idx;
            if idx < a_numel {
                unsafe {
                    min_val = min_val.min(*a_ptr.add(idx));
                }
            }
        }

        unsafe {
            *out_ptr.add(block) = min_val;
        }
    }

    vec![output]
}

fn mean_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let a_shape = a.shape();
        let ndim = a_shape.len() as i32;
        // Handle negative dimensions
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let dim_size = a_shape[dim] as f32;

    let sum_result = sum_kernel(args);
    let mut sum_tensor = sum_result[0].clone();

    // sum_kernel already handles keepdim correctly
    // If keepdim is true, sum_tensor has shape [..., 1, ...]
    // If keepdim is false, sum_tensor has shape with the dimension removed
    // We need to unsqueeze for broadcasting with the input, then squeeze back if needed
    let needs_unsqueeze = !keepdim;
    if needs_unsqueeze {
        sum_tensor = sum_tensor.unsqueeze(dim);
    }

    let scale = Tensor::full(vec![], 1.0 / dim_size, DType::F32, Device::Cpu);
    let result = sum_tensor * scale;

    // If we unsqueezed for broadcasting, remove it now
    let result = if needs_unsqueeze {
        result.squeeze(Some(dim))
    } else {
        result
    };

    vec![result]
}

fn max_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let a_shape = a.shape();
        let ndim = a_shape.len() as i32;
        // Handle negative dimensions
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.clone();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    if output_shape.is_empty() {
        output_shape = vec![1];
    }

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let a_ptr = a.data_ptr() as *const f32;
    let a_storage_offset = a.inner.storage_offset as usize;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    let dim_size = a_shape[dim] as usize;
    let mut strides_before = 1i64;
    let mut strides_after = 1i64;

    for i in 0..dim {
        strides_before *= a_shape[i];
    }
    for i in (dim + 1)..ndim {
        strides_after *= a_shape[i];
    }

    for block in 0..(strides_before as usize * strides_after as usize) {
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..dim_size {
            let a_idx = (block / (strides_after as usize)) * dim_size * (strides_after as usize)
                + i * (strides_after as usize)
                + block % (strides_after as usize);
            unsafe {
                let val = *a_ptr.add(a_idx + a_storage_offset);
                if val > max_val {
                    max_val = val;
                }
            }
        }
        unsafe {
            *out_ptr.add(block) = max_val;
        }
    }

    vec![output]
}

fn argmax_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let a_shape = a.shape();
        let ndim = a_shape.len() as i32;
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.clone();
    output_shape[dim] = 1;

    let mut output = Tensor::zeros(output_shape.clone(), DType::I32, a.device());

    let a_ptr = a.data_ptr() as *const f32;
    let a_storage_offset = a.inner.storage_offset as usize;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut i32;

    let dim_size = a_shape[dim] as usize;
    let mut strides_before = 1i64;
    let mut strides_after = 1i64;

    for i in 0..dim {
        strides_before *= a_shape[i];
    }
    for i in (dim + 1)..ndim {
        strides_after *= a_shape[i];
    }

    for block in 0..(strides_before as usize * strides_after as usize) {
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0usize;
        for i in 0..dim_size {
            let a_idx = (block / (strides_after as usize)) * dim_size * (strides_after as usize)
                + i * (strides_after as usize)
                + block % (strides_after as usize);
            unsafe {
                let val = *a_ptr.add(a_idx + a_storage_offset);
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
        }
        unsafe {
            *out_ptr.add(block) = max_idx as i32;
        }
    }

    vec![output]
}

fn argmin_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let a_shape = a.shape();
        let ndim = a_shape.len() as i32;
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.clone();
    output_shape[dim] = 1;

    let mut output = Tensor::zeros(output_shape.clone(), DType::I32, a.device());

    let a_ptr = a.data_ptr() as *const f32;
    let a_storage_offset = a.inner.storage_offset as usize;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut i32;

    let dim_size = a_shape[dim] as usize;
    let mut strides_before = 1i64;
    let mut strides_after = 1i64;

    for i in 0..dim {
        strides_before *= a_shape[i];
    }
    for i in (dim + 1)..ndim {
        strides_after *= a_shape[i];
    }

    for block in 0..(strides_before as usize * strides_after as usize) {
        let mut min_val = f32::INFINITY;
        let mut min_idx = 0usize;
        for i in 0..dim_size {
            let a_idx = (block / (strides_after as usize)) * dim_size * (strides_after as usize)
                + i * (strides_after as usize)
                + block % (strides_after as usize);
            unsafe {
                let val = *a_ptr.add(a_idx + a_storage_offset);
                if val < min_val {
                    min_val = val;
                    min_idx = i;
                }
            }
        }
        unsafe {
            *out_ptr.add(block) = min_idx as i32;
        }
    }

    vec![output]
}

fn softmax_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let x_shape = x.shape();
        let ndim = x_shape.len() as i32;
        // Handle negative dimensions
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };

    let x_shape = x.shape();
    let ndim = x_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let dim_size = x_shape[dim] as usize;

    if dim == ndim - 1 && x.is_contiguous() && dim_size > 32 {
        return vec![softmax_last_dim_simd(x, dim_size)];
    }

    let max_vals = max_kernel(&[
        x,
        &Tensor::from_scalar(dim as f32),
        &Tensor::from_scalar(1.0),
    ])[0]
        .clone();
    // max_kernel with keepdim=true already keeps the dimension, so no need to unsqueeze
    let max_exp = x.sub(&max_vals).exp();
    let sum_exp = max_exp.sum(dim as i32, true);

    vec![max_exp.div(&sum_exp)]
}

#[inline]
fn softmax_last_dim_simd(x: &Tensor, dim_size: usize) -> Tensor {
    let x_shape = x.shape();
    let _batch_size: i64 = x_shape[..x_shape.len() - 1].iter().product();

    let x_ptr = x.data_ptr() as *const f32;
    let numel = x.numel() as usize;

    let mut output = Tensor::zeros(x_shape.to_vec(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    #[cfg(feature = "parallel")]
    {
        let a_slice = unsafe { std::slice::from_raw_parts(x_ptr, numel) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

        out_slice
            .par_chunks_mut(dim_size)
            .zip(a_slice.par_chunks(dim_size))
            .for_each(|(out_row, x_row)| {
                let mut max_val = f32::MIN;

                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            let mut max_vec = _mm256_set1_ps(f32::MIN);
                            let mut i = 0;
                            while i + 8 <= dim_size {
                                let vec = _mm256_loadu_ps(x_row.as_ptr().add(i));
                                max_vec = _mm256_max_ps(max_vec, vec);
                                i += 8;
                            }
                            let mut max_arr = [0.0f32; 8];
                            _mm256_storeu_ps(max_arr.as_mut_ptr(), max_vec);
                            for j in 0..8 {
                                max_val = max_val.max(max_arr[j]);
                            }
                            // Handle remaining elements
                            for j in i..dim_size {
                                max_val = max_val.max(x_row[j]);
                            }
                        }
                    } else {
                        for j in 0..dim_size {
                            max_val = max_val.max(x_row[j]);
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for j in 0..dim_size {
                        max_val = max_val.max(x_row[j]);
                    }
                }

                let mut sum_exp = 0.0f32;

                for j in 0..dim_size {
                    sum_exp += (x_row[j] - max_val).exp();
                }

                let inv_sum = 1.0 / sum_exp;

                for j in 0..dim_size {
                    out_row[j] = (x_row[j] - max_val).exp() * inv_sum;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let num_rows = numel / dim_size;
        let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, numel) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

        for row in 0..num_rows {
            let row_start = row * dim_size;
            let x_row = &x_slice[row_start..row_start + dim_size];
            let out_row = &mut out_slice[row_start..row_start + dim_size];

            let mut max_val = f32::MIN;
            for j in 0..dim_size {
                max_val = max_val.max(x_row[j]);
            }

            let mut sum_exp = 0.0f32;
            for j in 0..dim_size {
                sum_exp += (x_row[j] - max_val).exp();
            }

            let inv_sum = 1.0 / sum_exp;
            for j in 0..dim_size {
                out_row[j] = (x_row[j] - max_val).exp() * inv_sum;
            }
        }
    }

    output
}

fn log_softmax_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let x_shape = x.shape();
        let ndim = x_shape.len() as i32;
        // Handle negative dimensions
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };

    let x_shape = x.shape();
    let ndim = x_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let max_vals = max_kernel(&[
        x,
        &Tensor::from_scalar(dim as f32),
        &Tensor::from_scalar(1.0),
    ])[0]
        .clone();
    let x_shifted = x.sub(&max_vals.clone().unsqueeze(dim));
    let log_sum_exp = x_shifted.exp().sum(dim as i32, true).ln();

    vec![x_shifted.sub(&log_sum_exp)]
}

fn mse_loss_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let pred = args[0];
    let target = args[1];
    let reduction = if args.len() > 2 {
        match args[2].item() as i32 {
            0 => "none",
            1 => "mean",
            _ => "sum",
        }
    } else {
        "mean"
    };

    let diff = pred.sub(target);
    let loss = diff.mul(&diff);

    match reduction {
        "none" => vec![loss],
        "mean" => vec![loss
            .sum(0, false)
            .div(&Tensor::from_scalar(loss.numel() as f32))],
        "sum" => vec![loss.sum(0, false)],
        _ => vec![loss.sum(0, false)],
    }
}

fn cross_entropy_loss_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let logits = args[0];
    let targets = args[1];
    let reduction = if args.len() > 2 {
        match args[2].item() as i32 {
            0 => "none",
            1 => "mean",
            _ => "sum",
        }
    } else {
        "mean"
    };

    let logits_data = logits.as_f32_slice();
    let targets_data = targets.as_f32_slice();

    let batch_size = logits.shape()[0] as usize;
    let num_classes = logits.shape()[1] as usize;

    let mut total_loss = 0.0f32;
    let mut losses = vec![0.0f32; batch_size];

    for b in 0..batch_size {
        let base_idx = b * num_classes;

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            use std::arch::x86_64::*;
            let mut max_logit = f32::MIN;
            let mut i = 0;

            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let ptr = logits_data.as_ptr().add(base_idx);
                    let mut max_vec = _mm256_set1_ps(f32::MIN);
                    for _ in 0..(num_classes / 8) {
                        let vec = _mm256_loadu_ps(ptr.add(i));
                        max_vec = _mm256_max_ps(max_vec, vec);
                        i += 8;
                    }
                    let mut max_arr = [0.0f32; 8];
                    _mm256_storeu_ps(max_arr.as_mut_ptr(), max_vec);
                    for j in 0..8 {
                        if max_arr[j] > max_logit {
                            max_logit = max_arr[j];
                        }
                    }
                }
            }
            for j in i..num_classes {
                let val = logits_data[base_idx + j];
                if val > max_logit {
                    max_logit = val;
                }
            }

            let mut sum_exp = 0.0f32;
            i = 0;
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let ptr = logits_data.as_ptr().add(base_idx);
                    let mut sum_vec = _mm256_setzero_ps();
                    for _ in 0..(num_classes / 8) {
                        let vec = _mm256_loadu_ps(ptr.add(i));
                        let diff = _mm256_sub_ps(vec, _mm256_set1_ps(max_logit));
                        let x = std::slice::from_raw_parts(&diff as *const _ as *const f32, 8);
                        let mut exp_res = [0.0f32; 8];
                        for k in 0..8 {
                            exp_res[k] = x[k].exp();
                        }
                        let exp_vec = _mm256_loadu_ps(exp_res.as_ptr());
                        sum_vec = _mm256_add_ps(sum_vec, exp_vec);
                        i += 8;
                    }
                    let mut sum_arr = [0.0f32; 8];
                    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
                    for j in 0..8 {
                        sum_exp += sum_arr[j];
                    }
                }
            }
            for j in i..num_classes {
                sum_exp += (logits_data[base_idx + j] - max_logit).exp();
            }
            let log_sum_exp = sum_exp.ln();

            let target_class = targets_data[b] as usize;
            let class_logit = logits_data[base_idx + target_class];

            // Add max_logit back (subtracted before exp for numerical stability)
            losses[b] = log_sum_exp + max_logit - class_logit;
            total_loss += losses[b];
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            let mut max_logit = f32::MIN;
            for c in 0..num_classes {
                let val = logits_data[base_idx + c];
                if val > max_logit {
                    max_logit = val;
                }
            }

            let mut sum_exp = 0.0f32;
            for c in 0..num_classes {
                sum_exp += (logits_data[base_idx + c] - max_logit).exp();
            }
            let log_sum_exp = sum_exp.ln();

            let target_class = targets_data[b] as usize;
            let class_logit = logits_data[base_idx + target_class];

            // Add max_logit back (subtracted before exp for numerical stability)
            losses[b] = log_sum_exp + max_logit - class_logit;
            total_loss += losses[b];
        }
    }

    match reduction {
        "none" => {
            let output = Tensor::from_vec(losses, vec![batch_size as i64]);
            vec![output]
        }
        "mean" => vec![Tensor::from_scalar(total_loss / batch_size as f32)],
        "sum" | _ => vec![Tensor::from_scalar(total_loss)],
    }
}

#[allow(dead_code)]
fn im2col_kernel(
    x: &Tensor,
    kernel_height: usize,
    kernel_width: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_height: usize,
    out_width: usize,
) -> Tensor {
    let x_shape = x.shape();
    let batch_size = x_shape[0] as usize;
    let in_channels = x_shape[1] as usize;
    let in_height = x_shape[2] as usize;
    let in_width = x_shape[3] as usize;

    let col_rows = batch_size * out_height * out_width;
    let col_cols = in_channels * kernel_height * kernel_width;
    let mut col_data = vec![0.0f32; col_rows * col_cols];

    let x_ptr = x.data_ptr() as *const f32;

    let _in_height_pad = in_height + 2 * padding;
    let _in_width_pad = in_width + 2 * padding;

    for n in 0..batch_size {
        for oh in 0..out_height {
            for ow in 0..out_width {
                let col_row = (n * out_height + oh) * out_width + ow;

                for ic in 0..in_channels {
                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            let ih = oh * stride + kh * dilation;
                            let iw = ow * stride + kw * dilation;

                            let col_col = ((ic * kernel_height) + kh) * kernel_width + kw;

                            if ih >= padding
                                && ih < padding + in_height
                                && iw >= padding
                                && iw < padding + in_width
                            {
                                let x_ih = ih - padding;
                                let x_iw = iw - padding;
                                let x_idx =
                                    ((n * in_channels + ic) * in_height + x_ih) * in_width + x_iw;
                                col_data[col_row * col_cols + col_col] =
                                    unsafe { *x_ptr.add(x_idx) };
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::from_vec(col_data, vec![col_rows as i64, col_cols as i64])
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn simd_dot_product(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;

    #[cfg(feature = "simd_avx512")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                let mut acc0 = _mm512_setzero_ps();
                let mut acc1 = _mm512_setzero_ps();
                let mut acc2 = _mm512_setzero_ps();
                let mut acc3 = _mm512_setzero_ps();

                while i + 64 <= len {
                    let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
                    let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
                    let a1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
                    let b1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
                    let a2 = _mm512_loadu_ps(a.as_ptr().add(i + 32));
                    let b2 = _mm512_loadu_ps(b.as_ptr().add(i + 32));
                    let a3 = _mm512_loadu_ps(a.as_ptr().add(i + 48));
                    let b3 = _mm512_loadu_ps(b.as_ptr().add(i + 48));

                    acc0 = _mm512_fmadd_ps(a0, b0, acc0);
                    acc1 = _mm512_fmadd_ps(a1, b1, acc1);
                    acc2 = _mm512_fmadd_ps(a2, b2, acc2);
                    acc3 = _mm512_fmadd_ps(a3, b3, acc3);

                    i += 64;
                }

                let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
                sum += _mm512_reduce_add_ps(acc);

                while i + 16 <= len {
                    let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
                    acc0 = _mm512_fmadd_ps(a_vec, b_vec, _mm512_setzero_ps());
                    sum += _mm512_reduce_add_ps(acc0);
                    i += 16;
                }
            }
        } else if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();

                while i + 32 <= len {
                    let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
                    let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
                    let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
                    let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
                    let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
                    let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
                    let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));

                    acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                    acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                    acc2 = _mm256_fmadd_ps(a2, b2, acc2);
                    acc3 = _mm256_fmadd_ps(a3, b3, acc3);

                    i += 32;
                }

                let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                let acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                _mm256_storeu_ps(acc_arr.as_ptr() as *mut f32, acc);
                sum += acc_arr.assume_init().iter().sum::<f32>();

                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                    let prod = _mm256_mul_ps(a_vec, b_vec);
                    let prod_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                    _mm256_storeu_ps(prod_arr.as_ptr() as *mut f32, prod);
                    sum += prod_arr.assume_init().iter().sum::<f32>();
                    i += 8;
                }
            }
        }
    }

    #[cfg(not(feature = "simd_avx512"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();

                while i + 32 <= len {
                    let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
                    let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
                    let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
                    let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
                    let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
                    let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
                    let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));

                    acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                    acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                    acc2 = _mm256_fmadd_ps(a2, b2, acc2);
                    acc3 = _mm256_fmadd_ps(a3, b3, acc3);

                    i += 32;
                }

                let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                let acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                _mm256_storeu_ps(acc_arr.as_ptr() as *mut f32, acc);
                sum += acc_arr.assume_init().iter().sum::<f32>();

                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                    let prod = _mm256_mul_ps(a_vec, b_vec);
                    let prod_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                    _mm256_storeu_ps(prod_arr.as_ptr() as *mut f32, prod);
                    sum += prod_arr.assume_init().iter().sum::<f32>();
                    i += 8;
                }
            }
        }
    }

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
#[inline]
fn simd_dot_product(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + 4 <= len {
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

fn conv2d_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 && args[2].numel() > 0 {
        Some(args[2])
    } else {
        None
    };
    let stride = if args.len() > 3 {
        args[3].item() as i64
    } else {
        1
    };
    let padding = if args.len() > 4 {
        args[4].item() as i64
    } else {
        0
    };
    let dilation = if args.len() > 5 {
        args[5].item() as i64
    } else {
        1
    };
    let groups = if args.len() > 6 {
        args[6].item() as i64
    } else {
        1
    };

    let x_shape = x.shape();
    let w_shape = w.shape();

    let batch_size = x_shape[0] as usize;
    let in_channels = x_shape[1] as usize;
    let in_height = x_shape[2] as usize;
    let in_width = x_shape[3] as usize;

    let out_channels = w_shape[0] as usize;
    let kernel_height = w_shape[2] as usize;
    let kernel_width = w_shape[3] as usize;

    let stride = stride as usize;
    let padding = padding as usize;
    let dilation = dilation as usize;
    let groups = groups as usize;

    let out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    let out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    if groups > 1 && groups == in_channels && groups == out_channels {
        return vec![depthwise_conv2d(
            x, w, bias, stride, padding, dilation, out_height, out_width,
        )];
    }

    if kernel_height == 1
        && kernel_width == 1
        && stride == 1
        && padding == 0
        && dilation == 1
        && groups == 1
    {
        return vec![conv2d_1x1(
            x,
            w,
            bias,
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
        )];
    }

    // Use im2col for all 3x3 convolutions
    // Winograd F(2x2,3x3) was implemented but has too much overhead
    // The transforms add latency that exceeds the arithmetic savings
    if kernel_height == 3 && kernel_width == 3 && stride == 1 && dilation == 1 && groups == 1 {
        return vec![conv2d_im2col(
            x,
            w,
            bias,
            stride,
            padding,
            dilation,
            out_height,
            out_width,
            batch_size,
            in_channels,
            out_channels,
            kernel_height,
            kernel_width,
            groups,
        )];
    }

    vec![conv2d_im2col(
        x,
        w,
        bias,
        stride,
        padding,
        dilation,
        out_height,
        out_width,
        batch_size,
        in_channels,
        out_channels,
        kernel_height,
        kernel_width,
        groups,
    )]
}

fn conv2d_1x1(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    in_height: usize,
    in_width: usize,
) -> Tensor {
    let x_ptr = x.data_ptr() as *const f32;
    let w_ptr = w.data_ptr() as *const f32;

    let output_shape = vec![
        batch_size as i64,
        out_channels as i64,
        in_height as i64,
        in_width as i64,
    ];
    let mut output = Tensor::zeros(output_shape.clone(), x.dtype(), x.device());

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    let w_data: Vec<f32> =
        unsafe { std::slice::from_raw_parts(w_ptr, out_channels * in_channels).to_vec() };

    let bias_data: Option<Vec<f32>> = bias.map(|b| {
        let b_ptr = b.data_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(b_ptr, out_channels).to_vec() }
    });

    let _n = batch_size * in_height * in_width;
    let _k = in_channels;
    let _m = out_channels;

    for b in 0..batch_size {
        for h in 0..in_height {
            for w_idx in 0..in_width {
                let row = (b * in_height + h) * in_width + w_idx;
                let x_row = unsafe {
                    std::slice::from_raw_parts(x_ptr.add(row * in_channels), in_channels)
                };

                for oc in 0..out_channels {
                    let w_row = &w_data[oc * in_channels..];
                    let sum = simd_dot_product(x_row, w_row, in_channels);
                    let bias_val = bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0);

                    let out_idx = ((b * out_channels + oc) * in_height + h) * in_width + w_idx;
                    unsafe { *out_ptr.add(out_idx) = sum + bias_val };
                }
            }
        }
    }

    output
}

fn depthwise_conv2d(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    _padding: usize,
    dilation: usize,
    out_height: usize,
    out_width: usize,
) -> Tensor {
    let x_shape = x.shape();
    let batch_size = x_shape[0] as usize;
    let in_channels = x_shape[1] as usize;
    let in_height = x_shape[2] as usize;
    let in_width = x_shape[3] as usize;

    let w_shape = w.shape();
    let kernel_height = w_shape[2] as usize;
    let kernel_width = w_shape[3] as usize;

    let output_shape = vec![
        batch_size as i64,
        in_channels as i64,
        out_height as i64,
        out_width as i64,
    ];
    let mut output = Tensor::zeros(output_shape.clone(), x.dtype(), x.device());

    let x_ptr = x.data_ptr() as *const f32;
    let w_ptr = w.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    // Use direct pointers instead of copying data
    let w_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(w_ptr, in_channels * kernel_height * kernel_width).to_vec()
    };

    let x_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(x_ptr, batch_size * in_channels * in_height * in_width).to_vec()
    };

    let bias_data: Option<Vec<f32>> = bias.map(|b| {
        let ptr = b.data_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, in_channels).to_vec() }
    });

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let total = batch_size * in_channels * out_height * out_width;

        let results: Vec<f32> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let n = idx / (in_channels * out_height * out_width);
                let rem = idx % (in_channels * out_height * out_width);
                let ic = rem / (out_height * out_width);
                let rem2 = rem % (out_height * out_width);
                let oh = rem2 / out_width;
                let ow = rem2 % out_width;

                let mut sum = 0.0f32;

                for kh in 0..kernel_height {
                    for kw in 0..kernel_width {
                        let ih = oh * stride + kh * dilation;
                        let iw = ow * stride + kw * dilation;

                        if ih < in_height && iw < in_width {
                            let x_idx = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                            let w_idx = ((ic * kernel_height) + kh) * kernel_width + kw;
                            sum += x_data[x_idx] * w_data[w_idx];
                        }
                    }
                }

                if let Some(ref b) = bias_data {
                    sum += b[ic];
                }

                sum
            })
            .collect();

        unsafe {
            std::ptr::copy_nonoverlapping(results.as_ptr(), out_ptr, total);
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        for n in 0..batch_size {
            for ic in 0..in_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0f32;

                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride + kh * dilation;
                                let iw = ow * stride + kw * dilation;

                                if ih < in_height && iw < in_width {
                                    let x_idx =
                                        ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                                    let w_idx = ((ic * kernel_height) + kh) * kernel_width + kw;
                                    sum += unsafe { *x_ptr.add(x_idx) } * w_data[w_idx];
                                }
                            }
                        }

                        if let Some(ref b) = bias_data {
                            sum += b[ic];
                        }

                        let out_idx = ((n * in_channels + ic) * out_height + oh) * out_width + ow;
                        unsafe { *out_ptr.add(out_idx) = sum };
                    }
                }
            }
        }
    }

    output
}

fn conv2d_im2col(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_height: usize,
    out_width: usize,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_height: usize,
    kernel_width: usize,
    groups: usize,
) -> Tensor {
    let x_shape = x.shape();
    let in_height = x_shape[2] as usize;
    let in_width = x_shape[3] as usize;

    let col_rows = batch_size * out_height * out_width;
    let col_cols = in_channels * kernel_height * kernel_width;
    let mut col_data = vec![0.0f32; col_rows * col_cols];

    let x_ptr = x.data_ptr() as *const f32;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let x_usize = x_ptr as usize;

        col_data
            .par_chunks_mut(col_cols)
            .enumerate()
            .for_each(|(row, col_chunk)| {
                let n = row / (out_height * out_width);
                let rem = row % (out_height * out_width);
                let oh = rem / out_width;
                let ow = rem % out_width;

                for ic in 0..in_channels {
                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            let ih = oh * stride + kh * dilation;
                            let iw = ow * stride + kw * dilation;

                            let col_col = ((ic * kernel_height) + kh) * kernel_width + kw;

                            if ih >= padding
                                && ih < padding + in_height
                                && iw >= padding
                                && iw < padding + in_width
                            {
                                let x_ih = ih - padding;
                                let x_iw = iw - padding;
                                let x_idx =
                                    ((n * in_channels + ic) * in_height + x_ih) * in_width + x_iw;
                                let x_ptr = x_usize as *const f32;
                                col_chunk[col_col] = unsafe { *x_ptr.add(x_idx) };
                            }
                        }
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for n in 0..batch_size {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let col_row = (n * out_height + oh) * out_width + ow;

                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride + kh * dilation;
                                let iw = ow * stride + kw * dilation;

                                let col_col = ((ic * kernel_height) + kh) * kernel_width + kw;

                                if ih >= padding
                                    && ih < padding + in_height
                                    && iw >= padding
                                    && iw < padding + in_width
                                {
                                    let x_ih = ih - padding;
                                    let x_iw = iw - padding;
                                    let x_idx = ((n * in_channels + ic) * in_height + x_ih)
                                        * in_width
                                        + x_iw;
                                    col_data[col_row * col_cols + col_col] =
                                        unsafe { *x_ptr.add(x_idx) };
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let w_data: Vec<f32> = unsafe {
        let w_ptr = w.data_ptr() as *const f32;
        std::slice::from_raw_parts(
            w_ptr,
            out_channels * in_channels * kernel_height * kernel_width / groups,
        )
        .to_vec()
    };

    let output_shape = vec![
        batch_size as i64,
        out_channels as i64,
        out_height as i64,
        out_width as i64,
    ];
    let mut output = Tensor::zeros(output_shape.clone(), x.dtype(), x.device());

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    // Use the col_data Vec directly as a slice
    let col_slice = col_data.as_slice();

    // Use optimized matrix multiplication for large matrices
    // Threshold: use GEMM when all dimensions >= 12
    const GEMM_MIN_SIZE: usize = 12;
    let use_gemm =
        col_rows >= GEMM_MIN_SIZE && out_channels >= GEMM_MIN_SIZE && col_cols >= GEMM_MIN_SIZE;

    let bias_data: Option<Vec<f32>> = bias.map(|b| {
        let b_ptr = b.data_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(b_ptr, out_channels).to_vec() }
    });

    if use_gemm {
        // Transpose weights for GEMM: (out_channels, col_cols) -> (col_cols, out_channels)
        let w_t: Vec<f32> = {
            let w = &w_data;
            (0..col_cols)
                .flat_map(|i| (0..out_channels).map(move |j| w[j * col_cols + i]))
                .collect()
        };

        let result = matmul_blas(col_slice, &w_t, col_rows, col_cols, out_channels);

        for row in 0..col_rows {
            let n = row / (out_height * out_width);
            let rem = row % (out_height * out_width);
            let oh = rem / out_width;
            let ow = rem % out_width;

            for oc in 0..out_channels {
                let out_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                let bias_val = bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0);
                unsafe { *out_ptr.add(out_idx) = result[row * out_channels + oc] + bias_val };
            }
        }
    } else {
        for oc in 0..out_channels {
            let w_row = &w_data[oc * col_cols..(oc + 1) * col_cols];
            let bias_val = bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0);

            for row in 0..col_rows {
                let col_row = &col_slice[row * col_cols..(row + 1) * col_cols];
                let sum = simd_dot_product(col_row, w_row, col_cols);

                let n = row / (out_height * out_width);
                let rem = row % (out_height * out_width);
                let oh = rem / out_width;
                let ow = rem % out_width;

                let out_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                unsafe { *out_ptr.add(out_idx) = sum + bias_val };
            }
        }
    }

    output
}

fn layer_norm_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let _normalized_shape = args[1].shape();
    let weight = if args.len() > 2 && args[2].numel() > 0 {
        Some(args[2])
    } else {
        None
    };
    let bias = if args.len() > 3 && args[3].numel() > 0 {
        Some(args[3])
    } else {
        None
    };
    let eps = if args.len() > 4 {
        args[4].item() as f64
    } else {
        1e-5
    };

    let x_shape = x.shape();
    let ndim = x_shape.len();
    let _normalized_shape: i64 = x_shape.iter().skip(ndim - 1).product();

    let mean = x.mean((ndim - 1) as i32, true);
    let var = x
        .sub(&mean.clone())
        .mul(&x.sub(&mean.clone()))
        .mean((ndim - 1) as i32, true);
    let std = var.add(&Tensor::from_scalar(eps as f32)).sqrt();

    let mut normalized = x.sub(&mean).div(&std);
    let x_hat = normalized.clone(); // Store x_hat before scaling/shifting

    if let Some(w) = weight {
        normalized = normalized.mul(w);
    }
    if let Some(b) = bias {
        normalized = normalized.add(b);
    }

    // Return: output, mean, variance, x_hat (normalized before scaling/shifting)
    vec![normalized, mean, var, x_hat]
}

fn batch_norm_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let weight = if args.len() > 1 && args[1].numel() > 0 {
        Some(args[1])
    } else {
        None
    };
    let bias = if args.len() > 2 && args[2].numel() > 0 {
        Some(args[2])
    } else {
        None
    };
    let _training = if args.len() > 5 {
        args[5].item() != 0.0
    } else {
        false
    };
    let eps = if args.len() > 6 {
        args[6].item() as f64
    } else {
        1e-5
    };

    let x_shape = x.shape();
    let _num_features = x_shape[1];

    let mean = x.mean(0, false).unsqueeze(0);
    let var = x
        .sub(&mean.clone())
        .mul(&x.sub(&mean.clone()))
        .mean(0, false)
        .unsqueeze(0);
    let std = var.add(&Tensor::from_scalar(eps as f32)).sqrt();

    let mut normalized = x.sub(&mean).div(&std);

    if let Some(w) = weight {
        normalized = normalized.mul(&w.unsqueeze(0));
    }
    if let Some(b) = bias {
        normalized = normalized.add(&b.unsqueeze(0));
    }

    vec![normalized]
}

fn embedding_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let weight = args[0];
    let indices = args[1];

    let weight_shape = weight.shape();
    let num_embeddings = weight_shape[0];
    let embedding_dim = weight_shape[1];

    let indices_shape = indices.shape();
    let batch_size: i64 = indices_shape.iter().product();

    let output_shape: Vec<i64> = indices_shape
        .iter()
        .chain(std::iter::once(&embedding_dim))
        .copied()
        .collect();
    let mut output = Tensor::zeros(output_shape.clone(), weight.dtype(), weight.device());

    // Read indices as i32 (or i64 depending on dtype) to avoid precision loss with f32
    let indices_ptr = indices.data_ptr() as *const i32;
    let weight_ptr = weight.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    for i in 0..batch_size as usize {
        let idx = unsafe { *indices_ptr.add(i) } as usize;
        if idx < num_embeddings as usize {
            for j in 0..embedding_dim as usize {
                let w_idx = idx * embedding_dim as usize + j;
                let o_idx = i * embedding_dim as usize + j;
                unsafe {
                    *out_ptr.add(o_idx) = *weight_ptr.add(w_idx);
                }
            }
        }
    }

    // Set up gradient tracking for embedding
    if weight.requires_grad() {
        let backward = EmbeddingBackward::new(weight.clone(), indices.clone());
        let mut meta = AutogradMeta::new_non_leaf(true);
        meta.grad_fn = Some(Arc::new(backward));
        Arc::make_mut(&mut output.inner).autograd_meta =
            Some(Arc::new(std::sync::Mutex::new(meta)));
    }

    vec![output]
}

pub struct EmbeddingBackward {
    pub inputs: Vec<Tensor>,
    pub edges: Vec<Edge>,
}

impl EmbeddingBackward {
    pub fn new(weight: Tensor, indices: Tensor) -> Self {
        EmbeddingBackward {
            inputs: vec![weight, indices],
            edges: vec![],
        }
    }
}

impl Node for EmbeddingBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad_output = grad_outputs[0].clone().unwrap();
        let weight = &self.inputs[0];
        let indices = &self.inputs[1];

        let weight_shape = weight.shape().clone();
        let embedding_dim = weight_shape[1];
        let batch_size = grad_output.shape()[0];

        // Create gradient for weight (same shape as weight)
        let mut weight_grad = Tensor::zeros(weight_shape.clone(), weight.dtype(), weight.device());

        // Accumulate gradients from output
        let grad_output_ptr = grad_output.data_ptr() as *const f32;
        // Read indices as i32 to avoid precision loss with f32
        let indices_ptr = indices.data_ptr() as *const i32;
        let weight_grad_inner = Arc::make_mut(&mut weight_grad.inner);
        let weight_grad_storage = Arc::make_mut(&mut weight_grad_inner.storage);
        let Storage::Cpu(cpu_storage) = weight_grad_storage else {
            panic!("Expected CPU storage");
        };
        let weight_grad_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

        for i in 0..batch_size as usize {
            let idx = unsafe { *indices_ptr.add(i) } as usize;
            if idx < weight_shape[0] as usize {
                for j in 0..embedding_dim as usize {
                    let w_idx = idx * embedding_dim as usize + j;
                    let o_idx = i * embedding_dim as usize + j;
                    unsafe {
                        *weight_grad_ptr.add(w_idx) += *grad_output_ptr.add(o_idx);
                    }
                }
            }
        }

        vec![Some(weight_grad)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "EmbeddingBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

fn zeros_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0].shape();
    let dtype = if args.len() > 1 {
        let dtype_slice = args[1].as_f32_slice();
        if !dtype_slice.is_empty() {
            DType::from_str(
                &dtype_slice
                    .iter()
                    .map(|&x| x as u8 as char)
                    .collect::<String>(),
            )
            .unwrap_or(DType::F32)
        } else {
            DType::F32
        }
    } else {
        DType::F32
    };
    let device = Device::Cpu;

    vec![Tensor::zeros(shape, dtype, device)]
}

fn ones_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0].shape();
    let dtype = DType::F32;
    let device = Device::Cpu;

    vec![Tensor::ones(shape, dtype, device)]
}

fn full_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0].shape();
    let value = args[1].item();
    let dtype = DType::F32;
    let device = Device::Cpu;

    vec![Tensor::full(shape, value, dtype, device)]
}

fn arange_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let start = args[0].item();
    let end = args[1].item();
    let step = if args.len() > 2 { args[2].item() } else { 1.0 };

    let numel = ((end - start) / step).ceil() as usize;
    let values: Vec<f32> = (0..numel).map(|i| start + i as f32 * step).collect();

    vec![Tensor::from_vec(values, vec![numel as i64])]
}

fn linspace_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let start = args[0].item();
    let end = args[1].item();
    let steps = args[2].item() as usize;

    let values: Vec<f32> = (0..steps)
        .map(|i| {
            let t = if steps <= 1 {
                0.0
            } else {
                i as f32 / (steps - 1) as f32
            };
            start * (1.0 - t) + end * t
        })
        .collect();

    vec![Tensor::from_vec(values, vec![steps as i64])]
}

fn eye_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let n = args[0].item() as usize;
    let m = if args.len() > 1 {
        args[1].item() as usize
    } else {
        n
    };

    let mut values = vec![0.0f32; n * m];
    for i in 0..n.min(m) {
        values[i * m + i] = 1.0;
    }

    vec![Tensor::from_vec(values, vec![n as i64, m as i64])]
}

fn randn_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape_tensor = args[0];
    let shape: Vec<i64> = shape_tensor.inner.sizes.iter().copied().collect();
    let numel: i64 = shape.iter().product();

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut values = vec![0.0f32; numel as usize];

    for v in &mut values {
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }

    vec![Tensor::from_vec(values, shape)]
}

fn rand_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape_tensor = args[0];
    let shape: Vec<i64> = shape_tensor.inner.sizes.iter().copied().collect();
    let numel: i64 = shape.iter().product();

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let values: Vec<f32> = (0..numel as usize).map(|_| rng.gen()).collect();

    vec![Tensor::from_vec(values, shape)]
}

#[allow(dead_code)]
fn randint_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape_tensor = args[0];
    let shape: Vec<i64> = shape_tensor.inner.sizes.iter().copied().collect();
    let low = args[1].item() as i32;
    let high = args[2].item() as i32;

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let numel: i64 = shape.iter().product();
    let values: Vec<f32> = (0..numel as usize)
        .map(|_| rng.gen_range(low..high) as f32)
        .collect();

    vec![Tensor::from_vec(values, shape)]
}

#[allow(dead_code)]
fn read_f32(slice: &[u8], dtype: DType) -> f32 {
    match dtype {
        DType::F32 => {
            let ptr = slice.as_ptr() as *const f32;
            unsafe { *ptr }
        }
        DType::F64 => {
            let ptr = slice.as_ptr() as *const f64;
            unsafe { *ptr as f32 }
        }
        DType::I32 => {
            let ptr = slice.as_ptr() as *const i32;
            unsafe { *ptr as f32 }
        }
        DType::I64 => {
            let ptr = slice.as_ptr() as *const i64;
            unsafe { *ptr as f32 }
        }
        _ => 0.0,
    }
}

#[allow(dead_code)]
fn write_f32(slice: &[u8], val: f32) {
    let ptr = slice.as_ptr() as *mut u8;
    unsafe {
        *(ptr as *mut f32) = val;
    }
}

fn clamp_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let min_val = args[1].item();
    let max_val = args[2].item();

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice
                .par_iter_mut()
                .zip(a_slice.par_iter())
                .for_each(|(out, &val)| {
                    *out = val.clamp(min_val, max_val);
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    let x = *a_ptr.add(idx);
                    *out_ptr.add(idx) = x.clamp(min_val, max_val);
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.clamp(min_val, max_val);
            }
        }
    }

    vec![output]
}

fn pow_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let exponent = args[1].item();

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice
                .par_iter_mut()
                .zip(a_slice.par_iter())
                .for_each(|(out, &val)| {
                    *out = val.powf(exponent);
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    let x = *a_ptr.add(idx);
                    *out_ptr.add(idx) = x.powf(exponent);
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.powf(exponent);
            }
        }
    }

    vec![output]
}

fn maximum_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.inner.sizes.as_slice();
    let b_shape = b.inner.sizes.as_slice();
    let a_contig = a.is_contiguous();
    let b_contig = b.is_contiguous();
    let a_numel = a.inner.numel() as usize;

    if a_contig && b_contig && a_shape == b_shape && a_numel > 2048 {
        let output_shape = a_shape.to_vec();
        let numel = a_numel;

        let mut output = Tensor::empty(output_shape, a.dtype(), a.device());

        let a_ptr = a.data_ptr_f32();
        let b_ptr = b.data_ptr_f32();

        let output_inner = Arc::make_mut(&mut output.inner);
        let output_storage = Arc::make_mut(&mut output_inner.storage);
        let Storage::Cpu(cpu_storage) = output_storage else {
            panic!("Expected CPU storage");
        };
        let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;

            (0..num_chunks)
                .into_par_iter()
                .for_each(|chunk_idx| unsafe {
                    let start = chunk_idx * chunk_size;
                    let end = std::cmp::min(start + chunk_size, numel);
                    for i in start..end {
                        let val_a = *((a_usize + i * 4) as *const f32);
                        let val_b = *((b_usize + i * 4) as *const f32);
                        *((out_usize + i * 4) as *mut f32) = val_a.max(val_b);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    let val_a = *a_ptr.add(idx);
                    let val_b = *b_ptr.add(idx);
                    *out_ptr.add(idx) = val_a.max(val_b);
                }
            }
        }

        vec![output]
    } else {
        // Fallback for non-contiguous or smaller tensors
        let iter = TensorIterator::build_for_binary(a, b);
        let output_shape = iter.output_shape.to_vec();

        let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

        let numel = output_shape.iter().product::<i64>() as usize;
        let output_inner = Arc::make_mut(&mut output.inner);
        let output_storage = Arc::make_mut(&mut output_inner.storage);
        let Storage::Cpu(cpu_storage) = output_storage else {
            panic!("Expected CPU storage");
        };
        let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

        for idx in 0..numel {
            unsafe {
                let a_idx = iter.input_index(0, idx);
                let b_idx = iter.input_index(1, idx);
                let val_a = *(a.data_ptr() as *const f32).add(a_idx);
                let val_b = *(b.data_ptr() as *const f32).add(b_idx);
                *out_ptr.add(idx) = val_a.max(val_b);
            }
        }

        vec![output]
    }
}

fn gt_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let threshold = args[1].item();

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    for idx in 0..numel {
        unsafe {
            let val = *a_ptr.add(idx);
            *out_ptr.add(idx) = if val > threshold { 1.0 } else { 0.0 };
        }
    }

    vec![output]
}

fn sign_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_ptr = cpu_storage.data.as_mut_ptr() as *mut f32;

    for idx in 0..numel {
        unsafe {
            let val = *a_ptr.add(idx);
            *out_ptr.add(idx) = if val > 0.0 {
                1.0
            } else if val < 0.0 {
                -1.0
            } else {
                0.0
            };
        }
    }

    vec![output]
}

#[ctor::ctor]
fn register_kernels() {
    register("add", DispatchKey::Cpu, add_kernel as KernelFn);
    register("sub", DispatchKey::Cpu, sub_kernel as KernelFn);
    register("mul", DispatchKey::Cpu, mul_kernel as KernelFn);
    register("div", DispatchKey::Cpu, div_kernel as KernelFn);
    register("neg", DispatchKey::Cpu, neg_kernel as KernelFn);
    register("abs", DispatchKey::Cpu, abs_kernel as KernelFn);
    register("exp", DispatchKey::Cpu, exp_kernel as KernelFn);
    register("log", DispatchKey::Cpu, log_kernel as KernelFn);
    register("sqrt", DispatchKey::Cpu, sqrt_kernel as KernelFn);
    register("relu", DispatchKey::Cpu, relu_kernel as KernelFn);
    register(
        "fused_add_relu",
        DispatchKey::Cpu,
        fused_add_relu_kernel as KernelFn,
    );
    register("gelu", DispatchKey::Cpu, gelu_kernel as KernelFn);
    register("sigmoid", DispatchKey::Cpu, sigmoid_kernel as KernelFn);
    register("tanh", DispatchKey::Cpu, tanh_kernel as KernelFn);
    register("silu", DispatchKey::Cpu, silu_kernel as KernelFn);
    register("clamp", DispatchKey::Cpu, clamp_kernel as KernelFn);
    register("pow", DispatchKey::Cpu, pow_kernel as KernelFn);
    register("maximum", DispatchKey::Cpu, maximum_kernel as KernelFn);
    register("matmul", DispatchKey::Cpu, matmul_kernel as KernelFn);
    register("linear", DispatchKey::Cpu, linear_kernel as KernelFn);
    register(
        "fused_linear_relu",
        DispatchKey::Cpu,
        fused_linear_relu_kernel as KernelFn,
    );
    register(
        "fused_linear_gelu",
        DispatchKey::Cpu,
        fused_linear_gelu_kernel as KernelFn,
    );
    register(
        "fused_mul_add",
        DispatchKey::Cpu,
        fused_mul_add_kernel as KernelFn,
    );
    register(
        "fused_linear_silu",
        DispatchKey::Cpu,
        fused_linear_silu_kernel as KernelFn,
    );
    register("sum", DispatchKey::Cpu, sum_kernel as KernelFn);
    register("mean", DispatchKey::Cpu, mean_kernel as KernelFn);
    register("max", DispatchKey::Cpu, max_kernel as KernelFn);
    register("min", DispatchKey::Cpu, min_kernel as KernelFn);
    register("argmax", DispatchKey::Cpu, argmax_kernel as KernelFn);
    register("argmin", DispatchKey::Cpu, argmin_kernel as KernelFn);
    register("softmax", DispatchKey::Cpu, softmax_kernel as KernelFn);
    register(
        "log_softmax",
        DispatchKey::Cpu,
        log_softmax_kernel as KernelFn,
    );
    register("mse_loss", DispatchKey::Cpu, mse_loss_kernel as KernelFn);
    register(
        "cross_entropy_loss",
        DispatchKey::Cpu,
        cross_entropy_loss_kernel as KernelFn,
    );

    // GPU fallback for cross_entropy_loss (moves to CPU for computation)
    fn cross_entropy_loss_gpu_fallback(args: &[&Tensor]) -> Vec<Tensor> {
        // Move inputs to CPU, compute, then move result back to GPU
        let pred_cpu = args[0].to_cpu();
        let target_cpu = args[1].to_cpu();
        let reduction_code = args[2].item();

        // Create CPU tensors for dispatch
        let cpu_args = [&pred_cpu, &target_cpu, &Tensor::from_scalar(reduction_code)];
        let result = cross_entropy_loss_kernel(&cpu_args);

        // Move result back to original GPU
        let device_id = match args[0].inner.storage.as_ref() {
            Storage::Wgpu(gpu) => gpu.device_id,
            _ => 0,
        };
        vec![result[0].to_gpu(device_id)]
    }

    register(
        "cross_entropy_loss",
        DispatchKey::Wgpu,
        cross_entropy_loss_gpu_fallback as KernelFn,
    );

    register("conv2d", DispatchKey::Cpu, conv2d_kernel as KernelFn);
    register(
        "layer_norm",
        DispatchKey::Cpu,
        layer_norm_kernel as KernelFn,
    );
    register(
        "batch_norm",
        DispatchKey::Cpu,
        batch_norm_kernel as KernelFn,
    );
    register("embedding", DispatchKey::Cpu, embedding_kernel as KernelFn);
    register("zeros", DispatchKey::Cpu, zeros_kernel as KernelFn);
    register("ones", DispatchKey::Cpu, ones_kernel as KernelFn);
    register("full", DispatchKey::Cpu, full_kernel as KernelFn);
    register("arange", DispatchKey::Cpu, arange_kernel as KernelFn);
    register("linspace", DispatchKey::Cpu, linspace_kernel as KernelFn);
    register("eye", DispatchKey::Cpu, eye_kernel as KernelFn);
    register("randn", DispatchKey::Cpu, randn_kernel as KernelFn);
    register("rand", DispatchKey::Cpu, rand_kernel as KernelFn);
    register("gt_scalar", DispatchKey::Cpu, gt_scalar_kernel as KernelFn);

    // GPU fallback for gt_scalar (moves to CPU for computation)
    fn gt_scalar_gpu_fallback(args: &[&Tensor]) -> Vec<Tensor> {
        let input_cpu = args[0].to_cpu();
        let threshold = args[1].item();
        let result_cpu = gt_scalar_kernel(&[&input_cpu, &Tensor::from_scalar(threshold)]);

        let device_id = match args[0].inner.storage.as_ref() {
            Storage::Wgpu(gpu) => gpu.device_id,
            _ => 0,
        };
        vec![result_cpu[0].to_gpu(device_id)]
    }

    register(
        "gt_scalar",
        DispatchKey::Wgpu,
        gt_scalar_gpu_fallback as KernelFn,
    );
    register("sign", DispatchKey::Cpu, sign_kernel as KernelFn);
}
