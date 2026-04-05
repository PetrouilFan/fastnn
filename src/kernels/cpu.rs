#![allow(unused_imports)]

use crate::autograd::{AutogradMeta, Edge, Node};
use crate::dispatcher::{register, DispatchKey, KernelFn};
use crate::iterator::TensorIterator;
use crate::kernels::blas::{
    matmul_blas, matmul_blas_into, matmul_blas_with_transpose, matmul_blas_with_transpose_into,
    MIN_BLAS_SIZE,
};
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use std::sync::Arc;
use std::sync::OnceLock;

/// Enable DAZ (Denormals-Are-Zero) and FTZ (Flush-To-Zero) for the current thread.
/// Subnormal floats are treated as zero, preventing catastrophic throughput drops
/// when weights approach zero during training. Thread-local since MXCSR is per-thread.
///
/// SAFETY: This changes floating-point behavior for denormal numbers on the current thread.
/// Denormals will be flushed to zero, which is acceptable for ML workloads where the
/// precision loss from subnormals is negligible compared to the performance gain.
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[inline(always)]
#[allow(deprecated)]
unsafe fn enable_daz_ftz() {
    use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
    // FTZ = bit 15, DAZ = bit 6 of MXCSR
    const FTZ: u32 = 1 << 15;
    const DAZ: u32 = 1 << 6;
    let mxcsr = _mm_getcsr();
    _mm_setcsr(mxcsr | FTZ | DAZ);
}

// Thread-local guard ensuring DAZ/FTZ is enabled once per thread.
// Called at the start of SIMD hot paths to guarantee correct CPU state.
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
thread_local! {
    static DAZ_FTZ_INIT: () = {
        // SAFETY: Called once per thread during TLS initialization.
        // Only affects denormal float handling, not normal arithmetic.
        unsafe { enable_daz_ftz(); }
    };
}

/// Ensure DAZ/FTZ is enabled on the current thread (no-op on non-x86).
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[inline(always)]
fn ensure_daz_ftz() {
    DAZ_FTZ_INIT.with(|_| {});
}

#[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
#[inline(always)]
fn ensure_daz_ftz() {}

// Thread-local reusable scratch buffer for conv2d operations.
// Avoids per-call heap allocations for im2col and GEMM output buffers.
// SAFETY with rayon: Each rayon worker thread gets its own independent
// CONV_SCRATCH instance. The RefCell is only borrowed within a single thread.
// rayon's par_chunks_mut guarantees non-overlapping mutable slices, so no
// aliasing occurs even when the borrow spans a parallel section. The buffer
// is never shared across threads.
thread_local! {
    static CONV_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

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

// Memory-bound elementwise ops: 128KB working set (L2 cache friendly)
const CHUNK_MEMBOUND: usize = 1024 * 32; // 32K f32 = 128KB

// Compute-bound transcendental ops: 32KB for better load balancing
const CHUNK_TRANSCENDENTAL: usize = 1024 * 8; // 8K f32 = 32KB

// Rayon parallel threshold: only parallelize above this size
// Prevents overhead from parallelization on small tensors
#[allow(dead_code)]
const PARALLEL_THRESHOLD: usize = 1024 * 32; // 32K elements

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

/// Horizontal sum of __m256 — used by softmax and other AVX2 kernels.
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

// Parallel sigmoid AVX2 kernel using vectorized fast_exp
#[cfg(all(feature = "parallel", feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_parallel_avx2(
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

// Sigmoid parallel AVX512 kernel using vectorized fast_exp
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

// Parallel tanh AVX2 kernel using vectorized fast_exp
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

// Tanh parallel AVX512 kernel using vectorized fast_exp
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
        if a_val != b_val && a_val != 1 && b_val != 1 {
            panic!("shapes {:?} and {:?} are not broadcast-compatible", a, b);
        }
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
    // Precompute multipliers on the stack: multipliers[i] = product(out_shape[i+1..])
    let mut multipliers: smallvec::SmallVec<[usize; 8]> = smallvec::smallvec![0usize; ndim];
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
        let out_data = Arc::make_mut(&mut cpu_storage.data);
        let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    // Convert to usize for the helper function (use SmallVec to avoid heap alloc)
    let out_shape_usize: smallvec::SmallVec<[usize; 8]> =
        out_shape.iter().map(|&x| x as usize).collect();
    let a_shape_usize: smallvec::SmallVec<[usize; 8]> =
        a_shape.iter().map(|&x| x as usize).collect();
    let b_shape_usize: smallvec::SmallVec<[usize; 8]> =
        b_shape.iter().map(|&x| x as usize).collect();
    let a_strides_usize: smallvec::SmallVec<[usize; 8]> =
        a_strides.iter().map(|&x| x as usize).collect();
    let b_strides_usize: smallvec::SmallVec<[usize; 8]> =
        b_strides.iter().map(|&x| x as usize).collect();

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
        let out_data = Arc::make_mut(&mut cpu_storage.data);
        let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    // Convert to usize for the helper function (use SmallVec to avoid heap alloc)
    let out_shape_usize: smallvec::SmallVec<[usize; 8]> =
        out_shape.iter().map(|&x| x as usize).collect();
    let a_shape_usize: smallvec::SmallVec<[usize; 8]> =
        a_shape.iter().map(|&x| x as usize).collect();
    let b_shape_usize: smallvec::SmallVec<[usize; 8]> =
        b_shape.iter().map(|&x| x as usize).collect();
    let a_strides_usize: smallvec::SmallVec<[usize; 8]> =
        a_strides.iter().map(|&x| x as usize).collect();
    let b_strides_usize: smallvec::SmallVec<[usize; 8]> =
        b_strides.iter().map(|&x| x as usize).collect();

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
        let out_data = Arc::make_mut(&mut cpu_storage.data);
        let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    // Convert to usize for the helper function (use SmallVec to avoid heap alloc)
    let out_shape_usize: smallvec::SmallVec<[usize; 8]> =
        out_shape.iter().map(|&x| x as usize).collect();
    let a_shape_usize: smallvec::SmallVec<[usize; 8]> =
        a_shape.iter().map(|&x| x as usize).collect();
    let b_shape_usize: smallvec::SmallVec<[usize; 8]> =
        b_shape.iter().map(|&x| x as usize).collect();
    let a_strides_usize: smallvec::SmallVec<[usize; 8]> =
        a_strides.iter().map(|&x| x as usize).collect();
    let b_strides_usize: smallvec::SmallVec<[usize; 8]> =
        b_strides.iter().map(|&x| x as usize).collect();

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    // Convert to usize for the helper function (use SmallVec to avoid heap alloc)
    let out_shape_usize: smallvec::SmallVec<[usize; 8]> =
        out_shape.iter().map(|&x| x as usize).collect();
    let a_shape_usize: smallvec::SmallVec<[usize; 8]> =
        a_shape.iter().map(|&x| x as usize).collect();
    let b_shape_usize: smallvec::SmallVec<[usize; 8]> =
        b_shape.iter().map(|&x| x as usize).collect();
    let a_strides_usize: smallvec::SmallVec<[usize; 8]> =
        a_strides.iter().map(|&x| x as usize).collect();
    let b_strides_usize: smallvec::SmallVec<[usize; 8]> =
        b_strides.iter().map(|&x| x as usize).collect();

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let numel = a.numel() as usize;

    // Fast path: skip TensorIterator for contiguous tensors
    if a.is_contiguous() && numel > 2048 {
        let mut output = Tensor::empty(a.shape().to_vec(), a.dtype(), a.device());
        {
            let inner = Arc::make_mut(&mut output.inner);
            let storage = Arc::make_mut(&mut inner.storage);
            let Storage::Cpu(cpu_storage) = storage else {
                panic!()
            };
            let out_data = Arc::make_mut(&mut cpu_storage.data);
            let a_ptr = a.data_ptr() as *const f32;
            let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
                                relu_parallel_avx512(
                                    chunk_idx, chunk_size, numel, a_usize, out_usize,
                                );
                            });
                    }
                    SimdLevel::Avx2 => {
                        (0..num_chunks)
                            .into_par_iter()
                            .for_each(|chunk_idx| unsafe {
                                relu_parallel_avx2(
                                    chunk_idx, chunk_size, numel, a_usize, out_usize,
                                );
                            });
                    }
                    SimdLevel::Scalar => {
                        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                            relu_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        relu_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                        relu_simd(a_slice, out_slice);
                    } else {
                        for idx in 0..numel {
                            unsafe {
                                *out_ptr.add(idx) = (*a_ptr.add(idx)).max(0.0);
                            }
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = (*a_ptr.add(idx)).max(0.0);
                        }
                    }
                }
            }
        }
        return vec![output];
    }

    // Fallback: use TensorIterator for non-contiguous tensors
    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();
    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());
    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!()
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    for idx in 0..numel {
        unsafe {
            let val = *a_ptr.add(idx);
            *out_ptr.add(idx) = val.max(0.0);
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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    // Removed debug file writing that was causing issues on Windows

    let a = args[0];
    let b = args[1];

    let a_shape = a.shape();
    let b_shape = b.shape();

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
    let _a_is_transposed =
        a_strides[a.ndim() - 2] == 1 && a_strides[a.ndim() - 1] >= a_shape[a_shape.len() - 2];
    let b_is_transposed =
        b_strides[b.ndim() - 2] == 1 && b_strides[b.ndim() - 1] >= b_shape[b_shape.len() - 2];

    // For matmul: A[m, k] @ B[k, n] = C[m, n]
    // When B is transposed (shape [n, k] representing original [k, n]):
    // The transposed view has shape [n, k] where n is the original outer dim
    // and k is the original inner dim
    if b_is_transposed {
        // B is transposed: shape [n, k] represents original matrix [k, n]
        // For matmul A[m,k] @ B[k,n], we need B's inner dim (k) to match A's inner dim (k)
        // In the transposed view [n, k], the inner dim k is at position 0 (b_shape[0])
        let b_inner_dim = b_shape[0] as i32; // k is at position 0 for transposed
        if b_inner_dim != k {
            panic!(
                "matmul: transposed B dimensions incompatible: A[{}, {}] @ B.T[{}, {}]",
                m, k, b_shape[0], b_shape[1]
            );
        }
    } else {
        // Standard case: B is not transposed, shape [k, n]
        if b_shape[b_shape.len() - 2] as i32 != k {
            panic!(
                "matmul: A[{}, {}] @ B[{}, {}] - B second-to-last dim {} != k {}",
                m,
                k,
                b_shape[b_shape.len() - 2],
                b_shape[b_shape.len() - 1],
                b_shape[b_shape.len() - 2],
                k
            );
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

    // Save original shapes for output reshape
    let orig_a_shape = a_shape.clone();

    // For N-D tensors (N > 3), flatten all batch dims into a single batch dim
    // by reshaping to 3D. This avoids incorrect batch stride calculations.
    let a_3d = if a_shape.len() > 3 {
        let flat_batch: i64 = a_shape[..a_shape.len() - 2].iter().product();
        a.reshape(vec![
            flat_batch,
            a_shape[a_shape.len() - 2],
            a_shape[a_shape.len() - 1],
        ])
    } else {
        a.clone()
    };
    let b_3d = if b_shape.len() > 3 {
        let flat_batch: i64 = b_shape[..b_shape.len() - 2].iter().product();
        b.reshape(vec![
            flat_batch,
            b_shape[b_shape.len() - 2],
            b_shape[b_shape.len() - 1],
        ])
    } else {
        b.clone()
    };

    let a = &a_3d;
    let b = &b_3d;
    let a_shape = a.shape();
    let b_shape = b.shape();
    let a_strides = a.strides();
    let b_strides = b.strides();

    let mut output_shape: smallvec::SmallVec<[i64; 4]> =
        smallvec::SmallVec::with_capacity(a_shape.len());
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

    let mut output = Tensor::empty(output_shape.to_vec(), a.dtype(), a.device());

    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let a_rows = a_shape[a_shape.len() - 2] as usize;
    let a_cols = a_shape[a_shape.len() - 1] as usize;
    let b_cols = b_shape[b_shape.len() - 1] as usize;

    // Detect transposed matrices by checking strides
    // For row-major contiguous matrix [rows, cols], stride_0 = cols, stride_1 = 1
    // For transposed [rows, cols] stored as [cols, rows], stride_0 = 1, stride_1 = rows
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

    // Reshape trick: For batched A [batch, m, k] @ 2D B [k, n], flatten to [batch*m, k] @ [k, n]
    // This enables a single BLAS call instead of looping over batch dimension
    let can_reshape_trick = batch > 1
        && b_shape.len() == 2  // B is 2D, not batched
        && a.is_contiguous()
        && b_valid_for_blas;

    // Batched 3D BLAS: both A and B are batched 3D contiguous tensors
    let can_batch_blas = batch > 1
        && a_shape.len() == 3
        && b_shape.len() == 3
        && a.is_contiguous()
        && b.is_contiguous();

    // Use BLAS for:
    // 1. Single batch with large enough matrices
    // 2. Batched operations with 2D weight (reshape trick)
    // 3. Batched 3D contiguous tensors (batched BLAS loop)
    let matrices_large = m as usize * n as usize >= MIN_BLAS_SIZE * MIN_BLAS_SIZE
        || m as usize * k as usize >= MIN_BLAS_SIZE * MIN_BLAS_SIZE
        || k as usize * n as usize >= MIN_BLAS_SIZE * MIN_BLAS_SIZE;
    let use_blas = (can_reshape_trick || batch == 1 || can_batch_blas)
        && matrices_large
        && a_valid_for_blas
        && b_valid_for_blas;

    if use_blas {
        if can_reshape_trick {
            // Reshape trick: treat [batch, m, k] as [batch*m, k]
            // Single BLAS call for entire batch
            let batch_m = batch * m as usize;
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, batch_m * k as usize) };
            let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, k as usize * b_cols) };
            let out_slice =
                unsafe { std::slice::from_raw_parts_mut(out_ptr, batch_m * n as usize) };
            matmul_blas_with_transpose_into(
                a_slice,
                b_slice,
                out_slice,
                batch_m,
                k as usize,
                n as usize,
                a_is_transposed,
                b_is_transposed,
            );
        } else if can_batch_blas {
            // Batched 3D BLAS loop: process each batch element separately
            let a_slice =
                unsafe { std::slice::from_raw_parts(a_ptr, batch * m as usize * k as usize) };
            let b_slice =
                unsafe { std::slice::from_raw_parts(b_ptr, batch * k as usize * n as usize) };
            let out_slice =
                unsafe { std::slice::from_raw_parts_mut(out_ptr, batch * m as usize * n as usize) };

            let m_usize = m as usize;
            let k_usize = k as usize;
            let n_usize = n as usize;
            let a_batch_elems = m_usize * k_usize;
            let b_batch_elems = k_usize * n_usize;
            let out_batch_elems = m_usize * n_usize;

            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                out_slice
                    .par_chunks_mut(out_batch_elems)
                    .enumerate()
                    .for_each(|(bat, out_chunk)| {
                        let a_offset = bat * a_batch_elems;
                        let b_offset = bat * b_batch_elems;
                        matmul_blas_with_transpose_into(
                            &a_slice[a_offset..],
                            &b_slice[b_offset..],
                            out_chunk,
                            m_usize,
                            k_usize,
                            n_usize,
                            false,
                            false,
                        );
                    });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for bat in 0..batch {
                    let a_offset = bat * a_batch_elems;
                    let b_offset = bat * b_batch_elems;
                    let out_offset = bat * out_batch_elems;
                    matmul_blas_with_transpose_into(
                        &a_slice[a_offset..],
                        &b_slice[b_offset..],
                        &mut out_slice[out_offset..],
                        m_usize,
                        k_usize,
                        n_usize,
                        false,
                        false,
                    );
                }
            }
        } else {
            // Single batch BLAS
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, a_rows * a_cols) };
            let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, k as usize * b_cols) };
            let out_slice =
                unsafe { std::slice::from_raw_parts_mut(out_ptr, m as usize * n as usize) };
            matmul_blas_with_transpose_into(
                a_slice,
                b_slice,
                out_slice,
                m as usize,
                k as usize,
                n as usize,
                a_is_transposed,
                b_is_transposed,
            );
        }
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

    // Reshape output back to original N-D shape if we flattened
    if orig_a_shape.len() > 3 {
        let mut final_shape: Vec<i64> = orig_a_shape[..orig_a_shape.len() - 2].to_vec();
        final_shape.push(m as i64);
        final_shape.push(n as i64);
        output = output.reshape(final_shape);
    }

    vec![output]
}

#[cfg(feature = "parallel")]
#[inline]
#[allow(clippy::too_many_arguments)]
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
    let total_rows = batch * m_usize;

    let abs = a_batch_stride as usize;
    let as0 = a_stride_0 as usize;
    let as1 = a_stride_1 as usize;
    let bbs = b_batch_stride as usize;
    let bs0 = b_stride_0 as usize;
    let bs1 = b_stride_1 as usize;

    // Parallelize at the row level (batch * m rows).
    // Each row calls blocked_row_matmul which tiles K and N for cache efficiency.
    (0..total_rows).into_par_iter().for_each(|row_idx| {
        blocked_row_matmul(
            a_usize as *const f32,
            b_usize as *const f32,
            out_usize as *mut f32,
            row_idx,
            m_usize,
            n_usize,
            k_usize,
            abs,
            as0,
            as1,
            bbs,
            bs0,
            bs1,
        );
    });
}

/// Cache-blocked scalar matmul for one row of output: C[row, :] += A[row, :] @ B.
/// Tiles the K dimension so that a TILE_K × TILE_N block of B stays in L1 cache.
///
/// When strides are contiguous (a_stride_1 == 1, b_stride_1 == 1) and the
/// target supports AVX2, uses an 8-wide SIMD inner loop that broadcasts
/// A[i, kk] and FMADDs with the contiguous B[kk, j..j+8] row slice.
///
/// Cache math (scalar TILE_N=4):
///   B tile: 64 × 4 × 4 bytes = 1 KB
///   A tile: 1 × 64 × 4 bytes = 256 bytes
///   Total working set ≪ 32 KB L1 cache.
///
/// SIMD cache math (TILE_N_SIMD=8):
///   B tile: 64 × 8 × 4 bytes = 2 KB
///   Still well within L1.
#[inline]
#[allow(clippy::too_many_arguments)]
fn blocked_row_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    row: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch_stride: usize,
    a_stride_0: usize,
    a_stride_1: usize,
    b_batch_stride: usize,
    b_stride_0: usize,
    b_stride_1: usize,
) {
    const TILE_K: usize = 64;
    const TILE_N: usize = 4;

    let bat = row / m;
    let i = row % m;
    let a_off = bat * a_batch_stride + i * a_stride_0;
    let b_off = bat * b_batch_stride;
    let out_off = row * n;

    // Clear output row
    for j in 0..n {
        unsafe {
            *out_ptr.add(out_off + j) = 0.0;
        }
    }

    // AVX2 SIMD fast path: 8-wide FMADD when A and B have contiguous columns.
    // For fixed (i, kk): C[i, j..j+8] += A[i, kk] * B[kk, j..j+8]
    // B[kk, j..j+8] is 8 contiguous floats — a correct plain load.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if a_stride_1 == 1 && b_stride_1 == 1 && n >= 8 {
        use std::arch::x86_64::*;

        const TILE_N_SIMD: usize = 8;
        let mut ko = 0;
        while ko < k {
            let kend = if ko + TILE_K < k { ko + TILE_K } else { k };

            // SIMD tiles: 8 columns at a time
            let mut jo = 0;
            while jo + TILE_N_SIMD <= n {
                unsafe {
                    let mut acc = _mm256_setzero_ps();

                    let mut kk = ko;
                    while kk + 4 <= kend {
                        // Unroll 4 k-steps for ILP
                        let a0 = _mm256_set1_ps(*a_ptr.add(a_off + kk * a_stride_1));
                        let b0 = _mm256_loadu_ps(b_ptr.add(b_off + kk * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a0, b0, acc);

                        let a1 = _mm256_set1_ps(*a_ptr.add(a_off + (kk + 1) * a_stride_1));
                        let b1 = _mm256_loadu_ps(b_ptr.add(b_off + (kk + 1) * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a1, b1, acc);

                        let a2 = _mm256_set1_ps(*a_ptr.add(a_off + (kk + 2) * a_stride_1));
                        let b2 = _mm256_loadu_ps(b_ptr.add(b_off + (kk + 2) * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a2, b2, acc);

                        let a3 = _mm256_set1_ps(*a_ptr.add(a_off + (kk + 3) * a_stride_1));
                        let b3 = _mm256_loadu_ps(b_ptr.add(b_off + (kk + 3) * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a3, b3, acc);

                        kk += 4;
                    }
                    // Scalar tail for remaining kk
                    while kk < kend {
                        let av = _mm256_set1_ps(*a_ptr.add(a_off + kk * a_stride_1));
                        let bv = _mm256_loadu_ps(b_ptr.add(b_off + kk * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(av, bv, acc);
                        kk += 1;
                    }

                    // Accumulate into output (add to existing, not overwrite)
                    let out_v = _mm256_loadu_ps(out_ptr.add(out_off + jo));
                    _mm256_storeu_ps(out_ptr.add(out_off + jo), _mm256_add_ps(out_v, acc));
                }
                jo += TILE_N_SIMD;
            }

            // Scalar tail for remaining columns (< 8)
            while jo < n {
                let mut sum = 0.0f32;
                for kk in ko..kend {
                    sum += unsafe {
                        *a_ptr.add(a_off + kk * a_stride_1)
                            * b_ptr.add(b_off + kk * b_stride_0 + jo).read()
                    };
                }
                unsafe {
                    let p = out_ptr.add(out_off + jo);
                    *p += sum;
                }
                jo += 1;
            }

            ko += TILE_K;
        }
        return;
    }

    // Scalar blocked path (handles non-contiguous strides and small n)
    let mut ko = 0;
    while ko < k {
        let kend = if ko + TILE_K < k { ko + TILE_K } else { k };

        let mut jo = 0;
        while jo + TILE_N <= n {
            let mut acc = [0.0f32; TILE_N];

            let mut kk = ko;
            while kk < kend {
                let av = unsafe { *a_ptr.add(a_off + kk * a_stride_1) };
                for t in 0..TILE_N {
                    acc[t] +=
                        av * unsafe { *b_ptr.add(b_off + kk * b_stride_0 + (jo + t) * b_stride_1) };
                }
                kk += 1;
            }

            for t in 0..TILE_N {
                unsafe {
                    let p = out_ptr.add(out_off + jo + t);
                    *p += acc[t];
                }
            }
            jo += TILE_N;
        }

        // Tail: remaining columns
        while jo < n {
            let mut sum = 0.0f32;
            for kk in ko..kend {
                sum += unsafe {
                    *a_ptr.add(a_off + kk * a_stride_1)
                        * b_ptr.add(b_off + kk * b_stride_0 + jo * b_stride_1).read()
                };
            }
            unsafe {
                let p = out_ptr.add(out_off + jo);
                *p += sum;
            }
            jo += 1;
        }

        ko += TILE_K;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
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
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let total_rows = batch * m_usize;

    for row in 0..total_rows {
        blocked_row_matmul(
            a_ptr,
            b_ptr,
            out_ptr,
            row,
            m_usize,
            n_usize,
            k_usize,
            a_batch_stride as usize,
            a_stride_0 as usize,
            a_stride_1 as usize,
            b_batch_stride as usize,
            b_stride_0 as usize,
            b_stride_1 as usize,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
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
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let total_rows = batch * m_usize;

    for row in 0..total_rows {
        blocked_row_matmul(
            a_ptr,
            b_ptr,
            out_ptr,
            row,
            m_usize,
            n_usize,
            k_usize,
            a_batch_stride as usize,
            a_stride_0 as usize,
            a_stride_1 as usize,
            b_batch_stride as usize,
            b_stride_0 as usize,
            b_stride_1 as usize,
        );
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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let batch_size = batch_size as usize;
    let in_features = in_features as usize;
    let out_features = out_features as usize;

    // Use BLAS for the GEMM when matrices are large enough, then fuse
    // bias + activation in a single parallel pass. This is 3-5x faster
    // than the scalar inner loop because BLAS uses SIMD + cache blocking.
    let use_blas = x.is_contiguous()
        && w.is_contiguous()
        && (batch_size >= MIN_BLAS_SIZE
            || in_features >= MIN_BLAS_SIZE
            || out_features >= MIN_BLAS_SIZE);

    if use_blas {
        let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, batch_size * in_features) };
        let w_slice = unsafe { std::slice::from_raw_parts(w_ptr, out_features * in_features) };
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(out_ptr, batch_size * out_features) };

        // GEMM: [batch, in] @ [out, in]^T = [batch, out]
        // w is [out, in] contiguous, use trans_b=true so BLAS reads it as [in, out]
        matmul_blas_with_transpose_into(
            x_slice,
            w_slice,
            out_slice,
            batch_size,
            in_features,
            out_features,
            false,
            true,
        );

        // Parallel bias + relu pass over all output elements
        let total = batch_size * out_features;
        let out_usize = out_ptr as usize;

        if let Some(b) = bias {
            let b_ptr = b.data_ptr_f32();
            let b_usize = b_ptr as usize;
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..total).into_par_iter().for_each(|idx| {
                    let out_idx = idx % out_features;
                    unsafe {
                        let val = *((out_usize + idx * 4) as *const f32)
                            + *((b_usize + out_idx * 4) as *const f32);
                        *((out_usize + idx * 4) as *mut f32) = val.max(0.0);
                    }
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for idx in 0..total {
                    let out_idx = idx % out_features;
                    unsafe {
                        let val = *((out_usize + idx * 4) as *const f32)
                            + *((b_usize + out_idx * 4) as *const f32);
                        *((out_usize + idx * 4) as *mut f32) = val.max(0.0);
                    }
                }
            }
        } else {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..total).into_par_iter().for_each(|idx| unsafe {
                    let val = *((out_usize + idx * 4) as *const f32);
                    *((out_usize + idx * 4) as *mut f32) = val.max(0.0);
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for idx in 0..total {
                    unsafe {
                        let val = *((out_usize + idx * 4) as *const f32);
                        *((out_usize + idx * 4) as *mut f32) = val.max(0.0);
                    }
                }
            }
        }
    } else {
        // Scalar fallback for small matrices
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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let batch_size = batch_size as usize;
    let in_features = in_features as usize;
    let out_features = out_features as usize;

    let use_blas = x.is_contiguous()
        && w.is_contiguous()
        && (batch_size >= MIN_BLAS_SIZE
            || in_features >= MIN_BLAS_SIZE
            || out_features >= MIN_BLAS_SIZE);

    if use_blas {
        let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, batch_size * in_features) };
        let w_slice = unsafe { std::slice::from_raw_parts(w_ptr, out_features * in_features) };
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(out_ptr, batch_size * out_features) };

        matmul_blas_with_transpose_into(
            x_slice,
            w_slice,
            out_slice,
            batch_size,
            in_features,
            out_features,
            false,
            true,
        );

        let total = batch_size * out_features;
        let out_usize = out_ptr as usize;

        if let Some(b) = bias {
            let b_ptr = b.data_ptr_f32();
            let b_usize = b_ptr as usize;
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..total).into_par_iter().for_each(|idx| {
                    let out_idx = idx % out_features;
                    unsafe {
                        let val = *((out_usize + idx * 4) as *const f32)
                            + *((b_usize + out_idx * 4) as *const f32);
                        *((out_usize + idx * 4) as *mut f32) = val / (1.0 + (-val).exp());
                    }
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for idx in 0..total {
                    let out_idx = idx % out_features;
                    unsafe {
                        let val = *((out_usize + idx * 4) as *const f32)
                            + *((b_usize + out_idx * 4) as *const f32);
                        *((out_usize + idx * 4) as *mut f32) = val / (1.0 + (-val).exp());
                    }
                }
            }
        } else {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..total).into_par_iter().for_each(|idx| unsafe {
                    let val = *((out_usize + idx * 4) as *const f32);
                    *((out_usize + idx * 4) as *mut f32) = val / (1.0 + (-val).exp());
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for idx in 0..total {
                    unsafe {
                        let val = *((out_usize + idx * 4) as *const f32);
                        *((out_usize + idx * 4) as *mut f32) = val / (1.0 + (-val).exp());
                    }
                }
            }
        }
    } else {
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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let batch_size = batch_size as usize;
    let in_features = in_features as usize;
    let out_features = out_features as usize;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let coeff = 0.044715f32;

    let use_blas = x.is_contiguous()
        && w.is_contiguous()
        && (batch_size >= MIN_BLAS_SIZE
            || in_features >= MIN_BLAS_SIZE
            || out_features >= MIN_BLAS_SIZE);

    if use_blas {
        let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, batch_size * in_features) };
        let w_slice = unsafe { std::slice::from_raw_parts(w_ptr, out_features * in_features) };
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(out_ptr, batch_size * out_features) };

        matmul_blas_with_transpose_into(
            x_slice,
            w_slice,
            out_slice,
            batch_size,
            in_features,
            out_features,
            false,
            true,
        );

        let total = batch_size * out_features;
        let out_usize = out_ptr as usize;

        if let Some(b) = bias {
            let b_ptr = b.data_ptr_f32();
            let b_usize = b_ptr as usize;
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..total).into_par_iter().for_each(|idx| {
                    let out_idx = idx % out_features;
                    unsafe {
                        let sum = *((out_usize + idx * 4) as *const f32)
                            + *((b_usize + out_idx * 4) as *const f32);
                        let x3 = sum * sum * sum;
                        let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                        *((out_usize + idx * 4) as *mut f32) = 0.5 * sum * (1.0 + t);
                    }
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for idx in 0..total {
                    let out_idx = idx % out_features;
                    unsafe {
                        let sum = *((out_usize + idx * 4) as *const f32)
                            + *((b_usize + out_idx * 4) as *const f32);
                        let x3 = sum * sum * sum;
                        let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                        *((out_usize + idx * 4) as *mut f32) = 0.5 * sum * (1.0 + t);
                    }
                }
            }
        } else {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..total).into_par_iter().for_each(|idx| unsafe {
                    let sum = *((out_usize + idx * 4) as *const f32);
                    let x3 = sum * sum * sum;
                    let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                    *((out_usize + idx * 4) as *mut f32) = 0.5 * sum * (1.0 + t);
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for idx in 0..total {
                    unsafe {
                        let sum = *((out_usize + idx * 4) as *const f32);
                        let x3 = sum * sum * sum;
                        let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                        *((out_usize + idx * 4) as *mut f32) = 0.5 * sum * (1.0 + t);
                    }
                }
            }
        }
    } else {
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
    }

    vec![output]
}

/// Fast contiguous last-dim sum with SIMD
pub fn sum_last_dim_contiguous(a: &Tensor, dim_size: usize, num_rows: usize) -> Tensor {
    let a_ptr = a.data_ptr() as *const f32;

    // Direct allocation without Arc::make_mut overhead
    let mut result_data = vec![0.0f32; num_rows];
    let out_ptr = result_data.as_mut_ptr();

    #[cfg(feature = "parallel")]
    {
        if num_rows > 64 {
            use rayon::prelude::*;
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_rows).into_par_iter().for_each(|row| {
                let row_ptr = unsafe { (a_usize as *const f32).add(row * dim_size) };
                let mut sum = 0.0f32;
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && dim_size >= 8 {
                        unsafe {
                            let mut acc = _mm256_setzero_ps();
                            let mut j = 0;
                            while j + 8 <= dim_size {
                                acc = _mm256_add_ps(acc, _mm256_loadu_ps(row_ptr.add(j)));
                                j += 8;
                            }
                            sum = hsum256_ps(acc);
                            for k in j..dim_size {
                                sum += *row_ptr.add(k);
                            }
                        }
                    } else {
                        for j in 0..dim_size {
                            unsafe {
                                sum += *row_ptr.add(j);
                            }
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for j in 0..dim_size {
                        unsafe {
                            sum += *row_ptr.add(j);
                        }
                    }
                }
                unsafe {
                    *(out_usize as *mut f32).add(row) = sum;
                }
            });
            return Tensor::from_vec(result_data, vec![num_rows as i64]);
        }
    }

    // Non-parallel: SIMD inline
    for row in 0..num_rows {
        let row_ptr = unsafe { a_ptr.add(row * dim_size) };
        let mut sum = 0.0f32;
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && dim_size >= 8 {
                unsafe {
                    let mut acc = _mm256_setzero_ps();
                    let mut j = 0;
                    while j + 8 <= dim_size {
                        acc = _mm256_add_ps(acc, _mm256_loadu_ps(row_ptr.add(j)));
                        j += 8;
                    }
                    sum = hsum256_ps(acc);
                    for k in j..dim_size {
                        sum += *row_ptr.add(k);
                    }
                }
            } else {
                for j in 0..dim_size {
                    unsafe {
                        sum += *row_ptr.add(j);
                    }
                }
            }
        }
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            for j in 0..dim_size {
                unsafe {
                    sum += *row_ptr.add(j);
                }
            }
        }
        result_data[row] = sum;
    }
    Tensor::from_vec(result_data, vec![num_rows as i64])
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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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

    // Fast path: contiguous sum along last dimension (2D only)
    if dim == ndim - 1 && ndim == 2 && a.is_contiguous() && a.inner.storage_offset == 0 {
        return vec![sum_last_dim_contiguous(a, dim_size, a_shape[0] as usize)];
    }

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let dim_size = a_shape[dim] as usize;
    let mut strides_before = 1i64;
    let mut strides_after = 1i64;

    for i in 0..dim {
        strides_before *= a_shape[i];
    }
    for i in (dim + 1)..ndim {
        strides_after *= a_shape[i];
    }

    let total_blocks = strides_before as usize * strides_after as usize;
    let a_usize = a_ptr as usize;
    let out_usize = out_ptr as usize;
    let a_off = a_storage_offset;

    #[cfg(feature = "parallel")]
    {
        if total_blocks > 64 {
            use rayon::prelude::*;
            (0..total_blocks).into_par_iter().for_each(|block| {
                let mut max_val = f32::NEG_INFINITY;
                let a_p = a_usize as *const f32;
                let ds = dim_size;
                let sa = strides_after as usize;
                for i in 0..ds {
                    let a_idx = (block / sa) * ds * sa + i * sa + block % sa;
                    unsafe {
                        let val = *a_p.add(a_idx + a_off);
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
                unsafe {
                    *(out_usize as *mut f32).add(block) = max_val;
                }
            });
            return vec![output];
        }
    }

    for block in 0..total_blocks {
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

    if dim == ndim - 1 && x.is_contiguous() {
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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
                            // Also handle 4-wide tail for small dims
                            if i + 4 <= dim_size {
                                let vec128 = _mm_loadu_ps(x_row.as_ptr().add(i));
                                let hi = _mm256_extractf128_ps(max_vec, 1);
                                let lo = _mm256_castps256_ps128(max_vec);
                                let merged_lo = _mm_max_ps(lo, vec128);
                                max_vec =
                                    _mm256_insertf128_ps(_mm256_castps128_ps256(merged_lo), hi, 1);
                                i += 4;
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

                // Fused pass: compute exp(x-max), store to output, and accumulate sum
                let mut sum_exp = 0.0f32;

                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            let max_vec = _mm256_set1_ps(max_val);
                            let mut sum_vec = _mm256_setzero_ps();
                            let mut j = 0;
                            while j + 8 <= dim_size {
                                let x_vec = _mm256_loadu_ps(x_row.as_ptr().add(j));
                                let shifted = _mm256_sub_ps(x_vec, max_vec);
                                let exp_vec = fast_exp_avx2(shifted);
                                sum_vec = _mm256_add_ps(sum_vec, exp_vec);

                                _mm256_storeu_ps(out_row.as_mut_ptr().add(j), exp_vec);
                                j += 8;
                            }
                            sum_exp = hsum256_ps(sum_vec);
                            for j2 in j..dim_size {
                                let e = (x_row[j2] - max_val).exp();
                                sum_exp += e;
                                out_row[j2] = e;
                            }
                        }
                    } else {
                        for j in 0..dim_size {
                            let e = (x_row[j] - max_val).exp();
                            sum_exp += e;
                            out_row[j] = e;
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for j in 0..dim_size {
                        let e = (x_row[j] - max_val).exp();
                        sum_exp += e;
                        out_row[j] = e;
                    }
                }

                let inv_sum = 1.0 / sum_exp;

                // Normalize: multiply stored exp values by inv_sum
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            let inv_vec = _mm256_set1_ps(inv_sum);
                            let mut j = 0;
                            while j + 8 <= dim_size {
                                let exp_vec = _mm256_loadu_ps(out_row.as_ptr().add(j));
                                let result = _mm256_mul_ps(exp_vec, inv_vec);
                                _mm256_storeu_ps(out_row.as_mut_ptr().add(j), result);
                                j += 8;
                            }
                            for j2 in j..dim_size {
                                out_row[j2] *= inv_sum;
                            }
                        }
                    } else {
                        for j in 0..dim_size {
                            out_row[j] *= inv_sum;
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for j in 0..dim_size {
                        out_row[j] *= inv_sum;
                    }
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

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        let mut max_vec = _mm256_set1_ps(f32::MIN);
                        let mut j = 0;
                        while j + 8 <= dim_size {
                            let vec = _mm256_loadu_ps(x_row.as_ptr().add(j));
                            max_vec = _mm256_max_ps(max_vec, vec);
                            j += 8;
                        }
                        let mut max_arr = [0.0f32; 8];
                        _mm256_storeu_ps(max_arr.as_mut_ptr(), max_vec);
                        for k in 0..8 {
                            max_val = max_val.max(max_arr[k]);
                        }
                        for k in j..dim_size {
                            max_val = max_val.max(x_row[k]);
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

            // Fused pass: compute exp(x-max), store to output, and accumulate sum
            let mut sum_exp = 0.0f32;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        let max_vec = _mm256_set1_ps(max_val);
                        let mut sum_vec = _mm256_setzero_ps();
                        let mut j = 0;
                        while j + 8 <= dim_size {
                            let x_vec = _mm256_loadu_ps(x_row.as_ptr().add(j));
                            let shifted = _mm256_sub_ps(x_vec, max_vec);
                            let exp_vec = fast_exp_avx2(shifted);
                            sum_vec = _mm256_add_ps(sum_vec, exp_vec);

                            _mm256_storeu_ps(out_row.as_mut_ptr().add(j), exp_vec);
                            j += 8;
                        }
                        sum_exp = hsum256_ps(sum_vec);
                        for j2 in j..dim_size {
                            let e = (x_row[j2] - max_val).exp();
                            sum_exp += e;
                            out_row[j2] = e;
                        }
                    }
                } else {
                    for j in 0..dim_size {
                        let e = (x_row[j] - max_val).exp();
                        sum_exp += e;
                        out_row[j] = e;
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for j in 0..dim_size {
                    let e = (x_row[j] - max_val).exp();
                    sum_exp += e;
                    out_row[j] = e;
                }
            }

            let inv_sum = 1.0 / sum_exp;

            // Normalize: multiply stored exp values by inv_sum
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        let inv_vec = _mm256_set1_ps(inv_sum);
                        let mut j = 0;
                        while j + 8 <= dim_size {
                            let exp_vec = _mm256_loadu_ps(out_row.as_ptr().add(j));
                            let result = _mm256_mul_ps(exp_vec, inv_vec);
                            _mm256_storeu_ps(out_row.as_mut_ptr().add(j), result);
                            j += 8;
                        }
                        for j2 in j..dim_size {
                            out_row[j2] *= inv_sum;
                        }
                    }
                } else {
                    for j in 0..dim_size {
                        out_row[j] *= inv_sum;
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for j in 0..dim_size {
                    out_row[j] *= inv_sum;
                }
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

    let dim_size = x_shape[dim] as usize;

    // Fast path: last-dim, contiguous - fused kernel avoids intermediate tensors
    if dim == ndim - 1 && x.is_contiguous() {
        return vec![log_softmax_last_dim_fused(x, dim_size)];
    }

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

#[inline]
fn log_softmax_last_dim_fused(x: &Tensor, dim_size: usize) -> Tensor {
    let x_shape = x.shape();
    let x_ptr = x.data_ptr() as *const f32;
    let numel = x.numel() as usize;

    let mut output = Tensor::zeros(x_shape.to_vec(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    #[cfg(feature = "parallel")]
    {
        let a_slice = unsafe { std::slice::from_raw_parts(x_ptr, numel) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

        out_slice
            .par_chunks_mut(dim_size)
            .zip(a_slice.par_chunks(dim_size))
            .for_each(|(out_row, x_row)| {
                // Pass 1: Find max
                let mut max_val = f32::MIN;
                for j in 0..dim_size {
                    max_val = max_val.max(x_row[j]);
                }

                // Pass 2: Compute sum of exp(x - max) and store (x - max) in output
                let mut sum_exp = 0.0f32;
                for j in 0..dim_size {
                    let shifted = x_row[j] - max_val;
                    sum_exp += shifted.exp();
                    out_row[j] = shifted;
                }

                // Pass 3: log_softmax = (x - max) - log(sum_exp)
                let log_sum = sum_exp.ln();
                for j in 0..dim_size {
                    out_row[j] -= log_sum;
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
                let shifted = x_row[j] - max_val;
                sum_exp += shifted.exp();
                out_row[j] = shifted;
            }

            let log_sum = sum_exp.ln();
            for j in 0..dim_size {
                out_row[j] -= log_sum;
            }
        }
    }

    output
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
        "mean" => {
            // Flatten then single sum instead of O(dims) kernel dispatches
            let numel = loss.numel() as usize;
            let flat = loss.reshape(vec![numel as i64]);
            let mut result = flat.sum(0, false);
            result.mul_scalar_(1.0 / numel as f32);
            vec![result]
        }
        "sum" => {
            // Flatten then single sum
            let numel = loss.numel() as usize;
            let flat = loss.reshape(vec![numel as i64]);
            vec![flat.sum(0, false)]
        }
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

    // Fused log-sum-exp forward pass, parallelized over batch rows.
    // For each row i:
    //   max_val = max(logits[i])
    //   sum_exp = sum(exp(logits[i][j] - max_val))
    //   loss[i] = -(logits[i][target[i]] - max_val - ln(sum_exp))
    let mut losses = vec![0.0f32; batch_size];

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let logits_usize = logits_data.as_ptr() as usize;
        let targets_usize = targets_data.as_ptr() as usize;
        let losses_usize = losses.as_mut_ptr() as usize;
        let nc = num_classes;

        (0..batch_size).into_par_iter().for_each(|b| {
            let base = b * nc;
            let mut max_val = f32::NEG_INFINITY;
            let mut j = 0;

            // Find max with 4-unroll
            while j + 4 <= nc {
                unsafe {
                    let v0 = *((logits_usize + (base + j) * 4) as *const f32);
                    let v1 = *((logits_usize + (base + j + 1) * 4) as *const f32);
                    let v2 = *((logits_usize + (base + j + 2) * 4) as *const f32);
                    let v3 = *((logits_usize + (base + j + 3) * 4) as *const f32);
                    max_val = max_val.max(v0).max(v1).max(v2).max(v3);
                }
                j += 4;
            }
            while j < nc {
                unsafe {
                    max_val = max_val.max(*((logits_usize + (base + j) * 4) as *const f32));
                }
                j += 1;
            }

            // sum_exp with 4-unroll
            let mut sum_exp = 0.0f32;
            j = 0;
            while j + 4 <= nc {
                unsafe {
                    sum_exp += (*((logits_usize + (base + j) * 4) as *const f32) - max_val).exp();
                    sum_exp +=
                        (*((logits_usize + (base + j + 1) * 4) as *const f32) - max_val).exp();
                    sum_exp +=
                        (*((logits_usize + (base + j + 2) * 4) as *const f32) - max_val).exp();
                    sum_exp +=
                        (*((logits_usize + (base + j + 3) * 4) as *const f32) - max_val).exp();
                }
                j += 4;
            }
            while j < nc {
                unsafe {
                    sum_exp += (*((logits_usize + (base + j) * 4) as *const f32) - max_val).exp();
                }
                j += 1;
            }

            let target_class = unsafe { *((targets_usize + b * 4) as *const f32) } as usize;
            let log_sum_exp = sum_exp.ln();
            let class_logit =
                unsafe { *((logits_usize + (base + target_class) * 4) as *const f32) };
            let loss = log_sum_exp + max_val - class_logit;
            unsafe {
                *((losses_usize + b * 4) as *mut f32) = if loss.is_finite() { loss } else { 0.0 };
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for b in 0..batch_size {
            let base = b * num_classes;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..num_classes {
                max_val = max_val.max(logits_data[base + j]);
            }

            let mut sum_exp = 0.0f32;
            for j in 0..num_classes {
                sum_exp += (logits_data[base + j] - max_val).exp();
            }

            let target_class = targets_data[b] as usize;
            let log_sum_exp = sum_exp.ln();
            let class_logit = logits_data[base + target_class];
            let loss = log_sum_exp + max_val - class_logit;
            losses[b] = if loss.is_finite() { loss } else { 0.0 };
        }
    }

    match reduction {
        "none" => {
            let output = Tensor::from_vec(losses, vec![batch_size as i64]);
            vec![output]
        }
        "mean" => {
            let total_loss: f32 = losses.iter().sum();
            vec![Tensor::from_scalar(total_loss / batch_size as f32)]
        }
        _ => {
            let total_loss: f32 = losses.iter().sum();
            vec![Tensor::from_scalar(total_loss)]
        }
    }
}

/// Cross-entropy backward: computes grad_logits = softmax(logits) - one_hot(target),
/// scaled by grad_output / batch_size for mean reduction.
/// Writes directly into a pre-allocated output buffer, parallelized over batch rows.
pub fn cross_entropy_backward_f32(
    logits_data: &[f32],
    targets_data: &[f32],
    grad_out: f32,
    batch_size: usize,
    num_classes: usize,
    reduction: &str,
    grad_logits_data: &mut [f32],
) {
    let scale = if reduction == "mean" {
        grad_out / batch_size as f32
    } else {
        grad_out
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let logits_usize = logits_data.as_ptr() as usize;
        let targets_usize = targets_data.as_ptr() as usize;
        let grad_usize = grad_logits_data.as_mut_ptr() as usize;
        let nc = num_classes;

        (0..batch_size).into_par_iter().for_each(|b| {
            let base = b * nc;
            let target_class = unsafe { *((targets_usize + b * 4) as *const f32) } as usize;

            // Find max
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..nc {
                unsafe {
                    max_val = max_val.max(*((logits_usize + (base + j) * 4) as *const f32));
                }
            }

            // Compute sum_exp
            let mut sum_exp = 0.0f32;
            for j in 0..nc {
                unsafe {
                    sum_exp += (*((logits_usize + (base + j) * 4) as *const f32) - max_val).exp();
                }
            }

            // Guard against degenerate inputs (all logits = -inf → sum_exp = 0)
            if sum_exp == 0.0 || !sum_exp.is_finite() {
                for j in 0..nc {
                    unsafe {
                        *((grad_usize + (base + j) * 4) as *mut f32) = 0.0;
                    }
                }
                return;
            }

            let inv_sum = scale / sum_exp;

            // Write gradient: softmax - one_hot, scaled
            for j in 0..nc {
                unsafe {
                    let p = (logits_usize + (base + j) * 4) as *const f32;
                    let grad = (*p - max_val).exp() * inv_sum
                        - if j == target_class { scale } else { 0.0 };
                    *((grad_usize + (base + j) * 4) as *mut f32) = grad;
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for b in 0..batch_size {
            let base = b * num_classes;
            let target_class = targets_data[b] as usize;

            let mut max_val = f32::NEG_INFINITY;
            for j in 0..num_classes {
                max_val = max_val.max(logits_data[base + j]);
            }

            let mut sum_exp = 0.0f32;
            for j in 0..num_classes {
                sum_exp += (logits_data[base + j] - max_val).exp();
            }

            if sum_exp == 0.0 || !sum_exp.is_finite() {
                for j in 0..num_classes {
                    grad_logits_data[base + j] = 0.0;
                }
                continue;
            }

            let inv_sum = scale / sum_exp;

            for j in 0..num_classes {
                let grad = (logits_data[base + j] - max_val).exp() * inv_sum
                    - if j == target_class { scale } else { 0.0 };
                grad_logits_data[base + j] = grad;
            }
        }
    }
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
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

    #[cfg(feature = "parallel")]
    {
        if batch_size > 1 {
            use rayon::prelude::*;
            let col_rows_per_batch = out_height * out_width;
            let x_usize = x_ptr as usize;
            let col_usize = col_data.as_mut_ptr() as usize;

            (0..batch_size).into_par_iter().for_each(|n| {
                unsafe {
                    let x_p = x_usize as *const f32;
                    let col_p = (col_usize as *mut f32).add(n * col_rows_per_batch * col_cols);
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let col_row = oh * out_width + ow;
                            for ic in 0..in_channels {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let ih = oh * stride + kh * dilation;
                                        let iw = ow * stride + kw * dilation;
                                        let col_col =
                                            ((ic * kernel_height) + kh) * kernel_width + kw;
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
                                            *col_p.add(col_row * col_cols + col_col) =
                                                *x_p.add(x_idx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } // end unsafe
            });
        } else {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let col_row = oh * out_width + ow;
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
                                    let x_idx = (ic * in_height + x_ih) * in_width + x_iw;
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

    Tensor::from_vec(col_data, vec![col_rows as i64, col_cols as i64])
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn simd_dot_product(a: &[f32], b: &[f32], len: usize) -> f32 {
    ensure_daz_ftz();
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
                let mut acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                _mm256_storeu_ps(acc_arr.as_mut_ptr() as *mut f32, acc);
                // SAFETY: All 8 lanes are initialized by _mm256_storeu_ps above.
                sum += acc_arr.assume_init_ref().iter().sum::<f32>();

                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                    let prod = _mm256_mul_ps(a_vec, b_vec);
                    let mut prod_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                    _mm256_storeu_ps(prod_arr.as_mut_ptr() as *mut f32, prod);
                    // SAFETY: All 8 lanes are initialized by _mm256_storeu_ps above.
                    sum += prod_arr.assume_init_ref().iter().sum::<f32>();
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
                let mut acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                _mm256_storeu_ps(acc_arr.as_mut_ptr() as *mut f32, acc);
                // SAFETY: All 8 lanes are initialized by _mm256_storeu_ps above.
                sum += acc_arr.assume_init_ref().iter().sum::<f32>();

                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                    let prod = _mm256_mul_ps(a_vec, b_vec);
                    let mut prod_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                    _mm256_storeu_ps(prod_arr.as_mut_ptr() as *mut f32, prod);
                    // SAFETY: All 8 lanes are initialized by _mm256_storeu_ps above.
                    sum += prod_arr.assume_init_ref().iter().sum::<f32>();
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

#[allow(clippy::too_many_arguments)]
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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let n = batch_size * in_height * in_width; // Total spatial positions
    let k = in_channels;
    let m = out_channels;

    let w_data = unsafe { std::slice::from_raw_parts(w_ptr, m * k) };
    let bias_data: Option<&[f32]> = bias.map(|b| {
        let b_ptr = b.data_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(b_ptr, m) }
    });

    // Thread-local scratch: w_t [k*m] + x_t [n*k] + result [n*m]
    let total_scratch = k * m + n * k + n * m;

    CONV_SCRATCH.with(|scratch| {
        let mut buf = scratch.borrow_mut();
        if buf.len() < total_scratch {
            buf.resize(total_scratch, 0.0f32);
        }

        let (w_t_buf, rest) = buf.split_at_mut(k * m);
        let (x_t_buf, result_buf) = rest.split_at_mut(n * k);

        // Transpose weights: [out_ch, in_ch] -> [in_ch, out_ch]
        for i in 0..k {
            for j in 0..m {
                w_t_buf[i * m + j] = w_data[j * k + i];
            }
        }

        // Transpose input: [batch, in_ch, h, w] -> [batch * h * w, in_ch]
        let spatial_size = in_height * in_width;
        for b in 0..batch_size {
            for ic in 0..k {
                for s in 0..spatial_size {
                    let src_idx = (b * k + ic) * spatial_size + s;
                    let dst_idx = b * spatial_size * k + s * k + ic;
                    x_t_buf[dst_idx] = unsafe { *x_ptr.add(src_idx) };
                }
            }
        }

        // Use BLAS for [n, k] @ [k, m] = [n, m]
        let result_slice = &mut result_buf[..n * m];
        matmul_blas_with_transpose_into(x_t_buf, w_t_buf, result_slice, n, k, m, false, false);

        // Reshape result [n, m] -> [batch, out_ch, h, w] and add bias
        for b in 0..batch_size {
            for oc in 0..m {
                let bval = bias_data.map_or(0.0, |b| b[oc]);
                for s in 0..spatial_size {
                    let src_idx = b * spatial_size * m + s * m + oc;
                    let dst_idx = (b * m + oc) * spatial_size + s;
                    unsafe { *out_ptr.add(dst_idx) = result_slice[src_idx] + bval };
                }
            }
        }

        output
    })
}

// Winograd F(2x2, 3x3) is disabled for now due to overhead from transform operations
// The im2col + BLAS approach is faster for current use cases

#[allow(clippy::too_many_arguments)]
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

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    // Use direct pointers instead of copying data
    let w_ptr = w.data_ptr() as *const f32;
    let x_ptr = x.data_ptr() as *const f32;
    let bias_ptr = bias.map(|b| b.data_ptr() as *const f32);

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let total = batch_size * in_channels * out_height * out_width;
        let x_usize = x_ptr as usize;
        let w_usize = w_ptr as usize;
        let bias_usize = bias_ptr.map(|p| p as usize);

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
                            unsafe {
                                sum += *(x_usize as *const f32).add(x_idx)
                                    * *(w_usize as *const f32).add(w_idx);
                            }
                        }
                    }
                }

                if let Some(b) = bias_usize {
                    unsafe {
                        sum += *(b as *const f32).add(ic);
                    }
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
                                    sum += unsafe { *x_ptr.add(x_idx) * *w_ptr.add(w_idx) };
                                }
                            }
                        }

                        if let Some(b) = bias_ptr {
                            sum += unsafe { *b.add(ic) };
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

#[allow(clippy::too_many_arguments)]
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
    let col_size = col_rows * col_cols;

    let x_ptr = x.data_ptr() as *const f32;

    // Use optimized matrix multiplication for large matrices
    // Threshold: use GEMM when all dimensions >= 12
    const GEMM_MIN_SIZE: usize = 12;
    let use_gemm =
        col_rows >= GEMM_MIN_SIZE && out_channels >= GEMM_MIN_SIZE && col_cols >= GEMM_MIN_SIZE;

    let gemm_size = if use_gemm { col_rows * out_channels } else { 0 };
    let total_scratch = col_size + gemm_size;

    // Borrow the thread-local scratch buffer, growing only on the cold path.
    CONV_SCRATCH.with(|scratch| {
        let mut buf = scratch.borrow_mut();
        if buf.len() < total_scratch {
            buf.resize(total_scratch, 0.0f32);
        }
        // Zero only the im2col portion. gemm_out is fully overwritten by BLAS.
        buf[..col_size].fill(0.0);
        if gemm_size > 0 {
            buf[col_size..col_size + gemm_size].fill(0.0);
        }

        let (col_buf, gemm_buf) = buf.split_at_mut(col_size);
        let col_data: &mut [f32] = col_buf;

        // --- im2col extraction ---
        // Loop order: (ic, kh, kw) instead of (ic, kh, kw).
        // ic is innermost so that col_data writes are sequential for each
        // kernel position, and when stride==1 && dilation==1, contiguous
        // kernel_w floats can be copied from x in one memcpy.
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

                    // Fast path: stride=1, dilation=1, entire kernel patch in-bounds.
                    // For fixed (n, oh, ow, kh), the kernel_w elements x[n,ic,ih,ow..ow+kw]
                    // are contiguous when stride=1, dilation=1. We can memcpy each row.
                    let fast_path = stride == 1 && dilation == 1;

                    for ic in 0..in_channels {
                        let col_base = ic * kernel_height * kernel_width;

                        if fast_path {
                            // Optimized path: stride=1, dilation=1
                            for kh in 0..kernel_height {
                                let ih = oh + kh;

                                if ih >= padding && ih < padding + in_height {
                                    let x_ih = ih - padding;
                                    let x_row_base = (n * in_channels + ic) * in_height * in_width
                                        + x_ih * in_width;

                                    let iw_start = ow;
                                    if iw_start >= padding
                                        && iw_start + kernel_width <= padding + in_width
                                    {
                                        // Entire row in-bounds: use memcpy
                                        let x_iw_start = iw_start - padding;
                                        let x_src = x_row_base + x_iw_start;
                                        let col_dst_base = col_base + kh * kernel_width;
                                        unsafe {
                                            std::ptr::copy_nonoverlapping(
                                                (x_usize as *const f32).add(x_src),
                                                col_chunk.as_mut_ptr().add(col_dst_base),
                                                kernel_width,
                                            );
                                        }
                                    } else {
                                        // Boundary: per-element
                                        for kw in 0..kernel_width {
                                            let iw = ow + kw;
                                            if iw >= padding && iw < padding + in_width {
                                                let x_iw = iw - padding;
                                                let col_col = col_base + kh * kernel_width + kw;
                                                unsafe {
                                                    col_chunk[col_col] = *((x_usize as *const f32)
                                                        .add(x_row_base + x_iw));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            // General path: arbitrary stride/dilation
                            for kh in 0..kernel_height {
                                let ih = oh * stride + kh * dilation;

                                if ih >= padding && ih < padding + in_height {
                                    let x_ih = ih - padding;
                                    let x_row_base = (n * in_channels + ic) * in_height * in_width
                                        + x_ih * in_width;

                                    for kw in 0..kernel_width {
                                        let iw = ow * stride + kw * dilation;
                                        if iw >= padding && iw < padding + in_width {
                                            let x_iw = iw - padding;
                                            let col_col = col_base + kh * kernel_width + kw;
                                            unsafe {
                                                col_chunk[col_col] = *((x_usize as *const f32)
                                                    .add(x_row_base + x_iw));
                                            }
                                        }
                                    }
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
                        let col_chunk = &mut col_data[col_row * col_cols..(col_row + 1) * col_cols];

                        let fast_path = stride == 1 && dilation == 1;

                        for ic in 0..in_channels {
                            let col_base = ic * kernel_height * kernel_width;

                            if fast_path {
                                for kh in 0..kernel_height {
                                    let ih = oh + kh;

                                    if ih >= padding && ih < padding + in_height {
                                        let x_ih = ih - padding;
                                        let x_row_base =
                                            (n * in_channels + ic) * in_height * in_width
                                                + x_ih * in_width;

                                        let iw_start = ow;
                                        if iw_start >= padding
                                            && iw_start + kernel_width <= padding + in_width
                                        {
                                            let x_iw_start = iw_start - padding;
                                            let x_src = x_row_base + x_iw_start;
                                            let col_dst_base = col_base + kh * kernel_width;
                                            unsafe {
                                                std::ptr::copy_nonoverlapping(
                                                    x_ptr.add(x_src),
                                                    col_chunk.as_mut_ptr().add(col_dst_base),
                                                    kernel_width,
                                                );
                                            }
                                        } else {
                                            for kw in 0..kernel_width {
                                                let iw = ow + kw;
                                                if iw >= padding && iw < padding + in_width {
                                                    let x_iw = iw - padding;
                                                    let col_col = col_base + kh * kernel_width + kw;
                                                    col_chunk[col_col] =
                                                        unsafe { *x_ptr.add(x_row_base + x_iw) };
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                for kh in 0..kernel_height {
                                    let ih = oh * stride + kh * dilation;

                                    if ih >= padding && ih < padding + in_height {
                                        let x_ih = ih - padding;
                                        let x_row_base =
                                            (n * in_channels + ic) * in_height * in_width
                                                + x_ih * in_width;

                                        for kw in 0..kernel_width {
                                            let iw = ow * stride + kw * dilation;
                                            if iw >= padding && iw < padding + in_width {
                                                let x_iw = iw - padding;
                                                let col_col = col_base + kh * kernel_width + kw;
                                                col_chunk[col_col] =
                                                    unsafe { *x_ptr.add(x_row_base + x_iw) };
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Weight data: direct borrow, no copy. BLAS only reads from it.
        let w_data: &[f32] = unsafe {
            let w_ptr = w.data_ptr() as *const f32;
            std::slice::from_raw_parts(
                w_ptr,
                out_channels * in_channels * kernel_height * kernel_width / groups,
            )
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
        let out_data = Arc::make_mut(&mut cpu_storage.data);
        let out_ptr = out_data.as_mut_ptr() as *mut f32;

        let col_slice: &[f32] = col_data;

        let bias_data: Option<Vec<f32>> = if let Some(b) = bias {
            if b.numel() == 1 {
                let bias_val = b.item();
                Some(vec![bias_val; out_channels])
            } else {
                let b_ptr = b.data_ptr() as *const f32;
                Some(unsafe { std::slice::from_raw_parts(b_ptr, out_channels).to_vec() })
            }
        } else {
            None
        };

        if use_gemm {
            let gemm_out = &mut gemm_buf[..col_rows * out_channels];

            matmul_blas_with_transpose_into(
                col_slice,
                w_data,
                gemm_out,
                col_rows,
                col_cols,
                out_channels,
                false,
                true,
            );

            // Parallel NHWC -> NCHW layout conversion + bias addition.
            let spatial = out_height * out_width;
            let oc_usize = out_channels;
            let out_usize = out_ptr as usize;
            let gemm_usize = gemm_out.as_ptr() as usize;

            if let Some(ref b) = bias_data {
                let b_usize = b.as_ptr() as usize;
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    (0..col_rows).into_par_iter().for_each(|row| {
                        let n = row / spatial;
                        let sp = row % spatial;
                        for oc in 0..oc_usize {
                            let out_idx = (n * oc_usize + oc) * spatial + sp;
                            unsafe {
                                let val = *((gemm_usize + (row * oc_usize + oc) * 4) as *const f32)
                                    + *((b_usize + oc * 4) as *const f32);
                                *((out_usize + out_idx * 4) as *mut f32) = val;
                            }
                        }
                    });
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for row in 0..col_rows {
                        let n = row / spatial;
                        let sp = row % spatial;
                        for oc in 0..oc_usize {
                            let out_idx = (n * oc_usize + oc) * spatial + sp;
                            unsafe {
                                *((out_usize + out_idx * 4) as *mut f32) =
                                    *((gemm_usize + (row * oc_usize + oc) * 4) as *const f32)
                                        + *((b_usize + oc * 4) as *const f32);
                            }
                        }
                    }
                }
            } else {
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    (0..col_rows).into_par_iter().for_each(|row| {
                        let n = row / spatial;
                        let sp = row % spatial;
                        for oc in 0..oc_usize {
                            let out_idx = (n * oc_usize + oc) * spatial + sp;
                            unsafe {
                                *((out_usize + out_idx * 4) as *mut f32) =
                                    *((gemm_usize + (row * oc_usize + oc) * 4) as *const f32);
                            }
                        }
                    });
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for row in 0..col_rows {
                        let n = row / spatial;
                        let sp = row % spatial;
                        for oc in 0..oc_usize {
                            let out_idx = (n * oc_usize + oc) * spatial + sp;
                            unsafe {
                                *((out_usize + out_idx * 4) as *mut f32) =
                                    *((gemm_usize + (row * oc_usize + oc) * 4) as *const f32);
                            }
                        }
                    }
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
    })
}

/// Fused layer norm forward: single-pass mean/variance, normalize, apply weight/bias.
/// Returns [output, mean, variance, x_hat].
/// Parallelized over outer dimensions (rows) with rayon.
/// Zero intermediate tensor allocations — writes directly into output buffers.
fn layer_norm_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let _normalized_shape_arg = args[1].shape();
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
    let eps = if args.len() > 4 { args[4].item() } else { 1e-5 };

    let x_shape = x.shape();
    let ndim = x_shape.len();
    let norm_dim = x_shape[ndim - 1] as usize;

    // Number of outer dimensions (product of all dims except last)
    let outer_size: usize = x_shape[..ndim - 1].iter().map(|&d| d as usize).product();
    let total = outer_size * norm_dim;

    let x_data = x.as_f32_slice();

    let mut output_data = vec![0.0f32; total];
    let mut mean_data = vec![0.0f32; outer_size];
    let mut var_data = vec![0.0f32; outer_size];
    let mut x_hat_data = vec![0.0f32; total];

    let w_data = weight.map(|w| w.as_f32_slice());
    let b_data = bias.map(|b| b.as_f32_slice());
    let nd = norm_dim;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let x_usize = x_data.as_ptr() as usize;
        let out_usize = output_data.as_mut_ptr() as usize;
        let xhat_usize = x_hat_data.as_mut_ptr() as usize;
        let mean_usize = mean_data.as_mut_ptr() as usize;
        let var_usize = var_data.as_mut_ptr() as usize;

        (0..outer_size).into_par_iter().for_each(|row| {
            let base = row * nd;

            // Two-pass: mean then variance (more numerically stable than single-pass)
            let mut sum = 0.0f32;
            for j in 0..nd {
                unsafe {
                    sum += *((x_usize + (base + j) * 4) as *const f32);
                }
            }
            let mean = sum / nd as f32;

            let mut sum_sq = 0.0f32;
            for j in 0..nd {
                unsafe {
                    let diff = *((x_usize + (base + j) * 4) as *const f32) - mean;
                    sum_sq += diff * diff;
                }
            }
            let var = sum_sq / nd as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            unsafe {
                *((mean_usize + row * 4) as *mut f32) = mean;
                *((var_usize + row * 4) as *mut f32) = var;
            }

            // Normalize, apply weight/bias, store x_hat and output
            for j in 0..nd {
                unsafe {
                    let val = *((x_usize + (base + j) * 4) as *const f32);
                    let xn = (val - mean) * inv_std;
                    *((xhat_usize + (base + j) * 4) as *mut f32) = xn;
                    let mut out_val = xn;
                    if let Some(w) = w_data {
                        out_val *= w[j];
                    }
                    if let Some(b) = b_data {
                        out_val += b[j];
                    }
                    *((out_usize + (base + j) * 4) as *mut f32) = out_val;
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..outer_size {
            let base = row * nd;
            let mut sum = 0.0f32;
            for j in 0..nd {
                sum += x_data[base + j];
            }
            let mean = sum / nd as f32;

            let mut sum_sq = 0.0f32;
            for j in 0..nd {
                let diff = x_data[base + j] - mean;
                sum_sq += diff * diff;
            }
            let var = sum_sq / nd as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            mean_data[row] = mean;
            var_data[row] = var;

            for j in 0..nd {
                let xn = (x_data[base + j] - mean) * inv_std;
                x_hat_data[base + j] = xn;
                let mut out_val = xn;
                if let Some(w) = w_data {
                    out_val *= w[j];
                }
                if let Some(b) = b_data {
                    out_val += b[j];
                }
                output_data[base + j] = out_val;
            }
        }
    }

    // Reshape mean and variance to [outer_size, 1] for broadcasting compatibility
    let mut mean_shape = x_shape[..ndim - 1].to_vec();
    mean_shape.push(1);
    let var_shape = mean_shape.clone();

    let output = Tensor::from_vec(output_data, x_shape.clone());
    let mean = Tensor::from_vec(mean_data, mean_shape);
    let var = Tensor::from_vec(var_data, var_shape);
    let x_hat = Tensor::from_vec(x_hat_data, x_shape.clone());

    vec![output, mean, var, x_hat]
}

/// Fused layer norm backward: computes dX, dW, dB without intermediate tensors.
/// Parallelized over outer dimensions (rows) for dX computation.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward_f32(
    grad_data: &[f32],
    x_hat_data: &[f32],
    weight_data: Option<&[f32]>,
    outer_size: usize,
    norm_dim: usize,
    eps: f32,
    var_data: &[f32],
    grad_input_data: &mut [f32],
    grad_weight_data: &mut [f32],
    grad_bias_data: &mut [f32],
) {
    let nd = norm_dim;

    // Zero weight/bias grads
    for j in 0..nd {
        grad_weight_data[j] = 0.0;
        grad_bias_data[j] = 0.0;
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let grad_usize = grad_data.as_ptr() as usize;
        let xhat_usize = x_hat_data.as_ptr() as usize;
        let ginput_usize = grad_input_data.as_mut_ptr() as usize;
        let var_usize = var_data.as_ptr() as usize;

        // Parallel dX computation
        (0..outer_size).into_par_iter().for_each(|row| {
            let base = row * nd;
            let inv_std = 1.0 / (unsafe { *((var_usize + row * 4) as *const f32) } + eps).sqrt();

            // Compute sum(dY * weight) and sum(dY * weight * x_hat)
            let mut sum_gw = 0.0f32;
            let mut sum_gw_xh = 0.0f32;
            for j in 0..nd {
                unsafe {
                    let g = *((grad_usize + (base + j) * 4) as *const f32);
                    let xh = *((xhat_usize + (base + j) * 4) as *const f32);
                    let gw = if let Some(w) = weight_data {
                        g * w[j]
                    } else {
                        g
                    };
                    sum_gw += gw;
                    sum_gw_xh += gw * xh;
                }
            }
            let mean_gw = sum_gw / nd as f32;
            let mean_gw_xh = sum_gw_xh / nd as f32;

            // dX
            for j in 0..nd {
                unsafe {
                    let g = *((grad_usize + (base + j) * 4) as *const f32);
                    let xh = *((xhat_usize + (base + j) * 4) as *const f32);
                    let gw = if let Some(w) = weight_data {
                        g * w[j]
                    } else {
                        g
                    };
                    let dx = (gw - mean_gw - xh * mean_gw_xh) * inv_std;
                    *((ginput_usize + (base + j) * 4) as *mut f32) = dx;
                }
            }
        });

        // Sequential accumulation for dW and dB (small array, not worth parallel reduction)
        for row in 0..outer_size {
            let base = row * nd;
            for j in 0..nd {
                unsafe {
                    let g = *((grad_usize + (base + j) * 4) as *const f32);
                    let xh = *((xhat_usize + (base + j) * 4) as *const f32);
                    grad_bias_data[j] += g;
                    grad_weight_data[j] += g * xh;
                }
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..outer_size {
            let base = row * nd;
            let inv_std = 1.0 / (var_data[row] + eps).sqrt();

            let mut sum_gw = 0.0f32;
            let mut sum_gw_xh = 0.0f32;
            for j in 0..nd {
                let g = grad_data[base + j];
                let xh = x_hat_data[base + j];
                let gw = if let Some(w) = weight_data {
                    g * w[j]
                } else {
                    g
                };
                sum_gw += gw;
                sum_gw_xh += gw * xh;
            }
            let mean_gw = sum_gw / nd as f32;
            let mean_gw_xh = sum_gw_xh / nd as f32;

            for j in 0..nd {
                let g = grad_data[base + j];
                let xh = x_hat_data[base + j];
                let gw = if let Some(w) = weight_data {
                    g * w[j]
                } else {
                    g
                };
                grad_input_data[base + j] = (gw - mean_gw - xh * mean_gw_xh) * inv_std;

                grad_bias_data[j] += g;
                grad_weight_data[j] += g * xh;
            }
        }
    }
}

fn max_pool2d_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let kernel_size = if args.len() > 1 {
        args[1].item() as i64
    } else {
        2
    };
    let stride = if args.len() > 2 {
        args[2].item() as i64
    } else {
        kernel_size
    };
    let padding = if args.len() > 3 {
        args[3].item() as i64
    } else {
        0
    };
    let dilation = if args.len() > 4 {
        args[4].item() as i64
    } else {
        1
    };

    let x_shape = x.shape();
    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_height = x_shape[2];
    let in_width = x_shape[3];

    let out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    let out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    let output = Tensor::zeros(
        vec![batch_size, channels, out_height, out_width],
        x.dtype(),
        x.device(),
    );

    let x_ptr = x.data_ptr() as *const f32;
    let out_ptr = output.data_ptr() as *mut f32;

    let total_bc = batch_size as usize * channels as usize;
    let stride_usize = stride as usize;
    let dilation_usize = dilation as usize;
    let kernel_size_usize = kernel_size as usize;
    let in_h = in_height as usize;
    let in_w = in_width as usize;
    let out_h = out_height as usize;
    let out_w = out_width as usize;
    let pad = padding;
    let channels_usize = channels as usize;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let x_usize = x_ptr as usize;
        let out_usize = out_ptr as usize;
        (0..total_bc).into_par_iter().for_each(|bc| {
            let b = bc / channels_usize;
            let c = bc % channels_usize;
            let x_p = x_usize as *const f32;
            let o_p = out_usize as *mut f32;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for kh in 0..kernel_size_usize {
                        for kw in 0..kernel_size_usize {
                            let h = (oh * stride_usize + kh * dilation_usize) as i64 - pad;
                            let w = (ow * stride_usize + kw * dilation_usize) as i64 - pad;
                            if h >= 0 && h < in_h as i64 && w >= 0 && w < in_w as i64 {
                                let idx = ((b * channels_usize + c) * in_h + h as usize) * in_w
                                    + w as usize;
                                let val = unsafe { *x_p.add(idx) };
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    let out_idx = ((b * channels_usize + c) * out_h + oh) * out_w + ow;
                    unsafe {
                        *o_p.add(out_idx) = max_val;
                    }
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for b in 0..batch_size as usize {
            for c in 0..channels as usize {
                for oh in 0..out_height as usize {
                    for ow in 0..out_width as usize {
                        let mut max_val = f32::NEG_INFINITY;

                        for kh in 0..kernel_size as usize {
                            for kw in 0..kernel_size as usize {
                                let h = (oh * stride as usize + kh * dilation as usize) as i64
                                    - padding;
                                let w = (ow * stride as usize + kw * dilation as usize) as i64
                                    - padding;

                                if h >= 0 && h < in_height && w >= 0 && w < in_width {
                                    let idx = ((b * channels as usize + c) * in_height as usize
                                        + h as usize)
                                        * in_width as usize
                                        + w as usize;
                                    let val = unsafe { *x_ptr.add(idx) };
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }

                        let out_idx = ((b * channels as usize + c) * out_height as usize + oh)
                            * out_width as usize
                            + ow;
                        unsafe { *out_ptr.add(out_idx) = max_val };
                    }
                }
            }
        }
    }

    vec![output]
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
    let running_mean = if args.len() > 3 && args[3].numel() > 0 {
        Some(args[3])
    } else {
        None
    };
    let running_var = if args.len() > 4 && args[4].numel() > 0 {
        Some(args[4])
    } else {
        None
    };
    let training = if args.len() > 5 {
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
    let ndim = x_shape.len();

    // BatchNorm normalizes across the channel dimension (dim=1)
    let num_channels = if ndim > 1 { x_shape[1] } else { 1 };
    let batch_size = x_shape[0];
    let spatial_size: i64 = if ndim > 2 {
        x_shape[2..].iter().product()
    } else {
        1
    };
    let total_per_channel = batch_size * spatial_size; // Elements per feature

    // Get weight and bias (default: gamma=1, beta=0)
    let w_data = weight.map(|w| w.as_f32_slice());
    let b_data = bias.map(|b| b.as_f32_slice());

    // Create output tensor with same shape as input
    let mut output = Tensor::empty(x_shape.clone(), x.dtype(), x.device());

    // Get raw pointers for fast access
    let x_ptr = x.data_ptr() as *const f32;
    let x_addr = x_ptr as usize; // Convert to usize for Sync
    let out_inner = Arc::make_mut(&mut output.inner);
    let out_storage = Arc::make_mut(&mut out_inner.storage);
    let Storage::Cpu(out_cpu) = out_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut out_cpu.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;
    let out_addr = out_ptr as usize; // Convert to usize for Sync

    // Helper function to compute mean/variance using Welford's algorithm
    // with optional SIMD acceleration
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    unsafe fn compute_stats_simd(
        x_ptr: *const f32,
        indices: &[(usize, usize)],
        n: usize,
    ) -> (f32, f32) {
        use std::arch::x86_64::*;
        ensure_daz_ftz();

        if is_x86_feature_detected!("avx2") && n >= 8 {
            // AVX2 SIMD path - process 8 elements at a time
            let mut sum = _mm256_setzero_ps();
            let mut sum_sq = _mm256_setzero_ps();

            // Process chunks of 8
            let chunks = n / 8;
            for i in 0..chunks {
                let mut vals = [0.0f32; 8];
                for j in 0..8 {
                    let (b, s) = indices[i * 8 + j];
                    vals[j] = *x_ptr.add(b + s);
                }
                let v = _mm256_loadu_ps(vals.as_ptr());
                sum = _mm256_add_ps(sum, v);
                sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
            }

            // Horizontal sum - use f64 accumulator to avoid catastrophic cancellation
            let mut sum_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
            _mm256_storeu_ps(sum_arr.as_mut_ptr() as *mut f32, sum);
            // SAFETY: All 8 lanes are initialized by _mm256_storeu_ps above.
            let mut total_sum: f64 = sum_arr.assume_init_ref().iter().map(|&x| x as f64).sum();

            let mut sum_sq_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
            _mm256_storeu_ps(sum_sq_arr.as_mut_ptr() as *mut f32, sum_sq);
            // SAFETY: All 8 lanes are initialized by _mm256_storeu_ps above.
            let mut total_sum_sq: f64 =
                sum_sq_arr.assume_init_ref().iter().map(|&x| x as f64).sum();

            // Handle remainder
            let _remainder = n % 8;
            let start = chunks * 8;
            for i in start..n {
                let (b, s) = indices[i];
                let val = *x_ptr.add(b + s);
                total_sum += val as f64;
                total_sum_sq += (val as f64) * (val as f64);
            }

            let mean = (total_sum / n as f64) as f32;
            let var = (total_sum_sq / n as f64) as f32 - mean * mean;
            (mean, var.max(0.0)) // Ensure non-negative variance
        } else {
            // Scalar fallback using Welford's algorithm
            let mut count = 0.0f32;
            let mut mean = 0.0f32;
            let mut m2 = 0.0f32;

            for &(b, s) in indices {
                let val = *x_ptr.add(b + s);
                count += 1.0;
                let delta = val - mean;
                mean += delta / count;
                let delta2 = val - mean;
                m2 += delta * delta2;
            }

            let var = m2 / count;
            (mean, var.max(0.0))
        }
    }

    // Non-SIMD version
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    #[inline]
    unsafe fn compute_stats_simd(
        x_ptr: *const f32,
        indices: &[(usize, usize)],
        n: usize,
    ) -> (f32, f32) {
        // Scalar fallback using Welford's algorithm
        let mut count = 0.0f32;
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;

        for &(b, s) in indices {
            let val = *x_ptr.add(b + s);
            count += 1.0;
            let delta = val - mean;
            mean += delta / count;
            let delta2 = val - mean;
            m2 += delta * delta2;
        }

        let var = m2 / count;
        (mean, var.max(0.0))
    }

    if training {
        // Training mode: compute batch statistics
        // Parallelize over channels using rayon
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            (0..num_channels as usize).into_par_iter().for_each(|c| {
                let x_ptr = x_addr as *const f32;
                let out_ptr = out_addr as *mut f32;

                // Pre-compute indices for this channel
                let mut indices = Vec::with_capacity(total_per_channel as usize);
                for b in 0..batch_size as usize {
                    let base = b * num_channels as usize * spatial_size as usize;
                    for s in 0..spatial_size as usize {
                        indices.push((base + c * spatial_size as usize, s));
                    }
                }

                // Compute mean and variance using SIMD-accelerated Welford
                let (mean, var) =
                    unsafe { compute_stats_simd(x_ptr, &indices, total_per_channel as usize) };
                let inv_std = 1.0 / (var + eps as f32).sqrt();

                // Get gamma and beta for this channel
                let gamma = w_data.map_or(1.0, |w| w[c]);
                let beta = b_data.map_or(0.0, |b| b[c]);

                // Normalize, scale, and shift in one pass
                for &(base, s) in &indices {
                    let idx = base + s;
                    let val = unsafe { *x_ptr.add(idx) };
                    let normed = (val - mean) * inv_std;
                    unsafe { *out_ptr.add(idx) = gamma * normed + beta };
                }
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for c in 0..num_channels as usize {
                // Pre-compute indices for this channel
                let mut indices = Vec::with_capacity(total_per_channel as usize);
                for b in 0..batch_size as usize {
                    let base = b * num_channels as usize * spatial_size as usize;
                    for s in 0..spatial_size as usize {
                        indices.push((base + c * spatial_size as usize, s));
                    }
                }

                // Compute mean and variance using SIMD-accelerated Welford
                let (mean, var) =
                    unsafe { compute_stats_simd(x_ptr, &indices, total_per_channel as usize) };
                let inv_std = 1.0 / (var + eps as f32).sqrt();

                // Get gamma and beta for this channel
                let gamma = w_data.map_or(1.0, |w| w[c]);
                let beta = b_data.map_or(0.0, |b| b[c]);

                // Normalize, scale, and shift in one pass
                for &(base, s) in &indices {
                    let idx = base + s;
                    let val = unsafe { *x_ptr.add(idx) };
                    let normed = (val - mean) * inv_std;
                    unsafe { *out_ptr.add(idx) = gamma * normed + beta };
                }
            }
        }
    } else {
        // Inference mode: use running statistics (no computation needed)
        let run_mean = running_mean.expect("running_mean required for inference");
        let run_var = running_var.expect("running_var required for inference");
        let mean_data = run_mean.as_f32_slice();
        let var_data = run_var.as_f32_slice();

        // Parallelize over channels
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            (0..num_channels as usize).into_par_iter().for_each(|c| {
                let x_ptr = x_addr as *const f32;
                let out_ptr = out_addr as *mut f32;

                let mean = mean_data[c];
                let inv_std = 1.0 / (var_data[c] + eps as f32).sqrt();
                let gamma = w_data.map_or(1.0, |w| w[c]);
                let beta = b_data.map_or(0.0, |b| b[c]);

                // Normalize, scale, and shift in one pass
                for b in 0..batch_size as usize {
                    for s in 0..spatial_size as usize {
                        let idx = (b * num_channels as usize + c) * spatial_size as usize + s;
                        let val = unsafe { *x_ptr.add(idx) };
                        let normed = (val - mean) * inv_std;
                        unsafe { *out_ptr.add(idx) = gamma * normed + beta };
                    }
                }
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for c in 0..num_channels as usize {
                let mean = mean_data[c];
                let inv_std = 1.0 / (var_data[c] + eps as f32).sqrt();
                let gamma = w_data.map_or(1.0, |w| w[c]);
                let beta = b_data.map_or(0.0, |b| b[c]);

                // Normalize, scale, and shift in one pass
                for b in 0..batch_size as usize {
                    for s in 0..spatial_size as usize {
                        let idx = (b * num_channels as usize + c) * spatial_size as usize + s;
                        let val = unsafe { *x_ptr.add(idx) };
                        let normed = (val - mean) * inv_std;
                        unsafe { *out_ptr.add(idx) = gamma * normed + beta };
                    }
                }
            }
        }
    }

    vec![output]
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

    let indices_ptr = indices.data_ptr() as *const f32;
    let weight_ptr = weight.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    for i in 0..batch_size as usize {
        let idx = unsafe { *indices_ptr.add(i) } as usize;
        if idx < num_embeddings as usize {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    weight_ptr.add(idx * embedding_dim as usize),
                    out_ptr.add(i * embedding_dim as usize),
                    embedding_dim as usize,
                );
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let Some(grad_output) = grad_outputs.into_iter().next().flatten() else {
            return vec![Some(Tensor::zeros(
                self.inputs[0].shape().clone(),
                self.inputs[0].dtype(),
                self.inputs[0].device(),
            ))];
        };
        let weight = &self.inputs[0];
        let indices = &self.inputs[1];

        let weight_shape = weight.shape().clone();
        let embedding_dim = weight_shape[1];
        let batch_size = grad_output.shape()[0];

        // Create gradient for weight (same shape as weight)
        let mut weight_grad = Tensor::zeros(weight_shape.clone(), weight.dtype(), weight.device());

        // Accumulate gradients from output
        let grad_output_ptr = grad_output.data_ptr() as *const f32;
        let indices_ptr = indices.data_ptr() as *const f32;
        let weight_grad_inner = Arc::make_mut(&mut weight_grad.inner);
        let weight_grad_storage = Arc::make_mut(&mut weight_grad_inner.storage);
        let Storage::Cpu(cpu_storage) = weight_grad_storage else {
            panic!("Expected CPU storage");
        };
        let weight_grad_data = Arc::make_mut(&mut cpu_storage.data);
        let weight_grad_ptr = weight_grad_data.as_mut_ptr() as *mut f32;

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
        1
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
    let start = args[0].item() as f64;
    let end = args[1].item() as f64;
    let step = if args.len() > 2 {
        args[2].item() as f64
    } else {
        1.0
    };

    let numel = ((end - start) / step).ceil() as usize;
    let values: Vec<f32> = (0..numel)
        .map(|i| (start + i as f64 * step) as f32)
        .collect();

    vec![Tensor::from_vec(values, vec![numel as i64])]
}

fn linspace_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let start = args[0].item();
    let end = args[1].item();
    let steps = args[2].item() as usize;

    if steps == 0 {
        return vec![Tensor::from_vec(vec![], vec![0i64])];
    }
    if steps == 1 {
        return vec![Tensor::from_vec(vec![start], vec![1i64])];
    }

    let values: Vec<f32> = (0..steps)
        .map(|i| {
            let t = i as f32 / (steps - 1) as f32;
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
    let numel: usize = shape.iter().product::<i64>() as usize;

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut values = vec![0.0f32; numel];

    // Box-Muller generates 2 normal samples from 2 uniform samples
    // Process in pairs to be 2x faster
    let mut i = 0;
    while i + 1 < numel {
        let u1: f32 = rng.gen::<f32>().max(1e-10); // avoid ln(0)
        let u2: f32 = rng.gen::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        values[i] = r * theta.cos();
        values[i + 1] = r * theta.sin();
        i += 2;
    }
    // Handle odd element
    if i < numel {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen::<f32>();
        values[i] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
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
        DType::BF16 => {
            let ptr = slice.as_ptr() as *const half::bf16;
            unsafe { f32::from(*ptr) }
        }
        DType::F16 => {
            let ptr = slice.as_ptr() as *const half::f16;
            unsafe { f32::from(*ptr) }
        }
        _ => 0.0,
    }
}

#[allow(dead_code)]
fn write_f32(slice: &[u8], val: f32, dtype: DType) {
    let ptr = slice.as_ptr() as *mut u8;
    unsafe {
        match dtype {
            DType::F32 => {
                *(ptr as *mut f32) = val;
            }
            DType::F64 => {
                *(ptr as *mut f64) = val as f64;
            }
            DType::I32 => {
                *(ptr as *mut i32) = val as i32;
            }
            DType::I64 => {
                *(ptr as *mut i64) = val as i64;
            }
            DType::BF16 => {
                *(ptr as *mut half::bf16) = half::bf16::from_f32(val);
            }
            DType::F16 => {
                *(ptr as *mut half::f16) = half::f16::from_f32(val);
            }
            _ => {}
        }
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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(numel);
                let a_p = a_usize as *const f32;
                let o_p = out_usize as *mut f32;
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            let thresh = _mm256_set1_ps(threshold);
                            let one = _mm256_set1_ps(1.0f32);
                            let zero = _mm256_set1_ps(0.0f32);
                            let mut i = start;
                            while i + 8 <= end {
                                let v = _mm256_loadu_ps(a_p.add(i));
                                let mask = _mm256_cmp_ps(v, thresh, _CMP_GT_OQ);
                                let r = _mm256_blendv_ps(zero, one, mask);
                                _mm256_storeu_ps(o_p.add(i), r);
                                i += 8;
                            }
                            for j in i..end {
                                *o_p.add(j) = if *a_p.add(j) > threshold { 1.0 } else { 0.0 };
                            }
                            return;
                        }
                    }
                }
                for j in start..end {
                    unsafe {
                        *o_p.add(j) = if *a_p.add(j) > threshold { 1.0 } else { 0.0 };
                    }
                }
            });
        }
    } else {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && numel >= 8 {
                unsafe {
                    let thresh = _mm256_set1_ps(threshold);
                    let one = _mm256_set1_ps(1.0f32);
                    let zero = _mm256_set1_ps(0.0f32);
                    let mut i = 0;
                    while i + 8 <= numel {
                        let v = _mm256_loadu_ps(a_ptr.add(i));
                        let mask = _mm256_cmp_ps(v, thresh, _CMP_GT_OQ);
                        let r = _mm256_blendv_ps(zero, one, mask);
                        _mm256_storeu_ps(out_ptr.add(i), r);
                        i += 8;
                    }
                    for j in i..numel {
                        *out_ptr.add(j) = if *a_ptr.add(j) > threshold { 1.0 } else { 0.0 };
                    }
                    return vec![output];
                }
            }
        }
        for idx in 0..numel {
            unsafe {
                *out_ptr.add(idx) = if *a_ptr.add(idx) > threshold {
                    1.0
                } else {
                    0.0
                };
            }
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
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(numel);
                for j in start..end {
                    unsafe {
                        let val = *(a_usize as *const f32).add(j);
                        *(out_usize as *mut f32).add(j) = if val > 0.0 {
                            1.0
                        } else if val < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                    }
                }
            });
        }
    } else {
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
    }

    vec![output]
}

fn maximum_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];
    let out_shape = broadcast_shapes_simple(&a.shape(), &b.shape());
    let numel = out_shape.iter().product::<i64>() as usize;
    let mut output = Tensor::zeros(out_shape.clone(), a.dtype(), a.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;
    let a_data = a.as_f32_slice();
    let b_data = b.as_f32_slice();
    let a_strides = &a.inner.strides;
    let b_strides = &b.inner.strides;
    let out_strides = &output.inner.strides;
    let ndim = out_shape.len();
    let a_offset = a.inner.storage_offset as usize;
    let b_offset = b.inner.storage_offset as usize;
    let mut indices = vec![0i64; ndim];
    for _out_idx in 0..numel {
        let mut a_idx: usize = a_offset;
        let mut b_idx: usize = b_offset;
        for d in 0..ndim {
            let a_dim_idx = if d >= ndim - a.ndim() {
                d - (ndim - a.ndim())
            } else {
                usize::MAX
            };
            let b_dim_idx = if d >= ndim - b.ndim() {
                d - (ndim - b.ndim())
            } else {
                usize::MAX
            };
            if a_dim_idx != usize::MAX && a.shape()[a_dim_idx] != 1 {
                a_idx +=
                    (indices[d] % a.shape()[a_dim_idx]) as usize * a_strides[a_dim_idx] as usize;
            }
            if b_dim_idx != usize::MAX && b.shape()[b_dim_idx] != 1 {
                b_idx +=
                    (indices[d] % b.shape()[b_dim_idx]) as usize * b_strides[b_dim_idx] as usize;
            }
        }
        let av = unsafe { *a_data.as_ptr().add(a_idx) };
        let bv = unsafe { *b_data.as_ptr().add(b_idx) };
        let mut out_linear: usize = 0;
        for d in 0..ndim {
            out_linear += indices[d] as usize * out_strides[d] as usize;
        }
        unsafe {
            *out_ptr.add(out_linear) = if av > bv { av } else { bv };
        }
        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < out_shape[d] {
                break;
            }
            indices[d] = 0;
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
    register(
        "leaky_relu",
        DispatchKey::Cpu,
        leaky_relu_kernel as KernelFn,
    );
    register("prelu", DispatchKey::Cpu, prelu_kernel as KernelFn);
    register("softplus", DispatchKey::Cpu, softplus_kernel as KernelFn);
    register("hardswish", DispatchKey::Cpu, hardswish_kernel as KernelFn);
    register("clamp", DispatchKey::Cpu, clamp_kernel as KernelFn);
    register("pow", DispatchKey::Cpu, pow_kernel as KernelFn);
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
    register("maximum", DispatchKey::Cpu, maximum_kernel as KernelFn);
    register("softmax", DispatchKey::Cpu, softmax_kernel as KernelFn);
    register(
        "log_softmax",
        DispatchKey::Cpu,
        log_softmax_kernel as KernelFn,
    );
    register("mse_loss", DispatchKey::Cpu, mse_loss_kernel as KernelFn);
    register(
        "bce_with_logits",
        DispatchKey::Cpu,
        bce_with_logits_kernel as KernelFn,
    );
    register(
        "huber_loss",
        DispatchKey::Cpu,
        huber_loss_kernel as KernelFn,
    );
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
        "conv_transpose2d",
        DispatchKey::Cpu,
        conv_transpose2d_kernel as KernelFn,
    );
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
    register(
        "max_pool2d",
        DispatchKey::Cpu,
        max_pool2d_kernel as KernelFn,
    );
    register("sign", DispatchKey::Cpu, sign_kernel as KernelFn);
    register("lt_scalar", DispatchKey::Cpu, lt_scalar_kernel as KernelFn);
    register(
        "add_scalar",
        DispatchKey::Cpu,
        add_scalar_kernel as KernelFn,
    );
    register(
        "div_scalar",
        DispatchKey::Cpu,
        div_scalar_kernel as KernelFn,
    );
    register(
        "logical_not",
        DispatchKey::Cpu,
        logical_not_kernel as KernelFn,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// Reference scalar matmul for a single batch: C = A @ B
    /// A is [m, k], B is [k, n], C is [m, n]
    fn reference_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

    #[test]
    fn test_parallel_matmul_fallback_3d_batched() {
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
    fn test_parallel_matmul_fallback_2d_small() {
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

    #[test]
    fn test_embedding_bulk_copy() {
        // Test that embedding forward produces correct results
        let num_embeddings: usize = 10;
        let embedding_dim: usize = 4;

        let weight_data: Vec<f32> = (0..num_embeddings * embedding_dim)
            .map(|i| i as f32 * 0.1)
            .collect();
        let weight = Tensor::from_vec(
            weight_data.clone(),
            vec![num_embeddings as i64, embedding_dim as i64],
        );

        // Indices: pick rows 3, 7, 1
        let indices_data = vec![3.0f32, 7.0, 1.0];
        let indices = Tensor::from_vec(indices_data, vec![3]);

        let result = crate::dispatcher::dispatch(
            "embedding",
            crate::dispatcher::DispatchKey::Cpu,
            &[&weight, &indices],
        );
        let result_data = result[0].as_f32_slice();

        // Expected: rows 3, 7, 1 from weight
        let mut expected = Vec::new();
        for &idx in &[3usize, 7, 1] {
            expected
                .extend_from_slice(&weight_data[idx * embedding_dim..(idx + 1) * embedding_dim]);
        }

        assert_eq!(result_data.len(), expected.len());
        for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "idx={}: got={}, expected={}",
                idx,
                got,
                exp
            );
        }
    }

    /// Benchmark: measure fused_linear_relu vs matmul+relu to quantify the gap.
    /// This test prints timing data and verifies correctness of both paths.
    #[test]
    fn bench_fused_linear_relu_vs_matmul() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};
        use std::time::Instant;

        let configs: Vec<(usize, usize, usize)> = vec![
            (32, 512, 512),   // small
            (32, 1024, 1024), // medium
            (64, 2048, 2048), // large
        ];

        for &(batch, in_feat, out_feat) in &configs {
            let x_data: Vec<f32> = (0..batch * in_feat)
                .map(|i| ((i as f32) * 0.001).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_feat * in_feat)
                .map(|i| ((i as f32) * 0.001).cos() * 0.1)
                .collect();

            let x = Tensor::from_vec(x_data, vec![batch as i64, in_feat as i64]);
            let w = Tensor::from_vec(w_data, vec![out_feat as i64, in_feat as i64]);

            // Warmup
            for _ in 0..3 {
                let _ = dispatch("fused_linear_relu", DispatchKey::Cpu, &[&x, &w]);
            }

            // Benchmark fused_linear_relu
            let iters = 20;
            let start = Instant::now();
            for _ in 0..iters {
                let _ = dispatch("fused_linear_relu", DispatchKey::Cpu, &[&x, &w]);
            }
            let fused_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // Benchmark matmul + relu (BLAS-backed)
            let start = Instant::now();
            for _ in 0..iters {
                let linear_out = x.matmul(&w);
                let _ = dispatch("relu", DispatchKey::Cpu, &[&linear_out]);
            }
            let blas_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let gflops =
                (2.0 * batch as f64 * out_feat as f64 * in_feat as f64) / (fused_ms / 1000.0) / 1e9;
            let blas_gflops =
                (2.0 * batch as f64 * out_feat as f64 * in_feat as f64) / (blas_ms / 1000.0) / 1e9;

            println!(
                "fused_linear_relu {}x{}x{}: fused={:.3}ms ({:.1} GFLOP/s), blas={:.3}ms ({:.1} GFLOP/s), gap={:.1}x",
                batch, in_feat, out_feat, fused_ms, gflops, blas_ms, blas_gflops, fused_ms / blas_ms
            );
        }
    }

    /// Reference: compute x @ w^T + bias then apply activation, all in scalar.
    fn reference_fused_linear(
        x_data: &[f32],
        w_data: &[f32],
        bias_data: Option<&[f32]>,
        batch: usize,
        in_feat: usize,
        out_feat: usize,
        activation: &str,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * out_feat];
        for b in 0..batch {
            for o in 0..out_feat {
                let mut sum = 0.0f32;
                for k in 0..in_feat {
                    sum += x_data[b * in_feat + k] * w_data[o * in_feat + k];
                }
                if let Some(bias) = bias_data {
                    sum += bias[o];
                }
                out[b * out_feat + o] = match activation {
                    "relu" => sum.max(0.0),
                    "silu" => sum / (1.0 + (-sum).exp()),
                    "gelu" => {
                        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
                        let coeff = 0.044715f32;
                        let x3 = sum * sum * sum;
                        let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                        0.5 * sum * (1.0 + t)
                    }
                    _ => panic!("unknown activation"),
                };
            }
        }
        out
    }

    /// Test fused_linear_relu correctness: BLAS path (large) and scalar path (small).
    #[test]
    fn test_fused_linear_relu_correctness() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};

        // Configs: (batch, in, out) - both large (BLAS) and small (scalar fallback)
        let configs: Vec<(usize, usize, usize)> = vec![
            (32, 256, 256), // above BLAS threshold
            (4, 8, 8),      // below BLAS threshold (scalar fallback)
            (1, 128, 64),   // single sample
        ];

        for &(batch, in_feat, out_feat) in &configs {
            let x_data: Vec<f32> = (0..batch * in_feat)
                .map(|i| ((i as f32) * 0.01).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_feat * in_feat)
                .map(|i| ((i as f32) * 0.01).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_feat).map(|i| (i as f32) * 0.01).collect();

            let x = Tensor::from_vec(x_data.clone(), vec![batch as i64, in_feat as i64]);
            let w = Tensor::from_vec(w_data.clone(), vec![out_feat as i64, in_feat as i64]);
            let bias = Tensor::from_vec(bias_data.clone(), vec![out_feat as i64]);

            // Without bias
            let result = dispatch("fused_linear_relu", DispatchKey::Cpu, &[&x, &w]);
            let result_data = result[0].as_f32_slice();
            let expected =
                reference_fused_linear(&x_data, &w_data, None, batch, in_feat, out_feat, "relu");
            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-4,
                    "relu no-bias batch={} in={} out={} idx={}: got={}, expected={}",
                    batch,
                    in_feat,
                    out_feat,
                    idx,
                    got,
                    exp
                );
            }

            // With bias
            let result = dispatch("fused_linear_relu", DispatchKey::Cpu, &[&x, &w, &bias]);
            let result_data = result[0].as_f32_slice();
            let expected = reference_fused_linear(
                &x_data,
                &w_data,
                Some(&bias_data),
                batch,
                in_feat,
                out_feat,
                "relu",
            );
            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-4,
                    "relu with-bias batch={} in={} out={} idx={}: got={}, expected={}",
                    batch,
                    in_feat,
                    out_feat,
                    idx,
                    got,
                    exp
                );
            }
        }
    }

    /// Test fused_linear_silu correctness: BLAS path and scalar path.
    #[test]
    fn test_fused_linear_silu_correctness() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};

        let configs: Vec<(usize, usize, usize)> = vec![
            (16, 512, 256), // above BLAS threshold
            (2, 16, 16),    // below BLAS threshold
        ];

        for &(batch, in_feat, out_feat) in &configs {
            let x_data: Vec<f32> = (0..batch * in_feat)
                .map(|i| ((i as f32) * 0.01).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_feat * in_feat)
                .map(|i| ((i as f32) * 0.01).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_feat).map(|i| (i as f32) * 0.01).collect();

            let x = Tensor::from_vec(x_data.clone(), vec![batch as i64, in_feat as i64]);
            let w = Tensor::from_vec(w_data.clone(), vec![out_feat as i64, in_feat as i64]);
            let bias = Tensor::from_vec(bias_data.clone(), vec![out_feat as i64]);

            // With bias
            let result = dispatch("fused_linear_silu", DispatchKey::Cpu, &[&x, &w, &bias]);
            let result_data = result[0].as_f32_slice();
            let expected = reference_fused_linear(
                &x_data,
                &w_data,
                Some(&bias_data),
                batch,
                in_feat,
                out_feat,
                "silu",
            );
            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-4,
                    "silu batch={} in={} out={} idx={}: got={}, expected={}",
                    batch,
                    in_feat,
                    out_feat,
                    idx,
                    got,
                    exp
                );
            }
        }
    }

    /// Test fused_linear_gelu correctness: BLAS path and scalar path.
    #[test]
    fn test_fused_linear_gelu_correctness() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};

        let configs: Vec<(usize, usize, usize)> = vec![
            (16, 512, 256), // above BLAS threshold
            (2, 16, 16),    // below BLAS threshold
        ];

        for &(batch, in_feat, out_feat) in &configs {
            let x_data: Vec<f32> = (0..batch * in_feat)
                .map(|i| ((i as f32) * 0.01).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_feat * in_feat)
                .map(|i| ((i as f32) * 0.01).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_feat).map(|i| (i as f32) * 0.01).collect();

            let x = Tensor::from_vec(x_data.clone(), vec![batch as i64, in_feat as i64]);
            let w = Tensor::from_vec(w_data.clone(), vec![out_feat as i64, in_feat as i64]);
            let bias = Tensor::from_vec(bias_data.clone(), vec![out_feat as i64]);

            let result = dispatch("fused_linear_gelu", DispatchKey::Cpu, &[&x, &w, &bias]);
            let result_data = result[0].as_f32_slice();
            let expected = reference_fused_linear(
                &x_data,
                &w_data,
                Some(&bias_data),
                batch,
                in_feat,
                out_feat,
                "gelu",
            );
            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-4,
                    "gelu batch={} in={} out={} idx={}: got={}, expected={}",
                    batch,
                    in_feat,
                    out_feat,
                    idx,
                    got,
                    exp
                );
            }
        }
    }

    /// Reference conv2d (im2col + scalar GEMM) for correctness verification.
    fn reference_conv2d(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        batch: usize,
        in_ch: usize,
        out_ch: usize,
        h: usize,
        w_img: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Vec<f32> {
        let oh = (h + 2 * pad - kh) / stride + 1;
        let ow = (w_img + 2 * pad - kw) / stride + 1;
        let col_cols = in_ch * kh * kw;
        let col_rows = batch * oh * ow;

        // im2col
        let mut col = vec![0.0f32; col_rows * col_cols];
        for row in 0..col_rows {
            let n = row / (oh * ow);
            let sp = row % (oh * ow);
            let sph = sp / ow;
            let spw = sp % ow;
            for ic in 0..in_ch {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let ih = sph * stride + ky;
                        let iw = spw * stride + kx;
                        let col_col = (ic * kh + ky) * kw + kx;
                        if ih >= pad && ih < pad + h && iw >= pad && iw < pad + w_img {
                            let xih = ih - pad;
                            let xiw = iw - pad;
                            let x_idx = ((n * in_ch + ic) * h + xih) * w_img + xiw;
                            col[row * col_cols + col_col] = x[x_idx];
                        }
                    }
                }
            }
        }

        // GEMM: col [col_rows, col_cols] @ w^T [col_cols, out_ch] = [col_rows, out_ch]
        let mut result = vec![0.0f32; col_rows * out_ch];
        for r in 0..col_rows {
            for oc in 0..out_ch {
                let mut sum = 0.0f32;
                for k in 0..col_cols {
                    sum += col[r * col_cols + k] * w[oc * col_cols + k];
                }
                result[r * out_ch + oc] = sum;
            }
        }

        // NHWC -> NCHW + bias
        let spatial = oh * ow;
        let mut out = vec![0.0f32; batch * out_ch * oh * ow];
        for row in 0..col_rows {
            let n = row / spatial;
            let sp = row % spatial;
            for oc in 0..out_ch {
                let out_idx = (n * out_ch + oc) * spatial + sp;
                let bias_val = bias.map_or(0.0, |b| b[oc]);
                out[out_idx] = result[row * out_ch + oc] + bias_val;
            }
        }
        out
    }

    /// Test conv2d_im2col correctness with various configs (padding, stride, bias).
    #[test]
    fn test_conv2d_im2col_correctness() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};

        let configs: Vec<(usize, usize, usize, usize, usize, usize, usize, usize)> = vec![
            // (batch, in_ch, out_ch, h, w, kernel, stride, pad)
            (1, 3, 16, 8, 8, 3, 1, 1),   // basic 3x3, pad=1
            (2, 8, 16, 16, 16, 3, 1, 1), // batched, above GEMM threshold
            (1, 4, 8, 12, 12, 3, 2, 0),  // stride=2, no padding
            (1, 3, 8, 10, 10, 5, 1, 2),  // 5x5 kernel
        ];

        for &(batch, in_ch, out_ch, h, w, kernel, stride, pad) in &configs {
            let x_data: Vec<f32> = (0..batch * in_ch * h * w)
                .map(|i| ((i as f32) * 0.001).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_ch * in_ch * kernel * kernel)
                .map(|i| ((i as f32) * 0.001).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_ch).map(|i| (i as f32) * 0.01).collect();

            let x_t = Tensor::from_vec(
                x_data.clone(),
                vec![batch as i64, in_ch as i64, h as i64, w as i64],
            );
            let w_t = Tensor::from_vec(
                w_data.clone(),
                vec![out_ch as i64, in_ch as i64, kernel as i64, kernel as i64],
            );
            let bias_t = Tensor::from_vec(bias_data.clone(), vec![out_ch as i64]);

            let stride_t = Tensor::from_scalar(stride as f32);
            let pad_t = Tensor::from_scalar(pad as f32);

            // With bias
            let result = dispatch(
                "conv2d",
                DispatchKey::Cpu,
                &[&x_t, &w_t, &bias_t, &stride_t, &pad_t],
            );
            let result_data = result[0].as_f32_slice();

            let expected = reference_conv2d(
                &x_data,
                &w_data,
                Some(&bias_data),
                batch,
                in_ch,
                out_ch,
                h,
                w,
                kernel,
                kernel,
                stride,
                pad,
            );

            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "conv2d b={} ic={} oc={} {}x{} k={} s={} p={} idx={}: got={}, expected={}",
                    batch,
                    in_ch,
                    out_ch,
                    h,
                    w,
                    kernel,
                    stride,
                    pad,
                    idx,
                    got,
                    exp
                );
            }

            // Without bias: pass zeros as bias
            let zero_bias = Tensor::from_vec(vec![0.0f32; out_ch], vec![out_ch as i64]);
            let result = dispatch(
                "conv2d",
                DispatchKey::Cpu,
                &[&x_t, &w_t, &zero_bias, &stride_t, &pad_t],
            );
            let result_data = result[0].as_f32_slice();

            let expected = reference_conv2d(
                &x_data, &w_data, None, batch, in_ch, out_ch, h, w, kernel, kernel, stride, pad,
            );

            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "conv2d-zero-bias b={} idx={}: got={}, expected={}",
                    batch,
                    idx,
                    got,
                    exp
                );
            }
        }
    }

    /// Benchmark conv2d_im2col to verify optimization impact.
    #[test]
    fn bench_conv2d_im2col() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};
        use std::time::Instant;

        let configs: Vec<(usize, usize, usize, usize, usize, usize, usize)> = vec![
            (1, 32, 32, 32, 32, 3, 1), // batch=1, 32ch, 32x32, 3x3
            (1, 64, 64, 64, 64, 3, 1), // batch=1, 64ch, 64x64, 3x3
        ];

        for &(batch, in_ch, out_ch, h, w, kernel, stride) in &configs {
            let pad = 1;
            let x_data: Vec<f32> = (0..batch * in_ch * h * w)
                .map(|i| ((i as f32) * 0.001).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_ch * in_ch * kernel * kernel)
                .map(|i| ((i as f32) * 0.001).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_ch).map(|i| (i as f32) * 0.01).collect();

            let x_t =
                Tensor::from_vec(x_data, vec![batch as i64, in_ch as i64, h as i64, w as i64]);
            let w_t = Tensor::from_vec(
                w_data,
                vec![out_ch as i64, in_ch as i64, kernel as i64, kernel as i64],
            );
            let bias_t = Tensor::from_vec(bias_data, vec![out_ch as i64]);
            let stride_t = Tensor::from_scalar(stride as f32);
            let pad_t = Tensor::from_scalar(pad as f32);

            // Warmup
            for _ in 0..3 {
                let _ = dispatch(
                    "conv2d",
                    DispatchKey::Cpu,
                    &[&x_t, &w_t, &bias_t, &stride_t, &pad_t],
                );
            }

            let iters = 20;
            let start = Instant::now();
            for _ in 0..iters {
                let _ = dispatch(
                    "conv2d",
                    DispatchKey::Cpu,
                    &[&x_t, &w_t, &bias_t, &stride_t, &pad_t],
                );
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            println!(
                "conv2d {}x{} {}x{} k={} s={}: {:.3}ms",
                batch, in_ch, h, w, kernel, stride, ms
            );
        }
    }

    /// Test that the thread-local scratch buffer correctly reuses memory
    /// across calls with different sizes (exercises buffer growth).
    #[test]
    fn test_conv2d_scratch_reuse() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};

        // Call conv2d 3 times with increasing sizes to exercise buffer growth.
        // Then call with the smallest size again to verify shrink is handled.
        let configs: Vec<(usize, usize, usize, usize, usize, usize, usize, usize)> = vec![
            (1, 4, 8, 8, 8, 3, 1, 1),     // small: grows buffer
            (1, 16, 32, 16, 16, 3, 1, 1), // medium: grows buffer more
            (2, 8, 16, 12, 12, 5, 1, 2),  // different shape: may reuse
            (1, 4, 8, 8, 8, 3, 1, 1),     // small again: reuse existing buffer
        ];

        for &(batch, in_ch, out_ch, h, w, kernel, stride, pad) in &configs {
            let x_data: Vec<f32> = (0..batch * in_ch * h * w)
                .map(|i| ((i as f32 + 1.0) * 0.01).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_ch * in_ch * kernel * kernel)
                .map(|i| ((i as f32 + 1.0) * 0.01).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_ch).map(|i| (i as f32) * 0.01).collect();

            let x_t = Tensor::from_vec(
                x_data.clone(),
                vec![batch as i64, in_ch as i64, h as i64, w as i64],
            );
            let w_t = Tensor::from_vec(
                w_data.clone(),
                vec![out_ch as i64, in_ch as i64, kernel as i64, kernel as i64],
            );
            let bias_t = Tensor::from_vec(bias_data.clone(), vec![out_ch as i64]);
            let stride_t = Tensor::from_scalar(stride as f32);
            let pad_t = Tensor::from_scalar(pad as f32);

            let result = dispatch(
                "conv2d",
                DispatchKey::Cpu,
                &[&x_t, &w_t, &bias_t, &stride_t, &pad_t],
            );
            let result_data = result[0].as_f32_slice();

            let expected = reference_conv2d(
                &x_data,
                &w_data,
                Some(&bias_data),
                batch,
                in_ch,
                out_ch,
                h,
                w,
                kernel,
                kernel,
                stride,
                pad,
            );

            assert_eq!(
                result_data.len(),
                expected.len(),
                "output size mismatch for config {:?}",
                (batch, in_ch, out_ch, h, w, kernel, stride, pad)
            );

            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "scratch_reuse b={} ic={} oc={} {}x{} k={} s={} p={} idx={}: got={}, expected={}",
                    batch, in_ch, out_ch, h, w, kernel, stride, pad, idx, got, exp
                );
            }
        }
    }

    /// Test conv2d with stride=2, dilation=2 — exercises the general (non-fast-path)
    /// im2col loop and verifies output matches a reference scalar implementation.
    #[test]
    fn test_conv2d_im2col_stride_dilation() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};

        let configs: Vec<(usize, usize, usize, usize, usize, usize, usize, usize)> = vec![
            (1, 4, 8, 16, 16, 3, 2, 1),  // stride=2, dilation=1
            (1, 3, 8, 12, 12, 3, 1, 1),  // stride=1, pad=1 (fast path)
            (2, 8, 16, 20, 20, 3, 2, 0), // stride=2, no pad, batched
            (1, 4, 8, 14, 14, 3, 2, 2),  // stride=2, pad=2
            (1, 4, 8, 10, 10, 5, 2, 2),  // stride=2, 5x5 kernel, pad=2
        ];

        for &(batch, in_ch, out_ch, h, w, kernel, stride, pad) in &configs {
            let x_data: Vec<f32> = (0..batch * in_ch * h * w)
                .map(|i| ((i as f32 + 1.0) * 0.01).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_ch * in_ch * kernel * kernel)
                .map(|i| ((i as f32 + 1.0) * 0.01).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_ch).map(|i| (i as f32) * 0.01).collect();

            let x_t = Tensor::from_vec(
                x_data.clone(),
                vec![batch as i64, in_ch as i64, h as i64, w as i64],
            );
            let w_t = Tensor::from_vec(
                w_data.clone(),
                vec![out_ch as i64, in_ch as i64, kernel as i64, kernel as i64],
            );
            let bias_t = Tensor::from_vec(bias_data.clone(), vec![out_ch as i64]);
            let stride_t = Tensor::from_scalar(stride as f32);
            let pad_t = Tensor::from_scalar(pad as f32);

            let result = dispatch(
                "conv2d",
                DispatchKey::Cpu,
                &[&x_t, &w_t, &bias_t, &stride_t, &pad_t],
            );
            let result_data = result[0].as_f32_slice();

            let expected = reference_conv2d(
                &x_data,
                &w_data,
                Some(&bias_data),
                batch,
                in_ch,
                out_ch,
                h,
                w,
                kernel,
                kernel,
                stride,
                pad,
            );

            assert_eq!(
                result_data.len(),
                expected.len(),
                "output size mismatch for stride_dilation config {:?}",
                (batch, in_ch, out_ch, h, w, kernel, stride, pad)
            );

            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "stride_dilation b={} ic={} oc={} {}x{} k={} s={} p={} idx={}: got={}, expected={}",
                    batch, in_ch, out_ch, h, w, kernel, stride, pad, idx, got, exp
                );
            }
        }
    }

    /// Reference scalar matmul with explicit strides, for testing blocked_matmul.
    fn reference_matmul_strided(
        a: &[f32],
        b: &[f32],
        batch: usize,
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
        let mut out = vec![0.0f32; batch * m * n];
        for bat in 0..batch {
            let a_base = bat * a_batch_stride;
            let b_base = bat * b_batch_stride;
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a[a_base + i * a_s0 + kk * a_s1] * b[b_base + kk * b_s0 + j * b_s1];
                    }
                    out[bat * m * n + i * n + j] = sum;
                }
            }
        }
        out
    }

    fn verify_blocked_matmul(
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        a_bs: usize,
        a_s0: usize,
        a_s1: usize,
        b_bs: usize,
        b_s0: usize,
        b_s1: usize,
        tag: &str,
    ) {
        let mut out = vec![0.0f32; batch * m * n];
        let total_rows = batch * m;
        for row in 0..total_rows {
            blocked_row_matmul(
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
                row,
                m,
                n,
                k,
                a_bs,
                a_s0,
                a_s1,
                b_bs,
                b_s0,
                b_s1,
            );
        }
        let expected =
            reference_matmul_strided(a, b, batch, m, n, k, a_bs, a_s0, a_s1, b_bs, b_s0, b_s1);
        for (idx, (got, exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "{} idx={}: got={}, expected={}",
                tag,
                idx,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_blocked_matmul_square() {
        for &(m, n, k) in &[(64usize, 64, 64), (128, 128, 128)] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.001).cos()).collect();
            verify_blocked_matmul(
                &a,
                &b,
                1,
                m,
                n,
                k,
                m * k,
                k,
                1,
                k * n,
                n,
                1,
                &format!("square {}x{}x{}", m, n, k),
            );
        }
    }

    #[test]
    fn test_blocked_matmul_non_square() {
        // Non-multiples of tile sizes (TILE_M=1, TILE_N=4, TILE_K=64)
        let (m, n, k) = (37usize, 64, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.001).cos()).collect();
        verify_blocked_matmul(
            &a,
            &b,
            1,
            m,
            n,
            k,
            m * k,
            k,
            1,
            k * n,
            n,
            1,
            &format!("non_square {}x{}x{}", m, n, k),
        );

        // Also test n not multiple of TILE_N
        let (m, n, k) = (16, 7, 32);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.001).cos()).collect();
        verify_blocked_matmul(
            &a,
            &b,
            1,
            m,
            n,
            k,
            m * k,
            k,
            1,
            k * n,
            n,
            1,
            &format!("non_square {}x{}x{}", m, n, k),
        );
    }

    #[test]
    fn test_blocked_matmul_small() {
        for &(m, n, k) in &[
            (4usize, 4, 4),
            (8, 8, 8),
            (3, 5, 7),
            (1, 1, 1),
            (16, 16, 16),
        ] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i as f32 + 1.0) * 0.1).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i as f32 + 1.0) * 0.1).cos()).collect();
            verify_blocked_matmul(
                &a,
                &b,
                1,
                m,
                n,
                k,
                m * k,
                k,
                1,
                k * n,
                n,
                1,
                &format!("small {}x{}x{}", m, n, k),
            );
        }
    }

    #[test]
    fn test_blocked_matmul_batched() {
        let (batch, m, n, k) = (4usize, 16, 16, 16);
        let a: Vec<f32> = (0..batch * m * k)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let b: Vec<f32> = (0..batch * k * n)
            .map(|i| ((i as f32) * 0.01).cos())
            .collect();
        verify_blocked_matmul(
            &a,
            &b,
            batch,
            m,
            n,
            k,
            m * k,
            k,
            1,
            k * n,
            n,
            1,
            &format!("batched {}x{}x{}x{}", batch, m, n, k),
        );
    }

    /// Benchmark: naive triple loop vs blocked scalar matmul for sizes below BLAS threshold.
    #[test]
    fn bench_scalar_matmul() {
        use std::time::Instant;

        fn naive_matmul_row(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a[i * k + kk] * b[kk * n + j];
                    }
                    out[i * n + j] = sum;
                }
            }
        }

        fn blocked_matmul_row(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
            for row in 0..m {
                blocked_row_matmul(
                    a.as_ptr(),
                    b.as_ptr(),
                    out.as_mut_ptr(),
                    row,
                    m,
                    n,
                    k,
                    m * k,
                    k,
                    1,
                    k * n,
                    n,
                    1,
                );
            }
        }

        for &(m, n, k) in &[(32usize, 32, 32), (48, 48, 48), (64, 64, 64)] {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.001).cos()).collect();
            let mut out_naive = vec![0.0f32; m * n];
            let mut out_blocked = vec![0.0f32; m * n];
            let iters = 100;

            // Warmup
            for _ in 0..3 {
                naive_matmul_row(&a, &b, &mut out_naive, m, n, k);
                blocked_matmul_row(&a, &b, &mut out_blocked, m, n, k);
            }

            let start = Instant::now();
            for _ in 0..iters {
                naive_matmul_row(&a, &b, &mut out_naive, m, n, k);
            }
            let naive_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let start = Instant::now();
            for _ in 0..iters {
                blocked_matmul_row(&a, &b, &mut out_blocked, m, n, k);
            }
            let blocked_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let gflops = |ms: f64| (2.0 * m as f64 * n as f64 * k as f64) / (ms / 1000.0) / 1e9;

            println!(
                "matmul {}x{}x{}: naive={:.3}ms ({:.2} GFLOP/s), blocked={:.3}ms ({:.2} GFLOP/s), speedup={:.2}x",
                m, n, k, naive_ms, gflops(naive_ms), blocked_ms, gflops(blocked_ms), naive_ms / blocked_ms
            );

            // Verify correctness
            for (idx, (got, exp)) in out_blocked.iter().zip(out_naive.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "bench blocked vs naive mismatch at idx={}: got={}, expected={}",
                    idx,
                    got,
                    exp
                );
            }
        }
    }

    /// Test SIMD path (contiguous a_stride_1 == 1, b_stride_1 == 1).
    #[test]
    fn test_blocked_matmul_simd_contiguous() {
        for &(m, n, k) in &[(64usize, 64, 64), (32, 32, 32), (16, 72, 32), (1, 64, 128)] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.001).cos()).collect();
            verify_blocked_matmul(
                &a,
                &b,
                1,
                m,
                n,
                k,
                m * k,
                k,
                1,
                k * n,
                n,
                1,
                &format!("simd_contiguous {}x{}x{}", m, n, k),
            );
        }
    }

    /// Test non-contiguous B (b_stride_1 != 1) falls through to scalar path correctly.
    #[test]
    fn test_blocked_matmul_simd_noncontiguous() {
        let (m, n, k) = (64usize, 64, 64);
        // Create B with stride_1 = 2 (every other column, rest are padding)
        let b_stride_1 = 2usize;
        let b_data_size = k * n * b_stride_1;
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
        let b: Vec<f32> = (0..b_data_size)
            .map(|i| {
                if i % b_stride_1 == 0 {
                    ((i as f32) * 0.001).cos()
                } else {
                    0.0
                }
            })
            .collect();
        // Reference: B is stored with stride_1 = 2, logical n = 64
        let mut out = vec![0.0f32; m * n];
        let total_rows = m;
        for row in 0..total_rows {
            blocked_row_matmul(
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
                row,
                m,
                n,
                k,
                m * k,
                k,
                1,
                k * n * b_stride_1,
                n * b_stride_1,
                b_stride_1,
            );
        }
        // Verify against reference scalar matmul with same strides
        let expected = reference_matmul_strided(
            &a,
            &b,
            1,
            m,
            n,
            k,
            m * k,
            k,
            1,
            k * n * b_stride_1,
            n * b_stride_1,
            b_stride_1,
        );
        for (idx, (got, exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "noncontiguous idx={}: got={}, expected={}",
                idx,
                got,
                exp
            );
        }
    }
}

fn leaky_relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let slope = args[1].item();
    let mut output = x.clone();
    let numel = output.inner.numel() as usize;
    let ptr = output.data_ptr_f32_mut();
    for i in 0..numel {
        unsafe {
            let v = *ptr.add(i);
            *ptr.add(i) = if v > 0.0 { v } else { v * slope };
        }
    }
    vec![output]
}

fn prelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let weight = args[1];
    let w_data = weight.as_f32_slice();
    let x_shape = x.shape();
    let numel = x.inner.numel() as usize;
    let ndim = x.ndim();
    let w_numel = w_data.len();
    let mut output = x.clone();
    let ptr = output.data_ptr_f32_mut();
    let mut indices = vec![0i64; ndim];
    for i in 0..numel {
        unsafe {
            let v = *ptr.add(i);
            let w_idx = if w_numel == 1 {
                0
            } else {
                indices[1] as usize % w_numel
            };
            let w = *w_data.get_unchecked(w_idx);
            *ptr.add(i) = if v > 0.0 { v } else { v * w };
        }
        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < x_shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }
    vec![output]
}

fn softplus_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let beta = args[1].item() as f32;
    let threshold = args[2].item() as f32;
    let numel = x.inner.numel() as usize;
    let mut output_data = vec![0.0f32; numel];
    let x_data = x.as_f32_slice();
    for i in 0..numel {
        let bx = beta * x_data[i];
        output_data[i] = if bx > threshold {
            x_data[i]
        } else {
            (1.0 + (bx).exp()).ln() / beta
        };
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

fn hardswish_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let numel = x.inner.numel() as usize;
    let mut output_data = vec![0.0f32; numel];
    let x_data = x.as_f32_slice();
    for i in 0..numel {
        let v = x_data[i];
        let relu6 = (v + 3.0).max(0.0).min(6.0);
        output_data[i] = v * relu6 / 6.0;
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

fn lt_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let threshold = args[1].item();
    let numel = x.inner.numel() as usize;
    let x_data = x.as_f32_slice();
    let mut output_data = vec![0.0f32; numel];
    for i in 0..numel {
        output_data[i] = if x_data[i] < threshold { 1.0 } else { 0.0 };
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

fn add_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let scalar = args[1].item();
    let numel = x.inner.numel() as usize;
    let x_data = x.as_f32_slice();
    let mut output_data = vec![0.0f32; numel];
    for i in 0..numel {
        output_data[i] = x_data[i] + scalar;
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

fn div_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let scalar = args[1].item();
    let numel = x.inner.numel() as usize;
    let x_data = x.as_f32_slice();
    let mut output_data = vec![0.0f32; numel];
    for i in 0..numel {
        output_data[i] = x_data[i] / scalar;
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

fn logical_not_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let numel = x.inner.numel() as usize;
    let x_data = x.as_f32_slice();
    let mut output_data = vec![0.0f32; numel];
    for i in 0..numel {
        output_data[i] = if x_data[i] == 0.0 { 1.0 } else { 0.0 };
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

fn bce_with_logits_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let target = args[1];
    let numel = input.inner.numel() as usize;
    let input_data = input.as_f32_slice();
    let target_data = target.as_f32_slice();
    let mut loss = 0.0f32;
    for i in 0..numel {
        let x = input_data[i];
        let t = target_data[i];
        let max_val = if x > 0.0 { x } else { 0.0 };
        loss += max_val - x * t + (1.0 + (-x.abs()).exp()).ln();
    }
    let avg_loss = loss / numel as f32;
    vec![Tensor::from_vec(vec![avg_loss], vec![1])]
}

fn huber_loss_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let target = args[1];
    let delta = args[2].item() as f32;
    let numel = input.inner.numel() as usize;
    let input_data = input.as_f32_slice();
    let target_data = target.as_f32_slice();
    let mut loss = 0.0f32;
    for i in 0..numel {
        let diff = input_data[i] - target_data[i];
        let abs_diff = diff.abs();
        loss += if abs_diff < delta {
            0.5 * diff * diff
        } else {
            delta * (abs_diff - 0.5 * delta)
        };
    }
    let avg_loss = loss / numel as f32;
    vec![Tensor::from_vec(vec![avg_loss], vec![1])]
}

fn conv_transpose2d_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let weight = args[1];
    let stride = args[2].item() as i64;
    let padding = args[3].item() as i64;

    let x_shape = x.shape();
    let w_shape = weight.shape();
    let batch = x_shape[0];
    let in_channels = x_shape[1];
    let h_in = x_shape[2];
    let w_in = x_shape[3];
    let out_channels = w_shape[1];
    let kernel_h = w_shape[2];
    let kernel_w = w_shape[3];

    let h_out = (h_in - 1) * stride - 2 * padding + kernel_h;
    let w_out = (w_in - 1) * stride - 2 * padding + kernel_w;

    let output_shape = vec![batch, out_channels, h_out, w_out];
    let mut output = Tensor::zeros(output_shape.clone(), x.dtype(), x.device());

    let x_data = x.as_f32_slice();
    let w_data = weight.as_f32_slice();
    let out_ptr = output.data_ptr_f32_mut();

    for b in 0..batch as usize {
        for ic in 0..in_channels as usize {
            for oc in 0..out_channels as usize {
                for hi in 0..h_in as usize {
                    for wi in 0..w_in as usize {
                        let x_val = x_data[b
                            * (in_channels as usize * h_in as usize * w_in as usize)
                            + ic * (h_in as usize * w_in as usize)
                            + hi * w_in as usize
                            + wi];
                        for kh in 0..kernel_h as usize {
                            for kw in 0..kernel_w as usize {
                                let ho = hi as i64 * stride - padding + kh as i64;
                                let wo = wi as i64 * stride - padding + kw as i64;
                                if ho >= 0 && ho < h_out && wo >= 0 && wo < w_out {
                                    let w_idx = ic
                                        * (out_channels as usize
                                            * kernel_h as usize
                                            * kernel_w as usize)
                                        + oc * (kernel_h as usize * kernel_w as usize)
                                        + kh * kernel_w as usize
                                        + kw;
                                    let out_idx = b
                                        * (out_channels as usize * h_out as usize * w_out as usize)
                                        + oc * (h_out as usize * w_out as usize)
                                        + ho as usize * w_out as usize
                                        + wo as usize;
                                    unsafe {
                                        *out_ptr.add(out_idx) += x_val * w_data[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    vec![output]
}
