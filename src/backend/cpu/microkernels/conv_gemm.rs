//! Conv-specific GEMM fused with bias and activation.
//!
//! Uses matrixmultiply for the core GEMM, then applies bias and activation
//! as a SIMD-accelerated post-processing pass.  This avoids the separate
//! scalar loop while keeping matrixmultiply's well-optimised multi-level
//! tiling and SIMD.
//!
//! Dimensions:
//!   A = weight   — row-major [M, K], contiguous (rs_a=K, cs_a=1)
//!   B = col      — stored as [N, K] row-major, accessed as [K, N] (rs_b=1, cs_b=K)
//!   C = output   — row-major [M, N] (rs_c=N, cs_c=1)
//!
//! When bias and activation are both absent the call is forwarded directly
//! to matrixmultiply with zero overhead.

use super::ConvActivation;
use crate::backend::cpu::microkernels::conv::apply_conv_activation;
use crate::backend::cpu::microkernels::simd_avx2_available;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use core::arch::x86_64::*;

// ── Scalar bias + activation pass ────────────────────────────

#[inline(never)]
fn apply_bias_activation_scalar(
    m: usize,
    n: usize,
    c: &mut [f32],
    bias: Option<&[f32]>,
    activation: Option<ConvActivation>,
) {
    for oc in 0..m {
        let bv = bias.map(|b| b[oc]).unwrap_or(0.0);
        for s in 0..n {
            let idx = oc * n + s;
            let v = c[idx] + bv;
            c[idx] = match activation {
                None => v,
                Some(act) => apply_conv_activation(v, act),
            };
        }
    }
}

// ── AVX2 bias + activation pass ──────────────────────────────

/// Apply bias and activation in-place on C[M, N] using AVX2 for bias
/// addition and the existing AVX2 activation kernels from `activations.rs`.
///
/// A small stack buffer avoids heap allocation for n ≤ 64.  For larger n a
/// single heap allocation is made per call (not per row).
///
/// # Safety
/// Caller must guarantee AVX2+FMA availability and valid pointers to
/// M × N contiguous f32 values.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_bias_activation_avx2(
    m: usize,
    n: usize,
    c: *mut f32,
    bias: Option<&[f32]>,
    activation: Option<ConvActivation>,
) {
    use crate::backend::cpu::microkernels::activations::*;

    // Temp buffer for safe non-overlapping activation input.
    // Use stack for n ≤ 64 (common for small spatial sizes).
    let mut stack_buf = [0.0f32; 64];
    let mut heap_buf: Vec<f32>;
    let tmp: &mut [f32] = if n <= 64 {
        &mut stack_buf[..n]
    } else {
        heap_buf = vec![0.0f32; n];
        &mut heap_buf[..]
    };

    for oc in 0..m {
        let bv = bias.map(|b| b[oc]).unwrap_or(0.0);
        let vbias = _mm256_set1_ps(bv);
        let row = std::slice::from_raw_parts_mut(c.add(oc * n), n);

        // Add bias via AVX2
        let mut i = 0usize;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(row.as_ptr().add(i));
            _mm256_storeu_ps(row.as_mut_ptr().add(i), _mm256_add_ps(v, vbias));
            i += 8;
        }
        for j in i..n {
            row[j] += bv;
        }

        // Apply activation via existing SIMD kernels (require non-overlapping
        // input/output).  Copy row → tmp → activate → row.
        match activation {
            Some(ConvActivation::Silu) => {
                tmp.copy_from_slice(row);
                silu_f32_avx2(tmp, row);
            }
            Some(ConvActivation::Relu) => {
                tmp.copy_from_slice(row);
                relu_f32_avx2(tmp, row);
            }
            Some(ConvActivation::Gelu) => {
                tmp.copy_from_slice(row);
                gelu_f32_avx2(tmp, row);
            }
            None => {}
        }
    }
}

// ── Top-level dispatch ───────────────────────────────────────

/// Conv GEMM with optional fused bias+activation.
///
/// Delegates the core GEMM to matrixmultiply.  When bias or activation
/// is present the output pass uses AVX2 (on supported x86_64) or scalar.
/// When both are absent the call is forwarded directly with zero overhead.
pub fn conv_gemm_f32(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    rs_a: isize,
    cs_a: isize,
    b: *const f32,
    rs_b: isize,
    cs_b: isize,
    c: *mut f32,
    rs_c: isize,
    cs_c: isize,
    bias: Option<&[f32]>,
    activation: Option<ConvActivation>,
) {
    // ── No bias or activation: straight GEMM ──
    if bias.is_none() && activation.is_none() {
        unsafe {
            matrixmultiply::sgemm(
                m, k, n, 1.0, a, rs_a, cs_a, b, rs_b, cs_b, 0.0, c, rs_c, cs_c,
            );
        }
        return;
    }

    // ── With bias/activation: matrixmultiply then post-process ──
    unsafe {
        matrixmultiply::sgemm(
            m, k, n, 1.0, a, rs_a, cs_a, b, rs_b, cs_b, 0.0, c, rs_c, cs_c,
        );
    }

    // Post-process: contiguous row-major output (common case)
    if rs_c == n as isize && cs_c == 1 {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if simd_avx2_available() {
            unsafe {
                apply_bias_activation_avx2(m, n, c, bias, activation);
            }
            return;
        }

        let slice = unsafe { std::slice::from_raw_parts_mut(c, m * n) };
        apply_bias_activation_scalar(m, n, slice, bias, activation);
        return;
    }

    // Non-standard strides — fall back to elementwise via indices
    for oc in 0..m {
        let bv = bias.map(|b| b[oc]).unwrap_or(0.0);
        for s in 0..n {
            let idx = (oc as isize * rs_c + s as isize * cs_c) as usize;
            let v = unsafe { *c.add(idx) + bv };
            unsafe {
                *c.add(idx) = match activation {
                    None => v,
                    Some(act) => apply_conv_activation(v, act),
                };
            }
        }
    }
}
