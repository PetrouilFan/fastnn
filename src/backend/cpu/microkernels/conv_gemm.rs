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
use crate::backend::cpu::microkernels::activations::{exp_avx2_vec, tanh_avx2_vec};
use crate::backend::cpu::microkernels::conv::apply_conv_activation;
use crate::backend::cpu::microkernels::simd_avx2_available;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use core::arch::x86_64::*;

// ── Scalar bias + activation pass (already fused) ────────────

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

// ── Fused AVX2 bias + activation pass (single load→fma→store) ─

/// Apply bias and activation in a single fused SIMD pass.
///
/// No temp buffer, no copy_from_slice, no second pass over the data.
/// Each 8-element chunk is loaded once, combined with bias, passed
/// through the activation function, and stored.
///
/// # Safety
/// Caller must guarantee AVX2+FMA availability and valid pointers to
/// M × N contiguous f32 values.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn bias_activation_fused_avx2(
    m: usize,
    n: usize,
    c: *mut f32,
    bias: Option<&[f32]>,
    activation: Option<ConvActivation>,
) {
    let none = activation.is_none();

    for oc in 0..m {
        let bv = bias.map(|b| b[oc]).unwrap_or(0.0);
        let vbias = _mm256_set1_ps(bv);
        let row = std::slice::from_raw_parts_mut(c.add(oc * n), n);
        let mut i = 0usize;

        if none {
            // Bias only: single add + store
            while i + 8 <= n {
                let v = _mm256_loadu_ps(row.as_ptr().add(i));
                _mm256_storeu_ps(row.as_mut_ptr().add(i), _mm256_add_ps(v, vbias));
                i += 8;
            }
            if i < n {
                for j in i..n {
                    row[j] += bv;
                }
            }
        } else {
            match activation {
                Some(ConvActivation::Relu) => {
                    let zero = _mm256_setzero_ps();
                    while i + 8 <= n {
                        let v = _mm256_loadu_ps(row.as_ptr().add(i));
                        _mm256_storeu_ps(
                            row.as_mut_ptr().add(i),
                            _mm256_max_ps(_mm256_add_ps(v, vbias), zero),
                        );
                        i += 8;
                    }
                    for j in i..n {
                        row[j] = (row[j] + bv).max(0.0);
                    }
                }
                Some(ConvActivation::Silu) => {
                    let one = _mm256_set1_ps(1.0);
                    let neg_zero = _mm256_set1_ps(-0.0f32);
                    while i + 8 <= n {
                        let v = _mm256_loadu_ps(row.as_ptr().add(i));
                        let x = _mm256_add_ps(v, vbias);
                        let neg_x = _mm256_xor_ps(x, neg_zero);
                        let e_neg = exp_avx2_vec(neg_x);
                        let sig = _mm256_div_ps(one, _mm256_add_ps(one, e_neg));
                        _mm256_storeu_ps(row.as_mut_ptr().add(i), _mm256_mul_ps(x, sig));
                        i += 8;
                    }
                    for j in i..n {
                        let x = row[j] + bv;
                        row[j] = x / (1.0 + (-x).exp());
                    }
                }
                Some(ConvActivation::Gelu) => {
                    let half = _mm256_set1_ps(0.5);
                    let one = _mm256_set1_ps(1.0);
                    let sqrt_2pi = _mm256_set1_ps(0.797_884_6_f32);
                    let coeff = _mm256_set1_ps(0.044715f32);
                    while i + 8 <= n {
                        let v = _mm256_loadu_ps(row.as_ptr().add(i));
                        let x = _mm256_add_ps(v, vbias);
                        let x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
                        let tanh_arg =
                            _mm256_mul_ps(sqrt_2pi, _mm256_add_ps(x, _mm256_mul_ps(coeff, x3)));
                        let t = tanh_avx2_vec(tanh_arg);
                        _mm256_storeu_ps(
                            row.as_mut_ptr().add(i),
                            _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_add_ps(one, t)),
                        );
                        i += 8;
                    }
                    for j in i..n {
                        let x = row[j] + bv;
                        let x3 = x * x * x;
                        row[j] = 0.5 * x * (1.0 + (0.7978846 * (x + 0.044715 * x3)).tanh());
                    }
                }
                _ => {}
            }
        }
    }
}

// ── Top-level dispatch ───────────────────────────────────────

/// Conv GEMM with optional fused bias+activation.
///
/// Delegates the core GEMM to matrixmultiply.  When bias or activation
/// is present the output pass uses AVX2 (on supported x86_64) or scalar.
/// When both are absent the call is forwarded directly with zero overhead.
///
/// For M=1 (depthwise) a fast-path dot-product is used to avoid
/// matrixmultiply's per-call overhead.
pub unsafe fn conv_gemm_f32(
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
    // ── M=1 fast path: dot product per spatial position ──
    // Avoids matrixmultiply's per-call overhead for depthwise GEMMs
    // where a single output channel is computed per group.
    if m == 1
        && rs_c == n as isize
        && cs_c == 1
        && rs_a == k as isize
        && cs_a == 1
        && rs_b == 1
        && cs_b == k as isize
    {
        let bv = bias.and_then(|b| b.first().copied()).unwrap_or(0.0);
        let out = unsafe { std::slice::from_raw_parts_mut(c, n) };
        let wgt = unsafe { std::slice::from_raw_parts(a, k) };
        for s in 0..n {
            let mut sum = bv;
            for kk in 0..k {
                sum += wgt[kk] * unsafe { *b.add(kk + s * k) };
            }
            out[s] = match activation {
                Some(act) => apply_conv_activation(sum, act),
                None => sum,
            };
        }
        return;
    }

    // ── No bias or activation: straight GEMM ──
    if bias.is_none() && activation.is_none() {
        unsafe {
            crate::backend::cpu::sgemm::sgemm(
                m, k, n, 1.0, a, rs_a, cs_a, b, rs_b, cs_b, 0.0, c, rs_c, cs_c,
            );
        }
        return;
    }

    // ── With bias/activation: GEMM then post-process ──
    unsafe {
        crate::backend::cpu::sgemm::sgemm(
            m, k, n, 1.0, a, rs_a, cs_a, b, rs_b, cs_b, 0.0, c, rs_c, cs_c,
        );
    }

    // Post-process: contiguous row-major output (common case)
    if rs_c == n as isize && cs_c == 1 {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if simd_avx2_available() {
            unsafe {
                bias_activation_fused_avx2(m, n, c, bias, activation);
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
