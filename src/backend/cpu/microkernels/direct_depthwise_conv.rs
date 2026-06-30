//! Direct depthwise convolution — no im2col expansion.
//!
//! For depthwise 3×3 conv (groups == in_channels == out_channels),
//! im2col expands the input by 9×, thrashing L1/L2 cache.
//! This kernel processes one group at a time with spatial AVX2
//! vectorization, reading input tiles directly from NCHW layout.
//!
//! Supported: 3×3 stride-1 pad-1 (AVX2), generic (scalar fallback).

#![allow(dead_code)]

use super::ConvActivation;
use crate::backend::cpu::microkernels::conv::apply_conv_activation;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use core::arch::x86_64::*;

use crate::backend::cpu::microkernels::activations::{exp_avx2_vec, tanh_avx2_vec};

#[cfg(feature = "debug_canary")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "debug_canary")]
static DID_AVX2: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "debug_canary")]
static DID_SCALAR: AtomicBool = AtomicBool::new(false);

// ── Helper: apply activation to an AVX2 vector ──────────────────

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_activation_vec(v: __m256, act: ConvActivation) -> __m256 {
    match act {
        ConvActivation::Relu => {
            _mm256_max_ps(v, _mm256_setzero_ps())
        }
        ConvActivation::Silu => {
            let neg_v = _mm256_xor_ps(v, _mm256_set1_ps(-0.0f32));
            let e_neg = exp_avx2_vec(neg_v);
            let one = _mm256_set1_ps(1.0);
            let sig = _mm256_div_ps(one, _mm256_add_ps(one, e_neg));
            _mm256_mul_ps(v, sig)
        }
        ConvActivation::Gelu => {
            let half = _mm256_set1_ps(0.5);
            let one = _mm256_set1_ps(1.0);
            let sqrt_2pi = _mm256_set1_ps(0.7978845608028654f32);
            let coeff = _mm256_set1_ps(0.044715f32);
            let v3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
            let tanh_arg = _mm256_mul_ps(sqrt_2pi, _mm256_add_ps(v, _mm256_mul_ps(coeff, v3)));
            let t = tanh_avx2_vec(tanh_arg);
            _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, t))
        }
    }
}

// ── Scalar: process one depthwise pixel ────────────────────────

fn depthwise_conv3x3_scalar_pixel(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    img: usize,
    g: usize,
    oh: usize,
    ow: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    h_out: usize,
    w_out: usize,
    stride: usize,
    padding: usize,
    bv: f32,
    activation: Option<ConvActivation>,
) {
    let mut sum = bv;
    let inp_plane = img * (c * h * w) + g * h * w;
    for kr in 0..3isize {
        let ih = oh as isize * stride as isize - padding as isize + kr;
        if ih < 0 || ih >= h as isize {
            continue;
        }
        let row_base = inp_plane + (ih as usize) * w;
        for kw in 0..3isize {
            let iw = ow as isize * stride as isize - padding as isize + kw;
            if iw < 0 || iw >= w as isize {
                continue;
            }
            let wv = weight[g * 9 + (kr * 3 + kw) as usize];
            sum += input[row_base + (iw as usize)] * wv;
        }
    }
    let out_idx = img * (f * h_out * w_out) + g * h_out * w_out + oh * w_out + ow;
    output[out_idx] = match activation {
        Some(act) => apply_conv_activation(sum, act),
        None => sum,
    };
}

// ── AVX2: 3×3 stride-1 pad-1 depthwise conv ────────────────────

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn depthwise_conv3x3_f32_avx2_path(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    h_out: usize,
    w_out: usize,
    activation: Option<ConvActivation>,
) {
    let inp_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let wgt_ptr = weight.as_ptr();
    let hw = h * w;
    let ho_wo = h_out * w_out;

    for img in 0..n {
        let inp_img_off = img * (c * hw);

        for g in 0..f {
            let bv = if bias.is_empty() { 0.0 } else { bias[g] };
            let vbias = _mm256_set1_ps(bv);

            // Pre-broadcast 9 weight values (hoisted outside spatial loops)
            let w_base = wgt_ptr.add(g * 9);
            let w0 = _mm256_set1_ps(*w_base.add(0));
            let w1 = _mm256_set1_ps(*w_base.add(1));
            let w2 = _mm256_set1_ps(*w_base.add(2));
            let w3 = _mm256_set1_ps(*w_base.add(3));
            let w4 = _mm256_set1_ps(*w_base.add(4));
            let w5 = _mm256_set1_ps(*w_base.add(5));
            let w6 = _mm256_set1_ps(*w_base.add(6));
            let w7 = _mm256_set1_ps(*w_base.add(7));
            let w8 = _mm256_set1_ps(*w_base.add(8));

            let inp_plane_off = inp_img_off + g * hw;

            for oh in 0..h_out {
                let mut ow = 0usize;

                // Left-edge scalar pixel: the maskload trick (ow=0, kw=0)
                // incorrectly shifts lanes 1-7 by +1 (loads row_off+1..row_off+7
                // instead of row_off+0..row_off+6), corrupting 7 output columns.
                // Process ow=0 via safe per-pixel scalar, then SIMD from ow=1.
                if w_out > 0 {
                    depthwise_conv3x3_scalar_pixel(
                        input, weight, output,
                        img, g, oh, 0, c, h, w, f, h_out, w_out,
                        1, 1, bv, activation,
                    );
                    ow = 1;
                }

                // ── SIMD main loop (8 output columns at a time) ──
                // Condition: ow + 8 < w_out ensures all input loads are in-bounds
                // (kw=2 at the last batch column loads at most ow+8 < w_out = w).
                // The scalar tail handles the rightmost column(s).
                while ow + 8 < w_out {
                    let mut acc = vbias;

                    for kr in 0..3isize {
                        let ih = oh as isize + kr - 1;
                        if ih < 0 || (ih as usize) >= h {
                            continue;
                        }
                        let row_off = inp_plane_off + (ih as usize) * w;

                        for kw in 0..3isize {
                            let iw_base = ow as isize + kw - 1;
                            let v = _mm256_loadu_ps(inp_ptr.add(row_off + iw_base as usize));

                            acc = match kw {
                                0 => match kr {
                                    0 => _mm256_fmadd_ps(w0, v, acc),
                                    1 => _mm256_fmadd_ps(w3, v, acc),
                                    _ => _mm256_fmadd_ps(w6, v, acc),
                                },
                                1 => match kr {
                                    0 => _mm256_fmadd_ps(w1, v, acc),
                                    1 => _mm256_fmadd_ps(w4, v, acc),
                                    _ => _mm256_fmadd_ps(w7, v, acc),
                                },
                                _ => match kr {
                                    0 => _mm256_fmadd_ps(w2, v, acc),
                                    1 => _mm256_fmadd_ps(w5, v, acc),
                                    _ => _mm256_fmadd_ps(w8, v, acc),
                                },
                            };
                        }
                    }

                    let sum = match activation {
                        Some(act) => apply_activation_vec(acc, act),
                        None => acc,
                    };
                    let out_base = img * (f * ho_wo) + g * ho_wo + oh * w_out + ow;
                    _mm256_storeu_ps(out_ptr.add(out_base), sum);

                    ow += 8;
                }

                // ── Scalar tail (rightmost columns) ──
                while ow < w_out {
                    depthwise_conv3x3_scalar_pixel(
                        input, weight, output,
                        img, g, oh, ow, c, h, w, f, h_out, w_out,
                        1, 1, bv, activation,
                    );
                    ow += 1;
                }
            }
        }
    }
}

// ── Top-level depthwise 3×3 conv ────────────────────────────────

/// 3×3 depthwise convolution with fused bias+activation.
///
/// Only stride-1 pad-1 uses AVX2; other configurations fall back to scalar.
pub fn direct_depthwise_conv3x3_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    groups: usize,
    stride: usize,
    padding: usize,
    activation: Option<ConvActivation>,
) {
    let _ = &groups; // used only under debug_canary
    let h_out = (h + 2 * padding).saturating_sub(3) / stride + 1;
    let w_out = (w + 2 * padding).saturating_sub(3) / stride + 1;

    // Fast path: AVX2 stride-1 pad-1
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if padding == 1 && stride == 1 && is_x86_feature_detected!("avx2") {
        #[cfg(feature = "debug_canary")]
        if !DID_AVX2.swap(true, Ordering::Relaxed) {
            eprintln!("[direct_depthwise_conv] AVX2 PATH ACTIVE (c={} h={} w={} f={} groups={})", c, h, w, f, groups);
        }
        unsafe {
            depthwise_conv3x3_f32_avx2_path(
                input, weight, bias, output,
                n, c, h, w, f, h_out, w_out, activation,
            );
        }
        return;
    }

    // Scalar path (handles stride-2, any padding)
    #[cfg(feature = "debug_canary")]
    if !DID_SCALAR.swap(true, Ordering::Relaxed) {
        eprintln!("[direct_depthwise_conv] SCALAR PATH (padding={} stride={} c={} h={} w={} f={})", padding, stride, c, h, w, f);
    }
    for img in 0..n {
        for g in 0..f {
            let bv = if bias.is_empty() { 0.0 } else { bias[g] };
            for oh in 0..h_out {
                for ow in 0..w_out {
                    depthwise_conv3x3_scalar_pixel(
                        input, weight, output,
                        img, g, oh, ow, c, h, w, f, h_out, w_out,
                        stride, padding, bv, activation,
                    );
                }
            }
        }
    }
}
