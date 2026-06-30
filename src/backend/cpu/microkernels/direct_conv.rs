//! Direct convolution kernels — no im2col expansion.
//!
//! For 3×3 conv the im2col approach expands the input by 9× (e.g. 1.6 MB
//! → 14 MB), thrashing the CPU cache.  The direct kernel reads the input
//! in NCHW layout and processes one output channel at a time, keeping the
//! working set small and cache-friendly.
//!
//! Supported: 3×3 stride-1/2, 1×1 stride-1 (via matmul view, no copy).

use std::sync::atomic::{AtomicBool, Ordering};

use super::ConvActivation;
use crate::backend::cpu::microkernels::conv::apply_conv_activation;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use core::arch::x86_64::*;

// ── Helper: apply activation to an AVX2 vector ──────────────────

use crate::backend::cpu::microkernels::activations::{exp_avx2_vec, tanh_avx2_vec};

static DID_AVX2: AtomicBool = AtomicBool::new(false);
static DID_SCALAR: AtomicBool = AtomicBool::new(false);

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

// ── Scalar: process one output pixel for 3×3 ───────────────────

fn conv3x3_scalar_pixel(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    img: usize,
    oc: usize,
    oh: usize,
    ow: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    stride: usize,
    padding: usize,
    h_out: usize,
    w_out: usize,
    activation: Option<ConvActivation>,
) {
    let bv = if bias.is_empty() { 0.0 } else { bias[oc] };
    let mut sum = bv;
    let inp_img_off = img * (c * h * w);
    let w_row_off = oc * (c * 9);

    let mut wi = w_row_off;
    for ic in 0..c {
        let ic_off = inp_img_off + ic * (h * w);
        for kh in 0..3isize {
            let ih = oh as isize * stride as isize - padding as isize + kh;
            for kw in 0..3isize {
                let iw = ow as isize * stride as isize - padding as isize + kw;
                let wv = weight[wi];
                wi += 1;
                if ih >= 0 && iw >= 0 && (ih as usize) < h && (iw as usize) < w {
                    sum += input[ic_off + (ih as usize) * w + (iw as usize)] * wv;
                }
            }
        }
    }

    let out_idx = img * (f * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
    output[out_idx] = match activation {
        Some(act) => apply_conv_activation(sum, act),
        None => sum,
    };
}

// ── AVX2: full 3×3 stride-1 pad-1 conv ────────────────────────

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn direct_conv3x3_f32_avx2_path(
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
    let f_main = f / 8 * 8;
    let nsp = h_out * w_out;

    for img in 0..n {
        let inp_img_off = img * (c * h * w);

        // ── Batched output channels (groups of 8) ──
        for oc_batch in (0..f_main).step_by(8) {
            for oh in 0..h_out {
                let mut ow = 0usize;

                // Left-edge scalar pixel: when ow=0 the SIMD loop below would
                // skip kw=0 entirely (iw_base = -1), dropping the left kernel
                // column contribution for output columns 1-7.  Process ow=0
                // via the safe per-pixel scalar function.
                if w_out > 0 {
                    for i in 0..8 {
                        conv3x3_scalar_pixel(
                            input, weight, bias, output,
                            img, oc_batch + i, oh, 0,
                            c, h, w, f, 1, 1, h_out, w_out, activation,
                        );
                    }
                    ow = 1;
                }

                // interior 8-column SIMD blocks
                // Use strict < to avoid OOB read: kw=2 loads 8 floats starting at
                // ow+1, so max index = ow+1+7 = ow+8.  When ow+8 == w_out that
                // reads one past the valid range.  The scalar tail handles the
                // rightmost column(s).
                while ow + 8 < w_out {
                    // 8 accumulators, one per output channel in the batch
                    let mut s = [_mm256_setzero_ps(); 8];
                    for i in 0..8 {
                        let bv = if bias.is_empty() { 0.0 } else { bias[oc_batch + i] };
                        s[i] = _mm256_set1_ps(bv);
                    }

                    for ic in 0..c {
                        let ic_off = inp_img_off + ic * (h * w);

                        for kh in 0..3isize {
                            let ih = oh as isize + kh - 1;
                            if ih < 0 || (ih as usize) >= h { continue; }
                            let row_off = ic_off + (ih as usize) * w;

                            for kw in 0..3isize {
                                let iw_base = ow as isize + kw - 1;
                                if iw_base < 0 { continue; }

                                let src = inp_ptr.add(row_off + iw_base as usize);
                                let v = _mm256_loadu_ps(src);
                                let w_pos = (kh * 3 + kw) as usize;

                                // Load weight for each oc and FMA
                                // weight[oc][ic][kh][kw] = weight[oc * c * 9 + ic * 9 + kh * 3 + kw]
                                let w0 = *wgt_ptr.add((oc_batch + 0) * (c * 9) + ic * 9 + w_pos);
                                let w1 = *wgt_ptr.add((oc_batch + 1) * (c * 9) + ic * 9 + w_pos);
                                let w2 = *wgt_ptr.add((oc_batch + 2) * (c * 9) + ic * 9 + w_pos);
                                let w3 = *wgt_ptr.add((oc_batch + 3) * (c * 9) + ic * 9 + w_pos);
                                let w4 = *wgt_ptr.add((oc_batch + 4) * (c * 9) + ic * 9 + w_pos);
                                let w5 = *wgt_ptr.add((oc_batch + 5) * (c * 9) + ic * 9 + w_pos);
                                let w6 = *wgt_ptr.add((oc_batch + 6) * (c * 9) + ic * 9 + w_pos);
                                let w7 = *wgt_ptr.add((oc_batch + 7) * (c * 9) + ic * 9 + w_pos);

                                s[0] = _mm256_fmadd_ps(_mm256_set1_ps(w0), v, s[0]);
                                s[1] = _mm256_fmadd_ps(_mm256_set1_ps(w1), v, s[1]);
                                s[2] = _mm256_fmadd_ps(_mm256_set1_ps(w2), v, s[2]);
                                s[3] = _mm256_fmadd_ps(_mm256_set1_ps(w3), v, s[3]);
                                s[4] = _mm256_fmadd_ps(_mm256_set1_ps(w4), v, s[4]);
                                s[5] = _mm256_fmadd_ps(_mm256_set1_ps(w5), v, s[5]);
                                s[6] = _mm256_fmadd_ps(_mm256_set1_ps(w6), v, s[6]);
                                s[7] = _mm256_fmadd_ps(_mm256_set1_ps(w7), v, s[7]);
                            }
                        }
                    }

                    for i in 0..8 {
                        let sum = match activation {
                            Some(act) => apply_activation_vec(s[i], act),
                            None => s[i],
                        };
                        let out_base = img * (f * nsp) + (oc_batch + i) * nsp + oh * w_out + ow;
                        _mm256_storeu_ps(out_ptr.add(out_base), sum);
                    }

                    ow += 8;
                }

                // scalar tail for rightmost columns
                while ow < w_out {
                    for i in 0..8 {
                        conv3x3_scalar_pixel(
                            input, weight, bias, output,
                            img, oc_batch + i, oh, ow,
                            c, h, w, f, 1, 1, h_out, w_out, activation,
                        );
                    }
                    ow += 1;
                }
            }
        }

        // ── Tail output channels (f % 8) via single-oc loop ──
        for oc in f_main..f {
            let bv = if bias.is_empty() { 0.0 } else { bias[oc] };
            let w_base = wgt_ptr.add(oc * (c * 9));

            for oh in 0..h_out {
                let mut ow = 0usize;

                // Left-edge scalar pixel — same fix as the batched loop above.
                if w_out > 0 {
                    conv3x3_scalar_pixel(
                        input, weight, bias, output,
                        img, oc, oh, 0,
                        c, h, w, f, 1, 1, h_out, w_out, activation,
                    );
                    ow = 1;
                }

                // Strict < for same reason as above — avoid OOB read at right edge.
                while ow + 8 < w_out {
                    let mut sum = _mm256_set1_ps(bv);

                    for ic in 0..c {
                        let ic_off = inp_img_off + ic * (h * w);
                        let w_ptr = w_base.add(ic * 9);

                        for kh in 0..3isize {
                            let ih = oh as isize + kh - 1;
                            if ih < 0 || (ih as usize) >= h { continue; }
                            let row_off = ic_off + (ih as usize) * w;

                            for kw in 0..3isize {
                                let iw_base = ow as isize + kw - 1;
                                if iw_base < 0 { continue; }

                                let wv = _mm256_set1_ps(*w_ptr.add((kh * 3 + kw) as usize));
                                let src = inp_ptr.add(row_off + iw_base as usize);
                                let v = _mm256_loadu_ps(src);
                                sum = _mm256_fmadd_ps(wv, v, sum);
                            }
                        }
                    }

                    sum = match activation {
                        Some(act) => apply_activation_vec(sum, act),
                        None => sum,
                    };
                    let out_base = img * (f * nsp) + oc * nsp + oh * w_out + ow;
                    _mm256_storeu_ps(out_ptr.add(out_base), sum);
                    ow += 8;
                }

                while ow < w_out {
                    conv3x3_scalar_pixel(
                        input, weight, bias, output,
                        img, oc, oh, ow,
                        c, h, w, f, 1, 1, h_out, w_out, activation,
                    );
                    ow += 1;
                }
            }
        }
    }
}

// ── Top-level 3×3 direct conv ─────────────────────────────────

/// 3×3 direct convolution with fused bias+activation.
/// stride-1 pad-1 uses AVX2; stride-2 uses scalar.
pub fn direct_conv3x3_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    stride: usize,
    padding: usize,
    activation: Option<ConvActivation>,
) {
    let h_out = (h + 2 * padding).saturating_sub(3) / stride + 1;
    let w_out = (w + 2 * padding).saturating_sub(3) / stride + 1;

    // Fast path: AVX2 stride-1 pad-1
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if padding == 1 && stride == 1 && is_x86_feature_detected!("avx2") {
        if !DID_AVX2.swap(true, Ordering::Relaxed) {
            eprintln!("[direct_conv3x3] AVX2 PATH ACTIVE (c={} h={} w={} f={})", c, h, w, f);
        }
        unsafe {
            direct_conv3x3_f32_avx2_path(
                input, weight, bias, output,
                n, c, h, w, f, h_out, w_out, activation,
            );
        }
        return;
    }

    // Scalar path (handles stride-2, any padding)
    if !DID_SCALAR.swap(true, Ordering::Relaxed) {
        eprintln!("[direct_conv3x3] SCALAR PATH (padding={} stride={} c={} h={} w={} f={})", padding, stride, c, h, w, f);
    }
    for img in 0..n {
        for oc in 0..f {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    conv3x3_scalar_pixel(
                        input, weight, bias, output,
                        img, oc, oh, ow,
                        c, h, w, f, stride, padding,
                        h_out, w_out, activation,
                    );
                }
            }
        }
    }
}

// ── 1×1 convolution (via matmul view) ─────────────────────────

/// 1×1 convolution — dispatch directly to matrixmultiply with views.
/// Bypasses im2col entirely (no expansion needed).
pub fn direct_conv1x1_f32(
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
    activation: Option<ConvActivation>,
) {
    let c_per_group = c / groups;
    let f_per_group = f / groups;
    let sp = h * w;

    for g in 0..groups {
        let c_g = c_per_group;
        let f_g = f_per_group;
        let f_off = g * f_g;
        let w_off = g * f_g * c_g;
        let wo = f_g * sp;
        let group_bias = if !bias.is_empty() {
            Some(&bias[f_off..f_off + f_g])
        } else {
            None
        };

        for img in 0..n {
            let inp_img = img * (c * h * w) + g * c_g * sp;
            let out_img = img * (f * sp) + g * wo;

            unsafe {
                crate::backend::cpu::microkernels::conv_gemm::conv_gemm_f32(
                    f_g,
                    c_g,
                    sp,
                    weight.as_ptr().add(w_off),
                    c_g as isize,
                    1isize,
                    input.as_ptr().add(inp_img),
                    sp as isize,
                    1isize,
                    output.as_mut_ptr().add(out_img),
                    sp as isize,
                    1isize,
                    group_bias,
                    activation,
                );
            }
        }
    }
}
