//! CPU ops microkernels (norm, pool, scalar, optimizers) — extracted from microkernels.rs

#![allow(dead_code, unused_imports)]

use crate::dtypes::{F32x1, I4x8, I8x4, PackedWord};
use crate::packed_tensor::PackedTensor;
use std::sync::OnceLock;

use super::*;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

// BatchNorm inference — scalar + AVX2
// ============================================================

/// BatchNorm inference: out[i] = data[i] * scale[ch] + shift[ch] where
///   scale[ch] = weight[ch] / sqrt(var[ch] + eps)
///   shift[ch] = bias[ch] - mean[ch] * scale[ch]
///   ch = i % c  (matches existing dispatch behavior)
pub fn batch_norm_inference_f32(
    data: &[f32],
    weight: &[f32],
    bias: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let c = weight.len();
    debug_assert!(c > 0);
    debug_assert_eq!(bias.len(), c);
    debug_assert_eq!(running_mean.len(), c);
    debug_assert_eq!(running_var.len(), c);
    debug_assert_eq!(data.len(), output.len());
    debug_assert!(data.len().is_multiple_of(c));
    let len = output.len();
    // Pre-compute per-channel scale and shift
    for ch in 0..c {
        let scale = weight[ch] / (running_var[ch] + eps).sqrt();
        let shift = bias[ch] - running_mean[ch] * scale;
        // Process all positions where i % c == ch
        let mut i = ch;
        while i < len {
            output[i] = data[i] * scale + shift;
            i += c;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure all slice arguments are valid and non-overlapping,
// with `output.len()` matching `data.len()`, and at least 8 elements.
pub unsafe fn batch_norm_inference_f32_avx2(
    data: &[f32],
    weight: &[f32],
    bias: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let c = weight.len();
    debug_assert!(c > 0);
    debug_assert_eq!(bias.len(), c);
    debug_assert_eq!(running_mean.len(), c);
    debug_assert_eq!(running_var.len(), c);
    debug_assert_eq!(data.len(), output.len());
    debug_assert!(data.len().is_multiple_of(c));
    let len = output.len();
    // Pre-compute per-channel scale and shift
    let mut scale = vec![0.0f32; c];
    let mut shift = vec![0.0f32; c];
    for ch in 0..c {
        scale[ch] = weight[ch] / (running_var[ch] + eps).sqrt();
        shift[ch] = bias[ch] - running_mean[ch] * scale[ch];
    }

    let vscale_base = scale.as_ptr();
    let vshift_base = shift.as_ptr();
    let vc = _mm256_set1_ps(c as f32);
    let vinc = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);

    let mut i = 0usize;
    while i + 8 <= len {
        // Channel indices ch_j = (i + j) % c  via float:  idx - floor(idx/c)*c
        let vi = _mm256_set1_ps(i as f32);
        let vindices_f = _mm256_add_ps(vi, vinc);
        let vdiv = _mm256_round_ps(_mm256_div_ps(vindices_f, vc), _MM_FROUND_TO_NEG_INF);
        let vch_f = _mm256_sub_ps(vindices_f, _mm256_mul_ps(vdiv, vc));
        let vch = _mm256_cvttps_epi32(vch_f);

        // Gather scale[ch] and shift[ch]
        let vscale = _mm256_i32gather_ps(vscale_base, vch, 4);
        let vshift = _mm256_i32gather_ps(vshift_base, vch, 4);

        // out[i] = data[i] * scale + shift
        let vdata = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_fmadd_ps(vdata, vscale, vshift),
        );
        i += 8;
    }
    for j in i..len {
        let ch = j % c;
        output[j] = data[j] * scale[ch] + shift[ch];
    }
}

// ============================================================
// Pooling microkernels — MaxPool2d + AvgPool2d (scalar + AVX2)
// ============================================================

#[inline]
pub fn pool_max_f32_scalar(
    input: &[f32],
    output: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kernel: usize,
    stride_val: usize,
    padding_val: usize,
    h_out: usize,
    w_out: usize,
    mut indices_out: Option<&mut [i64]>,
) {
    let hw_out = h_out * w_out;
    for nn in 0..n {
        for cc in 0..c {
            for hh in 0..h_out {
                for ww in 0..w_out {
                    let mut val = f32::NEG_INFINITY;
                    let mut best_kh = 0usize;
                    let mut best_kw = 0usize;
                    for kh in 0..kernel {
                        for kw in 0..kernel {
                            let h_in = hh * stride_val + kh;
                            let w_in = ww * stride_val + kw;
                            if h_in >= padding_val && w_in >= padding_val {
                                let h_in_s = h_in - padding_val;
                                let w_in_s = w_in - padding_val;
                                if h_in_s < h && w_in_s < w {
                                    let idx = nn * (c * h * w) + cc * (h * w) + h_in_s * w + w_in_s;
                                    if idx < input.len() {
                                        if input[idx] > val {
                                            val = input[idx];
                                            best_kh = kh;
                                            best_kw = kw;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let out_idx = nn * (c * hw_out) + cc * hw_out + hh * w_out + ww;
                    if out_idx < output.len() {
                        output[out_idx] = val;
                    }
                    if let Some(ref mut idx_out) = indices_out {
                        if out_idx < idx_out.len() {
                            idx_out[out_idx] = (best_kh * kernel + best_kw) as i64;
                        }
                    }
                }
            }
        }
    }
}

#[inline]
pub fn pool_avg_f32_scalar(
    input: &[f32],
    output: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kernel: usize,
    stride_val: usize,
    padding_val: usize,
    h_out: usize,
    w_out: usize,
) {
    let hw_out = h_out * w_out;
    for nn in 0..n {
        for cc in 0..c {
            for hh in 0..h_out {
                for ww in 0..w_out {
                    let mut val = 0.0f32;
                    let mut count = 0usize;
                    for kh in 0..kernel {
                        for kw in 0..kernel {
                            let h_in = hh * stride_val + kh;
                            let w_in = ww * stride_val + kw;
                            if h_in >= padding_val && w_in >= padding_val {
                                let h_in_s = h_in - padding_val;
                                let w_in_s = w_in - padding_val;
                                if h_in_s < h && w_in_s < w {
                                    let idx = nn * (c * h * w) + cc * (h * w) + h_in_s * w + w_in_s;
                                    if idx < input.len() {
                                        val += input[idx];
                                        count += 1;
                                    }
                                }
                            }
                        }
                    }
                    if count > 0 {
                        val /= count as f32;
                    }
                    let out_idx = nn * (c * hw_out) + cc * hw_out + hh * w_out + ww;
                    if out_idx < output.len() {
                        output[out_idx] = val;
                    }
                }
            }
        }
    }
}

#[inline]
pub fn adaptive_avg_pool2d_f32_scalar(
    input: &[f32],
    output: &mut [f32],
    nc: usize,
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
) {
    let hw = h * w;
    for nci in 0..nc {
        for ohi in 0..out_h {
            for owi in 0..out_w {
                let h_start = ohi * h / out_h;
                let h_end = (ohi + 1) * h / out_h;
                let w_start = owi * w / out_w;
                let w_end = (owi + 1) * w / out_w;
                let mut sum = 0.0f32;
                let mut count = 0;
                for hi in h_start..h_end {
                    for wi in w_start..w_end {
                        sum += input[nci * hw + hi * w + wi];
                        count += 1;
                    }
                }
                let out_idx = nci * out_h * out_w + ohi * out_w + owi;
                if out_idx < output.len() {
                    output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and sized according to the pooling dimensions (n, c, h, w, kernel, ...).
pub unsafe fn pool_max_f32_avx2(
    input: &[f32],
    output: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kernel: usize,
    stride_val: usize,
    padding_val: usize,
    h_out: usize,
    w_out: usize,
    indices_out: Option<&mut [i64]>,
) {
    if stride_val != 1 || indices_out.is_some() {
        return pool_max_f32_scalar(
            input,
            output,
            n,
            c,
            h,
            w,
            kernel,
            stride_val,
            padding_val,
            h_out,
            w_out,
            indices_out,
        );
    }
    let hw_out = h_out * w_out;

    // Interior hh where all kh are valid (stride=1): hh in [padding_val, padding_val + h - kernel]
    let interior_h_start = if padding_val < h_out {
        padding_val
    } else {
        h_out
    };
    let interior_h_end = if padding_val + h >= kernel {
        (padding_val + h - kernel + 1).min(h_out)
    } else {
        interior_h_start
    };

    // Interior ww where 8 consecutive outputs all have valid kw: ww in [padding_val, w+padding_val-kernel-7]
    // exclusive upper bound: min(w_out, w+padding_val-kernel-6)
    let interior_w_start = if padding_val < w_out {
        padding_val
    } else {
        w_out
    };
    let interior_w_end = if w + padding_val >= kernel + 7 {
        (w + padding_val - kernel - 6).min(w_out)
    } else {
        interior_w_start
    };

    let vneg_inf = _mm256_set1_ps(f32::NEG_INFINITY);

    for nn in 0..n {
        for cc in 0..c {
            let ch_in_off = nn * (c * h * w) + cc * (h * w);
            let ch_out_off = nn * (c * hw_out) + cc * hw_out;

            // Top edge rows (scalar)
            for hh in 0..interior_h_start {
                for ww in 0..w_out {
                    let mut val = f32::NEG_INFINITY;
                    for kh in 0..kernel {
                        let h_in = hh + kh;
                        if h_in >= padding_val && h_in - padding_val < h {
                            let rbase = ch_in_off + (h_in - padding_val) * w;
                            for kw in 0..kernel {
                                let w_in = ww + kw;
                                if w_in >= padding_val && w_in - padding_val < w {
                                    let inp = input[rbase + w_in - padding_val];
                                    if inp > val {
                                        val = inp;
                                    }
                                }
                            }
                        }
                    }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }
            }

            // Interior rows
            for hh in interior_h_start..interior_h_end {
                // Left edge (scalar)
                for ww in 0..interior_w_start {
                    let mut val = f32::NEG_INFINITY;
                    let h_s = hh - padding_val;
                    for kh in 0..kernel {
                        let rbase = ch_in_off + (h_s + kh) * w;
                        for kw in 0..kernel {
                            let w_in = ww + kw;
                            if w_in >= padding_val && w_in - padding_val < w {
                                let inp = input[rbase + w_in - padding_val];
                                if inp > val {
                                    val = inp;
                                }
                            }
                        }
                    }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }

                // Interior W (SIMD: 8 outputs at once)
                let mut ww = interior_w_start;
                while ww + 8 <= interior_w_end {
                    let mut vmax = vneg_inf;
                    let h_s = hh - padding_val;
                    for kh in 0..kernel {
                        let rbase = ch_in_off + (h_s + kh) * w;
                        for kw in 0..kernel {
                            let load_off = rbase + ww + kw - padding_val;
                            let v = _mm256_loadu_ps(input.as_ptr().add(load_off));
                            vmax = _mm256_max_ps(vmax, v);
                        }
                    }
                    _mm256_storeu_ps(output.as_mut_ptr().add(ch_out_off + hh * w_out + ww), vmax);
                    ww += 8;
                }

                // Right edge (scalar)
                for ww in ww..w_out {
                    let mut val = f32::NEG_INFINITY;
                    let h_s = hh - padding_val;
                    for kh in 0..kernel {
                        let rbase = ch_in_off + (h_s + kh) * w;
                        for kw in 0..kernel {
                            let w_in = ww + kw;
                            if w_in >= padding_val && w_in - padding_val < w {
                                let inp = input[rbase + w_in - padding_val];
                                if inp > val {
                                    val = inp;
                                }
                            }
                        }
                    }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }
            }

            // Bottom edge rows (scalar)
            for hh in interior_h_end..h_out {
                for ww in 0..w_out {
                    let mut val = f32::NEG_INFINITY;
                    for kh in 0..kernel {
                        let h_in = hh + kh;
                        if h_in >= padding_val && h_in - padding_val < h {
                            let rbase = ch_in_off + (h_in - padding_val) * w;
                            for kw in 0..kernel {
                                let w_in = ww + kw;
                                if w_in >= padding_val && w_in - padding_val < w {
                                    let inp = input[rbase + w_in - padding_val];
                                    if inp > val {
                                        val = inp;
                                    }
                                }
                            }
                        }
                    }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }
            }
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as pool_max_f32_avx2 — caller ensures valid, non-overlapping
// input/output buffers sized per the pooling parameters.
pub unsafe fn pool_avg_f32_avx2(
    input: &[f32],
    output: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kernel: usize,
    stride_val: usize,
    padding_val: usize,
    h_out: usize,
    w_out: usize,
) {
    if stride_val != 1 {
        return pool_avg_f32_scalar(
            input,
            output,
            n,
            c,
            h,
            w,
            kernel,
            stride_val,
            padding_val,
            h_out,
            w_out,
        );
    }
    let hw_out = h_out * w_out;
    let kernel_area = (kernel * kernel) as f32;

    let interior_h_start = if padding_val < h_out {
        padding_val
    } else {
        h_out
    };
    let interior_h_end = if padding_val + h >= kernel {
        (padding_val + h - kernel + 1).min(h_out)
    } else {
        interior_h_start
    };
    let interior_w_start = if padding_val < w_out {
        padding_val
    } else {
        w_out
    };
    let interior_w_end = if w + padding_val >= kernel + 7 {
        (w + padding_val - kernel - 6).min(w_out)
    } else {
        interior_w_start
    };

    let vkernel_area = _mm256_set1_ps(kernel_area);

    for nn in 0..n {
        for cc in 0..c {
            let ch_in_off = nn * (c * h * w) + cc * (h * w);
            let ch_out_off = nn * (c * hw_out) + cc * hw_out;

            // Top edge rows (scalar)
            for hh in 0..interior_h_start {
                for ww in 0..w_out {
                    let mut val = 0.0f32;
                    let mut count = 0usize;
                    for kh in 0..kernel {
                        let h_in = hh + kh;
                        if h_in >= padding_val && h_in - padding_val < h {
                            let rbase = ch_in_off + (h_in - padding_val) * w;
                            for kw in 0..kernel {
                                let w_in = ww + kw;
                                if w_in >= padding_val && w_in - padding_val < w {
                                    val += input[rbase + w_in - padding_val];
                                    count += 1;
                                }
                            }
                        }
                    }
                    if count > 0 {
                        val /= count as f32;
                    }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }
            }

            // Interior rows
            for hh in interior_h_start..interior_h_end {
                // Left edge (scalar)
                for ww in 0..interior_w_start {
                    let mut val = 0.0f32;
                    let mut count = 0usize;
                    let h_s = hh - padding_val;
                    for kh in 0..kernel {
                        let rbase = ch_in_off + (h_s + kh) * w;
                        for kw in 0..kernel {
                            let w_in = ww + kw;
                            if w_in >= padding_val && w_in - padding_val < w {
                                val += input[rbase + w_in - padding_val];
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        val /= count as f32;
                    }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }

                // Interior W (SIMD)
                let mut ww = interior_w_start;
                while ww + 8 <= interior_w_end {
                    let mut vacc = _mm256_setzero_ps();
                    let h_s = hh - padding_val;
                    for kh in 0..kernel {
                        let rbase = ch_in_off + (h_s + kh) * w;
                        for kw in 0..kernel {
                            let load_off = rbase + ww + kw - padding_val;
                            let v = _mm256_loadu_ps(input.as_ptr().add(load_off));
                            vacc = _mm256_add_ps(vacc, v);
                        }
                    }
                    let vavg = _mm256_div_ps(vacc, vkernel_area);
                    _mm256_storeu_ps(output.as_mut_ptr().add(ch_out_off + hh * w_out + ww), vavg);
                    ww += 8;
                }

                // Right edge (scalar)
                for ww in ww..w_out {
                    let mut val = 0.0f32;
                    let mut count = 0usize;
                    let h_s = hh - padding_val;
                    for kh in 0..kernel {
                        let rbase = ch_in_off + (h_s + kh) * w;
                        for kw in 0..kernel {
                            let w_in = ww + kw;
                            if w_in >= padding_val && w_in - padding_val < w {
                                val += input[rbase + w_in - padding_val];
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        val /= count as f32;
                    }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }
            }

            // Bottom edge rows (scalar)
            for hh in interior_h_end..h_out {
                for ww in 0..w_out {
                    let mut val = 0.0f32;
                    let mut count = 0usize;
                    for kh in 0..kernel {
                        let h_in = hh + kh;
                        if h_in >= padding_val && h_in - padding_val < h {
                            let rbase = ch_in_off + (h_in - padding_val) * w;
                            for kw in 0..kernel {
                                let w_in = ww + kw;
                                if w_in >= padding_val && w_in - padding_val < w {
                                    val += input[rbase + w_in - padding_val];
                                    count += 1;
                                }
                            }
                        }
                    }
                    if count > 0 {
                        val /= count as f32;
                    }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }
            }
        }
    }
}

// ============================================================
// Remaining scalar ops — AVX2 microkernels for ops without dedicated kernels
// ============================================================

// ── reduce_sum_f32 — group sum/mean ─────────────────────────

#[inline]
pub fn reduce_sum_f32_scalar(input: &[f32], output: &mut [f32], group_size: usize, is_mean: bool) {
    debug_assert!(group_size > 0);
    debug_assert!(input.len().is_multiple_of(group_size));
    debug_assert_eq!(input.len() / group_size, output.len());
    let num_groups = output.len();
    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        let mut sum = 0.0f32;
        for i in start..end {
            sum += input[i];
        }
        output[g] = if is_mean {
            sum / group_size as f32
        } else {
            sum
        };
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and `input` has at least `num_groups * group_size` elements.
pub unsafe fn reduce_sum_f32_avx2(
    input: &[f32],
    output: &mut [f32],
    group_size: usize,
    is_mean: bool,
) {
    debug_assert!(group_size > 0);
    debug_assert!(input.len().is_multiple_of(group_size));
    debug_assert_eq!(input.len() / group_size, output.len());
    let num_groups = output.len();
    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        let mut i = start;
        let mut acc = _mm256_setzero_ps();
        while i + 8 <= end {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(input.as_ptr().add(i)));
            i += 8;
        }
        let mut sum = hsum256_ps(acc);
        for j in i..end {
            sum += input[j];
        }
        output[g] = if is_mean {
            sum / group_size as f32
        } else {
            sum
        };
    }
}

// ── reduce_max_f32 — group max ──────────────────────────────

#[inline]
pub fn reduce_max_f32_scalar(input: &[f32], output: &mut [f32], group_size: usize) {
    debug_assert!(group_size > 0);
    debug_assert!(input.len().is_multiple_of(group_size));
    debug_assert_eq!(input.len() / group_size, output.len());
    let num_groups = output.len();
    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        let mut val = f32::NEG_INFINITY;
        for i in start..end {
            if input[i] > val {
                val = input[i];
            }
        }
        output[g] = val;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as reduce_sum_f32_avx2 — caller ensures valid, non-overlapping
// slices with `input` sized at least `num_groups * group_size`.
pub unsafe fn reduce_max_f32_avx2(input: &[f32], output: &mut [f32], group_size: usize) {
    debug_assert!(group_size > 0);
    debug_assert!(input.len().is_multiple_of(group_size));
    debug_assert_eq!(input.len() / group_size, output.len());
    let num_groups = output.len();
    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        let mut i = start;
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        while i + 8 <= end {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(input.as_ptr().add(i)));
            i += 8;
        }
        // horizontal max
        let mx128 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        let mx = _mm_max_ps(mx128, _mm_shuffle_ps(mx128, mx128, 0x0E));
        let mx = _mm_max_ps(mx, _mm_shuffle_ps(mx, mx, 0x01));
        let mut max_val = _mm_cvtss_f32(mx);
        for j in i..end {
            if input[j] > max_val {
                max_val = input[j];
            }
        }
        output[g] = max_val;
    }
}

// ── biasadd ─────────────────────────────────────────────────

#[inline]
pub fn biasadd_f32_scalar(data: &[f32], bias: &[f32], output: &mut [f32], channel_stride: usize) {
    debug_assert_eq!(data.len(), output.len());
    debug_assert!(!bias.is_empty());
    debug_assert!(channel_stride > 0);
    let len = output.len();
    let bias_len = bias.len();
    for i in 0..len {
        output[i] = data[i] + bias[(i / channel_stride) % bias_len];
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `data` and `output` are valid, non-overlapping
// and at least `len` elements; `bias` is valid for the channel index range.
pub unsafe fn biasadd_f32_avx2(
    data: &[f32],
    bias: &[f32],
    output: &mut [f32],
    channel_stride: usize,
) {
    debug_assert_eq!(data.len(), output.len());
    debug_assert!(!bias.is_empty());
    debug_assert!(channel_stride > 0);
    let len = output.len();
    let bias_len = bias.len();
    let mut i = 0;
    while i + 8 <= len {
        let vdata = _mm256_loadu_ps(data.as_ptr().add(i));
        let b0 = bias[(i / channel_stride) % bias_len];
        let b1 = bias[((i + 1) / channel_stride) % bias_len];
        let b2 = bias[((i + 2) / channel_stride) % bias_len];
        let b3 = bias[((i + 3) / channel_stride) % bias_len];
        let b4 = bias[((i + 4) / channel_stride) % bias_len];
        let b5 = bias[((i + 5) / channel_stride) % bias_len];
        let b6 = bias[((i + 6) / channel_stride) % bias_len];
        let b7 = bias[((i + 7) / channel_stride) % bias_len];
        let vbias = _mm256_set_ps(b7, b6, b5, b4, b3, b2, b1, b0);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_add_ps(vdata, vbias));
        i += 8;
    }
    for j in i..len {
        *output.as_mut_ptr().add(j) = data[j] + bias[(j / channel_stride) % bias_len];
    }
}

// ── norm_layernorm_f32 — LayerNorm ──────────────────────────

/// Single-pass Welford-based LayerNorm: computes mean + variance in one pass
/// instead of two, reducing memory traffic by ~33%.
#[inline]
pub fn norm_layernorm_f32_scalar(input: &[f32], output: &mut [f32], row_size: usize, eps: f32) {
    debug_assert!(row_size > 0);
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(input.len().is_multiple_of(row_size));
    let num_rows = input.len() / row_size;
    for r in 0..num_rows {
        let start = r * row_size;
        let end = start + row_size;
        // Welford's online algorithm: single pass for mean + variance
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;
        let mut count = 0u32;
        for i in start..end {
            count += 1;
            let x = input[i];
            let delta = x - mean;
            mean += delta / (count as f32);
            let delta2 = x - mean;
            m2 += delta * delta2;
        }
        let var = if count > 1 { m2 / (count as f32) } else { 0.0 };
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in start..end {
            output[i] = (input[i] - mean) * inv_std;
        }
    }
}

#[inline]
pub fn fused_residual_add_layer_norm_f32_scalar(
    residual: &[f32],
    main: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    row_size: usize,
    eps: f32,
) {
    debug_assert!(row_size > 0);
    debug_assert_eq!(main.len(), output.len());
    debug_assert_eq!(residual.len(), output.len());
    debug_assert_eq!(weight.len(), row_size);
    debug_assert_eq!(bias.len(), row_size);
    debug_assert!(output.len().is_multiple_of(row_size));
    let num_rows = output.len() / row_size;
    for r in 0..num_rows {
        let start = r * row_size;
        let end = start + row_size;

        // Welford single-pass mean + variance on (main + residual)
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;
        let mut count = 0u32;
        for i in start..end {
            count += 1;
            let x = main[i] + residual[i];
            let delta = x - mean;
            mean += delta / (count as f32);
            let delta2 = x - mean;
            m2 += delta * delta2;
        }
        let var = if count > 1 { m2 / (count as f32) } else { 0.0 };
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in start..end {
            let idx = i - start;
            output[i] = (main[i] + residual[i] - mean) * inv_std * weight[idx] + bias[idx];
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and each at least `num_rows * row_size` elements.
pub unsafe fn norm_layernorm_f32_avx2(
    input: &[f32],
    output: &mut [f32],
    row_size: usize,
    eps: f32,
) {
    debug_assert!(row_size > 0);
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(input.len().is_multiple_of(row_size));
    let num_rows = input.len() / row_size;
    for r in 0..num_rows {
        let start = r * row_size;
        let end = start + row_size;
        let n = row_size as f32;
        // Pass 1: vectorized sum
        let mut i = start;
        let mut vsum = _mm256_setzero_ps();
        while i + 8 <= end {
            vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(input.as_ptr().add(i)));
            i += 8;
        }
        let mut sum = hsum256_ps(vsum);
        for j in i..end {
            sum += input[j];
        }
        let mean = if n > 0.0 { sum / n } else { 0.0 };
        let vmean = _mm256_set1_ps(mean);
        // Pass 2: vectorized sum of (x - mean)^2
        i = start;
        let mut vvar = _mm256_setzero_ps();
        while i + 8 <= end {
            let d = _mm256_sub_ps(_mm256_loadu_ps(input.as_ptr().add(i)), vmean);
            vvar = _mm256_fmadd_ps(d, d, vvar);
            i += 8;
        }
        let mut var = hsum256_ps(vvar);
        for j in i..end {
            let d = input[j] - mean;
            var += d * d;
        }
        var /= n;
        let inv_std = 1.0 / (var + eps).sqrt();
        // Pass 3: vectorized normalize
        let vinv_std = _mm256_set1_ps(inv_std);
        i = start;
        while i + 8 <= end {
            _mm256_storeu_ps(
                output.as_mut_ptr().add(i),
                _mm256_mul_ps(
                    _mm256_sub_ps(_mm256_loadu_ps(input.as_ptr().add(i)), vmean),
                    vinv_std,
                ),
            );
            i += 8;
        }
        for j in i..end {
            output[j] = (input[j] - mean) * inv_std;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `residual`, `main`, `weight`, `bias`, and `output` are valid,
// non-overlapping, and each at least `num_rows * row_size` elements.
pub unsafe fn fused_residual_add_layer_norm_f32_avx2(
    residual: &[f32],
    main: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    row_size: usize,
    eps: f32,
) {
    debug_assert!(row_size > 0);
    debug_assert_eq!(main.len(), output.len());
    debug_assert_eq!(residual.len(), output.len());
    debug_assert_eq!(weight.len(), row_size);
    debug_assert_eq!(bias.len(), row_size);
    debug_assert!(output.len().is_multiple_of(row_size));
    let num_rows = output.len() / row_size;
    for r in 0..num_rows {
        let start = r * row_size;
        let end = start + row_size;
        let n = row_size as f32;

        // Pass 1: vectorized sum of (main + residual)
        let mut i = start;
        let mut vsum = _mm256_setzero_ps();
        while i + 8 <= end {
            let vm = _mm256_loadu_ps(main.as_ptr().add(i));
            let vr = _mm256_loadu_ps(residual.as_ptr().add(i));
            vsum = _mm256_add_ps(vsum, _mm256_add_ps(vm, vr));
            i += 8;
        }
        let mut sum = hsum256_ps(vsum);
        for j in i..end {
            sum += main[j] + residual[j];
        }
        let mean = if n > 0.0 { sum / n } else { 0.0 };
        let vmean = _mm256_set1_ps(mean);

        // Pass 2: vectorized sum of ((main + residual) - mean)^2
        i = start;
        let mut vvar = _mm256_setzero_ps();
        while i + 8 <= end {
            let vm = _mm256_loadu_ps(main.as_ptr().add(i));
            let vr = _mm256_loadu_ps(residual.as_ptr().add(i));
            let d = _mm256_sub_ps(_mm256_add_ps(vm, vr), vmean);
            vvar = _mm256_fmadd_ps(d, d, vvar);
            i += 8;
        }
        let mut var = hsum256_ps(vvar);
        for j in i..end {
            let d = main[j] + residual[j] - mean;
            var += d * d;
        }
        var /= n;
        let inv_std = 1.0 / (var + eps).sqrt();
        let vinv_std = _mm256_set1_ps(inv_std);

        // Pass 3: vectorized normalize with weight + bias
        i = start;
        while i + 8 <= end {
            let idx = i - start;
            let vm = _mm256_loadu_ps(main.as_ptr().add(i));
            let vr = _mm256_loadu_ps(residual.as_ptr().add(i));
            let vnorm = _mm256_mul_ps(_mm256_sub_ps(_mm256_add_ps(vm, vr), vmean), vinv_std);
            let vw = _mm256_loadu_ps(weight.as_ptr().add(idx));
            let vb = _mm256_loadu_ps(bias.as_ptr().add(idx));
            _mm256_storeu_ps(
                output.as_mut_ptr().add(i),
                _mm256_add_ps(_mm256_mul_ps(vnorm, vw), vb),
            );
            i += 8;
        }
        for j in i..end {
            let idx = j - start;
            output[j] = (main[j] + residual[j] - mean) * inv_std * weight[idx] + bias[idx];
        }
    }
}

// ── rms_norm_f32 ────────────────────────────────────────────

#[inline]
pub fn rms_norm_f32_scalar(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    row_size: usize,
    eps: f32,
) {
    debug_assert!(row_size > 0);
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(weight.len(), row_size);
    debug_assert!(input.len().is_multiple_of(row_size));
    let num_rows = input.len() / row_size;
    for r in 0..num_rows {
        let start = r * row_size;
        let end = start + row_size;
        let mut sq_sum = 0.0f32;
        for i in start..end {
            sq_sum += input[i] * input[i];
        }
        let n = row_size as f32;
        let rms = if n > 0.0 {
            (sq_sum / n + eps).sqrt()
        } else {
            1.0
        };
        for i in start..end {
            output[i] = input[i] / rms * weight[i - start];
        }
    }
}

#[inline]
pub fn fused_residual_add_rms_norm_f32_scalar(
    residual: &[f32],
    main: &[f32],
    weight: &[f32],
    output: &mut [f32],
    row_size: usize,
    eps: f32,
) {
    debug_assert!(row_size > 0);
    debug_assert_eq!(main.len(), output.len());
    debug_assert_eq!(residual.len(), output.len());
    debug_assert_eq!(weight.len(), row_size);
    debug_assert!(output.len().is_multiple_of(row_size));
    let num_rows = output.len() / row_size;
    for r in 0..num_rows {
        let start = r * row_size;
        let end = start + row_size;

        let mut sq_sum = 0.0f32;
        for i in start..end {
            let x = main[i] + residual[i];
            sq_sum += x * x;
        }
        let n = row_size as f32;
        let rms = if n > 0.0 {
            (sq_sum / n + eps).sqrt()
        } else {
            1.0
        };

        for i in start..end {
            let idx = i - start;
            output[i] = (main[i] + residual[i]) / rms * weight[idx];
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `residual`, `main`, `weight`, and `output` are valid,
// non-overlapping, and each at least `num_rows * row_size` elements.
pub unsafe fn fused_residual_add_rms_norm_f32_avx2(
    residual: &[f32],
    main: &[f32],
    weight: &[f32],
    output: &mut [f32],
    row_size: usize,
    eps: f32,
) {
    debug_assert!(row_size > 0);
    debug_assert_eq!(main.len(), output.len());
    debug_assert_eq!(residual.len(), output.len());
    debug_assert_eq!(weight.len(), row_size);
    debug_assert!(output.len().is_multiple_of(row_size));
    let num_rows = output.len() / row_size;
    for r in 0..num_rows {
        let start = r * row_size;
        let end = start + row_size;
        // Vectorized sum of (main + residual)^2
        let mut i = start;
        let mut vsq = _mm256_setzero_ps();
        while i + 8 <= end {
            let vm = _mm256_loadu_ps(main.as_ptr().add(i));
            let vr = _mm256_loadu_ps(residual.as_ptr().add(i));
            let vx = _mm256_add_ps(vm, vr);
            vsq = _mm256_fmadd_ps(vx, vx, vsq);
            i += 8;
        }
        let mut sq_sum = hsum256_ps(vsq);
        for j in i..end {
            let x = main[j] + residual[j];
            sq_sum += x * x;
        }
        let n = row_size as f32;
        let rms = if n > 0.0 {
            (sq_sum / n + eps).sqrt()
        } else {
            1.0
        };
        let inv_rms = _mm256_set1_ps(1.0 / rms);
        // Vectorized writeback with weight
        i = start;
        while i + 8 <= end {
            let idx = i - start;
            let vm = _mm256_loadu_ps(main.as_ptr().add(i));
            let vr = _mm256_loadu_ps(residual.as_ptr().add(i));
            let vx = _mm256_add_ps(vm, vr);
            let w = _mm256_loadu_ps(weight.as_ptr().add(idx));
            _mm256_storeu_ps(
                output.as_mut_ptr().add(i),
                _mm256_mul_ps(_mm256_mul_ps(vx, inv_rms), w),
            );
            i += 8;
        }
        for j in i..end {
            let idx = j - start;
            output[j] = (main[j] + residual[j]) / rms * weight[idx];
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `input`, `weight`, and `output` are valid,
// non-overlapping, and each at least `num_rows * row_size` elements.
pub unsafe fn rms_norm_f32_avx2(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    row_size: usize,
    eps: f32,
) {
    debug_assert!(row_size > 0);
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(weight.len(), row_size);
    debug_assert!(input.len().is_multiple_of(row_size));
    let num_rows = input.len() / row_size;
    for r in 0..num_rows {
        let start = r * row_size;
        let end = start + row_size;
        // Vectorized sum of squares
        let mut i = start;
        let mut vsq = _mm256_setzero_ps();
        while i + 8 <= end {
            let vx = _mm256_loadu_ps(input.as_ptr().add(i));
            vsq = _mm256_fmadd_ps(vx, vx, vsq);
            i += 8;
        }
        let mut sq_sum = hsum256_ps(vsq);
        for j in i..end {
            sq_sum += input[j] * input[j];
        }
        let n = row_size as f32;
        let rms = if n > 0.0 {
            (sq_sum / n + eps).sqrt()
        } else {
            1.0
        };
        let inv_rms = _mm256_set1_ps(1.0 / rms);
        // Vectorized writeback with weight
        i = start;
        while i + 8 <= end {
            let vx = _mm256_loadu_ps(input.as_ptr().add(i));
            let w = _mm256_loadu_ps(weight.as_ptr().add(i - start));
            _mm256_storeu_ps(
                output.as_mut_ptr().add(i),
                _mm256_mul_ps(_mm256_mul_ps(vx, inv_rms), w),
            );
            i += 8;
        }
        for j in i..end {
            *output.as_mut_ptr().add(j) = input[j] / rms * weight[j - start];
        }
    }
}

// ── Scalar arithmetic ───────────────────────────────────────

#[inline]
pub fn add_scalar_f32_scalar(data: &[f32], s: f32, output: &mut [f32]) {
    debug_assert_eq!(data.len(), output.len());
    let len = output.len();
    for i in 0..len {
        output[i] = data[i] + s;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `data` and `output` are valid, non-overlapping,
// and each at least 8 elements long.
pub unsafe fn add_scalar_f32_avx2(data: &[f32], s: f32, output: &mut [f32]) {
    debug_assert_eq!(data.len(), output.len());
    let len = output.len();
    let mut i = 0;
    let vs = _mm256_set1_ps(s);
    while i + 8 <= len {
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_add_ps(_mm256_loadu_ps(data.as_ptr().add(i)), vs),
        );
        i += 8;
    }
    for j in i..len {
        *output.as_mut_ptr().add(j) = data[j] + s;
    }
}

#[inline]
pub fn mul_scalar_f32_scalar(data: &[f32], s: f32, output: &mut [f32]) {
    debug_assert_eq!(data.len(), output.len());
    let len = output.len();
    for i in 0..len {
        output[i] = data[i] * s;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as add_scalar_f32_avx2 — caller ensures valid, non-overlapping
// data/output slices with at least 8 elements.
pub unsafe fn mul_scalar_f32_avx2(data: &[f32], s: f32, output: &mut [f32]) {
    debug_assert_eq!(data.len(), output.len());
    let len = output.len();
    let mut i = 0;
    let vs = _mm256_set1_ps(s);
    while i + 8 <= len {
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_mul_ps(_mm256_loadu_ps(data.as_ptr().add(i)), vs),
        );
        i += 8;
    }
    for j in i..len {
        *output.as_mut_ptr().add(j) = data[j] * s;
    }
}

#[inline]
pub fn div_scalar_f32_scalar(data: &[f32], s: f32, output: &mut [f32]) {
    debug_assert_eq!(data.len(), output.len());
    let len = output.len();
    for i in 0..len {
        output[i] = data[i] / s;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as add_scalar_f32_avx2 — caller ensures valid, non-overlapping
// data/output slices with at least 8 elements.
pub unsafe fn div_scalar_f32_avx2(data: &[f32], s: f32, output: &mut [f32]) {
    debug_assert_eq!(data.len(), output.len());
    let len = output.len();
    let mut i = 0;
    let vs = _mm256_set1_ps(s);
    while i + 8 <= len {
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_div_ps(_mm256_loadu_ps(data.as_ptr().add(i)), vs),
        );
        i += 8;
    }
    for j in i..len {
        *output.as_mut_ptr().add(j) = data[j] / s;
    }
}

// ── Scalar comparison ───────────────────────────────────────

#[inline]
pub fn gt_scalar_f32_scalar(data: &[f32], s: f32, output: &mut [f32]) {
    debug_assert_eq!(data.len(), output.len());
    let len = output.len();
    for i in 0..len {
        output[i] = if data[i] > s { 1.0 } else { 0.0 };
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as add_scalar_f32_avx2 — caller ensures valid, non-overlapping
// data/output slices with at least 8 elements.
pub unsafe fn gt_scalar_f32_avx2(data: &[f32], s: f32, output: &mut [f32]) {
    debug_assert_eq!(data.len(), output.len());
    let len = output.len();
    let mut i = 0;
    let vs = _mm256_set1_ps(s);
    let vone = _mm256_set1_ps(1.0);
    let vzero = _mm256_setzero_ps();
    while i + 8 <= len {
        let mask = _mm256_cmp_ps::<{ _CMP_GT_OQ }>(_mm256_loadu_ps(data.as_ptr().add(i)), vs);
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_blendv_ps(vzero, vone, mask),
        );
        i += 8;
    }
    for j in i..len {
        *output.as_mut_ptr().add(j) = if data[j] > s { 1.0 } else { 0.0 };
    }
}

#[inline]
pub fn lt_scalar_f32_scalar(data: &[f32], s: f32, output: &mut [f32]) {
    debug_assert_eq!(data.len(), output.len());
    let len = output.len();
    for i in 0..len {
        output[i] = if data[i] < s { 1.0 } else { 0.0 };
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as add_scalar_f32_avx2 — caller ensures valid, non-overlapping
// data/output slices with at least 8 elements.
pub unsafe fn lt_scalar_f32_avx2(data: &[f32], s: f32, output: &mut [f32]) {
    debug_assert_eq!(data.len(), output.len());
    let len = output.len();
    let mut i = 0;
    let vs = _mm256_set1_ps(s);
    let vone = _mm256_set1_ps(1.0);
    let vzero = _mm256_setzero_ps();
    while i + 8 <= len {
        let mask = _mm256_cmp_ps::<{ _CMP_LT_OQ }>(_mm256_loadu_ps(data.as_ptr().add(i)), vs);
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_blendv_ps(vzero, vone, mask),
        );
        i += 8;
    }
    for j in i..len {
        *output.as_mut_ptr().add(j) = if data[j] < s { 1.0 } else { 0.0 };
    }
}

#[inline]
pub fn eq_scalar_f32_scalar(data: &[f32], s: f32, output: &mut [f32]) {
    debug_assert_eq!(data.len(), output.len());
    let len = output.len();
    for i in 0..len {
        output[i] = if (data[i] - s).abs() < 1e-6 { 1.0 } else { 0.0 };
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as add_scalar_f32_avx2 — caller ensures valid, non-overlapping
// data/output slices with at least 8 elements.
pub unsafe fn eq_scalar_f32_avx2(data: &[f32], s: f32, output: &mut [f32]) {
    debug_assert_eq!(data.len(), output.len());
    let len = output.len();
    let mut i = 0;
    let vs = _mm256_set1_ps(s);
    let vone = _mm256_set1_ps(1.0);
    let vzero = _mm256_setzero_ps();
    let veps = _mm256_set1_ps(1e-6);
    let vabsmask = _mm256_set1_ps(f32::from_bits(0x7fffffff));
    while i + 8 <= len {
        let vdiff = _mm256_sub_ps(_mm256_loadu_ps(data.as_ptr().add(i)), vs);
        let vabsdiff = _mm256_and_ps(vdiff, vabsmask);
        let mask = _mm256_cmp_ps::<{ _CMP_LT_OQ }>(vabsdiff, veps);
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_blendv_ps(vzero, vone, mask),
        );
        i += 8;
    }
    for j in i..len {
        *output.as_mut_ptr().add(j) = if (data[j] - s).abs() < 1e-6 { 1.0 } else { 0.0 };
    }
}

// ── softmax ─────────────────────────────────────────────────

#[inline]
pub fn softmax_f32_scalar_strided(
    input: &[f32],
    output: &mut [f32],
    axis_dim_size: usize,
    stride: usize,
    num_rows: usize,
) {
    for r in 0..num_rows {
        let outer = r / stride;
        let inner = r % stride;
        let base = (outer * axis_dim_size * stride) + inner;
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..axis_dim_size {
            let idx = base + i * stride;
            if input[idx] > max_val {
                max_val = input[idx];
            }
        }
        let mut sum = 0.0f32;
        for i in 0..axis_dim_size {
            let idx = base + i * stride;
            let e = (input[idx] - max_val).exp();
            output[idx] = e;
            sum += e;
        }
        if sum > 0.0 {
            for i in 0..axis_dim_size {
                let idx = base + i * stride;
                output[idx] /= sum;
            }
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and each at least `num_rows * axis_dim_size * stride` elements.
pub unsafe fn softmax_f32_avx2_strided(
    input: &[f32],
    output: &mut [f32],
    axis_dim_size: usize,
    stride: usize,
    num_rows: usize,
) {
    if stride != 1 {
        // Strided access — fall back to scalar for correctness
        softmax_f32_scalar_strided(input, output, axis_dim_size, stride, num_rows);
        return;
    }
    // Contiguous softmax: each row is axis_dim_size elements
    for r in 0..num_rows {
        let base = r * axis_dim_size;
        // 1. Vectorized max
        let mut i = base;
        let end = base + axis_dim_size;
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        while i + 8 <= end {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(input.as_ptr().add(i)));
            i += 8;
        }
        // horizontal max
        let mx128 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        let mx = _mm_max_ps(mx128, _mm_shuffle_ps(mx128, mx128, 0x0E));
        let mx = _mm_max_ps(mx, _mm_shuffle_ps(mx, mx, 0x01));
        let mut max_val = _mm_cvtss_f32(mx);
        for j in i..end {
            if input[j] > max_val {
                max_val = input[j];
            }
        }
        let vmax_bcast = _mm256_set1_ps(max_val);
        // 2. Vectorized exp + sum using scalar f64 accumulator for precision
        let mut sum = 0.0f64;
        i = base;
        while i + 8 <= end {
            let vx = _mm256_loadu_ps(input.as_ptr().add(i));
            let vshifted = _mm256_sub_ps(vx, vmax_bcast);
            // Vectorized exp using existing polynomial approximation
            let vexp = exp_avx2_vec(vshifted);
            let arr: [f32; 8] = std::mem::transmute(vexp);
            for k in 0..8 {
                sum += arr[k] as f64;
            }
            _mm256_storeu_ps(output.as_mut_ptr().add(i), vexp);
            i += 8;
        }
        for j in i..end {
            let e = (input[j] - max_val).exp();
            *output.as_mut_ptr().add(j) = e;
            sum += e as f64;
        }
        let sum_f32 = sum as f32;
        if sum_f32 > 0.0 {
            let vsum = _mm256_set1_ps(sum_f32);
            i = base;
            while i + 8 <= end {
                let vexp = _mm256_loadu_ps(output.as_ptr().add(i));
                _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_div_ps(vexp, vsum));
                i += 8;
            }
            for j in i..end {
                *output.as_mut_ptr().add(j) /= sum_f32;
            }
        }
    }
}

// ── Optimizer: sgd_update_f32 ───────────────────────────────

#[inline]
pub fn sgd_update_f32_scalar(w: &[f32], g: &[f32], lr: f32, wd: f32) -> Vec<f32> {
    let mut result = vec![0.0; w.len()];
    sgd_update_f32_scalar_into(w, g, lr, wd, &mut result);
    result
}

#[inline]
pub fn sgd_update_f32_scalar_into(w: &[f32], g: &[f32], lr: f32, wd: f32, out: &mut [f32]) {
    let len = w.len().min(out.len());
    for i in 0..len {
        // w - lr * (g + wd * w)  =  w - lr*g - lr*wd*w
        out[i] = w[i] - lr * g.get(i % g.len()).copied().unwrap_or(0.0);
        out[i] -= lr * wd * w[i];
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `w` and `g` are valid, non-overlapping slices
// of equal length (or g is broadcast-able).
pub unsafe fn sgd_update_f32_avx2(w: &[f32], g: &[f32], lr: f32, wd: f32) -> Vec<f32> {
    let mut result = vec![0.0; w.len()];
    sgd_update_f32_avx2_into(w, g, lr, wd, &mut result);
    result
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as sgd_update_f32_avx2 — caller ensures `w`, `g`, and `out` are
// valid, with `out` at least as long as `w`.
pub unsafe fn sgd_update_f32_avx2_into(w: &[f32], g: &[f32], lr: f32, wd: f32, out: &mut [f32]) {
    let len = w.len().min(out.len());
    let mut i = 0;
    let vlr = _mm256_set1_ps(lr);
    let vwd = _mm256_set1_ps(wd);
    if g.len() == len {
        while i + 8 <= len {
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vg = _mm256_loadu_ps(g.as_ptr().add(i));
            // w - lr * (g + wd * w)
            // = w - lr*g - lr*wd*w
            // = vw - vlr * (vg + vwd * vw)
            let vg_wd = _mm256_fmadd_ps(vwd, vw, vg); // vwd * vw + vg
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_fnmadd_ps(vg_wd, vlr, vw));
            i += 8;
        }
    }
    for j in i..len {
        out[j] = w[j] - lr * g.get(j % g.len()).copied().unwrap_or(0.0);
        out[j] -= lr * wd * w[j];
    }
}

// ── Optimizer: adam_update_f32 ──────────────────────────────

#[inline]
pub fn adam_update_f32_scalar(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    let mut v_new = v.to_vec();
    adam_update_f32_scalar_into(
        w, g, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2, &mut w_new, &mut m_new,
        &mut v_new,
    );
    (w_new, m_new, v_new)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn adam_update_f32_scalar_into(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
    w_out: &mut [f32],
    m_out: &mut [f32],
    v_out: &mut [f32],
) {
    let len = w.len().min(w_out.len()).min(m_out.len()).min(v_out.len());
    for i in 0..len {
        let gi = g.get(i % g.len()).copied().unwrap_or(0.0);
        m_out[i] = beta1 * m[i] + (1.0 - beta1) * gi;
        v_out[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi;
        let m_hat = m_out[i] / bias_corr1;
        let v_hat = v_out[i] / bias_corr2;
        w_out[i] = w[i] - lr * m_hat / (v_hat.sqrt() + eps);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `w`, `g`, `m`, `v` are valid, non-overlapping,
// and of equal length; `bias_corr1/2` are pre-computed correction factors.
pub unsafe fn adam_update_f32_avx2(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    let mut v_new = v.to_vec();
    adam_update_f32_avx2_into(
        w, g, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2, &mut w_new, &mut m_new,
        &mut v_new,
    );
    (w_new, m_new, v_new)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
// SAFETY: Caller ensures `w`, `g`, `m`, `v`, `w_out`, `m_out`, `v_out` are all
// valid. `w`/`w_out` may alias; the function reads `w` before writing `w_out`.
pub unsafe fn adam_update_f32_avx2_into(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
    w_out: &mut [f32],
    m_out: &mut [f32],
    v_out: &mut [f32],
) {
    let len = w.len().min(w_out.len()).min(m_out.len()).min(v_out.len());
    let vlr = _mm256_set1_ps(lr);
    let vb1 = _mm256_set1_ps(beta1);
    let vb1c = _mm256_set1_ps(1.0 - beta1);
    let vb2 = _mm256_set1_ps(beta2);
    let vb2c = _mm256_set1_ps(1.0 - beta2);
    let veps = _mm256_set1_ps(eps);
    let vbc1 = _mm256_set1_ps(bias_corr1);
    let vbc2 = _mm256_set1_ps(bias_corr2);
    let mut i = 0;
    if g.len() == len {
        while i + 8 <= len {
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vg = _mm256_loadu_ps(g.as_ptr().add(i));
            let vm = _mm256_loadu_ps(m.as_ptr().add(i));
            let vv = _mm256_loadu_ps(v.as_ptr().add(i));
            let vm_new = _mm256_fmadd_ps(vb1c, vg, _mm256_mul_ps(vb1, vm));
            let vv_new = _mm256_fmadd_ps(vb2c, _mm256_mul_ps(vg, vg), _mm256_mul_ps(vb2, vv));
            let vm_hat = _mm256_div_ps(vm_new, vbc1);
            let vv_hat = _mm256_div_ps(vv_new, vbc2);
            let vdenom = _mm256_add_ps(_mm256_sqrt_ps(vv_hat), veps);
            let vupdate = _mm256_div_ps(_mm256_mul_ps(vlr, vm_hat), vdenom);
            _mm256_storeu_ps(w_out.as_mut_ptr().add(i), _mm256_sub_ps(vw, vupdate));
            _mm256_storeu_ps(m_out.as_mut_ptr().add(i), vm_new);
            _mm256_storeu_ps(v_out.as_mut_ptr().add(i), vv_new);
            i += 8;
        }
    }
    for j in i..len {
        let gi = g.get(j % g.len()).copied().unwrap_or(0.0);
        m_out[j] = beta1 * m[j] + (1.0 - beta1) * gi;
        v_out[j] = beta2 * v[j] + (1.0 - beta2) * gi * gi;
        let m_hat = m_out[j] / bias_corr1;
        let v_hat = v_out[j] / bias_corr2;
        w_out[j] = w[j] - lr * m_hat / (v_hat.sqrt() + eps);
    }
}

// ── Optimizer: adamw_update_f32 ─────────────────────────────

#[inline]
pub fn adamw_update_f32_scalar(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
    wd: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut w_new = vec![0.0; w.len()];
    let mut m_new = vec![0.0; m.len()];
    let mut v_new = vec![0.0; v.len()];
    adamw_update_f32_scalar_into(
        w, g, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2, wd, &mut w_new, &mut m_new,
        &mut v_new,
    );
    (w_new, m_new, v_new)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn adamw_update_f32_scalar_into(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
    wd: f32,
    w_out: &mut [f32],
    m_out: &mut [f32],
    v_out: &mut [f32],
) {
    let len = w.len().min(w_out.len()).min(m_out.len()).min(v_out.len());
    for i in 0..len {
        w_out[i] = w[i] - lr * wd * w[i];
        let gi = g.get(i % g.len()).copied().unwrap_or(0.0);
        m_out[i] = beta1 * m[i] + (1.0 - beta1) * gi;
        v_out[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi;
        let m_hat = m_out[i] / bias_corr1;
        let v_hat = v_out[i] / bias_corr2;
        w_out[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as adam_update_f32_avx2 — caller ensures all slices are valid
// and non-overlapping with equal length.
pub unsafe fn adamw_update_f32_avx2(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
    wd: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut w_new = vec![0.0; w.len()];
    let mut m_new = vec![0.0; m.len()];
    let mut v_new = vec![0.0; v.len()];
    adamw_update_f32_avx2_into(
        w, g, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2, wd, &mut w_new, &mut m_new,
        &mut v_new,
    );
    (w_new, m_new, v_new)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
// SAFETY: Same as adam_update_f32_avx2 — caller ensures `w`, `g`, `m`, `v`,
// `w_out`, `m_out`, `v_out` are all valid and non-overlapping.
pub unsafe fn adamw_update_f32_avx2_into(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
    wd: f32,
    w_out: &mut [f32],
    m_out: &mut [f32],
    v_out: &mut [f32],
) {
    let len = w.len().min(w_out.len()).min(m_out.len()).min(v_out.len());
    let vlr = _mm256_set1_ps(lr);
    let vwd = _mm256_set1_ps(wd);
    let vb1 = _mm256_set1_ps(beta1);
    let vb1c = _mm256_set1_ps(1.0 - beta1);
    let vb2 = _mm256_set1_ps(beta2);
    let vb2c = _mm256_set1_ps(1.0 - beta2);
    let veps = _mm256_set1_ps(eps);
    let vbc1 = _mm256_set1_ps(bias_corr1);
    let vbc2 = _mm256_set1_ps(bias_corr2);
    let mut i = 0;
    if g.len() == len {
        while i + 8 <= len {
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vg = _mm256_loadu_ps(g.as_ptr().add(i));
            let vm = _mm256_loadu_ps(m.as_ptr().add(i));
            let vv = _mm256_loadu_ps(v.as_ptr().add(i));
            let vw_decayed = _mm256_fnmadd_ps(_mm256_mul_ps(vlr, vwd), vw, vw);
            let vm_new = _mm256_fmadd_ps(vb1c, vg, _mm256_mul_ps(vb1, vm));
            let vv_new = _mm256_fmadd_ps(vb2c, _mm256_mul_ps(vg, vg), _mm256_mul_ps(vb2, vv));
            let vm_hat = _mm256_div_ps(vm_new, vbc1);
            let vv_hat = _mm256_div_ps(vv_new, vbc2);
            let vdenom = _mm256_add_ps(_mm256_sqrt_ps(vv_hat), veps);
            let vupdate = _mm256_div_ps(_mm256_mul_ps(vlr, vm_hat), vdenom);
            _mm256_storeu_ps(
                w_out.as_mut_ptr().add(i),
                _mm256_sub_ps(vw_decayed, vupdate),
            );
            _mm256_storeu_ps(m_out.as_mut_ptr().add(i), vm_new);
            _mm256_storeu_ps(v_out.as_mut_ptr().add(i), vv_new);
            i += 8;
        }
    }
    for j in i..len {
        w_out[j] = w[j] - lr * wd * w[j];
        let gi = g.get(j % g.len()).copied().unwrap_or(0.0);
        m_out[j] = beta1 * m[j] + (1.0 - beta1) * gi;
        v_out[j] = beta2 * v[j] + (1.0 - beta2) * gi * gi;
        let m_hat = m_out[j] / bias_corr1;
        let v_hat = v_out[j] / bias_corr2;
        w_out[j] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

// ── Optimizer: lion_update_f32 ──────────────────────────────

#[inline]
pub fn lion_update_f32_scalar(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    wd: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut w_new = vec![0.0; w.len()];
    let mut m_new = vec![0.0; m.len()];
    lion_update_f32_scalar_into(w, g, m, lr, beta1, beta2, wd, &mut w_new, &mut m_new);
    (w_new, m_new)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn lion_update_f32_scalar_into(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    wd: f32,
    w_out: &mut [f32],
    m_out: &mut [f32],
) {
    let len = w.len().min(w_out.len()).min(m_out.len());
    for i in 0..len {
        let gi = g.get(i % g.len()).copied().unwrap_or(0.0);
        m_out[i] = beta2 * m[i] + (1.0 - beta2) * gi;
        let update = beta1 * m_out[i] + (1.0 - beta1) * gi;
        w_out[i] = w[i] - lr * (update.signum() + wd * w[i]);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `w`, `g`, `m` are valid, non-overlapping,
// and of equal length; results are returned in new Vecs.
pub unsafe fn lion_update_f32_avx2(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    wd: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut w_new = vec![0.0; w.len()];
    let mut m_new = vec![0.0; m.len()];
    lion_update_f32_avx2_into(w, g, m, lr, beta1, beta2, wd, &mut w_new, &mut m_new);
    (w_new, m_new)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
// SAFETY: Same as lion_update_f32_avx2 — caller ensures `w`, `g`, `m`,
// `w_out`, `m_out` are all valid and non-overlapping.
pub unsafe fn lion_update_f32_avx2_into(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    wd: f32,
    w_out: &mut [f32],
    m_out: &mut [f32],
) {
    let len = w.len().min(w_out.len()).min(m_out.len());
    let vlr = _mm256_set1_ps(lr);
    let vwd = _mm256_set1_ps(wd);
    let vb1 = _mm256_set1_ps(beta1);
    let vb2 = _mm256_set1_ps(beta2);
    let vb1c = _mm256_set1_ps(1.0 - beta1);
    let vb2c = _mm256_set1_ps(1.0 - beta2);
    let vone = _mm256_set1_ps(1.0);
    let vneg_one = _mm256_set1_ps(-1.0);
    let vzero = _mm256_setzero_ps();
    let mut i = 0;
    if g.len() == len {
        while i + 8 <= len {
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vg = _mm256_loadu_ps(g.as_ptr().add(i));
            let vm = _mm256_loadu_ps(m.as_ptr().add(i));
            let vm_new = _mm256_fmadd_ps(vb2c, vg, _mm256_mul_ps(vb2, vm));
            let vupdate = _mm256_fmadd_ps(vb1c, vg, _mm256_mul_ps(vb1, vm_new));
            let vpos_mask = _mm256_cmp_ps::<{ _CMP_GT_OQ }>(vupdate, vzero);
            let vneg_mask = _mm256_cmp_ps::<{ _CMP_LT_OQ }>(vupdate, vzero);
            let vsign = _mm256_or_ps(
                _mm256_and_ps(vpos_mask, vone),
                _mm256_and_ps(vneg_mask, vneg_one),
            );
            let vsign_wd = _mm256_fmadd_ps(vwd, vw, vsign);
            _mm256_storeu_ps(
                w_out.as_mut_ptr().add(i),
                _mm256_fnmadd_ps(vlr, vsign_wd, vw),
            );
            _mm256_storeu_ps(m_out.as_mut_ptr().add(i), vm_new);
            i += 8;
        }
    }
    for j in i..len {
        let gi = g.get(j % g.len()).copied().unwrap_or(0.0);
        m_out[j] = beta2 * m[j] + (1.0 - beta2) * gi;
        let update = beta1 * m_out[j] + (1.0 - beta1) * gi;
        w_out[j] = w[j] - lr * (update.signum() + wd * w[j]);
    }
}

// ── Optimizer: rmsprop_update_f32 ───────────────────────────

#[inline]
pub fn rmsprop_update_f32_scalar(
    w: &[f32],
    g: &[f32],
    v: &[f32],
    lr: f32,
    beta: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut w_new = vec![0.0; w.len()];
    let mut v_new = vec![0.0; v.len()];
    rmsprop_update_f32_scalar_into(w, g, v, lr, beta, eps, &mut w_new, &mut v_new);
    (w_new, v_new)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn rmsprop_update_f32_scalar_into(
    w: &[f32],
    g: &[f32],
    v: &[f32],
    lr: f32,
    beta: f32,
    eps: f32,
    w_out: &mut [f32],
    v_out: &mut [f32],
) {
    let len = w.len().min(w_out.len()).min(v_out.len());
    for i in 0..len {
        let gi = g.get(i % g.len()).copied().unwrap_or(0.0);
        v_out[i] = beta * v[i] + (1.0 - beta) * gi * gi;
        w_out[i] = w[i] - lr * gi / (v_out[i].sqrt() + eps);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `w`, `g`, `v` are valid, non-overlapping,
// and of equal length; results are returned in new Vecs.
pub unsafe fn rmsprop_update_f32_avx2(
    w: &[f32],
    g: &[f32],
    v: &[f32],
    lr: f32,
    beta: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut w_new = vec![0.0; w.len()];
    let mut v_new = vec![0.0; v.len()];
    rmsprop_update_f32_avx2_into(w, g, v, lr, beta, eps, &mut w_new, &mut v_new);
    (w_new, v_new)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
// SAFETY: Same as rmsprop_update_f32_avx2 — caller ensures `w`, `g`, `v`,
// `w_out`, `v_out` are all valid and non-overlapping.
pub unsafe fn rmsprop_update_f32_avx2_into(
    w: &[f32],
    g: &[f32],
    v: &[f32],
    lr: f32,
    beta: f32,
    eps: f32,
    w_out: &mut [f32],
    v_out: &mut [f32],
) {
    let len = w.len().min(w_out.len()).min(v_out.len());
    let vlr = _mm256_set1_ps(lr);
    let vb = _mm256_set1_ps(beta);
    let vbc = _mm256_set1_ps(1.0 - beta);
    let veps = _mm256_set1_ps(eps);
    let mut i = 0;
    if g.len() == len {
        while i + 8 <= len {
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vg = _mm256_loadu_ps(g.as_ptr().add(i));
            let vv = _mm256_loadu_ps(v.as_ptr().add(i));
            let vv_new = _mm256_fmadd_ps(vbc, _mm256_mul_ps(vg, vg), _mm256_mul_ps(vb, vv));
            let vdenom = _mm256_add_ps(_mm256_sqrt_ps(vv_new), veps);
            let vupdate = _mm256_div_ps(_mm256_mul_ps(vlr, vg), vdenom);
            _mm256_storeu_ps(w_out.as_mut_ptr().add(i), _mm256_sub_ps(vw, vupdate));
            _mm256_storeu_ps(v_out.as_mut_ptr().add(i), vv_new);
            i += 8;
        }
    }
    for j in i..len {
        let gi = g.get(j % g.len()).copied().unwrap_or(0.0);
        v_out[j] = beta * v[j] + (1.0 - beta) * gi * gi;
        w_out[j] = w[j] - lr * gi / (v_out[j].sqrt() + eps);
    }
}

// ── Optimizer: muon_update_f32 ──────────────────────────────

#[inline]
pub fn muon_update_f32_scalar(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    lr: f32,
    beta: f32,
    wd: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = w.len();
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    for i in 0..len {
        let gi = g.get(i % g.len()).copied().unwrap_or(0.0);
        m_new[i] = beta * m[i] + (1.0 - beta) * gi;
        w_new[i] = w[i] - lr * (m_new[i] + wd * w[i]);
    }
    (w_new, m_new)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `w`, `g`, `m` are valid, non-overlapping,
// and of equal length; results are returned in new Vecs.
pub unsafe fn muon_update_f32_avx2(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    lr: f32,
    beta: f32,
    wd: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = w.len();
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    let vlr = _mm256_set1_ps(lr);
    let vb = _mm256_set1_ps(beta);
    let vbc = _mm256_set1_ps(1.0 - beta);
    let vwd = _mm256_set1_ps(wd);
    let mut i = 0;
    if g.len() == len {
        while i + 8 <= len {
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vg = _mm256_loadu_ps(g.as_ptr().add(i));
            let vm = _mm256_loadu_ps(m.as_ptr().add(i));
            let vm_new = _mm256_fmadd_ps(vbc, vg, _mm256_mul_ps(vb, vm));
            let vwd_term = _mm256_mul_ps(vwd, vw);
            let vinner = _mm256_add_ps(vm_new, vwd_term);
            let vw_new = _mm256_fnmadd_ps(vlr, vinner, vw);
            _mm256_storeu_ps(w_new.as_mut_ptr().add(i), vw_new);
            _mm256_storeu_ps(m_new.as_mut_ptr().add(i), vm_new);
            i += 8;
        }
    }
    for j in i..len {
        let gi = g.get(j % g.len()).copied().unwrap_or(0.0);
        m_new[j] = beta * m[j] + (1.0 - beta) * gi;
        w_new[j] = w[j] - lr * (m_new[j] + wd * w[j]);
    }
    (w_new, m_new)
}
