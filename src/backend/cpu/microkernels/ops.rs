//! CPU ops microkernels (norm, pool, scalar) — extracted from microkernels.rs

#![allow(dead_code, unused_imports)]

use crate::dtypes::{F32x1, I4x8, I8x4, PackedWord};
use crate::packed_tensor::PackedTensor;
use std::sync::OnceLock;

use super::*;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

// BatchNorm inference — scalar + AVX2
// ============================================================

/// BatchNorm inference: `out[i] = data[i] * scale[ch] + shift[ch]` where
///   `scale[ch] = weight[ch] / sqrt(var[ch] + eps)`
///   `shift[ch] = bias[ch] - mean[ch] * scale[ch]`
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
    let weight_base = weight.as_ptr();
    let bias_base = bias.as_ptr();
    let mean_base = running_mean.as_ptr();
    let var_base = running_var.as_ptr();
    let vc = _mm256_set1_ps(c as f32);
    let veps = _mm256_set1_ps(eps);
    let vinc = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);

    let mut i = 0usize;
    while i + 8 <= len {
        // Channel indices ch_j = (i + j) % c  via float:  idx - floor(idx/c)*c
        let vi = _mm256_set1_ps(i as f32);
        let vindices_f = _mm256_add_ps(vi, vinc);
        let vdiv = _mm256_round_ps(_mm256_div_ps(vindices_f, vc), _MM_FROUND_TO_NEG_INF);
        let vch_f = _mm256_sub_ps(vindices_f, _mm256_mul_ps(vdiv, vc));
        let vch = _mm256_cvttps_epi32(vch_f);

        let vweight = _mm256_i32gather_ps(weight_base, vch, 4);
        let vbias = _mm256_i32gather_ps(bias_base, vch, 4);
        let vmean = _mm256_i32gather_ps(mean_base, vch, 4);
        let vvar = _mm256_i32gather_ps(var_base, vch, 4);
        let vscale = _mm256_div_ps(vweight, _mm256_sqrt_ps(_mm256_add_ps(vvar, veps)));
        let vshift = _mm256_sub_ps(vbias, _mm256_mul_ps(vmean, vscale));

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
        let scale = weight[ch] / (running_var[ch] + eps).sqrt();
        let shift = bias[ch] - running_mean[ch] * scale;
        output[j] = data[j] * scale + shift;
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
    debug_assert!(kernel > 0 && stride_val > 0);
    debug_assert_eq!(input.len(), n * c * h * w);
    debug_assert_eq!(output.len(), n * c * h_out * w_out);
    debug_assert!(indices_out
        .as_ref()
        .is_none_or(|indices| indices.len() == output.len()));
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
                                    if input[idx] > val {
                                        val = input[idx];
                                        best_kh = kh;
                                        best_kw = kw;
                                    }
                                }
                            }
                        }
                    }
                    let out_idx = nn * (c * hw_out) + cc * hw_out + hh * w_out + ww;
                    output[out_idx] = val;
                    if let Some(ref mut idx_out) = indices_out {
                        idx_out[out_idx] = (best_kh * kernel + best_kw) as i64;
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
    debug_assert!(kernel > 0 && stride_val > 0);
    debug_assert_eq!(input.len(), n * c * h * w);
    debug_assert_eq!(output.len(), n * c * h_out * w_out);
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
                                    val += input[idx];
                                    count += 1;
                                }
                            }
                        }
                    }
                    if count > 0 {
                        val /= count as f32;
                    }
                    let out_idx = nn * (c * hw_out) + cc * hw_out + hh * w_out + ww;
                    output[out_idx] = val;
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
    debug_assert!(h > 0 && w > 0 && out_h > 0 && out_w > 0);
    debug_assert_eq!(input.len(), nc * h * w);
    debug_assert_eq!(output.len(), nc * out_h * out_w);
    let hw = h * w;
    for nci in 0..nc {
        for ohi in 0..out_h {
            for owi in 0..out_w {
                let h_start = ohi * h / out_h;
                let h_end = ((ohi + 1) * h).div_ceil(out_h);
                let w_start = owi * w / out_w;
                let w_end = ((owi + 1) * w).div_ceil(out_w);
                let mut sum = 0.0f32;
                let mut count = 0;
                for hi in h_start..h_end {
                    for wi in w_start..w_end {
                        sum += input[nci * hw + hi * w + wi];
                        count += 1;
                    }
                }
                let out_idx = nci * out_h * out_w + ohi * out_w + owi;
                debug_assert!(count > 0);
                output[out_idx] = sum / count as f32;
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
