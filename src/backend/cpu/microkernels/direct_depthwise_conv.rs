//! Direct depthwise convolution — no im2col expansion.
//!
//! For depthwise 3×3 conv (groups == in_channels == out_channels), im2col
//! expands the input by 9×, thrashing L1/L2 cache and writing a patch matrix
//! that is immediately re-read. The channel is the only axis with no
//! reduction across it, so this kernel splits work over `(image, channel)`
//! planes: every plane owns one contiguous output slice and reads only its
//! own input plane. That makes the split disjoint — no atomics, no locks, no
//! false sharing — which is what lets the kernel use every core.
//!
//! Supported: 3×3 stride-1 pad-1 (AVX2), generic (scalar fallback).

#![allow(dead_code)]

use super::ConvActivation;
use crate::backend::cpu::microkernels::conv::apply_conv_activation;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use core::arch::x86_64::*;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::backend::cpu::microkernels::activations::{exp_avx2_vec, tanh_avx2_vec};

#[cfg(feature = "debug_canary")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "debug_canary")]
static DID_AVX2: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "debug_canary")]
static DID_SCALAR: AtomicBool = AtomicBool::new(false);

/// Output below this many elements stays on one thread. A plane split hands
/// each worker a few dozen planes, so the dispatch only pays for itself once
/// the output is a few hundred kilobytes.
const PARALLEL_MIN_OUTPUT_ELEMENTS: usize = 1 << 16;

// ── Helper: apply activation to an AVX2 vector ──────────────────

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_activation_vec(v: __m256, act: ConvActivation) -> __m256 {
    match act {
        ConvActivation::Relu => _mm256_max_ps(v, _mm256_setzero_ps()),
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
            let sqrt_2pi = _mm256_set1_ps(0.797_884_6_f32);
            let coeff = _mm256_set1_ps(0.044715f32);
            let v3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
            let tanh_arg = _mm256_mul_ps(sqrt_2pi, _mm256_add_ps(v, _mm256_mul_ps(coeff, v3)));
            let t = tanh_avx2_vec(tanh_arg);
            _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, t))
        }
    }
}

// ── Scalar: one depthwise pixel, generic stride/padding ─────────

/// One output pixel, reading only taps that fall inside the input plane.
///
/// Used for the whole tensor on the generic path and for the two edge columns
/// of each row on the AVX2 path, where a tap necessarily leaves the plane.
#[allow(clippy::too_many_arguments)]
fn depthwise_pixel_generic(
    input: &[f32],
    weights9: &[f32],
    out_plane: &mut [f32],
    inp_plane: usize,
    oh: usize,
    ow: usize,
    h: usize,
    w: usize,
    w_out: usize,
    stride: usize,
    padding: usize,
    bv: f32,
    activation: Option<ConvActivation>,
) {
    let mut sum = bv;
    for kr in 0..3isize {
        let ih = oh as isize * stride as isize - padding as isize + kr;
        if ih < 0 || ih >= h as isize {
            continue;
        }
        let row_base = inp_plane + ih as usize * w;
        for kw in 0..3isize {
            let iw = ow as isize * stride as isize - padding as isize + kw;
            if iw < 0 || iw >= w as isize {
                continue;
            }
            sum += input[row_base + iw as usize] * weights9[(kr as usize) * 3 + kw as usize];
        }
    }
    out_plane[oh * w_out + ow] = match activation {
        Some(act) => apply_conv_activation(sum, act),
        None => sum,
    };
}

// ── AVX2: interior of one stride-1 pad-1 row ────────────────────

/// Accumulate 8 output columns at a time for `ow` in `1..=w_out-2`.
///
/// Only the interior is vectorised: at `ow == 0` and `ow == w_out-1` a tap
/// leaves the input plane, so those two columns stay with the caller. The
/// final block is rewound to start at `w_out - 9` and recomputes a few
/// columns rather than dropping into a scalar tail, which used to leave the
/// rightmost seven columns of every row (6 % of a 112-wide plane) on the
/// per-pixel path.
///
/// # Safety
/// Caller must guarantee AVX2+FMA. `top`/`bottom` rows and `mid` must be
/// complete `w`-wide rows inside `input`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn depthwise_row_interior_avx2(
    input: &[f32],
    out_row: &mut [f32],
    top: Option<usize>,
    mid: usize,
    bottom: Option<usize>,
    weights: &[__m256; 9],
    vbias: __m256,
    activation: Option<ConvActivation>,
) {
    let w_out = out_row.len();
    if w_out < 10 {
        return;
    }
    let inp = input.as_ptr();
    let out = out_row.as_mut_ptr();
    let interior = w_out - 2;
    let mut done = 0usize;
    while done < interior {
        // Blocks start at ow = 1 + 8k; the last one is clamped so its eighth
        // lane still lands on w_out - 2 and every tap load stays in the row.
        let candidate = 1 + done;
        let start = if candidate + 8 < w_out {
            candidate
        } else {
            w_out - 9
        };

        let mut acc = vbias;
        if let Some(row) = top {
            let p = inp.add(row + start - 1);
            acc = _mm256_fmadd_ps(weights[0], _mm256_loadu_ps(p), acc);
            acc = _mm256_fmadd_ps(weights[1], _mm256_loadu_ps(p.add(1)), acc);
            acc = _mm256_fmadd_ps(weights[2], _mm256_loadu_ps(p.add(2)), acc);
        }
        {
            let p = inp.add(mid + start - 1);
            acc = _mm256_fmadd_ps(weights[3], _mm256_loadu_ps(p), acc);
            acc = _mm256_fmadd_ps(weights[4], _mm256_loadu_ps(p.add(1)), acc);
            acc = _mm256_fmadd_ps(weights[5], _mm256_loadu_ps(p.add(2)), acc);
        }
        if let Some(row) = bottom {
            let p = inp.add(row + start - 1);
            acc = _mm256_fmadd_ps(weights[6], _mm256_loadu_ps(p), acc);
            acc = _mm256_fmadd_ps(weights[7], _mm256_loadu_ps(p.add(1)), acc);
            acc = _mm256_fmadd_ps(weights[8], _mm256_loadu_ps(p.add(2)), acc);
        }

        let sum = match activation {
            Some(act) => apply_activation_vec(acc, act),
            None => acc,
        };
        _mm256_storeu_ps(out.add(start), sum);

        done += 8;
    }
}

// ── AVX2: one (image, channel) plane ────────────────────────────

/// Whole output plane of one channel, stride 1 and padding 1.
///
/// # Safety
/// Caller must guarantee AVX2+FMA and that `inp_plane .. inp_plane + h*w` and
/// `out_plane` cover the plane being written.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn depthwise_plane_avx2(
    input: &[f32],
    out_plane: &mut [f32],
    weights9: &[f32],
    bv: f32,
    inp_plane: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    activation: Option<ConvActivation>,
) {
    let vbias = _mm256_set1_ps(bv);
    let weights = [
        _mm256_set1_ps(*weights9.get_unchecked(0)),
        _mm256_set1_ps(*weights9.get_unchecked(1)),
        _mm256_set1_ps(*weights9.get_unchecked(2)),
        _mm256_set1_ps(*weights9.get_unchecked(3)),
        _mm256_set1_ps(*weights9.get_unchecked(4)),
        _mm256_set1_ps(*weights9.get_unchecked(5)),
        _mm256_set1_ps(*weights9.get_unchecked(6)),
        _mm256_set1_ps(*weights9.get_unchecked(7)),
        _mm256_set1_ps(*weights9.get_unchecked(8)),
    ];

    for oh in 0..h_out {
        let top = if oh > 0 {
            Some(inp_plane + (oh - 1) * w)
        } else {
            None
        };
        let mid = inp_plane + oh * w;
        let bottom = if oh + 1 < h {
            Some(inp_plane + (oh + 1) * w)
        } else {
            None
        };
        let (row_start, row_end) = (oh * w_out, (oh + 1) * w_out);

        if w_out >= 10 {
            let out_row = &mut out_plane[row_start..row_end];
            depthwise_row_interior_avx2(
                input, out_row, top, mid, bottom, &weights, vbias, activation,
            );
            depthwise_pixel_generic(
                input, weights9, out_plane, inp_plane, oh, 0, h, w, w_out, 1, 1, bv, activation,
            );
            depthwise_pixel_generic(
                input,
                weights9,
                out_plane,
                inp_plane,
                oh,
                w_out - 1,
                h,
                w,
                w_out,
                1,
                1,
                bv,
                activation,
            );
        } else {
            for ow in 0..w_out {
                depthwise_pixel_generic(
                    input, weights9, out_plane, inp_plane, oh, ow, h, w, w_out, 1, 1, bv,
                    activation,
                );
            }
        }
    }
}

// ── Plane runner: serial or split across (image, channel) ───────

/// Hand every `(image, channel)` plane to exactly one worker.
#[cfg_attr(feature = "parallel", allow(clippy::type_complexity))]
fn for_each_plane<F>(output: &mut [f32], plane_len: usize, work: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let planes = if plane_len == 0 {
        0
    } else {
        output.len() / plane_len
    };
    #[cfg(feature = "parallel")]
    if planes > 1 && output.len() >= PARALLEL_MIN_OUTPUT_ELEMENTS {
        use rayon::prelude::*;
        output
            .par_chunks_mut(plane_len)
            .enumerate()
            .for_each(|(plane, out_plane)| work(plane, out_plane));
        return;
    }
    for plane in 0..planes {
        let start = plane * plane_len;
        work(plane, &mut output[start..start + plane_len]);
    }
}

// ── Top-level depthwise 3×3 conv ────────────────────────────────

/// 3×3 depthwise convolution with fused bias+activation.
///
/// Only stride-1 pad-1 uses AVX2; other configurations fall back to scalar.
/// Both paths split over `(image, channel)` planes when the output is large
/// enough, because that is the one axis with no reduction across it.
#[allow(clippy::too_many_arguments)]
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
    assert!(
        groups == 0 || (groups == c && groups == f),
        "direct depthwise conv needs one channel per group"
    );
    let h_out = (h + 2 * padding)
        .checked_sub(3)
        .map_or(0, |span| span / stride + 1);
    let w_out = (w + 2 * padding)
        .checked_sub(3)
        .map_or(0, |span| span / stride + 1);
    if h_out == 0 || w_out == 0 {
        output.fill(0.0);
        return;
    }
    let plane_len = h_out * w_out;
    let total = n * f * plane_len;
    assert!(
        output.len() >= total,
        "depthwise conv output too small: {} < {total}",
        output.len()
    );
    assert!(
        weight.len() >= f * 9,
        "depthwise conv needs 9 weights per channel, got {} for f={f}",
        weight.len()
    );
    let hw = h * w;

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let avx2 = padding == 1 && stride == 1 && is_x86_feature_detected!("avx2");
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    let avx2 = false;

    #[cfg(feature = "debug_canary")]
    if avx2 {
        if !DID_AVX2.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[direct_depthwise_conv] AVX2 PATH ACTIVE (c={} h={} w={} f={} groups={})",
                c, h, w, f, groups
            );
        }
    } else if !DID_SCALAR.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[direct_depthwise_conv] SCALAR PATH (padding={} stride={} c={} h={} w={} f={})",
            padding, stride, c, h, w, f
        );
    }

    let work = |plane: usize, out_plane: &mut [f32]| {
        let img = plane / f;
        let g = plane % f;
        let inp_plane = img * c * hw + g * hw;
        let bv = if bias.is_empty() { 0.0 } else { bias[g] };
        let weights9 = &weight[g * 9..g * 9 + 9];

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if avx2 {
            // SAFETY: `avx2` was detected above, the input plane lies inside
            // `input`, and `out_plane` is this plane's own output slice.
            unsafe {
                depthwise_plane_avx2(
                    input, out_plane, weights9, bv, inp_plane, h, w, h_out, w_out, activation,
                );
            }
            return;
        }

        for oh in 0..h_out {
            for ow in 0..w_out {
                depthwise_pixel_generic(
                    input, weights9, out_plane, inp_plane, oh, ow, h, w, w_out, stride, padding,
                    bv, activation,
                );
            }
        }
    };

    for_each_plane(&mut output[..total], plane_len, work);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plane split, the rewind block and the scalar edge columns all have
    /// to agree with a straight per-pixel convolution, including widths that
    /// are not a multiple of 8 and planes smaller than one vector.
    #[test]
    fn depthwise_3x3_matches_a_naive_convolution_across_widths() {
        for &(n, c, h, w, act) in &[
            (1usize, 1usize, 5usize, 5usize, None),
            (1, 3, 7, 9, None),
            (1, 4, 12, 11, Some(ConvActivation::Relu)),
            (2, 5, 9, 16, None),
            (1, 8, 10, 10, Some(ConvActivation::Silu)),
            (1, 2, 3, 17, None),
        ] {
            let input: Vec<f32> = (0..n * c * h * w)
                .map(|i| ((i as f32) * 0.31).sin())
                .collect();
            let weight: Vec<f32> = (0..c * 9).map(|i| 0.05 + (i as f32) * 0.011).collect();
            let bias: Vec<f32> = (0..c).map(|i| 0.1 * i as f32).collect();
            let mut output = vec![0.0f32; n * c * h * w];
            direct_depthwise_conv3x3_f32(
                &input,
                &weight,
                &bias,
                &mut output,
                n,
                c,
                h,
                w,
                c,
                c,
                1,
                1,
                act,
            );

            for img in 0..n {
                for g in 0..c {
                    for oh in 0..h {
                        for ow in 0..w {
                            let mut expected = bias[g];
                            for kr in 0..3isize {
                                let ih = oh as isize + kr - 1;
                                if ih < 0 || ih >= h as isize {
                                    continue;
                                }
                                for kw in 0..3isize {
                                    let iw = ow as isize + kw - 1;
                                    if iw < 0 || iw >= w as isize {
                                        continue;
                                    }
                                    let src = (img * c + g) * h * w + ih as usize * w + iw as usize;
                                    expected += input[src]
                                        * weight[g * 9 + (kr as usize) * 3 + kw as usize];
                                }
                            }
                            let expected = match act {
                                Some(a) => apply_conv_activation(expected, a),
                                None => expected,
                            };
                            let actual = output[(img * c + g) * h * w + oh * w + ow];
                            assert!(
                                (actual - expected).abs() <= 1e-5,
                                "n={n} c={c} h={h} w={w} img={img} g={g} ({oh},{ow}): \
                                 expected {expected}, got {actual}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_kernel_larger_than_the_padded_input_clears_the_output() {
        let input = [1.0f32; 4];
        let weight = [1.0f32; 9];
        let mut output = [7.0f32];
        direct_depthwise_conv3x3_f32(
            &input,
            &weight,
            &[],
            &mut output,
            1,
            1,
            2,
            2,
            1,
            1,
            1,
            0,
            None,
        );
        assert_eq!(output, [0.0]);
    }
}
