//! Optimized SWAR + AVX2 packed Conv2d kernels
//!
//! im2col + fused packed GEMM with:
//! - AVX2 int8 dot products (fallback to scalar SWAR)
//! - Cache tiling (TILE_K×TILE_N)
//! - Fused qa_sum/qb_sum into dot product loop
//! - Thread-local scratch buffers
//! - Rayon parallelism (row loop)
//! - im2col_dispatch (AVX2/parallel/scalar)
//! - Single-pass min/max scan
//! - i8 direct pack path (no FP32 buffer)
//! - AVX2 batch dequantization (per-tensor weights)

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::backend::cpu::swar::{
    sum_u4x8_packed, sum_u8x4_packed, u4x8_dot_packed, u8x4_dot_packed,
};
use crate::dtypes::{PackedWord, U4x8, U8x4};
use crate::packed_tensor::PackedTensor;

// Change 4: Thread-local scratch buffers for im2col+pack
thread_local! {
    static PACKED_CONV_COL_BUF: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
    static PACKED_CONV_I8_BUF: std::cell::RefCell<Vec<i8>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

/// Helper: reuse a thread-local Vec<f32>, return an `&mut [f32]` of at least `size`.
/// The buffer stays alive for `f`'s duration.
#[inline(always)]
fn with_col_buf<R>(size: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    PACKED_CONV_COL_BUF.with(|cell| {
        let mut b = cell.borrow_mut();
        if b.len() < size {
            b.resize(size, 0.0);
        }
        f(&mut b[..size])
    })
}

/// Helper: reuse a thread-local Vec<i8>, return an `&mut [i8]` of at least `size`.
#[inline(always)]
fn with_i8_buf<R>(size: usize, f: impl FnOnce(&mut [i8]) -> R) -> R {
    PACKED_CONV_I8_BUF.with(|cell| {
        let mut b = cell.borrow_mut();
        if b.len() < size {
            b.resize(size, 0i8);
        }
        f(&mut b[..size])
    })
}

#[inline(always)]
pub(crate) fn conv_out_size(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let dk = (kernel - 1) * dilation + 1;
    if input + 2 * padding >= dk {
        (input + 2 * padding - dk) / stride + 1
    } else {
        0
    }
}

/// im2col for i8 data (rearrange i8 values without FP32 conversion).
/// Used by the precopied i8 activation dispatch path in mod.rs.
#[inline(always)]
pub(crate) unsafe fn im2col_i8(
    input_n: &[i8],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    col: &mut [i8],
) {
    let dh = (kh - 1) * dilation + 1;
    let dw = (kw - 1) * dilation + 1;
    let h_out = if h + 2 * padding >= dh {
        (h + 2 * padding - dh) / stride + 1
    } else {
        0
    };
    let w_out = if w + 2 * padding >= dw {
        (w + 2 * padding - dw) / stride + 1
    } else {
        0
    };
    let col_w = c * kh * kw;

    for oh in 0..h_out {
        for ow in 0..w_out {
            let row = oh * w_out + ow;
            for ic in 0..c {
                for kkh in 0..kh {
                    for kkw in 0..kw {
                        let ih = oh * stride + kkh * dilation;
                        let iw = ow * stride + kkw * dilation;
                        let dst = row * col_w + ic * kh * kw + kkh * kw + kkw;
                        if ih < h + padding && iw < w + padding && ih >= padding && iw >= padding {
                            let src = (ic * h + (ih - padding)) * w + (iw - padding);
                            col[dst] = input_n[src];
                        } else {
                            col[dst] = 0i8;
                        }
                    }
                }
            }
        }
    }
}

// ── AVX2 dot-product helpers ─────────────────────────────────────
// Change 1: AVX2 int8 dot products

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x0E);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// AVX2 dot product over two U8x4-packed slices (2 words at a time → 8 int8 values).
/// Returns f64-precise sum (accumulated in f32 via FMA).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_u8x4_slice_avx2(a: &[U8x4], b: &[U8x4], k_packed: usize, k: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc_v = _mm256_setzero_ps();
    let mut p = 0usize;
    while p + 1 < k_packed && p * 4 + 8 <= k {
        let a0 = a[p].0;
        let a1 = a[p + 1].0;
        let b0 = b[p].0;
        let b1 = b[p + 1].0;
        let a128 = _mm_set_epi32(0, 0, a1 as i32, a0 as i32);
        let b128 = _mm_set_epi32(0, 0, b1 as i32, b0 as i32);
        let a_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a128));
        let b_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b128));
        acc_v = _mm256_fmadd_ps(a_f32, b_f32, acc_v);
        p += 2;
    }
    let mut acc = hsum256_ps(acc_v);
    while p < k_packed {
        let a_w = a[p].0;
        let b_w = b[p].0;
        let a0 = (a_w & 0xFF) as i8 as i32;
        let a1 = ((a_w >> 8) & 0xFF) as i8 as i32;
        let a2 = ((a_w >> 16) & 0xFF) as i8 as i32;
        let a3 = ((a_w >> 24) & 0xFF) as i8 as i32;
        let b0 = (b_w & 0xFF) as i8 as i32;
        let b1 = ((b_w >> 8) & 0xFF) as i8 as i32;
        let b2 = ((b_w >> 16) & 0xFF) as i8 as i32;
        let b3 = ((b_w >> 24) & 0xFF) as i8 as i32;
        acc += (a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3) as f32;
        p += 1;
    }
    acc
}

/// AVX2 dot product over two U4x8-packed slices (2 words at a time → 16 nibbles).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_u4x8_slice_avx2(a: &[U4x8], b: &[U4x8], k_packed: usize, k: usize) -> f32 {
    use std::arch::x86_64::*;
    let shift = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
    let mask_lo = _mm256_set1_epi32(0xF);
    let sign_ext = _mm256_set1_epi32(8);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut p = 0usize;
    while p + 1 < k_packed && p * 8 + 16 <= k {
        // Weight nibbles
        let w0 = b[p].0;
        let w1 = b[p + 1].0;
        let w0v = _mm256_set1_epi32(w0 as i32);
        let w1v = _mm256_set1_epi32(w1 as i32);
        let nb_w0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift), mask_lo);
        let nb_w1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift), mask_lo);
        let s_w0 = _mm256_sub_epi32(_mm256_xor_si256(nb_w0, sign_ext), sign_ext);
        let s_w1 = _mm256_sub_epi32(_mm256_xor_si256(nb_w1, sign_ext), sign_ext);
        let w_f0 = _mm256_cvtepi32_ps(s_w0);
        let w_f1 = _mm256_cvtepi32_ps(s_w1);

        // Activation nibbles
        let a0 = a[p].0;
        let a1 = a[p + 1].0;
        let a0v = _mm256_set1_epi32(a0 as i32);
        let a1v = _mm256_set1_epi32(a1 as i32);
        let nb_a0 = _mm256_and_si256(_mm256_srlv_epi32(a0v, shift), mask_lo);
        let nb_a1 = _mm256_and_si256(_mm256_srlv_epi32(a1v, shift), mask_lo);
        let s_a0 = _mm256_sub_epi32(_mm256_xor_si256(nb_a0, sign_ext), sign_ext);
        let s_a1 = _mm256_sub_epi32(_mm256_xor_si256(nb_a1, sign_ext), sign_ext);
        let a_f0 = _mm256_cvtepi32_ps(s_a0);
        let a_f1 = _mm256_cvtepi32_ps(s_a1);

        acc0 = _mm256_fmadd_ps(w_f0, a_f0, acc0);
        acc1 = _mm256_fmadd_ps(w_f1, a_f1, acc1);
        p += 2;
    }

    let mut acc = hsum256_ps(_mm256_add_ps(acc0, acc1));
    while p < k_packed {
        let a_w = a[p].0;
        let b_w = b[p].0;
        for lane in 0..8 {
            let a_nib = ((a_w >> (lane * 4)) & 0xF) as i32;
            let b_nib = ((b_w >> (lane * 4)) & 0xF) as i32;
            let av = if a_nib >= 8 { a_nib - 16 } else { a_nib };
            let bv = if b_nib >= 8 { b_nib - 16 } else { b_nib };
            acc += (av * bv) as f32;
        }
        p += 1;
    }
    acc
}

// ── im2col + pack (FP32 input) ──────────────────────────────────
// Changes 4, 8, 9: thread-local buf, im2col_dispatch, single-pass min/max

unsafe fn im2col_pack_u8x4(
    input_n: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> PackedTensor<U8x4> {
    let h_out = conv_out_size(h, kh, stride, padding, dilation);
    let w_out = conv_out_size(w, kw, stride, padding, dilation);
    let m = h_out * w_out;
    let k = c * kh * kw;
    let k_packed = k.div_ceil(4);
    let col_size = m * k;

    with_col_buf(col_size, |col| {
        // Change 8: Use im2col_dispatch (AVX2 → parallel → scalar)
        crate::backend::cpu::im2col::im2col_dispatch(
            input_n, c, h, w, kh, kw, stride, padding, dilation, col,
        );

        // Change 9: Single-pass min/max
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in col.iter() {
            min = min.min(v);
            max = max.max(v);
        }
        let range = max - min;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        let zero_point = if range > 0.0 {
            min + 128.0 * scale
        } else {
            0.0
        };
        let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        // Pack each row independently for correct row-wise alignment
        let mut packed: Vec<U8x4> = Vec::with_capacity(m * k_packed);
        for row in 0..m {
            let row_start = row * k;
            for word in 0..k_packed {
                let mut w = 0u32;
                let base = row_start + word * 4;
                for i in 0..4 {
                    let elem_idx = base + i;
                    if elem_idx < row_start + k {
                        let val = col[elem_idx];
                        let q = ((val - zero_point) * inv_scale)
                            .round()
                            .clamp(-128.0, 127.0) as i8;
                        w |= (q as u8 as u32) << (i * 8);
                    }
                }
                packed.push(U8x4(w));
            }
        }

        PackedTensor::from_raw(packed, vec![m, k], vec![scale], vec![zero_point])
    })
}

unsafe fn im2col_pack_u4x8(
    input_n: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> PackedTensor<U4x8> {
    let h_out = conv_out_size(h, kh, stride, padding, dilation);
    let w_out = conv_out_size(w, kw, stride, padding, dilation);
    let m = h_out * w_out;
    let k = c * kh * kw;
    let k_packed = k.div_ceil(8);
    let col_size = m * k;

    with_col_buf(col_size, |col| {
        crate::backend::cpu::im2col::im2col_dispatch(
            input_n, c, h, w, kh, kw, stride, padding, dilation, col,
        );

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in col.iter() {
            min = min.min(v);
            max = max.max(v);
        }
        let range = max - min;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let zero_point = if range > 0.0 { min + 8.0 * scale } else { 0.0 };
        let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        let mut packed: Vec<U4x8> = Vec::with_capacity(m * k_packed);
        for row in 0..m {
            let row_start = row * k;
            for word in 0..k_packed {
                let mut w = 0u32;
                let base = row_start + word * 8;
                for i in 0..8 {
                    let elem_idx = base + i;
                    if elem_idx < row_start + k {
                        let val = col[elem_idx];
                        let q = ((val - zero_point) * inv_scale).round().clamp(-8.0, 7.0) as i32;
                        w |= ((q as u32) & 0xF) << (i * 4);
                    }
                }
                packed.push(U4x8(w));
            }
        }

        PackedTensor::from_raw(packed, vec![m, k], vec![scale], vec![zero_point])
    })
}

// ── Change 10: im2col + pack from i8 (no FP32 buffer) ──────────

unsafe fn im2col_pack_u8x4_from_i8(
    input_n: &[i8],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> PackedTensor<U8x4> {
    let h_out = conv_out_size(h, kh, stride, padding, dilation);
    let w_out = conv_out_size(w, kw, stride, padding, dilation);
    let m = h_out * w_out;
    let k = c * kh * kw;
    let k_packed = k.div_ceil(4);
    let col_w = c * kh * kw;

    // Use thread-local i8 buffer for im2col rearrangement
    with_i8_buf(m * k, |col| {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row = oh * w_out + ow;
                for ic in 0..c {
                    for kkh in 0..kh {
                        for kkw in 0..kw {
                            let ih = oh * stride + kkh * dilation;
                            let iw = ow * stride + kkw * dilation;
                            let dst = row * col_w + ic * kh * kw + kkh * kw + kkw;
                            if ih < h + padding
                                && iw < w + padding
                                && ih >= padding
                                && iw >= padding
                            {
                                let src = (ic * h + (ih - padding)) * w + (iw - padding);
                                col[dst] = input_n[src];
                            } else {
                                col[dst] = 0i8;
                            }
                        }
                    }
                }
            }
        }

        // Compute min/max on i8 data for scale/zp determination
        let mut min_i8 = 127i8;
        let mut max_i8 = -128i8;
        for &v in col.iter() {
            min_i8 = min_i8.min(v);
            max_i8 = max_i8.max(v);
        }
        let min_f = min_i8 as f32;
        let max_f = max_i8 as f32;
        let range = max_f - min_f;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        let zero_point = if range > 0.0 {
            min_f + 128.0 * scale
        } else {
            0.0
        };

        // Pack i8 → U8x4
        let mut packed: Vec<U8x4> = Vec::with_capacity(m * k_packed);
        for row in 0..m {
            let row_start = row * k;
            for word in 0..k_packed {
                let mut w = 0u32;
                let base = row_start + word * 4;
                for i in 0..4 {
                    let elem_idx = base + i;
                    if elem_idx < row_start + k {
                        w |= (col[elem_idx] as u8 as u32) << (i * 8);
                    }
                }
                packed.push(U8x4(w));
            }
        }

        PackedTensor::from_raw(packed, vec![m, k], vec![scale], vec![zero_point])
    })
}

// ── Slice helper ────────────────────────────────────────────────

#[inline(always)]
unsafe fn slice_packed<U: PackedWord>(
    t: &PackedTensor<U>,
    row_start: usize,
    row_count: usize,
) -> PackedTensor<U> {
    let inner: usize = t.shape()[1..].iter().product();
    let k_packed = inner.div_ceil(U::ITEMS);
    let data = t.as_packed()[row_start * k_packed..(row_start + row_count) * k_packed].to_vec();
    let scales = if t.scales.len() > 1 {
        t.scales[row_start..row_start + row_count].to_vec()
    } else {
        t.scales.clone()
    };
    let zeros = if t.zeros.len() > 1 {
        t.zeros[row_start..row_start + row_count].to_vec()
    } else {
        t.zeros.clone()
    };
    PackedTensor::from_raw(data, vec![row_count, inner], scales, zeros)
}

// ── GEMM: U8x4 with fused dequantize + bias + activation ────────
// Changes 1, 2, 3, 7, 11

// (Tile constants moved to per-GEMM scope below)

/// Raw slice-based U8x4 packed GEMM with fused dequantize + bias + activation.
/// Avoids PackedTensor wrapping overhead — takes slices and scalars directly.
/// Activation scale/zp are per-tensor (scalar). Weight scales/zps can be
/// per-channel (len == n) or per-tensor (len == 1, broadcast).
pub fn gemm_packed_u8x4_fused_raw(
    act_data: &[U8x4],
    m: usize,
    k: usize,
    act_scale: f32,
    act_zp: f32,
    w_data: &[U8x4],
    n: usize,
    w_scales: &[f32],
    w_zps: &[f32],
    bias: Option<&[f32]>,
    activation: Option<&str>,
    c: &mut [f32],
) {
    let k_packed = k.div_ceil(4);
    debug_assert_eq!(c.len(), m * n);
    debug_assert!(w_scales.len() == 1 || w_scales.len() == n);
    debug_assert!(w_zps.len() == 1 || w_zps.len() == n);
    let k_f32 = k as f32;
    let per_channel_w = w_scales.len() > 1;

    // Precompute qb_sum once per weight column
    let mut qb_sum: smallvec::SmallVec<[i32; 256]> = smallvec::SmallVec::with_capacity(n);
    for col in 0..n {
        let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
        qb_sum.push(w_row.iter().map(|&w| sum_u8x4_packed(w.0)).sum::<i32>());
    }

    // Determine row iteration strategy
    #[cfg(feature = "parallel")]
    let parallel = m >= 512;

    // Row computation lambda
    let compute_row = |row: usize, c_row: &mut [f32]| {
        let act_row_start = row * k_packed;
        let act_row = &act_data[act_row_start..act_row_start + k_packed];

        let qa_sum: i32 = act_row.iter().map(|&w| sum_u8x4_packed(w.0)).sum();

        for col in 0..n {
            let w_row_start = col * k_packed;
            let w_row = &w_data[w_row_start..w_row_start + k_packed];

            let w_scale = if per_channel_w { w_scales[col] } else { w_scales[0] };
            let w_zp = if per_channel_w { w_zps[col] } else { w_zps[0] };

            let mut acc = 0i32;
            for kk in 0..k_packed {
                acc += u8x4_dot_packed(act_row[kk].0, w_row[kk].0);
            }

            let scale_ab = act_scale * w_scale;
            let mut val = (acc as f32) * scale_ab
                + w_zp * (act_scale * qa_sum as f32)
                + act_zp * (w_scale * qb_sum[col] as f32)
                + act_zp * w_zp * k_f32;

            if let Some(b) = bias {
                val += b[col];
            }
            if let Some(act) = activation {
                val = match act {
                    "relu" => val.max(0.0),
                    "silu" => val / (1.0 + (-val).exp()),
                    _ => val,
                };
            }
            c_row[col] = val;
        }
    };

    #[cfg(feature = "parallel")]
    if parallel {
        c.par_chunks_mut(n)
            .enumerate()
            .for_each(|(row, c_row)| compute_row(row, c_row));
    } else {
        for (row, c_row) in c.chunks_mut(n).enumerate() {
            compute_row(row, c_row);
        }
    }

    #[cfg(not(feature = "parallel"))]
    for (row, c_row) in c.chunks_mut(n).enumerate() {
        compute_row(row, c_row);
    }
}

/// PackedTensor-wrapping variant of `gemm_packed_u8x4_fused_raw`.
/// Extracts shape/scales/zps from PackedTensor and delegates to the raw version.
#[inline]
pub fn gemm_packed_u8x4_fused(
    act_packed: &PackedTensor<U8x4>,
    weight_packed: &PackedTensor<U8x4>,
    bias: Option<&[f32]>,
    activation: Option<&str>,
    c: &mut [f32],
) {
    let m = act_packed.shape()[0];
    let k = act_packed.shape()[1];
    let n = weight_packed.shape()[0];
    gemm_packed_u8x4_fused_raw(
        act_packed.as_packed(),
        m,
        k,
        act_packed.scale_for_row(0),
        act_packed.zero_for_row(0),
        weight_packed.as_packed(),
        n,
        &weight_packed.scales,
        &weight_packed.zeros,
        bias,
        activation,
        c,
    );
}

// ── GEMM: U4x8 with fused dequantize + bias + activation ────────

/// Raw slice-based U4x8 packed GEMM with fused dequantize + bias + activation.
pub fn gemm_packed_u4x8_fused_raw(
    act_data: &[U4x8],
    m: usize,
    k: usize,
    act_scale: f32,
    act_zp: f32,
    w_data: &[U4x8],
    n: usize,
    w_scales: &[f32],
    w_zps: &[f32],
    bias: Option<&[f32]>,
    activation: Option<&str>,
    c: &mut [f32],
) {
    let k_packed = k.div_ceil(8);
    debug_assert_eq!(c.len(), m * n);
    debug_assert!(w_scales.len() == 1 || w_scales.len() == n);
    debug_assert!(w_zps.len() == 1 || w_zps.len() == n);
    let k_f32 = k as f32;
    let per_channel_w = w_scales.len() > 1;

    // Precompute qb_sum per weight column
    let mut qb_sum: smallvec::SmallVec<[i32; 256]> = smallvec::SmallVec::with_capacity(n);
    for col in 0..n {
        let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
        qb_sum.push(w_row.iter().map(|&w| sum_u4x8_packed(w.0)).sum::<i32>());
    }

    let compute_row = |row: usize, c_row: &mut [f32]| {
        let a_row_start = row * k_packed;
        let a_row = &act_data[a_row_start..a_row_start + k_packed];

        let qa_sum: i32 = a_row.iter().map(|&w| sum_u4x8_packed(w.0)).sum();

        for col in 0..n {
            let w_row_start = col * k_packed;
            let w_row = &w_data[w_row_start..w_row_start + k_packed];

            let w_scale = if per_channel_w { w_scales[col] } else { w_scales[0] };
            let w_zp = if per_channel_w { w_zps[col] } else { w_zps[0] };

            let mut acc = 0i32;
            for kk in 0..k_packed {
                acc += u4x8_dot_packed(a_row[kk].0, w_row[kk].0);
            }

            let scale_ab = act_scale * w_scale;
            let mut val = (acc as f32) * scale_ab
                + w_zp * (act_scale * qa_sum as f32)
                + act_zp * (w_scale * qb_sum[col] as f32)
                + act_zp * w_zp * k_f32;

            if let Some(b) = bias {
                val += b[col];
            }
            if let Some(act) = activation {
                val = match act {
                    "relu" => val.max(0.0),
                    "silu" => val / (1.0 + (-val).exp()),
                    _ => val,
                };
            }
            c_row[col] = val;
        }
    };

    #[cfg(feature = "parallel")]
    if m >= 512 {
        c.par_chunks_mut(n)
            .enumerate()
            .for_each(|(row, c_row)| compute_row(row, c_row));
    } else {
        for (row, c_row) in c.chunks_mut(n).enumerate() {
            compute_row(row, c_row);
        }
    }

    #[cfg(not(feature = "parallel"))]
    for (row, c_row) in c.chunks_mut(n).enumerate() {
        compute_row(row, c_row);
    }
}

/// PackedTensor-wrapping variant. Delegates to `gemm_packed_u4x8_fused_raw`.
#[inline]
pub fn gemm_packed_u4x8_fused(
    act_packed: &PackedTensor<U4x8>,
    weight_packed: &PackedTensor<U4x8>,
    bias: Option<&[f32]>,
    activation: Option<&str>,
    c: &mut [f32],
) {
    let m = act_packed.shape()[0];
    let k = act_packed.shape()[1];
    let n = weight_packed.shape()[0];
    gemm_packed_u4x8_fused_raw(
        act_packed.as_packed(),
        m,
        k,
        act_packed.scale_for_row(0),
        act_packed.zero_for_row(0),
        weight_packed.as_packed(),
        n,
        &weight_packed.scales,
        &weight_packed.zeros,
        bias,
        activation,
        c,
    );
}

/// Pack flat i8 column-buffer data into U8x4 packed words.
///
/// Every 4 consecutive i8 bytes are packed into one u32 word:
/// `word[0] = bytes[0], word[1] = bytes[1] << 8, ...`
/// The i8→u8 conversion uses two's complement (safe cast).
/// The output `packed` must have at least `m * k_packed` capacity.
#[inline]
pub fn pack_i8_col_to_u8x4(col: &[i8], m: usize, k: usize, packed: &mut [U8x4]) {
    let k_packed = k.div_ceil(4);
    let full_words = if k % 4 == 0 {
        k_packed
    } else {
        k_packed - 1
    };
    for row in 0..m {
        let row_start = row * k;
        let word_base = row * k_packed;
        // Fast path: read full u32 words via unaligned load (avoids byte loop)
        for w in 0..full_words {
            let byte_base = row_start + w * 4;
            // SAFETY: aligned within the slice bounds, u32 has same repr as 4×i8
            let word = unsafe { std::ptr::read_unaligned(col.as_ptr().add(byte_base) as *const u32) };
            packed[word_base + w] = U8x4(word);
        }
        // Tail: handle the last partial word when k is not a multiple of 4
        if k % 4 != 0 {
            let mut word = 0u32;
            let byte_base = row_start + full_words * 4;
            for i in 0..(k % 4) {
                let idx = byte_base + i;
                word |= (col[idx] as u8 as u32) << (i * 8);
            }
            packed[word_base + full_words] = U8x4(word);
        }
    }
}

/// Pack flat i8 column-buffer data into U4x8 packed words (2 nibbles per byte).
///
/// Every 8 i8 values (each clamped to [-8, 7]) are packed into one u32 word
/// as 8× 4-bit nibbles. Out-of-range values are clamped.
#[inline]
pub fn pack_i8_col_to_u4x8(col: &[i8], m: usize, k: usize, packed: &mut [U4x8]) {
    let k_packed = k.div_ceil(8);
    for row in 0..m {
        let row_start = row * k;
        let word_base = row * k_packed;
        for w in 0..k_packed {
            let mut word = 0u32;
            let byte_base = row_start + w * 8;
            for i in 0..8 {
                let idx = byte_base + i;
                if idx < row_start + k {
                    let val = col[idx].clamp(-8i8, 7i8);
                    word |= ((val as u8 as u32) & 0xF) << (i * 4);
                }
            }
            packed[word_base + w] = U4x8(word);
        }
    }
}

// ── Public Conv2d entry points ──────────────────────────────────

pub unsafe fn conv2d_packed_u8x4(
    input: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    weight: &PackedTensor<U8x4>,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    kh: usize,
    kw: usize,
    activation: Option<&str>,
    output: &mut [f32],
) {
    let oc = weight.shape()[0];
    let c_per_g = c / groups;
    let oc_per_g = oc / groups;
    let h_out = conv_out_size(h, kh, stride, padding, dilation);
    let w_out = conv_out_size(w, kw, stride, padding, dilation);
    let num_pixels = h_out * w_out;

    for nn in 0..n {
        let input_base = nn * c * h * w;
        let out_base = nn * oc * h_out * w_out;

        for g in 0..groups {
            let g_c_off = g * c_per_g;
            let g_oc_off = g * oc_per_g;

            let act_packed = im2col_pack_u8x4(
                &input[input_base + g_c_off * h * w..],
                c_per_g,
                h,
                w,
                kh,
                kw,
                stride,
                padding,
                dilation,
            );

            let w_slice = if groups > 1 {
                slice_packed(weight, g_oc_off, oc_per_g)
            } else {
                slice_packed(weight, 0, oc)
            };

            let local_oc = w_slice.shape()[0];
            let mut temp = vec![0.0f32; num_pixels * local_oc];
            let b = bias.map(|b| {
                if groups > 1 {
                    &b[g_oc_off..g_oc_off + oc_per_g]
                } else {
                    b
                }
            });
            gemm_packed_u8x4_fused(&act_packed, &w_slice, b, activation, &mut temp);

            for pixel in 0..num_pixels {
                for f in 0..local_oc {
                    output[out_base + (g_oc_off + f) * num_pixels + pixel] =
                        temp[pixel * local_oc + f];
                }
            }
        }
    }
}

pub unsafe fn conv2d_packed_u4x8(
    input: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    weight: &PackedTensor<U4x8>,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    kh: usize,
    kw: usize,
    activation: Option<&str>,
    output: &mut [f32],
) {
    let oc = weight.shape()[0];
    let c_per_g = c / groups;
    let oc_per_g = oc / groups;
    let h_out = conv_out_size(h, kh, stride, padding, dilation);
    let w_out = conv_out_size(w, kw, stride, padding, dilation);
    let num_pixels = h_out * w_out;

    for nn in 0..n {
        let input_base = nn * c * h * w;
        let out_base = nn * oc * h_out * w_out;

        for g in 0..groups {
            let g_c_off = g * c_per_g;
            let g_oc_off = g * oc_per_g;

            let act_packed = im2col_pack_u4x8(
                &input[input_base + g_c_off * h * w..],
                c_per_g,
                h,
                w,
                kh,
                kw,
                stride,
                padding,
                dilation,
            );

            let w_slice = if groups > 1 {
                slice_packed(weight, g_oc_off, oc_per_g)
            } else {
                slice_packed(weight, 0, oc)
            };

            let local_oc = w_slice.shape()[0];
            let mut temp = vec![0.0f32; num_pixels * local_oc];
            let b = bias.map(|b| {
                if groups > 1 {
                    &b[g_oc_off..g_oc_off + oc_per_g]
                } else {
                    b
                }
            });
            gemm_packed_u4x8_fused(&act_packed, &w_slice, b, activation, &mut temp);

            for pixel in 0..num_pixels {
                for f in 0..local_oc {
                    output[out_base + (g_oc_off + f) * num_pixels + pixel] =
                        temp[pixel * local_oc + f];
                }
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::packed_gemm::quantize_activations_to_u4x8;

    #[test]
    fn test_conv2d_packed_u4x8_basic() {
        let n = 1;
        let c = 8;
        let h = 1;
        let w = 1;
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let wdata = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let weight = PackedTensor::<U4x8>::from_f32_per_channel(&wdata, &[2, 8]);
        let mut output = vec![0.0f32; 2];
        unsafe {
            conv2d_packed_u4x8(
                &input,
                n,
                c,
                h,
                w,
                &weight,
                None,
                1,
                0,
                1,
                1,
                1,
                1,
                None,
                &mut output,
            );
        }
        assert_eq!(output.len(), 2);
        assert!(
            output[0] > 0.0,
            "Expected positive output[0], got {}",
            output[0]
        );
    }

    #[test]
    fn test_gemm_packed_u4x8_fused_basic() {
        let input = vec![-4.0, -2.0, 0.0, 2.0, -3.0, -1.0, 1.0, 3.0];
        let act = quantize_activations_to_u4x8(&input);
        let weight_data = vec![
            4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let weight = PackedTensor::<U4x8>::from_f32_per_channel(&weight_data, &[2, 8]);
        let mut c = vec![0.0f32; 2];
        gemm_packed_u4x8_fused(&act, &weight, None, None, &mut c);
        assert_eq!(c.len(), 2);
        assert!(c[1] > 0.0, "expected second output positive, got {:?}", c);
    }

    #[test]
    fn test_conv2d_packed_u4x8_non_multiple_k() {
        let n = 1;
        let c = 3;
        let h = 4;
        let w = 4;
        let kh = 3;
        let kw = 3;
        let stride = 1;
        let padding = 1;
        let dilation = 1;
        let oc = 2;
        let groups = 1;

        let input: Vec<f32> = (0..(c * h * w)).map(|i| (i as f32 - 24.0) * 0.1).collect();
        let k = c * kh * kw;
        let mut wdata = vec![0.0f32; oc * k];
        for i in 0..c {
            wdata[i * kh * kw + kh * kw / 2] = 1.0;
        }
        for i in 0..c {
            wdata[k + i * kh * kw + kh * kw / 2] = 0.5;
        }

        let weight = PackedTensor::<U4x8>::from_f32_per_channel(&wdata, &[oc, k]);
        let h_out = h;
        let w_out = w;
        let mut output = vec![0.0f32; oc * h_out * w_out];
        unsafe {
            conv2d_packed_u4x8(
                &input,
                n,
                c,
                h,
                w,
                &weight,
                None,
                stride,
                padding,
                dilation,
                groups,
                kh,
                kw,
                None,
                &mut output,
            );
        }
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite value at output[{}]: {}", i, v);
        }
        let max_abs = output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.01, "Output is all zeros (max_abs={})", max_abs);
    }

    #[test]
    fn test_conv2d_packed_u8x4_non_multiple_k() {
        let n = 1;
        let c = 3;
        let h = 4;
        let w = 4;
        let kh = 3;
        let kw = 3;
        let stride = 1;
        let padding = 1;
        let dilation = 1;
        let oc = 2;
        let groups = 1;

        let input: Vec<f32> = (0..(c * h * w)).map(|i| (i as f32 - 24.0) * 0.1).collect();
        let k = c * kh * kw;
        let mut wdata = vec![0.0f32; oc * k];
        for i in 0..c {
            wdata[i * kh * kw + kh * kw / 2] = 1.0;
        }
        for i in 0..c {
            wdata[k + i * kh * kw + kh * kw / 2] = 0.5;
        }

        let weight = PackedTensor::<U8x4>::from_f32_per_channel(&wdata, &[oc, k]);
        let h_out = h;
        let w_out = w;
        let mut output = vec![0.0f32; oc * h_out * w_out];
        unsafe {
            conv2d_packed_u8x4(
                &input,
                n,
                c,
                h,
                w,
                &weight,
                None,
                stride,
                padding,
                dilation,
                groups,
                kh,
                kw,
                None,
                &mut output,
            );
        }
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite value at output[{}]: {}", i, v);
        }
        let max_abs = output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.01, "Output is all zeros (max_abs={})", max_abs);
    }
}
