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

use crate::backend::cpu::packed_gemm::gemm_packed_float_fused;
use crate::backend::cpu::swar::{
    i4x8_dot_packed, i8x4_dot_packed, sum_i4x8_packed, sum_i8x4_packed,
};
use crate::dtypes::f4x8::f4x8_dot_packed;
use crate::dtypes::{F4x8, I4x8, I8x4, PackedWord};
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
// SAFETY: Caller must ensure all pointer arguments are valid, non-overlapping,
// and point to allocations of sufficient size for the convolution parameters.
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

// ── im2col + pack (FP32 input) ──────────────────────────────────
// Changes 4, 8, 9: thread-local buf, im2col_dispatch, single-pass min/max

// SAFETY: Caller must ensure `input_n` is valid for the full NCHW input range
// and the returned PackedTensor is properly constructed.
unsafe fn im2col_pack_i8x4(
    input_n: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> PackedTensor<I8x4> {
    let h_out = conv_out_size(h, kh, stride, padding, dilation);
    let w_out = conv_out_size(w, kw, stride, padding, dilation);
    let m = h_out * w_out;
    let k = c * kh * kw;
    let k_packed = k.div_ceil(4);
    let col_size = m * k;

    with_col_buf(col_size, |col| {
        let (global_min, global_max) = crate::backend::cpu::im2col::im2col_dispatch(
            input_n, c, h, w, kh, kw, stride, padding, dilation, col,
            true, // I8x4 needs min/max for asym quantization
        );

        let range = global_max - global_min;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        let zero_point = if range > 0.0 {
            global_min + 128.0 * scale
        } else {
            0.0
        };
        let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        // Pack each row independently for correct row-wise alignment
        let mut packed: Vec<I8x4> = Vec::with_capacity(m * k_packed);
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            use std::arch::x86_64::*;
            let zv = _mm256_set1_ps(zero_point);
            let iv = _mm256_set1_ps(inv_scale);
            let c_lo = _mm256_set1_ps(-128.0);
            let c_hi = _mm256_set1_ps(127.0);
            let mut row_packed = [0u8; 32];
            for row in 0..m {
                let row_start = row * k;
                let mut wp = 0usize;
                while wp + 7 < k_packed {
                    let base = row_start + wp * 4;
                    let v = _mm256_loadu_ps(&col[base]);
                    let x = _mm256_round_ps(_mm256_mul_ps(_mm256_sub_ps(v, zv), iv), 0);
                    let clamped = _mm256_min_ps(_mm256_max_ps(x, c_lo), c_hi);
                    let i32v = _mm256_cvttps_epi32(clamped);
                    let lo128 = _mm256_castsi256_si128(i32v);
                    let hi128 = _mm256_extracti128_si256(i32v, 1);
                    let lo16 = _mm_packs_epi32(lo128, _mm_setzero_si128());
                    let hi16 = _mm_packs_epi32(hi128, _mm_setzero_si128());
                    let bytes = _mm_packus_epi16(lo16, hi16);
                    _mm_storeu_si128(row_packed[0..16].as_mut_ptr() as *mut __m128i, bytes);
                    for j in 0..8 {
                        packed[row * k_packed + wp + j] = I8x4(u32::from_le_bytes([
                            row_packed[4 * j],
                            row_packed[4 * j + 1],
                            row_packed[4 * j + 2],
                            row_packed[4 * j + 3],
                        ]));
                    }
                    wp += 8;
                }
                while wp < k_packed {
                    let base = row_start + wp * 4;
                    let mut w = 0u32;
                    for i in 0..4 {
                        let ei = base + i;
                        if ei < row_start + k {
                            let q = ((col[ei] - zero_point) * inv_scale)
                                .round()
                                .clamp(-128.0, 127.0) as i8;
                            w |= (q as u8 as u32) << (i * 8);
                        }
                    }
                    packed.push(I8x4(w));
                    wp += 1;
                }
            }
        }
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
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
                packed.push(I8x4(w));
            }
        }

        PackedTensor::from_raw(packed, vec![m, k], vec![scale], vec![zero_point])
    })
}

// SAFETY: Same as im2col_pack_i8x4 — caller must ensure `input_n` is valid
// for the full NCHW input range.
unsafe fn im2col_pack_i4x8(
    input_n: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> PackedTensor<I4x8> {
    let h_out = conv_out_size(h, kh, stride, padding, dilation);
    let w_out = conv_out_size(w, kw, stride, padding, dilation);
    let m = h_out * w_out;
    let k = c * kh * kw;
    let k_packed = k.div_ceil(8);
    let col_size = m * k;

    with_col_buf(col_size, |col| {
        let (global_min, global_max) = crate::backend::cpu::im2col::im2col_dispatch(
            input_n, c, h, w, kh, kw, stride, padding, dilation, col,
            true, // I4x8 needs min/max for asym quantization
        );
        let range = global_max - global_min;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let zero_point = global_min + 8.0 * scale;
        let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        let mut packed: Vec<I4x8> = Vec::with_capacity(m * k_packed);
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
                packed.push(I4x8(w));
            }
        }

        PackedTensor::from_raw(packed, vec![m, k], vec![scale], vec![zero_point])
    })
}

// SAFETY: Same as im2col_pack_i4x8.
unsafe fn im2col_pack_f4x8(
    input_n: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> PackedTensor<F4x8> {
    let h_out = conv_out_size(h, kh, stride, padding, dilation);
    let w_out = conv_out_size(w, kw, stride, padding, dilation);
    let m = h_out * w_out;
    let k = c * kh * kw;
    let k_packed = k.div_ceil(8);
    let col_size = m * k;

    with_col_buf(col_size, |col| {
        let (_global_min, _global_max) = crate::backend::cpu::im2col::im2col_dispatch(
            input_n, c, h, w, kh, kw, stride, padding, dilation, col,
            false, // F4x8 symmetric quant — uses max_abs from separate scan below
        );

        let max_abs = col.iter().copied().map(|v| v.abs()).fold(0.0f32, f32::max);
        let inv_scale = if max_abs > 0.0 {
            F4x8::MAX_REPRESENTABLE / max_abs
        } else {
            1.0
        };

        let mut packed: Vec<F4x8> = Vec::with_capacity(m * k_packed);
        for row in 0..m {
            let row_start = row * k;
            for word in 0..k_packed {
                let mut arr = [0.0f32; 8];
                let base = row_start + word * 8;
                for i in 0..8 {
                    let elem_idx = base + i;
                    if elem_idx < row_start + k {
                        arr[i] = col[elem_idx] * inv_scale;
                    }
                }
                packed.push(F4x8::pack_from_f32(arr));
            }
        }

        let scale = if max_abs > 0.0 {
            max_abs / F4x8::MAX_REPRESENTABLE
        } else {
            1.0
        };
        PackedTensor::from_raw(packed, vec![m, k], vec![scale], vec![0.0])
    })
}

// ── FP8 im2col + pack (generic for F8x4 and F8x4R) ─────────────

/// Generic im2col + pack for 8-bit float types (F8x4, F8x4R — both have 4 items per word).
///
/// Symmetric quantization: values are scaled so max_abs maps to T::MAX_REPRESENTABLE.
/// Zero point is always 0.0.
unsafe fn im2col_pack_float8<T: PackedWord>(
    input_n: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> PackedTensor<T> {
    debug_assert_eq!(T::ITEMS, 4);
    let h_out = conv_out_size(h, kh, stride, padding, dilation);
    let w_out = conv_out_size(w, kw, stride, padding, dilation);
    let m = h_out * w_out;
    let k = c * kh * kw;
    let k_packed = k.div_ceil(4);
    let col_size = m * k;

    with_col_buf(col_size, |col| {
        let (_global_min, _global_max) = crate::backend::cpu::im2col::im2col_dispatch(
            input_n, c, h, w, kh, kw, stride, padding, dilation, col,
            false, // F8x symmetric quant — uses separate max_abs scan below
        );

        let max_abs = col.iter().copied().map(|v| v.abs()).fold(0.0f32, f32::max);
        let inv_scale = if max_abs > 0.0 {
            T::MAX_REPRESENTABLE / max_abs
        } else {
            1.0
        };

        let mut packed: Vec<T> = Vec::with_capacity(m * k_packed);
        for row in 0..m {
            let row_start = row * k;
            for word in 0..k_packed {
                let mut arr: T::Array = Default::default();
                let slice = arr.as_mut();
                let base = row_start + word * 4;
                for i in 0..4 {
                    let elem_idx = base + i;
                    slice[i] = if elem_idx < row_start + k {
                        col[elem_idx] * inv_scale
                    } else {
                        0.0
                    };
                }
                packed.push(T::pack_from_f32(arr));
            }
        }

        let scale = if max_abs > 0.0 {
            max_abs / T::MAX_REPRESENTABLE
        } else {
            1.0
        };
        PackedTensor::from_raw(packed, vec![m, k], vec![scale], vec![0.0])
    })
}

/// Generic conv2d for 8-bit float packed types (F8x4, F8x4R).
///
/// Uses im2col_pack_float8 for activation packing and gemm_packed_float_fused for the GEMM.
pub unsafe fn conv2d_packed_float<T: PackedWord>(
    input: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    weight: &PackedTensor<T>,
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

            let act_packed = im2col_pack_float8::<T>(
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
            let b = bias.map(|b| {
                if groups > 1 {
                    &b[g_oc_off..g_oc_off + oc_per_g]
                } else {
                    b
                }
            });

            with_col_buf(num_pixels * local_oc, |temp| {
                gemm_packed_float_fused(&act_packed, &w_slice, b, activation, temp);
                for pixel in 0..num_pixels {
                    for f in 0..local_oc {
                        output[out_base + (g_oc_off + f) * num_pixels + pixel] =
                            temp[pixel * local_oc + f];
                    }
                }
            });
        }
    }
}

// ── Slice helper ────────────────────────────────────────────────

#[inline(always)]
// SAFETY: Caller must ensure `row_start + row_count` does not exceed the number
// of rows in the packed tensor, and the resulting tensor owns its data.
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

// ── GEMM: I8x4 with fused dequantize + bias + activation ────────
// Changes 1, 2, 3, 7, 11

// (Tile constants moved to per-GEMM scope below)

/// Raw slice-based I8x4 packed GEMM with fused dequantize + bias + activation.
/// Avoids PackedTensor wrapping overhead — takes slices and scalars directly.
/// Activation scale/zp are per-tensor (scalar). Weight scales/zps can be
/// per-channel (len == n) or per-tensor (len == 1, broadcast).
pub fn gemm_packed_i8x4_fused_raw(
    act_data: &[I8x4],
    m: usize,
    k: usize,
    act_scale: f32,
    act_zp: f32,
    w_data: &[I8x4],
    n: usize,
    w_scales: &[f32],
    w_zps: &[f32],
    bias: Option<&[f32]>,
    activation: Option<&str>,
    c: &mut [f32],
) {
    let k_packed = k.div_ceil(4);
    debug_assert_eq!(c.len(), m * n);
    let k_f32 = k as f32;
    let per_channel_w = w_scales.len() > 1;

    // Precompute per-column arrays (hoists per_channel_w and bias branches)
    let mut scale_ab_col: smallvec::SmallVec<[f32; 256]> = smallvec::SmallVec::with_capacity(n);
    let mut w_zp_col: smallvec::SmallVec<[f32; 256]> = smallvec::SmallVec::with_capacity(n);
    let mut bias_q_col: smallvec::SmallVec<[i32; 256]> = smallvec::SmallVec::with_capacity(n);
    let mut w_term_col: smallvec::SmallVec<[f32; 256]> = smallvec::SmallVec::with_capacity(n);
    let mut zp_prod_col: smallvec::SmallVec<[f32; 256]> = smallvec::SmallVec::with_capacity(n);
    for col in 0..n {
        let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
        let qb = w_row.iter().map(|&w| sum_i8x4_packed(w.0)).sum::<i32>();
        let ws = if per_channel_w {
            w_scales[col]
        } else {
            w_scales[0]
        };
        let wz = if per_channel_w { w_zps[col] } else { w_zps[0] };
        let sab = act_scale * ws;
        scale_ab_col.push(sab);
        w_zp_col.push(wz);
        bias_q_col.push(bias.map(|b| (b[col] / sab).round() as i32).unwrap_or(0));
        w_term_col.push(ws * qb as f32);
        zp_prod_col.push(act_zp * wz);
    }

    // Determine row iteration strategy
    #[cfg(feature = "parallel")]
    let parallel = m >= crate::backend::cpu::topology::physical_core_count();

    match activation {
        None | Some("") => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let act_row = &act_data[row * k_packed..(row + 1) * k_packed];
                let qa_sum: i32 = act_row.iter().map(|&w| sum_i8x4_packed(w.0)).sum();
                let r = act_scale * qa_sum as f32;
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = bias_q_col[col];
                    for kk in 0..k_packed {
                        acc += i8x4_dot_packed(act_row[kk].0, w_row[kk].0);
                    }
                    c_row[col] = (acc as f32) * scale_ab_col[col]
                        + w_zp_col[col] * r
                        + act_zp * w_term_col[col]
                        + zp_prod_col[col] * k_f32;
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
        Some("relu") => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let act_row = &act_data[row * k_packed..(row + 1) * k_packed];
                let qa_sum: i32 = act_row.iter().map(|&w| sum_i8x4_packed(w.0)).sum();
                let r = act_scale * qa_sum as f32;
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = bias_q_col[col];
                    for kk in 0..k_packed {
                        acc += i8x4_dot_packed(act_row[kk].0, w_row[kk].0);
                    }
                    let val = (acc as f32) * scale_ab_col[col]
                        + w_zp_col[col] * r
                        + act_zp * w_term_col[col]
                        + zp_prod_col[col] * k_f32;
                    c_row[col] = val.max(0.0);
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
        Some("silu") => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let act_row = &act_data[row * k_packed..(row + 1) * k_packed];
                let qa_sum: i32 = act_row.iter().map(|&w| sum_i8x4_packed(w.0)).sum();
                let r = act_scale * qa_sum as f32;
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = bias_q_col[col];
                    for kk in 0..k_packed {
                        acc += i8x4_dot_packed(act_row[kk].0, w_row[kk].0);
                    }
                    let val = (acc as f32) * scale_ab_col[col]
                        + w_zp_col[col] * r
                        + act_zp * w_term_col[col]
                        + zp_prod_col[col] * k_f32;
                    c_row[col] = val / (1.0 + (-val).exp());
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
        _ => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let act_row = &act_data[row * k_packed..(row + 1) * k_packed];
                let qa_sum: i32 = act_row.iter().map(|&w| sum_i8x4_packed(w.0)).sum();
                let r = act_scale * qa_sum as f32;
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = bias_q_col[col];
                    for kk in 0..k_packed {
                        acc += i8x4_dot_packed(act_row[kk].0, w_row[kk].0);
                    }
                    c_row[col] = (acc as f32) * scale_ab_col[col]
                        + w_zp_col[col] * r
                        + act_zp * w_term_col[col]
                        + zp_prod_col[col] * k_f32;
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
    }
}

/// PackedTensor-wrapping variant of `gemm_packed_i8x4_fused_raw`.
/// Extracts shape/scales/zps from PackedTensor and delegates to the raw version.
#[inline]
pub fn gemm_packed_i8x4_fused(
    act_packed: &PackedTensor<I8x4>,
    weight_packed: &PackedTensor<I8x4>,
    bias: Option<&[f32]>,
    activation: Option<&str>,
    c: &mut [f32],
) {
    let m = act_packed.shape()[0];
    let k = act_packed.shape()[1];
    let n = weight_packed.shape()[0];
    gemm_packed_i8x4_fused_raw(
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

// ── GEMM: I4x8 with fused dequantize + bias + activation ────────

/// Raw slice-based I4x8 packed GEMM with fused dequantize + bias + activation.
pub fn gemm_packed_i4x8_fused_raw(
    act_data: &[I4x8],
    m: usize,
    k: usize,
    act_scale: f32,
    act_zp: f32,
    w_data: &[I4x8],
    n: usize,
    w_scales: &[f32],
    w_zps: &[f32],
    bias: Option<&[f32]>,
    activation: Option<&str>,
    c: &mut [f32],
) {
    let k_packed = k.div_ceil(8);
    debug_assert_eq!(c.len(), m * n);
    let k_f32 = k as f32;
    let per_channel_w = w_scales.len() > 1;

    // Precompute per-column arrays (hoists per_channel_w and bias branches)
    let mut scale_ab_col: smallvec::SmallVec<[f32; 256]> = smallvec::SmallVec::with_capacity(n);
    let mut w_zp_col: smallvec::SmallVec<[f32; 256]> = smallvec::SmallVec::with_capacity(n);
    let mut bias_q_col: smallvec::SmallVec<[i32; 256]> = smallvec::SmallVec::with_capacity(n);
    let mut w_term_col: smallvec::SmallVec<[f32; 256]> = smallvec::SmallVec::with_capacity(n);
    let mut zp_prod_col: smallvec::SmallVec<[f32; 256]> = smallvec::SmallVec::with_capacity(n);
    for col in 0..n {
        let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
        let qb = w_row.iter().map(|&w| sum_i4x8_packed(w.0)).sum::<i32>();
        let ws = if per_channel_w {
            w_scales[col]
        } else {
            w_scales[0]
        };
        let wz = if per_channel_w { w_zps[col] } else { w_zps[0] };
        let sab = act_scale * ws;
        scale_ab_col.push(sab);
        w_zp_col.push(wz);
        bias_q_col.push(bias.map(|b| (b[col] / sab).round() as i32).unwrap_or(0));
        w_term_col.push(ws * qb as f32);
        zp_prod_col.push(act_zp * wz);
    }

    #[cfg(feature = "parallel")]
    let parallel = m >= crate::backend::cpu::topology::physical_core_count();

    match activation {
        None | Some("") => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let a_row = &act_data[row * k_packed..(row + 1) * k_packed];
                let qa_sum: i32 = a_row.iter().map(|&w| sum_i4x8_packed(w.0)).sum();
                let r = act_scale * qa_sum as f32;
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = bias_q_col[col];
                    for kk in 0..k_packed {
                        acc += i4x8_dot_packed(a_row[kk].0, w_row[kk].0);
                    }
                    c_row[col] = (acc as f32) * scale_ab_col[col]
                        + w_zp_col[col] * r
                        + act_zp * w_term_col[col]
                        + zp_prod_col[col] * k_f32;
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
        Some("relu") => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let a_row = &act_data[row * k_packed..(row + 1) * k_packed];
                let qa_sum: i32 = a_row.iter().map(|&w| sum_i4x8_packed(w.0)).sum();
                let r = act_scale * qa_sum as f32;
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = bias_q_col[col];
                    for kk in 0..k_packed {
                        acc += i4x8_dot_packed(a_row[kk].0, w_row[kk].0);
                    }
                    let val = (acc as f32) * scale_ab_col[col]
                        + w_zp_col[col] * r
                        + act_zp * w_term_col[col]
                        + zp_prod_col[col] * k_f32;
                    c_row[col] = val.max(0.0);
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
        Some("silu") => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let a_row = &act_data[row * k_packed..(row + 1) * k_packed];
                let qa_sum: i32 = a_row.iter().map(|&w| sum_i4x8_packed(w.0)).sum();
                let r = act_scale * qa_sum as f32;
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = bias_q_col[col];
                    for kk in 0..k_packed {
                        acc += i4x8_dot_packed(a_row[kk].0, w_row[kk].0);
                    }
                    let val = (acc as f32) * scale_ab_col[col]
                        + w_zp_col[col] * r
                        + act_zp * w_term_col[col]
                        + zp_prod_col[col] * k_f32;
                    c_row[col] = val / (1.0 + (-val).exp());
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
        _ => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let a_row = &act_data[row * k_packed..(row + 1) * k_packed];
                let qa_sum: i32 = a_row.iter().map(|&w| sum_i4x8_packed(w.0)).sum();
                let r = act_scale * qa_sum as f32;
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = bias_q_col[col];
                    for kk in 0..k_packed {
                        acc += i4x8_dot_packed(a_row[kk].0, w_row[kk].0);
                    }
                    c_row[col] = (acc as f32) * scale_ab_col[col]
                        + w_zp_col[col] * r
                        + act_zp * w_term_col[col]
                        + zp_prod_col[col] * k_f32;
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
    }
}

/// PackedTensor-wrapping variant. Delegates to `gemm_packed_i4x8_fused_raw`.
#[inline]
pub fn gemm_packed_i4x8_fused(
    act_packed: &PackedTensor<I4x8>,
    weight_packed: &PackedTensor<I4x8>,
    bias: Option<&[f32]>,
    activation: Option<&str>,
    c: &mut [f32],
) {
    let m = act_packed.shape()[0];
    let k = act_packed.shape()[1];
    let n = weight_packed.shape()[0];
    gemm_packed_i4x8_fused_raw(
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

/// Packed F4x8 GEMM: fused dequantize + bias + activation (raw slice entry).
///
/// Symmetric FP4 quantization, no zero-point.
/// Dequantization: output = acc * scale_a * scale_b / 4.0
/// Since activations use a single per-tensor scale, `act_scale` is a scalar.
/// Weight scales can be per-channel (len=n) or per-tensor (len=1).
pub fn gemm_packed_f4x8_fused_raw(
    act_data: &[F4x8],
    m: usize,
    k: usize,
    act_scale: f32,
    w_data: &[F4x8],
    n: usize,
    w_scales: &[f32],
    bias: Option<&[f32]>,
    activation: Option<&str>,
    c: &mut [f32],
) {
    let k_packed = k.div_ceil(8);
    debug_assert_eq!(c.len(), m * n);
    let per_channel_w = w_scales.len() > 1;

    // Precompute per-column arrays (hoists per_channel_w and bias branches)
    let mut scale_ab_col: smallvec::SmallVec<[f32; 256]> = smallvec::SmallVec::with_capacity(n);
    let mut bias_col: smallvec::SmallVec<[f32; 256]> = smallvec::SmallVec::with_capacity(n);
    for col in 0..n {
        let ws = if per_channel_w {
            w_scales[col]
        } else {
            w_scales[0]
        };
        scale_ab_col.push(act_scale * ws / 4.0);
        bias_col.push(bias.map(|b| b[col]).unwrap_or(0.0));
    }

    #[cfg(feature = "parallel")]
    let parallel = m >= crate::backend::cpu::topology::physical_core_count();

    match activation {
        None | Some("") => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let a_row = &act_data[row * k_packed..(row + 1) * k_packed];
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = 0i32;
                    for kk in 0..k_packed {
                        acc += f4x8_dot_packed(a_row[kk].0, w_row[kk].0);
                    }
                    c_row[col] = (acc as f32) * scale_ab_col[col] + bias_col[col];
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
        Some("relu") => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let a_row = &act_data[row * k_packed..(row + 1) * k_packed];
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = 0i32;
                    for kk in 0..k_packed {
                        acc += f4x8_dot_packed(a_row[kk].0, w_row[kk].0);
                    }
                    c_row[col] = ((acc as f32) * scale_ab_col[col] + bias_col[col]).max(0.0);
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
        Some("silu") => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let a_row = &act_data[row * k_packed..(row + 1) * k_packed];
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = 0i32;
                    for kk in 0..k_packed {
                        acc += f4x8_dot_packed(a_row[kk].0, w_row[kk].0);
                    }
                    let val = (acc as f32) * scale_ab_col[col] + bias_col[col];
                    c_row[col] = val / (1.0 + (-val).exp());
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
        _ => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let a_row = &act_data[row * k_packed..(row + 1) * k_packed];
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = 0i32;
                    for kk in 0..k_packed {
                        acc += f4x8_dot_packed(a_row[kk].0, w_row[kk].0);
                    }
                    c_row[col] = (acc as f32) * scale_ab_col[col] + bias_col[col];
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
    }
}

/// PackedTensor-wrapping variant of `gemm_packed_f4x8_fused_raw`.
#[inline]
pub fn gemm_packed_f4x8_fused(
    act_packed: &PackedTensor<F4x8>,
    weight_packed: &PackedTensor<F4x8>,
    bias: Option<&[f32]>,
    activation: Option<&str>,
    c: &mut [f32],
) {
    let m = act_packed.shape()[0];
    let k = act_packed.shape()[1];
    let n = weight_packed.shape()[0];
    gemm_packed_f4x8_fused_raw(
        act_packed.as_packed(),
        m,
        k,
        act_packed.scale_for_row(0),
        weight_packed.as_packed(),
        n,
        &weight_packed.scales,
        bias,
        activation,
        c,
    );
}

/// Pack flat i8 column-buffer data into I8x4 packed words.
///
/// Every 4 consecutive i8 bytes are packed into one u32 word:
/// `word[0] = bytes[0], word[1] = bytes[1] << 8, ...`
/// The i8→u8 conversion uses two's complement (safe cast).
/// The output `packed` must have at least `m * k_packed` capacity.
#[inline]
pub fn pack_i8_col_to_i8x4(col: &[i8], m: usize, k: usize, packed: &mut [I8x4]) {
    let k_packed = k.div_ceil(4);
    let full_words = if k % 4 == 0 { k_packed } else { k_packed - 1 };
    for row in 0..m {
        let row_start = row * k;
        let word_base = row * k_packed;
        // Fast path: read full u32 words via unaligned load (avoids byte loop)
        for w in 0..full_words {
            let byte_base = row_start + w * 4;
            // SAFETY: aligned within the slice bounds, u32 has same repr as 4×i8
            let word =
                unsafe { std::ptr::read_unaligned(col.as_ptr().add(byte_base) as *const u32) };
            packed[word_base + w] = I8x4(word);
        }
        // Tail: handle the last partial word when k is not a multiple of 4
        if k % 4 != 0 {
            let mut word = 0u32;
            let byte_base = row_start + full_words * 4;
            for i in 0..(k % 4) {
                let idx = byte_base + i;
                word |= (col[idx] as u8 as u32) << (i * 8);
            }
            packed[word_base + full_words] = I8x4(word);
        }
    }
}

/// Pack flat i8 column-buffer data into I4x8 packed words (2 nibbles per byte).
///
/// Every 8 i8 values (each clamped to [-8, 7]) are packed into one u32 word
/// as 8× 4-bit nibbles. Out-of-range values are clamped.
#[inline]
pub fn pack_i8_col_to_i4x8(col: &[i8], m: usize, k: usize, packed: &mut [I4x8]) {
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
            packed[word_base + w] = I4x8(word);
        }
    }
}

// ── Public Conv2d entry points ──────────────────────────────────

// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and sized according to the convolution parameters (n, c, h, w, oc, kh, kw, ...).
pub unsafe fn conv2d_packed_i8x4(
    input: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    weight: &PackedTensor<I8x4>,
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

            let act_packed = im2col_pack_i8x4(
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
            let b = bias.map(|b| {
                if groups > 1 {
                    &b[g_oc_off..g_oc_off + oc_per_g]
                } else {
                    b
                }
            });
            let nan_found = with_col_buf(num_pixels * local_oc, |temp| {
                gemm_packed_i8x4_fused(&act_packed, &w_slice, b, activation, temp);
                let mut found = false;
                for pixel in 0..num_pixels {
                    for f in 0..local_oc {
                        let v = temp[pixel * local_oc + f];
                        if v.is_nan() {
                            found = true;
                            eprintln!("[FNN_NAN] conv2d_u8 out_base={} g_oc_off={} f={} pixel={} v=nan temp_slice={:?}", out_base, g_oc_off, f, pixel, &temp[..10.min(temp.len())]);
                        }
                        output[out_base + (g_oc_off + f) * num_pixels + pixel] = v;
                    }
                }
                found
            });
            if nan_found {
                panic!("NaN in u8 conv output");
            }
        }
    }
}

// SAFETY: Same as conv2d_packed_i8x4 — caller ensures valid, non-overlapping
// input/output buffers sized for the convolution parameters.
pub unsafe fn conv2d_packed_i4x8(
    input: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    weight: &PackedTensor<I4x8>,
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

            let act_packed = im2col_pack_i4x8(
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
            let b = bias.map(|b| {
                if groups > 1 {
                    &b[g_oc_off..g_oc_off + oc_per_g]
                } else {
                    b
                }
            });
            let nan_found = with_col_buf(num_pixels * local_oc, |temp| {
                gemm_packed_i4x8_fused(&act_packed, &w_slice, b, activation, temp);
                let mut found = false;
                for pixel in 0..num_pixels {
                    for f in 0..local_oc {
                        let v = temp[pixel * local_oc + f];
                        if v.is_nan() {
                            found = true;
                            eprintln!("[FNN_NAN] conv2d_u8 out_base={} g_oc_off={} f={} pixel={} v=nan temp_slice={:?}", out_base, g_oc_off, f, pixel, &temp[..10.min(temp.len())]);
                        }
                        output[out_base + (g_oc_off + f) * num_pixels + pixel] = v;
                    }
                }
                found
            });
            if nan_found {
                panic!("NaN in u8 conv output");
            }
        }
    }
}

// SAFETY: Same as conv2d_packed_i4x8 — caller ensures valid, non-overlapping buffers.
pub unsafe fn conv2d_packed_f4x8(
    input: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    weight: &PackedTensor<F4x8>,
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

            let act_packed = im2col_pack_f4x8(
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
            let b = bias.map(|b| {
                if groups > 1 {
                    &b[g_oc_off..g_oc_off + oc_per_g]
                } else {
                    b
                }
            });
            with_col_buf(num_pixels * local_oc, |temp| {
                gemm_packed_f4x8_fused(&act_packed, &w_slice, b, activation, temp);
                for pixel in 0..num_pixels {
                    for f in 0..local_oc {
                        let v = temp[pixel * local_oc + f];
                        if v.is_nan() {
                            eprintln!(
                                "[FNN_NAN] conv2d_f4 out_base={} g_oc_off={} f={} pixel={} v=nan",
                                out_base, g_oc_off, f, pixel
                            );
                        }
                        output[out_base + (g_oc_off + f) * num_pixels + pixel] = v;
                    }
                }
            });
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::packed_gemm::quantize_activations_to_i4x8;
    use crate::dtypes::{F8x4, F8x4R};

    #[test]
    fn test_conv2d_packed_i4x8_basic() {
        let n = 1;
        let c = 8;
        let h = 1;
        let w = 1;
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let wdata = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let weight = PackedTensor::<I4x8>::from_f32_per_channel(&wdata, &[2, 8]);
        let mut output = vec![0.0f32; 2];
        unsafe {
            conv2d_packed_i4x8(
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
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite value at output[{}]: {}", i, v);
        }
        let max_abs = output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.01, "Output is all zeros (max_abs={})", max_abs);
    }

    #[test]
    fn test_gemm_packed_i4x8_fused_basic() {
        let input = vec![-4.0, -2.0, 0.0, 2.0, -3.0, -1.0, 1.0, 3.0];
        let act = quantize_activations_to_i4x8(&input);
        let weight_data = vec![
            4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let weight = PackedTensor::<I4x8>::from_f32_per_channel(&weight_data, &[2, 8]);
        let mut c = vec![0.0f32; 2];
        gemm_packed_i4x8_fused(&act, &weight, None, None, &mut c);
        assert_eq!(c.len(), 2);
        assert!(c[1] > 0.0, "expected second output positive, got {:?}", c);
    }

    #[test]
    fn test_conv2d_packed_i4x8_non_multiple_k() {
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

        let weight = PackedTensor::<I4x8>::from_f32_per_channel(&wdata, &[oc, k]);
        let h_out = h;
        let w_out = w;
        let mut output = vec![0.0f32; oc * h_out * w_out];
        unsafe {
            conv2d_packed_i4x8(
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
    fn test_conv2d_packed_i8x4_non_multiple_k() {
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

        let weight = PackedTensor::<I8x4>::from_f32_per_channel(&wdata, &[oc, k]);
        let h_out = h;
        let w_out = w;
        let mut output = vec![0.0f32; oc * h_out * w_out];
        unsafe {
            conv2d_packed_i8x4(
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
    fn test_conv2d_packed_f4x8_basic() {
        let n = 1;
        let c = 8;
        let h = 1;
        let w = 1;
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let wdata = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let weight = PackedTensor::<F4x8>::from_f32_per_channel(&wdata, &[2, 8]);
        let mut output = vec![0.0f32; 2];
        unsafe {
            conv2d_packed_f4x8(
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
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite value at output[{}]: {}", i, v);
        }
        let max_abs = output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.01, "Output is all zeros (max_abs={})", max_abs);
    }

    #[test]
    fn test_gemm_packed_f4x8_fused_basic() {
        let input = vec![-4.0, -2.0, 0.0, 2.0, -3.0, -1.0, 1.0, 3.0];
        let act = crate::backend::cpu::packed_gemm::quantize_activations_to_f4x8(&input);
        let weight_data = vec![
            4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let weight = PackedTensor::<F4x8>::from_f32_per_channel(&weight_data, &[2, 8]);
        let mut c = vec![0.0f32; 2];
        gemm_packed_f4x8_fused(&act, &weight, None, None, &mut c);
        assert_eq!(c.len(), 2);
        assert!(
            c[0] != 0.0 || c[1] != 0.0,
            "Expected non-zero output, got {:?}",
            c
        );
        for (i, &v) in c.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite value at output[{}]: {}", i, v);
        }
    }

    #[test]
    fn test_conv2d_packed_f8x4_basic() {
        let n = 1;
        let c = 4;
        let h = 1;
        let w = 1;
        let input: Vec<f32> = (1..=4).map(|i| i as f32).collect();
        let wdata = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let weight = PackedTensor::<F8x4>::from_f32_per_channel(&wdata, &[2, 4]);
        let mut output = vec![0.0f32; 2];
        unsafe {
            conv2d_packed_float::<F8x4>(
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
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite value at output[{}]: {}", i, v);
        }
        let max_abs = output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.01, "Output is all zeros (max_abs={})", max_abs);
    }

    #[test]
    fn test_conv2d_packed_f8x4r_basic() {
        let n = 1;
        let c = 4;
        let h = 1;
        let w = 1;
        let input: Vec<f32> = (1..=4).map(|i| i as f32).collect();
        let wdata = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let weight = PackedTensor::<F8x4R>::from_f32_per_channel(&wdata, &[2, 4]);
        let mut output = vec![0.0f32; 2];
        unsafe {
            conv2d_packed_float::<F8x4R>(
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
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite value at output[{}]: {}", i, v);
        }
        let max_abs = output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.01, "Output is all zeros (max_abs={})", max_abs);
    }
}
