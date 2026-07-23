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
    i4x8_dot_packed, i8x4_dot_packed, sum_i4x8_packed, sum_i8x4_packed,
};
use crate::backend::prepared::PreparedActivation;
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
        let mut packed: Vec<I8x4> = vec![I8x4(0); m * k_packed];
        // Keep one correctness path here. The former AVX2 branch wrote tail words
        // past the pre-sized vector and used unsigned saturation for negative i8
        // values, corrupting every convolution whose K did not fill its SIMD tile.
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
                packed[row * k_packed + word] = I8x4(w);
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

/// Generic conv2d for 8-bit float packed types (F8x4, F8x4R).
///
/// Fast path: does im2col in f32, unpacks weights to f32 via LUT, calls gemm::gemm::<f32>.
/// Skips the wasteful f32→packed→f32 quantize/dequantize cycle on activations.
///
/// # Safety
///
/// Caller must ensure `input` and `output` are valid, non-overlapping,
/// and sized according to the convolution parameters (n, c, h, w, oc, kh, kw, ...).
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
    activation: PreparedActivation,
    output: &mut [f32],
) {
    let oc = weight.shape()[0];

    let f32_weights = weight.get_or_init_f32_weights();

    let conv_act = prepared_to_conv_activation(activation);
    let bias_slice = bias.unwrap_or(&[]);

    super::microkernels::conv::conv2d_f32_im2col_gemm(
        input,
        f32_weights,
        bias_slice,
        output,
        n,
        c,
        h,
        w,
        oc,
        kh,
        kw,
        stride,
        padding,
        dilation,
        groups,
        conv_act,
    );
}

#[inline(always)]
fn prepared_to_conv_activation(
    a: PreparedActivation,
) -> Option<super::microkernels::ConvActivation> {
    use super::microkernels::ConvActivation;
    match a {
        PreparedActivation::None => None,
        PreparedActivation::Relu => Some(ConvActivation::Relu),
        PreparedActivation::Gelu => Some(ConvActivation::Gelu),
        PreparedActivation::Silu => Some(ConvActivation::Silu),
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

#[inline]
fn dot_i8x4_rows(a: &[I8x4], b: &[I8x4]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 availability is checked above. The helper performs only
        // unaligned in-bounds loads and uses the scalar reference for its tail.
        return unsafe { dot_i8x4_rows_avx2(a, b) };
    }
    a.iter()
        .zip(b)
        .map(|(&lhs, &rhs)| i8x4_dot_packed(lhs.0, rhs.0))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8x4_rows_avx2(a: &[I8x4], b: &[I8x4]) -> i32 {
    use std::arch::x86_64::*;

    let mut words = 0;
    let mut sums = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(1);
    while words + 4 <= a.len() {
        // SAFETY: The loop condition proves both 16-byte source ranges exist;
        // loadu has no alignment requirement.
        let lhs = unsafe { _mm_loadu_si128(a.as_ptr().add(words).cast::<__m128i>()) };
        let rhs = unsafe { _mm_loadu_si128(b.as_ptr().add(words).cast::<__m128i>()) };
        let lhs_i16 = _mm256_cvtepi8_epi16(lhs);
        let rhs_i16 = _mm256_cvtepi8_epi16(rhs);
        let products = _mm256_mullo_epi16(lhs_i16, rhs_i16);
        sums = _mm256_add_epi32(sums, _mm256_madd_epi16(products, ones));
        words += 4;
    }

    let hi = _mm256_extracti128_si256::<1>(sums);
    let mut sum128 = _mm_add_epi32(_mm256_castsi256_si128(sums), hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    let mut total = _mm_cvtsi128_si32(sum128);
    for index in words..a.len() {
        total += i8x4_dot_packed(a[index].0, b[index].0);
    }
    total
}

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
    activation: PreparedActivation,
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
        PreparedActivation::None => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let act_row = &act_data[row * k_packed..(row + 1) * k_packed];
                let qa_sum: i32 = act_row.iter().map(|&w| sum_i8x4_packed(w.0)).sum();
                let r = act_scale * qa_sum as f32;
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let acc = bias_q_col[col] + dot_i8x4_rows(act_row, w_row);
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
        PreparedActivation::Relu => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let act_row = &act_data[row * k_packed..(row + 1) * k_packed];
                let qa_sum: i32 = act_row.iter().map(|&w| sum_i8x4_packed(w.0)).sum();
                let r = act_scale * qa_sum as f32;
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let acc = bias_q_col[col] + dot_i8x4_rows(act_row, w_row);
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
        PreparedActivation::Silu => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let act_row = &act_data[row * k_packed..(row + 1) * k_packed];
                let qa_sum: i32 = act_row.iter().map(|&w| sum_i8x4_packed(w.0)).sum();
                let r = act_scale * qa_sum as f32;
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let acc = bias_q_col[col] + dot_i8x4_rows(act_row, w_row);
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
        PreparedActivation::Gelu => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let act_row = &act_data[row * k_packed..(row + 1) * k_packed];
                let qa_sum: i32 = act_row.iter().map(|&w| sum_i8x4_packed(w.0)).sum();
                let r = act_scale * qa_sum as f32;
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let acc = bias_q_col[col] + dot_i8x4_rows(act_row, w_row);
                    let val = (acc as f32) * scale_ab_col[col]
                        + w_zp_col[col] * r
                        + act_zp * w_term_col[col]
                        + zp_prod_col[col] * k_f32;
                    c_row[col] = val
                        * 0.5
                        * (1.0 + (val * 0.797_884_6 * (1.0 + 0.044715 * val * val)).tanh());
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
    activation: PreparedActivation,
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
    activation: PreparedActivation,
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
        PreparedActivation::None => {
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
        PreparedActivation::Relu => {
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
        PreparedActivation::Silu => {
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
        PreparedActivation::Gelu => {
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
                    c_row[col] = val
                        * 0.5
                        * (1.0 + (val * 0.797_884_6 * (1.0 + 0.044715 * val * val)).tanh());
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
    activation: PreparedActivation,
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
    activation: PreparedActivation,
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
        PreparedActivation::None => {
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
        PreparedActivation::Relu => {
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
        PreparedActivation::Silu => {
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
        PreparedActivation::Gelu => {
            let compute_row = |row: usize, c_row: &mut [f32]| {
                let a_row = &act_data[row * k_packed..(row + 1) * k_packed];
                for col in 0..n {
                    let w_row = &w_data[col * k_packed..(col + 1) * k_packed];
                    let mut acc = 0i32;
                    for kk in 0..k_packed {
                        acc += f4x8_dot_packed(a_row[kk].0, w_row[kk].0);
                    }
                    let val = (acc as f32) * scale_ab_col[col] + bias_col[col];
                    c_row[col] = val
                        * 0.5
                        * (1.0 + (val * 0.797_884_6 * (1.0 + 0.044715 * val * val)).tanh());
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
    activation: PreparedActivation,
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
    let full_words = if k.is_multiple_of(4) {
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
            let word =
                unsafe { std::ptr::read_unaligned(col.as_ptr().add(byte_base) as *const u32) };
            packed[word_base + w] = I8x4(word);
        }
        // Tail: handle the last partial word when k is not a multiple of 4
        if !k.is_multiple_of(4) {
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

/// # Safety
///
/// Caller must ensure `input` and `output` are valid, non-overlapping,
/// and sized according to the convolution parameters (n, c, h, w, oc, kh, kw, ...).
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
    activation: PreparedActivation,
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
            with_col_buf(num_pixels * local_oc, |temp| {
                gemm_packed_i8x4_fused(&act_packed, &w_slice, b, activation, temp);
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

/// # Safety
///
/// Same as conv2d_packed_i8x4 — caller ensures valid, non-overlapping
/// input/output buffers sized for the convolution parameters.
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
    activation: PreparedActivation,
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
            with_col_buf(num_pixels * local_oc, |temp| {
                gemm_packed_i4x8_fused(&act_packed, &w_slice, b, activation, temp);
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

/// # Safety
///
/// Same as conv2d_packed_i4x8 — caller ensures valid, non-overlapping buffers.
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
    activation: PreparedActivation,
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
    fn test_i8x4_row_dot_matches_scalar_at_signed_endpoints_and_tails() {
        let values = [-128i8, -127, -65, -1, 0, 1, 63, 126, 127];
        for words in 1..=33 {
            let lhs: Vec<I8x4> = (0..words)
                .map(|word| {
                    let mut packed = 0u32;
                    for lane in 0..4 {
                        let value = values[(word * 4 + lane) % values.len()];
                        packed |= (value as u8 as u32) << (lane * 8);
                    }
                    I8x4(packed)
                })
                .collect();
            let rhs: Vec<I8x4> = (0..words)
                .map(|word| {
                    let mut packed = 0u32;
                    for lane in 0..4 {
                        let value = values[values.len() - 1 - (word * 3 + lane) % values.len()];
                        packed |= (value as u8 as u32) << (lane * 8);
                    }
                    I8x4(packed)
                })
                .collect();
            let expected: i32 = lhs
                .iter()
                .zip(&rhs)
                .map(|(&a, &b)| i8x4_dot_packed(a.0, b.0))
                .sum();
            assert_eq!(dot_i8x4_rows(&lhs, &rhs), expected, "words={words}");
        }
    }

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
                PreparedActivation::None,
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
    fn test_conv2d_packed_i4x8_nonfinite_input_never_panics() {
        let input = vec![f32::NAN; 8];
        let weight = PackedTensor::<I4x8>::from_f32_per_channel(&[1.0; 8], &[1, 8]);
        let mut output = vec![0.0f32; 1];
        unsafe {
            conv2d_packed_i4x8(
                &input,
                1,
                8,
                1,
                1,
                &weight,
                None,
                1,
                0,
                1,
                1,
                1,
                1,
                PreparedActivation::None,
                &mut output,
            );
        }
        assert_eq!(output.len(), 1);
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
        gemm_packed_i4x8_fused(&act, &weight, None, PreparedActivation::None, &mut c);
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
                PreparedActivation::None,
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
    fn test_conv2d_packed_i4x8_matches_its_dequantized_operands() {
        let (n, c, h, w, oc, kh, kw) = (1, 3, 8, 8, 5, 3, 3);
        let (stride, padding, dilation, groups) = (2, 1, 1, 1);
        let input: Vec<f32> = (0..n * c * h * w)
            .map(|i| ((i * 37 % 211) as f32 - 105.0) / 29.0)
            .collect();
        let k = c * kh * kw;
        let weights: Vec<f32> = (0..oc * k)
            .map(|i| ((i * 53 % 173) as f32 - 86.0) / 113.0)
            .collect();
        let weight = PackedTensor::<I4x8>::from_f32_per_channel(&weights, &[oc, k]);
        let h_out = conv_out_size(h, kh, stride, padding, dilation);
        let w_out = conv_out_size(w, kw, stride, padding, dilation);
        let pixels = h_out * w_out;
        let mut actual = vec![0.0; n * oc * pixels];

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
                PreparedActivation::None,
                &mut actual,
            );
        }

        let activations =
            unsafe { im2col_pack_i4x8(&input, c, h, w, kh, kw, stride, padding, dilation) }
                .to_f32_vec();
        let mut reference_im2col = vec![0.0; pixels * k];
        // SAFETY: `input` contains one complete NCHW image and
        // `reference_im2col` is sized to the exact output matrix extent.
        let _ = unsafe {
            crate::backend::cpu::im2col::im2col_dispatch(
                &input,
                c,
                h,
                w,
                kh,
                kw,
                stride,
                padding,
                dilation,
                &mut reference_im2col,
                false,
            )
        };
        let activation_step = (reference_im2col
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            - reference_im2col
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min))
            / 15.0;
        for (index, (&got, &want)) in activations.iter().zip(&reference_im2col).enumerate() {
            assert!(
                (got - want).abs() <= activation_step + 1e-6,
                "packed activation[{index}] mismatch: got {got}, expected {want}, step {activation_step}"
            );
        }

        let dequantized_weights = weight.to_f32_vec();
        let mut expected = vec![0.0; actual.len()];
        for pixel in 0..pixels {
            for out_channel in 0..oc {
                let mut sum = 0.0;
                for depth in 0..k {
                    sum += activations[pixel * k + depth]
                        * dequantized_weights[out_channel * k + depth];
                }
                expected[out_channel * pixels + pixel] = sum;
            }
        }

        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            let error = (got - want).abs();
            assert!(
                error <= 2e-4,
                "output[{index}] mismatch: got {got}, expected {want}, error {error}"
            );
        }
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
                PreparedActivation::None,
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
    fn test_conv2d_packed_i8x4_matches_its_dequantized_operands() {
        let (n, c, h, w, oc, kh, kw) = (1, 3, 8, 8, 5, 3, 3);
        let (stride, padding, dilation, groups) = (2, 1, 1, 1);
        let input: Vec<f32> = (0..n * c * h * w)
            .map(|i| ((i * 37 % 211) as f32 - 105.0) / 29.0)
            .collect();
        let k = c * kh * kw;
        let weights: Vec<f32> = (0..oc * k)
            .map(|i| ((i * 53 % 173) as f32 - 86.0) / 113.0)
            .collect();
        let bias: Vec<f32> = (0..oc).map(|i| i as f32 * 0.07 - 0.11).collect();
        let weight = PackedTensor::<I8x4>::from_f32_per_channel(&weights, &[oc, k]);
        let h_out = conv_out_size(h, kh, stride, padding, dilation);
        let w_out = conv_out_size(w, kw, stride, padding, dilation);
        let pixels = h_out * w_out;
        let mut actual = vec![0.0; n * oc * pixels];

        unsafe {
            conv2d_packed_i8x4(
                &input,
                n,
                c,
                h,
                w,
                &weight,
                Some(&bias),
                stride,
                padding,
                dilation,
                groups,
                kh,
                kw,
                PreparedActivation::None,
                &mut actual,
            );
        }

        let activations =
            unsafe { im2col_pack_i8x4(&input, c, h, w, kh, kw, stride, padding, dilation) }
                .to_f32_vec();
        let mut reference_im2col = vec![0.0; pixels * k];
        // SAFETY: `input` contains one complete NCHW image and
        // `reference_im2col` is sized to the exact output matrix extent.
        let _ = unsafe {
            crate::backend::cpu::im2col::im2col_dispatch(
                &input,
                c,
                h,
                w,
                kh,
                kw,
                stride,
                padding,
                dilation,
                &mut reference_im2col,
                false,
            )
        };
        let activation_step = (reference_im2col
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            - reference_im2col
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min))
            / 255.0;
        for (index, (&got, &want)) in activations.iter().zip(&reference_im2col).enumerate() {
            assert!(
                (got - want).abs() <= activation_step + 1e-6,
                "packed activation[{index}] mismatch: got {got}, expected {want}, step {activation_step}"
            );
        }
        let dequantized_weights = weight.to_f32_vec();
        let mut expected = vec![0.0; actual.len()];
        for pixel in 0..pixels {
            for out_channel in 0..oc {
                let mut sum = bias[out_channel];
                for depth in 0..k {
                    sum += activations[pixel * k + depth]
                        * dequantized_weights[out_channel * k + depth];
                }
                expected[out_channel * pixels + pixel] = sum;
            }
        }

        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            let error = (got - want).abs();
            assert!(
                error <= 2e-4,
                "output[{index}] mismatch: got {got}, expected {want}, error {error}"
            );
        }
    }

    #[test]
    fn test_conv2d_packed_i8x4_simd_path() {
        // k_packed >= 8 exercises the SIMD fast path (AVX2) that was
        // panicking before the fix — Vec::with_capacity gave len==0
        // while the SIMD loop did direct index assignment.
        let n = 1;
        let c = 8; // c*kh*kw = 8*3*3 = 72, k_packed = 18 >= 8
        let h = 4;
        let w = 4;
        let kh = 3;
        let kw = 3;
        let stride = 1;
        let padding = 1;
        let dilation = 1;
        let oc = 4;
        let groups = 1;

        let input: Vec<f32> = (0..(c * h * w)).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let k = c * kh * kw;
        let mut wdata = vec![0.0f32; oc * k];
        for i in 0..c {
            wdata[i * kh * kw + kh * kw / 2] = 1.0 / c as f32;
        }
        for i in 0..c {
            wdata[k + i * kh * kw + kh * kw / 2] = 0.5 / c as f32;
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
                PreparedActivation::None,
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
                PreparedActivation::None,
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
        gemm_packed_f4x8_fused(&act, &weight, None, PreparedActivation::None, &mut c);
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
                PreparedActivation::None,
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
                PreparedActivation::None,
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
