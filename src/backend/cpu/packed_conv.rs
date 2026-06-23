//! SWAR-based packed Conv2d kernels
//!
//! im2col + fused packed GEMM using SWAR dot products.
//! No unpacking of weights in the hot loop — activations are
//! quantized to the same packed format as weights and the dot
//! product runs entirely in i32 via SWAR.

use crate::backend::cpu::swar::{u4x8_dot_packed, u8x4_dot_packed};
use crate::dtypes::{PackedWord, U4x8, U8x4};
use crate::packed_tensor::PackedTensor;

#[inline(always)]
fn conv_out_size(input: usize, kernel: usize, stride: usize, padding: usize, dilation: usize) -> usize {
    let dk = (kernel - 1) * dilation + 1;
    if input + 2 * padding >= dk {
        (input + 2 * padding - dk) / stride + 1
    } else {
        0
    }
}

/// im2col for a single image (no batch).
#[inline(always)]
unsafe fn im2col_f32(
    input_n: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    col: &mut [f32],
) {
    let h_out = conv_out_size(h, kh, stride, padding, dilation);
    let w_out = conv_out_size(w, kw, stride, padding, dilation);
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
                            col[dst] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

/// im2col → quantize to PackedTensor<U8x4> with shape [M, K].
///
/// Each row is packed independently (row-wise layout) so that the GEMM kernel
/// can index into `row * ceil(K/4)` packed words. This is critical when K is
/// not a multiple of 4 — flat packing would cross row boundaries and corrupt
/// data.
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

    let mut col = vec![0.0f32; m * k];
    im2col_f32(input_n, c, h, w, kh, kw, stride, padding, dilation, &mut col);

    // Per-tensor quantization from the full buffer
    let min = col.iter().copied().fold(f32::INFINITY, f32::min);
    let max = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
    let zero_point = if range > 0.0 { min + 128.0 * scale } else { 0.0 };
    let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

    // Pack each row independently for correct row-wise alignment
    let mut packed: Vec<U8x4> = Vec::with_capacity(m * k_packed);
    for row in 0..m {
        let row_start = row * k;
        for word in 0..k_packed {
            let mut w = 0u32;
            for i in 0..4 {
                let elem_idx = row_start + word * 4 + i;
                if elem_idx < row_start + k && elem_idx < col.len() {
                    let val = col[elem_idx];
                    let q = ((val - zero_point) * inv_scale).round().clamp(-128.0, 127.0) as i8;
                    w |= (q as u8 as u32) << (i * 8);
                }
            }
            packed.push(U8x4(w));
        }
    }

    PackedTensor::from_raw(packed, vec![m, k], vec![scale], vec![zero_point])
}

/// im2col → quantize to PackedTensor<U4x8> with shape [M, K].
///
/// Each row is packed independently (row-wise layout) so that the GEMM kernel
/// can index into `row * ceil(K/8)` packed words. This is critical when K is
/// not a multiple of 8 — flat packing would cross row boundaries and corrupt
/// data.
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

    let mut col = vec![0.0f32; m * k];
    im2col_f32(input_n, c, h, w, kh, kw, stride, padding, dilation, &mut col);

    // Per-tensor quantization from the full buffer
    let min = col.iter().copied().fold(f32::INFINITY, f32::min);
    let max = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
    let zero_point = if range > 0.0 { min + 8.0 * scale } else { 0.0 };
    let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

    // Pack each row independently for correct row-wise alignment
    let mut packed: Vec<U4x8> = Vec::with_capacity(m * k_packed);
    for row in 0..m {
        let row_start = row * k;
        for word in 0..k_packed {
            let mut w = 0u32;
            for i in 0..8 {
                let elem_idx = row_start + word * 8 + i;
                if elem_idx < row_start + k && elem_idx < col.len() {
                    let val = col[elem_idx];
                    let q = ((val - zero_point) * inv_scale).round().clamp(-8.0, 7.0) as i32;
                    let q_u = (q as u32) & 0xF;
                    w |= q_u << (i * 4);
                }
            }
            packed.push(U4x8(w));
        }
    }

    PackedTensor::from_raw(packed, vec![m, k], vec![scale], vec![zero_point])
}

/// Extract a row-subset of a PackedTensor (copies data).
#[inline(always)]
unsafe fn slice_packed<U: PackedWord>(t: &PackedTensor<U>, row_start: usize, row_count: usize) -> PackedTensor<U> {
    let inner: usize = t.shape()[1..].iter().product();
    let k_packed = inner.div_ceil(U::ITEMS);
    let data = t.as_packed()[row_start * k_packed..(row_start + row_count) * k_packed].to_vec();
    let scales = if t.scales.len() > 1 { t.scales[row_start..row_start + row_count].to_vec() } else { t.scales.clone() };
    let zeros = if t.zeros.len() > 1 { t.zeros[row_start..row_start + row_count].to_vec() } else { t.zeros.clone() };
    PackedTensor::from_raw(data, vec![row_count, inner], scales, zeros)
}

/// SWAR Conv2d with U8x4 packed weights (kernel name conv2d_u8 / conv2d_u8_i8).
///
/// # Parameters
/// - `input`: FP32 activations [N, C, H, W] row-major NCHW
/// - `n, c, h, w`: input tensor dimensions
/// - `weight`: PackedTensor<U8x4> with shape [OC, C_per_group * KH * KW]
/// - `bias`: optional bias [OC]
/// - `stride, padding, dilation, groups`: standard conv2d hyper-parameters
/// - `kh, kw`: kernel spatial dimensions
/// - `activation`: optional fused activation ("relu", "silu")
/// - `output`: pre-allocated [N, OC, H_out, W_out] NCHW
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
                c_per_g, h, w, kh, kw, stride, padding, dilation,
            );

            let w_slice = if groups > 1 {
                slice_packed(weight, g_oc_off, oc_per_g)
            } else {
                slice_packed(weight, 0, oc)
            };

            let local_oc = w_slice.shape()[0];
            let mut temp = vec![0.0f32; num_pixels * local_oc];
            let b = bias.map(|b| {
                if groups > 1 { &b[g_oc_off..g_oc_off + oc_per_g] } else { b }
            });
            gemm_packed_u8x4_fused(&act_packed, &w_slice, b, activation, &mut temp);

            for pixel in 0..num_pixels {
                for f in 0..local_oc {
                    output[out_base + (g_oc_off + f) * num_pixels + pixel] = temp[pixel * local_oc + f];
                }
            }
        }
    }
}

/// SWAR Conv2d with U4x8 packed weights (kernel name conv2d_u4 / conv2d_u4_i8).
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
                c_per_g, h, w, kh, kw, stride, padding, dilation,
            );

            let w_slice = if groups > 1 {
                slice_packed(weight, g_oc_off, oc_per_g)
            } else {
                slice_packed(weight, 0, oc)
            };

            let local_oc = w_slice.shape()[0];
            let mut temp = vec![0.0f32; num_pixels * local_oc];
            let b = bias.map(|b| {
                if groups > 1 { &b[g_oc_off..g_oc_off + oc_per_g] } else { b }
            });
            gemm_packed_u4x8_fused(&act_packed, &w_slice, b, activation, &mut temp);

            for pixel in 0..num_pixels {
                for f in 0..local_oc {
                    output[out_base + (g_oc_off + f) * num_pixels + pixel] = temp[pixel * local_oc + f];
                }
            }
        }
    }
}

/// Fused U4x8 GEMM: C = A × Bᵀ with dequantize + bias + activation.
///
/// Full 4-term dequantization with per-channel scales for both activations and weights:
///   output = acc * scaleA * scaleB + zpB * scaleA * ΣqA + zpA * scaleB * ΣqB + zpA * zpB * K
#[inline]
pub fn gemm_packed_u4x8_fused(
    act_packed: &PackedTensor<U4x8>,
    weight_packed: &PackedTensor<U4x8>,
    bias: Option<&[f32]>,
    activation: Option<&str>,
    c: &mut [f32],
) {
    let m = act_packed.shape()[0];
    let k_packed = act_packed.shape()[1].div_ceil(8);
    let n = weight_packed.shape()[0];

    assert_eq!(weight_packed.shape()[1].div_ceil(8), k_packed, "K dimension mismatch in U4x8");
    assert_eq!(c.len(), m * n, "Output buffer size mismatch");

    let act_packed_slice = act_packed.as_packed();
    let weight_packed_slice = weight_packed.as_packed();

    let k_f32 = act_packed.shape()[1] as f32; // actual K, not padded K

    for row in 0..m {
        let a_row_start = row * k_packed;
        let a_row = &act_packed_slice[a_row_start..a_row_start + k_packed];

        // Per-channel activation scale/zero_point (falls back to per-tensor if only 1)
        let act_scale = act_packed.scale_for_row(row);
        let act_zp = act_packed.zero_for_row(row);

        let qa_sum: i32 = a_row.iter().map(|&w| {
            (0..8).map(|i| {
                let nib = ((w.0 >> (i * 4)) & 0xF) as i32;
                if nib >= 8 { nib - 16 } else { nib }
            }).sum::<i32>()
        }).sum();

        for col in 0..n {
            let w_row_start = col * k_packed;
            let w_row = &weight_packed_slice[w_row_start..w_row_start + k_packed];

            // Per-channel weight scale/zero_point
            let w_scale = weight_packed.scale_for_row(col);
            let w_zp = weight_packed.zero_for_row(col);

            let mut acc = 0i32;
            for k in 0..k_packed {
                acc += u4x8_dot_packed(a_row[k].0, w_row[k].0);
            }

            let qb_sum: i32 = w_row.iter().map(|&w| {
                (0..8).map(|i| {
                    let nib = ((w.0 >> (i * 4)) & 0xF) as i32;
                    if nib >= 8 { nib - 16 } else { nib }
                }).sum::<i32>()
            }).sum();

            let scale_ab = act_scale * w_scale;
            let mut val = (acc as f32) * scale_ab
                + w_zp * (act_scale * qa_sum as f32)
                + act_zp * (w_scale * qb_sum as f32)
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
            c[row * n + col] = val;
        }
    }
}

/// Fused U8x4 GEMM: C = A × Bᵀ with dequantize + bias + activation.
fn gemm_packed_u8x4_fused(
    act_packed: &PackedTensor<U8x4>,
    weight_packed: &PackedTensor<U8x4>,
    bias: Option<&[f32]>,
    activation: Option<&str>,
    c: &mut [f32],
) {
    let m = act_packed.shape()[0];
    let k_packed = act_packed.shape()[1].div_ceil(4);
    let n = weight_packed.shape()[0];

    assert_eq!(weight_packed.shape()[1].div_ceil(4), k_packed);
    assert_eq!(c.len(), m * n);

    let act_packed_slice = act_packed.as_packed();
    let weight_packed_slice = weight_packed.as_packed();

    for row in 0..m {
        let act_row_start = row * k_packed;
        let act_row = &act_packed_slice[act_row_start..act_row_start + k_packed];

        let act_scale = act_packed.scale_for_row(row);
        let act_zp = act_packed.zero_for_row(row);

        for col in 0..n {
            let w_row_start = col * k_packed;
            let w_row = &weight_packed_slice[w_row_start..w_row_start + k_packed];

            let w_scale = weight_packed.scale_for_row(col);
            let w_zp = weight_packed.zero_for_row(col);

            let mut acc = 0i32;
            for k in 0..k_packed {
                acc += u8x4_dot_packed(act_row[k].0, w_row[k].0);
            }

            let qa_sum: i32 = act_row.iter()
                .map(|&w| {
                    let b0 = (w.0 & 0xFF) as i8 as i32;
                    let b1 = ((w.0 >> 8) & 0xFF) as i8 as i32;
                    let b2 = ((w.0 >> 16) & 0xFF) as i8 as i32;
                    let b3 = ((w.0 >> 24) & 0xFF) as i8 as i32;
                    b0 + b1 + b2 + b3
                })
                .sum();
            let qb_sum: i32 = w_row.iter()
                .map(|&w| {
                    let b0 = (w.0 & 0xFF) as i8 as i32;
                    let b1 = ((w.0 >> 8) & 0xFF) as i8 as i32;
                    let b2 = ((w.0 >> 16) & 0xFF) as i8 as i32;
                    let b3 = ((w.0 >> 24) & 0xFF) as i8 as i32;
                    b0 + b1 + b2 + b3
                })
                .sum();
            let k_f32 = act_packed.shape()[1] as f32; // actual K, not padded K
            let scale_ab = act_scale * w_scale;

            let mut val = (acc as f32) * scale_ab
                + w_zp * (act_scale * qa_sum as f32)
                + act_zp * (w_scale * qb_sum as f32)
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
            c[row * n + col] = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::packed_gemm::quantize_activations_to_u4x8;

    #[test]
    fn test_conv2d_packed_u4x8_basic() {
        // 1x1 conv: N=1, C=8, H=1, W=1, OC=2, groups=1
        // Input [1..8] reshaped as NCHW
        let n = 1; let c = 8; let h = 1; let w = 1;
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        // Weight [OC=2, IC=8]: row0 picks first 4, row1 picks next 4
        let wdata = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let weight = PackedTensor::<U4x8>::from_f32_per_channel(&wdata, &[2, 8]);
        let mut output = vec![0.0f32; 2];
        unsafe {
            conv2d_packed_u4x8(
                &input, n, c, h, w, &weight, None,
                1, 0, 1, 1, 1, 1, None, &mut output,
            );
        }
        assert_eq!(output.len(), 2);
        assert!(output[0] > 0.0, "Expected positive output[0], got {}", output[0]);
    }

    #[test]
    fn test_gemm_packed_u4x8_fused_basic() {
        let input = vec![-4.0, -2.0, 0.0, 2.0, -3.0, -1.0, 1.0, 3.0];
        let act = quantize_activations_to_u4x8(&input);
        let weight_data = vec![
            4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let weight = PackedTensor::<U4x8>::from_f32_per_channel(&weight_data, &[2, 8]);
        let mut c = vec![0.0f32; 2];
        gemm_packed_u4x8_fused(&act, &weight, None, None, &mut c);
        assert_eq!(c.len(), 2);
        assert!(c[1] > 0.0, "expected second output positive, got {:?}", c);
    }

    /// Test conv2d U4 with K not a multiple of 8 (e.g. C=3, KH=3, KW=3 → K=27).
    /// This is the real-world case for the first conv layer in YOLO and catches the
    /// row-wise packing bug where flat packing would cross row boundaries.
    #[test]
    fn test_conv2d_packed_u4x8_non_multiple_k() {
        // 3x3 conv: N=1, C=3, H=4, W=4, OC=2, KH=KW=3, stride=1, pad=1
        // K = 3*3*3 = 27 (not a multiple of 8!)
        let n = 1; let c = 3; let h = 4; let w = 4;
        let kh = 3; let kw = 3;
        let stride = 1; let padding = 1; let dilation = 1;
        let oc = 2; let groups = 1;

        let input: Vec<f32> = (0..(c * h * w)).map(|i| (i as f32 - 24.0) * 0.1).collect();
        // Simple identity-ish weights (each output channel picks a specific input channel)
        let k = c * kh * kw; // 27
        let mut wdata = vec![0.0f32; oc * k];
        // OC=0 picks first C elements with stride=kh*kw=9
        for i in 0..c { wdata[i * kh * kw + kh * kw / 2] = 1.0; }
        // OC=1 picks differently
        for i in 0..c { wdata[k + i * kh * kw + kh * kw / 2] = 0.5; }

        let weight = PackedTensor::<U4x8>::from_f32_per_channel(&wdata, &[oc, k]);
        let h_out = h; // stride=1, pad=1, 4x4→4x4
        let w_out = w;
        let mut output = vec![0.0f32; oc * h_out * w_out];
        unsafe {
            conv2d_packed_u4x8(
                &input, n, c, h, w, &weight, None,
                stride, padding, dilation, groups, kh, kw, None, &mut output,
            );
        }
        // Output should not contain NaN or Inf — verifies row packing alignment
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite value at output[{}]: {}", i, v);
        }
        // Output should be non-trivial (not all zeros)
        let max_abs = output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.01, "Output is all zeros (max_abs={})", max_abs);
    }

    /// Test conv2d U8 with K not a multiple of 4 (e.g. C=3, KH=3, KW=3 → K=27).
    #[test]
    fn test_conv2d_packed_u8x4_non_multiple_k() {
        let n = 1; let c = 3; let h = 4; let w = 4;
        let kh = 3; let kw = 3;
        let stride = 1; let padding = 1; let dilation = 1;
        let oc = 2; let groups = 1;

        let input: Vec<f32> = (0..(c * h * w)).map(|i| (i as f32 - 24.0) * 0.1).collect();
        let k = c * kh * kw; // 27
        let mut wdata = vec![0.0f32; oc * k];
        for i in 0..c { wdata[i * kh * kw + kh * kw / 2] = 1.0; }
        for i in 0..c { wdata[k + i * kh * kw + kh * kw / 2] = 0.5; }

        let weight = PackedTensor::<U8x4>::from_f32_per_channel(&wdata, &[oc, k]);
        let h_out = h;
        let w_out = w;
        let mut output = vec![0.0f32; oc * h_out * w_out];
        unsafe {
            conv2d_packed_u8x4(
                &input, n, c, h, w, &weight, None,
                stride, padding, dilation, groups, kh, kw, None, &mut output,
            );
        }
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite value at output[{}]: {}", i, v);
        }
        let max_abs = output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.01, "Output is all zeros (max_abs={})", max_abs);
    }
}
