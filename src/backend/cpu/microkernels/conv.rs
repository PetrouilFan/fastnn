//! CPU convolution microkernels â€” extracted from microkernels.rs

#![allow(dead_code, unused_imports)]

use super::*;
use crate::dtypes::{F32x1, I4x8, I8x4, PackedWord};
use crate::packed_tensor::PackedTensor;
use std::sync::OnceLock;

thread_local! {
    static CONV_COL_BUF: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
    static CONV_WEIGHT_T_BUF: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
    static CONV_TEMP_OUT_BUF: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

macro_rules! get_conv_buf {
    ($buf:expr, $size:expr) => {{
        let size: usize = $size;
        (*$buf).with(|b| {
            let mut b = b.borrow_mut();
            if b.len() < size {
                b.resize(size, 0.0);
            }
            unsafe {
                std::mem::transmute::<
                    std::cell::RefMut<'_, Vec<f32>>,
                    std::cell::RefMut<'static, Vec<f32>>,
                >(b)
            }
        })
    }};
}

// ============================================================
// Conv2d im2col kernel
// ============================================================

thread_local! {
    static CONV_SCRATCH: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

// SAFETY: Caller must ensure all pointer arguments are valid, non-overlapping,
// and point to allocations of sufficient size for the convolution parameters.
pub unsafe fn conv2d_im2col(
    x_ptr: *const f32,
    w_data: &[f32],
    bias_data: Option<&[f32]>,
    out_ptr: *mut f32,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    in_height: usize,
    in_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) {
    let out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    let out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    let col_rows = batch_size * out_height * out_width;
    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;
    let cols_per_group = in_channels_per_group * kernel_height * kernel_width;
    let col_size = col_rows * cols_per_group;

    for group in 0..groups {
        let in_ch_start = group * in_channels_per_group;
        let in_ch_end = in_ch_start + in_channels_per_group;

        CONV_SCRATCH.with(|scratch| {
            let mut buf = scratch.borrow_mut();
            if buf.len() < col_size {
                buf.resize(col_size, 0.0f32);
            }
            buf[..col_size].fill(0.0);

            let col_data: &mut [f32] = &mut buf[..col_size];

            for n in 0..batch_size {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let col_row = (n * out_height + oh) * out_width + ow;
                        let col_chunk =
                            &mut col_data[col_row * cols_per_group..(col_row + 1) * cols_per_group];

                        let fast_path = stride == 1 && dilation == 1;

                        for ic_idx in in_ch_start..in_ch_end {
                            let ic = ic_idx;
                            let col_base = (ic_idx - in_ch_start) * kernel_height * kernel_width;

                            if fast_path {
                                for kh in 0..kernel_height {
                                    let ih = oh + kh;
                                    if ih >= padding && ih < padding + in_height {
                                        let x_ih = ih - padding;
                                        let x_row_base =
                                            (n * in_channels + ic) * in_height * in_width
                                                + x_ih * in_width;

                                        let iw_start = ow;
                                        if iw_start >= padding
                                            && iw_start + kernel_width <= padding + in_width
                                        {
                                            let x_iw_start = iw_start - padding;
                                            let x_src = x_row_base + x_iw_start;
                                            let col_dst_base = col_base + kh * kernel_width;

                                            std::ptr::copy_nonoverlapping(
                                                x_ptr.add(x_src),
                                                col_chunk.as_mut_ptr().add(col_dst_base),
                                                kernel_width,
                                            );
                                        } else {
                                            for kw in 0..kernel_width {
                                                let iw = ow + kw;
                                                if iw >= padding && iw < padding + in_width {
                                                    let x_iw = iw - padding;
                                                    let col_col = col_base + kh * kernel_width + kw;
                                                    col_chunk[col_col] =
                                                        *x_ptr.add(x_row_base + x_iw);
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                for kh in 0..kernel_height {
                                    let ih = (oh * stride + kh * dilation).wrapping_sub(padding);
                                    if ih < in_height {
                                        let x_row_base =
                                            (n * in_channels + ic) * in_height * in_width
                                                + ih * in_width;
                                        for kw in 0..kernel_width {
                                            let iw =
                                                (ow * stride + kw * dilation).wrapping_sub(padding);
                                            if iw < in_width {
                                                let col_col = col_base + kh * kernel_width + kw;
                                                col_chunk[col_col] = *x_ptr.add(x_row_base + iw);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let col_slice: &[f32] = col_data;
            let oc_start = group * out_channels_per_group;
            let w_group_start = oc_start * cols_per_group;

            let mut group_out = vec![0.0f32; col_rows * out_channels_per_group];

            matmul_blas_with_transpose_into(
                &col_slice[..col_rows * cols_per_group],
                &w_data[w_group_start..],
                &mut group_out,
                col_rows,
                cols_per_group,
                out_channels_per_group,
                false,
                true,
            );

            let spatial = out_height * out_width;
            let block_rows = 64;

            if let Some(bias) = bias_data {
                let bias_ptr = bias.as_ptr();
                for n in 0..batch_size {
                    for sp_block in (0..spatial).step_by(block_rows) {
                        let blk = std::cmp::min(block_rows, spatial - sp_block);
                        for oc_idx_in_group in 0..out_channels_per_group {
                            let oc_idx = oc_start + oc_idx_in_group;
                            let bias_val = *bias_ptr.add(oc_idx);
                            for i in 0..blk {
                                let row = n * spatial + sp_block + i;
                                let out_idx =
                                    ((n * out_channels + oc_idx) * spatial + sp_block) + i;
                                let val = group_out[row * out_channels_per_group + oc_idx_in_group];
                                *out_ptr.add(out_idx) = val + bias_val;
                            }
                        }
                    }
                }
            } else {
                for n in 0..batch_size {
                    for sp_block in (0..spatial).step_by(block_rows) {
                        let blk = std::cmp::min(block_rows, spatial - sp_block);
                        for oc_idx_in_group in 0..out_channels_per_group {
                            let oc_idx = oc_start + oc_idx_in_group;
                            for i in 0..blk {
                                let row = n * spatial + sp_block + i;
                                let out_idx =
                                    ((n * out_channels + oc_idx) * spatial + sp_block) + i;
                                let val = group_out[row * out_channels_per_group + oc_idx_in_group];
                                *out_ptr.add(out_idx) = val;
                            }
                        }
                    }
                }
            }
        });
    }
}

// ============================================================
// Conv2d f32 tiled microkernel (cache-blocked SIMD-friendly)
// ============================================================

/// Tiled f32 conv2d with OCÃ—2Ã—2 register blocking.
///
/// Tiles over output channels (OC_TILE=4) and output positions (2Ã—2)
/// to keep filter weights and partial sums in registers. The inner
/// (cc, kh, kw) loop broadcasts the input value and does 4 FMAs,
/// which the compiler auto-vectorizes.
pub fn conv2d_f32_tiled(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) {
    assert!(
        bias.is_empty() || bias.len() == f,
        "conv2d bias must have one entry per output channel"
    );
    let c_per_group = c / groups.max(1);
    let f_per_group = f / groups.max(1);
    let h_out =
        (h + 2 * padding).saturating_sub(dilation * (kh.saturating_sub(1)) + 1) / stride + 1;
    let w_out =
        (w + 2 * padding).saturating_sub(dilation * (kw.saturating_sub(1)) + 1) / stride + 1;

    const OC_TILE: usize = 4;

    for nn in 0..n {
        for g in 0..groups {
            let ff_base = g * f_per_group;
            let input_group_off = g * c_per_group * (h * w);

            for ff_tile in (0..f_per_group).step_by(OC_TILE) {
                let ff_abs = ff_base + ff_tile;
                let oc_end = (ff_tile + OC_TILE).min(f_per_group);
                let oc_tile = oc_end - ff_tile;

                let weight_off = ff_abs * c_per_group * kh * kw;

                for hh in (0..h_out).step_by(2) {
                    for ww in (0..w_out).step_by(2) {
                        let mut acc = [[0.0f32; 4]; 4];

                        for cc in 0..c_per_group {
                            let input_ch_off = nn * (c * h * w) + input_group_off + cc * (h * w);
                            let weight_ch_off = weight_off + cc * kh * kw;

                            for kkh in 0..kh {
                                for kkw in 0..kw {
                                    let weight_val_base = weight_ch_off + kkh * kw + kkw;

                                    for pos_h in 0..2 {
                                        let oh = hh + pos_h;
                                        if oh >= h_out {
                                            continue;
                                        }
                                        for pos_w in 0..2 {
                                            let ow = ww + pos_w;
                                            if ow >= w_out {
                                                continue;
                                            }

                                            let h_in = oh * stride + kkh * dilation;
                                            let w_in = ow * stride + kkw * dilation;
                                            if h_in < padding || w_in < padding {
                                                continue;
                                            }
                                            let h_in_s = h_in - padding;
                                            let w_in_s = w_in - padding;
                                            if h_in_s >= h || w_in_s >= w {
                                                continue;
                                            }

                                            let input_val =
                                                input[input_ch_off + h_in_s * w + w_in_s];
                                            let pos = pos_h * 2 + pos_w;

                                            for oc in 0..oc_tile {
                                                acc[pos][oc] += input_val
                                                    * weight[weight_val_base
                                                        + oc * c_per_group * kh * kw];
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        for pos_h in 0..2 {
                            let oh = hh + pos_h;
                            if oh >= h_out {
                                continue;
                            }
                            for pos_w in 0..2 {
                                let ow = ww + pos_w;
                                if ow >= w_out {
                                    continue;
                                }
                                let pos = pos_h * 2 + pos_w;
                                let out_base = nn * (f * h_out * w_out) + (oh * w_out + ow);
                                for oc in 0..oc_tile {
                                    let mut val = acc[pos][oc];
                                    if !bias.is_empty() {
                                        val += bias[ff_abs + oc];
                                    }
                                    output[out_base + (ff_abs + oc) * (h_out * w_out)] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
// im2col + GEMM based conv2d
// ============================================================

/// Optional fused activation for conv2d kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvActivation {
    Relu,
    Gelu,
    Silu,
}

#[inline]
pub(crate) fn apply_conv_activation(x: f32, activation: ConvActivation) -> f32 {
    match activation {
        ConvActivation::Relu => x.max(0.0),
        ConvActivation::Gelu => {
            let x3 = x * x * x;
            0.5 * x * (1.0 + (0.797_884_6_f32 * (x + 0.044_715_f32 * x3)).tanh())
        }
        ConvActivation::Silu => x / (1.0 + (-x).exp()),
    }
}

#[cfg(feature = "openblas")]
#[link(name = "openblas")]
extern "C" {
    fn cblas_sgemm(
        layout: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(feature = "openblas")]
#[inline]
pub fn openblas_conv_gemm_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FASTNN_DISABLE_OPENBLAS_CONV_GEMM")
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                !matches!(value.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(true)
    })
}

#[cfg(feature = "openblas")]
#[inline]
// SAFETY: Caller must ensure all pointer arguments are valid, non-overlapping,
// and point to allocations of sufficient size for the matrix dimensions (m, k, n).
pub(crate) unsafe fn conv_sgemm(
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
) {
    if openblas_conv_gemm_enabled()
        && rs_a == k as isize
        && cs_a == 1
        && rs_c == n as isize
        && cs_c == 1
        && m <= i32::MAX as usize
        && n <= i32::MAX as usize
        && k <= i32::MAX as usize
    {
        const CBLAS_ROW_MAJOR: i32 = 101;
        const CBLAS_NO_TRANS: i32 = 111;
        const CBLAS_TRANS: i32 = 112;
        if rs_b == n as isize && cs_b == 1 {
            // 1x1 Conv fast path: input is already B=[K,N] row-major.
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a,
                k as i32,
                b,
                n as i32,
                0.0,
                c,
                n as i32,
            );
        } else if rs_b == 1 && cs_b == k as isize {
            // General Conv im2col stores Col as [spatial, K] row-major. The
            // GEMM view is B=[K,spatial] with strides rs_b=1, cs_b=K, i.e.
            // row-major B^T. Use CBLAS TransB to consume without repacking.
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a,
                k as i32,
                b,
                k as i32,
                0.0,
                c,
                n as i32,
            );
        } else {
            crate::backend::cpu::sgemm::sgemm(
                m, k, n, 1.0, a, rs_a, cs_a, b, rs_b, cs_b, 0.0, c, rs_c, cs_c,
            );
        }
    } else {
        crate::backend::cpu::sgemm::sgemm(
            m, k, n, 1.0, a, rs_a, cs_a, b, rs_b, cs_b, 0.0, c, rs_c, cs_c,
        );
    }
}

#[cfg(not(feature = "openblas"))]
#[inline]
// SAFETY: Same as the openblas conv_sgemm â€” caller ensures valid, non-overlapping
// pointers with correct dimensions.
pub(crate) unsafe fn conv_sgemm(
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
) {
    crate::backend::cpu::sgemm::sgemm(
        m, k, n, 1.0, a, rs_a, cs_a, b, rs_b, cs_b, 0.0, c, rs_c, cs_c,
    );
}

/// Optimized conv2d using im2col transformation + sgemm.
/// Fast path for 1x1 convolutions (no im2col needed).
pub fn conv2d_f32_im2col_gemm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    activation: Option<ConvActivation>,
) {
    let c_per_group = c / groups.max(1);
    let f_per_group = f / groups.max(1);
    // ONNX defines an empty output when the dilated kernel does not fit in the
    // padded input. `saturating_sub` used to turn that case into one output
    // row, so im2col wrote nothing and the GEMM consumed stale bytes from the
    // thread-local column buffer. Compute zero and stop before either happens.
    let effective_kh = dilation * (kh.saturating_sub(1)) + 1;
    let effective_kw = dilation * (kw.saturating_sub(1)) + 1;
    let h_out = (h + 2 * padding)
        .checked_sub(effective_kh)
        .map_or(0, |span| span / stride + 1);
    let w_out = (w + 2 * padding)
        .checked_sub(effective_kw)
        .map_or(0, |span| span / stride + 1);
    if h_out == 0 || w_out == 0 {
        output.fill(0.0);
        return;
    }
    let spatial_size = h_out * w_out;

    // Small tensor fallback to avoid im2col overhead
    if spatial_size * f < 64 {
        conv2d_f32_tiled(
            input, weight, bias, output, n, c, h, w, f, kh, kw, stride, padding, dilation, groups,
        );
        // Apply activation separately for tiled path
        if let Some(act) = activation {
            for x in output.iter_mut() {
                *x = apply_conv_activation(*x, act);
            }
        }
        return;
    }

    // Fast path for 1x1 convolutions: GEMM directly on the NCHW views
    // (no im2col expansion needed).
    if kh == 1 && kw == 1 && stride == 1 && padding == 0 && dilation == 1 {
        return crate::backend::cpu::microkernels::direct_conv::direct_conv1x1_f32(
            input, weight, bias, output, n, c, h, w, f, groups, activation,
        );
    }

    // Fast path for depthwise 3×3 stride-1 pad-1 direct convolution
    // (avoids 9× im2col expansion for mobileNet/YOLO depthwise convs).
    if kh == 3
        && kw == 3
        && stride == 1
        && padding == 1
        && dilation == 1
        && c_per_group == 1
        && f_per_group == 1
    {
        return crate::backend::cpu::microkernels::direct_depthwise_conv::direct_depthwise_conv3x3_f32(
            input, weight, bias, output,
            n, c, h, w, f, groups, stride, padding, activation,
        );
    }

    // General case: im2col + GEMM (for non-1x1 convolutions).
    // Also used for 3×3 s=1 p=1 g=1 — matrixmultiply's tiled GEMM
    // (~90 GFLOP/s) outperforms direct_conv3x3 (~25 GFLOP/s).
    // im2col is [spatial, col_w] row-major. Run GEMM as
    // C[f, spatial] = W[f, col_w] * Col[col_w, spatial], with
    // Col[col, spatial] = col_matrix[spatial * col_w + col]
    // (rs_b=1, cs_b=col_w), so C lands directly in NCHW output order.
    for g in 0..groups {
        let f_start = g * f_per_group;
        let input_group_off = g * c_per_group * (h * w);

        let col_w = c_per_group * kh * kw;
        let num_pixels = n * spatial_size;
        let mut col_matrix = get_conv_buf!(&CONV_COL_BUF, num_pixels * col_w);

        for nn in 0..n {
            let col_start = nn * spatial_size * col_w;
            // SAFETY: slice pointers and offsets are valid per the tensor dimensions;
            // `im2col_dispatch` only writes within the provided slice bounds.
            unsafe {
                crate::backend::cpu::im2col::im2col_dispatch(
                    &input[nn * (c * h * w) + input_group_off..],
                    c_per_group,
                    h,
                    w,
                    kh,
                    kw,
                    stride,
                    padding,
                    dilation,
                    &mut col_matrix[col_start..],
                    false, // FP32 conv does not need min/max
                );
            }
        }

        let weight_off = f_start * col_w;
        let group_out_off = g * f_per_group * spatial_size;
        let batch_stride = f * spatial_size;

        for nn in 0..n {
            let col_start = nn * spatial_size * col_w;
            if !bias.is_empty() || activation.is_some() {
                let group_bias = if !bias.is_empty() {
                    Some(&bias[f_start..f_start + f_per_group])
                } else {
                    None
                };
                // SAFETY: all pointer offsets stay within their allocated slices
                unsafe {
                    crate::backend::cpu::microkernels::conv_gemm::conv_gemm_f32(
                        f_per_group,
                        col_w,
                        spatial_size,
                        weight.as_ptr().add(weight_off),
                        col_w as isize,
                        1isize,
                        col_matrix.as_ptr().add(col_start),
                        1isize,
                        col_w as isize,
                        output.as_mut_ptr().add(group_out_off + nn * batch_stride),
                        spatial_size as isize,
                        1isize,
                        group_bias,
                        activation,
                    );
                }
            } else {
                // SAFETY: slice pointers and strides match the GEMM dimensions
                unsafe {
                    conv_sgemm(
                        f_per_group,
                        col_w,
                        spatial_size,
                        weight.as_ptr().add(weight_off),
                        col_w as isize,
                        1isize,
                        col_matrix.as_ptr().add(col_start),
                        1isize,
                        col_w as isize,
                        output.as_mut_ptr().add(group_out_off + nn * batch_stride),
                        spatial_size as isize,
                        1isize,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_uses_the_tanh_approximation_the_avx2_paths_use() {
        for x in [-3.0f32, -1.0, -0.25, 0.0, 0.5, 1.0, 2.5] {
            let expected = 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh());
            assert_eq!(apply_conv_activation(x, ConvActivation::Gelu), expected);
        }
    }

    #[test]
    fn a_kernel_larger_than_the_padded_input_clears_the_output() {
        // 2x2 input, 3x3 kernel, no padding: the output is empty by ONNX
        // geometry. The caller's one-element buffer must not keep an earlier
        // convolution's bytes, which is what the saturating formula caused.
        let input = [1.0f32; 4];
        let weight = [1.0f32; 9];
        let mut output = [7.0f32];
        conv2d_f32_im2col_gemm(
            &input,
            &weight,
            &[],
            &mut output,
            1,
            1,
            2,
            2,
            1,
            3,
            3,
            1,
            0,
            1,
            1,
            None,
        );
        assert_eq!(output, [0.0]);
    }

    #[test]
    fn stride_two_odd_input_matches_a_naive_convolution() {
        // 17x17 with 3x3, stride 2, pad 1: output 9x9 (81 elements), above the
        // small-tensor fallback, so this runs the stride-2 im2col path that
        // used to store the wrong lanes.
        const H: usize = 17;
        const W: usize = 17;
        const K: usize = 3;
        const OUT: usize = 9;
        let input: Vec<f32> = (0..H * W).map(|i| ((i as f32) * 0.37).sin()).collect();
        let weight: Vec<f32> = (0..K * K).map(|i| 0.1 + i as f32 * 0.01).collect();
        let bias = [0.25f32];
        let mut output = vec![0.0f32; OUT * OUT];
        conv2d_f32_im2col_gemm(
            &input,
            &weight,
            &bias,
            &mut output,
            1,
            1,
            H,
            W,
            1,
            K,
            K,
            2,
            1,
            1,
            1,
            None,
        );

        for oy in 0..OUT {
            for ox in 0..OUT {
                let mut expected = bias[0];
                for ky in 0..K {
                    let iy = oy as isize * 2 + ky as isize - 1;
                    if !(0..H as isize).contains(&iy) {
                        continue;
                    }
                    for kx in 0..K {
                        let ix = ox as isize * 2 + kx as isize - 1;
                        if !(0..W as isize).contains(&ix) {
                            continue;
                        }
                        expected += input[iy as usize * W + ix as usize] * weight[ky * K + kx];
                    }
                }
                let actual = output[oy * OUT + ox];
                assert!(
                    (actual - expected).abs() <= 1e-5,
                    "at ({oy},{ox}): expected {expected}, got {actual}"
                );
            }
        }
    }
}
