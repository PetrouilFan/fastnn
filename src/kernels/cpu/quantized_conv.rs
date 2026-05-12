//! Quantized convolution kernels for packed precision types.
//!
//! Architecture
//! ============
//! These kernels perform conv2d using weights stored in a packed format
//! (U8x4, U4x8, or F16x2 — controlled by the `T: PackedWord` type parameter).
//! Activations are kept in f32 (weight-only quantization), which avoids
//! calibration overhead and preserves accuracy.
//!
//! Dispatch strategy (mirrors f32 `conv2d_kernel` in `conv.rs`):
//!
//!   quantized_conv2d_kernel<T>()
//!     ├── groups==in_ch==out_ch → depthwise_quantized_conv2d()
//!     ├── 1×1, stride=1, pad=0, groups=1 → quantized_conv2d_1x1()
//!     ├── 3×3, stride∈{1,2}, pad=1, groups=1 → quantized_conv2d_3x3()
//!     └── (all other cases) → quantized_conv2d_im2col()
//!
//! Data flow (minimal dequantization)
//! ==================================
//! Weights stay packed in memory. Inside each kernel, packed weight words
//! are dequantized ON-THE-FLY using SIMD (e.g. `_mm256_cvtepi8_epi32` for U8).
//! The dequantized f32 weight is immediately consumed by an FMA with the
//! f32 activation. There is NO bulk dequantization pass and NO f32 copy
//! of the full weight matrix.
//!
//! For the im2col path, activation patches are gathered into f32 column
//! buffers (same as the f32 conv im2col), and then fed to the batched
//! packed GEMM (`gemm_batch_packed_simd`).  The GEMM tiles K, unpacks
//! each weight row once per tile, and reuses the unpacked tile across
//! all batch positions.
//!
//! Multi-bit support
//! =================
//! All kernels are generic over `T: PackedWord`:
//!   - `T = U8x4`  → 8-bit integer weights, 4× memory compression
//!   - `T = U4x8`  → 4-bit integer weights, 8× compression
//!   - `T = F16x2` → 16-bit float weights, 2× compression
//! The SIMD GEMV kernels in `backends/packed_simd.rs` handle the
//! type-specific dequantization (int8→f32, nibble→f32, f16→f32).
//!
//! Future: full integer path (VNNI `_mm256_dpbusd_epi32`)
//! These kernels support f32 activations + packed weights.  A future
//! enhancement can add activation quantization + VNNI-based integer
//! dot products for zero-dequantization inference.

use crate::backends::cpu::gemm_batch_packed;
use crate::backends::TlsVecPool;
use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;
use crate::tensor::Tensor;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compute_out_dim(in_dim: usize, kernel: usize, stride: usize, padding: usize, dilation: usize) -> usize {
    let dk = (kernel - 1) * dilation + 1;
    if in_dim + 2 * padding < dk {
        0
    } else {
        (in_dim + 2 * padding - dk) / stride + 1
    }
}

// ---------------------------------------------------------------------------
// Quantized 1×1 conv
// ---------------------------------------------------------------------------

/// Quantized 1×1 convolution.
///
/// Treats conv as a batched matrix multiply: input [N*H*W, C] × weight [OC, C].
/// The weight is packed; dequantization happens inside `gemm_batch_packed`.
///
/// # Arguments
/// - `x`: f32 input tensor [N, C, H, W]
/// - `weight`: packed weight matrix [OC, C]
/// - `bias`: optional f32 bias [OC]
/// - `n, c, h, w, oc`: dimension sizes
pub unsafe fn quantized_conv2d_1x1<T: PackedWord>(
    x: &Tensor,
    weight: &PackedTensor<T>,
    bias: Option<&[f32]>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    oc: usize,
) -> Tensor {
    let spatial = n * h * w;
    let x_data = x.as_f32_slice();
    let mut output = TlsVecPool::alloc_zeroed(spatial * oc);

    // Transpose activations from NCHW to NHWC for the GEMM.
    // x_data is NCHW flat:  x[n, c, h, w] at n*C*H*W + c*H*W + h*W + w
    // gemm_batch_packed expects NHWC row-major: x[n, h, w, c] at (n*H*W + h*W + w)*C + c
    // This is only needed for 1×1; the 3×3 and im2col paths build their own column buffers.
    let mut x_nhwc = TlsVecPool::alloc_zeroed(spatial * c);
    for n_idx in 0..n {
        let nchw_batch = n_idx * c * h * w;
        let nhwc_batch = n_idx * h * w * c;
        for oy in 0..h {
            for ox in 0..w {
                let nhwc_spatial_base = nhwc_batch + (oy * w + ox) * c;
                let nchw_spatial_base = nchw_batch + oy * w + ox;
                for c_idx in 0..c {
                    x_nhwc[nhwc_spatial_base + c_idx] =
                        x_data[nchw_spatial_base + c_idx * h * w];
                }
            }
        }
    }

    // Batched packed GEMM: weight [OC, C] × act [spatial, C] → out [spatial, OC]
    // This does on-the-fly SIMD dequantization with K-tiling and per-row scale/zero.
    gemm_batch_packed(weight, &x_nhwc, &mut output, spatial, c, oc);

    // output is [spatial, OC] in NHWC-interleaved layout from gemm_batch_packed.
    // Permute to NCHW and add bias in one fused pass.
    let mut nchw = TlsVecPool::alloc_zeroed(n * oc * h * w);
    for n_idx in 0..n {
        for oc_idx in 0..oc {
            let b = bias.map(|b| b[oc_idx]).unwrap_or(0.0);
            let nchw_base = n_idx * oc * h * w + oc_idx * h * w;
            for oy in 0..h {
                for ox in 0..w {
                    let nhwc = (n_idx * h * w + oy * w + ox) * oc + oc_idx;
                    let nchw_idx = nchw_base + oy * w + ox;
                    nchw[nchw_idx] = output[nhwc] + b;
                }
            }
        }
    }
    Tensor::from_vec(nchw.take(), vec![n as i64, oc as i64, h as i64, w as i64])
}

// ---------------------------------------------------------------------------
// Quantized 3×3 direct conv
// ---------------------------------------------------------------------------

/// Quantized 3×3 convolution using im2col + batched packed GEMM.
///
/// For each output position, the 3×3 input patch (per input channel) is
/// gathered into a column buffer (im2col), then `gemm_batch_packed` computes
/// the dot product with the packed weight matrix in one batched call.
///
/// TODO: Replace im2col with a true direct kernel that processes 8 OC at
/// once using the same S_BUF + WT_TRANS_BUF pattern as `conv2d_3x3_direct`.
/// That will eliminate the temporary im2col buffer allocation.
pub unsafe fn quantized_conv2d_3x3<T: PackedWord>(
    x: &Tensor,
    weight: &PackedTensor<T>,
    bias: Option<&[f32]>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    oc: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
) -> Tensor {
    let k = c * 9; // inner dimension for 3×3
    let spatial = n * oh * ow;
    let x_data = x.as_f32_slice();
    let mut output = TlsVecPool::alloc(spatial * oc);

    // Build im2col column matrix: [spatial, k]
    // Each row is the flattened 3×3 patch for one output position.
    // Every element is written by the im2col loop below (no zero-init needed).
    let mut col = TlsVecPool::alloc(spatial * k);

    for batch in 0..n {
        for oy in 0..oh {
            for ox in 0..ow {
                let in_y = (oy as i64 * stride as i64) - pad as i64;
                let in_x = (ox as i64 * stride as i64) - pad as i64;
                let col_row = (batch * oh * ow + oy * ow + ox) * k;

                // Gather 3×3 patch for each input channel
                for ic in 0..c {
                    let ch_base = batch * (c * h * w) + ic * h * w;
                    let out_base = col_row + ic * 9;
                    for ky in 0..3i64 {
                        for kx in 0..3i64 {
                            let py = in_y + ky;
                            let px = in_x + kx;
                            let val = if py >= 0 && py < h as i64 && px >= 0 && px < w as i64 {
                                x_data[ch_base + (py as usize) * w + (px as usize)]
                            } else {
                                0.0
                            };
                            col[out_base + (ky as usize) * 3 + (kx as usize)] = val;
                        }
                    }
                }
            }
        }
    }

    // Batched packed GEMM: weight [OC, k] × col [spatial, k] → out [spatial, OC]
    gemm_batch_packed(weight, &col, &mut output, spatial, k, oc);

    // output is [spatial, OC] in NHWC-interleaved layout from gemm_batch_packed.
    // Permute to NCHW and add bias in one fused pass.
    let mut nchw = TlsVecPool::alloc_zeroed(n * oc * oh * ow);
    for n_idx in 0..n {
        for oc_idx in 0..oc {
            let b = bias.map(|b| b[oc_idx]).unwrap_or(0.0);
            let nchw_base = n_idx * oc * oh * ow + oc_idx * oh * ow;
            for oy in 0..oh {
                for ox in 0..ow {
                    let nhwc = (n_idx * oh * ow + oy * ow + ox) * oc + oc_idx;
                    let nchw_idx = nchw_base + oy * ow + ox;
                    nchw[nchw_idx] = output[nhwc] + b;
                }
            }
        }
    }
    Tensor::from_vec(nchw.take(), vec![n as i64, oc as i64, oh as i64, ow as i64])
}

// ---------------------------------------------------------------------------
// Quantized depthwise conv
// ---------------------------------------------------------------------------

/// Quantized depthwise convolution (groups == in_channels == out_channels).
/// Each input channel is convolved with its own kernel (1 output channel).
pub unsafe fn quantized_conv2d_depthwise<T: PackedWord>(
    x: &Tensor,
    weight: &PackedTensor<T>,
    bias: Option<&[f32]>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kernel_size: usize,
    stride: usize,
    pad: usize,
    _dilation: usize,
    oh: usize,
    ow: usize,
) -> Tensor {
    let x_data = x.as_f32_slice();
    let oc = c; // depthwise: out_channels == in_channels
    let spatial = n * oh * ow;
    let mut output = TlsVecPool::alloc(spatial * oc);

    // For depthwise, each output channel depends on exactly one input channel.
    // The weight shape is [C, 1*kh*kw] (packed per-row).
    // We process each channel independently.
    let kh = kernel_size;
    let kw = kernel_size;
    let k = kh * kw; // elements per kernel, NOT multiplied by C

    // We need to re-interpret the packed weight: it's [C, 1*kh*kw] per-row.
    // The existing PackedTensor's shape might be [C, kh*kw] which is perfect.
    // gemm_batch_packed expects weight [M, K] × act [N, K] → out [N, M].
    // For depthwise, M=C, K=kh*kw, and each output channel uses a different
    // activation patch.  We can't use a single GEMM call because each output
    // row needs a different activation row.
    //
    // Instead, we process per output position, per channel:
    for batch in 0..n {
        for oy in 0..oh {
            for ox in 0..ow {
                let in_y = (oy as i64 * stride as i64) - pad as i64;
                let in_x = (ox as i64 * stride as i64) - pad as i64;

                for ch in 0..c {
                    // Build activation patch for this (channel, output position)
                    let mut patch = [0.0f32; 9]; // max kernel_size=3
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let py = in_y + ky as i64;
                            let px = in_x + kx as i64;
                            let val = if py >= 0 && py < h as i64 && px >= 0 && px < w as i64 {
                                x_data[batch * (c * h * w) + ch * (h * w)
                                    + (py as usize) * w + (px as usize)]
                            } else {
                                0.0
                            };
                            patch[ky * kw + kx] = val;
                        }
                    }

                    // Dot product: weight row ch with patch
                    // For depthwise, weight row ch has k elements packed.
                    let mut dot = 0.0f32;
                    let wp = weight.as_packed();
                    if kh == 3 && kw == 3 && T::ITEMS >= 8 {
                        // Unrolled for 3×3 kernel (k=9, 9.div_ceil(8) = 2 words):
                        let row_offset = ch * 2;
                        let w0 = wp[row_offset].unpack_to_f32();
                        let w0_arr = w0.as_ref();
                        dot += patch[0] * w0_arr[0]
                            + patch[1] * w0_arr[1]
                            + patch[2] * w0_arr[2]
                            + patch[3] * w0_arr[3]
                            + patch[4] * w0_arr[4]
                            + patch[5] * w0_arr[5]
                            + patch[6] * w0_arr[6]
                            + patch[7] * w0_arr[7];
                        let w1 = wp[row_offset + 1].unpack_to_f32();
                        let w1_arr = w1.as_ref();
                        dot += patch[8] * w1_arr[0];
                    } else {
                        let row_offset = ch * k.div_ceil(T::ITEMS);
                        for p in 0..k.div_ceil(T::ITEMS) {
                            let word = wp[row_offset + p];
                            let unpacked = word.unpack_to_f32();
                            for j in 0..T::ITEMS {
                                let idx = p * T::ITEMS + j;
                                if idx < k {
                                    dot += unpacked.as_ref()[j] * patch[idx];
                                }
                            }
                        }
                    }

                    let scale = weight.scale_for_row(ch);
                    let zero = weight.zero_for_row(ch);
                    let nchw_idx = batch * oc * oh * ow + ch * oh * ow + oy * ow + ox;
                    output[nchw_idx] = dot * scale + zero;
                }
            }
        }
    }

    // Add bias (output is NCHW)
    if let Some(bias) = bias {
        for n_idx in 0..n {
            for oc_idx in 0..oc {
                let b = bias[oc_idx];
                for oy in 0..oh {
                    for ox in 0..ow {
                        let idx = n_idx * oc * oh * ow + oc_idx * oh * ow + oy * ow + ox;
                        output[idx] += b;
                    }
                }
            }
        }
    }

    Tensor::from_vec(output.take(), vec![n as i64, oc as i64, oh as i64, ow as i64])
}

// ---------------------------------------------------------------------------
// General quantized conv: im2col fallback
// ---------------------------------------------------------------------------

/// General quantized convolution via im2col + batched packed GEMM.
///
/// Supports arbitrary kernel sizes, strides, padding, dilation, and groups.
/// For groups > 1 (but not full depthwise), processes each group independently
/// with its own im2col + GEMM call.
pub unsafe fn quantized_conv2d_im2col<T: PackedWord>(
    x: &Tensor,
    weight: &PackedTensor<T>,
    bias: Option<&[f32]>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    oc: usize,
    kernel_size: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    groups: usize,
    oh: usize,
    ow: usize,
) -> Tensor {
    let kh = kernel_size;
    let kw = kernel_size;
    let c_per_group = c / groups;
    let oc_per_group = oc / groups;
    let k_per_group = c_per_group * kh * kw;
    let spatial = n * oh * ow;
    let x_data = x.as_f32_slice();
    let mut output = TlsVecPool::alloc(spatial * oc);

    for g in 0..groups {
        let c_start = g * c_per_group;
        let oc_start = g * oc_per_group;

        // Tiled im2col: process spatial positions in chunks to bound
        // column buffer memory to TILE_SIZE * k_per_group.
        const TILE_SIZE: usize = 256;
        let total_positions = spatial;

        let mut tile_start = 0;
        while tile_start < total_positions {
            let tile_end = (tile_start + TILE_SIZE).min(total_positions);
            let cur_tile = tile_end - tile_start;

            // col_tile is fully written by the im2col loop below (no zero-init needed).
            // Using the pool here means the tile buffer is recycled across iterations.
            let mut col_tile = TlsVecPool::alloc(cur_tile * k_per_group);

            // Fill this tile's im2col rows
            for local_pos in 0..cur_tile {
                let global_pos = tile_start + local_pos;
                let batch = global_pos / (oh * ow);
                let hw_rem = global_pos % (oh * ow);
                let oy = hw_rem / ow;
                let ox = hw_rem % ow;

                let in_y = (oy as i64 * stride as i64) - pad as i64;
                let in_x = (ox as i64 * stride as i64) - pad as i64;
                let col_row_local = local_pos * k_per_group;

                for ic in 0..c_per_group {
                    let ch = c_start + ic;
                    let ch_base = batch * (c * h * w) + ch * h * w;
                    let out_base = col_row_local + ic * kh * kw;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let py = in_y + ky as i64 * dilation as i64;
                            let px = in_x + kx as i64 * dilation as i64;
                            let val = if py >= 0 && py < h as i64 && px >= 0 && px < w as i64 {
                                x_data[ch_base + (py as usize) * w + (px as usize)]
                            } else {
                                0.0
                            };
                            col_tile[out_base + ky * kw + kx] = val;
                        }
                    }
                }
            }

            // Compute dot products for this tile
            for local_pos in 0..cur_tile {
                let global_pos = tile_start + local_pos;
                let batch = global_pos / (oh * ow);
                let hw_rem = global_pos % (oh * ow);
                let oy = hw_rem / ow;
                let ox = hw_rem % ow;

                let col_row = &col_tile[local_pos * k_per_group..(local_pos + 1) * k_per_group];

                for oc_local in 0..oc_per_group {
                    let oc_global = oc_start + oc_local;
                    let weight_row = oc_global;

                    // Dot product of weight row with col_row
                    let mut dot = 0.0f32;
                    let wp = weight.as_packed();
                    let row_offset = weight_row * k_per_group.div_ceil(T::ITEMS);
                    for p in 0..k_per_group.div_ceil(T::ITEMS) {
                        let word = wp[row_offset + p];
                        let unpacked = word.unpack_to_f32();
                        for j in 0..T::ITEMS {
                            let idx = p * T::ITEMS + j;
                            if idx < k_per_group {
                                dot += unpacked.as_ref()[j] * col_row[idx];
                            }
                        }
                    }

                    let scale = weight.scale_for_row(oc_global);
                    let zero = weight.zero_for_row(oc_global);
                    output[batch * oc * oh * ow + oc_global * oh * ow + oy * ow + ox] = dot * scale + zero;
                }
            }

            tile_start += TILE_SIZE;
        }
    }

    // Add bias (output is NCHW)
    if let Some(bias) = bias {
        for n_idx in 0..n {
            for oc_idx in 0..oc {
                let b = bias[oc_idx];
                for oy in 0..oh {
                    for ox in 0..ow {
                        let idx = n_idx * oc * oh * ow + oc_idx * oh * ow + oy * ow + ox;
                        output[idx] += b;
                    }
                }
            }
        }
    }

    Tensor::from_vec(output.take(), vec![n as i64, oc as i64, oh as i64, ow as i64])
}

// ---------------------------------------------------------------------------
// Top-level dispatch
// ---------------------------------------------------------------------------

/// Dispatch a quantized convolution to the most efficient kernel.
///
/// Mirrors the dispatch logic of `conv2d_kernel` in `conv.rs`.
///
/// # Arguments
/// - `x`: f32 input tensor [N, C, H, W]
/// - `weight`: packed weight tensor [OC, C * KH * KW] (per-row packed)
/// - `bias`: optional f32 bias [OC]
/// - `kernel_size`: height/width of the square convolution kernel
/// - `stride, padding, dilation, groups`: conv parameters
///
/// # Returns
/// f32 output tensor [N, OC, OH, OW]
pub unsafe fn dispatch_quantized_conv2d<T: PackedWord>(
    x: &Tensor,
    weight: &PackedTensor<T>,
    bias: Option<&[f32]>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let shape = x.shape_ref();
    let n = shape[0] as usize;
    let c = shape[1] as usize;
    let h = shape[2] as usize;
    let w = shape[3] as usize;

    let pshape = weight.shape();
    let oc = pshape[0];

    let oh = compute_out_dim(h, kernel_size, stride, padding, dilation);
    let ow = compute_out_dim(w, kernel_size, stride, padding, dilation);

    // --- Dispatch ---

    // 1. Depthwise: groups == in_channels == out_channels
    if groups > 1 && groups == c && groups == oc {
        return quantized_conv2d_depthwise(
            x, weight, bias, n, c, h, w, kernel_size, stride, padding, dilation, oh, ow,
        );
    }

    // 2. 1×1 fast path
    if kernel_size == 1 && stride == 1 && padding == 0 && dilation == 1 && groups == 1 {
        return quantized_conv2d_1x1(x, weight, bias, n, c, h, w, oc);
    }

    // 3. 3×3 direct path (im2col + gemm for now)
    if kernel_size == 3 && (stride == 1 || stride == 2) && padding == 1 && dilation == 1 && groups == 1 {
        return quantized_conv2d_3x3(x, weight, bias, n, c, h, w, oc, stride, padding, oh, ow);
    }

    // 4. General im2col fallback
    quantized_conv2d_im2col(
        x, weight, bias, n, c, h, w, oc, kernel_size, stride, padding, dilation, groups, oh, ow,
    )
}
