//! CPU conv kernels.

#![allow(unused_imports)]
#![allow(clippy::missing_safety_doc)]

use super::*;

use crate::autograd::{AutogradMeta, Edge, Node};


use crate::iterator::TensorIterator;

use crate::kernels::blas::{
    matmul_blas, matmul_blas_into, matmul_blas_with_transpose, matmul_blas_with_transpose_into,
    MIN_BLAS_SIZE,
};

use crate::storage::{DType, Device, Storage};

use crate::tensor::Tensor;

use std::sync::Arc;

const GELU_SQRT_2_OVER_PI: f32 = 0.7978846;

const GELU_COEFF: f32 = 0.044715;

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn im2col_kernel(
    x: &Tensor,

    kernel_height: usize,

    kernel_width: usize,

    stride: usize,

    padding: usize,

    dilation: usize,

    out_height: usize,

    out_width: usize,
) -> Tensor {
    let x_shape = x.shape_ref();

    let batch_size = x_shape[0] as usize;

    let in_channels = x_shape[1] as usize;

    let in_height = x_shape[2] as usize;

    let in_width = x_shape[3] as usize;

    let col_rows = batch_size * out_height * out_width;

    let col_cols = in_channels * kernel_height * kernel_width;

    let mut col_data = vec![0.0f32; col_rows * col_cols];

    let x_ptr = x.data_ptr() as *const f32;

    #[cfg(feature = "parallel")]
    {
        if batch_size > 1 {
            use rayon::prelude::*;

            let col_rows_per_batch = out_height * out_width;

            let x_usize = x_ptr as usize;

            let col_usize = col_data.as_mut_ptr() as usize;

            (0..batch_size).into_par_iter().for_each(|n| {
                // SAFETY: Pointer arithmetic stays within bounds of the allocated tensor storage.
                unsafe {
                    let x_p = x_usize as *const f32;

                    let col_p = (col_usize as *mut f32).add(n * col_rows_per_batch * col_cols);

                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let col_row = oh * out_width + ow;

                            for ic in 0..in_channels {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let ih =
                                            (oh * stride + kh * dilation).wrapping_sub(padding);

                                        let iw =
                                            (ow * stride + kw * dilation).wrapping_sub(padding);

                                        // NOTE: ih/iw are in input-coordinate space (wrapping_sub handles
                                        // negative padding positions via usize overflow). The fast path
                                        // (stride=1) uses padded coordinates with `x_ih = ih - padding`,
                                        // but this general path already subtracted padding above.

                                        let col_col =
                                            ((ic * kernel_height) + kh) * kernel_width + kw;

                                        if ih < in_height && iw < in_width {
                                            let x_idx = ((n * in_channels + ic) * in_height + ih)
                                                * in_width
                                                + iw;

                                            *col_p.add(col_row * col_cols + col_col) =
                                                *x_p.add(x_idx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } // end unsafe
            });
        } else {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let col_row = oh * out_width + ow;

                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = (oh * stride + kh * dilation).wrapping_sub(padding);

                                let iw = (ow * stride + kw * dilation).wrapping_sub(padding);

                                let col_col = ((ic * kernel_height) + kh) * kernel_width + kw;

                                if ih < in_height && iw < in_width {
                                    let x_idx = (ic * in_height + ih) * in_width + iw;

                                    col_data[col_row * col_cols + col_col] =

                                        // SAFETY: x_idx computed from n, ic, ih, iw is bounded by x tensor element count.
                                        unsafe { *x_ptr.add(x_idx) };
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        for n in 0..batch_size {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let col_row = (n * out_height + oh) * out_width + ow;

                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = (oh * stride + kh * dilation).wrapping_sub(padding);

                                let iw = (ow * stride + kw * dilation).wrapping_sub(padding);

                                let col_col = ((ic * kernel_height) + kh) * kernel_width + kw;

                                if ih < in_height && iw < in_width {
                                    let x_idx =
                                        ((n * in_channels + ic) * in_height + ih) * in_width + iw;

                                    col_data[col_row * col_cols + col_col] =

                                        // SAFETY: x_idx computed from n, ic, ih, iw is bounded by x tensor element count.
                                        unsafe { *x_ptr.add(x_idx) };
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::from_vec(col_data, vec![col_rows as i64, col_cols as i64])
}

pub unsafe fn conv2d_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    // Ensure contiguous input to handle non-contiguous views (e.g., from Split/Slice ops).
    // All downstream kernels (im2col, 1x1, 3x3 direct, depthwise) assume contiguous
    // layout with storage_offset == 0, and use raw pointer arithmetic via data_ptr().
    // We must NOT use as_f32_slice() for non-contiguous tensors because it returns elements
    // in storage order (ignoring strides). Instead we use contiguous() which properly
    // iterates over logical indices respecting strides.
    let x = args[0].contiguous();

    let w = args[1];

    let bias = if args.len() > 2 && args[2].numel() > 0 {
        Some(args[2])
    } else {
        None
    };

    let stride = if args.len() > 3 {
        args[3].item() as i64
    } else {
        1
    };

    let padding = if args.len() > 4 {
        args[4].item() as i64
    } else {
        0
    };

    let dilation = if args.len() > 5 {
        args[5].item() as i64
    } else {
        1
    };

    let groups = if args.len() > 6 {
        args[6].item() as i64
    } else {
        1
    };

    // Optional 8th arg: pre-transposed weight for 1x1 or WT_TRANS_BUF-layout for 3x3
    let w_pre_t = if args.len() > 7 { Some(args[7]) } else { None };

    let x_shape = x.shape_ref();

    let w_shape = w.shape_ref();

    let batch_size = x_shape[0] as usize;

    let in_channels = x_shape[1] as usize;

    let in_height = x_shape[2] as usize;

    let in_width = x_shape[3] as usize;

    let out_channels = w_shape[0] as usize;

    let kernel_height = w_shape[2] as usize;

    let kernel_width = w_shape[3] as usize;

    let stride = stride as usize;

    let padding = padding as usize;

    let dilation = dilation as usize;

    let groups = groups as usize;

    let out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;

    let out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    if groups > 1 && groups == in_channels && groups == out_channels {
        return vec![depthwise_conv2d(
            &x, w, bias, stride, padding, dilation, out_height, out_width,
        )];
    }

    if kernel_height == 1
        && kernel_width == 1
        && stride == 1
        && padding == 0
        && dilation == 1
        && groups == 1
    {
        return vec![conv2d_1x1(
            &x,
            w,
            w_pre_t,
            bias,
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
        )];
    }

    // Fast path: 3x3 conv (stride=1 or stride=2, padding=1, dilation=1, groups=1)

    if kernel_height == 3
        && kernel_width == 3
        && (stride == 1 || stride == 2)
        && padding == 1
        && dilation == 1
        && groups == 1
    {
        return vec![conv2d_3x3_direct(
            &x,
            w,
            w_pre_t,
            bias,
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
            stride,
            padding,
            out_height,
            out_width,
        )];
    }

    vec![conv2d_im2col(
        &x,
        w,
        bias,
        stride,
        padding,
        dilation,
        out_height,
        out_width,
        batch_size,
        in_channels,
        out_channels,
        kernel_height,
        kernel_width,
        groups,
    )]
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn conv2d_1x1(
    x: &Tensor,

    w: &Tensor,

    w_pre_t: Option<&Tensor>,

    bias: Option<&Tensor>,

    batch_size: usize,

    in_channels: usize,

    out_channels: usize,

    in_height: usize,

    in_width: usize,
) -> Tensor {
    let x_ptr = x.data_ptr() as *const f32;

    let w_ptr = w.data_ptr() as *const f32;

    let output_shape = vec![
        batch_size as i64,
        out_channels as i64,
        in_height as i64,
        in_width as i64,
    ];

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());

    let output_inner = Arc::make_mut(&mut output.inner);

    let output_storage = Arc::make_mut(&mut output_inner.storage);

    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };

    let out_data = Arc::make_mut(&mut cpu_storage.data);

    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let total_spatial = batch_size * in_height * in_width;

    let k = in_channels;

    let m = out_channels;

    let spatial = in_height * in_width; // per-batch spatial size

    // Get pre-transposed weight [IC, OC] or transpose inline

    let use_pre_t = w_pre_t.is_some_and(|wt| wt.numel() as usize == k * m);

    let wt_ptr: *const f32;

    // Temporary vec for inline transpose (only used when !use_pre_t)

    let mut temp_wt: Vec<f32> = Vec::new();

    if use_pre_t {
        let wt = w_pre_t.unwrap();
        wt_ptr = wt.data_ptr() as *const f32;
    } else {
        let w_data = std::slice::from_raw_parts(w_ptr, m * k);
        temp_wt.resize(k * m, 0.0);

        for ic in 0..k {
            for oc in 0..m {
                temp_wt[ic * m + oc] = w_data[oc * k + ic];
            }
        }

        wt_ptr = temp_wt.as_ptr();
    }

    // bias_data from a scalar tensor would read m elements from a 1-element storage (OOB)
    let bias_data: Option<&[f32]> = bias.and_then(|b| {
        if b.numel() > 1 {
            let b_ptr = b.data_ptr() as *const f32;
            Some(unsafe { std::slice::from_raw_parts(b_ptr, m) })
        } else {
            None
        }
    });
    let bias_scalar: Option<f32> =
        bias.and_then(|b| if b.numel() == 1 { Some(b.item()) } else { None });

    let x_usize = x_ptr as usize;

    let out_usize = out_ptr as usize;

    let wt_usize = wt_ptr as usize;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        (0..total_spatial).into_par_iter().for_each(|sp_idx| {

            let n = sp_idx / (in_height * in_width);

            let hw = sp_idx % (in_height * in_width);

            let oh = hw / in_width;

            let ow = hw % in_width;

            let x_ptr = x_usize as *const f32;

            let out_ptr = out_usize as *mut f32;

            let wt_ptr = wt_usize as *const f32;

            // Input: x[n, ic, oh, ow] at stride spatial per channel

            let x_base = n * k * spatial + oh * in_width + ow;

            // Output: out[n, oc, oh, ow] at stride spatial per channel

            let out_base = n * m * spatial + oh * in_width + ow;

            let mut oc = 0;

            // SIMD blocks of 8 (x86_64)

            // Use explicit f32 accumulator arrays to avoid SIMD type alignment issues
            // on worker threads (e.g. rayon thread pool stacks).
            // Perform manual SIMD via `[f32; 8]` arithmetic and let the compiler
            // auto-vectorize — this is more portable and avoids subtle LLVM bugs
            // with `#[repr(align(32))]` types on spawned thread stacks.

            while oc + 8 <= m {

                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let mut acc = [0.0f32; 8];

                    for ic in 0..k {
                        let x_val = unsafe { *x_ptr.add(x_base + ic * spatial) };

                        let w = unsafe {
                            std::slice::from_raw_parts(wt_ptr.add(ic * m + oc), 8)
                        };

                        for lane in 0..8 {
                            acc[lane] += w[lane] * x_val;
                        }
                    }

                    let b = if let Some(bias) = bias_data {
                        unsafe {
                            std::slice::from_raw_parts(bias.as_ptr().add(oc), 8)
                        }
                    } else {
                        &[bias_scalar.unwrap_or(0.0); 8]
                    };

                    for i in 0..8 {
                        unsafe {
                            *out_ptr.add(out_base + (oc + i) * spatial) = acc[i] + b[i];
                        }
                    }
                }

                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                {
                    use wide::f32x4;

                    let mut acc_lo = f32x4::ZERO;

                    let mut acc_hi = f32x4::ZERO;

                    for ic in 0..k {
                        let x_val = unsafe { *x_ptr.add(x_base + ic * spatial) };

                        let w_lo = from_slice_unaligned_f32x4(unsafe {
                            std::slice::from_raw_parts(wt_ptr.add(ic * m + oc), 4)
                        });

                        let w_hi = from_slice_unaligned_f32x4(unsafe {
                            std::slice::from_raw_parts(wt_ptr.add(ic * m + oc + 4), 4)
                        });

                        acc_lo += w_lo * f32x4::splat(x_val);

                        acc_hi += w_hi * f32x4::splat(x_val);
                    }

                    let b_lo = if let Some(bias) = bias_data {
                        from_slice_unaligned_f32x4(unsafe {
                            std::slice::from_raw_parts(bias.as_ptr().add(oc), 4)
                        })
                    } else {
                        f32x4::splat(bias_scalar.unwrap_or(0.0))
                    };

                    let b_hi = if let Some(bias) = bias_data {
                        from_slice_unaligned_f32x4(unsafe {
                            std::slice::from_raw_parts(bias.as_ptr().add(oc + 4), 4)
                        })
                    } else {
                        f32x4::splat(bias_scalar.unwrap_or(0.0))
                    };

                    let res_lo = (acc_lo + b_lo).to_array();

                    let res_hi = (acc_hi + b_hi).to_array();

                    for i in 0..4 {
                        unsafe { *out_ptr.add(out_base + (oc + i) * spatial) = res_lo[i]; }
                    }

                    for i in 0..4 {
                        unsafe { *out_ptr.add(out_base + (oc + 4 + i) * spatial) = res_hi[i]; }
                    }
                }

                #[cfg(not(all(
                    feature = "simd",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )))]
                {
                    for oc_off in 0..8 {
                        let oc_i = oc + oc_off;

                        let mut sum = 0.0f32;

                        for ic in 0..k {
                            sum += unsafe { *x_ptr.add(x_base + ic * spatial) }
                                * unsafe { *wt_ptr.add(ic * m + oc_i) };
                        }

                        let b = bias_data.map_or(bias_scalar.unwrap_or(0.0), |bias| bias[oc_i]);

                        unsafe { *out_ptr.add(out_base + oc_i * spatial) = sum + b; }
                    }
                }

                oc += 8;
            }

            // Remainder channels (scalar)

            while oc < m {
                let mut sum = 0.0f32;

                for ic in 0..k {
                    sum += unsafe { *x_ptr.add(x_base + ic * spatial) }
                        * unsafe { *wt_ptr.add(ic * m + oc) };
                }

                let b = bias_data.map_or(bias_scalar.unwrap_or(0.0), |bias| bias[oc]);

                unsafe { *out_ptr.add(out_base + oc * spatial) = sum + b; }

                oc += 1;
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let x_ptr = x_usize as *const f32;

        let out_ptr = out_usize as *mut f32;

        let wt_ptr = wt_usize as *const f32;

        for sp_idx in 0..total_spatial {
            let n = sp_idx / (in_height * in_width);

            let hw = sp_idx % (in_height * in_width);

            let oh = hw / in_width;

            let ow = hw % in_width;

            let x_base = n * k * spatial + oh * in_width + ow;

            let out_base = n * m * spatial + oh * in_width + ow;

            let mut oc = 0;

            while oc + 8 <= m {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let mut acc = [0.0f32; 8];

                    for ic in 0..k {
                        let x_val = unsafe { *x_ptr.add(x_base + ic * spatial) };

                        let w = unsafe {
                            std::slice::from_raw_parts(wt_ptr.add(ic * m + oc), 8)
                        };

                        for lane in 0..8 {
                            acc[lane] += w[lane] * x_val;
                        }
                    }

                    let b = if let Some(bias) = bias_data {
                        unsafe {
                            std::slice::from_raw_parts(bias.as_ptr().add(oc), 8)
                        }
                    } else {
                        &[bias_scalar.unwrap_or(0.0); 8]
                    };

                    for i in 0..8 {
                        unsafe {
                            *out_ptr.add(out_base + (oc + i) * spatial) = acc[i] + b[i];
                        }
                    }
                }

                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                {
                    use wide::f32x4;

                    let mut acc_lo = f32x4::ZERO;

                    let mut acc_hi = f32x4::ZERO;

                    for ic in 0..k {
                        let x_val = unsafe { *x_ptr.add(x_base + ic * spatial) };

                        let w_lo = from_slice_unaligned_f32x4(unsafe {
                            std::slice::from_raw_parts(wt_ptr.add(ic * m + oc), 4)
                        });

                        let w_hi = from_slice_unaligned_f32x4(unsafe {
                            std::slice::from_raw_parts(wt_ptr.add(ic * m + oc + 4), 4)
                        });

                        acc_lo += w_lo * f32x4::splat(x_val);

                        acc_hi += w_hi * f32x4::splat(x_val);
                    }

                    let b_lo = if let Some(bias) = bias_data {
                        from_slice_unaligned_f32x4(unsafe {
                            std::slice::from_raw_parts(bias.as_ptr().add(oc), 4)
                        })
                    } else {
                        f32x4::splat(bias_scalar.unwrap_or(0.0))
                    };

                    let b_hi = if let Some(bias) = bias_data {
                        from_slice_unaligned_f32x4(unsafe {
                            std::slice::from_raw_parts(bias.as_ptr().add(oc + 4), 4)
                        })
                    } else {
                        f32x4::splat(bias_scalar.unwrap_or(0.0))
                    };

                    let res_lo = (acc_lo + b_lo).to_array();

                    let res_hi = (acc_hi + b_hi).to_array();

                    for i in 0..4 {
                        unsafe { *out_ptr.add(out_base + (oc + i) * spatial) = res_lo[i]; }
                    }

                    for i in 0..4 {
                        unsafe { *out_ptr.add(out_base + (oc + 4 + i) * spatial) = res_hi[i]; }
                    }
                }

                #[cfg(not(all(
                    feature = "simd",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )))]
                {
                    for oc_off in 0..8 {
                        let oc_i = oc + oc_off;

                        let mut sum = 0.0f32;

                        for ic in 0..k {
                            sum += unsafe { *x_ptr.add(x_base + ic * spatial) }
                                * unsafe { *wt_ptr.add(ic * m + oc_i) };
                        }

                        let b = bias_data.map_or(bias_scalar.unwrap_or(0.0), |bias| bias[oc_i]);

                        unsafe { *out_ptr.add(out_base + oc_i * spatial) = sum + b; }
                    }
                }

                oc += 8;
            }

            while oc < m {
                let mut sum = 0.0f32;

                for ic in 0..k {
                    sum += unsafe { *x_ptr.add(x_base + ic * spatial) }
                        * unsafe { *wt_ptr.add(ic * m + oc) };
                }

                let b = bias_data.map_or(bias_scalar.unwrap_or(0.0), |bias| bias[oc]);

                unsafe { *out_ptr.add(out_base + oc * spatial) = sum + b; }

                oc += 1;
            }
        }
    }

    output
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn conv2d_3x3_direct(
    x: &Tensor,

    w: &Tensor,

    w_pre_t: Option<&Tensor>,

    bias: Option<&Tensor>,

    batch_size: usize,

    in_channels: usize,

    out_channels: usize,

    in_height: usize,

    in_width: usize,

    stride: usize,

    padding: usize,

    out_height: usize,

    out_width: usize,
) -> Tensor {
    let output_shape = vec![
        batch_size as i64,
        out_channels as i64,
        out_height as i64,
        out_width as i64,
    ];

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());

    let output_inner = Arc::make_mut(&mut output.inner);

    let output_storage = Arc::make_mut(&mut output_inner.storage);

    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };

    let out_data = Arc::make_mut(&mut cpu_storage.data);

    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let x_ptr = x.data_ptr() as *const f32;

    let w_ptr = w.data_ptr() as *const f32;

    let bias_scalar: Option<f32> =
        bias.and_then(|b| if b.numel() == 1 { Some(b.item()) } else { None });

    let bias_data: Option<Vec<f32>> = bias.filter(|b| b.numel() > 1).map(|b| {
        let b_ptr = b.data_ptr() as *const f32;

        // SAFETY: bias tensor has out_channels elements as verified by tensor shape.

        unsafe { std::slice::from_raw_parts(b_ptr, out_channels).to_vec() }
    });

    let in_h = in_height as isize;

    let in_w = in_width as isize;

    // --- Step 1: Pre-transpose weights into thread-local buffer, reuse across calls

    let k = in_channels * 9;

    let oc_count = out_channels;

    let needed = k * oc_count;

    let use_pre_t = w_pre_t.is_some_and(|wt| wt.numel() as usize == needed);

    let wt_trans_ptr: *const f32 = if use_pre_t {
        // Use pre-transposed weight from DAG executor

        let wt_ptr = w_pre_t.unwrap().data_ptr() as *const f32;

        // SAFETY: wt has shape [k, oc_count] and needed elements (verified above).

        let wt_data = unsafe { std::slice::from_raw_parts(wt_ptr, needed) };

        WT_TRANS_BUF.with(|buf| {
            let mut b = buf.borrow_mut();

            let v = if let Some(ref mut vec) = &mut *b {
                vec
            } else {
                let new_vec = vec![0.0f32; needed];

                *b = Some(new_vec);

                b.as_mut().unwrap()
            };

            v.resize(needed, 0.0);

            v.copy_from_slice(wt_data);

            v.as_ptr()
        })
    } else {
        // Fill WT_TRANS_BUF from original weight (fallback for grouped convs or missing pre-transposed weight)

        WT_TRANS_BUF.with(|buf| {
            let mut b = buf.borrow_mut();

            let v = if let Some(ref mut vec) = &mut *b {
                vec
            } else {
                let new_vec = vec![0.0f32; needed];

                *b = Some(new_vec);

                b.as_mut().unwrap()
            };

            v.resize(needed, 0.0);

            for ic in 0..in_channels {
                for kh in 0..3 {
                    for kw in 0..3 {
                        let k_idx = ic * 9 + kh * 3 + kw;

                        for oc in 0..out_channels {
                            let w_idx = ((oc * in_channels + ic) * 3 + kh) * 3 + kw;

                            // SAFETY: k_idx < k, oc < out_channels = oc_count, v.len() >= needed = k*oc_count.

                            // w_idx is bounded by the weight tensor dimensions.

                            unsafe {
                                *v.get_unchecked_mut(k_idx * oc_count + oc) = *w_ptr.add(w_idx)
                            };
                        }
                    }
                }
            }

            v.as_ptr()
        })
    };

    // SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
    let wt_trans_slice = unsafe { std::slice::from_raw_parts(wt_trans_ptr, needed) };

    // --- Step 2: Main kernel (parallel over output spatial+batch positions)

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let total_work = batch_size * out_height * out_width;

        let x_usize = x_ptr as usize;

        let out_usize = out_ptr as usize;

        (0..total_work).into_par_iter().for_each(|idx| {

            let n = idx / (out_height * out_width);

            let pos = idx % (out_height * out_width);

            let oh = pos / out_width;

            let ow = pos % out_width;

            let x_ptr = x_usize as *const f32;

            let out_ptr = out_usize as *mut f32;



                    S_BUF.with(|s| {

                        let mut s_buf = s.borrow_mut();

                        if s_buf.len() < k {

                            s_buf.resize(k, 0.0);

                        }

                        let s = &mut s_buf[..k];

                        let mut k_idx = 0;

                        for ic in 0..in_channels {

                            for kh in 0..3 {

                                for kw in 0..3 {

                                    let ih_s = (oh as isize) * stride as isize + (kh as isize) - padding as isize;

                                    let iw_s = (ow as isize) * stride as isize + (kw as isize) - padding as isize;

                                    s[k_idx] =

                                        if ih_s >= 0 && ih_s < in_h && iw_s >= 0 && iw_s < in_w {

                                            let ih = ih_s as usize;

                                            let iw = iw_s as usize;

                                            let x_idx = ((n * in_channels + ic) * in_height + ih)

                                                * in_width

                                                + iw;

                                            // SAFETY: x_idx computed from n, ic, ih, iw is bounded by batch_size*in_channels*in_height*in_width = x tensor element count.

                                            unsafe { *x_ptr.add(x_idx) }

                                        } else {

                                            0.0

                                        };

                                    k_idx += 1;

                                }

                            }

                        }



                        let mut oc = 0;

                        // Process output channels in blocks of 8 (x86) or 8 (aarch64 does 4+4)

                        while oc + 8 <= out_channels {

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]

                            {

                                use wide::f32x8;

                                let mut acc = f32x8::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in blocks of 8.

                                    let w_ptr_k =

                                        // SAFETY: Pointer arithmetic stays within bounds of the allocated tensor storage.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    // SAFETY: w_ptr_k points to 8 valid f32 elements within wt_trans_slice when oc+8 <= out_channels.

                                    let w_simd = from_slice_unaligned_f32x8(unsafe {

                                        std::slice::from_raw_parts(w_ptr_k, 8)

                                    });

                                    acc += w_simd * f32x8::splat(x_val);

                                }



                                let bias_vec = if let Some(ref b) = bias_data {

                                    // SAFETY: All pointer accesses are within bounds of their respective tensor allocations.
                                    let b_slice = unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 8)

                                    };

                                    from_slice_unaligned_f32x8(b_slice)

                                } else if let Some(b) = bias_scalar {

                                    f32x8::splat(b)

                                } else {

                                    f32x8::ZERO

                                };



                                let res = acc + bias_vec;

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;

                                let res_arr = res.to_array();

                                for i in 0..8 {

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    // SAFETY: out_idx computed from n, oc+i, oh, ow is bounded by batch_size*out_channels*out_height*out_width = output tensor element count.

                                    unsafe {

                                        *out_ptr.add(out_idx) = res_arr[i];

                                    }

                                }

                            }

                            #[cfg(all(feature = "simd", target_arch = "aarch64"))]

                            {

                                use wide::f32x4;

                                let mut acc_lo = f32x4::ZERO;

                                let mut acc_hi = f32x4::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let base_ptr =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    // lower 4

                                    // SAFETY: All pointer accesses are within bounds of their respective tensor allocations.
                                    let w_lo = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr, 4)

                                    });

                                    acc_lo = acc_lo + w_lo * f32x4::splat(x_val);

                                    // upper 4

                                    // SAFETY: All pointer accesses are within bounds of their respective tensor allocations.
                                    let w_hi = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr.add(4), 4)

                                    });

                                    acc_hi = acc_hi + w_hi * f32x4::splat(x_val);

                                }



                                let bias_lo = if let Some(ref b) = bias_data {

                                    // SAFETY: oc < out_channels and oc+4 <= out_channels, so b[oc..oc+4] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };

                                let bias_hi = if let Some(ref b) = bias_data {

                                    // SAFETY: oc+4 < out_channels, so b[oc+4..oc+8] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc + 4), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };



                                let res_lo = acc_lo + bias_lo;

                                let res_hi = acc_hi + bias_hi;

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;

                                let res_lo_arr = res_lo.to_array();

                                let res_hi_arr = res_hi.to_array();

                                for i in 0..4 {

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    // SAFETY: Pointer arithmetic stays within bounds of the allocated tensor storage.
                                    unsafe {

                                        *out_ptr.add(out_idx) = res_lo_arr[i];

                                    }

                                }

                                for i in 0..4 {

                                    let out_idx = out_idx_base + (4 + i) * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = res_hi_arr[i];

                                    }

                                }

                            }

                            #[cfg(not(all(

                                feature = "simd",

                                any(target_arch = "x86_64", target_arch = "aarch64")

                            )))]

                            {

                                for oc_off in 0..8 {

                                    let oc_i = oc + oc_off;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut sum = 0.0f32;

                                    for k_idx in 0..k {

                                        let x_val = s[k_idx];

                                        // SAFETY: k_idx*oc_count+oc_i < k*oc_count = wt_trans_slice.len() since oc_i < out_channels.

                                        let w_val = unsafe {

                                            *wt_trans_slice.get_unchecked(k_idx * oc_count + oc_i)

                                        };

                                        sum += x_val * w_val;

                                    }

                                    let bias_val = bias_scalar.unwrap_or_else(|| {

                                        bias_data.as_ref().map(|b| b[oc_i]).unwrap_or(0.0)

                                    });

                                    let out_idx = ((n * out_channels + oc_i) * out_height + oh)

                                        * out_width

                                        + ow;

                                    // SAFETY: out_idx bounded by tensor dimensions.

                                    unsafe {

                                        *out_ptr.add(out_idx) = sum + bias_val;

                                    }

                                }

                            }

                            oc += 8;

                        }



                        // Remainder channels (scalar)

                        while oc < out_channels {

                            let mut sum = 0.0f32;

                            for k_idx in 0..k {

                                let x_val = s[k_idx];

                                // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() since oc < out_channels.

                                let w_val =

                                    // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() since oc < out_channels.

                                    unsafe { *wt_trans_slice.get_unchecked(k_idx * oc_count + oc) };

                                sum += x_val * w_val;

                            }

                            let bias_val = bias_scalar.unwrap_or_else(|| {

                                bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0)

                            });

                            let out_idx =

                                ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                            // SAFETY: out_idx bounded by tensor dimensions.

                            unsafe {

                                *out_ptr.add(out_idx) = sum + bias_val;

                            }

                            oc += 1;

                        }

                    });

        })
    }

    #[cfg(not(feature = "parallel"))]
    {
        let total_work = batch_size * out_height * out_width;

        for idx in 0..total_work {

            let n = idx / (out_height * out_width);

            let pos = idx % (out_height * out_width);

            let oh = pos / out_width;

            let ow = pos % out_width;

                    S_BUF.with(|s| {

                        let mut s_buf = s.borrow_mut();

                        if s_buf.len() < k {

                            s_buf.resize(k, 0.0);

                        }

                        let s = &mut s_buf[..k];

                        let mut k_idx = 0;

                        for ic in 0..in_channels {

                            for kh in 0..3 {

                                for kw in 0..3 {

                                    let ih_s = (oh as isize) * stride as isize + (kh as isize) - padding as isize;

                                    let iw_s = (ow as isize) * stride as isize + (kw as isize) - padding as isize;

                                    s[k_idx] =

                                        if ih_s >= 0 && ih_s < in_h && iw_s >= 0 && iw_s < in_w {

                                            let ih = ih_s as usize;

                                            let iw = iw_s as usize;

                                            let x_idx = ((n * in_channels + ic) * in_height + ih)

                                                * in_width

                                                + iw;

                                            // SAFETY: x_idx computed from n, ic, ih, iw is bounded by batch_size*in_channels*in_height*in_width = x tensor element count.

                                            unsafe { *x_ptr.add(x_idx) }

                                        } else {

                                            0.0

                                        };

                                    k_idx += 1;

                                }

                            }

                        }



                        let mut oc = 0;

                        while oc + 8 <= out_channels {

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]

                            {

                                use wide::f32x8;

                                let mut acc = f32x8::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let w_ptr_k =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_simd = from_slice_unaligned_f32x8(unsafe {

                                        std::slice::from_raw_parts(w_ptr_k, 8)

                                    });

                                    acc += w_simd * f32x8::splat(x_val);

                                }



                                let bias_vec = if let Some(ref b) = bias_data {

                                    let b_slice = unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 8)

                                    };

                                    from_slice_unaligned_f32x8(b_slice)

                                } else if let Some(b) = bias_scalar {

                                    f32x8::splat(b)

                                } else {

                                    f32x8::ZERO

                                };



                                let res = acc + bias_vec;

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;

                                let res_arr = res.to_array();

                                for i in 0..8 {

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    // SAFETY: out_idx computed from n, oc+i, oh, ow is bounded by batch_size*out_channels*out_height*out_width = output tensor element count.

                                    unsafe {

                                        *out_ptr.add(out_idx) = res_arr[i];

                                    }

                                }

                            }

                            #[cfg(all(feature = "simd", target_arch = "aarch64"))]

                            {

                                use wide::f32x4;

                                let mut acc_lo = f32x4::ZERO;

                                let mut acc_hi = f32x4::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let base_ptr =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    // lower 4

                                    let w_lo = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr, 4)

                                    });

                                    acc_lo = acc_lo + w_lo * f32x4::splat(x_val);

                                    // upper 4

                                    let w_hi = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr.add(4), 4)

                                    });

                                    acc_hi = acc_hi + w_hi * f32x4::splat(x_val);

                                }



                                let bias_lo = if let Some(ref b) = bias_data {

                                    // SAFETY: oc < out_channels and oc+4 <= out_channels, so b[oc..oc+4] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };

                                let bias_hi = if let Some(ref b) = bias_data {

                                    // SAFETY: oc+4 < out_channels, so b[oc+4..oc+8] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc + 4), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };



                                let res_lo = acc_lo + bias_lo;

                                let res_hi = acc_hi + bias_hi;

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;

                                let res_lo_arr = res_lo.to_array();

                                let res_hi_arr = res_hi.to_array();

                                for i in 0..4 {

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = res_lo_arr[i];

                                    }

                                }

                                for i in 0..4 {

                                    let out_idx = out_idx_base + (4 + i) * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = res_hi_arr[i];

                                    }

                                }

                            }

                            #[cfg(not(all(

                                feature = "simd",

                                any(target_arch = "x86_64", target_arch = "aarch64")

                            )))]

                            {

                                for oc_off in 0..8 {

                                    let oc_i = oc + oc_off;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut sum = 0.0f32;

                                    for k_idx in 0..k {

                                        let x_val = s[k_idx];

                                        // SAFETY: k_idx*oc_count+oc_i < k*oc_count = wt_trans_slice.len() since oc_i < out_channels.

                                        let w_val = unsafe {

                                            *wt_trans_slice.get_unchecked(k_idx * oc_count + oc_i)

                                        };

                                        sum += x_val * w_val;

                                    }

                                    let bias_val = bias_scalar.unwrap_or_else(|| {

                                        bias_data.as_ref().map(|b| b[oc_i]).unwrap_or(0.0)

                                    });

                                    let out_idx = ((n * out_channels + oc_i) * out_height + oh)

                                        * out_width

                                        + ow;

                                    // SAFETY: out_idx bounded by tensor dimensions.

                                    unsafe {

                                        *out_ptr.add(out_idx) = sum + bias_val;

                                    }

                                }

                            }

                            oc += 8;

                        }



                        // Remainder channels (scalar)

                        while oc < out_channels {

                            let mut sum = 0.0f32;

                            for k_idx in 0..k {

                                let x_val = s[k_idx];

                                // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() since oc < out_channels.

                                let w_val =

                                    // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() since oc < out_channels.

                                    unsafe { *wt_trans_slice.get_unchecked(k_idx * oc_count + oc) };

                                sum += x_val * w_val;

                            }

                            let bias_val = bias_scalar.unwrap_or_else(|| {

                                bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0)

                            });

                            let out_idx =

                                ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                            // SAFETY: out_idx bounded by tensor dimensions.

                            unsafe {

                                *out_ptr.add(out_idx) = sum + bias_val;

                            }

                            oc += 1;

                        }

                    });
        }
    }

    output
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_conv_bn_silu_3x3_direct(
    x: &Tensor,

    w: &Tensor,

    bias: Option<&Tensor>,

    bn_weight: &Tensor,

    bn_bias: &Tensor,

    bn_running_mean: &Tensor,

    bn_running_var: &Tensor,

    bn_eps: f32,

    batch_size: usize,

    in_channels: usize,

    out_channels: usize,

    in_height: usize,

    in_width: usize,

    out_height: usize,

    out_width: usize,
) -> Tensor {
    let output_shape = vec![
        batch_size as i64,
        out_channels as i64,
        out_height as i64,
        out_width as i64,
    ];

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());

    let output_inner = Arc::make_mut(&mut output.inner);

    let output_storage = Arc::make_mut(&mut output_inner.storage);

    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };

    let out_data = Arc::make_mut(&mut cpu_storage.data);

    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let x_ptr = x.data_ptr() as *const f32;

    let w_ptr = w.data_ptr() as *const f32;

    let bn_weight_ptr = bn_weight.data_ptr() as *const f32;

    let bn_bias_ptr = bn_bias.data_ptr() as *const f32;

    let bn_mean_ptr = bn_running_mean.data_ptr() as *const f32;

    let bn_var_ptr = bn_running_var.data_ptr() as *const f32;

    let bias_scalar: Option<f32> =
        bias.and_then(|b| if b.numel() == 1 { Some(b.item()) } else { None });

    let bias_data: Option<Vec<f32>> = bias.filter(|b| b.numel() > 1).map(|b| {
        let b_ptr = b.data_ptr() as *const f32;

        // SAFETY: bias tensor has out_channels elements as verified by tensor shape.

        unsafe { std::slice::from_raw_parts(b_ptr, out_channels).to_vec() }
    });

    let in_h = in_height as isize;

    let in_w = in_width as isize;

    // --- Step 1: Get transposed weights from thread-local buffer (recompute per call)

    let k = in_channels * 9;

    let oc_count = out_channels;

    let wt_trans_ptr: *const f32 = WT_TRANS_BUF.with(|buf| {
        let needed = k * oc_count;

        let mut b = buf.borrow_mut();

        let v = if let Some(ref mut v) = &mut *b {
            v
        } else {
            let new_vec = vec![0.0f32; needed];

            *b = Some(new_vec);

            b.as_mut().unwrap()
        };

        if v.len() < needed {
            v.resize(needed, 0.0);
        }

        for ic in 0..in_channels {
            for kh in 0..3 {
                for kw in 0..3 {
                    let k_idx = ic * 9 + kh * 3 + kw;

                    for oc in 0..out_channels {
                        let w_idx = ((oc * in_channels + ic) * 3 + kh) * 3 + kw;

                        // SAFETY: k_idx < k, oc < out_channels = oc_count, v.len() >= needed = k*oc_count.

                        // w_idx is bounded by the weight tensor dimensions.

                        unsafe { *v.get_unchecked_mut(k_idx * oc_count + oc) = *w_ptr.add(w_idx) };
                    }
                }
            }
        }

        v.as_ptr()
    });

    // SAFETY: wt_trans_ptr points to the thread-local buffer resized to `needed = k*oc_count` elements.

    let wt_trans_slice = unsafe { std::slice::from_raw_parts(wt_trans_ptr, k * oc_count) };

    // --- Step 2: Main kernel (parallel over output spatial+batch positions)

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let total_work = batch_size * out_height * out_width;

        let x_usize = x_ptr as usize;

        let out_usize = out_ptr as usize;

        let bn_weight_usize = bn_weight_ptr as usize;

        let bn_bias_usize = bn_bias_ptr as usize;

        let bn_mean_usize = bn_mean_ptr as usize;

        let bn_var_usize = bn_var_ptr as usize;

        (0..total_work).into_par_iter().for_each(|idx| {

            let n = idx / (out_height * out_width);

            let pos = idx % (out_height * out_width);

            let oh = pos / out_width;

            let ow = pos % out_width;

            let x_ptr = x_usize as *const f32;

            let out_ptr = out_usize as *mut f32;

                    let bn_weight_ptr = bn_weight_usize as *const f32;

                    let bn_bias_ptr = bn_bias_usize as *const f32;

                    let bn_mean_ptr = bn_mean_usize as *const f32;

                    let bn_var_ptr = bn_var_usize as *const f32;



                    S_BUF.with(|s| {

                        let mut s_buf = s.borrow_mut();

                        if s_buf.len() < k {

                            s_buf.resize(k, 0.0);

                        }

                        let s = &mut s_buf[..k];

                        let mut k_idx = 0;

                        for ic in 0..in_channels {

                            for kh in 0..3 {

                                for kw in 0..3 {

                                    let ih_s = (oh as isize) + (kh as isize) - 1;

                                    let iw_s = (ow as isize) + (kw as isize) - 1;

                                    s[k_idx] =

                                        if ih_s >= 0 && ih_s < in_h && iw_s >= 0 && iw_s < in_w {

                                            let ih = ih_s as usize;

                                            let iw = iw_s as usize;

                                            let x_idx = ((n * in_channels + ic) * in_height + ih)

                                                * in_width

                                                + iw;

                                            // SAFETY: x_idx computed from n, ic, ih, iw is bounded by batch_size*in_channels*in_height*in_width = x tensor element count.

                                            unsafe { *x_ptr.add(x_idx) }

                                        } else {

                                            0.0

                                        };

                                    k_idx += 1;

                                }

                            }

                        }



                        let mut oc = 0;

                        while oc + 8 <= out_channels {

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]

                            {

                                use wide::f32x8;

                                let mut acc = f32x8::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let w_ptr_k =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_simd = from_slice_unaligned_f32x8(unsafe {

                                        std::slice::from_raw_parts(w_ptr_k, 8)

                                    });

                                    acc += w_simd * f32x8::splat(x_val);

                                }



                                let bias_vec = if let Some(ref b) = bias_data {

                                    let b_slice = unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 8)

                                    };

                                    from_slice_unaligned_f32x8(b_slice)

                                } else if let Some(b) = bias_scalar {

                                    f32x8::splat(b)

                                } else {

                                    f32x8::ZERO

                                };



                                let res_arr = (acc + bias_vec).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                // Apply BN+SiLU per-element and store

                                for i in 0..8 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_arr[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v * (1.0 / (1.0 + (-v).exp()));

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(all(feature = "simd", target_arch = "aarch64"))]

                            {

                                use wide::f32x4;

                                let mut acc_lo = f32x4::ZERO;

                                let mut acc_hi = f32x4::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let base_ptr =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_lo = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr, 4)

                                    });

                                    acc_lo = acc_lo + w_lo * f32x4::splat(x_val);

                                    let w_hi = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr.add(4), 4)

                                    });

                                    acc_hi = acc_hi + w_hi * f32x4::splat(x_val);

                                }



                                let bias_lo = if let Some(ref b) = bias_data {

                                    // SAFETY: oc < out_channels and oc+4 <= out_channels, so b[oc..oc+4] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };

                                let bias_hi = if let Some(ref b) = bias_data {

                                    // SAFETY: oc+4 < out_channels, so b[oc+4..oc+8] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc + 4), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };



                                let res_lo = (acc_lo + bias_lo).to_array();

                                let res_hi = (acc_hi + bias_hi).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                // Apply BN+SiLU for lo (4)

                                for i in 0..4 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_lo[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v * (1.0 / (1.0 + (-v).exp()));

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                                // Apply BN+SiLU for hi (4)

                                for i in 0..4 {

                                    let oc_i = oc + 4 + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_hi[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v * (1.0 / (1.0 + (-v).exp()));

                                    let out_idx = out_idx_base + (4 + i) * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(not(all(

                                feature = "simd",

                                any(target_arch = "x86_64", target_arch = "aarch64")

                            )))]

                            {

                                for oc_off in 0..8 {

                                    let oc_i = oc + oc_off;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut sum = 0.0f32;

                                    for k_idx in 0..k {

                                        let x_val = s[k_idx];

                                        let w_val = unsafe {

                                            *wt_trans_slice.get_unchecked(k_idx * oc_count + oc_i)

                                        };

                                        sum += x_val * w_val;

                                    }

                                    let bias_val = bias_scalar.unwrap_or_else(|| {

                                        bias_data.as_ref().map(|b| b[oc_i]).unwrap_or(0.0)

                                    });

                                    let mut v = sum + bias_val;

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v * (1.0 / (1.0 + (-v).exp()));

                                    let out_idx = ((n * out_channels + oc_i) * out_height + oh)

                                        * out_width

                                        + ow;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            oc += 8;

                        }



                        // Remainder channels (scalar)

                        while oc < out_channels {

                            let mut sum = 0.0f32;

                            for k_idx in 0..k {

                                let x_val = s[k_idx];

                                let w_val =

                                    // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() since oc < out_channels.

                                    unsafe { *wt_trans_slice.get_unchecked(k_idx * oc_count + oc) };

                                sum += x_val * w_val;

                            }

                            let bias_val = bias_scalar.unwrap_or_else(|| {

                                bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0)

                            });

                            let mut v = sum + bias_val;

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let mean = unsafe { *bn_mean_ptr.add(oc) };

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let var = unsafe { *bn_var_ptr.add(oc) };

                            let inv_std = 1.0 / (var + bn_eps).sqrt();

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc) }

                                // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                + unsafe { *bn_bias_ptr.add(oc) };

                            v = v * (1.0 / (1.0 + (-v).exp()));

                            let out_idx =

                                ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                            unsafe {

                                *out_ptr.add(out_idx) = v;

                            }

                            oc += 1;

                        }

                    });

        })
    }

    #[cfg(not(feature = "parallel"))]
    {
        let total_work = batch_size * out_height * out_width;

        for idx in 0..total_work {

            let n = idx / (out_height * out_width);

            let pos = idx % (out_height * out_width);

            let oh = pos / out_width;

            let ow = pos % out_width;

                    S_BUF.with(|s| {

                        let mut s_buf = s.borrow_mut();

                        if s_buf.len() < k {

                            s_buf.resize(k, 0.0);

                        }

                        let s = &mut s_buf[..k];

                        let mut k_idx = 0;

                        for ic in 0..in_channels {

                            for kh in 0..3 {

                                for kw in 0..3 {

                                    let ih_s = (oh as isize) + (kh as isize) - 1;

                                    let iw_s = (ow as isize) + (kw as isize) - 1;

                                    s[k_idx] =

                                        if ih_s >= 0 && ih_s < in_h && iw_s >= 0 && iw_s < in_w {

                                            let ih = ih_s as usize;

                                            let iw = iw_s as usize;

                                            let x_idx = ((n * in_channels + ic) * in_height + ih)

                                                * in_width

                                                + iw;

                                            // SAFETY: x_idx computed from n, ic, ih, iw is bounded by batch_size*in_channels*in_height*in_width = x tensor element count.

                                            unsafe { *x_ptr.add(x_idx) }

                                        } else {

                                            0.0

                                        };

                                    k_idx += 1;

                                }

                            }

                        }



                        let mut oc = 0;

                        while oc + 8 <= out_channels {

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]

                            {

                                use wide::f32x8;

                                let mut acc = f32x8::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let w_ptr_k =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_simd = from_slice_unaligned_f32x8(unsafe {

                                        std::slice::from_raw_parts(w_ptr_k, 8)

                                    });

                                    acc += w_simd * f32x8::splat(x_val);

                                }



                                let bias_vec = if let Some(ref b) = bias_data {

                                    let b_slice = unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 8)

                                    };

                                    from_slice_unaligned_f32x8(b_slice)

                                } else if let Some(b) = bias_scalar {

                                    f32x8::splat(b)

                                } else {

                                    f32x8::ZERO

                                };



                                let res_arr = (acc + bias_vec).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                // Apply BN+SiLU per-element and store

                                for i in 0..8 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_arr[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v * (1.0 / (1.0 + (-v).exp()));

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(all(feature = "simd", target_arch = "aarch64"))]

                            {

                                use wide::f32x4;

                                let mut acc_lo = f32x4::ZERO;

                                let mut acc_hi = f32x4::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let base_ptr =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_lo = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr, 4)

                                    });

                                    acc_lo = acc_lo + w_lo * f32x4::splat(x_val);

                                    let w_hi = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr.add(4), 4)

                                    });

                                    acc_hi = acc_hi + w_hi * f32x4::splat(x_val);

                                }



                                let bias_lo = if let Some(ref b) = bias_data {

                                    // SAFETY: oc < out_channels and oc+4 <= out_channels, so b[oc..oc+4] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };

                                let bias_hi = if let Some(ref b) = bias_data {

                                    // SAFETY: oc+4 < out_channels, so b[oc+4..oc+8] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc + 4), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };



                                let res_lo = (acc_lo + bias_lo).to_array();

                                let res_hi = (acc_hi + bias_hi).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                // Apply BN+SiLU for lo (4)

                                for i in 0..4 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_lo[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v * (1.0 / (1.0 + (-v).exp()));

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                                // Apply BN+SiLU for hi (4)

                                for i in 0..4 {

                                    let oc_i = oc + 4 + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_hi[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v * (1.0 / (1.0 + (-v).exp()));

                                    let out_idx = out_idx_base + (4 + i) * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(not(all(

                                feature = "simd",

                                any(target_arch = "x86_64", target_arch = "aarch64")

                            )))]

                            {

                                for oc_off in 0..8 {

                                    let oc_i = oc + oc_off;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut sum = 0.0f32;

                                    for k_idx in 0..k {

                                        let x_val = s[k_idx];

                                        let w_val = unsafe {

                                            *wt_trans_slice.get_unchecked(k_idx * oc_count + oc_i)

                                        };

                                        sum += x_val * w_val;

                                    }

                                    let bias_val = bias_scalar.unwrap_or_else(|| {

                                        bias_data.as_ref().map(|b| b[oc_i]).unwrap_or(0.0)

                                    });

                                    let mut v = sum + bias_val;

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v * (1.0 / (1.0 + (-v).exp()));

                                    let out_idx = ((n * out_channels + oc_i) * out_height + oh)

                                        * out_width

                                        + ow;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            oc += 8;

                        }



                        // Remainder channels (scalar)

                        while oc < out_channels {

                            let mut sum = 0.0f32;

                            for k_idx in 0..k {

                                let x_val = s[k_idx];

                                let w_val =

                                    // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() since oc < out_channels.

                                    unsafe { *wt_trans_slice.get_unchecked(k_idx * oc_count + oc) };

                                sum += x_val * w_val;

                            }

                            let bias_val = bias_scalar.unwrap_or_else(|| {

                                bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0)

                            });

                            let mut v = sum + bias_val;

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let mean = unsafe { *bn_mean_ptr.add(oc) };

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let var = unsafe { *bn_var_ptr.add(oc) };

                            let inv_std = 1.0 / (var + bn_eps).sqrt();

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc) }

                                // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                + unsafe { *bn_bias_ptr.add(oc) };

                            v = v * (1.0 / (1.0 + (-v).exp()));

                            let out_idx =

                                ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                            unsafe {

                                *out_ptr.add(out_idx) = v;

                            }

                            oc += 1;

                        }

                    });
        }
    }

    output
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_conv_bn_3x3_direct(
    x: &Tensor,

    w: &Tensor,

    bias: Option<&Tensor>,

    bn_weight: &Tensor,

    bn_bias: &Tensor,

    bn_running_mean: &Tensor,

    bn_running_var: &Tensor,

    bn_eps: f32,

    batch_size: usize,

    in_channels: usize,

    out_channels: usize,

    in_height: usize,

    in_width: usize,

    out_height: usize,

    out_width: usize,
) -> Tensor {
    let output_shape = vec![
        batch_size as i64,
        out_channels as i64,
        out_height as i64,
        out_width as i64,
    ];

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());

    let output_inner = Arc::make_mut(&mut output.inner);

    let output_storage = Arc::make_mut(&mut output_inner.storage);

    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };

    let out_data = Arc::make_mut(&mut cpu_storage.data);

    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let x_ptr = x.data_ptr() as *const f32;

    let w_ptr = w.data_ptr() as *const f32;

    let bn_weight_ptr = bn_weight.data_ptr() as *const f32;

    let bn_bias_ptr = bn_bias.data_ptr() as *const f32;

    let bn_mean_ptr = bn_running_mean.data_ptr() as *const f32;

    let bn_var_ptr = bn_running_var.data_ptr() as *const f32;

    let bias_scalar: Option<f32> =
        bias.and_then(|b| if b.numel() == 1 { Some(b.item()) } else { None });

    let bias_data: Option<Vec<f32>> = bias.filter(|b| b.numel() > 1).map(|b| {
        let b_ptr = b.data_ptr() as *const f32;

        // SAFETY: bias tensor has out_channels elements as verified by tensor shape.

        unsafe { std::slice::from_raw_parts(b_ptr, out_channels).to_vec() }
    });

    let in_h = in_height as isize;

    let in_w = in_width as isize;

    let k = in_channels * 9;

    let oc_count = out_channels;

    let wt_trans_ptr: *const f32 = WT_TRANS_BUF.with(|buf| {
        let needed = k * oc_count;

        let mut b = buf.borrow_mut();

        let v = if let Some(ref mut v) = &mut *b {
            v
        } else {
            let new_vec = vec![0.0f32; needed];

            *b = Some(new_vec);

            b.as_mut().unwrap()
        };

        if v.len() < needed {
            v.resize(needed, 0.0);
        }

        for ic in 0..in_channels {
            for kh in 0..3 {
                for kw in 0..3 {
                    let k_idx = ic * 9 + kh * 3 + kw;

                    for oc in 0..out_channels {
                        let w_idx = ((oc * in_channels + ic) * 3 + kh) * 3 + kw;

                        // SAFETY: k_idx < k, oc < out_channels = oc_count, v.len() >= needed = k*oc_count.

                        // w_idx is bounded by the weight tensor dimensions.

                        unsafe { *v.get_unchecked_mut(k_idx * oc_count + oc) = *w_ptr.add(w_idx) };
                    }
                }
            }
        }

        v.as_ptr()
    });

    // SAFETY: wt_trans_ptr points to the thread-local buffer resized to `needed = k*oc_count` elements.

    let wt_trans_slice = unsafe { std::slice::from_raw_parts(wt_trans_ptr, k * oc_count) };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let total_work = batch_size * out_height * out_width;

        let x_usize = x_ptr as usize;

        let out_usize = out_ptr as usize;

        let bn_weight_usize = bn_weight_ptr as usize;

        let bn_bias_usize = bn_bias_ptr as usize;

        let bn_mean_usize = bn_mean_ptr as usize;

        let bn_var_usize = bn_var_ptr as usize;

        (0..total_work).into_par_iter().for_each(|idx| {

            let n = idx / (out_height * out_width);

            let pos = idx % (out_height * out_width);

            let oh = pos / out_width;

            let ow = pos % out_width;

            let x_ptr = x_usize as *const f32;

            let out_ptr = out_usize as *mut f32;

                    let bn_weight_ptr = bn_weight_usize as *const f32;

                    let bn_bias_ptr = bn_bias_usize as *const f32;

                    let bn_mean_ptr = bn_mean_usize as *const f32;

                    let bn_var_ptr = bn_var_usize as *const f32;



                    S_BUF.with(|s| {

                        let mut s_buf = s.borrow_mut();

                        if s_buf.len() < k {

                            s_buf.resize(k, 0.0);

                        }

                        let s = &mut s_buf[..k];

                        let mut k_idx = 0;

                        for ic in 0..in_channels {

                            for kh in 0..3 {

                                for kw in 0..3 {

                                    let ih_s = (oh as isize) + (kh as isize) - 1;

                                    let iw_s = (ow as isize) + (kw as isize) - 1;

                                    s[k_idx] =

                                        if ih_s >= 0 && ih_s < in_h && iw_s >= 0 && iw_s < in_w {

                                            let ih = ih_s as usize;

                                            let iw = iw_s as usize;

                                            let x_idx = ((n * in_channels + ic) * in_height + ih)

                                                * in_width

                                                + iw;

                                            // SAFETY: x_idx computed from n, ic, ih, iw is bounded by batch_size*in_channels*in_height*in_width = x tensor element count.

                                            unsafe { *x_ptr.add(x_idx) }

                                        } else {

                                            0.0

                                        };

                                    k_idx += 1;

                                }

                            }

                        }



                        let mut oc = 0;

                        while oc + 8 <= out_channels {

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]

                            {

                                use wide::f32x8;

                                let mut acc = f32x8::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let w_ptr_k =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_simd = from_slice_unaligned_f32x8(unsafe {

                                        std::slice::from_raw_parts(w_ptr_k, 8)

                                    });

                                    acc += w_simd * f32x8::splat(x_val);

                                }



                                let bias_vec = if let Some(ref b) = bias_data {

                                    let b_slice = unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 8)

                                    };

                                    from_slice_unaligned_f32x8(b_slice)

                                } else if let Some(b) = bias_scalar {

                                    f32x8::splat(b)

                                } else {

                                    f32x8::ZERO

                                };



                                let res_arr = (acc + bias_vec).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                for i in 0..8 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_arr[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(all(feature = "simd", target_arch = "aarch64"))]

                            {

                                use wide::f32x4;

                                let mut acc_lo = f32x4::ZERO;

                                let mut acc_hi = f32x4::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let base_ptr =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_lo = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr, 4)

                                    });

                                    acc_lo = acc_lo + w_lo * f32x4::splat(x_val);

                                    let w_hi = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr.add(4), 4)

                                    });

                                    acc_hi = acc_hi + w_hi * f32x4::splat(x_val);

                                }



                                let bias_lo = if let Some(ref b) = bias_data {

                                    // SAFETY: oc < out_channels and oc+4 <= out_channels, so b[oc..oc+4] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };

                                let bias_hi = if let Some(ref b) = bias_data {

                                    // SAFETY: oc+4 < out_channels, so b[oc+4..oc+8] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc + 4), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };



                                let res_lo = (acc_lo + bias_lo).to_array();

                                let res_hi = (acc_hi + bias_hi).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                for i in 0..4 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_lo[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                                for i in 0..4 {

                                    let oc_i = oc + 4 + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_hi[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    let out_idx = out_idx_base + (4 + i) * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(not(all(

                                feature = "simd",

                                any(target_arch = "x86_64", target_arch = "aarch64")

                            )))]

                            {

                                for oc_off in 0..8 {

                                    let oc_i = oc + oc_off;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut sum = 0.0f32;

                                    for k_idx in 0..k {

                                        let x_val = s[k_idx];

                                        let w_val = unsafe {

                                            *wt_trans_slice.get_unchecked(k_idx * oc_count + oc_i)

                                        };

                                        sum += x_val * w_val;

                                    }

                                    let bias_val = bias_scalar.unwrap_or_else(|| {

                                        bias_data.as_ref().map(|b| b[oc_i]).unwrap_or(0.0)

                                    });

                                    let mut v = sum + bias_val;

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    let out_idx = ((n * out_channels + oc_i) * out_height + oh)

                                        * out_width

                                        + ow;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            oc += 8;

                        }



                        while oc < out_channels {

                            let mut sum = 0.0f32;

                            for k_idx in 0..k {

                                let x_val = s[k_idx];

                                let w_val =

                                    // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() since oc < out_channels.

                                    unsafe { *wt_trans_slice.get_unchecked(k_idx * oc_count + oc) };

                                sum += x_val * w_val;

                            }

                            let bias_val = bias_scalar.unwrap_or_else(|| {

                                bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0)

                            });

                            let mut v = sum + bias_val;

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let mean = unsafe { *bn_mean_ptr.add(oc) };

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let var = unsafe { *bn_var_ptr.add(oc) };

                            let inv_std = 1.0 / (var + bn_eps).sqrt();

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc) }

                                // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                + unsafe { *bn_bias_ptr.add(oc) };

                            let out_idx =

                                ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                            unsafe {

                                *out_ptr.add(out_idx) = v;

                            }

                            oc += 1;

                        }

                    });

        })
    }

    #[cfg(not(feature = "parallel"))]
    {
        let total_work = batch_size * out_height * out_width;

        for idx in 0..total_work {

            let n = idx / (out_height * out_width);

            let pos = idx % (out_height * out_width);

            let oh = pos / out_width;

            let ow = pos % out_width;

                    S_BUF.with(|s| {

                        let mut s_buf = s.borrow_mut();

                        if s_buf.len() < k {

                            s_buf.resize(k, 0.0);

                        }

                        let s = &mut s_buf[..k];

                        let mut k_idx = 0;

                        for ic in 0..in_channels {

                            for kh in 0..3 {

                                for kw in 0..3 {

                                    let ih_s = (oh as isize) + (kh as isize) - 1;

                                    let iw_s = (ow as isize) + (kw as isize) - 1;

                                    s[k_idx] =

                                        if ih_s >= 0 && ih_s < in_h && iw_s >= 0 && iw_s < in_w {

                                            let ih = ih_s as usize;

                                            let iw = iw_s as usize;

                                            let x_idx = ((n * in_channels + ic) * in_height + ih)

                                                * in_width

                                                + iw;

                                            // SAFETY: x_idx computed from n, ic, ih, iw is bounded by batch_size*in_channels*in_height*in_width = x tensor element count.

                                            unsafe { *x_ptr.add(x_idx) }

                                        } else {

                                            0.0

                                        };

                                    k_idx += 1;

                                }

                            }

                        }



                        let mut oc = 0;

                        while oc + 8 <= out_channels {

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]

                            {

                                use wide::f32x8;

                                let mut acc = f32x8::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let w_ptr_k =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_simd = from_slice_unaligned_f32x8(unsafe {

                                        std::slice::from_raw_parts(w_ptr_k, 8)

                                    });

                                    acc += w_simd * f32x8::splat(x_val);

                                }



                                let bias_vec = if let Some(ref b) = bias_data {

                                    let b_slice = unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 8)

                                    };

                                    from_slice_unaligned_f32x8(b_slice)

                                } else if let Some(b) = bias_scalar {

                                    f32x8::splat(b)

                                } else {

                                    f32x8::ZERO

                                };



                                let res_arr = (acc + bias_vec).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                for i in 0..8 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_arr[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(all(feature = "simd", target_arch = "aarch64"))]

                            {

                                use wide::f32x4;

                                let mut acc_lo = f32x4::ZERO;

                                let mut acc_hi = f32x4::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let base_ptr =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_lo = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr, 4)

                                    });

                                    acc_lo = acc_lo + w_lo * f32x4::splat(x_val);

                                    let w_hi = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr.add(4), 4)

                                    });

                                    acc_hi = acc_hi + w_hi * f32x4::splat(x_val);

                                }



                                let bias_lo = if let Some(ref b) = bias_data {

                                    // SAFETY: oc < out_channels and oc+4 <= out_channels, so b[oc..oc+4] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };

                                let bias_hi = if let Some(ref b) = bias_data {

                                    // SAFETY: oc+4 < out_channels, so b[oc+4..oc+8] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc + 4), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };



                                let res_lo = (acc_lo + bias_lo).to_array();

                                let res_hi = (acc_hi + bias_hi).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                for i in 0..4 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_lo[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                                for i in 0..4 {

                                    let oc_i = oc + 4 + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_hi[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    let out_idx = out_idx_base + (4 + i) * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(not(all(

                                feature = "simd",

                                any(target_arch = "x86_64", target_arch = "aarch64")

                            )))]

                            {

                                for oc_off in 0..8 {

                                    let oc_i = oc + oc_off;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut sum = 0.0f32;

                                    for k_idx in 0..k {

                                        let x_val = s[k_idx];

                                        let w_val = unsafe {

                                            *wt_trans_slice.get_unchecked(k_idx * oc_count + oc_i)

                                        };

                                        sum += x_val * w_val;

                                    }

                                    let bias_val = bias_scalar.unwrap_or_else(|| {

                                        bias_data.as_ref().map(|b| b[oc_i]).unwrap_or(0.0)

                                    });

                                    let mut v = sum + bias_val;

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    let out_idx = ((n * out_channels + oc_i) * out_height + oh)

                                        * out_width

                                        + ow;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            oc += 8;

                        }



                        while oc < out_channels {

                            let mut sum = 0.0f32;

                            for k_idx in 0..k {

                                let x_val = s[k_idx];

                                let w_val =

                                    // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() since oc < out_channels.

                                    unsafe { *wt_trans_slice.get_unchecked(k_idx * oc_count + oc) };

                                sum += x_val * w_val;

                            }

                            let bias_val = bias_scalar.unwrap_or_else(|| {

                                bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0)

                            });

                            let mut v = sum + bias_val;

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let mean = unsafe { *bn_mean_ptr.add(oc) };

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let var = unsafe { *bn_var_ptr.add(oc) };

                            let inv_std = 1.0 / (var + bn_eps).sqrt();

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc) }

                                // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                + unsafe { *bn_bias_ptr.add(oc) };

                            let out_idx =

                                ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                            unsafe {

                                *out_ptr.add(out_idx) = v;

                            }

                            oc += 1;

                        }

                    });
        }
    }

    output
}

pub unsafe fn fused_conv_bn_silu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];

    let w = args[1];

    let mut idx = 2;

    let bias = if args.len() > idx && args[idx].numel() > 0 {
        let b = Some(args[idx]);

        idx += 1;

        b
    } else {
        None
    };

    if args.len() < idx + 9 {
        panic!("fused_conv_bn_silu: insufficient arguments");
    }

    let bn_weight = args[idx];

    idx += 1;

    let bn_bias = args[idx];

    idx += 1;

    let bn_running_mean = args[idx];

    idx += 1;

    let bn_running_var = args[idx];

    idx += 1;

    let stride_t = args[idx];

    idx += 1;

    let padding_t = args[idx];

    idx += 1;

    let dilation_t = args[idx];

    idx += 1;

    let groups_t = args[idx];

    idx += 1;

    let eps_t = args[idx];

    let stride = stride_t.item() as i64;

    let padding = padding_t.item() as i64;

    let dilation = dilation_t.item() as i64;

    let groups = groups_t.item() as i64;

    let eps = eps_t.item();

    // Only support 3x3 kernel with stride=1, padding=1, dilation=1, groups=1

    let w_shape = w.shape_ref();

    let kernel_h = w_shape[2] as usize;

    let kernel_w = w_shape[3] as usize;

    if kernel_h != 3 || kernel_w != 3 || stride != 1 || padding != 1 || dilation != 1 || groups != 1
    {
        panic!("fused_conv_bn_silu: only 3x3 kernel with stride=1, padding=1, dilation=1, groups=1 is supported");
    }

    let x_shape = x.shape_ref();

    let batch_size = x_shape[0] as usize;

    let in_channels = x_shape[1] as usize;

    let in_height = x_shape[2] as usize;

    let in_width = x_shape[3] as usize;

    let out_channels = w_shape[0] as usize;

    // Output spatial dimensions are the same as input for stride=1, padding=1, 3x3 kernel

    let out_height = in_height;

    let out_width = in_width;

    vec![fused_conv_bn_silu_3x3_direct(
        x,
        w,
        bias,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        eps,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
    )]
}

pub unsafe fn fused_conv_bn_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];

    let w = args[1];

    let mut idx = 2;

    let bias = if args.len() > idx && args[idx].numel() > 0 {
        let b = Some(args[idx]);

        idx += 1;

        b
    } else {
        None
    };

    if args.len() < idx + 9 {
        panic!("fused_conv_bn: insufficient arguments");
    }

    let bn_weight = args[idx];

    idx += 1;

    let bn_bias = args[idx];

    idx += 1;

    let bn_running_mean = args[idx];

    idx += 1;

    let bn_running_var = args[idx];

    idx += 1;

    let stride_t = args[idx];

    idx += 1;

    let padding_t = args[idx];

    idx += 1;

    let dilation_t = args[idx];

    idx += 1;

    let groups_t = args[idx];

    idx += 1;

    let eps_t = args[idx];

    let stride = stride_t.item() as i64;

    let padding = padding_t.item() as i64;

    let dilation = dilation_t.item() as i64;

    let groups = groups_t.item() as i64;

    let eps = eps_t.item();

    // Only support 3x3 kernel with stride=1, padding=1, dilation=1, groups=1

    let w_shape = w.shape_ref();

    let kernel_h = w_shape[2] as usize;

    let kernel_w = w_shape[3] as usize;

    if kernel_h != 3 || kernel_w != 3 || stride != 1 || padding != 1 || dilation != 1 || groups != 1
    {
        panic!("fused_conv_bn: only 3x3 kernel with stride=1, padding=1, dilation=1, groups=1 is supported");
    }

    let x_shape = x.shape_ref();

    let batch_size = x_shape[0] as usize;

    let in_channels = x_shape[1] as usize;

    let in_height = x_shape[2] as usize;

    let in_width = x_shape[3] as usize;

    let out_channels = w_shape[0] as usize;

    // Output spatial dimensions are the same as input for stride=1, padding=1, 3x3 kernel

    let out_height = in_height;

    let out_width = in_width;

    vec![fused_conv_bn_3x3_direct(
        x,
        w,
        bias,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        eps,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
    )]
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn depthwise_conv2d(
    x: &Tensor,

    w: &Tensor,

    bias: Option<&Tensor>,

    stride: usize,

    padding: usize,

    dilation: usize,

    out_height: usize,

    out_width: usize,
) -> Tensor {
    let x_shape = x.shape_ref();

    let batch_size = x_shape[0] as usize;

    let in_channels = x_shape[1] as usize;

    let in_height = x_shape[2] as usize;

    let in_width = x_shape[3] as usize;

    let w_shape = w.shape_ref();

    let kernel_height = w_shape[2] as usize;

    let kernel_width = w_shape[3] as usize;

    let output_shape = vec![
        batch_size as i64,
        in_channels as i64,
        out_height as i64,
        out_width as i64,
    ];

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());

    let output_inner = Arc::make_mut(&mut output.inner);

    let output_storage = Arc::make_mut(&mut output_inner.storage);

    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };

    let out_data = Arc::make_mut(&mut cpu_storage.data);

    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    // Use direct pointers instead of copying data

    let w_ptr = w.data_ptr() as *const f32;

    let x_ptr = x.data_ptr() as *const f32;

    let bias_ptr = bias.map(|b| b.data_ptr() as *const f32);

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let total = batch_size * in_channels * out_height * out_width;

        let x_usize = x_ptr as usize;

        let w_usize = w_ptr as usize;

        let bias_usize = bias_ptr.map(|p| p as usize);

        let results: Vec<f32> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let n = idx / (in_channels * out_height * out_width);

                let rem = idx % (in_channels * out_height * out_width);

                let ic = rem / (out_height * out_width);

                let rem2 = rem % (out_height * out_width);

                let oh = rem2 / out_width;

                let ow = rem2 % out_width;

                let mut sum = 0.0f32;

                for kh in 0..kernel_height {
                    for kw in 0..kernel_width {
                        let ih = (oh * stride + kh * dilation).wrapping_sub(padding);

                        let iw = (ow * stride + kw * dilation).wrapping_sub(padding);

                        if ih < in_height && iw < in_width {
                            let x_idx = ((n * in_channels + ic) * in_height + ih) * in_width + iw;

                            let w_idx = ((ic * kernel_height) + kh) * kernel_width + kw;

                            unsafe {
                                sum += *(x_usize as *const f32).add(x_idx)
                                    * *(w_usize as *const f32).add(w_idx);
                            }
                        }
                    }
                }

                if let Some(b) = bias_usize {
                    unsafe {
                        sum += *(b as *const f32).add(ic);
                    }
                }

                sum
            })
            .collect();

        unsafe {
            std::ptr::copy_nonoverlapping(results.as_ptr(), out_ptr, total);
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        for n in 0..batch_size {
            for ic in 0..in_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0f32;

                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = (oh * stride + kh * dilation).wrapping_sub(padding);

                                let iw = (ow * stride + kw * dilation).wrapping_sub(padding);

                                if ih < in_height && iw < in_width {
                                    let x_idx =
                                        ((n * in_channels + ic) * in_height + ih) * in_width + iw;

                                    let w_idx = ((ic * kernel_height) + kh) * kernel_width + kw;

                                    sum += unsafe { *x_ptr.add(x_idx) * *w_ptr.add(w_idx) };
                                }
                            }
                        }

                        if let Some(b) = bias_ptr {
                            sum += unsafe { *b.add(ic) };
                        }

                        let out_idx = ((n * in_channels + ic) * out_height + oh) * out_width + ow;

                        // SAFETY: out_idx bounded by tensor dimensions.

                        unsafe { *out_ptr.add(out_idx) = sum };
                    }
                }
            }
        }
    }

    output
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn conv2d_im2col(
    x: &Tensor,

    w: &Tensor,

    bias: Option<&Tensor>,

    stride: usize,

    padding: usize,

    dilation: usize,

    out_height: usize,

    out_width: usize,

    batch_size: usize,

    in_channels: usize,

    out_channels: usize,

    kernel_height: usize,

    kernel_width: usize,

    groups: usize,
) -> Tensor {
    let x_shape = x.shape_ref();

    let in_height = x_shape[2] as usize;

    let in_width = x_shape[3] as usize;

    let col_rows = batch_size * out_height * out_width;

    let in_channels_per_group = in_channels / groups;

    let out_channels_per_group = out_channels / groups;

    let cols_per_group = in_channels_per_group * kernel_height * kernel_width;

    let col_size = col_rows * cols_per_group;

    let x_ptr = x.data_ptr() as *const f32;

    // Use optimized matrix multiplication for large matrices

    // Threshold: use GEMM when all dimensions >= 1

    const GEMM_MIN_SIZE: usize = 1;

    let use_gemm = col_rows >= GEMM_MIN_SIZE
        && out_channels_per_group >= GEMM_MIN_SIZE
        && cols_per_group >= GEMM_MIN_SIZE;

    let gemm_size = if use_gemm { col_rows * out_channels } else { 0 };

    let total_scratch = col_size + gemm_size;

    // Borrow the thread-local scratch buffer, growing only on the cold path.

    CONV_SCRATCH.with(|scratch| {
        let mut buf = scratch.borrow_mut();

        if buf.len() < total_scratch {
            buf.resize(total_scratch);
        }

        // Zero the im2col buffer for this group before splitting

        buf.data[..col_size].fill(0.0);

        let (col_buf, _gemm_buf) = buf.split_at_mut(col_size);

        let output_shape = vec![
            batch_size as i64,
            out_channels as i64,
            out_height as i64,
            out_width as i64,
        ];

        let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());

        let output_inner = Arc::make_mut(&mut output.inner);

        let output_storage = Arc::make_mut(&mut output_inner.storage);

        let Storage::Cpu(cpu_storage) = output_storage else {
            panic!("Expected CPU storage");
        };

        let out_data = Arc::make_mut(&mut cpu_storage.data);

        let out_ptr = out_data.as_mut_ptr() as *mut f32;

        // Weight data: direct borrow, no copy.

        let w_data: &[f32] = unsafe {
            let w_ptr = w.data_ptr() as *const f32;

            std::slice::from_raw_parts(
                w_ptr,
                out_channels * in_channels * kernel_height * kernel_width / groups,
            )
        };

        let bias_data: Option<Vec<f32>> = if let Some(b) = bias {
            if b.numel() == 1 {
                let bias_val = b.item();

                Some(vec![bias_val; out_channels])
            } else {
                let b_ptr = b.data_ptr() as *const f32;

                // SAFETY: bias tensor has out_channels elements as verified by tensor shape.

                Some(unsafe { std::slice::from_raw_parts(b_ptr, out_channels).to_vec() })
            }
        } else {
            None
        };

        // --- Process each group independently ---

        // For each group, im2col extracts the group's input channels, then GEMM or

        // scalar loop computes the group's output channels.

        for group in 0..groups {
            let col_data: &mut [f32] = col_buf;

            let in_ch_start = group * in_channels_per_group;

            let in_ch_end = in_ch_start + in_channels_per_group;

            // --- im2col extraction for this group's channels ---

            // Optimized loop order: process input channels in blocks first for better cache efficiency.

            // This maintains NCHW->NCHW locality by processing channels contiguously before spatial dims.

            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;

                let x_usize = x_ptr as usize;

                col_data
                    .par_chunks_mut(cols_per_group)
                    .with_min_len(128)
                    .enumerate()
                    .for_each(|(row, col_chunk)| {
                        let n = row / (out_height * out_width);

                        let rem = row % (out_height * out_width);

                        let oh = rem / out_width;

                        let ow = rem % out_width;

                        // Fast path: dilation=1, entire kernel patch in-bounds.

                        let fast_path = stride == 1 && dilation == 1;

                        for ic_idx in in_ch_start..in_ch_end {
                            let ic = ic_idx;

                            let col_base = (ic_idx - in_ch_start) * kernel_height * kernel_width;

                            if fast_path {
                                // Optimized path: dilation=1

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
                                            // Entire row in-bounds: use memcpy

                                            let x_iw_start = iw_start - padding;

                                            let x_src = x_row_base + x_iw_start;

                                            let col_dst_base = col_base + kh * kernel_width;

                                            unsafe {
                                                std::ptr::copy_nonoverlapping(
                                                    (x_usize as *const f32).add(x_src),
                                                    col_chunk.as_mut_ptr().add(col_dst_base),
                                                    kernel_width,
                                                );
                                            }
                                        } else {
                                            // Boundary: per-element

                                            for kw in 0..kernel_width {
                                                let iw = ow + kw;

                                                if iw >= padding && iw < padding + in_width {
                                                    let x_iw = iw - padding;

                                                    let col_col = col_base + kh * kernel_width + kw;

                                                    unsafe {
                                                        col_chunk[col_col] = *((x_usize
                                                            as *const f32)
                                                            .add(x_row_base + x_iw));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                // General path: arbitrary stride/dilation
                                // Note: ih/iw are computed in input-coordinate space via wrapping_sub;
                                // no further padding adjustment is needed.
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

                                                unsafe {
                                                    col_chunk[col_col] = *((x_usize as *const f32)
                                                        .add(x_row_base + iw));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });
            }

            #[cfg(not(feature = "parallel"))]
            {
                for n in 0..batch_size {
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let col_row = (n * out_height + oh) * out_width + ow;

                            let col_chunk = &mut col_data
                                [col_row * cols_per_group..(col_row + 1) * cols_per_group];

                            let fast_path = stride == 1 && dilation == 1;

                            for ic_idx in in_ch_start..in_ch_end {
                                let ic = ic_idx;

                                let col_base =
                                    (ic_idx - in_ch_start) * kernel_height * kernel_width;

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

                                                unsafe {
                                                    std::ptr::copy_nonoverlapping(
                                                        x_ptr.add(x_src),
                                                        col_chunk.as_mut_ptr().add(col_dst_base),
                                                        kernel_width,
                                                    );
                                                }
                                            } else {
                                                for kw in 0..kernel_width {
                                                    let iw = ow + kw;

                                                    if iw >= padding && iw < padding + in_width {
                                                        let x_iw = iw - padding;

                                                        let col_col =
                                                            col_base + kh * kernel_width + kw;

                                                        col_chunk[col_col] =

                                                        // SAFETY: x_row_base + x_iw computed from input indices is bounded by x tensor element count.
                                                        unsafe { *x_ptr.add(x_row_base + x_iw) };
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    for kh in 0..kernel_height {
                                        let ih =
                                            (oh * stride + kh * dilation).wrapping_sub(padding);

                                        if ih < in_height {
                                            let x_row_base =
                                                (n * in_channels + ic) * in_height * in_width
                                                    + ih * in_width;

                                            for kw in 0..kernel_width {
                                                let iw = (ow * stride + kw * dilation)
                                                    .wrapping_sub(padding);

                                                if iw < in_width {
                                                    let col_col = col_base + kh * kernel_width + kw;

                                                    col_chunk[col_col] =

                                                    // SAFETY: x_row_base + iw computed from input indices is bounded by x tensor element count.
                                                    unsafe { *x_ptr.add(x_row_base + iw) };
                                                }
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

            let _oc_end = oc_start + out_channels_per_group;

            // Weight slice for this group: rows oc_start..oc_end, each with cols_per_group elements

            let w_group_start = oc_start * cols_per_group;

            if use_gemm {
                // Use a temporary buffer for this group's GEMM output (contiguous row-major)

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

                // Scatter group_out into the output tensor in NCHW layout with bias

                let spatial = out_height * out_width;

                let block_rows = 64;

                if let Some(b) = &bias_data {
                    let bias_ptr = b.as_ptr();

                    for n in 0..batch_size {
                        for sp_block in (0..spatial).step_by(block_rows) {
                            let blk = std::cmp::min(block_rows, spatial - sp_block);

                            // Write blk rows from group_out to output with bias

                            for oc_idx_in_group in 0..out_channels_per_group {
                                let oc_idx = oc_start + oc_idx_in_group;

                                let bias_val = unsafe { *bias_ptr.add(oc_idx) };

                                for i in 0..blk {
                                    let row = n * spatial + sp_block + i;

                                    let out_idx =
                                        ((n * out_channels + oc_idx) * spatial + sp_block) + i;

                                    let val =
                                        group_out[row * out_channels_per_group + oc_idx_in_group];

                                    unsafe {
                                        *out_ptr.add(out_idx) = val + bias_val;
                                    }
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

                                    let val =
                                        group_out[row * out_channels_per_group + oc_idx_in_group];

                                    unsafe {
                                        *out_ptr.add(out_idx) = val;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                for oc_in_group in 0..out_channels_per_group {
                    let oc = oc_start + oc_in_group;

                    let w_row = &w_data[(oc_start + oc_in_group) * cols_per_group
                        ..(oc_start + oc_in_group + 1) * cols_per_group];

                    let bias_val = bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0);

                    for row in 0..col_rows {
                        let col_row = &col_slice[row * cols_per_group..(row + 1) * cols_per_group];

                        let sum = unsafe { simd_dot_product(col_row, w_row, cols_per_group) };

                        let n = row / (out_height * out_width);

                        let rem = row % (out_height * out_width);

                        let oh = rem / out_width;

                        let ow = rem % out_width;

                        let out_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                        // SAFETY: out_idx bounded by tensor dimensions.

                        unsafe { *out_ptr.add(out_idx) = sum + bias_val };
                    }
                }
            }
        }

        output
    })
}

pub unsafe fn conv_transpose2d_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];

    let weight = args[1];

    let stride = args[2].item() as i64;

    let padding = args[3].item() as i64;

    let x_shape = x.shape_ref();

    let w_shape = weight.shape_ref();

    let batch = x_shape[0];

    let in_channels = x_shape[1];

    let h_in = x_shape[2];

    let w_in = x_shape[3];

    let out_channels = w_shape[1];

    let kernel_h = w_shape[2];

    let kernel_w = w_shape[3];

    let h_out = (h_in - 1) * stride - 2 * padding + kernel_h;

    let w_out = (w_in - 1) * stride - 2 * padding + kernel_w;

    let output_shape = vec![batch, out_channels, h_out, w_out];

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());

    let x_data = x.as_f32_slice();

    let w_data = weight.as_f32_slice();

    let out_ptr = output.data_ptr_f32_mut();

    for b in 0..batch as usize {
        for ic in 0..in_channels as usize {
            for oc in 0..out_channels as usize {
                for hi in 0..h_in as usize {
                    for wi in 0..w_in as usize {
                        let x_val = x_data[b
                            * (in_channels as usize * h_in as usize * w_in as usize)
                            + ic * (h_in as usize * w_in as usize)
                            + hi * w_in as usize
                            + wi];

                        for kh in 0..kernel_h as usize {
                            for kw in 0..kernel_w as usize {
                                let ho = hi as i64 * stride - padding + kh as i64;

                                let wo = wi as i64 * stride - padding + kw as i64;

                                if ho >= 0 && ho < h_out && wo >= 0 && wo < w_out {
                                    let w_idx = ic
                                        * (out_channels as usize
                                            * kernel_h as usize
                                            * kernel_w as usize)
                                        + oc * (kernel_h as usize * kernel_w as usize)
                                        + kh * kernel_w as usize
                                        + kw;

                                    let out_idx = b
                                        * (out_channels as usize * h_out as usize * w_out as usize)
                                        + oc * (h_out as usize * w_out as usize)
                                        + ho as usize * w_out as usize
                                        + wo as usize;

                                    unsafe {
                                        *out_ptr.add(out_idx) += x_val * w_data[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    vec![output]
}

pub unsafe fn conv1d_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];

    let weight = args[1];

    let stride = args[2].item() as i64;

    let padding = args[3].item() as i64;

    let dilation = args[4].item() as i64;

    let x_shape = x.shape_ref();

    let w_shape = weight.shape_ref();

    let batch = x_shape[0];

    let in_channels = x_shape[1];

    let l_in = x_shape[2];

    let out_channels = w_shape[0];

    let kernel_l = w_shape[2];

    let l_out = ((l_in + 2 * padding - dilation * (kernel_l - 1) - 1) / stride) + 1;

    let output_shape = vec![batch, out_channels, l_out];

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());

    let x_data = x.as_f32_slice();

    let w_data = weight.as_f32_slice();

    let out_ptr = output.data_ptr_f32_mut();

    for b in 0..batch as usize {
        for oc in 0..out_channels as usize {
            for ic in 0..in_channels as usize {
                for lo in 0..l_out as usize {
                    for kl in 0..kernel_l as usize {
                        let li = lo as i64 * stride - padding + kl as i64 * dilation;

                        if li >= 0 && li < l_in {
                            let x_idx = b * (in_channels as usize * l_in as usize)
                                + ic * l_in as usize
                                + li as usize;

                            let w_idx = oc * (in_channels as usize * kernel_l as usize)
                                + ic * kernel_l as usize
                                + kl;

                            let out_idx = b * (out_channels as usize * l_out as usize)
                                + oc * l_out as usize
                                + lo;

                            unsafe {
                                *out_ptr.add(out_idx) += x_data[x_idx] * w_data[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    vec![output]
}

pub unsafe fn conv3d_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];

    let weight = args[1];

    let stride = args[2].item() as i64;

    let padding = args[3].item() as i64;

    let dilation = args[4].item() as i64;

    let x_shape = x.shape_ref();

    let w_shape = weight.shape_ref();

    let batch = x_shape[0];

    let in_channels = x_shape[1];

    let d_in = x_shape[2];

    let h_in = x_shape[3];

    let w_in = x_shape[4];

    let out_channels = w_shape[0];

    let kernel_d = w_shape[2];

    let kernel_h = w_shape[3];

    let kernel_w = w_shape[4];

    let d_out = ((d_in + 2 * padding - dilation * (kernel_d - 1) - 1) / stride) + 1;

    let h_out = ((h_in + 2 * padding - dilation * (kernel_h - 1) - 1) / stride) + 1;

    let w_out = ((w_in + 2 * padding - dilation * (kernel_w - 1) - 1) / stride) + 1;

    let output_shape = vec![batch, out_channels, d_out, h_out, w_out];

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());

    let x_data = x.as_f32_slice();

    let w_data = weight.as_f32_slice();

    let out_ptr = output.data_ptr_f32_mut();

    for b in 0..batch as usize {
        for oc in 0..out_channels as usize {
            for ic in 0..in_channels as usize {
                for do_ in 0..d_out as usize {
                    for ho in 0..h_out as usize {
                        for wo in 0..w_out as usize {
                            for kd in 0..kernel_d as usize {
                                for kh in 0..kernel_h as usize {
                                    for kw in 0..kernel_w as usize {
                                        let di =
                                            do_ as i64 * stride - padding + kd as i64 * dilation;

                                        let hi =
                                            ho as i64 * stride - padding + kh as i64 * dilation;

                                        let wi =
                                            wo as i64 * stride - padding + kw as i64 * dilation;

                                        if di >= 0
                                            && di < d_in
                                            && hi >= 0
                                            && hi < h_in
                                            && wi >= 0
                                            && wi < w_in
                                        {
                                            let x_idx = b
                                                * (in_channels as usize
                                                    * d_in as usize
                                                    * h_in as usize
                                                    * w_in as usize)
                                                + ic * (d_in as usize
                                                    * h_in as usize
                                                    * w_in as usize)
                                                + di as usize * (h_in as usize * w_in as usize)
                                                + hi as usize * w_in as usize
                                                + wi as usize;

                                            let w_idx = oc
                                                * (in_channels as usize
                                                    * kernel_d as usize
                                                    * kernel_h as usize
                                                    * kernel_w as usize)
                                                + ic * (kernel_d as usize
                                                    * kernel_h as usize
                                                    * kernel_w as usize)
                                                + kd * (kernel_h as usize * kernel_w as usize)
                                                + kh * kernel_w as usize
                                                + kw;

                                            let out_idx = b
                                                * (out_channels as usize
                                                    * d_out as usize
                                                    * h_out as usize
                                                    * w_out as usize)
                                                + oc * (d_out as usize
                                                    * h_out as usize
                                                    * w_out as usize)
                                                + do_ * (h_out as usize * w_out as usize)
                                                + ho * w_out as usize
                                                + wo;

                                            unsafe {
                                                *out_ptr.add(out_idx) +=
                                                    x_data[x_idx] * w_data[w_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    vec![output]
}

/// FlashAttention-inspired kernel for memory-efficient attention.
/// Computes softmax(Q @ K^T) @ V using block-wise tiling to avoid
/// materializing the full N×N attention scores matrix.
/// This is mathematically equivalent to standard attention but uses O(N) memory
/// for the attention scores instead of O(N²).
pub unsafe fn flash_attention_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let q = args[0];

    let k = args[1];

    let v = args[2];

    let scale = if args.len() > 3 { args[3].item() } else { 1.0 };

    let causal = if args.len() > 4 {
        args[4].item() as i64 != 0
    } else {
        false
    };

    let q_shape = q.shape_ref();

    let batch = q_shape[0] as usize;

    let num_heads = q_shape[1] as usize;

    let seq_len = q_shape[2] as usize;

    let head_dim = q_shape[3] as usize;

    let q_data = q.as_f32_slice();

    let k_data = k.as_f32_slice();

    let v_data = v.as_f32_slice();

    let block_size = 64.min(seq_len);

    let num_blocks = seq_len.div_ceil(block_size);

    let mut output = vec![0.0f32; batch * num_heads * seq_len * head_dim];

    let mut l = vec![0.0f32; batch * num_heads * seq_len];

    let mut m = vec![f32::NEG_INFINITY; batch * num_heads * seq_len];

    for b in 0..batch {
        for h in 0..num_heads {
            for j in 0..num_blocks {
                let j_start = j * block_size;

                let j_end = (j * block_size + block_size).min(seq_len);

                let block_j_len = j_end - j_start;

                // Load K_j and V_j blocks into local buffers

                let mut k_block = vec![0.0f32; block_j_len * head_dim];

                let mut v_block = vec![0.0f32; block_j_len * head_dim];

                for jj in 0..block_j_len {
                    for d in 0..head_dim {
                        let src = b * num_heads * seq_len * head_dim
                            + h * seq_len * head_dim
                            + (j_start + jj) * head_dim
                            + d;

                        k_block[jj * head_dim + d] = k_data[src];

                        v_block[jj * head_dim + d] = v_data[src];
                    }
                }

                for i in 0..num_blocks {
                    let i_start = i * block_size;

                    let i_end = (i * block_size + block_size).min(seq_len);

                    let block_i_len = i_end - i_start;

                    // Skip blocks above the diagonal for causal attention

                    if causal && i_start > j_start {
                        continue;
                    }

                    // Load Q_i block

                    let mut q_block = vec![0.0f32; block_i_len * head_dim];

                    for ii in 0..block_i_len {
                        for d in 0..head_dim {
                            let src = b * num_heads * seq_len * head_dim
                                + h * seq_len * head_dim
                                + (i_start + ii) * head_dim
                                + d;

                            q_block[ii * head_dim + d] = q_data[src];
                        }
                    }

                    // Compute S_ij = Q_i @ K_j^T * scale

                    let mut s_block = vec![0.0f32; block_i_len * block_j_len];

                    for ii in 0..block_i_len {
                        for jj in 0..block_j_len {
                            // Apply causal mask within the block

                            if causal && (i_start + ii) < (j_start + jj) {
                                s_block[ii * block_j_len + jj] = f32::NEG_INFINITY;

                                continue;
                            }

                            let mut sum = 0.0f32;

                            for d in 0..head_dim {
                                sum += q_block[ii * head_dim + d] * k_block[jj * head_dim + d];
                            }

                            s_block[ii * block_j_len + jj] = sum * scale;
                        }
                    }

                    // Compute row-wise max of S_ij

                    let mut m_block = vec![f32::NEG_INFINITY; block_i_len];

                    for ii in 0..block_i_len {
                        for jj in 0..block_j_len {
                            let s = s_block[ii * block_j_len + jj];

                            if s > m_block[ii] {
                                m_block[ii] = s;
                            }
                        }
                    }

                    // Compute row-wise exp-sum of S_ij

                    let mut l_block = vec![0.0f32; block_i_len];

                    for ii in 0..block_i_len {
                        let mut sum = 0.0f32;

                        for jj in 0..block_j_len {
                            let s = s_block[ii * block_j_len + jj];

                            if s != f32::NEG_INFINITY {
                                sum += (s - m_block[ii]).exp();
                            }
                        }

                        l_block[ii] = sum;
                    }

                    // Update global max and exp-sum, then update output

                    for ii in 0..block_i_len {
                        let global_idx = b * num_heads * seq_len + h * seq_len + i_start + ii;

                        let m_old = m[global_idx];

                        let m_new = m_old.max(m_block[ii]);

                        // Compute new exp-sum

                        let l_new = (m_old - m_new).exp() * l[global_idx]
                            + (m_block[ii] - m_new).exp() * l_block[ii];

                        // Update output: O_new = (exp(m_old - m_new) * O_old + exp(m_block - m_new) * S_ij @ V_j) / l_new

                        for d in 0..head_dim {
                            let out_idx = b * num_heads * seq_len * head_dim
                                + h * seq_len * head_dim
                                + (i_start + ii) * head_dim
                                + d;

                            let old_val = output[out_idx] * (m_old - m_new).exp();

                            let mut new_val = 0.0f32;

                            for jj in 0..block_j_len {
                                let s = s_block[ii * block_j_len + jj];

                                if s != f32::NEG_INFINITY {
                                    new_val += (s - m_block[ii]).exp() * v_block[jj * head_dim + d];
                                }
                            }

                            output[out_idx] =
                                (old_val + new_val * (m_block[ii] - m_new).exp()) / l_new;
                        }

                        m[global_idx] = m_new;

                        l[global_idx] = l_new;
                    }
                }
            }
        }
    }

    let output_shape = vec![
        batch as i64,
        num_heads as i64,
        seq_len as i64,
        head_dim as i64,
    ];

    vec![Tensor::from_vec(output, output_shape)]
}

// --- FusedConvBnReLU (copied from SiLU, activation changed) ---

#[allow(dead_code, clippy::too_many_arguments)]
pub unsafe fn fused_conv_bn_relu_3x3_direct(
    x: &Tensor,

    w: &Tensor,

    bias: Option<&Tensor>,

    bn_weight: &Tensor,

    bn_bias: &Tensor,

    bn_running_mean: &Tensor,

    bn_running_var: &Tensor,

    bn_eps: f32,

    batch_size: usize,

    in_channels: usize,

    out_channels: usize,

    in_height: usize,

    in_width: usize,

    out_height: usize,

    out_width: usize,
) -> Tensor {
    let output_shape = vec![
        batch_size as i64,
        out_channels as i64,
        out_height as i64,
        out_width as i64,
    ];

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());

    let output_inner = Arc::make_mut(&mut output.inner);

    let output_storage = Arc::make_mut(&mut output_inner.storage);

    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };

    let out_data = Arc::make_mut(&mut cpu_storage.data);

    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let x_ptr = x.data_ptr() as *const f32;

    let w_ptr = w.data_ptr() as *const f32;

    let bn_weight_ptr = bn_weight.data_ptr() as *const f32;

    let bn_bias_ptr = bn_bias.data_ptr() as *const f32;

    let bn_mean_ptr = bn_running_mean.data_ptr() as *const f32;

    let bn_var_ptr = bn_running_var.data_ptr() as *const f32;

    let bias_scalar: Option<f32> =
        bias.and_then(|b| if b.numel() == 1 { Some(b.item()) } else { None });

    let bias_data: Option<Vec<f32>> = bias.filter(|b| b.numel() > 1).map(|b| {
        let b_ptr = b.data_ptr() as *const f32;

        // SAFETY: bias tensor has out_channels elements as verified by tensor shape.

        unsafe { std::slice::from_raw_parts(b_ptr, out_channels).to_vec() }
    });

    let in_h = in_height as isize;

    let in_w = in_width as isize;

    // --- Step 1: Get transposed weights from thread-local buffer (recompute per call)

    let k = in_channels * 9;

    let oc_count = out_channels;

    let wt_trans_ptr: *const f32 = WT_TRANS_BUF.with(|buf| {
        let needed = k * oc_count;

        let mut b = buf.borrow_mut();

        let v = if let Some(ref mut v) = &mut *b {
            v
        } else {
            let new_vec = vec![0.0f32; needed];

            *b = Some(new_vec);

            b.as_mut().unwrap()
        };

        if v.len() < needed {
            v.resize(needed, 0.0);
        }

        for ic in 0..in_channels {
            for kh in 0..3 {
                for kw in 0..3 {
                    let k_idx = ic * 9 + kh * 3 + kw;

                    for oc in 0..out_channels {
                        let w_idx = ((oc * in_channels + ic) * 3 + kh) * 3 + kw;

                        // SAFETY: k_idx < k, oc < out_channels = oc_count, v.len() >= needed = k*oc_count.

                        // w_idx is bounded by the weight tensor dimensions.

                        unsafe { *v.get_unchecked_mut(k_idx * oc_count + oc) = *w_ptr.add(w_idx) };
                    }
                }
            }
        }

        v.as_ptr()
    });

    // SAFETY: wt_trans_ptr points to the thread-local buffer resized to `needed = k*oc_count` elements.

    let wt_trans_slice = unsafe { std::slice::from_raw_parts(wt_trans_ptr, k * oc_count) };

    // --- Step 2: Main kernel (parallel over output spatial+batch positions)

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let total_work = batch_size * out_height * out_width;

        let x_usize = x_ptr as usize;

        let out_usize = out_ptr as usize;

        let bn_weight_usize = bn_weight_ptr as usize;

        let bn_bias_usize = bn_bias_ptr as usize;

        let bn_mean_usize = bn_mean_ptr as usize;

        let bn_var_usize = bn_var_ptr as usize;

        (0..total_work).into_par_iter().for_each(|idx| {

            let n = idx / (out_height * out_width);

            let pos = idx % (out_height * out_width);

            let oh = pos / out_width;

            let ow = pos % out_width;

            let x_ptr = x_usize as *const f32;

            let out_ptr = out_usize as *mut f32;

                    let bn_weight_ptr = bn_weight_usize as *const f32;

                    let bn_bias_ptr = bn_bias_usize as *const f32;

                    let bn_mean_ptr = bn_mean_usize as *const f32;

                    let bn_var_ptr = bn_var_usize as *const f32;



                    S_BUF.with(|s| {

                        let mut s_buf = s.borrow_mut();

                        if s_buf.len() < k {

                            s_buf.resize(k, 0.0);

                        }

                        let s = &mut s_buf[..k];

                        let mut k_idx = 0;

                        for ic in 0..in_channels {

                            for kh in 0..3 {

                                for kw in 0..3 {

                                    let ih_s = (oh as isize) + (kh as isize) - 1;

                                    let iw_s = (ow as isize) + (kw as isize) - 1;

                                    s[k_idx] =

                                        if ih_s >= 0 && ih_s < in_h && iw_s >= 0 && iw_s < in_w {

                                            let ih = ih_s as usize;

                                            let iw = iw_s as usize;

                                            let x_idx = ((n * in_channels + ic) * in_height + ih)

                                                * in_width

                                                + iw;

                                            // SAFETY: x_idx computed from n, ic, ih, iw is bounded by batch_size*in_channels*in_height*in_width = x tensor element count.

                                            unsafe { *x_ptr.add(x_idx) }

                                        } else {

                                            0.0

                                        };

                                    k_idx += 1;

                                }

                            }

                        }



                        let mut oc = 0;

                        while oc + 8 <= out_channels {

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]

                            {

                                use wide::f32x8;

                                let mut acc = f32x8::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let w_ptr_k =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_simd = from_slice_unaligned_f32x8(unsafe {

                                        std::slice::from_raw_parts(w_ptr_k, 8)

                                    });

                                    acc += w_simd * f32x8::splat(x_val);

                                }



                                let bias_vec = if let Some(ref b) = bias_data {

                                    let b_slice = unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 8)

                                    };

                                    from_slice_unaligned_f32x8(b_slice)

                                } else if let Some(b) = bias_scalar {

                                    f32x8::splat(b)

                                } else {

                                    f32x8::ZERO

                                };



                                let res_arr = (acc + bias_vec).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                // Apply BN+SiLU per-element and store

                                for i in 0..8 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_arr[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v.max(0.0); // ReLU activation

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(all(feature = "simd", target_arch = "aarch64"))]

                            {

                                use wide::f32x4;

                                let mut acc_lo = f32x4::ZERO;

                                let mut acc_hi = f32x4::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let base_ptr =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_lo = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr, 4)

                                    });

                                    acc_lo = acc_lo + w_lo * f32x4::splat(x_val);

                                    let w_hi = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr.add(4), 4)

                                    });

                                    acc_hi = acc_hi + w_hi * f32x4::splat(x_val);

                                }



                                let bias_lo = if let Some(ref b) = bias_data {

                                    // SAFETY: oc < out_channels and oc+4 <= out_channels, so b[oc..oc+4] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };

                                let bias_hi = if let Some(ref b) = bias_data {

                                    // SAFETY: oc+4 < out_channels, so b[oc+4..oc+8] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc + 4), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };



                                let res_lo = (acc_lo + bias_lo).to_array();

                                let res_hi = (acc_hi + bias_hi).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                // Apply BN+SiLU for lo (4)

                                for i in 0..4 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_lo[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v.max(0.0); // ReLU activation

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                                // Apply BN+SiLU for hi (4)

                                for i in 0..4 {

                                    let oc_i = oc + 4 + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_hi[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v.max(0.0); // ReLU activation

                                    let out_idx = out_idx_base + (4 + i) * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(not(all(

                                feature = "simd",

                                any(target_arch = "x86_64", target_arch = "aarch64")

                            )))]

                            {

                                for oc_off in 0..8 {

                                    let oc_i = oc + oc_off;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut sum = 0.0f32;

                                    for k_idx in 0..k {

                                        let x_val = s[k_idx];

                                        let w_val = unsafe {

                                            *wt_trans_slice.get_unchecked(k_idx * oc_count + oc_i)

                                        };

                                        sum += x_val * w_val;

                                    }

                                    let bias_val = bias_scalar.unwrap_or_else(|| {

                                        bias_data.as_ref().map(|b| b[oc_i]).unwrap_or(0.0)

                                    });

                                    let mut v = sum + bias_val;

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v.max(0.0); // ReLU activation

                                    let out_idx = ((n * out_channels + oc_i) * out_height + oh)

                                        * out_width

                                        + ow;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            oc += 8;

                        }



                        // Remainder channels (scalar)

                        while oc < out_channels {

                            let mut sum = 0.0f32;

                            for k_idx in 0..k {

                                let x_val = s[k_idx];

                                let w_val =

                                    // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() since oc < out_channels.

                                    unsafe { *wt_trans_slice.get_unchecked(k_idx * oc_count + oc) };

                                sum += x_val * w_val;

                            }

                            let bias_val = bias_scalar.unwrap_or_else(|| {

                                bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0)

                            });

                            let mut v = sum + bias_val;

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let mean = unsafe { *bn_mean_ptr.add(oc) };

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let var = unsafe { *bn_var_ptr.add(oc) };

                            let inv_std = 1.0 / (var + bn_eps).sqrt();

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc) }

                                // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                + unsafe { *bn_bias_ptr.add(oc) };

                            v = v.max(0.0); // ReLU activation

                            let out_idx =

                                ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                            unsafe {

                                *out_ptr.add(out_idx) = v;

                            }

                            oc += 1;

                        }

                    });

        })
    }

    #[cfg(not(feature = "parallel"))]
    {
        let total_work = batch_size * out_height * out_width;

        for idx in 0..total_work {

            let n = idx / (out_height * out_width);

            let pos = idx % (out_height * out_width);

            let oh = pos / out_width;

            let ow = pos % out_width;

                    S_BUF.with(|s| {

                        let mut s_buf = s.borrow_mut();

                        if s_buf.len() < k {

                            s_buf.resize(k, 0.0);

                        }

                        let s = &mut s_buf[..k];

                        let mut k_idx = 0;

                        for ic in 0..in_channels {

                            for kh in 0..3 {

                                for kw in 0..3 {

                                    let ih_s = (oh as isize) + (kh as isize) - 1;

                                    let iw_s = (ow as isize) + (kw as isize) - 1;

                                    s[k_idx] =

                                        if ih_s >= 0 && ih_s < in_h && iw_s >= 0 && iw_s < in_w {

                                            let ih = ih_s as usize;

                                            let iw = iw_s as usize;

                                            let x_idx = ((n * in_channels + ic) * in_height + ih)

                                                * in_width

                                                + iw;

                                            // SAFETY: x_idx computed from n, ic, ih, iw is bounded by batch_size*in_channels*in_height*in_width = x tensor element count.

                                            unsafe { *x_ptr.add(x_idx) }

                                        } else {

                                            0.0

                                        };

                                    k_idx += 1;

                                }

                            }

                        }



                        let mut oc = 0;

                        while oc + 8 <= out_channels {

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]

                            {

                                use wide::f32x8;

                                let mut acc = f32x8::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let w_ptr_k =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_simd = from_slice_unaligned_f32x8(unsafe {

                                        std::slice::from_raw_parts(w_ptr_k, 8)

                                    });

                                    acc += w_simd * f32x8::splat(x_val);

                                }



                                let bias_vec = if let Some(ref b) = bias_data {

                                    let b_slice = unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 8)

                                    };

                                    from_slice_unaligned_f32x8(b_slice)

                                } else if let Some(b) = bias_scalar {

                                    f32x8::splat(b)

                                } else {

                                    f32x8::ZERO

                                };



                                let res_arr = (acc + bias_vec).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                // Apply BN+SiLU per-element and store

                                for i in 0..8 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_arr[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v.max(0.0); // ReLU activation

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(all(feature = "simd", target_arch = "aarch64"))]

                            {

                                use wide::f32x4;

                                let mut acc_lo = f32x4::ZERO;

                                let mut acc_hi = f32x4::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let base_ptr =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_lo = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr, 4)

                                    });

                                    acc_lo = acc_lo + w_lo * f32x4::splat(x_val);

                                    let w_hi = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr.add(4), 4)

                                    });

                                    acc_hi = acc_hi + w_hi * f32x4::splat(x_val);

                                }



                                let bias_lo = if let Some(ref b) = bias_data {

                                    // SAFETY: oc < out_channels and oc+4 <= out_channels, so b[oc..oc+4] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };

                                let bias_hi = if let Some(ref b) = bias_data {

                                    // SAFETY: oc+4 < out_channels, so b[oc+4..oc+8] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc + 4), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };



                                let res_lo = (acc_lo + bias_lo).to_array();

                                let res_hi = (acc_hi + bias_hi).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                // Apply BN+SiLU for lo (4)

                                for i in 0..4 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_lo[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v.max(0.0); // ReLU activation

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                                // Apply BN+SiLU for hi (4)

                                for i in 0..4 {

                                    let oc_i = oc + 4 + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_hi[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v.max(0.0); // ReLU activation

                                    let out_idx = out_idx_base + (4 + i) * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(not(all(

                                feature = "simd",

                                any(target_arch = "x86_64", target_arch = "aarch64")

                            )))]

                            {

                                for oc_off in 0..8 {

                                    let oc_i = oc + oc_off;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut sum = 0.0f32;

                                    for k_idx in 0..k {

                                        let x_val = s[k_idx];

                                        let w_val = unsafe {

                                            *wt_trans_slice.get_unchecked(k_idx * oc_count + oc_i)

                                        };

                                        sum += x_val * w_val;

                                    }

                                    let bias_val = bias_scalar.unwrap_or_else(|| {

                                        bias_data.as_ref().map(|b| b[oc_i]).unwrap_or(0.0)

                                    });

                                    let mut v = sum + bias_val;

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v.max(0.0); // ReLU activation

                                    let out_idx = ((n * out_channels + oc_i) * out_height + oh)

                                        * out_width

                                        + ow;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            oc += 8;

                        }



                        // Remainder channels (scalar)

                        while oc < out_channels {

                            let mut sum = 0.0f32;

                            for k_idx in 0..k {

                                let x_val = s[k_idx];

                                let w_val =

                                    // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() since oc < out_channels.

                                    unsafe { *wt_trans_slice.get_unchecked(k_idx * oc_count + oc) };

                                sum += x_val * w_val;

                            }

                            let bias_val = bias_scalar.unwrap_or_else(|| {

                                bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0)

                            });

                            let mut v = sum + bias_val;

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let mean = unsafe { *bn_mean_ptr.add(oc) };

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let var = unsafe { *bn_var_ptr.add(oc) };

                            let inv_std = 1.0 / (var + bn_eps).sqrt();

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc) }

                                // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                + unsafe { *bn_bias_ptr.add(oc) };

                            v = v.max(0.0); // ReLU activation

                            let out_idx =

                                ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                            unsafe {

                                *out_ptr.add(out_idx) = v;

                            }

                            oc += 1;

                        }

                    });
        }
    }

    output
}

#[allow(clippy::too_many_arguments)]
// --- FusedConvBnGELU (copied from SiLU, activation changed) ---
pub unsafe fn fused_conv_bn_gelu_3x3_direct(
    x: &Tensor,

    w: &Tensor,

    bias: Option<&Tensor>,

    bn_weight: &Tensor,

    bn_bias: &Tensor,

    bn_running_mean: &Tensor,

    bn_running_var: &Tensor,

    bn_eps: f32,

    batch_size: usize,

    in_channels: usize,

    out_channels: usize,

    in_height: usize,

    in_width: usize,

    out_height: usize,

    out_width: usize,
) -> Tensor {
    let output_shape = vec![
        batch_size as i64,
        out_channels as i64,
        out_height as i64,
        out_width as i64,
    ];

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());

    let output_inner = Arc::make_mut(&mut output.inner);

    let output_storage = Arc::make_mut(&mut output_inner.storage);

    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };

    let out_data = Arc::make_mut(&mut cpu_storage.data);

    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let x_ptr = x.data_ptr() as *const f32;

    let w_ptr = w.data_ptr() as *const f32;

    let bn_weight_ptr = bn_weight.data_ptr() as *const f32;

    let bn_bias_ptr = bn_bias.data_ptr() as *const f32;

    let bn_mean_ptr = bn_running_mean.data_ptr() as *const f32;

    let bn_var_ptr = bn_running_var.data_ptr() as *const f32;

    let bias_scalar: Option<f32> =
        bias.and_then(|b| if b.numel() == 1 { Some(b.item()) } else { None });

    let bias_data: Option<Vec<f32>> = bias.filter(|b| b.numel() > 1).map(|b| {
        let b_ptr = b.data_ptr() as *const f32;

        // SAFETY: bias tensor has out_channels elements as verified by tensor shape.

        unsafe { std::slice::from_raw_parts(b_ptr, out_channels).to_vec() }
    });

    let in_h = in_height as isize;

    let in_w = in_width as isize;

    // --- Step 1: Get transposed weights from thread-local buffer (recompute per call)

    let k = in_channels * 9;

    let oc_count = out_channels;

    let wt_trans_ptr: *const f32 = WT_TRANS_BUF.with(|buf| {
        let needed = k * oc_count;

        let mut b = buf.borrow_mut();

        let v = if let Some(ref mut v) = &mut *b {
            v
        } else {
            let new_vec = vec![0.0f32; needed];

            *b = Some(new_vec);

            b.as_mut().unwrap()
        };

        if v.len() < needed {
            v.resize(needed, 0.0);
        }

        for ic in 0..in_channels {
            for kh in 0..3 {
                for kw in 0..3 {
                    let k_idx = ic * 9 + kh * 3 + kw;

                    for oc in 0..out_channels {
                        let w_idx = ((oc * in_channels + ic) * 3 + kh) * 3 + kw;

                        // SAFETY: k_idx < k, oc < out_channels = oc_count, v.len() >= needed = k*oc_count.

                        // w_idx is bounded by the weight tensor dimensions.

                        unsafe { *v.get_unchecked_mut(k_idx * oc_count + oc) = *w_ptr.add(w_idx) };
                    }
                }
            }
        }

        v.as_ptr()
    });

    // SAFETY: wt_trans_ptr points to the thread-local buffer resized to `needed = k*oc_count` elements.

    let wt_trans_slice = unsafe { std::slice::from_raw_parts(wt_trans_ptr, k * oc_count) };

    // --- Step 2: Main kernel (parallel over output spatial+batch positions)

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let total_work = batch_size * out_height * out_width;

        let x_usize = x_ptr as usize;

        let out_usize = out_ptr as usize;

        let bn_weight_usize = bn_weight_ptr as usize;

        let bn_bias_usize = bn_bias_ptr as usize;

        let bn_mean_usize = bn_mean_ptr as usize;

        let bn_var_usize = bn_var_ptr as usize;

        (0..total_work).into_par_iter().for_each(|idx| {

            let n = idx / (out_height * out_width);

            let pos = idx % (out_height * out_width);

            let oh = pos / out_width;

            let ow = pos % out_width;

            let x_ptr = x_usize as *const f32;

            let out_ptr = out_usize as *mut f32;

                    let bn_weight_ptr = bn_weight_usize as *const f32;

                    let bn_bias_ptr = bn_bias_usize as *const f32;

                    let bn_mean_ptr = bn_mean_usize as *const f32;

                    let bn_var_ptr = bn_var_usize as *const f32;



                    S_BUF.with(|s| {

                        let mut s_buf = s.borrow_mut();

                        if s_buf.len() < k {

                            s_buf.resize(k, 0.0);

                        }

                        let s = &mut s_buf[..k];

                        let mut k_idx = 0;

                        for ic in 0..in_channels {

                            for kh in 0..3 {

                                for kw in 0..3 {

                                    let ih_s = (oh as isize) + (kh as isize) - 1;

                                    let iw_s = (ow as isize) + (kw as isize) - 1;

                                    s[k_idx] =

                                        if ih_s >= 0 && ih_s < in_h && iw_s >= 0 && iw_s < in_w {

                                            let ih = ih_s as usize;

                                            let iw = iw_s as usize;

                                            let x_idx = ((n * in_channels + ic) * in_height + ih)

                                                * in_width

                                                + iw;

                                            // SAFETY: x_idx computed from n, ic, ih, iw is bounded by batch_size*in_channels*in_height*in_width = x tensor element count.

                                            unsafe { *x_ptr.add(x_idx) }

                                        } else {

                                            0.0

                                        };

                                    k_idx += 1;

                                }

                            }

                        }



                        let mut oc = 0;

                        while oc + 8 <= out_channels {

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]

                            {

                                use wide::f32x8;

                                let mut acc = f32x8::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let w_ptr_k =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_simd = from_slice_unaligned_f32x8(unsafe {

                                        std::slice::from_raw_parts(w_ptr_k, 8)

                                    });

                                    acc += w_simd * f32x8::splat(x_val);

                                }



                                let bias_vec = if let Some(ref b) = bias_data {

                                    let b_slice = unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 8)

                                    };

                                    from_slice_unaligned_f32x8(b_slice)

                                } else if let Some(b) = bias_scalar {

                                    f32x8::splat(b)

                                } else {

                                    f32x8::ZERO

                                };



                                let res_arr = (acc + bias_vec).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                // Apply BN+SiLU per-element and store

                                for i in 0..8 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_arr[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v

                                        * 0.5

                                        * (1.0

                                            + (GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v))

                                                .tanh()); // GELU activation

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(all(feature = "simd", target_arch = "aarch64"))]

                            {

                                use wide::f32x4;

                                let mut acc_lo = f32x4::ZERO;

                                let mut acc_hi = f32x4::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let base_ptr =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_lo = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr, 4)

                                    });

                                    acc_lo = acc_lo + w_lo * f32x4::splat(x_val);

                                    let w_hi = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr.add(4), 4)

                                    });

                                    acc_hi = acc_hi + w_hi * f32x4::splat(x_val);

                                }



                                let bias_lo = if let Some(ref b) = bias_data {

                                    // SAFETY: oc < out_channels and oc+4 <= out_channels, so b[oc..oc+4] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };

                                let bias_hi = if let Some(ref b) = bias_data {

                                    // SAFETY: oc+4 < out_channels, so b[oc+4..oc+8] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc + 4), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };



                                let res_lo = (acc_lo + bias_lo).to_array();

                                let res_hi = (acc_hi + bias_hi).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                // Apply BN+SiLU for lo (4)

                                for i in 0..4 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_lo[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v

                                        * 0.5

                                        * (1.0

                                            + (GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v))

                                                .tanh()); // GELU activation

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                                // Apply BN+SiLU for hi (4)

                                for i in 0..4 {

                                    let oc_i = oc + 4 + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_hi[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v

                                        * 0.5

                                        * (1.0

                                            + (GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v))

                                                .tanh()); // GELU activation

                                    let out_idx = out_idx_base + (4 + i) * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(not(all(

                                feature = "simd",

                                any(target_arch = "x86_64", target_arch = "aarch64")

                            )))]

                            {

                                for oc_off in 0..8 {

                                    let oc_i = oc + oc_off;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut sum = 0.0f32;

                                    for k_idx in 0..k {

                                        let x_val = s[k_idx];

                                        let w_val = unsafe {

                                            *wt_trans_slice.get_unchecked(k_idx * oc_count + oc_i)

                                        };

                                        sum += x_val * w_val;

                                    }

                                    let bias_val = bias_scalar.unwrap_or_else(|| {

                                        bias_data.as_ref().map(|b| b[oc_i]).unwrap_or(0.0)

                                    });

                                    let mut v = sum + bias_val;

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v

                                        * 0.5

                                        * (1.0

                                            + (GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v))

                                                .tanh()); // GELU activation

                                    let out_idx = ((n * out_channels + oc_i) * out_height + oh)

                                        * out_width

                                        + ow;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            oc += 8;

                        }



                        // Remainder channels (scalar)

                        while oc < out_channels {

                            let mut sum = 0.0f32;

                            for k_idx in 0..k {

                                let x_val = s[k_idx];

                                let w_val =

                                    // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() since oc < out_channels.

                                    unsafe { *wt_trans_slice.get_unchecked(k_idx * oc_count + oc) };

                                sum += x_val * w_val;

                            }

                            let bias_val = bias_scalar.unwrap_or_else(|| {

                                bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0)

                            });

                            let mut v = sum + bias_val;

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let mean = unsafe { *bn_mean_ptr.add(oc) };

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let var = unsafe { *bn_var_ptr.add(oc) };

                            let inv_std = 1.0 / (var + bn_eps).sqrt();

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc) }

                                // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                + unsafe { *bn_bias_ptr.add(oc) };

                            v = v

                                * 0.5

                                * (1.0

                                    + (GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v)).tanh()); // GELU activation

                            let out_idx =

                                ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                            unsafe {

                                *out_ptr.add(out_idx) = v;

                            }

                            oc += 1;

                        }

                    });

        })
    }

    #[cfg(not(feature = "parallel"))]
    {
        let total_work = batch_size * out_height * out_width;

        for idx in 0..total_work {

            let n = idx / (out_height * out_width);

            let pos = idx % (out_height * out_width);

            let oh = pos / out_width;

            let ow = pos % out_width;

                    S_BUF.with(|s| {

                        let mut s_buf = s.borrow_mut();

                        if s_buf.len() < k {

                            s_buf.resize(k, 0.0);

                        }

                        let s = &mut s_buf[..k];

                        let mut k_idx = 0;

                        for ic in 0..in_channels {

                            for kh in 0..3 {

                                for kw in 0..3 {

                                    let ih_s = (oh as isize) + (kh as isize) - 1;

                                    let iw_s = (ow as isize) + (kw as isize) - 1;

                                    s[k_idx] =

                                        if ih_s >= 0 && ih_s < in_h && iw_s >= 0 && iw_s < in_w {

                                            let ih = ih_s as usize;

                                            let iw = iw_s as usize;

                                            let x_idx = ((n * in_channels + ic) * in_height + ih)

                                                * in_width

                                                + iw;

                                            // SAFETY: x_idx computed from n, ic, ih, iw is bounded by batch_size*in_channels*in_height*in_width = x tensor element count.

                                            unsafe { *x_ptr.add(x_idx) }

                                        } else {

                                            0.0

                                        };

                                    k_idx += 1;

                                }

                            }

                        }



                        let mut oc = 0;

                        while oc + 8 <= out_channels {

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]

                            {

                                use wide::f32x8;

                                let mut acc = f32x8::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let w_ptr_k =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_simd = from_slice_unaligned_f32x8(unsafe {

                                        std::slice::from_raw_parts(w_ptr_k, 8)

                                    });

                                    acc += w_simd * f32x8::splat(x_val);

                                }



                                let bias_vec = if let Some(ref b) = bias_data {

                                    let b_slice = unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 8)

                                    };

                                    from_slice_unaligned_f32x8(b_slice)

                                } else if let Some(b) = bias_scalar {

                                    f32x8::splat(b)

                                } else {

                                    f32x8::ZERO

                                };



                                let res_arr = (acc + bias_vec).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                // Apply BN+SiLU per-element and store

                                for i in 0..8 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_arr[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v

                                        * 0.5

                                        * (1.0

                                            + (GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v))

                                                .tanh()); // GELU activation

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(all(feature = "simd", target_arch = "aarch64"))]

                            {

                                use wide::f32x4;

                                let mut acc_lo = f32x4::ZERO;

                                let mut acc_hi = f32x4::ZERO;



                                for k_idx in 0..k {

                                    let x_val = s[k_idx];

                                    let base_ptr =

                                        // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() for oc < out_channels in SIMD blocks.
                                        unsafe { wt_trans_slice.as_ptr().add(k_idx * oc_count + oc) };

                                    let w_lo = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr, 4)

                                    });

                                    acc_lo = acc_lo + w_lo * f32x4::splat(x_val);

                                    let w_hi = from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(base_ptr.add(4), 4)

                                    });

                                    acc_hi = acc_hi + w_hi * f32x4::splat(x_val);

                                }



                                let bias_lo = if let Some(ref b) = bias_data {

                                    // SAFETY: oc < out_channels and oc+4 <= out_channels, so b[oc..oc+4] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };

                                let bias_hi = if let Some(ref b) = bias_data {

                                    // SAFETY: oc+4 < out_channels, so b[oc+4..oc+8] is in bounds.

                                    from_slice_unaligned_f32x4(unsafe {

                                        std::slice::from_raw_parts(b.as_ptr().add(oc + 4), 4)

                                    })

                                } else if let Some(b) = bias_scalar {

                                    f32x4::splat(b)

                                } else {

                                    f32x4::ZERO

                                };



                                let res_lo = (acc_lo + bias_lo).to_array();

                                let res_hi = (acc_hi + bias_hi).to_array();

                                let out_idx_base =

                                    ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                                let spatial_stride = out_height * out_width;



                                // Apply BN+SiLU for lo (4)

                                for i in 0..4 {

                                    let oc_i = oc + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_lo[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v

                                        * 0.5

                                        * (1.0

                                            + (GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v))

                                                .tanh()); // GELU activation

                                    let out_idx = out_idx_base + i * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                                // Apply BN+SiLU for hi (4)

                                for i in 0..4 {

                                    let oc_i = oc + 4 + i;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut v = res_hi[i];

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v

                                        * 0.5

                                        * (1.0

                                            + (GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v))

                                                .tanh()); // GELU activation

                                    let out_idx = out_idx_base + (4 + i) * spatial_stride;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            #[cfg(not(all(

                                feature = "simd",

                                any(target_arch = "x86_64", target_arch = "aarch64")

                            )))]

                            {

                                for oc_off in 0..8 {

                                    let oc_i = oc + oc_off;

                                    if oc_i >= out_channels {

                                        break;

                                    }

                                    let mut sum = 0.0f32;

                                    for k_idx in 0..k {

                                        let x_val = s[k_idx];

                                        let w_val = unsafe {

                                            *wt_trans_slice.get_unchecked(k_idx * oc_count + oc_i)

                                        };

                                        sum += x_val * w_val;

                                    }

                                    let bias_val = bias_scalar.unwrap_or_else(|| {

                                        bias_data.as_ref().map(|b| b[oc_i]).unwrap_or(0.0)

                                    });

                                    let mut v = sum + bias_val;

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let mean = unsafe { *bn_mean_ptr.add(oc_i) };

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    let var = unsafe { *bn_var_ptr.add(oc_i) };

                                    let inv_std = 1.0 / (var + bn_eps).sqrt();

                                    // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                    v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc_i) }

                                        // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                        + unsafe { *bn_bias_ptr.add(oc_i) };

                                    v = v

                                        * 0.5

                                        * (1.0

                                            + (GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v))

                                                .tanh()); // GELU activation

                                    let out_idx = ((n * out_channels + oc_i) * out_height + oh)

                                        * out_width

                                        + ow;

                                    unsafe {

                                        *out_ptr.add(out_idx) = v;

                                    }

                                }

                            }

                            oc += 8;

                        }



                        // Remainder channels (scalar)

                        while oc < out_channels {

                            let mut sum = 0.0f32;

                            for k_idx in 0..k {

                                let x_val = s[k_idx];

                                let w_val =

                                    // SAFETY: k_idx*oc_count+oc < k*oc_count = wt_trans_slice.len() since oc < out_channels.

                                    unsafe { *wt_trans_slice.get_unchecked(k_idx * oc_count + oc) };

                                sum += x_val * w_val;

                            }

                            let bias_val = bias_scalar.unwrap_or_else(|| {

                                bias_data.as_ref().map(|b| b[oc]).unwrap_or(0.0)

                            });

                            let mut v = sum + bias_val;

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let mean = unsafe { *bn_mean_ptr.add(oc) };

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            let var = unsafe { *bn_var_ptr.add(oc) };

                            let inv_std = 1.0 / (var + bn_eps).sqrt();

                            // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                            v = (v - mean) * inv_std * unsafe { *bn_weight_ptr.add(oc) }

                                // SAFETY: oc_i < out_channels, so bn arrays are indexed in bounds.

                                + unsafe { *bn_bias_ptr.add(oc) };

                            v = v

                                * 0.5

                                * (1.0

                                    + (GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v)).tanh()); // GELU activation

                            let out_idx =

                                ((n * out_channels + oc) * out_height + oh) * out_width + ow;

                            unsafe {

                                *out_ptr.add(out_idx) = v;

                            }

                            oc += 1;

                        }

                    });
        }
    }

    output
}

#[allow(clippy::too_many_arguments)]
// --- ReLU kernel (dispatch entry) ---
pub unsafe fn fused_conv_bn_relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];

    let w = args[1];

    let mut idx = 2;

    let bias = if args.len() > idx && args[idx].numel() > 0 {
        let b = Some(args[idx]);

        idx += 1;

        b
    } else {
        None
    };

    if args.len() < idx + 9 {
        panic!("fused_conv_bn_relu: insufficient arguments");
    }

    let bn_weight = args[idx];

    idx += 1;

    let bn_bias = args[idx];

    idx += 1;

    let bn_running_mean = args[idx];

    idx += 1;

    let bn_running_var = args[idx];

    idx += 1;

    let stride_t = args[idx];

    idx += 1;

    let padding_t = args[idx];

    idx += 1;

    let dilation_t = args[idx];

    idx += 1;

    let groups_t = args[idx];

    idx += 1;

    let eps_t = args[idx];

    let stride = stride_t.item() as i64;

    let padding = padding_t.item() as i64;

    let dilation = dilation_t.item() as i64;

    let groups = groups_t.item() as i64;

    let eps = eps_t.item();

    if stride != 1 || padding != 1 || dilation != 1 || groups != 1 {
        panic!("fused_conv_bn_relu: only 3x3 kernel with stride=1, padding=1, dilation=1, groups=1 is supported");
    }

    let x_shape = x.shape_ref();

    let w_shape = w.shape_ref();

    let batch_size = x_shape[0] as usize;

    let in_channels = x_shape[1] as usize;

    let in_height = x_shape[2] as usize;

    let in_width = x_shape[3] as usize;

    let out_channels = w_shape[0] as usize;

    let out_height = in_height;

    let out_width = in_width;

    vec![fused_conv_bn_relu_3x3_direct(
        x,
        w,
        bias,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        eps,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
    )]
}

// --- GELU kernel (dispatch entry) ---

pub unsafe fn fused_conv_bn_gelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];

    let w = args[1];

    let mut idx = 2;

    let bias = if args.len() > idx && args[idx].numel() > 0 {
        let b = Some(args[idx]);

        idx += 1;

        b
    } else {
        None
    };

    if args.len() < idx + 9 {
        panic!("fused_conv_bn_gelu: insufficient arguments");
    }

    let bn_weight = args[idx];

    idx += 1;

    let bn_bias = args[idx];

    idx += 1;

    let bn_running_mean = args[idx];

    idx += 1;

    let bn_running_var = args[idx];

    idx += 1;

    let stride_t = args[idx];

    idx += 1;

    let padding_t = args[idx];

    idx += 1;

    let dilation_t = args[idx];

    idx += 1;

    let groups_t = args[idx];

    idx += 1;

    let eps_t = args[idx];

    let stride = stride_t.item() as i64;

    let padding = padding_t.item() as i64;

    let dilation = dilation_t.item() as i64;

    let groups = groups_t.item() as i64;

    let eps = eps_t.item();

    if stride != 1 || padding != 1 || dilation != 1 || groups != 1 {
        panic!("fused_conv_bn_gelu: only 3x3 kernel with stride=1, padding=1, dilation=1, groups=1 is supported");
    }

    let x_shape = x.shape_ref();

    let w_shape = w.shape_ref();

    let batch_size = x_shape[0] as usize;

    let in_channels = x_shape[1] as usize;

    let in_height = x_shape[2] as usize;

    let in_width = x_shape[3] as usize;

    let out_channels = w_shape[0] as usize;

    let out_height = in_height;

    let out_width = in_width;

    vec![fused_conv_bn_gelu_3x3_direct(
        x,
        w,
        bias,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        eps,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
    )]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;

    fn reference_conv2d_1x1(
        x_data: &[f32],
        w_data: &[f32],
        bias_data: Option<&[f32]>,
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        h: usize,
        w: usize,
    ) -> Vec<f32> {
        let spatial = h * w;
        let mut out = vec![0.0f32; batch_size * out_channels * spatial];
        for n in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..h {
                    for ow in 0..w {
                        let out_idx = (n * out_channels + oc) * spatial + oh * w + ow;
                        let mut sum = 0.0f32;
                        for ic in 0..in_channels {
                            let x_idx = (n * in_channels + ic) * spatial + oh * w + ow;
                            // w[oc, ic, 0, 0] = w_data[oc * in_channels + ic]
                            sum += x_data[x_idx] * w_data[oc * in_channels + ic];
                        }
                        let b = bias_data.map(|b| b[oc]).unwrap_or(0.0);
                        out[out_idx] = sum + b;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn test_conv2d_1x1_sparse_small() {
        let batch_size = 1;
        let in_channels = 2;
        let out_channels = 4;
        let h = 2;
        let w = 2;

        let spatial = h * w;

        // Sparse weights: each oc sees exactly one ic
        // w[oc, ic] = if ic == oc % in_channels { 1.0 } else { 0.0 }
        let mut w_data = vec![0.0f32; out_channels * in_channels];
        for oc in 0..out_channels {
            w_data[oc * in_channels + oc % in_channels] = 1.0;
        }

        // x[n, ic, oh, ow] = ic * 100 + oh * 10 + ow
        let mut x_data = vec![0.0f32; batch_size * in_channels * spatial];
        for n in 0..batch_size {
            for ic in 0..in_channels {
                for oh in 0..h {
                    for ow in 0..w {
                        let x_idx = (n * in_channels + ic) * spatial + oh * w + ow;
                        x_data[x_idx] = (ic * 100 + oh * 10 + ow) as f32;
                    }
                }
            }
        }

        let x_t = Tensor::from_vec(x_data.clone(), vec![batch_size as i64, in_channels as i64, h as i64, w as i64]);
        let w_t = Tensor::from_vec(w_data.clone(), vec![out_channels as i64, in_channels as i64, 1i64, 1i64]);

        let expected = reference_conv2d_1x1(&x_data, &w_data, None, batch_size, in_channels, out_channels, h, w);

        unsafe {
            let result = conv2d_1x1(
                &x_t,
                &w_t,
                None,
                None,
                batch_size,
                in_channels,
                out_channels,
                h,
                w,
            );
            let out = result.as_f32_slice();
            assert_eq!(out.len(), expected.len());
            for i in 0..out.len() {
                let diff = (out[i] - expected[i]).abs();
                assert!(diff < 1e-5, "idx={}: got={}, expected={}", i, out[i], expected[i]);
            }
        }
        println!("test_conv2d_1x1_sparse_small PASSED");
    }

    fn run_conv2d_1x1_with_pre_t(
        x_data: &[f32],
        w_data: &[f32],
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        h: usize,
        w: usize,
    ) -> Tensor {
        let w_t = Tensor::from_vec(w_data.to_vec(), vec![out_channels as i64, in_channels as i64, 1i64, 1i64]);
        // Build pre-transposed weight [IC, OC]
        let mut wt_data = vec![0.0f32; in_channels * out_channels];
        for ic in 0..in_channels {
            for oc in 0..out_channels {
                wt_data[ic * out_channels + oc] = w_data[oc * in_channels + ic];
            }
        }
        let wt_t = Tensor::from_vec(wt_data, vec![in_channels as i64, out_channels as i64]);
        let x_t = Tensor::from_vec(x_data.to_vec(), vec![batch_size as i64, in_channels as i64, h as i64, w as i64]);

        unsafe {
            conv2d_1x1(
                &x_t,
                &w_t,
                Some(&wt_t),
                None,
                batch_size,
                in_channels,
                out_channels,
                h,
                w,
            )
        }
    }

    fn reference_conv2d_1x1_simple(
        x_data: &[f32],
        w_data: &[f32],
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        h: usize,
        w: usize,
    ) -> Vec<f32> {
        let spatial = h * w;
        let mut out = vec![0.0f32; batch_size * out_channels * spatial];
        for n in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..h {
                    for ow in 0..w {
                        let out_idx = (n * out_channels + oc) * spatial + oh * w + ow;
                        let mut sum = 0.0f32;
                        for ic in 0..in_channels {
                            let x_idx = (n * in_channels + ic) * spatial + oh * w + ow;
                            sum += x_data[x_idx] * w_data[oc * in_channels + ic];
                        }
                        out[out_idx] = sum;
                    }
                }
            }
        }
        out
    }

    /// Compare conv2d_kernel vs direct conv2d_1x1 call with contiguous x
    #[test]
    fn test_conv2d_1x1_vs_kernel() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for trial in 0..50 {
            let out_channels: usize = rng.gen_range(8..=32) / 8 * 8;
            let in_channels = rng.gen_range(1..64);
            let h = rng.gen_range(1..30);
            let w = rng.gen_range(1..30);
            let batch_size = 1;
            let spatial = h * w;

            let mut x_data = vec![0.0f32; batch_size * in_channels * spatial];
            let mut w_data = vec![0.0f32; out_channels * in_channels];
            for v in x_data.iter_mut() { *v = rng.gen_range(-1.0..1.0); }
            for v in w_data.iter_mut() { *v = rng.gen_range(-1.0..1.0); }

            let expected = reference_conv2d_1x1_simple(
                &x_data, &w_data, batch_size, in_channels, out_channels, h, w,
            );

            let x_t = Tensor::from_vec(x_data.clone(), vec![batch_size as i64, in_channels as i64, h as i64, w as i64]);
            let w_t = Tensor::from_vec(w_data.clone(), vec![out_channels as i64, in_channels as i64, 1i64, 1i64]);
            let reshaped = w_t.reshape(vec![out_channels as i64, in_channels as i64]);
            let pre_t = reshaped.transpose(0_usize, 1_usize).contiguous();

            // METHOD A: direct conv2d_1x1 call with contiguous clone of x_t (like kernel does)
            let x_contiguous = x_t.contiguous();  // same as kernel's args[0].contiguous()
            let result_a = unsafe {
                conv2d_1x1(&x_contiguous, &w_t, Some(&pre_t), None, batch_size, in_channels, out_channels, h, w)
            };
            let out_a = result_a.as_f32_slice();
            let mut a_failed = false;
            for i in 0..out_a.len() {
                if (out_a[i] - expected[i]).abs() > 1e-3 {
                    a_failed = true;
                    break;
                }
            }

            // METHOD B: via conv2d_kernel
            let zero_bias = Tensor::from_scalar(0.0f32);
            let stride_t = Tensor::from_scalar(1.0f32);
            let pad_t = Tensor::from_scalar(0.0f32);
            let dilation_t = Tensor::from_scalar(1.0f32);
            let groups_t = Tensor::from_scalar(1.0f32);
            let result_b = unsafe {
                conv2d_kernel(&[&x_t, &w_t, &zero_bias, &stride_t, &pad_t, &dilation_t, &groups_t, &pre_t])
            };
            let out_b = result_b[0].as_f32_slice();
            let mut b_failed = false;
            let mut b_fail_idx = 0;
            for i in 0..out_b.len() {
                if (out_b[i] - expected[i]).abs() > 1e-3 {
                    b_failed = true;
                    b_fail_idx = i;
                    break;
                }
            }

            if a_failed && !b_failed {
                panic!("A (contiguous x) failed but B passed");
            }
            if !a_failed && b_failed {
                let oc_idx = b_fail_idx / spatial;
                let sp_idx = b_fail_idx % spatial;
                panic!(
                    "B (kernel) FAILED but A (contiguous x) PASSED: OC={} IC={} H={} W={} trial={}: ch[{}] sp[{}] got={}, expected={}",
                    out_channels, in_channels, h, w, trial, oc_idx, sp_idx, out_b[b_fail_idx], expected[b_fail_idx]
                );
            }
        }
        println!("test_conv2d_1x1_vs_kernel: Both agree on all {} trials", 50);
    }

    /// Test with DAG-executor-style pre-t weight construction: reshape → transpose → contiguous
    #[test]
    fn test_conv2d_1x1_dag_style_pre_t() {
        // Use the same construction as DAGExecutor:
        // let reshaped = w.reshape(vec![oc, ic]);
        // let transposed = reshaped.transpose(0, 1).contiguous();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let oc_values = [8, 16, 20, 24, 28, 32];
        for &out_channels in &oc_values {
            for trial in 0..200 {
                let in_channels = rng.gen_range(1..64);
                let h = rng.gen_range(1..30);
                let w = rng.gen_range(1..30);
                let batch_size = 1;
                let spatial = h * w;

                let mut x_data = vec![0.0f32; batch_size * in_channels * spatial];
                let mut w_data = vec![0.0f32; out_channels * in_channels];
                for v in x_data.iter_mut() { *v = rng.gen_range(-1.0..1.0); }
                for v in w_data.iter_mut() { *v = rng.gen_range(-1.0..1.0); }

                let expected = reference_conv2d_1x1_simple(
                    &x_data, &w_data, batch_size, in_channels, out_channels, h, w,
                );

                let x_t = Tensor::from_vec(x_data.clone(), vec![batch_size as i64, in_channels as i64, h as i64, w as i64]);
                // Create original 4D weight [OC, IC, 1, 1]
                let w_t = Tensor::from_vec(w_data.clone(), vec![out_channels as i64, in_channels as i64, 1i64, 1i64]);
                // Build pre-t exactly as DAGExecutor does
                let reshaped = w_t.reshape(vec![out_channels as i64, in_channels as i64]);
                let pre_t = reshaped.transpose(0_usize, 1_usize).contiguous();

                unsafe {
                    let result = conv2d_1x1(
                        &x_t,
                        &w_t,
                        Some(&pre_t),
                        None,
                        batch_size,
                        in_channels,
                        out_channels,
                        h,
                        w,
                    );
                    let out = result.as_f32_slice();
                    assert_eq!(out.len(), expected.len());
                    for i in 0..out.len() {
                        let diff = (out[i] - expected[i]).abs();
                        if diff > 1e-3 {
                            let oc_idx = i / spatial;
                            let sp_idx = i % spatial;
                            panic!(
                                "FAIL: OC={}, IC={}, H={}, W={}, trial={}: ch[{}] sp[{}] got={}, expected={}, diff={}",
                                out_channels, in_channels, h, w, trial, oc_idx, sp_idx, out[i], expected[i], diff
                            );
                        }
                    }
                }
            }
        }
        println!("test_conv2d_1x1_dag_style_pre_t PASSED (all {} combos)", oc_values.len() * 200);
    }

    /// Test the inline transpose path directly by calling conv2d_1x1 WITHOUT pre-transposed weight
    #[test]
    fn test_conv2d_1x1_inline_transpose() {
        let oc_values = [8, 16, 20, 24, 32];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for &out_channels in &oc_values {
            for trial in 0..50 {
                let in_channels = rng.gen_range(1..64);
                let h = rng.gen_range(1..30);
                let w = rng.gen_range(1..30);
                let batch_size = 1;
                let spatial = h * w;

                let mut x_data = vec![0.0f32; batch_size * in_channels * spatial];
                let mut w_data = vec![0.0f32; out_channels * in_channels];
                for v in x_data.iter_mut() { *v = rng.gen_range(-1.0..1.0); }
                for v in w_data.iter_mut() { *v = rng.gen_range(-1.0..1.0); }

                let expected = reference_conv2d_1x1_simple(
                    &x_data, &w_data, batch_size, in_channels, out_channels, h, w,
                );

                let x_t = Tensor::from_vec(x_data.clone(), vec![batch_size as i64, in_channels as i64, h as i64, w as i64]);
                let w_t = Tensor::from_vec(w_data.clone(), vec![out_channels as i64, in_channels as i64, 1i64, 1i64]);

                unsafe {
                    // NO pre-transposed weight - this forces the inline transpose path
                    let result = conv2d_1x1(
                        &x_t,
                        &w_t,
                        None,
                        None,
                        batch_size,
                        in_channels,
                        out_channels,
                        h,
                        w,
                    );
                    let out = result.as_f32_slice();
                    assert_eq!(out.len(), expected.len());
                    for i in 0..out.len() {
                        let diff = (out[i] - expected[i]).abs();
                        if diff > 1e-3 {
                            let oc_idx = i / spatial;
                            let sp_idx = i % spatial;
                            panic!(
                                "FAIL: OC={}, IC={}, H={}, W={}, trial={}: ch[{}] sp[{}] got={}, expected={}, diff={}",
                                out_channels, in_channels, h, w, trial, oc_idx, sp_idx, out[i], expected[i], diff
                            );
                        }
                    }
                }
            }
        }
        println!("test_conv2d_1x1_inline_transpose PASSED (all {} combos)", oc_values.len() * 50);
    }

    #[test]
    fn test_conv2d_1x1_random() {
        let seed: u64 = std::env::var("TEST_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(42);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let oc_values = [8, 16, 20, 24, 28, 32, 40];
        for &out_channels in &oc_values {
            for _trial in 0..50 {
                let in_channels = rng.gen_range(1..64);
                let h = rng.gen_range(1..30);
                let w = rng.gen_range(1..30);
                let batch_size = 1;

                let spatial = h * w;

                let mut x_data = vec![0.0f32; batch_size * in_channels * spatial];
                let mut w_data = vec![0.0f32; out_channels * in_channels];
                for v in x_data.iter_mut() { *v = rng.gen_range(-1.0..1.0); }
                for v in w_data.iter_mut() { *v = rng.gen_range(-1.0..1.0); }

                let expected = reference_conv2d_1x1(&x_data, &w_data, None, batch_size, in_channels, out_channels, h, w);

                unsafe {
                    let result = run_conv2d_1x1_with_pre_t(
                        &x_data, &w_data,
                        batch_size, in_channels, out_channels, h, w,
                    );
                    let out = result.as_f32_slice();
                    assert_eq!(out.len(), expected.len());
                    for i in 0..out.len() {
                        let diff = (out[i] - expected[i]).abs();
                        if diff > 1e-3 {
                            let oc_idx = i / spatial;
                            let sp_idx = i % spatial;
                            panic!(
                                "FAIL: OC={}, IC={}, H={}, W={}, trial={}, seed={}: ch[{}] sp[{}] got={}, expected={}, diff={}",
                                out_channels, in_channels, h, w, _trial, seed, oc_idx, sp_idx, out[i], expected[i], diff
                            );
                        }
                    }
                }
            }
        }
        println!("test_conv2d_1x1_random seed={} PASSED (all {} combos)", seed, oc_values.len() * 50);
    }

    #[test]
    fn test_conv2d_1x1_specific_failing() {
        // Known failing cases from previous debugging sessions
        let cases = vec![
            (24, 62, 55, 55),  // channel 10 fails with seed=13
            (32, 34, 33, 33),  // channels 26,27,29 fail with seed=2
            (32, 16, 4, 4),    // channels 8,9 fail
        ];
        for &(out_channels, in_channels, h, w) in &cases {
            let batch_size = 1;
            let spatial = h * w;
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);

            let mut x_data = vec![0.0f32; batch_size * in_channels * spatial];
            let mut w_data = vec![0.0f32; out_channels * in_channels];
            for v in x_data.iter_mut() { *v = rng.gen_range(-1.0..1.0); }
            for v in w_data.iter_mut() { *v = rng.gen_range(-1.0..1.0); }

            let expected = reference_conv2d_1x1(&x_data, &w_data, None, batch_size, in_channels, out_channels, h, w);

            unsafe {
                let result = run_conv2d_1x1_with_pre_t(
                    &x_data, &w_data,
                    batch_size, in_channels, out_channels, h, w,
                );
                let out = result.as_f32_slice();
                assert_eq!(out.len(), expected.len());
                let mut worst_diff = 0.0f32;
                let mut worst_idx = 0;
                for i in 0..out.len() {
                    let diff = (out[i] - expected[i]).abs();
                    if diff > worst_diff {
                        worst_diff = diff;
                        worst_idx = i;
                    }
                }
                let oc_idx = worst_idx / spatial;
                let sp_idx = worst_idx % spatial;
                assert!(
                    worst_diff < 1e-3,
                    "FAIL: OC={}, IC={}, H={}, W={}: ch[{}] sp[{}] got={}, expected={}, diff={}",
                    out_channels, in_channels, h, w, oc_idx, sp_idx, out[worst_idx], expected[worst_idx], worst_diff
                );
            }
        }
        println!("test_conv2d_1x1_specific_failing PASSED");
    }
}
