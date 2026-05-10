//! CPU pooling kernels.

#![allow(unused_imports)]
#![allow(clippy::missing_safety_doc)]

use super::*;
use crate::autograd::{AutogradMeta, Edge, Node};
use crate::dispatcher::{register, DispatchKey, KernelFn};
use crate::iterator::TensorIterator;
use crate::kernels::blas::{
    matmul_blas, matmul_blas_into, matmul_blas_with_transpose, matmul_blas_with_transpose_into,
    MIN_BLAS_SIZE,
};
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use std::sync::Arc;

pub unsafe fn max_pool2d_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let kernel_size = if args.len() > 1 {
        args[1].item() as i64
    } else {
        2
    };
    let stride = if args.len() > 2 {
        args[2].item() as i64
    } else {
        kernel_size
    };
    let padding = if args.len() > 3 {
        args[3].item() as i64
    } else {
        0
    };
    let dilation = if args.len() > 4 {
        args[4].item() as i64
    } else {
        1
    };

    let x_shape = x.shape_ref();
    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_height = x_shape[2];
    let in_width = x_shape[3];

    let out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    let out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    let output = Tensor::empty(
        vec![batch_size, channels, out_height, out_width],
        x.dtype(),
        x.device(),
    );

    let x_ptr = x.data_ptr() as *const f32;
    let out_ptr = output.data_ptr() as *mut f32;

    let total_bc = batch_size as usize * channels as usize;
    let stride_usize = stride as usize;
    let dilation_usize = dilation as usize;
    let kernel_size_usize = kernel_size as usize;
    let in_h = in_height as usize;
    let in_w = in_width as usize;
    let out_h = out_height as usize;
    let out_w = out_width as usize;
    let pad = padding;
    let channels_usize = channels as usize;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let x_usize = x_ptr as usize;
        let out_usize = out_ptr as usize;
        (0..total_bc).into_par_iter().for_each(|bc| {
            let b = bc / channels_usize;
            let c = bc % channels_usize;
            let x_p = x_usize as *const f32;
            let o_p = out_usize as *mut f32;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for kh in 0..kernel_size_usize {
                        for kw in 0..kernel_size_usize {
                            let h = (oh * stride_usize + kh * dilation_usize) as i64 - pad;
                            let w = (ow * stride_usize + kw * dilation_usize) as i64 - pad;
                            if h >= 0 && h < in_h as i64 && w >= 0 && w < in_w as i64 {
                                let idx = ((b * channels_usize + c) * in_h + h as usize) * in_w
                                    + w as usize;
                                let val = unsafe { *x_p.add(idx) };
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    let out_idx = ((b * channels_usize + c) * out_h + oh) * out_w + ow;
                    unsafe {
                        *o_p.add(out_idx) = max_val;
                    }
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for b in 0..batch_size as usize {
            for c in 0..channels as usize {
                for oh in 0..out_height as usize {
                    for ow in 0..out_width as usize {
                        let mut max_val = f32::NEG_INFINITY;

                        for kh in 0..kernel_size as usize {
                            for kw in 0..kernel_size as usize {
                                let h = (oh * stride as usize + kh * dilation as usize) as i64
                                    - padding;
                                let w = (ow * stride as usize + kw * dilation as usize) as i64
                                    - padding;

                                if h >= 0 && h < in_height && w >= 0 && w < in_width {
                                    let idx = ((b * channels as usize + c) * in_height as usize
                                        + h as usize)
                                        * in_width as usize
                                        + w as usize;
                                    let val = unsafe { *x_ptr.add(idx) };
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }

                        let out_idx = ((b * channels as usize + c) * out_height as usize + oh)
                            * out_width as usize
                            + ow;
                        unsafe { *out_ptr.add(out_idx) = max_val };
                    }
                }
            }
        }
    }

    vec![output]
}

pub unsafe fn avg_pool2d_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let kernel_size = if args.len() > 1 {
        args[1].item() as i64
    } else {
        2
    };
    let stride = if args.len() > 2 {
        args[2].item() as i64
    } else {
        kernel_size
    };
    let padding = if args.len() > 3 {
        args[3].item() as i64
    } else {
        0
    };

    let x_shape = x.shape_ref();
    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_height = x_shape[2];
    let in_width = x_shape[3];

    let out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    let out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    let output = Tensor::empty(
        vec![batch_size, channels, out_height, out_width],
        x.dtype(),
        x.device(),
    );

    let x_ptr = x.data_ptr() as *const f32;
    let out_ptr = output.data_ptr() as *mut f32;

    let total_bc = batch_size as usize * channels as usize;
    let stride_usize = stride as usize;
    let kernel_size_usize = kernel_size as usize;
    let in_h = in_height as usize;
    let in_w = in_width as usize;
    let out_h = out_height as usize;
    let out_w = out_width as usize;
    let pad = padding;
    let channels_usize = channels as usize;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let x_usize = x_ptr as usize;
        let out_usize = out_ptr as usize;
        (0..total_bc).into_par_iter().for_each(|bc| {
            let b = bc / channels_usize;
            let c = bc % channels_usize;
            let x_p = x_usize as *const f32;
            let o_p = out_usize as *mut f32;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    let mut count = 0i64;
                    for kh in 0..kernel_size_usize {
                        for kw in 0..kernel_size_usize {
                            let h = (oh * stride_usize + kh) as i64 - pad;
                            let w = (ow * stride_usize + kw) as i64 - pad;
                            if h >= 0 && h < in_h as i64 && w >= 0 && w < in_w as i64 {
                                let idx = ((b * channels_usize + c) * in_h + h as usize) * in_w
                                    + w as usize;
                                unsafe {
                                    sum += *x_p.add(idx);
                                }
                                count += 1;
                            }
                        }
                    }
                    let out_idx = ((b * channels_usize + c) * out_h + oh) * out_w + ow;
                    unsafe {
                        *o_p.add(out_idx) = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for b in 0..batch_size as usize {
            for c in 0..channels as usize {
                for oh in 0..out_height as usize {
                    for ow in 0..out_width as usize {
                        let mut sum = 0.0f32;
                        let mut count = 0i64;

                        for kh in 0..kernel_size as usize {
                            for kw in 0..kernel_size as usize {
                                let h = (oh * stride as usize + kh) as i64 - padding;
                                let w = (ow * stride as usize + kw) as i64 - padding;

                                if h >= 0 && h < in_height && w >= 0 && w < in_width {
                                    let idx = ((b * channels as usize + c) * in_height as usize
                                        + h as usize)
                                        * in_width as usize
                                        + w as usize;
                                    sum += unsafe { *x_ptr.add(idx) };
                                    count += 1;
                                }
                            }
                        }

                        let out_idx = ((b * channels as usize + c) * out_height as usize + oh)
                            * out_width as usize
                            + ow;
                        unsafe {
                            *out_ptr.add(out_idx) =
                                if count > 0 { sum / count as f32 } else { 0.0 };
                        }
                    }
                }
            }
        }
    }

    vec![output]
}
