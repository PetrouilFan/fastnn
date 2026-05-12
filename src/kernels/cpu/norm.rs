//! CPU norm kernels.

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

/// Fused layer norm forward: single-pass mean/variance, normalize, apply weight/bias.
/// Returns [output, mean, variance, x_hat].
/// Parallelized over outer dimensions (rows) with rayon.
/// Zero intermediate tensor allocations — writes directly into output buffers.
pub unsafe fn layer_norm_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let _normalized_shape_arg = args[1].shape_ref();
    let weight = if args.len() > 2 && args[2].numel() > 0 {
        Some(args[2])
    } else {
        None
    };
    let bias = if args.len() > 3 && args[3].numel() > 0 {
        Some(args[3])
    } else {
        None
    };
    let eps = if args.len() > 4 { args[4].item() } else { 1e-5 };

    let x_shape = x.shape_ref();
    let ndim = x_shape.len();
    let norm_dim = x_shape[ndim - 1] as usize;

    // Number of outer dimensions (product of all dims except last)
    let outer_size: usize = x_shape[..ndim - 1].iter().map(|&d| d as usize).product();
    let total = outer_size * norm_dim;

    let x_data = x.as_f32_slice();

    let mut output_data = vec![0.0f32; total];
    let mut mean_data = vec![0.0f32; outer_size];
    let mut var_data = vec![0.0f32; outer_size];
    let mut x_hat_data = vec![0.0f32; total];

    let w_data = weight.map(|w| w.as_f32_slice());
    let b_data = bias.map(|b| b.as_f32_slice());
    let nd = norm_dim;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let x_usize = x_data.as_ptr() as usize;
        let out_usize = output_data.as_mut_ptr() as usize;
        let xhat_usize = x_hat_data.as_mut_ptr() as usize;
        let mean_usize = mean_data.as_mut_ptr() as usize;
        let var_usize = var_data.as_mut_ptr() as usize;

        (0..outer_size).into_par_iter().for_each(|row| {
            let base = row * nd;

            // Two-pass: mean then variance (more numerically stable than single-pass)
            let mut sum = 0.0f32;
            for j in 0..nd {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    sum += *((x_usize + (base + j) * 4) as *const f32);
                }
            }
            let mean = sum / nd as f32;

            let mut sum_sq = 0.0f32;
            for j in 0..nd {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    let diff = *((x_usize + (base + j) * 4) as *const f32) - mean;
                    sum_sq += diff * diff;
                }
            }
            let var = sum_sq / nd as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            // SAFETY: The offset stays within the bounds of the allocated tensor storage.
            // The pointer is valid for this element access.
            unsafe {
                *((mean_usize + row * 4) as *mut f32) = mean;
                *((var_usize + row * 4) as *mut f32) = var;
            }

            // Normalize, apply weight/bias, store x_hat and output
            for j in 0..nd {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    let val = *((x_usize + (base + j) * 4) as *const f32);
                    let xn = (val - mean) * inv_std;
                    *((xhat_usize + (base + j) * 4) as *mut f32) = xn;
                    let mut out_val = xn;
                    if let Some(w) = w_data {
                        out_val *= w[j];
                    }
                    if let Some(b) = b_data {
                        out_val += b[j];
                    }
                    *((out_usize + (base + j) * 4) as *mut f32) = out_val;
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..outer_size {
            let base = row * nd;
            let mut sum = 0.0f32;
            for j in 0..nd {
                sum += x_data[base + j];
            }
            let mean = sum / nd as f32;

            let mut sum_sq = 0.0f32;
            for j in 0..nd {
                let diff = x_data[base + j] - mean;
                sum_sq += diff * diff;
            }
            let var = sum_sq / nd as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            mean_data[row] = mean;
            var_data[row] = var;

            for j in 0..nd {
                let xn = (x_data[base + j] - mean) * inv_std;
                x_hat_data[base + j] = xn;
                let mut out_val = xn;
                if let Some(w) = w_data {
                    out_val *= w[j];
                }
                if let Some(b) = b_data {
                    out_val += b[j];
                }
                output_data[base + j] = out_val;
            }
        }
    }

    // Reshape mean and variance to [outer_size, 1] for broadcasting compatibility
    let mut mean_shape = x_shape[..ndim - 1].to_vec();
    mean_shape.push(1);
    let var_shape = mean_shape.clone();

    let output = Tensor::from_vec(output_data, x_shape.to_vec());
    let mean = Tensor::from_vec(mean_data, mean_shape);
    let var = Tensor::from_vec(var_data, var_shape);
    let x_hat = Tensor::from_vec(x_hat_data, x_shape.to_vec());

    vec![output, mean, var, x_hat]
}

/// Fused RMSNorm kernel: single-pass computation combining x^2, mean, add(eps), sqrt, div, mul(weight).
/// Eliminates 6+ intermediate operations and allocations.
/// Returns the normalized output tensor.
pub unsafe fn rms_norm_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let weight = if args.len() > 1 && args[1].numel() > 0 {
        Some(args[1])
    } else {
        None
    };
    let eps = if args.len() > 2 { args[2].item() } else { 1e-5 };

    let x_shape = x.shape_ref();
    let ndim = x_shape.len();
    let norm_dim = x_shape[ndim - 1] as usize;

    // Number of outer dimensions (product of all dims except last)
    let outer_size: usize = x_shape[..ndim - 1].iter().map(|&d| d as usize).product();
    let total = outer_size * norm_dim;

    let x_data = x.as_f32_slice();
    let mut output_data = vec![0.0f32; total];

    let w_data = weight.map(|w| w.as_f32_slice());
    let nd = norm_dim;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let x_usize = x_data.as_ptr() as usize;
        let out_usize = output_data.as_mut_ptr() as usize;

        (0..outer_size).into_par_iter().for_each(|row| {
            let base = row * nd;

            // Compute mean of squares
            let mut sum_sq = 0.0f32;
            for j in 0..nd {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    let val = *((x_usize + (base + j) * 4) as *const f32);
                    sum_sq += val * val;
                }
            }
            let mean_sq = sum_sq / nd as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();

            // Normalize and apply weight in one pass
            for j in 0..nd {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    let val = *((x_usize + (base + j) * 4) as *const f32);
                    let mut out_val = val * inv_rms;
                    if let Some(w) = w_data {
                        out_val *= w[j];
                    }
                    *((out_usize + (base + j) * 4) as *mut f32) = out_val;
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..outer_size {
            let base = row * nd;

            // Compute mean of squares
            let mut sum_sq = 0.0f32;
            for j in 0..nd {
                let val = x_data[base + j];
                sum_sq += val * val;
            }
            let mean_sq = sum_sq / nd as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();

            // Normalize and apply weight in one pass
            for j in 0..nd {
                let val = x_data[base + j];
                let mut out_val = val * inv_rms;
                if let Some(w) = w_data {
                    out_val *= w[j];
                }
                output_data[base + j] = out_val;
            }
        }
    }

    let output = Tensor::from_vec(output_data, x_shape.to_vec());
    vec![output]
}

pub unsafe fn batch_norm_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let weight = if args.len() > 1 && args[1].numel() > 0 {
        Some(args[1])
    } else {
        None
    };
    let bias = if args.len() > 2 && args[2].numel() > 0 {
        Some(args[2])
    } else {
        None
    };
    let running_mean = if args.len() > 3 && args[3].numel() > 0 {
        Some(args[3])
    } else {
        None
    };
    let running_var = if args.len() > 4 && args[4].numel() > 0 {
        Some(args[4])
    } else {
        None
    };
    let training = if args.len() > 5 {
        args[5].item() != 0.0
    } else {
        false
    };
    let eps = if args.len() > 6 {
        args[6].item() as f64
    } else {
        1e-5
    };

    let x_shape = x.shape_ref();
    let ndim = x_shape.len();

    // BatchNorm normalizes across the channel dimension (dim=1)
    let num_channels = if ndim > 1 { x_shape[1] } else { 1 };
    let batch_size = x_shape[0];
    let spatial_size: i64 = if ndim > 2 {
        x_shape[2..].iter().product()
    } else {
        1
    };
    let total_per_channel = batch_size * spatial_size; // Elements per feature

    // Get weight and bias (default: gamma=1, beta=0)
    let w_data = weight.map(|w| w.as_f32_slice());
    let b_data = bias.map(|b| b.as_f32_slice());

    // Create output tensor with same shape as input
    let mut output = Tensor::empty(x_shape.to_vec(), x.dtype(), x.device());

    // Get raw pointers for fast access
    let x_ptr = x.data_ptr() as *const f32;
    let x_addr = x_ptr as usize; // Convert to usize for Sync
    let out_inner = Arc::make_mut(&mut output.inner);
    let out_storage = Arc::make_mut(&mut out_inner.storage);
    let Storage::Cpu(out_cpu) = out_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut out_cpu.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;
    let out_addr = out_ptr as usize; // Convert to usize for Sync

    // Helper function to compute mean/variance using Welford's algorithm
    // with optional SIMD acceleration
    // Uses on-the-fly index computation to avoid Vec allocation
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    unsafe fn compute_stats_for_channel(
        x_ptr: *const f32,
        num_channels: usize,
        spatial_size: usize,
        channel: usize,
        _batch_size: usize,
        total_per_channel: usize,
    ) -> (f32, f32) {
        use std::arch::x86_64::*;
        ensure_daz_ftz();

        if is_x86_feature_detected!("avx2") && total_per_channel >= 8 {
            let mut sum = _mm256_setzero_ps();
            let mut sum_sq = _mm256_setzero_ps();
            let mut total_sum: f64 = 0.0;
            let mut total_sum_sq: f64 = 0.0;

            for b in 0.._batch_size {
                let batch_offset = (b * num_channels + channel) * spatial_size;
                let mut s = 0;

                while s + 8 <= spatial_size {
                    let v = _mm256_loadu_ps(x_ptr.add(batch_offset + s));
                    sum = _mm256_add_ps(sum, v);
                    sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
                    s += 8;
                }

                for s_rem in s..spatial_size {
                    let val = *x_ptr.add(batch_offset + s_rem);
                    total_sum += val as f64;
                    total_sum_sq += (val as f64) * (val as f64);
                }
            }

            let mut sum_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
            _mm256_storeu_ps(sum_arr.as_mut_ptr() as *mut f32, sum);
            let sum_arr = sum_arr.assume_init();
            for &v in &sum_arr {
                total_sum += v as f64;
            }

            let mut sum_sq_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
            _mm256_storeu_ps(sum_sq_arr.as_mut_ptr() as *mut f32, sum_sq);
            let sum_sq_arr = sum_sq_arr.assume_init();
            for &v in &sum_sq_arr {
                total_sum_sq += v as f64;
            }

            let mean = (total_sum / total_per_channel as f64) as f32;
            let var = (total_sum_sq / total_per_channel as f64) as f32 - mean * mean;
            (mean, var.max(0.0))
        } else {
            let mut count = 0.0f32;
            let mut mean = 0.0f32;
            let mut m2 = 0.0f32;

            for b in 0.._batch_size {
                let batch_offset = (b * num_channels + channel) * spatial_size;
                for s in 0..spatial_size {
                    let val = *x_ptr.add(batch_offset + s);
                    count += 1.0;
                    let delta = val - mean;
                    mean += delta / count;
                    let delta2 = val - mean;
                    m2 += delta * delta2;
                }
            }

            let var = m2 / count;
            (mean, var.max(0.0))
        }
    }

    // Non-SIMD version
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    #[inline]
    unsafe fn compute_stats_for_channel(
        x_ptr: *const f32,
        num_channels: usize,
        spatial_size: usize,
        channel: usize,
        _batch_size: usize,
        _total_per_channel: usize,
    ) -> (f32, f32) {
        let mut count = 0.0f32;
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;

        for b in 0.._batch_size {
            let batch_offset = (b * num_channels + channel) * spatial_size;
            for s in 0..spatial_size {
                let val = *x_ptr.add(batch_offset + s);
                count += 1.0;
                let delta = val - mean;
                mean += delta / count;
                let delta2 = val - mean;
                m2 += delta * delta2;
            }
        }

        let var = m2 / count;
        (mean, var.max(0.0))
    }

    if training {
        // Training mode: compute batch statistics
        // Parallelize over channels using rayon
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            (0..num_channels as usize).into_par_iter().for_each(|c| {
                let x_ptr = x_addr as *const f32;
                let out_ptr = out_addr as *mut f32;

                // Compute mean and variance using SIMD-accelerated Welford (no Vec allocation)
                // SAFETY: Each rayon iteration accesses disjoint memory regions because
                // the loop index maps to non-overlapping chunks of the buffer.
                let (mean, var) = unsafe {
                    compute_stats_for_channel(
                        x_ptr,
                        num_channels as usize,
                        spatial_size as usize,
                        c,
                        batch_size as usize,
                        total_per_channel as usize,
                    )
                };
                let inv_std = 1.0 / (var + eps as f32).sqrt();

                // Get gamma and beta for this channel
                let gamma = w_data.map_or(1.0, |w| w[c]);
                let beta = b_data.map_or(0.0, |b| b[c]);

                // Normalize, scale, and shift in one pass
                for i in 0..total_per_channel as usize {
                    let b_idx = i / spatial_size as usize;
                    let s = i % spatial_size as usize;
                    let idx = (b_idx * num_channels as usize + c) * spatial_size as usize + s;
                    // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                    // The pointer is valid for this element access.
                    let val = unsafe { *x_ptr.add(idx) };
                    let normed = (val - mean) * inv_std;
                    // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                    // The pointer is valid for this element access.
                    unsafe { *out_ptr.add(idx) = gamma * normed + beta };
                }
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for c in 0..num_channels as usize {
                // Compute mean and variance using SIMD-accelerated Welford (no Vec allocation)
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                let (mean, var) = unsafe {
                    compute_stats_for_channel(
                        x_ptr,
                        num_channels as usize,
                        spatial_size as usize,
                        c,
                        batch_size as usize,
                        total_per_channel as usize,
                    )
                };
                let inv_std = 1.0 / (var + eps as f32).sqrt();

                // Get gamma and beta for this channel
                let gamma = w_data.map_or(1.0, |w| w[c]);
                let beta = b_data.map_or(0.0, |b| b[c]);

                // Normalize, scale, and shift in one pass
                for i in 0..total_per_channel as usize {
                    let b_idx = i / spatial_size as usize;
                    let s = i % spatial_size as usize;
                    let idx = (b_idx * num_channels as usize + c) * spatial_size as usize + s;
                    // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                    // The pointer is valid for this element access.
                    let val = unsafe { *x_ptr.add(idx) };
                    let normed = (val - mean) * inv_std;
                    // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                    // The pointer is valid for this element access.
                    unsafe { *out_ptr.add(idx) = gamma * normed + beta };
                }
            }
        }
    } else {
        // Inference mode: use running statistics (no computation needed)
        let run_mean = running_mean.expect("running_mean required for inference");
        let run_var = running_var.expect("running_var required for inference");
        let mean_data = run_mean.as_f32_slice();
        let var_data = run_var.as_f32_slice();

        // Parallelize over channels
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            (0..num_channels as usize).into_par_iter().for_each(|c| {
                let x_ptr = x_addr as *const f32;
                let out_ptr = out_addr as *mut f32;

                let mean = mean_data[c];
                let inv_std = 1.0 / (var_data[c] + eps as f32).sqrt();
                let gamma = w_data.map_or(1.0, |w| w[c]);
                let beta = b_data.map_or(0.0, |b| b[c]);

                // Normalize, scale, and shift in one pass
                for b in 0..batch_size as usize {
                    for s in 0..spatial_size as usize {
                        let idx = (b * num_channels as usize + c) * spatial_size as usize + s;
                        // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                        // The pointer is valid for this element access.
                        let val = unsafe { *x_ptr.add(idx) };
                        let normed = (val - mean) * inv_std;
                        // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                        // The pointer is valid for this element access.
                        unsafe { *out_ptr.add(idx) = gamma * normed + beta };
                    }
                }
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for c in 0..num_channels as usize {
                let mean = mean_data[c];
                let inv_std = 1.0 / (var_data[c] + eps as f32).sqrt();
                let gamma = w_data.map_or(1.0, |w| w[c]);
                let beta = b_data.map_or(0.0, |b| b[c]);

                // Normalize, scale, and shift in one pass
                for b in 0..batch_size as usize {
                    for s in 0..spatial_size as usize {
                        let idx = (b * num_channels as usize + c) * spatial_size as usize + s;
                        // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                        // The pointer is valid for this element access.
                        let val = unsafe { *x_ptr.add(idx) };
                        let normed = (val - mean) * inv_std;
                        // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                        // The pointer is valid for this element access.
                        unsafe { *out_ptr.add(idx) = gamma * normed + beta };
                    }
                }
            }
        }
    }

    vec![output]
}

/// Fused layer norm + GELU: single-pass with GELU activation.
/// Returns the normalized output tensor.
pub unsafe fn fused_layer_norm_gelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let weight = if args.len() > 1 && args[1].numel() > 0 {
        Some(args[1])
    } else {
        None
    };
    let bias = if args.len() > 2 && args[2].numel() > 0 {
        Some(args[2])
    } else {
        None
    };
    let eps = if args.len() > 3 { args[3].item() } else { 1e-5 };

    let x_shape = x.shape_ref();
    let ndim = x_shape.len();
    let norm_dim = x_shape[ndim - 1] as usize;

    let outer_size: usize = x_shape[..ndim - 1].iter().map(|&d| d as usize).product();
    let total = outer_size * norm_dim;

    let x_data = x.as_f32_slice();
    let mut output_data = vec![0.0f32; total];

    let w_data = weight.map(|w| w.as_f32_slice());
    let b_data = bias.map(|b| b.as_f32_slice());

    const SQRT_2_OVER_PI: f32 = 0.7978846;
    const GELU_COEFF: f32 = 0.044715;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let x_usize = x_data.as_ptr() as usize;
        let out_usize = output_data.as_mut_ptr() as usize;

        (0..outer_size).into_par_iter().for_each(|row| {
            let base = row * norm_dim;

            // Compute mean
            let mut sum = 0.0f32;
            for j in 0..norm_dim {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    sum += *((x_usize as *const f32).add(base + j));
                }
            }
            let mean = sum / norm_dim as f32;

            // Compute variance
            let mut sum_sq = 0.0f32;
            for j in 0..norm_dim {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    let diff = *((x_usize as *const f32).add(base + j)) - mean;
                    sum_sq += diff * diff;
                }
            }
            let var = sum_sq / norm_dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            // Normalize, apply weight/bias, then GELU
            for j in 0..norm_dim {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    let val = *((x_usize as *const f32).add(base + j));
                    let xn = (val - mean) * inv_std;
                    let mut out_val = xn;
                    if let Some(w) = w_data {
                        out_val *= w[j];
                    }
                    if let Some(b) = b_data {
                        out_val += b[j];
                    }
                    // GELU activation
                    let x3 = out_val * out_val * out_val;
                    let t = (SQRT_2_OVER_PI * (out_val + GELU_COEFF * x3)).tanh();
                    let gelu = 0.5 * out_val * (1.0 + t);
                    *((out_usize as *mut f32).add(base + j)) = gelu;
                }
            }
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..outer_size {
            let base = row * norm_dim;

            // Compute mean
            let mut sum = 0.0f32;
            for j in 0..norm_dim {
                sum += x_data[base + j];
            }
            let mean = sum / norm_dim as f32;

            // Compute variance
            let mut sum_sq = 0.0f32;
            for j in 0..norm_dim {
                let diff = x_data[base + j] - mean;
                sum_sq += diff * diff;
            }
            let var = sum_sq / norm_dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            // Normalize, apply weight/bias, then GELU
            for j in 0..norm_dim {
                let xn = (x_data[base + j] - mean) * inv_std;
                let mut out_val = xn;
                if let Some(w) = w_data {
                    out_val *= w[j];
                }
                if let Some(b) = b_data {
                    out_val += b[j];
                }
                // GELU activation
                let x3 = out_val * out_val * out_val;
                let t = (SQRT_2_OVER_PI * (out_val + GELU_COEFF * x3)).tanh();
                output_data[base + j] = 0.5 * out_val * (1.0 + t);
            }
        }
    }

    let output_shape = x_shape.to_vec();
    vec![Tensor::from_vec(output_data, output_shape)]
}

/// Fused RMSNorm + GELU: single-pass with GELU activation.
/// Returns the normalized output tensor.
pub unsafe fn fused_rms_norm_gelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let weight = if args.len() > 1 && args[1].numel() > 0 {
        Some(args[1])
    } else {
        None
    };
    let eps = if args.len() > 2 { args[2].item() } else { 1e-5 };

    let x_shape = x.shape_ref();
    let ndim = x_shape.len();
    let norm_dim = x_shape[ndim - 1] as usize;

    let outer_size: usize = x_shape[..ndim - 1].iter().map(|&d| d as usize).product();
    let total = outer_size * norm_dim;

    let x_data = x.as_f32_slice();
    let mut output_data = vec![0.0f32; total];

    let w_data = weight.map(|w| w.as_f32_slice());

    const SQRT_2_OVER_PI: f32 = 0.7978846;
    const GELU_COEFF: f32 = 0.044715;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let x_usize = x_data.as_ptr() as usize;
        let out_usize = output_data.as_mut_ptr() as usize;

        (0..outer_size).into_par_iter().for_each(|row| {
            let base = row * norm_dim;

            // Compute mean of squares
            let mut sum_sq = 0.0f32;
            for j in 0..norm_dim {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    let val = *((x_usize + (base + j) * 4) as *const f32);
                    sum_sq += val * val;
                }
            }
            let mean_sq = sum_sq / norm_dim as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();

            // Normalize, apply weight, then GELU
            for j in 0..norm_dim {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    let val = *((x_usize + (base + j) * 4) as *const f32);
                    let mut out_val = val * inv_rms;
                    if let Some(w) = w_data {
                        out_val *= w[j];
                    }
                    // GELU activation
                    let x3 = out_val * out_val * out_val;
                    let t = (SQRT_2_OVER_PI * (out_val + GELU_COEFF * x3)).tanh();
                    let gelu = 0.5 * out_val * (1.0 + t);
                    *((out_usize + (base + j) * 4) as *mut f32) = gelu;
                }
            }
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..outer_size {
            let base = row * norm_dim;

            // Compute mean of squares
            let mut sum_sq = 0.0f32;
            for j in 0..norm_dim {
                sum_sq += x_data[base + j] * x_data[base + j];
            }
            let mean_sq = sum_sq / norm_dim as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();

            // Normalize, apply weight, then GELU
            for j in 0..norm_dim {
                let val = x_data[base + j];
                let mut out_val = val * inv_rms;
                if let Some(w) = w_data {
                    out_val *= w[j];
                }
                // GELU activation
                let x3 = out_val * out_val * out_val;
                let t = (SQRT_2_OVER_PI * (out_val + GELU_COEFF * x3)).tanh();
                output_data[base + j] = 0.5 * out_val * (1.0 + t);
            }
        }
    }

    vec![Tensor::from_vec(output_data, x_shape.to_vec())]
}
