//! CPU losses kernels.

#![allow(unused_imports)]
#![allow(clippy::missing_safety_doc)]

use crate::autograd::{AutogradMeta, Edge, Node};
use crate::dispatcher::{register, DispatchKey, KernelFn};
use crate::iterator::TensorIterator;
use crate::kernels::blas::{
    matmul_blas, matmul_blas_into, matmul_blas_with_transpose,
    matmul_blas_with_transpose_into, MIN_BLAS_SIZE,
};
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use std::sync::Arc;
use super::*;

pub unsafe fn mse_loss_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let pred = args[0];
    let target = args[1];
    let reduction = if args.len() > 2 {
        match args[2].item() as i32 {
            0 => "none",
            1 => "mean",
            _ => "sum",
        }
    } else {
        "mean"
    };

    let diff = pred.sub(target);
    let loss = diff.mul(&diff);

    match reduction {
        "none" => vec![loss],
        "mean" => {
            // Flatten then single sum instead of O(dims) kernel dispatches
            let numel = loss.numel() as usize;
            let flat = loss.reshape(vec![numel as i64]);
            let mut result = flat.sum(0, false);
            result.mul_scalar_(1.0 / numel as f32);
            vec![result]
        }
        "sum" => {
            // Flatten then single sum
            let numel = loss.numel() as usize;
            let flat = loss.reshape(vec![numel as i64]);
            vec![flat.sum(0, false)]
        }
        _ => vec![loss.sum(0, false)],
    }
}

pub unsafe fn cross_entropy_loss_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let logits = args[0];
    let targets = args[1];
    let reduction = if args.len() > 2 {
        match args[2].item() as i32 {
            0 => "none",
            1 => "mean",
            _ => "sum",
        }
    } else {
        "mean"
    };

    let logits_data = logits.as_f32_slice();
    let targets_data = targets.as_f32_slice();

    let batch_size = logits.shape()[0] as usize;
    let num_classes = logits.shape()[1] as usize;

    // Fused log-sum-exp forward pass, parallelized over batch rows.
    // For each row i:
    //   max_val = max(logits[i])
    //   sum_exp = sum(exp(logits[i][j] - max_val))
    //   loss[i] = -(logits[i][target[i]] - max_val - ln(sum_exp))
    let mut losses = vec![0.0f32; batch_size];

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let logits_usize = logits_data.as_ptr() as usize;
        let targets_usize = targets_data.as_ptr() as usize;
        let losses_usize = losses.as_mut_ptr() as usize;
        let nc = num_classes;

        (0..batch_size).into_par_iter().for_each(|b| {
            let base = b * nc;
            let mut max_val = f32::NEG_INFINITY;
            let mut j = 0;

            // Find max with 4-unroll
            while j + 4 <= nc {
                unsafe {
                    let v0 = *((logits_usize + (base + j) * 4) as *const f32);
                    let v1 = *((logits_usize + (base + j + 1) * 4) as *const f32);
                    let v2 = *((logits_usize + (base + j + 2) * 4) as *const f32);
                    let v3 = *((logits_usize + (base + j + 3) * 4) as *const f32);
                    max_val = max_val.max(v0).max(v1).max(v2).max(v3);
                }
                j += 4;
            }
            while j < nc {
                unsafe {
                    max_val = max_val.max(*((logits_usize + (base + j) * 4) as *const f32));
                }
                j += 1;
            }

            // sum_exp with 4-unroll
            let mut sum_exp = 0.0f32;
            j = 0;
            while j + 4 <= nc {
                unsafe {
                    sum_exp += (*((logits_usize + (base + j) * 4) as *const f32) - max_val).exp();
                    sum_exp +=
                        (*((logits_usize + (base + j + 1) * 4) as *const f32) - max_val).exp();
                    sum_exp +=
                        (*((logits_usize + (base + j + 2) * 4) as *const f32) - max_val).exp();
                    sum_exp +=
                        (*((logits_usize + (base + j + 3) * 4) as *const f32) - max_val).exp();
                }
                j += 4;
            }
            while j < nc {
                unsafe {
                    sum_exp += (*((logits_usize + (base + j) * 4) as *const f32) - max_val).exp();
                }
                j += 1;
            }

            let target_class = unsafe { *((targets_usize + b * 4) as *const f32) } as usize;
            let log_sum_exp = sum_exp.ln();
            let class_logit =
                unsafe { *((logits_usize + (base + target_class) * 4) as *const f32) };
            let loss = log_sum_exp + max_val - class_logit;
            unsafe {
                *((losses_usize + b * 4) as *mut f32) = if loss.is_finite() { loss } else { 0.0 };
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for b in 0..batch_size {
            let base = b * num_classes;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..num_classes {
                max_val = max_val.max(logits_data[base + j]);
            }

            let mut sum_exp = 0.0f32;
            for j in 0..num_classes {
                sum_exp += (logits_data[base + j] - max_val).exp();
            }

            let target_class = targets_data[b] as usize;
            let log_sum_exp = sum_exp.ln();
            let class_logit = logits_data[base + target_class];
            let loss = log_sum_exp + max_val - class_logit;
            losses[b] = if loss.is_finite() { loss } else { 0.0 };
        }
    }

    match reduction {
        "none" => {
            let output = Tensor::from_vec(losses, vec![batch_size as i64]);
            vec![output]
        }
        "mean" => {
            let total_loss: f32 = losses.iter().sum();
            vec![Tensor::from_scalar(total_loss / batch_size as f32)]
        }
        _ => {
            let total_loss: f32 = losses.iter().sum();
            vec![Tensor::from_scalar(total_loss)]
        }
    }
}

pub unsafe fn bce_with_logits_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let target = args[1];
    let numel = input.inner.numel() as usize;
    let input_data = input.as_f32_slice();
    let target_data = target.as_f32_slice();
    let mut loss = 0.0f32;
    for i in 0..numel {
        let x = input_data[i];
        let t = target_data[i];
        let max_val = if x > 0.0 { x } else { 0.0 };
        loss += max_val - x * t + (1.0_f32 + (-x.abs()).exp()).ln();
    }
    let avg_loss = loss / numel as f32;
    vec![Tensor::from_vec(vec![avg_loss], vec![1])]
}

pub unsafe fn huber_loss_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let target = args[1];
    let delta = args[2].item();
    let numel = input.inner.numel() as usize;
    let input_data = input.as_f32_slice();
    let target_data = target.as_f32_slice();
    let mut loss = 0.0f32;
    for i in 0..numel {
        let diff = input_data[i] - target_data[i];
        let abs_diff = diff.abs();
        loss += if abs_diff < delta {
            0.5 * diff * diff
        } else {
            delta * (abs_diff - 0.5 * delta)
        };
    }
    let avg_loss = loss / numel as f32;
    vec![Tensor::from_vec(vec![avg_loss], vec![1])]
}

