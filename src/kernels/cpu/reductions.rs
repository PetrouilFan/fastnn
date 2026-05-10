//! CPU reductions kernels.

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

pub unsafe fn sum_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let a_shape = a.shape_ref();
        let ndim = a_shape.len() as i32;
        // Handle negative dimensions
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape_ref();
    let ndim = a_shape.len();

    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    if output_shape.is_empty() {
        output_shape = vec![1];
    }

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let a_ptr = a.data_ptr_f32();

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let dim_size = a_shape[dim] as usize;
    let mut strides_before = 1i64;
    let mut strides_after = 1i64;

    for i in 0..dim {
        strides_before *= a_shape[i];
    }
    for i in (dim + 1)..ndim {
        strides_after *= a_shape[i];
    }

    let total_blocks = (strides_before * strides_after) as usize;
    let a_numel = a.numel() as usize;

    // Fast path: contiguous sum along last dimension (2D only)
    if dim == ndim - 1 && ndim == 2 && a.is_contiguous() && a.inner.storage_offset == 0 {
        return vec![sum_last_dim_contiguous(a, dim_size, a_shape[0] as usize)];
    }

    #[cfg(feature = "parallel")]
    {
        if total_blocks > 64 && dim_size > 8 {
            let a_slice: &[f32] = unsafe { std::slice::from_raw_parts(a_ptr, a_numel) };
            let results: Vec<f32> = (0..total_blocks)
                .into_par_iter()
                .map(|block| {
                    let block_before = block / (strides_after as usize);
                    let block_after = block % (strides_after as usize);

                    let mut sum_val = 0.0f32;
                    for d in 0..dim_size {
                        let linear_idx =
                            (block_before * dim_size + d) * strides_after as usize + block_after;
                        if linear_idx < a_numel {
                            sum_val += a_slice[linear_idx];
                        }
                    }
                    sum_val
                })
                .collect();

            for (i, &sum_val) in results.iter().enumerate() {
                unsafe {
                    *out_ptr.add(i) = sum_val;
                }
            }
            return vec![output];
        }
    }

    for block in 0..total_blocks {
        let block_before = block / (strides_after as usize);
        let block_after = block % (strides_after as usize);

        let mut sum_val = 0.0f32;
        for d in 0..dim_size {
            let linear_idx = (block_before * dim_size + d) * strides_after as usize + block_after;
            if linear_idx < a_numel {
                unsafe {
                    sum_val += *a_ptr.add(linear_idx);
                }
            }
        }

        unsafe {
            *out_ptr.add(block) = sum_val;
        }
    }

    vec![output]
}

pub unsafe fn min_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let a_shape = a.shape_ref();
        let ndim = a_shape.len() as i32;
        // Handle negative dimensions
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape_ref();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    if output_shape.is_empty() {
        output_shape = vec![1];
    }

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let a_ptr = a.data_ptr_f32();
    let a_numel = a.numel() as usize;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let dim_size = a_shape[dim] as usize;
    let mut strides_before = 1i64;
    let mut strides_after = 1i64;

    for i in 0..dim {
        strides_before *= a_shape[i];
    }
    for i in (dim + 1)..ndim {
        strides_after *= a_shape[i];
    }

    let total_blocks = (strides_before * strides_after) as usize;

    #[cfg(feature = "parallel")]
    {
        if total_blocks > 64 && dim_size > 8 {
            let a_slice: &[f32] = unsafe { std::slice::from_raw_parts(a_ptr, a_numel) };
            let results: Vec<f32> = (0..total_blocks)
                .into_par_iter()
                .map(|block| {
                    let block_before = block / (strides_after as usize);
                    let block_after = block % (strides_after as usize);

                    let mut min_val = f32::MAX;
                    for d in 0..dim_size {
                        let linear_idx =
                            (block_before * dim_size + d) * strides_after as usize + block_after;
                        if linear_idx < a_numel {
                            min_val = min_val.min(a_slice[linear_idx]);
                        }
                    }
                    min_val
                })
                .collect();

            for (i, &min_val) in results.iter().enumerate() {
                unsafe {
                    *out_ptr.add(i) = min_val;
                }
            }
            return vec![output];
        }
    }

    for block in 0..total_blocks {
        let block_before = block / (strides_after as usize);
        let block_after = block % (strides_after as usize);

        let mut min_val = f32::MAX;
        for d in 0..dim_size {
            let linear_idx = (block_before * dim_size + d) * strides_after as usize + block_after;
            if linear_idx < a_numel {
                unsafe {
                    min_val = min_val.min(*a_ptr.add(linear_idx));
                }
            }
        }

        unsafe {
            *out_ptr.add(block) = min_val;
        }
    }

    vec![output]
}

pub unsafe fn mean_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let a_shape = a.shape_ref();
        let ndim = a_shape.len() as i32;
        // Handle negative dimensions
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape_ref();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let dim_size = a_shape[dim] as f32;

    let sum_result = sum_kernel(args);
    let mut sum_tensor = sum_result[0].clone();

    // sum_kernel already handles keepdim correctly
    // If keepdim is true, sum_tensor has shape [..., 1, ...]
    // If keepdim is false, sum_tensor has shape with the dimension removed
    // We need to unsqueeze for broadcasting with the input, then squeeze back if needed
    let needs_unsqueeze = !keepdim;
    if needs_unsqueeze {
        sum_tensor = sum_tensor.unsqueeze(dim);
    }

    let scale = Tensor::full(vec![], 1.0 / dim_size, DType::F32, Device::Cpu);
    let result = sum_tensor * scale;

    // If we unsqueezed for broadcasting, remove it now
    let result = if needs_unsqueeze {
        result.squeeze(Some(dim))
    } else {
        result
    };

    vec![result]
}

pub unsafe fn max_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let a_shape = a.shape_ref();
        let ndim = a_shape.len() as i32;
        // Handle negative dimensions
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape_ref();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    if output_shape.is_empty() {
        output_shape = vec![1];
    }

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let a_ptr = a.data_ptr_f32();

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let dim_size = a_shape[dim] as usize;
    let mut strides_before = 1i64;
    let mut strides_after = 1i64;

    for i in 0..dim {
        strides_before *= a_shape[i];
    }
    for i in (dim + 1)..ndim {
        strides_after *= a_shape[i];
    }

    let total_blocks = strides_before as usize * strides_after as usize;
    let a_usize = a_ptr as usize;
    let out_usize = out_ptr as usize;

    #[cfg(feature = "parallel")]
    {
        if total_blocks > 64 {
            use rayon::prelude::*;
            (0..total_blocks).into_par_iter().for_each(|block| {
                let mut max_val = f32::NEG_INFINITY;
                let a_p = a_usize as *const f32;
                let ds = dim_size;
                let sa = strides_after as usize;
                for i in 0..ds {
                    let a_idx = (block / sa) * ds * sa + i * sa + block % sa;
                    unsafe {
                        let val = *a_p.add(a_idx);
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
                unsafe {
                    *(out_usize as *mut f32).add(block) = max_val;
                }
            });
            return vec![output];
        }
    }

    for block in 0..total_blocks {
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..dim_size {
            let a_idx = (block / (strides_after as usize)) * dim_size * (strides_after as usize)
                + i * (strides_after as usize)
                + block % (strides_after as usize);
            unsafe {
                let val = *a_ptr.add(a_idx);
                if val > max_val {
                    max_val = val;
                }
            }
        }
        unsafe {
            *out_ptr.add(block) = max_val;
        }
    }

    vec![output]
}

pub unsafe fn softmax_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let x_shape = x.shape_ref();
        let ndim = x_shape.len() as i32;
        // Handle negative dimensions
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };

    let x_shape = x.shape_ref();
    let ndim = x_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let dim_size = x_shape[dim] as usize;

    if dim == ndim - 1 && x.is_contiguous() {
        return vec![softmax_last_dim_simd(x, dim_size)];
    }

    let max_vals = max_kernel(&[
        x,
        &Tensor::from_scalar(dim as f32),
        &Tensor::from_scalar(1.0),
    ])[0]
        .clone();
    // max_kernel with keepdim=true already keeps the dimension, so no need to unsqueeze
    let max_exp = x.sub(&max_vals).exp();
    let sum_exp = max_exp.sum(dim as i32, true);

    vec![max_exp.div(&sum_exp)]
}

#[inline]
pub fn softmax_last_dim_simd(x: &Tensor, dim_size: usize) -> Tensor {
    let x_shape = x.shape_ref();
    let _batch_size: i64 = x_shape[..x_shape.len() - 1].iter().product();

    let x_ptr = x.data_ptr() as *const f32;
    let numel = x.numel() as usize;

    let mut output = Tensor::empty(x_shape.to_vec(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    #[cfg(feature = "parallel")]
    {
        let a_slice = unsafe { std::slice::from_raw_parts(x_ptr, numel) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

        out_slice
            .par_chunks_mut(dim_size)
            .zip(a_slice.par_chunks(dim_size))
            .for_each(|(out_row, x_row)| {
                let mut max_val = f32::MIN;

                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            let mut max_vec = _mm256_set1_ps(f32::MIN);
                            let mut i = 0;
                            while i + 8 <= dim_size {
                                let vec = _mm256_loadu_ps(x_row.as_ptr().add(i));
                                max_vec = _mm256_max_ps(max_vec, vec);
                                i += 8;
                            }
                            // Also handle 4-wide tail for small dims
                            if i + 4 <= dim_size {
                                let vec128 = _mm_loadu_ps(x_row.as_ptr().add(i));
                                let hi = _mm256_extractf128_ps(max_vec, 1);
                                let lo = _mm256_castps256_ps128(max_vec);
                                let merged_lo = _mm_max_ps(lo, vec128);
                                max_vec =
                                    _mm256_insertf128_ps(_mm256_castps128_ps256(merged_lo), hi, 1);
                                i += 4;
                            }
                            let mut max_arr = [0.0f32; 8];
                            _mm256_storeu_ps(max_arr.as_mut_ptr(), max_vec);
                            for j in 0..8 {
                                max_val = max_val.max(max_arr[j]);
                            }
                            // Handle remaining elements
                            for j in i..dim_size {
                                max_val = max_val.max(x_row[j]);
                            }
                        }
                    } else {
                        for j in 0..dim_size {
                            max_val = max_val.max(x_row[j]);
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for j in 0..dim_size {
                        max_val = max_val.max(x_row[j]);
                    }
                }

                // Fused pass: compute exp(x-max), store to output, and accumulate sum
                let mut sum_exp = 0.0f32;

                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            let max_vec = _mm256_set1_ps(max_val);
                            let mut sum_vec = _mm256_setzero_ps();
                            let mut j = 0;
                            while j + 8 <= dim_size {
                                let x_vec = _mm256_loadu_ps(x_row.as_ptr().add(j));
                                let shifted = _mm256_sub_ps(x_vec, max_vec);
                                let exp_vec = fast_exp_avx2(shifted);
                                sum_vec = _mm256_add_ps(sum_vec, exp_vec);

                                _mm256_storeu_ps(out_row.as_mut_ptr().add(j), exp_vec);
                                j += 8;
                            }
                            sum_exp = hsum256_ps(sum_vec);
                            for j2 in j..dim_size {
                                let e = (x_row[j2] - max_val).exp();
                                sum_exp += e;
                                out_row[j2] = e;
                            }
                        }
                    } else {
                        for j in 0..dim_size {
                            let e = (x_row[j] - max_val).exp();
                            sum_exp += e;
                            out_row[j] = e;
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for j in 0..dim_size {
                        let e = (x_row[j] - max_val).exp();
                        sum_exp += e;
                        out_row[j] = e;
                    }
                }

                let inv_sum = 1.0 / sum_exp;

                // Normalize: multiply stored exp values by inv_sum
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            let inv_vec = _mm256_set1_ps(inv_sum);
                            let mut j = 0;
                            while j + 8 <= dim_size {
                                let exp_vec = _mm256_loadu_ps(out_row.as_ptr().add(j));
                                let result = _mm256_mul_ps(exp_vec, inv_vec);
                                _mm256_storeu_ps(out_row.as_mut_ptr().add(j), result);
                                j += 8;
                            }
                            for j2 in j..dim_size {
                                out_row[j2] *= inv_sum;
                            }
                        }
                    } else {
                        for j in 0..dim_size {
                            out_row[j] *= inv_sum;
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for j in 0..dim_size {
                        out_row[j] *= inv_sum;
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let num_rows = numel / dim_size;
        let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, numel) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

        for row in 0..num_rows {
            let row_start = row * dim_size;
            let x_row = &x_slice[row_start..row_start + dim_size];
            let out_row = &mut out_slice[row_start..row_start + dim_size];

            let mut max_val = f32::MIN;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        let mut max_vec = _mm256_set1_ps(f32::MIN);
                        let mut j = 0;
                        while j + 8 <= dim_size {
                            let vec = _mm256_loadu_ps(x_row.as_ptr().add(j));
                            max_vec = _mm256_max_ps(max_vec, vec);
                            j += 8;
                        }
                        let mut max_arr = [0.0f32; 8];
                        _mm256_storeu_ps(max_arr.as_mut_ptr(), max_vec);
                        for k in 0..8 {
                            max_val = max_val.max(max_arr[k]);
                        }
                        for k in j..dim_size {
                            max_val = max_val.max(x_row[k]);
                        }
                    }
                } else {
                    for j in 0..dim_size {
                        max_val = max_val.max(x_row[j]);
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for j in 0..dim_size {
                    max_val = max_val.max(x_row[j]);
                }
            }

            // Fused pass: compute exp(x-max), store to output, and accumulate sum
            let mut sum_exp = 0.0f32;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        let max_vec = _mm256_set1_ps(max_val);
                        let mut sum_vec = _mm256_setzero_ps();
                        let mut j = 0;
                        while j + 8 <= dim_size {
                            let x_vec = _mm256_loadu_ps(x_row.as_ptr().add(j));
                            let shifted = _mm256_sub_ps(x_vec, max_vec);
                            let exp_vec = fast_exp_avx2(shifted);
                            sum_vec = _mm256_add_ps(sum_vec, exp_vec);

                            _mm256_storeu_ps(out_row.as_mut_ptr().add(j), exp_vec);
                            j += 8;
                        }
                        sum_exp = hsum256_ps(sum_vec);
                        for j2 in j..dim_size {
                            let e = (x_row[j2] - max_val).exp();
                            sum_exp += e;
                            out_row[j2] = e;
                        }
                    }
                } else {
                    for j in 0..dim_size {
                        let e = (x_row[j] - max_val).exp();
                        sum_exp += e;
                        out_row[j] = e;
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for j in 0..dim_size {
                    let e = (x_row[j] - max_val).exp();
                    sum_exp += e;
                    out_row[j] = e;
                }
            }

            let inv_sum = 1.0 / sum_exp;

            // Normalize: multiply stored exp values by inv_sum
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        let inv_vec = _mm256_set1_ps(inv_sum);
                        let mut j = 0;
                        while j + 8 <= dim_size {
                            let exp_vec = _mm256_loadu_ps(out_row.as_ptr().add(j));
                            let result = _mm256_mul_ps(exp_vec, inv_vec);
                            _mm256_storeu_ps(out_row.as_mut_ptr().add(j), result);
                            j += 8;
                        }
                        for j2 in j..dim_size {
                            out_row[j2] *= inv_sum;
                        }
                    }
                } else {
                    for j in 0..dim_size {
                        out_row[j] *= inv_sum;
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for j in 0..dim_size {
                    out_row[j] *= inv_sum;
                }
            }
        }
    }

    output
}

pub unsafe fn log_softmax_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let dim = if args.len() > 1 {
        let dim_i32 = args[1].item() as i32;
        let x_shape = x.shape_ref();
        let ndim = x_shape.len() as i32;
        // Handle negative dimensions
        let dim_normalized = if dim_i32 < 0 { ndim + dim_i32 } else { dim_i32 };
        dim_normalized as usize
    } else {
        0
    };

    let x_shape = x.shape_ref();
    let ndim = x_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let dim_size = x_shape[dim] as usize;

    // Fast path: last-dim, contiguous - fused kernel avoids intermediate tensors
    if dim == ndim - 1 && x.is_contiguous() {
        return vec![log_softmax_last_dim_fused(x, dim_size)];
    }

    let max_vals = max_kernel(&[
        x,
        &Tensor::from_scalar(dim as f32),
        &Tensor::from_scalar(1.0),
    ])[0]
        .clone();
    let x_shifted = x.sub(&max_vals.clone().unsqueeze(dim));
    let log_sum_exp = x_shifted.exp().sum(dim as i32, true).ln();

    vec![x_shifted.sub(&log_sum_exp)]
}

#[inline]
pub fn log_softmax_last_dim_fused(x: &Tensor, dim_size: usize) -> Tensor {
    let x_shape = x.shape_ref();
    let x_ptr = x.data_ptr() as *const f32;
    let numel = x.numel() as usize;

    let mut output = Tensor::empty(x_shape.to_vec(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    #[cfg(feature = "parallel")]
    {
        let a_slice = unsafe { std::slice::from_raw_parts(x_ptr, numel) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

        out_slice
            .par_chunks_mut(dim_size)
            .zip(a_slice.par_chunks(dim_size))
            .for_each(|(out_row, x_row)| {
                // Pass 1: Find max
                let mut max_val = f32::MIN;
                for j in 0..dim_size {
                    max_val = max_val.max(x_row[j]);
                }

                // Pass 2: Compute sum of exp(x - max) and store (x - max) in output
                let mut sum_exp = 0.0f32;
                for j in 0..dim_size {
                    let shifted = x_row[j] - max_val;
                    sum_exp += shifted.exp();
                    out_row[j] = shifted;
                }

                // Pass 3: log_softmax = (x - max) - log(sum_exp)
                let log_sum = sum_exp.ln();
                for j in 0..dim_size {
                    out_row[j] -= log_sum;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let num_rows = numel / dim_size;
        let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, numel) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

        for row in 0..num_rows {
            let row_start = row * dim_size;
            let x_row = &x_slice[row_start..row_start + dim_size];
            let out_row = &mut out_slice[row_start..row_start + dim_size];

            let mut max_val = f32::MIN;
            for j in 0..dim_size {
                max_val = max_val.max(x_row[j]);
            }

            let mut sum_exp = 0.0f32;
            for j in 0..dim_size {
                let shifted = x_row[j] - max_val;
                sum_exp += shifted.exp();
                out_row[j] = shifted;
            }

            let log_sum = sum_exp.ln();
            for j in 0..dim_size {
                out_row[j] -= log_sum;
            }
        }
    }

    output
}

/// Fused SoftmaxBackward kernel: computes grad_input = s * (grad - sum(grad * s))
/// where s is the softmax output from forward pass.
/// This eliminates 3 intermediate tensor allocations.
pub unsafe fn softmax_backward_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let s = args[0]; // softmax output
    let grad = args[1]; // gradient from next layer
    let dim = if args.len() > 2 {
        args[2].item() as usize
    } else {
        0
    };

    let s_shape = s.shape_ref();
    let ndim = s_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };
    let dim_size = s_shape[dim] as usize;

    // For last dimension case with contiguous memory, use optimized version
    if dim == ndim - 1 && s.is_contiguous() && grad.is_contiguous() {
        return vec![softmax_backward_last_dim(s, grad, dim_size)];
    }

    // General case: iterate over all elements
    let _numel = s.numel() as usize;
    let mut output = Tensor::empty(s_shape.to_vec(), s.dtype(), s.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);

    // Use raw pointers for parallel access (raw pointers are Copy + Send)
    let s_ptr = s.data_ptr() as *const f32;
    let grad_ptr = grad.data_ptr() as *const f32;
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    // Compute outer_size (all dimensions before dim) and inner_size (all after dim)
    let outer_size: usize = s_shape[..dim].iter().map(|&d| d as usize).product();
    let inner_size: usize = s_shape[dim + 1..].iter().map(|&d| d as usize).product();
    let stride_dim = dim_size * inner_size;

    // Sequential implementation (raw pointers are not Sync, so we can't parallelize directly)
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let base = outer * stride_dim + inner;

            // Compute dot = sum(grad * s) over the dim dimension
            let mut dot = 0.0f32;
            for i in 0..dim_size {
                let offset = base + i * inner_size;
                unsafe {
                    dot += *grad_ptr.add(offset) * *s_ptr.add(offset);
                }
            }

            // Compute grad_input = s * (grad - dot)
            for i in 0..dim_size {
                let offset = base + i * inner_size;
                unsafe {
                    let s_val = *s_ptr.add(offset);
                    let g_val = *grad_ptr.add(offset);
                    *out_ptr.add(offset) = s_val * (g_val - dot);
                }
            }
        }
    }

    vec![output]
}

/// Optimized softmax backward for last dimension using SIMD
#[inline]
fn softmax_backward_last_dim(s: &Tensor, grad: &Tensor, dim_size: usize) -> Tensor {
    let s_shape = s.shape_ref();
    let outer_size: usize = s_shape[..s_shape.len() - 1]
        .iter()
        .map(|&d| d as usize)
        .product();
    let _numel = s.numel() as usize;

    let mut output = Tensor::empty(s_shape.to_vec(), s.dtype(), s.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);

    // Use raw pointers for parallel access (raw pointers are Copy + Send)
    let s_ptr = s.data_ptr() as *const f32;
    let grad_ptr = grad.data_ptr() as *const f32;
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    // Sequential implementation (raw pointers are not Sync, so we can't parallelize directly)
    for batch in 0..outer_size {
        let base = batch * dim_size;

        // Compute dot = sum(grad * s)
        let mut dot = 0.0f32;
        for i in 0..dim_size {
            let offset = base + i;
            unsafe {
                dot += *grad_ptr.add(offset) * *s_ptr.add(offset);
            }
        }

        // Compute grad_input = s * (grad - dot)
        for i in 0..dim_size {
            let offset = base + i;
            unsafe {
                let s_val = *s_ptr.add(offset);
                let g_val = *grad_ptr.add(offset);
                *out_ptr.add(offset) = s_val * (g_val - dot);
            }
        }
    }

    output
}
