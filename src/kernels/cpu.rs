#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::wildcard_in_or_patterns)]
#![allow(unused_imports)]

use crate::dispatcher::{register, DispatchKey, KernelFn};
use crate::iterator::TensorIterator;
use crate::kernels::blas::{matmul_blas, MIN_BLAS_SIZE};
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use std::sync::Arc;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "simd")]
use wide::f32x8;

#[cfg(feature = "simd")]
#[allow(dead_code)]
fn relu_simd(input: &[f32], output: &mut [f32]) {
    let zero = f32x8::ZERO;
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();
    
    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x8::from(*in_chunk);
        let result = v.max(zero);
        *out_chunk = result.into();
    }
    
    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.max(0.0);
    }
}

#[cfg(feature = "simd")]
#[allow(dead_code)]
fn gelu_simd(input: &[f32], output: &mut [f32]) {
    let sqrt_2_over_pi = f32x8::new([
        0.797_884_6, 0.797_884_6, 0.797_884_6, 0.797_884_6,
        0.797_884_6, 0.797_884_6, 0.797_884_6, 0.797_884_6,
    ]);
    let coeff = f32x8::new([
        0.044715f32, 0.044715f32, 0.044715f32, 0.044715f32,
        0.044715f32, 0.044715f32, 0.044715f32, 0.044715f32,
    ]);
    let half = f32x8::new([
        0.5f32, 0.5f32, 0.5f32, 0.5f32,
        0.5f32, 0.5f32, 0.5f32, 0.5f32,
    ]);
    let one = f32x8::new([
        1.0f32, 1.0f32, 1.0f32, 1.0f32,
        1.0f32, 1.0f32, 1.0f32, 1.0f32,
    ]);
    
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();
    
    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let x = f32x8::from(*in_chunk);
        let x3 = x * x * x;
        let y = sqrt_2_over_pi * (x + coeff * x3);
        let exp_2y = (y + y).exp();
        let t = (exp_2y - one) / (exp_2y + one);
        let result = half * x * (one + t);
        *out_chunk = result.into();
    }
    
    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        let x = *in_val;
        let x3 = x * x * x;
        let t = (0.797_884_6 * (x + 0.044715 * x3)).tanh();
        *out_val = 0.5 * x * (1.0 + t);
    }
}

#[cfg(feature = "simd")]
#[allow(dead_code)]
fn tanh_simd(input: &[f32], output: &mut [f32]) {
    let two = f32x8::new([2.0; 8]);
    let one = f32x8::new([1.0; 8]);
    
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();
    
    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x8::from(*in_chunk);
        let abs_v = v.abs();
        let exp_2x = (two * abs_v).exp();
        let result = (exp_2x - one) / (exp_2x + one);
        let sign = v.sign_bit();
        let result = result.blend(-result, sign);
        *out_chunk = result.into();
    }
    
    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        let x = *in_val;
        let abs_x = x.abs();
        let exp_2x = (2.0 * abs_x).exp();
        let mut result = (exp_2x - 1.0) / (exp_2x + 1.0);
        if x < 0.0 {
            result = -result;
        }
        *out_val = result;
    }
}

#[allow(dead_code)]
fn create_output(tensor: &Tensor, shape: Vec<i64>) -> Tensor {
    let sizes: smallvec::SmallVec<[i64; 8]> = shape.into();
    let numel: i64 = sizes.iter().product();
    let nbytes = (numel * tensor.dtype().size() as i64) as usize;
    let storage = Arc::new(Storage::new(tensor.dtype(), tensor.device(), nbytes));
    Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))
}

#[inline]
#[allow(dead_code)]
fn broadcast_shapes_simple(a: &[i64], b: &[i64]) -> Vec<i64> {
    let ndim = std::cmp::max(a.len(), b.len());
    let mut result = vec![1i64; ndim];
    
    let offset_a = ndim - a.len();
    let offset_b = ndim - b.len();
    
    for i in 0..ndim {
        let a_val = if i < offset_a { 1 } else { a[i - offset_a] };
        let b_val = if i < offset_b { 1 } else { b[i - offset_b] };
        result[i] = a_val.max(b_val);
    }
    result
}

fn add_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.inner.sizes.as_slice();
    let b_shape = b.inner.sizes.as_slice();
    let a_contig = a.is_contiguous();
    let b_contig = b.is_contiguous();
    let a_numel = a.inner.numel() as usize;

    if a_contig && b_contig && a_shape == b_shape && a_numel > 4096 {
        let output_shape = a_shape.to_vec();
        let numel = a_numel;
        
        let mut output = Tensor::zeros(output_shape, a.dtype(), a.device());
        
        let a_ptr = a.data_ptr_f32();
        let b_ptr = b.data_ptr_f32();
        
        let output_inner = Arc::make_mut(&mut output.inner);
        let output_storage = Arc::make_mut(&mut output_inner.storage);
        let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

        #[cfg(feature = "parallel")]
        {
            const CHUNK_SIZE: usize = 8192;
            let num_chunks = (numel + CHUNK_SIZE - 1) / CHUNK_SIZE;
            
            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;
            
            if num_chunks > 1 {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    let start = chunk_idx * CHUNK_SIZE;
                    let end = (start + CHUNK_SIZE).min(numel);
                    
                    let a_ptr = a_usize as *const f32;
                    let b_ptr = b_usize as *const f32;
                    let out_ptr = out_usize as *mut f32;
                    
                    for i in start..end {
                        unsafe {
                            *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                        }
                    }
                });
            } else {
                let a_ptr = a_usize as *const f32;
                let b_ptr = b_usize as *const f32;
                let out_ptr = out_usize as *mut f32;
                
                for i in 0..numel {
                    unsafe {
                        *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                    }
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        let mut i = 0usize;
                        while i + 8 <= numel {
                            let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                            let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                            let result = _mm256_add_ps(a_vec, b_vec);
                            _mm256_storeu_ps(out_ptr.add(i), result);
                            i += 8;
                        }
                        while i < numel {
                            *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                            i += 1;
                        }
                    }
                } else {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) + *b_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) + *b_ptr.add(idx);
                    }
                }
            }
        }
        return vec![output];
    }

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();
    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    for idx in 0..numel {
        let mut remaining = idx;
        let mut a_idx = 0;
        let mut b_idx = 0;

        for i in (0..out_shape.len()).rev() {
            let dim_idx = remaining % out_shape[i] as usize;
            remaining /= out_shape[i] as usize;

            if i >= a_shape.len() || a_shape[i] == 1 {
            } else {
                a_idx += dim_idx * a_strides[i] as usize;
            }

            if i >= b_shape.len() || b_shape[i] == 1 {
            } else {
                b_idx += dim_idx * b_strides[i] as usize;
            }
        }

        a_idx += a_storage_offset;
        b_idx += b_storage_offset;

        unsafe {
            let a_val = *a_ptr.add(a_idx);
            let b_val = *b_ptr.add(b_idx);
            *out_ptr.add(idx) = a_val + b_val;
        }
    }

    vec![output]
}

fn sub_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    if a.is_contiguous() && b.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            const CHUNK_SIZE: usize = 8192;
            let num_chunks = (numel + CHUNK_SIZE - 1) / CHUNK_SIZE;
            
            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;
            
            if num_chunks > 1 {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    let start = chunk_idx * CHUNK_SIZE;
                    let end = (start + CHUNK_SIZE).min(numel);
                    
                    let a_ptr = a_usize as *const f32;
                    let b_ptr = b_usize as *const f32;
                    let out_ptr = out_usize as *mut f32;
                    
                    for i in start..end {
                        unsafe {
                            *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i);
                        }
                    }
                });
            } else {
                let a_ptr = a_usize as *const f32;
                let b_ptr = b_usize as *const f32;
                let out_ptr = out_usize as *mut f32;
                
                for i in 0..numel {
                    unsafe {
                        *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i);
                    }
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        let mut i = 0usize;
                        while i + 8 <= numel {
                            let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                            let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                            let result = _mm256_sub_ps(a_vec, b_vec);
                            _mm256_storeu_ps(out_ptr.add(i), result);
                            i += 8;
                        }
                        while i < numel {
                            *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i);
                            i += 1;
                        }
                    }
                } else {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) - *b_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) - *b_ptr.add(idx);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            let mut remaining = idx;
            let mut a_idx = 0;
            let mut b_idx = 0;

            for i in (0..out_shape.len()).rev() {
                let dim_idx = remaining % out_shape[i] as usize;
                remaining /= out_shape[i] as usize;

                if i >= a_shape.len() || a_shape[i] == 1 {
                } else {
                    a_idx += dim_idx * a_strides[i] as usize;
                }

                if i >= b_shape.len() || b_shape[i] == 1 {
                } else {
                    b_idx += dim_idx * b_strides[i] as usize;
                }
            }

            a_idx += a_storage_offset;
            b_idx += b_storage_offset;

            unsafe {
                let a_val = *a_ptr.add(a_idx);
                let b_val = *b_ptr.add(b_idx);
                *out_ptr.add(idx) = a_val - b_val;
            }
        }
    }

    vec![output]
}

fn mul_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.inner.sizes.as_slice();
    let b_shape = b.inner.sizes.as_slice();
    let a_contig = a.is_contiguous();
    let b_contig = b.is_contiguous();
    let a_numel = a.inner.numel() as usize;

    if a_contig && b_contig && a_shape == b_shape && a_numel > 4096 {
        let output_shape = a_shape.to_vec();
        let numel = a_numel;
        
        let mut output = Tensor::zeros(output_shape, a.dtype(), a.device());
        
        let a_ptr = a.data_ptr_f32();
        let b_ptr = b.data_ptr_f32();
        
        let output_inner = Arc::make_mut(&mut output.inner);
        let output_storage = Arc::make_mut(&mut output_inner.storage);
        let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

        #[cfg(feature = "parallel")]
        {
            const CHUNK_SIZE: usize = 8192;
            let num_chunks = (numel + CHUNK_SIZE - 1) / CHUNK_SIZE;
            
            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;
            
            if num_chunks > 1 {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    let start = chunk_idx * CHUNK_SIZE;
                    let end = (start + CHUNK_SIZE).min(numel);
                    
                    let a_ptr = a_usize as *const f32;
                    let b_ptr = b_usize as *const f32;
                    let out_ptr = out_usize as *mut f32;
                    
                    for i in start..end {
                        unsafe {
                            *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                        }
                    }
                });
            } else {
                let a_ptr = a_usize as *const f32;
                let b_ptr = b_usize as *const f32;
                let out_ptr = out_usize as *mut f32;
                
                for i in 0..numel {
                    unsafe {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                    }
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        let mut i = 0usize;
                        while i + 8 <= numel {
                            let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                            let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                            let result = _mm256_mul_ps(a_vec, b_vec);
                            _mm256_storeu_ps(out_ptr.add(i), result);
                            i += 8;
                        }
                        while i < numel {
                            *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                            i += 1;
                        }
                    }
                } else {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) * *b_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) * *b_ptr.add(idx);
                    }
                }
            }
        }
        return vec![output];
    }

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();
    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    for idx in 0..numel {
        let mut remaining = idx;
        let mut a_idx = 0;
        let mut b_idx = 0;

        for i in (0..out_shape.len()).rev() {
            let dim_idx = remaining % out_shape[i] as usize;
            remaining /= out_shape[i] as usize;

            if i >= a_shape.len() || a_shape[i] == 1 {
            } else {
                a_idx += dim_idx * a_strides[i] as usize;
            }

            if i >= b_shape.len() || b_shape[i] == 1 {
            } else {
                b_idx += dim_idx * b_strides[i] as usize;
            }
        }

        a_idx += a_storage_offset;
        b_idx += b_storage_offset;

        unsafe {
            let a_val = *a_ptr.add(a_idx);
            let b_val = *b_ptr.add(b_idx);
            *out_ptr.add(idx) = a_val * b_val;
        }
    }

    vec![output]
}

fn div_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    if a.is_contiguous() && b.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).zip(b_slice.par_iter()).for_each(|((out, &a_val), &b_val)| {
                *out = a_val / b_val;
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        let mut i = 0usize;
                        while i + 8 <= numel {
                            let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                            let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                            let result = _mm256_div_ps(a_vec, b_vec);
                            _mm256_storeu_ps(out_ptr.add(i), result);
                            i += 8;
                        }
                        while i < numel {
                            *out_ptr.add(i) = *a_ptr.add(i) / *b_ptr.add(i);
                            i += 1;
                        }
                    }
                } else {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) / *b_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) / *b_ptr.add(idx);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            let mut remaining = idx;
            let mut a_idx = 0;
            let mut b_idx = 0;

            for i in (0..out_shape.len()).rev() {
                let dim_idx = remaining % out_shape[i] as usize;
                remaining /= out_shape[i] as usize;

                if i >= a_shape.len() || a_shape[i] == 1 {
                } else {
                    a_idx += dim_idx * a_strides[i] as usize;
                }

                if i >= b_shape.len() || b_shape[i] == 1 {
                } else {
                    b_idx += dim_idx * b_strides[i] as usize;
                }
            }

            a_idx += a_storage_offset;
            b_idx += b_storage_offset;

            unsafe {
                let a_val = *a_ptr.add(a_idx);
                let b_val = *b_ptr.add(b_idx);
                *out_ptr.add(idx) = a_val / b_val;
            }
        }
    }

    vec![output]
}

fn neg_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).for_each(|(out, &a_val)| {
                *out = -a_val;
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    *out_ptr.add(idx) = -*a_ptr.add(idx);
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = -val;
            }
        }
    }

    vec![output]
}

fn abs_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).for_each(|(out, &a_val)| {
                *out = a_val.abs();
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        let mut i = 0usize;
                        while i + 8 <= numel {
                            let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                            let abs_vec = _mm256_andnot_ps(_mm256_set1_ps(-0.0f32), a_vec);
                            _mm256_storeu_ps(out_ptr.add(i), abs_vec);
                            i += 8;
                        }
                        while i < numel {
                            *out_ptr.add(i) = a_ptr.add(i).read().abs();
                            i += 1;
                        }
                    }
                } else {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = (*a_ptr.add(idx)).abs();
                        }
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).abs();
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.abs();
            }
        }
    }

    vec![output]
}

fn exp_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).for_each(|(out, &a_val)| {
                *out = a_val.exp();
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    *out_ptr.add(idx) = (*a_ptr.add(idx)).exp();
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.exp();
            }
        }
    }

    vec![output]
}

fn log_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).for_each(|(out, &a_val)| {
                *out = a_val.ln();
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    *out_ptr.add(idx) = (*a_ptr.add(idx)).ln();
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.ln();
            }
        }
    }

    vec![output]
}

fn sqrt_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).for_each(|(out, &a_val)| {
                *out = a_val.sqrt();
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    *out_ptr.add(idx) = (*a_ptr.add(idx)).sqrt();
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.sqrt();
            }
        }
    }

    vec![output]
}

fn relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).for_each(|(out, &a_val)| {
                *out = a_val.max(0.0);
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(feature = "simd")]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                relu_simd(a_slice, out_slice);
            }
            #[cfg(not(feature = "simd"))]
            {
                for idx in 0..numel {
                    unsafe {
                        let val = *a_ptr.add(idx);
                        *out_ptr.add(idx) = val.max(0.0);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.max(0.0);
            }
        }
    }

    vec![output]
}

#[inline]
fn fused_add_relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.shape();
    let _b_shape = b.shape();
    let output_shape = a_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && b.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            const CHUNK_SIZE: usize = 8192;
            let num_chunks = (numel + CHUNK_SIZE - 1) / CHUNK_SIZE;
            
            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;
            
            if num_chunks > 1 {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    let start = chunk_idx * CHUNK_SIZE;
                    let end = (start + CHUNK_SIZE).min(numel);
                    
                    let a_ptr = a_usize as *const f32;
                    let b_ptr = b_usize as *const f32;
                    let out_ptr = out_usize as *mut f32;
                    
                    for i in start..end {
                        let val = unsafe { *a_ptr.add(i) + *b_ptr.add(i) };
                        unsafe { *out_ptr.add(i) = val.max(0.0); }
                    }
                });
            } else {
                let a_ptr = a_usize as *const f32;
                let b_ptr = b_usize as *const f32;
                let out_ptr = out_usize as *mut f32;
                
                for i in 0..numel {
                    let val = unsafe { *a_ptr.add(i) + *b_ptr.add(i) };
                    unsafe { *out_ptr.add(i) = val.max(0.0); }
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    let val = *a_ptr.add(idx) + *b_ptr.add(idx);
                    *out_ptr.add(idx) = val.max(0.0);
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx) + *b_ptr.add(idx);
                *out_ptr.add(idx) = val.max(0.0);
            }
        }
    }

    vec![output]
}

fn gelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).for_each(|(out, &x)| {
                let x3 = x * x * x;
                let t = (0.797_884_6 * (x + 0.044715 * x3)).tanh();
                *out = 0.5 * x * (1.0 + t);
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(feature = "simd")]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                gelu_simd(a_slice, out_slice);
            }
            #[cfg(not(feature = "simd"))]
            {
                for idx in 0..numel {
                    unsafe {
                        let x = *a_ptr.add(idx);
                        let x3 = x * x * x;
                        let t = (0.797_884_6 * (x + 0.044715 * x3)).tanh();
                        *out_ptr.add(idx) = 0.5 * x * (1.0 + t);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                let x3 = x * x * x;
                let t = (0.797_884_6 * (x + 0.044715 * x3)).tanh();
                *out_ptr.add(idx) = 0.5 * x * (1.0 + t);
            }
        }
    }

    vec![output]
}

fn sigmoid_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).for_each(|(out, &x)| {
                *out = 1.0 / (1.0 + (-x).exp());
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    let x = *a_ptr.add(idx);
                    *out_ptr.add(idx) = 1.0 / (1.0 + (-x).exp());
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                *out_ptr.add(idx) = 1.0 / (1.0 + (-x).exp());
            }
        }
    }

    vec![output]
}

fn tanh_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).for_each(|(out, &a_val)| {
                *out = a_val.tanh();
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(feature = "simd")]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                tanh_simd(a_slice, out_slice);
            }
            #[cfg(not(feature = "simd"))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).tanh();
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                *out_ptr.add(idx) = x.tanh();
            }
        }
    }

    vec![output]
}

fn silu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).for_each(|(out, &x)| {
                *out = x / (1.0 + (-x).exp());
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    let x = *a_ptr.add(idx);
                    *out_ptr.add(idx) = x / (1.0 + (-x).exp());
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                *out_ptr.add(idx) = x / (1.0 + (-x).exp());
            }
        }
    }

    vec![output]
}

fn matmul_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() < 2 || b_shape.len() < 2 {
        panic!("matmul: both tensors must have at least 2 dimensions");
    }

    let m = a_shape[a_shape.len() - 2] as i32;
    let k = a_shape[a_shape.len() - 1] as i32;
    let n = b_shape[b_shape.len() - 1] as i32;

    if b_shape[b_shape.len() - 2] as i32 != k {
        panic!("matmul: {} != {}", b_shape[b_shape.len() - 2], k);
    }

    // Use custom tiled matmul
    let batch_a = if a_shape.len() > 2 {
        a_shape[..a_shape.len() - 2].iter().product::<i64>() as usize
    } else {
        1
    };
    let batch_b = if b_shape.len() > 2 {
        b_shape[..b_shape.len() - 2].iter().product::<i64>() as usize
    } else {
        1
    };
    let batch = batch_a.max(batch_b);

    let mut output_shape: Vec<i64> = vec![];
    if a_shape.len() > 2 {
        for i in 0..a_shape.len() - 2 {
            output_shape.push(a_shape[i].max(b_shape[i]));
        }
    }
    output_shape.push(m as i64);
    output_shape.push(n as i64);

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let a_rows = a_shape[a_shape.len() - 2] as usize;
    let a_cols = a_shape[a_shape.len() - 1] as usize;
    let b_cols = b_shape[b_shape.len() - 1] as usize;

    let use_blas = batch == 1
        && m as usize >= MIN_BLAS_SIZE
        && n as usize >= MIN_BLAS_SIZE
        && k as usize >= MIN_BLAS_SIZE
        && a.is_contiguous()
        && b.is_contiguous();

    if use_blas {
        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, a_rows * a_cols) };
        let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, k as usize * b_cols) };
        let result = matmul_blas(a_slice, b_slice, m as usize, k as usize, n as usize);
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, m as usize * n as usize) };
        out_slice.copy_from_slice(&result);
    } else {
        #[cfg(feature = "parallel")]
        {
            if batch > 1 || m as usize * n as usize > 10000 {
                parallel_matmul(
                    a_ptr, b_ptr, out_ptr, batch, m, n, k, a_rows, a_cols, b_cols,
                );
            } else if m as usize <= 64 && n as usize <= 64 && k as usize <= 64 {
                small_matrix_matmul(
                    a_ptr, b_ptr, out_ptr, batch, m, n, k, a_rows, a_cols, b_cols,
                );
            } else {
                single_threaded_matmul(
                    a_ptr, b_ptr, out_ptr, batch, m, n, k, a_rows, a_cols, b_cols,
                );
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            if m as usize <= 64 && n as usize <= 64 && k as usize <= 64 {
                small_matrix_matmul(
                    a_ptr, b_ptr, out_ptr, batch, m, n, k, a_rows, a_cols, b_cols,
                );
            } else {
                single_threaded_matmul(
                    a_ptr, b_ptr, out_ptr, batch, m, n, k, a_rows, a_cols, b_cols,
                );
            }
        }
    }

    vec![output]
}

#[cfg(feature = "parallel")]
#[inline]
fn parallel_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    batch: usize,
    m: i32,
    n: i32,
    k: i32,
    a_rows: usize,
    a_cols: usize,
    b_cols: usize,
) {
    let total_outputs = batch * m as usize * n as usize;
    
    let a_usize = a_ptr as usize;
    let b_usize = b_ptr as usize;
    let out_usize = out_ptr as usize;
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let _batch_usize = batch;

    (0..total_outputs).into_par_iter().for_each(|idx| {
        let bat = idx / (m_usize * n_usize);
        let rem = idx % (m_usize * n_usize);
        let i = rem / n_usize;
        let j = rem % n_usize;

        let mut sum = 0.0f32;
        let mut kk = 0;

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("fma") {
                unsafe {
                    while kk + 8 <= k_usize {
                        let a0 = _mm256_loadu_ps((a_usize + (bat * a_rows * a_cols + i * a_cols + kk) * 4) as *const f32);
                        let a1 = _mm256_loadu_ps((a_usize + (bat * a_rows * a_cols + i * a_cols + kk + 8) * 4) as *const f32);

                        let b0 = _mm256_loadu_ps((b_usize + (bat * k_usize * b_cols + kk * b_cols + j) * 4) as *const f32);
                        let b1 = _mm256_loadu_ps((b_usize + (bat * k_usize * b_cols + (kk + 8) * b_cols + j) * 4) as *const f32);

                        let acc0 = _mm256_mul_ps(a0, b0);
                        let acc1 = _mm256_mul_ps(a1, b1);
                        let acc = _mm256_add_ps(acc0, acc1);
                        
                        let acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                        _mm256_storeu_ps(acc_arr.as_ptr() as *mut f32, acc);
                        sum += acc_arr.assume_init().iter().sum::<f32>();
                        
                        kk += 16;
                    }
                }
            }
        }

        while kk + 4 <= k_usize {
            unsafe {
                let bat_offset = bat * a_rows * a_cols + i * a_cols;
                let a_val = *((a_usize + (bat_offset + kk) * 4) as *const f32);
                let a_val1 = *((a_usize + (bat_offset + kk + 1) * 4) as *const f32);
                let a_val2 = *((a_usize + (bat_offset + kk + 2) * 4) as *const f32);
                let a_val3 = *((a_usize + (bat_offset + kk + 3) * 4) as *const f32);

                let b_base = bat * k_usize * b_cols + kk * b_cols + j;
                let b_val = *((b_usize + b_base * 4) as *const f32);
                let b_val1 = *((b_usize + (b_base + b_cols) * 4) as *const f32);
                let b_val2 = *((b_usize + (b_base + 2 * b_cols) * 4) as *const f32);
                let b_val3 = *((b_usize + (b_base + 3 * b_cols) * 4) as *const f32);

                sum += a_val * b_val + a_val1 * b_val1 + a_val2 * b_val2 + a_val3 * b_val3;
            }
            kk += 4;
        }

        while kk < k_usize {
            unsafe {
                let a_val = *((a_usize + (bat * a_rows * a_cols + i * a_cols + kk) * 4) as *const f32);
                let b_val = *((b_usize + (bat * k_usize * b_cols + kk * b_cols + j) * 4) as *const f32);
                sum += a_val * b_val;
            }
            kk += 1;
        }

        unsafe { *((out_usize + idx * 4) as *mut f32) = sum; };
    });
}

#[inline]
fn small_matrix_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    batch: usize,
    m: i32,
    n: i32,
    k: i32,
    a_rows: usize,
    a_cols: usize,
    b_cols: usize,
) {
    for bat in 0..batch {
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut sum = 0.0f32;

                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("fma") {
                        unsafe {
                            let mut acc0 = _mm256_setzero_ps();
                            let mut acc1 = _mm256_setzero_ps();
                            let mut kk = 0usize;

                            while kk + 8 <= k as usize {
                                let a0 = _mm256_loadu_ps(a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk));
                                let a1 = _mm256_loadu_ps(a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 8));

                                let b0 = _mm256_loadu_ps(b_ptr.add(bat * k as usize * b_cols + kk * b_cols + j));
                                let b1 = _mm256_loadu_ps(b_ptr.add(bat * k as usize * b_cols + (kk + 8) * b_cols + j));

                                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                                acc1 = _mm256_fmadd_ps(a1, b1, acc1);

                                kk += 16;
                            }

                            let acc = _mm256_add_ps(acc0, acc1);
                            let acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                            _mm256_storeu_ps(acc_arr.as_ptr() as *mut f32, acc);
                            sum += acc_arr.assume_init().iter().sum::<f32>();

                            while kk < k as usize {
                                let a_val = *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk);
                                let b_val = *b_ptr.add(bat * k as usize * b_cols + kk * b_cols + j);
                                sum += a_val * b_val;
                                kk += 1;
                            }
                        }
                    } else {
                        let mut sum0 = 0.0f32;
                        let mut sum1 = 0.0f32;
                        let mut sum2 = 0.0f32;
                        let mut sum3 = 0.0f32;

                        let mut kk = 0usize;
                        let k_limit = k as usize;

                        while kk + 4 <= k_limit {
                            let a0 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk) };
                            let a1 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 1) };
                            let a2 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 2) };
                            let a3 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 3) };

                            let b_base = bat * k as usize * b_cols + kk * b_cols + j;
                            let b0 = unsafe { *b_ptr.add(b_base) };
                            let b1 = unsafe { *b_ptr.add(b_base + b_cols) };
                            let b2 = unsafe { *b_ptr.add(b_base + 2 * b_cols) };
                            let b3 = unsafe { *b_ptr.add(b_base + 3 * b_cols) };

                            sum0 += a0 * b0;
                            sum1 += a1 * b1;
                            sum2 += a2 * b2;
                            sum3 += a3 * b3;
                            kk += 4;
                        }

                        sum = sum0 + sum1 + sum2 + sum3;

                        while kk < k_limit {
                            let a_val = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk) };
                            let b_val = unsafe { *b_ptr.add(bat * k as usize * b_cols + kk * b_cols + j) };
                            sum += a_val * b_val;
                            kk += 1;
                        }
                    }
                }

                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    let mut sum0 = 0.0f32;
                    let mut sum1 = 0.0f32;
                    let mut sum2 = 0.0f32;
                    let mut sum3 = 0.0f32;

                    let mut kk = 0usize;
                    let k_limit = k as usize;

                    while kk + 4 <= k_limit {
                        let a0 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk) };
                        let a1 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 1) };
                        let a2 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 2) };
                        let a3 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 3) };

                        let b_base = bat * k as usize * b_cols + kk * b_cols + j;
                        let b0 = unsafe { *b_ptr.add(b_base) };
                        let b1 = unsafe { *b_ptr.add(b_base + b_cols) };
                        let b2 = unsafe { *b_ptr.add(b_base + 2 * b_cols) };
                        let b3 = unsafe { *b_ptr.add(b_base + 3 * b_cols) };

                        sum0 += a0 * b0;
                        sum1 += a1 * b1;
                        sum2 += a2 * b2;
                        sum3 += a3 * b3;
                        kk += 4;
                    }

                    sum = sum0 + sum1 + sum2 + sum3;

                    while kk < k_limit {
                        let a_val = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk) };
                        let b_val = unsafe { *b_ptr.add(bat * k as usize * b_cols + kk * b_cols + j) };
                        sum += a_val * b_val;
                        kk += 1;
                    }
                }

                let out_idx = bat * m as usize * n as usize + i * n as usize + j;
                unsafe { *out_ptr.add(out_idx) = sum };
            }
        }
    }
}

#[inline]
fn single_threaded_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    batch: usize,
    m: i32,
    n: i32,
    k: i32,
    a_rows: usize,
    a_cols: usize,
    b_cols: usize,
) {
    const TILE_SIZE: usize = 64;

    for bat in 0..batch {
        for i_tile in (0..m as usize).step_by(TILE_SIZE) {
            for j_tile in (0..n as usize).step_by(TILE_SIZE) {
                for k_tile in (0..k as usize).step_by(TILE_SIZE) {
                    let i_max = (i_tile + TILE_SIZE).min(m as usize);
                    let j_max = (j_tile + TILE_SIZE).min(n as usize);
                    let k_max = (k_tile + TILE_SIZE).min(k as usize);

                    for i in i_tile..i_max {
                        for j in j_tile..j_max {
                            let mut sum = 0.0f32;

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                            {
                                if is_x86_feature_detected!("fma") {
                                    unsafe {
                                        let mut acc0 = _mm256_setzero_ps();
                                        let mut acc1 = _mm256_setzero_ps();
                                        let mut kk = k_tile;

                                        while kk + 8 <= k_max {
                                            let a0 = _mm256_loadu_ps(a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk));
                                            let a1 = _mm256_loadu_ps(a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 8));

                                            let b0 = _mm256_loadu_ps(b_ptr.add(bat * k as usize * b_cols + kk * b_cols + j));
                                            let b1 = _mm256_loadu_ps(b_ptr.add(bat * k as usize * b_cols + (kk + 8) * b_cols + j));

                                            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                                            acc1 = _mm256_fmadd_ps(a1, b1, acc1);

                                            #[cfg(feature = "prefetch")]
                                            {
                                                _mm_prefetch(b_ptr.add(bat * k as usize * b_cols + (kk + TILE_SIZE) * b_cols + j), _MM_HINT_T0);
                                                _mm_prefetch(a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + TILE_SIZE), _MM_HINT_T0);
                                            }

                                            kk += 16;
                                        }

                                        let acc = _mm256_add_ps(acc0, acc1);
                                        let acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                                        _mm256_storeu_ps(acc_arr.as_ptr() as *mut f32, acc);
                                        sum += acc_arr.assume_init().iter().sum::<f32>();

                                        while kk < k_max {
                                            let a_val = *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk);
                                            let b_val = *b_ptr.add(bat * k as usize * b_cols + kk * b_cols + j);
                                            sum += a_val * b_val;
                                            kk += 1;
                                        }
                                    }
                                } else {
                                    let mut kk = k_tile;
                                    while kk + 4 <= k_max {
                                        let a0 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk) };
                                        let a1 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 1) };
                                        let a2 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 2) };
                                        let a3 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 3) };

                                        let b0 = unsafe { *b_ptr.add(bat * k as usize * b_cols + kk * b_cols + j) };
                                        let b1 = unsafe { *b_ptr.add(bat * k as usize * b_cols + (kk + 1) * b_cols + j) };
                                        let b2 = unsafe { *b_ptr.add(bat * k as usize * b_cols + (kk + 2) * b_cols + j) };
                                        let b3 = unsafe { *b_ptr.add(bat * k as usize * b_cols + (kk + 3) * b_cols + j) };

                                        sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
                                        kk += 4;
                                    }

                                    while kk < k_max {
                                        let a_val = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk) };
                                        let b_val = unsafe { *b_ptr.add(bat * k as usize * b_cols + kk * b_cols + j) };
                                        sum += a_val * b_val;
                                        kk += 1;
                                    }
                                }
                            }

                            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                            {
                                let mut kk = k_tile;
                                while kk + 4 <= k_max {
                                    let a0 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk) };
                                    let a1 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 1) };
                                    let a2 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 2) };
                                    let a3 = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 3) };

                                    let b0 = unsafe { *b_ptr.add(bat * k as usize * b_cols + kk * b_cols + j) };
                                    let b1 = unsafe { *b_ptr.add(bat * k as usize * b_cols + (kk + 1) * b_cols + j) };
                                    let b2 = unsafe { *b_ptr.add(bat * k as usize * b_cols + (kk + 2) * b_cols + j) };
                                    let b3 = unsafe { *b_ptr.add(bat * k as usize * b_cols + (kk + 3) * b_cols + j) };

                                    sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
                                    kk += 4;
                                }

                                while kk < k_max {
                                    let a_val = unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk) };
                                    let b_val = unsafe { *b_ptr.add(bat * k as usize * b_cols + kk * b_cols + j) };
                                    sum += a_val * b_val;
                                    kk += 1;
                                }
                            }

                            let out_idx = bat * m as usize * n as usize + i * n as usize + j;
                            unsafe { *out_ptr.add(out_idx) = sum };
                        }
                    }
                }
            }
        }
    }
}

fn linear_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 { Some(args[2]) } else { None };

    let x_shape = x.shape();
    let w_shape = w.shape();

    let batch_size: i64 = if x_shape.len() > 1 {
        x_shape[..x_shape.len() - 1].iter().product()
    } else {
        1
    };
    let in_features = x_shape[x_shape.len() - 1];
    let out_features = w_shape[0];

    let x_flat = x.reshape(vec![batch_size, in_features]);
    let w_t = w.transpose(0, 1);

    let mut result = (x_flat.matmul(&w_t)).reshape(
        x_shape[..x_shape.len() - 1]
            .iter()
            .copied()
            .chain(std::iter::once(out_features))
            .collect(),
    );

    if let Some(b) = bias {
        result = result.add(b);
    }

    vec![result]
}

fn fused_linear_relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 { Some(args[2]) } else { None };

    let x_shape = x.shape();
    let w_shape = w.shape();

    let batch_size: i64 = if x_shape.len() > 1 {
        x_shape[..x_shape.len() - 1].iter().product()
    } else {
        1
    };
    let in_features = x_shape[x_shape.len() - 1];
    let out_features = w_shape[0];

    let x_ptr = x.data_ptr_f32();
    let w_ptr = w.data_ptr_f32();
    debug_assert!(w.is_contiguous(), "fused_linear_relu: weight tensor must be contiguous");

    let output_shape: Vec<i64> = if x_shape.len() > 1 {
        let mut s = x_shape[..x_shape.len() - 1].to_vec();
        s.push(out_features);
        s
    } else {
        vec![out_features]
    };

    let mut output = Tensor::zeros(output_shape.clone(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let batch_size = batch_size as usize;
    let in_features = in_features as usize;
    let out_features = out_features as usize;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let total = batch_size * out_features;
        
        let x_usize = x_ptr as usize;
        let w_usize = w_ptr as usize;
        let out_usize = out_ptr as usize;
        
        (0..total).into_par_iter().for_each(|idx| {
            let batch_idx = idx / out_features;
            let out_idx = idx % out_features;
            
            let mut sum = 0.0f32;
            for k in 0..in_features {
                let x_offset = batch_idx * in_features + k;
                let w_offset = out_idx * in_features + k;
                let x_val = unsafe { *((x_usize + x_offset * 4) as *const f32) };
                let w_val = unsafe { *((w_usize + w_offset * 4) as *const f32) };
                sum += x_val * w_val;
            }
            
            if let Some(b) = bias {
                let b_ptr = b.data_ptr_f32();
                sum += unsafe { *b_ptr.add(out_idx) };
            }
            
            unsafe { *((out_usize + idx * 4) as *mut f32) = sum.max(0.0); };
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for batch_idx in 0..batch_size {
            for out_idx in 0..out_features {
                let mut sum = 0.0f32;
                for k in 0..in_features {
                    let x_offset = batch_idx * in_features + k;
                    let w_offset = out_idx * in_features + k;
                    let x_val = unsafe { *x_ptr.add(x_offset) };
                    let w_val = unsafe { *w_ptr.add(w_offset) };
                    sum += x_val * w_val;
                }
                
                if let Some(b) = bias {
                    let b_ptr = b.data_ptr_f32();
                    sum += unsafe { *b_ptr.add(out_idx) };
                }
                
                unsafe { *out_ptr.add(batch_idx * out_features + out_idx) = sum.max(0.0); };
            }
        }
    }

    vec![output]
}

fn fused_linear_gelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 { Some(args[2]) } else { None };

    let x_shape = x.shape();
    let w_shape = w.shape();

    let batch_size: i64 = if x_shape.len() > 1 {
        x_shape[..x_shape.len() - 1].iter().product()
    } else {
        1
    };
    let in_features = x_shape[x_shape.len() - 1];
    let out_features = w_shape[0];

    let x_ptr = x.data_ptr_f32();
    let w_ptr = w.data_ptr_f32();
    debug_assert!(w.is_contiguous(), "fused_linear_gelu: weight tensor must be contiguous");

    let output_shape: Vec<i64> = if x_shape.len() > 1 {
        let mut s = x_shape[..x_shape.len() - 1].to_vec();
        s.push(out_features);
        s
    } else {
        vec![out_features]
    };

    let mut output = Tensor::zeros(output_shape.clone(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let batch_size = batch_size as usize;
    let in_features = in_features as usize;
    let out_features = out_features as usize;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let coeff = 0.044715f32;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let total = batch_size * out_features;
        
        let x_usize = x_ptr as usize;
        let w_usize = w_ptr as usize;
        let out_usize = out_ptr as usize;
        
        (0..total).into_par_iter().for_each(|idx| {
            let batch_idx = idx / out_features;
            let out_idx = idx % out_features;
            
            let mut sum = 0.0f32;
            for k in 0..in_features {
                let x_offset = batch_idx * in_features + k;
                let w_offset = out_idx * in_features + k;
                let x_val = unsafe { *((x_usize + x_offset * 4) as *const f32) };
                let w_val = unsafe { *((w_usize + w_offset * 4) as *const f32) };
                sum += x_val * w_val;
            }
            
            if let Some(b) = bias {
                let b_ptr = b.data_ptr_f32();
                sum += unsafe { *b_ptr.add(out_idx) };
            }
            
            let x3 = sum * sum * sum;
            let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
            let gelu = 0.5 * sum * (1.0 + t);
            
            unsafe { *((out_usize + idx * 4) as *mut f32) = gelu; };
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for batch_idx in 0..batch_size {
            for out_idx in 0..out_features {
                let mut sum = 0.0f32;
                for k in 0..in_features {
                    let x_offset = batch_idx * in_features + k;
                    let w_offset = out_idx * in_features + k;
                    let x_val = unsafe { *x_ptr.add(x_offset) };
                    let w_val = unsafe { *w_ptr.add(w_offset) };
                    sum += x_val * w_val;
                }
                
                if let Some(b) = bias {
                    let b_ptr = b.data_ptr_f32();
                    sum += unsafe { *b_ptr.add(out_idx) };
                }
                
                let x3 = sum * sum * sum;
                let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                let gelu = 0.5 * sum * (1.0 + t);
                
                unsafe { *out_ptr.add(batch_idx * out_features + out_idx) = gelu; };
            }
        }
    }

    vec![output]
}

fn sum_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();

    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.clone();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    if output_shape.is_empty() {
        output_shape = vec![1];
    }

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let a_ptr = a.data_ptr() as *const f32;
    let _a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

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

    #[cfg(feature = "parallel")]
    {
        if total_blocks > 256 && dim_size > 32 {
            let a_slice: &[f32] = unsafe { std::slice::from_raw_parts(a_ptr, a_numel) };
            let results: Vec<f32> = (0..total_blocks)
                .into_par_iter()
                .map(|block| {
                    let block_before = block / (strides_after as usize);
                    let block_after = block % (strides_after as usize);

                    let mut sum = 0.0f32;
                    for d in 0..dim_size {
                        let linear_idx = (block_before * dim_size + d) * strides_after as usize + block_after;
                        let idx = a_storage_offset + linear_idx;
                        if idx < a_numel {
                            sum += a_slice[idx];
                        }
                    }
                    sum
                })
                .collect();

            for (i, &sum) in results.iter().enumerate() {
                unsafe {
                    *out_ptr.add(i) = sum;
                }
            }
            return vec![output];
        }
    }

    for block in 0..total_blocks {
        let block_before = block / (strides_after as usize);
        let block_after = block % (strides_after as usize);

        let mut sum = 0.0f32;
        for d in 0..dim_size {
            let linear_idx = (block_before * dim_size + d) * strides_after as usize + block_after;
            let idx = a_storage_offset + linear_idx;
            if idx < a_numel {
                unsafe {
                    sum += *a_ptr.add(idx);
                }
            }
        }

        unsafe {
            *out_ptr.add(block) = sum;
        }
    }

    vec![output]
}

fn mean_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let dim_size = a_shape[dim] as f32;

    let sum_result = sum_kernel(args);
    let mut sum_tensor = sum_result[0].clone();

    if keepdim {
        sum_tensor = sum_tensor.unsqueeze(dim);
    }

    let scale = Tensor::full(vec![], 1.0 / dim_size, DType::F32, Device::Cpu);
    vec![sum_tensor * scale]
}

fn max_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.clone();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    if output_shape.is_empty() {
        output_shape = vec![1];
    }

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let a_ptr = a.data_ptr() as *const f32;
    let a_storage_offset = a.inner.storage_offset as usize;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

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

    #[cfg(feature = "parallel")]
    {
        if total_blocks > 256 && dim_size > 32 {
            let a_slice: &[f32] = unsafe { std::slice::from_raw_parts(a_ptr, a_numel) };
            let results: Vec<f32> = (0..total_blocks)
                .into_par_iter()
                .map(|block| {
                    let block_before = block / (strides_after as usize);
                    let block_after = block % (strides_after as usize);

                    let mut max_val = f32::MIN;
                    for d in 0..dim_size {
                        let linear_idx = (block_before * dim_size + d) * strides_after as usize + block_after;
                        let idx = a_storage_offset + linear_idx;
                        if idx < a_numel {
                            max_val = max_val.max(a_slice[idx]);
                        }
                    }
                    max_val
                })
                .collect();

            for (i, &max_val) in results.iter().enumerate() {
                unsafe {
                    *out_ptr.add(i) = max_val;
                }
            }
            return vec![output];
        }
    }

    for block in 0..total_blocks {
        let block_before = block / (strides_after as usize);
        let block_after = block % (strides_after as usize);

        let mut max_val = f32::MIN;
        for d in 0..dim_size {
            let linear_idx = (block_before * dim_size + d) * strides_after as usize + block_after;
            let idx = a_storage_offset + linear_idx;
            if idx < a_numel {
                unsafe {
                    max_val = max_val.max(*a_ptr.add(idx));
                }
            }
        }

        unsafe {
            *out_ptr.add(block) = max_val;
        }
    }

    vec![output]
}

fn min_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.clone();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    if output_shape.is_empty() {
        output_shape = vec![1];
    }

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let a_ptr = a.data_ptr() as *const f32;
    let a_storage_offset = a.inner.storage_offset as usize;
    let a_numel = a.numel() as usize;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

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
        if total_blocks > 256 && dim_size > 32 {
            let a_slice: &[f32] = unsafe { std::slice::from_raw_parts(a_ptr, a_numel) };
            let results: Vec<f32> = (0..total_blocks)
                .into_par_iter()
                .map(|block| {
                    let block_before = block / (strides_after as usize);
                    let block_after = block % (strides_after as usize);

                    let mut min_val = f32::MAX;
                    for d in 0..dim_size {
                        let linear_idx = (block_before * dim_size + d) * strides_after as usize + block_after;
                        let idx = a_storage_offset + linear_idx;
                        if idx < a_numel {
                            min_val = min_val.min(a_slice[idx]);
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
            let idx = a_storage_offset + linear_idx;
            if idx < a_numel {
                unsafe {
                    min_val = min_val.min(*a_ptr.add(idx));
                }
            }
        }

        unsafe {
            *out_ptr.add(block) = min_val;
        }
    }

    vec![output]
}

fn softmax_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };

    let x_shape = x.shape();
    let ndim = x_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let dim_size = x_shape[dim] as usize;

    if dim == ndim - 1 && x.is_contiguous() && dim_size > 32 {
        return vec![softmax_last_dim_simd(x, dim_size)];
    }

    let max_vals = max_kernel(&[
        x,
        &Tensor::from_scalar(dim as f32),
        &Tensor::from_scalar(1.0),
    ])[0]
        .clone();
    let max_exp = x.sub(&max_vals.clone().unsqueeze(dim)).exp();
    let sum_exp = max_exp.sum(dim as i32, true);

    vec![max_exp.div(&sum_exp)]
}

#[inline]
fn softmax_last_dim_simd(x: &Tensor, dim_size: usize) -> Tensor {
    let x_shape = x.shape();
    let _batch_size: i64 = x_shape[..x_shape.len() - 1].iter().product();
    
    let x_ptr = x.data_ptr() as *const f32;
    let numel = x.numel() as usize;
    let _num_rows = numel / dim_size;
    
    let mut output = Tensor::zeros(x_shape.to_vec(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    #[cfg(feature = "parallel")]
    {
        let a_slice = unsafe { std::slice::from_raw_parts(x_ptr, numel) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
        
        out_slice.par_chunks_mut(dim_size).zip(a_slice.par_chunks(dim_size)).for_each(|(out_row, x_row)| {
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
                        let mut max_arr = [0.0f32; 8];
                        _mm256_storeu_ps(max_arr.as_mut_ptr(), max_vec);
                        for j in 0..8 {
                            max_val = max_val.max(max_arr[j]);
                        }
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
            
            let mut sum_exp = 0.0f32;
            
            for j in 0..dim_size {
                sum_exp += (x_row[j] - max_val).exp();
            }
            
            let inv_sum = 1.0 / sum_exp;
            
            for j in 0..dim_size {
                out_row[j] = (x_row[j] - max_val).exp() * inv_sum;
            }
        });
    }
    
    #[cfg(not(feature = "parallel"))]
    {
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
                sum_exp += (x_row[j] - max_val).exp();
            }
            
            let inv_sum = 1.0 / sum_exp;
            for j in 0..dim_size {
                out_row[j] = (x_row[j] - max_val).exp() * inv_sum;
            }
        }
    }
    
    output
}

fn log_softmax_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };

    let x_shape = x.shape();
    let ndim = x_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

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

fn mse_loss_kernel(args: &[&Tensor]) -> Vec<Tensor> {
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
        "mean" => vec![loss
            .sum(0, false)
            .div(&Tensor::from_scalar(loss.numel() as f32))],
        "sum" => vec![loss.sum(0, false)],
        _ => vec![loss.sum(0, false)],
    }
}

fn cross_entropy_loss_kernel(args: &[&Tensor]) -> Vec<Tensor> {
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

    let mut total_loss = 0.0f32;
    let mut losses = vec![0.0f32; batch_size];

    for b in 0..batch_size {
        let base_idx = b * num_classes;
        
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            use std::arch::x86_64::*;
            let mut max_logit = f32::MIN;
            let mut i = 0;
            
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let ptr = logits_data.as_ptr().add(base_idx);
                    let mut max_vec = _mm256_set1_ps(f32::MIN);
                    for _ in 0..(num_classes / 8) {
                        let vec = _mm256_loadu_ps(ptr.add(i));
                        max_vec = _mm256_max_ps(max_vec, vec);
                        i += 8;
                    }
                    let mut max_arr = [0.0f32; 8];
                    _mm256_storeu_ps(max_arr.as_mut_ptr(), max_vec);
                    for j in 0..8 {
                        if max_arr[j] > max_logit {
                            max_logit = max_arr[j];
                        }
                    }
                }
            }
            for j in i..num_classes {
                let val = logits_data[base_idx + j];
                if val > max_logit {
                    max_logit = val;
                }
            }

            let mut sum_exp = 0.0f32;
            i = 0;
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let ptr = logits_data.as_ptr().add(base_idx);
                    let mut sum_vec = _mm256_setzero_ps();
                    for _ in 0..(num_classes / 8) {
                        let vec = _mm256_loadu_ps(ptr.add(i));
                        let diff = _mm256_sub_ps(vec, _mm256_set1_ps(max_logit));
                        let x = std::slice::from_raw_parts(&diff as *const _ as *const f32, 8);
                        let mut exp_res = [0.0f32; 8];
                        for k in 0..8 {
                            exp_res[k] = x[k].exp();
                        }
                        let exp_vec = _mm256_loadu_ps(exp_res.as_ptr());
                        sum_vec = _mm256_add_ps(sum_vec, exp_vec);
                        i += 8;
                    }
                    let mut sum_arr = [0.0f32; 8];
                    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
                    for j in 0..8 {
                        sum_exp += sum_arr[j];
                    }
                }
            }
            for j in i..num_classes {
                sum_exp += (logits_data[base_idx + j] - max_logit).exp();
            }
            let log_sum_exp = sum_exp.ln();

            let target_class = targets_data[b] as usize;
            let class_logit = logits_data[base_idx + target_class];

            losses[b] = log_sum_exp - class_logit;
            total_loss += losses[b];
        }

        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            let mut max_logit = f32::MIN;
            for c in 0..num_classes {
                let val = logits_data[base_idx + c];
                if val > max_logit {
                    max_logit = val;
                }
            }

            let mut sum_exp = 0.0f32;
            for c in 0..num_classes {
                sum_exp += (logits_data[base_idx + c] - max_logit).exp();
            }
            let log_sum_exp = sum_exp.ln();

            let target_class = targets_data[b] as usize;
            let class_logit = logits_data[base_idx + target_class];

            losses[b] = log_sum_exp - class_logit;
            total_loss += losses[b];
        }
    }

    match reduction {
        "none" => {
            let output = Tensor::from_vec(losses, vec![batch_size as i64]);
            vec![output]
        }
        "mean" => vec![Tensor::from_scalar(total_loss / batch_size as f32)],
        "sum" | _ => vec![Tensor::from_scalar(total_loss)],
    }
}

fn im2col_kernel(
    x: &Tensor,
    kernel_height: usize,
    kernel_width: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_height: usize,
    out_width: usize,
) -> Tensor {
    let x_shape = x.shape();
    let batch_size = x_shape[0] as usize;
    let in_channels = x_shape[1] as usize;
    let in_height = x_shape[2] as usize;
    let in_width = x_shape[3] as usize;

    let col_rows = batch_size * out_height * out_width;
    let col_cols = in_channels * kernel_height * kernel_width;
    let mut col_data = vec![0.0f32; col_rows * col_cols];

    let x_ptr = x.data_ptr() as *const f32;

    let _in_height_pad = in_height + 2 * padding;
    let _in_width_pad = in_width + 2 * padding;

    for n in 0..batch_size {
        for oh in 0..out_height {
            for ow in 0..out_width {
                let col_row = (n * out_height + oh) * out_width + ow;

                for ic in 0..in_channels {
                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            let ih = oh * stride + kh * dilation;
                            let iw = ow * stride + kw * dilation;

                            let col_col = ((ic * kernel_height) + kh) * kernel_width + kw;

                            if ih >= padding
                                && ih < padding + in_height
                                && iw >= padding
                                && iw < padding + in_width
                            {
                                let x_ih = ih - padding;
                                let x_iw = iw - padding;
                                let x_idx =
                                    ((n * in_channels + ic) * in_height + x_ih) * in_width + x_iw;
                                col_data[col_row * col_cols + col_col] =
                                    unsafe { *x_ptr.add(x_idx) };
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::from_vec(col_data, vec![col_rows as i64, col_cols as i64])
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn simd_dot_product(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;

    #[cfg(feature = "simd_avx512")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                let mut acc0 = _mm512_setzero_ps();
                let mut acc1 = _mm512_setzero_ps();
                let mut acc2 = _mm512_setzero_ps();
                let mut acc3 = _mm512_setzero_ps();

                while i + 64 <= len {
                    let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
                    let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
                    let a1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
                    let b1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
                    let a2 = _mm512_loadu_ps(a.as_ptr().add(i + 32));
                    let b2 = _mm512_loadu_ps(b.as_ptr().add(i + 32));
                    let a3 = _mm512_loadu_ps(a.as_ptr().add(i + 48));
                    let b3 = _mm512_loadu_ps(b.as_ptr().add(i + 48));

                    acc0 = _mm512_fmadd_ps(a0, b0, acc0);
                    acc1 = _mm512_fmadd_ps(a1, b1, acc1);
                    acc2 = _mm512_fmadd_ps(a2, b2, acc2);
                    acc3 = _mm512_fmadd_ps(a3, b3, acc3);

                    i += 64;
                }

                let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
                sum += _mm512_reduce_add_ps(acc);

                while i + 16 <= len {
                    let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
                    acc0 = _mm512_fmadd_ps(a_vec, b_vec, _mm512_setzero_ps());
                    sum += _mm512_reduce_add_ps(acc0);
                    i += 16;
                }
            }
        } else if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();

                while i + 32 <= len {
                    let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
                    let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
                    let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
                    let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
                    let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
                    let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
                    let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));

                    acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                    acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                    acc2 = _mm256_fmadd_ps(a2, b2, acc2);
                    acc3 = _mm256_fmadd_ps(a3, b3, acc3);

                    i += 32;
                }

                let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                let acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                _mm256_storeu_ps(acc_arr.as_ptr() as *mut f32, acc);
                sum += acc_arr.assume_init().iter().sum::<f32>();

                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                    let prod = _mm256_mul_ps(a_vec, b_vec);
                    let sum_vec = _mm256_hadd_ps(prod, prod);
                    let sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                    sum += _mm256_cvtss_f32(sum_vec);
                    i += 8;
                }
            }
        }
    }

    #[cfg(not(feature = "simd_avx512"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();

                while i + 32 <= len {
                    let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
                    let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
                    let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
                    let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
                    let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
                    let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
                    let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));

                    acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                    acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                    acc2 = _mm256_fmadd_ps(a2, b2, acc2);
                    acc3 = _mm256_fmadd_ps(a3, b3, acc3);

                    i += 32;
                }

                let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                let acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                _mm256_storeu_ps(acc_arr.as_ptr() as *mut f32, acc);
                sum += acc_arr.assume_init().iter().sum::<f32>();

                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                    let prod = _mm256_mul_ps(a_vec, b_vec);
                    let sum_vec = _mm256_hadd_ps(prod, prod);
                    let sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                    sum += _mm256_cvtss_f32(sum_vec);
                    i += 8;
                }
            }
        }
    }

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
#[inline]
fn simd_dot_product(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + 4 <= len {
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

fn winograd_conv3x3_kernel(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    in_height: usize,
    in_width: usize,
    padding: usize,
) -> Tensor {
    let out_height = in_height + 2 * padding - 2;
    let out_width = in_width + 2 * padding - 2;
    let output_shape = vec![
        batch_size as i64,
        out_channels as i64,
        out_height as i64,
        out_width as i64,
    ];
    let mut output = Tensor::zeros(output_shape.clone(), x.dtype(), x.device());

    let x_ptr = x.data_ptr() as *const f32;
    let w_ptr = w.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let w_data: Vec<f32> =
        unsafe { std::slice::from_raw_parts(w_ptr, out_channels * in_channels * 9).to_vec() };

    let x_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(x_ptr, batch_size * in_channels * in_height * in_width).to_vec()
    };

    let bias_data: Option<Vec<f32>> = bias.map(|b| {
        let ptr = b.data_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, out_channels).to_vec() }
    });

    #[cfg(feature = "parallel")]
    {
        let total = batch_size * out_channels * out_height * out_width;

        let x_data_owned = x_data;
        let w_data_owned = w_data;
        let bias_data_owned = bias_data;

        let results: Vec<f32> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let n = idx / (out_channels * out_height * out_width);
                let rem = idx % (out_channels * out_height * out_width);
                let oc = rem / (out_height * out_width);
                let rem2 = rem % (out_height * out_width);
                let oh = rem2 / out_width;
                let ow = rem2 % out_width;

                let ih_start = oh.saturating_sub(padding);
                let iw_start = ow.saturating_sub(padding);

                let mut sum = 0.0f32;

                for ic in 0..in_channels {
                    let w_offset = oc * in_channels * 9 + ic * 9;
                    let mut patch_values = [0.0f32; 9];
                    let mut patch_idx = 0;

                    for kh in 0..3 {
                        for kw in 0..3 {
                            let ih = ih_start + kh;
                            let iw = iw_start + kw;
                            if ih < in_height && iw < in_width {
                                let x_idx =
                                    ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                                patch_values[patch_idx] = x_data_owned[x_idx];
                            }
                            patch_idx += 1;
                        }
                    }

                    for kw in 0..9 {
                        sum += patch_values[kw] * w_data_owned[w_offset + kw];
                    }
                }

                if let Some(ref b) = bias_data_owned {
                    sum += b[oc];
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
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0f32;

                        let ih_start = oh.saturating_sub(padding);
                        let iw_start = ow.saturating_sub(padding);

                        for ic in 0..in_channels {
                            let w_base = (oc * in_channels + ic) * 9;

                            for kh in 0..3 {
                                for kw in 0..3 {
                                    let ih = ih_start + kh;
                                    let iw = iw_start + kw;

                                    if ih < in_height && iw < in_width {
                                        let x_idx = ((n * in_channels + ic) * in_height + ih)
                                            * in_width
                                            + iw;
                                        let w_idx = w_base + kh * 3 + kw;
                                        sum += x_data[x_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }

                        if let Some(ref b) = bias_data {
                            sum += b[oc];
                        }

                        let out_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                        unsafe { *out_ptr.add(out_idx) = sum };
                    }
                }
            }
        }
    }

    output
}

fn conv2d_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
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

    let x_shape = x.shape();
    let w_shape = w.shape();

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
            x, w, bias, stride, padding, dilation, out_height, out_width,
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
            x,
            w,
            bias,
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
        )];
    }

    if kernel_height == 3 && kernel_width == 3 && stride == 1 && dilation == 1 && groups == 1 {
        return vec![winograd_conv3x3_kernel(
            x,
            w,
            bias,
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
            padding,
        )];
    }

    vec![conv2d_im2col(
        x,
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

fn conv2d_1x1(
    x: &Tensor,
    w: &Tensor,
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
    let mut output = Tensor::zeros(output_shape.clone(), x.dtype(), x.device());

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let w_data: Vec<f32> =
        unsafe { std::slice::from_raw_parts(w_ptr, out_channels * in_channels).to_vec() };

    let _n = batch_size * in_height * in_width;
    let _k = in_channels;
    let _m = out_channels;

    for b in 0..batch_size {
        for h in 0..in_height {
            for w_idx in 0..in_width {
                let row = (b * in_height + h) * in_width + w_idx;
                let x_row = unsafe {
                    std::slice::from_raw_parts(x_ptr.add(row * in_channels), in_channels)
                };

                for oc in 0..out_channels {
                    let w_row = &w_data[oc * in_channels..];
                    let sum = simd_dot_product(x_row, w_row, in_channels);

                    let out_idx = ((b * out_channels + oc) * in_height + h) * in_width + w_idx;
                    unsafe { *out_ptr.add(out_idx) = sum };
                }
            }
        }
    }

    if let Some(b) = bias {
        let b_ptr = b.data_ptr() as *const f32;
        let b_data: Vec<f32> = unsafe { std::slice::from_raw_parts(b_ptr, out_channels).to_vec() };

        for n in 0..batch_size {
            for oc in 0..out_channels {
                for h in 0..in_height {
                    for w_idx in 0..in_width {
                        let out_idx = ((n * out_channels + oc) * in_height + h) * in_width + w_idx;
                        unsafe { *out_ptr.add(out_idx) += b_data[oc] };
                    }
                }
            }
        }
    }

    output
}

fn depthwise_conv2d(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    _padding: usize,
    dilation: usize,
    out_height: usize,
    out_width: usize,
) -> Tensor {
    let x_shape = x.shape();
    let batch_size = x_shape[0] as usize;
    let in_channels = x_shape[1] as usize;
    let in_height = x_shape[2] as usize;
    let in_width = x_shape[3] as usize;

    let w_shape = w.shape();
    let kernel_height = w_shape[2] as usize;
    let kernel_width = w_shape[3] as usize;

    let output_shape = vec![
        batch_size as i64,
        in_channels as i64,
        out_height as i64,
        out_width as i64,
    ];
    let mut output = Tensor::zeros(output_shape.clone(), x.dtype(), x.device());

    let x_ptr = x.data_ptr() as *const f32;
    let w_ptr = w.data_ptr() as *const f32;

    let x_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(x_ptr, batch_size * in_channels * in_height * in_width).to_vec()
    };

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let w_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(w_ptr, in_channels * kernel_height * kernel_width).to_vec()
    };

    let bias_data: Option<Vec<f32>> = bias.map(|b| {
        let ptr = b.data_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, in_channels).to_vec() }
    });

    #[cfg(feature = "parallel")]
    {
        let total = batch_size * in_channels * out_height * out_width;

        let x_data_owned = x_data;
        let w_data_owned = w_data;
        let bias_data_owned = bias_data;

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
                        let ih = oh * stride + kh * dilation;
                        let iw = ow * stride + kw * dilation;

                        if ih < in_height && iw < in_width {
                            let x_idx = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                            let w_idx = ((ic * kernel_height) + kh) * kernel_width + kw;
                            sum += x_data_owned[x_idx] * w_data_owned[w_idx];
                        }
                    }
                }

                if let Some(ref b) = bias_data_owned {
                    sum += b[ic];
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
                                let ih = oh * stride + kh * dilation;
                                let iw = ow * stride + kw * dilation;

                                if ih < in_height && iw < in_width {
                                    let x_idx =
                                        ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                                    let w_idx = ((ic * kernel_height) + kh) * kernel_width + kw;
                                    sum += unsafe { *x_ptr.add(x_idx) } * w_data[w_idx];
                                }
                            }
                        }

                        if let Some(ref b) = bias_data {
                            sum += b[ic];
                        }

                        let out_idx = ((n * in_channels + ic) * out_height + oh) * out_width + ow;
                        unsafe { *out_ptr.add(out_idx) = sum };
                    }
                }
            }
        }
    }

    output
}

fn conv2d_im2col(
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
    let col = im2col_kernel(
        x,
        kernel_height,
        kernel_width,
        stride,
        padding,
        dilation,
        out_height,
        out_width,
    );
    let col_shape = col.shape();
    let col_rows = col_shape[0] as usize;
    let col_cols = col_shape[1] as usize;

    let w_data: Vec<f32> = unsafe {
        let w_ptr = w.data_ptr() as *const f32;
        std::slice::from_raw_parts(
            w_ptr,
            out_channels * in_channels * kernel_height * kernel_width / groups,
        )
        .to_vec()
    };

    let output_shape = vec![
        batch_size as i64,
        out_channels as i64,
        out_height as i64,
        out_width as i64,
    ];
    let mut output = Tensor::zeros(output_shape.clone(), x.dtype(), x.device());

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let col_ptr = col.data_ptr() as *const f32;
    let col_data: Vec<f32> =
        unsafe { std::slice::from_raw_parts(col_ptr, col_rows * col_cols).to_vec() };

    for oc in 0..out_channels {
        let w_row = &w_data[oc * col_cols..(oc + 1) * col_cols];

        for row in 0..col_rows {
            let col_row = &col_data[row * col_cols..(row + 1) * col_cols];
            let sum = simd_dot_product(col_row, w_row, col_cols);

            let n = row / (out_height * out_width);
            let rem = row % (out_height * out_width);
            let oh = rem / out_width;
            let ow = rem % out_width;

            let out_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
            unsafe { *out_ptr.add(out_idx) = sum };
        }
    }

    if let Some(b) = bias {
        let b_ptr = b.data_ptr() as *const f32;
        let b_data: Vec<f32> = unsafe { std::slice::from_raw_parts(b_ptr, out_channels).to_vec() };

        for n in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let out_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                        unsafe { *out_ptr.add(out_idx) += b_data[oc] };
                    }
                }
            }
        }
    }

    output
}

fn layer_norm_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let _normalized_shape = args[1].shape();
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
    let eps = if args.len() > 4 {
        args[4].item() as f64
    } else {
        1e-5
    };

    let x_shape = x.shape();
    let ndim = x_shape.len();
    let _normalized_shape: i64 = x_shape.iter().skip(ndim - 1).product();

    let mean = x.mean((ndim - 1) as i32, true);
    let var = x
        .sub(&mean.clone())
        .mul(&x.sub(&mean.clone()))
        .mean((ndim - 1) as i32, true);
    let std = var.add(&Tensor::from_scalar(eps as f32)).sqrt();

    let mut normalized = x.sub(&mean).div(&std);

    if let Some(w) = weight {
        normalized = normalized.mul(w);
    }
    if let Some(b) = bias {
        normalized = normalized.add(b);
    }

    vec![normalized]
}

fn batch_norm_kernel(args: &[&Tensor]) -> Vec<Tensor> {
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
    let _training = if args.len() > 5 {
        args[5].item() != 0.0
    } else {
        false
    };
    let eps = if args.len() > 6 {
        args[6].item() as f64
    } else {
        1e-5
    };

    let x_shape = x.shape();
    let _num_features = x_shape[1];

    let mean = x.mean(0, false).unsqueeze(0);
    let var = x
        .sub(&mean.clone())
        .mul(&x.sub(&mean.clone()))
        .mean(0, false)
        .unsqueeze(0);
    let std = var.add(&Tensor::from_scalar(eps as f32)).sqrt();

    let mut normalized = x.sub(&mean).div(&std);

    if let Some(w) = weight {
        normalized = normalized.mul(&w.unsqueeze(0));
    }
    if let Some(b) = bias {
        normalized = normalized.add(&b.unsqueeze(0));
    }

    vec![normalized]
}

fn embedding_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let weight = args[0];
    let indices = args[1];

    let weight_shape = weight.shape();
    let num_embeddings = weight_shape[0];
    let embedding_dim = weight_shape[1];

    let indices_shape = indices.shape();
    let batch_size: i64 = indices_shape.iter().product();

    let output_shape: Vec<i64> = indices_shape
        .iter()
        .chain(std::iter::once(&embedding_dim))
        .copied()
        .collect();
    let mut output = Tensor::zeros(output_shape.clone(), weight.dtype(), weight.device());

    let indices_ptr = indices.data_ptr() as *const f32;
    let weight_ptr = weight.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    for i in 0..batch_size as usize {
        let idx = unsafe { *indices_ptr.add(i) } as usize;
        if idx < num_embeddings as usize {
            for j in 0..embedding_dim as usize {
                let w_idx = idx * embedding_dim as usize + j;
                let o_idx = i * embedding_dim as usize + j;
                unsafe {
                    *out_ptr.add(o_idx) = *weight_ptr.add(w_idx);
                }
            }
        }
    }

    vec![output]
}

fn zeros_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0].shape();
    let dtype = if args.len() > 1 {
        let dtype_slice = args[1].as_f32_slice();
        if !dtype_slice.is_empty() {
            DType::from_str(
                &dtype_slice
                    .iter()
                    .map(|&x| x as u8 as char)
                    .collect::<String>(),
            )
            .unwrap_or(DType::F32)
        } else {
            DType::F32
        }
    } else {
        DType::F32
    };
    let device = Device::Cpu;

    vec![Tensor::zeros(shape, dtype, device)]
}

fn ones_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0].shape();
    let dtype = DType::F32;
    let device = Device::Cpu;

    vec![Tensor::ones(shape, dtype, device)]
}

fn full_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0].shape();
    let value = args[1].item();
    let dtype = DType::F32;
    let device = Device::Cpu;

    vec![Tensor::full(shape, value, dtype, device)]
}

fn arange_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let start = args[0].item();
    let end = args[1].item();
    let step = if args.len() > 2 { args[2].item() } else { 1.0 };

    let numel = ((end - start) / step).ceil() as usize;
    let values: Vec<f32> = (0..numel).map(|i| start + i as f32 * step).collect();

    vec![Tensor::from_vec(values, vec![numel as i64])]
}

fn linspace_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let start = args[0].item();
    let end = args[1].item();
    let steps = args[2].item() as usize;

    let values: Vec<f32> = (0..steps)
        .map(|i| {
            let t = i as f32 / (steps - 1) as f32;
            start * (1.0 - t) + end * t
        })
        .collect();

    vec![Tensor::from_vec(values, vec![steps as i64])]
}

fn eye_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let n = args[0].item() as usize;
    let m = if args.len() > 1 {
        args[1].item() as usize
    } else {
        n
    };

    let mut values = vec![0.0f32; n * m];
    for i in 0..n.min(m) {
        values[i * m + i] = 1.0;
    }

    vec![Tensor::from_vec(values, vec![n as i64, m as i64])]
}

fn randn_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape_tensor = args[0];
    let shape: Vec<i64> = shape_tensor.inner.sizes.iter().copied().collect();
    let numel: i64 = shape.iter().product();

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut values = vec![0.0f32; numel as usize];

    for v in &mut values {
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }

    vec![Tensor::from_vec(values, shape)]
}

fn rand_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape_tensor = args[0];
    let shape: Vec<i64> = shape_tensor.inner.sizes.iter().copied().collect();
    let numel: i64 = shape.iter().product();

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let values: Vec<f32> = (0..numel as usize).map(|_| rng.gen()).collect();

    vec![Tensor::from_vec(values, shape)]
}

#[allow(dead_code)]
fn randint_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape_tensor = args[0];
    let shape: Vec<i64> = shape_tensor.inner.sizes.iter().copied().collect();
    let low = args[1].item() as i32;
    let high = args[2].item() as i32;

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let numel: i64 = shape.iter().product();
    let values: Vec<f32> = (0..numel as usize)
        .map(|_| (rng.gen::<i32>() % (high - low) + low) as f32)
        .collect();

    vec![Tensor::from_vec(values, shape)]
}

#[allow(dead_code)]
fn read_f32(slice: &[u8], dtype: DType) -> f32 {
    match dtype {
        DType::F32 => {
            let ptr = slice.as_ptr() as *const f32;
            unsafe { *ptr }
        }
        DType::F64 => {
            let ptr = slice.as_ptr() as *const f64;
            unsafe { *ptr as f32 }
        }
        DType::I32 => {
            let ptr = slice.as_ptr() as *const i32;
            unsafe { *ptr as f32 }
        }
        DType::I64 => {
            let ptr = slice.as_ptr() as *const i64;
            unsafe { *ptr as f32 }
        }
        _ => 0.0,
    }
}

#[allow(dead_code)]
fn write_f32(slice: &[u8], val: f32) {
    let ptr = slice.as_ptr() as *mut u8;
    unsafe {
        *(ptr as *mut f32) = val;
    }
}

fn clamp_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let min_val = args[1].item();
    let max_val = args[2].item();

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).for_each(|(out, &val)| {
                *out = val.clamp(min_val, max_val);
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    let x = *a_ptr.add(idx);
                    *out_ptr.add(idx) = x.clamp(min_val, max_val);
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.clamp(min_val, max_val);
            }
        }
    }

    vec![output]
}

fn pow_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let exponent = args[1].item();

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 4096 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice.par_iter_mut().zip(a_slice.par_iter()).for_each(|(out, &val)| {
                *out = val.powf(exponent);
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    let x = *a_ptr.add(idx);
                    *out_ptr.add(idx) = x.powf(exponent);
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.powf(exponent);
            }
        }
    }

    vec![output]
}

#[ctor::ctor]
fn register_kernels() {
    register("add", DispatchKey::Cpu, add_kernel as KernelFn);
    register("sub", DispatchKey::Cpu, sub_kernel as KernelFn);
    register("mul", DispatchKey::Cpu, mul_kernel as KernelFn);
    register("div", DispatchKey::Cpu, div_kernel as KernelFn);
    register("neg", DispatchKey::Cpu, neg_kernel as KernelFn);
    register("abs", DispatchKey::Cpu, abs_kernel as KernelFn);
    register("exp", DispatchKey::Cpu, exp_kernel as KernelFn);
    register("log", DispatchKey::Cpu, log_kernel as KernelFn);
    register("sqrt", DispatchKey::Cpu, sqrt_kernel as KernelFn);
    register("relu", DispatchKey::Cpu, relu_kernel as KernelFn);
    register("fused_add_relu", DispatchKey::Cpu, fused_add_relu_kernel as KernelFn);
    register("gelu", DispatchKey::Cpu, gelu_kernel as KernelFn);
    register("sigmoid", DispatchKey::Cpu, sigmoid_kernel as KernelFn);
    register("tanh", DispatchKey::Cpu, tanh_kernel as KernelFn);
    register("silu", DispatchKey::Cpu, silu_kernel as KernelFn);
    register("clamp", DispatchKey::Cpu, clamp_kernel as KernelFn);
    register("pow", DispatchKey::Cpu, pow_kernel as KernelFn);
    register("matmul", DispatchKey::Cpu, matmul_kernel as KernelFn);
    register("linear", DispatchKey::Cpu, linear_kernel as KernelFn);
    register("fused_linear_relu", DispatchKey::Cpu, fused_linear_relu_kernel as KernelFn);
    register("fused_linear_gelu", DispatchKey::Cpu, fused_linear_gelu_kernel as KernelFn);
    register("sum", DispatchKey::Cpu, sum_kernel as KernelFn);
    register("mean", DispatchKey::Cpu, mean_kernel as KernelFn);
    register("max", DispatchKey::Cpu, max_kernel as KernelFn);
    register("min", DispatchKey::Cpu, min_kernel as KernelFn);
    register("softmax", DispatchKey::Cpu, softmax_kernel as KernelFn);
    register(
        "log_softmax",
        DispatchKey::Cpu,
        log_softmax_kernel as KernelFn,
    );
    register("mse_loss", DispatchKey::Cpu, mse_loss_kernel as KernelFn);
    register(
        "cross_entropy_loss",
        DispatchKey::Cpu,
        cross_entropy_loss_kernel as KernelFn,
    );
    register("conv2d", DispatchKey::Cpu, conv2d_kernel as KernelFn);
    register(
        "layer_norm",
        DispatchKey::Cpu,
        layer_norm_kernel as KernelFn,
    );
    register(
        "batch_norm",
        DispatchKey::Cpu,
        batch_norm_kernel as KernelFn,
    );
    register("embedding", DispatchKey::Cpu, embedding_kernel as KernelFn);
    register("zeros", DispatchKey::Cpu, zeros_kernel as KernelFn);
    register("ones", DispatchKey::Cpu, ones_kernel as KernelFn);
    register("full", DispatchKey::Cpu, full_kernel as KernelFn);
    register("arange", DispatchKey::Cpu, arange_kernel as KernelFn);
    register("linspace", DispatchKey::Cpu, linspace_kernel as KernelFn);
    register("eye", DispatchKey::Cpu, eye_kernel as KernelFn);
    register("randn", DispatchKey::Cpu, randn_kernel as KernelFn);
    register("rand", DispatchKey::Cpu, rand_kernel as KernelFn);
}
