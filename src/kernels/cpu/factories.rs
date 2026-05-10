//! CPU factories kernels.

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

pub unsafe fn zeros_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0].shape();
    let dtype = if args.len() > 1 {
        let dtype_slice = args[1].as_f32_slice();
        if !dtype_slice.is_empty() {
            DType::from_str_label(
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

pub unsafe fn ones_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0].shape();
    let dtype = DType::F32;
    let device = Device::Cpu;

    vec![Tensor::ones(shape, dtype, device)]
}

pub unsafe fn full_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0].shape();
    let value = args[1].item();
    let dtype = DType::F32;
    let device = Device::Cpu;

    vec![Tensor::full(shape, value, dtype, device)]
}

pub unsafe fn arange_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let start = args[0].item() as f64;
    let end = args[1].item() as f64;
    let step = if args.len() > 2 {
        args[2].item() as f64
    } else {
        1.0
    };

    let numel = ((end - start) / step).ceil() as usize;
    let values: Vec<f32> = (0..numel)
        .map(|i| (start + i as f64 * step) as f32)
        .collect();

    vec![Tensor::from_vec(values, vec![numel as i64])]
}

pub unsafe fn linspace_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let start = args[0].item();
    let end = args[1].item();
    let steps = args[2].item() as usize;

    if steps == 0 {
        return vec![Tensor::from_vec(vec![], vec![0i64])];
    }
    if steps == 1 {
        return vec![Tensor::from_vec(vec![start], vec![1i64])];
    }

    let values: Vec<f32> = (0..steps)
        .map(|i| {
            let t = i as f32 / (steps - 1) as f32;
            start * (1.0 - t) + end * t
        })
        .collect();

    vec![Tensor::from_vec(values, vec![steps as i64])]
}

pub unsafe fn eye_kernel(args: &[&Tensor]) -> Vec<Tensor> {
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

pub unsafe fn randn_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape_tensor = args[0];
    let shape: Vec<i64> = shape_tensor.inner.sizes.iter().copied().collect();
    let numel: usize = shape.iter().product::<i64>() as usize;

    let mut values = vec![0.0f32; numel];

    // Box-Muller generates 2 normal samples from 2 uniform samples
    // Process in pairs to be 2x faster
    let mut i = 0;
    while i + 1 < numel {
        let u1: f32 = crate::random_f32().max(1e-10); // avoid ln(0)
        let u2: f32 = crate::random_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        values[i] = r * theta.cos();
        values[i + 1] = r * theta.sin();
        i += 2;
    }
    // Handle odd element
    if i < numel {
        let u1: f32 = crate::random_f32().max(1e-10);
        let u2: f32 = crate::random_f32();
        values[i] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }

    vec![Tensor::from_vec(values, shape)]
}

pub unsafe fn rand_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape_tensor = args[0];
    let shape: Vec<i64> = shape_tensor.inner.sizes.iter().copied().collect();
    let numel: i64 = shape.iter().product();

    let values: Vec<f32> = (0..numel as usize).map(|_| crate::random_f32()).collect();

    vec![Tensor::from_vec(values, shape)]
}

#[allow(dead_code)]
pub unsafe fn randint_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape_tensor = args[0];
    let shape: Vec<i64> = shape_tensor.inner.sizes.iter().copied().collect();
    let low = args[1].item() as i32;
    let high = args[2].item() as i32;

    let numel: i64 = shape.iter().product();
    let range = high - low;
    let values: Vec<f32> = (0..numel as usize)
        .map(|_| ((crate::random_f32() * range as f32) as i32 + low) as f32)
        .collect();

    vec![Tensor::from_vec(values, shape)]
}

#[allow(dead_code)]
pub fn read_f32(slice: &[u8], dtype: DType) -> f32 {
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
        DType::BF16 => {
            let ptr = slice.as_ptr() as *const half::bf16;
            unsafe { f32::from(*ptr) }
        }
        DType::F16 => {
            let ptr = slice.as_ptr() as *const half::f16;
            unsafe { f32::from(*ptr) }
        }
        _ => 0.0,
    }
}

#[allow(dead_code)]
pub fn write_f32(slice: &[u8], val: f32, dtype: DType) {
    let ptr = slice.as_ptr() as *mut u8;
    unsafe {
        match dtype {
            DType::F32 => {
                *(ptr as *mut f32) = val;
            }
            DType::F64 => {
                *(ptr as *mut f64) = val as f64;
            }
            DType::I32 => {
                *(ptr as *mut i32) = val as i32;
            }
            DType::I64 => {
                *(ptr as *mut i64) = val as i64;
            }
            DType::BF16 => {
                *(ptr as *mut half::bf16) = half::bf16::from_f32(val);
            }
            DType::F16 => {
                *(ptr as *mut half::f16) = half::f16::from_f32(val);
            }
            _ => {}
        }
    }
}
