//! Pure f32 elementwise kernel functions (v2.0.0).
//! No Tensor struct, no dispatcher, no autograd — just raw slices.

#![allow(dead_code)]

// ── GELU polynomial constants ──────────────────────────────────────────────
// These are empirically tuned; keep as-is.
const GELU_SQRT_2_OVER_PI: f32 = 0.7978846;
const GELU_COEFF: f32 = 0.044715;

// ── DAZ / FTZ ──────────────────────────────────────────────────────────────
/// Enable DAZ/FTZ for this thread (x86 only).
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[inline(always)]
#[allow(deprecated)]
unsafe fn enable_daz_ftz() {
    use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
    const FTZ: u32 = 1 << 15;
    const DAZ: u32 = 1 << 6;
    let csr = _mm_getcsr();
    _mm_setcsr(csr | FTZ | DAZ);
}

// ── Binary ops ─────────────────────────────────────────────────────────────
pub unsafe fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len().min(a.len()).min(b.len());
    for i in 0..len {
        out[i] = a[i] + b[i];
    }
}

pub unsafe fn sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len().min(a.len()).min(b.len());
    for i in 0..len {
        out[i] = a[i] - b[i];
    }
}

pub unsafe fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len().min(a.len()).min(b.len());
    for i in 0..len {
        out[i] = a[i] * b[i];
    }
}

pub unsafe fn div_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len().min(a.len()).min(b.len());
    for i in 0..len {
        out[i] = a[i] / b[i];
    }
}

// ── Unary ops ──────────────────────────────────────────────────────────────
pub unsafe fn relu_f32(input: &[f32], out: &mut [f32]) {
    let len = out.len().min(input.len());
    for i in 0..len {
        out[i] = if input[i] > 0.0 { input[i] } else { 0.0 };
    }
}

pub unsafe fn gelu_f32(input: &[f32], out: &mut [f32]) {
    let len = out.len().min(input.len());
    for i in 0..len {
        let x = input[i];
        let x3 = x * x * x;
        let t = (GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3)).tanh();
        out[i] = 0.5 * x * (1.0 + t);
    }
}

pub unsafe fn silu_f32(input: &[f32], out: &mut [f32]) {
    let len = out.len().min(input.len());
    for i in 0..len {
        let x = input[i];
        out[i] = x / (1.0 + (-x).exp());
    }
}

pub unsafe fn sigmoid_f32(input: &[f32], out: &mut [f32]) {
    let len = out.len().min(input.len());
    for i in 0..len {
        out[i] = 1.0 / (1.0 + (-input[i]).exp());
    }
}

pub unsafe fn tanh_f32(input: &[f32], out: &mut [f32]) {
    let len = out.len().min(input.len());
    for i in 0..len {
        out[i] = input[i].tanh();
    }
}

pub unsafe fn exp_f32(input: &[f32], out: &mut [f32]) {
    let len = out.len().min(input.len());
    for i in 0..len {
        out[i] = input[i].exp();
    }
}

pub unsafe fn log_f32(input: &[f32], out: &mut [f32]) {
    let len = out.len().min(input.len());
    for i in 0..len {
        out[i] = input[i].ln();
    }
}

pub unsafe fn sqrt_f32(input: &[f32], out: &mut [f32]) {
    let len = out.len().min(input.len());
    for i in 0..len {
        out[i] = input[i].sqrt();
    }
}

pub unsafe fn neg_f32(input: &[f32], out: &mut [f32]) {
    let len = out.len().min(input.len());
    for i in 0..len {
        out[i] = -input[i];
    }
}

pub unsafe fn abs_f32(input: &[f32], out: &mut [f32]) {
    let len = out.len().min(input.len());
    for i in 0..len {
        out[i] = input[i].abs();
    }
}

// ===========================================================================
// v1.x backward-compat Tensor-level kernels (bridge to the f32 slices above)
// These are registered in the runtime dispatch table so that nn/ and tensor/
// code compiles and works during the v2.0.0 migration.
// They will be removed once all code uses the AOT compiler pipeline.
// ===========================================================================

use crate::tensor::Tensor;
use crate::storage::DType;

/// Helper: create a f32 output tensor with the same dtype/device as the input.
fn output_tensor_like(input: &Tensor, shape: &[i64]) -> Tensor {
    Tensor::empty(shape.to_vec(), DType::F32, input.device())
}

// ── Unary elementwise kernels ──────────────────────────────────────────────

macro_rules! unary_kernel {
    ($name:ident, $f32_fn:ident) => {
        pub unsafe fn $name(args: &[&Tensor]) -> Vec<Tensor> {
            let input = args[0];
            let shape = input.shape_ref().to_vec();
            let mut out = output_tensor_like(input, &shape);
            let _n = out.numel() as usize;
            let in_data = input.as_f32_slice();
            let out_data = out.as_f32_slice_mut();
            $f32_fn(in_data, out_data);
            vec![out]
        }
    };
}

unary_kernel!(neg_kernel, neg_f32);
unary_kernel!(abs_kernel, abs_f32);
unary_kernel!(exp_kernel, exp_f32);
unary_kernel!(log_kernel, log_f32);
unary_kernel!(sqrt_kernel, sqrt_f32);
unary_kernel!(relu_kernel, relu_f32);
unary_kernel!(gelu_kernel, gelu_f32);
unary_kernel!(sigmoid_kernel, sigmoid_f32);
unary_kernel!(tanh_kernel, tanh_f32);
unary_kernel!(silu_kernel, silu_f32);

// ── Binary elementwise kernels ─────────────────────────────────────────────

macro_rules! binary_kernel {
    ($name:ident, $f32_fn:ident) => {
        pub unsafe fn $name(args: &[&Tensor]) -> Vec<Tensor> {
            let a = args[0];
            let b = args[1];
            let shape: Vec<i64> = broadcast_shape(a.shape_ref(), b.shape_ref());
            let mut out = output_tensor_like(a, &shape);
            let _n = out.numel() as usize;
            let a_data = a.as_f32_slice();
            let b_data = b.as_f32_slice();
            let out_data = out.as_f32_slice_mut();
            $f32_fn(a_data, b_data, out_data);
            vec![out]
        }
    };
}

/// Simple numpy-style broadcast: take the elementwise-max of two shapes.
fn broadcast_shape(a: &[i64], b: &[i64]) -> Vec<i64> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        result.push(da.max(db));
    }
    result.reverse();
    result
}

binary_kernel!(add_kernel, add_f32);
binary_kernel!(sub_kernel, sub_f32);
binary_kernel!(mul_kernel, mul_f32);
binary_kernel!(div_kernel, div_f32);

// ── Fused add+relu ─────────────────────────────────────────────────────────

pub unsafe fn fused_add_relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];
    let shape: Vec<i64> = broadcast_shape(a.shape_ref(), b.shape_ref());
    let mut out = output_tensor_like(a, &shape);
    let n = out.numel() as usize;
    let a_data = a.as_f32_slice();
    let b_data = b.as_f32_slice();
    let out_data = out.as_f32_slice_mut();
    add_f32(a_data, b_data, out_data);
    // In-place ReLU: read and write the same buffer
    for i in 0..n.min(out_data.len()) {
        if out_data[i] < 0.0 {
            out_data[i] = 0.0;
        }
    }
    vec![out]
}

// ── Fused mul+add ──────────────────────────────────────────────────────────

pub unsafe fn fused_mul_add_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];
    let c = args[2];
    let shape: Vec<i64> = broadcast_shape(a.shape_ref(), b.shape_ref());
    let shape = broadcast_shape(&shape, c.shape_ref());
    let mut out = output_tensor_like(a, &shape);
    let n = out.numel() as usize;
    let a_data = a.as_f32_slice();
    let b_data = b.as_f32_slice();
    let c_data = c.as_f32_slice();
    let out_data = out.as_f32_slice_mut();
    mul_f32(a_data, b_data, out_data);
    for i in 0..n.min(c_data.len()) {
        out_data[i] += c_data[i];
    }
    vec![out]
}

// ── Backward kernels ───────────────────────────────────────────────────────

pub unsafe fn gelu_backward_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let grad = args[0];
    let input = args[1];
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let _n = out.numel() as usize;
    let g = grad.as_f32_slice();
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0.._n {
        let xi = x[i];
        let x3 = xi * xi * xi;
        let tanh_arg = GELU_SQRT_2_OVER_PI * (xi + GELU_COEFF * x3);
        let t = tanh_arg.tanh();
        let sech2 = 1.0 - t * t;
        let dtanh = sech2 * GELU_SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEFF * xi * xi);
        let _gelu = 0.5 * xi * (1.0 + t);
        o[i] = g[i] * (0.5 * (1.0 + t) + xi * dtanh);
    }
    vec![out]
}

pub unsafe fn silu_backward_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let grad = args[0];
    let input = args[1];
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let _n = out.numel() as usize;
    let g = grad.as_f32_slice();
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0.._n {
        let sig = 1.0 / (1.0 + (-x[i]).exp());
        o[i] = g[i] * (sig * (1.0 + x[i] * (1.0 - sig)));
    }
    vec![out]
}

// ── Activation kernels with parameters ─────────────────────────────────────

pub unsafe fn leaky_relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let neg_slope = if args.len() > 1 { args[1].item() } else { 0.01 };
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = if x[i] > 0.0 { x[i] } else { x[i] * neg_slope };
    }
    vec![out]
}

pub unsafe fn prelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let weight = args[1].item();
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = if x[i] > 0.0 { x[i] } else { x[i] * weight };
    }
    vec![out]
}

pub unsafe fn softplus_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let beta = if args.len() > 1 { args[1].item() } else { 1.0 };
    let threshold = if args.len() > 2 { args[2].item() } else { 20.0 };
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        let bx = x[i] * beta;
        o[i] = if bx > threshold { bx } else { (1.0 + bx.exp()).ln() / beta };
    }
    vec![out]
}

pub unsafe fn hardswish_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        let xi = x[i];
        o[i] = if xi <= -3.0 { 0.0 }
               else if xi >= 3.0 { xi }
               else { xi * (xi + 3.0) / 6.0 };
    }
    vec![out]
}

pub unsafe fn elu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let alpha = if args.len() > 1 { args[1].item() } else { 1.0 };
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = if x[i] > 0.0 { x[i] } else { alpha * (x[i].exp() - 1.0) };
    }
    vec![out]
}

pub unsafe fn clamp_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let min_val = if args.len() > 1 { args[1].item() } else { 0.0 };
    let max_val = if args.len() > 2 { args[2].item() } else { 1.0 };
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = x[i].clamp(min_val, max_val);
    }
    vec![out]
}

pub unsafe fn pow_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let exponent = if args.len() > 1 { args[1].item() } else { 1.0 };
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = x[i].powf(exponent);
    }
    vec![out]
}

pub unsafe fn sign_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = if x[i] > 0.0 { 1.0 } else if x[i] < 0.0 { -1.0 } else { 0.0 };
    }
    vec![out]
}

// ── Scalar comparison / logical kernels ────────────────────────────────────

pub unsafe fn gt_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let scalar = if args.len() > 1 { args[1].item() } else { 0.0 };
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = if x[i] > scalar { 1.0 } else { 0.0 };
    }
    vec![out]
}

pub unsafe fn lt_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let scalar = if args.len() > 1 { args[1].item() } else { 0.0 };
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = if x[i] < scalar { 1.0 } else { 0.0 };
    }
    vec![out]
}

pub unsafe fn add_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let scalar = if args.len() > 1 { args[1].item() } else { 0.0 };
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = x[i] + scalar;
    }
    vec![out]
}

pub unsafe fn div_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let scalar = if args.len() > 1 { args[1].item() } else { 1.0 };
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = x[i] / scalar;
    }
    vec![out]
}

pub unsafe fn logical_not_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let input = args[0];
    let shape = input.shape_ref().to_vec();
    let mut out = output_tensor_like(input, &shape);
    let n = out.numel() as usize;
    let x = input.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = if x[i] == 0.0 { 1.0 } else { 0.0 };
    }
    vec![out]
}

// ── Reduction-like binary kernels ──────────────────────────────────────────

pub unsafe fn maximum_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];
    let shape: Vec<i64> = broadcast_shape(a.shape_ref(), b.shape_ref());
    let mut out = output_tensor_like(a, &shape);
    let n = out.numel() as usize;
    let a_data = a.as_f32_slice();
    let b_data = b.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = a_data[i].max(b_data[i]);
    }
    vec![out]
}

pub unsafe fn minimum_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];
    let shape: Vec<i64> = broadcast_shape(a.shape_ref(), b.shape_ref());
    let mut out = output_tensor_like(a, &shape);
    let n = out.numel() as usize;
    let a_data = a.as_f32_slice();
    let b_data = b.as_f32_slice();
    let o = out.as_f32_slice_mut();
    for i in 0..n {
        o[i] = a_data[i].min(b_data[i]);
    }
    vec![out]
}

// ── Embedding kernel ───────────────────────────────────────────────────────

pub unsafe fn embedding_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let weight = args[0];
    let indices = args[1];
    let num_embeddings = weight.shape_ref()[0] as usize;
    let embedding_dim = weight.shape_ref()[1] as usize;
    let n_indices = indices.numel() as usize;
    let w = weight.as_f32_slice();
    let idx = indices.as_f32_slice();

    let mut out_data = vec![0.0f32; n_indices * embedding_dim];
    for i in 0..n_indices {
        let ix = idx[i] as usize;
        if ix < num_embeddings {
            let src = &w[ix * embedding_dim..(ix + 1) * embedding_dim];
            let dst = &mut out_data[i * embedding_dim..(i + 1) * embedding_dim];
            dst.copy_from_slice(src);
        }
    }
    let out = Tensor::from_vec(out_data, vec![n_indices as i64, embedding_dim as i64]);
    vec![out]
}
