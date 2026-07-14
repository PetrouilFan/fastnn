//! SIMD-aware dispatch wrappers for elementwise ops, reductions,
//! scalar arithmetic, normalisation, softmax, and optimiser updates.
//!
//! Each wrapper checks `simd_avx2_available()` at runtime and delegates
//! to the AVX2 microkernel when possible, falling back to a scalar loop.
//! These were extracted from `mod.rs` to reduce the size of the main
//! CPU backend dispatch file.

#![allow(dead_code)]

use super::microkernels;

macro_rules! impl_simd_unary_wrapper {
    ($name:ident, $avx2:path, $scalar:path) => {
        #[inline]
        pub(super) fn $name(input: &[f32], output: &mut [f32]) {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if microkernels::simd_avx2_available() {
                return unsafe { $avx2(input, output) };
            }
            let len = output.len().min(input.len());
            #[cfg(feature = "parallel")]
            if len >= 4096 {
                use rayon::prelude::*;
                output[..len].par_iter_mut().enumerate().for_each(|(i, o)| {
                    *o = $scalar(input[i]);
                });
            } else {
                for i in 0..len {
                    output[i] = $scalar(input[i]);
                }
            }
            #[cfg(not(feature = "parallel"))]
            for i in 0..len {
                output[i] = $scalar(input[i]);
            }
        }
    };
}

macro_rules! impl_simd_binary_wrapper {
    ($name:ident, $avx2:path, $scalar:path, $op:expr) => {
        #[inline]
        pub(super) fn $name(a: &[f32], b: &[f32], output: &mut [f32]) {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if (a.len() == output.len() || b.len() == output.len())
                && microkernels::simd_avx2_available()
            {
                return unsafe { $avx2(a, b, output) };
            }
            #[cfg(feature = "parallel")]
            let len = output.len().min(a.len().max(b.len()));
            #[cfg(feature = "parallel")]
            if len >= 4096 {
                use rayon::prelude::*;
                let a_len = a.len();
                let b_len = b.len();
                output[..len].par_iter_mut().enumerate().for_each(|(i, o)| {
                    *o = $op(a[i % a_len], b[i % b_len]);
                });
            } else {
                let a_len = a.len();
                let b_len = b.len();
                for i in 0..len {
                    output[i] = $op(a[i % a_len], b[i % b_len]);
                }
            }
            #[cfg(not(feature = "parallel"))]
            $scalar(a, b, output);
        }
    };
}

macro_rules! impl_simd_scalar_wrapper {
    ($name:ident, $avx2:path, $scalar:path, $op:expr) => {
        #[inline]
        pub(super) fn $name(data: &[f32], s: f32, output: &mut [f32]) {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if microkernels::simd_avx2_available() {
                return unsafe { $avx2(data, s, output) };
            }
            #[cfg(feature = "parallel")]
            let len = output.len().min(data.len());
            #[cfg(feature = "parallel")]
            if len >= 4096 {
                use rayon::prelude::*;
                output[..len].par_iter_mut().enumerate().for_each(|(i, o)| {
                    *o = $op(data[i], s);
                });
            } else {
                for i in 0..len {
                    output[i] = $op(data[i], s);
                }
            }
            #[cfg(not(feature = "parallel"))]
            $scalar(data, s, output);
        }
    };
}

// ============================================================
// SIMD-aware elementwise dispatch wrappers
// ============================================================

impl_simd_unary_wrapper!(
    relu_f32,
    microkernels::relu_f32_avx2,
    microkernels::relu_f32_scalar
);
impl_simd_unary_wrapper!(
    gelu_f32,
    microkernels::gelu_f32_avx2,
    microkernels::gelu_f32_scalar
);
impl_simd_unary_wrapper!(
    silu_f32,
    microkernels::silu_f32_avx2,
    microkernels::silu_f32_scalar
);
impl_simd_unary_wrapper!(
    sigmoid_f32,
    microkernels::sigmoid_f32_avx2,
    microkernels::sigmoid_f32_scalar
);
impl_simd_unary_wrapper!(
    tanh_f32,
    microkernels::tanh_f32_avx2,
    microkernels::tanh_f32_scalar
);
impl_simd_unary_wrapper!(
    exp_f32,
    microkernels::exp_f32_avx2,
    microkernels::exp_f32_scalar
);
impl_simd_unary_wrapper!(
    log_f32,
    microkernels::log_f32_avx2,
    microkernels::log_f32_scalar
);
impl_simd_unary_wrapper!(
    sqrt_f32,
    microkernels::sqrt_f32_avx2,
    microkernels::sqrt_f32_scalar
);
impl_simd_unary_wrapper!(
    neg_f32,
    microkernels::neg_f32_avx2,
    microkernels::neg_f32_scalar
);
impl_simd_unary_wrapper!(
    abs_f32,
    microkernels::abs_f32_avx2,
    microkernels::abs_f32_scalar
);
impl_simd_unary_wrapper!(
    elu_f32,
    microkernels::elu_f32_avx2,
    microkernels::elu_f32_scalar
);
impl_simd_unary_wrapper!(
    softplus_f32,
    microkernels::softplus_f32_avx2,
    microkernels::softplus_f32_scalar
);
impl_simd_unary_wrapper!(
    hardswish_f32,
    microkernels::hardswish_f32_avx2,
    microkernels::hardswish_f32_scalar
);
impl_simd_unary_wrapper!(
    sign_f32,
    microkernels::sign_f32_avx2,
    microkernels::sign_f32_scalar
);
impl_simd_unary_wrapper!(
    round_f32,
    microkernels::round_f32_avx2,
    microkernels::round_f32_scalar
);
impl_simd_unary_wrapper!(
    logical_not_f32,
    microkernels::logical_not_f32_avx2,
    microkernels::logical_not_f32_scalar
);
impl_simd_unary_wrapper!(
    mish_f32,
    microkernels::mish_f32_avx2,
    microkernels::mish_f32_scalar
);

#[inline]
pub(super) fn leaky_relu_f32(input: &[f32], output: &mut [f32], slope: f32) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe { microkernels::leaky_relu_f32_avx2(input, output, slope) };
    }
    let len = output.len().min(input.len());
    for i in 0..len {
        output[i] = microkernels::leaky_relu_f32_scalar(input[i], slope);
    }
}

#[inline]
pub(super) fn clamp_f32(input: &[f32], output: &mut [f32], min_val: f32, max_val: f32) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe { microkernels::clamp_f32_avx2(input, output, min_val, max_val) };
    }
    let len = output.len().min(input.len());
    for i in 0..len {
        output[i] = microkernels::clamp_f32_scalar(input[i], min_val, max_val);
    }
}

#[inline]
pub(super) fn log_softmax_f32(input: &[f32], output: &mut [f32]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe { microkernels::log_softmax_f32_avx2(input, output) };
    }
    microkernels::log_softmax_f32_scalar_all(input, output);
}

// ── Binary ops ──────────────────────────────────────────────

impl_simd_binary_wrapper!(
    add_f32,
    microkernels::add_f32_avx2_broadcast,
    microkernels::add_f32_scalar_broadcast,
    |a, b| a + b
);
impl_simd_binary_wrapper!(
    sub_f32,
    microkernels::sub_f32_avx2_broadcast,
    microkernels::sub_f32_scalar_broadcast,
    |a, b| a - b
);
impl_simd_binary_wrapper!(
    mul_f32,
    microkernels::mul_f32_avx2_broadcast,
    microkernels::mul_f32_scalar_broadcast,
    |a, b| a * b
);
impl_simd_binary_wrapper!(
    div_f32,
    microkernels::div_f32_avx2_broadcast,
    microkernels::div_f32_scalar_broadcast,
    |a, b| a / b
);
impl_simd_binary_wrapper!(
    max_f32,
    microkernels::max_f32_avx2_broadcast,
    microkernels::max_f32_scalar_broadcast,
    |a: f32, b: f32| a.max(b)
);
impl_simd_binary_wrapper!(
    min_f32,
    microkernels::min_f32_avx2_broadcast,
    microkernels::min_f32_scalar_broadcast,
    |a: f32, b: f32| a.min(b)
);

// ============================================================
// Reductions
// ============================================================

#[inline]
pub(super) fn reduce_f32(
    input: &[f32],
    output: &mut [f32],
    group_size: usize,
    is_mean: bool,
    is_max: bool,
) {
    if is_max {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if microkernels::simd_avx2_available() {
            return unsafe { microkernels::reduce_max_f32_avx2(input, output, group_size) };
        }
        microkernels::reduce_max_f32_scalar(input, output, group_size);
    } else {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if microkernels::simd_avx2_available() {
            return unsafe {
                microkernels::reduce_sum_f32_avx2(input, output, group_size, is_mean)
            };
        }
        microkernels::reduce_sum_f32_scalar(input, output, group_size, is_mean);
    }
}

// ============================================================
// Scalar arithmetic
// ============================================================

impl_simd_scalar_wrapper!(
    add_scalar_f32,
    microkernels::add_scalar_f32_avx2,
    microkernels::add_scalar_f32_scalar,
    |a, s| a + s
);
impl_simd_scalar_wrapper!(
    mul_scalar_f32,
    microkernels::mul_scalar_f32_avx2,
    microkernels::mul_scalar_f32_scalar,
    |a, s| a * s
);
impl_simd_scalar_wrapper!(
    div_scalar_f32,
    microkernels::div_scalar_f32_avx2,
    microkernels::div_scalar_f32_scalar,
    |a, s| a / s
);

// ============================================================
// Scalar comparison
// ============================================================

impl_simd_scalar_wrapper!(
    gt_scalar_f32,
    microkernels::gt_scalar_f32_avx2,
    microkernels::gt_scalar_f32_scalar,
    |a, s| if a > s { 1.0 } else { 0.0 }
);
impl_simd_scalar_wrapper!(
    lt_scalar_f32,
    microkernels::lt_scalar_f32_avx2,
    microkernels::lt_scalar_f32_scalar,
    |a, s| if a < s { 1.0 } else { 0.0 }
);
impl_simd_scalar_wrapper!(
    eq_scalar_f32,
    microkernels::eq_scalar_f32_avx2,
    microkernels::eq_scalar_f32_scalar,
    |a, s| if a == s { 1.0 } else { 0.0 }
);

// ============================================================
// BiasAdd, Norm, RMS Norm, Softmax, Argmax
// ============================================================

#[inline]
pub(super) fn biasadd_f32(data: &[f32], bias: &[f32], output: &mut [f32], channel_stride: usize) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe { microkernels::biasadd_f32_avx2(data, bias, output, channel_stride) };
    }
    microkernels::biasadd_f32_scalar(data, bias, output, channel_stride);
}

#[inline]
pub(super) fn norm_layernorm_f32(input: &[f32], output: &mut [f32], row_size: usize, eps: f32) {
    #[cfg(feature = "parallel")]
    let num_rows = input.len() / row_size;

    #[cfg(feature = "parallel")]
    if num_rows > 1 {
        use rayon::prelude::*;
        #[allow(unused_variables)]
        let has_avx2 = {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                microkernels::simd_avx2_available()
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                false
            }
        };
        let input_ref: &[f32] = input;
        output
            .par_chunks_mut(row_size)
            .enumerate()
            .for_each(|(r, out)| {
                let start = r * row_size;
                let inp = &input_ref[start..start + row_size];
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                if has_avx2 {
                    unsafe { microkernels::norm_layernorm_f32_avx2(inp, out, row_size, eps) };
                    return;
                }
                microkernels::norm_layernorm_f32_scalar(inp, out, row_size, eps);
            });
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe { microkernels::norm_layernorm_f32_avx2(input, output, row_size, eps) };
    }
    microkernels::norm_layernorm_f32_scalar(input, output, row_size, eps);
}

#[inline]
pub(super) fn rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    row_size: usize,
    eps: f32,
) {
    #[cfg(feature = "parallel")]
    let num_rows = input.len() / row_size;

    #[cfg(feature = "parallel")]
    if num_rows > 1 {
        use rayon::prelude::*;
        #[allow(unused_variables)]
        let has_avx2 = {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                microkernels::simd_avx2_available()
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                false
            }
        };
        let input_ref: &[f32] = input;
        let weight_ref: &[f32] = weight;
        output
            .par_chunks_mut(row_size)
            .enumerate()
            .for_each(|(r, out)| {
                let start = r * row_size;
                let inp = &input_ref[start..start + row_size];
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                if has_avx2 {
                    unsafe { microkernels::rms_norm_f32_avx2(inp, weight_ref, out, row_size, eps) };
                    return;
                }
                microkernels::rms_norm_f32_scalar(inp, weight_ref, out, row_size, eps);
            });
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe { microkernels::rms_norm_f32_avx2(input, weight, output, row_size, eps) };
    }
    microkernels::rms_norm_f32_scalar(input, weight, output, row_size, eps);
}

#[inline]
pub(super) fn softmax_f32(
    input: &[f32],
    output: &mut [f32],
    axis_dim_size: usize,
    stride: usize,
    num_rows: usize,
) {
    #[cfg(feature = "parallel")]
    if num_rows > 1 && stride == 1 {
        use rayon::prelude::*;
        #[allow(unused_variables)]
        let has_avx2 = {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                microkernels::simd_avx2_available()
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                false
            }
        };
        let input_ref: &[f32] = input;
        output
            .par_chunks_mut(axis_dim_size)
            .enumerate()
            .for_each(|(row, out)| {
                let offset = row * axis_dim_size;
                let inp = &input_ref[offset..offset + axis_dim_size];
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                if has_avx2 {
                    unsafe {
                        microkernels::softmax_f32_avx2_strided(inp, out, axis_dim_size, 1, 1)
                    };
                    return;
                }
                microkernels::softmax_f32_scalar_strided(inp, out, axis_dim_size, 1, 1);
            });
        return;
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe {
            microkernels::softmax_f32_avx2_strided(input, output, axis_dim_size, stride, num_rows)
        };
    }
    microkernels::softmax_f32_scalar_strided(input, output, axis_dim_size, stride, num_rows);
}

pub(super) fn argmax_f32(
    input: &[f32],
    output: &mut [u64],
    axis: usize,
    dim_size: usize,
    inner: usize,
) {
    if axis == usize::MAX || dim_size == 0 || dim_size > input.len() {
        let max_idx = input
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u64)
            .unwrap_or(0);
        for v in output.iter_mut() {
            *v = max_idx;
        }
    } else {
        let outer = input.len() / (dim_size * inner);
        for o in 0..outer {
            for i in 0..inner {
                let base = o * dim_size * inner + i;
                let mut best_flat = base as u64;
                let mut best_val = input[base];
                for k in 1..dim_size {
                    let flat_idx = base + k * inner;
                    let val = input[flat_idx];
                    if val > best_val {
                        best_val = val;
                        best_flat = flat_idx as u64;
                    }
                }
                let out_idx = o * inner + i;
                if out_idx < output.len() {
                    output[out_idx] = best_flat;
                }
            }
        }
    }
}

// ============================================================
// Optimizer update wrappers
// ============================================================

#[inline]
pub(super) fn sgd_update_f32(w: &[f32], g: &[f32], lr: f32, wd: f32) -> Vec<f32> {
    let mut out = vec![0.0; w.len()];
    sgd_update_f32_into(w, g, lr, wd, &mut out);
    out
}

#[inline]
pub(super) fn sgd_update_f32_into(w: &[f32], g: &[f32], lr: f32, wd: f32, out: &mut [f32]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        unsafe { microkernels::sgd_update_f32_avx2_into(w, g, lr, wd, out) };
        return;
    }
    microkernels::sgd_update_f32_scalar_into(w, g, lr, wd, out);
}

#[inline]
pub(super) fn adam_update_f32(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    let mut v_new = v.to_vec();
    adam_update_f32_into(
        w, g, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2, &mut w_new, &mut m_new,
        &mut v_new,
    );
    (w_new, m_new, v_new)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn adam_update_f32_into(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
    w_out: &mut [f32],
    m_out: &mut [f32],
    v_out: &mut [f32],
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        unsafe {
            microkernels::adam_update_f32_avx2_into(
                w, g, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2, w_out, m_out, v_out,
            );
        }
        return;
    }
    microkernels::adam_update_f32_scalar_into(
        w, g, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2, w_out, m_out, v_out,
    );
}

#[inline]
pub(super) fn adamw_update_f32(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
    wd: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut w_out = vec![0.0; w.len()];
    let mut m_out = vec![0.0; m.len()];
    let mut v_out = vec![0.0; v.len()];
    adamw_update_f32_into(
        w, g, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2, wd, &mut w_out, &mut m_out,
        &mut v_out,
    );
    (w_out, m_out, v_out)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn adamw_update_f32_into(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
    wd: f32,
    w_out: &mut [f32],
    m_out: &mut [f32],
    v_out: &mut [f32],
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        unsafe {
            microkernels::adamw_update_f32_avx2_into(
                w, g, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2, wd, w_out, m_out, v_out,
            )
        };
        return;
    }
    microkernels::adamw_update_f32_scalar_into(
        w, g, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2, wd, w_out, m_out, v_out,
    );
}

#[inline]
pub(super) fn lion_update_f32(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    wd: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut w_out = vec![0.0; w.len()];
    let mut m_out = vec![0.0; m.len()];
    lion_update_f32_into(w, g, m, lr, beta1, beta2, wd, &mut w_out, &mut m_out);
    (w_out, m_out)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn lion_update_f32_into(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    wd: f32,
    w_out: &mut [f32],
    m_out: &mut [f32],
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        unsafe {
            microkernels::lion_update_f32_avx2_into(w, g, m, lr, beta1, beta2, wd, w_out, m_out)
        };
        return;
    }
    microkernels::lion_update_f32_scalar_into(w, g, m, lr, beta1, beta2, wd, w_out, m_out);
}

#[inline]
pub(super) fn rmsprop_update_f32(
    w: &[f32],
    g: &[f32],
    v: &[f32],
    lr: f32,
    beta: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut w_out = vec![0.0; w.len()];
    let mut v_out = vec![0.0; v.len()];
    rmsprop_update_f32_into(w, g, v, lr, beta, eps, &mut w_out, &mut v_out);
    (w_out, v_out)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn rmsprop_update_f32_into(
    w: &[f32],
    g: &[f32],
    v: &[f32],
    lr: f32,
    beta: f32,
    eps: f32,
    w_out: &mut [f32],
    v_out: &mut [f32],
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        unsafe { microkernels::rmsprop_update_f32_avx2_into(w, g, v, lr, beta, eps, w_out, v_out) };
        return;
    }
    microkernels::rmsprop_update_f32_scalar_into(w, g, v, lr, beta, eps, w_out, v_out);
}

#[inline]
pub(super) fn muon_update_f32(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    lr: f32,
    beta: f32,
    wd: f32,
) -> (Vec<f32>, Vec<f32>) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe { microkernels::muon_update_f32_avx2(w, g, m, lr, beta, wd) };
    }
    microkernels::muon_update_f32_scalar(w, g, m, lr, beta, wd)
}
