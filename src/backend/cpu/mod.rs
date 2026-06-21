#![allow(dead_code)]
#![allow(clippy::shadow_unrelated)]
#![allow(clippy::let_and_return)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_checked_ops)]
#![allow(clippy::redundant_locals)]
#![allow(clippy::get_first)]
#![allow(clippy::if_same_then_else)]

use crate::backend::cpu::blas::matmul_blas_into;
use crate::backend::{Backend, BackendError, BufferSlice, ExecutablePlan, Instruction};
use crate::compiler::passes::memory_planning::MemoryPlan;
use crate::dtypes::{PackedWord, U4x8, U8x4};
use crate::ir::node::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, ShapeEnv, TensorValue};
use crate::packed_tensor::PackedTensor;
use bytemuck;
use std::cell::UnsafeCell;
use std::sync::atomic::Ordering;

mod arena;
pub mod blas;
pub mod flash_attn;
pub mod im2col;
pub mod microkernels;
pub mod reductions_fast;
pub mod telemetry;

mod elementwise;
use elementwise::fused_binary_activation_dispatch;
mod scalar;
use scalar::{scalar_kernel_instruction, scalar_op_dispatch, unary_op_dispatch};
mod params;
use params::resolve_params;
mod matmul;
use matmul::{
    matmul_activation_dispatch, packed_tensor_from_meta, quantized_matmul_dispatch,
    quantized_matmul_dispatch_i8_u4, quantized_matmul_dispatch_i8_u8,
};

/// Minimum number of elements for a parallel dispatch loop to be beneficial.
/// Below this threshold, sequential execution avoids rayon's task-spawning
/// overhead without measurable throughput loss.
#[cfg(feature = "parallel")]
const PARALLEL_MIN_ELEMS: usize = 1024;

// ============================================================
// Pre-copied kernel dispatch helpers
// ============================================================
// These are variants of the arena-based dispatch helpers that work
// with pre-copied input buffers (&[Vec<u8>]) and a single output
// buffer (&mut [u8]), suitable for out-of-arena execution.

/// Helper: extract two f32 slices from pre-copied inputs, broadcast-loop
/// with a binary op and activation function, and write to output.
#[inline]
fn fused_binary_activation_dispatch_precopied(
    kernel_name: &str,
    inputs: &[Vec<u8>],
    output: &mut [u8],
    op: impl Fn(f32, f32) -> f32 + Sync,
    act: impl Fn(f32) -> f32 + Sync,
) {
    if inputs.len() >= 2 {
        let a = bytemuck::cast_slice::<_, f32>(&inputs[0]);
        let b = bytemuck::cast_slice::<_, f32>(&inputs[1]);
        let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if microkernels::simd_avx2_available() && out_f32.len() >= 8 {
            let binary_dispatched = if kernel_name.starts_with("add_") {
                add_f32(a, b, out_f32);
                true
            } else if kernel_name.starts_with("sub_") {
                sub_f32(a, b, out_f32);
                true
            } else if kernel_name.starts_with("mul_") {
                mul_f32(a, b, out_f32);
                true
            } else if kernel_name.starts_with("div_") {
                div_f32(a, b, out_f32);
                true
            } else {
                false
            };

            if binary_dispatched {
                let len = out_f32.len();
                let ptr = out_f32.as_mut_ptr();
                if kernel_name.ends_with("relu_f32") {
                    unsafe {
                        let v = std::slice::from_raw_parts(ptr, len);
                        microkernels::relu_f32_avx2(v, std::slice::from_raw_parts_mut(ptr, len));
                    }
                } else if kernel_name.ends_with("gelu_f32") {
                    unsafe {
                        let v = std::slice::from_raw_parts(ptr, len);
                        microkernels::gelu_f32_avx2(v, std::slice::from_raw_parts_mut(ptr, len));
                    }
                } else if kernel_name.ends_with("silu_f32") {
                    unsafe {
                        let v = std::slice::from_raw_parts(ptr, len);
                        microkernels::silu_f32_avx2(v, std::slice::from_raw_parts_mut(ptr, len));
                    }
                }
                return;
            }
        }

        let out_len = out_f32.len();
        let a_len = a.len();
        let b_len = b.len();
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..out_len {
                let x = op(a[i % a_len], b[i % b_len]);
                out_f32[i] = act(x);
            }
        }
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out_f32[..out_len]
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| {
                    let x = op(a[i % a_len], b[i % b_len]);
                    *o = act(x);
                });
        }
    }
}

/// Helper: dispatch a scalar op (gt, lt, eq, add, mul, div) from pre-copied inputs.
#[inline]
fn scalar_op_dispatch_precopied(
    inputs: &[Vec<u8>],
    output: &mut [u8],
    op: impl Fn(&[f32], f32, &mut [f32]),
) {
    if inputs.len() >= 2 {
        let data = bytemuck::cast_slice::<_, f32>(&inputs[0]);
        let scalar_data = bytemuck::cast_slice::<_, f32>(&inputs[1]);
        let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
        let s = scalar_data.first().copied().unwrap_or(0.0);
        op(data, s, out_f32);
    }
}

/// Helper: dispatch a fused matmul + bias + activation from pre-copied inputs.
#[inline]
fn matmul_activation_dispatch_precopied(
    inputs: &[Vec<u8>],
    output: &mut [u8],
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    kernel_name: &str,
    act: impl Fn(f32) -> f32,
) -> Result<(), BackendError> {
    if inputs.len() >= 2 {
        let a = bytemuck::cast_slice::<_, f32>(&inputs[0]);
        let b = bytemuck::cast_slice::<_, f32>(&inputs[1]);
        let bias = if inputs.len() >= 3 {
            bytemuck::cast_slice::<_, f32>(&inputs[2]).to_vec()
        } else {
            Vec::new()
        };
        let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
        let &[m, _k, n] = &matmul_params[..] else {
            return Err(BackendError::Dispatch(format!(
                "{kernel_name}: expected params [M,K,N]"
            )));
        };
        let has_bias = !bias.is_empty();
        let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
        matmul_blas_into(a, b, out_f32, m, _k, n);
        for i in 0..out_f32.len() {
            let x = out_f32[i]
                + if has_bias && i % n < bias.len() {
                    bias[i % n]
                } else {
                    0.0
                };
            out_f32[i] = act(x);
        }
    }
    Ok(())
}

/// Helper: dispatch a quantized matmul (u4 or u8) from pre-copied inputs.
#[inline]
fn quantized_matmul_dispatch_precopied<T: PackedWord>(
    inputs: &[Vec<u8>],
    output: &mut [u8],
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    weight_meta: &Option<crate::backend::QuantizedWeightMeta>,
    bit_width: usize,
    kernel_name: &str,
    persistent_view: Option<&crate::backend::prepared::PersistentPreparedWeights>,
) -> Result<(), BackendError> {
    if inputs.len() >= 2 {
        let activations = bytemuck::cast_slice::<_, f32>(&inputs[0]);
        let raw = &inputs[1];
        let mut packed_bytes = vec![0u32; raw.len().div_ceil(4)];
        let byte_slice = bytemuck::cast_slice_mut::<_, u8>(&mut packed_bytes);
        byte_slice[..raw.len()].copy_from_slice(raw);
        let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
        let &[m, k, n] = &matmul_params[..] else {
            return Err(BackendError::Dispatch(format!(
                "{kernel_name}: expected params [M,K,N]"
            )));
        };
        let meta = weight_meta
            .clone()
            .unwrap_or_else(|| crate::backend::QuantizedWeightMeta {
                bit_width,
                scales: vec![1.0],
                zero_points: vec![0.0],
                shape: vec![m, k],
            });
        let typed_data: Vec<T> = bytemuck::cast_slice(&packed_bytes).to_vec();
        let pt = packed_tensor_from_meta(typed_data, meta, kernel_name)?;
        let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
        crate::backend::cpu::microkernels::gemm_cpu_flat::<T>(&pt, activations, out_f32, m, k, n);
    }
    Ok(())
}

/// Pre-copied dispatch for I8 activation × U8x4 packed-weight MatMul.
///
/// Reads activation as raw bytes (I8 payload format), weight as raw bytes,
/// builds a `PackedTensor<U8x4>`, and calls the scalar I8×U8x4 microkernel.
fn quantized_matmul_dispatch_precopied_i8_u8(
    inputs: &[Vec<u8>],
    output: &mut [u8],
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    weight_meta: &Option<crate::backend::QuantizedWeightMeta>,
    kernel_name: &str,
) -> Result<(), BackendError> {
    if inputs.len() >= 2 {
        let activation_payload = &inputs[0];
        let raw = &inputs[1];
        let mut packed_bytes = vec![0u32; raw.len().div_ceil(4)];
        let byte_slice = bytemuck::cast_slice_mut::<_, u8>(&mut packed_bytes);
        byte_slice[..raw.len()].copy_from_slice(raw);
        let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
        let &[m, k, n] = &matmul_params[..] else {
            return Err(BackendError::Dispatch(format!(
                "{kernel_name}: expected params [M,K,N]"
            )));
        };
        let meta = weight_meta
            .clone()
            .unwrap_or_else(|| crate::backend::QuantizedWeightMeta {
                bit_width: 8,
                scales: vec![1.0],
                zero_points: vec![0.0],
                shape: vec![m, k],
            });
        let typed_data: Vec<U8x4> = bytemuck::cast_slice(&packed_bytes).to_vec();
        let pt = packed_tensor_from_meta(typed_data, meta, kernel_name)?;
        let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
        crate::backend::cpu::microkernels::gemm_cpu_flat_i8_u8x4(
            &pt,
            activation_payload,
            out_f32,
            m,
            k,
            n,
        );
    }
    Ok(())
}

/// Pre-copied dispatch for I8 activation × U4x8 packed-weight MatMul.
///
/// Reads activation as raw bytes (I8 payload format), weight as raw bytes,
/// builds a `PackedTensor<U4x8>`, and calls the scalar I8×U4x8 microkernel.
fn quantized_matmul_dispatch_precopied_i8_u4(
    inputs: &[Vec<u8>],
    output: &mut [u8],
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    weight_meta: &Option<crate::backend::QuantizedWeightMeta>,
    kernel_name: &str,
) -> Result<(), BackendError> {
    if inputs.len() >= 2 {
        let activation_payload = &inputs[0];
        let raw = &inputs[1];
        let mut packed_bytes = vec![0u32; raw.len().div_ceil(4)];
        let byte_slice = bytemuck::cast_slice_mut::<_, u8>(&mut packed_bytes);
        byte_slice[..raw.len()].copy_from_slice(raw);
        let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
        let &[m, k, n] = &matmul_params[..] else {
            return Err(BackendError::Dispatch(format!(
                "{kernel_name}: expected params [M,K,N]"
            )));
        };
        let meta = weight_meta
            .clone()
            .unwrap_or_else(|| crate::backend::QuantizedWeightMeta {
                bit_width: 4,
                scales: vec![1.0],
                zero_points: vec![0.0],
                shape: vec![m, k],
            });
        let typed_data: Vec<U4x8> = bytemuck::cast_slice(&packed_bytes).to_vec();
        let pt = packed_tensor_from_meta(typed_data, meta, kernel_name)?;
        let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
        crate::backend::cpu::microkernels::gemm_cpu_flat_i8_u4x8(
            &pt,
            activation_payload,
            out_f32,
            m,
            k,
            n,
        );
    }
    Ok(())
}

/// Execute a kernel with pre-copied input buffers and an exclusive output buffer.
///
/// Transformed version of the `dispatch()` match arms — replaces arena-based
/// I/O (`CpuBuffer` + `BufferSlice`) with direct slice access.
///
/// # Parameters
///
/// * `kernel_name` — identifies the kernel to run
/// * `inputs` — pre-copied input data (each element is an owned byte vector)
/// * `output` — mutable output byte buffer
// ── Helper macros for run_kernel_precopied ──────────────────────────
/// Expands to the body of a unary match arm: cast input[0]→f32, cast output→f32, call fn
macro_rules! precopied_unary_body {
    ($inputs:expr, $output:expr, $fn:ident) => {{
        if !$inputs.is_empty() {
            let a = bytemuck::cast_slice::<_, f32>(&$inputs[0]);
            let out = bytemuck::cast_slice_mut::<_, f32>($output);
            $fn(a, out);
        }
    }};
}
/// Expands to the body of a binary match arm: cast input[0], input[1], output, call fn
macro_rules! precopied_binary_body {
    ($inputs:expr, $output:expr, $fn:ident) => {{
        let a = bytemuck::cast_slice::<_, f32>(&$inputs[0]);
        let b = bytemuck::cast_slice::<_, f32>(&$inputs[1]);
        let out = bytemuck::cast_slice_mut::<_, f32>($output);
        $fn(a, b, out);
    }};
}

/// * `params` — integer kernel parameters
/// * `param_dims` — optional symbolic dimension expressions
/// * `weight_meta` — optional quantized weight metadata
/// * `shape_env` — runtime shape environment for symbolic resolution
#[allow(clippy::cognitive_complexity)]
fn run_kernel_precopied(
    kernel_name: &str,
    inputs: &[Vec<u8>],
    output: &mut [u8],
    _secondary_output: Option<&mut [u8]>,
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    weight_meta: &Option<crate::backend::QuantizedWeightMeta>,
    shape_env: &ShapeEnv,
) -> Result<(), BackendError> {
    match kernel_name {
        // ── Binary elementwise ────────────────────────────────
        "add_f32" => precopied_binary_body!(inputs, output, add_f32),
        "sub_f32" => precopied_binary_body!(inputs, output, sub_f32),
        "mul_f32" => precopied_binary_body!(inputs, output, mul_f32),
        "div_f32" => precopied_binary_body!(inputs, output, div_f32),
        // ── Unary elementwise ────────────────────────────────
        "relu_f32" => precopied_unary_body!(inputs, output, relu_f32),
        "gelu_f32" => precopied_unary_body!(inputs, output, gelu_f32),
        "silu_f32" => precopied_unary_body!(inputs, output, silu_f32),
        "exp_f32" => precopied_unary_body!(inputs, output, exp_f32),
        "log_f32" => precopied_unary_body!(inputs, output, log_f32),
        "sqrt_f32" => precopied_unary_body!(inputs, output, sqrt_f32),
        "neg_f32" => precopied_unary_body!(inputs, output, neg_f32),
        "abs_f32" => precopied_unary_body!(inputs, output, abs_f32),
        "sigmoid_f32" => precopied_unary_body!(inputs, output, sigmoid_f32),
        "tanh_f32" => precopied_unary_body!(inputs, output, tanh_f32),
        "elu_f32" => precopied_unary_body!(inputs, output, elu_f32),
        "softplus_f32" => precopied_unary_body!(inputs, output, softplus_f32),
        "hardswish_f32" => precopied_unary_body!(inputs, output, hardswish_f32),
        "sign_f32" => precopied_unary_body!(inputs, output, sign_f32),
        "round_f32" => precopied_unary_body!(inputs, output, round_f32),
        "logical_not_f32" => precopied_unary_body!(inputs, output, logical_not_f32),
        "log_softmax_f32" => precopied_unary_body!(inputs, output, log_softmax_f32),
        "mish_f32" => precopied_unary_body!(inputs, output, mish_f32),
        "max_f32" => precopied_binary_body!(inputs, output, max_f32),
        "min_f32" => precopied_binary_body!(inputs, output, min_f32),
        // ── Unary with scalar params ───────────────────────────
        "leaky_relu_f32" => {
            if !inputs.is_empty() {
                let a = bytemuck::cast_slice::<_, f32>(&inputs[0]);
                let out = bytemuck::cast_slice_mut::<_, f32>(output);
                let slope = if !params.is_empty() {
                    f32::from_bits(params[0] as u32)
                } else {
                    0.01
                };
                leaky_relu_f32(a, out, slope);
            }
        }
        "clamp_f32" => {
            if !inputs.is_empty() {
                let a = bytemuck::cast_slice::<_, f32>(&inputs[0]);
                let out = bytemuck::cast_slice_mut::<_, f32>(output);
                let min_val = if !params.is_empty() {
                    f32::from_bits(params[0] as u32)
                } else {
                    0.0
                };
                let max_val = if params.len() > 1 {
                    f32::from_bits(params[1] as u32)
                } else {
                    1.0
                };
                clamp_f32(a, out, min_val, max_val);
            }
        }
        // ── Matmul ────────────────────────────────────────────
        "matmul" => {
            if inputs.len() >= 2 {
                let a = bytemuck::cast_slice::<_, f32>(&inputs[0]);
                let b = bytemuck::cast_slice::<_, f32>(&inputs[1]);
                let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
                let &[m, _k, n] = &matmul_params[..] else {
                    return Err(BackendError::Dispatch(
                        "matmul: expected params [M,K,N]".into(),
                    ));
                };
                let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                let a_stride = m * _k;
                let b_stride = _k * n;
                let out_stride = m * n;
                let batch_count = out_f32.len() / out_stride;
                let b_batched = b.len() > b_stride;
                let use_blas = m * _k * n >= blas::MIN_BLAS_SIZE * 64;
                if use_blas {
                    for batch in 0..batch_count {
                        let a_s = batch * a_stride;
                        let b_s = if b_batched { batch * b_stride } else { 0 };
                        let out_s = batch * out_stride;
                        matmul_blas_into(
                            &a[a_s..a_s + a_stride],
                            &b[b_s..b_s + b_stride],
                            &mut out_f32[out_s..out_s + out_stride],
                            m,
                            _k,
                            n,
                        );
                    }
                } else {
                    let total_rows = batch_count * m;
                    let b_batch_stride = if b_batched { b_stride } else { 0 };
                    #[cfg(feature = "parallel")]
                    {
                        use rayon::prelude::*;
                        let a_raw = a.as_ptr() as usize;
                        let b_raw = b.as_ptr() as usize;
                        let out_raw = out_f32.as_mut_ptr() as usize;
                        (0..total_rows).into_par_iter().for_each(move |row| {
                            let a_ptr = a_raw as *const f32;
                            let b_ptr = b_raw as *const f32;
                            let out_ptr = out_raw as *mut f32;
                            unsafe {
                                crate::backend::cpu::microkernels::blocked_row_matmul(
                                    a_ptr,
                                    b_ptr,
                                    out_ptr,
                                    row,
                                    m,
                                    n,
                                    _k,
                                    a_stride,
                                    _k,
                                    1,
                                    b_batch_stride,
                                    n,
                                    1,
                                );
                            }
                        });
                    }
                    #[cfg(not(feature = "parallel"))]
                    for row in 0..total_rows {
                        unsafe {
                            crate::backend::cpu::microkernels::blocked_row_matmul(
                                a.as_ptr(),
                                b.as_ptr(),
                                out_f32.as_mut_ptr(),
                                row,
                                m,
                                n,
                                _k,
                                a_stride,
                                _k,
                                1,
                                b_batch_stride,
                                n,
                                1,
                            );
                        }
                    }
                }
            }
        }
        // ── Fused matmul + activation ──────────────────────────
        "fused_matmul_add_relu" => {
            matmul_activation_dispatch_precopied(
                inputs,
                output,
                params,
                param_dims,
                shape_env,
                "fused_matmul_add_relu",
                |x| x.max(0.0),
            )?;
        }
        "fused_matmul_add_gelu" => {
            matmul_activation_dispatch_precopied(
                inputs,
                output,
                params,
                param_dims,
                shape_env,
                "fused_matmul_add_gelu",
                |x| {
                    let x3 = x * x * x;
                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                    let t = tanh_arg.tanh();
                    0.5 * x * (1.0 + t)
                },
            )?;
        }
        "fused_matmul_add_silu" => {
            matmul_activation_dispatch_precopied(
                inputs,
                output,
                params,
                param_dims,
                shape_env,
                "fused_matmul_add_silu",
                |x| x / (1.0 + (-x).exp()),
            )?;
        }
        "matmul_relu" => {
            matmul_activation_dispatch_precopied(
                inputs,
                output,
                params,
                param_dims,
                shape_env,
                "matmul_relu",
                |x| x.max(0.0),
            )?;
        }
        "matmul_gelu" => {
            matmul_activation_dispatch_precopied(
                inputs,
                output,
                params,
                param_dims,
                shape_env,
                "matmul_gelu",
                |x| {
                    let x3 = x * x * x;
                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                    let t = tanh_arg.tanh();
                    0.5 * x * (1.0 + t)
                },
            )?;
        }
        "matmul_silu" => {
            matmul_activation_dispatch_precopied(
                inputs,
                output,
                params,
                param_dims,
                shape_env,
                "matmul_silu",
                |x| x / (1.0 + (-x).exp()),
            )?;
        }
        "matmul_u4" => {
            quantized_matmul_dispatch_precopied::<U4x8>(
                inputs,
                output,
                params,
                param_dims,
                shape_env,
                weight_meta,
                4,
                "matmul_u4",
                None,
            )?;
        }
        "matmul_u4_i8" => {
            quantized_matmul_dispatch_precopied_i8_u4(
                inputs,
                output,
                params,
                param_dims,
                shape_env,
                weight_meta,
                "matmul_u4_i8",
            )?;
        }
        "matmul_u8_i8" => {
            quantized_matmul_dispatch_precopied_i8_u8(
                inputs,
                output,
                params,
                param_dims,
                shape_env,
                weight_meta,
                "matmul_u8_i8",
            )?;
        }
        "matmul_u8" => {
            quantized_matmul_dispatch_precopied::<U8x4>(
                inputs,
                output,
                params,
                param_dims,
                shape_env,
                weight_meta,
                8,
                "matmul_u8",
                None,
            )?;
        }
        // ── Reduce ────────────────────────────────────────────
        "reduce_f32" => {
            let input = bytemuck::cast_slice::<_, f32>(&inputs[0]);
            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
            let &[group_size, is_mean, is_max] = &params[..3] else {
                return Err(BackendError::Dispatch(
                    "reduce_f32: expected params [group_size, is_mean, is_max]".into(),
                ));
            };
            let effective_group_size = match param_dims {
                Some(dims) if !dims.is_empty() => dims[0]
                    .evaluate_with_env(shape_env)
                    .map_err(|e| BackendError::Dispatch(format!("reduce_f32: {e}")))?
                    as usize,
                _ => group_size,
            };
            reduce_f32(
                input,
                out_f32,
                effective_group_size,
                is_mean == 1,
                is_max == 1,
            );
        }
        // ── Transpose ──────────────────────────────────────────
        "transpose_f32" => {
            if !inputs.is_empty() {
                let input = bytemuck::cast_slice::<_, f32>(&inputs[0]);
                let transpose_params = resolve_params(params, param_dims, shape_env, 2)?;
                let &[m, n] = &transpose_params[..] else {
                    return Err(BackendError::Dispatch(
                        "transpose_f32: expected params [M,N]".into(),
                    ));
                };
                let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                if microkernels::simd_avx2_available() && m >= 8 && n >= 8 {
                    unsafe {
                        microkernels::transpose_f32_avx2(input, out_f32, m, n);
                    }
                } else {
                    #[cfg(not(feature = "parallel"))]
                    {
                        for i in 0..m {
                            for j in 0..n {
                                out_f32[j * m + i] = input[i * n + j];
                            }
                        }
                    }
                    #[cfg(feature = "parallel")]
                    {
                        use rayon::prelude::*;
                        out_f32.par_chunks_mut(m).enumerate().for_each(|(j, col)| {
                            for i in 0..m {
                                col[i] = input[i * n + j];
                            }
                        });
                    }
                }
            }
        }
        "transpose_perm_f32" => {
            if !inputs.is_empty() {
                let input = bytemuck::cast_slice::<_, f32>(&inputs[0]);
                let rank = params.first().copied().unwrap_or(2);
                let nd_params = resolve_params(params, param_dims, shape_env, 1 + 2 * rank)?;
                let dims: Vec<usize> = nd_params[1..1 + rank].to_vec();
                let perm: Vec<usize> = nd_params[1 + rank..1 + 2 * rank].to_vec();
                let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                let mut in_strides = vec![1usize; rank];
                let mut out_strides = vec![1usize; rank];
                for i in (0..rank - 1).rev() {
                    in_strides[i] = in_strides[i + 1] * dims[i + 1];
                }
                for i in (0..rank - 1).rev() {
                    out_strides[perm[i]] = out_strides[perm[i + 1]] * dims[perm[i + 1]];
                }
                let _total = out_f32.len();
                #[cfg(not(feature = "parallel"))]
                {
                    let total = _total;
                    for out_idx in 0..total {
                        let mut in_idx = 0usize;
                        let mut remaining = out_idx;
                        for k in 0..rank {
                            let coord = remaining / out_strides[perm[k]];
                            remaining %= out_strides[perm[k]];
                            in_idx += coord * in_strides[perm[k]];
                        }
                        out_f32[out_idx] = input[in_idx];
                    }
                }
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    out_f32.par_iter_mut().enumerate().for_each(|(out_idx, v)| {
                        let mut in_idx = 0usize;
                        let mut remaining = out_idx;
                        for k in 0..rank {
                            let coord = remaining / out_strides[perm[k]];
                            remaining %= out_strides[perm[k]];
                            in_idx += coord * in_strides[perm[k]];
                        }
                        *v = input[in_idx];
                    });
                }
            }
        }
        // ── Fused binary + activation ──────────────────────────
        "add_relu_f32" => {
            fused_binary_activation_dispatch_precopied(
                "add_relu_f32",
                inputs,
                output,
                |a, b| a + b,
                |x| x.max(0.0),
            );
        }
        "sub_relu_f32" => {
            fused_binary_activation_dispatch_precopied(
                "sub_relu_f32",
                inputs,
                output,
                |a, b| a - b,
                |x| x.max(0.0),
            );
        }
        "mul_relu_f32" => {
            fused_binary_activation_dispatch_precopied(
                "mul_relu_f32",
                inputs,
                output,
                |a, b| a * b,
                |x| x.max(0.0),
            );
        }
        "div_relu_f32" => {
            fused_binary_activation_dispatch_precopied(
                "div_relu_f32",
                inputs,
                output,
                |a, b| a / b,
                |x| x.max(0.0),
            );
        }
        "add_gelu_f32" => {
            fused_binary_activation_dispatch_precopied(
                "add_gelu_f32",
                inputs,
                output,
                |a, b| a + b,
                |x| {
                    let x3 = x * x * x;
                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                    let t = tanh_arg.tanh();
                    0.5 * x * (1.0 + t)
                },
            );
        }
        "sub_gelu_f32" => {
            fused_binary_activation_dispatch_precopied(
                "sub_gelu_f32",
                inputs,
                output,
                |a, b| a - b,
                |x| {
                    let x3 = x * x * x;
                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                    let t = tanh_arg.tanh();
                    0.5 * x * (1.0 + t)
                },
            );
        }
        "mul_gelu_f32" => {
            fused_binary_activation_dispatch_precopied(
                "mul_gelu_f32",
                inputs,
                output,
                |a, b| a * b,
                |x| {
                    let x3 = x * x * x;
                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                    let t = tanh_arg.tanh();
                    0.5 * x * (1.0 + t)
                },
            );
        }
        "div_gelu_f32" => {
            fused_binary_activation_dispatch_precopied(
                "div_gelu_f32",
                inputs,
                output,
                |a, b| a / b,
                |x| {
                    let x3 = x * x * x;
                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                    let t = tanh_arg.tanh();
                    0.5 * x * (1.0 + t)
                },
            );
        }
        "add_silu_f32" => {
            fused_binary_activation_dispatch_precopied(
                "add_silu_f32",
                inputs,
                output,
                |a, b| a + b,
                |x| x / (1.0 + (-x).exp()),
            );
        }
        "sub_silu_f32" => {
            fused_binary_activation_dispatch_precopied(
                "sub_silu_f32",
                inputs,
                output,
                |a, b| a - b,
                |x| x / (1.0 + (-x).exp()),
            );
        }
        "mul_silu_f32" => {
            fused_binary_activation_dispatch_precopied(
                "mul_silu_f32",
                inputs,
                output,
                |a, b| a * b,
                |x| x / (1.0 + (-x).exp()),
            );
        }
        "div_silu_f32" => {
            fused_binary_activation_dispatch_precopied(
                "div_silu_f32",
                inputs,
                output,
                |a, b| a / b,
                |x| x / (1.0 + (-x).exp()),
            );
        }
        // ── Softmax ────────────────────────────────────────────
        "softmax" => {
            if !inputs.is_empty() {
                let input = bytemuck::cast_slice::<_, f32>(&inputs[0]);
                let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                let softmax_params = resolve_params(params, param_dims, shape_env, 2)
                    .unwrap_or_else(|_| vec![input.len(), 1]);
                let axis_dim_size = softmax_params[0].max(1);
                let stride = softmax_params.get(1).copied().unwrap_or(1).max(1);
                let num_rows = input.len() / axis_dim_size.max(1);
                softmax_f32(input, out_f32, axis_dim_size, stride, num_rows);
            }
        }
        // ── BiasAdd ────────────────────────────────────────────
        "biasadd" => {
            if inputs.len() >= 2 {
                let data = bytemuck::cast_slice::<_, f32>(&inputs[0]);
                let bias = bytemuck::cast_slice::<_, f32>(&inputs[1]);
                let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                let &[channel_stride] = params else {
                    return Err(BackendError::Dispatch(
                        "biasadd: expected params [channel_stride]".into(),
                    ));
                };
                biasadd_f32(data, bias, out_f32, channel_stride);
            }
        }
        // ── Normalization ──────────────────────────────────────
        "norm_f32" => {
            let &[eps_bits, is_batch_norm] = params else {
                return Err(BackendError::Dispatch(
                    "norm_f32: expected params [eps_bits, is_batch_norm]".into(),
                ));
            };
            let eps = f32::from_bits(eps_bits as u32);
            if is_batch_norm == 1 {
                if inputs.len() >= 5 {
                    let data = bytemuck::cast_slice::<_, f32>(&inputs[0]);
                    let weight = bytemuck::cast_slice::<_, f32>(&inputs[1]);
                    let bias = bytemuck::cast_slice::<_, f32>(&inputs[2]);
                    let running_mean = bytemuck::cast_slice::<_, f32>(&inputs[3]);
                    let running_var = bytemuck::cast_slice::<_, f32>(&inputs[4]);
                    let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                    {
                        use crate::backend::cpu::microkernels::has_avx2;
                        if has_avx2() {
                            unsafe {
                                crate::backend::cpu::microkernels::batch_norm_inference_f32_avx2(
                                    data,
                                    weight,
                                    bias,
                                    running_mean,
                                    running_var,
                                    out_f32,
                                    eps,
                                );
                            }
                        } else {
                            crate::backend::cpu::microkernels::batch_norm_inference_f32(
                                data,
                                weight,
                                bias,
                                running_mean,
                                running_var,
                                out_f32,
                                eps,
                            );
                        }
                    }
                    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                    {
                        crate::backend::cpu::microkernels::batch_norm_inference_f32(
                            data,
                            weight,
                            bias,
                            running_mean,
                            running_var,
                            out_f32,
                            eps,
                        );
                    }
                }
            } else {
                if !inputs.is_empty() {
                    let input = bytemuck::cast_slice::<_, f32>(&inputs[0]);
                    let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                    let row_size = input.len() / out_f32.len().max(1);
                    norm_layernorm_f32(input, out_f32, row_size, eps);
                }
            }
        }
        // ── Conv2d ─────────────────────────────────────────────
        "conv2d" | "conv2d_relu" | "conv2d_gelu" | "conv2d_silu" => {
            let fused_act = match kernel_name {
                "conv2d_relu" => Some("relu"),
                "conv2d_gelu" => Some("gelu"),
                "conv2d_silu" => Some("silu"),
                _ => None,
            };
            if inputs.len() >= 2 {
                let &[stride, padding, dilation, groups, c, h, w, kh, kw] = params else {
                    return Err(BackendError::Dispatch("conv2d: expected params [stride, padding, dilation, groups, c, h, w, kh, kw]".into()));
                };
                let c_per_group = c / groups.max(1);
                let input_data = bytemuck::cast_slice::<_, f32>(&inputs[0]).to_vec();
                let weight_data = bytemuck::cast_slice::<_, f32>(&inputs[1]).to_vec();
                let bias_data = if inputs.len() >= 3 {
                    bytemuck::cast_slice::<_, f32>(&inputs[2]).to_vec()
                } else {
                    vec![]
                };
                let n = input_data.len() / (c * h * w).max(1);
                let f = weight_data.len() / (c_per_group * kh * kw).max(1);
                let _h_out = (h + 2 * padding).saturating_sub(dilation * (kh - 1) + 1) / stride + 1;
                let _w_out = (w + 2 * padding).saturating_sub(dilation * (kw - 1) + 1) / stride + 1;
                let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                let conv_act = fused_act.map(|act| match act {
                    "relu" => crate::backend::cpu::microkernels::ConvActivation::Relu,
                    "gelu" => crate::backend::cpu::microkernels::ConvActivation::Gelu,
                    "silu" => crate::backend::cpu::microkernels::ConvActivation::Silu,
                    _ => unreachable!(),
                });
                crate::backend::cpu::microkernels::conv2d_f32_im2col_gemm(
                    &input_data,
                    &weight_data,
                    &bias_data,
                    out_f32,
                    n,
                    c,
                    h,
                    w,
                    f,
                    kh,
                    kw,
                    stride,
                    padding,
                    dilation,
                    groups,
                    conv_act,
                );
            }
        }
        // ── Concat ─────────────────────────────────────────────
        "concat" => {
            if !inputs.is_empty() && params.len() >= 3 {
                let _axis = params[0];
                let _inner_stride = params[1];
                let outer_count = params[2];
                let num_inputs = inputs.len();
                let mut block_sizes: Vec<usize> = Vec::with_capacity(num_inputs);
                for inp in inputs {
                    let elems = inp.len() / std::mem::size_of::<f32>();
                    block_sizes.push(elems / outer_count.max(1));
                }
                let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                let mut output_offset = 0;
                for outer_pos in 0..outer_count {
                    for (si, inp) in inputs.iter().enumerate() {
                        let input_data = bytemuck::cast_slice::<_, f32>(inp);
                        let block = block_sizes[si];
                        let src_start = outer_pos * block;
                        let src_end = (src_start + block).min(input_data.len());
                        let dst_end = (output_offset + src_end - src_start).min(out_f32.len());
                        out_f32[output_offset..dst_end]
                            .copy_from_slice(&input_data[src_start..src_end]);
                        output_offset += src_end - src_start;
                    }
                }
            }
        }
        // ── Scalar ops ─────────────────────────────────────────
        "gt_f32" => scalar_op_dispatch_precopied(inputs, output, |data, s, out| {
            for i in 0..out.len() {
                out[i] = if data[i % data.len()] > s { 1.0 } else { 0.0 };
            }
        }),
        "lt_f32" => scalar_op_dispatch_precopied(inputs, output, |data, s, out| {
            for i in 0..out.len() {
                out[i] = if data[i % data.len()] < s { 1.0 } else { 0.0 };
            }
        }),
        "eq_f32" => scalar_op_dispatch_precopied(inputs, output, |data, s, out| {
            for i in 0..out.len() {
                out[i] = if (data[i % data.len()] - s).abs() < 1e-6 {
                    1.0
                } else {
                    0.0
                };
            }
        }),
        "add_scalar_f32" => scalar_op_dispatch_precopied(inputs, output, |data, s, out| {
            for i in 0..out.len() {
                out[i] = data[i % data.len()] + s;
            }
        }),
        "mul_scalar_f32" => scalar_op_dispatch_precopied(inputs, output, |data, s, out| {
            for i in 0..out.len() {
                out[i] = data[i % data.len()] * s;
            }
        }),
        "div_scalar_f32" => scalar_op_dispatch_precopied(inputs, output, |data, s, out| {
            for i in 0..out.len() {
                out[i] = data[i % data.len()] / s;
            }
        }),
        "where_f32" => {
            if inputs.len() >= 3 {
                let cond = bytemuck::cast_slice::<_, f32>(&inputs[0]);
                let a = bytemuck::cast_slice::<_, f32>(&inputs[1]);
                let b = bytemuck::cast_slice::<_, f32>(&inputs[2]);
                let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                for i in 0..out_f32.len() {
                    out_f32[i] = if cond[i % cond.len()] != 0.0 {
                        a[i % a.len()]
                    } else {
                        b[i % b.len()]
                    };
                }
            }
        }
        "cast_f32_i64" => {
            let input = bytemuck::cast_slice::<_, f32>(&inputs[0]);
            let out_i64 = bytemuck::cast_slice_mut::<_, i64>(output);
            for i in 0..out_i64.len() {
                out_i64[i] = input[i % input.len()] as i64;
            }
        }
        "cast_i64_f32" => {
            let input = bytemuck::cast_slice::<_, i64>(&inputs[0]);
            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
            for i in 0..out_f32.len() {
                out_f32[i] = input[i % input.len()] as f32;
            }
        }
        "slice_f32" => {
            if !inputs.is_empty() {
                let input = bytemuck::cast_slice::<_, f32>(&inputs[0]);
                let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                let dst_len = out_f32.len();
                let copy_len = dst_len.min(input.len());
                out_f32[..copy_len].copy_from_slice(&input[..copy_len]);
            }
        }
        "shape_f32" | "shape_i64" => {
            if !inputs.is_empty() {
                let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                for (i, inp) in inputs.iter().enumerate().take(out_f32.len()) {
                    let f = bytemuck::cast_slice::<_, f32>(inp);
                    out_f32[i] = f.first().copied().unwrap_or(0.0);
                }
            }
        }
        "squeeze_f32" => {
            if !inputs.is_empty() {
                let input = bytemuck::cast_slice::<_, f32>(&inputs[0]);
                let out_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
                let copy_len = out_f32.len().min(input.len());
                out_f32[..copy_len].copy_from_slice(&input[..copy_len]);
            }
        }
        // ── Catch-all for any kernel not yet covered ───────────
        other => {
            return Err(BackendError::UnsupportedOp(format!(
                "run_kernel_precopied: unsupported kernel '{}'",
                other
            )));
        }
    }
    Ok(())
}

/// Execute a group of independent instructions, processing all inputs in
/// parallel when the `parallel` feature is enabled.
///
/// All instructions in the group have the same topological level and no
/// data dependencies between them. Inputs are pre-copied from the arena
/// (serial phase), then each instruction is dispatched to its own kernel
/// via [`run_kernel_precopied`] with an exclusive output slice.
///
/// When the `parallel` feature is disabled, instructions execute sequentially
/// (still safe — just no overlap).
fn execute_parallel_level(
    group: &[usize],
    plan: &ExecutablePlan,
    arena: &CpuBuffer,
    shape_env: &ShapeEnv,
) -> Result<(), BackendError> {
    // ── Phase 1: Pre-copy all inputs from the arena (serial) ──────
    struct ParallelTask {
        kernel_name: String,
        inputs: Vec<Vec<u8>>,
        output_offset: usize,
        output_size: usize,
        secondary_output_offset: Option<usize>,
        secondary_output_size: Option<usize>,
        params: Vec<usize>,
        param_dims: Option<Vec<DimExpr>>,
        weight_meta: Option<crate::backend::QuantizedWeightMeta>,
    }

    let mut ptasks: Vec<ParallelTask> = Vec::with_capacity(group.len());
    for &instr_idx in group {
        let instr = &plan.instructions[instr_idx];
        match instr {
            Instruction::CallKernel {
                kernel_name,
                input_slices,
                output_slice,
                secondary_output_slice,
                params,
                param_dims,
                weight_meta,
                ..
            } => {
                let d = arena.data_mut();
                let inputs: Vec<Vec<u8>> = input_slices
                    .iter()
                    .map(|slice| {
                        let start = slice.offset;
                        let end = start + slice.size;
                        d[start..end].to_vec()
                    })
                    .collect();
                ptasks.push(ParallelTask {
                    kernel_name: kernel_name.clone(),
                    inputs,
                    output_offset: output_slice.offset,
                    output_size: output_slice.size,
                    secondary_output_offset: secondary_output_slice.as_ref().map(|s| s.offset),
                    secondary_output_size: secondary_output_slice.as_ref().map(|s| s.size),
                    params: params.clone(),
                    param_dims: param_dims.clone(),
                    weight_meta: weight_meta.clone(),
                });
            }
            // MemCopy / Fill / WriteConst are fast — execute inline (serial)
            Instruction::MemCopy { dst, src } => {
                let data = arena.data_mut();
                let len = dst.size.min(src.size);
                data.copy_within(src.offset..src.offset + len, dst.offset);
            }
            Instruction::Fill { dst, value } => {
                let data = arena.data_mut();
                let bytes = &mut data[dst.offset..dst.offset + dst.size];
                let f32_slice = bytemuck::cast_slice_mut::<_, f32>(bytes);
                f32_slice.fill(*value);
            }
            Instruction::WriteConst { dst, data } => {
                let arena_data = arena.data_mut();
                let end = (dst.offset + data.len()).min(arena_data.len());
                arena_data[dst.offset..end].copy_from_slice(&data[..end - dst.offset]);
            }
        }
    }

    #[cfg(feature = "parallel")]
    {
        // ── Phase 2: Execute kernel tasks in parallel ────────────────
        use rayon::prelude::*;
        let arena_base = arena.data_mut().as_mut_ptr() as usize;
        // SAFETY: ShapeEnv is read-only during dispatch; no concurrent writes occur.
        // We pass it as a raw pointer (isize) to satisfy Send requirements.
        let shape_env_ptr = shape_env as *const ShapeEnv as usize;

        let errors: Vec<_> = ptasks
            .into_par_iter()
            .map(|task| {
                // SAFETY: each task writes to a non-overlapping region of the arena
                // (guaranteed by the memory planner — same-level instructions have
                // distinct output regions).
                let output = unsafe {
                    std::slice::from_raw_parts_mut(
                        (arena_base + task.output_offset) as *mut u8,
                        task.output_size,
                    )
                };
                let secondary = task.secondary_output_offset.map(|off| unsafe {
                    std::slice::from_raw_parts_mut(
                        (arena_base + off) as *mut u8,
                        task.secondary_output_size.unwrap_or(0),
                    )
                });
                // SAFETY: ShapeEnv is read-only and lives for the duration of dispatch().
                let se = unsafe { &*(shape_env_ptr as *const ShapeEnv) };
                run_kernel_precopied(
                    &task.kernel_name,
                    &task.inputs,
                    output,
                    secondary,
                    &task.params,
                    &task.param_dims,
                    &task.weight_meta,
                    se,
                )
            })
            .collect();

        // Check for errors from parallel tasks
        for result in errors {
            result?;
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        // Fallback: execute sequentially using pre-copied inputs (same code path,
        // just without the parallelism).
        for task in &ptasks {
            let output = {
                let d = arena.data_mut();
                &mut d[task.output_offset..task.output_offset + task.output_size]
            };
            let secondary = task.secondary_output_offset.map(|off| {
                let d = arena.data_mut();
                let sz = task.secondary_output_size.unwrap_or(0);
                &mut d[off..off + sz]
            });
            run_kernel_precopied(
                &task.kernel_name,
                &task.inputs,
                output,
                secondary,
                &task.params,
                &task.param_dims,
                &task.weight_meta,
                shape_env,
            )?;
        }
    }
    Ok(())
}

/// Build level groups from the plan's topological levels.
///
/// Returns a `Vec<Vec<usize>>` where `groups[lvl]` contains the instruction
/// indices at that level. Groups are in ascending level order. Levels with
/// no instructions produce empty vectors.
fn build_level_groups(levels: &[usize]) -> Vec<Vec<usize>> {
    if levels.is_empty() {
        return vec![];
    }
    let max_level = *levels.iter().max().unwrap_or(&0);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for (idx, &lvl) in levels.iter().enumerate() {
        groups[lvl].push(idx);
    }
    groups
}

/// CPU memory arena with interior mutability for zero-allocation dispatch.
///
/// # Soundness
///
/// `CpuBuffer` wraps [`Vec<u8>`] in an [`UnsafeCell`] so that the
/// [`dispatch`](Backend::dispatch) method can mutate the arena through
/// a shared `&CpuBuffer` reference.  Dispatch is single-threaded and
/// processes instructions sequentially, so the `&mut [u8]` slices
/// returned by [`data_mut`](CpuBuffer::data_mut) are never aliased.
///
/// `Send + Sync` are safe because the inner `Vec<u8>` is itself
/// `Send + Sync` and the arena is never accessed concurrently.
pub struct CpuBuffer(UnsafeCell<Vec<u8>>);

impl CpuBuffer {
    pub fn new(data: Vec<u8>) -> Self {
        CpuBuffer(UnsafeCell::new(data))
    }

    /// Get a mutable slice to the arena data.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other `&mut [u8]` reference
    /// derived from this arena is live when this method is called.
    /// This is satisfied by the sequential dispatch loop — each
    /// `data_mut` call's borrow ends before the next one begins.
    #[allow(clippy::mut_from_ref)]
    pub fn data_mut(&self) -> &mut [u8] {
        unsafe { &mut *self.0.get() }.as_mut_slice()
    }
}

// SAFETY: `Vec<u8>` is `Send + Sync`.  The arena is never accessed
// concurrently — dispatch is single-threaded — so interior mutability
// via `UnsafeCell` does not introduce data races.
unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

/// CPU execution context. Zero allocation during dispatch.
#[derive(Clone)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Buffer = CpuBuffer;

    fn name(&self) -> &str {
        "cpu"
    }

    fn allocate_arena(&self, total_bytes: usize) -> CpuBuffer {
        // Zero-fill the arena to guarantee deterministic behavior across
        // platform allocators.  On Linux, mmap-backed pages are typically
        // zero-filled by the kernel, but macOS and Windows allocators may
        // return reused memory with non-zero contents.  Zero-initialization
        // ensures that any kernel that fails to write its output slot
        // (e.g. due to a missed dispatch path) produces zeros instead of
        // platform-dependent garbage.
        let buf = vec![0u8; total_bytes];
        CpuBuffer::new(buf)
    }

    fn compile(
        &self,
        graph: &ComputeGraph,
        memory_plan: &MemoryPlan,
    ) -> Result<ExecutablePlan, BackendError> {
        let mut instructions = Vec::new();
        let order = graph.topological_sort();

        // ── Pre‑compute topological levels for parallel dispatch ──────
        // level[node] = max(level[input]) + 1   (with input level = 0 for graph inputs)
        let mut node_level: std::collections::HashMap<NodeId, usize> =
            std::collections::HashMap::new();
        for &node_id in &order {
            let node = graph
                .get_node(node_id)
                .expect("node in topological order must exist");
            let level = node
                .inputs
                .iter()
                .filter_map(|id| node_level.get(id))
                .max()
                .map(|l| l + 1)
                .unwrap_or(0);
            node_level.insert(node_id, level);
        }
        let mut instruction_levels: Vec<usize> = Vec::with_capacity(order.len());

        for &node_id in &order {
            let node = graph
                .get_node(node_id)
                .ok_or_else(|| BackendError::Compilation(format!("Node {} not found", node_id)))?;
            let level = node_level[&node_id];

            let input_slices: Vec<BufferSlice> = node
                .inputs
                .iter()
                .filter_map(|&input_id| {
                    memory_plan
                        .slots
                        .get(&input_id)
                        .map(|slot| BufferSlice::new(slot.offset, slot.size))
                })
                .collect();

            // Collect input shapes for dimension-dependent kernels.
            // Symbolic dims that can't be resolved at compile time are
            // replaced with SYMBOL_DIM_MAX to preserve shape rank.
            let symbol_max = crate::ir::node::SYMBOL_DIM_MAX.load(Ordering::Relaxed);
            let input_shapes: Vec<Vec<u64>> = node
                .inputs
                .iter()
                .filter_map(|&input_id| graph.get_node(input_id))
                .map(|n| {
                    n.output_type
                        .shape
                        .iter()
                        .map(|d| d.evaluate().unwrap_or(symbol_max))
                        .collect()
                })
                .collect();
            // Also collect raw DimExpr shapes for symbolic dispatch resolution
            let input_shape_dims: Vec<Vec<DimExpr>> = node
                .inputs
                .iter()
                .filter_map(|&input_id| graph.get_node(input_id))
                .map(|n| n.output_type.shape.clone())
                .collect();

            let output_slice = memory_plan
                .slots
                .get(&node_id)
                .map(|slot| BufferSlice::new(slot.offset, slot.size))
                .ok_or_else(|| {
                    BackendError::Compilation(format!(
                        "node {} ({:?}) has no memory slot",
                        node_id, node.opcode
                    ))
                })?;
            let secondary_output_slice = memory_plan
                .secondary_slots
                .get(&(node_id, 1))
                .map(|slot| BufferSlice::new(slot.offset, slot.size));

            match &node.opcode {
                Opcode::Constant(val) => match val {
                    TensorValue::Float(v) => {
                        instructions.push(Instruction::Fill {
                            dst: output_slice,
                            value: *v,
                        });
                    }
                    TensorValue::Int(v) => {
                        instructions.push(Instruction::Fill {
                            dst: output_slice,
                            value: *v as f32,
                        });
                    }
                    TensorValue::Data { bytes, .. } => {
                        instructions.push(Instruction::WriteConst {
                            dst: output_slice,
                            data: bytes.clone(),
                        });
                    }
                },
                Opcode::MatMul => {
                    // Detect quantized dtypes from input nodes to select the right kernel
                    let input_dtypes: Vec<_> = node
                        .inputs
                        .iter()
                        .filter_map(|&input_id| graph.get_node(input_id))
                        .map(|n| n.output_type.dtype.clone())
                        .collect();
                    let is_quantized = input_dtypes
                        .iter()
                        .any(|d| matches!(d, IrDType::U4 { .. } | IrDType::U8 { .. }));

                    let fused_type = node.attrs.get("fused_op").map(|s| s.as_str());
                    // Determine fusion params for the unified "matmul" kernel
                    let has_bias = input_slices.len() >= 3;
                    let activation_type = match fused_type {
                        Some("MatMulAddRelu") | Some("OpRelu") => 1, // ReLU
                        Some("MatMulAddGelu") | Some("OpGelu") => 2,  // GELU
                        Some("MatMulAddSilu") | Some("OpSilu") => 3,  // SiLU
                        _ => 0,                                        // No activation
                    };
                    let kernel_name = match (fused_type, is_quantized) {
                        // Non-quantized fused patterns: use specialized kernels for now
                        // (they have mature, tested implementations)
                        (Some("MatMulAddRelu"), false) => "fused_matmul_add_relu",
                        (Some("MatMulAddGelu"), false) => "fused_matmul_add_gelu",
                        (Some("MatMulAddSilu"), false) => "fused_matmul_add_silu",
                        (Some("OpRelu"), false) => "matmul_relu",
                        (Some("OpGelu"), false) => "matmul_gelu",
                        (Some("OpSilu"), false) => "matmul_silu",
                        // Non-quantized non-fused: use unified "matmul" kernel
                        (_, false) => "matmul",
                        // Quantized: keep specialized kernels
                        (_, true)
                            if input_dtypes.iter().any(|d| matches!(d, IrDType::I8))
                                && input_dtypes.iter().any(|d| matches!(d, IrDType::U4 { .. })) =>
                        {
                            "matmul_u4_i8"
                        }
                        (_, true)
                            if input_dtypes.iter().any(|d| matches!(d, IrDType::I8))
                                && input_dtypes.iter().any(|d| matches!(d, IrDType::U8 { .. }))
                                && !input_dtypes
                                    .iter()
                                    .any(|d| matches!(d, IrDType::U4 { .. })) =>
                        {
                            "matmul_u8_i8"
                        }
                        (_, true)
                            if input_dtypes.iter().any(|d| matches!(d, IrDType::U4 { .. })) =>
                        {
                            "matmul_u4"
                        }
                        (_, true) => "matmul_u8",
                    };
                    // Extract M, K, N from input shapes
                    let m = input_shapes
                        .first()
                        .and_then(|s| s.get(s.len().saturating_sub(2)).copied())
                        .unwrap_or(1) as usize;
                    let k = input_shapes
                        .first()
                        .and_then(|s| s.last().copied())
                        .unwrap_or(1) as usize;
                    let n = input_shapes
                        .get(1)
                        .and_then(|s| s.last().copied())
                        .unwrap_or(1) as usize;
                    // Capture symbolic dims for runtime resolution
                    let m_dim = input_shape_dims
                        .first()
                        .and_then(|s| s.get(s.len().saturating_sub(2)).cloned())
                        .unwrap_or(DimExpr::Known(m as u64));
                    let k_dim = input_shape_dims
                        .first()
                        .and_then(|s| s.last().cloned())
                        .unwrap_or(DimExpr::Known(k as u64));
                    let n_dim = input_shape_dims
                        .get(1)
                        .and_then(|s| s.last().cloned())
                        .unwrap_or(DimExpr::Known(n as u64));
                    // Extract weight metadata for quantized matmul kernels.
                    // For 2D weights, the quantized data is stored transposed
                    // ([N, K] instead of [K, N]) so the shape must match.
                    let weight_meta = if is_quantized {
                        node.inputs.get(1).and_then(|&w_id| {
                            graph.get_node(w_id).map(|wn| {
                                let (bit_width, scales, zero_points) = match &wn.output_type.dtype {
                                    IrDType::U4 {
                                        scales,
                                        zero_points,
                                    } => (4usize, scales.clone(), zero_points.clone()),
                                    IrDType::U8 {
                                        scales,
                                        zero_points,
                                    } => (8usize, scales.clone(), zero_points.clone()),
                                    _ => (0usize, vec![], vec![]),
                                };
                                let mut w_shape: Vec<usize> = wn
                                    .output_type
                                    .shape
                                    .iter()
                                    .map(|d| d.evaluate().unwrap_or(symbol_max) as usize)
                                    .collect();
                                // The quantization pass transposes 2D weights from
                                // [K, N] to [N, K] for gemm_packed_batched convention.
                                if w_shape.len() == 2 {
                                    w_shape.reverse();
                                }
                                crate::backend::QuantizedWeightMeta {
                                    bit_width,
                                    scales,
                                    zero_points,
                                    shape: w_shape,
                                }
                            })
                        })
                    } else {
                        None
                    };
                    // For non-quantized matmul, pass fusion params (has_bias, activation_type)
                    // Quantized kernels keep the original [M, K, N] params
                    let (params, param_dims) = if kernel_name == "matmul" {
                        (
                            vec![m, k, n, has_bias as usize, activation_type],
                            Some(vec![
                                m_dim,
                                k_dim,
                                n_dim,
                                DimExpr::Known(has_bias as u64),
                                DimExpr::Known(activation_type as u64),
                            ]),
                        )
                    } else {
                        (
                            vec![m, k, n],
                            Some(vec![m_dim, k_dim, n_dim]),
                        )
                    };
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel_name.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims,
                        weight_meta,
                    });
                }
                Opcode::Add
                | Opcode::Sub
                | Opcode::Mul
                | Opcode::Div
                | Opcode::Maximum
                | Opcode::Minimum => {
                    let mut kernel = match node.opcode {
                        Opcode::Add => "add_f32",
                        Opcode::Sub => "sub_f32",
                        Opcode::Mul => "mul_f32",
                        Opcode::Div => "div_f32",
                        Opcode::Maximum => "max_f32",
                        Opcode::Minimum => "min_f32",
                        _ => unreachable!(),
                    };
                    // Op+Activation fusion: if fused_op is set, use the fused kernel name
                    match node.attrs.get("fused_op").map(|s| s.as_str()) {
                        Some("OpRelu") => {
                            kernel = match node.opcode {
                                Opcode::Add => "add_relu_f32",
                                Opcode::Sub => "sub_relu_f32",
                                Opcode::Mul => "mul_relu_f32",
                                Opcode::Div => "div_relu_f32",
                                _ => unreachable!(),
                            };
                        }
                        Some("OpGelu") => {
                            kernel = match node.opcode {
                                Opcode::Add => "add_gelu_f32",
                                Opcode::Sub => "sub_gelu_f32",
                                Opcode::Mul => "mul_gelu_f32",
                                Opcode::Div => "div_gelu_f32",
                                _ => unreachable!(),
                            };
                        }
                        Some("OpSilu") => {
                            kernel = match node.opcode {
                                Opcode::Add => "add_silu_f32",
                                Opcode::Sub => "sub_silu_f32",
                                Opcode::Mul => "mul_silu_f32",
                                Opcode::Div => "div_silu_f32",
                                _ => unreachable!(),
                            };
                        }
                        _ => {}
                    }
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Relu
                | Opcode::Gelu
                | Opcode::Silu
                | Opcode::Sigmoid
                | Opcode::Tanh
                | Opcode::Exp
                | Opcode::Log
                | Opcode::Sqrt
                | Opcode::Neg
                | Opcode::Abs
                | Opcode::LeakyRelu
                | Opcode::Elu
                | Opcode::Softplus
                | Opcode::Hardswish
                | Opcode::Clamp
                | Opcode::Sign
                | Opcode::Round
                | Opcode::LogicalNot
                | Opcode::LogSoftmax
                | Opcode::Mish => {
                    let kernel = match node.opcode {
                        Opcode::Relu => "relu_f32",
                        Opcode::Gelu => "gelu_f32",
                        Opcode::Silu => "silu_f32",
                        Opcode::Sigmoid => "sigmoid_f32",
                        Opcode::Tanh => "tanh_f32",
                        Opcode::Exp => "exp_f32",
                        Opcode::Log => "log_f32",
                        Opcode::Sqrt => "sqrt_f32",
                        Opcode::Neg => "neg_f32",
                        Opcode::Abs => "abs_f32",
                        Opcode::LeakyRelu => "leaky_relu_f32",
                        Opcode::Elu => "elu_f32",
                        Opcode::Softplus => "softplus_f32",
                        Opcode::Hardswish => "hardswish_f32",
                        Opcode::Clamp => "clamp_f32",
                        Opcode::Sign => "sign_f32",
                        Opcode::Round => "round_f32",
                        Opcode::LogicalNot => "logical_not_f32",
                        Opcode::LogSoftmax => "log_softmax_f32",
                        Opcode::Mish => "mish_f32",
                        _ => unreachable!(),
                    };
                    let mut extra_params: Vec<usize> = Vec::new();
                    if let Opcode::LeakyRelu = node.opcode {
                        let slope: f32 = node
                            .attrs
                            .get("negative_slope")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.01);
                        extra_params.push(slope.to_bits() as usize);
                    }
                    if let Opcode::Clamp = node.opcode {
                        let min: f32 = node
                            .attrs
                            .get("min")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.0);
                        let max: f32 = node
                            .attrs
                            .get("max")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(1.0);
                        extra_params.push(min.to_bits() as usize);
                        extra_params.push(max.to_bits() as usize);
                    }
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: extra_params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Reshape | Opcode::Flatten | Opcode::Squeeze | Opcode::Unsqueeze => {
                    if let Some(&input_id) = node.inputs.first() {
                        if let (Some(in_slot), Some(out_slot)) = (
                            memory_plan.slots.get(&input_id),
                            memory_plan.slots.get(&node_id),
                        ) {
                            if in_slot.offset != out_slot.offset || in_slot.size != out_slot.size {
                                instructions.push(Instruction::MemCopy {
                                    dst: output_slice,
                                    src: BufferSlice::new(in_slot.offset, in_slot.size),
                                });
                            }
                        }
                    }
                }
                Opcode::Conv2d => {
                    let stride: usize = node
                        .attrs
                        .get("stride")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let padding: usize = node
                        .attrs
                        .get("padding")
                        .and_then(|p| p.parse().ok())
                        .unwrap_or(0);
                    let dilation: usize = node
                        .attrs
                        .get("dilation")
                        .and_then(|d| d.parse().ok())
                        .unwrap_or(1);
                    let groups: usize = node
                        .attrs
                        .get("groups")
                        .and_then(|g| g.parse().ok())
                        .unwrap_or(1);
                    // Detect quantized weights for packed conv2d dispatch
                    let weight_dtype = node
                        .inputs
                        .get(1)
                        .and_then(|&w_id| graph.get_node(w_id))
                        .map(|wn| wn.output_type.dtype.clone());
                    let is_quantized = weight_dtype
                        .as_ref()
                        .is_some_and(|d| matches!(d, IrDType::U4 { .. } | IrDType::U8 { .. }));
                    let (kernel_name, weight_meta) = if is_quantized {
                        let dtype = weight_dtype.as_ref().unwrap();
                        let mut kernel = if matches!(dtype, IrDType::U4 { .. }) {
                            "conv2d_u4".to_string()
                        } else {
                            "conv2d_u8".to_string()
                        };
                        let bit_width = if matches!(dtype, IrDType::U4 { .. }) {
                            4usize
                        } else {
                            8usize
                        };

                        // Detect I8 activations from QuantizeActivations and append suffix
                        let act_dtype = node
                            .inputs
                            .first()
                            .and_then(|&a_id| graph.get_node(a_id))
                            .map(|an| an.output_type.dtype.clone());
                        if act_dtype
                            .as_ref()
                            .is_some_and(|d| matches!(d, IrDType::I8))
                        {
                            kernel = format!("{}_i8", kernel);
                        }

                        let meta = node.inputs.get(1).and_then(|&w_id| {
                            graph.get_node(w_id).map(|wn| {
                                let (bw, scales, zero_points) = match &wn.output_type.dtype {
                                    IrDType::U4 {
                                        scales,
                                        zero_points,
                                    } => (4usize, scales.clone(), zero_points.clone()),
                                    IrDType::U8 {
                                        scales,
                                        zero_points,
                                    } => (8usize, scales.clone(), zero_points.clone()),
                                    _ => (bit_width, vec![], vec![]),
                                };
                                let w_shape: Vec<usize> = wn
                                    .output_type
                                    .shape
                                    .iter()
                                    .map(|d| d.evaluate().unwrap_or(symbol_max) as usize)
                                    .collect();
                                crate::backend::QuantizedWeightMeta {
                                    bit_width: bw,
                                    scales,
                                    zero_points,
                                    shape: w_shape,
                                }
                            })
                        });
                        (kernel.to_string(), meta)
                    } else {
                        let fused_type = node.attrs.get("fused_op").map(|s| s.as_str());
                        let base_name = match fused_type {
                            Some("OpRelu") => "conv2d_relu",
                            Some("OpGelu") => "conv2d_gelu",
                            Some("OpSilu") => "conv2d_silu",
                            _ => "conv2d",
                        };
                        (base_name.to_string(), None)
                    };
                    // Extract spatial dims from input shapes to avoid
                    // ambiguous dim inference at dispatch time.
                    let input_c = input_shapes
                        .first()
                        .and_then(|s| s.get(1).copied())
                        .unwrap_or(1) as usize;
                    let input_h = input_shapes
                        .first()
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let input_w = input_shapes
                        .first()
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    let kernel_h = input_shapes
                        .get(1)
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let kernel_w = input_shapes
                        .get(1)
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name,
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        // params: [stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w]
                        params: vec![
                            stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h,
                            kernel_w,
                        ],
                        param_dims: None,
                        weight_meta,
                    });
                }
                Opcode::BatchNorm | Opcode::LayerNorm => {
                    let eps = node
                        .attrs
                        .get("eps")
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(1e-5);
                    let is_batch_norm = if matches!(node.opcode, Opcode::BatchNorm) {
                        1
                    } else {
                        0
                    };
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "norm_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![eps.to_bits() as usize, is_batch_norm],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Softmax => {
                    // Read and normalize the axis attribute.
                    let axis: i64 = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0);
                    let rank = input_shapes.first().map(|s| s.len()).unwrap_or(1);
                    let normalized_axis = if axis < 0 {
                        (rank as i64 + axis) as usize
                    } else {
                        axis as usize
                    };
                    // Capture the axis-dimension size for the dispatch handler,
                    // which needs to know how many elements comprise a single
                    // softmax "row" (all elements along the reduction axis).
                    let axis_dim = input_shapes
                        .first()
                        .and_then(|s| s.get(normalized_axis).copied())
                        .unwrap_or(1) as usize;
                    let axis_dim_dim = input_shape_dims
                        .first()
                        .and_then(|s| s.get(normalized_axis).cloned())
                        .unwrap_or(DimExpr::Known(1));
                    // Compute stride: product of dims after the axis.
                    // This is needed because softmax rows are strided for
                    // non-last dimensions (e.g. axis=2 on [N,C,H,W]).
                    let stride = input_shapes
                        .first()
                        .map(|s| {
                            s[normalized_axis + 1..]
                                .iter()
                                .copied()
                                .map(|x| x as usize)
                                .product::<usize>()
                                .max(1)
                        })
                        .unwrap_or(1);

                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "softmax".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![axis_dim, stride],
                        param_dims: Some(vec![axis_dim_dim, DimExpr::Known(stride as u64)]),
                        weight_meta: None,
                    });
                }
                Opcode::BiasAdd => {
                    // Channel stride = product of spatial dims (H*W for 4D NCHW).
                    // This is needed for correct NCHW channel-wise broadcast:
                    //   bias_idx = (flat_idx / channel_stride) % num_channels
                    let channel_stride = input_shapes
                        .first()
                        .map(|s| s.iter().skip(2).product::<u64>())
                        .unwrap_or(1) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "biasadd".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![channel_stride],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Concat => {
                    let axis: usize = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0);
                    // Compute inner_stride (product of dims after axis) and
                    // outer_count (product of dims before axis) from output shape.
                    let output_shape: Vec<u64> = node
                        .output_type
                        .shape
                        .iter()
                        .map(|d| d.evaluate().unwrap_or(symbol_max))
                        .collect();
                    let rank = output_shape.len();
                    let inner_stride: u64 = if axis + 1 < rank {
                        output_shape[axis + 1..].iter().product()
                    } else {
                        1
                    };
                    let outer_count: u64 = if axis > 0 {
                        output_shape[..axis].iter().product()
                    } else {
                        1
                    };
                    let _input_ids_str = node
                        .inputs
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    #[cfg(feature = "debug_canary")]
                    eprintln!(
                        "[FNN_DBG_CONCAT_COMPILE] nid={} op=Concat inputs=[{}] axis={} inner_stride={} outer_count={} output_shape={:?}",
                        node_id,
                        _input_ids_str,
                        axis,
                        inner_stride,
                        outer_count,
                        output_shape,
                    );
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "concat".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![axis, inner_stride as usize, outer_count as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::MaxPool | Opcode::AvgPool => {
                    let kernel_size: usize = node
                        .attrs
                        .get("kernel_size")
                        .and_then(|k| k.parse().ok())
                        .unwrap_or(2);
                    let stride: usize = node
                        .attrs
                        .get("stride")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(2);
                    let padding: usize = node
                        .attrs
                        .get("padding")
                        .and_then(|p| p.parse().ok())
                        .unwrap_or(0);
                    let is_max = if matches!(node.opcode, Opcode::MaxPool) {
                        1
                    } else {
                        0
                    };
                    let secondary_output_slice = if is_max == 1 {
                        memory_plan
                            .secondary_slots
                            .get(&(node_id, 1))
                            .map(|slot| BufferSlice::new(slot.offset, slot.size))
                    } else {
                        None
                    };
                    // Pass explicit input dims so pool_f32 doesn't need to
                    // infer N,C,H,W from flat element counts (ambiguous).
                    let input_n = input_shapes
                        .first()
                        .and_then(|s| s.get(0).copied())
                        .unwrap_or(1) as usize;
                    let input_c = input_shapes
                        .first()
                        .and_then(|s| s.get(1).copied())
                        .unwrap_or(1) as usize;
                    let input_h = input_shapes
                        .first()
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let input_w = input_shapes
                        .first()
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "pool_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice,
                        // params: [kernel_size, stride, padding, is_max, N, C, H, W]
                        params: vec![
                            kernel_size,
                            stride,
                            padding,
                            is_max,
                            input_n,
                            input_c,
                            input_h,
                            input_w,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Pad => {
                    let pads_str = node.attrs.get("pads").cloned().unwrap_or_default();
                    let pads: Vec<usize> = pads_str
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "pad_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: pads,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Gather => {
                    let axis: usize = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "gather".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![axis],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Slice => {
                    let dim: usize = node
                        .attrs
                        .get("dim")
                        .and_then(|d| d.parse().ok())
                        .unwrap_or(0);
                    let start: usize = node
                        .attrs
                        .get("start")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    let end: usize = node
                        .attrs
                        .get("end")
                        .and_then(|e| e.parse().ok())
                        .unwrap_or(1);
                    // Compute the stride (product of dim sizes after `dim`)
                    // so the kernel can correctly compute the offset for non-batch dims.
                    let stride = input_shapes
                        .first()
                        .filter(|s| dim < s.len())
                        .map(|s| s[dim + 1..].iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "slice_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![dim, start, end, stride],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ScatterNd => {
                    let data_shape: Vec<u64> = input_shapes.first().cloned().unwrap_or_default();
                    let indices_shape: Vec<u64> = input_shapes.get(1).cloned().unwrap_or_default();
                    let index_depth = match indices_shape.len() {
                        0 | 1 => 1,
                        _ => indices_shape[indices_shape.len() - 1] as usize,
                    };
                    let mut params: Vec<usize> = vec![index_depth];
                    params.extend(data_shape.iter().map(|&d| d as usize));
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "scatter_nd".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ReduceSum | Opcode::ReduceMean | Opcode::ReduceMax => {
                    // Group size = product of dims being reduced over.
                    // For a single-axis reduce this is just input_shape[axis],
                    // which is typically Known (e.g. reduce over dim 1 of [N,4]
                    // has group_size=Known(4)).
                    let axis_i64: i64 = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0);
                    let rank = input_shapes.first().map(|s| s.len() as i64).unwrap_or(0);
                    let axis = if axis_i64 < 0 {
                        (rank + axis_i64).max(0)
                    } else {
                        axis_i64.min((rank - 1).max(0))
                    } as usize;
                    let group_size_dim = input_shape_dims
                        .first()
                        .and_then(|s| s.get(axis).cloned())
                        .unwrap_or(DimExpr::Known(1));
                    let (is_mean, is_max) = match node.opcode {
                        Opcode::ReduceMean => (1, 0),
                        Opcode::ReduceMax => (0, 1),
                        _ => (0, 0), // ReduceSum
                    };
                    let group_size = group_size_dim.evaluate().unwrap_or_else(|| {
                        // Symbolic dim — use SYMBOL_DIM_MAX as compile-time
                        // estimate; runtime resolves via param_dims.
                        crate::ir::node::SYMBOL_DIM_MAX.load(Ordering::Relaxed)
                    }) as usize;

                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "reduce_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![group_size, is_mean, is_max],
                        // Pass the group_size as a symbolic DimExpr so dispatch
                        // can re-evaluate it when shape_env is available (e.g.
                        // reduce over symbolic batch dim N).
                        param_dims: Some(vec![group_size_dim]),
                        weight_meta: None,
                    });
                }
                Opcode::Transpose => {
                    let input_shape: Vec<usize> = input_shapes
                        .first()
                        .map(|s| s.iter().map(|&d| d as usize).collect())
                        .unwrap_or_default();
                    let rank = input_shape.len();

                    // Read perm from node attrs (e.g. "0,3,1,2")
                    let perm_str: String = node.attrs.get("perm").cloned().unwrap_or_default();
                    let perm: Vec<usize> = if perm_str.is_empty() {
                        (0..rank).rev().collect()
                    } else {
                        perm_str.split(',').filter_map(|s| s.parse().ok()).collect()
                    };

                    // Simple 2D transpose [1,0] on a rank-2 tensor → use fast kernel
                    if rank == 2 && perm.len() >= 2 && perm[0] == 1 && perm[1] == 0 {
                        let m = input_shape[0];
                        let n = input_shape[1];
                        let m_dim = input_shape_dims
                            .first()
                            .and_then(|s| s.first().cloned())
                            .unwrap_or(DimExpr::Known(m as u64));
                        let n_dim = input_shape_dims
                            .first()
                            .and_then(|s| s.get(1).cloned())
                            .unwrap_or(DimExpr::Known(n as u64));
                        instructions.push(Instruction::CallKernel {
                            node_id: Some(node_id),
                            kernel_name: "transpose_f32".to_string(),
                            input_slices,
                            output_slice,
                            secondary_output_slice: None,
                            params: vec![m, n],
                            param_dims: Some(vec![m_dim, n_dim]),
                            weight_meta: None,
                        });
                    } else {
                        // N-D permute transpose: params = [rank, d0..dN, p0..pN]
                        let mut nd_params: Vec<usize> = Vec::with_capacity(1 + 2 * rank);
                        nd_params.push(rank);
                        nd_params.extend_from_slice(&input_shape);
                        for i in 0..rank {
                            nd_params.push(perm.get(i).copied().unwrap_or(i));
                        }
                        instructions.push(Instruction::CallKernel {
                            node_id: Some(node_id),
                            kernel_name: "transpose_perm_f32".to_string(),
                            input_slices,
                            output_slice,
                            secondary_output_slice: None,
                            params: nd_params,
                            param_dims: None,
                            weight_meta: None,
                        });
                    }
                }
                Opcode::Conv1d => {
                    let stride: usize = node
                        .attrs
                        .get("stride")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let padding: usize = node
                        .attrs
                        .get("padding")
                        .and_then(|p| p.parse().ok())
                        .unwrap_or(0);
                    let input_c = input_shapes
                        .first()
                        .and_then(|s| s.get(1).copied())
                        .unwrap_or(1) as usize;
                    let input_w = input_shapes
                        .first()
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let kernel_w = input_shapes
                        .get(1)
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "conv1d".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![stride, padding, input_c, input_w, kernel_w],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Conv3d => {
                    let stride: usize = node
                        .attrs
                        .get("stride")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let padding: usize = node
                        .attrs
                        .get("padding")
                        .and_then(|p| p.parse().ok())
                        .unwrap_or(0);
                    let dilation: usize = node
                        .attrs
                        .get("dilation")
                        .and_then(|d| d.parse().ok())
                        .unwrap_or(1);
                    let input_c = input_shapes
                        .first()
                        .and_then(|s| s.get(1).copied())
                        .unwrap_or(1) as usize;
                    let input_d = input_shapes
                        .first()
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let input_h = input_shapes
                        .first()
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    let input_w = input_shapes
                        .first()
                        .and_then(|s| s.get(4).copied())
                        .unwrap_or(0) as usize;
                    let kernel_d = input_shapes
                        .get(1)
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let kernel_h = input_shapes
                        .get(1)
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    let kernel_w = input_shapes
                        .get(1)
                        .and_then(|s| s.get(4).copied())
                        .unwrap_or(0) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "conv3d".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![
                            stride, padding, dilation, input_c, input_d, input_h, input_w,
                            kernel_d, kernel_h, kernel_w,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ConvTranspose2d => {
                    let stride: usize = node
                        .attrs
                        .get("stride")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let padding: usize = node
                        .attrs
                        .get("padding")
                        .and_then(|p| p.parse().ok())
                        .unwrap_or(0);
                    let input_c = input_shapes
                        .first()
                        .and_then(|s| s.get(1).copied())
                        .unwrap_or(1) as usize;
                    let input_h = input_shapes
                        .first()
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let input_w = input_shapes
                        .first()
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    let kernel_h = input_shapes
                        .get(1)
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let kernel_w = input_shapes
                        .get(1)
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "conv_transpose2d".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![
                            stride, padding, input_c, input_h, input_w, kernel_h, kernel_w,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Prelu => {
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "prelu".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::RMSNorm => {
                    let eps = node
                        .attrs
                        .get("eps")
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(1e-5);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "rms_norm".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![eps.to_bits() as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Embedding => {
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "embedding".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Pow => {
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "pow_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::GtScalar => {
                    instructions.push(scalar_kernel_instruction(
                        node_id,
                        "gt_scalar_f32",
                        input_slices,
                        output_slice,
                    ));
                }
                Opcode::LtScalar => {
                    instructions.push(scalar_kernel_instruction(
                        node_id,
                        "lt_scalar_f32",
                        input_slices,
                        output_slice,
                    ));
                }
                Opcode::EqScalar => {
                    instructions.push(scalar_kernel_instruction(
                        node_id,
                        "eq_scalar_f32",
                        input_slices,
                        output_slice,
                    ));
                }
                Opcode::AddScalar => {
                    instructions.push(scalar_kernel_instruction(
                        node_id,
                        "add_scalar_f32",
                        input_slices,
                        output_slice,
                    ));
                }
                Opcode::MulScalar => {
                    instructions.push(scalar_kernel_instruction(
                        node_id,
                        "mul_scalar_f32",
                        input_slices,
                        output_slice,
                    ));
                }
                Opcode::DivScalar => {
                    instructions.push(scalar_kernel_instruction(
                        node_id,
                        "div_scalar_f32",
                        input_slices,
                        output_slice,
                    ));
                }
                // Input nodes have no producer instruction — data is written
                // by the executor before dispatch.
                Opcode::Input => {
                    // No instruction needed.
                }
                Opcode::ArgMax => {
                    let axis: i64 = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(-1);
                    let rank = input_shapes.first().map(|s| s.len() as i64).unwrap_or(0);
                    let normalized = if axis < 0 {
                        (rank + axis).max(0)
                    } else {
                        axis.min((rank - 1).max(0))
                    };
                    let (dim_size, inner) = input_shapes
                        .first()
                        .and_then(|s| {
                            let idx = normalized as usize;
                            if idx < s.len() {
                                let ds = s[idx] as usize;
                                let inn: usize =
                                    s[idx + 1..].iter().copied().map(|x| x as usize).product();
                                Some((ds, inn))
                            } else {
                                None
                            }
                        })
                        .unwrap_or((0, 1));
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "argmax".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![normalized as usize, dim_size, inner],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::UpsampleNearest2d | Opcode::UpsampleBilinear2d => {
                    let scale_h: usize = node
                        .attrs
                        .get("scale_h")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(2);
                    let scale_w: usize = node
                        .attrs
                        .get("scale_w")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(2);
                    // Pass input spatial dims so the kernel doesn't need to guess H,W
                    // from flat buffer size (which is ambiguous for NCHW layouts).
                    let h_in = input_shapes
                        .first()
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(1) as usize;
                    let w_in = input_shapes
                        .first()
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(1) as usize;
                    let kernel_name = match node.opcode {
                        Opcode::UpsampleNearest2d => "upsample_nearest2d",
                        _ => "upsample_bilinear2d",
                    };
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel_name.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![scale_h, scale_w, h_in, w_in],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::AdaptiveAvgPool2d => {
                    let out_h: usize = node
                        .attrs
                        .get("output_h")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let out_w: usize = node
                        .attrs
                        .get("output_w")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "adaptive_avg_pool2d".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![out_h, out_w],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Repeat => {
                    let repeats_str = node.attrs.get("repeats").cloned().unwrap_or_default();
                    let repeats: Vec<usize> = repeats_str
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "repeat".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: repeats,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::CumSum => {
                    let dim: usize = node
                        .attrs
                        .get("dim")
                        .and_then(|d| d.parse().ok())
                        .unwrap_or(0);
                    let exclusive: usize = node
                        .attrs
                        .get("exclusive")
                        .and_then(|e| e.parse().ok())
                        .unwrap_or(0);
                    let rev: usize = node
                        .attrs
                        .get("reverse")
                        .and_then(|r| r.parse().ok())
                        .unwrap_or(0);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "cumsum".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![dim, exclusive, rev],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Erf => {
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "erf_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Flip => {
                    let dims_str = node.attrs.get("dims").cloned().unwrap_or_default();
                    let dims: Vec<usize> = dims_str
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    let input_shape: Vec<u64> = input_shapes.first().cloned().unwrap_or_default();
                    let mut params = vec![dims.len()];
                    params.extend_from_slice(&dims);
                    params.extend(input_shape.iter().map(|&s| s as usize));
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "flip".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Where => {
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "where_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::TopK => {
                    let k: usize = node
                        .attrs
                        .get("k")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let axis: i64 = node
                        .attrs
                        .get("axis")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(-1);
                    let rank = input_shapes.first().map(|s| s.len() as i64).unwrap_or(0);
                    let normalized = if axis < 0 {
                        (rank + axis).max(0)
                    } else {
                        axis.min((rank - 1).max(0))
                    };
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "topk_fused".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice,
                        params: vec![k, normalized as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                // ── Optimizer ops ──────────────────────────────────
                Opcode::SgdUpdate => {
                    let lr: f32 = node
                        .attrs
                        .get("lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.01);
                    let wd: f32 = node
                        .attrs
                        .get("weight_decay")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "sgd_update_f32".to_string(),
                        input_slices, // [weight, grad] — weight must be same slot as output
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![lr.to_bits() as usize, wd.to_bits() as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::AdamUpdate => {
                    let lr: f32 = node
                        .attrs
                        .get("lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.001);
                    let beta1: f32 = node
                        .attrs
                        .get("beta1")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.9);
                    let beta2: f32 = node
                        .attrs
                        .get("beta2")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.999);
                    let eps: f32 = node
                        .attrs
                        .get("eps")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-8);
                    let t: u64 = node
                        .attrs
                        .get("t")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    // Detect F16 state tensors (m and v at inputs[2] and inputs[3]).
                    let has_f16_state = node.inputs.len() >= 4
                        && graph
                            .get_node(node.inputs[2])
                            .map(|n| n.output_type.dtype == IrDType::F16)
                            .unwrap_or(false)
                        && graph
                            .get_node(node.inputs[3])
                            .map(|n| n.output_type.dtype == IrDType::F16)
                            .unwrap_or(false);
                    let kernel_name = if has_f16_state {
                        "adam_update_f16_state"
                    } else {
                        "adam_update_f32"
                    };
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel_name.to_string(),
                        input_slices, // [weight, grad, m, v]
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![
                            lr.to_bits() as usize,
                            beta1.to_bits() as usize,
                            beta2.to_bits() as usize,
                            eps.to_bits() as usize,
                            t as usize,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::AdamWUpdate => {
                    let lr: f32 = node
                        .attrs
                        .get("lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.001);
                    let beta1: f32 = node
                        .attrs
                        .get("beta1")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.9);
                    let beta2: f32 = node
                        .attrs
                        .get("beta2")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.999);
                    let eps: f32 = node
                        .attrs
                        .get("eps")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-8);
                    let wd: f32 = node
                        .attrs
                        .get("weight_decay")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.01);
                    let has_f16_state = node.inputs.len() >= 4
                        && graph
                            .get_node(node.inputs[2])
                            .map(|n| n.output_type.dtype == IrDType::F16)
                            .unwrap_or(false)
                        && graph
                            .get_node(node.inputs[3])
                            .map(|n| n.output_type.dtype == IrDType::F16)
                            .unwrap_or(false);
                    let kernel_name = if has_f16_state {
                        "adamw_update_f16_state"
                    } else {
                        "adamw_update_f32"
                    };
                    let t_attr: u64 = node
                        .attrs
                        .get("t")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let has_t_input = node.inputs.len() >= 5;
                    let mut adamw_params = vec![
                        lr.to_bits() as usize,
                        beta1.to_bits() as usize,
                        beta2.to_bits() as usize,
                        eps.to_bits() as usize,
                    ];
                    if has_t_input {
                        // New path: t is a runtime tensor (5th input slice)
                        adamw_params.push(wd.to_bits() as usize);
                    } else {
                        // Old path: t is stored in params
                        adamw_params.push(t_attr as usize);
                        adamw_params.push(wd.to_bits() as usize);
                    }
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel_name.to_string(),
                        input_slices, // [weight, grad, m, v, t] (t only for training pass path)
                        output_slice,
                        secondary_output_slice: None,
                        params: adamw_params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::MuonUpdate => {
                    let lr: f32 = node
                        .attrs
                        .get("lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.01);
                    let beta1: f32 = node
                        .attrs
                        .get("beta1")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.9);
                    let weight_decay: f32 = node
                        .attrs
                        .get("weight_decay")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "muon_update_f32".to_string(),
                        input_slices, // [weight, grad, m]
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![
                            lr.to_bits() as usize,
                            beta1.to_bits() as usize,
                            weight_decay.to_bits() as usize,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::LionUpdate => {
                    let lr: f32 = node
                        .attrs
                        .get("lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.001);
                    let beta1: f32 = node
                        .attrs
                        .get("beta1")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.9);
                    let beta2: f32 = node
                        .attrs
                        .get("beta2")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.999);
                    let wd: f32 = node
                        .attrs
                        .get("weight_decay")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "lion_update_f32".to_string(),
                        input_slices, // [weight, grad, m]
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![
                            lr.to_bits() as usize,
                            beta1.to_bits() as usize,
                            beta2.to_bits() as usize,
                            wd.to_bits() as usize,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::RmspropUpdate => {
                    let lr: f32 = node
                        .attrs
                        .get("lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.001);
                    let beta: f32 = node
                        .attrs
                        .get("beta")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.99);
                    let eps: f32 = node
                        .attrs
                        .get("eps")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-8);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "rmsprop_update_f32".to_string(),
                        input_slices, // [weight, grad, v]
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![
                            lr.to_bits() as usize,
                            beta.to_bits() as usize,
                            eps.to_bits() as usize,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Shape => {
                    // Write the shape of the input tensor as F32 values.
                    // The arena stores all data as f32 (4 bytes/element), so we
                    // write f32-le bytes.  Downstream ops (Gather, Concat, etc.)
                    // read from the arena as f32 slices and get correct values.
                    // Resolve input shape at compile time (known dims directly,
                    // symbolic dims use SYMBOL_DIM_MAX — they'll be resolved
                    // at dispatch by param_dims).
                    use std::io::Write;
                    let in_shape = input_shapes.first().cloned().unwrap_or_default();
                    let mut shape_bytes = Vec::with_capacity(in_shape.len() * 4);
                    for &d in &in_shape {
                        shape_bytes.write_all(&(d as f32).to_le_bytes()).unwrap();
                    }
                    instructions.push(Instruction::WriteConst {
                        dst: output_slice,
                        data: shape_bytes,
                    });
                }
                Opcode::Cast => {
                    // Cast: same-shape, potentially different dtype.
                    // For same-byte-size casts, MemCopy is sufficient.
                    // For different byte sizes, do element conversion.
                    let in_type = node
                        .inputs
                        .first()
                        .and_then(|id| graph.get_node(*id))
                        .map(|n| n.output_type.dtype.clone());
                    let out_type = node.output_type.dtype.clone();
                    let in_byte_size = in_type.as_ref().map(|d| d.byte_size()).unwrap_or(4);
                    let out_byte_size = out_type.byte_size();
                    if in_byte_size == out_byte_size {
                        // Same byte size: just copy.
                        if let Some(&input_id) = node.inputs.first() {
                            if let Some(in_slot) = memory_plan.slots.get(&input_id) {
                                instructions.push(Instruction::MemCopy {
                                    dst: output_slice,
                                    src: BufferSlice::new(in_slot.offset, in_slot.size),
                                });
                            }
                        }
                    } else {
                        // Different byte size: use a kernel call.
                        let in_slot = node.inputs.first().and_then(|id| memory_plan.slots.get(id));
                        let input_slices = in_slot
                            .map(|s| vec![BufferSlice::new(s.offset, s.size)])
                            .unwrap_or_default();
                        instructions.push(Instruction::CallKernel {
                            node_id: Some(node_id),
                            kernel_name: "cast".to_string(),
                            input_slices,
                            output_slice,
                            secondary_output_slice: None,
                            params: vec![in_byte_size, out_byte_size],
                            param_dims: None,
                            weight_meta: None,
                        });
                    }
                }
                Opcode::Quantize => {
                    let bit_width: usize = node
                        .attrs
                        .get("bit_width")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(4);
                    let kernel_name = match bit_width {
                        4 => "quantize_f32_u4",
                        8 => "quantize_f32_u8",
                        _ => {
                            return Err(BackendError::Compilation(format!(
                                "Quantize: unsupported bit_width={bit_width}"
                            )))
                        }
                    };
                    // num_channels = dim 0 (rows), num_elems_per_channel = product of rest
                    let num_channels = input_shapes
                        .first()
                        .and_then(|s| s.first().copied())
                        .unwrap_or(1) as usize;
                    let numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    let num_elems_per_channel = if num_channels > 0 {
                        numel / num_channels
                    } else {
                        numel
                    };

                    // If the predecessor node already carries calibrated scales/zeros
                    // (e.g. from wrap_quantized_optimizer re-quant path), forward
                    // them through params so the kernel can skip the O(N×K) scan.
                    let (cached_scales, cached_zeros) = node
                        .inputs
                        .first()
                        .and_then(|&input_id| graph.get_node(input_id))
                        .map(|n| match &n.output_type.dtype {
                            IrDType::U4 { scales, zero_points, .. }
                            | IrDType::U8 { scales, zero_points, .. } => {
                                (scales.clone(), zero_points.clone())
                            }
                            _ => (vec![], vec![]),
                        })
                        .unwrap_or_default();

                    let mut params = vec![num_channels, num_elems_per_channel, numel];
                    if !cached_scales.is_empty() && cached_scales.len() == num_channels {
                        params.push(1); // flag: cached
                        for &s in &cached_scales {
                            params.push(s.to_bits() as usize);
                        }
                        for &zp in &cached_zeros {
                            params.push(zp.to_bits() as usize);
                        }
                    }

                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel_name.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params, // includes cached scales/zeros when available
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Dequantize => {
                    let numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    // Check if the input has per-channel scale metadata (from
                    // quantized Constants or Quantize ops). These are passed as
                    // additional params so the dequantize kernel can reconstruct
                    // f32 values without relying on an inline header.
                    let (scales, zero_points) = node
                        .inputs
                        .first()
                        .and_then(|&input_id| graph.get_node(input_id))
                        .map(|n| match &n.output_type.dtype {
                            IrDType::U4 {
                                scales,
                                zero_points,
                            } => (scales.clone(), zero_points.clone()),
                            IrDType::U8 {
                                scales,
                                zero_points,
                            } => (scales.clone(), zero_points.clone()),
                            _ => (vec![], vec![]),
                        })
                        .unwrap_or_default();
                    let has_metadata = !scales.is_empty() && !zero_points.is_empty();
                    let format_flag: usize = if has_metadata { 1 } else { 0 }; // 0=header, 1=metadata
                                                                               // Flatten scales and zero_points into params (f32 bits as usize)
                    let mut params = vec![numel, format_flag];
                    let num_channels = scales.len();
                    params.push(num_channels);
                    for &s in &scales {
                        params.push(s.to_bits() as usize);
                    }
                    for &zp in &zero_points {
                        params.push(zp.to_bits() as usize);
                    }
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "dequantize_kernel".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ToF16 => {
                    let numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "to_f16".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![numel],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ToF32 => {
                    let numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "to_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![numel],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::QuantizeActivations => {
                    let numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    // Output buffer: [scale(f32)][zp(f32)][i8_data(numel bytes)]
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "quantize_activations".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![numel],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::DequantizeActivations => {
                    let numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "dequantize_activations".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![numel],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::FusedResidualAddNorm => {
                    let eps: f32 = node
                        .attrs
                        .get("eps")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-5);
                    let norm_type = node
                        .attrs
                        .get("norm_type")
                        .map(|s| s.as_str())
                        .unwrap_or("layer_norm");
                    let kernel_name = format!("fused_residual_add_{}", norm_type);

                    let output_numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    let row_size = node
                        .attrs
                        .get("normalized_ndims")
                        .and_then(|s| s.parse::<usize>().ok())
                        .and_then(|ndims| {
                            input_shapes.first().map(|shape| {
                                let start = shape.len().saturating_sub(ndims);
                                shape[start..]
                                    .iter()
                                    .copied()
                                    .map(|d| d as usize)
                                    .product::<usize>()
                                    .max(1)
                            })
                        })
                        .or_else(|| {
                            input_shapes
                                .get(2)
                                .map(|s| s.iter().product::<u64>() as usize)
                        })
                        .unwrap_or(output_numel.max(1));
                    let row_size_dim = node
                        .attrs
                        .get("normalized_ndims")
                        .and_then(|s| s.parse::<usize>().ok())
                        .and_then(|ndims| {
                            input_shape_dims.first().map(|shape| {
                                let start = shape.len().saturating_sub(ndims);
                                shape[start..]
                                    .iter()
                                    .cloned()
                                    .fold(DimExpr::Known(1), |acc, dim| acc.mul(&dim))
                            })
                        })
                        .unwrap_or(DimExpr::Known(row_size as u64));
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name,
                        input_slices, // [residual, main_output, weight, optional_bias]
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![eps.to_bits() as usize, row_size],
                        param_dims: Some(vec![row_size_dim]),
                        weight_meta: None,
                    });
                }
                Opcode::GradientScale => {
                    let scale: f32 = node
                        .attrs
                        .get("scale")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1.0);
                    let numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    if !input_slices.is_empty() {
                        instructions.push(Instruction::CallKernel {
                            node_id: Some(node_id),
                            kernel_name: "gradient_scale".to_string(),
                            input_slices,
                            output_slice,
                            secondary_output_slice: None,
                            params: vec![numel, scale.to_bits() as usize],
                            param_dims: None,
                            weight_meta: None,
                        });
                    }
                }
                Opcode::Expand => {
                    // Expand broadcasts input[0] to the shape specified by input[1].
                    // Resolve input and output shapes at compile time and pack them
                    // into params: [max_rank, in_d0, in_d1, ..., out_d0, out_d1, ...].
                    // The kernel uses these to compute broadcast strides.
                    let data_shape_dims = input_shape_dims.first().cloned().unwrap_or_default();
                    let out_shape_dims = node.output_type.shape.clone();
                    let out_rank = out_shape_dims.len();
                    let data_rank = data_shape_dims.len();
                    let max_rank = out_rank.max(data_rank);

                    // Resolve to concrete values (Known dims, safe in YOLO pipeline)
                    let resolve = |d: &DimExpr| d.evaluate().unwrap_or(1) as usize;

                    // Build params: [max_rank, padded_in_dims..., padded_out_dims...]
                    let mut params = vec![max_rank];
                    for i in 0..max_rank {
                        if i < max_rank - data_rank {
                            params.push(1);
                        } else {
                            params.push(resolve(&data_shape_dims[i - (max_rank - data_rank)]));
                        }
                    }
                    for i in 0..max_rank {
                        if i < max_rank - out_rank {
                            params.push(1);
                        } else {
                            params.push(resolve(&out_shape_dims[i - (max_rank - out_rank)]));
                        }
                    }

                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "expand_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Tile => {
                    // Tile copies first input (no broadcast yet).
                    if let Some(&input_id) = node.inputs.first() {
                        if let Some(in_slot) = memory_plan.slots.get(&input_id) {
                            instructions.push(Instruction::MemCopy {
                                dst: output_slice,
                                src: BufferSlice::new(in_slot.offset, in_slot.size),
                            });
                        }
                    }
                }
                Opcode::Range => {
                    // Range(start, limit, step) — produce 1D F32 tensor.
                    // All 3 inputs are scalars (4 bytes each).
                    let input_slices: Vec<BufferSlice> = node
                        .inputs
                        .iter()
                        .filter_map(|id| memory_plan.slots.get(id))
                        .map(|slot| BufferSlice::new(slot.offset, slot.size))
                        .collect();
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "range_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                #[allow(unreachable_patterns)]
                _ => {
                    if let Some(&input_id) = node.inputs.first() {
                        if let Some(in_slot) = memory_plan.slots.get(&input_id) {
                            instructions.push(Instruction::MemCopy {
                                dst: output_slice,
                                src: BufferSlice::new(in_slot.offset, in_slot.size),
                            });
                        } else {
                            // no input slot — user‑error / unexpected
                        }
                    }
                }
            }
            instruction_levels.push(level);
        }

        Ok(ExecutablePlan {
            instructions,
            arena_size: memory_plan.total_size,
            levels: instruction_levels,
        })
    }

    fn dispatch(
        &self,
        plan: &ExecutablePlan,
        arena: &CpuBuffer,
        shape_env: &ShapeEnv,
    ) -> Result<(), BackendError> {
        // ── Debug: collect MaxPool primary output ranges ──────────────
        //     (only active with `debug_canary` feature — expensive)
        // ANCHOR: debug-canary-start
        #[cfg(feature = "debug_canary")]
        let maxpool_ranges: Vec<(usize, usize)> = {
            let mut v = Vec::new();
            for instr in &plan.instructions {
                if let Instruction::CallKernel {
                    kernel_name,
                    params,
                    output_slice,
                    ..
                } = instr
                {
                    if kernel_name == "pool_f32" && params.len() >= 4 && params[3] == 1 {
                        v.push((output_slice.offset, output_slice.size));
                    }
                }
            }
            v
        };
        // Track: after each MaxPool kernel, snapshot its first f32 value;
        // after every other instruction, check if any snapshot changed.
        #[cfg(feature = "debug_canary")]
        let mut maxpool_snapshot: Vec<Option<f32>> = vec![None; maxpool_ranges.len()];
        #[cfg(feature = "debug_canary")]
        let mut maxpool_seen: Vec<bool> = vec![false; maxpool_ranges.len()];
        // ANCHOR: debug-canary-end

        for (_instr_idx, instr) in plan.instructions.iter().enumerate() {
            match instr {
                Instruction::CallKernel {
                    kernel_name,
                    input_slices,
                    output_slice,
                    secondary_output_slice,
                    params,
                    param_dims,
                    weight_meta,
                    ..
                } => {
                    let out_start = output_slice.offset;
                    let out_end = output_slice.offset + output_slice.size;

                    match kernel_name.as_str() {
                        "add_f32" => {
                            fused_binary_activation_dispatch(
                                "add_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a + b,
                                |x| x,
                            );
                        }
                        "sub_f32" => {
                            fused_binary_activation_dispatch(
                                "sub_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a - b,
                                |x| x,
                            );
                        }
                        "mul_f32" => {
                            fused_binary_activation_dispatch(
                                "mul_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a * b,
                                |x| x,
                            );
                        }
                        "div_f32" => {
                            fused_binary_activation_dispatch(
                                "div_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a / b,
                                |x| x,
                            );
                        }
                        "relu_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, relu_f32);
                        }
                        "gelu_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, gelu_f32);
                        }
                        "silu_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, silu_f32);
                        }
                        "exp_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, exp_f32);
                        }
                        "log_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, log_f32);
                        }
                        "sqrt_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, sqrt_f32);
                        }
                        "neg_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, neg_f32);
                        }
                        "abs_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, abs_f32);
                        }
                        "sigmoid_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, sigmoid_f32);
                        }
                        "tanh_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, tanh_f32);
                        }
                        "leaky_relu_f32" => {
                            let slope = if !params.is_empty() {
                                f32::from_bits(params[0] as u32)
                            } else {
                                0.01
                            };
                            unary_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |input, out_f32| {
                                    leaky_relu_f32(input, out_f32, slope);
                                },
                            );
                        }
                        "elu_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, elu_f32);
                        }
                        "softplus_f32" => {
                            unary_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                softplus_f32,
                            );
                        }
                        "hardswish_f32" => {
                            unary_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                hardswish_f32,
                            );
                        }
                        "clamp_f32" => {
                            let min_val = if !params.is_empty() {
                                f32::from_bits(params[0] as u32)
                            } else {
                                0.0
                            };
                            let max_val = if params.len() > 1 {
                                f32::from_bits(params[1] as u32)
                            } else {
                                1.0
                            };
                            unary_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |input, out_f32| {
                                    clamp_f32(input, out_f32, min_val, max_val);
                                },
                            );
                        }
                        "sign_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, sign_f32);
                        }
                        "round_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, round_f32);
                        }
                        "logical_not_f32" => {
                            unary_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                logical_not_f32,
                            );
                        }
                        "log_softmax_f32" => {
                            unary_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                log_softmax_f32,
                            );
                        }
                        "mish_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, mish_f32);
                        }
                        "max_f32" => {
                            fused_binary_activation_dispatch(
                                "max_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a: f32, b: f32| a.max(b),
                                |x| x,
                            );
                        }
                        "min_f32" => {
                            fused_binary_activation_dispatch(
                                "min_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a: f32, b: f32| a.min(b),
                                |x| x,
                            );
                        }
                        "matmul" => {
                            if let [a_slice, b_slice] = &input_slices[..] {
                                // params: [M, K, N, has_bias(0/1), activation(0=none, 1=relu, 2=gelu, 3=silu)]
                                let matmul_params =
                                    resolve_params(params, param_dims, shape_env, 5)?;
                                let &[m, _k, n, has_bias, activation] = &matmul_params[..] else {
                                    return Err(BackendError::Dispatch(
                                        "matmul: expected params [M,K,N,has_bias,activation]".into(),
                                    ));
                                };
                                let has_bias = has_bias != 0;
                                let bias_slice = if has_bias {
                                    Some(input_slices[2])
                                } else {
                                    None
                                };
                                let (a, b, bias) = {
                                    let d = arena.data_mut();
                                    let a_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[a_slice.offset..a_slice.offset + a_slice.size],
                                    )
                                    .to_vec();
                                    let b_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[b_slice.offset..b_slice.offset + b_slice.size],
                                    )
                                    .to_vec();
                                    let bias_f32 = if let Some(bs) = bias_slice {
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[bs.offset..bs.offset + bs.size],
                                        )
                                        .to_vec()
                                    } else {
                                        Vec::new()
                                    };
                                    (a_f32, b_f32, bias_f32)
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                // *** BATCHED MATMUL ***
                                // Compute batch count from output buffer size. Each batch element is
                                // M*K elements in A, K*N in B, and M*N in the output (all contiguous).
                                let a_stride = m * _k;
                                let b_stride = _k * n;
                                let out_stride = m * n;
                                let batch_count = out_f32.len() / out_stride;
                                // B may be batched (same batch dims as A, e.g. Q*K^T in attention)
                                // or shared across all batches (2D weight matrix). Detect by comparing
                                // total B elements against a single batch's K*N slice.
                                let b_batched = b.len() > b_stride;
                                // Skip BLAS for tiny matrices — dispatch overhead dominates.
                                let use_blas = m * _k * n >= blas::MIN_BLAS_SIZE * 64;
                                let apply_fusion = has_bias || activation != 0;
                                if use_blas {
                                    for batch in 0..batch_count {
                                        let a_s = batch * a_stride;
                                        let b_s = if b_batched { batch * b_stride } else { 0 };
                                        let out_s = batch * out_stride;
                                        matmul_blas_into(
                                            &a[a_s..a_s + a_stride],
                                            &b[b_s..b_s + b_stride],
                                            &mut out_f32[out_s..out_s + out_stride],
                                            m,
                                            _k,
                                            n,
                                        );
                                        if apply_fusion {
                                            let out_batch = &mut out_f32[out_s..out_s + out_stride];
                                            for i in 0..out_batch.len() {
                                                let x = out_batch[i]
                                                    + if has_bias && i % n < bias.len() {
                                                        bias[i % n]
                                                    } else {
                                                        0.0
                                                    };
                                                out_batch[i] = match activation {
                                                    1 => x.max(0.0),
                                                    2 => {
                                                        let x3 = x * x * x;
                                                        let tanh_arg =
                                                            0.797_884_6 * (x + 0.044_715 * x3);
                                                        0.5 * x * (1.0 + tanh_arg.tanh())
                                                    }
                                                    3 => x / (1.0 + (-x).exp()),
                                                    _ => x,
                                                };
                                            }
                                        }
                                    }
                                } else {
                                    // Use SIMD+tiled blocked_row_matmul for small-to-medium matmuls
                                    let total_rows = batch_count * m;
                                    let b_batch_stride = if b_batched { b_stride } else { 0 };
                                    #[cfg(feature = "parallel")]
                                    {
                                        use rayon::prelude::*;
                                        let a_raw = a.as_ptr() as usize;
                                        let b_raw = b.as_ptr() as usize;
                                        let out_raw = out_f32.as_mut_ptr() as usize;
                                        (0..total_rows).into_par_iter().for_each(move |row| {
                                            let a_ptr = a_raw as *const f32;
                                            let b_ptr = b_raw as *const f32;
                                            let out_ptr = out_raw as *mut f32;
                                            unsafe {
                                                crate::backend::cpu::microkernels::blocked_row_matmul(
                                                    a_ptr, b_ptr, out_ptr, row,
                                                    m, n, _k,
                                                    a_stride, _k, 1,
                                                    b_batch_stride,
                                                    n, 1,
                                                );
                                            }
                                            if apply_fusion {
                                                let row_start = row * n;
                                                unsafe {
                                                    let out_row =
                                                        std::slice::from_raw_parts_mut(out_ptr.add(row_start), n);
                                                    for i in 0..n {
                                                        let x = out_row[i]
                                                            + if has_bias && i < bias.len() {
                                                                bias[i]
                                                            } else {
                                                                0.0
                                                            };
                                                        out_row[i] = match activation {
                                                            1 => x.max(0.0),
                                                            2 => {
                                                                let x3 = x * x * x;
                                                                let tanh_arg =
                                                                    0.797_884_6 * (x + 0.044_715 * x3);
                                                                0.5 * x * (1.0 + tanh_arg.tanh())
                                                            }
                                                            3 => x / (1.0 + (-x).exp()),
                                                            _ => x,
                                                        };
                                                    }
                                                }
                                            }
                                        });
                                    }
                                    #[cfg(not(feature = "parallel"))]
                                    for row in 0..total_rows {
                                        unsafe {
                                            crate::backend::cpu::microkernels::blocked_row_matmul(
                                                a.as_ptr(),
                                                b.as_ptr(),
                                                out_f32.as_mut_ptr(),
                                                row,
                                                m, n, _k,
                                                a_stride, _k, 1,
                                                b_batch_stride,
                                                n, 1,
                                            );
                                        }
                                        if apply_fusion {
                                            let row_start = row * n;
                                            let out_row = &mut out_f32[row_start..row_start + n];
                                            for i in 0..n {
                                                let x = out_row[i]
                                                    + if has_bias && i < bias.len() {
                                                        bias[i]
                                                    } else {
                                                        0.0
                                                    };
                                                out_row[i] = match activation {
                                                    1 => x.max(0.0),
                                                    2 => {
                                                        let x3 = x * x * x;
                                                        let tanh_arg =
                                                            0.797_884_6 * (x + 0.044_715 * x3);
                                                        0.5 * x * (1.0 + tanh_arg.tanh())
                                                    }
                                                    3 => x / (1.0 + (-x).exp()),
                                                    _ => x,
                                                };
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "fused_matmul_add_relu" => {
                            matmul_activation_dispatch(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                out_start,
                                out_end,
                                "fused_matmul_add_relu",
                                |x| x.max(0.0),
                            )?;
                        }
                        "fused_matmul_add_gelu" => {
                            matmul_activation_dispatch(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                out_start,
                                out_end,
                                "fused_matmul_add_gelu",
                                |x| {
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    0.5 * x * (1.0 + t)
                                },
                            )?;
                        }
                        "fused_matmul_add_silu" => {
                            matmul_activation_dispatch(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                out_start,
                                out_end,
                                "fused_matmul_add_silu",
                                |x| x / (1.0 + (-x).exp()),
                            )?;
                        }
                        "matmul_relu" => {
                            matmul_activation_dispatch(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                out_start,
                                out_end,
                                "matmul_relu",
                                |x| x.max(0.0),
                            )?;
                        }
                        "matmul_gelu" => {
                            matmul_activation_dispatch(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                out_start,
                                out_end,
                                "matmul_gelu",
                                |x| {
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    0.5 * x * (1.0 + t)
                                },
                            )?;
                        }
                        "matmul_silu" => {
                            matmul_activation_dispatch(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                out_start,
                                out_end,
                                "matmul_silu",
                                |x| x / (1.0 + (-x).exp()),
                            )?;
                        }
                        "matmul_u4" => {
                            quantized_matmul_dispatch::<U4x8>(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                4,
                                "matmul_u4",
                            )?;
                        }
                        "matmul_u4_i8" => {
                            quantized_matmul_dispatch_i8_u4(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                "matmul_u4_i8",
                            )?;
                        }
                        "matmul_u8_i8" => {
                            quantized_matmul_dispatch_i8_u8(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                "matmul_u8_i8",
                            )?;
                        }
                        "matmul_u8" => {
                            quantized_matmul_dispatch::<U8x4>(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                8,
                                "matmul_u8",
                            )?;
                        }
                        "reduce_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let &[group_size, is_mean, is_max] = &params[..3] else {
                                    return Err(BackendError::Dispatch(
                                        "reduce_f32: expected params [group_size, is_mean, is_max]"
                                            .into(),
                                    ));
                                };
                                let effective_group_size = match param_dims {
                                    Some(dims) if !dims.is_empty() => {
                                        dims[0].evaluate_with_env(shape_env).map_err(|e| {
                                            BackendError::Dispatch(format!("reduce_f32: {e}"))
                                        })? as usize
                                    }
                                    _ => group_size,
                                };
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        reduce_f32(
                                            input,
                                            out_f32,
                                            effective_group_size,
                                            is_mean == 1,
                                            is_max == 1,
                                        );
                                    },
                                );
                            }
                        }
                        "transpose_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let transpose_params =
                                    resolve_params(params, param_dims, shape_env, 2)?;
                                let &[m, n] = &transpose_params[..] else {
                                    return Err(BackendError::Dispatch(
                                        "transpose_f32: expected params [M,N]".into(),
                                    ));
                                };
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                        if microkernels::simd_avx2_available() && m >= 8 && n >= 8 {
                                            unsafe {
                                                microkernels::transpose_f32_avx2(input, out_f32, m, n);
                                            }
                                        } else {
                                            #[cfg(not(feature = "parallel"))]
                                            {
                                                for i in 0..m {
                                                    for j in 0..n {
                                                        out_f32[j * m + i] = input[i * n + j];
                                                    }
                                                }
                                            }
                                            #[cfg(feature = "parallel")]
                                            {
                                                use rayon::prelude::*;
                                                out_f32.par_chunks_mut(m).enumerate().for_each(
                                                    |(j, col)| {
                                                        for i in 0..m {
                                                            col[i] = input[i * n + j];
                                                        }
                                                    },
                                                );
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        "transpose_perm_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                // params: [rank, d0..dN, p0..pN]
                                let rank = params.first().copied().unwrap_or(2);
                                let nd_params =
                                    resolve_params(params, param_dims, shape_env, 1 + 2 * rank)?;
                                let dims: Vec<usize> = nd_params[1..1 + rank].to_vec();
                                let perm: Vec<usize> = nd_params[1 + rank..1 + 2 * rank].to_vec();
                                // Pre-compute input and output strides
                                let mut in_strides = vec![1usize; rank];
                                let mut out_strides = vec![1usize; rank];
                                for i in (0..rank - 1).rev() {
                                    in_strides[i] = in_strides[i + 1] * dims[i + 1];
                                }
                                for i in (0..rank - 1).rev() {
                                    out_strides[perm[i]] =
                                        out_strides[perm[i + 1]] * dims[perm[i + 1]];
                                }
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        let _total = out_f32.len();
                                        #[cfg(not(feature = "parallel"))]
                                        {
                                            for out_idx in 0..total {
                                                let mut in_idx = 0usize;
                                                let mut remaining = out_idx;
                                                for k in 0..rank {
                                                    let coord = remaining / out_strides[perm[k]];
                                                    remaining %= out_strides[perm[k]];
                                                    in_idx += coord * in_strides[perm[k]];
                                                }
                                                out_f32[out_idx] = input[in_idx];
                                            }
                                        }
                                        #[cfg(feature = "parallel")]
                                        {
                                            use rayon::prelude::*;
                                            out_f32.par_iter_mut().enumerate().for_each(
                                                |(out_idx, v)| {
                                                    let mut in_idx = 0usize;
                                                    let mut remaining = out_idx;
                                                    for k in 0..rank {
                                                        let coord = remaining / out_strides[perm[k]];
                                                        remaining %= out_strides[perm[k]];
                                                        in_idx += coord * in_strides[perm[k]];
                                                    }
                                                    *v = input[in_idx];
                                                },
                                            );
                                        }
                                    },
                                );
                            }
                        }
                        "add_relu_f32" => {
                            fused_binary_activation_dispatch(
                                "add_relu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a + b,
                                |x| x.max(0.0),
                            );
                        }
                        "sub_relu_f32" => {
                            fused_binary_activation_dispatch(
                                "sub_relu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a - b,
                                |x| x.max(0.0),
                            );
                        }
                        "mul_relu_f32" => {
                            fused_binary_activation_dispatch(
                                "mul_relu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a * b,
                                |x| x.max(0.0),
                            );
                        }
                        "softmax" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        let softmax_params =
                                            resolve_params(params, param_dims, shape_env, 2)
                                                .unwrap_or_else(|_| vec![input.len(), 1]);
                                        let axis_dim_size = softmax_params[0].max(1);
                                        let stride =
                                            softmax_params.get(1).copied().unwrap_or(1).max(1);
                                        let num_rows = input.len() / axis_dim_size.max(1);
                                        softmax_f32(
                                            input,
                                            out_f32,
                                            axis_dim_size,
                                            stride,
                                            num_rows,
                                        );
                                    },
                                );
                            }
                        }
                        "biasadd" => {
                            if let [data_slice, bias_slice] = &input_slices[..] {
                                let (data, bias) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[data_slice.offset
                                                ..data_slice.offset + data_slice.size],
                                        )
                                        .to_vec(),
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[bias_slice.offset
                                                ..bias_slice.offset + bias_slice.size],
                                        )
                                        .to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let &[channel_stride] = &params[..] else {
                                    return Err(BackendError::Dispatch(
                                        "biasadd: expected params [channel_stride]".into(),
                                    ));
                                };
                                biasadd_f32(&data, &bias, out_f32, channel_stride);
                            }
                        }
                        "norm_f32" => {
                            let &[eps_bits, is_batch_norm] = &params[..] else {
                                return Err(BackendError::Dispatch(
                                    "norm_f32: expected params [eps_bits, is_batch_norm]".into(),
                                ));
                            };
                            let eps = f32::from_bits(eps_bits as u32);
                            if is_batch_norm == 1 {
                                // Batch norm (evaluation mode): use running_mean and running_var
                                if let [data_slice, weight_slice, bias_slice, mean_slice, var_slice] =
                                    &input_slices[..]
                                {
                                    let (data, weight, bias, running_mean, running_var) = {
                                        let d = arena.data_mut();
                                        (
                                            bytemuck::cast_slice::<_, f32>(
                                                &d[data_slice.offset
                                                    ..data_slice.offset + data_slice.size],
                                            )
                                            .to_vec(),
                                            bytemuck::cast_slice::<_, f32>(
                                                &d[weight_slice.offset
                                                    ..weight_slice.offset + weight_slice.size],
                                            )
                                            .to_vec(),
                                            bytemuck::cast_slice::<_, f32>(
                                                &d[bias_slice.offset
                                                    ..bias_slice.offset + bias_slice.size],
                                            )
                                            .to_vec(),
                                            bytemuck::cast_slice::<_, f32>(
                                                &d[mean_slice.offset
                                                    ..mean_slice.offset + mean_slice.size],
                                            )
                                            .to_vec(),
                                            bytemuck::cast_slice::<_, f32>(
                                                &d[var_slice.offset
                                                    ..var_slice.offset + var_slice.size],
                                            )
                                            .to_vec(),
                                        )
                                    };
                                    let out_f32 = {
                                        let d = arena.data_mut();
                                        bytemuck::cast_slice_mut::<_, f32>(
                                            &mut d[out_start..out_end],
                                        )
                                    };
                                    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                    {
                                        use crate::backend::cpu::microkernels::has_avx2;
                                        if has_avx2() {
                                            // SAFETY: AVX2 feature checked by has_avx2()
                                            unsafe {
                                                crate::backend::cpu::microkernels::batch_norm_inference_f32_avx2(
                                                    &data, &weight, &bias, &running_mean, &running_var,
                                                    out_f32, eps,
                                                );
                                            }
                                        } else {
                                            crate::backend::cpu::microkernels::batch_norm_inference_f32(
                                                &data, &weight, &bias, &running_mean, &running_var,
                                                out_f32, eps,
                                            );
                                        }
                                    }
                                    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                                    {
                                        crate::backend::cpu::microkernels::batch_norm_inference_f32(
                                            &data,
                                            &weight,
                                            &bias,
                                            &running_mean,
                                            &running_var,
                                            out_f32,
                                            eps,
                                        );
                                    }
                                }
                            } else if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        let row_size = input.len() / out_f32.len().max(1);
                                        norm_layernorm_f32(input, out_f32, row_size, eps);
                                    },
                                );
                            }
                        }
                        "div_relu_f32" => {
                            fused_binary_activation_dispatch(
                                "div_relu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a / b,
                                |x| x.max(0.0),
                            );
                        }
                        // ── Fused elementwise + GELU ─────────────────────
                        "add_gelu_f32" => {
                            fused_binary_activation_dispatch(
                                "add_gelu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a + b,
                                |x| {
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    0.5 * x * (1.0 + t)
                                },
                            );
                        }
                        "sub_gelu_f32" => {
                            fused_binary_activation_dispatch(
                                "sub_gelu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a - b,
                                |x| {
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    0.5 * x * (1.0 + t)
                                },
                            );
                        }
                        "mul_gelu_f32" => {
                            fused_binary_activation_dispatch(
                                "mul_gelu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a * b,
                                |x| {
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    0.5 * x * (1.0 + t)
                                },
                            );
                        }
                        "div_gelu_f32" => {
                            fused_binary_activation_dispatch(
                                "div_gelu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a / b,
                                |x| {
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    0.5 * x * (1.0 + t)
                                },
                            );
                        }
                        // ── Fused elementwise + SiLU ─────────────────────
                        "add_silu_f32" => {
                            fused_binary_activation_dispatch(
                                "add_silu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a + b,
                                |x| x / (1.0 + (-x).exp()),
                            );
                        }
                        "sub_silu_f32" => {
                            fused_binary_activation_dispatch(
                                "sub_silu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a - b,
                                |x| x / (1.0 + (-x).exp()),
                            );
                        }
                        "mul_silu_f32" => {
                            fused_binary_activation_dispatch(
                                "mul_silu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a * b,
                                |x| x / (1.0 + (-x).exp()),
                            );
                        }
                        "div_silu_f32" => {
                            fused_binary_activation_dispatch(
                                "div_silu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a / b,
                                |x| x / (1.0 + (-x).exp()),
                            );
                        }
                        "conv2d" | "conv2d_relu" | "conv2d_gelu" | "conv2d_silu" => {
                            let fused_act = match kernel_name.as_str() {
                                "conv2d_relu" => Some("relu"),
                                "conv2d_gelu" => Some("gelu"),
                                "conv2d_silu" => Some("silu"),
                                _ => None,
                            };
                            if let [input_slice, weight_slice] = &input_slices[..2] {
                                let &[stride, padding, dilation, groups, c, h, w, kh, kw] =
                                    &params[..]
                                else {
                                    return Err(BackendError::Dispatch("conv2d: expected params [stride, padding, dilation, groups, c, h, w, kh, kw]".into()));
                                };
                                let c_per_group = c / groups.max(1);
                                // Recover batch (n) and output-channel (f) counts from the input
                                // and weight byte counts: each f32 is 4 bytes. The previous code
                                // computed these from copied `Vec<f32>::len()` which was the
                                // whole point of the per-call copy.
                                let f32_size = std::mem::size_of::<f32>();
                                let n_in = (input_slice.size / f32_size) / (c * h * w).max(1);
                                let f_out =
                                    (weight_slice.size / f32_size) / (c_per_group * kh * kw).max(1);
                                let _h_out = (h + 2 * padding)
                                    .saturating_sub(dilation * (kh - 1) + 1)
                                    / stride
                                    + 1;
                                let _w_out = (w + 2 * padding)
                                    .saturating_sub(dilation * (kw - 1) + 1)
                                    / stride
                                    + 1;
                                let out_slice = BufferSlice::new(out_start, out_end - out_start);
                                // Build optional activation closure for fused conv+activation.
                                // The activation is applied inside the scatter loop, avoiding a
                                // separate memory round-trip over the output tensor.
                                let conv_act = fused_act.map(|act| match act {
                                    "relu" => {
                                        crate::backend::cpu::microkernels::ConvActivation::Relu
                                    }
                                    "gelu" => {
                                        crate::backend::cpu::microkernels::ConvActivation::Gelu
                                    }
                                    "silu" => {
                                        crate::backend::cpu::microkernels::ConvActivation::Silu
                                    }
                                    _ => unreachable!(),
                                });
                                // Borrow the kernel call with arena slices directly: input, weight,
                                // and bias (if present) are read-only, output is the only mut
                                // region. with_nary_f32_slices handles disjoint/overlap correctly
                                // and avoids the previous per-call `.to_vec()` copies of the input
                                // and weight tensors.
                                if let [input_s, weight_s, bias_s @ ..] = &input_slices[..] {
                                    let inputs_for_kernel: Vec<BufferSlice> = if bias_s.is_empty() {
                                        vec![*input_s, *weight_s]
                                    } else {
                                        vec![*input_s, *weight_s, bias_s[0]]
                                    };
                                    arena::with_nary_f32_slices(
                                        arena,
                                        &inputs_for_kernel,
                                        out_slice,
                                        |inputs, out_f32| {
                                            let input = inputs[0];
                                            let weight = inputs[1];
                                            let bias = if inputs.len() >= 3 {
                                                inputs[2]
                                            } else {
                                                &[][..]
                                            };
                                            // Delegate to the im2col + GEMM microkernel
                                            // (falls back to tiled for small tensors).
                                            crate::backend::cpu::microkernels::conv2d_f32_im2col_gemm(
                                                input, weight, bias, out_f32, n_in, c, h, w, f_out,
                                                kh, kw, stride, padding, dilation, groups, conv_act,
                                            );
                                        },
                                    );
                                }
                            }
                        }
                        "conv2d_u4" | "conv2d_u8" => {
                            // Quantized conv2d using im2col + packed GEMM
                            // input_slices: [activation (f32), weight (packed)]  optional: [bias (f32)]
                            // weight_meta carries bit_width, shape=[OC, IC_per_group*KH*KW], scales[], zero_points[]
                            // Approach: build im2col matrix [N*OH*OW, col_w], then call gemm_cpu per batch
                            // where PackedTensor weights have shape [OC, col_w] and each GEMV produces OC outputs.
                            if let [a_slice, w_slice] = &input_slices[..] {
                                let (input_data, packed_bytes) = {
                                    let d = arena.data_mut();
                                    (
                                        if kernel_name.ends_with("_i8") {
                                            crate::backend::cpu::matmul::dequantize_i8_activation(
                                                &d[a_slice.offset..a_slice.offset + a_slice.size],
                                            )
                                        } else {
                                            bytemuck::cast_slice::<_, f32>(
                                                &d[a_slice.offset..a_slice.offset + a_slice.size],
                                            )
                                            .to_vec()
                                        },
                                        {
                                            let raw =
                                                &d[w_slice.offset..w_slice.offset + w_slice.size];
                                            // Copy to u32-aligned buffer (arena may not be u32-aligned)
                                            let mut aligned: Vec<u32> =
                                                vec![0u32; raw.len().div_ceil(4)];
                                            let byte_slice =
                                                bytemuck::cast_slice_mut::<_, u8>(&mut aligned);
                                            byte_slice[..raw.len()].copy_from_slice(raw);
                                            aligned
                                        },
                                    )
                                };
                                let bias_data: Vec<f32> = if input_slices.len() >= 3 {
                                    let b_slice = &input_slices[2];
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[b_slice.offset..b_slice.offset + b_slice.size],
                                    )
                                    .to_vec()
                                } else {
                                    vec![]
                                };
                                let &[stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w] =
                                    &params[..]
                                else {
                                    return Err(BackendError::Dispatch("conv2d_u4/u8: expected params [stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w]".into()));
                                };
                                let meta = weight_meta.clone().ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "conv2d_u4/u8: missing weight_meta".into(),
                                    )
                                })?;
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                // Flatten conv2d weight shape [OC, IC, KH, KW] → [OC, IC*KH*KW]
                                // so the GEMM dispatch sees 2D weight with inner = IC*KH*KW.
                                let flat_shape = if meta.shape.len() >= 4 {
                                    let oc = meta.shape[0];
                                    let inner: usize = meta.shape[1..].iter().product();
                                    vec![oc, inner]
                                } else {
                                    meta.shape.clone()
                                };
                                let oc = flat_shape.first().copied().unwrap_or(1);
                                let c = input_c;
                                let h = input_h;
                                let w = input_w;
                                if groups == 0 {
                                    return Err(BackendError::Dispatch(
                                        "conv2d_u4/u8: groups=0".into(),
                                    ));
                                }
                                let c_per_g = c / groups;
                                let dk_h = (kernel_h - 1) * dilation + 1;
                                let dk_w = (kernel_w - 1) * dilation + 1;
                                let n = input_data.len() / (c * h * w).max(1);
                                let h_out = if h + 2 * padding >= dk_h {
                                    (h + 2 * padding - dk_h) / stride + 1
                                } else {
                                    0
                                };
                                let w_out = if w + 2 * padding >= dk_w {
                                    (w + 2 * padding - dk_w) / stride + 1
                                } else {
                                    0
                                };
                                // Dispatch based on bit_width to handle different PackedTensor types
                                // (PackedTensor<U4x8> and PackedTensor<U8x4> are different types)
                                macro_rules! dispatch_conv2d_quant {
                                    ($PackedType:ty, $byte_cast:expr) => {{
                                        let packed_data: Vec<$PackedType> =
                                            bytemuck::cast_slice(&packed_bytes).to_vec();
                                        let pt = PackedTensor::from_raw(
                                            packed_data,
                                            flat_shape.clone(),
                                            meta.scales,
                                            meta.zero_points,
                                        );
                                        let col_w = c_per_g * kernel_h * kernel_w;
                                        for nn in 0..n {
                                            let num_pixels = h_out * w_out;
                                            let mut col_matrix = vec![0.0f32; num_pixels * col_w];
                                            unsafe {
                                                crate::backend::cpu::im2col::im2col_kernel_rect(
                                                    &input_data[nn * (c * h * w)..],
                                                    c_per_g,
                                                    h,
                                                    w,
                                                    kernel_h,
                                                    kernel_w,
                                                    stride,
                                                    padding,
                                                    dilation,
                                                    &mut col_matrix,
                                                );
                                            }
                                            // Use flat-buffer GEMM — avoids Vec<Vec> allocation storm
                                            let mut temp_out = vec![0.0f32; num_pixels * oc];
                                            crate::backend::cpu::microkernels::gemm_cpu_flat(
                                                &pt,
                                                &col_matrix,
                                                &mut temp_out,
                                                num_pixels,
                                                col_w,
                                                oc,
                                            );
                                            for pixel in 0..num_pixels {
                                                for ff in 0..oc {
                                                    let mut val = temp_out[pixel * oc + ff];
                                                    if !bias_data.is_empty() && ff < bias_data.len()
                                                    {
                                                        val += bias_data[ff];
                                                    }
                                                    let out_idx = nn * (oc * h_out * w_out)
                                                        + ff * (h_out * w_out)
                                                        + pixel;
                                                    if out_idx < out_f32.len() {
                                                        out_f32[out_idx] = val;
                                                    }
                                                }
                                            }
                                        }
                                    }};
                                }
                                if meta.bit_width == 4 {
                                    dispatch_conv2d_quant!(U4x8, 4);
                                } else {
                                    dispatch_conv2d_quant!(U8x4, 8);
                                }
                            }
                        }
                        "concat" => {
                            if !input_slices.is_empty() && params.len() >= 3 {
                                let _axis = params[0];
                                let _inner_stride = params[1];
                                let outer_count = params[2];
                                // For each input, compute block_size = elements per outer position.
                                let num_inputs = input_slices.len();
                                let mut block_sizes: Vec<usize> = Vec::with_capacity(num_inputs);
                                for slice in input_slices {
                                    let elems = slice.size / std::mem::size_of::<f32>();
                                    block_sizes.push(elems / outer_count.max(1));
                                }
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let mut output_offset = 0;
                                for outer_pos in 0..outer_count {
                                    for (si, slice) in input_slices.iter().enumerate() {
                                        let input_data = {
                                            let d = arena.data_mut();
                                            bytemuck::cast_slice::<_, f32>(
                                                &d[slice.offset..slice.offset + slice.size],
                                            )
                                        };
                                        let bs = block_sizes[si];
                                        let src_start = outer_pos * bs;
                                        let src_end = (src_start + bs).min(input_data.len());
                                        let copy_len = src_end - src_start;
                                        let dst_end = (output_offset + copy_len).min(out_f32.len());
                                        let actual_copy = dst_end - output_offset;
                                        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                        if microkernels::simd_avx2_available() {
                                            unsafe {
                                                microkernels::concat_f32_avx2(
                                                    &input_data[src_start..src_start + actual_copy],
                                                    out_f32,
                                                    output_offset,
                                                );
                                            }
                                        } else {
                                            out_f32[output_offset..dst_end].copy_from_slice(
                                                &input_data[src_start..src_start + actual_copy],
                                            );
                                        }
                                        #[cfg(not(all(
                                            feature = "simd",
                                            target_arch = "x86_64"
                                        )))]
                                        out_f32[output_offset..dst_end].copy_from_slice(
                                            &input_data[src_start..src_start + actual_copy],
                                        );
                                        #[cfg(feature = "debug_canary")]
                                        eprintln!(
                                            "[FNN_DBG_CONCAT] out=[{},{}) outer={} input[{}]: off={} sz={} block={} copy={}",
                                            out_start, out_end, outer_pos, si, slice.offset, slice.size, bs, copy_len
                                        );
                                        output_offset += copy_len;
                                    }
                                }
                            } else if !input_slices.is_empty() {
                                // Fallback: flat concat (legacy, no axis info)
                                let mut output_offset = 0;
                                for (_si, slice) in input_slices.iter().enumerate() {
                                    let input_data = {
                                        let d = arena.data_mut();
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[slice.offset..slice.offset + slice.size],
                                        )
                                        .to_vec()
                                    };
                                    let out_f32 = {
                                        let d = arena.data_mut();
                                        bytemuck::cast_slice_mut::<_, f32>(
                                            &mut d[out_start..out_end],
                                        )
                                    };
                                    let end = (output_offset + input_data.len()).min(out_f32.len());
                                    let actual_copy = end - output_offset;
                                    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                    if microkernels::simd_avx2_available() {
                                        unsafe {
                                            microkernels::concat_f32_avx2(
                                                &input_data[..actual_copy],
                                                out_f32,
                                                output_offset,
                                            );
                                        }
                                    } else {
                                        out_f32[output_offset..end]
                                            .copy_from_slice(&input_data[..actual_copy]);
                                    }
                                    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                                    out_f32[output_offset..end]
                                        .copy_from_slice(&input_data[..actual_copy]);
                                    #[cfg(feature = "debug_canary")]
                                    eprintln!(
                                        "[FNN_DBG_CONCAT] out=[{},{}) input[{}]: off={} sz={} numel={} (flat fallback)",
                                        out_start, out_end, si, slice.offset, slice.size, input_data.len()
                                    );
                                    output_offset += input_data.len();
                                }
                            }
                        }

                        "pool_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset
                                            ..input_slice.offset + input_slice.size],
                                    )
                                    .to_vec()
                                };
                                // params: [kernel, stride, padding, is_max, N, C, H, W]
                                let &[kernel, stride_val, padding_val, is_max, n, c, h, w] =
                                    &params[..]
                                else {
                                    return Err(BackendError::Dispatch("pool_f32: expected params [kernel, stride, padding, is_max, N, C, H, W]".into()));
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let h_out =
                                    (h + 2 * padding_val).saturating_sub(kernel) / stride_val + 1;
                                let w_out =
                                    (w + 2 * padding_val).saturating_sub(kernel) / stride_val + 1;
                                let hw_out = h_out * w_out;
                                // ── Sequential path ──────────────────────
                                #[cfg(not(feature = "parallel"))]
                                {
                                    if is_max == 1 {
                                        let indices_out: Option<&mut [i64]> =
                                            secondary_output_slice.as_ref().map(|sec_slice| {
                                                let d = arena.data_mut();
                                                bytemuck::cast_slice_mut::<_, i64>(
                                                    &mut d[sec_slice.offset
                                                        ..sec_slice.offset + sec_slice.size],
                                                )
                                            });
                                        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                        if microkernels::simd_avx2_available() {
                                            unsafe {
                                                microkernels::pool_max_f32_avx2(
                                                    &input,
                                                    out_f32,
                                                    n,
                                                    c,
                                                    h,
                                                    w,
                                                    kernel,
                                                    stride_val,
                                                    padding_val,
                                                    h_out,
                                                    w_out,
                                                    indices_out,
                                                );
                                            }
                                        } else {
                                            microkernels::pool_max_f32_scalar(
                                                &input,
                                                out_f32,
                                                n,
                                                c,
                                                h,
                                                w,
                                                kernel,
                                                stride_val,
                                                padding_val,
                                                h_out,
                                                w_out,
                                                indices_out,
                                            );
                                        }
                                        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                                        {
                                            microkernels::pool_max_f32_scalar(
                                                &input,
                                                out_f32,
                                                n,
                                                c,
                                                h,
                                                w,
                                                kernel,
                                                stride_val,
                                                padding_val,
                                                h_out,
                                                w_out,
                                                indices_out,
                                            );
                                        }
                                    } else {
                                        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                        if microkernels::simd_avx2_available() {
                                            unsafe {
                                                microkernels::pool_avg_f32_avx2(
                                                    &input,
                                                    out_f32,
                                                    n,
                                                    c,
                                                    h,
                                                    w,
                                                    kernel,
                                                    stride_val,
                                                    padding_val,
                                                    h_out,
                                                    w_out,
                                                );
                                            }
                                        } else {
                                            microkernels::pool_avg_f32_scalar(
                                                &input,
                                                out_f32,
                                                n,
                                                c,
                                                h,
                                                w,
                                                kernel,
                                                stride_val,
                                                padding_val,
                                                h_out,
                                                w_out,
                                            );
                                        }
                                        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                                        {
                                            microkernels::pool_avg_f32_scalar(
                                                &input,
                                                out_f32,
                                                n,
                                                c,
                                                h,
                                                w,
                                                kernel,
                                                stride_val,
                                                padding_val,
                                                h_out,
                                                w_out,
                                            );
                                        }
                                    }
                                }
                                // ── Parallel path (rayon) ────────────────
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    let nc = n * c;
                                    let input_ptr_val = input.as_ptr() as usize;
                                    let out_ptr_val = out_f32.as_mut_ptr() as usize;
                                    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                    let use_simd = microkernels::simd_avx2_available();
                                    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                                    let use_simd = false;
                                    if is_max == 1 {
                                        let (idx_ptr_val, has_indices) =
                                            if let Some(sec_slice) = secondary_output_slice {
                                                let d = arena.data_mut();
                                                let idx_end = sec_slice.offset + sec_slice.size;
                                                let idx_slice = bytemuck::cast_slice_mut::<_, i64>(
                                                    &mut d[sec_slice.offset..idx_end],
                                                );
                                                (idx_slice.as_mut_ptr() as usize, true)
                                            } else {
                                                (0usize, false)
                                            };
                                        (0..nc).into_par_iter().for_each(|nc_idx| {
                                            let nn = nc_idx / c;
                                            let cc = nc_idx % c;
                                            let inp = unsafe {
                                                std::slice::from_raw_parts(
                                                    (input_ptr_val
                                                        + (nn * (c * h * w) + cc * (h * w))
                                                            * std::mem::size_of::<f32>())
                                                        as *const f32,
                                                    h * w,
                                                )
                                            };
                                            let out = unsafe {
                                                std::slice::from_raw_parts_mut(
                                                    (out_ptr_val
                                                        + (nn * (c * hw_out) + cc * hw_out)
                                                            * std::mem::size_of::<f32>())
                                                        as *mut f32,
                                                    hw_out,
                                                )
                                            };
                                            let idx: Option<&mut [i64]> = if has_indices {
                                                Some(unsafe {
                                                    std::slice::from_raw_parts_mut(
                                                        (idx_ptr_val
                                                            + (nn * (c * hw_out) + cc * hw_out)
                                                                * core::mem::size_of::<i64>())
                                                            as *mut i64,
                                                        hw_out,
                                                    )
                                                })
                                            } else {
                                                None
                                            };
                                            if use_simd {
                                                #[cfg(all(
                                                    feature = "simd",
                                                    target_arch = "x86_64"
                                                ))]
                                                unsafe {
                                                    microkernels::pool_max_f32_avx2(
                                                        inp,
                                                        out,
                                                        1,
                                                        1,
                                                        h,
                                                        w,
                                                        kernel,
                                                        stride_val,
                                                        padding_val,
                                                        h_out,
                                                        w_out,
                                                        idx,
                                                    );
                                                }
                                                #[cfg(not(all(
                                                    feature = "simd",
                                                    target_arch = "x86_64"
                                                )))]
                                                {
                                                    unreachable!();
                                                }
                                            } else {
                                                microkernels::pool_max_f32_scalar(
                                                    inp,
                                                    out,
                                                    1,
                                                    1,
                                                    h,
                                                    w,
                                                    kernel,
                                                    stride_val,
                                                    padding_val,
                                                    h_out,
                                                    w_out,
                                                    idx,
                                                );
                                            }
                                        });
                                    } else {
                                        (0..nc).into_par_iter().for_each(|nc_idx| {
                                            let nn = nc_idx / c;
                                            let cc = nc_idx % c;
                                            let inp = unsafe {
                                                std::slice::from_raw_parts(
                                                    (input_ptr_val
                                                        + (nn * (c * h * w) + cc * (h * w))
                                                            * std::mem::size_of::<f32>())
                                                        as *const f32,
                                                    h * w,
                                                )
                                            };
                                            let out = unsafe {
                                                std::slice::from_raw_parts_mut(
                                                    (out_ptr_val
                                                        + (nn * (c * hw_out) + cc * hw_out)
                                                            * std::mem::size_of::<f32>())
                                                        as *mut f32,
                                                    hw_out,
                                                )
                                            };
                                            if use_simd {
                                                #[cfg(all(
                                                    feature = "simd",
                                                    target_arch = "x86_64"
                                                ))]
                                                unsafe {
                                                    microkernels::pool_avg_f32_avx2(
                                                        inp,
                                                        out,
                                                        1,
                                                        1,
                                                        h,
                                                        w,
                                                        kernel,
                                                        stride_val,
                                                        padding_val,
                                                        h_out,
                                                        w_out,
                                                    );
                                                }
                                                #[cfg(not(all(
                                                    feature = "simd",
                                                    target_arch = "x86_64"
                                                )))]
                                                {
                                                    unreachable!();
                                                }
                                            } else {
                                                microkernels::pool_avg_f32_scalar(
                                                    inp,
                                                    out,
                                                    1,
                                                    1,
                                                    h,
                                                    w,
                                                    kernel,
                                                    stride_val,
                                                    padding_val,
                                                    h_out,
                                                    w_out,
                                                );
                                            }
                                        });
                                    }
                                }
                            }
                        }
                        "pad_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset
                                            ..input_slice.offset + input_slice.size],
                                    )
                                    .to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                // Simple flat copy: the caller right-sizes the output buffer.
                                // Output includes padding zeros already allocated.
                                let end = input.len().min(out_f32.len());
                                out_f32[..end].copy_from_slice(&input[..end]);
                            }
                        }
                        "gather" => {
                            if let [data_slice, indices_slice] = &input_slices[..] {
                                let (input, indices) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[data_slice.offset
                                                ..data_slice.offset + data_slice.size],
                                        )
                                        .to_vec(),
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[indices_slice.offset
                                                ..indices_slice.offset + indices_slice.size],
                                        )
                                        .to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let axis = if !params.is_empty() { params[0] } else { 0 };
                                let inner = if axis == 0 {
                                    input.len() / out_f32.len().max(1)
                                } else {
                                    1
                                };
                                for i in 0..out_f32.len() {
                                    let idx_idx = i.checked_div(inner).unwrap_or(i);
                                    let idx = indices[idx_idx.min(indices.len().saturating_sub(1))]
                                        as usize;
                                    let src = idx * inner + (i % inner);
                                    out_f32[i] = if src < input.len() { input[src] } else { 0.0 };
                                }
                            }
                        }
                        "slice_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset
                                            ..input_slice.offset + input_slice.size],
                                    )
                                    .to_vec()
                                };
                                let &[_dim, start, end, stride] = &params[..] else {
                                    return Err(BackendError::Dispatch(
                                        "slice_f32: expected params [dim, start, end, stride]"
                                            .into(),
                                    ));
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                // General strided slice along dimension `dim`.
                                // input layout (row-major): outer * dim_size * stride
                                // output layout:             outer * (end-start) * stride
                                let in_len = input.len();
                                let out_len = out_f32.len();
                                let range_len = (end - start).max(1);
                                // dim_size = in_len * range_len / out_len
                                let dim_size = if out_len > 0 {
                                    (in_len * range_len) / out_len
                                } else {
                                    0
                                };
                                let outer = if dim_size > 0 && stride > 0 {
                                    in_len / dim_size / stride
                                } else {
                                    1
                                };
                                let slice_elems = range_len * stride;
                                for i in 0..outer {
                                    let src_off = i * dim_size * stride + start * stride;
                                    let dst_off = i * slice_elems;
                                    let copy_len = slice_elems.min(in_len.saturating_sub(src_off));
                                    if copy_len > 0 {
                                        out_f32[dst_off..(dst_off + copy_len)]
                                            .copy_from_slice(&input[src_off..(src_off + copy_len)]);
                                    }
                                }
                            }
                        }
                        "scatter_nd" => {
                            if let [data_slice, indices_slice, updates_slice] = &input_slices[..] {
                                let (data, indices_f32, updates) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[data_slice.offset
                                                ..data_slice.offset + data_slice.size],
                                        )
                                        .to_vec(),
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[indices_slice.offset
                                                ..indices_slice.offset + indices_slice.size],
                                        )
                                        .to_vec(),
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[updates_slice.offset
                                                ..updates_slice.offset + updates_slice.size],
                                        )
                                        .to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                out_f32.copy_from_slice(&data);
                                if let Some((&index_depth, data_dims)) = params.split_first() {
                                    let data_rank = data_dims.len();
                                    if index_depth > 0
                                        && index_depth <= data_rank
                                        && indices_f32.len() >= index_depth
                                    {
                                        let num_indices = indices_f32.len() / index_depth;
                                        let inner_size: usize =
                                            data_dims[index_depth..].iter().product();
                                        for i in 0..num_indices {
                                            let mut linear_offset = 0usize;
                                            for j in 0..index_depth {
                                                let idx = indices_f32[i * index_depth + j] as usize;
                                                let mut stride = 1usize;
                                                for k in (j + 1)..data_rank {
                                                    stride *= data_dims[k];
                                                }
                                                linear_offset += idx * stride;
                                            }
                                            let update_start = i * inner_size;
                                            let update_end = update_start + inner_size;
                                            if linear_offset + inner_size <= out_f32.len()
                                                && update_end <= updates.len()
                                            {
                                                out_f32[linear_offset..linear_offset + inner_size]
                                                    .copy_from_slice(
                                                        &updates[update_start..update_end],
                                                    );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "conv1d" => {
                            if let [input_slice, weight_slice] = &input_slices[..] {
                                let (input, weight) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[input_slice.offset
                                                ..input_slice.offset + input_slice.size],
                                        )
                                        .to_vec(),
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[weight_slice.offset
                                                ..weight_slice.offset + weight_slice.size],
                                        )
                                        .to_vec(),
                                    )
                                };
                                let &[stride, padding, c, w, kw] = &params[..] else {
                                    return Err(BackendError::Dispatch(
                                        "conv1d: expected params [stride, padding, input_c, input_w, kernel_w]".into(),
                                    ));
                                };
                                let n = input.len() / (c * w).max(1);
                                let f = weight.len() / (c * kw).max(1);
                                let w_out = (w + 2 * padding).saturating_sub(kw) / stride + 1;
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for nn in 0..n {
                                    for ff in 0..f {
                                        for ww in 0..w_out {
                                            let mut sum = 0.0f32;
                                            for cc in 0..c {
                                                for kkw in 0..kw {
                                                    let w_in = ww * stride + kkw;
                                                    if w_in >= padding {
                                                        let w_in_s = w_in - padding;
                                                        if w_in_s < w {
                                                            sum += input
                                                                [nn * (c * w) + cc * w + w_in_s]
                                                                * weight
                                                                    [ff * (c * kw) + cc * kw + kkw];
                                                        }
                                                    }
                                                }
                                            }
                                            out_f32[nn * (f * w_out) + ff * w_out + ww] = sum;
                                        }
                                    }
                                }
                            }
                        }
                        "conv3d" => {
                            if let [input_slice, weight_slice] = &input_slices[..] {
                                let (input, weight) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[input_slice.offset
                                                ..input_slice.offset + input_slice.size],
                                        )
                                        .to_vec(),
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[weight_slice.offset
                                                ..weight_slice.offset + weight_slice.size],
                                        )
                                        .to_vec(),
                                    )
                                };
                                let &[stride, padding, dilation, c, d, h, w, kd, kh, kw] =
                                    &params[..]
                                else {
                                    return Err(BackendError::Dispatch(
                                        "conv3d: expected params [stride, padding, dilation, input_c, input_d, input_h, input_w, kernel_d, kernel_h, kernel_w]".into(),
                                    ));
                                };
                                let n = input.len() / (c * d * h * w).max(1);
                                let f = weight.len() / (c * kd * kh * kw).max(1);
                                let d_out = (d + 2 * padding)
                                    .saturating_sub(dilation * (kd - 1) + 1)
                                    / stride
                                    + 1;
                                let h_out = (h + 2 * padding)
                                    .saturating_sub(dilation * (kh - 1) + 1)
                                    / stride
                                    + 1;
                                let w_out = (w + 2 * padding)
                                    .saturating_sub(dilation * (kw - 1) + 1)
                                    / stride
                                    + 1;
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for nn in 0..n {
                                    for ff in 0..f {
                                        for dd in 0..d_out {
                                            for hh in 0..h_out {
                                                for ww in 0..w_out {
                                                    let mut sum = 0.0f32;
                                                    for cc in 0..c {
                                                        for kkd in 0..kd {
                                                            for kkh in 0..kh {
                                                                for kkw in 0..kw {
                                                                    let d_in = dd * stride
                                                                        + kkd * dilation;
                                                                    let h_in = hh * stride
                                                                        + kkh * dilation;
                                                                    let w_in = ww * stride
                                                                        + kkw * dilation;
                                                                    if d_in >= padding
                                                                        && h_in >= padding
                                                                        && w_in >= padding
                                                                    {
                                                                        let d_in_s = d_in - padding;
                                                                        let h_in_s = h_in - padding;
                                                                        let w_in_s = w_in - padding;
                                                                        if d_in_s < d
                                                                            && h_in_s < h
                                                                            && w_in_s < w
                                                                        {
                                                                            let input_idx = nn
                                                                                * (c * d * h * w)
                                                                                + cc * (d * h * w)
                                                                                + d_in_s * (h * w)
                                                                                + h_in_s * w
                                                                                + w_in_s;
                                                                            let weight_idx = ff
                                                                                * (c * kd
                                                                                    * kh
                                                                                    * kw)
                                                                                + cc * (kd
                                                                                    * kh
                                                                                    * kw)
                                                                                + kkd * (kh * kw)
                                                                                + kkh * kw
                                                                                + kkw;
                                                                            sum += input[input_idx]
                                                                                * weight
                                                                                    [weight_idx];
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    let out_idx = nn * (f * d_out * h_out * w_out)
                                                        + ff * (d_out * h_out * w_out)
                                                        + dd * (h_out * w_out)
                                                        + hh * w_out
                                                        + ww;
                                                    out_f32[out_idx] = sum;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "conv_transpose2d" => {
                            if let [input_slice, weight_slice] = &input_slices[..] {
                                let (input, weight) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[input_slice.offset
                                                ..input_slice.offset + input_slice.size],
                                        )
                                        .to_vec(),
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[weight_slice.offset
                                                ..weight_slice.offset + weight_slice.size],
                                        )
                                        .to_vec(),
                                    )
                                };
                                let &[stride, padding, c, hin, win, kh, kw] = &params[..] else {
                                    return Err(BackendError::Dispatch(
                                        "conv_transpose2d: expected params [stride, padding, input_c, input_h, input_w, kernel_h, kernel_w]"
                                            .into(),
                                    ));
                                };
                                let n = input.len() / (c * hin * win).max(1);
                                let f = weight.len() / (c * kh * kw).max(1);
                                let h_out = ((hin - 1) * stride + kh).saturating_sub(2 * padding);
                                let w_out = ((win - 1) * stride + kw).saturating_sub(2 * padding);
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                out_f32.fill(0.0f32);
                                for nn in 0..n {
                                    for cc in 0..c {
                                        for hh in 0..hin {
                                            for ww in 0..win {
                                                for ff in 0..f {
                                                    for kkh in 0..kh {
                                                        for kkw in 0..kw {
                                                            let h_out_idx = hh * stride + kkh;
                                                            let w_out_idx = ww * stride + kkw;
                                                            if h_out_idx >= padding
                                                                && w_out_idx >= padding
                                                            {
                                                                let h_out_s = h_out_idx - padding;
                                                                let w_out_s = w_out_idx - padding;
                                                                if h_out_s < h_out
                                                                    && w_out_s < w_out
                                                                {
                                                                    let out_idx = nn
                                                                        * (f * h_out * w_out)
                                                                        + ff * (h_out * w_out)
                                                                        + h_out_s * w_out
                                                                        + w_out_s;
                                                                    let input_idx = nn
                                                                        * (c * hin * win)
                                                                        + cc * (hin * win)
                                                                        + hh * win
                                                                        + ww;
                                                                    let weight_idx = cc
                                                                        * (f * kh * kw)
                                                                        + ff * (kh * kw)
                                                                        + kkh * kw
                                                                        + kkw;
                                                                    if out_idx < out_f32.len()
                                                                        && input_idx < input.len()
                                                                        && weight_idx < weight.len()
                                                                    {
                                                                        out_f32[out_idx] += input
                                                                            [input_idx]
                                                                            * weight[weight_idx];
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
                        }
                        "prelu" => {
                            if let [data_slice, weight_slice] = &input_slices[..] {
                                let (input, weight) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[data_slice.offset
                                                ..data_slice.offset + data_slice.size],
                                        )
                                        .to_vec(),
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[weight_slice.offset
                                                ..weight_slice.offset + weight_slice.size],
                                        )
                                        .to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let channel_stride =
                                    if !weight.is_empty() && input.len() > weight.len() {
                                        input.len() / weight.len()
                                    } else {
                                        1
                                    };
                                for i in 0..out_f32.len().min(input.len()) {
                                    let w_idx = if weight.len() == 1 {
                                        0
                                    } else {
                                        (i / channel_stride) % weight.len()
                                    };
                                    let slope = if w_idx < weight.len() {
                                        weight[w_idx]
                                    } else {
                                        0.0
                                    };
                                    out_f32[i] = if input[i] > 0.0 {
                                        input[i]
                                    } else {
                                        input[i] * slope
                                    };
                                }
                            }
                        }
                        "rms_norm" => {
                            if let [data_slice, weight_slice] = &input_slices[..] {
                                let eps =
                                    f32::from_bits(params.first().copied().unwrap_or(0) as u32);
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                arena::with_nary_f32_slices(
                                    arena,
                                    &[*data_slice, *weight_slice],
                                    output_slice,
                                    |inputs, out_f32| {
                                        let input = inputs[0];
                                        let weight = inputs[1];
                                        let row_size = if !weight.is_empty() {
                                            input.len() / weight.len()
                                        } else {
                                            input.len()
                                        };
                                        rms_norm_f32(input, weight, out_f32, row_size, eps);
                                    },
                                );
                            }
                        }
                        "fused_residual_add_layer_norm" => {
                            if let [residual_slice, main_slice, rest @ ..] = &input_slices[..] {
                                let eps =
                                    f32::from_bits(params.first().copied().unwrap_or(0) as u32);
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let weight_slice = rest.first().copied();
                                let bias_slice = rest.get(1).copied();
                                let fallback_row_size = weight_slice
                                    .or(bias_slice)
                                    .map(|slice| slice.size / std::mem::size_of::<f32>())
                                    .unwrap_or(output_slice.size / std::mem::size_of::<f32>());
                                let row_size = params
                                    .get(1)
                                    .copied()
                                    .filter(|&value| value > 0)
                                    .unwrap_or(fallback_row_size);

                                if residual_slice.size == main_slice.size
                                    && main_slice.size == output_slice.size
                                    && row_size > 0
                                    && (output_slice.size / std::mem::size_of::<f32>())
                                        .is_multiple_of(row_size)
                                    && weight_slice.is_none_or(|w| {
                                        w.size / std::mem::size_of::<f32>() == row_size
                                    })
                                    && bias_slice.is_none_or(|b| {
                                        b.size / std::mem::size_of::<f32>() == row_size
                                    })
                                {
                                    let mut inputs = vec![*residual_slice, *main_slice];
                                    if let Some(w) = weight_slice {
                                        inputs.push(w);
                                    }
                                    if let Some(b) = bias_slice {
                                        inputs.push(b);
                                    }
                                    arena::with_nary_f32_slices(
                                        arena,
                                        &inputs,
                                        output_slice,
                                        |inputs, out_f32| {
                                            let weight = if weight_slice.is_some() {
                                                inputs[2]
                                            } else {
                                                &[]
                                            };
                                            let bias = match (
                                                weight_slice.is_some(),
                                                bias_slice.is_some(),
                                            ) {
                                                (true, true) => inputs[3],
                                                (false, true) => inputs[2],
                                                _ => &[],
                                            };
                                            microkernels::fused_residual_add_layer_norm_f32_scalar(
                                                inputs[0], inputs[1], weight, bias, out_f32,
                                                row_size, eps,
                                            );
                                        },
                                    );
                                }
                            }
                        }
                        "fused_residual_add_rms_norm" => {
                            if let [residual_slice, main_slice, rest @ ..] = &input_slices[..] {
                                let eps =
                                    f32::from_bits(params.first().copied().unwrap_or(0) as u32);
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let weight_slice = rest.first().copied();
                                let fallback_row_size = weight_slice
                                    .map(|slice| slice.size / std::mem::size_of::<f32>())
                                    .unwrap_or(output_slice.size / std::mem::size_of::<f32>());
                                let row_size = params
                                    .get(1)
                                    .copied()
                                    .filter(|&value| value > 0)
                                    .unwrap_or(fallback_row_size);

                                if residual_slice.size == main_slice.size
                                    && main_slice.size == output_slice.size
                                    && row_size > 0
                                    && (output_slice.size / std::mem::size_of::<f32>())
                                        .is_multiple_of(row_size)
                                    && weight_slice.is_none_or(|w| {
                                        w.size / std::mem::size_of::<f32>() == row_size
                                    })
                                {
                                    let mut inputs = vec![*residual_slice, *main_slice];
                                    if let Some(w) = weight_slice {
                                        inputs.push(w);
                                    }
                                    arena::with_nary_f32_slices(
                                        arena,
                                        &inputs,
                                        output_slice,
                                        |inputs, out_f32| {
                                            let weight = if weight_slice.is_some() {
                                                inputs[2]
                                            } else {
                                                &[]
                                            };
                                            microkernels::fused_residual_add_rms_norm_f32_scalar(
                                                inputs[0], inputs[1], weight, out_f32, row_size,
                                                eps,
                                            );
                                        },
                                    );
                                }
                            }
                        }
                        "embedding" => {
                            if let [weight_slice, indices_slice] = &input_slices[..] {
                                let (weight, indices) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[weight_slice.offset
                                                ..weight_slice.offset + weight_slice.size],
                                        )
                                        .to_vec(),
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[indices_slice.offset
                                                ..indices_slice.offset + indices_slice.size],
                                        )
                                        .to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let dim = if !weight.is_empty() && !indices.is_empty() {
                                    out_f32.len() / indices.len()
                                } else {
                                    1
                                };
                                for i in 0..indices.len() {
                                    let idx = indices[i] as usize;
                                    let src_start = idx * dim;
                                    let dst_start = i * dim;
                                    let len = dim
                                        .min(weight.len().saturating_sub(src_start))
                                        .min(out_f32.len().saturating_sub(dst_start));
                                    if len > 0 {
                                        out_f32[dst_start..dst_start + len]
                                            .copy_from_slice(&weight[src_start..src_start + len]);
                                    }
                                }
                            }
                        }
                        "pow_f32" => {
                            if let [data_slice, exp_slice] = &input_slices[..] {
                                let (data, exponent) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[data_slice.offset
                                                ..data_slice.offset + data_slice.size],
                                        )
                                        .to_vec(),
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[exp_slice.offset..exp_slice.offset + exp_slice.size],
                                        )
                                        .to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let len = out_f32.len().min(data.len());
                                #[cfg(not(feature = "parallel"))]
                                {
                                    for i in 0..len {
                                        let e = if i < exponent.len() {
                                            exponent[i]
                                        } else {
                                            exponent[exponent.len().saturating_sub(1)]
                                        };
                                        out_f32[i] = data[i].powf(e);
                                    }
                                }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    let exponent = &exponent;
                                    out_f32[..len]
                                        .par_iter_mut()
                                        .enumerate()
                                        .for_each(|(i, o)| {
                                            let e = if i < exponent.len() {
                                                exponent[i]
                                            } else {
                                                exponent[exponent.len().saturating_sub(1)]
                                            };
                                            *o = data[i].powf(e);
                                        });
                                }
                            }
                        }
                        "gt_scalar_f32" => {
                            scalar_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |data, s, out| gt_scalar_f32(data, s, out),
                            );
                        }
                        "lt_scalar_f32" => {
                            scalar_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |data, s, out| lt_scalar_f32(data, s, out),
                            );
                        }
                        "eq_scalar_f32" => {
                            scalar_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |data, s, out| eq_scalar_f32(data, s, out),
                            );
                        }
                        "add_scalar_f32" => {
                            scalar_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |data, s, out| add_scalar_f32(data, s, out),
                            );
                        }
                        "mul_scalar_f32" => {
                            scalar_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |data, s, out| mul_scalar_f32(data, s, out),
                            );
                        }
                        "div_scalar_f32" => {
                            scalar_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |data, s, out| div_scalar_f32(data, s, out),
                            );
                        }
                        "argmax" => {
                            if let Some(input_slice) = input_slices.first() {
                                let axis = params.first().copied().unwrap_or(usize::MAX);
                                let dim_size = params.get(1).copied().unwrap_or(0);
                                let inner = params.get(2).copied().unwrap_or(1);
                                let input_end = input_slice.offset + input_slice.size;
                                let output_bytes = BufferSlice::new(out_start, out_end - out_start);

                                if !arena::ranges_overlap(
                                    input_slice.offset,
                                    input_end,
                                    output_bytes.offset,
                                    output_bytes.offset + output_bytes.size,
                                ) {
                                    let d = arena.data_mut();
                                    assert!(input_end <= d.len());
                                    assert!(out_end <= d.len());
                                    // SAFETY: bounds were checked above and the input/output byte
                                    // ranges are disjoint, so a shared f32 input slice cannot alias
                                    // the mutable u64 output slice.
                                    unsafe {
                                        let input = std::slice::from_raw_parts(
                                            d.as_ptr().add(input_slice.offset).cast::<f32>(),
                                            input_slice.size / std::mem::size_of::<f32>(),
                                        );
                                        let out_u64 = std::slice::from_raw_parts_mut(
                                            d.as_mut_ptr().add(out_start).cast::<u64>(),
                                            (out_end - out_start) / std::mem::size_of::<u64>(),
                                        );
                                        argmax_f32(input, out_u64, axis, dim_size, inner);
                                    }
                                } else {
                                    let input = {
                                        let d = arena.data_mut();
                                        let src = bytemuck::cast_slice::<_, f32>(
                                            &d[input_slice.offset..input_end],
                                        );
                                        let mut copy =
                                            crate::backend::cpu::microkernels::TlsVecPool::alloc(
                                                src.len(),
                                            );
                                        copy.copy_from_slice(src);
                                        crate::backend::cpu::telemetry::record_arena_temp_copy(
                                            input_slice.size,
                                        );
                                        copy
                                    };
                                    let d = arena.data_mut();
                                    let out_u64 = bytemuck::cast_slice_mut::<_, u64>(
                                        &mut d[out_start..out_end],
                                    );
                                    argmax_f32(&input, out_u64, axis, dim_size, inner);
                                }
                            }
                        }
                        "topk_fused" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let k = params.first().copied().unwrap_or(1);
                                arena::with_unary_f32_slices(arena, *input_slice, output_slice, |input, out_slice| {
                                    let mut indexed: Vec<(usize, f32)> =
                                        input.iter().copied().enumerate().collect();
                                    if input.len() > k {
                                        indexed.select_nth_unstable_by(
                                            input.len().saturating_sub(k),
                                            |a, b| {
                                                a.1.partial_cmp(&b.1)
                                                    .unwrap_or(std::cmp::Ordering::Equal)
                                            },
                                        );
                                    }

                                    // Write values (f32) to primary output
                                    for i in 0..k.min(out_slice.len()) {
                                        out_slice[i] = indexed[input.len().saturating_sub(k) + i].1;
                                    }

                                    // Write indices (i64) to secondary output
                                    if let Some(sec_slice) = secondary_output_slice {
                                        let d = arena.data_mut();
                                        let sec_start = sec_slice.offset;
                                        let sec_end = sec_slice.offset + sec_slice.size;
                                        let idx_slice = bytemuck::cast_slice_mut::<_, u64>(
                                            &mut d[sec_start..sec_end],
                                        );
                                        for i in 0..k.min(idx_slice.len()) {
                                            idx_slice[i] =
                                                indexed[input.len().saturating_sub(k) + i].0 as u64;
                                        }
                                    }
                                });
                            }
                        }
                        "topk_values" | "topk_indices" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let k = params.first().copied().unwrap_or(1);
                                let _axis = params.get(1).copied().unwrap_or(usize::MAX);
                                let is_values = kernel_name == "topk_values";
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_slice| {
                                        let mut indexed: Vec<(usize, f32)> =
                                            input.iter().copied().enumerate().collect();
                                        if input.len() > k {
                                            indexed.select_nth_unstable_by(
                                                input.len().saturating_sub(k),
                                                |a, b| {
                                                    a.1.partial_cmp(&b.1)
                                                        .unwrap_or(std::cmp::Ordering::Equal)
                                                },
                                            );
                                        }
                                        if is_values {
                                            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(out_slice);
                                            for i in 0..k.min(out_f32.len()) {
                                                out_f32[i] =
                                                    indexed[input.len().saturating_sub(k) + i].1;
                                            }
                                        } else {
                                            let out_u64 = bytemuck::cast_slice_mut::<_, u64>(out_slice);
                                            for i in 0..k.min(out_u64.len()) {
                                                out_u64[i] =
                                                    indexed[input.len().saturating_sub(k) + i].0
                                                        as u64;
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        "upsample_nearest2d" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let scale_h = params.first().copied().unwrap_or(2);
                                let scale_w = params.get(1).copied().unwrap_or(2);
                                let h_in = params.get(2).copied().unwrap_or(1);
                                let w_in = params.get(3).copied().unwrap_or(1);
                                let hw = h_in * w_in;
                                arena::with_unary_f32_slices(arena, *input_slice, output_slice, |input, out_f32| {
                                    let in_len = input.len();
                                    if scale_h > 0
                                        && scale_w > 0
                                        && hw > 0
                                        && in_len > 0
                                        && in_len % hw == 0
                                    {
                                        let nc = in_len / hw;
                                        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                        {
                                            use crate::backend::cpu::microkernels::has_avx2;
                                            if has_avx2() {
                                                unsafe {
                                                    crate::backend::cpu::microkernels::upsample_nearest2d_f32_avx2(
                                                        input, out_f32, nc, h_in, w_in, scale_h, scale_w,
                                                    );
                                                }
                                                return;
                                            }
                                        }
                                        crate::backend::cpu::microkernels::upsample_nearest2d_f32(
                                            input, out_f32, nc, h_in, w_in, scale_h, scale_w,
                                        );
                                    }
                                });
                            }
                        }
                        "upsample_bilinear2d" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let scale_h = params.first().copied().unwrap_or(2);
                                let scale_w = params.get(1).copied().unwrap_or(2);
                                let h_in = params.get(2).copied().unwrap_or(1);
                                let w_in = params.get(3).copied().unwrap_or(1);
                                let hw = h_in * w_in;
                                arena::with_unary_f32_slices(arena, *input_slice, output_slice, |input, out_f32| {
                                    let out_len = out_f32.len();
                                    let in_len = input.len();
                                    if scale_h > 0
                                        && scale_w > 0
                                        && hw > 0
                                        && out_len == in_len * scale_h * scale_w
                                        && in_len > 0
                                        && in_len % hw == 0
                                    {
                                        let nc = in_len / hw;
                                        for nci in 0..nc {
                                            for hi in 0..h_in * scale_h {
                                                for wi in 0..w_in * scale_w {
                                                    let src_h = (hi as f64 / scale_h as f64)
                                                        .min((h_in - 1) as f64);
                                                    let src_w = (wi as f64 / scale_w as f64)
                                                        .min((w_in - 1) as f64);
                                                    let h0 = src_h.floor() as usize;
                                                    let w0 = src_w.floor() as usize;
                                                    let h1 = (h0 + 1).min(h_in - 1);
                                                    let w1 = (w0 + 1).min(w_in - 1);
                                                    let dh = src_h - h0 as f64;
                                                    let dw = src_w - w0 as f64;
                                                    let v00 = input[nci * hw + h0 * w_in + w0];
                                                    let v01 = input[nci * hw + h0 * w_in + w1];
                                                    let v10 = input[nci * hw + h1 * w_in + w0];
                                                    let v11 = input[nci * hw + h1 * w_in + w1];
                                                    let v0 = v00 * (1.0 - dw as f32) + v01 * dw as f32;
                                                    let v1 = v10 * (1.0 - dw as f32) + v11 * dw as f32;
                                                    let val = v0 * (1.0 - dh as f32) + v1 * dh as f32;
                                                    let out_idx = nci * h_in * scale_h * w_in * scale_w
                                                        + hi * w_in * scale_w
                                                        + wi;
                                                    if out_idx < out_len {
                                                        out_f32[out_idx] = val;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        "adaptive_avg_pool2d" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let out_h = params.first().copied().unwrap_or(1);
                                let out_w = params.get(1).copied().unwrap_or(1);
                                arena::with_unary_f32_slices(arena, *input_slice, output_slice, |input, out_f32| {
                                    let out_len = out_f32.len();
                                    if out_len > 0 {
                                        let in_len = input.len();
                                        let nc = if out_h > 0 && out_w > 0 {
                                            out_len / (out_h * out_w)
                                        } else {
                                            0
                                        };
                                        if nc > 0 && in_len > 0 && in_len % nc == 0 {
                                            let hw = in_len / nc;
                                            let mut h = (hw as f64).sqrt() as usize;
                                            while h > 0 && hw % h != 0 {
                                                h -= 1;
                                            }
                                            let w = hw / h;
                                            if h >= out_h && w >= out_w && h > 0 && w > 0 {
                                                microkernels::adaptive_avg_pool2d_f32_scalar(
                                                    input, out_f32, nc, h, w, out_h, out_w,
                                                );
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        "repeat" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                arena::with_unary_f32_slices(arena, *input_slice, output_slice, |input, out_f32| {
                                    let in_len = input.len();
                                    let out_len = out_f32.len();
                                    if in_len > 0 && out_len >= in_len && out_len % in_len == 0 {
                                        let factor = out_len / in_len;
                                        for i in 0..in_len {
                                            let val = input[i];
                                            for f in 0..factor {
                                                out_f32[i * factor + f] = val;
                                            }
                                        }
                                    } else if in_len > 0 {
                                        for i in 0..out_len {
                                            out_f32[i] = input[i % in_len];
                                        }
                                    }
                                });
                            }
                        }
                        "cumsum" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let exclusive = params.get(1).copied().unwrap_or(0);
                                let rev = params.get(2).copied().unwrap_or(0);
                                arena::with_unary_f32_slices(arena, *input_slice, output_slice, |input, out_f32| {
                                    let len = out_f32.len().min(input.len());
                                    if rev == 0 {
                                        let mut s = 0.0f32;
                                        for i in 0..len {
                                            s += input[i];
                                            out_f32[i] = if exclusive != 0 { s - input[i] } else { s };
                                        }
                                    } else {
                                        let mut s = 0.0f32;
                                        for i in (0..len).rev() {
                                            s += input[i];
                                            out_f32[i] = if exclusive != 0 { s - input[i] } else { s };
                                        }
                                    }
                                });
                            }
                        }
                        "erf_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                arena::with_unary_f32_slices(arena, *input_slice, output_slice, |input, out_f32| {
                                    for i in 0..out_f32.len().min(input.len()) {
                                        let x = input[i];
                                        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
                                        #[allow(clippy::excessive_precision)]
                                        let y = 1.0
                                            - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741)
                                                * t
                                                - 0.284496736)
                                                * t
                                                + 0.254829592)
                                                * t
                                                * (-x * x).exp();
                                        out_f32[i] = x.signum() * y;
                                    }
                                });
                            }
                        }
                        "flip" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let num_dims = params.first().copied().unwrap_or(0);
                                let flip_dims: Vec<usize> = if num_dims > 0 {
                                    params[1..1 + num_dims].to_vec()
                                } else {
                                    vec![]
                                };
                                let shape: Vec<usize> = if num_dims > 0 {
                                    params[1 + num_dims..].to_vec()
                                } else {
                                    vec![]
                                };
                                let ndim = shape.len();
                                arena::with_unary_f32_slices(arena, *input_slice, output_slice, |input, out_f32| {
                                    let len = out_f32.len().min(input.len());
                                    if params.is_empty() {
                                        for i in 0..len {
                                            out_f32[i] = input[len - 1 - i];
                                        }
                                    } else {
                                        let mut indices = vec![0i64; ndim];
                                        let mut strides = vec![0i64; ndim];
                                        let mut stride = 1i64;
                                        for d in (0..ndim).rev() {
                                            strides[d] = stride;
                                            stride *= shape[d] as i64;
                                        }
                                        for out_idx in 0..len {
                                            let mut src_idx = 0i64;
                                            for d in 0..ndim {
                                                let idx = if flip_dims.contains(&d) {
                                                    shape[d] as i64 - 1 - indices[d]
                                                } else {
                                                    indices[d]
                                                };
                                                src_idx += idx * strides[d];
                                            }
                                            out_f32[out_idx] = input[src_idx as usize];
                                            for d in (0..ndim).rev() {
                                                indices[d] += 1;
                                                if indices[d] < shape[d] as i64 {
                                                    break;
                                                }
                                                indices[d] = 0;
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        "where_f32" => {
                            if input_slices.len() >= 3 {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let cond_slice = input_slices[0];
                                let x_slice = input_slices[1];
                                let y_slice = input_slices[2];
                                arena::with_nary_f32_slices(
                                    arena,
                                    &[cond_slice, x_slice, y_slice],
                                    output_slice,
                                    |inputs, out_f32| {
                                        let cond = inputs[0];
                                        let x = inputs[1];
                                        let y = inputs[2];
                                        let len = out_f32.len();
                                        for i in 0..len {
                                            let c = cond.get(i % cond.len()).copied().unwrap_or(0.0);
                                            out_f32[i] = if c != 0.0 {
                                                x.get(i % x.len()).copied().unwrap_or(0.0)
                                            } else {
                                                y.get(i % y.len()).copied().unwrap_or(0.0)
                                            };
                                        }
                                    },
                                );
                            }
                        }
                        // ── Optimizer kernels ───────────────────────
                        "sgd_update_f32" => {
                            let w_new = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                );
                                let g_slice = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                );
                                let lr = f32::from_bits(params[0] as u32);
                                let wd = if params.len() > 1 {
                                    f32::from_bits(params[1] as u32)
                                } else {
                                    0.0
                                };
                                sgd_update_f32(w_init, g_slice, lr, wd)
                            };
                            let d = arena.data_mut();
                            let w_off = input_slices[0].offset;
                            let w_sz = input_slices[0].size;
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[w_off..w_off + w_sz])
                                .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                        }
                        "gradient_scale" => {
                            if let Some(input_slice) = input_slices.first() {
                                let numel = *params.first().unwrap_or(&0);
                                let scale = f32::from_bits(*params.get(1).unwrap_or(&0) as u32);
                                let in_f32 = {
                                    let d = arena.data_mut();
                                    let d_ref: &[u8] = &*d;
                                    bytemuck::cast_slice::<_, f32>(
                                        &d_ref[input_slice.offset
                                            ..input_slice.offset + input_slice.size],
                                    )
                                    .to_vec()
                                };
                                let d = arena.data_mut();
                                let out_f32 =
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end]);
                                let len = out_f32.len().min(in_f32.len()).min(numel);
                                #[cfg(not(feature = "parallel"))]
                                {
                                    for i in 0..len {
                                        out_f32[i] = in_f32[i] * scale;
                                    }
                                }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len]
                                        .par_iter_mut()
                                        .enumerate()
                                        .for_each(|(i, o)| *o = in_f32[i] * scale);
                                }
                            }
                        }
                        "adam_update_f32" => {
                            let (w_new, m_new, v_new) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                );
                                let g_slice = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                );
                                let m_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[2].offset
                                        ..input_slices[2].offset + input_slices[2].size],
                                );
                                let v_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[3].offset
                                        ..input_slices[3].offset + input_slices[3].size],
                                );
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let eps = f32::from_bits(params[3] as u32);
                                let t = params[4] as f32;
                                let bias_corr1 = 1.0 - beta1.powi(t as i32);
                                let bias_corr2 = 1.0 - beta2.powi(t as i32);
                                adam_update_f32(
                                    w_init, g_slice, m_init, v_init, lr, beta1, beta2, eps,
                                    bias_corr1, bias_corr2,
                                )
                            };
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[0].offset
                                    ..input_slices[0].offset + input_slices[0].size],
                            )
                            .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[2].offset
                                    ..input_slices[2].offset + input_slices[2].size],
                            )
                            .copy_from_slice(&m_new);
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[3].offset
                                    ..input_slices[3].offset + input_slices[3].size],
                            )
                            .copy_from_slice(&v_new);
                        }
                        "adamw_update_f32" => {
                            let (w_new, m_new, v_new) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                );
                                let g_slice = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                );
                                let m_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[2].offset
                                        ..input_slices[2].offset + input_slices[2].size],
                                );
                                let v_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[3].offset
                                        ..input_slices[3].offset + input_slices[3].size],
                                );
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let eps = f32::from_bits(params[3] as u32);
                                let (t, wd) = if input_slices.len() >= 5 {
                                    // New path: t is a runtime tensor (5th input slice)
                                    let t = u64::from_le_bytes(
                                        d_ref[input_slices[4].offset..input_slices[4].offset + 8]
                                            .try_into()
                                            .unwrap(),
                                    ) as f32;
                                    let wd = f32::from_bits(params[4] as u32);
                                    (t, wd)
                                } else {
                                    // Old path: t is in params[4], wd in params[5]
                                    let t = params[4] as f32;
                                    let wd = f32::from_bits(params[5] as u32);
                                    (t, wd)
                                };
                                let bias_corr1 = 1.0 - beta1.powi(t as i32);
                                let bias_corr2 = 1.0 - beta2.powi(t as i32);
                                adamw_update_f32(
                                    w_init, g_slice, m_init, v_init, lr, beta1, beta2, eps,
                                    bias_corr1, bias_corr2, wd,
                                )
                            };
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[0].offset
                                    ..input_slices[0].offset + input_slices[0].size],
                            )
                            .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[2].offset
                                    ..input_slices[2].offset + input_slices[2].size],
                            )
                            .copy_from_slice(&m_new);
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[3].offset
                                    ..input_slices[3].offset + input_slices[3].size],
                            )
                            .copy_from_slice(&v_new);
                        }
                        "muon_update_f32" => {
                            let (w_new, m_new) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                );
                                let g_slice = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                );
                                let m_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[2].offset
                                        ..input_slices[2].offset + input_slices[2].size],
                                );
                                let lr = f32::from_bits(params[0] as u32);
                                let beta = f32::from_bits(params[1] as u32);
                                let wd = f32::from_bits(params[2] as u32);
                                muon_update_f32(w_init, g_slice, m_init, lr, beta, wd)
                            };
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[0].offset
                                    ..input_slices[0].offset + input_slices[0].size],
                            )
                            .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[2].offset
                                    ..input_slices[2].offset + input_slices[2].size],
                            )
                            .copy_from_slice(&m_new);
                        }
                        "lion_update_f32" => {
                            let (w_new, m_new) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                );
                                let g_slice = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                );
                                let m_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[2].offset
                                        ..input_slices[2].offset + input_slices[2].size],
                                );
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let wd = if params.len() > 3 {
                                    f32::from_bits(params[3] as u32)
                                } else {
                                    0.0
                                };
                                lion_update_f32(w_init, g_slice, m_init, lr, beta1, beta2, wd)
                            };
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[0].offset
                                    ..input_slices[0].offset + input_slices[0].size],
                            )
                            .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[2].offset
                                    ..input_slices[2].offset + input_slices[2].size],
                            )
                            .copy_from_slice(&m_new);
                        }
                        "rmsprop_update_f32" => {
                            let (w_new, v_new) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                );
                                let g_slice = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                );
                                let v_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[2].offset
                                        ..input_slices[2].offset + input_slices[2].size],
                                );
                                let lr = f32::from_bits(params[0] as u32);
                                let beta = f32::from_bits(params[1] as u32);
                                let eps = f32::from_bits(params[2] as u32);
                                rmsprop_update_f32(w_init, g_slice, v_init, lr, beta, eps)
                            };
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[0].offset
                                    ..input_slices[0].offset + input_slices[0].size],
                            )
                            .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[2].offset
                                    ..input_slices[2].offset + input_slices[2].size],
                            )
                            .copy_from_slice(&v_new);
                        }
                        // ── F16 state optimizer kernels ──────────────
                        // m and v are stored as F16 (2 bytes/elem), w and grad are F32 (4 bytes/elem).
                        // Read F16 state, convert to f32 internally, apply update, write back as F16.
                        "adam_update_f16_state" => {
                            let n = input_slices[0].size / 4;
                            let (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                )
                                .to_vec();
                                let m_raw: Vec<u16> = d_ref[input_slices[2].offset
                                    ..input_slices[2].offset + input_slices[2].size]
                                    .chunks_exact(2)
                                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                                    .collect();
                                let v_raw: Vec<u16> = d_ref[input_slices[3].offset
                                    ..input_slices[3].offset + input_slices[3].size]
                                    .chunks_exact(2)
                                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                                    .collect();
                                let m_init: Vec<f32> = m_raw
                                    .iter()
                                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                                    .collect();
                                let v_init: Vec<f32> = v_raw
                                    .iter()
                                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                                    .collect();
                                let grad = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                )
                                .to_vec();
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let eps = f32::from_bits(params[3] as u32);
                                let t = params[4] as f32;
                                (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t)
                            };
                            let bias_corr1 = 1.0 - beta1.powi(t as i32);
                            let bias_corr2 = 1.0 - beta2.powi(t as i32);
                            let len = n.min(w_init.len()).min(m_init.len()).min(v_init.len());
                            let mut w_new = w_init.clone();
                            let mut m_new_f32 = vec![0.0f32; len];
                            let mut v_new_f32 = vec![0.0f32; len];
                            #[cfg(not(feature = "parallel"))]
                            {
                                for i in 0..len {
                                    let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                    m_new_f32[i] = beta1 * m_init[i] + (1.0 - beta1) * g;
                                    v_new_f32[i] = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                    let m_hat = m_new_f32[i] / bias_corr1;
                                    let v_hat = v_new_f32[i] / bias_corr2;
                                    w_new[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                                }
                            }
                            #[cfg(feature = "parallel")]
                            {
                                use rayon::prelude::*;
                                w_new
                                    .par_iter_mut()
                                    .zip(m_new_f32.par_iter_mut())
                                    .zip(v_new_f32.par_iter_mut())
                                    .enumerate()
                                    .for_each(|(i, ((w, m), v))| {
                                        let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                        *m = beta1 * m_init[i] + (1.0 - beta1) * g;
                                        *v = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                        let m_hat = *m / bias_corr1;
                                        let v_hat = *v / bias_corr2;
                                        *w -= lr * m_hat / (v_hat.sqrt() + eps);
                                    });
                            }
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[0].offset
                                    ..input_slices[0].offset + input_slices[0].size],
                            )
                            .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                            let m_bytes: Vec<u8> = m_new_f32
                                .iter()
                                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                                .collect();
                            let v_bytes: Vec<u8> = v_new_f32
                                .iter()
                                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                                .collect();
                            let m_end = (input_slices[2].offset + m_bytes.len()).min(d.len());
                            let v_end = (input_slices[3].offset + v_bytes.len()).min(d.len());
                            d[input_slices[2].offset..m_end]
                                .copy_from_slice(&m_bytes[..m_end - input_slices[2].offset]);
                            d[input_slices[3].offset..v_end]
                                .copy_from_slice(&v_bytes[..v_end - input_slices[3].offset]);
                        }
                        "adamw_update_f16_state" => {
                            let n = input_slices[0].size / 4;
                            let (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t, wd) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                )
                                .to_vec();
                                let m_raw: Vec<u16> = d_ref[input_slices[2].offset
                                    ..input_slices[2].offset + input_slices[2].size]
                                    .chunks_exact(2)
                                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                                    .collect();
                                let v_raw: Vec<u16> = d_ref[input_slices[3].offset
                                    ..input_slices[3].offset + input_slices[3].size]
                                    .chunks_exact(2)
                                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                                    .collect();
                                let m_init: Vec<f32> = m_raw
                                    .iter()
                                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                                    .collect();
                                let v_init: Vec<f32> = v_raw
                                    .iter()
                                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                                    .collect();
                                let grad = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                )
                                .to_vec();
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let eps = f32::from_bits(params[3] as u32);
                                let (t, wd) = if input_slices.len() >= 5 {
                                    // New path: t is a runtime tensor (5th input slice)
                                    let t = u64::from_le_bytes(
                                        d_ref[input_slices[4].offset..input_slices[4].offset + 8]
                                            .try_into()
                                            .unwrap(),
                                    ) as f32;
                                    let wd = f32::from_bits(params[4] as u32);
                                    (t, wd)
                                } else {
                                    // Old path: t is in params[4], wd in params[5]
                                    let t = params[4] as f32;
                                    let wd = f32::from_bits(params[5] as u32);
                                    (t, wd)
                                };
                                (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t, wd)
                            };
                            let bias_corr1 = 1.0 - beta1.powi(t as i32);
                            let bias_corr2 = 1.0 - beta2.powi(t as i32);
                            let len = n.min(w_init.len()).min(m_init.len()).min(v_init.len());
                            let mut w_new = w_init.clone();
                            let mut m_new_f32 = vec![0.0f32; len];
                            let mut v_new_f32 = vec![0.0f32; len];
                            #[cfg(not(feature = "parallel"))]
                            {
                                for i in 0..len {
                                    w_new[i] -= lr * wd * w_init[i];
                                    let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                    m_new_f32[i] = beta1 * m_init[i] + (1.0 - beta1) * g;
                                    v_new_f32[i] = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                    let m_hat = m_new_f32[i] / bias_corr1;
                                    let v_hat = v_new_f32[i] / bias_corr2;
                                    w_new[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                                }
                            }
                            #[cfg(feature = "parallel")]
                            {
                                use rayon::prelude::*;
                                w_new
                                    .par_iter_mut()
                                    .zip(m_new_f32.par_iter_mut())
                                    .zip(v_new_f32.par_iter_mut())
                                    .enumerate()
                                    .for_each(|(i, ((w, m), v))| {
                                        *w -= lr * wd * w_init[i];
                                        let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                        *m = beta1 * m_init[i] + (1.0 - beta1) * g;
                                        *v = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                        let m_hat = *m / bias_corr1;
                                        let v_hat = *v / bias_corr2;
                                        *w -= lr * m_hat / (v_hat.sqrt() + eps);
                                    });
                            }
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[0].offset
                                    ..input_slices[0].offset + input_slices[0].size],
                            )
                            .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                            let m_bytes: Vec<u8> = m_new_f32
                                .iter()
                                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                                .collect();
                            let v_bytes: Vec<u8> = v_new_f32
                                .iter()
                                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                                .collect();
                            let m_end = (input_slices[2].offset + m_bytes.len()).min(d.len());
                            let v_end = (input_slices[3].offset + v_bytes.len()).min(d.len());
                            d[input_slices[2].offset..m_end]
                                .copy_from_slice(&m_bytes[..m_end - input_slices[2].offset]);
                            d[input_slices[3].offset..v_end]
                                .copy_from_slice(&v_bytes[..v_end - input_slices[3].offset]);
                        }
                        "cast" => {
                            let in_byte_size = *params.first().unwrap_or(&4);
                            let out_byte_size = *params.get(1).unwrap_or(&4);
                            if let Some(input_slice) = input_slices.first() {
                                let d = arena.data_mut();
                                let in_data =
                                    &d[input_slice.offset..input_slice.offset + input_slice.size];
                                if in_byte_size == 4 && out_byte_size == 8 {
                                    // F32/I32 → I64: widen
                                    let in_f32 = bytemuck::cast_slice::<_, f32>(in_data).to_vec();
                                    let mut out_bytes = Vec::with_capacity(in_f32.len() * 8);
                                    for &v in &in_f32 {
                                        out_bytes.extend_from_slice(&(v as i64).to_le_bytes());
                                    }
                                    let end = (out_start + out_bytes.len()).min(d.len());
                                    d[out_start..end]
                                        .copy_from_slice(&out_bytes[..end - out_start]);
                                } else if in_byte_size == 8 && out_byte_size == 4 {
                                    // I64 → F32/I32: narrow
                                    let in_i64 = bytemuck::cast_slice::<_, i64>(in_data).to_vec();
                                    let mut out_bytes = Vec::with_capacity(in_i64.len() * 4);
                                    for &v in &in_i64 {
                                        out_bytes.extend_from_slice(&(v as f32).to_le_bytes());
                                    }
                                    let end = (out_start + out_bytes.len()).min(d.len());
                                    d[out_start..end]
                                        .copy_from_slice(&out_bytes[..end - out_start]);
                                }
                                // Same-size casts handled by MemCopy at compile time
                            }
                        }
                        "expand_f32" => {
                            // Expand broadcasts input[0] (f32 data) using target
                            // shape input[1] (i64 dims).  params layout:
                            //   [max_rank, in_d0..in_dN, out_d0..out_dN]
                            if input_slices.len() < 2 {
                                return Err(BackendError::Dispatch(
                                    "expand_f32 needs 2 inputs (data + shape)".into(),
                                ));
                            }
                            let max_rank = *params.first().ok_or_else(|| {
                                BackendError::Dispatch("expand_f32: missing max_rank".into())
                            })?;
                            if params.len() < 1 + max_rank * 2 {
                                return Err(BackendError::Dispatch(format!(
                                    "expand_f32: expected {} params, got {}",
                                    1 + max_rank * 2,
                                    params.len()
                                )));
                            }
                            // Extract padded input dims and output dims
                            let in_dims: Vec<usize> = params[1..1 + max_rank].to_vec();
                            let out_dims: Vec<usize> =
                                params[1 + max_rank..1 + max_rank * 2].to_vec();

                            let data_slice = &input_slices[0];
                            let _shape_slice = &input_slices[1];
                            let data_numel = data_slice.size / 4; // f32 = 4 bytes
                            let out_numel = output_slice.size / 4;

                            let d = arena.data_mut();

                            // NOTE: We do NOT read the runtime shape tensor here.
                            // The compile-time broadcast dims from the instruction params
                            // (in_dims, out_dims) are the source of truth.  The shape
                            // tensor (input[1]) is stored as F32 (4 bytes/elem) by the
                            // Shape/Gather/Concat pipeline, but the old code attempted to
                            // read it as i64 (8 bytes/elem) — a latent bytemuck panic for
                            // tensors with odd element counts (e.g. 3 dims → 12 bytes,
                            // not a multiple of 8).

                            // Read input data
                            let in_f32: Vec<f32> = bytemuck::cast_slice::<_, f32>(
                                &d[data_slice.offset..data_slice.offset + data_slice.size],
                            )
                            .to_vec();

                            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[out_start..out_start + output_slice.size],
                            );

                            // Broadcast: for each output element, map back to input coords
                            #[cfg(not(feature = "parallel"))]
                            {
                                for out_linear in 0..out_numel {
                                    let mut out_coord = vec![0usize; max_rank];
                                    let mut remaining = out_linear;
                                    for i in (0..max_rank).rev() {
                                        out_coord[i] = remaining % out_dims[i];
                                        remaining /= out_dims[i];
                                    }
                                    let mut in_linear: usize = 0;
                                    let mut in_stride = 1usize;
                                    for i in (0..max_rank).rev() {
                                        let in_dim = in_dims[i];
                                        let out_dim = out_dims[i];
                                        let in_coord = if in_dim == out_dim {
                                            out_coord[i]
                                        } else if in_dim == 1 {
                                            0
                                        } else {
                                            panic!("expand_f32: invalid broadcast: input dim {} cannot expand to output dim {} (must be 1 or match)", in_dim, out_dim);
                                        };
                                        in_linear += in_coord * in_stride;
                                        in_stride *= in_dim;
                                    }
                                    if in_linear < data_numel {
                                        out_f32[out_linear] = in_f32[in_linear];
                                    }
                                }
                            }
                            #[cfg(feature = "parallel")]
                            {
                                use rayon::prelude::*;
                                let max_rank = max_rank;
                                let out_addr = out_f32.as_mut_ptr() as usize;
                                (0..out_numel).into_par_iter().for_each(|out_linear| {
                                    let mut out_coord = vec![0usize; max_rank];
                                    let mut remaining = out_linear;
                                    for i in (0..max_rank).rev() {
                                        out_coord[i] = remaining % out_dims[i];
                                        remaining /= out_dims[i];
                                    }
                                    let mut in_linear: usize = 0;
                                    let mut in_stride = 1usize;
                                    for i in (0..max_rank).rev() {
                                        let in_dim = in_dims[i];
                                        let out_dim = out_dims[i];
                                        let in_coord = if in_dim == out_dim {
                                            out_coord[i]
                                        } else if in_dim == 1 {
                                            0
                                        } else {
                                            panic!("expand_f32: invalid broadcast: input dim {} cannot expand to output dim {} (must be 1 or match)", in_dim, out_dim);
                                        };
                                        in_linear += in_coord * in_stride;
                                        in_stride *= in_dim;
                                    }
                                    if in_linear < data_numel {
                                        unsafe {
                                            *(out_addr as *mut f32).add(out_linear) =
                                                in_f32[in_linear];
                                        }
                                    }
                                });
                            }
                        }
                        "range_f32" => {
                            // Range(start, limit, step): produce 1D F32 tensor.
                            let d = arena.data_mut();
                            let start_val = if let Some(s) = input_slices.first() {
                                f32::from_le_bytes(
                                    d[s.offset..s.offset + 4].try_into().unwrap_or([0u8; 4]),
                                )
                            } else {
                                0.0
                            };
                            let limit_val = if let Some(s) = input_slices.get(1) {
                                f32::from_le_bytes(
                                    d[s.offset..s.offset + 4].try_into().unwrap_or([0u8; 4]),
                                )
                            } else {
                                0.0
                            };
                            let step_val = if let Some(s) = input_slices.get(2) {
                                f32::from_le_bytes(
                                    d[s.offset..s.offset + 4].try_into().unwrap_or([1u8; 4]),
                                )
                            } else {
                                1.0
                            };
                            let n = if step_val > 0.0 {
                                ((limit_val - start_val) / step_val).ceil().max(0.0) as usize
                            } else if step_val < 0.0 {
                                ((start_val - limit_val) / (-step_val)).ceil().max(0.0) as usize
                            } else {
                                0
                            };
                            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[out_start..out_start + output_slice.size],
                            );
                            let actual_len = out_f32.len().min(n);
                            for i in 0..actual_len {
                                out_f32[i] = start_val + i as f32 * step_val;
                            }
                        }
                        "quantize_f32_u4" | "quantize_f32_u8" => {
                            let num_channels = *params.first().unwrap_or(&1);
                            let num_elems_per_channel = *params.get(1).unwrap_or(&1);
                            let numel = *params.get(2).unwrap_or(&1);
                            let bit_width = if kernel_name == "quantize_f32_u4" {
                                4
                            } else {
                                8
                            };
                            let max_q = (1i32 << (bit_width - 1)) - 1; // 7 for U4, 127 for U8
                            let items_per_word = 32 / bit_width; // 8 for U4, 4 for U8

                            // Check for cached scales from wrap_quantized_optimizer
                            let has_cached = params.get(3).copied().unwrap_or(0) == 1;
                            let mut cached_scales = vec![];
                            let mut cached_zeros = vec![];
                            if has_cached {
                                let sc_start = 4;
                                let sc_end = sc_start + num_channels;
                                let zp_start = sc_end;
                                let zp_end = zp_start + num_channels;
                                for i in sc_start..sc_end {
                                    let bits = *params.get(i).unwrap_or(&0);
                                    cached_scales.push(f32::from_bits(bits as u32));
                                }
                                for i in zp_start..zp_end {
                                    let bits = *params.get(i).unwrap_or(&0);
                                    cached_zeros.push(f32::from_bits(bits as u32));
                                }
                            }

                            if let Some(input_slice) = input_slices.first() {
                                let d = arena.data_mut();
                                let f32_data: Vec<f32> = bytemuck::cast_slice::<_, f32>(
                                    &d[input_slice.offset..input_slice.offset + input_slice.size],
                                )
                                .to_vec();

                                let mut scales: Vec<f32> = Vec::with_capacity(num_channels);
                                let mut zero_points: Vec<f32> = vec![0.0; num_channels];
                                let packed_words = numel.div_ceil(items_per_word);
                                let mut packed: Vec<u32> = vec![0u32; packed_words];

                                if has_cached
                                    && cached_scales.len() == num_channels
                                    && cached_zeros.len() == num_channels
                                {
                                    scales = cached_scales;
                                    zero_points = cached_zeros;
                                    // Pack using cached scales
                                    for ch in 0..num_channels {
                                        let start = ch * num_elems_per_channel;
                                        let end = (start + num_elems_per_channel).min(f32_data.len());
                                        let scale = scales[ch];
                                        let inv_s = if scale != 0.0 { 1.0 / scale } else { 0.0 };
                                        let zp = zero_points[ch];
                                        for j in start..end {
                                            let q = ((f32_data[j] - zp) * inv_s)
                                                .round()
                                                .clamp(-(max_q as f32), max_q as f32)
                                                as i32;
                                            let word_idx = j / items_per_word;
                                            let shift = (j % items_per_word) * bit_width;
                                            packed[word_idx] |= ((q as u32) & ((1 << bit_width) - 1)) << shift;
                                        }
                                    }
                                } else {
                                    // Original path: recompute per-channel scales
                                    for ch in 0..num_channels {
                                        let start = ch * num_elems_per_channel;
                                        let end = (start + num_elems_per_channel).min(f32_data.len());
                                        let max_abs = f32_data[start..end]
                                            .iter()
                                            .map(|v| v.abs())
                                            .fold(0.0f32, f32::max);
                                        let scale = if max_abs == 0.0 {
                                            1.0
                                        } else {
                                            max_abs / max_q as f32
                                        };
                                        scales.push(scale);

                                        // Quantize and pack
                                        for j in start..end {
                                            let q = (f32_data[j] / scale)
                                                .round()
                                                .clamp(-(max_q as f32), max_q as f32)
                                                as i32;
                                            let word_idx = j / items_per_word;
                                            let shift = (j % items_per_word) * bit_width;
                                            packed[word_idx] |=
                                                ((q as u32) & ((1 << bit_width) - 1)) << shift;
                                        }
                                    }
                                }

                                // Write output: [num_channels(u32)][num_elems_per_channel(u32)]
                                //             [scales(f32 x N)][zero_points(f32 x N)][packed_data]
                                let header_size = 8 + 8 * num_channels; // 2 u32 + N f32 + N f32
                                let total_size = header_size + packed.len() * 4;
                                let out_end = (out_start + total_size).min(d.len());
                                let out = &mut d[out_start..out_end];

                                let mut offset = 0;
                                out[offset..offset + 4]
                                    .copy_from_slice(&(num_channels as u32).to_le_bytes());
                                offset += 4;
                                out[offset..offset + 4]
                                    .copy_from_slice(&(num_elems_per_channel as u32).to_le_bytes());
                                offset += 4;
                                for &s in &scales {
                                    out[offset..offset + 4].copy_from_slice(&s.to_le_bytes());
                                    offset += 4;
                                }
                                for &z in &zero_points {
                                    out[offset..offset + 4].copy_from_slice(&z.to_le_bytes());
                                    offset += 4;
                                }
                                for &w in &packed {
                                    out[offset..offset + 4].copy_from_slice(&w.to_le_bytes());
                                    offset += 4;
                                }
                            }
                        }
                        "dequantize_kernel" => {
                            if let Some(input_slice) = input_slices.first() {
                                let numel = *params.first().unwrap_or(&0);
                                let format_flag = *params.get(1).unwrap_or(&0); // 0=header, 1=metadata
                                let in_data = {
                                    let d = arena.data_mut();
                                    d[input_slice.offset..input_slice.offset + input_slice.size]
                                        .to_vec()
                                };

                                let (
                                    num_channels,
                                    num_elems_per_channel,
                                    scales,
                                    zero_points,
                                    data_offset,
                                    bit_width,
                                ) = if format_flag == 1 {
                                    // Metadata-based: scales/zero_points passed as params
                                    let num_channels = *params.get(2).unwrap_or(&0);
                                    let mut scales: Vec<f32> = Vec::with_capacity(num_channels);
                                    for j in 0..num_channels {
                                        let bits = *params.get(3 + j).unwrap_or(&0);
                                        scales.push(f32::from_bits(bits as u32));
                                    }
                                    let mut zero_points: Vec<f32> =
                                        Vec::with_capacity(num_channels);
                                    for j in 0..num_channels {
                                        let bits = *params.get(3 + num_channels + j).unwrap_or(&0);
                                        zero_points.push(f32::from_bits(bits as u32));
                                    }
                                    // The packed data starts at offset 0 (no header)
                                    let data_offset = 0;
                                    // Infer num_elems_per_channel from numel
                                    let num_elems_per_channel = if num_channels > 0 {
                                        numel / num_channels
                                    } else {
                                        numel
                                    };
                                    // Infer bit_width from packed byte ratio
                                    let total_packed_bytes =
                                        in_data.len().saturating_sub(data_offset);
                                    let packed_words = total_packed_bytes / 4;
                                    let bit_width = if packed_words > 0 && numel > 0 {
                                        let ratio = (packed_words * 4) as f64 / numel as f64;
                                        if ratio < 0.6 {
                                            4
                                        } else {
                                            8
                                        }
                                    } else {
                                        8
                                    };
                                    (
                                        num_channels,
                                        num_elems_per_channel,
                                        scales,
                                        zero_points,
                                        data_offset,
                                        bit_width,
                                    )
                                } else {
                                    // Header-based: parse [num_channels][num_elems][scales...][zps...][packed_data]
                                    let num_channels = u32::from_le_bytes(
                                        in_data[0..4].try_into().unwrap_or([0u8; 4]),
                                    )
                                        as usize;
                                    let num_elems_per_channel = u32::from_le_bytes(
                                        in_data[4..8].try_into().unwrap_or([0u8; 4]),
                                    )
                                        as usize;
                                    let mut hdr_offset = 8usize;
                                    let mut scales: Vec<f32> = Vec::with_capacity(num_channels);
                                    for _ in 0..num_channels {
                                        if hdr_offset + 4 <= in_data.len() {
                                            let s = f32::from_le_bytes(
                                                in_data[hdr_offset..hdr_offset + 4]
                                                    .try_into()
                                                    .unwrap(),
                                            );
                                            scales.push(s);
                                            hdr_offset += 4;
                                        }
                                    }
                                    // zero_points (currently all zero for symmetric quant)
                                    for _ in 0..num_channels {
                                        if hdr_offset + 4 <= in_data.len() {
                                            hdr_offset += 4;
                                        }
                                    }
                                    let zero_points: Vec<f32> = vec![0.0; num_channels];
                                    let data_offset = hdr_offset;
                                    let total_packed_bytes =
                                        in_data.len().saturating_sub(data_offset);
                                    let packed_words = total_packed_bytes / 4;
                                    let bit_width = if packed_words > 0 && numel > 0 {
                                        let ratio = (packed_words * 4) as f64 / numel as f64;
                                        if ratio < 0.6 {
                                            4
                                        } else {
                                            8
                                        }
                                    } else {
                                        8
                                    };
                                    (
                                        num_channels,
                                        num_elems_per_channel,
                                        scales,
                                        zero_points,
                                        data_offset,
                                        bit_width,
                                    )
                                };

                                let items_per_word = 32 / bit_width;
                                let total_packed_bytes = in_data.len().saturating_sub(data_offset);
                                let packed_words = total_packed_bytes / 4;

                                // Write output
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let max_out = out_f32.len().min(numel);
                                for i in 0..max_out {
                                    let ch = if num_channels > 0 {
                                        i / num_elems_per_channel % num_channels
                                    } else {
                                        0
                                    };
                                    let word_idx = i / items_per_word;
                                    let shift = (i % items_per_word) * bit_width;
                                    if word_idx < packed_words {
                                        let word_start = data_offset + word_idx * 4;
                                        let word = if word_start + 4 <= in_data.len() {
                                            u32::from_le_bytes(
                                                in_data[word_start..word_start + 4]
                                                    .try_into()
                                                    .unwrap(),
                                            )
                                        } else {
                                            0
                                        };
                                        let q = ((word >> shift) & ((1 << bit_width) - 1)) as i32;
                                        // Sign-extend for signed types
                                        let sign_bit = 1 << (bit_width - 1);
                                        let q_signed = if (q & sign_bit) != 0 {
                                            q | (!((1 << bit_width) - 1))
                                        } else {
                                            q
                                        };
                                        let scale = scales.get(ch).copied().unwrap_or(1.0);
                                        let zp = zero_points.get(ch).copied().unwrap_or(0.0);
                                        out_f32[i] = q_signed as f32 * scale + zp;
                                    }
                                }
                            }
                        }
                        "to_f16" => {
                            if let Some(input_slice) = input_slices.first() {
                                let f32_data = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset
                                            ..input_slice.offset + input_slice.size],
                                    )
                                    .to_vec()
                                };
                                let out_bytes = {
                                    let d = arena.data_mut();
                                    &mut d[out_start..out_end]
                                };
                                for (i, &v) in f32_data.iter().enumerate() {
                                    let f16_val = half::f16::from_f32(v);
                                    let bytes = f16_val.to_le_bytes();
                                    let start = i * 2;
                                    let end = (start + 2).min(out_bytes.len());
                                    if start < out_bytes.len() {
                                        out_bytes[start..end]
                                            .copy_from_slice(&bytes[..end - start]);
                                    }
                                }
                            }
                        }
                        "to_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let in_data = {
                                    let d = arena.data_mut();
                                    d[input_slice.offset..input_slice.offset + input_slice.size]
                                        .to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let max_out = out_f32.len().min(in_data.len() / 2);
                                for i in 0..max_out {
                                    let start = i * 2;
                                    if start + 2 <= in_data.len() {
                                        let f16_val = half::f16::from_le_bytes(
                                            in_data[start..start + 2].try_into().unwrap(),
                                        );
                                        out_f32[i] = f16_val.to_f32();
                                    }
                                }
                            }
                        }
                        "quantize_activations" => {
                            let numel = *params.first().unwrap_or(&0);
                            if let Some(input_slice) = input_slices.first() {
                                let f32_data = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset
                                            ..input_slice.offset + input_slice.size],
                                    )
                                    .to_vec()
                                };
                                // Symmetric INT8 quantization: scale = max_abs / 127
                                let max_abs =
                                    f32_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                                let zp = 0.0f32;

                                let out_bytes = {
                                    let d = arena.data_mut();
                                    &mut d[out_start..out_end]
                                };
                                // Write scale, zp, then quantized i8 data
                                let header_size = 8; // scale + zp
                                out_bytes[0..4].copy_from_slice(&scale.to_le_bytes());
                                out_bytes[4..8].copy_from_slice(&zp.to_le_bytes());
                                let max_out = (out_bytes.len() - header_size).min(numel);
                                for i in 0..max_out {
                                    let q =
                                        (f32_data[i] / scale).round().clamp(-128.0, 127.0) as i8;
                                    out_bytes[header_size + i] = q as u8;
                                }
                            }
                        }
                        "dequantize_activations" => {
                            let numel = *params.first().unwrap_or(&0);
                            if let Some(input_slice) = input_slices.first() {
                                let in_data = {
                                    let d = arena.data_mut();
                                    d[input_slice.offset..input_slice.offset + input_slice.size]
                                        .to_vec()
                                };
                                let header_size = 8;
                                let scale = if in_data.len() >= 4 {
                                    f32::from_le_bytes(in_data[0..4].try_into().unwrap())
                                } else {
                                    1.0
                                };
                                let zp = if in_data.len() >= 8 {
                                    f32::from_le_bytes(in_data[4..8].try_into().unwrap())
                                } else {
                                    0.0
                                };

                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let max_out = out_f32.len().min(numel);
                                for i in 0..max_out {
                                    let idx = header_size + i;
                                    let q = if idx < in_data.len() {
                                        in_data[idx] as i8
                                    } else {
                                        0i8
                                    };
                                    out_f32[i] = (q as f32) * scale + zp;
                                }
                            }
                        }
                        _ => {
                            return Err(BackendError::UnsupportedOp(kernel_name.clone()));
                        }
                    }
                }
                Instruction::MemCopy { dst, src } => {
                    let data = arena.data_mut();
                    let src_start = src.offset;
                    let dst_start = dst.offset;
                    let len = dst.size.min(src.size);
                    let src_range = src_start..src_start + len;
                    data.copy_within(src_range, dst_start);
                }
                Instruction::Fill { dst, value } => {
                    let data = arena.data_mut();
                    let start = dst.offset;
                    let end = dst.offset + dst.size;
                    let bytes = &mut data[start..end];
                    let f32_slice = bytemuck::cast_slice_mut::<_, f32>(bytes);
                    f32_slice.fill(*value);
                }
                Instruction::WriteConst { dst, data } => {
                    let arena_data = arena.data_mut();
                    let end = (dst.offset + data.len()).min(arena_data.len());
                    arena_data[dst.offset..end].copy_from_slice(&data[..end - dst.offset]);
                }
            }

            // ── Debug: per-instruction MaxPool canary check ──────────
            // After the current instruction has executed, check whether
            // any MaxPool primary slot has been overwritten.
            // Only active with `debug_canary` feature (expensive).
            #[cfg(feature = "debug_canary")]
            {
                let d = arena.data_mut();
                // Determine if this instruction IS a MaxPool kernel, and
                // if so, which index in maxpool_ranges it corresponds to.
                let is_mp_and_idx: Option<usize> = match instr {
                    Instruction::CallKernel {
                        kernel_name,
                        params,
                        output_slice,
                        ..
                    } if kernel_name == "pool_f32" && params.len() >= 4 && params[3] == 1 => {
                        maxpool_ranges
                            .iter()
                            .position(|&(off, _)| off == output_slice.offset)
                    }
                    _ => None,
                };

                if let Some(mp_idx) = is_mp_and_idx {
                    // This instruction just wrote MaxPool data → snapshot
                    let (mp_off, mp_sz) = maxpool_ranges[mp_idx];
                    if mp_sz >= 4 && mp_off + 4 <= d.len() {
                        let bytes: [u8; 4] = d[mp_off..mp_off + 4].try_into().unwrap_or([0; 4]);
                        maxpool_snapshot[mp_idx] = Some(f32::from_le_bytes(bytes));
                        maxpool_seen[mp_idx] = true;
                        // Also log the kernel's output_slice for reference
                        if let Instruction::CallKernel {
                            kernel_name,
                            output_slice,
                            ..
                        } = instr
                        {
                            eprintln!(
                                "[FNN_DBG_CANARY] MaxPool nid={}: off={} sz={} first_f32={} (AFTER kernel={} out=[{},{})",
                                mp_idx, mp_off, mp_sz,
                                maxpool_snapshot[mp_idx].unwrap(),
                                kernel_name,
                                output_slice.offset,
                                output_slice.offset + output_slice.size,
                            );
                        }
                    }
                } else {
                    // Not a MaxPool kernel — check if any MaxPool was corrupted
                    for (mp_idx, &(mp_off, mp_sz)) in maxpool_ranges.iter().enumerate() {
                        if !maxpool_seen[mp_idx] {
                            continue; // MaxPool hasn't executed yet
                        }
                        if let Some(expected) = maxpool_snapshot[mp_idx] {
                            if mp_sz >= 4 && mp_off + 4 <= d.len() {
                                let bytes: [u8; 4] =
                                    d[mp_off..mp_off + 4].try_into().unwrap_or([0; 4]);
                                let actual = f32::from_le_bytes(bytes);
                                if actual.to_bits() != expected.to_bits() {
                                    let desc = match instr {
                                        Instruction::CallKernel {
                                            kernel_name,
                                            output_slice,
                                            ..
                                        } => format!(
                                            "kernel={} out=[{},{})",
                                            kernel_name,
                                            output_slice.offset,
                                            output_slice.offset + output_slice.size
                                        ),
                                        Instruction::MemCopy { dst, src } => {
                                            format!(
                                                "MemCopy dst=[{},{}) src=[{},{})",
                                                dst.offset,
                                                dst.offset + dst.size,
                                                src.offset,
                                                src.offset + src.size
                                            )
                                        }
                                        Instruction::Fill { dst, value } => {
                                            format!(
                                                "Fill dst=[{},{}) value={}",
                                                dst.offset,
                                                dst.offset + dst.size,
                                                value
                                            )
                                        }
                                        Instruction::WriteConst { dst, .. } => {
                                            format!(
                                                "WriteConst dst=[{},{})",
                                                dst.offset,
                                                dst.offset + dst.size
                                            )
                                        }
                                    };
                                    eprintln!(
                                        "[FNN_DBG_CORRUPT] MaxPool mp_off={} expected={} actual={} AFTER instr_idx={} {}",
                                        mp_off, expected, actual, _instr_idx, desc
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn write_arena(&self, arena: &CpuBuffer, offset: usize, data: &[u8]) {
        let buf = arena.data_mut();
        let end = (offset + data.len()).min(buf.len());
        buf[offset..end].copy_from_slice(&data[..end - offset]);
    }

    fn read_arena(&self, arena: &CpuBuffer, offset: usize, size: usize) -> Vec<u8> {
        let buf = arena.data_mut();
        let end = (offset + size).min(buf.len());
        buf[offset..end].to_vec()
    }

    #[cfg(feature = "prepared-plan")]
    fn dispatch_with_persistent_view(
        &self,
        plan: &ExecutablePlan,
        arena: &CpuBuffer,
        shape_env: &ShapeEnv,
        persistent_view: Option<&crate::backend::prepared::PersistentPreparedWeights>,
    ) -> Result<(), BackendError> {
        use crate::backend::prepared::PersistentPreparedWeights;

        // Fast path: empty / missing view degrades to the standard
        // dispatch without any per-instruction overhead.
        let view: &PersistentPreparedWeights = match persistent_view {
            Some(v) if !v.is_empty() => v,
            _ => return self.dispatch(plan, arena, shape_env),
        };

        for instruction in &plan.instructions {
            match instruction {
                // Skip WriteConst for slots the persistent view will
                // satisfy directly.  The slot offset/size is the
                // exact byte range that the WriteConst would have
                // materialised; the dispatch kernel will read the
                // same range from the persistent payload instead.
                Instruction::WriteConst { dst, .. }
                    if view.get(&(dst.offset, dst.size)).is_some() =>
                {
                    continue;
                }

                // Conv2d fp32 + (optional) bias.  Resolved directly
                // from the persistent view when available; otherwise
                // falls through to the standard per-instruction
                // dispatch path which already handles the
                // input/weight/bias reads from the mutable arena.
                Instruction::CallKernel {
                    kernel_name,
                    input_slices,
                    output_slice,
                    params,
                    param_dims,
                    weight_meta,
                    node_id,
                    ..
                } if matches!(
                    kernel_name.as_str(),
                    "conv2d" | "conv2d_relu" | "conv2d_gelu" | "conv2d_silu"
                ) =>
                {
                    let has_override = input_slices
                        .get(1)
                        .map(|w| view.get(&(w.offset, w.size)).is_some())
                        .unwrap_or(false)
                        || input_slices
                            .get(2)
                            .map(|b| view.get(&(b.offset, b.size)).is_some())
                            .unwrap_or(false);
                    if has_override {
                        dispatch_conv2d_fp32_with_view(
                            arena,
                            shape_env,
                            kernel_name,
                            input_slices,
                            *output_slice,
                            params,
                            node_id.unwrap_or(0),
                            view,
                        )?;
                    } else {
                        let single = ExecutablePlan {
                            instructions: vec![instruction.clone()],
                            arena_size: plan.arena_size,
                            levels: vec![0],
                        };
                        self.dispatch(&single, arena, shape_env)?;
                    }
                }

                // Fp32 MatMul family (matmul, matmul_relu/gelu/silu,
                // fused_matmul_add_*).  Resolved from the persistent
                // view for both the B slot and the optional bias slot.
                Instruction::CallKernel {
                    kernel_name,
                    input_slices,
                    output_slice,
                    params,
                    param_dims,
                    weight_meta,
                    node_id,
                    ..
                } if matches!(
                    kernel_name.as_str(),
                    "matmul"
                        | "matmul_relu"
                        | "matmul_gelu"
                        | "matmul_silu"
                        | "fused_matmul_add_relu"
                        | "fused_matmul_add_gelu"
                        | "fused_matmul_add_silu"
                ) =>
                {
                    let has_override = input_slices
                        .get(1)
                        .map(|b| view.get(&(b.offset, b.size)).is_some())
                        .unwrap_or(false)
                        || input_slices
                            .get(2)
                            .map(|b| view.get(&(b.offset, b.size)).is_some())
                            .unwrap_or(false);
                    if has_override {
                        dispatch_matmul_fp32_with_view(
                            arena,
                            shape_env,
                            kernel_name,
                            input_slices,
                            *output_slice,
                            params,
                            param_dims,
                            view,
                        )?;
                    } else {
                        let single = ExecutablePlan {
                            instructions: vec![instruction.clone()],
                            arena_size: plan.arena_size,
                            levels: vec![0],
                        };
                        self.dispatch(&single, arena, shape_env)?;
                    }
                }

                // Quantized MatMul family (u4, u8, and i8-activation
                // variants).  Same persistent-view override logic as
                // the fp32 path above.
                Instruction::CallKernel {
                    kernel_name,
                    input_slices,
                    output_slice,
                    params,
                    param_dims,
                    weight_meta,
                    node_id,
                    ..
                } if matches!(
                    kernel_name.as_str(),
                    "matmul_u4"
                        | "matmul_u4_i8"
                        | "matmul_u8"
                        | "matmul_u8_i8"
                ) =>
                {
                    let has_override = input_slices
                        .get(1)
                        .map(|b| view.get(&(b.offset, b.size)).is_some())
                        .unwrap_or(false)
                        || input_slices
                            .get(2)
                            .map(|b| view.get(&(b.offset, b.size)).is_some())
                            .unwrap_or(false);
                    if has_override {
                        dispatch_matmul_quantized_with_view(
                            arena,
                            shape_env,
                            kernel_name,
                            input_slices,
                            *output_slice,
                            params,
                            param_dims,
                            weight_meta,
                            view,
                        )?;
                    } else {
                        let single = ExecutablePlan {
                            instructions: vec![instruction.clone()],
                            arena_size: plan.arena_size,
                            levels: vec![0],
                        };
                        self.dispatch(&single, arena, shape_env)?;
                    }
                }

                // Everything else: defer to the standard per-instruction
                // dispatch so we keep the existing single-threaded,
                // sequential behaviour for the rest of the plan.
                _ => {
                    let single = ExecutablePlan {
                        instructions: vec![instruction.clone()],
                        arena_size: plan.arena_size,
                        levels: vec![0],
                    };
                    self.dispatch(&single, arena, shape_env)?;
                }
            }
        }
        Ok(())
    }
}

// ── Persistent-view dispatch helpers ────────────────────────
//
// These helpers are private to the CpuBackend. They are intentionally
// near-clones of the corresponding branches in `dispatch()` so the
// fp32 Conv2d / MatMul kernels can borrow the weight/bias bytes
// directly from the [`PersistentPreparedWeights`] view instead of
// pulling them out of the mutable arena.  The input tensor is still
// read from the mutable arena; only the static weight / bias slots
// are routed through the view.

#[cfg(feature = "prepared-plan")]
fn dispatch_conv2d_fp32_with_view(
    arena: &CpuBuffer,
    _shape_env: &ShapeEnv,
    kernel_name: &str,
    input_slices: &[BufferSlice],
    output_slice: BufferSlice,
    params: &[usize],
    _node_id: usize,
    view: &crate::backend::prepared::PersistentPreparedWeights,
) -> Result<(), BackendError> {
    use crate::backend::cpu::microkernels::{conv2d_f32_im2col_gemm, ConvActivation};

    let fused_act: Option<ConvActivation> = match kernel_name {
        "conv2d_relu" => Some(ConvActivation::Relu),
        "conv2d_gelu" => Some(ConvActivation::Gelu),
        "conv2d_silu" => Some(ConvActivation::Silu),
        _ => None,
    };

    if input_slices.len() < 2 {
        return Err(BackendError::Dispatch(format!(
            "conv2d_persistent: expected ≥ 2 input slices, got {}",
            input_slices.len()
        )));
    }
    let input_slice = input_slices[0];
    let weight_slice = input_slices[1];
    let bias_slice = input_slices.get(2).copied();

    let &[stride, padding, dilation, groups, c, h, w, kh, kw] = params else {
        return Err(BackendError::Dispatch(
            "conv2d_persistent: expected params [stride, padding, dilation, groups, c, h, w, kh, kw]"
                .into(),
        ));
    };
    let c_per_group = c / groups.max(1);
    let f32_size = std::mem::size_of::<f32>();
    let n_in = (input_slice.size / f32_size) / (c * h * w).max(1);
    let f_out = (weight_slice.size / f32_size) / (c_per_group * kh * kw).max(1);
    let _h_out = (h + 2 * padding).saturating_sub(dilation * (kh - 1) + 1) / stride + 1;
    let _w_out = (w + 2 * padding).saturating_sub(dilation * (kw - 1) + 1) / stride + 1;

    // Resolve the weight / bias f32 slices.  Persistent-view entries
    // are borrowed directly (no copy); non-overridden slots fall
    // back to a Vec copy of the arena bytes (these are rare in
    // practice — the no-copy plan only filters WriteConst for
    // overridden slots, so any non-overridden slot still has its
    // WriteConst running and the arena bytes are valid).
    let weight_f32: Vec<f32> = match view.get(&(weight_slice.offset, weight_slice.size)) {
        Some(payload) => payload.to_vec(),
        None => {
            let d = arena.data_mut();
            bytemuck::cast_slice::<_, f32>(
                &d[weight_slice.offset..weight_slice.offset + weight_slice.size],
            )
            .to_vec()
        }
    };
    let bias_f32: Vec<f32> = if let Some(b) = bias_slice {
        match view.get(&(b.offset, b.size)) {
            Some(payload) => payload.to_vec(),
            None => {
                let d = arena.data_mut();
                bytemuck::cast_slice::<_, f32>(&d[b.offset..b.offset + b.size]).to_vec()
            }
        }
    } else {
        Vec::new()
    };

    // Borrow the input tensor from the arena.
    let input_f32: Vec<f32> = {
        let d = arena.data_mut();
        bytemuck::cast_slice::<_, f32>(
            &d[input_slice.offset..input_slice.offset + input_slice.size],
        )
        .to_vec()
    };
    let out_f32: &mut [f32] = {
        let d = arena.data_mut();
        bytemuck::cast_slice_mut::<_, f32>(
            &mut d[output_slice.offset..output_slice.offset + output_slice.size],
        )
    };

    conv2d_f32_im2col_gemm(
        &input_f32,
        &weight_f32,
        &bias_f32,
        out_f32,
        n_in,
        c,
        h,
        w,
        f_out,
        kh,
        kw,
        stride,
        padding,
        dilation,
        groups,
        fused_act,
    );
    Ok(())
}

#[cfg(feature = "prepared-plan")]
fn dispatch_matmul_fp32_with_view(
    arena: &CpuBuffer,
    shape_env: &ShapeEnv,
    kernel_name: &str,
    input_slices: &[BufferSlice],
    output_slice: BufferSlice,
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    view: &crate::backend::prepared::PersistentPreparedWeights,
) -> Result<(), BackendError> {
    use crate::backend::cpu::blas::matmul_blas_into;

    if input_slices.len() < 2 {
        return Err(BackendError::Dispatch(format!(
            "matmul_persistent: expected ≥ 2 input slices, got {}",
            input_slices.len()
        )));
    }
    let a_slice = input_slices[0];
    let b_slice = input_slices[1];
    let bias_slice = input_slices.get(2).copied();

    // Resolve the B (weight) / bias f32 slices.
    let b_f32: Vec<f32> = match view.get(&(b_slice.offset, b_slice.size)) {
        Some(payload) => payload.to_vec(),
        None => {
            let d = arena.data_mut();
            bytemuck::cast_slice::<_, f32>(&d[b_slice.offset..b_slice.offset + b_slice.size])
                .to_vec()
        }
    };
    let bias_f32: Vec<f32> = if let Some(b) = bias_slice {
        match view.get(&(b.offset, b.size)) {
            Some(payload) => payload.to_vec(),
            None => {
                let d = arena.data_mut();
                bytemuck::cast_slice::<_, f32>(&d[b.offset..b.offset + b.size]).to_vec()
            }
        }
    } else {
        Vec::new()
    };

    // Borrow the activation tensor from the arena.
    let a_f32: Vec<f32> = {
        let d = arena.data_mut();
        bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size]).to_vec()
    };

    let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
    let &[m, _k, n] = matmul_params.as_slice() else {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: expected params [M,K,N]"
        )));
    };

    let out_f32: &mut [f32] = {
        let d = arena.data_mut();
        bytemuck::cast_slice_mut::<_, f32>(
            &mut d[output_slice.offset..output_slice.offset + output_slice.size],
        )
    };

    matmul_blas_into(&a_f32, &b_f32, out_f32, m, _k, n);

    // Apply fused activation / bias on top of the GEMM output, mirroring
    // the `matmul_activation_dispatch` semantics.
    let has_bias = !bias_f32.is_empty();
    let act: fn(f32) -> f32 = match kernel_name {
        "matmul_relu" | "fused_matmul_add_relu" => |x| x.max(0.0),
        "matmul_gelu" | "fused_matmul_add_gelu" => |x: f32| {
            let x3 = x * x * x;
            let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
            let t = tanh_arg.tanh();
            0.5 * x * (1.0 + t)
        },
        "matmul_silu" | "fused_matmul_add_silu" => |x: f32| x / (1.0 + (-x).exp()),
        _ => |x| x,
    };
    for i in 0..out_f32.len() {
        let x = out_f32[i]
            + if has_bias && i % n < bias_f32.len() {
                bias_f32[i % n]
            } else {
                0.0
            };
        out_f32[i] = act(x);
    }
    Ok(())
}

#[cfg(feature = "prepared-plan")]
fn dispatch_matmul_quantized_with_view(
    arena: &CpuBuffer,
    shape_env: &ShapeEnv,
    kernel_name: &str,
    input_slices: &[BufferSlice],
    output_slice: BufferSlice,
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    weight_meta: &Option<crate::backend::QuantizedWeightMeta>,
    view: &crate::backend::prepared::PersistentPreparedWeights,
) -> Result<(), BackendError> {
    if input_slices.len() < 2 {
        return Err(BackendError::Dispatch(format!(
            "matmul_quantized_persistent: expected ≥ 2 input slices, got {}",
            input_slices.len()
        )));
    }
    let a_slice = input_slices[0];
    let b_slice = input_slices[1];
    let bias_slice = input_slices.get(2).copied();

    // Resolve the B (weight) raw bytes from the persistent view
    // (no per-copy memcpy — we pass the payload directly to the
    // typed microkernel).
    let raw_weight: &[u8] = match view.get(&(b_slice.offset, b_slice.size)) {
        Some(payload) => payload,
        None => {
            let d = arena.data_mut();
            &d[b_slice.offset..b_slice.offset + b_slice.size]
        }
    };

    let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
    let &[m, k, n] = matmul_params.as_slice() else {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: expected params [M,K,N]"
        )));
    };

    let meta = weight_meta
        .clone()
        .unwrap_or_else(|| crate::backend::QuantizedWeightMeta {
            bit_width: 8,
            scales: vec![1.0],
            zero_points: vec![0.0],
            shape: vec![m, k],
        });

    let out_f32: &mut [f32] = {
        let d = arena.data_mut();
        bytemuck::cast_slice_mut::<_, f32>(
            &mut d[output_slice.offset..output_slice.offset + output_slice.size],
        )
    };

    match kernel_name {
        "matmul_u4" | "matmul_u4_i8" => {
            let typed = pack_bytes_to_u4x8(raw_weight);
            let pt = packed_tensor_from_meta(typed, meta, kernel_name)?;
            if kernel_name == "matmul_u4_i8" {
                let a_raw = {
                    let d = arena.data_mut();
                    &d[a_slice.offset..a_slice.offset + a_slice.size]
                };
                crate::backend::cpu::microkernels::gemm_cpu_flat_i8_u4(
                    &pt, a_raw, out_f32, m, k, n,
                );
            } else {
                let a_f32 = {
                    let d = arena.data_mut();
                    bytemuck::cast_slice::<_, f32>(
                        &d[a_slice.offset..a_slice.offset + a_slice.size],
                    )
                };
                crate::backend::cpu::microkernels::gemm_cpu_flat::<
                    crate::backend::cpu::u4x8::U4x8,
                >(&pt, a_f32, out_f32, m, k, n);
            }
        }
        "matmul_u8" | "matmul_u8_i8" => {
            let typed = pack_bytes_to_u8x4(raw_weight);
            let pt = packed_tensor_from_meta(typed, meta, kernel_name)?;
            if kernel_name == "matmul_u8_i8" {
                let a_raw = {
                    let d = arena.data_mut();
                    &d[a_slice.offset..a_slice.offset + a_slice.size]
                };
                crate::backend::cpu::microkernels::gemm_cpu_flat_i8_u8x4(
                    &pt, a_raw, out_f32, m, k, n,
                );
            } else {
                let a_f32 = {
                    let d = arena.data_mut();
                    bytemuck::cast_slice::<_, f32>(
                        &d[a_slice.offset..a_slice.offset + a_slice.size],
                    )
                };
                crate::backend::cpu::microkernels::gemm_cpu_flat::<
                    crate::backend::cpu::u4x8::U8x4,
                >(&pt, a_f32, out_f32, m, k, n);
            }
        }
        other => {
            return Err(BackendError::Dispatch(format!(
                "matmul_quantized_persistent: unsupported kernel '{other}'"
            )));
        }
    }

    // Apply optional bias (quantized matmul paths don't fuse activations
    // at the kernel level today, but bias is still supported via the
    // standard post-process path).
    if let Some(b_slice) = bias_slice {
        let bias_f32: Vec<f32> = match view.get(&(b_slice.offset, b_slice.size)) {
            Some(payload) => payload.to_vec(),
            None => {
                let d = arena.data_mut();
                bytemuck::cast_slice::<_, f32>(&d[b_slice.offset..b_slice.offset + b_slice.size])
                    .to_vec()
            }
        };
        for i in 0..out_f32.len() {
            if i % n < bias_f32.len() {
                out_f32[i] += bias_f32[i % n];
            }
        }
    }

    Ok(())
}

#[cfg(feature = "prepared-plan")]
fn pack_bytes_to_u4x8(raw: &[u8]) -> Vec<crate::backend::cpu::u4x8::U4x8> {
    let mut packed = vec![0u32; raw.len().div_ceil(4)];
    {
        let bytes = bytemuck::cast_slice_mut::<_, u8>(&mut packed);
        bytes[..raw.len()].copy_from_slice(raw);
    }
    bytemuck::cast_slice(&packed).to_vec()
}

#[cfg(feature = "prepared-plan")]
fn pack_bytes_to_u8x4(raw: &[u8]) -> Vec<crate::backend::cpu::u4x8::U8x4> {
    let mut packed = vec![0u32; raw.len().div_ceil(4)];
    {
        let bytes = bytemuck::cast_slice_mut::<_, u8>(&mut packed);
        bytes[..raw.len()].copy_from_slice(raw);
    }
    bytemuck::cast_slice(&packed).to_vec()
}

macro_rules! impl_simd_unary_wrapper {
    ($name:ident, $avx2:path, $scalar:path) => {
        #[inline]
        fn $name(input: &[f32], output: &mut [f32]) {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if microkernels::simd_avx2_available() {
                return unsafe { $avx2(input, output) };
            }
            let len = output.len().min(input.len());
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                output[..len].par_iter_mut().enumerate().for_each(|(i, o)| {
                    *o = $scalar(input[i]);
                });
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
        fn $name(a: &[f32], b: &[f32], output: &mut [f32]) {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if (a.len() == output.len() || b.len() == output.len())
                && microkernels::simd_avx2_available()
            {
                return unsafe { $avx2(a, b, output) };
            }
            let len = output.len().min(a.len().max(b.len()));
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                let a_len = a.len();
                let b_len = b.len();
                output[..len].par_iter_mut().enumerate().for_each(|(i, o)| {
                    *o = $op(a[i % a_len], b[i % b_len]);
                });
            }
            #[cfg(not(feature = "parallel"))]
            $scalar(a, b, output);
        }
    };
}

macro_rules! impl_simd_scalar_wrapper {
    ($name:ident, $avx2:path, $scalar:path, $op:expr) => {
        #[inline]
        fn $name(data: &[f32], s: f32, output: &mut [f32]) {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if microkernels::simd_avx2_available() {
                return unsafe { $avx2(data, s, output) };
            }
            let len = output.len().min(data.len());
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                output[..len].par_iter_mut().enumerate().for_each(|(i, o)| {
                    *o = $op(data[i], s);
                });
            }
            #[cfg(not(feature = "parallel"))]
            $scalar(data, s, output);
        }
    };
}

// ============================================================
// SIMD-aware elementwise dispatch wrappers
// ============================================================
// Each wrapper checks `simd_avx2_available()` at runtime and
// delegates to the AVX2 microkernel when possible, falling back
// to a scalar loop.  The scalar fallback is the same code that
// was previously inlined in the big `dispatch()` match arms —
// flattened to reduce duplication.

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
fn leaky_relu_f32(input: &[f32], output: &mut [f32], slope: f32) {
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
fn clamp_f32(input: &[f32], output: &mut [f32], min_val: f32, max_val: f32) {
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
fn log_softmax_f32(input: &[f32], output: &mut [f32]) {
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
// New dispatch wrappers — Reductions
// ============================================================

#[inline]
fn reduce_f32(input: &[f32], output: &mut [f32], group_size: usize, is_mean: bool, is_max: bool) {
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
// New dispatch wrappers — Scalar arithmetic
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
// New dispatch wrappers — Scalar comparison
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
// New dispatch wrappers — BiasAdd, Norm, RMS Norm, Softmax
// ============================================================

#[inline]
fn biasadd_f32(data: &[f32], bias: &[f32], output: &mut [f32], channel_stride: usize) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe { microkernels::biasadd_f32_avx2(data, bias, output, channel_stride) };
    }
    microkernels::biasadd_f32_scalar(data, bias, output, channel_stride);
}

#[inline]
fn norm_layernorm_f32(input: &[f32], output: &mut [f32], row_size: usize, eps: f32) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe { microkernels::norm_layernorm_f32_avx2(input, output, row_size, eps) };
    }
    microkernels::norm_layernorm_f32_scalar(input, output, row_size, eps);
}

#[inline]
fn rms_norm_f32(input: &[f32], weight: &[f32], output: &mut [f32], row_size: usize, eps: f32) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe { microkernels::rms_norm_f32_avx2(input, weight, output, row_size, eps) };
    }
    microkernels::rms_norm_f32_scalar(input, weight, output, row_size, eps);
}

#[inline]
fn softmax_f32(
    input: &[f32],
    output: &mut [f32],
    axis_dim_size: usize,
    stride: usize,
    num_rows: usize,
) {
    #[cfg(feature = "parallel")]
    if num_rows > 1 && stride == 1 {
        use rayon::prelude::*;
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
        // Build non-overlapping row slices before the parallel section
        // to satisfy the borrow checker and rayon's Send requirements.
        let mut row_slices: Vec<(&[f32], &mut [f32])> = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            let offset = row * axis_dim_size;
            let inp = &input[offset..offset + axis_dim_size];
            // SAFETY: Each row writes to a unique non-overlapping region of output.
            let out = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr().add(offset), axis_dim_size)
            };
            row_slices.push((inp, out));
        }
        row_slices.par_iter_mut().for_each(|(inp, out)| {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            if has_avx2 {
                unsafe {
                    microkernels::softmax_f32_avx2_strided(inp, out, axis_dim_size, 1, 1);
                }
                return;
            }
            microkernels::softmax_f32_scalar_strided(inp, out, axis_dim_size, 1, 1);
        });
        return;
    }
    // Single-threaded fallback
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe {
            microkernels::softmax_f32_avx2_strided(input, output, axis_dim_size, stride, num_rows)
        };
    }
    microkernels::softmax_f32_scalar_strided(input, output, axis_dim_size, stride, num_rows);
}

fn argmax_f32(input: &[f32], output: &mut [u64], axis: usize, dim_size: usize, inner: usize) {
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
fn sgd_update_f32(w: &[f32], g: &[f32], lr: f32, wd: f32) -> Vec<f32> {
    let mut out = vec![0.0; w.len()];
    sgd_update_f32_into(w, g, lr, wd, &mut out);
    out
}

#[inline]
fn sgd_update_f32_into(w: &[f32], g: &[f32], lr: f32, wd: f32, out: &mut [f32]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        unsafe { microkernels::sgd_update_f32_avx2_into(w, g, lr, wd, out) };
        return;
    }
    microkernels::sgd_update_f32_scalar_into(w, g, lr, wd, out);
}

#[inline]
fn adam_update_f32(
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
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() {
        return unsafe {
            microkernels::adam_update_f32_avx2(
                w, g, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2,
            )
        };
    }
    microkernels::adam_update_f32_scalar(w, g, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2)
}

#[inline]
fn adamw_update_f32(
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
fn adamw_update_f32_into(
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
fn lion_update_f32(
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
fn lion_update_f32_into(
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
fn rmsprop_update_f32(
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
fn rmsprop_update_f32_into(
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
fn muon_update_f32(
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
