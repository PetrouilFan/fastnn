#![allow(clippy::too_many_arguments)]

use crate::backend::cpu::blas::matmul_blas_into;
use crate::backend::{BackendError, BufferSlice};
use crate::dtypes::{PackedWord, U4x8, U8x4};
use crate::ir::node::{DimExpr, ShapeEnv};
use crate::packed_tensor::PackedTensor;

use super::{resolve_params, CpuBuffer};

// Runtime dispatch helpers intentionally take the IR/kernel call-site context
// directly so they do not allocate wrapper structs on hot paths.

/// Helper: dispatch a fused matmul + bias + activation kernel at runtime.
///
/// Handles both fused (with bias, 3 input slices) and non-fused (without bias,
/// 2 input slices) matmul activation variants.  Extracts A and B (and optionally
/// bias) from the arena, resolves [M,K,N] params, calls `matmul_blas_into`, and
/// applies `act(x + bias[i % n])`.
///
/// Generic closure `act` is monomorphized and inlined — zero overhead vs the
/// handwritten per-activation blocks.
#[inline]
pub(super) fn matmul_activation_dispatch(
    input_slices: &[BufferSlice],
    arena: &CpuBuffer,
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    out_start: usize,
    out_end: usize,
    kernel_name: &str,
    act: impl Fn(f32) -> f32,
) -> Result<(), BackendError> {
    if let [a_slice, b_slice] = &input_slices[..2] {
        let has_bias = input_slices.len() >= 3;
        let (a, b, bias) = {
            let d = arena.data_mut();
            let a_f32 =
                bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size])
                    .to_vec();
            let b_f32 =
                bytemuck::cast_slice::<_, f32>(&d[b_slice.offset..b_slice.offset + b_slice.size])
                    .to_vec();
            let bias_f32 = if has_bias {
                let bias_slice = &input_slices[2];
                bytemuck::cast_slice::<_, f32>(
                    &d[bias_slice.offset..bias_slice.offset + bias_slice.size],
                )
                .to_vec()
            } else {
                Vec::new()
            };
            (a_f32, b_f32, bias_f32)
        };
        let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
        let &[m, _k, n] = &matmul_params[..] else {
            return Err(BackendError::Dispatch(format!(
                "{kernel_name}: expected params [M,K,N]"
            )));
        };
        let out_f32 = {
            let d = arena.data_mut();
            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
        };
        matmul_blas_into(&a, &b, out_f32, m, _k, n);
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

/// Validate quantized weight metadata and build a packed tensor.
pub(super) fn packed_tensor_from_meta<T: PackedWord>(
    data: Vec<T>,
    meta: crate::backend::QuantizedWeightMeta,
    kernel_name: &str,
) -> Result<PackedTensor<T>, BackendError> {
    if meta.scales.is_empty() {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: quantized weight metadata missing scales"
        )));
    }
    if meta.zero_points.is_empty() {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: quantized weight metadata missing zero_points"
        )));
    }
    if meta.scales.len() != meta.zero_points.len() {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: quantized weight metadata length mismatch: {} scales vs {} zero_points",
            meta.scales.len(),
            meta.zero_points.len()
        )));
    }

    let rows = if meta.shape.len() >= 2 {
        meta.shape[0]
    } else {
        1
    };
    let valid_meta_len = meta.scales.len() == 1 || meta.scales.len() == rows;
    if !valid_meta_len {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: quantized weight metadata length {} incompatible with shape {:?} (expected 1 or {})",
            meta.scales.len(),
            meta.shape,
            rows
        )));
    }

    let numel: usize = meta.shape.iter().product();
    let expected_words = if meta.shape.len() >= 2 {
        let inner: usize = meta.shape[1..].iter().product();
        rows * inner.div_ceil(T::ITEMS)
    } else {
        numel.div_ceil(T::ITEMS)
    };
    if data.len() < expected_words {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: quantized weight payload too short: got {} packed words for shape {:?}, expected at least {}",
            data.len(),
            meta.shape,
            expected_words
        )));
    }

    Ok(PackedTensor::from_raw(
        data,
        meta.shape,
        meta.scales,
        meta.zero_points,
    ))
}

/// Helper: dispatch a quantized matmul (u4 or u8) at runtime.
///
/// Generic over `T: PackedWord` — monomorphized as `U4x8` or `U8x4`.
/// Handles arena extraction, u32-aligned weight copy, PackedTensor
/// construction, and `gemm_cpu_flat` dispatch.
#[inline]
pub(super) fn quantized_matmul_dispatch<T: PackedWord>(
    input_slices: &[BufferSlice],
    arena: &CpuBuffer,
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    weight_meta: &Option<crate::backend::QuantizedWeightMeta>,
    out_start: usize,
    out_end: usize,
    bit_width: usize,
    kernel_name: &str,
) -> Result<(), BackendError> {
    if let [a_slice, w_slice] = input_slices {
        let (activations, packed_bytes) = {
            let d = arena.data_mut();
            (
                bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size])
                    .to_vec(),
                {
                    let raw = &d[w_slice.offset..w_slice.offset + w_slice.size];
                    // Copy to u32-aligned buffer (arena may not be u32-aligned)
                    let mut aligned: Vec<u32> = vec![0u32; raw.len().div_ceil(4)];
                    let byte_slice = bytemuck::cast_slice_mut::<_, u8>(&mut aligned);
                    byte_slice[..raw.len()].copy_from_slice(raw);
                    aligned
                },
            )
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
                bit_width,
                scales: vec![1.0],
                zero_points: vec![0.0],
                shape: vec![m, k],
            });
        let typed_data: Vec<T> = bytemuck::cast_slice(&packed_bytes).to_vec();
        let pt = packed_tensor_from_meta(typed_data, meta, kernel_name)?;
        let out_f32 = {
            let d = arena.data_mut();
            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
        };
        crate::backend::cpu::microkernels::gemm_cpu_flat::<T>(&pt, &activations, out_f32, m, k, n);
    }
    Ok(())
}

/// Dispatch pre-quantized I8 activation × U8x4 packed-weight MatMul.
///
/// Reads activation from arena as raw bytes (I8 payload format:
/// [scale_f32][zp_f32][i8_data...]), builds a `PackedTensor<U8x4>` from
/// the weight bytes, and calls the scalar I8×U8x4 microkernel.
#[inline]
pub(super) fn quantized_matmul_dispatch_i8_u8(
    input_slices: &[BufferSlice],
    arena: &CpuBuffer,
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    weight_meta: &Option<crate::backend::QuantizedWeightMeta>,
    out_start: usize,
    out_end: usize,
    kernel_name: &str,
) -> Result<(), BackendError> {
    if let [a_slice, w_slice] = input_slices {
        let (activation_payload, packed_bytes) = {
            let d = arena.data_mut();
            (d[a_slice.offset..a_slice.offset + a_slice.size].to_vec(), {
                let raw = &d[w_slice.offset..w_slice.offset + w_slice.size];
                let mut aligned: Vec<u32> = vec![0u32; raw.len().div_ceil(4)];
                let byte_slice = bytemuck::cast_slice_mut::<_, u8>(&mut aligned);
                byte_slice[..raw.len()].copy_from_slice(raw);
                aligned
            })
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
        let typed_data: Vec<U8x4> = bytemuck::cast_slice(&packed_bytes).to_vec();
        let pt = packed_tensor_from_meta(typed_data, meta, kernel_name)?;
        let out_f32 = {
            let d = arena.data_mut();
            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
        };
        crate::backend::cpu::microkernels::gemm_cpu_flat_i8_u8x4(
            &pt,
            &activation_payload,
            out_f32,
            m,
            k,
            n,
        );
    }
    Ok(())
}

/// Dispatch pre-quantized I8 activation × U4x8 packed-weight MatMul.
///
/// Reads activation from arena as raw bytes (I8 payload format:
/// [scale_f32][zp_f32][i8_data...]), builds a `PackedTensor<U4x8>` from
/// the weight bytes, and calls the scalar I8×U4x8 microkernel.
#[inline]
pub(super) fn quantized_matmul_dispatch_i8_u4(
    input_slices: &[BufferSlice],
    arena: &CpuBuffer,
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    weight_meta: &Option<crate::backend::QuantizedWeightMeta>,
    out_start: usize,
    out_end: usize,
    kernel_name: &str,
) -> Result<(), BackendError> {
    if let [a_slice, w_slice] = input_slices {
        let (activation_payload, packed_bytes) = {
            let d = arena.data_mut();
            (d[a_slice.offset..a_slice.offset + a_slice.size].to_vec(), {
                let raw = &d[w_slice.offset..w_slice.offset + w_slice.size];
                let mut aligned: Vec<u32> = vec![0u32; raw.len().div_ceil(4)];
                let byte_slice = bytemuck::cast_slice_mut::<_, u8>(&mut aligned);
                byte_slice[..raw.len()].copy_from_slice(raw);
                aligned
            })
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
                bit_width: 4,
                scales: vec![1.0],
                zero_points: vec![0.0],
                shape: vec![m, k],
            });
        let typed_data: Vec<U4x8> = bytemuck::cast_slice(&packed_bytes).to_vec();
        let pt = packed_tensor_from_meta(typed_data, meta, kernel_name)?;
        let out_f32 = {
            let d = arena.data_mut();
            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
        };
        crate::backend::cpu::microkernels::gemm_cpu_flat_i8_u4x8(
            &pt,
            &activation_payload,
            out_f32,
            m,
            k,
            n,
        );
    }
    Ok(())
}
