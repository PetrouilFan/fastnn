#![allow(clippy::too_many_arguments)]

use crate::backend::cpu::blas::matmul_blas_into;
use crate::backend::{BackendError, BufferSlice};
use crate::dtypes::{PackedWord, U4x8, U8x4};
use crate::ir::node::{DimExpr, ShapeEnv};
use crate::packed_tensor::PackedTensor;

use super::{aligned_packed_slice, resolve_params, CpuBuffer};

/// Dequantize I8 activation payload to f32.
///
/// Supports two formats:
/// - **Per-tensor** (legacy): `[scale_f32][zp_f32][i8_data...]` (header_size = 8)
/// - **Per-channel**: `[num_channels(u32)][chunk_size(u32)][scale_1..scale_n(zp_1..zp_n)][ch_data...]`
///
/// Detection: if the first 4 bytes, read as u32, are > 0 and the payload is
/// large enough for the per-channel header, we treat it as per-channel.
pub(super) fn dequantize_i8_activation(payload: &[u8]) -> Vec<f32> {
    // Attempt to detect per-channel format: num_channels > 0 and payload
    // is larger than the minimal per-channel header (4 + 4 + 2*4 = 16 bytes
    // for 1 channel).
    let per_channel_detected = {
        if payload.len() >= 16 {
            let nc = u32::from_le_bytes(payload[0..4].try_into().unwrap_or([0; 4])) as usize;
            let chunk_size =
                u32::from_le_bytes(payload[4..8].try_into().unwrap_or([0; 4])) as usize;
            let expected = 8 + nc * 8 + nc * chunk_size;
            nc > 0 && chunk_size > 0 && payload.len() >= expected
        } else {
            false
        }
    };

    if per_channel_detected {
        let nc = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
        let chunk_size = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
        let data_start = 8 + nc * 8;
        let numel = nc * chunk_size;
        let mut out = Vec::with_capacity(numel);
        for ch in 0..nc {
            let scale = f32::from_le_bytes(
                payload[8 + ch * 4..8 + (ch + 1) * 4]
                    .try_into()
                    .unwrap_or([0; 4]),
            );
            let zp = f32::from_le_bytes(
                payload[8 + nc * 4 + ch * 4..8 + nc * 4 + (ch + 1) * 4]
                    .try_into()
                    .unwrap_or([0; 4]),
            );
            let ch_start = ch * chunk_size;
            for j in 0..chunk_size {
                let di = data_start + j;
                let q = if di < payload.len() {
                    payload[di] as i8
                } else {
                    0i8
                };
                out.push((q as f32) * scale + zp);
            }
        }
        out
    } else {
        let header_size = 8;
        let numel = payload.len().saturating_sub(header_size);
        let scale = if payload.len() >= 4 {
            f32::from_le_bytes(payload[0..4].try_into().unwrap())
        } else {
            1.0
        };
        let zp = if payload.len() >= 8 {
            f32::from_le_bytes(payload[4..8].try_into().unwrap())
        } else {
            0.0
        };
        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let idx = header_size + i;
            let q = payload[idx] as i8;
            out.push((q as f32) * scale + zp);
        }
        out
    }
}

/// Dequantize I8 activation payload into a pre-allocated output slice.
pub(super) fn dequantize_i8_activation_into(payload: &[u8], out: &mut [f32]) {
    let per_channel_detected = {
        if payload.len() >= 16 {
            let nc = u32::from_le_bytes(payload[0..4].try_into().unwrap_or([0; 4])) as usize;
            let chunk_size =
                u32::from_le_bytes(payload[4..8].try_into().unwrap_or([0; 4])) as usize;
            let expected = 8 + nc * 8 + nc * chunk_size;
            nc > 0 && chunk_size > 0 && payload.len() >= expected
        } else {
            false
        }
    };

    if per_channel_detected {
        let nc = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
        let chunk_size = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
        let data_start = 8 + nc * 8;
        let mut idx = 0;
        for ch in 0..nc {
            let scale = f32::from_le_bytes(
                payload[8 + ch * 4..8 + (ch + 1) * 4]
                    .try_into()
                    .unwrap_or([0; 4]),
            );
            let zp = f32::from_le_bytes(
                payload[8 + nc * 4 + ch * 4..8 + nc * 4 + (ch + 1) * 4]
                    .try_into()
                    .unwrap_or([0; 4]),
            );
            for j in 0..chunk_size {
                let di = data_start + j;
                let q = if di < payload.len() {
                    payload[di] as i8
                } else {
                    0i8
                };
                out[idx] = (q as f32) * scale + zp;
                idx += 1;
            }
        }
    } else {
        let header_size = 8;
        let scale = if payload.len() >= 4 {
            f32::from_le_bytes(payload[0..4].try_into().unwrap())
        } else {
            1.0
        };
        let zp = if payload.len() >= 8 {
            f32::from_le_bytes(payload[4..8].try_into().unwrap())
        } else {
            0.0
        };
        for i in 0..out.len() {
            let pidx = header_size + i;
            let q = if pidx < payload.len() {
                payload[pidx] as i8
            } else {
                0i8
            };
            out[i] = (q as f32) * scale + zp;
        }
    }
}

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
    meta: std::sync::Arc<crate::backend::QuantizedWeightMeta>,
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
    let valid_meta_len = meta.scales.len() == 1
        || meta.scales.len() == rows
        || (meta.scales.len() > 1 && rows > meta.scales.len() && rows % meta.scales.len() == 0);
    if !valid_meta_len {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: quantized weight metadata length {} incompatible with shape {:?} (expected 1, {}, or a divisor of {})",
            meta.scales.len(),
            meta.shape,
            rows,
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

    Ok({
        let scales_len = meta.scales.len();
        let mut pt = PackedTensor::from_raw(data, meta.shape.clone(), meta.scales.clone(), meta.zero_points.clone());
        if pt.group_size == 0 && scales_len > 1 && rows > scales_len {
            pt.group_size = rows / scales_len;
        }
        pt
    })
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
    weight_meta: &Option<std::sync::Arc<crate::backend::QuantizedWeightMeta>>,
    out_start: usize,
    out_end: usize,
    bit_width: usize,
    kernel_name: &str,
) -> Result<(), BackendError> {
    if let [a_slice, w_slice] = input_slices {
        let (activations, typed_data) = {
            let d = arena.data_mut();
            (
                bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size])
                    .to_vec(),
                {
                    let raw = &d[w_slice.offset..w_slice.offset + w_slice.size];
                    aligned_packed_slice::<T>(raw)
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
            .unwrap_or_else(|| std::sync::Arc::new(crate::backend::QuantizedWeightMeta {
                bit_width,
                scales: vec![1.0],
                zero_points: vec![0.0],
                shape: vec![m, k],
            }));
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
    weight_meta: &Option<std::sync::Arc<crate::backend::QuantizedWeightMeta>>,
    out_start: usize,
    out_end: usize,
    kernel_name: &str,
) -> Result<(), BackendError> {
    if let [a_slice, w_slice] = input_slices {
        let (activation_payload, typed_data) = {
            let d = arena.data_mut();
            (
                d[a_slice.offset..a_slice.offset + a_slice.size].to_vec(),
                {
                    let raw = &d[w_slice.offset..w_slice.offset + w_slice.size];
                    aligned_packed_slice::<U8x4>(raw)
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
            .unwrap_or_else(|| std::sync::Arc::new(crate::backend::QuantizedWeightMeta {
                bit_width: 8,
                scales: vec![1.0],
                zero_points: vec![0.0],
                shape: vec![m, k],
            }));
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
    weight_meta: &Option<std::sync::Arc<crate::backend::QuantizedWeightMeta>>,
    out_start: usize,
    out_end: usize,
    kernel_name: &str,
) -> Result<(), BackendError> {
    if let [a_slice, w_slice] = input_slices {
        let (activation_payload, typed_data) = {
            let d = arena.data_mut();
            (
                d[a_slice.offset..a_slice.offset + a_slice.size].to_vec(),
                {
                    let raw = &d[w_slice.offset..w_slice.offset + w_slice.size];
                    aligned_packed_slice::<U4x8>(raw)
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
            .unwrap_or_else(|| std::sync::Arc::new(crate::backend::QuantizedWeightMeta {
                bit_width: 4,
                scales: vec![1.0],
                zero_points: vec![0.0],
                shape: vec![m, k],
            }));
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
