#![allow(clippy::too_many_arguments)]

use crate::backend::cpu::blas::matmul_blas_into;
use crate::backend::{BackendError, BufferSlice};
use crate::dtypes::{I4x8, I8x4, PackedWord};
use crate::ir::{DimExpr, ShapeEnv};
use crate::packed_tensor::PackedTensor;
use std::sync::Arc;

use super::{get_or_cache_packed, resolve_params, CpuBuffer};

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
///
/// # Zero-copy
///
/// Reads A, B, bias directly from the arena without intermediate `to_vec()` copies.
/// The arena's memory is pre-planned with non-overlapping slots, and the `UnsafeCell`
/// backing allows simultaneous read-only (input) and read-write (output) views.
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
        let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
        let &[m, _k, n] = &matmul_params[..] else {
            return Err(BackendError::Dispatch(format!(
                "{kernel_name}: expected params [M,K,N]"
            )));
        };
        let out_size = out_end - out_start;

        // Zero-copy views into the arena.
        // SAFETY: The memory planner guarantees non-overlapping BufferSlice ranges.
        // Dispatch is single-threaded via the CpuBackend sequential loop.
        let a: &[f32] = unsafe { arena.view_f32(a_slice.offset, a_slice.size) };
        let b: &[f32] = unsafe { arena.view_f32(b_slice.offset, b_slice.size) };
        let out_f32: &mut [f32] = unsafe { arena.view_f32_mut(out_start, out_size) };

        matmul_blas_into(a, b, out_f32, m, _k, n);

        if has_bias {
            let bias: &[f32] =
                unsafe { arena.view_f32(input_slices[2].offset, input_slices[2].size) };
            for i in 0..out_f32.len() {
                out_f32[i] = act(out_f32[i] + bias[i % n]);
            }
        } else {
            for x in out_f32.iter_mut() {
                *x = act(*x);
            }
        }
    }
    Ok(())
}

/// Validate quantized weight metadata and build a packed tensor.
///
/// FP8 types (F8x4/F8x4R/F4x8) use symmetric quantization with zero point = 0.
/// When `zero_points` is empty (because the IR dtype only stores scales for FP8),
/// we synthesize a matching zeros vector of 0.0 values.
pub(super) fn packed_tensor_from_meta<T: PackedWord>(
    data: Arc<Vec<T>>,
    meta: std::sync::Arc<crate::backend::QuantizedWeightMeta>,
    kernel_name: &str,
) -> Result<PackedTensor<T>, BackendError> {
    if meta.scales.is_empty() {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: quantized weight metadata missing scales"
        )));
    }
    let zero_points = if meta.dequant_offsets.is_empty() {
        vec![0.0; meta.scales.len()]
    } else {
        meta.dequant_offsets.clone()
    };
    if meta.scales.len() != zero_points.len() {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: quantized weight metadata length mismatch: {} scales vs {} zero_points",
            meta.scales.len(),
            zero_points.len()
        )));
    }

    let rows = if meta.shape.len() >= 2 {
        meta.shape[0]
    } else {
        1
    };
    let valid_meta_len = meta.scales.len() == 1
        || meta.scales.len() == rows
        || (meta.scales.len() > 1 && rows > meta.scales.len() && rows % meta.scales.len() == 0)
        || (meta.quant_block_size > 0
            && meta.scales.len() > rows
            && meta.scales.len().is_multiple_of(rows));
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
        let mut pt =
            PackedTensor::from_raw_arc(data, meta.shape.clone(), meta.scales.clone(), zero_points);
        pt.quant_block_size = meta.quant_block_size;
        pt.codebooks = meta.codebooks.clone();
        if pt.group_size == 0 && scales_len > 1 && rows > scales_len {
            pt.group_size = rows / scales_len;
        }
        pt
    })
}

/// Helper: dispatch a quantized matmul (I4x8/I8x4/F4x8/F8x4/F8x4R) at runtime.
///
/// Generic over `T: PackedWord` — monomorphized as the concrete packed type.
/// Handles arena extraction, u32-aligned weight copy, PackedTensor
/// construction, and `gemm_cpu_flat` dispatch.
///
/// # Zero-copy
///
/// Activations are read directly from the arena via a zero-copy `&[f32]` view.
/// Weight data is copied into a `PackedTensor` (required by the microkernel API).
#[inline]
pub(super) fn quantized_matmul_dispatch<T: PackedWord + 'static>(
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
        let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
        let &[m, k, n] = &matmul_params[..] else {
            return Err(BackendError::Dispatch(format!(
                "{kernel_name}: expected params [M,K,N]"
            )));
        };

        // Zero-copy activation view.
        // SAFETY: input/output slices are non-overlapping; dispatch is single-threaded.
        let activations: &[f32] = unsafe { arena.view_f32(a_slice.offset, a_slice.size) };
        let typed_data = {
            // SAFETY: weight slice does not overlap with activation or output.
            let raw: &[u8] = unsafe { arena.view_u8(w_slice.offset, w_slice.size) };
            get_or_cache_packed::<T>(w_slice.offset, w_slice.size, raw)
        };
        let meta = weight_meta.clone().unwrap_or_else(|| {
            std::sync::Arc::new(crate::backend::QuantizedWeightMeta {
                bit_width,
                scales: vec![1.0],
                dequant_offsets: vec![0.0],
                shape: vec![m, k],
                quant_block_size: 0,
                codebooks: vec![],
            })
        });
        let pt = packed_tensor_from_meta(typed_data, meta, kernel_name)?;
        let out_f32: &mut [f32] = unsafe { arena.view_f32_mut(out_start, out_end - out_start) };

        if !pt.codebooks.is_empty() {
            // Codebook quantization (I4Codebook): dequant to f32 using codebook
            // lookup, then use standard f32 BLAS matmul. This mirrors the Conv2d
            // path which uses `dispatch_packed_conv_cached!` → `get_or_init_f32_weights()`
            // → `conv2d_f32_im2col_gemm`.
            let f32_weights = pt.get_or_init_f32_weights();
            matmul_blas_into(activations, f32_weights, out_f32, m, k, n);
        } else {
            crate::backend::cpu::microkernels::gemm_cpu_flat::<T>(
                &pt,
                activations,
                out_f32,
                m,
                k,
                n,
            );
        }
    }
    Ok(())
}

fn validate_i8_activation_affine(payload: &[u8], kernel_name: &str) -> Result<(), BackendError> {
    let scale = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let offset = f32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
    if !scale.is_finite() || scale <= 0.0 || !offset.is_finite() {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: activation affine metadata must be finite with a positive scale"
        )));
    }
    Ok(())
}

fn validate_i8_matmul_contract(
    activation: BufferSlice,
    output_start: usize,
    output_end: usize,
    m: usize,
    k: usize,
    n: usize,
    weight_meta: &Option<std::sync::Arc<crate::backend::QuantizedWeightMeta>>,
    kernel_name: &str,
) -> Result<(), BackendError> {
    let activation_values = m.checked_mul(k).ok_or_else(|| {
        BackendError::Dispatch(format!("{kernel_name}: activation element count overflows"))
    })?;
    let expected_activation = activation_values.checked_add(8).ok_or_else(|| {
        BackendError::Dispatch(format!("{kernel_name}: activation payload size overflows"))
    })?;
    if activation.size != expected_activation {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: activation payload has {} bytes, expected {expected_activation}",
            activation.size
        )));
    }
    let output_values = m.checked_mul(n).ok_or_else(|| {
        BackendError::Dispatch(format!("{kernel_name}: output element count overflows"))
    })?;
    let expected_output = output_values.checked_mul(4).ok_or_else(|| {
        BackendError::Dispatch(format!("{kernel_name}: output byte size overflows"))
    })?;
    if output_end.checked_sub(output_start) != Some(expected_output)
        || !output_start.is_multiple_of(4)
    {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: output slice does not match the declared dimensions"
        )));
    }
    let meta = weight_meta.as_ref().ok_or_else(|| {
        BackendError::Dispatch(format!("{kernel_name}: missing quantized weight metadata"))
    })?;
    if meta.shape.as_slice() != [n, k] {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: weight shape {:?} does not match [{n}, {k}]",
            meta.shape
        )));
    }
    Ok(())
}

/// Dispatch pre-quantized I8 activation × I8x4 packed-weight MatMul.
///
/// Reads activation from arena as raw bytes (I8 payload format:
/// [scale_f32][zp_f32][i8_data...]), builds a `PackedTensor<I8x4>` from
/// the weight bytes, and calls the scalar I8×I8x4 microkernel.
///
/// # Zero-copy
///
/// Activation payload and output are read/written directly in the arena.
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
    if input_slices.len() != 2 {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: expected activation and weight inputs"
        )));
    }
    if let [a_slice, w_slice] = input_slices {
        let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
        let &[m, k, n] = &matmul_params[..] else {
            return Err(BackendError::Dispatch(format!(
                "{kernel_name}: expected params [M,K,N]"
            )));
        };
        validate_i8_matmul_contract(
            *a_slice,
            out_start,
            out_end,
            m,
            k,
            n,
            weight_meta,
            kernel_name,
        )?;

        // Zero-copy views.
        let activation_payload: &[u8] = unsafe { arena.view_u8(a_slice.offset, a_slice.size) };
        validate_i8_activation_affine(activation_payload, kernel_name)?;
        let typed_data = {
            let raw: &[u8] = unsafe { arena.view_u8(w_slice.offset, w_slice.size) };
            get_or_cache_packed::<I8x4>(w_slice.offset, w_slice.size, raw)
        };
        let meta = weight_meta.clone().ok_or_else(|| {
            BackendError::Dispatch(format!("{kernel_name}: missing quantized weight metadata"))
        })?;
        let pt = packed_tensor_from_meta(typed_data, meta, kernel_name)?;
        let out_f32: &mut [f32] = unsafe { arena.view_f32_mut(out_start, out_end - out_start) };

        crate::backend::cpu::microkernels::gemm_cpu_flat_i8_i8x4(
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

/// Dispatch pre-quantized I8 activation × I4x8 packed-weight MatMul.
///
/// Reads activation from arena as raw bytes (I8 payload format:
/// [scale_f32][zp_f32][i8_data...]), builds a `PackedTensor<I4x8>` from
/// the weight bytes, and calls the scalar I8×I4x8 microkernel.
///
/// # Zero-copy
///
/// Activation payload and output are read/written directly in the arena.
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
    if input_slices.len() != 2 {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: expected activation and weight inputs"
        )));
    }
    if let [a_slice, w_slice] = input_slices {
        let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
        let &[m, k, n] = &matmul_params[..] else {
            return Err(BackendError::Dispatch(format!(
                "{kernel_name}: expected params [M,K,N]"
            )));
        };
        validate_i8_matmul_contract(
            *a_slice,
            out_start,
            out_end,
            m,
            k,
            n,
            weight_meta,
            kernel_name,
        )?;

        // Zero-copy views.
        let activation_payload: &[u8] = unsafe { arena.view_u8(a_slice.offset, a_slice.size) };
        validate_i8_activation_affine(activation_payload, kernel_name)?;
        let typed_data = {
            let raw: &[u8] = unsafe { arena.view_u8(w_slice.offset, w_slice.size) };
            get_or_cache_packed::<I4x8>(w_slice.offset, w_slice.size, raw)
        };
        let meta = weight_meta.clone().ok_or_else(|| {
            BackendError::Dispatch(format!("{kernel_name}: missing quantized weight metadata"))
        })?;
        let pt = packed_tensor_from_meta(typed_data, meta, kernel_name)?;
        let out_f32: &mut [f32] = unsafe { arena.view_f32_mut(out_start, out_end - out_start) };

        crate::backend::cpu::microkernels::gemm_cpu_flat_i8_i4x8(
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
