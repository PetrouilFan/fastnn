use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;
use crate::swar::ops_4bit::{swar_add_u4x8, swar_max_u4x8, swar_min_u4x8, swar_sub_u4x8};
use crate::swar::ops_8bit::{swar_add_u8x4, swar_max_u8x4, swar_min_u8x4, swar_sub_u8x4};

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Threshold K above which tiled BLAS kernels are used.
/// Below this, per-row SIMD streaming is faster (no temp buffer copy).
const TILED_K_THRESHOLD: usize = 4096;

/// Generic GEMV (matrix × vector) on CPU using packed representation.
#[inline(always)]
pub fn gemv_cpu<T: PackedWord>(weights: &PackedTensor<T>, activation: &[f32], output: &mut [f32]) {
    let k = weights.shape()[1];
    if k > TILED_K_THRESHOLD {
        // Use cache-blocked tiled kernel for large K
        super::packed_blas::gemv_packed_tiled(weights, activation, output);
    } else {
        // Use streaming SIMD per-row kernel for small/medium K
        super::packed_simd::gemv_packed_simd(weights, activation, output);
    }
}

/// Generic GEMM (matrix × matrix) as a batch of GEMV calls on CPU.
/// Uses batched kernel that unpacks each weight row once for all inputs.
#[inline(always)]
pub fn gemm_cpu<T: PackedWord>(
    weights: &PackedTensor<T>,
    batch_inputs: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
) {
    assert_eq!(batch_inputs.len(), outputs.len());
    // Use batched GEMM: unpack weights once, process all inputs
    super::packed_simd::gemm_packed_batched(weights, batch_inputs, outputs);
}

/// Element-wise ReLU on a packed tensor using SWAR for ALL types.
/// Phase 2: SWAR ReLU for float types via IEEE 754 sign bit masking.
#[inline(always)]
pub fn relu_cpu<T: PackedWord>(tensor: &mut PackedTensor<T>) {
    let packed = tensor.as_packed_mut();
    let raw =
        unsafe { std::slice::from_raw_parts_mut(packed.as_mut_ptr() as *mut u32, packed.len()) };

    // Dispatch to appropriate SWAR kernel based on bit width
    // ALL types now use SWAR — no unpack-repack needed
    match (T::BIT_WIDTH, T::IS_FLOAT) {
        (4, false) => {
            // U4x8: no SIMD shortcut for nibble-level ops, use scalar SWAR
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if raw.len() > 4096 {
                    raw.par_iter_mut().for_each(|word| {
                        *word = crate::swar::ops_4bit::swar_relu_s4x8(*word);
                    });
                    return;
                }
            }
            for word in raw.iter_mut() {
                *word = crate::swar::ops_4bit::swar_relu_s4x8(*word);
            }
        }
        (8, false) => {
            // U8x4: AVX2 _mm256_cmpgt_epi8 for per-byte sign detection
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && raw.len() >= 8 {
                    unsafe {
                        relu_swar_simd_u8x4(raw);
                    }
                    return;
                }
            }
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if raw.len() > 4096 {
                    raw.par_iter_mut().for_each(|word| {
                        *word = crate::swar::ops_8bit::swar_relu_s8x4(*word);
                    });
                    return;
                }
            }
            for word in raw.iter_mut() {
                *word = crate::swar::ops_8bit::swar_relu_s8x4(*word);
            }
        }
        (16, true) => {
            // F16x2: use scalar SWAR (bit masking on sign bits is correct and fast)
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if raw.len() > 4096 {
                    raw.par_iter_mut().for_each(|word| {
                        *word = crate::swar::ops_16bit::swar_relu_f16x2(*word);
                    });
                    return;
                }
            }
            for word in raw.iter_mut() {
                *word = crate::swar::ops_16bit::swar_relu_f16x2(*word);
            }
        }
        (32, true) => {
            // F32x1: AVX2 _mm256_max_ps (correct for IEEE 754 f32)
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && raw.len() >= 8 {
                    unsafe {
                        relu_swar_simd_f32(raw);
                    }
                    return;
                }
            }
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if raw.len() > 4096 {
                    raw.par_iter_mut().for_each(|word| {
                        *word = crate::swar::ops_32bit::swar_relu_f32x1(*word);
                    });
                    return;
                }
            }
            for word in raw.iter_mut() {
                *word = crate::swar::ops_32bit::swar_relu_f32x1(*word);
            }
        }
        _ => {
            // Fallback: unpack → relu → repack (shouldn't reach here)
            // All 4 concrete types (U4x8, U8x4, F16x2, F32x1) are matched above,
            // so this arm is dead code. Kept as a safety net for future types.
            for word in packed.iter_mut() {
                let mut arr = word.unpack_to_f32();
                for v in arr.as_mut().iter_mut() {
                    *v = v.max(0.0);
                }
                *word = T::pack_from_f32(arr);
            }
        }
    }
}

/// SWAR ReLU backward on CPU — all types now use bitwise operations.
#[inline(always)]
pub fn relu_backward_cpu<T: PackedWord>(grad: &mut PackedTensor<T>, pre_relu: &PackedTensor<T>) {
    assert_eq!(grad.packed_len(), pre_relu.packed_len());
    let grad_packed = grad.as_packed_mut();
    let pre_packed = pre_relu.as_packed();

    let grad_raw = unsafe {
        std::slice::from_raw_parts_mut(grad_packed.as_mut_ptr() as *mut u32, grad_packed.len())
    };
    let pre_raw =
        unsafe { std::slice::from_raw_parts(pre_packed.as_ptr() as *const u32, pre_packed.len()) };

    match (T::BIT_WIDTH, T::IS_FLOAT) {
        (4, false) => {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if grad_raw.len() > 4096 {
                    grad_raw.par_iter_mut().zip(pre_raw.par_iter()).for_each(|(g, p)| {
                        *g = crate::swar::ops_4bit::swar_relu_backward_u4x8(*g, *p);
                    });
                    return;
                }
            }
            for (g, p) in grad_raw.iter_mut().zip(pre_raw.iter()) {
                *g = crate::swar::ops_4bit::swar_relu_backward_u4x8(*g, *p);
            }
        }
        (8, false) => {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && grad_raw.len() >= 8 {
                    unsafe {
                        relu_backward_swar_simd_u8x4(grad_raw, pre_raw);
                    }
                    return;
                }
            }
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if grad_raw.len() > 4096 {
                    grad_raw.par_iter_mut().zip(pre_raw.par_iter()).for_each(|(g, p)| {
                        *g = crate::swar::ops_8bit::swar_relu_backward_u8x4(*g, *p);
                    });
                    return;
                }
            }
            for (g, p) in grad_raw.iter_mut().zip(pre_raw.iter()) {
                *g = crate::swar::ops_8bit::swar_relu_backward_u8x4(*g, *p);
            }
        }
        (16, true) => {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if grad_raw.len() > 4096 {
                    grad_raw.par_iter_mut().zip(pre_raw.par_iter()).for_each(|(g, p)| {
                        *g = crate::swar::ops_16bit::swar_relu_backward_f16x2(*g, *p);
                    });
                    return;
                }
            }
            for (g, p) in grad_raw.iter_mut().zip(pre_raw.iter()) {
                *g = crate::swar::ops_16bit::swar_relu_backward_f16x2(*g, *p);
            }
        }
        (32, true) => {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && grad_raw.len() >= 8 {
                    unsafe {
                        relu_backward_swar_simd_f32(grad_raw, pre_raw);
                    }
                    return;
                }
            }
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if grad_raw.len() > 4096 {
                    grad_raw.par_iter_mut().zip(pre_raw.par_iter()).for_each(|(g, p)| {
                        *g = crate::swar::ops_32bit::swar_relu_backward_f32x1(*g, *p);
                    });
                    return;
                }
            }
            for (g, p) in grad_raw.iter_mut().zip(pre_raw.iter()) {
                *g = crate::swar::ops_32bit::swar_relu_backward_f32x1(*g, *p);
            }
        }
        _ => {
            for (g_word, p_word) in grad_packed.iter_mut().zip(pre_packed.iter()) {
                let pre_arr = p_word.unpack_to_f32();
                let mut grad_arr = g_word.unpack_to_f32();
                for (g, p) in grad_arr.as_mut().iter_mut().zip(pre_arr.as_ref().iter()) {
                    if *p <= 0.0 {
                        *g = 0.0;
                    }
                }
                *g_word = T::pack_from_f32(grad_arr);
            }
        }
    }
}

/// Compute input gradient for packed linear backward pass.
///
/// grad_input[i] = sum_j grad_output[j] * W[j][i]
/// Where W has shape [out_features, in_features], row-major.
pub fn packed_linear_backward_input_cpu<T: PackedWord>(
    weight: &PackedTensor<T>,
    grad_output: &[f32],
    _cached_input: &[f32],
    out_f: usize,
    in_f: usize,
) -> Vec<f32> {
    assert_eq!(grad_output.len(), out_f);
    let mut grad_input = vec![0.0; in_f];

    for j in 0..out_f {
        let go = grad_output[j];
        if go == 0.0 {
            continue;
        }
        let row_start = j * in_f;
        for i in 0..in_f {
            grad_input[i] += weight.get(row_start + i) * go;
        }
    }

    grad_input
}

/// Accumulate weight gradient for packed linear backward pass.
///
/// master_grad[j * in_f + i] += grad_output[j] * cached_input[i]
/// Accumulates into existing master_grad values.
pub fn packed_linear_backward_weight_cpu<T: PackedWord>(
    _weight: &PackedTensor<T>,
    grad_output: &[f32],
    cached_input: &[f32],
    master_grad: &mut [f32],
    out_f: usize,
    in_f: usize,
) {
    assert_eq!(grad_output.len(), out_f);
    assert_eq!(cached_input.len(), in_f);
    assert_eq!(master_grad.len(), out_f * in_f);

    for j in 0..out_f {
        let go = grad_output[j];
        if go == 0.0 {
            continue;
        }
        let row_start = j * in_f;
        // Vectorize this inner loop with SIMD if available
        for i in 0..in_f {
            master_grad[row_start + i] += go * cached_input[i];
        }
    }
}

/// Fused ReLU forward + backward on CPU — computes both in a single pass
/// sharing sign-bit computation for integer SWAR types.
#[inline(always)]
pub fn relu_fused_cpu<T: PackedWord>(tensor: &mut PackedTensor<T>, grad: &mut PackedTensor<T>) {
    assert_eq!(tensor.packed_len(), grad.packed_len());
    let tensor_packed = tensor.as_packed_mut();
    let grad_packed = grad.as_packed_mut();
    let data_raw = unsafe {
        std::slice::from_raw_parts_mut(tensor_packed.as_mut_ptr() as *mut u32, tensor_packed.len())
    };
    let grad_raw = unsafe {
        std::slice::from_raw_parts_mut(grad_packed.as_mut_ptr() as *mut u32, grad_packed.len())
    };

    match (T::BIT_WIDTH, T::IS_FLOAT) {
        (4, false) => {
            for (d, g) in data_raw.iter_mut().zip(grad_raw.iter_mut()) {
                let (relu_out, mask) = crate::swar::ops_4bit::swar_fused_relu_u4x8(*d);
                *d = relu_out;
                *g = *g & mask;
            }
        }
        (8, false) => {
            for (d, g) in data_raw.iter_mut().zip(grad_raw.iter_mut()) {
                let (relu_out, mask) = crate::swar::ops_8bit::swar_fused_relu_u8x4(*d);
                *d = relu_out;
                *g = *g & mask;
            }
        }
        _ => {
            relu_cpu(tensor);
            relu_backward_cpu(grad, tensor);
        }
    }
}

/// Element-wise addition on packed tensors using SWAR for integer types.
pub fn add_cpu<T: PackedWord>(a: &PackedTensor<T>, b: &PackedTensor<T>) -> PackedTensor<T> {
    assert_eq!(a.shape(), b.shape(), "add_cpu: shape mismatch");
    match (T::BIT_WIDTH, T::IS_FLOAT) {
        (4, false) => {
            let mut result = a.clone();
            for (ra, rb) in result.as_packed_mut().iter_mut().zip(b.as_packed().iter()) {
                let ra_raw = bytemuck::cast::<T, u32>(*ra);
                let rb_raw = bytemuck::cast::<T, u32>(*rb);
                *ra = bytemuck::cast(swar_add_u4x8(ra_raw, rb_raw));
            }
            result
        }
        (8, false) => {
            let mut result = a.clone();
            for (ra, rb) in result.as_packed_mut().iter_mut().zip(b.as_packed().iter()) {
                let ra_raw = bytemuck::cast::<T, u32>(*ra);
                let rb_raw = bytemuck::cast::<T, u32>(*rb);
                *ra = bytemuck::cast(swar_add_u8x4(ra_raw, rb_raw));
            }
            result
        }
        _ => {
            let a_f32 = a.to_f32_vec();
            let b_f32 = b.to_f32_vec();
            let r: Vec<f32> = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x + y).collect();
            let scale = PackedTensor::<T>::compute_scale(&r);
            PackedTensor::from_f32_slice(&r, a.shape(), scale, 0.0)
        }
    }
}

/// Element-wise subtraction on packed tensors using SWAR for integer types.
pub fn sub_cpu<T: PackedWord>(a: &PackedTensor<T>, b: &PackedTensor<T>) -> PackedTensor<T> {
    assert_eq!(a.shape(), b.shape(), "sub_cpu: shape mismatch");
    match (T::BIT_WIDTH, T::IS_FLOAT) {
        (4, false) => {
            let mut result = a.clone();
            for (ra, rb) in result.as_packed_mut().iter_mut().zip(b.as_packed().iter()) {
                let ra_raw = bytemuck::cast::<T, u32>(*ra);
                let rb_raw = bytemuck::cast::<T, u32>(*rb);
                *ra = bytemuck::cast(swar_sub_u4x8(ra_raw, rb_raw));
            }
            result
        }
        (8, false) => {
            let mut result = a.clone();
            for (ra, rb) in result.as_packed_mut().iter_mut().zip(b.as_packed().iter()) {
                let ra_raw = bytemuck::cast::<T, u32>(*ra);
                let rb_raw = bytemuck::cast::<T, u32>(*rb);
                *ra = bytemuck::cast(swar_sub_u8x4(ra_raw, rb_raw));
            }
            result
        }
        _ => {
            let a_f32 = a.to_f32_vec();
            let b_f32 = b.to_f32_vec();
            let r: Vec<f32> = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x - y).collect();
            let scale = PackedTensor::<T>::compute_scale(&r);
            PackedTensor::from_f32_slice(&r, a.shape(), scale, 0.0)
        }
    }
}

/// Element-wise max on packed tensors using SWAR for integer types.
pub fn max_cpu<T: PackedWord>(a: &PackedTensor<T>, b: &PackedTensor<T>) -> PackedTensor<T> {
    assert_eq!(a.shape(), b.shape(), "max_cpu: shape mismatch");
    match (T::BIT_WIDTH, T::IS_FLOAT) {
        (4, false) => {
            let mut result = a.clone();
            for (ra, rb) in result.as_packed_mut().iter_mut().zip(b.as_packed().iter()) {
                let ra_raw = bytemuck::cast::<T, u32>(*ra);
                let rb_raw = bytemuck::cast::<T, u32>(*rb);
                *ra = bytemuck::cast(swar_max_u4x8(ra_raw, rb_raw));
            }
            result
        }
        (8, false) => {
            let mut result = a.clone();
            for (ra, rb) in result.as_packed_mut().iter_mut().zip(b.as_packed().iter()) {
                let ra_raw = bytemuck::cast::<T, u32>(*ra);
                let rb_raw = bytemuck::cast::<T, u32>(*rb);
                *ra = bytemuck::cast(swar_max_u8x4(ra_raw, rb_raw));
            }
            result
        }
        _ => {
            let a_f32 = a.to_f32_vec();
            let b_f32 = b.to_f32_vec();
            let r: Vec<f32> = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x.max(*y)).collect();
            let scale = PackedTensor::<T>::compute_scale(&r);
            PackedTensor::from_f32_slice(&r, a.shape(), scale, 0.0)
        }
    }
}

/// Element-wise min on packed tensors using SWAR for integer types.
pub fn min_cpu<T: PackedWord>(a: &PackedTensor<T>, b: &PackedTensor<T>) -> PackedTensor<T> {
    assert_eq!(a.shape(), b.shape(), "min_cpu: shape mismatch");
    match (T::BIT_WIDTH, T::IS_FLOAT) {
        (4, false) => {
            let mut result = a.clone();
            for (ra, rb) in result.as_packed_mut().iter_mut().zip(b.as_packed().iter()) {
                let ra_raw = bytemuck::cast::<T, u32>(*ra);
                let rb_raw = bytemuck::cast::<T, u32>(*rb);
                *ra = bytemuck::cast(swar_min_u4x8(ra_raw, rb_raw));
            }
            result
        }
        (8, false) => {
            let mut result = a.clone();
            for (ra, rb) in result.as_packed_mut().iter_mut().zip(b.as_packed().iter()) {
                let ra_raw = bytemuck::cast::<T, u32>(*ra);
                let rb_raw = bytemuck::cast::<T, u32>(*rb);
                *ra = bytemuck::cast(swar_min_u8x4(ra_raw, rb_raw));
            }
            result
        }
        _ => {
            let a_f32 = a.to_f32_vec();
            let b_f32 = b.to_f32_vec();
            let r: Vec<f32> = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x.min(*y)).collect();
            let scale = PackedTensor::<T>::compute_scale(&r);
            PackedTensor::from_f32_slice(&r, a.shape(), scale, 0.0)
        }
    }
}

// ============================================================
// SIMD SWAR ReLU — per-type AVX2 implementations
// ============================================================

/// SIMD SWAR ReLU for U8x4 — zero out bytes where sign bit is set.
/// Uses AVX2 _mm256_cmpgt_epi8 for per-byte sign detection.
/// Correctly handles sub-element boundaries (no cross-byte contamination).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_swar_simd_u8x4(raw: &mut [u32]) {
    let len = raw.len();
    let mut i = 0;
    let zero = _mm256_setzero_si256();
    let ptr = raw.as_mut_ptr();

    while i + 8 <= len {
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        // _mm256_cmpgt_epi8(zero, v): 0xFF per byte where v < 0 (sign bit set)
        let neg_mask = _mm256_cmpgt_epi8(zero, v);
        // AND with complement: zero out negative bytes
        let result = _mm256_andnot_si256(neg_mask, v);
        _mm256_storeu_si256(ptr.add(i) as *mut __m256i, result);
        i += 8;
    }

    // Scalar tail
    while i < len {
        raw[i] = crate::swar::ops_8bit::swar_relu_s8x4(raw[i]);
        i += 1;
    }
}

/// SIMD SWAR ReLU for F32x1 — sign bit is bit 31.
/// Processes 8 f32 values at once using AVX2 _mm256_max_ps.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_swar_simd_f32(raw: &mut [u32]) {
    let len = raw.len();
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    let ptr = raw.as_mut_ptr() as *mut f32;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        let result = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(ptr.add(i), result);
        i += 8;
    }

    while i < len {
        let v = f32::from_bits(raw[i]);
        if v < 0.0 {
            raw[i] = 0;
        }
        i += 1;
    }
}

/// SIMD SWAR ReLU backward for U8x4 — zero out gradient bytes where pre_relu <= 0.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_backward_swar_simd_u8x4(grad_raw: &mut [u32], pre_raw: &[u32]) {
    let len = grad_raw.len().min(pre_raw.len());
    let mut i = 0;
    let zero = _mm256_setzero_si256();
    let g_ptr = grad_raw.as_mut_ptr();
    let p_ptr = pre_raw.as_ptr();

    while i + 8 <= len {
        let g = _mm256_loadu_si256(g_ptr.add(i) as *const __m256i);
        let p = _mm256_loadu_si256(p_ptr.add(i) as *const __m256i);
        // Per-byte signed compare: 0xFF where p > 0
        let mask = _mm256_cmpgt_epi8(p, zero);
        let result = _mm256_and_si256(g, mask);
        _mm256_storeu_si256(g_ptr.add(i) as *mut __m256i, result);
        i += 8;
    }

    while i < len {
        grad_raw[i] = crate::swar::ops_8bit::swar_relu_backward_u8x4(grad_raw[i], pre_raw[i]);
        i += 1;
    }
}

/// SIMD SWAR ReLU backward for F32x1 — zero out gradient where pre_relu <= 0.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_backward_swar_simd_f32(grad_raw: &mut [u32], pre_raw: &[u32]) {
    let len = grad_raw.len().min(pre_raw.len());
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    let g_ptr = grad_raw.as_mut_ptr() as *mut f32;
    let p_ptr = pre_raw.as_ptr() as *const f32;

    while i + 8 <= len {
        let g = _mm256_loadu_ps(g_ptr.add(i));
        let p = _mm256_loadu_ps(p_ptr.add(i));
        // Compare: all-ones per lane where p > 0
        let mask = _mm256_cmp_ps(p, zero, _CMP_GT_OQ);
        let result = _mm256_and_ps(g, mask);
        _mm256_storeu_ps(g_ptr.add(i), result);
        i += 8;
    }

    while i < len {
        let pre_val = f32::from_bits(pre_raw[i]);
        if pre_val <= 0.0 {
            grad_raw[i] = 0;
        }
        i += 1;
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::{F16x2, F32x1, U4x8, U8x4};

    #[test]
    fn test_gemv_cpu_f32x1() {
        let weights_data = vec![1.0, 2.0, 3.0, 4.0];
        let weights = PackedTensor::<F32x1>::from_f32_auto(&weights_data, &[2, 2]);
        let activation = vec![1.0, 1.0];
        let mut output = vec![0.0; 2];
        gemv_cpu(&weights, &activation, &mut output);
        assert_eq!(output[0], 3.0);
        assert_eq!(output[1], 7.0);
    }

    #[test]
    fn test_relu_cpu_u4x8() {
        let data: Vec<f32> = (0..8).map(|i| (i as f32) - 4.0).collect();
        let mut t = PackedTensor::<U4x8>::from_f32_auto(&data, &[8]);
        relu_cpu(&mut t);
        let result = t.to_f32_vec();
        for v in result {
            assert!(
                v >= 0.0,
                "ReLU should produce non-negative values, got {}",
                v
            );
        }
    }

    #[test]
    fn test_relu_cpu_u8x4() {
        let data: Vec<f32> = vec![-50.0, 0.0, 50.0, 100.0];
        let mut t = PackedTensor::<U8x4>::from_f32_auto(&data, &[4]);
        relu_cpu(&mut t);
        let result = t.to_f32_vec();
        for v in result {
            assert!(
                v >= 0.0,
                "ReLU should produce non-negative values, got {}",
                v
            );
        }
    }

    #[test]
    fn test_relu_cpu_f16x2() {
        let data: Vec<f32> = vec![1.5, -2.5, -1.0, 3.0];
        let mut t = PackedTensor::<F16x2>::from_f32_auto(&data, &[4]);
        relu_cpu(&mut t);
        let result = t.to_f32_vec();
        // Positive values preserved, negative zeroed
        assert!(result[0] > 0.0, "1.5 should be positive");
        assert_eq!(result[1], 0.0, "-2.5 should be zeroed");
        assert_eq!(result[2], 0.0, "-1.0 should be zeroed");
        assert!(result[3] > 0.0, "3.0 should be positive");
    }

    #[test]
    fn test_relu_cpu_f32x1() {
        let data: Vec<f32> = vec![1.5, -2.5, 0.0, 3.0];
        let mut t = PackedTensor::<F32x1>::from_f32_auto(&data, &[4]);
        relu_cpu(&mut t);
        let result = t.to_f32_vec();
        assert!(result[0] > 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
        assert!(result[3] > 0.0);
    }
}
