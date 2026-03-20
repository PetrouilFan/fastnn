use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Generic GEMV (matrix × vector) on CPU using packed representation.
/// Uses streaming SIMD unpack+FMA (optimal for single vector multiplication).
pub fn gemv_cpu<T: PackedWord>(weights: &PackedTensor<T>, activation: &[f32], output: &mut [f32]) {
    // Streaming SIMD: unpack word → FMA immediately, no temp buffers
    super::packed_simd::gemv_packed_simd(weights, activation, output);
}

/// Generic GEMM (matrix × matrix) as a batch of GEMV calls on CPU.
/// Uses batched kernel that unpacks each weight row once for all inputs.
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
pub fn relu_cpu<T: PackedWord>(tensor: &mut PackedTensor<T>) {
    let packed = tensor.as_packed_mut();
    let raw =
        unsafe { std::slice::from_raw_parts_mut(packed.as_mut_ptr() as *mut u32, packed.len()) };

    // Dispatch to appropriate SWAR kernel based on bit width
    // ALL types now use SWAR — no unpack-repack needed
    match (T::BIT_WIDTH, T::IS_FLOAT) {
        (4, false) => {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && raw.len() >= 8 {
                    unsafe { relu_swar_simd_u32(raw, 0x0F0F0F0F, 0x88888888); }
                    return;
                }
            }
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
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && raw.len() >= 8 {
                    unsafe { relu_swar_simd_u32(raw, 0x00FF00FF, 0x80808080); }
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
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && raw.len() >= 8 {
                    unsafe { relu_swar_simd_u32(raw, 0x0000FFFF, 0x80008000); }
                    return;
                }
            }
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
            // F32x1: SWAR ReLU using IEEE 754 sign bit
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && raw.len() >= 8 {
                    unsafe { relu_swar_simd_f32(raw); }
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
            for (g, p) in grad_raw.iter_mut().zip(pre_raw.iter()) {
                *g = crate::swar::ops_4bit::swar_relu_backward_u4x8(*g, *p);
            }
        }
        (8, false) => {
            for (g, p) in grad_raw.iter_mut().zip(pre_raw.iter()) {
                *g = crate::swar::ops_8bit::swar_relu_backward_u8x4(*g, *p);
            }
        }
        (16, true) => {
            for (g, p) in grad_raw.iter_mut().zip(pre_raw.iter()) {
                *g = crate::swar::ops_16bit::swar_relu_backward_f16x2(*g, *p);
            }
        }
        (32, true) => {
            for (g, p) in grad_raw.iter_mut().zip(pre_raw.iter()) {
                *g = crate::swar::ops_32bit::swar_relu_backward_f32x1(*g, *p);
            }
        }
        _ => {
            // Fallback
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

// ============================================================
// SIMD SWAR ReLU — process 8 u32 words at once via AVX2
// ============================================================

/// SIMD SWAR ReLU for integer types (4-bit, 8-bit, 16-bit).
/// Processes 8 u32 words at once using AVX2, spreading sign bits
/// to fill each lane and ANDing with complement.
///
/// `data_mask`: mask for data bits (0x0F0F0F0F for 4-bit, etc.)
/// `sign_mask`: mask for sign bits (0x88888888 for 4-bit, etc.)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_swar_simd_u32(raw: &mut [u32], _data_mask: u32, sign_mask: u32) {
    let len = raw.len();
    let mut i = 0;

    // Process 8 u32 words at a time (256 bits = 1 AVX2 register)
    let mask_vec = _mm256_set1_epi32(sign_mask as i32);
    let ptr = raw.as_mut_ptr();

    while i + 8 <= len {
        // Load 8 u32 words
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);

        // Extract sign bits
        let signs = _mm256_and_si256(v, mask_vec);

        // Spread sign bits to fill entire lanes (unrolled — _mm256_srli_epi32 needs const)
        let mut spread = signs;
        spread = _mm256_or_si256(spread, _mm256_srli_epi32(spread, 1));
        spread = _mm256_or_si256(spread, _mm256_srli_epi32(spread, 2));
        spread = _mm256_or_si256(spread, _mm256_srli_epi32(spread, 4));
        spread = _mm256_or_si256(spread, _mm256_srli_epi32(spread, 8));
        spread = _mm256_or_si256(spread, _mm256_srli_epi32(spread, 16));

        // Zero out negative lanes
        let result = _mm256_andnot_si256(spread, v);
        _mm256_storeu_si256(ptr.add(i) as *mut __m256i, result);

        i += 8;
    }

    // Scalar tail
    if sign_mask == 0x88888888 {
        while i < len {
            raw[i] = crate::swar::ops_4bit::swar_relu_s4x8(raw[i]);
            i += 1;
        }
    } else if sign_mask == 0x80808080 {
        while i < len {
            raw[i] = crate::swar::ops_8bit::swar_relu_s8x4(raw[i]);
            i += 1;
        }
    } else {
        while i < len {
            raw[i] = crate::swar::ops_16bit::swar_relu_f16x2(raw[i]);
            i += 1;
        }
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
