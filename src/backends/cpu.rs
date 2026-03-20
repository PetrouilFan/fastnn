use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;

/// Generic GEMV (matrix × vector) on CPU using packed representation.
/// weights: [M, K] packed tensor
/// activation: [K] f32 vector
/// output: [M] f32 vector
pub fn gemv_cpu<T: PackedWord>(weights: &PackedTensor<T>, activation: &[f32], output: &mut [f32]) {
    let shape = weights.shape();
    assert!(shape.len() >= 2, "Weights must be at least 2D");
    let m = shape[0];
    let k = shape[1];
    assert_eq!(activation.len(), k, "Activation length must match K");
    assert_eq!(output.len(), m, "Output length must match M");

    let k_packed = (k + T::ITEMS - 1) / T::ITEMS;
    let scale = weights.scale();
    let zero = weights.zero();

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let mut acc: f32 = 0.0;
            let row_offset = row * k_packed;
            for p in 0..k_packed {
                let word = weights.as_packed()[row_offset + p];
                let unpacked = word.unpack_to_f32();
                let arr = unpacked.as_ref();
                for j in 0..T::ITEMS {
                    let act_idx = p * T::ITEMS + j;
                    if act_idx < k {
                        acc += arr[j] * activation[act_idx];
                    }
                }
            }
            *out = acc * scale - zero;
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..m {
            let mut acc: f32 = 0.0;
            let row_offset = row * k_packed;
            for p in 0..k_packed {
                let word = weights.as_packed()[row_offset + p];
                let unpacked = word.unpack_to_f32();
                let arr = unpacked.as_ref();
                for j in 0..T::ITEMS {
                    let act_idx = p * T::ITEMS + j;
                    if act_idx < k {
                        acc += arr[j] * activation[act_idx];
                    }
                }
            }
            output[row] = acc * scale - zero;
        }
    }
}

/// Generic GEMM (matrix × matrix) as a batch of GEMV calls on CPU.
/// weights: [M, K] packed tensor
/// batch_inputs: batch of [K] f32 vectors
/// outputs: batch of [M] f32 vectors
pub fn gemm_cpu<T: PackedWord>(
    weights: &PackedTensor<T>,
    batch_inputs: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
) {
    assert_eq!(batch_inputs.len(), outputs.len());
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        batch_inputs
            .par_iter()
            .zip(outputs.par_iter_mut())
            .for_each(|(input, output)| {
                gemv_cpu(weights, input, output);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (input, output) in batch_inputs.iter().zip(outputs.iter_mut()) {
            gemv_cpu(weights, input, output);
        }
    }
}

/// Element-wise ReLU on a packed tensor using SWAR for integer types.
pub fn relu_cpu<T: PackedWord>(tensor: &mut PackedTensor<T>) {
    let packed = tensor.as_packed_mut();

    // For integer types, use SWAR operations directly on raw u32
    if T::BIT_WIDTH == 4 && !T::IS_FLOAT {
        let raw = unsafe {
            std::slice::from_raw_parts_mut(packed.as_mut_ptr() as *mut u32, packed.len())
        };
        for word in raw.iter_mut() {
            *word = crate::swar::ops_4bit::swar_relu_s4x8(*word);
        }
    } else if T::BIT_WIDTH == 8 && !T::IS_FLOAT {
        let raw = unsafe {
            std::slice::from_raw_parts_mut(packed.as_mut_ptr() as *mut u32, packed.len())
        };
        for word in raw.iter_mut() {
            *word = crate::swar::ops_8bit::swar_relu_s8x4(*word);
        }
    } else {
        // For float types, unpack → relu → repack
        for word in packed.iter_mut() {
            let mut arr = word.unpack_to_f32();
            for v in arr.as_mut().iter_mut() {
                *v = v.max(0.0);
            }
            *word = T::pack_from_f32(arr);
        }
    }
}

/// SWAR ReLU backward on CPU.
pub fn relu_backward_cpu<T: PackedWord>(grad: &mut PackedTensor<T>, pre_relu: &PackedTensor<T>) {
    assert_eq!(grad.packed_len(), pre_relu.packed_len());
    let grad_packed = grad.as_packed_mut();
    let pre_packed = pre_relu.as_packed();

    if T::BIT_WIDTH == 4 && !T::IS_FLOAT {
        let grad_raw = unsafe {
            std::slice::from_raw_parts_mut(grad_packed.as_mut_ptr() as *mut u32, grad_packed.len())
        };
        let pre_raw = unsafe {
            std::slice::from_raw_parts(pre_packed.as_ptr() as *const u32, pre_packed.len())
        };
        for (g, p) in grad_raw.iter_mut().zip(pre_raw.iter()) {
            *g = crate::swar::ops_4bit::swar_relu_backward_u4x8(*g, *p);
        }
    } else if T::BIT_WIDTH == 8 && !T::IS_FLOAT {
        let grad_raw = unsafe {
            std::slice::from_raw_parts_mut(grad_packed.as_mut_ptr() as *mut u32, grad_packed.len())
        };
        let pre_raw = unsafe {
            std::slice::from_raw_parts(pre_packed.as_ptr() as *const u32, pre_packed.len())
        };
        for (g, p) in grad_raw.iter_mut().zip(pre_raw.iter()) {
            *g = crate::swar::ops_8bit::swar_relu_backward_u8x4(*g, *p);
        }
    } else {
        // Float types: unpack → compare → repack
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::{F32x1, U4x8, U8x4};

    #[test]
    fn test_gemv_cpu_f32x1() {
        // Simple 2x2 matrix × 2-vector
        let weights_data = vec![1.0, 2.0, 3.0, 4.0];
        let weights = PackedTensor::<F32x1>::from_f32_auto(&weights_data, &[2, 2]);
        let activation = vec![1.0, 1.0];
        let mut output = vec![0.0; 2];
        gemv_cpu(&weights, &activation, &mut output);
        assert_eq!(output[0], 3.0); // 1*1 + 2*1
        assert_eq!(output[1], 7.0); // 3*1 + 4*1
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
}
