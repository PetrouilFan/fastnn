//! CPU microkernels correctness tests — extracted from microkernels.rs

use super::*;
use crate::dtypes::{F32x1, I4x8, I8x4, PackedWord};
use crate::packed_tensor::PackedTensor;

// ============================================================
// OpenBLAS Conv GEMM layout correctness tests
// ============================================================

#[cfg(all(test, feature = "openblas"))]
mod openblas_conv_gemm_tests {
    use super::*;

    fn deterministic(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + phase) * 0.017).sin() * 0.5)
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], eps: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff <= eps,
                "mismatch at {i}: actual={a} expected={e} diff={diff} eps={eps}"
            );
        }
    }

    #[test]
    fn openblas_conv_sgemm_matches_matrixmultiply_for_1x1_layout() {
        let (m, k, n) = (17usize, 31usize, 73usize);
        let a = deterministic(m * k, 1.0);
        let b = deterministic(k * n, 7.0);
        let mut expected = vec![0.0f32; m * n];
        let mut actual = vec![0.0f32; m * n];
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                expected.as_mut_ptr(),
                n as isize,
                1,
            );
            conv_sgemm(
                m,
                k,
                n,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                actual.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        assert_close(&actual, &expected, 2e-5);
    }

    #[test]
    fn openblas_conv_sgemm_matches_matrixmultiply_for_im2col_transb_layout() {
        let (m, k, n) = (19usize, 29usize, 67usize);
        let a = deterministic(m * k, 3.0);
        // General Conv im2col is physically [N,K] row-major but logically B=[K,N].
        let col_nk = deterministic(n * k, 11.0);
        let mut expected = vec![0.0f32; m * n];
        let mut actual = vec![0.0f32; m * n];
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                col_nk.as_ptr(),
                1,
                k as isize,
                0.0,
                expected.as_mut_ptr(),
                n as isize,
                1,
            );
            conv_sgemm(
                m,
                k,
                n,
                a.as_ptr(),
                k as isize,
                1,
                col_nk.as_ptr(),
                1,
                k as isize,
                actual.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        assert_close(&actual, &expected, 2e-5);
    }
}

// ============================================================
// NEON kernel correctness tests
// ============================================================

#[cfg(test)]
mod neon_tests {
    use super::*;

    fn test_vector(len: usize) -> Vec<f32> {
        (0..len).map(|i| (i as f32) * 0.25).collect()
    }

    fn random_vector(len: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    fn assert_f32_slice_eq(actual: &[f32], expected: &[f32], eps: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff < eps,
                "mismatch at index {}: actual={} expected={} diff={}",
                i,
                a,
                e,
                diff
            );
        }
    }

    fn softmax_scalar(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum).collect()
    }

    #[test]
    fn test_fma_f32_scalar() {
        let a = test_vector(16);
        let b = test_vector(16);
        let result = fma_f32_scalar(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-6, "fma scalar mismatch");
    }

    #[test]
    fn test_fma_f32_slice_consistency() {
        let a = random_vector(256);
        let b = random_vector(256);
        let result = fma_f32_slice(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-4, "fma slice mismatch");
    }

    #[test]
    fn test_fma_f32_slice_various_lengths() {
        for len in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 128] {
            let a = random_vector(len);
            let b = random_vector(len);
            let result = fma_f32_slice(&a, &b);
            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            assert!(
                (result - expected).abs() < 1e-4,
                "fma slice mismatch for len={}: result={} expected={}",
                len,
                result,
                expected
            );
        }
    }

    #[test]
    fn test_gemv_generic_fallback_vs_scalar() {
        let m = 4;
        let k = 16;
        let weights_f32 = random_vector(m * k);
        let activations = random_vector(k);

        let packed = PackedTensor::<F32x1>::from_f32_slice(&weights_f32, &[m, k], 1.0, 0.0);

        let mut output_generic = vec![0.0f32; m];
        gemv_generic_fallback(&packed, &activations, &mut output_generic);

        for row in 0..m {
            let dot: f32 = (0..k)
                .map(|j| weights_f32[row * k + j] * activations[j])
                .sum();
            assert!(
                (output_generic[row] - dot).abs() < 1e-4,
                "row {} mismatch: {} vs {}",
                row,
                output_generic[row],
                dot
            );
        }
    }

    #[test]
    fn test_gemv_dispatch_f32x1_vs_scalar() {
        let m = 4;
        let k = 32;
        let weights_f32 = random_vector(m * k);
        let activations = random_vector(k);

        let packed = PackedTensor::<F32x1>::from_f32_slice(&weights_f32, &[m, k], 1.0, 0.0);

        let mut output = vec![0.0f32; m];
        gemv_cpu(&packed, &activations, &mut output);

        for row in 0..m {
            let dot: f32 = (0..k)
                .map(|j| weights_f32[row * k + j] * activations[j])
                .sum();
            assert!(
                (output[row] - dot).abs() < 1e-4,
                "row {} mismatch: {} vs {}",
                row,
                output[row],
                dot
            );
        }
    }

    #[test]
    fn test_gemv_dispatch_i8x4_self_consistency() {
        let m = 4;
        let k = 32;
        let weights_f32: Vec<f32> = (0..m * k).map(|i| ((i % 32) as f32) - 16.0).collect();
        let activations = random_vector(k);

        let packed = PackedTensor::<I8x4>::from_f32_slice(&weights_f32, &[m, k], 1.0, 0.0);

        let mut output_simd = vec![0.0f32; m];
        gemv_cpu(&packed, &activations, &mut output_simd);

        let mut output_fallback = vec![0.0f32; m];
        gemv_generic_fallback(&packed, &activations, &mut output_fallback);

        assert_f32_slice_eq(&output_simd, &output_fallback, 1e-5);
    }

    #[test]
    fn test_gemv_dispatch_i4x8_self_consistency() {
        let m = 4;
        let k = 32;
        let weights_f32: Vec<f32> = (0..m * k).map(|i| ((i % 8) as f32) - 4.0).collect();
        let activations = random_vector(k);

        let packed = PackedTensor::<I4x8>::from_f32_slice(&weights_f32, &[m, k], 1.0, 0.0);

        let mut output_simd = vec![0.0f32; m];
        gemv_cpu(&packed, &activations, &mut output_simd);

        let mut output_fallback = vec![0.0f32; m];
        gemv_generic_fallback(&packed, &activations, &mut output_fallback);

        assert_f32_slice_eq(&output_simd, &output_fallback, 1e-5);
    }

    #[test]
    fn test_simd_dot_product() {
        let a = random_vector(256);
        let b = random_vector(256);
        let result = unsafe { simd_dot_product(&a, &b, a.len()) };
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-4, "simd dot mismatch");
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let logits = vec![1000.0, 1010.0, 1000.0];
        let result = softmax_scalar(&logits);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={} != 1.0", sum);
        assert!(result[1] > 0.5, "softmax should peak at max logit");
    }

    #[test]
    fn test_softmax_consistency() {
        let logits = random_vector(128);
        let result = softmax_scalar(&logits);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={} != 1.0", sum);
        for &v in &result {
            assert!(v >= 0.0 && v <= 1.0, "softmax value {} out of range", v);
        }
    }

    #[test]
    fn test_gemm_cpu_batched_consistency() {
        let batch = 4;
        let m = 4;
        let k = 32;
        let weights_f32 = random_vector(m * k);
        let packed = PackedTensor::<F32x1>::from_f32_slice(&weights_f32, &[m, k], 1.0, 0.0);

        let batch_inputs: Vec<Vec<f32>> = (0..batch).map(|_| random_vector(k)).collect();
        let mut outputs = vec![vec![0.0f32; m]; batch];

        gemm_cpu(&packed, &batch_inputs, &mut outputs);

        for (bi, input) in batch_inputs.iter().enumerate() {
            for row in 0..m {
                let dot: f32 = (0..k).map(|j| weights_f32[row * k + j] * input[j]).sum();
                assert!(
                    (outputs[bi][row] - dot).abs() < 1e-3,
                    "batch {} row {} mismatch: {} vs {}",
                    bi,
                    row,
                    outputs[bi][row],
                    dot
                );
            }
        }
    }

    #[test]
    fn test_gemv_packed_tiled_consistency() {
        let m = 8;
        let k = 128;
        let weights_f32 = random_vector(m * k);
        let activations = random_vector(k);

        let packed = PackedTensor::<F32x1>::from_f32_slice(&weights_f32, &[m, k], 1.0, 0.0);
        let mut output_tiled = vec![0.0f32; m];
        gemv_packed_tiled(&packed, &activations, &mut output_tiled);

        for row in 0..m {
            let dot: f32 = (0..k)
                .map(|j| weights_f32[row * k + j] * activations[j])
                .sum();
            assert!(
                (output_tiled[row] - dot).abs() < 1e-3,
                "row {} mismatch: {} vs {}",
                row,
                output_tiled[row],
                dot
            );
        }
    }

    #[test]
    fn test_gemm_cpu_flat_i8_i8x4_basic() {
        let n = 2;
        let k = 4;
        let m = 1;

        // f32 weights: row 0 = [1, 2, 3, 4], row 1 = [5, 6, 7, 8]
        let weights_f32 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let packed = PackedTensor::<I8x4>::from_f32_slice(&weights_f32, &[n, k], 1.0, 0.0);

        let act_scale = 1.0f32;
        let act_zp = 0.0f32;
        let act_i8: Vec<i8> = vec![1, 2, 3, 4];
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(&act_scale.to_le_bytes());
        payload.extend_from_slice(&act_zp.to_le_bytes());
        for &v in &act_i8 {
            payload.push(v as u8);
        }

        let mut output = vec![0.0f32; m * n];
        gemm_cpu_flat_i8_i8x4(&packed, &payload, &mut output, m, k, n);

        // Expected: row 0 = 1*1 + 2*2 + 3*3 + 4*4 = 30
        //          row 1 = 5*1 + 6*2 + 7*3 + 8*4 = 70
        assert!(
            (output[0] - 30.0).abs() < 1e-3,
            "row 0: expected 30.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 70.0).abs() < 1e-3,
            "row 1: expected 70.0, got {}",
            output[1]
        );
    }

    #[test]
    fn test_gemm_cpu_flat_i8_i8x4_with_scale() {
        let n = 2;
        let k = 8;
        let m = 1;

        let w_scale = 2.0f32;
        let weights_f32: Vec<f32> = (0..n * k).map(|i| i as f32).collect();
        let packed = PackedTensor::<I8x4>::from_f32_slice(&weights_f32, &[n, k], w_scale, 0.0);

        let act_scale = 1.5f32;
        let act_zp = 0.0f32;
        let act_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.5).collect();
        let mut act_i8: Vec<i8> = Vec::with_capacity(k);
        for &v in &act_data {
            act_i8.push((v / act_scale).round().clamp(-128.0, 127.0) as i8);
        }
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(&act_scale.to_le_bytes());
        payload.extend_from_slice(&act_zp.to_le_bytes());
        for &v in &act_i8 {
            payload.push(v as u8);
        }

        let mut output = vec![0.0f32; m * n];
        gemm_cpu_flat_i8_i8x4(&packed, &payload, &mut output, m, k, n);

        // Manual reference
        let quantized_weights: Vec<Vec<i8>> = (0..n)
            .map(|row| {
                (0..k)
                    .map(|kk| {
                        (weights_f32[row * k + kk] / w_scale)
                            .round()
                            .clamp(-128.0, 127.0) as i8
                    })
                    .collect()
            })
            .collect();

        for row in 0..n {
            let mut dot: i32 = 0;
            for kk in 0..k {
                dot += quantized_weights[row][kk] as i32 * act_i8[kk] as i32;
            }
            let expected = (dot as f32) * w_scale * act_scale;
            assert!(
                (output[row] - expected).abs() < 1e-3,
                "row {}: expected {}, got {}",
                row,
                expected,
                output[row]
            );
        }
    }

    #[test]
    fn test_gemm_cpu_flat_i8_i8x4_batched() {
        let n = 3;
        let k = 4;
        let m = 2;

        let weights_f32: Vec<f32> = (0..n * k).map(|i| ((i % 4) as f32) - 1.5).collect();
        let packed = PackedTensor::<I8x4>::from_f32_slice(&weights_f32, &[n, k], 1.0, 0.0);

        // Use round numbers that are exactly representable as i8
        let act_scale = 1.0f32;
        let act_zp = 0.0f32;
        let act_i8: Vec<i8> = vec![1, 2, 3, 4, -1, -2, -3, -4];
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(&act_scale.to_le_bytes());
        payload.extend_from_slice(&act_zp.to_le_bytes());
        for &v in &act_i8 {
            payload.push(v as u8);
        }

        let mut output = vec![0.0f32; m * n];
        gemm_cpu_flat_i8_i8x4(&packed, &payload, &mut output, m, k, n);

        // Reference: use existing gemm_cpu_flat with the same dequantized activations
        let act_f32: Vec<f32> = act_i8.iter().map(|&v| v as f32).collect();
        let mut ref_output = vec![0.0f32; m * n];
        gemm_cpu_flat::<I8x4>(&packed, &act_f32, &mut ref_output, m, k, n);

        for i in 0..m * n {
            let err = (output[i] - ref_output[i]).abs();
            assert!(
                err < 1e-4,
                "batch {} row {}: i8_i8x4={}, ref={}, err={}",
                i / n,
                i % n,
                output[i],
                ref_output[i],
                err
            );
        }
    }

    #[test]
    fn test_gemm_cpu_flat_i8x4_asymmetric_zero_is_per_input_sum() {
        let n = 1;
        let k = 4;
        let m = 2;

        // Dequantized weight is q_w * 2.0 + 10.0 = [12, 14, 16, 18].
        // The zero/offset contribution must be 10.0 * sum(input), not +10 once.
        let packed =
            PackedTensor::<I8x4>::from_f32_slice(&[12.0, 14.0, 16.0, 18.0], &[n, k], 2.0, 10.0);
        let input = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0];
        let mut output = vec![0.0f32; m * n];

        gemm_cpu_flat::<I8x4>(&packed, &input, &mut output, m, k, n);

        let expected0 = 12.0 * 1.0 + 14.0 * 2.0 + 16.0 * 3.0 + 18.0 * 4.0;
        let expected1 = 12.0 * -1.0 + 14.0 * 0.5 + 16.0 * 2.0 + 18.0 * -3.0;
        assert!(
            (output[0] - expected0).abs() < 1e-4,
            "row0 got {}, expected {}",
            output[0],
            expected0
        );
        assert!(
            (output[1] - expected1).abs() < 1e-4,
            "row1 got {}, expected {}",
            output[1],
            expected1
        );
    }

    #[test]
    fn test_gemm_cpu_flat_i8_i8x4_asymmetric_weight_zero_is_per_activation_sum() {
        let n = 1;
        let k = 4;
        let m = 1;

        // q_w = [1,2,3,4], scale_w=2, zero_w=10 -> w=[12,14,16,18].
        let packed =
            PackedTensor::<I8x4>::from_f32_slice(&[12.0, 14.0, 16.0, 18.0], &[n, k], 2.0, 10.0);
        let act_scale = 1.0f32;
        let act_zp = 0.0f32;
        let act_i8: Vec<i8> = vec![1, 2, 3, 4];
        let mut payload = Vec::new();
        payload.extend_from_slice(&act_scale.to_le_bytes());
        payload.extend_from_slice(&act_zp.to_le_bytes());
        for &v in &act_i8 {
            payload.push(v as u8);
        }

        let mut output = vec![0.0f32; m * n];
        gemm_cpu_flat_i8_i8x4(&packed, &payload, &mut output, m, k, n);

        let expected = 12.0 * 1.0 + 14.0 * 2.0 + 16.0 * 3.0 + 18.0 * 4.0;
        assert!(
            (output[0] - expected).abs() < 1e-4,
            "got {}, expected {}",
            output[0],
            expected
        );
    }

    #[test]
    fn test_gemm_cpu_flat_i8_i4x8_signed_nibbles_match_f32_reference() {
        let n = 2;
        let k = 8;
        let m = 1;

        let weights_f32 = vec![
            -8.0, -4.0, -1.0, 0.0, 1.0, 3.0, 6.0, 7.0, 7.0, 6.0, 3.0, 1.0, 0.0, -1.0, -4.0, -8.0,
        ];
        let packed = PackedTensor::<I4x8>::from_f32_slice(&weights_f32, &[n, k], 1.0, 0.0);

        let act_scale = 1.0f32;
        let act_zp = 0.0f32;
        let act_i8: Vec<i8> = vec![1, -2, 3, -4, 5, -6, 7, -8];
        let mut payload = Vec::new();
        payload.extend_from_slice(&act_scale.to_le_bytes());
        payload.extend_from_slice(&act_zp.to_le_bytes());
        for &v in &act_i8 {
            payload.push(v as u8);
        }

        let mut output = vec![0.0f32; m * n];
        gemm_cpu_flat_i8_i4x8(&packed, &payload, &mut output, m, k, n);

        let act_f32: Vec<f32> = act_i8.iter().map(|&v| v as f32).collect();
        let mut ref_output = vec![0.0f32; m * n];
        gemm_cpu_flat::<I4x8>(&packed, &act_f32, &mut ref_output, m, k, n);

        for i in 0..(m * n) {
            let err = (output[i] - ref_output[i]).abs();
            assert!(
                err < 1e-4,
                "lane {}: i8_i4x8={}, ref={}, err={}",
                i,
                output[i],
                ref_output[i],
                err
            );
        }
    }

    #[test]
    fn test_gemm_cpu_flat_i8_i4x8_asymmetric_weight_zero_is_per_activation_sum() {
        let n = 1;
        let k = 8;
        let m = 1;

        let weights = [8.0, 10.0, 12.0, 14.0, -4.0, -2.0, 0.0, 2.0];
        let packed = PackedTensor::<I4x8>::from_f32_slice(&weights, &[n, k], 2.0, 10.0);
        let act_scale = 1.0f32;
        let act_zp = 0.0f32;
        let act_i8: Vec<i8> = vec![1, 2, 3, 4, -1, -2, -3, -4];
        let mut payload = Vec::new();
        payload.extend_from_slice(&act_scale.to_le_bytes());
        payload.extend_from_slice(&act_zp.to_le_bytes());
        for &v in &act_i8 {
            payload.push(v as u8);
        }

        let mut output = vec![0.0f32; m * n];
        gemm_cpu_flat_i8_i4x8(&packed, &payload, &mut output, m, k, n);

        let expected: f32 = weights
            .iter()
            .zip(act_i8.iter())
            .map(|(w, a)| *w * (*a as f32))
            .sum();
        assert!(
            (output[0] - expected).abs() < 1e-4,
            "got {}, expected {}",
            output[0],
            expected
        );
    }

    #[test]
    fn test_gemm_cpu_flat_i8_i4x8_nonzero_activation_zp_matches_affine_reference() {
        let n = 1;
        let k = 8;
        let m = 1;

        let packed = PackedTensor::<I4x8>::from_f32_slice(
            &[-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0],
            &[n, k],
            2.0,
            0.0,
        );
        let act_scale = 0.5f32;
        let act_zp = 1.5f32;
        let act_i8: Vec<i8> = vec![-4, -3, -2, -1, 0, 1, 2, 3];
        let mut payload = Vec::new();
        payload.extend_from_slice(&act_scale.to_le_bytes());
        payload.extend_from_slice(&act_zp.to_le_bytes());
        for &v in &act_i8 {
            payload.push(v as u8);
        }

        let mut output = vec![0.0f32; m * n];
        gemm_cpu_flat_i8_i4x8(&packed, &payload, &mut output, m, k, n);

        let deq_weights = packed.to_f32_vec();
        let expected: f32 = deq_weights
            .iter()
            .zip(act_i8.iter())
            .map(|(w, a)| *w * ((*a as f32) * act_scale + act_zp))
            .sum();
        assert!(
            (output[0] - expected).abs() < 1e-4,
            "got {}, expected {}",
            output[0],
            expected
        );
    }

    #[test]
    fn test_gemv_cpu_i8x4_asymmetric_zero() {
        let m: usize = 2;
        let k: usize = 4;

        // Dequantized weights: row0=[12,14,16,18], row1=[1,2,3,4]
        // Packed with scale=2, zero=10 for row0 and scale=1, zero=0 for row1
        let w0: Vec<f32> = vec![12.0, 14.0, 16.0, 18.0];
        let w1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        // Per-channel packing for per-row scale/zero
        let scales = vec![2.0, 1.0];
        let zeros = vec![10.0, 0.0];
        let k_packed = k.div_ceil(I8x4::ITEMS);
        let packed_len = m * k_packed;
        let mut packed = PackedTensor::<I8x4>::from_raw(
            vec![I8x4::default(); packed_len],
            vec![m, k],
            scales,
            zeros,
        );
        // Manually pack: q = (x - zero) / scale
        for row in 0..m {
            let row_data = if row == 0 { &w0 } else { &w1 };
            let s = packed.scale_for_row(row);
            let z = packed.zero_for_row(row);
            for word in 0..k_packed {
                let mut arr = [0.0f32; 4];
                for i in 0..4 {
                    let idx = row * k + word * 4 + i;
                    if idx < (row + 1) * k {
                        arr[i] = (row_data[word * 4 + i] - z) / s;
                    }
                }
                let data_mut = std::sync::Arc::make_mut(&mut packed.data);
                data_mut[row * k_packed + word] = I8x4::pack_from_f32(arr);
            }
        }

        let activation: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; m];
        gemv_cpu(&packed, &activation, &mut output);

        // Reference: direct dot product with dequantized weights
        let expected0 = 12.0 * 1.0 + 14.0 * 2.0 + 16.0 * 3.0 + 18.0 * 4.0;
        let expected1 = 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0;
        assert!(
            (output[0] - expected0).abs() < 1e-4,
            "row0: got {}, expected {}",
            output[0],
            expected0
        );
        assert!(
            (output[1] - expected1).abs() < 1e-4,
            "row1: got {}, expected {}",
            output[1],
            expected1
        );
    }

    #[test]
    fn test_gemv_cpu_i4x8_asymmetric_zero() {
        let m: usize = 1;
        let k: usize = 8;

        // I4x8 range is [-8, 7]; with scale=2, zero=10:
        //   w = q * 2 + 10 → representable w: even numbers in [-6, 24]
        let w: Vec<f32> = vec![12.0, 14.0, 16.0, 18.0, 0.0, 2.0, 4.0, 6.0];
        let packed = PackedTensor::<I4x8>::from_f32_slice(&w, &[m, k], 2.0, 10.0);

        // Reference from dequantized packed values
        let deq = packed.to_f32_vec();
        let activation: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, -3.0];
        let mut output = vec![0.0f32; m];
        gemv_cpu(&packed, &activation, &mut output);

        let expected: f32 = deq.iter().zip(activation.iter()).map(|(w, a)| w * a).sum();
        assert!(
            (output[0] - expected).abs() < 1e-4,
            "got {}, expected {}, deq={:?}",
            output[0],
            expected,
            deq
        );
    }

    #[test]
    fn test_gemv_cpu_i8x4_asymmetric_zero_negative_activations() {
        let m: usize = 2;
        let k: usize = 4;

        // The test is designed to fail with the old formula:
        //   old: output = dot * scale + zero
        //   new: output = dot * scale + zero * Σ(activation)
        // With Σ(activation) != 1, the old formula gives wrong results.
        let scales = vec![2.0, 3.0];
        let zeros = vec![10.0, -5.0];
        let k_packed = k.div_ceil(I8x4::ITEMS);
        let packed_len = m * k_packed;

        // Round-trip via from_f32_slice (symmetric, zero=0) then manually set scale/zero
        let w_vals: Vec<f32> = vec![12.0, 4.0, 20.0, -6.0, 7.0, -14.0, 1.0, 13.0];
        let mut packed_data = vec![I8x4::default(); packed_len + 16];
        for row in 0..m {
            let s = scales[row];
            let z = zeros[row];
            for word in 0..k_packed {
                let mut arr = [0.0f32; 4];
                for i in 0..4 {
                    let idx = row * k + word * 4 + i;
                    if idx < (row + 1) * k {
                        arr[i] = (w_vals[idx] - z) / s;
                    }
                }
                packed_data[row * k_packed + word] = I8x4::pack_from_f32(arr);
            }
        }
        let packed = PackedTensor::<I8x4>::from_raw(packed_data, vec![m, k], scales, zeros);

        let activation: Vec<f32> = vec![2.0, -3.0, 1.0, -2.0];
        let mut output = vec![0.0f32; m];
        gemv_cpu(&packed, &activation, &mut output);

        let expected0 = 12.0 * 2.0 + 4.0 * -3.0 + 20.0 * 1.0 + (-6.0) * -2.0;
        let expected1 = 7.0 * 2.0 + (-14.0) * -3.0 + 1.0 * 1.0 + 13.0 * -2.0;
        assert!(
            (output[0] - expected0).abs() < 1e-4,
            "row0: got {}, expected {}",
            output[0],
            expected0
        );
        assert!(
            (output[1] - expected1).abs() < 1e-4,
            "row1: got {}, expected {}",
            output[1],
            expected1
        );
    }

    #[test]
    fn test_gemv_packed_inner_i8x4_asymmetric_zero() {
        let m: usize = 2;
        let k: usize = 4;

        let w_vals: Vec<f32> = vec![5.0, 15.0, 25.0, 35.0, 0.5, 1.5, 2.5, 3.5];
        let scales = vec![2.0, 0.5];
        let zeros = vec![3.0, -1.0];
        let k_packed = k.div_ceil(I8x4::ITEMS);
        let packed_len = m * k_packed;
        let mut packed_data = vec![I8x4::default(); packed_len + 16];
        for row in 0..m {
            let s = scales[row];
            let z = zeros[row];
            for word in 0..k_packed {
                let mut arr = [0.0f32; 4];
                for i in 0..4 {
                    let idx = row * k + word * 4 + i;
                    if idx < (row + 1) * k {
                        arr[i] = (w_vals[idx] - z) / s;
                    }
                }
                packed_data[row * k_packed + word] = I8x4::pack_from_f32(arr);
            }
        }
        let packed = PackedTensor::<I8x4>::from_raw(packed_data, vec![m, k], scales, zeros);

        let activation: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0];
        let mut output = vec![0.0f32; m];
        // Call inner directly to bypass SIMD dispatch
        let k_packed_calc = k.div_ceil(I8x4::ITEMS);
        gemv_packed_inner::<I8x4>(&packed, &activation, &mut output, m, k, k_packed_calc);

        let expected0 = 5.0 * 1.0 + 15.0 * -2.0 + 25.0 * 3.0 + 35.0 * -4.0;
        let expected1 = 0.5 * 1.0 + 1.5 * -2.0 + 2.5 * 3.0 + 3.5 * -4.0;
        assert!(
            (output[0] - expected0).abs() < 1e-4,
            "row0: got {}, expected {}",
            output[0],
            expected0
        );
        assert!(
            (output[1] - expected1).abs() < 1e-4,
            "row1: got {}, expected {}",
            output[1],
            expected1
        );
    }

    #[test]
    fn test_gemv_packed_blocked_i8x4_asymmetric_zero() {
        let m: usize = 2;
        let k: usize = 48;

        let scales = vec![2.0, 10.0];
        let zeros = vec![3.0, -5.0];
        let k_packed = k.div_ceil(I8x4::ITEMS);
        let packed_len = m * k_packed;
        let mut packed_data = vec![I8x4::default(); packed_len + 16];
        let w_vals: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 1.5).collect();
        for row in 0..m {
            let s = scales[row];
            let z = zeros[row];
            for word in 0..k_packed {
                let mut arr = [0.0f32; 4];
                for i in 0..4 {
                    let idx = row * k + word * 4 + i;
                    if idx < (row + 1) * k {
                        arr[i] = (w_vals[idx] - z) / s;
                    }
                }
                packed_data[row * k_packed + word] = I8x4::pack_from_f32(arr);
            }
        }
        let packed = PackedTensor::<I8x4>::from_raw(packed_data, vec![m, k], scales, zeros);

        // Reference from dequantized values (not original w_vals, which lose precision)
        let deq = packed.to_f32_vec();
        let activation: Vec<f32> = (0..k).map(|i| ((i % 7) as f32) - 3.0).collect();
        let mut output = vec![0.0f32; m];
        gemv_packed_blocked::<I8x4>(&packed, &activation, &mut output, m, k, k_packed);

        let expected0: f32 = (0..k).map(|i| deq[i] * activation[i]).sum();
        let expected1: f32 = (0..k).map(|i| deq[k + i] * activation[i]).sum();
        assert!(
            (output[0] - expected0).abs() < 1e-3,
            "row0: got {}, expected {}",
            output[0],
            expected0
        );
        assert!(
            (output[1] - expected1).abs() < 1e-3,
            "row1: got {}, expected {}",
            output[1],
            expected1
        );
    }

    #[test]
    fn test_gemv_packed_tiled_i8x4_asymmetric_zero() {
        let m: usize = 4;
        let k: usize = 8192;

        let scales = vec![2.0; m];
        let zeros = vec![10.0; m];
        let k_packed = k.div_ceil(I8x4::ITEMS);
        let packed_len = m * k_packed;
        let mut packed_data = vec![I8x4::default(); packed_len + 16];
        let w_vals: Vec<f32> = (0..m * k).map(|i| (i % 127) as f32).collect();
        for row in 0..m {
            let s = scales[row];
            let z = zeros[row];
            for word in 0..k_packed {
                let mut arr = [0.0f32; 4];
                for i in 0..4 {
                    let idx = row * k + word * 4 + i;
                    if idx < (row + 1) * k {
                        arr[i] = (w_vals[idx] - z) / s;
                    }
                }
                packed_data[row * k_packed + word] = I8x4::pack_from_f32(arr);
            }
        }
        let packed = PackedTensor::<I8x4>::from_raw(packed_data, vec![m, k], scales, zeros);

        let deq = packed.to_f32_vec();
        let activation: Vec<f32> = (0..k).map(|i| ((i % 31) as f32) - 15.0).collect();
        let mut output = vec![0.0f32; m];
        gemv_packed_tiled::<I8x4>(&packed, &activation, &mut output);

        for row in 0..m {
            let expected: f32 = (0..k).map(|i| deq[row * k + i] * activation[i]).sum();
            assert!(
                (output[row] - expected).abs() < 1e-2,
                "row{}: got {}, expected {}",
                row,
                output[row],
                expected
            );
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[test]
    fn test_neon_compiles_and_dispatches() {
        eprintln!(
            "NEON arch detected — verifying neon feature: {}",
            cfg!(feature = "neon")
        );
        let a = random_vector(128);
        let b = random_vector(128);
        let _result = fma_f32_slice(&a, &b);
    }
}
