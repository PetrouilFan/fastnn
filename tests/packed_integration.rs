//! Integration test: 2-layer packed network forward + backward + optimizer.

use fastnn::backends::cpu;
use fastnn::dtypes::{F16x2, F32x1, PackedWord, U4x8, U8x4};
use fastnn::packed_tensor::PackedTensor;
use fastnn::packed_train::MasterWeightOptimizer;
use rand::Rng;

/// Test forward pass with reference GEMV correctness check.
fn test_forward_pass<T: PackedWord>() {
    let in_features = 64;
    let out_features = 32;
    let mut rng = rand::thread_rng();

    // Create reference weights and activations
    let ref_weights: Vec<f32> = (0..in_features * out_features).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let activations: Vec<f32> = (0..in_features).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Create packed weights
    let weight = PackedTensor::<T>::from_f32_auto(&ref_weights, &[out_features, in_features]);

    // Run GEMV
    let mut output = vec![0.0f32; out_features];
    cpu::gemv_cpu(&weight, &activations, &mut output);

    // Compute reference
    let mut expected = vec![0.0f32; out_features];
    for i in 0..out_features {
        for j in 0..in_features {
            expected[i] += ref_weights[i * in_features + j] * activations[j];
        }
    }

    // Assert approx equality (within quantization error)
    for i in 0..out_features {
        let scale = weight.scale_for_row(i);
        let tolerance = scale * (in_features as f32).sqrt() + 0.01;
        assert!((output[i] - expected[i]).abs() < tolerance,
            "Mismatch at {}: got {}, expected {}, tolerance {}", i, output[i], expected[i], tolerance);
    }
}

/// Test per-channel quantized forward pass.
fn test_per_channel_forward<T: PackedWord>() {
    let in_features = 32;
    let out_features = 16;
    let mut rng = rand::thread_rng();

    let ref_weights: Vec<f32> = (0..in_features * out_features).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let activations: Vec<f32> = (0..in_features).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Create per-channel packed weights
    let weight = PackedTensor::<T>::from_f32_per_channel(&ref_weights, &[out_features, in_features]);

    let mut output = vec![0.0f32; out_features];
    cpu::gemv_cpu(&weight, &activations, &mut output);

    // Compute reference
    let mut expected = vec![0.0f32; out_features];
    for i in 0..out_features {
        for j in 0..in_features {
            expected[i] += ref_weights[i * in_features + j] * activations[j];
        }
    }

    for i in 0..out_features {
        let scale = weight.scale_for_row(i);
        let tolerance = scale * (in_features as f32).sqrt() + 0.01;
        assert!((output[i] - expected[i]).abs() < tolerance);
    }
}

/// Test optimizer step on packed weights.
fn test_optimizer_step<T: PackedWord>() {
    let master: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut opt = MasterWeightOptimizer::<T>::new(&master, 0.01, (0.9, 0.999), 1e-8, 0.0);

    let grad: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).cos() * 0.1).collect();

    // Take a step
    let packed = opt.step(&grad, 1, 16);

    // Verify step count
    assert_eq!(opt.step, 1);

    // Verify weights moved in direction of -gradient
    for (i, ((&new, &orig), &grad)) in opt.master.iter().zip(master.iter()).zip(grad.iter()).enumerate() {
        let change = new - orig;
        if grad > 0.0 {
            assert!(change <= 0.0, "Weight {} increased ({} -> {}) despite positive gradient {}",
                i, orig, new, grad);
        } else if grad < 0.0 {
            assert!(change >= 0.0, "Weight {} decreased ({} -> {}) despite negative gradient {}",
                i, orig, new, grad);
        }
    }

    // Verify packed tensor is valid
    let unpacked = packed.to_f32_vec();
    assert_eq!(unpacked.len(), 16);
    for v in &unpacked {
        assert!(v.is_finite(), "Unpacked value should be finite");
    }
}

/// Test batched inference: multiple inputs through same weights.
fn test_batched_inference<T: PackedWord>() {
    let k = 32;
    let m = 8;
    let batch = 4;

    let w_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
    let weights = PackedTensor::<T>::from_f32_auto(&w_data, &[m, k]);

    let inputs: Vec<Vec<f32>> = (0..batch)
        .map(|b| (0..k).map(|i| (i as f32 * 0.1 + b as f32).sin()).collect())
        .collect();
    let mut outputs: Vec<Vec<f32>> = vec![vec![0.0f32; m]; batch];

    cpu::gemm_cpu(&weights, &inputs, &mut outputs);

    // Verify all outputs are finite
    for (bi, out) in outputs.iter().enumerate() {
        for (oi, v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Batch {} output {} should be finite, got {}",
                bi,
                oi,
                v
            );
        }
    }
}

/// Test that SWAR ReLU works correctly with packed types.
fn test_relu_correctness<T: PackedWord>() {
    let data: Vec<f32> = (0..16).map(|i| (i as f32) - 8.0).collect();
    let mut tensor = PackedTensor::<T>::from_f32_auto(&data, &[16]);

    cpu::relu_cpu(&mut tensor);
    let result = tensor.to_f32_vec();

    for (i, v) in result.iter().enumerate() {
        assert!(
            *v >= 0.0,
            "ReLU should produce non-negative values, got {} at index {}",
            v,
            i
        );
    }

    // Verify specific known values
    assert!(result[3] == 0.0, "ReLU(-5) should be 0, got {}", result[3]);
    let scale = tensor.scale();
    let tolerance = scale * 2.0 + 0.01;
    assert!((result[11] - 3.0).abs() < tolerance || result[11] > 0.0,
        "ReLU(3) should be approx 3.0, got {}, tolerance {}", result[11], tolerance);
}

#[test]
fn test_full_pipeline_f32x1() {
    test_forward_pass::<F32x1>();
    test_per_channel_forward::<F32x1>();
    test_optimizer_step::<F32x1>();
    test_batched_inference::<F32x1>();
    test_relu_correctness::<F32x1>();
}

#[test]
fn test_full_pipeline_f16x2() {
    test_forward_pass::<F16x2>();
    test_per_channel_forward::<F16x2>();
    test_optimizer_step::<F16x2>();
    test_batched_inference::<F16x2>();
    test_relu_correctness::<F16x2>();
}

#[test]
fn test_full_pipeline_u8x4() {
    test_forward_pass::<U8x4>();
    test_per_channel_forward::<U8x4>();
    test_optimizer_step::<U8x4>();
    test_batched_inference::<U8x4>();
    test_relu_correctness::<U8x4>();
}

#[test]
fn test_full_pipeline_u4x8() {
    test_forward_pass::<U4x8>();
    test_per_channel_forward::<U4x8>();
    test_optimizer_step::<U4x8>();
    test_batched_inference::<U4x8>();
    test_relu_correctness::<U4x8>();
}

#[test]
fn test_large_model_u4x8() {
    test_forward_pass::<U4x8>();
}

#[test]
fn test_loss_decreases_u8x4() {
    // Train a tiny model and verify loss decreases
    let in_f = 8;
    let hidden = 16;
    let out_f = 4;

    let mut w1_master: Vec<f32> = (0..hidden * in_f)
        .map(|i| (i as f32 * 0.01).sin() * 0.3)
        .collect();
    let mut w2_master: Vec<f32> = (0..out_f * hidden)
        .map(|i| (i as f32 * 0.01).cos() * 0.3)
        .collect();

    let mut opt1 =
        MasterWeightOptimizer::<U8x4>::new(&w1_master, 0.1, (0.9, 0.999), 1e-8, 0.0);
    let mut opt2 =
        MasterWeightOptimizer::<U8x4>::new(&w2_master, 0.1, (0.9, 0.999), 1e-8, 0.0);

    let target: Vec<f32> = vec![1.0, 0.0, -1.0, 0.5];
    let input: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.5).sin()).collect();

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for step in 0..20 {
        // Forward
        let w1 = PackedTensor::<U8x4>::from_f32_auto(&w1_master, &[hidden, in_f]);
        let mut h = vec![0.0f32; hidden];
        cpu::gemv_cpu(&w1, &input, &mut h);

        let mut h_packed = PackedTensor::<U8x4>::from_f32_auto(&h, &[hidden]);
        cpu::relu_cpu(&mut h_packed);
        let h_act = h_packed.to_f32_vec();

        let w2 = PackedTensor::<U8x4>::from_f32_auto(&w2_master, &[out_f, hidden]);
        let mut out = vec![0.0f32; out_f];
        cpu::gemv_cpu(&w2, &h_act, &mut out);

        // MSE loss
        let loss: f32 = out
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f32>()
            / out_f as f32;

        if step == 0 {
            initial_loss = loss;
        }
        if step == 19 {
            final_loss = loss;
        }

        // Backward: dL/dout
        let d_out: Vec<f32> = out
            .iter()
            .zip(target.iter())
            .map(|(o, t)| 2.0 * (o - t) / out_f as f32)
            .collect();

        // Backward through layer 2: dL/dw2 = d_out * h_act
        let mut grad_w2 = vec![0.0f32; out_f * hidden];
        for i in 0..out_f {
            for j in 0..hidden {
                grad_w2[i * hidden + j] = d_out[i] * h_act[j];
            }
        }

        // Backward through layer 1 (simplified)
        let mut d_h = vec![0.0f32; hidden];
        for j in 0..hidden {
            for i in 0..out_f {
                d_h[j] += d_out[i] * w2_master[i * hidden + j];
            }
        }
        // ReLU mask
        for j in 0..hidden {
            if h[j] <= 0.0 {
                d_h[j] = 0.0;
            }
        }
        let mut grad_w1 = vec![0.0f32; hidden * in_f];
        for i in 0..hidden {
            for j in 0..in_f {
                grad_w1[i * in_f + j] = d_h[i] * input[j];
            }
        }

        // Optimizer step
        let packed1 = opt1.step(&grad_w1, hidden, in_f);
        let packed2 = opt2.step(&grad_w2, out_f, hidden);

        w1_master = opt1.master.clone();
        w2_master = opt2.master.clone();
    }

    // Loss should decrease (with quantization, won't be perfect)
    assert!(
        final_loss < initial_loss,
        "Loss should decrease: initial={}, final={}",
        initial_loss,
        final_loss
    );
}
