//! Integration test: 2-layer packed network forward + backward + optimizer.

use fastnn::backends::cpu;
use fastnn::dtypes::{F16x2, F32x1, PackedWord, U4x8, U8x4};
use fastnn::packed_tensor::PackedTensor;
use fastnn::packed_train::MasterWeightOptimizer;

/// Test a 2-layer forward pass with packed types.
/// Layer 1: [in_features → hidden], Layer 2: [hidden → out_features]
fn test_forward_pass<T: PackedWord>(in_features: usize, hidden: usize, out_features: usize) {
    // Create weight matrices
    let w1_data: Vec<f32> = (0..hidden * in_features)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();
    let w2_data: Vec<f32> = (0..out_features * hidden)
        .map(|i| (i as f32 * 0.01).cos() * 0.5)
        .collect();

    let w1 = PackedTensor::<T>::from_f32_auto(&w1_data, &[hidden, in_features]);
    let w2 = PackedTensor::<T>::from_f32_auto(&w2_data, &[out_features, hidden]);

    // Input
    let input: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.1).sin()).collect();

    // Forward: layer 1
    let mut hidden_out = vec![0.0f32; hidden];
    cpu::gemv_cpu(&w1, &input, &mut hidden_out);

    // ReLU
    let mut hidden_packed = PackedTensor::<T>::from_f32_auto(&hidden_out, &[hidden]);
    cpu::relu_cpu(&mut hidden_packed);
    let hidden_activated = hidden_packed.to_f32_vec();

    // Forward: layer 2
    let mut output = vec![0.0f32; out_features];
    cpu::gemv_cpu(&w2, &hidden_activated, &mut output);

    // Verify output is finite and non-zero
    for (i, o) in output.iter().enumerate() {
        assert!(o.is_finite(), "Output {} should be finite, got {}", i, o);
    }

    // At least some outputs should be non-zero (unless weights are all zero)
    let nonzero_count = output.iter().filter(|v| v.abs() > 1e-6).count();
    assert!(nonzero_count > 0, "Expected some non-zero outputs");
}

/// Test optimizer step on packed weights.
fn test_optimizer_step<T: PackedWord>() {
    let master: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut opt = MasterWeightOptimizer::<T>::new(master.clone(), 0.01, (0.9, 0.999), 1e-8, 0.0);

    let grad: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).cos() * 0.1).collect();

    // Take a step
    let packed = opt.step(&grad);

    // Verify step count
    assert_eq!(opt.step, 1);

    // Verify master weights changed
    for (i, (m, orig)) in opt.master.iter().zip(master.iter()).enumerate() {
        assert!(
            (m - orig).abs() > 1e-6,
            "Master weight {} should have changed: was {}, now {}",
            i,
            orig,
            m
        );
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
fn test_relu_correctness<T: PackedWord>(bit_width: usize) {
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
}

#[test]
fn test_full_pipeline_f32x1() {
    test_forward_pass::<F32x1>(64, 32, 16);
    test_optimizer_step::<F32x1>();
    test_batched_inference::<F32x1>();
    test_relu_correctness::<F32x1>(32);
}

#[test]
fn test_full_pipeline_f16x2() {
    test_forward_pass::<F16x2>(64, 32, 16);
    test_optimizer_step::<F16x2>();
    test_batched_inference::<F16x2>();
    test_relu_correctness::<F16x2>(16);
}

#[test]
fn test_full_pipeline_u8x4() {
    test_forward_pass::<U8x4>(64, 32, 16);
    test_optimizer_step::<U8x4>();
    test_batched_inference::<U8x4>();
    test_relu_correctness::<U8x4>(8);
}

#[test]
fn test_full_pipeline_u4x8() {
    test_forward_pass::<U4x8>(64, 32, 16);
    test_optimizer_step::<U4x8>();
    test_batched_inference::<U4x8>();
    test_relu_correctness::<U4x8>(4);
}

#[test]
fn test_large_model_u4x8() {
    // Test with model sizes similar to real use (512→2048→512)
    test_forward_pass::<U4x8>(512, 2048, 512);
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
        MasterWeightOptimizer::<U8x4>::new(w1_master.clone(), 0.1, (0.9, 0.999), 1e-8, 0.0);
    let mut opt2 =
        MasterWeightOptimizer::<U8x4>::new(w2_master.clone(), 0.1, (0.9, 0.999), 1e-8, 0.0);

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
        w1_master = opt1.master.clone();
        w2_master = opt2.master.clone();

        let packed1 = opt1.step(&grad_w1);
        let packed2 = opt2.step(&grad_w2);

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
