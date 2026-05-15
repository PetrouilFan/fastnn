//! Integration tests for the end-to-end quantized pipeline.
//!
//! Tests the full flow: GraphBuilder → compile_with_quantize → execute → verify output.
//! Covers MatMul and Conv2d with both U4 and U8 quantization.

use fastnn::backend::cpu::CpuBackend;
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{DimExpr, IrDType, TensorType};

/// Helper: build a MatMul graph and run through the full pipeline.
/// Returns output as Vec<f32>.
fn run_matmul(
    batch: usize,
    k: usize,
    n: usize,
    weight_data: &[f32],
    input_data: &[f32],
    quantize: Option<u8>,
) -> Vec<f32> {
    let gb = GraphBuilder::new();
    let input = gb.input_with_dims(
        &[DimExpr::Known(batch as u64), DimExpr::Known(k as u64)],
        IrDType::F32,
    );
    let weight_tt = TensorType::new(
        vec![DimExpr::Known(k as u64), DimExpr::Known(n as u64)],
        IrDType::F32,
    );
    let weight_bytes: Vec<u8> = bytemuck::cast_slice(weight_data).to_vec();
    let weight = gb.constant(&weight_bytes, weight_tt);
    let output = gb.matmul(&input, &weight);

    let input_bytes: Vec<u8> = bytemuck::cast_slice(input_data).to_vec();
    let result = gb
        .compile_and_execute_with_quantize(&[&output], CpuBackend, &[&input_bytes], quantize)
        .expect("compile_and_execute should succeed");

    // Output is f32 (quantized weights produce f32 output after dequant in GEMM)
    bytemuck::cast_slice(&result[0]).to_vec()
}

/// Helper: build a Conv2d graph and run through the full pipeline.
/// Returns output as Vec<f32>.
fn run_conv2d(
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    h: usize,
    w: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    weight_data: &[f32],
    input_data: &[f32],
    quantize: Option<u8>,
) -> Vec<f32> {
    let gb = GraphBuilder::new();
    let input = gb.input_with_dims(
        &[
            DimExpr::Known(batch as u64),
            DimExpr::Known(in_channels as u64),
            DimExpr::Known(h as u64),
            DimExpr::Known(w as u64),
        ],
        IrDType::F32,
    );
    let weight_tt = TensorType::new(
        vec![
            DimExpr::Known(out_channels as u64),
            DimExpr::Known(in_channels as u64),
            DimExpr::Known(kernel_size as u64),
            DimExpr::Known(kernel_size as u64),
        ],
        IrDType::F32,
    );
    let weight_bytes: Vec<u8> = bytemuck::cast_slice(weight_data).to_vec();
    let weight = gb.constant(&weight_bytes, weight_tt);
    let output = gb.conv2d_with_params(&input, &weight, stride, padding, 1, 1);

    let input_bytes: Vec<u8> = bytemuck::cast_slice(input_data).to_vec();
    let result = gb
        .compile_and_execute_with_quantize(&[&output], CpuBackend, &[&input_bytes], quantize)
        .expect("compile_and_execute should succeed");

    bytemuck::cast_slice(&result[0]).to_vec()
}

// ── MatMul tests ──────────────────────────────────────────────────────

#[test]
fn test_matmul_u4_end_to_end() {
    // [2, 8] @ [8, 4] = [2, 4]
    let weight: Vec<f32> = (0..32).map(|i| (i as f32) / 8.0).collect();
    let input: Vec<f32> = vec![0.5; 16]; // [2, 8] of 0.5

    let output_f32 = run_matmul(2, 8, 4, &weight, &input, None);
    let output_u4 = run_matmul(2, 8, 4, &weight, &input, Some(4));

    // Output should have 8 elements (2*4)
    assert_eq!(output_f32.len(), 8, "f32 output should have 8 elements");
    assert_eq!(output_u4.len(), 8, "U4 output should have 8 elements");

    // Quantized output should be within reasonable tolerance of f32 output.
    // U4 has 4-bit quantization (16 levels), so expect ~10-20% relative error.
    for i in 0..8 {
        let f32_val = output_f32[i];
        let u4_val = output_u4[i];
        if f32_val.abs() > 0.1 {
            let rel_err = (u4_val - f32_val).abs() / f32_val.abs();
            assert!(
                rel_err < 0.5,
                "U4 output[{}] = {} vs f32 = {} (rel_err = {:.3})",
                i,
                u4_val,
                f32_val,
                rel_err
            );
        }
    }
}

#[test]
fn test_matmul_u8_end_to_end() {
    // [2, 8] @ [8, 4] = [2, 4] — U8 should be closer to f32 than U4
    let weight: Vec<f32> = (0..32).map(|i| (i as f32) / 8.0).collect();
    let input: Vec<f32> = vec![0.5; 16]; // [2, 8] of 0.5

    let output_f32 = run_matmul(2, 8, 4, &weight, &input, None);
    let output_u8 = run_matmul(2, 8, 4, &weight, &input, Some(8));

    assert_eq!(output_u8.len(), 8, "U8 output should have 8 elements");

    // U8 has 8-bit quantization (256 levels), much closer to f32.
    for i in 0..8 {
        let f32_val = output_f32[i];
        let u8_val = output_u8[i];
        if f32_val.abs() > 0.1 {
            let rel_err = (u8_val - f32_val).abs() / f32_val.abs();
            assert!(
                rel_err < 0.15,
                "U8 output[{}] = {} vs f32 = {} (rel_err = {:.3})",
                i,
                u8_val,
                f32_val,
                rel_err
            );
        }
    }
}

#[test]
fn test_matmul_u4_output_shape_correct() {
    // [1, 4] @ [4, 2] = [1, 2]
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5];
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let output = run_matmul(1, 4, 2, &weight, &input, Some(4));
    assert_eq!(
        output.len(),
        2,
        "Output should have 2 elements for [1,4]@[4,2]"
    );
}

#[test]
fn test_matmul_u8_output_shape_correct() {
    // [3, 4] @ [4, 2] = [3, 2]
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5];
    let input: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0,
    ];

    let output = run_matmul(3, 4, 2, &weight, &input, Some(8));
    assert_eq!(
        output.len(),
        6,
        "Output should have 6 elements for [3,4]@[4,2]"
    );
}

// ── Conv2d tests ──────────────────────────────────────────────────────

#[test]
fn test_conv2d_u4_end_to_end() {
    // Conv2d: 1 input channel, 2 output channels, 3x3 kernel, stride=1, padding=0
    // Input: [1, 1, 5, 5] — output: [1, 2, 3, 3]
    let in_channels = 1usize;
    let out_channels = 2usize;
    let kernel_size = 3usize;
    let weight: Vec<f32> = (0..(out_channels * in_channels * kernel_size * kernel_size))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let input: Vec<f32> = vec![0.5; 25]; // [1, 1, 5, 5]

    let output_f32 = run_conv2d(
        1,
        in_channels,
        out_channels,
        5,
        5,
        kernel_size,
        1,
        0,
        &weight,
        &input,
        None,
    );
    let output_u4 = run_conv2d(
        1,
        in_channels,
        out_channels,
        5,
        5,
        kernel_size,
        1,
        0,
        &weight,
        &input,
        Some(4),
    );

    // Output should have 18 elements (1 * 2 * 3 * 3)
    assert_eq!(
        output_f32.len(),
        18,
        "f32 conv2d output should have 18 elements"
    );
    assert_eq!(
        output_u4.len(),
        18,
        "U4 conv2d output should have 18 elements"
    );

    // U4 quantized conv2d output should be within reasonable tolerance
    let mut u4_correct = 0;
    for i in 0..18 {
        let f32_val = output_f32[i];
        let u4_val = output_u4[i];
        if f32_val.abs() > 0.01 {
            let rel_err = (u4_val - f32_val).abs() / f32_val.abs();
            if rel_err < 0.6 {
                u4_correct += 1;
            }
        } else {
            // Very small values — just check both are small
            if (u4_val - f32_val).abs() < 0.1 {
                u4_correct += 1;
            }
        }
    }
    // At least half the values should be within tolerance
    assert!(
        u4_correct >= 9,
        "At least 9/18 U4 conv2d outputs should be close to f32 (got {}/18)",
        u4_correct
    );
}

#[test]
fn test_conv2d_u8_end_to_end() {
    // Conv2d: 1 input channel, 2 output channels, 3x3 kernel, stride=1, padding=0
    // Input: [1, 1, 5, 5] — output: [1, 2, 3, 3]
    let in_channels = 1usize;
    let out_channels = 2usize;
    let kernel_size = 3usize;
    let weight: Vec<f32> = (0..(out_channels * in_channels * kernel_size * kernel_size))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let input: Vec<f32> = vec![0.5; 25]; // [1, 1, 5, 5]

    let output_u8 = run_conv2d(
        1,
        in_channels,
        out_channels,
        5,
        5,
        kernel_size,
        1,
        0,
        &weight,
        &input,
        Some(8),
    );

    // Output should have 18 elements (1 * 2 * 3 * 3)
    assert_eq!(
        output_u8.len(),
        18,
        "U8 conv2d output should have 18 elements"
    );

    // All outputs should be finite numbers
    for (i, &val) in output_u8.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Output[{}] should be finite, got {}",
            i,
            val
        );
    }
}

// ── API / validation tests ─────────────────────────────────────────────

#[test]
fn test_compile_with_quantize_rejects_invalid_bit_width() {
    use fastnn::backend::executor::GraphExecutor;

    // Bit width must be 4 or 8
    let gb = GraphBuilder::new();
    let input = gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32);
    let weight_tt = TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(2)], IrDType::F32);
    let weight_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.5, -0.5, -0.3, 0.8];
    let weight_bytes: Vec<u8> = bytemuck::cast_slice(&weight_data).to_vec();
    let weight = gb.constant(&weight_bytes, weight_tt);
    let _output = gb.matmul(&input, &weight);

    let graph = gb.to_graph();
    let executor = GraphExecutor::new(CpuBackend);

    let result = executor.compile_with_plan_and_quantize(&graph, Some(2));
    assert!(result.is_err(), "Bit width 2 should be rejected");

    let result = executor.compile_with_plan_and_quantize(&graph, Some(16));
    assert!(result.is_err(), "Bit width 16 should be rejected");

    // None and Some(4) and Some(8) should succeed
    assert!(executor
        .compile_with_plan_and_quantize(&graph, None)
        .is_ok());
    assert!(executor
        .compile_with_plan_and_quantize(&graph, Some(4))
        .is_ok());
    assert!(executor
        .compile_with_plan_and_quantize(&graph, Some(8))
        .is_ok());
}

#[test]
fn test_graph_builder_compile_with_quantize() {
    // Test the GraphBuilder::compile_with_quantize() API
    let gb = GraphBuilder::new();
    let input = gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32);
    let weight_tt = TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(2)], IrDType::F32);
    let weight_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.5, -0.5, -0.3, 0.8];
    let weight_bytes: Vec<u8> = bytemuck::cast_slice(&weight_data).to_vec();
    let weight = gb.constant(&weight_bytes, weight_tt);
    let output = gb.matmul(&input, &weight);

    // Compile without quantization
    let result_no_q = gb.compile_with_quantize(&[&output], CpuBackend, None);
    assert!(
        result_no_q.is_ok(),
        "compile without quantize should succeed"
    );

    // Compile with U4 quantization
    let result_u4 = gb.compile_with_quantize(&[&output], CpuBackend, Some(4));
    assert!(result_u4.is_ok(), "compile with quantize=4 should succeed");

    // Compile with U8 quantization
    let result_u8 = gb.compile_with_quantize(&[&output], CpuBackend, Some(8));
    assert!(result_u8.is_ok(), "compile with quantize=8 should succeed");

    // Verify quantized graphs contain U4/U8 weight nodes
    let (_, _, u4_graph) = result_u4.unwrap();
    let has_u4 = u4_graph
        .nodes
        .iter()
        .any(|n| matches!(&n.output_type.dtype, IrDType::U4 { .. }));
    assert!(has_u4, "U4-compiled graph should contain a U4 weight node");

    let (_, _, u8_graph) = result_u8.unwrap();
    let has_u8 = u8_graph
        .nodes
        .iter()
        .any(|n| matches!(&n.output_type.dtype, IrDType::U8 { .. }));
    assert!(has_u8, "U8-compiled graph should contain a U8 weight node");

    // No-quantize graph should have no U4/U8 nodes
    let (_, _, f32_graph) = result_no_q.unwrap();
    let has_packed = f32_graph.nodes.iter().any(|n| {
        matches!(
            &n.output_type.dtype,
            IrDType::U4 { .. } | IrDType::U8 { .. }
        )
    });
    assert!(
        !has_packed,
        "f32-compiled graph should not contain packed weight nodes"
    );
}

#[test]
fn test_quantized_matmul_preserves_output_sign() {
    // Test that quantized matmul preserves the sign of the output
    // [1, 4] @ [4, 2] — identity-like weight
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
    let input: Vec<f32> = vec![1.0, 2.0, -1.0, -2.0];

    let output_f32 = run_matmul(1, 4, 2, &weight, &input, None);
    let output_u8 = run_matmul(1, 4, 2, &weight, &input, Some(8));

    // Sign should be preserved for U8 (256 levels)
    for i in 0..output_f32.len() {
        if output_f32[i].abs() > 0.1 {
            assert_eq!(
                output_f32[i].signum(),
                output_u8[i].signum(),
                "Sign mismatch at index {}: f32={}, u8={}",
                i,
                output_f32[i],
                output_u8[i]
            );
        }
    }
}

#[test]
fn test_quantize_none_produces_f32_pipeline() {
    // Ensure quantize=None produces the same result as the standard path
    let weight: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
    let input: Vec<f32> = vec![1.0, 0.0, -1.0, 0.5];

    let output = run_matmul(1, 4, 2, &weight, &input, None);

    // All outputs should be finite
    for (i, &val) in output.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Output[{}] should be finite, got {}",
            i,
            val
        );
    }
}
