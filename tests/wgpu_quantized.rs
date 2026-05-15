//! Tests for WGPU quantized inference (U4/U8 matmul & conv2d).
//!
//! Note: These tests require a GPU and are `#[ignore]` by default.
//! Run with `cargo test -- --include-ignored` on a machine with a GPU.

use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::wgpu::WgpuBackend;
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{DimExpr, IrDType, TensorType};

/// Helper: run matmul through the full pipeline with a given backend and quantization.
fn run_matmul<B: fastnn::backend::Backend>(
    batch: usize,
    k: usize,
    n: usize,
    weight_data: &[f32],
    input_data: &[f32],
    quantize: Option<u8>,
    backend: B,
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
        .compile_and_execute_with_quantize(&[&output], backend, &[&input_bytes], quantize)
        .expect("compile_and_execute should succeed");

    bytemuck::cast_slice(&result[0]).to_vec()
}

/// Helper: run conv2d through the full pipeline with a given backend and quantization.
fn run_conv2d<B: fastnn::backend::Backend>(
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
    backend: B,
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
        .compile_and_execute_with_quantize(&[&output], backend, &[&input_bytes], quantize)
        .expect("compile_and_execute should succeed");

    bytemuck::cast_slice(&result[0]).to_vec()
}

// ── MatMul GPU tests ───────────────────────────────────────────────────

#[test]
#[ignore] // Requires GPU
fn test_wgpu_matmul_u4() {
    let weight: Vec<f32> = (0..32).map(|i| (i as f32) / 8.0).collect();
    let input: Vec<f32> = vec![0.5; 16];

    let output_u4 = run_matmul(2, 8, 4, &weight, &input, Some(4), WgpuBackend);
    assert_eq!(output_u4.len(), 8, "U4 output should have 8 elements");
    for &v in &output_u4 {
        assert!(v.is_finite(), "U4 output should be finite, got {}", v);
    }
}

#[test]
#[ignore] // Requires GPU
fn test_wgpu_matmul_u8() {
    let weight: Vec<f32> = (0..32).map(|i| (i as f32) / 8.0).collect();
    let input: Vec<f32> = vec![0.5; 16];

    let output_u8 = run_matmul(2, 8, 4, &weight, &input, Some(8), WgpuBackend);
    assert_eq!(output_u8.len(), 8, "U8 output should have 8 elements");
    for &v in &output_u8 {
        assert!(v.is_finite(), "U8 output should be finite, got {}", v);
    }
}

#[test]
#[ignore] // Requires GPU
fn test_wgpu_matmul_u4_cpu_comparison() {
    let weight: Vec<f32> = (0..32).map(|i| (i as f32) / 8.0).collect();
    let input: Vec<f32> = vec![0.5; 16];

    let cpu_result = run_matmul(2, 8, 4, &weight, &input, Some(4), CpuBackend);
    let gpu_result = run_matmul(2, 8, 4, &weight, &input, Some(4), WgpuBackend);

    assert_eq!(cpu_result.len(), gpu_result.len());
    for (i, (c, g)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
        let diff = (c - g).abs();
        let tolerance = if c.abs() > 0.1 { c.abs() * 0.01 } else { 0.01 };
        assert!(
            diff < tolerance,
            "GPU vs CPU mismatch at [{}]: GPU={} CPU={} diff={}",
            i, g, c, diff
        );
    }
}

#[test]
#[ignore] // Requires GPU
fn test_wgpu_matmul_u8_cpu_comparison() {
    let weight: Vec<f32> = (0..32).map(|i| (i as f32) / 8.0).collect();
    let input: Vec<f32> = vec![0.5; 16];

    let cpu_result = run_matmul(2, 8, 4, &weight, &input, Some(8), CpuBackend);
    let gpu_result = run_matmul(2, 8, 4, &weight, &input, Some(8), WgpuBackend);

    assert_eq!(cpu_result.len(), gpu_result.len());
    for (i, (c, g)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
        let diff = (c - g).abs();
        let tolerance = if c.abs() > 0.1 { c.abs() * 0.01 } else { 0.01 };
        assert!(
            diff < tolerance,
            "GPU vs CPU mismatch at [{}]: GPU={} CPU={} diff={}",
            i, g, c, diff
        );
    }
}

// ── Conv2d GPU tests ───────────────────────────────────────────────────

#[test]
#[ignore] // Requires GPU
fn test_wgpu_conv2d_u4() {
    let weight: Vec<f32> = (0..18).map(|i| (i as f32) * 0.1).collect();
    let input: Vec<f32> = vec![0.5; 25];

    let output_u4 = run_conv2d(
        1, 1, 2, 5, 5, 3, 1, 0, &weight, &input, Some(4), WgpuBackend,
    );
    assert_eq!(output_u4.len(), 18, "U4 conv2d output should have 18 elements");
    for &v in &output_u4 {
        assert!(v.is_finite(), "U4 conv2d output should be finite, got {}", v);
    }
}

#[test]
#[ignore] // Requires GPU
fn test_wgpu_conv2d_u8() {
    let weight: Vec<f32> = (0..18).map(|i| (i as f32) * 0.1).collect();
    let input: Vec<f32> = vec![0.5; 25];

    let output_u8 = run_conv2d(
        1, 1, 2, 5, 5, 3, 1, 0, &weight, &input, Some(8), WgpuBackend,
    );
    assert_eq!(output_u8.len(), 18, "U8 conv2d output should have 18 elements");
    for &v in &output_u8 {
        assert!(v.is_finite(), "U8 conv2d output should be finite, got {}", v);
    }
}

#[test]
#[ignore] // Requires GPU
fn test_wgpu_conv2d_u4_cpu_comparison() {
    let weight: Vec<f32> = (0..18).map(|i| (i as f32) * 0.1).collect();
    let input: Vec<f32> = vec![0.5; 25];

    let cpu_result = run_conv2d(
        1, 1, 2, 5, 5, 3, 1, 0, &weight, &input, Some(4), CpuBackend,
    );
    let gpu_result = run_conv2d(
        1, 1, 2, 5, 5, 3, 1, 0, &weight, &input, Some(4), WgpuBackend,
    );

    assert_eq!(cpu_result.len(), gpu_result.len());
    for (i, (c, g)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
        let diff = (c - g).abs();
        let tolerance = if c.abs() > 0.1 { c.abs() * 0.05 } else { 0.05 };
        assert!(
            diff < tolerance,
            "GPU vs CPU mismatch at [{}]: GPU={} CPU={} diff={}",
            i, g, c, diff
        );
    }
}
