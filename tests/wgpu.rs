//! Integration tests for the WGPU (GPU) backend.
//!
//! Tests all ops that have dedicated GPU shaders: MatMul, element-wise,
//! softmax, convolution, pooling, normalization, reduction, transpose,
//! embedding, and argmax.  Also tests CPU fallback for unsupported ops
//! and multi-op graphs (matmul -> add -> relu -> softmax -> reduce).
//!
//! All tests are `#[ignore]` by default since they require a GPU.
//! Run with:
//!     cargo test --test wgpu -- --ignored
//! Or on a GPU-available CI runner:
//!     cargo test --test wgpu

use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::wgpu::WgpuBackend;
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{DimExpr, IrDType, TensorType};

// ── Helpers ───────────────────────────────────────────────────────────

fn f32_data(values: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

fn read_f32(bytes: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice(bytes).to_vec()
}

fn i64_data(values: &[i64]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

/// Run the same graph on both CPU and GPU backends, returning (cpu_f32, wgpu_f32) per output.
/// Uses a relative tolerance of 1% for values > 0.1, absolute 0.01 otherwise.
fn run_both<B: fastnn::backend::Backend>(
    g: &GraphBuilder,
    outputs: &[&fastnn::ir::builder::GraphTensor],
    inputs: &[&[u8]],
    backend: B,
) -> Vec<Vec<f32>> {
    let result = g.compile_and_execute(outputs, backend, inputs).unwrap();
    result.iter().map(|r| read_f32(r)).collect()
}

fn compare_gpu_vs_cpu(
    g: &GraphBuilder,
    outputs: &[&fastnn::ir::builder::GraphTensor],
    inputs: &[&[u8]],
) -> Vec<(Vec<f32>, Vec<f32>)> {
    let cpu = run_both(g, outputs, inputs, CpuBackend);
    let wgpu = run_both(g, outputs, inputs, WgpuBackend);

    assert_eq!(cpu.len(), wgpu.len());

    cpu.into_iter()
        .zip(wgpu)
        .enumerate()
        .map(|(i, (cpu_f32, wgpu_f32))| {
            assert_eq!(
                cpu_f32.len(),
                wgpu_f32.len(),
                "output {} length mismatch",
                i
            );
            for (j, (c, g)) in cpu_f32.iter().zip(wgpu_f32.iter()).enumerate() {
                let diff = (c - g).abs();
                let tol = if c.abs() > 0.1 { c.abs() * 0.01 } else { 0.01 };
                assert!(
                    diff < tol,
                    "output {} elem {}: GPU={} CPU={} diff={}",
                    i,
                    j,
                    g,
                    c,
                    diff
                );
            }
            (cpu_f32, wgpu_f32)
        })
        .collect()
}

// ── MatMul ────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_wgpu_matmul_small() {
    let g = GraphBuilder::new();
    let a = g.input(&[2, 4], IrDType::F32);
    let b = g.input(&[4, 3], IrDType::F32);
    let c = g.matmul(&a, &b);

    let a_data = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b_data = f32_data(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

    let result = g
        .compile_and_execute(&[&c], WgpuBackend, &[&a_data, &b_data])
        .unwrap();
    let out = read_f32(&result[0]);
    assert_eq!(out.len(), 6);
    for &v in &out {
        assert!(v.is_finite(), "output should be finite, got {}", v);
    }
}

#[test]
#[ignore]
fn test_wgpu_matmul_large() {
    let g = GraphBuilder::new();
    let a = g.input(&[128, 256], IrDType::F32);
    let b = g.input(&[256, 64], IrDType::F32);
    let c = g.matmul(&a, &b);

    let a_data: Vec<f32> = (0..128 * 256).map(|i| (i as f32) / 256.0).collect();
    let b_data: Vec<f32> = (0..256 * 64).map(|i| (i as f32) / 64.0).collect();

    let pairs = compare_gpu_vs_cpu(&g, &[&c], &[&f32_data(&a_data), &f32_data(&b_data)]);
    assert!(!pairs.is_empty());
}

#[test]
#[ignore]
fn test_wgpu_matmul_batched() {
    let g = GraphBuilder::new();
    let a = g.input(&[8, 16, 32], IrDType::F32);
    let b = g.input(&[8, 32, 64], IrDType::F32);
    let c = g.matmul(&a, &b);

    let a_data: Vec<f32> = (0..8 * 16 * 32).map(|i| (i as f32) / 32.0).collect();
    let b_data: Vec<f32> = (0..8 * 32 * 64).map(|i| (i as f32) / 64.0).collect();

    let pairs = compare_gpu_vs_cpu(&g, &[&c], &[&f32_data(&a_data), &f32_data(&b_data)]);
    assert!(!pairs.is_empty());
}

// ── Elementwise ───────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_wgpu_elementwise_add() {
    let g = GraphBuilder::new();
    let a = g.input(&[4], IrDType::F32);
    let b = g.input(&[4], IrDType::F32);
    let c = g.add(&a, &b);

    let pairs = compare_gpu_vs_cpu(
        &g,
        &[&c],
        &[
            &f32_data(&[1.0, 2.0, 3.0, 4.0]),
            &f32_data(&[5.0, 6.0, 7.0, 8.0]),
        ],
    );
    assert_eq!(
        read_f32(&bytemuck::cast_slice(&pairs[0].0)),
        &[6.0, 8.0, 10.0, 12.0]
    );
}

#[test]
#[ignore]
fn test_wgpu_elementwise_relu() {
    let g = GraphBuilder::new();
    let a = g.input(&[5], IrDType::F32);
    let r = g.relu(&a);

    let pairs = compare_gpu_vs_cpu(&g, &[&r], &[&f32_data(&[-2.0, -1.0, 0.0, 1.0, 2.0])]);
    assert_eq!(pairs[0].0, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
#[ignore]
fn test_wgpu_elementwise_gelu() {
    let g = GraphBuilder::new();
    let a = g.input(&[4], IrDType::F32);
    let r = g.gelu(&a);

    let pairs = compare_gpu_vs_cpu(&g, &[&r], &[&f32_data(&[-2.0, 0.0, 1.0, 3.0])]);
    assert_eq!(pairs[0].0.len(), 4);
}

#[test]
#[ignore]
fn test_wgpu_elementwise_sigmoid() {
    let g = GraphBuilder::new();
    let a = g.input(&[4], IrDType::F32);
    let r = g.sigmoid(&a);

    let pairs = compare_gpu_vs_cpu(&g, &[&r], &[&f32_data(&[-2.0, 0.0, 1.0, 5.0])]);
    assert_eq!(pairs[0].0.len(), 4);
}

#[test]
#[ignore]
fn test_wgpu_elementwise_mul_sub_div() {
    let g = GraphBuilder::new();
    let a = g.input(&[2], IrDType::F32);
    let b = g.input(&[2], IrDType::F32);
    let m = g.mul(&a, &b);
    let s = g.sub(&a, &b);
    let d = g.div(&a, &b);

    let a_data = f32_data(&[10.0, 20.0]);
    let b_data = f32_data(&[2.0, 5.0]);

    let pairs = compare_gpu_vs_cpu(&g, &[&m, &s, &d], &[&a_data, &b_data]);
    assert_eq!(pairs[0].0, vec![20.0, 100.0]);
    assert_eq!(pairs[1].0, vec![8.0, 15.0]);
    assert_eq!(pairs[2].0, vec![5.0, 4.0]);
}

#[test]
#[ignore]
fn test_wgpu_elementwise_neg_abs_exp_log_sqrt() {
    let g = GraphBuilder::new();
    let a = g.input(&[4], IrDType::F32);
    let n = g.neg(&a);
    let ab = g.abs(&a);
    let e = g.exp(&a);
    let l = g.log(&a);
    let sq = g.sqrt(&a);

    let a_data = f32_data(&[1.0, 2.0, 3.0, 4.0]);

    compare_gpu_vs_cpu(&g, &[&n, &ab, &e, &l, &sq], &[&a_data]);
}

#[test]
#[ignore]
fn test_wgpu_elementwise_tanh_silu() {
    let g = GraphBuilder::new();
    let a = g.input(&[4], IrDType::F32);
    let t = g.tanh(&a);
    let s = g.silu(&a);

    let a_data = f32_data(&[-2.0, -1.0, 0.0, 2.0]);

    compare_gpu_vs_cpu(&g, &[&t, &s], &[&a_data]);
}

// ── Softmax ───────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_wgpu_softmax_basic() {
    let g = GraphBuilder::new();
    let a = g.input(&[2, 3], IrDType::F32);
    let s = g.softmax(&a, 1);

    let a_data = f32_data(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

    let pairs = compare_gpu_vs_cpu(&g, &[&s], &[&a_data]);
    let softmax_out = &pairs[0].0;
    assert_eq!(softmax_out.len(), 6);
    for &v in softmax_out {
        assert!(v.is_finite(), "softmax values should be finite, got {}", v);
    }
}

#[test]
#[ignore]
fn test_wgpu_softmax_stability() {
    let g = GraphBuilder::new();
    let a = g.input(&[1, 4], IrDType::F32);
    let s = g.softmax(&a, 1);

    let a_data = f32_data(&[100.0, 101.0, 102.0, 103.0]);

    let pairs = compare_gpu_vs_cpu(&g, &[&s], &[&a_data]);
    let softmax_out = &pairs[0].0;
    assert_eq!(softmax_out.len(), 4);
    for &v in softmax_out {
        assert!(
            v.is_finite(),
            "softmax with large values should be stable, got {}",
            v
        );
    }
}

// ── Convolution ───────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_wgpu_conv2d_small() {
    let g = GraphBuilder::new();
    let input = g.input(&[1, 1, 4, 4], IrDType::F32);
    let weight_tt = TensorType::new(
        vec![
            DimExpr::Known(1),
            DimExpr::Known(1),
            DimExpr::Known(3),
            DimExpr::Known(3),
        ],
        IrDType::F32,
    );
    let weight_data: Vec<f32> = vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0];
    let weight_bytes = f32_data(&weight_data);
    let weight = g.constant(&weight_bytes, weight_tt);
    let out = g.conv2d_with_params(&input, &weight, 1, 0, 1, 1);

    let input_data = f32_data(&(0..16).map(|i| i as f32).collect::<Vec<_>>());

    let result = g
        .compile_and_execute(&[&out], WgpuBackend, &[&input_data])
        .unwrap();
    let out_f32 = read_f32(&result[0]);
    assert_eq!(
        out_f32.len(),
        4,
        "conv2d [1,1,4,4] with 3x3 filter, stride 1, pad 0 -> [1,1,2,2]"
    );
    for &v in &out_f32 {
        assert!(v.is_finite(), "conv2d output should be finite, got {}", v);
    }
}

#[test]
#[ignore]
fn test_wgpu_conv2d_multichannel() {
    let g = GraphBuilder::new();
    let input = g.input(&[1, 2, 4, 4], IrDType::F32);
    let weight_tt = TensorType::new(
        vec![
            DimExpr::Known(4),
            DimExpr::Known(2),
            DimExpr::Known(3),
            DimExpr::Known(3),
        ],
        IrDType::F32,
    );
    let weight_data: Vec<f32> = (0..4 * 2 * 3 * 3).map(|i| (i as f32) * 0.1).collect();
    let weight_bytes = f32_data(&weight_data);
    let weight = g.constant(&weight_bytes, weight_tt);
    let out = g.conv2d_with_params(&input, &weight, 1, 0, 1, 1);

    let input_data: Vec<f32> = (0..1 * 2 * 4 * 4).map(|i| (i as f32) * 0.5).collect();

    let result = g
        .compile_and_execute(&[&out], WgpuBackend, &[&f32_data(&input_data)])
        .unwrap();
    let out_f32 = read_f32(&result[0]);
    assert_eq!(out_f32.len(), 4 * 2 * 2, "conv out [1,4,2,2]");
    for &v in &out_f32 {
        assert!(
            v.is_finite(),
            "conv multichannel output should be finite, got {}",
            v
        );
    }
}

// ── Pooling ───────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_wgpu_maxpool2d() {
    let g = GraphBuilder::new();
    let input = g.input(&[1, 1, 4, 4], IrDType::F32);
    let (pooled, _indices) = g.max_pool2d(&input, 2, 2, 0);

    let input_data: Vec<f32> = (0..16).map(|i| i as f32).collect();

    let result = g
        .compile_and_execute(&[&pooled], WgpuBackend, &[&f32_data(&input_data)])
        .unwrap();
    let out_f32 = read_f32(&result[0]);
    assert_eq!(out_f32.len(), 4, "maxpool [1,1,4,4] k=2 s=2 -> [1,1,2,2]");
    for &v in &out_f32 {
        assert!(v.is_finite(), "maxpool output should be finite, got {}", v);
    }
}

#[test]
#[ignore]
fn test_wgpu_avgpool2d() {
    let g = GraphBuilder::new();
    let input = g.input(&[1, 1, 4, 4], IrDType::F32);
    let pooled = g.avg_pool2d(&input, 2, 2, 0);

    let input_data: Vec<f32> = (0..16).map(|i| i as f32).collect();

    let result = g
        .compile_and_execute(&[&pooled], WgpuBackend, &[&f32_data(&input_data)])
        .unwrap();
    let out_f32 = read_f32(&result[0]);
    assert_eq!(out_f32.len(), 4, "avgpool [1,1,4,4] k=2 s=2 -> [1,1,2,2]");
    for &v in &out_f32 {
        assert!(v.is_finite(), "avgpool output should be finite, got {}", v);
    }
}

// ── Normalization ─────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_wgpu_layernorm() {
    let g = GraphBuilder::new();
    let input = g.input(&[2, 4], IrDType::F32);
    let weight_tt = TensorType::new(vec![DimExpr::Known(4)], IrDType::F32);
    let bias_tt = TensorType::new(vec![DimExpr::Known(4)], IrDType::F32);
    let weight = g.constant(&f32_data(&[1.0, 1.0, 1.0, 1.0]), weight_tt);
    let bias = g.constant(&f32_data(&[0.0, 0.0, 0.0, 0.0]), bias_tt);
    let out = g.layer_norm(&input, &weight, &bias, 1e-5);

    let input_data = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    compare_gpu_vs_cpu(&g, &[&out], &[&input_data]);
}

#[test]
#[ignore]
fn test_wgpu_rmsnorm() {
    let g = GraphBuilder::new();
    let input = g.input(&[2, 4], IrDType::F32);
    let weight_tt = TensorType::new(vec![DimExpr::Known(4)], IrDType::F32);
    let weight = g.constant(&f32_data(&[0.5, 0.5, 0.5, 0.5]), weight_tt);
    let out = g.rms_norm(&input, &weight, 1e-5);

    let input_data = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    compare_gpu_vs_cpu(&g, &[&out], &[&input_data]);
}

// ── Reductions ────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_wgpu_reduce_sum() {
    let g = GraphBuilder::new();
    let input = g.input(&[2, 4], IrDType::F32);
    let sum = g.reduce_sum(&input, 1, false);

    let input_data = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let pairs = compare_gpu_vs_cpu(&g, &[&sum], &[&input_data]);
    assert_eq!(pairs[0].0, vec![10.0, 26.0]);
}

#[test]
#[ignore]
fn test_wgpu_reduce_mean() {
    let g = GraphBuilder::new();
    let input = g.input(&[2, 4], IrDType::F32);
    let mean = g.reduce_mean(&input, 1, false);

    let input_data = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let pairs = compare_gpu_vs_cpu(&g, &[&mean], &[&input_data]);
    assert_eq!(pairs[0].0, vec![2.5, 6.5]);
}

// ── Transpose ─────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_wgpu_transpose_2d() {
    let g = GraphBuilder::new();
    let input = g.input(&[2, 3], IrDType::F32);
    let t = g.transpose(&input);

    let input_data = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let pairs = compare_gpu_vs_cpu(&g, &[&t], &[&input_data]);
    assert_eq!(pairs[0].1, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
#[ignore]
fn test_wgpu_transpose_4d() {
    let g = GraphBuilder::new();
    let input = g.input(&[2, 4, 3, 5], IrDType::F32);
    // transpose_with_perm [0,1,3,2] swaps last two dims (attention pattern)
    let t = g.transpose_with_perm(&input, &[0, 1, 3, 2]);

    let input_data: Vec<f32> = (0..2 * 4 * 3 * 5).map(|i| i as f32).collect();

    compare_gpu_vs_cpu(&g, &[&t], &[&f32_data(&input_data)]);
}

// ── Embedding ─────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_wgpu_embedding() {
    let g = GraphBuilder::new();
    let weight_tt = TensorType::new(vec![DimExpr::Known(10), DimExpr::Known(4)], IrDType::F32);
    let weight_data: Vec<f32> = (0..40).map(|i| (i as f32) * 0.1).collect();
    let weight = g.constant(&f32_data(&weight_data), weight_tt);
    let indices = g.input(&[3], IrDType::I64);
    let emb = g.embedding(&weight, &indices);

    let idx_data = i64_data(&[0, 5, 9]);

    let result = g
        .compile_and_execute(&[&emb], WgpuBackend, &[&idx_data])
        .unwrap();
    let out_f32 = read_f32(&result[0]);
    assert_eq!(out_f32.len(), 12, "embed 3 indices * 4 dim = 12 floats");
    for &v in &out_f32 {
        assert!(v.is_finite(), "embed output should be finite, got {}", v);
    }
}

// ── ArgMax ────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_wgpu_argmax() {
    let g = GraphBuilder::new();
    let input = g.input(&[2, 4], IrDType::F32);
    let amax = g.argmax(&input, Some(DimExpr::Known(1)));

    let input_data = f32_data(&[0.1, 0.5, 0.3, 0.2, 0.9, 0.1, 0.4, 0.8]);

    let result = g
        .compile_and_execute(&[&amax], WgpuBackend, &[&input_data])
        .unwrap();
    let out: &[i64] = bytemuck::cast_slice(&result[0]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], 1); // argmax of [0.1, 0.5, 0.3, 0.2] is index 1
    assert_eq!(out[1], 0); // argmax of [0.9, 0.1, 0.4, 0.8] is index 0
}

// ── Error Handling ────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_wgpu_unsupported_op_fallback() {
    // Pad is a CPU-fallback op.  Verify it still works (runs on CPU).
    let g = GraphBuilder::new();
    let input = g.input(&[2, 3], IrDType::F32);
    let padded = g.pad(&input, &[(1, 1), (0, 0)]);

    let input_data = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let pairs = compare_gpu_vs_cpu(&g, &[&padded], &[&input_data]);
    assert_eq!(pairs[0].0.len(), 8);
}

// ── Large Graph ───────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_wgpu_large_graph() {
    // Multi-op graph: matmul -> add -> relu -> softmax -> reduce_mean
    let g = GraphBuilder::new();
    let a = g.input(&[4, 8], IrDType::F32);
    let w = g.input(&[8, 4], IrDType::F32);
    let b = g.input(&[4, 4], IrDType::F32);

    let mm = g.matmul(&a, &w);
    let added = g.add(&mm, &b);
    let relu = g.relu(&added);
    let sm = g.softmax(&relu, 1);
    let reduced = g.reduce_mean(&sm, 1, false);

    let a_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5).collect();
    let w_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.2).collect();

    let pairs = compare_gpu_vs_cpu(
        &g,
        &[&reduced],
        &[&f32_data(&a_data), &f32_data(&w_data), &f32_data(&b_data)],
    );
    assert_eq!(pairs[0].0.len(), 4);
}
