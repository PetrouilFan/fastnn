//! Benchmark: WGPU quantized inference vs CPU quantized inference.
//!
//! Measures end-to-end latency for quantized matmul on GPU vs CPU.
//! Both backends use the same AOT-compiled graph with 4-bit weight quantization.

use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::GraphExecutor;
use fastnn::backend::wgpu::WgpuBackend;
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{DimExpr, IrDType, TensorType};
use std::hint::black_box;
use std::time::Instant;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

/// Run the WGPU quantized inference benchmark, returning the mean time in ms,
/// or None if WGPU is unavailable or fails.
fn bench_wgpu_quantized(graph: &fastnn::ir::node::ComputeGraph, input_bytes: &[u8], iters: usize) -> Option<f64> {
    let gpu_executor = GraphExecutor::new(WgpuBackend);
    let (gpu_plan, gpu_mem, gpu_graph) = gpu_executor
        .compile_with_plan_and_quantize(graph, Some(4))
        .ok()?;

    for _ in 0..10 {
        gpu_executor
            .execute(&gpu_graph, &gpu_plan, &gpu_mem, &[input_bytes])
            .ok()?;
    }

    let start = Instant::now();
    for _ in 0..iters {
        gpu_executor
            .execute(&gpu_graph, &gpu_plan, &gpu_mem, &[input_bytes])
            .ok()?;
    }
    let gpu_ns = start.elapsed().as_nanos() as f64 / iters as f64;
    Some(gpu_ns / 1_000_000.0)
}

fn main() {
    println!("=== WGPU Quantized Inference vs CPU Quantized Inference ===\n");

    // Build a quantized matmul graph: input[1, 256] @ weight[256, 512] = output[1, 512]
    let g = GraphBuilder::new();
    let x = g.input(&[1, 256], IrDType::F32);

    // Weight as a constant so the quantization pass can convert it to U4
    let w_data: Vec<f32> = (0..256 * 512)
        .map(|i| (i as f32 * 0.001).cos() * 0.1)
        .collect();
    let w_bytes = f32_bytes(&w_data);
    let w_tt = TensorType::new(
        vec![DimExpr::Known(256), DimExpr::Known(512)],
        IrDType::F32,
    );
    let w = g.constant(&w_bytes, w_tt);

    let mm = g.matmul(&x, &w);

    let mut graph = g.to_graph();
    graph.inputs = vec![x.node_id()];
    graph.outputs = vec![mm.node_id()];

    let input_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
    let input_bytes = f32_bytes(&input_data);

    let iters = 500;

    // ═══════════════════════════════════════════════════════════════════════
    // CPU: Quantized inference
    // ═══════════════════════════════════════════════════════════════════════
    println!("--- CPU Quantized MatMul (256x512, U4) ---");

    let cpu_executor = GraphExecutor::new(CpuBackend);
    let (cpu_plan, cpu_mem, cpu_graph) = cpu_executor
        .compile_with_plan_and_quantize(&graph, Some(4))
        .expect("CPU quantized compile should succeed");

    for _ in 0..10 {
        cpu_executor
            .execute(&cpu_graph, &cpu_plan, &cpu_mem, &[&input_bytes])
            .unwrap();
    }

    let start = Instant::now();
    for _ in 0..iters {
        black_box(
            cpu_executor
                .execute(&cpu_graph, &cpu_plan, &cpu_mem, &[&input_bytes])
                .unwrap(),
        );
    }
    let cpu_ns = start.elapsed().as_nanos() as f64 / iters as f64;
    let cpu_ms = cpu_ns / 1_000_000.0;

    println!(
        "  Mean inference time: {:>10.4} ms  ({:>8.2} inf/sec)",
        cpu_ms,
        1000.0 / cpu_ms
    );

    // ═══════════════════════════════════════════════════════════════════════
    // WGPU: Quantized inference (may fail if GPU doesn't support enough
    // storage buffers or if no GPU is available)
    // ═══════════════════════════════════════════════════════════════════════
    println!();
    println!("--- WGPU Quantized MatMul (256x512, U4) ---");

    let wgpu_ms = std::panic::catch_unwind(|| bench_wgpu_quantized(&graph, &input_bytes, iters))
        .ok()
        .flatten();

    match wgpu_ms {
        Some(ms) => {
            println!(
                "  Mean inference time: {:>10.4} ms  ({:>8.2} inf/sec)",
                ms,
                1000.0 / ms
            );
            println!();
            println!("=== Comparison ===");
            println!(
                "  {:<30} {:>12} {:>10}",
                "Backend", "Mean Time", "Speedup"
            );
            println!(
                "  {:<30} {:>10.4} ms  {:>6.2}x",
                "CPU (U4 quantized)", cpu_ms, 1.0
            );
            println!(
                "  {:<30} {:>10.4} ms  {:>6.2}x",
                "WGPU (GPU)", ms, cpu_ms / ms
            );
        }
        None => {
            println!("  WGPU not available or benchmark failed (GPU device limits)");
            println!();
            println!("=== Comparison ===");
            println!(
                "  {:<30} {:>12} {:>10}",
                "Backend", "Mean Time", "Speedup"
            );
            println!(
                "  {:<30} {:>10.4} ms  {:>6.2}x",
                "CPU (U4 quantized)", cpu_ms, 1.0
            );
            println!("  WGPU (GPU)               N/A (device limits / not available)");
        }
    }
    println!();
}
