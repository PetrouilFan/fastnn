//! Shared benchmark utilities for fastnn Rust benchmarks.
//!
//! This module provides common benchmark functions to eliminate code duplication
//! across benchmark files (packed_bench.rs, wgpu_bench.rs, quantized_vs_pytorch.rs).

use fastnn::backends::cpu;
use fastnn::dtypes::PackedWord;
use fastnn::packed_tensor::PackedTensor;
use std::time::Instant;

/// Benchmark GEMV (General Matrix-Vector multiplication) operation.
///
/// # Arguments
/// * `m` - Number of output rows
/// * `k` - Inner dimension (length of activation vector)
/// * `iters` - Number of iterations for timing
///
/// # Returns
/// Tuple of (elapsed_time_ms, packed_bytes)
pub fn bench_gemv<T: PackedWord>(m: usize, k: usize, iters: usize) -> (f64, usize) {
    let weight_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin() * 2.0).collect();
    let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).cos()).collect();
    let mut output = vec![0.0f32; m];

    let weights = PackedTensor::<T>::from_f32_auto(&weight_data, &[m, k]);

    // Warmup
    for _ in 0..5 {
        cpu::gemv_cpu(&weights, &activation, &mut output);
    }

    let start = Instant::now();
    for _ in 0..iters {
        cpu::gemv_cpu(&weights, &activation, &mut output);
    }
    let elapsed = start.elapsed();

    let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    let packed_bytes = weights.packed_len() * 4;
    (ms, packed_bytes)
}

/// Benchmark GEMV with additional metrics (GFLOPS, memory).
///
/// # Arguments
/// * `m` - Number of output rows
/// * `k` - Inner dimension
/// * `iters` - Number of iterations
/// * `label` - Label for output
///
/// # Returns
/// Tuple of (ms, gflops, memory_mb)
pub fn bench_gemv_with_metrics<T: PackedWord>(
    m: usize,
    k: usize,
    iters: usize,
    label: &str,
) -> (f64, f64, f64) {
    let weight_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin() * 2.0).collect();
    let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).cos()).collect();
    let mut output = vec![0.0f32; m];

    let weights = PackedTensor::<T>::from_f32_auto(&weight_data, &[m, k]);

    // Warmup
    for _ in 0..5 {
        cpu::gemv_cpu(&weights, &activation, &mut output);
    }

    let start = Instant::now();
    for _ in 0..iters {
        cpu::gemv_cpu(&weights, &activation, &mut output);
    }
    let elapsed = start.elapsed();

    let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    let gflops = (2.0 * m as f64 * k as f64) / (ms / 1000.0) / 1e9;
    let memory_mb = (weights.packed_len() * 4) as f64 / (1024.0 * 1024.0);

    println!(
        "  {:<10} {:>10.3} ms {:>10.2} GFLOP/s {:>10.2} MB {:>10.1}x speedup",
        label,
        ms,
        gflops,
        memory_mb,
        gflops / 8.3
    );

    (ms, gflops, memory_mb)
}

/// Benchmark ReLU operation on packed tensor.
///
/// # Arguments
/// * `data` - Input data slice
/// * `shape` - Shape of the tensor
/// * `iters` - Number of iterations
///
/// # Returns
/// Elapsed time in ms
pub fn bench_relu<T: PackedWord>(data: &[f32], shape: &[usize], iters: usize) -> f64 {
    // Create tensor OUTSIDE the timing loop
    let mut tensor = PackedTensor::<T>::from_f32_auto(data, shape);

    // Warmup
    for _ in 0..5 {
        fastnn::backends::cpu::relu_cpu(&mut tensor);
        tensor = PackedTensor::<T>::from_f32_auto(data, shape); // reset
    }

    let start = Instant::now();
    for _ in 0..iters {
        fastnn::backends::cpu::relu_cpu(&mut tensor);
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1000.0 / iters as f64
}

/// Calculate speedup ratio with safe division.
pub fn speedup(base: f64, compare: f64) -> f64 {
    if compare > 0.0 {
        base / compare
    } else {
        0.0
    }
}

/// Print benchmark comparison header.
pub fn print_comparison_header() {
    println!(
        "{:<10} {:>10} {:>10} {:>12} {:>10} {:>10}",
        "dtype", "ms", "GFLOP/s", "mem (bytes)", "vs F32", "mem save"
    );
}

/// Print benchmark result row.
pub fn print_result_row(
    label: &str,
    ms: f64,
    gflops: f64,
    bytes: usize,
    vs_f32: f64,
    mem_save: f64,
) {
    println!(
        "{:<10} {:>10.3} {:>10.2} {:>12} {:>10.1}x {:>9.1}x",
        label, ms, gflops, bytes, vs_f32, mem_save
    );
}
