//! Shared benchmark utilities for the `packed_bench` example.
//!
//! The example pulls this file in with `#[path = "bench_util.rs"] mod bench_util;`.

use fastnn::backend::cpu::microkernels::gemv_cpu;
use fastnn::dtypes::PackedWord;
use fastnn::packed_tensor::PackedTensor;
use std::time::Instant;

/// Benchmark GEMV (General Matrix-Vector multiplication) operation.
pub fn bench_gemv<T: PackedWord>(m: usize, k: usize, iters: usize) -> (f64, usize) {
    let weight_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin() * 2.0).collect();
    let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).cos()).collect();
    let mut output = vec![0.0f32; m];

    let weights = PackedTensor::<T>::from_f32_auto(&weight_data, &[m, k]);

    for _ in 0..5 {
        gemv_cpu(&weights, &activation, &mut output);
    }

    let start = Instant::now();
    for _ in 0..iters {
        gemv_cpu(&weights, &activation, &mut output);
    }
    let elapsed = start.elapsed();

    let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    let packed_bytes = weights.packed_len() * 4;
    (ms, packed_bytes)
}

pub fn bench_relu<T: PackedWord>(_data: &[f32], _shape: &[usize], _iters: usize) -> f64 {
    let _ = std::marker::PhantomData::<T>;
    0.0
}

/// Calculate speedup ratio with safe division.
pub fn speedup(base: f64, compare: f64) -> f64 {
    if compare > 0.0 {
        base / compare
    } else {
        0.0
    }
}
