#[path = "bench_util.rs"]
mod bench_util;

use bench_util::bench_gemv;
use fastnn::dtypes::{F16x2, F32x1, U4x8, U8x4};
use serde::Serialize;

#[derive(Serialize)]
struct BenchRow {
    precision: String,
    size: usize,
    ms: f64,
    memory_mb: f64,
}

#[derive(Serialize)]
struct Output {
    benchmark: String,
    results: Vec<BenchRow>,
}

fn main() {
    let sizes = [256usize, 512, 1024, 4096];
    let precisions: &[( &str, fn(usize, usize, usize) -> (f64, usize))] = &[
        ("F32x1", bench_gemv::<F32x1>),
        ("F16x2", bench_gemv::<F16x2>),
        ("U8x4", bench_gemv::<U8x4>),
        ("U4x8", bench_gemv::<U4x8>),
    ];

    let mut results = Vec::new();

    for &(name, bench_fn) in precisions {
        for &size in &sizes {
            let iters = if size <= 1024 { 200 } else { 20 };
            let (ms, packed_bytes) = bench_fn(size, size, iters);
            let memory_mb = packed_bytes as f64 / (1024.0 * 1024.0);
            results.push(BenchRow {
                precision: name.to_string(),
                size,
                ms: (ms * 1000.0).round() / 1000.0,  // 3 decimal places
                memory_mb: (memory_mb * 100.0).round() / 100.0,  // 2 decimal places
            });
        }
    }

    let output = Output {
        benchmark: "packed_gemv".to_string(),
        results,
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}
