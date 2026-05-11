/// Benchmarks comparing packed vs f32 dispatch for MatMul, Conv, and Embedding.
use fastnn::backends::cpu;
use fastnn::dtypes::{U4x8, U8x4};
use fastnn::packed_tensor::PackedTensor;
use std::time::Instant;

fn bench_matmul_f32(m: usize, k: usize, n: usize, iters: usize) -> f64 {
    let w_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let act: Vec<f32> = (0..m * k).map(|i| (i as f32) / 256.0).collect();

    let start = Instant::now();
    for _ in 0..iters {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            let inp = &act[i * k..(i + 1) * k];
            let outp = &mut out[i * n..(i + 1) * n];
            for j in 0..n {
                let mut sum = 0.0;
                for t in 0..k {
                    sum += inp[t] * w_f32[j * k + t];
                }
                outp[j] = sum;
            }
        }
        std::hint::black_box(out);
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1000.0 / iters as f64
}

fn bench_gemv_packed<T: fastnn::dtypes::PackedWord>(
    m: usize,
    k: usize,
    n: usize,
    iters: usize,
) -> f64 {
    let w_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let weight = PackedTensor::<T>::from_f32_auto(&w_f32, &[n, k]);
    let act: Vec<f32> = (0..m * k).map(|i| (i as f32) / 256.0).collect();

    let start = Instant::now();
    for _ in 0..iters {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            let inp = &act[i * k..(i + 1) * k];
            let outp = &mut out[i * n..(i + 1) * n];
            cpu::gemv_cpu(&weight, inp, outp);
        }
        std::hint::black_box(out);
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1000.0 / iters as f64
}

fn bench_batch_gemm_packed<T: fastnn::dtypes::PackedWord>(
    m: usize,
    k: usize,
    n: usize,
    iters: usize,
) -> f64 {
    let w_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let weight = PackedTensor::<T>::from_f32_auto(&w_f32, &[n, k]);
    let act: Vec<f32> = (0..m * k).map(|i| (i as f32) / 256.0).collect();
    let mut out = vec![0.0f32; m * n];

    let start = Instant::now();
    for _ in 0..iters {
        out.fill(0.0);
        cpu::gemm_batch_packed(&weight, &act, &mut out, m, k, n);
        std::hint::black_box(&out);
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    println!("=== Packed vs f32 Dispatch Benchmark ===\n");

    let sizes = [(64, 256, 512), (128, 512, 1024), (32, 1024, 2048)];
    let warmup_iters = 10;
    let bench_iters = 100;

    for &(m, k, n) in &sizes {
        println!("Matrix: batch={}, K={}, N={}", m, k, n);

        // Warmup
        for _ in 0..warmup_iters {
            let _ = bench_matmul_f32(m, k, n, 1);
            let _ = bench_gemv_packed::<U4x8>(m, k, n, 1);
            let _ = bench_gemv_packed::<U8x4>(m, k, n, 1);
            let _ = bench_batch_gemm_packed::<U4x8>(m, k, n, 1);
        }

        let f32_ms = bench_matmul_f32(m, k, n, bench_iters);
        let u4_gemv_ms = bench_gemv_packed::<U4x8>(m, k, n, bench_iters);
        let u8_gemv_ms = bench_gemv_packed::<U8x4>(m, k, n, bench_iters);
        let u4_batch_ms = bench_batch_gemm_packed::<U4x8>(m, k, n, bench_iters);
        let u8_batch_ms = bench_batch_gemm_packed::<U8x4>(m, k, n, bench_iters);

        let gflops = |ms: f64| (2.0 * m as f64 * k as f64 * n as f64) / (ms / 1000.0) / 1e9;
        let speedup = |base: f64, opt: f64| base / opt;

        println!(
            "  {:<20} {:>10} {:>12} {:>10}",
            "variant", "ms", "GFLOP/s", "vs f32"
        );
        println!(
            "  {:<20} {:>10.3} {:>12.2} {:>10}",
            "f32 (naive)",
            f32_ms,
            gflops(f32_ms),
            "1.0x"
        );
        println!(
            "  {:<20} {:>10.3} {:>12.2} {:>10.1}x",
            "U4 GEMV",
            u4_gemv_ms,
            gflops(u4_gemv_ms),
            speedup(f32_ms, u4_gemv_ms)
        );
        println!(
            "  {:<20} {:>10.3} {:>12.2} {:>10.1}x",
            "U8 GEMV",
            u8_gemv_ms,
            gflops(u8_gemv_ms),
            speedup(f32_ms, u8_gemv_ms)
        );
        println!(
            "  {:<20} {:>10.3} {:>12.2} {:>10.1}x",
            "U4 batch GEMM",
            u4_batch_ms,
            gflops(u4_batch_ms),
            speedup(f32_ms, u4_batch_ms)
        );
        println!(
            "  {:<20} {:>10.3} {:>12.2} {:>10.1}x",
            "U8 batch GEMM",
            u8_batch_ms,
            gflops(u8_batch_ms),
            speedup(f32_ms, u8_batch_ms)
        );
        println!();
    }

    println!("Done.");
}
