use fastnn::backends::cpu;
use fastnn::dtypes::{F16x2, F32x1, U4x8, U8x4};
use fastnn::packed_tensor::PackedTensor;
use std::time::Instant;

fn bench_gemv<T: fastnn::dtypes::PackedWord>(
    m: usize,
    k: usize,
    iters: usize,
) -> (f64, f64) {
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
    (ms, gflops)
}

fn main() {
    println!("=== Packed Precision GEMV Performance ===\n");

    let sizes = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ];

    for &(m, k) in &sizes {
        let iters = if m * k <= 1024 * 1024 { 100 } else { 20 };

        println!("GEMV {}x{} (K={})", m, k, k);
        println!(
            "{:<10} {:>10} {:>12} {:>12}",
            "dtype", "ms", "GFLOP/s", "vs F32"
        );

        let (f32_ms, f32_gflops) = bench_gemv::<F32x1>(m, k, iters);
        println!(
            "{:<10} {:>10.3} {:>12.2} {:>12.1}x",
            "F32x1", f32_ms, f32_gflops, 1.0
        );

        let (f16_ms, f16_gflops) = bench_gemv::<F16x2>(m, k, iters);
        println!(
            "{:<10} {:>10.3} {:>12.2} {:>12.1}x",
            "F16x2",
            f16_ms,
            f16_gflops,
            f32_ms / f16_ms
        );

        let (u8_ms, u8_gflops) = bench_gemv::<U8x4>(m, k, iters);
        println!(
            "{:<10} {:>10.3} {:>12.2} {:>12.1}x",
            "U8x4",
            u8_ms,
            u8_gflops,
            f32_ms / u8_ms
        );

        let (u4_ms, u4_gflops) = bench_gemv::<U4x8>(m, k, iters);
        println!(
            "{:<10} {:>10.3} {:>12.2} {:>12.1}x",
            "U4x8",
            u4_ms,
            u4_gflops,
            f32_ms / u4_ms
        );

        println!();
    }

    println!("=== Large K (Tiled BLAS path) ===\n");

    let large_k_sizes = [(512, 8192), (1024, 8192), (2048, 8192)];

    for &(m, k) in &large_k_sizes {
        let iters = 10;

        println!("GEMV {}x{} (K={})", m, k, k);
        println!(
            "{:<10} {:>10} {:>12} {:>12}",
            "dtype", "ms", "GFLOP/s", "vs F32"
        );

        let (f32_ms, f32_gflops) = bench_gemv::<F32x1>(m, k, iters);
        println!(
            "{:<10} {:>10.3} {:>12.2} {:>12.1}x",
            "F32x1", f32_ms, f32_gflops, 1.0
        );

        let (f16_ms, f16_gflops) = bench_gemv::<F16x2>(m, k, iters);
        println!(
            "{:<10} {:>10.3} {:>12.2} {:>12.1}x",
            "F16x2",
            f16_ms,
            f16_gflops,
            f32_ms / f16_ms
        );

        let (u8_ms, u8_gflops) = bench_gemv::<U8x4>(m, k, iters);
        println!(
            "{:<10} {:>10.3} {:>12.2} {:>12.1}x",
            "U8x4",
            u8_ms,
            u8_gflops,
            f32_ms / u8_ms
        );

        let (u4_ms, u4_gflops) = bench_gemv::<U4x8>(m, k, iters);
        println!(
            "{:<10} {:>10.3} {:>12.2} {:>12.1}x",
            "U4x8",
            u4_ms,
            u4_gflops,
            f32_ms / u4_ms
        );

        println!();
    }
}
