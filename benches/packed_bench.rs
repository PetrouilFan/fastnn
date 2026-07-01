// Use bench_util module for shared benchmark functions
#[path = "bench_util.rs"]
mod bench_util;

use bench_util::{bench_gemv, bench_relu, print_comparison_header, print_result_row, speedup};
use fastnn::dtypes::{F16x2, F32x1, F4x8, F8x4, F8x4R, I4x8, I8x4, PackedWord};

fn main() {
    println!("=== fastnn Native Packed Precision Benchmark ===\n");

    let sizes = [(256, 256), (512, 512), (1024, 1024), (4096, 4096)];

    for &(m, k) in &sizes {
        println!("GEMV {}x{} x {}", m, k, k);
        println!(
            "{:<10} {:>10} {:>10} {:>12} {:>10} {:>10}",
            "dtype", "ms", "GFLOP/s", "mem (bytes)", "vs F32", "mem save"
        );

        let iters = if m <= 1024 { 200 } else { 20 };

        let (f32_ms, f32_bytes) = bench_gemv::<F32x1>(m, k, iters);
        let gflops = (2.0 * m as f64 * k as f64) / (f32_ms / 1000.0) / 1e9;
        println!(
            "{:<10} {:>10.3} {:>10.2} {:>12} {:>10} {:>10}",
            "F32x1", f32_ms, gflops, f32_bytes, "1.0x", "1.0x"
        );

        let (f16_ms, f16_bytes) = bench_gemv::<F16x2>(m, k, iters);
        let gflops = (2.0 * m as f64 * k as f64) / (f16_ms / 1000.0) / 1e9;
        println!(
            "{:<10} {:>10.3} {:>10.2} {:>12} {:>10.1}x {:>9.1}x",
            "F16x2",
            f16_ms,
            gflops,
            f16_bytes,
            speedup(f32_ms, f16_ms),
            speedup(f32_bytes as f64, f16_bytes as f64)
        );

        let (u8_ms, u8_bytes) = bench_gemv::<I8x4>(m, k, iters);
        let gflops = (2.0 * m as f64 * k as f64) / (u8_ms / 1000.0) / 1e9;
        println!(
            "{:<10} {:>10.3} {:>10.2} {:>12} {:>10.1}x {:>9.1}x",
            "I8x4",
            u8_ms,
            gflops,
            u8_bytes,
            speedup(f32_ms, u8_ms),
            speedup(f32_bytes as f64, u8_bytes as f64)
        );

        let (u4_ms, u4_bytes) = bench_gemv::<I4x8>(m, k, iters);
        let gflops = (2.0 * m as f64 * k as f64) / (u4_ms / 1000.0) / 1e9;
        println!(
            "{:<10} {:>10.3} {:>10.2} {:>12} {:>10.1}x {:>9.1}x",
            "I4x8",
            u4_ms,
            gflops,
            u4_bytes,
            speedup(f32_ms, u4_ms),
            speedup(f32_bytes as f64, u4_bytes as f64)
        );

        let (f4_ms, f4_bytes) = bench_gemv::<F4x8>(m, k, iters);
        let gflops = (2.0 * m as f64 * k as f64) / (f4_ms / 1000.0) / 1e9;
        println!(
            "{:<10} {:>10.3} {:>10.2} {:>12} {:>10.1}x {:>9.1}x",
            "F4x8",
            f4_ms,
            gflops,
            f4_bytes,
            speedup(f32_ms, f4_ms),
            speedup(f32_bytes as f64, f4_bytes as f64)
        );

        let (f8_ms, f8_bytes) = bench_gemv::<F8x4>(m, k, iters);
        let gflops = (2.0 * m as f64 * k as f64) / (f8_ms / 1000.0) / 1e9;
        println!(
            "{:<10} {:>10.3} {:>10.2} {:>12} {:>10.1}x {:>9.1}x",
            "F8x4",
            f8_ms,
            gflops,
            f8_bytes,
            speedup(f32_ms, f8_ms),
            speedup(f32_bytes as f64, f8_bytes as f64)
        );

        let (f8r_ms, f8r_bytes) = bench_gemv::<F8x4R>(m, k, iters);
        let gflops = (2.0 * m as f64 * k as f64) / (f8r_ms / 1000.0) / 1e9;
        println!(
            "{:<10} {:>10.3} {:>10.2} {:>12} {:>10.1}x {:>9.1}x",
            "F8x4R",
            f8r_ms,
            gflops,
            f8r_bytes,
            speedup(f32_ms, f8r_ms),
            speedup(f32_bytes as f64, f8r_bytes as f64)
        );

        println!();
    }

    println!("ReLU (SWAR for int types, unpack-repack for float types)");
    println!(
        "{:<10} {:>10} {:>10} {:>12} {:>10}",
        "dtype", "ms", "elems/ms", "mem (bytes)", "vs F32"
    );

    for numel in [4096, 16384, 65536, 262144] {
        let iters = 500;
        let data: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.1).sin() * 10.0).collect();
        let shape = [numel];

        println!("\n  numel={}", numel);

        let f32_ms = bench_relu::<F32x1>(&data, &shape, iters);
        let f32_bytes = numel * 4;
        println!(
            "  {:<10} {:>10.4} {:>10.0} {:>12} {:>10}",
            "F32x1",
            f32_ms,
            numel as f64 / f32_ms,
            f32_bytes,
            "1.0x"
        );

        let f16_ms = bench_relu::<F16x2>(&data, &shape, iters);
        let f16_bytes = numel.div_ceil(2) * 4;
        println!(
            "  {:<10} {:>10.4} {:>10.0} {:>12} {:>9.1}x",
            "F16x2",
            f16_ms,
            numel as f64 / f16_ms,
            f16_bytes,
            speedup(f32_ms, f16_ms)
        );

        let u8_ms = bench_relu::<I8x4>(&data, &shape, iters);
        let u8_bytes = numel.div_ceil(4) * 4;
        println!(
            "  {:<10} {:>10.4} {:>10.0} {:>12} {:>9.1}x",
            "I8x4",
            u8_ms,
            numel as f64 / u8_ms,
            u8_bytes,
            speedup(f32_ms, u8_ms)
        );

        let u4_ms = bench_relu::<I4x8>(&data, &shape, iters);
        let u4_bytes = numel.div_ceil(8) * 4;
        println!(
            "  {:<10} {:>10.4} {:>10.0} {:>12} {:>9.1}x",
            "I4x8",
            u4_ms,
            numel as f64 / u4_ms,
            u4_bytes,
            speedup(f32_ms, u4_ms)
        );
    }

    println!("\n=== Summary ===");
    println!("GEMV is memory-bandwidth-bound: packed types reduce cache misses");
    println!("ReLU SWAR for int types avoids unpacking entirely (bitwise-only)");
    println!("On CPU, gains are modest due to scalar unpack in GEMV inner loop");
    println!("On GPU (wgpu), gains are larger due to reduced memory traffic");
}
