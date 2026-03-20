use fastnn::backends::cpu;
use fastnn::dtypes::{F16x2, F32x1, PackedWord, U4x8, U8x4};
use fastnn::packed_tensor::PackedTensor;
use std::time::Instant;

fn bench_gemv<T: PackedWord>(m: usize, k: usize, iters: usize) -> (f64, usize) {
    let weight_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin() * 2.0).collect();
    let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).cos()).collect();
    let mut output = vec![0.0f32; m];

    let weights = PackedTensor::<T>::from_f32_auto(&weight_data, &[m, k]);

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

fn bench_relu<T: PackedWord>(data: &[f32], shape: &[usize], iters: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let mut tensor = PackedTensor::<T>::from_f32_auto(data, shape);
        fastnn::backends::cpu::relu_cpu(&mut tensor);
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1000.0 / iters as f64
}

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
            f32_ms / f16_ms,
            f32_bytes as f64 / f16_bytes as f64
        );

        let (u8_ms, u8_bytes) = bench_gemv::<U8x4>(m, k, iters);
        let gflops = (2.0 * m as f64 * k as f64) / (u8_ms / 1000.0) / 1e9;
        println!(
            "{:<10} {:>10.3} {:>10.2} {:>12} {:>10.1}x {:>9.1}x",
            "U8x4",
            u8_ms,
            gflops,
            u8_bytes,
            f32_ms / u8_ms,
            f32_bytes as f64 / u8_bytes as f64
        );

        let (u4_ms, u4_bytes) = bench_gemv::<U4x8>(m, k, iters);
        let gflops = (2.0 * m as f64 * k as f64) / (u4_ms / 1000.0) / 1e9;
        println!(
            "{:<10} {:>10.3} {:>10.2} {:>12} {:>10.1}x {:>9.1}x",
            "U4x8",
            u4_ms,
            gflops,
            u4_bytes,
            f32_ms / u4_ms,
            f32_bytes as f64 / u4_bytes as f64
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
        let f32_bytes = ((numel + 0) / 1) * 4;
        println!(
            "  {:<10} {:>10.4} {:>10.0} {:>12} {:>10}",
            "F32x1",
            f32_ms,
            numel as f64 / f32_ms,
            f32_bytes,
            "1.0x"
        );

        let f16_ms = bench_relu::<F16x2>(&data, &shape, iters);
        let f16_bytes = ((numel + 1) / 2) * 4;
        println!(
            "  {:<10} {:>10.4} {:>10.0} {:>12} {:>9.1}x",
            "F16x2",
            f16_ms,
            numel as f64 / f16_ms,
            f16_bytes,
            f32_ms / f16_ms
        );

        let u8_ms = bench_relu::<U8x4>(&data, &shape, iters);
        let u8_bytes = ((numel + 3) / 4) * 4;
        println!(
            "  {:<10} {:>10.4} {:>10.0} {:>12} {:>9.1}x",
            "U8x4",
            u8_ms,
            numel as f64 / u8_ms,
            u8_bytes,
            f32_ms / u8_ms
        );

        let u4_ms = bench_relu::<U4x8>(&data, &shape, iters);
        let u4_bytes = ((numel + 7) / 8) * 4;
        println!(
            "  {:<10} {:>10.4} {:>10.0} {:>12} {:>9.1}x",
            "U4x8",
            u4_ms,
            numel as f64 / u4_ms,
            u4_bytes,
            f32_ms / u4_ms
        );
    }

    println!("\n=== Summary ===");
    println!("GEMV is memory-bandwidth-bound: packed types reduce cache misses");
    println!("ReLU SWAR for int types avoids unpacking entirely (bitwise-only)");
    println!("On CPU, gains are modest due to scalar unpack in GEMV inner loop");
    println!("On GPU (wgpu), gains are larger due to reduced memory traffic");
}
