use fastnn::backends::cpu;
use fastnn::dtypes::PackedWord;
use fastnn::packed_tensor::PackedTensor;
use std::time::Instant;

fn bench_gemv<T: PackedWord>(m: usize, k: usize, iters: usize) -> (f64, usize) {
    let weight_data: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32 * 0.01).sin()) * 2.0)
        .collect();
    let activation: Vec<f32> = (0..k)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();
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

fn main() {
    println!("=== Cache/Memory Hierarchy Analysis for Packed GEMM ===\n");

    let sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 8192)];
    for &(m, k) in &sizes {
        println!("GEMV {}x{}:", m, k);
        let (f32_ms, f32_bytes) = bench_gemv::<fastnn::dtypes::F32x1>(m, k, 100);
        let (f16_ms, f16_bytes) = bench_gemv::<fastnn::dtypes::F16x2>(m, k, 100);
        let (u8_ms, u8_bytes) = bench_gemv::<fastnn::dtypes::U8x4>(m, k, 100);
        let (u4_ms, u4_bytes) = bench_gemv::<fastnn::dtypes::U4x8>(m, k, 100);

        println!("  F32x1: {:.3} ms, {} bytes", f32_ms, f32_bytes);
        println!("  F16x2: {:.3} ms, {} bytes", f16_ms, f16_bytes);
        println!("  U8x4:  {:.3} ms, {} bytes", u8_ms, u8_bytes);
        println!("  U4x8:  {:.3} ms, {} bytes", u4_ms, u4_bytes);
        println!();
    }
}
