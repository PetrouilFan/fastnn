use fastnn::backends::cpu;
use fastnn::backends::wgpu::gemv_wgpu;
use fastnn::dtypes::{F32x1, PackedWord, U8x4};
use fastnn::packed_layer::PackedLinear;
use fastnn::packed_tensor::PackedTensor;
use std::time::Instant;

fn bench_cpu_gemv<T: PackedWord>(m: usize, k: usize, iters: usize) -> f64 {
    let weights = PackedTensor::<T>::from_f32_auto(
        &(0..m * k)
            .map(|i| (i as f32 * 0.01).sin() * 2.0)
            .collect::<Vec<f32>>(),
        &[m, k],
    );
    let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).cos()).collect();
    let mut output = vec![0.0f32; m];
    for _ in 0..5 {
        cpu::gemv_cpu(&weights, &activation, &mut output);
    }
    let start = Instant::now();
    for _ in 0..iters {
        cpu::gemv_cpu(&weights, &activation, &mut output);
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn bench_wgpu_original<T: PackedWord>(m: usize, k: usize, iters: usize) -> f64 {
    let weights = PackedTensor::<T>::from_f32_auto(
        &(0..m * k)
            .map(|i| (i as f32 * 0.01).sin() * 2.0)
            .collect::<Vec<f32>>(),
        &[m, k],
    );
    let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).cos()).collect();
    for _ in 0..3 {
        let _ = gemv_wgpu::<T>(&weights, &activation);
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = gemv_wgpu::<T>(&weights, &activation);
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn bench_wgpu_persistent<T: PackedWord>(m: usize, k: usize, iters: usize) -> f64 {
    let layer = PackedLinear::<T>::from_packed(
        PackedTensor::<T>::from_f32_auto(
            &(0..m * k)
                .map(|i| (i as f32 * 0.01).sin() * 2.0)
                .collect::<Vec<f32>>(),
            &[m, k],
        ),
        None,
        k,
        m,
    );
    let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).cos()).collect();
    for _ in 0..3 {
        let _ = layer.forward_wgpu(&activation);
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = layer.forward_wgpu(&activation);
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    println!("=== fastnn GPU vs CPU GEMV Benchmark ===\n");
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    for (i, a) in instance
        .enumerate_adapters(wgpu::Backends::all())
        .iter()
        .enumerate()
    {
        let info = a.get_info();
        println!("GPU {}: {} ({:?})", i, info.name, info.device_type);
    }
    println!();

    for &(m, k) in &[(256, 256), (512, 512), (1024, 1024), (4096, 4096)] {
        let iters = if m <= 1024 { 100 } else { 20 };
        println!("GEMV {}x{} x {}", m, k, k);
        println!(
            "{:<8} {:>10} {:>10} {:>12}",
            "dtype", "cpu ms", "orig ms", "persist ms"
        );

        let cpu = bench_cpu_gemv::<F32x1>(m, k, iters);
        let orig = bench_wgpu_original::<F32x1>(m, k, iters);
        let persist = bench_wgpu_persistent::<F32x1>(m, k, iters);
        println!(
            "{:<8} {:>10.3} {:>10.3} {:>12.3}",
            "F32x1", cpu, orig, persist
        );

        let cpu = bench_cpu_gemv::<U8x4>(m, k, iters);
        let orig = bench_wgpu_original::<U8x4>(m, k, iters);
        let persist = bench_wgpu_persistent::<U8x4>(m, k, iters);
        println!(
            "{:<8} {:>10.3} {:>10.3} {:>12.3}",
            "U8x4", cpu, orig, persist
        );
        println!();
    }
}
