use fastnn::kernels::blas;
use fastnn::Tensor;
use std::time::Instant;

/// Generate deterministic random-like data for benchmarking
fn generate_data(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| ((i as f32 * 0.001).sin() * 1000.0).sin())
        .collect()
}

fn main() {
    println!("=== fastnn Tensor::matmul Direct Benchmark ===\n");

    let size = 1024;
    let m = size;
    let k = size;
    let n = size;

    // Create two 1024x1024 contiguous f32 tensors on CPU
    let a_data = generate_data(m * k);
    let b_data = generate_data(k * n);

    let a = Tensor::from_vec(a_data.clone(), vec![m as i64, k as i64]);
    let b = Tensor::from_vec(b_data.clone(), vec![k as i64, n as i64]);

    // Verify tensors are contiguous
    assert!(a.is_contiguous(), "Tensor A must be contiguous");
    assert!(b.is_contiguous(), "Tensor B must be contiguous");

    // Warmup: run Tensor::matmul a few times
    for _ in 0..5 {
        let _ = a.matmul(&b);
    }

    // Benchmark Tensor::matmul
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        let _c = a.matmul(&b);
    }
    let elapsed_tensor = start.elapsed();

    let ms_tensor = elapsed_tensor.as_secs_f64() * 1000.0;
    let avg_ms_tensor = ms_tensor / iters as f64;
    let gflops_tensor = (2.0 * m as f64 * k as f64 * n as f64) / (ms_tensor / 1000.0) / 1e9;

    println!("--- fastnn Tensor::matmul ---");
    println!("Matrix size: {} x {} x {}", m, k, n);
    println!("Iterations: {}", iters);
    println!("Total time: {:.3} ms", ms_tensor);
    println!("Average time: {:.3} ms / iter", avg_ms_tensor);
    println!("Performance: {:.2} GFLOP/s", gflops_tensor);
    println!("Total FLOPs: {}\n", 2 * m * k * n);

    // Benchmark direct cblas_sgemm call
    let a_slice = &a_data;
    let b_slice = &b_data;
    let mut c_direct = vec![0.0f32; m * n];

    // Warmup
    for _ in 0..5 {
        blas::matmul_blas_into(a_slice, b_slice, &mut c_direct, m, k, n);
    }

    // Reset output
    c_direct.fill(0.0);

    let start = Instant::now();
    for _ in 0..iters {
        blas::matmul_blas_into(a_slice, b_slice, &mut c_direct, m, k, n);
    }
    let elapsed_direct = start.elapsed();

    let ms_direct = elapsed_direct.as_secs_f64() * 1000.0;
    let avg_ms_direct = ms_direct / iters as f64;
    let gflops_direct = (2.0 * m as f64 * k as f64 * n as f64) / (ms_direct / 1000.0) / 1e9;

    println!("--- Direct cblas_sgemm ---");
    println!("Matrix size: {} x {} x {}", m, k, n);
    println!("Iterations: {}", iters);
    println!("Total time: {:.3} ms", ms_direct);
    println!("Average time: {:.3} ms / iter", avg_ms_direct);
    println!("Performance: {:.2} GFLOP/s", gflops_direct);
    println!("Total FLOPs: {}\n", 2 * m * k * n);

    // Comparison
    let ratio = avg_ms_tensor / avg_ms_direct;
    let overhead_pct = (ratio - 1.0) * 100.0;

    println!("--- Comparison ---");
    println!("Tensor::matmul avg:  {:.3} ms/iter", avg_ms_tensor);
    println!("cblas_sgemm avg:     {:.3} ms/iter", avg_ms_direct);
    println!("Overhead ratio:      {:.3}x", ratio);
    println!("Overhead:            {:.2}%", overhead_pct);
}
