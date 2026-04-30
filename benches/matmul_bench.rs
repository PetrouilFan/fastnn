use fastnn::kernels::blas;
use std::time::Instant;

fn main() {
    println!("=== fastnn BLAS MatMul Benchmark ===\n");

    let m = 1024;
    let k = 1024;
    let n = 1024;

    // Create two 1024x1024 matrices with deterministic data
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.001).cos()).collect();
    let mut c = vec![0.0f32; m * n];

    // Warmup
    for _ in 0..5 {
        blas::matmul_blas_into(&a, &b, &mut c, m, k, n);
    }

    // Reset output to ensure consistent timing
    c.fill(0.0);

    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        blas::matmul_blas_into(&a, &b, &mut c, m, k, n);
    }
    let elapsed = start.elapsed();

    let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    let gflops = (2.0 * m as f64 * k as f64 * n as f64) / (ms / 1000.0) / 1e9;

    println!("Matrix size: {} x {} x {}", m, k, n);
    println!("Iterations: {}", iters);
    println!("Total time: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    println!("Average time: {:.3} ms / iter", ms);
    println!("Performance: {:.2} GFLOP/s", gflops);
    println!("Total FLOPs: {}", 2 * m * k * n);
}
