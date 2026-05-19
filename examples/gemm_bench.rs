use std::time::Instant;

struct GemmShape {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

fn main() {
    let shapes = [
        GemmShape { name: "backbone_3x3_s2_16ch", m: 160 * 160, k: 3 * 3 * 3, n: 16 },
        GemmShape { name: "backbone_3x3_s2_32ch", m: 80 * 80, k: 16 * 3 * 3, n: 32 },
        GemmShape { name: "backbone_3x3_s2_64ch", m: 40 * 40, k: 32 * 3 * 3, n: 64 },
        GemmShape { name: "mid_3x3_64ch_80", m: 80 * 80, k: 64 * 3 * 3, n: 64 },
        GemmShape { name: "mid_3x3_128ch_40", m: 40 * 40, k: 128 * 3 * 3, n: 128 },
        GemmShape { name: "1x1_up_64to128", m: 80 * 80, k: 64, n: 128 },
        GemmShape { name: "1x1_down_128to64", m: 80 * 80, k: 128, n: 64 },
        GemmShape { name: "neck_3x3_128ch_80", m: 80 * 80, k: 128 * 3 * 3, n: 128 },
        GemmShape { name: "dw_80ch", m: 80 * 80, k: 9, n: 1 },
    ];

    let iters: u32 = match std::env::var("GEMM_ITERS") {
        Ok(v) => v.parse().unwrap_or(400),
        Err(_) => 400,
    };

    println!("GEMM microbenchmark: matrixmultiply::sgemm");
    println!("CPU feature detection: checking...");
    println!("  avx2:  {}", is_x86_feature_detected!("avx2"));
    println!("  fma:   {}", is_x86_feature_detected!("fma"));
    println!("  avx:   {}", is_x86_feature_detected!("avx"));
    println!("  sse2:  {}", is_x86_feature_detected!("sse2"));
    println!("Iterations per shape: {}\n", iters);
    println!("{:<30} {:>8} {:>6} {:>6} {:>12} {:>14} {:>12}", 
             "Shape", "M", "K", "N", "GFLOPS", "GB/s", "ms");
    println!("{}", "-".repeat(92));

    for shape in &shapes {
        let m = shape.m;
        let k = shape.k;
        let n = shape.n;

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.01).cos()).collect();
        let mut c: Vec<f32> = vec![0.0; m * n];

        // warmup
        for _ in 0..5 {
            unsafe {
                matrixmultiply::sgemm(
                    m, k, n,
                    1.0,
                    a.as_ptr(), k as isize, 1,
                    b.as_ptr(), n as isize, 1,
                    0.0,
                    c.as_mut_ptr(), n as isize, 1,
                );
            }
        }

        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                matrixmultiply::sgemm(
                    m, k, n,
                    1.0,
                    a.as_ptr(), k as isize, 1,
                    b.as_ptr(), n as isize, 1,
                    0.0,
                    c.as_mut_ptr(), n as isize, 1,
                );
            }
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let gflops = flops / (ms / 1000.0) / 1e9;

        let bytes_read = (m * k + k * n) as f64 * 4.0;
        let bytes_written = (m * n) as f64 * 4.0;
        let total_bytes = bytes_read + bytes_written;
        let bw = total_bytes / (ms / 1000.0) / 1e9;

        println!("{:<30} {:>8} {:>6} {:>6} {:>12.2} {:>14.2} {:>12.3}",
                 shape.name, m, k, n, gflops, bw, ms);
    }

    println!("\nNotes:");
    println!("  - GFLOPS = 2*M*K*N / time(s) / 1e9");
    println!("  - Ryzen 3700X theoretical peak (8-core): ~920 GFLOPS (AVX2 FMA @ 3.6 GHz)");
    println!("  - Per-core peak: ~115 GFLOPS");
}
