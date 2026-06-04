use std::time::Instant;

#[derive(Clone, Copy)]
struct GemmShape {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

#[cfg(feature = "openblas")]
#[link(name = "openblas")]
extern "C" {
    fn cblas_sgemm(
        layout: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

fn fill(len: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((x >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn matrixmultiply_sgemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[cfg(feature = "openblas")]
fn openblas_sgemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn bench_fn(
    f: fn(usize, usize, usize, &[f32], &[f32], &mut [f32]),
    shape: GemmShape,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    warmup: usize,
    iters: usize,
) -> f64 {
    for _ in 0..warmup {
        f(shape.m, shape.k, shape.n, a, b, c);
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f(shape.m, shape.k, shape.n, a, b, c);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    mean(&times)
}

#[cfg(feature = "openblas")]
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let iters = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100);
    let warmup = 5;
    let shapes = [
        GemmShape {
            name: "stem_m16_k27_n25600",
            m: 16,
            k: 27,
            n: 25_600,
        },
        GemmShape {
            name: "yolo_m32_k144_n6400",
            m: 32,
            k: 144,
            n: 6_400,
        },
        GemmShape {
            name: "yolo_m16_k144_n6400",
            m: 16,
            k: 144,
            n: 6_400,
        },
        GemmShape {
            name: "yolo_m32_k288_n1600",
            m: 32,
            k: 288,
            n: 1_600,
        },
        GemmShape {
            name: "yolo_m64_k576_n400",
            m: 64,
            k: 576,
            n: 400,
        },
        GemmShape {
            name: "yolo_m64_k576_n1600",
            m: 64,
            k: 576,
            n: 1_600,
        },
        GemmShape {
            name: "yolo_m80_k720_n1600",
            m: 80,
            k: 720,
            n: 1_600,
        },
        GemmShape {
            name: "yolo_m128_k1152_n100",
            m: 128,
            k: 1_152,
            n: 100,
        },
    ];

    println!(
        "iters={} warmup={} openblas_feature={}",
        iters,
        warmup,
        cfg!(feature = "openblas")
    );
    println!(
        "{:<26} {:>8} {:>8} {:>8} {:>12} {:>12} {:>10} {:>12}",
        "shape", "M", "K", "N", "mm_ms", "blas_ms", "ratio", "max_abs"
    );

    for shape in shapes {
        let a = fill(shape.m * shape.k, 1);
        let b = fill(shape.k * shape.n, 2);
        let mut c_mm = vec![0.0f32; shape.m * shape.n];
        let mm_ms = bench_fn(
            matrixmultiply_sgemm,
            shape,
            &a,
            &b,
            &mut c_mm,
            warmup,
            iters,
        );
        let flops = 2.0 * shape.m as f64 * shape.k as f64 * shape.n as f64;
        let mm_gflops = flops / (mm_ms * 1.0e6);

        #[cfg(feature = "openblas")]
        {
            let mut c_blas = vec![0.0f32; shape.m * shape.n];
            let blas_ms = bench_fn(openblas_sgemm, shape, &a, &b, &mut c_blas, warmup, iters);
            println!(
                "{:<26} {:>8} {:>8} {:>8} {:>8.4}ms {:>8.4}ms {:>10.3} {:>12.3e}  mm_gf/s={:>6.1} blas_gf/s={:>6.1}",
                shape.name,
                shape.m,
                shape.k,
                shape.n,
                mm_ms,
                blas_ms,
                blas_ms / mm_ms,
                max_abs(&c_mm, &c_blas),
                mm_gflops,
                flops / (blas_ms * 1.0e6),
            );
        }

        #[cfg(not(feature = "openblas"))]
        {
            println!(
                "{:<26} {:>8} {:>8} {:>8} {:>8.4}ms {:>12} {:>10} {:>12}  mm_gf/s={:>6.1}",
                shape.name, shape.m, shape.k, shape.n, mm_ms, "n/a", "n/a", "n/a", mm_gflops,
            );
        }
    }
}
