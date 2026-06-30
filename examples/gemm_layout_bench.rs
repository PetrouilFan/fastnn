use std::time::Instant;

fn fill(len: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((x >> 32) as u32) as f32 / (u32::MAX as f32);
            v * 2.0 - 1.0
        })
        .collect()
}

fn transpose_mk_to_km(src: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut dst = vec![0.0; src.len()];
    for i in 0..m {
        for j in 0..k {
            dst[j * m + i] = src[i * k + j];
        }
    }
    dst
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn bench_case(m: usize, k: usize, n: usize, iters: usize) {
    let a = fill(m * k, 1);
    let a_t = transpose_mk_to_km(&a, m, k);
    let b = fill(k * n, 2);
    let mut c_ref = vec![0.0f32; m * n];
    let mut c = vec![0.0f32; m * n];

    unsafe {
        fastnn::backend::cpu::sgemm::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            1,
            k as isize,
            0.0,
            c_ref.as_mut_ptr(),
            n as isize,
            1,
        );
    }

    unsafe {
        fastnn::backend::cpu::sgemm::sgemm(
            m,
            k,
            n,
            1.0,
            a_t.as_ptr(),
            1,
            m as isize,
            b.as_ptr(),
            1,
            k as isize,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    let diff = max_abs(&c_ref, &c);

    for _ in 0..5 {
        unsafe {
            fastnn::backend::cpu::sgemm::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                1,
                k as isize,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            fastnn::backend::cpu::sgemm::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                1,
                k as isize,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }
    let row_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    for _ in 0..5 {
        unsafe {
            fastnn::backend::cpu::sgemm::sgemm(
                m,
                k,
                n,
                1.0,
                a_t.as_ptr(),
                1,
                m as isize,
                b.as_ptr(),
                1,
                k as isize,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }
    let t1 = Instant::now();
    for _ in 0..iters {
        unsafe {
            fastnn::backend::cpu::sgemm::sgemm(
                m,
                k,
                n,
                1.0,
                a_t.as_ptr(),
                1,
                m as isize,
                b.as_ptr(),
                1,
                k as isize,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }
    let transposed_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!(
        "m={m:4} k={k:4} n={n:5} row_major_a={row_ms:.4}ms transposed_a={transposed_ms:.4}ms ratio={:.3} max_abs={diff:.3e}",
        transposed_ms / row_ms
    );
}

fn main() {
    let iters = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100);
    println!("matrixmultiply layout bench, iters={iters}");
    bench_case(64, 576, 400, iters);
    bench_case(64, 576, 1600, iters);
    bench_case(32, 288, 1600, iters);
    bench_case(80, 720, 1600, iters);
    bench_case(128, 1152, 100, iters);
}
