use std::time::Instant;
use std::ptr;

mod ffi {
    extern "C" {
        pub fn cblas_sgemm(
            Order: i32,
            TransA: i32,
            TransB: i32,
            M: i32,
            N: i32,
            K: i32,
            alpha: f32,
            A: *const f32,
            lda: i32,
            B: *const f32,
            ldb: i32,
            beta: f32,
            C: *mut f32,
            ldc: i32,
        );
    }
}

const MATRIX_SIZE: usize = 1024;
const ITERATIONS: u32 = 100;

fn main() {
    // Allocate and initialize matrices
    let mut a = vec![1.0f32; MATRIX_SIZE * MATRIX_SIZE];
    let mut b = vec![1.0f32; MATRIX_SIZE * MATRIX_SIZE];
    let mut c = vec![0.0f32; MATRIX_SIZE * MATRIX_SIZE];

    // BLAS constants
    const CblasRowMajor: i32 = 101;
    const CblasNoTrans: i32 = 111;

    let m = MATRIX_SIZE as i32;
    let n = MATRIX_SIZE as i32;
    let k = MATRIX_SIZE as i32;
    let lda = MATRIX_SIZE as i32;
    let ldb = MATRIX_SIZE as i32;
    let ldc = MATRIX_SIZE as i32;

    let alpha = 1.0f32;
    let beta = 0.0f32;

    // Warm-up run
    unsafe {
        ptr::write_bytes(c.as_mut_ptr(), 0, c.len());
        ffi::cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        );
    }

    // Timed iterations
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        unsafe {
            ptr::write_bytes(c.as_mut_ptr(), 0, c.len());
            ffi::cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                beta,
                c.as_mut_ptr(),
                ldc,
            );
        }
    }
    let duration = start.elapsed();

    let avg_time = duration.as_secs_f64() / ITERATIONS as f64;
    let gflops = (2.0 * MATRIX_SIZE as f64 * MATRIX_SIZE as f64 * MATRIX_SIZE as f64) / (avg_time * 1e9);

    println!("Matrix multiplication {}x{}", MATRIX_SIZE, MATRIX_SIZE);
    println!("Iterations: {}", ITERATIONS);
    println!("Total time: {:.3} s", duration.as_secs_f64());
    println!("Average time: {:.3} ms", avg_time * 1000.0);
    println!("Performance: {:.2} GFLOPS", gflops);
}
