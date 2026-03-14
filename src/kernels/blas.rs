#![allow(dead_code)]

#[cfg(target_arch = "aarch64")]
pub const MIN_BLAS_SIZE: usize = 32;

#[cfg(not(target_arch = "aarch64"))]
pub const MIN_BLAS_SIZE: usize = 64;

#[cfg(all(feature = "openblas", not(target_os = "windows")))]
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

#[cfg(all(feature = "openblas", not(target_os = "windows")))]
pub fn matmul_blas(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    unsafe {
        cblas_sgemm(
            101, // RowMajor
            111, // NoTrans
            111, // NoTrans
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

    c
}

#[cfg(not(all(feature = "openblas", not(target_os = "windows"))))]
pub fn matmul_blas(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    use matrixmultiply::sgemm;

    let mut c = vec![0.0f32; m * n];

    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0f32,
            a.as_ptr(),
            k as isize,
            1isize,
            b.as_ptr(),
            n as isize,
            1isize,
            0.0f32,
            c.as_mut_ptr(),
            n as isize,
            1isize,
        );
    }

    c
}
