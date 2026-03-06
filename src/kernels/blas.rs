#[cfg(feature = "blas")]
pub const MIN_BLAS_SIZE: usize = 1;

#[cfg(feature = "blas")]
pub fn matmul_blas(a: &[f32], b: &[f32], m: i32, k: i32, n: i32) -> Vec<f32> {
    use matrixmultiply::sgemm;

    let m = m as usize;
    let k = k as usize;
    let n = n as usize;

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

#[cfg(not(feature = "blas"))]
pub const MIN_BLAS_SIZE: usize = 1;

#[cfg(not(feature = "blas"))]
pub fn matmul_blas(_a: &[f32], _b: &[f32], _m: i32, _k: i32, _n: i32) -> Vec<f32> {
    vec![]
}
