pub const MIN_BLAS_SIZE: usize = 256;

#[cfg(feature = "blas")]
pub fn matmul_blas(a: &[f32], b: &[f32], m: i32, k: i32, n: i32) -> Vec<f32> {
    use blas_sys::sgemm_;

    let mut c = vec![0.0f32; (m * n) as usize];
    let alpha = 1.0f32;
    let beta = 0.0f32;

    let m = m as i32;
    let n = n as i32;
    let k = k as i32;
    let lda = k;
    let ldb = n;
    let ldc = n;

    let transa: [i8; 1] = [b'N' as i8];
    let transb: [i8; 1] = [b'N' as i8];

    unsafe {
        sgemm_(
            transa.as_ptr(),
            transb.as_ptr(),
            &m,
            &n,
            &k,
            &alpha,
            a.as_ptr(),
            &lda,
            b.as_ptr(),
            &ldb,
            &beta,
            c.as_mut_ptr(),
            &ldc,
        );
    }

    c
}

#[cfg(not(feature = "blas"))]
pub fn matmul_blas(_a: &[f32], _b: &[f32], _m: i32, _k: i32, _n: i32) -> Vec<f32> {
    vec![]
}
