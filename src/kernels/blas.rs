pub const MIN_BLAS_SIZE: usize = 64;

#[cfg(feature = "openblas")]
pub fn matmul_blas(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    use openblas_src::cblas::{sgemm, Transpose, Layout};

    let mut c = vec![0.0f32; m * n];

    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
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

#[cfg(not(feature = "openblas"))]
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
