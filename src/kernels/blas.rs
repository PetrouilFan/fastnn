#![allow(dead_code)]

#[cfg(target_arch = "aarch64")]
pub const MIN_BLAS_SIZE: usize = 32;

#[cfg(not(target_arch = "aarch64"))]
pub const MIN_BLAS_SIZE: usize = 64;

// Try to use system BLAS (cblas) when available
#[cfg(all(feature = "blas", not(target_os = "windows")))]
#[link(name = "openblas", kind = "static")]
extern "C" {
    fn cblas_sgemm(
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

#[cfg(all(feature = "blas", not(target_os = "windows")))]
pub fn matmul_blas(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    matmul_blas_with_transpose(a, b, m, k, n, false, false)
}

#[cfg(all(feature = "blas", not(target_os = "windows")))]
pub fn matmul_blas_with_transpose(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    matmul_blas_with_transpose_into(a, b, &mut c, m, k, n, trans_a, trans_b);
    c
}

#[cfg(all(feature = "blas", not(target_os = "windows")))]
pub fn matmul_blas_into(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    matmul_blas_with_transpose_into(a, b, out, m, k, n, false, false)
}

#[cfg(all(feature = "blas", not(target_os = "windows")))]
#[allow(clippy::too_many_arguments)]
pub fn matmul_blas_with_transpose_into(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) {
    // CBLAS transpose flags: 111 = NoTrans, 112 = Trans
    let transa = if trans_a { 112 } else { 111 };
    let transb = if trans_b { 112 } else { 111 };

    // For RowMajor:
    // If A is NoTrans (m×k): lda = k
    // If A is Trans (k×m stored as m×k): lda = m
    // If B is NoTrans (k×n): ldb = n
    // If B is Trans (n×k stored as k×n): ldb = k
    let lda = if trans_a { m as i32 } else { k as i32 };
    let ldb = if trans_b { k as i32 } else { n as i32 };

    unsafe {
        cblas_sgemm(
            101, // RowMajor
            transa,
            transb,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            0.0,
            out.as_mut_ptr(),
            n as i32,
        );
    }
}

#[cfg(not(all(feature = "blas", not(target_os = "windows"))))]
pub fn matmul_blas(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    matmul_blas_with_transpose(a, b, m, k, n, false, false)
}

#[cfg(not(all(feature = "blas", not(target_os = "windows"))))]
pub fn matmul_blas_with_transpose(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    matmul_blas_with_transpose_into(a, b, &mut c, m, k, n, trans_a, trans_b);
    c
}

#[cfg(not(all(feature = "blas", not(target_os = "windows"))))]
pub fn matmul_blas_into(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    matmul_blas_with_transpose_into(a, b, out, m, k, n, false, false)
}

#[cfg(not(all(feature = "blas", not(target_os = "windows"))))]
#[allow(clippy::too_many_arguments)]
pub fn matmul_blas_with_transpose_into(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) {
    use matrixmultiply::sgemm;

    // matrixmultiply expects row-major order
    // For C = A @ B with A[m×k], B[k×n], C[m×n]
    // If A is transposed: A is stored as k×m, so dimensions become A[k×m], B[m×n], C[k×n]
    // If B is transposed: B is stored as n×k, so dimensions become A[m×n], B[n×k], C[m×k]
    let (m_final, k_final, n_final, lda, ldb, ldc) = if trans_a && trans_b {
        // A transposed: k×m, B transposed: n×k -> C[k×n]
        (k, n, m, k as isize, 1isize, k as isize)
    } else if trans_a {
        // A transposed: k×m, B not transposed: k×n -> C[k×n]
        (k, m, n, k as isize, n as isize, k as isize)
    } else if trans_b {
        // A not transposed: m×k, B transposed: n×k -> C[m×n]
        (m, n, k, k as isize, 1isize, n as isize)
    } else {
        // No transposes: A[m×k], B[k×n] -> C[m×n]
        (m, k, n, k as isize, n as isize, n as isize)
    };

    unsafe {
        sgemm(
            m_final,
            k_final,
            n_final,
            1.0f32,
            a.as_ptr(),
            lda,
            1isize,
            b.as_ptr(),
            ldb,
            1isize,
            0.0f32,
            out.as_mut_ptr(),
            ldc,
            1isize,
        );
    }
}
