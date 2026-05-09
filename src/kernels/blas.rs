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

    // matrixmultiply::sgemm(M, K, N, A, rsa, csa, B, rsb, csb, C, rsc, csc) computes:
    //   C[i][j] = Σ_{p=0}^{K-1} A[rsa·i + csa·p] · B[rsb·p + csb·j]
    // where A is M×K, B is K×N, C is M×N.
    //
    // For row-major storage: rsa=K, csa=1 (next row = +K, next col = +1)
    // For transposed (stored as [K][M], accessed as [M][K]): rsa=1, csa=M
    //   because stored_A[p][i] = base + p·M + i = A[i][p] = base + rsa·i + csa·p
    //   gives rsa=1, csa=M.
    //
    // We keep M=m, K=k, N=n constant and adjust strides for transposition:
    let rsa: isize = if trans_a { 1 } else { k as isize };
    let csa: isize = if trans_a { m as isize } else { 1 };
    let rsb: isize = if trans_b { 1 } else { n as isize };
    let csb: isize = if trans_b { k as isize } else { 1 };

    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0f32,
            a.as_ptr(),
            rsa,
            csa,
            b.as_ptr(),
            rsb,
            csb,
            0.0f32,
            out.as_mut_ptr(),
            n as isize,
            1isize,
        );
    }
}
