//! Drop-in replacement for `matrixmultiply::sgemm` using the `gemm` crate
//! with built-in rayon parallelism.
//!
//! Signature matches `matrixmultiply::sgemm` exactly so callers need only
//! change their `use` import.
//!
//! ```text
//! sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
//!
//! Computes: C = alpha * A*B + beta * C
//!   A is m×k, accessed as a[row*rsa + col*csa]
//!   B is k×n, accessed as b[row*rsb + col*csb]
//!   C is m×n, accessed as c[row*rsc + col*csc]
//! ```

use std::sync::LazyLock;

/// Number of threads for parallel GEMM — computed once from system topology.
static NUM_THREADS: LazyLock<usize> = LazyLock::new(|| {
    crate::backend::cpu::topology::physical_core_count().max(1)
});

/// Parallel SGEMM: C ← alpha·A·B + beta·C
///
/// # Safety
/// Same contract as `matrixmultiply::sgemm`: all pointers must be valid,
/// non-overlapping, and strides must be consistent with the matrix dimensions.
pub unsafe fn sgemm(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    rsa: isize,
    csa: isize,
    b: *const f32,
    rsb: isize,
    csb: isize,
    beta: f32,
    c: *mut f32,
    rsc: isize,
    csc: isize,
) {
    // gemm computes: dst = alpha_g * dst + beta_g * lhs * rhs
    // matrixmultiply computes: C = alpha_mm * A*B + beta_mm * C
    // So: alpha_g = beta_mm, beta_g = alpha_mm, and read_dst = beta_mm != 0
    let read_dst = beta != 0.0;

    gemm::gemm::<f32>(
        m,
        n,
        k,
        c,
        csc,
        rsc,
        read_dst,
        a,
        csa,
        rsa,
        b,
        csb,
        rsb,
        beta,  // alpha_g = beta_mm (scales existing dst)
        alpha, // beta_g = alpha_mm (scales product)
        false,
        false,
        false,
        gemm::Parallelism::Rayon(*NUM_THREADS),
    );
}
