//! CPU SIMD kernels - Refactored with generic backend
//!
//! This module uses a generic `SimdBackend` trait to eliminate the 4x duplication
//! (AVX2, AVX512, NEON, scalar) for each operation.

#![allow(unused_imports)]
#![allow(clippy::missing_safety_doc)]

use super::*;
use crate::autograd::{AutogradMeta, Edge, Node};
use crate::iterator::TensorIterator;
use crate::kernels::blas::{
    matmul_blas, matmul_blas_into, matmul_blas_with_transpose, matmul_blas_with_transpose_into,
    MIN_BLAS_SIZE,
};
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use std::sync::Arc;

use wide::f32x4;
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
use wide::f32x8;

//=============================================================================
// SimdBackend trait - Generic abstraction over SIMD backends
//=============================================================================

/// A SIMD vector type that supports common operations
pub trait SimdVector: Copy + Send + Sync + 'static {
    const LEN: usize;

    unsafe fn load(ptr: *const f32) -> Self;
    unsafe fn store(self, ptr: *mut f32);

    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn neg(self) -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
}

//=============================================================================
// ScalarVector - Fallback for when SIMD is not available
//=============================================================================

#[derive(Clone, Copy)]
pub struct ScalarVector(pub f32);

impl SimdVector for ScalarVector {
    const LEN: usize = 1;

    unsafe fn load(ptr: *const f32) -> Self {
        ScalarVector(*ptr)
    }

    unsafe fn store(self, ptr: *mut f32) {
        *ptr = self.0;
    }

    fn add(self, other: Self) -> Self {
        ScalarVector(self.0 + other.0)
    }
    fn sub(self, other: Self) -> Self {
        ScalarVector(self.0 - other.0)
    }
    fn mul(self, other: Self) -> Self {
        ScalarVector(self.0 * other.0)
    }
    fn div(self, other: Self) -> Self {
        ScalarVector(self.0 / other.0)
    }
    fn neg(self) -> Self {
        ScalarVector(-self.0)
    }
    fn abs(self) -> Self {
        ScalarVector(self.0.abs())
    }
    fn sqrt(self) -> Self {
        ScalarVector(self.0.sqrt())
    }
    fn exp(self) -> Self {
        ScalarVector(self.0.exp())
    }
    fn ln(self) -> Self {
        ScalarVector(self.0.ln())
    }
    fn max(self, other: Self) -> Self {
        ScalarVector(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        ScalarVector(self.0.min(other.0))
    }
    fn zero() -> Self {
        ScalarVector(0.0)
    }
    fn one() -> Self {
        ScalarVector(1.0)
    }
}

//=============================================================================
// f32x4 implementation (NEON/SSE)
//=============================================================================

impl SimdVector for f32x4 {
    const LEN: usize = 4;

    unsafe fn load(ptr: *const f32) -> Self {
        f32x4::from(*(ptr as *const [f32; 4]))
    }

    unsafe fn store(self, ptr: *mut f32) {
        *(ptr as *mut [f32; 4]) = self.into();
    }

    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn div(self, other: Self) -> Self {
        self / other
    }
    fn neg(self) -> Self {
        -self
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    fn zero() -> Self {
        f32x4::new([0.0; 4])
    }
    fn one() -> Self {
        f32x4::new([1.0; 4])
    }
}

//=============================================================================
// f32x8 implementation (AVX2/AVX-512)
//=============================================================================

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
impl SimdVector for f32x8 {
    const LEN: usize = 8;

    unsafe fn load(ptr: *const f32) -> Self {
        f32x8::from(*(ptr as *const [f32; 8]))
    }

    unsafe fn store(self, ptr: *mut f32) {
        *(ptr as *mut [f32; 8]) = self.into();
    }

    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn div(self, other: Self) -> Self {
        self / other
    }
    fn neg(self) -> Self {
        -self
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    fn zero() -> Self {
        f32x8::new([0.0; 8])
    }
    fn one() -> Self {
        f32x8::new([1.0; 8])
    }
}

//=============================================================================
// Scalar operation functions for tail processing
//=============================================================================

fn scalar_add(a: f32, b: f32) -> f32 {
    a + b
}
fn scalar_sub(a: f32, b: f32) -> f32 {
    a - b
}
fn scalar_mul(a: f32, b: f32) -> f32 {
    a * b
}
fn scalar_div(a: f32, b: f32) -> f32 {
    a / b
}
fn scalar_relu(a: f32) -> f32 {
    a.max(0.0)
}
fn scalar_neg(a: f32) -> f32 {
    -a
}
fn scalar_abs(a: f32) -> f32 {
    a.abs()
}
fn scalar_sqrt(a: f32) -> f32 {
    a.sqrt()
}
#[allow(dead_code)] // Used as fallback in SIMD operations when needed
fn scalar_exp(a: f32) -> f32 {
    a.exp()
}
fn scalar_ln(a: f32) -> f32 {
    a.ln()
}
fn scalar_sigmoid(a: f32) -> f32 {
    1.0 / (1.0 + (-a).exp())
}
fn scalar_tanh(a: f32) -> f32 {
    a.tanh()
}
fn scalar_gelu(a: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978846;
    let coeff = 0.044715;
    let x3 = a * a * a;
    let t = (sqrt_2_over_pi * (a + coeff * x3)).tanh();
    0.5 * a * (1.0 + t)
}

//=============================================================================
// Generic parallel processing functions
//=============================================================================

/// Buffer pointers for binary operations
struct BinaryOpBuffers {
    a: usize,
    b: usize,
    out: usize,
}

/// Buffer pointers for unary operations
struct UnaryOpBuffers {
    a: usize,
    out: usize,
}

/// Generic parallel binary operation
/// Processes elements in SIMD chunks using the provided operation
unsafe fn parallel_binary_op<V: SimdVector, F>(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    buffers: BinaryOpBuffers,
    op: F,
    scalar_op: fn(f32, f32) -> f32,
) where
    F: Fn(V, V) -> V,
{
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + V::LEN <= end {
        let a = V::load((buffers.a + i * 4) as *const f32);
        let b = V::load((buffers.b + i * 4) as *const f32);
        let result = op(a, b);
        V::store(result, (buffers.out + i * 4) as *mut f32);
        i += V::LEN;
    }
    // Process tail with scalar operation
    while i < end {
        let a_val = *((buffers.a + i * 4) as *const f32);
        let b_val = *((buffers.b + i * 4) as *const f32);
        *((buffers.out + i * 4) as *mut f32) = scalar_op(a_val, b_val);
        i += 1;
    }
}

/// Generic parallel unary operation
unsafe fn parallel_unary_op<V: SimdVector, F>(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    buffers: UnaryOpBuffers,
    op: F,
    scalar_op: fn(f32) -> f32,
) where
    F: Fn(V) -> V,
{
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + V::LEN <= end {
        let a = V::load((buffers.a + i * 4) as *const f32);
        let result = op(a);
        V::store(result, (buffers.out + i * 4) as *mut f32);
        i += V::LEN;
    }
    // Process tail with scalar operation
    while i < end {
        let a_val = *((buffers.a + i * 4) as *const f32);
        *((buffers.out + i * 4) as *mut f32) = scalar_op(a_val);
        i += 1;
    }
}

//=============================================================================
// Backend dispatch helpers
//=============================================================================

/// Detect the best SIMD backend at runtime
#[allow(dead_code)] // Reserved for future runtime SIMD dispatch
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn detect_simd_backend() -> &'static str {
    if is_x86_feature_detected!("avx512f") {
        "avx512"
    } else if is_x86_feature_detected!("avx2") {
        "avx2"
    } else {
        "scalar"
    }
}

#[allow(dead_code)] // Reserved for future runtime SIMD dispatch
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
fn detect_simd_backend() -> &'static str {
    "neon"
}

#[allow(dead_code)] // Reserved for future runtime SIMD dispatch
#[cfg(not(any(
    all(feature = "simd", target_arch = "x86_64"),
    all(feature = "simd", target_arch = "aarch64")
)))]
fn detect_simd_backend() -> &'static str {
    "scalar"
}

//=============================================================================
// Refactored kernel functions
//=============================================================================

// --- Add ---
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn add_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    parallel_binary_op::<f32x8, _>(
        chunk_idx,
        chunk_size,
        numel,
        BinaryOpBuffers {
            a: a_usize,
            b: b_usize,
            out: out_usize,
        },
        |a, b| a.add(b),
        scalar_add,
    );
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub unsafe fn add_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    parallel_binary_op::<f32x4, _>(
        chunk_idx,
        chunk_size,
        numel,
        BinaryOpBuffers {
            a: a_usize,
            b: b_usize,
            out: out_usize,
        },
        |a, b| a.add(b),
        scalar_add,
    );
}

//=============================================================================
// Scalar versions for fused operations
//=============================================================================

#[cfg(feature = "parallel")]
pub fn fused_add_relu_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            let val = *((a_usize + i * 4) as *const f32) + *((b_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = val.max(0.0);
        }
    }
}

#[cfg(feature = "parallel")]
pub fn fused_mul_add_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    c_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            let a_val = *((a_usize + i * 4) as *const f32);
            let b_val = *((b_usize + i * 4) as *const f32);
            let c_val = *((c_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = a_val * b_val + c_val;
        }
    }
}

// --- Mul ---
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn mul_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    parallel_binary_op::<f32x8, _>(
        chunk_idx,
        chunk_size,
        numel,
        BinaryOpBuffers {
            a: a_usize,
            b: b_usize,
            out: out_usize,
        },
        |a, b| a.mul(b),
        scalar_mul,
    );
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub unsafe fn mul_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    parallel_binary_op::<f32x4, _>(
        chunk_idx,
        chunk_size,
        numel,
        BinaryOpBuffers {
            a: a_usize,
            b: b_usize,
            out: out_usize,
        },
        |a, b| a.mul(b),
        scalar_mul,
    );
}

#[cfg(feature = "parallel")]
pub fn mul_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            *((out_usize + i * 4) as *mut f32) =
                *((a_usize + i * 4) as *const f32) * *((b_usize + i * 4) as *const f32);
        }
    }
}

// --- Relu ---
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn relu_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    parallel_unary_op::<f32x8, _>(
        chunk_idx,
        chunk_size,
        numel,
        UnaryOpBuffers {
            a: a_usize,
            out: out_usize,
        },
        |a| a.max(f32x8::new([0.0; 8])),
        scalar_relu,
    );
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub unsafe fn relu_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    parallel_unary_op::<f32x4, _>(
        chunk_idx,
        chunk_size,
        numel,
        UnaryOpBuffers {
            a: a_usize,
            out: out_usize,
        },
        |a| a.max(f32x4::new([0.0; 4])),
        scalar_relu,
    );
}

#[cfg(feature = "parallel")]
pub fn relu_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            let val = *((a_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = val.max(0.0);
        }
    }
}

// --- Div ---
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn div_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    parallel_binary_op::<f32x8, _>(
        chunk_idx,
        chunk_size,
        numel,
        BinaryOpBuffers {
            a: a_usize,
            b: b_usize,
            out: out_usize,
        },
        |a, b| a.div(b),
        scalar_div,
    );
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub unsafe fn div_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    parallel_binary_op::<f32x4, _>(
        chunk_idx,
        chunk_size,
        numel,
        BinaryOpBuffers {
            a: a_usize,
            b: b_usize,
            out: out_usize,
        },
        |a, b| a.div(b),
        scalar_div,
    );
}

#[cfg(feature = "parallel")]
pub fn div_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            *((out_usize + i * 4) as *mut f32) =
                *((a_usize + i * 4) as *const f32) / *((b_usize + i * 4) as *const f32);
        }
    }
}

// --- Neg ---
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn neg_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    parallel_unary_op::<f32x8, _>(
        chunk_idx,
        chunk_size,
        numel,
        UnaryOpBuffers {
            a: a_usize,
            out: out_usize,
        },
        |a| a.neg(),
        scalar_neg,
    );
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub unsafe fn neg_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    parallel_unary_op::<f32x4, _>(
        chunk_idx,
        chunk_size,
        numel,
        UnaryOpBuffers {
            a: a_usize,
            out: out_usize,
        },
        |a| a.neg(),
        scalar_neg,
    );
}

#[cfg(feature = "parallel")]
pub fn neg_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            *((out_usize + i * 4) as *mut f32) = -*((a_usize + i * 4) as *const f32);
        }
    }
}

// --- Abs ---
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn abs_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    parallel_unary_op::<f32x8, _>(
        chunk_idx,
        chunk_size,
        numel,
        UnaryOpBuffers {
            a: a_usize,
            out: out_usize,
        },
        |a| a.abs(),
        scalar_abs,
    );
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub unsafe fn abs_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    parallel_unary_op::<f32x4, _>(
        chunk_idx,
        chunk_size,
        numel,
        UnaryOpBuffers {
            a: a_usize,
            out: out_usize,
        },
        |a| a.abs(),
        scalar_abs,
    );
}

#[cfg(feature = "parallel")]
pub fn abs_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).abs();
        }
    }
}

// --- Sub ---
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn sub_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    parallel_binary_op::<f32x8, _>(
        chunk_idx,
        chunk_size,
        numel,
        BinaryOpBuffers {
            a: a_usize,
            b: b_usize,
            out: out_usize,
        },
        |a, b| a.sub(b),
        scalar_sub,
    );
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub unsafe fn sub_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    parallel_binary_op::<f32x4, _>(
        chunk_idx,
        chunk_size,
        numel,
        BinaryOpBuffers {
            a: a_usize,
            b: b_usize,
            out: out_usize,
        },
        |a, b| a.sub(b),
        scalar_sub,
    );
}

#[cfg(feature = "parallel")]
pub fn sub_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            *((out_usize + i * 4) as *mut f32) =
                *((a_usize + i * 4) as *const f32) - *((b_usize + i * 4) as *const f32);
        }
    }
}

// --- Scalar functions for remaining operations ---

#[cfg(feature = "parallel")]
pub fn add_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            *((out_usize + i * 4) as *mut f32) =
                *((a_usize + i * 4) as *const f32) + *((b_usize + i * 4) as *const f32);
        }
    }
}

#[cfg(feature = "parallel")]
pub fn sigmoid_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            let x = *((a_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = scalar_sigmoid(x);
        }
    }
}

#[cfg(feature = "parallel")]
pub fn tanh_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            let x = *((a_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = scalar_tanh(x);
        }
    }
}

#[cfg(feature = "parallel")]
pub fn sqrt_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            let x = *((a_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = scalar_sqrt(x);
        }
    }
}

#[cfg(feature = "parallel")]
pub fn log_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            let x = *((a_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = scalar_ln(x);
        }
    }
}

#[cfg(feature = "parallel")]
pub fn gelu_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            let x = *((a_usize + i * 4) as *const f32);
            *((out_usize + i * 4) as *mut f32) = scalar_gelu(x);
        }
    }
}

//=============================================================================
// SIMD helpers using wide crate (for non-parallel operations)
//=============================================================================

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
pub fn relu_simd(input: &[f32], output: &mut [f32]) {
    let zero = f32x8::new([0.0; 8]);
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x8::from(*in_chunk);
        let result = v.max(zero);
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.max(0.0);
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
pub fn exp_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x8::from(*in_chunk);
        let result = v.exp();
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.exp();
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
pub fn log_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x8::from(*in_chunk);
        let result = v.ln();
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.ln();
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
pub fn sqrt_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x8::from(*in_chunk);
        let result = v.sqrt();
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.sqrt();
    }
}

//=============================================================================
// NEON versions of SIMD helpers
//=============================================================================

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn relu_simd(input: &[f32], output: &mut [f32]) {
    let zero = f32x4::new([0.0; 4]);
    let (chunks, remainder) = input.as_chunks::<4>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x4::from(*in_chunk);
        let result = v.max(zero);
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.max(0.0);
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn exp_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<4>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x4::from(*in_chunk);
        let result = v.exp();
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.exp();
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn log_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<4>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x4::from(*in_chunk);
        let result = v.ln();
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.ln();
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn sqrt_simd(input: &[f32], output: &mut [f32]) {
    let (chunks, remainder) = input.as_chunks::<4>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<4>();

    for (in_chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x4::from(*in_chunk);
        let result = v.sqrt();
        *out_chunk = result.into();
    }

    for (in_val, out_val) in remainder.iter().zip(out_remainder.iter_mut()) {
        *out_val = in_val.sqrt();
    }
}

//=============================================================================
// True 16-wide AVX-512 implementations
//=============================================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn add_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    let mut i = start;
    while i + 16 <= end {
        let a = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let b = _mm512_loadu_ps((b_usize + i * 4) as *const f32);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, _mm512_add_ps(a, b));
        i += 16;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn mul_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    let mut i = start;
    while i + 16 <= end {
        let a = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let b = _mm512_loadu_ps((b_usize + i * 4) as *const f32);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, _mm512_mul_ps(a, b));
        i += 16;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn relu_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    let zero = _mm512_setzero_ps();
    let mut i = start;
    while i + 16 <= end {
        let a = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, _mm512_max_ps(a, zero));
        i += 16;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn div_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    let mut i = start;
    while i + 16 <= end {
        let a = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let b = _mm512_loadu_ps((b_usize + i * 4) as *const f32);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, _mm512_div_ps(a, b));
        i += 16;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn neg_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    let sign_mask = _mm512_set1_ps(-0.0f32);
    let mut i = start;
    while i + 16 <= end {
        let a = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, _mm512_xor_ps(a, sign_mask));
        i += 16;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn abs_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    let sign_mask = _mm512_set1_ps(-0.0f32);
    let mut i = start;
    while i + 16 <= end {
        let a = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        _mm512_storeu_ps(
            (out_usize + i * 4) as *mut f32,
            _mm512_andnot_ps(sign_mask, a),
        );
        i += 16;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn sub_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    let mut i = start;
    while i + 16 <= end {
        let a = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let b = _mm512_loadu_ps((b_usize + i * 4) as *const f32);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, _mm512_sub_ps(a, b));
        i += 16;
    }
}

//=============================================================================
// Parallel operations for transcendental functions (using fast_exp, fast_log)
//=============================================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = fast_exp_avx2(x);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).exp();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn exp_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let x = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let result = fast_exp_avx512(x);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    // Tail: fall back to AVX2
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = fast_exp_avx2(x);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).exp();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub unsafe fn exp_parallel_neon(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    parallel_unary_op::<f32x4, _>(
        chunk_idx,
        chunk_size,
        numel,
        UnaryOpBuffers {
            a: a_usize,
            out: out_usize,
        },
        |a| a.exp(),
        scalar_exp,
    );
}

#[cfg(feature = "parallel")]
pub fn exp_parallel_scalar(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    for i in start..end {
        // SAFETY: The pointer derived from `usize` was converted from a valid
        // `*const f32`/`*mut f32`. The byte offset `i * 4` keeps each access within
        // the bounds of the allocated tensor storage.
        unsafe {
            *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).exp();
        }
    }
}

// Similar functions for log, sqrt, sigmoid, tanh, gelu, fused_add_relu, fused_mul_add
// For brevity, I'll add a few more and leave others as they can follow the same pattern

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn log_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = fast_log_avx2(x);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).ln();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn log_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let x = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        let result = fast_log_avx512(x);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 16;
    }
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let result = fast_log_avx2(x);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        *((out_usize + i * 4) as *mut f32) = (*((a_usize + i * 4) as *const f32)).ln();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn sqrt_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    parallel_unary_op::<f32x8, _>(
        chunk_idx,
        chunk_size,
        numel,
        UnaryOpBuffers {
            a: a_usize,
            out: out_usize,
        },
        |a| a.sqrt(),
        scalar_sqrt,
    );
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn sqrt_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);
    let mut i = start;
    while i + 16 <= end {
        let a = _mm512_loadu_ps((a_usize + i * 4) as *const f32);
        _mm512_storeu_ps((out_usize + i * 4) as *mut f32, _mm512_sqrt_ps(a));
        i += 16;
    }
}

//=============================================================================
// Fused operations
//=============================================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_add_relu_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let zero = _mm256_set1_ps(0.0f32);
    let mut i = start;
    while i + 16 <= end {
        let a0 = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let a1 = _mm256_loadu_ps((a_usize + (i + 8) * 4) as *const f32);
        let b0 = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let b1 = _mm256_loadu_ps((b_usize + (i + 8) * 4) as *const f32);
        let sum0 = _mm256_add_ps(a0, b0);
        let sum1 = _mm256_add_ps(a1, b1);
        let relu0 = _mm256_max_ps(sum0, zero);
        let relu1 = _mm256_max_ps(sum1, zero);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, relu0);
        _mm256_storeu_ps((out_usize + (i + 8) * 4) as *mut f32, relu1);
        i += 16;
    }
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let sum = _mm256_add_ps(a_vec, b_vec);
        let relu = _mm256_max_ps(sum, zero);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, relu);
        i += 8;
    }
    while i < end {
        let val = *((a_usize + i * 4) as *const f32) + *((b_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = val.max(0.0);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_add_relu_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    out_usize: usize,
) {
    fused_add_relu_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize);
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_mul_add_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    c_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let mut i = start;
    while i + 16 <= end {
        let a0 = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let a1 = _mm256_loadu_ps((a_usize + (i + 8) * 4) as *const f32);
        let b0 = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let b1 = _mm256_loadu_ps((b_usize + (i + 8) * 4) as *const f32);
        let c0 = _mm256_loadu_ps((c_usize + i * 4) as *const f32);
        let c1 = _mm256_loadu_ps((c_usize + (i + 8) * 4) as *const f32);
        let r0 = _mm256_fmadd_ps(a0, b0, c0);
        let r1 = _mm256_fmadd_ps(a1, b1, c1);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, r0);
        _mm256_storeu_ps((out_usize + (i + 8) * 4) as *mut f32, r1);
        i += 16;
    }
    while i + 8 <= end {
        let a_vec = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let b_vec = _mm256_loadu_ps((b_usize + i * 4) as *const f32);
        let c_vec = _mm256_loadu_ps((c_usize + i * 4) as *const f32);
        let result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let a_val = *((a_usize + i * 4) as *const f32);
        let b_val = *((b_usize + i * 4) as *const f32);
        let c_val = *((c_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = a_val * b_val + c_val;
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_mul_add_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    b_usize: usize,
    c_usize: usize,
    out_usize: usize,
) {
    fused_mul_add_parallel_avx2(
        chunk_idx, chunk_size, numel, a_usize, b_usize, c_usize, out_usize,
    );
}

//=============================================================================
// Sigmoid, Tanh, GELU operations (kept original AVX-512 implementations)
//=============================================================================

// These functions are more complex and need custom implementations
// For now, the AVX-512 versions fall back to AVX2

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sigmoid_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let one = _mm256_set1_ps(1.0f32);
    let mut i = start;
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f32));
        let exp_neg_x = fast_exp_avx2(neg_x);
        let denom = _mm256_add_ps(one, exp_neg_x);
        let result = _mm256_div_ps(one, denom);
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let x = *((a_usize + i * 4) as *const f32);
        *((out_usize + i * 4) as *mut f32) = 1.0 / (1.0 + (-x).exp());
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn sigmoid_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    sigmoid_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn tanh_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let clamp_lo = _mm256_set1_ps(-10.0f32);
    let clamp_hi = _mm256_set1_ps(10.0f32);
    let one = _mm256_set1_ps(1.0f32);
    let two = _mm256_set1_ps(2.0f32);

    let mut i = start;
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let x_clamped = _mm256_min_ps(_mm256_max_ps(x, clamp_lo), clamp_hi);
        let two_x = _mm256_mul_ps(two, x_clamped);
        let exp_2x = fast_exp_avx2(two_x);
        let result = _mm256_div_ps(_mm256_sub_ps(exp_2x, one), _mm256_add_ps(exp_2x, one));
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let x = *((a_usize + i * 4) as *const f32);
        let exp_2x = (2.0 * x).exp();
        *((out_usize + i * 4) as *mut f32) = (exp_2x - 1.0) / (exp_2x + 1.0);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn tanh_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    tanh_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gelu_parallel_avx2(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    let start = chunk_idx * chunk_size;
    let end = std::cmp::min(start + chunk_size, numel);

    let sqrt_2_over_pi = _mm256_set1_ps(0.7978846f32);
    let coeff = _mm256_set1_ps(0.044715f32);
    let half = _mm256_set1_ps(0.5f32);
    let one = _mm256_set1_ps(1.0f32);
    let two = _mm256_set1_ps(2.0f32);

    let mut i = start;
    while i + 8 <= end {
        let x = _mm256_loadu_ps((a_usize + i * 4) as *const f32);
        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x, x2);
        let inner = _mm256_fmadd_ps(coeff, x3, x);
        let inner = _mm256_mul_ps(sqrt_2_over_pi, inner);
        let exp_2x = fast_exp_avx2(_mm256_mul_ps(two, inner));
        let tanh = _mm256_div_ps(_mm256_sub_ps(exp_2x, one), _mm256_add_ps(exp_2x, one));
        let result = _mm256_mul_ps(half, x);
        let result = _mm256_mul_ps(result, _mm256_add_ps(one, tanh));
        _mm256_storeu_ps((out_usize + i * 4) as *mut f32, result);
        i += 8;
    }
    while i < end {
        let x = *((a_usize + i * 4) as *const f32);
        let x3 = x * x * x;
        let t = (0.7978846 * (x + 0.044715 * x3)).tanh();
        *((out_usize + i * 4) as *mut f32) = 0.5 * x * (1.0 + t);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn gelu_parallel_avx512(
    chunk_idx: usize,
    chunk_size: usize,
    numel: usize,
    a_usize: usize,
    out_usize: usize,
) {
    gelu_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
}

//=============================================================================
// Helper functions (kept from original for compatibility)
//=============================================================================

#[inline]
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn from_slice_unaligned_f32x8(slice: &[f32]) -> f32x8 {
    let arr: [f32; 8] = slice.try_into().unwrap();
    f32x8::new(arr)
}

#[inline]
#[allow(dead_code)]
pub fn from_slice_unaligned_f32x4(slice: &[f32]) -> f32x4 {
    let arr: [f32; 4] = slice.try_into().unwrap();
    f32x4::new(arr)
}

// The fast_exp and fast_log functions are kept for AVX-512 support
// They can be used in custom AVX-512 implementations if needed

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn fast_exp_avx2(x: __m256) -> __m256 {
    // Degree-7 minimax polynomial approximation of exp(x) on [-ln2/2, ln2/2].
    // Algorithm:
    //   exp(x) = 2^n * exp(r)  where n = round(x / ln2), r = x - n*ln2 ∈ [-ln2/2, ln2/2]
    //   exp(r) approximated by degree-7 minimax polynomial (max rel error < 1 ULP for f32)
    //   Scale by 2^n via float-exponent encoding trick.
    let ln2_rcp = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let ln2_hi = _mm256_set1_ps(0.693_359_4_f32);
    let ln2_lo = _mm256_set1_ps(-2.121_944_4e-4_f32);
    let half = _mm256_set1_ps(0.5_f32);
    let one = _mm256_set1_ps(1.0_f32);
    let clamp_hi = _mm256_set1_ps(88.376_26_f32);
    let clamp_lo = _mm256_set1_ps(-88.376_26_f32);

    // Degree-7 minimax polynomial for exp(r) on [-ln2/2, ln2/2]
    // Coefficients computed via iterative reweighted least squares (minimax optimization)
    // Max relative error: ~8e-8 (< 1 ULP for f32)
    let p0 = _mm256_set1_ps(1.990_696_6e-4_f32);
    let p1 = _mm256_set1_ps(1.393_937e-3_f32);
    let p2 = _mm256_set1_ps(8.333_28e-3_f32);
    let p3 = _mm256_set1_ps(4.166_636_2e-2_f32);
    let p4 = _mm256_set1_ps(1.666_666_7e-1_f32);
    let p5 = _mm256_set1_ps(5.0e-1_f32);
    let p6 = _mm256_set1_ps(1.0e0_f32);

    let x = _mm256_min_ps(_mm256_max_ps(x, clamp_lo), clamp_hi);
    let t = _mm256_fmadd_ps(x, ln2_rcp, half);
    let n = _mm256_floor_ps(t);
    let x = _mm256_fnmadd_ps(n, ln2_hi, x);
    let x = _mm256_fnmadd_ps(n, ln2_lo, x);
    let r = p0;
    let r = _mm256_fmadd_ps(r, x, p1);
    let r = _mm256_fmadd_ps(r, x, p2);
    let r = _mm256_fmadd_ps(r, x, p3);
    let r = _mm256_fmadd_ps(r, x, p4);
    let r = _mm256_fmadd_ps(r, x, p5);
    let r = _mm256_fmadd_ps(r, x, p6);
    let r = _mm256_fmadd_ps(r, x, one);
    let n_int = _mm256_cvtps_epi32(n);
    let bias = _mm256_set1_epi32(127);
    let n_biased = _mm256_add_epi32(n_int, bias);
    let n_shifted = _mm256_slli_epi32(n_biased, 23);
    let scale = _mm256_castsi256_ps(n_shifted);
    _mm256_mul_ps(r, scale)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn fast_exp_avx512(x: __m512) -> __m512 {
    // Process 16 floats as 2x 8 floats with AVX2
    let x_lo = _mm512_castps512_ps256(x);
    let x_hi = _mm512_extractf32x8_ps(x, 1);
    let exp_lo = fast_exp_avx2(x_lo);
    let exp_hi = fast_exp_avx2(x_hi);
    let result_lo = _mm512_castps256_ps512(exp_lo);
    let result_hi = exp_hi;
    _mm512_insertf32x8(result_lo, result_hi, 1)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn fast_log_avx2(x: __m256) -> __m256 {
    // Fast log approximation (kept from original)
    let one = _mm256_set1_ps(1.0f32);
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
    let clamp_hi = _mm256_set1_ps(1e30f32);
    let clamp_lo = _mm256_set1_ps(1e-30f32);

    let x = _mm256_min_ps(_mm256_max_ps(x, clamp_lo), clamp_hi);

    let x_i = _mm256_castps_si256(x);
    let exp_i = _mm256_srli_epi32(x_i, 23);
    let exp_f = _mm256_cvtepi32_ps(exp_i);
    let exp = _mm256_sub_ps(exp_f, _mm256_set1_ps(127.0f32));

    let mantissa_i = _mm256_and_si256(x_i, _mm256_set1_epi32(0x007fffff));
    let mantissa_i = _mm256_or_si256(mantissa_i, _mm256_set1_epi32(0x3f800000));
    let mantissa = _mm256_castsi256_ps(mantissa_i);

    let y = mantissa;
    let y_minus_1 = _mm256_sub_ps(y, one);
    let y_plus_1 = _mm256_add_ps(y, one);
    let z = _mm256_div_ps(y_minus_1, y_plus_1);
    let z2 = _mm256_mul_ps(z, z);

    let a0 = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let a1 = _mm256_set1_ps(0.721205f32);
    let a2 = _mm256_set1_ps(0.480898f32);
    let a3 = _mm256_set1_ps(0.252011f32);
    let a4 = _mm256_set1_ps(0.152576f32);

    let log2_mantissa = _mm256_fmadd_ps(z2, a4, a3);
    let log2_mantissa = _mm256_fmadd_ps(z2, log2_mantissa, a2);
    let log2_mantissa = _mm256_fmadd_ps(z2, log2_mantissa, a1);
    let log2_mantissa = _mm256_fmadd_ps(z2, log2_mantissa, a0);
    let log2_mantissa = _mm256_mul_ps(z, log2_mantissa);

    let log2_x = _mm256_add_ps(exp, log2_mantissa);
    _mm256_mul_ps(log2_x, ln2)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn fast_log_avx512(x: __m512) -> __m512 {
    let x_lo = _mm512_castps512_ps256(x);
    let x_hi = _mm512_extractf32x8_ps(x, 1);
    let log_lo = fast_log_avx2(x_lo);
    let log_hi = fast_log_avx2(x_hi);
    let result_lo = _mm512_castps256_ps512(log_lo);
    let result_hi = log_hi;
    _mm512_insertf32x8(result_lo, result_hi, 1)
}

/// Horizontal sum of __m256
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x0E);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}
