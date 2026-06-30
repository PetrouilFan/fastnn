//! CPU microkernels for the v2.0.0 AOT compiler.
//!
//! Hand-tuned AVX2/AVX-512/SWAR microkernels extracted from the v1.x codebase.
//! The CPU backend's compile() step routes matched IR nodes to these functions.

#![allow(clippy::missing_safety_doc, dead_code)]

use std::sync::OnceLock;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

pub mod activations;
pub mod conv;
pub mod conv_gemm;
pub mod direct_conv;
pub mod direct_depthwise_conv;
pub mod gemm;
pub mod misc;
pub mod ops;
#[cfg(test)]
pub mod tests;

use crate::dtypes::{F4x8, I4x8, I8x4};

pub use activations::*;
pub use conv::*;
pub use gemm::*;
pub use misc::*;
pub use ops::*;

// ============================================================
// Feature detection — checked once, reused forever.
// ============================================================

static HAS_AVX512: OnceLock<bool> = OnceLock::new();
static HAS_AVX2: OnceLock<bool> = OnceLock::new();
static HAS_F16C: OnceLock<bool> = OnceLock::new();

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn has_avx512() -> bool {
    *HAS_AVX512
        .get_or_init(|| is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw"))
}
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(crate) fn has_avx2() -> bool {
    *HAS_AVX2.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

/// Public runtime check — returns true if AVX2+FMA is available.
/// Always callable (returns false when SIMD support is not compiled in).
#[inline]
pub fn simd_avx2_available() -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        has_avx2()
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        false
    }
}
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn has_f16c() -> bool {
    *HAS_F16C.get_or_init(|| is_x86_feature_detected!("f16c"))
}

// ============================================================
// Thread-local scratch buffers
// ============================================================

thread_local! {
    static PACKED_SCRATCH: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

pub(crate) fn with_scratch<R>(size: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    PACKED_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        if size > buf.len() {
            buf.resize(size, 0.0);
        } else {
            buf.truncate(size);
        }
        f(&mut buf)
    })
}

// ============================================================
// Generic thread-local Vec pool — one pool per element type
// ============================================================

macro_rules! tls_pool {
    ($ty:ty, $pool:ident, $scoped:ident, $alloc:ident, $alloc_zeroed:ident) => {
        thread_local! {
            static $pool: std::cell::RefCell<Vec<Vec<$ty>>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }

        pub(crate) struct $scoped {
            inner: Option<Vec<$ty>>,
        }

        impl std::ops::Deref for $scoped {
            type Target = Vec<$ty>;
            fn deref(&self) -> &Vec<$ty> {
                self.inner.as_ref().expect("ScopedVec already consumed")
            }
        }

        impl std::ops::DerefMut for $scoped {
            fn deref_mut(&mut self) -> &mut Vec<$ty> {
                self.inner.as_mut().expect("ScopedVec already consumed")
            }
        }

        impl $scoped {
            pub(crate) fn take(mut self) -> Vec<$ty> {
                self.inner
                    .take()
                    .expect("ScopedVec::take called on already-consumed ScopedVec")
            }
        }

        impl Drop for $scoped {
            fn drop(&mut self) {
                if let Some(v) = self.inner.take() {
                    if v.capacity() <= 100_000_000 {
                        $pool.with(|pool| {
                            pool.borrow_mut().push(v);
                        });
                    }
                }
            }
        }

        pub(crate) fn $alloc(min_capacity: usize) -> $scoped {
            let pooled = $pool.with(|pool| pool.borrow_mut().pop());
            let mut v = if let Some(v) = pooled {
                crate::backend::cpu::telemetry::record_tls_vec_reuse();
                v
            } else {
                crate::backend::cpu::telemetry::record_tls_vec_alloc();
                Vec::new()
            };
            if v.capacity() < min_capacity {
                v.reserve(min_capacity - v.len());
            }
            // SAFETY: `v` was just reserved with sufficient capacity; `set_len` is
            // safe because the elements are uninitialized but immediately overwritten.
            unsafe {
                v.set_len(min_capacity);
            }
            $scoped { inner: Some(v) }
        }

        pub(crate) fn $alloc_zeroed(len: usize) -> $scoped {
            let mut v = $alloc(len);
            v.fill(Default::default());
            v
        }
    };
}

tls_pool!(f32, TLS_VEC_POOL_F32, ScopedVec, tls_alloc_f32, tls_alloc_zeroed_f32);
tls_pool!(i8, TLS_VEC_POOL_I8, ScopedVecI8, tls_alloc_i8, tls_alloc_zeroed_i8);
tls_pool!(u8, TLS_VEC_POOL_U8, ScopedVecU8, tls_alloc_u8, tls_alloc_zeroed_u8);
tls_pool!(i32, TLS_VEC_POOL_I32, ScopedVecI32, tls_alloc_i32, tls_alloc_zeroed_i32);
tls_pool!(u32, TLS_VEC_POOL_U32, ScopedVecU32, tls_alloc_u32, tls_alloc_zeroed_u32);
tls_pool!(I8x4, TLS_VEC_POOL_I8X4, ScopedVecI8x4, tls_alloc_i8x4, tls_alloc_zeroed_i8x4);
tls_pool!(I4x8, TLS_VEC_POOL_I4X8, ScopedVecI4x8, tls_alloc_i4x8, tls_alloc_zeroed_i4x8);
tls_pool!(F4x8, TLS_VEC_POOL_F4X8, ScopedVecF4x8, tls_alloc_f4x8, tls_alloc_zeroed_f4x8);

/// Backward-compatible alias: `TlsVecPool::alloc(...)` still returns `ScopedVec` (f32).
pub(crate) struct TlsVecPool;

impl TlsVecPool {
    pub(crate) fn alloc(min_capacity: usize) -> ScopedVec {
        tls_alloc_f32(min_capacity)
    }
    pub(crate) fn alloc_zeroed(len: usize) -> ScopedVec {
        tls_alloc_zeroed_f32(len)
    }
}

// ============================================================
// Shared SIMD utilities
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: `v` is a valid __m256 register value from an AVX2 intrinsic operation.
pub(crate) unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x0E);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[inline]
fn fma_f32_slice(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if has_avx512() {
            // SAFETY: Feature check ensures AVX512F is available; slices are valid
            // and non-overlapping per the caller.
            return unsafe { fma_f32_avx512(a, b) };
        }
        if has_avx2() {
            // SAFETY: Feature check ensures AVX2+FMA are available; slices are valid
            // and non-overlapping per the caller.
            return unsafe { fma_f32_avx2(a, b) };
        }
    }

    fma_f32_scalar(a, b)
}

#[inline]
fn fma_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[inline(always)]
fn affine_sum_term(sum_source: &[f32], len: usize, zero: f32) -> f32 {
    if zero == 0.0 {
        0.0
    } else {
        sum_source.iter().take(len).copied().sum()
    }
}

#[inline(always)]
fn apply_affine_dot(dot: f32, scale: f32, zero: f32, input_sum: f32) -> f32 {
    dot * scale + zero * input_sum
}

const I8_ACTIVATION_PAYLOAD_HEADER_LEN: usize = 8;

#[derive(Clone, Copy)]
pub(crate) struct I8ActivationAffine {
    pub(crate) scale: f32,
    pub(crate) zero: f32,
}

impl I8ActivationAffine {
    #[inline(always)]
    fn dequantize(self, q: i8) -> f32 {
        (q as f32) * self.scale + self.zero
    }

    #[inline(always)]
    fn sum_from_q_sum(self, q_sum: i32, len: usize) -> f32 {
        (q_sum as f32) * self.scale + (len as f32) * self.zero
    }
}

#[inline(always)]
fn parse_i8_activation_payload(activation_payload: &[u8]) -> (I8ActivationAffine, &[u8]) {
    let scale = if activation_payload.len() >= 4 {
        f32::from_le_bytes(activation_payload[0..4].try_into().unwrap())
    } else {
        1.0
    };
    let zero = if activation_payload.len() >= I8_ACTIVATION_PAYLOAD_HEADER_LEN {
        f32::from_le_bytes(
            activation_payload[4..I8_ACTIVATION_PAYLOAD_HEADER_LEN]
                .try_into()
                .unwrap(),
        )
    } else {
        0.0
    };
    let payload = activation_payload
        .get(I8_ACTIVATION_PAYLOAD_HEADER_LEN..)
        .unwrap_or(&[]);
    (I8ActivationAffine { scale, zero }, payload)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
// SAFETY: Caller must ensure `a` and `b` are valid, non-overlapping slices
// of equal length, and AVX2+FMA are available at the call site.
unsafe fn fma_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 32 <= len {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        acc2 = _mm256_fmadd_ps(a2, b2, acc2);
        acc3 = _mm256_fmadd_ps(a3, b3, acc3);
        i += 32;
    }

    while i + 8 <= len {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        i += 8;
    }

    let combined = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut total = hsum256_ps(combined);

    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
// SAFETY: Caller must ensure `a` and `b` are valid, non-overlapping slices
// of equal length, and AVX512F is available at the call site.
unsafe fn fma_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut i = 0;

    while i + 64 <= len {
        let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
        let a1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
        let b1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
        let a2 = _mm512_loadu_ps(a.as_ptr().add(i + 32));
        let b2 = _mm512_loadu_ps(b.as_ptr().add(i + 32));
        let a3 = _mm512_loadu_ps(a.as_ptr().add(i + 48));
        let b3 = _mm512_loadu_ps(b.as_ptr().add(i + 48));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        acc2 = _mm512_fmadd_ps(a2, b2, acc2);
        acc3 = _mm512_fmadd_ps(a3, b3, acc3);
        i += 64;
    }

    while i + 16 <= len {
        let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        i += 16;
    }

    let combined = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    let mut total = _mm512_reduce_add_ps(combined);

    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}
