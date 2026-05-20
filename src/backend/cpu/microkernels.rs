//! CPU microkernels for the v2.0.0 AOT compiler.
//!
//! Hand-tuned AVX2/AVX-512/SWAR microkernels extracted from the v1.x codebase.
//! The CPU backend's compile() step routes matched IR nodes to these functions.

#![allow(dead_code)]
#![allow(clippy::missing_safety_doc)]

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::dtypes::{F16x2, U4x8, U8x4};
use crate::dtypes::{F32x1, PackedWord};
use crate::packed_tensor::PackedTensor;
use std::sync::OnceLock;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

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
    { has_avx2() }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    { false }
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

fn with_scratch<R>(size: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
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

thread_local! {
    static TLS_VEC_POOL: std::cell::RefCell<Vec<Vec<f32>>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

pub struct ScopedVec {
    inner: Option<Vec<f32>>,
}

impl std::ops::Deref for ScopedVec {
    type Target = Vec<f32>;
    fn deref(&self) -> &Vec<f32> {
        self.inner.as_ref().expect("ScopedVec already consumed")
    }
}

impl std::ops::DerefMut for ScopedVec {
    fn deref_mut(&mut self) -> &mut Vec<f32> {
        self.inner.as_mut().expect("ScopedVec already consumed")
    }
}

impl ScopedVec {
    pub fn take(mut self) -> Vec<f32> {
        self.inner
            .take()
            .expect("ScopedVec::take called on already-consumed ScopedVec")
    }
}

impl Drop for ScopedVec {
    fn drop(&mut self) {
        if let Some(v) = self.inner.take() {
            if v.capacity() <= 100_000_000 {
                TLS_VEC_POOL.with(|pool| {
                    pool.borrow_mut().push(v);
                });
            }
        }
    }
}

pub struct TlsVecPool;

impl TlsVecPool {
    pub fn alloc(min_capacity: usize) -> ScopedVec {
        let mut v = TLS_VEC_POOL
            .with(|pool| pool.borrow_mut().pop())
            .unwrap_or_default();
        if v.capacity() < min_capacity {
            v.reserve(min_capacity - v.len());
        }
        unsafe {
            v.set_len(min_capacity);
        }
        ScopedVec { inner: Some(v) }
    }

    pub fn alloc_zeroed(len: usize) -> ScopedVec {
        let mut v = Self::alloc(len);
        v.fill(0.0);
        v
    }
}

thread_local! {
    static CONV_COL_BUF: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
    static CONV_WEIGHT_T_BUF: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
    static CONV_TEMP_OUT_BUF: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

macro_rules! get_conv_buf {
    ($buf:expr, $size:expr) => {{
        let size: usize = $size;
        (*$buf).with(|b| {
            let mut b = b.borrow_mut();
            if b.len() < size {
                b.resize(size, 0.0);
            }
            // SAFETY: thread-local data lives for the duration of the thread,
            // so extending the lifetime to 'static is sound.
            unsafe {
                std::mem::transmute::<
                    std::cell::RefMut<'_, Vec<f32>>,
                    std::cell::RefMut<'static, Vec<f32>>,
                >(b)
            }
        })
    }};
}

// ============================================================
// Shared SIMD utilities
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: __m256) -> f32 {
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
            return unsafe { fma_f32_avx512(a, b) };
        }
        if has_avx2() {
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

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
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

// ============================================================
// AVX2 elementwise microkernels — 19 unary + 6 binary f32 ops
// ============================================================

// Strategy:
//   - Simple ops (relu, neg, abs, sqrt, sign, logical_not): direct AVX2 intrinsic
//   - Complex ops (exp, log, sigmoid, tanh, gelu, silu, mish, softplus, hardswish, elu):
//     SIMD load/store struct (8-wide loop) with per-element scalar math.
//     Still wins ~2-3× from memory bandwidth.
//   - Param ops (leaky_relu, clamp): blend/max/min intrinsics
//   - Binary ops (add, sub, mul, div, max, min): direct AVX2 when dense (no broadcast)

// ── Simple ops (direct AVX2 intrinsics) ──────────────────────

#[inline]
pub fn relu_f32_scalar(x: f32) -> f32 { x.max(0.0) }

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn relu_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    while i + 8 <= len {
        _mm256_storeu_ps(output.as_mut_ptr().add(i),
            _mm256_max_ps(_mm256_loadu_ps(input.as_ptr().add(i)), zero));
        i += 8;
    }
    for j in i..len { output[j] = relu_f32_scalar(input[j]); }
}

#[inline]
pub fn neg_f32_scalar(x: f32) -> f32 { -x }

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn neg_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let sign_mask = _mm256_set1_ps(-0.0f32);
    while i + 8 <= len {
        _mm256_storeu_ps(output.as_mut_ptr().add(i),
            _mm256_xor_ps(_mm256_loadu_ps(input.as_ptr().add(i)), sign_mask));
        i += 8;
    }
    for j in i..len { output[j] = neg_f32_scalar(input[j]); }
}

#[inline]
pub fn abs_f32_scalar(x: f32) -> f32 { x.abs() }

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn abs_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let inv_sign_mask = _mm256_set1_ps(f32::from_bits(0x7fff_ffff));
    while i + 8 <= len {
        _mm256_storeu_ps(output.as_mut_ptr().add(i),
            _mm256_and_ps(_mm256_loadu_ps(input.as_ptr().add(i)), inv_sign_mask));
        i += 8;
    }
    for j in i..len { output[j] = abs_f32_scalar(input[j]); }
}

#[inline]
pub fn sqrt_f32_scalar(x: f32) -> f32 { x.sqrt() }

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn sqrt_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    while i + 8 <= len {
        _mm256_storeu_ps(output.as_mut_ptr().add(i),
            _mm256_sqrt_ps(_mm256_loadu_ps(input.as_ptr().add(i))));
        i += 8;
    }
    for j in i..len { output[j] = sqrt_f32_scalar(input[j]); }
}

#[inline]
pub fn sign_f32_scalar(x: f32) -> f32 { x.signum() }

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn sign_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        _mm256_storeu_ps(output.as_mut_ptr().add(i),
            _mm256_or_ps(_mm256_and_ps(_mm256_cmp_ps::<{_CMP_GT_OQ}>(x, zero), one),
                          _mm256_and_ps(_mm256_cmp_ps::<{_CMP_LT_OQ}>(x, zero), neg_one)));
        i += 8;
    }
    for j in i..len { output[j] = sign_f32_scalar(input[j]); }
}

#[inline]
pub fn round_f32_scalar(x: f32) -> f32 { x.round() }

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn round_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    while i + 8 <= len {
        _mm256_storeu_ps(output.as_mut_ptr().add(i),
            _mm256_round_ps(_mm256_loadu_ps(input.as_ptr().add(i)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        i += 8;
    }
    for j in i..len { output[j] = round_f32_scalar(input[j]); }
}

#[inline]
pub fn logical_not_f32_scalar(x: f32) -> f32 { if x == 0.0 { 1.0 } else { 0.0 } }

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn logical_not_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    while i + 8 <= len {
        _mm256_storeu_ps(output.as_mut_ptr().add(i),
            _mm256_and_ps(_mm256_cmp_ps::<{_CMP_EQ_OQ}>(_mm256_loadu_ps(input.as_ptr().add(i)), zero), one));
        i += 8;
    }
    for j in i..len { output[j] = logical_not_f32_scalar(input[j]); }
}

// ── Parametric ops ───────────────────────────────────────────

#[inline]
pub fn clamp_f32_scalar(x: f32, min_val: f32, max_val: f32) -> f32 { x.max(min_val).min(max_val) }

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn clamp_f32_avx2(input: &[f32], output: &mut [f32], min_val: f32, max_val: f32) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let vmin = _mm256_set1_ps(min_val);
    let vmax = _mm256_set1_ps(max_val);
    while i + 8 <= len {
        _mm256_storeu_ps(output.as_mut_ptr().add(i),
            _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(input.as_ptr().add(i)), vmin), vmax));
        i += 8;
    }
    for j in i..len { output[j] = clamp_f32_scalar(input[j], min_val, max_val); }
}

#[inline]
pub fn leaky_relu_f32_scalar(x: f32, slope: f32) -> f32 { if x > 0.0 { x } else { x * slope } }

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn leaky_relu_f32_avx2(input: &[f32], output: &mut [f32], slope: f32) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    let vslope = _mm256_set1_ps(slope);
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        _mm256_storeu_ps(output.as_mut_ptr().add(i),
            _mm256_blendv_ps(_mm256_mul_ps(x, vslope), x,
                _mm256_cmp_ps::<{_CMP_GT_OQ}>(x, zero)));
        i += 8;
    }
    for j in i..len { output[j] = leaky_relu_f32_scalar(input[j], slope); }
}

// ── Complex ops (SIMD struct load/store, per-element scalar math) ───

/// Macro: unrolls 8-element SIMD struct loads/stores, calling `$scalar_fn` per lane.
macro_rules! avx2_elwise_fallback {
    ($name:ident, $scalar_fn:expr) => {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        pub unsafe fn $name(input: &[f32], output: &mut [f32]) {
            let len = output.len().min(input.len());
            let mut i = 0;
            while i + 8 <= len {
                let x0 = *input.as_ptr().add(i);
                let x1 = *input.as_ptr().add(i + 1);
                let x2 = *input.as_ptr().add(i + 2);
                let x3 = *input.as_ptr().add(i + 3);
                let x4 = *input.as_ptr().add(i + 4);
                let x5 = *input.as_ptr().add(i + 5);
                let x6 = *input.as_ptr().add(i + 6);
                let x7 = *input.as_ptr().add(i + 7);
                *output.as_mut_ptr().add(i)     = $scalar_fn(x0);
                *output.as_mut_ptr().add(i + 1) = $scalar_fn(x1);
                *output.as_mut_ptr().add(i + 2) = $scalar_fn(x2);
                *output.as_mut_ptr().add(i + 3) = $scalar_fn(x3);
                *output.as_mut_ptr().add(i + 4) = $scalar_fn(x4);
                *output.as_mut_ptr().add(i + 5) = $scalar_fn(x5);
                *output.as_mut_ptr().add(i + 6) = $scalar_fn(x6);
                *output.as_mut_ptr().add(i + 7) = $scalar_fn(x7);
                i += 8;
            }
            for j in i..len {
                *output.as_mut_ptr().add(j) = $scalar_fn(*input.as_ptr().add(j));
            }
        }
    };
}

#[inline] pub fn exp_f32_scalar(x: f32) -> f32 { x.exp() }

#[inline] pub fn sigmoid_f32_scalar(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[inline] pub fn tanh_f32_scalar(x: f32) -> f32 { x.tanh() }

// ============================================================
// SIMD vector helpers — reusable across activation microkernels
// ============================================================

/// Vector exp(x) for a single `__m256`. Degree-5 polynomial approximation
/// with range reduction via `x·log2(e)`, Horner with FMA, 2^k reconstruction.
/// Relative error < 2e-5 over [-88, 88].
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn exp_avx2_vec(x: __m256) -> __m256 {
    let vlog2e = _mm256_set1_ps(1.4426950408889634f32);
    let vln2_hi = _mm256_set1_ps(0.693359375f32);
    let vln2_lo = _mm256_set1_ps(-2.12194440e-4f32);
    let v_120 = _mm256_set1_ps(1.0f32 / 120.0f32);
    let v_24 = _mm256_set1_ps(1.0f32 / 24.0f32);
    let v_6 = _mm256_set1_ps(1.0f32 / 6.0f32);
    let v_half = _mm256_set1_ps(0.5f32);

    let k_float = _mm256_round_ps(_mm256_mul_ps(x, vlog2e), _MM_FROUND_TO_NEAREST_INT);
    let t = _mm256_fnmadd_ps(k_float, vln2_hi, x);
    let t = _mm256_fnmadd_ps(k_float, vln2_lo, t);

    let mut p = _mm256_fmadd_ps(v_120, t, v_24);
    p = _mm256_fmadd_ps(p, t, v_6);
    p = _mm256_fmadd_ps(p, t, v_half);
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.0f32));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.0f32));

    let k_int = _mm256_cvtps_epi32(k_float);
    let k_clamped = _mm256_min_epi32(
        _mm256_max_epi32(k_int, _mm256_set1_epi32(-126)),
        _mm256_set1_epi32(127),
    );
    let exp_factor =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(k_clamped, _mm256_set1_epi32(127)), 23));
    _mm256_mul_ps(p, exp_factor)
}

/// Vector tanh(x) for a single `__m256`. Uses identity: tanh(x) = 2·sigmoid(2x) - 1.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tanh_avx2_vec(x: __m256) -> __m256 {
    let vone = _mm256_set1_ps(1.0f32);
    let vtwo = _mm256_set1_ps(2.0f32);
    let neg_2x = _mm256_xor_ps(_mm256_mul_ps(x, vtwo), _mm256_set1_ps(-0.0f32));
    let e_neg2 = exp_avx2_vec(neg_2x);
    let sig_2x = _mm256_div_ps(vone, _mm256_add_ps(vone, e_neg2));
    _mm256_sub_ps(_mm256_mul_ps(vtwo, sig_2x), vone)
}

/// Vector ln(x) for a single `__m256`. Uses frexp decomposition to extract
/// exponent and mantissa in [1, 2), then degree-4 minimax polynomial for ln(m).
/// Relative error < 1e-3 over normal f32 range, acceptable for NN activations.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn log_avx2_vec(x: __m256) -> __m256 {
    // Extract exponent: e = floor(log2(x))
    let raw = _mm256_castps_si256(x);
    let biased_exp = _mm256_srli_epi32(raw, 23);
    let exp_i = _mm256_sub_epi32(biased_exp, _mm256_set1_epi32(127));
    let exp_f = _mm256_cvtepi32_ps(exp_i);

    // Extract mantissa in [1, 2): clear exponent, set to 127 (bias)
    let mant_raw = _mm256_and_si256(raw, _mm256_set1_epi32(0x007FFFFF));
    let mant_f = _mm256_castsi256_ps(
        _mm256_or_si256(mant_raw, _mm256_set1_epi32(0x3F800000)),
    );

    // Degree-4 minimax polynomial for ln(m) on m ∈ [1, 2]
    // Coefficients from Sollya: max error ~3e-4
    // ln(m) ≈ c0 + c1*m + c2*m² + c3*m³ + c4*m⁴
    let c0 = _mm256_set1_ps(-1.7417939f32);
    let c1 = _mm256_set1_ps(2.8212026f32);
    let c2 = _mm256_set1_ps(-1.4699568f32);
    let c3 = _mm256_set1_ps(0.4477982f32);
    let c4 = _mm256_set1_ps(-0.05657085f32);

    let m2 = _mm256_mul_ps(mant_f, mant_f);
    let m3 = _mm256_mul_ps(m2, mant_f);
    let m4 = _mm256_mul_ps(m3, mant_f);

    // Horner: (((c4*m + c3)*m + c2)*m + c1)*m + c0
    let mut log_m = _mm256_fmadd_ps(c4, mant_f, c3);
    log_m = _mm256_fmadd_ps(log_m, mant_f, c2);
    log_m = _mm256_fmadd_ps(log_m, mant_f, c1);
    log_m = _mm256_fmadd_ps(log_m, mant_f, c0);

    // ln(x) = ln(m * 2^e) = ln(m) + e * ln(2)
    _mm256_fmadd_ps(exp_f, _mm256_set1_ps(0.6931471805599453f32), log_m)
}

// ============================================================
// Public AVX2 activation microkernels — all use the helpers above
// ============================================================

/// True AVX2 exp(x) via polynomial approximation.
/// Delegates to `exp_avx2_vec` for the vectorized computation.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn exp_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        _mm256_storeu_ps(output.as_mut_ptr().add(i), exp_avx2_vec(x));
        i += 8;
    }
    for j in i..len {
        *output.as_mut_ptr().add(j) = input.as_ptr().add(j).read().exp();
    }
}

#[inline] pub fn log_f32_scalar(x: f32) -> f32 { x.ln() }

/// True AVX2 ln(x) using frexp decomposition + polynomial on [1, 2).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn log_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        _mm256_storeu_ps(output.as_mut_ptr().add(i), log_avx2_vec(x));
        i += 8;
    }
    for j in i..len {
        *output.as_mut_ptr().add(j) = input.as_ptr().add(j).read().ln();
    }
}

/// True AVX2 sigmoid(x) = 1 / (1 + exp(-x)).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn sigmoid_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let vone = _mm256_set1_ps(1.0f32);
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f32));
        let e_neg = exp_avx2_vec(neg_x);
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_div_ps(vone, _mm256_add_ps(vone, e_neg)),
        );
        i += 8;
    }
    for j in i..len {
        let x = *input.as_ptr().add(j);
        *output.as_mut_ptr().add(j) = 1.0 / (1.0 + (-x).exp());
    }
}

/// True AVX2 tanh(x) = 2·sigmoid(2x) - 1.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn tanh_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        _mm256_storeu_ps(output.as_mut_ptr().add(i), tanh_avx2_vec(x));
        i += 8;
    }
    for j in i..len {
        let x = *input.as_ptr().add(j);
        *output.as_mut_ptr().add(j) = x.tanh();
    }
}

#[inline] pub fn elu_f32_scalar(x: f32) -> f32 { if x > 0.0 { x } else { x.exp() - 1.0 } }

/// True AVX2 ELU(x) = if x > 0 then x else exp(x) - 1.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn elu_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let vzero = _mm256_setzero_ps();
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let exp_x = exp_avx2_vec(x);
        let exp_minus_1 = _mm256_sub_ps(exp_x, _mm256_set1_ps(1.0f32));
        // blend: when x > 0, select x; otherwise select exp(x)-1
        let mask = _mm256_cmp_ps::<{_CMP_GT_OQ}>(x, vzero);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_blendv_ps(exp_minus_1, x, mask));
        i += 8;
    }
    for j in i..len {
        let x = *input.as_ptr().add(j);
        *output.as_mut_ptr().add(j) = if x > 0.0 { x } else { x.exp() - 1.0 };
    }
}

#[inline]
pub fn gelu_f32_scalar(x: f32) -> f32 {
    let x3 = x * x * x;
    let tanh_arg = 0.7978846f32 * (x + 0.044715f32 * x3);
    0.5 * x * (1.0 + tanh_arg.tanh())
}

/// True AVX2 GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715·x³))).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn gelu_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let vhalf = _mm256_set1_ps(0.5f32);
    let vone = _mm256_set1_ps(1.0f32);
    let vsqrt2pi = _mm256_set1_ps(0.7978845608028654f32); // √(2/π)
    let vcoeff = _mm256_set1_ps(0.044715f32);
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
        let tanh_arg = _mm256_mul_ps(vsqrt2pi, _mm256_add_ps(x, _mm256_mul_ps(vcoeff, x3)));
        let t = tanh_avx2_vec(tanh_arg);
        let result = _mm256_mul_ps(_mm256_mul_ps(vhalf, x), _mm256_add_ps(vone, t));
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }
    for j in i..len {
        let x = *input.as_ptr().add(j);
        let x3 = x * x * x;
        let tanh_arg = 0.7978846f32 * (x + 0.044715f32 * x3);
        *output.as_mut_ptr().add(j) = 0.5 * x * (1.0 + tanh_arg.tanh());
    }
}

#[inline] pub fn silu_f32_scalar(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// True AVX2 SiLU(x) = x * sigmoid(x).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn silu_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let vone = _mm256_set1_ps(1.0f32);
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f32));
        let e_neg = exp_avx2_vec(neg_x);
        let sig = _mm256_div_ps(vone, _mm256_add_ps(vone, e_neg));
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_mul_ps(x, sig));
        i += 8;
    }
    for j in i..len {
        let x = *input.as_ptr().add(j);
        *output.as_mut_ptr().add(j) = x / (1.0 + (-x).exp());
    }
}

#[inline] pub fn softplus_f32_scalar(x: f32) -> f32 { (1.0 + x.exp()).ln() }

/// True AVX2 softplus(x) = ln(1 + exp(x)).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn softplus_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let vone = _mm256_set1_ps(1.0f32);
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let e_x = exp_avx2_vec(x);
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            log_avx2_vec(_mm256_add_ps(vone, e_x)),
        );
        i += 8;
    }
    for j in i..len {
        let x = *input.as_ptr().add(j);
        *output.as_mut_ptr().add(j) = (1.0 + x.exp()).ln();
    }
}

#[inline] pub fn hardswish_f32_scalar(x: f32) -> f32 { x * (x + 3.0).clamp(0.0, 6.0) / 6.0 }

/// True AVX2 hardswish(x) = x * clamp(x+3, 0, 6) / 6.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn hardswish_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let vthree = _mm256_set1_ps(3.0f32);
    let vsix = _mm256_set1_ps(6.0f32);
    let vzero = _mm256_setzero_ps();
    let vinvsix = _mm256_set1_ps(1.0f32 / 6.0f32);
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let x3 = _mm256_add_ps(x, vthree);
        let clamped = _mm256_min_ps(_mm256_max_ps(x3, vzero), vsix);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_mul_ps(x, clamped), vinvsix));
        i += 8;
    }
    for j in i..len {
        let x = *input.as_ptr().add(j);
        *output.as_mut_ptr().add(j) = x * (x + 3.0).clamp(0.0, 6.0) / 6.0;
    }
}

#[inline] pub fn mish_f32_scalar(x: f32) -> f32 { let sp = (1.0 + x.exp()).ln(); x * sp.tanh() }

/// True AVX2 mish(x) = x * tanh(ln(1 + exp(x))).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn mish_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let vone = _mm256_set1_ps(1.0f32);
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let sp = log_avx2_vec(_mm256_add_ps(vone, exp_avx2_vec(x))); // ln(1+exp(x))
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_mul_ps(x, tanh_avx2_vec(sp)));
        i += 8;
    }
    for j in i..len {
        let x = *input.as_ptr().add(j);
        let sp = (1.0 + x.exp()).ln();
        *output.as_mut_ptr().add(j) = x * sp.tanh();
    }
}

/// log_softmax: needs max + sum reduction. AVX2 for max + output phases;
/// sum phase uses scalar exp accumulator for numeric precision.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn log_softmax_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    if len == 0 { return; }
    // 1. Find max with AVX2 horizontal reduction
    let mut max_val = f32::NEG_INFINITY;
    let mut i = 0;
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        // permute+max to get horizontal max across all 8 lanes
        let mx = _mm256_max_ps(x, _mm256_permute2f128_ps(x, x, 1));
        let mx = _mm256_max_ps(mx, _mm256_shuffle_ps(mx, mx, 0b_00_11_10_01));
        let mx = _mm256_max_ps(mx, _mm256_shuffle_ps(mx, mx, 0b_00_00_00_10));
        let candidate = _mm256_cvtss_f32(mx);
        if candidate > max_val { max_val = candidate; }
        i += 8;
    }
    for j in i..len {
        let v = *input.as_ptr().add(j);
        if v > max_val { max_val = v; }
    }
    // 2. Sum of exp(x - max) using f64 accumulator for precision
    let mut sum = 0.0f64;
    for j in 0..len {
        sum += ((*input.as_ptr().add(j)) - max_val).exp() as f64;
    }
    let log_sum = (sum as f32).ln();
    // 3. Write output with AVX2
    let mut i = 0;
    let vmax = _mm256_set1_ps(max_val);
    let vlogsum = _mm256_set1_ps(log_sum);
    while i + 8 <= len {
        let shifted = _mm256_sub_ps(_mm256_loadu_ps(input.as_ptr().add(i)), vmax);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_sub_ps(shifted, vlogsum));
        i += 8;
    }
    for j in i..len {
        *output.as_mut_ptr().add(j) = (*input.as_ptr().add(j)) - max_val - log_sum;
    }
}

/// Scalar fallback for log_softmax
pub fn log_softmax_f32_scalar_all(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    if len == 0 { return; }
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &x in &input[..len] { sum += ((x - max_val).exp()) as f64; }
    let log_sum = (sum as f32).ln();
    for i in 0..len { output[i] = input[i] - max_val - log_sum; }
}

// ── Binary ops (AVX2 when dense, scalar broadcast fallback) ──

macro_rules! impl_binary_arith_broadcast {
    ($scalar_fn:ident, $avx2_fn:ident, $op:tt, $avx2_intrin:ident) => {
        #[inline]
        pub fn $scalar_fn(a: &[f32], b: &[f32], output: &mut [f32]) {
            let a_len = a.len();
            let b_len = b.len();
            for i in 0..output.len() {
                output[i] = a[i % a_len] $op b[i % b_len];
            }
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        pub unsafe fn $avx2_fn(a: &[f32], b: &[f32], output: &mut [f32]) {
            let len = output.len();
            let a_len = a.len();
            let b_len = b.len();
            let mut i = 0;
            if a_len == len && b_len == len {
                while i + 8 <= len {
                    let va = _mm256_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                    _mm256_storeu_ps(output.as_mut_ptr().add(i), $avx2_intrin(va, vb));
                    i += 8;
                }
            } else if a_len == 1 && b_len == len {
                let va = _mm256_broadcast_ss(&a[0]);
                while i + 8 <= len {
                    let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                    _mm256_storeu_ps(output.as_mut_ptr().add(i), $avx2_intrin(va, vb));
                    i += 8;
                }
            } else if a_len == len && b_len == 1 {
                let vb = _mm256_broadcast_ss(&b[0]);
                while i + 8 <= len {
                    let va = _mm256_loadu_ps(a.as_ptr().add(i));
                    _mm256_storeu_ps(output.as_mut_ptr().add(i), $avx2_intrin(va, vb));
                    i += 8;
                }
            }
            for j in i..len {
                *output.as_mut_ptr().add(j) = a[j % a_len] $op b[j % b_len];
            }
        }
    };
}

macro_rules! impl_binary_minmax_broadcast {
    ($scalar_fn:ident, $avx2_fn:ident, $method:ident, $avx2_intrin:ident) => {
        #[inline]
        pub fn $scalar_fn(a: &[f32], b: &[f32], output: &mut [f32]) {
            let a_len = a.len();
            let b_len = b.len();
            for i in 0..output.len() {
                output[i] = a[i % a_len].$method(b[i % b_len]);
            }
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        pub unsafe fn $avx2_fn(a: &[f32], b: &[f32], output: &mut [f32]) {
            let len = output.len();
            let a_len = a.len();
            let b_len = b.len();
            let mut i = 0;
            if a_len == len && b_len == len {
                while i + 8 <= len {
                    let va = _mm256_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                    _mm256_storeu_ps(output.as_mut_ptr().add(i), $avx2_intrin(va, vb));
                    i += 8;
                }
            } else if a_len == 1 && b_len == len {
                let va = _mm256_broadcast_ss(&a[0]);
                while i + 8 <= len {
                    let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                    _mm256_storeu_ps(output.as_mut_ptr().add(i), $avx2_intrin(va, vb));
                    i += 8;
                }
            } else if a_len == len && b_len == 1 {
                let vb = _mm256_broadcast_ss(&b[0]);
                while i + 8 <= len {
                    let va = _mm256_loadu_ps(a.as_ptr().add(i));
                    _mm256_storeu_ps(output.as_mut_ptr().add(i), $avx2_intrin(va, vb));
                    i += 8;
                }
            }
            for j in i..len {
                *output.as_mut_ptr().add(j) = a[j % a_len].$method(b[j % b_len]);
            }
        }
    };
}

impl_binary_arith_broadcast!(add_f32_scalar_broadcast, add_f32_avx2_broadcast, +, _mm256_add_ps);
impl_binary_arith_broadcast!(sub_f32_scalar_broadcast, sub_f32_avx2_broadcast, -, _mm256_sub_ps);
impl_binary_arith_broadcast!(mul_f32_scalar_broadcast, mul_f32_avx2_broadcast, *, _mm256_mul_ps);
impl_binary_arith_broadcast!(div_f32_scalar_broadcast, div_f32_avx2_broadcast, /, _mm256_div_ps);
impl_binary_minmax_broadcast!(max_f32_scalar_broadcast, max_f32_avx2_broadcast, max, _mm256_max_ps);
impl_binary_minmax_broadcast!(min_f32_scalar_broadcast, min_f32_avx2_broadcast, min, _mm256_min_ps);

// ============================================================
// GEMM constants
// ============================================================

/// Cache-blocked GEMV for packed types.
/// Tiles the K dimension to keep activation blocks in L2 cache.
const K_BLOCK_SIZE: usize = 4096;

// ============================================================
// Type-dispatched SIMD GEMV entry point
// ============================================================

#[inline(always)]
pub fn gemv_packed_simd<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    match (T::BIT_WIDTH, T::IS_FLOAT) {
        (8, false) => {
            let w = unsafe { &*(weights as *const _ as *const PackedTensor<U8x4>) };
            return gemv_u8x4_dispatch(w, activation, output);
        }
        (4, false) => {
            let w = unsafe { &*(weights as *const _ as *const PackedTensor<U4x8>) };
            return gemv_u4x8_dispatch(w, activation, output);
        }
        (16, true) => {
            let w = unsafe { &*(weights as *const _ as *const PackedTensor<F16x2>) };
            return gemv_f16x2_dispatch(w, activation, output);
        }
        (32, true) => {
            let w = unsafe { &*(weights as *const _ as *const PackedTensor<F32x1>) };
            return gemv_f32x1_dispatch(w, activation, output);
        }
        _ => {}
    }

    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(T::ITEMS);

    if k <= K_BLOCK_SIZE {
        gemv_packed_inner::<T>(weights, activation, output, m, k, k_packed);
    } else {
        gemv_packed_blocked::<T>(weights, activation, output, m, k, k_packed);
    }
}

// ============================================================
// U8x4: AVX2 int8→f32 widening + FMA
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(unused_variables)]
fn gemv_u8x4_dispatch(weights: &PackedTensor<U8x4>, activation: &[f32], output: &mut [f32]) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(4);
    let weights_u32 = weights.as_u32();

    if has_avx512() {
        for row in 0..m {
            let dot = unsafe {
                gemv_row_u8x4_avx512(weights_u32, activation, row * k_packed, k, k_packed)
            };
            output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    } else if has_avx2() {
        for row in 0..m {
            let dot =
                unsafe { gemv_row_u8x4_avx2(weights_u32, activation, row * k_packed, k, k_packed) };
            output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    } else {
        for row in 0..m {
            let row_offset = row * k_packed;
            let dot =
                unsafe { gemv_row_u8x4_scalar(weights_u32, activation, row_offset, k, k_packed) };
            output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn gemv_row_u8x4_scalar(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    // NOTE: U8x4 values are stored as SIGNED i8 (range -128..127), not unsigned u8.
    // The cast (bytes[j] as i8) as f32 correctly interprets the packed data.
    let mut total = 0.0f32;
    for p in 0..k_packed {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        for j in 0..4 {
            let idx = p * 4 + j;
            if idx < k {
                total += (bytes[j] as i8) as f32 * activation[idx];
            }
        }
    }
    total
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn gemv_row_u8x4_avx2(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    while p + 1 < k_packed && act_idx + 8 <= k {
        if p + 4 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 4) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];

        let word_pair = _mm_set_epi32(0, 0, w1 as i32, w0 as i32);
        let weight_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(word_pair));

        let act = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc = _mm256_fmadd_ps(weight_f32, act, acc);

        p += 2;
        act_idx += 8;
    }

    let mut total = hsum256_ps(acc);

    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        for j in 0..4 {
            let idx = act_idx + j;
            if idx < k {
                total += (bytes[j] as i8) as f32 * activation[idx];
            }
        }
        p += 1;
        act_idx += 4;
    }

    total
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn gemv_row_u8x4_avx512(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    while p + 4 <= k_packed && act_idx + 16 <= k {
        if p + 8 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 8) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];
        let w2 = weights_u32[row_offset + p + 2];
        let w3 = weights_u32[row_offset + p + 3];

        let bytes = _mm_set_epi32(w3 as i32, w2 as i32, w1 as i32, w0 as i32);
        let i32x16 = _mm512_cvtepi8_epi32(bytes);
        let weight_f32 = _mm512_cvtepi32_ps(i32x16);

        let act = _mm512_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm512_fmadd_ps(weight_f32, act, acc0);

        p += 4;
        act_idx += 16;
    }

    let mut total = _mm512_reduce_add_ps(acc0);

    while p + 2 <= k_packed && act_idx + 8 <= k {
        for word_off in 0..2 {
            let w = weights_u32[row_offset + p + word_off];
            let bytes = w.to_le_bytes();
            for j in 0..4 {
                total += (bytes[j] as i8) as f32 * activation[act_idx + word_off * 4 + j];
            }
        }
        p += 2;
        act_idx += 8;
    }

    while p < k_packed && act_idx + 4 <= k {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        let weights_f32: [f32; 4] = [
            bytes[0] as i8 as f32,
            bytes[1] as i8 as f32,
            bytes[2] as i8 as f32,
            bytes[3] as i8 as f32,
        ];
        let acts = [
            activation[act_idx],
            activation[act_idx + 1],
            activation[act_idx + 2],
            activation[act_idx + 3],
        ];

        let w_vec = _mm_loadu_ps(weights_f32.as_ptr());
        let a_vec = _mm_loadu_ps(acts.as_ptr());
        let prod = _mm_mul_ps(w_vec, a_vec);
        let hsum = _mm_hadd_ps(prod, prod);
        let final_sum = _mm_hadd_ps(hsum, hsum);
        total += _mm_cvtss_f32(final_sum);

        p += 1;
        act_idx += 4;
    }

    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        for j in 0..4 {
            let idx = act_idx + j;
            if idx < k {
                total += (bytes[j] as i8) as f32 * activation[idx];
            }
        }
        p += 1;
        act_idx += 4;
    }

    total
}

// ============================================================
// U4x8: AVX2 nibble extraction → int32 → f32 → FMA
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(unused_variables)]
fn gemv_u4x8_dispatch(weights: &PackedTensor<U4x8>, activation: &[f32], output: &mut [f32]) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(8);
    let weights_u32 = weights.as_u32();

    if has_avx512() {
        for row in 0..m {
            let dot = unsafe {
                gemv_row_u4x8_avx512(weights_u32, activation, row * k_packed, k, k_packed)
            };
            output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    } else if has_avx2() {
        for row in 0..m {
            let dot =
                unsafe { gemv_row_u4x8_avx2(weights_u32, activation, row * k_packed, k, k_packed) };
            output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    } else {
        gemv_generic_fallback(weights, activation, output);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn gemv_row_u4x8_avx2(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;
    let mask_lo = _mm256_set1_epi32(0xF);
    let sign_ext = _mm256_set1_epi32(8);

    while p + 2 <= k_packed && act_idx + 16 <= k {
        if p + 4 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 4) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];

        let shift0 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
        let w0v = _mm256_set1_epi32(w0 as i32);
        let w1v = _mm256_set1_epi32(w1 as i32);

        let nib_lo0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift0), mask_lo);
        let nib_lo1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift0), mask_lo);

        let signed_lo0 = _mm256_sub_epi32(_mm256_xor_si256(nib_lo0, sign_ext), sign_ext);
        let signed_lo1 = _mm256_sub_epi32(_mm256_xor_si256(nib_lo1, sign_ext), sign_ext);

        let fl0 = _mm256_cvtepi32_ps(signed_lo0);
        let fl1 = _mm256_cvtepi32_ps(signed_lo1);

        let al0 = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm256_fmadd_ps(fl0, al0, acc0);

        let al1 = _mm256_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc1 = _mm256_fmadd_ps(fl1, al1, acc1);

        p += 2;
        act_idx += 16;
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    let mut total = hsum256_ps(acc0);

    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        for j in 0..8 {
            let idx = act_idx + j;
            if idx < k {
                let nibble = (w >> (j * 4)) & 0xF;
                let signed = if nibble & 0x8 != 0 {
                    (nibble | 0xFFFFFFF0) as i32
                } else {
                    nibble as i32
                };
                total += signed as f32 * activation[idx];
            }
        }
        p += 1;
        act_idx += 8;
    }

    total
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn gemv_row_u4x8_avx512(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;
    let mask_lo = _mm256_set1_epi32(0xF);
    let sign_ext = _mm256_set1_epi32(8);

    while p + 2 <= k_packed && act_idx + 16 <= k {
        if p + 4 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 4) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];
        let w0v = _mm256_set1_epi32(w0 as i32);
        let w1v = _mm256_set1_epi32(w1 as i32);

        let shift_lo = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
        let nib_lo0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift_lo), mask_lo);
        let nib_lo1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift_lo), mask_lo);

        let signed_lo0 = _mm256_sub_epi32(_mm256_xor_si256(nib_lo0, sign_ext), sign_ext);
        let signed_lo1 = _mm256_sub_epi32(_mm256_xor_si256(nib_lo1, sign_ext), sign_ext);

        let fl0 = _mm512_cvtepi32_ps(_mm512_castsi256_si512(signed_lo0));
        let fl1 = _mm512_cvtepi32_ps(_mm512_castsi256_si512(signed_lo1));
        let al0 = _mm512_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm512_fmadd_ps(fl0, al0, acc0);
        let al1 = _mm512_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc1 = _mm512_fmadd_ps(fl1, al1, acc1);

        p += 2;
        act_idx += 16;
    }

    let combined = _mm512_add_ps(acc0, acc1);
    let mut total = _mm512_reduce_add_ps(combined);

    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        for j in 0..8 {
            let idx = act_idx + j;
            if idx < k {
                let nibble = (w >> (j * 4)) & 0xF;
                let signed = if nibble & 0x8 != 0 {
                    (nibble | 0xFFFFFFF0) as i32
                } else {
                    nibble as i32
                };
                total += signed as f32 * activation[idx];
            }
        }
        p += 1;
        act_idx += 8;
    }

    total
}

// ============================================================
// F16x2: F16C hardware half→float conversion + FMA
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(unused_variables)]
fn gemv_f16x2_dispatch(weights: &PackedTensor<F16x2>, activation: &[f32], output: &mut [f32]) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(2);
    let weights_u32 = weights.as_u32();

    if has_f16c() {
        for row in 0..m {
            let dot = unsafe {
                gemv_row_f16x2_f16c(weights_u32, activation, row * k_packed, k, k_packed)
            };
            output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    } else {
        gemv_generic_fallback(weights, activation, output);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn u32x4_to_f32x8_f16c(w0: u32, w1: u32, w2: u32, w3: u32) -> __m256 {
    let half_bits = _mm_set_epi32(w3 as i32, w2 as i32, w1 as i32, w0 as i32);
    _mm256_cvtph_ps(half_bits)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn u32x2_to_f32x4_f16c(w0: u32, w1: u32) -> __m128 {
    let half_bits = _mm_set_epi32(0, 0, w1 as i32, w0 as i32);
    _mm_cvtph_ps(half_bits)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn gemv_row_f16x2_f16c(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    while p + 8 <= k_packed && act_idx + 16 <= k {
        if p + 16 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 16) as *const i8,
                _MM_HINT_T0,
            );
        }

        let f0 = u32x4_to_f32x8_f16c(
            weights_u32[row_offset + p],
            weights_u32[row_offset + p + 1],
            weights_u32[row_offset + p + 2],
            weights_u32[row_offset + p + 3],
        );
        let a0 = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm256_fmadd_ps(f0, a0, acc0);

        let f1 = u32x4_to_f32x8_f16c(
            weights_u32[row_offset + p + 4],
            weights_u32[row_offset + p + 5],
            weights_u32[row_offset + p + 6],
            weights_u32[row_offset + p + 7],
        );
        let a1 = _mm256_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc1 = _mm256_fmadd_ps(f1, a1, acc1);

        p += 8;
        act_idx += 16;
    }

    while p + 4 <= k_packed && act_idx + 8 <= k {
        let f = u32x4_to_f32x8_f16c(
            weights_u32[row_offset + p],
            weights_u32[row_offset + p + 1],
            weights_u32[row_offset + p + 2],
            weights_u32[row_offset + p + 3],
        );
        let a = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm256_fmadd_ps(f, a, acc0);
        p += 4;
        act_idx += 8;
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    let mut total = hsum256_ps(acc0);

    if p + 2 <= k_packed && act_idx + 4 <= k {
        let f = u32x2_to_f32x4_f16c(weights_u32[row_offset + p], weights_u32[row_offset + p + 1]);
        let a = _mm_loadu_ps(activation.as_ptr().add(act_idx));
        let prod = _mm_mul_ps(f, a);
        let shuf = _mm_shuffle_ps(prod, prod, 0x0E);
        let sums = _mm_add_ps(prod, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
        total += _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        p += 2;
        act_idx += 4;
    }

    if p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        let f = u32x2_to_f32x4_f16c(w, 0);
        let arr: [f32; 4] = std::mem::transmute(f);
        total += arr[0] * activation[act_idx];
        if act_idx + 1 < k {
            total += arr[1] * activation[act_idx + 1];
        }
    }

    total
}

// ============================================================
// F32x1: direct f32 loads (no unpack needed)
// ============================================================

#[allow(unused_variables)]
fn gemv_f32x1_dispatch(weights: &PackedTensor<F32x1>, activation: &[f32], output: &mut [f32]) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let weights_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(
            weights.as_u32().as_ptr() as *const f32,
            weights.packed_len(),
        )
    };

    for row in 0..m {
        let row_data = &weights_f32[row * k..(row + 1) * k];
        let dot = fma_f32_slice(row_data, activation);
        output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
    }
}

// ============================================================
// Generic fallback
// ============================================================

#[inline(always)]
fn gemv_generic_fallback<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(T::ITEMS);

    if k <= K_BLOCK_SIZE {
        gemv_packed_inner::<T>(weights, activation, output, m, k, k_packed);
    } else {
        gemv_packed_blocked::<T>(weights, activation, output, m, k, k_packed);
    }
}

#[inline(always)]
fn gemv_packed_inner<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    _m: usize,
    k: usize,
    k_packed: usize,
) {
    with_scratch(k_packed * T::ITEMS, |unpack_buf| {
        for row in 0.._m {
            let dot = gemv_row::<T>(weights, activation, row, k, k_packed, unpack_buf);
            output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    });
}

#[inline(always)]
fn gemv_packed_blocked<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    k_packed: usize,
) {
    for o in output.iter_mut() {
        *o = 0.0;
    }

    let items = T::ITEMS;

    with_scratch(K_BLOCK_SIZE, |unpack_buf| {
        let mut k_offset = 0;
        while k_offset < k {
            let k_end = (k_offset + K_BLOCK_SIZE).min(k);
            let k_block = k_end - k_offset;

            for row in 0..m {
                let row_offset = row * k_packed;
                let packed_start = k_offset / items;
                let packed_end = k_end.div_ceil(items);
                let unpack_len = (packed_end - packed_start) * items;

                if unpack_len <= K_BLOCK_SIZE {
                    for p in packed_start..packed_end {
                        let word = weights.as_packed()[row_offset + p];
                        let unpacked = word.unpack_to_f32();
                        let base = p * items;
                        for j in 0..items {
                            let idx = base + j;
                            if idx >= k_offset && idx < k_end {
                                unpack_buf[idx - k_offset] = unpacked.as_ref()[j];
                            }
                        }
                    }
                    let acc = fma_f32_slice(
                        &unpack_buf[..k_block.min(unpack_len)],
                        &activation[k_offset..k_end],
                    );
                    output[row] += acc;
                }
            }

            k_offset += K_BLOCK_SIZE;
        }

        for row in 0..m {
            output[row] = output[row] * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    });
}

#[inline]
fn gemv_row<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    row: usize,
    k: usize,
    k_packed: usize,
    unpack_buf: &mut [f32],
) -> f32 {
    let items = T::ITEMS;
    let row_offset = row * k_packed;

    for p in 0..k_packed {
        let word = weights.as_packed()[row_offset + p];
        let unpacked = word.unpack_to_f32();
        let base = p * items;
        for j in 0..items {
            let idx = base + j;
            if idx < k {
                unpack_buf[idx] = unpacked.as_ref()[j];
            }
        }
    }

    fma_f32_slice(&unpack_buf[..k], activation)
}

// ============================================================
// BLIS-style tiled GEMV (cache-blocked K, register-blocked M)
// ============================================================

thread_local! {
    static BLAS_SCRATCH: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

fn with_blas_scratch<R>(f: impl FnOnce(&mut [f32]) -> R) -> R {
    BLAS_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.resize(KC * MR, 0.0);
        f(&mut buf)
    })
}

const MR: usize = 4;
const KC: usize = 8192;

#[inline(always)]
pub fn gemv_packed_tiled<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(T::ITEMS);

    for o in output.iter_mut() {
        *o = 0.0;
    }

    with_blas_scratch(|row_bufs| {
        let mut k_offset = 0;
        while k_offset < k {
            let k_end = (k_offset + KC).min(k);
            let k_block = k_end - k_offset;

            let mut row = 0;
            while row + MR <= m {
                micro_kernel::<T>(
                    weights,
                    &activation[k_offset..k_end],
                    &mut output[row..row + MR],
                    row,
                    k_offset,
                    k_end,
                    k_packed,
                    k_block,
                    row_bufs,
                );
                row += MR;
            }

            while row < m {
                let mut acc = 0.0f32;
                let row_offset = row * k_packed;
                let packed_start = k_offset / T::ITEMS;
                let packed_end = k_end.div_ceil(T::ITEMS);
                for p in packed_start..packed_end {
                    let word = weights.as_packed()[row_offset + p];
                    let unpacked = word.unpack_to_f32();
                    let base = p * T::ITEMS;
                    for j in 0..T::ITEMS {
                        let idx = base + j;
                        if idx >= k_offset && idx < k_end {
                            acc += unpacked.as_ref()[j] * activation[idx];
                        }
                    }
                }
                output[row] += acc;
                row += 1;
            }

            k_offset += KC;
        }

        for row in 0..m {
            output[row] = output[row] * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    });
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn micro_kernel<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    start_row: usize,
    k_start: usize,
    k_end: usize,
    k_packed: usize,
    k_block: usize,
    row_bufs: &mut [f32],
) {
    debug_assert_eq!(output.len(), MR);
    debug_assert!(row_bufs.len() >= KC * MR);

    for r in 0..MR {
        let row_start = r * KC;
        if k_block < KC {
            row_bufs[row_start + k_block..row_start + KC].fill(0.0);
        }
        let row = start_row + r;
        let row_offset = row * k_packed;
        let packed_start = k_start / T::ITEMS;
        let packed_end = k_end.div_ceil(T::ITEMS);

        for p in packed_start..packed_end {
            let word = weights.as_packed()[row_offset + p];
            let unpacked = word.unpack_to_f32();
            let base = p * T::ITEMS;
            for j in 0..T::ITEMS {
                let idx = base + j;
                if idx >= k_start && idx < k_end {
                    row_bufs[row_start + (idx - k_start)] = unpacked.as_ref()[j];
                }
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                micro_kernel_avx2(row_bufs, activation, output, k_block);
            }
            return;
        }
    }

    for r in 0..MR {
        let row_start = r * KC;
        let mut acc = 0.0f32;
        for kk in 0..k_block {
            acc += row_bufs[row_start + kk] * activation[kk];
        }
        output[r] += acc;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn micro_kernel_avx2(row_bufs: &[f32], activation: &[f32], output: &mut [f32], k: usize) {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut kk = 0;

    while kk + 8 <= k {
        if kk + 32 < k {
            _mm_prefetch(activation.as_ptr().add(kk + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(row_bufs.as_ptr().add(kk + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(
                row_bufs.as_ptr().add(KC + kk + 32) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                row_bufs.as_ptr().add(2 * KC + kk + 32) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                row_bufs.as_ptr().add(3 * KC + kk + 32) as *const i8,
                _MM_HINT_T0,
            );
        }

        let act = _mm256_loadu_ps(activation.as_ptr().add(kk));

        let row0 = _mm256_loadu_ps(row_bufs.as_ptr().add(kk));
        acc0 = _mm256_fmadd_ps(row0, act, acc0);

        let row1 = _mm256_loadu_ps(row_bufs.as_ptr().add(KC + kk));
        acc1 = _mm256_fmadd_ps(row1, act, acc1);

        let row2 = _mm256_loadu_ps(row_bufs.as_ptr().add(2 * KC + kk));
        acc2 = _mm256_fmadd_ps(row2, act, acc2);

        let row3 = _mm256_loadu_ps(row_bufs.as_ptr().add(3 * KC + kk));
        acc3 = _mm256_fmadd_ps(row3, act, acc3);

        kk += 8;
    }

    let mut tail0 = 0.0f32;
    let mut tail1 = 0.0f32;
    let mut tail2 = 0.0f32;
    let mut tail3 = 0.0f32;
    while kk < k {
        let act = *activation.as_ptr().add(kk);
        tail0 += *row_bufs.as_ptr().add(kk) * act;
        tail1 += *row_bufs.as_ptr().add(KC + kk) * act;
        tail2 += *row_bufs.as_ptr().add(2 * KC + kk) * act;
        tail3 += *row_bufs.as_ptr().add(3 * KC + kk) * act;
        kk += 1;
    }

    *output.as_mut_ptr().add(0) += hsum256_ps(acc0) + tail0;
    *output.as_mut_ptr().add(1) += hsum256_ps(acc1) + tail1;
    *output.as_mut_ptr().add(2) += hsum256_ps(acc2) + tail2;
    *output.as_mut_ptr().add(3) += hsum256_ps(acc3) + tail3;
}

// ============================================================
// SIMD unpack helpers
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn unpack_u8x4_sse4(w: u32, dst: *mut f32) {
    let wvec = _mm_cvtsi32_si128(w as i32);
    let i32_vals = _mm_cvtepi8_epi32(wvec);
    let f32_vals = _mm_cvtepi32_ps(i32_vals);
    _mm_storeu_ps(dst, f32_vals);
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn unpack_u4x8_wordpair_simd(
    weights_u32: &[u32],
    row_off: usize,
    p: usize,
    row_bufs: &mut [f32],
    row_start: usize,
    k_offset: usize,
) {
    let w0 = weights_u32[row_off + p];
    let w1 = weights_u32[row_off + p + 1];

    let shift0 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
    let mask_lo = _mm256_set1_epi32(0xF);
    let sign_bit = _mm256_set1_epi32(0x8);

    let w0v = _mm256_set1_epi32(w0 as i32);
    let w1v = _mm256_set1_epi32(w1 as i32);

    let nib0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift0), mask_lo);
    let nib1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift0), mask_lo);

    let signed0 = _mm256_sub_epi32(_mm256_xor_si256(nib0, sign_bit), sign_bit);
    let signed1 = _mm256_sub_epi32(_mm256_xor_si256(nib1, sign_bit), sign_bit);

    let f0 = _mm256_cvtepi32_ps(signed0);
    let f1 = _mm256_cvtepi32_ps(signed1);

    let base = p * 8;
    _mm256_storeu_ps(row_bufs.as_mut_ptr().add(row_start + (base - k_offset)), f0);
    _mm256_storeu_ps(
        row_bufs.as_mut_ptr().add(row_start + (base + 8 - k_offset)),
        f1,
    );
}

// ============================================================
// Top-level CPU GEMV dispatcher
// ============================================================

const TILED_K_THRESHOLD: usize = 4096;

#[inline(always)]
pub fn gemv_cpu<T: PackedWord>(weights: &PackedTensor<T>, activation: &[f32], output: &mut [f32]) {
    let k = weights.shape()[1];
    if k > TILED_K_THRESHOLD {
        gemv_packed_tiled(weights, activation, output);
    } else {
        gemv_packed_simd(weights, activation, output);
    }
}

// ============================================================
// Top-level CPU GEMM (batch of GEMV calls)
// ============================================================

#[inline(always)]
pub fn gemm_cpu<T: PackedWord>(
    weights: &PackedTensor<T>,
    batch_inputs: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
) {
    assert_eq!(batch_inputs.len(), outputs.len());
    gemm_packed_batched(weights, batch_inputs, outputs);
}

#[inline(always)]
pub fn gemm_packed_batched<T: PackedWord>(
    weights: &PackedTensor<T>,
    batch_inputs: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
) {
    let n = batch_inputs.len();
    assert_eq!(n, outputs.len());
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(T::ITEMS);

    for o in outputs.iter() {
        assert!(o.len() >= m, "output vector length {} < m={}", o.len(), m);
    }

    with_scratch(k, |unpack_buf| {
        for row in 0..m {
            let scale = weights.scale_for_row(row);
            let zero = weights.zero_for_row(row);
            let row_offset = row * k_packed;
            for p in 0..k_packed {
                let word = weights.as_packed()[row_offset + p];
                let unpacked = word.unpack_to_f32();
                let base = p * T::ITEMS;
                for j in 0..T::ITEMS {
                    let idx = base + j;
                    if idx < k {
                        unpack_buf[idx] = unpacked.as_ref()[j];
                    }
                }
            }
            for (bi, input) in batch_inputs.iter().enumerate() {
                let acc = fma_f32_slice(unpack_buf, input);
                outputs[bi][row] = acc * scale + zero;
            }
        }
    });
}

// ============================================================
// Flat-buffer GEMM (avoids Vec<Vec> allocation storm)
// ============================================================

#[inline(always)]
pub fn gemm_cpu_flat<T: PackedWord>(
    weights: &PackedTensor<T>,
    batch_inputs: &[f32],
    outputs: &mut [f32],
    num_pixels: usize,
    col_w: usize,
    oc: usize,
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(T::ITEMS);

    with_scratch(k, |unpack_buf| {
        for row in 0..m {
            let scale = weights.scale_for_row(row);
            let zero = weights.zero_for_row(row);
            let row_offset = row * k_packed;
            for p in 0..k_packed {
                let word = weights.as_packed()[row_offset + p];
                let unpacked = word.unpack_to_f32();
                let base = p * T::ITEMS;
                for j in 0..T::ITEMS {
                    let idx = base + j;
                    if idx < k {
                        unpack_buf[idx] = unpacked.as_ref()[j];
                    }
                }
            }
            for bi in 0..num_pixels {
                let input = &batch_inputs[bi * col_w..(bi + 1) * col_w];
                let acc = fma_f32_slice(unpack_buf, input);
                outputs[bi * oc + row] = acc * scale + zero;
            }
        }
    });
}

// ============================================================
// Batch GEMM — K-tiled for cache reuse across batch rows
// ============================================================

#[inline(always)]
pub fn gemm_batch_packed_simd<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    n: usize,
    k: usize,
    m: usize,
) {
    let k_packed = k.div_ceil(T::ITEMS);

    for o in output.iter_mut() {
        *o = 0.0;
    }

    let mut k_start = 0;
    while k_start < k {
        let k_end = (k_start + K_BLOCK_SIZE).min(k);
        let k_len = k_end - k_start;
        let packed_start = k_start / T::ITEMS;
        let packed_end = k_end.div_ceil(T::ITEMS);

        with_scratch(k_len, |unpack_buf| {
            for row in 0..m {
                let row_offset = row * k_packed;

                for p in packed_start..packed_end {
                    let word = weights.as_packed()[row_offset + p];
                    let unpacked = word.unpack_to_f32();
                    let base = p * T::ITEMS;
                    for j in 0..T::ITEMS {
                        let idx = base + j;
                        if idx >= k_start && idx < k_end {
                            unpack_buf[idx - k_start] = unpacked.as_ref()[j];
                        }
                    }
                }

                for bi in 0..n {
                    let act_slice = &activation[bi * k + k_start..bi * k + k_end];
                    let acc = fma_f32_slice(unpack_buf, act_slice);
                    output[bi * m + row] += acc;
                }
            }
        });

        k_start += K_BLOCK_SIZE;
    }

    for bi in 0..n {
        for row in 0..m {
            let scale = weights.scale_for_row(row);
            let zero = weights.zero_for_row(row);
            output[bi * m + row] = output[bi * m + row] * scale + zero;
        }
    }
}

// ============================================================
// SWAR ReLU kernels (AVX2)
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_swar_simd_u8x4(raw: &mut [u32]) {
    let len = raw.len();
    let mut i = 0;
    let zero = _mm256_setzero_si256();
    let ptr = raw.as_mut_ptr();

    while i + 8 <= len {
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let neg_mask = _mm256_cmpgt_epi8(zero, v);
        let result = _mm256_andnot_si256(neg_mask, v);
        _mm256_storeu_si256(ptr.add(i) as *mut __m256i, result);
        i += 8;
    }

    while i < len {
        raw[i] = crate::swar::ops_8bit::swar_relu_s8x4(raw[i]);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_swar_simd_f32(raw: &mut [u32]) {
    let len = raw.len();
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    let ptr = raw.as_mut_ptr() as *mut f32;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        let result = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(ptr.add(i), result);
        i += 8;
    }

    while i < len {
        let v = f32::from_bits(raw[i]);
        if v < 0.0 {
            raw[i] = 0;
        }
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_backward_swar_simd_u8x4(grad_raw: &mut [u32], pre_raw: &[u32]) {
    let len = grad_raw.len().min(pre_raw.len());
    let mut i = 0;
    let zero = _mm256_setzero_si256();
    let g_ptr = grad_raw.as_mut_ptr();
    let p_ptr = pre_raw.as_ptr();

    while i + 8 <= len {
        let g = _mm256_loadu_si256(g_ptr.add(i) as *const __m256i);
        let p = _mm256_loadu_si256(p_ptr.add(i) as *const __m256i);
        let mask = _mm256_cmpgt_epi8(p, zero);
        let result = _mm256_and_si256(g, mask);
        _mm256_storeu_si256(g_ptr.add(i) as *mut __m256i, result);
        i += 8;
    }

    while i < len {
        grad_raw[i] = crate::swar::ops_8bit::swar_relu_backward_u8x4(grad_raw[i], pre_raw[i]);
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_backward_swar_simd_f32(grad_raw: &mut [u32], pre_raw: &[u32]) {
    let len = grad_raw.len().min(pre_raw.len());
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    let g_ptr = grad_raw.as_mut_ptr() as *mut f32;
    let p_ptr = pre_raw.as_ptr() as *const f32;

    while i + 8 <= len {
        let g = _mm256_loadu_ps(g_ptr.add(i));
        let p = _mm256_loadu_ps(p_ptr.add(i));
        let mask = _mm256_cmp_ps(p, zero, _CMP_GT_OQ);
        let result = _mm256_and_ps(g, mask);
        _mm256_storeu_ps(g_ptr.add(i), result);
        i += 8;
    }

    while i < len {
        let pre_val = f32::from_bits(pre_raw[i]);
        if pre_val <= 0.0 {
            grad_raw[i] = 0;
        }
        i += 1;
    }
}

// ============================================================
// Matmul — cache-blocked row-wise microkernel
// ============================================================

pub const MIN_BLAS_SIZE: usize = 64;

pub fn matmul_blas(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    matmul_blas_with_transpose(a, b, m, k, n, false, false)
}

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

pub fn matmul_blas_into(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    matmul_blas_with_transpose_into(a, b, out, m, k, n, false, false)
}

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

#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn blocked_row_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    row: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch_stride: usize,
    a_stride_0: usize,
    a_stride_1: usize,
    b_batch_stride: usize,
    b_stride_0: usize,
    b_stride_1: usize,
) {
    const TILE_K: usize = 64;
    const TILE_N: usize = 4;

    let bat = row / m;
    let i = row % m;
    let a_off = bat * a_batch_stride + i * a_stride_0;
    let b_off = bat * b_batch_stride;
    let out_off = row * n;

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let use_simd = a_stride_1 == 1 && b_stride_1 == 1 && n >= 8;

    for j in 0..n {
        *out_ptr.add(out_off + j) = 0.0;
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if use_simd {
        use std::arch::x86_64::*;

        const TILE_N_SIMD: usize = 8;
        let mut ko = 0;
        while ko < k {
            let kend = if ko + TILE_K < k { ko + TILE_K } else { k };

            let mut jo = 0;
            while jo + TILE_N_SIMD <= n {
                unsafe {
                    let mut acc = _mm256_setzero_ps();

                    let mut kk = ko;
                    while kk + 4 <= kend {
                        // Prefetch next tile's data — 32 elements (128 bytes) ahead
                        if kk + 32 < k {
                            _mm_prefetch(
                                a_ptr.add(a_off + (kk + 32) * a_stride_1) as *const i8,
                                _MM_HINT_T0,
                            );
                            _mm_prefetch(
                                b_ptr.add(b_off + (kk + 32) * b_stride_0 + jo) as *const i8,
                                _MM_HINT_T0,
                            );
                        }

                        let a0 = _mm256_set1_ps(*a_ptr.add(a_off + kk * a_stride_1));
                        let b0 = _mm256_loadu_ps(b_ptr.add(b_off + kk * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a0, b0, acc);

                        let a1 = _mm256_set1_ps(*a_ptr.add(a_off + (kk + 1) * a_stride_1));
                        let b1 = _mm256_loadu_ps(b_ptr.add(b_off + (kk + 1) * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a1, b1, acc);

                        let a2 = _mm256_set1_ps(*a_ptr.add(a_off + (kk + 2) * a_stride_1));
                        let b2 = _mm256_loadu_ps(b_ptr.add(b_off + (kk + 2) * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a2, b2, acc);

                        let a3 = _mm256_set1_ps(*a_ptr.add(a_off + (kk + 3) * a_stride_1));
                        let b3 = _mm256_loadu_ps(b_ptr.add(b_off + (kk + 3) * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a3, b3, acc);

                        kk += 4;
                    }
                    while kk < kend {
                        let av = _mm256_set1_ps(*a_ptr.add(a_off + kk * a_stride_1));
                        let bv = _mm256_loadu_ps(b_ptr.add(b_off + kk * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(av, bv, acc);
                        kk += 1;
                    }

                    let out_v = _mm256_loadu_ps(out_ptr.add(out_off + jo));
                    _mm256_storeu_ps(out_ptr.add(out_off + jo), _mm256_add_ps(out_v, acc));
                }
                jo += TILE_N_SIMD;
            }

            while jo < n {
                let mut sum = 0.0f32;
                for kk in ko..kend {
                    sum += *a_ptr.add(a_off + kk * a_stride_1)
                        * b_ptr.add(b_off + kk * b_stride_0 + jo).read();
                }
                let p = out_ptr.add(out_off + jo);
                *p += sum;
                jo += 1;
            }

            ko += TILE_K;
        }
        return;
    }

    let mut ko = 0;
    while ko < k {
        let kend = if ko + TILE_K < k { ko + TILE_K } else { k };

        let mut jo = 0;
        while jo + TILE_N <= n {
            let mut acc = [0.0f32; TILE_N];

            let mut kk = ko;
            while kk < kend {
                let av = *a_ptr.add(a_off + kk * a_stride_1);
                for t in 0..TILE_N {
                    acc[t] += av * *b_ptr.add(b_off + kk * b_stride_0 + (jo + t) * b_stride_1);
                }
                kk += 1;
            }

            for t in 0..TILE_N {
                let p = out_ptr.add(out_off + jo + t);
                *p += acc[t];
            }
            jo += TILE_N;
        }

        while jo < n {
            let mut sum = 0.0f32;
            for kk in ko..kend {
                sum += *a_ptr.add(a_off + kk * a_stride_1)
                    * b_ptr.add(b_off + kk * b_stride_0 + jo * b_stride_1).read();
            }
            let p = out_ptr.add(out_off + jo);
            *p += sum;
            jo += 1;
        }

        ko += TILE_K;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn small_matrix_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    batch: usize,
    m: i32,
    n: i32,
    k: i32,
    a_batch_stride: i64,
    a_stride_0: i64,
    a_stride_1: i64,
    b_batch_stride: i64,
    b_stride_0: i64,
    b_stride_1: i64,
) {
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let total_rows = batch * m_usize;

    for row in 0..total_rows {
        blocked_row_matmul(
            a_ptr,
            b_ptr,
            out_ptr,
            row,
            m_usize,
            n_usize,
            k_usize,
            a_batch_stride as usize,
            a_stride_0 as usize,
            a_stride_1 as usize,
            b_batch_stride as usize,
            b_stride_0 as usize,
            b_stride_1 as usize,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn single_threaded_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    batch: usize,
    m: i32,
    n: i32,
    k: i32,
    a_batch_stride: i64,
    a_stride_0: i64,
    a_stride_1: i64,
    b_batch_stride: i64,
    b_stride_0: i64,
    b_stride_1: i64,
) {
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let total_rows = batch * m_usize;

    for row in 0..total_rows {
        blocked_row_matmul(
            a_ptr,
            b_ptr,
            out_ptr,
            row,
            m_usize,
            n_usize,
            k_usize,
            a_batch_stride as usize,
            a_stride_0 as usize,
            a_stride_1 as usize,
            b_batch_stride as usize,
            b_stride_0 as usize,
            b_stride_1 as usize,
        );
    }
}

// ============================================================
// SIMD dot product
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub unsafe fn simd_dot_product(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;

    if is_x86_feature_detected!("avx512f") {
        unsafe {
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();

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

            let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
            sum += _mm512_reduce_add_ps(acc);

            while i + 16 <= len {
                let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
                acc0 = _mm512_fmadd_ps(a_vec, b_vec, _mm512_setzero_ps());
                sum += _mm512_reduce_add_ps(acc0);
                i += 16;
            }
        }
    } else if is_x86_feature_detected!("avx2") {
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

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

            let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            sum += hsum256_ps(acc);

            while i + 8 <= len {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                sum += hsum256_ps(_mm256_mul_ps(a_vec, b_vec));
                i += 8;
            }
        }
    }

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
#[inline]
pub unsafe fn simd_dot_product(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + 4 <= len {
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

// ============================================================
// Conv2d im2col kernel
// ============================================================

thread_local! {
    static CONV_SCRATCH: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn conv2d_im2col(
    x_ptr: *const f32,
    w_data: &[f32],
    bias_data: Option<&[f32]>,
    out_ptr: *mut f32,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    in_height: usize,
    in_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) {
    let out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    let out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    let col_rows = batch_size * out_height * out_width;
    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;
    let cols_per_group = in_channels_per_group * kernel_height * kernel_width;
    let col_size = col_rows * cols_per_group;

    for group in 0..groups {
        let in_ch_start = group * in_channels_per_group;
        let in_ch_end = in_ch_start + in_channels_per_group;

        CONV_SCRATCH.with(|scratch| {
            let mut buf = scratch.borrow_mut();
            if buf.len() < col_size {
                buf.resize(col_size, 0.0f32);
            }
            buf[..col_size].fill(0.0);

            let col_data: &mut [f32] = &mut buf[..col_size];

            for n in 0..batch_size {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let col_row = (n * out_height + oh) * out_width + ow;
                        let col_chunk =
                            &mut col_data[col_row * cols_per_group..(col_row + 1) * cols_per_group];

                        let fast_path = stride == 1 && dilation == 1;

                        for ic_idx in in_ch_start..in_ch_end {
                            let ic = ic_idx;
                            let col_base = (ic_idx - in_ch_start) * kernel_height * kernel_width;

                            if fast_path {
                                for kh in 0..kernel_height {
                                    let ih = oh + kh;
                                    if ih >= padding && ih < padding + in_height {
                                        let x_ih = ih - padding;
                                        let x_row_base =
                                            (n * in_channels + ic) * in_height * in_width
                                                + x_ih * in_width;

                                        let iw_start = ow;
                                        if iw_start >= padding
                                            && iw_start + kernel_width <= padding + in_width
                                        {
                                            let x_iw_start = iw_start - padding;
                                            let x_src = x_row_base + x_iw_start;
                                            let col_dst_base = col_base + kh * kernel_width;

                                            std::ptr::copy_nonoverlapping(
                                                x_ptr.add(x_src),
                                                col_chunk.as_mut_ptr().add(col_dst_base),
                                                kernel_width,
                                            );
                                        } else {
                                            for kw in 0..kernel_width {
                                                let iw = ow + kw;
                                                if iw >= padding && iw < padding + in_width {
                                                    let x_iw = iw - padding;
                                                    let col_col = col_base + kh * kernel_width + kw;
                                                    col_chunk[col_col] =
                                                        *x_ptr.add(x_row_base + x_iw);
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                for kh in 0..kernel_height {
                                    let ih = (oh * stride + kh * dilation).wrapping_sub(padding);
                                    if ih < in_height {
                                        let x_row_base =
                                            (n * in_channels + ic) * in_height * in_width
                                                + ih * in_width;
                                        for kw in 0..kernel_width {
                                            let iw =
                                                (ow * stride + kw * dilation).wrapping_sub(padding);
                                            if iw < in_width {
                                                let col_col = col_base + kh * kernel_width + kw;
                                                col_chunk[col_col] = *x_ptr.add(x_row_base + iw);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let col_slice: &[f32] = col_data;
            let oc_start = group * out_channels_per_group;
            let w_group_start = oc_start * cols_per_group;

            let mut group_out = vec![0.0f32; col_rows * out_channels_per_group];

            matmul_blas_with_transpose_into(
                &col_slice[..col_rows * cols_per_group],
                &w_data[w_group_start..],
                &mut group_out,
                col_rows,
                cols_per_group,
                out_channels_per_group,
                false,
                true,
            );

            let spatial = out_height * out_width;
            let block_rows = 64;

            if let Some(bias) = bias_data {
                let bias_ptr = bias.as_ptr();
                for n in 0..batch_size {
                    for sp_block in (0..spatial).step_by(block_rows) {
                        let blk = std::cmp::min(block_rows, spatial - sp_block);
                        for oc_idx_in_group in 0..out_channels_per_group {
                            let oc_idx = oc_start + oc_idx_in_group;
                            let bias_val = *bias_ptr.add(oc_idx);
                            for i in 0..blk {
                                let row = n * spatial + sp_block + i;
                                let out_idx =
                                    ((n * out_channels + oc_idx) * spatial + sp_block) + i;
                                let val = group_out[row * out_channels_per_group + oc_idx_in_group];
                                *out_ptr.add(out_idx) = val + bias_val;
                            }
                        }
                    }
                }
            } else {
                for n in 0..batch_size {
                    for sp_block in (0..spatial).step_by(block_rows) {
                        let blk = std::cmp::min(block_rows, spatial - sp_block);
                        for oc_idx_in_group in 0..out_channels_per_group {
                            let oc_idx = oc_start + oc_idx_in_group;
                            for i in 0..blk {
                                let row = n * spatial + sp_block + i;
                                let out_idx =
                                    ((n * out_channels + oc_idx) * spatial + sp_block) + i;
                                let val = group_out[row * out_channels_per_group + oc_idx_in_group];
                                *out_ptr.add(out_idx) = val;
                            }
                        }
                    }
                }
            }
        });
    }
}

// ============================================================
// Conv2d f32 tiled microkernel (cache-blocked SIMD-friendly)
// ============================================================

/// Tiled f32 conv2d with OC×2×2 register blocking.
///
/// Tiles over output channels (OC_TILE=4) and output positions (2×2)
/// to keep filter weights and partial sums in registers. The inner
/// (cc, kh, kw) loop broadcasts the input value and does 4 FMAs,
/// which the compiler auto-vectorizes.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32_tiled(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) {
    let c_per_group = c / groups.max(1);
    let f_per_group = f / groups.max(1);
    let h_out =
        (h + 2 * padding).saturating_sub(dilation * (kh.saturating_sub(1)) + 1) / stride + 1;
    let w_out =
        (w + 2 * padding).saturating_sub(dilation * (kw.saturating_sub(1)) + 1) / stride + 1;

    const OC_TILE: usize = 4;

    for nn in 0..n {
        for g in 0..groups {
            let ff_base = g * f_per_group;
            let input_group_off = g * c_per_group * (h * w);

            for ff_tile in (0..f_per_group).step_by(OC_TILE) {
                let ff_abs = ff_base + ff_tile;
                let oc_end = (ff_tile + OC_TILE).min(f_per_group);
                let oc_tile = oc_end - ff_tile;

                let weight_off = ff_abs * c_per_group * kh * kw;

                for hh in (0..h_out).step_by(2) {
                    for ww in (0..w_out).step_by(2) {
                        let mut acc = [[0.0f32; 4]; 4];

                        for cc in 0..c_per_group {
                            let input_ch_off = nn * (c * h * w) + input_group_off + cc * (h * w);
                            let weight_ch_off = weight_off + cc * kh * kw;

                            for kkh in 0..kh {
                                for kkw in 0..kw {
                                    let weight_val_base = weight_ch_off + kkh * kw + kkw;

                                    for pos_h in 0..2 {
                                        let oh = hh + pos_h;
                                        if oh >= h_out {
                                            continue;
                                        }
                                        for pos_w in 0..2 {
                                            let ow = ww + pos_w;
                                            if ow >= w_out {
                                                continue;
                                            }

                                            let h_in = oh * stride + kkh * dilation;
                                            let w_in = ow * stride + kkw * dilation;
                                            if h_in < padding || w_in < padding {
                                                continue;
                                            }
                                            let h_in_s = h_in - padding;
                                            let w_in_s = w_in - padding;
                                            if h_in_s >= h || w_in_s >= w {
                                                continue;
                                            }

                                            let input_val =
                                                input[input_ch_off + h_in_s * w + w_in_s];
                                            let pos = pos_h * 2 + pos_w;

                                            for oc in 0..oc_tile {
                                                acc[pos][oc] += input_val
                                                    * weight[weight_val_base
                                                        + oc * c_per_group * kh * kw];
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        for pos_h in 0..2 {
                            let oh = hh + pos_h;
                            if oh >= h_out {
                                continue;
                            }
                            for pos_w in 0..2 {
                                let ow = ww + pos_w;
                                if ow >= w_out {
                                    continue;
                                }
                                let pos = pos_h * 2 + pos_w;
                                let out_base = nn * (f * h_out * w_out) + (oh * w_out + ow);
                                for oc in 0..oc_tile {
                                    let mut val = acc[pos][oc];
                                    if !bias.is_empty() {
                                        let b = (ff_abs + oc) % bias.len();
                                        val += bias[b];
                                    }
                                    output[out_base + (ff_abs + oc) * (h_out * w_out)] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================
// BatchNorm inference — scalar + AVX2
// ============================================================

/// BatchNorm inference: out[i] = data[i] * scale[ch] + shift[ch] where
///   scale[ch] = weight[ch] / sqrt(var[ch] + eps)
///   shift[ch] = bias[ch] - mean[ch] * scale[ch]
///   ch = i % c  (matches existing dispatch behavior)
pub fn batch_norm_inference_f32(
    data: &[f32],
    weight: &[f32],
    bias: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let c = weight.len();
    let len = output.len().min(data.len());
    // Pre-compute per-channel scale and shift
    for ch in 0..c {
        let scale = weight[ch] / (running_var[ch] + eps).sqrt();
        let shift = bias[ch] - running_mean[ch] * scale;
        // Process all positions where i % c == ch
        let mut i = ch;
        while i < len {
            output[i] = data[i] * scale + shift;
            i += c;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn batch_norm_inference_f32_avx2(
    data: &[f32],
    weight: &[f32],
    bias: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let c = weight.len();
    let len = output.len().min(data.len());
    // Pre-compute per-channel scale and shift
    let mut scale = vec![0.0f32; c];
    let mut shift = vec![0.0f32; c];
    for ch in 0..c {
        scale[ch] = weight[ch] / (running_var[ch] + eps).sqrt();
        shift[ch] = bias[ch] - running_mean[ch] * scale[ch];
    }

    let vscale_base = scale.as_ptr();
    let vshift_base = shift.as_ptr();
    let vc = _mm256_set1_ps(c as f32);
    let vinc = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);

    let mut i = 0usize;
    while i + 8 <= len {
        // Channel indices ch_j = (i + j) % c  via float:  idx - floor(idx/c)*c
        let vi = _mm256_set1_ps(i as f32);
        let vindices_f = _mm256_add_ps(vi, vinc);
        let vdiv = _mm256_round_ps(
            _mm256_div_ps(vindices_f, vc),
            _MM_FROUND_TO_NEG_INF,
        );
        let vch_f = _mm256_sub_ps(vindices_f, _mm256_mul_ps(vdiv, vc));
        let vch = _mm256_cvttps_epi32(vch_f);

        // Gather scale[ch] and shift[ch]
        let vscale = _mm256_i32gather_ps(vscale_base, vch, 4);
        let vshift = _mm256_i32gather_ps(vshift_base, vch, 4);

        // out[i] = data[i] * scale + shift
        let vdata = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_fmadd_ps(vdata, vscale, vshift));
        i += 8;
    }
    for j in i..len {
        let ch = j % c;
        output[j] = data[j] * scale[ch] + shift[ch];
    }
}

// ============================================================
// Pooling microkernels — MaxPool2d + AvgPool2d (scalar + AVX2)
// ============================================================

#[inline]
pub fn pool_max_f32_scalar(
    input: &[f32], output: &mut [f32],
    n: usize, c: usize, h: usize, w: usize,
    kernel: usize, stride_val: usize, padding_val: usize,
    h_out: usize, w_out: usize,
    mut indices_out: Option<&mut [i64]>,
) {
    let hw_out = h_out * w_out;
    for nn in 0..n {
        for cc in 0..c {
            for hh in 0..h_out {
                for ww in 0..w_out {
                    let mut val = f32::NEG_INFINITY;
                    let mut best_kh = 0usize;
                    let mut best_kw = 0usize;
                    for kh in 0..kernel {
                        for kw in 0..kernel {
                            let h_in = hh * stride_val + kh;
                            let w_in = ww * stride_val + kw;
                            if h_in >= padding_val && w_in >= padding_val {
                                let h_in_s = h_in - padding_val;
                                let w_in_s = w_in - padding_val;
                                if h_in_s < h && w_in_s < w {
                                    let idx = nn * (c * h * w) + cc * (h * w) + h_in_s * w + w_in_s;
                                    if idx < input.len() {
                                        if input[idx] > val {
                                            val = input[idx];
                                            best_kh = kh;
                                            best_kw = kw;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let out_idx = nn * (c * hw_out) + cc * hw_out + hh * w_out + ww;
                    if out_idx < output.len() {
                        output[out_idx] = val;
                    }
                    if let Some(ref mut idx_out) = indices_out {
                        if out_idx < idx_out.len() {
                            idx_out[out_idx] = (best_kh * kernel + best_kw) as i64;
                        }
                    }
                }
            }
        }
    }
}

#[inline]
pub fn pool_avg_f32_scalar(
    input: &[f32], output: &mut [f32],
    n: usize, c: usize, h: usize, w: usize,
    kernel: usize, stride_val: usize, padding_val: usize,
    h_out: usize, w_out: usize,
) {
    let hw_out = h_out * w_out;
    for nn in 0..n {
        for cc in 0..c {
            for hh in 0..h_out {
                for ww in 0..w_out {
                    let mut val = 0.0f32;
                    let mut count = 0usize;
                    for kh in 0..kernel {
                        for kw in 0..kernel {
                            let h_in = hh * stride_val + kh;
                            let w_in = ww * stride_val + kw;
                            if h_in >= padding_val && w_in >= padding_val {
                                let h_in_s = h_in - padding_val;
                                let w_in_s = w_in - padding_val;
                                if h_in_s < h && w_in_s < w {
                                    let idx = nn * (c * h * w) + cc * (h * w) + h_in_s * w + w_in_s;
                                    if idx < input.len() {
                                        val += input[idx];
                                        count += 1;
                                    }
                                }
                            }
                        }
                    }
                    if count > 0 {
                        val /= count as f32;
                    }
                    let out_idx = nn * (c * hw_out) + cc * hw_out + hh * w_out + ww;
                    if out_idx < output.len() {
                        output[out_idx] = val;
                    }
                }
            }
        }
    }
}

#[inline]
pub fn adaptive_avg_pool2d_f32_scalar(
    input: &[f32], output: &mut [f32],
    nc: usize, h: usize, w: usize,
    out_h: usize, out_w: usize,
) {
    let hw = h * w;
    for nci in 0..nc {
        for ohi in 0..out_h {
            for owi in 0..out_w {
                let h_start = ohi * h / out_h;
                let h_end = (ohi + 1) * h / out_h;
                let w_start = owi * w / out_w;
                let w_end = (owi + 1) * w / out_w;
                let mut sum = 0.0f32;
                let mut count = 0;
                for hi in h_start..h_end {
                    for wi in w_start..w_end {
                        sum += input[nci * hw + hi * w + wi];
                        count += 1;
                    }
                }
                let out_idx = nci * out_h * out_w + ohi * out_w + owi;
                if out_idx < output.len() {
                    output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn pool_max_f32_avx2(
    input: &[f32], output: &mut [f32],
    n: usize, c: usize, h: usize, w: usize,
    kernel: usize, stride_val: usize, padding_val: usize,
    h_out: usize, w_out: usize,
    indices_out: Option<&mut [i64]>,
) {
    if stride_val != 1 || indices_out.is_some() {
        return pool_max_f32_scalar(
            input, output, n, c, h, w, kernel, stride_val, padding_val, h_out, w_out, indices_out,
        );
    }
    let hw_out = h_out * w_out;

    // Interior hh where all kh are valid (stride=1): hh in [padding_val, padding_val + h - kernel]
    let interior_h_start = if padding_val < h_out { padding_val } else { h_out };
    let interior_h_end = if padding_val + h >= kernel {
        (padding_val + h - kernel + 1).min(h_out)
    } else {
        interior_h_start
    };

    // Interior ww where 8 consecutive outputs all have valid kw: ww in [padding_val, w+padding_val-kernel-7]
    // exclusive upper bound: min(w_out, w+padding_val-kernel-6)
    let interior_w_start = if padding_val < w_out { padding_val } else { w_out };
    let interior_w_end = if w + padding_val >= kernel + 7 {
        (w + padding_val - kernel - 6).min(w_out)
    } else {
        interior_w_start
    };

    let vneg_inf = _mm256_set1_ps(f32::NEG_INFINITY);

    for nn in 0..n {
        for cc in 0..c {
            let ch_in_off = nn * (c * h * w) + cc * (h * w);
            let ch_out_off = nn * (c * hw_out) + cc * hw_out;

            // Top edge rows (scalar)
            for hh in 0..interior_h_start {
                for ww in 0..w_out {
                    let mut val = f32::NEG_INFINITY;
                    for kh in 0..kernel {
                        let h_in = hh + kh;
                        if h_in >= padding_val && h_in - padding_val < h {
                            let rbase = ch_in_off + (h_in - padding_val) * w;
                            for kw in 0..kernel {
                                let w_in = ww + kw;
                                if w_in >= padding_val && w_in - padding_val < w {
                                    let inp = input[rbase + w_in - padding_val];
                                    if inp > val { val = inp; }
                                }
                            }
                        }
                    }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }
            }

            // Interior rows
            for hh in interior_h_start..interior_h_end {
                // Left edge (scalar)
                for ww in 0..interior_w_start {
                    let mut val = f32::NEG_INFINITY;
                    let h_s = hh - padding_val;
                    for kh in 0..kernel {
                        let rbase = ch_in_off + (h_s + kh) * w;
                        for kw in 0..kernel {
                            let w_in = ww + kw;
                            if w_in >= padding_val && w_in - padding_val < w {
                                let inp = input[rbase + w_in - padding_val];
                                if inp > val { val = inp; }
                            }
                        }
                    }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }

                // Interior W (SIMD: 8 outputs at once)
                let mut ww = interior_w_start;
                while ww + 8 <= interior_w_end {
                    let mut vmax = vneg_inf;
                    let h_s = hh - padding_val;
                    for kh in 0..kernel {
                        let rbase = ch_in_off + (h_s + kh) * w;
                        for kw in 0..kernel {
                            let load_off = rbase + ww + kw - padding_val;
                            let v = _mm256_loadu_ps(input.as_ptr().add(load_off));
                            vmax = _mm256_max_ps(vmax, v);
                        }
                    }
                    _mm256_storeu_ps(output.as_mut_ptr().add(ch_out_off + hh * w_out + ww), vmax);
                    ww += 8;
                }

                // Right edge (scalar)
                for ww in ww..w_out {
                    let mut val = f32::NEG_INFINITY;
                    let h_s = hh - padding_val;
                    for kh in 0..kernel {
                        let rbase = ch_in_off + (h_s + kh) * w;
                        for kw in 0..kernel {
                            let w_in = ww + kw;
                            if w_in >= padding_val && w_in - padding_val < w {
                                let inp = input[rbase + w_in - padding_val];
                                if inp > val { val = inp; }
                            }
                        }
                    }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }
            }

            // Bottom edge rows (scalar)
            for hh in interior_h_end..h_out {
                for ww in 0..w_out {
                    let mut val = f32::NEG_INFINITY;
                    for kh in 0..kernel {
                        let h_in = hh + kh;
                        if h_in >= padding_val && h_in - padding_val < h {
                            let rbase = ch_in_off + (h_in - padding_val) * w;
                            for kw in 0..kernel {
                                let w_in = ww + kw;
                                if w_in >= padding_val && w_in - padding_val < w {
                                    let inp = input[rbase + w_in - padding_val];
                                    if inp > val { val = inp; }
                                }
                            }
                        }
                    }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }
            }
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn pool_avg_f32_avx2(
    input: &[f32], output: &mut [f32],
    n: usize, c: usize, h: usize, w: usize,
    kernel: usize, stride_val: usize, padding_val: usize,
    h_out: usize, w_out: usize,
) {
    if stride_val != 1 {
        return pool_avg_f32_scalar(
            input, output, n, c, h, w, kernel, stride_val, padding_val, h_out, w_out,
        );
    }
    let hw_out = h_out * w_out;
    let kernel_area = (kernel * kernel) as f32;

    let interior_h_start = if padding_val < h_out { padding_val } else { h_out };
    let interior_h_end = if padding_val + h >= kernel {
        (padding_val + h - kernel + 1).min(h_out)
    } else {
        interior_h_start
    };
    let interior_w_start = if padding_val < w_out { padding_val } else { w_out };
    let interior_w_end = if w + padding_val >= kernel + 7 {
        (w + padding_val - kernel - 6).min(w_out)
    } else {
        interior_w_start
    };

    let vkernel_area = _mm256_set1_ps(kernel_area);

    for nn in 0..n {
        for cc in 0..c {
            let ch_in_off = nn * (c * h * w) + cc * (h * w);
            let ch_out_off = nn * (c * hw_out) + cc * hw_out;

            // Top edge rows (scalar)
            for hh in 0..interior_h_start {
                for ww in 0..w_out {
                    let mut val = 0.0f32;
                    let mut count = 0usize;
                    for kh in 0..kernel {
                        let h_in = hh + kh;
                        if h_in >= padding_val && h_in - padding_val < h {
                            let rbase = ch_in_off + (h_in - padding_val) * w;
                            for kw in 0..kernel {
                                let w_in = ww + kw;
                                if w_in >= padding_val && w_in - padding_val < w {
                                    val += input[rbase + w_in - padding_val];
                                    count += 1;
                                }
                            }
                        }
                    }
                    if count > 0 { val /= count as f32; }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }
            }

            // Interior rows
            for hh in interior_h_start..interior_h_end {
                // Left edge (scalar)
                for ww in 0..interior_w_start {
                    let mut val = 0.0f32;
                    let mut count = 0usize;
                    let h_s = hh - padding_val;
                    for kh in 0..kernel {
                        let rbase = ch_in_off + (h_s + kh) * w;
                        for kw in 0..kernel {
                            let w_in = ww + kw;
                            if w_in >= padding_val && w_in - padding_val < w {
                                val += input[rbase + w_in - padding_val];
                                count += 1;
                            }
                        }
                    }
                    if count > 0 { val /= count as f32; }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }

                // Interior W (SIMD)
                let mut ww = interior_w_start;
                while ww + 8 <= interior_w_end {
                    let mut vacc = _mm256_setzero_ps();
                    let h_s = hh - padding_val;
                    for kh in 0..kernel {
                        let rbase = ch_in_off + (h_s + kh) * w;
                        for kw in 0..kernel {
                            let load_off = rbase + ww + kw - padding_val;
                            let v = _mm256_loadu_ps(input.as_ptr().add(load_off));
                            vacc = _mm256_add_ps(vacc, v);
                        }
                    }
                    let vavg = _mm256_div_ps(vacc, vkernel_area);
                    _mm256_storeu_ps(output.as_mut_ptr().add(ch_out_off + hh * w_out + ww), vavg);
                    ww += 8;
                }

                // Right edge (scalar)
                for ww in ww..w_out {
                    let mut val = 0.0f32;
                    let mut count = 0usize;
                    let h_s = hh - padding_val;
                    for kh in 0..kernel {
                        let rbase = ch_in_off + (h_s + kh) * w;
                        for kw in 0..kernel {
                            let w_in = ww + kw;
                            if w_in >= padding_val && w_in - padding_val < w {
                                val += input[rbase + w_in - padding_val];
                                count += 1;
                            }
                        }
                    }
                    if count > 0 { val /= count as f32; }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }
            }

            // Bottom edge rows (scalar)
            for hh in interior_h_end..h_out {
                for ww in 0..w_out {
                    let mut val = 0.0f32;
                    let mut count = 0usize;
                    for kh in 0..kernel {
                        let h_in = hh + kh;
                        if h_in >= padding_val && h_in - padding_val < h {
                            let rbase = ch_in_off + (h_in - padding_val) * w;
                            for kw in 0..kernel {
                                let w_in = ww + kw;
                                if w_in >= padding_val && w_in - padding_val < w {
                                    val += input[rbase + w_in - padding_val];
                                    count += 1;
                                }
                            }
                        }
                    }
                    if count > 0 { val /= count as f32; }
                    *output.as_mut_ptr().add(ch_out_off + hh * w_out + ww) = val;
                }
            }
        }
    }
}

// ============================================================
// Remaining scalar ops — AVX2 microkernels for ops without dedicated kernels
// ============================================================

// ── reduce_sum_f32 — group sum/mean ─────────────────────────

#[inline]
pub fn reduce_sum_f32_scalar(input: &[f32], output: &mut [f32], group_size: usize, is_mean: bool) {
    let num_groups = output.len();
    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(input.len());
        let cnt = (end - start) as f32;
        let mut sum = 0.0f32;
        for i in start..end { sum += input[i]; }
        output[g] = if is_mean { if cnt > 0.0 { sum / cnt } else { 0.0 } } else { sum };
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn reduce_sum_f32_avx2(input: &[f32], output: &mut [f32], group_size: usize, is_mean: bool) {
    let num_groups = output.len();
    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(input.len());
        let cnt = (end - start) as f32;
        let mut i = start;
        let mut acc = _mm256_setzero_ps();
        while i + 8 <= end {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(input.as_ptr().add(i)));
            i += 8;
        }
        let mut sum = hsum256_ps(acc);
        for j in i..end { sum += input[j]; }
        output[g] = if is_mean { if cnt > 0.0 { sum / cnt } else { 0.0 } } else { sum };
    }
}

// ── reduce_max_f32 — group max ──────────────────────────────

#[inline]
pub fn reduce_max_f32_scalar(input: &[f32], output: &mut [f32], group_size: usize) {
    let num_groups = output.len();
    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(input.len());
        let mut val = f32::NEG_INFINITY;
        for i in start..end { if input[i] > val { val = input[i]; } }
        output[g] = val;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn reduce_max_f32_avx2(input: &[f32], output: &mut [f32], group_size: usize) {
    let num_groups = output.len();
    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(input.len());
        let mut i = start;
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        while i + 8 <= end {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(input.as_ptr().add(i)));
            i += 8;
        }
        // horizontal max
        let mx128 = _mm_max_ps(
            _mm256_castps256_ps128(vmax),
            _mm256_extractf128_ps(vmax, 1),
        );
        let mx = _mm_max_ps(mx128, _mm_shuffle_ps(mx128, mx128, 0x0E));
        let mx = _mm_max_ps(mx, _mm_shuffle_ps(mx, mx, 0x01));
        let mut max_val = _mm_cvtss_f32(mx);
        for j in i..end { if input[j] > max_val { max_val = input[j]; } }
        output[g] = max_val;
    }
}

// ── biasadd ─────────────────────────────────────────────────

#[inline]
pub fn biasadd_f32_scalar(data: &[f32], bias: &[f32], output: &mut [f32], channel_stride: usize) {
    let len = output.len().min(data.len());
    let bias_len = bias.len().max(1);
    for i in 0..len {
        output[i] = data[i] + bias[(i / channel_stride) % bias_len];
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn biasadd_f32_avx2(data: &[f32], bias: &[f32], output: &mut [f32], channel_stride: usize) {
    let len = output.len().min(data.len());
    let bias_len = bias.len().max(1);
    let mut i = 0;
    while i + 8 <= len {
        let vdata = _mm256_loadu_ps(data.as_ptr().add(i));
        let b0 = bias[((i + 0) / channel_stride) % bias_len];
        let b1 = bias[((i + 1) / channel_stride) % bias_len];
        let b2 = bias[((i + 2) / channel_stride) % bias_len];
        let b3 = bias[((i + 3) / channel_stride) % bias_len];
        let b4 = bias[((i + 4) / channel_stride) % bias_len];
        let b5 = bias[((i + 5) / channel_stride) % bias_len];
        let b6 = bias[((i + 6) / channel_stride) % bias_len];
        let b7 = bias[((i + 7) / channel_stride) % bias_len];
        let vbias = _mm256_set_ps(b7, b6, b5, b4, b3, b2, b1, b0);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_add_ps(vdata, vbias));
        i += 8;
    }
    for j in i..len {
        *output.as_mut_ptr().add(j) = data[j] + bias[(j / channel_stride) % bias_len];
    }
}

// ── norm_layernorm_f32 — LayerNorm ──────────────────────────

#[inline]
pub fn norm_layernorm_f32_scalar(input: &[f32], output: &mut [f32], row_size: usize, eps: f32) {
    let num_rows = input.len() / row_size;
    for r in 0..num_rows {
        let start = r * row_size;
        let end = start + row_size;
        let n = row_size as f32;
        // Pass 1: sum
        let mut sum = 0.0f32;
        for i in start..end { sum += input[i]; }
        let mean = if n > 0.0 { sum / n } else { 0.0 };
        // Pass 2: sum of squared diffs
        let mut var = 0.0f32;
        for i in start..end { let d = input[i] - mean; var += d * d; }
        var = var / n;
        let inv_std = 1.0 / (var + eps).sqrt();
        // Pass 3: normalize
        for i in start..end { output[i] = (input[i] - mean) * inv_std; }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn norm_layernorm_f32_avx2(input: &[f32], output: &mut [f32], row_size: usize, eps: f32) {
    let num_rows = input.len() / row_size;
    for r in 0..num_rows {
        let start = r * row_size;
        let end = start + row_size;
        let n = row_size as f32;
        // Pass 1: vectorized sum
        let mut i = start;
        let mut vsum = _mm256_setzero_ps();
        while i + 8 <= end {
            vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(input.as_ptr().add(i)));
            i += 8;
        }
        let mut sum = hsum256_ps(vsum);
        for j in i..end { sum += input[j]; }
        let mean = if n > 0.0 { sum / n } else { 0.0 };
        let vmean = _mm256_set1_ps(mean);
        // Pass 2: vectorized sum of (x - mean)^2
        i = start;
        let mut vvar = _mm256_setzero_ps();
        while i + 8 <= end {
            let d = _mm256_sub_ps(_mm256_loadu_ps(input.as_ptr().add(i)), vmean);
            vvar = _mm256_fmadd_ps(d, d, vvar);
            i += 8;
        }
        let mut var = hsum256_ps(vvar);
        for j in i..end { let d = input[j] - mean; var += d * d; }
        var = var / n;
        let inv_std = 1.0 / (var + eps).sqrt();
        // Pass 3: vectorized normalize
        let vinv_std = _mm256_set1_ps(inv_std);
        i = start;
        while i + 8 <= end {
            _mm256_storeu_ps(
                output.as_mut_ptr().add(i),
                _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(input.as_ptr().add(i)), vmean), vinv_std),
            );
            i += 8;
        }
        for j in i..end { output[j] = (input[j] - mean) * inv_std; }
    }
}

// ── rms_norm_f32 ────────────────────────────────────────────

#[inline]
pub fn rms_norm_f32_scalar(input: &[f32], weight: &[f32], output: &mut [f32], row_size: usize, eps: f32) {
    let num_rows = input.len() / row_size;
    for r in 0..num_rows {
        let start = r * row_size;
        let end = start + row_size;
        let mut sq_sum = 0.0f32;
        for i in start..end { sq_sum += input[i] * input[i]; }
        let n = row_size as f32;
        let rms = if n > 0.0 { (sq_sum / n + eps).sqrt() } else { 1.0 };
        for i in start..end {
            let w = if i - start < weight.len() { weight[i - start] } else { 1.0 };
            output[i] = input[i] / rms * w;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn rms_norm_f32_avx2(input: &[f32], weight: &[f32], output: &mut [f32], row_size: usize, eps: f32) {
    let num_rows = input.len() / row_size;
    for r in 0..num_rows {
        let start = r * row_size;
        let end = start + row_size;
        // Vectorized sum of squares
        let mut i = start;
        let mut vsq = _mm256_setzero_ps();
        while i + 8 <= end {
            let vx = _mm256_loadu_ps(input.as_ptr().add(i));
            vsq = _mm256_fmadd_ps(vx, vx, vsq);
            i += 8;
        }
        let mut sq_sum = hsum256_ps(vsq);
        for j in i..end { sq_sum += input[j] * input[j]; }
        let n = row_size as f32;
        let rms = if n > 0.0 { (sq_sum / n + eps).sqrt() } else { 1.0 };
        let inv_rms = _mm256_set1_ps(1.0 / rms);
        // Vectorized writeback with weight
        i = start;
        while i + 8 <= end {
            let vx = _mm256_loadu_ps(input.as_ptr().add(i));
            let w = if weight.len() >= 8 {
                _mm256_loadu_ps(weight.as_ptr().add(i - start))
            } else {
                _mm256_set1_ps(if i - start < weight.len() { weight[i - start] } else { 1.0 })
            };
            _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_mul_ps(vx, inv_rms), w));
            i += 8;
        }
        for j in i..end {
            let w = if j - start < weight.len() { weight[j - start] } else { 1.0 };
            *output.as_mut_ptr().add(j) = input[j] / rms * w;
        }
    }
}

// ── Scalar arithmetic ───────────────────────────────────────

#[inline]
pub fn add_scalar_f32_scalar(data: &[f32], s: f32, output: &mut [f32]) {
    let len = output.len().min(data.len());
    for i in 0..len { output[i] = data[i] + s; }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn add_scalar_f32_avx2(data: &[f32], s: f32, output: &mut [f32]) {
    let len = output.len().min(data.len());
    let mut i = 0;
    let vs = _mm256_set1_ps(s);
    while i + 8 <= len {
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_add_ps(_mm256_loadu_ps(data.as_ptr().add(i)), vs));
        i += 8;
    }
    for j in i..len { *output.as_mut_ptr().add(j) = data[j] + s; }
}

#[inline]
pub fn mul_scalar_f32_scalar(data: &[f32], s: f32, output: &mut [f32]) {
    let len = output.len().min(data.len());
    for i in 0..len { output[i] = data[i] * s; }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn mul_scalar_f32_avx2(data: &[f32], s: f32, output: &mut [f32]) {
    let len = output.len().min(data.len());
    let mut i = 0;
    let vs = _mm256_set1_ps(s);
    while i + 8 <= len {
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_loadu_ps(data.as_ptr().add(i)), vs));
        i += 8;
    }
    for j in i..len { *output.as_mut_ptr().add(j) = data[j] * s; }
}

#[inline]
pub fn div_scalar_f32_scalar(data: &[f32], s: f32, output: &mut [f32]) {
    let len = output.len().min(data.len());
    for i in 0..len { output[i] = data[i] / s; }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn div_scalar_f32_avx2(data: &[f32], s: f32, output: &mut [f32]) {
    let len = output.len().min(data.len());
    let mut i = 0;
    let vs = _mm256_set1_ps(s);
    while i + 8 <= len {
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_div_ps(_mm256_loadu_ps(data.as_ptr().add(i)), vs));
        i += 8;
    }
    for j in i..len { *output.as_mut_ptr().add(j) = data[j] / s; }
}

// ── Scalar comparison ───────────────────────────────────────

#[inline]
pub fn gt_scalar_f32_scalar(data: &[f32], s: f32, output: &mut [f32]) {
    let len = output.len().min(data.len());
    for i in 0..len { output[i] = if data[i] > s { 1.0 } else { 0.0 }; }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn gt_scalar_f32_avx2(data: &[f32], s: f32, output: &mut [f32]) {
    let len = output.len().min(data.len());
    let mut i = 0;
    let vs = _mm256_set1_ps(s);
    let vone = _mm256_set1_ps(1.0);
    let vzero = _mm256_setzero_ps();
    while i + 8 <= len {
        let mask = _mm256_cmp_ps::<{_CMP_GT_OQ}>(_mm256_loadu_ps(data.as_ptr().add(i)), vs);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_blendv_ps(vzero, vone, mask));
        i += 8;
    }
    for j in i..len { *output.as_mut_ptr().add(j) = if data[j] > s { 1.0 } else { 0.0 }; }
}

#[inline]
pub fn lt_scalar_f32_scalar(data: &[f32], s: f32, output: &mut [f32]) {
    let len = output.len().min(data.len());
    for i in 0..len { output[i] = if data[i] < s { 1.0 } else { 0.0 }; }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn lt_scalar_f32_avx2(data: &[f32], s: f32, output: &mut [f32]) {
    let len = output.len().min(data.len());
    let mut i = 0;
    let vs = _mm256_set1_ps(s);
    let vone = _mm256_set1_ps(1.0);
    let vzero = _mm256_setzero_ps();
    while i + 8 <= len {
        let mask = _mm256_cmp_ps::<{_CMP_LT_OQ}>(_mm256_loadu_ps(data.as_ptr().add(i)), vs);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_blendv_ps(vzero, vone, mask));
        i += 8;
    }
    for j in i..len { *output.as_mut_ptr().add(j) = if data[j] < s { 1.0 } else { 0.0 }; }
}

#[inline]
pub fn eq_scalar_f32_scalar(data: &[f32], s: f32, output: &mut [f32]) {
    let len = output.len().min(data.len());
    for i in 0..len { output[i] = if (data[i] - s).abs() < 1e-6 { 1.0 } else { 0.0 }; }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn eq_scalar_f32_avx2(data: &[f32], s: f32, output: &mut [f32]) {
    let len = output.len().min(data.len());
    let mut i = 0;
    let vs = _mm256_set1_ps(s);
    let vone = _mm256_set1_ps(1.0);
    let vzero = _mm256_setzero_ps();
    let veps = _mm256_set1_ps(1e-6);
    let vabsmask = _mm256_set1_ps(f32::from_bits(0x7fffffff));
    while i + 8 <= len {
        let vdiff = _mm256_sub_ps(_mm256_loadu_ps(data.as_ptr().add(i)), vs);
        let vabsdiff = _mm256_and_ps(vdiff, vabsmask);
        let mask = _mm256_cmp_ps::<{_CMP_LT_OQ}>(vabsdiff, veps);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_blendv_ps(vzero, vone, mask));
        i += 8;
    }
    for j in i..len { *output.as_mut_ptr().add(j) = if (data[j] - s).abs() < 1e-6 { 1.0 } else { 0.0 }; }
}

// ── softmax ─────────────────────────────────────────────────

#[inline]
pub fn softmax_f32_scalar_strided(input: &[f32], output: &mut [f32], axis_dim_size: usize, stride: usize, num_rows: usize) {
    for r in 0..num_rows {
        let outer = r / stride;
        let inner = r % stride;
        let base = (outer * axis_dim_size * stride) + inner;
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..axis_dim_size {
            let idx = base + i * stride;
            if input[idx] > max_val { max_val = input[idx]; }
        }
        let mut sum = 0.0f32;
        for i in 0..axis_dim_size {
            let idx = base + i * stride;
            let e = (input[idx] - max_val).exp();
            output[idx] = e;
            sum += e;
        }
        if sum > 0.0 {
            for i in 0..axis_dim_size {
                let idx = base + i * stride;
                output[idx] /= sum;
            }
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn softmax_f32_avx2_strided(input: &[f32], output: &mut [f32], axis_dim_size: usize, stride: usize, num_rows: usize) {
    if stride != 1 {
        // Strided access — fall back to scalar for correctness
        softmax_f32_scalar_strided(input, output, axis_dim_size, stride, num_rows);
        return;
    }
    // Contiguous softmax: each row is axis_dim_size elements
    for r in 0..num_rows {
        let base = r * axis_dim_size;
        // 1. Vectorized max
        let mut i = base;
        let end = base + axis_dim_size;
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        while i + 8 <= end {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(input.as_ptr().add(i)));
            i += 8;
        }
        // horizontal max
        let mx128 = _mm_max_ps(
            _mm256_castps256_ps128(vmax),
            _mm256_extractf128_ps(vmax, 1),
        );
        let mx = _mm_max_ps(mx128, _mm_shuffle_ps(mx128, mx128, 0x0E));
        let mx = _mm_max_ps(mx, _mm_shuffle_ps(mx, mx, 0x01));
        let mut max_val = _mm_cvtss_f32(mx);
        for j in i..end { if input[j] > max_val { max_val = input[j]; } }
        let vmax_bcast = _mm256_set1_ps(max_val);
        // 2. Vectorized exp + sum using scalar f64 accumulator for precision
        let mut sum = 0.0f64;
        i = base;
        while i + 8 <= end {
            let vx = _mm256_loadu_ps(input.as_ptr().add(i));
            let vshifted = _mm256_sub_ps(vx, vmax_bcast);
            // Vectorized exp using existing polynomial approximation
            let vexp = exp_avx2_vec(vshifted);
            let arr: [f32; 8] = std::mem::transmute(vexp);
            for k in 0..8 {
                sum += arr[k] as f64;
            }
            _mm256_storeu_ps(output.as_mut_ptr().add(i), vexp);
            i += 8;
        }
        for j in i..end {
            let e = (input[j] - max_val).exp();
            *output.as_mut_ptr().add(j) = e;
            sum += e as f64;
        }
        let sum_f32 = sum as f32;
        if sum_f32 > 0.0 {
            let vsum = _mm256_set1_ps(sum_f32);
            i = base;
            while i + 8 <= end {
                let vexp = _mm256_loadu_ps(output.as_ptr().add(i));
                _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_div_ps(vexp, vsum));
                i += 8;
            }
            for j in i..end { *output.as_mut_ptr().add(j) /= sum_f32; }
        }
    }
}

// ── Optimizer: sgd_update_f32 ───────────────────────────────

#[inline]
pub fn sgd_update_f32_scalar(w: &[f32], g: &[f32], lr: f32, wd: f32) -> Vec<f32> {
    let len = w.len();
    let mut result = w.to_vec();
    for i in 0..len {
        // w - lr * (g + wd * w)  =  w - lr*g - lr*wd*w
        result[i] -= lr * g.get(i % g.len()).copied().unwrap_or(0.0);
        result[i] -= lr * wd * w[i];
    }
    result
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn sgd_update_f32_avx2(w: &[f32], g: &[f32], lr: f32, wd: f32) -> Vec<f32> {
    let len = w.len();
    let mut result = w.to_vec();
    let mut i = 0;
    let vlr = _mm256_set1_ps(lr);
    let vwd = _mm256_set1_ps(wd);
    if g.len() == len {
        while i + 8 <= len {
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vg = _mm256_loadu_ps(g.as_ptr().add(i));
            // w - lr * (g + wd * w)
            // = w - lr*g - lr*wd*w
            // = vw - vlr * (vg + vwd * vw)
            let vg_wd = _mm256_fmadd_ps(vwd, vw, vg); // vwd * vw + vg
            _mm256_storeu_ps(result.as_mut_ptr().add(i), _mm256_fnmadd_ps(vg_wd, vlr, vw));
            i += 8;
        }
    }
    for j in i..len {
        result[j] -= lr * g.get(j % g.len()).copied().unwrap_or(0.0);
        result[j] -= lr * wd * w[j];
    }
    result
}

// ── Optimizer: adam_update_f32 ──────────────────────────────

#[inline]
pub fn adam_update_f32_scalar(
    w: &[f32], g: &[f32], m: &[f32], v: &[f32],
    lr: f32, beta1: f32, beta2: f32, eps: f32, bias_corr1: f32, bias_corr2: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = w.len();
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    let mut v_new = v.to_vec();
    for i in 0..len {
        let gi = g.get(i % g.len()).copied().unwrap_or(0.0);
        m_new[i] = beta1 * m[i] + (1.0 - beta1) * gi;
        v_new[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi;
        let m_hat = m_new[i] / bias_corr1;
        let v_hat = v_new[i] / bias_corr2;
        w_new[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
    (w_new, m_new, v_new)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn adam_update_f32_avx2(
    w: &[f32], g: &[f32], m: &[f32], v: &[f32],
    lr: f32, beta1: f32, beta2: f32, eps: f32, bias_corr1: f32, bias_corr2: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = w.len();
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    let mut v_new = v.to_vec();
    let vlr = _mm256_set1_ps(lr);
    let vb1 = _mm256_set1_ps(beta1);
    let vb1c = _mm256_set1_ps(1.0 - beta1);
    let vb2 = _mm256_set1_ps(beta2);
    let vb2c = _mm256_set1_ps(1.0 - beta2);
    let veps = _mm256_set1_ps(eps);
    let vbc1 = _mm256_set1_ps(bias_corr1);
    let vbc2 = _mm256_set1_ps(bias_corr2);
    let mut i = 0;
    if g.len() == len {
        while i + 8 <= len {
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vg = _mm256_loadu_ps(g.as_ptr().add(i));
            let vm = _mm256_loadu_ps(m.as_ptr().add(i));
            let vv = _mm256_loadu_ps(v.as_ptr().add(i));
            // m_new = beta1 * m + (1-beta1) * g
            let vm_new = _mm256_fmadd_ps(vb1c, vg, _mm256_mul_ps(vb1, vm));
            // v_new = beta2 * v + (1-beta2) * g * g
            let vv_new = _mm256_fmadd_ps(vb2c, _mm256_mul_ps(vg, vg), _mm256_mul_ps(vb2, vv));
            // m_hat = m_new / bias_corr1
            let vm_hat = _mm256_div_ps(vm_new, vbc1);
            // v_hat = v_new / bias_corr2
            let vv_hat = _mm256_div_ps(vv_new, vbc2);
            // w -= lr * m_hat / (sqrt(v_hat) + eps)
            let vdenom = _mm256_add_ps(_mm256_sqrt_ps(vv_hat), veps);
            let vupdate = _mm256_div_ps(_mm256_mul_ps(vlr, vm_hat), vdenom);
            _mm256_storeu_ps(w_new.as_mut_ptr().add(i), _mm256_sub_ps(vw, vupdate));
            _mm256_storeu_ps(m_new.as_mut_ptr().add(i), vm_new);
            _mm256_storeu_ps(v_new.as_mut_ptr().add(i), vv_new);
            i += 8;
        }
    }
    for j in i..len {
        let gi = g.get(j % g.len()).copied().unwrap_or(0.0);
        m_new[j] = beta1 * m[j] + (1.0 - beta1) * gi;
        v_new[j] = beta2 * v[j] + (1.0 - beta2) * gi * gi;
        let m_hat = m_new[j] / bias_corr1;
        let v_hat = v_new[j] / bias_corr2;
        w_new[j] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
    (w_new, m_new, v_new)
}

// ── Optimizer: adamw_update_f32 ─────────────────────────────

#[inline]
pub fn adamw_update_f32_scalar(
    w: &[f32], g: &[f32], m: &[f32], v: &[f32],
    lr: f32, beta1: f32, beta2: f32, eps: f32, bias_corr1: f32, bias_corr2: f32, wd: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = w.len();
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    let mut v_new = v.to_vec();
    for i in 0..len {
        w_new[i] -= lr * wd * w[i];
        let gi = g.get(i % g.len()).copied().unwrap_or(0.0);
        m_new[i] = beta1 * m[i] + (1.0 - beta1) * gi;
        v_new[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi;
        let m_hat = m_new[i] / bias_corr1;
        let v_hat = v_new[i] / bias_corr2;
        w_new[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
    (w_new, m_new, v_new)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn adamw_update_f32_avx2(
    w: &[f32], g: &[f32], m: &[f32], v: &[f32],
    lr: f32, beta1: f32, beta2: f32, eps: f32, bias_corr1: f32, bias_corr2: f32, wd: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = w.len();
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    let mut v_new = v.to_vec();
    let vlr = _mm256_set1_ps(lr);
    let vwd = _mm256_set1_ps(wd);
    let vb1 = _mm256_set1_ps(beta1);
    let vb1c = _mm256_set1_ps(1.0 - beta1);
    let vb2 = _mm256_set1_ps(beta2);
    let vb2c = _mm256_set1_ps(1.0 - beta2);
    let veps = _mm256_set1_ps(eps);
    let vbc1 = _mm256_set1_ps(bias_corr1);
    let vbc2 = _mm256_set1_ps(bias_corr2);
    let mut i = 0;
    if g.len() == len {
        while i + 8 <= len {
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vg = _mm256_loadu_ps(g.as_ptr().add(i));
            let vm = _mm256_loadu_ps(m.as_ptr().add(i));
            let vv = _mm256_loadu_ps(v.as_ptr().add(i));
            // w -= lr * wd * w  (weight decay)
            let vw_decayed = _mm256_fnmadd_ps(_mm256_mul_ps(vlr, vwd), vw, vw);
            // m_new = beta1 * m + (1-beta1) * g
            let vm_new = _mm256_fmadd_ps(vb1c, vg, _mm256_mul_ps(vb1, vm));
            // v_new = beta2 * v + (1-beta2) * g * g
            let vv_new = _mm256_fmadd_ps(vb2c, _mm256_mul_ps(vg, vg), _mm256_mul_ps(vb2, vv));
            // m_hat = m_new / bias_corr1; v_hat = v_new / bias_corr2
            let vm_hat = _mm256_div_ps(vm_new, vbc1);
            let vv_hat = _mm256_div_ps(vv_new, vbc2);
            // w -= lr * m_hat / (sqrt(v_hat) + eps)
            let vdenom = _mm256_add_ps(_mm256_sqrt_ps(vv_hat), veps);
            let vupdate = _mm256_div_ps(_mm256_mul_ps(vlr, vm_hat), vdenom);
            _mm256_storeu_ps(w_new.as_mut_ptr().add(i), _mm256_sub_ps(vw_decayed, vupdate));
            _mm256_storeu_ps(m_new.as_mut_ptr().add(i), vm_new);
            _mm256_storeu_ps(v_new.as_mut_ptr().add(i), vv_new);
            i += 8;
        }
    }
    for j in i..len {
        w_new[j] -= lr * wd * w[j];
        let gi = g.get(j % g.len()).copied().unwrap_or(0.0);
        m_new[j] = beta1 * m[j] + (1.0 - beta1) * gi;
        v_new[j] = beta2 * v[j] + (1.0 - beta2) * gi * gi;
        let m_hat = m_new[j] / bias_corr1;
        let v_hat = v_new[j] / bias_corr2;
        w_new[j] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
    (w_new, m_new, v_new)
}

// ── Optimizer: lion_update_f32 ──────────────────────────────

#[inline]
pub fn lion_update_f32_scalar(
    w: &[f32], g: &[f32], m: &[f32],
    lr: f32, beta1: f32, beta2: f32, wd: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = w.len();
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    for i in 0..len {
        let gi = g.get(i % g.len()).copied().unwrap_or(0.0);
        m_new[i] = beta2 * m[i] + (1.0 - beta2) * gi;
        let update = beta1 * m_new[i] + (1.0 - beta1) * gi;
        // Lion with decoupled weight decay: w -= lr * (sign(update) + wd * w)
        w_new[i] -= lr * (update.signum() + wd * w[i]);
    }
    (w_new, m_new)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn lion_update_f32_avx2(
    w: &[f32], g: &[f32], m: &[f32],
    lr: f32, beta1: f32, beta2: f32, wd: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = w.len();
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    let vlr = _mm256_set1_ps(lr);
    let vwd = _mm256_set1_ps(wd);
    let vb1 = _mm256_set1_ps(beta1);
    let vb2 = _mm256_set1_ps(beta2);
    let vb1c = _mm256_set1_ps(1.0 - beta1);
    let vb2c = _mm256_set1_ps(1.0 - beta2);
    let vone = _mm256_set1_ps(1.0);
    let vneg_one = _mm256_set1_ps(-1.0);
    let vzero = _mm256_setzero_ps();
    let mut i = 0;
    if g.len() == len {
        while i + 8 <= len {
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vg = _mm256_loadu_ps(g.as_ptr().add(i));
            let vm = _mm256_loadu_ps(m.as_ptr().add(i));
            // m_new = beta2 * m + (1-beta2) * g
            let vm_new = _mm256_fmadd_ps(vb2c, vg, _mm256_mul_ps(vb2, vm));
            // update = beta1 * m_new + (1-beta1) * g
            let vupdate = _mm256_fmadd_ps(vb1c, vg, _mm256_mul_ps(vb1, vm_new));
            // signum(update) with blendv
            let vpos_mask = _mm256_cmp_ps::<{_CMP_GT_OQ}>(vupdate, vzero);
            let vneg_mask = _mm256_cmp_ps::<{_CMP_LT_OQ}>(vupdate, vzero);
            let vsign = _mm256_or_ps(
                _mm256_and_ps(vpos_mask, vone),
                _mm256_and_ps(vneg_mask, vneg_one),
            );
            // w -= lr * (sign + wd * w)  =  vw - vlr * (vsign + vwd * vw)
            let vsign_wd = _mm256_fmadd_ps(vwd, vw, vsign); // vwd * vw + vsign
            _mm256_storeu_ps(w_new.as_mut_ptr().add(i), _mm256_fnmadd_ps(vlr, vsign_wd, vw));
            _mm256_storeu_ps(m_new.as_mut_ptr().add(i), vm_new);
            i += 8;
        }
    }
    for j in i..len {
        let gi = g.get(j % g.len()).copied().unwrap_or(0.0);
        m_new[j] = beta2 * m[j] + (1.0 - beta2) * gi;
        let update = beta1 * m_new[j] + (1.0 - beta1) * gi;
        w_new[j] -= lr * (update.signum() + wd * w[j]);
    }
    (w_new, m_new)
}

// ── Optimizer: rmsprop_update_f32 ───────────────────────────

#[inline]
pub fn rmsprop_update_f32_scalar(
    w: &[f32], g: &[f32], v: &[f32],
    lr: f32, beta: f32, eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = w.len();
    let mut w_new = w.to_vec();
    let mut v_new = v.to_vec();
    for i in 0..len {
        let gi = g.get(i % g.len()).copied().unwrap_or(0.0);
        v_new[i] = beta * v[i] + (1.0 - beta) * gi * gi;
        w_new[i] -= lr * gi / (v_new[i].sqrt() + eps);
    }
    (w_new, v_new)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn rmsprop_update_f32_avx2(
    w: &[f32], g: &[f32], v: &[f32],
    lr: f32, beta: f32, eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = w.len();
    let mut w_new = w.to_vec();
    let mut v_new = v.to_vec();
    let vlr = _mm256_set1_ps(lr);
    let vb = _mm256_set1_ps(beta);
    let vbc = _mm256_set1_ps(1.0 - beta);
    let veps = _mm256_set1_ps(eps);
    let mut i = 0;
    if g.len() == len {
        while i + 8 <= len {
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vg = _mm256_loadu_ps(g.as_ptr().add(i));
            let vv = _mm256_loadu_ps(v.as_ptr().add(i));
            // v_new = beta * v + (1-beta) * g * g
            let vv_new = _mm256_fmadd_ps(vbc, _mm256_mul_ps(vg, vg), _mm256_mul_ps(vb, vv));
            // w -= lr * g / (sqrt(v_new) + eps)
            let vdenom = _mm256_add_ps(_mm256_sqrt_ps(vv_new), veps);
            let vupdate = _mm256_div_ps(_mm256_mul_ps(vlr, vg), vdenom);
            _mm256_storeu_ps(w_new.as_mut_ptr().add(i), _mm256_sub_ps(vw, vupdate));
            _mm256_storeu_ps(v_new.as_mut_ptr().add(i), vv_new);
            i += 8;
        }
    }
    for j in i..len {
        let gi = g.get(j % g.len()).copied().unwrap_or(0.0);
        v_new[j] = beta * v[j] + (1.0 - beta) * gi * gi;
        w_new[j] -= lr * gi / (v_new[j].sqrt() + eps);
    }
    (w_new, v_new)
}

// ── Optimizer: muon_update_f32 ──────────────────────────────

#[inline]
pub fn muon_update_f32_scalar(
    w: &[f32], g: &[f32], m: &[f32],
    lr: f32, beta: f32, wd: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = w.len();
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    for i in 0..len {
        let gi = g.get(i % g.len()).copied().unwrap_or(0.0);
        m_new[i] = beta * m[i] + (1.0 - beta) * gi;
        w_new[i] = w[i] - lr * (m_new[i] + wd * w[i]);
    }
    (w_new, m_new)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn muon_update_f32_avx2(
    w: &[f32], g: &[f32], m: &[f32],
    lr: f32, beta: f32, wd: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = w.len();
    let mut w_new = w.to_vec();
    let mut m_new = m.to_vec();
    let vlr = _mm256_set1_ps(lr);
    let vb = _mm256_set1_ps(beta);
    let vbc = _mm256_set1_ps(1.0 - beta);
    let vwd = _mm256_set1_ps(wd);
    let mut i = 0;
    if g.len() == len {
        while i + 8 <= len {
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vg = _mm256_loadu_ps(g.as_ptr().add(i));
            let vm = _mm256_loadu_ps(m.as_ptr().add(i));
            // m_new = beta * m + (1-beta) * g
            let vm_new = _mm256_fmadd_ps(vbc, vg, _mm256_mul_ps(vb, vm));
            // w_new = w - lr * (m_new + wd * w)
            let vwd_term = _mm256_mul_ps(vwd, vw);
            let vinner = _mm256_add_ps(vm_new, vwd_term);
            let vw_new = _mm256_fnmadd_ps(vlr, vinner, vw);
            _mm256_storeu_ps(w_new.as_mut_ptr().add(i), vw_new);
            _mm256_storeu_ps(m_new.as_mut_ptr().add(i), vm_new);
            i += 8;
        }
    }
    for j in i..len {
        let gi = g.get(j % g.len()).copied().unwrap_or(0.0);
        m_new[j] = beta * m[j] + (1.0 - beta) * gi;
        w_new[j] = w[j] - lr * (m_new[j] + wd * w[j]);
    }
    (w_new, m_new)
}

// ============================================================
// im2col + GEMM based conv2d
// ============================================================

/// Optimized conv2d using im2col transformation + sgemm.
/// Fast path for 1x1 convolutions (no im2col needed).
#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32_im2col_gemm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    activation: Option<fn(f32) -> f32>,
) {
    let c_per_group = c / groups.max(1);
    let f_per_group = f / groups.max(1);
    let h_out = (h + 2 * padding).saturating_sub(dilation * (kh.saturating_sub(1)) + 1) / stride + 1;
    let w_out = (w + 2 * padding).saturating_sub(dilation * (kw.saturating_sub(1)) + 1) / stride + 1;
    let spatial_size = h_out * w_out;

    // Small tensor fallback to avoid im2col overhead
    if spatial_size * f < 64 {
        conv2d_f32_tiled(
            input, weight, bias, output, n, c, h, w, f, kh, kw, stride, padding, dilation, groups,
        );
        // Apply activation separately for tiled path
        if let Some(act) = activation {
            for x in output.iter_mut() {
                *x = act(*x);
            }
        }
        return;
    }

    // Fast path for 1x1 convolutions (no im2col, no weight transpose).
    // The input NCHW data is treated as a column-major [c_per_group, num_pixels] matrix
    // (rs_a=1, cs_a=spatial_size_per_batch) and the weight as column-major [c, f]
    // (rs_b=1, cs_b=c_per_group). This avoids the im2col copy and the weight transpose,
    // saving ~2 * num_pixels * c elements of memory traffic per group.
    if kh == 1 && kw == 1 && stride == 1 && padding == 0 && dilation == 1 && groups == 1 {
        let col_w = c_per_group; // = c since groups == 1
        let num_pixels = n * spatial_size;
        let hw_per_img = h * w;

        for g in 0..groups {
            let f_start = g * f_per_group;
            let input_group_off = g * col_w * hw_per_img;
            let weight_off = f_start * col_w;

            let mut temp_out = get_conv_buf!(&CONV_TEMP_OUT_BUF, num_pixels * f_per_group);
            unsafe {
                // Direct GEMM: A[num_pixels, col_w] = input[col_w, hw] with rs_a=1, cs_a=hw
                //   A[spatial][ch] = input[ch * hw + spatial]  (NCHW: within each batch, channels are outer)
                // B[col_w, f] = weight[f, col_w] with rs_b=1, cs_b=col_w
                //   B[ch][oc] = weight[oc * col_w + ch]
                matrixmultiply::sgemm(
                    num_pixels,
                    col_w,
                    f_per_group,
                    1.0,
                    input.as_ptr().add(input_group_off),
                    1isize,
                    hw_per_img as isize,
                    weight.as_ptr().add(weight_off),
                    1isize,
                    col_w as isize,
                    0.0,
                    temp_out.as_mut_ptr(),
                    f_per_group as isize,
                    1isize,
                );
            }

            // Scatter temp_out [num_pixels, f] to NCHW output with bias and optional activation
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                let num_pixels = num_pixels;
                let f_per_group = f_per_group;
                let f_start = f_start;
                let spatial_size = spatial_size;
                let f = f;
                let bias = bias;
                let activation = activation;
                // RefMut<Vec<f32>> is !Sync, so cast to usize (Sync) and
                // reconstruct the pointer inside the closure.
                let temp_ptr = temp_out.as_ptr() as usize;
                let out_ptr = output.as_mut_ptr() as usize;
                let f_whole = f;
                (0..num_pixels * f_per_group).into_par_iter().for_each(|idx| {
                    let pixel = idx / f_per_group;
                    let ff = idx % f_per_group;
                    let nn = pixel / spatial_size;
                    let spatial = pixel % spatial_size;
                    let mut val = unsafe { *(temp_ptr as *const f32).add(pixel * f_per_group + ff) };
                    if !bias.is_empty() {
                        val += bias[f_start + ff];
                    }
                    if let Some(act) = activation {
                        val = act(val);
                    }
                    unsafe {
                        *(out_ptr as *mut f32).add(nn * (f_whole * spatial_size) + (f_start + ff) * spatial_size + spatial) = val;
                    }
                });
            }
            #[cfg(not(feature = "parallel"))]
            for pixel in 0..num_pixels {
                let nn = pixel / spatial_size;
                let spatial = pixel % spatial_size;
                for ff in 0..f_per_group {
                    let mut val = temp_out[pixel * f_per_group + ff];
                    if !bias.is_empty() {
                        val += bias[f_start + ff];
                    }
                    if let Some(act) = activation {
                        val = act(val);
                    }
                    output[nn * (f * spatial_size) + (f_start + ff) * spatial_size + spatial] = val;
                }
            }
        }
        return;
    }

    // General case: im2col + GEMM (for non-1x1 convolutions)
    for g in 0..groups {
        let f_start = g * f_per_group;
        let input_group_off = g * c_per_group * (h * w);

        let col_w = c_per_group * kh * kw;
        let num_pixels = n * spatial_size;
        let mut col_matrix = get_conv_buf!(&CONV_COL_BUF, num_pixels * col_w);

        for nn in 0..n {
            let col_start = nn * spatial_size * col_w;
            unsafe {
                crate::backend::cpu::im2col::im2col_dispatch(
                    &input[nn * (c * h * w) + input_group_off..],
                    c_per_group,
                    h,
                    w,
                    kh,
                    kw,
                    stride,
                    padding,
                    dilation,
                    &mut col_matrix[col_start..],
                );
            }
        }

        let weight_off = f_start * col_w;

        let mut temp_out = get_conv_buf!(&CONV_TEMP_OUT_BUF, num_pixels * f_per_group);
        unsafe {
            // Read weight directly as column-major to avoid explicit transpose.
            // Weight is stored as [f_per_group][col_w] row-major. With rs_b=1, cs_b=col_w,
            // sgemm reads it as [col_w][f_per_group] column-major, eliminating the transpose.
            matrixmultiply::sgemm(
                num_pixels,
                col_w,
                f_per_group,
                1.0,
                col_matrix.as_ptr(),
                col_w as isize,
                1isize,
                weight.as_ptr().add(weight_off),
                1isize,
                col_w as isize,
                0.0,
                temp_out.as_mut_ptr(),
                f_per_group as isize,
                1isize,
            );
        }

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let num_pixels = num_pixels;
            let f_per_group = f_per_group;
            let f_start = f_start;
            let spatial_size = spatial_size;
            let f = f;
            let bias = bias;
            let activation = activation;
            // Same usize-cast Sync-safe approach as the 1×1 fast-path scatter above.
            let temp_ptr = temp_out.as_ptr() as usize;
            let out_ptr = output.as_mut_ptr() as usize;
            let f_whole = f;
            (0..num_pixels * f_per_group).into_par_iter().for_each(|idx| {
                let pixel = idx / f_per_group;
                let ff = idx % f_per_group;
                let nn = pixel / spatial_size;
                let spatial = pixel % spatial_size;
                let mut val = unsafe { *(temp_ptr as *const f32).add(pixel * f_per_group + ff) };
                if !bias.is_empty() {
                    val += bias[f_start + ff];
                }
                if let Some(act) = activation {
                    val = act(val);
                }
                unsafe {
                    *(out_ptr as *mut f32).add(nn * (f_whole * spatial_size) + (f_start + ff) * spatial_size + spatial) = val;
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        for pixel in 0..num_pixels {
            let nn = pixel / spatial_size;
            let spatial = pixel % spatial_size;
            for ff in 0..f_per_group {
                let mut val = temp_out[pixel * f_per_group + ff];
                if !bias.is_empty() {
                    val += bias[f_start + ff];
                }
                if let Some(act) = activation {
                    val = act(val);
                }
                output[nn * (f * spatial_size) + (f_start + ff) * spatial_size + spatial] = val;
            }
        }
    }
}

// ============================================================
// UpsampleNearest2d — scalar and AVX2
// ============================================================

/// Nearest-neighbor upsampling: each input pixel replicates into a
/// `scale_h × scale_w` block in the output.
///
/// # Layout
/// Input is NCHW: shape `[nc, h_in, w_in]` stored flat (`nc = N * C`).
/// Output is `[nc, h_in * scale_h, w_in * scale_w]`.
#[inline]
pub fn upsample_nearest2d_f32(
    input: &[f32],
    output: &mut [f32],
    nc: usize,
    h_in: usize,
    w_in: usize,
    scale_h: usize,
    scale_w: usize,
) {
    let hw = h_in * w_in;
    let out_row_stride = w_in * scale_w;
    for nci in 0..nc {
        for hi in 0..h_in {
            for wi in 0..w_in {
                let val = input[nci * hw + hi * w_in + wi];
                for sh in 0..scale_h {
                    let out_row = nci * hw * scale_h * scale_w
                        + (hi * scale_h + sh) * out_row_stride
                        + wi * scale_w;
                    for sw in 0..scale_w {
                        output[out_row + sw] = val;
                    }
                }
            }
        }
    }
}

/// AVX2 nearest-neighbor upsampling using broadcast stores.
///
/// For `scale_w ∈ {1, 2, 4}`, processes `8 / scale_w` input columns per
/// vector iteration, broadcasting each value `scale_w` times into an 8-wide
/// store. This reduces store instructions by up to 87.5% vs scalar.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn upsample_nearest2d_f32_avx2(
    input: &[f32],
    output: &mut [f32],
    nc: usize,
    h_in: usize,
    w_in: usize,
    scale_h: usize,
    scale_w: usize,
) {
    // Fallback to scalar for unsupported scale factors
    if scale_w != 1 && scale_w != 2 && scale_w != 4 {
        return upsample_nearest2d_f32(input, output, nc, h_in, w_in, scale_h, scale_w);
    }
    let hw = h_in * w_in;
    let out_row_stride = w_in * scale_w;
    // Number of input columns processed per AVX2 iteration (8 output floats / scale_w)
    let vec_step = 8 / scale_w;

    for nci in 0..nc {
        for hi in 0..h_in {
            let mut wi = 0;
            while wi + vec_step <= w_in {
                let in_base = nci * hw + hi * w_in + wi;
                // Build a vector of 8 floats with each input value repeated scale_w times.
                // _mm256_set_ps args go from highest lane (index 7) to lowest (index 0).
                // Memory order after store: arg0 at lowest address, arg7 at highest.
                let v = match scale_w {
                    1 => {
                        // Direct copy: 8 input values, no repetition
                        _mm256_loadu_ps(input.as_ptr().add(in_base))
                    }
                    2 => {
                        // 4 input values → each repeated twice
                        _mm256_set_ps(
                            *input.get_unchecked(in_base + 3),
                            *input.get_unchecked(in_base + 3),
                            *input.get_unchecked(in_base + 2),
                            *input.get_unchecked(in_base + 2),
                            *input.get_unchecked(in_base + 1),
                            *input.get_unchecked(in_base + 1),
                            *input.get_unchecked(in_base + 0),
                            *input.get_unchecked(in_base + 0),
                        )
                    }
                    4 => {
                        // 2 input values → each repeated 4 times
                        _mm256_set_ps(
                            *input.get_unchecked(in_base + 1),
                            *input.get_unchecked(in_base + 1),
                            *input.get_unchecked(in_base + 1),
                            *input.get_unchecked(in_base + 1),
                            *input.get_unchecked(in_base + 0),
                            *input.get_unchecked(in_base + 0),
                            *input.get_unchecked(in_base + 0),
                            *input.get_unchecked(in_base + 0),
                        )
                    }
                    _ => unreachable!(),
                };

                // Write the same 8-wide block to each output row (sh = 0..scale_h)
                for sh in 0..scale_h {
                    let out_base = nci * hw * scale_h * scale_w
                        + (hi * scale_h + sh) * out_row_stride
                        + wi * scale_w;
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_base), v);
                }

                wi += vec_step;
            }
            // Handle remaining columns (wi < vec_step from end)
            for wi in wi..w_in {
                let val = input[nci * hw + hi * w_in + wi];
                for sh in 0..scale_h {
                    let out_base = nci * hw * scale_h * scale_w
                        + (hi * scale_h + sh) * out_row_stride
                        + wi * scale_w;
                    for sw in 0..scale_w {
                        *output.get_unchecked_mut(out_base + sw) = val;
                    }
                }
            }
        }
    }
}

// ============================================================
// Concat — scalar and AVX2
// ============================================================

/// Scalar concat: copies `input` into `output` at `output_offset`.
/// Returns the new output offset after copy.
#[inline]
pub fn concat_f32_scalar(input: &[f32], output: &mut [f32], output_offset: usize) -> usize {
    let end = (output_offset + input.len()).min(output.len());
    let len = end - output_offset;
    output[output_offset..end].copy_from_slice(&input[..len]);
    output_offset + len
}

/// AVX2 concat: copies `input` into `output` at `output_offset` using
/// 8-wide vector stores for the bulk of the copy, falling back to scalar
/// for the remainder.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn concat_f32_avx2(input: &[f32], output: &mut [f32], output_offset: usize) -> usize {
    let out_start = output_offset;
    let copy_len = input.len().min(output.len().saturating_sub(out_start));
    let mut i = 0usize;
    // 8-wide vector copy
    while i + 8 <= copy_len {
        let v = _mm256_loadu_ps(input.as_ptr().add(i));
        _mm256_storeu_ps(output.as_mut_ptr().add(out_start + i), v);
        i += 8;
    }
    // Scalar remainder
    for j in i..copy_len {
        *output.as_mut_ptr().add(out_start + j) = *input.as_ptr().add(j);
    }
    out_start + copy_len
}


// ============================================================
// Transpose 2D — scalar and AVX2 tiled 8×8
// ============================================================

/// Scalar 2D transpose: out[j * m + i] = in[i * n + j]
#[inline]
pub fn transpose_f32_scalar(input: &[f32], output: &mut [f32], m: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            output[j * m + i] = input[i * n + j];
        }
    }
}

/// AVX2 tiled 8×8 2D transpose. Processes 8×8 tiles using unpack
/// and permute instructions, falling back to scalar for edge tiles.
///
/// Uses the classic:
///   t0 = _mm256_unpacklo_ps(a0, a1)  // a0[0], a1[0], a0[1], a1[1], ...
///   t1 = _mm256_unpackhi_ps(a0, a1)  // a0[2], a1[2], a0[3], a1[3], ...
///   ... then combine across rows with permute.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn transpose_f32_avx2(input: &[f32], output: &mut [f32], m: usize, n: usize) {
    use std::arch::x86_64::*;

    // Process 8×8 tiles
    let mut i = 0;
    while i + 8 <= m {
        let mut j = 0;
        while j + 8 <= n {
            // Load 8 rows of 8 elements each
            let r0 = _mm256_loadu_ps(input.as_ptr().add((i + 0) * n + j));
            let r1 = _mm256_loadu_ps(input.as_ptr().add((i + 1) * n + j));
            let r2 = _mm256_loadu_ps(input.as_ptr().add((i + 2) * n + j));
            let r3 = _mm256_loadu_ps(input.as_ptr().add((i + 3) * n + j));
            let r4 = _mm256_loadu_ps(input.as_ptr().add((i + 4) * n + j));
            let r5 = _mm256_loadu_ps(input.as_ptr().add((i + 5) * n + j));
            let r6 = _mm256_loadu_ps(input.as_ptr().add((i + 6) * n + j));
            let r7 = _mm256_loadu_ps(input.as_ptr().add((i + 7) * n + j));

            // Unpack low/high 64-bit halves (4-element chunks)
            let t01a = _mm256_unpacklo_ps(r0, r1);  // r0[0], r1[0], r0[1], r1[1]
            let t01b = _mm256_unpackhi_ps(r0, r1);  // r0[2], r1[2], r0[3], r1[3]
            let t23a = _mm256_unpacklo_ps(r2, r3);
            let t23b = _mm256_unpackhi_ps(r2, r3);
            let t45a = _mm256_unpacklo_ps(r4, r5);
            let t45b = _mm256_unpackhi_ps(r4, r5);
            let t67a = _mm256_unpacklo_ps(r6, r7);
            let t67b = _mm256_unpackhi_ps(r6, r7);

            // Combine into 8-wide transpose result
            let q0 = _mm256_shuffle_ps(t01a, t23a, 0b_01_00_01_00);  // cols 0,1 of rows 0-3
            let q1 = _mm256_shuffle_ps(t01a, t23a, 0b_11_10_11_10);  // cols 2,3 of rows 0-3
            let q2 = _mm256_shuffle_ps(t01b, t23b, 0b_01_00_01_00);  // cols 4,5 of rows 0-3
            let q3 = _mm256_shuffle_ps(t01b, t23b, 0b_11_10_11_10);  // cols 6,7 of rows 0-3
            let q4 = _mm256_shuffle_ps(t45a, t67a, 0b_01_00_01_00);  // cols 0,1 of rows 4-7
            let q5 = _mm256_shuffle_ps(t45a, t67a, 0b_11_10_11_10);  // cols 2,3 of rows 4-7
            let q6 = _mm256_shuffle_ps(t45b, t67b, 0b_01_00_01_00);  // cols 4,5 of rows 4-7
            let q7 = _mm256_shuffle_ps(t45b, t67b, 0b_11_10_11_10);  // cols 6,7 of rows 4-7

            // Permute to final layout — now rows are in order 0,4,1,5,2,6,3,7
            // We need them as 0,1,2,3,4,5,6,7
            let out0 = _mm256_permute2f128_ps(q0, q4, 0x20);
            let out4 = _mm256_permute2f128_ps(q0, q4, 0x31);
            let out1 = _mm256_permute2f128_ps(q1, q5, 0x20);
            let out5 = _mm256_permute2f128_ps(q1, q5, 0x31);
            let out2 = _mm256_permute2f128_ps(q2, q6, 0x20);
            let out6 = _mm256_permute2f128_ps(q2, q6, 0x31);
            let out3 = _mm256_permute2f128_ps(q3, q7, 0x20);
            let out7 = _mm256_permute2f128_ps(q3, q7, 0x31);

            // Store 8 cols × 8 rows in transposed position
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 0) * m + i), out0);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 1) * m + i), out1);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 2) * m + i), out2);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 3) * m + i), out3);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 4) * m + i), out4);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 5) * m + i), out5);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 6) * m + i), out6);
            _mm256_storeu_ps(output.as_mut_ptr().add((j + 7) * m + i), out7);

            j += 8;
        }
        // Scalar remainder columns
        for jj in j..n {
            for ii in i..i + 8 {
                *output.as_mut_ptr().add(jj * m + ii) = *input.as_ptr().add(ii * n + jj);
            }
        }
        i += 8;
    }
    // Scalar remainder rows
    for ii in i..m {
        for jj in 0..n {
            *output.as_mut_ptr().add(jj * m + ii) = *input.as_ptr().add(ii * n + jj);
        }
    }
}


// ============================================================
// NEON kernel correctness tests
// ============================================================

#[cfg(test)]
mod neon_tests {
    use super::*;

    fn test_vector(len: usize) -> Vec<f32> {
        (0..len).map(|i| (i as f32) * 0.25).collect()
    }

    fn random_vector(len: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    fn assert_f32_slice_eq(actual: &[f32], expected: &[f32], eps: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff < eps,
                "mismatch at index {}: actual={} expected={} diff={}",
                i,
                a,
                e,
                diff
            );
        }
    }

    fn softmax_scalar(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum).collect()
    }

    #[test]
    fn test_fma_f32_scalar() {
        let a = test_vector(16);
        let b = test_vector(16);
        let result = fma_f32_scalar(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-6, "fma scalar mismatch");
    }

    #[test]
    fn test_fma_f32_slice_consistency() {
        let a = random_vector(256);
        let b = random_vector(256);
        let result = fma_f32_slice(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-4, "fma slice mismatch");
    }

    #[test]
    fn test_fma_f32_slice_various_lengths() {
        for len in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 128] {
            let a = random_vector(len);
            let b = random_vector(len);
            let result = fma_f32_slice(&a, &b);
            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            assert!(
                (result - expected).abs() < 1e-4,
                "fma slice mismatch for len={}: result={} expected={}",
                len,
                result,
                expected
            );
        }
    }

    #[test]
    fn test_gemv_generic_fallback_vs_scalar() {
        let m = 4;
        let k = 16;
        let weights_f32 = random_vector(m * k);
        let activations = random_vector(k);

        let packed = PackedTensor::<F32x1>::from_f32_slice(&weights_f32, &[m, k], 1.0, 0.0);

        let mut output_generic = vec![0.0f32; m];
        gemv_generic_fallback(&packed, &activations, &mut output_generic);

        for row in 0..m {
            let dot: f32 = (0..k)
                .map(|j| weights_f32[row * k + j] * activations[j])
                .sum();
            assert!(
                (output_generic[row] - dot).abs() < 1e-4,
                "row {} mismatch: {} vs {}",
                row,
                output_generic[row],
                dot
            );
        }
    }

    #[test]
    fn test_gemv_dispatch_f32x1_vs_scalar() {
        let m = 4;
        let k = 32;
        let weights_f32 = random_vector(m * k);
        let activations = random_vector(k);

        let packed = PackedTensor::<F32x1>::from_f32_slice(&weights_f32, &[m, k], 1.0, 0.0);

        let mut output = vec![0.0f32; m];
        gemv_cpu(&packed, &activations, &mut output);

        for row in 0..m {
            let dot: f32 = (0..k)
                .map(|j| weights_f32[row * k + j] * activations[j])
                .sum();
            assert!(
                (output[row] - dot).abs() < 1e-4,
                "row {} mismatch: {} vs {}",
                row,
                output[row],
                dot
            );
        }
    }

    #[test]
    fn test_gemv_dispatch_u8x4_self_consistency() {
        let m = 4;
        let k = 32;
        let weights_f32: Vec<f32> = (0..m * k).map(|i| ((i % 32) as f32) - 16.0).collect();
        let activations = random_vector(k);

        let packed = PackedTensor::<U8x4>::from_f32_slice(&weights_f32, &[m, k], 1.0, 0.0);

        let mut output_simd = vec![0.0f32; m];
        gemv_cpu(&packed, &activations, &mut output_simd);

        let mut output_fallback = vec![0.0f32; m];
        gemv_generic_fallback(&packed, &activations, &mut output_fallback);

        assert_f32_slice_eq(&output_simd, &output_fallback, 1e-5);
    }

    #[test]
    fn test_gemv_dispatch_u4x8_self_consistency() {
        let m = 4;
        let k = 32;
        let weights_f32: Vec<f32> = (0..m * k).map(|i| ((i % 8) as f32) - 4.0).collect();
        let activations = random_vector(k);

        let packed = PackedTensor::<U4x8>::from_f32_slice(&weights_f32, &[m, k], 1.0, 0.0);

        let mut output_simd = vec![0.0f32; m];
        gemv_cpu(&packed, &activations, &mut output_simd);

        let mut output_fallback = vec![0.0f32; m];
        gemv_generic_fallback(&packed, &activations, &mut output_fallback);

        assert_f32_slice_eq(&output_simd, &output_fallback, 1e-5);
    }

    #[test]
    fn test_simd_dot_product() {
        let a = random_vector(256);
        let b = random_vector(256);
        let result = unsafe { simd_dot_product(&a, &b, a.len()) };
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-4, "simd dot mismatch");
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let logits = vec![1000.0, 1010.0, 1000.0];
        let result = softmax_scalar(&logits);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={} != 1.0", sum);
        assert!(result[1] > 0.5, "softmax should peak at max logit");
    }

    #[test]
    fn test_softmax_consistency() {
        let logits = random_vector(128);
        let result = softmax_scalar(&logits);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={} != 1.0", sum);
        for &v in &result {
            assert!(v >= 0.0 && v <= 1.0, "softmax value {} out of range", v);
        }
    }

    #[test]
    fn test_gemm_cpu_batched_consistency() {
        let batch = 4;
        let m = 4;
        let k = 32;
        let weights_f32 = random_vector(m * k);
        let packed = PackedTensor::<F32x1>::from_f32_slice(&weights_f32, &[m, k], 1.0, 0.0);

        let batch_inputs: Vec<Vec<f32>> = (0..batch).map(|_| random_vector(k)).collect();
        let mut outputs = vec![vec![0.0f32; m]; batch];

        gemm_cpu(&packed, &batch_inputs, &mut outputs);

        for (bi, input) in batch_inputs.iter().enumerate() {
            for row in 0..m {
                let dot: f32 = (0..k).map(|j| weights_f32[row * k + j] * input[j]).sum();
                assert!(
                    (outputs[bi][row] - dot).abs() < 1e-3,
                    "batch {} row {} mismatch: {} vs {}",
                    bi,
                    row,
                    outputs[bi][row],
                    dot
                );
            }
        }
    }

    #[test]
    fn test_gemv_packed_tiled_consistency() {
        let m = 8;
        let k = 128;
        let weights_f32 = random_vector(m * k);
        let activations = random_vector(k);

        let packed = PackedTensor::<F32x1>::from_f32_slice(&weights_f32, &[m, k], 1.0, 0.0);
        let mut output_tiled = vec![0.0f32; m];
        gemv_packed_tiled(&packed, &activations, &mut output_tiled);

        for row in 0..m {
            let dot: f32 = (0..k)
                .map(|j| weights_f32[row * k + j] * activations[j])
                .sum();
            assert!(
                (output_tiled[row] - dot).abs() < 1e-3,
                "row {} mismatch: {} vs {}",
                row,
                output_tiled[row],
                dot
            );
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[test]
    fn test_neon_compiles_and_dispatches() {
        eprintln!(
            "NEON arch detected — verifying neon feature: {}",
            cfg!(feature = "neon")
        );
        let a = random_vector(128);
        let b = random_vector(128);
        let _result = fma_f32_slice(&a, &b);
    }
}
