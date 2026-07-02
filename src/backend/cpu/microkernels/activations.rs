//! CPU activation microkernels — extracted from microkernels.rs

#![allow(dead_code)]

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

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
pub fn relu_f32_scalar(x: f32) -> f32 {
    x.max(0.0)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and each at least 8 elements long.
pub unsafe fn relu_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    while i + 8 <= len {
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_max_ps(_mm256_loadu_ps(input.as_ptr().add(i)), zero),
        );
        i += 8;
    }
    for j in i..len {
        output[j] = relu_f32_scalar(input[j]);
    }
}

#[inline]
pub fn neg_f32_scalar(x: f32) -> f32 {
    -x
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as relu_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
pub unsafe fn neg_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let sign_mask = _mm256_set1_ps(-0.0f32);
    while i + 8 <= len {
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_xor_ps(_mm256_loadu_ps(input.as_ptr().add(i)), sign_mask),
        );
        i += 8;
    }
    for j in i..len {
        output[j] = neg_f32_scalar(input[j]);
    }
}

#[inline]
pub fn abs_f32_scalar(x: f32) -> f32 {
    x.abs()
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as relu_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
pub unsafe fn abs_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let inv_sign_mask = _mm256_set1_ps(f32::from_bits(0x7fff_ffff));
    while i + 8 <= len {
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_and_ps(_mm256_loadu_ps(input.as_ptr().add(i)), inv_sign_mask),
        );
        i += 8;
    }
    for j in i..len {
        output[j] = abs_f32_scalar(input[j]);
    }
}

#[inline]
pub fn sqrt_f32_scalar(x: f32) -> f32 {
    x.sqrt()
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as relu_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
pub unsafe fn sqrt_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    while i + 8 <= len {
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_sqrt_ps(_mm256_loadu_ps(input.as_ptr().add(i))),
        );
        i += 8;
    }
    for j in i..len {
        output[j] = sqrt_f32_scalar(input[j]);
    }
}

#[inline]
pub fn sign_f32_scalar(x: f32) -> f32 {
    x.signum()
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as relu_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
pub unsafe fn sign_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_or_ps(
                _mm256_and_ps(_mm256_cmp_ps::<{ _CMP_GT_OQ }>(x, zero), one),
                _mm256_and_ps(_mm256_cmp_ps::<{ _CMP_LT_OQ }>(x, zero), neg_one),
            ),
        );
        i += 8;
    }
    for j in i..len {
        output[j] = sign_f32_scalar(input[j]);
    }
}

#[inline]
pub fn round_f32_scalar(x: f32) -> f32 {
    x.round()
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as relu_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
pub unsafe fn round_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    while i + 8 <= len {
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_round_ps(
                _mm256_loadu_ps(input.as_ptr().add(i)),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            ),
        );
        i += 8;
    }
    for j in i..len {
        output[j] = round_f32_scalar(input[j]);
    }
}

#[inline]
pub fn logical_not_f32_scalar(x: f32) -> f32 {
    if x == 0.0 {
        1.0
    } else {
        0.0
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as relu_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
pub unsafe fn logical_not_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    while i + 8 <= len {
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_and_ps(
                _mm256_cmp_ps::<{ _CMP_EQ_OQ }>(_mm256_loadu_ps(input.as_ptr().add(i)), zero),
                one,
            ),
        );
        i += 8;
    }
    for j in i..len {
        output[j] = logical_not_f32_scalar(input[j]);
    }
}

// ── Parametric ops ───────────────────────────────────────────

#[inline]
pub fn clamp_f32_scalar(x: f32, min_val: f32, max_val: f32) -> f32 {
    x.max(min_val).min(max_val)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and each at least 8 elements long.
pub unsafe fn clamp_f32_avx2(input: &[f32], output: &mut [f32], min_val: f32, max_val: f32) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let vmin = _mm256_set1_ps(min_val);
    let vmax = _mm256_set1_ps(max_val);
    while i + 8 <= len {
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_min_ps(
                _mm256_max_ps(_mm256_loadu_ps(input.as_ptr().add(i)), vmin),
                vmax,
            ),
        );
        i += 8;
    }
    for j in i..len {
        output[j] = clamp_f32_scalar(input[j], min_val, max_val);
    }
}

#[inline]
pub fn leaky_relu_f32_scalar(x: f32, slope: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        x * slope
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as clamp_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
pub unsafe fn leaky_relu_f32_avx2(input: &[f32], output: &mut [f32], slope: f32) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let zero = _mm256_setzero_ps();
    let vslope = _mm256_set1_ps(slope);
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_blendv_ps(
                _mm256_mul_ps(x, vslope),
                x,
                _mm256_cmp_ps::<{ _CMP_GT_OQ }>(x, zero),
            ),
        );
        i += 8;
    }
    for j in i..len {
        output[j] = leaky_relu_f32_scalar(input[j], slope);
    }
}

// ── Complex ops (SIMD struct load/store, per-element scalar math) ───

/// Macro: unrolls 8-element SIMD struct loads/stores, calling `$scalar_fn` per lane.
#[allow(unused_macros)]
macro_rules! avx2_elwise_fallback {
    ($name:ident, $scalar_fn:expr) => {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        // SAFETY: Same as other SIMD elementwise ops — caller ensures valid, non-overlapping
        // input/output slices with sufficient length.
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
                *output.as_mut_ptr().add(i) = $scalar_fn(x0);
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

#[inline]
pub fn exp_f32_scalar(x: f32) -> f32 {
    x.exp()
}

#[inline]
pub fn sigmoid_f32_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn tanh_f32_scalar(x: f32) -> f32 {
    x.tanh()
}

// ============================================================
// SIMD vector helpers — reusable across activation microkernels
// ============================================================

/// Vector exp(x) for a single `__m256`. Degree-5 polynomial approximation
/// with range reduction via `x·log2(e)`, Horner with FMA, 2^k reconstruction.
/// Relative error < 2e-5 over [-88, 88].
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: `x` is a valid __m256 register value; this helper is always called
// from within a function that guarantees AVX2+FMA availability.
pub(crate) unsafe fn exp_avx2_vec(x: __m256) -> __m256 {
    let vlog2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let vln2_hi = _mm256_set1_ps(0.693_359_4_f32);
    let vln2_lo = _mm256_set1_ps(-2.121_944_4e-4_f32);
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
    let exp_factor = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(k_clamped, _mm256_set1_epi32(127)),
        23,
    ));
    _mm256_mul_ps(p, exp_factor)
}

/// Vector tanh(x) for a single `__m256`. Uses identity: tanh(x) = 2·sigmoid(2x) - 1.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: Same as exp_avx2_vec — `x` is a valid __m256, AVX2+FMA is available.
pub(crate) unsafe fn tanh_avx2_vec(x: __m256) -> __m256 {
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
// SAFETY: Same as exp_avx2_vec — `x` is a valid __m256, AVX2+FMA is available.
pub(crate) unsafe fn log_avx2_vec(x: __m256) -> __m256 {
    // Extract exponent: e = floor(log2(x))
    let raw = _mm256_castps_si256(x);
    let biased_exp = _mm256_srli_epi32(raw, 23);
    let exp_i = _mm256_sub_epi32(biased_exp, _mm256_set1_epi32(127));
    let exp_f = _mm256_cvtepi32_ps(exp_i);

    // Extract mantissa in [1, 2): clear exponent, set to 127 (bias)
    let mant_raw = _mm256_and_si256(raw, _mm256_set1_epi32(0x007FFFFF));
    let mant_f = _mm256_castsi256_ps(_mm256_or_si256(mant_raw, _mm256_set1_epi32(0x3F800000)));

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
    let _m4 = _mm256_mul_ps(m3, mant_f);

    // Horner: (((c4*m + c3)*m + c2)*m + c1)*m + c0
    let mut log_m = _mm256_fmadd_ps(c4, mant_f, c3);
    log_m = _mm256_fmadd_ps(log_m, mant_f, c2);
    log_m = _mm256_fmadd_ps(log_m, mant_f, c1);
    log_m = _mm256_fmadd_ps(log_m, mant_f, c0);

    // ln(x) = ln(m * 2^e) = ln(m) + e * ln(2)
    _mm256_fmadd_ps(exp_f, _mm256_set1_ps(std::f32::consts::LN_2), log_m)
}

// ============================================================
// Public AVX2 activation microkernels — all use the helpers above
// ============================================================

/// True AVX2 exp(x) via polynomial approximation.
/// Delegates to `exp_avx2_vec` for the vectorized computation.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and each at least 8 elements long.
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

#[inline]
pub fn log_f32_scalar(x: f32) -> f32 {
    x.ln()
}

/// True AVX2 ln(x) using frexp decomposition + polynomial on [1, 2).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and each at least 8 elements long.
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
// SAFETY: Caller must ensure `input` and `output` are valid, non-overlapping,
// and each at least 8 elements long.
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
// SAFETY: Same as sigmoid_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
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

#[inline]
pub fn elu_f32_scalar(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        x.exp() - 1.0
    }
}

/// True AVX2 ELU(x) = if x > 0 then x else exp(x) - 1.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as sigmoid_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
pub unsafe fn elu_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let vzero = _mm256_setzero_ps();
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let exp_x = exp_avx2_vec(x);
        let exp_minus_1 = _mm256_sub_ps(exp_x, _mm256_set1_ps(1.0f32));
        // blend: when x > 0, select x; otherwise select exp(x)-1
        let mask = _mm256_cmp_ps::<{ _CMP_GT_OQ }>(x, vzero);
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_blendv_ps(exp_minus_1, x, mask),
        );
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
// SAFETY: Same as sigmoid_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
pub unsafe fn gelu_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let vhalf = _mm256_set1_ps(0.5f32);
    let vone = _mm256_set1_ps(1.0f32);
    let vsqrt2pi = _mm256_set1_ps(0.797_884_6_f32); // √(2/π)
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

#[inline]
pub fn silu_f32_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// True AVX2 SiLU(x) = x * sigmoid(x).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as sigmoid_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
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

#[inline]
pub fn softplus_f32_scalar(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

/// True AVX2 softplus(x) = ln(1 + exp(x)).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as sigmoid_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
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

#[inline]
pub fn hardswish_f32_scalar(x: f32) -> f32 {
    x * (x + 3.0).clamp(0.0, 6.0) / 6.0
}

/// True AVX2 hardswish(x) = x * clamp(x+3, 0, 6) / 6.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as sigmoid_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
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
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_mul_ps(_mm256_mul_ps(x, clamped), vinvsix),
        );
        i += 8;
    }
    for j in i..len {
        let x = *input.as_ptr().add(j);
        *output.as_mut_ptr().add(j) = x * (x + 3.0).clamp(0.0, 6.0) / 6.0;
    }
}

#[inline]
pub fn mish_f32_scalar(x: f32) -> f32 {
    let sp = (1.0 + x.exp()).ln();
    x * sp.tanh()
}

/// True AVX2 mish(x) = x * tanh(ln(1 + exp(x))).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Same as sigmoid_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
pub unsafe fn mish_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    let mut i = 0;
    let vone = _mm256_set1_ps(1.0f32);
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let sp = log_avx2_vec(_mm256_add_ps(vone, exp_avx2_vec(x))); // ln(1+exp(x))
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_mul_ps(x, tanh_avx2_vec(sp)),
        );
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
// SAFETY: Same as sigmoid_f32_avx2 — caller ensures valid, non-overlapping
// input/output slices with at least 8 elements.
pub unsafe fn log_softmax_f32_avx2(input: &[f32], output: &mut [f32]) {
    let len = output.len().min(input.len());
    if len == 0 {
        return;
    }
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
        if candidate > max_val {
            max_val = candidate;
        }
        i += 8;
    }
    for j in i..len {
        let v = *input.as_ptr().add(j);
        if v > max_val {
            max_val = v;
        }
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
    if len == 0 {
        return;
    }
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &x in &input[..len] {
        sum += ((x - max_val).exp()) as f64;
    }
    let log_sum = (sum as f32).ln();
    for i in 0..len {
        output[i] = input[i] - max_val - log_sum;
    }
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
        // SAFETY: Same as other SIMD broadcast ops — caller ensures valid, non-overlapping
        // slices with sufficient length for the operation (either dense or broadcast).
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
            } else if b_len == len && a_len > 1 && len % a_len == 0 {
                let tile = a_len;
                let num_tiles = len / tile;
                for t in 0..num_tiles {
                    let chunk_start = t * tile;
                    let mut j = chunk_start;
                    while j + 8 <= chunk_start + tile {
                        let va = _mm256_loadu_ps(a.as_ptr().add(j - chunk_start));
                        let vb = _mm256_loadu_ps(b.as_ptr().add(j));
                        _mm256_storeu_ps(output.as_mut_ptr().add(j), $avx2_intrin(va, vb));
                        j += 8;
                    }
                    for r in j..chunk_start + tile {
                        *output.as_mut_ptr().add(r) = a[r - chunk_start] $op b[r];
                    }
                }
                i = len;
            } else if a_len == len && b_len > 1 && len % b_len == 0 {
                let tile = b_len;
                let num_tiles = len / tile;
                for t in 0..num_tiles {
                    let chunk_start = t * tile;
                    let mut j = chunk_start;
                    while j + 8 <= chunk_start + tile {
                        let va = _mm256_loadu_ps(a.as_ptr().add(j));
                        let vb = _mm256_loadu_ps(b.as_ptr().add(j - chunk_start));
                        _mm256_storeu_ps(output.as_mut_ptr().add(j), $avx2_intrin(va, vb));
                        j += 8;
                    }
                    for r in j..chunk_start + tile {
                        *output.as_mut_ptr().add(r) = a[r] $op b[r - chunk_start];
                    }
                }
                i = len;
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
        // SAFETY: Same as other SIMD broadcast ops — caller ensures valid, non-overlapping
        // slices with sufficient length for the operation (either dense or broadcast).
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
            } else if b_len == len && a_len > 1 && len % a_len == 0 {
                let tile = a_len;
                let num_tiles = len / tile;
                for t in 0..num_tiles {
                    let chunk_start = t * tile;
                    let mut j = chunk_start;
                    while j + 8 <= chunk_start + tile {
                        let va = _mm256_loadu_ps(a.as_ptr().add(j - chunk_start));
                        let vb = _mm256_loadu_ps(b.as_ptr().add(j));
                        _mm256_storeu_ps(output.as_mut_ptr().add(j), $avx2_intrin(va, vb));
                        j += 8;
                    }
                    for r in j..chunk_start + tile {
                        *output.as_mut_ptr().add(r) = a[r - chunk_start].$method(b[r]);
                    }
                }
                i = len;
            } else if a_len == len && b_len > 1 && len % b_len == 0 {
                let tile = b_len;
                let num_tiles = len / tile;
                for t in 0..num_tiles {
                    let chunk_start = t * tile;
                    let mut j = chunk_start;
                    while j + 8 <= chunk_start + tile {
                        let va = _mm256_loadu_ps(a.as_ptr().add(j));
                        let vb = _mm256_loadu_ps(b.as_ptr().add(j - chunk_start));
                        _mm256_storeu_ps(output.as_mut_ptr().add(j), $avx2_intrin(va, vb));
                        j += 8;
                    }
                    for r in j..chunk_start + tile {
                        *output.as_mut_ptr().add(r) = a[r].$method(b[r - chunk_start]);
                    }
                }
                i = len;
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
impl_binary_minmax_broadcast!(
    max_f32_scalar_broadcast,
    max_f32_avx2_broadcast,
    max,
    _mm256_max_ps
);
impl_binary_minmax_broadcast!(
    min_f32_scalar_broadcast,
    min_f32_avx2_broadcast,
    min,
    _mm256_min_ps
);

// ═══════════════════════════════════════════════════════════════
// Fused binary + activation — single-pass kernels
// Combines elementwise op (add/mul) + activation (relu/silu/gelu)
// in one AVX2 pass, halving memory bandwidth vs two separate kernels.
// ═══════════════════════════════════════════════════════════════

// ── Fused: add + relu ──────────────────────────────────────────

/// add + ReLU: out[i] = max(0, a[i % a_len] + b[i % b_len])
#[inline]
pub fn fused_add_relu_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let a_len = a.len();
    let b_len = b.len();
    let len = out.len();
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if len >= 8 {
        unsafe { return fused_add_relu_f32_avx2(a, b, out) };
    }
    for i in 0..len {
        out[i] = relu_f32_scalar(a[i % a_len] + b[i % b_len]);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must ensure slices are valid, non-overlapping, and at least 8 elements.
pub unsafe fn fused_add_relu_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len();
    let a_len = a.len();
    let b_len = b.len();
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    if a_len == len && b_len == len {
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_max_ps(_mm256_add_ps(va, vb), zero),
            );
            i += 8;
        }
    } else if a_len == 1 && b_len == len {
        let va = _mm256_broadcast_ss(&a[0]);
        while i + 8 <= len {
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_max_ps(_mm256_add_ps(va, vb), zero),
            );
            i += 8;
        }
    } else if a_len == len && b_len == 1 {
        let vb = _mm256_broadcast_ss(&b[0]);
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_max_ps(_mm256_add_ps(va, vb), zero),
            );
            i += 8;
        }
    } else if b_len == len && a_len > 1 && len.is_multiple_of(a_len) {
        let tile = a_len;
        for t in 0..len / tile {
            let cs = t * tile;
            let mut j = cs;
            while j + 8 <= cs + tile {
                let va = _mm256_loadu_ps(a.as_ptr().add(j - cs));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j));
                _mm256_storeu_ps(
                    out.as_mut_ptr().add(j),
                    _mm256_max_ps(_mm256_add_ps(va, vb), zero),
                );
                j += 8;
            }
            for r in j..cs + tile {
                *out.as_mut_ptr().add(r) = relu_f32_scalar(a[r - cs] + b[r]);
            }
        }
        i = len;
    } else if a_len == len && b_len > 1 && len.is_multiple_of(b_len) {
        let tile = b_len;
        for t in 0..len / tile {
            let cs = t * tile;
            let mut j = cs;
            while j + 8 <= cs + tile {
                let va = _mm256_loadu_ps(a.as_ptr().add(j));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j - cs));
                _mm256_storeu_ps(
                    out.as_mut_ptr().add(j),
                    _mm256_max_ps(_mm256_add_ps(va, vb), zero),
                );
                j += 8;
            }
            for r in j..cs + tile {
                *out.as_mut_ptr().add(r) = relu_f32_scalar(a[r] + b[r - cs]);
            }
        }
        i = len;
    }
    for j in i..len {
        *out.as_mut_ptr().add(j) = relu_f32_scalar(a[j % a_len] + b[j % b_len]);
    }
}

// ── Fused: mul + relu ──────────────────────────────────────────

/// mul + ReLU: out[i] = max(0, a[i % a_len] * b[i % b_len])
pub fn fused_mul_relu_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let a_len = a.len();
    let b_len = b.len();
    let len = out.len();
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if len >= 8 {
        unsafe { return fused_mul_relu_f32_avx2(a, b, out) };
    }
    for i in 0..len {
        out[i] = relu_f32_scalar(a[i % a_len] * b[i % b_len]);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn fused_mul_relu_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len();
    let a_len = a.len();
    let b_len = b.len();
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    if a_len == len && b_len == len {
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_max_ps(_mm256_mul_ps(va, vb), zero),
            );
            i += 8;
        }
    } else if a_len == 1 && b_len == len {
        let va = _mm256_broadcast_ss(&a[0]);
        while i + 8 <= len {
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_max_ps(_mm256_mul_ps(va, vb), zero),
            );
            i += 8;
        }
    } else if a_len == len && b_len == 1 {
        let vb = _mm256_broadcast_ss(&b[0]);
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_max_ps(_mm256_mul_ps(va, vb), zero),
            );
            i += 8;
        }
    } else if b_len == len && a_len > 1 && len.is_multiple_of(a_len) {
        let tile = a_len;
        for t in 0..len / tile {
            let cs = t * tile;
            let mut j = cs;
            while j + 8 <= cs + tile {
                let va = _mm256_loadu_ps(a.as_ptr().add(j - cs));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j));
                _mm256_storeu_ps(
                    out.as_mut_ptr().add(j),
                    _mm256_max_ps(_mm256_mul_ps(va, vb), zero),
                );
                j += 8;
            }
            for r in j..cs + tile {
                *out.as_mut_ptr().add(r) = relu_f32_scalar(a[r - cs] * b[r]);
            }
        }
        i = len;
    } else if a_len == len && b_len > 1 && len.is_multiple_of(b_len) {
        let tile = b_len;
        for t in 0..len / tile {
            let cs = t * tile;
            let mut j = cs;
            while j + 8 <= cs + tile {
                let va = _mm256_loadu_ps(a.as_ptr().add(j));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j - cs));
                _mm256_storeu_ps(
                    out.as_mut_ptr().add(j),
                    _mm256_max_ps(_mm256_mul_ps(va, vb), zero),
                );
                j += 8;
            }
            for r in j..cs + tile {
                *out.as_mut_ptr().add(r) = relu_f32_scalar(a[r] * b[r - cs]);
            }
        }
        i = len;
    }
    for j in i..len {
        *out.as_mut_ptr().add(j) = relu_f32_scalar(a[j % a_len] * b[j % b_len]);
    }
}

// ── Fused: add + silu ──────────────────────────────────────────

/// add + SiLU: out[i] = (a[i] + b[i]) / (1 + exp(-(a[i] + b[i])))
#[inline]
fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn fused_add_silu_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let a_len = a.len();
    let b_len = b.len();
    let len = out.len();
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if len >= 8 {
        unsafe { return fused_add_silu_f32_avx2(a, b, out) };
    }
    for i in 0..len {
        out[i] = silu_scalar(a[i % a_len] + b[i % b_len]);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn fused_add_silu_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len();
    let a_len = a.len();
    let b_len = b.len();
    let one = _mm256_set1_ps(1.0);
    let neg_zero = _mm256_set1_ps(-0.0f32);
    let mut i = 0;
    if a_len == len && b_len == len {
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            silu_avx2_core(va, vb, one, neg_zero, out, &mut i);
        }
    } else if a_len == 1 && b_len == len {
        let va = _mm256_broadcast_ss(&a[0]);
        while i + 8 <= len {
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            silu_avx2_core(va, vb, one, neg_zero, out, &mut i);
        }
    } else if a_len == len && b_len == 1 {
        let vb = _mm256_broadcast_ss(&b[0]);
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            silu_avx2_core(va, vb, one, neg_zero, out, &mut i);
        }
    } else if b_len == len && a_len > 1 && len.is_multiple_of(a_len) {
        let tile = a_len;
        for t in 0..len / tile {
            let cs = t * tile;
            let mut j = cs;
            while j + 8 <= cs + tile {
                let va = _mm256_loadu_ps(a.as_ptr().add(j - cs));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j));
                silu_avx2_core(va, vb, one, neg_zero, out, &mut j);
            }
            for r in j..cs + tile {
                *out.as_mut_ptr().add(r) = silu_scalar(a[r - cs] + b[r]);
            }
        }
        i = len;
    } else if a_len == len && b_len > 1 && len.is_multiple_of(b_len) {
        let tile = b_len;
        for t in 0..len / tile {
            let cs = t * tile;
            let mut j = cs;
            while j + 8 <= cs + tile {
                let va = _mm256_loadu_ps(a.as_ptr().add(j));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j - cs));
                silu_avx2_core(va, vb, one, neg_zero, out, &mut j);
            }
            for r in j..cs + tile {
                *out.as_mut_ptr().add(r) = silu_scalar(a[r] + b[r - cs]);
            }
        }
        i = len;
    }
    for j in i..len {
        *out.as_mut_ptr().add(j) = silu_scalar(a[j % a_len] + b[j % b_len]);
    }
}

/// Core AVX2 SiLU computation: out[i] = silu(va + vb), advances i by 8
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn silu_avx2_core(
    va: __m256,
    vb: __m256,
    one: __m256,
    neg_zero: __m256,
    out: &mut [f32],
    i: &mut usize,
) {
    let vsum = _mm256_add_ps(va, vb);
    let vneg = _mm256_xor_ps(vsum, neg_zero);
    let vexp = exp_avx2_vec(vneg);
    let vsigmoid = _mm256_div_ps(one, _mm256_add_ps(one, vexp));
    _mm256_storeu_ps(out.as_mut_ptr().add(*i), _mm256_mul_ps(vsum, vsigmoid));
    *i += 8;
}

// ── Fused: mul + silu ──────────────────────────────────────────

pub fn fused_mul_silu_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let a_len = a.len();
    let b_len = b.len();
    let len = out.len();
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if len >= 8 {
        unsafe { return fused_mul_silu_f32_avx2(a, b, out) };
    }
    for i in 0..len {
        out[i] = silu_scalar(a[i % a_len] * b[i % b_len]);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn fused_mul_silu_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len();
    let a_len = a.len();
    let b_len = b.len();
    let one = _mm256_set1_ps(1.0);
    let neg_zero = _mm256_set1_ps(-0.0f32);
    let mut i = 0;
    if a_len == len && b_len == len {
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            silu_mul_avx2_core(va, vb, one, neg_zero, out, &mut i);
        }
    } else if a_len == 1 && b_len == len {
        let va = _mm256_broadcast_ss(&a[0]);
        while i + 8 <= len {
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            silu_mul_avx2_core(va, vb, one, neg_zero, out, &mut i);
        }
    } else if a_len == len && b_len == 1 {
        let vb = _mm256_broadcast_ss(&b[0]);
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            silu_mul_avx2_core(va, vb, one, neg_zero, out, &mut i);
        }
    } else if b_len == len && a_len > 1 && len.is_multiple_of(a_len) {
        let tile = a_len;
        for t in 0..len / tile {
            let cs = t * tile;
            let mut j = cs;
            while j + 8 <= cs + tile {
                let va = _mm256_loadu_ps(a.as_ptr().add(j - cs));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j));
                silu_mul_avx2_core(va, vb, one, neg_zero, out, &mut j);
            }
            for r in j..cs + tile {
                *out.as_mut_ptr().add(r) = silu_scalar(a[r - cs] * b[r]);
            }
        }
        i = len;
    } else if a_len == len && b_len > 1 && len.is_multiple_of(b_len) {
        let tile = b_len;
        for t in 0..len / tile {
            let cs = t * tile;
            let mut j = cs;
            while j + 8 <= cs + tile {
                let va = _mm256_loadu_ps(a.as_ptr().add(j));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j - cs));
                silu_mul_avx2_core(va, vb, one, neg_zero, out, &mut j);
            }
            for r in j..cs + tile {
                *out.as_mut_ptr().add(r) = silu_scalar(a[r] * b[r - cs]);
            }
        }
        i = len;
    }
    for j in i..len {
        *out.as_mut_ptr().add(j) = silu_scalar(a[j % a_len] * b[j % b_len]);
    }
}

/// Core AVX2 mul+SiLU computation: out[i] = silu(va * vb), advances i by 8
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn silu_mul_avx2_core(
    va: __m256,
    vb: __m256,
    one: __m256,
    neg_zero: __m256,
    out: &mut [f32],
    i: &mut usize,
) {
    let vprod = _mm256_mul_ps(va, vb);
    let vneg = _mm256_xor_ps(vprod, neg_zero);
    let vexp = exp_avx2_vec(vneg);
    let vsigmoid = _mm256_div_ps(one, _mm256_add_ps(one, vexp));
    _mm256_storeu_ps(out.as_mut_ptr().add(*i), _mm256_mul_ps(vprod, vsigmoid));
    *i += 8;
}

// ── Fused: add + gelu ──────────────────────────────────────────

/// add + GELU: out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// where x = a[i] + b[i]
#[inline]
fn gelu_scalar(x: f32) -> f32 {
    let x3 = x * x * x;
    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
    0.5 * x * (1.0 + tanh_arg.tanh())
}

pub fn fused_add_gelu_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let a_len = a.len();
    let b_len = b.len();
    let len = out.len();
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if len >= 8 {
        unsafe { return fused_add_gelu_f32_avx2(a, b, out) };
    }
    for i in 0..len {
        out[i] = gelu_scalar(a[i % a_len] + b[i % b_len]);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn fused_add_gelu_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len();
    let a_len = a.len();
    let b_len = b.len();
    let c0 = _mm256_set1_ps(0.7978846);
    let c1 = _mm256_set1_ps(0.044715);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;
    if a_len == len && b_len == len {
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            add_gelu_avx2_core(va, vb, c0, c1, half, one, out, &mut i);
        }
    } else if a_len == 1 && b_len == len {
        let va = _mm256_broadcast_ss(&a[0]);
        while i + 8 <= len {
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            add_gelu_avx2_core(va, vb, c0, c1, half, one, out, &mut i);
        }
    } else if a_len == len && b_len == 1 {
        let vb = _mm256_broadcast_ss(&b[0]);
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            add_gelu_avx2_core(va, vb, c0, c1, half, one, out, &mut i);
        }
    } else if b_len == len && a_len > 1 && len.is_multiple_of(a_len) {
        let tile = a_len;
        for t in 0..len / tile {
            let cs = t * tile;
            let mut j = cs;
            while j + 8 <= cs + tile {
                let va = _mm256_loadu_ps(a.as_ptr().add(j - cs));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j));
                add_gelu_avx2_core(va, vb, c0, c1, half, one, out, &mut j);
            }
            for r in j..cs + tile {
                *out.as_mut_ptr().add(r) = gelu_scalar(a[r - cs] + b[r]);
            }
        }
        i = len;
    } else if a_len == len && b_len > 1 && len.is_multiple_of(b_len) {
        let tile = b_len;
        for t in 0..len / tile {
            let cs = t * tile;
            let mut j = cs;
            while j + 8 <= cs + tile {
                let va = _mm256_loadu_ps(a.as_ptr().add(j));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j - cs));
                add_gelu_avx2_core(va, vb, c0, c1, half, one, out, &mut j);
            }
            for r in j..cs + tile {
                *out.as_mut_ptr().add(r) = gelu_scalar(a[r] + b[r - cs]);
            }
        }
        i = len;
    }
    for j in i..len {
        *out.as_mut_ptr().add(j) = gelu_scalar(a[j % a_len] + b[j % b_len]);
    }
}

/// Core AVX2 add+GELU computation: out[i] = gelu(va + vb), advances i by 8
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn add_gelu_avx2_core(
    va: __m256,
    vb: __m256,
    c0: __m256,
    c1: __m256,
    half: __m256,
    one: __m256,
    out: &mut [f32],
    i: &mut usize,
) {
    let x = _mm256_add_ps(va, vb);
    let x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
    let tanh_arg = _mm256_mul_ps(c0, _mm256_add_ps(x, _mm256_mul_ps(c1, x3)));
    let t = tanh_avx2_vec(tanh_arg);
    _mm256_storeu_ps(
        out.as_mut_ptr().add(*i),
        _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, t))),
    );
    *i += 8;
}
