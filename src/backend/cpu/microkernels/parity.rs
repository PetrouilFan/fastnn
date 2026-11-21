//! Table-driven parity between the pointwise AVX2 kernels and their scalar
//! references.
//!
//! Two numerical defects have shipped from this repository, and both had the
//! same shape: a AVX2 kernel that silently disagreed with the scalar path. The
//! stride-2 `im2col` gather emitted the wrong lanes (39 % of a column matrix
//! wrong), and the AVX2 `ln` polynomial returned `ln(2) = 0.69804` instead of
//! `0.69315`, which made every `softplus` in a model wrong by `4.9e-3`. Each
//! was found by an ad-hoc probe and then pinned by one hand-written test that
//! protects that single kernel and nothing else.
//!
//! This module turns the check into a property of the table. A kernel listed in
//! a `pointwise_parity_cases!` invocation is compared against its scalar
//! reference over several lengths — including lengths that leave a remainder —
//! and over the input range where it is mathematically defined. Adding a kernel
//! to the table is one line, and the table is the thing a reviewer can read to
//! see what is covered.

use super::has_avx2;

/// Input range a pointwise kernel is defined over.
#[derive(Clone, Copy)]
enum Domain {
    /// Any finite value in `[-12, 12]`, plus signed zeros.
    Signed,
    /// Strictly positive values, for `ln` and `sqrt`.
    Positive,
    /// Finite values plus the specials a masked or comparison kernel must
    /// handle: `NaN`, both infinities and both zeros.
    WithSpecials,
}

/// Lengths chosen to run the 8-lane vector body and every remainder class.
const LENGTHS: [usize; 11] = [1, 3, 7, 8, 9, 15, 16, 17, 33, 64, 129];

fn inputs(domain: Domain, length: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(length);
    for index in 0..length {
        let position = index as f32 / (length.max(2) - 1) as f32;
        let value = match domain {
            Domain::Signed => -12.0 + 24.0 * position,
            Domain::Positive => 10.0f32.powf(-6.0 + 9.0 * position),
            Domain::WithSpecials => -6.0 + 12.0 * position,
        };
        values.push(value);
    }
    match domain {
        Domain::Signed => {
            values[0] = 0.0;
            if length > 1 {
                values[1] = -0.0;
            }
        }
        Domain::WithSpecials => {
            let specials = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
            for (slot, special) in specials.iter().enumerate() {
                if slot < length {
                    values[slot] = *special;
                }
            }
        }
        Domain::Positive => {}
    }
    values
}

/// Compare one kernel against its scalar reference.
///
/// `tolerance` is mixed absolute/relative: the error must stay within
/// `tolerance * max(|want|, 1)`, so a result near zero is held to the absolute
/// bound rather than an unachievable relative one. `0.0` demands bit-identical
/// results, which is what a kernel built from copies, comparisons or exact
/// rounding must achieve.
///
/// A transcendental row is given `1e-5` rather than the smallest number that
/// happens to pass: the two defects this harness exists for were `39 %` and
/// `4.9e-3`, three orders of magnitude above that bound, while a polynomial
/// `exp` legitimately lands a few ulp from `f32::exp`. Tightening a row below
/// `1e-5` only makes it fail when an input length changes.
fn check(
    label: &str,
    domain: Domain,
    tolerance: f32,
    avx2: impl Fn(&[f32], &mut [f32]),
    scalar: impl Fn(f32) -> f32,
) {
    for length in LENGTHS {
        let input = inputs(domain, length);
        let mut output = vec![f32::NAN; length];
        avx2(&input, &mut output);
        for (index, (&value, &got)) in input.iter().zip(output.iter()).enumerate() {
            let want = scalar(value);
            let agrees = if want.is_nan() {
                got.is_nan()
            } else if want.is_infinite() {
                got == want
            } else if tolerance == 0.0 {
                got == want
            } else {
                let error = (got - want).abs();
                error <= tolerance * want.abs().max(1.0)
            };
            assert!(
                agrees,
                "{label}: length {length}, index {index}, x = {value}, \
                 scalar = {want}, avx2 = {got}, tolerance = {tolerance}"
            );
        }
        // A kernel must not write past the slice it was handed; the Vec trip
        // above would have caught a shorter write, and this checks the length is
        // untouched for the next iteration.
        assert_eq!(output.len(), length, "{label}: length changed");
    }
}

/// One row per pointwise kernel with the uniform
/// `(input: &[f32], output: &mut [f32])` signature.
macro_rules! pointwise_parity_cases {
    ($(
        $test:ident, $label:literal, $domain:ident, $tolerance:expr, $avx2:ident, $scalar:ident;
    )*) => {
        $(
            #[test]
            fn $test() {
                if !has_avx2() {
                    return;
                }
                check(
                    $label,
                    Domain::$domain,
                    $tolerance,
                    |input, output| unsafe { $avx2(input, output) },
                    $scalar,
                );
            }
        )*
    };
}

mod table {
    use super::super::activations::*;
    use super::*;

    pointwise_parity_cases! {
        relu_matches_its_scalar_reference, "relu", Signed, 0.0,
            relu_f32_avx2, relu_f32_scalar;
        neg_matches_its_scalar_reference, "neg", Signed, 0.0,
            neg_f32_avx2, neg_f32_scalar;
        abs_matches_its_scalar_reference, "abs", Signed, 0.0,
            abs_f32_avx2, abs_f32_scalar;
        sqrt_matches_its_scalar_reference, "sqrt", Positive, 1.0e-5,
            sqrt_f32_avx2, sqrt_f32_scalar;
        sign_matches_its_scalar_reference, "sign", Signed, 0.0,
            sign_f32_avx2, sign_f32_scalar;
        round_matches_its_scalar_reference, "round", Signed, 0.0,
            round_f32_avx2, round_f32_scalar;
        logical_not_matches_its_scalar_reference, "logical_not", WithSpecials, 0.0,
            logical_not_f32_avx2, logical_not_f32_scalar;
        is_nan_matches_its_scalar_reference, "is_nan", WithSpecials, 0.0,
            is_nan_f32_avx2, is_nan_f32_scalar;
        exp_matches_its_scalar_reference, "exp", Signed, 1.0e-5,
            exp_f32_avx2, exp_f32_scalar;
        sigmoid_matches_its_scalar_reference, "sigmoid", Signed, 1.0e-5,
            sigmoid_f32_avx2, sigmoid_f32_scalar;
        tanh_matches_its_scalar_reference, "tanh", Signed, 1.0e-5,
            tanh_f32_avx2, tanh_f32_scalar;
        log_matches_its_scalar_reference, "log", Positive, 1.0e-5,
            log_f32_avx2, log_f32_scalar;
        elu_matches_its_scalar_reference, "elu", Signed, 1.0e-5,
            elu_f32_avx2, elu_f32_scalar;
        gelu_matches_its_scalar_reference, "gelu", Signed, 1.0e-5,
            gelu_f32_avx2, gelu_f32_scalar;
        silu_matches_its_scalar_reference, "silu", Signed, 1.0e-5,
            silu_f32_avx2, silu_f32_scalar;
        softplus_matches_its_scalar_reference, "softplus", Signed, 1.0e-5,
            softplus_f32_avx2, softplus_f32_scalar;
        // The vector path multiplies by 1/6 where the scalar divides by 6, so
        // the two differ in the last place even though the algebra is exact.
        hardswish_matches_its_scalar_reference, "hardswish", Signed, 1.0e-6,
            hardswish_f32_avx2, hardswish_f32_scalar;
        mish_matches_its_scalar_reference, "mish", Signed, 1.0e-5,
            mish_f32_avx2, mish_f32_scalar;
    }

    /// `clamp` and `leaky_relu` carry parameters, so they state their own
    /// closure pair instead of fitting the uniform macro.
    #[test]
    fn clamp_matches_its_scalar_reference() {
        if !has_avx2() {
            return;
        }
        let (low, high) = (-2.5f32, 2.5f32);
        check(
            "clamp",
            Domain::Signed,
            0.0,
            |input, output| unsafe { clamp_f32_avx2(input, output, low, high) },
            |x| clamp_f32_scalar(x, low, high),
        );
    }

    #[test]
    fn leaky_relu_matches_its_scalar_reference() {
        if !has_avx2() {
            return;
        }
        let slope = 0.01f32;
        check(
            "leaky_relu",
            Domain::Signed,
            0.0,
            |input, output| unsafe { leaky_relu_f32_avx2(input, output, slope) },
            |x| leaky_relu_f32_scalar(x, slope),
        );
    }
}
