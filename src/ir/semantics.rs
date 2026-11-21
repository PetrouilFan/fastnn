//! Scalar operator semantics that must be identical at compile time and run time.
//!
//! Two independent implementations of the same arithmetic live in this crate:
//! the constant folder ([`crate::compiler::passes::constant_folding`]), which
//! evaluates an operator on literal inputs while a graph is built, and the CPU
//! kernels, which evaluate it on tensor data afterwards. The two have to agree,
//! because otherwise the value of an expression depends on whether its input
//! happened to be folded away.
//!
//! The Rust standard library is not a safe place to take the shared definition
//! from. `f32::round` rounds a half *away from zero* and `f32::signum` returns
//! ±1 for ±0.0, whereas ONNX specifies halves-to-even for `Round` and
//! `sign(0) == 0` for `Sign`. A folder written with the `std` definition
//! therefore produced a different constant than the kernel produced for the
//! same tensor — `Round(2.5)` folded to 3 where ONNX and the AVX2 kernel say 2,
//! and `Sign(0.0)` folded to 1 where ONNX and the AVX2 kernel say 0.
//!
//! These functions are the shared definition. They live in the IR, next to the
//! opcodes they implement, because the compiler and every backend need them and
//! neither may depend on the other.

/// `Opcode::Sign` on a float, as ONNX defines it: 1 above zero, -1 below, 0 at
/// both zeros, NaN propagated.
///
/// The sign *bit* is deliberately not consulted: `-0.0` is a zero, so its sign
/// is 0. `f32::signum` returns -1 there, which is a different function.
#[inline]
#[must_use]
pub fn sign_f32(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x.is_nan() {
        f32::NAN
    } else {
        0.0
    }
}

/// `Opcode::Round` on a float, as ONNX defines it: *"in case of halves, the rule
/// is to round them to the nearest even integer"*, so `round(2.5) == 2`,
/// `round(-2.5) == -2` and `round([-4.5]) == [-4]`.
///
/// `f32::round` rounds a half away from zero instead, so it is only correct for
/// the values that are not exact halves.
#[inline]
#[must_use]
pub fn round_f32(x: f32) -> f32 {
    let truncated = x.trunc();
    if (x - truncated).abs() != 0.5 {
        return x.round();
    }
    if (truncated / 2.0).fract() == 0.0 {
        truncated
    } else {
        truncated + x.signum()
    }
}

#[cfg(test)]
mod tests {
    use super::{round_f32, sign_f32};

    #[test]
    fn sign_ignores_the_sign_bit_of_zero() {
        assert_eq!(sign_f32(0.0), 0.0);
        assert_eq!(sign_f32(-0.0), 0.0);
        assert_eq!(sign_f32(2.0), 1.0);
        assert_eq!(sign_f32(-2.0), -1.0);
        assert!(sign_f32(f32::NAN).is_nan());
    }

    #[test]
    fn round_sends_halves_to_the_nearest_even_integer() {
        assert_eq!(round_f32(2.5), 2.0);
        assert_eq!(round_f32(3.5), 4.0);
        assert_eq!(round_f32(-2.5), -2.0);
        assert_eq!(round_f32(-4.5), -4.0);
        assert_eq!(round_f32(2.4), 2.0);
        assert_eq!(round_f32(2.6), 3.0);
        assert_eq!(round_f32(-2.6), -3.0);
        assert_eq!(round_f32(0.0), 0.0);
        assert_eq!(round_f32(-0.0), -0.0);
    }
}
