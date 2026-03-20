//! SWAR (SIMD Within A Register) operations for 16-bit values.
//! Since F16/BF16 are floating point, SWAR add isn't as useful (bit patterns
//! don't add cleanly). These ops are provided for integer-oriented use cases
//! and element-wise operations like relu that can work on sign bits.

/// Mask for the low 16-bit half
pub const U16_LO: u32 = 0x0000_FFFF;
/// Mask for the high 16-bit half
pub const U16_HI: u32 = 0xFFFF_0000;
/// Sign bit mask for each 16-bit lane (bit 15 of each half)
pub const U16_SIGN: u32 = 0x8000_8000;

/// SWAR element-wise max for two F16x2 words (treating bits as unsigned 16-bit).
/// This is approximate — proper F16 max requires unpacking.
#[inline]
pub fn swar_max_u16x2(a: u32, b: u32) -> u32 {
    let a_lo = a & U16_LO;
    let b_lo = b & U16_LO;
    // Compare low halves: if a >= b, no borrow
    let diff_lo = a_lo.wrapping_add(U16_LO).wrapping_sub(b_lo);
    let borrow_lo = (diff_lo >> 16) & 1;
    let mask_lo = 0u32.wrapping_sub(borrow_lo); // 0xFFFF if borrow, 0x0000 if no borrow
    let result_lo = (a_lo & !mask_lo) | (b_lo & mask_lo);

    let a_hi = a & U16_HI;
    let b_hi = b & U16_HI;
    // Compare high halves
    let a_hi_shifted = a >> 16;
    let b_hi_shifted = b >> 16;
    let result_hi = if a_hi_shifted >= b_hi_shifted {
        a_hi
    } else {
        b_hi
    };

    result_lo | result_hi
}

/// SWAR ReLU backward for F16x2 — blocks gradient where sign bit is set.
/// This is approximate for F16 (sign bit layout matches IEEE 754).
#[inline]
pub fn swar_relu_backward_f16x2(grad: u32, pre_relu: u32) -> u32 {
    let sign_bits = pre_relu & U16_SIGN;
    // Spread sign bit to fill each 16-bit half
    let neg_mask = sign_bits
        | (sign_bits >> 1)
        | (sign_bits >> 2)
        | (sign_bits >> 3)
        | (sign_bits >> 4)
        | (sign_bits >> 5)
        | (sign_bits >> 6)
        | (sign_bits >> 7)
        | (sign_bits >> 8)
        | (sign_bits >> 9)
        | (sign_bits >> 10)
        | (sign_bits >> 11)
        | (sign_bits >> 12)
        | (sign_bits >> 13)
        | (sign_bits >> 14)
        | (sign_bits >> 15);
    grad & !neg_mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_swar_relu_backward_f16x2() {
        // Pack two f16 values: 1.5 (positive) and -2.5 (negative)
        let pos = f16::from_f32(1.5).to_bits() as u32;
        let neg = f16::from_f32(-2.5).to_bits() as u32;
        let pre_relu = pos | (neg << 16);

        let grad = 0xFFFF_FFFFu32;
        let result = swar_relu_backward_f16x2(grad, pre_relu);

        let result_lo = result & 0xFFFF;
        let result_hi = (result >> 16) & 0xFFFF;

        assert_eq!(result_lo, 0xFFFF); // positive, gradient passes
        assert_eq!(result_hi, 0x0000); // negative, gradient blocked
    }
}
