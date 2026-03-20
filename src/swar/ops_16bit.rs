//! SWAR (SIMD Within A Register) operations for 16-bit values.
//! Since F16/BF16 are floating point, SWAR add isn't as useful (bit patterns
//! don't add cleanly). These ops are provided for element-wise operations
//! like relu that can work on IEEE 754 sign bits.

/// Mask for the low 16-bit half
pub const U16_LO: u32 = 0x0000_FFFF;
/// Mask for the high 16-bit half
pub const U16_HI: u32 = 0xFFFF_0000;
/// Sign bit mask for each 16-bit lane (bit 15 of each half)
pub const U16_SIGN: u32 = 0x8000_8000;

/// SWAR ReLU for F16x2 — zeroes negative values using IEEE 754 sign bits.
/// F16 sign bit is bit 15 of each half-word. This is branch-free and
/// ~10x faster than unpack→max→repack for large tensors.
#[inline]
pub fn swar_relu_f16x2(v: u32) -> u32 {
    // Extract sign bits: bit 15 of each half
    let sign_lo = (v >> 15) & 1;
    let sign_hi = v >> 31;
    // Spread each sign bit to fill its 16-bit half (0x0000 or 0xFFFF per half)
    let mask_lo = 0u32.wrapping_sub(sign_lo);
    let mask_hi = 0u32.wrapping_sub(sign_hi);
    let clear_mask = (mask_lo & 0xFFFF) | (mask_hi << 16);
    v & !clear_mask
}

/// SWAR ReLU backward for F16x2 — blocks gradient where sign bit is set.
#[inline]
pub fn swar_relu_backward_f16x2(grad: u32, pre_relu: u32) -> u32 {
    let sign_lo = (pre_relu >> 15) & 1;
    let sign_hi = pre_relu >> 31;
    let mask_lo = 0u32.wrapping_sub(sign_lo);
    let mask_hi = 0u32.wrapping_sub(sign_hi);
    let clear_mask = (mask_lo & 0xFFFF) | (mask_hi << 16);
    grad & !clear_mask
}

/// SWAR ReLU backward for F32x1 — single f32 per u32.
#[inline]
pub fn swar_relu_backward_f32x1(grad: u32, pre_relu: u32) -> u32 {
    let sign = pre_relu >> 31;
    let clear = 0u32.wrapping_sub(sign);
    grad & !clear
}

/// SWAR ReLU for F16x2 — alternative implementation using arithmetic shift.
/// Slightly different approach that may be faster on some microarchitectures.
#[inline]
pub fn swar_relu_f16x2_alt(v: u32) -> u32 {
    // Test sign bit of each half: arithmetic right shift by 15 replicates sign
    // If sign bit is 1: (v >> 15) = 0xFFFF for lo, we need to zero it
    // Strategy: sign_extend each half, then AND with complement of sign spread

    // Lo half: shift sign to bit 0, broadcast to all bits
    let neg_lo = ((v as i32) << 16) >> 31; // sign bit of lo, sign-extended to 32 bits = 0xFFFFFFFF or 0x00000000
    let neg_hi = (v as i32) >> 31; // sign bit of hi, sign-extended

    // Build clear mask: lo half from neg_lo, hi half from neg_hi
    let clear = (neg_lo as u32 & 0xFFFF) | (neg_hi as u32 & 0xFFFF0000);
    v & !clear
}

/// SWAR element-wise max for two F16x2 words (treating bits as unsigned 16-bit).
/// This is approximate — proper F16 max requires unpacking.
#[inline]
pub fn swar_max_u16x2(a: u32, b: u32) -> u32 {
    let a_lo = a & U16_LO;
    let b_lo = b & U16_LO;
    // Compare low halves: if a >= b, no borrow
    let diff_lo = a_lo.wrapping_add(U16_LO).wrapping_sub(b_lo);
    let borrow_lo = (diff_lo >> 16) & 1;
    let mask_lo = 0u32.wrapping_sub(borrow_lo);
    let result_lo = (a_lo & !mask_lo) | (b_lo & mask_lo);

    // High halves
    let a_hi = a & U16_HI;
    let b_hi = b & U16_HI;
    let a_hi_shifted = a >> 16;
    let b_hi_shifted = b >> 16;
    let result_hi = if a_hi_shifted >= b_hi_shifted {
        a_hi
    } else {
        b_hi
    };

    result_lo | result_hi
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_swar_relu_f16x2_positive() {
        // Both positive: should pass through unchanged
        let pos1 = f16::from_f32(1.5).to_bits() as u32;
        let pos2 = f16::from_f32(2.5).to_bits() as u32;
        let word = pos1 | (pos2 << 16);
        let result = swar_relu_f16x2(word);
        assert_eq!(result, word); // unchanged
    }

    #[test]
    fn test_swar_relu_f16x2_negative() {
        // Both negative: should be zeroed
        let neg1 = f16::from_f32(-1.5).to_bits() as u32;
        let neg2 = f16::from_f32(-2.5).to_bits() as u32;
        let word = neg1 | (neg2 << 16);
        let result = swar_relu_f16x2(word);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_swar_relu_f16x2_mixed() {
        let pos = f16::from_f32(1.5).to_bits() as u32;
        let neg = f16::from_f32(-2.5).to_bits() as u32;
        let word = pos | (neg << 16);
        let result = swar_relu_f16x2(word);

        let result_lo = result & 0xFFFF;
        let result_hi = (result >> 16) & 0xFFFF;

        assert_eq!(result_lo, pos); // positive, kept
        assert_eq!(result_hi, 0x0000); // negative, zeroed
    }

    #[test]
    fn test_swar_relu_f16x2_zero() {
        let zero = f16::from_f32(0.0).to_bits() as u32;
        let word = zero | (zero << 16);
        let result = swar_relu_f16x2(word);
        assert_eq!(result, word); // zero passes through
    }

    #[test]
    fn test_swar_relu_backward_f16x2() {
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

    #[test]
    fn test_swar_relu_backward_f32x1() {
        let pos = 1.0f32.to_bits();
        let neg = (-1.0f32).to_bits();
        let grad = 0xFFFF_FFFFu32;

        assert_eq!(swar_relu_backward_f32x1(grad, pos), grad); // positive, pass
        assert_eq!(swar_relu_backward_f32x1(grad, neg), 0); // negative, block
    }
}
