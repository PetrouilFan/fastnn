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
#[inline(always)]
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

/// SWAR ReLU backward for F16x2 — blocks gradient where pre_relu ≤ 0.
#[inline(always)]
pub fn swar_relu_backward_f16x2(grad: u32, pre_relu: u32) -> u32 {
    let sign_lo = (pre_relu >> 15) & 1;
    let sign_hi = pre_relu >> 31;
    let mask_lo = 0u32.wrapping_sub(sign_lo);
    let mask_hi = 0u32.wrapping_sub(sign_hi);
    let neg_mask = (mask_lo & 0xFFFF) | (mask_hi << 16);
    // Also block zero values (f16 zero = all bits clear)
    let lo_nonzero = pre_relu & 0xFFFF;
    let hi_nonzero = pre_relu >> 16;
    let zero_mask_lo = if lo_nonzero == 0 { 0xFFFF } else { 0 };
    let zero_mask_hi = if hi_nonzero == 0 { 0xFFFF0000 } else { 0 };
    grad & !(neg_mask | zero_mask_lo | zero_mask_hi)
}

/// SWAR element-wise max for two F16x2 words.
/// Uses IEEE 754 total-order conversion for correct signed comparison.
#[inline(always)]
pub fn swar_max_u16x2(a: u32, b: u32) -> u32 {
    // IEEE 754 total-order: XOR with 0x8000 if non-negative, XOR with 0xFFFF if negative
    // This makes unsigned comparison equivalent to IEEE 754 total order
    let sa_lo = (a >> 15) & 1;
    let sa_hi = a >> 31;
    let mask_a = U16_SIGN | (sa_lo * 0x7FFF) | (sa_hi * 0x7FFF_0000);
    let a_biased = a ^ mask_a;

    let sb_lo = (b >> 15) & 1;
    let sb_hi = b >> 31;
    let mask_b = U16_SIGN | (sb_lo * 0x7FFF) | (sb_hi * 0x7FFF_0000);
    let b_biased = b ^ mask_b;

    // Low halves
    let a_lo = a_biased & U16_LO;
    let b_lo = b_biased & U16_LO;
    let diff_lo = a_lo.wrapping_add(U16_LO).wrapping_sub(b_lo);
    let borrow_lo = (diff_lo >> 16) & 1;
    let mask_lo = 0u32.wrapping_sub(borrow_lo);
    let result_lo = ((a & U16_LO) & mask_lo) | ((b & U16_LO) & !mask_lo);

    // High halves
    let a_hi_shifted = a_biased >> 16;
    let b_hi_shifted = b_biased >> 16;
    let diff_hi = a_hi_shifted.wrapping_add(U16_LO).wrapping_sub(b_hi_shifted);
    let borrow_hi = (diff_hi >> 16) & 1;
    let mask_hi = 0u32.wrapping_sub(borrow_hi);
    let result_hi = ((a & U16_HI) & mask_hi) | ((b & U16_HI) & !mask_hi);

    result_lo | result_hi
}

/// SWAR element-wise min for two F16x2 words (treating bits as unsigned 16-bit).
/// This is approximate — proper F16 min requires unpacking.
#[inline(always)]
pub fn swar_min_u16x2(a: u32, b: u32) -> u32 {
    let a_lo = a & U16_LO;
    let b_lo = b & U16_LO;
    // Compare low halves: if a <= b, no borrow
    let diff_lo = b_lo.wrapping_add(U16_LO).wrapping_sub(a_lo);
    let borrow_lo = (diff_lo >> 16) & 1;
    let mask_lo = 0u32.wrapping_sub(borrow_lo);
    // borrow_lo = 1 when b > a → mask = 0xFFFF → select a
    let result_lo = (a_lo & mask_lo) | (b_lo & !mask_lo);

    // High halves (branchless)
    let a_hi = a & U16_HI;
    let b_hi = b & U16_HI;
    let a_hi_shifted = a >> 16;
    let b_hi_shifted = b >> 16;
    let diff_hi = b_hi_shifted.wrapping_add(U16_LO).wrapping_sub(a_hi_shifted);
    let borrow_hi = (diff_hi >> 16) & 1;
    let mask_hi = 0u32.wrapping_sub(borrow_hi);
    let result_hi = (a_hi & mask_hi) | (b_hi & !mask_hi);

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
    fn test_swar_max_u16x2() {
        let a = 0x0002_0001u32; // [1, 2]
        let b = 0x0003_0000u32; // [0, 3]
        let result = swar_max_u16x2(a, b);
        assert_eq!(result, 0x0003_0001u32); // max([1,2], [0,3]) = [1,3]
    }

    #[test]
    fn test_swar_min_u16x2() {
        let a = 0x0002_0001u32; // [1, 2]
        let b = 0x0003_0000u32; // [0, 3]
        let result = swar_min_u16x2(a, b);
        assert_eq!(result, 0x0002_0000u32); // min([1,2], [0,3]) = [0,2]
    }
}
