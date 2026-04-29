//! SWAR operations for 32-bit values (F32x1).
//! While F32x1 doesn't pack multiple values per u32, SWAR ops on the
//! IEEE 754 sign bit are still useful for element-wise operations like ReLU.

/// SWAR ReLU for F32x1 — zeroes negative values using IEEE 754 sign bit.
/// Equivalent to max(0, x) but operates on raw u32 bits.
#[inline(always)]
pub fn swar_relu_f32x1(v: u32) -> u32 {
    let sign = v >> 31;
    let mask = sign.wrapping_neg();
    v & !mask
}

/// SWAR ReLU backward for F32x1 — passes gradient only where pre_relu > 0.
/// Blocks gradient at zero (matching scalar fallback convention).
#[inline(always)]
pub fn swar_relu_backward_f32x1(grad: u32, pre_relu: u32) -> u32 {
    // Block if sign bit set (negative) or all bits zero
    let sign = pre_relu >> 31;
    let is_zero = if pre_relu == 0 { 1u32 } else { 0 };
    let block = sign | is_zero;
    let clear = 0u32.wrapping_sub(block);
    grad & !clear
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swar_relu_f32x1_positive() {
        let v = 1.5f32.to_bits();
        assert_eq!(swar_relu_f32x1(v), v); // unchanged
    }

    #[test]
    fn test_swar_relu_f32x1_negative() {
        let v = (-1.5f32).to_bits();
        assert_eq!(swar_relu_f32x1(v), 0); // zeroed
    }

    #[test]
    fn test_swar_relu_f32x1_zero() {
        let v = 0.0f32.to_bits();
        assert_eq!(swar_relu_f32x1(v), v); // passes through (sign bit 0)
    }

    #[test]
    fn test_swar_relu_f32x1_neg_zero() {
        let v = (-0.0f32).to_bits();
        assert_eq!(swar_relu_f32x1(v), 0);
    }

    #[test]
    fn test_swar_relu_backward_f32x1() {
        let pos = 1.0f32.to_bits();
        let neg = (-1.0f32).to_bits();
        let zero = 0.0f32.to_bits();
        let grad = 0xFFFF_FFFFu32;

        assert_eq!(swar_relu_backward_f32x1(grad, pos), grad); // positive, pass
        assert_eq!(swar_relu_backward_f32x1(grad, neg), 0); // negative, block
        assert_eq!(swar_relu_backward_f32x1(grad, zero), 0); // zero, block
    }
}
