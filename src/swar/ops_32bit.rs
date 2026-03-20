//! SWAR operations for 32-bit values (F32x1).
//! While F32x1 doesn't pack multiple values per u32, SWAR ops on the
//! IEEE 754 sign bit are still useful for element-wise operations like ReLU.

/// SWAR ReLU for F32x1 — zeroes negative values using IEEE 754 sign bit.
/// Equivalent to max(0, x) but operates on raw u32 bits.
#[inline]
pub fn swar_relu_f32x1(v: u32) -> u32 {
    // IEEE 754 sign bit is bit 31
    // If sign bit is set (negative), zero the value
    let sign = v >> 31; // 1 if negative, 0 if positive
    let mask = sign.wrapping_neg(); // 0xFFFFFFFF if neg, 0x00000000 if pos
    v & !mask
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
        // -0.0 has sign bit set, so it gets zeroed
        // This is fine for ReLU: -0.0 == 0.0 mathematically
        assert_eq!(swar_relu_f32x1(v), 0);
    }
}
