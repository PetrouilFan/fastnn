//! SWAR (SIMD Within A Register) operations for 8-bit signed integers.
//! All operations work directly on the raw u32 word — no unpacking needed.

/// Mask for even byte positions (0, 2) at bits 0-7, 16-23
pub const U8_EVEN: u32 = 0x00FF_00FF;
/// Mask for odd byte positions (1, 3) at bits 8-15, 24-31
pub const U8_ODD: u32 = 0xFF00_FF00;
/// Mask for all sign bits (bit 7 of each byte)
pub const U8_SIGN: u32 = 0x8080_8080;

/// SWAR addition for two U8x4 words.
#[inline]
pub fn swar_add_u8x4(a: u32, b: u32) -> u32 {
    let a_odd_shifted = (a >> 8) & U8_EVEN;
    let b_odd_shifted = (b >> 8) & U8_EVEN;

    let sum_even = ((a & U8_EVEN).wrapping_add(b & U8_EVEN)) & U8_EVEN;
    let sum_odd = ((a_odd_shifted.wrapping_add(b_odd_shifted)) & U8_EVEN) << 8;

    sum_even | sum_odd
}

/// SWAR subtraction for two U8x4 words.
#[inline]
pub fn swar_sub_u8x4(a: u32, b: u32) -> u32 {
    let a_odd_shifted = (a >> 8) & U8_EVEN;
    let b_odd_shifted = (b >> 8) & U8_EVEN;

    let diff_even = ((a & U8_EVEN)
        .wrapping_add(U8_EVEN)
        .wrapping_sub(b & U8_EVEN))
        & U8_EVEN;
    let diff_odd = ((a_odd_shifted
        .wrapping_add(U8_EVEN)
        .wrapping_sub(b_odd_shifted))
        & U8_EVEN)
        << 8;

    diff_even | diff_odd
}

/// SWAR ReLU for signed 8-bit values in a U8x4 word.
#[inline]
pub fn swar_relu_s8x4(v: u32) -> u32 {
    let sign_bits = v & U8_SIGN;
    // Spread sign bit to fill byte
    let neg_mask = sign_bits
        | (sign_bits >> 1)
        | (sign_bits >> 2)
        | (sign_bits >> 3)
        | (sign_bits >> 4)
        | (sign_bits >> 5)
        | (sign_bits >> 6)
        | (sign_bits >> 7);
    v & !neg_mask
}

/// SWAR ReLU backward for 8-bit: passes gradient only where pre_relu > 0.
#[inline]
pub fn swar_relu_backward_u8x4(grad: u32, pre_relu: u32) -> u32 {
    let sign_bits = pre_relu & U8_SIGN;
    let neg_mask = sign_bits
        | (sign_bits >> 1)
        | (sign_bits >> 2)
        | (sign_bits >> 3)
        | (sign_bits >> 4)
        | (sign_bits >> 5)
        | (sign_bits >> 6)
        | (sign_bits >> 7);
    grad & !neg_mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swar_add_no_carry_leak() {
        // Pack [127, 0, 127, 0]
        let a = 0x007F_007Fu32;
        // Pack [1, 0, 1, 0]
        let b = 0x0001_0001u32;
        let result = swar_add_u8x4(a, b);
        // Expected: [128, 0, 128, 0] = 0x0080_0080
        assert_eq!(result, 0x0080_0080);
    }

    #[test]
    fn test_swar_relu() {
        // Pack [50, -30, 100, -1]
        let a: u8 = 50u8;
        let b: u8 = (-30i8) as u8;
        let c: u8 = 100u8;
        let d: u8 = (-1i8) as u8;
        let word = (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24);
        let result = swar_relu_s8x4(word);

        let bytes = result.to_le_bytes();
        assert_eq!(bytes[0] as i8, 50);
        assert_eq!(bytes[1] as i8, 0);
        assert_eq!(bytes[2] as i8, 100);
        assert_eq!(bytes[3] as i8, 0);
    }

    #[test]
    fn test_swar_relu_backward() {
        let grad = 0xFFFF_FFFFu32;
        let pre_relu: u32 = (50u32) | ((0xE2u32) << 8) | ((100u32) << 16) | ((0xFFu32) << 24);
        let result = swar_relu_backward_u8x4(grad, pre_relu);

        let bytes = result.to_le_bytes();
        assert_eq!(bytes[0], 0xFF); // 50 > 0, pass
        assert_eq!(bytes[1], 0x00); // -30 < 0, blocked
        assert_eq!(bytes[2], 0xFF); // 100 > 0, pass
        assert_eq!(bytes[3], 0x00); // -1 < 0, blocked
    }
}
