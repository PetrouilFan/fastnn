//! SWAR (SIMD Within A Register) operations for 4-bit signed integers.
//! All operations work directly on the raw u32 word — no unpacking needed.

/// Mask for even nibble positions (0, 2, 4, 6) at bits 0-3, 8-11, 16-19, 24-27
pub const U4_EVEN: u32 = 0x0F0F_0F0F;
/// Mask for odd nibble positions (1, 3, 5, 7) at bits 4-7, 12-15, 20-23, 28-31
pub const U4_ODD: u32 = 0xF0F0_F0F0;
/// Mask for all sign bits (bit 3 of each nibble)
pub const U4_SIGN: u32 = 0x8888_8888;

/// SWAR addition for two U4x8 words.
/// Splits into even and odd nibble groups, adds each separately,
/// and masks off carry bits to prevent lane boundary bleeding.
#[inline]
pub fn swar_add_u4x8(a: u32, b: u32) -> u32 {
    // Shift odd nibbles to even positions
    let a_odd_shifted = (a >> 4) & U4_EVEN;
    let b_odd_shifted = (b >> 4) & U4_EVEN;

    // Add even-positioned nibbles (mask prevents carry from crossing nibble boundary)
    let sum_even = ((a & U4_EVEN).wrapping_add(b & U4_EVEN)) & U4_EVEN;

    // Add odd-positioned nibbles (shifted to even positions, then shift back)
    let sum_odd = ((a_odd_shifted.wrapping_add(b_odd_shifted)) & U4_EVEN) << 4;

    sum_even | sum_odd
}

/// SWAR subtraction for two U4x8 words.
/// Uses the same even/odd split as addition.
#[inline]
pub fn swar_sub_u4x8(a: u32, b: u32) -> u32 {
    let a_odd_shifted = (a >> 4) & U4_EVEN;
    let b_odd_shifted = (b >> 4) & U4_EVEN;

    // Add 0x10 per nibble to handle underflow wrapping
    // For even nibbles: (a + 0x10 - b) masked to nibble
    let diff_even = ((a & U4_EVEN)
        .wrapping_add(U4_EVEN)
        .wrapping_sub(b & U4_EVEN))
        & U4_EVEN;
    let diff_odd = ((a_odd_shifted
        .wrapping_add(U4_EVEN)
        .wrapping_sub(b_odd_shifted))
        & U4_EVEN)
        << 4;

    diff_even | diff_odd
}

/// SWAR ReLU for signed 4-bit values in a U4x8 word.
/// Zeroes out any nibble whose sign bit is set (negative values).
#[inline]
pub fn swar_relu_s4x8(v: u32) -> u32 {
    // Extract sign bits (bit 3 of each nibble)
    let sign_bits = v & U4_SIGN;
    // Spread each sign bit to fill its entire nibble
    let neg_mask = sign_bits | (sign_bits >> 1) | (sign_bits >> 2) | (sign_bits >> 3);
    // Zero out negative lanes
    v & !neg_mask
}

/// SWAR element-wise max for two U4x8 words (treating as unsigned).
/// For each lane: returns a if a >= b, else b.
#[inline]
pub fn swar_max_u4x8(a: u32, b: u32) -> u32 {
    // Even lanes
    let a_even = a & U4_EVEN;
    let b_even = b & U4_EVEN;
    // If (a - b) underflows, borrow propagates into the gap bit above the nibble
    // We use wrapping_add(U4_EVEN) to prevent underflow wrapping around
    let diff_even = a_even.wrapping_add(U4_EVEN).wrapping_sub(b_even);
    // The bit just above each even nibble (position 4, 12, 20, 28) will be 0 if borrow occurred
    let borrow_even = (diff_even >> 4) & 0x0101_0101u32;
    // Spread borrow bit to fill the nibble
    let mask_even = borrow_even * 0x0F; // broadcast: 1 nibble → 0x0F, 0 → 0x00
    let result_even = (a_even & mask_even) | (b_even & !mask_even);

    // Odd lanes
    let a_odd = (a >> 4) & U4_EVEN;
    let b_odd = (b >> 4) & U4_EVEN;
    let diff_odd = a_odd.wrapping_add(U4_EVEN).wrapping_sub(b_odd);
    let borrow_odd = (diff_odd >> 4) & 0x0101_0101u32;
    let mask_odd = borrow_odd * 0x0F;
    let result_odd = ((a_odd & mask_odd) | (b_odd & !mask_odd)) << 4;

    result_even | result_odd
}

/// SWAR ReLU backward for 4-bit: passes gradient only where pre_relu > 0.
#[inline]
pub fn swar_relu_backward_u4x8(grad: u32, pre_relu: u32) -> u32 {
    let sign_bits = pre_relu & U4_SIGN;
    let neg_mask = sign_bits | (sign_bits >> 1) | (sign_bits >> 2) | (sign_bits >> 3);
    grad & !neg_mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swar_add_no_carry_leak() {
        // Pack values [7, 0, 7, 0, 7, 0, 7, 0] into a word
        let a = 0x0707_0707u32;
        // Pack values [1, 0, 1, 0, 1, 0, 1, 0]
        let b = 0x0101_0101u32;
        let result = swar_add_u4x8(a, b);
        // Expected: [8, 0, 8, 0, 8, 0, 8, 0] = 0x0808_0808
        assert_eq!(result, 0x0808_0808);
    }

    #[test]
    fn test_swar_add_mixed() {
        // a = [3, 5, -1, 0, ...] → 3 | (5<<4) | (0xF<<8) | (0<<12) = 0x0F53
        let a: u32 = 0x0000_F53;
        // b = [1, 2, -2, 0, ...] → 1 | (2<<4) | (0xE<<8) | (0<<12) = 0x0E21
        let b: u32 = 0x0000_E21;
        let result = swar_add_u4x8(a, b);
        // Expected nibbles: 3+1=4, 5+2=7, 0xF+0xE masked = (−1)+(−2) masked
        let result_nibbles: Vec<i32> = (0..8)
            .map(|i| {
                let n = (result >> (i * 4)) & 0xF;
                if n & 0x8 != 0 {
                    (n | 0xFFFFFFF0) as i32
                } else {
                    n as i32
                }
            })
            .collect();
        assert_eq!(result_nibbles[0], 4); // 3+1
        assert_eq!(result_nibbles[1], 7); // 5+2
                                          // Lane 2: -1 + (-2) = -3 → 0xD (signed 4-bit)
        assert_eq!(result_nibbles[2] as i32, -3);
    }

    #[test]
    fn test_swar_relu() {
        // Pack [3, -2, 5, -1, 0, -4, 7, -8]
        let vals: u32 = (0x3u32)
            | (0xEu32 << 4)
            | (0x5u32 << 8)
            | (0xFu32 << 12)
            | (0x0u32 << 16)
            | (0xCu32 << 20)
            | (0x7u32 << 24)
            | (0x8u32 << 28);
        let result = swar_relu_s4x8(vals);
        // Positive values preserved, negative zeroed
        let nibbles: Vec<i32> = (0..8)
            .map(|i| {
                let n = (result >> (i * 4)) & 0xF;
                if n & 0x8 != 0 {
                    (n | 0xFFFFFFF0) as i32
                } else {
                    n as i32
                }
            })
            .collect();
        assert_eq!(nibbles[0], 3); // positive, kept
        assert_eq!(nibbles[1], 0); // negative, zeroed
        assert_eq!(nibbles[2], 5); // positive, kept
        assert_eq!(nibbles[3], 0); // negative, zeroed
        assert_eq!(nibbles[4], 0); // zero, kept
        assert_eq!(nibbles[5], 0); // negative, zeroed
        assert_eq!(nibbles[6], 7); // positive, kept
        assert_eq!(nibbles[7], 0); // negative (-8), zeroed
    }

    #[test]
    fn test_swar_relu_backward() {
        // grad = all 1s pattern
        let grad = 0xFFFF_FFFFu32;
        // pre_relu: [3, -2, 5, -1, 0, -4, 7, -8]
        let pre_relu: u32 = (0x3u32)
            | (0xEu32 << 4)
            | (0x5u32 << 8)
            | (0xFu32 << 12)
            | (0x0u32 << 16)
            | (0xCu32 << 20)
            | (0x7u32 << 24)
            | (0x8u32 << 28);
        let result = swar_relu_backward_u4x8(grad, pre_relu);
        // Check: positive nibbles pass through, negative are zeroed
        assert_eq!((result >> 0) & 0xF, 0xF); // 3 > 0, pass
        assert_eq!((result >> 4) & 0xF, 0x0); // -2 < 0, blocked
        assert_eq!((result >> 8) & 0xF, 0xF); // 5 > 0, pass
        assert_eq!((result >> 12) & 0xF, 0x0); // -1 < 0, blocked
    }
}
