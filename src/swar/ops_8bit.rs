//! SWAR (SIMD Within A Register) operations for 8-bit signed integers.
//! All operations work directly on the raw u32 word — no unpacking needed.

pub const U8_EVEN: u32 = 0x00FF_00FF;
pub const U8_ODD: u32 = 0xFF00_FF00;
pub const U8_SIGN: u32 = 0x8080_8080;

#[inline(always)]
pub fn swar_add_u8x4(a: u32, b: u32) -> u32 {
    let a_odd_shifted = (a >> 8) & U8_EVEN;
    let b_odd_shifted = (b >> 8) & U8_EVEN;

    let sum_even = ((a & U8_EVEN).wrapping_add(b & U8_EVEN)) & U8_EVEN;
    let sum_odd = ((a_odd_shifted.wrapping_add(b_odd_shifted)) & U8_EVEN) << 8;

    sum_even | sum_odd
}

#[inline(always)]
pub fn swar_sub_u8x4(a: u32, b: u32) -> u32 {
    let a_odd_shifted = (a >> 8) & U8_EVEN;
    let b_odd_shifted = (b >> 8) & U8_EVEN;

    let diff_even = ((a & U8_EVEN)
        .wrapping_add(0x0100_0100)
        .wrapping_sub(b & U8_EVEN))
        & U8_EVEN;
    let diff_odd = ((a_odd_shifted
        .wrapping_add(0x0100_0100)
        .wrapping_sub(b_odd_shifted))
        & U8_EVEN)
        << 8;

    diff_even | diff_odd
}

#[inline(always)]
pub fn swar_relu_s8x4(v: u32) -> u32 {
    let sign_bits = v & U8_SIGN;
    let neg_mask = sign_bits | sign_bits.wrapping_sub(sign_bits >> 7);
    v & !neg_mask
}

#[inline(always)]
pub fn swar_relu_backward_u8x4(grad: u32, pre_relu: u32) -> u32 {
    let sign_bits = pre_relu & U8_SIGN;
    let neg_mask = sign_bits | sign_bits.wrapping_sub(sign_bits >> 7);
    let nz = pre_relu | (pre_relu >> 1);
    let nz = nz | (nz >> 2);
    let nz = nz | (nz >> 4);
    let nz_bit0 = nz & 0x0101_0101;
    let not_zero = nz_bit0.wrapping_mul(0xFF);
    grad & not_zero & !neg_mask
}

/// Fused SWAR ReLU forward + backward for U8x4.
/// Returns (relu_output, mask_for_grad) computed in a single pass
/// sharing the sign-bit computation.
#[inline(always)]
pub fn swar_fused_relu_u8x4(v: u32) -> (u32, u32) {
    let sign_bits = v & U8_SIGN;
    let neg_mask = sign_bits | sign_bits.wrapping_sub(sign_bits >> 7);
    let nz = v | (v >> 1);
    let nz = nz | (nz >> 2);
    let nz = nz | (nz >> 4);
    let nz_bit0 = nz & 0x0101_0101;
    let not_zero = nz_bit0.wrapping_mul(0xFF);
    let relu_out = v & !neg_mask;
    let mask = not_zero & !neg_mask;
    (relu_out, mask)
}

#[inline(always)]
pub fn swar_max_u8x4(a: u32, b: u32) -> u32 {
    let a_biased = a ^ U8_SIGN;
    let b_biased = b ^ U8_SIGN;

    let a_even_biased = a_biased & U8_EVEN;
    let b_even_biased = b_biased & U8_EVEN;
    let diff_even = a_even_biased.wrapping_add(U8_EVEN).wrapping_sub(b_even_biased);
    let borrow_even = (diff_even >> 8) & 0x0001_0001;
    let mask_even = borrow_even * 0xFF;
    let result_even = ((a & U8_EVEN) & mask_even) | ((b & U8_EVEN) & !mask_even);

    let a_odd_biased = (a_biased >> 8) & U8_EVEN;
    let b_odd_biased = (b_biased >> 8) & U8_EVEN;
    let diff_odd = a_odd_biased.wrapping_add(U8_EVEN).wrapping_sub(b_odd_biased);
    let borrow_odd = (diff_odd >> 8) & 0x0001_0001;
    let mask_odd = borrow_odd * 0xFF;
    let result_odd = ((((a >> 8) & U8_EVEN) & mask_odd) | (((b >> 8) & U8_EVEN) & !mask_odd)) << 8;

    result_even | result_odd
}

#[inline(always)]
pub fn swar_min_u8x4(a: u32, b: u32) -> u32 {
    let a_biased = a ^ U8_SIGN;
    let b_biased = b ^ U8_SIGN;

    let a_even_biased = a_biased & U8_EVEN;
    let b_even_biased = b_biased & U8_EVEN;
    let diff_even = b_even_biased.wrapping_add(U8_EVEN).wrapping_sub(a_even_biased);
    let borrow_even = (diff_even >> 8) & 0x0001_0001;
    let mask_even = borrow_even * 0xFF;
    let result_even = ((a & U8_EVEN) & mask_even) | ((b & U8_EVEN) & !mask_even);

    let a_odd_biased = (a_biased >> 8) & U8_EVEN;
    let b_odd_biased = (b_biased >> 8) & U8_EVEN;
    let diff_odd = b_odd_biased.wrapping_add(U8_EVEN).wrapping_sub(a_odd_biased);
    let borrow_odd = (diff_odd >> 8) & 0x0001_0001;
    let mask_odd = borrow_odd * 0xFF;
    let result_odd = ((((a >> 8) & U8_EVEN) & mask_odd) | (((b >> 8) & U8_EVEN) & !mask_odd)) << 8;

    result_even | result_odd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swar_add_no_carry_leak() {
        let a = 0x007F_007Fu32;
        let b = 0x0001_0001u32;
        let result = swar_add_u8x4(a, b);
        assert_eq!(result, 0x0080_0080);
    }

    #[test]
    fn test_swar_max_u8x4() {
        // Note: 200 as u8 = -56 as i8, 0xCE as u8 = 206 = -50 as i8
        let a = (50u32) | ((0xE2u32) << 8) | ((100u32) << 16) | ((0xFFu32) << 24);
        let b = (10u32) | ((20u32) << 8) | ((200u32) << 16) | ((0xCEu32) << 24);
        let result = swar_max_u8x4(a, b);
        let bytes = result.to_le_bytes();
        assert_eq!(bytes[0] as i8, 50);   // max(50, 10) = 50
        assert_eq!(bytes[1] as i8, 20);   // max(-30, 20) = 20
        assert_eq!(bytes[2] as i8, 100);  // max(100, -56) = 100
        assert_eq!(bytes[3] as i8, -1);   // max(-1, -50) = -1
    }

    #[test]
    fn test_swar_min_u8x4() {
        let a = (50u32) | ((0xE2u32) << 8) | ((100u32) << 16) | ((0xFFu32) << 24);
        let b = (10u32) | ((20u32) << 8) | ((200u32) << 16) | ((0xCEu32) << 24);
        let result = swar_min_u8x4(a, b);
        let bytes = result.to_le_bytes();
        assert_eq!(bytes[0] as i8, 10);   // min(50, 10) = 10
        assert_eq!(bytes[1] as i8, -30);  // min(-30, 20) = -30
        assert_eq!(bytes[2] as i8, -56);  // min(100, -56) = -56
        assert_eq!(bytes[3] as i8, -50);  // min(-1, -50) = -50
    }

    #[test]
    fn test_swar_relu() {
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
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[1], 0x00);
        assert_eq!(bytes[2], 0xFF);
        assert_eq!(bytes[3], 0x00);
    }
}
