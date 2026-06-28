//! SWAR operations for float-packed formats (F8x4, F8x4R, F4x8).
//!
//! These formats store IEEE-like floats (sign-magnitude with bias).
//! For comparison, we apply the total-order transform:
//!   - Positive (S=0): keep bits as-is
//!   - Negative (S=1): flip all bits
//! After transform, unsigned comparison gives correct float ordering.

use crate::swar::ops_i4::swar_relu_i4x8;

// ── FP8 helpers (1 byte per element, 4 per u32) ──────────────────

const FP8_SIGN: u32 = 0x80808080;

/// Total-order transform for FP8 (E4M3 or E5M2): flip all bits if sign set.
#[inline(always)]
fn total_order_fp8x4(v: u32) -> u32 {
    let sign = v & FP8_SIGN;
    // For bytes where sign bit is 1, flip all bits
    // sign_byte * 0xFF gives 0x00 for positive, 0xFF for negative
    let flip_mask = (sign >> 7).wrapping_mul(0xFF);
    v ^ flip_mask
}

#[inline(always)]
fn fp8_sign_byte_mask(v: u32) -> u32 {
    // 0xFF per byte where sign bit is 1, 0x00 otherwise
    let sign = v & FP8_SIGN;
    (sign >> 7).wrapping_mul(0xFF)
}

#[inline(always)]
fn op2_fp8x4(a: u32, b: u32, cmp_t: impl Fn(u32, u32) -> u32) -> u32 {
    let a_t = total_order_fp8x4(a);
    let b_t = total_order_fp8x4(b);
    let select_a = cmp_t(a_t, b_t);
    (a & select_a) | (b & !select_a)
}

/// FP8 ReLU: zero out negative (sign bit set) values.
#[inline(always)]
pub fn swar_relu_fp8x4(v: u32) -> u32 {
    // For each byte: if sign bit = 1 (negative), output 0; else keep value
    v & !fp8_sign_byte_mask(v)
}

/// FP8 ReLU backward: zero out gradient where pre_relu was negative.
#[inline(always)]
pub fn swar_relu_backward_fp8x4(grad: u32, pre_relu: u32) -> u32 {
    grad & !fp8_sign_byte_mask(pre_relu)
}

/// Fused FP8 ReLU forward + backward.
#[inline(always)]
pub fn swar_fused_relu_fp8x4(v: u32) -> (u32, u32) {
    let mask = !fp8_sign_byte_mask(v);
    (v & mask, mask)
}

/// FP8 max: element-wise maximum using total-order comparison.
#[inline(always)]
pub fn swar_max_fp8x4(a: u32, b: u32) -> u32 {
    let a_t = total_order_fp8x4(a);
    let b_t = total_order_fp8x4(b);
    let mut r = 0u32;
    for i in 0..4 {
        let shift = i * 8;
        let aa = (a_t >> shift) & 0xFF;
        let bb = (b_t >> shift) & 0xFF;
        let orig_a = (a >> shift) & 0xFF;
        let orig_b = (b >> shift) & 0xFF;
        if aa >= bb {
            r |= orig_a << shift;
        } else {
            r |= orig_b << shift;
        }
    }
    r
}

/// FP8 min: element-wise minimum using total-order comparison.
#[inline(always)]
pub fn swar_min_fp8x4(a: u32, b: u32) -> u32 {
    let a_t = total_order_fp8x4(a);
    let b_t = total_order_fp8x4(b);
    let mut r = 0u32;
    for i in 0..4 {
        let shift = i * 8;
        let aa = (a_t >> shift) & 0xFF;
        let bb = (b_t >> shift) & 0xFF;
        let orig_a = (a >> shift) & 0xFF;
        let orig_b = (b >> shift) & 0xFF;
        if aa <= bb {
            r |= orig_a << shift;
        } else {
            r |= orig_b << shift;
        }
    }
    r
}

// ── FP4 helpers (4 bits per element, 8 per u32) ──────────────────

/// Total-order transform for FP4 (E2M1): flip all 4 bits if sign nibble set.
#[inline(always)]
pub fn total_order_fp4x8(v: u32) -> u32 {
    let sign = v & 0x88888888;
    // For nibbles where sign bit is 1, flip all 4 bits
    // For nibbles where sign bit is 1, flip all 4 bits.
    // Use even/odd split to avoid cross-byte multiplication artifacts.
    // Actually safer to do: spread 0x1 per nibble to 0xF directly.
    // But we need to ensure per-nibble masking.
    let flip_even = (sign >> 3) & 0x07070707;
    let flip_odd = (sign >> 7) & 0x07070707;
    // spread even nibbles: multiply each byte by 0xF, mask low nibble
    let spread_even = flip_even.wrapping_mul(0xF) & 0x0F0F0F0F;
    let spread_odd = (flip_odd.wrapping_mul(0xF) & 0x0F0F0F0F) << 4;
    let flip_mask = spread_even | spread_odd;
    v ^ flip_mask
}

/// FP4 ReLU: zero out negative (sign nibble set) values.
/// Same as I4x8 relu since sign bit is at the same position.
#[inline(always)]
pub fn swar_relu_fp4x8(v: u32) -> u32 {
    swar_relu_i4x8(v)
}

/// FP4 ReLU backward.
#[inline(always)]
pub fn swar_relu_backward_fp4x8(grad: u32, pre_relu: u32) -> u32 {
    let sign = pre_relu & 0x88888888;
    let neg_mask = sign | sign.wrapping_sub(sign >> 3);
    grad & !neg_mask
}

/// Fused FP4 ReLU forward + backward.
#[inline(always)]
pub fn swar_fused_relu_fp4x8(v: u32) -> (u32, u32) {
    let sign = v & 0x88888888;
    let neg_mask = sign | sign.wrapping_sub(sign >> 3);
    (v & !neg_mask, !neg_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FP8 tests ────────────────────────────────────────────────

    fn fp8_word(vals: &[i16; 4]) -> u32 {
        let mut w = 0u32;
        for i in 0..4 {
            w |= ((vals[i] as u32) & 0xFF) << (i * 8);
        }
        w
    }

    fn fp8_bytes(w: u32) -> [i16; 4] {
        let mut v = [0i16; 4];
        for i in 0..4 {
            v[i] = ((w >> (i * 8)) & 0xFF) as i16;
        }
        v
    }

    #[test]
    fn test_fp8_relu() {
        // E4M3: 0x80 = -NaN (negative), 0x00 = 0, 0x3F = 1.75, 0xFF = -NaN
        let v = fp8_word(&[0x00, 0x80, 0x3F, 0xFF]);
        let r = swar_relu_fp8x4(v);
        let bytes = fp8_bytes(r);
        assert_eq!(bytes[0], 0x00); // +0 stays
        assert_eq!(bytes[1], 0x00); // -NaN zeroed
        assert_eq!(bytes[2], 0x3F); // +1.75 stays
        assert_eq!(bytes[3], 0x00); // -NaN zeroed
    }

    #[test]
    fn test_fp8_max() {
        // 0x00=0, 0x3F=1.75 (positive), 0x80=-0, 0xBF=-1.75 (negative)
        let a = fp8_word(&[0x00, 0x3F, 0x80, 0xBF]);
        let b = fp8_word(&[0x3F, 0x00, 0xBF, 0x80]);
        let r = swar_max_fp8x4(a, b);
        let bytes = fp8_bytes(r);
        assert_eq!(bytes[0], 0x3F); // max(0, 1.75) = 1.75
        assert_eq!(bytes[1], 0x3F); // max(1.75, 0) = 1.75
        assert_eq!(bytes[2], 0x80); // max(-0, -1.75) = -0 (FP8 -0 = 0x80)
        assert_eq!(bytes[3], 0x80); // max(-1.75, -0) = -0
    }

    #[test]
    fn test_fp8_min() {
        let a = fp8_word(&[0x00, 0x3F, 0x80, 0xBF]);
        let b = fp8_word(&[0x3F, 0x00, 0xBF, 0x80]);
        let r = swar_min_fp8x4(a, b);
        let bytes = fp8_bytes(r);
        assert_eq!(bytes[0], 0x00); // min(0, 1.75) = 0
        assert_eq!(bytes[1], 0x00); // min(1.75, 0) = 0
        assert_eq!(bytes[2], 0xBF); // min(-0, -1.75) = -1.75
        assert_eq!(bytes[3], 0xBF); // min(-1.75, -0) = -1.75
    }

    #[test]
    fn test_fp8_relu_backward() {
        let grad = fp8_word(&[0x01, 0x02, 0x04, 0x08]);
        let pre = fp8_word(&[0x3F, 0x80, 0x00, 0xFF]);
        let r = swar_relu_backward_fp8x4(grad, pre);
        let bytes = fp8_bytes(r);
        assert_eq!(bytes[0], 0x01); // positive → keep grad
        assert_eq!(bytes[1], 0x00); // negative → zero
        assert_eq!(bytes[2], 0x04); // zero → keep grad (fp8 0x00 = +0, not negative)
        assert_eq!(bytes[3], 0x00); // negative → zero
    }

    // ── FP4 tests ────────────────────────────────────────────────

    fn fp4_word(vals: &[u8; 8]) -> u32 {
        let mut w = 0u32;
        for i in 0..8 {
            w |= (vals[i] as u32 & 0xF) << (i * 4);
        }
        w
    }

    fn fp4_nibbles(w: u32) -> [u8; 8] {
        let mut v = [0u8; 8];
        for i in 0..8 {
            v[i] = ((w >> (i * 4)) & 0xF) as u8;
        }
        v
    }

    #[test]
    fn test_fp4_total_order() {
        // FP4 codes: 0x0=0, 0x2=1.0 (positive), 0x8=-0, 0xA=-1.0 (negative)
        // After total-order: 0x0→0x0, 0x2→0x2, 0x8→0x7, 0xA→0x5
        let v = fp4_word(&[0x0, 0x2, 0x8, 0xA, 0x7, 0x1, 0x9, 0xF]);
        let t = total_order_fp4x8(v);
        // Expected: total order flips all nibble bits for sign=1
        // 0x0 → 0x0 (positive, no flip)
        // 0x2 → 0x2 (positive, no flip)
        // 0x8 → !0x8 = 0x7 (negative, flip)
        // 0xA → !0xA = 0x5 (negative, flip)
        // 0x7 → 0x7 (positive, no flip)
        // 0x1 → 0x1 (positive, no flip, 0x9 is positive 1.0 with sign=1? No, 0x9 = -0.5)
        //   0x9 → !0x9 = 0x6 (negative, flip)
        // 0xF → !0xF = 0x0 (negative, flip)
        let expected = fp4_word(&[0x0, 0x2, 0x7, 0x5, 0x7, 0x1, 0x6, 0x0]);
        assert_eq!(t, expected, "total_order: got {:08x}, expected {:08x}", t, expected);
    }

    #[test]
    fn test_fp4_relu() {
        let v = fp4_word(&[0x0, 0x2, 0x8, 0xA, 0x6, 0x9, 0x0, 0xF]);
        let r = swar_relu_fp4x8(v);
        let n = fp4_nibbles(r);
        // 0x0=0 → keep, 0x2=1.0 → keep, 0x8=-0 → 0, 0xA=-1.0 → 0,
        // 0x6=4.0 → keep, 0x9=-0.5 → 0, 0x0=0 → keep, 0xF=-6.0 → 0
        assert_eq!(n[0], 0x0);
        assert_eq!(n[1], 0x2);
        assert_eq!(n[2], 0x0);
        assert_eq!(n[3], 0x0);
        assert_eq!(n[4], 0x6);
        assert_eq!(n[5], 0x0);
        assert_eq!(n[6], 0x0);
        assert_eq!(n[7], 0x0);
    }

    #[test]
    fn test_fp4_relu_backward() {
        let grad = fp4_word(&[0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8]);
        let pre = fp4_word(&[0x2, 0x9, 0x0, 0xF, 0x6, 0x8, 0xA, 0x1]);
        let r = swar_relu_backward_fp4x8(grad, pre);
        let n = fp4_nibbles(r);
        assert_eq!(n[0], 0x1, "nibble 0: pre=0x2 pos, grad=0x1"); // 0x2 is positive → keep grad 0x1
        assert_eq!(n[1], 0x0, "nibble 1: pre=0xF neg, grad=0x2"); // 0xF is negative → zero
        assert_eq!(n[2], 0x3, "nibble 2: pre=0x0 zero, grad=0x3"); // 0x0 is zero → keep grad 0x3
        assert_eq!(n[3], 0x0, "nibble 3: pre=0x9 neg, grad=0x4"); // 0x9 is negative → zero
        assert_eq!(n[4], 0x5, "nibble 4: pre=0x6 pos, grad=0x5"); // 0x6 is positive → keep grad 0x5
        assert_eq!(n[5], 0x0, "nibble 5: pre=0x8 neg, grad=0x6"); // 0x8 is negative → zero
        assert_eq!(n[6], 0x0, "nibble 6: pre=0xA neg, grad=0x7"); // 0xA is negative → zero
        assert_eq!(n[7], 0x8, "nibble 7: pre=0x1 pos, grad=0x8"); // 0x1 is positive → keep grad 0x8
    }

    #[test]
    fn test_fp4_fused_relu() {
        let pre = fp4_word(&[0x2, 0x9, 0x0, 0xF, 0x6, 0x8, 0xA, 0x1]);
        let (out, mask) = swar_fused_relu_fp4x8(pre);
        let out_n = fp4_nibbles(out);
        let mask_n = fp4_nibbles(mask);
        assert_eq!(out_n[0], 0x2);
        assert_eq!(out_n[1], 0x0);
        assert_eq!(out_n[2], 0x0);
        assert_eq!(out_n[3], 0x0);
        assert_eq!(out_n[4], 0x6);
        assert_eq!(out_n[5], 0x0);
        assert_eq!(out_n[6], 0x0);
        assert_eq!(out_n[7], 0x1);
        assert_eq!(mask_n[0], 0xF); // keep
        assert_eq!(mask_n[1], 0x0); // mask out
        assert_eq!(mask_n[2], 0xF); // keep
        assert_eq!(mask_n[3], 0x0); // mask out
        assert_eq!(mask_n[4], 0xF); // keep
        assert_eq!(mask_n[5], 0x0); // mask out
        assert_eq!(mask_n[6], 0x0); // mask out
        assert_eq!(mask_n[7], 0xF); // keep
    }

    #[test]
    fn test_fp8_fused_relu() {
        let pre = fp8_word(&[0x3F, 0x80, 0x00, 0xFF]);
        let (out, mask) = swar_fused_relu_fp8x4(pre);
        let out_bytes = fp8_bytes(out);
        let mask_bytes = fp8_bytes(mask);
        assert_eq!(out_bytes[0], 0x3F); // positive → keep
        assert_eq!(out_bytes[1], 0x00); // negative → 0
        assert_eq!(out_bytes[2], 0x00); // zero → keep
        assert_eq!(out_bytes[3], 0x00); // negative → 0
        assert_eq!(mask_bytes[0], 0xFF); // keep grad
        assert_eq!(mask_bytes[1], 0x00); // mask out
        assert_eq!(mask_bytes[2], 0xFF); // keep grad
        assert_eq!(mask_bytes[3], 0x00); // mask out
    }
}
