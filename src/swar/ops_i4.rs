//! SWAR (SIMD Within A Register) operations for I4x8 (4-bit signed).
//! Each u32 holds 8 signed 4-bit values in [-8, 7].

/// Sign bit per nibble (bit 3 of each nibble).
const I4_SIGN: u32 = 0x88888888;

#[inline(always)]
fn nibble_to_signed(n: u32) -> i32 {
    let n = n & 0xF;
    if n >= 8 {
        (n as i32) - 16
    } else {
        n as i32
    }
}

#[inline(always)]
fn signed_to_nibble(v: i32) -> u32 {
    (v as u32) & 0xF
}

/// Per-nibble unary operation on a packed I4x8 word.
#[inline(always)]
fn op1_i4x8(v: u32, f: impl Fn(i32) -> i32) -> u32 {
    let mut r = 0u32;
    for i in 0..8 {
        let shift = i * 4;
        let nib = nibble_to_signed(v >> shift);
        r |= signed_to_nibble(f(nib)) << shift;
    }
    r
}

/// Per-nibble binary operation on two packed I4x8 words.
#[inline(always)]
fn op2_i4x8(a: u32, b: u32, f: impl Fn(i32, i32) -> i32) -> u32 {
    let mut r = 0u32;
    for i in 0..8 {
        let shift = i * 4;
        let an = nibble_to_signed(a >> shift);
        let bn = nibble_to_signed(b >> shift);
        r |= signed_to_nibble(f(an, bn)) << shift;
    }
    r
}

#[inline(always)]
pub fn swar_add_i4x8(a: u32, b: u32) -> u32 {
    op2_i4x8(a, b, |x, y| x + y)
}

#[inline(always)]
pub fn swar_sub_i4x8(a: u32, b: u32) -> u32 {
    op2_i4x8(a, b, |x, y| x - y)
}

#[inline(always)]
pub fn swar_relu_i4x8(v: u32) -> u32 {
    let sign = v & I4_SIGN;
    let neg_mask = sign | sign.wrapping_sub(sign >> 3);
    v & !neg_mask
}

#[inline(always)]
pub fn swar_relu_backward_i4x8(grad: u32, pre_relu: u32) -> u32 {
    op2_i4x8(grad, pre_relu, |g, x| if x > 0 { g } else { 0 })
}

#[inline(always)]
pub fn swar_fused_relu_i4x8(v: u32) -> (u32, u32) {
    let relu_out = swar_relu_i4x8(v);
    let mask = op1_i4x8(v, |x| if x > 0 { 0xF } else { 0 });
    (relu_out, mask)
}

#[inline(always)]
pub fn swar_max_i4x8(a: u32, b: u32) -> u32 {
    op2_i4x8(a, b, |x, y| if x >= y { x } else { y })
}

#[inline(always)]
pub fn swar_min_i4x8(a: u32, b: u32) -> u32 {
    op2_i4x8(a, b, |x, y| if x <= y { x } else { y })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vals_to_word(vals: &[i8; 8]) -> u32 {
        let mut w = 0u32;
        for i in 0..8 {
            w |= ((vals[i] as u32) & 0xF) << (i * 4);
        }
        w
    }

    #[allow(dead_code)]
    fn word_to_vals(w: u32) -> [i8; 8] {
        let mut v = [0i8; 8];
        for i in 0..8 {
            let shift = i * 4;
            v[i] = ((w >> shift) & 0xF) as i8;
        }
        v
    }

    fn to_i8(w: u32, i: usize) -> i8 {
        let n = (w >> (i * 4)) & 0xF;
        if n >= 8 {
            (n as i8) - 16
        } else {
            n as i8
        }
    }

    #[test]
    fn test_i4x8_add() {
        let a = vals_to_word(&[1, 2, 3, 4, -1, -2, -3, -4]);
        let b = vals_to_word(&[1, 1, 1, 1, 1, 1, 1, 1]);
        let r = swar_add_i4x8(a, b);
        let expected = [2, 3, 4, 5, 0, -1, -2, -3];
        for i in 0..8 {
            assert_eq!(to_i8(r, i), expected[i], "mismatch at nibble {}", i);
        }
    }

    #[test]
    fn test_i4x8_sub() {
        let a = vals_to_word(&[5, 3, 7, 2, -1, -4, -2, 0]);
        let b = vals_to_word(&[1, 1, 1, 1, 1, 1, 1, 1]);
        let r = swar_sub_i4x8(a, b);
        let expected = [4, 2, 6, 1, -2, -5, -3, -1];
        for i in 0..8 {
            assert_eq!(to_i8(r, i), expected[i]);
        }
    }

    #[test]
    fn test_i4x8_relu() {
        let a = vals_to_word(&[-8, -1, 0, 3, -5, 7, -3, 1]);
        let r = swar_relu_i4x8(a);
        let expected = [0, 0, 0, 3, 0, 7, 0, 1];
        for i in 0..8 {
            assert_eq!(to_i8(r, i), expected[i]);
        }
    }

    #[test]
    fn test_i4x8_relu_backward() {
        let grad = vals_to_word(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let pre = vals_to_word(&[-3, 0, 2, -1, 5, -7, 1, 0]);
        let r = swar_relu_backward_i4x8(grad, pre);
        let expected = [0, 0, 3, 0, 5, 0, 7, 0];
        for i in 0..8 {
            assert_eq!(to_i8(r, i), expected[i]);
        }
    }

    #[test]
    fn test_i4x8_fused_relu() {
        let v = vals_to_word(&[-3, 0, 2, -1, 5, -7, 1, 0]);
        let (relu_out, mask) = swar_fused_relu_i4x8(v);
        for i in 0..8 {
            let expected_relu = if to_i8(v, i) > 0 { to_i8(v, i) } else { 0 };
            let expected_mask = if to_i8(v, i) > 0 { 0xF } else { 0 };
            assert_eq!(to_i8(relu_out, i), expected_relu);
            assert_eq!(((mask >> (i * 4)) & 0xF) as i8, expected_mask);
        }
    }

    #[test]
    fn test_i4x8_max() {
        let a = vals_to_word(&[-8, 5, -3, 0, 7, -1, 2, -4]);
        let b = vals_to_word(&[0, -5, -3, 4, -8, 6, 2, -8]);
        let r = swar_max_i4x8(a, b);
        let expected = [0, 5, -3, 4, 7, 6, 2, -4];
        for i in 0..8 {
            assert_eq!(to_i8(r, i), expected[i]);
        }
    }

    #[test]
    fn test_i4x8_min() {
        let a = vals_to_word(&[-8, 5, -3, 0, 7, -1, 2, -4]);
        let b = vals_to_word(&[0, -5, -3, 4, -8, 6, 2, -8]);
        let r = swar_min_i4x8(a, b);
        let expected = [-8, -5, -3, 0, -8, -1, 2, -8];
        for i in 0..8 {
            assert_eq!(to_i8(r, i), expected[i]);
        }
    }

    #[test]
    fn test_i4x8_add_overflow_wrap() {
        // 7 + 1 = 8, wraps to -8 as signed 4-bit
        let a = vals_to_word(&[7, 0, 0, 0, 0, 0, 0, 0]);
        let b = vals_to_word(&[1, 0, 0, 0, 0, 0, 0, 0]);
        let r = swar_add_i4x8(a, b);
        assert_eq!(to_i8(r, 0), -8);
    }
}
