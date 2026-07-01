use std::sync::OnceLock;

use super::PackedWord;

/// FP8 E4M3 (4 exponent, 3 mantissa, bias=7), 4 values packed per u32 word.
///
/// Bit layout per byte: `SEEEEMMM`
///   - S: sign (1 bit)
///   - E: exponent (4 bits, bias=7)
///   - M: mantissa (3 bits, explicit leading bit)
///
/// Range: ±240, NaN at 0x7F/0xFF, no infinity.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct F8x4(pub u32);

unsafe impl bytemuck::Pod for F8x4 {}
unsafe impl bytemuck::Zeroable for F8x4 {}

const E4M3_BIAS: i32 = 7;

#[inline]
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if (byte & 0x80) == 0 { 1.0 } else { -1.0 };
    let biased_exp = (byte >> 3) & 0x0F;
    let mant = (byte & 0x07) as f32;
    match biased_exp {
        0 => {
            if mant == 0.0 {
                0.0 * sign
            } else {
                sign * 2.0f32.powi(1 - E4M3_BIAS) * (mant / 8.0)
            }
        }
        0x0F => f32::NAN,
        _ => sign * 2.0f32.powi(biased_exp as i32 - E4M3_BIAS) * (1.0 + mant / 8.0),
    }
}

#[inline]
fn f32_to_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let sign_bit = if v < 0.0 { 0x80 } else { 0x00 };
    let abs_v = v.abs();
    if abs_v == 0.0 {
        return if v.is_sign_negative() { 0x80 } else { 0x00 };
    }
    const MAX: f32 = 240.0;
    let abs_v = abs_v.min(MAX);
    if abs_v < 0.015625 {
        let mant_rounded = (abs_v * 512.0).round() as u8;
        if mant_rounded >= 8 {
            return sign_bit | 0x08;
        }
        if mant_rounded == 0 {
            return sign_bit;
        }
        return sign_bit | mant_rounded;
    }
    let e = abs_v.log2().floor() as i32;
    let biased_exp = (e + E4M3_BIAS) as u8;
    let normalized = abs_v / 2.0f32.powi(e);
    let mant = ((normalized - 1.0) * 8.0).round() as u8;
    if mant >= 8 {
        let new_biased = biased_exp + 1;
        if new_biased >= 15 {
            return sign_bit | 0x7B;
        }
        return sign_bit | ((new_biased) << 3);
    }
    if biased_exp >= 15 {
        return sign_bit | 0x7B;
    }
    sign_bit | ((biased_exp) << 3) | mant
}

// ── Packed dot product (LUT-based, avoids per-element unpack) ────

/// 256-entry f32 LUT: `LUT[byte] = e4m3_to_f32(byte)`.
/// Pre-computed once; avoids per-element `powi()` calls in hot path.
fn f8_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0.0f32; 256];
        for i in 0..256u16 {
            lut[i as usize] = e4m3_to_f32(i as u8);
        }
        lut
    })
}

/// Dot product of two F8x4-packed u32 words using LUT-based dequantize.
/// Returns f32 sum of (act[i] * weight[i]) for i in 0..4.
#[inline(always)]
pub fn f8x4_dot_packed_f32(a: u32, b: u32) -> f32 {
    let lut = f8_lut();
    let bytes_a = a.to_le_bytes();
    let bytes_b = b.to_le_bytes();
    lut[bytes_a[0] as usize] * lut[bytes_b[0] as usize]
        + lut[bytes_a[1] as usize] * lut[bytes_b[1] as usize]
        + lut[bytes_a[2] as usize] * lut[bytes_b[2] as usize]
        + lut[bytes_a[3] as usize] * lut[bytes_b[3] as usize]
}

impl PackedWord for F8x4 {
    const ITEMS: usize = 4;
    const BIT_WIDTH: usize = 8;
    const IS_FLOAT: bool = true;
    const MAX_REPRESENTABLE: f32 = 240.0;
    type Array = [f32; 4];

    #[inline]
    fn unpack_to_f32(self) -> [f32; 4] {
        let lut = f8_lut();
        let bytes = self.0.to_le_bytes();
        [
            lut[bytes[0] as usize],
            lut[bytes[1] as usize],
            lut[bytes[2] as usize],
            lut[bytes[3] as usize],
        ]
    }

    #[inline]
    fn pack_from_f32(vals: [f32; 4]) -> Self {
        let b0 = f32_to_e4m3(vals[0]) as u32;
        let b1 = f32_to_e4m3(vals[1]) as u32;
        let b2 = f32_to_e4m3(vals[2]) as u32;
        let b3 = f32_to_e4m3(vals[3]) as u32;
        F8x4(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
    }

    #[inline(always)]
    fn dot_packed_f32(a: Self, b: Self) -> f32 {
        f8x4_dot_packed_f32(a.0, b.0)
    }

    fn wgsl_unpack_body() -> &'static str {
        concat!(
            "var result: vec4<f32>;\n",
            "for (var i = 0u; i < 4u; i = i + 1u) {\n",
            "  let byte = (packed >> (i * 8u)) & 0xFFu;\n",
            "  let sign = byte & 0x80u;\n",
            "  let exp = (byte >> 3u) & 0xFu;\n",
            "  let mant = byte & 0x7u;\n",
            "  if (exp == 0u && mant == 0u) {\n",
            "    result[i] = select(0.0, -0.0, sign != 0u);\n",
            "  } else if (exp == 0xFu) {\n",
            "    if (mant == 0u) {\n",
            "      result[i] = select(1.0 / 0.0, -1.0 / 0.0, sign != 0u);\n",
            "    } else {\n",
            "      result[i] = 0.0 / 0.0;\n",
            "    }\n",
            "  } else if (exp == 0u) {\n",
            "    let v = 0.015625 * f32(mant) / 8.0;\n",
            "    result[i] = select(v, -v, sign != 0u);\n",
            "  } else {\n",
            "    let v = exp2(f32(i32(exp) - 7)) * (1.0 + f32(mant) / 8.0);\n",
            "    result[i] = select(v, -v, sign != 0u);\n",
            "  }\n",
            "}\n",
            "return result;\n",
        )
    }

    fn wgsl_return_type() -> &'static str {
        "vec4<f32>"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e4m3_to_f32_zero() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x80), -0.0);
    }

    #[test]
    fn test_e4m3_to_f32_one() {
        let val = e4m3_to_f32(0x38);
        assert!((val - 1.0).abs() < 0.1, "got {}", val);
    }

    #[test]
    fn test_e4m3_to_f32_max() {
        let val = e4m3_to_f32(0x77);
        assert!((val - 240.0).abs() < 1.0, "got {}", val);
    }

    #[test]
    fn test_f32_to_e4m3_roundtrip() {
        let test_vals = [0.0, 1.0, -1.0, 0.5, 2.0, 128.0, -128.0, 240.0, -240.0];
        for &v in &test_vals {
            let encoded = f32_to_e4m3(v);
            let decoded = e4m3_to_f32(encoded);
            let err = (decoded - v).abs();
            assert!(
                err < 0.5 * v.abs().max(1.0),
                "roundtrip err {} for input {}, encoded 0x{:02X}, decoded {}",
                err,
                v,
                encoded,
                decoded
            );
        }
    }

    #[test]
    fn test_f8x4_pack_unpack_roundtrip() {
        let vals = [1.0, -1.0, 128.0, -128.0];
        let packed = F8x4::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        for i in 0..4 {
            let err = (unpacked[i] - vals[i]).abs();
            assert!(
                err < 0.5 * vals[i].abs().max(1.0),
                "Mismatch at {}: got {}, expected {}, err {}",
                i,
                unpacked[i],
                vals[i],
                err
            );
        }
    }

    #[test]
    fn test_f8x4_nan() {
        let encoded = f32_to_e4m3(f32::NAN);
        assert_eq!(encoded, 0x7F);
        assert!(e4m3_to_f32(0x7F).is_nan());
    }

    #[test]
    fn test_e4m3_subnormal() {
        let val = e4m3_to_f32(0x01);
        assert!(
            val > 0.0 && val < 0.1,
            "Smallest subnormal should be tiny, got {}",
            val
        );
    }

    #[test]
    fn test_e4m3_neg_nan() {
        assert!(e4m3_to_f32(0xFF).is_nan());
    }

    #[test]
    fn test_e4m3_zero_nan_special_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x80), -0.0);
        assert!(e4m3_to_f32(0x7F).is_nan());
        assert!(e4m3_to_f32(0xFF).is_nan());
    }

    #[test]
    fn test_e4m3_roundtrip_extensive() {
        let test_vals = [
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, -128.0,
            240.0, -240.0, 0.25, 0.125, 1.5, 2.5, 3.5, 7.0, 15.0, 31.0, 63.0, 127.0,
        ];
        for &v in &test_vals {
            let encoded = f32_to_e4m3(v);
            let decoded = e4m3_to_f32(encoded);
            let err = (decoded - v).abs();
            assert!(
                err < 0.5 * v.abs().max(1.0),
                "E4M3 roundtrip err {} for input {}, encoded 0x{:02X}, decoded {}",
                err,
                v,
                encoded,
                decoded
            );
        }
    }

    #[test]
    fn test_f8x4_pack_unpack_all_representable() {
        let vals = [0.0, 240.0, -240.0, f32::NAN];
        let packed = F8x4::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        assert_eq!(unpacked[0], 0.0);
        assert!((unpacked[1] - 240.0).abs() < 1.0);
        assert!((unpacked[2] - (-240.0)).abs() < 1.0);
        assert!(unpacked[3].is_nan());
    }
}
