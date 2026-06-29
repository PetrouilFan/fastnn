use super::PackedWord;

/// FP8 E5M2 (5 exponent, 2 mantissa, bias=15), 4 values packed per u32 word.
///
/// Bit layout per byte: `SEEEE MM`
///   - S: sign (1 bit)
///   - E: exponent (5 bits, bias=15)
///   - M: mantissa (2 bits, explicit leading bit)
///
/// Range: ±57344, Inf at 0x7C/0xFC, NaN at 0x7E/0x7F/0xFE/0xFF.
/// Primarily for gradient storage during training (wider range than E4M3).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct F8x4R(pub u32);

unsafe impl bytemuck::Pod for F8x4R {}
unsafe impl bytemuck::Zeroable for F8x4R {}

const E5M2_BIAS: i32 = 15;

#[inline]
pub fn e5m2_to_f32(byte: u8) -> f32 {
    let sign = if (byte & 0x80) == 0 { 1.0 } else { -1.0 };
    let biased_exp = (byte >> 2) & 0x1F;
    let mant = (byte & 0x03) as f32;
    match biased_exp {
        0 => {
            if mant == 0.0 {
                0.0 * sign
            } else {
                sign * 2.0f32.powi(1 - E5M2_BIAS) * (mant / 4.0)
            }
        }
        0x1F => {
            if mant == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * 2.0f32.powi(biased_exp as i32 - E5M2_BIAS) * (1.0 + mant / 4.0),
    }
}

#[inline]
fn f32_to_e5m2(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let sign_bit = if v < 0.0 { 0x80 } else { 0x00 };
    let abs_v = v.abs();
    if abs_v == 0.0 {
        return if v.is_sign_negative() { 0x80 } else { 0x00 };
    }
    if v.is_infinite() {
        return if v > 0.0 { 0x7C } else { 0xFC };
    }
    const MAX: f32 = 57344.0;
    let abs_v = abs_v.min(MAX);
    if abs_v < (1.0 / 16384.0) {
        let mant_rounded = (abs_v * 16384.0).round() as u8;
        if mant_rounded >= 4 {
            return sign_bit | 0x04;
        }
        if mant_rounded == 0 {
            return sign_bit;
        }
        return sign_bit | mant_rounded;
    }
    let e = abs_v.log2().floor() as i32;
    let biased_exp = (e + E5M2_BIAS) as u8;
    let normalized = abs_v / 2.0f32.powi(e);
    let mant = ((normalized - 1.0) * 4.0).round() as u8;
    if mant >= 4 {
        let new_biased = biased_exp + 1;
        if new_biased >= 31 {
            return sign_bit | 0x7C;
        }
        return sign_bit | ((new_biased) << 2);
    }
    if biased_exp >= 31 {
        return sign_bit | 0x7C;
    }
    sign_bit | ((biased_exp) << 2) | mant
}

impl PackedWord for F8x4R {
    const ITEMS: usize = 4;
    const BIT_WIDTH: usize = 8;
    const IS_FLOAT: bool = true;
    const MAX_REPRESENTABLE: f32 = 57344.0;
    type Array = [f32; 4];

    #[inline]
    fn unpack_to_f32(self) -> [f32; 4] {
        let bytes = self.0.to_le_bytes();
        [
            e5m2_to_f32(bytes[0]),
            e5m2_to_f32(bytes[1]),
            e5m2_to_f32(bytes[2]),
            e5m2_to_f32(bytes[3]),
        ]
    }

    #[inline]
    fn pack_from_f32(vals: [f32; 4]) -> Self {
        let b0 = f32_to_e5m2(vals[0]) as u32;
        let b1 = f32_to_e5m2(vals[1]) as u32;
        let b2 = f32_to_e5m2(vals[2]) as u32;
        let b3 = f32_to_e5m2(vals[3]) as u32;
        F8x4R(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
    }

    fn wgsl_unpack_body() -> &'static str {
        concat!(
            "var result: vec4<f32>;\n",
            "for (var i = 0u; i < 4u; i = i + 1u) {\n",
            "  let byte = (packed >> (i * 8u)) & 0xFFu;\n",
            "  let sign = byte & 0x80u;\n",
            "  let exp = (byte >> 2u) & 0x1Fu;\n",
            "  let mant = byte & 0x3u;\n",
            "  if (exp == 0u && mant == 0u) {\n",
            "    result[i] = select(0.0, -0.0, sign != 0u);\n",
            "  } else if (exp == 0x1Fu) {\n",
            "    if (mant == 0u) {\n",
            "      result[i] = select(1.0 / 0.0, -1.0 / 0.0, sign != 0u);\n",
            "    } else {\n",
            "      result[i] = 0.0 / 0.0;\n",
            "    }\n",
            "  } else if (exp == 0u) {\n",
            "    let v = exp2(-14.0) * f32(mant) / 4.0;\n",
            "    result[i] = select(v, -v, sign != 0u);\n",
            "  } else {\n",
            "    let v = exp2(f32(i32(exp) - 15)) * (1.0 + f32(mant) / 4.0);\n",
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
    fn test_e5m2_to_f32_zero() {
        assert_eq!(e5m2_to_f32(0x00), 0.0);
        assert_eq!(e5m2_to_f32(0x80), -0.0);
    }

    #[test]
    fn test_e5m2_to_f32_one() {
        let val = e5m2_to_f32(0x3C);
        assert!((val - 1.0).abs() < 0.1, "got {}", val);
    }

    #[test]
    fn test_e5m2_to_f32_inf() {
        assert!(e5m2_to_f32(0x7C).is_infinite());
        assert!(e5m2_to_f32(0x7C) > 0.0);
        assert!(e5m2_to_f32(0xFC).is_infinite());
        assert!(e5m2_to_f32(0xFC) < 0.0);
    }

    #[test]
    fn test_e5m2_to_f32_nan() {
        assert!(e5m2_to_f32(0x7F).is_nan());
        assert!(e5m2_to_f32(0xFE).is_nan());
    }

    #[test]
    fn test_f32_to_e5m2_roundtrip() {
        let test_vals = [0.0, 1.0, -1.0, 2.0, 256.0, 32768.0, -32768.0, 57344.0];
        for &v in &test_vals {
            let encoded = f32_to_e5m2(v);
            let decoded = e5m2_to_f32(encoded);
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
    fn test_f8x4r_pack_unpack_roundtrip() {
        let vals = [1.0, -1.0, 256.0, -256.0];
        let packed = F8x4R::pack_from_f32(vals);
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
    fn test_e5m2_subnormal() {
        let val = e5m2_to_f32(0x01);
        assert!(
            val > 0.0 && val < 2e-5,
            "E5M2 smallest subnormal, got {}",
            val
        );
    }

    #[test]
    fn test_e5m2_roundtrip_extensive() {
        let test_vals = [
            0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0,
            1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, -32768.0, 57344.0, -57344.0, 0.5,
            0.25, 0.125, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0,
        ];
        for &v in &test_vals {
            let encoded = f32_to_e5m2(v);
            let decoded = e5m2_to_f32(encoded);
            let err = (decoded - v).abs();
            assert!(
                err < 0.5 * v.abs().max(1.0),
                "E5M2 roundtrip err {} for input {}, encoded 0x{:02X}, decoded {}",
                err,
                v,
                encoded,
                decoded
            );
        }
    }

    #[test]
    fn test_e5m2_inf_nan_values() {
        assert!(e5m2_to_f32(0x7C).is_infinite());
        assert!(e5m2_to_f32(0x7C) > 0.0);
        assert!(e5m2_to_f32(0xFC).is_infinite());
        assert!(e5m2_to_f32(0xFC) < 0.0);
        assert!(e5m2_to_f32(0x7E).is_nan());
        assert!(e5m2_to_f32(0x7F).is_nan());
        assert!(e5m2_to_f32(0xFE).is_nan());
        assert!(e5m2_to_f32(0xFF).is_nan());
    }
}
