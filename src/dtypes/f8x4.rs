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
        return sign_bit | ((new_biased as u8) << 3);
    }
    if biased_exp >= 15 {
        return sign_bit | 0x7B;
    }
    sign_bit | ((biased_exp as u8) << 3) | mant
}

impl PackedWord for F8x4 {
    const ITEMS: usize = 4;
    const BIT_WIDTH: usize = 8;
    const IS_FLOAT: bool = true;
    const MAX_REPRESENTABLE: f32 = 240.0;
    type Array = [f32; 4];

    #[inline]
    fn unpack_to_f32(self) -> [f32; 4] {
        let bytes = self.0.to_le_bytes();
        [
            e4m3_to_f32(bytes[0]),
            e4m3_to_f32(bytes[1]),
            e4m3_to_f32(bytes[2]),
            e4m3_to_f32(bytes[3]),
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

    fn wgsl_unpack_body() -> &'static str {
        "return vec4<f32>(unpack4xI8(packed));\n"
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
}
