use super::PackedWord;
use std::sync::OnceLock;

/// FP4 E2M1 (NVFP4-style), 8 values packed per u32 word.
///
/// 4-bit sign-magnitude encoding: `[S|E1|E0|M]`
/// - S = sign (bit 3)
/// - E = exponent (bits 2-1, bias=1)
/// - M = mantissa (bit 0)
///
/// Representable values:
///   0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct F4x8(pub u32);

unsafe impl bytemuck::Pod for F4x8 {}
unsafe impl bytemuck::Zeroable for F4x8 {}

/// FP4 real values indexed by magnitude code (0-7).
const FP4_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decode a single 4-bit FP4 nibble to f32.
#[inline(always)]
pub fn fp4_to_f32(code: u8) -> f32 {
    let mag = FP4_MAG[(code & 0x7) as usize];
    if (code & 0x8) != 0 && (code & 0x7) != 0 {
        -mag
    } else {
        mag
    }
}

/// Encode an f32 value to the nearest FP4 4-bit code.
#[inline(always)]
pub fn f32_to_fp4(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    let sign = v.is_sign_negative();
    let abs = v.abs();

    let mut best_i = 0u8;
    let mut best_d = abs;
    for (i, &mag) in FP4_MAG.iter().enumerate() {
        let d = (abs - mag).abs();
        if d < best_d {
            best_d = d;
            best_i = i as u8;
        }
    }
    if sign && best_i != 0 {
        best_i | 0x8
    } else {
        best_i
    }
}

/// 256-entry LUT: `LUT[a << 4 | b] = round(4 * fp4_to_f32(a) * fp4_to_f32(b))`
fn f4x8_lut() -> &'static [i16; 256] {
    static LUT: OnceLock<[i16; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0i16; 256];
        let vals: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for a in 0..16u8 {
            for b in 0..16u8 {
                let idx = ((a as usize) << 4) | b as usize;
                lut[idx] = (4.0 * vals[a as usize] * vals[b as usize]).round() as i16;
            }
        }
        lut
    })
}

/// Dot product of two F4x8-packed u32 words using the 256-entry LUT.
/// Returns i32 accumulator. Dequantize: `result = acc * scale_a * scale_b / 4.0`.
#[inline(always)]
pub fn f4x8_dot_packed(a: u32, b: u32) -> i32 {
    let lut = f4x8_lut();
    let mut sum = 0i32;
    // Manual unroll via indices to reduce loop overhead
    sum += lut[(((a >> 0) & 0xF) as usize) << 4 | ((b >> 0) & 0xF) as usize] as i32;
    sum += lut[(((a >> 4) & 0xF) as usize) << 4 | ((b >> 4) & 0xF) as usize] as i32;
    sum += lut[(((a >> 8) & 0xF) as usize) << 4 | ((b >> 8) & 0xF) as usize] as i32;
    sum += lut[(((a >> 12) & 0xF) as usize) << 4 | ((b >> 12) & 0xF) as usize] as i32;
    sum += lut[(((a >> 16) & 0xF) as usize) << 4 | ((b >> 16) & 0xF) as usize] as i32;
    sum += lut[(((a >> 20) & 0xF) as usize) << 4 | ((b >> 20) & 0xF) as usize] as i32;
    sum += lut[(((a >> 24) & 0xF) as usize) << 4 | ((b >> 24) & 0xF) as usize] as i32;
    sum += lut[(((a >> 28) & 0xF) as usize) << 4 | ((b >> 28) & 0xF) as usize] as i32;
    sum
}

impl PackedWord for F4x8 {
    const ITEMS: usize = 8;
    const BIT_WIDTH: usize = 4;
    const IS_FLOAT: bool = true;
    const MAX_REPRESENTABLE: f32 = 6.0;
    type Array = [f32; 8];

    #[inline]
    fn unpack_to_f32(self) -> [f32; 8] {
        let w = self.0;
        [
            fp4_to_f32((w >> 0) as u8 & 0xF),
            fp4_to_f32((w >> 4) as u8 & 0xF),
            fp4_to_f32((w >> 8) as u8 & 0xF),
            fp4_to_f32((w >> 12) as u8 & 0xF),
            fp4_to_f32((w >> 16) as u8 & 0xF),
            fp4_to_f32((w >> 20) as u8 & 0xF),
            fp4_to_f32((w >> 24) as u8 & 0xF),
            fp4_to_f32((w >> 28) as u8 & 0xF),
        ]
    }

    #[inline]
    fn pack_from_f32(vals: [f32; 8]) -> Self {
        let mut w = 0u32;
        w |= (f32_to_fp4(vals[0]) as u32) << 0;
        w |= (f32_to_fp4(vals[1]) as u32) << 4;
        w |= (f32_to_fp4(vals[2]) as u32) << 8;
        w |= (f32_to_fp4(vals[3]) as u32) << 12;
        w |= (f32_to_fp4(vals[4]) as u32) << 16;
        w |= (f32_to_fp4(vals[5]) as u32) << 20;
        w |= (f32_to_fp4(vals[6]) as u32) << 24;
        w |= (f32_to_fp4(vals[7]) as u32) << 28;
        F4x8(w)
    }

    fn wgsl_unpack_body() -> &'static str {
        concat!(
            "var out: mat2x4<f32>;\n",
            "  let nib0_0 = packed & 0xFu;\n",
            "  let mag0_0 = FP4_MAG[nib0_0 & 7u];\n",
            "  out[0][0] = select(mag0_0, -mag0_0, nib0_0 >> 3u != 0u && nib0_0 != 0u);\n",
            "  let nib1_0 = (packed >> 16u) & 0xFu;\n",
            "  let mag1_0 = FP4_MAG[nib1_0 & 7u];\n",
            "  out[1][0] = select(mag1_0, -mag1_0, nib1_0 >> 3u != 0u && nib1_0 != 0u);\n",
            "  let nib0_1 = (packed >> 4u) & 0xFu;\n",
            "  let mag0_1 = FP4_MAG[nib0_1 & 7u];\n",
            "  out[0][1] = select(mag0_1, -mag0_1, nib0_1 >> 3u != 0u && nib0_1 != 0u);\n",
            "  let nib1_1 = (packed >> 20u) & 0xFu;\n",
            "  let mag1_1 = FP4_MAG[nib1_1 & 7u];\n",
            "  out[1][1] = select(mag1_1, -mag1_1, nib1_1 >> 3u != 0u && nib1_1 != 0u);\n",
            "  let nib0_2 = (packed >> 8u) & 0xFu;\n",
            "  let mag0_2 = FP4_MAG[nib0_2 & 7u];\n",
            "  out[0][2] = select(mag0_2, -mag0_2, nib0_2 >> 3u != 0u && nib0_2 != 0u);\n",
            "  let nib1_2 = (packed >> 24u) & 0xFu;\n",
            "  let mag1_2 = FP4_MAG[nib1_2 & 7u];\n",
            "  out[1][2] = select(mag1_2, -mag1_2, nib1_2 >> 3u != 0u && nib1_2 != 0u);\n",
            "  let nib0_3 = (packed >> 12u) & 0xFu;\n",
            "  let mag0_3 = FP4_MAG[nib0_3 & 7u];\n",
            "  out[0][3] = select(mag0_3, -mag0_3, nib0_3 >> 3u != 0u && nib0_3 != 0u);\n",
            "  let nib1_3 = (packed >> 28u) & 0xFu;\n",
            "  let mag1_3 = FP4_MAG[nib1_3 & 7u];\n",
            "  out[1][3] = select(mag1_3, -mag1_3, nib1_3 >> 3u != 0u && nib1_3 != 0u);\n",
            "return out;\n",
        )
    }

    fn wgsl_return_type() -> &'static str {
        "mat2x4<f32>"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp4_to_f32_table() {
        assert_eq!(fp4_to_f32(0x0), 0.0);
        assert_eq!(fp4_to_f32(0x1), 0.5);
        assert_eq!(fp4_to_f32(0x2), 1.0);
        assert_eq!(fp4_to_f32(0x3), 1.5);
        assert_eq!(fp4_to_f32(0x4), 2.0);
        assert_eq!(fp4_to_f32(0x5), 3.0);
        assert_eq!(fp4_to_f32(0x6), 4.0);
        assert_eq!(fp4_to_f32(0x7), 6.0);
    }

    #[test]
    fn test_fp4_negative() {
        assert!(fp4_to_f32(0x8).is_sign_positive()); // -0 is treated as 0
        assert_eq!(fp4_to_f32(0x9), -0.5);
        assert_eq!(fp4_to_f32(0xF), -6.0);
    }

    #[test]
    fn test_f32_to_fp4_roundtrip() {
        let test_vals = [0.0, 0.5, 1.0, -1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -6.0, -0.5];
        for &v in &test_vals {
            let code = f32_to_fp4(v);
            let decoded = fp4_to_f32(code);
            let err = (decoded - v).abs();
            assert!(
                err < 0.01,
                "roundtrip err {:.6} for input {}, code 0x{:X}, decoded {}",
                err,
                v,
                code,
                decoded
            );
        }
    }

    #[test]
    fn test_f32_to_fp4_clamp() {
        let code = f32_to_fp4(100.0);
        assert_eq!(code & 0x7, 7); // clamped to 6.0
        assert_eq!(fp4_to_f32(code), 6.0);

        let code_neg = f32_to_fp4(-100.0);
        assert_eq!(code_neg & 0x7, 7);
        assert_eq!(fp4_to_f32(code_neg), -6.0);
    }

    #[test]
    fn test_f4x8_pack_unpack() {
        let vals = [1.0, -1.0, 0.5, 6.0, -6.0, 0.0, 3.0, -3.0];
        let packed = F4x8::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        for i in 0..8 {
            assert!(
                (unpacked[i] - vals[i]).abs() < 0.01,
                "Mismatch at {}: got {}, expected {}",
                i,
                unpacked[i],
                vals[i]
            );
        }
    }

    #[test]
    fn test_f4x8_dot_packed_exact() {
        // a = [1.0, 0.5, 0.0, 6.0, 2.0, 3.0, 4.0, 1.5]
        // b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        // dot = 1 + 0.5 + 0 + 6 + 2 + 3 + 4 + 1.5 = 18.0
        // LUT sum = 4 * 18.0 = 72
        let a_code: [u8; 8] = [2, 1, 0, 7, 4, 5, 6, 3]; // codes for the vals
        let b_code: [u8; 8] = [2, 2, 2, 2, 2, 2, 2, 2]; // all 1.0
        let mut a_w = 0u32;
        let mut b_w = 0u32;
        for i in 0..8 {
            a_w |= (a_code[i] as u32) << (i * 4);
            b_w |= (b_code[i] as u32) << (i * 4);
        }
        let sum = f4x8_dot_packed(a_w, b_w);
        assert_eq!(sum, 72, "expected 72 (4 * 18.0), got {}", sum);
    }

    #[test]
    fn test_f4x8_dot_packed_zero() {
        assert_eq!(f4x8_dot_packed(0, 0), 0);
        assert_eq!(f4x8_dot_packed(0x77777777, 0), 0); // all max magnitude, zero weight = 0
    }

    #[test]
    fn test_f4x8_zero_word() {
        let vals = [0.0; 8];
        let packed = F4x8::pack_from_f32(vals);
        assert_eq!(packed.0, 0);
    }
}
