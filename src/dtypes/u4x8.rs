use super::PackedWord;

/// 4-bit signed integer, 8 values packed per u32 word.
/// Values are in the range [-8, 7].
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct U4x8(pub u32);

unsafe impl bytemuck::Pod for U4x8 {}
unsafe impl bytemuck::Zeroable for U4x8 {}

impl PackedWord for U4x8 {
    const ITEMS: usize = 8;
    const BIT_WIDTH: usize = 4;
    const IS_FLOAT: bool = false;
    type Array = [f32; 8];

    #[inline]
    fn unpack_to_f32(self) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        let word = self.0;
        for i in 0..8 {
            let nibble = (word >> (i * 4)) & 0xF;
            let signed = if nibble & 0x8 != 0 {
                (nibble | 0xFFFFFFF0) as i32
            } else {
                nibble as i32
            };
            out[i] = signed as f32;
        }
        out
    }

    #[inline]
    fn pack_from_f32(vals: [f32; 8]) -> Self {
        let mut word: u32 = 0;
        for i in 0..8 {
            let clamped = vals[i].clamp(-8.0, 7.0);
            let rounded = clamped.round();
            let as_i32 = rounded as i32;
            let nibble = (as_i32 as u32) & 0xF;
            word |= nibble << (i * 4);
        }
        U4x8(word)
    }

    fn wgsl_unpack_body() -> &'static str {
        concat!(
            "var out: mat2x4<f32>;\n",
            "for (var i: u32 = 0u; i < 4u; i = i + 1u) {\n",
            "  let nib0 = (packed >> (i * 4u)) & 0xFu;\n",
            "  let s0 = i32(nib0) - (i32(nib0 >> 3u) * 16);\n",
            "  out[0][i] = f32(s0);\n",
            "  let nib1 = (packed >> ((i + 4u) * 4u)) & 0xFu;\n",
            "  let s1 = i32(nib1) - (i32(nib1 >> 3u) * 16);\n",
            "  out[1][i] = f32(s1);\n",
            "}\n",
            "return out;\n",
        )
    }

    fn wgsl_return_type() -> &'static str {
        "mat2x4<f32>"
    }

    fn wgsl_dot_logic() -> &'static str {
        "acc += dot(unpacked[0], act0) + dot(unpacked[1], act1);"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip_u4x8() {
        let vals = [0.0, 1.0, 2.0, 3.0, -1.0, -4.0, 7.0, -8.0];
        let packed = U4x8::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        for i in 0..8 {
            assert!(
                (unpacked[i] - vals[i]).abs() < 0.5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                unpacked[i],
                vals[i]
            );
        }
    }

    #[test]
    fn test_clamp_u4x8() {
        let vals = [100.0, -100.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let packed = U4x8::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        assert_eq!(unpacked[0], 7.0); // clamped to max
        assert_eq!(unpacked[1], -8.0); // clamped to min
    }

    #[test]
    fn test_zero_u4x8() {
        let vals = [0.0; 8];
        let packed = U4x8::pack_from_f32(vals);
        assert_eq!(packed.0, 0);
        let unpacked = packed.unpack_to_f32();
        for v in unpacked {
            assert_eq!(v, 0.0);
        }
    }
}
