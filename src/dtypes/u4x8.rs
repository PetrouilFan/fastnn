use super::PackedWord;

/// 4-bit signed integer, 8 values packed per u32 word.
/// Values are in the range [-8, 7].
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct U4x8(pub u32);

// SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
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
            let as_i32 = if rounded.is_nan() { 0 } else { rounded as i32 };
            let nibble = (as_i32 as u32) & 0xF;
            word |= nibble << (i * 4);
        }
        U4x8(word)
    }

    fn wgsl_unpack_body() -> &'static str {
        concat!(
            "var out: mat2x4<f32>;\n",
            // i = 0
            "  let nib0_0 = packed & 0xFu;\n",
            "  let s0_0 = i32(nib0_0) - (i32(nib0_0 >> 3u) * 16);\n",
            "  out[0][0] = f32(s0_0);\n",
            "  let nib1_0 = (packed >> 16u) & 0xFu;\n",
            "  let s1_0 = i32(nib1_0) - (i32(nib1_0 >> 3u) * 16);\n",
            "  out[1][0] = f32(s1_0);\n",
            // i = 1
            "  let nib0_1 = (packed >> 4u) & 0xFu;\n",
            "  let s0_1 = i32(nib0_1) - (i32(nib0_1 >> 3u) * 16);\n",
            "  out[0][1] = f32(s0_1);\n",
            "  let nib1_1 = (packed >> 20u) & 0xFu;\n",
            "  let s1_1 = i32(nib1_1) - (i32(nib1_1 >> 3u) * 16);\n",
            "  out[1][1] = f32(s1_1);\n",
            // i = 2
            "  let nib0_2 = (packed >> 8u) & 0xFu;\n",
            "  let s0_2 = i32(nib0_2) - (i32(nib0_2 >> 3u) * 16);\n",
            "  out[0][2] = f32(s0_2);\n",
            "  let nib1_2 = (packed >> 24u) & 0xFu;\n",
            "  let s1_2 = i32(nib1_2) - (i32(nib1_2 >> 3u) * 16);\n",
            "  out[1][2] = f32(s1_2);\n",
            // i = 3
            "  let nib0_3 = (packed >> 12u) & 0xFu;\n",
            "  let s0_3 = i32(nib0_3) - (i32(nib0_3 >> 3u) * 16);\n",
            "  out[0][3] = f32(s0_3);\n",
            "  let nib1_3 = (packed >> 28u) & 0xFu;\n",
            "  let s1_3 = i32(nib1_3) - (i32(nib1_3 >> 3u) * 16);\n",
            "  out[1][3] = f32(s1_3);\n",
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
