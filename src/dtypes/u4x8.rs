use super::PackedWord;

/// 4-bit unsigned integer, 8 values packed per u32 word.
/// Values are in the range [0, 15].
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct U4x8(pub u32);

// SAFETY: U4x8 is a `repr(transparent)` struct over `u32`, so it is safe to
// reinterpret cast from/to byte slices.
unsafe impl bytemuck::Pod for U4x8 {}
unsafe impl bytemuck::Zeroable for U4x8 {}

impl PackedWord for U4x8 {
    const ITEMS: usize = 8;
    const BIT_WIDTH: usize = 4;
    const IS_FLOAT: bool = false;
    const MAX_REPRESENTABLE: f32 = 15.0; // UNSIGNED: max is 15, not 7
    type Array = [f32; 8];

    #[inline]
    fn unpack_to_f32(self) -> [f32; 8] {
        let mut arr = [0.0f32; 8];
        let word = self.0;
        for i in 0..8 {
            // UNSIGNED: no sign extension, just extract nibble as-is
            let nib = ((word >> (i * 4)) & 0xF) as i32;
            arr[i] = nib as f32; // Range [0, 15]
        }
        arr
    }

    #[inline]
    fn pack_from_f32(vals: [f32; 8]) -> Self {
        let mut word: u32 = 0;
        for i in 0..8 {
            // UNSIGNED: clamp to [0, 15]
            let clamped = vals[i].clamp(0.0, 15.0).round() as i32;
            word |= ((clamped as u32) & 0xF) << (i * 4);
        }
        U4x8(word)
    }

    fn wgsl_unpack_body() -> &'static str {
        // WGSL for unsigned 4-bit: no sign extension
        concat!(
            "var out: mat2x4<f32>;\n",
            "  let nib0_0 = packed & 0xFu;\n",
            "  out[0][0] = f32(i32(nib0_0));\n",
            "  let nib1_0 = (packed >> 16u) & 0xFu;\n",
            "  out[1][0] = f32(i32(nib1_0));\n",
            "  let nib0_1 = (packed >> 4u) & 0xFu;\n",
            "  out[0][1] = f32(i32(nib0_1));\n",
            "  let nib1_1 = (packed >> 20u) & 0xFu;\n",
            "  out[1][1] = f32(i32(nib1_1));\n",
            "  let nib0_2 = (packed >> 8u) & 0xFu;\n",
            "  out[0][2] = f32(i32(nib0_2));\n",
            "  let nib1_2 = (packed >> 24u) & 0xFu;\n",
            "  out[1][2] = f32(i32(nib1_2));\n",
            "  let nib0_3 = (packed >> 12u) & 0xFu;\n",
            "  out[0][3] = f32(i32(nib0_3));\n",
            "  let nib1_3 = (packed >> 28u) & 0xFu;\n",
            "  out[1][3] = f32(i32(nib1_3));\n",
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
        // Test all values [0, 15]
        let vals = [0.0, 1.0, 5.0, 8.0, 10.0, 15.0, 7.0, 3.0];
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
        // UNSIGNED: clamp to [0, 15], not [-8, 7]
        let vals = [100.0, -100.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let packed = U4x8::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        assert_eq!(unpacked[0], 15.0); // clamped to max unsigned
        assert_eq!(unpacked[1], 0.0); // clamped to min unsigned
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

    #[test]
    fn test_all_ones_u4x8() {
        // All 15s
        let vals = [15.0; 8];
        let packed = U4x8::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        for v in unpacked {
            assert_eq!(v, 15.0);
        }
    }
}
