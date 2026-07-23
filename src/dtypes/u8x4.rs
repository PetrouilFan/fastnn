use super::PackedWord;

/// 8-bit unsigned integer, 4 values packed per u32 word.
/// Values are in the range [0, 255].
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct U8x4(pub u32);

// SAFETY: U8x4 is a `repr(transparent)` struct over `u32`, so it is safe to
// reinterpret cast from/to byte slices.
unsafe impl bytemuck::Pod for U8x4 {}
unsafe impl bytemuck::Zeroable for U8x4 {}

impl PackedWord for U8x4 {
    const SCALAR_TYPE: crate::types::ScalarType = crate::types::ScalarType::U8;
    const ITEMS: usize = 4;
    const BIT_WIDTH: usize = 8;
    const IS_FLOAT: bool = false;
    const MAX_REPRESENTABLE: f32 = 255.0;
    type Array = [f32; 4];

    #[inline]
    fn unpack_to_f32(self) -> [f32; 4] {
        let bytes = self.0.to_le_bytes();
        [
            bytes[0] as f32,
            bytes[1] as f32,
            bytes[2] as f32,
            bytes[3] as f32,
        ]
    }

    #[inline]
    fn pack_from_f32(vals: [f32; 4]) -> Self {
        let mut word: u32 = 0;
        for i in 0..4 {
            // UNSIGNED: clamp to [0, 255]
            let clamped = vals[i].clamp(0.0, 255.0).round() as u8;
            word |= (clamped as u32) << (i * 8);
        }
        U8x4(word)
    }

    fn wgsl_unpack_body() -> &'static str {
        // WGSL unsigned 8-bit unpack: u8 as f32 (no sign extension)
        concat!(
            "return vec4<f32>(",
            "  f32((packed & 0xFFu)),",
            "  f32(((packed >> 8u) & 0xFFu)),",
            "  f32(((packed >> 16u) & 0xFFu)),",
            "  f32(((packed >> 24u) & 0xFFu))",
            ");\n",
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
    fn test_pack_unpack_roundtrip_u8x4() {
        let vals = [0.0, 1.0, 128.0, 255.0];
        let packed = U8x4::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        for i in 0..4 {
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
    fn test_clamp_u8x4() {
        // UNSIGNED: clamp to [0, 255], not [-128, 127]
        let vals = [300.0, -100.0, 50.0, 200.0];
        let packed = U8x4::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        assert_eq!(unpacked[0], 255.0);
        assert_eq!(unpacked[1], 0.0);
    }

    #[test]
    fn test_zero_u8x4() {
        let vals = [0.0; 4];
        let packed = U8x4::pack_from_f32(vals);
        assert_eq!(packed.0, 0);
    }

    #[test]
    fn test_max_u8x4() {
        let vals = [255.0; 4];
        let packed = U8x4::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        for v in unpacked {
            assert_eq!(v, 255.0);
        }
    }
}
