use super::PackedWord;

/// 8-bit signed integer, 4 values packed per u32 word.
/// Values are in the range [-128, 127].
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct U8x4(pub u32);

unsafe impl bytemuck::Pod for U8x4 {}
unsafe impl bytemuck::Zeroable for U8x4 {}

impl PackedWord for U8x4 {
    const ITEMS: usize = 4;
    const BIT_WIDTH: usize = 8;
    const IS_FLOAT: bool = false;
    type Array = [f32; 4];

    #[inline]
    fn unpack_to_f32(self) -> [f32; 4] {
        let bytes = self.0.to_le_bytes();
        [
            (bytes[0] as i8) as f32,
            (bytes[1] as i8) as f32,
            (bytes[2] as i8) as f32,
            (bytes[3] as i8) as f32,
        ]
    }

    #[inline]
    fn pack_from_f32(vals: [f32; 4]) -> Self {
        let bytes = [
            vals[0].clamp(-128.0, 127.0).round() as i8 as u8,
            vals[1].clamp(-128.0, 127.0).round() as i8 as u8,
            vals[2].clamp(-128.0, 127.0).round() as i8 as u8,
            vals[3].clamp(-128.0, 127.0).round() as i8 as u8,
        ];
        U8x4(u32::from_le_bytes(bytes))
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
    fn test_pack_unpack_roundtrip_u8x4() {
        let vals = [0.0, 1.0, -1.0, 127.0];
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
        let vals = [200.0, -200.0, 50.0, -50.0];
        let packed = U8x4::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        assert_eq!(unpacked[0], 127.0);
        assert_eq!(unpacked[1], -128.0);
    }

    #[test]
    fn test_zero_u8x4() {
        let vals = [0.0; 4];
        let packed = U8x4::pack_from_f32(vals);
        assert_eq!(packed.0, 0);
    }
}
