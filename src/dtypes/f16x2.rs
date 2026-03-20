use super::PackedWord;
use half::f16;

/// IEEE 754 half-precision float, 2 values packed per u32 word.
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct F16x2(pub u32);

unsafe impl bytemuck::Pod for F16x2 {}
unsafe impl bytemuck::Zeroable for F16x2 {}

impl PackedWord for F16x2 {
    const ITEMS: usize = 2;
    const BIT_WIDTH: usize = 16;
    const IS_FLOAT: bool = true;
    type Array = [f32; 2];

    #[inline]
    fn unpack_to_f32(self) -> [f32; 2] {
        let lo = f16::from_bits((self.0 & 0xFFFF) as u16).to_f32();
        let hi = f16::from_bits((self.0 >> 16) as u16).to_f32();
        [lo, hi]
    }

    #[inline]
    fn pack_from_f32(vals: [f32; 2]) -> Self {
        let lo = f16::from_f32(vals[0]).to_bits() as u32;
        let hi = f16::from_f32(vals[1]).to_bits() as u32;
        F16x2(lo | (hi << 16))
    }

    fn wgsl_unpack_body() -> &'static str {
        "return unpack2x16float(packed);\n"
    }

    fn wgsl_return_type() -> &'static str {
        "vec2<f32>"
    }

    fn wgsl_dot_logic() -> &'static str {
        "acc += dot(unpacked, act0);"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip_f16x2() {
        let vals = [1.5, -2.5];
        let packed = F16x2::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        for i in 0..2 {
            assert!(
                (unpacked[i] - vals[i]).abs() < 0.01,
                "Mismatch at index {}: got {}, expected {}",
                i,
                unpacked[i],
                vals[i]
            );
        }
    }

    #[test]
    fn test_zero_f16x2() {
        let vals = [0.0; 2];
        let packed = F16x2::pack_from_f32(vals);
        assert_eq!(packed.0, 0);
    }

    #[test]
    fn test_large_values_f16x2() {
        let vals = [65504.0, -65504.0]; // f16 max
        let packed = F16x2::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        assert!((unpacked[0] - 65504.0).abs() < 1.0);
        assert!((unpacked[1] + 65504.0).abs() < 1.0);
    }
}
