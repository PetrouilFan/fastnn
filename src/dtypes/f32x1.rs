use super::PackedWord;

/// IEEE 754 single-precision float, 1 value per u32 word (no packing).
/// This is the baseline type — no memory savings, but provides a uniform API.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct F32x1(pub u32);

// SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
unsafe impl bytemuck::Pod for F32x1 {}
unsafe impl bytemuck::Zeroable for F32x1 {}

impl PackedWord for F32x1 {
    const ITEMS: usize = 1;
    const BIT_WIDTH: usize = 32;
    const IS_FLOAT: bool = true;
    type Array = [f32; 1];

    #[inline]
    fn unpack_to_f32(self) -> [f32; 1] {
        [f32::from_bits(self.0)]
    }

    #[inline]
    fn pack_from_f32(vals: [f32; 1]) -> Self {
        F32x1(vals[0].to_bits())
    }

    fn wgsl_unpack_body() -> &'static str {
        "return bitcast<f32>(packed);\n"
    }

    fn wgsl_return_type() -> &'static str {
        "f32"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip_f32x1() {
        let vals = [3.14159f32];
        let packed = F32x1::pack_from_f32(vals);
        let unpacked = packed.unpack_to_f32();
        assert_eq!(unpacked[0], vals[0]);
    }

    #[test]
    fn test_zero_f32x1() {
        let vals = [0.0f32];
        let packed = F32x1::pack_from_f32(vals);
        assert_eq!(packed.0, 0);
    }
}
