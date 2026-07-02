pub mod f16x2;
pub mod f32x1;
pub mod f4x8;
pub mod f8x4;
pub mod f8x4r;
pub mod i4x8;
pub mod i8x4;
pub mod u4x8;
pub mod u8x4;

pub use f16x2::F16x2;
pub use f32x1::F32x1;
pub use f4x8::F4x8;
pub use f8x4::F8x4;
pub use f8x4r::F8x4R;
pub use i4x8::I4x8;
pub use i8x4::I8x4;
pub use u4x8::U4x8;
pub use u8x4::U8x4;

/// Core trait for packed multi-precision types.
/// Each implementor packs N values into a single u32 word.
pub trait PackedWord: Send + Sync + Copy + bytemuck::Pod + bytemuck::Zeroable + Default {
    /// Number of values stored per u32 word
    const ITEMS: usize;
    /// Bit width of each individual value
    const BIT_WIDTH: usize;
    /// Whether the values are floating point (true) or integer (false)
    const IS_FLOAT: bool;
    /// Maximum representable value in this format (for scale computation).
    /// For integer types: 2^(BIT_WIDTH-1)-1. For float types: format-specific max.
    const MAX_REPRESENTABLE: f32;

    /// Unpacked representation as f32 array for compute
    type Array: AsRef<[f32]> + AsMut<[f32]> + Copy + Default;

    /// Unpack all ITEMS values from this u32 into f32
    fn unpack_to_f32(self) -> Self::Array;

    /// Pack ITEMS f32 values back into a u32 word
    fn pack_from_f32(vals: Self::Array) -> Self;

    /// Dot product of two packed words: sum(a[i] * b[i]) for i in 0..ITEMS.
    /// Default: unpack both to f32 arrays, multiply, sum.
    /// Override for types where LUT-based lookup is faster (e.g., F8x4, F8x4R).
    #[inline(always)]
    fn dot_packed_f32(a: Self, b: Self) -> f32 {
        let a_v = a.unpack_to_f32();
        let b_v = b.unpack_to_f32();
        a_v.as_ref()
            .iter()
            .zip(b_v.as_ref().iter())
            .map(|(x, y)| x * y)
            .sum()
    }

    /// WGSL unpack function body (injected into shader at compile time)
    fn wgsl_unpack_body() -> &'static str;

    /// WGSL type returned by unpack (e.g. `"mat2x4<f32>"`, `"vec4<f32>"`)
    fn wgsl_return_type() -> &'static str;
}
