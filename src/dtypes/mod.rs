pub mod f16x2;
pub mod f32x1;
pub mod u4x8;
pub mod u8x4;

pub use f16x2::F16x2;
pub use f32x1::F32x1;
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

    /// Unpacked representation as f32 array for compute
    type Array: AsRef<[f32]> + AsMut<[f32]> + Copy + Default;

    /// Unpack all ITEMS values from this u32 into f32
    fn unpack_to_f32(self) -> Self::Array;

    /// Pack ITEMS f32 values back into a u32 word
    fn pack_from_f32(vals: Self::Array) -> Self;

    /// WGSL unpack function body (injected into shader at compile time)
    fn wgsl_unpack_body() -> &'static str;

    /// WGSL type returned by unpack (e.g. `"mat2x4<f32>"`, `"vec4<f32>"`)
    fn wgsl_return_type() -> &'static str;

    /// WGSL dot product accumulation line (injected into shader)
    fn wgsl_dot_logic() -> &'static str;
}
