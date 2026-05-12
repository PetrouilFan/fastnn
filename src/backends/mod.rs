pub mod cpu;
pub mod packed_blas;
pub mod packed_simd;
pub mod wgpu;

pub use packed_simd::{ScopedVec, TlsVecPool};
