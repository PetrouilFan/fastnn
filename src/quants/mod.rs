//! Quantized GEMM types compatible with GGUF blockwise quantization.
//!
//! This module provides the `QuantizedGemm` trait for native execution of
//! blockwise-quantized weights (Q4_0, Q4_K, Q5_K, ...) without dequantizing
//! to f32 beforehand.  Each implementor is responsible for:
//!
//! - `gemv`: matrix-vector product (weights × activation) with on-the-fly
//!   dequantization.
//! - `shape`: [out_features, in_features] logical shape.
//! - `memory_bytes`: actual consumption on disk / in RAM.
//!
//! Adding a new quantization format means adding one file under `quants/`
//! and implementing `QuantizedGemm`.  Nothing else in the tree needs to change.

pub mod q4_0;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod bf16;
pub mod quantized_tensor;

pub use q4_0::Q4_0;
pub use q4_k::Q4_K;
pub use q5_k::Q5K;
pub use q6_k::Q6K;
pub use bf16::BF16;
pub use quantized_tensor::GgmlQuantizedTensor;

/// Identifiers for every GGUF quantization flavour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizedDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    F8_E5M2,
    F8_E4M3,
    Bf16,
    Iq1M,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Nvfp4,
    Q1_0,
}

impl QuantizedDType {
    /// Bits per weight for this dtype (approximate, for memory estimates).
    pub fn bits_per_weight(&self) -> f32 {
        use QuantizedDType::*;
        match self {
            Q4_0 | Q4_1 => 4.5,   // 2-byte scale + 16 bytes for 32 weights
            Q5_0 | Q5_1 => 5.5,   // 2-byte scale + 20 bytes for 32 weights
            Q8_0 | Q8_1 => 9.0,   // 2-byte scale + 32 bytes for 32 weights
            Q2_K => 2.625,
            Q3_K => 3.4375,
            Q4_K => 4.5,
            Q5_K => 5.5,
            Q6_K => 6.5625,
            Q8_K => 8.5,
            I8 => 8.0,
            I16 => 16.0,
            F16 | Bf16 | F8_E5M2 | F8_E4M3 => 16.0,
            F32 => 32.0,
            F64 => 64.0,
            _ => 16.0,
        }
    }

    /// Block size (elements per quantization block).  `None` means no blocking.
    pub fn block_size(&self) -> Option<usize> {
        use QuantizedDType::*;
        match self {
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => Some(32),
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K => Some(256),
            _ => None,
        }
    }

    /// Bytes per block (scale + quantized values).
    pub fn block_bytes(&self) -> Option<usize> {
        use QuantizedDType::*;
        match self {
            Q4_0 => Some(18),   // f16 scale + 16 bytes
            Q4_1 => Some(20),   // f16 scale + f16 min + 16 bytes
            Q5_0 => Some(22),   // f16 scale + 20 bytes + 2-bit padding
            Q5_1 => Some(24),   // f16 scale + f16 min + 20 bytes + 2-bit padding
            Q8_0 => Some(34),   // f16 scale + 32 bytes
            Q8_1 => Some(36),   // f16 scale + f16 min + 32 bytes
            Q2_K => Some(80),   // complex layout
            Q3_K => Some(110),  // complex layout
            Q4_K => Some(144),  // f16 scale + f16 min + per-subblock scales + 128 nibbles
            Q5_K => Some(176),
            Q6_K => Some(210),
            Q8_K => Some(292),  // f16 scale + f16 min + 256 bytes
            _ => None,
        }
    }
}

/// Native quantized GEMV contract.
pub trait QuantizedGemm: Send + Sync {
    /// Matrix-vector product: `output += weights @ activation`.
    ///
    /// The implementor may accumulate or overwrite; callers must zero `output`
    /// if accumulation is not desired.
    fn gemv(&self, activation: &[f32], output: &mut [f32]);

    /// Logical shape `[out_features, in_features]`.
    fn shape(&self) -> &[usize];

    fn out_features(&self) -> usize {
        self.shape()[0]
    }

    fn in_features(&self) -> usize {
        self.shape()[1]
    }

    /// Concrete dtype identifier (used by dispatchers and loaders).
    fn dtype(&self) -> QuantizedDType;

    /// Total bytes consumed **including** block metadata.
    fn memory_bytes(&self) -> usize;

/// Human-readable summary for diagnostics.
    fn summary(&self) -> String {
        format!(
            "{:?} [{:?}] {:.1}M × {:.1}M ({:.1} GB)",
            self.dtype(),
            self.shape(),
            self.out_features() as f32 / 1_048_576.0,
            self.in_features() as f32 / 1_048_576.0,
            self.memory_bytes() as f32 / (1024.0 * 1024.0 * 1024.0),
        )
    }

    fn row(&self, row_idx: usize) -> Vec<f32> {
        unimplemented!("row extraction not implemented for this dtype")
    }
}

/// Boxed handle so we can store heterogeneous quant types in collections.
pub type QuantizedTensorRef = Box<dyn QuantizedGemm>;
