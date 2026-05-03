#![allow(clippy::needless_range_loop)]

mod autograd;
mod dispatcher;
mod io;
mod iterator;
pub mod kernels;
mod llm;
mod nn;
mod optim;
mod python;
mod residual;
mod quants;
mod storage;
mod storage_pool;
mod storage_quantized;
mod tensor;
mod train;

pub use storage_quantized::QuantizedTensor;

// Re-export core types
pub use storage::{DType, Device};
pub use tensor::Tensor;

// Re-export GGUF loader and quantized types
pub use io::gguf::{GgufError, GgufFile, GgufTensorInfo};
pub use quants::{GgmlQuantizedTensor, QuantizedDType, QuantizedGemm};
pub use quants::{Q4_0, Q4_K, Q6K};

// Re-export LLM types
pub use llm::config::{LlmConfig, LayerConfig};
pub use llm::model::LlmModel;
pub use llm::embedding::Embedding;
pub use llm::attention::AttentionLayer;
pub use llm::model::TransformerLayer;

// Native packed precision modules
pub mod backends;
pub mod dtypes;
pub mod packed_layer;
pub mod packed_tensor;
pub mod packed_train;
pub mod swar;

// Re-export packed precision public API
pub use dtypes::{F16x2, F32x1, PackedWord, U4x8, U8x4};
pub use packed_layer::{is_wgpu, use_cpu, use_wgpu};
pub use packed_layer::{Linear16, Linear32, Linear4, Linear8, PackedLinear};
pub use packed_tensor::PackedTensor;
pub use packed_train::MasterWeightOptimizer;

use parking_lot::Mutex;
use rand::Rng;
use std::sync::OnceLock;

static SEEDED_RNG: OnceLock<Mutex<rand::rngs::StdRng>> = OnceLock::new();

pub fn random_f32() -> f32 {
    if let Some(rng_mutex) = SEEDED_RNG.get() {
        let mut rng = rng_mutex.lock();
        rng.gen()
    } else {
        let mut rng = rand::thread_rng();
        rng.gen()
    }
}

pub(crate) fn set_seeded_rng(seed: u64) {
    use rand::SeedableRng;
    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mutex = SEEDED_RNG.get_or_init(|| Mutex::new(rng.clone()));
    *mutex.lock() = rng;
}


