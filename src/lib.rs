#![allow(clippy::needless_range_loop)]

pub mod error;
pub mod io;
pub mod iterator;
// DEPRECATED: legacy v1.x kernel implementations — kept for dag.rs backward compat
pub mod kernels;
pub mod nn;
pub mod optim;
pub mod python;
pub mod residual;
pub mod storage;
pub mod storage_pool;
pub mod storage_quantized;
pub mod tensor;
pub use error::{FastnnError, FastnnResult};
pub use storage_quantized::QuantizedTensor;

// DEPRECATED: legacy v1.x runtime dispatch table — kept for dag.rs backward compat
pub mod dispatcher;

// Re-export core types
pub use storage::{DType, Device};
pub use tensor::Tensor;

// Native packed precision modules
pub mod backends;
pub mod dtypes;
pub mod packed_conv;
pub mod packed_layer;
pub mod packed_tensor;
pub mod packed_train;
pub mod swar;

// v2.0.0 AOT compiler infrastructure
pub mod ir;
pub mod backend;
pub mod compiler;
pub mod autograd;

// Re-export the graph builder API for convenience
pub use ir::builder::{GraphBuilder, GraphTensor};

// Re-export packed precision public API
pub use dtypes::{F16x2, F32x1, PackedWord, U4x8, U8x4};
pub use packed_conv::{Conv2d16, Conv2d32, Conv2d4, Conv2d8, PackedConv2d};
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
