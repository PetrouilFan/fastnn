#![allow(clippy::needless_range_loop)]

pub mod autograd;
pub mod dispatcher;
pub mod error;
pub mod io;
pub mod iterator;
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

// Re-export core types
pub use storage::{DType, Device};
pub use tensor::Tensor;

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

