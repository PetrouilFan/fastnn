//! # fastnn — High-Performance Neural Network Inference Library
//!
//! fastnn provides a complete **ahead-of-time (AOT) compiler pipeline** built on a
//! first-class IR (`ComputeGraph`). The pipeline compiles computation graphs through
//! shape inference, operator fusion, optional weight quantization (U4/U8), and memory
//! planning before dispatching to a backend (CPU or WGPU).
//!
//! # Quick Start
//!
//! ```rust
//! use fastnn::ir::builder::GraphBuilder;
//! use fastnn::ir::node::{DimExpr, IrDType, TensorType};
//! use fastnn::backend::cpu::CpuBackend;
//!
//! let gb = GraphBuilder::new();
//! let input = gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(784)], IrDType::F32);
//! let weight_tt = TensorType::new(vec![DimExpr::Known(784), DimExpr::Known(10)], IrDType::F32);
//! let weight_bytes: Vec<u8> = vec![0u8; 784 * 10 * 4]; // placeholder
//! let weight = gb.constant(&weight_bytes, weight_tt);
//! let output = gb.matmul(&input, &weight);
//!
//! // Compile with 4-bit quantization
//! let result = gb.compile_with_quantize(&[&output], CpuBackend, Some(4));
//! ```
//!
//! See [`ir::builder::GraphBuilder`] and [`backend::executor::GraphExecutor`] for the
//! full API.

#![allow(clippy::needless_range_loop)]

pub mod error;
pub mod io;
pub mod iterator;
pub mod onnx;
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

// v2.0.0: The old `kernels` module and DAG dispatcher have been removed.
// All operations route through the AOT pipeline (ir/ + backend/ + compiler/).

// Re-export core types
pub use storage::{DType, Device};
pub use tensor::Tensor;

// v2.1: Legacy packed layer classes (PackedLinear, PackedConv2d, etc.)
// and their kernel backends have been removed. The AOT pipeline's
// quantized dispatch (matmul_u4, conv2d_u4) uses PackedTensor and
// dtypes internally via backend/cpu/microkernels.rs.
pub mod packed_tensor;
pub mod dtypes;
// swar ops — used by microkernels.rs for quantized relu backward
pub mod swar;

// v2.0.0 AOT compiler infrastructure
pub mod ir;
pub mod backend;
pub mod compiler;
pub mod autograd;

// Re-export the graph builder API for convenience
pub use ir::builder::{GraphBuilder, GraphTensor};

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
