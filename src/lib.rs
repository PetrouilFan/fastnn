//! # fastnn — inference-only neural-network runtime
//!
//! FastNN 3.0 is an inference product. The primary API is [`fn3`]: tensor
//! storage, a portable CPU reference, a Vulkan compute backend, strict ONNX
//! import, a versioned prepared-model artifact, a bounded planner, sessions
//! that own their KV cache, an inference CLI and a Python binding. There is no
//! autograd, optimizer, training loop or wgpu backend in this crate.
//!
//! ```text
//! ONNX ──import──► graph ──plan──► PreparedModel ──session──► prefill/decode
//!                                   (CPU or Vulkan)
//! ```
//!
//! Start with [`fn3::model::PreparedModel`] and [`fn3::model::Session`]; the
//! CLI (`fn3-infer`) and the `fastnn.fn3` Python package call the same path.
//!
//! # Secondary 2.x AOT tree
//!
//! The 2.x modules remain compiled as a secondary, inference-capable surface:
//! the graph IR ([`ir`]), compiler passes ([`compiler`]), the CPU executor and
//! prepared-plan runtime ([`backend`]), the eager [`tensor`]/[`nn`] forward
//! API, ONNX conversion ([`onnx`]) and `.fnn` I/O ([`io`]). They are kept for
//! the legacy AOT path and its quantized CPU pipeline; new code should use
//! [`fn3`].
//!
//! ```rust
//! use fastnn::ir::builder::GraphBuilder;
//! use fastnn::ir::{DimExpr, IrDType, TensorType};
//! use fastnn::backend::cpu::CpuBackend;
//!
//! let gb = GraphBuilder::new();
//! let input = gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(784)], IrDType::F32);
//! let weight_tt = TensorType::new(vec![DimExpr::Known(784), DimExpr::Known(10)], IrDType::F32);
//! let weight_bytes: Vec<u8> = vec![0u8; 784 * 10 * 4]; // placeholder
//! let weight = gb.constant(&weight_bytes, weight_tt);
//! let output = gb.matmul(&input, &weight);
//!
//! let result = gb.compile_and_execute(&[&output], CpuBackend, &[&vec![0u8; 784 * 4], &weight_bytes]);
//! ```
//!
//! See [`ir::builder::GraphBuilder`] and [`backend::executor::GraphExecutor`]
//! for the 2.x AOT API.

// Keep byte decoding on slice chunks: Clippy's array-chunk rewrite changes
// TryInto inference in packed-type macros. This is a style lint, not validation.
#![allow(clippy::chunks_exact_to_as_chunks)]
#![allow(clippy::needless_range_loop)]
// SIMD compute kernels require many raw pointer + dimension parameters for zero-overhead dispatch.
#![allow(clippy::too_many_arguments)]

// ── 3.0 inference core: the primary API ─────────────────────────────────────
//
// Pure Rust, no optional features and no dependency on the 2.x modules. See
// docs/roadmap/fn3-architecture.md for the contract.
pub mod fn3;

// ── Shared infrastructure ───────────────────────────────────────────────────
pub mod error;
pub mod io;
pub mod utils;

// v2.4: shared macros for reducing boilerplate
pub mod macros;

// ── Secondary 2.x AOT tree (inference only; no training) ────────────────────
pub mod backend;
pub mod compiler;
pub mod ir;

pub mod nn;
pub mod onnx;
#[cfg(feature = "python")]
pub mod python;
pub mod residual;
pub mod storage;
pub mod storage_pool;

pub mod tensor;
pub mod types;

// v2.1 cleanup: legacy packed layer classes (PackedLinear, PackedConv2d, etc.)
// and their kernel backends have been removed. The AOT pipeline's quantized
// dispatch (matmul_u4, conv2d_u4) uses PackedTensor and dtypes internally via
// backend/cpu/microkernels.rs.
pub mod dtypes;
pub mod packed_tensor;
// swar ops — used by microkernels.rs for packed integer kernels
pub mod swar;

pub use error::{FastnnError, FastnnResult};

// Re-export core types
pub use storage::{DType, Device};
pub use tensor::Tensor;

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

pub(crate) fn fill_random_f32(values: &mut [f32]) {
    if let Some(rng_mutex) = SEEDED_RNG.get() {
        let mut rng = rng_mutex.lock();
        rng.fill(values);
    } else {
        rand::thread_rng().fill(values);
    }
}

#[cfg(any(feature = "python", test))]
pub(crate) fn set_seeded_rng(seed: u64) {
    use rand::SeedableRng;
    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mutex = SEEDED_RNG.get_or_init(|| Mutex::new(rng.clone()));
    *mutex.lock() = rng;
}

#[cfg(test)]
mod seeded_rng_tests {
    use super::*;
    use crate::nn::dropout::{Dropout, Dropout2d};
    use crate::nn::Module;
    use crate::tensor::Tensor;

    fn values(tensor: &Tensor) -> Vec<f32> {
        tensor.as_f32_slice().to_vec()
    }

    #[test]
    fn dropout_variants_use_the_process_seeded_rng() {
        let dropout = Dropout::new(0.5);
        let input = Tensor::from_vec(vec![1.0; 32], vec![32]);

        set_seeded_rng(17);
        let first = values(&dropout.forward(&input));
        set_seeded_rng(17);
        let second = values(&dropout.forward(&input));

        assert_eq!(first, second);

        let dropout = Dropout2d::new(0.5);
        let input = Tensor::from_vec(vec![1.0; 16], vec![1, 4, 2, 2]);

        set_seeded_rng(29);
        let first = values(&dropout.forward(&input));
        set_seeded_rng(29);
        let second = values(&dropout.forward(&input));

        assert_eq!(first, second);
    }
}
