//! # fastnn — inference-only neural-network runtime
//!
//! [`inference`] is the whole product: tensor storage, strict ONNX import, a
//! target-aware planner, a versioned prepared-model artifact, sessions that own
//! their thread pool and KV cache, a CPU backend with a pure-Rust reference and
//! an optional Vulkan backend, the `fastnn-infer` CLI and the Python binding.
//!
//! There is no autograd, optimizer, training loop or wgpu backend in this
//! crate, and no second inference surface: the 2.x AOT tree was archived at tag
//! `archive/3.0` and is not compiled here.
//!
//! ```text
//! ONNX ──import──► graph ──plan──► PreparedModel ──session──► run
//! ```
//!
//! Start at [`inference::model::PreparedModel`] and [`inference::model::Session`].

// Keep byte decoding on slice chunks: Clippy's array-chunk rewrite changes
// TryInto inference in the packed-type macros.
#![allow(clippy::chunks_exact_to_as_chunks)]
#![allow(clippy::needless_range_loop)]
// SIMD and GPU compute kernels take many raw dimension parameters for
// zero-overhead dispatch.
#![allow(clippy::too_many_arguments)]

#[path = "fn3/mod.rs"]
pub mod inference;

#[cfg(feature = "python")]
pub mod python;
