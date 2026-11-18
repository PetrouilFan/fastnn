//! The FastNN 3.0 inference core.
//!
//! **Status: temporary module name.** `src/fn3/` exists so the 3.0 rewrite can
//! land in parallel with the 2.x tree without touching `src/tensor`,
//! `src/ir` or `src/backend`. FN3-19 (#65) promotes this tree to its final
//! top-level name and deletes the 2.x modules in one reviewed change. Until
//! then, nothing outside `src/fn3/` may depend on it and nothing inside it may
//! depend on 2.x modules except through the standard library.
//!
//! # Layout
//!
//! ```text
//! dtype.rs        DType: f32 (compute), i64 (indices), i32 (aux), bool (masks)
//! shape.rs        Shape, Strides, DimBound, ShapeEnvelope, LiveDims
//! error.rs        Fn3Error, the single error type
//! device.rs       DeviceId, DeviceRequest, resolution (no silent fallback)
//! storage.rs      AlignedBuffer (64-byte), Arc-shared Storage
//! tensor.rs       TensorMeta, TensorView, TensorMut, Tensor
//! ops/            OpId registry with stable discriminants
//! backend/        Backend trait, Capabilities, Support; backend/cpu/ reference
//! model/          ExecutionPlan, ModelWeights, PreparedModel, Session
//! ```
//!
//! Dependency direction is a DAG:
//!
//! ```text
//! dtype   device   ops
//!   \       |      /
//!        error
//!          |
//!     shape  storage
//!          \   /
//!         tensor
//!           |
//!        backend
//!           |
//!         model
//! ```
//!
//! No module below may import a module above it.
//!
//! # Scope
//!
//! FastNN 3.0 is inference-only. There is no autograd, no optimizer, no
//! training state and no wgpu backend here, and there will not be. The old
//! roadmap's training/autograd/wgpu structural gates are superseded; see
//! `docs/roadmap/fn3-architecture.md` for the supersession table.
//!
//! # Contract entry points
//!
//! - [`Backend`] and [`Support`]: how a backend advertises and rejects work.
//! - [`crate::fn3::model::PreparedModel`] / [`crate::fn3::model::Session`]:
//!   the immutable-model / mutable-session split.
//! - [`Fn3Error`]: the only error type, with actionable context.
//! - `tests/fn3_contracts.rs`: pinned behaviour downstream workers rely on.

pub mod backend;
pub mod cli;
pub mod device;
pub mod dtype;
pub mod error;
pub mod import;
pub mod model;
pub mod ops;
pub mod shape;
pub mod storage;
pub mod tensor;

pub use backend::cpu::CpuBackend;
pub use backend::{Backend, Capabilities, CapabilityFeatures, OpParams, Support};
pub use device::{DeviceId, DeviceRequest, DeviceResolveError, PhysicalDeviceId};
pub use dtype::DType;
pub use error::{Fn3Error, Fn3Result};
pub use model::{ExecutionPlan, Instruction, ModelWeights, PreparedModel, Session};
pub use ops::OpId;
pub use shape::{DimBound, LiveDims, Shape, ShapeEnvelope, Strides, Symbol};
pub use storage::{AlignedBuffer, Storage};
pub use tensor::{Tensor, TensorMeta, TensorMut, TensorView};
