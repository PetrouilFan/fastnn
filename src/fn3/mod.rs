//! The FastNN 3.0 inference core.
//!
//! **This is the primary 3.0 API.** FN3-19 (#65) promoted `src/fn3/` from a
//! parallel tree to the crate's main surface and deleted training and wgpu.
//! The module keeps its `fn3` name so downstream paths, the `fastnn.fn3`
//! Python package and the CLI stay stable for the alpha. It depends on no 2.x
//! module, and no 2.x module depends on it; `src/lib.rs` is the only shared
//! file.
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
//! FastNN 3.0 is inference-only. There is no autograd, no optimizer and no
//! training state here. See `docs/roadmap/fn3-architecture.md` for the
//! contract and the superseded 2.x gates.
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
