//! The FastNN 3.0 inference core.
//!
//! **This is the primary 3.0 API.** FN3-19 (#65) promoted `src/fn3/` from a
//! parallel tree to the crate's main surface and deleted training and wgpu.
//! The public names are `fastnn::inference`, `fastnn.inference` and
//! `fastnn-infer`; the source stays in `src/fn3/`. The core depends on no 2.x
//! module, and no 2.x module depends on it.
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
//! ir/             typed graph: DimExpr, TensorType, SourceLoc, Graph
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
//! - [`crate::inference::model::PreparedModel`] / [`crate::inference::model::Session`]:
//!   the immutable-model / mutable-session split.
//! - [`Fn3Error`]: the only error type, with actionable context.
//! - `tests/fn3_contracts.rs`: pinned behaviour downstream workers rely on.

pub mod api;
pub mod backend;
pub mod cli;
pub mod device;
pub mod digest;
pub mod dtype;
pub mod error;
pub mod import;
pub mod ir;
pub mod model;
pub mod ops;
pub mod quant;
pub mod runtime;
pub mod shape;
pub mod storage;
pub mod tensor;

pub use api::{
    doctor, doctor_with_shapes, inspect, load, prepare, prepare_with_shapes, DoctorIssue,
    DoctorReport, DtypeUse, DynamicSymbolUse, PrepareReport, TargetPlanReport,
};
pub use backend::cpu::CpuBackend;
pub use backend::{Backend, Capabilities, CapabilityFeatures, OpParams, Support};
pub use device::{DeviceId, DeviceRequest, DeviceResolveError, PhysicalDeviceId};
pub use dtype::DType;
pub use error::{Fn3Error, Fn3Result};
pub use model::format::{
    Artifact, ArtifactHeader, ArtifactInspection, PlanSummary, Precision, PreparedArtifact,
    SectionHash, SectionInfo,
};
pub use model::{ExecutionPlan, Instruction, ModelWeights, PreparedModel, Session};
pub use ops::OpId;
pub use runtime::{
    Arena as RuntimeArena, HostTensor, Plan as RuntimePlan, Segment, SegmentOp,
    Session as RuntimeSession, SessionThreads, Slot, SlotId, SlotSource, Threading,
};
pub use shape::{DimBound, LiveDims, Shape, ShapeEnvelope, Strides, Symbol};
pub use storage::{AlignedBuffer, Storage};
pub use tensor::{Tensor, TensorMeta, TensorMut, TensorView};
