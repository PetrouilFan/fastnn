//! The FastNN inference core.
//!
//! **This is the crate's main surface.** The public names are
//! `fastnn::inference`, `fastnn.inference` and `fastnn-infer`; the source stays
//! in `src/fn3/`. There is no autograd, optimizer, training loop or wgpu
//! backend here.
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
//! ir/             typed graph: DimExpr, TensorType, SourceLoc, shape inference
//! import/         strict ONNX (opset 12..=22) -> typed graph, shape binding
//! ops/            OpId registry with stable discriminants
//! runtime/        plan, fusion, arena, session loop, scalar kernels
//! backend/        the optional engines: oneDNN (cpu-onednn), Vulkan (unwired)
//! model/format/   the prepared artifact: header, graph, plans, weights
//! ```
//!
//! The prepare path is import -> plan -> artifact; the run path is artifact ->
//! session -> arena. Planning and import belong in Rust; the Python binding and
//! the CLI are facades over [`api`].
//!
//! # Contract entry points
//!
//! - [`api::prepare`] / [`api::load`] / [`api::inspect`] / [`api::doctor`]:
//!   the four verbs every surface exposes.
//! - [`runtime::Session`]: one arena, one worker pool, one immutable plan.
//! - [`Fn3Error`]: the only error type, with actionable context.
//! - `tests/fn3_cnn_run.rs` and `tests/fn3_onednn.rs`: engine parity;
//!   `tests/fn3_artifact_v2.rs`: format contract.

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
pub use ops::OpId;
pub use runtime::{
    Arena as RuntimeArena, HostTensor, Plan as RuntimePlan, Segment, SegmentOp,
    Session as RuntimeSession, SessionThreads, Slot, SlotId, SlotSource, Threading,
};
pub use shape::{DimBound, LiveDims, Shape, ShapeEnvelope, Strides, Symbol};
pub use storage::{AlignedBuffer, Storage};
pub use tensor::{Tensor, TensorMeta, TensorMut, TensorView};
