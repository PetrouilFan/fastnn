//! Production CPU execution of the locked W8A16 format (FN3-23 / #69).
//!
//! `fn3-quant-format-v1` stores signed INT8 projection weights in groups of 64
//! along K, with one binary16 scale and one int16 zero-point per group, and
//! keeps activations in FP32. This module consumes that representation
//! directly:
//!
//! ```text
//! packed codes ──► Σ x·(q − z) per 64-element group, f32, ascending k
//! scales/zeros ─► one f32 multiply per group
//! ```
//!
//! No dequantized weight tensor is materialized, no Q/DQ node is inserted and
//! nothing is approximated: [`kernel::matmul_quantized`] is the normative
//! arithmetic of `docs/quantization/fn3-quant-format-v1.md` §5.
//!
//! # Pieces
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`kernel`] | the fused scalar kernel, a borrowed [`kernel::PackedView`], the [`WorkHint`] thread-budget seam |
//! | [`pack`] | the §4.5 encoder, an owned [`pack::PackedWeight`] with its accuracy report |
//! | [`planner`] | per-projection registration, the §6 fallback rules and the coverage report |
//! | [`backend`] | [`backend::QuantCpuBackend`]: the existing `Backend` trait with registered `MatMul` dispatch |
//!
//! # Ownership and contract
//!
//! This is an **additive** module under `backend/cpu/`. It adds no `OpId`, no
//! `DType` and no `Capabilities` flag, and it does not change
//! `Backend::admits`, `CpuBackend`, the planner, the session, the ONNX importer,
//! the prepared-model format or the Python surface. A caller builds a
//! [`planner::QuantWeightSet`] from a concrete plan and passes a
//! [`backend::QuantCpuBackend`] to `Session::with_runtime`; the session then runs
//! every instruction exactly as before, with the registered projections taking
//! the fused path.
//!
//! # Fallbacks are explicit
//!
//! A projection is registered only when the format's §6 policy accepts it:
//! a static 2-D FP32 right-hand side, `K >= 64`, at least 4096 elements, and a
//! permitted packed copy. Everything else — the token-embedding `wte`, the
//! lm_head transpose, attention score/value products, and any weight the policy
//! rejects — is reported by [`planner::CoverageReport`] with a typed reason and
//! executed by the FP32 reference. A strided activation is a `Partial`
//! registration: the kernel rejects it with `UnsupportedOp` and the backend
//! delegates, so no shape is ever claimed as `Full` that the kernel cannot
//! serve.
//!
//! # Thread budget (FN3-42 / #71)
//!
//! The kernel takes an explicit [`WorkHint`] and never spawns a thread or owns
//! a pool. [`WorkHint::chunks`] partitions *output rows* only; every output
//! element keeps its own ascending-`k`/ascending-group accumulation, so
//! distributing the chunks over a pool is bit-identical. `parallel.rs` does not
//! exist on this branch and is not edited; when #71 lands, its `ThreadBudget`
//! supplies `threads` and `min_rows` to [`WorkHint::new`] and replaces the
//! sequential chunk loop with a pool dispatch over the same `RowChunks`
//! iterator. See [`kernel`] for the full note.

pub mod backend;
pub mod kernel;
pub mod pack;
pub mod planner;

pub use backend::{QuantCpuBackend, QuantStats, WeightStats, QUANT_CPU_BACKEND_NAME, UNPACKED_RHS};
pub use kernel::{matmul_quantized, PackedView, RowChunks, WorkHint};
pub use pack::{PackedWeight, PackingReport, MIN_K, MIN_NUMEL};
pub use planner::{
    registration, ActivationSpec, CoverageReport, FallbackReason, ProjectionCoverage,
    ProjectionKind, ProjectionRegistration, ProjectionSpec, QuantDecision, QuantEntry, QuantPolicy,
    QuantWeightSet, RhsKind, PARTIAL_ACTIVATION_LAYOUT,
};
