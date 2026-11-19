//! Planner registration and coverage for the W8A16 CPU path (FN3-23 / #69).
//!
//! # What "registration" means here
//!
//! The format contract is per projection, not per operation: `MatMul` stays
//! `Support::Full` on the CPU because the FP32 kernel serves every documented
//! shape, and the W8A16 path is an alternative *weight representation* for one
//! concrete `MatMul` whose right-hand side is a static 2-D FP32 weight. Adding a
//! quantized `OpId`, changing [`crate::fn3::backend::Backend::admits`] or
//! claiming a new dtype would change the backend contract; none of that happens
//! here.
//!
//! Instead a [`ProjectionSpec`] describes one right-hand side and this module
//! answers the same three-valued question the backend contract uses:
//!
//! | [`ProjectionRegistration`] | [`Support`] | Meaning |
//! |---|---|---|
//! | `Admitted` | `Full` | the fused kernel serves this concrete invocation |
//! | `Partial { reason }` | `Partial { reason }` | the packed weight is valid, but this invocation's activation layout is outside the kernel's named subset; the kernel rejects it with `UnsupportedOp` and the backend delegates |
//! | `Fallback { reason }` | `Unsupported { reason }` | this projection is not registered at all; the caller must use FP32 |
//!
//! `Full` is only reported when the kernel can actually execute the invocation:
//! a 2-D FP32 static weight with `k >= 64`, at least 4096 elements, a permitted
//! packed copy, a contiguous FP32 activation of rank >= 2, and matching
//! contraction/output geometry. A `Partial` registration is admitted only
//! through the existing contract: [`super::kernel::matmul_quantized`] rejects
//! the subset it does not implement (it does not approximate), and
//! [`super::backend::QuantCpuBackend`] counts the rejection and delegates to the
//! FP32 path rather than running a different arithmetic.
//!
//! # Fallback policy (the format's §6)
//!
//! * a computed right-hand side (a transpose, a KV slice, an activation) is
//!   [`FallbackReason::DynamicRhs`];
//! * a non-2-D or non-F32 weight falls back;
//! * `K < 64` or `numel < 4096` falls back because the metadata would dominate;
//! * a weight shared with a non-MatMul consumer falls back unless a packed copy
//!   is permitted (a packed copy leaves the original FP32 weight untouched).
//!
//! Tails are *not* a fallback: `K` from 64 upward with any tail length is part
//! of the format and of the kernel.
//!
//! # Building the set from a plan
//!
//! [`QuantWeightSet::from_plan`] walks a concrete [`Plan`] and registers the
//! right-hand side of every `MatMul` whose slot role is
//! [`SlotRole::Weight`]. That is the same information the session's
//! `materialise` step uses, so the pointer key it records is the exact storage
//! the session will later pass to `Backend::execute_op`; a weight the plan does
//! not use as a MatMul right-hand side (the token embedding table, `wte`) is
//! never packed. The lm_head `MatMul` in the pinned GPT-2 graph takes
//! `transpose(wte)` — an instruction output, not a weight slot — so it is
//! reported as a dynamic right-hand side and stays FP32.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;

use crate::fn3::backend::Support;
use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::model::{ModelWeights, Plan, SlotRole};
use crate::fn3::ops::OpId;
use crate::fn3::shape::Shape;
use crate::fn3::tensor::{Tensor, TensorView};

use super::pack::{PackedWeight, MIN_K, MIN_NUMEL};

/// Which projection family a weight belongs to, from its model name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProjectionKind {
    /// `c_attn`: the fused Q/K/V projection.
    Qkv,
    /// `attn.c_proj`: the attention output projection.
    AttnOut,
    /// `mlp.c_fc`: the MLP input projection.
    MlpIn,
    /// `mlp.c_proj`: the MLP output projection.
    MlpOut,
    /// `lm_head` / the tied token-embedding transpose.
    LmHead,
    /// Anything else with a packed weight.
    Other,
}

impl ProjectionKind {
    /// Stable label for the coverage table and reports.
    pub fn label(self) -> &'static str {
        match self {
            ProjectionKind::Qkv => "qkv",
            ProjectionKind::AttnOut => "attn_out",
            ProjectionKind::MlpIn => "mlp_in",
            ProjectionKind::MlpOut => "mlp_out",
            ProjectionKind::LmHead => "lm_head",
            ProjectionKind::Other => "other",
        }
    }

    /// Classify a model weight name.
    pub fn from_name(name: &str) -> ProjectionKind {
        if name.contains("c_attn") {
            ProjectionKind::Qkv
        } else if name.contains("attn.c_proj") {
            ProjectionKind::AttnOut
        } else if name.contains("mlp.c_fc") {
            ProjectionKind::MlpIn
        } else if name.contains("mlp.c_proj") {
            ProjectionKind::MlpOut
        } else if name.contains("lm_head") || name.contains("wte") {
            ProjectionKind::LmHead
        } else {
            ProjectionKind::Other
        }
    }
}

impl fmt::Display for ProjectionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Where the right-hand side of a `MatMul` comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RhsKind {
    /// A static model weight slot.
    StaticWeight,
    /// A value computed in the step (dynamic right-hand side).
    Computed {
        /// What kind of value it is, for the fallback diagnostic.
        what: &'static str,
    },
}

/// One concrete projection the planner is considering.
#[derive(Debug, Clone)]
pub struct ProjectionSpec {
    /// Weight name, or the graph value name of a computed right-hand side.
    pub name: String,
    /// Projection family.
    pub kind: ProjectionKind,
    /// Static weight or computed value.
    pub rhs: RhsKind,
    /// Declared weight dtype (meaningless for a computed right-hand side).
    pub weight_dtype: DType,
    /// Declared weight rank (meaningless for a computed right-hand side).
    pub weight_rank: usize,
    /// Contraction length K.
    pub k: u64,
    /// Output rows N.
    pub n: u64,
    /// `true` if another consumer (not a MatMul) shares this weight.
    pub shared_with_non_matmul: bool,
    /// `true` if the caller permits a separate packed copy.
    pub packed_copy_permitted: bool,
}

impl ProjectionSpec {
    /// A static 2-D FP32 MatMul right-hand side of shape `[K, N]`.
    pub fn static_weight(name: impl Into<String>, shape: &Shape, dtype: DType) -> ProjectionSpec {
        let rank = shape.rank();
        let dims = shape.dims();
        let (k, n) = if rank == 2 {
            (dims[0], dims[1])
        } else {
            (0, 0)
        };
        let name = name.into();
        ProjectionSpec {
            kind: ProjectionKind::from_name(&name),
            name,
            rhs: RhsKind::StaticWeight,
            weight_dtype: dtype,
            weight_rank: rank,
            k,
            n,
            shared_with_non_matmul: false,
            packed_copy_permitted: true,
        }
    }

    /// A computed right-hand side with the shape the plan resolved.
    pub fn computed(name: impl Into<String>, what: &'static str, shape: &Shape) -> ProjectionSpec {
        let rank = shape.rank();
        let dims = shape.dims();
        let (k, n) = if rank == 2 {
            (dims[0], dims[1])
        } else {
            (0, 0)
        };
        let name = name.into();
        ProjectionSpec {
            kind: ProjectionKind::from_name(&name),
            name,
            rhs: RhsKind::Computed { what },
            weight_dtype: DType::F32,
            weight_rank: rank,
            k,
            n,
            shared_with_non_matmul: false,
            packed_copy_permitted: true,
        }
    }

    /// Builder: the weight is also read by a non-MatMul consumer.
    pub fn with_shared_weight(mut self, shared: bool) -> ProjectionSpec {
        self.shared_with_non_matmul = shared;
        self
    }

    /// Builder: a separate packed copy is (or is not) permitted.
    pub fn with_packed_copy(mut self, permitted: bool) -> ProjectionSpec {
        self.packed_copy_permitted = permitted;
        self
    }

    /// Builder: projection family, when the name is not conventional.
    pub fn with_kind(mut self, kind: ProjectionKind) -> ProjectionSpec {
        self.kind = kind;
        self
    }

    /// `k * n` without wrapping; `0` for a non-2-D shape.
    pub fn numel(&self) -> u64 {
        self.k.saturating_mul(self.n)
    }
}

/// Why a projection stays on the FP32 path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FallbackReason {
    /// The right-hand side is computed in the step, not a static weight.
    DynamicRhs {
        /// What kind of value it is.
        what: &'static str,
    },
    /// The weight is not 2-D.
    NonTwoDimensionalRhs {
        /// Declared rank.
        rank: usize,
    },
    /// The weight is not FP32.
    NonF32Weight {
        /// Declared dtype.
        dtype: DType,
    },
    /// A shared weight has no permitted packed copy.
    SharedWeightWithoutPackedCopy,
    /// `K` is below the policy minimum.
    KBelowMinimum {
        /// Found contraction length.
        k: u64,
        /// Policy minimum.
        minimum: u64,
    },
    /// The element count is below the policy minimum.
    NumelBelowMinimum {
        /// Found element count.
        numel: u64,
        /// Policy minimum.
        minimum: u64,
    },
    /// The activation is not FP32.
    NonF32Activation {
        /// Found dtype.
        dtype: DType,
    },
    /// The activation has rank below 2.
    ActivationRankBelowTwo {
        /// Found rank.
        rank: usize,
    },
    /// The activation's contraction length disagrees with the packed weight.
    ContractionMismatch {
        /// Found K.
        found: u64,
        /// Packed weight K.
        expected: u64,
    },
}

impl FallbackReason {
    /// Stable machine-readable code, used by the counters and the report.
    pub fn code(&self) -> &'static str {
        match self {
            FallbackReason::DynamicRhs { .. } => "dynamic-rhs",
            FallbackReason::NonTwoDimensionalRhs { .. } => "rhs-rank",
            FallbackReason::NonF32Weight { .. } => "weight-dtype",
            FallbackReason::SharedWeightWithoutPackedCopy => "shared-weight",
            FallbackReason::KBelowMinimum { .. } => "k-below-min",
            FallbackReason::NumelBelowMinimum { .. } => "numel-below-min",
            FallbackReason::NonF32Activation { .. } => "activation-dtype",
            FallbackReason::ActivationRankBelowTwo { .. } => "activation-rank",
            FallbackReason::ContractionMismatch { .. } => "contraction-mismatch",
        }
    }
}

impl fmt::Display for FallbackReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FallbackReason::DynamicRhs { what } => {
                write!(f, "the right-hand side is {what}, not a static weight")
            }
            FallbackReason::NonTwoDimensionalRhs { rank } => {
                write!(f, "the weight has rank {rank}, not 2")
            }
            FallbackReason::NonF32Weight { dtype } => {
                write!(f, "the weight is {dtype}, not f32")
            }
            FallbackReason::SharedWeightWithoutPackedCopy => f.write_str(
                "the weight is shared with a non-MatMul consumer and no packed copy is permitted",
            ),
            FallbackReason::KBelowMinimum { k, minimum } => {
                write!(f, "K = {k} is below the policy minimum {minimum}")
            }
            FallbackReason::NumelBelowMinimum { numel, minimum } => {
                write!(f, "{numel} elements are below the policy minimum {minimum}")
            }
            FallbackReason::NonF32Activation { dtype } => {
                write!(f, "the activation is {dtype}, not f32")
            }
            FallbackReason::ActivationRankBelowTwo { rank } => {
                write!(f, "the activation has rank {rank}, not >= 2")
            }
            FallbackReason::ContractionMismatch { found, expected } => {
                write!(
                    f,
                    "the activation contracts {found}, the packed weight {expected}"
                )
            }
        }
    }
}

/// The planner's decision for one projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantDecision {
    /// Register the weight and use the fused kernel.
    Quantize,
    /// Keep the FP32 path, for the named reason.
    Fallback(FallbackReason),
}

/// The format's §6 policy, as data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantPolicy {
    /// Smallest accepted contraction length.
    pub min_k: usize,
    /// Smallest accepted element count.
    pub min_numel: u64,
    /// Whether a packed copy may be created for a shared weight.
    pub allow_packed_copy: bool,
}

impl Default for QuantPolicy {
    fn default() -> QuantPolicy {
        QuantPolicy {
            min_k: MIN_K,
            min_numel: MIN_NUMEL,
            allow_packed_copy: true,
        }
    }
}

impl QuantPolicy {
    /// Apply the fallback rules to one projection.
    pub fn classify(&self, spec: &ProjectionSpec) -> QuantDecision {
        match &spec.rhs {
            RhsKind::Computed { what } => {
                return QuantDecision::Fallback(FallbackReason::DynamicRhs { what });
            }
            RhsKind::StaticWeight => {}
        }
        if spec.weight_rank != 2 {
            return QuantDecision::Fallback(FallbackReason::NonTwoDimensionalRhs {
                rank: spec.weight_rank,
            });
        }
        if spec.weight_dtype != DType::F32 {
            return QuantDecision::Fallback(FallbackReason::NonF32Weight {
                dtype: spec.weight_dtype,
            });
        }
        if spec.shared_with_non_matmul && !(spec.packed_copy_permitted && self.allow_packed_copy) {
            return QuantDecision::Fallback(FallbackReason::SharedWeightWithoutPackedCopy);
        }
        if spec.k < self.min_k as u64 {
            return QuantDecision::Fallback(FallbackReason::KBelowMinimum {
                k: spec.k,
                minimum: self.min_k as u64,
            });
        }
        let numel = spec.numel();
        if numel < self.min_numel {
            return QuantDecision::Fallback(FallbackReason::NumelBelowMinimum {
                numel,
                minimum: self.min_numel,
            });
        }
        QuantDecision::Quantize
    }
}

/// The activation layout of one concrete invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationSpec {
    /// Activation dtype.
    pub dtype: DType,
    /// Activation rank.
    pub rank: usize,
    /// Contraction length.
    pub k: u64,
    /// `true` when the activation is one contiguous row-major block.
    pub contiguous: bool,
}

impl ActivationSpec {
    /// Describe a tensor view.
    pub fn of(view: &TensorView<'_>) -> ActivationSpec {
        let rank = view.shape().rank();
        let k = if rank >= 2 {
            view.shape().dims()[rank - 1]
        } else {
            0
        };
        ActivationSpec {
            dtype: view.meta().dtype(),
            rank,
            k,
            contiguous: view.is_contiguous(),
        }
    }
}

/// The registration verdict for one concrete invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionRegistration {
    /// The fused kernel serves this invocation completely.
    Admitted,
    /// The packed weight is valid, but this activation layout is outside the
    /// kernel's named subset. The kernel rejects it; the backend delegates.
    Partial {
        /// What the kernel does serve.
        reason: &'static str,
    },
    /// The projection is not registered at all.
    Fallback {
        /// Why.
        reason: FallbackReason,
    },
}

/// The subset a `Partial` registration names.
pub const PARTIAL_ACTIVATION_LAYOUT: &str =
    "the fused W8A16 kernel serves contiguous FP32 activations of rank >= 2; a strided \
     invocation is rejected with UnsupportedOp and delegated to the FP32 MatMul kernel";

impl ProjectionRegistration {
    /// Map the registration onto the backend's three-valued contract.
    pub fn support(&self) -> Support {
        match self {
            ProjectionRegistration::Admitted => Support::Full,
            ProjectionRegistration::Partial { reason } => Support::Partial {
                reason: (*reason).to_string(),
            },
            ProjectionRegistration::Fallback { reason } => Support::Unsupported {
                reason: reason.to_string(),
            },
        }
    }

    /// Stable counter code.
    pub fn code(&self) -> &'static str {
        match self {
            ProjectionRegistration::Admitted => "admitted",
            ProjectionRegistration::Partial { .. } => "activation-layout",
            ProjectionRegistration::Fallback { reason } => reason.code(),
        }
    }

    /// `true` for [`ProjectionRegistration::Admitted`].
    pub fn is_admitted(&self) -> bool {
        matches!(self, ProjectionRegistration::Admitted)
    }
}

/// Classify one concrete invocation.
pub fn registration(
    spec: &ProjectionSpec,
    activation: ActivationSpec,
    policy: &QuantPolicy,
) -> ProjectionRegistration {
    if let QuantDecision::Fallback(reason) = policy.classify(spec) {
        return ProjectionRegistration::Fallback { reason };
    }
    if activation.dtype != DType::F32 {
        return ProjectionRegistration::Fallback {
            reason: FallbackReason::NonF32Activation {
                dtype: activation.dtype,
            },
        };
    }
    if activation.rank < 2 {
        return ProjectionRegistration::Fallback {
            reason: FallbackReason::ActivationRankBelowTwo {
                rank: activation.rank,
            },
        };
    }
    if activation.k != spec.k {
        return ProjectionRegistration::Fallback {
            reason: FallbackReason::ContractionMismatch {
                found: activation.k,
                expected: spec.k,
            },
        };
    }
    if !activation.contiguous {
        return ProjectionRegistration::Partial {
            reason: PARTIAL_ACTIVATION_LAYOUT,
        };
    }
    ProjectionRegistration::Admitted
}

/// One registered weight and the projection it serves.
#[derive(Debug, Clone)]
pub struct QuantEntry {
    weight: Arc<PackedWeight>,
    spec: ProjectionSpec,
}

impl QuantEntry {
    /// The packed weight.
    pub fn weight(&self) -> &PackedWeight {
        &self.weight
    }

    /// The projection description.
    pub fn spec(&self) -> &ProjectionSpec {
        &self.spec
    }
}

/// One row of the coverage report.
#[derive(Debug, Clone)]
pub struct ProjectionCoverage {
    /// Weight or value name.
    pub name: String,
    /// Projection family.
    pub kind: ProjectionKind,
    /// Planner decision.
    pub decision: QuantDecision,
    /// Contraction length.
    pub k: u64,
    /// Output rows.
    pub n: u64,
    /// Resident packed bytes (0 for a fallback).
    pub packed_bytes: u64,
    /// Per-weight relative Frobenius error (0 for a fallback).
    pub relative_frobenius: Option<f64>,
}

/// Every projection a plan considers, with its decision.
#[derive(Debug, Clone, Default)]
pub struct CoverageReport {
    projections: Vec<ProjectionCoverage>,
}

impl CoverageReport {
    /// Add one row.
    pub fn push(&mut self, row: ProjectionCoverage) {
        self.projections.push(row);
    }

    /// The rows, in plan order.
    pub fn projections(&self) -> &[ProjectionCoverage] {
        &self.projections
    }

    /// How many projections use the fused kernel.
    pub fn quantized(&self) -> usize {
        self.projections
            .iter()
            .filter(|row| row.decision == QuantDecision::Quantize)
            .count()
    }

    /// How many projections stay on FP32.
    pub fn fallback(&self) -> usize {
        self.projections.len() - self.quantized()
    }

    /// Fallback counts keyed by [`FallbackReason::code`].
    pub fn fallback_reasons(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for row in &self.projections {
            if let QuantDecision::Fallback(reason) = &row.decision {
                *counts.entry(reason.code()).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Quantized projections of one family.
    pub fn quantized_of(&self, kind: ProjectionKind) -> usize {
        self.projections
            .iter()
            .filter(|row| row.kind == kind && row.decision == QuantDecision::Quantize)
            .count()
    }

    /// FP32 bytes the quantized projections would occupy.
    pub fn fp32_bytes(&self) -> u64 {
        self.projections
            .iter()
            .filter(|row| row.decision == QuantDecision::Quantize)
            .map(|row| row.k.saturating_mul(row.n).saturating_mul(4))
            .sum()
    }

    /// Packed bytes the quantized projections occupy.
    pub fn packed_bytes(&self) -> u64 {
        self.projections
            .iter()
            .filter(|row| row.decision == QuantDecision::Quantize)
            .map(|row| row.packed_bytes)
            .sum()
    }

    /// Packed bytes as a fraction of the FP32 bytes they replace.
    pub fn storage_ratio(&self) -> f64 {
        let fp32 = self.fp32_bytes();
        if fp32 == 0 {
            0.0
        } else {
            self.packed_bytes() as f64 / fp32 as f64
        }
    }
}

/// Packed weights keyed by the storage pointer the session will present.
#[derive(Debug, Clone, Default)]
pub struct QuantWeightSet {
    entries: Vec<QuantEntry>,
    by_pointer: HashMap<usize, usize>,
    _keepalive: Vec<Arc<Tensor>>,
    coverage: CoverageReport,
}

impl QuantWeightSet {
    /// Register every static MatMul right-hand side of `plan`.
    ///
    /// Weights are packed from the immutable [`ModelWeights`] table, so the
    /// original FP32 tensor stays valid for any other consumer (the embedding
    /// table is never touched). The pointer key is read from the same `Storage`
    /// handle the session clones into its weight slots.
    pub fn from_plan(
        plan: &Plan,
        weights: &ModelWeights,
        policy: QuantPolicy,
    ) -> Result<QuantWeightSet, Fn3Error> {
        let mut entries: Vec<QuantEntry> = Vec::new();
        let mut by_pointer: HashMap<usize, usize> = HashMap::new();
        let mut keepalive: Vec<Arc<Tensor>> = Vec::new();
        let mut coverage = CoverageReport::default();
        let mut packed_by_weight: BTreeMap<usize, usize> = BTreeMap::new();

        for (position, instruction) in plan.instructions().iter().enumerate() {
            if instruction.op != OpId::MatMul {
                continue;
            }
            let Some(&rhs_index) = instruction.inputs.get(1) else {
                continue;
            };
            let slot =
                plan.slots()
                    .get(rhs_index as usize)
                    .ok_or_else(|| Fn3Error::InvalidShape {
                        context: "W8A16 registration",
                        detail: format!(
                            "instruction {position} reads slot {rhs_index}, the plan has {} slots",
                            plan.slots().len()
                        ),
                    })?;
            let shape = slot.shape();

            match slot.role() {
                SlotRole::Weight { id } => {
                    let index = id.index();
                    let tensor = weights.handle(index).ok_or_else(|| Fn3Error::OutOfBounds {
                        what: "W8A16 registration (weight index)",
                        index: index as u64,
                        len: weights.len() as u64,
                    })?;
                    // One weight used by several instructions (a tied tensor) is
                    // packed once and reused.
                    if let Some(&entry) = packed_by_weight.get(&index) {
                        let entry: &QuantEntry = &entries[entry];
                        coverage.push(ProjectionCoverage {
                            name: entry.spec.name.clone(),
                            kind: entry.spec.kind,
                            decision: QuantDecision::Quantize,
                            k: entry.spec.k,
                            n: entry.spec.n,
                            packed_bytes: entry.weight.report().packed_bytes(),
                            relative_frobenius: Some(entry.weight.report().relative_frobenius),
                        });
                        continue;
                    }

                    let name = weights.name(index).unwrap_or("weight").to_string();
                    let spec = ProjectionSpec::static_weight(name, shape, slot.dtype());
                    match policy.classify(&spec) {
                        QuantDecision::Quantize => {
                            let pointer = tensor.as_view()?.storage_bytes().as_ptr() as usize;
                            // A weight whose storage is already registered (two
                            // names for tied bytes) reuses that packed copy.
                            if let Some(&existing) = by_pointer.get(&pointer) {
                                let entry: &QuantEntry = &entries[existing];
                                if entry.weight.k() != spec.k as usize
                                    || entry.weight.n_rows() != spec.n as usize
                                {
                                    return Err(Fn3Error::InvalidShape {
                                        context: "W8A16 registration",
                                        detail: format!(
                                            "two weights share one storage but disagree on \
                                             shape: {} is {} x {}, {} is {} x {}",
                                            entry.spec.name,
                                            entry.weight.n_rows(),
                                            entry.weight.k(),
                                            spec.name,
                                            spec.n,
                                            spec.k
                                        ),
                                    });
                                }
                                coverage.push(ProjectionCoverage {
                                    name: spec.name.clone(),
                                    kind: spec.kind,
                                    decision: QuantDecision::Quantize,
                                    k: spec.k,
                                    n: spec.n,
                                    packed_bytes: entry.weight.report().packed_bytes(),
                                    relative_frobenius: Some(
                                        entry.weight.report().relative_frobenius,
                                    ),
                                });
                                packed_by_weight.insert(index, existing);
                                continue;
                            }

                            let weight = Arc::new(PackedWeight::from_tensor(&spec.name, tensor)?);
                            let entry_index = entries.len();
                            coverage.push(ProjectionCoverage {
                                name: spec.name.clone(),
                                kind: spec.kind,
                                decision: QuantDecision::Quantize,
                                k: spec.k,
                                n: spec.n,
                                packed_bytes: weight.report().packed_bytes(),
                                relative_frobenius: Some(weight.report().relative_frobenius),
                            });
                            entries.push(QuantEntry { weight, spec });
                            by_pointer.insert(pointer, entry_index);
                            packed_by_weight.insert(index, entry_index);
                            keepalive.push(Arc::clone(tensor));
                        }
                        QuantDecision::Fallback(reason) => coverage.push(ProjectionCoverage {
                            name: spec.name.clone(),
                            kind: spec.kind,
                            decision: QuantDecision::Fallback(reason),
                            k: spec.k,
                            n: spec.n,
                            packed_bytes: 0,
                            relative_frobenius: None,
                        }),
                    }
                }
                role => {
                    let what = match role {
                        SlotRole::GraphInput => "a graph input",
                        SlotRole::Constant { .. } => "a folded constant",
                        SlotRole::InstructionOutput { .. } => {
                            "a value computed in the step (a transpose, a KV slice or an activation)"
                        }
                        SlotRole::Weight { .. } => "a weight",
                    };
                    let spec = ProjectionSpec::computed(slot.name(), what, shape);
                    let reason = match policy.classify(&spec) {
                        QuantDecision::Fallback(reason) => reason,
                        QuantDecision::Quantize => unreachable!("computed RHS is a fallback"),
                    };
                    coverage.push(ProjectionCoverage {
                        name: spec.name.clone(),
                        kind: spec.kind,
                        decision: QuantDecision::Fallback(reason),
                        k: spec.k,
                        n: spec.n,
                        packed_bytes: 0,
                        relative_frobenius: None,
                    });
                }
            }
        }

        Ok(QuantWeightSet {
            entries,
            by_pointer,
            _keepalive: keepalive,
            coverage,
        })
    }

    /// Register one packed weight for a source tensor directly.
    ///
    /// [`QuantWeightSet::from_plan`] is the planner-driven path; this entry
    /// point exists for a caller that already has a packed weight and the
    /// tensor whose storage a session will present (tests, and a future
    /// prepared-artifact loader). The source must be the 2-D FP32 `[K, N]`
    /// right-hand side the weight was packed from.
    pub fn register(
        &mut self,
        spec: ProjectionSpec,
        weight: PackedWeight,
        source: &Tensor,
    ) -> Result<(), Fn3Error> {
        if spec.k != weight.k() as u64 || spec.n != weight.n_rows() as u64 {
            return Err(Fn3Error::InvalidShape {
                context: "W8A16 registration",
                detail: format!(
                    "projection {} describes {} x {}, the packed weight is {} x {}",
                    spec.name,
                    spec.n,
                    spec.k,
                    weight.n_rows(),
                    weight.k()
                ),
            });
        }
        if source.dtype() != DType::F32
            || source.shape().rank() != 2
            || source.shape().dims() != [weight.k() as u64, weight.n_rows() as u64].as_slice()
        {
            return Err(Fn3Error::InvalidShape {
                context: "W8A16 registration",
                detail: format!(
                    "the source tensor for {} is {} {}, the packed weight needs f32 [{}, {}]",
                    spec.name,
                    source.dtype(),
                    source.shape(),
                    weight.k(),
                    weight.n_rows()
                ),
            });
        }
        let pointer = source.as_view()?.storage_bytes().as_ptr() as usize;
        if self.by_pointer.contains_key(&pointer) {
            return Err(Fn3Error::InvalidShape {
                context: "W8A16 registration",
                detail: format!(
                    "the storage of {} is already registered; one packed copy serves one \
                     right-hand side",
                    spec.name
                ),
            });
        }
        let index = self.entries.len();
        self.coverage.push(ProjectionCoverage {
            name: spec.name.clone(),
            kind: spec.kind,
            decision: QuantDecision::Quantize,
            k: spec.k,
            n: spec.n,
            packed_bytes: weight.report().packed_bytes(),
            relative_frobenius: Some(weight.report().relative_frobenius),
        });
        self.by_pointer.insert(pointer, index);
        self._keepalive.push(Arc::new(source.clone()));
        self.entries.push(QuantEntry {
            weight: Arc::new(weight),
            spec,
        });
        Ok(())
    }

    /// Number of registered weights.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when nothing was registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The coverage report this set was built from.
    pub fn coverage(&self) -> &CoverageReport {
        &self.coverage
    }

    /// Registered entries.
    pub fn entries(&self) -> &[QuantEntry] {
        &self.entries
    }

    /// A registered weight by model name.
    pub fn weight(&self, name: &str) -> Option<&PackedWeight> {
        self.entries
            .iter()
            .find(|entry| entry.spec.name == name)
            .map(|entry| entry.weight.as_ref())
    }

    /// The entry serving a concrete right-hand-side view.
    ///
    /// The key is the address of the view's underlying host allocation, which
    /// is stable because the session clones `Arc` storage rather than the
    /// bytes. The offset and shape are re-checked so a view into the same
    /// allocation cannot be mistaken for the registered weight.
    pub fn lookup(&self, rhs: &TensorView<'_>) -> Option<(usize, &QuantEntry)> {
        if rhs.is_device_resident() || rhs.meta().offset() != 0 {
            return None;
        }
        let pointer = rhs.storage_bytes().as_ptr() as usize;
        let &index = self.by_pointer.get(&pointer)?;
        let entry = self.entries.get(index)?;
        if rhs.meta().dtype() != DType::F32
            || rhs.shape().rank() != 2
            || rhs.shape().dims() != entry.weight.source_shape()
        {
            return None;
        }
        Some((index, entry))
    }

    /// Resident bytes of every packed weight.
    pub fn packed_bytes(&self) -> u64 {
        self.entries
            .iter()
            .map(|entry| entry.weight.report().packed_bytes())
            .sum()
    }

    /// FP32 bytes of the weights the set replaces.
    pub fn fp32_bytes(&self) -> u64 {
        self.entries
            .iter()
            .map(|entry| entry.weight.report().fp32_bytes)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(k: u64, n: u64) -> ProjectionSpec {
        ProjectionSpec::static_weight(
            "transformer.h.0.mlp.c_fc.weight",
            &Shape::from_dims([k, n]),
            DType::F32,
        )
    }

    #[test]
    fn the_policy_names_every_fallback() {
        let policy = QuantPolicy::default();
        assert_eq!(policy.classify(&spec(768, 3072)), QuantDecision::Quantize);
        assert!(matches!(
            policy.classify(&spec(32, 3072)),
            QuantDecision::Fallback(FallbackReason::KBelowMinimum { k: 32, .. })
        ));
        assert!(matches!(
            policy.classify(&spec(64, 32)),
            QuantDecision::Fallback(FallbackReason::NumelBelowMinimum { numel: 2048, .. })
        ));
        let two_d = ProjectionSpec::static_weight("odd", &Shape::from_dims([2, 2, 2]), DType::F32);
        assert!(matches!(
            policy.classify(&two_d),
            QuantDecision::Fallback(FallbackReason::NonTwoDimensionalRhs { rank: 3 })
        ));
        let not_f32 =
            ProjectionSpec::static_weight("odd", &Shape::from_dims([768, 768]), DType::I32);
        assert!(matches!(
            policy.classify(&not_f32),
            QuantDecision::Fallback(FallbackReason::NonF32Weight { .. })
        ));
        let shared = spec(768, 768).with_shared_weight(true);
        assert_eq!(policy.classify(&shared), QuantDecision::Quantize);
        assert!(matches!(
            policy.classify(&shared.with_packed_copy(false)),
            QuantDecision::Fallback(FallbackReason::SharedWeightWithoutPackedCopy)
        ));
        let dynamic = ProjectionSpec::computed(
            "transformer.wte.weight_transposed",
            "a value computed in the step",
            &Shape::from_dims([768, 50257]),
        );
        assert!(matches!(
            policy.classify(&dynamic),
            QuantDecision::Fallback(FallbackReason::DynamicRhs { .. })
        ));
    }

    #[test]
    fn registration_is_full_only_for_the_kernels_subset() {
        let policy = QuantPolicy::default();
        let qkv = spec(768, 2304);
        let contiguous = ActivationSpec {
            dtype: DType::F32,
            rank: 2,
            k: 768,
            contiguous: true,
        };
        assert_eq!(
            registration(&qkv, contiguous, &policy).support(),
            Support::Full
        );
        let strided = ActivationSpec {
            contiguous: false,
            ..contiguous
        };
        let partial = registration(&qkv, strided, &policy);
        assert!(matches!(
            partial.support(),
            Support::Partial { ref reason } if reason.contains("contiguous")
        ));
        assert_eq!(partial.code(), "activation-layout");
        let one_d = ActivationSpec {
            rank: 1,
            ..contiguous
        };
        assert!(matches!(
            registration(&qkv, one_d, &policy).support(),
            Support::Unsupported { .. }
        ));
        let dynamic = ProjectionSpec::computed(
            "transformer.wte.weight_transposed",
            "a value computed in the step",
            &Shape::from_dims([768, 50257]),
        );
        assert!(matches!(
            registration(&dynamic, contiguous, &policy).support(),
            Support::Unsupported { .. }
        ));
    }

    #[test]
    fn projection_kinds_come_from_the_model_names() {
        assert_eq!(
            ProjectionKind::from_name("transformer.h.3.attn.c_attn.weight"),
            ProjectionKind::Qkv
        );
        assert_eq!(
            ProjectionKind::from_name("transformer.h.3.attn.c_proj.weight"),
            ProjectionKind::AttnOut
        );
        assert_eq!(
            ProjectionKind::from_name("transformer.h.3.mlp.c_fc.weight"),
            ProjectionKind::MlpIn
        );
        assert_eq!(
            ProjectionKind::from_name("transformer.h.3.mlp.c_proj.weight"),
            ProjectionKind::MlpOut
        );
        assert_eq!(
            ProjectionKind::from_name("transformer.wte.weight_transposed"),
            ProjectionKind::LmHead
        );
        assert_eq!(ProjectionKind::from_name("misc"), ProjectionKind::Other);
    }
}
