//! Bounded shape specialisation: turn an imported graph into a [`Plan`].
//!
//! # What specialisation has to guarantee
//!
//! [`super::plan::Plan`] is only useful if a *kernel* can run it without ever
//! asking a shape question. That means this module must make three promises
//! true at build time:
//!
//! 1. **Every slot has a concrete shape.** No symbolic axis and no `Unknown`
//!    reason survives into the plan.
//! 2. **No integer arithmetic executes.** The alpha CPU arithmetic kernels are
//!    documented `f32`-only (`src/fn3/backend/cpu/kernels/elementwise.rs`) and
//!    only [`OpId::Cast`] may convert, so `i64` `Add`/`Sub` — which the pinned
//!    exporter really does use for `past`, `total` and the causal index range —
//!    must be **constant-folded** here, not executed later.
//! 3. **Reject, never approximate.** A shape outside the envelope, an extent
//!    the graph cannot resolve, or an op the backend does not admit for the
//!    plan's concrete operands is a typed error naming the value or the op.
//!    There is no "try it and see".
//!
//! ```text
//! ModelImport ──► resolve (substitute LiveDims, fold integer shape algebra)
//!             ──► build   (liveness, slot table, constants, instructions)
//!             ──► preflight (Backend::admits, capabilities, masks)
//!             ──► Plan
//! ```
//!
//! # Why folding is the only honest answer
//!
//! The pinned graph computes `seq + past` and `total - seq` at ONNX nodes 17,
//! 21, 27 and 44, all in `i64`. Casting them to `f32` would be a silent
//! numerical change (`2^24 + 1` is not representable in `f32`), the CPU
//! `Add`/`Sub` kernels reject `i64` by contract, and an execution-time "shape
//! op" would be a shape question at exactly the moment the plan promised there
//! would be none. So this module evaluates every node whose operands are
//! *already known integers* — `Shape`, `Gather`, `Slice`, `Concat`,
//! `Unsqueeze`, `Squeeze`, `Reshape`, `Flatten`, `Expand`, `Range`,
//! `ConstantOfShape` (integer fill), `Cast` (integer domain), `Equal`, `Less`,
//! `Greater`, `Where`, `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Pow` — and stores the
//! result as a [`super::plan::Constant`]. A node that folds has no
//! [`super::Instruction`] at all, and [`super::plan::Plan::folded`] records it
//! as evidence.
//!
//! A second fold runs before that: a `Reshape` whose resolved shape is exactly
//! its input's shape is an identity (**not** an arithmetic fold), so it emits
//! no instruction and its value resolves to the input's slot. This is what
//! leaves a `Gather` that indexes with `Reshape(input_ids)` reading the host
//! graph-input slot it validates on the host, instead of a device-resident
//! instruction output (FN3-26 / #72). The fold is restricted to a `Reshape`
//! with the same shape, the same dtype and a non-graph-output value, so it can
//! only remove a copy that was already a byte-for-byte alias.
//!
//! If an integer *arithmetic* node cannot fold (because one operand comes from
//! a runtime tensor), the plan is refused with
//! [`Fn3Error::UnsupportedOp`] naming the node and both owning issues. That is a
//! blocker, never a runtime surprise.
//!
//! Floating-point arithmetic is deliberately **not** folded: it stays on the
//! backend, so there is exactly one numerical path for `f32` and the planner
//! never becomes a second, subtly different reference implementation.
//!
//! # Attribute mapping (`OpParams`)
//!
//! [`super::super::backend::OpParams`] is the alpha attribute carrier: three
//! generic fields (`axes`, `keep_dims`, `epsilon`) shared by every op. This
//! module defines the **typed** view of that encoding, [`PlanAttr`], and every
//! plan instruction's `OpParams` is produced by [`PlanAttr::encode`] — never by
//! copying the importer's fields through unread. The exact encoding:
//!
//! | Op | `OpParams` fields | Meaning |
//! |---|---|---|
//! | `Gather`, `Concat`, `Split` | `axes = [axis]` | the one axis |
//! | `Softmax` | `axes = [axis]` | softmax axis, may be negative |
//! | `ArgMax` | `axes = [axis]`, `keep_dims` | axis and `keepdims` |
//! | `Trilu` | `axes = [upper]` | `0` = lower, non-zero = upper |
//! | `Transpose` | `axes = perm` | the full permutation |
//! | `ReduceMean`, `ReduceSum` | `axes`, `keep_dims` | normalised non-negative axes |
//! | `LayerNorm` | `axes`, `epsilon` | normalised axes, epsilon |
//! | `MaskedAttention` | `keep_dims` | `true` = causal mask (see below) |
//! | every other op | all neutral | a non-neutral field is rejected |
//!
//! ## The `OpParams` gap this module works around
//!
//! There is no boolean `causal` field and no packed attribute word, so the fused
//! attention's causal flag rides in `keep_dims`
//! (`src/fn3/backend/vulkan/kernels/attention.rs`, FN3-15 / #61). [`PlanAttr`]
//! is the single place that knows it: `Attention { causal } <=> keep_dims`.
//! When #48 extends `OpParams`, only [`PlanAttr::decode`] and
//! [`PlanAttr::encode`] change. The exact requested shape is recorded in
//! `docs/reports/fn3-10-planner.md`:
//!
//! ```ignore
//! // requested addition to `OpParams` (FN3-02 / #48), one variant per family:
//! pub enum OpAttributes {
//!     MatMul { transpose_a: bool, transpose_b: bool },
//!     Attention { causal: bool, scale: Option<f32>, window: Option<u32>,
//!                 layout: AttentionLayout },
//!     Softmax { axis: i32 },
//!     Reduce { axes: Vec<u32>, keep_dims: bool },
//!     Transpose { perm: Vec<u32> },
//!     None,
//! }
//! ```
//!
//! `Flatten` has a second, smaller gap: the importer accepts `Flatten.axis` but
//! `OpParams` has no field to carry it, so this planner can only plan the ONNX
//! default (`axis = 1`). A non-default axis is caught by the
//! declared-vs-resolved shape check and refused; see the report for the
//! requested `Flatten { axis }` variant.

use crate::inference::backend::{Backend, OpParams};
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::import::{
    DecoderGeometry, Dim, GraphIr, GraphNode, ModelImport, ValueId, ValueOrigin, ValueShape,
};
use crate::inference::model::weights::{WeightSet, WeightTensor};
use crate::inference::ops::OpId;
use crate::inference::shape::{DimBound, LiveDims, Shape, ShapeEnvelope, Symbol};

use super::plan::{
    BoundValue, Constant, ConstantData, FoldKind, FoldedNode, MaskConstant, MaskRole, Plan,
    PlanBound, PlanParts, SlotPlan, SlotRole,
};
use super::{ExecutionPlan, Instruction};

/// Backend name used when the *planner* is the thing that cannot proceed, so a
/// diagnostic never pretends a device rejected the op.
const PLANNER: &str = "planner";

// ---------------------------------------------------------------------------
// Typed attributes
// ---------------------------------------------------------------------------

/// The typed meaning of a plan instruction's [`OpParams`].
///
/// One variant per encoding the alpha carrier can express; see the module docs
/// for the table. [`PlanAttr::encode`] is the only code in the planner that
/// writes [`OpParams`].
#[derive(Debug, Clone, PartialEq)]
pub enum PlanAttr {
    /// The op takes no attributes; every `OpParams` field must be neutral.
    None,
    /// Exactly one axis (`Gather`, `Concat`, `Split`, `Softmax`).
    Axis(i64),
    /// One axis plus a keep-dims flag (`ArgMax`).
    AxisKeepDims {
        /// Axis, possibly negative.
        axis: i64,
        /// Keep the reduced axis as extent 1.
        keep_dims: bool,
    },
    /// A full axis permutation (`Transpose`).
    Perm(Vec<i64>),
    /// A reduction axis set plus keep-dims (`ReduceMean`, `ReduceSum`).
    Reduce {
        /// Normalised non-negative axes.
        axes: Vec<i64>,
        /// Keep reduced axes as extent 1.
        keep_dims: bool,
    },
    /// Trailing-axis normalisation (`LayerNorm`).
    Epsilon {
        /// Normalised non-negative axes.
        axes: Vec<i64>,
        /// Numerical floor.
        epsilon: f32,
    },
    /// Fused causal attention (`MaskedAttention`).
    Attention {
        /// `true` applies the implicit causal mask.
        causal: bool,
    },
}

impl PlanAttr {
    /// Decode `params` for `op`, rejecting any field the op does not define.
    ///
    /// This is the authority: a plan instruction's attributes are
    /// `PlanAttr::decode(op, node.params)?.encode()`, so the encoding and the
    /// typed view cannot drift.
    pub fn decode(op: OpId, params: &OpParams) -> Result<PlanAttr, Fn3Error> {
        let neutral = params.axes.is_empty() && !params.keep_dims && params.epsilon == 0.0;
        let attribute_error = |detail: String| Fn3Error::InvalidShape {
            context: "PlanAttr::decode",
            detail: format!("op {op}: {detail}"),
        };
        match op {
            // One axis, nothing else.
            OpId::Gather | OpId::Concat | OpId::Split | OpId::Softmax => {
                if params.axes.len() != 1 {
                    return Err(attribute_error(format!(
                        "expected exactly one axis, found axes={:?}",
                        params.axes
                    )));
                }
                if params.keep_dims || params.epsilon != 0.0 {
                    return Err(attribute_error(format!(
                        "accepts a single axis only, but the plan supplies keep_dims={}, \
                         epsilon={}",
                        params.keep_dims, params.epsilon
                    )));
                }
                Ok(PlanAttr::Axis(params.axes[0]))
            }
            OpId::ArgMax => {
                if params.axes.len() != 1 {
                    return Err(attribute_error(format!(
                        "expected exactly one axis, found axes={:?}",
                        params.axes
                    )));
                }
                if params.epsilon != 0.0 {
                    return Err(attribute_error(format!(
                        "accepts no epsilon, found {}",
                        params.epsilon
                    )));
                }
                Ok(PlanAttr::AxisKeepDims {
                    axis: params.axes[0],
                    keep_dims: params.keep_dims,
                })
            }
            OpId::Trilu => {
                if params.axes.len() != 1 {
                    return Err(attribute_error(format!(
                        "expected the `upper` flag in axes[0], found axes={:?}",
                        params.axes
                    )));
                }
                if params.keep_dims || params.epsilon != 0.0 {
                    return Err(attribute_error(
                        "accepts the `upper` flag in axes[0] and nothing else".to_string(),
                    ));
                }
                Ok(PlanAttr::Axis(params.axes[0]))
            }
            OpId::Transpose => {
                if params.keep_dims || params.epsilon != 0.0 {
                    return Err(attribute_error(
                        "accepts a permutation only, but the plan supplies keep_dims/epsilon"
                            .to_string(),
                    ));
                }
                Ok(PlanAttr::Perm(params.axes.clone()))
            }
            OpId::ReduceMean | OpId::ReduceSum => {
                if params.epsilon != 0.0 {
                    return Err(attribute_error(format!(
                        "accepts no epsilon, found {}",
                        params.epsilon
                    )));
                }
                Ok(PlanAttr::Reduce {
                    axes: params.axes.clone(),
                    keep_dims: params.keep_dims,
                })
            }
            OpId::LayerNorm => {
                if params.epsilon == 0.0 {
                    return Err(attribute_error(
                        "LayerNorm needs an epsilon; the plan supplies 0.0".to_string(),
                    ));
                }
                Ok(PlanAttr::Epsilon {
                    axes: params.axes.clone(),
                    epsilon: params.epsilon,
                })
            }
            OpId::MaskedAttention => {
                if !params.axes.is_empty() || params.epsilon != 0.0 {
                    return Err(attribute_error(format!(
                        "the fused attention encoding uses keep_dims for the causal flag and \
                         nothing else, but the plan supplies axes={:?}, epsilon={}",
                        params.axes, params.epsilon
                    )));
                }
                Ok(PlanAttr::Attention {
                    causal: params.keep_dims,
                })
            }
            OpId::Conv
            | OpId::MaxPool
            | OpId::AveragePool
            | OpId::GlobalAveragePool
            | OpId::BatchNorm
            | OpId::Resize
            | OpId::Pad
            | OpId::Clip
            | OpId::Sigmoid
            | OpId::Maximum
            | OpId::Minimum
            | OpId::TopK => Err(attribute_error(format!(
                "op {op} has no execution-plan attribute encoding in this revision; the typed IR \
                 reads its OpParams, and the v3 CNN executor owns the plan encoding (phase 2, \
                 #102/#104)"
            ))),
            _ => {
                if neutral {
                    Ok(PlanAttr::None)
                } else {
                    Err(attribute_error(format!(
                        "takes no attributes, but the plan supplies axes={:?}, keep_dims={}, \
                         epsilon={}",
                        params.axes, params.keep_dims, params.epsilon
                    )))
                }
            }
        }
    }

    /// Decode an imported node's attributes.
    pub fn of(node: &GraphNode) -> Result<PlanAttr, Fn3Error> {
        PlanAttr::decode(node.op, &node.params).map_err(|error| match error {
            Fn3Error::InvalidShape { context, detail } => Fn3Error::InvalidShape {
                context,
                detail: format!(
                    "node {} `{}` ({}) failed attribute decoding: {detail}",
                    node.onnx_index, node.name, node.op
                ),
            },
            other => other,
        })
    }

    /// The `OpParams` this typed attribute encodes to.
    pub fn encode(&self) -> OpParams {
        match self {
            PlanAttr::None => OpParams::new(),
            PlanAttr::Axis(axis) => OpParams::new().with_axes([*axis]),
            PlanAttr::AxisKeepDims { axis, keep_dims } => OpParams::new()
                .with_axes([*axis])
                .with_keep_dims(*keep_dims),
            PlanAttr::Perm(perm) => OpParams::new().with_axes(perm.clone()),
            PlanAttr::Reduce { axes, keep_dims } => OpParams::new()
                .with_axes(axes.clone())
                .with_keep_dims(*keep_dims),
            PlanAttr::Epsilon { axes, epsilon } => OpParams::new()
                .with_axes(axes.clone())
                .with_epsilon(*epsilon),
            // The #61 workaround, isolated to exactly these two lines.
            PlanAttr::Attention { causal } => OpParams::new().with_keep_dims(*causal),
        }
    }
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

/// Builds [`Plan`]s for one imported model against one backend.
///
/// ```ignore
/// let imported = import_onnx_file(path)?;
/// let backend = CpuBackend::new();
/// let planner = Planner::new(&imported, &backend);
/// let envelope = planner.envelope(1, 13, 17, 18)?;
/// let bound = planner.bound(&envelope, 1)?;
///
/// let prefill = planner.plan(&envelope, LiveDims::new(1, 4, 0, 2, 2, 64)?)?;
/// let decode = planner.plan(&envelope, LiveDims::new(1, 1, 17, 2, 2, 64)?)?;
/// bound.admits(&prefill)?;
/// bound.admits(&decode)?;
/// ```
///
/// A `Planner` holds no mutable state: it borrows the import and the backend and
/// returns a fresh plan on every call. That is what makes repeated calls reuse
/// the *plan* (callers cache by [`PlanId`]) rather than re-derive it.
///
/// [`PlanId`]: super::plan::PlanId
pub struct Planner<'a> {
    import: &'a ModelImport,
    backend: &'a dyn Backend,
    session_token_aliases: bool,
}

impl<'a> Planner<'a> {
    /// Bind a planner to an import and the backend that will run its plans.
    pub fn new(import: &'a ModelImport, backend: &'a dyn Backend) -> Planner<'a> {
        Planner {
            import,
            backend,
            session_token_aliases: false,
        }
    }

    /// Session-only specialization: `Session` owns the binding for `input_ids`
    /// and always supplies a contiguous, host-backed I64 tensor. Use only for
    /// session execution or internal preflight for a session-backed entry point,
    /// never for a plan that accepts caller-supplied slot tensors.
    pub(crate) fn for_session(import: &'a ModelImport, backend: &'a dyn Backend) -> Self {
        Self {
            session_token_aliases: true,
            ..Self::new(import, backend)
        }
    }

    /// The imported model.
    pub fn import(&self) -> &'a ModelImport {
        self.import
    }

    /// The backend plans are preflighted against.
    pub fn backend(&self) -> &'a dyn Backend {
        self.backend
    }

    /// The envelope this model can use for the given bounds.
    ///
    /// Head counts, per-head width and hidden size are fixed by the weights, so
    /// those three symbols are pinned to the detected geometry; `batch`, `seq`,
    /// `past` and `total` stay free. Requires `max_seq <= max_total` and
    /// `max_past < max_total` (a decode step needs at least one token).
    pub fn envelope(
        &self,
        max_batch: u64,
        max_seq: u64,
        max_past: u64,
        max_total: u64,
    ) -> Result<ShapeEnvelope, Fn3Error> {
        let geometry = self.import.geometry();
        if max_total == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "Planner::envelope",
                detail: "max_total must be at least 1: a step consumes at least one token"
                    .to_string(),
            });
        }
        if max_seq > max_total {
            return Err(Fn3Error::InvalidShape {
                context: "Planner::envelope",
                detail: format!(
                    "max_seq ({max_seq}) exceeds max_total ({max_total}); a prefill cannot be \
                     longer than the total sequence it produces"
                ),
            });
        }
        if max_past >= max_total {
            return Err(Fn3Error::InvalidShape {
                context: "Planner::envelope",
                detail: format!(
                    "max_past ({max_past}) must leave room for one decode token below \
                     max_total ({max_total})"
                ),
            });
        }
        Ok(ShapeEnvelope::new([
            DimBound::new(1, max_batch)?,
            DimBound::new(1, max_seq)?,
            DimBound::new(0, max_past)?,
            DimBound::new(1, max_total)?,
            DimBound::exact(u64::from(geometry.query_heads))?,
            DimBound::exact(u64::from(geometry.kv_heads))?,
            DimBound::exact(u64::from(geometry.head_dim))?,
        ]))
    }

    /// Specialise the model for one concrete shape environment.
    ///
    /// Public planners keep reshape copies because execution accepts externally
    /// bound tensors, which need not have contiguous layout. The plan is
    /// deterministic: the same import, envelope and dims
    /// always produce an equal [`Plan`] with the same
    /// [`identity`](Plan::identity).
    pub fn plan(&self, envelope: &ShapeEnvelope, dims: LiveDims) -> Result<Plan, Fn3Error> {
        envelope.validate()?;
        envelope.check(&dims)?;
        envelope.check_consistency(&dims)?;
        self.check_geometry(&dims)?;
        let survey = Survey::resolve(self.import, dims, self.session_token_aliases)?;
        survey.finish(self.backend, *envelope)
    }

    /// The bound every admissible plan of `envelope` must respect.
    ///
    /// Plans the worst feasible corner(s) of the envelope and takes the maximum
    /// of each figure. Admissible `(past, seq)` pairs satisfy
    /// `past + seq == total` inside the envelope's bounds; every resolved extent
    /// in a decoder graph is a product of non-negative powers of `batch`, `seq`,
    /// `past` and `total`, so the figures are monotone and the maximum lies on a
    /// maximal element of that set. [`PlanBound::admits`] then proves a concrete
    /// plan against it.
    pub fn bound(&self, envelope: &ShapeEnvelope, batch: u64) -> Result<PlanBound, Fn3Error> {
        envelope.validate()?;
        envelope.get(Symbol::Batch).check(Symbol::Batch, batch)?;
        let geometry = self.import.geometry();
        let corners = feasible_corners(envelope, batch, geometry)?;
        let mut plans = Vec::with_capacity(corners.len());
        for dims in &corners {
            plans.push(self.plan(envelope, *dims)?);
        }
        PlanBound::from_corners(*envelope, batch, corners, &plans)
    }

    /// The heads and head width are baked into the weights; a plan cannot
    /// disagree with them.
    fn check_geometry(&self, dims: &LiveDims) -> Result<(), Fn3Error> {
        let geometry = self.import.geometry();
        let expected = [
            ("n_heads", u64::from(geometry.query_heads), dims.n_heads),
            ("n_kv_heads", u64::from(geometry.kv_heads), dims.n_kv_heads),
            ("head_dim", u64::from(geometry.head_dim), dims.head_dim),
        ];
        for (symbol, want, found) in expected {
            if want != found {
                return Err(Fn3Error::InvalidShape {
                    context: "Planner::plan",
                    detail: format!(
                        "{symbol} is fixed by the model weights to {want}, but the request says \
                         {found}; the plan cannot change the head geometry"
                    ),
                });
            }
        }
        Ok(())
    }
}

/// `(past, seq)` corners of the envelope that dominate every admissible pair.
fn feasible_corners(
    envelope: &ShapeEnvelope,
    batch: u64,
    geometry: &DecoderGeometry,
) -> Result<Vec<LiveDims>, Fn3Error> {
    let seq = envelope.get(Symbol::Seq);
    let past = envelope.get(Symbol::Past);
    let total = envelope.get(Symbol::Total);
    let mut pairs: Vec<(u64, u64)> = Vec::new();
    let mut push = |past_value: u64, seq_value: u64| {
        if past_value >= past.min()
            && past_value <= past.max()
            && seq_value >= seq.min()
            && seq_value <= seq.max()
        {
            if let Some(sum) = past_value.checked_add(seq_value) {
                if total.contains(sum) && !pairs.contains(&(past_value, seq_value)) {
                    pairs.push((past_value, seq_value));
                }
            }
        }
    };
    if past.max().saturating_add(seq.max()) <= total.max() {
        push(past.max(), seq.max());
    } else {
        // The frontier is `past + seq == total.max()`; its maximal elements are
        // the two ends of the segment.
        push(past.max(), total.max().saturating_sub(past.max()));
        push(total.max().saturating_sub(seq.max()), seq.max());
    }
    if pairs.is_empty() {
        return Err(Fn3Error::InvalidShape {
            context: "Planner::bound",
            detail: format!(
                "the envelope admits no (past, seq) pair: seq {seq}, past {past}, total {total}"
            ),
        });
    }
    pairs
        .into_iter()
        .map(|(past_value, seq_value)| {
            LiveDims::new(
                batch,
                seq_value,
                past_value,
                u64::from(geometry.query_heads),
                u64::from(geometry.kv_heads),
                u64::from(geometry.head_dim),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Resolution and folding
// ---------------------------------------------------------------------------

/// The op set the planner folds when its operands are known integers.
///
/// This is exactly the shape/control algebra of a decoder graph. Floating-point
/// ops are absent on purpose: they execute on the backend.
const FOLDABLE: [OpId; 22] = [
    OpId::Shape,
    OpId::Gather,
    OpId::Slice,
    OpId::Concat,
    OpId::Unsqueeze,
    OpId::Squeeze,
    OpId::Reshape,
    OpId::Flatten,
    OpId::Expand,
    OpId::ConstantOfShape,
    OpId::Range,
    OpId::Cast,
    OpId::Equal,
    OpId::Less,
    OpId::Greater,
    OpId::Where,
    OpId::Add,
    OpId::Sub,
    OpId::Mul,
    OpId::Div,
    OpId::Neg,
    OpId::Pow,
];

/// Ops with a compute kernel that the alpha CPU tree only implements for `f32`.
const F32_COMPUTE: [OpId; 16] = [
    OpId::MatMul,
    OpId::Add,
    OpId::Sub,
    OpId::Mul,
    OpId::Div,
    OpId::Pow,
    OpId::Neg,
    OpId::Sqrt,
    OpId::Erf,
    OpId::Tanh,
    OpId::Gelu,
    OpId::Relu,
    OpId::Softmax,
    OpId::LayerNorm,
    OpId::ReduceMean,
    OpId::ReduceSum,
];

fn is_integer_domain(dtype: DType) -> bool {
    matches!(dtype, DType::I64 | DType::I32 | DType::Bool)
}

/// Shape resolution plus integer folding for one concrete [`LiveDims`].
struct Survey<'a> {
    graph: &'a GraphIr,
    weights: &'a WeightSet,
    dims: LiveDims,
    session_token_aliases: bool,
    shapes: Vec<Option<Shape>>,
    data: Vec<Option<ConstantData>>,
    /// For an identity-folded value, the value whose slot it resolves to.
    ///
    /// The source is always a root (never another alias) and its `GraphIr`
    /// position is lower, because the importer emits nodes in topological
    /// order.
    alias_of: Vec<Option<ValueId>>,
    attrs: Vec<Option<PlanAttr>>,
    surviving: Vec<u32>,
    folded: Vec<FoldedNode>,
}

impl<'a> Survey<'a> {
    fn resolve(
        import: &'a ModelImport,
        dims: LiveDims,
        session_token_aliases: bool,
    ) -> Result<Survey<'a>, Fn3Error> {
        let graph = import.graph();
        let weights = import.weights();
        let count = graph.value_count();
        let mut survey = Survey {
            graph,
            weights,
            dims,
            session_token_aliases,
            shapes: vec![None; count],
            data: vec![None; count],
            alias_of: vec![None; count],
            attrs: vec![None; count],
            surviving: Vec::new(),
            folded: Vec::new(),
        };
        for (index, value) in graph.values().iter().enumerate() {
            match value.origin {
                ValueOrigin::GraphInput => {
                    survey.shapes[index] = Some(substitute(&value.shape, dims, &value.name)?);
                }
                ValueOrigin::Weight(id) => {
                    let shape = substitute(&value.shape, dims, &value.name)?;
                    let weight = weights.get(id).ok_or_else(|| Fn3Error::OutOfBounds {
                        what: "Survey::resolve (weight id)",
                        index: id.index() as u64,
                        len: weights.len() as u64,
                    })?;
                    let elements = shape.numel()?;
                    if elements != weight.numel()? {
                        return Err(Fn3Error::InvalidShape {
                            context: "Survey::resolve",
                            detail: format!(
                                "weight `{}` carries {} elements but its value declares the shape \
                                 {shape} ({elements} elements)",
                                value.name,
                                weight.numel()?
                            ),
                        });
                    }
                    survey.shapes[index] = Some(shape);
                    survey.data[index] = decode_weight(weight);
                }
                ValueOrigin::Node(_) => {}
            }
        }
        for node_index in 0..graph.nodes().len() {
            survey.visit(node_index)?;
        }
        Ok(survey)
    }

    fn visit(&mut self, node_index: usize) -> Result<(), Fn3Error> {
        let graph = self.graph;
        let node = &graph.nodes()[node_index];
        let attr = PlanAttr::of(node)?;
        let shapes = self.infer(node, &attr)?;
        if shapes.len() != node.outputs.len() {
            return Err(Fn3Error::InvalidShape {
                context: "Survey::visit",
                detail: format!(
                    "node {} ({}) has {} outputs but the planner inferred {} shapes",
                    node.onnx_index,
                    node.op,
                    node.outputs.len(),
                    shapes.len()
                ),
            });
        }
        // `Shape` reads metadata, not data: its result is a constant even when
        // its operand is a runtime tensor. Every other foldable op is a pure
        // function of its operand *values*, so all of them must be known.
        let foldable = match node.op {
            OpId::Shape => node
                .outputs
                .iter()
                .all(|id| is_integer_domain(self.graph.value(*id).dtype)),
            _ => {
                FOLDABLE.contains(&node.op)
                    && node
                        .outputs
                        .iter()
                        .all(|id| is_integer_domain(self.graph.value(*id).dtype))
                    && node.inputs.iter().all(|id| self.data[id.index()].is_some())
            }
        };
        if foldable {
            // `Shape` needs its operand's metadata, not its bytes.
            let inputs: Vec<ConstantData> = if node.op == OpId::Shape {
                Vec::new()
            } else {
                node.inputs
                    .iter()
                    .map(|id| {
                        self.data[id.index()]
                            .clone()
                            .expect("foldability checked that every operand is known")
                    })
                    .collect()
            };
            let folded = self.fold(node, &attr, &shapes, &inputs)?;
            if folded.len() != shapes.len() {
                return Err(Fn3Error::InvalidShape {
                    context: "Survey::visit (fold)",
                    detail: format!(
                        "folding node {} ({}) produced {} value(s) for {} output(s)",
                        node.onnx_index,
                        node.op,
                        folded.len(),
                        shapes.len()
                    ),
                });
            }
            for ((id, shape), data) in node.outputs.iter().zip(&shapes).zip(folded) {
                let elements = shape.numel()?;
                if elements != data.len() as u64 {
                    return Err(Fn3Error::InvalidShape {
                        context: "Survey::visit (fold)",
                        detail: format!(
                            "node {} ({}) folded `{}` into {} {shape} = {elements} elements, but \
                             produced {}",
                            node.onnx_index,
                            node.op,
                            self.graph.value(*id).name,
                            data.dtype(),
                            data.len()
                        ),
                    });
                }
                self.shapes[id.index()] = Some(shape.clone());
                self.data[id.index()] = Some(data);
            }
            self.folded.push(FoldedNode {
                onnx_index: node.onnx_index,
                name: node.name.clone(),
                op: node.op,
                outputs: node.outputs.iter().map(|id| id.index() as u32).collect(),
                kind: FoldKind::Constant,
            });
            return Ok(());
        }
        // A `Reshape` that resolves to exactly its input's shape, dtype and
        // element count on the session's contiguous token input is an identity:
        // same row-major layout, no reorder, broadcast or conversion. It emits
        // no instruction; its value resolves to the input's slot. A graph
        // output keeps its own binding and is never folded.
        //
        // This is also the fix for FN3-26 (#72): the trained GPT-2 feeds the
        // `wte` `Gather` through `Reshape(input_ids, [1, 13])`. Folding the
        // reshape leaves `Gather` reading the host `input_ids` graph-input
        // slot, which is the operand its host index validation requires.
        if let Some(source) = self.identity_reshape_source(node, &shapes) {
            self.check_declared(node, node.outputs[0], &shapes[0])?;
            let root = self.alias_of[source.index()].unwrap_or(source);
            self.alias_of[node.outputs[0].index()] = Some(root);
            self.shapes[node.outputs[0].index()] = Some(shapes[0].clone());
            self.data[node.outputs[0].index()] = None;
            self.folded.push(FoldedNode {
                onnx_index: node.onnx_index,
                name: node.name.clone(),
                op: node.op,
                outputs: node.outputs.iter().map(|id| id.index() as u32).collect(),
                kind: FoldKind::Identity,
            });
            return Ok(());
        }
        self.guard_integer_arithmetic(node)?;
        for (id, shape) in node.outputs.iter().zip(&shapes) {
            self.shapes[id.index()] = Some(shape.clone());
            self.data[id.index()] = None;
            self.check_declared(node, *id, shape)?;
        }
        self.attrs[node_index] = Some(attr);
        self.surviving.push(node_index as u32);
        Ok(())
    }

    /// The input value of a `Reshape` that is provably an identity.
    ///
    /// The proof is deliberately narrow:
    ///
    /// * the op is `Reshape` (which never reorders, broadcasts or converts);
    /// * the planner's resolved output shape equals the resolved input shape,
    ///   so every extent and the element count agree;
    /// * the element type is unchanged;
    /// * the value is not a graph output, which must keep its own slot binding.
    ///
    /// Only the session's own `input_ids` binding (or an alias of it) is
    /// eligible. Public execution accepts strided external roots, and a
    /// reshape copy can normalize their layout before downstream consumers.
    fn identity_reshape_source(&self, node: &GraphNode, shapes: &[Shape]) -> Option<ValueId> {
        if !self.session_token_aliases || node.op != OpId::Reshape {
            return None;
        }
        let source = *node.inputs.first()?;
        let root = self
            .graph
            .value(self.alias_of[source.index()].unwrap_or(source));
        if root.origin != ValueOrigin::GraphInput
            || root.name != "input_ids"
            || root.dtype != DType::I64
        {
            return None;
        }
        let [output] = node.outputs.as_slice() else {
            return None;
        };
        if self.graph.outputs().contains(output) {
            return None;
        }
        // The importer emits nodes in topological order, so an input's value
        // position is lower. Refuse the fold rather than rely on it: the slot
        // table resolves aliases in value order.
        if source.index() >= output.index() {
            return None;
        }
        let input = self.shapes[source.index()].as_ref()?;
        let resolved = shapes.first()?;
        if input != resolved {
            return None;
        }
        if self.graph.value(source).dtype != self.graph.value(*output).dtype {
            return None;
        }
        Some(source)
    }

    /// An integer op that survives folding may not be a *compute* op: the CPU
    /// arithmetic kernels are `f32`-only, and only `Cast` may convert.
    fn guard_integer_arithmetic(&self, node: &GraphNode) -> Result<(), Fn3Error> {
        if !F32_COMPUTE.contains(&node.op) {
            return Ok(());
        }
        for id in node.inputs.iter().chain(node.outputs.iter()) {
            let value = self.graph.value(*id);
            if value.dtype != DType::F32 {
                return Err(Fn3Error::UnsupportedOp {
                    op: node.op,
                    backend: PLANNER,
                    reason: format!(
                        "node {} `{}` needs `{}` on `{}`, and the planner could not constant-fold \
                         it: one operand is computed at run time. The alpha CPU arithmetic kernels \
                         are f32-only (FN3-07 #53) and only `Cast` may convert, so this plan is \
                         blocked on holding the value in a static I64 constant. Owner: FN3-10 \
                         (#56) for the fold, FN3-07 (#53) for an integer kernel if a fold is \
                         impossible.",
                        node.onnx_index, node.name, node.op, value.dtype
                    ),
                });
            }
        }
        Ok(())
    }

    /// Compare the planner's resolved shape with the importer's inference when
    /// that inference is fully static. A disagreement means the planner's shape
    /// semantics differ from the importer's, which must never be papered over.
    fn check_declared(
        &self,
        node: &GraphNode,
        id: ValueId,
        resolved: &Shape,
    ) -> Result<(), Fn3Error> {
        let value = self.graph.value(id);
        let Ok(declared) = substitute(&value.shape, self.dims, &value.name) else {
            return Ok(());
        };
        if &declared == resolved {
            return Ok(());
        }
        Err(node_error(
            node,
            format!(
                "the planner resolves `{}` as {resolved} but the import inferred {declared}; \
                 the two must agree before a plan can be bounded",
                value.name
            ),
        ))
    }

    fn shape_of(&self, id: ValueId) -> Result<&Shape, Fn3Error> {
        self.shapes[id.index()]
            .as_ref()
            .ok_or_else(|| Fn3Error::InvalidShape {
                context: "Survey::shape_of",
                detail: format!(
                    "value {} `{}` was read before its shape was resolved",
                    id.index(),
                    self.graph.value(id).name
                ),
            })
    }

    /// A constant integer operand, or a rejection naming the value.
    fn ints_of(&self, node: &GraphNode, id: ValueId, role: &str) -> Result<Vec<i64>, Fn3Error> {
        match &self.data[id.index()] {
            Some(data) => Ok(data.as_i64()),
            None => Err(Fn3Error::UnsupportedOp {
                op: node.op,
                backend: PLANNER,
                reason: format!(
                    "node {} `{}` needs `{}` to be a constant integer tensor ({role}), but it is \
                     computed at run time; a plan cannot commit to concrete extents without it. \
                     Owner: FN3-10 (#56)",
                    node.onnx_index,
                    node.name,
                    self.graph.value(id).name
                ),
            }),
        }
    }

    // -- shape rules ------------------------------------------------------

    fn infer(&self, node: &GraphNode, attr: &PlanAttr) -> Result<Vec<Shape>, Fn3Error> {
        let op = node.op;
        let first = node.inputs.first().copied();
        let first_shape = || -> Result<&Shape, Fn3Error> {
            self.shape_of(first.expect("every mapped op has at least one input"))
        };
        match op {
            OpId::MatMul => {
                let a = self.shape_of(node.inputs[0])?;
                let b = self.shape_of(node.inputs[1])?;
                Ok(vec![matmul_shape(a, b, node)?])
            }
            OpId::Add
            | OpId::Sub
            | OpId::Mul
            | OpId::Div
            | OpId::Pow
            | OpId::Equal
            | OpId::Less
            | OpId::Greater => {
                let a = self.shape_of(node.inputs[0])?;
                let b = self.shape_of(node.inputs[1])?;
                Ok(vec![a
                    .broadcast_with(b, op)
                    .map_err(|error| with_node(error, node))?])
            }
            OpId::Neg
            | OpId::Sqrt
            | OpId::Erf
            | OpId::Tanh
            | OpId::Gelu
            | OpId::Relu
            | OpId::Sin
            | OpId::Cos
            | OpId::Cast
            | OpId::Softmax
            | OpId::LayerNorm
            | OpId::Trilu => Ok(vec![first_shape()?.clone()]),
            OpId::ReduceMean | OpId::ReduceSum => {
                let axes = match attr {
                    PlanAttr::Reduce { axes, .. } => axes.clone(),
                    other => {
                        return Err(node_error(
                            node,
                            format!("expected a reduce attribute, found {other:?}"),
                        ))
                    }
                };
                let keep_dims = matches!(
                    attr,
                    PlanAttr::Reduce {
                        keep_dims: true,
                        ..
                    }
                );
                Ok(vec![reduce_shape(
                    self.shape_of(node.inputs[0])?,
                    &axes,
                    keep_dims,
                    node,
                )?])
            }
            OpId::ArgMax => {
                let (axis, keep_dims) = match attr {
                    PlanAttr::AxisKeepDims { axis, keep_dims } => (*axis, *keep_dims),
                    other => {
                        return Err(node_error(
                            node,
                            format!("expected an axis/keep_dims attribute, found {other:?}"),
                        ))
                    }
                };
                Ok(vec![reduce_shape(
                    self.shape_of(node.inputs[0])?,
                    &[axis],
                    keep_dims,
                    node,
                )?])
            }
            OpId::Transpose => {
                let perm = match attr {
                    PlanAttr::Perm(perm) => perm.clone(),
                    other => {
                        return Err(node_error(
                            node,
                            format!("expected a permutation attribute, found {other:?}"),
                        ))
                    }
                };
                let shape = self.shape_of(node.inputs[0])?;
                let perm = if perm.is_empty() {
                    (0..shape.rank() as i64).rev().collect::<Vec<_>>()
                } else {
                    perm
                };
                let perm: Vec<usize> = perm
                    .iter()
                    .map(|axis| normalise_axis(*axis, shape.rank(), node, "Transpose perm"))
                    .collect::<Result<_, _>>()?;
                Ok(vec![shape
                    .permuted(&perm)
                    .map_err(|error| with_node(error, node))?])
            }
            OpId::Reshape => {
                let target = self.ints_of(node, node.inputs[1], "Reshape target")?;
                Ok(vec![reshape_shape(
                    self.shape_of(node.inputs[0])?,
                    &target,
                    node,
                )?])
            }
            OpId::Flatten => Ok(vec![flatten_shape(self.shape_of(node.inputs[0])?, node)?]),
            OpId::Squeeze => {
                let axes = self.axes_input(node, attr, 1)?;
                Ok(vec![squeeze_shape(
                    self.shape_of(node.inputs[0])?,
                    axes.as_deref(),
                    node,
                )?])
            }
            OpId::Unsqueeze => {
                let axes = self
                    .axes_input(node, attr, 1)?
                    .ok_or_else(|| node_error(node, "Unsqueeze needs axes".to_string()))?;
                Ok(vec![unsqueeze_shape(
                    self.shape_of(node.inputs[0])?,
                    &axes,
                    node,
                )?])
            }
            OpId::Concat => {
                let axis = match attr {
                    PlanAttr::Axis(axis) => *axis,
                    other => {
                        return Err(node_error(
                            node,
                            format!("expected a concat axis attribute, found {other:?}"),
                        ))
                    }
                };
                let shapes = node
                    .inputs
                    .iter()
                    .map(|id| self.shape_of(*id))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(vec![concat_shape(&shapes, axis, node)?])
            }
            OpId::Slice => {
                let mut values = Vec::new();
                for role in ["starts", "ends", "axes", "steps"] {
                    let index = 1 + values.len();
                    if let Some(id) = node.inputs.get(index) {
                        values.push(self.ints_of(node, *id, role)?);
                    } else {
                        values.push(Vec::new());
                    }
                }
                Ok(vec![slice_shape(
                    self.shape_of(node.inputs[0])?,
                    &values[0],
                    &values[1],
                    &values[2],
                    &values[3],
                    node,
                )?])
            }
            OpId::Split => {
                let axis = match attr {
                    PlanAttr::Axis(axis) => *axis,
                    other => {
                        return Err(node_error(
                            node,
                            format!("expected a split axis attribute, found {other:?}"),
                        ))
                    }
                };
                let sizes = match node.inputs.get(1) {
                    Some(id) => Some(self.ints_of(node, *id, "Split sizes")?),
                    None => None,
                };
                split_shapes(
                    self.shape_of(node.inputs[0])?,
                    axis,
                    sizes.as_deref(),
                    node.outputs.len(),
                    node,
                )
            }
            OpId::Shape => {
                if !node.params.axes.is_empty() {
                    return Err(node_error(
                        node,
                        "Shape start/end are not representable in OpParams; the importer rejects \
                         them, so a non-empty axes field here is a contract violation"
                            .to_string(),
                    ));
                }
                Ok(vec![Shape::from_dims([
                    self.shape_of(node.inputs[0])?.rank() as u64,
                ])])
            }
            OpId::ConstantOfShape => {
                let target = self.ints_of(node, node.inputs[0], "ConstantOfShape shape")?;
                let dims = target
                    .iter()
                    .map(|&dim| {
                        u64::try_from(dim).map_err(|_| {
                            node_error(
                                node,
                                format!("ConstantOfShape shape contains the negative extent {dim}"),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(vec![Shape::from_dims(dims)])
            }
            OpId::Expand => {
                let target = self.ints_of(node, node.inputs[1], "Expand shape")?;
                let dims = target
                    .iter()
                    .map(|&dim| {
                        u64::try_from(dim).map_err(|_| {
                            node_error(
                                node,
                                format!("Expand shape contains the negative extent {dim}"),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let target = Shape::from_dims(dims);
                let input = self.shape_of(node.inputs[0])?;
                let broadcast = input
                    .broadcast_with(&target, op)
                    .map_err(|error| with_node(error, node))?;
                if broadcast != target {
                    return Err(node_error(
                        node,
                        format!(
                            "Expand target {target} is not the broadcast of its input {input} \
                             (which is {broadcast})"
                        ),
                    ));
                }
                Ok(vec![target])
            }
            OpId::Range => {
                let start = self.scalar_i64(node, node.inputs[0], "Range start")?;
                let limit = self.scalar_i64(node, node.inputs[1], "Range limit")?;
                let delta = self.scalar_i64(node, node.inputs[2], "Range delta")?;
                let count = range_len(start, limit, delta, node)?;
                Ok(vec![Shape::from_dims([count])])
            }
            OpId::Gather => {
                let axis = match attr {
                    PlanAttr::Axis(axis) => *axis,
                    other => {
                        return Err(node_error(
                            node,
                            format!("expected a gather axis attribute, found {other:?}"),
                        ))
                    }
                };
                let data = self.shape_of(node.inputs[0])?;
                let indices = self.shape_of(node.inputs[1])?;
                Ok(vec![gather_shape(data, indices, axis, node)?])
            }
            OpId::Where => {
                let a = self.shape_of(node.inputs[0])?;
                let b = self.shape_of(node.inputs[1])?;
                let c = self.shape_of(node.inputs[2])?;
                let first = a
                    .broadcast_with(b, op)
                    .map_err(|error| with_node(error, node))?;
                Ok(vec![first
                    .broadcast_with(c, op)
                    .map_err(|error| with_node(error, node))?])
            }
            OpId::ScatterND => Ok(vec![self.shape_of(node.inputs[0])?.clone()]),
            OpId::MaskedAttention => {
                let q = self.shape_of(node.inputs[0])?;
                let k = self.shape_of(node.inputs[1])?;
                if q.rank() != 4 || k.rank() != 4 {
                    return Err(node_error(
                        node,
                        format!(
                            "the fused attention contract needs rank-4 q and k, found {} and {}",
                            q.rank(),
                            k.rank()
                        ),
                    ));
                }
                let keys = k.dim(2).expect("rank checked");
                let heads = q.dim(1).expect("rank checked");
                Ok(vec![
                    q.clone(),
                    Shape::from_dims([
                        q.dim(0).expect("rank checked"),
                        heads,
                        q.dim(2).expect("rank checked"),
                        keys,
                    ]),
                ])
            } // Every `OpId` has a rule above: this match is exhaustive on
            // purpose, so adding an op id without a shape rule is a compile
            // error rather than a runtime "unsupported" surprise.
            OpId::Conv
            | OpId::MaxPool
            | OpId::AveragePool
            | OpId::GlobalAveragePool
            | OpId::BatchNorm
            | OpId::Resize
            | OpId::Pad
            | OpId::Clip
            | OpId::Sigmoid
            | OpId::Maximum
            | OpId::Minimum
            | OpId::TopK => Err(Fn3Error::UnsupportedOp {
                op,
                backend: "planner",
                reason: format!(
                    "{} has no execution-plan shape rule in this revision; the typed IR shape \
                     pass resolves it, and the v3 CNN executor owns the plan rule (phase 2, \
                     #102/#104)",
                    node.name
                ),
            }),
        }
    }

    /// Axes for `Squeeze`/`Unsqueeze`: the input when the op has one, otherwise
    /// the attribute (opset < 13 form).
    fn axes_input(
        &self,
        node: &GraphNode,
        attr: &PlanAttr,
        position: usize,
    ) -> Result<Option<Vec<i64>>, Fn3Error> {
        if let Some(id) = node.inputs.get(position) {
            return Ok(Some(self.ints_of(node, *id, "axes")?));
        }
        match attr {
            PlanAttr::None => Ok(None),
            PlanAttr::Axis(axis) => Ok(Some(vec![*axis])),
            PlanAttr::Reduce { axes, .. } | PlanAttr::Epsilon { axes, .. } => {
                Ok(Some(axes.clone()))
            }
            other => Err(node_error(
                node,
                format!("expected axes, found the attribute {other:?}"),
            )),
        }
    }

    fn scalar_i64(&self, node: &GraphNode, id: ValueId, role: &str) -> Result<i64, Fn3Error> {
        let values = self.ints_of(node, id, role)?;
        if values.len() != 1 {
            return Err(node_error(
                node,
                format!(
                    "{role} must be a scalar, but `{}` has {} elements",
                    self.graph.value(id).name,
                    values.len()
                ),
            ));
        }
        Ok(values[0])
    }

    // -- folding ---------------------------------------------------------

    fn fold(
        &self,
        node: &GraphNode,
        attr: &PlanAttr,
        shapes: &[Shape],
        inputs: &[ConstantData],
    ) -> Result<Vec<ConstantData>, Fn3Error> {
        let dtype = self.graph.value(node.outputs[0]).dtype;
        let output = || shapes[0].clone();
        let folded = match node.op {
            OpId::Shape => ConstantData::I64(
                self.shape_of(node.inputs[0])?
                    .dims()
                    .iter()
                    .map(|&dim| dim as i64)
                    .collect(),
            ),
            OpId::Gather => {
                let axis = match attr {
                    PlanAttr::Axis(axis) => *axis,
                    other => {
                        return Err(node_error(
                            node,
                            format!("expected a gather axis, found {other:?}"),
                        ))
                    }
                };
                let data = inputs[0].as_i64();
                let indices = inputs[1].as_i64();
                let axis = normalise_axis(
                    axis,
                    self.shape_of(node.inputs[0])?.rank(),
                    node,
                    "Gather axis",
                )?;
                ConstantData::I64(gather_values(
                    &data,
                    self.shape_of(node.inputs[0])?,
                    &indices,
                    axis,
                    node,
                )?)
            }
            OpId::Slice => {
                let data = inputs[0].as_i64();
                let mut rest = Vec::new();
                for position in 1..5 {
                    rest.push(if position < inputs.len() {
                        inputs[position].as_i64()
                    } else {
                        Vec::new()
                    });
                }
                ConstantData::I64(slice_values(
                    &data,
                    self.shape_of(node.inputs[0])?,
                    &rest[0],
                    &rest[1],
                    &rest[2],
                    &rest[3],
                    node,
                )?)
            }
            OpId::Concat => {
                let axis = match attr {
                    PlanAttr::Axis(axis) => *axis,
                    other => {
                        return Err(node_error(
                            node,
                            format!("expected a concat axis, found {other:?}"),
                        ))
                    }
                };
                let sources: Vec<Vec<i64>> = inputs.iter().map(ConstantData::as_i64).collect();
                let shapes: Vec<&Shape> = node
                    .inputs
                    .iter()
                    .map(|id| self.shape_of(*id))
                    .collect::<Result<Vec<_>, _>>()?;
                ConstantData::I64(concat_values(&sources, &shapes, axis, node)?)
            }
            OpId::Unsqueeze | OpId::Squeeze | OpId::Reshape | OpId::Flatten => {
                // Element order is unchanged; only the resolved shape differs.
                constant_from_i64(dtype, inputs[0].as_i64(), &output(), node)?
            }
            OpId::Expand => {
                // Expand broadcasts: every output coordinate reads the input
                // coordinate with broadcast axes pinned to 0.
                let input = self.shape_of(node.inputs[0])?.clone();
                let data = inputs[0].as_i64();
                let out_shape = output();
                let mut values = Vec::with_capacity(out_shape.numel()? as usize);
                for linear in 0..out_shape.numel()? {
                    let coordinate = coordinates_of(&out_shape, linear);
                    values.push(
                        data[ravel(&input, &broadcast_coordinate(&coordinate, &input)) as usize],
                    );
                }
                constant_from_i64(dtype, values, &out_shape, node)?
            }
            OpId::ConstantOfShape => {
                let fill = inputs[1].as_i64();
                if fill.len() != 1 {
                    return Err(node_error(
                        node,
                        format!(
                            "ConstantOfShape value must have exactly one element, found {}",
                            fill.len()
                        ),
                    ));
                }
                let out_shape = output();
                constant_from_i64(
                    dtype,
                    vec![fill[0]; out_shape.numel()? as usize],
                    &out_shape,
                    node,
                )?
            }
            OpId::Range => {
                let start = inputs[0].as_i64()[0];
                let limit = inputs[1].as_i64()[0];
                let delta = inputs[2].as_i64()[0];
                let count = range_len(start, limit, delta, node)?;
                let mut values = Vec::with_capacity(count as usize);
                for index in 0..count {
                    values.push(
                        start
                            .checked_add((index as i64).checked_mul(delta).ok_or_else(|| {
                                node_error(node, "Range multiplication overflowed i64".to_string())
                            })?)
                            .ok_or_else(|| {
                                node_error(node, "Range addition overflowed i64".to_string())
                            })?,
                    );
                }
                constant_from_i64(dtype, values, &output(), node)?
            }
            OpId::Cast => {
                let values = inputs[0].as_i64();
                constant_from_i64(dtype, values, &output(), node)?
            }
            OpId::Equal | OpId::Less | OpId::Greater => {
                let a = inputs[0].as_i64();
                let b = inputs[1].as_i64();
                let left = self.shape_of(node.inputs[0])?.clone();
                let right = self.shape_of(node.inputs[1])?.clone();
                let mut out = Vec::new();
                broadcast_zip(&left, &right, &output(), node, &mut |i, j| {
                    let x = a[i];
                    let y = b[j];
                    out.push(match node.op {
                        OpId::Equal => u8::from(x == y),
                        OpId::Less => u8::from(x < y),
                        _ => u8::from(x > y),
                    });
                })?;
                ConstantData::Bool(out)
            }
            OpId::Where => {
                let cond = inputs[0].as_i64();
                let x = inputs[1].as_i64();
                let y = inputs[2].as_i64();
                let shapes: Vec<Shape> = node
                    .inputs
                    .iter()
                    .map(|id| self.shape_of(*id).cloned())
                    .collect::<Result<Vec<_>, _>>()?;
                let mut out = Vec::new();
                broadcast_zip3(
                    &shapes[0],
                    &shapes[1],
                    &shapes[2],
                    &output(),
                    node,
                    &mut |c, i, j| {
                        out.push(if cond[c] != 0 { x[i] } else { y[j] });
                    },
                )?;
                constant_from_i64(dtype, out, &output(), node)?
            }
            OpId::Add | OpId::Sub | OpId::Mul | OpId::Div | OpId::Pow | OpId::Neg => {
                let values = if node.op == OpId::Neg {
                    inputs[0]
                        .as_i64()
                        .iter()
                        .map(|&value| value.wrapping_neg())
                        .collect()
                } else {
                    let a = inputs[0].as_i64();
                    let b = inputs[1].as_i64();
                    let left = self.shape_of(node.inputs[0])?.clone();
                    let right = self.shape_of(node.inputs[1])?.clone();
                    // Integer division by zero and a negative exponent are not
                    // "some value": refuse the plan instead of inventing one.
                    if node.op == OpId::Div && b.contains(&0) {
                        return Err(node_error(
                            node,
                            "integer division by zero has no ONNX value; the planner does not \
                             invent one"
                                .to_string(),
                        ));
                    }
                    if node.op == OpId::Pow && b.iter().any(|&exponent| exponent < 0) {
                        return Err(node_error(
                            node,
                            "integer Pow with a negative exponent is not an integer; the planner \
                             refuses it"
                                .to_string(),
                        ));
                    }
                    let mut out = Vec::new();
                    broadcast_zip(&left, &right, &output(), node, &mut |i, j| {
                        let x = a[i];
                        let y = b[j];
                        let value = match node.op {
                            OpId::Add => x.wrapping_add(y),
                            OpId::Sub => x.wrapping_sub(y),
                            OpId::Mul => x.wrapping_mul(y),
                            OpId::Div => x.wrapping_div(y),
                            _ => u32::try_from(y)
                                .map(|exponent| x.wrapping_pow(exponent))
                                .unwrap_or(i64::MAX),
                        };
                        out.push(value);
                    })?;
                    out
                };
                constant_from_i64(dtype, values, &output(), node)?
            }
            other => {
                return Err(Fn3Error::UnsupportedOp {
                    op: other,
                    backend: PLANNER,
                    reason:
                        "the planner listed this op as foldable but has no folding rule for it; \
                             that is a planner bug, not a model property. Owner: FN3-10 (#56)"
                            .to_string(),
                })
            }
        };
        Ok(vec![folded])
    }

    // -- plan construction -----------------------------------------------

    fn finish(self, backend: &dyn Backend, envelope: ShapeEnvelope) -> Result<Plan, Fn3Error> {
        let graph = self.graph;
        let mut needed = vec![false; graph.value_count()];
        for output in graph.outputs() {
            needed[output.index()] = true;
        }
        // Instruction position that writes each value, so a slot role can name
        // the instruction that produces it.
        let mut instruction_of_value: Vec<Option<u32>> = vec![None; graph.value_count()];
        for (position, &node_index) in self.surviving.iter().enumerate() {
            let node = &graph.nodes()[node_index as usize];
            for id in node.inputs.iter().chain(node.outputs.iter()) {
                needed[id.index()] = true;
            }
            for id in &node.outputs {
                instruction_of_value[id.index()] = Some(position as u32);
            }
        }
        // Folded values are not written by an instruction; their slot is a
        // constant (or, if dead, has no slot at all). An identity-folded value
        // has no slot of its own either: it resolves to the slot of the value
        // it aliases, so that source must stay live.
        for index in 0..graph.value_count() {
            if needed[index] {
                if let Some(source) = self.alias_of[index] {
                    needed[source.index()] = true;
                }
            }
        }

        let mut slots: Vec<SlotPlan> = Vec::new();
        let mut constants: Vec<Constant> = Vec::new();
        let mut slot_of: Vec<Option<u32>> = vec![None; graph.value_count()];
        for (index, value) in graph.values().iter().enumerate() {
            if !needed[index] {
                continue;
            }
            if let Some(source) = self.alias_of[index] {
                slot_of[index] =
                    Some(
                        slot_of[source.index()].ok_or_else(|| Fn3Error::InvalidShape {
                            context: "Survey::finish",
                            detail: format!(
                                "identity-folded value `{}` aliases `{}`, which has no slot; the \
                             source must be marked needed before the slot table is built",
                                value.name,
                                graph.value(source).name
                            ),
                        })?,
                    );
                continue;
            }
            let shape = self.shapes[index]
                .clone()
                .ok_or_else(|| Fn3Error::InvalidShape {
                    context: "Survey::finish",
                    detail: format!(
                        "value `{}` is used but its shape was never resolved",
                        value.name
                    ),
                })?;
            let role = match value.origin {
                ValueOrigin::GraphInput => SlotRole::GraphInput,
                ValueOrigin::Weight(id) => SlotRole::Weight { id },
                ValueOrigin::Node(_) => match &self.data[index] {
                    Some(data) => {
                        let constant = intern_constant(
                            &mut constants,
                            Constant::new(shape.clone(), data.clone())?,
                        );
                        SlotRole::Constant { index: constant }
                    }
                    None => match instruction_of_value[index] {
                        Some(instruction) => SlotRole::InstructionOutput { instruction },
                        None => {
                            return Err(Fn3Error::InvalidShape {
                                context: "Survey::finish",
                                detail: format!(
                                    "value `{}` is neither folded nor produced by a surviving \
                                     instruction",
                                    value.name
                                ),
                            })
                        }
                    },
                },
            };
            slot_of[index] = Some(slots.len() as u32);
            slots.push(SlotPlan::new(value.name.clone(), value.dtype, shape, role));
        }

        let mut instructions = Vec::with_capacity(self.surviving.len());
        for &node_index in &self.surviving {
            let node = &graph.nodes()[node_index as usize];
            let attr =
                self.attrs[node_index as usize]
                    .clone()
                    .ok_or_else(|| Fn3Error::InvalidShape {
                        context: "Survey::finish",
                        detail: format!(
                            "surviving node {} has no typed attribute",
                            node.onnx_index
                        ),
                    })?;
            let mut inputs = Vec::with_capacity(node.inputs.len());
            for id in &node.inputs {
                inputs.push(slot_of[id.index()].ok_or_else(|| Fn3Error::InvalidShape {
                    context: "Survey::finish",
                    detail: format!(
                        "node {} `{}` reads value `{}` which was pruned as dead; a live operand \
                         must keep its slot",
                        node.onnx_index,
                        node.name,
                        graph.value(*id).name
                    ),
                })?);
            }
            let mut outputs = Vec::with_capacity(node.outputs.len());
            for id in &node.outputs {
                outputs.push(slot_of[id.index()].ok_or_else(|| Fn3Error::InvalidShape {
                    context: "Survey::finish",
                    detail: format!(
                        "node {} `{}` writes value `{}` which has no slot",
                        node.onnx_index,
                        node.name,
                        graph.value(*id).name
                    ),
                })?);
            }
            instructions.push(Instruction {
                op: node.op,
                inputs,
                outputs,
                params: attr.encode(),
            });
        }
        let execution = ExecutionPlan::new(instructions, slots.len() as u32)?;

        // Preflight: the plan is a promise, so ask the backend now. A `Partial`
        // op-level answer is refined by `Backend::admits` against the plan's
        // concrete operands: a kernel whose subset is "f32 and 4- or 8-byte
        // elements" or "the trailing axis" can promise *this* instruction while
        // its op-level verdict stays honest. `admits` never widens an
        // `Unsupported` verdict, and the default body returns `supports`, so a
        // backend without a concrete rule keeps exactly its previous verdicts.
        //
        // Element types are checked here rather than through
        // `Capabilities::dtypes`: the aggregate table is a statement about the
        // backend, and the pinned decoder graph is the case where it is not a
        // statement about any instruction (Vulkan lists f32 as its compute
        // dtype while `gather`, `transpose`, `shape` and `cast` legitimately
        // move i64 indices). Every slot is an operand of at least one
        // instruction, so this loop is a complete dtype gate; a slot no
        // instruction touches falls back to the aggregate table below.
        let mut admitted = vec![false; slots.len()];
        for (position, instruction) in execution.instructions().iter().enumerate() {
            let operands = instruction.inputs.len() + instruction.outputs.len();
            let mut shapes = Vec::with_capacity(operands);
            let mut dtypes = Vec::with_capacity(operands);
            for &slot in instruction.inputs.iter().chain(instruction.outputs.iter()) {
                let plan = slots
                    .get(slot as usize)
                    .ok_or_else(|| Fn3Error::OutOfBounds {
                        what: "Survey::finish (admission slot)",
                        index: u64::from(slot),
                        len: slots.len() as u64,
                    })?;
                shapes.push(plan.shape().clone());
                dtypes.push(plan.dtype());
                admitted[slot as usize] = true;
            }
            backend
                .admits(instruction.op, &shapes, &dtypes, &instruction.params)
                .require_full(instruction.op, backend.name())
                .map_err(|error| at_instruction(error, position, instruction.op))?;
        }
        self.check_capabilities(backend, &slots, &admitted)?;

        let mask = self.mask_constants(&execution, &slots)?;
        let mut inputs = Vec::new();
        for id in graph.inputs() {
            let slot = slot_of[id.index()].ok_or_else(|| Fn3Error::InvalidShape {
                context: "Survey::finish",
                detail: format!("graph input `{}` has no slot", graph.value(*id).name),
            })?;
            let plan = &slots[slot as usize];
            inputs.push(BoundValue::new(
                graph.value(*id).name.clone(),
                slot,
                plan.dtype(),
                plan.shape().clone(),
            ));
        }
        let mut outputs = Vec::new();
        for id in graph.outputs() {
            let slot = slot_of[id.index()].ok_or_else(|| Fn3Error::InvalidShape {
                context: "Survey::finish",
                detail: format!("graph output `{}` has no slot", graph.value(*id).name),
            })?;
            let plan = &slots[slot as usize];
            outputs.push(BoundValue::new(
                graph.value(*id).name.clone(),
                slot,
                plan.dtype(),
                plan.shape().clone(),
            ));
        }

        Plan::assemble(PlanParts {
            execution,
            slots,
            constants,
            folded: self.folded,
            mask,
            inputs,
            outputs,
            dims: self.dims,
            envelope,
            backend: backend.name(),
        })
    }

    fn check_capabilities(
        &self,
        backend: &dyn Backend,
        slots: &[SlotPlan],
        admitted: &[bool],
    ) -> Result<(), Fn3Error> {
        let capabilities = backend.capabilities();
        for (index, slot) in slots.iter().enumerate() {
            // Ranks and buffer sizes are genuinely backend-wide limits.
            capabilities.check_rank(slot.shape().rank())?;
            capabilities.check_buffer(slot.byte_len()?)?;
            // Element types are admitted per instruction (see `finish`). A
            // slot that no instruction touches has no concrete kernel to ask,
            // so the aggregate table remains the gate for that case only.
            if !admitted.get(index).copied().unwrap_or(false) {
                capabilities.check_dtype(None, slot.dtype())?;
            }
        }
        Ok(())
    }

    /// Every float constant used as a mask fill, with a finiteness check.
    fn mask_constants(
        &self,
        execution: &ExecutionPlan,
        slots: &[SlotPlan],
    ) -> Result<Vec<MaskConstant>, Fn3Error> {
        let mut found = Vec::new();
        for (position, instruction) in execution.instructions().iter().enumerate() {
            let operands: &[(usize, MaskRole)] = match instruction.op {
                OpId::ConstantOfShape => &[(1, MaskRole::Fill)],
                OpId::Where => &[(1, MaskRole::Selected), (2, MaskRole::Alternative)],
                OpId::MaskedAttention => &[(2, MaskRole::AttentionMask)],
                _ => &[],
            };
            for &(operand, role) in operands {
                let Some(&slot) = instruction.inputs.get(operand) else {
                    continue;
                };
                let plan = &slots[slot as usize];
                if plan.dtype() != DType::F32 || plan.shape().numel()? != 1 {
                    continue;
                }
                let SlotRole::Weight { id } = plan.role() else {
                    continue;
                };
                let weight = self.weights.get(*id).ok_or_else(|| Fn3Error::OutOfBounds {
                    what: "Survey::mask_constants (weight id)",
                    index: id.index() as u64,
                    len: self.weights.len() as u64,
                })?;
                let bytes = weight.bytes();
                let value = f32::from_le_bytes(bytes[..4].try_into().expect("four bytes checked"));
                found.push(MaskConstant {
                    instruction: position as u32,
                    op: instruction.op,
                    role,
                    value,
                    origin: format!("weight `{}`", plan.name()),
                });
            }
        }
        Ok(found)
    }
}

// ---------------------------------------------------------------------------
// Shape algebra
// ---------------------------------------------------------------------------

/// Add the plan position to an unsupported-op diagnostic.
fn at_instruction(error: Fn3Error, position: usize, op: OpId) -> Fn3Error {
    match error {
        Fn3Error::UnsupportedOp {
            op: reported,
            backend,
            reason,
        } => Fn3Error::UnsupportedOp {
            op: reported,
            backend,
            reason: format!("{reason} (at plan instruction {position}, op {op})"),
        },
        other => other,
    }
}

fn node_error(node: &GraphNode, detail: impl Into<String>) -> Fn3Error {
    Fn3Error::InvalidShape {
        context: "specialize::shape",
        detail: format!(
            "node {} ({}) `{}`: {}",
            node.onnx_index,
            node.op,
            node.name,
            detail.into()
        ),
    }
}

fn with_node(error: Fn3Error, node: &GraphNode) -> Fn3Error {
    match error {
        Fn3Error::ShapeMismatch {
            op,
            expected,
            actual,
        } => Fn3Error::ShapeMismatch {
            op,
            expected: format!(
                "{expected} (node {} `{}` of {})",
                node.onnx_index, node.name, node.op
            ),
            actual,
        },
        other => other,
    }
}

/// Normalise an ONNX axis against `rank`.
fn normalise_axis(axis: i64, rank: usize, node: &GraphNode, role: &str) -> Result<usize, Fn3Error> {
    let rank_i64 = rank as i64;
    if axis >= rank_i64 || axis < -rank_i64 {
        return Err(node_error(
            node,
            format!("{role} {axis} is outside the rank-{rank} tensor"),
        ));
    }
    Ok(if axis < 0 {
        (axis + rank_i64) as usize
    } else {
        axis as usize
    })
}

/// Turn a declared [`ValueShape`] into a concrete [`Shape`], or name the value
/// and the axis that could not be resolved.
fn substitute(shape: &ValueShape, dims: LiveDims, name: &str) -> Result<Shape, Fn3Error> {
    match shape {
        ValueShape::Unknown => Err(Fn3Error::InvalidShape {
            context: "specialize::resolve",
            detail: format!("value `{name}` has an unknown rank; a plan needs a concrete shape"),
        }),
        ValueShape::Dims(axes) => {
            let mut extents = Vec::with_capacity(axes.len());
            for (axis, dim) in axes.iter().enumerate() {
                let extent = match dim {
                    Dim::Known(extent) => *extent,
                    Dim::Symbol(symbol) => dims.get(*symbol),
                    Dim::Unknown(reason) => {
                        return Err(Fn3Error::InvalidShape {
                            context: "specialize::resolve",
                            detail: format!(
                                "value `{name}` axis {axis} is unresolved ({reason}); the planner \
                                 rejects rather than guessing an extent"
                            ),
                        })
                    }
                };
                extents.push(extent);
            }
            Ok(Shape::from_dims(extents))
        }
    }
}

fn matmul_shape(a: &Shape, b: &Shape, node: &GraphNode) -> Result<Shape, Fn3Error> {
    if a.rank() < 2 || b.rank() < 2 {
        return Err(Fn3Error::UnsupportedOp {
            op: OpId::MatMul,
            backend: PLANNER,
            reason: format!(
                "node {} `{}` multiplies rank-{} and rank-{} operands; the alpha contract is \
                 rank >= 2 (a rank-1 ONNX operand would need the squeeze/unsqueeze composition). \
                 Owner: FN3-10 (#56)",
                node.onnx_index,
                node.name,
                a.rank(),
                b.rank()
            ),
        });
    }
    let m = a.dims()[a.rank() - 2];
    let k = a.dims()[a.rank() - 1];
    let k2 = b.dims()[b.rank() - 2];
    let n = b.dims()[b.rank() - 1];
    if k != k2 {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(OpId::MatMul),
            expected: format!("inner extent {k} from the left operand"),
            actual: format!(
                "inner extent {k2} from the right operand (node {})",
                node.onnx_index
            ),
        });
    }
    let left_batch = Shape::from_dims(a.dims()[..a.rank() - 2].to_vec());
    let right_batch = Shape::from_dims(b.dims()[..b.rank() - 2].to_vec());
    let batch = left_batch
        .broadcast_with(&right_batch, OpId::MatMul)
        .map_err(|error| with_node(error, node))?;
    let mut dims = batch.dims().to_vec();
    dims.push(m);
    dims.push(n);
    Ok(Shape::from_dims(dims))
}

fn reduce_shape(
    input: &Shape,
    axes: &[i64],
    keep_dims: bool,
    node: &GraphNode,
) -> Result<Shape, Fn3Error> {
    let mut normalised = Vec::with_capacity(axes.len());
    for &axis in axes {
        let axis = normalise_axis(axis, input.rank(), node, "reduction axis")?;
        if !normalised.contains(&axis) {
            normalised.push(axis);
        }
    }
    // ONNX: an absent or empty axis list reduces everything.
    if normalised.is_empty() {
        normalised.extend(0..input.rank());
    }
    let mut dims = Vec::with_capacity(input.rank());
    for (axis, &extent) in input.dims().iter().enumerate() {
        if normalised.contains(&axis) {
            if keep_dims {
                dims.push(1);
            }
        } else {
            dims.push(extent);
        }
    }
    Ok(Shape::from_dims(dims))
}

fn reshape_shape(input: &Shape, target: &[i64], node: &GraphNode) -> Result<Shape, Fn3Error> {
    let total = input.numel()?;
    let mut dims: Vec<u64> = Vec::with_capacity(target.len());
    let mut inferred: Option<usize> = None;
    for (position, &dim) in target.iter().enumerate() {
        match dim {
            0 => {
                // `allowzero = 0`, the only form the importer accepts: 0 copies
                // the input extent at the same position.
                let extent = input.dim(position).ok_or_else(|| {
                    node_error(
                        node,
                        format!(
                            "Reshape target position {position} is 0 but the input has only rank {}",
                            input.rank()
                        ),
                    )
                })?;
                dims.push(extent);
            }
            -1 => {
                if inferred.replace(position).is_some() {
                    return Err(node_error(
                        node,
                        "Reshape target contains more than one -1".to_string(),
                    ));
                }
                dims.push(1);
            }
            other if other < 0 => {
                return Err(node_error(
                    node,
                    format!("Reshape target contains the negative extent {other}"),
                ))
            }
            other => dims.push(other as u64),
        }
    }
    if let Some(position) = inferred {
        let known = dims.iter().try_fold(1u64, |product, &dim| {
            product
                .checked_mul(dim)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "Survey::reshape_shape",
                    axis: Some(position),
                    detail: format!("known extents of the target {target:?} exceed u64::MAX"),
                })
        })?;
        if known == 0 {
            return Err(node_error(
                node,
                "Reshape target has a 0 wherever -1 must be inferred, which is ambiguous"
                    .to_string(),
            ));
        }
        if !total.is_multiple_of(known) {
            return Err(node_error(
                node,
                format!(
                    "Reshape from {input} ({total} elements) to {target:?} is not divisible: \
                     {total} % {known} != 0"
                ),
            ));
        }
        dims[position] = total / known;
    }
    let shape = Shape::from_dims(dims);
    if shape.numel()? != total {
        return Err(node_error(
            node,
            format!(
                "Reshape target {target:?} resolves to {shape} with {} elements, not the input's \
                 {total}",
                shape.numel()?
            ),
        ));
    }
    Ok(shape)
}

fn flatten_shape(input: &Shape, node: &GraphNode) -> Result<Shape, Fn3Error> {
    // ONNX default axis is 1. `Flatten.axis` has no `OpParams` field, so the
    // planner can only honour the default; a non-default axis makes the
    // resolved shape disagree with the import's inference and is refused there.
    let axis = 1.min(input.rank());
    let outer = input.dims()[..axis]
        .iter()
        .try_fold(1u64, |product, &dim| product.checked_mul(dim))
        .ok_or_else(|| Fn3Error::ShapeOverflow {
            context: "Survey::flatten_shape",
            axis: None,
            detail: format!("outer extent of {input} exceeds u64::MAX"),
        })?;
    let inner = input.dims()[axis..]
        .iter()
        .try_fold(1u64, |product, &dim| product.checked_mul(dim))
        .ok_or_else(|| Fn3Error::ShapeOverflow {
            context: "Survey::flatten_shape",
            axis: None,
            detail: format!("inner extent of {input} exceeds u64::MAX"),
        })?;
    let _ = node;
    Ok(Shape::from_dims([outer, inner]))
}

fn squeeze_shape(input: &Shape, axes: Option<&[i64]>, node: &GraphNode) -> Result<Shape, Fn3Error> {
    let mut removed = Vec::new();
    match axes {
        Some(axes) => {
            for &axis in axes {
                let axis = normalise_axis(axis, input.rank(), node, "Squeeze axis")?;
                if input.dim(axis) != Some(1) {
                    return Err(node_error(
                        node,
                        format!(
                            "Squeeze axis {axis} has extent {:?}, not 1",
                            input.dim(axis)
                        ),
                    ));
                }
                removed.push(axis);
            }
        }
        None => {
            for (axis, &extent) in input.dims().iter().enumerate() {
                if extent == 1 {
                    removed.push(axis);
                }
            }
        }
    }
    Ok(Shape::from_dims(
        input
            .dims()
            .iter()
            .enumerate()
            .filter(|(axis, _)| !removed.contains(axis))
            .map(|(_, &extent)| extent)
            .collect::<Vec<_>>(),
    ))
}

fn unsqueeze_shape(input: &Shape, axes: &[i64], node: &GraphNode) -> Result<Shape, Fn3Error> {
    let output_rank = input.rank() + axes.len();
    let mut positions = Vec::with_capacity(axes.len());
    for &axis in axes {
        let position = normalise_axis(axis, output_rank, node, "Unsqueeze axis")?;
        if positions.contains(&position) {
            return Err(node_error(
                node,
                format!("Unsqueeze repeats the axis {position}"),
            ));
        }
        positions.push(position);
    }
    positions.sort_unstable();
    let mut dims = Vec::with_capacity(output_rank);
    let mut source = input.dims().iter();
    for position in 0..output_rank {
        if positions.contains(&position) {
            dims.push(1);
        } else {
            dims.push(
                *source
                    .next()
                    .expect("one source extent per non-inserted axis"),
            );
        }
    }
    Ok(Shape::from_dims(dims))
}

fn concat_shape(shapes: &[&Shape], axis: i64, node: &GraphNode) -> Result<Shape, Fn3Error> {
    let first = shapes
        .first()
        .ok_or_else(|| node_error(node, "Concat needs at least one input".to_string()))?;
    let axis = normalise_axis(axis, first.rank(), node, "Concat axis")?;
    let mut dims = first.dims().to_vec();
    let mut total = 0u64;
    for shape in shapes {
        if shape.rank() != first.rank() {
            return Err(node_error(
                node,
                format!(
                    "Concat inputs have ranks {} and {}",
                    first.rank(),
                    shape.rank()
                ),
            ));
        }
        for (position, (&mine, &theirs)) in dims.iter().zip(shape.dims()).enumerate() {
            if position != axis && mine != theirs {
                return Err(node_error(
                    node,
                    format!(
                        "Concat inputs disagree on axis {position}: {mine} vs {theirs} (shape \
                         {shape})"
                    ),
                ));
            }
        }
        total = total
            .checked_add(shape.dim(axis).expect("rank checked"))
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "Survey::concat_shape",
                axis: Some(axis),
                detail: format!(
                    "concatenated extent exceeds u64::MAX (node {})",
                    node.onnx_index
                ),
            })?;
    }
    dims[axis] = total;
    Ok(Shape::from_dims(dims))
}

/// Resolved Slice parameters, already normalised against the input rank.
struct SlicePlan {
    starts: Vec<i64>,
    steps: Vec<i64>,
    extents: Vec<u64>,
    axes: Vec<usize>,
}

fn slice_plan(
    input: &Shape,
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
    node: &GraphNode,
) -> Result<SlicePlan, Fn3Error> {
    let axes: Vec<i64> = if axes.is_empty() {
        (0..starts.len() as i64).collect()
    } else {
        axes.to_vec()
    };
    if starts.len() != axes.len() || ends.len() != axes.len() {
        return Err(node_error(
            node,
            format!(
                "Slice starts/ends/axes lengths disagree: {} / {} / {}",
                starts.len(),
                ends.len(),
                axes.len()
            ),
        ));
    }
    if !steps.is_empty() && steps.len() != axes.len() {
        return Err(node_error(
            node,
            format!(
                "Slice steps length {} does not match axes length {}",
                steps.len(),
                axes.len()
            ),
        ));
    }
    let mut plan = SlicePlan {
        starts: Vec::with_capacity(axes.len()),
        steps: Vec::with_capacity(axes.len()),
        extents: Vec::with_capacity(axes.len()),
        axes: Vec::with_capacity(axes.len()),
    };
    for (position, &axis) in axes.iter().enumerate() {
        let axis = normalise_axis(axis, input.rank(), node, "Slice axis")?;
        if plan.axes.contains(&axis) {
            return Err(node_error(node, format!("Slice repeats the axis {axis}")));
        }
        let extent = input.dim(axis).expect("axis normalised against the rank");
        let step = if steps.is_empty() { 1 } else { steps[position] };
        if step <= 0 {
            return Err(Fn3Error::UnsupportedOp {
                op: OpId::Slice,
                backend: PLANNER,
                reason: format!(
                    "node {} `{}` slices with step {step}; a negative or zero step needs a \
                     reversed view, which the alpha view contract does not have. Owner: FN3-10 \
                     (#56) for the plan rule, FN3-07 (#53) for the kernel",
                    node.onnx_index, node.name
                ),
            });
        }
        let extent_i64 = extent as i64;
        let mut start = starts[position];
        if start < 0 {
            start += extent_i64;
        }
        start = start.clamp(0, extent_i64);
        let mut end = ends[position];
        if end < 0 {
            end += extent_i64;
        }
        end = end.clamp(0, extent_i64);
        let length = if end > start {
            ((end - start + step - 1) / step) as u64
        } else {
            0
        };
        plan.starts.push(start);
        plan.steps.push(step);
        plan.extents.push(length);
        plan.axes.push(axis);
    }
    Ok(plan)
}

fn slice_shape(
    input: &Shape,
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
    node: &GraphNode,
) -> Result<Shape, Fn3Error> {
    let plan = slice_plan(input, starts, ends, axes, steps, node)?;
    let mut dims = input.dims().to_vec();
    for (position, &axis) in plan.axes.iter().enumerate() {
        dims[axis] = plan.extents[position];
    }
    Ok(Shape::from_dims(dims))
}

fn split_shapes(
    input: &Shape,
    axis: i64,
    sizes: Option<&[i64]>,
    outputs: usize,
    node: &GraphNode,
) -> Result<Vec<Shape>, Fn3Error> {
    let axis = normalise_axis(axis, input.rank(), node, "Split axis")?;
    let extent = input.dim(axis).expect("axis normalised against the rank");
    let sizes: Vec<u64> = match sizes {
        Some(sizes) => {
            if sizes.len() != outputs {
                return Err(node_error(
                    node,
                    format!(
                        "Split has {outputs} outputs but {} split sizes",
                        sizes.len()
                    ),
                ));
            }
            sizes
                .iter()
                .map(|&size| {
                    u64::try_from(size)
                        .map_err(|_| node_error(node, format!("Split size {size} is negative")))
                })
                .collect::<Result<_, _>>()?
        }
        None => {
            if outputs == 0 || !extent.is_multiple_of(outputs as u64) {
                return Err(node_error(
                    node,
                    format!("Split of extent {extent} into {outputs} equal parts is not exact"),
                ));
            }
            vec![extent / outputs as u64; outputs]
        }
    };
    let sum: u64 = sizes.iter().sum();
    if sum != extent {
        return Err(node_error(
            node,
            format!("Split sizes {sizes:?} sum to {sum}, not the axis extent {extent}"),
        ));
    }
    Ok(sizes
        .into_iter()
        .map(|size| {
            let mut dims = input.dims().to_vec();
            dims[axis] = size;
            Shape::from_dims(dims)
        })
        .collect())
}

fn gather_shape(
    data: &Shape,
    indices: &Shape,
    axis: i64,
    node: &GraphNode,
) -> Result<Shape, Fn3Error> {
    if data.rank() == 0 {
        return Err(node_error(
            node,
            "Gather data is a scalar; ONNX requires rank >= 1".to_string(),
        ));
    }
    let axis = normalise_axis(axis, data.rank(), node, "Gather axis")?;
    let mut dims = data.dims()[..axis].to_vec();
    dims.extend_from_slice(indices.dims());
    dims.extend_from_slice(&data.dims()[axis + 1..]);
    Ok(Shape::from_dims(dims))
}

fn range_len(start: i64, limit: i64, delta: i64, node: &GraphNode) -> Result<u64, Fn3Error> {
    if delta == 0 {
        return Err(node_error(node, "Range delta is zero".to_string()));
    }
    let span = if delta > 0 {
        limit.saturating_sub(start)
    } else {
        start.saturating_sub(limit)
    };
    if span <= 0 {
        return Ok(0);
    }
    let step = delta.unsigned_abs();
    Ok((span as u64).div_ceil(step))
}

// ---------------------------------------------------------------------------
// Constant evaluation helpers
// ---------------------------------------------------------------------------

fn decode_weight(weight: &WeightTensor) -> Option<ConstantData> {
    match weight.dtype() {
        DType::I64 => Some(ConstantData::I64(
            weight
                .bytes()
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
                .collect(),
        )),
        DType::I32 => Some(ConstantData::I32(
            weight
                .bytes()
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect(),
        )),
        DType::Bool => Some(ConstantData::Bool(weight.bytes().to_vec())),
        // Floating-point weights stay data, not planner state.
        DType::F32 => None,
    }
}

fn constant_from_i64(
    dtype: DType,
    values: Vec<i64>,
    shape: &Shape,
    node: &GraphNode,
) -> Result<ConstantData, Fn3Error> {
    let data = match dtype {
        DType::I64 => ConstantData::I64(values),
        DType::I32 => ConstantData::I32(
            values
                .into_iter()
                .map(|value| {
                    i32::try_from(value).map_err(|_| {
                        node_error(
                            node,
                            format!("folded value {value} does not fit the I32 output"),
                        )
                    })
                })
                .collect::<Result<_, _>>()?,
        ),
        DType::Bool => ConstantData::Bool(
            values
                .into_iter()
                .map(|value| u8::from(value != 0))
                .collect(),
        ),
        DType::F32 => {
            return Err(Fn3Error::UnsupportedOp {
                op: node.op,
                backend: PLANNER,
                reason: "the planner only folds integer/bool constants; an f32 constant would \
                         need a float evaluation path that the backend already owns. Owner: \
                         FN3-10 (#56)"
                    .to_string(),
            })
        }
    };
    if data.len() as u64 != shape.numel()? {
        return Err(node_error(
            node,
            format!(
                "folded {} payload has {} elements but {shape} needs {}",
                dtype,
                data.len(),
                shape.numel()?
            ),
        ));
    }
    Ok(data)
}

fn intern_constant(constants: &mut Vec<Constant>, constant: Constant) -> u32 {
    if let Some(position) = constants
        .iter()
        .position(|candidate| candidate == &constant)
    {
        return position as u32;
    }
    constants.push(constant);
    (constants.len() - 1) as u32
}

fn ravel(shape: &Shape, coordinates: &[u64]) -> u64 {
    let mut index = 0u64;
    for (axis, &extent) in shape.dims().iter().enumerate() {
        index = index * extent + coordinates[axis];
    }
    index
}

fn coordinates_of(shape: &Shape, mut index: u64) -> Vec<u64> {
    let mut coordinates = vec![0u64; shape.rank()];
    for axis in (0..shape.rank()).rev() {
        let extent = shape.dims()[axis];
        if extent != 0 {
            coordinates[axis] = index % extent;
            index /= extent;
        }
    }
    coordinates
}

/// Right-aligned broadcast of one coordinate into an operand's shape.
fn broadcast_coordinate(output: &[u64], operand: &Shape) -> Vec<u64> {
    let rank = operand.rank();
    let offset = output.len().saturating_sub(rank);
    let mut coordinates = vec![0u64; rank];
    for axis in 0..rank {
        let extent = operand.dims()[axis];
        coordinates[axis] = if extent == 1 {
            0
        } else {
            output[offset + axis]
        };
    }
    coordinates
}

fn broadcast_zip(
    left: &Shape,
    right: &Shape,
    output: &Shape,
    node: &GraphNode,
    f: &mut dyn FnMut(usize, usize),
) -> Result<(), Fn3Error> {
    let expected = left
        .broadcast_with(right, node.op)
        .map_err(|error| with_node(error, node))?;
    if &expected != output {
        return Err(node_error(
            node,
            format!("broadcast of {left} and {right} is {expected}, not {output}"),
        ));
    }
    for linear in 0..output.numel()? {
        let coordinate = coordinates_of(output, linear);
        let left_index = ravel(left, &broadcast_coordinate(&coordinate, left));
        let right_index = ravel(right, &broadcast_coordinate(&coordinate, right));
        f(left_index as usize, right_index as usize);
    }
    Ok(())
}

fn broadcast_zip3(
    a: &Shape,
    b: &Shape,
    c: &Shape,
    output: &Shape,
    node: &GraphNode,
    f: &mut dyn FnMut(usize, usize, usize),
) -> Result<(), Fn3Error> {
    let first = a
        .broadcast_with(b, node.op)
        .map_err(|error| with_node(error, node))?;
    let expected = first
        .broadcast_with(c, node.op)
        .map_err(|error| with_node(error, node))?;
    if &expected != output {
        return Err(node_error(
            node,
            format!("broadcast of {a}, {b} and {c} is {expected}, not {output}"),
        ));
    }
    for linear in 0..output.numel()? {
        let coordinate = coordinates_of(output, linear);
        f(
            ravel(a, &broadcast_coordinate(&coordinate, a)) as usize,
            ravel(b, &broadcast_coordinate(&coordinate, b)) as usize,
            ravel(c, &broadcast_coordinate(&coordinate, c)) as usize,
        );
    }
    Ok(())
}

/// Right-aligned gather along one axis, in row-major output order.
fn gather_values(
    data: &[i64],
    data_shape: &Shape,
    indices: &[i64],
    axis: usize,
    node: &GraphNode,
) -> Result<Vec<i64>, Fn3Error> {
    let outer: u64 = data_shape.dims()[..axis].iter().product();
    let extent = data_shape.dims()[axis];
    let inner: u64 = data_shape.dims()[axis + 1..].iter().product();
    let resolved: Vec<u64> = indices
        .iter()
        .map(|&index| {
            let mut index = index;
            if index < 0 {
                index += extent as i64;
            }
            if index < 0 || index as u64 >= extent {
                return Err(node_error(
                    node,
                    format!("Gather index {index} is outside the axis extent {extent}"),
                ));
            }
            Ok(index as u64)
        })
        .collect::<Result<_, _>>()?;
    let mut out = Vec::with_capacity(resolved.len() * inner as usize * outer as usize);
    for outer_index in 0..outer.max(1) {
        for &index in &resolved {
            let base = (outer_index * extent + index) * inner;
            out.extend_from_slice(&data[base as usize..(base + inner) as usize]);
        }
    }
    Ok(out)
}

fn slice_values(
    data: &[i64],
    shape: &Shape,
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
    node: &GraphNode,
) -> Result<Vec<i64>, Fn3Error> {
    let plan = slice_plan(shape, starts, ends, axes, steps, node)?;
    let mut dims = shape.dims().to_vec();
    for (position, &axis) in plan.axes.iter().enumerate() {
        dims[axis] = plan.extents[position];
    }
    let output = Shape::from_dims(dims);
    let mut values = Vec::with_capacity(output.numel()? as usize);
    for linear in 0..output.numel()? {
        let mut coordinate = coordinates_of(&output, linear);
        for (position, &axis) in plan.axes.iter().enumerate() {
            coordinate[axis] =
                (plan.starts[position] as u64) + coordinate[axis] * plan.steps[position] as u64;
        }
        values.push(data[ravel(shape, &coordinate) as usize]);
    }
    Ok(values)
}

fn concat_values(
    sources: &[Vec<i64>],
    shapes: &[&Shape],
    axis: i64,
    node: &GraphNode,
) -> Result<Vec<i64>, Fn3Error> {
    let first = shapes
        .first()
        .ok_or_else(|| node_error(node, "Concat needs at least one input".to_string()))?;
    let axis = normalise_axis(axis, first.rank(), node, "Concat axis")?;
    let output = concat_shape(shapes, axis as i64, node)?;
    let outer: u64 = output.dims()[..axis].iter().product();
    let inner: u64 = output.dims()[axis + 1..].iter().product();
    let pieces: Vec<u64> = shapes
        .iter()
        .map(|shape| shape.dim(axis).expect("rank checked"))
        .collect();
    let mut values = Vec::with_capacity(output.numel()? as usize);
    for outer_index in 0..outer {
        for (source, (&piece, shape)) in sources.iter().zip(pieces.iter().zip(shapes)) {
            let base = outer_index * piece * inner;
            let slice = &source[base as usize..(base + piece * inner) as usize];
            values.extend_from_slice(slice);
            let _ = shape;
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::model::weights::WeightId;

    fn node(op: OpId) -> GraphNode {
        GraphNode {
            onnx_index: 0,
            name: "test".to_string(),
            onnx_op: "Test".to_string(),
            op,
            inputs: Vec::new(),
            outputs: Vec::new(),
            params: OpParams::new(),
        }
    }

    #[test]
    fn plan_attr_round_trips_every_encoding() {
        let cases = [
            (OpId::Gather, PlanAttr::Axis(-2)),
            (OpId::Softmax, PlanAttr::Axis(-1)),
            (
                OpId::ArgMax,
                PlanAttr::AxisKeepDims {
                    axis: 0,
                    keep_dims: true,
                },
            ),
            (OpId::Transpose, PlanAttr::Perm(vec![3, 1, 0, 2])),
            (
                OpId::ReduceMean,
                PlanAttr::Reduce {
                    axes: vec![1, 2],
                    keep_dims: false,
                },
            ),
            (
                OpId::LayerNorm,
                PlanAttr::Epsilon {
                    axes: vec![2],
                    epsilon: 1e-5,
                },
            ),
            (OpId::MaskedAttention, PlanAttr::Attention { causal: true }),
            (OpId::Add, PlanAttr::None),
        ];
        for (op, attr) in cases {
            let params = attr.encode();
            let decoded = PlanAttr::decode(op, &params).unwrap();
            assert_eq!(decoded, attr, "op {op}");
            assert_eq!(decoded.encode(), params, "op {op}");
        }
    }

    #[test]
    fn plan_attr_rejects_an_attribute_the_op_does_not_define() {
        let params = OpParams::new().with_axes([1]);
        let error = PlanAttr::decode(OpId::Add, &params).unwrap_err();
        assert!(error.to_string().contains("takes no attributes"), "{error}");
        let error = PlanAttr::decode(OpId::MatMul, &params).unwrap_err();
        assert!(error.to_string().contains("takes no attributes"), "{error}");
    }

    #[test]
    fn causal_attention_rides_in_keep_dims() {
        let params = PlanAttr::Attention { causal: true }.encode();
        assert!(params.keep_dims);
        assert!(params.axes.is_empty());
        assert_eq!(params.epsilon, 0.0);
        assert_eq!(
            PlanAttr::decode(OpId::MaskedAttention, &params).unwrap(),
            PlanAttr::Attention { causal: true }
        );
    }

    #[test]
    fn reshape_infers_the_flattening_axis() {
        let input = Shape::from_dims([2, 7, 128]);
        let shape = reshape_shape(&input, &[-1, 128], &node(OpId::Reshape)).unwrap();
        assert_eq!(shape, Shape::from_dims([14, 128]));
        let shape = reshape_shape(&input, &[0, -1], &node(OpId::Reshape)).unwrap();
        assert_eq!(shape, Shape::from_dims([2, 896]));
        assert!(reshape_shape(&input, &[-1, -1], &node(OpId::Reshape)).is_err());
        assert!(reshape_shape(&input, &[-1, 5], &node(OpId::Reshape)).is_err());
    }

    #[test]
    fn slice_clamps_like_onnx() {
        let input = Shape::from_dims([4]);
        let shape = slice_shape(&input, &[-2], &[-1], &[0], &[1], &node(OpId::Slice)).unwrap();
        assert_eq!(shape, Shape::from_dims([1]));
        let shape =
            slice_shape(&input, &[-1], &[i64::MAX], &[0], &[1], &node(OpId::Slice)).unwrap();
        assert_eq!(shape, Shape::from_dims([1]));
        let values = slice_values(
            &[10, 20, 30, 40],
            &input,
            &[-2],
            &[-1],
            &[0],
            &[1],
            &node(OpId::Slice),
        )
        .unwrap();
        assert_eq!(values, vec![30]);
    }

    #[test]
    fn broadcast_zip_is_right_aligned() {
        let left = Shape::from_dims([2, 3]);
        let right = Shape::from_dims([3]);
        let mut visited = Vec::new();
        broadcast_zip(&left, &right, &left, &node(OpId::Add), &mut |i, j| {
            visited.push((i, j));
        })
        .unwrap();
        assert_eq!(visited.len(), 6);
        assert_eq!(visited[0], (0, 0));
        assert_eq!(visited[1], (1, 1));
        assert_eq!(visited[3], (3, 0));
    }

    #[test]
    fn range_length_matches_onnx() {
        assert_eq!(range_len(0, 5, 1, &node(OpId::Range)).unwrap(), 5);
        assert_eq!(range_len(5, 0, -1, &node(OpId::Range)).unwrap(), 5);
        assert_eq!(range_len(5, 0, 1, &node(OpId::Range)).unwrap(), 0);
        assert_eq!(range_len(0, 6, 2, &node(OpId::Range)).unwrap(), 3);
        assert!(range_len(0, 5, 0, &node(OpId::Range)).is_err());
    }

    #[test]
    fn feasible_corners_respect_the_total_bound() {
        let envelope = ShapeEnvelope::new([
            DimBound::new(1, 1).unwrap(),
            DimBound::new(1, 13).unwrap(),
            DimBound::new(0, 17).unwrap(),
            DimBound::new(1, 18).unwrap(),
            DimBound::exact(2).unwrap(),
            DimBound::exact(2).unwrap(),
            DimBound::exact(64).unwrap(),
        ]);
        let geometry = DecoderGeometry {
            layers: 2,
            hidden_size: 128,
            mlp_hidden_size: 512,
            query_heads: 2,
            kv_heads: 2,
            head_dim: 64,
            vocab_size: 50_257,
            positions: crate::inference::import::PositionEmbedding::Learned {
                weight: WeightId::new(0),
                max_positions: 1024,
            },
            layer_norm_epsilon: Some(1e-5),
            token_embedding: WeightId::new(1),
            logits: ValueId::new(0),
            kv: Vec::new(),
        };
        let corners = feasible_corners(&envelope, 1, &geometry).unwrap();
        assert_eq!(corners.len(), 2);
        for dims in &corners {
            assert_eq!(dims.total, dims.past + dims.seq);
            assert!(dims.past <= 17);
            assert!(dims.seq <= 13);
            assert!(dims.total <= 18);
        }
        assert!(corners.iter().any(|dims| dims.seq == 13));
        assert!(corners.iter().any(|dims| dims.past == 17));
    }
}
