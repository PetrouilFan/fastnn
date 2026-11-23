//! ONNX graph → fn3 intermediate representation: op mapping, attribute
//! resolution and static validation.
//!
//! # The one rule
//!
//! Each ONNX node maps to exactly one fn3 [`OpId`], with two documented
//! exceptions that are compositions, not approximations:
//!
//! 1. `Gemm` becomes [`OpId::MatMul`] followed by [`OpId::Add`] (bias). `Gemm`
//!    has no fn3 id because it is `alpha * A' * B' + beta * C`; this importer
//!    accepts only the identity form (`alpha = beta = 1`, `transA = transB = 0`)
//!    and reports anything else instead of dropping the scaling.
//! 2. `ConstantOfShape` carries its fill value in an attribute that
//!    [`crate::inference::backend::OpParams`] cannot express, so the value is lifted
//!    into a scalar constant input, which is the shape the CPU kernel
//!    documents. The constant is registered as a weight
//!    ([`WeightSource::AttributeConstant`]) so the planner can reference it.
//!
//! Everything else is one-to-one or an error naming the node, the ONNX op type
//! and (where an issue exists) the owner. There is no fallback that returns an
//! input.
//!
//! # Static validation
//!
//! The importer infers a dtype for every value and a shape where it can:
//!
//! - dtypes are always statically known and always checked against the ONNX
//!   operator's own rule (an `Add` with an `f32` and an `i64` operand is an
//!   error, not a cast);
//! - shapes are tracked symbolically, including the integer payloads that
//!   shape arithmetic flows through (`Shape` → `Gather` → `Concat` → `Reshape`),
//!   so `[batch, seq, hidden]` survives the exporter's `Reshape` targets;
//! - a dim that cannot be proven becomes [`Dim::Unknown`] *with a reason*; the
//!   importer never guesses an extent;
//! - a declared `value_info` shape that contradicts a fully static inferred
//!   shape is reported as a warning, because exporter `dim_param` naming is
//!   advisory (the pinned export names the mask axis
//!   `"past_sequence_length + 1"` for a value that is `past + seq`).
//!
//! Declared shapes of *graph inputs and outputs* are the interface and are
//! authoritative; a rank contradiction there is an error.
//!
//! # Determinism
//!
//! Value ids are assigned in file order: graph inputs, then initializers, then
//! node outputs in node order (and, for a composition, intermediate first).
//! Nothing iterates a hash map. The same bytes therefore produce the same value
//! table, the same node order and the same weight layout.

use std::collections::{BTreeMap, BTreeSet};

use crate::inference::backend::{
    AutoPad, CoordinateTransformationMode, NearestMode, OpParams, PadMode, ResizeMode,
};
use crate::inference::dtype::DType;
use crate::inference::model::{KvCacheConfig, KvCacheLayout};
use crate::inference::ops::OpId;
use crate::inference::shape::{Shape, Symbol};

use super::budget::{
    model_bytes_within_cap, MAX_MODEL_BYTES, MAX_NODES, MAX_RANK, MAX_VALUES, MAX_WEIGHTS,
};
use super::error::{Diagnostics, ImportProblemKind, NodeRef};
use super::proto;
use crate::inference::model::weights::{WeightId, WeightSet, WeightSource, WeightTensor};

/// Smallest ONNX IR version this importer accepts.
///
/// IR 4 is the first version where an initializer is no longer required to also
/// appear as a graph input.
pub const MIN_IR_VERSION: i64 = 4;

/// Largest ONNX IR version this importer accepts.
///
/// The reader ignores unknown fields, so a newer model decodes; the version is
/// still pinned so a semantic change in the message set is a deliberate bump
/// rather than a silent reinterpretation. ONNX IR 10 is current at the time of
/// writing.
pub const MAX_IR_VERSION: i64 = 10;

/// Smallest opset this importer implements.
///
/// Opset 12 is the last revision where `Split` carries its sizes in the `split`
/// attribute; opset 13 moved them to an input. Both forms are implemented.
/// `Squeeze`/`Unsqueeze`/`Slice` are read in their opset-13 input form (Slice
/// moved to inputs in opset 10).
pub const MIN_OPSET: i64 = 12;

/// Largest opset this importer implements.
///
/// Opset 18 moved `ReduceMean` axes from an attribute to an input, which this
/// mapping would silently ignore. The pinned model is opset 14.
pub const MAX_OPSET: i64 = 22;

/// Position of one value in [`GraphIr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(u32);

impl ValueId {
    /// Construct from a zero-based position.
    pub const fn new(index: u32) -> ValueId {
        ValueId(index)
    }

    /// Zero-based position in [`GraphIr::values`].
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "value {}", self.0)
    }
}

/// One axis of a value's shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    /// A statically known extent.
    Known(u64),
    /// One of the decoder symbols of [`crate::inference::shape`].
    Symbol(Symbol),
    /// An extent this importer cannot prove, with the reason.
    Unknown(String),
}

impl Dim {
    /// A known extent.
    pub fn known(extent: u64) -> Dim {
        Dim::Known(extent)
    }

    /// An unproven extent, with a reason that a diagnostic can quote.
    pub fn unknown(reason: impl Into<String>) -> Dim {
        Dim::Unknown(reason.into())
    }

    /// The extent when it is statically known.
    pub fn as_known(&self) -> Option<u64> {
        match self {
            Dim::Known(extent) => Some(*extent),
            _ => None,
        }
    }

    /// `true` when the extent is statically known.
    pub fn is_known(&self) -> bool {
        matches!(self, Dim::Known(_))
    }

    /// `past + seq` normalizes to `total`, the only sum the decoder contract
    /// names. Everything else that cannot be simplified becomes unknown.
    fn add(left: &Dim, right: &Dim) -> Dim {
        match (left, right) {
            (Dim::Known(a), Dim::Known(b)) => match a.checked_add(*b) {
                Some(sum) => Dim::Known(sum),
                None => Dim::unknown(format!("{a} + {b} exceeds u64")),
            },
            (Dim::Known(0), other) | (other, Dim::Known(0)) => other.clone(),
            (Dim::Symbol(Symbol::Past), Dim::Symbol(Symbol::Seq))
            | (Dim::Symbol(Symbol::Seq), Dim::Symbol(Symbol::Past)) => Dim::Symbol(Symbol::Total),
            (left, right) => Dim::unknown(format!("{} + {}", left, right)),
        }
    }

    /// `left - right`, using the decoder's own relations.
    ///
    /// `total - seq = past` and `total - past = seq` are the only symbolic
    /// subtractions the decoder contract names; everything else that cannot be
    /// reduced to a known difference becomes unknown.
    fn sub(left: &Dim, right: &Dim) -> Dim {
        match (left, right) {
            (Dim::Known(a), Dim::Known(b)) => match a.checked_sub(*b) {
                Some(difference) => Dim::Known(difference),
                None => Dim::unknown(format!("{a} - {b} is negative")),
            },
            (Dim::Symbol(Symbol::Total), Dim::Symbol(Symbol::Seq)) => Dim::Symbol(Symbol::Past),
            (Dim::Symbol(Symbol::Total), Dim::Symbol(Symbol::Past)) => Dim::Symbol(Symbol::Seq),
            (Dim::Symbol(a), Dim::Symbol(b)) if a == b => Dim::Known(0),
            (left, Dim::Known(0)) => left.clone(),
            (left, right) => Dim::unknown(format!("{} - {}", left, right)),
        }
    }
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Known(extent) => write!(f, "{extent}"),
            Dim::Symbol(symbol) => write!(f, "{symbol}"),
            Dim::Unknown(reason) => write!(f, "?({reason})"),
        }
    }
}

/// A value's shape: known rank with per-axis [`Dim`]s, or unknown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueShape {
    /// The rank could not be determined (for example the target of a `Reshape`
    /// that is computed at runtime).
    Unknown,
    /// Known rank; each axis is known, symbolic or unproven.
    Dims(Vec<Dim>),
}

impl ValueShape {
    /// Build from fully known extents.
    pub fn from_dims(dims: impl IntoIterator<Item = u64>) -> ValueShape {
        ValueShape::Dims(dims.into_iter().map(Dim::Known).collect())
    }

    /// Axes, or an empty slice when the rank is unknown.
    pub fn dims(&self) -> &[Dim] {
        match self {
            ValueShape::Unknown => &[],
            ValueShape::Dims(dims) => dims,
        }
    }

    /// Rank, when it is known.
    pub fn rank(&self) -> Option<usize> {
        match self {
            ValueShape::Unknown => None,
            ValueShape::Dims(dims) => Some(dims.len()),
        }
    }

    /// `true` when every axis is a known extent.
    pub fn is_static(&self) -> bool {
        match self {
            ValueShape::Unknown => false,
            ValueShape::Dims(dims) => dims.iter().all(Dim::is_known),
        }
    }

    /// The [`Shape`] when every axis is known.
    pub fn static_shape(&self) -> Option<Shape> {
        match self {
            ValueShape::Unknown => None,
            ValueShape::Dims(dims) => {
                let mut extents = Vec::with_capacity(dims.len());
                for dim in dims {
                    extents.push(dim.as_known()?);
                }
                Some(Shape::from_dims(extents))
            }
        }
    }

    /// Element count when every axis is known.
    pub fn numel(&self) -> Option<u64> {
        self.static_shape()?.numel().ok()
    }

    /// Render as `[batch, seq, 128]`, or `[?]` when the rank is unknown.
    pub fn render(&self) -> String {
        match self {
            ValueShape::Unknown => "[?]".to_string(),
            ValueShape::Dims(dims) => {
                let rendered: Vec<String> = dims.iter().map(Dim::to_string).collect();
                format!("[{}]", rendered.join(", "))
            }
        }
    }
}

impl std::fmt::Display for ValueShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render())
    }
}

/// What produced a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueOrigin {
    /// `GraphProto.input`.
    GraphInput,
    /// A `GraphProto.initializer` or a lifted attribute constant.
    Weight(WeightId),
    /// Output of [`GraphIr::nodes`] at this index.
    Node(u32),
}

/// One value in the graph: a name, a dtype, a shape and its producer.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueInfo {
    /// Name exactly as the model spelled it.
    pub name: String,
    /// Inferred element type.
    pub dtype: DType,
    /// Inferred shape.
    pub shape: ValueShape,
    /// What produced the value.
    pub origin: ValueOrigin,
}

/// One fn3 operation, in execution order.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphNode {
    /// Index of the ONNX node this operation came from. A `Gemm` produces two
    /// operations with the same index.
    pub onnx_index: u32,
    /// ONNX node name, empty when the exporter left it blank.
    pub name: String,
    /// ONNX operator type, exactly as the file spelled it (`"Gemm"`).
    pub onnx_op: String,
    /// The fn3 operation this node executes.
    pub op: OpId,
    /// Value handles read by the op, in operand order. The count matches the
    /// kernel's arity: two for `Add`, five for an opset-14 `Slice`, two for a
    /// `ConstantOfShape` whose fill value was lifted.
    pub inputs: Vec<ValueId>,
    /// Value handles written by the op. One for every mapped operator except
    /// `Split`, whose output count is the node's output count.
    pub outputs: Vec<ValueId>,
    /// Attributes, in the encoding the CPU reference reads (`axes` carries
    /// `Transpose.perm`, `Softmax.axis`, `ReduceMean.axes`, `Concat.axis`,
    /// `Gather.axis`, `Split.axis`, `Trilu.upper`, ...).
    pub params: OpParams,
}

impl GraphNode {
    /// The first output, which is the node's result for every op but `Split`.
    pub fn primary_output(&self) -> Option<ValueId> {
        self.outputs.first().copied()
    }
}

/// Values, nodes and the graph interface, in deterministic order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GraphIr {
    values: Vec<ValueInfo>,
    nodes: Vec<GraphNode>,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
}

impl GraphIr {
    /// Every value, indexed by [`ValueId`].
    pub fn values(&self) -> &[ValueInfo] {
        &self.values
    }

    /// One value.
    pub fn value(&self, id: ValueId) -> &ValueInfo {
        &self.values[id.index()]
    }

    /// Look a value up by exact name.
    pub fn lookup(&self, name: &str) -> Option<ValueId> {
        self.values
            .iter()
            .position(|value| value.name == name)
            .map(|index| ValueId(index as u32))
    }

    /// A value by exact name, with its info.
    pub fn find(&self, name: &str) -> Option<(ValueId, &ValueInfo)> {
        self.lookup(name).map(|id| (id, self.value(id)))
    }

    /// Operations in execution order.
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Graph inputs, in file order.
    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    /// Graph outputs, in file order.
    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    /// The node that produced `id`, if a node did.
    pub fn producer(&self, id: ValueId) -> Option<&GraphNode> {
        match self.value(id).origin {
            ValueOrigin::Node(index) => self.nodes.get(index as usize),
            _ => None,
        }
    }

    /// Number of values.
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// Build a graph from parts a caller already validated.
    ///
    /// The importer builds a `GraphIr` while decoding an ONNX file. A reader of
    /// a serialized model ([`crate::inference::ir::Graph`], which validates the
    /// same invariants) needs the identical container without an
    /// ONNX file present, and private fields make that otherwise impossible.
    /// This is the additive seam those callers use; it performs no validation,
    /// so the caller owns the invariants listed on
    /// [`crate::inference::ir::Graph`].
    pub fn from_parts(
        values: Vec<ValueInfo>,
        nodes: Vec<GraphNode>,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    ) -> GraphIr {
        GraphIr {
            values,
            nodes,
            inputs,
            outputs,
        }
    }
}

/// The `OperatorSetIdProto` the model declares for the default domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpsetImport {
    /// Domain; always empty for the import this build accepts.
    pub domain: String,
    /// Opset version.
    pub version: i64,
}

/// How positions are produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionEmbedding {
    /// A learned table gathered with `position_ids` (GPT-2).
    Learned {
        /// The `[max_positions, hidden]` table.
        weight: WeightId,
        /// Number of rows in the table.
        max_positions: u32,
    },
}

/// Per-layer key/value cache interface of the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvInterface {
    /// `past_key_values.{layer}.key`
    pub past_key: ValueId,
    /// `past_key_values.{layer}.value`
    pub past_value: ValueId,
    /// `present.{layer}.key`
    pub present_key: ValueId,
    /// `present.{layer}.value`
    pub present_value: ValueId,
}

/// Architecture geometry detected from the graph and validated against it.
///
/// This is the contract a planner and a session need: how many layers exist,
/// how wide they are, how the head split works, how large the vocabulary is and
/// where the KV interface lives. A graph that does not match is rejected with an
/// [`ImportProblemKind::Architecture`] problem naming the missing or
/// contradictory piece.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderGeometry {
    /// Transformer blocks.
    pub layers: u32,
    /// Model width.
    pub hidden_size: u32,
    /// MLP intermediate width.
    pub mlp_hidden_size: u32,
    /// Query heads (`hidden_size / head_dim`).
    pub query_heads: u32,
    /// Key/value heads (grouped-query attention when smaller than queries).
    pub kv_heads: u32,
    /// Per-head width.
    pub head_dim: u32,
    /// Vocabulary size, from the token embedding table.
    pub vocab_size: u32,
    /// Learned or computed positions.
    pub positions: PositionEmbedding,
    /// LayerNorm epsilon when it could be detected structurally.
    pub layer_norm_epsilon: Option<f32>,
    /// Token embedding table (`[vocab_size, hidden_size]`).
    pub token_embedding: WeightId,
    /// Final `logits` output.
    pub logits: ValueId,
    /// Per-layer KV interface, layer order.
    pub kv: Vec<KvInterface>,
}

impl DecoderGeometry {
    /// KV cache configuration for this model.
    ///
    /// `capacity_tokens` is the allocation bound and `batch` the batch size; a
    /// session decides both, so they are arguments rather than detected values.
    pub fn kv_cache_config(&self, capacity_tokens: u64, batch: u64) -> KvCacheConfig {
        KvCacheConfig {
            layers: u64::from(self.layers),
            capacity_tokens,
            batch,
            n_kv_heads: u64::from(self.kv_heads),
            head_dim: u64::from(self.head_dim),
            dtype: DType::F32,
            layout: KvCacheLayout::BatchHeadsCapacityHeadDim,
        }
    }
}

/// A declared shape that contradicts a fully static inferred shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportWarning {
    /// Node that produced the value, when a node did.
    pub node: Option<NodeRef>,
    /// What the file declared and what was inferred.
    pub kind: ImportWarningKind,
}

/// Warning payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportWarningKind {
    /// `value_info` (or a graph output) declares one static shape and the
    /// mapping infers another. The mapping's shape is used.
    DeclaredShape {
        /// Value name.
        value: String,
        /// Shape the file declares.
        declared: String,
        /// Shape the mapping infers.
        inferred: String,
    },
}

impl std::fmt::Display for ImportWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.node, &self.kind) {
            (
                Some(node),
                ImportWarningKind::DeclaredShape {
                    value,
                    declared,
                    inferred,
                },
            ) => write!(
                f,
                "{node}: value `{value}` declares shape {declared} but inference gives {inferred}; \
                 the inferred shape is used"
            ),
            (
                None,
                ImportWarningKind::DeclaredShape {
                    value,
                    declared,
                    inferred,
                },
            ) => write!(
                f,
                "value `{value}` declares shape {declared} but inference gives {inferred}; the \
                 inferred shape is used"
            ),
        }
    }
}

/// Counts and censuses of one import, for tests, reports and the CLI.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ImportSummary {
    /// Nodes in `GraphProto.node`.
    pub onnx_nodes: usize,
    /// Operations emitted into [`GraphIr::nodes`]. Larger than `onnx_nodes` when
    /// a node maps to a composition.
    pub fn3_nodes: usize,
    /// Values in [`GraphIr::values`].
    pub values: usize,
    /// `GraphProto.initializer` entries.
    pub initializers: usize,
    /// Values beyond the initializers that were lifted into the weight table:
    /// `ConstantOfShape` fill values and folded `Constant` nodes.
    pub attribute_constants: usize,
    /// Total weight payload bytes.
    pub weight_bytes: u64,
    /// ONNX operator census, sorted by operator name.
    pub onnx_ops: Vec<(String, usize)>,
    /// fn3 operator census, sorted by [`OpId::code`].
    pub ops: Vec<(OpId, usize)>,
    /// Value dtype census, sorted by [`DType::code`].
    pub dtypes: Vec<(DType, usize)>,
}

impl ImportSummary {
    /// Number of times `op` appears.
    pub fn count(&self, op: OpId) -> usize {
        self.ops
            .iter()
            .find(|(candidate, _)| *candidate == op)
            .map(|(_, count)| *count)
            .unwrap_or(0)
    }

    /// Number of times the ONNX operator `op_type` appears.
    pub fn count_onnx(&self, op_type: &str) -> usize {
        self.onnx_ops
            .iter()
            .find(|(candidate, _)| candidate == op_type)
            .map(|(_, count)| *count)
            .unwrap_or(0)
    }
}

/// The provenance an assembled model carries.
///
/// [`ModelImport`] normally gets these five values from the ONNX header. A
/// caller that rebuilds a model from a serialized artifact (where the metadata
/// is a section of the file) has them already and supplies them here instead of
/// re-reading the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProvenance {
    /// ONNX IR version the graph was imported from.
    pub ir_version: i64,
    /// Opset import for the default domain.
    pub opset: OpsetImport,
    /// Exporter name.
    pub producer: String,
    /// Exporter version.
    pub producer_version: String,
    /// Source graph name.
    pub graph_name: String,
}

/// The graph-level result of one import, before architecture detection.
///
/// [`Builder::decode`] produces it, [`Builder::run`] adds decoder geometry and
/// [`super::import_onnx_graph`] adds the typed IR. Keeping the two stages
/// separate is what lets a CNN — which has no decoder interface — import at
/// all, without relaxing the decoder contract for the models that do.
pub(crate) struct GraphParts {
    /// ONNX IR version of the source.
    pub(crate) ir_version: i64,
    /// Exporter name.
    pub(crate) producer: String,
    /// Exporter version.
    pub(crate) producer_version: String,
    /// Default-domain opset.
    pub(crate) opset: OpsetImport,
    /// Source graph name.
    pub(crate) graph_name: String,
    /// Values, nodes and the graph interface.
    pub(crate) graph: GraphIr,
    /// Weight table, in file order.
    pub(crate) weights: WeightSet,
    /// Counts and censuses.
    pub(crate) summary: ImportSummary,
    /// Declared-shape warnings.
    pub(crate) warnings: Vec<ImportWarning>,
    /// Per-node attributes the typed IR reads, when they differ from the
    /// [`GraphNode::params`] the artifact format serializes.
    pub(crate) typed_params: Vec<crate::inference::backend::OpParams>,
}

/// A validated ONNX model, ready for a planner.
///
/// Immutable after construction. Nothing here borrows the model file: names,
/// shapes and weight bytes are owned, so a caller can drop the 28 MB buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelImport {
    ir_version: i64,
    producer: String,
    producer_version: String,
    opset: OpsetImport,
    graph_name: String,
    graph: GraphIr,
    weights: WeightSet,
    geometry: DecoderGeometry,
    summary: ImportSummary,
    warnings: Vec<ImportWarning>,
}

impl ModelImport {
    /// Assemble a model from parts that were not decoded from an ONNX file.
    ///
    /// The artifact reader is the caller: it has already validated the graph,
    /// the weights and the geometry (see
    /// [`crate::inference::model::format::PreparedArtifact::validate`]), and the
    /// metadata section supplies [`ModelProvenance`]. Nothing here re-detects
    /// the architecture, so a caller that has no validated geometry must use
    /// the ONNX path instead.
    ///
    /// `summary` is recomputed from `graph` and `weights`. Every field is
    /// derivable because a graph node keeps its ONNX provenance and a weight
    /// keeps its source, so an artifact built from an import reproduces that
    /// import's summary exactly rather than carrying a partial one.
    pub fn from_parts(
        graph: GraphIr,
        weights: WeightSet,
        geometry: DecoderGeometry,
        provenance: ModelProvenance,
    ) -> ModelImport {
        let summary = summarize(&graph, &weights, None, 0);
        ModelImport {
            ir_version: provenance.ir_version,
            producer: provenance.producer,
            producer_version: provenance.producer_version,
            opset: provenance.opset,
            graph_name: provenance.graph_name,
            graph,
            weights,
            geometry,
            summary,
            warnings: Vec::new(),
        }
    }

    /// ONNX IR version of the file.
    pub fn ir_version(&self) -> i64 {
        self.ir_version
    }

    /// `ModelProto.producer_name`.
    pub fn producer(&self) -> &str {
        &self.producer
    }

    /// `ModelProto.producer_version`.
    pub fn producer_version(&self) -> &str {
        &self.producer_version
    }

    /// Opset import for the default domain.
    pub fn opset(&self) -> &OpsetImport {
        &self.opset
    }

    /// `GraphProto.name`.
    pub fn graph_name(&self) -> &str {
        &self.graph_name
    }

    /// The mapped graph.
    pub fn graph(&self) -> &GraphIr {
        &self.graph
    }

    /// Typed weights.
    pub fn weights(&self) -> &WeightSet {
        &self.weights
    }

    /// Detected architecture geometry.
    pub fn geometry(&self) -> &DecoderGeometry {
        &self.geometry
    }

    /// Counts and censuses.
    pub fn summary(&self) -> &ImportSummary {
        &self.summary
    }

    /// Non-fatal findings, in file order.
    pub fn warnings(&self) -> &[ImportWarning] {
        &self.warnings
    }

    /// Split into the two things a planner consumes.
    pub fn into_parts(self) -> (GraphIr, WeightSet) {
        (self.graph, self.weights)
    }

    /// Copy every weight into host tensors on the CPU.
    pub fn to_model_weights(
        &self,
    ) -> Result<crate::inference::model::ModelWeights, crate::inference::error::Fn3Error> {
        self.weights.to_model_weights()
    }
}

// ---------------------------------------------------------------------------
// Abstract values used by static inference
// ---------------------------------------------------------------------------

/// One statically tracked element of a value sequence.
#[derive(Debug, Clone, PartialEq)]
enum Atom {
    /// A known integer.
    Int(i64),
    /// A known boolean.
    Bool(bool),
    /// A known single-precision float.
    Float(f32),
    /// A symbolic extent used as an integer.
    Dim(Dim),
    /// A value that is not statically known.
    Unknown,
}

impl Atom {
    fn as_i64(&self) -> Option<i64> {
        match self {
            Atom::Int(value) => Some(*value),
            Atom::Dim(Dim::Known(extent)) => i64::try_from(*extent).ok(),
            _ => None,
        }
    }

    fn as_f32(&self) -> Option<f32> {
        match self {
            Atom::Float(value) => Some(*value),
            Atom::Int(value) => Some(*value as f32),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Atom::Bool(value) => Some(*value),
            Atom::Int(value) => Some(*value != 0),
            Atom::Dim(Dim::Known(extent)) => Some(*extent != 0),
            _ => None,
        }
    }

    fn as_dim(&self) -> Dim {
        atom_as_dim(self)
            .unwrap_or_else(|| Dim::unknown(format!("{} is not an extent", self.render())))
    }

    fn render(&self) -> String {
        match self {
            Atom::Int(value) => value.to_string(),
            Atom::Bool(value) => value.to_string(),
            Atom::Float(value) => value.to_string(),
            Atom::Dim(dim) => dim.to_string(),
            Atom::Unknown => "?".to_string(),
        }
    }

    /// `true` when both atoms are provably the same value.
    fn same(&self, other: &Atom) -> Option<bool> {
        match (self, other) {
            (Atom::Int(a), Atom::Int(b)) => Some(a == b),
            (Atom::Bool(a), Atom::Bool(b)) => Some(a == b),
            (Atom::Dim(Dim::Known(a)), Atom::Int(b)) => Some(i64::try_from(*a).ok()? == *b),
            (Atom::Int(a), Atom::Dim(Dim::Known(b))) => Some(*a == i64::try_from(*b).ok()?),
            (Atom::Dim(Dim::Symbol(a)), Atom::Dim(Dim::Symbol(b))) => Some(a == b),
            // A symbolic extent is a non-negative runtime value, so it can
            // never equal a negative constant.
            (Atom::Dim(_), Atom::Int(value)) | (Atom::Int(value), Atom::Dim(_)) => {
                Some(*value >= 0)
            }
            _ => None,
        }
    }
}

/// An atom as a non-negative extent, when it is one.
fn atom_as_dim(atom: &Atom) -> Option<Dim> {
    match atom {
        Atom::Int(value) if *value >= 0 => Some(Dim::Known(*value as u64)),
        Atom::Dim(dim) if !matches!(dim, Dim::Unknown(_)) => Some(dim.clone()),
        _ => None,
    }
}

/// `left + right` on tracked elements.
fn atom_add(left: &Atom, right: &Atom) -> Atom {
    match (left.as_i64(), right.as_i64()) {
        (Some(a), Some(b)) => Atom::Int(a.wrapping_add(b)),
        _ => match (atom_as_dim(left), atom_as_dim(right)) {
            (Some(a), Some(b)) => Atom::Dim(Dim::add(&a, &b)),
            _ => Atom::Unknown,
        },
    }
}

/// `left - right` on tracked elements.
fn atom_sub(left: &Atom, right: &Atom) -> Atom {
    match (left.as_i64(), right.as_i64()) {
        (Some(a), Some(b)) => Atom::Int(a.wrapping_sub(b)),
        _ => match (atom_as_dim(left), atom_as_dim(right)) {
            (Some(a), Some(b)) => Atom::Dim(Dim::sub(&a, &b)),
            _ => Atom::Unknown,
        },
    }
}

/// `left * right` on tracked elements.
fn atom_mul(left: &Atom, right: &Atom) -> Atom {
    match (left.as_i64(), right.as_i64()) {
        (Some(a), Some(b)) => Atom::Int(a.wrapping_mul(b)),
        _ => Atom::Unknown,
    }
}

/// `left / right` on tracked elements, with integer truncation.
///
/// ONNX `Div` on two integers truncates toward zero, which is Rust's `/`; a
/// float operand or a zero divisor stays [`Atom::Unknown`] rather than
/// guessing. The pinned YOLOv8 export computes a Slice bound this way
/// (`(dim + 1) / 2`), so folding it is what keeps that Slice statically shaped.
fn atom_div(left: &Atom, right: &Atom) -> Atom {
    match (left.as_i64(), right.as_i64()) {
        (Some(_), Some(0)) => Atom::Unknown,
        (Some(a), Some(b)) => Atom::Int(a / b),
        _ => Atom::Unknown,
    }
}

/// The element sequence of a value whose contents the importer tracks.
#[derive(Debug, Clone, PartialEq)]
struct AtomSeries {
    /// Number of elements; symbolic when it comes from a symbolic axis.
    len: Dim,
    /// Element values when every one is tracked.
    items: Option<Vec<Atom>>,
}

impl AtomSeries {
    fn from_atoms(items: Vec<Atom>) -> AtomSeries {
        AtomSeries {
            len: Dim::Known(items.len() as u64),
            items: Some(items),
        }
    }

    fn scalar(atom: Atom) -> AtomSeries {
        AtomSeries::from_atoms(vec![atom])
    }

    fn from_dims(dims: &[Dim]) -> AtomSeries {
        AtomSeries::from_atoms(dims.iter().cloned().map(Atom::Dim).collect())
    }

    /// Length as a `usize`, when it is statically known.
    fn len_usize(&self) -> Option<usize> {
        self.len
            .as_known()
            .and_then(|len| usize::try_from(len).ok())
    }

    /// The single element of a length-one series.
    fn only(&self) -> Option<&Atom> {
        match &self.items {
            Some(items) if items.len() == 1 => items.first(),
            _ => None,
        }
    }

    /// Index an element, counting negative indices from the end.
    fn item(&self, index: i64) -> Option<Atom> {
        let items = self.items.as_ref()?;
        let len = items.len() as i64;
        let index = if index < 0 { len + index } else { index };
        if index < 0 {
            return None;
        }
        items.get(index as usize).cloned()
    }

    /// Concatenate series end to end.
    fn concat(parts: &[AtomSeries]) -> AtomSeries {
        let mut len = Dim::Known(0);
        let mut items = Vec::new();
        let mut all_items = true;
        for part in parts {
            len = Dim::add(&len, &part.len);
            match &part.items {
                Some(part_items) => items.extend(part_items.iter().cloned()),
                None => all_items = false,
            }
        }
        AtomSeries {
            len,
            items: if all_items { Some(items) } else { None },
        }
    }

    /// Elementwise combine two sequences with NumPy broadcasting.
    fn zip(&self, other: &AtomSeries, combine: impl Fn(&Atom, &Atom) -> Atom) -> AtomSeries {
        let (len, a_items, b_items) = broadcast_series(self, other);
        let items = match (a_items, b_items) {
            (Some(a), Some(b)) => Some(
                a.iter()
                    .zip(b.iter())
                    .map(|(left, right)| combine(left, right))
                    .collect(),
            ),
            _ => None,
        };
        AtomSeries { len, items }
    }
}

/// Right-align two sequences and return the broadcast length plus per-element
/// views (a length-one side is repeated).
///
/// A broadcast repeat is only materialised for a small length: a declared extent
/// of a billion must not turn into a billion tracked atoms, so a larger repeat
/// keeps the length and drops the per-element detail.
fn broadcast_series(
    left: &AtomSeries,
    right: &AtomSeries,
) -> (Dim, Option<Vec<Atom>>, Option<Vec<Atom>>) {
    if left.len == Dim::Known(1) {
        let len = right.len.clone();
        let repeated = repeat_single(left.items.as_deref(), right.len_usize());
        return (len, repeated, right.items.clone());
    }
    if right.len == Dim::Known(1) {
        let len = left.len.clone();
        let repeated = repeat_single(right.items.as_deref(), left.len_usize());
        return (len, left.items.clone(), repeated);
    }
    if left.len == right.len {
        return (left.len.clone(), left.items.clone(), right.items.clone());
    }
    (
        Dim::unknown(format!("{} vs {}", left.len, right.len)),
        None,
        None,
    )
}

/// Largest element sequence the importer materialises.
///
/// The constant lives in [`super::budget`] with every other parser budget; it is
/// re-exported under the old name so the plan helpers below read the same.
const MAX_TRACKED_ELEMENTS: usize = super::budget::MAX_TRACKED_ELEMENTS;

/// Repeat a single-element sequence to `len`, when that is small enough to be
/// worth tracking.
fn repeat_single(items: Option<&[Atom]>, len: Option<usize>) -> Option<Vec<Atom>> {
    let items = items?;
    let len = len?;
    let atom = items.first()?;
    if len > MAX_TRACKED_ELEMENTS {
        return None;
    }
    Some(vec![atom.clone(); len])
}

/// What the importer knows about one value during inference.
#[derive(Debug, Clone, PartialEq)]
enum Abstract {
    /// A tensor whose dtype and shape are tracked.
    Tensor { dtype: DType, shape: ValueShape },
    /// A tensor whose element values are tracked as well (shape payloads and
    /// mask arithmetic).
    Values {
        dtype: DType,
        shape: ValueShape,
        series: AtomSeries,
    },
    /// A value produced by a node that already reported a problem, so its
    /// consumers stay quiet instead of cascading.
    Poisoned,
}

impl Abstract {
    fn tensor(dtype: DType, shape: ValueShape) -> Abstract {
        Abstract::Tensor { dtype, shape }
    }

    fn values(dtype: DType, shape: ValueShape, series: AtomSeries) -> Abstract {
        Abstract::Values {
            dtype,
            shape,
            series,
        }
    }
}

// ---------------------------------------------------------------------------
// Planned operations
// ---------------------------------------------------------------------------

/// Where one operand of a planned operation comes from.
#[derive(Debug, Clone, PartialEq)]
enum Operand {
    /// The node's resolved input at this position.
    Input(usize),
    /// The first output of an earlier planned operation of the same node.
    Intermediate(usize),
    /// A constant the mapping synthesizes, registered as a weight.
    NewConstant(Box<WeightTensor>),
}

/// One value a planned operation writes.
#[derive(Debug, Clone, PartialEq)]
struct PlannedValue {
    dtype: DType,
    shape: ValueShape,
    series: Option<AtomSeries>,
}

impl PlannedValue {
    fn tensor(dtype: DType, shape: ValueShape) -> PlannedValue {
        PlannedValue {
            dtype,
            shape,
            series: None,
        }
    }

    fn values(dtype: DType, shape: ValueShape, series: AtomSeries) -> PlannedValue {
        PlannedValue {
            dtype,
            shape,
            series: Some(series),
        }
    }
}

/// One fn3 operation a node maps to.
#[derive(Debug, Clone, PartialEq)]
struct PlannedOp {
    op: OpId,
    operands: Vec<Operand>,
    /// Attributes the imported [`GraphNode`] carries. They must survive the
    /// 3.0 artifact format, which stores `axes`, `keep_dims` and `epsilon`
    /// only, so a field the format does not carry goes in `typed` instead.
    params: OpParams,
    /// Attributes the typed IR reads when they differ from `params`.
    ///
    /// `Reshape`, `Slice` and `Split` need their constant operands to be
    /// visible to [`crate::inference::ir::infer_shapes`], which reads
    /// [`OpParams`] and never a tensor payload. Putting those bounds in
    /// `params` would also change what the decoder's artifact round trip
    /// stores, so they ride here instead and only the typed graph sees them.
    typed: Option<OpParams>,
    outputs: Vec<PlannedValue>,
}

impl PlannedOp {
    fn single(
        op: OpId,
        operands: Vec<Operand>,
        params: OpParams,
        output: PlannedValue,
    ) -> Vec<PlannedOp> {
        vec![PlannedOp {
            op,
            operands,
            params,
            typed: None,
            outputs: vec![output],
        }]
    }
}

/// Why a node could not be planned.
type PlanError = ImportProblemKind;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// ONNX mnemonics for a `TensorProto.DataType` code.
fn onnx_dtype_name(code: i32) -> &'static str {
    match code {
        0 => "UNDEFINED",
        1 => "FLOAT",
        2 => "UINT8",
        3 => "INT8",
        4 => "UINT16",
        5 => "INT16",
        6 => "INT32",
        7 => "INT64",
        8 => "STRING",
        9 => "BOOL",
        10 => "FLOAT16",
        11 => "DOUBLE",
        12 => "UINT32",
        13 => "UINT64",
        14 => "COMPLEX64",
        15 => "COMPLEX128",
        16 => "BFLOAT16",
        17 => "FLOAT8E4M3FN",
        18 => "FLOAT8E4M3FNUZ",
        19 => "FLOAT8E5M2",
        20 => "FLOAT8E5M2FNUZ",
        21 => "UINT4",
        22 => "INT4",
        23 => "FLOAT4E2M1",
        _ => "UNKNOWN",
    }
}

/// Map an ONNX dtype code onto the 3.0 dtype set.
fn onnx_dtype(code: i32, carrier: impl Into<String>) -> Result<DType, PlanError> {
    match code {
        1 => Ok(DType::F32),
        6 => Ok(DType::I32),
        7 => Ok(DType::I64),
        9 => Ok(DType::Bool),
        other => Err(ImportProblemKind::UnsupportedDType {
            code: other,
            onnx_name: onnx_dtype_name(other),
            carrier: carrier.into(),
        }),
    }
}

/// A missing required attribute.
fn missing_attribute(name: &str) -> PlanError {
    ImportProblemKind::UnsupportedAttribute {
        attribute: name.to_string(),
        detail: "the operator requires this attribute".to_string(),
    }
}

/// A shape contradiction with a readable rule.
fn shape_problem(detail: impl Into<String>) -> PlanError {
    ImportProblemKind::ShapeMismatch {
        detail: detail.into(),
    }
}

/// A dtype disagreement with a readable rule.
fn dtype_problem(detail: impl Into<String>) -> PlanError {
    ImportProblemKind::DTypeMismatch {
        detail: detail.into(),
    }
}

/// Right-aligned NumPy broadcast of two shapes.
fn broadcast_shapes(left: &ValueShape, right: &ValueShape) -> Result<ValueShape, PlanError> {
    let (ValueShape::Dims(left_dims), ValueShape::Dims(right_dims)) = (left, right) else {
        return Ok(ValueShape::Unknown);
    };
    let rank = left_dims.len().max(right_dims.len());
    let mut dims = Vec::with_capacity(rank);
    for position in 0..rank {
        let left_dim = offset_dim(left_dims, rank, position);
        let right_dim = offset_dim(right_dims, rank, position);
        dims.push(broadcast_dim(&left_dim, &right_dim, position)?);
    }
    Ok(ValueShape::Dims(dims))
}

/// The axis at `position` in a right-aligned view of `dims` (extent 1 when the
/// axis does not exist).
fn offset_dim(dims: &[Dim], rank: usize, position: usize) -> Dim {
    let start = rank - dims.len();
    if position < start {
        Dim::Known(1)
    } else {
        dims[position - start].clone()
    }
}

/// Broadcast one pair of axes.
fn broadcast_dim(left: &Dim, right: &Dim, axis: usize) -> Result<Dim, PlanError> {
    let mismatch = || {
        shape_problem(format!(
            "axis {axis} broadcasts {left} against {right}, and neither is 1"
        ))
    };
    match (left, right) {
        (Dim::Known(a), Dim::Known(b)) => {
            if a == b {
                Ok(Dim::Known(*a))
            } else if *a == 1 {
                Ok(Dim::Known(*b))
            } else if *b == 1 {
                Ok(Dim::Known(*a))
            } else {
                Err(mismatch())
            }
        }
        (Dim::Known(1), other) | (other, Dim::Known(1)) => Ok(other.clone()),
        (Dim::Symbol(a), Dim::Symbol(b)) if a == b => Ok(Dim::Symbol(*a)),
        (Dim::Symbol(symbol), Dim::Known(k)) | (Dim::Known(k), Dim::Symbol(symbol)) => Ok(
            Dim::unknown(format!("symbol {symbol} broadcast against extent {k}")),
        ),
        (Dim::Symbol(a), Dim::Symbol(b)) => {
            Ok(Dim::unknown(format!("symbols {a} and {b} broadcast")))
        }
        (Dim::Unknown(_), _) | (_, Dim::Unknown(_)) => Ok(Dim::unknown(format!(
            "axis {axis} broadcast of {left} and {right}"
        ))),
    }
}

/// ONNX `MatMul` shape rule: right-aligned batch broadcast plus matrix
/// contraction.
fn matmul_shape(left: &ValueShape, right: &ValueShape) -> Result<ValueShape, PlanError> {
    let (ValueShape::Dims(left_dims), ValueShape::Dims(right_dims)) = (left, right) else {
        return Ok(ValueShape::Unknown);
    };
    if left_dims.len() < 2 || right_dims.len() < 2 {
        return Err(shape_problem(format!(
            "matmul needs rank 2 or more, found {} and {}",
            left.render(),
            right.render()
        )));
    }
    let contract_left = &left_dims[left_dims.len() - 1];
    let contract_right = &right_dims[right_dims.len() - 2];
    match (contract_left.as_known(), contract_right.as_known()) {
        (Some(a), Some(b)) if a != b => {
            return Err(shape_problem(format!(
                "matmul contracts {a} against {b} ({} x {})",
                left.render(),
                right.render()
            )))
        }
        _ => {}
    }
    let left_batch = &left_dims[..left_dims.len() - 2];
    let right_batch = &right_dims[..right_dims.len() - 2];
    let mut batch = broadcast_shape_slices(left_batch, right_batch)?;
    batch.push(left_dims[left_dims.len() - 2].clone());
    batch.push(right_dims[right_dims.len() - 1].clone());
    Ok(ValueShape::Dims(batch))
}

/// Broadcast two batch-axis slices, right-aligned.
fn broadcast_shape_slices(left: &[Dim], right: &[Dim]) -> Result<Vec<Dim>, PlanError> {
    let rank = left.len().max(right.len());
    let mut dims = Vec::with_capacity(rank);
    for position in 0..rank {
        let left_dim = offset_dim(left, rank, position);
        let right_dim = offset_dim(right, rank, position);
        dims.push(broadcast_dim(&left_dim, &right_dim, position)?);
    }
    Ok(dims)
}

/// Normalize an ONNX axis against `rank`.
fn normalize_axis(axis: i64, rank: usize, what: &str) -> Result<usize, PlanError> {
    let rank_i = rank as i64;
    let normalized = if axis < 0 { axis + rank_i } else { axis };
    if normalized < 0 || normalized >= rank_i {
        return Err(shape_problem(format!(
            "{what} axis {axis} is outside a rank-{rank} tensor"
        )));
    }
    Ok(normalized as usize)
}

/// Resolve an ONNX `Reshape` target with `allowzero = 0` semantics.
///
/// `0` copies the input extent at the same position, `-1` infers the remaining
/// extent once, and a positive value is that extent. `allowzero = 1` is
/// rejected by the caller, because the CPU kernel implements `allowzero = 0`.
fn resolve_reshape(input: &ValueShape, target: &AtomSeries) -> Result<ValueShape, PlanError> {
    let Some(items) = target.items.as_ref() else {
        return Ok(ValueShape::Unknown);
    };
    let input_dims = input.dims();
    let mut dims: Vec<Dim> = Vec::with_capacity(items.len());
    let mut inferred: Option<usize> = None;
    let mut known_product: Option<u64> = Some(1);
    for (position, atom) in items.iter().enumerate() {
        let extent = match atom.as_i64() {
            Some(-1) => {
                if inferred.is_some() {
                    return Err(shape_problem(
                        "reshape target uses -1 more than once".to_string(),
                    ));
                }
                inferred = Some(position);
                dims.push(Dim::Known(0));
                continue;
            }
            Some(0) => input_dims.get(position).cloned().ok_or_else(|| {
                shape_problem(format!(
                    "reshape target copies axis {position} with 0, but the input {} has rank {}",
                    input.render(),
                    input.rank().unwrap_or(0)
                ))
            })?,
            Some(other) if other < 0 => {
                return Err(shape_problem(format!(
                    "reshape target has the negative extent {other}; only -1 infers"
                )))
            }
            Some(other) => Dim::Known(other as u64),
            // A symbolic extent in a reshape target is a concrete runtime
            // number; keep the symbol rather than losing the axis.
            None => match atom {
                Atom::Dim(dim) if !matches!(dim, Dim::Unknown(_)) => dim.clone(),
                other => {
                    return Err(shape_problem(format!(
                        "reshape target element {position} is {} and not a statically known \
                         extent",
                        other.render()
                    )))
                }
            },
        };
        match extent.as_known() {
            Some(value) => {
                if let Some(product) = known_product {
                    known_product = product.checked_mul(value);
                }
            }
            None => known_product = None,
        }
        dims.push(extent);
    }
    match inferred {
        Some(position) => {
            dims[position] = infer_reshape_extent(input, &dims, position)?;
        }
        None => {
            if let (Some(input_elements), Some(target_elements)) = (input.numel(), known_product) {
                if input_elements != target_elements {
                    return Err(shape_problem(format!(
                        "reshape target {} holds {target_elements} elements, the input {} holds \
                         {input_elements}",
                        render_dims(&dims),
                        input.render()
                    )));
                }
            }
        }
    }
    Ok(ValueShape::Dims(dims))
}

/// Infer the `-1` extent of a reshape target.
///
/// Identical factors of the input and of the target cancel, which is what makes
/// `[batch, seq, 128] -> [batch, seq, -1, 64]` resolve to `2` without inventing
/// algebra. When cancellation leaves only known extents the ratio is computed
/// exactly and a non-divisible request is an error. Anything else stays
/// [`Dim::Unknown`] with the expression that could not be reduced.
fn infer_reshape_extent(
    input: &ValueShape,
    target: &[Dim],
    inferred: usize,
) -> Result<Dim, PlanError> {
    let mut input_factors: Vec<Dim> = input
        .dims()
        .iter()
        .filter(|dim| dim.as_known() != Some(1))
        .cloned()
        .collect();
    let mut target_factors: Vec<Dim> = target
        .iter()
        .enumerate()
        .filter(|(position, _)| *position != inferred)
        .map(|(_, dim)| dim.clone())
        .filter(|dim| dim.as_known() != Some(1))
        .collect();
    let mut position = 0;
    while position < target_factors.len() {
        match input_factors
            .iter()
            .position(|dim| dims_identical(dim, &target_factors[position]))
        {
            Some(found) => {
                input_factors.remove(found);
                target_factors.remove(position);
            }
            None => position += 1,
        }
    }
    let input_product = factor_product(&input_factors);
    let target_product = factor_product(&target_factors);
    if let (Some(input_product), Some(target_product)) = (input_product, target_product) {
        if target_product == 0 {
            return Ok(Dim::unknown(
                "reshape infers -1 beside a zero-length target",
            ));
        }
        if input_product % target_product == 0 {
            return Ok(Dim::Known(input_product / target_product));
        }
        return Err(shape_problem(format!(
            "reshape cannot infer -1: {} elements are not divisible by the target factors {}",
            input_product,
            render_dims(&target_factors)
        )));
    }
    match input_factors.as_slice() {
        [] => Ok(Dim::Known(1)),
        [only] if target_factors.is_empty() => Ok(only.clone()),
        _ => Ok(Dim::unknown(format!(
            "reshape infers -1 as {} without {}",
            render_dims(&input_factors),
            render_dims(&target_factors)
        ))),
    }
}

/// `true` when two extents are provably the same factor.
fn dims_identical(left: &Dim, right: &Dim) -> bool {
    match (left, right) {
        (Dim::Known(a), Dim::Known(b)) => a == b,
        (Dim::Symbol(a), Dim::Symbol(b)) => a == b,
        _ => false,
    }
}

/// `true` when two extents provably differ.
///
/// Two symbols conflict only when they are different symbols; an unknown extent
/// never conflicts, because "not proven" is not "different".
fn dims_conflict(left: &Dim, right: &Dim) -> bool {
    match (left, right) {
        (Dim::Known(a), Dim::Known(b)) => a != b,
        (Dim::Symbol(a), Dim::Symbol(b)) => a != b,
        _ => false,
    }
}

/// The product of a factor list, when every factor is known.
fn factor_product(factors: &[Dim]) -> Option<u64> {
    let mut product: u64 = 1;
    for factor in factors {
        product = product.checked_mul(factor.as_known()?)?;
    }
    Some(product)
}

/// Render a factor list as `[batch, seq, 128]`.
fn render_dims(dims: &[Dim]) -> String {
    let rendered: Vec<String> = dims.iter().map(Dim::to_string).collect();
    format!("[{}]", rendered.join(", "))
}

// ---------------------------------------------------------------------------
// Attribute access
// ---------------------------------------------------------------------------

/// Consumes a node's attributes and records anything it cannot represent.
struct Attrs<'a, 'b> {
    items: &'b [proto::Attribute<'a>],
    consumed: Vec<bool>,
    problems: Vec<PlanError>,
}

impl<'a, 'b> Attrs<'a, 'b> {
    fn new(items: &'b [proto::Attribute<'a>]) -> Attrs<'a, 'b> {
        let mut problems = Vec::new();
        for attribute in items {
            if let Some(declared) = attribute.declared_type {
                if declared != 0 && !attribute.value.matches_type(declared) {
                    problems.push(ImportProblemKind::AttributeTypeMismatch {
                        attribute: attribute.name.to_string(),
                        declared: proto::AttributeValue::type_name(declared),
                        found: attribute.value.kind_name(),
                    });
                }
            }
            match attribute.value {
                proto::AttributeValue::Subgraph => problems.push(ImportProblemKind::Subgraph {
                    attribute: attribute.name.to_string(),
                }),
                proto::AttributeValue::SparseTensor => problems.push(ImportProblemKind::Subgraph {
                    attribute: attribute.name.to_string(),
                }),
                proto::AttributeValue::TypeProto => problems.push(ImportProblemKind::Subgraph {
                    attribute: attribute.name.to_string(),
                }),
                _ => {}
            }
        }
        Attrs {
            items,
            consumed: vec![false; items.len()],
            problems,
        }
    }

    /// Look up an attribute and mark it consumed.
    fn take(&mut self, name: &str) -> Option<&'b proto::Attribute<'a>> {
        let position = self
            .items
            .iter()
            .position(|attribute| attribute.name == name)?;
        self.consumed[position] = true;
        self.items.get(position)
    }

    fn kind_error(&mut self, name: &str, expected: &str, found: &str) -> PlanError {
        ImportProblemKind::UnsupportedAttribute {
            attribute: name.to_string(),
            detail: format!("expected a {expected} payload, found {found}"),
        }
    }

    /// An `INT` attribute.
    fn int(&mut self, name: &str) -> Result<Option<i64>, PlanError> {
        let Some(attribute) = self.take(name) else {
            return Ok(None);
        };
        match &attribute.value {
            proto::AttributeValue::Int(value) => Ok(Some(*value)),
            other => {
                let error = self.kind_error(name, "INT", other.kind_name());
                Err(error)
            }
        }
    }

    /// A `FLOAT` attribute.
    fn float(&mut self, name: &str) -> Result<Option<f32>, PlanError> {
        let Some(attribute) = self.take(name) else {
            return Ok(None);
        };
        match &attribute.value {
            proto::AttributeValue::Float(value) => Ok(Some(*value)),
            other => {
                let error = self.kind_error(name, "FLOAT", other.kind_name());
                Err(error)
            }
        }
    }

    /// An `INTS` attribute.
    fn ints(&mut self, name: &str) -> Result<Option<&'b [i64]>, PlanError> {
        let Some(attribute) = self.take(name) else {
            return Ok(None);
        };
        match &attribute.value {
            proto::AttributeValue::Ints(values) => Ok(Some(values)),
            other => {
                let error = self.kind_error(name, "INTS", other.kind_name());
                Err(error)
            }
        }
    }

    /// A `FLOATS` attribute.
    fn floats(&mut self, name: &str) -> Result<Option<&'b [f32]>, PlanError> {
        let Some(attribute) = self.take(name) else {
            return Ok(None);
        };
        match &attribute.value {
            proto::AttributeValue::Floats(values) => Ok(Some(values)),
            other => {
                let error = self.kind_error(name, "FLOATS", other.kind_name());
                Err(error)
            }
        }
    }

    /// A `STRING` attribute, which ONNX encodes as raw bytes.
    fn string(&mut self, name: &str) -> Result<Option<&'b str>, PlanError> {
        let Some(attribute) = self.take(name) else {
            return Ok(None);
        };
        match &attribute.value {
            proto::AttributeValue::String(bytes) => {
                std::str::from_utf8(bytes).map(Some).map_err(|_| {
                    ImportProblemKind::UnsupportedAttribute {
                        attribute: name.to_string(),
                        detail: "the value is not valid UTF-8".to_string(),
                    }
                })
            }
            other => {
                let error = self.kind_error(name, "STRING", other.kind_name());
                Err(error)
            }
        }
    }

    /// A `TENSOR` attribute.
    fn tensor(&mut self, name: &str) -> Result<Option<&'b proto::Tensor<'a>>, PlanError> {
        let Some(attribute) = self.take(name) else {
            return Ok(None);
        };
        match &attribute.value {
            proto::AttributeValue::Tensor(tensor) => Ok(Some(tensor)),
            other => {
                let error = self.kind_error(name, "TENSOR", other.kind_name());
                Err(error)
            }
        }
    }

    /// Report every attribute the mapping did not consume.
    fn finish(mut self, accepted: Option<&str>) -> Vec<PlanError> {
        let mut problems = std::mem::take(&mut self.problems);
        if let Some(accepted) = accepted {
            for (position, attribute) in self.items.iter().enumerate() {
                if !self.consumed[position] {
                    problems.push(ImportProblemKind::UnknownAttribute {
                        attribute: attribute.name.to_string(),
                        accepted: accepted.to_string(),
                    });
                }
            }
        }
        problems
    }
}

/// Attributes each mapped operator accepts, for the unknown-attribute
/// diagnostic.
fn accepted_attributes(op_type: &str) -> &'static str {
    match op_type {
        "Gather" => "axis, batch_dims",
        "Transpose" => "perm",
        "Reshape" => "allowzero",
        "Flatten" => "axis",
        "Concat" => "axis",
        "Split" => "axis, num_outputs, split",
        "Slice" => "(none)",
        "Squeeze" | "Unsqueeze" => "(none)",
        "Shape" => "start, end",
        "ConstantOfShape" => "value",
        "Expand" => "(none)",
        "Range" => "(none)",
        "Cast" => "to, saturate",
        "Where" => "(none)",
        "Equal" | "Less" | "Greater" => "(none)",
        "Trilu" => "upper",
        "ScatterND" => "(none)",
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => "(none)",
        "Sqrt" | "Erf" | "Tanh" | "Relu" | "Neg" | "Sin" | "Cos" => "(none)",
        "MatMul" => "(none)",
        "Gemm" => "alpha, beta, transA, transB",
        "Softmax" => "axis",
        "ReduceMean" | "ReduceSum" => "axes, keepdims",
        "ArgMax" => "axis, keepdims, select_last_index",
        "Conv" => "auto_pad, dilations, group, kernel_shape, pads, strides",
        "MaxPool" => "auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides",
        "AveragePool" => {
            "auto_pad, ceil_mode, count_include_pad, dilations, kernel_shape, pads, strides"
        }
        "GlobalAveragePool" => "(none)",
        "BatchNormalization" => "epsilon, momentum, training_mode",
        "Resize" => {
            "coordinate_transformation_mode, cubic_coeff_a, exclude_outside, \
             extrapolation_value, mode, nearest_mode"
        }
        "Pad" => "mode, pads, value",
        "Clip" => "min, max",
        "Sigmoid" | "Max" | "Min" => "(none)",
        "TopK" => "axis, k, largest, sorted",
        _ => "(none)",
    }
}

// ---------------------------------------------------------------------------
// Operand access
// ---------------------------------------------------------------------------

/// Number of elements a shape holds, as a [`Dim`].
fn len_dim(shape: &ValueShape) -> Dim {
    match shape {
        ValueShape::Unknown => Dim::unknown("rank is unknown"),
        ValueShape::Dims(dims) if dims.is_empty() => Dim::Known(1),
        ValueShape::Dims(dims) if dims.len() == 1 => dims[0].clone(),
        ValueShape::Dims(dims) => {
            let mut product: u64 = 1;
            for dim in dims {
                match dim.as_known() {
                    Some(extent) => match product.checked_mul(extent) {
                        Some(value) => product = value,
                        None => return Dim::unknown(format!("product of {}", shape.render())),
                    },
                    None => return Dim::unknown(format!("product of {}", shape.render())),
                }
            }
            Dim::Known(product)
        }
    }
}

/// Check the operand count of a node.
fn expect_inputs(ins: &[Abstract], expected: usize) -> Result<(), PlanError> {
    if ins.len() == expected {
        Ok(())
    } else {
        Err(ImportProblemKind::OperandCount {
            expected,
            found: ins.len(),
        })
    }
}

/// An operand's dtype and shape.
fn input_tensor(
    ins: &[Abstract],
    index: usize,
    what: &str,
) -> Result<(DType, ValueShape), PlanError> {
    match ins.get(index) {
        Some(Abstract::Tensor { dtype, shape }) | Some(Abstract::Values { dtype, shape, .. }) => {
            Ok((*dtype, shape.clone()))
        }
        Some(Abstract::Poisoned) | None => Err(shape_problem(format!(
            "{what} is not available (operand {index})"
        ))),
    }
}

/// An operand's dtype, shape and tracked element sequence.
fn input_series(
    ins: &[Abstract],
    index: usize,
    what: &str,
) -> Result<(DType, ValueShape, AtomSeries), PlanError> {
    match ins.get(index) {
        Some(Abstract::Values {
            dtype,
            shape,
            series,
        }) => Ok((*dtype, shape.clone(), series.clone())),
        Some(Abstract::Tensor { dtype, shape }) => Ok((
            *dtype,
            shape.clone(),
            AtomSeries {
                len: len_dim(shape),
                items: None,
            },
        )),
        Some(Abstract::Poisoned) | None => Err(shape_problem(format!(
            "{what} is not available (operand {index})"
        ))),
    }
}

/// The element sequence of an operand, when it has one.
fn series_of(value: &Abstract) -> Option<&AtomSeries> {
    match value {
        Abstract::Values { series, .. } => Some(series),
        _ => None,
    }
}

/// Require a list of known integers (an axis list, split sizes, a shape).
fn int_list(series: &AtomSeries, what: &str) -> Result<Vec<i64>, PlanError> {
    let items = series
        .items
        .as_ref()
        .ok_or_else(|| shape_problem(format!("{what} is not statically known")))?;
    items
        .iter()
        .map(|atom| {
            atom.as_i64().ok_or_else(|| {
                shape_problem(format!(
                    "{what} contains the unknown element {}",
                    atom.render()
                ))
            })
        })
        .collect()
}

/// A tensor's declared shape, rejecting negative extents.
fn tensor_shape(tensor: &proto::Tensor<'_>, carrier: &str) -> Result<Shape, PlanError> {
    // The protobuf reader already caps `dims`; this is the second reader on the
    // path (a `Tensor` can also be built by hand in a test), so the budget is
    // enforced where the `Shape` is created as well.
    if tensor.dims.len() > MAX_RANK {
        return Err(ImportProblemKind::LimitExceeded {
            what: "tensor rank",
            requested: tensor.dims.len() as u64,
            limit: MAX_RANK as u64,
        });
    }
    let mut dims = Vec::with_capacity(tensor.dims.len());
    for extent in &tensor.dims {
        match u64::try_from(*extent) {
            Ok(extent) => dims.push(extent),
            Err(_) => {
                return Err(ImportProblemKind::NegativeExtent {
                    tensor: carrier.to_string(),
                    shape: format!("{:?}", tensor.dims),
                })
            }
        }
    }
    Ok(Shape::from_dims(dims))
}

/// A tensor's payload bytes, validated against its dtype and shape.
fn tensor_bytes(
    tensor: &proto::Tensor<'_>,
    dtype: DType,
    shape: &Shape,
    carrier: &str,
) -> Result<Vec<u8>, PlanError> {
    if tensor.has_segment {
        return Err(ImportProblemKind::SegmentedTensor {
            tensor: carrier.to_string(),
        });
    }
    if tensor.has_external_data || tensor.data_location == 1 {
        return Err(ImportProblemKind::ExternalTensorData {
            tensor: carrier.to_string(),
        });
    }
    let elements = shape
        .numel()
        .map_err(|error| shape_problem(error.to_string()))?;
    let expected = elements
        .checked_mul(dtype.size_bytes() as u64)
        .ok_or_else(|| shape_problem(format!("{shape} of {dtype} overflows a byte count")))?;
    let mut typed = Vec::new();
    if !tensor.float_data.is_empty() {
        typed.push("float_data");
    }
    if !tensor.int32_data.is_empty() {
        typed.push("int32_data");
    }
    if !tensor.int64_data.is_empty() {
        typed.push("int64_data");
    }
    if !tensor.uint64_data.is_empty() {
        typed.push("uint64_data");
    }
    if !tensor.double_data.is_empty() {
        typed.push("double_data");
    }
    if tensor.string_data_count > 0 {
        typed.push("string_data");
    }
    if let Some(raw) = tensor.raw_data {
        if !typed.is_empty() {
            return Err(ImportProblemKind::AmbiguousTensorData {
                tensor: carrier.to_string(),
                fields: format!("raw_data and {}", typed.join(" + ")),
            });
        }
        if raw.len() as u64 != expected {
            return Err(ImportProblemKind::TensorLength {
                tensor: carrier.to_string(),
                dtype,
                shape: shape.to_string(),
                expected,
                actual: raw.len() as u64,
            });
        }
        return Ok(raw.to_vec());
    }
    if typed.len() > 1 {
        return Err(ImportProblemKind::AmbiguousTensorData {
            tensor: carrier.to_string(),
            fields: typed.join(" + "),
        });
    }
    // Pair the one populated field with the dtype *before* converting it. A
    // payload that the dtype does not use is a statement about the file, not a
    // buffer to materialise: converting first would allocate `elements ×
    // size_bytes` for a tensor that is about to be rejected.
    let (field, count, element_bytes) = match dtype {
        DType::F32 => ("float_data", tensor.float_data.len(), 4u64),
        DType::I64 => ("int64_data", tensor.int64_data.len(), 8),
        DType::I32 => ("int32_data", tensor.int32_data.len(), 4),
        DType::Bool => ("int32_data", tensor.int32_data.len(), 1),
    };
    if let Some(found) = typed.first() {
        if *found != field {
            return Err(ImportProblemKind::TensorDataKind {
                tensor: carrier.to_string(),
                dtype,
                found,
            });
        }
    } else if expected > 0 {
        return Err(ImportProblemKind::MissingTensorData {
            tensor: carrier.to_string(),
        });
    }
    if count as u64 != elements {
        return Err(ImportProblemKind::TensorLength {
            tensor: carrier.to_string(),
            dtype,
            shape: shape.to_string(),
            expected,
            actual: (count as u64).saturating_mul(element_bytes),
        });
    }
    let bytes: Vec<u8> = match dtype {
        DType::F32 => tensor
            .float_data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        DType::I64 => tensor
            .int64_data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        DType::I32 => tensor
            .int32_data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        DType::Bool => tensor
            .int32_data
            .iter()
            .map(|value| u8::from(*value != 0))
            .collect(),
    };
    Ok(bytes)
}

/// Build a weight from a decoded tensor.
fn build_weight(
    tensor: &proto::Tensor<'_>,
    source: WeightSource,
    carrier: impl FnOnce() -> String,
) -> Result<WeightTensor, PlanError> {
    let name = carrier();
    if name.is_empty() {
        return Err(ImportProblemKind::MissingName {
            what: if source == WeightSource::Initializer {
                "initializer"
            } else {
                "attribute tensor"
            },
        });
    }
    let dtype = onnx_dtype(tensor.data_type, format!("tensor `{name}`"))?;
    let shape = tensor_shape(tensor, &name)?;
    let bytes = tensor_bytes(tensor, dtype, &shape, &name)?;
    WeightTensor::new(name, dtype, shape, bytes, source)
        .map_err(|error| shape_problem(error.to_string()))
}

/// Build a weight from an attribute payload the importer synthesized.
fn attribute_weight(
    name: &str,
    dtype: DType,
    dims: Vec<u64>,
    bytes: Vec<u8>,
) -> Result<WeightTensor, PlanError> {
    if dims.len() > MAX_RANK {
        return Err(ImportProblemKind::LimitExceeded {
            what: "constant rank",
            requested: dims.len() as u64,
            limit: MAX_RANK as u64,
        });
    }
    WeightTensor::new(
        name,
        dtype,
        Shape::from_dims(dims),
        bytes,
        WeightSource::AttributeConstant,
    )
    .map_err(|error| shape_problem(error.to_string()))
}

/// The tracked form of a weight's value, so shape arithmetic can fold through
/// initializers.
fn weight_abstract(weight: &WeightTensor) -> Abstract {
    let shape = ValueShape::from_dims(weight.shape().dims().iter().copied());
    let elements = weight.shape().numel().unwrap_or(0);
    if elements > MAX_TRACKED_ELEMENTS as u64 {
        return Abstract::tensor(weight.dtype(), shape);
    }
    match weight.dtype() {
        DType::I64 => {
            let items: Vec<Atom> = weight
                .bytes()
                .chunks_exact(8)
                .map(|chunk| {
                    let mut word = [0u8; 8];
                    word.copy_from_slice(chunk);
                    Atom::Int(i64::from_le_bytes(word))
                })
                .collect();
            Abstract::values(DType::I64, shape, AtomSeries::from_atoms(items))
        }
        // Float constants are tracked so `Resize` scales, `Clip` bounds and
        // `Pad` fill values can be folded at import; a data initializer of the
        // same dtype usually exceeds the sequence budget and stays a tensor.
        DType::F32 => {
            let items: Vec<Atom> = weight
                .bytes()
                .chunks_exact(4)
                .map(|chunk| {
                    let mut word = [0u8; 4];
                    word.copy_from_slice(chunk);
                    Atom::Float(f32::from_le_bytes(word))
                })
                .collect();
            Abstract::values(DType::F32, shape, AtomSeries::from_atoms(items))
        }
        _ => Abstract::tensor(weight.dtype(), shape),
    }
}

/// A planned value from a `PlannedValue`.
fn planned_abstract(value: &PlannedValue) -> Abstract {
    match &value.series {
        Some(series) => Abstract::values(value.dtype, value.shape.clone(), series.clone()),
        None => Abstract::tensor(value.dtype, value.shape.clone()),
    }
}

// ---------------------------------------------------------------------------
// Per-operator planning
// ---------------------------------------------------------------------------

/// `Gather(data, indices, axis)`.
fn plan_gather(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 2)?;
    let (data_dtype, data_shape) = input_tensor(ins, 0, "Gather data")?;
    let (index_dtype, index_shape, index_series) = input_series(ins, 1, "Gather indices")?;
    if index_dtype != DType::I64 {
        return Err(ImportProblemKind::IndexDType {
            actual: index_dtype,
            operand: "Gather indices".to_string(),
        });
    }
    let batch_dims = attrs.int("batch_dims")?.unwrap_or(0);
    if batch_dims != 0 {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "batch_dims".to_string(),
            detail: format!(
                "batch_dims={batch_dims} is not implemented; only 0 (a flat gather) is"
            ),
        });
    }
    let axis = attrs.int("axis")?.unwrap_or(0);
    let rank = data_shape
        .rank()
        .ok_or_else(|| shape_problem("Gather data rank is unknown".to_string()))?;
    let axis = normalize_axis(axis, rank, "Gather")?;
    let params = OpParams::new().with_axes([axis as i64]);
    let (ValueShape::Dims(index_dims), ValueShape::Dims(data_dims)) = (&index_shape, &data_shape)
    else {
        return Ok(PlannedOp::single(
            OpId::Gather,
            vec![Operand::Input(0), Operand::Input(1)],
            params,
            PlannedValue::tensor(data_dtype, ValueShape::Unknown),
        ));
    };
    let mut dims = data_dims[..axis].to_vec();
    dims.extend(index_dims.iter().cloned());
    dims.extend(data_dims[axis + 1..].iter().cloned());
    let shape = ValueShape::Dims(dims);
    if let (Some(data_series), Some(index)) = (
        series_of(&ins[0]),
        index_series.only().and_then(Atom::as_i64),
    ) {
        if let Some(atom) = data_series.item(index) {
            return Ok(PlannedOp::single(
                OpId::Gather,
                vec![Operand::Input(0), Operand::Input(1)],
                params,
                PlannedValue::values(data_dtype, shape, AtomSeries::scalar(atom)),
            ));
        }
    }
    Ok(PlannedOp::single(
        OpId::Gather,
        vec![Operand::Input(0), Operand::Input(1)],
        params,
        PlannedValue::tensor(data_dtype, shape),
    ))
}

/// `Transpose(data, perm)`.
fn plan_transpose(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 1)?;
    let (dtype, shape) = input_tensor(ins, 0, "Transpose data")?;
    let rank = shape
        .rank()
        .ok_or_else(|| shape_problem("Transpose input rank is unknown".to_string()))?;
    let perm: Vec<i64> = match attrs.ints("perm")? {
        Some(values) => values.to_vec(),
        None => (0..rank as i64).rev().collect(),
    };
    if perm.len() != rank {
        return Err(shape_problem(format!(
            "Transpose perm has {} axes for a rank-{rank} input",
            perm.len()
        )));
    }
    let mut normalized = Vec::with_capacity(rank);
    for axis in &perm {
        normalized.push(normalize_axis(*axis, rank, "Transpose perm")?);
    }
    // `normalize_axis` bounds each entry but does not notice a repeat. A
    // permutation that names an axis twice would reach the kernel as
    // `OpParams.axes = [0, 0]`, which is not a transpose; refuse it here where
    // the node can be named.
    Shape::validate_permutation(&normalized, rank)
        .map_err(|error| shape_problem(format!("Transpose perm {perm:?}: {error}")))?;
    let ValueShape::Dims(dims) = &shape else {
        return Ok(PlannedOp::single(
            OpId::Transpose,
            vec![Operand::Input(0)],
            OpParams::new().with_axes(perm),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    let mut permuted = Vec::with_capacity(rank);
    for axis in &normalized {
        permuted.push(dims[*axis].clone());
    }
    Ok(PlannedOp::single(
        OpId::Transpose,
        vec![Operand::Input(0)],
        OpParams::new().with_axes(perm),
        PlannedValue::tensor(dtype, ValueShape::Dims(permuted)),
    ))
}

/// `Reshape(data, shape, allowzero = 0)`.
fn plan_reshape(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 2)?;
    let allowzero = attrs.int("allowzero")?.unwrap_or(0);
    if allowzero != 0 {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "allowzero".to_string(),
            detail: "allowzero=1 is not representable: the fn3 reshape contract implements \
                     allowzero=0 (0 copies the input extent, -1 infers). Owner: FN3-07 (#53) \
                     would need a kernel mode"
                .to_string(),
        });
    }
    let (dtype, shape) = input_tensor(ins, 0, "Reshape data")?;
    let (target_dtype, _, target) = input_series(ins, 1, "Reshape target")?;
    if target_dtype != DType::I64 {
        return Err(ImportProblemKind::IndexDType {
            actual: target_dtype,
            operand: "Reshape target".to_string(),
        });
    }
    let output = resolve_reshape(&shape, &target)?;
    // The typed IR resolves a reshape from `OpParams::axes`; export the raw
    // ONNX target (0 copies, -1 infers) when every element is a constant. A
    // symbolic target cannot be encoded as `i64` and stays runtime-only.
    let typed_target: Option<Vec<i64>> = target
        .items
        .as_ref()
        .and_then(|items| items.iter().map(Atom::as_i64).collect::<Option<Vec<i64>>>());
    let series = series_of(&ins[0]).cloned();
    let value = match series {
        Some(series) => PlannedValue::values(dtype, output, series),
        None => PlannedValue::tensor(dtype, output),
    };
    let mut operation = PlannedOp::single(
        OpId::Reshape,
        vec![Operand::Input(0), Operand::Input(1)],
        OpParams::new(),
        value,
    );
    if let Some(target) = typed_target {
        operation[0].typed = Some(OpParams::new().with_axes(target));
    }
    Ok(operation)
}

/// `Flatten(data, axis)`.
fn plan_flatten(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 1)?;
    let (dtype, shape) = input_tensor(ins, 0, "Flatten data")?;
    let Some(rank) = shape.rank() else {
        return Ok(PlannedOp::single(
            OpId::Flatten,
            vec![Operand::Input(0)],
            OpParams::new(),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    let axis = attrs.int("axis")?.unwrap_or(1);
    if axis < 0 || axis as usize > rank {
        return Err(shape_problem(format!(
            "Flatten axis {axis} is outside 0..={rank}"
        )));
    }
    let axis = axis as usize;
    let before = ValueShape::Dims(shape.dims()[..axis].to_vec());
    let after = ValueShape::Dims(shape.dims()[axis..].to_vec());
    let output = ValueShape::Dims(vec![len_dim(&before), len_dim(&after)]);
    Ok(PlannedOp::single(
        OpId::Flatten,
        vec![Operand::Input(0)],
        OpParams::new().with_axes([axis as i64]),
        PlannedValue::tensor(dtype, output),
    ))
}

/// `Squeeze(data, axes?)`.
fn plan_squeeze(
    _ctx: &NodeRef,
    _attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    if ins.is_empty() || ins.len() > 2 {
        return Err(ImportProblemKind::OperandCount {
            expected: 2,
            found: ins.len(),
        });
    }
    let (dtype, shape) = input_tensor(ins, 0, "Squeeze data")?;
    let Some(rank) = shape.rank() else {
        return Ok(PlannedOp::single(
            OpId::Squeeze,
            vec![Operand::Input(0)],
            OpParams::new(),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    let axes: Vec<usize> = if ins.len() == 2 {
        let (axes_dtype, _, series) = input_series(ins, 1, "Squeeze axes")?;
        if axes_dtype != DType::I64 {
            return Err(ImportProblemKind::IndexDType {
                actual: axes_dtype,
                operand: "Squeeze axes".to_string(),
            });
        }
        let axes = int_list(&series, "Squeeze axes")?;
        let mut normalized = Vec::with_capacity(axes.len());
        for axis in axes {
            normalized.push(normalize_axis(axis, rank, "Squeeze")?);
        }
        normalized
    } else {
        shape
            .dims()
            .iter()
            .enumerate()
            .filter(|(_, dim)| dim.as_known() == Some(1))
            .map(|(axis, _)| axis)
            .collect()
    };
    for axis in &axes {
        if shape.dims()[*axis]
            .as_known()
            .is_some_and(|extent| extent != 1)
        {
            return Err(shape_problem(format!(
                "Squeeze removes axis {axis} of extent {}",
                shape.dims()[*axis]
            )));
        }
    }
    let dims: Vec<Dim> = shape
        .dims()
        .iter()
        .enumerate()
        .filter(|(axis, _)| !axes.contains(axis))
        .map(|(_, dim)| dim.clone())
        .collect();
    let output = ValueShape::Dims(dims);
    let operands = if ins.len() == 2 {
        vec![Operand::Input(0), Operand::Input(1)]
    } else {
        vec![Operand::Input(0)]
    };
    let series = series_of(&ins[0]).cloned();
    let value = match series {
        Some(series) => PlannedValue::values(dtype, output, series),
        None => PlannedValue::tensor(dtype, output),
    };
    Ok(PlannedOp::single(
        OpId::Squeeze,
        operands,
        OpParams::new(),
        value,
    ))
}

/// `Unsqueeze(data, axes)`.
fn plan_unsqueeze(
    _ctx: &NodeRef,
    _attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 2)?;
    let (dtype, shape) = input_tensor(ins, 0, "Unsqueeze data")?;
    let (axes_dtype, _, series) = input_series(ins, 1, "Unsqueeze axes")?;
    if axes_dtype != DType::I64 {
        return Err(ImportProblemKind::IndexDType {
            actual: axes_dtype,
            operand: "Unsqueeze axes".to_string(),
        });
    }
    let axes = int_list(&series, "Unsqueeze axes")?;
    let Some(rank) = shape.rank() else {
        return Ok(PlannedOp::single(
            OpId::Unsqueeze,
            vec![Operand::Input(0), Operand::Input(1)],
            OpParams::new(),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    let output_rank = rank + axes.len();
    let mut normalized = Vec::with_capacity(axes.len());
    for axis in &axes {
        let axis = normalize_axis(*axis, output_rank, "Unsqueeze")?;
        if normalized.contains(&axis) {
            return Err(shape_problem(format!(
                "Unsqueeze inserts axis {axis} twice"
            )));
        }
        normalized.push(axis);
    }
    let mut dims: Vec<Dim> = Vec::with_capacity(output_rank);
    let mut source = shape.dims().iter();
    for axis in 0..output_rank {
        if normalized.contains(&axis) {
            dims.push(Dim::Known(1));
        } else {
            dims.push(source.next().cloned().unwrap_or(Dim::Known(1)));
        }
    }
    let output = ValueShape::Dims(dims);
    let series = series_of(&ins[0]).cloned();
    let value = match series {
        Some(series) => PlannedValue::values(dtype, output, series),
        None => PlannedValue::tensor(dtype, output),
    };
    Ok(PlannedOp::single(
        OpId::Unsqueeze,
        vec![Operand::Input(0), Operand::Input(1)],
        OpParams::new(),
        value,
    ))
}

/// `Concat(inputs..., axis)`.
fn plan_concat(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    if ins.len() < 2 {
        return Err(ImportProblemKind::OperandCount {
            expected: 2,
            found: ins.len(),
        });
    }
    let axis = attrs
        .int("axis")?
        .ok_or_else(|| missing_attribute("axis"))?;
    let mut dtype: Option<DType> = None;
    let mut shapes = Vec::with_capacity(ins.len());
    for index in 0..ins.len() {
        let (item_dtype, shape) = input_tensor(ins, index, "Concat input")?;
        match dtype {
            None => dtype = Some(item_dtype),
            Some(expected) if expected == item_dtype => {}
            Some(expected) => {
                return Err(dtype_problem(format!(
                    "Concat input {index} is {item_dtype}, input 0 is {expected}"
                )))
            }
        }
        shapes.push(shape);
    }
    let dtype = dtype.expect("at least two inputs were checked");
    let rank = shapes[0].rank();
    if shapes.iter().any(|shape| shape.rank() != rank) {
        return Err(shape_problem("Concat inputs have different ranks"));
    }
    let Some(rank) = rank else {
        return Ok(PlannedOp::single(
            OpId::Concat,
            (0..ins.len()).map(Operand::Input).collect(),
            OpParams::new().with_axes([axis]),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    let axis = normalize_axis(axis, rank, "Concat")?;
    let mut dims = shapes[0].dims().to_vec();
    for shape in &shapes[1..] {
        for (position, dim) in shape.dims().iter().enumerate() {
            if position == axis {
                dims[position] = Dim::add(&dims[position], dim);
            } else if dims_conflict(&dims[position], dim) {
                return Err(shape_problem(format!(
                    "Concat keeps axis {position} at {}, but an input has {dim}",
                    dims[position]
                )));
            }
        }
    }
    let output = ValueShape::Dims(dims);
    let series: Vec<AtomSeries> = ins.iter().filter_map(series_of).cloned().collect();
    let value = if series.len() == ins.len() {
        PlannedValue::values(dtype, output, AtomSeries::concat(&series))
    } else {
        PlannedValue::tensor(dtype, output)
    };
    Ok(PlannedOp::single(
        OpId::Concat,
        (0..ins.len()).map(Operand::Input).collect(),
        OpParams::new().with_axes([axis as i64]),
        value,
    ))
}

/// `Slice(data, starts, ends, axes?, steps?)`.
///
/// Extents are computed when the bounds are known or when an end is the axis
/// symbol itself (`Slice(x, 0, total, axes=[3])` keeps a `total`-wide axis). Any
/// other symbolic bound leaves the axis [`Dim::Unknown`]; the runtime slice
/// input is still passed through unchanged.
fn plan_slice(
    _ctx: &NodeRef,
    _attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    if !(3..=5).contains(&ins.len()) {
        return Err(ImportProblemKind::OperandCount {
            expected: 3,
            found: ins.len(),
        });
    }
    let (dtype, shape) = input_tensor(ins, 0, "Slice data")?;
    let Some(rank) = shape.rank() else {
        return Ok(PlannedOp::single(
            OpId::Slice,
            (0..ins.len()).map(Operand::Input).collect(),
            OpParams::new(),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    let atoms = |index: usize, what: &str| -> Result<Vec<Atom>, PlanError> {
        let (_, _, series) = input_series(ins, index, what)?;
        series
            .items
            .clone()
            .ok_or_else(|| shape_problem(format!("{what} is not statically known")))
    };
    let starts = atoms(1, "Slice starts")?;
    let ends = atoms(2, "Slice ends")?;
    if starts.len() != ends.len() {
        return Err(shape_problem(format!(
            "Slice has {} starts and {} ends",
            starts.len(),
            ends.len()
        )));
    }
    let axes: Vec<i64> = if ins.len() >= 4 {
        let (_, _, series) = input_series(ins, 3, "Slice axes")?;
        int_list(&series, "Slice axes")?
    } else {
        (0..starts.len() as i64).collect()
    };
    if axes.len() != starts.len() {
        return Err(shape_problem(format!(
            "Slice has {} starts and {} axes",
            starts.len(),
            axes.len()
        )));
    }
    let mut steps = vec![1i64; starts.len()];
    if ins.len() == 5 {
        let (_, _, series) = input_series(ins, 4, "Slice steps")?;
        steps = int_list(&series, "Slice steps")?;
        // `starts`, `ends` and `axes` are length-checked against each other
        // above; `steps` is a separate operand and must be too. Without this
        // check a five-operand Slice with one step and two axes indexes past the
        // end of this vector.
        if steps.len() != starts.len() {
            return Err(shape_problem(format!(
                "Slice has {} starts and {} steps",
                starts.len(),
                steps.len()
            )));
        }
    }
    let mut dims = shape.dims().to_vec();
    for (position, axis) in axes.iter().enumerate() {
        let axis = normalize_axis(*axis, rank, "Slice")?;
        dims[axis] = slice_extent(
            &dims[axis],
            &starts[position],
            &ends[position],
            steps[position],
            axis,
        );
    }
    let output = ValueShape::Dims(dims.clone());
    // Fold the tracked payload only when every bound is a known integer.
    let folded = if dims.len() == 1 && !axes.is_empty() {
        match (
            starts[0].as_i64(),
            ends[0].as_i64(),
            shape.dims()[0].as_known(),
            series_of(&ins[0]),
        ) {
            (Some(start), Some(end), Some(extent), Some(series))
                if axes[0] == 0 && steps[0] != 0 =>
            {
                fold_slice(series, start, end, steps[0], Some(extent))
            }
            _ => None,
        }
    } else {
        None
    };
    let value = match folded {
        Some(atoms) => PlannedValue::values(dtype, output, AtomSeries::from_atoms(atoms)),
        None => PlannedValue::tensor(dtype, output),
    };
    // The typed IR reads slice bounds from `OpParams`, never from a payload, so
    // export them when every bound is a constant. A symbolic bound leaves the
    // node unrepresentable to the typed pass, which reports it by name.
    let typed = typed_slice_params(&starts, &ends, &axes, &steps);
    let mut operation = PlannedOp::single(
        OpId::Slice,
        (0..ins.len()).map(Operand::Input).collect(),
        OpParams::new(),
        value,
    );
    if let Some(typed) = typed {
        operation[0].typed = Some(typed);
    }
    Ok(operation)
}

/// `OpParams` for a `Slice` whose bounds are all constants.
fn typed_slice_params(
    starts: &[Atom],
    ends: &[Atom],
    axes: &[i64],
    steps: &[i64],
) -> Option<OpParams> {
    let starts: Vec<i64> = starts.iter().map(Atom::as_i64).collect::<Option<_>>()?;
    let ends: Vec<i64> = ends.iter().map(Atom::as_i64).collect::<Option<_>>()?;
    Some(
        OpParams::new()
            .with_axes(axes.to_vec())
            .with_slice(starts, ends, steps.to_vec()),
    )
}

/// The extent of one sliced axis.
fn slice_extent(extent: &Dim, start: &Atom, end: &Atom, step: i64, axis: usize) -> Dim {
    if step == 0 {
        return Dim::unknown(format!("slice of axis {axis} has step 0"));
    }
    match extent.as_known() {
        Some(extent) => {
            let extent = extent as i64;
            let (Some(start), Some(end)) = (start.as_i64(), end.as_i64()) else {
                // An end equal to the symbolic axis extent keeps the whole axis.
                if start.as_i64() == Some(0) && end_matches_axis(end, extent as u64) {
                    return Dim::Known(extent as u64);
                }
                return Dim::unknown(format!("slice of axis {axis} has a symbolic bound"));
            };
            let start = clamp_index(start, extent);
            let end = clamp_index(end, extent);
            let count = if step > 0 {
                if end <= start {
                    0
                } else {
                    (end - start + step - 1) / step
                }
            } else if start <= end {
                0
            } else {
                (start - end + (-step) - 1) / (-step)
            };
            Dim::Known(count as u64)
        }
        None => {
            if start.as_i64() == Some(0) && end_matches_axis(end, 0) {
                extent.clone()
            } else {
                Dim::unknown(format!(
                    "slice of the symbolic axis {axis} ({extent}) from {} to {}",
                    start.render(),
                    end.render()
                ))
            }
        }
    }
}

/// `true` when a slice end denotes the whole axis of `extent`.
///
/// `extent` is `0` when the axis is symbolic, in which case only the symbolic
/// comparison is available.
fn end_matches_axis(end: &Atom, extent: u64) -> bool {
    match end {
        Atom::Int(value) if *value >= 0 => {
            if extent == 0 {
                false
            } else {
                *value as u64 >= extent
            }
        }
        Atom::Dim(_) => true,
        _ => false,
    }
}

/// Clamp a Slice index the way ONNX defines it.
fn clamp_index(index: i64, extent: i64) -> i64 {
    if index < 0 {
        (extent + index).clamp(0, extent)
    } else {
        index.min(extent)
    }
}

/// `Expand(data, shape)`.
fn plan_expand(
    _ctx: &NodeRef,
    _attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 2)?;
    let (dtype, shape) = input_tensor(ins, 0, "Expand data")?;
    let (target_dtype, _, target) = input_series(ins, 1, "Expand shape")?;
    if target_dtype != DType::I64 {
        return Err(ImportProblemKind::IndexDType {
            actual: target_dtype,
            operand: "Expand shape".to_string(),
        });
    }
    let Some(items) = target.items.as_ref() else {
        return Ok(PlannedOp::single(
            OpId::Expand,
            vec![Operand::Input(0), Operand::Input(1)],
            OpParams::new(),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    let dims: Vec<Dim> = items.iter().map(Atom::as_dim).collect();
    let target_shape = ValueShape::Dims(dims);
    let output = broadcast_shapes(&shape, &target_shape)?;
    Ok(PlannedOp::single(
        OpId::Expand,
        vec![Operand::Input(0), Operand::Input(1)],
        OpParams::new(),
        PlannedValue::tensor(dtype, output),
    ))
}

/// `Shape(data)`.
fn plan_shape(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 1)?;
    if attrs.int("start")?.is_some() || attrs.int("end")?.is_some() {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "start/end".to_string(),
            detail: "the opset-15 Shape slice attributes are not implemented; the pinned export \
                     slices the result with Slice instead"
                .to_string(),
        });
    }
    let (_, shape) = input_tensor(ins, 0, "Shape data")?;
    let Some(rank) = shape.rank() else {
        return Ok(PlannedOp::single(
            OpId::Shape,
            vec![Operand::Input(0)],
            OpParams::new(),
            PlannedValue::tensor(DType::I64, ValueShape::Unknown),
        ));
    };
    let series = AtomSeries::from_dims(shape.dims());
    Ok(PlannedOp::single(
        OpId::Shape,
        vec![Operand::Input(0)],
        OpParams::new(),
        PlannedValue::values(DType::I64, ValueShape::from_dims([rank as u64]), series),
    ))
}

/// `ConstantOfShape(shape)` with the `value` attribute lifted to an input.
fn plan_constant_of_shape(
    ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 1)?;
    let (shape_dtype, _, shape_series) = input_series(ins, 0, "ConstantOfShape shape")?;
    if shape_dtype != DType::I64 {
        return Err(ImportProblemKind::IndexDType {
            actual: shape_dtype,
            operand: "ConstantOfShape shape".to_string(),
        });
    }
    let (dtype, value_shape, value_bytes) = match attrs.tensor("value")? {
        Some(tensor) => {
            let carrier = format!("{}#value", ctx.name);
            let dtype = onnx_dtype(
                tensor.data_type,
                format!("attribute `value` of {}", ctx.name),
            )?;
            let shape = tensor_shape(tensor, &carrier)?;
            let bytes = tensor_bytes(tensor, dtype, &shape, &carrier)?;
            (dtype, shape, bytes)
        }
        None => (
            DType::F32,
            Shape::from_dims([1u64]),
            0.0f32.to_le_bytes().to_vec(),
        ),
    };
    if value_shape
        .numel()
        .map_err(|error| shape_problem(error.to_string()))?
        != 1
    {
        return Err(shape_problem(format!(
            "ConstantOfShape value {value_shape} is not a single element"
        )));
    }
    let output = match shape_series.items.as_ref() {
        Some(items) => ValueShape::Dims(items.iter().map(Atom::as_dim).collect()),
        None => ValueShape::Unknown,
    };
    let name = if ctx.name.is_empty() {
        format!("node{}#value", ctx.index)
    } else {
        format!("{}#value", ctx.name)
    };
    let constant = WeightTensor::new(
        name,
        dtype,
        value_shape,
        value_bytes,
        WeightSource::AttributeConstant,
    )
    .map_err(|error| shape_problem(error.to_string()))?;
    Ok(PlannedOp::single(
        OpId::ConstantOfShape,
        vec![Operand::Input(0), Operand::NewConstant(Box::new(constant))],
        OpParams::new(),
        PlannedValue::tensor(dtype, output),
    ))
}

/// `Range(start, limit, delta)`.
fn plan_range(
    _ctx: &NodeRef,
    _attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 3)?;
    let (start_dtype, _, start) = input_series(ins, 0, "Range start")?;
    let (limit_dtype, _, limit) = input_series(ins, 1, "Range limit")?;
    let (delta_dtype, _, delta) = input_series(ins, 2, "Range delta")?;
    if start_dtype != limit_dtype || start_dtype != delta_dtype {
        return Err(dtype_problem(format!(
            "Range operands are {start_dtype}, {limit_dtype} and {delta_dtype}"
        )));
    }
    let start_value = start.only().and_then(Atom::as_i64);
    let limit_value = limit.only().and_then(Atom::as_i64);
    let delta_value = delta.only().and_then(Atom::as_i64);
    let len = match (start_value, limit_value, delta_value) {
        (Some(start), Some(limit), Some(delta)) if delta != 0 => {
            let count = ((limit - start) as f64 / delta as f64).ceil();
            Dim::Known(count.max(0.0) as u64)
        }
        (Some(0), None, Some(1)) => match limit.only() {
            Some(Atom::Dim(dim)) => dim.clone(),
            Some(Atom::Int(value)) if *value >= 0 => Dim::Known(*value as u64),
            _ => Dim::unknown("Range length"),
        },
        _ => Dim::unknown("Range length from non-constant operands"),
    };
    let items = match (len.as_known(), start_value, delta_value) {
        (Some(len), Some(start), Some(delta)) if len <= MAX_TRACKED_ELEMENTS as u64 => {
            let mut items = Vec::with_capacity(len as usize);
            for index in 0..len {
                items.push(Atom::Int(
                    start.wrapping_add(index as i64).wrapping_mul(delta),
                ));
            }
            Some(items)
        }
        _ => None,
    };
    let series = AtomSeries {
        len: len.clone(),
        items,
    };
    Ok(PlannedOp::single(
        OpId::Range,
        (0..3).map(Operand::Input).collect(),
        OpParams::new(),
        PlannedValue::values(start_dtype, ValueShape::Dims(vec![len]), series),
    ))
}

/// `Cast(data, to)`.
fn plan_cast(
    ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 1)?;
    let (_, shape) = input_tensor(ins, 0, "Cast data")?;
    let to = attrs.int("to")?.ok_or_else(|| missing_attribute("to"))?;
    let to = i32::try_from(to).map_err(|_| ImportProblemKind::UnsupportedAttribute {
        attribute: "to".to_string(),
        detail: format!("Cast target {to} is not an ONNX dtype code"),
    })?;
    if attrs.int("saturate")?.is_some() {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "saturate".to_string(),
            detail: "Cast.saturate applies to float8 targets only, which this build does not \
                     implement"
                .to_string(),
        });
    }
    let dtype = onnx_dtype(to, format!("the `to` attribute of {}", ctx.name))?;
    Ok(PlannedOp::single(
        OpId::Cast,
        vec![Operand::Input(0)],
        OpParams::new(),
        PlannedValue::tensor(dtype, shape),
    ))
}

/// `Where(condition, x, y)`.
fn plan_where(
    _ctx: &NodeRef,
    _attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 3)?;
    let (condition_dtype, condition_shape) = input_tensor(ins, 0, "Where condition")?;
    if condition_dtype != DType::Bool {
        return Err(dtype_problem(format!(
            "Where condition is {condition_dtype}, not bool"
        )));
    }
    let (dtype, left_shape) = input_tensor(ins, 1, "Where x")?;
    let (right_dtype, right_shape) = input_tensor(ins, 2, "Where y")?;
    if dtype != right_dtype {
        return Err(dtype_problem(format!(
            "Where selects between {dtype} and {right_dtype}"
        )));
    }
    let shape = broadcast_shapes(&condition_shape, &left_shape)?;
    let shape = broadcast_shapes(&shape, &right_shape)?;
    let value = match (series_of(&ins[0]), series_of(&ins[1]), series_of(&ins[2])) {
        (Some(condition), Some(left), Some(right)) => {
            let len = len_dim(&shape);
            let items = match (
                condition.items.as_ref(),
                left.items.as_ref(),
                right.items.as_ref(),
            ) {
                (Some(flags), Some(lefts), Some(rights))
                    if flags.len() == lefts.len() && lefts.len() == rights.len() =>
                {
                    Some(
                        flags
                            .iter()
                            .zip(lefts.iter())
                            .zip(rights.iter())
                            .map(|((flag, left), right)| {
                                if flag.as_bool() == Some(true) {
                                    left.clone()
                                } else if flag.as_bool() == Some(false) {
                                    right.clone()
                                } else {
                                    Atom::Unknown
                                }
                            })
                            .collect(),
                    )
                }
                _ => None,
            };
            PlannedValue::values(dtype, shape, AtomSeries { len, items })
        }
        _ => PlannedValue::tensor(dtype, shape),
    };
    Ok(PlannedOp::single(
        OpId::Where,
        (0..3).map(Operand::Input).collect(),
        OpParams::new(),
        value,
    ))
}

/// `Equal`/`Less`/`Greater`.
fn plan_compare(
    ctx: &NodeRef,
    _attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
    op: OpId,
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 2)?;
    let (dtype, left_shape) = input_tensor(ins, 0, "comparison left")?;
    let (right_dtype, right_shape) = input_tensor(ins, 1, "comparison right")?;
    if dtype != right_dtype {
        return Err(dtype_problem(format!(
            "{} compares {dtype} with {right_dtype}",
            ctx.op_type
        )));
    }
    let shape = broadcast_shapes(&left_shape, &right_shape)?;
    let value = match (series_of(&ins[0]), series_of(&ins[1])) {
        (Some(left), Some(right)) => {
            let fold = |left: &Atom, right: &Atom| match op {
                OpId::Equal => match left.same(right) {
                    Some(equal) => Atom::Bool(equal),
                    None => Atom::Unknown,
                },
                OpId::Less => match (left.as_i64(), right.as_i64()) {
                    (Some(a), Some(b)) => Atom::Bool(a < b),
                    _ => Atom::Unknown,
                },
                _ => match (left.as_i64(), right.as_i64()) {
                    (Some(a), Some(b)) => Atom::Bool(a > b),
                    _ => Atom::Unknown,
                },
            };
            let series = left.zip(right, fold);
            PlannedValue::values(DType::Bool, shape, series)
        }
        _ => PlannedValue::tensor(DType::Bool, shape),
    };
    Ok(PlannedOp::single(
        op,
        vec![Operand::Input(0), Operand::Input(1)],
        OpParams::new(),
        value,
    ))
}

/// `Trilu(data, k?)`.
fn plan_trilu(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    if ins.is_empty() || ins.len() > 2 {
        return Err(ImportProblemKind::OperandCount {
            expected: 2,
            found: ins.len(),
        });
    }
    let (dtype, shape) = input_tensor(ins, 0, "Trilu data")?;
    let upper = attrs.int("upper")?.unwrap_or(1);
    let operands = (0..ins.len()).map(Operand::Input).collect();
    Ok(PlannedOp::single(
        OpId::Trilu,
        operands,
        OpParams::new().with_axes([upper]),
        PlannedValue::tensor(dtype, shape),
    ))
}

/// `ScatterND(data, indices, updates)`.
fn plan_scatter_nd(
    _ctx: &NodeRef,
    _attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 3)?;
    let (dtype, shape) = input_tensor(ins, 0, "ScatterND data")?;
    let (index_dtype, _) = input_tensor(ins, 1, "ScatterND indices")?;
    if index_dtype != DType::I64 {
        return Err(ImportProblemKind::IndexDType {
            actual: index_dtype,
            operand: "ScatterND indices".to_string(),
        });
    }
    let (update_dtype, _) = input_tensor(ins, 2, "ScatterND updates")?;
    if update_dtype != dtype {
        return Err(dtype_problem(format!(
            "ScatterND updates are {update_dtype}, data is {dtype}"
        )));
    }
    Ok(PlannedOp::single(
        OpId::ScatterND,
        (0..3).map(Operand::Input).collect(),
        OpParams::new(),
        PlannedValue::tensor(dtype, shape),
    ))
}

/// Binary elementwise arithmetic with right-aligned broadcasting.
fn plan_binary(
    ctx: &NodeRef,
    _attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
    op: OpId,
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 2)?;
    let (dtype, left_shape) = input_tensor(ins, 0, "left operand")?;
    let (right_dtype, right_shape) = input_tensor(ins, 1, "right operand")?;
    if dtype != right_dtype {
        return Err(dtype_problem(format!(
            "{} takes {dtype} and {right_dtype}",
            ctx.op_type
        )));
    }
    let shape = broadcast_shapes(&left_shape, &right_shape)?;
    let value = match (series_of(&ins[0]), series_of(&ins[1])) {
        (Some(left), Some(right)) if !matches!(op, OpId::Pow | OpId::Maximum | OpId::Minimum) => {
            let series = left.zip(right, |left, right| match op {
                OpId::Add => atom_add(left, right),
                OpId::Sub => atom_sub(left, right),
                OpId::Mul => atom_mul(left, right),
                OpId::Div => atom_div(left, right),
                _ => Atom::Unknown,
            });
            PlannedValue::values(dtype, shape, series)
        }
        _ => PlannedValue::tensor(dtype, shape),
    };
    Ok(PlannedOp::single(
        op,
        vec![Operand::Input(0), Operand::Input(1)],
        OpParams::new(),
        value,
    ))
}

/// Unary elementwise arithmetic or activation.
fn plan_unary(
    _ctx: &NodeRef,
    _attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
    op: OpId,
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 1)?;
    let (dtype, shape) = input_tensor(ins, 0, "operand")?;
    Ok(PlannedOp::single(
        op,
        vec![Operand::Input(0)],
        OpParams::new(),
        PlannedValue::tensor(dtype, shape),
    ))
}

/// `MatMul`.
fn plan_matmul(
    _ctx: &NodeRef,
    _attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 2)?;
    let (dtype, left) = input_tensor(ins, 0, "MatMul left")?;
    let (right_dtype, right) = input_tensor(ins, 1, "MatMul right")?;
    if dtype != right_dtype {
        return Err(dtype_problem(format!(
            "MatMul multiplies {dtype} by {right_dtype}"
        )));
    }
    let shape = matmul_shape(&left, &right)?;
    Ok(PlannedOp::single(
        OpId::MatMul,
        vec![Operand::Input(0), Operand::Input(1)],
        OpParams::new(),
        PlannedValue::tensor(dtype, shape),
    ))
}

/// `Gemm(A, B, C?)` as `MatMul` followed by `Add`.
fn plan_gemm(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    if ins.len() < 2 || ins.len() > 3 {
        return Err(ImportProblemKind::OperandCount {
            expected: 3,
            found: ins.len(),
        });
    }
    let alpha = attrs.float("alpha")?.unwrap_or(1.0);
    let beta = attrs.float("beta")?.unwrap_or(1.0);
    let trans_a = attrs.int("transA")?.unwrap_or(0);
    let trans_b = attrs.int("transB")?.unwrap_or(0);
    if alpha != 1.0 || beta != 1.0 || trans_a != 0 || trans_b != 0 {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "Gemm".to_string(),
            detail: format!(
                "only the identity form alpha = beta = 1, transA = transB = 0 maps to MatMul + \
                 Add; this node has alpha = {alpha}, beta = {beta}, transA = {trans_a}, transB = \
                 {trans_b}"
            ),
        });
    }
    let (dtype, left) = input_tensor(ins, 0, "Gemm A")?;
    let (right_dtype, right) = input_tensor(ins, 1, "Gemm B")?;
    if dtype != right_dtype {
        return Err(dtype_problem(format!(
            "Gemm multiplies {dtype} by {right_dtype}"
        )));
    }
    let shape = matmul_shape(&left, &right)?;
    let mut planned = vec![PlannedOp {
        op: OpId::MatMul,
        operands: vec![Operand::Input(0), Operand::Input(1)],
        params: OpParams::new(),
        typed: None,
        outputs: vec![PlannedValue::tensor(dtype, shape.clone())],
    }];
    if ins.len() == 3 {
        let (bias_dtype, bias_shape) = input_tensor(ins, 2, "Gemm C")?;
        if bias_dtype != dtype {
            return Err(dtype_problem(format!(
                "Gemm bias is {bias_dtype}, A is {dtype}"
            )));
        }
        let output = broadcast_shapes(&shape, &bias_shape)?;
        planned.push(PlannedOp {
            op: OpId::Add,
            operands: vec![Operand::Intermediate(0), Operand::Input(2)],
            params: OpParams::new(),
            typed: None,
            outputs: vec![PlannedValue::tensor(dtype, output)],
        });
    }
    Ok(planned)
}

/// `Softmax(data, axis)`.
fn plan_softmax(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 1)?;
    let (dtype, shape) = input_tensor(ins, 0, "Softmax data")?;
    let axis = attrs.int("axis")?.unwrap_or(-1);
    if let Some(rank) = shape.rank() {
        normalize_axis(axis, rank, "Softmax")?;
    }
    Ok(PlannedOp::single(
        OpId::Softmax,
        vec![Operand::Input(0)],
        OpParams::new().with_axes([axis]),
        PlannedValue::tensor(dtype, shape),
    ))
}

/// `ReduceMean`/`ReduceSum`.
fn plan_reduce(
    ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
    op: OpId,
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 1)?;
    let (dtype, shape) = input_tensor(ins, 0, "reduction input")?;
    let keep_dims = attrs.int("keepdims")?.unwrap_or(1) != 0;
    let Some(rank) = shape.rank() else {
        return Ok(PlannedOp::single(
            op,
            vec![Operand::Input(0)],
            OpParams::new().with_keep_dims(keep_dims),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    let axes: Vec<i64> = match attrs.ints("axes")? {
        Some(values) => values.to_vec(),
        None => (0..rank as i64).collect(),
    };
    let mut normalized = Vec::with_capacity(axes.len());
    for axis in &axes {
        let axis = normalize_axis(*axis, rank, ctx.op_type.as_str())?;
        if !normalized.contains(&axis) {
            normalized.push(axis);
        }
    }
    let dims: Vec<Dim> = shape
        .dims()
        .iter()
        .enumerate()
        .filter_map(|(axis, dim)| {
            if normalized.contains(&axis) {
                if keep_dims {
                    Some(Dim::Known(1))
                } else {
                    None
                }
            } else {
                Some(dim.clone())
            }
        })
        .collect();
    Ok(PlannedOp::single(
        op,
        vec![Operand::Input(0)],
        OpParams::new()
            .with_axes(normalized.iter().map(|axis| *axis as i64))
            .with_keep_dims(keep_dims),
        PlannedValue::tensor(dtype, ValueShape::Dims(dims)),
    ))
}

/// `ArgMax`.
fn plan_argmax(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 1)?;
    let (_, shape) = input_tensor(ins, 0, "ArgMax data")?;
    let axis = attrs.int("axis")?.unwrap_or(0);
    let keep_dims = attrs.int("keepdims")?.unwrap_or(1) != 0;
    if attrs.int("select_last_index")?.unwrap_or(0) != 0 {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "select_last_index".to_string(),
            detail: "select_last_index=1 is not implemented by the fn3 argmax kernel".to_string(),
        });
    }
    let output = match shape.rank() {
        None => ValueShape::Unknown,
        Some(rank) => {
            let axis = normalize_axis(axis, rank, "ArgMax")?;
            let dims = shape
                .dims()
                .iter()
                .enumerate()
                .filter_map(|(position, dim)| {
                    if position == axis {
                        if keep_dims {
                            Some(Dim::Known(1))
                        } else {
                            None
                        }
                    } else {
                        Some(dim.clone())
                    }
                })
                .collect();
            ValueShape::Dims(dims)
        }
    };
    Ok(PlannedOp::single(
        OpId::ArgMax,
        vec![Operand::Input(0)],
        OpParams::new().with_axes([axis]).with_keep_dims(keep_dims),
        PlannedValue::tensor(DType::I64, output),
    ))
}

// ---------------------------------------------------------------------------
// Convolution, pooling, normalization and the remaining CNN operators
// ---------------------------------------------------------------------------

/// Parse the ONNX `auto_pad` attribute.
fn auto_pad_attribute(attrs: &mut Attrs<'_, '_>) -> Result<AutoPad, PlanError> {
    let Some(value) = attrs.string("auto_pad")? else {
        return Ok(AutoPad::NotSet);
    };
    match value {
        "NOTSET" => Ok(AutoPad::NotSet),
        "SAME_UPPER" => Ok(AutoPad::SameUpper),
        "SAME_LOWER" => Ok(AutoPad::SameLower),
        "VALID" => Ok(AutoPad::Valid),
        other => Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "auto_pad".to_string(),
            detail: format!(
                "`{other}` is not an ONNX auto_pad value; expected NOTSET, SAME_UPPER, \
                 SAME_LOWER or VALID"
            ),
        }),
    }
}

/// Parse a per-spatial-axis positive extent list.
fn positive_extents(values: &[i64], expected: usize, what: &str) -> Result<Vec<u64>, PlanError> {
    if values.len() != expected {
        return Err(shape_problem(format!(
            "{what} has {} extents for a {expected}-axis spatial rank",
            values.len()
        )));
    }
    let mut extents = Vec::with_capacity(expected);
    for &value in values {
        let extent = u64::try_from(value)
            .ok()
            .filter(|extent| *extent > 0)
            .ok_or_else(|| shape_problem(format!("{what} contains the non-positive {value}")))?;
        extents.push(extent);
    }
    Ok(extents)
}

/// A positive extent list with an ONNX default when it is absent.
fn positive_extents_or(
    values: Option<&[i64]>,
    expected: usize,
    default: u64,
    what: &str,
) -> Result<Vec<u64>, PlanError> {
    match values {
        Some(values) => positive_extents(values, expected, what),
        None => Ok(vec![default; expected]),
    }
}

/// Resolve explicit pads, or the pads `auto_pad` derives, as begin/end pairs.
fn window_pads(
    auto_pad: AutoPad,
    explicit: Option<&[i64]>,
    spatial: usize,
    input: &[Option<u64>],
    kernel: &[u64],
    strides: &[u64],
    dilations: &[u64],
    what: &str,
) -> Result<Vec<(i64, i64)>, PlanError> {
    match auto_pad {
        AutoPad::NotSet => {
            let Some(pads) = explicit else {
                return Ok(vec![(0, 0); spatial]);
            };
            if pads.len() != 2 * spatial {
                return Err(shape_problem(format!(
                    "{what} has {} pads for a {spatial}-axis spatial rank; ONNX supplies \
                     {spatial} begin extents followed by {spatial} end extents",
                    pads.len()
                )));
            }
            let mut pairs = Vec::with_capacity(spatial);
            for axis in 0..spatial {
                let (begin, end) = (pads[axis], pads[spatial + axis]);
                if begin < 0 || end < 0 {
                    return Err(shape_problem(format!(
                        "{what} has the negative pads ({begin}, {end}) on spatial axis {axis}; \
                         convolution and pooling pads are non-negative"
                    )));
                }
                pairs.push((begin, end));
            }
            Ok(pairs)
        }
        AutoPad::Valid => {
            if explicit.is_some() {
                return Err(shape_problem(format!(
                    "{what} sets auto_pad = VALID and explicit pads; ONNX forbids combining them"
                )));
            }
            Ok(vec![(0, 0); spatial])
        }
        upper @ (AutoPad::SameUpper | AutoPad::SameLower) => {
            if explicit.is_some() {
                return Err(shape_problem(format!(
                    "{what} sets auto_pad = {} and explicit pads; ONNX forbids combining them",
                    upper.name()
                )));
            }
            let mut pairs = Vec::with_capacity(spatial);
            for axis in 0..spatial {
                let Some(extent) = input[axis] else {
                    return Err(shape_problem(format!(
                        "{what} uses auto_pad = {} on a symbolic spatial axis {axis}; that \
                         needs the extent's value",
                        upper.name()
                    )));
                };
                let effective = dilations[axis] * (kernel[axis] - 1) + 1;
                let out = extent.div_ceil(strides[axis]);
                let total =
                    ((out.saturating_sub(1)) * strides[axis] + effective).saturating_sub(extent);
                let (begin, end) = if upper == AutoPad::SameUpper {
                    (total / 2, total - total / 2)
                } else {
                    (total - total / 2, total / 2)
                };
                pairs.push((
                    i64::try_from(begin).map_err(|_| {
                        shape_problem(format!("{what} auto_pad begin extent overflows i64"))
                    })?,
                    i64::try_from(end).map_err(|_| {
                        shape_problem(format!("{what} auto_pad end extent overflows i64"))
                    })?,
                ));
            }
            Ok(pairs)
        }
    }
}

/// One windowed output extent, or an unproven [`Dim`] for a symbolic input.
fn window_dim(
    input: &Dim,
    kernel: u64,
    stride: u64,
    dilation: u64,
    begin: i64,
    end: i64,
    ceil_mode: bool,
    what: &str,
) -> Dim {
    let Some(extent) = input.as_known() else {
        return Dim::unknown(format!("{what} of {input}"));
    };
    Dim::Known(window_extent(
        extent, kernel, stride, dilation, begin, end, ceil_mode,
    ))
}

/// The ONNX window formula: floor or ceil division, with pooling's last-window
/// adjustment.
fn window_extent(
    input: u64,
    kernel: u64,
    stride: u64,
    dilation: u64,
    begin: i64,
    end: i64,
    ceil_mode: bool,
) -> u64 {
    let effective = dilation * (kernel - 1) + 1;
    let total = i128::from(input) + i128::from(begin) + i128::from(end);
    let stride = i128::from(stride);
    let delta = total - i128::from(effective);
    let mut out = if ceil_mode {
        -((-delta).div_euclid(stride))
    } else {
        delta.div_euclid(stride)
    } + 1;
    if ceil_mode && out > 0 && (out - 1) * stride >= i128::from(input) + i128::from(begin) {
        out -= 1;
    }
    u64::try_from(out.max(0)).unwrap_or(0)
}

/// `Conv(data, weight, bias?)`.
fn plan_conv(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    if !(2..=3).contains(&ins.len()) {
        return Err(ImportProblemKind::OperandCount {
            expected: 3,
            found: ins.len(),
        });
    }
    let (dtype, shape) = input_tensor(ins, 0, "Conv data")?;
    let (weight_dtype, weight_shape) = input_tensor(ins, 1, "Conv weight")?;
    if dtype != weight_dtype {
        return Err(dtype_problem(format!(
            "Conv data is {dtype}, the weight is {weight_dtype}"
        )));
    }
    let auto_pad = auto_pad_attribute(attrs)?;
    let group = attrs.int("group")?.unwrap_or(1);
    if group <= 0 {
        return Err(shape_problem(format!("Conv group {group} is not positive")));
    }
    let kernel_attr = attrs.ints("kernel_shape")?.map(<[i64]>::to_vec);
    let dilations_attr = attrs.ints("dilations")?.map(<[i64]>::to_vec);
    let strides_attr = attrs.ints("strides")?.map(<[i64]>::to_vec);
    let pads_attr = attrs.ints("pads")?.map(<[i64]>::to_vec);
    let Some(rank) = shape.rank() else {
        return Ok(PlannedOp::single(
            OpId::Conv,
            (0..ins.len()).map(Operand::Input).collect(),
            OpParams::new().with_group(group).with_auto_pad(auto_pad),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    if rank < 3 {
        return Err(shape_problem(format!(
            "Conv needs an [N, C, spatial...] operand, found {}",
            shape.render()
        )));
    }
    let spatial = rank - 2;
    let kernel = match &kernel_attr {
        Some(values) => positive_extents(values, spatial, "Conv kernel_shape")?,
        None => {
            if weight_shape.rank().is_none() {
                return Err(shape_problem("Conv weight rank is unknown".to_string()));
            }
            let mut kernel = Vec::with_capacity(spatial);
            for (axis, dim) in weight_shape.dims().iter().skip(2).enumerate() {
                let Some(extent) = dim.as_known().filter(|extent| *extent > 0) else {
                    return Err(shape_problem(format!(
                        "Conv weight kernel axis {axis} is {dim}; kernel_shape is absent and the \
                         weight does not pin it"
                    )));
                };
                kernel.push(extent);
            }
            if kernel.len() != spatial {
                return Err(shape_problem(format!(
                    "Conv weight {} does not have {spatial} kernel axes",
                    weight_shape.render()
                )));
            }
            kernel
        }
    };
    let strides = positive_extents_or(strides_attr.as_deref(), spatial, 1, "Conv strides")?;
    let dilations = positive_extents_or(dilations_attr.as_deref(), spatial, 1, "Conv dilations")?;
    let spatial_input: Vec<Option<u64>> =
        shape.dims()[2..].iter().map(|dim| dim.as_known()).collect();
    let pads = window_pads(
        auto_pad,
        pads_attr.as_deref(),
        spatial,
        &spatial_input,
        &kernel,
        &strides,
        &dilations,
        "Conv",
    )?;
    let in_channels = shape.dims()[1].as_known().ok_or_else(|| {
        shape_problem(format!(
            "Conv input channels are {}; this importer needs a constant channel count",
            shape.dims()[1]
        ))
    })?;
    let group_u64 = group as u64;
    let weight_in = weight_shape.dims().get(1).and_then(Dim::as_known);
    if let Some(weight_in) = weight_in {
        if weight_in.saturating_mul(group_u64) != in_channels {
            return Err(shape_problem(format!(
                "Conv weight has {weight_in} input channels for group = {group}, but the data \
                 has {in_channels}; expected {}",
                in_channels / group_u64
            )));
        }
    }
    let out_channels = weight_shape.dims().first().and_then(Dim::as_known);
    let Some(out_channels) = out_channels else {
        return Err(shape_problem(format!(
            "Conv weight {} has no constant output channel count",
            weight_shape.render()
        )));
    };
    if out_channels % group_u64 != 0 {
        return Err(shape_problem(format!(
            "Conv output channels {out_channels} are not divisible by group {group}"
        )));
    }
    for (axis, extent) in kernel.iter().enumerate() {
        if let Some(declared) = weight_shape.dims().get(axis + 2).and_then(Dim::as_known) {
            if declared != *extent {
                return Err(shape_problem(format!(
                    "Conv kernel_shape says {extent} on spatial axis {axis}, the weight declares \
                     {declared}"
                )));
            }
        }
    }
    if ins.len() == 3 {
        let (bias_dtype, bias_shape) = input_tensor(ins, 2, "Conv bias")?;
        if bias_dtype != dtype {
            return Err(dtype_problem(format!(
                "Conv bias is {bias_dtype}, the data is {dtype}"
            )));
        }
        if bias_shape.rank() != Some(1)
            || bias_shape.dims().first().and_then(Dim::as_known) != Some(out_channels)
        {
            return Err(shape_problem(format!(
                "Conv bias must be rank 1 [{out_channels}], found {}",
                bias_shape.render()
            )));
        }
    }
    let mut dims = Vec::with_capacity(rank);
    dims.push(shape.dims()[0].clone());
    dims.push(Dim::Known(out_channels));
    for axis in 0..spatial {
        let (begin, end) = pads[axis];
        dims.push(window_dim(
            &shape.dims()[axis + 2],
            kernel[axis],
            strides[axis],
            dilations[axis],
            begin,
            end,
            false,
            "Conv output extent",
        ));
    }
    let params = OpParams::new()
        .with_kernel_shape(kernel.iter().map(|extent| *extent as i64))
        .with_strides(strides.iter().map(|extent| *extent as i64))
        .with_dilations(dilations.iter().map(|extent| *extent as i64))
        .with_pads(pads_attr.unwrap_or_default())
        .with_group(group)
        .with_auto_pad(auto_pad);
    Ok(PlannedOp::single(
        OpId::Conv,
        (0..ins.len()).map(Operand::Input).collect(),
        params,
        PlannedValue::tensor(dtype, ValueShape::Dims(dims)),
    ))
}

/// `MaxPool`/`AveragePool`.
fn plan_pool(
    ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
    output_count: usize,
    op: OpId,
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 1)?;
    if output_count != 1 {
        return Err(ImportProblemKind::UnsupportedOutputCount {
            found: output_count,
            expected: 1,
        });
    }
    let (dtype, shape) = input_tensor(ins, 0, "pool data")?;
    let auto_pad = auto_pad_attribute(attrs)?;
    let kernel_attr = attrs.ints("kernel_shape")?.map(<[i64]>::to_vec);
    let strides_attr = attrs.ints("strides")?.map(<[i64]>::to_vec);
    // `dilations`, `storage_order` and `count_include_pad` are MaxPool-only
    // attributes. Reading them for AveragePool would silently accept a field
    // ONNX does not define for it, so they are read only where they exist and
    // an AveragePool that carries one is reported as an unknown attribute.
    let dilations_attr = if op == OpId::MaxPool {
        attrs.ints("dilations")?.map(<[i64]>::to_vec)
    } else {
        None
    };
    let pads_attr = attrs.ints("pads")?.map(<[i64]>::to_vec);
    let ceil_mode = attrs.int("ceil_mode")?.unwrap_or(0) != 0;
    let count_include_pad = if op == OpId::AveragePool {
        attrs.int("count_include_pad")?.unwrap_or(0) != 0
    } else {
        false
    };
    if op == OpId::MaxPool {
        if let Some(order) = attrs.int("storage_order")? {
            if order != 0 {
                return Err(ImportProblemKind::UnsupportedAttribute {
                    attribute: "storage_order".to_string(),
                    detail: "storage_order = 1 orders the MaxPool indices column-major; this \
                             build does not map the indices output"
                        .to_string(),
                });
            }
        }
    }
    let Some(rank) = shape.rank() else {
        return Ok(PlannedOp::single(
            op,
            vec![Operand::Input(0)],
            OpParams::new()
                .with_auto_pad(auto_pad)
                .with_ceil_mode(ceil_mode),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    if rank < 3 {
        return Err(shape_problem(format!(
            "{} needs an [N, C, spatial...] operand, found {}",
            op,
            shape.render()
        )));
    }
    let spatial = rank - 2;
    let kernel = match &kernel_attr {
        Some(values) => positive_extents(values, spatial, "kernel_shape")?,
        None => {
            return Err(shape_problem(format!(
                "{op} needs the kernel_shape attribute"
            )))
        }
    };
    let strides = positive_extents_or(strides_attr.as_deref(), spatial, 1, "strides")?;
    let dilations = positive_extents_or(dilations_attr.as_deref(), spatial, 1, "dilations")?;
    let spatial_input: Vec<Option<u64>> =
        shape.dims()[2..].iter().map(|dim| dim.as_known()).collect();
    let pads = window_pads(
        auto_pad,
        pads_attr.as_deref(),
        spatial,
        &spatial_input,
        &kernel,
        &strides,
        &dilations,
        ctx.op_type.as_str(),
    )?;
    let mut dims = Vec::with_capacity(rank);
    dims.push(shape.dims()[0].clone());
    dims.push(shape.dims()[1].clone());
    for axis in 0..spatial {
        let (begin, end) = pads[axis];
        dims.push(window_dim(
            &shape.dims()[axis + 2],
            kernel[axis],
            strides[axis],
            dilations[axis],
            begin,
            end,
            ceil_mode,
            "pool output extent",
        ));
    }
    let params = OpParams::new()
        .with_kernel_shape(kernel.iter().map(|extent| *extent as i64))
        .with_strides(strides.iter().map(|extent| *extent as i64))
        .with_dilations(dilations.iter().map(|extent| *extent as i64))
        .with_pads(pads_attr.unwrap_or_default())
        .with_count_include_pad(count_include_pad)
        .with_auto_pad(auto_pad)
        .with_ceil_mode(ceil_mode);
    Ok(PlannedOp::single(
        op,
        vec![Operand::Input(0)],
        params,
        PlannedValue::tensor(dtype, ValueShape::Dims(dims)),
    ))
}

/// `GlobalAveragePool`.
fn plan_global_average_pool(
    _ctx: &NodeRef,
    _attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    expect_inputs(ins, 1)?;
    let (dtype, shape) = input_tensor(ins, 0, "GlobalAveragePool data")?;
    let Some(rank) = shape.rank() else {
        return Ok(PlannedOp::single(
            OpId::GlobalAveragePool,
            vec![Operand::Input(0)],
            OpParams::new(),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    if rank < 3 {
        return Err(shape_problem(format!(
            "GlobalAveragePool needs an [N, C, spatial...] operand, found {}",
            shape.render()
        )));
    }
    let mut dims = Vec::with_capacity(rank);
    dims.push(shape.dims()[0].clone());
    dims.push(shape.dims()[1].clone());
    dims.extend(std::iter::repeat_n(Dim::Known(1), rank - 2));
    Ok(PlannedOp::single(
        OpId::GlobalAveragePool,
        vec![Operand::Input(0)],
        OpParams::new(),
        PlannedValue::tensor(dtype, ValueShape::Dims(dims)),
    ))
}

/// `BatchNormalization` in its inference form.
fn plan_batch_norm(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
    output_count: usize,
) -> Result<Vec<PlannedOp>, PlanError> {
    if ins.len() != 5 {
        return Err(ImportProblemKind::OperandCount {
            expected: 5,
            found: ins.len(),
        });
    }
    if output_count != 1 {
        return Err(ImportProblemKind::UnsupportedOutputCount {
            found: output_count,
            expected: 1,
        });
    }
    let (dtype, shape) = input_tensor(ins, 0, "BatchNormalization data")?;
    let epsilon = attrs.float("epsilon")?.unwrap_or(1e-5);
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(shape_problem(format!(
            "BatchNormalization epsilon {epsilon} must be finite and non-negative"
        )));
    }
    // `momentum` only affects training; ONNX defines it as unused at inference.
    let _momentum = attrs.float("momentum")?;
    if attrs.int("training_mode")?.unwrap_or(0) != 0 {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "training_mode".to_string(),
            detail: "training_mode = 1 produces running-statistic outputs; this is an \
                     inference-only runtime"
                .to_string(),
        });
    }
    let channels = shape.dims().get(1).cloned();
    for (index, name) in ["scale", "B", "mean", "var"].iter().enumerate() {
        let (parameter_dtype, parameter_shape) =
            input_tensor(ins, index + 1, "BatchNorm parameter")?;
        if parameter_dtype != dtype {
            return Err(dtype_problem(format!(
                "BatchNormalization {name} is {parameter_dtype}, the data is {dtype}"
            )));
        }
        if parameter_shape.rank() != Some(1) || parameter_shape.dims().first() != channels.as_ref()
        {
            return Err(shape_problem(format!(
                "BatchNormalization {name} must be rank 1 matching the channel axis, found {}",
                parameter_shape.render()
            )));
        }
    }
    Ok(PlannedOp::single(
        OpId::BatchNorm,
        (0..5).map(Operand::Input).collect(),
        OpParams::new().with_epsilon(epsilon),
        PlannedValue::tensor(dtype, shape),
    ))
}

/// Parse a `Resize` mode string.
fn resize_mode_attribute(attrs: &mut Attrs<'_, '_>) -> Result<ResizeMode, PlanError> {
    match attrs.string("mode")?.unwrap_or("nearest") {
        "nearest" => Ok(ResizeMode::Nearest),
        "linear" => Ok(ResizeMode::Linear),
        "cubic" => Ok(ResizeMode::Cubic),
        other => Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "mode".to_string(),
            detail: format!(
                "`{other}` is not an ONNX Resize mode; expected nearest, linear or cubic"
            ),
        }),
    }
}

/// `Resize(data, roi?, scales?, sizes?)`.
fn plan_resize(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    if !(1..=4).contains(&ins.len()) {
        return Err(ImportProblemKind::OperandCount {
            expected: 3,
            found: ins.len(),
        });
    }
    let (dtype, shape) = input_tensor(ins, 0, "Resize data")?;
    let resize_mode = resize_mode_attribute(attrs)?;
    let coordinate_transformation_mode = match attrs
        .string("coordinate_transformation_mode")?
        .unwrap_or("half_pixel")
    {
        "half_pixel" => CoordinateTransformationMode::HalfPixel,
        "pytorch_half_pixel" => CoordinateTransformationMode::PytorchHalfPixel,
        "align_corners" => CoordinateTransformationMode::AlignCorners,
        "asymmetric" => CoordinateTransformationMode::Asymmetric,
        "tf_crop_and_resize" => CoordinateTransformationMode::TfCropAndResize,
        other => {
            return Err(ImportProblemKind::UnsupportedAttribute {
                attribute: "coordinate_transformation_mode".to_string(),
                detail: format!(
                    "`{other}` is not an ONNX coordinate transformation mode; expected \
                     half_pixel, pytorch_half_pixel, align_corners, asymmetric or \
                     tf_crop_and_resize"
                ),
            })
        }
    };
    let nearest_mode = match attrs
        .string("nearest_mode")?
        .unwrap_or("round_prefer_floor")
    {
        "round_prefer_floor" => NearestMode::RoundPreferFloor,
        "round_prefer_ceil" => NearestMode::RoundPreferCeil,
        "floor" => NearestMode::Floor,
        "ceil" => NearestMode::Ceil,
        other => {
            return Err(ImportProblemKind::UnsupportedAttribute {
                attribute: "nearest_mode".to_string(),
                detail: format!(
                    "`{other}` is not an ONNX nearest mode; expected round_prefer_floor, \
                     round_prefer_ceil, floor or ceil"
                ),
            })
        }
    };
    // `cubic_coeff_a` is emitted by PyTorch even for nearest/linear and only
    // affects cubic mode; consume it so it is not reported as unknown.
    let _cubic_coeff_a = attrs.float("cubic_coeff_a")?;
    if attrs.int("exclude_outside")?.unwrap_or(0) != 0 {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "exclude_outside".to_string(),
            detail: "exclude_outside = 1 is only defined for cubic interpolation".to_string(),
        });
    }
    let _extrapolation_value = attrs.float("extrapolation_value")?;
    if resize_mode == ResizeMode::Cubic {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "mode".to_string(),
            detail: "cubic Resize has no shape rule or kernel in this revision".to_string(),
        });
    }
    if coordinate_transformation_mode == CoordinateTransformationMode::TfCropAndResize {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "coordinate_transformation_mode".to_string(),
            detail: "tf_crop_and_resize is an ROI crop, which this runtime does not represent"
                .to_string(),
        });
    }
    // `roi` is read and then ignored, which is what the operator says it is:
    // ONNX defines its values only for `tf_crop_and_resize`, and that mode is
    // rejected above. An exporter that writes the placeholder as four zeros
    // instead of an empty tensor therefore describes the same resize as one
    // that omits it. The shape is still validated so a mis-typed operand is
    // reported here rather than silently dropped. A two-operand node has no
    // roi: ONNX requires exactly one of scales/sizes, so a lone roi is not a
    // node this importer has to read.
    if ins.len() >= 3 {
        let _ = input_tensor(ins, 1, "Resize roi")?;
    }
    // `X, roi?, scales?, sizes?` with at most one of scales/sizes. Empty
    // optional inputs are dropped while the operand list is built, so the
    // length says how many are present, not their positions: two operands mean
    // `X` and one target, three mean `X`, a roi and one target. The target's
    // dtype decides whether it is f32 scales or i64 sizes.
    let (scales_index, sizes_index) = match ins.len() {
        4 => (Some(2), Some(3)),
        present @ (2 | 3) => {
            let index = present - 1;
            let (target_dtype, _, _) = input_series(ins, index, "Resize target operand")?;
            if target_dtype == DType::I64 {
                (None, Some(index))
            } else {
                (Some(index), None)
            }
        }
        _ => (None, None),
    };
    let mut scales: Vec<f32> = Vec::new();
    if let Some(index) = scales_index {
        let (scales_dtype, _, series) = input_series(ins, index, "Resize scales")?;
        if scales_dtype != DType::F32 {
            return Err(dtype_problem(format!(
                "Resize scales are {scales_dtype}, expected f32"
            )));
        }
        let items = series
            .items
            .as_ref()
            .ok_or_else(|| shape_problem("Resize scales are not statically known".to_string()))?;
        scales = items
            .iter()
            .map(|atom| {
                atom.as_f32()
                    .filter(|scale| scale.is_finite() && *scale > 0.0)
                    .ok_or_else(|| {
                        shape_problem(format!(
                            "Resize scale {} is not a positive finite f32",
                            atom.render()
                        ))
                    })
            })
            .collect::<Result<Vec<f32>, PlanError>>()?;
    }
    let mut sizes: Vec<i64> = Vec::new();
    if let Some(index) = sizes_index {
        let (sizes_dtype, _, series) = input_series(ins, index, "Resize sizes")?;
        if sizes_dtype != DType::I64 {
            return Err(ImportProblemKind::IndexDType {
                actual: sizes_dtype,
                operand: "Resize sizes".to_string(),
            });
        }
        sizes = int_list(&series, "Resize sizes")?;
    }
    if !scales.is_empty() && !sizes.is_empty() {
        return Err(shape_problem(
            "Resize supplies both scales and sizes; ONNX allows exactly one".to_string(),
        ));
    }
    if scales.is_empty() && sizes.is_empty() {
        return Err(shape_problem(
            "Resize supplies neither scales nor sizes; this importer needs one constant form"
                .to_string(),
        ));
    }
    let Some(rank) = shape.rank() else {
        return Ok(PlannedOp::single(
            OpId::Resize,
            (0..ins.len()).map(Operand::Input).collect(),
            OpParams::new()
                .with_resize(resize_mode, coordinate_transformation_mode, nearest_mode)
                .with_resize_scales(scales)
                .with_resize_sizes(sizes),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    let output = if !sizes.is_empty() {
        if sizes.len() != rank {
            return Err(shape_problem(format!(
                "Resize sizes has {} extents for a rank-{rank} operand",
                sizes.len()
            )));
        }
        ValueShape::Dims(
            sizes
                .iter()
                .map(|size| {
                    u64::try_from(*size)
                        .ok()
                        .filter(|size| *size > 0)
                        .map(Dim::Known)
                        .ok_or_else(|| shape_problem(format!("Resize size {size} is not positive")))
                })
                .collect::<Result<Vec<Dim>, PlanError>>()?,
        )
    } else {
        if scales.len() != rank {
            return Err(shape_problem(format!(
                "Resize scales has {} extents for a rank-{rank} operand",
                scales.len()
            )));
        }
        ValueShape::Dims(
            shape
                .dims()
                .iter()
                .zip(&scales)
                .map(|(dim, scale)| match dim.as_known() {
                    Some(extent) => Dim::Known((extent as f64 * f64::from(*scale)).floor() as u64),
                    None => Dim::unknown(format!("resize of {dim} by {scale}")),
                })
                .collect(),
        )
    };
    let params = OpParams::new()
        .with_resize(resize_mode, coordinate_transformation_mode, nearest_mode)
        .with_resize_scales(scales)
        .with_resize_sizes(sizes);
    Ok(PlannedOp::single(
        OpId::Resize,
        (0..ins.len()).map(Operand::Input).collect(),
        params,
        PlannedValue::tensor(dtype, output),
    ))
}

/// `Pad(data, pads?, constant_value?)`.
fn plan_pad(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    if !(1..=3).contains(&ins.len()) {
        return Err(ImportProblemKind::OperandCount {
            expected: 3,
            found: ins.len(),
        });
    }
    let (dtype, shape) = input_tensor(ins, 0, "Pad data")?;
    let mode = match attrs.string("mode")?.unwrap_or("constant") {
        "constant" => PadMode::Constant,
        "reflect" => PadMode::Reflect,
        "edge" => PadMode::Edge,
        "wrap" => PadMode::Wrap,
        other => {
            return Err(ImportProblemKind::UnsupportedAttribute {
                attribute: "mode".to_string(),
                detail: format!(
                    "`{other}` is not an ONNX Pad mode; expected constant, reflect, edge or wrap"
                ),
            })
        }
    };
    if mode == PadMode::Wrap {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "mode".to_string(),
            detail: "wrap padding is not implemented; constant, reflect and edge are".to_string(),
        });
    }
    let attribute_pads = attrs.ints("pads")?.map(<[i64]>::to_vec);
    if attribute_pads.is_some() && ins.len() >= 2 {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "pads".to_string(),
            detail: "the node supplies both the opset-2 `pads` attribute and an opset-11 pads \
                     input"
                .to_string(),
        });
    }
    let pads: Vec<i64> = match (&attribute_pads, ins.len()) {
        (Some(values), _) => values.clone(),
        (None, 2..) => {
            let (pads_dtype, _, series) = input_series(ins, 1, "Pad pads")?;
            if pads_dtype != DType::I64 {
                return Err(ImportProblemKind::IndexDType {
                    actual: pads_dtype,
                    operand: "Pad pads".to_string(),
                });
            }
            int_list(&series, "Pad pads")?
        }
        _ => Vec::new(),
    };
    if pads.is_empty() {
        return Err(shape_problem(
            "Pad has no pads; a Pad node without pads is not representable".to_string(),
        ));
    }
    let mut pad_value = attrs.float("value")?.unwrap_or(0.0);
    if ins.len() >= 3 {
        let (value_dtype, _, series) = input_series(ins, 2, "Pad constant_value")?;
        if value_dtype != DType::F32 {
            return Err(dtype_problem(format!(
                "Pad constant_value is {value_dtype}, expected f32"
            )));
        }
        pad_value = series
            .only()
            .and_then(Atom::as_f32)
            .ok_or_else(|| shape_problem("Pad constant_value is not a known scalar".to_string()))?;
    }
    let Some(rank) = shape.rank() else {
        return Ok(PlannedOp::single(
            OpId::Pad,
            (0..ins.len()).map(Operand::Input).collect(),
            OpParams::new().with_pads(pads).with_pad(mode, pad_value),
            PlannedValue::tensor(dtype, ValueShape::Unknown),
        ));
    };
    if pads.len() != 2 * rank {
        return Err(shape_problem(format!(
            "Pad has {} pad extents for a rank-{rank} operand; ONNX supplies {rank} begin \
             extents followed by {rank} end extents",
            pads.len()
        )));
    }
    let mut dims = Vec::with_capacity(rank);
    for axis in 0..rank {
        let begin = pads[axis];
        let end = pads[rank + axis];
        match shape.dims()[axis].as_known() {
            Some(extent) => {
                let padded = extent as i128 + i128::from(begin) + i128::from(end);
                if padded < 0 {
                    return Err(shape_problem(format!(
                        "Pad crops axis {axis} of extent {extent} by {begin} + {end}, leaving \
                         {padded} elements"
                    )));
                }
                dims.push(Dim::Known(padded as u64));
            }
            None => dims.push(Dim::unknown(format!(
                "padding {begin} + {end} of {}",
                shape.dims()[axis]
            ))),
        }
    }
    Ok(PlannedOp::single(
        OpId::Pad,
        (0..ins.len()).map(Operand::Input).collect(),
        OpParams::new().with_pads(pads).with_pad(mode, pad_value),
        PlannedValue::tensor(dtype, ValueShape::Dims(dims)),
    ))
}

/// `Clip(data, min?, max?)`.
fn plan_clip(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
) -> Result<Vec<PlannedOp>, PlanError> {
    if !(1..=3).contains(&ins.len()) {
        return Err(ImportProblemKind::OperandCount {
            expected: 3,
            found: ins.len(),
        });
    }
    let (dtype, shape) = input_tensor(ins, 0, "Clip data")?;
    let attribute_min = attrs.float("min")?;
    let attribute_max = attrs.float("max")?;
    if (attribute_min.is_some() || attribute_max.is_some()) && ins.len() > 1 {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "min/max".to_string(),
            detail: "the node supplies both the opset-6 min/max attributes and the opset-11 \
                     min/max inputs"
                .to_string(),
        });
    }
    let mut clip_min = attribute_min;
    let mut clip_max = attribute_max;
    if ins.len() >= 2 {
        let (min_dtype, _, series) = input_series(ins, 1, "Clip min")?;
        if min_dtype != DType::F32 {
            return Err(dtype_problem(format!(
                "Clip min is {min_dtype}, expected f32"
            )));
        }
        clip_min = Some(
            series
                .only()
                .and_then(Atom::as_f32)
                .ok_or_else(|| shape_problem("Clip min is not a known scalar".to_string()))?,
        );
    }
    if ins.len() >= 3 {
        let (max_dtype, _, series) = input_series(ins, 2, "Clip max")?;
        if max_dtype != DType::F32 {
            return Err(dtype_problem(format!(
                "Clip max is {max_dtype}, expected f32"
            )));
        }
        clip_max = Some(
            series
                .only()
                .and_then(Atom::as_f32)
                .ok_or_else(|| shape_problem("Clip max is not a known scalar".to_string()))?,
        );
    }
    if let (Some(min), Some(max)) = (clip_min, clip_max) {
        if min > max {
            return Err(shape_problem(format!("Clip min {min} is above max {max}")));
        }
    }
    Ok(PlannedOp::single(
        OpId::Clip,
        (0..ins.len()).map(Operand::Input).collect(),
        OpParams::new().with_clip(clip_min, clip_max),
        PlannedValue::tensor(dtype, shape),
    ))
}

/// `TopK(data, k?)`.
fn plan_top_k(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
    output_count: usize,
) -> Result<Vec<PlannedOp>, PlanError> {
    if !(1..=2).contains(&ins.len()) {
        return Err(ImportProblemKind::OperandCount {
            expected: 2,
            found: ins.len(),
        });
    }
    if output_count != 2 {
        return Err(ImportProblemKind::UnsupportedOutputCount {
            found: output_count,
            expected: 2,
        });
    }
    let (dtype, shape) = input_tensor(ins, 0, "TopK data")?;
    let axis = attrs.int("axis")?.unwrap_or(-1);
    let largest = attrs.int("largest")?.unwrap_or(1) != 0;
    let sorted = attrs.int("sorted")?.unwrap_or(1) != 0;
    let attribute_k = attrs.int("k")?;
    if attribute_k.is_some() && ins.len() == 2 {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "k".to_string(),
            detail: "the node supplies both the opset-1 `k` attribute and an opset-10 k input"
                .to_string(),
        });
    }
    let k = match (attribute_k, ins.len()) {
        (Some(k), _) => k,
        (None, 2) => {
            let (k_dtype, _, series) = input_series(ins, 1, "TopK k")?;
            if k_dtype != DType::I64 {
                return Err(ImportProblemKind::IndexDType {
                    actual: k_dtype,
                    operand: "TopK k".to_string(),
                });
            }
            int_list(&series, "TopK k")?
                .first()
                .copied()
                .ok_or_else(|| shape_problem("TopK k is empty".to_string()))?
        }
        (None, _) => {
            return Err(shape_problem(
                "TopK has no k attribute and no k input".to_string(),
            ))
        }
    };
    if k < 0 {
        return Err(shape_problem(format!("TopK k {k} is negative")));
    }
    let Some(rank) = shape.rank() else {
        return Ok(vec![PlannedOp {
            op: OpId::TopK,
            operands: (0..ins.len()).map(Operand::Input).collect(),
            params: OpParams::new().with_topk(axis, k, largest, sorted),
            typed: None,
            outputs: vec![
                PlannedValue::tensor(dtype, ValueShape::Unknown),
                PlannedValue::tensor(DType::I64, ValueShape::Unknown),
            ],
        }]);
    };
    let axis = normalize_axis(axis, rank, "TopK")?;
    let mut dims = shape.dims().to_vec();
    dims[axis] = Dim::Known(k as u64);
    Ok(vec![PlannedOp {
        op: OpId::TopK,
        operands: (0..ins.len()).map(Operand::Input).collect(),
        params: OpParams::new().with_topk(axis as i64, k, largest, sorted),
        typed: None,
        outputs: vec![
            PlannedValue::tensor(dtype, ValueShape::Dims(dims.clone())),
            PlannedValue::tensor(DType::I64, ValueShape::Dims(dims)),
        ],
    }])
}

/// Slice one tracked sequence along axis 0.
fn fold_slice(
    series: &AtomSeries,
    start: i64,
    end: i64,
    step: i64,
    extent: Option<u64>,
) -> Option<Vec<Atom>> {
    let items = series.items.as_ref()?;
    let extent = extent? as i64;
    if extent as usize != items.len() || step == 0 {
        return None;
    }
    let start = clamp_index(start, extent);
    let end = clamp_index(end, extent);
    let mut selected = Vec::new();
    let mut position = start;
    while (step > 0 && position < end) || (step < 0 && position > end) {
        let index = usize::try_from(position).ok()?;
        selected.push(items.get(index)?.clone());
        position += step;
    }
    Some(selected)
}

/// `Split` (the only mapped operator with more than one output).
///
/// Opset 12 carries the sizes in the `split` attribute, opset 13 in an optional
/// second input, and opset 18 adds `num_outputs` for an equal split. All three
/// forms are accepted; the resolved sizes are exported to the typed IR because
/// its shape pass reads attributes, never tensor payloads.
fn plan_split_full(
    _ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
    output_count: usize,
) -> Result<Vec<PlannedOp>, PlanError> {
    if ins.is_empty() || ins.len() > 2 {
        return Err(ImportProblemKind::OperandCount {
            expected: 2,
            found: ins.len(),
        });
    }
    if output_count == 0 {
        return Err(ImportProblemKind::UnsupportedOutputCount {
            found: 0,
            expected: 1,
        });
    }
    let (dtype, shape) = input_tensor(ins, 0, "Split data")?;
    let axis = attrs.int("axis")?.unwrap_or(0);
    if let Some(declared) = attrs.int("num_outputs")? {
        if declared != output_count as i64 {
            return Err(shape_problem(format!(
                "Split declares num_outputs = {declared} but has {output_count} outputs"
            )));
        }
    }
    let attribute_sizes = attrs.ints("split")?.map(<[i64]>::to_vec);
    if attribute_sizes.is_some() && ins.len() == 2 {
        return Err(ImportProblemKind::UnsupportedAttribute {
            attribute: "split".to_string(),
            detail: "the node supplies both the opset-12 `split` attribute and an opset-13 \
                     split input"
                .to_string(),
        });
    }
    let Some(rank) = shape.rank() else {
        return Ok(vec![PlannedOp {
            op: OpId::Split,
            operands: (0..ins.len()).map(Operand::Input).collect(),
            params: OpParams::new().with_axes([axis]),
            typed: None,
            outputs: (0..output_count)
                .map(|_| PlannedValue::tensor(dtype, ValueShape::Unknown))
                .collect(),
        }]);
    };
    let axis = normalize_axis(axis, rank, "Split")?;
    let extent = shape.dims()[axis].as_known();
    let raw_sizes: Option<Vec<i64>> = match (ins.len(), attribute_sizes) {
        (2, _) => {
            let (sizes_dtype, _, series) = input_series(ins, 1, "Split sizes")?;
            if sizes_dtype != DType::I64 {
                return Err(ImportProblemKind::IndexDType {
                    actual: sizes_dtype,
                    operand: "Split sizes".to_string(),
                });
            }
            int_list(&series, "Split sizes").ok()
        }
        (_, Some(values)) => Some(values),
        _ => None,
    };
    let sizes: Option<Vec<u64>> = match &raw_sizes {
        Some(values) => {
            let mut sizes = Vec::with_capacity(values.len());
            for value in values {
                match u64::try_from(*value) {
                    Ok(value) => sizes.push(value),
                    Err(_) => return Err(shape_problem(format!("Split size {value} is negative"))),
                }
            }
            Some(sizes)
        }
        None => None,
    };
    if let Some(sizes) = &sizes {
        if sizes.len() != output_count {
            return Err(shape_problem(format!(
                "Split has {} sizes for {output_count} outputs",
                sizes.len()
            )));
        }
        if let Some(extent) = extent {
            let total: u64 = sizes.iter().sum();
            if total != extent {
                return Err(shape_problem(format!(
                    "Split sizes {sizes:?} sum to {total}, but axis {axis} has extent {extent}"
                )));
            }
        }
    }
    let outputs = (0..output_count)
        .map(|position| {
            let mut dims = shape.dims().to_vec();
            dims[axis] = match &sizes {
                Some(sizes) => Dim::Known(sizes[position]),
                None => match extent {
                    Some(extent) if extent % output_count as u64 == 0 => {
                        Dim::Known(extent / output_count as u64)
                    }
                    _ => Dim::unknown(format!("equal split of {extent:?} into {output_count}")),
                },
            };
            PlannedValue::tensor(dtype, ValueShape::Dims(dims))
        })
        .collect();
    let mut typed = OpParams::new().with_axes([axis as i64]);
    if let Some(sizes) = &sizes {
        typed = typed.with_split_sizes(sizes.iter().map(|size| *size as i64));
    }
    Ok(vec![PlannedOp {
        op: OpId::Split,
        operands: (0..ins.len()).map(Operand::Input).collect(),
        params: OpParams::new().with_axes([axis as i64]),
        typed: Some(typed),
        outputs,
    }])
}

/// Plan one node, reporting every problem it has.
fn plan_node(
    ctx: &NodeRef,
    node: &proto::Node<'_>,
    inputs: &[Abstract],
    output_count: usize,
    diagnostics: &mut Diagnostics,
) -> Option<Vec<PlannedOp>> {
    let mut attrs = Attrs::new(&node.attributes);
    let op_type = node.op_type.to_string();
    let result = dispatch(ctx, &mut attrs, inputs, output_count);
    let mut problems = std::mem::take(&mut attrs.problems);
    match result {
        Ok(planned) => {
            problems.extend(attrs.finish(Some(accepted_attributes(&op_type))));
            for problem in problems {
                diagnostics.push(Some(ctx.clone()), problem);
            }
            Some(planned)
        }
        Err(error) => {
            problems.push(error);
            // Keep the payload problems, skip the unknown-attribute scan: the
            // node already failed for a more specific reason.
            problems.extend(attrs.finish(None));
            for problem in problems {
                diagnostics.push(Some(ctx.clone()), problem);
            }
            None
        }
    }
}

/// Dispatch on the ONNX operator type.
fn dispatch(
    ctx: &NodeRef,
    attrs: &mut Attrs<'_, '_>,
    ins: &[Abstract],
    output_count: usize,
) -> Result<Vec<PlannedOp>, PlanError> {
    match ctx.op_type.as_str() {
        "Gather" => plan_gather(ctx, attrs, ins),
        "Transpose" => plan_transpose(ctx, attrs, ins),
        "Reshape" => plan_reshape(ctx, attrs, ins),
        "Flatten" => plan_flatten(ctx, attrs, ins),
        "Squeeze" => plan_squeeze(ctx, attrs, ins),
        "Unsqueeze" => plan_unsqueeze(ctx, attrs, ins),
        "Concat" => plan_concat(ctx, attrs, ins),
        "Slice" => plan_slice(ctx, attrs, ins),
        "Split" => plan_split_full(ctx, attrs, ins, output_count),
        "Expand" => plan_expand(ctx, attrs, ins),
        "Shape" => plan_shape(ctx, attrs, ins),
        "ConstantOfShape" => plan_constant_of_shape(ctx, attrs, ins),
        "Range" => plan_range(ctx, attrs, ins),
        "Cast" => plan_cast(ctx, attrs, ins),
        "Where" => plan_where(ctx, attrs, ins),
        "Equal" => plan_compare(ctx, attrs, ins, OpId::Equal),
        "Less" => plan_compare(ctx, attrs, ins, OpId::Less),
        "Greater" => plan_compare(ctx, attrs, ins, OpId::Greater),
        "Trilu" => plan_trilu(ctx, attrs, ins),
        "ScatterND" => plan_scatter_nd(ctx, attrs, ins),
        "Add" => plan_binary(ctx, attrs, ins, OpId::Add),
        "Sub" => plan_binary(ctx, attrs, ins, OpId::Sub),
        "Mul" => plan_binary(ctx, attrs, ins, OpId::Mul),
        "Div" => plan_binary(ctx, attrs, ins, OpId::Div),
        "Pow" => plan_binary(ctx, attrs, ins, OpId::Pow),
        "Sqrt" => plan_unary(ctx, attrs, ins, OpId::Sqrt),
        "Erf" => plan_unary(ctx, attrs, ins, OpId::Erf),
        "Tanh" => plan_unary(ctx, attrs, ins, OpId::Tanh),
        "Relu" => plan_unary(ctx, attrs, ins, OpId::Relu),
        "Neg" => plan_unary(ctx, attrs, ins, OpId::Neg),
        "Sin" => plan_unary(ctx, attrs, ins, OpId::Sin),
        "Cos" => plan_unary(ctx, attrs, ins, OpId::Cos),
        "MatMul" => plan_matmul(ctx, attrs, ins),
        "Gemm" => plan_gemm(ctx, attrs, ins),
        "Softmax" => plan_softmax(ctx, attrs, ins),
        "ReduceMean" => plan_reduce(ctx, attrs, ins, OpId::ReduceMean),
        "ReduceSum" => plan_reduce(ctx, attrs, ins, OpId::ReduceSum),
        "ArgMax" => plan_argmax(ctx, attrs, ins),
        // The CNN operator set.
        "Conv" => plan_conv(ctx, attrs, ins),
        "MaxPool" => plan_pool(ctx, attrs, ins, output_count, OpId::MaxPool),
        "AveragePool" => plan_pool(ctx, attrs, ins, output_count, OpId::AveragePool),
        "GlobalAveragePool" => plan_global_average_pool(ctx, attrs, ins),
        "BatchNormalization" => plan_batch_norm(ctx, attrs, ins, output_count),
        "Resize" => plan_resize(ctx, attrs, ins),
        "Pad" => plan_pad(ctx, attrs, ins),
        "Clip" => plan_clip(ctx, attrs, ins),
        "Sigmoid" => plan_unary(ctx, attrs, ins, OpId::Sigmoid),
        "Max" => plan_binary(ctx, attrs, ins, OpId::Maximum),
        "Min" => plan_binary(ctx, attrs, ins, OpId::Minimum),
        "TopK" => plan_top_k(ctx, attrs, ins, output_count),
        "If" | "Loop" | "Scan" => Err(ImportProblemKind::ControlFlow {
            op_type: ctx.op_type.clone(),
        }),
        "Attention" | "MultiHeadAttention" | "GroupQueryAttention" => {
            Err(ImportProblemKind::UnimplementedOpId {
                op: OpId::MaskedAttention,
                owner: "FN3-15 (#61)",
            })
        }
        other => Err(ImportProblemKind::UnmappedOp {
            op_type: other.to_string(),
            domain: String::new(),
            detail: "no fn3 op id corresponds to it, and this importer never substitutes one. \
                     Extend the mapping in FN3-08 (#54) with the op that is needed"
                .to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Import driver
// ---------------------------------------------------------------------------

/// Decode and import a complete ONNX model.
pub(crate) fn build_model(bytes: &[u8]) -> Result<ModelImport, super::error::ImportError> {
    Builder::new(decode_bytes(bytes)?).run()
}

/// Decode and import a model's graph, without decoder architecture detection.
pub(crate) fn build_graph_parts(
    bytes: &[u8],
    bindings: &BTreeMap<String, Vec<u64>>,
) -> Result<GraphParts, super::error::ImportError> {
    let (parts, diagnostics) = Builder::new(decode_bytes(bytes)?)
        .with_bindings(bindings)
        .decode();
    if let Some(error) = diagnostics.finish() {
        return Err(error);
    }
    parts.ok_or_else(|| super::error::ImportError::single(ImportProblemKind::MissingGraph))
}

/// The one size check and protobuf decode both entry points share.
fn decode_bytes(bytes: &[u8]) -> Result<proto::Model<'_>, super::error::ImportError> {
    // One comparison before a single byte is decoded: the model size is the
    // budget the format and the importer share, and a refusal names it.
    if !model_bytes_within_cap(bytes.len() as u64) {
        return Err(super::error::ImportError::single(
            ImportProblemKind::LimitExceeded {
                what: "model size in bytes",
                requested: bytes.len() as u64,
                limit: MAX_MODEL_BYTES,
            },
        ));
    }
    proto::decode_model(bytes)
        .map_err(|error| super::error::ImportError::single(ImportProblemKind::Proto(error)))
}

/// The graph inputs and their declared shapes, read without shape inference.
///
/// `doctor` needs this exactly when the typed pass refuses the model, so the
/// builder's diagnostics are ignored rather than returned: a value table with a
/// contradiction still has an interface.
pub(crate) fn declared_inputs_of(
    bytes: &[u8],
) -> Result<Vec<(String, String)>, super::error::ImportError> {
    let model = decode_bytes(bytes)?;
    let (parts, _diagnostics) = Builder::new(model).decode();
    let Some(parts) = parts else {
        return Ok(Vec::new());
    };
    Ok(parts
        .graph
        .inputs()
        .iter()
        .map(|&id| parts.graph.value(id))
        .map(|value| (value.name.clone(), value.shape.to_string()))
        .collect())
}

/// A declared dtype and shape from `ValueInfoProto`.
#[derive(Debug, Clone)]
struct DeclaredValue {
    dtype: DType,
    shape: ValueShape,
}

/// The importer's working state.
struct Builder<'a> {
    model: proto::Model<'a>,
    diagnostics: Diagnostics,
    warnings: Vec<ImportWarning>,
    values: Vec<ValueInfo>,
    abstracts: Vec<Abstract>,
    by_name: BTreeMap<String, ValueId>,
    nodes: Vec<GraphNode>,
    /// Typed-IR attributes, parallel to `nodes`. See [`GraphParts`].
    typed_params: Vec<crate::inference::backend::OpParams>,
    weights: Vec<WeightTensor>,
    weight_ids: BTreeMap<String, WeightId>,
    declared: BTreeMap<String, DeclaredValue>,
    /// Concrete extents for graph inputs, as supplied by the caller.
    bindings: BTreeMap<String, Vec<u64>>,
    /// Bindings that matched a declared graph input.
    used_bindings: BTreeSet<String>,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
    /// Set when a parser budget was reached, so `run` can stop before geometry
    /// detection instead of reporting a cascade of architecture problems.
    over_budget: bool,
    /// `Constant` nodes folded into weights. They produce no fn3 node but stay
    /// in the source census.
    folded_constants: usize,
}

impl<'a> Builder<'a> {
    fn new(model: proto::Model<'a>) -> Builder<'a> {
        Builder {
            model,
            diagnostics: Diagnostics::default(),
            warnings: Vec::new(),
            values: Vec::new(),
            abstracts: Vec::new(),
            by_name: BTreeMap::new(),
            nodes: Vec::new(),
            typed_params: Vec::new(),
            weights: Vec::new(),
            weight_ids: BTreeMap::new(),
            declared: BTreeMap::new(),
            bindings: BTreeMap::new(),
            used_bindings: BTreeSet::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            over_budget: false,
            folded_constants: 0,
        }
    }

    /// Attach the caller's input-shape bindings.
    fn with_bindings(mut self, bindings: &BTreeMap<String, Vec<u64>>) -> Builder<'a> {
        self.bindings = bindings.clone();
        self
    }

    fn run(self) -> Result<ModelImport, super::error::ImportError> {
        let (parts, mut diagnostics) = self.decode();
        let Some(parts) = parts else {
            return Err(diagnostics
                .finish()
                .expect("a missing graph or an exhausted budget is a problem"));
        };
        let GraphParts {
            ir_version,
            producer,
            producer_version,
            opset,
            graph_name,
            graph,
            weights,
            summary,
            warnings,
            typed_params: _,
        } = parts;
        let Some(geometry) = detect_geometry(&graph, &weights, &mut diagnostics) else {
            // Defensive: every `None` path in the detector records a problem, so
            // this only fires if one is added without one. Report it instead of
            // panicking.
            diagnostics.push_global(ImportProblemKind::Architecture {
                detail: "architecture detection produced no geometry".to_string(),
            });
            return Err(diagnostics
                .finish()
                .expect("a missing geometry is a problem"));
        };
        if let Some(error) = diagnostics.finish() {
            return Err(error);
        }
        Ok(ModelImport {
            ir_version,
            producer,
            producer_version,
            opset,
            graph_name,
            graph,
            weights,
            geometry,
            summary,
            warnings,
        })
    }

    /// Decode, validate and assemble the graph without architecture detection.
    ///
    /// This is the graph-level half of [`Builder::run`]. A CNN has no decoder
    /// geometry, so [`super::import_onnx_graph`] uses this directly; the
    /// decoder path adds [`detect_geometry`] afterwards.
    ///
    /// Some problems are recorded while the graph is still assembled, so the
    /// caller decides whether to stop: the decoder path runs architecture
    /// detection over the partial graph to report those problems too, while
    /// [`super::import_onnx_graph`] rejects any non-empty diagnostics.
    fn decode(mut self) -> (Option<GraphParts>, Diagnostics) {
        self.check_header();
        let Some(graph) = self.model.graph.take() else {
            self.diagnostics
                .push_global(ImportProblemKind::MissingGraph);
            return (None, self.diagnostics);
        };
        if graph.sparse_initializer_count > 0 {
            self.diagnostics
                .push_global(ImportProblemKind::SparseInitializer {
                    count: graph.sparse_initializer_count,
                });
        }
        let source_nodes = graph.nodes.len();
        self.register_declared(&graph);
        self.register_inputs(&graph);
        self.report_unused_bindings();
        self.register_initializers(&graph);
        self.register_nodes(&graph);
        if self.over_budget {
            // The value/node/weight table is already beyond what this build (and
            // the artifact format) accepts. Running architecture detection over a
            // truncated graph would produce a page of misleading "missing
            // operand" problems after the one that matters.
            return (None, self.diagnostics);
        }
        self.resolve_outputs(&graph);

        let graph_ir = GraphIr {
            values: std::mem::take(&mut self.values),
            nodes: std::mem::take(&mut self.nodes),
            inputs: std::mem::take(&mut self.inputs),
            outputs: std::mem::take(&mut self.outputs),
        };
        if graph_ir.nodes.len() != self.typed_params.len() {
            self.diagnostics
                .push_global(ImportProblemKind::Architecture {
                    detail: format!(
                        "internal: {} imported nodes carry {} typed attribute sets",
                        graph_ir.nodes.len(),
                        self.typed_params.len()
                    ),
                });
        }
        let weights = match WeightSet::new(std::mem::take(&mut self.weights)) {
            Ok(weights) => weights,
            Err(error) => {
                self.diagnostics
                    .push_global(ImportProblemKind::Architecture {
                        detail: error.to_string(),
                    });
                WeightSet::default()
            }
        };
        let summary = summarize(
            &graph_ir,
            &weights,
            Some(source_nodes),
            self.folded_constants,
        );
        let opset = self
            .model
            .opset_imports
            .iter()
            .find(|opset| opset.domain.is_empty())
            .map(|opset| OpsetImport {
                domain: String::new(),
                version: opset.version,
            })
            .unwrap_or(OpsetImport {
                domain: String::new(),
                version: 0,
            });
        (
            Some(GraphParts {
                ir_version: self.model.ir_version,
                producer: self.model.producer_name.to_string(),
                producer_version: self.model.producer_version.to_string(),
                opset,
                graph_name: graph.name.to_string(),
                graph: graph_ir,
                weights,
                summary,
                warnings: std::mem::take(&mut self.warnings),
                typed_params: std::mem::take(&mut self.typed_params),
            }),
            self.diagnostics,
        )
    }

    /// Model-level checks that do not need the graph.
    fn check_header(&mut self) {
        let version = self.model.ir_version;
        if !(MIN_IR_VERSION..=MAX_IR_VERSION).contains(&version) {
            self.diagnostics
                .push_global(ImportProblemKind::UnsupportedIrVersion {
                    found: version,
                    supported: format!("{MIN_IR_VERSION}..={MAX_IR_VERSION}"),
                });
        }
        if self.model.function_count > 0 {
            self.diagnostics
                .push_global(ImportProblemKind::LocalFunctions {
                    count: self.model.function_count,
                });
        }
        if self.model.has_training_info {
            self.diagnostics
                .push_global(ImportProblemKind::TrainingInfo);
        }
        let mut default_opset = None;
        for opset in &self.model.opset_imports {
            if opset.domain.is_empty() {
                default_opset = Some(opset.version);
            } else {
                self.diagnostics
                    .push_global(ImportProblemKind::UnsupportedDomain {
                        domain: opset.domain.to_string(),
                    });
            }
        }
        match default_opset {
            None => self
                .diagnostics
                .push_global(ImportProblemKind::MissingOpset),
            Some(version) if !(MIN_OPSET..=MAX_OPSET).contains(&version) => self
                .diagnostics
                .push_global(ImportProblemKind::UnsupportedOpset {
                    version,
                    supported: format!(
                        "{MIN_OPSET}..={MAX_OPSET}: the importer implements the opset-12 \
                             Split attribute form, the opset-13 input form for \
                             Squeeze/Unsqueeze/Split/Slice and the attribute form of ReduceMean \
                             axes"
                    ),
                }),
            Some(_) => {}
        }
    }

    /// Collect the declared dtypes and shapes of the graph interface and of
    /// every annotated intermediate.
    fn register_declared(&mut self, graph: &proto::Graph<'_>) {
        for value_info in graph
            .inputs
            .iter()
            .chain(graph.outputs.iter())
            .chain(graph.value_info.iter())
        {
            if value_info.name.is_empty() {
                continue;
            }
            if let Some(unsupported) = value_info.unsupported_type {
                self.diagnostics
                    .push_global(ImportProblemKind::UnsupportedType {
                        value: value_info.name.to_string(),
                        detail: unsupported.to_string(),
                    });
                continue;
            }
            let Some(ty) = &value_info.ty else {
                continue;
            };
            let carrier = format!("value `{}`", value_info.name);
            let dtype = match onnx_dtype(ty.elem_type, carrier.clone()) {
                Ok(dtype) => dtype,
                Err(kind) => {
                    self.diagnostics.push_global(kind);
                    continue;
                }
            };
            let mut shape = match &ty.dims {
                Some(dims) => ValueShape::Dims(dims.iter().map(declared_dim).collect()),
                None => ValueShape::Unknown,
            };
            // A caller-supplied binding replaces a graph input's declared
            // extents. A dynamic ONNX export spells those as `dim_param` names
            // with no numeric meaning, and every extent downstream — a
            // `Reshape` target, a `Concat` axis, a convolution's geometry —
            // depends on them being concrete.
            let is_input = graph
                .inputs
                .iter()
                .any(|input| input.name == value_info.name);
            if is_input {
                if let Some(extents) = self.bindings.get(value_info.name) {
                    match &shape {
                        ValueShape::Dims(dims) if dims.len() != extents.len() => {
                            self.diagnostics
                                .push_global(ImportProblemKind::ShapeBinding {
                                    input: value_info.name.to_string(),
                                    detail: format!(
                                        "the binding has {} extents, the model declares rank {}",
                                        extents.len(),
                                        dims.len()
                                    ),
                                });
                        }
                        _ => {
                            shape =
                                ValueShape::Dims(extents.iter().copied().map(Dim::Known).collect());
                            self.used_bindings.insert(value_info.name.to_string());
                        }
                    }
                }
            }
            self.declared
                .insert(value_info.name.to_string(), DeclaredValue { dtype, shape });
        }
    }

    fn register_inputs(&mut self, graph: &proto::Graph<'_>) {
        for value_info in &graph.inputs {
            let name = value_info.name.to_string();
            let Some(declared) = self.declared.get(&name).cloned() else {
                // The dtype/shape problem was already reported by
                // register_declared; keep going with an unknown shape.
                let dtype = value_info
                    .ty
                    .as_ref()
                    .and_then(|ty| onnx_dtype(ty.elem_type, name.clone()).ok())
                    .unwrap_or(DType::F32);
                self.push_value(
                    name,
                    dtype,
                    ValueShape::Unknown,
                    ValueOrigin::GraphInput,
                    Abstract::tensor(dtype, ValueShape::Unknown),
                    None,
                );
                continue;
            };
            let id = self.push_value(
                name.clone(),
                declared.dtype,
                declared.shape.clone(),
                ValueOrigin::GraphInput,
                Abstract::tensor(declared.dtype, declared.shape.clone()),
                None,
            );
            self.inputs.push(id);
        }
    }

    fn register_initializers(&mut self, graph: &proto::Graph<'_>) {
        for tensor in &graph.initializers {
            let name = tensor.name.to_string();
            match build_weight(tensor, WeightSource::Initializer, || name.clone()) {
                Ok(weight) => {
                    self.add_weight(weight, None);
                }
                Err(kind) => self.diagnostics.push_global(kind),
            }
        }
    }

    /// Fold one `Constant` node into a weight value.
    ///
    /// The typed IR has no constant payload, and a runtime `Constant` op would
    /// execute once per step to materialise a value the file already fixed.
    /// Folding it into the weight table makes the value an initializer-like
    /// operand, which is also how `ConstantOfShape`'s fill value is handled.
    fn register_constant(&mut self, ctx: &NodeRef, node: &proto::Node<'_>) {
        let mut attrs = Attrs::new(&node.attributes);
        let result = (|| -> Result<WeightTensor, PlanError> {
            let name = node
                .outputs
                .first()
                .copied()
                .filter(|name| !name.is_empty())
                .ok_or(ImportProblemKind::MissingName {
                    what: "Constant output",
                })?;
            let mut forms: Vec<&'static str> = Vec::new();
            let mut weight: Option<WeightTensor> = None;
            if let Some(tensor) = attrs.tensor("value")? {
                forms.push("value");
                weight = Some(build_weight(
                    tensor,
                    WeightSource::AttributeConstant,
                    || name.to_string(),
                )?);
            }
            if let Some(value) = attrs.float("value_float")? {
                forms.push("value_float");
                weight = Some(attribute_weight(
                    name,
                    DType::F32,
                    Vec::new(),
                    value.to_le_bytes().to_vec(),
                )?);
            }
            if let Some(values) = attrs.floats("value_floats")? {
                forms.push("value_floats");
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                weight = Some(attribute_weight(
                    name,
                    DType::F32,
                    vec![values.len() as u64],
                    bytes,
                )?);
            }
            if let Some(value) = attrs.int("value_int")? {
                forms.push("value_int");
                weight = Some(attribute_weight(
                    name,
                    DType::I64,
                    Vec::new(),
                    value.to_le_bytes().to_vec(),
                )?);
            }
            if let Some(values) = attrs.ints("value_ints")? {
                forms.push("value_ints");
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                weight = Some(attribute_weight(
                    name,
                    DType::I64,
                    vec![values.len() as u64],
                    bytes,
                )?);
            }
            if forms.len() > 1 {
                return Err(ImportProblemKind::UnsupportedAttribute {
                    attribute: "Constant".to_string(),
                    detail: format!(
                        "the node carries more than one value payload ({}); exactly one is \
                         allowed",
                        forms.join(", ")
                    ),
                });
            }
            weight.ok_or_else(|| missing_attribute("value"))
        })();
        let mut problems = attrs.finish(Some(
            "value, value_float, value_floats, value_int, value_ints",
        ));
        match result {
            Ok(weight) => {
                self.add_weight(weight, Some(ctx));
                self.folded_constants += 1;
            }
            Err(error) => problems.push(error),
        }
        for problem in problems {
            self.diagnostics.push(Some(ctx.clone()), problem);
        }
    }

    /// Register a weight and its value. Returns the value handle.
    fn add_weight(&mut self, weight: WeightTensor, node: Option<&NodeRef>) -> ValueId {
        let name = weight.name().to_string();
        let dtype = weight.dtype();
        let shape = ValueShape::from_dims(weight.shape().dims().iter().copied());
        let tracked = weight_abstract(&weight);
        if let Some(existing) = self.weight_ids.get(&name).copied() {
            self.diagnostics.push(
                node.cloned(),
                ImportProblemKind::DuplicateValue {
                    value: name.clone(),
                    existing: format!("weight {existing}"),
                },
            );
            return self.push_value(
                name,
                dtype,
                shape,
                ValueOrigin::Weight(existing),
                tracked,
                node,
            );
        }
        let id = WeightId::new(self.weights.len() as u32);
        if self.by_name.contains_key(&name) {
            self.diagnostics.push(
                node.cloned(),
                ImportProblemKind::DuplicateValue {
                    value: name.clone(),
                    existing: "a graph input".to_string(),
                },
            );
        }
        self.weights.push(weight);
        self.weight_ids.insert(name.clone(), id);
        self.push_value(name, dtype, shape, ValueOrigin::Weight(id), tracked, node)
    }

    /// Register one value, uniquifying a duplicated name so the rest of the
    /// graph stays addressable.
    fn push_value(
        &mut self,
        name: String,
        dtype: DType,
        shape: ValueShape,
        origin: ValueOrigin,
        tracked: Abstract,
        node: Option<&NodeRef>,
    ) -> ValueId {
        let mut unique = name.clone();
        if self.by_name.contains_key(&unique) {
            let existing = self.by_name[&unique];
            self.diagnostics.push(
                node.cloned(),
                ImportProblemKind::DuplicateValue {
                    value: name.clone(),
                    existing: describe_origin(&self.values[existing.index()].origin),
                },
            );
            let mut counter = 1;
            while self.by_name.contains_key(&unique) {
                unique = format!("{name}#duplicate{counter}");
                counter += 1;
            }
        }
        let id = ValueId::new(self.values.len() as u32);
        self.values.push(ValueInfo {
            name: unique.clone(),
            dtype,
            shape: shape.clone(),
            origin,
        });
        self.abstracts.push(tracked);
        self.by_name.insert(unique, id);
        if let Some(node) = node {
            self.check_declared(&name, &shape, node);
        }
        id
    }

    /// Compare an inferred shape with the file's declared one.
    fn check_declared(&mut self, name: &str, shape: &ValueShape, node: &NodeRef) {
        let Some(declared) = self.declared.get(name) else {
            return;
        };
        if let (Some(declared_rank), Some(rank)) = (declared.shape.rank(), shape.rank()) {
            if declared_rank != rank {
                self.warnings.push(ImportWarning {
                    node: Some(node.clone()),
                    kind: ImportWarningKind::DeclaredShape {
                        value: name.to_string(),
                        declared: declared.shape.render(),
                        inferred: shape.render(),
                    },
                });
                return;
            }
        }
        if declared.shape.is_static() && shape.is_static() && declared.shape != *shape {
            self.warnings.push(ImportWarning {
                node: Some(node.clone()),
                kind: ImportWarningKind::DeclaredShape {
                    value: name.to_string(),
                    declared: declared.shape.render(),
                    inferred: shape.render(),
                },
            });
        }
    }

    /// Which parser budget the working tables have reached, if any.
    ///
    /// Returns `(what, requested, limit)` for the first table at its cap, so the
    /// caller records one problem naming the budget rather than continuing to
    /// grow past it.
    fn budget_headroom(&self) -> Option<(&'static str, u64, u64)> {
        if self.values.len() >= MAX_VALUES {
            return Some((
                "graph value table",
                self.values.len() as u64,
                MAX_VALUES as u64,
            ));
        }
        if self.nodes.len() >= MAX_NODES {
            return Some((
                "graph node table",
                self.nodes.len() as u64,
                MAX_NODES as u64,
            ));
        }
        if self.weights.len() >= MAX_WEIGHTS {
            return Some((
                "graph weight table",
                self.weights.len() as u64,
                MAX_WEIGHTS as u64,
            ));
        }
        None
    }

    fn register_nodes(&mut self, graph: &proto::Graph<'_>) {
        // Every node output name and its producer, so a use-before-define can
        // say whether the value exists later in the file (a topological-order
        // violation) or nowhere at all (an undefined value).
        let mut producer_of: BTreeMap<&str, u32> = BTreeMap::new();
        for (index, node) in graph.nodes.iter().enumerate() {
            for name in &node.outputs {
                if !name.is_empty() {
                    producer_of.entry(*name).or_insert(index as u32);
                }
            }
        }
        for (index, node) in graph.nodes.iter().enumerate() {
            // A composition writes two ops for one node (`Gemm` → `MatMul` +
            // `Add`) and `Split` writes many values for one node, so the decoded
            // node count is not an upper bound on the tables this loop grows.
            // The budget is checked before each node, which bounds the overshoot
            // to one node's operands and outputs.
            if let Some((what, requested, limit)) = self.budget_headroom() {
                self.diagnostics
                    .push_global(ImportProblemKind::LimitExceeded {
                        what,
                        requested,
                        limit,
                    });
                self.over_budget = true;
                return;
            }
            let ctx = NodeRef::new(index as u32, node.name, node.op_type);
            if !node.domain.is_empty() {
                self.diagnostics.push(
                    Some(ctx),
                    ImportProblemKind::UnsupportedDomain {
                        domain: node.domain.to_string(),
                    },
                );
                continue;
            }
            // A `Constant` is data, not work: fold it into the weight table so
            // no runtime op materialises it and the typed IR sees an ordinary
            // initializer-like value.
            if node.op_type == "Constant" {
                self.register_constant(&ctx, node);
                continue;
            }
            let mut input_ids = Vec::with_capacity(node.inputs.len());
            let mut input_abstracts = Vec::with_capacity(node.inputs.len());
            let mut failed = false;
            let mut omitted_before_present = false;
            for (position, name) in node.inputs.iter().enumerate() {
                if name.is_empty() {
                    // ONNX `Resize` in opset 13+ carries `X, roi?, scales?,
                    // sizes?`; a sizes-only node omits `scales`, which is a
                    // hole in the middle. The planner identifies the surviving
                    // tail by dtype, so the hole is allowed there and nowhere
                    // else.
                    // `Resize` in opset 13+ carries `X, roi?, scales?,
                    // sizes?`; an exporter may omit `roi` (position 1), `scales`
                    // (position 2) or both and keep `sizes`. Any hole the
                    // planner can resolve by dtype is allowed.
                    let resize_scales_hole = node.op_type == "Resize"
                        && matches!(position, 1 | 2)
                        && node
                            .inputs
                            .iter()
                            .skip(position + 1)
                            .any(|later| !later.is_empty());
                    if !resize_scales_hole {
                        omitted_before_present = true;
                    }
                    continue;
                }
                if omitted_before_present {
                    self.diagnostics.push(
                        Some(ctx.clone()),
                        ImportProblemKind::UnsupportedAttribute {
                            attribute: "input".to_string(),
                            detail: format!(
                                "`{name}` follows an omitted optional input; this importer only \
                                 supports omitting trailing optional inputs"
                            ),
                        },
                    );
                    failed = true;
                }
                match self.by_name.get(*name) {
                    Some(id) => {
                        input_ids.push(*id);
                        input_abstracts.push(self.abstracts[id.index()].clone());
                    }
                    None => {
                        let kind = match producer_of.get(*name) {
                            Some(producer_index) => ImportProblemKind::NotTopological {
                                value: (*name).to_string(),
                                producer_index: *producer_index,
                            },
                            None => ImportProblemKind::UndefinedValue {
                                value: (*name).to_string(),
                            },
                        };
                        self.diagnostics.push(Some(ctx.clone()), kind);
                        failed = true;
                        break;
                    }
                }
            }
            if input_abstracts
                .iter()
                .any(|abstract_value| matches!(abstract_value, Abstract::Poisoned))
            {
                failed = true;
            }
            if failed {
                self.poison_outputs(&ctx, node);
                continue;
            }
            let planned = plan_node(
                &ctx,
                node,
                &input_abstracts,
                node.outputs.len(),
                &mut self.diagnostics,
            );
            match planned {
                Some(planned) => self.register_planned(&ctx, node, &input_ids, planned),
                None => self.poison_outputs(&ctx, node),
            }
        }
    }

    /// A binding that matched nothing is a caller mistake, not a model
    /// property: name it rather than ignore it.
    fn report_unused_bindings(&mut self) {
        let unused: Vec<String> = self
            .bindings
            .keys()
            .filter(|name| !self.used_bindings.contains(*name))
            .cloned()
            .collect();
        for name in unused {
            self.diagnostics
                .push_global(ImportProblemKind::ShapeBinding {
                    input: name,
                    detail: "the model declares no graph input with this name".to_string(),
                });
        }
    }

    /// Register a node's outputs as unavailable so its consumers stay quiet.
    ///
    /// Only reached after a diagnostic, so the resulting [`ModelImport`] is
    /// discarded; the placeholder origin exists to keep the value table
    /// addressable while the remaining nodes are validated.
    fn poison_outputs(&mut self, _ctx: &NodeRef, node: &proto::Node<'_>) {
        for name in &node.outputs {
            if name.is_empty() {
                continue;
            }
            let dtype = self
                .declared
                .get(*name)
                .map(|declared| declared.dtype)
                .unwrap_or(DType::F32);
            self.push_value(
                (*name).to_string(),
                dtype,
                ValueShape::Unknown,
                ValueOrigin::GraphInput,
                Abstract::Poisoned,
                None,
            );
        }
    }

    /// Turn planned operations into [`GraphNode`]s and values.
    fn register_planned(
        &mut self,
        ctx: &NodeRef,
        node: &proto::Node<'_>,
        input_ids: &[ValueId],
        planned: Vec<PlannedOp>,
    ) {
        let last = planned.len().saturating_sub(1);
        let mut intermediates: Vec<ValueId> = Vec::new();
        for (position, operation) in planned.iter().enumerate() {
            let is_last = position == last;
            let mut operands = Vec::with_capacity(operation.operands.len());
            for operand in &operation.operands {
                match operand {
                    Operand::Input(index) => match input_ids.get(*index) {
                        Some(id) => operands.push(*id),
                        None => operands.push(ValueId::new(0)),
                    },
                    Operand::Intermediate(index) => match intermediates.get(*index) {
                        Some(id) => operands.push(*id),
                        None => {
                            debug_assert!(false, "planned operand refers to a missing op");
                            operands.push(*input_ids.first().unwrap_or(&ValueId::new(0)));
                        }
                    },
                    Operand::NewConstant(weight) => {
                        let id = self.add_weight((**weight).clone(), Some(ctx));
                        operands.push(id);
                    }
                }
            }
            let node_index = self.nodes.len() as u32;
            let mut outputs = Vec::with_capacity(operation.outputs.len());
            for (position_in_node, value) in operation.outputs.iter().enumerate() {
                let name = if is_last {
                    match node.outputs.get(position_in_node) {
                        Some(name) if !name.is_empty() => (*name).to_string(),
                        _ => format!("{}#out{position_in_node}", describe_node(ctx)),
                    }
                } else {
                    format!("{}#op{position}", describe_node(ctx))
                };
                let abstract_value = planned_abstract(value);
                outputs.push(self.push_value(
                    name,
                    value.dtype,
                    value.shape.clone(),
                    ValueOrigin::Node(node_index),
                    abstract_value,
                    Some(ctx),
                ));
            }
            self.nodes.push(GraphNode {
                onnx_index: ctx.index,
                name: ctx.name.clone(),
                onnx_op: ctx.op_type.clone(),
                op: operation.op,
                inputs: operands,
                outputs: outputs.clone(),
                params: operation.params.clone(),
            });
            self.typed_params.push(
                operation
                    .typed
                    .clone()
                    .unwrap_or_else(|| operation.params.clone()),
            );
            intermediates.push(
                outputs
                    .first()
                    .copied()
                    .unwrap_or_else(|| *input_ids.first().unwrap_or(&ValueId::new(0))),
            );
        }
    }

    /// Resolve the graph's output names.
    fn resolve_outputs(&mut self, graph: &proto::Graph<'_>) {
        for value_info in &graph.outputs {
            let name = value_info.name.to_string();
            match self.by_name.get(&name).copied() {
                Some(id) => {
                    if let Some(declared) = self.declared.get(&name).cloned() {
                        let value = &self.values[id.index()];
                        if declared.dtype != value.dtype {
                            self.diagnostics.push(
                                None,
                                ImportProblemKind::DTypeMismatch {
                                    detail: format!(
                                        "graph output `{name}` is declared {} but produced as {}",
                                        declared.dtype, value.dtype
                                    ),
                                },
                            );
                        }
                        if let (Some(declared_rank), Some(rank)) =
                            (declared.shape.rank(), value.shape.rank())
                        {
                            if declared_rank != rank {
                                self.diagnostics.push(
                                    None,
                                    ImportProblemKind::ShapeMismatch {
                                        detail: format!(
                                            "graph output `{name}` is declared {} but produced as \
                                             the rank-{rank} {}",
                                            declared.shape.render(),
                                            value.shape.render()
                                        ),
                                    },
                                );
                            }
                        }
                    }
                    self.outputs.push(id);
                }
                None => self
                    .diagnostics
                    .push(None, ImportProblemKind::UndefinedValue { value: name }),
            }
        }
    }
}

/// Describe what produced a value, for a duplicate-name diagnostic.
fn describe_origin(origin: &ValueOrigin) -> String {
    match origin {
        ValueOrigin::GraphInput => "a graph input".to_string(),
        ValueOrigin::Weight(id) => format!("weight {id}"),
        ValueOrigin::Node(index) => format!("node {index}"),
    }
}

/// A node name for synthesized value names.
fn describe_node(ctx: &NodeRef) -> String {
    if ctx.name.is_empty() {
        format!("node{}", ctx.index)
    } else {
        ctx.name.clone()
    }
}

/// Convert a declared ONNX dimension.
fn declared_dim(dim: &proto::Dim) -> Dim {
    match dim {
        proto::Dim::Value(extent) => match u64::try_from(*extent) {
            Ok(extent) => Dim::Known(extent),
            Err(_) => Dim::unknown(format!("declared negative extent {extent}")),
        },
        proto::Dim::Param(name) => symbol_for(name),
        proto::Dim::Unknown => Dim::unknown("declared dimension is neither a value nor a symbol"),
    }
}

/// Map an ONNX `dim_param` onto an fn3 [`Symbol`].
///
/// The pinned export names the attention-mask axis `"past_sequence_length + 1"`
/// and the `present.*` axes the same way. At runtime that axis is `past + seq`
/// (`attention_mask` is `[batch, past + seq]` and the causal mask is built from
/// the prompt length), so it is [`Symbol::Total`]; the exporter's `+ 1` only
/// holds for a single new token. That is recorded here rather than silently
/// treating the axis as one token wide.
fn symbol_for(name: &str) -> Dim {
    let normalized = name.trim();
    match normalized {
        "batch_size" | "batch" | "batch_dim" | "B" => Dim::Symbol(Symbol::Batch),
        "sequence_length" | "sequence" | "seq" | "seq_len" | "S" => Dim::Symbol(Symbol::Seq),
        "past_sequence_length" | "past" | "past_len" | "P" => Dim::Symbol(Symbol::Past),
        "past_sequence_length + 1"
        | "past_sequence_length + sequence_length"
        | "total"
        | "total_sequence_length" => Dim::Symbol(Symbol::Total),
        "n_heads" | "num_heads" | "num_attention_heads" => Dim::Symbol(Symbol::NHeads),
        "n_kv_heads" | "num_kv_heads" => Dim::Symbol(Symbol::NKvHeads),
        "head_dim" | "hidden_size_per_head" => Dim::Symbol(Symbol::HeadDim),
        other => {
            if let Some(symbol) = Symbol::from_name(other) {
                Dim::Symbol(symbol)
            } else {
                Dim::unknown(format!("dim_param `{other}` has no fn3 symbol"))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Architecture detection
// ---------------------------------------------------------------------------

/// Find a weight by name.
fn weight_named<'w>(
    weights: &'w WeightSet,
    name: &str,
    problems: &mut Vec<ImportProblemKind>,
) -> Option<(WeightId, DType, &'w Shape)> {
    match weights.find(name) {
        Some(weight) => Some((weights.id_of(name)?, weight.dtype(), weight.shape())),
        None => {
            problems.push(ImportProblemKind::Architecture {
                detail: format!("required tensor `{name}` is missing"),
            });
            None
        }
    }
}

/// Require a rank-2 `f32` weight and return its shape.
fn expect_matrix(
    weights: &WeightSet,
    name: &str,
    problems: &mut Vec<ImportProblemKind>,
) -> Option<(WeightId, u64, u64)> {
    let (id, dtype, shape) = weight_named(weights, name, problems)?;
    if dtype != DType::F32 {
        problems.push(ImportProblemKind::Architecture {
            detail: format!("`{name}` is {dtype}; decoder weights are f32"),
        });
        return None;
    }
    match shape.dims() {
        [rows, columns] => Some((id, *rows, *columns)),
        other => {
            problems.push(ImportProblemKind::Architecture {
                detail: format!("`{name}` has shape {other:?}; expected a rank-2 matrix"),
            });
            None
        }
    }
}

/// Parse the layer index out of `transformer.h.{i}.<suffix>`.
fn layer_index(name: &str, suffix: &str) -> Option<u32> {
    let rest = name.strip_prefix("transformer.h.")?;
    let index = rest.strip_suffix(suffix)?;
    index.parse().ok()
}

/// Detect and validate the decoder geometry.
fn detect_geometry(
    graph: &GraphIr,
    weights: &WeightSet,
    diagnostics: &mut Diagnostics,
) -> Option<DecoderGeometry> {
    let mut problems: Vec<ImportProblemKind> = Vec::new();
    let architecture = |detail: String| ImportProblemKind::Architecture { detail };

    let Some((token_embedding, vocab_size, hidden_size)) =
        expect_matrix(weights, "transformer.wte.weight", &mut problems)
    else {
        for problem in problems {
            diagnostics.push_global(problem);
        }
        return None;
    };
    let Some((position_weight, max_positions, position_width)) =
        expect_matrix(weights, "transformer.wpe.weight", &mut problems)
    else {
        for problem in problems {
            diagnostics.push_global(problem);
        }
        return None;
    };
    if position_width != hidden_size {
        problems.push(architecture(format!(
            "the position table is {position_width} wide but the token table is {hidden_size}"
        )));
    }

    // Layer inventory: every block must have the four matrices this decoder
    // shape requires, and the indices must be contiguous from zero.
    let mut layers: Vec<u32> = weights
        .iter()
        .filter_map(|weight| layer_index(weight.name(), ".attn.c_attn.weight"))
        .collect();
    layers.sort_unstable();
    layers.dedup();
    let layer_count = layers.len() as u32;
    if layer_count == 0 {
        problems.push(architecture(
            "no `transformer.h.{i}.attn.c_attn.weight` tensors; the graph is not a GPT-2 decoder"
                .to_string(),
        ));
        for problem in problems {
            diagnostics.push_global(problem);
        }
        return None;
    }
    for (expected, found) in layers.iter().enumerate() {
        if *found != expected as u32 {
            problems.push(architecture(format!(
                "layer indices are not contiguous from 0: expected {expected}, found {found}"
            )));
            break;
        }
    }

    let mut mlp_hidden: Option<u64> = None;
    for layer in 0..layer_count {
        let prefix = format!("transformer.h.{layer}");
        let Some((_, attention_in, attention_out)) = expect_matrix(
            weights,
            &format!("{prefix}.attn.c_attn.weight"),
            &mut problems,
        ) else {
            continue;
        };
        if attention_in != hidden_size {
            problems.push(architecture(format!(
                "layer {layer}: c_attn expects {attention_in} inputs, the model width is \
                 {hidden_size}"
            )));
        }
        if attention_out != 3 * hidden_size {
            problems.push(architecture(format!(
                "layer {layer}: c_attn produces {attention_out} values; a GPT-2 block packs \
                 query/key/value into 3 x {hidden_size}"
            )));
        }
        if let Some((_, rows, columns)) = expect_matrix(
            weights,
            &format!("{prefix}.attn.c_proj.weight"),
            &mut problems,
        ) {
            if rows != hidden_size || columns != hidden_size {
                problems.push(architecture(format!(
                    "layer {layer}: c_proj is [{rows}, {columns}], expected \
                     [{hidden_size}, {hidden_size}]"
                )));
            }
        }
        if let Some((_, rows, columns)) =
            expect_matrix(weights, &format!("{prefix}.mlp.c_fc.weight"), &mut problems)
        {
            if rows != hidden_size {
                problems.push(architecture(format!(
                    "layer {layer}: c_fc expects {rows} inputs, the model width is {hidden_size}"
                )));
            }
            match mlp_hidden {
                None => mlp_hidden = Some(columns),
                Some(known) if known != columns => problems.push(architecture(format!(
                    "layer {layer}: c_fc width {columns} differs from layer 0 width {known}"
                ))),
                Some(_) => {}
            }
        }
        if let Some((_, rows, columns)) = expect_matrix(
            weights,
            &format!("{prefix}.mlp.c_proj.weight"),
            &mut problems,
        ) {
            if rows != mlp_hidden.unwrap_or(rows) || columns != hidden_size {
                problems.push(architecture(format!(
                    "layer {layer}: c_proj is [{rows}, {columns}], expected \
                     [{}, {hidden_size}]",
                    mlp_hidden.unwrap_or(0)
                )));
            }
        }
    }

    // The KV interface gives the head split.
    let mut kv = Vec::with_capacity(layer_count as usize);
    let mut kv_heads: Option<u64> = None;
    let mut head_dim: Option<u64> = None;
    for layer in 0..layer_count {
        let key_name = format!("past_key_values.{layer}.key");
        let value_name = format!("past_key_values.{layer}.value");
        let present_key_name = format!("present.{layer}.key");
        let present_value_name = format!("present.{layer}.value");
        let Some((past_key, key_info)) = graph.find(&key_name) else {
            problems.push(architecture(format!(
                "graph input `{key_name}` is missing; the KV-cache decoder interface is \
                 `past_key_values.{{i}}.{{key,value}}`"
            )));
            continue;
        };
        let Some((past_value, _)) = graph.find(&value_name) else {
            problems.push(architecture(format!(
                "graph input `{value_name}` is missing"
            )));
            continue;
        };
        let Some((present_key, _)) = graph.find(&present_key_name) else {
            problems.push(architecture(format!(
                "graph output `{present_key_name}` is missing"
            )));
            continue;
        };
        let Some((present_value, _)) = graph.find(&present_value_name) else {
            problems.push(architecture(format!(
                "graph output `{present_value_name}` is missing"
            )));
            continue;
        };
        if key_info.shape.rank() != Some(4) {
            problems.push(architecture(format!(
                "`{key_name}` has shape {}; the cache interface is \
                 [batch, kv_heads, past, head_dim]",
                key_info.shape.render()
            )));
            continue;
        }
        let dims = key_info.shape.dims();
        let (Some(heads), Some(width)) = (dims[1].as_known(), dims[3].as_known()) else {
            problems.push(architecture(format!(
                "`{key_name}` has the symbolic shape {}; kv_heads and head_dim must be concrete \
                 extents",
                key_info.shape.render()
            )));
            continue;
        };
        if let Some(known) = kv_heads {
            if known != heads {
                problems.push(architecture(format!(
                    "layer {layer} declares {heads} KV heads, layer 0 declares {known}"
                )));
            }
        } else {
            kv_heads = Some(heads);
        }
        if let Some(known) = head_dim {
            if known != width {
                problems.push(architecture(format!(
                    "layer {layer} declares head_dim {width}, layer 0 declares {known}"
                )));
            }
        } else {
            head_dim = Some(width);
        }
        if key_info.dtype != DType::F32 {
            problems.push(architecture(format!(
                "`{key_name}` is {}; the alpha KV cache is f32",
                key_info.dtype
            )));
        }
        kv.push(KvInterface {
            past_key,
            past_value,
            present_key,
            present_value,
        });
    }

    let (Some(kv_heads), Some(head_dim)) = (kv_heads, head_dim) else {
        problems.push(architecture(
            "the KV interface does not declare concrete head counts".to_string(),
        ));
        for problem in problems {
            diagnostics.push_global(problem);
        }
        return None;
    };
    if head_dim == 0 || hidden_size % head_dim != 0 {
        problems.push(architecture(format!(
            "model width {hidden_size} is not divisible by head_dim {head_dim}"
        )));
    }
    let query_heads = hidden_size / head_dim;
    if kv_heads == 0 || query_heads % kv_heads != 0 {
        problems.push(architecture(format!(
            "{query_heads} query heads do not group into {kv_heads} KV heads"
        )));
    }

    let Some((logits, logits_info)) = graph.find("logits") else {
        problems.push(architecture("graph output `logits` is missing".to_string()));
        for problem in problems {
            diagnostics.push_global(problem);
        }
        return None;
    };
    if logits_info.dtype != DType::F32 {
        problems.push(architecture(format!(
            "`logits` is {}; the alpha compute dtype is f32",
            logits_info.dtype
        )));
    }
    if logits_info.shape.rank() != Some(3) {
        problems.push(architecture(format!(
            "`logits` has shape {}; expected [batch, seq, vocab]",
            logits_info.shape.render()
        )));
    } else if logits_info.shape.dims()[2].as_known() != Some(vocab_size) {
        problems.push(architecture(format!(
            "`logits` has a vocabulary of {} but the embedding table has {vocab_size} rows",
            logits_info.shape.dims()[2]
        )));
    }

    let layer_norm_epsilon = detect_layer_norm_epsilon(graph, weights, &mut problems);

    if !problems.is_empty() {
        for problem in problems {
            diagnostics.push_global(problem);
        }
        return None;
    }
    let Some(mlp_hidden) = mlp_hidden else {
        diagnostics.push_global(architecture(
            "the MLP width could not be detected from `mlp.c_fc.weight`".to_string(),
        ));
        return None;
    };
    // Narrowing to `u32` is a real bound (the geometry is reported in `u32`), so
    // an overflow is reported rather than silently dropping the geometry.
    let mut narrow = |value: u64, what: &str| -> Option<u32> {
        match u32::try_from(value) {
            Ok(narrowed) => Some(narrowed),
            Err(_) => {
                diagnostics
                    .push_global(architecture(format!("{what} {value} does not fit in u32")));
                None
            }
        }
    };
    let geometry = DecoderGeometry {
        layers: narrow(u64::from(layer_count), "the layer count")?,
        hidden_size: narrow(hidden_size, "the model width")?,
        mlp_hidden_size: narrow(mlp_hidden, "the MLP width")?,
        query_heads: narrow(query_heads, "the query head count")?,
        kv_heads: narrow(kv_heads, "the KV head count")?,
        head_dim: narrow(head_dim, "the head width")?,
        vocab_size: narrow(vocab_size, "the vocabulary size")?,
        positions: PositionEmbedding::Learned {
            weight: position_weight,
            max_positions: narrow(max_positions, "the position table size")?,
        },
        layer_norm_epsilon,
        token_embedding,
        logits,
        kv,
    };
    Some(geometry)
}

/// Find the LayerNorm epsilon structurally: `Add(ReduceMean(variance), eps)`.
///
/// The exporter decomposes LayerNorm into primitives, so the epsilon is an
/// ordinary `f32` scalar constant in the graph rather than configuration. This
/// looks for the exact shape of that constant (a scalar `f32` in `(0, 0.01]`
/// added to the mean of a squared deviation) and requires every candidate to
/// agree. It is reporting information, not a runtime input: the graph already
/// carries the value.
fn detect_layer_norm_epsilon(
    graph: &GraphIr,
    weights: &WeightSet,
    problems: &mut Vec<ImportProblemKind>,
) -> Option<f32> {
    let mut found: Vec<f32> = Vec::new();
    for node in graph.nodes() {
        if node.op != OpId::Add || node.inputs.len() != 2 {
            continue;
        }
        let mean_node = match graph.value(node.inputs[0]).origin {
            ValueOrigin::Node(index) => graph.nodes().get(index as usize),
            _ => None,
        };
        let Some(mean_node) = mean_node else {
            continue;
        };
        if mean_node.op != OpId::ReduceMean {
            continue;
        }
        let ValueOrigin::Weight(weight_id) = graph.value(node.inputs[1]).origin else {
            continue;
        };
        let Some(weight) = weights.get(weight_id) else {
            continue;
        };
        if weight.dtype() != DType::F32 || weight.byte_len() != 4 {
            continue;
        }
        let mut word = [0u8; 4];
        word.copy_from_slice(weight.bytes());
        let value = f32::from_le_bytes(word);
        // `NaN > 0.0` is false, so a NaN scale would silently be read as "this
        // node is not a LayerNorm" and the geometry would carry no epsilon. A
        // non-finite scalar in the epsilon slot is corrupt input, not a shape
        // the detector should decline.
        if !value.is_finite() {
            problems.push(ImportProblemKind::Architecture {
                detail: format!(
                    "node {} carries the non-finite LayerNorm scale {value}; a decoder epsilon \
                     must be a positive finite float",
                    node.onnx_index
                ),
            });
            continue;
        }
        if value > 0.0 && value <= 0.01 {
            found.push(value);
        }
    }
    match found.first().copied() {
        None => None,
        Some(first) => {
            if found.iter().any(|value| *value != first) {
                problems.push(ImportProblemKind::Architecture {
                    detail: format!(
                        "LayerNorm epsilon differs between blocks: {found:?}; the decoder contract \
                         has one epsilon"
                    ),
                });
                None
            } else {
                Some(first)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

/// Count nodes, operators, weights and dtypes.
/// Count the graph's nodes and weights.
///
/// `source_nodes` is `GraphProto.node.len()` and `folded_constants` is how many
/// `Constant` nodes were folded into weights: a folded node still belongs in the
/// ONNX census even though it produces no fn3 node. The artifact path, which
/// has no source file, passes the emitted count and zero.
fn summarize(
    graph: &GraphIr,
    weights: &WeightSet,
    source_nodes: Option<usize>,
    folded_constants: usize,
) -> ImportSummary {
    // The ONNX census counts *ONNX nodes*: a `Gemm` that becomes MatMul + Add is
    // still one ONNX node.
    let mut onnx_node_ops: BTreeMap<u32, &str> = BTreeMap::new();
    let mut ops: BTreeMap<u16, usize> = BTreeMap::new();
    for node in graph.nodes() {
        // A composition (`Gemm` -> MatMul + Add) is one ONNX node, so the
        // source census is keyed by source index, not by emitted node.
        onnx_node_ops
            .entry(node.onnx_index)
            .or_insert(&node.onnx_op);
        *ops.entry(node.op.code()).or_insert(0) += 1;
    }
    let mut onnx_ops: BTreeMap<&str, usize> = BTreeMap::new();
    for op_type in onnx_node_ops.values() {
        *onnx_ops.entry(op_type).or_insert(0) += 1;
    }
    if folded_constants > 0 {
        *onnx_ops.entry("Constant").or_insert(0) += folded_constants;
    }
    let mut dtypes: BTreeMap<u8, usize> = BTreeMap::new();
    for value in graph.values() {
        *dtypes.entry(value.dtype.code()).or_insert(0) += 1;
    }
    let initializers = weights.initializers().count();
    let onnx_nodes = source_nodes.unwrap_or(onnx_node_ops.len());
    ImportSummary {
        onnx_nodes,
        fn3_nodes: graph.nodes().len(),
        values: graph.value_count(),
        initializers,
        attribute_constants: weights.len() - initializers,
        weight_bytes: weights.total_bytes(),
        onnx_ops: onnx_ops
            .into_iter()
            .map(|(name, count)| (name.to_string(), count))
            .collect(),
        ops: ops
            .into_iter()
            .filter_map(|(code, count)| OpId::from_u16(code).map(|op| (op, count)))
            .collect(),
        dtypes: dtypes
            .into_iter()
            .filter_map(|(code, count)| {
                DType::from_code(code, "import summary")
                    .ok()
                    .map(|dtype| (dtype, count))
            })
            .collect(),
    }
}
