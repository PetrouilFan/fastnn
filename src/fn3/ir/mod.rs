//! The v3 typed graph: the one representation import, planning and execution
//! agree on.
//!
//! # Why a separate graph and not the runtime tensors
//!
//! A [`TensorType`] is a **logical** type: a dtype and an ordered list of
//! extents, each either a constant or one of the decoder's named symbols. It
//! deliberately says nothing about strides, blocking, storage or device. Those
//! are plan data in v3 (ADR-0001 §3), so encoding one here would force every
//! backend that prefers a different layout to re-derive the shape it was given.
//! A shape that needs a stride is a view, and views belong to
//! [`crate::inference::tensor`], not to the graph.
//!
//! # Why a source location on every node
//!
//! An unsupported model must fail before it allocates or executes anything, and
//! the failure has to name the thing the user can edit. [`SourceLoc`] keeps the
//! source node index, the source operator spelling and the output name, so a
//! pass can say `node 7 (onnx "Conv", output "y")` instead of a bare `OpId`.
//! `node_index` is the index in the *source* graph, which is not always the
//! [`NodeId`]: a composition such as `Gemm → MatMul + Add` emits two IR nodes
//! for one source node.
//!
//! # Why an append-only builder
//!
//! [`Graph::add_value`] and [`Graph::add_node`] only append: an id is the
//! position it was assigned, a value is written by exactly one node, and every
//! input already exists when the node is added. That makes the table
//! topologically ordered by construction, so a single forward pass (see
//! [`infer_shapes`]) is sufficient and no cycle check is needed. It also means
//! the graph cannot be mutated into an inconsistent state through its public
//! surface. [`Graph::from_parts`] is the one exception, for a reader that already
//! owns a decoded table; it re-validates every invariant the builder enforces,
//! including that value ids are unique.
//!
//! # Attributes
//!
//! [`OpParams`] is the crate's one attribute carrier. This module does not
//! invent a parallel attribute system; an op's meaning is a documented reading
//! of those fields, and
//! [`infer`] rejects any field the op does not
//! define. The fields the starter set reads:
//!
//! | Attribute | Ops |
//! |---|---|
//! | `axes` = `Transpose.perm` | [`OpId::Transpose`] |
//! | `axes` = `Flatten.axis` | [`OpId::Flatten`] |
//! | `axes` = remove/insert axes | [`OpId::Squeeze`], [`OpId::Unsqueeze`] |
//! | `axes` = constant target shape | [`OpId::Reshape`] (see below) |
//! | `axes` = `[]`/single axis | [`OpId::Concat`], [`OpId::Softmax`], [`OpId::Gather`] |
//! | `axes` + `keep_dims` | [`OpId::ReduceMean`], [`OpId::ReduceSum`] |
//! | `axes` + `epsilon` | [`OpId::LayerNorm`] |
//! | `kernel_shape` + `strides` + `pads` + `dilations` + `group` + `auto_pad` | [`OpId::Conv`] |
//! | the same window fields + `ceil_mode`/`count_include_pad` | [`OpId::MaxPool`], [`OpId::AveragePool`] |
//! | `epsilon` | [`OpId::BatchNorm`] |
//! | `resize_mode` + `coordinate_transformation_mode` + `nearest_mode` + `resize_scales`/`resize_sizes` | [`OpId::Resize`] |
//! | `pads` + `pad_mode` + `pad_value` | [`OpId::Pad`] |
//! | `clip_min` + `clip_max` | [`OpId::Clip`] |
//! | `topk_axis` + `topk_k` + `largest` + `sorted` | [`OpId::TopK`] |
//! | `axes` + `split_sizes` | [`OpId::Split`] |
//! | `axes` + `slice_starts`/`slice_ends`/`slice_steps` | [`OpId::Slice`] |
//!
//! [`OpId::Reshape`], `Resize`, `Pad`, `Clip`, `TopK`, `Split` and `Slice` are
//! the ops whose ONNX kernel reads a *second input*. The IR has no constant
//! payload, so the importer folds that input into the fields above and a shape
//! rule that finds the operand without the folded value reports the node rather
//! than guessing. A runtime shape input is therefore rejected by shape
//! inference instead of being passed through as an unchecked extent.
//!
//! # Determinism
//!
//! Every table is a `Vec`, every iteration is in index order, and no public
//! output depends on hash iteration. Two graphs built from the same source in
//! the same order are equal, and two inference runs produce the same types.
//!
//! # Validation is typed, never a panic
//!
//! Construction returns [`Fn3Error`]. Ids are bounds-checked, arity comes from
//! [`OpId`] (the same counts the alpha kernels check with
//! `backend::check_arity`), each node writes at least one value, a value is
//! written once, and a fully static element count is checked in `u64` and
//! `usize` exactly as `Shape::numel_usize` does. An extent of zero is legal, as
//! it is for [`Shape`]: the tensor is simply empty. A symbolic extent is checked
//! when the plan binds it (`ShapeEnvelope` caps every symbol at
//! [`DimBound::MAX_EXTENT`](crate::inference::shape::DimBound::MAX_EXTENT) and
//! `Shape::numel` checks the product); the IR deliberately does not multiply
//! every symbol's independent cap, because that would reject an ordinary
//! four-symbol shape such as `[batch, seq, n_heads, head_dim]` for a bound no
//! plan could actually request.

use std::fmt;

use crate::inference::backend::OpParams;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::shape::{Shape, Symbol};

pub mod infer;

pub use infer::infer_shapes;

/// Largest rank a [`TensorType`] may declare.
///
/// Equal to the importer's and artifact's policy cap. The IR cannot import
/// `import::budget` (that module sits above it), so the value is stated here
/// once; a graph is rejected at construction rather than costing an unbounded
/// `Vec<DimExpr>` per value.
pub const MAX_RANK: usize = 64;

/// Largest number of inputs or outputs one node may declare.
///
/// Equal to the artifact's operand cap. Every concrete op has a tighter
/// per-op arity (see `op_arity`); the cap only bounds the variable-arity ops
/// (`Concat`, `Split`) and any future op the table does not describe yet.
pub const MAX_OPERANDS: usize = 1024;

/// One axis of a logical shape.
///
/// There is no `Unknown` variant on purpose: a pass that cannot prove an extent
/// returns a typed error naming the node instead of carrying an extent it does
/// not know. Symbols are the only legitimate uncertainty, because the session's
/// [`LiveDims`](crate::inference::shape::LiveDims) resolves them before
/// execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DimExpr {
    /// A fixed extent. Zero is legal and means the tensor has no elements on
    /// this axis.
    Const(u64),
    /// A named dynamic dimension, resolved by the plan's envelope.
    Symbol(Symbol),
}

impl DimExpr {
    /// A fixed extent.
    pub const fn constant(extent: u64) -> DimExpr {
        DimExpr::Const(extent)
    }

    /// A named dynamic dimension.
    pub const fn symbol(symbol: Symbol) -> DimExpr {
        DimExpr::Symbol(symbol)
    }

    /// The extent when it is fixed.
    pub const fn as_const(self) -> Option<u64> {
        match self {
            DimExpr::Const(extent) => Some(extent),
            DimExpr::Symbol(_) => None,
        }
    }

    /// The symbol when this axis is dynamic.
    pub const fn as_symbol(self) -> Option<Symbol> {
        match self {
            DimExpr::Const(_) => None,
            DimExpr::Symbol(symbol) => Some(symbol),
        }
    }
}

impl fmt::Display for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimExpr::Const(extent) => write!(f, "{extent}"),
            DimExpr::Symbol(symbol) => f.write_str(symbol.name()),
        }
    }
}

/// The logical type of one value: a dtype and constant-or-symbolic extents.
///
/// This is not a layout, a view or an allocation request. See the module docs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    /// Element type.
    pub dtype: DType,
    /// Extents, outermost first.
    pub shape: Vec<DimExpr>,
}

impl TensorType {
    /// Construct from a dtype and extents.
    ///
    /// The result is not validated until it is handed to [`Graph::add_value`],
    /// which is the only public way into a graph.
    pub fn new(dtype: DType, shape: impl IntoIterator<Item = DimExpr>) -> TensorType {
        TensorType {
            dtype,
            shape: shape.into_iter().collect(),
        }
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The extents, outermost first.
    pub fn dims(&self) -> &[DimExpr] {
        &self.shape
    }

    /// `true` when every axis is a constant.
    pub fn is_static(&self) -> bool {
        self.shape.iter().all(|dim| dim.as_const().is_some())
    }

    /// The concrete shape, when every axis is constant.
    pub fn static_shape(&self) -> Option<Shape> {
        let mut extents = Vec::with_capacity(self.shape.len());
        for dim in &self.shape {
            extents.push(dim.as_const()?);
        }
        Some(Shape::from_dims(extents))
    }

    /// The concrete element count, when every axis is constant.
    ///
    /// `None` when a symbol is present: the count depends on the session's
    /// envelope and is checked by [`Shape::numel`] when the plan binds it.
    /// Multiplying every symbol's independent cap here would overflow an
    /// ordinary four-symbol shape and reject a graph no allocation could reach.
    /// A zero extent short-circuits before multiplication, matching `Shape::numel`.
    pub fn static_numel(&self) -> Result<Option<u64>, Fn3Error> {
        match self.static_shape() {
            Some(shape) => shape.numel().map(Some),
            None => Ok(None),
        }
    }

    /// Validate the rank and, for a fully static type, the element count.
    ///
    /// The count goes through [`Shape::numel_usize`], which checks both `u64`
    /// overflow and the `usize` width and never truncates on a 32-bit target.
    /// A zero extent is accepted: an empty tensor is legal. A symbolic extent is
    /// not a count yet; it is validated against the envelope when it is bound.
    pub fn validate(&self) -> Result<(), Fn3Error> {
        if self.rank() > MAX_RANK {
            return Err(Fn3Error::InvalidShape {
                context: "ir TensorType::validate",
                detail: format!(
                    "rank {} exceeds the IR limit of {MAX_RANK} axes",
                    self.rank()
                ),
            });
        }
        if let Some(shape) = self.static_shape() {
            shape.numel_usize()?;
        }
        Ok(())
    }

    /// The same dtype on a new shape, as an inference pass produces.
    pub fn with_shape(&self, shape: Vec<DimExpr>) -> TensorType {
        TensorType {
            dtype: self.dtype,
            shape,
        }
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[", self.dtype)?;
        for (index, dim) in self.shape.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{dim}")?;
        }
        f.write_str("]")
    }
}

/// Where a node came from in the source model.
///
/// Kept on every [`Node`] so a pass can reject an unsupported model with a
/// node-level message before any allocation or execution. `node_index` is the
/// source index, not a [`NodeId`]: one source node may become several IR nodes
/// (`Gemm` becomes `MatMul` and `Add`), and all of them keep the source index so
/// the user can find the line in their model.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceLoc {
    /// Index of the node in the source graph.
    pub node_index: usize,
    /// Operator spelling as the source wrote it (`"Conv"`, `"Add"`, `"Gemm"`).
    pub op_type: String,
    /// Name of the source node's output, when it had one.
    pub output: Option<String>,
}

impl SourceLoc {
    /// A location with no output name.
    pub fn new(node_index: usize, op_type: impl Into<String>) -> SourceLoc {
        SourceLoc {
            node_index,
            op_type: op_type.into(),
            output: None,
        }
    }

    /// Builder: attach the source output name.
    pub fn with_output(mut self, output: impl Into<String>) -> SourceLoc {
        self.output = Some(output.into());
        self
    }

    /// Render for a diagnostic: `node 7 (onnx "Conv", output "y")`.
    pub fn describe(&self) -> String {
        let mut detail = String::new();
        if !self.op_type.is_empty() {
            detail.push_str(&format!("onnx {:?}", self.op_type));
        }
        if let Some(output) = &self.output {
            if !detail.is_empty() {
                detail.push_str(", ");
            }
            detail.push_str(&format!("output {output:?}"));
        }
        if detail.is_empty() {
            format!("node {}", self.node_index)
        } else {
            format!("node {} ({detail})", self.node_index)
        }
    }
}

impl fmt::Display for SourceLoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

/// Position of a value in the graph's value table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(u32);

impl ValueId {
    /// Construct from a zero-based table position.
    pub const fn new(index: u32) -> ValueId {
        ValueId(index)
    }

    /// Zero-based table position.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "value {}", self.0)
    }
}

/// Position of a node in the graph's node table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u32);

impl NodeId {
    /// Construct from a zero-based table position.
    pub const fn new(index: u32) -> NodeId {
        NodeId(index)
    }

    /// Zero-based table position.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node {}", self.0)
    }
}

/// One value: a logical type and the node that writes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    /// Position in the value table. Always equal to the position.
    pub id: ValueId,
    /// Source name, when the source had one. Names are diagnostics, not keys;
    /// a source may reuse one, so lookup is by [`ValueId`].
    pub name: Option<String>,
    /// Logical type.
    pub ty: TensorType,
    /// The node that writes this value. `None` means a graph input or a
    /// constant supplied by the loader.
    pub producer: Option<NodeId>,
}

/// One operation, in execution order.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Position in the node table.
    pub id: NodeId,
    /// Operation to execute.
    pub op: OpId,
    /// Values read, in operand order.
    pub inputs: Vec<ValueId>,
    /// Values written, in result order. Never empty.
    pub outputs: Vec<ValueId>,
    /// Attributes, in the crate's single attribute carrier.
    pub params: OpParams,
    /// Where the node came from.
    pub loc: SourceLoc,
}

impl Node {
    /// The first output, which is the node's result for every one-output op.
    pub fn primary_output(&self) -> Option<ValueId> {
        self.outputs.first().copied()
    }

    /// Source-level label for a diagnostic.
    pub fn label(&self) -> String {
        self.loc.describe()
    }
}

/// Input and output counts accepted for one op.
#[derive(Debug, Clone, Copy)]
struct Arity {
    min_inputs: usize,
    max_inputs: usize,
    min_outputs: usize,
    max_outputs: usize,
}

impl Arity {
    const fn fixed(inputs: usize, outputs: usize) -> Arity {
        Arity {
            min_inputs: inputs,
            max_inputs: inputs,
            min_outputs: outputs,
            max_outputs: outputs,
        }
    }
}

/// The operand counts the alpha kernels enforce, expressed once for the IR.
///
/// `None` means no authoritative arity exists yet (the reserved fused id); the
/// generic [`MAX_OPERANDS`] cap still applies. Keeping the counts here lets a
/// malformed graph fail while it is built, before a backend ever sees it.
fn op_arity(op: OpId) -> Option<Arity> {
    match op {
        OpId::Gather | OpId::MatMul => Some(Arity::fixed(2, 1)),
        OpId::Add | OpId::Sub | OpId::Mul | OpId::Div | OpId::Pow => Some(Arity::fixed(2, 1)),
        OpId::Neg
        | OpId::Sqrt
        | OpId::Erf
        | OpId::Tanh
        | OpId::Gelu
        | OpId::Relu
        | OpId::Sin
        | OpId::Cos => Some(Arity::fixed(1, 1)),
        OpId::Softmax => Some(Arity::fixed(1, 1)),
        OpId::LayerNorm => Some(Arity::fixed(3, 1)),
        OpId::ReduceMean | OpId::ReduceSum | OpId::ArgMax => Some(Arity::fixed(1, 1)),
        OpId::Transpose | OpId::Flatten => Some(Arity::fixed(1, 1)),
        // Reshape has a constant-target form (one input, target in `axes`) and
        // a runtime-shape form (two inputs); shape inference only resolves the
        // first, but both are valid IR.
        OpId::Reshape => Some(Arity {
            min_inputs: 1,
            max_inputs: 2,
            min_outputs: 1,
            max_outputs: 1,
        }),
        OpId::Squeeze | OpId::Unsqueeze => Some(Arity {
            min_inputs: 1,
            max_inputs: 2,
            min_outputs: 1,
            max_outputs: 1,
        }),
        OpId::Concat => Some(Arity {
            min_inputs: 1,
            max_inputs: MAX_OPERANDS,
            min_outputs: 1,
            max_outputs: 1,
        }),
        // The typed IR may carry the constant bounds in `OpParams` instead
        // of the starts/ends/axes/steps operands, so one input is legal.
        OpId::Slice => Some(Arity {
            min_inputs: 1,
            max_inputs: 5,
            min_outputs: 1,
            max_outputs: 1,
        }),
        OpId::Split => Some(Arity {
            min_inputs: 1,
            max_inputs: 2,
            min_outputs: 1,
            max_outputs: MAX_OPERANDS,
        }),
        OpId::Shape | OpId::Cast => Some(Arity::fixed(1, 1)),
        OpId::ConstantOfShape => Some(Arity::fixed(2, 1)),
        OpId::Expand => Some(Arity::fixed(2, 1)),
        OpId::Range => Some(Arity::fixed(3, 1)),
        OpId::Equal | OpId::Less | OpId::Greater => Some(Arity::fixed(2, 1)),
        OpId::Where | OpId::ScatterND => Some(Arity::fixed(3, 1)),
        OpId::Trilu => Some(Arity {
            min_inputs: 1,
            max_inputs: 2,
            min_outputs: 1,
            max_outputs: 1,
        }),
        // CNN ops. A `Resize` may carry the ONNX roi, scales and sizes inputs,
        // a `Pad` its pads and fill value, and a `Clip` its min and max; the
        // importer folds the constant payloads into `OpParams`, but the
        // operands stay in the graph so the node mirrors its source.
        OpId::Conv => Some(Arity {
            min_inputs: 2,
            max_inputs: 3,
            min_outputs: 1,
            max_outputs: 1,
        }),
        OpId::MaxPool | OpId::AveragePool | OpId::GlobalAveragePool => Some(Arity::fixed(1, 1)),
        OpId::BatchNorm => Some(Arity::fixed(5, 1)),
        OpId::Resize => Some(Arity {
            min_inputs: 1,
            max_inputs: 4,
            min_outputs: 1,
            max_outputs: 1,
        }),
        OpId::Pad | OpId::Clip => Some(Arity {
            min_inputs: 1,
            max_inputs: 3,
            min_outputs: 1,
            max_outputs: 1,
        }),
        OpId::Sigmoid => Some(Arity::fixed(1, 1)),
        OpId::Maximum | OpId::Minimum => Some(Arity::fixed(2, 1)),
        // TopK writes both values and i64 indices.
        OpId::TopK => Some(Arity {
            min_inputs: 1,
            max_inputs: 2,
            min_outputs: 2,
            max_outputs: 2,
        }),
        // Reserved for a fused backend and always rejected at execution; an
        // arity would be a guess about a contract that does not exist yet.
        OpId::MaskedAttention => None,
    }
}

fn check_node_arity(op: OpId, inputs: usize, outputs: usize) -> Result<(), Fn3Error> {
    match op_arity(op) {
        Some(arity) => {
            if inputs >= arity.min_inputs
                && inputs <= arity.max_inputs
                && outputs >= arity.min_outputs
                && outputs <= arity.max_outputs
            {
                Ok(())
            } else {
                Err(Fn3Error::ArityMismatch {
                    op,
                    expected_inputs: if inputs < arity.min_inputs {
                        arity.min_inputs
                    } else {
                        arity.max_inputs
                    },
                    actual_inputs: inputs,
                    expected_outputs: if outputs < arity.min_outputs {
                        arity.min_outputs
                    } else {
                        arity.max_outputs
                    },
                    actual_outputs: outputs,
                })
            }
        }
        None => {
            if inputs <= MAX_OPERANDS && outputs <= MAX_OPERANDS {
                Ok(())
            } else {
                Err(Fn3Error::ArityMismatch {
                    op,
                    expected_inputs: MAX_OPERANDS,
                    actual_inputs: inputs,
                    expected_outputs: MAX_OPERANDS,
                    actual_outputs: outputs,
                })
            }
        }
    }
}

/// Values and nodes, in execution order.
///
/// Build with [`Graph::new`], [`Graph::add_value`] and [`Graph::add_node`].
/// There is no public mutable access to the tables: a pass that resolves types
/// uses [`infer_shapes`], which re-validates every type it writes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Graph {
    values: Vec<Value>,
    nodes: Vec<Node>,
}

impl Graph {
    /// An empty graph.
    pub fn new() -> Graph {
        Graph::default()
    }

    /// Every value, indexed by [`ValueId`].
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Every node, in execution order.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// One value, or `None` when the id is out of range.
    pub fn value(&self, id: ValueId) -> Option<&Value> {
        self.values.get(id.index())
    }

    /// One node, or `None` when the id is out of range.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.index())
    }

    /// Number of values.
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The producer of a value, if a node writes it.
    pub fn producer(&self, id: ValueId) -> Option<NodeId> {
        self.value(id).and_then(|value| value.producer)
    }

    /// Append a value and return its id.
    ///
    /// The id is the table position, so it cannot collide and the caller does
    /// not supply it. `name` is optional and only used in diagnostics. The type
    /// is validated here: rank, checked element count and the `usize` bound.
    pub fn add_value(&mut self, name: Option<&str>, ty: TensorType) -> Result<ValueId, Fn3Error> {
        ty.validate()?;
        let index = u32::try_from(self.values.len()).map_err(|_| Fn3Error::CapacityExceeded {
            what: "ir value table",
            requested: self.values.len() as u64 + 1,
            capacity: u64::from(u32::MAX),
        })?;
        let id = ValueId::new(index);
        self.values.push(Value {
            id,
            name: name.map(str::to_string),
            ty,
            producer: None,
        });
        Ok(id)
    }

    /// Append a node and return its id.
    ///
    /// Checks, in order: at least one output; the op's arity; every input and
    /// output id is in range; an output is not also an input; no output already
    /// has a producer; no output appears twice. Only after all of that does the
    /// table grow, so a rejected node leaves the graph unchanged.
    pub fn add_node(
        &mut self,
        op: OpId,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        params: OpParams,
        loc: SourceLoc,
    ) -> Result<NodeId, Fn3Error> {
        if outputs.is_empty() {
            return Err(Fn3Error::InvalidShape {
                context: "ir Graph::add_node",
                detail: format!("{}: op {op} has no output value", loc.describe()),
            });
        }
        check_node_arity(op, inputs.len(), outputs.len())?;
        for &input in &inputs {
            self.require_value(input)?;
        }
        for (position, &output) in outputs.iter().enumerate() {
            let value = self.require_value(output)?;
            if inputs.contains(&output) {
                return Err(Fn3Error::InvalidShape {
                    context: "ir Graph::add_node",
                    detail: format!("{}: op {op} reads and writes {output}", loc.describe()),
                });
            }
            if outputs[..position].contains(&output) {
                return Err(Fn3Error::InvalidShape {
                    context: "ir Graph::add_node",
                    detail: format!("{}: op {op} writes {output} twice", loc.describe()),
                });
            }
            if let Some(producer) = value.producer {
                return Err(Fn3Error::InvalidShape {
                    context: "ir Graph::add_node",
                    detail: format!(
                        "{}: op {op} writes {output}, which is already produced by {producer}",
                        loc.describe()
                    ),
                });
            }
        }
        let index = u32::try_from(self.nodes.len()).map_err(|_| Fn3Error::CapacityExceeded {
            what: "ir node table",
            requested: self.nodes.len() as u64 + 1,
            capacity: u64::from(u32::MAX),
        })?;
        let id = NodeId::new(index);
        for &output in &outputs {
            if let Some(value) = self.values.get_mut(output.index()) {
                value.producer = Some(id);
            }
        }
        self.nodes.push(Node {
            id,
            op,
            inputs,
            outputs,
            params,
            loc,
        });
        Ok(id)
    }

    /// Rebuild a graph from tables a reader already decoded.
    ///
    /// The builder is the normal path. A prepared-model reader has the tables
    /// first and needs the identical container without an ONNX file present
    /// (the same seam `import::GraphIr::from_parts` provides). This performs no
    /// guessing: it validates id positions and uniqueness, producer
    /// consistency, topological order, arity, and every type.
    pub fn from_parts(values: Vec<Value>, nodes: Vec<Node>) -> Result<Graph, Fn3Error> {
        for (position, value) in values.iter().enumerate() {
            let expected = u32::try_from(position).map_err(|_| Fn3Error::CapacityExceeded {
                what: "ir value table",
                requested: (position as u64).saturating_add(1),
                capacity: u64::from(u32::MAX),
            })?;
            if value.id != ValueId::new(expected) {
                return Err(Fn3Error::InvalidShape {
                    context: "ir Graph::from_parts",
                    detail: format!(
                        "value table position {position} holds {}; value ids must be unique and \
                         equal to their position",
                        value.id
                    ),
                });
            }
            value.ty.validate()?;
        }
        for (position, node) in nodes.iter().enumerate() {
            let expected = u32::try_from(position).map_err(|_| Fn3Error::CapacityExceeded {
                what: "ir node table",
                requested: (position as u64).saturating_add(1),
                capacity: u64::from(u32::MAX),
            })?;
            if node.id != NodeId::new(expected) {
                return Err(Fn3Error::InvalidShape {
                    context: "ir Graph::from_parts",
                    detail: format!(
                        "node table position {position} holds {}; node ids must be unique and \
                         equal to their position",
                        node.id
                    ),
                });
            }
            if node.outputs.is_empty() {
                return Err(Fn3Error::InvalidShape {
                    context: "ir Graph::from_parts",
                    detail: format!("{}: op {} has no output value", node.label(), node.op),
                });
            }
            check_node_arity(node.op, node.inputs.len(), node.outputs.len())?;
            for &input in &node.inputs {
                let value = value_at(&values, input)?;
                if let Some(producer) = value.producer {
                    if producer.index() >= position {
                        return Err(Fn3Error::InvalidShape {
                            context: "ir Graph::from_parts",
                            detail: format!(
                                "{}: reads {input}, produced by {producer} at or after this \
                                 node; the IR must be in topological order",
                                node.label()
                            ),
                        });
                    }
                }
            }
            for (offset, &output) in node.outputs.iter().enumerate() {
                let value = value_at(&values, output)?;
                if value.producer != Some(node.id) {
                    return Err(Fn3Error::InvalidShape {
                        context: "ir Graph::from_parts",
                        detail: format!(
                            "{}: writes {output}, but the value records producer {:?}",
                            node.label(),
                            value.producer
                        ),
                    });
                }
                if node.inputs.contains(&output) {
                    return Err(Fn3Error::InvalidShape {
                        context: "ir Graph::from_parts",
                        detail: format!(
                            "{}: op {} reads and writes {output}",
                            node.label(),
                            node.op
                        ),
                    });
                }
                if node.outputs[..offset].contains(&output) {
                    return Err(Fn3Error::InvalidShape {
                        context: "ir Graph::from_parts",
                        detail: format!("{}: op {} writes {output} twice", node.label(), node.op),
                    });
                }
            }
        }
        for value in &values {
            if let Some(producer) = value.producer {
                let node = nodes
                    .get(producer.index())
                    .ok_or_else(|| Fn3Error::InvalidShape {
                        context: "ir Graph::from_parts",
                        detail: format!(
                            "{} records {producer} as its producer, which is out of range",
                            value.id
                        ),
                    })?;
                if !node.outputs.contains(&value.id) {
                    return Err(Fn3Error::InvalidShape {
                        context: "ir Graph::from_parts",
                        detail: format!(
                            "{} records {producer} as its producer, but that node does not \
                             write it",
                            value.id
                        ),
                    });
                }
            }
        }
        Ok(Graph { values, nodes })
    }

    fn require_value(&self, id: ValueId) -> Result<&Value, Fn3Error> {
        self.value(id).ok_or(Fn3Error::OutOfBounds {
            what: "ir value id",
            index: u64::from(id.0),
            len: self.values.len() as u64,
        })
    }

    /// Overwrite one value's type, for the inference pass only.
    ///
    /// The caller has already validated the type with
    /// [`TensorType::validate`]; this only bounds-checks the id.
    pub(super) fn set_value_type(&mut self, id: ValueId, ty: TensorType) -> Result<(), Fn3Error> {
        let len = self.values.len() as u64;
        let value = self
            .values
            .get_mut(id.index())
            .ok_or(Fn3Error::OutOfBounds {
                what: "ir value id",
                index: u64::from(id.0),
                len,
            })?;
        value.ty = ty;
        Ok(())
    }
}

fn value_at(values: &[Value], id: ValueId) -> Result<&Value, Fn3Error> {
    values.get(id.index()).ok_or(Fn3Error::OutOfBounds {
        what: "ir value id",
        index: u64::from(id.0),
        len: values.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_type(shape: Vec<DimExpr>) -> TensorType {
        TensorType::new(DType::F32, shape)
    }

    fn consts(extents: &[u64]) -> Vec<DimExpr> {
        extents.iter().copied().map(DimExpr::constant).collect()
    }

    fn add(
        graph: &mut Graph,
        op: OpId,
        inputs: &[ValueId],
        outputs: &[ValueId],
        node_index: usize,
    ) -> NodeId {
        graph
            .add_node(
                op,
                inputs.to_vec(),
                outputs.to_vec(),
                OpParams::new(),
                SourceLoc::new(node_index, op.name()),
            )
            .expect("the test graph is valid")
    }

    #[test]
    fn builder_records_values_producers_and_order() {
        let mut graph = Graph::new();
        let lhs = graph
            .add_value(Some("x"), f32_type(consts(&[2, 3])))
            .unwrap();
        let rhs = graph
            .add_value(Some("w"), f32_type(consts(&[3, 4])))
            .unwrap();
        let out = graph.add_value(None, f32_type(Vec::new())).unwrap();
        // The source index (5) is deliberately not the IR node index (0): a
        // source node may become several IR nodes, and this test pins that the
        // diagnostic keeps the source index.
        let node = graph
            .add_node(
                OpId::MatMul,
                vec![lhs, rhs],
                vec![out],
                OpParams::new(),
                SourceLoc::new(5, "MatMul").with_output("logits"),
            )
            .unwrap();

        assert_eq!(graph.value_count(), 3);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.value(lhs).unwrap().name.as_deref(), Some("x"));
        assert_eq!(graph.value(lhs).unwrap().producer, None);
        assert_eq!(graph.producer(out), Some(node));
        assert_eq!(graph.node(node).unwrap().primary_output(), Some(out));
        assert_eq!(
            graph.node(node).unwrap().loc.describe(),
            "node 5 (onnx \"MatMul\", output \"logits\")"
        );
    }

    #[test]
    fn out_of_range_input_id_is_rejected_without_growing_the_graph() {
        let mut graph = Graph::new();
        let value = graph.add_value(None, f32_type(consts(&[4]))).unwrap();
        let missing = ValueId::new(7);
        let error = graph
            .add_node(
                OpId::Relu,
                vec![missing],
                vec![value],
                OpParams::new(),
                SourceLoc::new(0, "Relu"),
            )
            .unwrap_err();
        assert_eq!(
            error,
            Fn3Error::OutOfBounds {
                what: "ir value id",
                index: 7,
                len: 1
            }
        );
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn arity_and_empty_outputs_are_rejected() {
        let mut graph = Graph::new();
        let a = graph.add_value(None, f32_type(consts(&[2]))).unwrap();
        let b = graph.add_value(None, f32_type(consts(&[2]))).unwrap();

        let arity = graph
            .add_node(
                OpId::Add,
                vec![a],
                vec![b],
                OpParams::new(),
                SourceLoc::new(0, "Add"),
            )
            .unwrap_err();
        assert!(matches!(
            arity,
            Fn3Error::ArityMismatch {
                op: OpId::Add,
                expected_inputs: 2,
                actual_inputs: 1,
                ..
            }
        ));

        let empty = graph
            .add_node(
                OpId::Relu,
                vec![a],
                Vec::new(),
                OpParams::new(),
                SourceLoc::new(1, "Relu"),
            )
            .unwrap_err();
        let text = empty.to_string();
        assert!(text.contains("node 1"), "{text}");
        assert!(text.contains("no output"), "{text}");
    }

    #[test]
    fn a_value_is_written_once() {
        let mut graph = Graph::new();
        let input = graph.add_value(None, f32_type(consts(&[2]))).unwrap();
        let output = graph.add_value(None, f32_type(consts(&[2]))).unwrap();
        add(&mut graph, OpId::Relu, &[input], &[output], 0);

        let second = graph
            .add_node(
                OpId::Neg,
                vec![input],
                vec![output],
                OpParams::new(),
                SourceLoc::new(1, "Neg"),
            )
            .unwrap_err();
        let text = second.to_string();
        assert!(text.contains("already produced"), "{text}");
    }

    #[test]
    fn duplicate_value_ids_are_rejected_by_from_parts() {
        let duplicated = Value {
            id: ValueId::new(0),
            name: None,
            ty: f32_type(consts(&[2])),
            producer: None,
        };
        let error =
            Graph::from_parts(vec![duplicated.clone(), duplicated], Vec::new()).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("position 1"), "{text}");
        assert!(text.contains("unique"), "{text}");
    }

    #[test]
    fn zero_extents_are_legal_and_short_circuit_the_count() {
        let mut graph = Graph::new();
        let empty = graph
            .add_value(
                None,
                f32_type(vec![
                    DimExpr::symbol(Symbol::Batch),
                    DimExpr::constant(0),
                    DimExpr::constant(8),
                ]),
            )
            .unwrap();
        let ty = &graph.value(empty).unwrap().ty;
        // A symbolic type has no static count yet; the envelope resolves it.
        assert_eq!(ty.static_numel(), Ok(None));
        assert_eq!(ty.to_string(), "f32[batch, 0, 8]");
        assert!(!ty.is_static());

        let fully_static = graph.add_value(None, f32_type(consts(&[0, 8]))).unwrap();
        assert_eq!(
            graph.value(fully_static).unwrap().ty.static_numel(),
            Ok(Some(0))
        );
    }

    #[test]
    fn a_four_symbol_shape_is_not_rejected() {
        // Each symbol's independent cap is 2^24; multiplying four of them
        // overflows u64. That is a false bound, not a plan, so the IR must
        // accept the shape and leave the count to the envelope.
        let mut graph = Graph::new();
        graph
            .add_value(
                None,
                f32_type(vec![
                    DimExpr::symbol(Symbol::Batch),
                    DimExpr::symbol(Symbol::Seq),
                    DimExpr::symbol(Symbol::NHeads),
                    DimExpr::symbol(Symbol::HeadDim),
                ]),
            )
            .unwrap();
    }

    #[test]
    fn rank_above_the_policy_cap_is_rejected() {
        let mut graph = Graph::new();
        let error = graph
            .add_value(None, f32_type(vec![DimExpr::constant(1); MAX_RANK + 1]))
            .unwrap_err();
        assert!(matches!(error, Fn3Error::InvalidShape { .. }));
    }
}
