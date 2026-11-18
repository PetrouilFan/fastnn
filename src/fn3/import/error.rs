//! Typed import errors: a bounds-checked wire error and an accumulated
//! diagnostic list.
//!
//! # Why two layers
//!
//! A malformed protobuf stream cannot be walked any further: the byte that says
//! where the next field starts is the byte that failed, so the reader reports
//! exactly one [`ProtoError`] and stops. Once the stream decodes, every
//! remaining check is independent, so the importer keeps going and reports
//! *all* of them: a model with three unsupported nodes yields three problems in
//! file order, not the first one.
//!
//! # Why not [`crate::fn3::error::Fn3Error`]
//!
//! `Fn3Error` has one payload per variant and belongs to the core contract
//! (FN3-02 / #48). "Six problems, each naming a node" has no representation
//! there, and inventing one would edit a shared contract. [`ImportError`]
//! therefore lives here and converts into `Fn3Error::InvalidFormat` for callers
//! that only speak the core error type.
//!
//! Every message names the node index, the node name and the ONNX op type, so a
//! user can find the node in the file. A check that cannot name its node is a
//! check that has not been designed yet.

use std::fmt;

use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;

/// Maximum number of problems one import reports before summarising.
///
/// The cap exists so a hostile or badly broken file cannot make the error value
/// larger than the file. Hitting it is itself reported
/// ([`ImportProblemKind::TooManyProblems`]), so a truncated list is never
/// mistaken for a complete one.
pub const MAX_PROBLEMS: usize = 100;

/// A byte-level failure while decoding the protobuf wire format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtoError {
    /// A length prefix or fixed-width field claimed more bytes than remain.
    Truncated {
        /// Absolute byte offset of the field payload.
        offset: usize,
        /// Bytes the field claimed.
        needed: usize,
        /// Bytes that were actually left.
        available: usize,
    },
    /// A varint did not terminate within ten bytes.
    VarintTooLong {
        /// Absolute byte offset where the varint started.
        offset: usize,
    },
    /// A tag decoded to field number zero, which protobuf forbids.
    InvalidFieldNumber {
        /// Absolute byte offset of the tag.
        offset: usize,
    },
    /// A tag used a wire type this reader does not implement (3, 4, 6, 7).
    ///
    /// ONNX never emits groups; a group tag means the bytes are not an ONNX
    /// model (or are corrupt), and skipping it would be guesswork.
    UnsupportedWireType {
        /// Field number that carried the wire type.
        field: u32,
        /// The wire type code.
        wire: u8,
        /// Absolute byte offset of the tag.
        offset: usize,
    },
    /// A known field used a wire type its schema does not allow.
    WrongWireType {
        /// Field number.
        field: u32,
        /// What the ONNX schema requires, e.g. `"length-delimited"`.
        expected: &'static str,
        /// The wire type code found.
        found: u8,
        /// Absolute byte offset of the tag.
        offset: usize,
    },
    /// A field that the ONNX schema requires was absent.
    MissingField {
        /// Enclosing message, e.g. `"ModelProto"`.
        message: &'static str,
        /// Required field name.
        field: &'static str,
    },
    /// A string field was not valid UTF-8.
    InvalidUtf8 {
        /// What the field describes, e.g. `"node name"`.
        what: &'static str,
        /// Absolute byte offset of the offending bytes.
        offset: usize,
    },
    /// A length-delimited field was nested deeper than the ONNX schema allows.
    DepthExceeded {
        /// Absolute byte offset of the nested message.
        offset: usize,
    },
    /// One protobuf message carried two payload fields that are mutually
    /// exclusive (for example `AttributeProto.f` and `AttributeProto.ints`).
    ConflictingPayload {
        /// Enclosing message.
        message: &'static str,
        /// Which payloads were found.
        detail: String,
    },
}

impl fmt::Display for ProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtoError::Truncated {
                offset,
                needed,
                available,
            } => write!(
                f,
                "truncated protobuf at byte {offset}: a field needs {needed} bytes but only \
                 {available} remain"
            ),
            ProtoError::VarintTooLong { offset } => write!(
                f,
                "malformed protobuf varint at byte {offset}: no terminator within ten bytes"
            ),
            ProtoError::InvalidFieldNumber { offset } => write!(
                f,
                "malformed protobuf tag at byte {offset}: field number zero is not allowed"
            ),
            ProtoError::UnsupportedWireType {
                field,
                wire,
                offset,
            } => write!(
                f,
                "unsupported protobuf wire type {wire} for field {field} at byte {offset}; this \
                 reader implements varint (0), 64-bit (1), length-delimited (2) and 32-bit (5) and \
                 treats protobuf groups as corrupt input"
            ),
            ProtoError::WrongWireType {
                field,
                expected,
                found,
                offset,
            } => write!(
                f,
                "protobuf field {field} at byte {offset} must be {expected}, found wire type {found}"
            ),
            ProtoError::MissingField { message, field } => {
                write!(f, "{message} is missing the required field {field}")
            }
            ProtoError::InvalidUtf8 { what, offset } => {
                write!(f, "{what} at byte {offset} is not valid UTF-8")
            }
            ProtoError::DepthExceeded { offset } => write!(
                f,
                "protobuf message at byte {offset} is nested deeper than the ONNX schema allows"
            ),
            ProtoError::ConflictingPayload { message, detail } => write!(
                f,
                "{message} carries conflicting payloads: {detail}; exactly one payload field is \
                 allowed"
            ),
        }
    }
}

impl std::error::Error for ProtoError {}

/// The node a problem belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeRef {
    /// Zero-based index of the node in `GraphProto::node`.
    pub index: u32,
    /// `NodeProto::name`, empty when the exporter left it blank.
    pub name: String,
    /// `NodeProto::op_type`, exactly as the file spelled it.
    pub op_type: String,
}

impl NodeRef {
    /// Build a node reference.
    pub fn new(index: u32, name: impl Into<String>, op_type: impl Into<String>) -> NodeRef {
        NodeRef {
            index,
            name: name.into(),
            op_type: op_type.into(),
        }
    }
}

impl fmt::Display for NodeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.name.is_empty() {
            write!(f, "node {} ({})", self.index, self.op_type)
        } else {
            write!(f, "node {} `{}` ({})", self.index, self.name, self.op_type)
        }
    }
}

/// What went wrong, independent of which node it belongs to.
#[derive(Debug, Clone, PartialEq)]
pub enum ImportProblemKind {
    /// The bytes are not a decodable protobuf stream.
    Proto(ProtoError),
    /// `ModelProto.ir_version` is outside the range this reader understands.
    UnsupportedIrVersion {
        /// Version found in the file.
        found: i64,
        /// Range this build accepts.
        supported: String,
    },
    /// The model has no `GraphProto`.
    MissingGraph,
    /// No `OperatorSetIdProto` for the default ONNX domain.
    MissingOpset,
    /// The default-domain opset is outside the range this reader implements.
    UnsupportedOpset {
        /// Version found in the file.
        version: i64,
        /// Range this build accepts, with the reason.
        supported: String,
    },
    /// The model imports operators from a domain other than the default one.
    UnsupportedDomain {
        /// The domain as spelled in the file.
        domain: String,
    },
    /// A tensor (or a graph input/output) uses a dtype the 3.0 core does not
    /// define.
    UnsupportedDType {
        /// ONNX `TensorProto.DataType` code.
        code: i32,
        /// ONNX mnemonics for that code, e.g. `"FLOAT16"`, when known.
        onnx_name: &'static str,
        /// What carried the dtype, e.g. `"initializer `w`"`.
        carrier: String,
    },
    /// A tensor stores its payload outside the model file.
    ExternalTensorData {
        /// Tensor name, possibly empty.
        tensor: String,
    },
    /// A tensor is stored in segments, which ONNX allows and this build does
    /// not implement.
    SegmentedTensor {
        /// Tensor name.
        tensor: String,
    },
    /// A tensor carries both `raw_data` and a typed payload field.
    AmbiguousTensorData {
        /// Tensor name.
        tensor: String,
        /// Which payload fields were set.
        fields: String,
    },
    /// A tensor has no payload at all.
    MissingTensorData {
        /// Tensor name.
        tensor: String,
    },
    /// A tensor's payload does not match its declared shape and dtype.
    TensorLength {
        /// Tensor name.
        tensor: String,
        /// Declared dtype.
        dtype: DType,
        /// Declared shape, rendered.
        shape: String,
        /// Bytes the shape and dtype require.
        expected: u64,
        /// Bytes the payload actually holds.
        actual: u64,
    },
    /// An initializer or attribute tensor has a negative extent.
    NegativeExtent {
        /// Tensor name.
        tensor: String,
        /// Declared shape, rendered.
        shape: String,
    },
    /// The model defines local functions (`ModelProto.functions`), which this
    /// reader does not expand.
    LocalFunctions {
        /// Number of function definitions found.
        count: usize,
    },
    /// The model carries training info, which is not an inference artifact.
    TrainingInfo,
    /// The graph has sparse initializers, which this build does not read.
    SparseInitializer {
        /// How many were found.
        count: usize,
    },
    /// A tensor or node that must be named has no name.
    MissingName {
        /// What is unnamed, e.g. `"initializer"`.
        what: &'static str,
    },
    /// A node was given the wrong number of operands.
    OperandCount {
        /// What the operator requires.
        expected: usize,
        /// What the node has.
        found: usize,
    },
    /// A node has more than one output in a position this mapping cannot
    /// express.
    UnsupportedOutputCount {
        /// Number of outputs the node declares.
        found: usize,
        /// What the mapping supports.
        expected: usize,
    },
    /// An attribute this mapping does not implement.
    UnsupportedAttribute {
        /// Attribute name.
        attribute: String,
        /// Why it cannot be represented, and what would fix it.
        detail: String,
    },
    /// A node carries an attribute its op does not define in this opset.
    UnknownAttribute {
        /// Attribute name.
        attribute: String,
        /// Attributes the op accepts.
        accepted: String,
    },
    /// An attribute's payload does not match the type the file declares for it.
    AttributeTypeMismatch {
        /// Attribute name.
        attribute: String,
        /// Declared `AttributeType` name.
        declared: &'static str,
        /// Payload actually present.
        found: &'static str,
    },
    /// A node that is not mapped to an fn3 [`OpId`].
    UnmappedOp {
        /// Operator type.
        op_type: String,
        /// ONNX domain (empty for the default domain).
        domain: String,
        /// Why, and which issue owns the work.
        detail: String,
    },
    /// A node that would need an [`OpId`] that exists but is not implemented.
    UnimplementedOpId {
        /// The id that would be needed.
        op: OpId,
        /// The issue that owns the implementation.
        owner: &'static str,
    },
    /// Subgraph or control-flow operators (`If`, `Loop`, `Scan`).
    ControlFlow {
        /// Operator type.
        op_type: String,
    },
    /// A node carries a subgraph attribute (attribute types GRAPH, GRAPHS,
    /// SPARSE_TENSOR or TYPE_PROTO).
    Subgraph {
        /// Attribute name.
        attribute: String,
    },
    /// A node input or output name is used before it is defined.
    UndefinedValue {
        /// Value name.
        value: String,
    },
    /// Two producers define the same value name.
    DuplicateValue {
        /// Value name.
        value: String,
        /// What already defined it.
        existing: String,
    },
    /// The graph is not topologically sorted, so file order is not execution
    /// order.
    NotTopological {
        /// Value name that is used before its producer.
        value: String,
        /// Index of the node that produces it, later in the file.
        producer_index: u32,
    },
    /// Two operands of one node disagree on dtype.
    DTypeMismatch {
        /// What the node requires.
        detail: String,
    },
    /// An index/shape operand is not `i64`, which the 3.0 core requires.
    IndexDType {
        /// The dtype found.
        actual: DType,
        /// What the operand is, e.g. `"shape"`.
        operand: String,
    },
    /// Static shape inference found a contradiction between the operands.
    ShapeMismatch {
        /// What the node requires.
        detail: String,
    },
    /// A graph input or output has a type this build cannot represent (a
    /// sequence, map, optional or sparse tensor).
    UnsupportedType {
        /// Value name.
        value: String,
        /// What kind of type was declared.
        detail: String,
    },
    /// The graph is not a decoder architecture this build supports.
    Architecture {
        /// What was expected and what was found.
        detail: String,
    },
    /// Reading the model file failed.
    Io {
        /// Path that was read.
        path: String,
        /// Operating-system error text.
        detail: String,
    },
    /// More problems were found than [`MAX_PROBLEMS`].
    TooManyProblems {
        /// How many additional problems were dropped.
        omitted: usize,
    },
}

impl fmt::Display for ImportProblemKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImportProblemKind::Proto(error) => write!(f, "{error}"),
            ImportProblemKind::UnsupportedIrVersion { found, supported } => write!(
                f,
                "IR version {found} is not supported; this reader implements ONNX IR {supported}"
            ),
            ImportProblemKind::MissingGraph => f.write_str(
                "the model has no GraphProto; a model without a graph has nothing to import",
            ),
            ImportProblemKind::SparseInitializer { count } => write!(
                f,
                "the graph has {count} sparse initializer(s); only dense initializers are \
                 implemented"
            ),
            ImportProblemKind::MissingName { what } => {
                write!(f, "the {what} has no name; every value must be addressable")
            }
            ImportProblemKind::OperandCount { expected, found } => write!(
                f,
                "the operator takes {expected} operands, the node supplies {found}"
            ),
            ImportProblemKind::MissingOpset => f.write_str(
                "the model declares no opset import for the default ONNX domain; add an \
                 OperatorSetIdProto with domain \"\" (the pinned export uses version 14)",
            ),
            ImportProblemKind::UnsupportedOpset { version, supported } => write!(
                f,
                "ONNX opset {version} is outside the implemented range {supported}"
            ),
            ImportProblemKind::UnsupportedDomain { domain } => write!(
                f,
                "the graph imports operators from domain `{domain}`; only the default ONNX \
                 domain is implemented"
            ),
            ImportProblemKind::UnsupportedDType {
                code,
                onnx_name,
                carrier,
            } => write!(
                f,
                "{carrier} uses ONNX dtype {code} ({onnx_name}), which the 3.0 core cannot \
                 represent; the alpha dtypes are f32 (compute), i64 (indices and shapes), i32 \
                 (auxiliary) and bool (masks)"
            ),
            ImportProblemKind::ExternalTensorData { tensor } => write!(
                f,
                "tensor `{tensor}` stores its payload outside the model file (ONNX external \
                 data); the importer reads a self-contained model only. Convert the model with \
                 `onnx.external_data_helper.convert_model_from_external_data` or save it without \
                 external data"
            ),
            ImportProblemKind::SegmentedTensor { tensor } => write!(
                f,
                "tensor `{tensor}` is stored in segments (`TensorProto.segment`); segmented \
                 tensors are not implemented for the 3.0 alpha"
            ),
            ImportProblemKind::AmbiguousTensorData { tensor, fields } => write!(
                f,
                "tensor `{tensor}` carries more than one payload ({fields}); ONNX allows exactly \
                 one of raw_data or a typed data field"
            ),
            ImportProblemKind::MissingTensorData { tensor } => write!(
                f,
                "tensor `{tensor}` declares a shape and dtype but carries no payload"
            ),
            ImportProblemKind::TensorLength {
                tensor,
                dtype,
                shape,
                expected,
                actual,
            } => write!(
                f,
                "tensor `{tensor}` declares {dtype} {shape}, which needs {expected} payload \
                 bytes, but the payload holds {actual}"
            ),
            ImportProblemKind::NegativeExtent { tensor, shape } => write!(
                f,
                "tensor `{tensor}` declares the negative extent in {shape}; ONNX dimensions are \
                 non-negative"
            ),
            ImportProblemKind::LocalFunctions { count } => write!(
                f,
                "the model defines {count} local function(s) (`ModelProto.functions`); function \
                 bodies must be expanded before import"
            ),
            ImportProblemKind::TrainingInfo => f.write_str(
                "the model carries training info (`ModelProto.training_info`); 3.0 imports \
                 inference models only",
            ),
            ImportProblemKind::UnsupportedOutputCount { found, expected } => write!(
                f,
                "the node declares {found} outputs; this mapping emits {expected}"
            ),
            ImportProblemKind::UnsupportedAttribute { attribute, detail } => {
                write!(f, "attribute `{attribute}` is not representable: {detail}")
            }
            ImportProblemKind::UnknownAttribute {
                attribute,
                accepted,
            } => write!(
                f,
                "attribute `{attribute}` is not defined for this op; accepted attributes: \
                 {accepted}"
            ),
            ImportProblemKind::AttributeTypeMismatch {
                attribute,
                declared,
                found,
            } => write!(
                f,
                "attribute `{attribute}` declares type {declared} but carries a {found} payload"
            ),
            ImportProblemKind::UnmappedOp {
                op_type,
                domain,
                detail,
            } => {
                let domain = if domain.is_empty() {
                    "the default ONNX domain"
                } else {
                    domain.as_str()
                };
                write!(
                    f,
                    "operator `{op_type}` from {domain} has no fn3 op id: {detail}"
                )
            }
            ImportProblemKind::UnimplementedOpId { op, owner } => write!(
                f,
                "this node needs the fn3 op `{op}`, which is reserved but not implemented \
                 (owner: {owner}); it is reported instead of being approximated"
            ),
            ImportProblemKind::ControlFlow { op_type } => write!(
                f,
                "control-flow operator `{op_type}` is not supported; 3.0 imports straight-line \
                 inference graphs only"
            ),
            ImportProblemKind::Subgraph { attribute } => write!(
                f,
                "attribute `{attribute}` carries a nested graph or type payload; subgraphs are \
                 not supported"
            ),
            ImportProblemKind::UndefinedValue { value } => write!(
                f,
                "value `{value}` is read before anything defines it (it is not a graph input, an \
                 initializer or an earlier node output)"
            ),
            ImportProblemKind::DuplicateValue { value, existing } => write!(
                f,
                "value `{value}` is defined twice; it is already provided by {existing}"
            ),
            ImportProblemKind::NotTopological {
                value,
                producer_index,
            } => write!(
                f,
                "value `{value}` is read before node {producer_index} produces it; ONNX requires a \
                 topologically sorted graph"
            ),
            ImportProblemKind::DTypeMismatch { detail } => {
                write!(f, "operand dtypes disagree: {detail}")
            }
            ImportProblemKind::IndexDType { actual, operand } => write!(
                f,
                "the {operand} operand is {actual}, but the 3.0 core requires i64 for indices, \
                 shapes and axis lists (no implicit conversion)"
            ),
            ImportProblemKind::ShapeMismatch { detail } => {
                write!(f, "shape contradiction: {detail}")
            }
            ImportProblemKind::UnsupportedType { value, detail } => write!(
                f,
                "value `{value}` is declared as {detail}; 3.0 imports tensors only"
            ),
            ImportProblemKind::Architecture { detail } => write!(
                f,
                "the graph is not a decoder architecture this build supports: {detail}"
            ),
            ImportProblemKind::Io { path, detail } => {
                write!(f, "cannot read the ONNX model at `{path}`: {detail}")
            }
            ImportProblemKind::TooManyProblems { omitted } => write!(
                f,
                "{omitted} further problem(s) were not reported (the limit is {MAX_PROBLEMS})"
            ),
        }
    }
}

/// One importer finding, optionally attributed to a node.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportProblem {
    /// Node the problem belongs to, when there is one.
    pub node: Option<NodeRef>,
    /// What went wrong.
    pub kind: ImportProblemKind,
}

impl fmt::Display for ImportProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.node {
            Some(node) => write!(f, "{node}: {}", self.kind),
            None => write!(f, "{}", self.kind),
        }
    }
}

/// Every problem found in one model, in file order.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportError {
    problems: Vec<ImportProblem>,
}

impl ImportError {
    /// Build from a problem list. The list must not be empty.
    pub fn new(problems: Vec<ImportProblem>) -> ImportError {
        debug_assert!(!problems.is_empty(), "an import error needs a problem");
        ImportError { problems }
    }

    /// One problem, with no node attribution.
    pub fn single(kind: ImportProblemKind) -> ImportError {
        ImportError {
            problems: vec![ImportProblem { node: None, kind }],
        }
    }

    /// Every problem found, in file order.
    pub fn problems(&self) -> &[ImportProblem] {
        &self.problems
    }

    /// Number of problems.
    pub fn len(&self) -> usize {
        self.problems.len()
    }

    /// `true` if the list is empty. Never true for a value that was returned as
    /// an error; present so clippy's `len_without_is_empty` stays honest.
    pub fn is_empty(&self) -> bool {
        self.problems.is_empty()
    }

    /// `true` if any problem has this discriminant test satisfied.
    pub fn contains(&self, predicate: impl Fn(&ImportProblemKind) -> bool) -> bool {
        self.problems.iter().any(|problem| predicate(&problem.kind))
    }

    /// The first problem that is a byte-level decode failure, if any.
    pub fn proto_error(&self) -> Option<&ProtoError> {
        self.problems
            .iter()
            .find_map(|problem| match &problem.kind {
                ImportProblemKind::Proto(error) => Some(error),
                _ => None,
            })
    }
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.problems.len();
        write!(
            f,
            "ONNX import failed with {count} problem{}:",
            if count == 1 { "" } else { "s" }
        )?;
        for (position, problem) in self.problems.iter().enumerate() {
            write!(f, "\n  {}. {problem}", position + 1)?;
        }
        Ok(())
    }
}

impl std::error::Error for ImportError {}

impl From<ImportError> for Fn3Error {
    fn from(error: ImportError) -> Fn3Error {
        Fn3Error::InvalidFormat {
            format: "onnx model",
            detail: error.to_string(),
        }
    }
}

/// Accumulates problems while the importer keeps working.
#[derive(Debug, Default)]
pub(crate) struct Diagnostics {
    problems: Vec<ImportProblem>,
    omitted: usize,
}

impl Diagnostics {
    /// Record a problem.
    pub(crate) fn push(&mut self, node: Option<NodeRef>, kind: ImportProblemKind) {
        if self.problems.len() >= MAX_PROBLEMS {
            self.omitted += 1;
            return;
        }
        self.problems.push(ImportProblem { node, kind });
    }

    /// Record a problem with no node attribution.
    pub(crate) fn push_global(&mut self, kind: ImportProblemKind) {
        self.push(None, kind);
    }

    /// Consume into an error, or `None` when clean.
    pub(crate) fn finish(mut self) -> Option<ImportError> {
        if self.omitted > 0 {
            self.problems.push(ImportProblem {
                node: None,
                kind: ImportProblemKind::TooManyProblems {
                    omitted: self.omitted,
                },
            });
        }
        if self.problems.is_empty() {
            None
        } else {
            Some(ImportError::new(self.problems))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn problem_messages_name_the_node_and_the_value() {
        let problem = ImportProblem {
            node: Some(NodeRef::new(12, "/transformer/Add_2", "Add")),
            kind: ImportProblemKind::DTypeMismatch {
                detail: "left is f32, right is i64".to_string(),
            },
        };
        let text = problem.to_string();
        assert!(text.contains("node 12"), "{text}");
        assert!(text.contains("/transformer/Add_2"), "{text}");
        assert!(text.contains("(Add)"), "{text}");
        assert!(text.contains("left is f32, right is i64"), "{text}");
    }

    #[test]
    fn errors_render_every_problem_in_order() {
        let error = ImportError::new(vec![
            ImportProblem {
                node: Some(NodeRef::new(1, "a", "Reshape")),
                kind: ImportProblemKind::ShapeMismatch {
                    detail: "first".to_string(),
                },
            },
            ImportProblem {
                node: None,
                kind: ImportProblemKind::MissingOpset,
            },
        ]);
        let text = error.to_string();
        assert!(text.contains("2 problems"), "{text}");
        let first = text.find("first").unwrap();
        let second = text.find("no opset import").unwrap();
        assert!(first < second, "{text}");
    }

    #[test]
    fn the_problem_cap_is_reported_not_hidden() {
        let mut diagnostics = Diagnostics::default();
        for index in 0..(MAX_PROBLEMS + 3) {
            diagnostics.push_global(ImportProblemKind::MissingOpset);
            let _ = index;
        }
        let error = diagnostics.finish().unwrap();
        assert_eq!(error.len(), MAX_PROBLEMS + 1);
        assert!(error.to_string().contains("3 further problem(s)"));
    }

    #[test]
    fn converts_into_the_core_error_with_the_full_text() {
        let error = ImportError::single(ImportProblemKind::MissingOpset);
        let core: Fn3Error = error.clone().into();
        match core {
            Fn3Error::InvalidFormat { format, detail } => {
                assert_eq!(format, "onnx model");
                assert!(detail.contains("no opset import"), "{detail}");
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}
