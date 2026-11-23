//! Strict ONNX model import for the 3.0 inference core.
//!
//! # What this module is for
//!
//! One thing: turn the bytes of a self-contained ONNX model into the fn3 IR that
//! a planner ([`crate::inference::import::ModelImport`]) and a serializer can consume,
//! or fail with a list of problems that names every node it could not represent.
//!
//! ```text
//! bytes ──► proto   (bounds-checked protobuf wire reader, no dependency)
//!       ──► graph   (op mapping, attributes, static dtype/shape inference)
//!       ──► weights (name + dtype + shape + exact payload bytes)
//!       ──► ModelImport   { graph, weights, decoder geometry, summary }
//!       ──► ImportedGraph { typed ir::Graph, weights, summary }
//! ```
//!
//! The two results share one decode and differ only after it:
//! [`import_onnx_model`] adds the GPT-2-shaped decoder profile, while
//! [`import_onnx_graph`] converts the decoded graph into
//! [`crate::inference::ir`] and runs its shape pass, so a CNN imports without
//! being required to look like a decoder.
//!
//! # What it deliberately does not do
//!
//! - It does not allocate device tensors. [`ModelImport::to_model_weights`] is
//!   the explicit, CPU-only conversion; the versioned format (FN3-09 / #55) and
//!   the planner (FN3-10 / #56) decide what happens next.
//! - It does not plan. Values are named, typed and shaped; slot allocation,
//!   specialisation and dynamic-bound enforcement belong to #56.
//! - It does not approximate. A node that cannot be represented exactly is an
//!   error that names the node, the ONNX operator and the owning issue. The 2.x
//!   importer's "return the first input" fallback is forbidden here.
//! - It does not require Python, `protoc` or any crate outside `core`/`alloc`.
//!
//! # Entry points
//!
//! The decoder profile:
//!
//! ```no_run
//! # fn main() -> Result<(), fastnn::inference::import::ImportError> {
//! let bytes = std::fs::read("model.onnx").map_err(|error| {
//!     fastnn::inference::import::ImportError::single(
//!         fastnn::inference::import::ImportProblemKind::Io {
//!             path: "model.onnx".to_string(),
//!             detail: error.to_string(),
//!         },
//!     )
//! })?;
//! let imported = fastnn::inference::import::import_onnx_model(&bytes)?;
//! let geometry = imported.geometry();
//! assert!(geometry.layers > 0);
//! let (graph, weights) = imported.into_parts();
//! # let _ = (graph, weights);
//! # Ok(())
//! # }
//! ```
//!
//! The typed graph (no decoder profile):
//!
//! ```no_run
//! # fn main() -> Result<(), fastnn::inference::import::ImportError> {
//! let bytes = std::fs::read("yolov8n.onnx").unwrap();
//! let imported = fastnn::inference::import::import_onnx_graph(&bytes)?;
//! let output = imported.lookup("output0").expect("the model names its output");
//! let ty = &imported.graph().value(output).unwrap().ty;
//! assert!(ty.is_static());
//! # let _ = ty;
//! # Ok(())
//! # }
//! ```
//!
//! # Seam for the next issues
//!
//! | Consumer | What to call |
//! |---|---|
//! | #55 prepared-model format | [`ModelImport::weights`] ([`WeightSet`]: name, [`DType`](crate::inference::DType), [`Shape`](crate::inference::Shape), bytes in file order) and [`ModelImport::ir_version`]/[`ModelImport::opset`] |
//! | #56 planner | [`ModelImport::graph`] ([`GraphIr`]: values, then nodes with [`OpId`](crate::inference::OpId), operand [`ValueId`]s, [`crate::inference::backend::OpParams`]) and [`ModelImport::geometry`] |
//! | #58 session | [`DecoderGeometry::kv`] for the cache interface and [`DecoderGeometry::kv_cache_config`] for the allocation |
//! | #62 end-to-end | [`ModelImport::to_model_weights`] plus [`ModelImport::summary`] for an inventory assertion |
//!
//! # Determinism
//!
//! The same bytes always produce the same value table, node order and weight
//! layout: ordering is file order throughout, and no map iteration reaches the
//! output.
//!
//! # Supported input
//!
//! | Dimension | Accepted |
//! |---|---|
//! | Container | one self-contained model file; external data and sparse initializers are rejected by name |
//! | IR version | 4..=10 |
//! | Opset (default domain) | 12..=17; the opset-12 `Split` attribute form and the opset-13 input form of `Squeeze`/`Unsqueeze`/`Split`/`Slice`; the attribute form of `ReduceMean.axes` |
//! | Domains | the default ONNX domain only |
//! | Dtypes | `f32` compute, `i64` indices/shapes, `i32` auxiliary, `bool` masks |
//! | Architecture | [`import_onnx_model`] requires a GPT-2-shaped decoder (token + learned position table, per-layer projections, `past_key_values.{i}` inputs, `present.{i}` outputs, `logits`); [`import_onnx_graph`] has no architecture profile |
//!
//! `Constant` nodes are folded into the weight table rather than executed: an
//! ONNX `Constant` is data the file already fixed, so it becomes an
//! initializer-like value and never reaches a plan.
//!
//! # Dtypes: reject, preserve, never cast
//!
//! "Reject non-F32 compute paths" is implemented as three separate rules:
//!
//! 1. an element type outside [`crate::inference::dtype::DType`] (`FLOAT16`, `DOUBLE`,
//!    `BFLOAT16`, quantized types, strings) is
//!    [`ImportProblemKind::UnsupportedDType`] naming the code, the ONNX mnemonic
//!    and the carrier;
//! 2. two operands of one node that disagree on dtype are
//!    [`ImportProblemKind::DTypeMismatch`];
//! 3. an index/shape operand that is not `i64` is
//!    [`ImportProblemKind::IndexDType`].
//!
//! `i64` arithmetic is *not* rejected: the pinned exporter computes `total`,
//! `past` and the causal index range with `i64` `Add`/`Sub`, and casting those
//! to `f32` would be a silent numerical change. The mapping keeps `i64` intact
//! and `tests/fn3_import.rs` pins the exact four nodes that need it.

pub mod budget;
pub mod error;
mod graph;
mod proto;
mod typed;

/// The weight table is typed weight data, not import logic, so it lives in
/// `model/` (ADR §5 reserves `src/fn3/model/weights.rs` for this issue) and is
/// re-exported here for import callers.
pub use crate::inference::model::weights::{WeightId, WeightSet, WeightSource, WeightTensor};
pub use budget::{
    MAX_ATTRIBUTES, MAX_MODEL_BYTES, MAX_NODES, MAX_OPERANDS, MAX_OPSET_IMPORTS, MAX_PARAM_AXES,
    MAX_RANK, MAX_STRING_BYTES, MAX_TRACKED_ELEMENTS, MAX_TYPED_ELEMENTS, MAX_VALUES, MAX_WEIGHTS,
};
pub use error::{ImportError, ImportProblem, ImportProblemKind, NodeRef, ProtoError, MAX_PROBLEMS};
pub use graph::{
    DecoderGeometry, Dim, GraphIr, GraphNode, ImportSummary, ImportWarning, ImportWarningKind,
    KvCacheConfig, KvCacheLayout, KvInterface, ModelImport, ModelProvenance, OpsetImport,
    PositionEmbedding, ValueId, ValueInfo, ValueOrigin, ValueShape, MAX_IR_VERSION, MAX_OPSET,
    MIN_IR_VERSION, MIN_OPSET,
};
pub use typed::ImportedGraph;

/// Import an ONNX model into the typed v3 graph.
///
/// Unlike [`import_onnx_model`], this stops before the decoder architecture
/// profile: it validates every node, resolves every value's dtype and shape, and
/// runs [`crate::inference::ir::infer_shapes`], so a CNN (or any graph the
/// operator table covers) imports without pretending to be a GPT-2 decoder.
/// The typed IR has no unknown extent, so a value the importer could not prove
/// is a node-named error here rather than an unchecked shape.
///
/// [`ImportedGraph`] carries the typed graph, the weights and the import
/// summary. The decoder path remains [`import_onnx_model`], which additionally
/// requires the `past_key_values`/`present` interface.
pub fn import_onnx_graph(bytes: &[u8]) -> Result<ImportedGraph, ImportError> {
    import_onnx_graph_with_shapes(bytes, &[])
}

/// Import an ONNX model, binding concrete extents for its graph inputs.
///
/// A dynamic ONNX export declares `[batch, 3, height, width]` as `dim_param`
/// names, so the typed IR has nothing to prove a `Reshape` target against. A
/// binding replaces one graph input's declared extents with the shape the model
/// will actually run at; every downstream extent then follows from the model's
/// own rules. A name the model does not declare, a rank that does not match and
/// the same name twice are all typed import problems.
pub fn import_onnx_graph_with_shapes(
    bytes: &[u8],
    shapes: &[(String, Vec<u64>)],
) -> Result<ImportedGraph, ImportError> {
    let mut bindings: std::collections::BTreeMap<String, Vec<u64>> =
        std::collections::BTreeMap::new();
    for (name, extents) in shapes {
        if bindings.insert(name.clone(), extents.clone()).is_some() {
            return Err(ImportError::single(ImportProblemKind::ShapeBinding {
                input: name.clone(),
                detail: "the input was bound more than once".to_string(),
            }));
        }
    }
    typed::build(graph::build_graph_parts(bytes, &bindings)?)
}

/// One graph input as the model declares it, before any binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredInput {
    /// Value name.
    pub name: String,
    /// Declared shape, rendered (`[batch, 3, height, width]`).
    pub shape: String,
}

/// The graph inputs and their declared shapes.
///
/// Read without shape inference, so a model whose symbolic extents block the
/// typed pass still reports what a caller would have to bind.
pub fn declared_inputs(bytes: &[u8]) -> Result<Vec<DeclaredInput>, ImportError> {
    Ok(graph::declared_inputs_of(bytes)?
        .into_iter()
        .map(|(name, shape)| DeclaredInput { name, shape })
        .collect())
}

/// Import an ONNX model from bytes.
///
/// Fails with every problem found, in file order. A byte-level decode failure is
/// reported alone, because nothing after it can be trusted.
pub fn import_onnx_model(bytes: &[u8]) -> Result<ModelImport, ImportError> {
    graph::build_model(bytes)
}

/// Import an ONNX model from a file.
///
/// Convenience wrapper: the file is read once and dropped, because
/// [`ModelImport`] owns everything it needs.
pub fn import_onnx_file(path: impl AsRef<std::path::Path>) -> Result<ModelImport, ImportError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|error| {
        ImportError::single(ImportProblemKind::Io {
            path: path.display().to_string(),
            detail: error.to_string(),
        })
    })?;
    import_onnx_model(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_a_typed_proto_error_not_a_panic() {
        let error = import_onnx_model(&[]).unwrap_err();
        assert_eq!(error.len(), 1);
        assert!(matches!(
            error.proto_error(),
            Some(ProtoError::MissingField { .. })
        ));
    }

    #[test]
    fn a_missing_file_names_the_path() {
        let error = import_onnx_file("/nonexistent/model.onnx").unwrap_err();
        match &error.problems()[0].kind {
            ImportProblemKind::Io { path, detail } => {
                assert!(path.contains("model.onnx"), "{path}");
                assert!(!detail.is_empty());
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn the_public_surface_names_its_seams() {
        // Compile-time documentation of the seam: these are the types #55 and
        // #56 consume.
        fn _takes_graph(_: &GraphIr) {}
        fn _takes_weights(_: &WeightSet) {}
        fn _takes_geometry(_: &DecoderGeometry) {}
    }
}
