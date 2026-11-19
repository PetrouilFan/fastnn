//! Strict ONNX model import for the 3.0 inference core.
//!
//! # What this module is for
//!
//! One thing: turn the bytes of a self-contained ONNX model into the fn3 IR that
//! a planner ([`crate::fn3::import::ModelImport`]) and a serializer can consume,
//! or fail with a list of problems that names every node it could not represent.
//!
//! ```text
//! bytes ──► proto   (bounds-checked protobuf wire reader, no dependency)
//!       ──► graph   (op mapping, attributes, static dtype/shape inference)
//!       ──► weights (name + dtype + shape + exact payload bytes)
//!       ──► ModelImport { graph, weights, geometry, summary }
//! ```
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
//! ```no_run
//! # fn main() -> Result<(), fastnn::fn3::import::ImportError> {
//! let bytes = std::fs::read("model.onnx").map_err(|error| {
//!     fastnn::fn3::import::ImportError::single(
//!         fastnn::fn3::import::ImportProblemKind::Io {
//!             path: "model.onnx".to_string(),
//!             detail: error.to_string(),
//!         },
//!     )
//! })?;
//! let imported = fastnn::fn3::import::import_onnx_model(&bytes)?;
//! let geometry = imported.geometry();
//! assert!(geometry.layers > 0);
//! let (graph, weights) = imported.into_parts();
//! # let _ = (graph, weights);
//! # Ok(())
//! # }
//! ```
//!
//! # Seam for the next issues
//!
//! | Consumer | What to call |
//! |---|---|
//! | #55 prepared-model format | [`ModelImport::weights`] ([`WeightSet`]: name, [`DType`], [`Shape`], bytes in file order) and [`ModelImport::ir_version`]/[`ModelImport::opset`] |
//! | #56 planner | [`ModelImport::graph`] ([`GraphIr`]: values, then nodes with [`OpId`], operand [`ValueId`]s, [`crate::fn3::backend::OpParams`]) and [`ModelImport::geometry`] |
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
//! | Opset (default domain) | 13..=17, the input form of `Squeeze`/`Unsqueeze`/`Split`/`Slice` and the attribute form of `ReduceMean.axes` |
//! | Domains | the default ONNX domain only |
//! | Dtypes | `f32` compute, `i64` indices/shapes, `i32` auxiliary, `bool` masks |
//! | Architecture | a GPT-2-shaped decoder: token + learned position table, per-layer `c_attn`/`c_proj`/`c_fc`/`mlp c_proj`, `past_key_values.{i}` inputs, `present.{i}` outputs, `logits` |
//!
//! # Dtypes: reject, preserve, never cast
//!
//! "Reject non-F32 compute paths" is implemented as three separate rules:
//!
//! 1. an element type outside [`crate::fn3::dtype::DType`] (`FLOAT16`, `DOUBLE`,
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

pub mod error;
mod graph;
mod proto;

/// The weight table is typed weight data, not import logic, so it lives in
/// `model/` (ADR §5 reserves `src/fn3/model/weights.rs` for this issue) and is
/// re-exported here for import callers.
pub use crate::fn3::model::weights::{WeightId, WeightSet, WeightSource, WeightTensor};
pub use error::{ImportError, ImportProblem, ImportProblemKind, NodeRef, ProtoError, MAX_PROBLEMS};
pub use graph::{
    DecoderGeometry, Dim, GraphIr, GraphNode, ImportSummary, ImportWarning, ImportWarningKind,
    KvInterface, ModelImport, ModelProvenance, OpsetImport, PositionEmbedding, ValueId, ValueInfo,
    ValueOrigin, ValueShape, MAX_IR_VERSION, MAX_OPSET, MIN_IR_VERSION, MIN_OPSET,
};

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
