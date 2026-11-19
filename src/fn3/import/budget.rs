//! Parser budgets for the untrusted ONNX input.
//!
//! # Why this module exists
//!
//! A protobuf stream carries its own counts, and an encoded element is smaller
//! than the Rust value it becomes: one packed varint is one byte and one `i64`,
//! one `NodeProto` with a three-character `op_type` is five bytes and a
//! hundred-byte [`super::GraphNode`]. Without a budget, a file's *size* is not a
//! bound on what decoding it costs, and "the input is only 10 MB" is not an
//! answer to "why is the process using four gigabytes".
//!
//! `src/fn3/model/format/` made this choice first for the artifact reader and
//! named its caps `MAX_VALUES`, `MAX_NODES`, `MAX_WEIGHTS`, `MAX_RANK` and
//! `MAX_PARAM_AXES`. The importer uses the **same numbers** on purpose: a model
//! whose value, node or weight table is beyond what the format can write would
//! import successfully and then fail in `PreparedArtifact::from_import` with no
//! hint that the importer was the place to notice. The equality is pinned by
//! `import::budget`'s own tests, because `import` is the lower layer and may not
//! import the format.
//!
//! # What is bounded, and where
//!
//! | Budget | Limit | Enforced in | Protects |
//! |---|---|---|---|
//! | [`MAX_MODEL_BYTES`] | 8 GiB | [`super::graph::build_model`] | the one `Vec` the format also caps |
//! | [`MAX_NODES`] | 2^20 | `proto::decode_graph`, `Builder::register_nodes` | `Vec<NodeProto>`, `Vec<GraphNode>` |
//! | [`MAX_VALUES`] | 2^20 | `proto::decode_graph`, `Builder::register_nodes` | `Vec<ValueInfo>` |
//! | [`MAX_WEIGHTS`] | 2^20 | `proto::decode_graph`, `Builder::register_nodes` | `Vec<WeightTensor>` |
//! | [`MAX_RANK`] | 64 | `proto::decode_tensor`, `proto::decode_shape`, [`super::graph::tensor_shape`] | `TensorProto.dims`, `Shape` |
//! | [`MAX_PARAM_AXES`] | 1024 | `proto::decode_attribute` | `Vec<f32>`/`Vec<i64>` attribute lists |
//! | [`MAX_ATTRIBUTES`] | 1024 | `proto::decode_node` | `Vec<Attribute>` |
//! | [`MAX_OPERANDS`] | 1024 | `proto::decode_node` | `Vec<&str>` inputs and outputs |
//! | [`MAX_STRING_BYTES`] | 4096 | `proto::Reader::text` | one `String` per name |
//! | [`MAX_OPSET_IMPORTS`] | 16 | `proto::decode_model` | `Vec<OpsetId>` |
//! | [`MAX_TRACKED_ELEMENTS`] | 4096 | [`super::graph`] | constant-folded element sequences |
//!
//! # What is *not* bounded here, and why that is honest
//!
//! [`MAX_MODEL_BYTES`] cannot prevent the caller from holding the input: the
//! importer is handed `&[u8]` and the file is already in memory. It states the
//! largest model this build claims to support, so a reader knows that a failure
//! beyond it is a documented limit rather than a mysterious OOM.
//!
//! A graph node costs at least two bytes on the wire (a tag and a zero length),
//! so a node budget cannot be derived from the bytes that remain — the encoded
//! form is always smaller than the decoded one. [`MAX_NODES`] is therefore an
//! absolute policy cap, and the documented worst case is
//! `MAX_NODES × size_of::<NodeProto>()`; `tests/fn3_malformed.rs` measures it
//! rather than asserting it from the struct layout.
//!
//! # What is bounded elsewhere
//!
//! Element counts and byte counts inside one tensor are *not* capped by a
//! constant here: they must equal the declared shape, and the shape's product is
//! checked against `u64` overflow by `Shape::numel`. A tensor whose declared
//! shape does not match its payload is rejected before the payload is converted
//! (see [`super::graph::tensor_bytes`]), so a typed payload cannot be expanded
//! into a larger buffer than the file itself.

/// Largest ONNX model file this build claims to import, in bytes.
///
/// Equal to the artifact's `MAX_PAYLOAD_BYTES`: an imported model is written to
/// an artifact, and a model beyond the format's payload cap could never be
/// saved. Eight GiB is far above the 498 MB trained GPT-2 and far below "the
/// process dies before it can report a problem".
pub const MAX_MODEL_BYTES: u64 = 8 * (1 << 30);

/// Largest number of `GraphProto.node` entries this build imports.
pub const MAX_NODES: usize = 1 << 20;

/// Largest value table a model may produce, counting graph inputs,
/// initializers and node outputs.
pub const MAX_VALUES: usize = 1 << 20;

/// Largest weight table a model may produce.
pub const MAX_WEIGHTS: usize = 1 << 20;

/// Largest tensor rank accepted in a shape, a declared value or a weight.
pub const MAX_RANK: usize = 64;

/// Largest repeated numeric attribute (`ints`, `floats`) this build decodes.
///
/// These carry axes, permutation and split sizes, and the artifact stores them
/// in `OpParams.axes`, which has its own `MAX_PARAM_AXES` cap.
pub const MAX_PARAM_AXES: usize = 1024;

/// Largest number of attributes one node may carry.
pub const MAX_ATTRIBUTES: usize = 1024;

/// Largest number of inputs or outputs one node may declare.
pub const MAX_OPERANDS: usize = 1024;

/// Largest name, domain or `dim_param` this build decodes, in bytes.
///
/// Equal to the artifact's string cap. Names are borrowed from the input and
/// copied once into the value table; this bounds that copy.
pub const MAX_STRING_BYTES: usize = 4096;

/// Largest number of `OperatorSetIdProto` entries a model may declare.
pub const MAX_OPSET_IMPORTS: usize = 16;

/// Largest element sequence the importer folds into `AtomSeries`.
///
/// A declared extent of a billion must not become a billion tracked atoms, so a
/// larger sequence keeps its length and drops the per-element detail.
pub const MAX_TRACKED_ELEMENTS: usize = 4096;

/// Largest element count accepted in one *typed* `TensorProto` payload
/// (`float_data`, `int32_data`, `int64_data`, `uint64_data`, `double_data`).
///
/// A typed payload stores one element per field occurrence, so a one-byte varint
/// becomes an eight-byte `i64`: decoding is where the amplification would happen
/// if the count were unbounded. The value is the artifact's `MAX_VOCAB_SIZE`,
/// which is the largest single constant a decoder of this shape can legitimately
/// carry. Bulk weights use `raw_data`, which is a direct slice of the input and
/// needs no cap of its own.
pub const MAX_TYPED_ELEMENTS: usize = 1 << 24;

/// `true` when a model of `len` bytes is inside [`MAX_MODEL_BYTES`].
///
/// Split out from [`super::graph::build_model`] so the comparison is testable
/// without allocating an eight-gigabyte buffer.
pub fn model_bytes_within_cap(len: u64) -> bool {
    len <= MAX_MODEL_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_model_cap_is_the_artifact_payload_cap() {
        // The importer must not accept a model the format cannot write. The
        // comparison lives here rather than in a dependency because `import` is
        // the lower layer; `model::format` asserts the other direction.
        assert_eq!(
            MAX_MODEL_BYTES,
            crate::fn3::model::format::MAX_PAYLOAD_BYTES
        );
    }

    #[test]
    fn the_size_check_is_inclusive_and_reports_the_cap() {
        assert!(model_bytes_within_cap(0));
        assert!(model_bytes_within_cap(MAX_MODEL_BYTES));
        assert!(!model_bytes_within_cap(MAX_MODEL_BYTES + 1));
    }

    #[test]
    fn the_importer_budgets_equal_the_artifact_budgets() {
        use crate::fn3::model::format;
        assert_eq!(MAX_NODES as u64, format::MAX_NODES);
        assert_eq!(MAX_VALUES as u64, format::MAX_VALUES);
        assert_eq!(MAX_WEIGHTS as u64, format::MAX_WEIGHTS);
        assert_eq!(MAX_RANK as u64, format::MAX_RANK);
        assert_eq!(MAX_PARAM_AXES as u64, format::MAX_PARAM_AXES);
    }
}
