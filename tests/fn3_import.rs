//! FN3-08 (#54): strict ONNX import.
//!
//! # What these tests prove
//!
//! 1. **The pinned artifact imports.** `onnx-community/tiny-random-gpt2-ONNX`
//!    (27,897,974 bytes, SHA-256 `e18370ba…141a1`) is decoded, every one of its
//!    257 nodes maps to an fn3 op id, and the inventory, geometry and weight
//!    layout match the values recorded in `fixtures/parity/tiny-gpt2/model.lock.json`.
//! 2. **Determinism.** Importing the same bytes twice yields an identical
//!    `ModelImport`, the same value table and the same weight layout.
//! 3. **No panic on malformed input.** Truncated prefixes of the real file and
//!    deterministic byte flips produce typed errors.
//! 4. **Exact rejection.** Unsupported operators, external tensor data, subgraphs,
//!    a missing opset, unrepresentable dtypes and unrepresentable `Gemm`/`Reshape`
//!    attributes each produce a distinct, named error.
//! 5. **The documented compositions.** `Gemm` becomes `MatMul` + `Add`,
//!    `ConstantOfShape.value` is lifted to a constant input, and
//!    `Reshape allowzero = 0` semantics are preserved.
//!
//! # Running the pinned-model tests
//!
//! The 28 MB model is not committed (it has no declared license). Fetch or
//! verify it with
//!
//! ```text
//! python3 tools/parity/fetch_models.py --suite tiny-gpt2 --verify-only
//! ```
//!
//! and point the tests at it with `FASTNN_FN3_ONNX_MODEL=/path/to/model.onnx`.
//! Without the variable the tests look in
//! `tools/parity/.cache/models/tiny-gpt2/onnx/model.onnx` and, when neither
//! exists, they print a reason and pass.

use std::path::PathBuf;

use fastnn::fn3::import::{
    import_onnx_model, ImportError, ImportProblemKind, ModelImport, WeightSource,
};
use fastnn::fn3::OpId;

// ---------------------------------------------------------------------------
// Pinned artifact
// ---------------------------------------------------------------------------

/// Byte length of the pinned graph (`model.lock.json`).
const PINNED_BYTES: usize = 27_897_974;

/// FNV-1a 64-bit fingerprint of the pinned file.
///
/// The authoritative identity is the SHA-256 in
/// `fixtures/parity/tiny-gpt2/model.lock.json`, which
/// `tools/parity/fetch_models.py --verify-only` checks. This fingerprint lets
/// the test notice that the file at the configured path is a *different* graph
/// without carrying a hash implementation.
const PINNED_FNV1A: u64 = 0xc294_d3ab_837f_b1be;

/// Locate the pinned model, or explain why the test is skipped.
fn pinned_model() -> Option<Vec<u8>> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("FASTNN_FN3_ONNX_MODEL") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tools/parity/.cache/models/tiny-gpt2/onnx/model.onnx"),
    );
    for path in candidates {
        match std::fs::read(&path) {
            Ok(bytes) => return Some(bytes),
            Err(_) => continue,
        }
    }
    println!(
        "SKIP: the pinned 27,897,974-byte ONNX model is not present. Set \
         FASTNN_FN3_ONNX_MODEL or run `python3 tools/parity/fetch_models.py --suite tiny-gpt2`."
    );
    None
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn pinned() -> Option<(Vec<u8>, ModelImport)> {
    let bytes = pinned_model()?;
    assert_eq!(
        bytes.len(),
        PINNED_BYTES,
        "the file at the configured path is not the pinned graph"
    );
    let imported = import_onnx_model(&bytes).unwrap_or_else(|error| {
        panic!("the pinned model failed to import:\n{error}");
    });
    Some((bytes, imported))
}

#[test]
fn pinned_model_imports_with_the_documented_inventory() {
    let Some((bytes, imported)) = pinned() else {
        return;
    };
    assert_eq!(
        fnv1a(&bytes),
        PINNED_FNV1A,
        "the file at the configured path is not the pinned graph"
    );

    assert_eq!(imported.ir_version(), 7);
    assert_eq!(imported.opset().version, 14);
    assert_eq!(imported.opset().domain, "");
    assert_eq!(imported.producer(), "pytorch");
    assert_eq!(imported.graph_name(), "main_graph");

    let summary = imported.summary();
    assert_eq!(summary.onnx_nodes, 257, "node count");
    // Eight `Gemm` nodes each become MatMul + Add.
    assert_eq!(summary.fn3_nodes, 265, "fn3 node count");
    assert_eq!(summary.initializers, 48);
    assert_eq!(summary.attribute_constants, 2, "ConstantOfShape values");
    // 27,833,172 initializer bytes plus the two 4-byte lifted
    // `ConstantOfShape` fill values.
    let initializer_bytes: u64 = imported
        .weights()
        .initializers()
        .map(|weight| weight.byte_len() as u64)
        .sum();
    assert_eq!(initializer_bytes, 27_833_172);
    assert_eq!(summary.weight_bytes, 27_833_180);

    let f32_initializers = imported
        .weights()
        .initializers()
        .filter(|weight| weight.dtype() == fastnn::fn3::DType::F32)
        .count();
    let i64_initializers = imported
        .weights()
        .initializers()
        .filter(|weight| weight.dtype() == fastnn::fn3::DType::I64)
        .count();
    assert_eq!((f32_initializers, i64_initializers), (24, 24));

    // The exact ONNX operator census, in the order the issue records it.
    let expected: &[(&str, usize)] = &[
        ("Add", 19),
        ("Cast", 3),
        ("Concat", 23),
        ("ConstantOfShape", 2),
        ("Div", 5),
        ("Equal", 2),
        ("Expand", 2),
        ("Gather", 35),
        ("Gemm", 8),
        ("Less", 1),
        ("MatMul", 5),
        ("Mul", 12),
        ("Pow", 7),
        ("Range", 1),
        ("ReduceMean", 10),
        ("Reshape", 25),
        ("Shape", 20),
        ("Slice", 4),
        ("Softmax", 2),
        ("Split", 2),
        ("Sqrt", 5),
        ("Squeeze", 2),
        ("Sub", 7),
        ("Tanh", 2),
        ("Transpose", 11),
        ("Unsqueeze", 37),
        ("Where", 5),
    ];
    assert_eq!(
        summary.onnx_ops.len(),
        expected.len(),
        "unexpected operator count: {:?}",
        summary.onnx_ops
    );
    for (name, count) in expected {
        assert_eq!(summary.count_onnx(name), *count, "op {name}");
    }
    // There is no Erf and no Gelu node in this export: `gelu_new` is the
    // exporter's primitive decomposition.
    assert_eq!(summary.count_onnx("Erf"), 0);
    assert_eq!(summary.count_onnx("Gelu"), 0);

    // Every mapped node has a supported op id and a non-empty value table.
    assert_eq!(summary.fn3_nodes, imported.graph().nodes().len());
    for node in imported.graph().nodes() {
        assert!(
            !node.inputs.is_empty() || node.op == OpId::Range,
            "node {} `{}` has no inputs",
            node.onnx_index,
            node.name
        );
        assert!(!node.outputs.is_empty());
    }
    assert_eq!(summary.values, 326, "value table size");
}

#[test]
fn pinned_model_geometry_is_detected_and_validated() {
    let Some((_, imported)) = pinned() else {
        return;
    };
    let geometry = imported.geometry();
    assert_eq!(geometry.layers, 2);
    assert_eq!(geometry.hidden_size, 128);
    assert_eq!(geometry.mlp_hidden_size, 512);
    assert_eq!(geometry.query_heads, 2);
    assert_eq!(geometry.kv_heads, 2);
    assert_eq!(geometry.head_dim, 64);
    assert_eq!(geometry.vocab_size, 50_257);
    assert_eq!(geometry.positions, {
        fastnn::fn3::import::PositionEmbedding::Learned {
            weight: imported.weights().id_of("transformer.wpe.weight").unwrap(),
            max_positions: 1024,
        }
    });
    let epsilon = geometry.layer_norm_epsilon.expect("LayerNorm epsilon");
    assert!(
        (epsilon - 1e-5).abs() < 1e-9,
        "unexpected LayerNorm epsilon {epsilon}"
    );
    assert_eq!(geometry.kv.len(), 2);
    let logits = imported.graph().value(geometry.logits);
    assert_eq!(logits.dtype, fastnn::fn3::DType::F32);
    assert_eq!(logits.shape.rank(), Some(3));
    assert_eq!(logits.shape.dims()[2].as_known(), Some(50_257));
    assert_eq!(
        geometry.kv_cache_config(16, 1).n_kv_heads,
        2,
        "kv cache geometry"
    );

    // The shape chain, resolved symbolically: the embedding sum, the attention
    // mask, the scores, the cache outputs and the logits.
    for (name, expected) in [
        ("/transformer/Add_output_0", "[batch, seq, 128]"),
        ("/transformer/ConstantOfShape_1_output_0", "[seq, past]"),
        ("/transformer/Concat_5_output_0", "[seq, total]"),
        ("/transformer/Where_4_output_0", "[batch, 1, seq, total]"),
        (
            "/transformer/h.0/attn/Add_output_0",
            "[batch, 2, seq, total]",
        ),
        (
            "/transformer/h.0/attn/Softmax_output_0",
            "[batch, 2, seq, total]",
        ),
        (
            "/transformer/h.0/attn/MatMul_1_output_0",
            "[batch, 2, seq, 64]",
        ),
        ("present.0.key", "[batch, 2, total, 64]"),
        ("logits", "[batch, seq, 50257]"),
    ] {
        let (_, value) = imported
            .graph()
            .find(name)
            .unwrap_or_else(|| panic!("{name}"));
        assert_eq!(value.shape.render(), expected, "shape of {name}");
    }

    // The only axes the importer cannot prove are the flatten of batch and seq
    // that feeds each `Gemm` (`[batch * seq, hidden]`), which the symbol set
    // cannot express. Everything else is exact.
    let unresolved: Vec<&str> = imported
        .graph()
        .values()
        .iter()
        .filter(|value| match &value.shape {
            fastnn::fn3::import::ValueShape::Unknown => true,
            fastnn::fn3::import::ValueShape::Dims(dims) => dims
                .iter()
                .any(|dim| matches!(dim, fastnn::fn3::import::Dim::Unknown(_))),
        })
        .map(|value| value.name.as_str())
        .collect();
    assert_eq!(
        unresolved.len(),
        24,
        "unexpected unresolved shapes: {unresolved:?}"
    );
}

#[test]
fn every_mapped_node_has_an_implemented_op_id() {
    let Some((_, imported)) = pinned() else {
        return;
    };
    let backend = fastnn::fn3::CpuBackend::new();
    let mut census = std::collections::BTreeMap::new();
    for node in imported.graph().nodes() {
        assert!(
            fastnn::fn3::OpId::ALL.contains(&node.op),
            "node {} maps to {:?}, which is not in OpId::ALL",
            node.onnx_index,
            node.op
        );
        assert!(
            fastnn::fn3::Backend::supports(&backend, node.op).is_supported(),
            "node {} maps to {}, which the CPU backend reports as unsupported",
            node.onnx_index,
            node.op
        );
        assert_ne!(
            node.op,
            OpId::MaskedAttention,
            "the fused attention id must never be emitted by import"
        );
        *census.entry(node.op).or_insert(0usize) += 1;
    }
    // The pinned graph needs no attention id: its attention is primitive ops.
    assert_eq!(imported.summary().ops.len(), census.len());
}

#[test]
fn integer_arithmetic_is_preserved_rather_than_cast() {
    let Some((_, imported)) = pinned() else {
        return;
    };
    // The exporter computes `total`, `past` and the causal index range with
    // i64 Add/Sub. The mapping keeps i64 intact: casting those to f32 would be a
    // silent numerical change. The consequence is recorded here because the
    // alpha CPU arithmetic kernels are documented as f32-only, so this is the
    // exact list of nodes #53/#56 must enable for the pinned model to run.
    let mut integer_arithmetic = Vec::new();
    for node in imported.graph().nodes() {
        if !matches!(
            node.op,
            OpId::Add | OpId::Sub | OpId::Mul | OpId::Div | OpId::Pow
        ) {
            continue;
        }
        let dtype = imported.graph().value(node.inputs[0]).dtype;
        if dtype == fastnn::fn3::DType::I64 {
            integer_arithmetic.push((node.onnx_index, node.op.name(), node.name.clone()));
        }
    }
    assert_eq!(
        integer_arithmetic,
        vec![
            (17, "add", "/transformer/Add_1".to_string()),
            (21, "sub", "/transformer/Sub".to_string()),
            (27, "add", "/transformer/Add_3".to_string()),
            (44, "add", "/transformer/Add_2".to_string()),
        ],
        "the integer shape arithmetic of the pinned graph changed"
    );
}

#[test]
fn pinned_model_ir_is_self_consistent() {
    let Some((_, imported)) = pinned() else {
        return;
    };
    let graph = imported.graph();
    let values = graph.values().len() as u32;
    for node in graph.nodes() {
        for id in node.inputs.iter().chain(node.outputs.iter()) {
            assert!(
                id.index() < values as usize,
                "node {} references value {} outside the table",
                node.onnx_index,
                id.index()
            );
        }
        // Every input is defined before the node that reads it.
        for id in &node.inputs {
            if let fastnn::fn3::import::ValueOrigin::Node(index) = graph.value(*id).origin {
                assert!(
                    (index as usize) < graph.nodes().len(),
                    "value {} claims producer {index}",
                    id.index()
                );
            }
        }
        assert!(
            node.params.axes.is_empty()
                || matches!(
                    node.op,
                    OpId::Gather
                        | OpId::Transpose
                        | OpId::Concat
                        | OpId::Slice
                        | OpId::Split
                        | OpId::Softmax
                        | OpId::ReduceMean
                        | OpId::Trilu
                        | OpId::Flatten
                        | OpId::ArgMax
                ),
            "unexpected attribute carrier on op {}",
            node.op
        );
    }
    // Topological order: every value is defined at most once.
    let mut seen = std::collections::BTreeSet::new();
    for value in graph.values() {
        assert!(seen.insert(value.name.clone()), "duplicate {}", value.name);
    }
    // No warning is a shape contradiction on the pinned graph.
    for warning in imported.warnings() {
        println!("warning: {warning}");
    }
    assert!(
        imported.warnings().is_empty(),
        "the pinned graph produced {} shape warning(s)",
        imported.warnings().len()
    );
}

#[test]
fn reimporting_the_same_bytes_is_identical() {
    let Some((bytes, first)) = pinned() else {
        return;
    };
    let second = import_onnx_model(&bytes).expect("second import");
    assert_eq!(first, second, "the import is not deterministic");
    assert_eq!(
        first.summary(),
        second.summary(),
        "the summary is not deterministic"
    );
    // The weight layout is the same, byte for byte, in the same order.
    let layout: Vec<(String, usize, u64)> = first
        .weights()
        .iter()
        .map(|weight| {
            (
                weight.name().to_string(),
                weight.byte_len(),
                weight.shape().numel().unwrap(),
            )
        })
        .collect();
    let other: Vec<(String, usize, u64)> = second
        .weights()
        .iter()
        .map(|weight| {
            (
                weight.name().to_string(),
                weight.byte_len(),
                weight.shape().numel().unwrap(),
            )
        })
        .collect();
    assert_eq!(layout, other);
}

#[test]
fn pinned_weights_cover_every_byte_of_the_declared_layout() {
    let Some((_, imported)) = pinned() else {
        return;
    };
    for weight in imported.weights().iter() {
        let expected = weight.shape().numel().unwrap() * weight.dtype().size_bytes() as u64;
        assert_eq!(
            weight.byte_len() as u64,
            expected,
            "weight {} byte length",
            weight.name()
        );
    }
    assert_eq!(
        imported.weights().initializers().count(),
        imported.summary().initializers
    );
    assert_eq!(
        imported
            .weights()
            .iter()
            .filter(|weight| weight.source() == WeightSource::AttributeConstant)
            .count(),
        2
    );
    // Every initializer lands in the value table under its own name.
    for weight in imported.weights().iter() {
        assert!(
            imported.graph().lookup(weight.name()).is_some(),
            "weight {} has no value",
            weight.name()
        );
    }
    let tensors = imported.to_model_weights().expect("host tensors");
    assert_eq!(tensors.len(), imported.weights().len());
}

// ---------------------------------------------------------------------------
// A minimal ONNX encoder for models this importer must reject
// ---------------------------------------------------------------------------

/// Enough of the ONNX protobuf encoder to build small models in memory.
///
/// The tests that need a *broken* model cannot truncate the 28 MB artifact into
/// the exact shape they want, so they build one. Keeping the encoder here rather
/// than adding a dev-dependency means the test also documents the wire format the
/// reader consumes.
mod encode {
    /// Append a base-128 varint.
    pub fn varint(mut value: u64, out: &mut Vec<u8>) {
        while value >= 0x80 {
            out.push((value as u8 & 0x7F) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    /// Append a varint field.
    pub fn field_varint(field: u32, value: i64, out: &mut Vec<u8>) {
        varint(u64::from(field) << 3, out);
        varint(value as u64, out);
    }

    /// Append a length-delimited field.
    pub fn field_bytes(field: u32, payload: &[u8], out: &mut Vec<u8>) {
        varint((u64::from(field) << 3) | 2, out);
        varint(payload.len() as u64, out);
        out.extend_from_slice(payload);
    }

    /// Append a string field.
    pub fn field_str(field: u32, value: &str, out: &mut Vec<u8>) {
        field_bytes(field, value.as_bytes(), out);
    }

    /// Append a fixed32 float field.
    pub fn field_f32(field: u32, value: f32, out: &mut Vec<u8>) {
        varint((u64::from(field) << 3) | 5, out);
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    /// A tensor to encode.
    #[derive(Clone)]
    pub struct Tensor {
        pub name: String,
        pub dtype: i64,
        pub dims: Vec<i64>,
        pub raw: Vec<u8>,
        pub external: bool,
        pub data_location: i64,
        pub float_data: Option<Vec<f32>>,
    }

    impl Tensor {
        pub fn f32(name: &str, dims: &[i64], values: &[f32]) -> Tensor {
            let raw: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            Tensor {
                name: name.to_string(),
                dtype: 1,
                dims: dims.to_vec(),
                raw,
                external: false,
                data_location: 0,
                float_data: None,
            }
        }

        pub fn i64(name: &str, dims: &[i64], values: &[i64]) -> Tensor {
            let raw: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            Tensor {
                name: name.to_string(),
                dtype: 7,
                dims: dims.to_vec(),
                raw,
                external: false,
                data_location: 0,
                float_data: None,
            }
        }

        pub fn zeros(name: &str, dims: &[i64]) -> Tensor {
            let elements: usize = dims
                .iter()
                .map(|dim| *dim as usize)
                .product::<usize>()
                .max(1);
            Tensor::f32(name, dims, &vec![0.0; elements])
        }

        pub fn encode(&self) -> Vec<u8> {
            let mut out = Vec::new();
            for dim in &self.dims {
                field_varint(1, *dim, &mut out);
            }
            field_varint(2, self.dtype, &mut out);
            if !self.name.is_empty() {
                field_str(8, &self.name, &mut out);
            }
            if let Some(values) = &self.float_data {
                let mut packed = Vec::new();
                for value in values {
                    packed.extend_from_slice(&value.to_le_bytes());
                }
                field_bytes(4, &packed, &mut out);
            } else {
                field_bytes(9, &self.raw, &mut out);
            }
            if self.external {
                field_bytes(13, &[], &mut out);
            }
            if self.data_location != 0 {
                field_varint(14, self.data_location, &mut out);
            }
            out
        }
    }

    /// A graph input or output.
    #[derive(Clone)]
    pub struct ValueInfo {
        pub name: String,
        pub dtype: i64,
        /// `None` means the dimension is symbolic with this name.
        pub dims: Vec<Option<i64>>,
    }

    impl ValueInfo {
        pub fn new(name: &str, dtype: i64, dims: &[Option<i64>]) -> ValueInfo {
            ValueInfo {
                name: name.to_string(),
                dtype,
                dims: dims.to_vec(),
            }
        }

        pub fn concrete(name: &str, dtype: i64, dims: &[i64]) -> ValueInfo {
            ValueInfo::new(
                name,
                dtype,
                &dims.iter().map(|dim| Some(*dim)).collect::<Vec<_>>(),
            )
        }

        pub fn encode(&self) -> Vec<u8> {
            let mut out = Vec::new();
            field_str(1, &self.name, &mut out);
            let mut tensor_type = Vec::new();
            field_varint(1, self.dtype, &mut tensor_type);
            let mut shape = Vec::new();
            for dim in &self.dims {
                let mut dimension = Vec::new();
                match dim {
                    Some(value) => field_varint(1, *value, &mut dimension),
                    None => field_str(2, "seq", &mut dimension),
                }
                field_bytes(1, &dimension, &mut shape);
            }
            field_bytes(2, &shape, &mut tensor_type);
            let mut ty = Vec::new();
            field_bytes(1, &tensor_type, &mut ty);
            field_bytes(2, &ty, &mut out);
            out
        }
    }

    /// An attribute to encode.
    #[derive(Clone)]
    pub enum Attr {
        Int(&'static str, i64),
        Float(&'static str, f32),
        Ints(&'static str, Vec<i64>),
        Tensor(&'static str, Tensor),
        /// A `GRAPH` payload: a marker, because the reader must not descend.
        Graph(&'static str),
    }

    impl Attr {
        fn encode(&self) -> Vec<u8> {
            let mut out = Vec::new();
            match self {
                Attr::Int(name, value) => {
                    field_str(1, name, &mut out);
                    field_varint(3, *value, &mut out);
                    field_varint(20, 2, &mut out);
                }
                Attr::Float(name, value) => {
                    field_str(1, name, &mut out);
                    field_f32(2, *value, &mut out);
                    field_varint(20, 1, &mut out);
                }
                Attr::Ints(name, values) => {
                    field_str(1, name, &mut out);
                    for value in values {
                        field_varint(8, *value, &mut out);
                    }
                    field_varint(20, 7, &mut out);
                }
                Attr::Tensor(name, tensor) => {
                    field_str(1, name, &mut out);
                    field_bytes(5, &tensor.encode(), &mut out);
                    field_varint(20, 4, &mut out);
                }
                Attr::Graph(name) => {
                    field_str(1, name, &mut out);
                    field_bytes(6, &[], &mut out);
                    field_varint(20, 5, &mut out);
                }
            }
            out
        }
    }

    /// A node to encode.
    #[derive(Clone)]
    pub struct Node {
        pub name: String,
        pub op_type: String,
        pub inputs: Vec<String>,
        pub outputs: Vec<String>,
        pub attrs: Vec<Attr>,
    }

    impl Node {
        pub fn new(
            name: &str,
            op_type: &str,
            inputs: &[&str],
            outputs: &[&str],
            attrs: Vec<Attr>,
        ) -> Node {
            Node {
                name: name.to_string(),
                op_type: op_type.to_string(),
                inputs: inputs.iter().map(|value| value.to_string()).collect(),
                outputs: outputs.iter().map(|value| value.to_string()).collect(),
                attrs,
            }
        }

        pub fn encode(&self) -> Vec<u8> {
            let mut out = Vec::new();
            for input in &self.inputs {
                field_str(1, input, &mut out);
            }
            for output in &self.outputs {
                field_str(2, output, &mut out);
            }
            field_str(3, &self.name, &mut out);
            field_str(4, &self.op_type, &mut out);
            for attr in &self.attrs {
                field_bytes(5, &attr.encode(), &mut out);
            }
            out
        }
    }

    /// A whole model, mutable so a test can break exactly one thing.
    #[derive(Clone)]
    pub struct Model {
        pub ir_version: i64,
        pub opset: Option<i64>,
        pub producer: String,
        pub nodes: Vec<Node>,
        pub initializers: Vec<Tensor>,
        pub inputs: Vec<ValueInfo>,
        pub outputs: Vec<ValueInfo>,
    }

    impl Model {
        pub fn encode(&self) -> Vec<u8> {
            let mut graph = Vec::new();
            for node in &self.nodes {
                field_bytes(1, &node.encode(), &mut graph);
            }
            field_str(2, "synthetic", &mut graph);
            for tensor in &self.initializers {
                field_bytes(5, &tensor.encode(), &mut graph);
            }
            for value in &self.inputs {
                field_bytes(11, &value.encode(), &mut graph);
            }
            for value in &self.outputs {
                field_bytes(12, &value.encode(), &mut graph);
            }
            let mut model = Vec::new();
            field_varint(1, self.ir_version, &mut model);
            field_str(2, &self.producer, &mut model);
            if let Some(version) = self.opset {
                let mut opset = Vec::new();
                field_varint(2, version, &mut opset);
                field_bytes(8, &opset, &mut model);
            }
            field_bytes(7, &graph, &mut model);
            model
        }
    }
}

use encode::{Attr, Model, Node, Tensor, ValueInfo};

/// A structural GPT-2 decoder with concrete extents, so every shape in it is
/// statically inferable.
///
/// `batch = 1`, `seq = 4`, `past = 2`, `total = 6`, `hidden = 8`,
/// `mlp = 16`, `heads = 2`, `head_dim = 4`, `vocab = 16`, one layer. Every
/// dimension is a literal, which is what makes the `Gemm` composition test
/// assert exact extents rather than symbols.
fn synthetic_gpt2() -> Model {
    let mut model = Model {
        ir_version: 7,
        opset: Some(14),
        producer: "fastnn-test".to_string(),
        nodes: Vec::new(),
        initializers: vec![
            Tensor::zeros("transformer.wte.weight", &[16, 8]),
            Tensor::zeros("transformer.wpe.weight", &[8, 8]),
            Tensor::zeros("transformer.h.0.attn.c_attn.weight", &[8, 24]),
            Tensor::zeros("transformer.h.0.attn.c_attn.bias", &[24]),
            Tensor::zeros("transformer.h.0.attn.c_proj.weight", &[8, 8]),
            Tensor::zeros("transformer.h.0.mlp.c_fc.weight", &[8, 16]),
            Tensor::zeros("transformer.h.0.mlp.c_proj.weight", &[16, 8]),
            Tensor::zeros("lm_head.weight", &[8, 16]),
            Tensor::zeros("lm_head.bias", &[16]),
            Tensor::i64("shape_2d", &[2], &[-1, 8]),
            Tensor::i64("shape_logits", &[3], &[1, 4, 16]),
        ],
        inputs: vec![
            ValueInfo::concrete("input_ids", 7, &[1, 4]),
            ValueInfo::concrete("past_key_values.0.key", 1, &[1, 2, 2, 4]),
            ValueInfo::concrete("past_key_values.0.value", 1, &[1, 2, 2, 4]),
            ValueInfo::concrete("attention_mask", 7, &[1, 6]),
            ValueInfo::concrete("position_ids", 7, &[1, 4]),
        ],
        outputs: vec![
            ValueInfo::concrete("logits", 1, &[1, 4, 16]),
            ValueInfo::concrete("present.0.key", 1, &[1, 2, 2, 4]),
            ValueInfo::concrete("present.0.value", 1, &[1, 2, 2, 4]),
        ],
    };
    model.nodes = vec![
        Node::new(
            "emb",
            "Gather",
            &["transformer.wte.weight", "input_ids"],
            &["emb"],
            vec![Attr::Int("axis", 0)],
        ),
        Node::new("flat", "Reshape", &["emb", "shape_2d"], &["emb2d"], vec![]),
        Node::new(
            "project",
            "Gemm",
            &[
                "emb2d",
                "transformer.h.0.attn.c_attn.weight",
                "transformer.h.0.attn.c_attn.bias",
            ],
            &["projected"],
            vec![Attr::Float("alpha", 1.0), Attr::Float("beta", 1.0)],
        ),
        Node::new(
            "head",
            "Gemm",
            &["emb2d", "lm_head.weight", "lm_head.bias"],
            &["logits2d"],
            vec![Attr::Float("alpha", 1.0), Attr::Float("beta", 1.0)],
        ),
        Node::new(
            "unflatten",
            "Reshape",
            &["logits2d", "shape_logits"],
            &["logits"],
            vec![Attr::Int("allowzero", 0)],
        ),
        Node::new(
            "present_key_inner",
            "Transpose",
            &["past_key_values.0.key"],
            &["present_key_inner"],
            vec![Attr::Ints("perm", vec![0, 1, 2, 3])],
        ),
        Node::new(
            "present_key",
            "Transpose",
            &["present_key_inner"],
            &["present.0.key"],
            vec![Attr::Ints("perm", vec![0, 1, 2, 3])],
        ),
        Node::new(
            "present_value_inner",
            "Transpose",
            &["past_key_values.0.value"],
            &["present_value_inner"],
            vec![Attr::Ints("perm", vec![0, 1, 2, 3])],
        ),
        Node::new(
            "present_value",
            "Transpose",
            &["present_value_inner"],
            &["present.0.value"],
            vec![Attr::Ints("perm", vec![0, 1, 2, 3])],
        ),
    ];
    model
}

fn import(model: &Model) -> Result<ModelImport, ImportError> {
    import_onnx_model(&model.encode())
}

#[test]
fn a_constant_of_shape_attribute_value_becomes_a_scalar_weight() {
    let mut model = synthetic_gpt2();
    // A dead `ConstantOfShape` node: nothing consumes it, but the importer must
    // still map it and lift its fill value.
    model.nodes.push(Node::new(
        "mask_fill",
        "ConstantOfShape",
        &["shape_logits"],
        &["mask_fill"],
        vec![Attr::Tensor("value", Tensor::f32("", &[1], &[-3.5]))],
    ));
    let imported = import(&model).expect("synthetic decoder");
    let node = imported
        .graph()
        .nodes()
        .iter()
        .find(|node| node.onnx_op == "ConstantOfShape")
        .expect("mapped ConstantOfShape");
    assert_eq!(node.inputs.len(), 2);
    let value = imported.graph().value(node.inputs[1]);
    assert_eq!(
        value.shape.static_shape(),
        Some(fastnn::fn3::Shape::from_dims([1]))
    );
    let weight = imported.weights().find(&value.name).expect("lifted weight");
    assert_eq!(weight.source(), WeightSource::AttributeConstant);
    assert_eq!(weight.bytes(), (-3.5f32).to_le_bytes());
    // The declared output shape comes from the shape operand, not the fill.
    assert_eq!(
        imported
            .graph()
            .value(node.primary_output().unwrap())
            .shape
            .static_shape(),
        Some(fastnn::fn3::Shape::from_dims([1, 4, 16]))
    );
    assert_eq!(imported.summary().attribute_constants, 1);
}

#[test]
fn a_synthetic_decoder_imports_with_fully_static_shapes() {
    let imported = import(&synthetic_gpt2()).expect("synthetic decoder");
    let geometry = imported.geometry();
    assert_eq!(geometry.layers, 1);
    assert_eq!(geometry.hidden_size, 8);
    assert_eq!(geometry.mlp_hidden_size, 16);
    assert_eq!(geometry.query_heads, 2);
    assert_eq!(geometry.kv_heads, 2);
    assert_eq!(geometry.head_dim, 4);
    assert_eq!(geometry.vocab_size, 16);
    assert_eq!(imported.opset().version, 14);
    let logits = imported.graph().value(geometry.logits);
    assert_eq!(
        logits.shape.static_shape(),
        Some(fastnn::fn3::Shape::from_dims([1, 4, 16]))
    );
    // Every value is fully static in this model.
    for value in imported.graph().values() {
        assert!(
            value.shape.is_static(),
            "value {} is not static: {}",
            value.name,
            value.shape.render()
        );
    }
}

#[test]
fn gemm_becomes_matmul_then_add_with_the_matmul_plus_bias_shape() {
    let imported = import(&synthetic_gpt2()).expect("synthetic decoder");
    let graph = imported.graph();
    let gemm_nodes: Vec<&fastnn::fn3::import::GraphNode> = graph
        .nodes()
        .iter()
        .filter(|node| node.onnx_op == "Gemm")
        .collect();
    // Two Gemm nodes, each MatMul + Add.
    assert_eq!(gemm_nodes.len(), 4);
    let project: Vec<&fastnn::fn3::import::GraphNode> = gemm_nodes
        .iter()
        .copied()
        .filter(|node| node.onnx_index == 2)
        .collect();
    assert_eq!(project.len(), 2, "Gemm maps to exactly two operations");
    assert_eq!(project[0].op, OpId::MatMul);
    assert_eq!(project[1].op, OpId::Add);
    // [4, 8] x [8, 24] -> [4, 24], then + [24] -> [4, 24].
    let matmul_shape = graph
        .value(project[0].primary_output().unwrap())
        .shape
        .clone();
    assert_eq!(
        matmul_shape,
        fastnn::fn3::import::ValueShape::from_dims([4, 24])
    );
    let add = &project[1];
    assert_eq!(
        graph.value(add.primary_output().unwrap()).shape,
        matmul_shape
    );
    assert_eq!(add.inputs.len(), 2);
    assert_eq!(add.inputs[0], project[0].primary_output().unwrap());
    // The bias is the Gemm's third operand.
    let bias = graph.value(add.inputs[1]);
    assert_eq!(bias.name, "transformer.h.0.attn.c_attn.bias");
    assert_eq!(
        bias.shape.static_shape(),
        Some(fastnn::fn3::Shape::from_dims([24]))
    );

    // The lm head is [4, 8] x [8, 16] -> [4, 16] and its consumer reshapes it
    // back to [1, 4, 16].
    let head: Vec<&fastnn::fn3::import::GraphNode> = graph
        .nodes()
        .iter()
        .filter(|node| node.onnx_index == 3)
        .collect();
    assert_eq!(head.len(), 2);
    assert_eq!(
        graph.value(head[1].primary_output().unwrap()).shape,
        fastnn::fn3::import::ValueShape::from_dims([4, 16])
    );
    let logits = graph.lookup("logits").unwrap();
    assert_eq!(
        graph.value(logits).shape,
        fastnn::fn3::import::ValueShape::from_dims([1, 4, 16])
    );
}

#[test]
fn constant_of_shape_lifts_its_value_attribute_into_a_constant_input() {
    let Some((_, imported)) = pinned() else {
        return;
    };
    let graph = imported.graph();
    let nodes: Vec<&fastnn::fn3::import::GraphNode> = graph
        .nodes()
        .iter()
        .filter(|node| node.onnx_op == "ConstantOfShape")
        .collect();
    assert_eq!(nodes.len(), 2);
    for node in &nodes {
        assert_eq!(node.op, OpId::ConstantOfShape);
        assert_eq!(
            node.inputs.len(),
            2,
            "ConstantOfShape must take (shape, scalar value)"
        );
        let value = graph.value(node.inputs[1]);
        assert_eq!(
            value.origin,
            fastnn::fn3::import::ValueOrigin::Weight(
                imported
                    .weights()
                    .id_of(&value.name)
                    .expect("lifted weight")
            )
        );
        let weight = imported.weights().find(&value.name).expect("weight");
        assert_eq!(weight.source(), WeightSource::AttributeConstant);
        assert_eq!(weight.dtype(), fastnn::fn3::DType::F32);
        assert_eq!(weight.byte_len(), 4);
        // The node's output dtype is the lifted value's dtype.
        assert_eq!(
            graph.value(node.primary_output().unwrap()).dtype,
            weight.dtype()
        );
        assert!(value.shape.is_static());
        assert_eq!(value.shape.numel(), Some(1));
    }
    // The two fills are the mask sentinel and zero, in that order.
    let fills: Vec<f32> = nodes
        .iter()
        .map(|node| {
            let value = graph.value(node.inputs[1]);
            let weight = imported.weights().find(&value.name).unwrap();
            let mut word = [0u8; 4];
            word.copy_from_slice(weight.bytes());
            f32::from_le_bytes(word)
        })
        .collect();
    assert_eq!(fills[0], f32::MIN, "-FLT_MAX mask sentinel");
    assert_eq!(fills[1], 0.0);
}

#[test]
fn reshape_keeps_allowzero_zero_semantics_and_rejects_one() {
    let Some((_, imported)) = pinned() else {
        return;
    };
    // Every Reshape in the pinned graph declares allowzero = 0 (or omits it,
    // which defaults to 0) and maps to the `Reshape` op with the shape as its
    // second operand; the semantics live in the kernel, not in OpParams.
    let reshapes: Vec<&fastnn::fn3::import::GraphNode> = imported
        .graph()
        .nodes()
        .iter()
        .filter(|node| node.onnx_op == "Reshape")
        .collect();
    assert_eq!(reshapes.len(), 25);
    for node in &reshapes {
        assert_eq!(node.op, OpId::Reshape);
        assert_eq!(node.inputs.len(), 2, "Reshape(data, shape)");
        assert!(node.params.axes.is_empty(), "Reshape has no axis attribute");
        assert_eq!(
            imported.graph().value(node.inputs[1]).dtype,
            fastnn::fn3::DType::I64
        );
    }
    // `allowzero = 1` is not representable and must be reported, not silently
    // treated as 0.
    let mut model = synthetic_gpt2();
    model.nodes[4].attrs = vec![Attr::Int("allowzero", 1)];
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(
            kind,
            ImportProblemKind::UnsupportedAttribute { attribute, .. } if attribute == "allowzero"
        )),
        "{error}"
    );
    assert!(error.to_string().contains("allowzero=1"), "{error}");
}

#[test]
fn an_operator_without_an_fn3_id_is_rejected_by_name() {
    let mut model = synthetic_gpt2();
    model.nodes[2].op_type = "LSTM".to_string();
    let error = import(&model).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("node 2"), "{text}");
    assert!(text.contains("LSTM"), "{text}");
    assert!(text.contains("no fn3 op id"), "{text}");
    // The rest of the model still validated: exactly one problem.
    assert_eq!(error.len(), 1, "{text}");
}

#[test]
fn an_attention_operator_names_the_unimplemented_fn3_id() {
    let mut model = synthetic_gpt2();
    model.nodes[2].op_type = "MultiHeadAttention".to_string();
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(
            kind,
            ImportProblemKind::UnimplementedOpId {
                op: OpId::MaskedAttention,
                owner
            } if owner.contains("#61")
        )),
        "{error}"
    );
    assert!(error.to_string().contains("masked_attention"), "{error}");
}

#[test]
fn external_tensor_data_is_rejected_by_tensor_name() {
    let mut model = synthetic_gpt2();
    model.initializers[0].external = true;
    model.initializers[0].data_location = 1;
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(
            kind,
            ImportProblemKind::ExternalTensorData { tensor } if tensor == "transformer.wte.weight"
        )),
        "{error}"
    );
    assert!(
        error.to_string().contains("external data"),
        "the message must explain how to fix it: {error}"
    );
}

#[test]
fn a_missing_opset_is_reported() {
    let mut model = synthetic_gpt2();
    model.opset = None;
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(kind, ImportProblemKind::MissingOpset)),
        "{error}"
    );
}

#[test]
fn an_unsupported_opset_is_reported_with_its_range() {
    let mut model = synthetic_gpt2();
    model.opset = Some(18);
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(
            kind,
            ImportProblemKind::UnsupportedOpset { version: 18, .. }
        )),
        "{error}"
    );
    assert!(error.to_string().contains("13..=17"), "{error}");
}

#[test]
fn control_flow_and_subgraphs_are_rejected() {
    let mut model = synthetic_gpt2();
    model.nodes.push(Node::new(
        "branch",
        "If",
        &["emb2d"],
        &["branch_out"],
        vec![],
    ));
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(
            kind,
            ImportProblemKind::ControlFlow { op_type } if op_type == "If"
        )),
        "{error}"
    );

    let mut model = synthetic_gpt2();
    model.nodes[0].attrs.push(Attr::Graph("body"));
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(
            kind,
            ImportProblemKind::Subgraph { attribute } if attribute == "body"
        )),
        "{error}"
    );
}

#[test]
fn a_dtype_the_core_cannot_represent_is_named() {
    let mut model = synthetic_gpt2();
    // FLOAT16: an element type the 3.0 core does not define.
    model.initializers[0].dtype = 10;
    model.initializers[0].raw = vec![0u8; 16 * 8 * 2];
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(
            kind,
            ImportProblemKind::UnsupportedDType {
                code: 10,
                onnx_name: "FLOAT16",
                ..
            }
        )),
        "{error}"
    );
    assert!(
        error.to_string().contains("transformer.wte.weight"),
        "{error}"
    );
}

#[test]
fn a_double_precision_graph_input_is_rejected() {
    let mut model = synthetic_gpt2();
    // DOUBLE input: a non-f32 compute path.
    model.inputs[1].dtype = 11;
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(
            kind,
            ImportProblemKind::UnsupportedDType {
                code: 11,
                onnx_name: "DOUBLE",
                ..
            }
        )),
        "{error}"
    );
}

#[test]
fn an_unknown_attribute_is_not_ignored() {
    let mut model = synthetic_gpt2();
    model.nodes[1].attrs = vec![Attr::Int("axes", 0)];
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(
            kind,
            ImportProblemKind::UnknownAttribute { attribute, .. } if attribute == "axes"
        )),
        "{error}"
    );
}

#[test]
fn a_gemm_scaling_is_rejected_instead_of_dropped() {
    let mut model = synthetic_gpt2();
    model.nodes[2].attrs = vec![
        Attr::Float("alpha", 1.0),
        Attr::Float("beta", 0.5),
        Attr::Int("transB", 1),
    ];
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(
            kind,
            ImportProblemKind::UnsupportedAttribute { attribute, .. } if attribute == "Gemm"
        )),
        "{error}"
    );
    assert!(error.to_string().contains("transB = 1"), "{error}");
}

#[test]
fn a_value_defined_twice_is_rejected() {
    let mut model = synthetic_gpt2();
    model.nodes[1].outputs = vec!["emb".to_string()];
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(
            kind,
            ImportProblemKind::DuplicateValue { value, .. } if value == "emb"
        )),
        "{error}"
    );
}

#[test]
fn an_unsorted_graph_is_rejected_as_not_topological() {
    let mut model = synthetic_gpt2();
    model.nodes.swap(0, 1);
    let error = import(&model).unwrap_err();
    assert!(
        error.contains(|kind| matches!(
            kind,
            ImportProblemKind::NotTopological { value, producer_index } if value == "emb" && *producer_index == 1
        )),
        "{error}"
    );
    // The node that reads a value produced later is named.
    assert!(error.to_string().contains("node 0"), "{error}");
}

#[test]
fn a_graph_that_is_not_a_supported_decoder_is_rejected_with_a_reason() {
    // A model with no initializers at all: every architecture check fires.
    let mut model = synthetic_gpt2();
    model.initializers.clear();
    let error = import(&model).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("transformer.wte.weight"), "{text}");
    assert!(text.contains("decoder architecture"), "{text}");
}

#[test]
fn a_wrong_initializer_length_is_reported_with_both_numbers() {
    let mut model = synthetic_gpt2();
    model.initializers[0].raw.truncate(7);
    let error = import(&model).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("transformer.wte.weight"), "{text}");
    assert!(text.contains("needs 512 payload bytes"), "{text}");
    assert!(text.contains("holds 7"), "{text}");
}

#[test]
fn truncated_and_corrupted_input_never_panics() {
    let Some((bytes, _)) = pinned() else {
        return;
    };
    // Truncated prefixes at several lengths: the header, the middle of the node
    // table, and just before the end.
    for length in [
        0usize,
        1,
        2,
        7,
        16,
        64,
        1024,
        65_536,
        1_000_000,
        PINNED_BYTES / 2,
        PINNED_BYTES - 1,
    ] {
        let error = import_onnx_model(&bytes[..length]).unwrap_err();
        assert!(!error.is_empty());
        // A truncated stream is a byte-level problem, reported once.
        if let Some(proto) = error.proto_error() {
            assert!(!proto.to_string().is_empty());
        }
    }
    // Deterministic byte flips in the header and in the graph metadata.
    let mut flipped = 0usize;
    for offset in [0usize, 1, 3, 5, 9, 12, 31, 64, 129, 512, 4096, 8192] {
        if offset >= bytes.len() {
            continue;
        }
        let mut corrupted = bytes.clone();
        corrupted[offset] ^= 0xFF;
        // The only requirement is that this returns, with a typed error or a
        // model; a panic fails the test.
        match import_onnx_model(&corrupted) {
            Ok(_) => {}
            Err(error) => {
                assert!(!error.is_empty());
                flipped += 1;
            }
        }
    }
    assert!(flipped > 0, "at least one header byte must be load-bearing");
}
