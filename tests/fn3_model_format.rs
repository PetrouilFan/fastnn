//! FN3-09 (#55): the versioned prepared-model artifact.
//!
//! # What these tests prove
//!
//! 1. **Round trip of the real pinned model.** `onnx-community/tiny-random-gpt2-ONNX`
//!    (27,897,974 bytes) imports, writes, reads back and re-writes
//!    byte-identically, with every weight, geometry field, value, node and graph
//!    handle compared.
//! 2. **Determinism.** Two independent imports write the same bytes, and writing
//!    the same artifact twice is byte-identical.
//! 3. **Untrusted input is rejected, never absorbed.** Bad magic, a future
//!    version, a past version, a truncated header, a truncated payload, a wrong
//!    checksum, an unknown dtype code, an unknown op code, counts larger than
//!    the file, counts large enough to be a hostile allocation, reserved fields,
//!    a foreign dtype encoding, trailing bytes and invalid UTF-8 all produce
//!    typed errors. Every single-byte change of a small artifact is detected.
//! 4. **No session state.** The section inventory is exactly the six structural
//!    sections, an unknown section tag is rejected, a section with extra fields
//!    is rejected, and creating sessions over a model built from the artifact
//!    changes neither the artifact nor its bytes.
//! 5. **The 2.x policy.** A `.fnnc`-shaped file (raw `bincode`, no magic) is
//!    rejected at the magic check, and nothing in this build reads or re-exports
//!    a 2.x artifact.
//!
//! # Running the pinned-model tests
//!
//! The 28 MB model is not committed (it has no declared license). Point the
//! tests at it with `FASTNN_FN3_ONNX_MODEL=/path/to/model.onnx`, or place it at
//! `tools/parity/.cache/models/tiny-gpt2/onnx/model.onnx`. Without either, those
//! tests print a reason and pass; the negative and structural tests always run
//! against a small artifact built through the public API.

use std::path::PathBuf;
use std::time::Instant;

use fastnn::fn3::import::{
    import_onnx_model, DecoderGeometry, Dim, GraphNode, KvInterface, ModelImport, OpsetImport,
    PositionEmbedding, ValueId, ValueInfo, ValueOrigin, ValueShape,
};
use fastnn::fn3::model::format::{
    ArtifactMetadata, FormatError, FormatErrorKind, PreparedArtifact, PreparedGraph,
    FORMAT_VERSION, HEADER_BYTES, MAGIC, MAX_RANK, MAX_VALUES,
};
use fastnn::fn3::model::{
    weights::{WeightId, WeightSet, WeightSource, WeightTensor},
    ExecutionPlan, ModelWeights, PreparedModel, Session, MAX_KV_LAYERS,
};
use fastnn::fn3::shape::{DimBound, Shape, ShapeEnvelope, Symbol};
use fastnn::fn3::{DType, DeviceId, OpId, OpParams};

// ---------------------------------------------------------------------------
// Pinned artifact
// ---------------------------------------------------------------------------

/// Byte length of the pinned graph (`fixtures/parity/tiny-gpt2/model.lock.json`).
const PINNED_BYTES: usize = 27_897_974;

/// FNV-1a 64-bit fingerprint of the pinned file.
///
/// The authoritative identity is the SHA-256 in `model.lock.json`
/// (`e18370ba…141a1`), checked out of band; this digest lets the test notice
/// that the file at the configured path is a different graph without carrying a
/// hash implementation.
const PINNED_FNV1A: u64 = 0xc294_d3ab_837f_b1be;

/// Locate the pinned model, or explain why the test is skipped.
fn pinned_bytes() -> Option<Vec<u8>> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("FASTNN_FN3_ONNX_MODEL") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tools/parity/.cache/models/tiny-gpt2/onnx/model.onnx"),
    );
    for path in candidates {
        if let Ok(bytes) = std::fs::read(&path) {
            return Some(bytes);
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

/// Import the pinned model, or `None` with a printed reason.
fn pinned_import() -> Option<ModelImport> {
    let bytes = pinned_bytes()?;
    assert_eq!(
        bytes.len(),
        PINNED_BYTES,
        "the file at the configured path is not the pinned graph"
    );
    assert_eq!(
        fnv1a(&bytes),
        PINNED_FNV1A,
        "the file at the configured path is not the pinned graph"
    );
    let imported = import_onnx_model(&bytes)
        .unwrap_or_else(|error| panic!("the pinned model failed to import:\n{error}"));
    Some(imported)
}

/// The declared dynamic bounds the pinned tests use: at most 1024 tokens.
fn pinned_envelope() -> ShapeEnvelope {
    ShapeEnvelope::uniform(1024).expect("1024 is within DimBound::MAX_EXTENT")
}

// ---------------------------------------------------------------------------
// Byte surgery helpers
//
// These duplicate the documented v1 layout on purpose: the integration test is
// an independent check that the implementation matches the published
// specification, not a mirror of the implementation's own constants.
// ---------------------------------------------------------------------------

const TAG_META: u32 = 1;
const TAG_GEOM: u32 = 2;
const TAG_ENV: u32 = 3;
const TAG_VALS: u32 = 4;
const TAG_NODES: u32 = 5;
const TAG_WEIGHTS: u32 = 6;

/// `(tag, file offset of the first byte, unpadded length)` for every section.
fn directory(bytes: &[u8]) -> Vec<(u32, usize, u64)> {
    assert!(bytes.len() > HEADER_BYTES, "not a framed file");
    let count = u32::from_le_bytes(bytes[HEADER_BYTES..HEADER_BYTES + 4].try_into().unwrap());
    let mut entries = Vec::new();
    let mut offset = HEADER_BYTES + 8 + count as usize * 16;
    for index in 0..count as usize {
        let at = HEADER_BYTES + 8 + index * 16;
        let tag = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let length = u64::from_le_bytes(bytes[at + 8..at + 16].try_into().unwrap());
        entries.push((tag, offset, length));
        offset += length as usize;
        offset += (8 - (length as usize % 8)) % 8;
    }
    entries
}

fn section_offset(bytes: &[u8], tag: u32) -> usize {
    directory(bytes)
        .into_iter()
        .find(|(found, _, _)| *found == tag)
        .map(|(_, offset, _)| offset)
        .unwrap_or_else(|| panic!("section {tag} is not present"))
}

/// Recompute the header checksum after a payload edit.
fn repair_checksum(bytes: &mut [u8]) {
    let crc = crc32(&bytes[HEADER_BYTES..]);
    bytes[40..44].copy_from_slice(&crc.to_le_bytes());
}

/// CRC-32/IEEE, reimplemented in the test so the checksum is verified against
/// the published polynomial rather than the implementation's own helper.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    crc ^ 0xFFFF_FFFF
}

fn expect_error(bytes: &[u8]) -> FormatError {
    PreparedArtifact::from_bytes(bytes).expect_err("the mutated artifact must be rejected")
}

// ---------------------------------------------------------------------------
// A small artifact built through the public API
// ---------------------------------------------------------------------------

fn tiny_geometry() -> DecoderGeometry {
    DecoderGeometry {
        layers: 1,
        hidden_size: 4,
        mlp_hidden_size: 8,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 2,
        vocab_size: 6,
        positions: PositionEmbedding::Learned {
            weight: WeightId::new(1),
            max_positions: 3,
        },
        layer_norm_epsilon: Some(1e-5),
        token_embedding: WeightId::new(0),
        logits: ValueId::new(4),
        kv: vec![KvInterface {
            past_key: ValueId::new(1),
            past_value: ValueId::new(2),
            present_key: ValueId::new(3),
            present_value: ValueId::new(4),
        }],
    }
}

fn tiny_graph() -> PreparedGraph {
    let f32_value = |name: &str, shape: ValueShape, origin: ValueOrigin| ValueInfo {
        name: name.to_string(),
        dtype: DType::F32,
        shape,
        origin,
    };
    PreparedGraph::new(
        vec![
            f32_value(
                "past.0.key",
                ValueShape::Dims(vec![Dim::Known(1), Dim::Known(2)]),
                ValueOrigin::GraphInput,
            ),
            f32_value(
                "past.0.value",
                ValueShape::Dims(vec![Dim::Symbol(Symbol::Past)]),
                ValueOrigin::GraphInput,
            ),
            f32_value(
                "present.0.key",
                ValueShape::Unknown,
                ValueOrigin::GraphInput,
            ),
            f32_value(
                "present.0.value",
                ValueShape::Dims(vec![Dim::unknown("past + seq")]),
                ValueOrigin::GraphInput,
            ),
            f32_value(
                "logits",
                ValueShape::Dims(vec![Dim::Known(6)]),
                ValueOrigin::Node(0),
            ),
            f32_value(
                "wpe",
                ValueShape::Dims(vec![Dim::Known(3), Dim::Known(4)]),
                ValueOrigin::Weight(WeightId::new(1)),
            ),
            f32_value(
                "wte",
                ValueShape::Dims(vec![Dim::Known(6), Dim::Known(4)]),
                ValueOrigin::Weight(WeightId::new(0)),
            ),
        ],
        vec![GraphNode {
            onnx_index: 3,
            name: "matmul".to_string(),
            onnx_op: "MatMul".to_string(),
            op: OpId::MatMul,
            inputs: vec![ValueId::new(0), ValueId::new(6)],
            outputs: vec![ValueId::new(4)],
            params: OpParams {
                axes: vec![-1, 0, 3],
                keep_dims: true,
                epsilon: 1e-5,
            },
        }],
        vec![
            ValueId::new(0),
            ValueId::new(1),
            ValueId::new(2),
            ValueId::new(3),
        ],
        vec![ValueId::new(4)],
    )
    .expect("the synthetic graph is consistent")
}

fn tiny_weights() -> WeightSet {
    WeightSet::new(vec![
        WeightTensor::new(
            "wte",
            DType::F32,
            Shape::from_dims([6, 4]),
            vec![0x11; 6 * 4 * 4],
            WeightSource::Initializer,
        )
        .expect("payload matches shape"),
        WeightTensor::new(
            "wpe",
            DType::F32,
            Shape::from_dims([3, 4]),
            vec![0x22; 3 * 4 * 4],
            WeightSource::AttributeConstant,
        )
        .expect("payload matches shape"),
    ])
    .expect("names are unique")
}

fn tiny_metadata() -> ArtifactMetadata {
    ArtifactMetadata {
        ir_version: 7,
        opset: OpsetImport {
            domain: String::new(),
            version: 14,
        },
        producer: "pytorch".to_string(),
        producer_version: "2.8.0".to_string(),
        graph_name: "main_graph".to_string(),
    }
}

fn tiny_artifact() -> PreparedArtifact {
    PreparedArtifact::from_parts(
        tiny_metadata(),
        tiny_graph(),
        tiny_weights(),
        tiny_geometry(),
        ShapeEnvelope::uniform(8).expect("8 is within DimBound::MAX_EXTENT"),
    )
    .expect("the synthetic artifact is consistent")
}

// ---------------------------------------------------------------------------
// 1. Round trip of the pinned model
// ---------------------------------------------------------------------------

#[test]
fn pinned_model_round_trips_with_exact_bytes_and_structure() {
    let Some(imported) = pinned_import() else {
        return;
    };

    let artifact = PreparedArtifact::from_import(&imported, pinned_envelope())
        .expect("the pinned import is a consistent artifact");

    // Warm the allocator, then measure the two operations that matter.
    let warm = artifact.to_bytes().expect("the artifact encodes");
    assert_eq!(
        warm,
        artifact.to_bytes().expect("encoding is deterministic")
    );
    let started = Instant::now();
    let bytes = artifact.to_bytes().expect("the artifact encodes");
    let write_time = started.elapsed();
    let started = Instant::now();
    let loaded = PreparedArtifact::from_bytes(&bytes).expect("the artifact decodes");
    let read_time = started.elapsed();
    println!(
        "[fn3fmt][measure] pinned artifact: {} bytes ({} header + {} payload), \
         write={:?} read={:?} (second pass, warm allocator, {} build)",
        bytes.len(),
        HEADER_BYTES,
        bytes.len() - HEADER_BYTES,
        write_time,
        read_time,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    let sections: Vec<String> = directory(&bytes)
        .into_iter()
        .map(|(tag, _, length)| format!("{tag}:{length}"))
        .collect();
    println!(
        "[fn3fmt][measure] sections (tag:unpadded bytes): {}",
        sections.join(" ")
    );

    // Size: the payload is the weight bytes plus the tables plus padding.
    assert_eq!(
        artifact.weight_bytes(),
        imported.summary().weight_bytes,
        "the artifact must carry every weight byte the import found"
    );
    let overhead = bytes.len() as u64 - artifact.weight_bytes();
    assert!(
        overhead < 1 << 20,
        "structural overhead of {overhead} bytes is larger than expected"
    );
    assert!(bytes.len() > PINNED_BYTES - 1_000_000);
    assert!(bytes.len() < PINNED_BYTES + 1_000_000);

    // Metadata, geometry, envelope.
    assert_eq!(loaded.metadata(), artifact.metadata());
    assert_eq!(loaded.metadata().ir_version, 7);
    assert_eq!(loaded.metadata().opset.version, 14);
    assert_eq!(loaded.metadata().producer, "pytorch");
    assert_eq!(loaded.metadata().graph_name, "main_graph");
    assert_eq!(loaded.geometry(), artifact.geometry());
    assert_eq!(loaded.geometry().layers, 2);
    assert_eq!(loaded.geometry().hidden_size, 128);
    assert_eq!(loaded.geometry().mlp_hidden_size, 512);
    assert_eq!(loaded.geometry().query_heads, 2);
    assert_eq!(loaded.geometry().kv_heads, 2);
    assert_eq!(loaded.geometry().head_dim, 64);
    assert_eq!(loaded.geometry().vocab_size, 50_257);
    assert_eq!(loaded.geometry().layer_norm_epsilon, Some(1e-5));
    assert_eq!(loaded.envelope(), pinned_envelope());
    assert_eq!(loaded.envelope().get(Symbol::Total).max(), 1024);

    // Graph: values, nodes and the interface, element by element.
    assert_eq!(loaded.graph().value_count(), 326);
    assert_eq!(loaded.graph().nodes().len(), 265);
    assert_eq!(
        loaded.graph().values(),
        artifact.graph().values(),
        "the value table changed across the round trip"
    );
    assert_eq!(
        loaded.graph().nodes(),
        artifact.graph().nodes(),
        "the op list changed across the round trip"
    );
    assert_eq!(loaded.graph().inputs(), artifact.graph().inputs());
    assert_eq!(loaded.graph().outputs(), artifact.graph().outputs());

    // Every weight, in order, with its exact bytes.
    assert_eq!(loaded.weights().len(), 50);
    assert_eq!(loaded.weights().len(), artifact.weights().len());
    for (position, (read_back, original)) in loaded
        .weights()
        .iter()
        .zip(artifact.weights().iter())
        .enumerate()
    {
        assert_eq!(read_back.name(), original.name(), "weight {position} name");
        assert_eq!(
            read_back.dtype(),
            original.dtype(),
            "weight {position} dtype"
        );
        assert_eq!(
            read_back.shape(),
            original.shape(),
            "weight {position} shape"
        );
        assert_eq!(
            read_back.source(),
            original.source(),
            "weight {position} source"
        );
        assert_eq!(
            read_back.bytes(),
            original.bytes(),
            "weight {position} (`{}`) payload",
            original.name()
        );
    }
    for (read_back, original) in loaded
        .weights()
        .iter_ids()
        .zip(artifact.weights().iter_ids())
    {
        assert_eq!(read_back.0, original.0, "weight handle order");
    }

    // Loading and re-saving is byte-identical.
    assert_eq!(
        loaded.to_bytes().expect("re-encoding works"),
        bytes,
        "load -> save must reproduce the artifact byte for byte"
    );

    // A loaded artifact can produce device tensors, and they match the import.
    let weights = loaded
        .to_model_weights()
        .expect("the loaded weights become host tensors");
    assert_eq!(weights.len(), 50);
    let imported_weights = imported
        .to_model_weights()
        .expect("the imported weights become host tensors");
    assert_eq!(weights.len(), imported_weights.len());
    for index in 0..weights.len() {
        let loaded_tensor = weights.get(index).expect("tensor exists");
        let imported_tensor = imported_weights.get(index).expect("tensor exists");
        assert_eq!(loaded_tensor.shape(), imported_tensor.shape());
        assert_eq!(loaded_tensor.dtype(), imported_tensor.dtype());
        assert_eq!(
            loaded_tensor.storage().as_bytes(),
            imported_tensor.storage().as_bytes(),
            "tensor {index} bytes"
        );
    }
}

#[test]
fn an_artifact_alone_plans_and_runs_with_no_onnx_source() {
    let Some(imported) = pinned_import() else {
        return;
    };

    // The two-file path, as a reference: plan and prefill from the import.
    let reference = prefill_logits(&imported);
    let expected_summary = imported.summary().clone();

    // Prepare, encode, and then drop the import and the artifact object: from
    // here on the only thing that exists is a file.
    let artifact = PreparedArtifact::from_import(&imported, pinned_envelope())
        .expect("the pinned import is a consistent artifact");
    let bytes = artifact.to_bytes().expect("the artifact encodes");
    drop(artifact);
    drop(imported);

    let directory =
        std::env::temp_dir().join(format!("fastnn-fn3-artifact-only-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).expect("scratch directory");
    let path = directory.join("tiny-gpt2.fnn3");
    std::fs::write(&path, &bytes).expect("write the artifact");
    drop(bytes);

    // `read_file` is the only input. No ONNX path is passed, stored or read.
    let loaded = PreparedArtifact::read_file(&path).expect("the artifact reads back");
    let metadata = loaded.metadata().clone();
    let weight_bytes = loaded.weight_bytes();

    // Both conversion forms agree, so a caller holding `&PreparedArtifact`
    // gets the same input as one that can consume it.
    assert_eq!(loaded.to_model_import(), loaded.clone().into_model_import());
    let graph = loaded.into_model_import();

    // The artifact rebuilt the importer's exact input: its own metadata
    // round-trips (a verifier can check this), and the summary is reproduced
    // from the graph and weight tables rather than degraded.
    assert_eq!(ArtifactMetadata::from_import(&graph), metadata);
    assert_eq!(*graph.summary(), expected_summary);
    assert_eq!(graph.summary().onnx_nodes, 257);
    assert_eq!(graph.summary().fn3_nodes, 265);
    assert_eq!(graph.summary().weight_bytes, weight_bytes);
    assert_eq!(graph.weights().len(), 50);

    let artifact_only = prefill_logits(&graph);
    assert_eq!(
        artifact_only.len(),
        reference.len(),
        "the logits shape changed between the two paths"
    );
    for (index, (left, right)) in artifact_only.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "logit {index} differs: {left} from the artifact alone, {right} with the ONNX"
        );
    }

    // The documented top-5 for the 10-token prompt, from the artifact alone.
    let top5 = top_k(&artifact_only, 5);
    assert_eq!(top5, vec![46_737, 28_597, 16_571, 7_868, 13]);

    let _ = std::fs::remove_dir_all(&directory);
}

/// The pinned 10-token prompt used by the CLI, the Python tests and the docs.
const PROMPT_10: [u32; 10] = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13];

/// Plan, run and prefill `prompt` through one CPU session over `import`.
fn prefill_logits(import: &ModelImport) -> Vec<f32> {
    use fastnn::fn3::model::Planner;
    use fastnn::fn3::{CpuBackend, LiveDims};

    let backend = CpuBackend::new();
    let envelope = pinned_envelope();
    let geometry = import.geometry();
    let dims = LiveDims::new(
        1,
        4,
        0,
        u64::from(geometry.query_heads),
        u64::from(geometry.kv_heads),
        u64::from(geometry.head_dim),
    )
    .expect("head geometry");
    let plan = Planner::new(import, &backend)
        .plan(&envelope, dims)
        .expect("the envelope and dims are admissible")
        .execution()
        .clone();
    let config = geometry.kv_cache_config(envelope.get(Symbol::Total).max(), 1);
    let weights = import.to_model_weights().expect("host weights");
    let model = PreparedModel::new(plan, weights, DeviceId::Cpu, envelope, Some(config))
        .expect("the model is consistent");
    let mut session =
        Session::with_runtime(&model, import, &backend).expect("the session opens on cpu");
    let logits = session.prefill(&PROMPT_10).expect("prefill runs");
    logits
        .as_f32_slice()
        .expect("cpu logits are host memory")
        .to_vec()
}

/// Token IDs of the `k` highest logits, ties broken by the lower ID.
fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked[..k].iter().map(|(index, _)| *index).collect()
}

#[test]
fn two_independent_imports_write_identical_artifacts() {
    let Some(first_import) = pinned_import() else {
        return;
    };
    let Some(second_import) = pinned_import() else {
        return;
    };
    let first = PreparedArtifact::from_import(&first_import, pinned_envelope()).unwrap();
    let second = PreparedArtifact::from_import(&second_import, pinned_envelope()).unwrap();
    let first_bytes = first.to_bytes().unwrap();
    let second_bytes = second.to_bytes().unwrap();
    assert_eq!(
        first_bytes, second_bytes,
        "two imports of the same bytes must write the same artifact"
    );
    assert_eq!(first_bytes.len(), second_bytes.len());
    // And the artifact is self-consistent in the sense a consumer relies on.
    assert_eq!(first.geometry(), second.geometry());
    assert_eq!(first.metadata(), second.metadata());
}

// ---------------------------------------------------------------------------
// 2. Framing and version rejections
// ---------------------------------------------------------------------------

#[test]
fn bad_magic_is_rejected() {
    let mut bytes = tiny_artifact().to_bytes().unwrap();
    bytes[0] = b'X';
    let error = expect_error(&bytes);
    assert!(matches!(error.kind(), FormatErrorKind::BadMagic { .. }));
    assert!(error.is_corruption());
    let text = error.to_string();
    assert!(text.contains("not a FastNN 3 prepared model"), "{text}");
    assert!(text.contains("46 4e 4e 33 50 52 45 50"), "{text}");
}

#[test]
fn a_future_version_is_rejected_by_version_not_by_parsing() {
    let mut bytes = tiny_artifact().to_bytes().unwrap();
    bytes[8..10].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
    let error = expect_error(&bytes);
    assert!(matches!(
        error.kind(),
        FormatErrorKind::UnsupportedVersion {
            found: 2,
            supported: 1
        }
    ));
    assert!(error.is_version_mismatch());
    let text = error.to_string();
    assert!(text.contains("version 2"), "{text}");
    assert!(text.contains("version 1 only"), "{text}");
    // Everything after the version is untouched: the rejection happened at byte
    // 8, before any structure was interpreted.
    assert_eq!(&bytes[64..], &tiny_artifact().to_bytes().unwrap()[64..]);
}

#[test]
fn a_past_version_is_rejected() {
    let mut bytes = tiny_artifact().to_bytes().unwrap();
    bytes[8..10].copy_from_slice(&0u16.to_le_bytes());
    let error = expect_error(&bytes);
    assert!(matches!(
        error.kind(),
        FormatErrorKind::UnsupportedVersion {
            found: 0,
            supported: 1
        }
    ));
    assert!(error.to_string().contains("version 0"), "{error}");
}

#[test]
fn a_different_dtype_encoding_is_rejected() {
    let mut bytes = tiny_artifact().to_bytes().unwrap();
    bytes[12..16].copy_from_slice(&2u32.to_le_bytes());
    let error = expect_error(&bytes);
    assert!(matches!(
        error.kind(),
        FormatErrorKind::UnsupportedDtypeEncoding {
            found: 2,
            supported: 1
        }
    ));
    assert!(error.is_version_mismatch());
    assert!(error.to_string().contains("dtype encoding 2"), "{error}");
}

#[test]
fn header_reserved_fields_must_be_zero() {
    for (offset, width) in [(16usize, 4usize), (20, 4), (44, 20)] {
        let mut bytes = tiny_artifact().to_bytes().unwrap();
        bytes[offset] = 0x7F;
        let error = expect_error(&bytes);
        assert!(
            matches!(error.kind(), FormatErrorKind::ReservedField { .. }),
            "offset {offset} (width {width}): {error}"
        );
    }
}

#[test]
fn a_header_length_this_build_does_not_use_is_rejected() {
    let mut bytes = tiny_artifact().to_bytes().unwrap();
    bytes[10..12].copy_from_slice(&128u16.to_le_bytes());
    let error = expect_error(&bytes);
    assert!(matches!(
        error.kind(),
        FormatErrorKind::LengthMismatch {
            what: "header length",
            ..
        }
    ));
}

#[test]
fn a_truncated_header_is_rejected() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    for length in [0usize, 1, 7, 8, 9, 10, 12, 16, 24, 40, 44, 63] {
        let error = expect_error(&bytes[..length]);
        assert!(
            matches!(error.kind(), FormatErrorKind::Truncated { .. }),
            "length {length}: {error}"
        );
        assert!(error.is_corruption());
        assert!(
            error.to_string().contains("prepared-model header"),
            "length {length}: {error}"
        );
    }
}

#[test]
fn a_truncated_file_is_rejected() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    for length in [HEADER_BYTES + 1, HEADER_BYTES + 8, bytes.len() - 1] {
        let error = expect_error(&bytes[..length]);
        assert!(
            matches!(error.kind(), FormatErrorKind::LengthMismatch { .. }),
            "length {length}: {error}"
        );
        assert!(error.to_string().contains("file length"), "{error}");
    }
}

#[test]
fn a_section_length_beyond_the_payload_is_reported_as_truncated() {
    let mut bytes = tiny_artifact().to_bytes().unwrap();
    // The first directory entry's length field, with the checksum repaired so the
    // section check is what fails.
    bytes[HEADER_BYTES + 16..HEADER_BYTES + 24].copy_from_slice(&u64::MAX.to_le_bytes());
    repair_checksum(&mut bytes);
    let error = expect_error(&bytes);
    assert!(
        matches!(error.kind(), FormatErrorKind::Truncated { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("META section"), "{error}");

    let mut bytes = tiny_artifact().to_bytes().unwrap();
    let length = directory(&bytes)
        .into_iter()
        .find(|(tag, _, _)| *tag == TAG_WEIGHTS)
        .unwrap()
        .2;
    bytes[HEADER_BYTES + 8 + 5 * 16 + 8..HEADER_BYTES + 8 + 5 * 16 + 16]
        .copy_from_slice(&(length + 8).to_le_bytes());
    repair_checksum(&mut bytes);
    let error = expect_error(&bytes);
    assert!(
        matches!(error.kind(), FormatErrorKind::Truncated { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("WGHT section"), "{error}");
}

#[test]
fn a_truncated_section_body_is_reported() {
    // Cut the file inside the last section, and make the header agree with the
    // new length so the *section* check is what fails, not the file check.
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut cut = bytes[..bytes.len() - 8].to_vec();
    let file_bytes = cut.len() as u64;
    cut[24..32].copy_from_slice(&file_bytes.to_le_bytes());
    cut[32..40].copy_from_slice(&(file_bytes - HEADER_BYTES as u64).to_le_bytes());
    repair_checksum(&mut cut);
    let error = expect_error(&cut);
    assert!(
        matches!(error.kind(), FormatErrorKind::Truncated { .. }),
        "{error}"
    );
}

#[test]
fn a_checksum_mismatch_is_rejected() {
    let mut bytes = tiny_artifact().to_bytes().unwrap();
    // Flip one byte of the last section's payload without fixing the checksum.
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    let error = expect_error(&bytes);
    assert!(matches!(
        error.kind(),
        FormatErrorKind::ChecksumMismatch { .. }
    ));
    assert!(error.is_corruption());
    let text = error.to_string();
    assert!(text.contains("payload checksum mismatch"), "{text}");
    assert!(text.contains("crc32 0x"), "{text}");
}

#[test]
fn one_flipped_payload_byte_names_both_checksums() {
    let artifact = tiny_artifact();
    let bytes = artifact.to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let wght = section_offset(&bytes, TAG_WEIGHTS);
    mutated[wght + 40] ^= 0x80;
    let error = expect_error(&mutated);
    match error.kind() {
        FormatErrorKind::ChecksumMismatch { stored, computed } => {
            assert_ne!(stored, computed);
            assert_eq!(
                *stored,
                u32::from_le_bytes(bytes[40..44].try_into().unwrap())
            );
            assert_eq!(
                *computed,
                crc32(&mutated[HEADER_BYTES..]),
                "the reported checksum is the CRC-32 of the payload"
            );
        }
        other => panic!("unexpected {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 3. Content rejections
// ---------------------------------------------------------------------------

#[test]
fn an_unknown_dtype_code_names_the_code_and_the_reader() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let wght = section_offset(&bytes, TAG_WEIGHTS);
    // count, name length, "wte", then the dtype byte.
    mutated[wght + 4 + 4 + 3] = 9;
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    assert!(
        matches!(error.kind(), FormatErrorKind::UnknownDType(_)),
        "{error}"
    );
    let text = error.to_string();
    assert!(text.contains("unknown dtype code 9"), "{text}");
    assert!(text.contains("weight table"), "{text}");
    assert!(text.contains("f32=0"), "{text}");
}

#[test]
fn an_unknown_op_code_names_the_code() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let nodes = section_offset(&bytes, TAG_NODES);
    // count, onnx_index, name length, "matmul", op-name length, "MatMul", code.
    let code = nodes + 4 + 4 + 4 + 6 + 4 + 6;
    mutated[code..code + 2].copy_from_slice(&0x0101u16.to_le_bytes());
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    assert!(matches!(
        error.kind(),
        FormatErrorKind::UnknownOpCode { code: 0x0101 }
    ));
    assert!(error.to_string().contains("reserved range"), "{error}");
}

#[test]
fn a_declared_count_larger_than_the_file_is_rejected() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let vals = section_offset(&bytes, TAG_VALS);
    // 1000 values need at least 7000 bytes; the section is far smaller.
    mutated[vals..vals + 4].copy_from_slice(&1000u32.to_le_bytes());
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    assert!(
        matches!(error.kind(), FormatErrorKind::Truncated { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("value table"), "{error}");
}

#[test]
fn a_hostile_count_fails_before_it_is_allocated() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let vals = section_offset(&bytes, TAG_VALS);
    // 4.29e9 values would need hundreds of gigabytes if the reader believed the
    // count. Reaching the assertion below proves the count was rejected
    // arithmetically, before any Vec was created.
    mutated[vals..vals + 4].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
    repair_checksum(&mut mutated);
    let started = Instant::now();
    let error = expect_error(&mutated);
    let elapsed = started.elapsed();
    match error.kind() {
        FormatErrorKind::LimitExceeded {
            requested, limit, ..
        } => {
            assert_eq!(*requested, 0xFFFF_FFF0u64);
            assert_eq!(*limit, MAX_VALUES);
        }
        other => panic!("unexpected {other:?}"),
    }
    assert!(
        elapsed.as_millis() < 100,
        "a hostile count must not be attempted: {elapsed:?}"
    );
    assert!(error.to_string().contains("limit of 1048576"), "{error}");
}

#[test]
fn a_hostile_rank_is_rejected() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let vals = section_offset(&bytes, TAG_VALS);
    // First value: name length, name, dtype, shape kind, rank.
    let name_len = u32::from_le_bytes(mutated[vals + 4..vals + 8].try_into().unwrap()) as usize;
    let rank = vals + 4 + 4 + name_len + 1 + 1;
    mutated[rank..rank + 4].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    match error.kind() {
        FormatErrorKind::LimitExceeded {
            requested, limit, ..
        } => {
            assert_eq!(*requested, u64::from(u32::MAX));
            assert_eq!(*limit, MAX_RANK);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn invalid_utf8_in_a_name_is_rejected() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let wght = section_offset(&bytes, TAG_WEIGHTS);
    mutated[wght + 8] = 0xFF; // first byte of the first weight name
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    assert!(
        matches!(error.kind(), FormatErrorKind::Malformed { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("not valid UTF-8"), "{error}");
}

#[test]
fn trailing_payload_bytes_are_rejected() {
    let mut bytes = tiny_artifact().to_bytes().unwrap();
    bytes.extend_from_slice(&[0u8; 8]);
    let file_bytes = bytes.len() as u64;
    bytes[24..32].copy_from_slice(&file_bytes.to_le_bytes());
    bytes[32..40].copy_from_slice(&(file_bytes - HEADER_BYTES as u64).to_le_bytes());
    repair_checksum(&mut bytes);
    let error = expect_error(&bytes);
    assert!(
        matches!(error.kind(), FormatErrorKind::Malformed { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("trailing bytes"), "{error}");
}

#[test]
fn a_payload_edit_with_a_repaired_checksum_is_still_validated() {
    // A hostile writer can recompute the checksum; the structural checks are
    // independent of it. Corrupt an origin kind and fix the CRC: the reader
    // must still reject the value table.
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let vals = section_offset(&bytes, TAG_VALS);
    let name_len = u32::from_le_bytes(mutated[vals + 4..vals + 8].try_into().unwrap()) as usize;
    // name, dtype, shape kind 1, rank 2, two dims of (kind 0 + u64), origin kind.
    let origin = vals + 4 + (4 + name_len) + 1 + 1 + 4 + 2 * (1 + 8);
    mutated[origin] = 9;
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    assert!(
        matches!(error.kind(), FormatErrorKind::Malformed { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("origin kind 9"), "{error}");
}

#[test]
fn an_envelope_above_the_hard_cap_is_rejected() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let env = section_offset(&bytes, TAG_ENV);
    // `seq` is symbol 1: minimum at 16..24, maximum at 24..32.
    mutated[env + 24..env + 32].copy_from_slice(&(DimBound::MAX_EXTENT + 1).to_le_bytes());
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    assert!(
        matches!(error.kind(), FormatErrorKind::Contract { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("hard cap"), "{error}");
}

#[test]
fn an_envelope_that_excludes_the_fixed_heads_is_rejected() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let env = section_offset(&bytes, TAG_ENV);
    // `n_heads` is symbol 4: minimum at 64..72, maximum at 72..80.
    mutated[env + 64..env + 72].copy_from_slice(&8u64.to_le_bytes());
    mutated[env + 72..env + 80].copy_from_slice(&9u64.to_le_bytes());
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    assert!(
        matches!(error.kind(), FormatErrorKind::Inconsistent { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("fixed n_heads is 2"), "{error}");
}

#[test]
fn a_geometry_that_contradicts_its_weights_is_rejected() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let geom = section_offset(&bytes, TAG_GEOM);
    // Vocabulary size is the seventh u32.
    mutated[geom + 24..geom + 28].copy_from_slice(&7u32.to_le_bytes());
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    assert!(
        matches!(error.kind(), FormatErrorKind::Inconsistent { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("token embedding"), "{error}");
}

#[test]
fn a_layer_count_beyond_the_kv_bound_is_rejected() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let geom = section_offset(&bytes, TAG_GEOM);
    mutated[geom..geom + 4].copy_from_slice(&((MAX_KV_LAYERS as u32) + 1).to_le_bytes());
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    match error.kind() {
        FormatErrorKind::LimitExceeded { limit, .. } => assert_eq!(*limit, MAX_KV_LAYERS),
        // A file with one KV interface and a larger layer count is also fine to
        // reject as inconsistent; both are correct, so accept either.
        FormatErrorKind::Inconsistent { .. } => {}
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn every_single_byte_change_is_detected() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    assert!(bytes.len() > 128);
    let mut by_checksum = 0;
    let mut by_header_field = 0;
    for position in 0..bytes.len() {
        let mut mutated = bytes.clone();
        mutated[position] ^= 0xFF;
        // Never silently accepted: every byte of the file is covered either by a
        // header field check or by the payload checksum.
        let error = expect_error(&mutated);
        assert!(!error.to_string().is_empty(), "byte {position}");
        if matches!(error.kind(), FormatErrorKind::ChecksumMismatch { .. }) {
            by_checksum += 1;
        } else {
            by_header_field += 1;
        }
    }
    // The payload is the larger part of a tiny artifact, and the header has
    // fields that are individually validated.
    assert!(by_checksum > 0, "no byte was caught by the checksum");
    assert!(by_header_field > 0, "no byte was caught by a field check");
    assert_eq!(by_checksum + by_header_field, bytes.len());
}

#[test]
fn a_weight_shape_that_overflows_is_rejected() {
    // The weight table is decoded before `WeightTensor::new` checks it, so this
    // exercises the checked element-count arithmetic on untrusted extents.
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    let wght = section_offset(&bytes, TAG_WEIGHTS);
    // count, name length, "wte", dtype, rank, then two u64 extents.
    let dims = wght + 4 + 4 + 3 + 1 + 4;
    mutated[dims..dims + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    mutated[dims + 8..dims + 16].copy_from_slice(&u64::MAX.to_le_bytes());
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    assert!(
        matches!(error.kind(), FormatErrorKind::Contract { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("overflow"), "{error}");

    // A declared payload larger than the section is a truncation, not an
    // allocation.
    let mut mutated = bytes.clone();
    let payload_len = wght + 4 + 4 + 3 + 1 + 4 + 16 + 1;
    mutated[payload_len..payload_len + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    assert!(
        matches!(error.kind(), FormatErrorKind::Truncated { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("payload"), "{error}");
}

// ---------------------------------------------------------------------------
// 4. No session state
// ---------------------------------------------------------------------------

#[test]
fn the_artifact_declares_exactly_the_six_structural_sections() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let entries = directory(&bytes);
    let tags: Vec<u32> = entries.iter().map(|(tag, _, _)| *tag).collect();
    assert_eq!(
        tags,
        vec![
            TAG_META,
            TAG_GEOM,
            TAG_ENV,
            TAG_VALS,
            TAG_NODES,
            TAG_WEIGHTS
        ],
        "the section inventory is closed: there is no cache, extent or counter section"
    );
    assert_eq!(
        u32::from_le_bytes(bytes[HEADER_BYTES..HEADER_BYTES + 4].try_into().unwrap()),
        6
    );
    // Every section is fully consumed by its decoder, so no section can carry a
    // trailing field of its own (see `a_section_with_extra_fields_is_rejected`).
    for (tag, offset, length) in entries {
        assert!(offset >= HEADER_BYTES);
        assert!(offset as u64 + length <= bytes.len() as u64, "tag {tag}");
    }
}

#[test]
fn an_unknown_section_tag_is_rejected() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    for tag in [0x4B56u32, 7, 0, u32::MAX] {
        let mut mutated = bytes.clone();
        mutated[HEADER_BYTES + 8..HEADER_BYTES + 12].copy_from_slice(&tag.to_le_bytes());
        repair_checksum(&mut mutated);
        let error = expect_error(&mutated);
        assert!(
            matches!(error.kind(), FormatErrorKind::Malformed { .. }),
            "tag {tag}: {error}"
        );
        assert!(error.to_string().contains("expected 1 (META)"), "{error}");
    }
}

#[test]
fn an_extra_section_entry_is_rejected() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let mut mutated = bytes.clone();
    mutated[HEADER_BYTES..HEADER_BYTES + 4].copy_from_slice(&7u32.to_le_bytes());
    repair_checksum(&mut mutated);
    let error = expect_error(&mutated);
    assert!(
        matches!(error.kind(), FormatErrorKind::LimitExceeded { .. }),
        "{error}"
    );
}

#[test]
fn a_section_with_extra_fields_is_rejected() {
    // Extend the last section over its own alignment padding and eight appended
    // bytes: the weights decoder must consume exactly the declared bytes, so a
    // field a writer invented (a live extent, a step counter, ...) cannot be
    // smuggled into a section.
    let mut bytes = tiny_artifact().to_bytes().unwrap();
    let last_entry = HEADER_BYTES + 8 + 5 * 16;
    let length = u64::from_le_bytes(bytes[last_entry + 8..last_entry + 16].try_into().unwrap());
    let padding = (8 - (length as usize % 8)) % 8;
    bytes[last_entry + 8..last_entry + 16]
        .copy_from_slice(&(length + padding as u64 + 8).to_le_bytes());
    bytes.extend_from_slice(&[0xAA; 8]);
    let file_bytes = bytes.len() as u64;
    bytes[24..32].copy_from_slice(&file_bytes.to_le_bytes());
    bytes[32..40].copy_from_slice(&(file_bytes - HEADER_BYTES as u64).to_le_bytes());
    repair_checksum(&mut bytes);
    let error = expect_error(&bytes);
    assert!(
        matches!(error.kind(), FormatErrorKind::Malformed { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("trailing bytes"), "{error}");
    assert!(error.to_string().contains("WGHT section"), "{error}");
}

#[test]
fn a_session_adds_no_state_to_the_artifact() {
    let artifact = tiny_artifact();
    let before = artifact.to_bytes().unwrap();

    let config = artifact.geometry().kv_cache_config(4, 1);
    assert_eq!(config.capacity_tokens, 4);
    assert_eq!(config.batch, 1);
    let plan = ExecutionPlan::new(Vec::new(), 1).expect("an empty plan over one slot");
    let model = PreparedModel::new(
        plan,
        artifact
            .to_model_weights()
            .expect("the artifact weight table becomes tensors"),
        DeviceId::Cpu,
        artifact.envelope(),
        Some(config),
    )
    .expect("the geometry's cache configuration is valid against the envelope");

    let first = Session::new(&model).expect("session one allocates its own cache");
    let second = Session::new(&model).expect("session two allocates its own cache");
    assert_eq!(first.step(), 0, "a fresh session has no steps");
    assert_eq!(first.cache_len(), 0, "a fresh session has no cached tokens");
    assert_eq!(first.caches().len(), 1);
    assert_eq!(first.caches()[0].capacity(), 4);
    assert_eq!(second.caches()[0].capacity(), 4);
    assert_ne!(
        first.caches()[0].keys().storage().as_bytes().as_ptr(),
        second.caches()[0].keys().storage().as_bytes().as_ptr(),
        "two sessions must not share cache storage"
    );

    // The artifact is immutable data: creating sessions cannot change it, and a
    // different allocation choice (a bigger cache, another batch) cannot reach
    // the file because capacity and batch are session inputs, not model fields.
    assert_eq!(
        artifact.to_bytes().unwrap(),
        before,
        "sessions must not mutate the artifact"
    );
    let other = artifact.geometry().kv_cache_config(64, 1);
    assert_eq!(other.capacity_tokens, 64);
    assert_eq!(
        artifact.to_bytes().unwrap(),
        before,
        "the cache capacity is not part of the artifact"
    );
}

// ---------------------------------------------------------------------------
// 5. 2.x policy
// ---------------------------------------------------------------------------

#[test]
fn legacy_fnnc_bytes_are_rejected_at_the_magic() {
    // A 2.x `.fnnc` is raw `bincode`: a `u64` little-endian length followed by
    // the payload, with no magic and no version. This is that shape.
    let mut fnnc = Vec::new();
    fnnc.extend_from_slice(&64u64.to_le_bytes());
    fnnc.extend(std::iter::repeat_n(0x5Au8, 128));
    let error = expect_error(&fnnc);
    assert!(matches!(error.kind(), FormatErrorKind::BadMagic { .. }));
    assert!(error.is_corruption());
    assert!(
        error.to_string().contains("not a FastNN 3 prepared model"),
        "{error}"
    );

    // A 2.x artifact can never be *re-exported* either: there is no API that
    // accepts one. The only way to obtain an artifact is from an import or from
    // a version-1 artifact, which is what this whole test file uses.
    let bytes = tiny_artifact().to_bytes().unwrap();
    assert_eq!(&bytes[..8], &MAGIC);
    assert_eq!(
        u16::from_le_bytes(bytes[8..10].try_into().unwrap()),
        FORMAT_VERSION
    );
}

// ---------------------------------------------------------------------------
// 6. Alignment and the file API
// ---------------------------------------------------------------------------

#[test]
fn weight_payloads_are_eight_byte_aligned_and_zero_padded() {
    let bytes = tiny_artifact().to_bytes().unwrap();
    let wght = section_offset(&bytes, TAG_WEIGHTS);
    // Walk the weight section exactly as documented.
    let count = u32::from_le_bytes(bytes[wght..wght + 4].try_into().unwrap());
    assert_eq!(count, 2);
    let mut position = wght + 4;
    let mut padding_positions = Vec::new();
    for _ in 0..count {
        let name_len =
            u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap()) as usize;
        position += 4 + name_len;
        position += 1; // dtype
        let rank = u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap()) as usize;
        position += 4;
        position += 8 * rank;
        position += 1; // source
        let payload_len =
            u64::from_le_bytes(bytes[position..position + 8].try_into().unwrap()) as usize;
        position += 8;
        let local = position - wght;
        let padding = (8 - (local % 8)) % 8;
        assert!(padding < 8);
        assert_eq!(
            (position + padding) % 8,
            0,
            "weight payload must start on an eight-byte boundary relative to the file"
        );
        assert_eq!(
            &bytes[position..position + padding],
            vec![0u8; padding].as_slice(),
            "alignment padding must be zeros"
        );
        padding_positions.extend(position..position + padding);
        position += padding + payload_len;
    }
    // The decoder accepts the file we just walked, and rejects non-zero padding.
    PreparedArtifact::from_bytes(&bytes).unwrap();
    assert!(
        !padding_positions.is_empty(),
        "the synthetic weights exercise the padding rule"
    );
    let mut dirty = bytes.clone();
    dirty[padding_positions[0]] = 1;
    repair_checksum(&mut dirty);
    let error = expect_error(&dirty);
    assert!(
        matches!(error.kind(), FormatErrorKind::Malformed { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("padding"), "{error}");
}

#[test]
fn a_file_round_trip_is_byte_identical() {
    let artifact = tiny_artifact();
    let path = std::env::temp_dir().join(format!(
        "fastnn-fn3-format-{}-{}.fn3",
        std::process::id(),
        "round-trip"
    ));
    artifact.write_file(&path).expect("the artifact is written");
    let bytes = std::fs::read(&path).expect("the file exists");
    let loaded = PreparedArtifact::read_file(&path).expect("the artifact is read");
    let _ = std::fs::remove_file(&path);
    assert_eq!(loaded, artifact);
    assert_eq!(loaded.to_bytes().unwrap(), bytes);
}

#[test]
fn a_missing_file_names_the_path() {
    let error = PreparedArtifact::read_file("/nonexistent/model.fn3").unwrap_err();
    assert!(
        matches!(error.kind(), FormatErrorKind::Io { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("model.fn3"), "{error}");
}

#[test]
fn the_public_surface_names_its_seams() {
    // Compile-time documentation of the seam #56 and #62 consume.
    fn _graph(_: &PreparedGraph) {}
    fn _weights(_: &WeightSet) {}
    fn _geometry(_: &DecoderGeometry) {}
    fn _envelope(_: ShapeEnvelope) {}
    fn _weights_host(_: ModelWeights) {}
    let artifact = tiny_artifact();
    _graph(artifact.graph());
    _weights(artifact.weights());
    _geometry(artifact.geometry());
    _envelope(artifact.envelope());
    _weights_host(artifact.to_model_weights().unwrap());
    assert_eq!(artifact.metadata().producer, "pytorch");
    assert_eq!(artifact.graph().lookup("logits"), Some(ValueId::new(4)));
}
