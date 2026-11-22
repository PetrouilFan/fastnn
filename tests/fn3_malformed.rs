//! FN3-27 (#73): malformed-input regressions, an allocation bound and the
//! committed byte corpus.
//!
//! # What this file is for
//!
//! `tests/fn3_import.rs` and `tests/fn3_model_format.rs` prove that *semantically*
//! malformed models are refused by name. This file proves the property that
//! matters before any of that: for arbitrary bytes, both untrusted readers either
//! return a typed error or accept the input, and they never panic, never read out
//! of bounds and never allocate from a declared count.
//!
//! Three independent mechanisms:
//!
//! 1. **A committed corpus** (`tests/corpus/fn3/`, bytes on disk) driven by
//!    `MANIFEST.tsv`, which maps every file to the error class it must produce.
//!    The manifest is compared against the generator below, so adding a case
//!    without committing its bytes, or changing the bytes without updating the
//!    manifest, fails.
//! 2. **Deterministic sweeps**: every single-byte flip and every truncation
//!    prefix of a small synthetic model and of a minimal artifact. Exhaustive for
//!    one-byte changes, which is the class of corruption a bad download or a
//!    flipped bit actually produces.
//! 3. **A counting global allocator**: the refusal of a declared multi-gigabyte
//!    count is asserted to happen with no large allocation, not merely to happen.
//!
//! # Regenerating the corpus
//!
//! ```text
//! FASTNN_WRITE_CORPUS=1 cargo test --test fn3_malformed
//! ```
//!
//! writes every generated case to `tests/corpus/fn3/` and the manifest beside
//! it. Review the diff: the corpus is deliberately tiny (each file is under a
//! kilobyte) because its job is to name error classes, not to be a fuzz corpus.
//! The fuzz targets in `fuzz/fuzz_targets/` cover the unbounded search.
//!
//! # Tooling
//!
//! No `unsafe` exists in `src/fn3/import/` or `src/fn3/model/format/`; the one
//! `unsafe` block in this repository's fn3 tree is the allocator below, which is
//! test-only and forwards to `System`. `cargo +nightly miri test --test
//! fn3_malformed` covers the same paths without the allocator (Miri does not run
//! a global allocator's arithmetic meaningfully) — see
//! `docs/reports/fn3-27-hardening.md`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use fastnn::inference::backend::OpParams;
use fastnn::inference::dtype::DType;
use fastnn::inference::import::{
    import_onnx_model, DecoderGeometry, Dim, GraphNode, KvInterface, PositionEmbedding, ValueId,
    ValueInfo, ValueOrigin, ValueShape,
};
use fastnn::inference::model::format::{
    ArtifactMetadata, FormatErrorKind, PreparedArtifact, PreparedGraph, HEADER_BYTES, MAGIC,
    MAX_VALUES,
};
use fastnn::inference::model::{WeightId, WeightSet, WeightSource, WeightTensor};
use fastnn::inference::ops::OpId;
use fastnn::inference::shape::{Shape, ShapeEnvelope, Symbol};

// ---------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------

/// Tracks the largest single allocation and the running total.
///
/// Only this test binary installs it, so it cannot perturb another suite.
struct Counting;

static LARGEST: AtomicUsize = AtomicUsize::new(0);
static TOTAL: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to `System` with the caller's own `Layout` and
// pointer, exactly as `System` requires. The only added work is two atomic
// counters, which cannot alias, free or move the allocation. `alloc_zeroed` and
// `realloc` forward to their `System` counterparts rather than reimplementing
// them, so the returned pointer has the same provenance and alignment the
// `GlobalAlloc` contract requires. Nothing here dereferences a pointer.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        TOTAL.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        TOTAL.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator with the same `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        TOTAL.fetch_add(new_size, Ordering::Relaxed);
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Serialises the allocation-sensitive tests: the counters are process-wide.
static SERIAL: Mutex<()> = Mutex::new(());

struct Measured {
    largest: usize,
    total: usize,
}

/// Reset the counters and run `body`, returning what it allocated.
fn measure<T>(body: impl FnOnce() -> T) -> (T, Measured) {
    LARGEST.store(0, Ordering::Relaxed);
    TOTAL.store(0, Ordering::Relaxed);
    let value = body();
    (
        value,
        Measured {
            largest: LARGEST.load(Ordering::Relaxed),
            total: TOTAL.load(Ordering::Relaxed),
        },
    )
}

// ---------------------------------------------------------------------------
// Error classification
// ---------------------------------------------------------------------------

/// `VariantName` from a `Debug` rendering such as `Truncated { what: ... }`.
fn variant_name(debug: &str) -> &str {
    debug
        .split(['(', ' ', '{'])
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("Unknown")
}

/// The error class an ONNX import produces, as the manifest spells it.
fn classify_onnx(bytes: &[u8]) -> String {
    match import_onnx_model(bytes) {
        Ok(_) => "accepted".to_string(),
        Err(error) => match error.proto_error() {
            Some(proto) => format!("Proto.{}", variant_name(&format!("{proto:?}"))),
            None => variant_name(&format!("{:?}", error.problems()[0].kind)).to_string(),
        },
    }
}

/// The error class an artifact load produces, as the manifest spells it.
fn classify_artifact(bytes: &[u8]) -> String {
    match PreparedArtifact::from_bytes(bytes) {
        Ok(_) => "accepted".to_string(),
        Err(error) => variant_name(&format!("{:?}", error.kind())).to_string(),
    }
}

fn classify(reader: Reader, bytes: &[u8]) -> String {
    match reader {
        Reader::Onnx => classify_onnx(bytes),
        Reader::Artifact => classify_artifact(bytes),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reader {
    Onnx,
    Artifact,
}

impl Reader {
    fn name(self) -> &'static str {
        match self {
            Reader::Onnx => "onnx",
            Reader::Artifact => "artifact",
        }
    }

    fn parse(name: &str) -> Reader {
        match name {
            "onnx" => Reader::Onnx,
            "artifact" => Reader::Artifact,
            other => panic!("unknown reader `{other}` in the corpus manifest"),
        }
    }
}

// ---------------------------------------------------------------------------
// A minimal protobuf encoder, enough for malformed ONNX cases
// ---------------------------------------------------------------------------

fn varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7F) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn tag(field: u32, wire: u8, out: &mut Vec<u8>) {
    varint((u64::from(field) << 3) | u64::from(wire), out);
}

fn field_varint(field: u32, value: u64, out: &mut Vec<u8>) {
    tag(field, 0, out);
    varint(value, out);
}

fn field_bytes(field: u32, payload: &[u8], out: &mut Vec<u8>) {
    tag(field, 2, out);
    varint(payload.len() as u64, out);
    out.extend_from_slice(payload);
}

/// A `TensorProto` with raw payload, enough for dims/dtype/name/raw/external.
struct RawTensor {
    name: &'static str,
    dtype: i64,
    dims: Vec<i64>,
    raw: Vec<u8>,
    external: bool,
}

impl RawTensor {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for dim in &self.dims {
            field_varint(1, *dim as u64, &mut out);
        }
        field_varint(2, self.dtype as u64, &mut out);
        field_bytes(8, self.name.as_bytes(), &mut out);
        field_bytes(9, &self.raw, &mut out);
        if self.external {
            field_bytes(13, &[], &mut out);
        }
        out
    }
}

/// A model with an IR version, an opset and one graph.
fn model(ir_version: u64, body: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut graph = Vec::new();
    body(&mut graph);
    let mut out = Vec::new();
    field_varint(1, ir_version, &mut out);
    let mut opset = Vec::new();
    field_varint(2, 14, &mut opset);
    field_bytes(8, &opset, &mut out);
    field_bytes(7, &graph, &mut out);
    out
}

// ---------------------------------------------------------------------------
// A minimal valid artifact, so its bytes can be mutated
// ---------------------------------------------------------------------------

fn tiny_graph() -> PreparedGraph {
    let value = |name: &str, shape: ValueShape, origin: ValueOrigin| ValueInfo {
        name: name.to_string(),
        dtype: DType::F32,
        shape,
        origin,
    };
    PreparedGraph::new(
        vec![
            value(
                "past.key",
                ValueShape::Dims(vec![Dim::Known(1), Dim::Known(2)]),
                ValueOrigin::GraphInput,
            ),
            value(
                "past.value",
                ValueShape::Dims(vec![Dim::Symbol(Symbol::Past)]),
                ValueOrigin::GraphInput,
            ),
            value("present.key", ValueShape::Unknown, ValueOrigin::GraphInput),
            value(
                "present.value",
                ValueShape::Dims(vec![Dim::unknown("past + seq")]),
                ValueOrigin::GraphInput,
            ),
            value(
                "logits",
                ValueShape::Dims(vec![Dim::Known(6)]),
                ValueOrigin::Node(0),
            ),
            value(
                "wpe",
                ValueShape::Dims(vec![Dim::Known(3), Dim::Known(4)]),
                ValueOrigin::Weight(WeightId::new(1)),
            ),
            value(
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
                axes: vec![-1, 0],
                keep_dims: true,
                epsilon: 1e-5,
                ..OpParams::default()
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
            past_key: ValueId::new(0),
            past_value: ValueId::new(1),
            present_key: ValueId::new(2),
            present_value: ValueId::new(3),
        }],
    }
}

fn tiny_artifact() -> Vec<u8> {
    let weights = WeightSet::new(vec![
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
    .expect("names are unique");
    let metadata = ArtifactMetadata {
        ir_version: 7,
        opset: fastnn::inference::import::OpsetImport {
            domain: String::new(),
            version: 14,
        },
        producer: "fastnn-malformed-corpus".to_string(),
        producer_version: "1".to_string(),
        graph_name: "tiny".to_string(),
    };
    PreparedArtifact::from_parts(
        metadata,
        tiny_graph(),
        weights,
        tiny_geometry(),
        ShapeEnvelope::uniform(8).expect("8 is inside the envelope cap"),
    )
    .expect("the synthetic artifact is consistent")
    .to_bytes()
    .expect("the synthetic artifact encodes")
}

/// CRC-32/IEEE over the payload, so an edited artifact fails at the field under
/// test rather than at the checksum.
fn repair_checksum(bytes: &mut [u8]) {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in &bytes[HEADER_BYTES..] {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    let crc = crc ^ 0xFFFF_FFFF;
    bytes[40..44].copy_from_slice(&crc.to_le_bytes());
}

/// File offset of a section's first payload byte, from the directory the format
/// documents (not from a private helper).
fn section_offset(bytes: &[u8], wanted: u32) -> usize {
    let count = u32::from_le_bytes(bytes[HEADER_BYTES..HEADER_BYTES + 4].try_into().unwrap());
    let mut offset = HEADER_BYTES + 8 + count as usize * 16;
    for index in 0..count as usize {
        let at = HEADER_BYTES + 8 + index * 16;
        let found = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let length = u64::from_le_bytes(bytes[at + 8..at + 16].try_into().unwrap());
        if found == wanted {
            return offset;
        }
        offset += length as usize;
        offset += (8 - (length as usize % 8)) % 8;
    }
    panic!("section {wanted} is absent");
}

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------

/// One committed case: a name, the reader it drives, the bytes, and the error
/// class the reader must produce.
struct Case {
    name: &'static str,
    reader: Reader,
    bytes: Vec<u8>,
    expected: &'static str,
}

fn corpus() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut add = |name: &'static str, reader: Reader, bytes: Vec<u8>, expected: &'static str| {
        cases.push(Case {
            name,
            reader,
            bytes,
            expected,
        });
    };

    // --- ONNX: protobuf framing -------------------------------------------
    add("onnx_empty", Reader::Onnx, Vec::new(), "Proto.MissingField");

    let mut truncated_varint = Vec::new();
    field_varint(1, 7, &mut truncated_varint);
    tag(7, 2, &mut truncated_varint); // a graph tag with no length
    add(
        "onnx_truncated_varint",
        Reader::Onnx,
        truncated_varint,
        "Proto.Truncated",
    );

    let mut hostile_length = Vec::new();
    field_varint(1, 7, &mut hostile_length);
    tag(7, 2, &mut hostile_length);
    varint(0xFFFF_FFFF, &mut hostile_length);
    add(
        "onnx_hostile_length",
        Reader::Onnx,
        hostile_length,
        "Proto.Truncated",
    );

    let mut eleven_byte_varint = Vec::new();
    tag(1, 0, &mut eleven_byte_varint);
    eleven_byte_varint.extend_from_slice(&[0xFF; 11]);
    add(
        "onnx_varint_too_long",
        Reader::Onnx,
        eleven_byte_varint,
        "Proto.VarintTooLong",
    );

    let mut group = Vec::new();
    field_varint(1, 7, &mut group);
    tag(30, 3, &mut group);
    add(
        "onnx_group_wire_type",
        Reader::Onnx,
        group,
        "Proto.UnsupportedWireType",
    );

    let mut wrong_wire = Vec::new();
    field_bytes(1, b"seven", &mut wrong_wire);
    add(
        "onnx_wrong_wire_type",
        Reader::Onnx,
        wrong_wire,
        "Proto.WrongWireType",
    );

    // --- ONNX: budgets and tensor metadata --------------------------------
    add(
        "onnx_rank_65",
        Reader::Onnx,
        model(7, |graph| {
            field_bytes(
                5,
                &RawTensor {
                    name: "deep",
                    dtype: 1,
                    dims: vec![1; 65],
                    raw: vec![0; 4],
                    external: false,
                }
                .encode(),
                graph,
            );
        }),
        "Proto.LimitExceeded",
    );

    add(
        "onnx_name_5000_bytes",
        Reader::Onnx,
        model(7, |graph| {
            let mut node = Vec::new();
            field_bytes(3, &vec![b'n'; 5_000], &mut node);
            field_bytes(4, b"Transpose", &mut node);
            field_bytes(1, &node, graph);
        }),
        "Proto.LimitExceeded",
    );

    add(
        "onnx_max_extent",
        Reader::Onnx,
        model(7, |graph| {
            field_bytes(
                5,
                &RawTensor {
                    name: "huge",
                    dtype: 1,
                    dims: vec![i64::MAX],
                    raw: Vec::new(),
                    external: false,
                }
                .encode(),
                graph,
            );
        }),
        "ShapeMismatch",
    );

    add(
        "onnx_negative_extent",
        Reader::Onnx,
        model(7, |graph| {
            field_bytes(
                5,
                &RawTensor {
                    name: "negative",
                    dtype: 1,
                    dims: vec![-1, 4],
                    raw: Vec::new(),
                    external: false,
                }
                .encode(),
                graph,
            );
        }),
        "NegativeExtent",
    );

    add(
        "onnx_external_data",
        Reader::Onnx,
        model(7, |graph| {
            field_bytes(
                5,
                &RawTensor {
                    name: "external",
                    dtype: 1,
                    dims: vec![1],
                    raw: vec![0; 4],
                    external: true,
                }
                .encode(),
                graph,
            );
        }),
        "ExternalTensorData",
    );

    add(
        "onnx_int32_on_float",
        Reader::Onnx,
        model(7, |graph| {
            let mut tensor = Vec::new();
            field_varint(1, 2, &mut tensor);
            field_varint(2, 1, &mut tensor); // FLOAT
            field_bytes(8, b"wrong_kind", &mut tensor);
            let mut packed = Vec::new();
            packed.extend_from_slice(&7i32.to_le_bytes());
            packed.extend_from_slice(&9i32.to_le_bytes());
            field_bytes(5, &packed, &mut tensor); // int32_data
            field_bytes(5, &tensor, graph);
        }),
        "TensorDataKind",
    );

    // --- Artifact: framing -------------------------------------------------
    add("artifact_empty", Reader::Artifact, Vec::new(), "Truncated");

    let mut bad_magic = tiny_artifact();
    bad_magic[0] = b'X';
    add(
        "artifact_bad_magic",
        Reader::Artifact,
        bad_magic,
        "BadMagic",
    );

    let mut truncated_header = tiny_artifact();
    truncated_header.truncate(HEADER_BYTES - 1);
    add(
        "artifact_truncated_header",
        Reader::Artifact,
        truncated_header,
        "Truncated",
    );

    let mut truncated_payload = tiny_artifact();
    truncated_payload.truncate(truncated_payload.len() - 8);
    add(
        "artifact_truncated_payload",
        Reader::Artifact,
        truncated_payload,
        "LengthMismatch",
    );

    let mut flipped = tiny_artifact();
    let last = flipped.len() - 1;
    flipped[last] ^= 0x01;
    add(
        "artifact_flipped_byte",
        Reader::Artifact,
        flipped,
        "ChecksumMismatch",
    );

    let mut future_version = tiny_artifact();
    future_version[8..10].copy_from_slice(&2u16.to_le_bytes());
    add(
        "artifact_future_version",
        Reader::Artifact,
        future_version,
        "UnsupportedVersion",
    );

    let mut reserved = tiny_artifact();
    reserved[16..20].copy_from_slice(&1u32.to_le_bytes());
    add(
        "artifact_reserved_flags",
        Reader::Artifact,
        reserved,
        "ReservedField",
    );

    // --- Artifact: directory and counts ------------------------------------
    let mut out_of_order = tiny_artifact();
    let first = HEADER_BYTES + 8;
    let second = HEADER_BYTES + 8 + 16;
    out_of_order[first..first + 4].copy_from_slice(&2u32.to_le_bytes());
    out_of_order[second..second + 4].copy_from_slice(&1u32.to_le_bytes());
    repair_checksum(&mut out_of_order);
    add(
        "artifact_sections_out_of_order",
        Reader::Artifact,
        out_of_order,
        "Malformed",
    );

    let mut swallowed = tiny_artifact();
    let meta_length = u64::from_le_bytes(
        swallowed[HEADER_BYTES + 16..HEADER_BYTES + 24]
            .try_into()
            .unwrap(),
    );
    let geom_length = u64::from_le_bytes(
        swallowed[HEADER_BYTES + 32..HEADER_BYTES + 40]
            .try_into()
            .unwrap(),
    );
    swallowed[HEADER_BYTES + 16..HEADER_BYTES + 24]
        .copy_from_slice(&(meta_length + geom_length).to_le_bytes());
    repair_checksum(&mut swallowed);
    add(
        "artifact_overlapping_sections",
        Reader::Artifact,
        swallowed,
        "Malformed",
    );

    let mut huge_count = tiny_artifact();
    let vals = section_offset(&huge_count, 4);
    huge_count[vals..vals + 4].copy_from_slice(&(MAX_VALUES as u32).to_le_bytes());
    repair_checksum(&mut huge_count);
    add(
        "artifact_count_at_the_cap",
        Reader::Artifact,
        huge_count,
        "Truncated",
    );

    let mut over_cap = tiny_artifact();
    over_cap[vals..vals + 4].copy_from_slice(&((MAX_VALUES + 1) as u32).to_le_bytes());
    repair_checksum(&mut over_cap);
    add(
        "artifact_count_above_the_cap",
        Reader::Artifact,
        over_cap,
        "LimitExceeded",
    );

    let mut huge_section = tiny_artifact();
    huge_section[HEADER_BYTES + 16..HEADER_BYTES + 24].copy_from_slice(&u64::MAX.to_le_bytes());
    repair_checksum(&mut huge_section);
    add(
        "artifact_section_length_overflow",
        Reader::Artifact,
        huge_section,
        "Truncated",
    );

    // --- Artifact: metadata -----------------------------------------------
    let mut nan_epsilon = tiny_artifact();
    let geom = section_offset(&nan_epsilon, 2);
    let epsilon = geom + 7 * 4 + 1 + 4 + 4;
    nan_epsilon[epsilon] = 1;
    nan_epsilon[epsilon + 1..epsilon + 5].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    repair_checksum(&mut nan_epsilon);
    add(
        "artifact_nan_epsilon",
        Reader::Artifact,
        nan_epsilon,
        "Inconsistent",
    );

    let mut inf_epsilon = tiny_artifact();
    inf_epsilon[epsilon + 1..epsilon + 5].copy_from_slice(&f32::INFINITY.to_bits().to_le_bytes());
    repair_checksum(&mut inf_epsilon);
    add(
        "artifact_infinite_epsilon",
        Reader::Artifact,
        inf_epsilon,
        "Inconsistent",
    );

    let mut max_extent = tiny_artifact();
    let wght = section_offset(&max_extent, 6);
    // WGHT entry 0 is `wte` [6, 4]: count (4), name (4 + 3), dtype (1),
    // rank (4), then the first extent.
    let first_extent = wght + 4 + 4 + 3 + 1 + 4;
    max_extent[first_extent..first_extent + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    repair_checksum(&mut max_extent);
    add(
        "artifact_max_extent",
        Reader::Artifact,
        max_extent,
        "Contract",
    );

    let mut zero_vocab = tiny_artifact();
    zero_vocab[geom + 24..geom + 28].copy_from_slice(&0u32.to_le_bytes());
    repair_checksum(&mut zero_vocab);
    add(
        "artifact_zero_vocabulary",
        Reader::Artifact,
        zero_vocab,
        "Inconsistent",
    );

    cases
}

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/fn3")
}

fn manifest_path() -> PathBuf {
    corpus_dir().join("MANIFEST.tsv")
}

/// Write the corpus and its manifest. Invoked through `FASTNN_WRITE_CORPUS=1`.
fn write_corpus() {
    let dir = corpus_dir();
    std::fs::create_dir_all(&dir).expect("create the corpus directory");
    let cases = corpus();
    let mut manifest = String::from("# name\treader\texpected error class\n");
    for case in &cases {
        std::fs::write(dir.join(case.name), &case.bytes).expect("write a corpus file");
        manifest.push_str(&format!(
            "{}\t{}\t{}\n",
            case.name,
            case.reader.name(),
            case.expected
        ));
    }
    std::fs::write(manifest_path(), manifest).expect("write the manifest");
    println!(
        "[fn3malformed][corpus] wrote {} cases, {} bytes total",
        cases.len(),
        cases.iter().map(|case| case.bytes.len()).sum::<usize>()
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn the_committed_corpus_matches_its_generator() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    if std::env::var_os("FASTNN_WRITE_CORPUS").is_some() {
        write_corpus();
    }
    let text = std::fs::read_to_string(manifest_path()).expect(
        "tests/corpus/fn3/MANIFEST.tsv is committed; regenerate it with FASTNN_WRITE_CORPUS=1",
    );
    let listed: Vec<(String, String, String)> = text
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(|line| {
            let mut fields = line.split('\t');
            (
                fields.next().expect("a name").to_string(),
                fields.next().expect("a reader").to_string(),
                fields.next().expect("an error class").to_string(),
            )
        })
        .collect();
    let generated = corpus();
    assert_eq!(
        listed.len(),
        generated.len(),
        "the manifest and the generator disagree on case count"
    );
    for (case, (name, reader, expected)) in generated.iter().zip(&listed) {
        assert_eq!(case.name, name, "corpus order and names must match");
        assert_eq!(case.reader.name(), reader, "case {} reader", case.name);
        assert_eq!(case.expected, expected, "case {} class", case.name);
        let on_disk = std::fs::read(corpus_dir().join(case.name))
            .unwrap_or_else(|error| panic!("corpus file {}: {error}", case.name));
        assert_eq!(
            on_disk, case.bytes,
            "case {} on disk differs from the generator",
            case.name
        );
    }
}

#[test]
fn every_corpus_case_produces_its_documented_error_class() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let manifest = std::fs::read_to_string(manifest_path()).expect("a committed manifest");
    let mut checked = 0usize;
    let mut total_bytes = 0usize;
    for line in manifest.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut fields = line.split('\t');
        let name = fields.next().expect("a name");
        let reader = Reader::parse(fields.next().expect("a reader"));
        let expected = fields.next().expect("an error class");
        let bytes = std::fs::read(corpus_dir().join(name))
            .unwrap_or_else(|error| panic!("corpus file {name}: {error}"));
        total_bytes += bytes.len();
        let observed = classify(reader, &bytes);
        assert_eq!(
            observed, expected,
            "corpus case {name}: expected {expected}, observed {observed}"
        );
        checked += 1;
    }
    println!(
        "[fn3malformed][corpus] {checked} cases, {total_bytes} bytes, all refused with the \
         documented class"
    );
    assert!(checked >= 20, "the corpus must stay meaningful");
}

#[test]
fn a_declared_multi_gigabyte_onnx_length_allocates_nothing_large() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    // An initializer that claims four gigabytes of `raw_data` in a 20-byte file.
    let bytes = model(7, |graph| {
        let mut tensor = Vec::new();
        field_varint(1, 1, &mut tensor);
        field_varint(2, 1, &mut tensor);
        field_bytes(8, b"w", &mut tensor);
        tag(9, 2, &mut tensor);
        varint(0xFFFF_FFFF, &mut tensor);
        field_bytes(5, &tensor, graph);
    });
    assert!(bytes.len() < 64, "the hostile file must stay tiny");
    let (observed, measured) = measure(|| classify_onnx(&bytes));
    assert_eq!(observed, "Proto.Truncated", "observed {observed}");
    assert!(
        measured.largest < 1024 * 1024,
        "a declared 4 GiB payload caused a {} byte allocation",
        measured.largest
    );
    println!(
        "[fn3malformed][alloc] hostile ONNX length: {} byte file, largest allocation {} bytes, \
         total {} bytes",
        bytes.len(),
        measured.largest,
        measured.total
    );
}

#[test]
fn a_declared_million_entry_artifact_table_allocates_nothing_large() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let mut bytes = tiny_artifact();
    let vals = section_offset(&bytes, 4);
    bytes[vals..vals + 4].copy_from_slice(&(MAX_VALUES as u32).to_le_bytes());
    repair_checksum(&mut bytes);
    let (observed, measured) = measure(|| classify_artifact(&bytes));
    assert_eq!(observed, "Truncated", "observed {observed}");
    assert!(
        measured.largest < 1024 * 1024,
        "a declared {MAX_VALUES}-entry table caused a {} byte allocation",
        measured.largest
    );
    println!(
        "[fn3malformed][alloc] declared {MAX_VALUES} values: {} byte file, largest allocation {} \
         bytes, total {} bytes",
        bytes.len(),
        measured.largest,
        measured.total
    );
}

#[test]
fn a_graph_of_a_million_nodes_is_refused_by_the_node_budget() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    // Each node is two bytes (a tag and a zero length), so a two-megabyte file
    // declares one node over the budget. This is the case the budget exists for:
    // the decoded node is far larger than its encoding.
    let over = fastnn::inference::import::MAX_NODES + 1;
    let mut graph = Vec::with_capacity(over * 2);
    for _ in 0..over {
        tag(1, 2, &mut graph);
        varint(0, &mut graph);
    }
    let mut bytes = Vec::new();
    field_varint(1, 7, &mut bytes);
    field_bytes(7, &graph, &mut bytes);
    let (observed, measured) = measure(|| classify_onnx(&bytes));
    assert_eq!(observed, "Proto.LimitExceeded", "observed {observed}");
    println!(
        "[fn3malformed][alloc] {} node entries in {} bytes refused: largest allocation {} bytes, \
         total {} bytes",
        over,
        bytes.len(),
        measured.largest,
        measured.total
    );
    // The cap is what makes the worst case finite. It is not small: one million
    // decoded nodes cost more than their two-byte encodings. Assert only that a
    // documented ceiling holds, so a regression that removes the cap (and would
    // grow without bound) fails here.
    assert!(
        measured.largest < 512 * 1024 * 1024,
        "the node budget did not bound the allocation: {} bytes",
        measured.largest
    );
}

#[test]
fn every_truncation_and_single_byte_flip_of_a_small_model_is_typed() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    // Small but structurally complete: an initializer, a graph input and output.
    let bytes = model(7, |graph| {
        field_bytes(
            5,
            &RawTensor {
                name: "w",
                dtype: 1,
                dims: vec![2],
                raw: vec![0; 8],
                external: false,
            }
            .encode(),
            graph,
        );
        let mut input = Vec::new();
        field_bytes(1, b"x", &mut input);
        let mut tensor_type = Vec::new();
        field_varint(1, 1, &mut tensor_type);
        let mut shape = Vec::new();
        let mut dimension = Vec::new();
        field_varint(1, 2, &mut dimension);
        field_bytes(1, &dimension, &mut shape);
        field_bytes(2, &shape, &mut tensor_type);
        let mut ty = Vec::new();
        field_bytes(1, &tensor_type, &mut ty);
        field_bytes(2, &ty, &mut input);
        field_bytes(11, &input, graph);
    });
    for length in 0..bytes.len() {
        assert!(
            import_onnx_model(&bytes[..length]).is_err(),
            "prefix of {length} bytes was accepted"
        );
    }
    let ((flips, refused), measured) = measure(|| {
        let mut flips = 0usize;
        let mut refused = 0usize;
        for offset in 0..bytes.len() {
            for mask in [0x01u8, 0x80, 0xFF] {
                let mut mutated = bytes.clone();
                mutated[offset] ^= mask;
                flips += 1;
                match import_onnx_model(&mutated) {
                    Ok(_) => {}
                    Err(error) => {
                        assert!(!error.is_empty());
                        refused += 1;
                    }
                }
            }
        }
        (flips, refused)
    });
    println!(
        "[fn3malformed][sweep] {} byte model: {} prefixes, {flips} flips, {refused} refused, \
         largest allocation {} bytes (total {})",
        bytes.len(),
        bytes.len(),
        measured.largest,
        measured.total
    );
}

#[test]
fn every_truncation_and_single_byte_flip_of_an_artifact_is_typed() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let bytes = tiny_artifact();
    let mut accepted = 0usize;
    for length in 0..bytes.len() {
        match PreparedArtifact::from_bytes(&bytes[..length]) {
            Ok(_) => panic!("a prefix of {length} bytes decoded"),
            Err(error) => {
                assert!(!error.to_string().is_empty());
            }
        }
    }
    for offset in 0..bytes.len() {
        for mask in [0x01u8, 0x80, 0xFF] {
            let mut mutated = bytes.clone();
            mutated[offset] ^= mask;
            match PreparedArtifact::from_bytes(&mutated) {
                Ok(_) => accepted += 1,
                Err(error) => {
                    let kind: &FormatErrorKind = error.kind();
                    assert!(
                        !matches!(kind, FormatErrorKind::Io { .. }),
                        "a byte flip produced an I/O error"
                    );
                }
            }
        }
    }
    // Every byte is covered by the header fields, a section length or the CRC,
    // so no single-byte flip may produce a different valid artifact.
    assert_eq!(accepted, 0, "a single-byte flip was accepted");
    let _ = MAGIC;
}

#[test]
fn the_corpus_stays_small_enough_to_commit() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let mut bytes = 0usize;
    let mut files = 0usize;
    for entry in std::fs::read_dir(corpus_dir()).expect("the corpus directory") {
        let entry = entry.expect("a directory entry");
        if entry.path().is_file() {
            files += 1;
            bytes += entry.metadata().expect("metadata").len() as usize;
        }
    }
    println!("[fn3malformed][corpus] {files} files, {bytes} bytes on disk");
    assert!(files >= 20, "the corpus lost files");
    assert!(
        bytes < 64 * 1024,
        "the corpus grew to {bytes} bytes; keep the committed corpus small"
    );
}
