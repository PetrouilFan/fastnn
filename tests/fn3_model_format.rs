//! FN3-09 follow-up: the version-2 artifact's structural contract.
//!
//! The 47 version-1 tests in this file asserted the decoder-shaped
//! `FNN3PREP` format (geometry, envelope, the three-attribute node table and a
//! CRC-32 payload). That format was replaced, so those cases were deleted rather
//! than adapted: a test that keeps passing while the format it checks no longer
//! exists is worse than no test. The v2 behaviour is covered here (framing,
//! section hashes, alignment, budgets, inspect and determinism) and end to end in
//! `tests/fn3_artifact_v2.rs`.

use std::path::PathBuf;

use fastnn::inference::backend::ScalarOnly;
use fastnn::inference::digest::sha256_hex;
use fastnn::inference::import::OpsetImport;
use fastnn::inference::ir::{DimExpr, Graph, SourceLoc, TensorType};
use fastnn::inference::model::format::{
    inspect_file, parse_directory, Artifact, ArtifactHeader, ArtifactMetadata, ArtifactPlan,
    FormatErrorKind, Precision, PreparedArtifact, DIRECTORY_BYTES, FORMAT_VERSION, HEADER_BYTES,
    LEGACY_MAGIC, MAGIC, MAX_TARGETS, SECTION_COUNT,
};
use fastnn::inference::model::{WeightSet, WeightSource, WeightTensor};
use fastnn::inference::runtime::{Plan, PlanOptions, Session, Threading};
use fastnn::inference::shape::Shape;
use fastnn::inference::{DType, OpId, OpParams};

struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "fastnn-fn3-model-format-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch");
        Scratch { path }
    }

    fn file(&self) -> PathBuf {
        self.path.join("model.fnn3")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn tiny() -> (PreparedArtifact, Vec<u8>) {
    let weights = WeightSet::new(vec![
        WeightTensor::new(
            "w",
            DType::F32,
            Shape::from_dims([2]),
            vec![0u8; 8],
            WeightSource::Initializer,
        )
        .expect("weight"),
        WeightTensor::new(
            "bias",
            DType::F32,
            Shape::from_dims([1]),
            vec![0u8; 4],
            WeightSource::Initializer,
        )
        .expect("bias"),
    ])
    .expect("weight set");
    let mut graph = Graph::new();
    let x = graph
        .add_value(
            Some("x"),
            TensorType::new(DType::F32, [DimExpr::constant(2)]),
        )
        .expect("x");
    let w = graph
        .add_value(
            Some("w"),
            TensorType::new(DType::F32, [DimExpr::constant(2)]),
        )
        .expect("w");
    let bias = graph
        .add_value(
            Some("bias"),
            TensorType::new(DType::F32, [DimExpr::constant(1)]),
        )
        .expect("bias");
    let sum = graph
        .add_value(
            Some("sum"),
            TensorType::new(DType::F32, [DimExpr::constant(2)]),
        )
        .expect("sum");
    let y = graph
        .add_value(
            Some("y"),
            TensorType::new(DType::F32, [DimExpr::constant(2)]),
        )
        .expect("y");
    graph
        .add_node(
            OpId::Mul,
            vec![x, w],
            vec![sum],
            OpParams::new(),
            SourceLoc::new(0, "Mul"),
        )
        .expect("Mul");
    graph
        .add_node(
            OpId::Add,
            vec![sum, bias],
            vec![y],
            OpParams::new(),
            SourceLoc::new(1, "Add").with_output("y"),
        )
        .expect("Add");
    let plan = Plan::from_graph_with_options(&graph, &weights, &ScalarOnly, PlanOptions::new())
        .expect("plan");
    let metadata = ArtifactMetadata {
        ir_version: 7,
        opset: OpsetImport {
            domain: String::new(),
            version: 13,
        },
        producer: "fastnn-tests".to_string(),
        producer_version: "0".to_string(),
        graph_name: "tiny".to_string(),
    };
    let prepared = PreparedArtifact::from_parts(
        metadata,
        [0x11; 32],
        graph,
        vec!["x".to_string()],
        vec!["y".to_string()],
        weights,
        vec![ArtifactPlan::new("cpu", Precision::F32, plan)],
    )
    .expect("artifact");
    let bytes = prepared.to_bytes().expect("encode");
    (prepared, bytes)
}

fn sections(bytes: &[u8]) -> Vec<(u32, u64, u64)> {
    let count =
        u32::from_le_bytes(bytes[HEADER_BYTES..HEADER_BYTES + 4].try_into().unwrap()) as usize;
    assert_eq!(count, SECTION_COUNT);
    (0..count)
        .map(|index| {
            let start = HEADER_BYTES + 8 + index * 24;
            (
                u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()),
                u64::from_le_bytes(bytes[start + 8..start + 16].try_into().unwrap()),
                u64::from_le_bytes(bytes[start + 16..start + 24].try_into().unwrap()),
            )
        })
        .collect()
}

#[test]
fn the_artifact_declares_five_sections_at_64_byte_boundaries() {
    let (_prepared, bytes) = tiny();
    let header = ArtifactHeader::parse(&bytes, bytes.len() as u64).expect("header");
    assert_eq!(header.version, FORMAT_VERSION);
    assert_eq!(header.section_count, SECTION_COUNT as u32);
    assert_eq!(header.target_count, 1);
    assert_eq!(header.file_bytes, bytes.len() as u64);
    assert_eq!(header.directory_offset, HEADER_BYTES as u64);
    assert_eq!(header.directory_bytes, DIRECTORY_BYTES as u64);
    let ranges = parse_directory(
        &bytes[HEADER_BYTES..HEADER_BYTES + DIRECTORY_BYTES],
        &header,
    )
    .expect("directory");
    assert_eq!(
        ranges.map(|range| range.name()),
        ["GRAPH", "PLANS", "WEIGHTS", "QUANT", "HASHES"]
    );
    let mut cursor = (HEADER_BYTES + DIRECTORY_BYTES) as u64;
    for range in ranges {
        assert_eq!(range.offset, cursor);
        assert_eq!(range.offset % 64, 0);
        cursor = (range.offset + range.length).div_ceil(64) * 64;
    }
    assert_eq!(cursor, bytes.len() as u64);
    assert_eq!(&bytes[..8], b"FNN3ART2");
}

#[test]
fn every_section_hash_matches_the_stored_digest() {
    let (_prepared, bytes) = tiny();
    let ranges = sections(&bytes);
    let hash_section = ranges[SECTION_COUNT - 1];
    let hash_bytes = &bytes[hash_section.1 as usize..(hash_section.1 + hash_section.2) as usize];
    let count = u32::from_le_bytes(hash_bytes[..4].try_into().unwrap());
    assert_eq!(count, 5, "HEADER, GRAPH, PLANS, WEIGHTS, QUANT");
    let mut stored = Vec::new();
    for index in 0..5 {
        let start = 8 + index * 40;
        assert_eq!(
            u32::from_le_bytes(hash_bytes[start..start + 4].try_into().unwrap()),
            index as u32
        );
        stored.push(&hash_bytes[start + 8..start + 40]);
    }
    let header_hash = sha256_hex(&bytes[..HEADER_BYTES]);
    assert_eq!(hex(stored[0]), header_hash);
    for index in 0..4 {
        let (_, offset, length) = ranges[index];
        let computed = sha256_hex(&bytes[offset as usize..(offset + length) as usize]);
        assert_eq!(hex(stored[index + 1]), computed, "section {}", index + 1);
    }
    // WEIGHTS is the third section; its hash covers the table and the payload.
    let weights = ranges[2];
    assert_eq!(
        hex(stored[3]),
        sha256_hex(&bytes[weights.1 as usize..(weights.1 + weights.2) as usize])
    );
    let quant = ranges[3];
    assert_eq!(
        hex(stored[4]),
        sha256_hex(&bytes[quant.1 as usize..(quant.1 + quant.2) as usize])
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn writing_twice_is_byte_identical() {
    let (prepared, first) = tiny();
    let second = prepared.to_bytes().expect("second encode");
    assert_eq!(first, second);
    let decoded = Artifact::from_bytes(first.clone()).expect("decode");
    assert_eq!(decoded.graph().node_count(), prepared.graph().node_count());
    assert_eq!(decoded.weights().len(), prepared.weights().len());
    assert_eq!(decoded.default_target(), "cpu");
}

#[test]
fn inspect_never_reads_the_weight_payload() {
    let (_prepared, bytes) = tiny();
    let scratch = Scratch::new("inspect");
    let path = scratch.file();
    std::fs::write(&path, &bytes).expect("write");
    let report = inspect_file(&path).expect("inspect");
    assert_eq!(report.mapped_bytes, 0);
    let weight_payload_bytes = report.weight_bytes;
    assert!(weight_payload_bytes >= 12);
    assert!(
        report.bytes_read + weight_payload_bytes < report.file_bytes,
        "inspect read {} of {} bytes; the payload is {} bytes",
        report.bytes_read,
        report.file_bytes,
        weight_payload_bytes
    );
    assert_eq!(report.sections[3].name, "WEIGHTS");
    assert!(
        !report.sections[3].verified,
        "the weight payload hash is stored, not re-read"
    );
    assert!(report.sections[0].verified);
    assert!(report.plans[0].segments >= 1);
}

#[test]
fn a_hostile_target_count_is_rejected_by_the_header() {
    let (_prepared, bytes) = tiny();
    let mut mutated = bytes.clone();
    mutated[16..20].copy_from_slice(&(MAX_TARGETS as u32 + 1).to_le_bytes());
    let error = ArtifactHeader::parse(&mutated, mutated.len() as u64).unwrap_err();
    assert!(matches!(
        error.kind(),
        FormatErrorKind::LimitExceeded { .. }
    ));
}

#[test]
fn a_truncated_directory_is_typed() {
    let (_prepared, bytes) = tiny();
    let header = ArtifactHeader::parse(&bytes, bytes.len() as u64).expect("header");
    let short = &bytes[HEADER_BYTES..HEADER_BYTES + DIRECTORY_BYTES - 1];
    let error = parse_directory(short, &header).unwrap_err();
    assert!(matches!(error.kind(), FormatErrorKind::Truncated { .. }));
    assert!(error.to_string().contains("directory"), "{error}");
}

#[test]
fn a_section_in_the_wrong_order_is_rejected() {
    let (_prepared, bytes) = tiny();
    let header = ArtifactHeader::parse(&bytes, bytes.len() as u64).expect("header");
    let mut directory = bytes[HEADER_BYTES..HEADER_BYTES + DIRECTORY_BYTES].to_vec();
    // Swap GRAPH's tag with PLANS'.
    let graph_tag: [u8; 4] = directory[8..12].try_into().unwrap();
    let plans_tag: [u8; 4] = directory[32..36].try_into().unwrap();
    directory[8..12].copy_from_slice(&plans_tag);
    directory[32..36].copy_from_slice(&graph_tag);
    let error = parse_directory(&directory, &header).unwrap_err();
    match error.kind() {
        FormatErrorKind::MalformedAt { section, .. } => assert_eq!(*section, "HEADER"),
        other => panic!("expected MalformedAt, found {other:?}"),
    }
}

#[test]
fn the_weight_payload_is_64_byte_aligned_and_zero_padded() {
    let (_prepared, bytes) = tiny();
    let header = ArtifactHeader::parse(&bytes, bytes.len() as u64).expect("header");
    assert_eq!(header.weight_payload_offset % 64, 0);
    assert_eq!(header.weights_offset % 64, 0);
    assert!(header.weight_payload_offset > header.weights_offset);
    let payload = &bytes[header.weight_payload_offset as usize
        ..(header.weight_payload_offset + header.weight_payload_bytes) as usize];
    // Two 8-byte weights padded to 64 each.
    assert_eq!(payload.len(), 128);
    assert_eq!(payload[8..64], [0u8; 56]);
    assert_eq!(payload[64 + 4..128], [0u8; 60]);
}

#[test]
fn a_future_version_and_the_version_one_magic_are_typed() {
    let (_prepared, bytes) = tiny();
    let mut future = bytes.clone();
    future[8..10].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
    assert!(matches!(
        ArtifactHeader::parse(&future, future.len() as u64)
            .unwrap_err()
            .kind(),
        FormatErrorKind::UnsupportedVersion { .. }
    ));
    let mut old = bytes.clone();
    old[..8].copy_from_slice(&LEGACY_MAGIC);
    old[8..10].copy_from_slice(&1u16.to_le_bytes());
    let error = ArtifactHeader::parse(&old, old.len() as u64).unwrap_err();
    assert!(matches!(
        error.kind(),
        FormatErrorKind::ObsoleteFormat { version: 1, .. }
    ));
    assert_eq!(&MAGIC, b"FNN3ART2");
}

#[test]
fn two_sessions_do_not_change_the_artifact() {
    let (_prepared, bytes) = tiny();
    let scratch = Scratch::new("sessions");
    let path = scratch.file();
    std::fs::write(&path, &bytes).expect("write");
    let artifact = Artifact::open(&path).expect("open");
    let before = sha256_hex(&bytes);
    let first = Session::from_artifact(&artifact, Threading::Fixed(1)).expect("first");
    let second = Session::from_artifact(&artifact, Threading::Fixed(1)).expect("second");
    assert_eq!(first.plan(), second.plan());
    assert_eq!(sha256_hex(&std::fs::read(&path).expect("read")), before);
}

#[test]
fn a_header_file_length_mismatch_is_typed() {
    let (_prepared, bytes) = tiny();
    let error = ArtifactHeader::parse(&bytes, bytes.len() as u64 + 1).unwrap_err();
    assert!(matches!(
        error.kind(),
        FormatErrorKind::LengthMismatch { .. }
    ));
    assert!(error.is_corruption());
}

#[test]
fn the_public_surface_names_its_seams() {
    fn _takes_artifact(_: &Artifact) {}
    fn _takes_prepared(_: &PreparedArtifact) {}
    fn _takes_header(_: &ArtifactHeader) {}
    fn _takes_plan(_: &ArtifactPlan) {}
    assert_eq!(Precision::F32.name(), "f32");
}
