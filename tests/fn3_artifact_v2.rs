//! Format version 2: one artifact, mmap-backed weights and stored plans.
//!
//! # What these tests prove
//!
//! 1. **Round trip.** `yolov8n.onnx` prepares to one artifact, loads with no ONNX
//!    source, runs, and produces bit-identical outputs to an in-memory session
//!    over the same typed graph.
//! 2. **Per-target plans.** `target="all"` writes a fused `cpu` plan and an
//!    unfused `cpu-reference` plan; the file selects the default and
//!    `Session::from_artifact_for` opens the other without re-planning.
//! 3. **mmap sharing.** Two sessions opened on one artifact have the same weight
//!    pointer, and that pointer lies inside the mapping.
//! 4. **Corruption is typed.** Every truncation, a flipped byte in every section,
//!    a bad directory length, a bad directory offset, a future version and a
//!    version-1 file fail with a [`FormatErrorKind`] naming the section and
//!    offset where there is one.
//! 5. **A 200 MB model does not load into 200 MB of RSS.** The measured numbers
//!    are printed; the assertion is that RSS growth stays far below the weight
//!    payload.
//!
//! The pinned YOLO file is not committed. A machine without it prints why and
//! returns for the tests that need it; the synthetic, corruption, sharing and
//! RSS tests always run.

use std::path::PathBuf;
use std::sync::Arc;

use fastnn::inference::backend::ScalarOnly;
use fastnn::inference::digest::sha256_hex;
use fastnn::inference::import::{import_onnx_graph, OpsetImport};
use fastnn::inference::ir::{DimExpr, Graph, SourceLoc, TensorType};
use fastnn::inference::model::format::{
    Artifact, ArtifactMetadata, ArtifactPlan, FormatErrorKind, Precision, PreparedArtifact,
    DIRECTORY_BYTES, FORMAT_VERSION, HEADER_BYTES, LEGACY_MAGIC, MAGIC, SECTION_COUNT,
};
use fastnn::inference::model::{WeightSet, WeightSource, WeightTensor};
use fastnn::inference::runtime::{EngineChoice, Plan, PlanOptions, Session, Threading};
use fastnn::inference::shape::Shape;
use fastnn::inference::{DType, OpId, OpParams};

const YOLOV8N: &str = "/home/petrouil/.hermes/yolov8n.onnx";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ty(dtype: DType, dims: &[u64]) -> TensorType {
    TensorType::new(dtype, dims.iter().copied().map(DimExpr::constant))
}

/// A one-node graph with a large weight and a tiny activation footprint:
/// `Gather(indices[1], w[N,1]) -> y[1,1]`.
fn synthetic(weight_elements: u64, name: &str) -> (PathBuf, tempdir::Scratch) {
    let shape = Shape::from_dims([weight_elements, 1]);
    let bytes = vec![0u8; (weight_elements * 4) as usize];
    let weight = WeightTensor::new("w", DType::F32, shape, bytes, WeightSource::Initializer)
        .expect("weight shape and payload agree");
    let weights = WeightSet::new(vec![weight]).expect("one weight");
    let mut graph = Graph::new();
    let indices = graph
        .add_value(Some("indices"), ty(DType::I64, &[1]))
        .expect("indices value");
    let w = graph
        .add_value(Some("w"), ty(DType::F32, &[weight_elements, 1]))
        .expect("weight value");
    let y = graph
        .add_value(Some("y"), ty(DType::F32, &[1, 1]))
        .expect("output");
    graph
        .add_node(
            OpId::Gather,
            vec![indices, w],
            vec![y],
            OpParams::new().with_axes([0]),
            SourceLoc::new(0, "Gather").with_output("y"),
        )
        .expect("Gather node");
    let plan = Plan::from_graph_with_options(&graph, &weights, &ScalarOnly, PlanOptions::new())
        .expect("synthetic plan");
    let metadata = ArtifactMetadata {
        ir_version: 7,
        opset: OpsetImport {
            domain: String::new(),
            version: 13,
        },
        producer: "fastnn-tests".to_string(),
        producer_version: "0".to_string(),
        graph_name: "synthetic-gather".to_string(),
    };
    let prepared = PreparedArtifact::from_parts(
        metadata,
        [0x5a; 32],
        graph,
        vec!["indices".to_string()],
        vec!["y".to_string()],
        weights,
        vec![ArtifactPlan::new("cpu", Precision::F32, plan)],
    )
    .expect("synthetic artifact");
    let directory = tempdir::Scratch::new(name);
    let path = directory.path().join("model.fnn3");
    prepared
        .write_file(&path)
        .expect("write synthetic artifact");
    drop(prepared);
    (path, directory)
}

/// A small scratch-directory owner, removed on drop.
mod tempdir {
    use std::path::{Path, PathBuf};

    pub struct Scratch {
        path: PathBuf,
    }

    impl Scratch {
        pub fn new(name: &str) -> Scratch {
            let path = std::env::temp_dir().join(format!(
                "fastnn-fn3-artifact-v2-{}-{name}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("scratch directory");
            Scratch { path }
        }

        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

fn yolov8n_bytes() -> Option<Vec<u8>> {
    match std::fs::read(YOLOV8N) {
        Ok(bytes) => Some(bytes),
        Err(_) => {
            eprintln!(
                "{YOLOV8N} is not present; skipping the pinned-artifact tests in \
                 tests/fn3_artifact_v2.rs"
            );
            None
        }
    }
}

fn rss_bytes() -> u64 {
    // Linux reports VmRSS in kB. The tests are gated on Linux for the RSS proof;
    // other platforms get a zero so the assertion is skipped rather than
    // claiming a measurement they did not take.
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Some(value) = rest.split_whitespace().next() {
                if let Ok(kb) = value.parse::<u64>() {
                    return kb * 1024;
                }
            }
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Round trip, no-ONNX load and per-target plans
// ---------------------------------------------------------------------------

#[test]
fn yolov8n_prepares_loads_and_runs_bitwise_from_the_artifact() {
    let Some(source) = yolov8n_bytes() else {
        return;
    };
    let imported = import_onnx_graph(&source).expect("the pinned CNN imports");
    let directory = tempdir::Scratch::new("roundtrip");
    let artifact_path = directory.path().join("yolov8n.fnn3");

    let source_digest = digest(&source);
    let prepared = PreparedArtifact::from_imported(
        &imported,
        &["cpu-reference"],
        Precision::F32,
        source_digest,
    )
    .expect("prepare");
    let bytes = prepared.to_bytes().expect("encode");
    assert_eq!(
        prepared.to_bytes().expect("encode again"),
        bytes,
        "the encoding must be deterministic"
    );
    std::fs::write(&artifact_path, &bytes).expect("write");

    let artifact = Artifact::open(&artifact_path).expect("mmap the artifact");
    assert_eq!(artifact.graph().node_count(), imported.graph().node_count());
    assert_eq!(
        artifact.graph().value_count(),
        imported.graph().value_count()
    );
    assert_eq!(artifact.weights().len(), imported.weights().len());
    assert_eq!(artifact.default_target(), "cpu-reference");
    assert_eq!(
        artifact.header().source_onnx_sha256,
        source_digest,
        "the header carries the source digest"
    );

    // No ONNX is passed to the session path: the artifact is the whole input.
    let mut from_artifact =
        Session::from_artifact(&artifact, Threading::Fixed(2)).expect("session from artifact");
    let imported = Arc::new(imported);
    let mut in_memory = Session::new_with_options(
        Arc::clone(&imported),
        Threading::Fixed(2),
        EngineChoice::Scalar,
        PlanOptions::unfused(),
    )
    .expect("in-memory session");

    let input_shape = imported
        .graph()
        .value(imported.inputs()[0])
        .expect("input value")
        .ty
        .static_shape()
        .expect("the pinned input is static");
    let elements: usize = input_shape.dims().iter().map(|dim| *dim as usize).product();
    let input: Vec<f32> = (0..elements)
        .map(|index| ((index % 251) as f32 - 125.0) * 0.0078125)
        .collect();
    let name = imported
        .graph()
        .value(imported.inputs()[0])
        .and_then(|value| value.name.clone())
        .expect("the input is named");

    let expected = in_memory
        .run(&[(name.as_str(), &input)])
        .expect("in-memory forward");
    let actual = from_artifact
        .run(&[(name.as_str(), &input)])
        .expect("artifact forward");
    assert_eq!(actual.len(), expected.len());
    for ((actual_name, actual_values), (expected_name, expected_values)) in
        actual.iter().zip(expected.iter())
    {
        assert_eq!(actual_name, expected_name);
        assert_eq!(actual_values.len(), expected_values.len());
        for (position, (left, right)) in actual_values.iter().zip(expected_values).enumerate() {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "output {actual_name}[{position}]: artifact {left} != in-memory {right}"
            );
        }
    }
    println!(
        "[fn3artifact][roundtrip] {} nodes, {} values, {} weights, {} output elements, bitwise",
        artifact.graph().node_count(),
        artifact.graph().value_count(),
        artifact.weights().len(),
        actual.iter().map(|(_, values)| values.len()).sum::<usize>()
    );
}

#[test]
fn per_target_plans_are_stored_and_selected_without_replanning() {
    let Some(source) = yolov8n_bytes() else {
        return;
    };
    let imported = import_onnx_graph(&source).expect("the pinned CNN imports");
    let directory = tempdir::Scratch::new("targets");
    let path = directory.path().join("all.fnn3");
    let prepared = PreparedArtifact::from_imported(
        &imported,
        &["cpu", "cpu-reference"],
        Precision::F32,
        digest(&source),
    )
    .expect("prepare two targets");
    prepared.write_file(&path).expect("write");

    let artifact = Artifact::open(&path).expect("open");
    assert_eq!(artifact.plans().len(), 2);
    assert_eq!(artifact.default_target(), "cpu");
    let cpu = artifact.plan_for("cpu").expect("cpu plan");
    let reference = artifact.plan_for("cpu-reference").expect("reference plan");
    assert!(
        cpu.segments.len() < reference.segments.len(),
        "the fused plan must be shorter: cpu {} vs reference {}",
        cpu.segments.len(),
        reference.segments.len()
    );
    assert!(!cpu.fusions.is_empty());
    assert!(reference.fusions.is_empty());
    assert_eq!(
        reference.unfused_segment_count(),
        imported.graph().node_count()
    );
    assert!(reference
        .segments
        .iter()
        .all(|segment| segment.engine == fastnn::inference::backend::SegmentEngine::Scalar));

    // Both plans open from the one mapping, with no new plan built.
    let default = Session::from_artifact(&artifact, Threading::Fixed(1)).expect("default plan");
    let other = Session::from_artifact_for(&artifact, "cpu-reference", Threading::Fixed(1))
        .expect("reference plan");
    assert_eq!(default.plan(), cpu);
    assert_eq!(other.plan(), reference);
    println!(
        "[fn3artifact][targets] cpu segments={} fusions={} reference segments={}",
        cpu.segments.len(),
        cpu.fusions.len(),
        reference.segments.len()
    );
}

// ---------------------------------------------------------------------------
// mmap sharing
// ---------------------------------------------------------------------------

#[test]
fn two_sessions_share_one_weight_mapping() {
    let (path, _directory) = synthetic(1024, "shared");
    let artifact = Artifact::open(&path).expect("open");
    let weight_slots = artifact.weight_slots();
    assert!(!weight_slots.is_empty(), "the synthetic graph has a weight");
    let (slot, weight_id) = weight_slots[0];

    let first = Session::from_shared_artifact(Arc::new(artifact.clone()), Threading::Fixed(1))
        .expect("first session");
    let second = Session::from_shared_artifact(Arc::new(artifact.clone()), Threading::Fixed(1))
        .expect("second session");
    let left = first.weight_slot_bytes(slot).expect("first weight");
    let right = second.weight_slot_bytes(slot).expect("second weight");
    assert_eq!(
        left.as_ptr(),
        right.as_ptr(),
        "two sessions must share the same weight address"
    );
    assert!(
        artifact.contains_address(left.as_ptr(), left.len()),
        "the weight pointer must lie inside the artifact mapping"
    );
    let direct = artifact
        .weight_bytes_at(weight_id)
        .expect("weight bytes from the mapping");
    assert_eq!(left.as_ptr(), direct.as_ptr());
    assert_eq!(left.len(), 4096);
    for (position, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(a, b, "byte {position}");
    }
    let mapped = artifact.mapped_bytes();
    println!(
        "[fn3artifact][mmap] weight={} bytes at {:p}, mapping={} bytes, two sessions share it",
        left.len(),
        left.as_ptr(),
        mapped
    );
}

// ---------------------------------------------------------------------------
// Corruption
// ---------------------------------------------------------------------------

/// `(tag, absolute offset, length)` for every section, parsed independently of
/// the implementation's constants.
fn directory(bytes: &[u8]) -> Vec<(u32, usize, u64)> {
    assert!(bytes.len() > HEADER_BYTES + DIRECTORY_BYTES);
    let count =
        u32::from_le_bytes(bytes[HEADER_BYTES..HEADER_BYTES + 4].try_into().unwrap()) as usize;
    assert_eq!(count, SECTION_COUNT);
    let mut entries = Vec::new();
    let mut cursor = HEADER_BYTES + 8;
    for _ in 0..count {
        let tag = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        let offset =
            u64::from_le_bytes(bytes[cursor + 8..cursor + 16].try_into().unwrap()) as usize;
        let length = u64::from_le_bytes(bytes[cursor + 16..cursor + 24].try_into().unwrap());
        entries.push((tag, offset, length));
        cursor += 24;
    }
    entries
}

#[test]
fn every_section_flip_and_truncation_is_a_typed_error() {
    let (path, _directory) = synthetic(64, "corrupt");
    let bytes = std::fs::read(&path).expect("read");
    let sections = directory(&bytes);
    assert_eq!(sections.len(), SECTION_COUNT);

    // Every truncation prefix is refused.
    for length in 0..bytes.len() {
        let error = Artifact::from_bytes(bytes[..length].to_vec()).unwrap_err();
        assert!(error.is_corruption(), "prefix {length}: {error}");
    }

    // A flipped byte inside each non-weight section is a hash mismatch naming
    // that section.
    for (tag, offset, length) in &sections {
        if *tag == 5 {
            // HASHES is tested separately below; a flip there may surface as
            // any section's mismatch.
            continue;
        }
        if *tag == 3 {
            // WEIGHTS is covered too, but choose a byte in the table rather than
            // a payload byte so the test does not depend on the payload.
            let position = *offset + 3;
            let mut mutated = bytes.clone();
            mutated[position] ^= 0x01;
            let error = Artifact::from_bytes(mutated).unwrap_err();
            assert!(
                matches!(error.kind(), FormatErrorKind::HashMismatch { .. }),
                "WEIGHTS table flip: {error}"
            );
            continue;
        }
        let position = *offset + (*length as usize / 2);
        let mut mutated = bytes.clone();
        mutated[position] ^= 0x40;
        let error = Artifact::from_bytes(mutated).unwrap_err();
        assert!(
            matches!(error.kind(), FormatErrorKind::HashMismatch { .. }),
            "section {tag} flip at {position}: {error}"
        );
        assert_eq!(error.section(), Some(section_name(*tag)));
    }

    // A flipped byte in HASHES is either an unknown tag or a mismatch: never
    // accepted.
    let (_, hash_offset, hash_length) = sections[SECTION_COUNT - 1];
    for position in hash_offset..(hash_offset + hash_length as usize) {
        let mut mutated = bytes.clone();
        mutated[position] ^= 0x20;
        assert!(
            Artifact::from_bytes(mutated).is_err(),
            "HASHES byte {position} was accepted"
        );
    }

    // A flipped header byte is rejected by a field check or the HEADER hash.
    for position in 0..HEADER_BYTES {
        let mut mutated = bytes.clone();
        mutated[position] ^= 0x10;
        assert!(
            Artifact::from_bytes(mutated).is_err(),
            "header byte {position} was accepted"
        );
    }

    // Every directory byte is either checked structurally or covered by an
    // offset/length equality, so no single-bit flip may be accepted.
    for position in HEADER_BYTES..(HEADER_BYTES + DIRECTORY_BYTES) {
        for mask in [0x01u8, 0x80] {
            let mut mutated = bytes.clone();
            mutated[position] ^= mask;
            assert!(
                Artifact::from_bytes(mutated).is_err(),
                "directory byte {position} mask {mask:#x} was accepted"
            );
        }
    }
}

fn section_name(tag: u32) -> &'static str {
    match tag {
        1 => "GRAPH",
        2 => "PLANS",
        3 => "WEIGHTS",
        4 => "QUANT",
        5 => "HASHES",
        _ => "HEADER",
    }
}

#[test]
fn a_bad_section_length_or_offset_is_malformed_at_its_byte() {
    let (path, _directory) = synthetic(16, "length");
    let bytes = std::fs::read(&path).expect("read");
    let header = Artifact::open(&path).expect("open").header().clone();
    let directory = &bytes[HEADER_BYTES..HEADER_BYTES + DIRECTORY_BYTES];

    // The first section header starts at directory byte 8: tag, reserved,
    // offset, length.
    let mut bad_length = directory.to_vec();
    bad_length[8 + 16..8 + 24].copy_from_slice(&(1u64 << 40).to_le_bytes());
    let error = fastnn::inference::model::format::parse_directory(&bad_length, &header)
        .expect_err("an overlong section must fail");
    assert!(
        matches!(error.kind(), FormatErrorKind::Truncated { .. }),
        "{error}"
    );
    assert!(error.offset().is_some());

    let mut bad_offset = directory.to_vec();
    bad_offset[8 + 8..8 + 16].copy_from_slice(&(HEADER_BYTES as u64 + 1024).to_le_bytes());
    let error = fastnn::inference::model::format::parse_directory(&bad_offset, &header)
        .expect_err("a misaligned section offset must fail");
    match error.kind() {
        FormatErrorKind::MalformedAt { section, .. } => assert_eq!(*section, "HEADER"),
        other => panic!("expected MalformedAt, found {other:?}"),
    }
    assert!(error.to_string().contains("byte "), "{error}");
}

#[test]
fn a_future_version_and_a_version_one_file_are_typed() {
    let (path, _directory) = synthetic(8, "version");
    let bytes = std::fs::read(&path).expect("read");

    let mut future = bytes.clone();
    future[8..10].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
    let error = Artifact::from_bytes(future).unwrap_err();
    assert!(matches!(
        error.kind(),
        FormatErrorKind::UnsupportedVersion {
            found,
            supported
        } if *found == FORMAT_VERSION + 1 && *supported == FORMAT_VERSION
    ));

    let mut version_one = bytes.clone();
    version_one[..8].copy_from_slice(&LEGACY_MAGIC);
    version_one[8..10].copy_from_slice(&1u16.to_le_bytes());
    let error = Artifact::from_bytes(version_one).unwrap_err();
    match error.kind() {
        FormatErrorKind::ObsoleteFormat { magic, version } => {
            assert_eq!(*magic, LEGACY_MAGIC);
            assert_eq!(*version, 1);
        }
        other => panic!("expected ObsoleteFormat, found {other:?}"),
    }
    assert!(error.to_string().contains("Re-prepare"), "{error}");
    assert!(error.is_version_mismatch());

    let mut bad_magic = bytes.clone();
    bad_magic[..8].copy_from_slice(b"FNN2PREP");
    assert!(matches!(
        Artifact::from_bytes(bad_magic).unwrap_err().kind(),
        FormatErrorKind::BadMagic { .. }
    ));
    assert_eq!(&MAGIC, b"FNN3ART2");
}

// ---------------------------------------------------------------------------
// RSS: a 200 MB model must not be copied on load
// ---------------------------------------------------------------------------

#[test]
fn a_200_mb_model_loads_without_copying_its_weights() {
    const WEIGHT_BYTES: u64 = 200_000_000;
    let elements = WEIGHT_BYTES / 4;
    let (path, _directory) = synthetic(elements, "rss");
    let file_bytes = std::fs::metadata(&path).expect("artifact metadata").len();
    assert!(file_bytes >= WEIGHT_BYTES);

    let before = rss_bytes();
    let artifact = Artifact::open(&path).expect("mmap the 200 MB artifact");
    let after_open = rss_bytes();
    let first = Session::from_shared_artifact(Arc::new(artifact.clone()), Threading::Fixed(1))
        .expect("first session");
    let second = Session::from_shared_artifact(Arc::new(artifact.clone()), Threading::Fixed(1))
        .expect("second session");
    let slots = artifact.weight_slots();
    let (slot, _) = slots[0];
    let first_weight = first.weight_slot_bytes(slot).expect("first weight slice");
    let second_weight = second.weight_slot_bytes(slot).expect("second weight slice");
    assert_eq!(first_weight.as_ptr(), second_weight.as_ptr());
    assert!(artifact.contains_address(first_weight.as_ptr(), first_weight.len()));
    let after_sessions = rss_bytes();

    let growth = after_sessions.saturating_sub(before);
    println!(
        "[fn3artifact][rss] file={} B weights={} B rss before={} B after_open={} B \
         after_two_sessions={} B growth={} B ({}% of the weight payload)",
        file_bytes,
        WEIGHT_BYTES,
        before,
        after_open,
        after_sessions,
        growth,
        growth * 100 / WEIGHT_BYTES
    );
    if before == 0 {
        // /proc is unavailable: do not claim a measurement that was not taken.
        return;
    }
    assert!(
        growth < WEIGHT_BYTES / 4,
        "RSS grew by {growth} B while loading a {WEIGHT_BYTES}-byte weight payload; the mapping \
         is being copied (file {file_bytes} B, before {before} B, after {after_sessions} B)"
    );
    assert_eq!(artifact.weight_bytes(), WEIGHT_BYTES);
    assert_eq!(artifact.header().weight_payload_bytes, WEIGHT_BYTES);
}

// ---------------------------------------------------------------------------
// Byte helpers
// ---------------------------------------------------------------------------

fn digest(bytes: &[u8]) -> [u8; 32] {
    let text = sha256_hex(bytes);
    let mut digest = [0u8; 32];
    for (index, chunk) in text.as_bytes().chunks_exact(2).enumerate() {
        if let Ok(chunk) = std::str::from_utf8(chunk) {
            digest[index] = u8::from_str_radix(chunk, 16).unwrap_or(0);
        }
    }
    digest
}

#[test]
fn the_high_level_api_prepares_loads_inspects_and_doctors() {
    let Some(_source) = yolov8n_bytes() else {
        return;
    };
    let directory = tempdir::Scratch::new("api");
    let path = directory.path().join("api.fnn3");
    let report = fastnn::inference::prepare(YOLOV8N, &path, "cpu", "f32").expect("api prepare");
    assert_eq!(report.nodes, 240);
    assert_eq!(report.values, 398);
    assert_eq!(report.weights, 149);
    assert!(report.segments > 0);
    assert_eq!(
        report.artifact_bytes,
        std::fs::metadata(&path).unwrap().len()
    );

    let inspection = fastnn::inference::inspect(&path).expect("api inspect");
    assert_eq!(inspection.mapped_bytes, 0);
    assert_eq!(inspection.nodes, report.nodes);
    assert_eq!(inspection.plans.len(), 1);
    assert!(inspection.bytes_read < inspection.file_bytes);

    let mut session = fastnn::inference::load(&path, Threading::Fixed(2)).expect("api load");
    let name = session.plan().input_slot("images").expect("images input");
    let elements = session.plan().slot(name).unwrap().bytes / 4;
    let outputs = session
        .run(&[("images", &vec![0.0f32; elements])])
        .expect("api run");
    assert_eq!(outputs[0].0, "output0");

    let doctor = fastnn::inference::doctor(YOLOV8N).expect("api doctor");
    assert!(doctor.ok);
    assert_eq!(doctor.nodes, 240);
    assert!(doctor.expected_memory_bytes > 0);
}
