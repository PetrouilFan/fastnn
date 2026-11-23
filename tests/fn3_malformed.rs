//! Malformed-input regressions for the version-2 artifact and the ONNX frontend.
//!
//! The version-1 suite in this file drove a committed corpus of `FNN3PREP`
//! bytes and a CRC-32 payload. Those cases encoded the old layout byte for byte,
//! so they were deleted rather than rewritten: their offsets and error classes no
//! longer exist. The property they protected is kept here against the v2 reader:
//!
//! 1. **A counting allocator** proves a declared multi-gigabyte length or a
//!    hostile target count is refused without a matching allocation.
//! 2. **Exhaustive sweeps** over every truncation prefix and every single-byte
//!    flip of a small artifact and a small ONNX model: the readers return a typed
//!    error or accept the input, and never panic.
//!
//! The committed corpus directory is left in place for the historical record;
//! nothing here compares against it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use fastnn::inference::backend::ScalarOnly;
use fastnn::inference::import::{import_onnx_graph, OpsetImport};
use fastnn::inference::ir::{DimExpr, Graph, SourceLoc, TensorType};
use fastnn::inference::model::format::{
    Artifact, ArtifactHeader, ArtifactMetadata, ArtifactPlan, FormatErrorKind, Precision,
    PreparedArtifact,
};
use fastnn::inference::model::{WeightSet, WeightSource, WeightTensor};
use fastnn::inference::runtime::{Plan, PlanOptions};
use fastnn::inference::shape::Shape;
use fastnn::inference::{DType, OpId, OpParams};

// ---------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------

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

fn measure<T>(run: impl FnOnce() -> T) -> (T, Measured) {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    LARGEST.store(0, Ordering::Relaxed);
    TOTAL.store(0, Ordering::Relaxed);
    let value = run();
    (
        value,
        Measured {
            largest: LARGEST.load(Ordering::Relaxed),
            total: TOTAL.load(Ordering::Relaxed),
        },
    )
}

// ---------------------------------------------------------------------------
// A small valid v2 artifact
// ---------------------------------------------------------------------------

fn tiny_artifact() -> Vec<u8> {
    let weights = WeightSet::new(vec![WeightTensor::new(
        "w",
        DType::F32,
        Shape::from_dims([2, 2]),
        vec![0x11; 16],
        WeightSource::Initializer,
    )
    .expect("payload matches shape")])
    .expect("one weight");
    let mut graph = Graph::new();
    let x = graph
        .add_value(
            Some("x"),
            TensorType::new(DType::F32, [DimExpr::constant(2), DimExpr::constant(2)]),
        )
        .expect("x");
    let w = graph
        .add_value(
            Some("w"),
            TensorType::new(DType::F32, [DimExpr::constant(2), DimExpr::constant(2)]),
        )
        .expect("w");
    let y = graph
        .add_value(
            Some("y"),
            TensorType::new(DType::F32, [DimExpr::constant(2), DimExpr::constant(2)]),
        )
        .expect("y");
    graph
        .add_node(
            OpId::Mul,
            vec![x, w],
            vec![y],
            OpParams::new(),
            SourceLoc::new(0, "Mul").with_output("y"),
        )
        .expect("Mul");
    let plan = Plan::from_graph_with_options(&graph, &weights, &ScalarOnly, PlanOptions::new())
        .expect("plan");
    let metadata = ArtifactMetadata {
        ir_version: 7,
        opset: OpsetImport {
            domain: String::new(),
            version: 13,
        },
        producer: "fastnn-malformed".to_string(),
        producer_version: "0".to_string(),
        graph_name: "tiny".to_string(),
    };
    PreparedArtifact::from_parts(
        metadata,
        [0x33; 32],
        graph,
        vec!["x".to_string()],
        vec!["y".to_string()],
        weights,
        vec![ArtifactPlan::new("cpu", Precision::F32, plan)],
    )
    .expect("artifact")
    .to_bytes()
    .expect("encode")
}

/// A minimal ONNX model: `ModelProto` with an IR version, a default-domain
/// opset and an empty graph with one `x` input and one `y` output.
fn tiny_onnx() -> Vec<u8> {
    fn varint(mut value: u64, out: &mut Vec<u8>) {
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }
    fn field(number: u32, wire: u8, payload: &[u8], out: &mut Vec<u8>) {
        varint(((number << 3) | u32::from(wire)) as u64, out);
        if wire == 2 {
            varint(payload.len() as u64, out);
        }
        out.extend_from_slice(payload);
    }
    fn scalar(number: u32, value: i64, out: &mut Vec<u8>) {
        varint((number << 3) as u64, out);
        varint(value as u64, out);
    }
    // ValueInfoProto with a name and a float tensor type of shape [2].
    fn value_info(name: &str) -> Vec<u8> {
        let mut value = Vec::new();
        field(1, 2, name.as_bytes(), &mut value);
        let mut tensor = Vec::new();
        scalar(1, 1, &mut tensor); // elem_type = FLOAT
        let mut shape = Vec::new();
        let mut dim = Vec::new();
        scalar(1, 2, &mut dim); // dim_value = 2
        field(1, 2, &dim, &mut shape);
        field(2, 2, &shape, &mut tensor);
        // TypeProto.tensor_type (field 1), then ValueInfoProto.type (field 2).
        let mut ty = Vec::new();
        field(1, 2, &tensor, &mut ty);
        field(2, 2, &ty, &mut value);
        value
    }
    let mut node = Vec::new();
    field(1, 2, b"x", &mut node);
    field(2, 2, b"y", &mut node);
    field(4, 2, b"Relu", &mut node);
    let mut graph = Vec::new();
    field(1, 2, &node, &mut graph);
    field(11, 2, &value_info("x"), &mut graph);
    field(12, 2, &value_info("y"), &mut graph);
    let mut model = Vec::new();
    scalar(1, 7, &mut model); // ir_version
    let mut opset = Vec::new();
    scalar(2, 13, &mut opset); // version
    field(8, 2, &opset, &mut model);
    field(7, 2, &graph, &mut model);
    model
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn a_declared_gigabyte_onnx_length_allocates_nothing_large() {
    let (_, measured) = measure(|| {
        let mut bytes = tiny_onnx();
        // The outer ModelProto length is the first byte's varint; claim a
        // multi-gigabyte nested message instead by truncating the graph.
        bytes.extend_from_slice(&[0x3a, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F]);
        let _ = import_onnx_graph(&bytes);
    });
    println!(
        "[fn3malformed] hostile ONNX length: largest allocation {} bytes (total {})",
        measured.largest, measured.total
    );
    assert!(
        measured.largest < 8 * 1024 * 1024,
        "a declared length allocated {} bytes",
        measured.largest
    );
}

#[test]
fn a_hostile_target_count_allocates_nothing_large() {
    let bytes = tiny_artifact();
    let (_, measured) = measure(|| {
        let mut mutated = bytes.clone();
        mutated[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        let error = ArtifactHeader::parse(&mutated, mutated.len() as u64).unwrap_err();
        assert!(matches!(
            error.kind(),
            FormatErrorKind::LimitExceeded { .. }
        ));
    });
    println!(
        "[fn3malformed] hostile target count: largest allocation {} bytes (total {})",
        measured.largest, measured.total
    );
    assert!(measured.largest < 1024 * 1024);
}

#[test]
fn every_truncation_of_a_small_artifact_is_a_typed_error() {
    let bytes = tiny_artifact();
    for length in 0..bytes.len() {
        let error = Artifact::from_bytes(bytes[..length].to_vec()).unwrap_err();
        assert!(!error.to_string().is_empty());
        assert!(error.offset().is_some() || error.is_version_mismatch());
    }
}

#[test]
fn every_single_byte_flip_of_a_small_artifact_is_refused() {
    let bytes = tiny_artifact();
    for position in 0..bytes.len() {
        for mask in [0x01u8, 0x80, 0xFF] {
            let mut mutated = bytes.clone();
            mutated[position] ^= mask;
            match Artifact::from_bytes(mutated) {
                Ok(_) => panic!("byte {position} mask {mask:#x} produced a valid artifact"),
                Err(error) => assert!(
                    !matches!(error.kind(), FormatErrorKind::Io { .. }),
                    "byte {position} produced an I/O error"
                ),
            }
        }
    }
}

#[test]
fn every_truncation_and_flip_of_a_small_onnx_is_typed_or_accepted() {
    let bytes = tiny_onnx();
    if let Err(error) = import_onnx_graph(&bytes) {
        panic!("the hand-built model must be a valid import:\n{error}");
    }
    for length in 0..bytes.len() {
        match import_onnx_graph(&bytes[..length]) {
            Ok(_) => panic!("a prefix of {length} bytes imported"),
            Err(error) => assert!(!error.is_empty()),
        }
    }
    let mut accepted = 0usize;
    for position in 0..bytes.len() {
        for mask in [0x01u8, 0x80, 0xFF] {
            let mut mutated = bytes.clone();
            mutated[position] ^= mask;
            if import_onnx_graph(&mutated).is_ok() {
                accepted += 1;
            }
        }
    }
    // A flipped byte may change a name or an unused field without making the
    // model invalid; the property is "no panic, no out-of-bounds read".
    println!(
        "[fn3malformed] {} bytes, {} single-byte mutations accepted",
        bytes.len(),
        accepted
    );
}

#[test]
fn every_corpus_file_is_still_small_enough_to_commit() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/fn3");
    if !directory.is_dir() {
        return;
    }
    let mut bytes = 0usize;
    let mut files = 0usize;
    for entry in std::fs::read_dir(&directory).expect("the corpus directory") {
        let entry = entry.expect("a directory entry");
        if entry.path().is_file() {
            files += 1;
            bytes += entry.metadata().expect("metadata").len() as usize;
        }
    }
    println!("[fn3malformed][corpus] {files} files, {bytes} bytes on disk");
    assert!(bytes < 64 * 1024, "the corpus grew to {bytes} bytes");
}
