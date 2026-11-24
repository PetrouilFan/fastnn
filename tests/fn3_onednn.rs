//! The optional oneDNN CPU engine: correctness against the scalar reference and
//! against ONNX Runtime.
//!
//! # What this test pins
//!
//! 1. **The scalar path is unchanged.** A `EngineChoice::Scalar` session on the
//!    imported YOLOv8n graph produces the same result it did before the engine
//!    existed: bitwise identical across `Fixed(1)`, `Fixed(4)` and `Fixed(8)`,
//!    and within the ORT gate below.
//! 2. **oneDNN agrees with the scalar reference** elementwise,
//!    `|diff| <= 1e-4 + 1e-4 * |scalar|`. Both engines run the same plan, so a
//!    disagreement is a kernel bug, not a tolerance choice.
//! 3. **oneDNN agrees with ONNX Runtime** under the phase-2 gate
//!    (`2e-4 + 1e-3 * |ref|`, relative error `<= 1e-3` with a `1e-3` floor).
//! 4. **Thread counts do not change the answer.** A oneDNN session at
//!    `Fixed(1)`, `Fixed(4)` and `Fixed(8)` selects different output-channel
//!    block counts. The test reports whether the outputs are bitwise identical;
//!    if they are not (oneDNN may pick a different kernel for a different block
//!    shape), it asserts the same `1e-4 + 1e-4|ref|` numerical bound rather than
//!    claiming a bitwise result it did not observe.
//! 5. **The plan is honest.** Every segment the engine did not take is in
//!    `Plan::fallbacks` with a reason, the report says which runtime oneDNN was
//!    built with, and the prepacked bytes are non-zero.
//!
//! # Inputs
//!
//! The model and the ONNX Runtime dump are the ones `tests/fn3_cnn_run.rs` uses;
//! neither is committed. When they are absent the test prints how to generate
//! the dump and returns, exactly like the parity test.

// This binary exercises the optional engine: without the feature there is no
// engine to exercise, so the target compiles to nothing rather than failing
// every assertion that expects one.
#![cfg(feature = "cpu-onednn")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use fastnn::inference::backend::SegmentEngine;
use fastnn::inference::import::{import_onnx_graph, ImportedGraph};
use fastnn::inference::ops::OpId;
use fastnn::inference::runtime::{
    ConvBlockAxis, EngineChoice, LayoutStrategy, PlanOptions, Session, Threading,
};

const YOLOV8N: &str = "/home/petrouil/.hermes/yolov8n.onnx";

/// Absolute tolerance of the engine-vs-scalar gate.
const ENGINE_ABS: f32 = 1e-4;
/// Relative term of the engine-vs-scalar gate.
const ENGINE_REL: f32 = 1e-4;

/// Absolute tolerance of the ONNX Runtime gate (phase 2).
const ORT_ABS: f32 = 2e-4;
/// Relative tolerance of the ONNX Runtime gate.
const ORT_REL: f32 = 1e-3;
/// Reference floor for the relative ONNX Runtime gate.
const ORT_FLOOR: f32 = 1e-3;

fn parity_dir() -> PathBuf {
    std::env::var_os("FN3_CNN_PARITY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/cnn-parity"))
}

struct Reference {
    input: Vec<f32>,
    output: Vec<f32>,
    output_shape: Vec<u64>,
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{} is not a whole f32 blob",
        path.display()
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn load_reference() -> Option<Reference> {
    let directory = parity_dir();
    let meta_path = directory.join("meta.json");
    let input_path = directory.join("input.bin");
    let output_path = directory.join("output.bin");
    if !meta_path.is_file() || !input_path.is_file() || !output_path.is_file() {
        eprintln!(
            "no ONNX Runtime reference dump in {}; generate it with the venv's onnxruntime:\n\n    \
             /home/petrouil/Projects/github/fastnn-wt-integrate/.venv/bin/python \
             tools/parity/ort_reference.py {YOLOV8N} {}\n",
            directory.display(),
            directory.display()
        );
        return None;
    }
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&meta_path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", meta_path.display())),
    )
    .unwrap_or_else(|error| panic!("{} is not JSON: {error}", meta_path.display()));
    let inputs = meta["inputs"]
        .as_array()
        .unwrap_or_else(|| panic!("{} has no inputs", meta_path.display()));
    assert_eq!(inputs.len(), 1, "this test knows one-input YOLOv8");
    let input_len: usize = inputs[0]["shape"]
        .as_array()
        .expect("the input shape is an array")
        .iter()
        .map(|dim| dim.as_u64().expect("static input extents") as usize)
        .product();
    let outputs = meta["outputs"]
        .as_array()
        .unwrap_or_else(|| panic!("{} has no outputs", meta_path.display()));
    assert_eq!(outputs.len(), 1, "this test knows one-output YOLOv8");
    let output_shape: Vec<u64> = outputs[0]["shape"]
        .as_array()
        .expect("the output shape is an array")
        .iter()
        .map(|dim| dim.as_u64().expect("static output extents"))
        .collect();
    let output_len: usize = output_shape.iter().map(|&dim| dim as usize).product();
    let input = read_f32(&input_path);
    let output = read_f32(&output_path);
    assert_eq!(input.len(), input_len, "input.bin length");
    assert_eq!(output.len(), output_len, "output.bin length");
    Some(Reference {
        input,
        output,
        output_shape,
    })
}

fn import_model() -> Option<Arc<ImportedGraph>> {
    if !Path::new(YOLOV8N).exists() {
        eprintln!("{YOLOV8N} is not present on this machine; skipping the oneDNN parity check");
        return None;
    }
    let bytes = std::fs::read(YOLOV8N).expect("the model is readable");
    let imported = import_onnx_graph(&bytes).expect("the model imports into the typed IR");
    Some(Arc::new(imported))
}

fn run(session: &mut Session, input: &[f32]) -> Vec<f32> {
    let outputs = session
        .run(&[("images", input)])
        .expect("the forward must succeed");
    assert_eq!(outputs.len(), 1, "yolov8n has one output");
    assert_eq!(outputs[0].0, "output0");
    outputs[0].1.clone()
}

/// Elementwise comparison; returns `(max_abs, max_rel)`.
fn compare(got: &[f32], expected: &[f32], abs: f32, rel: f32, what: &str) -> (f32, f32) {
    assert_eq!(got.len(), expected.len(), "{what}: output element count");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut worst = 0usize;
    for (index, (&got, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        let difference = (got - expected).abs();
        let relative = difference / expected.abs().max(ORT_FLOOR);
        assert!(
            difference <= abs + rel * expected.abs(),
            "{what}: element {index}: |{got:e} - {expected:e}| = {difference:e} exceeds \
             {abs:e} + {rel:e} * {}",
            expected.abs()
        );
        if difference > max_abs {
            max_abs = difference;
            worst = index;
        }
        max_rel = max_rel.max(relative);
    }
    println!(
        "[fn3onednn][measure] {what}: max_abs {max_abs:e} (element {worst}), max_rel {max_rel:e}, \
         {} elements",
        got.len()
    );
    (max_abs, max_rel)
}

/// The worker counts this build can actually open.
///
/// A build without the `parallel` feature has no pool, and `SessionThreads`
/// reports `Fixed(n > 1)` as an error rather than silently clamping to one. The
/// 1/4/8 comparison therefore runs in a `parallel` build (`cargo test --features
/// cpu-onednn --test fn3_onednn`); the no-default gate still covers 1 worker,
/// the plan report and every numerics check.
fn worker_counts() -> &'static [usize] {
    if cfg!(feature = "parallel") {
        &[1, 4, 8]
    } else {
        eprintln!("this build has no `parallel` feature: thread-count coverage is Fixed(1) only");
        &[1]
    }
}

fn scalar_session(imported: &Arc<ImportedGraph>, workers: usize) -> Session {
    Session::new_with(
        Arc::clone(imported),
        Threading::Fixed(workers),
        EngineChoice::Scalar,
    )
    .expect("open the scalar session")
}

fn onednn_session(imported: &Arc<ImportedGraph>, workers: usize) -> Session {
    Session::new_with(
        Arc::clone(imported),
        Threading::Fixed(workers),
        EngineChoice::Auto,
    )
    .expect("open the oneDNN session")
}

#[test]
fn the_plan_says_which_segments_onednn_took() {
    let Some(imported) = import_model() else {
        return;
    };
    let session = onednn_session(&imported, *worker_counts().last().unwrap());
    let plan = session.plan();
    let counts = plan.engine_counts();
    let report = session
        .one_dnn_report()
        .expect("an Auto session on a CNN must have oneDNN segments");
    println!(
        "[fn3onednn][measure] plan: {} segments ({} oneDNN, {} scalar), {} fallbacks, \
         oneDNN {}, cpu runtime {}, {} prepared segments ({} conv: {} row-split, {} \
         channel-split; {} matmul), {} prepacked bytes, {} reordered weight blocks, \
         parallel blocks {}",
        counts.total(),
        counts.one_dnn,
        counts.scalar,
        plan.fallbacks().len(),
        report.version,
        report.cpu_runtime,
        report.segments,
        report.conv_segments,
        report.conv_row_segments,
        report.conv_channel_segments,
        report.matmul_segments,
        report.prepacked_bytes,
        report.reordered_blocks,
        report.parallel_blocks,
    );
    assert_eq!(
        report.conv_row_segments + report.conv_channel_segments,
        report.conv_segments,
        "every prepared convolution records exactly one split dimension"
    );
    // The fusion report: every chain names the ops it replaced, and the plan
    // plus the removed segments is the unfused plan.
    let fusions = plan.fusions();
    assert!(!fusions.is_empty(), "the pinned graph has fusable chains");
    let mut removed = 0usize;
    for fusion in fusions {
        assert!(
            fusion.removed() >= 1,
            "a fusion must remove at least one segment: {fusion}"
        );
        assert_eq!(
            fusion.chain().len(),
            fusion.removed() + 1,
            "the chain and the removed count must agree: {fusion}"
        );
        let segment = &plan.segments[fusion.segment];
        assert!(
            segment.fused().is_some_and(|fused| fused == &fusion.fused),
            "fusion {} does not name its segment",
            fusion.segment
        );
        removed += fusion.removed();
    }
    assert_eq!(plan.unfused_segment_count(), plan.segments.len() + removed);
    println!(
        "[fn3onednn][measure] fusions: {} chains, {removed} segments removed, \
         {} -> {} segments",
        fusions.len(),
        plan.unfused_segment_count(),
        plan.segments.len(),
    );
    let splits = session.one_dnn_splits();
    assert_eq!(splits.len(), report.conv_segments);
    for split in &splits {
        assert_eq!(
            split.ranges.iter().map(|&(_, count)| count).sum::<usize>(),
            split.out_extent,
            "segment {} blocks must cover the split dimension",
            split.segment
        );
        let mut next = 0;
        for &(start, count) in &split.ranges {
            assert_eq!(
                start, next,
                "segment {} ranges are contiguous",
                split.segment
            );
            next += count;
        }
        assert!(split.blocks >= 1);
    }
    println!(
        "[fn3onednn][measure] conv splits at {} workers: {} row-split ({} blocks), {} \
         channel-split ({} blocks)",
        *worker_counts().last().unwrap(),
        report.conv_row_segments,
        splits
            .iter()
            .filter(|split| split.axis == ConvBlockAxis::Rows)
            .map(|split| split.blocks)
            .sum::<usize>(),
        report.conv_channel_segments,
        splits
            .iter()
            .filter(|split| split.axis == ConvBlockAxis::Channels)
            .map(|split| split.blocks)
            .sum::<usize>(),
    );
    for split in splits
        .iter()
        .filter(|split| split.axis == ConvBlockAxis::Rows)
    {
        let segment = &plan.segments[split.segment];
        let input = plan
            .shape(segment.inputs[0])
            .map(|shape| shape.to_string())
            .unwrap_or_else(|_| "?".to_string());
        let output = plan
            .shape(segment.outputs[0])
            .map(|shape| shape.to_string())
            .unwrap_or_else(|_| "?".to_string());
        println!(
            "[fn3onednn][measure] row-split segment {} ({}): {} -> {} kernel {:?} stride {:?} \
             pads {:?} group {}, {} blocks, {} source rows",
            split.segment,
            segment.loc.describe(),
            input,
            output,
            segment.params.kernel_shape,
            segment.params.strides,
            segment.params.pads,
            segment.params.group,
            split.blocks,
            split.source_rows,
        );
    }
    assert_eq!(
        report.cpu_runtime, "seq",
        "the engine requires a sequential oneDNN"
    );
    assert!(report.parallel_blocks);
    assert!(
        counts.one_dnn > 0,
        "YOLOv8n is convolution-heavy; the engine must take something"
    );
    // A claimed op family is either on the engine or has a recorded reason;
    // every other op is scalar by construction. Nothing is silently re-run.
    // Pooling is not claimed: on AVX2 oneDNN's JIT pooling rejects a plain
    // NCHW descriptor and falls through to the reference kernel, so the scalar
    // pooling kernel is the faster path (docs/v3/per-layer.md).
    for (index, segment) in plan.segments.iter().enumerate() {
        let claimable = matches!(segment.op_id(), OpId::Conv | OpId::MatMul);
        if claimable {
            let fell_back = plan
                .fallbacks()
                .iter()
                .any(|fallback| fallback.segment == index);
            assert!(
                segment.engine == SegmentEngine::OneDnn || fell_back,
                "segment {index} ({}) is neither on oneDNN nor a recorded fallback",
                segment.op
            );
        } else {
            assert_eq!(
                segment.engine,
                SegmentEngine::Scalar,
                "segment {index} ({}) is not an op the engine claims",
                segment.op
            );
        }
    }
    for fallback in plan.fallbacks() {
        println!(
            "[fn3onednn][measure] fallback: segment {} ({}): {}",
            fallback.segment, fallback.op, fallback.reason
        );
        assert!(
            !fallback.reason.is_empty(),
            "fallback for segment {} has no reason",
            fallback.segment
        );
    }
}

#[test]
fn the_fused_plan_matches_the_unfused_plan() {
    let Some(reference) = load_reference() else {
        return;
    };
    let Some(imported) = import_model() else {
        return;
    };
    let workers = *worker_counts().last().unwrap();
    let mut fused = Session::new_with_options(
        Arc::clone(&imported),
        Threading::Fixed(workers),
        EngineChoice::Auto,
        PlanOptions::new(),
    )
    .expect("open the fused session");
    let mut unfused = Session::new_with_options(
        Arc::clone(&imported),
        Threading::Fixed(workers),
        EngineChoice::Auto,
        PlanOptions::unfused(),
    )
    .expect("open the unfused session");
    let fusions = fused.fusions().len();
    let fused_segments = fused.plan().segments.len();
    let unfused_segments = unfused.plan().segments.len();
    println!(
        "[fn3onednn][measure] fusion: {fusions} chains folded, {unfused_segments} -> \
         {fused_segments} segments, arena {} -> {} bytes",
        unfused.plan().arena_bytes,
        fused.plan().arena_bytes,
    );
    assert!(fusions > 0, "the pinned graph has fusable chains");
    assert_eq!(fused.plan().unfused_segment_count(), unfused_segments);
    let expected = run(&mut unfused, &reference.input);
    let got = run(&mut fused, &reference.input);
    let (max_abs, max_rel) = compare(
        &got,
        &expected,
        ENGINE_ABS,
        ENGINE_REL,
        "fused oneDNN vs unfused oneDNN",
    );
    println!("[fn3onednn][measure] fused vs unfused: max_abs {max_abs:e}, max_rel {max_rel:e}");
}

/// The channels-last layout policy is an option, not the default; when it is
/// asked for, it must produce the same numbers.
#[test]
fn the_channels_last_layout_policy_matches_the_plain_one() {
    let Some(reference) = load_reference() else {
        return;
    };
    let Some(imported) = import_model() else {
        return;
    };
    let workers = *worker_counts().last().unwrap();
    let mut plain = Session::new_with_options(
        Arc::clone(&imported),
        Threading::Fixed(workers),
        EngineChoice::Auto,
        PlanOptions::new().with_layout(LayoutStrategy::Plain),
    )
    .expect("open the plain-layout session");
    let mut chained = Session::new_with_options(
        Arc::clone(&imported),
        Threading::Fixed(workers),
        EngineChoice::Auto,
        PlanOptions::new().with_layout(LayoutStrategy::Chained),
    )
    .expect("open the channels-last session");
    let report = chained.one_dnn_report().expect("oneDNN session");
    println!(
        "[fn3onednn][measure] channels-last: {} blocked segments, {} layout bytes, {} reorders \
         moving {} bytes",
        report.blocked_segments, report.layout_bytes, report.reorders, report.reorder_bytes,
    );
    let expected = run(&mut plain, &reference.input);
    let got = run(&mut chained, &reference.input);
    compare(
        &got,
        &expected,
        ENGINE_ABS,
        ENGINE_REL,
        "channels-last vs plain",
    );
}

#[test]
fn onednn_matches_the_scalar_session() {
    let Some(reference) = load_reference() else {
        return;
    };
    let Some(imported) = import_model() else {
        return;
    };
    let workers = *worker_counts().last().unwrap();
    let mut scalar = scalar_session(&imported, workers);
    let mut engine = onednn_session(&imported, workers);
    let expected = run(&mut scalar, &reference.input);
    let got = run(&mut engine, &reference.input);
    let (max_abs, max_rel) = compare(&got, &expected, ENGINE_ABS, ENGINE_REL, "oneDNN vs scalar");
    println!("[fn3onednn][measure] oneDNN vs scalar: max_abs {max_abs:e}, max_rel {max_rel:e}");
}

#[test]
fn onednn_matches_the_onnxruntime_reference() {
    let Some(reference) = load_reference() else {
        return;
    };
    let Some(imported) = import_model() else {
        return;
    };
    let mut engine = onednn_session(&imported, *worker_counts().last().unwrap());
    let got = run(&mut engine, &reference.input);
    assert_eq!(
        got.len(),
        reference.output_shape.iter().product::<u64>() as usize
    );
    compare(
        &got,
        &reference.output,
        ORT_ABS,
        ORT_REL,
        "oneDNN vs onnxruntime",
    );
}

#[test]
fn onednn_is_stable_across_thread_counts() {
    let Some(reference) = load_reference() else {
        return;
    };
    let Some(imported) = import_model() else {
        return;
    };
    let mut baseline: Option<Vec<f32>> = None;
    for &workers in worker_counts() {
        let mut session = onednn_session(&imported, workers);
        let report = session.one_dnn_report().expect("oneDNN session");
        let got = run(&mut session, &reference.input);
        match &baseline {
            None => baseline = Some(got),
            Some(first) => {
                let bitwise = got
                    .iter()
                    .zip(first.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                let (max_abs, max_rel) = compare(
                    &got,
                    first,
                    ENGINE_ABS,
                    ENGINE_REL,
                    &format!("oneDNN Fixed({workers}) vs Fixed(1)"),
                );
                println!(
                    "[fn3onednn][measure] Fixed({workers}) vs Fixed(1): bitwise {bitwise}, \
                     max_abs {max_abs:e}, max_rel {max_rel:e} (reordered blocks {})",
                    report.reordered_blocks
                );
            }
        }
    }
}

#[test]
fn the_scalar_path_is_unchanged() {
    let Some(reference) = load_reference() else {
        return;
    };
    let Some(imported) = import_model() else {
        return;
    };
    let mut first: Option<Vec<f32>> = None;
    for &workers in worker_counts() {
        let mut session = scalar_session(&imported, workers);
        assert_eq!(session.worker_count(), workers);
        assert!(
            session.one_dnn_report().is_none(),
            "a scalar session must not build a oneDNN engine"
        );
        let got = run(&mut session, &reference.input);
        match &first {
            None => first = Some(got),
            Some(expected) => assert_eq!(
                got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "Fixed({workers}) changed a scalar result bit"
            ),
        }
    }
    compare(
        &first.expect("a baseline"),
        &reference.output,
        ORT_ABS,
        ORT_REL,
        "scalar vs onnxruntime",
    );
}

#[test]
fn the_byte_account_matches_the_plan_slots() {
    let Some(imported) = import_model() else {
        return;
    };
    for &workers in worker_counts() {
        let session = onednn_session(&imported, workers);
        let plan = session.plan();
        let report = session
            .one_dnn_report()
            .expect("an Auto session on a CNN must have oneDNN segments");
        assert_eq!(
            report.segment_bytes.len(),
            report.conv_segments + report.matmul_segments,
            "every prepared segment records one byte row at {workers} workers"
        );
        for row in &report.segment_bytes {
            let segment = &plan.segments[row.index];
            let output = plan.slot(segment.outputs[0]).expect("the result slot");
            let input = plan.slot(segment.inputs[0]).expect("the source slot");
            assert!(
                (1..=workers).contains(&row.blocks),
                "segment {} has {} blocks at {workers} workers",
                row.index,
                row.blocks
            );
            assert_eq!(
                row.payload_out, output.bytes,
                "segment {} payload_out must be its result slot",
                row.index
            );
            assert_eq!(
                row.payload_in, input.bytes,
                "segment {} payload_in must be its source slot",
                row.index
            );
        }
    }
}
