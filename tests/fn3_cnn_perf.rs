//! The v3 CNN performance harness (opt-in).
//!
//! # Why it is opt-in
//!
//! A performance number is only meaningful next to the machine, the thread
//! count and the process's memory. This test therefore:
//!
//! 1. does nothing unless `FN3_CNN_PERF` names an engine — a normal `cargo test`
//!    must not spend thirty forwards on it;
//! 2. measures **one** engine per process, so `VmHWM` (peak RSS) is that
//!    engine's number and not a high-water mark shared with another session;
//! 3. times warm forwards only (three warm-ups, then thirty timed), and reports
//!    median and p95 rather than a mean.
//!
//! Run it as:
//!
//! ```text
//! FN3_CNN_PERF=scalar FN3_CNN_THREADS=1 cargo test --release --test fn3_cnn_perf -- --nocapture
//! FN3_CNN_PERF=onednn FN3_CNN_THREADS=8 cargo test --release --features cpu-onednn \
//!     --test fn3_cnn_perf -- --nocapture
//! ```
//!
//! `Evidence` lines start with `[fn3perf]`.
//!
//! The ONNX Runtime side of the table is produced by
//! `tools/parity/ort_reference.py --time --threads N`, which uses
//! `SessionOptions.intra_op_num_threads`.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use fastnn::inference::import::{import_onnx_graph, ImportedGraph};
use fastnn::inference::runtime::{physical_core_count, EngineChoice, Session, Threading};

const YOLOV8N: &str = "/home/petrouil/.hermes/yolov8n.onnx";

/// Warm-ups before the timed forwards.
const WARMUP: usize = 3;

/// Timed forwards.
const ITERATIONS: usize = 30;

fn machine() -> String {
    let model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_string())
        })
        .unwrap_or_else(|| "unknown CPU".to_string());
    format!(
        "{model}, {} logical CPUs, {} physical cores",
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(0),
        physical_core_count().unwrap_or(0)
    )
}

/// Peak resident set size in MiB (`VmHWM`), or `None` off Linux.
fn peak_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / 1024.0)
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn measure(
    engine: &str,
    workers: usize,
    threads: usize,
    mut session: Session,
    input: &[f32],
    report: Option<String>,
) {
    for _ in 0..WARMUP {
        session
            .run(&[("images", input)])
            .expect("a warm-up forward must succeed");
    }
    let mut samples = Vec::with_capacity(ITERATIONS);
    let mut first: Option<Vec<u32>> = None;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let outputs = session
            .run(&[("images", input)])
            .expect("a timed forward must succeed");
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        let bits: Vec<u32> = outputs[0].1.iter().map(|value| value.to_bits()).collect();
        match &first {
            None => first = Some(bits),
            Some(expected) => assert_eq!(&bits, expected, "a forward changed its result"),
        }
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite durations"));
    let median = percentile(&samples, 0.50);
    let p95 = percentile(&samples, 0.95);
    let min = samples[0];
    let max = samples[samples.len() - 1];
    println!(
        "[fn3perf] engine {engine}, threads {workers}, machine \"{}\"",
        machine()
    );
    if let Some(report) = report {
        println!("[fn3perf] plan: {report}");
    }
    println!(
        "[fn3perf] {engine} threads={workers} median_ms={median:.2} p95_ms={p95:.2} \
         min_ms={min:.2} max_ms={max:.2} iterations={ITERATIONS} peak_rss_mib={} \
         arena_bytes={}",
        peak_rss_mib().map_or("unknown".to_string(), |rss| format!("{rss:.1}")),
        session.plan().arena_bytes,
    );
    let _ = threads;
}

#[test]
fn cnn_forward_latency() {
    let Some(engine) =
        std::env::var_os("FN3_CNN_PERF").map(|value| value.to_string_lossy().into_owned())
    else {
        eprintln!(
            "FN3_CNN_PERF is not set; this harness is opt-in. Set it to `scalar` or `onednn` and \
             run with --release --nocapture."
        );
        return;
    };
    let workers: usize = std::env::var("FN3_CNN_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    if !Path::new(YOLOV8N).exists() {
        eprintln!("{YOLOV8N} is not present on this machine; nothing to measure");
        return;
    }
    let bytes = std::fs::read(YOLOV8N).expect("the model is readable");
    let imported: Arc<ImportedGraph> =
        Arc::new(import_onnx_graph(&bytes).expect("the model imports into the typed IR"));
    let input: Vec<f32> = {
        // Deterministic input independent of the parity dump: a reproducible
        // pseudo-random vector of the graph's declared input extent.
        let input_id = imported.inputs()[0];
        let shape = imported
            .graph()
            .value(input_id)
            .expect("the declared input exists")
            .ty
            .static_shape()
            .expect("static input");
        let elements = shape.numel().expect("a countable input");
        let mut state = 0x1234_5678u32;
        (0..elements)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    };
    match engine.as_str() {
        "scalar" => {
            let session = Session::new_with(
                Arc::clone(&imported),
                Threading::Fixed(workers),
                EngineChoice::Scalar,
            )
            .expect("open the scalar session");
            measure("scalar", workers, workers, session, &input, None);
        }
        "onednn" => {
            #[cfg(feature = "cpu-onednn")]
            {
                let session = Session::new_with(
                    Arc::clone(&imported),
                    Threading::Fixed(workers),
                    EngineChoice::Auto,
                )
                .expect("open the oneDNN session");
                let report = session.one_dnn_report().map(|report| {
                    format!(
                        "oneDNN {} ({} runtime), {} segments ({} conv, {} pool, {} matmul), \
                         {} prepacked bytes, {} reordered weight blocks, parallel blocks {}",
                        report.version,
                        report.cpu_runtime,
                        report.segments,
                        report.conv_segments,
                        report.pool_segments,
                        report.matmul_segments,
                        report.prepacked_bytes,
                        report.reordered_blocks,
                        report.parallel_blocks
                    )
                });
                measure("onednn", workers, workers, session, &input, report);
            }
            #[cfg(not(feature = "cpu-onednn"))]
            panic!("FN3_CNN_PERF=onednn needs --features cpu-onednn");
        }
        other => panic!("FN3_CNN_PERF must be `scalar` or `onednn`, not {other:?}"),
    }
}
