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
//! FN3_CNN_MODEL=/path/to/model.onnx FN3_CNN_PERF=onednn FN3_CNN_THREADS=4 \
//!     cargo test --release --features cpu-onednn --test fn3_cnn_perf -- --nocapture
//! ```
//!
//! Add `--features segment-timing` to print the per-op table (`calls`, median
//! total ms, share) at one and eight threads. That feature compiles the timing
//! into `Session`; a normal build has no clock read and no timing field.
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
#[cfg(feature = "segment-timing")]
use fastnn::inference::ops::OpId;
use fastnn::inference::runtime::{
    physical_core_count, EngineChoice, PlanOptions, Session, Threading,
};

const YOLOV8N: &str = "/home/petrouil/.hermes/yolov8n.onnx";

/// Warm-ups before the timed forwards.
const WARMUP: usize = 3;

/// Timed forwards.
const ITERATIONS: usize = 30;

/// Groups of forwards per op in the per-op table (the median is over groups).
#[cfg(feature = "segment-timing")]
const TIMING_REPEATS: usize = 3;

/// Forwards per group in the per-op table.
#[cfg(feature = "segment-timing")]
const TIMING_FORWARDS: usize = 3;

fn machine() -> String {
    #[cfg(target_os = "linux")]
    let model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_string())
        })
        .unwrap_or_else(|| "unknown CPU".to_string());
    #[cfg(target_os = "windows")]
    let model = std::process::Command::new("reg")
        .args([
            "query",
            r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            "/v",
            "ProcessorNameString",
        ])
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(output.stdout).ok().and_then(|text| {
                text.lines()
                    .find(|line| line.contains("ProcessorNameString"))
                    .and_then(|line| line.split("REG_SZ").nth(1))
                    .map(|value| value.trim().to_string())
            })
        })
        .unwrap_or_else(|| "unknown CPU".to_string());
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    let model = "unknown CPU".to_string();
    format!(
        "{model}, {} logical CPUs, {} physical cores",
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(0),
        physical_core_count().unwrap_or(0)
    )
}

/// Peak working set in MiB. Linux reports `VmHWM`; Windows uses
/// `GetProcessMemoryInfo` so the number is that of this one-engine process.
#[cfg(target_os = "windows")]
fn peak_rss_mib() -> Option<f64> {
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        page_file_usage: usize,
        peak_page_file_usage: usize,
    }

    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(
            process: isize,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
        fn GetCurrentProcess() -> isize;
    }

    let mut counters = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        page_file_usage: 0,
        peak_page_file_usage: 0,
    };
    // SAFETY: the pseudo-handle from GetCurrentProcess is process-local and
    // remains valid for the call; `counters` is a live, correctly initialised
    // struct whose `cb` field describes its own size; the API writes only
    // within that struct and returns before the borrow ends.
    let ok = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            std::mem::size_of::<ProcessMemoryCounters>() as u32,
        )
    };
    if ok == 0 {
        None
    } else {
        Some(counters.peak_working_set_size as f64 / (1024.0 * 1024.0))
    }
}

#[cfg(target_os = "linux")]
fn peak_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / 1024.0)
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn peak_rss_mib() -> Option<f64> {
    None
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

/// Whether the plan this process builds folds recognised chains.
///
/// `FN3_CNN_FUSE=0` builds the unfused oracle (one segment per graph node) with
/// the same engine and thread count, which is how the phase-4a report measures
/// what fusion is worth. Default: fused.
fn fuse_enabled() -> bool {
    std::env::var("FN3_CNN_FUSE")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

fn options() -> PlanOptions {
    PlanOptions::new().with_fuse(fuse_enabled())
}

fn measure(
    engine: &str,
    workers: usize,
    threads: usize,
    mut session: Session,
    input_name: &str,
    input: &[f32],
    report: Option<String>,
) {
    for _ in 0..WARMUP {
        session
            .run(&[(input_name, input)])
            .expect("a warm-up forward must succeed");
    }
    let mut samples = Vec::with_capacity(ITERATIONS);
    let mut first: Option<Vec<u32>> = None;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let outputs = session
            .run(&[(input_name, input)])
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
        "[fn3perf] engine {engine}, threads {workers}, fuse {}, machine \"{}\"",
        fuse_enabled(),
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

/// One op's row in the timing table.
#[cfg(feature = "segment-timing")]
struct OpRow {
    label: String,
    calls: f64,
    total_ms: f64,
    longest_ms: f64,
}

/// Multiply-accumulate count of one segment, when the op has one.
///
/// `Conv` uses the output extent and the weight extent, so groups and depthwise
/// are included without special cases; `MatMul` uses `M*N*K` from its output
/// and left operand. Every other op returns `None` and is reported by time
/// alone.
#[cfg(feature = "segment-timing")]
fn segment_macs(
    plan: &fastnn::inference::RuntimePlan,
    segment: &fastnn::inference::Segment,
) -> Option<u64> {
    let out = plan.slot(*segment.outputs.first()?).ok()?;
    let out_dims: Option<Vec<u64>> = out.ty.dims().iter().map(|d| d.as_const()).collect();
    let out_dims = out_dims?;
    match segment.op_id() {
        OpId::Conv => {
            let weights = plan.slot(*segment.inputs.get(1)?).ok()?;
            let w: Option<Vec<u64>> = weights.ty.dims().iter().map(|d| d.as_const()).collect();
            let w = w?;
            if out_dims.len() != 4 || w.len() != 4 {
                return None;
            }
            let mut macs = out_dims.iter().product::<u64>();
            macs = macs
                .checked_mul(w[1])?
                .checked_mul(w[2])?
                .checked_mul(w[3])?;
            Some(macs)
        }
        OpId::MatMul => {
            let left = plan.slot(*segment.inputs.first()?).ok()?;
            let left_dims: Option<Vec<u64>> = left.ty.dims().iter().map(|d| d.as_const()).collect();
            let left_dims = left_dims?;
            if out_dims.len() != 2 || left_dims.len() != 2 {
                return None;
            }
            Some(
                out_dims[0]
                    .checked_mul(out_dims[1])?
                    .checked_mul(left_dims[1])?,
            )
        }
        _ => None,
    }
}

/// Print the per-op table for one engine and worker count.
///
/// Each repeat is [`TIMING_FORWARDS`] forwards with the session's timing table
/// reset first; the reported total is the median of the repeats divided by the
/// forwards, so a single scheduling hiccup cannot move the table. Warm-up
/// forwards are not counted.
#[cfg(feature = "segment-timing")]
fn print_segment_timings(
    engine: &str,
    imported: &Arc<ImportedGraph>,
    input_name: &str,
    input: &[f32],
    workers: usize,
) {
    let choice = if engine == "scalar" {
        EngineChoice::Scalar
    } else {
        EngineChoice::Auto
    };
    let mut session = Session::new_with_options(
        Arc::clone(imported),
        Threading::Fixed(workers),
        choice,
        options(),
    )
    .expect("open the timing session");
    for _ in 0..WARMUP {
        session
            .run(&[(input_name, input)])
            .expect("a timing warm-up forward must succeed");
    }
    let mut repeats: Vec<Vec<(String, u64, f64, f64)>> = Vec::with_capacity(TIMING_REPEATS);
    let mut occurrences: Vec<Vec<(usize, String, u64, f64)>> = Vec::with_capacity(TIMING_REPEATS);
    let mut wall = Vec::with_capacity(TIMING_REPEATS);
    for _ in 0..TIMING_REPEATS {
        session.reset_segment_timings();
        let started = Instant::now();
        for _ in 0..TIMING_FORWARDS {
            session
                .run(&[(input_name, input)])
                .expect("a timing forward must succeed");
        }
        wall.push(started.elapsed().as_secs_f64() * 1000.0 / TIMING_FORWARDS as f64);
        repeats.push(
            session
                .segment_timings()
                .entries()
                .iter()
                .map(|entry| {
                    (
                        entry.label.clone(),
                        entry.calls,
                        entry.total.as_secs_f64() * 1000.0,
                        entry.longest.as_secs_f64() * 1000.0,
                    )
                })
                .collect(),
        );
        occurrences.push(
            session
                .segment_timings()
                .occurrences()
                .iter()
                .map(|entry| {
                    (
                        entry.index,
                        entry.label.clone(),
                        entry.calls,
                        entry.total.as_secs_f64() * 1000.0,
                    )
                })
                .collect(),
        );
    }
    let mut rows: Vec<OpRow> = Vec::with_capacity(repeats[0].len());
    for (index, (label, _, _, _)) in repeats[0].iter().enumerate() {
        let mut totals = Vec::with_capacity(TIMING_REPEATS);
        let mut longest = 0.0f64;
        let mut calls = 0u64;
        for repeat in &repeats {
            let entry = repeat
                .get(index)
                .filter(|entry| &entry.0 == label)
                .expect("the same plan must record the same ops in the same order");
            totals.push(entry.2 / TIMING_FORWARDS as f64);
            longest = longest.max(entry.3);
            calls = entry.1;
        }
        totals.sort_by(|a, b| a.partial_cmp(b).expect("finite durations"));
        rows.push(OpRow {
            label: label.clone(),
            calls: calls as f64 / TIMING_FORWARDS as f64,
            total_ms: percentile(&totals, 0.50),
            longest_ms: longest,
        });
    }
    rows.sort_by(|a, b| {
        b.total_ms
            .partial_cmp(&a.total_ms)
            .expect("finite durations")
    });
    let accounted: f64 = rows.iter().map(|row| row.total_ms).sum();
    wall.sort_by(|a, b| a.partial_cmp(b).expect("finite durations"));
    let wall_median = percentile(&wall, 0.50);
    println!(
        "[fn3perf][timing] engine {engine}, threads {workers}, median forward {wall_median:.2} ms, \
         segments {accounted:.2} ms, glue {:.2} ms (median of {TIMING_REPEATS} groups of \
         {TIMING_FORWARDS})",
        wall_median - accounted
    );
    println!("[fn3perf][timing]   op              calls  total_ms   share   mean_ms  longest_ms");
    for row in &rows {
        println!(
            "[fn3perf][timing]   {:<15} {:>5.1}  {:>8.2}  {:>6.1}%  {:>8.3}  {:>8.3}",
            row.label,
            row.calls,
            row.total_ms,
            100.0 * row.total_ms / accounted.max(f64::MIN_POSITIVE),
            row.total_ms / row.calls.max(1.0),
            row.longest_ms,
        );
    }
    if std::env::var_os("FN3_CNN_TRACE").is_some() {
        print_segment_trace(&session, &occurrences);
    }
}

/// Print one line per plan segment: median time, MACs and achieved GMAC/s.
///
/// Only with `FN3_CNN_TRACE` set: this is the per-layer view the class table
/// hides, and it is meant to be joined with an ORT per-node profile.
#[cfg(feature = "segment-timing")]
fn print_segment_trace(session: &Session, repeats: &[Vec<(usize, String, u64, f64)>]) {
    let mut by_index: Vec<(usize, String, u64, Vec<f64>)> = Vec::new();
    for repeat in repeats {
        for (index, label, calls, total) in repeat {
            let per_forward = *total / TIMING_FORWARDS as f64;
            match by_index.iter_mut().find(|entry| entry.0 == *index) {
                Some(entry) => entry.3.push(per_forward),
                None => by_index.push((*index, label.clone(), *calls, vec![per_forward])),
            }
        }
    }
    by_index.sort_by_key(|entry| entry.0);
    println!("[fn3perf][trace] index label ms calls macs gmac_s bytes source");
    for (index, label, calls, mut samples) in by_index {
        samples.sort_by(|a, b| a.partial_cmp(b).expect("finite durations"));
        let ms = percentile(&samples, 0.5);
        let segment = &session.plan().segments[index];
        let macs = segment_macs(session.plan(), segment);
        let macs_text = macs.map_or("-".to_string(), |value| value.to_string());
        let gmac = macs.map_or(0.0, |value| value as f64 / ms / 1e6);
        let bytes = session
            .plan()
            .slot(segment.outputs[0])
            .map_or(0, |slot| slot.bytes);
        println!(
            "[fn3perf][trace] {index} {label} {ms:.4} {:.2} {macs_text} {gmac:.2} {bytes} {}",
            calls as f64 / TIMING_FORWARDS as f64,
            segment.loc.describe()
        );
    }
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
    let model = std::env::var("FN3_CNN_MODEL").unwrap_or_else(|_| YOLOV8N.to_string());
    if !Path::new(&model).exists() {
        eprintln!("{model} is not present on this machine; nothing to measure");
        return;
    }
    let bytes = std::fs::read(&model).expect("the model is readable");
    let imported: Arc<ImportedGraph> =
        Arc::new(import_onnx_graph(&bytes).expect("the model imports into the typed IR"));
    let (input_name, input): (String, Vec<f32>) = {
        // Deterministic input independent of the parity dump: a reproducible
        // pseudo-random vector of the graph's declared input extent. The name
        // comes from the graph, so any supported model can be measured here.
        let input_id = imported.inputs()[0];
        let value = imported
            .graph()
            .value(input_id)
            .expect("the declared input exists");
        let name = value.name.clone().unwrap_or_else(|| "input".to_string());
        let shape = value.ty.static_shape().expect("static input");
        let elements = shape.numel().expect("a countable input");
        let mut state = 0x1234_5678u32;
        let values: Vec<f32> = (0..elements)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect();
        (name, values)
    };
    match engine.as_str() {
        "scalar" => {
            let session = Session::new_with_options(
                Arc::clone(&imported),
                Threading::Fixed(workers),
                EngineChoice::Scalar,
                options(),
            )
            .expect("open the scalar session");
            measure(
                "scalar",
                workers,
                workers,
                session,
                &input_name,
                &input,
                None,
            );
            #[cfg(feature = "segment-timing")]
            print_timing_tables(&engine, &imported, &input_name, &input);
        }
        "onednn" => {
            #[cfg(feature = "cpu-onednn")]
            {
                let session = Session::new_with_options(
                    Arc::clone(&imported),
                    Threading::Fixed(workers),
                    EngineChoice::Auto,
                    options(),
                )
                .expect("open the oneDNN session");
                let report = session.one_dnn_report().map(|report| {
                    format!(
                        "oneDNN {} ({} runtime), {} segments ({} conv: {} row-split, {} \
                         channel-split; {} pool, {} matmul), {} fused, {} channels-last ({} \
                         bytes), {} reorders ({} bytes), {} prepacked bytes, {} staging bytes, \
                         {} scratchpad bytes, {} reordered weight blocks, parallel blocks {}",
                        report.version,
                        report.cpu_runtime,
                        report.segments,
                        report.conv_segments,
                        report.conv_row_segments,
                        report.conv_channel_segments,
                        report.pool_segments,
                        report.matmul_segments,
                        report.fused_segments,
                        report.blocked_segments,
                        report.layout_bytes,
                        report.reorders,
                        report.reorder_bytes,
                        report.prepacked_bytes,
                        report.staging_bytes,
                        report.scratchpad_bytes,
                        report.reordered_blocks,
                        report.parallel_blocks
                    )
                });
                measure(
                    "onednn",
                    workers,
                    workers,
                    session,
                    &input_name,
                    &input,
                    report,
                );
                #[cfg(feature = "segment-timing")]
                print_timing_tables(&engine, &imported, &input_name, &input);
            }
            #[cfg(not(feature = "cpu-onednn"))]
            panic!("FN3_CNN_PERF=onednn needs --features cpu-onednn");
        }
        other => panic!("FN3_CNN_PERF must be `scalar` or `onednn`, not {other:?}"),
    }
}

/// The per-op table at one and eight threads.
///
/// Both tables come from fresh sessions on the same imported graph, after the
/// latency measurement above, so the peak-RSS line stays a single-engine number.
#[cfg(feature = "segment-timing")]
fn print_timing_tables(
    engine: &str,
    imported: &Arc<ImportedGraph>,
    input_name: &str,
    input: &[f32],
) {
    for threads in [1usize, 8] {
        if threads > 1 && !cfg!(feature = "parallel") {
            continue;
        }
        print_segment_timings(engine, imported, input_name, input, threads);
    }
}
