//! FN3-26 (#72) matched latency and traffic harness.
//!
//! One binary, one workload, one machine: prefill a 10-token prompt and decode
//! three tokens, with the *same* warmup count (3) and iteration count (20) on
//! every arm of the comparison. It is deliberately portable across the #72
//! commits, so the baseline (`672cec9`) and each step measure the same code:
//!
//! - timings: `std::time::Instant` around the production `Session` calls;
//! - traffic: `VulkanBackend::residency()` (per-op records) and
//!   `StagingPool` counters, which exist at the alpha;
//! - memory: `MemoryStats::peak_live_bytes` (device) and `VmHWM` (host RSS).
//!
//! The `[fn3-26][json]` line is the machine-readable result; the CSV lines are
//! the raw samples. Nothing here asserts a latency: a benchmark that fails a
//! test on a busy host is a flaky gate, not evidence. Correctness is asserted by
//! `fn3_e2e` and `fn3_vulkan_resident`.
//!
//! ```bash
//! FN3_PERF=1 FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
//! FN3_VK_REQUIRE_PHYSICAL=1 FASTNN_FN3_ONNX_MODEL=/path/to/model.onnx \
//!   flock /tmp/opencode/fn3-bench.lock -c \
//!   'CARGO_BUILD_JOBS=5 cargo test --release --features vulkan \
//!      --test fn3_vulkan_perf -- --nocapture --test-threads=1'
//! ```

#![cfg(feature = "vulkan")]

use std::time::Instant;

use fastnn::fn3::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
use fastnn::fn3::device::DeviceRequest;
use fastnn::fn3::OpId;
use fastnn::fn3::Session;

#[path = "fn3_e2e/harness.rs"]
mod harness;

use harness::*;

/// The prompt the #72 brief pins: the first ten tokens of the fixture sequence.
const PROMPT_TOKENS: [u32; 10] = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13];

/// The three decode tokens the #72 brief pins.
const DECODE_TOKENS: [u32; 3] = [640, 11, 612];

fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse()
            .unwrap_or_else(|error| panic!("{name}=`{value}` is not a number: {error}")),
        Err(_) => default,
    }
}

fn physical_adapter() -> Option<u32> {
    let pinned = std::env::var("FN3_VK_DEVICE_UUID")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty());
    adapters()
        .iter()
        .find(|adapter| {
            adapter.has_compute_queue()
                && !adapter.is_software()
                && pinned
                    .as_deref()
                    .map(|uuid| adapter.uuid_string().eq_ignore_ascii_case(uuid))
                    .unwrap_or(true)
        })
        .map(|adapter| adapter.index)
}

fn open_physical() -> Option<std::sync::Arc<VulkanBackend>> {
    let index = physical_adapter()?;
    let backend = VulkanBackend::open_with(
        &DeviceRequest::Vulkan { index: Some(index) },
        VulkanConfig::validated(),
    )
    .unwrap_or_else(|error| panic!("opening adapter {index} failed: {error}"));
    assert!(!backend.is_software(), "the perf harness needs a real GPU");
    Some(backend)
}

/// Peak host resident set from `/proc/self/status` (`VmHWM`), in bytes.
fn peak_rss_bytes() -> u64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Some(kib) = rest.split_whitespace().next() {
                return kib.parse::<u64>().unwrap_or(0) * 1024;
            }
        }
    }
    0
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() as f64 - 1.0) * fraction).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[test]
fn prefill_and_decode_latency() {
    if std::env::var("FN3_PERF").as_deref() != Ok("1") {
        println!("SKIP: set FN3_PERF=1 to run the #72 latency harness");
        return;
    }
    let Some(backend) = open_physical() else {
        panic!("FN3_PERF=1 but no physical Vulkan device is present");
    };
    let Some(import) = pinned() else {
        panic!("FN3_PERF=1 but the pinned ONNX model is not present");
    };
    let iterations = env_u64("FN3_PERF_ITERATIONS", 20) as usize;
    let warmups = env_u64("FN3_PERF_WARMUPS", 3) as usize;
    let device = backend.device_id();
    let (model, _envelope) = prepared(&import, device, FIXTURE_CONTEXT);

    let mut ttft: Vec<f64> = Vec::with_capacity(iterations);
    let mut prefill: Vec<f64> = Vec::with_capacity(iterations);
    let mut decode: Vec<f64> = Vec::with_capacity(iterations * DECODE_TOKENS.len());

    let run_once =
        |measure: bool, ttft: &mut Vec<f64>, prefill: &mut Vec<f64>, decode: &mut Vec<f64>| {
            let started = Instant::now();
            let mut session =
                Session::with_runtime(&model, &import, &*backend).expect("session for the harness");
            let prepared_at = started.elapsed();
            let prefill_start = Instant::now();
            let logits = session
                .prefill(&PROMPT_TOKENS)
                .expect("prefill in the harness");
            let prefill_done = prefill_start.elapsed();
            // Force the materialisation a caller performs: the logits are read.
            let host = logits.as_f32_slice().expect("logits are host-readable");
            assert!(!host.is_empty());
            let first_token = started.elapsed();
            let mut decoded = Vec::new();
            for token in DECODE_TOKENS {
                let start = Instant::now();
                let logits = session.decode(&[token]).expect("decode in the harness");
                let _ = logits.as_f32_slice().expect("logits are host-readable");
                decoded.push(start.elapsed());
            }
            if measure {
                ttft.push(first_token.as_secs_f64() * 1e3);
                prefill.push(prefill_done.as_secs_f64() * 1e3);
                for sample in decoded {
                    decode.push(sample.as_secs_f64() * 1e3);
                }
            }
            let _ = prepared_at;
        };

    for _ in 0..warmups {
        let mut scratch_a = Vec::new();
        let mut scratch_b = Vec::new();
        let mut scratch_c = Vec::new();
        run_once(false, &mut scratch_a, &mut scratch_b, &mut scratch_c);
    }

    backend.residency().clear();
    let memory_before = backend.memory_stats();
    let staging_before = backend.staging_stats();
    for _ in 0..iterations {
        run_once(true, &mut ttft, &mut prefill, &mut decode);
    }
    let memory_after = backend.memory_stats();
    let staging_after = backend.staging_stats();
    let summary = backend.residency().summary();

    let ceiling = |values: &mut Vec<f64>| {
        values.sort_by(|a, b| a.partial_cmp(b).expect("finite sample"));
        (
            percentile(values, 0.5),
            percentile(values, 0.95),
            values.first().copied().unwrap_or(0.0),
            values.last().copied().unwrap_or(0.0),
        )
    };
    let (ttft_p50, ttft_p95, ttft_min, ttft_max) = ceiling(&mut ttft);
    let (prefill_p50, prefill_p95, prefill_min, prefill_max) = ceiling(&mut prefill);
    let (decode_p50, decode_p95, decode_min, decode_max) = ceiling(&mut decode);

    // Per-op traffic, so the report can say *which* op moves the bytes.
    let mut by_op: std::collections::BTreeMap<OpId, (u64, u64, u64)> =
        std::collections::BTreeMap::new();
    for record in backend.residency().records() {
        let entry = by_op.entry(record.op).or_insert((0, 0, 0));
        entry.0 += 1;
        entry.1 += record.upload_bytes;
        entry.2 += record.readback_bytes;
    }
    let mut ops: Vec<(String, u64, u64, u64)> = by_op
        .into_iter()
        .map(|(op, (count, upload, readback))| {
            (
                op.name().to_string(),
                count / iterations.max(1) as u64,
                upload / iterations.max(1) as u64,
                readback / iterations.max(1) as u64,
            )
        })
        .collect();
    ops.sort_by_key(|entry| std::cmp::Reverse(entry.2));
    println!(
        "[fn3-26][per-op] op instructions upload_bytes_per_iteration readback_bytes_per_iteration"
    );
    for (name, count, upload, readback) in &ops {
        println!("[fn3-26][per-op] {name} {count} {upload} {readback}");
    }

    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let load = load.split_whitespace().next().unwrap_or("?").to_string();
    println!(
        "[fn3-26][perf] ttft_ms p50={ttft_p50:.3} p95={ttft_p95:.3} min={ttft_min:.3} \
         max={ttft_max:.3} n={}",
        ttft.len()
    );
    println!(
        "[fn3-26][perf] prefill_ms p50={prefill_p50:.3} p95={prefill_p95:.3} min={prefill_min:.3} \
         max={prefill_max:.3} n={}",
        prefill.len()
    );
    println!(
        "[fn3-26][perf] decode_ms p50={decode_p50:.3} p95={decode_p95:.3} min={decode_min:.3} \
         max={decode_max:.3} n={}",
        decode.len()
    );
    println!(
        "[fn3-26][perf] traffic upload_bytes={} readback_bytes={} executions={} submissions={} \
         input_readbacks={} iterations={iterations}",
        summary.upload_bytes,
        summary.readback_bytes,
        summary.executions,
        summary.submissions,
        summary.input_readbacks
    );
    println!(
        "[fn3-26][perf] memory device_peak_live_bytes={} device_live_bytes={} \
         allocator_chunks={} allocation_count={} host_peak_rss_bytes={} staging_upload_bytes={} \
         staging_readback_bytes={} staging_blocks_created={}",
        memory_after
            .peak_live_bytes
            .max(memory_before.peak_live_bytes),
        memory_after.live_bytes,
        memory_after.chunks,
        memory_after.allocation_count,
        peak_rss_bytes(),
        staging_after.upload_bytes - staging_before.upload_bytes,
        staging_after.readback_bytes - staging_before.readback_bytes,
        staging_after.created_blocks - staging_before.created_blocks,
    );
    let json = format!(
        "{{\"iterations\":{iterations},\"warmups\":{warmups},\
         \"ttft_ms\":{{\"p50\":{ttft_p50:.4},\"p95\":{ttft_p95:.4},\"min\":{ttft_min:.4},\
         \"max\":{ttft_max:.4}}},\
         \"prefill_ms\":{{\"p50\":{prefill_p50:.4},\"p95\":{prefill_p95:.4},\
         \"min\":{prefill_min:.4},\"max\":{prefill_max:.4}}},\
         \"decode_ms\":{{\"p50\":{decode_p50:.4},\"p95\":{decode_p95:.4},\
         \"min\":{decode_min:.4},\"max\":{decode_max:.4}}},\
         \"upload_bytes\":{},\"readback_bytes\":{},\"executions\":{},\
         \"submissions\":{},\"input_readbacks\":{},\
         \"device_peak_live_bytes\":{},\"device_live_bytes\":{},\
         \"allocation_count\":{},\"host_peak_rss_bytes\":{},\
         \"staging_upload_bytes\":{},\"staging_readback_bytes\":{},\"loadavg\":\"{}\"}}",
        summary.upload_bytes,
        summary.readback_bytes,
        summary.executions,
        summary.submissions,
        summary.input_readbacks,
        memory_after
            .peak_live_bytes
            .max(memory_before.peak_live_bytes),
        memory_after.live_bytes,
        memory_after.allocation_count,
        peak_rss_bytes(),
        staging_after.upload_bytes - staging_before.upload_bytes,
        staging_after.readback_bytes - staging_before.readback_bytes,
        load,
    );
    println!("[fn3-26][json] {json}");

    println!("[fn3-26][csv] sample,kind,ms");
    for (index, sample) in ttft.iter().enumerate() {
        println!("[fn3-26][csv] {index},ttft,{sample:.4}");
    }
    for (index, sample) in prefill.iter().enumerate() {
        println!("[fn3-26][csv] {index},prefill,{sample:.4}");
    }
    for (index, sample) in decode.iter().enumerate() {
        println!("[fn3-26][csv] {index},decode,{sample:.4}");
    }

    let (errors, warnings) = fastnn::fn3::backend::vulkan::instance::validation_counts();
    assert_eq!(
        (errors, warnings),
        (0, 0),
        "the measured run must be validation-clean"
    );
}
