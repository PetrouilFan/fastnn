//! FN3-21 alpha follow-up (#67): the matched release-build timing baseline.
//!
//! The alpha handoff quoted the 3.0 harness (`fn3-bench --suite model --quick`, removed with the 2.x surface) numbers measured
//! on a debug build with unmatched backends. This binary is the replacement
//! measurement: it drives the **production** [`Session`] path — the same call
//! sequence the CLI and the Python facade use — through one concrete backend at
//! a time, with a fixed workload and a fixed number of warmups and iterations.
//!
//! ```bash
//! FASTNN_FN3_ONNX_MODEL=/path/to/model.onnx FN3_BENCH_BACKEND=cpu \
//!   FN3_BENCH_OUT=docs/reports/fn3-21-alpha-followup/release-cpu.json \
//!   cargo test --release --test fn3_alpha_bench -- --nocapture --test-threads=1
//!
//! FN3_VK_DEVICE_UUID=<uuid> FN3_VK_REQUIRE_PHYSICAL=1 FN3_BENCH_BACKEND=vulkan \
//!   FN3_BENCH_OUT=docs/reports/fn3-21-alpha-followup/release-vulkan.json \
//!   cargo test --release --features vulkan --test fn3_alpha_bench -- \
//!   --nocapture --test-threads=1
//! ```
//!
//! # What is timed, exactly
//!
//! * **Host wall clock**: a monotonic `Instant` around `Session::prefill` /
//!   `Session::decode` *plus* the acquisition of the returned logits slice. The
//!   alpha `Tensor` is host memory on every device, so "logits available" is
//!   the host slice of a tensor whose every instruction already completed: the
//!   Vulkan submission path waits on a fence per instruction. No readback is
//!   needed to observe them, which is why the residency ledger also reports
//!   `input_readbacks == 0`.
//! * **Device time**, Vulkan only: the sum of the per-instruction timestamp
//!   queries recorded by the residency ledger (`gpu_ns`) for the phase. It
//!   measures kernel execution, not the host gaps between submissions. The CPU
//!   backend has no device clock and says so; it is never zero-filled.
//! * **Counters**: upload bytes, readback bytes, queue submissions, executions
//!   and proven device executions are ledger *deltas* around each phase.
//! * **Memory**: peak host RSS from `/proc/self/status` `VmHWM`; Vulkan peak
//!   device memory from the allocator's `MemoryStats::peak_live_bytes`. A
//!   device-local-only high-water mark is not exposed at this commit; the JSON
//!   records that as `unavailable` and names the missing accessor.
//!
//! # Workload
//!
//! Prompt 10 tokens, then three cached decode steps (`640, 11, 612`), so the
//! context after each step is 11, 12 and 13. Every iteration resets the session
//! and replays the same shapes, which keeps the plan cache warm (four plans:
//! prefill at `past = 0` and decode at `past = 10, 11, 12`) and makes every
//! measured prefill a `(10, 0)` prefill.
//!
//! Profile is recorded, not assumed: a debug run writes `"evidence": false`
//! and prints that its numbers are not release evidence. Only `--release` runs
//! may be cited.
//!
//! The pinned model is not committed (upstream declares no licence). The test
//! verifies the file against `fixtures/parity/tiny-gpt2/model.lock.json` —
//! bytes and SHA-256 are read from that file, never copied from a brief — and
//! prints a skip reason when the file is absent.

#[allow(unused)]
#[path = "fn3_e2e/harness.rs"]
mod harness;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fastnn::inference::import::{import_onnx_model, ModelImport};
use fastnn::inference::{Backend, DeviceId, Session};
use serde_json::{json, Value};

#[cfg(feature = "vulkan")]
use fastnn::inference::backend::vulkan::{
    adapters, instance::validation_counts, VulkanBackend, VulkanConfig,
};

/// Measured iterations. Fixed: a committed table must state the same `n`.
const ITERATIONS: usize = 20;
/// Untimed warmup iterations before the measured loop.
const WARMUPS: usize = 3;
/// KV capacity: 10 prompt tokens + 3 decode tokens.
const KV_CAPACITY: u64 = 13;

const PROMPT: [u32; 10] = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13];
const DECODE_TOKENS: [u32; 3] = [640, 11, 612];

/// The pinned prompt's top-5 token IDs, as asserted by
/// `tools/ci/wheel_smoke.py` and the FN3-21 handoff.
const EXPECTED_TOP5: [usize; 5] = [46737, 28597, 16571, 7868, 13];

fn model_path() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(path) = std::env::var("FASTNN_FN3_ONNX_MODEL") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tools/parity/.cache/models/tiny-gpt2/onnx/model.onnx"),
    );
    candidates.into_iter().find(|path| path.is_file())
}

/// The pinned `onnx/model.onnx` entry from the committed lock file.
fn lock_entry() -> (u64, String, String) {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/tiny-gpt2/model.lock.json");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("model lock {}: {error}", path.display()));
    let lock: Value = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("model lock is not JSON: {error}"));
    let files = lock
        .get("files")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("model lock has no `files` array"));
    let entry = files
        .iter()
        .find(|entry| entry.get("path").and_then(Value::as_str) == Some("onnx/model.onnx"))
        .unwrap_or_else(|| panic!("model lock names no `onnx/model.onnx` entry"));
    let bytes = entry
        .get("bytes")
        .and_then(Value::as_u64)
        .expect("lock entry has `bytes`");
    let sha256 = entry
        .get("sha256")
        .and_then(Value::as_str)
        .expect("lock entry has `sha256`")
        .to_string();
    let revision = lock
        .get("revision")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    (bytes, sha256, revision)
}

fn requested_backends() -> Vec<&'static str> {
    // Without the `vulkan` feature the only buildable backend is the CPU, so
    // `cargo test --test fn3_alpha_bench` is a complete run; an explicit
    // `FN3_BENCH_BACKEND=vulkan` on such a build is an error, below.
    let default = if cfg!(feature = "vulkan") {
        "both"
    } else {
        "cpu"
    };
    let raw = std::env::var("FN3_BENCH_BACKEND").unwrap_or_else(|_| default.to_string());
    match raw.trim().to_ascii_lowercase().as_str() {
        "cpu" => vec!["cpu"],
        "vulkan" | "gpu" => vec!["vulkan"],
        "both" => vec!["cpu", "vulkan"],
        other => panic!("FN3_BENCH_BACKEND=`{other}` is not `cpu`, `vulkan` or `both`"),
    }
}

#[cfg(feature = "vulkan")]
fn require_physical() -> bool {
    std::env::var("FN3_VK_REQUIRE_PHYSICAL").as_deref() == Ok("1")
}

#[cfg(feature = "vulkan")]
fn pinned_uuid() -> Option<String> {
    let value = std::env::var("FN3_VK_DEVICE_UUID").ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_ascii_lowercase())
}

// ---------------------------------------------------------------------------
// Small statistics
// ---------------------------------------------------------------------------

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite samples"));
    sorted
}

/// Nearest-rank percentile: rank `ceil(p/100 * n)`, 1-based.
fn percentile(values: &[f64], p: f64) -> f64 {
    let sorted = sorted(values);
    let n = sorted.len();
    let rank = ((p / 100.0) * n as f64).ceil() as usize;
    sorted[rank.clamp(1, n) - 1]
}

fn summary_of(values: &[f64]) -> Value {
    let ordered = sorted(values);
    json!({
        "n": values.len(),
        "min": ordered[0],
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "max": ordered[ordered.len() - 1],
        "mean": values.iter().sum::<f64>() / values.len() as f64,
        "percentile_rule": "nearest-rank, rank = ceil(p/100 * n), 1-based over sorted samples",
        "samples_ms": values,
    })
}

fn rss_bytes() -> (u64, u64) {
    let text = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let read = |key: &str| -> u64 {
        text.lines()
            .find(|line| line.starts_with(key))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
            .map(|kilobytes| kilobytes * 1024)
            .unwrap_or(0)
    };
    // VmHWM is the process-wide high-water mark; VmRSS is the current size.
    (read("VmHWM:"), read("VmRSS:"))
}

fn loadavg() -> String {
    fs::read_to_string("/proc/loadavg")
        .map(|text| text.trim().to_string())
        .unwrap_or_else(|error| format!("unavailable: {error}"))
}

fn cpu_model() -> String {
    let text = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    text.lines()
        .find(|line| line.starts_with("model name"))
        .and_then(|line| line.split(':').nth(1))
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.truncate(k);
    order
}

// ---------------------------------------------------------------------------
// Residency deltas
// ---------------------------------------------------------------------------

/// The subset of `ResidencySummary` this binary consumes.
#[derive(Clone, Copy)]
struct Ledger {
    executions: u64,
    submissions: u64,
    upload_bytes: u64,
    readback_bytes: u64,
    input_readbacks: u64,
    proven_device_executions: u64,
    gpu_ns: Option<u64>,
}

/// One timed phase's counter delta; `None` on a backend without a ledger.
struct PhaseCounters {
    upload_bytes: u64,
    readback_bytes: u64,
    submissions: u64,
    executions: u64,
    proven_device_executions: u64,
    input_readbacks: u64,
    device_ms: Option<f64>,
}

impl PhaseCounters {
    fn to_json(&self) -> Value {
        json!({
            "upload_bytes": self.upload_bytes,
            "readback_bytes": self.readback_bytes,
            "submissions": self.submissions,
            "executions": self.executions,
            "proven_device_executions": self.proven_device_executions,
            "input_readbacks": self.input_readbacks,
        })
    }
}

#[cfg(feature = "vulkan")]
fn ledger(vk: Option<&VulkanBackend>) -> Option<Ledger> {
    vk.map(|backend| {
        let summary = backend.residency().summary();
        Ledger {
            executions: summary.executions,
            submissions: summary.submissions,
            upload_bytes: summary.upload_bytes,
            readback_bytes: summary.readback_bytes,
            input_readbacks: summary.input_readbacks,
            proven_device_executions: summary.proven_device_executions,
            gpu_ns: summary.gpu_ns,
        }
    })
}

#[cfg(not(feature = "vulkan"))]
fn ledger(_vk: Option<&()>) -> Option<Ledger> {
    None
}

fn delta(before: Option<Ledger>, after: Option<Ledger>) -> Option<PhaseCounters> {
    let (before, after) = (before?, after?);
    Some(PhaseCounters {
        upload_bytes: after.upload_bytes - before.upload_bytes,
        readback_bytes: after.readback_bytes - before.readback_bytes,
        submissions: after.submissions - before.submissions,
        executions: after.executions - before.executions,
        proven_device_executions: after.proven_device_executions - before.proven_device_executions,
        input_readbacks: after.input_readbacks - before.input_readbacks,
        device_ms: match (before.gpu_ns, after.gpu_ns) {
            (Some(start), Some(end)) => Some((end - start) as f64 / 1e6),
            _ => None,
        },
    })
}

/// Counters for a backend with no residency ledger (the CPU).
fn no_counters() -> Value {
    json!({
        "status": "unavailable",
        "reason": "the CPU backend keeps no residency ledger: it has no queue \
                   submissions, no upload/readback transfers and no device timestamps",
    })
}

/// The device-time summary of a phase, from its per-iteration samples.
fn device_summary(samples: &[f64], expected: usize) -> Value {
    if samples.len() != expected {
        return json!({
            "status": "unavailable",
            "reason": format!(
                "only {}/{} phases reported a device timestamp for every record; the backend \
                 has no device clock for the rest",
                samples.len(),
                expected
            ),
        });
    }
    let mut summary = summary_of(samples);
    summary["status"] = Value::from("measured");
    summary["source"] =
        Value::from("sum of per-instruction Vulkan timestamp queries (residency ledger gpu_ns)");
    summary
}

fn counters_json(counters: &Option<PhaseCounters>) -> Value {
    counters
        .as_ref()
        .map_or_else(no_counters, PhaseCounters::to_json)
}

// ---------------------------------------------------------------------------
// Device contexts
// ---------------------------------------------------------------------------

struct DeviceContext<'a> {
    label: &'a str,
    device: DeviceId,
    backend: &'a dyn Backend,
    #[cfg(feature = "vulkan")]
    vulkan: Option<&'a VulkanBackend>,
}

// ---------------------------------------------------------------------------
// The measurement
// ---------------------------------------------------------------------------

struct PhaseRecord {
    host_ms: Vec<f64>,
    device_ms: Vec<f64>,
    counters: Vec<Option<PhaseCounters>>,
}

impl PhaseRecord {
    fn new() -> Self {
        Self {
            host_ms: Vec::new(),
            device_ms: Vec::new(),
            counters: Vec::new(),
        }
    }

    fn push(&mut self, host_ms: f64, counters: Option<PhaseCounters>) {
        if let Some(device_ms) = counters.as_ref().and_then(|counters| counters.device_ms) {
            self.device_ms.push(device_ms);
        }
        self.counters.push(counters);
        self.host_ms.push(host_ms);
    }
}

fn measure(context: &DeviceContext<'_>, import: &ModelImport) -> Value {
    let prepare_started = Instant::now();
    let (model, _envelope) = harness::prepared(import, context.device, KV_CAPACITY);
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;

    let mut session =
        Session::with_runtime(&model, import, context.backend).expect("session for the bench");

    #[cfg(feature = "vulkan")]
    let vk = context.vulkan;
    #[cfg(not(feature = "vulkan"))]
    let vk: Option<&()> = None;

    #[cfg(feature = "vulkan")]
    let validation_before = vk.map(|_| validation_counts());

    let (rss_before, _) = rss_bytes();

    // Untimed warmups: the first prefill is the first plan, so it is not a
    // measured sample.
    for _ in 0..WARMUPS {
        session.reset();
        session.prefill(&PROMPT).expect("warmup prefill");
        for token in DECODE_TOKENS {
            session.decode(&[token]).expect("warmup decode");
        }
    }
    let (warmup_hits, warmup_misses, warmup_evictions) = (
        session.plan_hits(),
        session.plan_misses(),
        session.plan_evictions(),
    );

    let mut prefill = PhaseRecord::new();
    let mut decode = [PhaseRecord::new(), PhaseRecord::new(), PhaseRecord::new()];
    let mut top5_seen: Option<Vec<usize>> = None;
    let mut decode_top1: Vec<Vec<usize>> = vec![Vec::new(), Vec::new(), Vec::new()];

    for _ in 0..ITERATIONS {
        session.reset();

        let before = ledger(vk);
        let started = Instant::now();
        let logits = session.prefill(&PROMPT).expect("measured prefill");
        let values = logits.as_f32_slice().expect("host logits");
        let host_ms = started.elapsed().as_secs_f64() * 1e3;
        let after = ledger(vk);
        let top5 = top_k(values, 5);
        match &top5_seen {
            Some(previous) => assert_eq!(
                &top5, previous,
                "the measured prefill changed the top-5 tokens"
            ),
            None => top5_seen = Some(top5),
        }
        prefill.push(host_ms, delta(before, after));

        for (index, token) in DECODE_TOKENS.iter().enumerate() {
            let before = ledger(vk);
            let started = Instant::now();
            let logits = session.decode(&[*token]).expect("measured decode");
            let values = logits.as_f32_slice().expect("host logits");
            let host_ms = started.elapsed().as_secs_f64() * 1e3;
            let after = ledger(vk);
            decode[index].push(host_ms, delta(before, after));
            decode_top1[index].push(harness::argmax(values));
        }
    }

    let top5 = top5_seen.expect("at least one measured prefill");
    assert_eq!(
        top5, EXPECTED_TOP5,
        "the installed path did not reproduce the pinned top-5"
    );
    for (index, observed) in decode_top1.iter().enumerate() {
        let first = *observed.first().expect("at least one measured decode");
        assert!(
            observed.iter().all(|value| *value == first),
            "decode step {} changed its argmax across iterations: {observed:?}",
            index + 1
        );
    }

    // The timed path must be the device path: every instruction executed,
    // every execution proven, and no input read back to the host.
    for (label, record) in std::iter::once(("prefill", &prefill)).chain(
        decode.iter().enumerate().map(|(index, record)| {
            let label: &'static str = ["decode 1", "decode 2", "decode 3"][index];
            (label, record)
        }),
    ) {
        for counters in record.counters.iter().flatten() {
            assert_eq!(counters.input_readbacks, 0, "{label}: input readback");
            assert_eq!(
                counters.proven_device_executions, counters.executions,
                "{label}: an execution was not proven on the device"
            );
        }
    }

    // Memory before the session is dropped: the KV buffers and the slot pool
    // are live, which is the state a running decode has.
    let (peak_rss, rss_now) = rss_bytes();
    #[cfg(feature = "vulkan")]
    let stats_live = vk.map(|backend| backend.memory_stats());

    let plan_counters = json!({
        "hits_after_warmup": warmup_hits,
        "misses_after_warmup": warmup_misses,
        "evictions_after_warmup": warmup_evictions,
        "hits_total": session.plan_hits(),
        "misses_total": session.plan_misses(),
        "evictions_total": session.plan_evictions(),
        "cached": session.plan_cache_len(),
        "kv_bytes": session.kv_bytes(),
        "slot_allocations": session.slot_allocations(),
        "slot_reuses": session.slot_reuses(),
    });

    drop(session);
    #[cfg(feature = "vulkan")]
    let stats_dropped = vk.map(|backend| backend.memory_stats());

    let validation = {
        #[cfg(feature = "vulkan")]
        {
            vk.map(|_| {
                let (errors, warnings) = validation_counts();
                json!({
                    "errors": errors,
                    "warnings": warnings,
                    "delta": match validation_before {
                        Some((before_errors, before_warnings)) => format!(
                            "{}/{}",
                            errors as i64 - before_errors as i64,
                            warnings as i64 - before_warnings as i64
                        ),
                        None => "unavailable".to_string(),
                    },
                })
            })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            None::<Value>
        }
    };

    if let Some(validation) = &validation {
        assert_eq!(
            validation.get("errors").and_then(Value::as_u64),
            Some(0),
            "the validation layer recorded an error"
        );
        assert_eq!(
            validation.get("warnings").and_then(Value::as_u64),
            Some(0),
            "the validation layer recorded a warning"
        );
    }

    // A representative chain's counters, so the report can state one chain's
    // transfer volume as well as the twenty-iteration total.
    let chain: Vec<&PhaseCounters> = std::iter::once(prefill.counters[0].as_ref())
        .chain(decode.iter().map(|record| record.counters[0].as_ref()))
        .flatten()
        .collect();
    let sum = |pick: fn(&PhaseCounters) -> u64| -> u64 { chain.iter().map(|c| pick(c)).sum() };
    let total = |pick: fn(&PhaseCounters) -> u64| -> u64 {
        std::iter::once(&prefill)
            .chain(decode.iter())
            .flat_map(|record| record.counters.iter().flatten())
            .map(pick)
            .sum()
    };

    let device_memory = {
        #[cfg(feature = "vulkan")]
        {
            stats_live.map(|stats| {
                json!({
                    "peak_device_allocator_live_bytes": stats.peak_live_bytes,
                    "peak_device_local_bytes": {
                        "status": "unavailable",
                        "reason": "MemoryStats exposes only peak_live_bytes (all memory types); \
                                   there is no separate high-water mark for live_device_local_bytes. \
                                   Add `MemoryStats::peak_live_device_local_bytes` to report it.",
                    },
                    "live_bytes_with_session": stats.live_bytes,
                    "live_device_local_bytes": stats.live_device_local_bytes,
                    "live_host_visible_bytes": stats.live_host_visible_bytes,
                    "reserved_bytes": stats.reserved_bytes,
                    "chunks": stats.chunks,
                    "allocation_count": stats.allocation_count,
                    "free_count": stats.free_count,
                    "live_bytes_after_session_drop": stats_dropped.map(|after| after.live_bytes),
                    "accessor": "VulkanBackend::memory_stats() -> MemoryStats",
                })
            })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            None::<Value>
        }
    };

    let decode_all: Vec<f64> = decode
        .iter()
        .flat_map(|record| record.host_ms.iter().copied())
        .collect();
    let decode_device_all: Vec<f64> = decode
        .iter()
        .flat_map(|record| record.device_ms.iter().copied())
        .collect();
    let device_memory_present = device_memory.is_some();

    json!({
        "label": context.label,
        "device": context.device.to_string(),
        "backend": context.backend.name(),
        "prepare": {
            "ms": prepare_ms,
            "n": 1,
            "what": "Planner + envelope + host weights + PreparedModel; a single shot, not a percentile",
        },
        "prefill": {
            "host": summary_of(&prefill.host_ms),
            "tokens_per_second_p50": PROMPT.len() as f64 / (percentile(&prefill.host_ms, 50.0) / 1e3),
            "device": device_summary(&prefill.device_ms, ITERATIONS),
            "counters": counters_json(&prefill.counters[0]),
        },
        "decode": {
            "host": summary_of(&decode_all),
            "tokens_per_second_p50": 1.0 / (percentile(&decode_all, 50.0) / 1e3),
            "per_step": DECODE_TOKENS.iter().enumerate().map(|(index, token)| {
                json!({
                    "step": index + 1,
                    "token": token,
                    "past": 10 + index as u64,
                    "context_after_step": 11 + index as u64,
                    "host": summary_of(&decode[index].host_ms),
                    "device": device_summary(&decode[index].device_ms, ITERATIONS),
                    "counters": counters_json(&decode[index].counters[0]),
                })
            }).collect::<Vec<_>>(),
            "device_ms_per_iteration": if decode_device_all.len() == decode_all.len() {
                Value::from(decode_device_all.iter().sum::<f64>() / ITERATIONS as f64)
            } else {
                json!({
                    "status": "unavailable",
                    "reason": "a decode step reported no device timestamp",
                })
            },
            "argmax_per_step": decode_top1,
        },
        "transfers": {
            "status": if device_memory_present { "measured" } else { "unavailable" },
            "one_chain": {
                "upload_bytes": sum(|c| c.upload_bytes),
                "readback_bytes": sum(|c| c.readback_bytes),
                "submissions": sum(|c| c.submissions),
                "what": "prefill(10) + three cached decode steps, iteration 1",
            },
            "measured_total": {
                "upload_bytes": total(|c| c.upload_bytes),
                "readback_bytes": total(|c| c.readback_bytes),
                "submissions": total(|c| c.submissions),
                "executions": total(|c| c.executions),
                "proven_device_executions": total(|c| c.proven_device_executions),
                "input_readbacks": total(|c| c.input_readbacks),
                "iterations": ITERATIONS,
            },
        },
        "plan": plan_counters,
        "memory": {
            "peak_host_rss_bytes": peak_rss,
            "rss_before_run_bytes": rss_before,
            "rss_at_end_bytes": rss_now,
            "host_source": "/proc/self/status VmHWM (process high-water) and VmRSS",
            "device": device_memory.unwrap_or_else(|| json!({
                "status": "unavailable",
                "reason": "the CPU backend has no device allocator; device memory is not a \
                           metric it can report and it is never zero-filled",
            })),
        },
        "validation": validation.unwrap_or_else(|| json!({
            "status": "unavailable",
            "reason": "no Vulkan device in this run",
        })),
        "correctness": {
            "prefill_top5": top5,
            "expected_top5": EXPECTED_TOP5,
            "decode_argmax_per_step": decode_top1,
        },
    })
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[test]
fn alpha_release_baseline() {
    let Some(path) = model_path() else {
        let reason = "the pinned 27,897,974-byte ONNX model is absent: set \
                      FASTNN_FN3_ONNX_MODEL or run `python3 tools/parity/fetch_models.py \
                      --suite tiny-gpt2`";
        println!("[fn3bench][skip] {reason}");
        if std::env::var("FN3_BENCH_REQUIRE_MODEL").as_deref() == Ok("1") {
            panic!("FN3_BENCH_REQUIRE_MODEL=1 but {reason}");
        }
        return;
    };

    let (want_bytes, want_sha256, revision) = lock_entry();
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(
        bytes.len() as u64,
        want_bytes,
        "{} is not the pinned graph",
        path.display()
    );
    let sha256 = harness::sha256(&bytes);
    assert_eq!(
        sha256,
        want_sha256,
        "{} has the pinned length but not the pinned bytes",
        path.display()
    );
    let load_started = Instant::now();
    let import = import_onnx_model(&bytes)
        .unwrap_or_else(|error| panic!("the pinned model failed to import:\n{error}"));
    let load_ms = load_started.elapsed().as_secs_f64() * 1e3;
    drop(bytes);

    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let evidence = profile == "release";
    if !evidence {
        println!(
            "[fn3bench][warn] this is a debug build: the timings below are NOT release \
             evidence and the JSON records evidence=false"
        );
    }

    let loadavg_start = loadavg();
    let started_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let mut backends: Vec<Value> = Vec::new();

    for requested in requested_backends() {
        match requested {
            "cpu" => {
                let backend = fastnn::inference::CpuBackend::new();
                let context = DeviceContext {
                    label: "cpu",
                    device: DeviceId::Cpu,
                    backend: &backend,
                    #[cfg(feature = "vulkan")]
                    vulkan: None,
                };
                backends.push(measure(&context, &import));
            }
            "vulkan" => {
                #[cfg(feature = "vulkan")]
                {
                    match open_physical() {
                        Ok(backend) => {
                            let adapter = backend.adapter();
                            let label: &'static str = "vulkan";
                            let context = DeviceContext {
                                label,
                                device: backend.device_id(),
                                backend: &*backend,
                                vulkan: Some(&*backend),
                            };
                            let mut measured = measure(&context, &import);
                            measured["adapter"] = json!({
                                "name": adapter.name,
                                "uuid": adapter.uuid_string(),
                                "api_version": fastnn::inference::backend::vulkan::device_info::format_version(adapter.api_version),
                                "driver_version": adapter.driver_version_string(),
                                "software": backend.is_software(),
                                "validation_enabled_in_code": backend.validation_enabled(),
                            });
                            backends.push(measured);
                        }
                        Err(reason) => {
                            println!("[fn3bench][unavailable] vulkan: {reason}");
                            if require_physical() {
                                panic!(
                                    "FN3_VK_REQUIRE_PHYSICAL=1 but vulkan is unavailable: {reason}"
                                );
                            }
                            backends.push(json!({
                                "label": "vulkan",
                                "status": "unavailable",
                                "reason": reason,
                            }));
                        }
                    }
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    panic!(
                        "FN3_BENCH_BACKEND=vulkan but this binary was built without the `vulkan` \
                         feature; rebuild with `--features vulkan`"
                    );
                }
            }
            other => panic!("unexpected backend `{other}`"),
        }
    }

    let (peak_rss, rss_now) = rss_bytes();
    let record = json!({
        "schema": 1,
        "report": "fn3-21-alpha-followup",
        "purpose": "matched release-build baseline for the FastNN 3.0 alpha handoff",
        "command": std::env::var("FN3_BENCH_COMMAND").unwrap_or_else(|_| "unavailable".to_string()),
        "generated_unix_s": started_unix,
        "profile": profile,
        "evidence": evidence,
        "features": {
            "vulkan": cfg!(feature = "vulkan"),
        },
        "crate_version": env!("CARGO_PKG_VERSION"),
        "host": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "logical_cpus": std::thread::available_parallelism().map(|value| value.get()).unwrap_or(0),
            "cpu_model": cpu_model(),
            "loadavg_start": loadavg_start,
            "loadavg_end": loadavg(),
            "peak_host_rss_bytes": peak_rss,
            "rss_at_end_bytes": rss_now,
        },
        "model": {
            "path": path.display().to_string(),
            "bytes": want_bytes,
            "sha256": sha256,
            "revision": revision,
            "lock": "fixtures/parity/tiny-gpt2/model.lock.json",
            "load_ms": load_ms,
            "load_what": "read the ONNX file, verify it against the lock, parse/import the graph; single shot",
            "geometry": {
                "layers": import.geometry().layers,
                "hidden_size": import.geometry().hidden_size,
                "mlp_hidden_size": import.geometry().mlp_hidden_size,
                "query_heads": import.geometry().query_heads,
                "kv_heads": import.geometry().kv_heads,
                "head_dim": import.geometry().head_dim,
                "vocab_size": import.geometry().vocab_size,
            },
        },
        "workload": {
            "prompt_tokens": PROMPT,
            "prompt_length": PROMPT.len(),
            "decode_tokens": DECODE_TOKENS,
            "decode_context_lengths_after_each_step": [11, 12, 13],
            "kv_capacity_tokens": KV_CAPACITY,
            "warmup_iterations": WARMUPS,
            "measured_iterations": ITERATIONS,
            "session_reset_between_iterations": true,
            "reset_note": "reset() keeps the KV buffers and the four cached plans and clears \
                           the value-slot pool; every measured prefill therefore runs at \
                           (seq=10, past=0) and every measured decode at past=10,11,12",
        },
        "timing": {
            "host_clock": "std::time::Instant (monotonic) around Session::prefill/decode plus \
                           acquisition of the returned logits slice; top-k is computed outside \
                           the timed region",
            "host_clock_covers": "call entry to logits available on the host",
            "device_clock": "Vulkan: sum of per-instruction timestamp queries recorded by the \
                             residency ledger (gpu_ns), so kernel time only, not host submission \
                             gaps or fences. CPU: unavailable, the backend has no device clock.",
            "iterations": ITERATIONS,
            "warmups": WARMUPS,
        },
        "backends": backends,
    });

    let json = serde_json::to_string_pretty(&record).expect("the record is JSON");
    if let Ok(out) = std::env::var("FN3_BENCH_OUT") {
        let out = PathBuf::from(out);
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent).expect("create the report directory");
        }
        fs::write(&out, format!("{json}\n")).expect("write the benchmark record");
        println!("[fn3bench] record written to {}", out.display());
    }
    println!("{json}");
}

// ---------------------------------------------------------------------------
// The physical-device gate (same selection rules as tests/fn3_e2e/gpu.rs)
// ---------------------------------------------------------------------------

#[cfg(feature = "vulkan")]
fn open_physical() -> Result<std::sync::Arc<VulkanBackend>, String> {
    use fastnn::inference::device::DeviceRequest;

    let pinned = pinned_uuid();
    let adapter = adapters()
        .iter()
        .find(|adapter| {
            adapter.has_compute_queue()
                && !adapter.is_software()
                && pinned
                    .as_deref()
                    .map(|uuid| adapter.uuid_string().eq_ignore_ascii_case(uuid))
                    .unwrap_or(true)
        })
        .ok_or_else(|| {
            format!(
                "no physical Vulkan device with a compute queue is present{}",
                match pinned {
                    Some(uuid) => format!(" (pinned uuid {uuid})"),
                    None => String::new(),
                }
            )
        })?;
    let backend = VulkanBackend::open_with(
        &DeviceRequest::Vulkan {
            index: Some(adapter.index),
        },
        VulkanConfig::validated(),
    )
    .map_err(|error| format!("adapter {} could not be opened: {error}", adapter.index))?;
    if backend.is_software() {
        return Err("the bench refuses a software ICD".to_string());
    }
    if !backend.validation_enabled() {
        return Err("the bench requires the validation layer enabled in code".to_string());
    }
    Ok(backend)
}
