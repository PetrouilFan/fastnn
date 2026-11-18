//! Versioned, machine-readable result schema for the FN3 benchmark harness.
//!
//! `SCHEMA_VERSION` changes only on a breaking change. Additive fields keep the
//! same version; consumers must ignore unknown fields (`additionalProperties`
//! is permissive in the companion JSON Schema).
//!
//! Design rules encoded here:
//! - Nothing is omitted or zero-filled. A measurement that could not be taken
//!   is a metric with `status: "unavailable"` and a human-readable `reason`.
//! - Host timing and device timing are separate series. `host_time` is the
//!   primary measured series (host monotonic wall clock for timing metrics);
//!   `device_time` is populated only when the backend reports device clocks.
//! - Units are explicit per metric (`ms`, `s`, `tokens/s`, `MiB`, `count`).
//! - Raw samples are always retained; percentiles are a documented convenience.
//! - Environment, build configuration, model identity, input geometry, and
//!   precision are recorded with every run so comparisons can reject
//!   mismatched measurements.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA_VERSION: u32 = 1;
pub const SCHEMA_NAME: &str = "fastnn.fn3.bench.result";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunResult {
    pub schema_version: u32,
    pub schema_name: String,
    pub run: RunInfo,
    pub git: GitInfo,
    pub build: BuildInfo,
    pub host: HostInfo,
    pub gpu: GpuInfo,
    pub model: ModelInfo,
    pub geometry: Geometry,
    pub precision: Precision,
    pub metrics: Vec<Metric>,
    /// Suite-level caveats and methodology notes (KV cache, masking, reuse).
    pub notes: Vec<String>,
    /// Harness warnings that are not attached to a single metric (load guard,
    /// environment capture failures, adapter caveats).
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunInfo {
    /// Random identifier, unique per invocation (not derived from the seed).
    pub run_id: String,
    /// RFC 3339 UTC timestamp, second resolution.
    pub timestamp_utc: String,
    pub started_unix_s: u64,
    pub duration_s: f64,
    /// `baseline` (2.x runtime) or `fn3` (3.0 runtime).
    pub adapter: String,
    pub suite: String,
    pub quick: bool,
    /// Seed used for deterministic weight/input synthesis.
    pub seed: u64,
    /// Free-form operator label recorded with the run.
    pub label: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GitInfo {
    pub commit: Option<String>,
    pub branch: Option<String>,
    /// True when `git status --porcelain` reported any tracked or untracked
    /// change at start. `None` when git or the repository is unavailable.
    pub dirty: Option<bool>,
    pub dirty_entries: Option<u64>,
    pub describe: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildInfo {
    pub package_version: String,
    /// `release` when debug assertions are off, `debug` otherwise.
    pub profile: String,
    pub debug_assertions: bool,
    pub rustc_version: Option<String>,
    pub cargo_version: Option<String>,
    /// Compile-time feature flags that change the measured runtime.
    pub features: BTreeMap<String, bool>,
    pub target_arch: String,
    pub target_os: String,
    pub target_family: Option<String>,
    /// Exact argv of the benchmark process, for reproducibility.
    pub command_line: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostInfo {
    pub os: Option<String>,
    pub kernel: Option<String>,
    pub arch: String,
    pub cpu_model: Option<String>,
    pub physical_cores: Option<u64>,
    pub logical_cores: u64,
    /// Thread count the runtime was asked to use (RAYON_NUM_THREADS).
    pub threads_used: u64,
    pub memory_total_bytes: Option<u64>,
    pub load: LoadInfo,
    /// PIDs of `cargo` / `rustc` / `rustdoc` processes other than this run's
    /// own process tree. Non-empty means a competing build was running.
    pub competing_build_pids: Vec<u64>,
    /// PIDs of other live `fn3-bench` processes. Non-empty means another
    /// benchmark (possibly GPU) overlapped this one.
    pub competing_benchmark_pids: Vec<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoadInfo {
    pub one: Option<f64>,
    pub five: Option<f64>,
    pub fifteen: Option<f64>,
    /// Warning threshold for the 1-minute load average at start.
    pub threshold: f64,
    pub exceeded: bool,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    /// `present` when at least one Vulkan device was enumerated.
    pub status: String,
    pub reason: Option<String>,
    pub vulkan_instance_version: Option<String>,
    pub devices: Vec<GpuDevice>,
    /// Index into `devices` used for measured work, or null when no GPU was
    /// used for the reported metrics.
    pub selected_device_index: Option<usize>,
    /// Execution backend that produced the reported metrics.
    pub execution_backend: String,
    pub execution_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuDevice {
    pub index: usize,
    pub name: String,
    pub vendor: Option<String>,
    pub vendor_id: Option<String>,
    pub device_id: Option<String>,
    /// `physical`, `software`, or `unknown`.
    pub device_type: String,
    pub software_icd: bool,
    pub driver_name: Option<String>,
    pub driver_info: Option<String>,
    pub driver_version: Option<String>,
    pub api_version: Option<String>,
    pub device_uuid: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    /// Immutable revision for downloaded models; `synthetic:<seed>` or
    /// `not_applicable` otherwise.
    pub revision: String,
    /// Where the model identity comes from: `synthetic-random-weights`,
    /// `local-artifact`, `downloaded`.
    pub source: String,
    pub parameter_count: Option<u64>,
    pub weights_bytes: Option<u64>,
    pub seed: Option<u64>,
    pub checksums: Vec<Checksum>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checksum {
    /// What was hashed: `weights`, `graph`, `inputs`.
    pub subject: String,
    /// `fnv1a64` today (no cryptographic hash dependency in this revision).
    pub algorithm: String,
    pub value: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Geometry {
    pub batch: u64,
    /// Number of tokens in the prefill / time-to-first-token measurement.
    pub prompt_tokens: u64,
    /// Number of steady-state decode steps measured.
    pub decode_tokens: u64,
    pub context_capacity: u64,
    pub hidden_size: u64,
    pub num_heads: u64,
    /// `None` for multi-head attention where KV heads equal query heads.
    pub num_kv_heads: Option<u64>,
    pub head_dim: u64,
    pub num_layers: u64,
    pub intermediate_size: u64,
    pub vocab_size: u64,
    /// `full`, `causal`, `causal_kv_cache`, or `not_applicable`.
    pub attention: String,
    pub kv_cache: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Precision {
    pub weights_dtype: String,
    pub activations_dtype: String,
    pub accumulation_dtype: String,
    /// `None` for unquantized f32; otherwise e.g. `i8_weight_only`.
    pub quantization: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricStatus {
    Ok,
    Unavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Clock {
    /// Host monotonic clock (`Instant`), wall-clock duration of the work.
    HostMonotonic,
    /// Derived arithmetically from a latency series.
    Derived,
    /// Quantity metric (memory, allocation counts): no clock applies.
    NotApplicable,
}

/// A raw sample series plus its documented percentile set.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleBlock {
    pub unit: String,
    pub samples: Vec<f64>,
    pub sample_count: u64,
    /// Warmup iterations per measurement batch (not measured).
    pub warmup_count: u64,
    /// Number of independent measurement batches that produced `samples`.
    pub repetition_count: u64,
    pub stats: Option<crate::stats::Stats>,
}

impl SampleBlock {
    pub fn new(unit: &str, samples: Vec<f64>, warmup_count: u64, repetition_count: u64) -> Self {
        // NaN/Inf cannot be represented in JSON and are not usable measurements.
        // They are dropped here and become an explicitly unavailable metric when
        // nothing finite remains.
        let samples: Vec<f64> = samples.into_iter().filter(|v| v.is_finite()).collect();
        let stats = crate::stats::summarize(&samples);
        Self {
            unit: unit.to_string(),
            sample_count: samples.len() as u64,
            samples,
            warmup_count,
            repetition_count,
            stats,
        }
    }
}

/// Device-reported timing. Always present on a metric; `unavailable` carries
/// the reason instead of silently dropping the field.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceTime {
    pub status: MetricStatus,
    pub reason: Option<String>,
    /// Identifier of the reporting backend, e.g. `vulkan-timestamp-query`.
    pub source: Option<String>,
    pub samples: Vec<f64>,
    pub stats: Option<crate::stats::Stats>,
}

impl DeviceTime {
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            status: MetricStatus::Unavailable,
            reason: Some(reason.into()),
            source: None,
            samples: Vec::new(),
            stats: None,
        }
    }

    /// Record backend-reported device time. Kept for the 3.0 adapter, which is
    /// the first backend expected to report device clocks.
    #[allow(dead_code)]
    pub fn measured(source: impl Into<String>, samples: Vec<f64>) -> Self {
        let stats = crate::stats::summarize(&samples);
        let status = if stats.is_some() {
            MetricStatus::Ok
        } else {
            MetricStatus::Unavailable
        };
        Self {
            status,
            reason: if stats.is_some() {
                None
            } else {
                Some("backend reported no finite device-time samples".to_string())
            },
            source: Some(source.into()),
            samples,
            stats,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metric {
    /// Stable dotted name, e.g. `decode.per_token` or
    /// `microbench.mlp.fc.gemm_f32`.
    pub name: String,
    /// `load | prepare | prefill | decode | throughput | memory | allocation |
    /// microbench | harness`.
    pub category: String,
    pub unit: String,
    pub status: MetricStatus,
    pub reason: Option<String>,
    pub clock: Clock,
    /// Primary measured series. Populated for `ok` metrics even when the unit
    /// is not a time (memory/allocation), in which case `clock` is
    /// `not_applicable`.
    pub host_time: Option<SampleBlock>,
    pub device_time: DeviceTime,
    /// Runtime counters observed during the measured work (for example CPU
    /// arena telemetry). Values are totals across all samples.
    pub counters: BTreeMap<String, u64>,
    /// Measurement-specific context: shapes, FLOP counts, cache mode, etc.
    pub metadata: BTreeMap<String, String>,
}

impl Metric {
    pub fn unavailable(name: &str, category: &str, unit: &str, reason: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            category: category.to_string(),
            unit: unit.to_string(),
            status: MetricStatus::Unavailable,
            reason: Some(reason.into()),
            clock: Clock::NotApplicable,
            host_time: None,
            device_time: DeviceTime::unavailable(
                "measurement unavailable; no device timing was reported",
            ),
            counters: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn measured_time(
        name: &str,
        category: &str,
        samples_ms: Vec<f64>,
        warmup_count: u64,
        repetition_count: u64,
    ) -> Self {
        let block = SampleBlock::new("ms", samples_ms, warmup_count, repetition_count);
        if block.stats.is_none() {
            return Self::unavailable(
                name,
                category,
                "ms",
                "no finite timing samples were collected",
            );
        }
        Self {
            name: name.to_string(),
            category: category.to_string(),
            unit: "ms".to_string(),
            status: MetricStatus::Ok,
            reason: None,
            clock: Clock::HostMonotonic,
            host_time: Some(block),
            device_time: DeviceTime::unavailable(
                "CPU backend does not report per-operation device clocks",
            ),
            counters: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn measured_value(
        name: &str,
        category: &str,
        unit: &str,
        samples: Vec<f64>,
        warmup_count: u64,
        repetition_count: u64,
    ) -> Self {
        let block = SampleBlock::new(unit, samples, warmup_count, repetition_count);
        if block.stats.is_none() {
            return Self::unavailable(
                name,
                category,
                unit,
                "no finite value samples were collected",
            );
        }
        Self {
            name: name.to_string(),
            category: category.to_string(),
            unit: unit.to_string(),
            status: MetricStatus::Ok,
            reason: None,
            clock: Clock::NotApplicable,
            host_time: Some(block),
            device_time: DeviceTime::unavailable("quantity metric has no device-time counterpart"),
            counters: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Attach device-reported timing (used by the future GPU/fn3 adapters).
    #[allow(dead_code)]
    pub fn with_device_time(mut self, device_time: DeviceTime) -> Self {
        self.device_time = device_time;
        self
    }

    pub fn is_ok(&self) -> bool {
        self.status == MetricStatus::Ok
    }
}

impl ModelInfo {
    pub fn empty() -> Self {
        Self {
            name: "unset".to_string(),
            revision: "not_applicable".to_string(),
            source: "not_applicable".to_string(),
            parameter_count: None,
            weights_bytes: None,
            seed: None,
            checksums: Vec::new(),
        }
    }
}

impl RunResult {
    pub fn new(run: RunInfo) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            schema_name: SCHEMA_NAME.to_string(),
            run,
            git: GitInfo {
                commit: None,
                branch: None,
                dirty: None,
                dirty_entries: None,
                describe: None,
            },
            build: BuildInfo {
                package_version: env!("CARGO_PKG_VERSION").to_string(),
                profile: if cfg!(debug_assertions) {
                    "debug".to_string()
                } else {
                    "release".to_string()
                },
                debug_assertions: cfg!(debug_assertions),
                rustc_version: None,
                cargo_version: None,
                features: BTreeMap::new(),
                target_arch: std::env::consts::ARCH.to_string(),
                target_os: std::env::consts::OS.to_string(),
                target_family: None,
                command_line: std::env::args().collect(),
            },
            host: HostInfo {
                os: None,
                kernel: None,
                arch: std::env::consts::ARCH.to_string(),
                cpu_model: None,
                physical_cores: None,
                logical_cores: 1,
                threads_used: 1,
                memory_total_bytes: None,
                load: LoadInfo {
                    one: None,
                    five: None,
                    fifteen: None,
                    threshold: 2.0,
                    exceeded: false,
                    reason: None,
                },
                competing_build_pids: Vec::new(),
                competing_benchmark_pids: Vec::new(),
            },
            gpu: GpuInfo {
                status: "unavailable".to_string(),
                reason: Some("GPU probe did not run".to_string()),
                vulkan_instance_version: None,
                devices: Vec::new(),
                selected_device_index: None,
                execution_backend: "cpu".to_string(),
                execution_reason: Some("CPU measurement".to_string()),
            },
            model: ModelInfo::empty(),
            geometry: Geometry {
                batch: 1,
                prompt_tokens: 0,
                decode_tokens: 0,
                context_capacity: 0,
                hidden_size: 0,
                num_heads: 0,
                num_kv_heads: None,
                head_dim: 0,
                num_layers: 0,
                intermediate_size: 0,
                vocab_size: 0,
                attention: "not_applicable".to_string(),
                kv_cache: false,
            },
            precision: Precision {
                weights_dtype: "f32".to_string(),
                activations_dtype: "f32".to_string(),
                accumulation_dtype: "f32".to_string(),
                quantization: None,
            },
            metrics: Vec::new(),
            notes: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_metric_keeps_explicit_reason() {
        let metric = Metric::unavailable(
            "memory.device_peak",
            "memory",
            "MiB",
            "no GPU device selected",
        );
        assert_eq!(metric.status, MetricStatus::Unavailable);
        assert!(metric.reason.as_deref().is_some_and(|r| r.contains("GPU")));
        assert!(metric.host_time.is_none());
        assert_eq!(metric.device_time.status, MetricStatus::Unavailable);
        let json = serde_json::to_string(&metric).expect("serialize");
        assert!(json.contains("\"samples\":[]"));
    }

    #[test]
    fn run_result_round_trips() {
        let run = RunInfo {
            run_id: "abc".to_string(),
            timestamp_utc: "2026-11-18T00:00:00Z".to_string(),
            started_unix_s: 0,
            duration_s: 0.1,
            adapter: "baseline".to_string(),
            suite: "smoke".to_string(),
            quick: true,
            seed: 7,
            label: None,
        };
        let mut result = RunResult::new(run);
        result.metrics.push(Metric::measured_time(
            "model.load",
            "load",
            vec![1.0, 2.0, 3.0],
            0,
            3,
        ));
        let json = serde_json::to_string(&result).expect("serialize");
        let back: RunResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.schema_version, SCHEMA_VERSION);
        assert_eq!(back.metrics.len(), 1);
        assert_eq!(back.metrics[0].host_time.as_ref().unwrap().sample_count, 3);
    }

    #[test]
    fn non_finite_samples_are_dropped_not_serialized_as_nan() {
        let metric = Metric::measured_time("x", "microbench", vec![f64::NAN, 2.0], 0, 1);
        assert_eq!(metric.status, MetricStatus::Ok);
        assert_eq!(metric.host_time.as_ref().expect("block").samples, vec![2.0]);
        let json = serde_json::to_string(&metric).expect("serialize");
        assert!(!json.contains("NaN"));
    }

    #[test]
    fn all_non_finite_is_unavailable_with_reason() {
        let metric = Metric::measured_time("x", "microbench", vec![f64::NAN], 0, 1);
        assert_eq!(metric.status, MetricStatus::Unavailable);
        assert!(metric.reason.is_some());
    }
}
