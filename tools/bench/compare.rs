//! Compare two FN3 result JSON files.
//!
//! Refuses (exit code 2) to compare runs whose environment or input geometry
//! differ, unless `--force` is passed. This is the tool FN3-21/25/26/29 use to
//! turn two schema-v1 documents into a delta table.

use crate::schema::{Clock, Metric, MetricStatus, RunResult};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stat {
    P50,
    Mean,
    Min,
    P95,
    Max,
}

impl Stat {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "p50" | "median" => Some(Self::P50),
            "mean" => Some(Self::Mean),
            "min" => Some(Self::Min),
            "p95" => Some(Self::P95),
            "max" => Some(Self::Max),
            _ => None,
        }
    }

    fn of(self, metric: &Metric) -> Option<f64> {
        let host = metric.host_time.as_ref()?;
        let stats = host.stats.as_ref()?;
        match self {
            Self::P50 => Some(stats.p50),
            Self::Mean => Some(stats.mean),
            Self::Min => Some(stats.min),
            Self::P95 => Some(stats.p95),
            Self::Max => Some(stats.max),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Mismatch {
    /// `hard` mismatches block comparison; `soft` mismatches only warn.
    pub level: String,
    pub field: String,
    pub a: String,
    pub b: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Row {
    pub metric: String,
    pub category: String,
    pub unit: String,
    pub a: Option<f64>,
    pub b: Option<f64>,
    pub delta: Option<f64>,
    pub ratio: Option<f64>,
    pub a_status: String,
    pub b_status: String,
    pub a_samples: Option<u64>,
    pub b_samples: Option<u64>,
    pub note: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DiffReport {
    pub schema_version: u32,
    pub stat: String,
    pub a: RunSummary,
    pub b: RunSummary,
    pub mismatches: Vec<Mismatch>,
    pub blocked: bool,
    pub rows: Vec<Row>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunSummary {
    pub run_id: String,
    pub suite: String,
    pub adapter: String,
    pub quick: bool,
    pub commit: Option<String>,
    pub branch: Option<String>,
    pub dirty: Option<bool>,
    pub timestamp_utc: String,
}

fn compare_hard<T: PartialEq + std::fmt::Debug>(
    mismatches: &mut Vec<Mismatch>,
    name: &str,
    a: &T,
    b: &T,
) {
    if a != b {
        mismatches.push(Mismatch {
            level: "hard".to_string(),
            field: name.to_string(),
            a: format!("{a:?}"),
            b: format!("{b:?}"),
        });
    }
}

fn compare_soft<T: PartialEq + std::fmt::Debug>(
    mismatches: &mut Vec<Mismatch>,
    name: &str,
    a: &T,
    b: &T,
) {
    if a != b {
        mismatches.push(Mismatch {
            level: "soft".to_string(),
            field: name.to_string(),
            a: format!("{a:?}"),
            b: format!("{b:?}"),
        });
    }
}

pub fn compare(a: &RunResult, b: &RunResult, stat: Stat) -> DiffReport {
    let mut mismatches = Vec::new();

    compare_hard(
        &mut mismatches,
        "schema_version",
        &a.schema_version,
        &b.schema_version,
    );
    compare_hard(&mut mismatches, "run.suite", &a.run.suite, &b.run.suite);
    compare_hard(
        &mut mismatches,
        "geometry.batch",
        &a.geometry.batch,
        &b.geometry.batch,
    );
    compare_hard(
        &mut mismatches,
        "geometry.prompt_tokens",
        &a.geometry.prompt_tokens,
        &b.geometry.prompt_tokens,
    );
    compare_hard(
        &mut mismatches,
        "geometry.decode_tokens",
        &a.geometry.decode_tokens,
        &b.geometry.decode_tokens,
    );
    compare_hard(
        &mut mismatches,
        "geometry.context_capacity",
        &a.geometry.context_capacity,
        &b.geometry.context_capacity,
    );
    compare_hard(
        &mut mismatches,
        "geometry.hidden_size",
        &a.geometry.hidden_size,
        &b.geometry.hidden_size,
    );
    compare_hard(
        &mut mismatches,
        "geometry.num_heads",
        &a.geometry.num_heads,
        &b.geometry.num_heads,
    );
    compare_hard(
        &mut mismatches,
        "geometry.num_kv_heads",
        &a.geometry.num_kv_heads,
        &b.geometry.num_kv_heads,
    );
    compare_hard(
        &mut mismatches,
        "geometry.head_dim",
        &a.geometry.head_dim,
        &b.geometry.head_dim,
    );
    compare_hard(
        &mut mismatches,
        "geometry.num_layers",
        &a.geometry.num_layers,
        &b.geometry.num_layers,
    );
    compare_hard(
        &mut mismatches,
        "geometry.intermediate_size",
        &a.geometry.intermediate_size,
        &b.geometry.intermediate_size,
    );
    compare_hard(
        &mut mismatches,
        "geometry.vocab_size",
        &a.geometry.vocab_size,
        &b.geometry.vocab_size,
    );
    compare_hard(
        &mut mismatches,
        "geometry.attention",
        &a.geometry.attention,
        &b.geometry.attention,
    );
    compare_hard(
        &mut mismatches,
        "geometry.kv_cache",
        &a.geometry.kv_cache,
        &b.geometry.kv_cache,
    );
    compare_hard(
        &mut mismatches,
        "precision.weights_dtype",
        &a.precision.weights_dtype,
        &b.precision.weights_dtype,
    );
    compare_hard(
        &mut mismatches,
        "precision.activations_dtype",
        &a.precision.activations_dtype,
        &b.precision.activations_dtype,
    );
    compare_hard(
        &mut mismatches,
        "precision.accumulation_dtype",
        &a.precision.accumulation_dtype,
        &b.precision.accumulation_dtype,
    );
    compare_hard(
        &mut mismatches,
        "precision.quantization",
        &a.precision.quantization,
        &b.precision.quantization,
    );
    compare_hard(&mut mismatches, "model.name", &a.model.name, &b.model.name);
    compare_hard(
        &mut mismatches,
        "model.revision",
        &a.model.revision,
        &b.model.revision,
    );
    compare_hard(&mut mismatches, "model.seed", &a.model.seed, &b.model.seed);
    compare_hard(
        &mut mismatches,
        "host.cpu_model",
        &a.host.cpu_model,
        &b.host.cpu_model,
    );
    compare_hard(
        &mut mismatches,
        "host.logical_cores",
        &a.host.logical_cores,
        &b.host.logical_cores,
    );
    compare_hard(
        &mut mismatches,
        "host.threads_used",
        &a.host.threads_used,
        &b.host.threads_used,
    );
    compare_hard(
        &mut mismatches,
        "host.kernel",
        &a.host.kernel,
        &b.host.kernel,
    );
    compare_hard(
        &mut mismatches,
        "gpu.selected_device",
        &selected_gpu_name(a),
        &selected_gpu_name(b),
    );

    // Soft differences are useful context but legitimate across the 2.x -> 3.0
    // transition (new commit, different feature set, different load).
    compare_soft(&mut mismatches, "git.commit", &a.git.commit, &b.git.commit);
    compare_soft(&mut mismatches, "git.branch", &a.git.branch, &b.git.branch);
    compare_soft(&mut mismatches, "git.dirty", &a.git.dirty, &b.git.dirty);
    compare_soft(
        &mut mismatches,
        "build.profile",
        &a.build.profile,
        &b.build.profile,
    );
    compare_soft(
        &mut mismatches,
        "build.features",
        &format!("{:?}", a.build.features),
        &format!("{:?}", b.build.features),
    );
    compare_soft(
        &mut mismatches,
        "build.rustc_version",
        &a.build.rustc_version,
        &b.build.rustc_version,
    );
    compare_soft(
        &mut mismatches,
        "host.load.one",
        &format!("{:?}", a.host.load.one),
        &format!("{:?}", b.host.load.one),
    );
    compare_soft(
        &mut mismatches,
        "host.competing_build_pids",
        &a.host.competing_build_pids,
        &b.host.competing_build_pids,
    );

    let blocked = mismatches.iter().any(|m| m.level == "hard");

    let a_metrics: BTreeMap<&str, &Metric> = a
        .metrics
        .iter()
        .map(|metric| (metric.name.as_str(), metric))
        .collect();
    let b_metrics: BTreeMap<&str, &Metric> = b
        .metrics
        .iter()
        .map(|metric| (metric.name.as_str(), metric))
        .collect();
    let mut names: Vec<&str> = a_metrics.keys().chain(b_metrics.keys()).copied().collect();
    names.sort_unstable();
    names.dedup();

    let mut rows = Vec::new();
    for name in names {
        let a_metric = a_metrics.get(name).copied();
        let b_metric = b_metrics.get(name).copied();
        let a_value = a_metric.and_then(|metric| stat.of(metric));
        let b_value = b_metric.and_then(|metric| stat.of(metric));
        let a_status = status_of(a_metric);
        let b_status = status_of(b_metric);
        let delta = match (a_value, b_value) {
            (Some(a_value), Some(b_value)) => Some(b_value - a_value),
            _ => None,
        };
        let ratio = match (a_value, b_value) {
            (Some(a_value), Some(b_value)) if a_value != 0.0 => Some(b_value / a_value),
            _ => None,
        };
        let note = match (a_metric, b_metric) {
            (None, _) => Some("absent in A".to_string()),
            (_, None) => Some("absent in B".to_string()),
            (Some(a_metric), Some(b_metric)) => {
                if a_metric.status == MetricStatus::Unavailable
                    || b_metric.status == MetricStatus::Unavailable
                {
                    Some(
                        [
                            a_metric.reason.as_deref().map(|r| format!("A: {r}")),
                            b_metric.reason.as_deref().map(|r| format!("B: {r}")),
                        ]
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                        .join("; "),
                    )
                } else if a_metric.clock != b_metric.clock {
                    Some(format!(
                        "clock differs: {} vs {}",
                        clock_name(a_metric.clock),
                        clock_name(b_metric.clock)
                    ))
                } else {
                    None
                }
            }
        };
        let category = a_metric
            .or(b_metric)
            .map(|metric| metric.category.clone())
            .unwrap_or_default();
        let unit = a_metric
            .or(b_metric)
            .map(|metric| metric.unit.clone())
            .unwrap_or_default();
        let a_samples = a_metric
            .and_then(|metric| metric.host_time.as_ref())
            .map(|block| block.sample_count);
        let b_samples = b_metric
            .and_then(|metric| metric.host_time.as_ref())
            .map(|block| block.sample_count);
        rows.push(Row {
            metric: name.to_string(),
            category,
            unit,
            a: a_value,
            b: b_value,
            delta,
            ratio,
            a_status,
            b_status,
            a_samples,
            b_samples,
            note,
        });
    }

    DiffReport {
        schema_version: a.schema_version,
        stat: stat_name(stat).to_string(),
        a: summary(a),
        b: summary(b),
        mismatches,
        blocked,
        rows,
    }
}

fn selected_gpu_name(run: &RunResult) -> Option<String> {
    run.gpu
        .selected_device_index
        .and_then(|index| run.gpu.devices.get(index))
        .map(|device| {
            format!(
                "{} ({})",
                device.name,
                device.device_uuid.clone().unwrap_or_default()
            )
        })
}

fn status_of(metric: Option<&Metric>) -> String {
    match metric {
        None => "absent".to_string(),
        Some(metric) if metric.status == MetricStatus::Ok => "ok".to_string(),
        Some(_) => "unavailable".to_string(),
    }
}

fn clock_name(clock: Clock) -> &'static str {
    match clock {
        Clock::HostMonotonic => "host_monotonic",
        Clock::Derived => "derived",
        Clock::NotApplicable => "not_applicable",
    }
}

pub fn stat_name(stat: Stat) -> &'static str {
    match stat {
        Stat::P50 => "p50",
        Stat::Mean => "mean",
        Stat::Min => "min",
        Stat::P95 => "p95",
        Stat::Max => "max",
    }
}

fn summary(run: &RunResult) -> RunSummary {
    RunSummary {
        run_id: run.run.run_id.clone(),
        suite: run.run.suite.clone(),
        adapter: run.run.adapter.clone(),
        quick: run.run.quick,
        commit: run.git.commit.clone(),
        branch: run.git.branch.clone(),
        dirty: run.git.dirty,
        timestamp_utc: run.run.timestamp_utc.clone(),
    }
}

/// Render the human-readable table used by the CLI.
pub fn render_table(report: &DiffReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "A: {} suite={} adapter={} {}@{} dirty={:?} {}\n",
        report.a.run_id,
        report.a.suite,
        report.a.adapter,
        report.a.branch.as_deref().unwrap_or("?"),
        report.a.commit.as_deref().unwrap_or("?"),
        report.a.dirty,
        report.a.timestamp_utc
    ));
    out.push_str(&format!(
        "B: {} suite={} adapter={} {}@{} dirty={:?} {}\n",
        report.b.run_id,
        report.b.suite,
        report.b.adapter,
        report.b.branch.as_deref().unwrap_or("?"),
        report.b.commit.as_deref().unwrap_or("?"),
        report.b.dirty,
        report.b.timestamp_utc
    ));
    if report.mismatches.is_empty() {
        out.push_str("compatibility: all checked environment/geometry fields match\n");
    } else {
        let forced = report
            .mismatches
            .iter()
            .any(|mismatch| mismatch.level == "forced");
        out.push_str(&format!(
            "compatibility: {} mismatch(es){}\n",
            report.mismatches.len(),
            if report.blocked {
                " (HARD: comparison refused without --force)"
            } else if forced {
                " (FORCED: hard mismatches ignored)"
            } else {
                " (soft only)"
            }
        ));
        for mismatch in &report.mismatches {
            out.push_str(&format!(
                "  [{}] {}: A={} B={}\n",
                mismatch.level, mismatch.field, mismatch.a, mismatch.b
            ));
        }
    }
    out.push_str(&format!("\nstat: {}\n", report.stat));
    out.push_str(&format!(
        "{:<58} {:>7} {:>12} {:>12} {:>12} {:>9} {:>9}  {}\n",
        "metric", "unit", "A", "B", "delta", "ratio", "n(A/B)", "note"
    ));
    out.push_str(&format!("{}\n", "-".repeat(150)));
    for row in &report.rows {
        let a = row
            .a
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| row.a_status.clone());
        let b = row
            .b
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| row.b_status.clone());
        let delta = row
            .delta
            .map(|value| format!("{value:+.4}"))
            .unwrap_or_else(|| "-".to_string());
        let ratio = row
            .ratio
            .map(|value| format!("{value:.3}x"))
            .unwrap_or_else(|| "-".to_string());
        let samples = match (row.a_samples, row.b_samples) {
            (Some(a_samples), Some(b_samples)) if a_samples == b_samples => a_samples.to_string(),
            (Some(a_samples), Some(b_samples)) => format!("{a_samples}/{b_samples}"),
            (Some(a_samples), None) => format!("{a_samples}/-"),
            (None, Some(b_samples)) => format!("-/{b_samples}"),
            (None, None) => "-".to_string(),
        };
        out.push_str(&format!(
            "{:<58} {:>7} {:>12} {:>12} {:>12} {:>9} {:>9}  {}\n",
            truncate(&row.metric, 58),
            row.unit,
            a,
            b,
            delta,
            ratio,
            samples,
            row.note.as_deref().unwrap_or("")
        ));
    }
    out
}

fn truncate(value: &str, max: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max {
        value.to_string()
    } else {
        let suffix: String = value
            .chars()
            .rev()
            .take(max.saturating_sub(1))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        format!("…{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_parsing() {
        assert_eq!(Stat::parse("p50"), Some(Stat::P50));
        assert_eq!(Stat::parse("median"), Some(Stat::P50));
        assert_eq!(Stat::parse("bogus"), None);
    }

    #[test]
    fn truncation_keeps_suffix() {
        assert_eq!(truncate("short", 10), "short");
        let text = truncate("abcdefghijklmnop", 8);
        assert_eq!(text.chars().count(), 8);
        assert!(text.ends_with("mnop"));
    }
}
