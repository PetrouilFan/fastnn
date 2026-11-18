//! Wraps the existing Criterion benches in `benches/cpu_baselines.rs`.
//!
//! The measurement itself stays in Criterion (warmup, sampling, statistics).
//! This module runs the documented command, imports the raw Criterion samples
//! and converts them into the FN3 result schema. It deliberately does not
//! reimplement any kernel measurement.

use super::{empty_geometry, BenchConfig, SuiteOutcome};
use crate::model;
use crate::schema::{Metric, ModelInfo};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const FEATURES: &str = "simd,parallel,prepared-plan,fusion-forward";

pub fn run(config: &BenchConfig) -> Result<SuiteOutcome, String> {
    let manifest_dir = &config.manifest_dir;
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                manifest_dir.join(path)
            }
        })
        .unwrap_or_else(|| manifest_dir.join("target"));
    let criterion_dir = target_dir.join("criterion");

    if config.reuse_criterion {
        if !criterion_dir.exists() {
            return Ok(unavailable_outcome(format!(
                "--reuse-criterion was requested but {} does not exist",
                criterion_dir.display()
            )));
        }
    } else if criterion_dir.exists() {
        fs::remove_dir_all(&criterion_dir).map_err(|error| {
            format!(
                "could not clear previous Criterion output at {}: {error}",
                criterion_dir.display()
            )
        })?;
    }

    let mut command = Command::new("cargo");
    command
        .current_dir(manifest_dir)
        .env("CARGO_BUILD_JOBS", "4")
        .args([
            "bench",
            "--no-default-features",
            "--features",
            FEATURES,
            "--bench",
            "cpu_baselines",
            "--",
            "--sample-size",
            "10",
            "--warm-up-time",
            "0.1",
            "--measurement-time",
            "0.3",
            "--noplot",
        ]);

    let display = format!(
        "cargo bench --no-default-features --features {FEATURES} --bench cpu_baselines -- \
         --sample-size 10 --warm-up-time 0.1 --measurement-time 0.3 --noplot"
    );

    let output = if config.reuse_criterion {
        None
    } else {
        Some(match command.output() {
            Ok(output) => output,
            Err(error) => {
                return Ok(unavailable_outcome(format!(
                    "could not execute `{display}`: {error}"
                )))
            }
        })
    };
    if let Some(output) = output.as_ref() {
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: String = stderr
                .lines()
                .rev()
                .take(12)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n");
            return Ok(unavailable_outcome(format!(
                "`{display}` exited with {}:\n{tail}",
                output.status
            )));
        }
    }

    let mut metrics = Vec::new();
    collect_metrics(&criterion_dir, &mut metrics)?;
    if metrics.is_empty() {
        return Ok(unavailable_outcome(format!(
            "Criterion produced no estimates under {}",
            criterion_dir.display()
        )));
    }

    let mut outcome = SuiteOutcome::new(metrics);
    outcome.model = ModelInfo {
        name: "criterion:cpu_baselines".to_string(),
        revision: "bench-profile".to_string(),
        source: "criterion".to_string(),
        parameter_count: None,
        weights_bytes: None,
        seed: None,
        checksums: Vec::new(),
    };
    outcome.geometry = empty_geometry();
    outcome.precision = model::precision();
    outcome.notes.push(if config.reuse_criterion {
        format!(
            "Imported existing Criterion output from {} (cargo bench not re-run)",
            criterion_dir.display()
        )
    } else {
        format!("Executed: {display}")
    });
    outcome.notes.push(
        "Criterion owns warmup and sampling; raw per-iteration times were imported from \
         new/sample.json and percentiles recomputed by this harness."
            .to_string(),
    );
    Ok(outcome)
}

fn unavailable_outcome(reason: String) -> SuiteOutcome {
    let mut outcome = SuiteOutcome::new(vec![Metric::unavailable(
        "criterion.cpu_baselines",
        "microbench",
        "ms",
        reason.clone(),
    )]);
    outcome.geometry = empty_geometry();
    outcome.precision = model::precision();
    outcome.warnings.push(reason);
    outcome
}

fn collect_metrics(root: &Path, metrics: &mut Vec<Metric>) -> Result<(), String> {
    let mut benchmark_dirs = Vec::new();
    find_benchmark_dirs(root, &mut benchmark_dirs)?;
    benchmark_dirs.sort();
    for benchmark_dir in benchmark_dirs {
        let relative = benchmark_dir
            .strip_prefix(root)
            .map_err(|error| format!("bad Criterion path: {error}"))?;
        let parts: Vec<String> = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy().to_string())
            .collect();
        if parts.len() < 2 {
            continue;
        }
        let group = parts[0].clone();
        let benchmark = parts[1..].join("/");
        let estimates = read_json(&benchmark_dir.join("new").join("estimates.json"))?;
        let samples = read_samples(&benchmark_dir.join("new").join("sample.json"))?;

        let name = format!("criterion.{group}.{benchmark}");
        let mut metric = if samples.is_empty() {
            Metric::unavailable(
                &name,
                "microbench",
                "ms",
                "Criterion new/sample.json is missing or has no usable samples",
            )
        } else {
            Metric::measured_time(&name, "microbench", samples, 0, 1)
        };
        metric.metadata.insert("group".to_string(), group.clone());
        metric
            .metadata
            .insert("benchmark".to_string(), benchmark.clone());
        metric.metadata.insert(
            "criterion_mean_ns".to_string(),
            point_estimate(&estimates, "mean"),
        );
        metric.metadata.insert(
            "criterion_median_ns".to_string(),
            point_estimate(&estimates, "median"),
        );
        metric.metadata.insert(
            "criterion_std_dev_ns".to_string(),
            point_estimate(&estimates, "std_dev"),
        );
        metric.metadata.insert(
            "source".to_string(),
            "benches/cpu_baselines.rs via criterion".to_string(),
        );
        metric.metadata.insert(
            "warmup".to_string(),
            "criterion-managed; warmup count is not recorded in sample.json".to_string(),
        );
        metrics.push(metric);
    }
    Ok(())
}

fn find_benchmark_dirs(root: &Path, found: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(format!(
                "could not read Criterion directory {}: {error}",
                root.display()
            ))
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "new")
                && path.join("estimates.json").is_file()
            {
                found.push(path.parent().expect("new has a parent").to_path_buf());
            } else {
                find_benchmark_dirs(&path, found)?;
            }
        }
    }
    Ok(())
}

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| format!("invalid JSON in {}: {error}", path.display()))
}

fn point_estimate(value: &serde_json::Value, key: &str) -> String {
    value
        .get(key)
        .and_then(|entry| entry.get("point_estimate"))
        .and_then(|estimate| estimate.as_f64())
        .map(|estimate| format!("{estimate:.3}"))
        .unwrap_or_else(|| "unavailable".to_string())
}

/// Criterion stores batch times in nanoseconds under `times` and the batch size
/// under `iters`; convert to per-iteration milliseconds.
fn read_samples(path: &Path) -> Result<Vec<f64>, String> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let value = read_json(path)?;
    let times = value
        .get("times")
        .and_then(|times| times.as_array())
        .map(|times| times.iter().filter_map(|v| v.as_f64()).collect::<Vec<_>>())
        .unwrap_or_default();
    let iters = value
        .get("iters")
        .and_then(|iters| iters.as_array())
        .map(|iters| iters.iter().filter_map(|v| v.as_f64()).collect::<Vec<_>>())
        .unwrap_or_default();
    let mut samples = Vec::with_capacity(times.len().min(iters.len()));
    for (time, iters) in times.iter().zip(iters.iter()) {
        if *iters > 0.0 {
            samples.push(time / iters / 1e6);
        }
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn criterion_sample_conversion_is_per_iteration_ms() {
        let dir = std::env::temp_dir().join(format!("fn3-criterion-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("sample.json");
        fs::write(&path, r#"{"iters":[10,10],"times":[1000000.0,2000000.0]}"#).expect("write");
        let samples = read_samples(&path).expect("parse");
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 0.1).abs() < 1e-12);
        assert!((samples[1] - 0.2).abs() < 1e-12);
        fs::remove_dir_all(&dir).ok();
    }
}
