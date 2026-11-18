//! Sample collection and percentile statistics for the FN3 benchmark harness.
//!
//! The harness records every raw sample and, in addition, a documented
//! percentile set (min / p50 / p95 / p99 / max, plus mean and sample standard
//! deviation). Percentiles use linear interpolation over the sorted samples
//! (the R-7 / NumPy default definition), so `p50` equals the median.

use serde::{Deserialize, Serialize};

/// Summary statistics over a finite, non-empty sample set.
///
/// All values are in the unit declared by the owning metric.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    pub min: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub mean: f64,
    pub stddev: f64,
}

/// Linear-interpolation percentile (`q` in `[0, 100]`) over sorted samples.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = (q / 100.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
    }
}

/// Summarize samples, ignoring non-finite values.
///
/// Returns `None` when no finite sample remains. Callers must surface that as
/// an explicitly unavailable metric rather than zero-filling it.
pub fn summarize(samples: &[f64]) -> Option<Stats> {
    let mut finite: Vec<f64> = samples.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).expect("finite values are ordered"));

    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let variance = if finite.len() > 1 {
        finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };

    Some(Stats {
        min: finite[0],
        p50: percentile(&finite, 50.0),
        p95: percentile(&finite, 95.0),
        p99: percentile(&finite, 99.0),
        max: *finite.last().expect("non-empty"),
        mean,
        stddev: variance.sqrt(),
    })
}

/// Convert a latency sample (ms) into a derived throughput sample (unit/s).
pub fn throughput_samples(latency_ms: &[f64], units_per_sample: f64) -> Vec<f64> {
    latency_ms
        .iter()
        .filter(|ms| **ms > 0.0 && ms.is_finite())
        .map(|ms| units_per_sample / (ms / 1000.0))
        .filter(|v| v.is_finite())
        .collect()
}

/// Convert a latency sample (ms) into a derived GFLOP/s sample.
pub fn gflops_samples(latency_ms: &[f64], flops: f64) -> Vec<f64> {
    latency_ms
        .iter()
        .filter(|ms| **ms > 0.0 && ms.is_finite())
        .map(|ms| flops / (*ms / 1000.0) / 1e9)
        .filter(|v| v.is_finite())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_sample_summary() {
        let stats = summarize(&[3.5]).expect("finite sample");
        assert_eq!(stats.min, 3.5);
        assert_eq!(stats.p50, 3.5);
        assert_eq!(stats.max, 3.5);
        assert_eq!(stats.stddev, 0.0);
    }

    #[test]
    fn percentiles_are_interpolated() {
        let stats = summarize(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite samples");
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.p50, 3.0);
        assert_eq!(stats.p95, 4.8);
        assert_eq!(stats.p99, 4.96);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn non_finite_samples_are_ignored() {
        let stats = summarize(&[1.0, f64::NAN, 3.0, f64::INFINITY]).expect("finite samples");
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 3.0);
        assert_eq!(stats.p50, 2.0);
    }

    #[test]
    fn all_non_finite_is_unavailable() {
        assert!(summarize(&[f64::NAN, f64::INFINITY]).is_none());
    }

    #[test]
    fn derived_throughput() {
        let samples = throughput_samples(&[10.0, 20.0], 128.0);
        assert_eq!(samples, vec![12_800.0, 6_400.0]);
    }
}
