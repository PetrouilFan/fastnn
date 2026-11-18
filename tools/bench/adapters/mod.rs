//! Adapter seam between the benchmark harness and the runtime under test.
//!
//! `baseline` measures the 2.x runtime in this revision. `fn3` is the seam for
//! the 3.0 `fastnn::fn3` runtime; it is not built here and fails loudly rather
//! than emitting placeholder numbers.

pub mod baseline;
pub mod criterion_wrap;
pub mod fn3;

use crate::schema::{Geometry, Metric, ModelInfo, Precision};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Suite {
    /// Small deterministic fixture covering every metric category in seconds.
    Smoke,
    /// GPT-2-shaped projection GEMM/GEMV microbenchmarks.
    Projections,
    /// Synthetic GPT-2 decoder stack: load, prepare, prefill, decode, memory.
    Model,
    /// Wraps the existing Criterion benches in `benches/cpu_baselines.rs`.
    Criterion,
}

impl Suite {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "smoke" => Some(Self::Smoke),
            "projections" => Some(Self::Projections),
            "model" => Some(Self::Model),
            "criterion" => Some(Self::Criterion),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Projections => "projections",
            Self::Model => "model",
            Self::Criterion => "criterion",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Smoke => {
                "tiny decoder fixture + GPT-2-shaped projections; target < 60 s with --quick"
            }
            Self::Projections => "GPT-2 projection GEMM (prefill) / GEMV (decode) microbenchmarks",
            Self::Model => {
                "synthetic GPT-2 decoder stack: load, prepare, prefill, decode, peak RSS, allocations"
            }
            Self::Criterion => {
                "wraps benches/cpu_baselines.rs and imports Criterion samples (slow: bench-profile build)"
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterKind {
    Baseline,
    Fn3,
}

impl AdapterKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "baseline" => Some(Self::Baseline),
            "fn3" => Some(Self::Fn3),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Fn3 => "fn3",
        }
    }
}

#[derive(Clone, Debug)]
pub struct BenchConfig {
    pub quick: bool,
    pub seed: u64,
    pub prompt_tokens: usize,
    pub layers: usize,
    pub decode_tokens: u64,
    /// Overrides the per-metric measured sample count.
    pub samples: Option<u64>,
    /// Overrides the per-metric warmup count.
    pub warmup: Option<u64>,
    pub manifest_dir: PathBuf,
    /// Import existing `target/criterion` data instead of running cargo bench.
    pub reuse_criterion: bool,
}

impl BenchConfig {
    pub fn plan(&self, default_warmup: u64, default_samples: u64) -> crate::timing::SamplePlan {
        crate::timing::SamplePlan::new(
            self.warmup.unwrap_or(default_warmup),
            self.samples.unwrap_or(default_samples),
            1,
        )
    }

    pub fn quick_warmup(&self, quick: u64, normal: u64) -> u64 {
        self.warmup
            .unwrap_or(if self.quick { quick } else { normal })
    }

    pub fn quick_samples(&self, quick: u64, normal: u64) -> u64 {
        self.samples
            .unwrap_or(if self.quick { quick } else { normal })
    }

    pub fn load_repetitions(&self) -> u64 {
        self.samples.unwrap_or(if self.quick { 2 } else { 3 })
    }
}

#[derive(Debug)]
pub struct SuiteOutcome {
    pub metrics: Vec<Metric>,
    pub model: ModelInfo,
    pub geometry: Geometry,
    pub precision: Precision,
    /// Caveats that belong to the suite as a whole (for example "2.x has no KV
    /// cache, decode re-runs the whole stack").
    pub notes: Vec<String>,
    pub warnings: Vec<String>,
}

impl SuiteOutcome {
    pub fn new(metrics: Vec<Metric>) -> Self {
        Self {
            metrics,
            model: crate::schema::ModelInfo::empty(),
            geometry: empty_geometry(),
            precision: crate::model::precision(),
            notes: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

pub fn empty_geometry() -> Geometry {
    Geometry {
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
    }
}

pub fn run(
    adapter: AdapterKind,
    suite: Suite,
    config: &BenchConfig,
) -> Result<SuiteOutcome, String> {
    match adapter {
        AdapterKind::Baseline => baseline::run(suite, config),
        AdapterKind::Fn3 => fn3::run(suite, config),
    }
}
