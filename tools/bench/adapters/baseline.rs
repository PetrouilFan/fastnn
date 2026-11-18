//! Baseline (2.x) adapter: measures this revision's CPU runtime.
//!
//! Reuse policy:
//! - Projection microbenchmarks call the same public CPU kernels that
//!   `benches/cpu_baselines.rs` measures (`gemm_cpu_flat` on a packed `F32x1`
//!   weight and `gemv_cpu` for the one-token case) instead of reimplementing
//!   kernels.
//! - The `criterion` suite shells out to the existing `cpu_baselines` bench and
//!   imports its raw Criterion samples; it does not reimplement those
//!   measurements.
//! - The whole-model path uses the real AOT pipeline (`GraphBuilder` ->
//!   `GraphExecutor` -> `CpuBackend`) because 2.x ships no weight artifact or
//!   whole-model loader.

use super::{BenchConfig, Suite, SuiteOutcome};
use crate::env;
use crate::model::{self, BuiltModel, Gpt2Spec};
use crate::schema::{Checksum, Geometry, Metric, ModelInfo, Precision};
use crate::stats::gflops_samples;
use crate::timing::{measure_ms, OpCounters};
use fastnn::backend::cpu::microkernels::{gemm_cpu_flat, gemv_cpu};
use fastnn::backend::cpu::telemetry::{cpu_telemetry_snapshot, reset_cpu_telemetry};
use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::GraphExecutor;
use fastnn::backend::ExecutablePlan;
use fastnn::compiler::MemoryPlan;
use fastnn::dtypes::F32x1;
use fastnn::ir::ComputeGraph;
use fastnn::packed_tensor::PackedTensor;
use std::hint::black_box;
use std::time::Instant;

pub fn run(suite: Suite, config: &BenchConfig) -> Result<SuiteOutcome, String> {
    match suite {
        Suite::Smoke => run_smoke(config),
        Suite::Projections => run_projections(config),
        Suite::Model => run_model(config),
        Suite::Criterion => super::criterion_wrap::run(config),
    }
}

pub fn cpu_precision() -> Precision {
    model::precision()
}

fn geometry(
    prompt_tokens: usize,
    decode_tokens: u64,
    layers: usize,
    attention: &str,
    kv_cache: bool,
) -> Geometry {
    Geometry {
        batch: 1,
        prompt_tokens: prompt_tokens as u64,
        decode_tokens,
        context_capacity: prompt_tokens as u64,
        hidden_size: 768,
        num_heads: 12,
        num_kv_heads: None,
        head_dim: 64,
        num_layers: layers as u64,
        intermediate_size: 3_072,
        vocab_size: 50_257,
        attention: attention.to_string(),
        kv_cache,
    }
}

struct ProjectionCase {
    name: &'static str,
    n_in: usize,
    n_out: usize,
}

/// GPT-2 small projection shapes: hidden 768, 12 heads x 64, FFN 3072,
/// vocabulary 50257.
const PROJECTION_CASES: [ProjectionCase; 5] = [
    ProjectionCase {
        name: "attn.qkv",
        n_in: 768,
        n_out: 2_304,
    },
    ProjectionCase {
        name: "attn.out",
        n_in: 768,
        n_out: 768,
    },
    ProjectionCase {
        name: "mlp.fc",
        n_in: 768,
        n_out: 3_072,
    },
    ProjectionCase {
        name: "mlp.proj",
        n_in: 3_072,
        n_out: 768,
    },
    ProjectionCase {
        name: "lm_head",
        n_in: 768,
        n_out: 50_257,
    },
];

fn projection_metrics(
    config: &BenchConfig,
    cases: &[ProjectionCase],
    prompt_tokens: usize,
    category: &str,
) -> (Vec<Metric>, u64, Checksum) {
    let mut metrics = Vec::new();
    let mut total_weight_bytes = 0u64;
    let mut spec_hash = env::fnv1a64(
        format!(
            "gpt2-projections:h768:a12:d64:f3072:v50257:seed{}",
            config.seed
        )
        .as_bytes(),
    );

    let prefill_plan = config.plan(config.quick_warmup(2, 3), config.quick_samples(5, 10));
    let decode_plan = config.plan(config.quick_warmup(2, 3), config.quick_samples(5, 10));

    for case in cases {
        let weight_seed = config.seed ^ env::fnv1a64(case.name.as_bytes());
        let weights = model::deterministic_f32(case.n_in * case.n_out, weight_seed, 0.05);
        let packed = PackedTensor::<F32x1>::from_f32_auto(&weights, &[case.n_out, case.n_in]);
        total_weight_bytes += (case.n_in * case.n_out * 4) as u64;
        spec_hash = env::fnv1a64(&spec_hash.to_le_bytes());

        // Prefill: M tokens through the projection.
        let inputs =
            model::deterministic_f32(prompt_tokens * case.n_in, weight_seed ^ 0x51ed_2701, 0.1);
        let mut outputs = vec![0.0f32; prompt_tokens * case.n_out];
        let flops = 2.0 * prompt_tokens as f64 * case.n_in as f64 * case.n_out as f64;
        let samples = measure_ms(prefill_plan, || {
            gemm_cpu_flat::<F32x1>(
                &packed,
                &inputs,
                &mut outputs,
                prompt_tokens,
                case.n_in,
                case.n_out,
            );
            // Keep every output store live without cloning the buffer.
            outputs.iter().sum::<f32>()
        });
        metrics.push(
            Metric::measured_time(
                &format!("microbench.{}.gemm_f32", case.name),
                category,
                samples.clone(),
                prefill_plan.warmup,
                prefill_plan.repetitions,
            )
            .with_meta("m", prompt_tokens.to_string())
            .with_meta("k", case.n_in.to_string())
            .with_meta("n", case.n_out.to_string())
            .with_meta("flops", format!("{flops:.0}"))
            .with_meta("implementation", "gemm_cpu_flat::<F32x1>")
            .with_meta(
                "weights_fnv1a64",
                env::checksum_f32("weights", &weights).value,
            ),
        );
        if !samples.is_empty() {
            let gflops = gflops_samples(&samples, flops);
            metrics.push(
                Metric::measured_value(
                    &format!("microbench.{}.gemm_gflops", case.name),
                    category,
                    "GFLOP/s",
                    gflops,
                    0,
                    1,
                )
                .with_meta("clock", "derived from host_monotonic samples")
                .with_meta("m", prompt_tokens.to_string())
                .with_meta("k", case.n_in.to_string())
                .with_meta("n", case.n_out.to_string()),
            );
        }

        // Decode: one token through the same projection (GEMV path).
        let activation = model::deterministic_f32(case.n_in, weight_seed ^ 0x9e37_79b9, 0.1);
        let mut output = vec![0.0f32; case.n_out];
        let decode_flops = 2.0 * case.n_in as f64 * case.n_out as f64;
        let samples = measure_ms(decode_plan, || {
            gemv_cpu(&packed, &activation, &mut output);
            output.iter().sum::<f32>()
        });
        metrics.push(
            Metric::measured_time(
                &format!("microbench.{}.gemv_f32", case.name),
                category,
                samples.clone(),
                decode_plan.warmup,
                decode_plan.repetitions,
            )
            .with_meta("m", "1")
            .with_meta("k", case.n_in.to_string())
            .with_meta("n", case.n_out.to_string())
            .with_meta("flops", format!("{decode_flops:.0}"))
            .with_meta("implementation", "gemv_cpu::<F32x1>"),
        );
        if !samples.is_empty() {
            let gflops = gflops_samples(&samples, decode_flops);
            metrics.push(
                Metric::measured_value(
                    &format!("microbench.{}.gemv_gflops", case.name),
                    category,
                    "GFLOP/s",
                    gflops,
                    0,
                    1,
                )
                .with_meta("clock", "derived from host_monotonic samples"),
            );
        }
    }

    let checksum = Checksum {
        subject: "spec".to_string(),
        algorithm: "fnv1a64".to_string(),
        value: format!("{spec_hash:016x}"),
    };
    (metrics, total_weight_bytes, checksum)
}

fn projections_model_info(
    config: &BenchConfig,
    weight_bytes: u64,
    checksum: Checksum,
) -> ModelInfo {
    ModelInfo {
        name: "gpt2-projection-shapes".to_string(),
        revision: "spec:h768-a12-d64-f3072-v50257".to_string(),
        source: "spec-defined-shapes".to_string(),
        parameter_count: Some(weight_bytes / 4),
        weights_bytes: Some(weight_bytes),
        seed: Some(config.seed),
        checksums: vec![checksum],
    }
}

fn run_projections(config: &BenchConfig) -> Result<SuiteOutcome, String> {
    let prompt_tokens = config.prompt_tokens;
    let (metrics, weight_bytes, checksum) =
        projection_metrics(config, &PROJECTION_CASES, prompt_tokens, "microbench");
    let mut outcome = SuiteOutcome::new(metrics);
    outcome.model = projections_model_info(config, weight_bytes, checksum);
    outcome.geometry = geometry(prompt_tokens, 1, 12, "not_applicable", false);
    outcome.precision = cpu_precision();
    outcome.notes.push(
        "Projection microbenchmarks measure the public CPU kernels used by the AOT runtime; they are not a full model."
            .to_string(),
    );
    Ok(outcome)
}

/// Tiny fixture used by `--suite smoke --quick`: fast but exercises every
/// metric category (load, prepare, prefill, decode, throughput, memory,
/// allocation) plus the GPT-2-shaped projection microbenchmarks.
fn run_smoke(config: &BenchConfig) -> Result<SuiteOutcome, String> {
    let prompt_tokens = config.prompt_tokens.clamp(1, 8);
    let mut outcome = run_tiny_model(config, prompt_tokens)?;
    // The projection microbenchmarks are only a coverage check in smoke mode;
    // three measured samples keep the suite far below a minute.
    let mut projection_config = config.clone();
    if projection_config.samples.is_none() {
        projection_config.samples = Some(3);
    }
    let (projection_metrics, weight_bytes, checksum) = projection_metrics(
        &projection_config,
        &PROJECTION_CASES,
        prompt_tokens,
        "microbench",
    );
    outcome.metrics.extend(projection_metrics);
    if outcome.model.source == "not_applicable" {
        outcome.model = projections_model_info(config, weight_bytes, checksum);
    }
    outcome
        .notes
        .push("Smoke suite combines a tiny synthetic decoder with GPT-2-shaped projection microbenchmarks."
            .to_string());
    Ok(outcome)
}

fn run_tiny_model(config: &BenchConfig, prompt_tokens: usize) -> Result<SuiteOutcome, String> {
    let mut tiny = config.clone();
    tiny.prompt_tokens = prompt_tokens;
    tiny.layers = 1;
    run_model_with_specs(
        &tiny,
        Gpt2Spec::tiny(prompt_tokens, config.seed),
        Gpt2Spec::tiny(1, config.seed),
        "synthetic-tiny-decoder",
        tiny_geometry(prompt_tokens, config),
    )
}

fn tiny_geometry(prompt_tokens: usize, config: &BenchConfig) -> Geometry {
    Geometry {
        batch: 1,
        prompt_tokens: prompt_tokens as u64,
        decode_tokens: config.decode_tokens,
        context_capacity: prompt_tokens as u64,
        hidden_size: 96,
        num_heads: 2,
        num_kv_heads: None,
        head_dim: 48,
        num_layers: 1,
        intermediate_size: 192,
        vocab_size: 1_024,
        attention: "full".to_string(),
        kv_cache: false,
    }
}

fn run_model(config: &BenchConfig) -> Result<SuiteOutcome, String> {
    run_model_with_specs(
        config,
        Gpt2Spec::gpt2(config.prompt_tokens, config.layers, config.seed),
        Gpt2Spec::gpt2(1, config.layers, config.seed),
        "synthetic-gpt2-decoder-stack",
        geometry(
            config.prompt_tokens,
            config.decode_tokens,
            config.layers,
            "full",
            false,
        ),
    )
}

fn build_pair(
    prefill_spec: &Gpt2Spec,
    decode_spec: &Gpt2Spec,
) -> Result<(BuiltModel, BuiltModel), String> {
    let prefill = model::build(prefill_spec)?;
    let decode = model::build(decode_spec)?;
    Ok((prefill, decode))
}

type Compiled = (ExecutablePlan, MemoryPlan, ComputeGraph);

fn compile_pair(prefill: &BuiltModel, decode: &BuiltModel) -> Result<(Compiled, Compiled), String> {
    // A GraphExecutor memoizes a static-shape plan, so each sequence length
    // needs its own executor (prefill and decode have different input shapes).
    let prefill_compiled = GraphExecutor::new(CpuBackend)
        .compile_with_plan_and_quantize(prefill.graph.clone(), None, None)
        .map_err(|error| format!("prefill compile failed: {error}"))?;
    let decode_compiled = GraphExecutor::new(CpuBackend)
        .compile_with_plan_and_quantize(decode.graph.clone(), None, None)
        .map_err(|error| format!("decode compile failed: {error}"))?;
    Ok((prefill_compiled, decode_compiled))
}

fn input_bytes(tokens: usize, hidden: usize, seed: u64) -> Vec<u8> {
    let values = model::deterministic_f32(tokens * hidden, seed, 0.1);
    bytemuck::cast_slice::<f32, u8>(&values).to_vec()
}

fn all_unavailable(reason: &str, geometry: Geometry, names: &[(&str, &str, &str)]) -> SuiteOutcome {
    let metrics = names
        .iter()
        .map(|(name, category, unit)| Metric::unavailable(name, category, unit, reason.to_string()))
        .collect();
    let mut outcome = SuiteOutcome::new(metrics);
    outcome.geometry = geometry;
    outcome
        .warnings
        .push(format!("model suite produced no measurements: {reason}"));
    outcome
}

fn model_metric_names() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        ("model.load", "load", "ms"),
        ("model.prepare", "prepare", "ms"),
        ("prefill.time_to_first_token", "prefill", "ms"),
        ("prefill.throughput", "throughput", "tokens/s"),
        ("decode.per_token", "decode", "ms"),
        ("decode.throughput", "throughput", "tokens/s"),
        ("memory.peak_rss", "memory", "MiB"),
        ("memory.device_peak", "memory", "MiB"),
        (
            "allocation.tls_vec_allocs_per_decode_step",
            "allocation",
            "count",
        ),
        (
            "allocation.arena_temp_copies_per_decode_step",
            "allocation",
            "count",
        ),
    ]
}

fn run_model_with_specs(
    config: &BenchConfig,
    prefill_spec: Gpt2Spec,
    decode_spec: Gpt2Spec,
    model_name: &str,
    geometry: Geometry,
) -> Result<SuiteOutcome, String> {
    let names = model_metric_names();

    // 1. Load: deterministic weight synthesis + graph construction. 2.x has no
    //    on-disk model artifact, so this is the honest analogue of model load.
    let load_reps = config.load_repetitions();
    let mut load_samples = Vec::with_capacity(load_reps as usize);
    let mut loaded: Option<(BuiltModel, BuiltModel)> = None;
    for _ in 0..load_reps {
        let start = Instant::now();
        let pair = match build_pair(&prefill_spec, &decode_spec) {
            Ok(pair) => pair,
            Err(error) => return Ok(all_unavailable(&error, geometry.clone(), &names)),
        };
        load_samples.push(start.elapsed().as_secs_f64() * 1000.0);
        loaded = Some(pair);
    }
    let (prefill_model, decode_model) = loaded.expect("at least one load repetition");

    // 2. Prepare: compile both sequence lengths with a cold executor per sample.
    let prepare_samples_count = config.quick_samples(2, 3);
    let mut prepare_samples = Vec::with_capacity(prepare_samples_count as usize);
    let mut compiled: Option<(Compiled, Compiled)> = None;
    for _ in 0..prepare_samples_count {
        let start = Instant::now();
        let pair = match compile_pair(&prefill_model, &decode_model) {
            Ok(pair) => pair,
            Err(error) => return Ok(all_unavailable(&error, geometry.clone(), &names)),
        };
        prepare_samples.push(start.elapsed().as_secs_f64() * 1000.0);
        compiled = Some(pair);
    }
    let (
        (mut prefill_plan, prefill_memory, prefill_graph),
        (mut decode_plan, decode_memory, decode_graph),
    ) = compiled.expect("at least one prepare sample");

    let mut prefill_executor = GraphExecutor::new(CpuBackend);
    let mut decode_executor = GraphExecutor::new(CpuBackend);

    // 3. Prefill / time to first token.
    let prompt = prefill_spec.seq;
    let hidden = prefill_spec.hidden;
    let prefill_input = input_bytes(prompt, hidden, config.seed ^ 0x1234_5678);
    let prefill_inputs: [&[u8]; 1] = [&prefill_input];
    for _ in 0..config.quick_warmup(1, 2) {
        let out = prefill_executor
            .execute(
                &prefill_graph,
                &mut prefill_plan,
                &prefill_memory,
                &prefill_inputs,
            )
            .map_err(|error| format!("prefill warmup execute failed: {error}"))?;
        black_box(out);
    }
    let observed = prefill_executor.last_output_shapes();
    if !observed.is_empty() {
        let observed: Vec<u64> = observed[0].iter().map(|dim| *dim as u64).collect();
        if observed != prefill_model.output_shape {
            return Ok(all_unavailable(
                &format!(
                    "prefill output shape mismatch: graph declares {:?}, executor reported {:?}",
                    prefill_model.output_shape, observed
                ),
                geometry,
                &names,
            ));
        }
    }
    let prefill_plan_samples = config.plan(config.quick_warmup(1, 2), config.quick_samples(3, 10));
    let prefill_samples = measure_ms(prefill_plan_samples, || {
        prefill_executor
            .execute(
                &prefill_graph,
                &mut prefill_plan,
                &prefill_memory,
                &prefill_inputs,
            )
            .expect("prefill execute failed after successful warmup")
    });

    // 4. Steady decode: one-token forward per step, with CPU allocation telemetry.
    let decode_input = input_bytes(1, hidden, config.seed ^ 0x8765_4321);
    let decode_inputs: [&[u8]; 1] = [&decode_input];
    for _ in 0..config.quick_warmup(2, 5) {
        let out = decode_executor
            .execute(
                &decode_graph,
                &mut decode_plan,
                &decode_memory,
                &decode_inputs,
            )
            .map_err(|error| format!("decode warmup execute failed: {error}"))?;
        black_box(out);
    }
    let decode_steps = config.samples.unwrap_or(config.decode_tokens.max(1)).max(1);
    let decode_warmup = config.quick_warmup(2, 5);
    let mut decode_samples = Vec::with_capacity(decode_steps as usize);
    let mut tls_alloc_samples = Vec::with_capacity(decode_steps as usize);
    let mut arena_copy_samples = Vec::with_capacity(decode_steps as usize);
    let mut totals = OpCounters::default();
    for _ in 0..decode_steps {
        reset_cpu_telemetry();
        let start = Instant::now();
        let out = decode_executor
            .execute(
                &decode_graph,
                &mut decode_plan,
                &decode_memory,
                &decode_inputs,
            )
            .map_err(|error| format!("decode execute failed: {error}"))?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        black_box(&out);
        let snapshot = cpu_telemetry_snapshot();
        let step = OpCounters {
            arena_temp_copies: snapshot.arena_temp_copies,
            arena_temp_copy_bytes: snapshot.arena_temp_copy_bytes,
            tls_vec_allocs: snapshot.tls_vec_allocs,
            tls_vec_reuses: snapshot.tls_vec_reuses,
        };
        totals.add(step);
        decode_samples.push(elapsed);
        tls_alloc_samples.push(step.tls_vec_allocs as f64);
        arena_copy_samples.push(step.arena_temp_copies as f64);
    }

    let mut metrics = Vec::new();
    metrics.push(
        Metric::measured_time("model.load", "load", load_samples, 0, load_reps)
            .with_meta(
                "includes",
                "deterministic f32 weight synthesis + graph construction",
            )
            .with_meta("prefill_plan_tokens", prompt.to_string())
            .with_meta("decode_plan_tokens", "1")
            .with_meta(
                "parameter_count",
                prefill_model.spec.parameter_count().to_string(),
            )
            .with_meta(
                "weights_bytes_per_graph",
                prefill_model.spec.weights_bytes().to_string(),
            )
            .with_meta(
                "resident_weight_bytes",
                (prefill_model.weights_bytes * 2).to_string(),
            )
            .with_meta("attention_kv_cache", "false"),
    );
    metrics.push(
        Metric::measured_time(
            "model.prepare",
            "prepare",
            prepare_samples,
            0,
            prepare_samples_count,
        )
        .with_meta(
            "includes",
            "compiler passes + memory planning + CPU plan build, two sequence lengths",
        )
        .with_meta("executor", "cold GraphExecutor per sample"),
    );
    metrics.push(
        Metric::measured_time(
            "prefill.time_to_first_token",
            "prefill",
            prefill_samples.clone(),
            prefill_plan_samples.warmup,
            prefill_plan_samples.repetitions,
        )
        .with_meta("prompt_tokens", prompt.to_string())
        .with_meta("includes", "full prompt forward pass, no KV cache"),
    );
    if !prefill_samples.is_empty() {
        metrics.push(
            Metric::measured_value(
                "prefill.throughput",
                "throughput",
                "tokens/s",
                crate::stats::throughput_samples(&prefill_samples, prompt as f64),
                0,
                1,
            )
            .with_meta("clock", "derived from host_monotonic prefill samples"),
        );
    }
    metrics.push(
        Metric::measured_time(
            "decode.per_token",
            "decode",
            decode_samples.clone(),
            decode_warmup,
            1,
        )
        .with_meta("decode_steps", decode_steps.to_string())
        .with_meta(
            "includes",
            "one-token full-stack forward; 2.x has no KV cache",
        )
        .with_meta("attention", "full (unmasked)"),
    );
    if !decode_samples.is_empty() {
        metrics.push(
            Metric::measured_value(
                "decode.throughput",
                "throughput",
                "tokens/s",
                crate::stats::throughput_samples(&decode_samples, 1.0),
                0,
                1,
            )
            .with_meta("clock", "derived from host_monotonic decode samples"),
        );
    }

    metrics.push(match env::peak_rss_bytes() {
        Some(bytes) => Metric::measured_value(
            "memory.peak_rss",
            "memory",
            "MiB",
            vec![bytes as f64 / (1024.0 * 1024.0)],
            0,
            1,
        )
        .with_meta("scope", "whole process, VmHWM after all measurements")
        .with_meta("source", "/proc/self/status"),
        None => Metric::unavailable(
            "memory.peak_rss",
            "memory",
            "MiB",
            "/proc/self/status VmHWM is unavailable on this host",
        ),
    });
    metrics.push(
        Metric::unavailable(
            "memory.device_peak",
            "memory",
            "MiB",
            "no GPU execution in this run; device memory is not queryable from the CPU baseline adapter",
        )
        .with_meta("backend", "cpu"),
    );
    metrics.push(
        Metric::measured_value(
            "allocation.tls_vec_allocs_per_decode_step",
            "allocation",
            "count",
            tls_alloc_samples,
            0,
            1,
        )
        .with_meta("source", "fastnn::backend::cpu::telemetry")
        .with_meta("total_tls_vec_allocs", totals.tls_vec_allocs.to_string())
        .with_meta("total_tls_vec_reuses", totals.tls_vec_reuses.to_string()),
    );
    metrics.push(
        Metric::measured_value(
            "allocation.arena_temp_copies_per_decode_step",
            "allocation",
            "count",
            arena_copy_samples,
            0,
            1,
        )
        .with_meta("source", "fastnn::backend::cpu::telemetry")
        .with_meta(
            "total_arena_temp_copy_bytes",
            totals.arena_temp_copy_bytes.to_string(),
        ),
    );

    let mut outcome = SuiteOutcome::new(metrics);
    outcome.model = ModelInfo {
        name: model_name.to_string(),
        revision: format!("synthetic:{}", config.seed),
        source: "synthetic-random-weights".to_string(),
        parameter_count: Some(prefill_model.parameter_count),
        weights_bytes: Some(prefill_model.weights_bytes),
        seed: Some(config.seed),
        checksums: vec![prefill_model.checksum.clone()],
    };
    outcome.geometry = geometry;
    outcome.precision = cpu_precision();
    outcome.notes.push(
        "2.x has no KV cache and no weight artifact: load measures deterministic weight synthesis plus graph construction, prepare measures AOT compile for prefill and decode sequence lengths, and decode is a one-token full-stack forward."
            .to_string(),
    );
    outcome.notes.push(
        "Attention is unmasked full attention; causal masking would not change the FLOPs measured here."
            .to_string(),
    );
    outcome.notes.push(
        "Graphs are statically shaped per sequence length; context_capacity equals the prefill prompt length and the decode graph is built for one token."
            .to_string(),
    );
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tiny_config(decode_tokens: u64) -> BenchConfig {
        BenchConfig {
            quick: true,
            seed: 7,
            prompt_tokens: 2,
            layers: 1,
            decode_tokens,
            samples: None,
            warmup: Some(0),
            manifest_dir: PathBuf::from("."),
            reuse_criterion: false,
        }
    }

    #[test]
    fn decode_token_count_controls_measured_steps() {
        let outcome = run_tiny_model(&tiny_config(3), 2).expect("tiny model suite");
        let metric = outcome
            .metrics
            .iter()
            .find(|metric| metric.name == "decode.per_token")
            .expect("decode metric");
        assert_eq!(metric.host_time.as_ref().unwrap().sample_count, 3);
        assert_eq!(outcome.geometry.decode_tokens, 3);
    }

    #[test]
    fn projection_shapes_are_gpt2_small() {
        assert_eq!(PROJECTION_CASES[0].n_in, 768);
        assert_eq!(PROJECTION_CASES[0].n_out, 2_304);
        assert_eq!(PROJECTION_CASES[4].n_out, 50_257);
    }
}
