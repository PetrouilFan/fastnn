//! FN3 benchmark and profiling harness (FN3-04).
//!
//! One command runs a named suite against the 2.x baseline (or, later, the 3.0
//! `fn3` runtime) and writes a versioned JSON document plus a human summary.
//!
//! Build (the default `python` feature breaks standalone binary linking):
//!
//! ```text
//! CARGO_BUILD_JOBS=4 cargo build --release --no-default-features \
//!   --features simd,parallel,prepared-plan,fusion-forward --bin fn3-bench
//!
//! target/release/fn3-bench run --suite smoke --quick --out /tmp/smoke.json
//! target/release/fn3-bench compare /tmp/a.json /tmp/b.json
//! ```
//!
//! See `docs/reports/fn3-04/README.md` for the schema, discipline rules and the
//! GPU timing contract.

mod adapters;
mod compare;
mod env;
mod model;
mod schema;
mod stats;
mod timing;

use adapters::{AdapterKind, BenchConfig, Suite};
use schema::{MetricStatus, RunInfo, RunResult};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_SEED: u64 = 20_260_818;

fn main() {
    let code = dispatch(std::env::args().skip(1).collect());
    std::process::exit(code);
}

fn dispatch(args: Vec<String>) -> i32 {
    let Some(command) = args.first() else {
        print_usage();
        return 2;
    };
    match command.as_str() {
        "run" => cmd_run(&args[1..]),
        "compare" => cmd_compare(&args[1..]),
        "list-suites" => {
            cmd_list_suites();
            0
        }
        "help" | "--help" | "-h" => {
            print_usage();
            0
        }
        other => {
            eprintln!("error: unknown command `{other}`");
            print_usage();
            2
        }
    }
}

fn print_usage() {
    println!(
        "fn3-bench — reproducible FastNN benchmark harness (schema v{})\n\
         \n\
         USAGE:\n\
         \x20 fn3-bench run --suite <smoke|projections|model|criterion> [options]\n\
         \x20 fn3-bench compare <a.json> <b.json> [--force] [--stat p50] [--json]\n\
         \x20 fn3-bench list-suites\n\
         \n\
         RUN OPTIONS:\n\
         \x20 --adapter baseline|fn3   runtime under test (default: baseline)\n\
         \x20 --quick                  fast smoke parameters (required for < 60 s smoke)\n\
         \x20 --out <path|->           JSON output path (default target/fn3-bench/<suite>-<run>.json)\n\
         \x20 --json-stdout            print JSON instead of the human summary\n\
         \x20 --seed <u64>             deterministic fixture seed (default {DEFAULT_SEED})\n\
         \x20 --threads <n>            sets RAYON_NUM_THREADS before the run\n\
         \x20 --samples <n>            override measured sample count for every metric\n\
         \x20 --warmup <n>             override warmup count for every metric\n\
         \x20 --prompt-tokens <n>      prefill tokens (default: quick 8 / normal 128)\n\
         \x20 --layers <n>             decoder layers (default: quick 1 / normal 12)\n\
         \x20 --decode-tokens <n>      decode steps (default: quick 5 / normal 32)\n\
         \x20 --fn3-manifest <path>   model manifest for --adapter fn3 (see tools/bench/README.md)\n\
         \x20 --max-load <f>           load-average warning threshold (default: half logical cores, min 2.0)\n\
         \x20 --no-gpu-probe           skip `vulkaninfo --summary`\n\
         \x20 --reuse-criterion        import existing target/criterion output\n\
         \x20 --label <text>           free-form label recorded in the result\n\
         \n\
         COMPARE OPTIONS:\n\
         \x20 --force                  compare despite hard environment/geometry mismatches\n\
         \x20 --stat <p50|mean|min|p95|max>  statistic used for the table (default p50)\n\
         \x20 --json                   print a machine-readable diff instead of a table",
        schema::SCHEMA_VERSION
    );
}

fn cmd_list_suites() {
    let fn3 = if adapters::fn3::is_available() {
        "available"
    } else {
        "not built (rebuild with --features fn3)"
    };
    println!("Suites:");
    for suite in [
        Suite::Smoke,
        Suite::Projections,
        Suite::Model,
        Suite::Criterion,
    ] {
        println!("  {:<12} {}", suite.as_str(), suite.description());
    }
    println!("\nAdapters:");
    println!("  baseline     2.x CPU runtime measured through the existing AOT pipeline");
    println!(
        "  fn3          3.0 pinned prepared model ({fn3}); needs --fn3-manifest and --suite model"
    );
}

struct RunOptions {
    suite: Suite,
    adapter: AdapterKind,
    quick: bool,
    out: Option<PathBuf>,
    json_stdout: bool,
    seed: u64,
    threads: Option<u64>,
    samples: Option<u64>,
    warmup: Option<u64>,
    prompt_tokens: Option<usize>,
    layers: Option<usize>,
    decode_tokens: Option<u64>,
    max_load: Option<f64>,
    gpu_probe: bool,
    label: Option<String>,
    reuse_criterion: bool,
    fn3_manifest: Option<PathBuf>,
}

fn parse_u64(value: &str, flag: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|error| format!("{flag} expects an integer, got `{value}`: {error}"))
}

fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    let current = &args[*index];
    if let Some((_, value)) = current.split_once('=') {
        return Ok(value.to_string());
    }
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_run_options(args: &[String]) -> Result<RunOptions, String> {
    let mut suite = Suite::Smoke;
    let mut adapter = AdapterKind::Baseline;
    let mut quick = false;
    let mut out = None;
    let mut json_stdout = false;
    let mut seed = DEFAULT_SEED;
    let mut threads = None;
    let mut samples = None;
    let mut warmup = None;
    let mut prompt_tokens = None;
    let mut layers = None;
    let mut decode_tokens = None;
    let mut max_load = None;
    let mut gpu_probe = true;
    let mut label = None;
    let mut reuse_criterion = false;
    let mut fn3_manifest = None;

    let mut index = 0;
    while index < args.len() {
        let arg = args[index].clone();
        let flag = arg.split('=').next().unwrap_or("");
        match flag {
            "--suite" => {
                let value = take_value(args, &mut index, "--suite")?;
                suite = Suite::parse(&value)
                    .ok_or_else(|| format!("unknown suite `{value}`; use list-suites"))?;
            }
            "--adapter" => {
                let value = take_value(args, &mut index, "--adapter")?;
                adapter = AdapterKind::parse(&value)
                    .ok_or_else(|| format!("unknown adapter `{value}`; use baseline or fn3"))?;
            }
            "--quick" => quick = true,
            "--out" => out = Some(PathBuf::from(take_value(args, &mut index, "--out")?)),
            "--json-stdout" => json_stdout = true,
            "--seed" => seed = parse_u64(&take_value(args, &mut index, "--seed")?, "--seed")?,
            "--threads" => {
                threads = Some(parse_u64(
                    &take_value(args, &mut index, "--threads")?,
                    "--threads",
                )?)
            }
            "--samples" => {
                samples = Some(parse_u64(
                    &take_value(args, &mut index, "--samples")?,
                    "--samples",
                )?)
            }
            "--warmup" => {
                warmup = Some(parse_u64(
                    &take_value(args, &mut index, "--warmup")?,
                    "--warmup",
                )?)
            }
            "--prompt-tokens" => {
                prompt_tokens = Some(parse_u64(
                    &take_value(args, &mut index, "--prompt-tokens")?,
                    "--prompt-tokens",
                )? as usize)
            }
            "--layers" => {
                layers = Some(
                    parse_u64(&take_value(args, &mut index, "--layers")?, "--layers")? as usize,
                )
            }
            "--decode-tokens" => {
                decode_tokens = Some(parse_u64(
                    &take_value(args, &mut index, "--decode-tokens")?,
                    "--decode-tokens",
                )?)
            }
            "--max-load" => {
                let value = take_value(args, &mut index, "--max-load")?;
                max_load = Some(
                    value
                        .parse::<f64>()
                        .map_err(|error| format!("--max-load expects a float: {error}"))?,
                )
            }
            "--no-gpu-probe" => gpu_probe = false,
            "--reuse-criterion" => reuse_criterion = true,
            "--fn3-manifest" => {
                fn3_manifest = Some(PathBuf::from(take_value(
                    args,
                    &mut index,
                    "--fn3-manifest",
                )?))
            }
            "--label" => label = Some(take_value(args, &mut index, "--label")?),
            "--help" | "-h" => return Err("__help__".to_string()),
            other => return Err(format!("unknown run option `{other}`")),
        }
        index += 1;
    }

    if prompt_tokens == Some(0) {
        return Err("--prompt-tokens must be positive".to_string());
    }
    if layers == Some(0) {
        return Err("--layers must be positive".to_string());
    }

    Ok(RunOptions {
        suite,
        adapter,
        quick,
        out,
        json_stdout,
        seed,
        threads,
        samples,
        warmup,
        prompt_tokens,
        layers,
        decode_tokens,
        max_load,
        gpu_probe,
        label,
        reuse_criterion,
        fn3_manifest,
    })
}

fn cmd_run(args: &[String]) -> i32 {
    let options = match parse_run_options(args) {
        Ok(options) => options,
        Err(error) if error == "__help__" => {
            print_usage();
            return 0;
        }
        Err(error) => {
            eprintln!("error: {error}");
            return 2;
        }
    };

    // Threads must be fixed before Rayon's global pool is first used.
    if let Some(threads) = options.threads {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }

    let started_unix_s = env::unix_now();
    let started = Instant::now();
    let mut warnings = Vec::new();
    let git = env::capture_git(&mut warnings);
    let build = env::capture_build(&mut warnings);
    let host = env::capture_host(options.max_load, &mut warnings);
    let gpu = env::capture_gpu(options.gpu_probe, &mut warnings);

    if !host.competing_build_pids.is_empty() {
        warnings.push(format!(
            "competing build detected (cargo/rustc pids {:?}); performance numbers may be distorted",
            host.competing_build_pids
        ));
    }
    if !host.competing_benchmark_pids.is_empty() {
        warnings.push(format!(
            "another fn3-bench process is running (pids {:?}); only one GPU benchmark may run at a time",
            host.competing_benchmark_pids
        ));
    }

    let prompt_tokens = options
        .prompt_tokens
        .unwrap_or(if options.quick { 8 } else { 128 });
    let layers = options.layers.unwrap_or(if options.quick { 1 } else { 12 });
    let decode_tokens = options
        .decode_tokens
        .unwrap_or(if options.quick { 5 } else { 32 });
    let config = BenchConfig {
        quick: options.quick,
        seed: options.seed,
        prompt_tokens,
        layers,
        decode_tokens,
        samples: options.samples,
        warmup: options.warmup,
        manifest_dir: env::manifest_dir().to_path_buf(),
        reuse_criterion: options.reuse_criterion,
        fn3_manifest: options.fn3_manifest,
    };

    let outcome = match adapters::run(options.adapter, options.suite, &config) {
        Ok(outcome) => outcome,
        Err(error) => {
            eprintln!("error: {error}");
            return 3;
        }
    };

    let run = RunInfo {
        run_id: env::unique_run_id(),
        timestamp_utc: env::rfc3339_utc(started_unix_s),
        started_unix_s,
        duration_s: started.elapsed().as_secs_f64(),
        adapter: options.adapter.as_str().to_string(),
        suite: options.suite.as_str().to_string(),
        quick: options.quick,
        seed: options.seed,
        label: options.label.clone(),
    };

    let mut result = RunResult::new(run);
    result.git = git;
    result.build = build;
    result.host = host;
    result.gpu = gpu;
    result.model = outcome.model;
    result.geometry = outcome.geometry;
    result.precision = outcome.precision;
    result.metrics = outcome.metrics;
    result.notes = outcome.notes;
    result.warnings.extend(warnings);
    result.warnings.extend(outcome.warnings);
    result.run.duration_s = started.elapsed().as_secs_f64();

    let json = match serde_json::to_string_pretty(&result) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("error: could not serialize result: {error}");
            return 4;
        }
    };

    let output_path = options.out.unwrap_or_else(|| {
        env::manifest_dir()
            .join("target/fn3-bench")
            .join(format!("{}-{}.json", result.run.suite, result.run.run_id))
    });

    if options.json_stdout || output_path.as_os_str() == "-" {
        println!("{json}");
    } else {
        if let Some(parent) = output_path.parent() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                eprintln!(
                    "error: could not create output directory {}: {error}",
                    parent.display()
                );
                return 4;
            }
        }
        if let Err(error) = std::fs::write(&output_path, format!("{json}\n")) {
            eprintln!("error: could not write {}: {error}", output_path.display());
            return 4;
        }
        println!("{}", render_summary(&result));
        println!("\nwrote {}", output_path.display());
    }

    let measured = result
        .metrics
        .iter()
        .filter(|metric| metric.is_ok())
        .count();
    if result.metrics.is_empty() {
        eprintln!("error: suite produced no metrics");
        6
    } else if measured == 0 {
        eprintln!("error: every metric in this suite is unavailable; see reasons in the JSON");
        5
    } else {
        0
    }
}

fn render_summary(result: &RunResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "FN3 benchmark {}  suite={} adapter={} quick={}\n",
        result.run.run_id, result.run.suite, result.run.adapter, result.run.quick
    ));
    out.push_str(&format!(
        "git   {}@{} dirty={:?}  build={}  rustc={}\n",
        result.git.branch.as_deref().unwrap_or("?"),
        result.git.commit.as_deref().unwrap_or("?"),
        result.git.dirty,
        result.build.profile,
        result.build.rustc_version.as_deref().unwrap_or("?")
    ));
    let features: Vec<String> = result
        .build
        .features
        .iter()
        .filter(|(_, enabled)| **enabled)
        .map(|(name, _)| name.clone())
        .collect();
    out.push_str(&format!("features  {}\n", features.join(",")));
    out.push_str(&format!(
        "host  {}  {}C/{}T  threads={}  load={:.2}/{:.2}{}\n",
        result.host.cpu_model.as_deref().unwrap_or("unknown CPU"),
        result.host.physical_cores.unwrap_or(0),
        result.host.logical_cores,
        result.host.threads_used,
        result.host.load.one.unwrap_or(0.0),
        result.host.load.threshold,
        if result.host.load.exceeded {
            " WARNING"
        } else {
            ""
        }
    ));
    let gpu = if result.gpu.devices.is_empty() {
        format!(
            "unavailable ({})",
            result.gpu.reason.as_deref().unwrap_or("unknown")
        )
    } else {
        result
            .gpu
            .devices
            .iter()
            .map(|device| format!("{} [{}]", device.name, device.device_type))
            .collect::<Vec<_>>()
            .join(", ")
    };
    out.push_str(&format!(
        "gpu   {gpu}  execution={}\n",
        result.gpu.execution_backend
    ));
    out.push_str(&format!(
        "model {} rev={} params={} weights={:.1} MiB\n",
        result.model.name,
        result.model.revision,
        result
            .model
            .parameter_count
            .map(|value| value.to_string())
            .unwrap_or_else(|| "?".to_string()),
        result.model.weights_bytes.unwrap_or(0) as f64 / (1024.0 * 1024.0)
    ));
    out.push_str(&format!(
        "geometry prompt={} decode={} layers={} hidden={} heads={}x{} vocab={} kv_cache={}\n",
        result.geometry.prompt_tokens,
        result.geometry.decode_tokens,
        result.geometry.num_layers,
        result.geometry.hidden_size,
        result.geometry.num_heads,
        result.geometry.head_dim,
        result.geometry.vocab_size,
        result.geometry.kv_cache
    ));
    out.push_str("\nmetrics:\n");
    out.push_str(&format!(
        "  {:<58} {:>10} {:>10} {:>7} {:>6}\n",
        "name", "status", "p50", "unit", "n"
    ));
    for metric in &result.metrics {
        let (status, value) = match metric.status {
            MetricStatus::Ok => (
                "ok",
                metric
                    .host_time
                    .as_ref()
                    .and_then(|block| block.stats.as_ref())
                    .map(|stats| format!("{:.4}", stats.p50))
                    .unwrap_or_else(|| "?".to_string()),
            ),
            MetricStatus::Unavailable => ("unavailable", "-".to_string()),
        };
        let count = metric
            .host_time
            .as_ref()
            .map(|block| block.sample_count.to_string())
            .unwrap_or_else(|| "0".to_string());
        out.push_str(&format!(
            "  {:<58} {:>10} {:>10} {:>7} {:>6}\n",
            metric.name, status, value, metric.unit, count
        ));
        if let Some(reason) = &metric.reason {
            out.push_str(&format!("      reason: {reason}\n"));
        }
    }
    if !result.notes.is_empty() {
        out.push_str("\nnotes:\n");
        for note in &result.notes {
            out.push_str(&format!("  - {note}\n"));
        }
    }
    if !result.warnings.is_empty() {
        out.push_str("\nwarnings:\n");
        for warning in &result.warnings {
            out.push_str(&format!("  ! {warning}\n"));
        }
    }
    out
}

struct CompareOptions {
    a: PathBuf,
    b: PathBuf,
    force: bool,
    stat: compare::Stat,
    json: bool,
}

fn parse_compare_options(args: &[String]) -> Result<CompareOptions, String> {
    let mut positional = Vec::new();
    let mut force = false;
    let mut stat = compare::Stat::P50;
    let mut json = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].clone();
        let flag = arg.split('=').next().unwrap_or("");
        match flag {
            "--force" => force = true,
            "--json" => json = true,
            "--stat" => {
                let value = take_value(args, &mut index, "--stat")?;
                stat = compare::Stat::parse(&value)
                    .ok_or_else(|| format!("unknown --stat `{value}`"))?;
            }
            "--help" | "-h" => return Err("__help__".to_string()),
            other if other.starts_with('-') => {
                return Err(format!("unknown compare option `{other}`"))
            }
            _ => positional.push(PathBuf::from(arg)),
        }
        index += 1;
    }
    if positional.len() != 2 {
        return Err("compare requires exactly two result JSON paths".to_string());
    }
    Ok(CompareOptions {
        a: positional[0].clone(),
        b: positional[1].clone(),
        force,
        stat,
        json,
    })
}

fn load_result(path: &PathBuf) -> Result<RunResult, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let result: RunResult = serde_json::from_str(&text)
        .map_err(|error| format!("invalid result JSON in {}: {error}", path.display()))?;
    Ok(result)
}

fn cmd_compare(args: &[String]) -> i32 {
    let options = match parse_compare_options(args) {
        Ok(options) => options,
        Err(error) if error == "__help__" => {
            print_usage();
            return 0;
        }
        Err(error) => {
            eprintln!("error: {error}");
            return 2;
        }
    };
    let a = match load_result(&options.a) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("error: {error}");
            return 1;
        }
    };
    let b = match load_result(&options.b) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("error: {error}");
            return 1;
        }
    };
    if a.schema_version != schema::SCHEMA_VERSION || b.schema_version != schema::SCHEMA_VERSION {
        eprintln!(
            "error: schema version mismatch (tool {}, A {}, B {}); refusing to compare",
            schema::SCHEMA_VERSION,
            a.schema_version,
            b.schema_version
        );
        return 2;
    }

    let mut report = compare::compare(&a, &b, options.stat);
    if report.blocked && options.force {
        for mismatch in &mut report.mismatches {
            if mismatch.level == "hard" {
                mismatch.level = "forced".to_string();
            }
        }
        report.blocked = false;
    }

    if options.json {
        match serde_json::to_string_pretty(&report) {
            Ok(json) => println!("{json}"),
            Err(error) => {
                eprintln!("error: could not serialize diff: {error}");
                return 1;
            }
        }
    } else {
        println!("{}", compare::render_table(&report));
    }

    if report.blocked {
        eprintln!(
            "\nerror: refusing to compare runs whose environment/geometry differ; pass --force to override"
        );
        2
    } else {
        0
    }
}

/// Keeps the schema field list explicit for the JSON Schema document and tests.
#[allow(dead_code)]
pub fn schema_field_groups() -> BTreeMap<&'static str, Vec<&'static str>> {
    let mut groups = BTreeMap::new();
    groups.insert(
        "run",
        vec![
            "run_id",
            "timestamp_utc",
            "started_unix_s",
            "duration_s",
            "adapter",
            "suite",
            "quick",
            "seed",
            "label",
        ],
    );
    groups.insert(
        "git",
        vec!["commit", "branch", "dirty", "dirty_entries", "describe"],
    );
    groups.insert(
        "build",
        vec![
            "package_version",
            "profile",
            "debug_assertions",
            "rustc_version",
            "cargo_version",
            "features",
            "target_arch",
            "target_os",
            "target_family",
            "command_line",
        ],
    );
    groups.insert(
        "host",
        vec![
            "os",
            "kernel",
            "arch",
            "cpu_model",
            "physical_cores",
            "logical_cores",
            "threads_used",
            "memory_total_bytes",
            "load",
            "competing_build_pids",
            "competing_benchmark_pids",
        ],
    );
    groups.insert(
        "gpu",
        vec![
            "status",
            "reason",
            "vulkan_instance_version",
            "devices",
            "selected_device_index",
            "execution_backend",
            "execution_reason",
        ],
    );
    groups.insert(
        "model",
        vec![
            "name",
            "revision",
            "source",
            "parameter_count",
            "weights_bytes",
            "seed",
            "checksums",
        ],
    );
    groups.insert(
        "geometry",
        vec![
            "batch",
            "prompt_tokens",
            "decode_tokens",
            "context_capacity",
            "hidden_size",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "num_layers",
            "intermediate_size",
            "vocab_size",
            "attention",
            "kv_cache",
        ],
    );
    groups.insert(
        "precision",
        vec![
            "weights_dtype",
            "activations_dtype",
            "accumulation_dtype",
            "quantization",
        ],
    );
    groups.insert(
        "metric",
        vec![
            "name",
            "category",
            "unit",
            "status",
            "reason",
            "clock",
            "host_time",
            "device_time",
            "counters",
            "metadata",
        ],
    );
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Metric;

    #[test]
    fn schema_groups_cover_every_metric_field() {
        let groups = schema_field_groups();
        for field in [
            "name",
            "category",
            "unit",
            "status",
            "reason",
            "clock",
            "host_time",
            "device_time",
            "counters",
            "metadata",
        ] {
            assert!(groups["metric"].contains(&field), "missing {field}");
        }
    }

    #[test]
    fn metric_status_serializes_as_snake_case() {
        let metric = Metric::unavailable("x", "memory", "MiB", "not here");
        let json = serde_json::to_string(&metric).expect("serialize");
        assert!(json.contains("\"status\":\"unavailable\""));
    }

    #[test]
    fn json_schema_metric_fields_match_the_rust_schema() {
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../../docs/reports/fn3-04/result-schema.json"))
                .expect("schema JSON parses");
        let required: Vec<&str> = schema["$defs"]["metric"]["required"]
            .as_array()
            .expect("metric.required")
            .iter()
            .map(|value| value.as_str().expect("string"))
            .collect();
        let mut expected = schema_field_groups()["metric"].clone();
        expected.sort_unstable();
        let mut actual = required.clone();
        actual.sort_unstable();
        assert_eq!(
            actual, expected,
            "result-schema.json drifted from schema.rs"
        );

        let top_required: Vec<&str> = schema["required"]
            .as_array()
            .expect("top-level required")
            .iter()
            .map(|value| value.as_str().expect("string"))
            .collect();
        for group in [
            "run",
            "git",
            "build",
            "host",
            "gpu",
            "model",
            "geometry",
            "precision",
        ] {
            assert!(top_required.contains(&group), "missing top-level {group}");
        }
    }
}
