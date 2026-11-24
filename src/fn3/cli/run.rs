//! `run`: one forward from the artifact, with explicit named inputs.
//!
//! Inputs are `f32` values, exactly the element count the plan's slot declares.
//! `zeros` and `ones` exist for smoke runs; a file is raw little-endian `f32`
//! bytes. Nothing is inferred from the file extension: a short or long file is
//! a typed error, not a padded input.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::inference::model::format::Artifact;
use crate::inference::runtime::{HostTensor, Session};

use super::args::{InputArg, RunArgs};
use super::{format_error, thousands, CliError};

/// Run `run`.
pub fn execute(args: &RunArgs, out: &mut dyn Write) -> Result<(), CliError> {
    let artifact = Artifact::open(&args.model).map_err(format_error)?;
    let target = args
        .target
        .as_deref()
        .unwrap_or_else(|| artifact.default_target());
    if artifact.plan_for(target).is_none() {
        return Err(CliError::model(format!(
            "`{}` has no target `{target}`",
            args.model.display()
        ))
        .with_next("pass `--target` one of the names printed by `fastnn-infer inspect`"));
    }

    let mut session = if target == artifact.default_target() {
        Session::from_artifact(&artifact, args.threads.threading())
    } else {
        Session::from_artifact_for(&artifact, target, args.threads.threading())
    }
    .map_err(|error| CliError::inference(format!("the stored plan cannot be opened: {error}")))?;

    let expected_inputs: Vec<String> = session
        .plan()
        .inputs
        .iter()
        .filter_map(|&id| session.plan().slot(id).ok()?.name.clone())
        .collect();
    if args.inputs.is_empty() && !expected_inputs.is_empty() {
        return Err(CliError::usage(format!(
            "`run` needs one `--input name=...` per graph input; this model has {}",
            expected_inputs.join(", ")
        )));
    }
    for supplied in &args.inputs {
        if !expected_inputs.iter().any(|name| name == supplied.name()) {
            return Err(CliError::usage(format!(
                "the model has no input named `{}`; inputs are {}",
                supplied.name(),
                expected_inputs.join(", ")
            )));
        }
    }

    let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(expected_inputs.len());
    for name in &expected_inputs {
        let slot = session.plan().input_slot(name).ok_or_else(|| {
            CliError::inference(format!("internal: input `{name}` is not in the plan"))
        })?;
        let plan_slot = session
            .plan()
            .slot(slot)
            .map_err(|error| CliError::inference(error.to_string()))?;
        if plan_slot.ty.dtype != crate::inference::DType::F32 {
            return Err(CliError::inference(format!(
                "input `{name}` is {}, but this CLI feeds f32 inputs only",
                plan_slot.ty.dtype
            )));
        }
        let element_count = plan_slot.bytes / 4;
        let argument = args
            .inputs
            .iter()
            .find(|argument| argument.name() == name)
            .ok_or_else(|| CliError::usage(format!("missing `--input {name}=...`")))?;
        let values = match argument {
            InputArg::Zeros(_) => vec![0.0f32; element_count],
            InputArg::Ones(_) => vec![1.0f32; element_count],
            InputArg::File(_, path) => read_raw_f32(path, element_count, name)?,
        };
        owned.push((name.clone(), values));
    }
    let borrowed: Vec<(&str, &[f32])> = owned
        .iter()
        .map(|(name, values)| (name.as_str(), values.as_slice()))
        .collect();

    let forward_error = |error: crate::inference::Fn3Error| {
        CliError::inference(format!("the forward failed: {error}")).with_next(
            "run `fastnn-infer doctor` on the source ONNX, and `inspect` on the artifact",
        )
    };

    // A warm-up run exists because the first forward pays for page faults on
    // the mapped weights and for oneDNN's first primitive creation; measuring
    // it alongside steady state would report a number no user experiences.
    for _ in 0..args.warmup {
        session.run(&borrowed).map_err(forward_error)?;
    }

    // `execute` writes the arena and returns nothing; the outputs are copied
    // once, after the series. `run` would allocate and copy every output on
    // every measured forward, which is a cost of building a result rather than
    // of running the model.
    let mut series: Vec<f64> = Vec::with_capacity(args.repeat);
    for _ in 0..args.repeat {
        let started = std::time::Instant::now();
        session.execute(&borrowed).map_err(forward_error)?;
        series.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let mut outputs: Vec<(String, Vec<f32>)> = Vec::new();
    for (name, tensor) in session.outputs().map_err(forward_error)? {
        match tensor {
            crate::inference::runtime::HostTensor::F32(values) => outputs.push((name, values)),
            other => {
                return Err(CliError::inference(format!(
                    "output `{name}` is {}, but this CLI prints f32 outputs only",
                    other.dtype()
                )))
            }
        }
    }
    let elapsed =
        std::time::Duration::from_secs_f64(series.iter().copied().fold(0.0f64, f64::max) / 1e3);
    let peak_rss = peak_rss_kib();

    let _ = writeln!(out, "fastnn-infer run");
    let _ = writeln!(out, "  artifact: {}", args.model.display());
    let _ = writeln!(
        out,
        "  target:   {target} ({}), threads {}, workers {}",
        session.plan().layout.name(),
        args.threads.render(),
        session.worker_count()
    );
    let _ = writeln!(
        out,
        "  plan:     {} segments, {} fusions, {} reorders, arena {} B",
        session.plan().segments.len(),
        session.fusions().len(),
        session.plan().reorders.len(),
        thousands(session.arena().bytes() as u64)
    );
    if series.len() == 1 {
        let _ = writeln!(out, "  forward: {:.3} ms", elapsed.as_secs_f64() * 1e3);
    } else {
        let mut sorted = series.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let _ = writeln!(out, "  forward: median {median:.3} ms, p95 {p95:.3} ms");
        let _ = writeln!(
            out,
            "  series:  {} forwards, min {:.3}, mean {:.3}, max {:.3} ms (warm-up {})",
            series.len(),
            sorted[0],
            mean,
            sorted[sorted.len() - 1],
            args.warmup
        );
        if let Some(rss) = peak_rss {
            let _ = writeln!(out, "  memory:  peak RSS {rss} KiB");
        }
    }

    if let Some(directory) = &args.output_dir {
        std::fs::create_dir_all(directory).map_err(|error| {
            CliError::model(format!(
                "cannot create output directory `{}`: {error}",
                directory.display()
            ))
        })?;
    }
    for (name, values) in &outputs {
        let slot = session
            .plan()
            .output_slot(name)
            .and_then(|id| session.plan().slot(id).ok());
        let shape = slot
            .and_then(|slot| slot.ty.static_shape())
            .map(|shape| {
                shape
                    .dims()
                    .iter()
                    .map(|dim| dim.to_string())
                    .collect::<Vec<_>>()
                    .join("x")
            })
            .unwrap_or_else(|| "?".to_string());
        let mut finite = 0usize;
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        for &value in values {
            if value.is_finite() {
                finite += 1;
                minimum = minimum.min(value);
                maximum = maximum.max(value);
                sum += f64::from(value);
            }
        }
        let mean = if finite == 0 {
            f32::NAN
        } else {
            (sum / finite as f64) as f32
        };
        let _ = writeln!(
            out,
            "  output {name}: shape [{shape}], {} elements, finite {}/{}, min {minimum}, max {maximum}, mean {mean}",
            values.len(),
            finite,
            values.len()
        );
        if let Some(directory) = &args.output_dir {
            let path = safe_output_path(directory, name);
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(&path, bytes).map_err(|error| {
                CliError::model(format!("cannot write `{}`: {error}", path.display()))
            })?;
            let _ = writeln!(out, "    wrote: {}", path.display());
        }
    }
    // Typed outputs (i64 indices, bool masks) are reported without pretending
    // they are f32.
    for (name, tensor) in session.outputs().unwrap_or_default() {
        if outputs.iter().any(|(existing, _)| existing == &name) {
            continue;
        }
        let _ = writeln!(
            out,
            "  output {name}: {} elements, dtype {} (typed; not written by --output-dir)",
            tensor.len(),
            match tensor {
                HostTensor::F32(_) => "f32",
                HostTensor::I64(_) => "i64",
                HostTensor::I32(_) => "i32",
                HostTensor::Bool(_) => "bool",
            }
        );
    }
    Ok(())
}

fn read_raw_f32(path: &Path, expected: usize, name: &str) -> Result<Vec<f32>, CliError> {
    let bytes = std::fs::read(path).map_err(|error| {
        CliError::model(format!(
            "cannot read input `{name}` from `{}`: {error}",
            path.display()
        ))
    })?;
    let expected_bytes = expected
        .checked_mul(4)
        .ok_or_else(|| CliError::inference(format!("input `{name}` element count overflows")))?;
    if bytes.len() != expected_bytes {
        return Err(CliError::model(format!(
            "input `{name}` is {} bytes, but the model needs {expected} f32 elements \
             ({expected_bytes} bytes)",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// A filesystem-safe name for an output tensor.
fn safe_output_path(directory: &Path, name: &str) -> PathBuf {
    let mut safe = String::with_capacity(name.len() + 4);
    for character in name.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
            safe.push(character);
        } else {
            safe.push('_');
        }
    }
    if safe.is_empty() {
        safe.push_str("output");
    }
    directory.join(format!("{safe}.f32"))
}

/// Peak resident set size of this process, from `/proc/self/status` on Linux.
///
/// A benchmark claim needs memory next to latency, and reading the kernel's own
/// high-water mark is the cheapest way for the CLI to report the process it
/// just measured. Other platforms return `None` rather than a guess.
#[cfg(target_os = "linux")]
fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(value) = line.strip_prefix("VmHWM:") {
            return value.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn peak_rss_kib() -> Option<u64> {
    None
}
