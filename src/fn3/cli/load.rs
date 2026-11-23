//! `load`: map an artifact, validate it and build the recorded session.
//!
//! This command exists to make the deployment path inspectable: it reports the
//! plan the file carries, the engine split, the arena and the fusions without
//! executing a single segment.

use std::io::Write;

use crate::inference::model::format::Artifact;
use crate::inference::runtime::Session;

use super::args::LoadArgs;
use super::{format_error, thousands, CliError};

/// Run `load`.
pub fn execute(args: &LoadArgs, out: &mut dyn Write) -> Result<(), CliError> {
    let artifact = Artifact::open(&args.model).map_err(format_error)?;
    let target = args
        .target
        .as_deref()
        .unwrap_or_else(|| artifact.default_target());
    if artifact.plan_for(target).is_none() {
        let available: Vec<&str> = artifact.plans().iter().map(|plan| plan.target()).collect();
        return Err(CliError::model(format!(
            "`{}` has no target `{target}`; the artifact carries {}",
            args.model.display(),
            available.join(", ")
        ))
        .with_next("pass `--target` one of the names printed by `fastnn-infer inspect`"));
    }

    let _ = writeln!(out, "fastnn-infer load");
    let _ = writeln!(out, "  artifact: {}", args.model.display());
    let _ = writeln!(
        out,
        "    bytes:   {} (mapped {}, header version {})",
        thousands(artifact.header().file_bytes),
        thousands(artifact.mapped_bytes()),
        artifact.header().version
    );
    let _ = writeln!(
        out,
        "    source:  ONNX sha256 {}",
        artifact.header().source_sha256_hex()
    );
    let _ = writeln!(
        out,
        "    graph:   {} nodes, {} values, {} inputs, {} outputs, {} weights ({} bytes)",
        artifact.graph().node_count(),
        artifact.graph().value_count(),
        artifact.inputs().len(),
        artifact.outputs().len(),
        artifact.weights().len(),
        thousands(artifact.weight_bytes())
    );
    let _ = writeln!(
        out,
        "    quant:   precision {}, {} entries",
        artifact.quant().precision(),
        artifact.quant().tensors().len()
    );
    let _ = writeln!(out, "  target: {target}");
    for plan in artifact.plan_summaries() {
        let _ = writeln!(
            out,
            "    {:<14} segments={} (unfused {}) fusions={} reorders={} ({}) layout={} arena={} B peak={} B engines scalar={} onednn={}",
            plan.target,
            plan.segments,
            plan.unfused_segments,
            plan.fusions,
            plan.reorders,
            thousands(plan.reorder_bytes),
            plan.layout.name(),
            thousands(plan.arena_bytes),
            thousands(plan.peak_bytes),
            plan.engines.scalar,
            plan.engines.one_dnn
        );
    }

    let session = if target == artifact.default_target() {
        Session::from_artifact(&artifact, args.threads.threading())
    } else {
        Session::from_artifact_for(&artifact, target, args.threads.threading())
    }
    .map_err(|error| {
        CliError::inference(format!("the stored plan cannot be opened: {error}")).with_next(
            "run `fastnn-infer inspect` to check the plan and `fastnn-infer doctor` on the source \
             if the model should be prepared with another target",
        )
    })?;

    let plan = session.plan();
    let _ = writeln!(
        out,
        "  session: workers={} arena={} B segments={} fallbacks={}",
        session.worker_count(),
        thousands(session.arena().bytes() as u64),
        plan.segments.len(),
        plan.fallbacks.len()
    );
    for fallback in &plan.fallbacks {
        let _ = writeln!(
            out,
            "    fallback segment {} ({}): {}",
            fallback.segment, fallback.op, fallback.reason
        );
    }
    if let Some(report) = session.one_dnn_report() {
        let _ = writeln!(
            out,
            "    onednn: {} ({}), segments={}, prepacked={} B, reorders={} ({} B), parallel={}",
            report.version,
            report.cpu_runtime,
            report.segments,
            thousands(report.prepacked_bytes as u64),
            report.reorders,
            thousands(report.reorder_bytes as u64),
            report.parallel_blocks
        );
    }
    let _ = writeln!(
        out,
        "  inputs:  {}",
        plan.inputs
            .iter()
            .filter_map(|&id| plan.slot(id).ok()?.name.as_deref())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(
        out,
        "  outputs: {}",
        plan.outputs
            .iter()
            .filter_map(|&id| plan.slot(id).ok()?.name.as_deref())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(out, "  executed: nothing (load only opens the plan)");
    Ok(())
}
