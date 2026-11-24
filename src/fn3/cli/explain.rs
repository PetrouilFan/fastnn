//! `explain`: the stored plan, segment by segment.
//!
//! `inspect` answers "what is in this file"; `explain` answers "what will run,
//! on which engine, and why did a segment fall back". It reads the stored plan
//! and never maps the weight payload or opens a session.

use std::io::Write;

use crate::inference::model::format::Artifact;

use super::args::ExplainArgs;
use super::{format_error, thousands, CliError};

/// Run `explain`.
pub fn execute(args: &ExplainArgs, out: &mut dyn Write) -> Result<(), CliError> {
    let artifact = Artifact::open(&args.artifact).map_err(format_error)?;
    let plan = match args.target.as_deref() {
        Some(target) => artifact.plan_for(target).ok_or_else(|| {
            let available: Vec<&str> = artifact.plans().iter().map(|plan| plan.target()).collect();
            CliError::model(format!(
                "`{target}` is not a target in this artifact; available: {}",
                available.join(", ")
            ))
            .with_next("run `fastnn-infer inspect <artifact>` to list the stored plans")
        })?,
        None => artifact.default_plan(),
    };
    let target = match args.target.as_deref() {
        Some(target) => target,
        None => artifact.default_target(),
    };
    let counts = plan.engine_counts();
    let _ = writeln!(out, "fastnn-infer explain");
    let _ = writeln!(out, "  artifact: {}", args.artifact.display());
    let _ = writeln!(out, "  target:   {target}");
    let _ = writeln!(
        out,
        "  plan:     {} segments ({} oneDNN, {} scalar), {} fusions, {} fallbacks",
        counts.total(),
        counts.one_dnn,
        counts.scalar,
        plan.fusions.len(),
        plan.fallbacks.len()
    );
    let _ = writeln!(
        out,
        "            arena {} B (peak {} B), layout {} (runtime conversions are chosen at \
         session open)",
        thousands(plan.arena_bytes as u64),
        thousands(plan.peak_bytes as u64),
        plan.layout.name()
    );

    let _ = writeln!(out, "  inputs:");
    for &id in &plan.inputs {
        let slot = plan
            .slot(id)
            .map_err(|error| CliError::inference(error.to_string()))?;
        let name = slot.name.as_deref().unwrap_or("<unnamed>");
        let _ = writeln!(out, "    {name}: {}, {} B", slot.ty, slot.bytes);
    }
    let _ = writeln!(out, "  outputs:");
    for &id in &plan.outputs {
        let slot = plan
            .slot(id)
            .map_err(|error| CliError::inference(error.to_string()))?;
        let name = slot.name.as_deref().unwrap_or("<unnamed>");
        let _ = writeln!(out, "    {name}: {}, {} B", slot.ty, slot.bytes);
    }

    let _ = writeln!(out, "  segments:");
    let _ = writeln!(
        out,
        "    {:>4}  {:<7}  {:<18}  {:<12}  {:>10}  source",
        "#", "engine", "op", "outputs", "bytes"
    );
    for (index, segment) in plan.segments.iter().enumerate() {
        let engine = if segment.engine == crate::inference::backend::SegmentEngine::Scalar {
            "scalar"
        } else {
            "onednn"
        };
        let outputs: Vec<String> = segment
            .outputs
            .iter()
            .map(|id| format!("v{}", id.index()))
            .collect();
        let bytes: usize = segment
            .outputs
            .iter()
            .filter_map(|&id| plan.slot(id).ok())
            .map(|slot| slot.bytes)
            .sum();
        let _ = writeln!(
            out,
            "    {index:>4}  {engine:<7}  {:<18}  {:<12}  {:>10}  {}",
            segment.op.to_string(),
            outputs.join(","),
            thousands(bytes as u64),
            segment.loc.describe()
        );
    }

    if !plan.fallbacks.is_empty() {
        let _ = writeln!(out, "  fallbacks:");
        for fallback in &plan.fallbacks {
            let _ = writeln!(
                out,
                "    segment {:<4}  {:<12}  {}",
                fallback.segment, fallback.op, fallback.reason
            );
        }
    }
    if !plan.fusions.is_empty() {
        let _ = writeln!(out, "  fusions:");
        for fusion in &plan.fusions {
            let chain: Vec<String> = fusion.chain().iter().map(ToString::to_string).collect();
            let _ = writeln!(
                out,
                "    segment {:<4}  {} <- {} (removed {})",
                fusion.segment,
                fusion.name(),
                chain.join(", "),
                fusion.removed()
            );
        }
    }
    if !plan.reorders.is_empty() {
        let _ = writeln!(out, "  reorders:");
        for reorder in &plan.reorders {
            let _ = writeln!(
                out,
                "    segment {:<4}  {}  {} B",
                reorder.segment,
                reorder.direction.name(),
                thousands(reorder.bytes as u64)
            );
        }
    }
    let _ = writeln!(
        out,
        "  weights:  {} B prepacked at session open (payload not read here)",
        thousands(artifact.weight_bytes())
    );
    Ok(())
}
