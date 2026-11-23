//! `doctor`: diagnose an ONNX model before preparing it.
//!
//! The report is produced by the importer and the scalar planner; nothing is
//! executed and no artifact is written. An import failure is a report listing
//! every problem, each with its node, not a single first error.

use std::io::Write;

use crate::inference::doctor;

use super::args::DoctorArgs;
use super::{thousands, CliError};

/// Run `doctor`.
pub fn execute(args: &DoctorArgs, out: &mut dyn Write) -> Result<(), CliError> {
    let report = doctor(&args.onnx)
        .map_err(|error| CliError::model(format!("cannot diagnose the model: {error}")))?;

    let _ = writeln!(out, "fastnn-infer doctor");
    let _ = writeln!(out, "  source: {}", args.onnx.display());
    let _ = writeln!(
        out,
        "  result: {}",
        if report.ok {
            "ready to prepare"
        } else {
            "not ready to prepare"
        }
    );
    if report.nodes > 0 {
        let _ = writeln!(
            out,
            "  graph:  {} typed nodes, {} values, {} weights ({} B payload)",
            report.nodes,
            report.values,
            report.weights,
            thousands(report.weight_bytes)
        );
        let dtypes: Vec<String> = report
            .dtypes
            .iter()
            .map(|entry| format!("{}={}", entry.dtype, entry.values))
            .collect();
        let _ = writeln!(out, "  dtypes: {}", dtypes.join(", "));
        if report.dynamic_symbols.is_empty() {
            let _ = writeln!(out, "  shapes: every extent is static");
        } else {
            let symbols: Vec<String> = report
                .dynamic_symbols
                .iter()
                .map(|entry| format!("{}={}", entry.symbol, entry.values))
                .collect();
            let _ = writeln!(out, "  shapes: dynamic symbols {}", symbols.join(", "));
        }
        let _ = writeln!(
            out,
            "  memory: weights {} B + arena {} B = {} B resident per session (peak live {} B)",
            thousands(report.weight_bytes),
            thousands(report.arena_bytes),
            thousands(report.expected_memory_bytes),
            thousands(report.peak_bytes)
        );
    }
    if report.problems.is_empty() {
        let _ = writeln!(out, "  problems: none");
    } else {
        let _ = writeln!(out, "  problems: {}", report.problems.len());
        for (index, problem) in report.problems.iter().enumerate() {
            let _ = writeln!(
                out,
                "    {}. [{}] {}",
                index + 1,
                problem.category,
                problem.message
            );
        }
    }
    if !report.ok {
        return Err(CliError::model(format!(
            "`{}` is not ready to prepare ({} problem(s)); see the list above",
            args.onnx.display(),
            report.problems.len()
        ))
        .with_next(
            "fix the named nodes or export the model with the operator set this build implements",
        ));
    }
    Ok(())
}
