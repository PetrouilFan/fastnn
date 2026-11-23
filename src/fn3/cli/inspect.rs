//! `inspect`: describe an artifact without mapping or running it.
//!
//! The reader in `model::format::inspect_file` opens a `File`, reads the header,
//! the directory, `GRAPH`, `PLANS`, `QUANT`, `HASHES` and the `WEIGHTS` *table*,
//! and never reads the weight payload. No `Mmap` is constructed. The report
//! prints `mapped bytes: 0` and the number of bytes actually read as the
//! evidence.

use std::io::Write;

use crate::inference::model::format::{inspect_file, HEADER_BYTES};

use super::args::InspectArgs;
use super::{format_error, thousands, CliError};

/// Run `inspect`.
pub fn execute(args: &InspectArgs, out: &mut dyn Write) -> Result<(), CliError> {
    let report = inspect_file(&args.artifact).map_err(format_error)?;

    let _ = writeln!(out, "fastnn-infer inspect");
    let _ = writeln!(out, "  artifact: {}", args.artifact.display());
    let _ = writeln!(out, "    bytes:   {}", thousands(report.file_bytes));
    let _ = writeln!(
        out,
        "    header:  magic FNN3ART2, version {}, {} B, {} sections, {} target profiles",
        report.format_version, HEADER_BYTES, 5, report.target_count
    );
    let _ = writeln!(
        out,
        "    source:  ONNX sha256 {}",
        hex(&report.source_onnx_sha256)
    );
    let _ = writeln!(
        out,
        "  graph:     {} nodes, {} values, {} inputs, {} outputs, {} weights",
        report.nodes, report.values, report.inputs, report.outputs, report.weights
    );
    let _ = writeln!(
        out,
        "  weights:   {} B payload, payload range {}..{}",
        thousands(report.weight_bytes),
        thousands(report.sections[3].offset),
        thousands(report.sections[3].offset + report.sections[3].length)
    );
    let _ = writeln!(
        out,
        "  quant:     precision {}, {} tensor entries",
        report.quant.precision(),
        report.quant.tensors().len()
    );
    let _ = writeln!(out, "  plans:");
    for plan in &report.plans {
        let _ = writeln!(
            out,
            "    {:<14} precision={} layout={} segments={} (unfused {}) fusions={} reorders={} ({}) arena={} B peak={} B engines scalar={} onednn={}",
            plan.target,
            plan.precision,
            plan.layout.name(),
            plan.segments,
            plan.unfused_segments,
            plan.fusions,
            plan.reorders,
            thousands(plan.reorder_bytes),
            thousands(plan.arena_bytes),
            thousands(plan.peak_bytes),
            plan.engines.scalar,
            plan.engines.one_dnn
        );
    }
    let _ = writeln!(out, "  sections:");
    for section in &report.sections {
        let _ = writeln!(
            out,
            "    {:<8} offset {:>12} length {:>12} sha256 {} {}",
            section.name,
            thousands(section.offset),
            thousands(section.length),
            section.sha256_hex(),
            if section.verified {
                "verified"
            } else {
                "stored (payload not read)"
            }
        );
    }
    let _ = writeln!(
        out,
        "  evidence:  mapped bytes {} (no Mmap), bytes read {} of {} (weight payload untouched)",
        report.mapped_bytes,
        thousands(report.bytes_read),
        thousands(report.file_bytes)
    );
    Ok(())
}

fn hex(bytes: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in bytes {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}
