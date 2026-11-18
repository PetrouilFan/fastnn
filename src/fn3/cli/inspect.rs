//! `inspect`: identify an artifact before trusting or running it.
//!
//! The header path calls [`PreparedHeader::parse`] and stops. `--sections`
//! additionally decodes META/GEOM/ENV/VALS/NODE/WGHT through the strict reader
//! and reports the model's geometry and bounds. Session state is never loaded.

use std::io::Write;

use crate::fn3::import::PositionEmbedding;
use crate::fn3::model::format::{FormatErrorKind, PreparedArtifact, PreparedHeader, HEADER_BYTES};
use crate::fn3::Symbol;

use super::args::InspectArgs;
use super::digest::sha256_hex;
use super::{thousands, CliError};

/// Run `inspect`.
pub fn execute(args: &InspectArgs, out: &mut dyn Write) -> Result<(), CliError> {
    let bytes = std::fs::read(&args.artifact).map_err(|error| {
        CliError::model(format!(
            "cannot read artifact `{}`: {error}",
            args.artifact.display()
        ))
        .with_next(
            "check the path, or create the artifact with `fn3-infer prepare --onnx <model.onnx> \
             --out <model.fnn3>`",
        )
    })?;

    let header = PreparedHeader::parse(&bytes).map_err(|error| {
        let message = format!(
            "`{}` is not a usable prepared model: {error}",
            args.artifact.display()
        );
        let next = match error.kind() {
            FormatErrorKind::UnsupportedVersion { found, supported } => format!(
                "this file uses format version {found}; this build reads version {supported} \
                 only. Use a matching `fn3-infer`, or re-prepare the ONNX with this build"
            ),
            FormatErrorKind::BadMagic { .. } => {
                "the first eight bytes must be `FNN3PREP`. A 2.x `.fnnc` file is a different \
                 format and is never read by FastNN 3.0; prepare the source ONNX instead"
                    .to_string()
            }
            FormatErrorKind::UnsupportedDtypeEncoding { found, supported } => format!(
                "this file uses dtype encoding {found}; this build reads encoding {supported} \
                 only. Rebuild the tool or re-prepare with a build that uses this encoding"
            ),
            _ => "the file is truncated, corrupt or not an artifact; re-run `fn3-infer prepare` \
                  from the source ONNX"
                .to_string(),
        };
        CliError::model(message).with_next(next)
    })?;

    let _ = writeln!(out, "fn3-infer inspect");
    let _ = writeln!(out, "  artifact: {}", args.artifact.display());
    let _ = writeln!(out, "    bytes:   {}", thousands(header.file_bytes));
    let _ = writeln!(out, "    sha256:  {}", sha256_hex(&bytes));
    let _ = writeln!(
        out,
        "    header:  magic FNN3PREP, version {}, header {} B, dtype encoding {}",
        header.version, HEADER_BYTES, header.dtype_encoding
    );
    let _ = writeln!(
        out,
        "    payload: {} bytes, crc32 {:#010x}",
        thousands(header.payload_bytes),
        header.payload_crc32
    );
    let _ = writeln!(
        out,
        "    status:  framing valid; payload checksum {}",
        if args.sections {
            "verified below"
        } else {
            "not recomputed (pass --sections for a full decode)"
        }
    );

    if args.sections {
        let artifact = PreparedArtifact::from_bytes(&bytes).map_err(|error| {
            let next = if error.is_version_mismatch() {
                "this file was written by an incompatible build; use a matching `fn3-infer` or \
                 re-prepare the ONNX"
            } else if error.is_corruption() {
                "the payload does not match its checksum; re-run `fn3-infer prepare` from the \
                 source ONNX"
            } else {
                "the file is well framed but inconsistent; re-run `fn3-infer prepare`"
            };
            CliError::model(format!(
                "`{}` failed the full decode: {error}",
                args.artifact.display()
            ))
            .with_next(next)
        })?;
        print_sections(&artifact, out);
    }

    Ok(())
}

fn print_sections(artifact: &PreparedArtifact, out: &mut dyn Write) {
    let metadata = artifact.metadata();
    let geometry = artifact.geometry();
    let envelope = artifact.envelope();
    let graph = artifact.graph();
    let epsilon = geometry
        .layer_norm_epsilon
        .map(|value| format!("{value:e}"))
        .unwrap_or_else(|| "not detected".to_string());
    let positions = match geometry.positions {
        PositionEmbedding::Learned { max_positions, .. } => {
            format!("learned ({max_positions} rows)")
        }
    };

    let _ = writeln!(
        out,
        "    sections: META, GEOM, ENV, VALS ({} values), NODE ({} ops), WGHT ({} weights, {} bytes)",
        graph.value_count(),
        graph.nodes().len(),
        artifact.weights().len(),
        thousands(artifact.weight_bytes())
    );
    let _ = writeln!(
        out,
        "    metadata: ir_version={} opset={}{} producer=\"{}\" {} graph=\"{}\"",
        metadata.ir_version,
        metadata.opset.domain,
        metadata.opset.version,
        metadata.producer,
        metadata.producer_version,
        metadata.graph_name
    );
    let _ = writeln!(
        out,
        "    geometry: layers={} hidden={} mlp={} heads={}/{} head_dim={} vocab={}",
        geometry.layers,
        geometry.hidden_size,
        geometry.mlp_hidden_size,
        geometry.query_heads,
        geometry.kv_heads,
        geometry.head_dim,
        thousands(u64::from(geometry.vocab_size))
    );
    let _ = writeln!(
        out,
        "    model:    positions={} epsilon={} kv_layers={} graph_inputs={} graph_outputs={}",
        positions,
        epsilon,
        geometry.kv.len(),
        graph.inputs().len(),
        graph.outputs().len()
    );
    let _ = writeln!(
        out,
        "    envelope: batch {}..={}, seq {}..={}, past {}..={}, total {}..={}",
        envelope.get(Symbol::Batch).min(),
        envelope.get(Symbol::Batch).max(),
        envelope.get(Symbol::Seq).min(),
        envelope.get(Symbol::Seq).max(),
        envelope.get(Symbol::Past).min(),
        envelope.get(Symbol::Past).max(),
        envelope.get(Symbol::Total).min(),
        envelope.get(Symbol::Total).max(),
    );
    let _ = writeln!(out, "    payload checksum: verified (crc32 matches)");
}

#[cfg(test)]
mod tests {
    use crate::fn3::cli::{run, EXIT_MODEL};

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "fastnn-fn3-cli-inspect-{}-{name}",
            std::process::id()
        ))
    }

    fn header_bytes(version: u16, magic: &[u8; 8]) -> Vec<u8> {
        let mut bytes = vec![0u8; 64];
        bytes[..8].copy_from_slice(magic);
        bytes[8..10].copy_from_slice(&version.to_le_bytes());
        bytes[10..12].copy_from_slice(&64u16.to_le_bytes());
        bytes[12..16].copy_from_slice(&1u32.to_le_bytes());
        bytes[24..32].copy_from_slice(&64u64.to_le_bytes());
        bytes[32..40].copy_from_slice(&0u64.to_le_bytes());
        bytes
    }

    fn inspect(args: &[&str]) -> (i32, String, String) {
        let argv: Vec<String> = args.iter().map(|item| (*item).to_string()).collect();
        let mut out = Vec::new();
        let mut err = Vec::new();
        let code = run(&argv, &mut out, &mut err);
        (
            code,
            String::from_utf8_lossy(&out).into_owned(),
            String::from_utf8_lossy(&err).into_owned(),
        )
    }

    #[test]
    fn a_future_version_names_both_versions_and_the_next_step() {
        let path = scratch("future.fnn3");
        std::fs::write(&path, header_bytes(99, b"FNN3PREP")).expect("write");
        let (code, _, err) = inspect(&["inspect", path.to_str().expect("path")]);
        let _ = std::fs::remove_file(&path);
        assert_eq!(code, EXIT_MODEL);
        assert!(err.contains("version 99"), "{err}");
        assert!(err.contains("next:"), "{err}");
    }

    #[test]
    fn a_foreign_magic_is_rejected_and_explains_that_fnnc_is_not_read() {
        let path = scratch("legacy.fnn3");
        std::fs::write(&path, header_bytes(1, b"FNN2PREP")).expect("write");
        let (code, _, err) = inspect(&["inspect", path.to_str().expect("path")]);
        let _ = std::fs::remove_file(&path);
        assert_eq!(code, EXIT_MODEL);
        assert!(err.contains("FNN3PREP"), "{err}");
        assert!(err.contains(".fnnc"), "{err}");
    }

    #[test]
    fn a_missing_file_names_the_path_and_the_prepare_command() {
        let path = scratch("missing.fnn3");
        let (code, _, err) = inspect(&["inspect", path.to_str().expect("path")]);
        assert_eq!(code, EXIT_MODEL);
        assert!(err.contains("fn3-infer prepare"), "{err}");
    }
}
