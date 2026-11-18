//! `prepare`: strict ONNX import -> versioned artifact, atomically written.
//!
//! The write is not trusted until it has been read back through the strict
//! reader: a crash or a bad filesystem that leaves a partial file cannot be
//! reported as success. The source ONNX is never modified and no network access
//! happens here.

use std::io::Write;
use std::path::Path;

use crate::fn3::import::{import_onnx_model, ModelImport, PositionEmbedding};
use crate::fn3::model::format::PreparedArtifact;
use crate::fn3::model::Planner;
use crate::fn3::{CpuBackend, Symbol};

use super::args::PrepareArgs;
use super::digest::sha256_hex;
use super::{thousands, CliError};

/// Run `prepare`.
pub fn execute(args: &PrepareArgs, out: &mut dyn Write) -> Result<(), CliError> {
    if args.onnx == args.out {
        return Err(CliError::usage(
            "`--onnx` and `--out` name the same file; refusing to overwrite the source model",
        ));
    }
    if args.out.exists() {
        if let (Ok(source), Ok(destination)) = (
            std::fs::canonicalize(&args.onnx),
            std::fs::canonicalize(&args.out),
        ) {
            if source == destination {
                return Err(CliError::usage(
                    "`--onnx` and `--out` resolve to the same file; refusing to overwrite the \
                     source model",
                ));
            }
        }
        if !args.force {
            return Err(
                CliError::model(format!("`{}` already exists", args.out.display()))
                    .with_next("pass `--force` to replace it, or choose another `--out` path"),
            );
        }
    }

    let source_bytes = std::fs::read(&args.onnx).map_err(|error| {
        CliError::model(format!(
            "cannot read ONNX model `{}`: {error}",
            args.onnx.display()
        ))
        .with_next(
            "check the path; the pinned tiny-GPT-2 model is fetched by \
             `python3 tools/parity/fetch_models.py --suite tiny-gpt2`",
        )
    })?;
    let source_sha256 = sha256_hex(&source_bytes);

    let import = import_onnx_model(&source_bytes).map_err(|error| {
        CliError::model(format!(
            "`{}` is not a model this importer accepts:\n{error}",
            args.onnx.display()
        ))
        .with_next(
            "use the pinned ONNX export described in `docs/quickstart-fn3.md`; the importer is \
             strict by design and will not approximate an unknown node",
        )
    })?;

    let envelope = {
        let backend = CpuBackend::new();
        Planner::new(&import, &backend)
            .envelope(1, args.max_seq, args.max_past, args.max_total)
            .map_err(|error| {
                CliError::model(format!("the requested envelope is not valid: {error}")).with_next(
                    "keep 1 <= max-seq <= max-total and max-past < max-total; the defaults are \
                     seq 128, past 1023, total 1024",
                )
            })?
    };
    let max_positions = match import.geometry().positions {
        PositionEmbedding::Learned { max_positions, .. } => u64::from(max_positions),
    };
    if args.max_total > max_positions {
        return Err(CliError::model(format!(
            "the requested `--max-total {}` exceeds the model's learned position table ({max_positions} rows)",
            args.max_total
        ))
        .with_next(format!(
            "pass `--max-total {max_positions}` or less; a prompt plus its generated tokens \
             cannot address a position the model does not have"
        )));
    }

    let artifact = PreparedArtifact::from_import(&import, envelope).map_err(|error| {
        CliError::model(format!("the artifact could not be built: {error}"))
            .with_next("this is a writer/import disagreement; report it with the ONNX checksum")
    })?;
    let artifact_bytes = artifact
        .to_bytes()
        .map_err(|error| CliError::model(format!("the artifact could not be encoded: {error}")))?;
    let artifact_sha256 = sha256_hex(&artifact_bytes);
    atomic_write(&args.out, &artifact_bytes)?;

    // Read back through the strict reader: framing, CRC and every section.
    let loaded = PreparedArtifact::read_file(&args.out).map_err(|error| {
        CliError::model(format!(
            "`{}` was written but cannot be read back: {error}",
            args.out.display()
        ))
        .with_next("re-run prepare; a partial or corrupt write is never reported as success")
    })?;
    if loaded != artifact {
        return Err(CliError::model(format!(
            "`{}` was written but reads back differently",
            args.out.display()
        ))
        .with_next("re-run prepare; the filesystem or a concurrent writer changed the file"));
    }
    let header = crate::fn3::model::format::PreparedHeader::parse(&artifact_bytes)
        .map_err(|error| CliError::model(format!("the written header is invalid: {error}")))?;

    print_summary(
        args,
        &import,
        envelope,
        &source_bytes,
        &source_sha256,
        &artifact_bytes,
        &artifact_sha256,
        header.payload_crc32,
        out,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn print_summary(
    args: &PrepareArgs,
    import: &ModelImport,
    envelope: crate::fn3::ShapeEnvelope,
    source_bytes: &[u8],
    source_sha256: &str,
    artifact_bytes: &[u8],
    artifact_sha256: &str,
    payload_crc32: u32,
    out: &mut dyn Write,
) {
    let geometry = import.geometry();
    let summary = import.summary();
    let epsilon = geometry
        .layer_norm_epsilon
        .map(|value| format!("{value:e}"))
        .unwrap_or_else(|| "not detected".to_string());
    let positions = match geometry.positions {
        PositionEmbedding::Learned { max_positions, .. } => {
            format!("learned ({max_positions} rows)")
        }
    };

    let _ = writeln!(out, "fn3-infer prepare");
    let _ = writeln!(out, "  source: {}", args.onnx.display());
    let _ = writeln!(out, "    bytes:  {}", thousands(source_bytes.len() as u64));
    let _ = writeln!(out, "    sha256: {source_sha256}");
    let _ = writeln!(
        out,
        "  imported: {} fn3 ops from {} ONNX nodes, {} values, {} weights ({} initializers, {} bytes)",
        summary.fn3_nodes,
        summary.onnx_nodes,
        summary.values,
        import.weights().len(),
        summary.initializers,
        thousands(summary.weight_bytes)
    );
    let _ = writeln!(
        out,
        "  geometry: layers={} hidden={} mlp={} heads={}/{} x {} vocab={} positions={} epsilon={}",
        geometry.layers,
        geometry.hidden_size,
        geometry.mlp_hidden_size,
        geometry.query_heads,
        geometry.kv_heads,
        geometry.head_dim,
        thousands(u64::from(geometry.vocab_size)),
        positions,
        epsilon
    );
    let _ = writeln!(
        out,
        "  envelope: batch {}..={}, seq {}..={}, past {}..={}, total {}..={}",
        envelope.get(Symbol::Batch).min(),
        envelope.get(Symbol::Batch).max(),
        envelope.get(Symbol::Seq).min(),
        envelope.get(Symbol::Seq).max(),
        envelope.get(Symbol::Past).min(),
        envelope.get(Symbol::Past).max(),
        envelope.get(Symbol::Total).min(),
        envelope.get(Symbol::Total).max(),
    );
    let _ = writeln!(out, "  wrote: {}", args.out.display());
    let _ = writeln!(
        out,
        "    bytes:  {}",
        thousands(artifact_bytes.len() as u64)
    );
    let _ = writeln!(out, "    sha256: {artifact_sha256}");
    let _ = writeln!(out, "    payload crc32: {payload_crc32:#010x}");
    let _ = writeln!(
        out,
        "    sections: META, GEOM, ENV, VALS ({} values), NODE ({} ops), WGHT ({} weights, {} bytes)",
        summary.values,
        summary.fn3_nodes,
        import.weights().len(),
        thousands(summary.weight_bytes)
    );
    let _ = writeln!(
        out,
        "  verified: read back through PreparedArtifact::read_file (framing, checksum and sections)"
    );
}

/// Write `bytes` to `path` through a same-directory temporary and a rename.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    let directory = parent.unwrap_or_else(|| Path::new("."));
    if !directory.is_dir() {
        return Err(CliError::model(format!(
            "cannot write `{}`: directory `{}` does not exist",
            path.display(),
            directory.display()
        ))
        .with_next("create the directory first, or pass an `--out` path that exists"));
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.fnn3");
    let temporary = directory.join(format!(".{name}.tmp-{}", std::process::id()));
    let _ = std::fs::remove_file(&temporary);

    let result = (|| -> std::io::Result<()> {
        use std::io::Write as _;
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(
            CliError::model(format!("cannot write `{}`: {error}", path.display()))
                .with_next("check free space and write permission on the destination directory"),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_destination_directory_is_actionable() {
        let path = std::env::temp_dir().join("fastnn-fn3-cli-missing-dir/model.fnn3");
        let error = atomic_write(&path, b"x").expect_err("no directory");
        assert_eq!(error.code(), super::super::EXIT_MODEL);
        assert!(error.message().contains("does not exist"), "{error:?}");
        assert!(error.next_step().is_some());
    }

    #[test]
    fn atomic_write_replaces_and_leaves_no_temporary() {
        let directory =
            std::env::temp_dir().join(format!("fastnn-fn3-cli-atomic-{}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("temp dir");
        let path = directory.join("model.fnn3");
        atomic_write(&path, b"first").expect("first write");
        atomic_write(&path, b"second").expect("replace");
        assert_eq!(std::fs::read(&path).expect("read"), b"second");
        let leftovers = std::fs::read_dir(&directory)
            .expect("list")
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp-"))
            .count();
        assert_eq!(leftovers, 0);
        let _ = std::fs::remove_dir_all(&directory);
    }
}
