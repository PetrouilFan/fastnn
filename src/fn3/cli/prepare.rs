//! `prepare`: strict ONNX import -> versioned artifact, atomically written.
//!
//! The write is not trusted until it has been read back through the strict
//! reader: a crash or a bad filesystem that leaves a partial file cannot be
//! reported as success. The source ONNX is never modified and no network access
//! happens here.

use std::io::Write;
use std::path::Path;

use crate::inference::digest::sha256_hex;
use crate::inference::import::import_onnx_graph;
use crate::inference::model::format::{
    resolve_targets, Artifact, Precision, PreparedArtifact, HEADER_BYTES,
};

use super::args::PrepareArgs;
use super::{format_error, import_error, thousands, CliError};

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
        .with_next("check the path and its permissions")
    })?;
    let source_sha256 = sha256_hex(&source_bytes);
    let mut source_digest = [0u8; 32];
    for (index, chunk) in source_sha256.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(chunk).unwrap_or("00");
        source_digest[index] = u8::from_str_radix(text, 16).unwrap_or(0);
    }

    let imported = import_onnx_graph(&source_bytes).map_err(import_error)?;
    let precision = Precision::parse(&args.precision)
        .ok_or_else(|| CliError::usage(format!("`{}` is not a precision", args.precision)))?;
    let targets = resolve_targets(&args.target).map_err(format_error)?;
    let artifact = PreparedArtifact::from_imported(&imported, &targets, precision, source_digest)
        .map_err(format_error)?;
    let artifact_bytes = artifact.to_bytes().map_err(format_error)?;
    let artifact_sha256 = sha256_hex(&artifact_bytes);
    atomic_write(&args.out, &artifact_bytes)?;

    // Read the file back through the strict reader: framing, every section
    // hash, the graph, the weight table and every plan. `Artifact::open` is the
    // deployment reader, so a success here is a success for `load` too.
    let loaded = Artifact::open(&args.out).map_err(format_error)?;
    if loaded.graph().node_count() != artifact.graph().node_count()
        || loaded.graph().value_count() != artifact.graph().value_count()
        || loaded.plans().len() != artifact.plans().len()
        || loaded.weights().len() != artifact.weights().len()
    {
        return Err(CliError::model(format!(
            "`{}` was written but reads back differently",
            args.out.display()
        ))
        .with_next("re-run prepare; the filesystem or a concurrent writer changed the file"));
    }
    if loaded.header().file_bytes != artifact_bytes.len() as u64 {
        return Err(CliError::model(format!(
            "`{}` was written but its length changed",
            args.out.display()
        ))
        .with_next("re-run prepare; a partial write is never reported as success"));
    }

    let _ = writeln!(out, "fastnn-infer prepare");
    let _ = writeln!(out, "  source: {}", args.onnx.display());
    let _ = writeln!(out, "    bytes:  {}", thousands(source_bytes.len() as u64));
    let _ = writeln!(out, "    sha256: {source_sha256}");
    let summary = imported.summary();
    let _ = writeln!(
        out,
        "  imported: {} typed nodes from {} ONNX nodes, {} values, {} weights ({} bytes)",
        loaded.graph().node_count(),
        summary.onnx_nodes,
        loaded.graph().value_count(),
        loaded.weights().len(),
        thousands(loaded.weight_bytes())
    );
    let _ = writeln!(
        out,
        "  target:   {} ({} profiles, precision {})",
        args.target,
        loaded.plans().len(),
        precision
    );
    for plan in loaded.plan_summaries() {
        let _ = writeln!(
            out,
            "    {:<14} segments={} fusions={} reorders={} ({}) arena={} B peak={} B engines scalar={} onednn={}",
            plan.target,
            plan.segments,
            plan.fusions,
            plan.reorders,
            thousands(plan.reorder_bytes),
            thousands(plan.arena_bytes),
            thousands(plan.peak_bytes),
            plan.engines.scalar,
            plan.engines.one_dnn
        );
    }
    let _ = writeln!(out, "  wrote: {}", args.out.display());
    let _ = writeln!(
        out,
        "    bytes:  {}",
        thousands(artifact_bytes.len() as u64)
    );
    let _ = writeln!(out, "    sha256: {artifact_sha256}");
    let _ = writeln!(
        out,
        "    header: magic FNN3ART2, version {}, {} B, {} sections, weights at byte {} ({} B)",
        loaded.header().version,
        HEADER_BYTES,
        loaded.header().section_count,
        thousands(loaded.header().weight_payload_offset),
        thousands(loaded.header().weight_payload_bytes)
    );
    let _ = writeln!(
        out,
        "  verified: read back through Artifact::open (header, SHA-256 per section, graph, \
         weight table and every plan)"
    );
    Ok(())
}

/// Write `bytes` to `path` through a same-directory temporary and a rename.
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
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
