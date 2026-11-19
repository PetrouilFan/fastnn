//! `run` and `generate`: token-ID inference on one explicit device.
//!
//! Both commands prepare a [`PreparedModel`] from a `.fnn3` artifact alone: the
//! file carries the graph, the weights, the geometry and the declared envelope,
//! so a deployment never needs the ONNX it was prepared from. Passing `--onnx`
//! (or `--verify-source`) additionally proves the artifact still matches that
//! source before anything runs; it is a verification, not an input.
//!
//! The commands open exactly the requested backend, run `prefill`, and then
//! either feed explicit token IDs (`run --decode`) or pick the argmax greedily
//! (`generate`). Decoding is deterministic: greedy argmax with ties broken by
//! the lowest token ID, and no sampling path at all.
//!
//! The bounds checks here are deliberate duplicates of the session's own: the
//! CLI names the flag that fixes the problem before any step starts, instead of
//! discovering it halfway through a prompt.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::inference::import::{import_onnx_model, ModelImport};
use crate::inference::model::format::{
    FormatError, FormatErrorKind, PreparedArtifact, PreparedHeader,
};
use crate::inference::model::{
    ModelWeights, Planner, PreparedModel, Session, SessionOptions, WEIGHT_INT8_BACKEND_REASON,
};
use crate::inference::{
    DType, DeviceId, Fn3Error, LiveDims, ShapeEnvelope, Symbol, Tensor, TensorMeta,
};

use super::args::{GenerateArgs, InferArgs, RunArgs};
use super::devices::OpenedBackend;
use super::digest::sha256_hex;
use super::{thousands, CliError};

/// Where the graph a plan is built from came from.
enum Source {
    /// The artifact alone. This is the deployment path.
    ArtifactOnly,
    /// An ONNX that was verified against the artifact before any work.
    Verified { path: PathBuf, sha256: String },
}

/// Everything a run needs, owned in one place. The session borrows `model`,
/// `import` and the backend inside `backend`, so all three outlive it.
struct Setup {
    artifact_path: PathBuf,
    artifact_bytes: u64,
    artifact_sha256: String,
    artifact_crc32: u32,
    source: Source,
    import: ModelImport,
    backend: OpenedBackend,
    model: PreparedModel,
    envelope: ShapeEnvelope,
    capacity: u64,
    prompt: Vec<u32>,
}

/// Run `run`.
pub fn execute_run(args: &RunArgs, out: &mut dyn Write) -> Result<(), CliError> {
    let setup = build_setup(&args.infer)?;
    let decode = match args.decode.as_deref() {
        Some(spec) => parse_tokens(spec, setup.import.geometry().vocab_size)?,
        None => Vec::new(),
    };
    check_sequence(&setup, decode.len() as u64, "decode tokens")?;
    print_header(&setup, out);

    let mut session = open_session(&setup, &args.infer)?;
    print_counters(&setup, &session, args.infer.steps, out);

    let mut logits = prefill(&setup, &mut session, &args.infer, out)?;
    let mut summary = summarize(&logits, setup.import.geometry().vocab_size)?;
    print_logits(&summary, &logits, args.infer.top_k, out);

    for (index, &token) in decode.iter().enumerate() {
        let step = index + 1;
        logits = decode_one(&mut session, token, step, &args.infer, out)?;
        summary = summarize(&logits, setup.import.geometry().vocab_size)?;
        print_logits(&summary, &logits, args.infer.top_k, out);
    }

    let _ = writeln!(
        out,
        "  predicted next token: {} (logit {:.4})",
        summary.argmax, summary.argmax_value
    );
    Ok(())
}

/// Run `generate`.
pub fn execute_generate(args: &GenerateArgs, out: &mut dyn Write) -> Result<(), CliError> {
    let setup = build_setup(&args.infer)?;
    check_sequence(&setup, args.max_new_tokens, "generated tokens")?;
    print_header(&setup, out);

    let mut session = open_session(&setup, &args.infer)?;
    print_counters(&setup, &session, args.infer.steps, out);

    let mut logits = prefill(&setup, &mut session, &args.infer, out)?;
    let mut summary = summarize(&logits, setup.import.geometry().vocab_size)?;
    print_logits(&summary, &logits, args.infer.top_k, out);

    let mut generated = Vec::with_capacity(args.max_new_tokens as usize);
    for step in 1..=args.max_new_tokens {
        let token = summary.argmax as u32;
        generated.push(token);
        logits = decode_one(&mut session, token, step as usize, &args.infer, out)?;
        summary = summarize(&logits, setup.import.geometry().vocab_size)?;
    }

    let _ = writeln!(
        out,
        "  generated token IDs: [{}]",
        format_token_list(&generated)
    );
    let _ = writeln!(
        out,
        "  predicted next token: {} (logit {:.4})",
        summary.argmax, summary.argmax_value
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

fn build_setup(args: &InferArgs) -> Result<Setup, CliError> {
    // Resolve the device first: an unavailable explicit request must fail
    // before a 28 MB artifact is read, and it must never become a CPU run.
    let backend = OpenedBackend::open(&args.device)?;

    let (artifact, artifact_sha256, artifact_crc32, artifact_bytes) = read_artifact(&args.model)?;
    let envelope = artifact.envelope();

    // The artifact is the input. `--onnx` only adds a source check: when it is
    // given, the import that runs is the verified one, exactly as before; when
    // it is absent, the artifact's own graph and weights are what run.
    let (import, source) = match args.onnx.as_deref() {
        Some(onnx_path) => {
            let (import, sha256) = read_import(onnx_path)?;
            verify_pair(&artifact, &import, &args.model, onnx_path)?;
            (
                import,
                Source::Verified {
                    path: onnx_path.to_path_buf(),
                    sha256,
                },
            )
        }
        None => (artifact.into_model_import(), Source::ArtifactOnly),
    };

    let capacity = args
        .capacity
        .unwrap_or_else(|| envelope.get(Symbol::Total).max());
    if capacity > envelope.get(Symbol::Total).max() {
        return Err(CliError::inference(format!(
            "KV capacity {capacity} exceeds the artifact's total envelope maximum {}",
            envelope.get(Symbol::Total).max()
        ))
        .with_next(
            "pass `--capacity` at or below the envelope maximum, or re-prepare with a larger \
             `--max-total` (the model's learned position table is the hard cap)",
        ));
    }

    let prompt = parse_tokens(&args.tokens, import.geometry().vocab_size)?;
    if prompt.is_empty() {
        return Err(CliError::usage(
            "`--tokens` needs at least one token ID; an empty prompt cannot be prefilled",
        ));
    }
    let prompt_len = prompt.len() as u64;
    if prompt_len > envelope.get(Symbol::Seq).max() {
        return Err(CliError::inference(format!(
            "the prompt has {prompt_len} token IDs, above the artifact's seq envelope maximum {}",
            envelope.get(Symbol::Seq).max()
        ))
        .with_next(format!(
            "re-prepare with `--max-seq {prompt_len}` (or more), or shorten the prompt"
        )));
    }
    if prompt_len > envelope.get(Symbol::Total).max() {
        return Err(CliError::inference(format!(
            "the prompt has {prompt_len} token IDs, above the artifact's total envelope maximum {}",
            envelope.get(Symbol::Total).max()
        ))
        .with_next("re-prepare with a larger `--max-total`, or shorten the prompt"));
    }
    if prompt_len > capacity {
        return Err(CliError::inference(format!(
            "the prompt has {prompt_len} token IDs but the KV capacity is {capacity} tokens"
        ))
        .with_next(format!(
            "pass `--capacity {prompt_len}` (or more, up to the envelope maximum {})",
            envelope.get(Symbol::Total).max()
        )));
    }

    let geometry = import.geometry();
    let heads = u64::from(geometry.query_heads);
    let kv_heads = u64::from(geometry.kv_heads);
    let head_dim = u64::from(geometry.head_dim);
    let fixed_seq = envelope.get(Symbol::Seq).max().clamp(1, 4);
    let fixed_dims =
        LiveDims::new(1, fixed_seq, 0, heads, kv_heads, head_dim).map_err(|error| {
            CliError::model(format!("the model's head geometry is not usable: {error}"))
        })?;
    let weights = import.to_model_weights().map_err(|error| {
        failure(
            "copying the weights into host tensors",
            error,
            backend.device,
        )
    })?;
    let weights = retag_weights(&weights, backend.device)?;
    // This internal plan only preflights the session-backed decoder; the
    // session binds its own tokens and specializes again at each live shape.
    let plan = Planner::for_session(&import, backend.backend.as_ref())
        .plan(&envelope, fixed_dims)
        .map_err(|error| plan_failure(&backend, error))?;
    let kv_config = geometry.kv_cache_config(capacity, 1);
    let model = PreparedModel::new(
        plan.execution().clone(),
        weights,
        backend.device,
        envelope,
        Some(kv_config),
    )
    .map_err(|error| failure("PreparedModel::new", error, backend.device))?;

    Ok(Setup {
        artifact_path: args.model.clone(),
        artifact_bytes,
        artifact_sha256,
        artifact_crc32,
        source,
        import,
        backend,
        model,
        envelope,
        capacity,
        prompt,
    })
}

fn open_session<'a>(setup: &'a Setup, args: &InferArgs) -> Result<Session<'a>, CliError> {
    Session::with_runtime_options(
        &setup.model,
        &setup.import,
        setup.backend.backend.as_ref(),
        SessionOptions {
            weight_int8: args.weight_int8,
            ..SessionOptions::default()
        },
    )
    .map_err(|error| {
        let hint = weight_int8_hint(args.weight_int8, setup.backend.device, &error);
        let cli = failure("Session::with_runtime", error, setup.backend.device);
        match hint {
            Some(hint) => cli.with_next(hint),
            None => cli,
        }
    })
}

/// Advice for the one session-creation failure `--weight-int8` owns.
///
/// Only the typed backend rejection earns the flag-specific hint: any other
/// session error keeps its own next step, so the message never blames the flag
/// for an unrelated failure.
fn weight_int8_hint(weight_int8: bool, device: DeviceId, error: &Fn3Error) -> Option<&'static str> {
    if !weight_int8 || device.is_cpu() {
        return None;
    }
    match error {
        Fn3Error::UnsupportedOp { reason, .. }
            if reason.starts_with(WEIGHT_INT8_BACKEND_REASON) =>
        {
            Some(
                "`--weight-int8` needs a resident session: drop the flag, or unset \
                 FN3_VK_STAGED; a staged Vulkan session is rejected rather than run on the CPU",
            )
        }
        _ => None,
    }
}

fn read_artifact(path: &Path) -> Result<(PreparedArtifact, String, u32, u64), CliError> {
    let bytes = std::fs::read(path).map_err(|error| {
        CliError::model(format!(
            "cannot read artifact `{}`: {error}",
            path.display()
        ))
        .with_next(
            "create it with `fastnn-infer prepare --onnx <model.onnx> --out <model.fnn3>`, or check \
             the path",
        )
    })?;
    let byte_len = bytes.len() as u64;
    let header = PreparedHeader::parse(&bytes).map_err(|error| format_failure(path, &error))?;
    let sha256 = sha256_hex(&bytes);
    let artifact = PreparedArtifact::from_bytes(&bytes).map_err(|error| {
        let cli = format_failure(path, &error);
        if error.is_corruption() {
            cli.with_next(
                "the payload does not match its checksum; re-run `fastnn-infer prepare` from the \
                 source ONNX",
            )
        } else {
            cli
        }
    })?;
    Ok((artifact, sha256, header.payload_crc32, byte_len))
}

fn read_import(path: &Path) -> Result<(ModelImport, String), CliError> {
    let bytes = std::fs::read(path).map_err(|error| {
        CliError::model(format!(
            "cannot read ONNX model `{}`: {error}",
            path.display()
        ))
        .with_next(
            "`--onnx` is the optional source check; drop it to run the artifact alone, or pass the \
             same ONNX file the artifact was prepared from",
        )
    })?;
    let sha256 = sha256_hex(&bytes);
    let import = import_onnx_model(&bytes).map_err(|error| {
        CliError::model(format!(
            "`{}` is not an ONNX model this importer accepts:\n{error}",
            path.display()
        ))
        .with_next("pass the pinned ONNX export this artifact was prepared from")
    })?;
    Ok((import, sha256))
}

fn verify_pair(
    artifact: &PreparedArtifact,
    import: &ModelImport,
    artifact_path: &Path,
    onnx_path: &Path,
) -> Result<(), CliError> {
    let expected = crate::inference::model::format::ArtifactMetadata::from_import(import);
    if &expected != artifact.metadata() {
        return Err(mismatch(
            artifact_path,
            onnx_path,
            "provenance (IR version, opset, producer or graph name)",
        ));
    }
    if import.geometry() != artifact.geometry() {
        return Err(mismatch(artifact_path, onnx_path, "architecture geometry"));
    }
    if import.weights().len() != artifact.weights().len() {
        return Err(mismatch(
            artifact_path,
            onnx_path,
            &format!(
                "weight count ({} in the ONNX, {} in the artifact)",
                import.weights().len(),
                artifact.weights().len()
            ),
        ));
    }
    for (artifact_weight, import_weight) in artifact.weights().iter().zip(import.weights().iter()) {
        if artifact_weight.name() != import_weight.name()
            || artifact_weight.dtype() != import_weight.dtype()
            || artifact_weight.shape() != import_weight.shape()
            || artifact_weight.bytes() != import_weight.bytes()
        {
            return Err(mismatch(
                artifact_path,
                onnx_path,
                &format!("weight `{}`", artifact_weight.name()),
            ));
        }
    }
    Ok(())
}

fn mismatch(artifact: &Path, onnx: &Path, what: &str) -> CliError {
    CliError::model(format!(
        "`{}` and `{}` do not describe the same model: {what} differs",
        artifact.display(),
        onnx.display()
    ))
    .with_next(
        "pass the exact ONNX the artifact was prepared from, or re-run `fastnn-infer prepare` with \
         this ONNX; without `--onnx` the artifact runs on its own",
    )
}

fn format_failure(path: &Path, error: &FormatError) -> CliError {
    let next = match error.kind() {
        FormatErrorKind::UnsupportedVersion { found, supported } => format!(
            "this file uses format version {found}; this build reads version {supported} only. \
             Use a matching `fastnn-infer`, or re-prepare the ONNX with this build"
        ),
        FormatErrorKind::BadMagic { .. } => "the first eight bytes must be `FNN3PREP`; a 2.x \
             `.fnnc` file is a different format and is never read by FastNN 3.0"
            .to_string(),
        FormatErrorKind::UnsupportedDtypeEncoding { found, supported } => format!(
            "this file uses dtype encoding {found}; this build reads encoding {supported} only"
        ),
        FormatErrorKind::Io { .. } => {
            "check the path and file permissions, then create the artifact with `fastnn-infer prepare`"
                .to_string()
        }
        _ => "the file is truncated, corrupt or inconsistent; re-run `fastnn-infer prepare` from \
              the source ONNX"
            .to_string(),
    };
    CliError::model(format!(
        "`{}` is not a usable prepared model: {error}",
        path.display()
    ))
    .with_next(next)
}

fn retag_weights(weights: &ModelWeights, device: DeviceId) -> Result<ModelWeights, CliError> {
    let mut named = Vec::with_capacity(weights.len());
    for index in 0..weights.len() {
        let tensor = weights.get(index).ok_or_else(|| {
            CliError::model(format!(
                "weight index {index} disappeared from the imported weights"
            ))
        })?;
        let name = weights.name(index).ok_or_else(|| {
            CliError::model(format!(
                "weight index {index} has no name in the imported weights"
            ))
        })?;
        let meta = TensorMeta::contiguous(tensor.shape().clone(), tensor.dtype(), device)
            .map_err(|error| failure("tagging a weight for the requested device", error, device))?;
        let retagged = Tensor::from_storage(tensor.storage().clone(), meta)
            .map_err(|error| failure("tagging a weight for the requested device", error, device))?;
        named.push((name.to_string(), retagged));
    }
    ModelWeights::with_names(named)
        .map_err(|error| failure("ModelWeights::with_names", error, device))
}

fn plan_failure(backend: &OpenedBackend, error: Fn3Error) -> CliError {
    if error_is_unsupported(&error) && !backend.device.is_cpu() {
        return CliError::inference(format!(
            "the {} backend cannot build the full decoder plan on {}: {error}",
            backend.backend.name(),
            backend.device
        ))
        .with_next(
            "this plan contains an operation the requested backend does not implement for these              operand types; an explicit device never falls back to CPU. `fastnn-infer diagnose`              shows what the device opened, and `--device cpu` is a different, explicit choice",
        );
    }
    failure("building the initial decoder plan", error, backend.device)
}

fn failure(context: &str, error: Fn3Error, device: DeviceId) -> CliError {
    let code = match &error {
        Fn3Error::DeviceUnavailable(_) => super::EXIT_DEVICE,
        Fn3Error::InvalidFormat { .. } | Fn3Error::UnknownDTypeCode(_) => super::EXIT_MODEL,
        _ => super::EXIT_INFERENCE,
    };
    let mut cli = CliError::new(code, format!("{context}: {error}"));
    if matches!(
        &error,
        Fn3Error::UnsupportedOp { .. } | Fn3Error::NotImplemented { .. }
    ) {
        let next = if device.is_cpu() {
            "the reference backend is missing this operation in this build; report it with the \
             artifact SHA-256 printed above"
        } else {
            "this backend cannot run the full decoder graph yet; run `--device cpu`, and use \
             `fastnn-infer diagnose` to see which devices actually opened"
        };
        cli = cli.with_next(next);
    } else if cli.next_step().is_none() {
        cli = cli.with_next(
            "re-run with `--steps` to see the failing step; if the device is involved, run \
             `fastnn-infer diagnose`",
        );
    }
    cli
}

fn error_is_unsupported(error: &Fn3Error) -> bool {
    matches!(
        error,
        Fn3Error::UnsupportedOp { .. } | Fn3Error::NotImplemented { .. }
    )
}

// ---------------------------------------------------------------------------
// Bounds and tokens
// ---------------------------------------------------------------------------

fn check_sequence(setup: &Setup, extra: u64, what: &str) -> Result<(), CliError> {
    let prompt = setup.prompt.len() as u64;
    let total = prompt.checked_add(extra).ok_or_else(|| {
        CliError::inference("the prompt plus requested tokens overflows a u64 token count")
    })?;
    let total_max = setup.envelope.get(Symbol::Total).max();
    if total > total_max {
        return Err(CliError::inference(format!(
            "the prompt ({prompt}) plus {extra} {what} needs {total} positions, above the \
             artifact's total envelope maximum {total_max}"
        ))
        .with_next(format!(
            "re-prepare with `--max-total {total}` (or more), or request fewer {what}"
        )));
    }
    if total > setup.capacity {
        return Err(CliError::inference(format!(
            "{total} positions exceed the KV capacity of {} tokens",
            setup.capacity
        ))
        .with_next(format!(
            "pass `--capacity {total}` (up to the envelope maximum {total_max}), or request \
             fewer {what}"
        )));
    }
    if total > 0 {
        let last_past = total - 1;
        let past_max = setup.envelope.get(Symbol::Past).max();
        if last_past > past_max {
            return Err(CliError::inference(format!(
                "the final decode would start at past={last_past}, above the artifact's past \
                 envelope maximum {past_max}"
            ))
            .with_next("re-prepare with a larger `--max-past`, or request fewer tokens"));
        }
    }
    Ok(())
}

fn parse_tokens(spec: &str, vocab: u32) -> Result<Vec<u32>, CliError> {
    let text = if let Some(path) = spec.strip_prefix('@') {
        std::fs::read_to_string(path).map_err(|error| {
            CliError::model(format!("cannot read token file `{path}`: {error}")).with_next(
                "write one token ID per line or a comma-separated list, and pass `--tokens @file`",
            )
        })?
    } else {
        spec.to_string()
    };

    let mut tokens = Vec::new();
    for (position, item) in text
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|item| !item.is_empty())
        .enumerate()
    {
        let token = item.parse::<u32>().map_err(|_| {
            CliError::usage(format!(
                "token {} (`{item}`) is not a whole number; token IDs are unsigned 32-bit integers",
                position + 1
            ))
        })?;
        if token >= vocab {
            return Err(CliError::inference(format!(
                "token ID {token} is outside the model vocabulary 0..={}",
                vocab.saturating_sub(1)
            ))
            .with_next(
                "use the reference tokenizer to obtain IDs below the model's vocabulary; this \
                 alpha does not approximate byte-level BPE, because different tokens produce \
                 different logits",
            ));
        }
        tokens.push(token);
    }
    Ok(tokens)
}

fn format_token_list(tokens: &[u32]) -> String {
    if tokens.len() <= 16 {
        return tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
    }
    let head = tokens[..8]
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let tail = tokens[tokens.len() - 8..]
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{head}, … ({} more) …, {tail}", tokens.len() - 16)
}

// ---------------------------------------------------------------------------
// Inference and output
// ---------------------------------------------------------------------------

fn prefill(
    setup: &Setup,
    session: &mut Session<'_>,
    args: &InferArgs,
    out: &mut dyn Write,
) -> Result<Tensor, CliError> {
    let hits_before = session.plan_hits();
    let misses_before = session.plan_misses();
    let start = Instant::now();
    let logits = session
        .prefill(&setup.prompt)
        .map_err(|error| failure("Session::prefill", error, setup.backend.device))?;
    let elapsed = start.elapsed();
    let _ = writeln!(
        out,
        "  prefill: seq={} past=0 cache={} plan={} plan-cache={}hit/{}miss wall={}",
        setup.prompt.len(),
        session.cache_len(),
        plan_label(session),
        session.plan_hits() - hits_before,
        session.plan_misses() - misses_before,
        milliseconds(elapsed)
    );
    if args.steps {
        let _ = writeln!(
            out,
            "    counters: slot_allocations={} slot_reuses={} plan_cache={} phase={}",
            session.slot_allocations(),
            session.slot_reuses(),
            session.plan_cache_len(),
            session.phase().name()
        );
    }
    print_quantization(session, args.steps, out);
    print_weight_int8_traffic(session, out);
    Ok(logits)
}

fn decode_one(
    session: &mut Session<'_>,
    token: u32,
    step: usize,
    args: &InferArgs,
    out: &mut dyn Write,
) -> Result<Tensor, CliError> {
    let hits_before = session.plan_hits();
    let misses_before = session.plan_misses();
    let past = session.cache_len();
    let start = Instant::now();
    let logits = session
        .decode(&[token])
        .map_err(|error| failure("Session::decode", error, session.model().device()))?;
    let elapsed = start.elapsed();
    if args.steps {
        let _ = writeln!(
            out,
            "  decode {step}: token={token} past={past} seq=1 cache={} plan={} plan-cache={}hit/{}miss \
             wall={} slots={}/{} phase={}",
            session.cache_len(),
            plan_label(session),
            session.plan_hits() - hits_before,
            session.plan_misses() - misses_before,
            milliseconds(elapsed),
            session.slot_allocations(),
            session.slot_reuses(),
            session.phase().name()
        );
    } else {
        let _ = writeln!(
            out,
            "  decode {step}: token={token} cache={}",
            session.cache_len()
        );
    }
    print_quantization(session, args.steps, out);
    print_weight_int8_traffic(session, out);
    Ok(logits)
}

fn print_quantization(session: &Session<'_>, details: bool, out: &mut dyn Write) {
    if let Some(report) = session.weight_int8_report() {
        let weights = session.model().weights();
        let model_bytes: u64 = (0..weights.len())
            .filter_map(|index| weights.get(index))
            .map(|tensor| tensor.storage().len() as u64)
            .sum();
        let packed = report
            .routes
            .iter()
            .filter(|route| route.fallback.is_none())
            .count();
        // On the CPU nothing is uploaded: the dense figure is FP32 weights the
        // session kept, not bytes moved to a device.
        let dense_label = if session.model().device().is_cpu() {
            "fp32-retained"
        } else {
            "dense-weight"
        };
        let _ = writeln!(out, "    weight-int8: packed-matmuls={packed}/{} fp32-matmuls={} packed-payload={}B packed-source={}B {dense_label}={}B shared-weight={}B model-weight={model_bytes}B route/pack/upload={}",
            report.routes.len(), report.routes.len() - packed, report.packed_payload_bytes,
            report.packed_source_bytes, report.dense_weight_bytes, report.shared_weight_bytes,
            milliseconds(report.prepare));
        if details {
            for route in &report.routes {
                let _ = writeln!(
                    out,
                    "      matmul[{}] {}: {}",
                    route.instruction,
                    route.rhs,
                    route
                        .fallback
                        .as_ref()
                        .map_or("packed", |fallback| fallback.code())
                );
            }
        }
    }
}

/// The residency ledger's view of an INT8-weight step, on a real device.
///
/// Packed weights are immutable prepare traffic, not per-step host staging; the
/// staged figure here is the non-weight traffic a resident step still moves.
#[cfg(feature = "vulkan")]
fn print_weight_int8_traffic(session: &Session<'_>, out: &mut dyn Write) {
    let Some(vulkan) = session.backend().and_then(|backend| backend.as_vulkan()) else {
        return;
    };
    let stats = vulkan.residency_stats();
    let staged: u64 = stats.staged_upload_bytes.iter().sum();
    let _ = writeln!(
        out,
        "    weight-int8-traffic: packed-prepare={}B staged-host-bytes={staged}B \
         late-persistent={}B",
        stats.packed_prepare_bytes, stats.late_persistent_bytes
    );
}

#[cfg(not(feature = "vulkan"))]
fn print_weight_int8_traffic(_session: &Session<'_>, _out: &mut dyn Write) {}

fn plan_label(session: &Session<'_>) -> String {
    session
        .last_plan()
        .map(|plan| plan.to_hex())
        .unwrap_or_else(|| "-".to_string())
}

fn milliseconds(elapsed: std::time::Duration) -> String {
    let millis = elapsed.as_secs_f64() * 1_000.0;
    if millis >= 1_000.0 {
        format!("{:.2}s", millis / 1_000.0)
    } else {
        format!("{millis:.2}ms")
    }
}

struct Summary {
    shape: String,
    len: usize,
    min: f32,
    max: f32,
    mean: f64,
    argmax: usize,
    argmax_value: f32,
}

fn summarize(logits: &Tensor, vocab: u32) -> Result<Summary, CliError> {
    if logits.dtype() != DType::F32 {
        return Err(CliError::inference(format!(
            "the session returned {} logits, but the decoder contract is f32",
            logits.dtype()
        ))
        .with_next("this is a backend contract violation; report it with the artifact SHA-256"));
    }
    let dims = logits.shape().dims();
    let expected = [1, 1, u64::from(vocab)];
    if dims != expected {
        return Err(CliError::inference(format!(
            "the session returned logits of shape {}, expected [1, 1, {vocab}]",
            logits.shape()
        ))
        .with_next(
            "this is a session/backend contract violation; report it with the artifact SHA-256",
        ));
    }
    let values = logits.as_f32_slice().map_err(|error| {
        CliError::inference(format!("the logits tensor is not readable as f32: {error}"))
    })?;
    if values.is_empty() {
        return Err(CliError::inference(
            "the session returned an empty logits tensor",
        ));
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut argmax = 0usize;
    let mut argmax_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(CliError::inference(format!(
                "logits contain a non-finite value ({value}) at vocabulary index {index}"
            ))
            .with_next(
                "refusing to pick a token from an invalid distribution; check the model/artifact \
                 pair and the backend",
            ));
        }
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
        // `>` keeps the lowest token ID on a tie: deterministic argmax.
        if value > argmax_value {
            argmax = index;
            argmax_value = value;
        }
        sum += f64::from(value);
    }
    Ok(Summary {
        shape: logits.shape().to_string(),
        len: values.len(),
        min,
        max,
        mean: sum / values.len() as f64,
        argmax,
        argmax_value,
    })
}

fn print_logits(summary: &Summary, logits: &Tensor, top_k: usize, out: &mut dyn Write) {
    let _ = writeln!(
        out,
        "  logits: shape={} f32 finite={} min={:.4} max={:.4} mean={:.6}",
        summary.shape,
        thousands(summary.len as u64),
        summary.min,
        summary.max,
        summary.mean
    );
    let Ok(values) = logits.as_f32_slice() else {
        return;
    };
    let mut ranked: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let shown = top_k.min(ranked.len());
    let top = ranked[..shown]
        .iter()
        .map(|(index, value)| format!("{index} ({value:.4})"))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(out, "  top-{shown}: {top}");
    let _ = writeln!(
        out,
        "  argmax: {} (logit {:.4})",
        summary.argmax, summary.argmax_value
    );
}

fn print_header(setup: &Setup, out: &mut dyn Write) {
    let geometry = setup.import.geometry();
    let envelope = setup.envelope;
    let _ = writeln!(out, "fastnn-infer inference");
    let _ = writeln!(out, "  artifact: {}", setup.artifact_path.display());
    let _ = writeln!(
        out,
        "    bytes:  {}  sha256: {}",
        thousands(setup.artifact_bytes),
        setup.artifact_sha256
    );
    let _ = writeln!(out, "    payload crc32: {:#010x}", setup.artifact_crc32);
    match &setup.source {
        Source::ArtifactOnly => {
            let _ = writeln!(
                out,
                "  source: none (artifact-only: the artifact carries the graph, {} weight \
                 payloads, the geometry and the envelope)",
                setup.import.weights().len()
            );
        }
        Source::Verified { path, sha256 } => {
            let _ = writeln!(
                out,
                "  source: {} (sha256 {}, verified against the artifact: metadata, geometry and \
                 {} weight payloads match)",
                path.display(),
                sha256,
                setup.import.weights().len()
            );
        }
    }
    let _ = writeln!(
        out,
        "  device: requested {}, resolved {}, backend {} ({})",
        setup.backend.requested,
        setup.backend.device,
        setup.backend.backend.name(),
        setup.backend.detail
    );
    let _ = writeln!(
        out,
        "  envelope: batch {}..={} seq {}..={} past {}..={} total {}..={}",
        envelope.get(Symbol::Batch).min(),
        envelope.get(Symbol::Batch).max(),
        envelope.get(Symbol::Seq).min(),
        envelope.get(Symbol::Seq).max(),
        envelope.get(Symbol::Past).min(),
        envelope.get(Symbol::Past).max(),
        envelope.get(Symbol::Total).min(),
        envelope.get(Symbol::Total).max()
    );
    let layer_bytes = geometry
        .kv_cache_config(setup.capacity, 1)
        .buffer_bytes()
        .map(|bytes| bytes.saturating_mul(2))
        .unwrap_or(0);
    let _ = writeln!(
        out,
        "  kv: capacity={} tokens, {} layers x {} KV heads x {} head_dim = {} bytes",
        setup.capacity,
        geometry.layers,
        geometry.kv_heads,
        geometry.head_dim,
        thousands(layer_bytes.saturating_mul(u64::from(geometry.layers)))
    );
    let _ = writeln!(
        out,
        "  prompt: {} token IDs [{}]",
        setup.prompt.len(),
        format_token_list(&setup.prompt)
    );
    let _ = writeln!(
        out,
        "  input: token IDs (this release has no text tokenizer)"
    );
}

fn print_counters(setup: &Setup, session: &Session<'_>, steps: bool, out: &mut dyn Write) {
    let _ = writeln!(
        out,
        "  session: phase={} cache={} kv_bytes={}",
        session.phase().name(),
        session.cache_len(),
        thousands(session.kv_bytes())
    );
    let _ = writeln!(
        out,
        "  execution: {} on {}",
        setup.backend.backend.name(),
        setup.backend.device
    );
    if steps {
        let _ = writeln!(
            out,
            "  timing: wall-clock around each call; the Vulkan backend records per-dispatch \
             gpu_ns, but Session does not aggregate a device time in this build"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flag-specific hint belongs to the typed backend rejection only.
    #[test]
    fn the_int8_hint_is_reserved_for_the_typed_rejection() {
        let gpu = DeviceId::Vulkan(crate::inference::device::PhysicalDeviceId::new(0));
        let rejection = Fn3Error::UnsupportedOp {
            op: crate::inference::OpId::MatMul,
            backend: "vulkan",
            reason: format!("{WEIGHT_INT8_BACKEND_REASON} (the session is host-staged)"),
        };
        assert!(weight_int8_hint(true, gpu, &rejection).is_some());
        assert!(
            weight_int8_hint(false, gpu, &rejection).is_none(),
            "no flag, no flag-specific advice"
        );
        assert!(
            weight_int8_hint(true, DeviceId::Cpu, &rejection).is_none(),
            "the CPU never needs the staging hint"
        );

        // Some other session failure keeps its own next step.
        let unrelated = Fn3Error::UnsupportedOp {
            op: crate::inference::OpId::MatMul,
            backend: "vulkan",
            reason: "the device lost its queue".into(),
        };
        assert!(weight_int8_hint(true, gpu, &unrelated).is_none());
        let other = Fn3Error::ShapeMismatch {
            op: Some(crate::inference::OpId::MatMul),
            expected: "[1, 64]".into(),
            actual: "[1, 65]".into(),
        };
        assert!(weight_int8_hint(true, gpu, &other).is_none());
    }

    #[test]
    fn tokens_accept_commas_whitespace_and_files() {
        assert_eq!(
            parse_tokens("1, 2 3\n4", 10).expect("parse"),
            vec![1, 2, 3, 4]
        );
        assert_eq!(parse_tokens("0", 1).expect("parse"), vec![0]);
        let error = parse_tokens("1, x", 10).expect_err("not a number");
        assert_eq!(error.code(), super::super::EXIT_USAGE);
        assert!(error.message().contains("token 2"), "{error:?}");
    }

    #[test]
    fn out_of_vocabulary_tokens_are_actionable_and_not_silently_wrapped() {
        let error = parse_tokens("50257", 50_257).expect_err("out of range");
        assert_eq!(error.code(), super::super::EXIT_INFERENCE);
        assert!(error.message().contains("0..=50256"), "{error:?}");
        assert!(error.next_step().is_some());
    }

    #[test]
    fn token_lists_are_truncated_in_the_middle_for_long_prompts() {
        let tokens: Vec<u32> = (0..40).collect();
        let text = format_token_list(&tokens);
        assert!(text.contains("24 more"), "{text}");
        assert!(text.starts_with("0, 1, 2"), "{text}");
        assert!(text.ends_with("38, 39"), "{text}");
    }
}
