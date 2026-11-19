//! Command-line syntax for `fastnn-infer`.
//!
//! Hand-written rather than derived so every rejection names the offending flag
//! and the accepted forms, and so the same parser is callable from tests. The
//! parser accepts `--flag value` and `--flag=value`; it rejects duplicates for
//! options that carry a single value instead of letting the last one win.

use std::path::PathBuf;

use super::{CliError, DeviceSpec};

/// One parsed invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// ONNX -> prepared artifact.
    Prepare(PrepareArgs),
    /// Artifact header and optional section summary.
    Inspect(InspectArgs),
    /// Token-ID prefill plus optional explicit decode tokens.
    Run(RunArgs),
    /// Greedy token generation.
    Generate(GenerateArgs),
    /// Build/device/backend matrix.
    Diagnose(DiagnoseArgs),
    /// Print help.
    Help(HelpTopic),
    /// Print the tool version.
    Version,
}

/// Which help page to print.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpTopic {
    All,
    Prepare,
    Inspect,
    Run,
    Generate,
    Diagnose,
}

/// Arguments of `prepare`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareArgs {
    /// Source ONNX file.
    pub onnx: PathBuf,
    /// Destination artifact.
    pub out: PathBuf,
    /// Declared maximum new tokens in one call.
    pub max_seq: u64,
    /// Declared maximum already-cached tokens.
    pub max_past: u64,
    /// Declared maximum `past + seq`.
    pub max_total: u64,
    /// Replace an existing destination.
    pub force: bool,
}

/// Arguments of `inspect`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectArgs {
    /// Artifact to inspect.
    pub artifact: PathBuf,
    /// Decode META/GEOM/ENV/VALS/NODE/WGHT instead of stopping at the header.
    pub sections: bool,
}

/// Arguments shared by `run` and `generate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferArgs {
    /// Prepared artifact: graph, weights, geometry and envelope.
    pub model: PathBuf,
    /// Optional source ONNX, verified against the artifact when given.
    pub onnx: Option<PathBuf>,
    /// Prompt token IDs: comma/whitespace separated, or `@file`.
    pub tokens: String,
    /// Device request string.
    pub device: DeviceSpec,
    /// KV capacity in tokens; defaults to the envelope's `total.max()`.
    pub capacity: Option<u64>,
    /// Number of tokens to show in the logits table.
    pub top_k: usize,
    /// Per-step plan/cache diagnostics.
    pub steps: bool,
    /// Opt in to session-local INT8 projection weights.
    pub weight_int8: bool,
}

/// Arguments of `run`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunArgs {
    /// Shared inference arguments.
    pub infer: InferArgs,
    /// Optional token IDs to feed one at a time after the prompt.
    pub decode: Option<String>,
}

/// Arguments of `generate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateArgs {
    /// Shared inference arguments.
    pub infer: InferArgs,
    /// Tokens to generate greedily.
    pub max_new_tokens: u64,
}

/// Arguments of `diagnose`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnoseArgs {
    /// Optional artifact to validate while reporting devices.
    pub artifact: Option<PathBuf>,
    /// Optional device to resolve and open; absence reports every device.
    pub device: Option<DeviceSpec>,
}

/// Parse an argument vector (without the program name).
pub fn parse(args: &[String]) -> Result<Command, CliError> {
    let Some(first) = args.first() else {
        return Ok(Command::Help(HelpTopic::All));
    };
    match first.as_str() {
        "prepare" => parse_prepare(&args[1..]),
        "inspect" => parse_inspect(&args[1..]),
        "run" => parse_run(&args[1..]),
        "generate" => parse_generate(&args[1..]),
        "diagnose" => parse_diagnose(&args[1..]),
        "help" | "--help" | "-h" => parse_help(&args[1..]),
        "--version" | "-V" => Ok(Command::Version),
        other if other.starts_with('-') => Err(CliError::usage(format!("unknown flag `{other}`"))),
        other => Err(CliError::usage(format!("unknown command `{other}`"))),
    }
}

fn parse_help(args: &[String]) -> Result<Command, CliError> {
    let Some(topic) = args.first() else {
        return Ok(Command::Help(HelpTopic::All));
    };
    if args.len() > 1 {
        return Err(CliError::usage(format!(
            "`help` takes at most one topic, but {} were given",
            args.len()
        )));
    }
    let topic = match topic.as_str() {
        "all" => HelpTopic::All,
        "prepare" => HelpTopic::Prepare,
        "inspect" => HelpTopic::Inspect,
        "run" => HelpTopic::Run,
        "generate" => HelpTopic::Generate,
        "diagnose" => HelpTopic::Diagnose,
        other => {
            return Err(CliError::usage(format!(
                "unknown help topic `{other}`; topics: prepare, inspect, run, generate, diagnose"
            )))
        }
    };
    Ok(Command::Help(topic))
}

fn parse_prepare(args: &[String]) -> Result<Command, CliError> {
    let mut onnx: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut max_seq: Option<u64> = None;
    let mut max_past: Option<u64> = None;
    let mut max_total: Option<u64> = None;
    let mut force = false;

    let mut cursor = Cursor::new(args);
    while let Some((flag, inline)) = cursor.next_flag() {
        match flag {
            "--help" | "-h" => return Ok(Command::Help(HelpTopic::Prepare)),
            "--onnx" => set_path(&mut onnx, cursor.value(inline, flag)?, flag)?,
            "--out" => set_path(&mut out, cursor.value(inline, flag)?, flag)?,
            "--max-seq" => set_u64(&mut max_seq, cursor.value(inline, flag)?, flag)?,
            "--max-past" => set_u64(&mut max_past, cursor.value(inline, flag)?, flag)?,
            "--max-total" => set_u64(&mut max_total, cursor.value(inline, flag)?, flag)?,
            "--force" => {
                no_inline(inline, flag)?;
                force = true;
            }
            other => return Err(unknown_flag("prepare", other)),
        }
    }

    let prepared = PrepareArgs {
        onnx: required_path(onnx, "--onnx", "prepare")?,
        out: required_path(out, "--out", "prepare")?,
        max_seq: max_seq.unwrap_or(128),
        max_past: max_past.unwrap_or(1023),
        max_total: max_total.unwrap_or(1024),
        force,
    };
    if prepared.max_seq == 0 {
        return Err(CliError::usage("`--max-seq` must be at least 1"));
    }
    if prepared.max_total == 0 {
        return Err(CliError::usage("`--max-total` must be at least 1"));
    }
    Ok(Command::Prepare(prepared))
}

fn parse_inspect(args: &[String]) -> Result<Command, CliError> {
    let mut artifact: Option<PathBuf> = None;
    let mut sections = false;
    let mut positional: Option<PathBuf> = None;

    let mut cursor = Cursor::new(args);
    while let Some((flag, inline)) = cursor.next_flag() {
        match flag {
            "--help" | "-h" => return Ok(Command::Help(HelpTopic::Inspect)),
            "--model" | "--artifact" => set_path(&mut artifact, cursor.value(inline, flag)?, flag)?,
            "--sections" | "--full" => {
                no_inline(inline, flag)?;
                sections = true;
            }
            other if other.starts_with('-') => return Err(unknown_flag("inspect", other)),
            other => {
                if positional.is_some() {
                    return Err(CliError::usage(
                        "`inspect` takes one artifact path; give the others with `--model`",
                    ));
                }
                positional = Some(PathBuf::from(other));
            }
        }
    }
    if let (Some(positional), Some(flag)) = (positional.as_ref(), artifact.as_ref()) {
        if positional != flag {
            return Err(CliError::usage(format!(
                "`inspect` was given two artifact paths (`{}` and `{}`)",
                positional.display(),
                flag.display()
            )));
        }
    }
    let artifact = artifact.or(positional).ok_or_else(|| {
        CliError::usage("`inspect` needs an artifact path: `fastnn-infer inspect model.fnn3`")
    })?;
    Ok(Command::Inspect(InspectArgs { artifact, sections }))
}

fn parse_run(args: &[String]) -> Result<Command, CliError> {
    let mut model = None;
    let mut onnx = None;
    let mut verify_source = false;
    let mut tokens = None;
    let mut decode = None;
    let mut device = None;
    let mut capacity = None;
    let mut top_k = None;
    let mut steps = false;
    let mut weight_int8 = false;

    let mut cursor = Cursor::new(args);
    while let Some((flag, inline)) = cursor.next_flag() {
        match flag {
            "--help" | "-h" => return Ok(Command::Help(HelpTopic::Run)),
            "--model" => set_path(&mut model, cursor.value(inline, flag)?, flag)?,
            "--onnx" => set_path(&mut onnx, cursor.value(inline, flag)?, flag)?,
            "--verify-source" => {
                no_inline(inline, flag)?;
                verify_source = true;
            }
            "--tokens" => set_string(&mut tokens, cursor.value(inline, flag)?, flag)?,
            "--decode" => set_string(&mut decode, cursor.value(inline, flag)?, flag)?,
            "--device" => set_device(&mut device, cursor.value(inline, flag)?, flag)?,
            "--capacity" => set_u64(&mut capacity, cursor.value(inline, flag)?, flag)?,
            "--top-k" => set_usize(&mut top_k, cursor.value(inline, flag)?, flag)?,
            "--steps" => {
                no_inline(inline, flag)?;
                steps = true;
            }
            "--weight-int8" => {
                no_inline(inline, flag)?;
                weight_int8 = true;
            }
            other => return Err(unknown_flag("run", other)),
        }
    }

    let infer = finish_infer(
        "run",
        model,
        onnx,
        verify_source,
        tokens,
        device,
        capacity,
        top_k,
        steps,
        weight_int8,
    )?;
    Ok(Command::Run(RunArgs { infer, decode }))
}

fn parse_generate(args: &[String]) -> Result<Command, CliError> {
    let mut model = None;
    let mut onnx = None;
    let mut verify_source = false;
    let mut tokens = None;
    let mut device = None;
    let mut capacity = None;
    let mut top_k = None;
    let mut steps = false;
    let mut weight_int8 = false;
    let mut max_new_tokens = None;

    let mut cursor = Cursor::new(args);
    while let Some((flag, inline)) = cursor.next_flag() {
        match flag {
            "--help" | "-h" => return Ok(Command::Help(HelpTopic::Generate)),
            "--model" => set_path(&mut model, cursor.value(inline, flag)?, flag)?,
            "--onnx" => set_path(&mut onnx, cursor.value(inline, flag)?, flag)?,
            "--verify-source" => {
                no_inline(inline, flag)?;
                verify_source = true;
            }
            "--tokens" => set_string(&mut tokens, cursor.value(inline, flag)?, flag)?,
            "--device" => set_device(&mut device, cursor.value(inline, flag)?, flag)?,
            "--capacity" => set_u64(&mut capacity, cursor.value(inline, flag)?, flag)?,
            "--top-k" => set_usize(&mut top_k, cursor.value(inline, flag)?, flag)?,
            "--max-new-tokens" => set_u64(&mut max_new_tokens, cursor.value(inline, flag)?, flag)?,
            "--steps" => {
                no_inline(inline, flag)?;
                steps = true;
            }
            "--weight-int8" => {
                no_inline(inline, flag)?;
                weight_int8 = true;
            }
            other => return Err(unknown_flag("generate", other)),
        }
    }

    let infer = finish_infer(
        "generate",
        model,
        onnx,
        verify_source,
        tokens,
        device,
        capacity,
        top_k,
        steps,
        weight_int8,
    )?;
    let max_new_tokens =
        max_new_tokens.ok_or_else(|| CliError::usage("`generate` needs `--max-new-tokens <n>`"))?;
    if max_new_tokens == 0 {
        return Err(CliError::usage("`--max-new-tokens` must be at least 1"));
    }
    Ok(Command::Generate(GenerateArgs {
        infer,
        max_new_tokens,
    }))
}

#[allow(clippy::too_many_arguments)]
fn finish_infer(
    command: &str,
    model: Option<PathBuf>,
    onnx: Option<PathBuf>,
    verify_source: bool,
    tokens: Option<String>,
    device: Option<DeviceSpec>,
    capacity: Option<u64>,
    top_k: Option<usize>,
    steps: bool,
    weight_int8: bool,
) -> Result<InferArgs, CliError> {
    let top_k = top_k.unwrap_or(5);
    if top_k == 0 {
        return Err(CliError::usage("`--top-k` must be at least 1"));
    }
    if let Some(0) = capacity {
        return Err(CliError::usage("`--capacity` must be at least 1 token"));
    }
    if verify_source && onnx.is_none() {
        return Err(CliError::usage(
            "`--verify-source` asks for a source check, but no source was given",
        )
        .with_next(
            "pass the ONNX the artifact was prepared from with `--onnx <model.onnx>`, or drop \
             `--verify-source` to run the artifact alone",
        ));
    }
    Ok(InferArgs {
        model: required_path(model, "--model", command)?,
        onnx,
        tokens: tokens.ok_or_else(|| {
            CliError::usage(format!(
                "`{command}` needs `--tokens <id[,id...]>` (token IDs, not text)"
            ))
        })?,
        device: device.unwrap_or(DeviceSpec::Auto),
        capacity,
        top_k,
        steps,
        weight_int8,
    })
}

fn parse_diagnose(args: &[String]) -> Result<Command, CliError> {
    let mut artifact = None;
    let mut device = None;
    let mut cursor = Cursor::new(args);
    while let Some((flag, inline)) = cursor.next_flag() {
        match flag {
            "--help" | "-h" => return Ok(Command::Help(HelpTopic::Diagnose)),
            "--model" | "--artifact" => set_path(&mut artifact, cursor.value(inline, flag)?, flag)?,
            "--device" => set_device(&mut device, cursor.value(inline, flag)?, flag)?,
            other => return Err(unknown_flag("diagnose", other)),
        }
    }
    Ok(Command::Diagnose(DiagnoseArgs { artifact, device }))
}

// ---------------------------------------------------------------------------
// Cursor and setters
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    args: &'a [String],
    index: usize,
}

impl<'a> Cursor<'a> {
    fn new(args: &'a [String]) -> Cursor<'a> {
        Cursor { args, index: 0 }
    }

    /// Next flag and an inline `=value`, when present.
    fn next_flag(&mut self) -> Option<(&'a str, Option<&'a str>)> {
        let arg = self.args.get(self.index)?;
        self.index += 1;
        match arg.split_once('=') {
            Some((flag, value)) => Some((flag, Some(value))),
            None => Some((arg.as_str(), None)),
        }
    }

    /// The value for a flag, either inline or the next argument.
    fn value(&mut self, inline: Option<&'a str>, flag: &str) -> Result<&'a str, CliError> {
        match inline {
            Some(value) if !value.is_empty() => Ok(value),
            Some(_) => Err(CliError::usage(format!("`{flag}` needs a value"))),
            None => {
                let value = self
                    .args
                    .get(self.index)
                    .ok_or_else(|| CliError::usage(format!("`{flag}` needs a value")))?;
                self.index += 1;
                Ok(value.as_str())
            }
        }
    }
}

fn no_inline(inline: Option<&str>, flag: &str) -> Result<(), CliError> {
    if inline.is_some() {
        Err(CliError::usage(format!("`{flag}` takes no value")))
    } else {
        Ok(())
    }
}

fn unknown_flag(command: &str, flag: &str) -> CliError {
    CliError::usage(format!(
        "unknown flag `{flag}` for `{command}`; run `fastnn-infer help {command}`"
    ))
}

fn set_path(slot: &mut Option<PathBuf>, value: &str, flag: &str) -> Result<(), CliError> {
    if slot.is_some() {
        return Err(CliError::usage(format!("`{flag}` was given twice")));
    }
    *slot = Some(PathBuf::from(value));
    Ok(())
}

fn set_string(slot: &mut Option<String>, value: &str, flag: &str) -> Result<(), CliError> {
    if slot.is_some() {
        return Err(CliError::usage(format!("`{flag}` was given twice")));
    }
    *slot = Some(value.to_string());
    Ok(())
}

fn set_device(slot: &mut Option<DeviceSpec>, value: &str, flag: &str) -> Result<(), CliError> {
    if slot.is_some() {
        return Err(CliError::usage(format!("`{flag}` was given twice")));
    }
    *slot = Some(DeviceSpec::parse(value)?);
    Ok(())
}

fn set_u64(slot: &mut Option<u64>, value: &str, flag: &str) -> Result<(), CliError> {
    if slot.is_some() {
        return Err(CliError::usage(format!("`{flag}` was given twice")));
    }
    let parsed = value.parse::<u64>().map_err(|_| {
        CliError::usage(format!(
            "`{flag}` expects a whole number, but `{value}` is not one"
        ))
    })?;
    *slot = Some(parsed);
    Ok(())
}

fn set_usize(slot: &mut Option<usize>, value: &str, flag: &str) -> Result<(), CliError> {
    let parsed = value.parse::<usize>().map_err(|_| {
        CliError::usage(format!(
            "`{flag}` expects a whole number, but `{value}` is not one"
        ))
    })?;
    if slot.is_some() {
        return Err(CliError::usage(format!("`{flag}` was given twice")));
    }
    *slot = Some(parsed);
    Ok(())
}

fn required_path(value: Option<PathBuf>, flag: &str, command: &str) -> Result<PathBuf, CliError> {
    value.ok_or_else(|| CliError::usage(format!("`{command}` needs `{flag} <path>`")))
}

/// Help text for one topic.
pub fn help_text(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::All => HELP_ALL.to_string(),
        HelpTopic::Prepare => HELP_PREPARE.to_string(),
        HelpTopic::Inspect => HELP_INSPECT.to_string(),
        HelpTopic::Run => HELP_RUN.to_string(),
        HelpTopic::Generate => HELP_GENERATE.to_string(),
        HelpTopic::Diagnose => HELP_DIAGNOSE.to_string(),
    }
}

const HELP_ALL: &str = "\
fastnn-infer — FastNN 3.0 inference CLI (FN3-17)

USAGE:
  fastnn-infer <command> [options]

COMMANDS:
  prepare    ONNX -> versioned .fnn3 artifact (atomic write, read-back check)
  inspect    artifact header and, with --sections, its section summary
  run        token-ID prefill and optional cached decode on one device
  generate   greedy, deterministic token generation (no sampling)
  diagnose   build/device/backend matrix; says what was only enumerated

GLOBAL:
  -h, --help [topic]   this list, or one command's options
  -V, --version        print the tool version

COMMON INFERENCE FLAGS:
  --model <artifact>   prepared .fnn3 artifact; carries graph, weights, geometry
                       and envelope, so it runs with no ONNX file
  --onnx <model.onnx>  optional: verify this source against the artifact before
                       running (metadata, geometry and every weight byte)
  --verify-source      state that source check explicitly (requires --onnx)
  --tokens <ids>       prompt token IDs, comma/whitespace separated, or @file
  --device <request>   auto | cpu | vulkan | vulkan:<index> | vulkan:<uuid>
  --capacity <n>       KV capacity in tokens (default: envelope total.max)
  --top-k <n>          logits shown per step (default 5)
  --steps              per-step plan/cache/timing diagnostics

TEXT INPUT:
  This alpha takes token IDs. It does not ship a tokenizer and will not guess:
  a different BPE would silently produce different tokens and different logits.
  Use the reference tokenizer to obtain IDs, then pass them with --tokens.

EXIT CODES:
  0 success  2 usage  3 model/input  4 device  5 inference  1 internal

Run `fastnn-infer help <command>` for one command.
";

const HELP_PREPARE: &str = "\
fastnn-infer prepare — ONNX -> versioned prepared artifact

USAGE:
  fastnn-infer prepare --onnx <model.onnx> --out <model.fnn3> [options]

OPTIONS:
  --max-seq <n>     maximum new tokens in one call (default 128)
  --max-past <n>    maximum already-cached tokens (default 1023)
  --max-total <n>   maximum past + seq (default 1024)
  --force           replace an existing --out file

The artifact is written atomically and read back before success is reported.
The source ONNX is not modified.
";

const HELP_INSPECT: &str = "\
fastnn-infer inspect — what is this artifact?

USAGE:
  fastnn-infer inspect <model.fnn3> [--sections]

OPTIONS:
  --sections, --full   decode META, GEOM, ENV, VALS, NODE and WGHT and print
                       their summary; without it only the 64-byte header is
                       parsed and the payload checksum is not recomputed
";

const HELP_RUN: &str = "\
fastnn-infer run — token-ID prefill and cached decode

USAGE:
  fastnn-infer run --model <model.fnn3> --tokens <ids> [options]
  fastnn-infer run --model <model.fnn3> --onnx <model.onnx> --tokens <ids> [options]

OPTIONS:
  --decode <ids>    feed these token IDs one at a time after the prompt
  --onnx <path>     verify this ONNX against the artifact before running
                    (optional; the artifact alone carries everything that runs)
  --device ...      auto (default), cpu, vulkan, vulkan:<index>, vulkan:<uuid>
  --capacity <n>    KV capacity; must hold prompt + decoded tokens
  --top-k <n>       logits shown (default 5)
  --steps           print token, cache length, plan id, plan hits/misses, time
  --weight-int8     INT8 projection weights, FP32 activations (CPU/resident Vulkan)

Output names the requested and resolved device, the backend that executed, the
logits summary and the predicted next token. Decoding is greedy: the highest
logit, ties broken by the lowest token ID. No sampling.
";

const HELP_GENERATE: &str = "\
fastnn-infer generate — greedy, deterministic generation

USAGE:
  fastnn-infer generate --model <model.fnn3> --tokens <ids> \\
      --max-new-tokens <n> [options]
  fastnn-infer generate --model <model.fnn3> --onnx <model.onnx> --tokens <ids> \\
      --max-new-tokens <n> [options]

OPTIONS:
  --max-new-tokens <n>  how many tokens to generate (at least 1)
  --onnx <path>         verify this ONNX against the artifact before running
                        (optional; the artifact alone carries everything that runs)
  --device ...          auto (default), cpu, vulkan, vulkan:<index>, vulkan:<uuid>
  --capacity <n>        KV capacity; must hold prompt + generated tokens
  --top-k <n>           logits shown (default 5)
  --steps               per-step diagnostics
  --weight-int8         INT8 projection weights, FP32 activations (CPU/resident Vulkan)

Generation is greedy argmax on f32 logits: no temperature, top-k sampling or
randomness of any kind. The same artifact, prompt, device and capacity produce
the same IDs.
";

const HELP_DIAGNOSE: &str = "\
fastnn-infer diagnose — build, device and backend matrix

USAGE:
  fastnn-infer diagnose [--artifact <model.fnn3>] [--device <request>]

Without --device it reports every device the loader lists and, with the vulkan
feature, whether the backend can open each one. It does not execute the model:
'enumerated', 'opened' and 'ran the model' are printed as separate facts.
";

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    #[test]
    fn prepare_defaults_are_explicit() {
        let command = parse(&argv(&["prepare", "--onnx", "m.onnx", "--out", "m.fnn3"]))
            .expect("prepare parses");
        let Command::Prepare(args) = command else {
            panic!("wrong command");
        };
        assert_eq!(args.max_seq, 128);
        assert_eq!(args.max_past, 1023);
        assert_eq!(args.max_total, 1024);
        assert!(!args.force);
    }

    #[test]
    fn equals_form_and_positional_inspect_are_accepted() {
        let command = parse(&argv(&["inspect", "--sections", "m.fnn3"])).expect("inspect parses");
        let Command::Inspect(args) = command else {
            panic!("wrong command");
        };
        assert_eq!(args.artifact, PathBuf::from("m.fnn3"));
        assert!(args.sections);

        let command = parse(&argv(&[
            "run",
            "--tokens=1,2",
            "--model=m.fnn3",
            "--onnx=m.onnx",
        ]))
        .expect("equals form parses");
        let Command::Run(args) = command else {
            panic!("wrong command");
        };
        assert_eq!(args.infer.tokens, "1,2");
        assert_eq!(args.infer.top_k, 5);
        assert!(!args.infer.steps);
    }

    #[test]
    fn duplicate_single_value_flags_are_rejected() {
        let error = parse(&argv(&[
            "run", "--model", "a", "--model", "b", "--onnx", "m", "--tokens", "1",
        ]))
        .expect_err("duplicate must fail");
        assert!(error.message().contains("given twice"), "{error:?}");
    }

    #[test]
    fn missing_values_and_unknown_flags_name_themselves() {
        let error = parse(&argv(&["prepare", "--onnx"])).expect_err("missing value");
        assert!(error.message().contains("--onnx"), "{error:?}");

        let error = parse(&argv(&["run", "--wat"])).expect_err("unknown flag");
        assert!(error.message().contains("--wat"), "{error:?}");
        assert!(error.message().contains("help run"), "{error:?}");
    }

    #[test]
    fn subcommand_help_is_accepted_in_every_command() {
        for (command, topic) in [
            ("prepare", HelpTopic::Prepare),
            ("inspect", HelpTopic::Inspect),
            ("run", HelpTopic::Run),
            ("generate", HelpTopic::Generate),
            ("diagnose", HelpTopic::Diagnose),
        ] {
            let parsed = parse(&argv(&[command, "--help"])).expect("help parses");
            assert_eq!(parsed, Command::Help(topic), "{command}");
        }
    }

    #[test]
    fn devices_are_parsed_or_rejected_with_the_accepted_forms() {
        let command = parse(&argv(&[
            "run", "--model", "m", "--onnx", "o", "--tokens", "1", "--device", "vulkan:1",
        ]))
        .expect("device parses");
        let Command::Run(args) = command else {
            panic!("wrong command");
        };
        assert_eq!(args.infer.device, DeviceSpec::vulkan_index(1));

        let error = parse(&argv(&[
            "run", "--model", "m", "--onnx", "o", "--tokens", "1", "--device", "gpu",
        ]))
        .expect_err("bad device");
        assert!(error.message().contains("vulkan:<uuid>"), "{error:?}");
    }

    #[test]
    fn zero_length_requests_are_usage_errors() {
        for args in [
            argv(&[
                "generate",
                "--model",
                "m",
                "--onnx",
                "o",
                "--tokens",
                "1",
                "--max-new-tokens",
                "0",
            ]),
            argv(&[
                "run", "--model", "m", "--onnx", "o", "--tokens", "1", "--top-k", "0",
            ]),
            argv(&[
                "run",
                "--model",
                "m",
                "--onnx",
                "o",
                "--tokens",
                "1",
                "--capacity",
                "0",
            ]),
        ] {
            let error = parse(&args).expect_err("zero is not a valid count");
            assert_eq!(error.code(), super::super::EXIT_USAGE, "{error:?}");
        }
    }
}
