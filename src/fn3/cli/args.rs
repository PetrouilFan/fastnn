//! Command-line syntax for `fastnn-infer` (the v3 surface).
//!
//! Hand-written rather than derived so every rejection names the offending flag
//! and the accepted forms, and so the same parser is callable from tests. The
//! parser accepts `--flag value` and `--flag=value`; it rejects duplicates for
//! options that carry a single value instead of letting the last one win.

use std::path::PathBuf;

use crate::inference::runtime::Threading;

use super::CliError;

/// One parsed invocation.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// ONNX -> versioned artifact.
    Prepare(PrepareArgs),
    /// Map an artifact and build a session, without running it.
    Load(LoadArgs),
    /// Map an artifact, build a session and run one forward.
    Run(RunArgs),
    /// Describe an artifact without mapping or executing it.
    Inspect(InspectArgs),
    /// Explain the stored plan, segment by segment.
    Explain(ExplainArgs),
    /// Diagnose an ONNX model before producing an artifact.
    Doctor(DoctorArgs),
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
    Load,
    Run,
    Inspect,
    Explain,
    Doctor,
}

/// A thread request, parsed once and mapped to [`Threading`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadsArg {
    /// Physical cores.
    Auto,
    /// Exactly `n` workers.
    Fixed(usize),
}

impl ThreadsArg {
    /// Parse `auto` or a positive whole number.
    pub fn parse(value: &str) -> Result<ThreadsArg, CliError> {
        if value == "auto" {
            return Ok(ThreadsArg::Auto);
        }
        let workers = value.parse::<usize>().map_err(|_| {
            CliError::usage(format!(
                "`--threads` expects `auto` or a whole number, but `{value}` is neither"
            ))
        })?;
        if workers == 0 {
            return Err(CliError::usage("`--threads` must be at least 1"));
        }
        if workers > 1024 {
            return Err(CliError::usage(format!(
                "`--threads {workers}` is above this build's limit of 1024"
            )));
        }
        Ok(ThreadsArg::Fixed(workers))
    }

    /// The runtime request.
    pub fn threading(self) -> Threading {
        match self {
            ThreadsArg::Auto => Threading::Auto,
            ThreadsArg::Fixed(workers) => Threading::Fixed(workers),
        }
    }

    /// Render for a report.
    pub fn render(self) -> String {
        match self {
            ThreadsArg::Auto => "auto".to_string(),
            ThreadsArg::Fixed(workers) => workers.to_string(),
        }
    }
}

/// Arguments of `prepare`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareArgs {
    /// Source ONNX file.
    pub onnx: PathBuf,
    /// Destination artifact.
    pub out: PathBuf,
    /// Target profile: `cpu`, `cpu-reference` or `all`.
    pub target: String,
    /// Execution precision: `f32`.
    pub precision: String,
    /// Replace an existing destination.
    pub force: bool,
    /// Concrete extents for graph inputs, as `name=d0,d1,...`.
    pub shapes: Vec<(String, Vec<u64>)>,
}

/// Arguments of `load`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadArgs {
    /// Artifact to map.
    pub model: PathBuf,
    /// Target plan; the artifact default when absent.
    pub target: Option<String>,
    /// Worker request.
    pub threads: ThreadsArg,
}

/// One `--input` value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputArg {
    /// Fill the input with zeros.
    Zeros(String),
    /// Fill the input with ones.
    Ones(String),
    /// Raw little-endian `f32` bytes from this file.
    File(String, PathBuf),
}

impl InputArg {
    /// Input name.
    pub fn name(&self) -> &str {
        match self {
            InputArg::Zeros(name) | InputArg::Ones(name) | InputArg::File(name, _) => name,
        }
    }

    fn parse(value: &str) -> Result<InputArg, CliError> {
        let (name, source) = value.split_once('=').ok_or_else(|| {
            CliError::usage(format!(
                "`--input` expects `name=zeros`, `name=ones` or `name=<raw f32 file>`, found \
                 `{value}`"
            ))
        })?;
        if name.is_empty() {
            return Err(CliError::usage("`--input` has an empty input name"));
        }
        Ok(match source {
            "zeros" => InputArg::Zeros(name.to_string()),
            "ones" => InputArg::Ones(name.to_string()),
            other => {
                let path = other.strip_prefix('@').unwrap_or(other);
                if path.is_empty() {
                    return Err(CliError::usage(format!(
                        "`--input {name}=` has no file path"
                    )));
                }
                InputArg::File(name.to_string(), PathBuf::from(path))
            }
        })
    }
}

/// Arguments of `run`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunArgs {
    /// Artifact to map.
    pub model: PathBuf,
    /// Target plan; the artifact default when absent.
    pub target: Option<String>,
    /// Worker request.
    pub threads: ThreadsArg,
    /// Named inputs.
    pub inputs: Vec<InputArg>,
    /// Optional directory for raw `f32` outputs.
    pub output_dir: Option<PathBuf>,
    /// Warm-up forwards before the measured series.
    pub warmup: usize,
    /// Measured forwards; 1 reports the cold forward only.
    pub repeat: usize,
}

/// Arguments of `inspect`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectArgs {
    /// Artifact to describe.
    pub artifact: PathBuf,
}

/// Arguments of `explain`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplainArgs {
    /// Artifact whose stored plan is explained.
    pub artifact: PathBuf,
    /// Target plan; the artifact default when absent.
    pub target: Option<String>,
}

/// Arguments of `doctor`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoctorArgs {
    /// ONNX to diagnose.
    pub onnx: PathBuf,
    /// Input-shape bindings to diagnose with, as `name=d0,d1,...`.
    pub shapes: Vec<(String, Vec<u64>)>,
}

/// Parse an argument vector (without the program name).
pub fn parse(args: &[String]) -> Result<Command, CliError> {
    let Some(first) = args.first() else {
        return Ok(Command::Help(HelpTopic::All));
    };
    match first.as_str() {
        "prepare" => parse_prepare(&args[1..]),
        "load" => parse_load(&args[1..]),
        "run" => parse_run(&args[1..]),
        "inspect" => parse_inspect(&args[1..]),
        "explain" => parse_explain(&args[1..]),
        "doctor" => parse_doctor(&args[1..]),
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
        "load" => HelpTopic::Load,
        "run" => HelpTopic::Run,
        "inspect" => HelpTopic::Inspect,
        "explain" => HelpTopic::Explain,
        "doctor" => HelpTopic::Doctor,
        other => {
            return Err(CliError::usage(format!(
                "unknown help topic `{other}`; topics: prepare, load, run, inspect, explain, doctor"
            )))
        }
    };
    Ok(Command::Help(topic))
}

fn parse_prepare(args: &[String]) -> Result<Command, CliError> {
    let mut onnx = None;
    let mut out = None;
    let mut target = None;
    let mut precision = None;
    let mut force = false;
    let mut shapes = Vec::new();
    let mut cursor = Cursor::new(args);
    while let Some((flag, inline)) = cursor.next_flag() {
        match flag {
            "--help" | "-h" => return Ok(Command::Help(HelpTopic::Prepare)),
            "--onnx" => set_path(&mut onnx, cursor.value(inline, flag)?, flag)?,
            "--out" => set_path(&mut out, cursor.value(inline, flag)?, flag)?,
            "--target" => set_string(&mut target, cursor.value(inline, flag)?, flag)?,
            "--precision" => set_string(&mut precision, cursor.value(inline, flag)?, flag)?,
            "--shape" => push_shape(&mut shapes, cursor.value(inline, flag)?, flag)?,
            "--force" => {
                no_inline(inline, flag)?;
                force = true;
            }
            other => return Err(unknown_flag("prepare", other)),
        }
    }
    let target = target.unwrap_or_else(|| crate::inference::model::format::TARGET_CPU.to_string());
    if !crate::inference::model::format::TARGETS.contains(&target.as_str()) {
        return Err(CliError::usage(format!(
            "`--target {target}` is not defined; use {}",
            crate::inference::model::format::TARGETS.join(", ")
        )));
    }
    let precision = precision.unwrap_or_else(|| "f32".to_string());
    // The name is validated here; whether this build can *execute* it is
    // decided by `prepare` (which refuses w8a8 with the measured reason), so
    // the CLI does not duplicate that policy.
    if !matches!(precision.as_str(), "f32" | "fp32" | "w8a8" | "int8") {
        return Err(CliError::usage(format!(
            "`--precision {precision}` is not a precision this build knows; use `f32` or `w8a8`"
        )));
    }
    Ok(Command::Prepare(PrepareArgs {
        onnx: required_path(onnx, "--onnx", "prepare")?,
        out: required_path(out, "--out", "prepare")?,
        target,
        precision,
        force,
        shapes,
    }))
}

fn parse_load(args: &[String]) -> Result<Command, CliError> {
    let mut model = None;
    let mut target = None;
    let mut threads = None;
    let mut positional = None;
    let mut cursor = Cursor::new(args);
    while let Some((flag, inline)) = cursor.next_flag() {
        match flag {
            "--help" | "-h" => return Ok(Command::Help(HelpTopic::Load)),
            "--model" | "--artifact" => set_path(&mut model, cursor.value(inline, flag)?, flag)?,
            "--target" => set_string(&mut target, cursor.value(inline, flag)?, flag)?,
            "--threads" => set_threads(&mut threads, cursor.value(inline, flag)?, flag)?,
            other if other.starts_with('-') => return Err(unknown_flag("load", other)),
            other => set_positional(&mut positional, other, "load")?,
        }
    }
    Ok(Command::Load(LoadArgs {
        model: finish_model(model, positional, "load")?,
        target,
        threads: threads.unwrap_or(ThreadsArg::Auto),
    }))
}

fn parse_run(args: &[String]) -> Result<Command, CliError> {
    let mut model = None;
    let mut target = None;
    let mut threads = None;
    let mut inputs = Vec::new();
    let mut output_dir = None;
    let mut warmup = None;
    let mut repeat = None;
    let mut positional = None;
    let mut cursor = Cursor::new(args);
    while let Some((flag, inline)) = cursor.next_flag() {
        match flag {
            "--help" | "-h" => return Ok(Command::Help(HelpTopic::Run)),
            "--model" | "--artifact" => set_path(&mut model, cursor.value(inline, flag)?, flag)?,
            "--target" => set_string(&mut target, cursor.value(inline, flag)?, flag)?,
            "--threads" => set_threads(&mut threads, cursor.value(inline, flag)?, flag)?,
            "--input" => {
                let value = cursor.value(inline, flag)?;
                let input = InputArg::parse(value)?;
                if inputs
                    .iter()
                    .any(|existing: &InputArg| existing.name() == input.name())
                {
                    return Err(CliError::usage(format!(
                        "`--input {}` was given twice",
                        input.name()
                    )));
                }
                inputs.push(input);
            }
            "--output-dir" => set_path(&mut output_dir, cursor.value(inline, flag)?, flag)?,
            "--warmup" => set_count(&mut warmup, cursor.value(inline, flag)?, flag, true)?,
            "--repeat" => set_count(&mut repeat, cursor.value(inline, flag)?, flag, false)?,
            other if other.starts_with('-') => return Err(unknown_flag("run", other)),
            other => set_positional(&mut positional, other, "run")?,
        }
    }
    Ok(Command::Run(RunArgs {
        model: finish_model(model, positional, "run")?,
        target,
        threads: threads.unwrap_or(ThreadsArg::Auto),
        inputs,
        output_dir,
        warmup: warmup.unwrap_or(0),
        repeat: repeat.unwrap_or(1),
    }))
}

fn parse_inspect(args: &[String]) -> Result<Command, CliError> {
    let mut artifact = None;
    let mut positional = None;
    let mut cursor = Cursor::new(args);
    while let Some((flag, inline)) = cursor.next_flag() {
        match flag {
            "--help" | "-h" => return Ok(Command::Help(HelpTopic::Inspect)),
            "--model" | "--artifact" => set_path(&mut artifact, cursor.value(inline, flag)?, flag)?,
            other if other.starts_with('-') => return Err(unknown_flag("inspect", other)),
            other => set_positional(&mut positional, other, "inspect")?,
        }
    }
    Ok(Command::Inspect(InspectArgs {
        artifact: finish_model(artifact, positional, "inspect")?,
    }))
}

fn parse_explain(args: &[String]) -> Result<Command, CliError> {
    let mut artifact = None;
    let mut target = None;
    let mut positional = None;
    let mut cursor = Cursor::new(args);
    while let Some((flag, inline)) = cursor.next_flag() {
        match flag {
            "--help" | "-h" => return Ok(Command::Help(HelpTopic::Explain)),
            "--model" | "--artifact" => set_path(&mut artifact, cursor.value(inline, flag)?, flag)?,
            "--target" => set_string(&mut target, cursor.value(inline, flag)?, flag)?,
            other if other.starts_with('-') => return Err(unknown_flag("explain", other)),
            other => set_positional(&mut positional, other, "explain")?,
        }
    }
    Ok(Command::Explain(ExplainArgs {
        artifact: finish_model(artifact, positional, "explain")?,
        target,
    }))
}

fn parse_doctor(args: &[String]) -> Result<Command, CliError> {
    let mut onnx = None;
    let mut shapes = Vec::new();
    let mut positional = None;
    let mut cursor = Cursor::new(args);
    while let Some((flag, inline)) = cursor.next_flag() {
        match flag {
            "--help" | "-h" => return Ok(Command::Help(HelpTopic::Doctor)),
            "--onnx" | "--model" => set_path(&mut onnx, cursor.value(inline, flag)?, flag)?,
            "--shape" => push_shape(&mut shapes, cursor.value(inline, flag)?, flag)?,
            other if other.starts_with('-') => return Err(unknown_flag("doctor", other)),
            other => set_positional(&mut positional, other, "doctor")?,
        }
    }
    Ok(Command::Doctor(DoctorArgs {
        onnx: finish_model(onnx, positional, "doctor")?,
        shapes,
    }))
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
            Some((flag, value)) if flag.starts_with('-') => Some((flag, Some(value))),
            _ => Some((arg.as_str(), None)),
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

fn set_threads(slot: &mut Option<ThreadsArg>, value: &str, flag: &str) -> Result<(), CliError> {
    if slot.is_some() {
        return Err(CliError::usage(format!("`{flag}` was given twice")));
    }
    *slot = Some(ThreadsArg::parse(value)?);
    Ok(())
}

/// Parse a whole-number count. `allow_zero` distinguishes "repeat none" from
/// "repeat at least once": a warm-up may be zero, a measured series may not.
fn set_count(
    slot: &mut Option<usize>,
    value: &str,
    flag: &str,
    allow_zero: bool,
) -> Result<(), CliError> {
    if slot.is_some() {
        return Err(CliError::usage(format!("`{flag}` was given twice")));
    }
    let parsed: usize = value.parse().map_err(|_| {
        CliError::usage(format!(
            "`{flag}` expects a whole number, but `{value}` is not"
        ))
    })?;
    if parsed == 0 && !allow_zero {
        return Err(CliError::usage(format!("`{flag}` must be at least 1")));
    }
    if parsed > 1_000_000 {
        return Err(CliError::usage(format!(
            "`{flag} {parsed}` is unreasonably large; cap it at 1000000"
        )));
    }
    *slot = Some(parsed);
    Ok(())
}

/// Parse one `--shape name=d0,d1,...` binding.
///
/// A dynamic ONNX export has no numeric extents to plan against, so the caller
/// supplies the shape the model will be run at. Extents must be positive whole
/// numbers: a zero would describe an empty tensor, and nothing downstream can
/// plan one.
fn push_shape(
    shapes: &mut Vec<(String, Vec<u64>)>,
    value: &str,
    flag: &str,
) -> Result<(), CliError> {
    let Some((name, extents)) = value.split_once('=') else {
        return Err(CliError::usage(format!(
            "`{flag} {value}` has no `=`; write `{flag} images=1,3,640,640`"
        )));
    };
    let name = name.trim();
    if name.is_empty() {
        return Err(CliError::usage(format!(
            "`{flag}` needs a graph input name before `=`"
        )));
    }
    if shapes.iter().any(|(existing, _)| existing == name) {
        return Err(CliError::usage(format!(
            "`{flag}` was given twice for `{name}`"
        )));
    }
    let mut dims = Vec::new();
    for part in extents.split(',') {
        let part = part.trim();
        let extent: u64 = part.parse().map_err(|_| {
            CliError::usage(format!(
                "`{flag} {value}`: `{part}` is not a whole number of elements"
            ))
        })?;
        if extent == 0 {
            return Err(CliError::usage(format!(
                "`{flag} {value}`: a bound extent must be at least 1"
            )));
        }
        dims.push(extent);
    }
    if dims.is_empty() {
        return Err(CliError::usage(format!(
            "`{flag} {value}` has no extents after `=`"
        )));
    }
    if dims.len() > 8 {
        return Err(CliError::usage(format!(
            "`{flag} {value}` has {} extents; this build accepts at most 8",
            dims.len()
        )));
    }
    shapes.push((name.to_string(), dims));
    Ok(())
}

fn set_positional(slot: &mut Option<PathBuf>, value: &str, command: &str) -> Result<(), CliError> {
    if slot.is_some() {
        return Err(CliError::usage(format!(
            "`{command}` takes one path; give the others with a flag"
        )));
    }
    *slot = Some(PathBuf::from(value));
    Ok(())
}

fn required_path(value: Option<PathBuf>, flag: &str, command: &str) -> Result<PathBuf, CliError> {
    value.ok_or_else(|| CliError::usage(format!("`{command}` needs `{flag} <path>`")))
}

fn finish_model(
    flag: Option<PathBuf>,
    positional: Option<PathBuf>,
    command: &str,
) -> Result<PathBuf, CliError> {
    match (flag, positional) {
        (Some(flag), Some(positional)) if flag != positional => Err(CliError::usage(format!(
            "`{command}` was given two paths (`{}` and `{}`)",
            flag.display(),
            positional.display()
        ))),
        (Some(flag), _) => Ok(flag),
        (None, Some(positional)) => Ok(positional),
        (None, None) => Err(CliError::usage(format!(
            "`{command}` needs a path: `fastnn-infer {command} model.fnn3`"
        ))),
    }
}

/// Help text for one topic.
pub fn help_text(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::All => HELP_ALL.to_string(),
        HelpTopic::Prepare => HELP_PREPARE.to_string(),
        HelpTopic::Load => HELP_LOAD.to_string(),
        HelpTopic::Run => HELP_RUN.to_string(),
        HelpTopic::Inspect => HELP_INSPECT.to_string(),
        HelpTopic::Explain => HELP_EXPLAIN.to_string(),
        HelpTopic::Doctor => HELP_DOCTOR.to_string(),
    }
}

const HELP_ALL: &str = "\
fastnn-infer — FastNN 3.0 inference CLI

USAGE:
  fastnn-infer <command> [options]

COMMANDS:
  prepare    ONNX -> versioned .fnn3 artifact (atomic write, read-back check)
  load       map an artifact and build a session, without running it
  run        map an artifact, build a session and run one forward
  inspect    describe a header, graph, plans and hashes without mapping weights
  explain    list the stored plan segment by segment: engine, shape, fallback
  doctor     diagnose an ONNX model before preparing it

GLOBAL:
  -h, --help [topic]   this list, or one command's options
  -V, --version        print the tool version

EXIT CODES:
  0 success  2 usage  3 model/input  5 inference  1 internal

Run `fastnn-infer help <command>` for one command.
";

const HELP_PREPARE: &str = "\
fastnn-infer prepare — ONNX -> versioned prepared artifact

USAGE:
  fastnn-infer prepare --onnx <model.onnx> --out <model.fnn3> [options]

OPTIONS:
  --target <name>     cpu (default), cpu-reference or all
  --precision <name>  f32 (default) or w8a8 (refused with a measured reason: #103)
  --shape name=d,...  bind a graph input's extents (e.g. images=1,3,640,640)
  --force             replace an existing --out file

A dynamic ONNX export declares its input as `[batch, 3, height, width]`; the
plan needs one concrete shape per input, so bind it with --shape. `doctor`
prints the declared inputs and the dynamic symbols they use.

The artifact is written atomically and read back through the strict reader
before success is reported. The source ONNX is not modified.
";

const HELP_LOAD: &str = "\
fastnn-infer load — map an artifact and build a session

USAGE:
  fastnn-infer load <model.fnn3> [--target <name>] [--threads auto|<n>]

Maps the artifact, validates every section hash and builds the recorded plan.
No ONNX file is read and the plan is not rebuilt. Nothing executes.
";

const HELP_RUN: &str = "\
fastnn-infer run — one forward from an artifact

USAGE:
  fastnn-infer run <model.fnn3> --input <name>=<value> [options]

OPTIONS:
  --input name=zeros      fill the input with zeros
  --input name=ones       fill the input with ones
  --input name=<file>     raw little-endian f32 bytes (exact element count)
  --target <name>         target plan; the artifact default when absent
  --threads auto|<n>      worker request (default auto = physical cores)
  --output-dir <dir>      write each output as raw little-endian f32
  --warmup <n>            forwards before the measured series (default 0)
  --repeat <n>            measured forwards (default 1); >1 prints median/p95

Every graph input must be supplied exactly once. An output is printed as its
name, shape, element count, min/max/mean and finite count.

With `--repeat`, the reported number is the forward only: the artifact is
already mapped, the plan is already built and the inputs are already resident,
so the series measures the executor. Use `--warmup` to exclude the first
forward's page faults on the mapped weights.
";

const HELP_INSPECT: &str = "\
fastnn-infer inspect — what is in this artifact?

USAGE:
  fastnn-infer inspect <model.fnn3>

Prints the header, graph size, every target plan, weight bytes, quantization
metadata and the stored section hashes. It never maps the weight payload:
GRAPH, PLANS, QUANT and HASHES are read, the WEIGHTS table is read, and the
weight payload is not. `mapped bytes: 0` is printed as the evidence.
";

const HELP_EXPLAIN: &str = "\
fastnn-infer explain — what will run, segment by segment

USAGE:
  fastnn-infer explain <model.fnn3> [--target <name>]

Prints the stored plan: every segment with its engine (scalar or oneDNN), op or
fused chain, output slots and bytes, and the source node. Then lists the
provider's fallbacks with their reasons, the fused chains and the layout
reorders. The plan is read from the artifact; no session is opened and the
weight payload is not mapped.
";

const HELP_DOCTOR: &str = "\
fastnn-infer doctor — diagnose an ONNX model before preparing it

USAGE:
  fastnn-infer doctor <model.onnx>

Imports the model and reports unsupported operators, dynamic-shape limits,
dtype problems and an expected-memory estimate before any execution. Every
message names the source node, e.g. `node 7 (onnx \"Conv\", output \"y\")`.

OPTIONS:
  --shape name=d,...  diagnose with a graph input bound to concrete extents

Without --shape a dynamic export is reported as not ready: the plan requires
every extent to be static. Pass the same bindings you would pass to prepare to
see the verdict the model will actually get.
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
        assert_eq!(args.target, "cpu");
        assert_eq!(args.precision, "f32");
        assert!(!args.force);
        assert!(args.shapes.is_empty());
    }

    #[test]
    fn a_shape_binding_reaches_the_prepare_arguments() {
        let command = parse(&argv(&[
            "prepare",
            "--onnx",
            "m.onnx",
            "--out",
            "m.fnn3",
            "--shape",
            "images=1,3,640,640",
        ]))
        .expect("prepare parses a binding");
        let Command::Prepare(args) = command else {
            panic!("wrong command");
        };
        assert_eq!(
            args.shapes,
            vec![("images".to_string(), vec![1, 3, 640, 640])]
        );

        let command =
            parse(&argv(&["doctor", "m.onnx", "--shape=x=2,4"])).expect("doctor parses a binding");
        let Command::Doctor(args) = command else {
            panic!("wrong command");
        };
        assert_eq!(args.shapes, vec![("x".to_string(), vec![2, 4])]);
    }

    #[test]
    fn a_malformed_shape_binding_is_a_usage_error() {
        for (value, expected) in [
            ("images", "has no `=`"),
            ("=1,2,3", "graph input name"),
            ("images=1,x,3", "not a whole number"),
            ("images=1,0,3", "at least 1"),
            ("images=1,2,3,4,5,6,7,8,9", "at most 8"),
        ] {
            let error = parse(&argv(&[
                "prepare", "--onnx", "m.onnx", "--out", "m.fnn3", "--shape", value,
            ]))
            .expect_err(value);
            assert!(
                error.message().contains(expected),
                "{value}: expected `{expected}` in {}",
                error.message()
            );
        }

        let error = parse(&argv(&[
            "prepare",
            "--onnx",
            "m.onnx",
            "--out",
            "m.fnn3",
            "--shape",
            "images=1,3",
            "--shape",
            "images=2,3",
        ]))
        .expect_err("duplicate");
        assert!(error.message().contains("twice"), "{}", error.message());
    }

    #[test]
    fn equals_form_and_positional_paths_are_accepted() {
        let command = parse(&argv(&["inspect", "m.fnn3"])).expect("inspect parses");
        let Command::Inspect(args) = command else {
            panic!("wrong command");
        };
        assert_eq!(args.artifact, PathBuf::from("m.fnn3"));

        let command = parse(&argv(&[
            "run",
            "--model=m.fnn3",
            "--input=images=zeros",
            "--threads=4",
        ]))
        .expect("equals form parses");
        let Command::Run(args) = command else {
            panic!("wrong command");
        };
        assert_eq!(args.threads, ThreadsArg::Fixed(4));
        assert_eq!(args.inputs, vec![InputArg::Zeros("images".to_string())]);
    }

    #[test]
    fn duplicate_single_value_flags_are_rejected() {
        let error = parse(&argv(&[
            "run", "--model", "a", "--model", "b", "--input", "x=zeros",
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
            ("load", HelpTopic::Load),
            ("run", HelpTopic::Run),
            ("inspect", HelpTopic::Inspect),
            ("doctor", HelpTopic::Doctor),
        ] {
            let parsed = parse(&argv(&[command, "--help"])).expect("help parses");
            assert_eq!(parsed, Command::Help(topic), "{command}");
        }
    }

    #[test]
    fn zero_threads_and_unknown_targets_are_usage_errors() {
        let error = parse(&argv(&["load", "m.fnn3", "--threads", "0"]))
            .expect_err("zero threads is not a worker count");
        assert_eq!(error.code(), super::super::EXIT_USAGE);
        let error = parse(&argv(&[
            "prepare", "--onnx", "m", "--out", "o", "--target", "gpu",
        ]))
        .expect_err("unknown target");
        assert!(error.message().contains("cpu-reference"), "{error:?}");
    }
}
