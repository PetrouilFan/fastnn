//! `fastnn-infer`: the FastNN 3.0 inference command line (FN3-17 / #63).
//!
//! The binary is a thin wrapper over [`run`]: argument parsing, dispatch and
//! exit codes live here so the integration tests can drive the same code path
//! the user does, including the messages, without spawning a process.
//!
//! # Commands
//!
//! ```text
//! prepare   ONNX -> versioned FNN3PREP artifact (atomic write, read-back check)
//! inspect   header and optional section summary; no session, no weights
//! run       token-ID prefill and optional cached decode on one explicit device
//! generate  greedy, deterministic token generation (no sampling)
//! diagnose  build/device/backend matrix, honest about what was only enumerated
//! ```
//!
//! # Exit codes
//!
//! | Code | Meaning |
//! ---:|---|
//! | 0 | success |
//! | 1 | internal failure (an invariant this tool owns) |
//! | 2 | usage error: unknown command or flag, missing or malformed value |
//! | 3 | model/input error: missing file, bad magic, future version, corrupt payload, ONNX mismatch |
//! | 4 | device error: unavailable device, build without the requested backend, driver cannot open |
//! | 5 | inference error: envelope or KV capacity exceeded, unsupported op, invalid logits |
//!
//! Every error is printed as `error: <what>` followed by `next: <action>`, and
//! no user-controlled input is indexed, unwrapped or allowed to panic.

pub mod args;
mod devices;
mod digest;
mod infer;
mod inspect;
mod prepare;

use std::io::Write;

pub use args::{Command, HelpTopic};
pub use devices::DeviceSpec;

/// Success.
pub const EXIT_OK: i32 = 0;
/// An invariant of this tool failed.
pub const EXIT_INTERNAL: i32 = 1;
/// The command line was not understood.
pub const EXIT_USAGE: i32 = 2;
/// The model file, or the ONNX it was prepared from, is unusable.
pub const EXIT_MODEL: i32 = 3;
/// The requested device or backend cannot be used by this build.
pub const EXIT_DEVICE: i32 = 4;
/// Inference was refused by the model's bounds, the cache or a backend.
pub const EXIT_INFERENCE: i32 = 5;

/// A failure with a user-facing message, the next step, and an exit code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliError {
    code: i32,
    message: String,
    next: Option<String>,
}

impl CliError {
    pub fn new(code: i32, message: impl Into<String>) -> CliError {
        CliError {
            code,
            message: message.into(),
            next: None,
        }
    }

    /// The command line was not understood.
    pub fn usage(message: impl Into<String>) -> CliError {
        CliError::new(EXIT_USAGE, message).with_next("run `fastnn-infer help` for the command list")
    }

    /// The model file or its ONNX source is unusable.
    pub fn model(message: impl Into<String>) -> CliError {
        CliError::new(EXIT_MODEL, message)
    }

    /// The requested device is not available in this build or on this host.
    pub fn device(message: impl Into<String>) -> CliError {
        CliError::new(EXIT_DEVICE, message)
            .with_next("run `fastnn-infer diagnose` to list the devices this build can open")
    }

    /// Inference was refused by a bound, a capacity or a backend.
    pub fn inference(message: impl Into<String>) -> CliError {
        CliError::new(EXIT_INFERENCE, message)
    }

    /// Attach (or replace) the next-step line.
    pub fn with_next(mut self, next: impl Into<String>) -> CliError {
        self.next = Some(next.into());
        self
    }

    /// The process exit code.
    pub fn code(&self) -> i32 {
        self.code
    }

    /// The message without the `next:` line.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The next-step line, when one is set.
    pub fn next_step(&self) -> Option<&str> {
        self.next.as_deref()
    }

    /// Render the two-line error block.
    pub fn render(&self) -> String {
        match &self.next {
            Some(next) => format!("error: {}\nnext: {next}", self.message),
            None => format!("error: {}", self.message),
        }
    }
}

/// Run the CLI against explicit writers and return the process exit code.
///
/// `out` receives normal output, `err` receives the rendered error block. Write
/// failures are ignored: a closed pipe must not turn a successful inference into
/// a failure exit code.
pub fn run(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    match args::parse(args) {
        Ok(Command::Help(topic)) => {
            let _ = out.write_all(args::help_text(topic).as_bytes());
            EXIT_OK
        }
        Ok(Command::Version) => {
            let _ = writeln!(out, "fastnn-infer {}", env!("CARGO_PKG_VERSION"));
            EXIT_OK
        }
        Ok(command) => match dispatch(command, out) {
            Ok(()) => EXIT_OK,
            Err(error) => {
                let _ = writeln!(err, "{}", error.render());
                error.code()
            }
        },
        Err(error) => {
            let _ = writeln!(err, "{}", error.render());
            error.code()
        }
    }
}

fn dispatch(command: Command, out: &mut dyn Write) -> Result<(), CliError> {
    match command {
        Command::Prepare(args) => prepare::execute(&args, out),
        Command::Inspect(args) => inspect::execute(&args, out),
        Command::Run(args) => infer::execute_run(&args, out),
        Command::Generate(args) => infer::execute_generate(&args, out),
        Command::Diagnose(args) => devices::execute_diagnose(&args, out),
        Command::Help(_) | Command::Version => Ok(()),
    }
}

/// Group an integer with `_` every three digits for readable sizes.
pub(crate) fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut text = String::with_capacity(digits.len() + digits.len() / 3);
    for (position, character) in digits.chars().enumerate() {
        if position > 0 && (digits.len() - position).is_multiple_of(3) {
            text.push(',');
        }
        text.push(character);
    }
    text
}

/// The binary entry point: parse `std::env::args` and exit with the CLI code.
pub fn main_entry() -> ! {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    let code = run(&args, &mut stdout, &mut stderr);
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_render_two_lines_and_keep_the_code() {
        let error = CliError::device("requested device vulkan is unavailable")
            .with_next("run `fastnn-infer diagnose`");
        assert_eq!(error.code(), EXIT_DEVICE);
        assert_eq!(
            error.render(),
            "error: requested device vulkan is unavailable\nnext: run `fastnn-infer diagnose`"
        );
    }

    #[test]
    fn an_unknown_command_is_a_usage_error_with_a_next_step() {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let code = run(&["wat".to_string()], &mut out, &mut err);
        assert_eq!(code, EXIT_USAGE);
        let text = String::from_utf8(err).expect("utf-8");
        assert!(text.contains("unknown command `wat`"), "{text}");
        assert!(text.contains("next:"), "{text}");
    }

    #[test]
    fn help_and_version_succeed_without_touching_stderr() {
        for args in [vec!["help".to_string()], vec!["--version".to_string()]] {
            let mut out = Vec::new();
            let mut err = Vec::new();
            assert_eq!(run(&args, &mut out, &mut err), EXIT_OK);
            assert!(!out.is_empty());
            assert!(err.is_empty());
        }
    }
}
