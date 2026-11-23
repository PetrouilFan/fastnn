//! The v3 `fastnn-infer` surface: prepare, load, run, inspect and doctor.
//!
//! The CLI is driven through [`fastnn::inference::cli::run`] so these tests
//! exercise the same parsing, messages and exit codes a process would. The
//! pinned YOLO ONNX is external; tests that need it print why and return.

use std::path::PathBuf;

use fastnn::inference::cli::{run, EXIT_MODEL, EXIT_OK, EXIT_USAGE};
use fastnn::inference::model::format::Artifact;
use fastnn::inference::runtime::Threading;

const YOLOV8N: &str = "/home/petrouil/.hermes/yolov8n.onnx";

struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Scratch {
        let path =
            std::env::temp_dir().join(format!("fastnn-fn3-cli-v2-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch");
        Scratch { path }
    }

    fn file(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn invoke(args: &[&str]) -> (i32, String, String) {
    let argv: Vec<String> = args.iter().map(|arg| (*arg).to_string()).collect();
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = run(&argv, &mut out, &mut err);
    (
        code,
        String::from_utf8_lossy(&out).into_owned(),
        String::from_utf8_lossy(&err).into_owned(),
    )
}

fn prepared_yolo(scratch: &Scratch) -> Option<PathBuf> {
    if !std::path::Path::new(YOLOV8N).exists() {
        eprintln!("{YOLOV8N} is not present; skipping the CLI prepare test");
        return None;
    }
    let path = scratch.file("yolov8n.fnn3");
    let (code, out, err) = invoke(&[
        "prepare",
        "--onnx",
        YOLOV8N,
        "--out",
        path.to_str().expect("utf-8 path"),
        "--target",
        "cpu",
    ]);
    assert_eq!(code, EXIT_OK, "stdout:\n{out}\nstderr:\n{err}");
    assert!(out.contains("magic FNN3ART2"), "{out}");
    assert!(out.contains("verified:"), "{out}");
    Some(path)
}

#[test]
fn help_and_version_succeed_without_stderr() {
    for args in [vec!["help"], vec!["--version"], vec!["help", "prepare"]] {
        let (code, out, err) = invoke(&args);
        assert_eq!(code, EXIT_OK);
        assert!(!out.is_empty());
        assert!(err.is_empty(), "{err}");
    }
}

#[test]
fn an_unknown_command_is_a_usage_error_with_a_next_step() {
    let (code, _, err) = invoke(&["wat"]);
    assert_eq!(code, EXIT_USAGE);
    assert!(err.contains("unknown command `wat`"), "{err}");
    assert!(err.contains("next:"), "{err}");
}

#[test]
fn a_missing_artifact_names_the_path_and_the_prepare_command() {
    let scratch = Scratch::new("missing");
    let path = scratch.file("missing.fnn3");
    let (code, _, err) = invoke(&["inspect", path.to_str().expect("path")]);
    assert_eq!(code, EXIT_MODEL);
    assert!(err.contains("missing.fnn3"), "{err}");
    assert!(err.contains("next:"), "{err}");
}

#[test]
fn inspect_load_and_run_agree_with_the_rust_api() {
    let scratch = Scratch::new("roundtrip");
    let Some(path) = prepared_yolo(&scratch) else {
        return;
    };
    let path = path.to_str().expect("path").to_string();

    let (code, out, err) = invoke(&["inspect", &path]);
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(out.contains("mapped bytes 0"), "{out}");
    assert!(out.contains("WEIGHTS"), "{out}");
    assert!(out.contains("stored (payload not read)"), "{out}");

    let (code, out, err) = invoke(&["load", &path, "--threads", "2"]);
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(out.contains("executed: nothing"), "{out}");
    assert!(out.contains("workers=2"), "{out}");

    let (code, out, err) = invoke(&["run", &path, "--input", "images=zeros", "--threads", "2"]);
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(out.contains("output output0"), "{out}");
    assert!(out.contains("shape [1x84x2100]"), "{out}");
    assert!(out.contains("finite 176400/176400"), "{out}");

    // The CLI and the Rust API see the same artifact.
    let artifact = Artifact::open(&path).expect("open");
    let session = fastnn::inference::load(&path, Threading::Fixed(1)).expect("load");
    assert_eq!(session.plan(), artifact.default_plan());
}

#[test]
fn run_requires_every_graph_input_by_name() {
    let scratch = Scratch::new("inputs");
    let Some(path) = prepared_yolo(&scratch) else {
        return;
    };
    let path = path.to_str().expect("path").to_string();
    let (code, _, err) = invoke(&["run", &path]);
    assert_eq!(code, EXIT_USAGE);
    assert!(err.contains("images"), "{err}");
    let (code, _, err) = invoke(&["run", &path, "--input", "wrong=zeros"]);
    assert_eq!(code, EXIT_USAGE);
    assert!(err.contains("no input named `wrong`"), "{err}");
    let (code, _, err) = invoke(&["run", &path, "--input", "images=unknown"]);
    assert_eq!(code, EXIT_MODEL);
    assert!(err.contains("cannot read input"), "{err}");
}

#[test]
fn doctor_reports_a_ready_model_and_its_memory() {
    if !std::path::Path::new(YOLOV8N).exists() {
        eprintln!("{YOLOV8N} is not present; skipping the CLI doctor test");
        return;
    }
    let (code, out, err) = invoke(&["doctor", YOLOV8N]);
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(out.contains("ready to prepare"), "{out}");
    assert!(
        out.contains("expected resident") || out.contains("memory:"),
        "{out}"
    );
    assert!(out.contains("problems: none"), "{out}");
}

#[test]
fn all_targets_are_selectable_from_the_cli() {
    if !std::path::Path::new(YOLOV8N).exists() {
        return;
    }
    let scratch = Scratch::new("targets");
    let path = scratch.file("all.fnn3");
    let (code, out, err) = invoke(&[
        "prepare",
        "--onnx",
        YOLOV8N,
        "--out",
        path.to_str().expect("path"),
        "--target",
        "all",
    ]);
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(out.contains("cpu-reference"), "{out}");
    let path = path.to_str().expect("path").to_string();
    let (code, out, err) = invoke(&["load", &path, "--target", "cpu-reference"]);
    assert_eq!(code, EXIT_OK, "{err}");
    assert!(out.contains("target: cpu-reference"), "{out}");
    let (code, _, err) = invoke(&["load", &path, "--target", "gpu"]);
    assert_eq!(code, EXIT_MODEL);
    assert!(err.contains("no target `gpu`"), "{err}");
}

#[test]
fn a_version_one_or_foreign_file_is_refused_with_a_next_step() {
    let scratch = Scratch::new("versions");
    let path = scratch.file("old.fnn3");
    let mut bytes = vec![0u8; 256];
    bytes[..8].copy_from_slice(b"FNN3PREP");
    bytes[8..10].copy_from_slice(&1u16.to_le_bytes());
    std::fs::write(&path, &bytes).expect("write");
    let (code, _, err) = invoke(&["inspect", path.to_str().expect("path")]);
    assert_eq!(code, EXIT_MODEL);
    assert!(err.contains("Re-prepare"), "{err}");
    assert!(err.contains("next:"), "{err}");

    let path = scratch.file("foreign.fnn3");
    std::fs::write(&path, b"not an artifact at all").expect("write");
    let (code, _, err) = invoke(&["inspect", path.to_str().expect("path")]);
    assert_eq!(code, EXIT_MODEL);
    assert!(err.contains("next:"), "{err}");
}
