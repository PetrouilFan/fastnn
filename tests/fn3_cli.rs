//! FN3-17 (#63): the `fastnn-infer` command line on the pinned model.
//!
//! Most tests drive `fastnn::inference::cli::run` directly, with captured writers, so
//! they assert the same messages and exit codes a user sees. The pinned model is
//! not committed (no declared licence); every model-dependent test prints a
//! skip reason and passes when it is absent.
//!
//! ```bash
//! FASTNN_FN3_ONNX_MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx \
//!   CARGO_BUILD_JOBS=12 cargo test --test fn3_cli -- --nocapture --test-threads=1
//! ```

use std::path::{Path, PathBuf};

use fastnn::inference::cli::{run, EXIT_DEVICE, EXIT_INFERENCE, EXIT_MODEL, EXIT_OK, EXIT_USAGE};

#[cfg(feature = "cli")]
#[test]
fn fastnn_infer_binary_help() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_fastnn-infer"))
        .arg("--help")
        .output()
        .expect("launch fastnn-infer");
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
    let help = String::from_utf8(output.stdout).expect("UTF-8 help");
    assert!(help.contains("fastnn-infer <command> [options]"), "{help}");
}

#[cfg(feature = "cli")]
#[test]
fn fastnn_infer_binary_inspect_missing_file() {
    let directory = scratch("binary-inspect-missing");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_fastnn-infer"))
        .arg("inspect")
        .arg(directory.join("missing.fnn3"))
        .output();
    std::fs::remove_dir(&directory).expect("remove scratch directory");
    let output = output.expect("launch fastnn-infer inspect");
    assert_eq!(output.status.code(), Some(EXIT_MODEL), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 diagnostic");
    assert!(stderr.contains("fastnn-infer prepare"), "{stderr}");
    assert!(!stderr.contains("fn3-infer"), "{stderr}");
}

/// Bytes and FNV-1a fingerprint of the pinned graph, matching `fn3_session`.
const PINNED_BYTES: usize = 27_897_974;
const PINNED_FNV1A: u64 = 0xc294_d3ab_837f_b1be;

/// The 13-token prompt the committed fixtures use.
const PROMPT_13: &str = "464,2068,7586,21831,18045,625,262,16931,3290,13,4874,2402,257";
/// A 10-token prompt whose greedy next token is pinned by this test.
const PROMPT_10: &str = "464,2068,7586,21831,18045,625,262,16931,3290,13";

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn pinned() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("FASTNN_FN3_ONNX_MODEL") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tools/parity/.cache/models/tiny-gpt2/onnx/model.onnx"),
    );
    for path in candidates {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        assert_eq!(
            bytes.len(),
            PINNED_BYTES,
            "not the pinned graph: {}",
            path.display()
        );
        assert_eq!(
            fnv1a(&bytes),
            PINNED_FNV1A,
            "not the pinned graph: {}",
            path.display()
        );
        return Some(path);
    }
    println!(
        "SKIP: the pinned 27,897,974-byte ONNX model is not present. Set \
         FASTNN_FN3_ONNX_MODEL or run `python3 tools/parity/fetch_models.py --suite tiny-gpt2`."
    );
    None
}

struct Output {
    code: i32,
    stdout: String,
    stderr: String,
}

fn cli(args: &[&str]) -> Output {
    let argv: Vec<String> = args.iter().map(|arg| (*arg).to_string()).collect();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let code = run(&argv, &mut stdout, &mut stderr);
    Output {
        code,
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
    }
}

fn scratch(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("fastnn-fn3-cli-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).expect("scratch directory");
    directory
}

fn path_text(path: &Path) -> &str {
    path.to_str().expect("utf-8 temporary path")
}

#[test]
fn usage_errors_are_usage_exit_codes_with_a_next_step() {
    let output = cli(&["definitely-not-a-command"]);
    assert_eq!(output.code, EXIT_USAGE);
    assert!(output.stderr.contains("next:"), "{}", output.stderr);
    assert!(output.stdout.is_empty());

    // `--onnx` is optional now; the missing required flag is the prompt.
    let output = cli(&["run", "--model", "x"]);
    assert_eq!(output.code, EXIT_USAGE);
    assert!(output.stderr.contains("--tokens"), "{}", output.stderr);

    // `--verify-source` is an explicit request for the source check, so asking
    // for it without a source is a usage error, not a silent no-op.
    let output = cli(&["run", "--model", "x", "--verify-source", "--tokens", "1"]);
    assert_eq!(output.code, EXIT_USAGE);
    assert!(output.stderr.contains("--onnx"), "{}", output.stderr);
    assert!(
        output.stderr.contains("--verify-source"),
        "{}",
        output.stderr
    );
}

#[test]
fn inspect_reports_a_missing_file_as_a_model_error() {
    let directory = scratch("missing");
    let missing = directory.join("nope.fnn3");
    let output = cli(&["inspect", path_text(&missing)]);
    assert_eq!(output.code, EXIT_MODEL);
    assert!(
        output.stderr.contains("fastnn-infer prepare"),
        "{}",
        output.stderr
    );
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn inspect_names_a_future_format_version_and_a_foreign_magic() {
    let directory = scratch("headers");
    let future = directory.join("future.fnn3");
    let legacy = directory.join("legacy.fnn3");
    let mut header = vec![0u8; 64];
    header[..8].copy_from_slice(b"FNN3PREP");
    header[8..10].copy_from_slice(&2u16.to_le_bytes());
    header[10..12].copy_from_slice(&64u16.to_le_bytes());
    header[12..16].copy_from_slice(&1u32.to_le_bytes());
    header[24..32].copy_from_slice(&64u64.to_le_bytes());
    std::fs::write(&future, &header).expect("write future header");

    header[..8].copy_from_slice(b"FNN2PREP");
    header[8..10].copy_from_slice(&1u16.to_le_bytes());
    std::fs::write(&legacy, &header).expect("write legacy header");

    let output = cli(&["inspect", path_text(&future)]);
    assert_eq!(output.code, EXIT_MODEL);
    assert!(output.stderr.contains("version 2"), "{}", output.stderr);
    assert!(output.stderr.contains("next:"), "{}", output.stderr);

    let output = cli(&["inspect", path_text(&legacy)]);
    assert_eq!(output.code, EXIT_MODEL);
    assert!(output.stderr.contains("FNN3PREP"), "{}", output.stderr);
    assert!(output.stderr.contains(".fnnc"), "{}", output.stderr);
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn an_explicit_vulkan_request_is_never_turned_into_cpu() {
    if cfg!(feature = "vulkan") {
        println!("SKIP: this test asserts the build-without-vulkan message");
        return;
    }
    let directory = scratch("vulkan-request");
    let missing = directory.join("missing.fnn3");
    let output = cli(&[
        "run",
        "--model",
        path_text(&missing),
        "--onnx",
        path_text(&missing),
        "--tokens",
        "1",
        "--device",
        "vulkan",
    ]);
    // The device is resolved before any file is read, so this is a device
    // error even though both paths are missing.
    assert_eq!(output.code, EXIT_DEVICE, "{}", output.stderr);
    assert!(output.stderr.contains("not silently"), "{}", output.stderr);
    assert!(
        output.stderr.contains("--features cli,vulkan"),
        "{}",
        output.stderr
    );
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn diagnose_prints_the_matrix_and_the_enumerated_opened_distinction() {
    let output = cli(&["diagnose"]);
    assert_eq!(output.code, EXIT_OK, "{}", output.stderr);
    assert!(output.stdout.contains("cpu"), "{}", output.stdout);
    assert!(output.stdout.contains("enumerated"), "{}", output.stdout);
    assert!(output.stdout.contains("ran the model"), "{}", output.stdout);
}

#[test]
fn the_pinned_model_prepares_inspects_runs_and_generates_on_cpu() {
    let Some(onnx) = pinned() else {
        return;
    };
    let directory = scratch("pinned");
    let artifact = directory.join("tiny-gpt2.fnn3");

    let output = cli(&[
        "prepare",
        "--onnx",
        path_text(&onnx),
        "--out",
        path_text(&artifact),
    ]);
    assert_eq!(output.code, EXIT_OK, "{}", output.stderr);
    assert!(
        output
            .stdout
            .contains("e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1"),
        "{}",
        output.stdout
    );
    assert!(output.stdout.contains("verified"), "{}", output.stdout);
    assert!(artifact.is_file());

    let output = cli(&["inspect", path_text(&artifact)]);
    assert_eq!(output.code, EXIT_OK, "{}", output.stderr);
    assert!(output.stdout.contains("framing valid"), "{}", output.stdout);

    // A different ONNX with one weight byte changed must be refused, naming
    // the weight. This pins the "artifact and --onnx are the same model" check.
    let mut flipped = std::fs::read(&onnx).expect("read pinned ONNX");
    let offset = 27_000_000usize;
    flipped[offset] ^= 1;
    let flipped_path = directory.join("flipped.onnx");
    std::fs::write(&flipped_path, &flipped).expect("write flipped ONNX");
    let output = cli(&[
        "run",
        "--model",
        path_text(&artifact),
        "--onnx",
        path_text(&flipped_path),
        "--tokens",
        "464",
    ]);
    assert_eq!(output.code, EXIT_MODEL, "{}", output.stderr);
    assert!(
        output.stderr.contains("do not describe the same model"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("weight `"), "{}", output.stderr);
    let _ = std::fs::remove_file(&flipped_path);

    let output = cli(&["inspect", "--sections", path_text(&artifact)]);
    assert_eq!(output.code, EXIT_OK, "{}", output.stderr);
    assert!(output.stdout.contains("layers=2"), "{}", output.stdout);
    assert!(output.stdout.contains("vocab=50,257"), "{}", output.stdout);
    assert!(
        output.stdout.contains("payload checksum: verified"),
        "{}",
        output.stdout
    );

    let output = cli(&[
        "run",
        "--model",
        path_text(&artifact),
        "--onnx",
        path_text(&onnx),
        "--tokens",
        PROMPT_10,
        "--device",
        "cpu",
        "--top-k",
        "3",
    ]);
    assert_eq!(output.code, EXIT_OK, "{}", output.stderr);
    assert!(output.stdout.contains("backend cpu"), "{}", output.stdout);
    assert!(
        output.stdout.contains("shape=[1, 1, 50257]"),
        "{}",
        output.stdout
    );
    assert!(
        output.stdout.contains("predicted next token: 46737"),
        "{}",
        output.stdout
    );
    assert!(!output.stdout.contains("vulkan"), "{}", output.stdout);

    let output = cli(&[
        "generate",
        "--model",
        path_text(&artifact),
        "--onnx",
        path_text(&onnx),
        "--tokens",
        PROMPT_10,
        "--max-new-tokens",
        "2",
        "--device",
        "cpu",
        "--top-k",
        "1",
    ]);
    assert_eq!(output.code, EXIT_OK, "{}", output.stderr);
    assert!(
        output
            .stdout
            .contains("generated token IDs: [46737, 46737]"),
        "{}",
        output.stdout
    );

    // The 13-token prompt decodes through the cached path; the fixture sequence
    // is the same one `tests/fn3_session.rs` checks against ORT.
    let output = cli(&[
        "run",
        "--model",
        path_text(&artifact),
        "--onnx",
        path_text(&onnx),
        "--tokens",
        PROMPT_13,
        "--decode",
        "640,11,612",
        "--device",
        "cpu",
        "--top-k",
        "1",
    ]);
    assert_eq!(output.code, EXIT_OK, "{}", output.stderr);
    assert!(
        output.stdout.contains("prefill: seq=13"),
        "{}",
        output.stdout
    );
    assert!(
        output.stdout.contains("decode 3: token=612"),
        "{}",
        output.stdout
    );
    assert!(
        output.stdout.contains("predicted next token: 612"),
        "{}",
        output.stdout
    );

    // Bounds are refused before any step runs, with the flag that fixes them.
    let output = cli(&[
        "run",
        "--model",
        path_text(&artifact),
        "--onnx",
        path_text(&onnx),
        "--tokens",
        PROMPT_10,
        "--capacity",
        "5",
    ]);
    assert_eq!(output.code, EXIT_INFERENCE);
    assert!(output.stderr.contains("--capacity 10"), "{}", output.stderr);

    let long_prompt = (0..129).map(|_| "464").collect::<Vec<_>>().join(",");
    let output = cli(&[
        "run",
        "--model",
        path_text(&artifact),
        "--onnx",
        path_text(&onnx),
        "--tokens",
        &long_prompt,
    ]);
    assert_eq!(output.code, EXIT_INFERENCE);
    assert!(
        output.stderr.contains("seq envelope maximum 128"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("--max-seq"), "{}", output.stderr);

    let _ = std::fs::remove_dir_all(&directory);
}

/// Token IDs of the top-k line, in order.
fn top_ids(stdout: &str, heading: &str, k: usize) -> Vec<usize> {
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with(heading))
        .unwrap_or_else(|| panic!("no `{heading}` line in:\n{stdout}"));
    let rest = line
        .split_once(':')
        .unwrap_or_else(|| panic!("no colon in `{line}`"))
        .1;
    let ids: Vec<usize> = rest
        .split(',')
        .map(|item| {
            item.split_whitespace()
                .next()
                .and_then(|id| id.parse().ok())
                .unwrap_or_else(|| panic!("no token ID in `{item}`"))
        })
        .collect();
    assert_eq!(ids.len(), k, "`{line}` does not hold {k} tokens");
    ids
}

#[test]
fn the_artifact_runs_alone_with_its_onnx_moved_away() {
    let Some(onnx) = pinned() else {
        return;
    };
    let directory = scratch("artifact-only");
    // Prepare from a copy, so moving it away does not disturb the pinned model
    // the other tests and workers read.
    let source = directory.join("source.onnx");
    std::fs::copy(&onnx, &source).expect("copy the pinned ONNX");
    let artifact = directory.join("tiny-gpt2.fnn3");

    let output = cli(&[
        "prepare",
        "--onnx",
        path_text(&source),
        "--out",
        path_text(&artifact),
    ]);
    assert_eq!(output.code, EXIT_OK, "{}", output.stderr);
    assert!(artifact.is_file());

    // The two-file path first, for the reference answer.
    let two_file = cli(&[
        "run",
        "--model",
        path_text(&artifact),
        "--onnx",
        path_text(&source),
        "--tokens",
        PROMPT_10,
        "--device",
        "cpu",
        "--top-k",
        "5",
    ]);
    assert_eq!(two_file.code, EXIT_OK, "{}", two_file.stderr);
    let reference = top_ids(&two_file.stdout, "top-5:", 5);
    assert_eq!(reference, vec![46_737, 28_597, 16_571, 7_868, 13]);

    // Now the source is gone: renamed out of the path `prepare` used.
    let moved = directory.join("source.onnx.moved-away");
    std::fs::rename(&source, &moved).expect("move the ONNX away");
    assert!(!source.exists(), "the source ONNX is still at its path");

    let artifact_only = cli(&[
        "run",
        "--model",
        path_text(&artifact),
        "--tokens",
        PROMPT_10,
        "--device",
        "cpu",
        "--top-k",
        "5",
    ]);
    assert_eq!(artifact_only.code, EXIT_OK, "{}", artifact_only.stderr);
    assert!(
        artifact_only.stdout.contains("source: none (artifact-only"),
        "{}",
        artifact_only.stdout
    );
    assert!(
        !artifact_only.stdout.contains("source.onnx"),
        "the artifact-only run still names an ONNX path:\n{}",
        artifact_only.stdout
    );
    assert_eq!(
        top_ids(&artifact_only.stdout, "top-5:", 5),
        reference,
        "artifact-only top-5 differs from the two-file path:\n{}",
        artifact_only.stdout
    );
    assert!(
        artifact_only.stdout.contains("argmax: 46737"),
        "{}",
        artifact_only.stdout
    );
    assert!(!artifact_only.stdout.contains("weight-int8:"));

    let packed = cli(&[
        "run",
        "--model",
        path_text(&artifact),
        "--tokens",
        PROMPT_10,
        "--decode",
        "4874,2402",
        "--device",
        "cpu",
        "--weight-int8",
        "--steps",
    ]);
    assert_eq!(packed.code, EXIT_OK, "{}", packed.stderr);
    assert_eq!(
        packed
            .stdout
            .matches("packed-matmuls=8/13 fp32-matmuls=5")
            .count(),
        3,
        "{}",
        packed.stdout
    );
    assert!(
        packed.stdout.contains(": computed-rhs"),
        "{}",
        packed.stdout
    );
    assert!(
        packed.stdout.contains("fp32-retained="),
        "on the CPU the dense figure is retained FP32 bytes, not an upload:\n{}",
        packed.stdout
    );
    assert!(
        packed.stdout.contains("packed-source="),
        "the report states what the packed bytes replaced:\n{}",
        packed.stdout
    );
    for token in [46737, 4874, 2402] {
        assert!(
            packed.stdout.contains(&format!("argmax: {token} (")),
            "{}",
            packed.stdout
        );
    }

    // Generation is artifact-only too, and reproduces the documented IDs.
    let generated = cli(&[
        "generate",
        "--model",
        path_text(&artifact),
        "--tokens",
        PROMPT_10,
        "--max-new-tokens",
        "2",
        "--device",
        "cpu",
        "--top-k",
        "1",
    ]);
    assert_eq!(generated.code, EXIT_OK, "{}", generated.stderr);
    assert!(
        generated
            .stdout
            .contains("generated token IDs: [46737, 46737]"),
        "{}",
        generated.stdout
    );

    // `--verify-source` with `--onnx` still proves the pair; the moved file is
    // gone, so this is the one case where the check must fail loudly.
    let missing = cli(&[
        "run",
        "--model",
        path_text(&artifact),
        "--onnx",
        path_text(&source),
        "--verify-source",
        "--tokens",
        PROMPT_10,
    ]);
    assert_eq!(missing.code, EXIT_MODEL, "{}", missing.stderr);
    assert!(
        missing.stderr.contains("cannot read ONNX model"),
        "{}",
        missing.stderr
    );

    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn a_truncated_artifact_fails_as_a_model_error_with_no_onnx_present() {
    let Some(onnx) = pinned() else {
        return;
    };
    let directory = scratch("artifact-only-truncated");
    let source = directory.join("source.onnx");
    std::fs::copy(&onnx, &source).expect("copy the pinned ONNX");
    let artifact = directory.join("tiny-gpt2.fnn3");
    let output = cli(&[
        "prepare",
        "--onnx",
        path_text(&source),
        "--out",
        path_text(&artifact),
    ]);
    assert_eq!(output.code, EXIT_OK, "{}", output.stderr);

    let bytes = std::fs::read(&artifact).expect("read the artifact");
    let truncated = directory.join("truncated.fnn3");
    std::fs::write(&truncated, &bytes[..100_000]).expect("write a truncated artifact");
    std::fs::remove_file(&source).expect("remove the source ONNX");
    assert!(!source.exists());

    let output = cli(&[
        "run",
        "--model",
        path_text(&truncated),
        "--tokens",
        PROMPT_10,
    ]);
    assert_eq!(output.code, EXIT_MODEL, "{}", output.stderr);
    assert!(
        output.stderr.contains("is not a usable prepared model"),
        "{}",
        output.stderr
    );
    assert!(
        output.stderr.contains("truncated") || output.stderr.contains("declares"),
        "{}",
        output.stderr
    );
    assert!(output.stderr.contains("next:"), "{}", output.stderr);
    assert!(output.stdout.is_empty(), "{}", output.stdout);

    let _ = std::fs::remove_dir_all(&directory);
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_either_executes_or_reports_the_known_blocker() {
    let Some(onnx) = pinned() else {
        return;
    };
    let directory = scratch("vulkan-run");
    let artifact = directory.join("tiny-gpt2.fnn3");
    let prepared = cli(&[
        "prepare",
        "--onnx",
        path_text(&onnx),
        "--out",
        path_text(&artifact),
    ]);
    assert_eq!(prepared.code, EXIT_OK, "{}", prepared.stderr);

    let output = cli(&[
        "run",
        "--model",
        path_text(&artifact),
        "--onnx",
        path_text(&onnx),
        "--tokens",
        PROMPT_10,
        "--device",
        "vulkan",
        "--top-k",
        "1",
    ]);
    if output.code == EXIT_OK {
        assert!(
            output.stdout.contains("backend vulkan"),
            "{}",
            output.stdout
        );
        assert!(
            output.stdout.contains("predicted next token:"),
            "{}",
            output.stdout
        );
        println!("Vulkan prefill actually ran:\n{}", output.stdout);
    } else {
        assert_eq!(output.code, EXIT_INFERENCE, "{}", output.stderr);
        assert!(output.stderr.contains("Vulkan"), "{}", output.stderr);
        assert!(output.stderr.contains("--device cpu"), "{}", output.stderr);
        println!(
            "Vulkan failed with the documented blocker:\n{}",
            output.stderr
        );
    }
    let _ = std::fs::remove_dir_all(&directory);
}
