# Contributing to fastnn

## Repository Setup

### Prerequisites

- Rust stable (see `rust-toolchain.toml` for the pinned version)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for Python environment management

### Building

```bash
# Python package (editable install)
uv pip install -e .

# Rust library with the default features
cargo build --lib

# Portable library with no features
cargo build --no-default-features

# Vulkan backend (opt-in; a build opens no device)
cargo build --lib --features vulkan

# Inference CLI (the `fastnn-infer` target requires the `cli` feature)
cargo build --release --no-default-features --features cli,vulkan --bin fastnn-infer
```

### Running Tests

```bash
# Rust unit tests
cargo test --lib

# fn3 contract and CI smoke suites
cargo test --test fn3_contracts
cargo test --test fn3_ci_smoke

# Vulkan fault-injection matrix (test-only feature)
cargo test --features vulkan,vulkan-fault-injection \
  --test fn3_vulkan_faults -- --test-threads=1

# Rust half of CI: fmt, version agreement, clippy with warnings denied,
# --no-default-features build and the default-feature Rust suites
tools/ci/cpu-lane.sh

# Python half of CI: maturin develop and pytest -m "not slow"
tools/ci/python-lane.sh
```

Physical-GPU runs are serialised under the measurement lock and must demand a
physical adapter; `FN3_VK_VALIDATION=1` turns the validation layer on for ad-hoc
runs. `tools/ci/run-gpu-lane.sh` is the only accepted physical-evidence path.

```bash
flock /tmp/opencode/fn3-measure.lock env \
  FN3_VK_REQUIRE_PHYSICAL=1 FN3_VK_VALIDATION=1 \
  FASTNN_FN3_ONNX_MODEL=/path/to/model.onnx \
  cargo test --features vulkan --test fn3_e2e -- --test-threads=1
```

## Pull Request Process

1. **Branch from `dev`** — All PRs target `dev`. The `main` branch is for releases only.
2. **Keep changes focused** — A PR should address one concern. Split large changes across multiple PRs.
3. **Include tests** — Behavior changes must include tests. Performance changes should include a reproducible benchmark command or telemetry guardrail.
4. **Run the local quality gate** before opening a PR (see above).
5. **Update documentation** if you change public APIs, add features, or modify compiler passes.

## Coding Standards

### General

- Use the pinned stable toolchain from `rust-toolchain.toml` so local runs match CI.
- Follow existing code style. Run `cargo fmt` before committing.
- Avoid `unsafe` where possible. When `unsafe` is necessary, document safety invariants with `// SAFETY:` comments.
- Prefer iterator combinators and functional style over raw loops where clarity is maintained.
- Use strong types rather than raw primitives for semantic distinctions.

### rustfmt Baseline

CI uses a file baseline at `ci/rustfmt-baseline.txt`. CI fails only when a newly touched file adds formatting debt outside the allowed list. When you fix an entry in the baseline, remove it from the file in the same PR.

### Clippy Baseline

CI uses a diagnostic baseline at `ci/clippy-baseline.txt`. CI fails on new warnings; existing debt stays tracked until cleanup lands. When you fix a clippy warning covered by the baseline, remove it from the file in the same PR.

### Documentation

- Keep the README concise. Move detailed API, architecture, and performance material into `docs/`.
- Every markdown file in `docs/` must link back to `docs/index.md` and to related documents.
- Code examples in documentation should be runnable or clearly marked as pseudocode.

### Commit Messages

- Use conventional commit format where possible: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `chore`
- Example: `feat(compiler): add fuse-matmul-silu pass`

## Feature Flags

fastnn uses Cargo feature flags. The default set is the inference path, and CI
builds explicit feature sets rather than `--all-features` (see
`.github/workflows/ci.yml`).

| Feature | Description |
|---------|-------------|
| `default` | simd + parallel + fusion-forward + python + prepared-plan |
| `python` | PyO3 bindings for `fastnn.inference` (default) |
| `prepared-plan` | Prepared-plan construction and the persistent executor path (default) |
| `vulkan` | Vulkan compute backend; opt-in because an enabled feature advertises a device |
| `vulkan-fault-injection` | Test-only deterministic fault injection; implies `vulkan` |
| `cli` | Enables the `fastnn-infer` binary target |
| `fn3` | Links the prepared-model adapter into `fn3-bench` |
| `simd` | SIMD microkernels (default) |
| `parallel` | Rayon-based parallelism (default) |
| `openblas` | OpenBLAS-backed GEMM |

The default `python` feature builds PyO3 as an extension module, so standalone
binaries (`fastnn-infer`, `fn3-bench`) are built with `--no-default-features`
plus the features they need.

## Getting Help

- Open a [GitHub Issue](https://github.com/PetrouilFan/fastnn/issues) for bugs, questions, or feature requests.
- See [`docs/internals/development.md`](docs/internals/development.md) for the full codebase walkthrough, architecture overview, and guides for adding ops, fusion passes, and packed precision types.
