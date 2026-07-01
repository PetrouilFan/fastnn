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

# Rust library only
cargo build

# Rust with all features
cargo build --all-features
```

### Running Tests

```bash
# Rust unit tests
cargo test --lib

# Quantized pipeline tests
cargo test --test quantized_pipeline

# Optimizer tests
cargo test --test optim_test

# Python tests
uv run pytest tests/ -v

# Full local PR gate (matches CI)
./scripts/ci/check-rustfmt-baseline.sh
python3 ./scripts/ci/check-clippy-baseline.py
cargo test --lib
cargo test --test quantized_pipeline
cargo test --test optim_test
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

fastnn uses Cargo feature flags extensively. Enable relevant features when building:

| Feature | Description |
|---------|-------------|
| `default` | simd + parallel + fusion-forward + python + prepared-plan |
| `python` | PyO3 Python bindings |
| `gpu` | Experimental WGPU backend |
| `openblas` | OpenBLAS-backed GEMM |
| `simd` | SIMD microkernels (default) |
| `parallel` | Rayon-based parallelism (default) |

Add `--features <feature>` or `--all-features` to cargo commands to test with your changes.

## Getting Help

- Open a [GitHub Issue](https://github.com/PetrouilFan/fastnn/issues) for bugs, questions, or feature requests.
- See [`docs/internals/development.md`](docs/internals/development.md) for the full codebase walkthrough, architecture overview, and guides for adding ops, fusion passes, and packed precision types.
