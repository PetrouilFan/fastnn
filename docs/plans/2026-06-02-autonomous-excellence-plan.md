# fastnn Autonomous Excellence Plan

> **For Hermes:** Execute this plan autonomously every 2 hours with strict TDD, isolated worktrees, low-noise reporting, and CI verification.

**Goal:** Improve fastnn systematically across correctness, performance, memory efficiency, API quality, documentation, CI, maintainability, packaging, and backend maturity without requiring user input while the user is away.

**Operating model:** Every 2-hour run performs preflight, selects exactly one high-leverage lane, creates or reuses an isolated worktree, writes a failing guardrail/test first where code behavior changes, implements the smallest safe change, verifies locally, commits, merges to `main`, pushes, and watches CI. If no safe implementation lane is available, the run performs audit/planning/documentation cleanup instead of inventing risky changes.

**Tech stack:** Rust, PyO3, Python 3.12+, maturin, Criterion, GitHub Actions, cargo, uv, WGPU optional backend.

---

## Non-negotiable operating rules

1. **No user input required.** If a decision is ambiguous, choose the safer default: smaller patch, stronger tests, no risky API break, no unverified performance claim.
2. **Canonical repo only:** `/home/petrouil/Projects/github/fastnn`.
3. **Do not modify `.hermes/` inside the repo.** Leave it untracked.
4. **Use isolated worktrees** for implementation lanes under `/tmp/fastnn-autonomous-lanes/`.
5. **Use `/dev/shm` target dirs** for builds/tests when possible to reduce disk pressure.
6. **Never claim performance wins without a reproducible command or telemetry guardrail.**
7. **TDD for behavior/performance changes:** RED guardrail first, then implementation, then GREEN verification.
8. **Safety first for memory/copy optimizations:** preserve fallback paths for overlap, aliasing, broadcast, dtype mismatch, and runtime shape uncertainty.
9. **One merge at a time:** serialize changes to `main`; never push multiple unreviewed branches simultaneously.
10. **Keep output low-noise:** report only meaningful progress, failures, CI status, and next action. If a run has no meaningful new state, it should be silent or minimal.
11. **No history rewriting:** normal commits only, no force-push/amend after push unless explicitly requested.
12. **No broad rewrites without proof:** prefer small, reversible, independently testable improvements.

---

## Two-hour run loop

Each scheduled run follows this exact lifecycle.

### 1. Preflight

Run from `/home/petrouil/Projects/github/fastnn`:

```bash
git fetch origin --prune
git checkout main
git pull --ff-only origin main
git status --short --branch
gh run list --branch main --limit 8
```

Check:

- `main` is synced with `origin/main`.
- There are no tracked local changes.
- `.hermes/` may be untracked and must not be staged.
- Latest CI state is known.
- No active cargo/rustc process is already operating on the same worktree.

If CI on `main` is failing, priority becomes **CI rescue** before new work.

### 2. Lane selection

Pick exactly one lane per run using this priority order:

1. Fix red CI on `main`.
2. Fix correctness bugs or test failures discovered by previous runs.
3. Add missing guardrails for memory/copy/performance-sensitive code.
4. Implement small safe optimization with telemetry or benchmark coverage.
5. Remove warning/lint debt accepted by baseline.
6. Improve docs truthfulness and navigability.
7. Improve CI/release/tooling reliability.
8. Perform an audit and write a plan if no safe patch is available.

### 3. Worktree setup

Create a descriptive branch and worktree:

```bash
mkdir -p /tmp/fastnn-autonomous-lanes
BRANCH="agent/<lane-name>"
WT="/tmp/fastnn-autonomous-lanes/<lane-name>"
git worktree add -b "$BRANCH" "$WT" main
```

Use lane-specific build dirs:

```bash
export CARGO_TARGET_DIR="/dev/shm/fastnn-<lane-name>-target"
```

### 4. RED guardrail

For code/performance changes, write a focused failing test or telemetry assertion first.

Examples:

- CPU copy reduction: assert temp copy count and numerical output.
- Memory planner reuse: assert arena size/reuse relationship and output correctness.
- Optimizer update improvements: assert output write counts or state synchronization plus numerical update.
- API/error-path cleanup: assert public error behavior.
- Docs-only changes: use README/doc guardrails, link checks, and verified snippets instead of code tests.

Run the focused check and confirm it fails for the intended reason.

### 5. Implement minimal change

Implement only the smallest change that satisfies the guardrail. Keep risky cases on existing fallback paths unless the test covers them.

### 6. GREEN verification

Run the nearest focused checks, then broader checks appropriate to the change:

```bash
cargo fmt --check
cargo test --lib
cargo test --test <relevant-test>
uv run pytest tests/ -q
```

Do not run every expensive benchmark by default. Run Criterion only when changing benchmarked paths or adding benchmark guardrails.

### 7. Diff review

Before commit:

```bash
git diff --check
git diff --stat
git diff
```

Verify:

- No `.hermes/`, `target/`, `__pycache__/`, `.so`, or generated scratch artifacts were staged.
- Tests and code changes match the lane objective.
- No unverified performance claims were added.

### 8. Commit, merge, push

```bash
git add <intended-files>
git commit -m "<type>: <short lane summary>"
cd /home/petrouil/Projects/github/fastnn
git checkout main
git pull --ff-only origin main
git merge --no-ff "$BRANCH" -m "Merge <lane summary>"
git push origin main
```

### 9. Watch CI

```bash
gh run list --branch main --limit 4
gh run watch <new-run-id> --exit-status
```

If CI fails, diagnose and fix immediately. Do not leave `main` red if the failure was introduced by the run.

### 10. Cleanup

After CI is green:

```bash
git worktree remove "$WT" || true
git branch -d "$BRANCH" || true
```

Prune stale worktrees periodically:

```bash
git worktree prune
```

---

## Improvement lanes

### Lane A: CI rescue and workflow reliability

Purpose: keep `main` green and reduce future automation friction.

Targets:

- `.github/workflows/ci.yml`
- `.github/workflows/coverage.yml`
- `.github/workflows/release.yml`
- `scripts/ci/check-clippy-baseline.py`
- release and benchmark scripts

Tasks:

1. Inspect latest failed run before changing anything.
2. Reproduce failure locally or with the closest CI command.
3. Fix one failure class at a time.
4. Keep clippy baseline strict by fixing new lints rather than expanding baseline where practical.
5. Address Node.js action deprecation when safe by upgrading actions or setting a tested compatibility path.
6. Ensure duplicate workflows do not diverge on equivalent coverage/quality commands.
7. Commit and push only after local reproduction passes.

Acceptance:

- `main` CI and Coverage green.
- No new warning class added to CI output.
- Workflow changes are minimal and portable.

### Lane B: CPU copy reduction and arena execution

Purpose: reduce avoidable temporary copies and allocations in CPU execution without weakening alias safety.

Targets:

- `src/backend/cpu/arena.rs`
- `src/backend/cpu/elementwise.rs`
- `src/backend/cpu/scalar.rs`
- `src/backend/cpu/matmul.rs`
- `src/backend/cpu/reductions.rs` if present
- `src/backend/cpu/mod.rs`
- `tests/cpu_arena_telemetry.rs`
- `benches/cpu_baselines.rs`

Tasks:

1. Add telemetry guardrails for one operation family at a time.
2. Assert both numerical correctness and temp-copy metrics.
3. Preserve overlap and in-place fallbacks.
4. Add benchmark entrypoints only for maintained suites.
5. Remove obsolete helpers after final call site migration.

Candidate improvements:

- More zero-copy unary same-shape dispatch where disjoint.
- More zero-copy binary same-shape dispatch where one operand overlaps but the other does not.
- Reduction output reuse when input/output ranges are provably disjoint.
- Matmul output direct-write paths where shape/dtype/backing storage are known.
- Scalar op dispatch consolidation to avoid repeated copy helpers.

Acceptance:

- Focused telemetry tests pass.
- `cargo test --lib` and relevant integration tests pass.
- Benchmarks expose the path if it is performance-sensitive.

### Lane C: Memory planning and arena reuse

Purpose: make compiled graph memory planning more compact, predictable, and safe.

Targets:

- `src/compiler/passes/*memory*` if present
- `src/backend/executor.rs`
- `src/backend/runtime.rs`
- `src/ir/builder.rs`
- `tests/memory_planning_tests.rs`
- `tests/compiler_edge_cases.rs`

Tasks:

1. Audit live-range computation for conservative over-retention.
2. Add tests for disjoint lifetimes reusing arena slots.
3. Add tests for dynamic shape tightening preserving safety.
4. Detect and prevent overlap reuse for aliased/in-place-sensitive paths.
5. Add or improve memory-plan debug/telemetry output if missing.

Candidate improvements:

- Better first-fit slot reuse ordering.
- Runtime shape tightening guardrails.
- Distinguish persistent parameter/state buffers from ephemeral activations.
- Stable memory-plan summaries for regression tests.

Acceptance:

- Arena size decreases or telemetry becomes more precise for a tested graph.
- No correctness regressions under dynamic shape tests.

### Lane D: Compiler passes and IR correctness

Purpose: improve graph transformation reliability and reduce edge-case failures.

Targets:

- `src/compiler/passes/`
- `src/ir/builder.rs`
- `src/ir/node.rs`
- `tests/compiler_edge_cases.rs`
- `tests/shape_inference_tests.rs`

Tasks:

1. Add edge-case tests for one pass at a time.
2. Verify idempotence for transformation passes.
3. Verify pass ordering invariants.
4. Add negative tests for invalid shapes/dtypes.
5. Tighten public error messages without changing expected successful behavior.

Candidate improvements:

- Auto-cast edge cases.
- Quantize/dequantize pruning idempotence.
- Fusion pass safety with dynamic shapes.
- Dead-code elimination preserving required outputs.
- Shape assertion clarity.

Acceptance:

- New edge-case test fails before implementation and passes after.
- Pass remains idempotent where expected.

### Lane E: Optimizers and compiled training

Purpose: improve optimizer correctness, host/device synchronization, and update efficiency.

Targets:

- `src/optim/*.rs`
- `src/python/optim.rs`
- `src/python/trainer.rs`
- `src/ir/builder.rs`
- `tests/optim_test.rs`
- `benches/cpu_baselines.rs`

Tasks:

1. Add numerical regression tests for optimizer state transitions.
2. Add guardrails for state synchronization after compiled/eager updates.
3. Reduce output writes or allocations only with telemetry.
4. Keep public optimizer APIs stable unless tests show a bug.
5. Compare with simple reference formulas where feasible.

Candidate improvements:

- Fused update write-count reduction.
- Adam/AdamW state dtype consistency.
- Weight decay ordering verification.
- Python optimizer API smoke tests.

Acceptance:

- Optimizer tests pass locally and in CI.
- Numerical tests document expected update behavior.

### Lane F: Python API, packaging, and bindings

Purpose: make the Python package cleaner, more stable, and easier to use.

Targets:

- `fastnn/__init__.py`
- `fastnn/tensor.py`
- `fastnn/nn.py`
- `fastnn/optimizers.py`
- `fastnn/io/*.py`
- `src/python/*.rs`
- `pyproject.toml`
- `tests/test_*.py`

Tasks:

1. Verify public exports against docs.
2. Add smoke tests for public imports and common examples.
3. Remove stale docstrings/marketing wording from Python modules.
4. Improve error behavior for invalid arguments.
5. Ensure generated/native artifacts are not accidentally tracked.

Candidate improvements:

- Public API consistency tests.
- Better `__all__` hygiene.
- Verified quick-start snippets.
- Python packaging metadata cleanup.
- Import-time warning reduction.

Acceptance:

- `uv run pytest tests/ -q` passes.
- README/getting-started snippets are runnable.

### Lane G: WGPU backend maturity

Purpose: make optional WGPU code paths more testable, documented, and less fragile.

Targets:

- WGPU backend files under `src/backend/` and `src/macros/wgpu.rs`
- `benches/wgpu_*.rs`
- `docs/python-api.md`
- `docs/architecture.md`
- `BENCHMARKS.md`

Tasks:

1. Separate GPU-available tests from CPU-only CI-safe checks.
2. Add feature-gated compile checks where possible.
3. Document WGPU limitations clearly.
4. Avoid performance claims unless manually benchmarked on named hardware.
5. Improve shader/cache/pooling code only with compile/test guardrails.

Acceptance:

- CPU CI remains green without GPU hardware.
- WGPU docs distinguish maintained vs manual/hardware-dependent paths.

### Lane H: Documentation professionalism

Purpose: make docs precise, navigable, and non-promotional.

Targets:

- `README.md`
- `docs/index.md`
- `docs/getting-started.md`
- `docs/tensors.md`
- `docs/training.md`
- `docs/python-api.md`
- `docs/architecture.md`
- `docs/development.md`
- `docs/onnx.md`
- `BENCHMARKS.md`

Tasks:

1. Audit one doc per run for stale/promotional claims.
2. Add status/limitations sections where missing.
3. Keep README concise; move details into topic docs.
4. Verify all local links.
5. Run retained snippets where feasible.
6. Add a simple docs link checker script if absent.

Claim policy:

- Verified from source/tests.
- Linked to detailed docs.
- Qualified as experimental/currently supported for selected paths.
- Or removed.

Acceptance:

- Local links resolve.
- No unqualified performance claims.
- Docs are shorter, more precise, and easier to navigate.

### Lane I: Benchmarks and performance reporting

Purpose: make performance work reproducible and resistant to misleading claims.

Targets:

- `BENCHMARKS.md`
- `benches/cpu_baselines.rs`
- `benches/bench_util.rs`
- `scripts/criterion_to_json.py`
- `benchmark-results/` if used

Tasks:

1. Ensure maintained benchmark groups map to active optimized paths.
2. Add benchmark entries only for stable representative workloads.
3. Normalize JSON export and baseline naming.
4. Document hardware/commit/baseline requirements.
5. Avoid adding hard-coded speedup tables to README.

Acceptance:

- `cargo +stable bench --bench cpu_baselines -- --list` includes maintained entries.
- Benchmark documentation is reproducible.

### Lane J: Warning and lint debt

Purpose: reduce accepted warning debt so future CI failures are easier to interpret.

Targets already known:

- `src/autograd.rs`
- `src/compiler/passes/auto_cast.rs`
- `src/compiler/passes/quantization.rs`
- `src/ir/builder.rs`

Tasks:

1. Fix unused variables in tests where simple.
2. Rename non-snake-case local test variables if not semantically important.
3. Prefer meaningful `_name` only when the value documents setup but is intentionally unused.
4. Avoid broad `allow` attributes except for hot-path API shapes or intentionally verbose test names.

Acceptance:

- Local warning count decreases.
- Clippy baseline remains green.
- No semantic changes except names/cleanup.

### Lane K: Error handling and public reliability

Purpose: make failure modes explicit and user-facing errors useful.

Targets:

- `src/error.rs`
- `src/python/*.rs`
- `tests/public_error_paths.rs`
- Python tests for invalid inputs

Tasks:

1. Add tests for invalid shapes/dtypes/devices.
2. Ensure Python exceptions map to stable error classes.
3. Improve messages only where tests anchor expected semantics.
4. Avoid changing successful API behavior.

Acceptance:

- Public error path tests pass.
- Error messages are clearer and less panic-like.

### Lane L: Repository hygiene

Purpose: keep the repo clean after autonomous/agent work.

Targets:

- tracked files
- worktrees under `/tmp/fastnn-*`
- accidental generated files
- stale branches

Tasks:

1. Inspect `git status --ignored --short` periodically.
2. Remove accidentally tracked pycache/native artifacts only if tracked.
3. Prune merged worktrees and branches.
4. Keep canonical repo clean except `.hermes/` untracked.

Acceptance:

- `git status --short --branch` clean except `.hermes/`.
- No tracked scratch/generated artifacts introduced.

---

## Run reporting format

Each 2-hour run should report only when meaningful:

```text
fastnn autonomous run: <lane>
Verdict: ✅ merged and CI green | ❌ blocked/failure | ℹ️ audit only
Commit: <sha or none>
What changed:
- <bullet>
Validation:
- <commands/results>
Next safe lane:
- <one-liner>
```

If nothing changed and no issue was found:

```text
[SILENT]
```

---

## Stop and escalation conditions

Stop implementation and report if:

1. `main` cannot be fast-forwarded or has unexpected tracked local changes.
2. GitHub authentication fails for push or CI inspection.
3. A structural correctness bug is found and the safe fix is unclear.
4. CI is red and cannot be diagnosed after one focused attempt.
5. A change would require breaking public API.
6. A lane requires hardware not available locally, such as GPU-specific validation.
7. Disk/memory pressure makes builds unsafe.

Do not ask routine questions while the user is gone. Choose the conservative path and continue with another safe lane.

---

## Definition of exceptional progress

fastnn becomes exceptional by compounding small verified improvements:

- Correctness bugs are caught by focused regression tests.
- Memory/copy reductions are protected by telemetry and alias-safety tests.
- Benchmarks are reproducible and not marketing claims.
- Documentation is truthful, professional, and easy to navigate.
- CI is green, strict, and boring.
- Public Python APIs are tested and stable.
- Optional backend paths are clearly marked and validated where possible.
- Repository hygiene remains clean after every autonomous run.
