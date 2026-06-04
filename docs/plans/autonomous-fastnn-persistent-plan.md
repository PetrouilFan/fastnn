# fastnn persistent autonomous improvement plan

Purpose: give every human session, cron job, and coding agent the same durable execution plan so autonomous work compounds instead of producing scattered low-value changes.

Canonical repo: `/home/petrouil/Projects/github/fastnn`
Canonical branch: `dev`
Protected branch: `main` — do not commit or push main autonomously.

## Operating contract

Autonomous work must produce meaningful library improvements, not cosmetic churn.

Required every run:

1. Preflight
   - `git fetch origin --prune`
   - `git checkout dev`
   - merge `origin/main` into `dev` only if needed and cleanly possible
   - confirm `git branch --show-current` is `dev`
   - if the worktree is dirty before starting, inspect it; if it is not clearly your own resumable work, stop and report
   - never use `git reset --hard`

2. Read this file and `docs/plans/autonomous-fastnn-work-log.md` before choosing work.

3. Pick one substantial item from the active priority queue below. Do not make tiny 2-3 line noise commits unless they fix a real failing test, correctness bug, or measured performance issue.

4. Use measurement-first development:
   - add or extend benchmark/profiling tooling when the bottleneck is unclear
   - implement only changes with a plausible broad efficiency path
   - verify with tests and, where relevant, before/after runtime numbers

5. Commit and push only if the change is useful and validated.
   - commit to `dev`
   - push only `origin dev`
   - update `docs/plans/autonomous-fastnn-work-log.md` with findings, commands, results, and next recommended action

6. If no useful implementation can be completed in the run, commit nothing. Instead append a concise work-log note explaining the blocker and the next concrete experiment.

## Current strategic direction

The current priority is broad graph/compiler/runtime efficiency, especially memory and layout handling. Do not default to trying random kernels, AVX variants, or backend swaps. Kernel/backend work is acceptable only when connected to whole-model/runtime gates and supported by measurements.

The key hypothesis: reducing unnecessary memory movement, arena writes, and physical layout copies will improve many models, not just YOLO.

## Active priority queue

### P0 — Dynamic memory traffic profiler

Build on `AotExecutor.memory_stats()` and `scripts/graph_memory_stats.py`.

Goal: combine static byte estimates with profile timings so we can rank instructions/kernels by:

- elapsed time
- input bytes
- output bytes
- physical-copy/write bytes
- bytes per ms
- suspected memory-bound vs compute-bound behavior

Preferred deliverable:

- a script such as `scripts/graph_memory_profile.py`
- JSON output
- documentation in `docs/plans/graph-memory-profile.md`
- smoke on `yolov8n.onnx`

This is the best next task because it tells future agents which memory/layout change matters most.

### P1 — Persistent constants vs transient arena design

Use measurements from `graph_memory_stats` / `graph_memory_profile` to design and then implement a safe path for constants that should not occupy mutable transient arena slots or be rewritten every forward.

Do not repeat the unsafe naive `WriteConst` skipping problem. The safe design needs one of:

- separate persistent constant arena
- immutable/non-reused constant slots
- preloaded plan variant whose lifetimes prove constants are never overwritten

Required gate:

- exact output match vs default path
- meaningful full-model/runtime improvement or clearly documented neutral scaffold that enables the next runtime-changing step

### P2 — View/layout planning for copy-like ops

Investigate physical traffic from:

- `slice_f32`
- `concat`
- `transpose_perm_f32`
- reshape/flatten-like paths
- `MemCopy`

Goal: introduce metadata/view planning or fusion to avoid materializing copies where safe.

Start with measurement and a small isolated correctness test. Do not rewrite broad tensor semantics without tests.

### P3 — Memory planner quality metrics and improvements

The current YOLO sample shows low static slot reuse savings by the new metric (~3.9%). Determine whether this reflects true long live ranges, constants occupying arena slots, conservative transitive lifetime extension, or missed reuse opportunities.

Preferred deliverable:

- allocator/lifetime diagnostic script or Rust tests
- top live ranges / peak pressure report
- one safe improvement if discovered

### P4 — Continue prepared-plan runtime specialization only after overhead is understood

Prepared metadata exists, but previous transposed/runtime spikes did not beat default forward. Do not retry the same path. Only resume prepared runtime work if the run first removes/amortizes per-forward overhead or proves a different full-model path.

## Gates

- No noise commits.
- Small cleanup must improve relevant whole-model/runtime metric by >=1% or fix a real correctness/test issue.
- Backend/kernel change must improve full YOLO/runtime by >=5% before integration.
- Major lane should move toward <=35ms YOLO mean on local CPU, but general graph improvements can be accepted when they are model-agnostic and well measured.
- Accuracy for YOLO-style inference should remain around current baseline: `mean_abs_vs_pytorch <= ~2e-6`, `max_abs_vs_pytorch ~5e-4`.

## Required validation menu

Choose the applicable subset, but Rust source changes normally require all Rust suites:

```bash
python -m py_compile <changed python scripts>
cargo fmt --check
cargo test --release --lib
cargo test --release --lib --features prepared-plan
cargo test --release --lib --features 'prepared-plan openblas'
git diff --check
```

For Python-extension API changes:

```bash
VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv \
  .venv/bin/python -m maturin develop --release --features 'prepared-plan openblas'
```

For graph memory tooling smoke:

```bash
PYENV_VERSION=system .venv/bin/python scripts/graph_memory_stats.py \
  --onnx yolov8n.onnx \
  --json /tmp/graph_memory_stats_yolo.json
```

## Shared notes file

Append every nontrivial run to:

`docs/plans/autonomous-fastnn-work-log.md`

Use this format:

```markdown
## YYYY-MM-DD HH:MM — agent/session name

Intent:
- ...

Changed:
- ...

Validation:
- command: result

Findings:
- ...

Next recommended action:
- ...
```

This file is for persistent project-learning and coordination. Keep it concise but specific. Do not paste giant logs.
