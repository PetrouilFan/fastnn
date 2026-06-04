# autonomous fastnn work log

Persistent append-only coordination log for human sessions, cron jobs, and autonomous agents working from `docs/plans/autonomous-fastnn-persistent-plan.md`.

Keep entries concise. Record intent, actual changes, validation, findings, and next recommended action. Do not paste giant logs.

## 2026-06-05 — cost-aware OpenCode policy added

Intent:
- Reduce paid Hermes/GPT API token usage during autonomous fastnn work by delegating token-heavy codebase reading, coding, and summarization to free OpenCode MiniMax M3 where practical.

Changed:
- Added a cost-aware agent policy to `docs/plans/autonomous-fastnn-persistent-plan.md`.
- Updated the fastnn autonomous cron prompt to prefer `opencode/minimax-m3-free` for read-only investigations and bounded coding tasks, with Hermes still responsible for verification.

Validation:
- Documentation/cron policy change only.

Findings:
- OpenCode should be treated as a worker, not a source of truth. Its code changes and summaries still need independent diff/test/file verification before reporting success or pushing.

Next recommended action:
- Future P0/P1 autonomous runs should launch OpenCode MiniMax M3 for large code-reading or implementation lanes, then use Hermes for orchestration, validation, and final integration.

## 2026-06-05 — persistent plan initialized

Intent:
- Stop generic low-value autonomous cron work and make future runs follow a shared plan focused on broad graph/compiler/runtime efficiency.

Changed:
- Added `docs/plans/autonomous-fastnn-persistent-plan.md`.
- Added this shared work log.
- Current priority queue starts with dynamic memory traffic profiling on top of `AotExecutor.memory_stats()` and `scripts/graph_memory_stats.py`.

Validation:
- Plan/log creation only; no runtime code changes in this initialization entry.

Findings:
- Latest measured graph-memory stats for YOLOv8n showed estimated static traffic 92.81 MiB, write_const 12.06 MiB, memcpy 1.70 MiB, and only 3.9% logical slot reuse savings by the new metric.

Next recommended action:
- Implement `scripts/graph_memory_profile.py` or equivalent to combine profile timings with static byte estimates and rank memory/layout bottlenecks by bytes per ms.

## 2026-06-05 01:35 — autonomous cron

Intent:
- Complete P0 dynamic memory traffic profiling by combining `AotExecutor.memory_stats()` with dynamic profile timings.

Changed:
- Added `scripts/graph_memory_profile.py` with JSON output and a reusable `build_memory_profile()` helper.
- Added `tests/test_graph_memory_profile.py` covering kernel byte ranking, unprofiled copy/const traffic, and no-double-count handling for profiled `memcopy`/`write_const` entries.
- Added `docs/plans/graph-memory-profile.md` with usage, JSON schema notes, limitations, and a YOLOv8n smoke result.

Validation:
- RED: `.venv/bin/python -m pytest tests/test_graph_memory_profile.py -q` initially failed with `ModuleNotFoundError: No module named 'scripts.graph_memory_profile'`.
- GREEN: `.venv/bin/python -m pytest tests/test_graph_memory_profile.py -q && .venv/bin/python -m py_compile scripts/graph_memory_profile.py` → 3 passed.
- Smoke: `PYENV_VERSION=system .venv/bin/python scripts/graph_memory_profile.py --onnx yolov8n.onnx --json /tmp/graph_memory_profile_yolo.json --top 10` → wrote `/tmp/graph_memory_profile_yolo.json`; profiled total 311.920 ms; estimated static traffic 92.81 MiB; profiled kernel static traffic 79.05 MiB; unprofiled copy/const traffic 0 B.
- `git diff --check` → pass.

Findings:
- Top dynamic/static memory signals in the YOLO smoke: `conv2d_silu` 37.55 MiB / 233.180 ms, `write_const` 12.06 MiB / 1.821 ms, `slice_f32` 11.86 MiB / 1.377 ms, `concat` 11.20 MiB / 5.546 ms, `memcopy` 1.70 MiB / 0.831 ms.
- Profile entries include `write_const` and `memcopy`, so the profiler must treat those as special traffic classes instead of apportioning aggregate kernel bytes to them; tests now guard this.

Next recommended action:
- Start P1 with a measurement-first design for safe persistent constants or immutable constant slots; `write_const` is small in wall time but broad and clearly per-forward traffic.
