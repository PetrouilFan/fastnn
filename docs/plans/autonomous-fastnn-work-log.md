# autonomous fastnn work log

Persistent append-only coordination log for human sessions, cron jobs, and autonomous agents working from `docs/plans/autonomous-fastnn-persistent-plan.md`.

Keep entries concise. Record intent, actual changes, validation, findings, and next recommended action. Do not paste giant logs.

## 2026-06-05 09:xx EEST — interactive session

Intent:
- Proceed with the best P1 path after the overnight profiler work: measure whether extending prepared-arena constant preloading beyond Conv weights is useful.

Changed:
- Added `packed_bias` metadata to `PreparedConv2d` and `PreparedMatMul`.
- Updated prepared static binding application to attach `input_index == 2` bindings as `packed_bias`.
- Updated the opt-in prepared-arena fallback to preload fp32 Conv/MatMul weights and biases/static slots, then skip matching `WriteConst` instructions.
- Documented the result in `docs/plans/graph-memory-profile.md`.

Validation:
- `cargo test --release --lib --features prepared-plan prepared::tests::detects_conv_weight_and_bias` → 1 passed.
- `cargo test --release --lib --features prepared-plan prepared_fallback_tests` → 5 passed.
- `VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv .venv/bin/python -m maturin develop --release --features 'prepared-plan openblas'` → pass.
- `.venv/bin/python -m pytest tests/test_aot_executor_prepared_execute.py tests/test_graph_memory_profile.py -q` → 20 passed.
- `PYENV_VERSION=system .venv/bin/python scripts/graph_memory_profile.py --onnx yolov8n.onnx --json /tmp/graph_memory_profile_bias_preload_yolo.json --top 8 --also-prepared` → pass.
- `PYENV_VERSION=system OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/prepared_fallback_overhead.py --onnx yolov8n.onnx --warmup 2 --iters 8 --json /tmp/prepared_fallback_bias_preload_overhead.json` → pass, exact outputs.

Findings:
- The extended prepared-arena fallback reduced profiled `write_const` entries by 127 and static WriteConst traffic by 11.70 MiB, with `write_const` profiled time delta -4.558 ms in one instrumented run.
- It did not improve end-to-end timing: `forward` mean 46.185 ms vs `forward_prepared_arena_fallback` mean 48.913 ms in the 8-iter timing sample, with `max_abs=0` and `mean_abs=0`.
- Conclusion: more per-forward preload pruning is the wrong performance direction. It proves the traffic is removable, but copying constants from the prepared arena into the mutable arena each forward still costs too much.

Next recommended action:
- Design a persistent immutable constant/no-copy prepared plan so kernels consume static payloads without per-forward arena re-materialisation, or switch to P2 `concat` view/layout work if the no-copy prepared plan is too invasive.

## 2026-06-05 09:53 EEST — autonomous cron

Intent:
- Add the P3/P2 diagnostic needed before doing layout-copy work: exact instruction-level static traffic rows, not just aggregate kernel-kind bytes.

Changed:
- Extended `AotExecutor.memory_stats()` with `top_instructions_by_static_bytes` rows containing instruction index, kind, kernel name, node id, read bytes, write bytes, and static bytes.
- Updated `scripts/graph_memory_stats.py` to print the top instruction traffic rows and continue writing them in raw JSON.
- Added a Python regression test that first failed on the missing memory-stats key, then passed after the Rust API change.
- Updated `docs/plans/graph-memory-stats.md` with the new diagnostic and current YOLO instruction-level findings.

Validation:
- RED: `.venv/bin/python -m pytest tests/test_aot_executor_profile.py::test_memory_stats_reports_instruction_level_static_traffic -q` → failed with `KeyError: 'top_instructions_by_static_bytes'` before implementation.
- `cargo check --release --lib` → pass.
- `VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv .venv/bin/python -m maturin develop --release --features 'prepared-plan openblas'` → pass.
- `.venv/bin/python -m pytest tests/test_aot_executor_profile.py -q` → 2 passed.
- `.venv/bin/python -m py_compile scripts/graph_memory_stats.py` → pass.
- `PYENV_VERSION=system .venv/bin/python scripts/graph_memory_stats.py --onnx yolov8n.onnx --json /tmp/graph_memory_stats_instruction_traffic_yolo.json` → pass.
- `cargo fmt --check` → pass.
- `cargo test --release --lib` → 241 passed.
- `cargo test --release --lib --features prepared-plan` → 246 passed.
- `cargo test --release --lib --features 'prepared-plan openblas'` → 248 passed.
- `git diff --check` → pass.

Findings:
- YOLOv8n top instruction-level static traffic now identifies the concrete plan rows to inspect next: `#131 conv2d_silu` 2.74 MiB, `#132 conv2d_silu` 2.36 MiB, `#139 concat` 2.34 MiB, `#190 concat` 2.34 MiB, and `#140 conv2d_silu` 1.96 MiB.
- This confirms aggregate `concat` traffic is concentrated in a few large copy-like instructions, making P2 view/layout planning more actionable than aggregate counts alone.

Next recommended action:
- Use the new instruction indices/node ids to inspect the two largest YOLO `concat` rows and add a RED guardrail for any safe view/layout or copy-elision candidate; otherwise continue P1 persistent immutable constants if no safe concat pattern is found.

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

## 2026-06-05 03:11 EEST — autonomous cron

Intent:
- Improve the P0 graph memory profiler accuracy before choosing P1/P2 implementation work, so kernel categories are ranked by actual compiled-plan traffic instead of call-count apportioning.

Changed:
- Extended `AotExecutor.memory_stats()` `top_kernels_by_count` rows with exact per-kernel aggregate `read_bytes`, `write_bytes`, and `static_bytes`.
- Updated `scripts/graph_memory_profile.py` to prefer those exact fields when present and fall back to the old estimate for older payloads.
- Added a RED/GREEN Python regression test for exact per-kernel byte usage.
- Updated `docs/plans/graph-memory-profile.md` with the new schema and YOLO smoke result.

Validation:
- RED: `.venv/bin/python -m pytest tests/test_graph_memory_profile.py::test_build_memory_profile_prefers_exact_per_kernel_static_bytes_when_available -q` failed with `assert 5461 == 8000` before the profiler change.
- GREEN: `.venv/bin/python -m pytest tests/test_graph_memory_profile.py -q` → 4 passed.
- `.venv/bin/python -m py_compile scripts/graph_memory_profile.py` → pass.
- `cargo check --release --lib` → pass.
- `cargo fmt --check` → pass.
- `VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv .venv/bin/python -m maturin develop --release --features 'prepared-plan openblas'` → rebuilt and installed `fastnn-2.2.4`.
- `PYENV_VERSION=system .venv/bin/python scripts/graph_memory_profile.py --onnx yolov8n.onnx --json /tmp/graph_memory_profile_exact_bytes_yolo.json --top 10` → pass; profiled total 491.677 ms, estimated static traffic 92.81 MiB, profiled kernel static traffic 79.05 MiB.
- `cargo test --release --lib` → 241 passed.
- `cargo test --release --lib --features prepared-plan` → 246 passed.
- `cargo test --release --lib --features 'prepared-plan openblas'` → 248 passed.

Findings:
- Exact YOLOv8n static byte ranking changed the broad memory/layout priorities: `conv2d_silu` 43.05 MiB, `concat` 16.12 MiB, `write_const` 12.06 MiB, `slice_f32` 7.13 MiB, `add_f32` 3.17 MiB.
- The previous call-count estimate understated `concat` and overstated `slice_f32`; future P2 view/layout work should treat `concat` as a top copy-like target alongside persistent constants.

Next recommended action:
- Use the exact-byte profiler output to start P1/P2 with guardrail tests: first persistent constants/immutable constant slots for 12.06 MiB WriteConst traffic, or a `concat` layout/view diagnostic because it now ranks above `slice_f32` by actual bytes.

## 2026-06-05 04:53 EEST — autonomous cron

Intent:
- Continue P1 prepared/persistent-constant work by making the opt-in prepared-arena path profileable, so future agents can measure WriteConst removal instead of only timing full forward calls.

Changed:
- Added `GraphExecutor::execute_profile_prepared_arena_fallback()` behind `prepared-plan`.
- Exposed `AotExecutor.profile_prepared_arena_fallback(inputs)` with the same `{outputs, profile}` shape as `profile()`.
- Added a Python RED/GREEN regression test for the new public profiling hook.

Validation:
- RED: `.venv/bin/python -m pytest tests/test_aot_executor_prepared_execute.py::test_profile_prepared_arena_fallback_matches_profile_shape -q` failed with missing `profile_prepared_arena_fallback` attribute.
- Rebuild: `VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv .venv/bin/python -m maturin develop --release --features 'prepared-plan openblas'` → installed `fastnn-2.2.4`.
- GREEN: `.venv/bin/python -m pytest tests/test_aot_executor_prepared_execute.py::test_profile_prepared_arena_fallback_matches_profile_shape -q` → 1 passed.
- `.venv/bin/python -m pytest tests/test_aot_executor_prepared_execute.py -q` → 8 passed.
- `.venv/bin/python -m py_compile scripts/prepared_fallback_overhead.py` → pass.
- `cargo fmt --check` → pass.
- `cargo test --release --lib` → 241 passed.
- `cargo test --release --lib --features prepared-plan` → 246 passed.
- `cargo test --release --lib --features 'prepared-plan openblas'` → 248 passed.
- `git diff --check` → pass.

Findings:
- YOLOv8n profile smoke using the new hook wrote `/tmp/fastnn_prepared_arena_profile_yolo.json` and produced byte-identical outputs (`max_abs=0.0`, `mean_abs=0.0`).
- Default profile vs prepared-arena profile reduced `write_const` profile entries from 131 to 67 and measured `write_const` time from 1.037 ms to 0.008 ms for that profiled run; this confirms the existing arena-preload fallback skips Conv weight writes but leaves other constants/biases on the safe default path.

Next recommended action:
- Either extend `scripts/prepared_fallback_overhead.py`/`graph_memory_profile.py` to emit default-vs-prepared profile deltas automatically, or proceed to the next safe P1 step: include bias/non-Conv constants only with a RED guardrail that proves slot immutability/lifetime safety before skipping more WriteConst instructions.

## 2026-06-05 06:31 EEST — autonomous cron

Intent:
- Make P1 prepared-arena WriteConst traffic reduction measurable from profiler JSON instead of manual profile diffs.

Changed:
- Added `build_profile_delta(default_profile, prepared_profile)` to `scripts/graph_memory_profile.py` for default-vs-prepared memory profile deltas.
- Added RED/GREEN tests for removed/added kernels, profile-time/static-byte summary deltas, unprofiled write_const handling, and shared-`memory_stats()` WriteConst byte apportioning.
- Updated `docs/plans/graph-memory-profile.md` with the delta schema and a YOLOv8n smoke result.

Validation:
- RED: `.venv/bin/python -m pytest tests/test_graph_memory_profile.py -q` initially failed with missing `build_profile_delta`; the shared-memory-stats apportioning test then failed with `0 == -8192` before the count-based byte estimator.
- GREEN: `.venv/bin/python -m pytest tests/test_graph_memory_profile.py -q` → 9 passed.
- `.venv/bin/python -m py_compile scripts/graph_memory_profile.py` → pass.
- YOLO smoke: `PYENV_VERSION=system .venv/bin/python <default/profile_prepared_arena_fallback delta snippet>` wrote `/tmp/graph_memory_profile_prepared_delta_yolo.json`.
- `git diff --check` → pass.

Findings:
- YOLOv8n prepared-arena fallback delta: `write_const_count_delta=-64`, `write_const_static_bytes_delta=-6,179,967` bytes (~5.89 MiB), `write_const_total_ms_delta=-2.284 ms` in the latest single instrumented run.
- Because default and prepared profiles share one static `memory_stats()` payload, profiled `write_const` bytes must be apportioned by observed profiled count over `memory_stats.write_const_count`; otherwise both sides appear to write the full 12.06 MiB aggregate.

Next recommended action:
- Use this delta helper in a CLI `--also-prepared`/sidecar mode or proceed to the next P1 RED guardrail for safely skipping additional non-Conv/bias constants only where slot immutability/lifetime safety is proven.

## 2026-06-05 08:09 EEST — autonomous cron

Intent:
- Turn the prepared-arena profile delta helper into a one-command profiler mode so future P1/P2 runs can archive default-vs-prepared JSON without bespoke Python snippets.

Changed:
- Added `scripts/graph_memory_profile.py --also-prepared`, which runs `profile()` and `profile_prepared_arena_fallback()` on the same input and writes `{default, prepared, delta}` JSON.
- Added `build_profile_payload()` plus regression tests for bundle shape, preserved legacy single-profile shape, and profiled `memcpy` spelling handling.
- Updated `docs/plans/graph-memory-profile.md` with the CLI usage, JSON shape, and current YOLO smoke numbers.

Validation:
- RED: `.venv/bin/python -m pytest tests/test_graph_memory_profile.py::test_build_profile_payload_embeds_default_prepared_and_delta_profiles -q` failed with missing `build_profile_payload` import before implementation.
- GREEN: `.venv/bin/python -m pytest tests/test_graph_memory_profile.py -q` → 12 passed.
- `.venv/bin/python -m py_compile scripts/graph_memory_profile.py` → pass.
- Smoke: `PYENV_VERSION=system .venv/bin/python scripts/graph_memory_profile.py --onnx yolov8n.onnx --json /tmp/graph_memory_profile_also_prepared_yolo.json --top 5 --also-prepared` → pass; wrote `/tmp/graph_memory_profile_also_prepared_yolo.json`.
- `git diff --check` → pass.

Findings:
- Latest single-run YOLOv8n `--also-prepared` delta: profiled total `-30.093 ms`, `write_const_count_delta=-64`, `write_const_static_bytes_delta=-5.89 MiB`, `write_const_total_ms_delta=-8.901 ms`.
- OpenCode MiniMax M3 researcher review confirmed no legacy JSON shape regression with `--also-prepared` absent and recommended adding the flat-shape regression test that is now included.

Next recommended action:
- Proceed to the next P1 RED guardrail for safely skipping additional non-Conv/bias constants only where slot immutability/lifetime safety is proven, or add an instruction-level constant-slot diagnostic if the safe set is still unclear.
