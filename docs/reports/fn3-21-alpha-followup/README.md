# FN3-21 alpha follow-up — matched baseline and strict release smoke

**Branch:** `fn3/alpha-followup` · **Base:** `672cec9faf089ef0840179f6edfdf33779b576a1`
(`rewrite/fastnn-3`, the verified 3.0 alpha candidate) · **Worktree:**
`/home/petrouil/Projects/github/fastnn-wt-alpha` · **Tracking:** #67 (parent #46)

This directory holds the raw evidence for the follow-up:

| File | What it is |
|---|---|
| `release-cpu.json` | primary CPU run: environment, load averages, all 20 prefill and 60 decode samples |
| `release-vulkan.json` | primary GTX 1080 Ti run, same workload, plus the residency counters and device timestamps |
| `release-cpu-run2.json`, `release-vulkan-run2.json` | independent repeat of each backend |
| `release-*.log` | exact stdout, command and `/proc/loadavg` around each run; **local only** (`*.log` is git-ignored — the JSON repeats the same facts) |

The corrected timing section of `docs/reports/fn3-21-alpha-release.md` cites
this run; its superseded `--quick` table is kept there as a labelled correction.

## 1. Matched release-build baseline

### Method

`tests/fn3_alpha_bench.rs` drives the production `Session` path, one concrete
backend per process, after:

```bash
CARGO_BUILD_JOBS=3 cargo build --release --features cli,vulkan
```

The release profile is `opt-level=3`, `lto=true`, `codegen-units=1`. Every run
held `flock /tmp/opencode/fn3-bench.lock`; `/proc/loadavg` is recorded before
and after in each JSON.

Workload, identical on both backends:

- prompt `464,2068,7586,21831,18045,625,262,16931,3290,13` (10 tokens);
- decode tokens `640,11,612`, so the cache holds 11, 12 and 13 tokens after the
  three measured steps;
- 3 untimed warmups, 20 measured iterations;
- each iteration calls `Session::reset()` and replays the same four shapes
  (`prefill (10,0)`, decode `past=10,11,12`), so the four plans are built in
  warmup and every measured step is a plan-cache hit (88 hits, 4 misses,
  0 evictions);
- the model is verified against `fixtures/parity/tiny-gpt2/model.lock.json`
  (bytes and SHA-256 read from the lock, not copied from a brief).

What is timed:

- **Host wall**: monotonic `Instant` around `Session::prefill` / `Session::decode`
  plus acquisition of the returned logits host slice — call entry to logits
  available. Top-k is computed outside the timed region. The Vulkan submission
  path waits on a fence per instruction, so the call returns after every
  instruction completed.
- **Device time** (Vulkan only): the sum of the per-instruction timestamp
  queries in the residency ledger (`gpu_ns`). It is kernel time, not host
  submission time and not the wall clock. The CPU backend has no device clock;
  that row is `unavailable`, never zero.
- **Counters**: residency-ledger deltas for upload bytes, readback bytes,
  submissions, executions and proven device executions.
- **Memory**: peak host RSS from `/proc/self/status` `VmHWM`; Vulkan peak from
  `VulkanBackend::memory_stats().peak_live_bytes`.

Percentiles are nearest-rank (`rank = ceil(p/100 · n)`, 1-based); p95 is the
19th of 20 samples.

### Result (primary run, `release-*.json`)

| Metric | CPU | GTX 1080 Ti (Vulkan) |
|---|---|---|
| Model | 27,897,974 B, sha256 `e18370ba…141a1` | same |
| `model.load` (read + digest + ONNX import), 1 shot | 5.30 ms | 5.61 ms |
| `model.prepare`, 1 shot | 6.15 ms | 6.84 ms |
| **Prefill TTFT p50** | **240.31 ms** (41.61 tok/s) | **204.76 ms** (48.84 tok/s) |
| **Prefill TTFT p95** | **246.02 ms** | **245.12 ms** |
| Prefill device p50 / p95 | unavailable — no CPU clock | **13.22 / 55.33 ms** |
| **Decode p50 / token** | **158.33 ms** (6.32 tok/s) | **190.64 ms** (5.25 tok/s) |
| **Decode p95 / token** | **179.98 ms** | **228.22 ms** |
| Decode device p50 / step | unavailable | **12.93 ms** (steps: 12.93, 12.93, 28.03) |
| Decode device p95 / step | unavailable | 54.21 / 54.21 / 54.26 ms |
| Upload / readback / submissions, one chain | unavailable — no ledger | 318,880,260 B / 106,941,352 B / 575 |
| Executions over 20 iterations | unavailable — no ledger | 11,500, **11,500 proven**, 0 input readbacks |
| Peak host RSS | 88.76 MiB | 396.72 MiB |
| Peak device memory | unavailable — no allocator | allocator high-water **120,669,184 B**; device-local live **0 B**; device-local peak **unavailable** |
| Validation | n/a | **0 errors / 0 warnings** |
| 1-min load, start / end | 6.29 / 6.67 | 5.86 / 6.01 |

Correctness in every iteration: prefill top-5 `[46737, 28597, 16571, 7868, 13]`,
decode argmax `32493, 11724, 612`.

**Repeatability (`*-run2.json`).** CPU prefill p50 288.31 ms / p95 307.80 ms,
decode p50 204.94 ms; Vulkan prefill p50 231.52 ms (device 28.55 ms), decode
p50 195.93 ms (device 28.03 ms). Host-side p50 moves by about 20 % on the CPU
and 13 % on Vulkan between the two runs.

**The Vulkan device time is not stable yet.** Its p50 spans 12.93–28.55 ms per
step across runs and even within one run (steps 1–2 at 12.93 ms, step 3 at
28.03 ms). The card's clock is unpinned: `nvidia-smi` reports 139 MHz SM at idle
against a 1911 MHz maximum, and the per-instruction fenced kernels are too short
to hold a boost state. The p95 of 54 ms is the same effect at the low clock.
Host wall time is the comparable metric; the device timestamps are a lower
bound that also proves the kernels ran. `nvidia-smi -lgc` needs root — owner
decision, not taken.

### What the correction replaces

The handoff's `--quick` table (load 135.92 ms, prepare 2.90 ms, prefill
271.77 ms / 36.80 tok/s, decode 179.74 ms/token, peak RSS 165.67 MiB, device
peak unavailable) is kept in the handoff, labelled. Four problems:

1. no build profile recorded;
2. `--quick` sampled less, published a median with no tail, and committed no
   samples;
3. the CPU column came from `fn3-bench` while the Vulkan prose came from the CLI
   on a 13-token prompt — never a matched comparison;
4. `memory.device_peak` was an unpopulated field in
   `tools/bench/adapters/fn3.rs` (`Runtime::peak_device_memory_bytes` is
   initialised to `None` and never assigned), not a backend limitation. The
   backend does expose `VulkanBackend::memory_stats().peak_live_bytes`, which
   this test reads.

## 2. Strict release wheel smoke

`--require-inference` was added to `tools/ci/wheel_smoke.py` and
`tools/ci/packaging_lane.py`. The lane also passes the pinned sha256 from
`fixtures/parity/tiny-gpt2/model.lock.json` via `--expect-model-sha256`, so a
file at the right path with the wrong bytes fails.

Proven locally:

| Case | Command | Exit | Message / record |
|---|---|---|---|
| model present, strict | `python3 tools/ci/packaging_lane.py --require-inference` | **0** | `WHEEL INFERENCE: PASSED — top-5 [46737, 28597, 16571, 7868, 13]` |
| model absent, strict | same with `FASTNN_FN3_ONNX_MODEL=/tmp/opencode/absent-tiny.onnx` | **1** | `PACKAGING FAILED: strict wheel inference requires the pinned tiny-GPT-2 ONNX at …; fetch https://huggingface.co/onnx-community/tiny-random-gpt2-ONNX/resolve/90f61e71…/onnx/model.onnx and verify sha256 e18370ba…141a1` |
| model absent, non-strict | `python3 tools/ci/packaging_lane.py --skip-build` | **0** | `WHEEL INFERENCE: NOT RUN — …`; lane record `wheel_inference: not_run` with the reason |
| wheel_smoke strict, absent | `wheel_smoke.py --model …absent… --require-inference` | **2** | `WHEEL INFERENCE: FAILED — …` + `STRICT MODE: …`; report `status=failed, strict=true` |
| wheel_smoke non-strict, absent | same without the flag | **0** | `WHEEL INFERENCE: SKIPPED — …`; report `status=skipped, strict=false` |

The release workflow (`.github/workflows/release.yml`, `build-wheels`) now
fetches `onnx/model.onnx` on every wheel platform from the URL derived from the
lock's `model_id` and `revision`, checks `bytes` and `sha256` against the same
entry, and runs the lane with `--require-inference`. A fetch or digest failure
is a named `::error::pinned-model …` failure; it can never be `not_run`.

The URL was exercised from this machine with the workflow's own logic: HTTP 200,
27,897,974 bytes, sha256 `e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1`
— exactly the lock entry.

## 3. Crate metadata and package contents

`Cargo.toml` had no description, license, repository, homepage, readme,
keywords, categories, `rust-version`, or include/exclude. `pyproject.toml` had
none of the corresponding PEP 621 fields. Both are now declared:

- Cargo: `description`, `license = "MIT"`, `repository`, `readme`, `keywords`,
  `categories`, `rust-version = "1.89"`, and an include list;
- pyproject: `description`, `readme`, `license = "MIT"` (PEP 639),
  `license-files`, `authors`, `keywords`, `classifiers`, `[project.urls]`.

The licence was read from the committed `LICENSE` (MIT, 2026 Petros
Fanioudakis) and the README badge; it was not invented. `homepage` is omitted
from `Cargo.toml` because it is identical to `repository` and triggers cargo's
`redundant_homepage` lint; `pyproject.toml` keeps both URLs.

`rust-version` is measured: `cargo check` passes with 1.89.0 for default
features, `--no-default-features`, `--features cli,vulkan` and the `fn3-bench`
feature set; 1.88.0 rejects the AVX-512 intrinsics in the retained 2.x CPU
kernels. No bisection below 1.89 was attempted.

Package:

- before: 672 files, 10.4 MiB unpacked / 3.1 MiB compressed, 91 fixture files,
  docs, tests, artifacts and the Python package;
- after: **306 files**, 5.2 MiB unpacked / **1021.3 KiB compressed**, containing
  `src/`, `benches/`, `examples/`, `tools/bench/`, `README.md`,
  `CHANGELOG.md`, `LICENSE` and the Cargo files — no fixtures, models, `target/`,
  tests or multi-MB data (the largest single file is source);
- `cargo package` verifies the unpacked crate builds;
- `cargo publish --dry-run` stops at `Uploading …` / `aborting upload due to
  dry run` with no manifest warnings.

Wheel (`unzip -l`): 47 files — 41 under `fastnn/`, 6 under
`fastnn-3.0.0a1.dist-info/` (`METADATA`, `WHEEL`, `entry_points.txt`,
`licenses/LICENSE`, `sboms/fastnn.cyclonedx.json`, `RECORD`). No tests,
fixtures, `.onnx`, `.fnn3` or `target/` entries. `METADATA` is 2.4 with
`Summary`, `License-Expression: MIT`, `License-File: LICENSE`,
`Requires-Python: >=3.11`, the five keywords, five classifiers, two
`Project-URL`s and the README as the long description.

## 4. Hosted CI on the review candidate and the follow-up

`rewrite/fastnn-3` was confirmed at `672cec9faf089ef0840179f6edfdf33779b576a1`
before dispatch and was not modified.

| Run | Ref / headSha | Result |
|---|---|---|
| [44486950429](https://github.com/PetrouilFan/fastnn/actions/runs/44486950429) | `rewrite/fastnn-3` / `672cec9` | **failed**: `vulkan-software` and `lane summary`; every other job success |
| [44486993070](https://github.com/PetrouilFan/fastnn/actions/runs/44486993070) | `fn3/alpha-followup` / `476dbb5` | **failed**: same `vulkan-software` failure |
| [44488019087](https://github.com/PetrouilFan/fastnn/actions/runs/44488019087) | `fn3/alpha-followup` / `cc3c50b` | **success**: all jobs and the lane summary |
| [44488737026](https://github.com/PetrouilFan/fastnn/actions/runs/44488737026) | `fn3/alpha-followup` / `49ae595` | **success**: all jobs and the lane summary |
| [44489825235](https://github.com/PetrouilFan/fastnn/actions/runs/44489825235) | `fn3/alpha-followup` / `d8f101d` (this folder's first commit) | **success**: all jobs and the lane summary |

The final commit that adds this row is documentation-only; every code, test,
workflow and packaging change is covered by the `d8f101d` run.

The failure was a hosted-environment gap, not a code regression:
`tests/fn3_vulkan_smoke.rs::software_icd_lane_is_labelled_and_matches_cpu`
opens the lavapipe instance with validation enabled in code, and the runner
image ships lavapipe but not `VK_LAYER_KHRONOS_validation`:

```text
creating a Vulkan instance failed: validation was requested but
VK_LAYER_KHRONOS_validation is not installed; available layers:
VK_LAYER_MESA_device_select, VK_LAYER_MESA_anti_lag, VK_LAYER_INTEL_nullhw,
VK_LAYER_MESA_overlay, VK_LAYER_MESA_screenshot
```

13 of 14 smoke tests passed; that one failed and correctly took the lane with
it. The fix on `fn3/alpha-followup` is one install line,
`vulkan-validationlayers`, in the `vulkan-software` job. The gate was not
weakened: the lane still requires validation 0 errors / 0 warnings. The
`cc3c50b` run proves the fix, with `vulkan-software` and the lane summary green.

Per-job conclusions on the successful run: `cpu-rust` success, `cpu-python`
success, `vulkan-build` success, `vulkan-software` success, `packaging` success
on ubuntu/macos/windows, `pinned-model` success (recorded NOT VERIFIED by
design — `tools/ci/model-pin.json` is `unpinned`, #49), `gpu-physical` success
(validates the committed physical-GPU record; it never runs a GPU on a hosted
runner, by design), `lane-summary` success.

The `packaging` matrix in `ci.yml` stays non-strict and records
`wheel_inference: not_run` with the absent-model reason; strict inference is a
release-only gate (documented in `docs/internals/release-process.md`).

## 5. Commands and real outcomes

| Command | Outcome |
|---|---|
| `cargo build --release --features cli,vulkan` | ok, 2m34s (base commit) |
| `cargo fmt --check` | clean |
| `cargo clippy --all-targets -- -D warnings` | clean |
| `cargo clippy --all-targets --no-default-features --features vulkan -- -D warnings` | clean |
| `cargo test --lib` | **673 passed** |
| `cargo test --test fn3_alpha_bench` (debug) | 1 passed, 292.22s, records `evidence=false` |
| `FN3_BENCH_BACKEND=cpu … cargo test --release --features vulkan --test fn3_alpha_bench` | 1 passed; two runs committed |
| `FN3_BENCH_BACKEND=vulkan …` (UUID, `FN3_VK_REQUIRE_PHYSICAL=1`) | 1 passed; two runs committed, validation 0/0 |
| `cargo package --list` | 306 files, no forbidden path |
| `cargo package` | 306 files, 1021.3 KiB compressed, verify build ok |
| `cargo publish --dry-run --allow-dirty` | stops at upload; no manifest warnings |
| `cargo metadata --no-deps` | description/license/repository/readme/keywords/categories/rust-version present |
| `cargo +1.88.0 check` (default) | **fails**: AVX-512 intrinsics unstable |
| `cargo +1.89.0 check` (4 configurations) | all pass → `rust-version = "1.89"` |
| `python3 tools/ci/packaging_lane.py --require-inference` | **0**, strict inference PASSED |
| same, absent model | **1**, named strict failure |
| `python3 tools/ci/packaging_lane.py --skip-build` (absent) | **0**, `wheel_inference: not_run` |
| `wheel_smoke.py --require-inference` (absent) | **2**, `status=failed` |
| `wheel_smoke.py` (absent) | **0**, `status=skipped` |
| `python3 scripts/ci/check-versions.py --tag v3.0.0-alpha.1` | VERSIONS AGREE |
| `gh workflow run ci.yml --ref rewrite/fastnn-3` | run 44486950429 (failed, cause above) |
| `gh workflow run ci.yml --ref fn3/alpha-followup` | runs 44486993070 (failed), 44488019087 (green), 44488737026 (final) |

## 6. Unavailable metrics, skips and owner decisions

- `memory.device_local_peak` — **unavailable**. `MemoryStats` exposes
  `peak_live_bytes` (all memory types) but no per-category high-water mark.
  The missing accessor is named in the JSON:
  `MemoryStats::peak_live_device_local_bytes`. Live device-local is 0 B at rest
  because alpha tensor storage is host memory.
- CPU device time / upload / readback / submissions — **unavailable by
  construction**: the CPU backend keeps no residency ledger, and it is never
  zero-filled.
- `pinned-model` CI lane — **NOT VERIFIED by design**: `tools/ci/model-pin.json`
  is `unpinned` and names #49.
- `gpu-physical` CI lane — **PASSED as evidence validation only** on the hosted
  runner; no GPU is executed there and the lane never claims one.
- GPU clock pinning (`nvidia-smi -lgc`) — **owner decision**; needs root, and
  without it Vulkan device timestamps span a 2–4× range.
- `fn3-bench`'s `Runtime::peak_device_memory_bytes` is still never assigned in
  `tools/bench/adapters/fn3.rs`. This follow-up did not touch it (the benchmark
  adapter is outside the owned files); the new test reads the backend accessor
  directly. Assigning it would make the `fn3-bench` schema report the same
  number and is a small follow-up for the bench owner.
- **Strict inference is only proven on Linux here.** The release workflow will
  run it on macOS and Windows for the first time; those wheels import and pass
  the non-strict packaging lane (CI run above), but no local machine can run
  their inference. If a platform fails there, the release job will say so by
  name instead of publishing an unverified wheel.
- No tag, release, package upload, PR, issue comment or change to
  `rewrite/fastnn-3` was made. Only `fn3/alpha-followup` was pushed.
