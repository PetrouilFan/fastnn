# FN3-21 — FastNN 3.0 alpha release candidate

**Status:** prepared and verified locally. **Not tagged, not published and not
merged.** The owner actions that remain are listed at the end.

| Item | Value |
|---|---|
| Candidate SHA | `d0c7b00e66a981043c59328556fdf87fd5f84eb2` |
| Baseline SHA | `a2c0759fff8bdcb7c943e53442f3382b86e59b66` |
| Branch | `fn3/67-alpha-release` |
| Worktree | `/home/petrouil/Projects/github/fastnn-wt-67-release` |
| Tracking issue | #67 (parent #46) |
| Candidate report | this file; the commit that adds it is documentation only |

The candidate SHA is the tree every command in this report was run against.
The report and the ledger row are the only changes after it; neither touches
code, tests or packaging.

## Package versions

| Location | Spelling | Value |
|---|---|---|
| `Cargo.toml` | SemVer | `3.0.0-alpha.1` |
| `Cargo.lock` (`[[package]] fastnn`) | SemVer | `3.0.0-alpha.1` |
| `fuzz/Cargo.lock` (`[[package]] fastnn`) | SemVer | `3.0.0-alpha.1` |
| `pyproject.toml` | PEP 440 | `3.0.0a1` |
| `uv.lock` (`[[package]] fastnn`) | PEP 440 | `3.0.0a1` |
| `fastnn/__init__.py` | PEP 440 | `3.0.0a1` |
| `CHANGELOG.md` | SemVer | `## [3.0.0-alpha.1]` |
| Built wheel | PEP 440 | `fastnn-3.0.0a1-cp311-abi3-manylinux_2_39_x86_64.whl` |

`scripts/ci/check-versions.py` is the single version rule. It normalises the
SemVer and PEP 440 spellings, checks all seven locations, and accepts
`--tag v3.0.0-alpha.1` for the release workflow. It runs in:

- `tools/ci/cpu-lane.sh` (every CI run);
- `tools/ci/packaging_lane.py` (before the wheel build);
- `.github/workflows/release.yml` (with the pushed tag);
- `scripts/ci/preflight-release.sh`;
- `tests/test_release_versions.py`, which also proves a deliberately drifted
  copy fails.

There is no hand-maintained "three files" rule left: a lockfile, the facade or
the changelog that lags the manifest fails the lane.

## Verified environment

| Item | Value |
|---|---|
| OS | Arch Linux (rolling), kernel `7.2.14-arch1-1` |
| CPU | AMD Ryzen 7 3700X, 8 cores / 16 threads |
| RAM | 31 GiB |
| Rust | `rustc 1.100.0 (0ef1bc386 2026-11-09)`, cargo 1.100.0 |
| Python (packaging) | 3.14.9 in a fresh venv |
| Python (pytest) | 3.13.15 via `uv run --python 3.13` |
| Physical GPU | NVIDIA GeForce GTX 1080 Ti (GP102), UUID `471b2bda-51e1-6451-e692-ff5d1e146f1e` |
| GPU driver | 580.196.07 |
| Vulkan API | 1.4.312 (physical device), instance 1.4.363 |
| Validation layer | `VK_LAYER_KHRONOS_validation` 1.4.363, enabled in code (never `VK_INSTANCE_LAYERS`) |
| Software ICD | Mesa lavapipe `llvmpipe (LLVM 23.1.4, 256 bits)`, CI-only |
| Precision | weights f32, activations f32, accumulation f32; no quantization |
| Concurrency | `CARGO_BUILD_JOBS=12`; physical-GPU runs serialized |

### Model

| Item | Value |
|---|---|
| Model | `onnx-community/tiny-random-gpt2-ONNX` @ `90f61e71d6fa8e571d0ab0f95a637a5d7d8ed52f` |
| Path | `/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx` |
| Bytes | 27,897,974 |
| SHA-256 | `e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1` |
| Geometry | 2 layers, hidden 128, MLP 512, 2 query / 2 KV heads, head dim 64, vocab 50,257, 1,024 learned positions, LayerNorm eps 1e-5, `gelu_new` |
| License | none declared upstream; **not redistributed** |

The candidate artifact, prepared by the Rust CLI, is 27,879,928 bytes with
SHA-256 `33cf409006d9d9538712486d45e9d3cdf1e3cf9176fdfe3c9af477630ca30122`
and payload CRC-32 `0xe02e9330`. (The Python `prepare` defaults to a wider
`max_seq` window and therefore writes a different, equally valid artifact
digest; both execute the same weights and geometry.)

## Reproducible commands

All commands are copy-pasteable from the worktree root. They use the pinned
model path, the pinned 10-token prompt and the GTX 1080 Ti.

```bash
cd /home/petrouil/Projects/github/fastnn-wt-67-release
export CARGO_BUILD_JOBS=12
export MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx
export TOKENS=464,2068,7586,21831,18045,625,262,16931,3290,13
export ARTIFACT=/tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3
mkdir -p /tmp/opencode/fn3-cli/release

sha256sum "$MODEL"
# e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1  .../model.onnx

cargo build --release --no-default-features --features cli,vulkan

./target/release/fn3-infer --version
# fn3-infer 3.0.0-alpha.1

./target/release/fn3-infer prepare --onnx "$MODEL" --out "$ARTIFACT" --force
./target/release/fn3-infer inspect --sections "$ARTIFACT"

./target/release/fn3-infer run \
  --model "$ARTIFACT" --onnx "$MODEL" --tokens "$TOKENS" \
  --device cpu --top-k 5

./target/release/fn3-infer run \
  --model "$ARTIFACT" --onnx "$MODEL" --tokens "$TOKENS" \
  --device vulkan --top-k 5

./target/release/fn3-infer generate \
  --model "$ARTIFACT" --onnx "$MODEL" --tokens "$TOKENS" \
  --max-new-tokens 3 --device vulkan --top-k 3 --steps
```

The Python path, from a wheel installed outside the checkout:

```bash
PATH="$HOME/.local/bin:$PATH" python3 tools/ci/packaging_lane.py
# builds fastnn-3.0.0a1-*.whl, installs numpy only, then runs
# prepare -> load -> prefill -> decode from a temporary directory.
```

The benchmark adapter, which measures the real prepared model rather than the
2.x synthetic stack:

```bash
cargo build --release --no-default-features \
  --features fn3,simd,parallel,prepared-plan,fusion-forward --bin fn3-bench

./target/release/fn3-bench run --adapter fn3 --suite model --quick \
  --fn3-manifest tools/bench/fn3-model.example.json \
  --out /tmp/opencode/fn3-bench/model.json
```

The physical-GPU gate itself:

```bash
FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
FN3_VK_REQUIRE_PHYSICAL=1 CARGO_BUILD_JOBS=12 \
  cargo test --features vulkan --test fn3_e2e -- --test-threads=1 --nocapture
```

## Parity evidence and tolerances

`tests/fn3_e2e.rs` compares logits and every appended KV tensor against the
committed ONNX Runtime / PyTorch fixtures. The documented tolerances are
`atol = 1e-5`, `rtol = 1e-4`, and a mean-absolute systematic-bias bound of
`2.5e-6`; the sequence-scaling rule is `atol *= sqrt(S/16)` above 16 tokens.

Fresh run on the candidate, CPU and GTX 1080 Ti (10 tests, exit 0):

| Tensor | CPU max abs / mean abs | Vulkan max abs / mean abs |
|---|---|---|
| `prefill.p1.logits` | **5.96e-7** / 9.77e-8 | **6.56e-7** / 9.74e-8 |
| `prefill.p4.last_logits` | 5.16e-7 / 7.90e-8 | 5.36e-7 / 8.61e-8 |
| `prefill.p13.last_logits` | 5.36e-7 / 1.00e-7 | 5.96e-7 / 9.89e-8 |
| `decode.step1.logits` | 6.11e-7 / 1.02e-7 | 8.34e-7 / 1.19e-7 |
| decode argmax, steps 1–4 | 24671, 11, 612, 373 | 24671, 11, 612, 373 |

Every row is inside `atol`; the largest is 8.34e-7 against 1e-5. The all-masked
attention row is finite and exactly uniform `[0.25, 0.25, 0.25, 0.25]`.

Cache and execution facts from the same run:

- cached decode extents **14, 15, 16, 17**, KV allocation constant at 34,816 B
  for the 13-token fixture chain; long chain to context 32 has extents
  `14..=32` and constant 65,536 B;
- the 13-token prefill plus four decode steps produced **719/719 proven device
  executions** over 720 plan instructions (one zero-element `ConstantOfShape`
  no-op), **0 input readbacks**, 5 plan misses, 0 evictions;
- the long chain to context 32 produced **2,879/2,879 proven executions**, 0
  input readbacks, 20 plan misses, 0 evictions;
- validation layer: **0 errors, 0 warnings** in the smoke, matmul, pointwise,
  attention and e2e binaries.

Two sessions over one prepared model remain independent; a model prepared for
Vulkan is refused by the CPU backend with `DeviceMismatch` rather than silently
executing.

## Timing evidence

> **Corrected by the alpha follow-up.** This section previously presented a
> `fn3-bench --suite model --quick` table (load 135.92 ms, prepare 2.90 ms,
> prefill 271.77 ms / 36.80 tokens/s, decode 179.74 ms/token, peak RSS
> 165.67 MiB, device peak unavailable). Those figures are kept, labelled, at
> the end of this section. The matched release-build table below is the
> evidence of record.

### Matched release-build baseline

Measured on `672cec9faf089ef0840179f6edfdf33779b576a1` with
`tests/fn3_alpha_bench.rs`, one concrete backend per process, after
`cargo build --release --features cli,vulkan` (`CARGO_BUILD_JOBS=3`,
`profile.release`: `opt-level=3`, `lto=true`, `codegen-units=1`).
Each run held `flock /tmp/opencode/fn3-bench.lock`; `/proc/loadavg` was
recorded before and after, and the raw JSON with every per-iteration sample is
committed under `docs/reports/fn3-21-alpha-followup/`.

Same workload on both backends: prompt
`464,2068,7586,21831,18045,625,262,16931,3290,13` (10 tokens), decode tokens
`640,11,612`, so the cache holds 11, 12 and 13 tokens after the three measured
decode steps. 3 untimed warmups, 20 measured iterations; each iteration calls
`Session::reset()` and replays the same four shapes (`prefill (10,0)` and
`decode past=10,11,12`), so the four plans are built in warmup and every
measured step is a plan-cache hit (88 hits, 4 misses, 0 evictions).

| Metric | CPU | GTX 1080 Ti (Vulkan) |
|---|---|---|
| Model (verified against `model.lock.json`) | 27,897,974 B, sha256 `e18370ba…141a1` | same |
| `model.load` (read + digest + ONNX import), single shot | 5.30 ms | 5.61 ms |
| `model.prepare` (planner + weights + `PreparedModel`), single shot | 6.15 ms | 6.84 ms |
| **Prefill TTFT p50**, host wall | **240.31 ms** (41.61 tok/s) | **204.76 ms** (48.84 tok/s) |
| **Prefill TTFT p95**, host wall | **246.02 ms** | **245.12 ms** |
| Prefill device time p50 / p95 | unavailable — no CPU device clock | **13.22 ms / 55.33 ms** (bimodal, see below) |
| **Decode p50 per token**, host wall | **158.33 ms** (6.32 tok/s) | **190.64 ms** (5.25 tok/s) |
| **Decode p95 per token**, host wall | **179.98 ms** | **228.22 ms** |
| Decode device time p50 per step | unavailable | **12.93 ms** (steps 1/2/3: 12.93, 12.93, 28.03) |
| Decode device time p95 per step | unavailable | 54.21 / 54.21 / 54.26 ms |
| Upload bytes, one prefill + 3 decodes | unavailable — no ledger | 318,880,260 B |
| Readback bytes, one prefill + 3 decodes | unavailable — no ledger | 106,941,352 B |
| Submissions, one prefill + 3 decodes | unavailable — no ledger | 575 (143 prefill + 3 × 144) |
| Executions, 20 iterations | unavailable — no ledger | 11,500, **11,500 proven on the device**, 0 input readbacks |
| Peak host RSS (`VmHWM`) | 88.76 MiB | 396.72 MiB |
| Peak device memory | unavailable — no device allocator | allocator high-water **120,669,184 B** (115.08 MiB); `live_device_local_bytes` **0 B**; device-local-only peak **unavailable** |
| Validation layer | n/a | **0 errors / 0 warnings** |
| 1-min load average at start / end | 6.29 / 6.67 | 5.86 / 6.01 |

Percentiles are nearest-rank over 20 samples (`rank = ceil(p/100 · n)`, 1-based);
p95 is therefore the 19th of 20.

Definitions, so the numbers cannot be read as something they are not:

- **Host wall** is a monotonic `Instant` around `Session::prefill`/`decode` plus
  the acquisition of the returned logits host slice. That is "call entry to
  logits available". Top-k is computed outside the timed region. The Vulkan
  submission path waits on a fence per instruction, so the call returns only
  after every instruction completed; no extra readback is needed or performed.
- **Device time** is the sum of the per-instruction Vulkan timestamp queries in
  the residency ledger (`gpu_ns`) for that phase. It is kernel time, *not* host
  submission time and not the wall clock. The difference is the finding below.
- **Upload/readback/submission** counters are residency-ledger deltas around
  each phase. The CPU backend keeps no such ledger; that is `unavailable`, not
  zero.
- **Peak device memory** is `VulkanBackend::memory_stats().peak_live_bytes`, the
  allocator's high-water mark across all memory types. `MemoryStats` has no
  separate high-water mark for `live_device_local_bytes`, so the device-local
  peak is reported `unavailable` and the missing accessor is named:
  `MemoryStats::peak_live_device_local_bytes`. At rest the allocator holds
  16,777,216 B of host-visible staging and **0 B device-local**, because alpha
  tensor storage is host memory on every device.
- **Peak host RSS** is `/proc/self/status` `VmHWM`, process-wide. The two
  backends are separate processes, so the column is comparable.
- `model.load` and `model.prepare` here are *not* the artifact path the
  superseded table measured: this test imports the ONNX graph and builds the
  prepared model from it, while `fn3-bench` loads the 27,879,928-byte `.fnn3`
  artifact. Only the new column is a matched CPU/Vulkan comparison.

The dominant fact at this commit: the Vulkan device executes a decode step in
12.93–28.03 ms depending on the GPU clock state, but the call takes 190.64 ms.
The host-staged alpha path uploads 79,476,896 B and reads back 26,071,080 B per
decode step, one fence wait and one submission per instruction. That is the
alpha storage contract working as specified, not a kernel problem; closing the
gap is the device-resident storage work (#72) and the performance wave (#73),
not a release blocker.

**Repeatability and the GPU clock.** A second independent run per backend was
committed as `release-*-run2.json`: CPU prefill p50 288.31 ms / p95 307.80 ms,
decode p50 204.94 ms / p95 227.84 ms; Vulkan prefill p50 231.52 ms (device
28.55 ms), decode p50 195.93 ms (device 28.03 ms). The machine is shared with
other release work, so the host-side p50 moves by about 20 % on the CPU and
13 % on Vulkan between the two runs; the conclusions do not change.

The Vulkan device time is **not** a stable estimate at this commit. Its p50
spans 12.93–28.55 ms per step across runs, and within this run steps 1–2 report
12.93 ms while step 3 reports 28.03 ms. The cause is the power state of the
card, not the workload: `nvidia-smi` reports 139 MHz SM clock at idle against a
1911 MHz maximum, and the per-instruction fenced kernels in this small model are
too short to hold a boost clock. The p95 of 54 ms is the same effect at the low
clock. Pinning the clock (`nvidia-smi -lgc`) needs root and is an owner
decision; until then the host wall clock is the metric to compare and the
device timestamps are a lower bound that also proves the kernels ran.

Both runs agree on every correctness and counter fact: top-5
`[46737, 28597, 16571, 7868, 13]`, decode argmax 32493, 11724, 612, 4 plans and
0 evictions, 0 input readbacks, 0 validation errors/warnings.

### Correction: the superseded `--quick` figures

The following table is exactly what the previous handoff published. It is kept
so the change is visible; it is **not** evidence for the release.

| Metric | Superseded value |
|---|---|
| `model.load` | 135.92 ms |
| `model.prepare` | 2.90 ms |
| `prefill.time_to_first_token` (10 tokens) | 271.77 ms (36.80 tokens/s) |
| `decode.per_token` | 179.74 ms (5.56 tokens/s) |
| `memory.peak_rss` | 165.67 MiB |
| `memory.device_peak` | unavailable (`None`, never 0) |
| plan misses per prefill / decode | 0 / 0 |

What was wrong with it:

1. **Profile not pinned in the evidence.** The table did not record the build
   profile it was measured on; the follow-up was tasked on the basis that it
   was a debug/`--quick` run. The new measurement shows something more useful
   than a profile correction: the CPU p50 across the two committed release runs
   is 240.31–288.31 ms prefill and 158.33–204.94 ms decode, and the superseded
   271.77 ms / 179.74 ms sit inside that range. Code generation was not the
   problem — the alpha CPU backend is dominated by per-step overhead. But
   without a recorded profile the old table could not distinguish the two, and
   a number that cannot say what binary produced it is not release evidence.
2. **`--quick` mode.** It shortened the sample count, published a median with
   no tail, and committed no per-iteration samples. There was no way to see
   the spread or to reproduce the percentile. The new run fixes 3 warmups and
   20 iterations, reports p50 and p95 with the rule, and commits all 20
   prefill and 60 decode samples per backend.
3. **Unmatched backends.** The table's CPU row came from `fn3-bench`; the
   Vulkan sentence in the same section ("164.6 ms prefill and 122 ms per
   cached decode step") came from the CLI, a 13-token prompt and a different
   device path. The two columns were never the same workload or the same
   harness, so the table did not compare CPU with Vulkan at all.
4. **`memory.device_peak` was an instrumentation gap, not a backend limit.**
   `tools/bench/adapters/fn3.rs` initialises
   `Runtime::peak_device_memory_bytes` to `None` and never assigns it. The
   backend does expose the value: `VulkanBackend::memory_stats()` returns
   `MemoryStats { peak_live_bytes, live_device_local_bytes, .. }`. The new test
   reads that accessor directly and reports 120,669,184 B; only the
   device-local split remains unavailable, for the reason named above.

What the new methodology guarantees: a recorded release profile, one concrete
backend per process, the same model, prompt, decode tokens, capacity, warmups
and iteration count on both sides; percentiles with a stated rule and the raw
samples committed; device time kept distinct from host wall time; counters as
ledger deltas; and every phase asserted to be 143/144 proven device executions
with zero input readbacks before any number is written.


## Packaging and clean-wheel smoke

`python3 tools/ci/packaging_lane.py` exited 0 and reported:

```text
wheel: fastnn-3.0.0a1-cp311-abi3-manylinux_2_39_x86_64.whl
fastnn.__version__ = 3.0.0a1
inference smoke: fastnn.fn3 imported, devices=['cpu']
prefill: tokens=10 cache=10 top-5=[46737, 28597, 16571, 7868, 13]
decode: 3 cached steps (640->32493, 11->11724, 612->612), cache=13
WHEEL INFERENCE: PASSED — top-5 [46737, 28597, 16571, 7868, 13]
```

The venv contained exactly:

```text
fastnn==3.0.0a1
numpy==2.5.5
pip==26.2.2
```

`numpy` is the only declared runtime dependency in `pyproject.toml`; the lane
fails if any other distribution is present. The import was resolved from
`target/ci-packaging/venv/lib/python3.14/site-packages/fastnn/__init__.py`, not
from the checkout, and `fn3-infer --version` reports `3.0.0-alpha.1`.

When the 28 MB model is absent, the smoke prints and records
`WHEEL INFERENCE: SKIPPED — pinned tiny-GPT-2 ONNX not present at <path>`, and
the packaging lane records the `wheel_inference` check as `not_run` with that
reason. A skip is never reported as a pass.

## Test and gate results

All commands below were run on the candidate SHA with `CARGO_BUILD_JOBS=12`.
No command was waived.

| Command | Result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo build` | ok |
| `cargo build --no-default-features` | ok |
| `cargo build --no-default-features --features vulkan` | ok |
| `cargo build --release --no-default-features --features cli` | ok |
| `cargo clippy --all-targets -- -D warnings` | **clean** |
| clippy, `--no-default-features` | **clean** |
| clippy, `--no-default-features --features vulkan` | **clean** |
| clippy, `--no-default-features --features cli` | **clean** |
| clippy, `--no-default-features --features fn3,simd,parallel,prepared-plan,fusion-forward` | **clean** |
| clippy, `--no-default-features --features fn3,vulkan` | **clean** |
| `cargo test --lib` | **673 passed** |
| `cargo test --no-default-features --features fn3 --bin fn3-bench` | **30 passed** |
| `fn3_ci_smoke` | 5 passed |
| `fn3_cli` | 6 passed |
| `fn3_contracts` | 43 passed |
| `fn3_cpu_reference` | 43 passed |
| `fn3_e2e` (CPU) | 5 passed |
| `fn3_e2e --features vulkan` (GTX 1080 Ti) | **10 passed** |
| `fn3_import` | 27 passed |
| `fn3_model_format` | 38 passed |
| `fn3_planner` | 11 passed |
| `fn3_session` | 8 passed |
| `fn3_tensor_storage` | 36 passed |
| `fn3_vulkan_smoke` (physical, validation) | 14 passed, 0 errors / 0 warnings |
| `fn3_vulkan_matmul` (physical) | 17 passed, 0 errors / 0 warnings |
| `fn3_vulkan_pointwise` (physical) | 18 passed, 0 errors / 0 warnings |
| `fn3_vulkan_attention` (physical) | 19 passed, 0 errors / 0 warnings |
| `uv run --python 3.13 pytest tests/ -m "not slow" -q` | **378 passed, 2 deselected** |
| `bash tools/ci/cpu-lane.sh` | PASSED (fmt, versions, clippy 0, no-default build, lib/contracts/smoke, tooling) |
| `bash tools/ci/vulkan-build-lane.sh` | PASSED (SPIR-V, feature build, clippy) |
| `bash tools/ci/vulkan-software-lane.sh` | PASSED, labelled CI-only/lavapipe |
| `python3 tools/ci/packaging_lane.py` | PASSED (wheel + clean-venv inference) |
| `python3 tools/ci/lane_status.py render` | exit 0; cpu, vulkan-build, vulkan-software-smoke and packaging PASSED; pinned-model and gpu-physical NOT VERIFIED by design |
| `python3 scripts/ci/check-agents-freshness.py` | checked 12 guides, clean |
| `python3 scripts/ci/check-versions.py --tag v3.0.0-alpha.1` | VERSIONS AGREE |

The 378 Python tests are the 373 from the #64 baseline plus 5 release-version
tests added here.

## Clippy debt closed

`cargo clippy --all-targets -- -D warnings` now passes in every configuration
listed above. `tools/ci/cpu-lane.sh` runs clippy with `-D warnings` and a
budget of **0**; `tools/ci/vulkan-build-lane.sh` runs the Vulkan-feature lint
as well. What changed:

- elidable lifetimes in `src/fn3/backend/cpu/kernels/mod.rs` and
  `src/fn3/model/session.rs`;
- `manual_implementation_of_is_multiple_of` in `src/fn3/storage.rs`,
  `src/fn3/backend/vulkan/kernels/transform.rs` and
  `src/fn3/backend/vulkan/submission.rs`;
- unnecessary closures/`ok_or_else` in `src/fn3/tensor.rs` and
  `src/fn3/backend/vulkan/kernels/matmul.rs`;
- `write_literal`, `bool_comparison`, `cloned_ref_to_slice_refs`,
  `useless_ptr_null_checks`, `needless_borrow`, `explicit_auto_deref`,
  `chunks_exact_to_as_chunks`, `identity_op`, `excessive_precision`,
  `len_zero`, `unnecessary_cast`, `collapsible_match`, `type_complexity` and
  one unused import across the fn3 and Vulkan tests;
- two dead-code functions used only by the Python bindings are now gated on
  `feature = "python"` instead of being compiled unused.

Two deliberate suppressions remain, both `#[allow(clippy::too_many_arguments)]`
on test-only reference helpers (`tests/fn3_vulkan_matmul.rs` and
`tests/fn3_vulkan_attention.rs`). Each mirrors the tensor operands one-for-one;
grouping them into a struct would hide which shape belongs to which tensor.
No numeric or API lint was suppressed.

`scripts/ci/check-agents-freshness.py` passes. No AGENTS.md contains a version
literal; the checker runs in the CPU lane.

## fn3-bench adapter

The seam #50 asked for is implemented behind the new `fn3` Cargo feature; the
`vulkan` and `cli` features and the `[[bin]] fn3-bench` entry are unchanged.
Public entry points in `tools/bench/adapters/fn3.rs`:

- `ModelManifest::from_path` — JSON manifest naming ONNX source, artifact,
  optional enforced byte lengths, device, capacity and the prompt/decode IDs;
- `load(&manifest) -> LoadedModel` — parse artifact, import ONNX, prove the two
  describe the same graph and every weight byte; no plan or weight tensor;
- `prepare(LoadedModel) -> PreparedModel` — open the device, build the planner
  and initial plan, materialise/retag weights;
- `PreparedModel::session() -> RunSession` — one session reused across
  `prefill` and `decode`; `reset` keeps plans, so repeated shapes are cache
  hits;
- `Runtime::last_device_time_ms() -> Option<f64>`,
  `Runtime::peak_device_memory_bytes() -> Option<u64>` — `None`, never zero,
  when the backend cannot report them;
- `Runtime::last_operation_counters() -> BTreeMap<String, u64>` — plan
  hits/misses/evictions, slot allocations/reuses/bytes, KV bytes and session
  counters;
- `PreparedModel::geometry()` and `PreparedModel::identity()` — schema geometry
  and model identity (FNV-1a checksums and byte counts for source and artifact);
- `BenchError` implements `Display` and names the failing stage.

Without the feature the adapter fails with the exact rebuild command, never
with baseline numbers under the fn3 name.

## Fix found while verifying

The #63 quickstart claimed full-model Vulkan execution was a "known alpha
blocker" that refused at plan instruction 1. On this candidate the same command
runs prefill and three cached decode steps on the GTX 1080 Ti (exit 0). The
quickstart, the changelog and the CLI's generic unsupported-op `next:` line
were corrected; the historical `docs/reports/fn3-17-cli-quickstart.md` was left
as the record of what #63 observed.

## Pending owner actions

**This session did none of these.** No remote write, tag, package upload, pull
request or merge was performed.

| Action | State | Notes |
|---|---|---|
| Review the candidate | **pending** | branch `fn3/67-alpha-release`, candidate `d0c7b00e66a981043c59328556fdf87fd5f84eb2` |
| Merge to `main` | **pending** | not done; `main` is untouched |
| Create annotated tag `v3.0.0-alpha.1` | **pending** | not created; tag the candidate (or the docs-only handoff commit) |
| Publish wheels | **pending** | no upload to GitHub Releases, PyPI or crates.io |
| Verify the published wheel | **pending** | download and rerun the clean-venv smoke after publication |
| Regenerate physical-GPU JSON evidence | optional | the fresh e2e run is cited above; `tools/ci/run-gpu-lane.sh` writes a new record if the owner wants one committed |

The release workflow accepts `v3.0.0-alpha.1` (prerelease tags are allowed) and
will re-run the version checker, CPU lane, portable builds, wheel builds and
the clean-venv smoke. macOS and Windows wheels were not built or tested here;
only Linux `x86_64` packaging is claimed.
