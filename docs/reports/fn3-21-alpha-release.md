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

Measured on the candidate with `fn3-bench` (`--suite model --quick`,
`CARGO_BUILD_JOBS=12`; host wall clock, median):

| Metric | Value |
|---|---|
| `model.load` | 135.92 ms |
| `model.prepare` | 2.90 ms |
| `prefill.time_to_first_token` (10 tokens) | 271.77 ms (36.80 tokens/s) |
| `decode.per_token` | 179.74 ms (5.56 tokens/s) |
| `memory.peak_rss` | 165.67 MiB |
| `memory.device_peak` | unavailable (`None`, never 0) |
| plan misses per prefill / decode | 0 / 0 |

The zero plan misses are the no-recompilation proof: the same prepared model
and one session are reused, prefill repeats at `(10, 0)` and every measured
decode runs at `past = 10`; an untimed reset/replay restores that shape. The
CLI on Vulkan measured 164.6 ms prefill and 122 ms per cached decode step for
the same prompt.

The benchmark records `last_device_time_ms() = None` for the CPU backend,
because this build has no CPU device clock; the schema marks the metric
unavailable with that reason instead of writing a zero.

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
