# FN3-20 (#66) — CI lanes, stale jobs, and the evidence contract

**Workstream:** CI · **Milestone:** alpha
**Branch:** `fn3/66-ci` · **Base:** `cfea81bb4f972fc3635a54c7a8f677b78b2d4716`
**Status of this report:** describes the committed files on `fn3/66-ci`; every
status below was observed locally on 2026-11-18 or is explicitly marked as not
run.

The point of this change is not more CI. It is that a compile check, a
software-Vulkan smoke run, and a physical-GPU run must never be presented as
the same evidence. The mechanism is `tools/ci/lanes.json` plus
`tools/ci/lane_status.py`: every lane is rendered in the run summary, a lane
with no record is **NOT VERIFIED**, and a lane that applies to the current tree
fails the run unless it is **PASSED**. See `tools/ci/README.md` for the
operating contract.

## 1. Lane table

| Lane | Jobs | Proves | Explicitly does NOT prove | Status on 2026-11-18 |
|---|---|---|---|---|
| `cpu` | `cpu-rust`, `cpu-python` | fmt; clippy within a documented budget with zero `src/fn3/` warnings; `--no-default-features` build; lib + fn3 contract + fn3 CI-smoke tests; Python suite `-m "not slow"` | Nothing about Vulkan or any GPU. The pytest suite currently exercises the frozen 2.x facade, not a 3.0 surface. | **PASSED** locally (`583` lib, `43` contract, `5` smoke; `369 passed, 1 skipped, 2 deselected, 1 xfailed` pytest) |
| `frozen-2x` | `frozen-2x` | The frozen 2.x tree still compiles and its targeted suites pass while `src/fn3/` coexists — the only 2.x gate the architecture keeps | It is **not** 3.0 evidence; training/autograd/quantization/wgpu behaviour says nothing about the inference core. Deleted by #65. | **PASSED** locally (quantized, optimizer, autograd-gradient, arena-telemetry and full `--tests` suites); required in CI |
| `vulkan-build` | `vulkan-build` | Every committed `.spv` validates with `spirv-val`; shader sources recompile byte-identically through `spike/vk-bringup/scripts/compile-shaders.sh`; once the feature exists, the crate builds with `--features vulkan` | A build is not execution: no instance/device, no dispatch, no validation layer. SPIR-V validation is not driver acceptance or numerical correctness. | **NOT VERIFIED** — SPIR-V and reproducibility passed; the `vulkan` feature does not exist (#57) |
| `vulkan-software-smoke` | `vulkan-software` | Under `VK_ICD_FILENAMES=…/lvp_icd.json` with validation enabled, the Vulkan smoke test runs on lavapipe | **CI-ONLY.** Lavapipe is a CPU implementation. It proves nothing about a physical GPU, driver, limits or performance, and must never be cited as GPU evidence. | **NOT VERIFIED** — “Vulkan backend not yet implemented (#57)”; lavapipe precondition passed |
| `packaging` | `packaging` (ubuntu/macos/windows) | A wheel builds, installs into a clean venv, imports from outside the checkout, reports its version and runs a minimal inference smoke | Rust tests, model loading, GPU support, manylinux compliance. The smoke surface is the frozen 2.x facade until #64. | **PASSED** locally on Linux (`fastnn-2.6.0-cp311-abi3-manylinux_2_39_x86_64.whl`); macOS/Windows are CI-only |
| `pinned-model` | `pinned-model` | With `tools/ci/model-pin.json` pinned, the E2E fixtures named by that single file run offline against exactly that revision | An unpinned/missing pin is NOT VERIFIED, never a pass. A pass proves numerical agreement on the pinned fixtures only. | **NOT VERIFIED** — pin is `unpinned`, fixtures owned by #49 |
| `gpu-physical` | `gpu-evidence` (validation only — no GPU) | A validated `tools/ci/run-gpu-lane.sh` record names a physical device, UUID, driver, API version, validation result, pass/fail and timings | Hosted CI never runs this lane and has no GPU. Software records are rejected. A record proves only the named device/driver. | **NOT VERIFIED** — backend absent (#57); a truthful local record exists and validates structurally |

`lane_status.py resolve` marks `cpu`, `frozen-2x` and `packaging` required today.
`vulkan-build` and `vulkan-software-smoke` become required automatically when
`Cargo.toml` declares the `vulkan` feature (and, for the software lane, when
`tests/fn3_vulkan_smoke.rs` exists). `pinned-model` becomes required when the
pin status is `pinned`. No `required_when` condition can make `gpu-physical`
apply to a hosted runner: its lane entry has `hosted: false`.

## 2. Stale-job inventory (base SHA workflows)

Every job that existed at `cfea81b` is listed. “Replacement” means the lane or
job that now carries the surviving intent; “gap” means nothing covers it yet.

### `.github/workflows/ci.yml`

| Job | Still applies under inference-only 3.0? | Replacement / action |
|---|---|---|
| `quality` | Partly. `cargo fmt --check` and `cargo test --lib` apply. `clippy --all-targets -- -D warnings` **already fails at the base SHA** (unused `TensorView` import in `tests/fn3_contracts.rs`, owned by #48), and `scripts/ci/check-agents-freshness.py` **also fails at the base SHA** (it rejects the version literal “3.0” in `src/fn3/AGENTS.md`). Neither is a usable gate today. The quantized/optimizer/autograd targets are 2.x-only. | Split into `cpu-rust` (fmt, clippy budget, build, lib, fn3 tests) and `frozen-2x` (2.x suites + `--tests`). The freshness script is not run by the new CPU lane until its false positive is fixed (owned by #48/#65). `release_error_paths` is covered by `cargo test --tests` in the frozen lane. |
| `feature-profiles` (`core`, `runtime`, `gpu`) | **No.** `simd`/`parallel`/`fusion-forward`/`prepared-plan`/`cli` are 2.x feature policy; `gpu` is wgpu, which 3.0 removes. | Removed. `cpu` uses `--no-default-features`; `vulkan-build` owns the only accelerated feature. Do not re-add `--all-features` or `--features gpu`. |
| `python` (3 OSes) | Only as the frozen 2.x facade. The pinned `onnx==1.17.0` / `onnxruntime==1.20.1` stack is 2.x test infrastructure. | `cpu-python` on Linux preserves the suite until #65; `packaging` covers wheel install on 3 OSes. #64 rewrites the suite and #67 the smoke. |
| `aarch64` | **No as written**: builds with default features (incl. `python`) and a 2.x feature list. Cross-compilation itself is still meaningful. | **Gap.** Removed in this change. #65/#67 should add `cargo build --target aarch64-unknown-linux-gnu --no-default-features` as part of the portable lane. |
| `coverage` | **No as written**: `--features "simd,parallel,fusion-forward"` is 2.x. | **Gap.** Removed. Re-add against the promoted 3.0 tree after #65, with a lane entry. |

### `.github/workflows/release.yml`

| Job | Still applies under inference-only 3.0? | Replacement / action |
|---|---|---|
| `validate-tag` | Needed, but the regex `^v[0-9]+\.[0-9]+\.[0-9]+$` **rejects every prerelease**, so `v3.0.0-alpha.1` could never release. | Regex now accepts `-PRERELEASE` suffixes. Version checks (Cargo/pyproject/`__init__`/CHANGELOG) are unchanged; #64 must update them if the Python facade moves. |
| `quality` | Same 2.x gates as `ci.yml/quality`. | Now calls `tools/ci/cpu-lane.sh` and `tools/ci/frozen-2x-lane.sh`; the lane scripts are deleted/updated with #65. |
| `portable-build` | **No as written**: `--features "simd,parallel,fusion-forward"` is the 2.x profile. | `cargo build --release --no-default-features`. A 3.0 portable profile (e.g. with `vulkan` on supported targets) is #67’s decision. |
| `build-wheels` | Build applies; the inline smoke **breaks after #65** because it calls `fastnn.AotExecutor` and `fastnn.tensor`, which the 3.0 inference core does not keep. It also installs `libatlas-base-dev`, which is not the OpenBLAS package the build actually uses. | Build stays with `maturin-action` (manylinux); install/smoke now `tools/ci/packaging_lane.py --skip-build`. `wheel_smoke.py` names the surface it tests and #64 must replace the 2.x branch. |
| `create-release` | Applies. | Unchanged except `prerelease: ${{ contains(github.ref_name, '-') }}` so alpha tags are published as prereleases. |

### What would fail purely because training/wgpu was deleted (#65)

- `feature-profiles` with `gpu` and the 2.x feature names: `cargo check` fails
  on unknown features or missing wgpu modules.
- `coverage` and `portable-build`: unknown `simd`/`parallel`/`fusion-forward`
  features.
- `quality` / `frozen-2x` / release `quality`: `cargo test --tests` includes
  `tests/training.rs`, `tests/wgpu.rs`, `tests/wgpu_quantized.rs`, and the
  autograd/optimizer suites. These must be deleted from the CI command list in
  the same change that deletes the files.
- Release `build-wheels`: the inline smoke uses `AotExecutor`; it would fail
  against a 3.0 wheel even though packaging itself is fine.
- `python` job: imports the 2.x facade modules that #65 removes.

The lane table makes the last two visible instead of silent: `frozen-2x` is
marked transitional/removed-by-#65, and `packaging` says the smoke surface is
the 2.x facade.

## 3. Commands actually run and outcomes

All Cargo commands used `CARGO_BUILD_JOBS=4`; logs for the lane runs are in the
referenced status records.

| Command | Outcome |
|---|---|
| `git rev-parse --abbrev-ref HEAD` / `git rev-parse HEAD` | `fn3/66-ci` / `cfea81bb4f972fc3635a54c7a8f677b78b2d4716` (before editing) |
| `cargo fmt --check` | exit 0 |
| `cargo clippy --all-targets -- -D warnings` | **FAILED at base**: unused import `TensorView` in `tests/fn3_contracts.rs:14` (#48-owned) |
| `cargo clippy --all-targets` | 4 warnings, all pre-existing: 3 in `src/backend/{cpu,executor}.rs` (2.x), 1 in `tests/fn3_contracts.rs`; **0 in `src/fn3/` or the new test** |
| `python3 scripts/ci/check-agents-freshness.py` | **FAILED at base**: `src/fn3/AGENTS.md: volatile version literal '3.0'` (#48-owned; the new lane does not hide it) |
| `cargo build --no-default-features` | success (manifest note: `fnv` unused in this profile) |
| `cargo test --lib -- --test-threads=1` | **583 passed**, 0 failed |
| `cargo test --test fn3_contracts` | **43 passed**, 0 failed |
| `cargo test --test fn3_ci_smoke` | **5 passed**, 0 failed |
| `tools/ci/cpu-lane.sh` | **PASSED** (fmt, clippy budget, build, lib, contracts, smoke, CI-tool selftests) |
| `tools/ci/python-lane.sh` | **PASSED**: `369 passed, 1 skipped, 2 deselected, 1 xfailed` in 23.16 s (maturin develop) |
| `tools/ci/frozen-2x-lane.sh` | **PASSED**: quantized pipeline, optimizer, autograd gradients, arena telemetry, and `cargo test --tests` |
| `python3 tools/ci/model_pin.py selftest` | 5 cases passed |
| `python3 tools/ci/lane_status.py selftest` | passed |
| `tools/ci/vulkan-build-lane.sh` | **NOT VERIFIED**: `spirv-val` passed for `spike/vk-bringup/shaders/fma.comp.spv`; recompile byte-identical; no `vulkan` feature (#57) |
| `tools/ci/vulkan-software-lane.sh` | **NOT VERIFIED**: lavapipe enumerated; backend not implemented (#57) |
| `tools/ci/model-pin-lane.sh` | **NOT VERIFIED**: `tools/ci/model-pin.json` status `unpinned` (#49) |
| `tools/ci/gpu-evidence-status.sh` | **NOT VERIFIED**: no committed physical-GPU record; hosted runners have no GPU |
| `python3 tools/ci/packaging_lane.py` | **PASSED**: built and installed `fastnn-2.6.0-cp311-abi3-manylinux_2_39_x86_64.whl`, smoke from `/tmp` |
| `tools/ci/run-gpu-lane.sh --out …` | wrote a truthful record; result `not_verified` (#57); `verify_evidence.py` accepts its structure; exit 69 |
| `python3 -c "import yaml …" .github/workflows/*.yml` | both workflows parsed |
| `actionlint 1.7.7 .github/workflows/*.yml` | exit 0 |

The full `frozen-2x` lane ran locally and passed; the base-SHA pass counts for
its suites are in `docs/reports/fn3-01/baseline-2.6.0.md`.

A combined render of every lane record produced by the local runs above exits 0
with `cpu`, `frozen-2x` and `packaging` **PASSED**, and `vulkan-build`,
`vulkan-software-smoke`, `pinned-model` and `gpu-physical` **NOT VERIFIED** for
the reasons in section 1. The gate fails only if a required lane is not passed;
no non-required lane is ever counted as success.

## 4. YAML validation

```bash
python3 -c "import yaml,sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]" .github/workflows/*.yml
# PyYAML: parsed 2 workflow files

actionlint -no-color .github/workflows/ci.yml .github/workflows/release.yml
# exit 0
```

`actionlint` initially rejected a workflow-level `runner.temp` expression; the
status directory now uses `${{ github.workspace }}/target/ci-lane-status`,
which is valid at workflow scope and gitignored.

## 5. GPU absence is mechanical

`tools/ci/lane_status.py render` enumerates `lanes.json`, not the jobs that
happened to report. A lane with no record is printed as **NOT VERIFIED** with
the reason; if its `required_when` condition is true, the summary job exits 1.
The `gpu-physical` row therefore always appears, and always says NOT VERIFIED
unless a validated physical record exists. There is no code path that maps a
missing GPU run to success.

The hosted `gpu-evidence` job never creates a Vulkan device. It only validates
committed records. The local script refuses to run when `GITHUB_ACTIONS=true`,
so a maintainer cannot accidentally produce evidence on a hosted runner.

## 6. Physical-GPU evidence record

`tools/ci/run-gpu-lane.sh` writes a JSON record with:

- device name, canonical UUID, `deviceType`, vendor/device id, driver
  name/version, Vulkan device and instance API versions;
- validation layer, error/warning counts, and the source of the validation run;
- per-check results (`fastnn_vulkan_smoke` is the only check that can make the
  record `passed`; a device probe alone cannot);
- timings, git branch/HEAD/dirty flag, and the exact commands;
- an `auxiliary` section for the raw `spike/vk-bringup` run, explicitly labelled
  “NOT fastnn backend evidence”.

`tools/ci/verify_evidence.py` rejects CPU-typed devices, lavapipe/llvmpipe/
SwiftShader names, missing UUID/driver/API fields, non-clean validation, a
`passed` record without a passed `fastnn_vulkan_smoke` check, and any
`ci_only: true` record. A committed record with `result=not_verified` is valid
to keep (it is truthful) but never satisfies the lane.

Local record from the GTX 1080 Ti (committed as
`docs/reports/fn3-20/gpu-evidence/gtx1080ti-20261118.json`, generated on a clean
tree at `93fe607`; `verify_evidence.py --require-pass` rejects it because the
backend is absent):

| Field | Value |
|---|---|
| device | NVIDIA GeForce GTX 1080 Ti, `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` |
| UUID | `471b2bda-51e1-6451-e692-ff5d1e146f1e` |
| driver | NVIDIA `580.196.07` |
| Vulkan API | `1.4.312` (device and instance) |
| validation | `VK_LAYER_KHRONOS_validation`, 0 errors (auxiliary bring-up run) |
| fastnn smoke | `not_run` — no `vulkan` feature, no `tests/fn3_vulkan_smoke.rs` (#57) |
| result | `not_verified`, with the reason naming #57 |

The hosted `gpu-evidence` job validates that committed record and reports
**NOT VERIFIED** — it does not turn a truthful “not implemented” record into a
green GPU claim.

## 7. Skipped checks, risks, and handoff

### Skipped in this worktree

- `frozen-2x` full suite (allowed: baseline covers the counts). CI runs it.
- macOS and Windows packaging (CI matrix). Only Linux was executed locally.
- `vulkan` feature build and both Vulkan execution lanes: the feature does not
  exist until #57. They report NOT VERIFIED.
- Pinned-model E2E: the pin and fixtures belong to #49; the lane reports NOT
  VERIFIED.

### Known risks

1. **Clippy is not denied globally.** The base SHA has 4 pre-existing warnings
   (3 in 2.x, 1 in #48’s test). The lane forbids *any* `src/fn3/` warning and
   enforces a total budget of 4. #65 must set the budget to 0 and `-D warnings`.
2. **The AGENTS freshness check is red at the base SHA.** It rejects the “3.0”
   literal in `src/fn3/AGENTS.md`; the new CPU lane does not run it, so the
   failure is reported here rather than quietly inherited. #48/#65 must fix the
   script or the guide, then add it back to `cpu-rust`.
2. **ADR: `vulkan-software-smoke` is green while NOT VERIFIED.** That is
   intentional: the job finishes and records an honest status. Branch protection
   must require **`lane-summary`**, not the individual lane jobs, or a skipped
   lane could be mistaken for success.
3. **Shader toolchain drift.** `vulkan-build` recompiles with the host `glslc`
   and requires byte-identical `.spv`/`.spvasm`. A different shaderc version may
   legitimately produce different bytes; the lane then fails loudly and the
   toolchain must be pinned (container) rather than weakening the check. The
   spike script also exits 141 when its `glslc --version | head -n 1` banner
   hits SIGPIPE; the lane distinguishes that exact case from a compile failure
   and records a warning. #57 should fix the script.
4. **Packaging is not manylinux-audited in CI.** `packaging` builds on the host
   and proves install/import/smoke only. `release.yml` still uses
   `maturin-action` with `manylinux: auto` for the publishable wheel; #67 owns
   auditwheel/manylinux evidence.
5. **`wheel_smoke.py` still has a 2.x branch.** It is labelled in output, but
   #64 must make the 3.0 API the only branch.
6. **`model-pin.json` is committed in the `unpinned` state.** The checker treats
   missing, unpinned, invalid-revision and manifest-hash-mismatch as 69/1, so a
   placeholder cannot silently become evidence.
7. **No aarch64 and no coverage job.** Both were removed as stale 2.x feature
   builds; see the gaps above.

### What #62 (E2E) must add

- Publish `fixtures/` plus a manifest; set `tools/ci/model-pin.json` to
  `status: "pinned"` with a 40-hex commit and run
  `python3 tools/ci/model_pin.py seal` to record the manifest hash.
- Add `tests/fn3_e2e.rs` (or `tests/fn3_e2e/main.rs`). The `pinned-model` lane
  then becomes required automatically and runs with
  `HF_HUB_OFFLINE=1 CARGO_NET_OFFLINE=1`.
- Make the E2E output name the fixture hashes it used and fail if the pin’s
  manifest hash does not match.

### What #67 (packaging) must add

- Replace the 2.x branch in `tools/ci/wheel_smoke.py` with the 3.0 Python/CLI
  surface (#63/#64) and assert the CLI binary is absent from the wheel unless
  intended.
- Add manylinux/auditwheel (or equivalent) evidence and an sdist lane; keep the
  three-OS matrix.
- Re-add the removed `aarch64` portable build and, if coverage returns, add a
  lane entry with a 3.0 target rather than a 2.x feature list.
- When #65 deletes the 2.x tree, delete the `frozen-2x` lane/job, set
  `CLIPPY_MAX_WARNINGS=0`, enable `-D warnings`, and remove the 2.x commands
  from `release.yml`.

### What #57 (Vulkan) must add for these lanes

- Declare `feature = "vulkan"` and add `tests/fn3_vulkan_smoke.rs`. Both Vulkan
  lanes then become required without a workflow edit.
- The smoke test must print the chosen device and, under lavapipe, make the
  CI-only lane’s output contain `lavapipe`/`llvmpipe`; the software lane fails
  if it does not, so it cannot silently use a physical GPU.
- Provide a validation summary (or VUID-free output) so `run-gpu-lane.sh` can
  fill the evidence record’s validation block from the fastnn test rather than
  the auxiliary spike.
- Fix `spike/vk-bringup/scripts/compile-shaders.sh`’s SIGPIPE exit so the
  `vulkan-build` warning can be removed.
