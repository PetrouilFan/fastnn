# FastNN 3.0 coordination ledger

Tracking issue: #46. This file is the orchestrator's working ledger for the local
3.0 rewrite. It records ownership, worktree/branch/state per issue and the exact
evidence links. A GitHub issue being open or closed is not execution evidence.

## Frozen references

| Item | Value |
|---|---|
| Baseline SHA | `a2c0759fff8bdcb7c943e53442f3382b86e59b66` (2.6.0, post-PR #45) |
| Baseline `origin/main` | identical to baseline; no delta to account for |
| Integration branch | `rewrite/fastnn-3` |
| Integration worktree | `/home/petrouil/Projects/github/fastnn-wt-integrate` |
| Read-only 2.x reference | `/home/petrouil/Projects/github/fastnn-wt-baseline` (detached at baseline) |
| Original clone | `/home/petrouil/Projects/github/fastnn` (branch `main`, clean) |
| Remote writes | disabled; no push, tag, package or issue transition without owner approval |

## Verified host environment

| Item | Value |
|---|---|
| OS | Arch Linux (rolling), kernel-level toolchain current |
| CPU | 16 logical cores, 31 GiB RAM, 212 GiB free on `/home` |
| Rust | rustc/cargo 1.100.0 |
| Python | 3.14.9, uv 0.12.19 |
| Physical GPU | NVIDIA GeForce GTX 1080 Ti (GP102, deviceID `0x1b06`), driver 580.196.07 |
| Physical Vulkan | API 1.4.312 reported by device, instance 1.4.363 |
| Validation layers | `VK_LAYER_KHRONOS_validation` 1.4.363 installed |
| Software ICD | Mesa lavapipe `llvmpipe (LLVM 23.1.4, 256 bits)`, `lvp_icd.json` |
| Shader toolchain | `glslangValidator`, `glslc`, `spirv-val`, `spirv-dis` present |
| Python ML stack | torch / onnx / onnxruntime absent at start; to be provisioned by FN3-03 |

Concurrency policy: at most 5 worker sessions; each build uses
`CARGO_BUILD_JOBS=4`; physical-GPU benchmarks are serialized one at a time;
no heavy builds during a measured run.

## 3.0 architecture frame (integrator decision, FN3-02 owns the ADR)

The 3.0 inference core lands in a new top-level module tree so parallel workers
never collide with 2.x paths, and FN3-19 promotes/renames it at cleanup:

```text
src/fn3/                 # 3.0 inference core, always compiled
  dtype.rs  shape.rs  error.rs  device.rs
  storage.rs  tensor.rs
  backend/   mod.rs (op id registry, Backend trait, capability)
             cpu/...
             vulkan/...  (feature = "vulkan")
  model/     weights.rs  plan.rs  session.rs
  ops/       reference implementations and op-id constants
```

- Vulkan binding: `ash` (raw Vulkan loader), shaders authored in GLSL and
  compiled offline by a script into committed SPIR-V, validated with `spirv-val`.
- No autograd, no training, no wgpu on the 3.0 path.
- Precursor: FN3-05 validates its bring-up in a standalone spike crate
  (`spike/`) that is not a workspace member, so no root manifest conflicts.

## Ownership of shared surfaces

| Surface | Owner |
|---|---|
| `src/fn3/**` public types, op ids, error enum, capability model | FN3-02 (#48) |
| Root `Cargo.toml` `[features]`/`[dependencies]` | integrator; FN3-02 proposes, FN3-11 executes after review |
| Root `Cargo.toml` `[[bench]]`/`[[bin]]`/`[dev-dependencies]` | FN3-04 (#50) |
| `src/lib.rs` module declarations | FN3-02 (#48), then integrator |
| `spike/**` | FN3-05 (#51) |
| `fixtures/**`, `tools/parity/**` | FN3-03 (#49) |
| `docs/reports/**` | FN3-01 (#47), later FN3-29 |
| Vulkan dispatch/lifetime infrastructure | FN3-11 (#57) |
| Kernel modules under `src/fn3/backend/vulkan/kernels/**` | FN3-13/14/15 (#59/#60/#61) |

Any worker needing to edit a surface it does not own must ask the orchestrator
first. Shared manifests are edited by the integrator or by explicit delegation.

## Active worker sessions (wave 1)

| Issue | Agent session id | Worktree |
|---|---|---|
| #47 | `ses_e11c3d23affewonQSKKz1QOB3G` | `fastnn-wt-47-baseline` |
| #48 | `ses_e11c49331ffe8v3liiFWrN7Pkj` | `fastnn-wt-48-core` |
| #49 | `ses_e11c49330ffeXZmLdCMPyUPUox` | `fastnn-wt-49-parity` |
| #50 | `ses_e11c3d23affd9JofWOLwdRo6nZ` | `fastnn-wt-50-bench` |
| #51 | `ses_e11c3467dffeSkQXFcmJWoZRxq` | `fastnn-wt-51-vulkan` |

Critical path to the alpha end-to-end gate (#62):
`#48 → #52 → {#53 → #56 → #58}, {#54 → #55}` and `#51 → #57 → {#59,#60,#61}`.

## Issue ledger

Statuses: ready, in progress, review, integrated, verified, blocked, deferred.

| Issue | Workstream | Status | Branch | Worktree | Base SHA | Next action |
|---|---|---|---|---|---|---|
| #46 | Coordination | in progress | `rewrite/fastnn-3` | `fastnn-wt-integrate` | `a2c0759` | dispatch wave 1 |
| #47 | Baseline | dispatched | `fn3/47-baseline` | `fastnn-wt-47-baseline` | `a2c0759` | baseline report |
| #48 | Core | dispatched | `fn3/48-core-contract` | `fastnn-wt-48-core` | `a2c0759` | contracts + ADR |
| #49 | Correctness | dispatched | `fn3/49-parity` | `fastnn-wt-49-parity` | `a2c0759` | pinned fixtures |
| #50 | Performance | dispatched | `fn3/50-bench` | `fastnn-wt-50-bench` | `a2c0759` | harness + schema |
| #51 | Vulkan platform | dispatched | `fn3/51-vulkan` | `fastnn-wt-51-vulkan` | `a2c0759` | device bring-up |
| #52 | Core storage | ready | — | — | — | after #48 |
| #53 | CPU reference | ready | — | — | — | after #48, #52 |
| #54 | Model import | ready | — | — | — | after #48, #49, #52 |
| #55 | Model format | ready | — | — | — | after #48, #52, #54 |
| #56 | Runtime planner | ready | — | — | — | after #48, #52, #53, #54 |
| #57 | Vulkan dispatch | ready | — | — | — | after #48, #51, #52 |
| #58 | Session runtime | ready | — | — | — | after #48, #52, #53, #56 |
| #59 | GPU pointwise | ready | — | — | — | after #49, #53, #57 |
| #60 | GPU matmul | ready | — | — | — | after #49, #53, #57 |
| #61 | GPU attention | ready | — | — | — | after #49, #53, #57 |
| #62 | End-to-end | ready | — | — | — | after #49, #54-#61 |
| #63 | CLI/docs | ready | — | — | — | after #54, #55, #58 |
| #64 | Python | ready | — | — | — | after #48, #55, #58 |
| #65 | Cleanup | ready | — | — | — | after #62, #64 |
| #66 | CI | ready | — | — | — | after #48 |
| #67 | Alpha release | ready | — | — | — | after #50, #62-#66 |
| #68 | Quant format | deferred (stable) | — | — | — | after #62 |
| #69 | CPU quant | deferred (stable) | — | — | — | after #68 |
| #70 | GPU quant | deferred (stable) | — | — | — | after #68 |
| #71 | CPU perf | deferred (stable) | — | — | — | after #62 |
| #72 | GPU perf | deferred (stable) | — | — | — | after #62 |
| #73 | Hardening | deferred (stable) | — | — | — | after #62 |
| #74 | Hardware QA | deferred (stable) | — | — | — | after alpha |
| #75 | Docs/evaluation | deferred (stable) | — | — | — | after alpha |
| #76 | Stable gate | deferred (stable) | — | — | — | after stable work |
| #77 | CUDA | deferred | — | — | — | owner activation required |
| #78 | Training | deferred | — | — | — | owner activation required |

## Alpha gates

1. FP32 prefill and several cached decode steps, CPU and named physical Vulkan device.
2. Logits compared to an independent reference; cache bounds and session isolation validated.
3. Actual backend execution recorded; explicit device requests error instead of silently using CPU.
4. Python facade and CLI share the same Rust path.
5. Training, wgpu and obsolete paths removed after gates 1-4.
6. Package versions `3.0.0-alpha.1` / `3.0.0a1` install and smoke.

## Stable gates (deferred)

One tested quant format with CPU and Vulkan execution on real hardware, measured
performance work against baseline, hardened error paths, documented hardware
matrix, and clean install verification.

## Integration log

| SHA | Issue | Summary | Checks |
|---|---|---|---|
| `a2c0759` | — | integration base = 2.6.0 baseline | n/a |
