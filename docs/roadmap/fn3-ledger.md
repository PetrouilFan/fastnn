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

## Worker sessions

| Issue | Agent session id | Worktree | State |
|---|---|---|---|
| #47 | `ses_e11c3d23affewonQSKKz1QOB3G` | `fastnn-wt-47-baseline` | integrated |
| #48 | `ses_e11c49331ffe8v3liiFWrN7Pkj` | `fastnn-wt-48-core` | integrated |
| #49 | `ses_e11c49330ffeXZmLdCMPyUPUox` | `fastnn-wt-49-parity` | integrated |
| #50 | `ses_e11c3d23affd9JofWOLwdRo6nZ` | `fastnn-wt-50-bench` | running |
| #51 | `ses_e11c3467dffeSkQXFcmJWoZRxq` | `fastnn-wt-51-vulkan` | integrated |
| #52 | `ses_e11a5ebdcffe7Ps0pyCu4gkYTm` | `fastnn-wt-52-storage` | running |
| #57 | `ses_e11a5092bffeTKXdnjGr7z17wa` | `fastnn-wt-57-vulkan` | running |
| #66 | `ses_e11a5ebdbffeKwNXwAPT9SHX6a` | `fastnn-wt-66-ci` | running |

Critical path to the alpha end-to-end gate (#62):
`#48 → #52 → {#53 → #56 → #58}, {#54 → #55}` and `#51 → #57 → {#59,#60,#61}`.
With #48, #49 and #51 landed, every remaining alpha issue is gated on **#52**.

## Issue ledger

Statuses: ready, in progress, review, integrated, verified, blocked, deferred.

| Issue | Workstream | Status | Branch | Worktree | Base SHA | Next action |
|---|---|---|---|---|---|---|
| #46 | Coordination | in progress | `rewrite/fastnn-3` | `fastnn-wt-integrate` | `a2c0759` | dispatch wave 1 |
| #47 | Baseline | verified (integrated) | `fn3/47-baseline` | `fastnn-wt-47-baseline` | `a2c0759` | done; perf + salvage reference |
| #48 | Core | verified (integrated `cfea81b`) | `fn3/48-core-contract` | `fastnn-wt-48-core` | `a2c0759` | done; unblocks #52/#57/#66 |
| #49 | Correctness | verified (integrated `b335146`) | `fn3/49-parity` | `fastnn-wt-49-parity` | `a2c0759` | done; tolerances for #53 |
| #50 | Performance | dispatched | `fn3/50-bench` | `fastnn-wt-50-bench` | `a2c0759` | harness + schema |
| #51 | Vulkan platform | verified (integrated `5581349`) | `fn3/51-vulkan` | `fastnn-wt-51-vulkan` | `a2c0759` | done; feeds #57 |

### #47 finding that changes the alpha story

`a2c0759` is `v2.6.0-71`; the roadmap's 5.96e-7 tiny-GPT-2 parity claim was
measured at `v2.6.0` and **does not reproduce at the integration baseline**.
At baseline the stock dynamic artifact fails to build (200 errors, first is
unsupported symbolic division in `Reshape`) and the static variant produces
all-NaN logits for any multi-token prefill (layer 1 K/V is already NaN).
3.0 therefore has no working 2.x reference path to inherit; #53 must be proven
against the #49 ORT/PyTorch fixtures, not against 2.x output. wgpu is also
broken at baseline (10 `wgpu` + 4 `wgpu_quantized` failures), which independently
justifies #65 removing it rather than porting it.

### #49 evidence (independently reproduced by integrator)

`python3 tools/parity/verify_fixtures.py --root fixtures/parity` on system
Python 3.14.9 with no ML packages installed: `OK: 3 suites, 169 tensors
(85 stored, 1415488 bytes)`. Worker's fresh-venv / fresh-`HF_HOME`
regeneration was byte-identical (`diff -r` empty) across 31 SHA-pinned upstream
files. Committed payloads are real binary blobs (e.g.
`fixtures/parity/tiny-gpt2/tensors/decode.step1.logits.f32` is 201,028 B =
50,257 x 4), not LFS pointers.

Contracted parity inputs for #53+ (do not re-derive):

- CI model `onnx-community/tiny-random-gpt2-ONNX` @ `90f61e71d6fa...d52f`,
  27,897,974 B, hidden 128, 2 layers, 2 Q/KV heads, head dim 64, `gelu_new`
  (tanh GELU), learned positions, LayerNorm eps 1e-5. **No declared license:
  never redistribute the weights.**
- Trained `onnx-community/gpt2-ONNX` @ `1975fb9b4a2a...fde0`, 498,126,358 B,
  12 layers, hidden 768; base `openai-community/gpt2` is MIT. `"ci": false`,
  opt-in via `tools/parity/fetch_models.py --suite trained-gpt2`.
- Tolerance table and failure predicates live in
  `fixtures/parity/README.md`; tiny logits atol 1e-5 / rtol 1e-4, tiny KV
  1e-5 / 1e-4, sequence scaling `atol *= sqrt(S/16)` for `S > 16`.
- All-masked attention rows are finite `-FLT_MAX` producing uniform
  `[0.25, 0.25, 0.25, 0.25]`. A NaN here is a conformance failure, not a
  tolerance question.

### #51 evidence (independently reproduced by integrator)

`spike/vk-bringup/` — standalone crate (`ash` 0.38.0), no root manifest edits.
Physical NVIDIA GTX 1080 Ti selected by UUID, `--validation`:

```text
verify: max_abs_error=0 exact_matches=262147/262147
repeat: submissions=100 samples=300 sample_mismatches=0
handles: buffer 6/6, device_memory 6/6 (8388736 B), pipeline 1/1, fence 1/1
validation summary: errors=0 warnings=0
result: PASS
```

Logged evidence: `spike/vk-bringup/logs/physical-gtx1080ti-validation.log`;
software lane `logs/lavapipe-ci-only.log` is labelled CI-only and does not
satisfy the physical-device gate. Failure paths verified with verbatim
actionable messages (`logs/no-icd.log`, `bad-device-index.log`,
`bad-device-name.log`, `missing-extension.log`, `missing-loader.log`).

Device limits that constrain later kernels: `maxMemoryAllocationSize` ≈ 4 GiB
(GTX 1080 Ti heap is 11 GiB — suballocation is mandatory), shared memory
49152 B, invocations 1536, `maxStorageBufferRange` 4294967295 B, storage
offset alignment 16 B, uniform offset alignment 256 B, timestampPeriod 1 ns.
Per-dispatch fences and per-tensor allocations do not scale; design note for
#57 is `spike/vk-bringup/NOTES-FN3-11.md`.
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
| `6d48abd` | #46 | ledger created | n/a |
| `5581349` | #51 | Vulkan bring-up spike merged (`a1e6517`) | re-ran on GTX 1080 Ti: exact match 262147/262147, 0 validation errors |
| `c5cadc8` | #46 | ledger: #51 limits + evidence | n/a |
| `e45a337` | #47 | baseline report + salvage list merged | independent re-read of report; baseline test matrix green |
| `cfea81b` | #48 | `src/fn3` contracts merged (`027ac49`) | fmt, 43 contract tests, 583 lib tests, build with and without default features |
| `b335146` | #49 | parity fixtures + verifier merged (`d7d8db0`) | verifier OK 169 tensors/1,415,488 B; fmt; 43 contract tests still green |
