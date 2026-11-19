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
| #50 | Performance | verified (integrated `301e5a7`) | `fn3/50-bench` | `fastnn-wt-50-bench` | `a2c0759` | done; fn3 adapter is a #62/#67 seam |
| #51 | Vulkan platform | verified (integrated `5581349`) | `fn3/51-vulkan` | `fastnn-wt-51-vulkan` | `a2c0759` | done; feeds #57 |
| #52 | Core | verified (integrated `830f315`) | `fn3/52-storage` | `fastnn-wt-52-storage` | `cfea81b` | done |
| #53 | CPU reference | verified (integrated `89dabac`) | `fn3/53-cpu-reference` | `fastnn-wt-53-cpu` | `830f315` | done; numerical oracle for all GPU work |
| #54 | Model import | verified (integrated `5a055e1`) | `fn3/54-onnx-import-b` | `fastnn-wt-54b-import` | `27fc60d` | done; 27 import tests on the real model, 265 fn3 nodes, exact inventory |
| #57 | Vulkan dispatch | verified (integrated `b3db105`) | `fn3/57-vulkan-dispatch` | `fastnn-wt-57-vulkan` | `cfea81b` | done; unblocks #59/#60/#61 |
| #55 | Model format | verified (integrated `e3ac3e4`) | `fn3/55-model-format` | `fastnn-wt-55-model-format` | `456f3a3` | done; 38 format tests, 665 lib |
| #56 | Planner | verified (integrated `b9a0de6`) | `fn3/56-planner` | `fastnn-wt-56-planner` | `456f3a3` | done; 11 planner tests, CPU prefill parity 5.96e-7 |
| #59 | GPU pointwise | verified (integrated `3671a3c`) | `fn3/59-gpu-pointwise` | `fastnn-wt-59-gpu-pointwise` | `27fc60d` | done; 18 tests, byte-exact on 4 op families |
| #60 | GPU matmul | verified (integrated `2835c32`) | `fn3/60-gpu-matmul` | `fastnn-wt-60-gpu-matmul` | `27fc60d` | done; `MatMul` = Full, 17 tests, 0/0 validation |
| #61 | GPU attention | verified (integrated `117e51a`) | `fn3/61-gpu-attention` | `fastnn-wt-61-gpu-attention` | `27fc60d` | done; `MaskedAttention` now Full, 19 tests, 0/0 validation |
| #64 | Python API | verified (integrated `db94d6c`) | `fn3/64-python` | `fastnn-wt-64-python` | `c5b69c2` | done; wheel installs clean outside the checkout, 27 Python tests |
| #65 | Remove training/wgpu | verified (integrated `f772f10`) | `fn3/65-remove-training-wgpu` | `fastnn-wt-65-remove` | `0becaca` | done; 171 files, -27,921 lines, 81 lock entries gone |
| #66 | CI | verified (integrated `880b979`) | `fn3/66-ci` | `fastnn-wt-66-ci` | `cfea81b` | done; lanes render NOT VERIFIED honestly |
| #67 | Alpha release | review (local candidate) | `fn3/67-alpha-release` | `fastnn-wt-67-release` | `854afb6` | owner: review, then tag/publish/merge; see `docs/reports/fn3-21-alpha-release.md` |

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
| #55 | Model format | verified (integrated `e3ac3e4`) | `fn3/55-model-format` | `fastnn-wt-55-model-format` | `456f3a3` | done; 38 format tests, 665 lib |
| #56 | Planner | verified (integrated `b9a0de6`) | `fn3/56-planner` | `fastnn-wt-56-planner` | `456f3a3` | done; 11 planner tests, CPU prefill parity 5.96e-7 |
| #59 | GPU pointwise | ready | — | — | — | after #49, #53, #57 |
| #60 | GPU matmul | ready | — | — | — | after #49, #53, #57 |
| #61 | GPU attention | ready | — | — | — | after #49, #53, #57 |
| #62 | End-to-end | ready | — | — | — | after #49, #54-#61 |
| #63 | CLI/docs | ready | — | — | — | after #54, #55, #58 |
| #64 | Python | ready | — | — | — | after #48, #55, #58 |
| #65 | Cleanup | ready | — | — | — | after #62, #64 |
| #64 | Python API | verified (integrated `db94d6c`) | `fn3/64-python` | `fastnn-wt-64-python` | `c5b69c2` | done; wheel installs clean outside the checkout, 27 Python tests |
| #65 | Remove training/wgpu | verified (integrated `f772f10`) | `fn3/65-remove-training-wgpu` | `fastnn-wt-65-remove` | `0becaca` | done; 171 files, -27,921 lines, 81 lock entries gone |
| #66 | CI | ready | — | — | — | after #48 |
| #67 | Alpha release | review (local candidate) | `fn3/67-alpha-release` | `fastnn-wt-67-release` | `854afb6` | owner actions only |
| #67 | Alpha release | verified (integrated `c4c2401`) | `fn3/67-alpha-release` | `fastnn-wt-67-release` | `854afb6` | done; versions agree, clippy 0, wheel smoke passed |
| #68 | Quant format | verified (integrated `24aee9e`) | `fn3/68-quant-format` | `fastnn-wt-68-quant` | `672cec9` | done; W8A16 locked, trained fixture pinned |
| #69 | CPU quant | deferred (stable) | — | — | — | after #68 |
| #70 | GPU quant | deferred (stable) | — | — | — | after #68 |
| #71 | CPU perf | active (expanded scope) | `fn3/71-cpu-perf` | `fastnn-wt-71-cpu` | `672cec9` | thread sweep 1-16 + Auto, tiny and trained model |
| #72 | GPU perf | verified (integrated `dd1dd7f`) | `fn3/72-residency` | `fastnn-wt-72-resident` | `672cec9` | done; 9.14 ms prefill, 9.31 ms/token, 4 submissions/chain |
| #73 | Hardening | wave 1 verified (integrated `653cdda`) | `fn3/73-hardening` | `fastnn-wt-73-hardening` | `672cec9` | loaders done; runtime half after #72 |
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
| `02a8950` | #46 | ledger: #49 evidence, #47 baseline finding, parity inputs | n/a |
| `830f315` | #52 | storage windows/views merged (`0c017e9`) | fmt, 36 storage tests, 43 contracts, 588 lib, `--no-default-features` |
| `89dabac` | #53 | portable CPU reference merged (`2c49653`) | fmt, 43 reference tests, 613 lib, release == debug to the last digit |
| `4547b16` | #52 fix | extent-1 axes no longer count as stride overlap (found by #53) | 36 storage tests; unblocks `[batch,1,seq,seq]` masks for #56/#58 |
| `bd32607` | #48 test | contract assertions realigned to the implemented CPU matrix | 43 contracts |
| `b3db105` | #57 | Vulkan memory/dispatch/sync merged (`ddc4d5c`) | 14/14 smoke on GTX 1080 Ti, validation 0/0, add exact 100003/100003, device 106208 ns |
| `3cbd0cc` | #57 fix | device messages state the real reason a tensor cannot live on Vulkan | 43 contracts with and without `--features vulkan` |
| `880b979` | #66 | CI lanes merged (`08cf97c`) | lane renderer marks every absent lane NOT VERIFIED; cpu/python/frozen-2x/packaging pass |
| `8e73812` | #48 test | dropped the unused `TensorView` import | 0 fn3 clippy findings |
| `301e5a7` | #50 | bench harness + result schema merged (`f18fa37`) | 24 bench tests, real smoke run exit 0, schema valid |
| `a522fbf` | #66 fix | software lane proves it dispatched; physical lane pins its device by UUID | lavapipe proof line, `max_abs_error=0`; `verify_evidence.py` VALID for the GTX 1080 Ti record |
| `2353fc5` | #59/#60/#61 | scaffolded the kernel modules and their registration site | 14/14 smoke, 26 platform unit tests |
| `27fc60d` | #59/#60/#61 | kernel-set verdict tests made invariant-based | 26 platform unit tests; removes the three-way merge hazard |
| `4d188e8` | #57 test | Vulkan smoke support-gap assertions made invariant | 14/14 smoke; unblocked #60 and pre-empts #59/#61 |
| `2835c32` | #60 | Vulkan FP32 matmul merged (`4272aac`) | 17 matmul tests, 14/14 smoke, 30 platform, 43 contracts; 0 errors 0 warnings; fixtures 1.14e-7..2.86e-6 |
| `3671a3c` | #59 | Vulkan pointwise/transform/embedding merged (`793e97d`) | 18 pointwise tests, 14/14 smoke, 41 platform, 0/0 validation, byte-exact for the layout and comparison ops |
| `fcad314` | #59 fix | a capability flag is an aggregate, not a kernel's own claim | 17 matmul tests; caught a cross-issue test collision neither worker could see |
| `5a055e1` | #54 | strict ONNX import merged (`56852ca`) | 27 import tests on the pinned 27,897,974-byte model, 43 contracts, 637 lib; 9001 insertions, one shared line |
| `456f3a3` | #54 fix | `model::weights` declared instead of reached into by `#[path]` | 27 import tests with the model present |
| `e3ac3e4` | #55 | versioned prepared-model artifact merged (`48e4d46`) | 38 format tests, 27 import, 665 lib, fmt; pinned model round-trips byte-identically |
| `ae125e9` | #58 | session, KV cache and cached decode merged (`b1a0413`) | 8 session tests, 683 lib; prefill 5.96e-7, decode.step1 6.11e-7, extents 14–17, cached-vs-full bit-identical |
| `c5b69c2` | #58 fix | a session without a runtime names the constructor that works | 43 contracts |
| `15f6177` | #52/#57 fix | device enumeration asserted as a rule, not a snapshot | 36 storage tests in both configurations |
| `304d0f8` | #63 | inference CLI and offline quickstart merged (`669833b`) | 6 CLI tests, 24 CLI unit, 707 lib, 43 contracts; real 10-token prefill, honest Vulkan refusal |
| `f772f10` | #65 | wgpu + training removed, inference surface promoted (`43396e5`) | lib 673 (37 training tests enumerated), ten fn3 binaries unchanged, 10/10 e2e on the GTX 1080 Ti, 0/0 validation, no wgpu in `cargo tree` |
| `1cebabd` | #62 | end-to-end on CPU and the physical GPU merged (`44081e0`) | 10 e2e tests, 719/719 proven executions, 0 readbacks, validation 0/0 |
| `0becaca` | #52/#57/#62 fix | device errors name the request in any configuration | lib 710 default / 753 vulkan / 690 no-default |
| `d0c7b00` | #67 | alpha candidate: versions, wheel smoke, clippy zero, fn3 bench adapter | lib 673, 10 fn3 binaries, 378 pytest, wheel top-5 exact, clippy clean in six configurations, GPU e2e 10/10 with 0/0 validation; see `docs/reports/fn3-21-alpha-release.md` |
| `db94d6c` | #64 | Python facade `fastnn.fn3` merged (`1e32752`) | 27 Python tests, 710 lib, 43 contracts, 36 storage, 5 ci-smoke; wheel installed in a clean venv outside the checkout reproduces the CLI's top-5 exactly |
| `117e51a` | #61 | Vulkan norm/softmax/reduce/fused-attention merged (`c69a890`) | 19 attention tests, 14/14 smoke, 41 platform, 43 contracts; all-masked row exactly [0.25;4] |
| `82e0003` | #46 | ledger: #61 state, fused-id decision, gate rows | n/a |
| `4771083` | #46 | ledger: whole-tree gate run after #60/#61 | n/a |
| `3671a3c` | #59 | Vulkan pointwise/transform/embedding merged (`793e97d`) | 18 pointwise tests, 14/14 smoke, 30 platform, 43 contracts; 18/18 SPIR-V valid; byte-exact where exactness is expected |
| `117e51a` | #61 | Vulkan norm/softmax/reductions/fused attention merged (`c69a890`) | 19 attention tests, 14/14 smoke, 41 platform, 43 contracts, 17 matmul; 0 errors 0 warnings |

## Integration decisions that bind later issues

- **Kernel registration is wired by the integrator, not by kernel owners.**
  `KernelSet::alpha()` calls `pointwise`, `transform`, `embedding`, `matmul`,
  `norm`, `reduce` and `attention` in that order, and replacement is by op id,
  so a later registration supersedes the FN3-11 smoke kernel. Kernel issues
  fill in their own module and never edit `kernels/mod.rs`.
- **Verdict tests are invariants.** `KernelSet` support verdicts must equal the
  registered kernel's own verdict, and an op with no kernel must name its owner.
  This holds before and after every GPU kernel lands, so adding a kernel cannot
  break a test owned by another issue.
- **Validation is enabled in code, never through `VK_INSTANCE_LAYERS`.**
  Forcing the layer from the environment duplicates it and the loader then
  emits `env var VK_INSTANCE_LAYERS defined` messages, which the teardown
  test's zero-delta guard correctly counts as a leak. Both the software lane
  and the physical evidence lane were fixed to stop doing this.
- **Support gaps are asserted as rules, not snapshots.** Both the kernel-set
  unit tests and `tests/fn3_vulkan_smoke.rs` used to pin which ops were
  missing, so every GPU kernel merge broke a test owned by another issue.
  They now assert that a non-`Full` verdict names its owner, that asking for
  such an op is rejected by name, and that a capability flag agrees with the
  verdict it summarises. Do not reintroduce op lists in either place.
- **The fused attention id is real, not aspirational.** `#61` registered
  `MaskedAttention` (40) with `Support::Full` and `features.masked_attention`
  follows it. The kernel's `keys <= 4096` bound is part of the documented
  contract and is enforced loudly; a flash-style tiled kernel would lift it,
  and that is a later performance issue, not a hidden limit.
- **A capability flag is an aggregate, so never assert it is false.** #60's
  matmul test asserted `!capabilities.features.broadcasting` to say "this
  kernel does not claim the elementwise flag", and it broke the instant #59
  registered a broadcast-capable `Add`. A kernel test may only assert the
  implication: if the flag is advertised, the kernel that backs it is `Full`.
  The same rule applies to `masked_attention` and to any flag added later.
- **`shaders/compile.sh` must exit 0 on success.** It exited 141 after a fully
  successful build because `pipefail` plus `glslc --version | head -n 1` is a
  SIGPIPE, so a lane checking the exit status would have called a good build bad.
- **A lane that skipped is not a lane that passed.** `vulkan-software-lane.sh`
  now requires the smoke test's `[fn3vk][software-icd][proof]` line; a suite
  where every test returned early exits 0 and would otherwise have gone green.
- **Physical evidence is pinned by UUID.** `gpu_evidence.py` sets
  `FN3_VK_REQUIRE_PHYSICAL=1` and `FN3_VK_DEVICE_UUID`, and the smoke test
  honours both: selection is by UUID because the loader enumerates a software
  ICD next to the real GPU, and a missing device is a failure rather than a skip.
- **`Tensor` storage is host memory on every device.** Vulkan executes by
  staging through host memory and records `host_staged=true`; device-resident
  storage has no owner yet (it belongs with the session/KV work, #58). The
  device-resident path (`alloc_buffer`/`upload`/`dispatch`/`download`) exists
  for kernel chaining. Any claim of zero-copy residency on an op path is false
  until #58.

## Whole-tree gate run (after #60 and #61 merged, HEAD `82e0003`)

Run from the integration checkout with `CARGO_BUILD_JOBS=6`. Every line is an
observed exit status, not a plan.

| Check | Result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo build --no-default-features` | ok |
| `cargo build` (default features) | ok |
| `cargo build --no-default-features --features vulkan` | ok |
| `cargo test --lib` | **613 passed** |
| `cargo test --lib --no-default-features` | **596 passed** (one parallel run showed 595/1 — the known 2.x arena-telemetry flake; `--test-threads=1` is always green) |
| `cargo test --test fn3_contracts` | 43 |
| `cargo test --test fn3_tensor_storage` | 36 |
| `cargo test --test fn3_cpu_reference` | 43 |
| `cargo test --test fn3_ci_smoke` | 5 |
| `cargo test --features vulkan --test fn3_vulkan_smoke` | 14 (physical GPU, validation 0/0) |
| `cargo test --features vulkan --test fn3_vulkan_matmul` | 17 |
| `cargo test --features vulkan --test fn3_vulkan_attention` | 19 |
| `cargo test --bin fn3-bench` (release feature set) | 24 |

Known debt, cleared by #67:

- `cargo clippy --all-targets -- -D warnings` now passes in every shipped
  feature configuration (default, `--no-default-features`, `vulkan`, `cli`, and
  `fn3`). The budget in `tools/ci/cpu-lane.sh` is **0** and warnings are denied,
  so a new finding fails the lane. Suppressions were kept to two deliberate
  test-helper cases (an 8-argument reference/table helper in the Vulkan tests);
  everything else was fixed.
- `scripts/ci/check-agents-freshness.py` passes on this tree. The #67 version
  work added no version literal to any AGENTS.md; the checker is in the CPU lane.

## Alpha gate status

| Gate | State |
|---|---|
| CPU prefill on the pinned tiny model, parity to the #49 fixtures | **proven for prefill** (#56): `prefill.p1` max_abs 5.96e-7 / mean 9.77e-8, `prefill.p4.last` 5.16e-7 / 7.90e-8, argmax matches (tol 1e-5 / 2.5e-6). Cached decode still needs #58. |
| Named physical Vulkan device, real execution | platform proven (#57); model path open — needs #59/#60/#61 + #62 |
| Cache bounds and session isolation | open — needs #58 |
| Residency record proves no silent CPU execution | proven for Add (#57) and MatMul (#60); full-model assertion is #62 |
| GPU matmul on the pinned shapes | proven (#60): real prefill/decode/attention shapes within tolerance, 144,448 ns kernel-only decode `lm_head` |
| GPU norm/softmax/attention on the pinned shapes | proven (#61): LayerNorm 2.384e-7, softmax 5.960e-8, fused attention 5.960e-8, ReduceSum bit-exact, all-masked row exactly uniform and finite |
| GPU pointwise/transform/embedding | proven (#59): Add/Sub/Mul/Neg/Relu/Cast/comparisons/layout byte-exact, Gelu 4.77e-7, embedding exact |
| Explicit device requests never run model computation on CPU | proven (#48/#57, both feature configurations) |
| Python facade | **proven** (#64): `fastnn.fn3` prepare/load/session/prefill/decode. From a wheel installed in a fresh venv outside the checkout, a 10-token prefill gives the same top-5 as the Rust CLI (46737, 28597, 16571, 7868, 13); GIL released on every long call. |
| CLI + quickstart | **proven** (#63/#67): `fn3-infer prepare/inspect/run/generate/diagnose`, 27,879,928-byte artifact with verified CRC, 10-token prefill to top-5 [46737, 28597, 16571, 7868, 13]; token IDs only, no approximate tokeniser. `--device vulkan` runs prefill and three cached decode steps on the GTX 1080 Ti (top-3 [46737, 28597, 16571]); an explicit request never falls back to CPU. |
| Package versions 3.0.0-alpha.1 / 3.0.0a1 | **proven** (#67): both spellings agree in the manifests, both Cargo lockfiles, `uv.lock`, `fastnn/__init__.py` and the changelog; `scripts/ci/check-versions.py` enforces it and runs in the CPU and packaging lanes and the release workflow. The wheel is `fastnn-3.0.0a1-cp311-abi3-manylinux_2_39_x86_64.whl`; a clean venv with `numpy 2.5.5` only reproduces top-5 [46737, 28597, 16571, 7868, 13]. |
| CI lanes | machinery merged (#66); vulkan lanes now genuinely execute |

## #54 findings that bind #56, #58 and #62

- **The importer is a real protobuf parser.** `src/fn3/import/proto.rs` is a
  bounds-checked wire-format reader with no dependency; decoded messages borrow
  names and weight bytes, so 28 MB is never copied. Every failure is one of 36
  typed diagnostics and all problems are accumulated, not just the first.
  Eleven truncated prefixes and twelve byte flips of the real file produce
  errors and no panic.
- **`i64` shape arithmetic is preserved, not cast.** The pinned exporter
  computes `total`, `past` and the causal index range with `i64` `Add`/`Sub` at
  four nodes. Casting to `f32` would be a silent numerical change, and the CPU
  arithmetic kernels are `f32`-only by contract with `Cast` the only converter.
  **The planner must fold this arithmetic at specialisation**; executing it is
  not an acceptable answer. `tests/fn3_import.rs::integer_arithmetic_is_preserved_rather_than_cast`
  pins it, so a worker cannot quietly "fix" this by widening the kernels.
- **24 of 326 values keep one unresolved axis**, the `[batch*seq, hidden]`
  flatten feeding each `Gemm`, because the seven-symbol envelope cannot express
  a product. The count is pinned by a test; specialisation resolves or rejects.
- Geometry is detected and validated, not assumed: 2 layers, hidden 128, mlp
  512, 2 query and 2 KV heads, head_dim 64, vocab 50,257, 1024 learned
  positions, epsilon 1e-5. A non-decoder graph, a non-`{past_key_values,
  present, logits}` interface, opset outside 13..=17 or external tensor data
  fails with a named error.
- **The plan is bounded before it runs, and it has an identity.** `PlanBound`
  is derived from the envelope corners; every plan must be admitted by it
  (210 slots, 144 instructions, 28,507,141 peak bytes for the pinned model at
  envelope `batch[1,1] seq[1,13] past[0,17]`), so a session cannot grow past
  what was declared. `PlanId` is FNV-1a-128 over envelope, dims, slots,
  constant bytes, instructions and parameter bits — no map iteration, no
  addresses — so identical inputs give an identical id and a plan can be
  cached or shared.
- **Integer shape arithmetic is folded at plan time.** This is the constraint
  #54 identified. Folding is pinned twice: by node index, and by the content
  of the constants that depend on it (`[seq, total-seq]` and the 16-element
  causal mask). A surviving integer arithmetic op is refused by name, and f32
  is never folded, so there is exactly one numerical path.
- **`PlanAttr` is the only writer of `OpParams`.** `OpParams` is core-owned and
  still untyped; the fused-attention causal flag is isolated to two lines
  behind `PlanAttr`, which round-trips every variant in a unit test. The
  `OpAttributes` enum that should replace the encoding is specified in
  `docs/reports/fn3-10-planner.md` §7.2 for #48 to adopt.
- **The artifact stores the graph, not a specialised plan.** A plan is a
  function of graph + envelope + device, so caching one in a shareable file
  would bind the file to a device. `PreparedArtifact` therefore carries the
  value/node tables, the weight bytes, the geometry and the envelope, and the
  planner produces the plan at load. Recorded because the issue title says
  "prepared model" and the narrower reading is the correct one.
- **`GraphIr` has no constructor.** Its fields are private, so the artifact
  reader cannot rebuild one and owns an accessor-compatible `PreparedGraph`
  mirror instead. The additive `GraphIr::from_parts` should land with its
  first consumer (#62), not as unused code.
- The model is **not** committed: no license is declared upstream. Tests pin
  both the sha256 and an FNV-1a digest and skip with a printed reason when the
  file is absent, so an offline CI lane is honest rather than falsely green.

## Process finding: the orchestrator's own briefs need verification

Twice an orchestrator-supplied constant was wrong and a worker caught it: the
pinned model sha256 was quoted with a transposed nibble in the #58 brief and
again in the #63 brief. The authoritative value is
`e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1`, confirmed
by `sha256sum` against the file and by `fixtures/parity/tiny-gpt2/model.lock.json`.
Both workers checked the brief instead of trusting it and reported the
discrepancy, which is the behaviour the briefs ask for. Standing rule: a
digest in a brief is a claim to be checked against `model.lock.json`, never a
fact to be copied into code or a report.

## Remote state (owner authorised writes)

`rewrite/fastnn-3` is pushed to `origin` and tracks it; `origin/rewrite/fastnn-3`
is `0becaca1c6c4044ac60ea266bcb0a5b94f9b4261`. The #67 candidate
(`fn3/67-alpha-release`, `d0c7b00`) is **local only**: this issue performed no
push, tag, package upload, pull request or merge. The owner actions are listed
in `docs/reports/fn3-21-alpha-release.md`. No PR to `main` was created and
`main` is untouched, so the original prohibition on merging into `main` still
holds. **Nineteen issues are closed as completed** (#47-#64 except none, plus
#66) each with an evidence comment naming the branch HEAD and the measured
numbers. **Fourteen remain open** and every one has a comment saying why: #46
(alpha handoff, open until #65/#67 land), #65 (in progress), #67 (blocked on
#65), #68-#76 (stable wave, not started, each with its real prerequisite
named), #77 and #78 (deferred by the owner, not activated).

The closure rule applied: an issue was closed only where its own acceptance
criteria were independently reproduced by the integrator from the integration
checkout. No issue was closed because a compile passed, and no deferred issue
was moved to active.

## Alpha scope decision recorded at #65

> **Superseded for public naming:** use `fastnn::inference`,
> `fastnn.inference`, and `fastnn-infer`. The alpha decision below is retained
> as history; the rename does not change `.fnn3` artifacts or move `src/fn3/`.

FN3-19 was written when 3.0 was expected to move `src/fn3/` to the crate root
and drop the `fn3` prefix. The ADR still says so. That rename did **not**
happen, and the reason is recorded here rather than left as an omission: the
Python package (`fastnn.fn3`) and the CLI (`fn3-infer`) already ship under that
name, and the artifact magic is `FNN3PREP`, so the prefix is now part of the
published surface rather than a temporary marker. Renaming it would touch every
import, the Python package, the CLI and every committed artifact for no
behavioural gain. `src/fn3/AGENTS.md` no longer calls the tree temporary. A
future rename is mechanical and is not scheduled.

The 2.x forward tree is deliberately retained: it is inference-only now and
still exercises surviving CPU kernels through 348 passing tests. Deleting it
would have removed tests whose subject still exists, which the #65 brief
forbade.

## Alpha complete

Candidate **`c4c2401`**, baseline `a2c0759`, on `rewrite/fastnn-3` (pushed).
Versions `3.0.0-alpha.1` / `3.0.0a1`, enforced in seven stores by
`scripts/ci/check-versions.py`.

Gate results, all reproduced by the integrator from the integration checkout:
`cargo fmt --check` clean; `cargo clippy --all-targets -- -D warnings` clean in
six configurations; `cargo test --lib` 673; ten `fn3_*` binaries 222 passed;
four Vulkan binaries 68 passed with 0 errors / 0 warnings on the GTX 1080 Ti;
`fn3_e2e` 5 CPU / 10 Vulkan with 719/719 proven executions and 0 input
readbacks; `pytest -m "not slow"` 378 passed; packaging lane PASSED with a
wheel installed in a clean venv outside the checkout running real inference;
six CI lanes render with four PASSED and two deliberately NOT VERIFIED.

Parity, FP32, against the independent ORT/PyTorch fixtures: `prefill.p1`
5.96e-7 CPU / 6.56e-7 Vulkan, cached decode steps 2-4 argmax matching, extents
exactly 14-17, and cached-vs-full **bit-identical** for all four steps.

Issues #46-#67 and #66 are closed with evidence comments. Open: #68-#76
(stable wave, each naming its real prerequisite), #77 and #78 (deferred by the
owner, not activated).

Owner actions deliberately not taken: no tag, no GitHub release, no PyPI or
crates.io publication, no PR to `main`, `main` untouched. Model weights are not
redistributed (no license is declared upstream).

## Alpha follow-up (#67): matched baseline and strict release smoke

Branch `fn3/alpha-followup`, base and verified candidate
`672cec9faf089ef0840179f6edfdf33779b576a1` on `rewrite/fastnn-3`. Evidence
in `docs/reports/fn3-21-alpha-followup/`; the handoff's timing section is
corrected in place with the old figures kept and labelled.

- **Timing is a matched release-build measurement.** `tests/fn3_alpha_bench.rs`
  runs one concrete backend per process: 3 warmups and 20 measured iterations
  of the pinned 10-token prompt plus decode `640,11,612` (contexts 11/12/13),
  resetting between iterations so every prefill is `(10,0)`. CPU p50 240.31 ms
  prefill / 158.33 ms per decode token; GTX 1080 Ti p50 204.76 ms / 190.64 ms,
  with every phase proven on the device (11,500/11,500 executions, 0 input
  readbacks, validation 0/0). Percentiles are nearest-rank and all samples are
  committed. The superseded `--quick` table (271.77 / 179.74, peak RSS
  165.67 MiB, device peak unavailable) is kept as a labelled correction: it
  recorded no profile, used `--quick`, compared `fn3-bench` CPU against CLI
  Vulkan on a 13-token prompt, and read an unpopulated adapter field.
- **`memory.device_peak` is now measured, and its limit named.**
  `MemoryStats::peak_live_bytes` reports 120,669,184 B. A device-local-only
  high-water mark is not exposed; that row is `unavailable` and names the
  missing `MemoryStats::peak_live_device_local_bytes`, never zero. Live
  device-local is 0 B at rest because alpha tensor storage is host-staged.
- **Vulkan device time is not yet a stable number.** Its p50 spans
  12.93-28.55 ms per step across runs and even within one run, because the GPU
  clock is unpinned (139 MHz idle against a 1911 MHz maximum) and the fenced
  per-instruction kernels are too short to hold a boost state. Host wall time is
  the comparable metric. Pinning the clock needs root: owner decision.
- **A release wheel cannot pass on an import alone.** `--require-inference`
  makes a missing pinned model a failure in `wheel_smoke.py` (exit 2, named
  path) and `packaging_lane.py` (exit 1 before the build, with the pinned URL
  and sha256). The release workflow fetches the model from the URL derived from
  `fixtures/parity/tiny-gpt2/model.lock.json` and verifies size and sha256; a
  fetch failure is a named `::error::pinned-model` failure. The local default
  stays non-strict and records `not_run` with its reason.
- **The crate package is trimmed and described.** `cargo package` ships 306
  files / 1.0 MiB (was 672 / 3.1 MiB) with no fixtures, models, docs, tests or
  artifacts; `cargo publish --dry-run` stops at upload with no manifest
  warnings. `rust-version = "1.89"` is measured (1.88 rejects the retained
  AVX-512 kernels, 1.89 passes four feature configurations). The wheel carries
  the same metadata, the MIT licence read from the committed `LICENSE`, and no
  tests or fixtures.
- **Hosted CI found one real gap.** The first dispatch on `rewrite/fastnn-3`
  failed `vulkan-software` because the runner ships lavapipe but not
  `VK_LAYER_KHRONOS_validation`; installing `vulkan-validationlayers` made the
  lane green on the follow-up branch. The review candidate was not modified.
  `pinned-model` remains NOT VERIFIED by design and `gpu-physical` validates
  the committed record without running a GPU.

Owner actions deliberately not taken here: no merge, tag, release, package
upload, PR, issue comment, clock pinning or change to `rewrite/fastnn-3`.

## Wave 1: stable workstreams (post-alpha)

Five of six wave-1 branches are integrated on `rewrite/fastnn-3`; #71 alone
remains active. The alpha
review commit `672cec9` remains an ancestor and is still the named review
candidate; `main` is untouched.

| Merge | Branch | HEAD | Scope |
|---|---|---|---|
| `6275485` | `fn3/artifact-exec` | `a74817e` | `.fnn3` executes without the source ONNX in Rust, CLI and Python |
| `653cdda` | `fn3/73-hardening` | `06a4208` | loader/parser budgets, 38 malformed cases, fuzz campaigns, unsafe audit |
| `24aee9e` | `fn3/68-quant-format` | `ba7266e` | trained GPT-2 pin, W4A8 evaluation, `fn3-quant-format-v1` |
| `3d2870c` | `fn3/alpha-followup` | `792d063` | matched release baseline, strict wheel inference, hosted CI fix |
| `dd1dd7f` | `fn3/72-residency` | `7d4e6e5` | persistent weights, resident slots/KV, one submission per step |

Decisions taken with the evidence:

- **W8A16 is the starter format, not W4A8.** Every pure 4-bit candidate flipped
  at least one decode top-1 on the trained model. `fn3-quant-format-v1` is
  signed INT8 weights in groups of 64 along K with binary16 scales and int16
  zero-points, FP32 activations, and a fused `Σx·(q−z)` group accumulation.
  It keeps top-1 on all measured steps at 26.6% of FP32 projection storage
  (3.77x). The departure from the issue's "W4A8 first" default is documented
  in `docs/reports/fn3-22-quant-format.md`; the evaluation itself was done
  first and is what rejected W4A8.
- **The trained FP32 reference exceeds its own suite `mean_abs` bound**
  (prefill and decode step 2, 1.640e-4 and 2.975e-4 against 5e-5) while every
  elementwise tolerance and every top-1/top-5 agrees. This is recorded as a
  systematic-bias finding for #74/#75 rather than hidden by loosening the
  bound.
- **Hosted CI on the review candidate exposed a workflow defect, not a code
  defect**: run 44486950429 failed `vulkan-software` because the hosted image
  ships lavapipe but not `VK_LAYER_KHRONOS_validation`. Installing the layer
  fixed it without weakening the 0 errors / 0 warnings requirement. The
  candidate stays identifiable and red on that lane; the fix is in the
  follow-up.
- **Artifact-only is now the normal path.** `--onnx` is optional verification,
  and the Rust, CLI and Python paths all run from a `.fnn3` alone with the
  ONNX renamed away, producing identical plan ids and top-5.
- **Vulkan is resident by default and the gap was staging, not arithmetic.**
  At `672cec9` one chain uploaded 318,880,260 B and made 575 submissions; after
  #72 it uploads 5,328 B and makes 4 plan submissions. Prefill TTFT p50 goes
  142.36 -> 9.14 ms and decode 130.44 -> 9.31 ms per token on the GTX 1080 Ti,
  with the base arm reproducing the alpha traffic byte for byte. Weight
  re-upload is a hard error, not an observation: a second session moves 0 B.
  Two limits are recorded: a decode step is still 10 `vkQueueSubmit` calls
  because four KV appends and the logits download sit outside the plan batch,
  and peak device memory rises 62 -> 116 MB because a step's staging blocks are
  held until its fence (an arena is the follow-up).
- **The hosted CI fix is proven at the integration commit.** Run 44491089648
  on `3d2870c` completed green across cpu, cpu-python, vulkan-build,
  vulkan-software-smoke, gpu-physical evidence and the packaging matrix.

#73's **runtime half** (device loss, unavailable features, repeated reset,
session error recovery, release-build behaviour) is deliberately not in this
wave and is scheduled after #72, whose storage and submission rewrite changes
the failure surface it must test.

Still running at the time of this entry: **#71**, now under the owner's
expanded automatic-thread-scaling requirements (1/2/4/8/12/16/Auto on the
Ryzen 7 3700X, prefill and batch-1 decode, tiny and trained model, Auto
default with a shared budget and a #69 interface). #72 integrated at `dd1dd7f`;
the merged tree's `cargo test --lib` is 682 and every `fn3_*` binary passes.

## Wave 2: CPU scaling, runtime hardening, quantized paths

**#71 merged `42d023a`.** Column-tiled prefill and adaptive `Auto`. The cache
hypothesis was refuted by a six-arm experiment (25.7 MB working set, 16 threads,
idle recovery: 0.19 ms spread, smaller than within-arm p95; the same binary
varied 5.277-7.886 ms between windows). Column tiling reads the weight once and
is 0.77x on tiny prefill 32 and 0.85x on trained prefill. Auto narrows to one
thread after observing a one-row product that cannot use threads. Reproduced by
the integrator on trained GPT-2 at `aed9d95`: prefill 1->16 **1.38x**
(330.4->240.0 ms), 1->12 1.37x; decode 1->12 1.04x (142.5->137.4 ms); Auto
239.33/258.91/133.19 ms on prefill 10/13/decode-after versus best fixed
240.02/260.51/133.60. The chain Auto sample 152.82 ms was transient; a repeat
gave 143.76 vs 142.98 at 4 threads and 139.09 at 16, ratio 1.034 inside the
0.95-1.05 tolerance. The owner's ~1.2x reproduces for prefill, not decode.
Residual: the first turn of a session runs wide before narrowing, which needs a
session-level hook in `model/**`.

**#69 merged `67ca5ae`.** CPU W8A16 fused projection, new module only. Trained
prefill **1.48x**, decode **1.10x**, projection storage 26.56% of FP32, top-1
preserved on every measured step; final quantized accuracy sign-off remains tied
to the trained FP32 decision.

**#73 merged `29d9e69`.** Recoverable failures leave extent/step/tokens/phase
unchanged, release slots and roll back every KV layer; device loss poisons the
device and returns a typed terminal error naming adapter, UUID and call.
Injected matrix 4 debug + 4 release; 12 injected and 8 hardware reset rounds;
physical e2e 10 with resident and staged arms byte-identical, validation 0/0.

**#70 merged `aed9d95`.** Vulkan W8A16 fused kernel, resident tiny proof: 8
prefill / 24 decode quantized records with 0 upload, 0 readback, 3 device-local
bindings, readback exactly 804,112 B, submissions 6/30. Trained is **staged
only**: the resident path fails on identity `Reshape(input_ids) -> Gather`
(`Fn3Error::NotImplemented`), a defect in the plan/embedding boundary, not in
the quantized kernel. #70 stays open until that is fixed and trained resident
is measured.

**Trained FP32 investigation merged `6e1d564`.** The 5e-5 mean-absolute bound is
below the reference's own rounding error: an independent float64 evaluation
differs from ORT by 6.254e-5 at the same position. No numerical defect was
found; projections reproduce the documented order bit-for-bit, per-layer
relative RMS stays ~1e-6, and the error is a near-constant logit translation
that softmax removes. Proposed scale-relative replacement is documented but
**not applied**; the old bound remains visible and failing. This is an owner
decision before the stable gate.

Resident-trained fix in progress on `fn3/resident-trained` from `6e1d564`.
