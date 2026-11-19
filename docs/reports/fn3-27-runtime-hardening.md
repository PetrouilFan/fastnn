# FN3-27 (#73): resident-runtime hardening

**Owner:** #73 (FN3-27), runtime wave. **Branch:** `fn3/73-runtime`.
**Worktree:** `/home/petrouil/Projects/github/fastnn-wt-73-runtime`.
**Base:** `6389186d4599471c4cb3ebc1156a5c1b872cd423` (integrated #72).
**Implementation HEAD:** `8ef04bb4e7d607e61e5a25c988e045d1d519efdd` (the commit
before this report; the report is the follow-up evidence commit).

No remote writes were made: no push, tag, PR, merge, release, issue comment or
close. The loader wave from `653cdda` is untouched and its regressions still
pass (§12).

---

## 1. What was hardened, and how to read the evidence

The resident-default Vulkan path from #72 is one transaction over several
subsystems: a plan batch, KV appends outside that batch, a logits readback, and
a session whose only mutable state is the KV extent and its counters. This wave
audits that transaction as merged, defines what a failure leaves behind, and
makes the failure paths observable with deterministic injection.

**Two kinds of evidence, never mixed:**

- **Injected / simulated.** `[fn3-27][injected]` lines come from the
  non-default `vulkan-fault-injection` feature. They exercise the real backend
  but fail at an armed seam. `[fn3-27][simulated]` is the synthetic-adapter
  capability check. None of this is hardware proof.
- **Hardware.** The unmodified `fn3_e2e`, `fn3_vulkan_resident` and the new
  `fn3_vulkan_reset` binary run with only `--features vulkan`; no injection code
  is compiled into them. Their results are in §10 and are the physical
  evidence.

Device used for every GPU run: NVIDIA GTX 1080 Ti, UUID
`471b2bda-51e1-6451-e692-ff5d1e146f1e`, selected by UUID, validation enabled in
code, `VK_INSTANCE_LAYERS` unset, `FN3_VK_REQUIRE_PHYSICAL=1`. Model
`/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx`, 27,897,974 B,
sha256 `e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1`,
verified against `fixtures/parity/tiny-gpt2/model.lock.json` by the harness on
every run. Weights are not committed.

---

## 2. Lifecycle inventory (as merged at `6389186`)

The step transaction, in execution order. "Left behind?" is what a failure at
that point could strand *before* this wave.

| # | Point | File:line (base) | Left behind? |
|---|---|---|---|
| 1 | Capacity check, every layer | `src/fn3/model/session.rs:916` | nothing (pure) |
| 2 | Plan lookup / build | `session.rs:928` | a cache entry (diagnostic, not state) |
| 3 | Slot materialise (device alloc) | `session.rs:929`, `kv.rs:212` | **device buffers of the partial slot list** — dropped, but the plan batch was not open yet |
| 4 | Graph-input bind (host tensors) | `session.rs:935` | host allocation only |
| 5 | KV past-prefix refresh | `kv.rs:529` (`live_with`) | scratch `past_*` buffers (cache-owned, reused) |
| 6 | `begin_step` opens the batch | `vulkan/mod.rs:758`, `submission.rs:326` | **a recording command buffer** |
| 7 | Per-instruction record | `submission.rs:597` (`record_batched`) | **a partial command buffer**, retained staging blocks if a later record fails |
| 8 | `end_step`: end + submit | `submission.rs:392`, `device.rs:289` | **the partial plan would still be submitted** |
| 9 | Fence wait | `submission.rs:672`, `device.rs:318` | **`in_flight` stayed `true`; the submitter was unusable but reported a generic internal error** |
| 10 | Logits readback | `session.rs:1146`, `resident.rs:358` | nothing committed yet (KV commit is later) |
| 11 | `present` validation, every layer | `session.rs:1000` | nothing |
| 12 | KV append, layer by layer | `session.rs:1007` → `kv.rs:473` | **half-appended layers**: layer 0 committed, layer 1 failed |
| 13 | Recycle + counters | `session.rs:1011` | pool generation only |

Point 6/7/8 is the batch lifecycle; 12 is the KV commit. The two concrete
defects fixed here are:

1. **A recording failure did not mark the batch.** `end_step` ended the command
   buffer and submitted whatever prefix had been recorded, so a step the caller
   was told had failed still executed part of the plan on the device. Fixed by
   a `failed` flag on `BatchState` (`submission.rs:402`) and a discard path that
   resets the command buffer and drops the retained staging blocks.
2. **A fence-wait failure wedged the submitter.** `finish_wait` returned before
   `in_flight = false`, and the next step reported
   `"Submitter::record called while its previous submission is in flight"`
   (a generic `InvalidShape`). The completion of the submission was unknown, so
   reusing the command buffer or fence was not safe. Fixed by poisoning the
   device with a typed terminal `Error::Lost` (`device.rs:252`, `:266`).

The KV half-append was latent rather than observed: `check_append` and
`check_present` ran for every layer first, but the six device-to-device copies
of a two-layer append can still fail individually. Fixed by rollback (§3).

---

## 3. The failure contract

**Recoverable (every failure that is not device loss).** After
`prefill`/`decode` returns `Err`, the session is usable as-is:

- every layer's live extent, `step()`, `tokens_consumed()`, `phase()` and the
  plan cache are exactly what they were before the call;
- the failed step's value slots are released (nothing is recycled into the
  pool), and the previous generation stays valid;
- a retry appends each token exactly once. A multi-layer commit that fails
  after an earlier layer was written **rolls every layer back** to its pre-step
  extent (`session.rs:1094` → `kv.rs:347`). Bytes past the live extent are
  unobservable by the KV contract, so the retry overwrites the same
  deterministic rows; there is no double append and no stale extent.

`Session::recover()` (`session.rs:877`) is the explicit acknowledgement:

1. returns the backend's typed terminal error and touches nothing if the device
   is poisoned;
2. rejects disagreeing layer extents with a typed `SessionState` (a refactor
   guard; the transaction makes it unreachable in normal operation);
3. drops the previous step's pooled output buffers;
4. repairs a phase left at `Prefilling` by a panic.

**Terminal.** A device lost at `vkQueueSubmit`, or any fence wait whose
completion is unknown (device lost, timeout, driver error), poisons the
`Device`. The first failure and every later operation return
`Fn3Error::DeviceUnavailable` naming the adapter and the failing call:

```text
device unavailable: requested device the Vulkan device is unavailable:
adapter "NVIDIA GeForce GTX 1080 Ti (471b2bda-...)" is no longer usable:
vkQueueSubmit: injected device loss at submit (FN3-27; no command buffer was
submitted). The device is poisoned; drop the backend and open a new one;
available devices: [cpu, vulkan(0) (...), vulkan(1) (llvmpipe ...)]
```

Why terminal instead of an in-place reset: after a lost fence the driver still
owns the command buffer and every buffer bound in it; resetting the pool or
freeing the buffers would be use-after-free. The only safe recovery is to drop
the backend (teardown already calls `vkDeviceWaitIdle`) and open a new device.
There is **no silent retry** anywhere: `Session::recover` and every step return
the stored terminal error, and `begin_step`/`end_step` refuse to record.

Recovery was chosen over "terminal for everything" because allocation,
readback, recording and KV-copy failures are demonstrably non-destructive: the
matrix proves the session is byte-identical to a clean reference after each one.

---

## 4. Deterministic fault injection

`vulkan-fault-injection` is a non-default cargo feature (`Cargo.toml:73`),
compiled out of every shipped configuration. `fault.rs` holds a per-device
injector: arm one point, skip *N* matching events, fail the next one **once**,
then disarm so the retry runs the real path. `FaultInjector::fired_at` and
`calls(point)` prove the injected event fired.

| Point | Seam | Failure |
|---|---|---|
| `Allocation` | `Device::allocate_range` before the allocator (`device.rs:469`) | `vkAllocateMemory` / `ERROR_OUT_OF_DEVICE_MEMORY` |
| `Submit` | `Device::submit` before `vkQueueSubmit` (`device.rs:368`) | terminal `Error::Lost`; nothing was submitted |
| `Fence` | `Device::wait_fence` after the real wait (`device.rs:418`) | terminal `Error::Lost`; the submission had completed |
| `Readback` | `ResidentBuffer::download` / `VulkanBackend::download` | recoverable `Fn3Error::Allocation`, before any submission |
| `Record` | `Recorder::dispatch` inside the batch closure (`submission.rs:1005`) | the batch is marked partial and discarded |
| `ResidentCopy` | `copy_resident_storage_regions` (`resident.rs:852`) | recoverable `Fn3Error::Allocation`; both buffers intact |

---

## 5. Injected matrix: expected / observed

Binary: `tests/fn3_vulkan_faults.rs`, four tests, run on the physical GPU with
validation on. All lines are `[fn3-27][injected]`.

| Case | Expected | Observed |
|---|---|---|
| allocation before submission, decode 14→15 | `Allocation`, 0 submissions, state unchanged, retry = reference | `failed to allocate ... injected allocation failure ... for 128 bytes`; submission delta 0; extent 14/step 1/tokens 14/phase ready unchanged; KV prefix equal to reference; retry byte-identical |
| readback (one logits row) | `Allocation`, no KV append, state unchanged, retry = reference | `injected readback failure ... 201028 bytes at offset 0 from "f32[1, 1, 50257]"`; extent 15 unchanged; retry byte-identical |
| recording failure at dispatch 41 of 144 | plan batch discarded, only the 4 pre-batch prefix copies submitted, state unchanged | `injected recording failure ... the step batch is partial`; submission delta exactly 4 (= layers × keys/values); no partial plan executed; extent 16 unchanged; retry byte-identical |
| KV past-prefix copy (3rd copy, layer 1 keys) | `Allocation`, no state change | `34 region(s), 8704 bytes`; extent 17 unchanged; retry byte-identical |
| KV append mid multi-layer commit (8th copy = layer 1 values, after layer 0 keys/values and layer 1 keys) | `Allocation`, **all** extents rolled back to 18, no double append | `2 region(s), 512 bytes`; per-layer extents `[18, 18]` (pre-step); semantic KV equal to reference at 18; retry reaches 19 and equals reference at 19 with extents 19, not 20 |
| `vkQueueSubmit` device loss | terminal `DeviceUnavailable` naming the adapter, 0 submissions, state unchanged, recover/step/after-reset prefill all terminal, process recovers on a fresh device | as expected; `vkQueueSubmit` counter unchanged; `recover`/`decode`/`prefill` all return the same stored error; fresh backend prefill byte-identical |
| fence device loss | terminal `DeviceUnavailable` naming `vkWaitForFences`, real submission happened, state unchanged, later calls terminal | as expected; `vkQueueSubmit counter` increased before the fence failure; every later call terminal; fresh backend prefill byte-identical |

Every case asserts the post-failure state (`cache_len`, `step`,
`tokens_consumed`, phase, KV bytes, per-layer extents) and, where the KV is
committed, compares the **semantic live prefix** byte-for-byte with a clean
reference session at the same extent. Validation: absolute 0 errors / 0
warnings and delta 0/0 in all four tests.

---

## 6. Repeated reset

### 6.1 Injected matrix (`tests/fn3_vulkan_faults.rs`, `repeated_reset...`)

12 rounds at extents 13, 14, 15, 16, 17, 18 (cycled), resident path; round 3
arms an allocation failure on its last decode, asserts the failure, calls
`recover()`, and retries. After **every** reset:

- `cache_len == 0`, `step == 0`, `tokens_consumed == 0`, phase `Empty`, every
  layer extent 0;
- `kv_bytes` constant at 36,864 B;
- plan cache `(6 plans, 0 evictions)` — the six shapes (prefill 13, decodes at
  past 13..17) are built once and reused;
- allocator `live_bytes` and `live_device_local_bytes` exactly 58,226,256 B
  (persistent weights + KV + past buffers) after the first steady-state reset,
  every round.

```text
[fn3-27][injected] reset baseline: plans (6, 0), live_bytes 58226256, live_device_local 58226256
[fn3-27][injected] repeated reset: 12 rounds at extents 13..=18, one injected failure and
recovery in round 3; extents 0, KV 36864 B, plans (6, 0) and device memory stable
```

### 6.2 Hardware (`tests/fn3_vulkan_reset.rs`, no injection compiled)

Eight clean rounds on the GTX 1080 Ti, resident path, every round checked
against the committed logits and KV fixtures:

- prefill 13 against `prefill.p13.last_logits` and both `present` tensors;
- decode 1 against `decode.step1.logits`, steps 2–4 against their summary
  manifests, extents exactly 14, 15, 16, 17;
- decode 4 KV against `decode.step4.present.{0,1}.{key,value}`;
- reset to empty, KV 34,816 B, plan cache `(5, 0)`, live bytes 27,903,104 B,
  peak 116,794,416 B, stable for rounds 1–7; validation 0/0.

This is hardware evidence: the binary has no fault-injection feature and arms
nothing.

---

## 7. Unavailable features / device absence

- **Simulated missing capability** (`device_info.rs:381`,
  `missing_required_features`). Required before opening a device: a COMPUTE
  queue family, `maxComputeWorkGroupInvocations >= 256` (the 16×16 matmul tile),
  workgroup size x ≥ 64 / y ≥ 16, `maxStorageBufferRange >= 32 MiB` (the 25.7 MB
  embedding matrix), `maxMemoryAllocationSize >= 64 MiB` (the allocator's chunk
  floor). A synthetic adapter with all five deficiencies produces six messages,
  each naming the capability **and both numbers**, and `open_with` turns them
  into one `DeviceUnavailable` naming the adapter, its UUID, the request and the
  available devices (`vulkan/mod.rs:288`). Pooled evidence:
  `[fn3-27][simulated]`.
- **`backend-vulkan` absent at runtime.** `tests/fn3_contracts.rs::
  a_build_without_the_vulkan_backend_refuses_an_explicit_request` runs in the
  default and `--no-default-features` builds: the device list is `[cpu]`,
  `DeviceRequest::Vulkan { index: None }` fails with
  `"this build has no Vulkan backend (the `vulkan` feature is not enabled)"`,
  and an explicit request never resolves to the CPU. 44/44 contract tests pass
  in all three configurations.
- **Device loss** is the terminal contract in §3; the injected matrix checks
  that the error names the adapter and that no CPU fallback occurs (the
  backend's `device_id` stays `Vulkan(0)`; the model is prepared for Vulkan and
  a CPU session constructor would reject it).

---

## 8. Release-mode behaviour

The whole injected matrix was rebuilt and rerun with `--release`:

```text
cargo test --release --features vulkan,vulkan-fault-injection --test fn3_vulkan_faults
test result: ok. 4 passed; 0 failed
```

Same typed errors, same post-failure state, same retry bytes, same validation
0/0. No contract check is a `debug_assert!`; the only `cfg` in the contract is
the injection feature itself. The hardware reset gate was run in debug (it is a
correctness gate, not a timing gate).

---

## 9. Verification commands and real outcomes

All in `/home/petrouil/Projects/github/fastnn-wt-73-runtime`; builds with
`CARGO_BUILD_JOBS=3`.

| Command | Result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo build --no-default-features` | ok |
| `cargo build` | ok |
| `cargo build --no-default-features --features vulkan` | ok |
| `cargo test --lib` | **682 passed** |
| `cargo test --lib --features vulkan` | **730 passed** |
| `cargo test --lib --features vulkan-fault-injection` | **733 passed** (includes the 3 injector unit tests) |
| `cargo test --test fn3_contracts` | 44 |
| `cargo test --no-default-features --test fn3_contracts` | 44 |
| `cargo test --features vulkan --test fn3_contracts` | 44 |
| `cargo test --test fn3_session` | 8 |
| `cargo test --test fn3_e2e` (CPU) | 5 |
| `cargo test --features vulkan --test fn3_e2e` (GTX 1080 Ti) | 10 |
| `cargo test --features vulkan --test fn3_vulkan_resident` | 1 |
| `cargo test --features vulkan --test fn3_vulkan_smoke` | 14 |
| `cargo test --features vulkan --test fn3_vulkan_matmul` | 17 |
| `cargo test --features vulkan --test fn3_vulkan_pointwise` | 18 |
| `cargo test --features vulkan --test fn3_vulkan_attention` | 19 |
| `cargo test --features vulkan --test fn3_vulkan_reset` | 1 (hardware reset gate) |
| `cargo test --features vulkan,vulkan-fault-injection --test fn3_vulkan_faults` | 4 (injected matrix) |
| `cargo test --release --features vulkan,vulkan-fault-injection --test fn3_vulkan_faults` | 4 (release matrix) |
| `cargo clippy --all-targets -- -D warnings` | 0 findings |
| `cargo clippy --all-targets --no-default-features --features vulkan -- -D warnings` | 0 findings |
| `cargo clippy --all-targets --features vulkan,vulkan-fault-injection -- -D warnings` | 0 findings |
| `cargo test --test fn3_malformed` (loader wave) | 8 |
| `cargo test --test fn3_import` (pinned model present) | 42 passed, 1 ignored |
| `cargo test --test fn3_model_format` | 47 |
| `cargo test --test fn3_planner` | 11 |
| `cargo test --test fn3_tensor_storage` | 36 |

Physical GPU commands ran under `flock /tmp/opencode/fn3-gpu.lock`, with
`FN3_VK_DEVICE_UUID`, `FN3_VK_REQUIRE_PHYSICAL=1`,
`FASTNN_FN3_ONNX_MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx`
and `--test-threads=1`. No timing was measured, so the measure lock was not
taken.

---

## 10. Physical-device evidence (hardware, kept separate)

Clean chain, `tests/fn3_e2e.rs` on the GTX 1080 Ti (10 tests): prefill 13 and
cached decode to extents exactly 14, 15, 16, 17; every committed logits fixture
and all 16 KV tensors per point at the documented tolerance; resident arm and
staged arm byte-identical; 0 input readbacks; validation 0 errors / 0 warnings
(absolute and delta). The same binary also runs the long chain and the
no-driver / unsatisfiable-request refusals.

`tests/fn3_vulkan_resident.rs` (1 test): 50 persistent tensors, 27,833,180 B
uploaded once, zero weight bytes after prepare, zero late persistent uploads.

`tests/fn3_vulkan_reset.rs` (1 test): eight clean reset rounds, parity checked
every round, stable extents/KV/plans/memory, validation 0/0 (§6.2).

No injected result appears in this section, and no hardware result is used to
excuse an injected case.

---

## 11. Unsafe inventory for the touched paths

No `unsafe` block was added by this wave. The existing blocks on the failure
paths, and the tests that exercise them:

| File:line | Call | Invariant | Coverage |
|---|---|---|---|
| `vulkan/device.rs:393` | `vkQueueSubmit` | command buffers are recording-complete, owned by this thread's submitter; fence belongs to the submission; queue externally synchronised by `queue_lock` | injected submit case, e2e 10 |
| `vulkan/device.rs:424` | `vkWaitForFences` | fence was created and submitted by this device; wait is for that exact fence | injected fence case, every GPU test |
| `vulkan/device.rs:457` | `vkResetFences` | the submission completed before reset | every GPU test |
| `vulkan/device.rs:557` | `vkDeviceWaitIdle` (teardown) | last handle; every resource holds `Arc<Device>` | every GPU test teardown; ledger/validation checks |
| `vulkan/device.rs:580`, `:584` | destroy cache / device | teardown only, after idempotent wait | every GPU test teardown |
| `vulkan/submission.rs:412` | `vkResetCommandBuffer` (discard) | buffer is recording, not pending; no other thread uses it | injected record case |
| `vulkan/submission.rs:425` | `vkEndCommandBuffer` | recording was begun on this buffer; not in flight | every GPU test |
| `vulkan/submission.rs:446` | `get_query_pool_results` (batch) | all timestamp pairs belong to the completed submission | every resident step |
| `vulkan/submission.rs:527`-`:562` | reset cmd/descriptor pool, begin, reset queries | previous submission completed; buffer recording | every GPU test |
| `vulkan/submission.rs:604` | `vkEndCommandBuffer` (non-batched) | recording begun; not in flight | staged arm / smokes |
| `vulkan/submission.rs:732` | `get_query_pool_results` (wait) | submission completed | every GPU test |
| `vulkan/submission.rs:765`-`:781` | destroy query/descriptor/fence/pool | teardown only, device idle | every GPU test teardown |
| `vulkan/submission.rs:958`, `:990` | `cmd_copy_buffer`, `cmd_pipeline_barrier` | ranges checked before recording; live buffers; same queue family | KV prefix/append, transfers |
| `vulkan/submission.rs:1126`, `:1170`, `:1174` | descriptor alloc/update, bind/dispatch | layout matches spec, set written for every binding, buffer ranges checked and aligned | every dispatch |
| `vulkan/resident.rs` | none | the download/copy paths call the safe `Submitter`/`Recorder` API | readback and KV-copy injection |

`fault.rs`, `kv.rs`, `session.rs`, `device_info.rs`, `error.rs` and
`backend/mod.rs` contain no `unsafe`. The new `FaultInjector` is a `Mutex` over
plain counters and adds no pointer or lifetime assumption.

---

## 12. Loader regressions preserved

The loader wave (`653cdda`) is untouched: `fn3_malformed` 8/8, `fn3_import`
42 passed + 1 ignored (the ignored case is the opt-in trained model), and
`fn3_model_format` 47/47, all re-run on this tree. No loader, parser, fuzz or
unsafe-audit file was edited.

---

## 13. Skips and exact reasons

- **No real device-loss hardware event was induced.** Actually losing the GTX
  1080 Ti on a shared host is not a deterministic or safe test. Device loss is
  injected at the submit and fence seams; that is labelled simulated, and the
  typed-error contract is what is proven. The device physically stayed alive
  (validation 0/0 after each terminal case).
- **No timed measurements.** This is a correctness wave; GPU runs took the GPU
  lock only, not `/tmp/opencode/fn3-measure.lock`.
- **Python/CLI/import/format trees were not edited** (ownership) and no pytest
  run is part of this wave's gate. `fn3_import`/`fn3_model_format` were run
  read-only to preserve the loader regressions.
- **The `--release` hardware reset gate was not rerun in release**; it asserts
  correctness, not timing, and the release matrix covers the failure paths. The
  debug hardware gate is the evidence.

---

## 14. Cross-cutting observations not edited

1. **`Fn3Error::Allocation` loses the requested size for a Vulkan OOM**
   (`vulkan/error.rs`, `Error::Vk { ERROR_OUT_OF_DEVICE_MEMORY }` maps to
   `bytes: 0`). The size is present in the detail string, but a typed field would
   be better; changing the mapping is a shared-error-surface decision, not a
   #73 runtime fix.
2. **`prepare_persistent` is idempotent but not transactional.** If an upload
   fails halfway through the weight list, the already-uploaded entries stay in
   the table and a later session completes it. That is safe (no leak, no stale
   bytes) but not all-or-nothing; worth a documented `prepare` transaction if a
   second model path ever shares a backend.
3. **`VulkanBackend::end_step` reports `Ok(0)` for a discarded partial batch.**
   The execution error the caller receives comes from `plan.execute`, which is
   correct in the session path; a future direct `Backend` user should know that
   a discard is reported by the recording failure, not by `end_step`.
4. **`Submission::Drop` retries `finish_wait` after a failed wait** and prints a
   warning. The device is already poisoned, so this is harmless noise, not a
   state change; it could be suppressed once the device is terminal.

---

## 15. Artifacts

```text
src/fn3/backend/vulkan/fault.rs          injection seam (feature-gated)
tests/fn3_vulkan_faults.rs               injected matrix, 4 tests
tests/fn3_vulkan_reset.rs                hardware repeated-reset gate, 1 test
docs/reports/fn3-27-runtime-hardening.md this report
```

Commit sequence: implementation (code, tests, AGENTS) then this report.
