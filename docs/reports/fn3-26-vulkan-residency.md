# FN3-26 (#72): Vulkan residency, batching and memory

**Owner:** #72 (FN3-26). **Branch:** `fn3/72-residency`.
**Worktree:** `/home/petrouil/Projects/github/fastnn-wt-72-resident`.
**Base:** `672cec9faf089ef0840179f6edfdf33779b576a1` (`docs(fn3): the alpha candidate is complete`).
**HEAD at handoff:** `7ec666f` (three commits, one per measured step).

```text
7ec666f fn3(vulkan): one submission per resident step (#72 step 3)
0a8077e fn3(vulkan): device-resident intermediates and KV cache (#72 step 2)
ea8cd70 fn3(vulkan): persistent weights and device-resident storage (#72 step 1)
672cec9 docs(fn3): the alpha candidate is complete            (base)
```

No remote writes were made: no push, tag, PR, merge or release. The issue was
not commented on or closed. The four shared-contract files (`storage.rs`,
`tensor.rs`, `backend/mod.rs`, `session.rs`) changed additively; §7 lists every
addition for integrator review.

---

## 1. What was measured, and how

Workload (pinned by the brief): the pinned tiny-GPT-2 ONNX graph, prefill of
**10** tokens `464,2068,7586,21831,18045,625,262,16931,3290,13`, then decode of
`640,11,612`. Per iteration a **fresh session** is created over one shared
`PreparedModel`, the prompt is prefilled, the logits are read by a caller (the
materialisation a real caller performs), and the three tokens are decoded with
their logits read.

- 20 measured iterations after **3 warm-up iterations**, release build
  (`cargo test --release`), validation layer enabled in code
  (`VulkanConfig::validated()`).
- Device: NVIDIA GTX 1080 Ti, UUID `471b2bda-51e1-6451-e692-ff5d1e146f1e`,
  selected by UUID; lavapipe present but never selected;
  `FN3_VK_REQUIRE_PHYSICAL=1`; `VK_INSTANCE_LAYERS` unset; validation
  **0 errors / 0 warnings** on every measured run.
- Model: `/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx`,
  sha256 `e18370ba6247c564a4aefb6a83aa86d5d036f47a73c0935e51bbcfe178141a1`,
  verified against `fixtures/parity/tiny-gpt2/model.lock.json` on every run
  (the harness prints the digest and the FNV-1a of the 27,897,974 bytes).
- Harness: `tests/fn3_vulkan_perf.rs`. The **same file** is copied into each
  arm's tree, so all four arms run identical measurement code, model, prompt,
  warmups and iteration count.

### Measurement windows (orchestrator protocol of 2026-11-19)

Each arm ran as `flock /tmp/opencode/fn3-measure.lock -c measure_arm.sh`, which
**waits for a host with no `cargo`/`rustc` process** before starting, then runs
the release build and the timed run inside the window; the timed run itself is
additionally inside `flock /tmp/opencode/fn3-bench.lock`. `measure_arm.sh`
records the wall-clock boundaries, the load average and both `pgrep` snapshots;
the raw logs are committed next to this file.

| Arm | measure lock acquired | waited for a quiet host | build started (load) | `pgrep` before timed run | run started | load after | loadavg recorded by the run |
|---|---|---|---:|---|---|---:|---|
| base `672cec9` | 11:06:54 | 0 s¹ | 11:06:54 (2.26) | none | 11:09:0x | 2.41 | 2.41 |
| step 1 `ea8cd70` | 11:09:29 | 0 s | 11:09:34 (1.99) | none | 11:11:0x | 4.55 | 4.55 |
| step 2 `0a8077e` | 11:12:16 | 375 s | 11:18:31 (0.68) | none | 11:20:1x | 2.38 | 2.38 |
| step 3 `7ec666f` | 11:35:54 | 0 s | 11:36:11 (1.24) | none | 11:37:4x | 2.46 | 2.46 |

¹ The base window overlapped the tail of the #71 owner's `fn3_cpu_perf` build
(`cargo test --release --test fn3_cpu_perf --no-run`). It finished before the
base arm's build started and `pgrep` was empty before the timed run, so the
*timed* window was quiet (load 2.26 → 2.41). Every other arm waited for an
empty `pgrep` before building. The measured traffic, device time and allocation
counts are deterministic (they are identical to the earlier interleaved runs);
only the host wall clock varies with contention, which is why §2 quotes the
*minimum* and the full sample files.

Owner's matched baseline at `672cec9`, for comparison: CPU prefill 240.31 ms
p50 / decode 158.33 ms per token; Vulkan prefill 204.76 ms / decode 190.64 ms,
upload 318,880,260 B, readback 106,941,352 B, 575 submissions per chain,
11,500/11,500 proven executions, 0 input readbacks, validation 0/0. Our base
arm reproduces the Vulkan traffic **byte for byte** (318,880,260 B upload,
106,941,352 B readback, 575 submissions, 575 executions per chain); its wall
clock is lower because this host was quieter (load 2.4 versus the owner's run).

---

## 2. The three step tables

`upload`/`readback` are **per chain** (one prefill + three decodes) unless
stated. "ledger submissions" is the count in the residency ledger's records
(per-op submissions, or batch submissions after step 3); "vkQueueSubmit" is the
device-level counter (§6).

| Metric | base `672cec9` | step 1 `ea8cd70` | step 2 `0a8077e` | step 3 `7ec666f` |
|---|---:|---:|---:|---:|
| prefill TTFT p50 (ms) | 142.360 | 142.947 | **16.334** | **9.138** |
| prefill TTFT p95 (ms) | 182.221 | 169.314 | 18.569 | 9.389 |
| prefill p50 min (ms) | 138.305 | 125.529 | 15.484 | 8.849 |
| decode per token p50 (ms) | 130.444 | 124.558 | **16.437** | **9.314** |
| decode p95 (ms) | 167.913 | 157.129 | 18.544 | 9.682 |
| decode p50 min (ms) | 122.976 | 112.672 | 15.549 | 9.063 |
| upload bytes / chain | 318,880,260 | 104,601,668 | 5,328 | 5,328 |
| readback bytes / chain (op records) | 106,941,352 | 106,941,352 | 0 | 0 |
| readback bytes / chain (logits download) | — | — | 804,112 | 804,112 |
| proven executions / chain | 575 | 575 | 575 | 575 |
| ledger submissions / chain | 575 | 575 | 575 | **4** |
| `vkQueueSubmit` / prefill step | 143 | 143 | 143 | **6** |
| `vkQueueSubmit` / decode step | 144 | 144 | 144 | **10** |
| device-busy ms / chain | 83.60 | 51.10 | 14.61 | 14.06 |
| device peak live bytes | 120,669,184 | 98,572,080 | **62,306,800** | 115,785,616 |
| device live bytes at end | 16,777,216 | 44,610,688 | 32,027,776 | 27,833,472 |
| allocator suballocations / 20 iterations | 35,194 | 30,737 | **9,666** | 10,953 |
| host peak RSS (VmHWM) | 415,653,888 | 371,761,152 | **292,630,528** | 294,846,464 |
| staging blocks created / 20 iterations | 320 | 160 | **0** | 1,120 |
| input readbacks | 0 | 0 | 0 | 0 |

Raw JSON, per-sample CSV and per-op tables: `docs/reports/fn3-26/{base,step1,step2,step3}.json`,
`.csv`, `-per-op.txt`; the full harness logs with the measurement windows are
`…-measured.log`.

### 2.1 Reading the wall clock honestly

Host wall clock is the comparable metric the orchestrator asked for, and it is
also the noisiest thing on a shared host: one fence wait per submission means a
contended host moves p50 by tens of milliseconds between runs of the *same*
binary (the base arm alone measured p50 142.4, 142.9, 144.3, 147.1, 156.8 and
194.8 ms across six runs at loads 2.4–6.1). The conclusions that do not depend
on that noise:

- **step 1** (weights resident) removes 67% of the upload bytes and 39% of the
  device-busy time but leaves the wall clock unchanged: with 575 per-op fence
  waits, the host round trip dominates.
- **step 2** (resident intermediates and KV) removes 99.998% of the upload
  bytes and 83% of the device-busy time, and the wall clock follows:
  16.3 ms/step, an 8.7× / 7.9× improvement.
- **step 3** (one submission per step) leaves device work unchanged
  (14.61 → 14.06 ms/chain, i.e. noise) and halves the wall clock again
  (16.3 → 9.1 ms), because 575 submissions become 4 plan submissions per chain.
  That is the submission-count change made visible.

The owner's note that GPU timestamps are not yet stable applies to the
*aggregate* device span; the per-op pairs this work attaches are stable enough
to explain the submission change, and they are never used as the headline
metric.

### 2.2 Where the bytes went (per op, per iteration = 4 steps)

Base, upload bytes per iteration (top 5 of 20 ops):

```text
matmul      109,427,328   gather 105,023,696   transpose 103,026,688
add             386,612   mul        326,816
```

Step 1 (weights resident, intermediates still host):

```text
matmul      103,135,872   transpose 100,352   gather 208   add 349,636   mul 326,656
```

Step 2/3 (resident intermediates and KV):

```text
reduce_mean       4,640   unsqueeze 368   gather 208   where 112   everything else 0
```

The remaining uploads are kernel *parameter blocks*, not tensors: a
`reduce_mean` parameter block is 116 B, an `unsqueeze` push-constant block 46 B.
The graph inputs (token ids 208 B, positions 208 B, mask 272 B per chain) are
the only tensor bytes that still cross the host bridge, plus host constants
(~4.3 KiB per chain).

`transpose` is worth naming: at the alpha the plan transposes the 25.7 MB
token-embedding matrix per step, uploading 51.5 MB and reading 51.5 MB back
per iteration. With persistent weights its *input* is resident (upload → 100 KB
of parameter blocks), and with resident slots its *output* stays on the device
(readback → 0). It still costs one device copy per step; hoisting a pure
transpose of a persistent tensor out of the step is planner work (#56), not
backend work, and is listed in §9.

---

## 3. The weight-reupload-zero proof

`tests/fn3_vulkan_resident.rs` (physical GPU, validation on):

```text
[fn3-26][prepare] persistent_tensors=50 persistent_bytes=27833180 prepare_upload_bytes=27833180
[fn3-26][prefill] queue_submissions=6 persistent_hits=47 staged_uploads=4 staged_upload_bytes=481 late=0 executions=143 ledger_submissions=1 upload_bytes=1644 readback_bytes=0
[fn3-26][decode]  queue_submissions=30 persistent_hits=141 staged_uploads=12 staged_upload_bytes=411 late=0 executions=432 ledger_submissions=3 upload_bytes=3900 readback_bytes=0
[fn3-26][validation] errors 0 warnings 0
```

What is asserted, per phase:

| Counter | prepare | prefill | decode |
|---|---|---|---|
| `persistent_tensors` | 50 (every contiguous non-empty weight) | unchanged | unchanged |
| `persistent_bytes` | 27,833,180 (= the model's weight bytes) | unchanged | unchanged |
| `persistent_prepare_bytes` | 27,833,180 | **0** | **0** |
| `persistent_hits` | — | **47** | **141** |
| `late_persistent_uploads` | **0** | **0** | **0** |
| `staged_upload_bytes` | — | 481 (tokens+positions+mask) | 411 |

Three independent facts make this a proof rather than an observation:

1. **It is enforced.** `VulkanBackend::check_staged_input` is called by the
   operand resolver before any host upload. A tensor that
   `prepare_persistent` registered and that then reaches the staging path is a
   hard error (`Error::internal`) and increments `late_persistent_uploads`; a
   kernel that forgets the resident path fails instead of quietly moving
   27.8 MB.
2. **Prepare is complete.** `persistent_tensors == 50` and
   `persistent_prepare_bytes == 27,833,180` are exactly the weight inventory
   (24 `f32` + 24 `i64` initializers plus 2 tied matrices that share storage);
   the test computes count and bytes from `model.weights()` with the same rule,
   not from a constant.
3. **A second session is free.** Creating a second `Session` over the same
   `PreparedModel` after resetting the counters moves **0** bytes and the
   second session produces byte-identical logits (`assert_eq!` on the prompt
   logits).

Host tensors are untouched: the CPU path and the `FN3_VK_STAGED=1` path read
the same host storage they always did (the staged arm of the end-to-end gate
reads every KV tensor and every logits fixture through it).

---

## 4. Submission and allocation counts, explained

### 4.1 Submissions

| Path | Submissions per prefill step | per decode step | why |
|---|---:|---:|---|
| base / step 1 / step 2 (staged or resident, no batching) | 143 | 144 | one `vkQueueSubmit` per byte-producing instruction (143 prefill, 144 decode: the prefill has one zero-element no-op) |
| step 3, staged arm (`FN3_VK_STAGED=1` or `SessionOptions::staged()`) | 143 | 144 | unchanged by design: a staged op reads each output back, so it cannot defer |
| step 3, resident arm | **6** | **10** | 1 plan batch + up to 4 KV past-prefix copies + 4 KV appends + 1 logits download |

Breakdown of a resident decode step (measured: 30 `vkQueueSubmit` for three
decode steps, 6 for one prefill step):

```text
1  plan batch   : every dispatch of the plan, all past-prefix copies, the
                  step's input uploads, in one command buffer
4  KV appends   : keys and values, one region list each per layer (2 layers)
1  logits       : the explicit last-position row download
```

A prefill step has no past prefix yet, so it is 1 + 4 + 1 = 6. The appends and
the download happen after `end_step` because the append validates the `present`
outputs against the committed extent and the logits must be produced before
they can be read; both are one submission per operand, not per row (the region
list is what makes that possible).

Verification that the change is *only* the submission count: the ledger still
holds one record per byte-producing instruction with the same dispatch
geometry and bindings (719 records for the 13+4 chain), each record now carries
`batch = Some(step)` and its own `(start, end)` device-timestamp pair, and
`proves_device_execution()` accepts "joined a batch" as proof. The staged arm
still produces `host_staged = true` on every record and byte-identical logits.

### 4.2 Allocations

| Arm | suballocations / 20 iterations | device peak live | device live at end | staging blocks created |
|---|---:|---:|---:|---:|
| base | 35,194 | 120.7 MB | 16.8 MB | 320 |
| step 1 | 30,737 | 98.6 MB | 44.6 MB (weights) | 160 |
| step 2 | 9,666 | 62.3 MB | 32.0 MB | 0 |
| step 3 | 10,953 | 115.8 MB | 27.8 MB | 1,120 |

Step 2 is the allocation win: per-op device buffers disappear for intermediates
(they are plan slots, allocated once per shape and recycled by the existing
slot pool) and the KV past prefixes are capacity-sized buffers refreshed in
place. Step 3 *costs* allocation headroom: a batch cannot free an op's buffer
before the fence, so a whole step's buffers are live at once (peak 62 → 116 MB)
and a staged upload block is retained per recording (0 → 1,120 blocks created
per 20 iterations). Both are honest consequences of one submission per step and
are the first thing a buffer-reuse arena would fix (§9). Note that the
*persistent* weight footprint (27.8 MB) is part of every step-2/3 figure; base
paid it as transient uploads instead.

---

## 5. Verification after each step

Every step was committed only after the whole gate below was green at that
commit. Counts are the number of passed tests.

| Check | base | step 1 | step 2 | step 3 |
|---|---|---|---|---|
| `cargo fmt --check` | clean | clean | clean | clean |
| `cargo test --lib` | 678 | 678 | 678 | 678 |
| clippy `-D warnings`, default | 0 findings | 0 | 0 | 0 |
| clippy `--no-default-features` | 0 | 0 | 0 | 0 |
| clippy `vulkan` / `cli` / `fn3` / `fn3,vulkan` | 0 / 0 / 0 / 0 | 0 | 0 | 0 |
| `fn3_contracts` | 43 | 43 | 43 | 43 |
| `fn3_cpu_reference` | 43 | 43 | 43 | 43 |
| `fn3_tensor_storage` | 36 | 36 | 36 | 36 |
| `fn3_import` | 27 | 27 | 27 | 27 |
| `fn3_model_format` | 38 | 38 | 38 | 38 |
| `fn3_planner` | 11 | 11 | 11 | 11 |
| `fn3_session` | 8 | 8 | 8 | 8 |
| `fn3_ci_smoke` | 5 | 5 | 5 | 5 |
| `fn3_cli` | 6 | 6 | 6 | 6 |
| `fn3_e2e` CPU | 5 | 5 | 5 | 5 |
| `fn3_e2e` Vulkan | 5 | 5 | 5 | 5 |
| `fn3_vulkan_smoke` (physical) | 14 | 14 | 14 | 14 |
| `fn3_vulkan_matmul` (physical) | 17 | 17 | 17 | 17 |
| `fn3_vulkan_pointwise` (physical) | 18 | 18 | 18 | 18 |
| `fn3_vulkan_attention` (physical) | 19 | 19 | 19 | 19 |
| `fn3_vulkan_resident` (new, physical) | — | 1 | 1 | 1 |
| validation errors / warnings | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

### 5.1 Parity evidence carried through every step

`fn3_e2e` proves, on the physical device and with validation on:

- every committed logits fixture (`prefill.p1`, `p4`, `p13`, `decode.step1..4`)
  and all 16 KV tensors per prefill/decode point, at atol 1e-5 / rtol 1e-4, with
  the committed mean-abs bound;
- cache extents exactly `14,15,16,17` for the 13+4 chain, and `14..=32` for the
  long chain, with a constant 65,536-byte cache and 0 plan evictions;
- session isolation (two sessions over one model), reset, capacity overflow
  (a refused 18th token leaves extent, step, tokens, phase and every KV byte
  unchanged) and cached-vs-full equivalence at two context lengths;
- one residency record per byte-producing instruction with per-op histogram
  equality, `inputs_read_back == 0`, every binding device-local and
  `proves_device_execution()` true for all records.

Step 2 and 3 add to that: **the resident arm and the staged arm run the same
13+4 chain in the same process and their logits are compared byte for byte**
(`assert_eq!` on every step). The resident arm asserts
`stats.readback_bytes == steps × 50,257 × 4` — one logits row per step, nothing
else — and an upload budget of 2 MiB per step (measured: 6,860 B for 5 steps).
The staged arm asserts `host_staged == true` on every record, no resident bind,
and the alpha's per-step transfer volume (measured 130,945,760 B upload /
133,974,624 B readback for the 5-step chain).

---

## 6. Shared-contract changes for integrator review

All additive; no existing signature, default or invariant was weakened.

### `src/fn3/storage.rs`

- `trait DeviceResidency` (`backend`, `bytes`, `describe`, `as_any`,
  `download(offset, dst)`), implemented by the Vulkan backend. `storage.rs`
  keeps no dependency on `ash`.
- `Storage` gains an optional residency attachment and a logical length:
  `from_resident`, `is_resident`, `residency`, `residency_label`, `identity`,
  `host_bytes`, `download`, `download_range`. `len()` is the logical length for
  both kinds so window/bounds checks do not branch.
- Host byte accessors on a resident storage return
  `Fn3Error::NotImplemented { owner: "FN3-26 (#72)" }` naming
  `Storage::download`/`Tensor::to_host`; `as_bytes()` is empty (there is no
  host copy, so nothing stale can be returned); `try_make_mut`/`try_into_buffer`
  are rejected. `try_as_bytes` was added to `StorageWindow`.
- `Fn3Error` was **not** modified (no new variant); the remedy is carried in the
  `what` string.

### `src/fn3/tensor.rs`

- `Tensor::is_device_resident`, `residency`, `to_host`; `from_storage` now
  accepts resident storage.
- `TensorView::storage`, `host_bytes`, `is_device_resident`, `to_host`;
  `storage_bytes` documents that it is empty for a resident view.
- `TensorMut` has a device backing (`TensorMut::resident`), plus
  `is_device_resident`, `resident_storage`, `host_bytes`, `host_bytes_mut`.
  `as_slice`/`as_mut_slice`/`storage_slice_mut` return the typed error for a
  resident view instead of reading empty bytes.
- `Tensor::as_mut_view` keeps its exclusive-borrow rule for resident storage
  (`StorageShared` while another handle lives).

### `src/fn3/backend/mod.rs`

New `Backend` methods, every one with a host/no-op default so the CPU backend is
bit-for-bit unchanged:

| Method | Default | Purpose |
|---|---|---|
| `supports_resident_storage()` | `false` | the session's residency decision |
| `allocate_resident(shape, dtype)` | `NotImplemented` | device tensor allocation |
| `prepare_persistent(&[&Tensor])` | `Ok(())` | upload immutable weights once |
| `write_resident(storage, offset, bytes)` | `NotImplemented` | step-boundary input upload |
| `copy_resident_regions(dst, src, regions)` | `NotImplemented` | strided device copies in one submission |
| `note_readback(bytes)` | no-op | readback accounting |
| `begin_step(label, batch)` / `end_step()` | no-ops | one submission per step; `batch` is the session's decision and is `false` whenever a host read is needed |

`Backend::admits` and the support contract are untouched.

### `src/fn3/model/session.rs`

- `SessionOptions { staged }` with `SessionOptions::staged()`;
  `Session::with_runtime_options`. `with_runtime` is now a thin wrapper with
  `SessionOptions::default()`. `FN3_VK_STAGED=1` still forces staging.
- `Session` exports `SessionOptions` from `model/mod.rs`.
- A step opens one backend scope (`begin_step`/`end_step`), always closed —
  including on failure — before the logits are read.

### `src/fn3/model/kv.rs`

- `KvCacheState::new_resident`, `is_device_resident`, `live_with`, `append_with`
  (a superset of `append`, which now delegates with no executor). The
  two-phase check/commit protocol, the capacity rules and the failure atomicity
  are unchanged and shared by both paths.

### `src/fn3/model/mod.rs`

- Re-export `SessionOptions`. Nothing else.

---

## 7. Experiments that were tried and rejected

Recorded so the integrator does not have to re-derive them.

1. **Resident graph inputs.** Making the token ids, positions and mask resident
   is possible, but `embedding.gather` validates index values on the host (the
   documented `[-dim, dim)` check), so a resident index tensor turns the gather
   into a host read of a device buffer. The fix would be to download the indices
   per gather (a per-op readback inside the graph, exactly what #72 removes) or
   to move the range check into the shader (a kernel change with no measured
   benefit at 208 B/step). Graph inputs therefore stay host-backed, which the
   contract already allows: they are the step-boundary upload.
2. **Moving `begin_step` before the KV prefix binding** would fold the four
   past-prefix copies into the plan batch (10 → 6 submissions per decode step).
   It requires every fallible bind to be wrapped so the scope is always closed;
   at 4 × ~0.15 ms per step it was not worth destabilising a green step. Listed
   in §9.
3. **Caching the transposed embedding matrix.** A pure transpose of a
   persistent weight could be computed once, not per step. That is plan-level
   constant folding (#56), not a backend decision; the measured cost left on the
   table is one 25.7 MB device copy per step.
4. **Batching the staged path.** Impossible without weakening the model: its ops
   read their outputs back, and a deferred submission cannot satisfy a host
   read. A batched op that has a host output now returns a typed error.

---

## 8. Commands and their real outcomes

Run in `/home/petrouil/Projects/github/fastnn-wt-72-resident` unless stated.
`CARGO_BUILD_JOBS=5` on every build; `FN3_VK_DEVICE_UUID` and
`FN3_VK_REQUIRE_PHYSICAL=1` on every GPU command.

```text
cargo fmt --check                                    clean (after each fmt pass)
cargo test --lib                                     678 passed
cargo clippy --all-targets -- -D warnings            0 findings
cargo clippy --all-targets --no-default-features -- -D warnings
cargo clippy --all-targets --features vulkan -- -D warnings
cargo clippy --all-targets --features cli -- -D warnings
cargo clippy --all-targets --features fn3 -- -D warnings
cargo clippy --all-targets --features fn3,vulkan -- -D warnings
cargo test --test fn3_contracts                      43 passed
cargo test --test fn3_cpu_reference                  43 passed
cargo test --test fn3_tensor_storage                 36 passed
cargo test --test fn3_import                         27 passed
cargo test --test fn3_model_format                   38 passed
cargo test --test fn3_planner                        11 passed
cargo test --test fn3_session                        8 passed
cargo test --test fn3_ci_smoke                       5 passed
cargo test --test fn3_cli                            6 passed
cargo test --test fn3_e2e                            5 passed (CPU)
cargo test --features vulkan --test fn3_e2e -- --nocapture --test-threads=1
                                                     10 passed (5 CPU + 5 Vulkan)
cargo test --features vulkan --test fn3_vulkan_smoke -- --test-threads=1
                                                     14 passed
cargo test --features vulkan --test fn3_vulkan_matmul -- --test-threads=1
                                                     17 passed
cargo test --features vulkan --test fn3_vulkan_pointwise -- --test-threads=1
                                                     18 passed
cargo test --features vulkan --test fn3_vulkan_attention -- --test-threads=1
                                                     19 passed
cargo test --features vulkan --test fn3_vulkan_resident -- --nocapture --test-threads=1
                                                     1 passed (weight persistence proof)
FN3_PERF=1 cargo test --release --features vulkan --test fn3_vulkan_perf -- --nocapture
                                                     1 passed per arm (see §2)
```

Two failures were real and were fixed rather than relaxed:

- after step 1 a `Bool` output of one byte was read back into a four-byte padded
  buffer (`readback destination requested 4, capacity 1`); the `Job` readback now
  records the logical length and copies the padded transfer;
- after the first batching attempt the p1 logits were wrong
  (`0.33639297 vs 0.37179`): one submission gives no implicit inter-dispatch
  ordering. Each captured dispatch now records an explicit compute→compute
  barrier over its bindings, and the fixtures match to 6.6e-7.

## 9. Skipped checks and remaining risks

- **`uv run pytest tests/ -m "not slow"`** was not run: the Python tree is
  outside this work and the alpha's Python lane is #64/#67's; nothing in this
  branch touches `src/fn3/python/**` or `fastnn/fn3/**`. The gate the brief
  specifies for this issue does not include it.
- **`cargo test --test quantized_pipeline`, `optim_test`,
  `autograd_gradient_checks`, `cpu_arena_telemetry`** were not run: they are
  superseded for `src/fn3/` by `docs/roadmap/fn3-architecture.md` §1.1.
- **Long-chain parity beyond position 17** remains structural (finite logits,
  exact extents, determinism), because the committed fixtures stop at 17. That
  is unchanged from the alpha.
- **Wall-clock variance** on this shared host is large (§2.1). The traffic,
  allocation and device-time columns are deterministic; the wall clock is not,
  and the raw samples are committed so the integrator can re-derive p50/p95.
- **Step 3 raises peak device memory** (62 → 116 MB) and staging-block creation,
  because a batch holds a step's buffers and upload blocks until its fence. A
  per-step buffer arena (reuse one output buffer per (shape, dtype) within a
  batch) would recover it; it is not done here because it changes the
  allocator's lifetime contract and deserves its own measured step.
- **10 `vkQueueSubmit` per decode step, not 1**: the four KV appends and the
  logits download are outside the plan batch by construction (§4.1, §7.2).
- **`FN3_VK_STAGED=1` is honoured at session construction**, so a process that
  sets it after creating a session keeps the resident path for that session.
  Documented on `SessionOptions`; the supported switch is the environment
  variable or the option, both read at construction.
- **The persistent table is per backend, never evicted.** A model with many
  shapes and several sessions over one `Arc<VulkanBackend>` keeps one copy of
  each weight, which is the intent; a bounded cache would be needed for a
  server with many models per process.

---

## 10. Note for #70 (FN3-24): W8A16 kernels on the resident path

The quantized format is locked as **W8A16** (`fn3-quant-format-v1`): signed INT8
weights, one byte per element, groups of 64 along K, binary16 scale and int16
zero-point per `(row, group)`, FP32 activations, no nibble interleaving and no
dequantized weight buffer.

This is what the surface added here gives that kernel family, and what it
requires in return.

**Weights arrive already resident.** `Backend::prepare_persistent(&[&Tensor])`
is the only upload path for immutable model data, and the end-to-end gate proves
a second call moves zero bytes. A quantized artifact contributes three tensors
per quantized matrix — `int8 codes [rows, K]`, `binary16 scales [rows, K/64]`,
`int16 zero_points [rows, K/64]` — and all three are registered at prepare and
bound by handle for the life of the backend. No new API is needed: pass them in
the same slice. The residency counter classifies them as persistent, so the
"zero weight bytes after prepare" test extends to them unchanged.

**The GEMM reads codes in place.** `resident_input`/`operand_source` bind the
codes, scales and zero-point buffers directly; a W8A16 matmul takes three
device bindings plus the FP32 activation. Dequantization must be fused in the
kernel: keep `q`, `scale`, `zero` in device memory, accumulate `Σ x·(q − z)` in
FP32 per 64-element group, then apply the binary16 scale once per group. The
resident path does not offer, and must not grow, a dequantized weight buffer —
that would be a per-prepare 4× expansion of exactly the bytes this issue
removed from the step.

**No staging-only API.** Every kernel entry point already receives
`Operand`/`OperandMut` (resident buffer or host window) and never sees a
staging buffer it could assume. A quantized kernel must implement the same two
cases: bind the resident buffer, or stage a host window. It may not require a
host copy of the codes; `resident_input` returns the buffer, and
`check_staged_input` will refuse a prepared quantized tensor that reaches the
upload path.

**Tile-level dequantization is allowed, whole-tensor is not.** A backend without
a fused affine may dequantize a *tile* into scratch, and that scratch must be
accounted for in the residency budget: it is a per-dispatch buffer like any
other `Job` buffer, so it appears in `ResidencyStats::resident_tensors` /
`MemoryStats::peak_live_bytes`. The measured headroom on this device (GTX
1080 Ti, 11 GiB heap, `maxMemoryAllocationSize` ≈ 4 GiB; peak live 116 MB at
step 3 for the tiny model) is ample for tile scratch, but the report must state
the tile size and its bytes.

**Batching composes.** A quantized op records into the same step batch as
everything else, so it must record its own compute→compute barriers over its
bindings (the `Job`/`execute_staged` pattern) and may not read a host output
while batching. Integer accumulation stays in integer space: the batch is
`vkCmdDispatch` after `vkCmdDispatch`, not a Q/DQ pair per element.

**dtype and capability.** `DType` has no quantized variant and `OpId` has no
quantized op in the alpha; #70 adds both. `CapabilityFeatures::quantized`
already exists and is `false`; a W8A16 registration flips it only when every
kernel it needs is registered, and `KernelSet::alpha` already replaces a
registration by op id, so no shared file needs to change to add the family.

---

## 11. Artifacts

```text
docs/reports/fn3-26-vulkan-residency.md      this report
docs/reports/fn3-26/base.json                base     matched JSON result
docs/reports/fn3-26/step1.json               step 1   matched JSON result
docs/reports/fn3-26/step2.json               step 2   matched JSON result
docs/reports/fn3-26/step3.json               step 3   matched JSON result
docs/reports/fn3-26/{base,step1,step2,step3}.csv          raw per-sample timings
docs/reports/fn3-26/{base,step1,step2,step3}-per-op.txt   per-op traffic/time
docs/reports/fn3-26/{base,step1,step2,step3}-measured.log measurement windows,
                                                          load, pgrep, full run
tests/fn3_vulkan_perf.rs                     the portable harness (all arms)
tests/fn3_vulkan_resident.rs                 the persistence proof
```
