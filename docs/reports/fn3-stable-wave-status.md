# FastNN 3.0 stable wave — status report

Status as of integration HEAD **`a3e9723`** on `rewrite/fastnn-3`, pushed to
`origin`. The alpha review commit **`672cec9`** remains an ancestor and is
identifiable in every merge. `main` is untouched at `a2c0759`; no tag, release,
package publication or PR has been created.

This report covers the alpha result, every stable workstream merged so far, the
verification actually reproduced by the integrator, and the open items that
block a stable candidate. It is written to be read with the raw evidence files
it cites, not instead of them.

---

## 1. Executive status

**Delivered and merged**

| Area | Result |
|---|---|
| Alpha inference gate | FP32 prefill + cached decode on portable CPU and the physical GTX 1080 Ti, parity 5.96e-7 / 6.56e-7, 719/719 device executions, 0 input readbacks, validation 0/0 |
| Vulkan residency (#72) | Weights uploaded once at prepare; intermediates and KV device-resident; one submission per resident step. Prefill 142.36 → 9.14 ms, decode 130.44 → 9.31 ms, upload 318.9 MB → 5.3 KB per chain |
| CPU kernels and threading (#71) | Column-tiled prefill, blocked/AVX2 MatMul, cache-friendly transpose, adaptive Auto budget. Trained prefill 1.38× at 16 threads; decode 1.04×; Auto within 0.95–1.05 of best fixed |
| Quantization format (#68) | Grouped W4A8 evaluated first and rejected for the starter; **W8A16** locked as `fn3-quant-format-v1`, with a portable byte layout, independent Python reference and hand-checkable fixtures |
| CPU quantized path (#69) | Fused W8A16 projection: trained prefill 1.48×, decode 1.10×, projections 26.56% of FP32 storage, top-1 preserved on every measured step |
| Vulkan quantized path (#70) | Fused W8A16 resident kernel: 0 upload / 0 readback per quantized op, 417,792 B packed payload prepared once on tiny; trained path staged only because of one resident defect |
| Artifact-only execution | `.fnn3` prepares once and runs with the source ONNX moved away, across Rust, CLI and Python, with identical top-5 |
| Loader hardening (#73) | Budgets, five real defects fixed, 38 malformed cases, 14,072,087 fuzz executions with zero crashes, Miri and unsafe inventory |
| Runtime hardening (#73) | Recoverable/terminal failure contract, KV rollback, repeated reset, unavailable-feature errors, injected and physical evidence kept separate |
| Alpha follow-up | Matched release-build CPU/Vulkan baseline, strict wheel inference, crate/wheel metadata, hosted CI validation-layer repair |
| Trained FP32 investigation | Reproduced the 5e-5 exceedance, isolated it to reference rounding rather than a fn3 defect, and found one real resident-Vulkan defect |

**Blocking stable**

1. The trained resident `Reshape(input_ids) → Gather(wte)` defect. A fix is in
   progress; until it lands, trained resident Vulkan is not claimed and #70 is
   open.
2. The trained FP32 `mean_abs` tolerance decision. The evidence says the 5e-5
   bound is below the reference implementation's own float64 error; the bound
   has **not** been changed. An owner decision is pending.
3. #74–#76: hardware/platform matrix, docs/evaluation, and the stable gate
   itself, including the 3.0.0 version bump, hosted CI on the exact final SHA,
   physical-GPU validation, and a clean-install artifact-only replay.

---

## 2. Governance and method

- The orchestrator is the single integration writer. Workers commit locally on
  their own branch, never push, tag, merge, release, publish, comment on issues
  or open a PR. The alpha follow-up owner was the sole exception, limited to
  its own branch and `gh workflow run`.
- Every workstream got a separate worktree based on the then-current
  integration HEAD. Shared contracts were owned by one worker at a time; the
  integrator resolved the one real conflict (`cpu/mod.rs`) and reviewed every
  additive API change.
- Correctness is asserted against independent references: ONNX Runtime and
  PyTorch fixtures for FP32, a from-scratch Python unpacker for the quantized
  format, and a float64 graph evaluation for the trained FP32 investigation.
  Self-consistency is never accepted as parity.
- A skipped lane is not a pass. Hardware-unavailable status names the device,
  the steps taken, and what it blocks. Software Vulkan never satisfies a
  physical-device gate.
- Timed runs take `/tmp/opencode/fn3-measure.lock`; builds and timed runs are
  never overlapped with another owner's work, and `/proc/loadavg` is recorded
  with every table.
- CUDA (#77) and training (#78) remain deferred and untouched.

---

## 3. Alpha baseline (commit `672cec9`)

| Evidence | CPU | GTX 1080 Ti |
|---|---:|---:|
| `prefill.p1` max abs | 5.9604645e-7 | 6.556511e-7 |
| `prefill.p4.last` | 5.159527e-7 | 5.364418e-7 |
| `prefill.p13.last` | 5.364418e-7 | 5.9604645e-7 |
| `decode.step1` | 6.109476e-7 | 8.34465e-7 |
| Cache extents | 14/15/16/17 exact | same |
| Device executions proven | — | 719/719 |
| Input readbacks | — | 0 |
| Validation | — | 0 errors / 0 warnings |

Cached-vs-full decode was regenerated rather than trusted and measured
**bit-identical (0.0)** at all four steps. The top-5 for the 10-token prompt
`464,2068,7586,21831,18045,625,262,16931,3290,13` is
`[46737, 28597, 16571, 7868, 13]`, identical from the Rust CLI, the Python API
and a wheel installed into a clean venv outside the checkout.

Alpha gates: 673 library tests, 222 across ten `fn3_*` binaries, 68 Vulkan
tests, 378 pytest, `clippy -D warnings` clean in six configurations, versions
`3.0.0-alpha.1` / `3.0.0a1` agreeing across seven stores.

Two design decisions carry the alpha's evidential weight:

- `Backend::admits` was added instead of widening `Partial` verdicts to `Full`.
  It is consulted only for a `Partial` registration, returns
  `Result<(), String>`, and cannot invent `Unsupported` or promote itself.
- Prefill is honestly **143/144** executions because one `ConstantOfShape`
  fills the zero-element half of the causal mask at `past = 0`. A fake dispatch
  to reach a round 144 was refused.

---

## 4. Workstream results

### 4.1 #72 — Vulkan residency, KV and batching

Branch `fn3/72-residency`, HEAD `7d4e6e5`; merged as **`dd1dd7f`**. Four
commits, 41 files, +5,457/−603.

| Metric (release, 10-token prefill + `640,11,612`) | base | step 1 | step 2 | step 3 |
|---|---:|---:|---:|---:|
| Prefill p50/p95 ms | 142.36/182.22 | 142.95/169.31 | 16.33/18.57 | **9.14/9.39** |
| Decode p50/p95 ms/token | 130.44/167.91 | 124.56/157.13 | 16.44/18.54 | **9.31/9.68** |
| Upload B/chain | 318,880,260 | 104,601,668 | 5,328 | **5,328** |
| Readback B/chain | 106,941,352 | 106,941,352 | 804,112 | **804,112** |
| Ledger submissions/chain | 575 | 575 | 575 | **4** |
| `vkQueueSubmit`/decode | 144 | 144 | 144 | **10** |
| Device peak / host RSS MB | 120.7/415.7 | 98.6/371.8 | 62.3/292.6 | 115.8/294.8 |

Enforced proofs:

- Prepare uploads 50 persistent tensors / 27,833,180 B once; a second session
  moves **0 B** and produces identical logits. Prefill/decode record 47/141
  persistent hits and **zero late uploads**. A prepared tensor reaching the
  host upload path is a hard error.
- Resident readback is exactly `steps × 50,257 × 4` = 804,112 B; input
  readbacks 0.
- Resident and `FN3_VK_STAGED=1` arms run the same chain in one process with
  **byte-identical logits**. The staged arm still records `host_staged = true`
  and 719 per-op submissions.
- Two real defects were fixed rather than relaxed: a padded Bool readback
  length, and missing inter-dispatch barriers (the first batching attempt
  produced wrong logits; now 6.6e-7 max abs).

Recorded limits: a decode step is still 10 `vkQueueSubmit` because four KV
appends and the logits download sit outside the plan batch; batching raises
device peak from 62 to 116 MB until the staging arena is pooled; the persistent
table is per backend and never evicted.

Shared contracts were additive: `DeviceResidency`, resident `Storage`,
`Tensor::to_host`, seven defaulted backend hooks, `SessionOptions` and KV
resident helpers. `Backend::admits` is untouched; eight new `unsafe` blocks
each carry a local `SAFETY:` justification.

### 4.2 #71 — CPU inference and automatic thread scaling

Branch `fn3/71-cpu-perf`, HEAD `4b7be67`; merged as **`42d023a`**. Four
commits plus the final fix, 41 files, +9,238/−30.

**Algorithmic work.** Profiling put transpose at 65.7% of prefill and 94.9% of
decode. The retained changes are a cache-friendly last-two-axes transpose, a
blocked/AVX2 row-major MatMul behind the reference contract, and column-tiled
prefill. Measured against the serial base:

| Workload | Base p50 | After, 1 thread | Algorithmic |
|---|---:|---:|---:|
| tiny prefill 10 | 270.8 ms | 13.96 ms | 19.4× |
| tiny prefill 32 | 475.0 ms | 23.98 ms | 19.8× |
| tiny decode 10 | 188.7 ms | 6.15 ms | 30.7× |
| tiny decode 32 | 191.1 ms | 6.73 ms | 28.4× |

**The cache hypothesis was refuted, not assumed.** A six-arm experiment
cross-factored a 25.7 MB working set with 16-thread concurrency and idle
recovery. The transpose stayed within 0.19 ms across arms — smaller than the
within-arm p95 — while the same binary varied 5.277–7.886 ms between windows.
The earlier decode-after-prefill regression was host-state noise, not L3
eviction.

**Column tiling was retained on its own merit**: it reads the weight once
instead of once per output-row tile. Trained prefill 0.85×, tiny prefill 32
0.77×, tiny prefill 10 0.91×, at unchanged accumulation order; 15 shapes × 6
budgets remain bit-exact.

**Adaptive Auto.** Auto starts at the visible CPU count and narrows to one
thread once the runtime has seen a one-output-row product that cannot use
threads and none that can. `Fixed(n)` is never narrowed. On the trained model:

| Workload | 1 thread | Auto | Best fixed | Auto/best |
|---|---:|---:|---:|---:|
| prefill 10 | 330.44 ms | 239.33 ms | 237.08 (8t) | 1.01 |
| prefill 13 | 352.88 ms | 258.91 ms | 256.37 (12t) | 1.01 |
| decode after prefill 10 | 138.32 ms | 133.19 ms | 130.35 (4t) | 1.02 |
| decode chain | 142.50 ms | 143.76 ms | 139.09 (16t) | 1.034 |

The owner's reported ~1.2× reproduces as **1.38× for trained prefill**
(1→16 threads) and **1.04× for trained decode**. Tiny prefill is 1.31×; tiny
decode is flat because its one-token projections are below the 4 M-MAC
threshold. The single 152.82 ms chain sample was transient; a repeat measured
Auto at 1.034× best fixed, inside the stated 0.95–1.05 tolerance and the host's
3.4% cell spread.

Residual: the first turn of a session runs wide before narrowing. Removing it
needs a session-level turn hook in `src/fn3/model/**`; it is recorded, not
hidden.

### 4.3 #68 — quantization format

Branch `fn3/68-quant-format`, HEAD `ba7266e`; merged as **`24aee9e`**.

The existing grouped W4A8 was evaluated first, as required, against a
from-scratch Python unpacker: dequantized weights bit-exact, MatMul max 6.1e-5,
≤6 code differences of ±1. Every pure 4-bit candidate nonetheless flipped at
least one decode top-1, so W4A8 cannot be the starter format under the logit
contract.

Locked format **`fn3-quant-format-v1`** is **W8A16**: signed INT8 weights, one
byte per element, groups of 64 along K, binary16 scale and int16 zero-point per
(row, group), `w = (q − z)·s` in f32, FP32 activations, no nibble
interleaving. The normative kernel accumulates `Σ x·(q − z)` in f32 per group
and applies one scale multiply; there is no dequantized weight buffer and no
Q/DQ nodes. A 64-byte header, 64-byte descriptors, an 8-byte group directory
and a 16-byte-aligned payload carry dual CRC-32. The reader rejects
`qbits != 8`; 4-bit remains a future format version.

Fixtures: four valid blobs (1,072 B) with complete expected dequantized tensors
and hand arithmetic, 31 malformed blobs covering 27 reachable error kinds, an
independent Python encoder/decoder, and a Rust decoder that agrees byte-for-byte.

Accuracy gate for #69/#70: per-projection relative Frobenius ≤ 0.008,
prefill KL ≤ 1e-3, decode KL each step ≤ 0.02, centered max ≤ 0.8, centered
RMSE ≤ 0.25, top-1 100% with margin > 1.0.

Storage: projections 131,252,784 B vs 494,128,128 B FP32 (26.6%, 3.77×);
whole trained model 289,279,144 B (58.1%) with FP32 embeddings retained.

### 4.4 #69 — CPU quantized path

Branch `fn3/69-quant-cpu`, HEAD `c8f3b91`; merged **`67ca5ae`** after resolving
one module-declaration conflict in `cpu/mod.rs`.

`matmul_quantized` consumes the locked payload directly: groups ascending, k
ascending inside each group, one f32 accumulator per group, one f32 multiply by
the exact f16-widened scale. Up to four output rows share a pass over the
activation. Tails 1..64, `K < 64` and rank > 2 contiguous are supported; no
`unsafe`; geometry is checked before the first write.

Only admitted weights are quantized. Dynamic RHSs, strided activations and
unregistered slots fall back to FP32. `Backend::admits` is not overridden, and
`features.quantized` stays false.

Trained results: prefill 5,191.9 → 3,509.1 ms (**1.48×**), decode 1,401.1 →
1,269.5 ms (**1.10×**). Projection storage 339.7 → 90.2 MB (26.56%).
Accuracy: per-projection Frobenius max 6.418e-3 (bound 0.008), prefill KL
4.768e-6, decode KL 1.455e-4 / 3.262e-4, top-1 640/640, 11/11, 262/262.

Residual: the packed weights are additive in memory until a planner/format
change replaces the slots; `lm_head` stays FP32 because its RHS is a per-step
`Transpose(wte)`; the scalar kernel is ALU-bound, so the measured deltas are a
floor once #71's SIMD path is applied. Final accuracy sign-off is tied to the
trained FP32 decision.

### 4.5 #70 — Vulkan quantized path

Branch `fn3/70-quant-vulkan`, HEAD `d1d3f80`; merged **`aed9d95`**. 12 files,
+5,710; two additive lines in `kernels/mod.rs`.

The fused kernel computes
`y[m,n] = Σ_g (Σ_{k∈g} x[m,k]·(code[n,k] − zero[n,g])) · scale[n,g]`, one FP32
accumulator per 64-element group, one FP32 multiply by the binary16-widened
scale, no dequantized weight and no Q/DQ. Codes, scales and zeros are bound as
`uint` words (four codes or two metadata values per word), so no 8/16-bit
storage extension is needed; Vulkan 1.1 core only.

Tiny resident proof on the GTX 1080 Ti, validation 0/0: prepare packs eight
projections into **417,792 B**, uploaded once; 8 prefill / 24 decode quantized
records show **0 upload, 0 readback, 3 device-local weight bindings**, input
readbacks 0, readback exactly 804,112 B, submissions 6/30 identical to #72.
The launch takes `&PackedProjection` and has no host byte slice for a weight,
so per-op weight upload is structurally unrepresentable.

Tiny resident latency, 20 iterations, 3 warmups: FP32 prefill 9.251 / decode
9.430 ms; with eight quantized projections 11.019 / 9.517; with the head copy
11.875 / 9.258. **No tiny win is claimed**; prefill regresses 19–28% from the
row-per-invocation access at M=13.

Trained staged latency, 5 iterations: FP32 1,255.9 / 1,082.6 ms; with 48
projections 1,344.8 / 1,083.2; with 48 plus the head 1,245.1 / **980.9 ms
(−9.4%)**, upload 3,389,055,900 → 301,265,820 B. The head is the only per-step
weight-scale transfer (154 MB), and packing it removes exactly 617 MB per
iteration.

Kernel fidelity: ≤ 2.4e-7 max abs on tiny and ≤ 3.6e-7 relative on trained,
including `K = 8`, 65 and 70 tails; packer byte-identical to the Python
reference. Trained accuracy meets the #68 gates: worst projection 6.418e-3,
prefill KL 6.665e-6, decode KL 1.626e-3 / 1.281e-2 / 5.167e-3, centered max
0.774, centered RMSE 0.191, top-1 4/4.

**Trained resident is not claimed.** See §7.1. The trained arms are staged and
labelled; the test asserts its session is staged so the label cannot drift.

### 4.6 Artifact-only execution

Branch `fn3/artifact-exec`, HEAD `a74817e`; merged **`6275485`**. Six commits,
13 files, +1,079/−141.

`fn3-infer run` and `generate` accept `--model X.fnn3` alone. `--onnx` is now
optional source verification, and `--verify-source` makes that explicit;
`prepare` still requires ONNX. Python `load(artifact)` needs no source. The
planner is reached through additive `GraphIr::from_parts` /
`PreparedArtifact::into_model_import`.

Proven with the ONNX renamed away: plan `e6e4191a…`, top-5
`[46737, 28597, 16571, 7868, 13]`, identical to the two-file path; cached
decode plan ids also match. Error paths without ONNX: truncation, checksum and
future version exit 3 with named reasons; `--verify-source` without `--onnx`
exits 2; a disagreeing `--onnx` exits 3 naming the weight.

### 4.7 #73 — loader and parser hardening

Branch `fn3/73-hardening`, HEAD `06a4208`; merged **`653cdda`**. Two commits,
42 files, +2,458/−48.

Added an `import::budget` module (8 GiB model; 2^20 nodes/values/weights; rank
64; 1,024 parameter axes; 4,096-byte strings; 2^24 typed elements) and wired it
into the protobuf, graph and artifact readers. Five real defects were fixed:

- `Slice` panicked on a short `steps` list → `ShapeMismatch`.
- `tensor_bytes` allocated before validating length → `TensorDataKind`.
- A non-permutation `Transpose` reached the planner → `ShapeMismatch`.
- A NaN LayerNorm epsilon was silently skipped → `Architecture`.
- The artifact accepted `vocab_size = 0` → `Inconsistent`.

Evidence: 38 malformed cases mapped to typed errors, a 28-file byte corpus
(20,461 B), 44 ONNX prefixes and 132 flips refused, no single-byte artifact
flip yields a valid artifact, and three fuzz campaigns totalling **14,072,087
executions with zero crashes** under ASan. Allocation bounds measured: a 4 GiB
claim refused after **136 B**, 1M values after **96 B**, a 1,048,577-node input
peaking at 125,829,120 B (asserted < 512 MiB). Zero `unsafe` in the audited
import/format paths; the only new unsafe is a test-only counting `GlobalAlloc`
with a local `SAFETY:`. Miri: zero failures across five command sets.

Cross-cutting findings reported but not edited: the planner does not apply
`Shape::validate_permutation` to hand-built `GraphNode`s; `DecoderGeometry`
lacks a validating constructor; there is no streaming importer.

### 4.8 #73 — runtime hardening

Branch `fn3/73-runtime`, HEAD `81b7231`; merged **`29d9e69`**. Two commits,
17 files, +2,438/−21.

Four lifecycle defects were found: a partial command buffer at `begin_step`, a
partial plan still submitted at `end_step`, `in_flight` staying true after a
failed fence with a generic error, and half-appended KV layers when a mid-commit
device copy failed.

Failure contract:

- **Recoverable** failures leave extent, step, tokens, phase and plan cache
  unchanged, release slots, and roll back every KV layer to the pre-step
  extent. `Session::recover()` is the explicit boundary; a retry cannot
  double-append.
- **Terminal** device loss at submit, or any fence whose completion is unknown,
  poisons the device. Later calls return `Fn3Error::DeviceUnavailable` naming
  adapter, UUID and the failing call; there is no in-place recovery and the
  caller reopens.

Injected matrix, 4/4 debug and 4/4 release, validation 0/0: allocation before
submission; a 201,028 B logits readback; a recording failure at dispatch 41/144
(partial batch discarded, only the four pre-batch prefix copies submitted); the
third KV prefix copy; a KV append failure after layers 0 and 1 (extents rolled
back to `[18,18]`, retry reaches 19, semantic KV clean); `vkQueueSubmit` device
loss; fence device loss naming `vkWaitForFences`.

Repeated reset: 12 injected rounds, extents 13–18, failure and recovery in
round 3; every reset returns extents 0, KV 36,864 B, plans `(6,0)`, live
58,226,256 B stable. New hardware test `fn3_vulkan_reset`: 8 rounds, extents
14–17, fixtures each round, plans `(5,0)`, live 27,903,104 B, peak
116,794,416 B, validation 0/0.

Unavailable features: a synthetic adapter missing a compute queue, 256
invocations, a 64×16 workgroup, a 32 MiB range or a 64 MiB allocation produces
one `DeviceUnavailable` naming adapter, UUID, request and both numbers. A build
without the Vulkan feature lists only `[cpu]` and refuses an explicit Vulkan
request rather than falling back.

Physical evidence is kept separate from injection: `fn3_e2e` 10 (resident
default and staged, extents 14–17, byte-identical arms), resident 1, reset 1,
validation 0/0.

Four cross-cutting defects remain open and are listed in §7.3.

### 4.9 Alpha follow-up

Branch `fn3/alpha-followup`, HEAD `792d063`; merged **`3d2870c`**. Nine
commits, 15 files, +4,141/−41, no `src/fn3` change.

The handoff's debug/`--quick` timing table was replaced with a matched release
build (opt-level 3, LTO, one codegen unit), one concrete backend per process,
3 warmups and 20 measured iterations:

| Metric | CPU | GTX 1080 Ti |
|---|---:|---:|
| Prefill TTFT p50 | 240.31 ms | 204.76 ms |
| Prefill TTFT p95 | 246.02 ms | 245.12 ms |
| Decode p50/token | 158.33 ms | 190.64 ms |
| Decode p95/token | 179.98 ms | 228.22 ms |
| Upload / readback / submissions | unavailable | 318,880,260 B / 106,941,352 B / 575 |
| Executions proven | unavailable | 11,500/11,500 |
| Peak host RSS | 88.76 MiB | 396.72 MiB |
| Validation | n/a | 0/0 |

Device timestamps are not yet stable because the GPU clock is unpinned (139 MHz
idle / 1,911 MHz max); host wall clock is the comparable metric. The missing
device-local peak accessor is named `MemoryStats::peak_live_device_local_bytes`.

Strict wheel inference was added: model present → lane passes with top-5
`[46737, 28597, 16571, 7868, 13]`; model absent in strict mode → named failure;
non-strict → exit 0 with `not_run`. The release workflow derives the fetch URL
from the lock, verifies size and sha256, and fails with `::error::pinned-model`
if it cannot fetch.

Metadata: description, MIT read from the committed `LICENSE`, repository,
readme, keywords, categories, `rust-version = "1.89"`, full PEP 621 metadata.
Crate package went from 672 files / 3.1 MiB to **306 files / 1,021.3 KiB**,
with no fixtures, models, tests or `target/`. Wheel: 47 files, license included.

Hosted CI found a real defect: run `44486950429` at `672cec9` failed the
`vulkan-software` lane because the hosted image ships lavapipe but not
`VK_LAYER_KHRONOS_validation`. Installing the layer fixed it without weakening
the 0/0 requirement. Green runs: `44488019087`, `44488737026`, `44489825235`,
and `44491089648` on the integration HEAD `3d2870c`.

### 4.10 Trained GPT-2 FP32 investigation

Branch `fn3/trained-fp32`, HEAD `d4f14a2`; merged **`6e1d564`**. Diagnostics only:
no `src/fn3/backend/**`, `model/**`, `cli/**`, `python/**` or `fixtures/**` edit.

Reproduction on CPU release against the committed reference:

| Phase | max abs | mean abs | 5e-5 bound | Elementwise | Top-1 | Top-5 | KL |
|---|---:|---:|---|---|---|---|---:|
| prefill p13 | 1.640e-4 | **7.273e-5** | FAIL | 0 violations | 640/640 | 5/5 | 3.5e-13 |
| decode 1 | 9.537e-5 | 1.743e-5 | pass | 0 | 11/11 | 5/5 | 5.7e-11 |
| decode 2 | 2.975e-4 | **9.302e-5** | FAIL | 0 | 262/262 | 5/5 | 1.2e-9 |

Tiny control remains 24–25× inside its 2.5e-6 mean bound.

Isolation found **no numerical defect**:

- Per-layer relative RMS vs ONNX Runtime stays ~1e-6 from layer 0 to 11;
  mean absolute error grows smoothly from 1.05e-6 to 9.61e-6 with no stage jump.
- Three real projections recomputed in the documented sequential-f32,
  ascending-k, non-FMA order are **79,872/79,872 elements bit-identical**.
- LayerNorm epsilon, attention scale `0.35355338`, `gelu_new` constants, softmax
  and layouts all match the graph.
- The error is a near-constant translation of the 50,257-vector. Projecting the
  final hidden delta through the tied embedding in float64 predicts the
  observed mean error: 1.8508e-5 vs 1.8529e-5 (prefill), −2.696e-6 vs −2.686e-6
  (decode 1), 9.273e-5 vs 9.228e-5 (decode 2). The sign flips by phase and no
  ranking changes.
- **Decisive:** an independent float64 evaluation of the same graph differs from
  ONNX Runtime by **6.254e-5 mean abs** at the same position. The 5e-5 bound is
  below the reference implementation's own rounding error; its centered RMS is
  5.3e-6.

Proposed tolerance, **not applied** and not written into any fixture:

```
mean_abs <= 5e-5 + 5e-6 * mean(|ref|)
rms(delta - mean(delta)) <= 5e-5 * rms(ref - mean(ref))
```

Observed values sit at 6.63e-6–1.338e-5 of the proposed bound. The elementwise
bound, top-k, KL and cache rules are unchanged. The trained bound is 13.7×
tighter relative to output scale than the tiny bound (1.01e-6 vs 1.39e-5), and
the logits carry a −49…−108 common mode that softmax removes. The old bound
remains visible in the report as failing.

### 4.11 Resident trained fix — in progress

A worktree `fastnn-wt-resident-trained`, branch `fn3/resident-trained`, base
`6e1d564`, is fixing the defect in §7.1. Nothing from it is merged yet.

---

## 5. Integration ledger

| Commit | Content |
|---|---|
| `672cec9` | alpha candidate, pushed; alpha review commit |
| `3d2870c` | merge alpha follow-up: matched baseline, strict wheel, hosted CI fix |
| `6275485` | merge artifact-only execution |
| `653cdda` | merge #73 loader/parser hardening |
| `24aee9e` | merge #68 W8A16 format decision |
| `77c76ae` | ledger wave 1 |
| `dd1dd7f` | merge #72 residency, KV and batching |
| `6389186` | ledger #72 |
| `42d023a` | merge #71 CPU kernels and adaptive Auto |
| `67ca5ae` | merge #69 CPU W8A16 path |
| `29d9e69` | merge #73 runtime hardening |
| `6e1d564` | merge trained FP32 diagnostics |
| `aed9d95` | merge #70 Vulkan W8A16 path |
| `a3e9723` | ledger wave 2; current HEAD |

All merges are `--no-ff`; `672cec9` remains an ancestor. The branch is pushed
without force and `origin/rewrite/fastnn-3` matches `a3e9723`.

---

## 6. Verification reproduced by the integrator

| Gate | Result |
|---|---|
| `cargo fmt --check` | clean at every merged HEAD |
| `cargo test --lib` | **699** default; **759** `--features vulkan`; **750** `--features vulkan,vulkan-fault-injection` |
| `fn3_contracts` | 44 |
| `fn3_tensor_storage` | 36 |
| `fn3_cpu_reference` | 43 |
| `fn3_import` | 42 passed, 1 ignored |
| `fn3_model_format` | 47 |
| `fn3_planner` | 11 |
| `fn3_session` | 8 |
| `fn3_cli` | 8 |
| `fn3_malformed` | 8 |
| `fn3_quant_format` | 4 passed, 2 ignored |
| `fn3_quant_cpu` | 7 passed, 4 ignored (debug); 11 passed with the trained gates enabled in release |
| `fn3_e2e` | 5 CPU, 10 Vulkan |
| `fn3_vulkan_resident` | 1 |
| `fn3_vulkan_faults` | 4 debug, 4 release |
| `fn3_vulkan_reset` | 1 on the physical GTX 1080 Ti |
| `fn3_vulkan_quant` | 11 |
| Vulkan smoke / matmul / pointwise / attention | 14 / 17 / 18 / 19 |
| `fn3_cpu_perf` | 11 |
| Clippy `-D warnings` | clean in default, `--no-default-features`, `vulkan`, `cli`, `fn3`, `fn3,vulkan`, `vulkan+fault` |
| Versions | `3.0.0-alpha.1` / `3.0.0a1`, seven stores, enforced by `scripts/ci/check-versions.py` |
| Hosted CI | run `44491089648` green on `3d2870c`; **not yet run on `a3e9723`** |
| Physical GPU | GTX 1080 Ti UUID `471b2bda-51e1-6451-e692-ff5d1e146f1e`, validation 0 errors / 0 warnings on every run above |

The counts are the ones from the owning branch and the post-merge integration
checkout. They are not a claim that every binary was re-run after every merge;
the merged-tree lib counts and the directly affected binaries were.

---

## 7. Open defects and exceptions

### 7.1 Blocking: trained resident `Reshape → Gather`

Resident Vulkan fails on the trained model at
`src/fn3/backend/vulkan/kernels/embedding.rs:132` → `read_i64_values` at
`transform.rs:239-247` → `view.as_i64_slice()` on a device-resident tensor →
`Fn3Error::NotImplemented` from `tensor.rs:1209`.

Trained node 12 `Reshape(input_ids,[1,13])` (plan instruction 2, slot 382)
feeds `Gather(wte)`. The resident session places the identity Reshape output on
device, but Gather validates its indices on the host. Tiny is unaffected because
Gather reads the host `input_ids` directly.

Proposed fix: fold identity `Reshape` at plan time
(`src/fn3/model/specialize.rs:1455-1538`) so Gather reads the host slot. A
kernel-side `to_host()` is insufficient inside a batched step. Fallback:
host-allocating, host-validated index chains with batching disabled for that
step only. Workaround today: `FN3_VK_STAGED=1`.

**Blocks:** #70 trained-resident claim, #74 trained Vulkan measurement, #76
stable gate. A fix is in progress.

### 7.2 Awaiting owner decision: trained FP32 mean absolute error

The evidence is in §4.10. The proposed scale-relative bound is not applied and
no fixture was edited. Until the owner decides, the stable gate cannot claim
the trained FP32 comparison as passing under the current 5e-5 rule.

### 7.3 Recorded, non-blocking

- **Vulkan OOM reports `Allocation { bytes: 0 }`.** The allocation size is lost
  on one path; the error should name the requested bytes.
- **`prepare_persistent` is not transactional.** A failure mid-prepare can leave
  a partially populated persistent table.
- **`end_step` returns `Ok(0)` for a discarded partial batch.** The count is
  correct; the success code does not distinguish a discarded batch from an
  empty one.
- **`Submission::Drop` retries a failed wait on a poisoned device.** Harmless
  today because the device is terminal, but it is the wrong shape.
- **First turn of a CPU session runs wide before Auto narrows.** Needs a
  session-level turn hook; one sample in 30/4.
- **Decode is still 10 `vkQueueSubmit`.** Four KV appends and the logits
  download sit outside the plan batch.
- **Device peak rises 62 → 116 MB with batching.** Staging blocks are held to
  the fence; an arena is the follow-up.
- **`lm_head` stays FP32.** Its RHS is a per-step `Transpose(wte)`; a packed
  copy would cost 41,009,712 B and halve decode streaming.
- **Quantized CPU weights are additive in memory** until a planner/format
  change replaces the FP32 slots.
- **No tiny quantized speed win is claimed.** The tiny step is dispatch-bound;
  prefill regresses 19–28%.
- **GPU clock is unpinned.** Device timestamps span 12.93–28.55 ms; host wall
  is the comparable metric. Pinning needs root (`nvidia-smi -lgc`).
- **`MemoryStats::peak_live_device_local_bytes` is not wired into the bench
  adapter**, so the matched baseline reports device peak as the allocator
  high-water only.
- **macOS and Windows strict inference is unproven.** Only import/packaging was
  tested there.
- **`perf` is unavailable** (`perf_event_paranoid=2`), so profiles are internal
  timers and standalone studies, not hardware counters.
- **No arm64 claim.** No target, qemu or cross-gcc.
- **Only one physical GPU.** Cross-vendor Vulkan claims are impossible on this
  host; #74 must record that rather than imply coverage.

---

## 8. Issue state

Closed with evidence comments: **#46–#72** except none, plus #73.
Specifically, the stable-wave issues closed are #68, #69, #71, #72 and #73.
**#70 is open** pending the trained resident fix and the accuracy sign-off.

Open: **#70, #74, #75, #76**; deferred and untouched: **#77 (CUDA)** and
**#78 (training)**.

---

## 9. Stable-gate checklist (#76)

> **Naming recommendation superseded:** the canonical names are now
> `fastnn::inference`, `fastnn.inference`, and `fastnn-infer`. The checklist
> below remains a snapshot of the original report.

| Item | State |
|---|---|
| Exact final candidate SHA | pending; current HEAD `a3e9723` is not the stable candidate |
| Hosted CI on that exact SHA | pending; green run `44491089648` predates #72 and is workflow evidence only |
| Pinned-model CPU/Vulkan checks with missing fixtures/devices as failures | pending in the final lane configuration |
| Trained FP32 mean-abs resolved or owner-approved | **pending decision** |
| CPU/Vulkan quantized inference on the locked format | CPU done and merged; Vulkan tiny resident done; trained resident blocked by §7.1 |
| Stable public names, wheel/device coverage, supported hardware | pending decision; recommendation: keep `fastnn.fn3` / `fn3-infer` for 3.0.0 and record `fastnn::inference` / `fastnn.inference` / `fastnn-infer` as a separate post-3.0 breaking cleanup. The artifact format is independent of either name |
| Artifact-only prepare/run from a clean checkout | proven on `6275485`; must be replayed on the final candidate |
| Wheel install smoke from outside the tree | proven in the alpha follow-up; must be re-run on the final candidate |
| Version metadata aligned to 3.0.0 | **not done**; currently `3.0.0-alpha.1` / `3.0.0a1` |
| Release notes drafted | pending |
| Physical-GPU validation on the final SHA | pending |
| Tag, release, publication, merge to main | owner actions; none taken |

---

## 10. Evidence index

| Area | Path |
|---|---|
| Alpha handoff | `docs/reports/fn3-21-alpha-release.md` |
| Alpha follow-up baseline + strict wheel | `docs/reports/fn3-21-alpha-followup/`, `docs/internals/release-process.md` |
| Vulkan residency | `docs/reports/fn3-26-vulkan-residency.md`, `docs/reports/fn3-26/` |
| CPU profile and thread sweep | `docs/reports/fn3-25-cpu-performance.md`, `docs/reports/fn3-25/` |
| Quant format | `docs/reports/fn3-22-quant-format.md`, `docs/quantization/fn3-quant-format-v1.md`, `fixtures/quant/` |
| CPU quantized path | `docs/reports/fn3-23-quant-cpu.md` |
| Vulkan quantized path | `docs/reports/fn3-30-vulkan-quant.md`, `docs/reports/fn3-30/` |
| Loader hardening | `docs/reports/fn3-27-hardening.md`, `tests/corpus/fn3/` |
| Runtime hardening | `docs/reports/fn3-27-runtime-hardening.md` |
| Trained FP32 investigation | `docs/reports/fn3-trained-fp32.md`, `docs/reports/fn3-trained-fp32/` |
| Artifact-only execution | `docs/reports/fn3-artifact-exec.md` |
| Integration ledger | `docs/roadmap/fn3-ledger.md` |

---

## 11. What is deliberately not done

- No merge to `main`, no tag, no GitHub release, no PyPI or crates.io
  publication, no PR to `main`.
- No CUDA work (#77) and no training work (#78).
- No license change: MIT is read from the committed `LICENSE`; the owner should
  confirm it before publication.
- No tolerance change in any fixture: the old bound is still the bound.
- No claim that the trained resident path works until §7.1 is fixed and
  re-verified.
