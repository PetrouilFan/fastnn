# FN3-23 (#69): production CPU inference on the locked W8A16 format

**Issue:** [#69](https://github.com/PetrouilFan/fastnn/issues/69), backlog FN3-23,
workstream CPU quantization, milestone stable.
**Depends on:** #53 (CPU reference), #58 (session), #68 (format); #71
(performance) is concurrent and not a dependency of this layout.
**Branch:** `fn3/69-quant-cpu`, base `6389186d4599471c4cb3ebc1156a5c1b872cd423`.
**Format:** `fn3-quant-format-v1` — signed INT8 weights, one byte per element,
groups of 64 along K, binary16 scale + int16 zero-point per `(row, group)`,
`w = (q − z)·s`, FP32 activations, no nibble interleaving, no Q/DQ nodes.

## Decision

The CPU consumes the packed INT8 codes, the little-endian binary16 scales and
the int16 zero-points **directly** in the projection kernel. Nothing is
dequantized into a weight buffer. The kernel computes `Σ x·(q − z)` in f32 for
each 64-element group in ascending `k`, then applies the group's scale exactly
once. Tail groups (`1..=64` columns) and `K < 64` are supported.

The work is additive under `src/fn3/backend/cpu/quant/`; it adds **no** `OpId`,
no `DType`, no capability flag and no `Backend::admits` override. A
`QuantCpuBackend` implements the existing `Backend` trait, so `Session` runs the
plan unchanged and only the registered projections take the fused path. Every
other `MatMul` — a dynamic right-hand side, the lm_head transpose, a strided
activation, or a weight the policy leaves in FP32 — is delegated to the
reference kernel and counted as an explicit fallback.

## 1. Files

| Path | Lines | Purpose |
| --- | ---: | --- |
| `src/fn3/backend/cpu/quant/mod.rs` | 72 | module contract, re-exports |
| `src/fn3/backend/cpu/quant/kernel.rs` | 697 | fused kernel, `PackedView`, `WorkHint` |
| `src/fn3/backend/cpu/quant/pack.rs` | 409 | §4.5 encoder, owned packed weight, error report |
| `src/fn3/backend/cpu/quant/planner.rs` | 1074 | per-projection registration, §6 policy, coverage, weight set |
| `src/fn3/backend/cpu/quant/backend.rs` | 345 | `QuantCpuBackend`, counters |
| `src/fn3/backend/cpu/mod.rs` | +1 | `pub mod quant;` only |
| `tests/fn3_quant_cpu.rs` | 1712 | parity, policy, fallback, determinism, model evidence |
| `docs/reports/fn3-23-quant-cpu.md` | 404 | this report |

No other shared file is touched: `parallel.rs`, `gemm.rs`, `matmul.rs`,
`simd.rs`, `storage.rs`, `tensor.rs`, `backend/mod.rs`, `session.rs`, the CLI,
the Python surface, the importer and the prepared-model format are unchanged.

## 2. Kernel design and accumulation order

For one packed weight `W[N, K]` and one contiguous FP32 activation row `x[K]`:

```text
y[row] = 0
for g in 0..ceil(K/64):                     # ascending group
    (start, end) = (64g, min(64g + 64, K))
    acc = 0
    for k in start..end:                       # ascending k
        acc += x[k] * (code[row, k] - zero_point[row, g])
    y[row] += acc * scale[row, g]              # exactly one f32 multiply
```

* `code` is read as a two's-complement `i8` from the packed byte; `(q − z)` is
  an exact small integer in f32.
* The group sum is one f32 accumulator; there is no pairwise sum, no f64
  accumulator and no fused multiply-add.
* The scale is the binary16 bit pattern widened to f32 exactly, read once per
  group.
* Up to four output rows are computed in one pass over `x` (`ROW_BLOCK = 4`) so
  one activation load feeds several rows. Each output element keeps its own
  accumulator, so this changes which rows share the load, not any sum.
* Tails: the final group is `1..=64` columns; `K < 64` is one group. `K = 70`
  is `(0, 64)` + `(64, 6)`.
* Rank > 2 is supported when the activation is contiguous: the batch axes are
  iterated around the same `[M, K]` inner kernel, and a `[2, 3, 130] × [130, 16]`
  test compares every element with the reference matmul on the dequantized
  weight.

The kernel validates dtype, device, rank, contraction, output geometry and
contiguity before the first output write. It contains no `unsafe`; row slices
are formed once and indexed by a `k` that is bounded by the slice length.

## 3. Packing (the normative encoder)

`PackedWeight::from_f32_rhs` consumes the ONNX `MatMul` right-hand side exactly
as the graph carries it, `[K, N]`, and writes the format's `[N, K]` grouped
layout in one pass (no transposed FP32 copy). It reproduces
`tools/quant/fn3_quant.py::quantize_int8_grouped`:

* per `(row, group)` min/max over the 64 columns;
* `span = max − min`, `asym_scale = span/255`, `asym_zero = −128 −
  round_half_away(min/asym_scale)`; asymmetric when `−128 ≤ asym_zero ≤ 127`;
* constant group: `scale = 1.0` for an all-zero group, otherwise
  `max(|v|/127, 1e-6)`, zero-point 0;
* otherwise symmetric: `max(|min|, |max|)/127`, zero-point 0;
* the scale is rounded to binary16 **first**; codes are
  `clamp(round_half_away(v / scale_f16) + zero, −128, 127)` with round-half-away
  evaluated in f64 on the f32 ratio, exactly the Python rule.

The packer rejects a non-finite source value and a scale that rounds to zero or
infinity. It reports per-weight `||W − Ŵ||_F / ||W||_F`, the largest absolute
error, code/metadata bytes and the asymmetric/constant group counts.

Byte layout in memory: `N·K` codes, then `N·G` little-endian binary16 scales,
then `N·G` little-endian int16 zero-points. That is the container payload
without its per-row 16-byte alignment padding; `format_bytes()` reports the
padded size and the report keeps both numbers.

## 4. Registration, fallback and coverage

`ProjectionSpec` describes one `MatMul` right-hand side.
`QuantPolicy::classify` applies the format's §6 rules and returns either
`Quantize` or `Fallback(reason)`. `registration(spec, activation, policy)` maps
one concrete invocation onto the backend contract:

| Registration | `Support` | Behaviour |
| --- | --- | --- |
| `Admitted` | `Full` | kernel runs |
| `Partial { reason }` | `Partial { reason }` | kernel rejects with `UnsupportedOp`; backend delegates and counts |
| `Fallback { reason }` | `Unsupported { reason }` | not registered; FP32 path |

`Full` is reported only for the kernel's documented subset: a registered
static 2-D FP32 weight with `K ≥ 64`, at least 4096 elements, a permitted packed
copy, a contiguous FP32 activation of rank ≥ 2 and matching contraction/output
geometry. The only `Partial` case is a non-contiguous activation; the kernel
really rejects it and the backend really delegates.

`QuantWeightSet::from_plan` walks a concrete `Plan`, registers the right-hand
side of every `MatMul` whose slot role is `SlotRole::Weight`, and keys each
packed weight by the address of the storage the session later presents. The
token embedding `wte` is never packed (no MatMul reads it as a weight slot), and
the lm_head `MatMul` reads `transpose(wte)` — an instruction output — so it is
`dynamic-rhs` and stays FP32.

Fallback reasons, all stable strings used by the counters:

`dynamic-rhs`, `rhs-rank`, `weight-dtype`, `shared-weight`, `k-below-min`,
`numel-below-min`, `activation-dtype`, `activation-rank`,
`contraction-mismatch`, plus `unpacked-rhs` (the backend's lookup miss) and
`activation-layout` (a rejected strided invocation).

`Capabilities::features.quantized` deliberately stays `false`. There is no
quantized `DType` and no quantized `OpId`; the flag promises "quantized
dtypes/ops are implemented", and flipping it would advertise a contract this
weight representation does not add. The registration and its counters are the
honest, per-projection evidence instead.

## 5. Format parity

Measured by `cargo test --test fn3_quant_cpu`:

| Check | Result |
| --- | --- |
| `v1/basic.fn3q` codes vs `manifest.packed_hex` | byte-exact |
| `v1/tail_k70.fn3q` codes, 10 padding bytes, `(0,64)`+`(64,6)` | byte-exact |
| `v1/zero_point.fn3q` and `basic` row 1 non-zero zero-points | bit-exact |
| `v1/two_tensors.fn3q` including `K = 8` | byte-exact |
| every scale / zero-point vs the manifest | bit-exact |
| fused kernel vs `f64` re-derivation of the manifest's dequantized values | within `2e-6·(1+|ref|)` |
| fused kernel vs `QuantTensor::dequantize_into` + one f32 accumulator | within `1e-5·(1+|ref|)` |
| packer vs `tools/quant/fn3_quant.py` (`K=70`, `N=4`) | codes CRC `0x09f79257`, scales CRC `0xabed3e89`, zeros CRC `0xf83264c7`, all equal |
| packer group classes on that matrix | 1 asymmetric, 4 constant, 3 symmetric; first zero-point −59 |

The packer matrix deliberately contains a constant non-zero group (`0.25`), a
constant zero group, an all-positive group whose asymmetric zero-point would be
out of range (so it takes the symmetric branch) and a group with a non-zero
zero-point. The CRC comparison is over the raw `N·K` codes, `N·G` little-endian
binary16 scales and `N·G` little-endian int16 zero-points, so it covers the byte
layout and not only dequantized values. At full scale the packer also reproduces
the #68 record's weight error (mean 5.539e-3 vs 0.0055, max 6.418e-3 vs 0.0064
on the trained projections), which is an independent cross-check of the
encoder policy on 48 real matrices.

The committed malformed matrix is rejected before any arithmetic. The new suite
asserts `unsupported-qbits`, `zero-point-out-of-range`, `bad-scale` and
`trailing-or-truncated-bytes` by `kind()`, and separately that a `PackedView`
with truncated codes, a row stride shorter than K, or missing metadata is
`InvalidShape`, that a K mismatch is a `ShapeMismatch` with the output untouched,
and that `qbits != 8` can never reach the kernel.

## 6. Projection coverage

Measured from the real `Plan` of each model:

| Graph | Quantized | Fallback | Fallback reasons |
| --- | ---: | ---: | --- |
| tiny GPT-2 (2 layers) | 8 | 5 | `dynamic-rhs` ×5 (4 attention + lm_head) |
| trained GPT-2 (12 layers) | 48 | 25 | `dynamic-rhs` ×25 (24 attention + lm_head) |

Trained family table (from `CoverageReport`):

| Family | Quantized | Fallback |
| --- | ---: | ---: |
| `c_attn` (QKV) | 12 | 0 |
| `attn.c_proj` | 12 | 0 |
| `mlp.c_fc` | 12 | 0 |
| `mlp.c_proj` | 12 | 0 |
| lm_head | 0 | 1 |
| other (attention score/value) | 0 | 24 |

The lm_head `MatMul` reads `transpose(wte)`, an instruction output produced in
the step; it is not a static weight slot, so the format's "shared weight without
a permitted packed copy" story never arises in this graph. The FP32 `wte` table
stays for the embedding `Gather`.

Storage (measured from the packed bytes and the model's tensors):

| Quantity | Tiny | Trained |
| --- | ---: | ---: |
| FP32 projection weights | 1,572,864 B | 339,738,624 B |
| W8A16 packed projections | 417,792 B (26.56%) | 90,243,072 B (26.56%) |
| of which scale+zero-point metadata | 24,576 B (5.88%) | 5,308,416 B (5.88%) |
| saved | 1,155,072 B | 249,495,552 B |
| per-projection relative Frobenius, mean | 5.360e-3 | 5.539e-3 |
| per-projection relative Frobenius, max | 5.393e-3 | 6.418e-3 (bound 0.008) |
| whole model, FP32 weights | 27,833,180 B | 497,764,984 B |
| whole model, projections replaced | 26,678,108 B | 248,269,432 B (49.88%) |
| lm_head kept FP32 | 25,731,584 B | 154,389,504 B |
| lm_head packed copy (not created) | 6,834,952 B | 41,009,712 B |

The 248,269,432-byte trained figure matches the #68 decision record's
"whole model, lm_head left FP32" row exactly (`49.9%`). **This backend is
additive**: it keeps the FP32 model weights because the session's weight slots
still reference them, and adds the packed copies. Its resident bytes are
497,764,984 + 90,243,072 = 588,008,056 B, not 248 MB. Replacing the FP32
projection slots with a packed artifact is a planner/prepared-format change
outside this issue's ownership boundary.

## 7. Measurements

### Accuracy (measured; not the final sign-off)

Reference: the fn3 FP32 run of the same graph, prompt and decode tokens
(prefill 13 tokens, then tokens 640 and 11). KL is `KL(reference ‖ W8A16)` in
nats; `centered_*` subtract the mean logit delta. The #68 bounds are:

| Bound | Value | Prefill | Decode 1 | Decode 2 |
| --- | ---: | ---: | ---: | ---: |
| KL | `1e-3` prefill, `0.02` each decode | **4.768e-6** | **1.455e-4** | **3.262e-4** |
| centered max | 0.8 | **0.2104** | **0.1577** | **0.1448** |
| centered RMSE | 0.25 | **0.04905** | **0.03018** | **0.02868** |
| elementwise max | 1.5 | **1.209** | **0.6631** | **0.8217** |
| top-1 | equal when reference margin > 1.0 | **640 / 640** (margin 7.561) | **11 / 11** (2.222) | **262 / 262** (1.029) |
| top-5 | ≥ 4/5 when 5th gap > 2×centered max | **5/5** (gap 1.238) | **4/5** (gap 0.001) | **4/5** (gap 0.010) |
| mean_abs | reported, no bound | 0.9986 | 0.5054 | 0.6768 |

Tiny GPT-2 (random init; the #68 bounds were calibrated on the trained graph,
so these are reported, not gated):

| Position | max_abs | centered max | centered RMSE | KL | top-1 | top-5 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| prefill | 5.467e-3 | 5.461e-3 | 1.296e-3 | 8.393e-7 | 257/257 (margin 0.366) | 5/5 |
| decode 1 | 7.381e-3 | 7.376e-3 | 1.684e-3 | 1.420e-6 | 24671/24671 (0.063) | 5/5 |
| decode 2 | 7.254e-3 | 7.254e-3 | 1.558e-3 | 1.212e-6 | 11/11 (0.097) | 5/5 |

Every trained bound passes with zero violations. The measured per-projection
maximum (6.418e-3) is the same quantity #68 reported as 0.0064. **No final
accuracy sign-off is taken here**: the reference is the fn3 FP32 graph, and the
trained FP32 `mean_abs` delta documented in the format report (§2.3) is still
open. The quantized run adds its own error on top of that reference; it does not
resolve it.

### Latency and peak memory (locked runs)

Release build, end-to-end `Session` time (all instructions; the quantized
projections are 48 of 73 MatMuls on trained and 8 of 13 on tiny). Four separate
processes, each on the shared lock:

| Model | Phase | FP32 p50 / p95 | W8A16 p50 / p95 | W8A16 vs FP32 |
| --- | --- | ---: | ---: | ---: |
| tiny | prefill, 13 tokens | 300.783 / 305.839 ms | 300.546 / 301.616 ms | 1.00× |
| tiny | decode, batch 1 | 188.859 / 196.945 ms | 189.485 / 195.875 ms | 0.997× |
| trained | prefill, 13 tokens | 5191.922 / 5197.385 ms | 3509.078 / 3511.483 ms | **1.48×** |
| trained | decode, batch 1 | 1401.133 / 1457.739 ms | 1269.469 / 1326.766 ms | **1.10×** |

Tiny is dominated by the FP32 lm_head (6.4M MACs per token versus 0.39M for all
eight projections) and by the attention/transpose traffic, so quantizing the
projections cannot move it: the two configurations are within 0.3% and the
difference is inside the run-to-run spread. Trained has a 48-projection,
84.9M-MAC decode and 13× that in prefill; its prefill is 1.48× faster and its
decode 1.10×. These are scalar-kernel numbers; #71's blocked/SIMD kernels are
expected to move the decode ratio, and the format's 3.77× weight-bandwidth
reduction is not yet the limiter because the scalar loop is ALU-bound.

Peak memory (`VmHWM` of a process that ran only that configuration):

| Model | FP32 peak | W8A16 peak | Delta |
| --- | ---: | ---: | ---: |
| tiny | 92,432 kB | 92,892 kB | +460 kB (+0.5%) |
| trained | 1,247,060 kB | 1,335,312 kB | +88,252 kB (+7.1%) |

The delta is the packed set itself (90.2 MB trained, 0.42 MB tiny): this backend
is additive and keeps the FP32 weights, so the packed copies increase the peak
until the prepared artifact replaces the FP32 projection slots (§6).

Kernel time and fallback overhead from the same runs:

| Model | Fused kernel mean per projection (prefill + decode) | Registry/dispatch overhead |
| --- | ---: | --- |
| tiny | 0.064 – 0.261 ms | unregistered MatMul 5,556 vs 5,533 ns (inner); fused 28,241 vs 28,290 ns kernel |
| trained | 2.334 – 9.602 ms | unregistered MatMul 5,920 vs 5,938 ns (inner); fused 1,064,435 vs 1,066,573 ns kernel |

The signed differences for the fused dispatch are negative, i.e. smaller than
the timing noise of a ~1 ms kernel; the lookup, registration check and atomic
counters are below 0.3% of a projection. The unregistered-MatMul delegation
costs at most 23 ns on a 5.5 µs call, so a fallback costs what the FP32 kernel
costs, not a round trip.

Counters: trained `96 quantized (48/step), 50 fallback (25/step)`; tiny
`16 quantized (8/step), 10 fallback (5/step)`, all `unpacked-rhs` (the lm_head
transpose and the attention products). Every registered weight was used.

Observed host state: lock acquired at `2026-11-19T13:37:20+02:00` with
`loadavg 2.75 3.09 2.96`; the four runs saw 1.71–2.75; no `cargo`/`rustc`
process was running during the timed sections (the script checks before each
run and records the load before/after). The machine is shared by up to five
worker sessions, so treat the third significant figure as noise.

## 8. Thread-budget seam for #71

`matmul_quantized` takes an explicit `WorkHint { threads, min_rows }` and visits
output rows through `hint.chunks(n_rows)`. It never spawns a thread and owns no
pool. The chunks are contiguous, disjoint, ascending, cover every row once and
number at most `threads`; a weight with fewer rows than `min_rows` is one chunk.
Because each output element keeps its own ascending-`k`/ascending-group
accumulation, distributing those chunks over a pool is bit-identical. The test
runs `threads = 1, 2, 3, 8` with `min_rows = 1, 3, 7` and compares every bit,
and repeats the serial run.

When #71 lands, its `ThreadBudget` supplies `threads` and a minimum work unit
(`min_rows`) to `WorkHint::new`, and replaces the sequential `for chunk` loop in
`kernel::project_matrix` with a pool dispatch over the same `RowChunks`
iterator. Nothing else in this module changes. `parallel.rs` does not exist on
this branch and is not edited; the kernel's public signature already carries the
hint, so the swap is internal to `project_matrix`.

## 9. Commands and outcomes

All builds use `CARGO_BUILD_JOBS=3`. Model paths were verified against
`fixtures/parity/model.lock.json`:

```text
tiny    27,897,974 B  e18370ba…141a1  (sha256sum, matches the lock)
trained 498,126,358 B 42c1e92a…c7c0d  (sha256sum, matches the lock)
```

| Command | Outcome |
| --- | --- |
| `git rev-parse HEAD` at start | `6389186d4599471c4cb3ebc1156a5c1b872cd423` (base) |
| `cargo fmt --check` | clean |
| `cargo test --lib` | **689 passed, 0 failed** |
| `cargo test --test fn3_quant_format` | **4 passed, 2 ignored** (the two opt-in trained/W4 evidence tests) |
| `cargo test --test fn3_cpu_reference` | **43 passed, 0 failed** |
| `FASTNN_FN3_ONNX_MODEL=<tiny> cargo test --test fn3_session` | **8 passed, 0 failed** |
| `FASTNN_FN3_ONNX_MODEL=<tiny> cargo test --test fn3_e2e -- --test-threads=1` | **5 passed, 0 failed** |
| `cargo test --test fn3_quant_cpu` | **7 passed, 4 ignored** (the four pinned-model tests) |
| `cargo clippy --tests -- -D warnings` | clean |
| `cargo clippy --tests --features fn3 -- -D warnings` | clean |
| `fn3_quant_cpu tiny_gpt2_w8a16_matches_fp32 -- --ignored` | pass, 1.52 s |
| `fn3_quant_cpu trained_gpt2_w8a16_matches_fp32 -- --ignored` | pass, 17.51 s |

The four timed configurations were run as one serialized script inside the
shared lock, each in its own process and after a `pgrep -a 'cargo|rustc'`
check (all four printed `cargo/rustc: none`):

```bash
flock /tmp/opencode/fn3-measure.lock /tmp/opencode/fn3-69/measure.sh
#   FN3_QUANT_MODE=fp32  fn3_quant_cpu measure_trained --ignored --nocapture
#   FN3_QUANT_MODE=w8a16 fn3_quant_cpu measure_trained --ignored --nocapture
#   FN3_QUANT_MODE=fp32  fn3_quant_cpu measure_tiny    --ignored --nocapture
#   FN3_QUANT_MODE=w8a16 fn3_quant_cpu measure_tiny    --ignored --nocapture
```

Outcome: 4/4 `exit=0`; the lock was acquired at `13:37:20+02:00` after the #71
authoritative window released it. Raw numbers are in §7.

The accuracy runs are correctness runs, not timed measurements; they were run
while the measurement lock was held by the #71 window, and their internal
pack/kernel timings are not used as benchmark evidence.

## 10. Skips and open decisions

* **Final quantized accuracy sign-off is not taken.** All measured #68 bounds
  pass against the fn3 FP32 run, but the decision belongs to the trained FP32
  investigation: the reference itself still carries the pre-existing `mean_abs`
  delta recorded in `docs/reports/fn3-22-quant-format.md` §2.3. This report
  measures the quantized path against that reference and does not widen or
  reinterpret the bound.
* **Latency is not a tuned-kernel result.** The fused kernel is the portable
  scalar reference with a four-row block, no SIMD and one thread. #71 owns the
  optimised kernels, blocking and the thread pool; this module provides the
  `WorkHint`/`RowChunks` seam and proves partition-invariance. The measured
  prefill and decode deltas in §7 are therefore a floor for the format, not the
  performance ceiling.
* **No prepared-artifact integration.** The prepared-model format and the
  session's weight materialisation are unchanged, so this backend retains the
  FP32 projections in addition to the packed copies. A packed artifact that
  replaces the FP32 slots (and, if wanted, a packed lm_head copy) is a
  planner/format change owned outside #69.
* **lm_head stays FP32.** Its right-hand side is a transpose computed in the
  step, not a weight slot. A packed copy would cost 41,009,712 B on the trained
  model and make the largest decode MatMul 3.77× smaller to stream; that is the
  open follow-up if decode latency becomes the constraint.
* **The Vulkan W8A16 path is #70.** Nothing here is device-specific; the
  arithmetic and registration rules are the reference #70 must match.
* **Host contention.** The machine is shared by up to five worker sessions; the
  locked runs record the load average and the `cargo`/`rustc` state with every
  measurement so a reader can judge the noise.
