# FN3-74 (#74): measured support scope at `17eee5f`

**Issue:** [#74](https://github.com/PetrouilFan/fastnn/issues/74), FN3-25 hardware
and platform evaluation.
**Measured tree:** `17eee5f2568ba38e2c57c5df578d98260e775adb` (`rewrite/fastnn-3`),
built in the pristine detached worktree
`/home/petrouil/Projects/github/fastnn-wt-74`.
**Model:** trained GPT-2, `onnx/model.onnx`, 498,126,358 B, SHA-256
`42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d`.
**GPU:** NVIDIA GeForce GTX 1080 Ti, UUID
`471b2bda-51e1-6451-e692-ff5d1e146f1e`, driver 580.196.07, 11,264 MiB.
**Scope:** measurement and documentation only. No library, test, version,
workflow or packaging file was changed. No remote write, push, tag, PR, issue
comment or issue transition was made.

The shared integration worktree advanced from `17eee5f` to `14260bf` (a
Python-API change that also makes the INT8 report optional) while this work was
in progress. Every number below was therefore re-measured in a fresh worktree
checked out at `17eee5f`; the `14260bf` tree is not the source of any table
here. The prepared artifacts from both trees are byte-identical (same SHA-256).

---

## Decision summary

- **The supported scope is one platform, one device family, one batch of one,
  and two models.** Linux x86_64, FP32 CPU and resident Vulkan on the physical
  GTX 1080 Ti, with `--weight-int8` as an opt-in storage mode on CPU and
  resident Vulkan. Tiny and trained GPT-2 are inference-verified; trained GPT-2
  is the only model measured at longer context.
- **Staging pooling is not what the supported scope needs.** The resident path
  moves 11.4 KB / 77.5 KB / 1.08 MB of host staging per 64 / 256 / 1024-token
  prefill and 6.4–14.0 KB per decode. The staged diagnostic moves 1.31 GB per
  256-token prefill and 216 MB per decode, and is 26.8× slower to first token.
  Residency, not a larger pool, is the release-relevant boundary.
- **The extra KV submissions are material, but their number is not the problem.**
  Every decode makes **50 `vkQueueSubmit` calls** at 64, 256 and 1023 tokens
  (48 KV-region copies + one token upload + one logits download). What grows is
  the work inside them: the live-prefix refresh copies 4.7 / 18.9 / 75.5 MB as
  18,720 / 74,016 / 294,912 separate 256-byte copy commands. Decode wall time
  grows 154.9 → 243.6 → 574.5 ms while the ledger's decode GPU time stays nearly
  constant (≈71–78 ms in this loaded campaign), so the growth is the KV copy
  chain and host-side recording, not kernel arithmetic.
- **Repeated transposes dominate decode engine time, not long-context growth.**
  At 1023 tokens the decode's ledger GPU time is ≈70.8 ms: `reduce_mean`
  (LayerNorm) ≈42.7 ms (60%), `transpose` ≈25.3 ms (36%), `matmul` ≈0 ms. The
  same transposes cost ≈25 ms at 64 tokens: they are a fixed per-layer cost, not
  the term that grows with context. At 1023-token prefill, `matmul` is 403 ms of
  638 ms (63%) and `transpose` is 33.5 ms (5%).
- **Duplicate FP32 + INT8 storage fits at 256 tokens and does not fit at the
  window on this host.** Separate peaks are 2.25 GiB (FP32) and 2.02 GiB
  (INT8); both sessions alive peak at 3.40 GiB. At 1023 tokens, FP32 alone peaks
  at 6.39 GiB and INT8 alone at 6.16 GiB; both together fail with
  `ERROR_OUT_OF_DEVICE_MEMORY` while requesting a 64 MiB chunk at 6.63 GiB live
  / 6.91 GiB reserved, with 6.72 GiB of VRAM free (4.17 GiB already used by the
  desktop). The duplicate weights are only +90.2 MB; the failure is the two live
  sessions' long-context state, not the packed copy.
- **INT8 is a storage mode, not a speed mode, at long context.** Vulkan prefill
  is 2.36× slower at 1024 tokens (2.568 s vs 1.089 s, mostly the per-session
  90.2 MB pack); CPU prefill is 3.42× slower (78.9 s vs 23.1 s, the serial
  quantized matmul). Decode is at parity on Vulkan (576.2 vs 574.5 ms at the
  window) and 2.6% slower on CPU at the same context, but the CPU comparison is
  load-confounded.
- **Everything outside that scope is not claimed** (macOS, Windows, arm64,
  AMD/Intel Vulkan, software Vulkan, CUDA, training, batch > 1, multi-GPU,
  quantized head, Python/wheel install). The explicit list is in §7.

---

## 1. What was run

### 1.1 Host and measurement conditions

| Item | Value |
|---|---|
| OS | Arch Linux (rolling), kernel current as of 2026-11-19 |
| CPU | AMD Ryzen 7 3700X, 16 logical cores |
| RAM | 31 GiB |
| GPU | NVIDIA GeForce GTX 1080 Ti, 11,264 MiB, UUID `471b2bda-51e1-6451-e692-ff5d1e146f1e` |
| Driver | 580.196.07 (`580.196.7.0`), Vulkan device API 1.4.312 |
| Validation | `VK_LAYER_KHRONOS_validation` 1.4.363, enabled in-process |
| Rust | rustc/cargo 1.100.0 |
| External VRAM | 4,272 MiB used by the desktop, 6,886 MiB free (= 4.17 GiB / 6.72 GiB) at the `--duo` probe |
| Host load | **not quiet**: `/proc/loadavg` 3.8–7.6 during this campaign, from a Java process at ~96% CPU and desktop GPU clients |

The host was a working desktop, not a quiesced benchmark host. A Java process
held one core at ~96% and Chrome/vesktop held GPU memory and CPU throughout.
Every table therefore carries the in-process `/proc/loadavg`. The absolute
latencies are **upper bounds**; the same-host ratios and the byte/queue counters
are the stable evidence. These numbers should not be read as the engine's best
case.

All GPU runs were serialized with
`flock /tmp/opencode/fn3-measure.lock`, selected the device by UUID, set
`FN3_VK_VALIDATION=1`, `FN3_VK_REQUIRE_PHYSICAL=1`, unset `VK_INSTANCE_LAYERS`,
and did not set `FN3_VK_STAGED` except in the explicit staged control. The
harness rejects a software adapter. Every Vulkan run reported **0 validation
errors and 0 warnings**; the loader emitted 289 `[validation][INFO]` lines per
process, which are not diagnostics. `raw/validation.csv` records the counts for
each arm and the two independent sources (the library's
`instance::validation_counts()` and the stderr grep).

### 1.2 Artifact and declared window

Prepared with the CLI from the pinned ONNX:

```sh
fastnn-infer prepare --onnx .../trained-gpt2/onnx/model.onnx \
  --out target/fn3-74/artifacts/trained-1024.fnn3 \
  --max-seq 1024 --max-past 1023 --max-total 1024
```

| Item | Trained | Tiny |
|---|---|---|
| ONNX bytes / SHA-256 | 498,126,358 / `42c1e92a…6c7c0d` | 27,897,974 / `e18370ba…8141a1` |
| Artifact bytes / SHA-256 | 498,075,184 / `570ce35b…4abe97` | 27,879,928 / `dc03b3c2…a1ecf2f` |
| Envelope | batch 1, seq 1..=1024, past 0..=1023, total 1..=1024 | same |
| Geometry | 12 layers, 768 hidden, 3072 MLP, 12/12 heads × 64, vocab 50,257 | 2 layers, 128 hidden, 512 MLP, 2/2 × 64 |
| Position table | 1024 rows | 1024 rows |
| KV cache at capacity | 75,497,472 B | 2,097,152 B |

The declared window is the model's full learned position table, not the CLI's
`prepare` default (`max-seq 128`). A **1024-token prefill cannot be followed by
a decode**: one more token needs position 1025, above the `total` envelope of
1024. The window decode was therefore measured as **1023-token prefill + one
decode**, which lands on position 1024. This is a real envelope limit, not a
harness choice.

### 1.3 Harness and definitions

The measuring binary is an external Rust harness, not a repository test. Its
source is in `raw/harness-main.rs.txt` and `raw/harness-Cargo.toml.txt`; it
links the `17eee5f` crate with `default-features = false` and features
`simd,parallel,fusion-forward,prepared-plan,vulkan` (no Python), loads the
`.fnn3` artifact, builds the model exactly as the CLI does, and runs:

- a fresh `Session` per iteration over one shared `PreparedModel`;
- `warmups` unmeasured iterations, then `iterations` measured ones;
- prefill of N tokens, then K teacher-forced decode steps;
- a full read of every returned logits tensor as a real caller would.

| Term | Definition |
|---|---|
| **TTFT** | `Session::prefill` from entry to the caller's read of all 50,257 logits. Excludes session construction and model load. |
| **Prefill** | the `Session::prefill` call alone (TTFT minus the logits read; the read is one 201,028-byte download). |
| **Decode** | one `Session::decode([token])` plus the logits read. Each `past` is a new shape, so every decode includes a plan-cache miss and specialisation; this is the CLI's normal per-token path, not a pre-specialised hot loop. |
| **Submissions** | `VulkanBackend::queue_submissions()`, i.e. real `vkQueueSubmit` calls, snapshotted around each step. The residency ledger's `submissions` counts step batches; both are reported because they differ (§5). |
| **Upload / readback** | `ResidencyStats.staged_upload_bytes` per phase and `ResidencyStats.readback_bytes` (`note_readback`), plus the ledger's per-op readback in the staged arm. |
| **Peak device-local** | sum of the allocator's non-host-visible chunk sizes (chunks are never released, so this is the device-local reservation high-water), and `MemoryStats::peak_live_bytes` (all live bytes). These are process-local, not the GPU total. |
| **Host RSS** | `/proc/self/status` `VmHWM`. |
| **p50 / p95** | nearest-rank on the sorted sample set. With n = 3 or 5, p95 is the maximum; the table states n. |

Iteration counts: 10 for the 64/256-token resident arms, 5 for the 1024-token
prefill arms, 3 for the 1023+1 arms, 5 for the 256-token INT8 and staged arms.
Decode samples are 8 per iteration except the 1023+1 arms, which have 1.

The exact commands are in §8.

---

## 2. Trained GPT-2, FP32

### 2.1 CPU

| Prompt | Decode | Iters / decode samples | TTFT p50 / p95 | Decode p50 / p95 | Peak host RSS | Loadavg |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8 | 10 / 80 | 1,046.1 / 1,088.5 ms | 289.0 / 314.0 ms | 1.48 GiB | 5.02 |
| 256 | 8 | 5 / 40 | 3,986.6 / 4,003.3 ms | 420.3 / 444.6 ms | 2.51 GiB | 6.39 |
| 1024 | 0 | 5 / — | 23,055.0 / 25,568.4 ms | — | 6.65 GiB | 7.23 |
| 1023 | 1 | 3 / 3 | 21,719.3 / 23,119.8 ms | 946.6 / 967.8 ms | 6.83 GiB | 7.61 |

CPU has no queue or transfer counters: submissions and bytes are N/A, not zero
in the sense of "measured zero". The 1024-token prefill allocates a large
transient: peak RSS is 6.65 GiB while the process returns to ≈1.1 GiB after the
session is dropped. This is a real host-memory cost of a long prefill, not a
retained cache.

### 2.2 Resident Vulkan

| Prompt | Decode | Iters / decode samples | TTFT p50 / p95 | Decode p50 / p95 | Device-local reserved / peak live | Peak host RSS | Submits prefill / decode | Loadavg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8 | 10 / 80 | 171.6 / 179.9 ms | 154.9 / 171.8 ms | 1.06 / 1.27 GiB | 1.34 GiB | 26 / 50 | 5.24 |
| 256 | 8 | 10 / 80 | 335.8 / 358.0 ms | 243.6 / 270.9 ms | 2.06 / 2.25 GiB | 1.34 GiB | 26 / 50 | 4.74 |
| 1024 | 0 | 5 / — | 1,088.8 / 1,124.4 ms | — | 6.38 / 6.26 GiB | 1.30 GiB | 26 / — | 4.47 |
| 1023 | 1 | 5 / 5 | 1,114.2 / 1,142.3 ms | 574.5 / 618.2 ms | 6.38 / 6.39 GiB | 1.31 GiB | 26 / 50 | 4.32 |

The device-local reservation grows **1.06 → 2.06 → 6.38 GiB** from 64 to 1024
tokens. At 1024 tokens the process peaks at 6.26 GiB of live device memory; with
6.72 GiB free on this desktop that is ≈0.46 GiB of headroom. A 1024-token
prefill on an 11 GiB card with other VRAM users is at the edge of what fits.

### 2.3 Traffic, per step

| Prompt | Prefill upload | Decode upload | Logits readback | Prepared once |
|---:|---:|---:|---:|---:|
| 64 | 11,432 B | 6,368 B | 201,028 B/step (1,809,252 B per 9-step chain) | 852 tensors / 497,764,984 B |
| 256 | 77,480 B | 7,904 B | 201,028 B/step | same |
| 1024 | 1,078,952 B | — | 201,028 B | same |
| 1023 | 1,076,884 B | 14,012 B | 201,028 B/step (402,056 B for 2 steps) | same |

The prefill staging upload scales with `seq²` because the causal mask is one
byte per element: 5,632 B at 64, 71,680 B at 256, 1,073,152 B at 1024. The
resident path otherwise moves only token IDs and the final logits.

---

## 3. Trained GPT-2, `--weight-int8` (INT8 weights, FP32 activations)

### 3.1 Coverage and fallback

| Item | Value |
|---|---|
| Packed / total MatMul instructions | **48 / 73** (65.8%) |
| FP32 MatMul instructions | **25** |
| Fallback codes | 24 × `computed-rhs` (per-layer attention score `Mul_1` and value transpose feeding `present.N.value`), 1 × `computed-rhs` (tied `transformer.wte.weight_transposed` LM head) |
| Packed payload | 90,243,072 B per **session** |
| Packed source | 339,738,624 B (73.4% reduction of the packed set) |
| Dense weights kept | 158,022,256 B |
| Shared weight (packed + dense) | 0 B |
| Whole-model reduction | 249,495,552 B = 50.1% of 497,764,984 B |
| Prepare | 304 dense tensors / 158,022,256 B + 90,243,072 B packed, once per session for the packed set |
| Pack cost | 848–901 ms CPU, 1.04–1.09 s Vulkan, first step of each session |

The LM head is **not quantized**. It is the single largest projection and it
stays FP32 because its right-hand side is an instruction output (the transpose
of the tied embedding), not a static weight. The 24 attention products are
dynamic by construction. No instruction is partially approximated: each is
wholly packed or wholly FP32.

### 3.2 Vulkan resident

| Prompt | Decode | Iters / decode samples | TTFT p50 / p95 | Decode p50 / p95 | Device-local reserved / peak | Peak host RSS | Submits prefill / decode | Loadavg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 8 | 5 / 40 | 1,515.7 / 1,560.9 ms | 241.9 / 257.3 ms | 1.88 / 2.02 GiB | 1.44 GiB | 170 / 50 | 4.56 |
| 1024 | 0 | 3 / — | 2,568.1 / 2,593.7 ms | — | 5.88 / 6.02 GiB | 1.40 GiB | 170 / — | 4.55 |
| 1023 | 1 | 3 / 3 | 2,674.6 / 2,724.0 ms | 576.2 / 597.0 ms | 6.00 / 6.16 GiB | 1.41 GiB | 170 / 50 | 4.32 |

The 170 prefill submissions are 26 step submissions + 144 packed-buffer uploads
(48 projections × 3 buffers), all in the first step. Decode returns to 50.

### 3.3 CPU

| Prompt | Decode | Iters / decode samples | TTFT p50 / p95 | Decode p50 / p95 | Peak host RSS | Loadavg |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 8 | 5 / 40 | 17,899.8 / 18,160.6 ms | 348.9 / 366.7 ms | 2.59 GiB | 4.28 |
| 1024 | 0 | 3 / — | 78,902.3 / 79,589.1 ms | — | 6.73 GiB | 4.39 |
| 1023 | 1 | 3 / 3 | 76,035.5 / 76,949.4 ms | 970.7 / 978.3 ms | 6.92 GiB | 3.83 |

The CPU quantized projection is **serial**. At M = 256 the prefill goes from
3.99 s (FP32, threaded) to 17.90 s (INT8); at M = 1024 from 23.06 s to 78.90 s.
Packing is only ≈0.85 s of that. This is a material long-prefill regression
that the 10-token measurement in `artifact-int8.md` did not expose.

### 3.4 Accuracy spot-check at 256 tokens

The CLI was run artifact-only on the same 256-token prompt in four arms. All
returned finite 50,257-element logits with the same argmax at both the prefill
(2068) and the first decode (12). The logits' printed min/max agree to four
decimals between CPU and Vulkan FP32; INT8 shifts them (prefill max
−223.3006 vs −223.2936; decode max −32.0244 vs −30.3715) while preserving
argmax and top-5 membership. Per-element max deltas at 256 tokens were **not**
computed in this task; this is a spot-check, not a parity study. `#70` owns the
per-element INT8 accuracy evidence and the 0.81710 trained max delta at 10
tokens.

---

## 4. Staged control (one length)

256-token prefill + 8 decodes, `FN3_VK_STAGED=1`, same artifact, same GPU,
validation on:

| Metric | Resident FP32 (§2.2) | Staged FP32 | Ratio |
|---|---:|---:|---:|
| TTFT p50 / p95 | 335.8 / 358.0 ms | 8,991.0 / 9,470.8 ms | 26.8× / 26.5× |
| Decode p50 / p95 | 243.6 / 270.9 ms | 1,976.4 / 2,147.7 ms | 8.1× / 7.9× |
| `vkQueueSubmit` prefill / decode | 26 / 50 | 775 / 776 | 29.8× / 15.5× |
| Prefill upload / readback | 77,480 B / 201,028 B | 1,311,548,072 B / 1,226,094,592 B | 16,932× / 6,099× |
| Decode upload / readback | 7,904 B / 201,028 B | 216,327,914 B / 196,853,878 B | 27,370× / 979× |
| Readback calls per 9-step chain | 9 | 7,199 | 800× |
| Peak host RSS | 1.34 GiB | 3.06 GiB | 2.3× |
| Device-local reserved / peak | 2.06 / 2.25 GiB | 1.06 / 0.86 GiB | — |
| Validation | 0 / 0 | 0 / 0 | — |
| Loadavg | 4.74 | 5.64 | — |

The staged path keeps the large transfers in host memory, so its device-local
reservation is lower; its host RSS and bytes are much higher. It is a
diagnostic, not a supported deployment.

The staged **INT8** control is refused, not silently run:

```text
error: Session::with_runtime: op matmul (rank-2 f32 matrix product) is unsupported
by backend vulkan: INT8 weights need the CPU backend or a resident Vulkan session
(the session is host-staged)
next: `--weight-int8` needs a resident session: drop the flag, or unset FN3_VK_STAGED;
a staged Vulkan session is rejected rather than run on the CPU
```

Exit code 5. Reproduced in the pristine worktree; see
`raw/duo-and-rejection.txt`.

---

## 5. Queue submissions and per-op device time

### 5.1 The two submission counters disagree

`VulkanBackend::queue_submissions()` counts real `vkQueueSubmit` calls. The
residency ledger's `ResidencySummary::submissions` counts **step batches**
(one per `begin_step`/`end_step`). For a resident FP32 256-token chain the
ledger reports 9 submissions (1 prefill + 8 decodes), while the device reports
26 + 8×50 = 426. The ledger is correct for "one batch per step"; it is not a
`vkQueueSubmit` count. Release notes that repeat "one submission per resident
step" should say "one step batch", not "one queue submission".

### 5.2 Where the 50 decode submissions come from

The trained graph is 12 layers. Per decode the KV cache performs, for every
layer:

- 2 × `copy_resident_regions` in `KvCacheState::append_with` (the new key row
  and value row);
- 2 × `copy_resident_regions` in `KvCacheState::live_with` (the whole live K
  prefix and V prefix into the contiguous past buffer the attention kernel
  binds).

That is 4 fence-waited submissions per layer = 48, plus the token upload and the
logits download = **50**. These copies are not ledger ops, so their device time
is not in `ledger_gpu_ms`; they are the residual between wall time and ledger
time.

The live-prefix copy is the one that grows. It copies
`2 × layers × kv_heads × (past+1) × head_dim × 4 B`:

| Context (past) | Live-prefix bytes per decode | 256-byte copy commands | Wall decode p50 | Ledger GPU ms (decode) | Residual |
|---:|---:|---:|---:|---:|---:|
| 64 | 4.79 MB | 18,720 | 154.9 ms | 78.4 ms† | 76.5 ms |
| 256 | 18.95 MB | 74,016 | 243.6 ms | 76.2 ms† | 167.4 ms |
| 1023 | 75.50 MB | 294,912 | 574.5 ms | 70.8 ms† | 503.7 ms |

† Auxiliary 2-iteration probe, same source, loadavg 4.3–5.2 (see
`raw/gpu-busy-*`). The ledger GPU figure is the sum of per-op `gpu_ns` for
prefill+one decode minus the prefill-only run; the KV copies are outside it.
The residual per copy command is ≈1.7–4.1 µs, consistent with host-side
recording of hundreds of thousands of tiny `vkCmdCopyBuffer` calls rather than
bandwidth. The 75.5 MB itself would take well under a millisecond at this
GPU's bandwidth.

**Conclusion:** the number of KV submissions is constant (50/token) and is not
what makes long context slow. The *row-granular copy chain* is: it scales with
context, is outside the ledger, and accounts for the measured 3.7× decode wall
growth from 64 to 1023 tokens. This is the single most actionable long-context
finding in this report.

### 5.3 Per-op device time

Pristine worktree, auxiliary 2-iteration probes, `gpu_ms` summed from the
residency ledger. The host was loaded (loadavg 4.3–5.2), so these are inflated
in absolute terms; the shares are the point.

**1023-token prefill (638 ms ledger time):**

| Op | Executions | GPU ms | Share |
|---|---:|---:|---:|
| matmul | 73 | 403.4 | 63.3% |
| softmax | 12 | 69.4 | 10.9% |
| reduce_mean | 50 | 51.4 | 8.1% |
| add | 159 | 34.8 | 5.5% |
| transpose | 61 | 33.5 | 5.3% |
| mul | 97 | 22.8 | 3.6% |
| reshape | 146 | 8.6 | 1.3% |
| split | 12 | 4.2 | 0.7% |

**Decode at 1023 tokens (≈71 ms ledger time, residual ≈504 ms):**

| Op | GPU ms | Share of ledger |
|---|---:|---:|
| reduce_mean | 42.7 | 60.3% |
| transpose | 25.3 | 35.7% |
| softmax | 8.6 | 12.2% |
| matmul | ≈0 (inside noise) | ≈0% |
| other pointwise/reshape | residual | ~−8% (noise from the two-run subtraction) |

The same shape holds at 64 and 256 tokens: decode engine time is dominated by
LayerNorm (`reduce_mean`) and `transpose`, not by matmul, and is nearly
context-invariant. A caller optimising decode should fuse or eliminate the
transpose/LayerNorm pair; a caller optimising long context must fix the KV row
copy.

---

## 6. Memory: FP32, INT8 and both in one process

### 6.1 Separate and concurrent peaks

| Arm | Prompt | Device-local reserved | Peak live | Host peak RSS | Loadavg |
|---|---:|---:|---:|---:|---:|
| FP32 resident | 64 | 1.06 GiB | 1.27 GiB | 1.34 GiB | 5.24 |
| FP32 resident | 256 | 2.06 GiB | 2.25 GiB | 1.34 GiB | 4.74 |
| FP32 resident | 1024 | 6.38 GiB | 6.26 GiB | 1.30 GiB | 4.47 |
| INT8 resident | 256 | 1.88 GiB | 2.02 GiB | 1.44 GiB | 4.56 |
| INT8 resident | 1024 | 5.88 GiB | 6.02 GiB | 1.40 GiB | 4.55 |
| FP32 + INT8 both alive | 256 | 3.19 GiB | 3.40 GiB | 1.40 GiB | 5.67 |
| FP32 + INT8 both alive | 1023+1 | — | **OOM** | — | 5.67 |

The `--duo` 256 run keeps one FP32 and one INT8 session alive, prefills both,
and decodes one token on each. It reports `max_abs` FP32→INT8 logits delta
0.361069 at the 256-token prefill and the same prefill argmax (2068); the
separate CLI 256-token parity arms also agree on the first decode argmax (12)
and on top-5 membership. The device-local reservation is 3.19 GiB, **1.5×** the
FP32 alone, not 2×: the 497.8 MB dense weights are shared and the transient
buffers are reused between the two prefills. The duplicate storage is only the
90.2 MB packed payload.

At 1023 tokens, the same program fails on the INT8 prefill:

```text
INT8 prefill: Allocation { bytes: 67108864, detail:
"vkAllocateMemory failed: ERROR_OUT_OF_DEVICE_MEMORY (-2):
resident: 67108864-byte device-local chunk, memory type 7;
live=7121267888 reserved=7419723776 chunks=68" }
```

with `nvidia-smi` reporting **4,272 MiB used / 6,886 MiB free** (`4,277 /
6,881` after). The process had already reserved 6.91 GiB; the free VRAM was
6.72 GiB. The first attempt (in the integration worktree) failed at a 192 MiB
chunk with 7.23 GiB live / 7.54 GiB reserved, so the failure is reproducible;
the exact failing allocation is not stable.

**Interpretation.** The duplicate FP32 + INT8 *weights* cost 90.2 MB, not 2×.
The real device-memory problem at the window is that two live long-context
sessions each retain several GiB of KV/past and per-session buffers; on this
host, with 4.17 GiB already used by the desktop, they cannot coexist at 1023
tokens. A deployment that wants both routes in one process must serialize the
sessions or reduce context. We cannot say from this measurement whether two
sessions would fit on a clean 11 GiB card with no desktop VRAM; we did not have
one.

### 6.2 Host memory

- CPU 1024-token prefill: peak RSS 6.65 GiB (FP32) / 6.73 GiB (INT8), returning
  to ≈1.0–1.2 GiB after the session is dropped. The transient dominates.
- Vulkan resident: peak RSS 1.30–1.44 GiB; the model weights are 498 MB and the
  rest is the artifact/import copy and session state.
- Staged 256: peak RSS 3.06 GiB, because the per-op host tensors are live.

---

## 7. Support matrix

### 7.1 What the four words mean here

| Word | Meaning in this report |
|---|---|
| **Built** | The CLI and/or harness compiled from `17eee5f` on that platform in this task. |
| **Installed** | A user-facing install (system path, wheel, package) was performed and the installed artifact was exercised. |
| **Inference-verified** | A full prefill + cached decode (or a prefill at the declared window) produced finite 50,257-element logits; where stated, argmax/top-5 were compared across arms. |
| **Physically-verified** | The Vulkan run selected the GTX 1080 Ti by UUID, `software=false`, `FN3_VK_REQUIRE_PHYSICAL=1`, validation on, 0 errors / 0 warnings, and a real `vkQueueSubmit`/dispatch path. |

### 7.2 Matrix

| Platform | Feature / backend | Model | Built | Installed | Inference-verified | Physically-verified |
|---|---|---|---|---|---|---|
| Linux x86_64 | CPU FP32 | tiny GPT-2 | yes | no | yes | n/a |
| Linux x86_64 | CPU FP32 | trained GPT-2 | yes | no | yes (64/256/1024/1023) | n/a |
| Linux x86_64 | CPU INT8 weights | trained GPT-2 | yes | no | yes (256/1024/1023) | n/a |
| Linux x86_64 | CPU INT8 weights | tiny GPT-2 | not run | no | not measured | n/a |
| Linux x86_64 | Vulkan resident FP32 | tiny GPT-2 | yes | no | yes | yes |
| Linux x86_64 | Vulkan resident FP32 | trained GPT-2 | yes | no | yes (64/256/1024/1023) | yes |
| Linux x86_64 | Vulkan resident INT8 | tiny GPT-2 | yes | no | yes | yes |
| Linux x86_64 | Vulkan resident INT8 | trained GPT-2 | yes | no | yes (256/1024/1023) | yes |
| Linux x86_64 | Vulkan staged FP32 | trained GPT-2 | yes | no | yes (control only) | yes |
| Linux x86_64 | Vulkan staged INT8 | trained GPT-2 | yes | no | refused (typed `UnsupportedOp`) | n/a |
| Linux x86_64 | Python facade / wheel | any | not built here | no | not measured | not measured |
| Linux x86_64 | CLI artifact-only (`--model` alone) | trained/tiny | yes | no | yes (all arms) | yes where Vulkan |

**Installed is "no" for every row.** No `cargo install`, wheel install or
system-path deployment was performed. The binaries ran from
`target/release/`. A wheel exists at `target/ci-packaging/dist/` but it was
built at 03:00, before `17eee5f`, and is not evidence for this SHA. Clean
install and the release wheel are #76's gate.

### 7.3 Combinations explicitly **not** claimed

- **macOS** (Intel or Apple silicon), **Windows** (x86_64 or arm64), and
  **Linux aarch64**. Not built, not run, no platform code exercised.
- **AMD Vulkan** (RADV/AMDVLK) and **Intel Vulkan** (ANV). Only the NVIDIA
  580.196.07 driver and GP102 were opened. No AMD or Intel adapter was
  available on this host.
- **Software Vulkan** (Mesa lavapipe/llvmpipe). The ICD is installed and was
  enumerated, but never selected; a software run is explicitly not physical
  evidence in this harness.
- **CUDA**, ROCm, oneAPI, Metal, NPU or any non-Vulkan accelerator. There is no
  CUDA backend in 3.0.
- **Training / fine-tuning / autograd.** The 3.0 path is inference-only.
- **Batch > 1.** `Session` rejects a KV config with `batch != 1`; every
  measurement is one sequence.
- **Multi-GPU**, device groups, peer copies, or a device other than index 0.
- **A quantized LM head, quantized attention products, or a fully INT8 model.**
  Coverage is 48/73 MatMuls; the head and the 24 dynamic attention products
  stay FP32. No claim is made for a quantized head.
- **Python API / wheel install / `import fastnn`**. Not exercised here.
- **Generation quality, perplexity, long-form fidelity, or sampling.** The
  measurements are latency, memory, traffic and a spot-check of argmax/top-5.
- **Context longer than 1024 tokens.** The learned position table is 1024 rows;
  the envelope refuses more. A 1024-token prefill cannot be decoded in the same
  session (position 1025).
- **Two concurrent long-context sessions on this host.** Measured to OOM at 1023
  tokens with the desktop's 4.17 GiB of VRAM in use.
- **Staged Vulkan as a deployment mode.** It is kept as a comparison and
  debugging path; it is 8–27× slower and refuses INT8.
- **Thermal throttling, clock locking, ECC, or 24/7 reliability.** No clocks
  were locked and no temperature was recorded.
- **Any performance claim relative to ORT, PyTorch, llama.cpp or a server.**
  This report compares fn3 to itself under one host's conditions.

---

## 8. Reproduction

### 8.1 Environment and lock

```sh
export TMPDIR=$PWD/target/fn3-74/tmp
LOCK=/tmp/opencode/fn3-measure.lock
UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e
env -u VK_INSTANCE_LAYERS -u FN3_VK_STAGED \
  FN3_VK_VALIDATION=1 FN3_VK_REQUIRE_PHYSICAL=1 FN3_VK_DEVICE_UUID=$UUID \
  flock $LOCK -c '<command>'
```

The staged control replaces `-u FN3_VK_STAGED` with `FN3_VK_STAGED=1`. CPU arms
took the same lock (they are a device too) but need none of the Vulkan
variables.

### 8.2 Build

```sh
cd /home/petrouil/Projects/github/fastnn-wt-74     # detached at 17eee5f
CARGO_BUILD_JOBS=6 cargo build --release \
  --no-default-features --features cli,vulkan --bin fastnn-infer

# external harness (source in raw/harness-main.rs.txt)
cd target/fn3-74/harness
CARGO_TARGET_DIR=/home/petrouil/Projects/github/fastnn-wt-74/target \
  CARGO_BUILD_JOBS=6 cargo build --release
```

### 8.3 Prepare

```sh
fastnn-infer prepare --onnx /tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx \
  --out target/fn3-74/artifacts/trained-1024.fnn3 \
  --max-seq 1024 --max-past 1023 --max-total 1024
```

The tiny artifact uses the same envelope and the pinned tiny ONNX. Both
artifacts' SHA-256 are in §1.2.

### 8.4 Arms

The generic harness command is:

```sh
target/release/fn3-74-harness \
  --artifact target/fn3-74/artifacts/trained-1024.fnn3 \
  --device {cpu|vulkan} --prompt N --decode K \
  --iterations I --warmups W [--weight-int8] [--duo] --label LABEL
```

The exact arms and their `--iterations/--warmups`:

| Label | Device | Prompt | Decode | Iters | Warmups | Flag |
|---|---|---:|---:|---:|---:|---|
| `trained-fp32-cpu-p64` | cpu | 64 | 8 | 10 | 2 | — |
| `trained-fp32-cpu-p256` | cpu | 256 | 8 | 5 | 2 | — |
| `trained-fp32-cpu-p1024` | cpu | 1024 | 0 | 5 | 1 | — |
| `trained-fp32-cpu-p1023-d1` | cpu | 1023 | 1 | 3 | 1 | — |
| `trained-fp32-vk-p64` | vulkan | 64 | 8 | 10 | 2 | — |
| `trained-fp32-vk-p256` | vulkan | 256 | 8 | 10 | 2 | — |
| `trained-fp32-vk-p1024` | vulkan | 1024 | 0 | 5 | 1 | — |
| `trained-fp32-vk-p1023-d1` | vulkan | 1023 | 1 | 5 | 1 | — |
| `trained-int8-vk-p256` | vulkan | 256 | 8 | 5 | 1 | `--weight-int8` |
| `trained-int8-vk-p1024` | vulkan | 1024 | 0 | 3 | 1 | `--weight-int8` |
| `trained-int8-vk-p1023-d1` | vulkan | 1023 | 1 | 3 | 1 | `--weight-int8` |
| `trained-int8-cpu-p256` | cpu | 256 | 8 | 5 | 1 | `--weight-int8` |
| `trained-int8-cpu-p1024` | cpu | 1024 | 0 | 3 | 1 | `--weight-int8` |
| `trained-int8-cpu-p1023-d1` | cpu | 1023 | 1 | 3 | 1 | `--weight-int8` |
| `trained-fp32-vk-staged-p256` | vulkan | 256 | 8 | 5 | 1 | `FN3_VK_STAGED=1` |
| `trained-duo-vk-p256` | vulkan | 256 | 1 | 1 | 0 | `--duo` |
| `trained-duo-vk-p1023` | vulkan | 1023 | 1 | 1 | 0 | `--duo` (OOM) |
| `tiny-fp32-cpu-p64` | cpu | 64 | 4 | 5 | 1 | — |
| `tiny-fp32-vk-p64` | vulkan | 64 | 4 | 5 | 1 | — |
| `tiny-int8-vk-p64` | vulkan | 64 | 4 | 5 | 1 | `--weight-int8` |

The prompt is the deterministic 17-token fixture cycle repeated to N:
`464,2068,7586,21831,18045,625,262,16931,3290,13,4874,2402,257,640,11,612,373`.
The CLI parity arms used the same 256-token prompt with `--decode 640 --top-k 5
--steps`, artifact-only (`--model`, no `--onnx`).

### 8.5 Raw evidence

`docs/reports/fn3-74/raw/`:

- `*.out` — the harness stdout for every arm: human summary, JSON line, raw
  timing samples, per-op device time, memory and validation counts.
- `validation.csv` — per-arm validation errors/warnings from both the library
  counter and the stderr grep, plus loadavg.
- `duo-and-rejection.txt` — the 1023-token `--duo` OOM, the `nvidia-smi`
  before/after, and the staged-INT8 typed rejection.
- `harness-main.rs.txt`, `harness-Cargo.toml.txt` — the measuring binary.
- `parity-*.out` — the four 256-token CLI arms and their logits summaries.
- `prepare-fp32.out`, `prepare-int8.out` — zero-warmup runs that expose the
  one-time dense prepare and the per-session packed prepare counters.
- `gpu-*.out` — the auxiliary prefill-only and prefill+1-decode probes used for
  the per-op split and the ledger GPU totals.

---

## 9. What this report does not establish

1. **It is not a quiet-host benchmark.** Loadavg 3.8–7.6 and 4.17 GiB of
   external VRAM. The absolute numbers are conservative; a quiesced host will
   be faster, especially the CPU and the staged arm. The queue counts, byte
   counts, validation results and OOM are load-independent.
2. **It is not a long-context numerical parity gate.** The four 256-token arms
   agree on argmax/top-5 and the FP32 CPU/Vulkan logits' printed min/max; no
   per-element max delta or KL was computed at 256 or 1023 tokens.
3. **It does not establish a maximum safe context for every GPU.** 1024 tokens
   fits on this 11 GiB card with 6.72 GiB free; the process peaks at 6.26 GiB
   (FP32) and 6.02 GiB (INT8). Smaller cards, more desktop VRAM, or two live
   sessions will fail earlier. The device-local figures are process-local.
4. **It does not measure the Python path.** The wheel in the tree predates
   `17eee5f`; no install was performed.
5. **It does not measure tiny at long context.** Tiny was verified at 64 tokens
   on CPU, resident FP32 and resident INT8; its long-context memory profile is
   not reported.
6. **It does not resolve the 1024-prefill-then-decode envelope limitation.** A
   decode at the window requires a 1023-token prefill; a 1024-token prefill ends
   at the last position.
7. **It does not attribute the whole decode residual.** The gap between wall
   decode and ledger GPU time is consistent with the KV row-copy chain (§5.2),
   but the copies are not individually instrumented. The region counts and
   bytes are derived from the code path and the measured `50` submissions, not
   from a per-copy timestamp.
8. **It does not repeat the 10-token INT8 accuracy evidence.** That remains
   `docs/reports/fn3-30-vulkan-quant.md` and #70; this report only extends the
   latency, memory and coverage measurements.

### 9.1 Measurements not completed

- **FP32 + INT8 together at 1023 tokens.** Both attempts failed with
  `ERROR_OUT_OF_DEVICE_MEMORY`; the result is the OOM in §6.1, not a latency or
  peak-memory number. A clean-GPU (no desktop VRAM) attempt was not possible
  because the desktop held 4.17 GiB throughout.
- **Per-element INT8 accuracy at 256/1023 tokens.** Only argmax, top-5
  membership and the CLI's four-decimal logits summary were compared; no
  full-vector max/RMS/KL.
- **Tiny GPT-2 CPU INT8 and tiny long context.** Tiny was verified at 64 tokens
  on CPU FP32, resident FP32 and resident INT8; CPU INT8 and long context were
  not run.
- **Python facade and wheel install.** Not built or installed here; the wheel in
  the tree predates `17eee5f`.
- **A 1024-token prefill followed by a decode.** Impossible under the declared
  envelope (position 1025 exceeds the model's 1024-row table), so the window
  decode is a 1023-token prefill + one token.
- **Per-copy KV timestamps.** The 48 KV submissions are counted at the
  `vkQueueSubmit` level, but each of the 294,912 row copies is not individually
  timestamped; the residual attribution in §5.2 is derived from the code path
  and the measured submission counts.

---

## 10. Handoff

- Measurement SHA: `17eee5f2568ba38e2c57c5df578d98260e775adb`.
- Pristine measurement worktree:
  `/home/petrouil/Projects/github/fastnn-wt-74` (detached at `17eee5f`).
- Docs commit: the commit that adds this file and the ledger pointer; see
  `git log -1 -- docs/reports/fn3-74-support-matrix.md`.
- The shared integration worktree is at `14260bf` (Python INT8 API) and is not
  the source of any number here.
- No remote writes, no issue transitions.
