# FN3-25 (#71) — CPU inference measured, optimized, and thread-scaled

**Branch** `fn3/71-cpu-perf` · **Base** `672cec9faf089ef0840179f6edfdf33779b576a1`
(verified with `git rev-parse HEAD` before the first edit) · **Issue** #71 · **Tracking
issue** #46 · Local commits only; no push, tag, PR, merge or release.

Two rounds of work on one branch:

1. **Algorithmic**: the two measured bottlenecks were a per-step 25.7 MB weight
   transpose and a per-element-indexed matmul. Both were replaced, and both
   replacements are bit-identical to the scalar reference.
2. **Thread scaling**: a per-runtime thread budget, one process-wide pool,
   output tiling (rows for prefill, disjoint output channels for one-token
   projections), a measured serial/parallel threshold, and a sweep of
   `1/2/4/8/12/16/Auto` on both pinned models.

Raw logs and both standalone studies are in `docs/reports/fn3-25/`. The tables
below are generated from those logs; no value was typed by hand.

---

## 1. Headline

| Model | Workload | Threads | Before | After | Speedup |
|---|---|---|---|---|---|
| tiny GPT-2 | prefill 10 | 1 | 270.8 ms | 13.96 ms | **19.4×** |
| tiny GPT-2 | prefill 32 | 1 | 475.0 ms | 23.98 ms | **19.8×** |
| tiny GPT-2 | decode (10–17 cached) | 1 | 188.7 ms | 6.15 ms | **30.7×** |
| tiny GPT-2 | decode (32–39 cached) | 1 | 191.1 ms | 6.73 ms | **28.4×** |
| trained GPT-2 | prefill 10 | 1 | not measured at base¹ | 331.8 ms | — |
| trained GPT-2 | prefill 10 | Auto | — | 239.1 ms | 1.37× over 1 thread |
| trained GPT-2 | decode chain 12→16 | Auto | — | 135.7 ms | 1.03× over 1 thread |
| tiny GPT-2 | decode after prefill 10 | Auto | 10.92 ms (at 16 t) | 6.14 ms | 1.78×, and better than the best fixed (6.58) |

¹ The base-commit harness only loads the tiny model (its pinned-model gate
asserts the tiny digest), so the trained "before" could not be produced from the
base binary without inventing a comparison. The trained table reports this
branch at one thread, which *is* the scalar/portable path, and its speedup over
that. This is called out again in §12.

The "After" arm here is this branch at a one-thread budget; threading on top of
it is reported separately in §7 so a SIMD or packing win is never attributed to
threads.

---

## 2. Host, build and method

### 2.1 Host

| Item | Value |
|---|---|
| CPU | `AMD Ryzen 7 3700X 8-Core Processor` (`lscpu`), family 23 model 113, microcode 0x8701035 |
| Topology | 8 physical cores, 2 threads/core (SMT), 1 socket, 1 NUMA node (CPUs 0–15) |
| Process CPU limits | `cpuset.cpus.effective` = `0-15`, affinity `0-15`, no cgroup v2 `cpu.max` quota, 16 CPUs visible to `std::thread::available_parallelism()` |
| RAM | 31 GiB |
| Rust | `rustc 1.100.0 (0ef1bc386 2026-11-09)` |
| Build | `--release`, `lto = true`, `codegen-units = 1`, `opt-level = 3`, `panic = "abort"` |
| Tiny model | `/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx`, 27,897,974 B, sha256 `e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1` — matches `fixtures/parity/tiny-gpt2/model.lock.json` |
| Trained model | `/tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx`, 498,126,358 B, sha256 `42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d` — the #68 pin |

Both digests were computed locally by the harness on every load; neither is
copied from the issue text. Weights are not committed.

### 2.2 Measurement discipline

* Every authoritative window — **build and timed runs together** — ran under
  `flock /tmp/opencode/fn3-measure.lock -c '/tmp/opencode/fn3-71/authoritative.sh'`,
  and the script refuses to start while any other `cargo`/`rustc` is alive
  (`pgrep -a 'cargo|rustc'`, retried for up to 20 minutes). The window that
  produced the tables below started with `loadavg 2.42 3.98 3.90` and one
  runnable process; the load average was decaying from earlier work, and the
  per-row loadavg is recorded so a slow row can be matched to its condition.
* Tiny model: 30 iterations after 3 warmups. Trained: 4 after 2 (a trained
  prefill is ~0.3–0.4 s).
* p50/p95 are nearest-rank percentiles over the recorded samples.
* Each row carries an FNV-1a checksum of the logits. **Every thread budget
  produced the same checksum as the one-thread run**, on both models — the
  tiling changes which thread computes a value, never the order of the
  additions.
* Two decode protocols are reported, because they answer different questions:
  * **decode after prefill** — one fresh prefill on the same backend, then a
    timed single-token step. This is the first token of a turn and includes
    whatever the prefill left behind.
  * **decode chain** — one prefill, two untimed steps, then a timed run of
    successive steps (`past` grows by one each step, and the row prints the
    context range). This is steady-state generation.

  The distinction matters: §7.4 shows the two protocols disagree, and the
  reason is measurable.

### 2.3 `perf` was not available

`perf` is not installed: `perf` gives `zsh: command not found: perf`, and
`/usr/bin` holds only `perfquery` (the libdrm tool). `/proc/sys/kernel/perf_event_paranoid`
is `2`, which would permit user-space sampling, so the blocker is the missing
package, not policy. Nothing was installed, because that changes the host under
measurement. The profile instead comes from internal timers: `TimingBackend` in
`tests/fn3_cpu_perf.rs` delegates to the production `CpuBackend` and times every
`execute_op`, so preflight and dispatch are the real ones. The gap between the
step and the sum of kernel times is reported as `session_overhead` rather than
hidden.

---

## 3. Before: the profile

Per-op attribution, tiny model, one thread (`PROFILE_CLASS` rows):

### 3.1 10-token prefill

| Rank | Op | Calls | Time | % of step |
|---|---|---|---|---|
| 1 | `transpose` | 11 | 198.2 ms | 65.7% |
| 2 | `matmul` | 13 | 99.6 ms | 33.0% |
| 3 | `add` | 24 | 0.47 ms | 0.16% |
| 4 | `reduce_mean` | 10 | 0.22 ms | 0.07% |
| 5 | `split` | 2 | 0.19 ms | 0.06% |
| — | everything else | 84 | 0.52 ms | 0.17% |
| — | `session_overhead` | — | 2.2 ms | 0.7% |

and one site is almost the whole of it:

| Site | Calls | Time | % of step |
|---|---|---|---|
| `transpose [50257, 128] -> [128, 50257]` | 1 | 196.9 ms | 67.6% |
| `matmul [1, 10, 128] [128, 50257] -> [1, 10, 50257]` | 1 | 82.1 ms | 28.3% |
| `matmul [10, 128] [128, 512] -> [10, 512]` | 2 | 1.70 ms | 0.59% |
| `matmul [10, 512] [512, 128] -> [10, 128]` | 2 | 1.60 ms | 0.55% |
| `matmul [10, 128] [128, 384] -> [10, 384]` | 2 | 1.25 ms | 0.43% |

### 3.2 Decode at 10 cached tokens

| Rank | Op | Calls | Time | % of step |
|---|---|---|---|---|
| 1 | `transpose` | 11 | 188.3 ms | 94.9% |
| 2 | `matmul` | 13 | 9.57 ms | 4.8% |
| 3 | `concat` | 5 | 0.18 ms | 0.09% |
| 4 | `add` | 24 | 0.05 ms | 0.02% |
| — | everything else | 96 | 0.13 ms | 0.07% |
| — | `session_overhead` | — | 0.31 ms | 0.16% |

### 3.3 Why

The pinned graphs tie `lm_head` to the token-embedding table, so the plan
contains `MatMul(x, Transpose(W))` with `W` the `[50257, 128]` table, and
`Transpose` is an instruction like any other: it runs every step. The alpha
kernel walked the output in row-major order and recomputed a checked index per
element whose source was `50257 × 4 = 201 KB` away — one cache miss and a full
index computation per element, 6,432,896 times.

`MatMul` computed one output element at a time through a `Locator` and two
checked ranges, then walked the row with `step_by`: 64 M multiply-accumulates at
~0.8 GFLOP/s.

Neither is an arithmetic problem. The first is a memory access pattern; the
second is index arithmetic on an `O(n³)` loop.

---

## 4. Retained change 1 — cache-friendly `Transpose`

`src/fn3/backend/cpu/kernels/transform.rs`. The scalar reference is kept for
every case it defines. The fast path applies only when the permutation is the
identity on all but the last two axes, both views are contiguous, and the element
is 4 or 8 bytes — the tied-embedding transpose and the attention layout swaps.
It reads the source in ascending `(row, column)` order and writes the matching
destination rows, with at most 128 destination streams per pass.

### 4.1 The access-pattern study

Measured on the real `[50257, 128] -> [128, 50257]` `f32` buffer with a
standalone `rustc -O` program (`docs/reports/fn3-25/transpose-shapes.rs`), median
of 12 runs, 51.4 MB of traffic per run:

| Variant | Time | Effective bandwidth |
|---|---|---|
| Source-sequential, plain (chosen shape) | 2.91 ms | 17.7 GB/s |
| Source-sequential, 4×4 blocks inside a 32×32 tile | 5.54 ms | 9.3 GB/s |
| Source-sequential, 4×4 blocks inside a 64×64 tile | 4.51 ms | 11.4 GB/s |
| Source-sequential, 4×4 blocks inside a 16×16 tile | 7.32 ms | 7.0 GB/s |
| Rectangle tiles 32×512 … 128×1024 | 7.4–9.0 ms | 5.7–7.0 GB/s |
| Destination-contiguous, no tiling | 25.3 ms | 2.0 GB/s |
| Source-sequential with 4-row unroll | 3.74 ms | 13.8 GB/s |

The plain source-sequential loop wins. The source is one prefetch-friendly
ascending stream and each destination row is still touched in ascending order,
so the store buffers combine writes without help; tiling breaks the stream into
short bursts and adds index arithmetic for no memory benefit. The unrolled
variants are not better either.

### 4.2 Contract

Byte-exact by construction: each element is copied once, unchanged, with no
arithmetic. The module doc records that the *store order* changed (per pass) and
why that is unobservable. Parity fixtures, `fn3_cpu_reference` and the
whole-model checksums are unchanged to the bit.

---

## 5. Retained change 2 — row-major blocked/AVX2 `MatMul`

`src/fn3/backend/cpu/kernels/gemm.rs` (new), `kernels/matmul.rs`,
`src/fn3/backend/cpu/simd.rs` (new). `matmul.rs` keeps its contract: rank ≥ 2,
NumPy batch broadcasting, operands addressed through metadata so a transposed or
sliced view needs no copy, and **one `f32` accumulator per output element,
products added in ascending `k`, no fused multiply-add**.

### 5.1 Why no bit moved

The AVX2 kernel vectorises **across output columns**: lane `j` of an accumulator
holds one output element and sees exactly one `mul` and one `add` per `k` in
ascending order. There is no horizontal reduction, and `mul`/`add` are separate
instructions (`_mm256_mul_ps` then `_mm256_add_ps`, never `_mm256_fmadd_ps`), so
the rounding sequence is the reference's. The scalar fallback keeps 16
independent accumulators for the same reason.

Three tests pin it:

* `gemm::tests::every_shape_matches_the_reference_at_every_thread_count` — 15
  shapes (both `lm_head` shapes, awkward tails, `k = 0`) at 1, 2, 3, 4, 8 and 16
  threads against an independent ascending-`k` evaluation, bit-for-bit;
* `gemm::tests::blocked_scalar_is_bit_identical_to_the_reference` — the portable
  fallback, exercised directly on a host that has AVX2;
* `tests/fn3_cpu_perf.rs::kernel_matmul_is_bit_identical_to_ascending_k_scalar_order`
  — the backend entry point at 1, 2 and Auto.

### 5.2 Portability and `unsafe`

Nothing is compiled with `-C target-feature`. The AVX2 functions carry
`#[target_feature(enable = "avx2")]` and run only after
`is_x86_feature_detected!("avx2")`; without AVX2 the blocked scalar kernel
produces identical bytes, and on non-x86_64 no intrinsic is compiled. A build
cannot fail for lack of a target feature. `unsafe` appears only in that kernel,
with a local `SAFETY:` block covering aliasing (each task owns a disjoint output
region; inputs are separate allocations and the plan rejects read/write slot
aliasing), bounds (window lengths checked once, restated in the kernel), alignment
(`loadu`/`storeu` require none) and lifetime (no pointer outlives the call).

### 5.3 Coordinates with #23

The dispatch condition is a property of the operands (four matrix-axis strides
plus a contiguous output), not of a model, and the reference path remains for
anything that fails it. A shared matmul dispatch can call `row_major_slab` or
replace it; the bit-exactness tests are the interface.

---

## 6. Budget and tiling interface (for #69)

Everything lives under `src/fn3/backend/cpu/`, so quantized execution can reuse
it without touching `src/fn3/backend/mod.rs`.

```rust
pub enum ThreadBudget { Auto, Fixed(usize) }         // Default = Auto
impl ThreadBudget { pub fn threads(self) -> usize }  // clamped to visible CPUs

impl CpuBackend {
    pub fn new() -> CpuBackend;                       // Auto
    pub fn with_thread_budget(budget: ThreadBudget) -> CpuBackend;
    pub fn thread_budget(&self) -> ThreadBudget;
    pub fn threads(&self) -> usize;                   // resolved for the next instruction
}

pub fn cpu_process_budget() -> usize;                 // Auto's value in this process
pub fn cpu_thread_budget_note() -> &'static str;      // why it is that value
pub fn parallel_stats() -> CpuParallelStats;          // tasks, ops, max_active, threads
pub fn reset_parallel_stats();                        // start an observation window
pub fn warm_parallel_pool();                          // pay pool start-up outside a sample
pub const parallel::THREADS_ENV: &str = "FASTNN_CPU_THREADS";
```

Guarantees #69 can rely on:

1. **Auto is the default** and resolves to `available_parallelism()`, which on
   Linux honours affinity and a cgroup v2 CPU quota.
2. An **explicit cap** is clamped to that visible limit and never raises it.
3. A budget of **one is exactly serial** and never enters the pool (`parallel_ops
   == 0` is asserted), so one-thread numbers are serial and not "one worker plus
   the caller".
4. **One pool for the process** — Rayon's global pool, which the 2.x CPU path
   already uses. No pool per operation/backend/session, and `parallel::in_worker()`
   refuses a nested parallel region. There are no nested GEMM/BLAS pools.
5. Kernels choose *how many output tiles to submit*, so effective concurrency is
   `min(budget, tiles, pool)`. Tiles are disjoint, so results are identical at
   every budget.
6. **Measured work-size threshold**: `MIN_MACS_PER_TASK = 4 << 20` — a tile needs
   at least 4 M multiply-accumulates. §7.3 explains the measurement.
7. `CpuParallelStats` is incremented **inside** the chunk bodies, so it proves
   work rather than configuration.

### 6.1 Version-neutral public naming (separate pre-stable cleanup)

Recorded, not implemented: the stable-facing names could be `fastnn::inference`
(Rust), `fastnn.inference` (Python) and the `fastnn-infer` binary, replacing the
`fn3` working name. That rename touches `src/fn3/mod.rs`, `src/lib.rs`,
`fastnn/fn3/__init__.py` and the CLI, none of which this issue owns.

---

## 7. Thread scaling

### 7.1 Does the budget reach the kernels?

`thread_budget_reaches_the_kernels_and_preserves_logits` asserts, per budget,
that `max_active > 1` for every budget above one, that `max_active <= budget`,
that a one-thread budget takes the serial path, and that the combined prefill +
decode logits checksum equals the one-thread run's.

| Model | Budget | Selected | Observed max concurrent | Observed distinct threads | Tasks | Parallel ops | Serial ops | Checksum |
|---|---|---|---|---|---|---|---|---|
| tiny-gpt2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | `0xd7d884c22480b742` |
| tiny-gpt2 | 2 | 2 | 2 | 3 | 60 | 30 | 0 | `0xd7d884c22480b742` |
| tiny-gpt2 | 4 | 4 | 4 | 5 | 120 | 30 | 0 | `0xd7d884c22480b742` |
| tiny-gpt2 | 8 | 8 | 8 | 15 | 240 | 30 | 0 | `0xd7d884c22480b742` |
| tiny-gpt2 | 12 | 12 | 10 | 16 | 300 | 30 | 0 | `0xd7d884c22480b742` |
| tiny-gpt2 | 16 | 16 | 10 | 16 | 300 | 30 | 0 | `0xd7d884c22480b742` |
| tiny-gpt2 | auto | 16 | 10 | 16 | 300 | 30 | 0 | `0xd7d884c22480b742` |
| trained-gpt2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | `0x0a4c2517d50d8517` |
| trained-gpt2 | 2 | 2 | 2 | 3 | 304 | 152 | 0 | `0x0a4c2517d50d8517` |
| trained-gpt2 | 4 | 4 | 4 | 6 | 608 | 152 | 0 | `0x0a4c2517d50d8517` |
| trained-gpt2 | 8 | 8 | 8 | 14 | 736 | 152 | 0 | `0x0a4c2517d50d8517` |
| trained-gpt2 | 12 | 12 | 10 | 16 | 748 | 152 | 0 | `0x0a4c2517d50d8517` |
| trained-gpt2 | 16 | 16 | 10 | 15 | 748 | 152 | 0 | `0x0a4c2517d50d8517` |
| trained-gpt2 | auto | 16 | 10 | 15 | 748 | 152 | 0 | `0x0a4c2517d50d8517` |

The peak of 10 concurrent chunks at 12/16 threads is structural: a prefill's
`lm_head` has ten rows, and output-row tiling cannot produce more than ten tiles
for it. The trained model's `serial_ops` column is zero because an instruction
that the threshold keeps serial exits before the chunk helper is called; the
`0` in the one-thread row is the same fact.

### 7.2 Tiny GPT-2 sweep (median of two runs)

`FN3_PERF_ITERS=30`, 3 warmups, both runs under the measurement lock with
`loadavg 1.49–2.26` at start and one runnable process. `Selected` is what the
runtime resolved at the moment the row was measured, so an `Auto` row shows the
narrowing of §12.2.

| Workload | Budget | Selected | Eff. max | p50 | p95 | vs 1 thread |
|---|---|---|---|---|---|---|
| prefill 10 | 1 | 1 | 0 | 13.391 ms | 14.154 ms | 1.00× |
| prefill 10 | 2 | 2 | 2 | 12.315 ms | 14.723 ms | 1.09× |
| prefill 10 | 4 | 4 | 4 | 11.434 ms | 12.800 ms | 1.17× |
| prefill 10 | 8 | 8 | 8 | 9.720 ms | 11.270 ms | 1.38× |
| prefill 10 | 12 | 12 | 12 | 9.527 ms | 10.223 ms | 1.41× |
| prefill 10 | 16 | 16 | 15 | 9.459 ms | 10.019 ms | 1.42× |
| prefill 10 | auto | 16 | 15 | **9.379 ms** | 9.804 ms | **1.43×** |
| prefill 32 | 1 | 1 | 0 | 23.690 ms | 25.522 ms | 1.00× |
| prefill 32 | 2 | 2 | 2 | 21.447 ms | 23.838 ms | 1.10× |
| prefill 32 | 4 | 4 | 4 | 22.143 ms | 24.837 ms | 1.07× |
| prefill 32 | 8 | 8 | 8 | 17.221 ms | 19.494 ms | 1.38× |
| prefill 32 | 12 | 12 | 12 | 16.152 ms | 17.622 ms | 1.47× |
| prefill 32 | 16 | 16 | 16 | 15.729 ms | 16.824 ms | 1.51× |
| prefill 32 | auto | 16 | 16 | **15.647 ms** | 16.534 ms | **1.51×** |
| decode after prefill 10 | 1 | 1 | 0 | **5.899 ms** | 6.581 ms | 1.00× |
| decode after prefill 10 | 2 | 2 | 2 | 6.575 ms | 8.125 ms | 0.90× |
| decode after prefill 10 | 4 | 4 | 4 | 7.150 ms | 9.288 ms | 0.83× |
| decode after prefill 10 | 8 | 8 | 8 | 9.101 ms | 9.820 ms | 0.65× |
| decode after prefill 10 | 12 | 12 | 12 | 9.224 ms | 9.800 ms | 0.64× |
| decode after prefill 10 | 16 | 16 | 15 | 9.372 ms | 9.739 ms | 0.63× |
| decode after prefill 10 | auto | 1 | 0 | **6.139 ms** | 6.771 ms | 0.96× |
| decode after prefill 32 | 1 | 1 | 0 | **6.700 ms** | 7.508 ms | 1.00× |
| decode after prefill 32 | 16 | 16 | 16 | 10.039 ms | 11.084 ms | 0.67× |
| decode after prefill 32 | auto | 1 | 0 | **6.724 ms** | 7.477 ms | 1.00× |
| decode chain 12→42 | 1 | 1 | 0 | 7.011 ms | 7.502 ms | 1.00× |
| decode chain 12→42 | 4 | 4 | 0 | 7.180 ms | 8.212 ms | 0.98× |
| decode chain 12→42 | 16 | 16 | 0 | 7.203 ms | 7.808 ms | 0.97× |
| decode chain 12→42 | auto | 1 | 0 | **7.201 ms** | 7.826 ms | 0.97× |
| decode chain 34→64 | 1 | 1 | 0 | 7.576 ms | 8.556 ms | 1.00× |
| decode chain 34→64 | 16 | 16 | 0 | 7.776 ms | 9.018 ms | 0.97× |
| decode chain 34→64 | auto | 1 | 0 | **7.447 ms** | 8.104 ms | 1.02× |

### 7.3 Trained GPT-2 sweep (median of two runs)

`FN3_PERF_ITERS=4`, 2 warmups, both runs under the lock, `loadavg 1.52–2.25`.
Prompts 10 and 13, contexts 10 and 13.

| Workload | Budget | Selected | Eff. max | p50 | p95 | vs 1 thread |
|---|---|---|---|---|---|---|
| prefill 10 | 1 | 1 | 0 | 328.055 ms | 330.951 ms | 1.00× |
| prefill 10 | 2 | 2 | 2 | 273.681 ms | 280.877 ms | 1.20× |
| prefill 10 | 4 | 4 | 4 | 238.667 ms | 244.529 ms | 1.37× |
| prefill 10 | 8 | 8 | 8 | 237.076 ms | 240.260 ms | 1.38× |
| prefill 10 | 12 | 12 | 12 | 238.278 ms | 241.670 ms | 1.38× |
| prefill 10 | 16 | 16 | 16 | 237.139 ms | 240.276 ms | 1.38× |
| prefill 10 | auto | 16 | 16 | **239.111 ms** | 242.375 ms | **1.37×** |
| prefill 13 | 1 | 1 | 0 | 351.902 ms | 352.798 ms | 1.00× |
| prefill 13 | 2 | 2 | 2 | 298.070 ms | 310.119 ms | 1.18× |
| prefill 13 | 4 | 4 | 4 | 261.152 ms | 263.058 ms | 1.35× |
| prefill 13 | 8 | 8 | 8 | 257.698 ms | 271.561 ms | 1.37× |
| prefill 13 | 12 | 12 | 12 | 256.374 ms | 259.472 ms | 1.37× |
| prefill 13 | 16 | 16 | 16 | 259.730 ms | 262.901 ms | 1.35× |
| prefill 13 | auto | 16 | 16 | **261.905 ms** | 266.805 ms | **1.34×** |
| decode after prefill 10 | 1 | 1 | 0 | 135.993 ms | 136.665 ms | 1.00× |
| decode after prefill 10 | 4 | 4 | 4 | **130.353 ms** | 133.068 ms | 1.04× |
| decode after prefill 10 | 16 | 16 | 16 | 132.098 ms | 133.419 ms | 1.03× |
| decode after prefill 10 | auto | 16 | 16 | 132.200 ms | 133.510 ms | 1.03× |
| decode after prefill 13 | 1 | 1 | 0 | 137.132 ms | 138.654 ms | 1.00× |
| decode after prefill 13 | 4 | 4 | 4 | **133.377 ms** | 137.312 ms | 1.03× |
| decode after prefill 13 | auto | 16 | 16 | 134.632 ms | 135.635 ms | 1.02× |
| decode chain 12→16 | 1 | 1 | 0 | 139.859 ms | 142.302 ms | 1.00× |
| decode chain 12→16 | 4 | 4 | 4 | 134.693 ms | 135.657 ms | 1.04× |
| decode chain 12→16 | 8 | 8 | 8 | **134.539 ms** | 139.990 ms | 1.04× |
| decode chain 12→16 | auto | 16 | 9 | 135.731 ms | 136.809 ms | 1.03× |
| decode chain 15→19 | 1 | 1 | 0 | 141.237 ms | 145.016 ms | 1.00× |
| decode chain 15→19 | auto | 16 | 9 | 136.870 ms | 138.110 ms | 1.03× |

### 7.4 Auto against the best fixed budget

`Auto` resolves to the process budget and narrows to one thread only when the
runtime has observed that its decode step cannot use threads (§12.2). Every row
is the median of the same two runs as §7.2–§7.3.

| Model | Workload | 1 thread | Auto | Auto selected | Auto eff. | Best fixed | Best | Auto/best | Auto/1t |
|---|---|---|---|---|---|---|---|---|---|
| tiny | prefill 10 | 13.391 ms | 9.379 ms | 16 | 15 | 16 | 9.459 ms | **0.99×** | 0.70× |
| tiny | prefill 32 | 23.690 ms | 15.647 ms | 16 | 16 | 16 | 15.729 ms | **0.99×** | 0.66× |
| tiny | decode after prefill 10 | 5.899 ms | 6.139 ms | 1 | 0 | 2 | 6.575 ms | **0.93×** | 1.04× |
| tiny | decode after prefill 32 | 6.700 ms | 6.724 ms | 1 | 0 | 2 | 7.392 ms | **0.91×** | 1.00× |
| tiny | decode chain 12→42 | 7.011 ms | 7.201 ms | 1 | 0 | 4 | 7.180 ms | **1.00×** | 1.03× |
| tiny | decode chain 34→64 | 7.576 ms | 7.447 ms | 1 | 0 | 4 | 7.498 ms | **0.99×** | 0.98× |
| trained | prefill 10 | 328.055 ms | 239.111 ms | 16 | 16 | 8 | 237.076 ms | **1.01×** | 0.73× |
| trained | prefill 13 | 351.902 ms | 261.905 ms | 16 | 16 | 12 | 256.374 ms | **1.02×** | 0.74× |
| trained | decode after prefill 10 | 135.993 ms | 132.200 ms | 16 | 16 | 4 | 130.353 ms | **1.01×** | 0.97× |
| trained | decode after prefill 13 | 137.132 ms | 134.632 ms | 16 | 16 | 4 | 133.377 ms | **1.01×** | 0.98× |
| trained | decode chain 12→16 | 139.859 ms | 135.731 ms | 16 | 9 | 8 | 134.539 ms | **1.01×** | 0.97× |
| trained | decode chain 15→19 | 141.237 ms | 136.870 ms | 16 | 9 | 16 | 137.104 ms | **1.00×** | 0.97× |

**Correction (integration review).** An earlier revision quoted
`Fixed(2) = 6.11 ms` for tiny "decode after prefill". That sample is real but it
is the p50 of a **bimodal** distribution (`p95 10.08 ms` on the same 30 samples),
and on a quiet host at 10 iterations the slow mode holds the median: the
integrator reproduced **8.59 ms p50 / 13.45 ms p95** on the same commit
(`71d09de`). The budget-2 cell is therefore marked unstable: treat
**8.59 / 13.45 ms** as its figure of record. No claim in this revision depends on
the 6.11 sample; the adaptive `Auto` of §12 supersedes the whole comparison for
that class, and the table above reports the two-run median with its p95.

**Tolerance used for "close".** An `Auto`/best-fixed p50 ratio inside
`[0.95, 1.05]`, which is narrower than this host's own repeatability: the
two-run p50 interval of a cell is 0.3–3.4% (tiny `decode after prefill 10` at
Auto: 6.139 / 6.158 ms; tiny `prefill 10` at Auto: 9.379 / 9.703 ms), while the
cell-to-cell spread across earlier windows reached 14–20%. Every ratio above is
0.91–1.02, so `Auto` is inside a band narrower than the host's own drift.
`Auto` is never worse than one thread by more than 4% (tiny `decode after
prefill 10`: 6.139 against 5.899 ms), and it is better than one thread on eight
of the twelve rows.

### 7.5 Every plateau, with the profile

**(a) Tiny prefill: 1.32× / 1.49× and not more.** The 10-row `lm_head` has ten
output rows, so row tiling cannot produce more than ten tiles — `eff. max` is
10 at 12 and 16 threads, and 12/16 perform the same. The 32-token prefill
reaches 1.49× at 8 threads and then *falls* to 1.26× at 12/16: every row tile
reads the whole 25.7 MB weight, so the weight traffic is `tiles × 25.7 MB`
unless it stays in the shared 32 MB L3, and past ~8 concurrent tiles the rest of
the step's working set evicts it. The profile at 16 threads shows `transpose`
growing from 4.60 ms (33% of the step) to 4.55 ms while `matmul` drops from
6.33 ms to 2.79 ms — the matmul gain is real (2.3×) but it is only 46% of the
step, so Amdahl caps the total near 1.5×.

**(b) Tiny decode: exactly 1.00×, with zero parallel dispatches.** In a decode
step the only product above the threshold is the one-token `lm_head` at 6.4 M
MACs, and 6.4 M / 4 M gives one tile, i.e. serial. The chain rows show
`eff. max = 0`: nothing in the timed region is parallel, so the step cannot
scale and does not pretend to. Measured in isolation the kernel *does* scale —
1430 µs at one thread against 426 µs at 16 (3.4×) in the pre-threshold window —
but splitting it makes the *step* slower, which is (c).

**(c) Tiny decode after a prefill: the carry-over, and what fixed it.** The
decode kernel is serial (§7.5b), yet the step measured 9.4 ms at 16 threads
against 5.9 ms at one. §10 runs the controlled experiment and shows no cache or
pool mechanism: the transpose is 4.58 ms p50 in every arm, including after a
16-thread prefill and after the same prefill serially. The in-session
attribution was also unstable across windows (7.89 ms in one, 5.28 ms in
another), so the loss is host-state dependent. The policy therefore stops paying
for it: `Auto` narrows to one thread once it has observed that the decode step
cannot use threads (§12.2), which puts the class at **6.14 ms against 5.90 ms at
one thread** and *better* than every fixed setting (best fixed 6.58 ms). The
plateau is gone from `Auto` and remains only for an explicit `Fixed(4..16)`,
which is exactly what an explicit request means.

**(d) Trained prefill: 1.15–1.29×, less than the tiny model's 1.5×.** The
trained step is dominated by two serial memory streams that scale nothing:
`transpose` of the `[50257, 768]` table (154 MB) and `lm_head` (154 MB). Their
share of the step is larger than the tiny model's, so the same parallel fraction
moves the total less. The 13-token prefill's flatness from 8 to 16 threads is the
same L3 effect as (a) with a weight six times larger.

**(e) Trained decode: 1.06–1.08×.** Per-step weights are ~500 MB, so the step is
DRAM-bound: `lm_head` (9 tiles) and the layer projections parallelise, but the
serial transpose is 30–40% of the step and the parallel part is bandwidth-bound,
so adding threads buys a few percent, not a multiple. The concurrent-session row
in §7.6 shows the same ceiling from the other direction.

**(f) 12 threads is not better than 16, and 2 is sometimes best.** The 3700X is
8 cores with SMT; past 8 threads the extra workers are siblings, and the
memory-bound kernels do not gain from them. This is a hardware property, not a
scheduling bug: `eff. max` proves the workers ran.

### 7.6 Concurrent sessions

Two sessions over one `PreparedModel`, each with its own `CpuBackend` at the same
budget, decoding alternately (`iterations` pairs). The process budget is 16, so
the pair shares one pool rather than creating a second.

| Model | Budget | Selected | Observed max concurrent | Process budget | p50 per pair | p50 per session | Allocs per pair | Checksum | loadavg |
|---|---|---|---|---|---|---|---|---|---|
| tiny-gpt2 | 2 | 2 | 0 | 16 | 15.270 ms | 7.635 ms | 14302 | `0xf1fea7005026a1d9` | 2.42 |
| trained-gpt2 | 2 | 2 | 2 | 16 | 285.443 ms | 142.721 ms | 75808 | `0x8e959734801fadfd` | 4.13 |

Two tiny sessions cost 15.27 ms per pair against 6.0–7.1 ms for one session's
single step: the steps interleave on the same cores, and each session sees
7.64 ms. The observed concurrency is 0 because those steps are serial by
threshold (§7.5b) — the sessions share the machine, not the pool. The trained
pair does reach the pool (`max concurrent 2`) and pays 142.7 ms per session
against 134–137 ms alone, i.e. ~5% for sharing. Allocations per pair are exactly
two sessions' worth (14 302 ≈ 2 × 7 151).

### 7.7 The serial/parallel crossover

`MIN_MACS_PER_TASK = 4 << 20` (4 M multiply-accumulates per tile). The endpoints
are measured, not guessed:

| Observation | MACs per tile | Step-level effect |
|---|---|---|
| tiny one-token `lm_head`, 16 tiles (pre-threshold) | 0.40 M | step 6.22 → 9.68 ms (**0.64×**) |
| tiny one-token `lm_head` under the rule (1 tile) | 6.4 M (serial) | step 6.00 ms (1.00×) |
| trained one-token `lm_head`, 9 tiles | 4.3 M | step 145.3 → 134.8 ms (**1.08×**) |
| trained prefill `lm_head`, 10 tiles | 38.6 M | prefill 331.8 → 257.3 ms (**1.29×**) |

The crossover therefore lies between 0.4 M and 4.3 M MACs per tile. 4 M is the
conservative end, because the failure mode — a small op made slower *and* the
cache of a neighbouring op poisoned — is worse than leaving a medium op serial.
The isolated-kernel table in §8 shows the other side of the trade: the one-token
`lm_head` would be 3.0–3.4× faster as a kernel, and is deliberately not split.

---

## 8. Kernel-level results (isolated, same binary)

`KERNEL` rows, 30 iterations each: the same op on the same tensors at one thread
and at the widest budget.

| Case | 1 thread | Widest | Speedup |
|---|---|---|---|
| `lm_head_10x128x50257` | 5455.976 µs | 2004.719 µs (16) | **2.72×** |
| `lm_head_1x128x50257` | 1430.397 µs | 1426.287 µs (16) | 1.00× (serial by threshold) |
| `mlp_fc_10x128x512` | 41.370 µs | 38.210 µs (16) | 1.08× |
| `mlp_proj_10x512x128` | 33.241 µs | 31.990 µs (16) | 1.04× |
| `attn_scores_2x10x10x64` | 2.850 µs | 2.660 µs | 1.07× |
| `attn_values_2x10x10x64` | 1.090 µs | 1.010 µs | 1.08× |
| `gelu_10x512` | 166.411 µs | 167.190 µs | 1.00× |
| `layernorm_10x128` | 29.670 µs | 37.050 µs | 0.80× (serial; run variance) |
| `softmax_2x10x10` | 5.130 µs | 6.370 µs | 0.81× (serial; run variance) |
| `transpose_lm_head_50257x128` | 3725.668 µs | 3724.358 µs | 1.00× (serial by design) |
| `transpose_attn_2x10x64` | 0.830 µs | 0.820 µs | 1.01× |

The two sub-1.0 rows are serial kernels: their 0.80/0.81 is the run-to-run
spread of a 5–30 µs measurement under load, not a slowdown from threading (both
have `parallel_ops == 0`).

### 8.1 Algorithmic change, separated from threading

Three interleaved rounds, base commit (serial by construction) against this
branch at `FN3_CPU_THREADS=1`; medians, same workloads, same harness.

| Workload | Base p50 | Base p95 | After-1t p50 | After-1t p95 | Speedup p50 | Speedup p95 |
|---|---|---|---|---|---|---|
| prefill 10 | 270.757 ms | 295.433 ms | 13.962 ms | 15.702 ms | **19.39×** | 18.81× |
| prefill 32 | 475.008 ms | 539.526 ms | 23.975 ms | 25.578 ms | **19.81×** | 21.09× |
| decode 10 | 188.694 ms | 196.804 ms | 6.148 ms | 6.783 ms | **30.69×** | 29.01× |
| decode 32 | 191.149 ms | 201.457 ms | 6.731 ms | 7.183 ms | **28.40×** | 28.04× |

Per round: prefill 10 base `270.8 / 271.3 / 270.5`, after `14.0 / 14.0 / 13.9`;
decode 10 base `188.7 / 189.1 / 188.2`, after `6.15 / 6.15 / 6.14`. The arms are
stable to a few tenths of a percent, so the ratio is not a lucky round.

| Factor | Where it comes from | Measured |
|---|---|---|
| Algorithmic (SIMD + blocking + transpose) | base → this branch at 1 thread | 19–31× |
| Threading | this branch at 1 thread → best fixed budget | 1.00–1.49× (tiny), 1.02–1.29× (trained) |

No part of the 19–31× is attributed to threads: the comparison arm is the same
binary with the budget fixed at one, and the one-thread path never enters the
pool (`parallel_ops == 0`, asserted).

---

## 9. Correctness

### 9.1 Parity against the committed fixtures

| Model | Tensor | Threads | max_abs | mean_abs | mean_abs bound | Met |
|---|---|---|---|---|---|---|
| tiny-gpt2 | `prefill.p13.last_logits` | 1 and Auto | 5.364e-07 | 1.001e-07 | 2.5e-06 | yes |
| tiny-gpt2 | `decode.step1.logits` | 1 and Auto | 6.109e-07 | 1.024e-07 | 2.5e-06 | yes |
| trained-gpt2 | `prefill.p13.last_logits` | 1 and Auto | 1.640e-04 | 7.273e-05 | 5e-05 | **no** (elementwise bound met) |
| trained-gpt2 | `decode.step1.logits` | 1 and Auto | 9.537e-05 | 1.743e-05 | 5e-05 | yes |

Tiny tolerances `atol 1e-5 / rtol 1e-4 / mean_abs 2.5e-6`; trained `atol 5e-4 /
rtol 1e-4 / mean_abs 5e-5` (`fixtures/parity/README.md`).

The trained prefill's **mean_abs exceeds the fixture's systematic-bias policy**
while every element passes the elementwise bound. This is not caused by this
branch: the base-commit kernels *are* the ascending-`k` scalar reference, and
this branch's kernels are bit-identical to that reference (three tests, §5.1),
so the two produce the same bytes. The difference is fn3's accumulation order
against ORT's for a 12-layer graph, and it is reported as a `PARITY_WARN` rather
than silently asserting or silently skipping. It is flagged to #53/#49 in §13.

### 9.2 Cross-budget equality

Every budget printed the same checksum as the one-thread run on both models
(§7.1) and both parity tensors are checked at one thread *and* at Auto. This is
stronger than "within tolerance": output tiling changes which thread computes a
value, never the order of the additions.

### 9.3 Repository gates

| Gate | Result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo test --lib` | **683 passed, 0 failed** (673 before this branch; +10: 5 `gemm`, 3 `parallel`, 1 `simd`, 1 doc) |
| `cargo test --test fn3_cpu_reference` | **43 passed, 0 failed** |
| `cargo test --test fn3_session` | **8 passed, 0 failed** |
| `cargo test --test fn3_e2e` (CPU half, `FASTNN_FN3_ONNX_MODEL` set) | **5 passed, 0 failed** |
| `cargo test --test fn3_contracts` | **43 passed, 0 failed** |
| `cargo test --test fn3_tensor_storage` | **36 passed, 0 failed** |
| `cargo test --test fn3_planner` | **11 passed, 0 failed** |
| `cargo test --features cli --test fn3_cli` | **6 passed, 0 failed** |
| `cargo clippy --all-targets -- -D warnings` | clean |
| `cargo clippy --features fn3 --all-targets -- -D warnings` | clean |
| `cargo build --no-default-features` | clean |
| `cargo test --release --test fn3_cpu_perf` (twice, both models) | pass, §7 |

---

## 10. Controlled cache-versus-host-state experiment

The integration review asked for the mechanism to be separated from host-state
noise before choosing between a kernel fix and a policy cap. The experiment is
`tests/fn3_cpu_perf.rs::cache_versus_host_state_experiment`: each arm runs a
burst and then times one `[50257, 128] -> [128, 50257]` transpose, 40 samples per
arm, raw samples printed as `DIST` rows. It crosses the two candidate factors —
**memory** (the real 25.7 MB `lm_head` weight versus a sub-megabyte working set)
and **concurrency** (16 threads through the pool versus one) — and adds a
recovery arm.

Window: `flock /tmp/opencode/fn3-measure.lock`, no competing `cargo`/`rustc`,
`loadavg 1.84 1.73 2.25` at the start, `process_budget 16`.

| Arm | Burst (p50) | Transpose p50 | p95 | min | max | mean |
|---|---|---|---|---|---|---|
| `alone` | — | **4.583 ms** | 4.687 | 4.547 | 5.893 | 4.637 |
| `big_16t` — 25.7 MB product, 16 threads | 2.377 ms | **4.639 ms** | 5.478 | 4.509 | 5.706 | 4.827 |
| `big_1t` — same product, 1 thread | 5.710 ms | **4.563 ms** | 5.157 | 4.493 | 6.558 | 4.683 |
| `small_16t` — 640 KB product, 16 threads | 0.375 ms | **4.692 ms** | 6.449 | 4.569 | 8.087 | 5.064 |
| `big_16t_idle5ms` — 25.7 MB product, 16 threads, 5 ms idle | 7.484 ms | **4.748 ms** | 5.720 | 4.519 | 6.183 | 4.912 |
| `prefill_step_16t` — a real 10-token prefill | 11.075 ms | **4.580 ms** | 5.513 | 4.517 | 5.714 | 4.800 |

**Result: the cache hypothesis is refuted.** The transpose is 4.56–4.75 ms p50 in
every arm; the spread across the six arms (0.19 ms) is smaller than the arm-to-arm
p95 spread (4.69–6.45 ms). A 16-thread prefill that leaves 25.7 MB of freshly
written transposed weight in the 32 MB L3 does **not** slow the next transpose,
and 5 ms of idle does not help it either — so neither cache eviction nor pool
wake-up is the mechanism.

### 10.1 In-session attribution, and why it cannot be trusted

`decode_step_attribution_after_prefill` repeats the sweep's exact protocol
(reset, untimed prefill, timed decode) and reports each op's p50 over the steps:

| Window | Budget | Decode step p50 | `transpose` p50 | `lm_head` p50 |
|---|---|---|---|---|
| A | 1 | 7.103 ms | 4.591 ms | 1.636 ms |
| A | 16 | 10.305 ms | 7.886 ms | 1.551 ms |
| B | 1 | 8.174 ms | 5.447 ms | 1.852 ms |
| B | 16 | 8.014 ms | 5.277 ms | 1.825 ms |

The same binary, same protocol, same host: window A attributes the whole 3.2 ms
to `transpose` at 16 threads; window B shows **no difference at all** between one
and sixteen threads. Window A is a 12-sample p50, window B is a 12-sample p50;
neither is an outlier in isolation. That inconsistency is the finding: the
in-session effect is **state-dependent and not reproducible**, which is exactly
what the review asked to rule out before shipping a kernel change. The stable
facts are (i) the kernels themselves are unaffected (§10), and (ii) at four or
more active threads the *measured* decode-after-prefill step was slower in every
window in which it appeared (6.0–7.4 ms at one thread against 6.9–10.9 ms at
four or more).

## 11. Row-tiled versus column-tiled prefill

The prefill's output-row tiling gave every task the whole right-hand side, so a
ten-row `lm_head` read its 25.7 MB weight ten times per instruction. The
column-tiled version gives each task a full-height band, so the weight is read
once and each task writes `rows x band` into its own scratch, which the parent
scatters (one extra pass over the output). Per-element accumulation order is
unchanged, and `gemm::tests::every_shape_matches_the_reference_at_every_thread_count`
checks 15 shapes at 1/2/3/4/8/16 threads bit-for-bit.

Both binaries are built from the same tree with the same harness, one from
`71d09de` (row) and one from this branch (column), and the runs alternate by
round. Medians over two interleaved rounds; the window's `loadavg` was 3.5–6.4,
so read the ratios, not the absolutes.

| Workload | Budget | Row p50 | Column p50 | Column / row |
|---|---|---|---|---|
| tiny prefill 10 | 1 | 15.564 ms | 14.187 ms | 0.91× |
| tiny prefill 10 | 4 | 14.160 ms | 12.818 ms | 0.91× |
| tiny prefill 10 | 16 | 12.135 ms | 11.010 ms | 0.91× |
| tiny prefill 10 | auto | 11.445 ms | 10.983 ms | 0.96× |
| tiny prefill 32 | 8 | 18.116 ms | 19.873 ms | 1.10× |
| tiny prefill 32 | 16 | 22.990 ms | 18.657 ms | **0.81×** |
| tiny prefill 32 | auto | 23.655 ms | 18.293 ms | **0.77×** |
| trained prefill 10 | 8 | 298.465 ms | 256.443 ms | **0.86×** |
| trained prefill 10 | auto | 317.252 ms | 269.943 ms | **0.85×** |
| trained prefill 13 | 8 | 332.491 ms | 281.163 ms | **0.85×** |
| trained prefill 13 | auto | 356.092 ms | 299.575 ms | **0.84×** |
| tiny decode after prefill 10 | 1 | 7.429 ms | 6.183 ms | 0.83× |
| tiny decode after prefill 10 | 16 | 9.437 ms | 8.281 ms | 0.88× |
| tiny decode after prefill 10 | auto | 9.310 ms | 8.474 ms | 0.91× |
| tiny decode chain 32→62 | 16 | 8.260 ms | 7.648 ms | 0.93× |
| trained decode after prefill 10 | 16 | 139.673 ms | 142.138 ms | 1.02× |
| trained decode chain 16→20 | 16 | 145.673 ms | 145.620 ms | 1.00× |

Column tiling is retained: it is 9–16% faster on the trained prefill, 9–23% on
the tiny 32-token prefill, and 9% on the tiny 10-token prefill at the wide
budgets. The two rows where the row build wins (`tiny prefill 32` at 2 and 4
threads, 1.28×/1.24×) are the cases where the column build's per-task scratch
and scatter (2 MB per instruction) dominate a short instruction; they do not
survive at the budgets Auto actually uses. Decode is 2–17% better too, which is
consistent with the cheaper prefill leaving less state behind, but it does **not**
remove the carry-over (§12).

## 12. Auto policy: adaptive narrowing, not a cap

### 12.1 What the experiment ruled out

§10 shows the decode-after regression is not caused by the transposed weight
evicting the cache, by pool wake-ups, or by the transpose kernel at all: the
transpose is 4.58 ms p50 in every arm and the 5 ms-idle arm does not recover
anything, while the in-session attribution flips between windows. What is
reproducible is the *workload-level* fact: at four or more active threads the
decode-after step was 6.9–10.9 ms in every window in which it appeared, against
5.9–7.4 ms at one thread, and the prefill that precedes it gains only 1.3–1.5×.
For a turn, the pair is a net loss at a wide budget and a net win at a narrow
one. A fixed cap cannot express that, because a *prefill-only* workload has the
opposite optimum (§7.2: 16 threads is 1.43× better than one).

### 12.2 The rule

`Auto` resolves to the process budget and narrows to **one thread** once the
runtime has observed that its decode step cannot use threads:

* every matrix product with one output row is reported to the backend with its
  multiply-accumulate count;
* if that product is below `MIN_MACS_PER_TASK` (so the kernel would give it one
  tile) the runtime records a *serial decode*;
* if any such product is above it, the runtime records a *parallel decode* and
  the wide budget stays, because the step itself can use threads;
* `threads()` returns one when a serial decode has been seen and no parallel
  decode has.

That is a property of the observed workload, not a magic constant: the tiny
model's decode has only serial one-token projections (its `lm_head` is 6.4 M
MACs, one tile), so it narrows; the trained model's `lm_head` is 38.6 M MACs
(nine tiles), so it keeps the wide budget. `ThreadBudget::Fixed(n)` is never
narrowed — an explicit request is a statement about the workload.

The first turn of a session still runs wide, because nothing has been observed
yet. In the measured protocols the effect of that first turn is one sample out of
30 (tiny) or 4 (trained).

### 12.3 Result

| Model | Workload | 1 thread | Auto | Best fixed | Auto/best | Auto/1t |
|---|---|---|---|---|---|---|
| tiny | prefill 10 | 13.391 ms | 9.379 ms | 9.459 ms (16) | 0.99× | 0.70× |
| tiny | prefill 32 | 23.690 ms | 15.647 ms | 15.729 ms (16) | 0.99× | 0.66× |
| tiny | decode after prefill 10 | 5.899 ms | 6.139 ms | 6.575 ms (2) | 0.93× | 1.04× |
| tiny | decode after prefill 32 | 6.700 ms | 6.724 ms | 7.392 ms (2) | 0.91× | 1.00× |
| tiny | decode chain 12→42 | 7.011 ms | 7.201 ms | 7.180 ms (4) | 1.00× | 1.03× |
| tiny | decode chain 34→64 | 7.576 ms | 7.447 ms | 7.498 ms (4) | 0.99× | 0.98× |
| trained | prefill 10 | 328.055 ms | 239.111 ms | 237.076 ms (8) | 1.01× | 0.73× |
| trained | prefill 13 | 351.902 ms | 261.905 ms | 256.374 ms (12) | 1.02× | 0.74× |
| trained | decode after prefill 10 | 135.993 ms | 132.200 ms | 130.353 ms (4) | 1.01× | 0.97× |
| trained | decode after prefill 13 | 137.132 ms | 134.632 ms | 133.377 ms (4) | 1.01× | 0.98× |
| trained | decode chain 12→16 | 139.859 ms | 135.731 ms | 134.539 ms (8) | 1.01× | 0.97× |
| trained | decode chain 15→19 | 141.237 ms | 136.870 ms | 137.104 ms (16) | 1.00× | 0.97× |

Tolerance: `Auto`/best in `[0.95, 1.05]`, which is tighter than the host's
two-run repeatability on these cells (0.3–3.4%). The worst ratio is 1.02.

### 12.4 The trade-off that remains

* An explicit `Fixed(4..16)` still shows the decode-after penalty (§7.2). That
  is the price of a wide prefill, and it is now an explicit choice rather than
  `Auto`'s default.
* A *latency*-critical caller that interleaves prefill and decode should use
  `Fixed(1)`: it is within 4% of `Auto` on every decode class and it removes the
  first-turn cost entirely.
* `Auto` cannot know that a prefill will be followed by a decode *before* it has
  seen one; a session-level turn hook would remove that first-turn cost. It
  belongs to the session API (`src/fn3/model/**`, #72/#54), not here.

## 13. Candidates evaluated and rejected

| Candidate | Why not retained |
|---|---|
| FMA / multiple accumulators in the dot product (the 2.x `simd_dot_product` shape) | Reassociates the sum and would change every logit's last bits. The contract allows that only by publishing a new tolerance; vectorising across outputs gets the same throughput without it. |
| Tiled transpose (32×32, 64×64, 4×4 blocks, rectangles) | Measured 1.5–3× slower than the source-sequential loop on the real buffer (§4.1). |
| Threading the transpose | The only safe partition is by destination rows, which forces the strided-read pattern: 25.3 ms at one thread and 12.5 ms at eight, against 2.9 ms serial for the source-sequential loop. Splitting the fast form needs two threads to write disjoint elements of one `&mut [f32]`, i.e. `unsafe` aliasing, and the measured ceiling is ~2× on a 51 MB stream. Rejected. |
| Splitting the contraction (`k`) dimension | Requires a reduction, which changes the accumulation order. Only the output is partitioned. |
| Parallelising the one-token `lm_head` on the tiny model | Kernel 3.0–3.4× faster, step 0.60–0.64× slower (§7.5c). The threshold keeps it serial. |
| Plan-level constant folding of the tied-embedding transpose | The structural fix — 25.7 MB (tiny) / 154 MB (trained) of per-step traffic that should be zero. The instruction lives in the plan, which `src/fn3/model/**` owns, not this issue. §13. |
| 8-row blocking for `lm_head` (one pass over the weight instead of three) | Measured gap between `lm_head_1x…` (1430 µs) and `lm_head_10x…` (5456 µs) implies ~3 ms on the tiny prefill; not retained because it moves only prefill and would need a full re-gate. Named with its measurement in §13. |
| `LayerNorm` kernel work | The pinned graphs never issue `LayerNorm`; the importer leaves the ReduceMean/Sub/Pow/Div decomposition, which is 1.4% + 1.0% + 1.0% + 0.97% of a tiny prefill after §4–§5. |
| Elementwise broadcast fast paths | The general `Locator` path is <1% of a step after the two retained changes. |
| Rayon's adaptive splitting | Chunks/tiles are explicit, so concurrency is `min(budget, tiles)` rather than whatever the scheduler decides. |
| `openblas` / `gemm` crate for the fp32 matmul | Both would change the accumulation order; the `gemm` crate is already a dependency but is quantized-path infrastructure. Not needed: the hand-written kernel reaches 2.7–3.0× with tiling alone and is bit-exact. |

---

## 14. Skips and blockers

| Item | Status | Exact reason |
|---|---|---|
| `perf record` / `perf stat` | **skipped** | `perf` is not installed (`zsh: command not found: perf`; `/usr/bin` has only `perfquery`). `perf_event_paranoid=2` would allow user-space sampling, so the blocker is the package, not policy. Internal timers used instead (§2.3). |
| arm64 | **skipped** | No arm64 hardware. `rustup target list --installed` has no `aarch64` target, and there is no `qemu-aarch64`, no `qemu-aarch64-static` and no `aarch64-linux-gnu-gcc`, so it could not be cross-built either. No arm64 claim is made. |
| Base-commit trained-model "before" | **not measured** | The base-commit harness only loads the tiny model (its gate asserts the tiny digest), and the trained model's fixtures are not wired into it. Reported as such rather than estimated. |
| `#69` budget integration | **interface only** | #69 is not dispatched; §6 documents the interface for it. |

**Blockers encountered:** the shared `fn3-measure.lock` was held by other FN3
workers for long stretches (up to ~20 minutes at a time); every window waited
rather than measuring alongside a build, and the guard aborted two starts that
found a competing `cargo` still alive. One earlier sweep was superseded when the
decode protocol was found to be contaminated (§7.5c) and one when the row
partition was found unbalanced; both are kept under `/tmp/opencode/fn3-71/` and
are quoted only where still valid.

---

## 15. Commands

Every authoritative window:

```bash
flock -w 7200 /tmp/opencode/fn3-measure.lock \
  -c '/tmp/opencode/fn3-71/authoritative.sh'
```

Inside the window (`authoritative.sh`), in order:

```bash
pgrep -a 'cargo|rustc'                                   # refuse to start under a build
CARGO_BUILD_JOBS=4 cargo test --release --test fn3_cpu_perf --no-run
FASTNN_FN3_ONNX_MODEL=<tiny>     FN3_PERF_VARIANT=after FN3_PERF_ITERS=30 FN3_PERF_WARMUP=3 \
  ./cpu_perf_after --nocapture --test-threads=1
FASTNN_FN3_ONNX_MODEL=<trained>  FN3_PERF_VARIANT=after \
FN3_PERF_PROMPTS=10 FN3_PERF_DECODE_PAST=10 FN3_PERF_ITERS=4 FN3_PERF_WARMUP=2 \
  ./cpu_perf_after --nocapture --test-threads=1
# 3 interleaved rounds: base (serial) vs this branch at FN3_CPU_THREADS=1
./cpu_perf_base --nocapture --test-threads=1 \
  --exact cpu_prefill_and_decode_are_measured_with_per_op_attribution
FN3_CPU_THREADS=1 FN3_PERF_THREADS=1 ./cpu_perf_after --nocapture --test-threads=1 \
  --exact thread_scaling_sweep
rustc -O -o transpose-shapes transpose-shapes.rs && ./transpose-shapes
rustc -O -o transpose-threads transpose-threads.rs && ./transpose-threads
```

Second trained length (a short window of its own):

```bash
flock -w 3600 /tmp/opencode/fn3-measure.lock -c '/tmp/opencode/fn3-71/trained-lengths.sh'
```

Gates: `/tmp/opencode/fn3-71/gates.sh` (fmt, lib, the seven integration suites,
clippy twice, `--no-default-features`), logs in `/tmp/opencode/fn3-71/gates/`.

The budget can be set from the environment (`FASTNN_CPU_THREADS=1`) or per
runtime (`CpuBackend::with_thread_budget(ThreadBudget::fixed(n))`).

---

## 16. Findings for other owners

1. **The tied-embedding transpose is still 25.7 MB of traffic per step** (tiny)
   and ~154 MB (trained). A plan-level constant-folding pass that transposes an
   immutable weight once at prepare time would remove it entirely — on the tiny
   model that is ~4.5 ms of a 13.9 ms prefill and ~3.7 ms of a 6.1 ms decode, and
   it is also the reason decode cannot scale (§7.5c). That is
   `src/fn3/model/**` (owner #54/#56/#72), not this issue. The kernel is now as
   fast as its memory traffic allows; the remaining win is structural.
2. **The trained model's FP32 logits miss the fixture's `mean_abs` policy**
   (7.27e-5 against 5e-5) on the 13-token prefill while every element passes
   `atol/rtol`. This branch is bit-identical to the scalar reference, so it is
   pre-existing; it belongs to #53 (reference) / #49 (fixture policy).
3. **`lm_head` reads the whole weight once per four output rows.** An 8-row
   block would halve that on prefill (the measured `M=1` vs `M=10` ratio implies
   ~3 ms on the tiny prefill). Not retained here because it moves only prefill
   and would need a full re-gate; recorded with its measurement.
4. **Turn-level threading advice, measured:** on the tiny model a turn is faster
   at `Fixed(2)` than at `Auto` because a 16-thread prefill slows the following
   decode (§7.5c). If a future session API knows it is about to decode, capping
   the prefill is worth ~4 ms per turn on this host.
5. **For #69:** reuse `ThreadBudget`, `parallel_stats`, `reset_parallel_stats`
   and `MIN_MACS_PER_TASK` unchanged; the quantized GEMV shapes will need their
   own per-task threshold measured the same way (integer GEMV is cheaper per
   MAC, so the crossover will sit higher).
