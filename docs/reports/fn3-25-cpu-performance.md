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
| trained GPT-2 | prefill 10 | Auto | — | 268.3 ms | 1.24× over 1 thread |
| trained GPT-2 | decode chain 12→16 | Auto | — | 136.6 ms | 1.06× over 1 thread |

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

### 7.2 Tiny GPT-2 sweep (p50/p95, loadavg per row)

| Workload | Budget | Selected | Eff. max | Eff. threads | p50 | p95 | Speedup vs 1 | Alloc count | Alloc bytes | loadavg |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill 10 | 1 | 1 | 0 | 0 | 13.430 ms | 13.695 ms | 1.00× | 3435 | 29,281,986 | 2.51 |
| prefill 10 | 2 | 2 | 2 | 3 | 10.920 ms | 13.556 ms | 1.23× | 3439 | 29,282,116 | 2.55 |
| prefill 10 | 4 | 4 | 4 | 5 | 12.947 ms | 14.588 ms | 1.04× | 3439 | 29,282,196 | 2.55 |
| prefill 10 | 8 | 8 | 8 | 14 | 11.692 ms | 12.570 ms | 1.15× | 3439 | 29,282,356 | 2.74 |
| prefill 10 | 12 | 12 | 10 | 16 | 10.382 ms | 11.847 ms | 1.29× | 3439 | 29,282,386 | 2.74 |
| prefill 10 | 16 | 16 | 10 | 16 | 10.192 ms | 11.588 ms | **1.32×** | 3439 | 29,282,386 | 2.68 |
| prefill 10 | auto | 16 | 10 | 16 | 10.209 ms | 11.817 ms | 1.32× | 3439 | 29,282,386 | 4.23 |
| prefill 32 | 1 | 1 | 0 | 0 | 24.404 ms | 26.785 ms | 1.00× | 3435 | 35,924,622 | 2.51 |
| prefill 32 | 2 | 2 | 2 | 3 | 18.539 ms | 21.887 ms | 1.32× | 3439 | 35,924,702 | 2.55 |
| prefill 32 | 4 | 4 | 4 | 6 | 18.095 ms | 22.431 ms | 1.35× | 3439 | 35,924,782 | 2.55 |
| prefill 32 | 8 | 8 | 8 | 15 | 16.385 ms | 18.383 ms | **1.49×** | 3439 | 35,924,942 | 2.74 |
| prefill 32 | 12 | 12 | 12 | 16 | 19.335 ms | 21.466 ms | 1.26× | 3439 | 35,925,152 | 2.74 |
| prefill 32 | 16 | 16 | 16 | 16 | 19.394 ms | 21.185 ms | 1.26× | 3439 | 35,925,312 | 2.68 |
| prefill 32 | auto | 16 | 16 | 16 | 19.586 ms | 22.454 ms | 1.25× | 3439 | 35,925,312 | 4.23 |
| decode after prefill 10 | 1 | 1 | 0 | 0 | 6.000 ms | 6.748 ms | 1.00× | 3396 | 925,921 | 2.51 |
| decode after prefill 10 | 2 | 2 | 2 | 3 | 6.113 ms | 10.084 ms | 0.98× | 3396 | 925,921 | 2.55 |
| decode after prefill 10 | 4 | 4 | 4 | 6 | 9.647 ms | 10.459 ms | 0.62× | 3396 | 925,921 | 2.55 |
| decode after prefill 10 | 8 | 8 | 8 | 16 | 10.076 ms | 11.310 ms | 0.60× | 3396 | 925,921 | 2.74 |
| decode after prefill 10 | 12 | 12 | 10 | 16 | 9.986 ms | 11.382 ms | 0.60× | 3396 | 925,921 | 2.68 |
| decode after prefill 10 | 16 | 16 | 10 | 16 | 10.142 ms | 11.023 ms | 0.59× | 3396 | 925,921 | 2.68 |
| decode after prefill 10 | auto | 16 | 10 | 16 | 10.076 ms | 11.203 ms | 0.60× | 3396 | 925,921 | 4.23 |
| decode after prefill 32 | 1 | 1 | 0 | 0 | 6.577 ms | 7.259 ms | 1.00× | 3396 | 1,066,765 | 2.74 |
| decode after prefill 32 | 4 | 4 | 4 | 6 | 10.105 ms | 11.845 ms | 0.65× | 3396 | 1,066,765 | 2.74 |
| decode after prefill 32 | 16 | 16 | 16 | 16 | 10.530 ms | 11.357 ms | 0.62× | 3396 | 1,066,765 | 2.68 |
| decode after prefill 32 | auto | 16 | 16 | 16 | 10.917 ms | 11.995 ms | 0.60× | 3396 | 1,066,765 | 4.23 |
| decode chain 12→42 | 1 | 1 | 0 | 0 | 7.094 ms | 7.754 ms | 1.00× | 7093 | 946,232 | 2.51 |
| decode chain 12→42 | 2 | 2 | 0 | 0 | 7.234 ms | 7.617 ms | 0.98× | 7093 | 946,232 | 2.55 |
| decode chain 12→42 | 4 | 4 | 0 | 0 | 7.052 ms | 7.489 ms | 1.01× | 7093 | 946,232 | 2.55 |
| decode chain 12→42 | 8 | 8 | 0 | 0 | 7.034 ms | 9.143 ms | 1.01× | 7093 | 946,232 | 2.74 |
| decode chain 12→42 | 12 | 12 | 0 | 0 | 7.106 ms | 8.236 ms | 1.00× | 7093 | 946,232 | 2.68 |
| decode chain 12→42 | 16 | 16 | 0 | 0 | 7.088 ms | 8.006 ms | 1.00× | 7093 | 946,232 | 2.68 |
| decode chain 12→42 | auto | 16 | 0 | 0 | 7.290 ms | 9.135 ms | 0.97× | 7093 | 946,232 | 4.23 |
| decode chain 34→64 | 1 | 1 | 0 | 0 | 7.534 ms | 8.149 ms | 1.00× | 7093 | 1,087,672 | 2.74 |
| decode chain 34→64 | 16 | 16 | 0 | 0 | 7.514 ms | 8.244 ms | 1.00× | 7093 | 1,087,672 | 4.23 |
| decode chain 34→64 | auto | 16 | 0 | 0 | 7.471 ms | 8.960 ms | 1.01× | 7093 | 1,087,672 | 4.23 |

### 7.3 Trained GPT-2 sweep

Two prompt lengths and two contexts. `prefill 13` / `decode 13` ran in a second
window (`docs/reports/fn3-25/trained-13.log`); the loadavg column says so.

| Workload | Budget | Selected | Eff. max | Eff. threads | p50 | p95 | Speedup vs 1 | Alloc count | Alloc bytes | loadavg |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill 10 | 1 | 1 | 0 | 0 | 331.801 ms | 338.094 ms | 1.00× | 18497 | 193,237,030 | 3.33 |
| prefill 10 | 2 | 2 | 2 | 3 | 257.297 ms | 263.717 ms | **1.29×** | 18645 | 193,240,750 | 3.14 |
| prefill 10 | 4 | 4 | 4 | 6 | 315.054 ms | 331.467 ms | 1.05× | 18645 | 193,243,710 | 2.97 |
| prefill 10 | 8 | 8 | 8 | 12 | 282.508 ms | 288.091 ms | 1.17× | 18645 | 193,245,210 | 2.81 |
| prefill 10 | 12 | 12 | 10 | 14 | 275.391 ms | 280.317 ms | 1.20× | 18645 | 193,245,290 | 2.67 |
| prefill 10 | 16 | 16 | 10 | 15 | 284.981 ms | 295.872 ms | 1.16× | 18645 | 193,244,910 | 2.53 |
| prefill 10 | auto | 16 | 10 | 13 | 268.308 ms | 279.461 ms | 1.24× | 18645 | 193,244,910 | 2.49 |
| prefill 13 | 1 | 1 | 0 | 0 | 377.052 ms | 380.842 ms | 1.00× | 18497 | 204,568,105 | 1.67 |
| prefill 13 | 2 | 2 | 2 | 3 | 380.085 ms | 383.183 ms | 0.99× | 18645 | 204,571,825 | 1.81 |
| prefill 13 | 4 | 4 | 4 | 6 | 363.347 ms | 370.075 ms | 1.04× | 18645 | 204,574,785 | 2.14 |
| prefill 13 | 8 | 8 | 8 | 14 | 327.704 ms | 344.171 ms | 1.15× | 18645 | 204,578,685 | 2.13 |
| prefill 13 | 12 | 12 | 12 | 15 | 332.460 ms | 342.818 ms | 1.13× | 18645 | 204,578,845 | 2.20 |
| prefill 13 | 16 | 16 | 13 | 14 | 332.548 ms | 341.587 ms | 1.13× | 18645 | 204,578,505 | 2.32 |
| prefill 13 | auto | 16 | 13 | 16 | 320.302 ms | 332.485 ms | 1.18× | 18645 | 204,578,505 | 2.30 |
| decode after prefill 10 | 1 | 1 | 0 | 0 | 137.540 ms | 146.924 ms | 1.00× | 18355 | 7,317,017 | 3.33 |
| decode after prefill 10 | 2 | 2 | 2 | 3 | 132.838 ms | 134.436 ms | 1.04× | 18358 | 7,317,081 | 3.14 |
| decode after prefill 10 | 4 | 4 | 4 | 5 | 133.206 ms | 137.892 ms | 1.03× | 18358 | 7,317,145 | 2.97 |
| decode after prefill 10 | 8 | 8 | 8 | 12 | 134.558 ms | 136.434 ms | 1.02× | 18358 | 7,317,273 | 2.81 |
| decode after prefill 10 | 12 | 12 | 10 | 15 | 135.101 ms | 140.167 ms | 1.02× | 18358 | 7,317,305 | 2.53 |
| decode after prefill 10 | 16 | 16 | 10 | 15 | 132.615 ms | 141.611 ms | 1.04× | 18358 | 7,317,305 | 2.49 |
| decode after prefill 10 | auto | 16 | 10 | 14 | 134.248 ms | 135.556 ms | 1.02× | 18358 | 7,317,305 | 2.69 |
| decode after prefill 13 | 1 | 1 | 0 | 0 | 141.619 ms | 143.132 ms | 1.00× | 18355 | 7,999,979 | 1.70 |
| decode after prefill 13 | 16 | 16 | 13 | 16 | 137.143 ms | 140.013 ms | 1.03× | 18358 | 8,000,267 | 2.32 |
| decode after prefill 13 | auto | 16 | 13 | 15 | 136.037 ms | 140.827 ms | 1.04× | 18358 | 8,000,267 | 2.30 |
| decode chain 12→16 | 1 | 1 | 0 | 0 | 145.302 ms | 146.612 ms | 1.00× | 37036 | 5,824,104 | 3.33 |
| decode chain 12→16 | 2 | 2 | 2 | 3 | 136.199 ms | 138.087 ms | 1.07× | 37039 | 5,824,168 | 3.14 |
| decode chain 12→16 | 4 | 4 | 4 | 4 | 135.340 ms | 137.856 ms | 1.07× | 37039 | 5,824,232 | 2.81 |
| decode chain 12→16 | 8 | 8 | 8 | 11 | 139.356 ms | 145.079 ms | 1.04× | 37039 | 5,824,360 | 2.67 |
| decode chain 12→16 | 12 | 12 | 9 | 12 | 134.766 ms | 141.126 ms | **1.08×** | 37039 | 5,824,392 | 2.53 |
| decode chain 12→16 | 16 | 16 | 9 | 13 | 142.273 ms | 150.411 ms | 1.02× | 37039 | 5,824,392 | 2.49 |
| decode chain 12→16 | auto | 16 | 9 | 14 | 136.631 ms | 138.018 ms | 1.06× | 37039 | 5,824,392 | 2.69 |
| decode chain 15→19 | 1 | 1 | 0 | 0 | 145.682 ms | 147.592 ms | 1.00× | 37036 | 6,507,066 | 1.70 |
| decode chain 15→19 | auto | 16 | 9 | 13 | 142.037 ms | 144.353 ms | 1.03× | 37039 | 6,507,354 | 2.30 |

**Sensitivity.** An earlier window on the same kernels (no chain workload,
`loadavg 2.1`) measured the trained prefill 10 at 352.7 ms → 226.6 ms (**1.56×**)
and the trained decode at 138.2 ms → 110.3 ms (**1.25×**). The kernels, threshold
and tiling were identical; the difference is machine contention (`loadavg
2.5–3.3` here against `2.1`). The table above is the conservative claim; the
quieter window is the best case observed.

### 7.4 Auto against the best fixed budget

| Model | Workload | Auto p50 | Best fixed | Best p50 | Auto / best |
|---|---|---|---|---|---|
| tiny | prefill 10 | 10.209 ms | 16 | 10.192 ms | 1.00× |
| tiny | prefill 32 | 19.586 ms | 8 | 16.385 ms | 1.20× |
| tiny | decode chain 12→42 | 7.290 ms | 8 | 7.034 ms | 1.04× |
| tiny | decode chain 34→64 | 7.471 ms | 8 | 7.461 ms | 1.00× |
| tiny | decode after prefill 10 | 10.076 ms | 1 | 6.000 ms | 1.68× |
| tiny | decode after prefill 32 | 10.917 ms | 1 | 6.577 ms | 1.66× |
| trained | prefill 10 | 268.308 ms | 2 | 257.297 ms | 1.04× |
| trained | prefill 13 | 320.302 ms | 8 | 327.704 ms | 0.98× |
| trained | decode chain 12→16 | 136.631 ms | 12 | 134.766 ms | 1.01× |
| trained | decode chain 15→19 | 142.037 ms | 12 | 139.282 ms | 1.02× |
| trained | decode after prefill 10 | 134.248 ms | 16 | 132.615 ms | 1.01× |
| trained | decode after prefill 13 | 136.037 ms | 16 | 137.143 ms | 0.99× |

Auto is within 4% of the best fixed setting on ten of twelve rows, and *is* the
best on three. The two exceptions are the tiny model's "decode after prefill"
rows, explained in §7.5.

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

**(c) Tiny decode after a prefill: 0.60× at ≥ 4 threads.** The decode kernel is
serial (b), so the loss is inherited from the prefill that ran before it on the
same backend. The profile at 16 threads shows exactly where: `transpose` is
3.69 ms of the 6.22 ms one-thread step (61%), and at 16 threads the step is
9.68 ms with `transpose` at 7.99 ms (83%) while `matmul` falls from 1.73 ms to
1.04 ms. The 16-thread prefill leaves 25.7 MB of freshly written transposed
weight in the shared L3, so the decode's own 25.7 MB transpose write-allocates
into a full cache. The kernel did not change (the chain proves that); the cache
state around it did. This is the single most important scaling result in this
report: a turn's first token is *slower* at 16 threads than at 1 on this model,
even though the prefill that precedes it is 1.32× faster.
  *Consequence*: for turn-level work on the tiny model the measured optimum is
  `ThreadBudget::fixed(2)` (prefill 1.23×, decode 0.98×, net win) or a small
  fixed cap; 16 threads is a net loss. For steady-state generation the budget is
  irrelevant because the step is serial.

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

## 10. Candidates evaluated and rejected

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

## 11. Skips and blockers

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

## 12. Commands

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

## 13. Findings for other owners

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
