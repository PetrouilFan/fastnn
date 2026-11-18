# FN3-04 reproducible benchmark and profiling harness

Status: schema v1, baseline (2.x) adapter working, fn3 (3.0) adapter seam failing
loudly by design. Worktree `fn3/50-bench`, base commit
`a2c0759fff8bdcb7c943e53442f3382b86e59b66`.

The harness measures the runtime through public APIs only. It never edits runtime
state, never synthesises results for missing capabilities, and records raw
samples so any summary can be recomputed.

## Commands

The default feature set includes `python`, which builds PyO3 as an
`extension-module` and prevents linking standalone binaries. Build with
`--no-default-features` and re-add the CPU features explicitly so the measured
configuration matches the default CPU runtime:

```bash
CARGO_BUILD_JOBS=4 cargo build --release --no-default-features \
  --features simd,parallel,prepared-plan,fusion-forward --bin fn3-bench
```

Run a named suite and write schema-v1 JSON plus a human summary:

```bash
# mandatory smoke path: tiny fixture + GPT-2-shaped projections
./target/release/fn3-bench run --suite smoke --quick --out /tmp/smoke.json

# GPT-2 projection microbenchmarks only
./target/release/fn3-bench run --suite projections --quick --out /tmp/proj.json

# synthetic GPT-2 decoder stack: load/prepare/prefill/decode/memory/allocations
./target/release/fn3-bench run --suite model --quick --out /tmp/model.json

# wrap the existing Criterion benches (bench-profile build, slow)
./target/release/fn3-bench run --suite criterion --out /tmp/criterion.json

# import an existing target/criterion tree instead of re-running the benches
./target/release/fn3-bench run --suite criterion --reuse-criterion --out /tmp/criterion.json

# machine-readable output on stdout
./target/release/fn3-bench run --suite model --quick --json-stdout > /tmp/model.json

# compare two runs (refuses hard environment/geometry mismatches)
./target/release/fn3-bench compare a.json b.json
./target/release/fn3-bench compare a.json b.json --stat p95 --json
./target/release/fn3-bench compare a.json b.json --force

./target/release/fn3-bench list-suites
```

Determinism controls: `--seed` (default `20260818`), `--warmup`, `--samples`,
`--prompt-tokens`, `--layers`, `--decode-tokens`, `--threads` (sets
`RAYON_NUM_THREADS` before the pool is created). `--quick` selects small
parameters and sample counts; every suite is runnable without it, but the full
`model` suite takes minutes because 2.x's packed f32 projection path is slow at
GPT-2 prefill sizes.

## Result schema (version 1)

Companion JSON Schema: [`result-schema.json`](result-schema.json). `schema_version`
is `1`. Additive fields keep the version; breaking changes bump it. Consumers
must tolerate unknown fields.

| Group | Fields |
|---|---|
| `run` | `run_id`, `timestamp_utc`, `started_unix_s`, `duration_s`, `adapter`, `suite`, `quick`, `seed`, `label` |
| `git` | `commit`, `branch`, `dirty`, `dirty_entries`, `describe` |
| `build` | `package_version`, `profile`, `debug_assertions`, `rustc_version`, `cargo_version`, `features`, `target_arch`, `target_os`, `target_family`, `command_line` |
| `host` | `os`, `kernel`, `arch`, `cpu_model`, `physical_cores`, `logical_cores`, `threads_used`, `memory_total_bytes`, `load.*`, `competing_build_pids`, `competing_benchmark_pids` |
| `gpu` | `status`, `reason`, `vulkan_instance_version`, `devices[]` (`name`, `vendor`, `vendor_id`, `device_id`, `device_type` (physical/software/unknown), `software_icd`, `driver_name`, `driver_info`, `driver_version`, `api_version`, `device_uuid`), `selected_device_index`, `execution_backend`, `execution_reason` |
| `model` | `name`, `revision`, `source`, `parameter_count`, `weights_bytes`, `seed`, `checksums[]` (`subject`, `algorithm`, `value`) |
| `geometry` | `batch`, `prompt_tokens`, `decode_tokens`, `context_capacity`, `hidden_size`, `num_heads`, `num_kv_heads`, `head_dim`, `num_layers`, `intermediate_size`, `vocab_size`, `attention`, `kv_cache` |
| `precision` | `weights_dtype`, `activations_dtype`, `accumulation_dtype`, `quantization` |
| `metrics[]` | `name`, `category`, `unit`, `status`, `reason`, `clock`, `host_time`, `device_time`, `counters`, `metadata` |
| top level | `notes` (methodology), `warnings` (guards/capture failures) |

Per-metric blocks:

- `host_time` is the primary measured series: host monotonic wall-clock
  milliseconds for latency metrics, the measured quantity (`MiB`, `count`) for
  memory/allocation metrics. `clock` states which: `host_monotonic`, `derived`,
  or `not_applicable`.
- `host_time.samples` contains every finite raw sample. `sample_count`,
  `warmup_count` and `repetition_count` are explicit. `stats` is the documented
  percentile set: `min`, `p50`, `p95`, `p99`, `max`, `mean`, sample `stddev`.
  Percentiles use linear interpolation over sorted samples (R-7 / NumPy
  default), so `p50` equals the median.
- `device_time` always exists. When the backend reports device clocks it has
  `status: "ok"`, `source`, `samples`, `stats`; otherwise it is
  `status: "unavailable"` with a `reason` and empty samples.
- `counters` holds runtime counters summed over the measured samples.
- `metadata` carries shapes, FLOP counts, implementation names, cache mode,
  Criterion estimates, and other context.

Nothing is zero-filled. A measurement that cannot be taken is a metric with
`status: "unavailable"` and a reason, and the CLI exits non-zero when a whole
suite produced no usable metric.

## What is measured, and how

| Metric | Category / unit | Meaning |
|---|---|---|
| `model.load` | `load` / ms | 2.x: deterministic weight synthesis + AOT graph construction for both sequence lengths. 3.0: artifact read + runtime instantiation. |
| `model.prepare` | `prepare` / ms | Compiler passes + memory planning + backend plan build; prefill and decode plans together, cold executor per sample. |
| `prefill.time_to_first_token` | `prefill` / ms | Full prompt forward pass. 2.x has no KV cache. |
| `prefill.throughput` | `throughput` / tokens/s | Derived from the TTFT samples (`prompt_tokens / seconds`). |
| `decode.per_token` | `decode` / ms | One-token full-stack forward per sample. 2.x has no KV cache. |
| `decode.throughput` | `throughput` / tokens/s | Derived from decode samples. |
| `memory.peak_rss` | `memory` / MiB | Whole-process `VmHWM` at the end of the suite. |
| `memory.device_peak` | `memory` / MiB | Device memory where queryable; unavailable today (no GPU execution). |
| `allocation.tls_vec_allocs_per_decode_step` | `allocation` / count | `fastnn::backend::cpu::telemetry` per decode step. |
| `allocation.arena_temp_copies_per_decode_step` | `allocation` / count | Same telemetry; byte total is in `counters`/metadata. |
| `microbench.<proj>.gemm_f32` | `microbench` / ms | Prefill-shaped `gemm_cpu_flat::<F32x1>` over `PackedTensor`, same kernel as `benches/cpu_baselines.rs`. |
| `microbench.<proj>.gemv_f32` | `microbench` / ms | One-token `gemv_cpu::<F32x1>`, same kernel as `benches/bench_util.rs`. |
| `...gemm_gflops` / `...gemv_gflops` | `microbench` / GFLOP/s | Derived from the corresponding latency samples (`2*m*k*n`). |
| `criterion.<group>.<bench>` | `microbench` / ms | Raw per-iteration times imported from Criterion `new/sample.json`; Criterion's own estimates are retained in metadata. |

GPT-2 small shapes used: hidden 768, 12 attention heads, head dim 64,
intermediate 3072, vocabulary 50257, layers 12 (1 in quick mode). Projection
cases: `attn.qkv` 768→2304, `attn.out` 768→768, `mlp.fc` 768→3072,
`mlp.proj` 3072→768, `lm_head` 768→50257.

### Baseline whole-model path

This revision ships no weight artifact and no whole-model loader. The harness
therefore builds a real, deterministic AOT graph with f32 weights:

```
hidden [seq, 768] -> 12 x (pre-LN attention + pre-LN MLP) -> LN -> lm_head
```

Attention is expressed with rank-2 matmuls per head, which is inside the
operator set the existing CPU tests exercise. It is **unmasked full attention**
(causal masking would not change the measured FLOPs), and there is **no KV
cache** in 2.x, so decode is a one-token full-stack forward. These limitations
are recorded in `notes` and metric metadata rather than presented as an
incremental decoder.

## GPU timing rule

1. A GPU measurement must synchronize or await work completion before the host
   timer is stopped (queue submit + fence/wait, or backend-provided `poll`
   that blocks until completion). A metric collected without that wait is
   invalid and must not be emitted.
2. Host time and device time are separate series. Host time measures the full
   submission plus synchronization wall clock; device time is only recorded
   from backend counter/timestamp queries.
3. They are never blended. Consumers comparing GPU runs must use `device_time`
   when it is available and `host_time` otherwise, and must not mix the two
   within one table.
4. If device time is unavailable, record `device_time.status: "unavailable"`
   with a reason, empty samples, and `null` stats. Do not put host time into
   `device_time`.
5. Device memory is reported only when the backend exposes a query; otherwise
   `memory.device_peak` is explicitly unavailable.
6. One physical GPU benchmark at a time. The start guard records other live
   `fn3-bench` processes and warns.

This revision executes on the CPU (`gpu.execution_backend: "cpu"`). The GPU
block is populated from `vulkaninfo --summary` for identity and reproducibility
only (physical GTX 1080 Ti plus the lavapipe software ICD on this host);
`selected_device_index` is `null`. No GPU metric is reported.

## Performance discipline

- Schedule benchmark jobs outside competing builds. The harness records the
  1-minute load average at start, prints/records a warning when it exceeds
  `--max-load` (default 2.0), and lists other `cargo`/`rustc`/`rustdoc` PIDs.
  A non-empty list or a warning means the run is evidence, not a release claim.
- One GPU benchmark at a time. Other live `fn3-bench` PIDs are recorded and
  warned about.
- Raw samples are retained. Do not post-filter a run to make a percentile look
  better; rerun with `--samples` if more evidence is needed.
- Results are marked dirty when the worktree has changes. Compare warns about
  dirty and commit/branch differences (soft), and refuses geometry, precision,
  host CPU/kernel/threads, model identity/seed, GPU selection, or suite
  differences (hard) unless `--force`.
- The benchmark seed controls all fixture data; changing it changes the model.

## Comparison semantics

`fn3-bench compare A B` prints A/B `p50` (or `--stat mean|min|p95|max`) values,
absolute delta `B - A`, and ratio `B / A` per metric. Multiple `A@commit`
adapter combinations are allowed: comparing `baseline` to `fn3` is the intended
use.

Hard mismatches (exit code 2, no table unless `--force`): `schema_version`,
`run.suite`, every `geometry` field, every `precision` field,
`model.name`/`revision`/`seed`, `host.cpu_model`, `host.logical_cores`,
`host.threads_used`, `host.kernel`, `gpu.selected_device`.

Soft mismatches (printed, comparison proceeds): commit, branch, dirty flag,
build profile/features/rustc, start load, competing build PIDs.

`--json` emits the diff report (`mismatches`, `blocked`, per-metric rows) for
FN3-21/25/26/29 automation.

## Adapters

- `baseline` (working): 2.x CPU runtime in this revision.
- `fn3` (seam): points at the future `fastnn::fn3` runtime. It fails with
  `fn3 adapter is not built in this revision (base a2c0759…)` and exit code 3;
  it never emits baseline numbers under the fn3 name. The required runtime
  surface is documented in `tools/bench/adapters/fn3.rs`:

  1. `load() -> PreparedModel` — read/instantiate the artifact, separate from
     compilation.
  2. `prepare(model) -> PreparedModel` — compile/plan/upload, separate timing.
  3. `prefill(model, &[u32]) -> ()` — prompt forward, synchronized before
     return, first-token latency observable at the caller.
  4. `decode(model, u32) -> ()` — steady-state incremental step (KV cache).
  5. `last_device_time_ms() -> Option<f64>` — device time for the last op.
  6. `peak_device_memory_bytes() -> Option<u64>` — device memory where
     queryable.
  7. An allocation/telemetry snapshot accessor for per-step counters
     (optional; unavailable is acceptable if explicitly reported).

## Reuse inventory

- `benches/cpu_baselines.rs`: reused by the `criterion` suite, which runs it
  and imports its raw samples; no benchmark is reimplemented.
- `benches/bench_util.rs`: its GEMV measurement path uses the same
  `gemv_cpu::<F32x1>` kernel; the harness calls the kernel directly so it can
  retain per-sample timings.
- `scripts/criterion_to_json.py` / `tests/test_criterion_to_json.py`: the
  Criterion `estimates.json` layout is followed, but the harness reads
  `sample.json` directly to retain raw samples instead of only aggregates.
- `tests/benchmark_utils.py`: not used (Python-side timing); its
  warmup/iterations/statistics model informed this schema.
- `.benchmarks/` does not exist in this revision; no legacy data was migrated.
- `docs/roadmap/fn3-ledger.md` was not present in this worktree at implementation
  time.

## Known limits and risks

- No GPU execution path exists in this revision, so device timing and device
  memory are exercised only as explicitly unavailable fields. The GPU timing
  rule above is the contract the first GPU adapter must implement.
- Full (non-quick) `model` and `projections` runs take minutes on 2.x because
  packed f32 prefill throughput is low; `--quick` is the mandatory smoke path.
  Use `--samples` to trade evidence for time.
- The Criterion suite requires a bench-profile build (LTO) and is therefore
  slow; it is excluded from the smoke path.
- The load threshold of 2.0 is a warning heuristic, not a hard gate.
- Fixture checksums use FNV-1a 64 (fast, non-cryptographic). The algorithm is
  recorded next to every value; this is fixture identity, not supply-chain
  integrity.
- 2.x packed f32 GEMM numbers are much lower than dense BLAS because of
  per-row unpacking in the packed path. That is a measured property of the
  baseline runtime, not a harness defect.
