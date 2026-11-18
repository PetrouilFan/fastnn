# FN3-04 baseline smoke evidence — 2026-11-18

Clean-tree run of the mandatory `--quick` smoke suite against the 2.x baseline.

## Run identity

- Run id: `f0da74cde8c902486e77ff6fc499745b`
- Timestamp: `2026-11-18T11:18:32Z`
- Git: `fn3/50-bench@918ec7d7b58bbdd91383e78314b216de2059a408` dirty=`false`
- Adapter/suite: `baseline` / `smoke`, quick=`true`
- Seed: `20260818`; wall duration: `14.424 s`
- Build: `release`, rustc `rustc 1.100.0 (0ef1bc386 2026-11-09)`, features `fusion-forward, parallel, prepared-plan, simd`
- Host: AMD Ryzen 7 3700X 8-Core Processor, 8 physical / 16 logical cores, 16 threads, kernel `7.2.14-arch1-1`
- Start load: 1m `0.79` (threshold `2.0`), exceeded=`false`
- GPU enumerated: NVIDIA GeForce GTX 1080 Ti (physical, api 1.4.312, driver 580.196.7.0, uuid 471b2bda-51e1-6451-e692-ff5d1e146f1e), llvmpipe (LLVM 23.1.4, 256 bits) (software, api 1.4.354, driver 26.2.7, uuid 6d657361-3236-2e32-2e37-2d6172636800)
- Execution backend: `cpu` — CPU execution in this revision; 1 physical and 1 software Vulkan device(s) enumerated, none selected
- Model: `synthetic-tiny-decoder` rev `synthetic:20260818`, 173280 parameters, 693120 weight bytes, checksum `fnv1a64:a4062200d06b3e8b`
- Geometry: prompt 8, decode 5, layers 1, hidden 96, heads 2x48, vocab 1024, kv_cache=`false`
- Warnings: none

Command:

```bash
CARGO_BUILD_JOBS=4 cargo build --release --no-default-features \
  --features simd,parallel,prepared-plan,fusion-forward --bin fn3-bench
./target/release/fn3-bench run --suite smoke --quick \
  --out /tmp/opencode/fn3-04-smoke-final.json
cp /tmp/opencode/fn3-04-smoke-final.json docs/reports/fn3-04/smoke-run.json
```

The benchmark command completed in 14.4 s wall time (well under the one-minute
smoke budget). The JSON was written outside the worktree so the captured
`git.dirty` is `false`; it was copied into this directory afterwards.

## Measurements

| Metric | Category | Unit | p50 | min | p95 | max | n |
|---|---|---|---:|---:|---:|---:|---:|
| `model.load` | load | ms | 2.9334 | 2.5476 | 3.2806 | 3.3191 | 2 |
| `model.prepare` | prepare | ms | 0.3267 | 0.2867 | 0.3626 | 0.3666 | 2 |
| `prefill.time_to_first_token` | prefill | ms | 0.3640 | 0.3567 | 0.3717 | 0.3725 | 3 |
| `prefill.throughput` | throughput | tokens/s | 21975.5467 | 21476.4524 | 22383.0368 | 22428.3135 | 3 |
| `decode.per_token` | decode | ms | 0.1922 | 0.1912 | 0.2000 | 0.2019 | 5 |
| `decode.throughput` | throughput | tokens/s | 5203.9696 | 4954.1493 | 5229.2231 | 5229.5511 | 5 |
| `memory.peak_rss` | memory | MiB | 10.3594 | 10.3594 | 10.3594 | 10.3594 | 1 |
| `memory.device_peak` | memory | MiB | unavailable | – | – | – | 0 |
| `allocation.tls_vec_allocs_per_decode_step` | allocation | count | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5 |
| `allocation.arena_temp_copies_per_decode_step` | allocation | count | 3.0000 | 3.0000 | 3.0000 | 3.0000 | 5 |
| `microbench.attn.qkv.gemm_f32` | microbench | ms | 110.8262 | 110.1543 | 115.8140 | 116.3682 | 3 |
| `microbench.attn.qkv.gemm_gflops` | microbench | GFLOP/s | 0.2555 | 0.2433 | 0.2569 | 0.2570 | 3 |
| `microbench.attn.qkv.gemv_f32` | microbench | ms | 0.0995 | 0.0993 | 0.0998 | 0.0998 | 3 |
| `microbench.attn.qkv.gemv_gflops` | microbench | GFLOP/s | 35.5780 | 35.4462 | 35.6422 | 35.6493 | 3 |
| `microbench.attn.out.gemm_f32` | microbench | ms | 36.6315 | 36.4690 | 37.1028 | 37.1552 | 3 |
| `microbench.attn.out.gemm_gflops` | microbench | GFLOP/s | 0.2576 | 0.2540 | 0.2587 | 0.2588 | 3 |
| `microbench.attn.out.gemv_f32` | microbench | ms | 0.0354 | 0.0353 | 0.0354 | 0.0354 | 3 |
| `microbench.attn.out.gemv_gflops` | microbench | GFLOP/s | 33.3046 | 33.3036 | 33.3809 | 33.3894 | 3 |
| `microbench.mlp.fc.gemm_f32` | microbench | ms | 148.0773 | 146.1185 | 151.4176 | 151.7888 | 3 |
| `microbench.mlp.fc.gemm_gflops` | microbench | GFLOP/s | 0.2549 | 0.2487 | 0.2580 | 0.2583 | 3 |
| `microbench.mlp.fc.gemv_f32` | microbench | ms | 0.1290 | 0.1290 | 0.1344 | 0.1350 | 3 |
| `microbench.mlp.fc.gemv_gflops` | microbench | GFLOP/s | 36.5666 | 34.9603 | 36.5870 | 36.5893 | 3 |
| `microbench.mlp.proj.gemm_f32` | microbench | ms | 44.8249 | 44.4952 | 45.0991 | 45.1296 | 3 |
| `microbench.mlp.proj.gemm_gflops` | microbench | GFLOP/s | 0.8421 | 0.8365 | 0.8478 | 0.8484 | 3 |
| `microbench.mlp.proj.gemv_f32` | microbench | ms | 0.1329 | 0.1327 | 0.1330 | 0.1330 | 3 |
| `microbench.mlp.proj.gemv_gflops` | microbench | GFLOP/s | 35.5126 | 35.4888 | 35.5559 | 35.5608 | 3 |
| `microbench.lm_head.gemm_f32` | microbench | ms | 2308.7718 | 2303.7993 | 2338.7692 | 2342.1023 | 3 |
| `microbench.lm_head.gemm_gflops` | microbench | GFLOP/s | 0.2675 | 0.2637 | 0.2680 | 0.2681 | 3 |
| `microbench.lm_head.gemv_f32` | microbench | ms | 4.6676 | 4.5914 | 4.8994 | 4.9252 | 3 |
| `microbench.lm_head.gemv_gflops` | microbench | GFLOP/s | 16.5386 | 15.6736 | 16.7855 | 16.8129 | 3 |

Percentiles are linear-interpolation over the sorted raw samples (R-7); raw samples are in [`smoke-run.json`](smoke-run.json).

## Explicitly unavailable

- `memory.device_peak` (MiB) — no GPU execution in this run; device memory is not queryable from the CPU baseline adapter

`device_time` is `unavailable` for every metric because the CPU backend does not report per-operation device clocks; `host_time` carries the host monotonic timing. No value was zero-filled.

## Criterion wrapper

The `criterion` suite ran `cargo bench --no-default-features --features simd,parallel,prepared-plan,fusion-forward --bench cpu_baselines -- --sample-size 10 --warm-up-time 0.1 --measurement-time 0.3 --noplot`; the Criterion benches override some of those timings with their own group settings, so the run took several minutes. 136 benchmark entries were imported from Criterion's raw `new/sample.json` (per-iteration times, nanoseconds converted to milliseconds) with Criterion's own `point_estimate` values retained in metadata. Representative p50 values from that import:

| Criterion metric | p50 (ms) |
|---|---:|
| `criterion.cpu_gemv.fastnn_f32x1/1024x1024` | 0.0560 |
| `criterion.cpu_gemv.baseline_scalar_f32/1024x1024` | 0.7074 |
| `criterion.cpu_gemm.fastnn_f32x1/batch8_256x256` | 0.0769 |
| `criterion.cpu_training_updates.adamw/64k` | 0.0690 |

The import was exercised with `--reuse-criterion` on the tree produced by the successful `cargo bench` invocation; `--reuse-criterion` only skips the bench step and reads the same `new/estimates.json` + `new/sample.json` files.

## Comparison demo

A second clean-tree smoke run (`smoke-run-b.json`, run id
`3be15f71e9de831f94b9070698606fa6`, load 0.56) was captured and compared with
the first. Both runs used the same commit lineage, geometry, precision, model
identity and host; the only differences are the commit (evidence commit) and
start load, both reported as *soft* mismatches. The tool produced a diff table
and exited 0:

```bash
./target/release/fn3-bench compare \
  docs/reports/fn3-04/smoke-run.json \
  docs/reports/fn3-04/smoke-run-b.json
```

Full output: [`compare-example.txt`](compare-example.txt). Excerpt:

```
compatibility: 2 mismatch(es) (soft only)
  [soft] git.commit: A=Some("918ec7d...") B=Some("a1ff0a1...")
  [soft] host.load.one: A="Some(0.79)" B="Some(0.56)"
...
decode.per_token          ms   0.1922   0.1879   -0.0042  0.978x   5
decode.throughput  tokens/s   5203.9696 5321.1302 +117.1606 1.023x  5
microbench.lm_head.gemm_f32 ms 2308.7718 2331.9426 +23.1708 1.010x  3
```

Comparing runs with different suites/geometry was exercised separately and is
refused with exit code 2 unless `--force`, as documented in the harness README.
`--json` emits the same diff as structured data for downstream tooling.

## Interpretation limits

- The smoke suite uses a tiny decoder (hidden 96, 2 heads, vocab 1024) for load/prepare/prefill/decode coverage; the projection microbenchmarks use real GPT-2 small shapes (hidden 768, 12 heads x 64, FFN 3072, vocab 50257).
- The 2.x runtime has no KV cache and no weight artifact: `model.load` is deterministic weight synthesis plus graph construction, `model.prepare` is AOT compile for prefill and decode sequence lengths, and decode is a one-token full-stack forward.
- Attention in the synthetic model is unmasked full attention; causal masking would not change the measured FLOPs.
- Projection p50 values are for the packed f32 CPU path at 8 prompt tokens (quick mode); they are not a claim about dense BLAS performance.
- The smoke run is a harness conformance fixture, not a performance baseline. Use the non-quick `projections`/`model` suites on a quiet host for comparative numbers.
