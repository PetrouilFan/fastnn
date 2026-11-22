# FN3-04 baseline smoke evidence — 2026-11-18

Clean-tree run of the mandatory `--quick` smoke suite against the 2.x baseline.

## Run identity

- Run id: `1cc95c0bc27e609c343ef49949e12ec1`
- Timestamp: `2026-11-18T11:33:44Z`
- Git: `fn3/50-bench@55ab4bb474b6be593437047b61508add81b95340` dirty=`false`
- Adapter/suite: `baseline` / `smoke`, quick=`true`
- Seed: `20260818`; wall duration: `14.534 s`
- Build: `release`, rustc `rustc 1.100.0 (0ef1bc386 2026-11-09)`, features `fusion-forward, parallel, prepared-plan, simd`
- Host: AMD Ryzen 7 3700X 8-Core Processor, 8 physical / 16 logical cores, 16 threads, kernel `7.2.14-arch1-1`
- Start load: 1m `1.59` (threshold `8.0`), exceeded=`false`
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

The benchmark command completed in 14.5 s wall time (well under the one-minute
smoke budget). The JSON was written outside the worktree so the captured
`git.dirty` is `false`; it was copied into this directory afterwards.

## Measurements

| Metric | Category | Unit | p50 | min | p95 | max | n |
|---|---|---|---:|---:|---:|---:|---:|
| `model.load` | load | ms | 2.9167 | 2.5443 | 3.2518 | 3.2891 | 2 |
| `model.prepare` | prepare | ms | 0.3366 | 0.2644 | 0.4015 | 0.4087 | 2 |
| `prefill.time_to_first_token` | prefill | ms | 0.3624 | 0.3613 | 0.3639 | 0.3640 | 3 |
| `prefill.throughput` | throughput | tokens/s | 22077.4311 | 21976.7541 | 22136.7736 | 22143.3672 | 3 |
| `decode.per_token` | decode | ms | 0.1893 | 0.1886 | 0.1963 | 0.1980 | 5 |
| `decode.throughput` | throughput | tokens/s | 5281.2252 | 5049.9950 | 5301.3000 | 5302.1988 | 5 |
| `memory.peak_rss` | memory | MiB | 10.3164 | 10.3164 | 10.3164 | 10.3164 | 1 |
| `memory.device_peak` | memory | MiB | unavailable | – | – | – | 0 |
| `allocation.tls_vec_allocs_per_decode_step` | allocation | count | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5 |
| `allocation.arena_temp_copies_per_decode_step` | allocation | count | 3.0000 | 3.0000 | 3.0000 | 3.0000 | 5 |
| `microbench.attn.qkv.gemm_f32` | microbench | ms | 110.9113 | 109.5053 | 114.6429 | 115.0575 | 3 |
| `microbench.attn.qkv.gemm_gflops` | microbench | GFLOP/s | 0.2553 | 0.2461 | 0.2582 | 0.2585 | 3 |
| `microbench.attn.qkv.gemv_f32` | microbench | ms | 0.1111 | 0.1110 | 0.1115 | 0.1115 | 3 |
| `microbench.attn.qkv.gemv_gflops` | microbench | GFLOP/s | 31.8537 | 31.7363 | 31.8899 | 31.8939 | 3 |
| `microbench.attn.out.gemm_f32` | microbench | ms | 36.2963 | 36.2895 | 36.3056 | 36.3067 | 3 |
| `microbench.attn.out.gemm_gflops` | microbench | GFLOP/s | 0.2600 | 0.2599 | 0.2600 | 0.2601 | 3 |
| `microbench.attn.out.gemv_f32` | microbench | ms | 0.0342 | 0.0342 | 0.0342 | 0.0342 | 3 |
| `microbench.attn.out.gemv_gflops` | microbench | GFLOP/s | 34.4624 | 34.4523 | 34.4896 | 34.4926 | 3 |
| `microbench.mlp.fc.gemm_f32` | microbench | ms | 147.8348 | 146.3183 | 148.5870 | 148.6705 | 3 |
| `microbench.mlp.fc.gemm_gflops` | microbench | GFLOP/s | 0.2553 | 0.2539 | 0.2577 | 0.2580 | 3 |
| `microbench.mlp.fc.gemv_f32` | microbench | ms | 0.1382 | 0.1380 | 0.1420 | 0.1424 | 3 |
| `microbench.mlp.fc.gemv_gflops` | microbench | GFLOP/s | 34.1529 | 33.1292 | 34.1996 | 34.2048 | 3 |
| `microbench.mlp.proj.gemm_f32` | microbench | ms | 42.1349 | 40.0481 | 44.7955 | 45.0911 | 3 |
| `microbench.mlp.proj.gemm_gflops` | microbench | GFLOP/s | 0.8959 | 0.8372 | 0.9379 | 0.9426 | 3 |
| `microbench.mlp.proj.gemv_f32` | microbench | ms | 0.1187 | 0.1155 | 0.1230 | 0.1235 | 3 |
| `microbench.mlp.proj.gemv_gflops` | microbench | GFLOP/s | 39.7456 | 38.2196 | 40.7552 | 40.8674 | 3 |
| `microbench.lm_head.gemm_f32` | microbench | ms | 2348.9932 | 2344.0030 | 2353.4545 | 2353.9502 | 3 |
| `microbench.lm_head.gemm_gflops` | microbench | GFLOP/s | 0.2629 | 0.2623 | 0.2634 | 0.2635 | 3 |
| `microbench.lm_head.gemv_f32` | microbench | ms | 4.5746 | 4.5342 | 4.5751 | 4.5752 | 3 |
| `microbench.lm_head.gemv_gflops` | microbench | GFLOP/s | 16.8745 | 16.8724 | 17.0100 | 17.0250 | 3 |

Percentiles are linear-interpolation over the sorted raw samples (R-7); raw samples are in [`smoke-run.json`](smoke-run.json).

## Explicitly unavailable

- `memory.device_peak` (MiB) — no GPU execution in this run; device memory is not queryable from the CPU baseline adapter

`device_time` is `unavailable` for every metric because the CPU backend does not report per-operation device clocks; `host_time` carries the host monotonic timing. No value was zero-filled.

## Criterion wrapper

The `criterion` suite ran `cargo bench --no-default-features --features simd,parallel,prepared-plan,fusion-forward --bench cpu_baselines -- --sample-size 10 --warm-up-time 0.1 --measurement-time 0.3 --noplot`; the Criterion benches override some of those timings with their own group settings, so the run took several minutes. The fresh run completed with exit code 0 and 136 benchmark entries were imported from Criterion's raw `new/sample.json` (per-iteration times, nanoseconds converted to milliseconds) with Criterion's own `point_estimate` values retained in metadata. Representative p50 values:

| Criterion metric | p50 (ms) |
|---|---:|
| `criterion.cpu_gemv.fastnn_f32x1/1024x1024` | 0.0557 |
| `criterion.cpu_gemv.baseline_scalar_f32/1024x1024` | 0.7304 |
| `criterion.cpu_gemm.fastnn_f32x1/batch8_256x256` | 0.0772 |
| `criterion.cpu_training_updates.adamw/64k` | 0.0665 |

That fresh run carried one competing-build warning from another worktree's build at start; the import path was also re-verified with `--reuse-criterion`. `--reuse-criterion` only skips the bench step and reads the same `new/estimates.json` + `new/sample.json` files.

## Comparison demo

A second clean-tree smoke run (`smoke-run-b.json`, run id `dfe536a03bd7d04fb552733cdb14ea6d`, load `3.23`) was captured and compared with the first. Both used the same commit, geometry, precision, model identity, host and thread count; the only difference is the start load, reported as a *soft* mismatch. The tool produced a diff table and exited 0:

```bash
./target/release/fn3-bench compare \
  docs/reports/fn3-04/smoke-run.json \
  docs/reports/fn3-04/smoke-run-b.json
```

Full output: [`compare-example.txt`](compare-example.txt). Excerpt:

```
compatibility: 1 mismatch(es) (soft only)
  [soft] host.load.one: A="Some(1.59)" B="Some(3.23)"
decode.per_token                                                ms       0.1893       0.2157      +0.0263    1.139x         5  
decode.throughput                                          tokens/s    5281.2252    4636.9285    -644.2967    0.878x         5  
microbench.lm_head.gemm_f32                                     ms    2348.9932    2319.5613     -29.4319    0.987x         3  
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
