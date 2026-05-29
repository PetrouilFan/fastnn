# fastnn Benchmarks

fastnn currently has one maintained CI-friendly benchmark suite, `cpu_baselines`, plus several WGPU/GPU benchmark entrypoints that should be run manually on machines with compatible hardware.

## Maintained CPU suite: `cpu_baselines`

The maintained CPU suite measures execution only (weights are packed before timing) and compares:

- scalar f32 reference loops (`baseline_scalar_f32`)
- fastnn f32 packed execution (`fastnn_f32x1`)
- fastnn quantized execution (`fastnn_u8x4`, `fastnn_u4x8`)

Covered workloads:

- GEMV: matrix-vector multiply on CPU
- GEMM: batched matrix-matrix style multiply on CPU

### CPU commands

Run the maintained suite:

```bash
cargo +stable bench --bench cpu_baselines
```

Save a reusable Criterion baseline for future comparisons:

```bash
cargo +stable bench --bench cpu_baselines -- --save-baseline cpu-local
```

Compare the current run against a previously captured baseline:

```bash
cargo +stable bench --bench cpu_baselines -- --baseline cpu-local
```

Export the latest Criterion results to one JSON file:

```bash
python scripts/criterion_to_json.py \
  --criterion-dir target/criterion \
  --output benchmark-results/cpu-local.json
```

## Manual WGPU benchmark entrypoints

WGPU results are hardware-, driver-, adapter-, and backend-dependent. They are not a stable CI gate yet. Run them manually on the target GPU before publishing GPU performance claims.

Available entrypoints include:

```bash
cargo +stable bench --bench wgpu_bench --features gpu
cargo +stable bench --bench wgpu_inference --features gpu
```

For v2.3, the WGPU benchmark work should promote one of these entrypoints, or a replacement, to a maintained suite that records:

- cold vs warm shader/pipeline latency
- buffer-pool behavior or allocation count
- matmul latency
- fused matmul+activation latency
- quantized U4/U8 inference latency
- GPU model, driver/backend, OS, and feature flags

Until that maintained suite exists, WGPU benchmark numbers are useful engineering evidence but should not be treated as portable release claims.

## Baseline capture format

Regression baselines are stored in two layers:

1. Raw Criterion output in `target/criterion/...`
   - includes per-benchmark estimates and comparison data
2. Optional normalized JSON summary produced by `scripts/criterion_to_json.py`
   - one file per run for check-in, review, or artifact upload

Example normalized JSON shape:

```json
{
  "generated_at_utc": "2026-05-21T12:34:56Z",
  "criterion_dir": "target/criterion",
  "benchmarks": [
    {
      "group": "cpu_gemv",
      "benchmark": "fastnn_u4x8/1024x1024",
      "mean_ns": 12345.0,
      "median_ns": 12010.0,
      "std_dev_ns": 210.0
    }
  ]
}
```

## Reproducibility expectations

For regression work:

- use `cargo +stable`
- run benchmarks on an otherwise idle machine
- keep thread count, CPU governor, GPU power mode, and thermal state consistent between runs
- compare runs with Criterion baselines instead of copying numbers into docs by hand
- treat large regressions as actionable; treat small deltas as noise until reproduced

## Performance claim policy

Do not add README or release-note speed claims unless they come from a reproducible benchmark run in this suite or another checked-in benchmark with the same standards.

Every public performance claim should include:

- exact benchmark command
- hardware context
- baseline being compared against
- whether the number is latency, throughput, or memory footprint
- for GPU/WGPU claims: GPU model, driver/backend, OS, and whether the run was cold or warm

Unsupported claims to avoid:

- hard-coded speedup tables with no runnable command
- comparisons to external frameworks without checked-in reproduction steps
- GPU performance claims from machines where the GPU benchmark was not run successfully
