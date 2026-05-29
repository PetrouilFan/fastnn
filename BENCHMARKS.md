# fastnn Benchmarks

fastnn's maintained release-facing CPU benchmark suite is `cpu_baselines`. WGPU/GPU benchmark entrypoints remain manual and hardware-dependent.

## Maintained CPU suite: `cpu_baselines`

The maintained CPU suite measures deterministic CPU workloads used to ground regression work and release claims.

Current and planned groups:

- `cpu_gemv`: packed and scalar matrix-vector workloads.
- `cpu_gemm`: packed and scalar batched matrix-matrix style workloads.
- `cpu_elementwise`: same-shape, scalar, and broadcasted elementwise workloads.
- `cpu_reductions`: 1D and row-wise sum/mean/max workloads.
- `cpu_fusions`: matmul+bias+activation and residual+add+norm workloads.
- `cpu_training_updates`: optimizer update-loop workloads where stable benchmarkable APIs exist.

### CPU commands

List the maintained suite without running all benchmarks:

```bash
cargo +stable bench --bench cpu_baselines -- --list
```

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

### CPU reproducibility checklist

For CPU regression work:

- use `cargo +stable`
- run on an otherwise idle machine
- keep CPU governor/frequency policy consistent between runs
- keep thermal state consistent; avoid comparing cold and thermally throttled runs
- set or record Rayon/thread configuration, for example `RAYON_NUM_THREADS`
- record CPU model, OS, Rust version, and feature flags
- compare Criterion baselines rather than copying numbers by hand
- treat large regressions as actionable; reproduce small deltas before acting

## Manual WGPU benchmark entrypoints

WGPU results are hardware-, driver-, adapter-, and backend-dependent. They are not the v2.3 focus and are not a stable CI gate yet. Run them manually on the target GPU before publishing GPU performance claims.

Available entrypoints include:

```bash
cargo +stable bench --bench wgpu_bench --features gpu
cargo +stable bench --bench wgpu_inference --features gpu
```

Until a maintained WGPU suite exists, WGPU benchmark numbers are useful engineering evidence but should not be treated as portable release claims.

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

## Performance claim policy

Do not add README or release-note speed claims unless they come from a reproducible benchmark run in this suite or another checked-in benchmark with the same standards.

Every public performance claim should include:

- exact benchmark command
- hardware context
- baseline being compared against
- whether the number is latency, throughput, or memory footprint
- feature flags and thread configuration
- for GPU/WGPU claims: GPU model, driver/backend, OS, and whether the run was cold or warm

Unsupported claims to avoid:

- hard-coded speedup tables with no runnable command
- comparisons to external frameworks without checked-in reproduction steps
- GPU performance claims from machines where the GPU benchmark was not run successfully
