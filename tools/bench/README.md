# `fn3-bench`

Source for the FN3-04 benchmark harness. Build and usage instructions, the
versioned result schema, measurement definitions, GPU timing rules, performance
discipline, and the 3.0 adapter contract live in
[`docs/reports/fn3-04/README.md`](../../docs/reports/fn3-04/README.md).

Layout:

- `main.rs` — CLI (`run`, `compare`, `list-suites`).
- `schema.rs` — schema v1 Rust types and unavailable/device-time semantics.
- `env.rs` — git/build/host/GPU/load capture, deterministic run ids.
- `stats.rs` — raw samples, documented percentiles, derived throughput.
- `timing.rs` — explicit warmup/sample/repetition batches, CPU telemetry.
- `model.rs` — deterministic synthetic GPT-2 decoder graph for 2.x.
- `adapters/baseline.rs` — 2.x projection + whole-model suites.
- `adapters/criterion_wrap.rs` — imports `benches/cpu_baselines.rs` results.
- `adapters/fn3.rs` — 3.0 prepared-model adapter behind `--features fn3`.
- `fn3-model.example.json` — manifest for the pinned tiny-GPT-2 model.
- `compare.rs` — compatibility checks and delta table.
