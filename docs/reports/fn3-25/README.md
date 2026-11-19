# FN3-25 (#71) raw measurement data

Everything in this directory was produced by the release binary of branch
`fn3/71-cpu-perf` (base `672cec9faf089ef0840179f6edfdf33779b576a1`). No file
here is hand-edited: the `.log` files are the stdout of the commands named
below, with only the harness's banner lines kept and no numeric value changed.

## Files

| File | Produced by | What it is |
|---|---|---|
| `tiny-sweep.log` | `tests/fn3_cpu_perf.rs` on the tiny pinned model | thread sweep, per-op profile, kernel microbenchmarks, parity, allocations, concurrent sessions |
| `trained-sweep.log` | the same test binary on the trained pinned model, 10-token prompt | the same tables for the 12-layer model |
| `trained-13.log` | the same binary, 13-token prompt and context (second length) | the trained sweep at a second prompt/context length |
| `ab-base-round{1,2,3}.log` | the base-commit binary | the "before" arm of the algorithmic A/B |
| `ab-after1t-round{1,2,3}.log` | this branch at `FN3_CPU_THREADS=1` | the "after" arm at one thread |
| `smoke.log` | this branch | the budget-reaches-the-kernels assertion run |
| `ab-*.log` | both arms of the algorithmic A/B | three interleaved rounds, medians in the report |
| `build.log` | `cargo test --release --test fn3_cpu_perf --no-run` | build transcript for the measured binary |
| `transpose-shapes.log` | `transpose-shapes.rs` | the access-pattern study that chose the transpose loop |
| `transpose-threads.log` | `transpose-threads.rs` | the threading ceiling for the safe transpose partition |
| `transpose-shapes.rs` | source | standalone `rustc -O` program, not part of the crate |
| `transpose-threads.rs` | source | standalone `rustc -O` program, not part of the crate |

## Commands

Each window was run under the shared measurement lock, so no other FN3 worker
was building or timing at the same time:

```bash
flock -w 7200 /tmp/opencode/fn3-measure.lock -c '/tmp/opencode/fn3-71/authoritative.sh'
```

That script performs, in order: a quiet-machine check (`pgrep -a 'cargo|rustc'`),
the release build, the budget smoke test, the tiny sweep, the trained sweep, the
three interleaved A/B rounds, and the two standalone studies. The per-step
`/proc/loadavg` is inside each log (`Environment`, `SWEEP`, `Concurrent` rows).

The standalone studies are compiled and run directly:

```bash
rustc -O -o transpose-shapes transpose-shapes.rs && ./transpose-shapes
rustc -O -o transpose-threads transpose-threads.rs && ./transpose-threads
```

## Trimming

The `.log` files keep only the harness's machine-readable rows (`SWEEP`,
`KERNEL`, `PARITY`, `PROFILE*`, `Allocations`, `Budget`, `Concurrent`, `Census`,
`PERF`, `BitExact`, `Environment`, `Model`, `test result`) plus the build
transcript. Non-numeric banner lines and the `test ... ok` chatter were dropped;
no numeric value was edited. The full untrimmed output is in
`/tmp/opencode/fn3-71/final/` on the measurement host.

Two earlier windows are **superseded and not committed**: one measured the
decode workload with a repeated prefill on the same backend, which the chain
workload in the final protocol showed was contaminated (report §7.5c), and one
ran an unbalanced row partition. Both live in `/tmp/opencode/fn3-71/final-old/`
and `final-2/`; their conclusions are quoted in the report only where they are
still valid (the transpose-threading ceiling and the pre-threshold kernel
speedup of the one-token `lm_head`).

## Models

Neither model is committed (no declared licence for the tiny one; the trained
one is 498 MB). Digests are verified by the harness on every load:

* `tiny-gpt2` — 27,897,974 B, sha256 `e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1`,
  matches `fixtures/parity/tiny-gpt2/model.lock.json`.
* `trained-gpt2` — 498,126,358 B, sha256 `42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d`,
  the #68 pin.
