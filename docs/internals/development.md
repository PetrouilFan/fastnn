# Developing FastNN

## The prepare and run path

```text
ONNX ──import──► typed IR ──plan──► artifact v2 (FNN3ART2) ──load──► Session
                  (ir)      (runtime)      (model/format)                 │
                                                                          ▼
                                                    arena + worker pool, one forward
                                                                          │
                                                                     outputs
```

`import` is strict (opset 12..=22) and belongs to Rust; Python and the CLI are
facades over `api::{prepare, load, inspect, doctor}`. Planning chooses the
segment engines, the fusion chains and the layout policy; a session executes the
stored plan and owns every mutable byte (arena, threads). The contracts are in
`docs/v3/adr-0001-one-runtime.md`, `src/fn3/AGENTS.md` and the operator matrix
in `SUPPORTED.md`.

## Repository map

```text
src/fn3/
  dtype, shape, error, device, storage, tensor   # leaves
  ir/            typed graph, DimExpr, the shape pass
  import/        ONNX reader, graph construction, shape binding
  ops/           the OpId registry (never renumber)
  runtime/       plan, fusion, arena, session, scalar kernels
  model/format/  artifact v2: header, graph codec, plans, weights
  backend/cpu/   portable scalar kernels and the oneDNN engine (`cpu-onednn`)
  backend/vulkan/ platform and resident code, compiled but unwired (#108)
  cli/           the `fastnn-infer` implementation
  python/        the PyO3 binding
```

The dependency direction is a DAG
(`dtype/device/ops → error → shape/storage → tensor → backend → model`), with
`runtime` on top. One upward edge is intentional: `backend::cpu::dispatch_kernels`
calls `runtime::kernels::dispatch`, so `Support::Full` and the kernel behind it
cannot drift apart. `src/fn3/` imports no 2.x module and 2.x imports nothing
from it.

## Adding an operator

1. Assign a stable `OpId` in `src/fn3/ops/mod.rs`; never renumber an existing
   one.
2. Write the shape rule in `src/fn3/ir/infer.rs` and the `OpParams` mapping in
   `src/fn3/import/graph.rs`. A rule that needs a constant reads it from
   `OpParams`, never from a tensor payload.
3. Implement the scalar kernel under `src/fn3/runtime/kernels/` (or the shared
   portable one under `src/fn3/backend/cpu/kernels/`) and add it to
   `kernels/mod.rs` and `is_cnn_op`. A backend that cannot run the op returns
   `UnsupportedOp`; it never approximates, returns an input or returns zero.
4. If the oneDNN engine can take it, add the claim in
   `backend/cpu/onednn/geometry.rs` and the preparation in `mod.rs`. A declined
   segment must keep its reason in `Plan::fallbacks`.
5. Add a numerical test beside the kernel, an import case if the ONNX spelling
   is new, and a parity row when the op reaches the CNN gates.

## Adding a fused segment

`runtime/fuse.rs` recognises the chain and proves every intermediate has no
consumer outside it; `runtime/kernels/fused.rs` implements the scalar reference
in the same arithmetic order as the unfused chain. The oneDNN engine may take
the segment only when its epilogue reproduces that order. The planner report
(`Plan::fusions`) and a bit-identity test against the unfused plan are part of
the change.

## Tests

| Area | Command |
|---|---|
| Everything the CPU lane runs | `cargo test --lib -- --test-threads=1` then `cargo test --no-fail-fast --tests` |
| Import and shape binding | `cargo test --test fn3_import --test fn3_shape_binding` |
| Constant folding | `cargo test --test fn3_constant_folding` |
| Artifact v2 | `cargo test --test fn3_artifact_v2 --test fn3_model_format` |
| Scalar/oneDNN parity | `cargo test --test fn3_cnn_run` and `cargo test --no-default-features --features cpu-onednn --test fn3_onednn` |
| CLI surface | `cargo test --test fn3_cli` |
| Python binding | `maturin develop --release && pytest tests/ -m "not slow"` |

The CPU lane runs the library tests and then every integration target; the
narrow commands above are for iterating on one area, not a substitute for the
lane. The CNN gates need an external YOLO file and an ONNX Runtime dump;
generate the dump with `python3 tools/parity/ort_reference.py <model>
target/cnn-parity`. They print the command and return when the files are
absent, and never `#[ignore]` themselves.

Vulkan is compiled behind the `vulkan` feature but no session reaches it in
this tree, and no test executes it until phase 6 (#108). Do not re-add a Vulkan
lane that reports a pass without a device.

## Performance reports

`tools/perf/perf_report.py` writes one JSON report per measurement:

```bash
PYTHONPATH=. .venv/bin/python tools/perf/perf_report.py \
  --model evit-depth-5m=/path/to/efficientvit_depth_5m.onnx \
  --input pixel_values=1,3,336,448 --threads 4 --repeat 20 \
  --require-quiet 2 --out docs/reports/perf-evit-depth-5m-cpu.json
```

It records the commit, the machine (including the 1-minute load average), the
thread count, the dtype, the model's size and SHA-256, min/median/p95, RSS at
each stage and a SHA-256 of every output. `--require-quiet N` refuses to
measure when the load average is above `N`: a benchmark taken on a busy host is
noise, and the number is recorded so a reader can tell the two apart.

`tools/perf/compare_perf.py --base A --head B` prints the transitions and
fails when the median grows past 15 % or the peak RSS past 20 %, or when an
output digest changed. `tools/ci/perf-lane.sh` runs both, and skips with an
explanation when the model or the baseline is absent, because a model is not
committed to this repository.

The v3 measurement path is `fastnn-infer run model.fnn3 --input ... --warmup 5
--repeat 50`: it times `Session::execute` with the arena in place and prints
median and p95. `tests/fn3_cnn_perf.rs` is the opt-in in-process harness
(`FN3_CNN_PERF=scalar|onednn`); the permanent performance CI is #106.
