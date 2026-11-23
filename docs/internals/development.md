# Development Architecture

Codebase walkthrough for contributors adding ops, kernels, compiler passes and
tests.

## Rust source layout

```text
src/
  lib.rs, error.rs, residual.rs, storage.rs, storage_pool.rs, packed_tensor.rs
  fn3/                        # 3.0 inference core (primary API)
    dtype.rs, shape.rs, error.rs, device.rs, storage.rs, tensor.rs
    ops/                      # stable OpId registry
    backend/  mod.rs          # Backend trait, Capabilities, Support, OpParams
              cpu/            # reference kernels
              vulkan/         # instance, device, memory, dispatch, kernels, shaders
    import/                   # strict ONNX protobuf reader and graph builder
    model/   weights.rs plan.rs specialize.rs kv.rs session.rs format/
    cli/                      # fastnn-infer implementation
    python/                   # PyO3 binding, declared from src/python/mod.rs
  ir/                         # 2.x ComputeGraph, IRNode, Opcode, GraphBuilder
  compiler/                   # 2.x passes (shape, fusion, quantization, memory)
  backend/                    # 2.x executor, prepared plans, CPU kernels
  tensor/, nn/                # 2.x eager forward API
  onnx/, io/                  # 2.x ONNX conversion and .fnn I/O
  dtypes/, swar/              # packed integer types and kernels
  python/                     # 2.x PyO3 bindings (forward only)
```

## The 3.0 path

```text
ONNX ──import──► GraphIr ──prepare──► PreparedArtifact ──load──► PreparedModel
                                                        │
                                             plan(specialise) ──► ExecutionPlan
                                                        │
                                                     Session (KV cache)
                                                        │
                                              prefill / decode ──► logits
```

Contracts live in `docs/archive/roadmap/fn3-architecture.md` and
`src/fn3/AGENTS.md`. The dependency direction is a DAG
(`dtype/device/ops → error → shape/storage → tensor → backend → model`); do not
import upward. `src/fn3/` imports no 2.x module and 2.x imports nothing from it.

### Adding a 3.0 op

1. Assign a stable `OpId` in `src/fn3/ops/mod.rs`; never renumber.
2. Implement the kernel under `src/fn3/backend/cpu/kernels/`.
3. Add an honest `Support` arm in `CpuBackend::supports` and a dispatch arm in
   `execute_op`; validate arity, dtypes and shapes with explicit messages.
4. Extend the planner's `OpParams` mapping and add a contract test in
   `tests/fn3_contracts.rs` plus a numerical test beside the kernel.
5. A backend that cannot run the op returns `UnsupportedOp`; it never
   approximates or falls back.

### Adding a Vulkan kernel

Follow `src/fn3/backend/vulkan/AGENTS.md`: author GLSL under `shaders/`,
compile offline with `shaders/compile.sh`, register a `KernelSpec` in
`kernels/mod.rs`, and prove it with `cargo test --features vulkan` on the named
physical device. `kernels/reflect.rs` checks the declaration against the
committed SPIR-V.

## The secondary 2.x AOT path

```text
ONNX model or GraphBuilder
        │
        ▼
  ComputeGraph (ir/)
        │
        ▼
  shape inference → fusion → quantization (optional) → constant folding
  → dead code elimination → memory planning
        │
        ▼
  CpuBackend → ExecutablePlan → GraphExecutor::run()
```

This path remains inference-only. It has no autograd, no backward graph, no
optimizer-update opcodes and no wgpu backend; those were deleted with FN3-19
(#65). The pinned 2.x revision `a2c0759` is the historical reference for the
deleted training tree.

### Adding a 2.x op

1. Add an `Opcode` variant and constant folding/type/shape behaviour.
2. Implement lowering and dispatch in `src/backend/cpu/`.
3. Add IR, compiler and CPU tests; if the op is imported, extend
   `src/onnx/converter.rs` and cover it with a round-trip test.

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
each stage, the arena size when the executor reports one, and a SHA-256 of
every output. `--require-quiet N` refuses to measure when the load average is
above `N`: a benchmark taken on a busy host is noise, and the number is recorded
in the report so a reader can tell the two apart.

`tools/perf/compare_perf.py --base A --head B` prints the transitions and
fails when the median grows past 15 % or the peak RSS past 20 %, or when an
output digest changed. `tools/ci/perf-lane.sh` runs both, and skips with an
explanation when the model or the baseline is absent, because a model is not
committed to this repository.

The Python facade exposes no quantizing builder yet, so the harness refuses
`--dtype u8|i8` instead of reporting an F32 number under a quantized label.
Those dtypes are reachable from Rust through `CompileTarget`; measuring them
from Python needs a facade first.

## Tests

| Area | Command |
|---|---|
| Everything the CPU lane runs | `cargo test --lib -- --test-threads=1` then `cargo test --no-fail-fast --tests` |
| AVX2 kernel parity | `cargo test --lib parity` |
| 3.0 contracts | `cargo test --test fn3_contracts` |
| 3.0 CPU reference | `cargo test --test fn3_cpu_reference` |
| import / format / planner / session | `cargo test --test fn3_{import,model_format,planner,session}` |
| end-to-end (pinned model) | `cargo test --test fn3_e2e` |
| IR shapes and memory planning | `cargo test --test {shape_inference_tests,memory_planning_tests,arena_liveness}` |
| compiler passes | `cargo test --test {compiler_edge_cases,residual_add_norm,concat_to_conv_eligibility}` |
| Vulkan (physical GPU) | `cargo test --features vulkan --test fn3_vulkan_smoke` |
| Vulkan library unit tests | `VK_ICD_FILENAMES=… FN3_VK_TEST_SOFTWARE=1 cargo test --no-default-features --features vulkan --lib -- --test-threads=1` |
| 2.x CPU/quantized pipeline | `cargo test --test quantized_pipeline -- --test-threads=1` |
| Python binding | `maturin develop --release && pytest tests/ -m "not slow"` |

The CPU lane runs the library tests and then every integration target; the
narrow commands above are for iterating on one area, not a substitute for the
lane. `cargo test --lib parity` is the only place a pointwise AVX2 kernel is
checked against its scalar reference, so a new SIMD activation needs a row
there, not a bespoke test. The Vulkan library unit tests need a loader and an
ICD, which only the CI-only software lane installs, so that lane runs them
after the smoke test; `vulkan-build` compiles them with `--no-run` and executes
nothing.

The pinned model is never committed; tests print a skip reason when it is
absent. Do not weaken a tolerance to hide a regression, and do not delete a
parity row that disagrees: the disagreement is the finding.

## Benchmarks

The 3.0 Rust benchmark harness was removed with the 2.x surface (tag
`archive/3.0`). The v3 harness is issue #106: fixed runner, cold and
warm latency, p50/p95/p99, memory and a same-machine ORT control.
`tools/perf/` keeps the report schema and the comparison budgets until
that harness replaces them.