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

Contracts live in `docs/roadmap/fn3-architecture.md` and
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

## Tests

| Area | Command |
|---|---|
| 3.0 contracts | `cargo test --test fn3_contracts` |
| 3.0 CPU reference | `cargo test --test fn3_cpu_reference` |
| import / format / planner / session | `cargo test --test fn3_{import,model_format,planner,session}` |
| end-to-end (pinned model) | `cargo test --test fn3_e2e` |
| Vulkan (physical GPU) | `cargo test --features vulkan --test fn3_vulkan_smoke` |
| 2.x CPU/quantized pipeline | `cargo test --test quantized_pipeline -- --test-threads=1` |
| Python binding | `maturin develop --release && pytest tests/ -m "not slow"` |

The pinned model is never committed; tests print a skip reason when it is
absent. Do not weaken a tolerance to hide a regression.

## Benchmarks

`tools/bench/` is the 3.0 reproducible harness (`cargo run --release --bin
fn3-bench` with `--no-default-features`); `benches/cpu_baselines.rs` covers the
2.x CPU kernels. Both emit JSON with the build and host identity recorded.
