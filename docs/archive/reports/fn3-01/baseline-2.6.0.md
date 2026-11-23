# FastNN 2.6.0 inference baseline — FN3-01

Baseline frozen for the FastNN 3.0 rewrite. All work was done in
`/home/petrouil/Projects/github/fastnn-wt-47-baseline` on branch
`fn3/47-baseline`.

## Identity

| Item | Value |
|---|---|
| Branch | `fn3/47-baseline` |
| Base SHA | `a2c0759fff8bdcb7c943e53442f3382b86e59b66` |
| HEAD at measurement | `a2c0759fff8bdcb7c943e53442f3382b86e59b66` |
| `git describe` | `v2.6.0-71-ga2c0759` |
| Crate / wheel version | `2.6.0` |
| Reference checkout | `/home/petrouil/Projects/github/fastnn-wt-baseline` (read-only) |

`a2c0759` is 71 commits **after** the `v2.6.0` tag (`e63eda8`). The
`docs/roadmap/llm-runtime.md` M0 audit names `v2.6.0` at `e63eda8`; its
numerical claims are therefore not directly comparable to this SHA. This
matters for the tiny GPT-2 result below.

`docs/roadmap/fn3-ledger.md` does not exist at this SHA or on the sibling
`fn3/*` branches. Environment facts in this report were measured directly.

## Environment

Full record: `environment.txt`.

| Item | Value |
|---|---|
| OS | Arch Linux rolling |
| Kernel | `Linux 7.2.14-arch1-1 x86_64` |
| CPU | AMD Ryzen 7 3700X, 8 cores / 16 threads |
| RAM | 31 GiB |
| Rust | `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `cargo 1.100.0` |
| System Python | 3.14.9 (no torch/onnx/onnxruntime) |
| Measurement venv | CPython 3.13.15, NumPy 2.4.2, pytest 9.2.0 |
| ONNX stack in venv | onnx 1.23.2, onnxruntime 1.32.0 |
| Build tools | uv 0.13.7, maturin 1.16.0 |
| GPU | NVIDIA GeForce GTX 1080 Ti, 11 GiB, driver 580.196.07 |
| Vulkan | instance 1.4.363, device API 1.4.312 |
| Vulkan layers | `VK_LAYER_KHRONOS_validation` |
| Vulkan ICDs | `nvidia_icd.json`, `lvp_icd.json`, `radeon_icd.json` |
| Software ICD | lavapipe / llvmpipe, Mesa 26.2.7, LLVM 23.1.4 |
| Unavailable | system torch, onnx, onnxruntime; `optimum-cli`; `transformers`; nvcc |
| Display | `DISPLAY` / `WAYLAND_DISPLAY` unset; headless Vulkan works |

## Test matrix (real runs)

All commands were run from the worktree. `CARGO_BUILD_JOBS=4` was used for
Cargo builds. Logs are under `logs/`.

| Gate | Command | Result |
|---|---|---|
| Rust library | `cargo test --lib -- --test-threads=1` | **544 passed, 0 failed, 0 ignored** |
| Quantized pipeline | `cargo test --test quantized_pipeline -- --test-threads=1` | **21 passed, 0 failed** |
| Optimizer | `cargo test --test optim_test -- --test-threads=1` | **11 passed, 0 failed** |
| Autograd gradients | `cargo test --test autograd_gradient_checks -- --test-threads=1` | **8 passed, 0 failed** |
| Arena telemetry | `cargo test --test cpu_arena_telemetry -- --test-threads=1` | **7 passed, 0 failed** |
| Python, not slow | `uv run pytest tests/ -m "not slow" -v` | **369 passed, 1 skipped, 2 deselected, 1 xfailed** in 3.44 s |
| I/O / serialization | `pytest tests/test_io.py -v` | **8 passed** |
| ONNX import/shape | `pytest tests/test_onnx.py tests/test_onnx_shape_inputs.py -v` | **89 passed** |
| AOT prepared/profile | `pytest tests/test_aot_executor_prepared_execute.py tests/test_aot_executor_prepared_stats.py tests/test_aot_executor_profile.py -v` | **28 passed** |
| Formatting | `cargo fmt --check` | exit 0 |
| WGPU (unquantized) | `cargo test --features gpu --test wgpu -- --ignored --test-threads=1` | **16 passed, 10 failed** |
| WGPU (quantized) | `cargo test --features gpu --test wgpu_quantized -- --ignored --test-threads=1` | **0 passed, 4 failed** |

The single pytest skip is
`tests/test_v080_wgpu_oom.py::test_gpu_detection` (the Python facade reports no
GPU device). The two deselected tests carry the `slow` marker.

The 21 quantized-pipeline tests include grouped W4A8 compile/execute for all
group sizes, W4A8 exclusion, shared-projection activation reuse, rank-3
activations, and W4A8 output-constant rejection, plus U4/U8/F4 MatMul and
Conv2d end-to-end tests. `cargo test --lib` contains 160 packed/quantized kernel
and dtype tests, including `grouped_w4a8_per_token_matches_exact_quantized_operand_reference`.

## Inventory of the working 2.x inference path

### CPU eager tensor path

- Entry points: `fastnn.tensor` / `fastnn.Tensor` in `fastnn/__init__.py`,
  `src/tensor/`, `src/lib.rs`.
- It runs on this machine and is covered by the 369-test Python suite.
- This is an eager, autograd-capable path. It is not the AOT inference path and
  should not be the 3.0 inference architecture.

### CPU AOT / prepared-plan path

- Entry points: `GraphBuilder` (`src/ir/builder.rs`) →
  `GraphExecutor::compile_with_target` / `execute`
  (`src/backend/executor.rs`) → `CpuBackend` (`src/backend/cpu/mod.rs`).
- `prepared-plan` is a **default feature** (`Cargo.toml`). Prepared execution is
  reached through `execute_prepared_no_copy` and `prepare_executable_plan`
  (`src/backend/prepared.rs`).
- It runs: all Rust gates pass; the CLI compiles and executes a `.fnnc` plan;
  the 28 prepared/profile pytest tests pass.
- Dynamic-shape execution is wired through `ShapeEnv` and
  `MemoryPlan::tighten`. That path is where the stock tiny GPT-2 build fails
  (below).

### WGPU path (`--features gpu`)

- Entry point: `WgpuBackend` (`src/backend/wgpu/mod.rs`), selected through
  `GraphExecutor`.
- It initializes on this machine and selects adapter 0: Vulkan NVIDIA GTX
  1080 Ti. The independent probe is `raw/gpu-adapter-probe.json`.
- It is **partially working**: 16/26 WGPU tests pass. Failures are shader bugs,
  wrong reductions, an embedding index error, and a fallback geometry error.
- Quantized WGPU cannot even create pipelines: shaders need 5 storage buffers,
  but `GpuContext::new` requests `wgpu::Limits::downlevel_defaults()` (4).
- Exact failures are in `raw/gpu-evidence.json` and `logs/wgpu-tests.log`.

### ONNX import

- Entry point: `fastnn.convert_from_onnx` / `fastnn.io.onnx.import_onnx`
  (Python) → v3 `.fnn` → `fastnn.build_model_from_fnn` (Rust converter).
- The Python importer runs here. The tiny random GPT-2 artifact imports in
  0.73 s as 257 nodes / 48 initializers.
- 89 ONNX tests pass, including external-data validation, typed constants, and
  runtime shape-tensor subgraphs.
- **Compiling the stock tiny GPT-2 artifact fails** at this SHA; see below.
- M1/M2 blocker status at this SHA:
  - Strict unsupported-op errors: **fixed** in the Rust converter
    (`unsupported ONNX operation ...`, unit test present).
  - Dynamic Reshape: **honest failure, not supported**. Symbolic `-1` inference
    is only implemented when the known target product is 1. `[-1, 128]` with a
    symbolic input product fails with `Reshape inferred dimension requires
    unsupported symbolic division` (`src/onnx/converter.rs`,
    `resolve_reshape_dims`).
  - Dynamic Split: **honest failure** for a symbolic axis extent; constant
    split sizes are supported.
  - Slice opset-14 tensor inputs: **implemented** through `runtime_slice`;
    static Slice accepts step 1 only.
  - Typed constant round-trip: **fixed** for F32/I64/I32/Bool. F16/BF16
    initializers are rejected rather than silently converted.
  - Persistent state: **PyO3-only** `AotExecutor.configure_state/prefill/decode`
    exists and is exercised by synthetic-graph tests in
    `tests/test_onnx_shape_inputs.py`, but it is not a reusable Rust session
    contract and the tiny GPT-2 graph cannot be compiled to use it.
  - External data: **validated** in `fastnn/io/onnx.py`
    (`_validate_external_data`, path-escape and 64 GiB aggregate checks).

### `.fnnc` and `.fnn` serialization

- `.fnnc`: `ExecutablePlan::save/load` uses raw bincode with resource limits
  (`src/backend/mod.rs`), accompanied by a JSON `MemoryPlan`.
  `Runtime<B>::save/load/run` (`src/backend/runtime.rs`) executes it.
  There is **no magic or format version** in the `.fnnc` payload.
- Verified end-to-end with `tools/baseline/fixtures/relu_graph.json`:
  compile → info → run → bench all exit 0. The 4 MB Relu arena averaged
  981.59 µs over 100 iterations.
- `.fnn`: magic `FNN\0`, format version 3, typed tensors
  (`src/io/serialize.rs`, `fastnn/io/__init__.py`). Version 3 supports
  F32/I64/I32/Bool plus packed I4/I8/F4/F8/F8R/F16 tags. Python loads packed
  tags only through Rust; `build_dag_model` explicitly rejects packed `.fnn`
  parameters from Python.
- I/O round-trips, external-data rejection, and model save/load pass
  `tests/test_io.py` (8 tests).

### Quantization paths

- Integer W4A8 grouped per-token: `gemm_cpu_flat_i8_i4x8_grouped_per_token`,
  `PerTokenI8Activations`, `quantize_i8_per_token_symmetric`
  (`src/backend/cpu/microkernels/gemm.rs`); `PackedTensor::from_f32_k_grouped_i4`
  (`src/packed_tensor.rs`); `quantize_matmul_weights_k_grouped_i4`
  (`src/compiler/passes/quantization.rs`). Tested by 5 grouped W4A8 pipeline
  tests and the Python `test_w4a8_activation_audit.py` cases.
- I4/I8 and U4/U8: `src/dtypes/i4x8.rs`, `i8x4.rs`, `u4x8.rs`, `u8x4.rs`,
  `src/backend/cpu/swar.rs`, `packed_gemm.rs`, `packed_conv.rs`. Tested by
  U4/U8 MatMul/Conv2d end-to-end tests and SWAR/dtype unit tests.
- F4/F8: `src/dtypes/f4x8.rs`, `f8x4.rs`, `f8x4r.rs`, packed kernels and
  `test_matmul_f4x8_roundtrip`. Tested on CPU; not GPU.
- All CPU quantized tests pass. GPU quantized tests do not start due to the
  storage-buffer limit.

### Packaging / CLI

- `maturin develop --release` succeeds in 1 m 13 s using the pyproject feature
  set `pyo3/extension-module, simd, parallel, fusion-forward, prepared-plan`.
  Installed editable extension reports fastnn 2.6.0.
- `fastnn-runtime` (`--features cli`) builds in release in 1 m 40 s and its
  `compile`/`info`/`run`/`bench` subcommands work.
- The Python `fastnn-convert` console script was not exercised.
- `uv run pytest` works against the editable install with no extra environment
  variables.

## Baseline measurements

### Whole-model tiny GPT-2: stock artifact does not compile

Artifact:

- URL: `https://huggingface.co/onnx-community/tiny-random-gpt2-ONNX/resolve/main/onnx/model.onnx`
- SHA-256 `e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1`
- 27,897,974 bytes; HF repo SHA `90f61e71…`, last modified 2025-10-03.

Reproduction:

```bash
curl -sL -o /tmp/opencode/fn3-01/tiny-random-gpt2.onnx \
  https://huggingface.co/onnx-community/tiny-random-gpt2-ONNX/resolve/main/onnx/model.onnx
.venv/bin/python -c "import fastnn; fastnn.convert_from_onnx('/tmp/opencode/fn3-01/tiny-random-gpt2.onnx','/tmp/opencode/fn3-01/tiny-random-gpt2.fnn')"
.venv/bin/python -c "import fastnn; fastnn.build_model_from_fnn('/tmp/opencode/fn3-01/tiny-random-gpt2.fnn', symbolic_dim_bounds={'batch_size':1,'sequence_length':8,'past_sequence_length':40})"
```

Import succeeds. Build fails with 200 conversion errors; the first is:

```text
node '/transformer/h.0/attn/c_attn/Reshape' (Reshape): Reshape inferred dimension requires unsupported symbolic division
```

A static-shape workaround (`tools/baseline/tiny_gpt2_static.py`) rewrites the
nine `-1` Reshape targets and the recorded symbolic shapes for one
`(tokens, past)` pair. This is **not** evidence that the dynamic path works.
With that workaround:

| Workload | FastNN vs ONNX Runtime |
|---|---|
| Decode 1 token, 8 cached | logits max abs **7.15e-7**, K/V max abs **2.98e-7**, argmax 11 = 11 |
| Prefill 8 tokens, 1 cached | logits **402,056 / 402,056 NaN**; layer-0 K/V finite, layer-1 K/V and logits NaN |
| Prefill 32 tokens, 1 cached | logits **1,608,224 / 1,608,224 NaN**; layer-0 K/V finite, layer-1 K/V and logits NaN |

So the static 1-token decode path matches ONNX Runtime to float32 noise, while
the static 8- and 32-token prefill paths have a real numerical defect in layer 1.
The stock dynamic path cannot be built at all, so the roadmap's 5.96e-7
three-step cache claim is **not reproducible at this SHA**. Do not use it as a
3.0 parity baseline until the converter and layer-1 defect are resolved.

### CPU prefill/decode-shaped timings

All timings are release build, F32, whole 257-node graph, static-shape
workaround, plain `forward` (not the stateful `prefill`/`decode` API), on the
AMD Ryzen 7 3700X. `warmup` and `reps` are per workload. p50/p95 are over the
recorded samples. Arena/RSS are telemetry from a post-measurement call.

| Workload | Threads | Warmup | Reps | p50 (ms) | p95 (ms) | Arena (MiB) | RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prefill 8 tokens + 1 cached | 16 | 3 | 20 | 5.475 | 5.807 | 53.1 | 348.6 |
| Prefill 32 tokens + 1 cached | 16 | 3 | 20 | 12.582 | 15.059 | 59.0 | 478.6 |
| Decode 1 token after 8 cached | 16 | 3 | 30 | 3.636 | 3.801 | 51.4 | 544.7 |
| Decode 1 token after 16 cached | 16 | 3 | 30 | 4.107 | 5.388 | 51.4 | 618.4 |
| Decode 1 token after 32 cached | 16 | 3 | 30 | 3.919 | 4.087 | 51.5 | 692.1 |
| Prefill 8 tokens + 1 cached | 1 | 3 | 15 | 6.576 | 11.835 | — | — |
| Prefill 32 tokens + 1 cached | 1 | 3 | 15 | 12.652 | 14.255 | — | — |
| Decode 1 token after 8 cached | 1 | 3 | 15 | 3.910 | 4.227 | — | — |
| Decode 1 token after 16 cached | 1 | 3 | 15 | 4.154 | 5.161 | — | — |
| Decode 1 token after 32 cached | 1 | 3 | 15 | 4.151 | 4.412 | — | — |

Peak process RSS for the 16-thread benchmark run: **708,316 KiB**.
Every model variant reports 302 instructions after graph optimization.
Model build from `.fnn` is ~0.19 s.

Commands:

```bash
# 16 threads (Rayon default)
.venv/bin/python tools/baseline/bench_tiny_gpt2.py \
  --source-fnn /tmp/opencode/fn3-01/tiny-random-gpt2.fnn \
  --json docs/reports/fn3-01/raw/cpu-prefill-decode.json

# 1 thread
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python tools/baseline/bench_tiny_gpt2.py \
  --source-fnn /tmp/opencode/fn3-01/tiny-random-gpt2.fnn \
  --prefill-tokens 8 32 --decode-pasts 8 16 32 \
  --prefill-reps 15 --decode-reps 15 \
  --json docs/reports/fn3-01/raw/cpu-prefill-decode-1thread.json
```

Raw data: `raw/cpu-prefill-decode.json`,
`raw/cpu-prefill-decode-1thread.json`,
`raw/cpu-prefill-decode-parity.json`.

## GPU evidence

The backend is real but incomplete:

| Check | Result |
|---|---|
| Adapter enumeration | 3 adapters: Vulkan 1080 Ti, Vulkan llvmpipe, GL 1080 Ti |
| FastNN device 0 | Vulkan NVIDIA GTX 1080 Ti, driver 580.196.07 |
| Minimal compute shader | passes on both Vulkan adapters |
| GL device request | fails: `Parent device is lost` |
| WGPU tests | 16 passed / 10 failed |
| WGPU quantized tests | 0 passed / 4 failed |

Confirmed failure classes:

1. Naga rejects `active` as a WGSL reserved keyword in `norm_f32` and
   `rms_norm` shaders.
2. Naga rejects `of` as a reserved keyword in the Conv2d shader.
3. Argmax shader has an invalid WGSL type.
4. Reduce-sum/mean return 3 instead of 10 / 2.5 — the reduction is wrong, not
   merely imprecise.
5. Embedding dispatch rejects its index tensor.
6. The unsupported-op fallback returns a 12-byte output where CPU returns 8.
7. Quantized pipelines need 5 storage buffers but the device is created with
   `Limits::downlevel_defaults()` (4).

To reproduce the quantized blocker on this machine:

```bash
cargo test --features gpu --test wgpu_quantized -- --ignored --test-threads=1
```

Required fix for a future GPU gate: request at least
`max_storage_buffers_per_shader_stage = 5` (the adapter exposes a larger device
profile than `downlevel_defaults`), and fix the WGSL identifiers/types before
any performance measurement is meaningful.

## Salvage list — carry into 3.0

### CPU FP32 kernels and low-level helpers

| Path | Symbols | Why reusable | Coupling cost |
|---|---|---|---|
| `src/backend/cpu/microkernels/activations.rs` | `relu_f32_avx2`, `gelu_f32_avx2`, `silu_f32_avx2`, `sigmoid_f32_avx2`, `tanh_f32_avx2`, `exp_f32_avx2`, fused add/mul variants | Scalar + AVX2 reference kernels with straightforward `&[f32]` contracts | `std::arch` + `cfg(target_arch)`; no runtime dispatch abstraction |
| `src/backend/cpu/microkernels/ops.rs` | `norm_layernorm_f32_*`, `rms_norm_f32_*`, `fused_residual_add_rms_norm_f32_*`, `reduce_sum_f32_*`, `pool_*` | Directly covers LLM LayerNorm/RMSNorm/reduction; tested | Row-size/group-size conventions are implicit |
| `src/backend/cpu/microkernels/misc.rs` | `blocked_row_matmul`, `simd_dot_product`, `transpose_f32_*`, `concat_f32_*` | Small, allocation-free FP32 building blocks | Duplicated AVX2/scalar definitions behind cfg |
| `src/backend/cpu/microkernels/gemm.rs` | `gemm_cpu_flat`, `gemm_cpu_flat_i8_i4x8_grouped_per_token`, `PerTokenI8Activations`, `quantize_i8_per_token_symmetric` | Proven packed GEMM + integer activation path | `PackedTensor`, `PackedWord`, Rayon, IR-adjacent types |
| `src/backend/cpu/sgemm.rs`, `matmul.rs` | `sgemm`, grouped compensation metadata tests | Cache-blocked FP32 GEMM and exact grouped-scale reconstruction | Unsafe intrinsics; plan-specific metadata |
| `src/backend/cpu/packed_gemm.rs` | `gemm_packed_i4x8`, `gemm_packed_f4x8`, `gemm_packed_f4x8_fused` | Integer/float packed GEMM microkernels | `PackedTensor`, per-channel scale/zero conventions |
| `src/backend/cpu/packed_conv.rs` | `conv2d_packed_i4x8`, `conv2d_packed_f4x8`, `gemm_packed_*` | Packed Conv2d/im2col kernels | Im2col layout and conv metadata coupling |
| `src/backend/cpu/swar.rs`, `src/swar/` | `i4x8_dot_packed`, `i8x4_dot_packed`, `quantize_f32_to_i4x8`, `dequantize_*` | Branch-free packed integer dot products; portable fallback | `u32` packing assumptions |
| `src/dtypes/` | `I4x8`, `I8x4`, `U4x8`, `U8x4`, `F4x8`, `F8x4`, `F8x4R`, `F16x2`, `PackedWord` | Canonical packing, max representable, dot/unpack semantics | `PackedWord` also carries WGSL generation; split that out |
| `src/packed_tensor.rs` | `PackedTensor`, `from_f32_k_grouped_i4`, `from_f32_per_channel_*` | Calibrated grouped quantization and dequantization | Large API; metadata semantics need a new contract |
| `src/backend/cpu/flash_attn.rs` | `flash_attention` | Tiled online-softmax attention with causal option; CPU-only | Takes eager `Tensor`; not an AOT opcode; no KV-cache contract |

### Arena, telemetry, scheduling

| Path | Symbols | Why reusable | Coupling cost |
|---|---|---|---|
| `src/backend/cpu/arena.rs` | `with_disjoint_output_bytes`, `with_unary_f32_slices`, `with_binary_f32_slices`, overlap handling | Solves in-place/aliasing arena dispatch safely; heavily tested | Uses `CpuBuffer` and `BufferSlice` |
| `src/backend/cpu/telemetry.rs` | `CpuTelemetrySnapshot`, `record_arena_temp_copy`, `cpu_telemetry_snapshot`, `reset_cpu_telemetry` | Low-overhead counters for allocator/copy regressions | Global atomics; one process-wide snapshot |
| `src/backend/cpu/topology.rs`, `affinity.rs` | `l2_cache_size_bytes`, `physical_core_count`, `PinnedThreadPool`, `global_pinned_pool` | Real NUMA/core topology for LLM kernels | Global pool initialization; Linux-specific assumptions |
| `src/backend/runtime.rs` | `Runtime<B>`, `load`, `save`, `run` | Minimal `.fnnc` execution contract and resource-limited load | Raw bincode format; no versioning |

### Prepared plans

| Path | Symbols | Why reusable | Coupling cost |
|---|---|---|---|
| `src/backend/prepared.rs` | `PreparedExecutablePlan`, `PreparedConstantArena`, `PackedWeightStore`, `PreparedWeightKey`, `try_prepare_matmul`, `try_prepare_conv2d`, `prepare_executable_plan` | Persistent-weight ownership, Arc-shared sessions, prepared/no-copy execution contracts | Coupled to CPU kernel names, `ExecutablePlan`, instruction indices, and current memory plan |
| `src/backend/executor.rs` | `execute_prepared_no_copy*`, `execute_prepared_fallback`, plan cache/tightening | Useful behavior and regression tests for a future executor | Monolithic 5.5k-line executor; do not copy wholesale |

### ONNX and typed I/O

| Path | Symbols | Why reusable | Coupling cost |
|---|---|---|---|
| `fastnn/io/onnx.py` | `import_onnx_to_compute_graph`, `_validate_external_data`, `_plain_dtype_name`, `_constant_param`, `_extract_attrs`, `_OP_ATTRS`, `_OP_PARAM_SLOTS`, `import_onnx` | Working importer front end with typed constants and external-data safety | Python performs graph normalization and duplicate shape inference |
| `src/onnx/converter.rs` | `OnnxConverter`, Slice/Split/Shape/Range/Expand/Where/Gather/Softmax arms, `resolve_reshape_dims`, `split_sizes_from_tensor` | Authoritative Rust conversion boundary with honest errors | Monolithic match; coupled to `GraphBuilder`; symbolic division incomplete |
| `src/onnx/opcode_map.rs` | `simple_opcode_name`, `opcode_attr_rules`, `opcode_to_onnx` | Compact ONNX↔IR name/attribute table | Only covers a subset; exporter-oriented |
| `src/io/serialize.rs` | `save_state_dict`, `save_model`, `load_model`, version 1/2/3 readers | Versioned `.fnn` magic and typed tensor round-trip | Duplicated with Python writer/reader; no shared schema |
| `fastnn/io/__init__.py` | `write_fnn_file_v3`, `read_fnn_header`, `read_fnn_parameters`, dtype tags | Python-side typed `.fnn` producer/consumer; works with Rust | Two language implementations must stay in lockstep |
| `fastnn/io/graph_builder.py` | `build_model_from_fnn`, `build_dag_model`, `_expression_capacity`, `_bound_dimension_descriptor`, `_reshape_descriptors` | Symbolic-bound and dynamic-shape metadata translation | Python duplicates compiler/shape logic |

### Fixtures, tests, and benchmark utilities

- `tests/quantized_pipeline.rs` — 21 end-to-end quantized tests, including
  grouped W4A8. High-value regression corpus.
- `tests/test_onnx_shape_inputs.py` + `tests/test_onnx.py` — 89 ONNX
  import/execute tests, including runtime Shape/Slice/Expand/Range/Split and
  external data.
- `tests/cpu_arena_telemetry.rs` — zero-copy/overlap invariants.
- `tests/test_aot_executor_prepared_execute.py`,
  `test_aot_executor_prepared_stats.py`, `test_aot_executor_profile.py` — 28
  prepared/profile tests.
- `tests/test_w4a8_activation_audit.py` — activation-aware W4A8 clipping
  reference logic.
- `tools/baseline/fixtures/relu_graph.json` — minimal `.fnnc` round-trip
  fixture.
- Tiny random GPT-2 ONNX artifact (external, SHA-256 pinned above) — useful as
  a 257-node reference graph once the converter handles its Reshape pattern.
- `scripts/gpt2_decode_benchmark.py`, `scripts/llama_decode_benchmark.py` —
  parity and latency JSON schemas, warmup/iteration contracts, ORT comparison.
- `scripts/graph_memory_profile.py`, `scripts/graph_memory_stats.py` — arena
  and memory telemetry reporting.
- `scripts/criterion_to_json.py`, `tests/benchmark_utils.py` — percentile and
  Criterion JSON utilities.
- `benches/cpu_baselines.rs` — criterion coverage for GEMM/GEMV, arena
  broadcast, rowwise, fusion, and per-token I8xI4x8.
- `benches/wgpu_inference.rs` — WGPU-vs-CPU comparison harness (currently blocked
  by quantized GPU pipeline limits).

## Reject list — do not carry

| Path / symbol | Reason |
|---|---|
| `src/autograd.rs` (`backward`, `NodeInfo`, `AutogradMeta`, `Edge`) | Training tape and backward graph; not inference. 3.0 should not keep autograd in the inference crate. |
| `src/optim/*` (`Adam`, `AdamW`, `Sgd`, `Rmsprop`, `Lion`, `Muon`) | Optimizer state and training kernels; separate concern. |
| `src/nn/*` training-oriented modules and `tests/training.rs` paths | Backward rules and training graph construction dominate; only the FP32 numerical definitions are worth mining. |
| `src/python/nn.rs` `AotExecutor` stateful block (`StateDescriptor`, `configure_state`, `forward_stateful`, `prefill`, `decode`, `state_sizes`, `state_descriptors`) | The M4/M5/M6 state machine lives in the PyO3 binding, not a Rust core contract. Rebuild as a Rust session module; do not port this 3.7k-line binding wholesale. |
| `src/backend/wgpu/*` as-is | 10/26 unquantized tests fail; all 4 quantized tests fail. Shader keywords, wrong reductions, embedding index handling, and device limits are unresolved. |
| `src/backend/cpu/mod.rs` monolith | 13k lines mixing buffer safety, lowering, dispatch, kernels, and training kernels. Salvage kernels and arena primitives, not the file structure. |
| `GraphExecutor` dynamic-shape/plan-cache implementation wholesale | 5.5k lines coupled to backend, compiler memory planning, and prepared execution. Reuse tests and contracts, redesign ownership. |
| `fastnn/io/graph_optimizer.py`, `fastnn/io/shape_inference.py` | Duplicate Python compiler passes with unproven parity against Rust. Do not carry as a second authority. |
| `.fnnc` raw bincode format | No magic/version; cannot evolve safely. Redesign before any 3.0 serialized-plan commitment. |
| Silent-fallback import behavior | The Rust converter now rejects unsupported ops. Preserve that policy; never reintroduce pass-through input substitution. |
| `scripts/gpt2_stateful_parity.py`, `scripts/gpt2_decode_benchmark.py` hard-coded GPT-2 geometries (12 layers, 12 heads, 64 head dim) | Workload-specific; replace with a generic descriptor-driven benchmark. The JSON schema and timing contract are reusable. |
| Quantized WGPU kernels until device limits/shaders are fixed | Currently cannot allocate bind groups; results would be meaningless. |

## Skipped checks and known risks

- Full `cargo test` / `--all-targets` / `--all-features`: not run; the mapped
  gates above were the requested subset. Command for a later worker:
  `cargo test --all-targets --all-features -- --test-threads=1`.
- `pytest -m slow`: not run (benchmark-marked tests). Command:
  `uv run pytest tests/ -m slow -v`.
- TinyLlama 1.1B and Qwen2.5-0.5B: not run. The 4.4 GB TinyLlama external
  payload is absent and the export environment lacks `transformers`/`optimum`.
  `scripts/prepare_llm_model.py` requires `optimum-cli`; no end-to-end model
  path was available without external weights.
- YOLO/COCO: not run; model weights are not present in the repo.
- `cargo clippy --all-targets -- -D warnings`: not run.
- Wheel build/install outside editable `maturin develop`: not run.
- WGPU performance: not measured because the quantized pipeline cannot create
  bind groups; unquantized tests are correctness-only.
- **Risk: tiny GPT-2 regression.** Stock dynamic build fails; static 8/32-token
  prefill produces NaN layer-1 K/V. The roadmap's 2.6.0 parity numbers were
  measured 71 commits earlier (`e63eda8`), so they cannot be used as a 3.0
  baseline without re-establishing them at this SHA.
- **Risk: `.fnnc` has no format version.** Any prepared-plan serialization work
  in 3.0 must define a new versioned envelope.
- **Risk: stateful API is binding-local.** The session/state contract cannot be
  reused by a Rust-only runtime without extraction.
- **Risk: Python/Rust compiler duplication.** ONNX shape handling exists on
  both sides; parity is not established for all passes.
- **Risk: GPU path is not usable for LLM work yet.** 16/26 tests pass, and the
  quantized path is blocked by a 4-buffer limit. Do not treat WGPU as an
  available 3.0 backend without a fresh gate.

## Reproduction manifest

- `raw/test-matrix.json` — machine-readable gate results and environment.
- `raw/gpu-evidence.json` — adapter, pass/fail lists, blockers.
- `raw/gpu-adapter-probe.json` — raw wgpu adapter probe.
- `raw/cpu-prefill-decode.json` — 16-thread measurements.
- `raw/cpu-prefill-decode-1thread.json` — 1-thread measurements.
- `raw/cpu-prefill-decode-parity.json` — ONNX Runtime parity for static
  variants.
- `logs/` — raw command logs, including `dynamic-build-failure.log`,
  `wgpu-tests.log`, `wgpu-quantized-tests.log`, and the pytest/Cargo logs.
- `environment.txt` — OS/CPU/Rust/Python/GPU record.
- `manifest.json` — file sizes and SHA-256 checksums.

Reproduction scripts added under `tools/baseline/`:

- `run_cargo_gates.sh` — runs the five Rust gates and records one log each.
- `gpu_probe/` — standalone wgpu adapter/compute probe.
- `tiny_gpt2_static.py` — static-shape workaround for the tiny GPT-2 artifact.
- `bench_tiny_gpt2.py` — prefill/decode-shaped benchmark and parity reporter.
- `fixtures/relu_graph.json` — minimal `.fnnc` round-trip fixture.
