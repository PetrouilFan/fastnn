# CPU providers: the scalar reference and the optional oneDNN engine

- **Status:** implemented, opt-in (`cpu-onednn`)
- **Date:** 2026-11-22
- **Authority:** [ADR-0001](adr-0001-one-runtime.md). The live operator matrix is
  [SUPPORTED.md](../../SUPPORTED.md).

This document describes the two engines a `Session` can execute a CPU plan with,
why one of them is a library and the other is not, and exactly what the library
is allowed to decide. It is the reference for the numbers in
[§7](#7-measurements) and for the invariants in `src/fn3/AGENTS.md`.

## 1. Two engines, one plan

A `Plan` is built before a session exists. Each `Segment` carries the engine the
plan chose:

```text
Segment { op, engine: Scalar | OneDnn, inputs, outputs, params, loc }
```

The choice is made once, at plan-build time, by offering every segment to a
`SegmentProvider` (`backend::SegmentProvider`). The planner contains no feature
test: the same graph produces the same scalar plan in either build, and only the
provider the session passes in decides where a segment runs. The optional
provider is `backend::cpu::onednn::OneDnnCapability`; its claim is a pure
function of `(op, operand shapes, dtypes, "is this operand constant?", params)`.

A segment the provider declines is recorded in `Plan::fallbacks` with the
provider's reason and runs on the **scalar kernels**, whose result is the
reference result. Nothing is silently re-run on another engine, and a forward
never re-decides: `Session::execute` dispatches on `segment.engine`.

```text
                      plan build                    forward
graph ──► Plan ──► segment.engine = Scalar ──► runtime::kernels (NCHW scalar)
                   segment.engine = OneDnn ──► prepared blocks on session pool
                   plan.fallbacks: [ (segment, op, reason) ]
```

With `--no-default-features` (or without `--features cpu-onednn`) the oneDNN
module is not compiled, no C++ library is linked, and `SegmentEngine::OneDnn`
cannot appear in a plan; `Session::from_plan` rejects such a plan rather than
running it scalar.

## 2. Why oneDNN is sequential in this build

oneDNN can be built with several CPU threading runtimes (`SEQ`, `OMP`, `TBB`,
`THREADPOOL`). `onednn-src` 0.1.24 sets no `ONEDNN_CPU_RUNTIME`, and oneDNN's
CMake default is **OMP** when the compiler supports OpenMP. On this host that
default produced a `libdnnl.a` whose generated `dnnl_config.h` said
`DNNL_CPU_RUNTIME DNNL_RUNTIME_OMP`: every primitive would spawn its own OpenMP
team inside every call.

That is the wrong ownership model here. ADR-0001 §5 makes the session own its
threads, and the plan's parallelism is *segments and blocks*, not library teams.
The feature therefore pins the build to the sequential runtime:

- `.cargo/config.toml` sets `CMAKE_TOOLCHAIN_FILE` to
  `tools/cmake/onednn-seq.cmake`, which sets `ONEDNN_CPU_RUNTIME=SEQ` (the
  `cmake` crate forwards that environment variable; it has no per-define
  environment hook);
- `cargo` does not rerun a dependency build script when that environment
  variable changes, so a tree that previously built oneDNN with OMP needs
  `cargo clean -p onednn-src` once;
- an existing `CMAKE_TOOLCHAIN_FILE` in the environment wins, which is why the
  engine does **not** assume SEQ: it reads `dnnl_version()->cpu_runtime` and
  reports it. A foreign threaded build still produces correct output; it runs
  the blocks serially instead of nesting a team per block
  (`Session::one_dnn_report().parallel_blocks == false`).

`tests/fn3_onednn.rs` asserts the runtime is `seq` and prints the version.

## 3. What the engine claims

Claimed, all `f32`:

| Op | Shape | Notes |
|---|---|---|
| `Conv` | 4-D NCHW | groups, strides, dilations, asymmetric explicit pads (after `auto_pad`), optional bias |
| `MaxPool` | 4-D NCHW | explicit pads, strides |
| `AveragePool` | 4-D NCHW | explicit pads, strides, `count_include_pad` |
| `GlobalAveragePool` | 4-D NCHW | expressed as average pooling over the full input plane |
| `MatMul` | 2-D | the decoder `Gemm` form; no broadcast batch, no bias operand |

A weight operand (and a `Conv` bias) must be a **read-only model payload**
(`SegmentOperand::constant`, i.e. an initializer or a folded `Constant`). The
engine prepacks weights at session open; a weight produced by an earlier segment
has no bytes then, so such a segment falls back with a reason instead of being
prepacked as zeros.

Everything falls back to scalar for a reason that is recorded. The common ones:

| Segment | Reason recorded |
|---|---|
| `Conv` rank 3/5 (1-D/3-D spatial) | "claimed for 4-D NCHW/NCHW/NCHW operands only" |
| dynamic (arena) weight or bias | "prepacks constant weights at session open" |
| `MaxPool`/`AveragePool` with `ceil_mode = 1` | "oneDNN pooling has no ceil rule" |
| pooling with dilation | "claims the undilated pool forms only" |
| `MatMul` rank 3+ | "claimed for 2-D operands only" |
| any other op (`Clip`, `Sigmoid`, `Add`, `Resize`, `Pad`, `TopK`, …) | "every other op runs on the scalar kernels" |

On YOLOv8n every `Conv` and `MaxPool` segment is claimed; the plan report in
`tests/fn3_onednn.rs` prints the counts, the fallback list and the prepacked
bytes for the pinned graph.

## 4. Threading: output-channel blocks, not oneDNN

The engine splits a segment at plan time from the session's worker count:

```text
plain conv       channels [0, c) | [c, 2c) | ...        source = full input
grouped conv     whole groups only                      source = channel submemory per block
depthwise conv   one channel (one group) per block
pooling          channels [0, c) | [c, 2c) | ...
matmul           result columns [0, n) | [n, 2n) | ...
```

**Why whole groups for a grouped convolution.** The ONNX weights are
`[O, C/g, kh, kw]`. A block that spans whole groups is a contiguous slice of
that payload and can be described as a submemory. A boundary *inside* a group
would need a weight submemory of shape `[1, n, C/g, ...]` starting at a row
offset, which is expressible as a submemory but produces a one-group
convolution whose input is a channel slice — a second code path with no payoff
on the pinned graphs. Depthwise convolutions (`O/g == 1`) are the common case
where a group *is* one channel, so they still split per channel.

**Block count.** `blocks = min(workers, units / min_units)` with `units` the
indivisible pieces above and `min_units` the larger of:

- `ceil(8 channels / channels-per-unit)`;
- the smallest unit count whose output payload is at least 16 Ki `f32`
  elements (64 KiB) — about one cache line per worker.

A problem below both floors stays one block, and one worker is always one
block. The split is deterministic: the only host input is the worker count,
which the plan records.

**Descriptors.** Each block binds:

- its **weights** as `dnnl_memory_desc_create_submemory` of the ONNX payload,
  offsets `[o0, 0, 0, 0]` for a plain convolution (element offset
  `o0 * (C/g) * kh * kw`) or `[g0, 0, 0, 0, 0]` for a grouped one;
- its **destination** as a submemory of the output slot, offsets
  `[0, o0, 0, 0]` (element offset `o0 * H_out * W_out` per batch row);
- its **source** as the full input slot for a plain convolution (no spatial
  split, no halo), or a channel submemory for a grouped block.

Source and destination stay in plain NCHW (`dnnl_nchw`); only the weight
descriptor is offered as `dnnl_format_tag_any`, because oneDNN's blocked weight
layouts are the point of using it.

`SessionThreads::run_blocks` executes the blocks with the session's own pool
(`rayon::ThreadPool`, exact count, `fn3-worker-*`). Blocks write disjoint
destination ranges and each owns its primitive, stream and scratchpad, so they
are independent. With a non-sequential oneDNN build the same blocks run
serially and the library uses its own threads.

**Scratchpads.** Primitives are created with
`dnnl_scratchpad_mode_user` and each block owns one scratchpad buffer. oneDNN's
default library scratchpad is shared per primitive and, with
`DNNL_ENABLE_CONCURRENT_EXEC=OFF` (the default), tied to the thread that created
the primitive; a block executes on whichever pool worker Rayon assigns, so the
engine supplies the scratchpad instead. Distinct blocks use distinct buffers,
which is the documented condition for concurrent execution.

## 5. What is prepacked, and when

All of it happens at `Session::from_plan`, once; no forward reorders or copies a
weight.

| Step | When | Buffer |
|---|---|---|
| copy the weight payload into an aligned staging buffer | session open | temporary, dropped after the segment |
| create the primitive descriptor with the weight descriptor `any` | session open | — |
| clone the descriptor oneDNN chose (`dnnl_memory_desc_clone`) | session open | — |
| reorder the block's weights into it (or a plain copy if it chose the ONNX layout) | session open | session-owned, per block |
| copy the constant bias slice | session open | session-owned, per block |
| allocate the scratchpad | session open | session-owned, per block |
| compile the primitive | session open | session-owned, per block |

The result is `pub(crate) struct OneDnnEngine { prepared: Vec<Option<PreparedSegment>>, .. }`
indexed by plan position: a deterministic cache with no hash iteration order.
`Session::one_dnn_report()` reports the prepared segment counts, the total
prepacked bytes and how many weight blocks needed a real reorder (the rest were
already in oneDNN's chosen plain layout and were only copied). On the pinned
YOLOv8n graph oneDNN chose the plain layout for all 149 weight blocks, so the
report says `0 reordered` and the 12.6 MB of prepacked weights are aligned
copies; a graph whose block shape oneDNN wants blocked (another ISA, another
algorithm) reorders once at open instead, and the report counts it.

## 6. Buffers, lifetimes and errors

oneDNN borrows a host pointer only for the duration of `dnnl_primitive_execute`.
A block's memory objects are created with `DNNL_MEMORY_NONE` (no buffer), and
immediately before each call the engine installs:

- the executor's live arena windows for source and destination (disjoint by the
  plan, `&[u8]`/`&mut [u8]` held across the call);
- the block's own prepacked weight, bias and scratchpad buffers.

Nothing caches an arena pointer between forwards, so no `&mut` provenance has to
outlive a later reborrow of the arena. The engine is a session field declared
before the arena, the weights and the aligned weight copies, so it is dropped
first; its buffers outlive every call it makes.

Every oneDNN entry point returns `dnnl_status_t`; every call goes through one
`check` that maps a non-success status to `Fn3Error::ProviderFailure { provider:
"oneDNN", call, status, detail }`. There is no `unwrap` on a status. Destructors,
which run in `Drop` where no error can be returned, assert success in debug
builds and document why destroying a valid handle cannot fail.

The FFI is hand-written (`backend/cpu/onednn/ffi.rs`): opaque handle types from
the C header, the enums the engine passes, and one RAII wrapper per handle. No
bindgen, no generated file, no header path in the build.

## 7. Measurements

Measured with `tests/fn3_cnn_perf.rs` (Rust) and
`tools/parity/ort_reference.py --time` (ONNX Runtime), one process per
configuration, `--release`, 3 warm-ups + 30 timed forwards, YOLOv8n
`[1, 3, 320, 320] -> [1, 84, 2100]`.

<!-- MEASUREMENTS -->
### Machine and method

| | |
|---|---|
| CPU | AMD Ryzen 7 3700X, 8 physical cores / 16 logical, 32 GiB RAM |
| OS | Linux 6.x, shared desktop (load average 7–12 during the runs) |
| Rust | rustc 1.100.0 |
| oneDNN | 3.14.1, `DNNL_CPU_RUNTIME=SEQ` |
| ONNX Runtime | 1.32.0, `CPUExecutionProvider`, `intra_op_num_threads = N` |
| Build | `--release`, `--no-default-features --features cpu-onednn,parallel` |
| Work | YOLOv8n `[1,3,320,320] -> [1,84,2100]`, 3 warm-ups + 30 timed forwards |

The host is a shared desktop, not a quiet benchmark machine: three repetitions
of each configuration were taken and the table reports the **lowest median of
the three** (with that run's p95 and `VmHWM`), because scheduling interference
only ever makes a forward slower. A repeat during a busier period gave, for
example, 967 ms/431 ms for the scalar 4/8-thread configurations, so compare
numbers within this table rather than across sessions.

### Result

| Engine | Threads | Median ms | p95 ms | Peak RSS MiB |
|---|--:|--:|--:|--:|
| scalar | 1 | 1377.73 | 2319.10 | 43.8 |
| scalar | 4 | 520.28 | 860.77 | 44.0 |
| scalar | 8 | 386.02 | 449.46 | 44.2 |
| oneDNN | 1 | 73.45 | 85.27 | 79.1 |
| oneDNN | 4 | 82.01 | 88.37 | 91.6 |
| oneDNN | 8 | 74.28 | 76.47 | 96.4 |
| ORT | 1 | 23.18 | 25.50 | 136.9 |
| ORT | 4 | 10.19 | 11.30 | 134.1 |
| ORT | 8 | 8.74 | 10.01 | 966.4 |

The oneDNN engine makes a forward **18.8× faster than the scalar reference at
1 thread** (1378 → 73 ms) and **5.2× faster at 8 threads** (386 → 74 ms). It
does **not** reach the ≤2×-ORT goal: at 8 threads it is **8.5× ORT** (74 vs
8.7 ms); the closest configuration is 1 thread at 3.2×. That is a negative
result and it is not restated as a win.

### Where the time goes

A temporary instrumented release build (one `Instant` per segment, summed per
`OpId`; not in the shipped tree) at 8 threads on the oneDNN session:

| Part | ms | Engine |
|---|--:|---|
| `Conv` (64 segments, 149 blocks) | 27.2 | oneDNN |
| `Split` (8) | 22.9 | scalar |
| `Transpose` (2) | 9.1 | scalar |
| `Sigmoid` (58) | 7.9 | scalar |
| `Softmax` (1) | 3.9 | scalar |
| `MaxPool` (3) | 2.4 | oneDNN |
| `Mul`, `Concat`, `Resize`, … | 3.1 | scalar |
| total | ~76.5 | |

Two facts follow:

1. **The scalar glue now dominates**: 48 ms of the 76 ms forward is split,
   transpose, sigmoid, softmax and the other scalar segments. Those are outside
   the engine and run on the scalar path unchanged; no amount of oneDNN tuning
   reaches 2× ORT while they cost 5.5× ORT's whole model.
2. **Output-channel blocks barely scale the convolution part** (32.8 ms at one
   block per convolution → 27.2 ms at 149 blocks). Each block re-reads the full
   input plane set, so eight blocks move roughly eight times the activation
   bytes; on a memory-bound graph the extra traffic cancels most of the parallel
   gain. Spatial splitting with a halo would fix the bandwidth, but the phase-3a
   contract is explicitly "full input as source (no halo)", so it is a documented
   limit, not an implementation accident. It is also why 4 threads is slower
   than 1 here: the same bandwidth, more contending workers.
3. The prepacked weights are 12.6 MB and explain most of the engine's RSS
   increase (79–96 MiB vs 44 MiB scalar). ONNX Runtime's 966 MiB at 8 threads is
   its arena, not the model.

### Correctness of these numbers

Measured by `tests/fn3_onednn.rs` on the same machine:

| Comparison | Gate | Measured |
|---|---|---|
| oneDNN vs scalar, same plan | `|d| <= 1e-4 + 1e-4|ref|` | max_abs 3.5e-4, max_rel 7.2e-6 |
| oneDNN vs ONNX Runtime | `|d| <= 2e-4 + 1e-3|ref|`, rel `<= 1e-3` | passes |
| oneDNN `Fixed(4)` vs `Fixed(1)` | same 1e-4/1e-4 gate | **not bitwise**; max_abs 3.8e-4, max_rel 7.1e-6 |
| oneDNN `Fixed(8)` vs `Fixed(1)` | same 1e-4/1e-4 gate | **not bitwise**; max_abs 2.9e-4, max_rel 1.5e-5 |
| scalar `Fixed(1/4/8)` | bitwise | bitwise identical |
| oneDNN vs scalar plan report | | 240 segments: 67 oneDNN (64 conv, 3 pool), 173 scalar, **0 fallbacks** |

The oneDNN outputs are **not bitwise identical across thread counts**: the
worker count changes the block count, a different block shape can select a
different oneDNN kernel, and its accumulation order is then different. The test
asserts the numerical bound above and prints `bitwise false` rather than
claiming a bitwise result it did not observe.


## 8. Known limits

- 4-D NCHW convolution and pooling only; 1-D and 3-D spatial graphs fall back.
- 2-D `MatMul` only; batched matmul (attention) falls back.
- `f32` only; no `bf16`/`f16`/quantized path.
- Pooling with `ceil_mode` or dilation falls back (the ONNX ceil rule allows
  windows past the padded region, which oneDNN pooling does not express).
- Weights and biases must be read-only model data (prepacked at open).
- No activation fusion yet: `Sigmoid`, `Clip`, `Add`/`Mul` run as separate
  scalar segments. The FFI has the post-ops surface; the planner does not fuse
  in phase 3a.
- The engine is a per-session object; there is no global primitive cache and no
  process-wide oneDNN setting is touched.
