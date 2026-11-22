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

## 4. Threading: blocks over one dimension, not oneDNN

The engine splits a segment at plan time from the session's worker count:

```text
plain conv       channels [0, c) | [c, 2c) | ...   source = full input
                 or rows [0, r) | [r, 2r) | ...     source = row submemory + halo
grouped conv     whole groups only                 source = channel submemory per block
depthwise conv   one channel (one group) per block source = channel submemory per block
pooling          channels [0, c) | [c, 2c) | ...
matmul           result columns [0, n) | [n, 2n) | ...
```

**Why rows exist.** A plain convolution's channel block has no channel
submemory (every block needs every input channel), so `B` channel blocks move
`B` times the input bytes. On YOLOv8n that is why phase 3a's eight blocks
scaled the convolution part only 1.2×: the activation traffic, not the
arithmetic, was the limit. A row block instead reads only the input rows its
output rows can reach, so the input moves about `(H_in + halo) / H_in` times;
consecutive blocks overlap by the dilated kernel's halo and read it twice.
That is far cheaper than re-reading the whole input per block.

**The choice**, deterministic from `(shape, params, worker count)`:

1. compute the channel split (the phase-3a rule below);
2. compute the row split: `min_rows` is the larger of the 16 Ki-element
   payload floor divided by one output row (`N * Oc * W_out` elements) and one
   dilated kernel span `ceil((kh - 1) * dilation / stride) + 1`;
   `B_r = min(workers, H_out / min_rows)` with contiguous row ranges;
3. choose rows when `B_r > 1`, `B_r >= B_c` and the row split's estimated
   input cells (`sum over blocks of C_in * rows * W_in`) are fewer than the
   channel split's (`B_c * C_in * H_in * W_in`); otherwise keep channels.

Consequences on the pinned graph: the first convolution (`3 -> 16`, 320×320)
and the detection-head convolutions with 64 output channels on small planes
split rows (the channel split cannot fill eight workers); a depthwise
convolution stays on channels because each channel block already reads only
its own input channels, so a halo would add traffic for no bandwidth win. A
one-worker session always gets one block. `Session::one_dnn_splits()` reports
the recorded axis, block ranges and source-row count per convolution;
`OneDnnReport` counts row-split and channel-split segments. The decision is
stored in the prepared segment at session open (the plan cannot carry it: it is
built before a worker count exists), and a forward never recomputes it.

**Row geometry and padding.** For output rows `[r0, r1)` the parent
convolution reads input rows `[r0*s - pad_top, (r1-1)*s + (kh-1)*d + 1 -
pad_top)` (stride `s`, dilation `d`). The block's source range is that range
clamped to `[0, H_in)`, copied into its dense staging buffer; the rows outside
the real allocation become the block's own oneDNN `padding_l`/`padding_r` on
the height axis. No block ever reads outside the source slot, and the
first/last block's boundary work is exactly the parent's padding, split between
the block and the library. The width axis and its padding are unchanged. The
destination is a dense `[batch, O, r1 - r0, W_out]` staging buffer whose rows
are copied into the output slot at `out_row_start`.

**Grouped convolutions** still split whole groups, because the weights of one
group only multiply that group's input channels; a boundary inside a group
would make each block a smaller one-group convolution with a partial weight
row, which oneDNN cannot express as a submemory of the ONNX `[O, C/g, kh, kw]`
payload. Depthwise convolutions (`O/g == 1`) therefore split one channel per
group. This is also why the row-split traffic test keeps them on channels.

**Block count for the channel split.** `blocks = min(workers, units /
min_units)` with `units` the indivisible pieces above and `min_units` the
larger of:

- `ceil(8 channels / channels-per-unit)`;
- the smallest unit count whose output payload is at least 16 Ki `f32`
  elements (64 KiB) — about one cache line per worker.

A problem below both floors stays one block, and one worker is always one
block.

**Descriptors.** Each block binds:

- its **weights** as `dnnl_memory_desc_create_submemory` of the ONNX payload,
  offsets `[o0, 0, 0, 0]` for a plain channel block (element offset
  `o0 * (C/g) * kh * kw`) or `[g0, 0, 0, 0, 0]` for a grouped one; a row block
  uses the whole weight and the row blocks of one convolution share a single
  prepacked buffer (the descriptors are compared, so a block whose kernel
  wants a different layout gets its own copy);
- its **destination** as a submemory of the output slot, offsets
  `[0, o0, 0, 0]` (channel split) or `[0, 0, r0, 0]` (row split);
- its **source** as the full input slot for a plain channel block, a channel
  submemory for a grouped block, or a dense staging buffer for a row block.
  A row block copies its clamped input rows into the staging buffer and its
  dense output rows back into the output slot. The row submemory form was
  measured first and abandoned: a `[1, 16, 21, 160]` source submemory of
  `[1, 16, 160, 160]` keeps strides that no longer spell `nchw`, and one
  3×3 block took ~100 ms against ~4.8 ms for the whole dense tensor (oneDNN
  3.14.1). The dense buffer costs one extra input read and one extra output
  write per block — O(1) input traffic, still not the O(blocks) of a channel
  split — and puts the primitive back on its JIT path (~0.17 ms per block in
  the same measurement).

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
`Session::one_dnn_report()` reports the prepared segment counts, the split
dimensions, the total prepacked bytes, the row-staging bytes and how many weight
blocks needed a real reorder (the rest were already in oneDNN's chosen plain
layout and were only copied). On the pinned YOLOv8n graph oneDNN chose the plain
layout for every weight buffer, so the report says `0 reordered` and the 12.6 MB
of prepacked weights are aligned copies; a graph whose block shape oneDNN wants
blocked (another ISA, another algorithm) reorders once at open instead, and the
report counts it. The row blocks of one convolution compare the weight
descriptor oneDNN chose and share one prepacked buffer when it matches, which is
the case for every row-split convolution on this graph; the channel blocks keep
their own (their weights are disjoint slices). The dense row staging is 22.0 MB
on this graph and is reported separately (`staging_bytes`): it is the price of
keeping oneDNN's dense `nchw` descriptors for the row split.

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
`[1, 3, 320, 320] -> [1, 84, 2100]`. The per-op table comes from the same
harness built with `--features segment-timing`: that feature adds one `Instant`
pair per segment to `Session`, and is compiled out of a normal build. The table
is the median of three groups of three forwards per op, so one scheduling
hiccup cannot move it.

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
example, 967 ms/431 ms for the scalar 4/8-thread configurations in phase 3a, so
compare numbers within one table rather than across sessions.

### Result

Phase 3b, three repetitions per configuration:

| Engine | Threads | Median ms | p95 ms | Peak RSS MiB |
|---|--:|--:|--:|--:|
| scalar | 1 | 1289.65 | 1337.99 | 44.5 |
| scalar | 4 | 483.66 | 512.88 | 44.3 |
| scalar | 8 | 340.11 | 356.99 | 44.7 |
| oneDNN | 1 | 46.96 | 49.47 | 79.5 |
| oneDNN | 4 | 45.54 | 48.63 | 116.4 |
| oneDNN | 8 | 42.90 | 46.15 | 123.7 |
| ORT | 1 | 23.18 | 25.50 | 136.9 |
| ORT | 4 | 10.19 | 11.30 | 134.1 |
| ORT | 8 | 8.74 | 10.01 | 966.4 |

Against phase 3a on the same machine (re-measured from `b94922a` before the
changes: scalar 1356.51 / 509.70 / 363.98 ms, oneDNN 76.91 / 74.23 / 72.28 ms):

| Configuration | Phase 3a median ms | Phase 3b median ms | Change |
|---|--:|--:|--:|
| scalar, 1 | 1356.51 | 1289.65 | −4.9% |
| scalar, 4 | 509.70 | 483.66 | −5.1% |
| scalar, 8 | 363.98 | 340.11 | −6.6% |
| oneDNN, 1 | 76.91 | 46.96 | **−38.9%** |
| oneDNN, 4 | 74.23 | 45.54 | **−38.6%** |
| oneDNN, 8 | 72.28 | 42.90 | **−40.6%** |

The scalar path's arithmetic is untouched, so its change is exactly the two
data-movement kernels it shares with every engine: `Split` and `Transpose` are
now memcpy-shaped, and the 8-thread scalar session gains 24 ms from that. The
`tests/fn3_cnn_run.rs` bitwise gate still passes for `Fixed(1)`, `Fixed(4)` and
`Fixed(8)`.

The oneDNN engine is now **27.5× faster than the scalar reference at one
thread** (1290 → 47 ms) and **7.9× at eight** (340 → 42.9 ms). It still does not
reach the ≤2×-ORT goal: at 8 threads it is **4.9× ORT** (42.9 vs 8.74 ms), at 4
threads 4.5×, at 1 thread 2.0×. That is a negative result for the goal and is
not restated as a win.

### Where the time goes, before and after

Both tables are the 8-thread oneDNN session on the same host. "Before" is
`b94922a` with the `segment-timing` build; "after" is this tree.

| Part | Before ms | After ms | Engine |
|---|--:|--:|---|
| `Conv` (64 segments) | 24.9 | 22.7 | oneDNN |
| `Split` (8) | 21.3 | 0.23 | scalar |
| `Transpose` (2) | 8.7 | 0.25 | scalar |
| `Sigmoid` (58) | 8.4 | 8.9 | scalar |
| `Softmax` (1) | 3.7 | 3.8 | scalar |
| `MaxPool` (3) | 2.5 | 2.7 | oneDNN |
| `Mul` (60) | 1.8 | 2.0 | scalar |
| `Concat`, `Resize`, `Slice`, `Add`, … | 1.0 | 1.3 | scalar |
| total (segment sum) | 72.8 | 42.4 | |

The "after" column is the median of three timing runs (each already a median
of three groups of three forwards); the per-op numbers move by ±10% with the
desktop's load, the two data-movement rows by much less.

Three changes and one negative result produced that:

1. **`Split`** was one `decompose` + `checked_index` + `copy_element` per
   element, with an integer division per axis in the index walk. It is now one
   `copy_from_slice` per `(output, outer block)`, with the run length and source
   stride computed once; for YOLOv8n's eight channel splits that is one memcpy
   per output. 21.3 → 0.23 ms. The element path remains for a strided view and is
   covered by a unit test against the old semantics on axes 0/1/last, uneven
   sizes, a zero-size output and an equal split.
2. **`Transpose`** was the same per-element pattern. The two transposes in the
   graph are `[0, 3, 1, 2]` (`[1,4,16,2100] -> [1,2100,4,16]`) and `[0, 3, 2, 1]`
   (`[1,2100,4,16] -> [1,16,4,2100]`), plus the `[0,2,3,1]`/`[0,2,1,3]` shapes the
   contract names. They now take a 32×32-tile transpose of contiguous cells
   (`copy_from_slice` on the innermost run); anything else keeps the element
   loop. 8.7 → 0.25 ms. This is **not** the list of permutations the phase-3b
   brief gave: the pinned file's second transpose is `[0,3,2,1]`, not
   `[0,2,1,3]`, so both were implemented and tested.
3. **`Conv`** now chooses between output channels and output rows. On this
   graph 20 of 64 convolutions choose rows at 8 workers (102 row blocks, 44
   channel-split segments and 76 channel blocks). The row shapes are the first
   layers and the small-plane detection-head layers: `3→16` at 320×320,
   `16→32` at 160×160, `32→32` and `1×1` layers at 80×80, `32→32`/`64→64` 3×3
   at 40×40. Their channel split cannot fill eight workers because the 16-Ki
   payload floor limits the block count (`Oc / ceil(16384 / (N·Oc·W))`), while
   the row split gets 5–8 blocks and reads 40–327 input rows instead of
   `blocks × H_in` (for the first layer: 327 rows versus 2 × 320, and 8 blocks
   instead of 2). Depthwise and grouped convolutions stay on channels: their
   channel blocks already read only their own channels, so a halo would add
   traffic. The dense staging buffers cost 22.0 MB (reported as
   `staging_bytes`) and raise peak RSS from 96 to 124 MiB; the row blocks of one
   convolution share one prepacked weight buffer. Measured, the row split moves
   the convolution total from 24.9 to 22.7 ms — real but far less than the
   traffic arithmetic suggests, because the remaining 44 segments are already
   closer to their FLOP bound than to their bandwidth bound, and the split adds
   one staging copy per block.
4. **`Sigmoid` was left alone.** A flat-chunk variant was implemented and
   measured: at 8 workers, 6 400-element chunks ran a `[1,32,80,80]` sigmoid in
   359–463 µs against 137–169 µs for the existing per-row split, and smaller
   chunks only approached the row result. The chunking was reverted; the scalar
   expression and its bit pattern are unchanged, and a new unit test asserts
   bitwise equality between the serial and pooled sigmoid. 8 ms for ~12 M
   evaluations is ~1.5 G exp/s and already libm/bandwidth bound, so there is no
   honest win here without fusion or changing the accuracy contract.

### Remaining gap to ORT, segment by segment

At 8 threads the forward is 42.9 ms against ORT's 8.74 ms (4.9×):

| Segment | Ours ms | What ORT does | Why we are behind |
|---|--:|---|---|
| `Conv` | 22.7 | a fused conv+SiLU kernel per layer; total ≤ 8.7 | one segment per layer, one primitive per block; 44 segments still channel-split because a row split would give fewer blocks. Even a free conv would not close the gap alone. |
| `Sigmoid` | 8.9 | fused into the convolution | 58 separate scalar segments, each with a pool entry; ~12 M `expf` calls. Removing them needs oneDNN post-ops (fusion), not a faster `expf`. |
| `Softmax` | 3.8 | one vectorised 16-wide row pass | the legacy kernel does a `Locator::index` (a checked multiply-add per axis) for every element in three passes; 33.6 Ki elements at ~107 ns each. A contiguous last-axis fast path is the obvious next fix but is outside phase 3b. |
| `MaxPool` | 2.7 | fused/near-free | three oneDNN pool segments, channel-split; the win is in fusion, not block shape. |
| `Mul`/`Add`/`Concat` | 2.9 | fused or a vectorised pointwise pass | 60 `Mul` (the SiLU multiply), 9 `Add`, 17 `Concat` run as legacy segments, one at a time, with a one-thread budget inside the session pool. |
| `Split`/`Transpose`/`Resize`/`Slice`/`Reshape` | 1.1 | near-free | now memcpy-shaped; this is the part phase 3b fixed. |

The scalar glue outside `Conv` is 19.7 ms, 2.3× ORT's **whole** forward. The
honest conclusion is that phase 3b closed the data-movement hole (73 → 43 ms)
but the ≤2×-ORT goal needs activation fusion and parallel pointwise kernels
(phase 4), not more block-split tuning. ONNX Runtime's 966 MiB peak at 8
threads is its arena; our 124 MiB is 12.6 MB of prepacked weights plus 22.0 MB
of row staging plus the 6.1 MB arena.

### Correctness of these numbers

Measured by `tests/fn3_onednn.rs` on the same machine after the phase-3b
changes:

| Comparison | Gate | Measured |
|---|---|---|
| oneDNN vs scalar, same plan | `|d| <= 1e-4 + 1e-4|ref|` | max_abs 4.4e-4, max_rel 7.6e-6 |
| oneDNN vs ONNX Runtime | `|d| <= 2e-4 + 1e-3|ref|`, rel `<= 1e-3` | passes (max_abs 4.7e-4, max_rel 1.4e-4) |
| oneDNN `Fixed(4)` vs `Fixed(1)` | same 1e-4/1e-4 gate | **not bitwise**; max_abs 2.3e-4, max_rel 5.1e-6 |
| oneDNN `Fixed(8)` vs `Fixed(1)` | same 1e-4/1e-4 gate | **not bitwise**; max_abs 2.3e-4, max_rel 5.1e-6 |
| scalar `Fixed(1/4/8)` | bitwise | bitwise identical |
| scalar vs ONNX Runtime | phase-2 gate | max_abs 2.7e-4, max_rel 1.4e-4 |
| plan report | | 240 segments: 67 oneDNN (64 conv: 20 row-split, 44 channel-split; 3 pool), 173 scalar, **0 fallbacks** |

The oneDNN outputs are **not bitwise identical across thread counts**: the
worker count changes the block count and the split dimension, a different block
shape can select a different oneDNN kernel, and its accumulation order is then
different. The test asserts the numerical bound above and prints `bitwise false`
rather than claiming a bitwise result it did not observe. The scalar path is
bitwise identical across thread counts because its kernels do not reorder a
reduction.

## 8. Known limits

- 4-D NCHW convolution and pooling only; 1-D and 3-D spatial graphs fall back.
- 2-D `MatMul` only; batched matmul (attention) falls back.
- `f32` only; no `bf16`/`f16`/quantized path.
- Pooling with `ceil_mode` or dilation falls back (the ONNX ceil rule allows
  windows past the padded region, which oneDNN pooling does not express).
- Weights and biases must be read-only model data (prepacked at open).
- A row-split convolution stages its input and output rows in dense
  engine-owned buffers, because a row submemory's strides take oneDNN off its
  JIT path (measured 20–600× slower). The staging is 22.0 MB on YOLOv8n and is
  part of the 124 MiB peak at 8 threads.
- No activation fusion yet: `Sigmoid`, `Clip`, `Add`/`Mul` run as separate
  scalar segments. The FFI has the post-ops surface; the planner does not fuse
  in phase 3b.
- The engine is a per-session object; there is no global primitive cache and no
  process-wide oneDNN setting is touched.
