# CPU providers: the scalar reference and the optional oneDNN engine

- **Status:** implemented, opt-in (`cpu-onednn`); phase 4a adds fused segments,
  a channels-last layout policy and a row-wise softmax
- **Date:** 2026-11-23
- **Authority:** [ADR-0001](adr-0001-one-runtime.md). The live operator matrix is
  [SUPPORTED.md](../../SUPPORTED.md).

This document describes the two engines a `Session` can execute a CPU plan with,
why one of them is a library and the other is not, and exactly what the library
is allowed to decide. It is the reference for the numbers in
[§7](#7-measurements) and for the invariants in `src/fn3/AGENTS.md`.

## 1. Two engines, one plan

A `Plan` is built before a session exists. Each `Segment` carries the engine the
plan chose and either one graph op or one recognised chain:

```text
Segment { op: Op(OpId) | Fused(FusedOp), engine: Scalar | OneDnn,
          inputs, outputs, params, loc }
```

The planner folds chains before it measures or asks anyone (`runtime::fuse`):
`conv -> relu`, the ONNX SiLU `mul(conv, sigmoid(conv))` and `add(conv, other)`
become one segment, so the intermediate values take no arena bytes. A fusion
fires only when it removes at least one segment and every declared consumer of
each intermediate value is inside the chain. `Plan::fusions` is the report
(`tests/fn3_cnn_run.rs` pins that the fused scalar plan is **bit-identical** to
the unfused one). The scalar engine implements every fused form, so a provider
that cannot run one does not force the planner to re-expand it.

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

The engine also claims a **fused** segment whose anchor is a convolution:
`conv + activation` (`relu`, `clip`, `sigmoid`, `tanh`, `gelu`, `swish`),
`conv + sum`, or `conv + sum + activation`. A fused segment with a different
anchor is `NotMine`; a fused convolution outside the shape rules is `Declined`
with the same reasons as the plain form, plus two of its own (a broadcast
residual, a constant residual).

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

Source and destination are plain NCHW (`dnnl_nchw`) under the default layout
policy, and only the weight descriptor is offered as `dnnl_format_tag_any`,
because oneDNN's blocked weight layouts are the point of using it. The
[layout policy](#45-layout-policy-channels-last-on-internal-conv-edges) can keep
an internal conv-to-conv edge in oneDNN's channels-last layout instead.

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

## 4.5 Layout policy: channels-last on internal conv edges

The plan records a **policy**, [`Plan::layout`], not a descriptor: a concrete
oneDNN layout is the library's decision, taken at session open, and a build
without `cpu-onednn` has no library. `PlanOptions::new` uses
`LayoutStrategy::Chained`; `LayoutStrategy::Plain` is the phase-3 contract and
is kept for A/B measurement.

**Plain** is the description above: every conv reads and writes `dnnl_nchw`.

**Chained** lets the engine keep an edge whose producer and every consumer are
oneDNN convolutions in the channels-last layout oneDNN chooses
(`dnnl_nhwc`, `dnnl_acdb`). The rule is structural and decided in
`Session::from_plan`, where the whole plan is visible:

- the producer is a oneDNN convolution whose output is not a graph output;
- every consumer is a oneDNN convolution that reads it as its source;
- no fused residual on the producer (a `sum` post-op reads the destination
  buffer back, which must be the NCHW plan slot);
- the consumer is not grouped (a group's channel slice is strided in a
  channels-last tensor).

The engine then decides whether the **block split** can produce it. A row range
of a channels-last tensor `[N, H, W, C]` is one contiguous byte run, so a
segment whose result stays channels-last splits rows; a channel range is not
contiguous, so such a segment splits channels and writes the NCHW plan slot. A
segment whose source is channels-last and whose result is NCHW splits either
way: its channel blocks or row blocks read the dense channels-last buffer.

### Where the conversions are

A segment **never mixes layout families**. OneDNN's kernels are fast on
`nchw -> nchw` and on `channels-last -> channels-last`; measured on this host, a
one-block 64x40x40 3x3 convolution with a channels-last source and an NCHW
destination took **50 ms** against 0.5 ms for the same convolution with matching
families — the library leaves its JIT path. So each family boundary carries one
explicit conversion, run once before the segment's blocks:

| Edge | Conversion | Counted as |
|---|---|---|
| plain source → channels-last result (chain head) | NCHW → channels-last of the whole input slot | `Reorder { direction: NchwToBlocked, bytes }` |
| channels-last source → plain result (chain tail) | channels-last → NCHW of the whole source buffer into the plan slot | `Reorder { direction: BlockedToNchw, bytes }` |
| channels-last → channels-last (interior) | none | — |

`Plan::reorders` lists them, `Plan::reorder_bytes` sums them, and
`OneDnnReport` repeats the count and the bytes. On the pinned graph at eight
workers the chained policy installs 11 channels-last buffers (10.0 MB) and runs
16 conversions moving 7.6 MB per forward; at four workers it installs 13
(11.1 MB) and 16 conversions (7.7 MB).

The conversions are engine-owned buffers, not plan slots: the plan's slots are
the graph's NCHW values and a layout is a provider decision, so the blocked
bytes have no `SlotId`. This is the one place where the implementation is not
literally "a `Reorder` segment": the conversion is an explicit reorder primitive
and a step of the segment that needs it, and it is counted and reported exactly
as if it were. Making it a plan segment would mean extending the slot table past
`graph.value_count()` and renumbering the engine's prepared table after
preparation; the conversion is per-segment in either shape.

**A channels-last destination requires dense row blocks.** The obvious
alternative — a channel-submemory destination of a channels-last parent — was
measured at **233 ms** for four 8-channel blocks of a 32x80x80 3x3 convolution,
against 1.09 ms for the same tensor as one block: the submemory keeps the
parent's blocked strides and oneDNN falls off its JIT path, exactly as it did
for NCHW row submemories in phase 3b. Row ranges are dense in channels-last, so
the row split is what makes the layout usable. A convolution whose row split
would give fewer blocks than its channel split stays plain (the engine requires
`row_blocks >= 2` and `row_blocks >= channel_blocks`); at one worker every
convolution is one block, so the policy is inert there and the plan reports zero
conversions.

### What it is worth

| Policy | 4 threads | 8 threads | Peak RSS (8) | channels-last segments | conversions |
|---|--:|--:|--:|--:|--:|
| `Plain` | 27.96 | 29.65 | 120.6 MiB | 0 | 0 |
| `Chained` | **26.47** | **28.22** | **104.7 MiB** | 11 | 16 (7.6 MB) |

Best median of three repetitions each, same host and build. The win is 5% at
both thread counts and 16 MiB of peak RSS: the row blocks of a chained
convolution read dense channels-last rows instead of copying NCHW rows into
staging, which is why `staging_bytes` falls from 22.0 MB to 8.3 MB. The kernel
itself is much faster than 5% — a 3x3 32x80x80 convolution with one block runs in
0.88 ms in channels-last against 1.58 ms in NCHW (1.8x) — but the pinned export's
convolution chains are only two or three long, so a conversion at each end eats
most of it. On a graph with longer conv chains the policy should pay
substantially more; that is a prediction, not a measurement.

## 4.6 Fusion: what is folded, in which order, and what it saved

The planner folds three graph shapes into one segment:

| Graph | Segment | Folded ops |
|---|---|---|
| `act(conv(x))` | `conv` + activation | one of `Relu`, `Clip`, `Sigmoid`, `Tanh`, `Gelu` |
| `mul(conv(x), sigmoid(conv(x)))` | `conv` + swish | `Sigmoid`, `Mul` |
| `add(conv(x), other)` / `act(add(conv(x), other))` | `conv` + sum ( + activation) | `Add` ( + the activation) |

with `other` the same shape and element type as the result. The SiLU row is the
ONNX form every export writes for YOLOv8's activation; it is one graph shape, so
the recogniser matches the two-consumer pattern rather than hoping for a
`Swish` node.

**Order is fixed: sum, then activation.** oneDNN's direct convolution requires
its `sum` post-op to be the first post-op, and the sum post-op reads the
destination buffer back rather than taking a separate argument. The planner
therefore accepts `add(conv, other)` and `act(add(conv, other))`, and does
**not** accept `add(act(conv), other)` — which is what the pinned YOLOv8n export
writes for all six residual adds, because the residual is added after the SiLU.
Those six adds stay scalar segments. That is a graph-shape limitation, not a
missing implementation: the oneDNN sum post-op is implemented, claimed and
tested (`a_fused_sum_and_activation_matches_the_three_step_chain` in the engine,
`a_residual_add_on_the_convolution_fuses_as_a_sum` in the planner), it just does
not match the pinned file. On this graph the sum pattern therefore fires **zero**
times and the fusion report says so.

**The activation is evaluated by the engine, not by a oneDNN eltwise post-op.**
Measured on this host (oneDNN 3.14.1, AVX2, a 32x80x80 3x3 convolution with
204 800 outputs, one block):

| Epilogue | Cost per element |
|---|--:|
| oneDNN `relu` post-op | 1.3 ns |
| oneDNN `clip` post-op | 6 ns |
| oneDNN `logistic` post-op | 9 ns |
| oneDNN `swish` post-op | 9.8 ns |
| this engine's scalar loop, same buffer | 0.4–2.7 ns |

The post-ops run on a reference path on this build, and `swish` as a post-op
made a forward **3.4x slower at one thread** than the two scalar segments it
replaced (68.8 ms against 44.2 ms). The engine therefore applies
`Activation::apply` over each block's own destination — the same expression the
scalar kernel uses, so the fused result is bit-identical to the unfused chain —
and keeps `dnnl_post_ops_append_eltwise` behind
`onednn::EPILOGUE_POST_OP = false` so the comparison is reproducible elsewhere.
The `sum` post-op is kept as a post-op.

**What fusion bought on the pinned graph** (8-thread oneDNN, same build,
`FN3_CNN_FUSE=0` for the unfused column):

| | Unfused | Fused |
|---|--:|--:|
| Segments | 240 | 126 |
| Fusions | 0 | 57 (`conv+swish`), 114 segments removed |
| Arena | 6 144 000 B | 2 867 200 B |
| `Conv` | 20.4 ms | 23.5 ms |
| `Sigmoid` | 7.9 ms | 0.3 ms |
| `Mul` | 1.7 ms | 0.0 ms |
| Forward | 35.0 ms | 28.9 ms |

At one thread the same comparison is 44.5 → 41.1 ms: the in-place swish over a
block's hot destination costs about 0.55x what the two separate segments cost
(`1/(1+exp(-x))` is the same `expf`, but the intermediate slot and its two extra
memory passes are gone). The memory win is unambiguous: the arena falls by
3.3 MB (53%).

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
layout and were only copied). On the pinned YOLOv8n graph under the plain
layout policy oneDNN chose the plain layout for every weight buffer, so the
report says `0 reordered`; under the chained policy the channels-last
source/destination pair makes it want a blocked weight layout for 13 blocks, so
the session-open reorder runs 13 times and the report counts it. Either way the
12.6 MB of prepacked weights are reordered once, at open, and never per
forward. The row blocks of one convolution compare the weight
descriptor oneDNN chose and share one prepacked buffer when it matches, which is
the case for every row-split convolution on this graph; the channel blocks keep
their own (their weights are disjoint slices). Dense row staging is reported
separately (`staging_bytes`): it is the price of keeping dense `nchw`
descriptors for a row split, and it falls from 22.0 MB under the plain layout
policy to 8.3 MB under the chained one, because the chained row blocks read
dense channels-last rows instead of staging them.

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
only ever makes a forward slower. Compare numbers within one table rather than
across sessions.

### Result

Phase 4a, fused plans, `LayoutStrategy::Chained`, three repetitions per
configuration:

| Engine | Threads | Median ms | p95 ms | Peak RSS MiB |
|---|--:|--:|--:|--:|
| scalar | 1 | 1229.96 | 1327.19 | 37.2 |
| scalar | 4 | 365.44 | 373.29 | 37.3 |
| scalar | 8 | 237.47 | 245.59 | 37.6 |
| oneDNN | 1 | 40.79 | 41.62 | 75.9 |
| oneDNN | 4 | **26.47** | 30.40 | 98.7 |
| oneDNN | 8 | 28.22 | 29.36 | 104.7 |
| ORT | 1 | 22.90 | 23.30 | 135.6 |
| ORT | 4 | 7.10 | 7.24 | 132.5 |
| ORT | 8 | 6.63 | 7.21 | 793.0 |

Against phase 3b on the same machine (`b94922a`: scalar 1289.65 / 483.66 /
340.11 ms, oneDNN 46.96 / 45.54 / 42.90 ms, ORT 23.18 / 10.19 / 8.74 ms):

| Configuration | Phase 3b | Phase 4a | Change |
|---|--:|--:|--:|
| scalar, 1 | 1289.65 | 1229.96 | −4.6% |
| scalar, 4 | 483.66 | 365.44 | −24.4% |
| scalar, 8 | 340.11 | 237.47 | **−30.2%** |
| oneDNN, 1 | 46.96 | 40.79 | −13.1% |
| oneDNN, 4 | 45.54 | 26.47 | **−41.9%** |
| oneDNN, 8 | 42.90 | 28.22 | **−34.2%** |

The scalar path gains from the row-wise softmax and from fusion (fewer
segments, fewer arena bytes); its arithmetic and its bit pattern are unchanged,
which `tests/fn3_cnn_run.rs` asserts against the ORT dump and across 1/4/8
workers.

The oneDNN engine is now **30.2x faster than the scalar reference at one
thread** (1229.96 / 40.79), 13.8x at four and **8.4x at eight**. Against ONNX
Runtime it is **4.3x at eight threads** (28.22 vs 6.63), 3.7x at four
(26.47 vs 7.10) and 1.8x at one (40.79 vs 22.90). It still does not reach the
≤2×-ORT goal at 4/8 threads; that is a negative result for the goal and is not
restated as a win. Note that eight workers is **slower than four** in every
configuration on this host (28.22 vs 26.47). The per-op table shows the
difference is not in any one segment: with 8 blocks per row-split convolution the
halo traffic and the per-block fixed costs grow, and the desktop's load makes the
last two workers expensive. Four workers is the better configuration here; eight
is what the historical tables use, so both are reported.

### Where the time goes, before and after

Both columns are the 8-thread oneDNN session on the same host with the final
build; "unfused" is `FN3_CNN_FUSE=0`, which keeps the layout policy but has no
chain to keep channels-last.

| Part | Unfused ms | Fused ms | Engine |
|---|--:|--:|---|
| `Conv` (64 segments, 57 with the swish epilogue) | 20.4 | 23.5 | oneDNN |
| `Sigmoid` (58 → 1) | 7.9 | 0.3 | scalar |
| `Mul` (60 → 3) | 1.7 | 0.0 | scalar |
| `MaxPool` (3) | 2.4 | 2.4 | oneDNN |
| `Softmax` (1) | 0.47 | 0.47 | scalar |
| `Concat` (17) | 0.57 | 0.67 | scalar |
| `Resize`, `Slice`, `Split`, `Add`, `Transpose`, `Reshape`, … | 1.4 | 1.4 | scalar |
| total (segment sum) | 34.9 | 28.9 | |

The 8-thread per-op table, final build (median of three groups of three
forwards, per-op totals in ms):

| Op | calls | 1 thread | 8 threads | share (8) |
|---|--:|--:|--:|--:|
| `Conv` | 64 | 35.77 | 23.52 | 81.4% |
| `MaxPool` | 3 | 2.24 | 2.42 | 8.4% |
| `Concat` | 17 | 0.21 | 0.67 | 2.3% |
| `Softmax` | 1 | 0.44 | 0.47 | 1.6% |
| `Resize` | 2 | 1.47 | 0.34 | 1.2% |
| `Sigmoid` | 1 | 0.42 | 0.32 | 1.1% |
| `Slice` | 2 | 0.22 | 0.26 | 0.9% |
| `Split` | 8 | 0.06 | 0.25 | 0.9% |
| `Reshape` | 8 | 0.04 | 0.21 | 0.7% |
| `Add` | 9 | 0.04 | 0.19 | 0.7% |
| `Transpose` | 2 | 0.15 | 0.18 | 0.6% |
| `Mul` | 3 | 0.00 | 0.03 | 0.1% |
| `Sub`, `Div`, `Gather`, `Shape` | 6 | 0.01 | 0.06 | 0.2% |
| total | 126 | 41.06 | 28.90 | |

Four changes and three negative results produced that:

1. **Fusion (4.6).** 57 `conv+swish` segments replace 171 graph segments; the
   plan falls from 240 to 126 segments and the arena from 6 144 000 to
   2 867 200 bytes. `Sigmoid` 7.9 → 0.3 ms and `Mul` 1.7 → 0.0 ms at eight
   threads; `Conv` grows by the in-place epilogue (20.4 → 23.5 ms). Net 35.0 →
   28.9 ms. The residual-add pattern fires zero times on this graph because the
   export adds *after* the SiLU, and oneDNN's post-op order is sum-then-eltwise.
2. **Channels-last layout policy (4.5).** 11 internal edges stay channels-last
   and 16 conversions move 7.6 MB per forward. 29.65 → 28.22 ms at eight
   threads, 27.96 → 26.47 at four, and peak RSS 120.6 → 104.7 MiB because the
   row staging falls from 22.0 to 8.3 MB. The convolution kernel itself is 1.8x
   faster (0.88 vs 1.58 ms for a 32x80x80 3x3 one-block convolution); the
   graph's two-to-three-convolution chains mean the conversions consume most of
   that.
3. **Softmax** was the one genuinely pathological kernel: 3.6 ms for 134 400
   elements in 8 400 rows of 16, because it walked a `Locator::index` (a checked
   multiply-add per axis) for every element in three passes. The contiguous
   last-axis path is three slice walks per row, no allocation (the exponent pass
   writes the destination and the divide pass scales it in place), and the same
   arithmetic order, so it is bit-identical: a new unit test pins 2x4x512 values
   against the documented loop. 3.56 → 0.44 ms at one thread, 0.47 at eight.
4. **Parallel pointwise.** `for_each_row` now splits into whole rows whenever the
   output has at least 8 Ki elements and at least 1 024 elements per worker,
   instead of a flat 64 KiB (16 Ki elements). The split unit is one row, so the
   arithmetic order cannot change. This is **not measurable on the pinned
   graph**: after fusion the pointwise segments it targeted are gone (the
   surviving `Sigmoid` is 168 000 elements, above both floors), so the change is
   documented as a rule rather than claimed as a win.

### Negative results, recorded

| Attempt | Result |
|---|---|
| oneDNN eltwise post-op for the fused activation | 3.4x slower at one thread than the two scalar segments (68.8 vs 44.2 ms forward); kept behind `EPILOGUE_POST_OP = false` |
| activation after the residual as one oneDNN primitive | not expressible: `sum` must be the first post-op, and it reads the destination; the six YOLOv8n residual adds stay scalar |
| channels-last destination + channel split (dst submemory of a blocked parent) | 233 ms per 32x80x80 3x3 convolution against 1.09 ms; blocked destinations require row splits |
| mixing families inside one segment (channels-last src, NCHW dst) | 50 ms for a one-block 64x40x40 3x3 convolution against 0.5 ms; every family boundary gets an explicit conversion |
| channels-last at one worker | inert: one block per segment cannot row-split, so the policy installs zero buffers and the plan reports zero conversions |

### Correctness of these numbers

Measured by `tests/fn3_onednn.rs` on the same machine after the phase-4a
changes:

| Comparison | Gate | Measured |
|---|---|---|
| fused vs unfused oneDNN plan | `|d| <= 1e-4 + 1e-4|ref|` | max_abs 3.8e-4, max_rel 6.7e-6 |
| channels-last vs plain layout | same gate | max_abs 3.8e-4, max_rel 6.7e-6 |
| oneDNN vs scalar, same plan | `|d| <= 1e-4 + 1e-4|ref|` | max_abs 4.4e-4, max_rel 7.6e-6 |
| oneDNN vs ONNX Runtime | `|d| <= 2e-4 + 1e-3|ref|`, rel `<= 1e-3` | passes (max_abs 4.7e-4, max_rel 1.4e-4) |
| oneDNN `Fixed(4/8)` vs `Fixed(1)` | same 1e-4/1e-4 gate | **not bitwise**; max_abs 2.3e-4, max_rel 5.1e-6 |
| scalar fused vs scalar unfused | bitwise | **bitwise identical**, 176 400 elements |
| scalar `Fixed(1/4/8)` | bitwise | bitwise identical |
| plan report | | 126 segments: 67 oneDNN (64 conv: 20 row-split, 44 channel-split; 3 pool), 59 scalar, 0 fallbacks, 57 fusions (114 segments removed) |

The oneDNN outputs are **not bitwise identical across thread counts**: the
worker count changes the block count and the split dimension, a different block
shape can select a different oneDNN kernel, and its accumulation order is then
different. The test asserts the numerical bound above and prints `bitwise false`
rather than claiming a bitwise result it did not observe. The scalar path is
bitwise identical across thread counts because its kernels do not reorder a
reduction, and the scalar fused path is bitwise identical to the unfused one
because it evaluates the same expressions in the same order.

## 8. Known limits

- 4-D NCHW convolution and pooling only; 1-D and 3-D spatial graphs fall back.
- 2-D `MatMul` only; batched matmul (attention) falls back.
- `f32` only; no `bf16`/`f16`/quantized path.
- Pooling with `ceil_mode` or dilation falls back (the ONNX ceil rule allows
  windows past the padded region, which oneDNN pooling does not express).
- Weights and biases must be read-only model data (prepacked at open).
- A row-split convolution whose result is plain NCHW stages its input and
  output rows in dense engine-owned buffers, because a row submemory's strides
  take oneDNN off its JIT path (measured 20–600× slower). Under the chained
  layout policy the input side of such a segment usually reads dense
  channels-last rows instead, which is why the staging falls from 22.0 MB to
  8.3 MB on YOLOv8n.
- A channels-last edge must be produced by dense row blocks, so a convolution
  whose row split would give fewer blocks than its channel split stays plain.
  At one worker that is every convolution.
- Fusion covers an activation on a convolution, the ONNX SiLU form and a
  residual add that precedes (optionally) an activation. A residual add after an
  activation, a `Clip` with only one bound, and a broadcast residual are left as
  separate segments.
- The engine is a per-session object; there is no global primitive cache and no
  process-wide oneDNN setting is touched.
