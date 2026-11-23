# v3 status

Where the rewrite stands, what is measured, and what is still open. Every claim
here points at a commit, a test or a measurement; nothing is projected.

## Phases

| phase | state | evidence |
|---|---|---|
| 0 freeze 2.x, delete the legacy surface, ADR | done | `7401ff9`, `7bb7ebf`, `d492a19`, `3b549ab`; `src/` went from 140,779 to ~52,000 lines, dependencies from 17 to 6 |
| 1 typed IR + CNN frontend | done | `bda7dbc` (IR + shape pass), `543c6fd` (scope), `2887328` (CNN ops 41-52, import of YOLOv8n/s to 240 nodes / 398 static values) |
| 2 scalar segment executor, arena, session threads | done | `10cda68`; ORT parity on yolov8n, 1/4/8 workers exact, one arena allocation per session |
| 3 oneDNN provider | done | `b94922a` (engine behind `cpu-onednn`), `ca5801f` (memcpy-shaped glue, row-split convs) |
| 4 fused segments, blocked layouts, artifact v2, API | done | `13595be`, `6235f37`, `e004a46` |
| 5 true W8A8 + calibration + `precision="auto"` | open | not started |
| 6 decoder/KV on the same plan and artifact, Vulkan re-wire | open | fn3 decoder still runs the v1-shaped plan path |
| 7 packaging, fuzzing/OOM hardening, docs | partially | wheels, fuzz corpus refresh and the benchmark methodology are open |

## Measured, on this machine

AMD Ryzen 7 3700X, 8 physical cores / 16 logical, AVX2 only (no AVX-VNNI, no
AVX-512), oneDNN 3.14.1 built with the SEQ runtime, ONNX Runtime 1.32.0,
release, 3 warm-ups + 30 forwards, median of three runs.

YOLOv8n, `f32[1,3,320,320] -> f32[1,84,2100]`:

| engine | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|
| scalar | 1229.96 ms | 365.44 ms | 237.47 ms |
| oneDNN | 40.79 ms | **26.47 ms** | 28.22 ms |
| ONNX Runtime | 22.90 ms | 7.10 ms | 6.63 ms |

That table is the phase-4a tree. The block payload floor was lowered from 16 Ki
to 4 Ki elements (see `docs/v3/backends.md`), which is the scaling fix: measured
against the shipped baseline binary with both run alternately on the same
artifact,

| engine | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|
| oneDNN, baseline `57272fc` | 44.33 ms | 33.26 ms | 31.68 ms |
| oneDNN, current | 44.46 ms | **22.73 ms** | **18.90 ms** |
| ONNX Runtime | 23.86 ms | 7.82 ms | 7.64 ms |

so oneDNN is now **1.86x** ORT at one worker (inside the <=2x goal), 2.91x at
four and 2.47x at eight, down from 3.7x / 4.3x. The remaining gap is
convolution throughput, not block count: the 57 fused `Conv+Swish` segments are
75 % of the forward, and the small layers (10x10, 20x20) cannot fill eight
workers. `docs/v3/backends.md` records the per-op table, the negative results
(one pool region per forward: no change; graph-level scheduling: 1.59x ceiling
on this graph) and the cost in peak RSS (105 MB -> 125 MB at eight workers, 76 MB at one).

### Without the Python binding

The numbers above come from the in-process harness. Because a session is also
reachable from the CLI, the same forward can be measured with no Python in the
process at all, which is the honest way to answer "how much of this is the
binding":

```
fastnn-infer prepare --onnx yolov8n.onnx --out yolov8n.fnn3
fastnn-infer run yolov8n.fnn3 --input images=ones --threads 8 --warmup 5 --repeat 50
```

`--repeat` times `Session::execute` (arena in place, no output copy per
forward) and prints median and p95; the outputs are materialised once, after
the series. Measured on this machine, 5 warm-ups + 50 forwards, same `ones`
input for every engine:

| | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|
| fastnn scalar, Rust CLI | 1322.39 ms | 390.60 ms | 256.21 ms |
| fastnn oneDNN, Rust CLI | 44.26 ms | **32.29 ms** | 32.48 ms |
| ONNX Runtime, Python per-call | 23.86 ms | 7.82 ms | 7.64 ms |

The Python binding is not what separates them: the floor for a Python + ORT
`session.run` call measured **0.032 ms** on a trivial identity model, which is
0.13 % of ORT's 1-thread number and 0.4 % of its 8-thread number.

What the Python-free measurement does show is the real shape of the gap:

- At **one thread the gap is 1.86x** — inside the <=2x goal.
- From 1 to 8 threads fastnn gains only 1.36x, while ORT gains 3.12x. The
  problem is scaling, not single-core throughput: 59 of the 126 segments are
  still scalar and run with a one-thread budget, so Amdahl's law sets the
  ceiling long before the convolutions do.

The in-process harness (`fn3_cnn_perf`, which uses borrowed outputs) measures
40.79 / 26.47 / 28.22 ms on the seeded-random input; the CLI measures a fresh
process per run. Differences of 5-20 % between the two are host and process
noise, not binding overhead, which is why both are quoted rather than one
"official" number.

The accuracy gates are real: oneDNN against scalar is within
`1e-4 + 1e-4|ref|`, against ORT within `2e-4 + 1e-3|ref|`, and the scalar path
is bit-identical across 1/4/8 workers. oneDNN is numerically stable but not
bit-identical across worker counts because the block count changes the kernel;
the tests print that rather than hiding it.

Artifact v2 (200,001,344 bytes, 200,000,000 bytes of weights): opening it takes
RSS from 4.4 MB to 5.6 MB, and a second session adds 65 KB, so the weights are
mapped once and shared. `tests/fn3_artifact_v2.rs` also flips every section,
truncates at every boundary and rejects v1 and future versions by number.

## Recently fixed bugs from the issue tracker

| issue | state | commit |
|---|---|---|
| #112 stock YOLO11n import | opset 12..=22, `Resize(X, "", sizes)`, and batched attention `MatMul` now work; one blocker remains (below) | `84dec2e` |
| #113 Conv+Add fusion could read a residual produced later | rejected when the producer is at or after the anchor, with a regression test | `e004a46` |
| #116 softmax on a contiguous mid-axis tensor | second contiguous path, bit-identical to the reference loop | `e004a46` |
| #120 segment timing hid a fused activation under `Conv` | segments are reported by their whole chain (`Conv+Swish`) | `8ac736a` |

`fastnn-infer doctor` on the stock `yolo11n.onnx` (11 MB, upstream asset)
now gets past the operator set, the `Resize` hole and the rank-4 `MatMul`, and
stops at exactly one problem:

```
1. [shape] node 135 `/model.10/m/m.0/attn/Reshape` (Reshape):
   shape contradiction: reshape target element 3 is ? and not a statically known extent
```

The export declares `[0,3,0,0]`; the attention reshape multiplies symbolic
extents, so nothing can be proved static. This is the shape-bucket gap of #98
and #104, not an operator gap: a concrete input shape must be bound before
planning. Tracked as the first task of the next slice.

## What the CLI and API look like now

```
fastnn-infer prepare model.onnx -o model.fnn3 [--target cpu] [--threads N]
fastnn-infer load model.fnn3
fastnn-infer run model.fnn3 --input images=1,3,320,320
fastnn-infer inspect model.fnn3
fastnn-infer doctor model.onnx
```

The Rust surface is `fastnn::inference::{prepare, load, inspect, doctor}` with
`PrepareReport` and `DoctorReport`; Python mirrors it through
`fastnn.inference`. `load` is keyword-only (`target=`, `device=`), a `Model` is
immutable, and every `session()` owns its own arena and thread pool.

## Open work, in the order it should be taken

1. **Shape binding** (#98, #104): `prepare --shape images=1,3,640,640` and a
   `doctor` that reports which symbols a plan would need, so a stock export can
   be prepared without external graph surgery. Unblocks #112's last mile and
   the matched YOLO11n comparison.
2. **#117 remove the pre-v3 CPU runtime**: the 40 decoder ops still dispatch
   one op at a time through `backend::cpu::dispatch_kernels`, and
   `model/{plan,session,specialize,kv}.rs` overlap `runtime/`. This is the last
   place where the old shape survives.
3. **#115 peak RSS**: the 4 Ki floor raised eight-worker peak RSS on yolov8n
   from 105 MB to 125 MB (76 MB at one worker); the row-split staging buffers
   are per block and should be bounded per worker rather than per plan.
4. **Phase 5 W8A8** (#103): oneDNN int8 with per-channel scales and calibration,
   gated on accuracy rather than on a dtype name.
5. **Phase 6 decoder/KV** (#108) and **phase 7 packaging** (#109), hardening
   (#110), docs (#111).

## Reproducing any number here

```bash
cargo test --lib
cargo test --test fn3_artifact_v2
cargo test --no-default-features --features cpu-onednn --test fn3_onednn
FN3_CNN_PERF=onednn FN3_CNN_THREADS=8 FN3_CNN_MODEL=/path/yolov8n.onnx \
  cargo test --release --no-default-features --features cpu-onednn,parallel,segment-timing \
  --test fn3_cnn_perf -- --nocapture
python3 tools/parity/ort_reference.py /path/yolov8n.onnx target/cnn-parity --time
```

`FN3_CNN_PERF` is opt-in on purpose: a benchmark that runs inside the normal
test suite is a benchmark that fails on a busy machine and gets ignored.
