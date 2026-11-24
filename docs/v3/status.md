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
| 5 true W8A8 + calibration + `precision="auto"` | open, hardware-gated | `w8a8` parses and is refused with a measured reason (`prepare`, CLI, API; `tests/fn3_cli.rs`); on this AVX2 host oneDNN's AVX2 int8 convolution is only **0.97-1.20x** the blocked f32 path (0.83-0.94 ms against 0.94-1.07 ms for 32x80x80x3x3 and 64x40x40x3x3), because `vpmaddubsw`+`vpmaddwd` has no MAC-throughput advantage over FMA without VNNI. The earlier 848x figure came from an NCHW probe that fell back to the reference kernel. An executable path needs VNNI/AMX before it can be measured and shipped (#103) |
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

That table is the phase-4a tree. Two changes moved it. The block payload floor
went from 16 Ki to 4 Ki elements (`docs/v3/backends.md`), which is the scaling
fix. Then the scalar tail stopped paying a pool hand-off per segment and the two
largest tail ops got row-parallel kernels. Measured by running the pre-change
binary and the current one alternately on the same artifact, 20 forwards each:

| engine | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|
| oneDNN, pre-change `92af69c` | 44.43 ms | 20.61 ms | 20.19 ms |
| oneDNN, after the scalar-tail wave | 42.01 ms | 20.32 ms | 19.77 ms |
| oneDNN, after one-block channels-last | **36.60 ms** | **19.39 ms** | **18.70 ms** |
| ONNX Runtime | 23.86 ms | 7.82 ms | 7.64 ms |

so oneDNN is **1.53x** ORT at one worker (inside the <=2x goal), 2.48x at four
and 2.45x at eight. The one-block channels-last change is what moved the
one-worker number: the internal conv-to-conv edges no longer drop back to NCHW
when the row split cannot produce two blocks. The remaining gap is convolution throughput, not block
count or glue: the 57 fused `Conv+Swish` segments are 77 % of the forward
(14.87 of 19.31 ms at eight workers), the pool now takes every row-parallel
tail op, and the small layers (10x10, 20x20) cannot fill eight workers. `docs/v3/backends.md` records the per-op table, the negative results
(one pool region per forward: no change; graph-level scheduling: 1.59x ceiling
on this graph). Staging is now pooled per worker (#115): `staging_bytes` at
eight workers fell from 14,110,720 B to 2,048,000 B and CLI peak RSS at eight
workers from ~106 MB to ~94 MB. A follow-up pools the oneDNN scratchpads the
same way (391-672 blocks owned 27-29 MB; the pool is 3.7 MB at eight workers),
which takes the same measurement to **76.5 MB** at eight, 75.0 MB at four and
52.4 MB at one, with median wall time unchanged within noise (measured A/B,
three runs each; see `docs/v3/backends.md`).

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
forward) and prints median, p95 and the process's peak RSS (`VmHWM`); the
outputs are materialised once, after the series. Measured on this machine,
5 warm-ups + 50 forwards, same `ones` input for every engine:

| | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|
| fastnn scalar, Rust CLI | 1322.39 ms | 390.60 ms | 256.21 ms |
| fastnn oneDNN, Rust CLI | **36.60 ms** | **19.39 ms** | **18.70 ms** |
| ONNX Runtime, Python per-call | 23.86 ms | 7.82 ms | 7.64 ms |

Peak RSS for those oneDNN runs: 68.7 MB at one worker, 75.1 MB at four,
77.3 MB at eight.

### Peak RSS against ONNX Runtime

Same protocol, same `ones` input, peak resident set of the whole process. The
fastnn side is the CLI (no interpreter); the ORT side is a Python process, so
its number includes CPython, numpy and the binding (about 38 MB on this host).
The comparison therefore flatters fastnn: it is a process-level number, not an
engine-only one.

| graph | threads | fastnn | ONNX Runtime |
|---|---:|---:|---:|
| yolov8n, 320x320 | 1 | **68.7 MB** | 100.6 MB |
| | 4 | **75.1 MB** | 100.6 MB |
| | 8 | **77.3 MB** | 100.4 MB |
| yolo11n, 640x640 | 1 | **134.2 MB** | 149.7 MB |
| | 4 | 148.5 MB | **149.6 MB** |
| | 8 | 152.5 MB | **151.2 MB** |

On the 320 graph fastnn is 23-32 % smaller even counting the interpreter in
ORT's number. On the 640 graph the two are within 2 MB once the per-thread
workspace ORT allocates is included; that graph's arena is 16.4 MB and the
mapped weights 10.9 MB, so most of both numbers is the engine, not the model.
FastNN's own memory is the arena plus 14.4 MB of pooled channels-last buffers
and 3.7 MB of scratchpads at eight workers; the weight payload is mapped and
shared, not copied.

The Python binding is not what separates them: the floor for a Python + ORT
`session.run` call measured **0.032 ms** on a trivial identity model, which is
0.13 % of ORT's 1-thread number and 0.4 % of its 8-thread number.

The per-layer breakdown behind these totals — every segment's time, MACs and
achieved GMAC/s next to ORT's per-node profile — is in
[per-layer.md](per-layer.md). It is the document that says *which* layers are
behind, and it is why the next work is pool/resize and low-channel 1×1 policy,
not a new convolution algorithm.

What the Python-free measurement does show is the real shape of the gap:

- At **one thread the gap is 1.53x** — inside the <=2x goal.
- From 1 to 8 threads fastnn now gains 1.96x (ORT 3.12x). 59 of the 126
  segments still run the scalar reference engine, but the row-parallel ones
  (concat, softmax, activations, pools) take the session pool; what is left
  serial is small (the 10x10 and 20x20 head layers) and Amdahl's ceiling is
  the convolutions themselves. The remaining per-convolution gap is the
  kernel, not a missing fusion: MLAS runs these shapes through its tuned
  NCHWc/direct and im2col-GEMM micro-kernels, while oneDNN's x64 path here is
  its NCSP direct kernel. Winograd is not the difference: neither MLAS on x64
  nor oneDNN 3.x on x64 implements it — in the pinned oneDNN source the only
  CPU Winograd kernels are ACL/AArch64 and RVV/RISC-V. A 3x3 Winograd win on
  x86 would have to be a new kernel of ours, not a provider flag.

The attention-headed detector tells the same story from the other side:
stock YOLO11n 640 gets more from the tail work because it has 174 scalar
segments. Interleaved, pre-change vs current, 15 forwards:

| engine | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|
| oneDNN, pre-change `92af69c` | 164.46 ms | 66.40 ms | 62.17 ms |
| oneDNN, current | **156.26 ms** | 67.88 ms | **55.90 ms** |
| ONNX Runtime | 77.78 ms | 32.07 ms | 27.90 ms |

The four-thread difference is inside this host's noise (the same binary
measured 64.2-67.9 ms across runs); the one- and eight-worker differences
reproduce. Against ORT that is 2.01x / 2.12x / 2.00x, down from
2.13x / 2.34x / 2.21x.

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
| #112 stock YOLO11n import | opset 12..=22, `Resize(X, "", sizes)` / `Resize(X, roi, scales)`, batched attention `MatMul`, input-shape binding, and the DFL decode's `Range(Cast(Gather(Shape(conv))))` now import; the model prepares and runs with ORT parity | `84dec2e`, `3ee9295` |
| #113 Conv+Add fusion could read a residual produced later | rejected when the producer is at or after the anchor, with a regression test | `e004a46` |
| #116 softmax on a contiguous mid-axis tensor | second contiguous path, bit-identical to the reference loop | `e004a46` |
| #120 segment timing hid a fused activation under `Conv` | segments are reported by their whole chain (`Conv+Swish`) | `8ac736a` |

`fastnn-infer doctor` on the stock `yolo11n.onnx` (11 MB, upstream asset,
SHA-256 `634279b4…053b`, 421 ONNX nodes, opset 22) reports **ready to prepare**:
421 typed nodes, 641 values, every extent static. Inputs are declared
`[batch, 3, height, width]`; `--shape images=1,3,640,640` replaces those extents
before shape inference (see `docs/v3/scope.md` §6).

The last blocker was the DFL decode's index machinery:
`Range(0, Cast(Gather(Shape(conv), i)), 1)`. The importer already tracked
constant payloads, but `Cast` dropped them and `Range` only read integer atoms,
so the f32 spelling of the same indices was untrackable. `3ee9295` folds both
and adds the typed-IR shape rules for `Cast`, `Range`, `ConstantOfShape` and
`Expand` (ops whose result type or length is data), and lifts the opset-13
axes-as-input form of `Unsqueeze`/`Squeeze` into `OpParams.axes`.

Measured on the same host and protocol as the table above (5 warm-ups + 30
forwards, median first, p95 from the same series), yolov8n's successor with an
attention block and a 26k-anchor head:

| yolo11n, 640x640 | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|
| fastnn oneDNN, before | 165.74 ms | 75.14 ms | 61.79 ms |
| fastnn oneDNN, after one-block channels-last | **137.7 ms** | **62.0 ms** | **52.6 ms** |
| ONNX Runtime | 77.78 ms | 32.07 ms | 27.90 ms |

That is 1.77x / 1.93x / 1.89x, down from 2.13x / 2.34x / 2.21x, and still
worse than yolov8n's 1.53x because the head's scalar segments and the attention
block make the tail longer: 174 of 265 segments are scalar and run with a
one-thread budget. Output
parity on the same input, tolerance `2e-4 + 1e-3|ref|`: 0 of 705,600 values
outside it (max_abs 2.3e-3 on values up to 637, mean_abs 2.4e-6).

## What the CLI and API look like now

```
fastnn-infer prepare model.onnx --out model.fnn3 [--target cpu] [--shape images=1,3,640,640]
fastnn-infer load model.fnn3
fastnn-infer run model.fnn3 --input images=ones --warmup 5 --repeat 50
fastnn-infer inspect model.fnn3
fastnn-infer explain model.fnn3 [--target cpu]
fastnn-infer doctor model.onnx [--shape images=1,3,640,640]
```

The Rust surface is `fastnn::inference::{prepare, load, inspect, doctor}` with
`PrepareReport` and `DoctorReport`; Python mirrors it through
`fastnn.inference`. `explain` is a CLI view over the stored plan (the same data
is on `Artifact::plan_for`), useful before running a model whose engine split
you want to understand. `load` is keyword-only (`target=`, `device=`), a `Model` is
immutable, and every `session()` owns its own arena and thread pool.

## Open work, in the order it should be taken

1. ~~Constant folding through `Shape`~~ (done in `3ee9295`): stock YOLO11n
   imports, prepares and runs with ORT parity. The remaining shape work is
   arbitrary constant subgraphs that are neither shape arithmetic nor a single
   unary (a general `Constant` folder), which no pinned model needs yet.
2. ~~#117 remove the pre-v3 CPU runtime~~ (largely done): `model/{plan,
   session,specialize,kv,weight_int8}.rs`, `backend/cpu/quant/` and the
   quantized Vulkan kernels are deleted, the KV geometry types live in
   `import/graph.rs`, and the remaining test targets are v3-only. What is left
   is the portable kernel table in `backend/cpu/kernels/` (6.2k lines that
   `runtime/kernels` and `backend/vulkan` still call) and the `Backend`
   trait/`CpuBackend` exports; no session uses the legacy `Session`/`Plan`
   types because they no longer exist.
3. ~~#115 peak RSS~~ (done): row staging and oneDNN scratchpads are session-wide
   pools of one slot per worker instead of one buffer per block.
   `staging_bytes` at eight workers: 14,110,720 B → 2,048,000 B; the scratchpad
   pool replaces 27.3 MB with 3.7 MB. CLI peak RSS on yolov8n: ~106 MB → 76.5
   MB at eight, ~97 MB → 75.0 MB at four, ~59 MB → 52.4 MB at one, with no
   wall-time change outside host noise.
4. **Phase 5 W8A8** (#103): the gate refuses it with the measured reason on
   this AVX2 host: oneDNN's AVX2 int8 convolution is 0.97-1.20x the blocked
   f32 path (0.83-0.94 ms against 0.94-1.07 ms), because without VNNI the
   `vpmaddubsw`+`vpmaddwd` sequence has no MAC-throughput advantage over FMA.
   Real W8A8 needs VNNI/AMX hardware (where the same provider path uses
   `vpdpbusd`) and a calibration path; the precision plumbing
   (`Precision::W8A8`, ISA probe, refusal) exists.
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
