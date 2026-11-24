# Per-layer efficiency map: yolov8n at 320

- **Date:** 2026-11-24
- **Machine:** AMD Ryzen 7 3700X, 8 physical cores / 16 logical, AVX2 only
- **Model:** `yolov8n.onnx` (`/home/petrouil/.hermes/yolov8n.onnx`), input `f32[1,3,320,320]`
- **Build:** release, `--no-default-features --features cpu-onednn,parallel,segment-timing`,
  at Cargo's default `x86-64` target. The same tree built for `x86-64-v3` is
  5-9 % faster at two to eight workers (see
  [backends.md](backends.md#the-fallback-path-is-compiled-for-sse2)); the table
  here is the baseline build, and a switch of target means re-measuring this
  map.
- **Protocol:** the whole harness runs three times, interleaved with any A/B
  build being compared, and every number is the median of the three runs. Inside
  one run: 3 warm-ups, then 3 groups of 3 forwards per thread count, each
  segment's group total divided by 3 and medianed over the groups. ONNX Runtime 1.32.0 is profiled with `intra_op_num_threads` set to
  the same count, 2 warm-ups + 3 runs, and the median of the per-run node totals.

The point of this document is to say **where** the remaining gap against ORT
is, not to claim a new number: the forward medians it produces agree with
[status.md](status.md) within host noise.

## Class totals

| class | fastnn 1T | ORT 1T | fastnn 8T | ORT 8T |
|---|---:|---:|---:|---:|
| convolution (incl. fused SiLU) | 24.94 ms | 21.62 ms¹ | 13.14 ms | 4.99 ms¹ |
| max pool | 0.50 | 0.08 | 0.15 | 0.04 |
| resize / upsample | 0.30 | 0.04 | 0.17 | 0.15 |
| softmax | 0.43 | 0.18 | 0.37 | 0.03 |
| elementwise (sigmoid / add) | 0.50 | 0.15² | 0.45 | 0.20² |
| concat | 0.29 | 0.33 | 0.62 | 0.52 |
| slice | 0.24 | 0.01 | 0.25 | 0.02 |
| transpose | 0.16 | 0.50 | 0.26 | 0.78 |
| split + reshape | 0.11 | 0.11 | 0.37 | 0.23 |
| layout reorders | —³ | 0.45 | —³ | 0.78 |
| **forward** | **27.68** | **23.47** | **15.97** | **7.75** |

¹ ORT time plus its `QuickGelu` nodes (1.87 ms at 1T, 1.01 ms at 8T), which are
the same SiLU pattern fastnn fuses into the convolution. The fastnn number is
`conv+swish` plus the seven unfused `Conv` segments (0.59 ms at 1T, 0.53 ms at
8T), so both sides cover the same work.
² ORT's `elementwise` bucket; it fuses the SiLU but not the DFL sigmoid.
³ fastnn's session-open reorders are part of session open, not a forward; the
runtime conversions that remain are inside the segments that need them.

Both tails are in the table as they now run; the measurement and the two
kernels are in [The two tails](#the-two-tails-pooling-and-resize).

## Convolution detail

64 of the 126 segments are convolutions. At one thread the big 3×3 layers are
now at parity; what is left is concentrated in large-spatial, low-channel
layers:

| segment | shape (in → out) | fastnn 1T | ORT 1T | ratio | fastnn GMAC/s | 8T scaling |
|---|---|---:|---:|---:|---:|---:|
| `model.22/cv3.0.1` | 80×40×40 → 80×40×40, 3×3 | 1.661 ms | 1.677 ms | 0.99× | 55.5 | 3.5× |
| `model.22/cv3.0.0` | 64×40×40 → 80×40×40, 3×3 | 1.387 | 1.341 | 1.03× | 53.2 | 3.6× |
| `model.22/cv2.0.1` | 64×40×40 → 64×40×40, 3×3 | 1.097 | 0.973 | 1.13× | 53.7 | 3.5× |
| `model.22/cv2.0.0` | 64×40×40 → 64×40×40, 3×3 | 1.087 | 0.969 | 1.12× | 54.2 | 3.4× |
| `model.2/m.0/cv2` | 16×80×80 → 16×80×80, 3×3 | 0.693 | 0.385 | 1.80× | 21.3 | 2.2× |
| `model.3/conv` | 32×80×80 → 64×40×40, 3×3/2 | 0.652 | 0.487 | 1.34× | 45.3 | 3.6× |
| `model.1/conv` | 16×160×160 → 32×80×80, 3×3/2 | 0.635 | 0.487 | 1.30× | 46.5 | 3.5× |
| `model.0/conv` | 3×320×320 → 16×160×160, 3×3/2 | 0.622 | 0.373 | 1.67× | 17.8 | 2.3× |
| `model.2/cv2` | 48×80×80 → 32×80×80, 1×1 | 0.454 | 0.164 | 2.77× | 21.7 | 1.4× |
| `model.2/cv1` | 32×80×80 → 32×80×80, 1×1 | 0.331 | 0.112 | 2.96× | 19.8 | 1.4× |
| `model.4/cv1` | 64×40×40 → 64×40×40, 1×1 | 0.218 | 0.109 | 2.00× | 30.1 | 1.3× |

Matched convolution time, all 64 segments: **24.94 ms vs 19.70 ms (1.27×)** at
one thread, and **13.14 ms vs 3.92 ms (3.35×)** at eight. The 1T ratio becomes
**1.16×** when ORT's separate `QuickGelu` nodes are added to its convolution
time; the 8T ratio becomes **2.67×**.

Scaling from one to eight threads: fastnn convolutions **1.90×**, ORT
convolutions **5.02×**. The head 3×3 layers account for most of the absolute
difference: they scale 3.5× here against 6.6× for ORT.

## What this rules out, and what it points at

- **Winograd and a hand-written int8 kernel are not the next move.** The main
  3×3 shapes run at 51–55 GMAC/s at one thread, and the isolated f32 oneDNN
  probe ran three 59M-MAC shapes in 0.94–1.07 ms, i.e. ~90 % of this core's f32
  peak. The loss is in layers the big-kernel work does not address.
- **The 1T loss on low-channel layers was the scalar epilogue, not the
  layout.** When this map was first taken, `model.0`, `model.2/cv1`,
  `model.2/cv2` and `model.4/cv1` ran at 8–15 GMAC/s and looked like a
  layout-policy problem. The standalone-eltwise change (PR #124) took them to
  17.7, 19.7, 21.4 and 28.6 GMAC/s with no layout change at all: most of their
  time was the per-element `exp`. Two layout policies were then measured
  anyway — a channel floor on the blocked destination, and plain NCHW for narrow
  1×1 edges — and both lost or tied; the numbers are in
  [the threads section](#threads-where-eight-workers-stop-paying).
- **The tails were cheap wins, and they are taken.** Both are now fixed; the
  measurement is in [the next section](#the-two-tails-pooling-and-resize).
- **At 8T the question is per-parallel-region overhead, not block strategy.**
  The head 3×3 layers reach 196 GMAC/s (2.5× one core's peak) but four workers
  beat eight on the forward, and one to two workers is only 1.38×. A
  `prefer_rows` change that fixed the two worst segments at eight workers did
  not move the total, while a layout change did: the measurements and the
  rejections are in [the threads section](#threads-where-eight-workers-stop-paying).

## The two tails: pooling and resize

Pooling no longer goes through the provider. The engine handed oneDNN the
arena's plain NCHW descriptor, and on this AVX2 host `jit_uni_pool_kernel.cpp`
leaves `ncsp_fmt_tag` unset unless the ISA is AVX-512, so the primitive oneDNN
built was `ref_pooling`: 0.98 / 0.69 / 0.60 ms for the three SPPF windows of
`model.9`. The claim is gone (the engine implements convolution and matmul) and
the scalar kernel implements all three pool ops.

The scalar kernel also grew a 2-D form. A maximum over a rectangle is the
maximum over the rows of the per-row horizontal maxima, so one pass over the
plane (`in_h * out_w` values) feeds the vertical pass instead of re-scanning 25
taps per output, and one task is a whole `(n, c)` plane so the horizontal
scratch is allocated once. Average pooling keeps a direct sum, still without the
generic loop's per-element coordinate decomposition, and accumulates in the same
tap order so its values are bit-identical to the generic kernel.

`Resize` now resolves the width axis once per forward: a run-length table for
nearest (`output columns start..end read input column source`, written as a
cursor walk) and a two-tap table for linear. The tables hold exactly the indices
and weights the direct form computed, so the values are unchanged.

| row | before | after | ORT 1T |
|---|---:|---:|---:|
| max pool (3 segments) | 2.28 ms | 0.51 ms | 0.08 |
| resize (2 segments) | 1.51 ms | 0.31 ms | 0.04 |
| forward, 1 thread | 36.13 ms | 34.72 ms | 23.47 |
| forward, 8 threads | 18.95 ms | 17.34 ms | 7.75 |

The forward rows are one run each, and this host's run-to-run spread on an
unchanged build is about 6 % (measured at 36.13 and 38.43 ms), so the class rows
— which moved by 4.5x and 4.9x — are the evidence, not the totals. Eight-thread
resize is at ORT parity (0.17 against 0.15); the one-thread resize gap is 0.27 ms
and would need a vectorised row writer to close.

## Threads: where eight workers stop paying

The map above is one and eight workers. The sweep between them is the
measurement that separates arithmetic from scheduling: the same build, the same
plan, only `Threading::Fixed(n)` changes. Three interleaved runs per count,
median reported, the range in brackets.

| workers | forward | from one | head 3x3 `cv3.0.1` | its GMAC/s |
|---:|---:|---:|---:|---:|
| 1 | 27.68 ms [25.95, 29.81] | 1.00x | 1.661 ms | 55 |
| 2 | 20.03 [19.08, 20.79] | 1.38x | 0.978 | 94 |
| 4 | 15.40 [14.64, 15.73] | 1.80x | 0.534 | 172 |
| 8 | 15.97 [15.85, 16.52] | 1.73x | 0.470 | 196 |

Three things fall out of it:

- **Four workers is the optimum, not eight**, and the two ranges do not
  overlap. Every large convolution is still getting faster at eight
  (`cv3.0.1` 0.534 -> 0.470 ms, 196 GMAC/s, 2.5x one core's f32 peak), so the
  graph is not memory-bound at eight; the extra workers are being spent
  elsewhere.
- **41 of 126 segments are more than 30 % slower at eight workers than at
  four** (medians). They are the small, short-spatial ones:
  `cv3.1.0.0` (64x20x20 -> 80x20x20, 3x3) runs 0.211 ms at four workers and
  0.716 at eight, 3.4x worse. Those are the segments where the row split
  yields fewer blocks than workers, so `prefer_rows` gives up and the split
  moves to channels, where each channel block re-reads the whole source.
- **One to two workers is 1.38x, not 2x**, and 63 of the 126 segments gain
  less than 5 % at two workers (they total 2.15 ms at one thread). The shortfall
  is below, joined to two measurements that rule out the obvious suspects.

Three policy changes were measured against this, and all three were rejected:

1. *Plain NCHW for every narrow edge* (a 64-channel floor on the blocked
   destination). Interleaved A/B, three runs each: 1T 29.16 ms against 28.01,
   8T inside noise. It doubles `model.1`'s time (0.626 -> 1.432 ms at 1T) - a
   narrow 3x3 reads each source pixel nine times and does want the
   channels-last source.
2. *Plain NCHW for narrow 1x1 edges only.* One edge changes (`model.2/cv2`
   0.450 -> 0.342 ms at 1T), but its consumer is a 3x3 that loses what the
   producer gained (0.650 -> 0.710): 1T 27.63 against 27.87, 8T 14.96 against
   15.50, a wash inside this host's noise.
3. *Letting the row split win with half the channel blocks* (`prefer_rows`
   relaxed from equal counts). It fixes the two pathological segments
   (`cv3.1.0.0` 0.716 -> 0.419 ms at 8T) but costs two workers 6 % (19.1-20.8
   against 20.6-22.1, disjoint) and moves the eight-worker total only 3 %, so it
   is a loss. The pathology needs a two-dimensional split - rows, then channels
   inside a row block - not a different tie-break.

### Where the eight-worker loss actually is

The ~50 us per segment above was an inference from the 1->2 shortfall, and it is
wrong. Two measurements replace it:

**Dispatch is 5-8 us, not 50.** `SessionThreads::run_blocks` on an empty body,
release build, 2000 iterations per cell (`cargo test --release --lib --features
parallel dispatch_cost -- --ignored --nocapture`):

| workers | 1 block | 2 blocks | 4 blocks | 8 blocks | `install` only |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 us | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 0.00 | 4.58 | 5.26 | 5.11 | 2.52 |
| 4 | 0.00 | 5.23 | 5.15 | 5.68 | 3.92 |
| 8 | 0.00 | 6.12 | 7.22 | 7.92 | 5.53 |

A forward has 126 segments, so the whole scheduling bill at eight workers is
about **1 ms**, not the 6 ms the 1->2 shortfall loses.

**The runtime scales; the engine does not.** The same harness and the same
segments through the scalar reference engine: 1214.56 / 641.23 / 385.37 /
228.64 ms at 1/2/4/8 workers, i.e. **1.89x at two and 5.31x at eight**. Pinning
the oneDNN sweep to eight distinct physical CPUs (`taskset -c 0-7`) changes
little (15.12 ms at eight against 15.97 unpinned), so it is not the scheduler
putting two workers on one core, and a block cap of 2 is clearly worse (22.81
ms at eight) while 4 against 8 cannot be separated: the run-to-run spread at
four workers was 14.85-17.90 ms in that session.

So the loss is inside the engine's per-block work. `OneDnnReport::segment_bytes`
now carries the per-segment account (blocks, payload, staged bytes in and out,
weight bytes), which is a count and therefore does not move with host noise.
`FN3_CNN_TRACE=1` prints it after the timing lines as `[fn3perf][bytes]`.

### Where the bytes go

Per forward over the 64 convolution segments, at eight workers:

| stream | bytes |
|---|---:|
| activation payload in + out | 35.65 MB |
| staged copies (row blocks) | 14.11 MB |
| weight reads, summed over blocks | **60.77 MB** |

The copy hypothesis is dead twice over. A staged segment copies 1.0-1.2x its own
payload (the halo), the channels-last segments copy nothing, and staging totals
14 MB against 35 MB of payload: at 14.68 ms a forward that is under 1 GB/s. What
grows with the block count is **weight traffic** — every block reads the whole
weight tensor, so the forward reads 60.77 MB at eight workers against 7.6 MB at
one. That is the price of blocks, and it explains why the floor sweep below
4 Ki elements found nothing: more blocks buy parallelism and pay for it in
re-read weights.

The segments that still do not scale are the ones that cannot reach four blocks.
Thirteen of the 64 cannot, and they are a large share of what is left:

| conv segments | count | 1T | 8T | scaling |
|---|---:|---:|---:|---:|
| four blocks or more | 51 | 23.10 ms | 9.74 ms | 2.37x |
| three blocks or fewer | 13 | 2.80 ms | 2.40 ms | **1.17x** |

Those thirteen are 16 % of the 14.68 ms forward at eight workers while they are
10 % of it at one. They are the 10x10 grids: a 3x3 output row at that height
needs three input rows, so a row split produces three blocks, and a channel
split is capped by the same 4 Ki-element floor that the sweep measured as
optimal. The split policy is therefore at its measured optimum; what is left on
the eight-worker side is the engine's **per-block fixed cost** — a stream, a set
of memory objects and a scratchpad lease per block — which is a session-level
redesign (persistent per-worker state), not a policy knob.

## Reproducing

```bash
FN3_CNN_PERF=onednn FN3_CNN_THREADS=1 FN3_CNN_TRACE=1 \
  cargo test --release --no-default-features \
    --features cpu-onednn,parallel,segment-timing \
    --test fn3_cnn_perf -- --nocapture
```

`FN3_CNN_TRACE=1` prints one line per plan segment: index, label, median
milliseconds, calls per forward, MACs, achieved GMAC/s, output bytes and the
source node. Without it the table is the label aggregation only.

The ORT side is [`tools/parity/ort_profile.py`](../../tools/parity/ort_profile.py):

```bash
python tools/parity/ort_profile.py /path/yolov8n.onnx --threads 1 --runs 3
python tools/parity/ort_profile.py /path/yolov8n.onnx --threads 8 --runs 3 \
    --json-out /tmp/ort-8t.json
```

It opens an `InferenceSession` with `enable_profiling` and
`intra_op_num_threads` set to the same count, runs 2 warm-ups + 3 timed
forwards, groups per-node durations by `args.op_name`, and prints the class
medians and the 15 slowest nodes.
