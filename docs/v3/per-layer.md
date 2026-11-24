# Per-layer efficiency map: yolov8n at 320

- **Date:** 2026-11-24
- **Machine:** AMD Ryzen 7 3700X, 8 physical cores / 16 logical, AVX2 only
- **Model:** `yolov8n.onnx` (`/home/petrouil/.hermes/yolov8n.onnx`), input `f32[1,3,320,320]`
- **Build:** release, `--no-default-features --features cpu-onednn,parallel,segment-timing`
- **Protocol:** 3 warm-ups, then 3 groups of 3 forwards per thread count; each
  segment's group total is divided by 3 and the median over the three groups is
  reported. ONNX Runtime 1.32.0 is profiled with `intra_op_num_threads` set to
  the same count, 2 warm-ups + 3 runs, and the median of the per-run node totals.

The point of this document is to say **where** the remaining gap against ORT
is, not to claim a new number: the forward medians it produces agree with
[status.md](status.md) within host noise.

## Class totals

| class | fastnn 1T | ORT 1T | fastnn 8T | ORT 8T |
|---|---:|---:|---:|---:|
| convolution (incl. fused SiLU) | 30.67 ms | 21.62 ms¹ | 14.79 ms | 4.99 ms¹ |
| max pool † | 2.28 | 0.08 | 1.05 | 0.04 |
| resize / upsample † | 1.51 | 0.04 | 0.33 | 0.15 |
| softmax | 0.42 | 0.18 | 0.35 | 0.03 |
| elementwise (sigmoid / add) | 0.47 | 0.15² | 0.48 | 0.20² |
| concat | 0.30 | 0.33 | 0.65 | 0.52 |
| slice | 0.23 | 0.01 | 0.26 | 0.02 |
| transpose | 0.15 | 0.50 | 0.28 | 0.78 |
| split + reshape | 0.12 | 0.11 | 0.37 | 0.23 |
| layout reorders | —³ | 0.45 | —³ | 0.78 |
| **forward** | **36.13** | **23.47** | **18.95** | **7.75** |

¹ ORT time plus its `QuickGelu` nodes (1.87 ms at 1T, 1.01 ms at 8T), which are
the same SiLU pattern fastnn fuses into the convolution. The fastnn number is
`conv+swish` plus the seven unfused `Conv` segments (0.59 ms at 1T, 0.54 ms at
8T), so both sides cover the same work.
² ORT's `elementwise` bucket; it fuses the SiLU but not the DFL sigmoid.
³ fastnn's session-open reorders are part of session open, not a forward; the
runtime conversions that remain are inside the segments that need them.

† Both tails have been fixed since this map was taken; the class rows above are
the baseline they were measured against. See
[The two tails](#the-two-tails-pooling-and-resize).

## Convolution detail

64 of the 126 segments are convolutions. At one thread the big 3×3 layers are
already close to ORT; the loss is concentrated in large-spatial, low-channel
layers, most of them 1×1:

| segment | shape (in → out) | fastnn 1T | ORT 1T | ratio | fastnn GMAC/s | 8T scaling |
|---|---|---:|---:|---:|---:|---:|
| `model.0/conv` | 3×320×320 → 16×160×160, 3×3/2 | 1.387 ms | 0.359 ms | 3.86× | 8.0 | 2.8× |
| `model.2/cv2` | 48×80×80 → 32×80×80, 1×1 | 0.831 | 0.167 | 4.98× | 11.8 | 1.9× |
| `model.22/cv3.0.1` | 80×40×40 → 80×40×40, 3×3 | 1.840 | 1.679 | 1.10× | 50.1 | 3.6× |
| `model.22/cv3.0.0` | 64×40×40 → 80×40×40, 3×3 | 1.604 | 1.342 | 1.20× | 46.0 | 3.5× |
| `model.2/cv1` | 32×80×80 → 32×80×80, 1×1 | 0.733 | 0.112 | 6.54× | 8.9 | 2.0× |
| `model.4/cv1` | 64×40×40 → 64×40×40, 1×1 | 0.430 | 0.109 | 3.94× | 15.2 | 1.9× |
| `model.2/m.0/cv2` | 16×80×80 → 16×80×80, 3×3 | 0.814 | 0.385 | 2.12× | 18.1 | 2.4× |
| `model.22/cv2.0.1` | 64×40×40 → 64×40×40, 3×3 | 1.242 | 0.970 | 1.28× | 47.5 | 3.5× |
| `model.22/cv2.0.0` | 64×40×40 → 64×40×40, 3×3 | 1.237 | 0.975 | 1.27× | 47.7 | 3.5× |

Matched convolution time, all 64 segments: **30.67 ms vs 19.68 ms (1.56×)** at
one thread, and **14.79 ms vs 4.24 ms (3.49×)** at eight. The 1T ratio becomes
**1.42×** when ORT's separate `QuickGelu` nodes are added to its convolution
time; the 8T ratio becomes **2.96×**.

Scaling from one to eight threads: fastnn convolutions **2.07×**, ORT
convolutions **4.64×**. The head 3×3 layers account for most of the absolute
difference: they scale 3.5× here against 6.6× for ORT.

## What this rules out, and what it points at

- **Winograd and a hand-written int8 kernel are not the next move.** The main
  3×3 shapes run at 46–50 GMAC/s at one thread, and the isolated f32 oneDNN probe
  ran three 59M-MAC shapes in 0.94–1.07 ms, i.e. ~90 % of this core's f32
  peak. The loss is in layers the big-kernel work does not address.
- **The 1T loss is a policy problem on low-channel, large-spatial layers.**
  `model.0`, `model.2/cv1`, `model.2/cv2` and `model.4/cv1` are 1×1 or 3-channel
  input convolutions; they run at 8–15 GMAC/s while ORT's NCHWc kernels reach
  31–60. The engine sends every conv edge in a chain through the channels-last
  (NHWC) destination; for 16–48 channels that layout is leaving a 3–6× gap.
  The first experiment is a policy split: keep channels-last for the ≥64-channel
  layers, use NCHW for the small ones, and measure the same table.
- **The tails were cheap wins, and they are taken.** Both are now fixed; the
  measurement is in [the next section](#the-two-tails-pooling-and-resize).
- **At 8T the question is block strategy, not arithmetic.** The head 3×3 layers
  reach 165–180 GMAC/s (2.5–2.8× one core's peak) but scale only 3.5× on eight
  workers, while ORT's NCHWc kernels reach 6.6×. A thread-count sweep on one
  segment (1/2/4/8, same block split) is the measurement that separates pool
  dispatch cost from memory/CCX saturation.

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
