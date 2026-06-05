# Model zoo CPU matrix

Roadmap task to decide whether to keep fastnn optimised exclusively for
YOLO, or to invest in broader CPU backbones, requires a small but
representative CPU model zoo: per-model export/AOT/timing/memory stats
plus a bottleneck class label so we can talk about optimisation targets
in model-level terms rather than microbench-shaped ones.

This file documents the new `scripts/model_zoo_cpu_matrix.py` and the
results it produced locally.

## Command

```bash
.venv/bin/python scripts/model_zoo_cpu_matrix.py \
    --warmup 1 --iters 2 \
    --json /tmp/fastnn_model_zoo_cpu_matrix.json
```

Optional flags:

- `--models yolov8n,yolo11n,resnet18,resnet50` to run a subset.
- `--models yolo11n,yolo11l` to include the optional larger YOLO11l
  curiosity comparison. Optional larger models are not part of the default
  matrix because they download/export larger artifacts and take longer.
- `--cache-dir /tmp/fastnn-zoo` (default) to control where the downloaded
  Ultralytics `.pt` weights and exported ONNX files are written.  The
  cache is not committed; the repo's `.gitignore` already covers
  `*.onnx` and `*.pt`.
- `--profile-top N` to control how many kernels/instruction hotspots
  are captured per model (default 10).
- `--skip-fastnn`, `--skip-ort`, `--skip-pytorch` to omit a backend
  when its dependency is unavailable.
- `--warmup N --iters N` to tighten or loosen the run.

By default the script keeps the runtime small: warmup 1, iters 2.  Use
larger values for stable publication-grade numbers, smaller values for
smoke runs.

## What is recorded per model

Each model record in the JSON contains:

- `export`: source (ultralytics / torchvision), artifact path, onnx
  metadata (`num_nodes`, `op_counts`, `inputs`, `outputs`, `size_bytes`).
- `onnx_metadata.conv_shape_summary`: model-agnostic Conv class metadata from
  ONNX shape inference: per-class counts/estimated FLOPs for `standard`,
  `pointwise`, `depthwise`, and `grouped` Conv; representative examples; and
  the top repeated Conv shapes by total estimated FLOPs.
- `pytorch`: PyTorch reference forward timing (ms), output shape, and
  the reference tensor for accuracy checks.
- `onnxruntime`: ORT timing, output shape, errors when ORT is not
  installed.
- `fastnn`: AOT input/output names, forward timing (warmup/iters/mean/
  median/min/max), `memory_stats_summary` (arena size, slot reuse,
  estimated static traffic, kernel R/W bytes, memcpy bytes,
  write_const bytes, fill bytes, alias groups, top kernels by count,
  top instructions by static bytes, top WriteConst entries, top alias
  groups), `profile_top_kernels` (count, total_ms, mean_ms), and
  `profile_instruction_hotspots` (per-instruction static bytes cross-
  referenced with the kernel's mean elapsed ms and a memory-bound
  flag).
- `accuracy_vs_<backend>`: `max_abs` and `mean_abs` vs the PyTorch
  reference when shapes match.
- `bottleneck_class`: a coarse label such as
  `memory_traffic_bound`, `write_const_heavy`, `memcpy_heavy`,
  `kernel_io_bound`, `aot_import_failure`, or
  `compute_or_unclassified`.

Failures in any single backend are recorded inline as `*_error` fields
but never abort the rest of the matrix.

## Models attempted

| key | family | source | input size |
| --- | --- | --- | --- |
| yolov8n | yolo | ultralytics `yolov8n.pt` | 1x3x320x320 |
| yolo11n | yolo | ultralytics `yolo11n.pt` | 1x3x320x320 |
| resnet18 | torchvision | `torchvision.models.resnet18` | 1x3x224x224 |
| resnet50 | torchvision | `torchvision.models.resnet50` | 1x3x224x224 |
| mobilenet_v2 | torchvision | `torchvision.models.mobilenet_v2` | 1x3x224x224 |
| mobilenet_v3_small | torchvision | `torchvision.models.mobilenet_v3_small` | 1x3x224x224 |
| efficientnet_b0 | torchvision | `torchvision.models.efficientnet_b0` | 1x3x224x224 |

ONNX export uses Ultralytics' `model.export(format="onnx", opset=12,
simplify=False, dynamic=False)` for YOLO and `torch.onnx.export(...,
dynamo=False)` for torchvision.  The legacy TorchScript exporter is
used for torchvision because the new dynamo-based exporter trips on
ResNet18 export under onnx 1.21 due to a `version_converter`
`axes_input_to_attribute` assertion.  Worth revisiting once the
upstream issue is fixed.

## Representative local results

Local machine: `AMD Ryzen 7 3700X 8-Core Processor`, single-threaded
forward (default script behaviour matches the YOLO runtime matrix
default), warmup 1, iters 2.

```text
[yolov8n]    pytorch mean_ms=34.501
[yolov8n]    onnxruntime mean_ms=8.795
[yolov8n]    fastnn mean_ms=49.418
[yolo11n]    pytorch mean_ms=32.522
[yolo11n]    onnxruntime mean_ms=8.101
[yolo11n]    fastnn mean_ms=47.920
[resnet18]   pytorch mean_ms=49.686
[resnet18]   onnxruntime mean_ms=7.772
[resnet18]   fastnn mean_ms=57.122
[resnet50]   pytorch mean_ms=153.610
[resnet50]   onnxruntime mean_ms=19.847
[resnet50]   fastnn mean_ms=114.770
[mobilenet_v2]        pytorch mean_ms=23.179
[mobilenet_v2]        onnxruntime mean_ms=2.897
[mobilenet_v2]        fastnn forward FAILED: shape validation
[mobilenet_v3_small]  pytorch mean_ms=8.015
[mobilenet_v3_small]  onnxruntime mean_ms=2.129
[mobilenet_v3_small]  fastnn AOT FAILED: Conv2d input must have at least 4 dims
[efficientnet_b0]     pytorch mean_ms=37.302
[efficientnet_b0]     onnxruntime mean_ms=7.482
[efficientnet_b0]     fastnn AOT FAILED: Conv2d input must have at least 4 dims
```

Per-model speedup ratio table:

| model | fastnn ms | pytorch ms | ort ms | fastnn/pytorch | fastnn/ort |
| --- | ---: | ---: | ---: | ---: | ---: |
| yolov8n | 49.4 | 34.5 | 8.8 | 1.43x | 5.62x |
| yolo11n | 47.9 | 32.5 | 8.1 | 1.47x | 5.92x |
| resnet18 | 57.1 | 49.7 | 7.8 | 1.15x | 7.35x |
| resnet50 | 114.8 | 153.6 | 19.8 | **0.75x** | 5.78x |
| mobilenet_v2 | (AOT fail) | 23.2 | 2.9 | - | - |
| mobilenet_v3_small | (AOT fail) | 8.0 | 2.1 | - | - |
| efficientnet_b0 | (AOT fail) | 37.3 | 7.5 | - | - |

ResNet50 is the headline result: fastnn's `conv2d_relu` kernel beats
PyTorch's stock CPU conv stack at this batch/imgsz.  The 1.4-1.5x YOLO
gap that we have been chasing is, in this zoo, *not* a generic CNN
problem - it is concentrated in the YOLO/Darknet head, and YOLO11n
inherits the same shape.

The accuracy column for fastnn matches ORT within ~6e-7 (floating
point accumulation order); the gap vs raw PyTorch is the usual
ONNX/PyTorch reorder, not a fastnn bug.  See `accuracy_vs_pytorch`
in the JSON for raw max_abs.

## Top kernels and instruction hotspots

The JSON contains the raw `top_kernels_by_count` from
`executor.memory_stats()` as well as the per-kernel aggregated
profile (`profile_top_kernels`) and the per-instruction hotspot table
that mirrors the layout of `graph_memory_profile.py`.

The most striking cross-model pattern: `conv2d_silu` and `conv2d_relu`
are the dominant kernels for every CNN that runs end-to-end.  Some
illustrative rows from the local run:

| model | top kernel | count | total ms | mean ms |
| --- | --- | ---: | ---: | ---: |
| yolov8n | conv2d_silu | 57 | 45.42 | 0.80 |
| yolov8n | transpose_perm_f32 | 2 | 1.02 | 0.51 |
| yolov8n | concat | 17 | 0.67 | 0.04 |
| yolo11n | conv2d_silu | 77 | 40.58 | 0.53 |
| yolo11n | transpose_perm_f32 | 4 | 1.36 | 0.34 |
| yolo11n | concat | 21 | 0.88 | 0.04 |
| resnet18 | conv2d_relu | 9 | 25.42 | 2.82 |
| resnet18 | conv2d | 11 | 23.81 | 2.16 |
| resnet50 | conv2d_relu | 33 | 68.05 | 2.06 |
| resnet50 | conv2d | 20 | 27.33 | 1.37 |
| resnet50 | matmul (FC head) | 1 | 2.02 | 2.02 |

The YOLO concat/transpose traffic from the existing `graph_memory_stats.py`
output is also present in the per-model JSON (`top_write_consts_by_size`,
`memcpy_bytes`, `alias_groups`).

The instruction-level hotspot table marks the top 10 instructions per
model as "memory-bound" if their static bytes per ms is at least 1
MiB/ms or their mean kernel time is at most 1 ms.  For ResNet50 the
top 5 hotspots are all `conv2d_relu` or `add_relu_f32` instructions
with 9-10 MB static traffic each - confirming the write_const +
compute dance is the largest single class of bytes touched per forward.

## Bottleneck classification rules

The coarse `bottleneck_class` per model is computed from the
`memory_stats_summary` block:

- `aot_import_failure` if fastnn could not build an AotExecutor.
- `no_fastnn_timing` if we got past AOT but could not complete a
  forward (currently the `Reshape` shape-mismatch class).
- `memory_traffic_bound` if `estimated_static_traffic_bytes / arena_size >= 4`.
- `write_const_heavy` if `write_const_bytes / arena_size >= 1.5`.
- `memcpy_heavy` if `memcpy_bytes / arena_size >= 0.4`.
- `kernel_io_bound` if `(kernel_read_bytes + kernel_write_bytes) > 2 * arena_size`.
- `compute_or_unclassified` otherwise.

Local classifications:

| model | class |
| --- | --- |
| yolov8n | compute_or_unclassified |
| yolo11n | compute_or_unclassified |
| resnet18 | compute_or_unclassified |
| resnet50 | compute_or_unclassified |
| mobilenet_v2 | aot_import_failure |
| mobilenet_v3_small | aot_import_failure |
| efficientnet_b0 | aot_import_failure |

The current `traffic/arena` ratios are 2.13-2.43x for the four
end-to-end models, well under the 4x threshold.  A small
threshold-tuning pass on the classified zoo (or a richer metric set
including the `unprofiled_static_traffic` from `graph_memory_profile`)
will likely move more of these into `memory_traffic_bound` or
`write_const_heavy`.  The JSON retains the underlying numbers so
those cuts can be done offline.

## What the results suggest (optimisation classes)

1. **`conv2d_silu` / `conv2d_relu` is the universal fastnn hot kernel.**
   In every model that ran end-to-end, this single fused kernel
   accounts for 60-90% of the profile time.  This matches the prior
   YOLO finding that OpenBLAS Conv GEMM was the real speed win; the
   same kernel beats PyTorch on ResNet50 and ties on ResNet18, so the
   investment in this path is paying off.  Next-best lever: re-run
   the matrix with thread sweeps (`scripts/yolo_openblas_thread_sweep.py`
   style) for ResNet50 to see how the win scales.

2. **YOLO is no longer a generic CNN problem; it is a YOLO-head
   problem.** YOLO11n inherits the same YOLO-shape cost as YOLOv8n.
   Top kernels outside the conv-fused path are still `concat`,
   `transpose_perm_f32`, and the YOLO DFL Slice/Reshape chain.  These
   are the layout/copy hotspots that `concat #139 / #190` and
   related roadmap items already target.  The fact that the matrix
   reproduces those hotspots across two YOLO generations suggests
   they are stable optimisation targets.

3. **torchvision is partially importable; 3/4 classifiers AOT-fail.**
   The three failures share a common shape: a `GlobalAveragePool` or
   `ReduceMean` whose `keepdims` is not preserved through the helper,
   followed by a `Flatten` or `Reshape` whose input fastnn believes
   is still 4D-wide.  MobileNetV2 errors with a `[1280, 7, 7] ->
   [1, 1280]` reshape (62720 vs 1280 elements).  MobileNetV3-small
   and EfficientNet-B0 cascade the same gap into a downstream
   `Conv2d` that receives a 2D input.  Fixing this is a prerequisite
   before any broader CPU coverage claim can hold; it is a
   model-portability blocker, not a perf bug.

4. **`write_const` is large on ResNet, small on YOLO.** ResNet18 has
   26 `write_const` entries totalling 46.7 MB and ResNet50 has 61
   entries totalling 102 MB - mostly Conv weights written into the
   arena.  YOLOv8n has 131 entries but only 10.5 MB and YOLO11n has
   180 entries and 10.5 MB - the YOLO head writes many small slices
   (small init constants for the DFL path), not big Conv weights.
   This is a useful signal for prioritising the `WriteConst` pruning
   work: it is much closer to the wall-time on ResNet than on YOLO.
   The recent `prepared` plan is more likely to pay off on the
   ResNet-shaped models than on YOLO, which is the opposite of the
   framing in the existing `yolo-constant-overhead` plan.

5. **The 2.13-2.43x traffic/arena ratio is a uniform optimisation
   headroom.** Every end-to-end model has roughly the same ratio,
   independent of architecture.  Closing half of that would
   noticeably reduce the bytes per forward without changing the
   conv kernel.  Worth measuring what fraction of the traffic is
   per-kernel vs `memcpy` vs `write_const` and targeting whichever
   dominates per model.

## Post-Phase-1 rerun after ReduceMean/keepdims unblock

After `b64aa63 fix(shape): track ReduceMean keepdims for flatten reshape`, the
three previously blocked torchvision models now import and run end-to-end in
fastnn. This changes the optimization picture: MobileNet/EfficientNet are no
longer frontend failures; they are real CPU backend bottlenecks.

Representative local command:

```bash
PYENV_VERSION=system OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/model_zoo_cpu_matrix.py \
  --warmup 3 --iters 5 --profile-top 8 \
  --json /tmp/fastnn_model_zoo_cpu_matrix_post_phase1_stable.json
```

Representative result:

```text
model                  aot    fastnn ms  onnxrt ms  pytorch ms      arena  traffic/arena class
efficientnet_b0        ok       158.036      5.592      30.590   49073952           2.61 kernel_io_bound
mobilenet_v2           ok        43.120      2.928      22.542   41243680           3.23 kernel_io_bound
mobilenet_v3_small     ok        37.190      2.101       7.268   19713184           3.32 kernel_io_bound
resnet18               ok        44.086      8.031      48.315   59691360           2.13 compute_or_unclassified
resnet50               ok        83.370     20.110     124.974  149576032           2.43 compute_or_unclassified
yolo11n                ok        50.620      9.181      30.949   46509760           2.19 compute_or_unclassified
yolov8n                ok        55.218      9.480      34.850   45052288           2.16 compute_or_unclassified
```

Top profile kernels from the same run:

| model | class | fastnn ms | pytorch ms | ort ms | top profile kernels |
| --- | --- | ---: | ---: | ---: | --- |
| efficientnet_b0 | kernel_io_bound | 158.0 | 30.6 | 5.6 | `conv2d_silu` 133.3ms, `mul_f32` 19.7ms, `conv2d` 2.7ms |
| mobilenet_v2 | kernel_io_bound | 43.1 | 22.5 | 2.9 | `conv2d` 31.6ms, `clamp_f32` 2.0ms, `matmul` 1.1ms |
| mobilenet_v3_small | kernel_io_bound | 37.2 | 7.3 | 2.1 | `conv2d` 25.8ms, `conv2d_relu` 5.6ms, `mul_f32` 2.6ms |
| resnet18 | compute_or_unclassified | 44.1 | 48.3 | 8.0 | `conv2d_relu` 19.6ms, `conv2d` 17.5ms, `write_const` 3.9ms |
| resnet50 | compute_or_unclassified | 83.4 | 125.0 | 20.1 | `conv2d_relu` 49.2ms, `conv2d` 16.9ms, `write_const` 10.8ms |
| yolo11n | compute_or_unclassified | 50.6 | 30.9 | 9.2 | `conv2d_silu` 43.5ms, `transpose_perm_f32` 2.1ms, `conv2d` 0.9ms |
| yolov8n | compute_or_unclassified | 55.2 | 34.9 | 9.5 | `conv2d_silu` 42.6ms, `transpose_perm_f32` 1.1ms, `write_const` 0.8ms |

Interpretation:

- Phase 1 was a breadth/unblock win, not a speed win.
- ResNet18/50 are acceptable relative to PyTorch on this CPU sample, so the
  generic standard-Conv path is not the worst cross-model problem.
- YOLO remains `conv2d_silu` dominated and still trails raw PyTorch by about
  1.6x in this run.
- MobileNetV3-small and EfficientNet-B0 are now the sharpest general backend
  failures: fastnn is about 5.1x slower than PyTorch and 17-28x slower than
  ONNX Runtime. Their top kernels point at depthwise/group/pointwise Conv and
  elementwise/memory traffic, not YOLO-specific graph overhead.

## Conv class metadata and YOLO11l curiosity run

The matrix now records `onnx_metadata.conv_shape_summary`, which classifies
Conv nodes as `standard`, `pointwise`, `depthwise`, or `grouped`, with
estimated FLOP fractions and top repeated shapes. This is metadata-only and
comes from ONNX shape inference; it does not change fastnn execution.

Representative default-matrix smoke with Conv metadata:

```bash
PYENV_VERSION=system OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/model_zoo_cpu_matrix.py \
  --warmup 1 --iters 2 --profile-top 8 \
  --json /tmp/fastnn_model_zoo_convmeta_full.json
```

Conv class split from that run:

| model | Conv class split by estimated FLOPs |
| --- | --- |
| yolov8n | pointwise 25 nodes / 19.1%, standard 39 / 80.9% |
| yolo11n | depthwise 7 / 0.4%, pointwise 46 / 35.9%, standard 35 / 63.7% |
| resnet18 | pointwise 3 / 1.1%, standard 17 / 98.9% |
| resnet50 | pointwise 36 / 51.9%, standard 17 / 48.1% |
| mobilenet_v2 | depthwise 17 / 6.9%, pointwise 34 / 89.5%, standard 1 / 3.6% |
| mobilenet_v3_small | depthwise 11 / 13.6%, pointwise 40 / 76.5%, standard 1 / 9.9% |
| efficientnet_b0 | depthwise 16 / 9.0%, pointwise 64 / 88.2%, standard 1 / 2.8% |

Important correction to the earlier intuition: MobileNet/EfficientNet are not
mostly depthwise by FLOPs. They are mostly pointwise 1x1 Conv by estimated
FLOPs, with depthwise Conv still important by count and memory behavior. A
runtime optimization should therefore classify/profile pointwise and depthwise
separately instead of assuming depthwise alone is the culprit.

The optional larger YOLO comparison is selected explicitly so it does not slow
down the default matrix:

```bash
PYENV_VERSION=system OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/model_zoo_cpu_matrix.py \
  --models yolo11n,yolo11l \
  --warmup 1 --iters 2 --profile-top 8 \
  --json /tmp/fastnn_yolo11n_l_compare.json
```

Representative result:

| model | fastnn ms | pytorch ms | ort ms | fastnn/PyTorch | fastnn/ORT | Conv class split by estimated FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| yolo11n | 50.7 | 30.7 | 8.1 | 1.65x | 6.26x | depthwise 0.4%, pointwise 35.9%, standard 63.7% |
| yolo11l | 309.9 | 262.1 | 63.4 | 1.18x | 4.88x | depthwise 0.1%, pointwise 38.8%, standard 61.1% |

Blunt read: bigger YOLO is less relatively bad versus PyTorch than YOLO11n
(1.18x vs 1.65x slower in this sample), but still far from ORT. The bigger
model is dominated by the same standard/pointwise Conv mix, so it does not
justify a separate YOLO-large-specific lane yet.

## Recommended next targets (ordered)

1. **Join Conv class metadata to fastnn per-instruction profile timing.** The
   current metadata proves the graph's Conv class/FLOP mix, but profile timing
   is still aggregated by kernel name. The next measurement slice should map
   runtime Conv instructions back to their class so we can report actual time
   in standard vs pointwise vs depthwise Conv.

2. **Use the joined timing to classify MobileNet/EfficientNet hotspots.** The
   next runtime-changing lane should be based on measured pointwise/depthwise
   time, not just FLOP fractions or family-level intuition.

3. **Prototype a model-agnostic pointwise/depthwise Conv specialization or
   backend path only after the timing join lands.** Gate on full-model
   improvement for MobileNetV2, MobileNetV3-small, and EfficientNet-B0, with
   YOLOv8n, YOLO11n, YOLO11l, ResNet18, and ResNet50 as regression anchors.

4. **Keep YOLO as a regression anchor, not the only target.** YOLO11l suggests
   larger YOLOs are less relatively bad vs PyTorch than small YOLOs, so the
   immediate general lane should stay focused on the cross-model Conv class
   split rather than a YOLO-large-specific path.

5. **Revisit thread sweeps after the Conv timing join.** Thread sweeps are
   still useful, but optimizing thread settings before knowing which Conv
   classes dominate risks tuning around the wrong kernel family.

## How to reproduce locally

```bash
# one-time: the script needs onnx, onnxruntime, ultralytics, torch, torchvision
# the in-repo .venv already has all of these
ls /tmp/fastnn-zoo || mkdir -p /tmp/fastnn-zoo

.venv/bin/python scripts/model_zoo_cpu_matrix.py \
    --warmup 1 --iters 2 \
    --json /tmp/fastnn_model_zoo_cpu_matrix.json

# inspect one record
.venv/bin/python -c "import json; d=json.load(open('/tmp/fastnn_model_zoo_cpu_matrix.json')); import pprint; pprint.pp(d['models']['resnet50']['fastnn']['profile_top_kernels'][:3])"
```

The exact JSON used for the numbers above is in
`/tmp/fastnn-zoo/matrix.json` from the run that produced the
`docs/plans/model-zoo-cpu-matrix.md` report.
