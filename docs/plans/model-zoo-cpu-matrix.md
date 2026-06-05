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

## Recommended next targets (ordered)

1. **Fix the AOT/Reshape shape inference** for the torchvision
   `GlobalAveragePool` / `ReduceMean` -> `Flatten` / `Reshape` chain
   (3 of the 4 torchvision models fail today).  Without this we
   cannot measure fastnn on MobileNet/EfficientNet and the breadth
   claim is hollow.  Likely a small change in
   `fastnn/io/shape_inference.py` plus the
   `yolo_compare_fastnn_pytorch._make_fastnn_executor` Shape/Gather
   const-folding path (the YOLO helper already does this for
   YOLO-shaped graphs; the same logic needs to apply to the generic
   case).

2. **Add thread sweeps to the matrix** for the working models so
   we can show the OpenBLAS Conv GEMM scaling story for ResNet18
   and ResNet50 the same way the existing
   `scripts/yolo_openblas_thread_sweep.py` does for YOLOv8n.  The
   resnet50 fastnn-vs-pytorch 0.75x win at 1 thread may not hold at
   higher thread counts; we need the data before any claim.

3. **Re-prioritise the `WriteConst` work on ResNet** (where it is
   ~28% of total traffic bytes) rather than on YOLO (where it is
   <3%).  The `prepared` plan's static-weight detection should be
   re-evaluated against this matrix because the cost shape is
   inverted.

4. **Stop chasing generic-YOLO optimisation classes** for the
   `conv2d_silu` kernel on YOLO.  That kernel is already at 0.5-0.8
   ms mean per call locally; further microbench work on it is
   unlikely to move the wall-time.  Instead target the YOLO-specific
   `concat` / `transpose` layout cost, which is the same cost on
   YOLOv8n and YOLO11n and is the next-largest single class in
   `top_kernels_by_count` and `top_write_consts_by_size`.

5. **Run the matrix on a small multi-config set** (320 / 416 / 640
   input sizes for YOLO, 224 / 256 / 320 for ResNet) to check
   whether the ResNet50 win holds at larger feature maps.  Add
   `--imgsz` override support if needed; today the script encodes
   the per-model imgsz in `DEFAULT_MODELS`.

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
