# YOLO constant overhead preflight

Roadmap Task B2 starts by measuring default-path constant materialisation before changing executor behavior.

`write_const` instructions currently materialise ONNX constants into the runtime arena at the start of every forward. They are not the dominant cost, but they are measurable after OpenBLAS reduces Conv GEMM time.

## Command

```bash
OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/yolo_constant_overhead.py \
  --runs 3 --warmup 3 --iters 10 \
  --json /tmp/yolo_constant_overhead.json
```

The script runs `scripts/yolo_compare_fastnn_pytorch.py --profile --profile-json ...` in repeated subprocesses, aggregates `fastnn_profile_summary`, and reports `write_const` and other non-Conv totals.

## Representative result

```text
fastnn_mean_ms mean=78.914 min=77.682 max=79.742
kernel totals mean_ms
  write_const            count=131 total_ms=1.560
  conv2d_silu            count= 57 total_ms=62.241
  conv2d                 count=  7 total_ms=0.759
  transpose_perm_f32     count=  2 total_ms=3.619
  pool_f32               count=  3 total_ms=1.508
  concat                 count= 17 total_ms=1.089
  slice_f32              count= 18 total_ms=0.535
  sigmoid_f32            count=  1 total_ms=0.809
  softmax                count=  1 total_ms=0.502
  upsample_nearest2d     count=  2 total_ms=0.191
  add_f32                count=  8 total_ms=0.208
  memcopy                count=  8 total_ms=0.163
write_const pct=1.98%
non_conv pct=12.91%
json /tmp/yolo_constant_overhead.json
```

## Interpretation

- `write_const` is measurable at ~1.6 ms, about 2% of this noisy full-YOLO sample.
- All 131 `write_const` instructions execute before the first kernel instruction in the profiled YOLO plan.
- A theoretical full elimination would clear the 1% small-cleanup gate, but only if it is actually safe and does not add comparable overhead elsewhere.
- A naive cross-forward skip is unsafe: the memory planner can reuse constant slots after their last consumer, so a cached arena does not imply those slots still contain constants at the next forward.
- A safe default-path constant cache would need one of:
  - persistent/non-reused slots for selected constants, increasing arena size;
  - a dispatch-time preload path with selected writes removed, similar to prepared arena fallback but without per-forward temporary plan cloning;
  - or a compile-time plan variant that separates persistent constants from per-forward instructions.

Task B2 verdict for this slice: land the measurement tool first. Do not make an executor behavior change until the implementation can prove both safety and >=1% full-YOLO speedup with unchanged accuracy.
