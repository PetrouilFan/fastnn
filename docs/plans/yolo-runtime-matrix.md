# YOLO runtime matrix

Roadmap Task A2 adds `scripts/yolo_runtime_matrix.py` to compare full YOLO CPU runtime settings across fastnn, raw PyTorch, and raw Ultralytics forward paths.

Each backend/thread configuration runs in a fresh subprocess so native thread settings are applied before importing fastnn, torch, OpenBLAS, or Ultralytics.

## Command

```bash
.venv/bin/python scripts/yolo_runtime_matrix.py \
  --threads 1,2,4,8 \
  --warmup 2 \
  --iters 3 \
  --json /tmp/yolo_runtime_matrix.json
```

The fastnn cases call `scripts/yolo_compare_fastnn_pytorch.py --profile --profile-json ...` under the selected `OPENBLAS_NUM_THREADS` setting, then read the exported profile JSON. PyTorch and Ultralytics raw cases run an inline `YOLO(...).model.eval().cpu()` forward with `torch.set_num_threads(...)` in the subprocess.

## Representative local result

Run after Task A2 implementation:

```text
fastnn threads=1 mean_ms=55.304
pytorch_raw threads=1 mean_ms=34.529
ultralytics_raw threads=1 mean_ms=34.010
fastnn threads=2 mean_ms=49.164
pytorch_raw threads=2 mean_ms=25.350
ultralytics_raw threads=2 mean_ms=26.135
fastnn threads=4 mean_ms=40.911
pytorch_raw threads=4 mean_ms=16.901
ultralytics_raw threads=4 mean_ms=18.765
fastnn threads=8 mean_ms=59.709
pytorch_raw threads=8 mean_ms=18.929
ultralytics_raw threads=8 mean_ms=18.595
best
  fastnn: threads=4 mean_ms=40.911 median_ms=41.287
  pytorch_raw: threads=4 mean_ms=16.901 median_ms=16.565
  ultralytics_raw: threads=8 mean_ms=18.595 median_ms=18.937
  fastnn/pytorch_raw ratio=2.421
  fastnn/ultralytics_raw ratio=2.200
json /tmp/yolo_runtime_matrix.json
```

JSON validation:

```text
json_ok 12 {'fastnn': (4, 40.911), 'pytorch_raw': (4, 16.901), 'ultralytics_raw': (8, 18.595)} {'fastnn_vs_pytorch_raw': 2.421, 'fastnn_vs_ultralytics_raw': 2.2}
```

## Current findings

- Best fastnn setting in this sample: `OPENBLAS_NUM_THREADS=4`, mean ~40.9 ms.
- Best raw PyTorch setting in this sample: `torch.set_num_threads(4)`, mean ~16.9 ms.
- Best raw Ultralytics setting in this sample: `torch.set_num_threads(8)`, mean ~18.6 ms.
- fastnn remains ~2.2-2.4x slower than the best raw PyTorch/Ultralytics forward settings in this matrix.
- Thread scaling is still non-monotonic: fastnn regressed at 8 OpenBLAS threads, while PyTorch/Ultralytics preferred 4-8 torch threads.

Use this matrix before accepting future YOLO CPU performance claims so fastnn changes are compared against current local PyTorch/Ultralytics thread baselines, not only prior fastnn timings.
