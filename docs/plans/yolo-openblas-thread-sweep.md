# YOLO OpenBLAS thread sweep

`scripts/yolo_openblas_thread_sweep.py` benchmarks fastnn YOLO inference under multiple `OPENBLAS_NUM_THREADS` settings.

Build first with OpenBLAS enabled:

```bash
VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv \
  .venv/bin/python -m maturin develop --release --features 'prepared-plan openblas'
```

Run:

```bash
.venv/bin/python scripts/yolo_openblas_thread_sweep.py \
  --threads 1,2,4,8 \
  --warmup 2 \
  --iters 3 \
  --json /tmp/yolo_openblas_thread_sweep.json
```

The script launches each measurement in a fresh subprocess so `OPENBLAS_NUM_THREADS` is set before importing fastnn/OpenBLAS.

Representative local run after `8cba595`:

```text
OpenBLAS thread sweep
onnx=yolov8n.onnx warmup=2 iters=3
 threads    mean_ms  median_ms     min_ms     max_ms   conv_silu_ms      max_abs
       1     60.995     61.512     58.401     63.071         43.319    4.272e-04
       2     44.878     42.563     42.080     49.993         37.134    4.883e-04
       4     40.496     39.009     39.005     43.474         37.283    5.188e-04
       8     60.128     60.057     57.271     63.055         51.037    3.967e-04
best threads=4 mean_ms=40.496 median_ms=39.009
```

Earlier longer samples favored `OPENBLAS_NUM_THREADS=2`. The conclusion is not that one thread count is universal; it is that OpenBLAS threading has a strong non-monotonic effect and must be swept on the target machine/configuration. Avoid hard-coding 8 threads just because the machine has more cores.

Suggested policy:

- Keep OpenBLAS as an opt-in build feature.
- Do not set `OPENBLAS_NUM_THREADS` inside fastnn yet.
- For deployment/benchmark runs, sweep `1,2,4,8` and pin the best externally.
- Treat `8` as suspicious until measured; it regressed in multiple local samples.
