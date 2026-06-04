# YOLO Conv shape targets for PreparedPlan performance

This note records the current measured Conv geometry in local `yolov8n.onnx` after PreparedPlan Wave 7. Use this before proposing packed/prepared Conv work.

Generated with:

```bash
.venv/bin/python scripts/yolo_conv_shape_stats.py --onnx yolov8n.onnx --top 20 --json /tmp/yolo_conv_shape_stats.json
```

## Counts

```text
total conv: 64
conv+silu patterns: 57
```

## Top grouped Conv+SiLU targets

Sorted by total estimated FLOPs within the YOLO graph:

```text
count=9 act=silu input=[1,64,20,20] weight=[64,64,3,3] stride=1 pad=1 gemm=[64,576,400] total_gflop=0.265
count=2 act=silu input=[1,64,40,40] weight=[64,64,3,3] stride=1 pad=1 gemm=[64,576,1600] total_gflop=0.236
count=1 act=silu input=[1,80,40,40] weight=[80,80,3,3] stride=1 pad=1 gemm=[80,720,1600] total_gflop=0.184
count=6 act=silu input=[1,32,40,40] weight=[32,32,3,3] stride=1 pad=1 gemm=[32,288,1600] total_gflop=0.177
count=1 act=silu input=[1,64,40,40] weight=[80,64,3,3] stride=1 pad=1 gemm=[80,576,1600] total_gflop=0.147
count=4 act=silu input=[1,128,10,10] weight=[128,128,3,3] stride=1 pad=1 gemm=[128,1152,100] total_gflop=0.118
```

## Implication

The best next performance target is not the C=3 stem. A naive direct-loop stem specialization was tested and rejected in `docs/plans/rejected-direct-stem-conv-spike.md`.

Prioritize repeated 3x3 Conv+SiLU shapes and packed/pretransposed GEMM/im2col layout improvements. Keep all performance work gated by:

- `scripts/conv_shape_microbench.py`
- full YOLO smoke/profile via `scripts/yolo_compare_fastnn_pytorch.py --profile`
- exactness vs the default fastnn path
