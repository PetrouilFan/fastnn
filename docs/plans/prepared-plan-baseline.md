# PreparedPlan Baseline Metrics

> Captured on 2026-06-03 from branch `agent/baseline-real` (commit 769b935).

## Test Results

```
cargo test --release --lib
168 passed; 0 failed; 0 ignored
```

## YOLOv8n Comparison (batch=1, 320x320, CPU, warmup=3, iters=8)

| Backend        | Mean (ms) | Median (ms) | Min (ms) | Max (ms) |
|----------------|-----------|-------------|----------|----------|
| PyTorch        | 38.22     | 39.24       | 34.40    | 40.33    |
| ONNX Runtime   | 9.25      | 9.24        | 9.20     | 9.34     |
| FastNN         | 60.49     | 59.91       | 57.69    | 65.37    |

**Accuracy (FastNN vs PyTorch):** max_abs=4.58e-04, mean_abs=1.40e-06

**Accuracy (ONNX Runtime vs PyTorch):** max_abs=5.19e-04, mean_abs=1.39e-06

## YOLO Profile Top 12 Kernels (fastnn)

| Kernel              | Calls | Mean (ms) | Total (ms) | % of Total |
|---------------------|-------|-----------|------------|------------|
| conv2d_silu         | 57    | 0.922     | 52.55      | 86.9%      |
| transpose_perm_f32  | 2     | 1.476     | 2.95       | 4.9%       |
| write_const         | 131   | 0.007     | 0.98       | 1.6%       |
| concat              | 17    | 0.041     | 0.70       | 1.2%       |
| conv2d              | 7     | 0.097     | 0.68       | 1.1%       |
| sigmoid_f32         | 1     | 0.570     | 0.57       | 0.9%       |
| slice_f32           | 18    | 0.022     | 0.39       | 0.6%       |
| pool_f32            | 3     | 0.118     | 0.35       | 0.6%       |
| softmax             | 1     | 0.179     | 0.18       | 0.3%       |
| add_f32             | 8     | 0.019     | 0.15       | 0.2%       |
| upsample_nearest2d  | 2     | 0.056     | 0.11       | 0.2%       |
| memcopy             | 8     | 0.013     | 0.11       | 0.2%       |

**Key insight:** `conv2d_silu` dominates at 86.9% of total inference time (52.55ms / 60.49ms). This is the primary optimization target for PreparedPlan.

## Conv Shape Microbench (warmup=10, iters=40, threads=1)

| Shape                        | Ratio | FastNN (ms) | PyTorch (ms) | GFLOP/s Fast | GFLOP/s Torch | GEMM (M,N,K)    |
|------------------------------|-------|-------------|--------------|--------------|---------------|-----------------|
| yolo_3x3_f32_c16_sp6400      | 6.11x | 5.716       | 0.935        | 10.3         | 63.1          | (32, 144, 6400) |
| yolo_stem_3x3_f16_c3_sp25600 | 4.83x | 4.502       | 0.932        | 4.9          | 23.7          | (16, 27, 25600) |
| yolo_3x3_f16_c16_sp6400      | 3.16x | 1.827       | 0.578        | 16.1         | 51.0          | (16, 144, 6400) |
| yolo_3x3_f64_c64_sp400       | 2.26x | 0.853       | 0.377        | 34.6         | 78.3          | (64, 576, 400)  |
| yolo_1x1_f64_c64_sp1600      | 2.18x | 0.555       | 0.254        | 23.6         | 51.6          | (64, 64, 1600)  |
| yolo_3x3_f32_c32_sp1600      | 2.08x | 0.910       | 0.437        | 32.4         | 67.4          | (32, 288, 1600) |
| yolo_1x1_f128_c192_sp400     | 1.44x | 0.502       | 0.348        | 39.1         | 56.5          | (128, 192, 400) |
| yolo_3x3_f128_c128_sp100     | 0.95x | 0.460       | 0.482        | 64.1         | 61.2          | (128, 1152, 100)|

**Key insights:**
- Small-channel shapes (c3, c16) show the largest gaps (3-6x slower than PyTorch).
- The c128 shape is already competitive (0.95x ratio, actually faster than PyTorch).
- GFLOP/s utilization is 2-6x lower than PyTorch for small-channel convolutions.
- PreparedPlan weight prepacking should target the high-ratio shapes first.

## Commands Used

```bash
cargo test --release --lib
VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv /home/petrouil/Projects/github/fastnn/.venv/bin/python -m maturin develop --release
VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv /home/petrouil/Projects/github/fastnn/.venv/bin/python scripts/yolo_compare_fastnn_pytorch.py --profile --profile-top 12 --warmup 3 --iters 8
VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv /home/petrouil/Projects/github/fastnn/.venv/bin/python scripts/conv_shape_microbench.py --warmup 10 --iters 40 --threads 1
```
