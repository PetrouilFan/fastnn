# Rejected spike: prepared transposed Conv runtime fallback

## Context

After `cd78cac feat(prepared): store transposed fp32 conv weights`, the prepared constant arena contained metadata-only transposed fp32 weights for the selected YOLO Conv+SiLU family:

```text
M=64, K=576
transposed fp32 conv entries: 13
transposed fp32 conv bytes: 1916928
```

A local uncommitted runtime spike attempted to consume those entries in an opt-in fallback path.

## Attempted implementation

The spike added:

- `conv2d_f32_im2col_gemm_weight_t(...)` in `src/backend/cpu/microkernels.rs`
- CPU dispatch kernel name `conv2d_silu_transposed_fp32`
- `GraphExecutor::execute_prepared_transposed_fallback(...)`
- Python method `AotExecutor.forward_prepared_transposed_fallback(inputs)`

The executor preloaded transposed fp32 bytes into the existing Conv weight arena slot, skipped matching `WriteConst`, and rewrote only matching `conv2d_silu` instructions to `conv2d_silu_transposed_fp32`.

## Correctness result

The path was exact vs default fastnn on YOLOv8n:

```text
shape: (1, 84, 2100)
max_abs default vs transposed fallback: 0.0
mean_abs default vs transposed fallback: 0.0
```

## Speed result

15-iteration local timing on the same YOLO input:

```text
forward              mean_ms ~63.01, median_ms ~64.19
arena_fallback        mean_ms ~68.19, median_ms ~69.11
transposed_fallback   mean_ms ~64.66, median_ms ~65.10
```

The transposed fallback was faster than the existing arena fallback, but did not beat default `forward()`.

## Verdict

Rejected and reverted before commit.

Do not reintroduce this runtime path unless the next design also removes enough prepared-fallback overhead to beat default `forward()`. The issue is not just transposed GEMM; the opt-in prepared fallback currently adds enough overhead that a modest GEMM-layout win is erased.

Better next directions:

1. Add profiling for prepared fallback overhead by phase: preload, plan rewrite, dispatch.
2. Avoid rebuilding/cloning a temporary `ExecutablePlan` per forward.
3. Make prepared execution a persistent compiled plan variant instead of a per-call rewrite.
4. Only then retry transposed fp32 Conv consumption.
