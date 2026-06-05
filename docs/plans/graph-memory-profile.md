# Graph memory profile

`scripts/graph_memory_profile.py` overlays dynamic `AotExecutor.profile()` timings with static `AotExecutor.memory_stats()` traffic estimates.

The goal is to rank broad graph/runtime memory opportunities before choosing an optimization lane. It is not YOLO-specific, although the default input shape matches YOLOv8n 320x320.

## Usage

```bash
PYENV_VERSION=system .venv/bin/python scripts/graph_memory_profile.py \
  --onnx yolov8n.onnx \
  --input-shape 1,3,320,320 \
  --top 20 \
  --json /tmp/graph_memory_profile_yolo.json
```

The JSON contains:

- `summary.profiled_total_ms`
- `summary.estimated_static_traffic_bytes`
- `summary.profiled_static_traffic_bytes`
- `summary.profiled_kernel_static_bytes`
- `summary.unprofiled_static_traffic_bytes`
- `kernels[]` rows with `kernel_name`, dynamic time, exact static bytes when available, bytes/ms, and a memory-bound heuristic
- `unprofiled_static_traffic[]` for static copy/const/fill traffic that was not present in dynamic profile entries
- raw `memory_stats` for cross-checking

## Estimation limits

`memory_stats()` exposes exact per-kernel aggregate read/write bytes in `top_kernels_by_count` (`read_bytes`, `write_bytes`, `static_bytes`) for the top kernel kinds. `scripts/graph_memory_profile.py` uses those fields when present, so broad kernel categories such as `concat`, `slice_f32`, and `conv2d_silu` are ranked by their actual compiled-plan byte traffic instead of by call-count apportioning. Older `memory_stats()` payloads without those fields still fall back to the prior call-count estimate.

The tool still aggregates by kernel name rather than by individual instruction/node. It handles profiled `memcopy`, `write_const`, and `fill` entries separately so copy/constant traffic is not double-counted.

Use this as a prioritization signal, not as a cycle-accurate profiler. If a decision depends on exact per-node bytes, extend `memory_stats()` to expose instruction-level traffic first.

## Representative local YOLOv8n smoke

Command:

```bash
PYENV_VERSION=system .venv/bin/python scripts/graph_memory_profile.py \
  --onnx yolov8n.onnx \
  --json /tmp/graph_memory_profile_yolo.json \
  --top 10
```

Observed on 2026-06-05 after exact per-kernel byte accounting:

```text
profiled total: 491.677 ms in instrumented profile mode
estimated static traffic: 92.81 MiB
profiled kernel static traffic: 79.05 MiB
unprofiled copy/const traffic: 0 B when memcopy/write_const appear in profile
```

Top memory/layout signals from exact per-kernel bytes were `conv2d_silu` (43.05 MiB), `concat` (16.12 MiB), `write_const` (12.06 MiB), `slice_f32` (7.13 MiB), and `add_f32` (3.17 MiB). Compared with the original call-count apportioning, `concat` is a stronger broad memory/layout target than `slice_f32`; `write_const` remains a clear per-forward traffic target for safe persistent-constant handling.
