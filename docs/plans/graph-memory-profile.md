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
- `kernels[]` rows with `kernel_name`, dynamic time, estimated static bytes, bytes/ms, and a memory-bound heuristic
- `unprofiled_static_traffic[]` for static copy/const/fill traffic that was not present in dynamic profile entries
- raw `memory_stats` for cross-checking

## Estimation limits

`memory_stats()` currently exposes kernel read/write bytes as aggregate totals and kernel call counts, not exact per-instruction byte traffic. The script apportions aggregate kernel bytes by static call count. It handles profiled `memcopy`, `write_const`, and `fill` entries separately so copy/constant traffic is not double-counted.

Use this as a prioritization signal, not as a cycle-accurate profiler. If a decision depends on exact per-node bytes, extend `memory_stats()` to expose instruction-level traffic first.

## Representative local YOLOv8n smoke

Command:

```bash
PYENV_VERSION=system .venv/bin/python scripts/graph_memory_profile.py \
  --onnx yolov8n.onnx \
  --json /tmp/graph_memory_profile_yolo.json \
  --top 10
```

Observed on 2026-06-05 after the first implementation:

```text
profiled total: ~640 ms in instrumented profile mode
estimated static traffic: 92.81 MiB
profiled kernel static traffic: ~80 MiB
unprofiled copy/const traffic: 0 B when memcopy/write_const appear in profile
```

Top memory/layout signals were `write_const`, `conv2d_silu`, `slice_f32`, `concat`, `memcopy`, and `add_f32`. The immediate broad next step is to investigate safe persistent-constant handling, because `WriteConst` is both visible dynamically and accounts for about 12 MiB of static per-forward traffic in the existing memory stats.
