# Graph memory stats

General-purpose memory/layout efficiency work should not be tied only to YOLO or
to one Conv backend. This task adds model-agnostic compiled-plan memory
introspection so broad graph compiler/runtime inefficiencies are visible before
choosing an optimisation lane.

## Tooling

`AotExecutor.memory_stats()` now returns static compiler/runtime metrics for any
compiled ONNX graph:

- arena size and memory-plan total size
- logical slot bytes vs physical bytes by arena offset
- slot-reuse savings, alias group count, and largest alias groups
- instruction mix
- exact per-instruction traffic rows ranked by static bytes (`top_instructions_by_static_bytes`), including graph node id/name/opcode plus input/output shapes for kernel-call rows so layout/copy hotspots can be inspected without a separate IR dump
- largest WriteConst rows with destination offsets/sizes (`top_write_consts_by_size`);
  when built with `prepared-plan`, rows that feed prepared Conv/MatMul static
  weights or biases also include the prepared consumer instruction, input index,
  static role, and constant-arena name/id
- estimated static traffic bytes from kernel inputs/outputs, `MemCopy`, `Fill`,
  and `WriteConst`
- top kernels by instruction count

`scripts/graph_memory_stats.py` wraps the API for any ONNX file and can emit raw
JSON:

```bash
PYENV_VERSION=system .venv/bin/python scripts/graph_memory_stats.py \
  --onnx yolov8n.onnx \
  --json /tmp/graph_memory_stats_yolo.json
```

The script reuses the existing ONNX executor construction helper from
`scripts/yolo_compare_fastnn_pytorch.py`, but the reported metrics are not
YOLO-specific.

## Representative local result

After rebuilding the extension with `--features 'prepared-plan openblas'`, the
YOLOv8n 320x320 graph reported:

```text
instructions: 259
arena size: 42.97 MiB
logical slot bytes: 44.73 MiB
physical slot bytes by offset: 42.96 MiB
slot reuse saved: 1.76 MiB (3.9% of logical)
alias reuse: 11 groups, 23 aliased nodes
estimated static traffic: 92.81 MiB (2.16x arena)
  kernel reads: 49.26 MiB
  kernel writes: 29.79 MiB
  memcpy: 1.70 MiB
  write_const: 12.06 MiB
  fill: 0.00 B
instruction mix: call_kernel=120, memcpy=8, write_const=131, fill=0

top instruction-level static traffic rows after the instruction diagnostic
addition:

- `#131 conv2d_silu` (node 157): 2.74 MiB
- `#132 conv2d_silu` (node 161): 2.36 MiB
- `#139 concat` (node 180 `/model.2/Concat`): 2.34 MiB,
  inputs `[1,16,80,80]`, `[1,16,80,80]`, `[1,16,80,80]`, output `[1,48,80,80]`.
- `#190 concat` (node 311 `/model.14/Concat`): 2.34 MiB,
  inputs `[1,128,40,40]`, `[1,64,40,40]`, output `[1,192,40,40]`.
- `#140 conv2d_silu` (node 181): 1.96 MiB

largest WriteConst rows after the prepared-static cross-reference addition:

- `#118` 1.12 MiB at arena offset `10203968`, prepared Conv weight for
  consumer instruction `#165` (`conv_118_i1`).
- `#83` 720 KiB at `7853440`, prepared Conv weight for consumer instruction
  `#233` (`conv_83_i1`).
- several 576 KiB prepared Conv-weight rows (`#35`, `#49`, `#51`, `#65`,
  `#124`, `#126`).
```

## Interpretation

The current memory planner already performs some alias reuse, but the full graph
shows low static slot reuse by this metric: only ~3.9% of logical slot bytes are
saved by shared offsets. That does not mean the planner is wrong; many tensors
are genuinely live for long ranges, and constants currently occupy mutable arena
slots. It does mean broad compiler/runtime efficiency work should consider:

1. Persistent constants separated from transient arena slots.
2. More aggressive view/layout planning for physical-copy ops such as slice,
   concat, transpose, and reshape-like paths.
3. Reducing per-forward static traffic, especially `WriteConst` bytes, without
   unsafe reuse of mutable slots.
4. Adding dynamic/profiled bytes-per-ms overlays on top of this static view so
   optimisation targets can be ranked across models, not only YOLO.

This tool is deliberately measurement-only; it does not change default runtime
semantics.
