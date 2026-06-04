# YOLO Prepared-Plan Stats Script

Standalone tooling to inspect the prepared-plan composition of the YOLOv8n
ONNX model. Used as a regression signal: if a future change accidentally
drops Conv2d / MatMul specialization from the prepared plan, the counts
this script prints diverge from the baseline below.

## What it does

`scripts/yolo_prepared_stats.py`:

1. Loads an ONNX model from disk (`--onnx PATH`, default `yolov8n.onnx`
   in the current working directory).
2. Compiles it through `fastnn.AotExecutor`, reusing
   `_make_fastnn_executor` from `scripts/yolo_compare_fastnn_pytorch.py`
   so the ONNX -> fastnn lowering is identical to the comparison harness.
3. Calls `executor.prepared_stats()` and prints a small human-readable
   summary with counts for `total`, `conv2d`, `matmul`, `generic`,
   `static_weight_bindings`, `constant_arena_entries`, and
   `constant_arena_bytes`.
4. Optionally writes the raw dict to `--json OUT.json` (atomic
   `tempfile` + `os.replace`).

It uses only `argparse`, `json`, `os`, `sys`, `tempfile`, `pathlib`
from the stdlib, plus `fastnn`, `onnx`, and `numpy` (already required
by the project).

## Gating rule

`executor.prepared_stats()` only exists when the Rust extension was
built with the `prepared-plan` feature. If the method is missing the
script exits non-zero with a clear error message — it does NOT
silently skip. Build the extension with:

```bash
maturin develop --release --features prepared-plan
```

Quick precheck (should print `True`):

```bash
python -c "import fastnn; print(hasattr(fastnn.AotExecutor, 'prepared_stats'))"
```

## Commands

```bash
# Default: looks for yolov8n.onnx in CWD
python scripts/yolo_prepared_stats.py --onnx yolov8n.onnx

# Persist the raw dict for diffing against a known baseline
python scripts/yolo_prepared_stats.py --onnx yolov8n.onnx --json /tmp/yolo_prepared_stats.json
```

## Expected output (local `yolov8n.onnx`, opset 12, imgsz 320)

```
YOLO prepared stats
  model: yolov8n.onnx
  total instructions: 259
  conv2d: 64
  matmul: 0
  generic (other): 195
  static weight bindings: 64
  constant arena entries: 127
  constant arena bytes: 12607616
```

`conv2d == 64` matches the baseline doc (`docs/plans/prepared-plan-baseline.md`,
"~64 Conv2d"). YOLOv8n exported with `ultralytics opset=12` lowers the
detection head's matrix multiplies into 1x1 Conv2d nodes, so the prepared
plan reports `matmul == 0`. A future regression that demoted Conv2d nodes
to the `Generic` instruction bucket would shrink the `conv2d` count below
64 (and inflate `generic` accordingly), which is easy to spot by diffing
the JSON output.

`static_weight_bindings == 64` confirms every Conv2d picked up its
specialized static-weight binding (one per Conv2d kernel). The two
`constant_arena_*` counters track the size of the immutable activation
arena materialized for the prepared plan and are a useful regression
signal for unintended growth in constant footprint.

## Exit codes

| code | meaning                                       |
|-----:|-----------------------------------------------|
|    0 | success                                       |
|    2 | ONNX file not found                           |
|    3 | failed to import the executor helper          |
|    4 | `AotExecutor` build failed                    |
|    5 | `prepared_stats` missing (feature not built)  |
|    6 | `prepared_stats()` raised                     |
|    7 | `prepared_stats()` returned a non-dict        |
|    8 | JSON output write failed                      |

## Constraints

- No Rust changes; pure Python tooling.
- No new package dependencies.
- Reuses the existing ONNX-to-fastnn lowering from
  `scripts/yolo_compare_fastnn_pytorch.py` to avoid divergence.
