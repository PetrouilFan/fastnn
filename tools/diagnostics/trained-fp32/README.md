# Trained GPT-2 FP32 diagnostics (#76)

This directory owns the investigation that decides whether the trained GPT-2
`mean_abs` excess is a numerical defect or a tolerance-policy problem. The
report is [`docs/reports/fn3-trained-fp32.md`](../../../docs/reports/fn3-trained-fp32.md);
the raw reduced outputs are in
[`docs/reports/fn3-trained-fp32/`](../../../docs/reports/fn3-trained-fp32/).

Nothing here runs in CI and no model weights are stored in the repository. The
scripts are standard Python 3 plus NumPy/ONNX/ONNX Runtime; the Rust probes are
opt-in integration tests that skip with a printed reason unless
`FN3_TRAINED_ONNX` names the pinned graph.

## Files

| File | Purpose |
| --- | --- |
| `reproduce.sh` | Runs the whole investigation in order; every command is also listed in the report. |
| `extract_intermediates.py` | Rebuilds the ORT reference with per-layer values promoted to graph outputs. Proves the logits still equal the committed bytes. |
| `reference_float64.py` | Independent float64 evaluation of the same graph, stage by stage. |
| `compare_layers.py` | Reduces two probe directories to per-stage metrics (`layers.json`). |
| `attribute_accumulation.py` | Recomputes selected projections with the CPU kernel's documented sequential-f32 order and checks bit equality. |
| `summarize.py` | Builds `summary.json` and `layer-table.md` for the report. |

The fn3 side comes from two ignored tests:

| Test | Output |
| --- | --- |
| `tests/fn3_trained_parity.rs` | Full logits and a `parity-report.json` for CPU and, with the `vulkan` feature, the physical device. |
| `tests/fn3_trained_layers.rs` | Per-instruction value snapshots (the plan is executed one instruction at a time, through the same `CpuBackend`). |
| `tests/fn3_vulkan_trained_probe.rs` | Diagnostic only: wraps the Vulkan backend to name the instruction that fails on the resident path. |

## Quick start

```bash
export FN3_TRAINED_ONNX=/path/to/onnx/model.onnx
export FN3_PYTHON=/path/to/venv/bin/python   # numpy, onnx, onnxruntime
./tools/diagnostics/trained-fp32/reproduce.sh
```

The script checks the pinned 498,126,358-byte graph and its SHA-256 before any
run. It serialises model work under `/tmp/opencode/fn3-measure.lock`, prints
`/proc/loadavg`, and refuses to claim a pass when a model or device is absent.

## Reference conditions

* Graph: `onnx-community/gpt2-ONNX` @ `1975fb9b4a2a92bd99f457407337ee25ab6cfde0`,
  `onnx/model.onnx`, SHA-256
  `42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d`.
* ORT CPU 1.23.2 (it reproduces the committed ORT 1.32.0 payloads byte for
  byte), `ORT_DISABLE_ALL`, one intra-op and one inter-op thread, sequential.
* Prompt `[464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874,
  2402, 257]`, decode `[640, 11]`.

## Reading the result

`mean_abs` is the suite metric. `centered_rms` removes the common-mode shift
that a softmax cannot see, and `relative_rms` is `rms(delta)/rms(reference)`.
The finding is that the trained logits carry a large common-mode offset
(about -49 at the last prompt position, -108 at decode step 2), so a relative
error of about 1e-6 becomes a `mean_abs` of 7e-5..1e-4 even though the ranking
is unchanged (KL <= 1.2e-9, top-1/top-5 exact).
