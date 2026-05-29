# ONNX Training Export Contract

## Summary

FastNN **does not support** exporting training graphs to ONNX. The ONNX format
is designed for inference, and fastnn's training-specific opcodes (optimizer
updates, gradient scaling) have no standard ONNX representation.

Exporting a training graph to ONNX would silently drop these nodes, producing
a graph that appears valid but produces wrong results. The export system now
prevents this by raising an explicit error.

## What Is Exported

Only **inference graphs** — graphs containing forward-pass ops — can be
exported to ONNX JSON format. This includes:

- Matrix multiplication (`MatMul`)
- Convolutions (`Conv2d`)
- Activations (`Relu`, `Gelu`, `Silu`, `Sigmoid`, `Tanh`, etc.)
- Normalization (`BatchNorm`, `LayerNorm`, `RMSNorm`)
- Shape operations (`Reshape`, `Transpose`, `Concat`, `Slice`, etc.)
- Quantized patterns (`QLinearMatMul`, `QLinearConv`)

## What Is Explicitly Unsupported

The following opcodes have **no standard ONNX representation** and will cause
`export_to_onnx_json` to return an error if present in the graph:

| Opcode | Description |
|--------|-------------|
| `SgdUpdate` | SGD weight update step |
| `AdamUpdate` | Adam optimizer step |
| `AdamWUpdate` | AdamW optimizer step |
| `MuonUpdate` | Muon optimizer step |
| `LionUpdate` | Lion optimizer step |
| `RmspropUpdate` | RMSprop optimizer step |
| `GradientScale` | Gradient scaling for mixed-precision training |

These are training-only operations executed by fastnn's compiled training
pipeline. They modify model weights in-place and maintain optimizer state
(momentum buffers, etc.) — none of which ONNX can represent.

## Error Behavior

By default, `export_to_onnx_json` and `export_to_onnx_file` will return an
error like:

```
ONNX export failed: graph contains 1 training-only opcode(s) that have no
standard ONNX representation and would be silently dropped, producing a wrong
graph: node 5 (SgdUpdate).

Training opcodes in fastnn (SgdUpdate, AdamUpdate, AdamWUpdate, MuonUpdate,
LionUpdate, RmspropUpdate, GradientScale) are executed by the fastnn compiled
training pipeline and cannot be faithfully represented in the ONNX inference
format.

To export this graph for inference only, remove training nodes (optimizer
steps, gradient scaling) before export, or set
`ExportConfig { fail_on_training_ops: false }` to explicitly acknowledge that
training nodes will be dropped.
```

## Export Metadata

Every exported JSON includes a `metadata` block:

```json
{
  "metadata": {
    "export_mode": "inference",
    "training_ops_dropped": false,
    "training_ops_count": 0
  }
}
```

When `fail_on_training_ops` is set to `false` and training ops are present,
the metadata will report `training_ops_dropped: true` with the count of
dropped ops.

## Opt-In: Export with Training Ops Dropped

If you have verified that the training opcodes in your graph are intentionally
irrelevant (e.g., they exist in the IR but you only care about the inference
path), you can explicitly opt in:

```rust
use fastnn::onnx::export::{export_to_onnx_json_with_config, ExportConfig};

let config = ExportConfig {
    fail_on_training_ops: false,
};
let json = export_to_onnx_json_with_config(&graph, &config)?;
```

**Warning**: This will produce a graph that silently omits training nodes.
Only use this if you have verified the graph's training nodes are irrelevant
to your use case.

## Training in FastNN

For training, fastnn uses its own **compiled training pipeline** (v2.2), not
ONNX. The training pipeline:

1. Takes a `ComputeGraph` and compiles it with backward passes and optimizer
   steps into an `ExecutablePlan`
2. Executes training steps with a single dispatch call
3. Supports SGD, Adam, AdamW, Muon, Lion, and RMSprop optimizers

See `docs/training.md` for the full training API.

## Detection API

You can programmatically check whether a graph contains training-only opcodes:

```rust
use fastnn::onnx::export::detect_training_ops;

let training_ops = detect_training_ops(&graph);
if !training_ops.is_empty() {
    // Graph contains training nodes that cannot be exported to ONNX
    for (node_id, opcode_name) in &training_ops {
        eprintln!("node {}: {}", node_id, opcode_name);
    }
}
```

## Test Coverage

The safety contract is verified by `tests/onnx_export_training.rs`:

- Inference graphs export successfully with correct metadata
- Training graphs (SGD, Adam, GradientScale) produce explicit errors
- `detect_training_ops` correctly identifies training opcodes
- Opt-in `fail_on_training_ops: false` drops training ops with metadata
- Error messages are actionable (suggests removing training nodes)
- Quantized export (`QLinearMatMul`, `QLinearConv`) is not affected
