# Packed static Conv weights metadata

Roadmap task: C2 — Packed static Conv weights for OpenBLAS/custom kernels.

## Scope

This slice records the existing prepared-plan packed-weight metadata surface for YOLO Conv weights. It is intentionally metadata/storage only:

- `PackedWeightStore::TransposedFp32 { data, m, k }` stores a row-major `[K, M]` transpose for selected static Conv weights.
- `PreparedConv2d::transposed_weight` explicitly binds eligible prepared Conv instructions to their metadata-only transposed arena entry.
- `PreparedConstantArena::get_transposed_fp32()` is the only accessor for the alternate `[K, M]` layout; ordinary fallback `get()` still returns `None` for these entries.
- Runtime dispatch is unchanged.
- Default `forward()` is unchanged.
- Prepared fallback paths do not consume metadata-only transposed entries as runtime `f32` constants.

The current selector materializes transposed fp32 Conv weights for the repeated YOLO target family:

```text
Conv2d + SiLU
static fp32 weights
groups=1, dilation=1
3x3 kernel
M=64, K=576
```

This keeps the storage surface narrow and avoids a global layout switch. The measured backend probe showed a large candidate backend gap, but previous isolated layout work was shape-dependent, so C2 stays metadata-only until an opt-in runtime path clears full-YOLO gates.

## Verification command

```bash
PYENV_VERSION=system .venv/bin/python scripts/yolo_prepared_stats.py \
  --onnx yolov8n.onnx \
  --json /tmp/yolo_prepared_stats_c2.json
```

Representative local result:

```text
total instructions: 259
conv2d: 64
matmul: 0
generic: 195
static weight bindings: 64
constant arena entries: 140
constant arena bytes: 14524544
packed fp32 conv candidates: 64
packed fp32 conv candidate flops: 2185996800
transposed fp32 conv entries: 13
transposed fp32 conv bytes: 1916928
transposed fp32 conv bindings: 13
transposed fp32 conv binding flops: 538214400
```

## Acceptance status

C2 acceptance is satisfied:

- Packed/static Conv metadata exists in `src/backend/prepared.rs`.
- Prepared stats expose candidate counts and transposed-entry byte counts through `AotExecutor.prepared_stats()`.
- Prepared stats expose transposed-entry binding counts and bound FLOP coverage through `transposed_fp32_conv_bindings` / `transposed_fp32_conv_binding_flops` so a future opt-in runtime path can consume a direct handle and quantify its full-YOLO scope instead of name-scanning the arena.
- The direct handle is paired with a layout-specific accessor, preventing future runtime paths from overloading original-layout `packed_weight`/`get()` semantics.
- Runtime behavior is unchanged; metadata-only transposed entries return `None` from `PackedWeightStore::as_f32_slice()` and are not consumed by fallback constant lookup.

Next runtime-changing work must be opt-in and gated by full YOLO speed + accuracy, not microbench-only wins.
