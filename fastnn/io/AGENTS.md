# Python I/O guide

This directory provides Python-facing model serialization, import/export,
format adapters, calibration utilities, validation, and the `AotExecutor`
facade.

## Ownership rules

- Treat `.fnn` reading/writing as a format ownership boundary. Redesign magic,
  versioning, tensor metadata, and readers if consolidation requires it; test
  only the resulting supported format.
- Python ONNX and graph-dictionary code is an import-time adapter, not
  automatically the authority for executable compiler semantics.
- Rust owns compiled-IR shape inference, graph rewrites, quantization, Q/DQ
  pruning, memory planning, lowering, and dispatch unless an explicit parity
  decision says otherwise.
- Python may retain compatibility transforms until shared operation-level tests
  prove delegation/removal safe. Do not add duplicate optimizer rules casually.

## Boundaries

- Keep serialization primitives, importers, runtime wrappers, calibration,
  validation, and compatibility graph transforms in distinct modules.
- `dag_model.py` remains a thin adapter over `_core.AotExecutor`; it must not
  absorb backend semantics.

For ONNX or graph optimization changes, add import/execute parity coverage and
run the relevant Python graph optimizer, shape inference, ONNX, and quantized
pipeline tests.