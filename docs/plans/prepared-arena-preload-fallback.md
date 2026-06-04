# Prepared arena preload fallback

Wave 7 added the first opt-in runtime use of `PreparedConstantArena`; the follow-up slice makes that path skip redundant exact Conv2d weight `WriteConst` instructions.

## Public API

`AotExecutor.forward_prepared_arena_fallback(inputs)` is available when fastnn is built with the `prepared-plan` feature.

## Contract

This method is behaviour-equivalent to `AotExecutor.forward(inputs)` and remains opt-in:

- validates the prepared plan against the live executable plan
- materializes graph inputs into the runtime arena like the normal executor
- preloads fp32 Conv2d static weights from `PreparedConstantArena` into their original runtime arena slots
- dispatches a temporary plan with only the matching exact Conv2d weight-slot `WriteConst` instructions removed
- keeps bias constants, non-Conv constants, dynamic weights, unsupported packed kinds, and non-exact slots on the original dispatch path
- delegates all remaining work to the existing backend dispatch
- reads outputs through the same decoder as `forward()` and `forward_prepared_fallback()`

Default `AotExecutor.forward()` is unchanged.

## Performance note

This is a small dispatch-pruning/correctness slice, not the main packed-kernel optimization. On the local YOLOv8n smoke, the path is exact (`max_abs=0.0`, `mean_abs=0.0`) but timing is neutral/noisy rather than a clear win. The expected upside is bounded by the removed `write_const` work; real speedups require prepared Conv kernels and/or packed layouts.

## YOLO expectation

For local `yolov8n.onnx`, `forward()`, `forward_prepared_fallback()`, and `forward_prepared_arena_fallback()` should produce exactly equal fastnn outputs (`max_abs=0.0`, `mean_abs=0.0`) for the same input.
