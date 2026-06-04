# Prepared arena preload fallback

Wave 7 adds the first opt-in runtime use of `PreparedConstantArena`.

## Public API

`AotExecutor.forward_prepared_arena_fallback(inputs)` is available when fastnn is built with the `prepared-plan` feature.

## Contract

This method is intentionally behaviour-identical to `AotExecutor.forward(inputs)`:

- validates the prepared plan against the live executable plan
- materializes graph inputs into the runtime arena like the normal executor
- preloads fp32 Conv2d static weights from `PreparedConstantArena` into their original runtime arena slots
- keeps all `WriteConst` instructions in the plan, so the normal dispatch path writes the same bytes again
- delegates to the existing backend dispatch
- reads outputs through the same decoder as `forward()` and `forward_prepared_fallback()`

Because `WriteConst` is not skipped yet, this is a plumbing/correctness slice, not a speed optimization. It proves that the prepared arena can source bytes into runtime execution without changing kernels or default execution.

## What remains

A later lane can use this foundation to skip selected `WriteConst` instructions after proving exactness and measuring speed on YOLO. That should remain opt-in until YOLO accuracy and profile gates pass.

## YOLO expectation

For local `yolov8n.onnx`, `forward()`, `forward_prepared_fallback()`, and `forward_prepared_arena_fallback()` should produce exactly equal fastnn outputs (`max_abs=0.0`, `mean_abs=0.0`) for the same input.
