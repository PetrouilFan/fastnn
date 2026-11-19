# fastnn Documentation

fastnn 3.0 is an inference-only neural-network runtime: ONNX import → a
versioned prepared-model artifact → a planned session running on the CPU
reference backend or a physical Vulkan device. It has a Rust API, the
`fastnn-infer` CLI and the `fastnn.inference` Python package.

The crate also keeps the 2.x AOT tree as a secondary inference surface (graph
IR, compiler passes, CPU executor, eager forward tensor/nn API, ONNX conversion
and `.fnn` I/O). Training, autograd, optimizers and the wgpu backend were
deleted in FN3-19 (#65); the pinned 2.x revision `a2c0759` is the archive.

---

## FastNN 3.0 inference

- [Getting Started](guides/getting-started.md) — build, prepare a model, run a session
- [CLI Quickstart](quickstart-fn3.md) — `fastnn-infer` on CPU and Vulkan
- [Python API](reference/python-api.md) — the `fastnn.inference` session API
- [Inference Architecture](roadmap/fn3-architecture.md) — core contracts and runtime design
- [Quantized Weights Format](quantization/fn3-quant-format-v1.md) — 3.0 sidecar format
- [Migration from 2.x](guides/migrate-from-2x.md) — what replaces `fastnn-runtime` and when to keep using 2.x

## Retained 2.x reference

- [Disposition and Callers](internals/legacy-2x.md) — what remains, who uses it, and what was retired
- [Tensors](reference/tensors.md) — 2.x tensor creation, operations, devices and NumPy interop
- [Neural Network Modules](reference/nn-modules.md) — 2.x forward layer reference (Linear, Conv, Normalization, etc.)
- [I/O & Serialization](reference/io.md) — `fastnn.io`, `fastnn-convert`, and `.fnn` models
- [API Reference](reference/api-reference.md) — secondary forward Python API
- [Models](models/models.md) — legacy MLP, Transformer, and YOLO examples
- [ONNX Support](models/onnx.md) — legacy ONNX conversion and AOT execution
- [AOT Architecture](internals/architecture.md) — secondary IR, compiler and CPU executor

## Internals and 2.x engineering notes

Most of these describe the retained 2.x tree or the build/release process; they
are not 3.0 runtime documentation.

- [Development](internals/development.md) — codebase walkthrough for both trees and test layout
- [Release Process](internals/release-process.md) — release workflow and checklist
- [Performance Roadmap](internals/performance-roadmap.md) — 2.x CPU/Vulkan performance work
- [ARM NEON Backend](internals/arm-neon.md) — 2.x CPU SIMD kernel documentation

## Roadmap and evidence

- [Roadmap Overview](roadmap/index.md) — historical 2.x planning record
- [v2.3 Roadmap](roadmap/v2.3-roadmap.md) — historical planning document
- [2.x Baseline](reports/fn3-01/baseline-2.6.0.md) — preserved comparison evidence. Its
  companion `fn3-trained-fp32` report links three logs that were never committed, so
  `check-doc-truthfulness.py` still names them.

---

Start with [Getting Started](guides/getting-started.md).
