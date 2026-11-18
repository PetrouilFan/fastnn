# fastnn Documentation

fastnn 3.0 is an inference-only neural-network runtime: ONNX import → a
versioned prepared-model artifact → a planned session running on the CPU
reference backend or a physical Vulkan device. It has a Rust API, the
`fn3-infer` CLI and the `fastnn.fn3` Python package.

The crate also keeps the 2.x AOT tree as a secondary inference surface (graph
IR, compiler passes, CPU executor, eager forward tensor/nn API, ONNX conversion
and `.fnn` I/O). Training, autograd, optimizers and the wgpu backend were
deleted in FN3-19 (#65); the pinned 2.x revision `a2c0759` is the archive.

---

## Guides

- [Getting Started](guides/getting-started.md) — build, prepare a model, run a session

## Reference

- [Tensors](reference/tensors.md) — inference tensor creation, operations, devices and NumPy interop
- [Neural Network Modules](reference/nn-modules.md) — forward layer reference (Linear, Conv, Normalization, etc.)
- [IO & Serialization](reference/io.md) — save/load models, ONNX import, `.fnn` and `.fn3prep` artifacts
- [API Reference](reference/api-reference.md) — forward Rust and Python API
- [Python API](reference/python-api.md) — the `fastnn.fn3` inference surface

## Internals

- [Architecture](internals/architecture.md) — the 3.0 inference core and the secondary AOT pipeline
- [Development](internals/development.md) — codebase walkthrough and test layout
- [Release Process](internals/release-process.md) — release workflow and checklist
- [Performance Roadmap](internals/performance-roadmap.md) — CPU/Vulkan performance work
- [ARM NEON Backend](internals/arm-neon.md) — ARM NEON SIMD kernel documentation

## Models

- [Models](models/models.md) — supported architectures (MLP, Transformer, YOLO)
- [ONNX Support](models/onnx.md) — ONNX import, operator coverage, troubleshooting

## Roadmap

- [Roadmap Overview](roadmap/index.md) — current and future development priorities
- [v2.3 Roadmap](roadmap/v2.3-roadmap.md) — historical planning document

---

Start with [Getting Started](guides/getting-started.md).
