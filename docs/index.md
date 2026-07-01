# fastnn Documentation

Welcome to the fastnn documentation. fastnn is an ahead-of-time compiled neural
network runtime with a Rust core and Python bindings.

---

## Guides

Start here for tutorials and walkthroughs.

- [Getting Started](guides/getting-started.md) — Install fastnn and run your first program
- [Training](guides/training/index.md) — Training loops, data loading, callbacks, compiled training

## Reference

API documentation and module catalogs.

- [Tensors](reference/tensors.md) — Tensor creation, operations, autograd, broadcasting
- [Neural Network Modules](reference/nn-modules.md) — Layer and module reference (Linear, Conv, Normalization, etc.)
- [Optimizers](reference/optimizers.md) — Optimizer and scheduler APIs (SGD, Adam, AdamW, Muon, Lion, RMSprop)
- [IO & Serialization](reference/io.md) — Save/load models, ONNX import, serialization format
- [API Reference](reference/api-reference.md) — Complete Rust and Python API documentation
- [Python API](reference/python-api.md) — Python-facing APIs for compiled execution and runtime features

## Internals

Architecture and contributor documentation.

- [Architecture](internals/architecture.md) — AOT compiler pipeline: IR, compiler passes, backends
- [Development](internals/development.md) — Codebase walkthrough, module guide, adding ops and types
- [Release Process](internals/release-process.md) — Release workflow and checklist
- [Performance Roadmap](internals/performance-roadmap.md) — GPU, fusion, optimizer, and precision roadmap
- [ARM NEON Backend](internals/arm-neon.md) — ARM NEON SIMD kernel documentation

## Models

Pre-built model architectures and ONNX support.

- [Models](models/models.md) — Supported architectures (MLP, Transformer, YOLO)
- [ONNX Support](models/onnx.md) — ONNX import, operator coverage, troubleshooting
- [ONNX Training Export](models/onnx-training-export.md) — Training export contract and workflow

## Roadmap

- [Roadmap Overview](roadmap/index.md) — Current and future development priorities
- [v2.3 Roadmap](roadmap/v2.3-roadmap.md) — Historical planning document

---

*Each document includes cross-references to related guides and reference material.
Start with [Getting Started](guides/getting-started.md) if you are new to fastnn.*
