# Compiler passes guide

This directory owns individual graph analyses and rewrites. The authoritative
pipeline order is `CompilerPipeline::run` in `src/compiler/pipeline.rs`; do not
duplicate or infer it from this directory.

## Pass rules

- A structural rewrite must collect edits before applying them and call
  `graph.mark_mutated()` after graph structure changes.
- Analysis passes should take immutable graph references where possible;
  transformations must document their mutation assumptions.
- Keep a pass idempotent unless repeated application is explicitly part of the
  pipeline contract.
- Passes must not depend on backend dispatch details. Shared plan metadata needs
  a narrow neutral boundary rather than a new reverse dependency.

## Quantization

- Rust owns executable-IR quantization, activation/gradient rewrites, and Q/DQ
  pruning. Preserve U4/U8 integer-space execution.
- Keep calibration-data collection separate from graph rewriting.
- Do not add equivalent optimization logic to both Rust and Python without an
  operation-level ownership decision and parity tests.

## Testing

Place focused unit tests beside a pass when practical. For changed pass order or
semantics, add integration coverage through the graph executor, including
quantized pipeline coverage when Q/DQ or packed types are involved.