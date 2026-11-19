# PyO3 bridge guide

This directory exposes Rust functionality as `fastnn._core`. fastnn has no
external users, so module names, registrations, and Python behavior may change
as part of a deliberate binding redesign.

## Boundaries

- `mod.rs` owns module initialization and stable registration order, not the
  implementation of every binding.
- Binding modules adapt Rust APIs; they do not own graph compilation or create
  competing compiler semantics.
- The inference surface is declared by path from this directory:
  `#[path = "../fn3/python/mod.rs"] pub mod inference;`. `src/fn3/` itself stays free
  of PyO3.
- The 2.x binding surface is forward-only. There is no autograd, optimizer,
  loss, trainer or gradient binding to extend.

## Safety

- Python exceptions must not cross FFI as Rust panics.
- DLPack capsule ownership and destructors require explicit lifetime review.
- Any CPU-buffer exposure must preserve the underlying arena aliasing and
  lifetime guarantees.

Validate binding moves with extension build/install, modular API/import tests,
and targeted tensor, nn, executor and serialization tests.
