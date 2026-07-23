# PyO3 bridge guide

This directory exposes Rust functionality as `fastnn._core`. fastnn has no
external users, so module names, registrations, and Python behavior may change
as part of a deliberate binding redesign.

## Boundaries

- `mod.rs` should own module initialization and stable registration order, not
  the implementation of every binding.
- Binding modules adapt Rust APIs; they do not own graph compilation or create
  competing compiler semantics.
- Replace `include!` through normal Rust module boundaries and explicit
  per-domain registration functions. Rename exported Python names if that makes
  the public boundary clearer.

## Safety

- Python exceptions must not cross FFI as Rust panics.
- DLPack capsule ownership and destructors require explicit lifetime review.
- Any CPU-buffer exposure must preserve the underlying arena aliasing and
  lifetime guarantees.

Validate binding moves with extension build/install, modular API/import tests,
and targeted tensor, nn, optimizer, executor, and serialization tests.