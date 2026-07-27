# Rust core guide

`src/lib.rs` owns the Rust public module surface and root re-exports. fastnn has
no external users, so module paths may change when a cleaner boundary warrants
it; update internal callers rather than retaining obsolete facades.

## Dependency direction

The target direction is:

```text
storage + dtypes → tensor + packed_tensor → nn/optim/autograd → ir
→ compiler passes → backend → Python bindings and CLI adapters
```

Some current dependencies violate this target. Treat them as migration work,
not permission to add further cross-layer imports. In particular, compiler and
backend plan metadata need a narrow shared boundary, and Python bindings should
adapt stable Rust APIs rather than implement compiler semantics.

## Local rules

- Keep public types and serialized data definitions close to their owners.
- Keep `mod.rs` files as facades where practical; put cohesive implementation
  domains in named modules.
- Preserve graph mutation/cache invariants and backend instruction semantics.
- Unsafe code needs a local `SAFETY:` explanation.
- Follow the subsystem guide when changing `ir`, `backend/cpu`, compiler
  passes, or PyO3 bindings.

See `docs/roadmap/codebase-reorganization.md` for compatibility requirements
and validation gates.