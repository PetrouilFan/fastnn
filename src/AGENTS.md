# Rust core guide

`src/lib.rs` owns the Rust public module surface and root re-exports, and states
which modules are primary and which are secondary. fastnn has no external
users, so module paths may change when a cleaner boundary warrants it; update
internal callers rather than retaining obsolete facades.

## Dependency direction

`src/fn3/` is the primary tree and follows a strict DAG:

```text
dtype + device + ops → error → shape + storage → tensor → backend → model
```

The secondary 2.x tree follows the legacy direction:

```text
storage + dtypes → tensor + packed_tensor → nn → ir → compiler passes
→ backend → Python bindings and CLI adapters
```

Some 2.x dependencies still violate that target. Treat them as migration work,
not permission to add further cross-layer imports. In particular, compiler and
backend plan metadata need a narrow shared boundary, and Python bindings should
adapt stable Rust APIs rather than implement compiler semantics.

## Local rules

- Keep public types and serialized data definitions close to their owners.
- Keep `mod.rs` files as facades where practical; put cohesive implementation
  domains in named modules.
- Preserve graph mutation/cache invariants and backend instruction semantics.
- Do not add a training, autograd, optimizer or wgpu path; those are deleted.
- Unsafe code needs a local `SAFETY:` explanation.
- Follow the subsystem guide when changing `src/fn3/`, `ir`, `backend/cpu`, compiler
  passes or PyO3 bindings.

See `docs/roadmap/fn3-architecture.md` for the inference contract and
`docs/roadmap/codebase-reorganization.md` for the secondary tree.
