# Rust core guide

`src/lib.rs` owns the Rust public surface: `inference` (the whole runtime) and,
behind the `python` feature, `python` (the PyO3 registration). There is no
second inference tree; the 2.x AOT modules were archived (see docs/archive/README.md).

## Dependency direction

`src/fn3/`, mounted as `fastnn::inference`, follows a strict DAG:

```text
dtype + device + ops → error → shape + storage → tensor → backend → model
```

`import` depends on `dtype`, `shape`, `error` and `ops`; `cli` and `python` are
adapters over `model` and must not implement import, planning or numerics that
belong below them. Never add a dependency from a lower layer back to `model`,
`cli` or `python`.

## Local rules

- Keep public types and serialized data definitions close to their owners.
- Keep `mod.rs` files as facades; put cohesive domains in named modules.
- A new operation needs a traced path through `ops`, `shape`, `backend`,
  `model` and `import`, with a test at each boundary it crosses.
- `Support` is a promise: a backend either executes an operation within the
  declared contract or reports `Unsupported`; it never guesses or silently
  substitutes another dtype or layout.
- Do not add a training, autograd, optimizer or wgpu path; there is no such
  code and a feature that needs one is a new product decision.
- Rust unsafe code needs a local `SAFETY:` explanation covering aliasing,
  bounds, alignment and lifetime assumptions.
- Untrusted input (artifacts, ONNX files, quantization metadata) is
  bounds-checked and budgeted; a malformed file must produce a typed error, not
  a panic or an allocation the file asked for.

See `docs/v3/adr-0001-one-runtime.md` for the architecture decision and
`src/fn3/AGENTS.md` for the subsystem guide.
