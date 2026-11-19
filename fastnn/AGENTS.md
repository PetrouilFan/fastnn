# Python package guide

`fastnn/inference/` is the shipped inference API and a thin wrapper over
`fastnn._core.inference`. `fastnn/__init__.py` also re-exports the surviving 2.x
forward facade. fastnn has no external users, so imports, lazy names, `__all__`
and error behavior may be simplified.

## Ownership

- `fastnn/inference/` owns path coercion, docstrings and `__all__` for the inference
  API; the classes themselves are the extension's own objects.
- Python facade modules provide ergonomic composition and Python-only behavior
  such as model helpers and user-facing I/O.
- Compiled tensor, graph execution and executable compiler semantics belong to
  Rust.
- There is no training, optimizer, scheduler, loss, data-loader or callback
  module. Do not add one without a product decision.
- Keep package imports acyclic and avoid eager imports that change
  initialization or optional-dependency behavior.

## Compatibility

- `pyproject.toml` owns distribution metadata and the `fastnn._core` extension
  module configuration.
- Remove facade aliases and lazy `__getattr__` names that do not justify their
  maintenance cost.
- Do not make Python graph optimization or shape inference authoritative without
  parity evidence against the Rust compiled-IR path.

Use targeted pytest modules for the changed API; package/facade changes require
the modular API and import smoke tests.
