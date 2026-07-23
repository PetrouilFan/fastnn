# Python package guide

`fastnn/__init__.py` is the Python facade over `fastnn._core` and Python-level
utilities. Its imports, lazy names, `__all__`, callable `fastnn.tensor`, and
error behavior may be simplified because fastnn has no external users.

## Ownership

- Python facade modules provide ergonomic API composition and Python-only
  behavior such as data loading, callbacks, model helpers, and user-facing I/O.
- Compiled tensor, graph execution, and executable compiler semantics belong to
  Rust unless the ownership matrix in the reorganization roadmap says
  otherwise.
- Keep package imports acyclic and avoid eager imports that change initialization
  or optional-dependency behavior.

## Compatibility

- `pyproject.toml` owns distribution metadata and the `fastnn._core` extension
  module configuration.
- Remove facade aliases and lazy `__getattr__` names that do not justify their
  maintenance cost.
- Do not make Python graph optimization or shape inference authoritative without
  parity evidence against the Rust compiled-IR path.

Use targeted pytest modules for the changed API; package/facade changes require
the modular API and import smoke tests in the roadmap.