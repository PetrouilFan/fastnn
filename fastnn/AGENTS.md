# Python package guide

`fastnn/inference/` is the shipped inference API and a thin wrapper over
`fastnn._core.inference`. `fastnn/__init__.py` re-exports it and nothing else.
fastnn has no external users, so imports, lazy names, `__all__` and error
behavior may be simplified whenever a cleaner surface warrants it.

## Ownership

- `fastnn/inference/` owns path coercion, docstrings and `__all__` for the
  inference API; the classes themselves are the extension's own objects.
- Python code does not own graph compilation, shape inference, planning or
  numerical behavior: those live in Rust, and the extension is the only entry
  point for them.
- There is no training, optimizer, scheduler, loss, data-loader or callback
  module. Do not add one without a product decision.
- Keep package imports acyclic and free of eager optional dependencies.

## Compatibility

- `pyproject.toml` owns distribution metadata and the `fastnn._core` extension
  module configuration.
- Do not reintroduce a 2.x facade module, a `.fnn` loader or an
  `AotExecutor`-style class. The archived surface lives at tag `archive/3.0`.
- Remove aliases and lazy `__getattr__` names that do not justify their
  maintenance cost.

Use targeted pytest modules for the changed API; package/facade changes require
the import smoke test (`tests/test_fn3_python.py`) and the CLI tests.
