# fastnn documentation

fastnn is an inference-only neural-network runtime with a Rust core and PyO3
bindings. The v3 rewrite replaces the 3.0 surface with one compiler, one
runtime and one artifact: ONNX → canonical IR → a target-aware plan → a
session-owned executor. The 2.x AOT tree and the 3.0 runtime are archived at
tag `archive/3.0`; see [the archive note](archive/README.md).

## Start here

- [One runtime: the v3 decision](v3/adr-0001-one-runtime.md) — what is being
  rebuilt and why
- [v3 status](v3/status.md) — which phases are done, the measured numbers, and
  what is still open
- [v3 scope and model contract](v3/scope.md) — what v3 runs, rejects and
  promises; the live matrix is [SUPPORTED.md](../SUPPORTED.md)
- [CLI quickstart](quickstart-fn3.md) — `fastnn-infer` on CPU and Vulkan
- [Development](internals/development.md) — build, test and measure this tree
- [Release process](internals/release-process.md) — versioning and release checks

## Archive

Everything under [`docs/archive/`](archive/README.md) describes the surface that
was removed. It is kept for the measurements and reasoning it records; its
commands and API names no longer exist.
