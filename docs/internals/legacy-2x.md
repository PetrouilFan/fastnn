# Retained 2.x surfaces and callers

This inventory records the disposition for #75, audited against `2022de4`.
The primary decoder surface is [FastNN 3.0 inference](../guides/getting-started.md).
The 2.x code below is inference-only and does not sit on the 3.0 execution path.

| Surface | Caller or evidence | Disposition |
|---|---|---|
| Standalone `fastnn-runtime` Cargo target | [FN3-01 baseline](../reports/fn3-01/baseline-2.6.0.md), its raw command matrix and the changelog. No active tracked invocation in tests, CI, release, examples or scripts. | **Retired.** The binary, its direct `clap` dependency and now-unused transitive crates are gone. `cli` still gates `fastnn-infer`, which has a handwritten parser. `clap` remains in the lockfile via the Criterion dev dependency. |
| Python ONNX/`.fnn` conversion and model I/O | The `fastnn-convert` console script is still registered in [`pyproject.toml`](../../pyproject.toml) and calls `fastnn.io.convert:main`. [`onnx_pipeline.py`](../../examples/onnx_pipeline.py) and [`test_onnx.py`](../../tests/test_onnx.py) use legacy conversion; [`test_io.py`](../../tests/test_io.py) covers model serialization. The Rust [`OnnxConverter`](../../src/onnx/converter.rs) consumes graph data from [the PyO3 binding](../../src/python/nn.rs), not ONNX protobuf. | **Retain** the script and both language paths for legacy model conversion. Do not reroute them to the strict decoder importer. |
| AOT compiler, prepared CPU executor and quantization | [`DAGModel`](../../fastnn/io/dag_model.py) wraps `_core.AotExecutor`; [`test_aot_executor_prepared_execute.py`](../../tests/test_aot_executor_prepared_execute.py) exercises its prepared/persistent paths. [`quantized_pipeline.rs`](../../tests/quantized_pipeline.rs) exercises native, packed and grouped W4A8 graph execution. | **Retain** `GraphBuilder`, `GraphExecutor`, the prepared runtime and quantized CPU kernels. These operate on 2.x IR/plans, not `.fnn3`. |
| Eager forward Rust and Python API | [`fastnn/__init__.py`](../../fastnn/__init__.py) imports the eager facade when the package initializes (including when importing `fastnn.inference`); [the PyO3 registration](../../src/python/mod.rs) supplies its classes and functions. [`test_modular_api.py`](../../tests/test_modular_api.py), [`test_tensor.py`](../../tests/test_tensor.py) and [`test_nn.py`](../../tests/test_nn.py) exercise the facade. | **Retain.** Removing it would require untangling package initialization and migrating the model/test callers. |
| Legacy Rust plan `Runtime` and serialization | After the CLI, [`backend::runtime::Runtime`](../../src/backend/runtime.rs) has only its own unit-test callers in-tree (input, ownership and resource-limit validation). [`release_error_paths.rs`](../../tests/release_error_paths.rs) and the [executable deserialization fuzz target](../../fuzz/fuzz_targets/executable_deserialization.rs) separately exercise the shared plan decoder. `bincode` is also used by the legacy graph serializer and graph-builder cache keys. | **Retain for now** as a secondary Rust API and tested reference. Not a replacement executable or a compatibility promise for 3.0; any further removal should decide where its validation coverage belongs. |
| CPU comparison and baseline tools | [`fn3-bench`'s baseline adapter](../../tools/bench/adapters/baseline.rs) calls legacy CPU kernels, telemetry and `GraphExecutor`; [`cpu_baselines`](../../benches/cpu_baselines.rs) and [`cpu_reference_oracle.rs`](../../tests/cpu_reference_oracle.rs) provide measurements and numerical checks. [`bench_tiny_gpt2.py`](../../tools/baseline/bench_tiny_gpt2.py) is the 2.x measurement recipe. | **Retain** measurement and reference callers. Do not remove the shared CPU dependencies just because the old binary is gone. |

## Reproduce or migrate

- To reproduce the historical standalone CLI, use the baseline revision
  `a2c0759` referenced by the [FN3-01 report](../reports/fn3-01/baseline-2.6.0.md).
  To inspect the implementation immediately before removal, run
  `git show 2022de4:src/bin/runtime.rs`. Historical commands, logs and the
  ReLU graph fixture are preserved as evidence, not current instructions.
- To deploy a supported decoder, follow the [migration guide](../guides/migrate-from-2x.md).
  Do not rename a `.fnn` or `.fnnc` file to `.fnn3`.
- Before removing more 2.x code, migrate or replace its callers and replace any
  lost comparison or correctness coverage. The old roadmap and reports are
  context, not proof that everything described there still ships.

## Verification notes

The retirement was checked with `cargo fmt --check`, both non-Python build
profiles, `--lib`, `fn3_contracts`, `fn3_cli`, `quantized_pipeline`,
`cpu_arena_telemetry`, `onnx_export_quantized`, `release_error_paths`, and the
Python suite marked `not slow` including a `fastnn-convert` to `AotExecutor`
round trip. Two checks fail at `2022de4` as well and are not caused by this
change:

- `tests/test_release_versions.py` hardcodes the `3.0.0-alpha.1` spelling while
  `Cargo.toml` already reads `3.0.0`; changing either is a version decision.
- `scripts/ci/check-doc-truthfulness.py` reports three link targets under
  `docs/reports/fn3-trained-fp32/` that were never committed, and
  `check-agents-freshness.py` reports two `src/fn3/AGENTS.md` files over its
  line budget.
