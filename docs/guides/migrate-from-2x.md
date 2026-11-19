# Migrate a 2.x inference workflow

For supported ONNX decoders, prepare a new artifact from the **original binary
ONNX export**. `fastnn-infer` replaces `fastnn-runtime` for decoder deployment,
but it is not a drop-in rename: the old `compile` command read an ONNX-style
JSON graph, and `run` read raw tensor buffers. The new CLI accepts token IDs and
executes a session with a KV cache.

```bash
cargo build --release --no-default-features --features cli --bin fastnn-infer
./target/release/fastnn-infer prepare --onnx model.onnx --out model.fnn3
./target/release/fastnn-infer inspect model.fnn3
./target/release/fastnn-infer run --model model.fnn3 --tokens 464,2068,7586
```

Pass `--onnx model.onnx` to `run` if you also want to verify the source against
the artifact. Otherwise the artifact runs on its own. See the
[CLI quickstart](../quickstart-fn3.md) for explicit decode, generation and device
selection. In Python use [`fastnn.inference`](../reference/python-api.md)
(`prepare`, `load`, `model.session()`, `prefill`, `decode`); in Rust use
`fastnn::inference::model::PreparedModel` and `Session`.

| Starting point | What to do |
|---|---|
| `.fnnc` plus `.memory.json` | These are a legacy serialized `ExecutablePlan` and a separate memory layout. Neither is an input to 3.0; there is no translator. Re-prepare from the source ONNX if supported. If you only have the old files, use the matching 2.x revision to reproduce their execution. |
| `.fnn` model, general ONNX graph (e.g. YOLO), or eager `Tensor` / `nn` | Keep using the [secondary 2.x APIs](../index.md#retained-2x-reference). `fastnn-convert` still writes `.fnn`, and the legacy AOT/CPU path still runs it; neither is an alias for the decoder runtime. |
| `fastnn-runtime bench` or `quantize` | Use [`fn3-bench`](../../tools/bench/README.md) to benchmark supported prepared decoders; it does not consume raw `.fnnc` plans. There is no new equivalent of the old CLI's `quantize` subcommand (which copied the plan unchanged); do not treat legacy quantization flags or packed weights as portable to 3.0. |
| 2.x W4A8 packed weights or quantized `.fnn` models | The 3.0 quantized path is weight-only INT8 with FP32 activations, packed at runtime from a prepared artifact: `--weight-int8` on `run`/`generate`, or `weight_int8=True` in `load`. It is opt-in, covers static MatMul weights, reports its own coverage and byte counters, and is not a port of the 2.x AOT W4A8 pipeline. See [artifact INT8](../quantization/artifact-int8.md). |
| Already using a 3.0 alpha `.fnn3` | The entry-point renames are `fn3-infer` → `fastnn-infer`, `fastnn.fn3` → `fastnn.inference` and `fastnn::fn3` → `fastnn::inference`. The renames do not require regenerating the artifact. |

For the caller audit, source pointers and reproduction boundary, see the
[2.x disposition](../internals/legacy-2x.md).
