# FastNN

[![CI](https://github.com/PetrouilFan/fastnn/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PetrouilFan/fastnn/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/PetrouilFan/fastnn)](https://github.com/PetrouilFan/fastnn/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

FastNN is a Rust inference runtime for supported ONNX decoder models, with Python bindings and a command-line interface. Prepare a model once as a `.fnn3` artifact, then run prefill and cached decoding from that artifact on CPU or Vulkan. The artifact contains the graph and weights, so it can run offline without the original ONNX file.

- **CPU inference** uses an adaptive thread budget by default. Rust callers can set a fixed budget.
- **Vulkan inference** keeps weights, intermediate tensors, and the KV cache on the device across steps. Explicit device requests fail with an error if the device or operation is unavailable.
- **Weight-only INT8** is opt-in. Eligible static projections use INT8 weights and FP32 activations. The runtime reports coverage, and other operations run in FP32 on the selected backend.

## Installation

| Interface | Requirements | Backends |
| --- | --- | --- |
| Python wheel | Python 3.11+ | CPU |
| `fastnn-infer` CLI | Rust 1.89+ and a source build | CPU, optional Vulkan |
| Rust library (`fastnn::inference`) | Rust 1.89+ and a source dependency | CPU, optional Vulkan |

Download the wheel for your platform from [GitHub Releases](https://github.com/PetrouilFan/fastnn/releases/latest). Release 3.0.0 provides wheels for Linux x86-64, macOS arm64, and Windows x86-64. Install the downloaded file in a virtual environment:

```bash
python -m pip install ./path/to/downloaded.whl
```

The repository's [installation guide](docs/guides/getting-started.md) covers source builds. For reproducible release builds, check out the corresponding Git tag.

## Run a model

From the repository root, fetch the pinned tiny GPT-2 model and build the CLI. The fetcher checks the model against the committed lock file. You can substitute a local, supported ONNX decoder.

```bash
python3 tools/parity/fetch_models.py --suite tiny-gpt2
MODEL=tools/parity/.cache/models/tiny-gpt2/onnx/model.onnx

cargo build --release --no-default-features --features cli --bin fastnn-infer
CLI=./target/release/fastnn-infer

"$CLI" prepare --onnx "$MODEL" --out tiny-gpt2.fnn3
"$CLI" generate --model tiny-gpt2.fnn3 --device cpu \
  --tokens 464,2068,7586,21831 --max-new-tokens 4
```

The API and CLI accept token IDs. Use the model's tokenizer in your application to encode prompts and decode generated IDs. For a GPU source build, use `--features cli,vulkan`, run `fastnn-infer diagnose` with a compatible Vulkan driver installed, then select `--device vulkan`. Add `--weight-int8` to `run` or `generate` to enable packed INT8 projections.

### Python

Load the artifact from the CLI example, or prepare one in Python with `inference.prepare("model.onnx", "tiny-gpt2.fnn3")`.

```python
from fastnn import inference

model = inference.load("tiny-gpt2.fnn3", device="cpu")
session = model.session()

logits = session.prefill([464, 2068, 7586, 21831])
next_token = int(logits.argmax())
next_logits = session.decode([next_token])
```

Each session owns its KV cache. Set `weight_int8=True` on `inference.load` to enable INT8 weights.

## Verification and performance

The 3.0.0 evidence covers tiny and trained GPT-2 on Linux x86-64, with physical Vulkan runs on a GeForce GTX 1080 Ti. Wheel availability is broader than the verified inference matrix. The importer accepts a defined subset of ONNX graphs and returns errors for unsupported operators or shapes. See the [support matrix](docs/reports/fn3-74-support-matrix.md) for tested platforms, drivers, model variants, and context lengths.

Quantization reduces the size of eligible projection weights. End-to-end latency and memory use depend on the model, backend, and context. The [3.0.0 validation report](docs/reports/fn3-3.0.0-release.md) links the benchmark commands and raw results.

## Documentation

- [CLI quickstart](docs/quickstart-fn3.md) and [Python API](docs/reference/python-api.md)
- [INT8 execution](docs/quantization/artifact-int8.md) and [runtime architecture](docs/roadmap/fn3-architecture.md)
- [Migration from 2.x](docs/guides/migrate-from-2x.md) and [changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md) and [documentation index](docs/index.md)
