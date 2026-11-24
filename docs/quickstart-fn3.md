# FastNN 3.0 inference quickstart

One path: `prepare` a `.fnn3` artifact from ONNX once, then `load` and `run` it.
The version-2 artifact carries the typed graph, one full plan per target
profile, the weights in a memory-mapped payload and a SHA-256 per section, so a
deployment never needs the ONNX it was prepared from and does not re-plan. See
[docs/v3/artifact.md](v3/artifact.md) for the format.

The typed-graph runtime is CPU-first. The CNN operator set runs on the scalar
NCHW kernels and, with `--features cpu-onednn`, on the optional oneDNN engine.
The replaced version-1 decoder CLI (`prepare`/`inspect`/`run`/`generate`/
`diagnose` with token IDs) is documented only in
[the archive](archive/reports/fn3-17-cli-quickstart.md); it is not this surface.

## 0. Build

```bash
cargo build --release --features cli
```

The binary is `target/release/fastnn-infer`. Add `cpu-onednn` for the oneDNN CPU
engine: `cargo build --release --features cli,cpu-onednn`.

## 1. Prepare

```bash
./target/release/fastnn-infer prepare \
  --onnx yolov8n.onnx \
  --out yolov8n.fnn3 \
  --target all
```

`--target all` writes two plans from one import:

```text
fastnn-infer prepare
  source: yolov8n.onnx
  imported: 240 typed nodes from 262 ONNX nodes, 398 values, 149 weights (12,649,900 bytes)
  target:   all (2 profiles, precision f32)
    cpu            segments=126 fusions=57 reorders=0 (0) arena=2,867,200 B peak=2,867,200 B
    cpu-reference  segments=240 fusions=0 reorders=0 (0) arena=6,144,000 B peak=4,915,200 B
  wrote: yolov8n.fnn3
    header: magic FNN3ART2, version 2, 128 B, 5 sections, weights at byte 173,952
  verified: read back through Artifact::open (header, SHA-256 per section, graph, weight table and every plan)
```

The write is atomic and is read back through the strict reader before success is
reported. The source ONNX is not modified. `cpu` is the default plan; `cpu-reference`
is the scalar, unfused oracle.

### A dynamic export

A stock Ultralytics export declares its input as `[batch, 3, height, width]` and
builds `Reshape` targets from those symbols, so the plan has nothing concrete to
compile. Bind the shape the model will run at:

```bash
./target/release/fastnn-infer doctor yolo11n.onnx
#   inputs: images: [batch, 3, ?(dim_param `height` has no fn3 symbol), ...]
./target/release/fastnn-infer prepare \
  --onnx yolo11n.onnx --out yolo11n.fnn3 \
  --shape images=1,3,640,640
```

`--shape name=d0,d1,...` may be repeated, once per graph input. `doctor` takes
the same flag and prints both the declared input shapes and the verdict the
bound model would get; an unknown name, the wrong rank or the same name twice
is a typed error naming the input.

## 2. Inspect

```bash
./target/release/fastnn-infer inspect yolov8n.fnn3
```

It prints the header, graph size, every target plan, weight bytes, quantization
metadata and the stored section hashes. It never maps the weight payload:
`mapped bytes 0` and `bytes read … of … (weight payload untouched)` are printed
as the evidence. `Artifact::open` (the `load` path) does verify every hash,
including the payload, before the first segment can run.

`explain` goes one level deeper and prints the stored plan itself:

```bash
./target/release/fastnn-infer explain yolov8n.fnn3
```

Every segment is listed with its engine (`scalar` or `onednn`), its op or fused
chain (`conv+swish`), its output slots and bytes and the source node; then the
provider's fallbacks with their reasons, the fused chains and the layout
reorders. It reads the plan from the artifact, opens no session and does not map
the weight payload. `--target <name>` selects another stored plan.

## 3. Load

```bash
./target/release/fastnn-infer load yolov8n.fnn3 --target cpu --threads 4
```

This maps the artifact, builds the session and executes nothing. It prints the
plan the file carries, the engine split, the arena and any provider fallbacks.

## 4. Run

Inputs are named `f32` values. `zeros` and `ones` are smoke inputs; a file is raw
little-endian `f32` bytes with exactly the element count the plan declares.

```bash
./target/release/fastnn-infer run yolov8n.fnn3 \
  --input images=zeros \
  --threads 4 \
  --output-dir out/
```

Every graph input must be supplied exactly once. The output line names the
tensor, its shape, element count, min/max/mean and finite count. With
`--warmup <n> --repeat <n>` the same command prints the median and p95 of the
measured series and this process's peak RSS (`VmHWM`), so a latency claim and
its memory come from one run.

## 5. Doctor

Before preparing an unfamiliar model, run the preflight:

```bash
./target/release/fastnn-infer doctor yolov8n.onnx
```

It imports, checks backend support, dynamic-shape limits and dtypes, and prints
an expected resident-memory estimate without writing an artifact or executing a
segment. Every finding names its source node, e.g.
`node 7 (onnx "Conv", output "y")`.

## 6. Rust and Python

Rust:

```rust
use fastnn::inference::{prepare, load, inspect, doctor};
use fastnn::inference::runtime::Threading;

let report = prepare("yolov8n.onnx", "yolov8n.fnn3", "cpu", "f32")?;
let info = inspect("yolov8n.fnn3")?;
assert_eq!(info.mapped_bytes, 0);
let mut session = load("yolov8n.fnn3", Threading::Fixed(4))?;
let outputs = session.run(&[("images", &vec![0.0f32; 3 * 320 * 320])])?;
let report = doctor("yolov8n.onnx")?;
# Ok::<(), fastnn::inference::Fn3Error>(())
```

Python:

```python
import fastnn.inference as fn3

report = fn3.prepare("yolov8n.onnx", "yolov8n.fnn3", target="all")
info = fn3.inspect("yolov8n.fnn3")          # mapped_bytes == 0
model = fn3.load("yolov8n.fnn3")
session = model.session(threads=4)
outputs = session.run({"images": [0.0] * (3 * 320 * 320)})
print(outputs["output0"].shape)
doctor = fn3.doctor("yolov8n.onnx")
```

The Python raster is `Model.session().run({name: values})`; there is no
tokenizer, no sampling and no hidden device fallback.

## 7. Failures

Every failure is two lines: `error: …` and `next: …`. Exit codes are `0`
success, `2` usage, `3` model/artifact, `5` inference and `1` internal. An
obsolete version-1 file names itself and says to re-prepare the source ONNX; a
future version says to use a matching build. A corrupt section names the section
and the absolute byte offset.
