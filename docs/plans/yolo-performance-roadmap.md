# YOLO CPU Performance Roadmap

> **For Hermes:** Use this as the execution roadmap for bringing fastnn YOLO CPU inference closer to PyTorch/Ultralytics/ONNX Runtime.

**Goal:** Make fastnn significantly faster on YOLOv8n CPU inference while preserving accuracy against PyTorch and keeping default builds stable.

**Architecture:** Execute three coordinated lanes: graph/thread scheduling, non-GEMM overhead reduction/fusion, and optimized Conv/runtime kernels. Each lane must land only behind clear gates and must be measured against the same YOLO benchmark scripts.

**Tech Stack:** Rust CPU backend, Python AOT executor bindings, optional OpenBLAS, YOLOv8n ONNX/PT fixtures, PyTorch/Ultralytics/ONNX Runtime comparison scripts.

---

## Current measured baseline

Commit baseline: `e7f7693 perf(cpu): add openblas conv gemm escape hatch`.

Build:

```bash
VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv \
  .venv/bin/python -m maturin develop --release --features 'prepared-plan openblas'
```

Representative CPU timings on `[1,3,320,320] -> [1,84,2100]`:

```text
ONNX Runtime                 mean ~10.3ms
Ultralytics raw, 4-8 threads mean ~18.2ms
Ultralytics raw, 2 threads   mean ~26.0ms
Ultralytics raw, 1 thread    mean ~34.5ms
PyTorch script raw, 1 thread mean ~37.6ms
fastnn OpenBLAS, 2 threads   mean ~44.9ms
fastnn OpenBLAS disabled     mean ~50-58ms depending sample
```

Accuracy gate already passes:

```text
fastnn OpenBLAS vs PyTorch max_abs  ~4.88e-4
fastnn OpenBLAS vs PyTorch mean_abs ~1.41e-6
```

Blunt target:

```text
Gate 1: fastnn <= 35ms mean on current machine/config
Gate 2: fastnn <= 25ms mean
Stretch: approach Ultralytics 4-thread raw forward (~18ms)
```

Do not chase ONNX Runtime first; it is a much more mature optimized runtime. Use it as upper-bound reference.

---

## Lane A: graph-level/thread scheduling

### Hypothesis

fastnn currently uses multithreaded OpenBLAS inside Conv GEMM, but the graph executor itself is mostly serial and may repeatedly pay per-kernel overhead. YOLO has many independent or low-cost elementwise/slice/concat operations around Conv. We can reduce wall time by controlling parallelism at the graph/kernel level instead of only inside BLAS.

### Risks

- Nested parallelism can regress badly, as seen with `OPENBLAS_NUM_THREADS=8`.
- Running independent kernels in parallel may fight BLAS threads and memory bandwidth.
- Python benchmark noise can hide small wins.

### Task A1: Add execution profile JSON export with per-kernel totals

**Objective:** Make profile comparisons machine-readable for before/after gates.

**Files:**
- Modify: `scripts/yolo_compare_fastnn_pytorch.py`

**Steps:**
1. Add `--profile-json <path>` if not already present.
2. Store full `fastnn_profile_top` and raw totals.
3. Verify:

```bash
OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/yolo_compare_fastnn_pytorch.py \
  --profile --profile-top 20 --warmup 3 --iters 10 \
  --profile-json /tmp/yolo_profile.json
```

**Acceptance:** JSON contains per-kernel `count`, `total_ms`, `mean_ms` and top-level fastnn timing.

### Task A2: Add thread sweep matrix for Torch and OpenBLAS

**Objective:** Compare fastnn OpenBLAS threads vs Ultralytics/PyTorch threads on one command.

**Files:**
- Create: `scripts/yolo_runtime_matrix.py`

**Steps:**
1. Sweep fastnn `OPENBLAS_NUM_THREADS=1,2,4,8`.
2. Sweep PyTorch/Ultralytics `torch.set_num_threads(1,2,4,8)`.
3. Save JSON.
4. Verify:

```bash
.venv/bin/python scripts/yolo_runtime_matrix.py --iters 10 --json /tmp/yolo_runtime_matrix.json
```

**Acceptance:** Reports current best fastnn, best PyTorch raw, best Ultralytics raw, and ratios.

### Task A3: Prototype executor-level parallel regions only after profiling

**Objective:** Identify graph regions safe for parallel execution without interfering with BLAS.

**Files:**
- Inspect: `src/backend/executor.rs`
- Possibly modify behind feature flag only.

**Steps:**
1. Generate dependency graph from existing executable plan instructions.
2. Identify non-Conv independent kernels after Conv-heavy regions.
3. Prototype with `FASTNN_EXPERIMENTAL_GRAPH_THREADS=1` env guard.
4. Set `OPENBLAS_NUM_THREADS=1` when graph parallelism is active to avoid nested parallelism.

**Acceptance:** Land only if full YOLO improves by >=5% vs best OpenBLAS-only setting and accuracy is unchanged.

---

## Lane B: non-GEMM overhead reduction/fusion

### Hypothesis

OpenBLAS reduced GEMM cost, making non-GEMM overhead more visible. Current profile still shows time in:

```text
conv2d_silu total ~35-38ms
transpose_perm_f32 ~0.9ms
write_const ~0.8ms
pool_f32 ~0.9ms in some runs
concat/slice/sigmoid overhead
```

Within Conv+SiLU itself, im2col and SiLU epilogue remain nontrivial.

### Task B1: Split Conv profile under OpenBLAS

**Objective:** Update `examples/conv_phase_bench.rs` or add OpenBLAS mode to show phase split with OpenBLAS active.

**Files:**
- Modify: `examples/conv_phase_bench.rs`
- Possibly docs: `docs/plans/conv-phase-bench.md`

**Steps:**
1. Run phase bench with and without `--features openblas`.
2. Record whether im2col/SiLU now dominate target shapes.
3. Verify:

```bash
cargo run --release --example conv_phase_bench -- 120
OPENBLAS_NUM_THREADS=2 cargo run --release --features openblas --example conv_phase_bench -- 120
```

**Acceptance:** Clear phase split under OpenBLAS for heavy YOLO shapes.

### Task B2: Fuse or cache constant writes in default path

**Objective:** Reduce repeated `write_const` overhead without relying on prepared fallback.

**Files:**
- Inspect: `src/backend/executor.rs`
- Inspect: prepared arena work in `src/backend/prepared.rs`

**Steps:**
1. Measure `write_const` cost with profile JSON.
2. Identify constants that are written every forward but never mutate.
3. Add default-path static constant preload/cache only if safe.
4. Verify exactness vs current default.

**Acceptance:** YOLO full profile improves >=1%; no change in outputs. This is small but safe.

### Task B3: Conv epilogue improvement

**Objective:** Reduce bias+SiLU overhead, especially stem and small/medium Conv outputs.

**Files:**
- Modify: `src/backend/cpu/microkernels.rs`
- Benchmark: `examples/conv_phase_bench.rs`

**Steps:**
1. Add isolated SiLU epilogue benchmark.
2. Try vectorized/SIMD or approximation only if accuracy gate can be defined.
3. Prefer exact `exp` path first; approximation requires explicit user approval.

**Acceptance:** Isolated epilogue improves >=10% and full YOLO improves >=1-2% with max_abs still in current tolerance.

---

## Lane C: ONNX Runtime style optimized kernels

### Hypothesis

The biggest remaining gap is that fastnn still uses generic im2col + GEMM + epilogue, while ONNX Runtime/PyTorch/Ultralytics use better convolution scheduling, packed weights, threading, and possibly oneDNN-style kernels.

### Task C1: Add optional oneDNN/ORT benchmark, not integration

**Objective:** Determine whether oneDNN direct Conv kernels close the gap for YOLO shapes.

**Files:**
- Create: `examples/conv_backend_probe.rs` or Python benchmark if easier.
- Docs: `docs/plans/conv-backend-probe.md`

**Steps:**
1. Benchmark current Conv, OpenBLAS Conv, and oneDNN/ORT-style Conv if available.
2. Use exact YOLO shapes.
3. Do not integrate before isolated win.

**Acceptance:** Candidate backend beats OpenBLAS Conv by >=20% on repeated heavy shapes.

### Task C2: Packed static Conv weights for OpenBLAS/custom kernels

**Objective:** Prepare static weights in the layout needed by the selected backend/microkernel.

**Files:**
- Modify: `src/backend/prepared.rs`
- Modify: `src/backend/cpu/microkernels.rs`
- Tests: prepared stats tests

**Steps:**
1. Add metadata-only packed weight store for the selected layout.
2. Do not change runtime yet.
3. Verify prepared stats and memory counts.

**Acceptance:** Metadata lands with zero runtime behavior change.

### Task C3: Custom blocked Conv/GEMM only if backend probe fails

**Objective:** Build a pure-Rust optimized kernel for one repeated YOLO shape.

**Initial target:**

```text
M=64,K=576,N=400 count=9
```

**Steps:**
1. Create `examples/conv_blocked_kernel_bench.rs`.
2. Implement a tiny shape-specific AVX2/FMA block kernel behind `target_feature` guard.
3. Compare against OpenBLAS Conv, not old matrixmultiply baseline.
4. Only integrate after isolated win.

**Acceptance:** >=20% faster than OpenBLAS Conv on target shape, exactness within fp32 tolerance.

---

## Execution order

Recommended order:

1. Task A1: profile JSON export.
2. Task A2: runtime matrix benchmark.
3. Task B1: OpenBLAS phase split.
4. Task B2: constant write/cache cleanup if clearly measurable.
5. Task C1: backend probe.
6. Decide:
   - if backend probe wins, integrate backend feature-gated;
   - else start C3 custom blocked kernel.
7. Only then attempt graph-level parallel execution A3.

Reasoning:

- Better measurement first prevents false wins.
- Non-GEMM overhead cleanup is lower risk.
- Backend probe can tell us whether to build or borrow optimized kernels.
- Graph parallelism is risky because it interacts with OpenBLAS threading.

---

## Global acceptance gates

Every performance commit must report:

```bash
git branch --show-current  # must be dev
cargo fmt --check
cargo test --release --lib
cargo test --release --lib --features prepared-plan
cargo test --release --lib --features 'prepared-plan openblas'
VIRTUAL_ENV=/home/petrouil/Projects/github/fastnn/.venv \
  .venv/bin/python -m maturin develop --release --features 'prepared-plan openblas'
OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/yolo_compare_fastnn_pytorch.py --profile --profile-top 20 --warmup 3 --iters 10
```

For any speed claim, also run:

```bash
FASTNN_DISABLE_OPENBLAS_CONV_GEMM=1 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/yolo_compare_fastnn_pytorch.py --profile --profile-top 20 --warmup 3 --iters 10
```

Accuracy gates:

```text
mean_abs_vs_pytorch <= ~2e-6 unless explicitly justified
max_abs_vs_pytorch should stay around current ~5e-4 scale
```

Speed gates:

```text
Small cleanup: >=1% full YOLO improvement
Kernel/backend change: >=5% full YOLO improvement
Major lane claim: must move toward <=35ms mean
```

Reject conditions:

- Faster only on microbench, no full YOLO gain.
- Requires fragile thread setting without escape hatch.
- Changes default non-OpenBLAS build behavior unexpectedly.
- Accuracy drift not explained by normal fp32 accumulation order.
- Adds integration before isolated backend/kernel win.
