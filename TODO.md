# TODO: Comprehensive Bug Fixes and Performance Optimizations

## BLOCK 1 — CRITICAL CRASHES (fix these first, they make the library unusable)

- [x] **BUG-19** `src/optim/adamw.rs` — Remove the debug `panic!` at step 60:
  `if !self.step.is_empty() && self.step[0] == 60 { panic!(...) }`
  Delete this entire block. It unconditionally crashes ALL training at step 60.
  Commit: `fix(optim): remove debug panic at step 60 in AdamW` ✓

- [x] **BUG-20** `src/optim/adamw.rs` — Remove the debug `panic!` in `AdamW::new()`:
  `if p.shape() == vec![32, 64, 64] { panic!(...) }`
  Delete this entire block. It crashes initialization of any model with a [32,64,64] parameter.
  Commit: `fix(optim): remove debug panic for shape [32,64,64] in AdamW::new`

---

## BLOCK 2 — SILENT CORRECTNESS BUGS (training produces wrong results silently)

- [x] **BUG-35** `src/optim/sgd.rs` — SGD momentum velocity is reset to zeros on EVERY step.
  The velocity tensor is allocated fresh inside `step()` each call.
  Fix: move velocity state to a `Vec<Tensor>` field on the `SGD` struct (same pattern as Adam's `m`/`v` fields),
  initialized in `SGD::new()` as zeros matching param shapes, and updated in-place each step.
  Commit: `fix(optim): persist SGD momentum velocity across steps`

- [x] **BUG-18** `src/optim/adam.rs` — Weight decay is computed but NEVER applied.
  The `decay` tensor is computed but not subtracted from `param`.
  Fix: after computing `step_size`, apply:
  `param = param - step_size - decay`
  where `decay = lr * weight_decay * param` (the L2 regularization term).
  Note: AdamW correctly decouples weight decay; Adam should apply it as L2 on the gradient.
  Commit: `fix(optim): apply weight decay in Adam step`

- [x] **BUG-21** `src/optim/adamw.rs` — AMSGrad uses `.max(dim, keepdim)` (spatial max over dimension 0)
  instead of element-wise max across time steps.
  Fix: replace `self.v[i].clone().max(0, false)` with an element-wise max between
  `self.v_hat[i]` (previous max) and `self.v[i]` (current second moment):
  implement a `tensor_elementwise_max(a, b)` helper that takes max of corresponding elements,
  then assign `self.v_hat[i] = tensor_elementwise_max(self.v_hat[i].clone(), self.v[i].clone())`.
  Commit: `fix(optim): fix AMSGrad v_hat to use element-wise temporal max`

- [x] **BUG-25** `fastnn/__init__.py` — The `__getitem__` monkey-patch returns a raw numpy array,
  breaking autograd for any indexing inside a gradient-tracked computation.
  The comment in the file already says it was intentionally removed — but then it's added back.
  Fix: remove the `new_get_item` function and the line `tensorcls.__getitem__ = new_get_item` entirely.
  Users who need numpy indexing should call `.numpy()[idx]` explicitly.
  Commit: `fix(python): remove __getitem__ patch that breaks autograd`

- [x] **BUG-1** `src/kernels/cpu.rs` (`linspace_kernel`) and `src/lib.rs` (`linspace` Python binding) —
  Division by zero when `steps == 1`: `i as f32 / (steps - 1) as f32` = `0/0 = NaN`.
  Fix in BOTH places:
  ```rust
  let t = if steps <= 1 { 0.0 } else { i as f32 / (steps - 1) as f32 };
  ```
  Commit: `fix(kernels): fix linspace division by zero when steps=1`

- [x] **BUG-2** `src/kernels/cpu.rs` (`randint_kernel`) and `src/lib.rs` (`randint` Python binding) —
  `rng.gen::<i32>() % (high - low)` has modulo bias and panics on negative values.
  Fix in BOTH places: replace with `rng.gen_range(low..high)` from the `rand` crate.
  Commit: `fix(kernels): use gen_range for unbiased randint`

- [x] **BUG-4** `src/kernels/cpu.rs` (`EmbeddingBackward`) — `num_inputs()` returns `1` but
  the backward node stores `[weight, indices]` (2 inputs).
  Fix: change `fn num_inputs(&self) -> usize { 1 }` to return `2`.
  Commit: `fix(autograd): fix EmbeddingBackward num_inputs returning 1 instead of 2`

- [x] **BUG-5** `src/kernels/cpu.rs` (`embedding_kernel` and `EmbeddingBackward`) —
  Indices tensor is cast as `*const f32` and then read as float and cast to usize.
  For vocab sizes > 16M this loses precision (f32 has only 23-bit mantissa).
  Fix: read indices as `*const i32` (or check dtype and handle i32/i64 appropriately),
  then cast to usize directly:
  ```rust
  let indices_ptr = indices.data_ptr() as *const i32;
  let idx = unsafe { *indices_ptr.add(i) } as usize;
  ```
  Apply the same fix in `EmbeddingBackward::apply`.
  Commit: `fix(kernels): read embedding indices as i32 not f32`

- [x] **BUG-14** `src/lib.rs` — `sum()` and `mean()` Python wrappers default `dim=None` to `dim=0`
  silently via `dim.unwrap_or(0)`. A user calling `fnn.sum(t)` expecting full reduction gets
  only axis-0 reduction.
  Fix: when `dim` is `None`, perform a full reduction (flatten then sum/mean) rather than defaulting to 0.
  Implement a `full_reduce` path that reshapes to 1D and calls sum/mean with dim=0.
  Commit: `fix(python): sum/mean with dim=None performs full reduction`

- [x] **BUG-6** `src/train/loss.rs` — Cross-entropy with `reduction="none"` returns a zero tensor
  of the input shape instead of per-sample losses.
  Fix: implement the actual per-sample loss computation:
  for each sample i, compute `-log(softmax(logits[i])[target[i]])` and store in output[i].
  Commit: `fix(loss): cross_entropy reduction=none returns per-sample losses`

- [x] **BUG-32** `src/nn/attention.rs` — Causal mask created as `[seq_len, seq_len]` 2D tensor
  is added to attention scores of shape `[B, H, S, S]` without explicit unsqueeze for B and H dims.
  Fix: unsqueeze the mask to `[1, 1, seq_len, seq_len]` before adding, so it broadcasts correctly
  over batch and head dimensions:
  ```rust
  let mask = mask.unsqueeze(0).unsqueeze(0); // [1,1,S,S]
  scores = scores.add(&mask);
  ```
  Commit: `fix(attention): unsqueeze causal mask for correct batched broadcast`

- [x] **BUG-31** `src/tensor.rs` — `Tensor::slice` with `step != 1` uses integer truncation
  `(stop - start) / step` which silently drops the last partial window.
  Fix: use ceiling division:
  ```rust
  let new_size = (stop - start + step - 1) / step; // ceiling division
  ```
  Commit: `fix(tensor): fix slice size ceiling division for non-unit steps`

- [x] **BUG-15** `src/lib.rs` — `argmax` and `argmin` dispatch to the `max`/`min` kernel with
  `keepdim=1.0` as a proxy for "return indices mode". This conflates value-returning and
  index-returning semantics in the same kernel.
  Fix: implement dedicated `argmax_kernel` and `argmin_kernel` in `cpu.rs` that find the index
  of the max/min element and return it as a tensor, then register them separately in the dispatcher.
  Commit: `fix(kernels): implement dedicated argmax/argmin kernels`

- [x] **BUG-9** `src/kernels/cpu.rs` — FMA matmul loop guard is `kk + 8 <= k_max` but the loop
  body reads two 8-float windows at `kk` and `kk+8` (16 floats total) and advances `kk += 16`.
  The final 8-element window can read 8 bytes past the buffer end.
  Fix: change loop guard to `kk + 16 <= k_max`:
  ```rust
  while kk + 16 <= k_max { ... kk += 16; }
  // then handle remaining with the 8-wide loop
  while kk + 8 <= k_max { ... kk += 8; }
  // then scalar tail
  while kk < k_max { ... kk += 1; }
  ```
  Commit: `fix(kernels): fix FMA matmul loop guard to prevent OOB read`

- [x] **BUG-23** `src/kernels/cpu.rs` — `fused_linear_relu` and `fused_linear_silu` parallel paths
  cast raw pointers to `usize` and dereference them inside Rayon closures using byte-offset arithmetic
  with hardcoded `* 4` (assumes sizeof(f32)==4). This is technically UB and fragile.
  Fix: use `std::slice::from_raw_parts` to create slices before entering the parallel region,
  then pass slice references into the closure. Use `bytemuck` for safe casting if needed.
  Commit: `fix(kernels): replace raw pointer arithmetic with safe slices in fused linear kernels`

- [x] **BUG-24** `src/train/trainer.rs` — `Trainer::fit` only inserts `train_loss` into `TrainLogs`.
  `EarlyStopping(monitor="val_loss")` will never find the key and will never trigger.
  Fix: after the validation loop, compute `val_loss` and insert it into `logs.metrics`
  with key `"val_loss"` before calling `on_epoch_end` callbacks.
  Commit: `fix(trainer): pass val_loss to callbacks for early stopping`

- [x] **BUG-29** `fastnn/parallel.py` — `DataParallel.forward_backward` averages losses with
  `sum(losses) / len(losses)` ignoring that GPUs have unequal batch sizes (weighted split).
  Fix: compute weighted average using actual batch sizes:
  ```python
  total_samples = sum(batch_sizes)
  avg_loss = sum(l * n for l, n in zip(losses, batch_sizes)) / total_samples
  ```
  Commit: `fix(parallel): weighted loss averaging in DDP forward_backward`

- [x] **BUG-26** `src/train/callbacks.rs` — `LearningRateScheduler` "step" schedule uses
  `self.lr * self.gamma.powf(epoch / self.step_size as f64)` which is exponential decay,
  not step decay. Integer division is needed so LR only drops at multiples of `step_size`.
  Fix:
  ```rust
  "step" => self.lr * self.gamma.powf((epoch / self.step_size) as f64),
  ```
  (cast the integer division result to f64 AFTER dividing, not before)
  Commit: `fix(callbacks): fix step LR scheduler to use integer epoch/step_size`

---

## BLOCK 3 — INCOMPLETE / STUB IMPLEMENTATIONS

- [x] **BUG-7** `src/io/serialize.rs` — `save_model` prints "Saved model" but never writes a file.
  The `safetensors` crate is already in `Cargo.toml`. Implement actual saving:
  1. Collect all named parameters from the model via `parameters()`.
  2. Build a `HashMap<String, safetensors::tensor::TensorView>` from them.
  3. Call `safetensors::serialize_to_file(&tensors, &None, path)`.
  Also implement `load_model` symmetrically using `safetensors::load`.
  Commit: `feat(io): implement save_model and load_model using safetensors`

- [x] **BUG-27** `src/train/callbacks.rs` — `ModelCheckpoint::on_epoch_end` prints a message
  but never writes to disk. `dirpath` is stored but unused.
  Fix: when `is_best || !self.save_best_only`, call the now-working `save_model()` with path
  `format!("{}/checkpoint_epoch_{}.safetensors", self.dirpath, epoch)`.
  Requires passing a model reference into the callback — refactor `Callback::on_epoch_end`
  to accept an optional `&dyn Module` parameter, or store an `Arc<Mutex<dyn Module>>` in the callback.
  Commit: `feat(callbacks): implement actual file saving in ModelCheckpoint`

- [x] **BUG-28** `src/train/callbacks.rs` — `CSVLogger` never writes to a CSV file.
  Fix: on first `on_epoch_end` call, create the file at `self.filepath`, write a CSV header
  from `logs.metrics.keys()`. On subsequent calls, append a row of values.
  Use `std::fs::OpenOptions::append(true)` for appending.
  Commit: `feat(callbacks): implement CSV writing in CSVLogger`

- [x] **BUG-30** `fastnn/data.py` — `DataLoader` with `num_workers > 0` silently no-ops.
  Fix: either implement basic prefetching using Python `threading.Thread` with a queue,
  or raise `NotImplementedError("num_workers > 0 is not yet supported")` with a clear message.
  Do not silently ignore user intent.
  Commit: `fix(data): raise error or implement threading for num_workers > 0`

- [x] **BUG-22** `src/autograd/engine.rs` — No `retain_graph` support. Calling `.backward()` twice
  panics or silently returns wrong gradients.
  Fix: add `retain_graph: bool` parameter to the `backward()` function.
  When `retain_graph=false` (default), behavior is unchanged.
  When `retain_graph=true`, do not free intermediate tensors after backward traversal
  (do not drop `Arc` references to nodes after calling `apply()`).
  Commit: `feat(autograd): add retain_graph parameter to backward`

---

## BLOCK 4 — SIMD / VECTORIZATION BUGS

- [x] **BUG-8** `src/kernels/cpu.rs` — AVX-512 `tanh` kernel loads a `__m512` (16 floats) then
  immediately transmutes to `[f32; 16]` and processes element by element. The SIMD load is wasted.
  Fix: use the `wide` crate's `f32x8` (two iterations of 8) with `.exp()` which is fully vectorized:
  ```rust
  let v = f32x8::from(chunk[0..8]);
  let exp_2x = (f32x8::splat(2.0) * v).exp();
  let result = (exp_2x - f32x8::ONE) / (exp_2x + f32x8::ONE);
  ```
  Mirror the pattern already used correctly in the AArch64 tanh path.
  Commit: `perf(kernels): fix AVX-512 tanh to use vectorized exp via wide crate`

- [x] **PERF-7** `src/kernels/cpu.rs` — `sigmoid_simd_x86` loads 8 floats into AVX2 register
  then immediately transmutes back to scalar array and processes one by one.
  Fix: same as tanh — use `f32x8` from the `wide` crate:
  ```rust
  let x = f32x8::from(*in_chunk);
  let result = f32x8::ONE / (f32x8::ONE + (-x).exp());
  *out_chunk = result.into();
  ```
  Commit: `perf(kernels): fix sigmoid_simd_x86 to use vectorized exp`

---

## BLOCK 5 — PERFORMANCE OPTIMIZATIONS (implement after all bugs fixed)

- [x] **PERF-2** `src/kernels/cpu.rs` — `broadcast_index_decomposition` allocates a `Vec` on the
  heap inside a hot loop called once per output element.
  Fix: change `let mut multipliers = vec![0usize; ndim]` to
  `let mut multipliers = smallvec::SmallVec::<[usize; 8]>::from_elem(0, ndim)`.
  The `smallvec` crate is already in `Cargo.toml`. This eliminates heap allocation for tensors
  up to 8 dimensions (covers 99.9% of real use cases).
  Commit: `perf(kernels): use SmallVec for broadcast_index_decomposition multipliers`

- [x] **PERF-11** `src/kernels/cpu.rs` — The SIMD fast-path in `add_kernel` (and sub/mul/div)
  requires `a_shape == b_shape`. The most common case in neural nets — `[B, D] + [D]` bias add —
  always falls to the slow scalar broadcast path.
  Fix: add a second fast path specifically for the `[N, D] + [D]` broadcast pattern:
  detect when `b.shape() == [a.shape().last()]`, then use a SIMD loop that applies the
  same `b` slice to each row of `a`. This covers `Linear`, `LayerNorm`, and `BatchNorm` bias adds.
  Commit: `perf(kernels): add SIMD fast-path for [N,D]+[D] broadcast add`

- [x] **PERF-12** `src/kernels/cpu.rs` — `embedding_kernel` copies rows element by element.
  Fix: replace the inner `j` loop with `std::ptr::copy_nonoverlapping`:
  ```rust
  unsafe {
      std::ptr::copy_nonoverlapping(
          weight_ptr.add(idx * embedding_dim as usize),
          out_ptr.add(i * embedding_dim as usize),
          embedding_dim as usize,
      );
  }
  ```
  Same fix in `EmbeddingBackward::apply` for gradient accumulation (use `ptr::add` in a loop for
  accumulation since it's a += not =, but the inner loop over `j` can be replaced with a SIMD sum).
  Commit: `perf(kernels): use ptr::copy_nonoverlapping for embedding row copies`

- [x] **PERF-8** `src/lib.rs` and `src/kernels/cpu.rs` — `randn` uses Box-Muller transform
  (one `ln()` + one `cos()` per 2 samples). Replace with the `StandardNormal` distribution
  from `rand_distr` crate which uses the much faster Ziggurat method:
  ```rust
  use rand_distr::{Distribution, StandardNormal};
  let val: f32 = StandardNormal.sample(&mut rng);
  ```
  Apply in both the Rust kernel and the Python binding path.
  Commit: `perf(kernels): use Ziggurat (StandardNormal) instead of Box-Muller for randn`

- [ ] **PERF-13** `fastnn/data.py` — `DataLoader.shuffle()` re-allocates `indices` list every epoch.
  Fix: pre-allocate `self.indices = list(range(len(dataset)))` in `__init__`, then shuffle in-place:
  ```python
  random.shuffle(self.indices)  # in-place, no re-alloc
  ```
  Commit: `perf(data): shuffle DataLoader indices in-place to avoid reallocation`

- [ ] **PERF-1** `src/optim/adam.rs` and `src/optim/adamw.rs` — Each optimizer step allocates
  ~6–8 intermediate tensors per parameter.
  Fix: implement a fused in-place parameter update kernel `adam_update_kernel` in `cpu.rs` that
  takes raw pointers to `param`, `m`, `v`, `grad` and updates all in a single SIMD loop:
  ```
  m[i] = beta1 * m[i] + (1-beta1) * g[i]
  v[i] = beta2 * v[i] + (1-beta2) * g[i]^2
  m_hat = m[i] / (1 - beta1^t)
  v_hat = v[i] / (1 - beta2^t)
  param[i] -= lr * m_hat / (sqrt(v_hat) + eps) + wd * param[i]
  ```
  All in one pass, zero heap allocations. Register as `"adam_update"` in the dispatcher.
  Commit: `perf(optim): fused in-place Adam/AdamW parameter update kernel`

- [ ] **PERF-3** `src/kernels/cpu.rs` — `layer_norm_kernel` and `batch_norm_kernel` make 8+
  dispatcher roundtrips (mean, var, std, normalize each as separate ops).
  Fix: implement a single fused `layer_norm_fused` kernel using Welford's online algorithm:
  ```rust
  // Single pass over the normalized dimensions:
  let mut mean = 0.0f32;
  let mut m2 = 0.0f32;
  for (n, &x) in slice.iter().enumerate() {
      let delta = x - mean;
      mean += delta / (n + 1) as f32;
      m2 += delta * (x - mean);
  }
  let variance = m2 / count as f32;
  ```
  Then normalize in a second pass. Two passes total, no intermediate tensor allocations.
  Commit: `perf(kernels): fused Welford LayerNorm/BatchNorm kernel`

- [ ] **PERF-4** `src/kernels/cpu.rs` — `softmax_kernel` does 4 separate passes: max, subtract, exp, sum, divide.
  Fix: implement online softmax in 2 passes:
  Pass 1: find max value (or use online max+sum combined with the log-sum-exp trick).
  Pass 2: compute `exp(x - max)` and accumulate sum, then divide.
  Alternatively implement fully online single-pass softmax using the numerically stable recurrence.
  Remove the 4 intermediate tensor allocations.
  Commit: `perf(kernels): fused two-pass softmax with no intermediate allocations`

- [ ] **PERF-5** `src/kernels/cpu.rs` — `fused_linear_relu` and `fused_linear_silu` implement
  their own naive O(B*M*N) matmul loop instead of calling `matmul_blas`.
  Fix: restructure both kernels to:
  1. Call `matmul_blas(x_ptr, w_ptr, batch_size, in_features, out_features)` for the matmul.
  2. Add bias in a second SIMD pass if present.
  3. Apply activation (relu/silu) in a third in-place SIMD pass.
  This gets BLAS-level performance for the matmul while keeping the fusion benefit of no
  intermediate tensor allocation for the activation.
  Commit: `perf(kernels): use BLAS matmul inside fused_linear_relu/silu kernels`

- [ ] **PERF-14** `src/nn/conv.rs` — `conv2d_kernel` uses 6 nested scalar loops.
  Fix: implement `im2col` transformation followed by a single `matmul_blas` call:
  1. `im2col`: reshape input patches into a 2D matrix of shape `[out_h*out_w, C_in*kH*kW]`.
  2. Reshape weight to `[C_out, C_in*kH*kW]`.
  3. Call `matmul_blas(im2col_matrix, weight.T)` → result shape `[out_h*out_w, C_out]`.
  4. Reshape result to `[B, C_out, out_h, out_w]`.
  Allocate the `im2col` buffer once per forward call on the stack if small, heap otherwise.
  Commit: `perf(nn): implement im2col + BLAS for Conv2d kernel`

- [ ] **PERF-6** `src/dispatcher.rs` — Every tensor op does `REGISTRY.read().unwrap().get(op_name)`
  — a `RwLock` acquisition + `HashMap` string lookup + clone on the hot path.
  Fix:
  1. Define a numeric `OpId` enum with one variant per op (`Add`, `Sub`, `Mul`, ...).
  2. Replace the `HashMap<String, KernelFn>` with two flat arrays `CPU_KERNELS: [KernelFn; NUM_OPS]`
     and `WGPU_KERNELS: [KernelFn; NUM_OPS]` indexed by `OpId as usize`.
  3. Keep the string→OpId lookup only for the Python API boundary; internal Rust code uses `OpId` directly.
  Commit: `perf(dispatcher): replace HashMap dispatch with flat array indexed by OpId enum`

- [ ] **PERF-9** `src/autograd/engine.rs` — Topological sort is rebuilt on every `.backward()` call
  even for static graphs that never change between training steps.
  Fix: cache the topo order as `Option<Vec<Arc<dyn Node>>>` on the root tensor's autograd metadata.
  Invalidate (set to `None`) whenever `requires_grad` is changed or a new operation creates a new node.
  On backward, if cache is `Some`, use it directly; otherwise compute and store.
  Commit: `perf(autograd): cache topological sort for static computation graphs`

- [ ] **PERF-10** `src/autograd/engine.rs` — Gradient accumulation allocates a new tensor for each `add`.
  Fix: replace `grad_in = grad_in.add(&incoming)` with in-place accumulation.
  Implement `Tensor::add_assign_inplace(&mut self, other: &Tensor)` that writes directly into
  `self`'s storage buffer using a SIMD loop, with no allocation.
  Use it in the backward accumulation loop.
  Commit: `perf(autograd): in-place gradient accumulation to eliminate allocs`

- [ ] **PERF-15** `src/kernels/cpu.rs` — `CHUNK_MEMBOUND` threshold (`numel > 2048`) is used
  uniformly for both heavy and cheap ops. For cheap ops (relu, abs, neg, exp) the Rayon thread
  spawn overhead dominates for tensors under ~32K elements.
  Fix: define per-category thresholds:
  ```rust
  const THRESHOLD_CHEAP_UNARY: usize = 32_768;   // relu, abs, neg
  const THRESHOLD_TRANSCENDENTAL: usize = 8_192;  // exp, log, sqrt, sin
  const THRESHOLD_BINARY: usize = 4_096;           // add, sub, mul
  const THRESHOLD_REDUCTION: usize = 2_048;        // sum, mean, max
  ```
  Replace all `numel > 2048` checks with the appropriate category constant.
  Commit: `perf(kernels): tune per-operation parallelism thresholds`

- [x] **PERF-19** `src/kernels/cpu.rs` — The `wide` crate is already a dependency but only used
  in AArch64 tanh. All x86 paths for `gelu`, `silu`, `exp` use raw intrinsics and scalarize
  for transcendentals.
  Fix: for `gelu_kernel`, `silu_kernel`, `exp_kernel` on x86_64, use `f32x8` from `wide`:
  ```rust
  let v = f32x8::from(*in_chunk);
  let result = v * (f32x8::ONE / (f32x8::ONE + (-v).exp())); // SiLU
  *out_chunk = result.into();
  ```
  This gives vectorized transcendentals without needing AVX-512.
  Commit: `perf(kernels): use wide crate f32x8 for gelu/silu/exp on x86_64` ✓

- [ ] **PERF-18** `fastnn/parallel.py` — DDP `sync_gradients` does one Python-level `add` + `div`
  dispatch per parameter in a Python loop.
  Fix: implement a `allreduce_gradients(params_a, params_b)` function on the Rust side that
  takes two lists of parameter tensors and computes `(a.grad + b.grad) / 2` for all of them
  in a single batched pass, avoiding per-parameter Python→Rust dispatch overhead.
  Register it as `"allreduce"` in the Python module.
  Commit: `perf(parallel): implement batched allreduce in Rust for DDP gradient sync`

- [ ] **PERF-20** `src/storage.rs` — Every `Tensor::zeros/empty/ones` calls the global allocator.
  During training, thousands of temporary tensors are allocated and freed per step.
  Fix: implement a `StepArena` — a thread-local bump allocator that:
  1. Pre-allocates a large block (e.g., 256MB) at training start.
  2. Serves tensor storage from this block during forward+backward.
  3. Resets (bumps pointer back to start) at the end of each optimizer step.
  Tensors that should outlive a step (parameters, gradients) use the global allocator as today.
  Only intermediate activations use the arena.
  Commit: `perf(storage): implement step-scoped arena allocator for intermediate tensors`

---

## BLOCK 6 — CLEANUP

- [x] **BUG-17** `src/kernels/cpu.rs` — `read_f32` and `write_f32` are marked `#[allow(dead_code)]`
  and are never used anywhere. Either wire them up to replace the raw pointer reads in kernels
  (for a safer API), or delete them.
  Decision: delete them since all kernels now use safe slice patterns (from PERF-5 above).
  Commit: `chore(kernels): remove unused read_f32/write_f32 utilities`

- [x] **BUG-16** `src/kernels/cpu.rs` — GPU cross-entropy fallback silently transfers to CPU
  with no warning. Add a `eprintln!` or `log::warn!` so users know this is happening:
  ```rust
  eprintln!("[fastnn WARNING] cross_entropy_loss: GPU kernel not implemented, falling back to CPU. This incurs a PCIe transfer penalty.");
  ```
  Same for `gt_scalar` GPU fallback.
  Commit: `fix(kernels): warn on silent GPU→CPU fallbacks`

- [x] **BUG-3** `src/kernels/cpu.rs` — The `#[cfg(not(simd))]` scalar fallback block inside
  `single_threaded_matmul` is placed inside the `unsafe` block for the FMA path, referencing
  variables not in scope. Dead code, but it prevents clean compilation with `--cfg simd`.
  Fix: move the fallback block outside the `if is_x86_feature_detected!("fma")` block,
  or remove it entirely since the scalar path is handled by the loop below.
  Commit: `fix(kernels): fix misplaced cfg(not(simd)) block in matmul`
