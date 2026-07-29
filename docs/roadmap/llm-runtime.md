# Architecture-neutral LLM runtime roadmap

## Goal

Support transformer and LLM workloads through reusable ONNX, IR, compiler,
runtime-state, and CPU execution contracts. GPT-2 is the first whole-model
conformance fixture, not a model-specific backend architecture.

Initial model matrix:

- `onnx-community/tiny-random-gpt2-ONNX`: learned positions, LayerNorm, GELU,
  ordinary multi-head attention, and two-layer KV cache.
- `onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX`: RoPE, RMSNorm expressed as
  primitives, SiLU/SwiGLU, and grouped-query attention geometry (32 query heads,
  4 KV heads).
- `onnx-community/Qwen2.5-0.5B-Instruct-ONNX-GQA`: a second GQA layout (14 query
  heads, 2 KV heads), fused ONNX Runtime contrib operators, tied embeddings, and
  a much larger vocabulary/context contract.

## M0 live audit

Audit date: 2026-07-28. Baseline: `v2.6.0`, commit
`e63eda85c4d88daff446de80d57070c802244f04`.

### Graph inventory

| Model | Nodes | Key geometry | Distinct graph requirements |
|---|---:|---|---|
| tiny GPT-2 | 257 | 2 layers, 2 Q/KV heads, head dim 64 | LayerNorm decomposition, GELU decomposition, Split, dynamic shape arithmetic, MHA KV cache |
| TinyLlama | 2,133 | 22 layers, 32 Q heads, 4 KV heads, head dim 64 | RoPE through Sin/Cos, RMSNorm decomposition, SwiGLU, GQA expansion, Trilu mask, KV cache |
| Qwen2.5 GQA explorer graph | 298 | 24 layers, 14 Q heads, 2 KV heads, head dim 64 | `GroupQueryAttention`, `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`, large-vocabulary output |

The Qwen graph is an optimized model-explorer graph without weights. It is useful
for interface and fused-op reconnaissance, not numerical execution evidence.

### Primitive operator matrix

| Operator/pattern | GPT-2 | TinyLlama | Current v2.6 status |
|---|---:|---:|---|
| MatMul/Gemm | 13 | 200 | Opcode/import path exists; rank/broadcast/model parity unproven |
| Gather | 35 | 163 | Exists; dynamic index and dtype contracts unproven |
| Reshape | 25 | 136 | Import exists, but runtime shape inputs may silently become pass-through |
| Slice | 4 | 112 | Import currently consumes static attributes, not general opset-14 tensor inputs |
| Concat | 23 | 247 | Symbolic output expressions exist; runtime cache append semantics absent |
| Split | 2 | 0 | Symbolic dimensions can silently duplicate the full input to every output |
| Expand | 2 | 51 | Opcode exists; all audited model cases require runtime-produced target shapes |
| Shape | 20 | 95 | Exists, but typed shape-tensor execution and downstream propagation are unverified |
| Range | 1 | 5 | Exists; runtime limits and shape propagation unverified |
| ConstantOfShape | 2 | 1 | Runtime-produced shapes explicitly unsupported |
| Equal/Less/Greater/Where | 8 | 97 | Paths exist, but tensor comparison semantics and broadcasting need differential tests |
| Softmax | 2 | 22 | Exists; arbitrary-axis and attention-shape parity unverified |
| Sin/Cos | 0 | 2 | Missing from canonical IR/importer |
| Trilu | 0 | 1 | Missing from canonical IR/importer |
| ScatterND | 0 | 1 | Exists; model pattern and mutation semantics unverified |
| Fused GQA | 0 | 0 in primitive graph | Qwen optimized graph requires unsupported contrib op |
| Fused simplified norms | 0 | 0 in primitive graph | Qwen optimized graph requires unsupported contrib ops |

### Dynamic auxiliary-input pressure

Modern opsets encode shape behavior as tensor inputs rather than attributes.
The audited graphs contain:

| Pattern | tiny GPT-2 | TinyLlama |
|---|---:|---:|
| Reshape with runtime-produced shape | 16/25 | 133/136 |
| Slice with runtime-produced auxiliary input | 2/4 | 24/112 |
| Expand with runtime-produced shape | 2/2 | 51/51 |
| Range with runtime-produced bound | 1/1 | 5/5 |

Therefore static attribute parsing is not a viable general LLM import strategy.

## Confirmed blockers

### 1. Unsupported ONNX operations silently change semantics

`src/onnx/converter.rs` currently maps unknown operations to their first input (or
a scalar zero). This can produce a successfully converted graph with incorrect
numerics. LLM support requires strict unsupported-operation errors with node name,
domain, opset, inputs, and inferred types. Optional explicit decompositions may be
added only when covered by differential tests.

### 2. Dynamic Reshape silently changes semantics

When a Reshape target cannot be resolved from a constant parameter, the converter
currently returns the data input unchanged. This is not a valid fallback.

### 3. Dynamic Split silently changes semantics

When a split dimension is symbolic, every output currently receives the complete
input. This is not a valid approximation.

### 4. Slice does not implement general opset-14 inputs

The converter reads static `starts`, `ends`, and `axes` attributes, while audited
models supply these through tensor inputs. Steps, negative indices, multiple axes,
and runtime endpoints need explicit contracts.

### 5. Typed constant round-trip is broken for the GPT-2 fixture

The public conversion call accepted the complete 27.9 MB tiny GPT-2 ONNX graph and
wrote a 29.5 MB `.fnn` file. Reading that file immediately failed:

```text
ValueError: cannot reshape array of size 4 into shape (2,)
```

The graph has 24 F32 and 24 I64 initializers. The first implementation boundary is
therefore typed initializer serialization/deserialization, not a GPT-2 wrapper.
A conversion success result is not meaningful until the emitted artifact reads
back and preserves every initializer byte, dtype, shape, and graph edge.

### 6. Runtime allocation reuse is not persistent model state

`GraphExecutor` caches an arena by capacity and resolves runtime shapes for each
execution, but the arena is an execution scratch resource. It does not define
persistent mutable tensors, semantic cache extents, append operations, reset,
session isolation, or cache overflow behavior.

### 7. External tensor data needs an explicit import contract

TinyLlama's graph has 201 external F32 initializers. General LLM loading must safely
resolve external-data paths, enforce aggregate budgets, preserve tensor dtypes and
shapes, and report missing payloads before compilation.

## Dependency-ordered milestones

### Current implementation status

- **M1 complete:** strict unsupported-operation errors, exact F32/I64/I32/Bool
  initializer round trips, external-data validation, symbolic-name preservation,
  and structural reload tests.
- **M2 substantially complete for the GPT-2 path:** static opset-13/14 Slice,
  Split, Squeeze, and Unsqueeze inputs; live Shape and ConstantOfShape; bounded
  tensor-driven Reshape and Slice; per-symbol allocation capacities; symbolic and
  affine extent resolution; runtime Expand; and N-D Add/Where broadcasting are
  implemented and differentially tested.
- The tiny random GPT-2 cached-decode fixture now converts, builds, and executes
  end-to-end. Logits and returned K/V tensors match ONNX Runtime: one-step logits
  have maximum absolute error `5.96e-7`, and three successive externally-fed cache
  steps retain the same `5.96e-7` maximum logit error while cache lengths grow from
  1 to 4.
- This establishes whole-model F32 numerical parity for an externally-managed
  GPT-2 cache, but it is **not** yet the persistent KV/session API. The caller still
  feeds every layer's past K/V tensors and receives every present K/V tensor on each
  invocation.
- **M3 is now active:** validate modern Llama/GQA primitives and importer paths.
  TinyLlama currently fails preflight because its external payload
  `model.onnx_data` is missing from the audit artifact. The weightless optimized
  Qwen explorer graph remains structural reconnaissance only and exposes fused GQA
  and simplified-normalization boundaries; it is not numerical execution evidence.
- Persistent runtime-owned KV state, prefill/decode session APIs, modern-model
  reference parity, and W4A8 remain subsequent gates.

### M1: Honest, typed ONNX boundary

1. Reject unsupported ops instead of pass-through.
2. Make advertised Python mappings agree with Rust converter support.
3. Round-trip F32, F16/BF16 where supported, I64, I32, and Bool initializers.
4. Validate external-data paths and aggregate sizes.
5. Preserve symbolic dimension names instead of collapsing all non-positive or
   symbolic dimensions to an undifferentiated `Unknown`.
6. Add convert-write-read structural equality tests.

Exit gate: tiny GPT-2 converts, reloads, and retains exact typed constants and graph
structure; unsupported TinyLlama/Qwen nodes fail explicitly.

### M2: Shape-tensor semantics

Implement and differentially test the common runtime shape algebra:

- Shape and typed shape tensors
- Gather over shape tensors
- Add/Sub/Mul/Div on shape values
- Cast
- Unsqueeze/Squeeze
- Concat
- Reshape with tensor target
- Expand with tensor target
- Range with runtime bounds
- ConstantOfShape with bounded runtime shape
- Slice with opset-14 tensor inputs
- Split with tensor sizes and symbolic extents

Exit gate: synthetic ONNX subgraphs from all three model families match ONNX
Runtime over multiple batch, prompt, and past-cache lengths.

### M3: Transformer primitive conformance

Verify ordinary tensor execution before adding fused architecture-specific ops:

- N-D/batched MatMul and broadcast rules
- embedding Gather with I64 indices
- arbitrary-axis Softmax
- LayerNorm and RMSNorm decompositions
- GELU and SiLU/SwiGLU
- Sin/Cos for RoPE
- Trilu or an equivalent causal-mask decomposition
- comparison/Where broadcasting
- GQA head expansion/repetition

Exit gate: isolated subgraphs match PyTorch and ONNX Runtime, including malformed
shape and axis errors.

### M4: Dynamic semantic shapes versus capacity

Represent bounded allocation capacity separately from the live semantic shape.
At minimum this must cover batch, input sequence length, past sequence length,
total sequence length, query heads, KV heads, and head dimension.

Exit gate: one compiled bounded graph runs several prompt and cache lengths without
padding every operation to maximum context or reading beyond the live extent.

### M5: Generic persistent state

Add architecture-neutral session-owned mutable state with:

- stable state IDs,
- dtype/layout/capacity/live shape,
- append/overwrite policy,
- checked bounds,
- reset,
- independent sessions,
- explicit graph readers/writers.

KV cache is the first user of this contract, not a special Python convention.

### M6: Explicit prefill and decode

Compile and profile prefill and decode as distinct workloads. Prefill consumes a
prompt and initializes state; decode consumes one or more new tokens and appends to
state without recomputing the prompt.

### M7: Whole-model F32 gates

1. tiny GPT-2 prefill parity.
2. tiny GPT-2 cached per-token decode parity against full recomputation.
3. TinyLlama primitive-graph parity using the same state/runtime contracts.
4. GQA geometry and cache parity with unequal Q/KV head counts.

No backend dispatch may branch on a model name.

### M8: Generic model/session API

Separate immutable compiled model/prepared weights, mutable inference session, and
optional tokenizer/sampling utilities. Python must not copy every layer's K/V
outputs back as next-step inputs.

### M9: Quantization

Apply grouped W4A8 only after F32 prefill/decode gates pass. Report exact mixed
execution contracts and test G32/G64/G128 using logit, first-divergent-layer,
perplexity, prefill, decode, memory, and temporary-copy gates.

### M10: Profile-driven optimization

Optimize only measured model bottlenecks. Reassess the old branch's output decoding
and no-copy MatMul ideas against current v2.6 ownership and model-level evidence;
do not cherry-pick them mechanically.

### Later expansion

- BERT-like bidirectional encoders
- T5-like encoder-decoder and cross-attention state
- sliding-window attention
- beam-search state cloning/reordering
- MoE routing

These reuse the same typed importer, dynamic-shape, state, and session foundations.

## Immediate implementation boundary

Start with M1, specifically a typed ONNX artifact round-trip test using the tiny
GPT-2 graph's mixed F32/I64 initializers. In the same boundary, remove silent
unsupported-op, dynamic-Reshape, and symbolic-Split approximations so every later
probe produces trustworthy failures instead of plausible-looking wrong graphs.
