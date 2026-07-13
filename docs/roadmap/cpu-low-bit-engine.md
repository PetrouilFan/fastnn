# CPU Low-Bit Engine Direction

## Status

This document evaluates the design proposals in the external research report
`deep-research-report.md` against fastnn's current architecture roadmap. The
report is useful as architectural input, but its inline citation identifiers are
not independently resolvable from the document. Proposed formats, performance
claims, and dated implementation schedules are therefore not accepted as
verified requirements without source links and fastnn measurements.

The accepted conclusion is narrower and stronger: fastnn should compile
explicit storage and compute contracts into workload- and ISA-specific kernel
families. It should not treat every compressed representation as another scalar
dtype or route every low-bit format through one generic packed inner loop.

## Accepted findings

### Distinguish three bit measures

Every compressed representation must expose three different quantities:

- **Logical alphabet width**: the information represented by one value, such as
  a ternary alphabet or a signed four-bit integer domain.
- **Physical encoding width**: the payload bits and alignment used by the actual
  storage layout.
- **Effective bits per weight**: payload plus scales, offsets, codebooks,
  exception tables, masks, alignment slack, and other amortized metadata.

`DType::logical_bit_width()` and exact payload sizing are only the beginning of
this separation. Effective size belongs to the complete representation artifact,
not the scalar enum.

### Separate storage format from compute family

A compiled quantized operand needs explicit, orthogonal descriptors for:

- logical scalar or alphabet;
- payload bit width and signedness;
- packing layout, word order, tile order, and endianness;
- scale/offset schema and parameter scalar types;
- axis, group, block, and superblock granularity;
- optional codebook or exception metadata;
- preferred decode/compute family;
- required ISA capabilities;
- supported operator and workload phases.

The same nominal INT4 values may require different physical layouts for an
unpack-to-I8 dot-product kernel, a floating dequantization fallback, or another
specialized family. Those layouts are distinct compiled storage formats even
when their logical quantization is identical.

### Use kernel families, not a universal packed kernel

The compiler and capability system should distinguish at least:

- unpack-to-I8 integer dot product;
- unpack-to-F16/BF16/F32 fallback;
- direct packed integer accumulation where proven useful;
- lookup-table accumulation;
- binary/ternary masked or bit-serial accumulation;
- codebook/vector-quantized staged decode.

Kernel selection must consider operator class, workload shape, target ISA, and
reuse pattern. A storage format must never imply that one compute family is
universally optimal.

### Make execution phase explicit

For sequence-model workloads, decode and prefill are materially different
execution regimes. The compiler/runtime contract should be able to distinguish:

- single/few-token, weight-bandwidth-dominated decode;
- multi-token prefill or batched GEMM;
- convolution and other reuse-heavy operators;
- attention/KV-cache paths;
- training forward and backward.

This belongs with graph kind, operation capabilities, and typed compile options;
it must not be inferred from model layer names.

### Pack for the selected kernel

Payload and metadata layout are lowering decisions. Requirements include:

- reduction-dimension streaming appropriate to the microkernel tile;
- bounded, block-local decode;
- explicit metadata placement and alignment;
- no entropy-coded variable-length stream in a hot random-access execution
  format;
- separate distribution/cold-storage encoding when maximum compression conflicts
  with executable layout.

The current separate payload and metadata vectors should not be assumed optimal.
Any move to interleaved or adjacent metadata must be benchmarked rather than
adopted as a universal rule.

## Priority changes to the existing roadmap

### Tier 1: complete the production W4/W8 foundation

Before adding new bit widths:

1. Finish canonical representation and quantized operand descriptors.
2. Finish accumulator, bias, correction, requantization, rounding, saturation,
   and output-storage contracts.
3. Add CPU capability detection and typed kernel-family selection.
4. Establish one measured W4A8 or W4A16 path with scalar and ISA-specific
   implementations.
5. Compare direct packed accumulation, unpack-to-I8, and explicit floating
   fallback on the actual supported CPUs and workloads.

No new INT3, INT2, ternary, or VQ production format should bypass this gate.

### Tier 2: workload-aware low-bit experiments

Only after Tier 1 has end-to-end wins and stable contracts:

- compare word-local and dense-stream INT3 layouts;
- investigate W2 or ternary LUT/masked kernels for bandwidth-dominated decode;
- investigate vector-LUT layouts for parallel prefill;
- add grouped KV compression only together with a fused attention contract and
  online-quantization cost measurements.

These are experiments first. Promotion requires correctness, quality,
performance, memory, and binary-size evidence.

### Tier 3: research formats

General VQ, residual/additive codebooks, trellis/lattice formats, transformed
sub-4-bit KV, and integer-heavy QAT remain research-tier. Their storage gains do
not justify production status without a concrete decode-to-accumulate design.

## Decisions not accepted as current requirements

The following report proposals are deliberately not added as implementation
commitments:

- exact names or wire layouts such as `QSI4_G32`, `QSI3_WL30`, `QTRIT160`, or
  `QVQ2x8`;
- fixed group sizes such as 32 or 64 as universal defaults;
- unconditional K-major storage for every operator;
- specific calendar estimates;
- transformed three-bit KV, broad VQ, or CPU-only large-model QAT;
- performance claims whose cited source cannot be resolved from the report.

They may become benchmark candidates after the typed contracts exist.

## Compiler and IR additions

The canonical representation work should add types equivalent in responsibility
to, but not necessarily named exactly as:

- `StorageFormat`: payload encoding and executable layout;
- `QuantizationSchema`: scale, offset, group, axis, and codebook semantics;
- `DecodeFamily`: how payload reaches the accumulator;
- `KernelFamily`: operation-specific compute contract;
- `IsaRequirement`: required and preferred CPU features;
- `WorkloadPhase`: decode, prefill, general GEMM, convolution, attention,
  training-forward, or training-backward;
- `ExceptionSchema`: optional outlier/mask/mixed-precision side data.

These are compiler/lowering concepts. Mathematical IR opcodes should remain
independent of a particular microkernel.

## Capability and dispatch requirements

Capability metadata must answer whether a kernel family supports a combination
of:

- operator;
- logical representation;
- storage format;
- activation and weight quantization;
- bias and output representation;
- layout and dynamic shape;
- workload phase;
- target ISA;
- training/autograd mode.

Compilation rejects unsupported combinations before backend dispatch. Runtime
selection may choose among precompiled compatible variants, but must not silently
change numerical semantics.

## Benchmark and promotion gates

Measure four levels independently:

1. microkernel;
2. operator;
3. representative decode/prefill/conv/attention layer;
4. end-to-end model.

Required metrics include:

- latency and throughput separated by workload phase;
- cycles per element/weight/token;
- payload bytes and effective bytes including metadata;
- temporary/scratch bytes and allocation count;
- L1/L2/LLC and TLB behavior where tooling permits;
- quality delta against the same checkpoint and calibration data;
- generated binary/code size for AOT specialization;
- scalar-versus-ISA numerical parity.

A compressed format is promoted only when it beats the relevant wider baseline
end to end. Model-file size alone is not sufficient.

## AOT specialization policy

Specialize generated code only on dimensions that alter the kernel contract:

- ISA feature set;
- kernel/decode family;
- storage format and bit width;
- group and metadata schema;
- a bounded set of tile shapes.

Do not specialize on layer names or every concrete hidden size. Optional hot
shape presets require measured wins and binary-size accounting.

## Relationship to the main roadmap

This direction extends rather than replaces the current work order:

1. recoverable APIs and canonical representation;
2. typed attributes, graph/workload kinds, and capabilities;
3. explicit integer arithmetic and output contracts;
4. production W4/W8 kernel-family baseline;
5. only then experimental sub-4-bit, KV, ternary, or VQ families.

The immediate roadmap remains focused on semantic correctness and ownership.
Adding more packed formats before those foundations are complete would recreate
the dtype and dispatch ambiguity currently being removed.
