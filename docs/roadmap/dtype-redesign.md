# Dtype and Quantization Redesign

## Decision

The current dtype model mixes four independent concerns:

1. The mathematical value domain (`f32`, `i32`, boolean, etc.).
2. The physical storage encoding (plain bytes versus packed words).
3. Quantization semantics (signedness, scale, zero point, axis, group, codebook).
4. Compilation policy (which target representation the compiler should select).
5. The executable storage layout selected for a kernel family.
6. The decode/compute pathway used to reach the accumulator.

CPU low-bit execution makes the last two distinctions essential: a nominal bit
width does not identify either the physical layout or the inner loop.

These must not be represented by overlapping enums with embedded metadata.

Today the same concepts are spread across `storage::DType`, `ir::IrDType`,
`backend::WeightDtype`, Python `Precision`, `PackedWord`, `PackedTensor`, and
`QuantizedTensor`. In particular, `IrDType` owns quantization vectors directly,
while `DType` uses names such as `I8Scaled`, and `WeightDtype` duplicates target
selection. This makes type checking, serialization, memory sizing, lowering, and
Python configuration disagree by construction.

## Target model

Use one canonical Rust schema with orthogonal concepts.

```text
ValueType        mathematical scalar semantics
StorageEncoding  physical byte/word representation
Quantization     numerical mapping between logical and stored values
TensorLayout     shape, strides, order, alignment requirements
Device           execution location
CompileTarget    requested optimization/quantization policy
StorageFormat    executable payload and metadata layout
DecodeFamily     unpack, lookup, masked, codebook, or floating fallback path
```

For CPU-first sub-8-bit planning and the evidence accepted from the external
research report, see [`cpu-low-bit-engine.md`](cpu-low-bit-engine.md).

Suggested core types:

```rust
pub enum ScalarType {
    F32, F16, BF16,
    I64, I32, I8, U8, I4, U4,
    Bool,
    Fp8E4M3, Fp8E5M2, Fp4E2M1,
}

pub enum StorageEncoding {
    Plain,
    Packed { word_bits: u8, lanes: u8 },
}

pub struct Quantization {
    pub scheme: QuantScheme,
    pub axis: Option<usize>,
    pub group_size: Option<usize>,
    pub scale: QuantParams,
    pub zero_point: Option<QuantParams>,
    pub codebook: Option<CodebookSpec>,
}

pub enum ValueRepresentation {
    Native { scalar: ScalarType },
    Quantized {
        logical: ScalarType,
        stored: ScalarType,
        encoding: StorageEncoding,
        quantization: Quantization,
    },
}

pub enum CompileTarget {
    Native,
    WeightOnly(QuantTarget),
    IntegerInference(QuantTarget),
    TrainingMixedPrecision(PrecisionPolicy),
}
```

Names above are deliberately illustrative. The implementation must settle exact
ownership before changing code. The non-negotiable rule is that no single
`DType` variant carries unrelated scale vectors, zero-point vectors, codebooks,
packing geometry, and compilation intent.

## Semantic rules

- A logical tensor value describes semantics, not the backend's chosen kernel.
- Packing is storage layout, not a different mathematical dtype.
- Quantization parameters belong to a representation/constant artifact, not to
  the global scalar type enum.
- `I8` and `U8` are actual scalar domains. Do not use misleading aliases such
  as a signed representation named after an unsigned type.
- Codebook quantization is a quantization scheme, not an `I4` subtype.
- FP8/FP4 formats are scalar/storage formats; their scale policy is separate.
- Activation, weight, gradient, and optimizer-state quantization use the same
  representation schema but have distinct compiler policies.
- Kernels consume an explicit typed operand contract. They do not infer signedness,
  scales, packing, or accumulator requirements from strings or scattered attrs.
- Logical alphabet width, physical payload width, and effective bits per value
  including metadata are separate measurements.
- A nominal bit width does not identify an executable layout. Different kernel
  families may require different layouts for the same logical quantization.
- Cold-storage/distribution compression and hot executable packing may use
  different formats.

## Integer-space U4/U8 policy

For integer inference, the compiler must make the arithmetic domain explicit:

```text
quantized activation + quantized weight
  → integer dot-product / integer accumulator
  → bias in accumulator domain
  → requantize to declared output representation
```

Rules:

- Accumulator scalar type, bias scale, zero-point correction, and requantization
  scale are explicit lowering data.
- FP32 may be used only at explicit boundary operations or for calibration; it
  is not an implicit scratch representation for U4/U8 kernels.
- Q/DQ pairs are canonicalized after every quantization rewrite.
- Quantized-output and dequantized-output operations are distinct typed
  operations or lowering forms; do not encode that difference in ad hoc attrs.

## Ownership

| Concern | Single owner |
|---|---|
| Scalar and storage representation schema | new `src/types/` or `src/tensor/types/` module |
| Logical IR value types | `src/ir/types.rs` using the shared schema |
| Quantization policy and transformations | `src/compiler/quantization/` |
| Packed word implementation and CPU/WGPU kernels | `src/kernels/` or backend leaf modules |
| Tensor storage allocation | `src/tensor/storage/` |
| Python configuration/bindings | generated or thin bindings over the Rust schema |
| File-format tags | serialization module mapping canonical schema to an explicitly versioned format |

`PackedTensor` and `QuantizedTensor` should be evaluated for consolidation into a
single representation owner. They currently overlap in packed data, scale/zero
metadata, shape, and conversion behavior.

## Migration sequence

1. Inventory every current dtype/precision/packing use and classify it as value
   semantics, storage encoding, quantization metadata, or compile policy.
2. Write the canonical schema and a type-validation API before migrating kernels.
3. Move `DType`, `IrDType`, and `WeightDtype` consumers to the canonical schema;
   delete the old enums rather than maintaining permanent conversion layers.
4. Move quantization vectors and codebooks out of IR scalar enum variants into
   explicit representation metadata.
5. Consolidate `PackedTensor` and `QuantizedTensor`, with one owner for packing,
   dequantization, shape, metadata, and cache policy.
6. Rework compiler quantization into separate policy, calibration, rewrite,
   canonicalization, and lowering stages.
7. Change CPU kernels to accept typed quantized operand descriptors and explicit
   integer accumulator/requantization contracts.
8. Add typed storage-format, decode-family, workload-phase, and ISA capability
   selection; establish one measured production W4 path before adding sub-4-bit
   production formats.
9. Replace Python `Precision` duplication with a thin Rust-backed configuration
   API; remove manually synchronized dtype tags.
10. Version the resulting model/plan format once; no legacy reader is required.
11. Delete obsolete aliases, duplicate size calculations, and type conversions.

## Required tests

- Exhaustive representation validation: legal/illegal scalar, packing, scheme,
  scale, zero-point, axis, and group combinations.
- Exact packed byte-size/layout tests for every encoding.
- Cross-language serialization tests using only the new format.
- Integer reference tests for U4/U8 matmul and convolution, including asymmetric
  zero points, bias fusion, tails, groups, and requantization.
- No-FP32-scratch instrumentation tests for declared integer-space paths.
- Differential tests comparing compiler output against a simple reference
  quantizer and accumulator implementation.
- Property tests for Q/DQ canonicalization and representation-preserving graph
  rewrites.
- Effective-bits accounting tests that include metadata and alignment.
- Workload-separated benchmarks for decode, prefill, convolution, and attention.
- Kernel-family promotion gates comparing microkernel, operator, and end-to-end
  results against wider baselines.

## Explicit deletions

The redesign should remove, not preserve:

- `I8Scaled`, `U4Scaled`, and `U8Scaled` as overloaded global dtype labels.
- Quantization scale/zero/codebook vectors embedded in `IrDType` variants.
- `WeightDtype` as a parallel precision taxonomy.
- Manually synchronized Python integer tags and Rust dtype mappings.
- Duplicate packed storage abstractions unless they have genuinely distinct
  runtime roles after the audit.

## Implementation hazards found during the audit

- `DType::size()` reports an ambiguous per-element byte value for packed formats
  while allocation needs word rounding. Replace it with explicitly named logical
  bit-width and storage-byte calculations.
- `PackedWord` currently requires generic FP32 unpack/pack and dot-product
  methods. The new abstraction must not force integer kernels through FP32.
- `QuantizedTensor::to_packed()` performs a two-pass conversion that reduces
  blockwise metadata to global metadata. The consolidated representation must
  preserve declared quantization granularity.
- Activation quantization currently has two rewrite implementations. The new
  representation schema must feed one canonical rewrite/canonicalization path.
- Add a direct-output U4/U8 regression test for exact output byte sizing. The
  executor's packed-output special case currently treats U4/U8 differently from
  the other packed representations; resolve that inconsistency before migration.
- Add instrumentation tests proving declared integer-space U4/U8 paths have no
  implicit FP32 conversion or scratch allocation.
