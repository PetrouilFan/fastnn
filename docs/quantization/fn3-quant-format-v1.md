# fn3-quant-format-v1

**Status:** locked for FastNN 3.0 (#68). This document is authoritative for
#69 (CPU quantized inference) and #70 (Vulkan quantized inference). Any layout
change is a new `format_version` and a new identifier; v1 is not renegotiated.

**Identifier:** `fn3-quant-format-v1` (`fastnn::fn3::quant::FORMAT_ID`)
**Version:** 1
**Rust implementation:** `src/fn3/quant/`
**Independent implementation:** `tools/quant/fn3_quant.py` (standard library +
NumPy; shares no code with the Rust reader)
**Fixtures:** `fixtures/quant/`
**Decision record:** `docs/reports/fn3-22-quant-format.md`

## 1. Scope

v1 is **weight-only INT8** (W8A16):

* weights of the projection MatMuls are signed INT8 codes in groups of 64 along
  the reduction axis K, with an IEEE binary16 scale and an int16 integer
  zero-point per group;
* **activations stay FP32**; there is no activation quantization and no Q/DQ
  node;
* dequantization is fused into the projection kernel. No dequantized weight
  buffer is materialized;
* the trained GPT-2 evidence and acceptance thresholds are in the decision
  record. In summary: per-projection relative Frobenius error ≤ 0.0064,
  prefill/decode top-1 preserved, decode KL ≤ 0.013.

v1 has **no 4-bit payload and no nibble interleaving**. The grouped-W4
candidates were measured and rejected on accuracy; their layouts are documented
in the decision record, not here. The descriptor carries `qbits` so a reader can
reject a 4-bit blob with `UnsupportedQBits` rather than misinterpret it. A
4-bit format, if revisited, is `v2`.

## 2. Byte order and alignment

* Every integer is little-endian.
* The file is exactly `payload_offset + payload_bytes` bytes; no trailing bytes.
* `payload_offset` is a multiple of 16.
* Every tensor's code section starts at a multiple of 16 (checked by the
  reader). Row starts are therefore 16-byte aligned because `row_stride_bytes`
  is a multiple of 16.
* Metadata (header, descriptors, group directory) is contiguous and canonical;
  offsets are not free-form. The reader rejects a non-canonical layout even if
  it is internally consistent. This is deliberate: one logical object has one
  byte representation, so fixtures and checksums are meaningful.

## 3. Container

### 3.1 Header (64 bytes)

| Offset | Size | Field | Value / meaning |
| ---: | ---: | --- | --- |
| 0 | 4 | `magic` | ASCII `FN3Q` (`46 4E 33 51`) |
| 4 | 2 | `version` | `1` |
| 6 | 2 | `header_bytes` | `64` |
| 8 | 4 | `flags` | `0` in v1; any other value is rejected |
| 12 | 4 | `tensor_count` | `1..=4096` |
| 16 | 4 | `descriptor_bytes` | `tensor_count * 64` |
| 20 | 4 | `group_dir_bytes` | `total_groups * 8` |
| 24 | 8 | `descriptor_offset` | `64` |
| 32 | 8 | `group_offset` | `descriptor_offset + descriptor_bytes` |
| 40 | 8 | `payload_offset` | `align_up(group_offset + group_dir_bytes, 16)` |
| 48 | 8 | `payload_bytes` | length of the payload section |
| 56 | 4 | `payload_crc32` | IEEE CRC-32 of bytes `[payload_offset, end)` |
| 60 | 4 | `meta_crc32` | IEEE CRC-32 of bytes `[0, payload_offset)` with this field zeroed |

`total_groups` is the sum of every descriptor's `groups_per_row`.

### 3.2 Tensor descriptor (64 bytes each)

| Offset | Size | Field | Value / meaning |
| ---: | ---: | --- | --- |
| 0 | 32 | `name` | ASCII `0x20..=0x7E`, NUL-terminated when shorter than 32, unique |
| 32 | 4 | `n_rows` | N > 0 |
| 36 | 4 | `k` | K > 0 |
| 40 | 2 | `qbits` | `8` |
| 42 | 2 | `group_size` | `64` |
| 44 | 4 | `groups_per_row` | `ceil(k / 64)` |
| 48 | 4 | `group_dir_index` | index of this tensor's first group entry |
| 52 | 4 | `row_stride_bytes` | `align_up(k, 16)` |
| 56 | 8 | `data_offset` | absolute, multiple of 16 |

Descriptors are in tensor order; `group_dir_index` is the cumulative sum of the
previous descriptors' `groups_per_row`.

### 3.3 Group directory (8 bytes per group)

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 4 | `k_start` |
| 4 | 4 | `k_len` |

For tensor `t` with `G = groups_per_row`, entries
`[group_dir_index, group_dir_index + G)` describe its groups in order:

* group 0 starts at `k_start = 0`;
* every next group starts exactly where the previous ended (no gap, no overlap);
* every non-final `k_len` is 64;
* the final `k_len` is `k - 64 * (G - 1)`, in `1..=64`;
* the final end is exactly `k`;
* every directory entry is claimed by exactly one descriptor.

K does **not** need to be a multiple of 64. `k = 70` has groups `(0, 64)` and
`(64, 6)`; `k = 8` has one group `(0, 8)`.

### 3.4 Payload

Tensors appear in descriptor order. For a tensor with `N` rows, `G` groups and
`row_stride = align_up(k, 16)`, at `data_offset`:

```text
N * row_stride              code bytes, row-major, row padding zero
N * G * 2                   binary16 scales, row-major [row][group]
N * G * 2                   int16 zero-points, row-major [row][group]
padding to the next 16-byte boundary, zero
```

`data_offset` of each tensor is `align_up(previous tensor end, 16)`; the first
is `payload_offset`. All padding bytes are zero and are covered by the payload
CRC.

## 4. Element encoding

### 4.1 Codes

Each row is `k` signed INT8 codes followed by `row_stride - k` zero bytes.
Code `c` represents the integer `c` in `[-128, 127]` (two's complement).
There is no nibble packing in v1; every weight is one byte.

### 4.2 Scale

Per `(row, group)`, an IEEE 754 binary16 value in little-endian byte order.
It must be finite and strictly positive. The reader rejects `0`, negative,
infinite, NaN and any denormal that rounds to zero when widened (the canonical
encoder never emits one). The value used by all arithmetic is the binary16
value widened to f32 exactly; the encoder quantizes against that widened value,
so the stored scale is the operative scale.

### 4.3 Zero-point

Per `(row, group)`, a signed **int16** in `[-128, 127]`. It is the code that
represents 0.0. Storing 16 bits (rather than one byte) makes an out-of-range
value representable and therefore rejectable with a typed error. The valid range
is a property of the 8-bit code space; v1 does not accept a wider offset.

### 4.4 Dequantization

```text
w[row, col] = (code[row, col] - zero_point[row, col / 64]) * scale[row, col / 64]
```

`code - zero_point` is computed in `i32` (range `-255..=255`) and converted to
f32; the multiply and result are f32. This is exactly
`QuantTensor::dequantize_into` in `src/fn3/quant/format.rs` and
`dequantize()` in `tools/quant/fn3_quant.py`.

### 4.5 Encoder policy (normative for prepared artifacts)

The format does not require a particular encoder, but #69/#70 must be able to
reproduce the reference policy. For each row and group:

```text
span       = max - min
asym_scale = span / 255
asym_zero  = -128 - round_half_away(min / asym_scale)

if the group is constant:
    if value == 0: scale = 1.0
    else:          scale = max(|value| / 127, 1e-6)
    zero = 0
elif -128 <= asym_zero <= 127:
    scale = asym_scale, zero = asym_zero
else:
    # A signed code range cannot represent a group whose mean is far from zero
    # (for example an all-positive group with a small span). Symmetric is the
    # special case zero = 0.
    scale = max(|min|, |max|) / 127, zero = 0

scale is rounded to binary16 first; the codes are computed against that value:
code = clamp(round_half_away(value / scale_f16) + zero, -128, 127)
```

`round_half_away` is round-to-nearest with ties away from zero, for both signs.
The scale is a binary16 value; `scale_f16` is its exact f32 widening. The
quantizer never emits a zero or non-finite scale.

The asymmetry decision is per `(row, group)`, so one tensor can contain both
asymmetric and symmetric groups.

## 5. Projection arithmetic

For a prepared weight tensor `W[N, K]` and an FP32 activation row
`x[K]`, the reference kernel computes:

```text
for each group g = [start, end):
    acc = sum_{k in [start, end)} x[k] * (code[row, k] - zero_point[row, g])   // f32
    y[row] += acc * scale[row, g]                                               // f32
y[row] += bias[row]                                                             // f32, if present
```

The accumulation inside a group is f32 and proceeds in ascending `k`. The
per-group affine is applied once per group, not once per element. This is the
`Σ x·(q−z)` form; it avoids a separate correction term and keeps
`(q−z)` exact. An implementation may instead compute
`Σ x·q − z·Σx` (algebraically identical) or use integer dot products over
quantized activations, but such a path is a different precision contract and
must still meet the accuracy gate in the decision record. v1 itself has no
quantized activations.

No Q/DQ op is inserted in the graph:
the weight stays packed, the activation stays FP32, and the dequantization is
the per-group multiply inside the MatMul kernel. On Vulkan the packed codes,
scales and zero-points are device-resident from prepare (see §8).

## 6. Mixed-precision coverage

Quantized in v1:

| Graph op | Weight | Activation | Bias |
| --- | --- | --- | --- |
| GPT-2 `c_attn` (QKV projection) | INT8 v1 | FP32 | FP32 |
| GPT-2 `attn.c_proj` | INT8 v1 | FP32 | FP32 |
| GPT-2 `mlp.c_fc` | INT8 v1 | FP32 | FP32 |
| GPT-2 `mlp.c_proj` | INT8 v1 | FP32 | FP32 |
| `lm_head` (tied `wte` copy) | INT8 v1 | FP32 | none |

Stays FP32:

* token and position embeddings (`Gather`), including the token-embedding table
  that `lm_head` shares (prepare keeps the FP32 table and writes a separate
  packed copy for the head);
* LayerNorm / RMSNorm, GELU (tanh and erf), softmax, residual adds, attention
  score and value MatMuls, RoPE/position arithmetic, the final norm;
* every bias;
* any MatMul whose weight is not a 2-D F32 constant at prepare time, whose RHS
  is dynamic, or whose `K < 64` / `numel < 4096` (metadata would dominate);
* a weight shared with a non-MatMul consumer when a packed copy is not
  permitted. This is a per-tensor fallback, not a silent graph-wide one; the
  prepared artifact must report which projections were left in FP32.

## 7. Failure contract

The reader is total over `&[u8]`: it never panics, never reads out of bounds,
and returns a typed [`QuantError`] with a stable `kind()` string. Checks run in
this order:

1. header presence, magic, version, header size, flags, tensor count;
2. descriptor/group/payload offsets and file length;
3. metadata CRC, payload CRC;
4. descriptor names and uniqueness;
5. extents, `qbits`, group size, group count/index, row stride;
6. 16-byte alignment and canonical data offset;
7. payload bounds, inter-tensor padding, group contiguity/length/coverage;
8. row padding, every scale, every zero-point;
9. unused group entries and the final payload cursor.

The committed matrix `fixtures/quant/malformed/` has one blob per reachable
failure, including: bad magic/version/header size/flags/count, descriptor and
group offsets, payload offset, qbits, group size, group count/index, row stride,
misaligned and non-canonical data offset, empty/duplicate names, zero extents,
group overlap and gap, group length, truncated/trailing file, payload and
metadata checksums, zero/NaN/negative scale, out-of-range zero-point, non-zero
row padding and non-zero inter-tensor padding. `tests/fn3_quant_format.rs`
asserts the exact `kind()` for every one.

## 8. Coordination with #72 (Vulkan residency)

The quantized weights are **device-resident from prepare**. Prepare packs the
weights on the host, uploads the `fn3-quant-format-v1` payload (codes, scales,
zero-points) once with the prepared-model artifact, and the resident session
binds those buffers directly. There is no per-op upload, no per-token staging of
weights and no host round-trip during decode.

Dequantization happens **inside the GEMM kernel**:

* the packed INT8 codes remain in device memory;
* the activation tile is FP32 and stays FP32 (v1 is W8A16);
* the kernel accumulates `x·(q−z)` per 64-element group in FP32, multiplies the
  group partial by the binary16 scale (widened on device), and accumulates the
  groups in FP32;
* a dequantized weight buffer is never written. This is the integer-storage /
  FP32-accumulate choice: the code and zero-point stay integer for the whole
  dot, and the only floating-point work is the per-group affine. It avoids the
  bandwidth of an FP32 weight copy and keeps the #72 resident path free of
  Q/DQ nodes.

If a future backend cannot fuse the affine (for example a vendor int8 GEMM with
a scale-only epilogue), it must dequantize into a tile-sized scratch buffer, not
a whole-tensor FP32 copy, and must account for that buffer in its residency
budget. That is a backend implementation detail; the on-disk format and the
accuracy contract do not change.

## 9. Versioning

* `format_version` and `FORMAT_ID` change together for any byte-layout change.
* Adding a qbits value, changing the group size, changing the scale dtype, or
  changing the payload order is a new format.
* Adding a new metadata field inside the existing reserved/`qbits` space without
  changing the layout is not possible in v1: `flags` is reserved and must be 0.
* The v1 reader accepts exactly one representation of a logical object. A future
  reader may read v1, but a v1 reader must not accept v2.
