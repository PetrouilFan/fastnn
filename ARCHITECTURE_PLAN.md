# FastNN Architecture Plan: I4x8, I8x4, F4x8, F8x4, F8x4R

## Overview

Add three new FP packed types (F4x8, F8x4, F8x4R) alongside renaming U4→I4 and U8→I8. All types pack multiple values into a single u32 word for memory efficiency and direct computation via SWAR (SIMD Within A Register).

## Type Specifications

### Storage Efficiency Table

| Type | Items/u32 | Bits/element | Range | Memory vs F32 |
|------|-----------|-------------|-------|--------------|
| F32 | 1 | 32 | ±3.4e38 | 1× |
| F16 | 2 | 16 | ±65504 | 2× |
| I8x4 | 4 | 8 | [-128, 127] | 4× |
| **F8x4** | 4 | 8 | ±448 (E4M3) | 4× |
| **F8x4R** | 4 | 8 | ±57344 (E5M2) | 4× |
| I4x8 | 8 | 4 | [-8, 7] | 8× |
| **F4x8** | 8 | 4 | ±6 (E2M1) | 8× |

### I4x8 (renamed from I4x8)
- **8 × signed 4-bit integers per u32**
- Range: [-8, 7], two's complement
- Dot product: `u4x8_dot_packed` — extract 8 nibbles, sign-extend, multiply-accumulate in i32
- Storage: `(sign << 3) | magnitude` per nibble, where magnitude is 3-bit unsigned [0..7]

### I8x4 (renamed from I8x4)  
- **4 × signed 8-bit integers per u32**
- Range: [-128, 127], two's complement
- Dot product: `u8x4_dot_packed` — extract 4 bytes, i8 sign-extend, multiply-accumulate
- Storage: native byte order, each byte is a signed i8

### F8x4 (NEW — FP8 E4M3 format)
- **4 × FP8 E4M3 values per u32**
- Bit layout per byte: `S EEEE MMM` (1 sign, 4 exponent bias=7, 3 mantissa)
- Range: ±448, max normal 448.0, min normal 0.015625, min subnormal ~0.001953125
- NaN: 0x7F or 0xFF (all exponent 1s, mantissa != 0)
- Inf: NOT representable in E4M3 (unlike E5M2)
- Dot product: LNS approximate
  - `product = (int_encode(a) + int_encode(b) - 0x38)` where 0x38 = (7 << 3)
  - This gives an approximate FP8 product in FP8 bit-pattern form
  - Accumulate: extract each byte, decode to f32, accumulate in f32
- Dequant: 2-term (no zero-point): `result = f32_val * scale`
- Constant `B = bias << mantissa_bits = 7 << 3 = 56 = 0x38`
- LNS rewrite formula: `int_encode(a) + int_encode(b) - B + carry_in_correction`

### F8x4R (NEW — FP8 E5M2 format, for gradients/training)
- **4 × FP8 E5M2 values per u32**
- Bit layout per byte: `S EEEEE MM` (1 sign, 5 exponent bias=15, 2 mantissa)
- Range: ±57344, max normal 57344.0, min normal 0.000030517578125
- Inf: 0x7C or 0xFC (all exponent 1s, mantissa = 0)
- NaN: 0x7E/0x7F or 0xFE/0xFF
- Same computational pattern as F8x4 but with constant `B = 15 << 2 = 60 = 0x3C`

### F4x8 (NEW — FP4 E2M1 format, NVFP4-style)
- **8 × FP4 E2M1 values per u32**
- Bit layout per nibble: `S EE M` (1 sign, 2 exponent bias=1, 1 mantissa)
- Range: ±6.0
- 16 representable values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} × {±1}
- NaN/Inf: none (0xF = -6.0, the most negative value)
- **NO SWAR integer multiply possible** — K=2 encoding needs 5 bits (4 magnitude + 1 sign), but only 4 bits available
- Dot product: 256-entry i16 LUT (512 bytes in L1)
  - `LUT[a][b] = round(4.0 * real_val(a) * real_val(b))`
  - Accumulate: for each nibble pair, `sum += LUT[nib_a][nib_b]` as i32
  - Output: `result = sum as f32 * scale_a * scale_b / 4.0`
- Block scaling: NVFP4-style, block_size=16, FP8 E4M3 per-block scale
  - Each 16-element block has 1 E8M0 exponent (1 byte)
  - 8 bytes FP4 data + 1 byte scale = 9 bytes per 16 elements = 4.5 bpe
  - Block scales stored in PackedTensor::block_scales: Option<Vec<u8>>

## GEMM Dequantization Formulas

### I4x8 / I8x4 (4-term, with zero-point)
```
result = Σ(qA * sA + zpA) × (qB * sB + zpB)
       = acc * sA * sB 
       + zpB * sA * ΣqA 
       + zpA * sB * ΣqB 
       + zpA * zpB * K
```
Precompute `qa_sum = ΣqA` and `qb_sum = ΣqB` per row for zero-point terms.

### F8x4 / F8x4R (2-term, no zero-point)
```
result = Σ(F8_val_A * sA) × (F8_val_B * sB)
       = acc * sA * sB
       where acc = Σ decode_to_f32(F8 A[i] × F8 B[i])
```
LNS approximate per-element product via: `int_a + int_b - B` where B depends on format.
But each LNS "product" is still an FP8 pattern — must decode to f32 and accumulate.

### F4x8 (LUT-based with block scaling)
```
for each K-block of 16 elements:
    block_i32_sum = Σ LUT[nib_a[i]][nib_b[i]]
    result += block_i32_sum * block_scale_A * block_scale_B
output = result as f32 / 4.0 + bias
```
The `/4.0` corrects for the LUT's scaling factor (the LUT stores round(4.0 * real(a) * real(b))).

## SWAR Operations

### Existing (rename only) — `src/swar/ops_i8.rs`
- I8x4: swar_add_u8x4, swar_sub_u8x4, swar_relu_s8x4, swar_max_u8x4, swar_min_u8x4
- These work at byte granularity in a u32

### NEW — `src/swar/ops_i4.rs`
I4x8 operations use byte-expansion (expand nibbles to bytes, operate, repack):
```
fn i4x8_add(a: u32, b: u32) -> u32:
    lo = expand_low_nibbles(a) + expand_low_nibbles(b)   // byte-level SWAR add
    hi = expand_high_nibbles(a) + expand_high_nibbles(b)
    pack_nibbles(lo, hi)
```

**expand_low_nibbles**: `(packed & 0x0F0F0F0F)` — gives 4 bytes, each = 0x0N
**expand_high_nibbles**: `(packed >> 4) & 0x0F0F0F0F` — gives 4 bytes, each = 0x0N
**pack_nibbles**: `lo | (hi << 4)` — interleave low/high nibbles into same word

Functions:
- `i4x8_add(a, b) -> u32`
- `i4x8_sub(a, b) -> u32`
- `i4x8_relu(a) -> u32` — zero out nibbles with sign=1 (i.e., `packed & 0x77777777` AFTER byte-expansion)
- `i4x8_max(a, b) -> u32` — expand to bytes, use I8x4 max on absolute values, then apply sign
- `i4x8_min(a, b) -> u32`

### NEW — `src/swar/ops_fp.rs`
FP8 total-order transform for comparison (max/min/relu):

```
fn fp_total_order(packed: u32, sign_mask: u32) -> u32:
    // For IEEE sign-magnitude floats, total-order transform makes them 
    // comparable as unsigned integers.
    // sign: if 1 (negative), flip all magnitude bits
    //       if 0 (positive), flip the sign bit
    sign = packed & sign_mask
    // sign >> bit_pos gives 0x01 per lane if negative
    fill = (sign >> sign_mask.trailing_zeros()).wrapping_mul(0xFF)
    // fill = 0xFF per lane if negative, 0x00 if positive
    // XOR mask: neg -> flip all bits (0xFF), pos -> flip only sign bit
    xor_mask = fill ^ sign_mask  
    packed ^ xor_mask

// For F8x4 (4 bytes per u32):
fn f8x4_total_order(packed: u32) -> u32:
    fp_total_order(packed, 0x80808080)

// For F4x8 (need byte-expansion for comparison):
// Expand to bytes, apply fp_total_order at byte level, repack
fn f4x8_total_order(packed: u32) -> u32:
    lo = packed & 0x0F0F0F0F
    hi = (packed >> 4) & 0x0F0F0F0F
    lo_ord = fp_total_order(lo, 0x08080808)  // bit 3 = sign in each low nibble byte
    hi_ord = fp_total_order(hi, 0x08080808)
    (lo_ord & 0x0F0F0F0F) | ((hi_ord & 0x0F0F0F0F) << 4)
```

**Inverse transform** is the same operation (total_order is self-inverse):
```
fn fp_inverse_total_order(ordered: u32, sign_mask: u32) -> u32:
    fp_total_order(ordered, sign_mask)
```

**Max/Min using total-order**:
```
fn fp_max(a_ord: u32, b_ord: u32) -> u32:
    // Standard SWAR unsigned max on total-order values
    lt = a_ord.wrapping_sub(b_ord) & 0x80808080  // (or 0x88888888 for nibbles)
    mask = (lt >> lt.trailing_zeros()).wrapping_mul(0xFF)
    result_ord = a_ord ^ ((a_ord ^ b_ord) & mask)
    fp_inverse_total_order(result_ord, sign_mask)
```

**Relu for F8x4**: clear sign bits of negative values
```
fn f8x4_relu(packed: u32) -> u32:
    packed & !((packed & 0x80808080) >> 7).wrapping_mul(0xFF)
```

## File Changes

### Phase 1: Rename U→I (mechanical, no functional changes)

#### src/dtypes/
| File | Change |
|------|--------|
| `mod.rs` | Rename modules: `pub mod u4x8`→`pub mod i4x8`, `pub mod u8x4`→`pub mod i8x4`; rename re-exports: `I4x8`→`I4x8`, `I8x4`→`I8x4` |
| `u4x8.rs` | Rename file to `i4x8.rs`; struct `I4x8`→`I4x8`; rename tests |
| `u8x4.rs` | Rename file to `i8x4.rs`; struct `I8x4`→`I8x4`; rename tests |

#### src/ir/node.rs
- `IrDType::U4` → `IrDType::I4` (keep variant structure: `I4 { scales, zero_points }`)
- `IrDType::U8` → `IrDType::I8` (KEEP the name `I8` already exists for activation INT8!  
  **CRITICAL**: The existing `IrDType::I8` is for INT8 activations. The renamed U8 must become a different variant.  
  **Solution**: `I8 { scales, zero_points }` for quantized weights AND rename what was U8 weight quantization.  
  But `I8` already exists as a variant (line 536). We can't have two variants named `I8`.
  
  **Decision**: The existing `IrDType::I8` (activation int8) stays as-is. The quantized weight type that was `U8` becomes a variant that can store scales/zps. But `I8` already has no fields.
  
  **Better approach**: Add scales/zps to the existing `I8` variant. Currently `I8` is fieldless (for activations). Make it `I8 { scales: Vec<f32>, zero_points: Vec<f32> }` and use `I8 { .. }` with empty vecs for activations. This unifies the type.
  
  Wait, this is semantically wrong because the existing `I8` means INT8 activations (not quantized weights from U8).
  
  **Correct approach**: The existing `I8` activation type is distinct from weight quantization. Create a new variant:
  - Rename `U4`→`I4`
  - Rename `U8`→`I8_W` (or more consistently, rename to `I4` and `I8` for weights, keep `I8` for activations but they're the same type with/without scales)

  **Simplest**: 
  - `IrDType::U4 { scales, zps }` → `IrDType::I4 { scales, zps }`
  - `IrDType::U8 { scales, zps }` → `IrDType::I8 { scales, zps }` — but this conflicts with existing fieldless `I8`.
  
  **Resolution**: Remove the fieldless `I8` variant. Replace it with `I8 { scales: Vec<f32>, zero_points: Vec<f32> }`. For activations, use `I8 { scales: vec![], zero_points: vec![] }` (empty = no per-channel metadata). For quantized weights, the scales/zps are populated.
  
  This unifies the type system and simplifies matching. All existing `IrDType::I8` patterns need to change to `IrDType::I8 { .. }`.
  
  Update `impl_ir_dtype_props!`: Change `(Self::I8, 1, "i8", 8, 4)` to `(Self::I8 { .. }, 1, "i8", 8, 4)`.

- In `packed_byte_size()`, rename cases:
  - `IrDType::U4 { .. }` → `IrDType::I4 { .. }`
  - `IrDType::U8 { .. }` → `IrDType::I8 { .. }` (update the second I8 case, keep existing I8 as INT8 activation)
  
  Actually, we need to distinguish: the U8→I8 rename conflicts with the existing I8 activation type. They represent different things.
  
  **Best solution**: Keep the INT8 activation variant as `I8` (no fields), and create `I8Q { scales, zps }` for quantized I8 weights. Or more cleanly: the INT8 activation variant was already called `I8` with no fields. The weight quantization type was called `U8` with fields. If we rename `U8`→`I8`, it collides.
  
  **Therefore**: The cleanest rename is:
  - `U4` → `I4` (no collision)
  - `U8` **stays as `U8`** for now, or becomes `I8W`/`Q8` — but the user explicitly wants U→I rename.

  **Final decision**: 
  - Remove the fieldless `I8` activation variant
  - Replace with `I8 { scales: Vec<f32>, zero_points: Vec<f32> }` serving both roles
  - For activations: scales/zps are empty vecs
  - For weights: scales/zps are populated
  - Update all match patterns

#### `impl_ir_dtype_props!` in node.rs
Change the I8 line from `(Self::I8, 1, "i8", 8, 4)` to `(Self::I8 { .. }, 1, "i8", 8, 4)`.  
Change U4/U8 lines accordingly.

#### `packed_byte_size()` in node.rs
- `IrDType::U4 { .. }` → `IrDType::I4 { .. }`  
- `IrDType::U8 { .. }` → `IrDType::I8 { .. }` (merge with existing I8 case)
- Handle: original I8 and renamed U8 are now the same variant. The activation I8 was computed as `numel + 8` (with header). The quantized I8 weight uses `ceil(inner / 4) * rows + 16` SIMD margin.  
  Need a way to distinguish: use whether scales vector is empty (activation, one uint8 per element) or non-empty (packed I8 weights, 4 per word).

#### src/backend/cpu/mod.rs (7042 lines)
- Kernel dispatch strings:  
  `"matmul_u4"`→`"matmul_i4"`  
  `"matmul_u4_i8"`→`"matmul_i4_i8"`  
  `"matmul_u8"`→`"matmul_i8"`  
  `"matmul_u8_i8"`→`"matmul_i8_i8"`  
  `"conv2d_u4"`→`"conv2d_i4"`  
  `"conv2d_u8"`→`"conv2d_i8"`  
  etc. (all fusion variants)
- IrDType matches: `IrDType::U4 { .. }`→`IrDType::I4 { .. }`, `IrDType::U8 { .. }`→`IrDType::I8 { .. }`
- Helper function `aligned_packed_slice` references: rename I4x8→I4x8, I8x4→I8x4

#### src/backend/cpu/matmul.rs
- `quantized_matmul_dispatch::<I4x8>` → `quantized_matmul_dispatch::<I4x8>`
- `quantized_matmul_dispatch::<I8x4>` → `quantized_matmul_dispatch::<I8x4>`
- `quantized_matmul_dispatch_i8_u4` → `quantized_matmul_dispatch_i8_i4`
- `quantized_matmul_dispatch_i8_u8` → `quantized_matmul_dispatch_i8_i8`

#### src/backend/cpu/swar.rs
- Functions: `u8x4_dot_packed` → `i8x4_dot_packed`, `u4x8_dot_packed` → `i4x8_dot_packed`
- `sum_u8x4_packed` → `sum_i8x4_packed`, `sum_u4x8_packed` → `sum_i4x8_packed`
- `u8x4_dot_packed_slice` → `i8x4_dot_packed_slice`
- `u4x8_dot_packed_slice` → `i4x8_dot_packed_slice`
- Quantize/dequantize: `quantize_f32_to_u8x4` → `quantize_f32_to_i8x4`, etc.
- Dtype imports: `I4x8`→`I4x8`, `I8x4`→`I8x4`

#### src/backend/cpu/packed_gemm.rs
- Imports: `I4x8`→`I4x8`, `I8x4`→`I8x4`, `u8x4_dot_packed`→`i8x4_dot_packed`
- Function names: `gemm_packed_u8x4`→`gemm_packed_i8x4`, `gemm_packed_u4x8`→`gemm_packed_i4x8`
- `gemm_packed_u8x4_fused`→`gemm_packed_i8x4_fused`
- `quantize_activations_to_u8x4`→`quantize_activations_to_i8x4`
- `quantize_activations_to_u4x8`→`quantize_activations_to_i4x8`
- `u8x4_packed_to_tensor`→`i8x4_packed_to_tensor`
- `u4x8_packed_to_tensor`→`i4x8_packed_to_tensor`

#### src/backend/cpu/packed_conv.rs
- Imports: I4x8→I4x8, I8x4→I8x4
- All references to conv2d_packed_u4x8/u8x4 → conv2d_packed_i4x8/i8x4
- `im2col_pack_u4x8/i8x4` → i4x8/i8x4 versions
- `gemm_packed_u8x4_fused_raw` → `gemm_packed_i8x4_fused_raw`
- `gemm_packed_u4x8_fused_raw` → `gemm_packed_i4x8_fused_raw`
- `pack_i8_col_to_u8x4` → `pack_i8_col_to_i8x4`
- `pack_i8_col_to_u4x8` → `pack_i8_col_to_i4x8`

#### src/backend/cpu/microkernels/gemm.rs
- `gemm_cpu_flat_i8_u8x4` → `gemm_cpu_flat_i8_i8x4`
- `gemm_cpu_flat_i8_u4x8` → `gemm_cpu_flat_i8_i4x8`
- Imports: I4x8→I4x8, I8x4→I8x4

#### src/backend/prepared.rs
- `PackedWeightKind::U4 = 2` → `PackedWeightKind::I4 = 2`
- Update `name()`: `"u4"`→`"i4"`
- `kernel_kind_from_kernel_name`: `"conv2d_u4"`→`"conv2d_i4"`, `"conv2d_u8"`→`"conv2d_i8"`
- `is_matmul_kernel_name`: `"matmul_u4"`→`"matmul_i4"`, `"matmul_u8"`→`"matmul_i8"`

#### src/backend/wgpu/quantized.rs
- All references to I4x8→I4x8, I8x4→I8x4

#### src/swar/ops_8bit.rs → ops_i8.rs
- Rename file. Function names: `swar_add_u8x4`→`swar_add_i8x4`, etc.
- Update constants comments.

#### src/swar/mod.rs
- `pub mod ops_8bit` → `pub mod ops_i8`

#### src/packed_tensor.rs
- All references to I4x8→I4x8, I8x4→I8x4

#### src/storage_quantized.rs
- All references to I4x8→I4x8, I8x4→I8x4

#### src/storage.rs
- Any DType::U4/U8 string parsing → I4/I8

#### src/compiler/passes/
- `quantization.rs`, `activation_quantization.rs`, `quantize_activations.rs`, `calibration.rs`, `auto_cast.rs`, `prune_qdq_pairs.rs`
- All IrDType matches: U4→I4, U8→I8

#### src/lib.rs
- Update crate docs

#### fastnn/precision.py
```python
class Precision(IntEnum):
    F32 = 0
    F16 = 1
    I8 = 2   # renamed from U8
    I4 = 3   # renamed from U4
```
Update `is_quantized`, `from_string`, `from_dtype_tag`.

#### fastnn/io/*.py
- Update all U4/U8 references to I4/I8

#### tests/*.rs
- Update all references

#### benches/*.rs
- Update all references

#### src/onnx/
- Update export references

### Phase 2: Add F8x4 and F8x4R

#### src/dtypes/f8x4.rs (NEW)
```rust
/// 4 × FP8 E4M3 values packed per u32 word.
/// Range: ±448, bias=7, 3-bit mantissa.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct F8x4(pub u32);

unsafe impl bytemuck::Pod for F8x4 {}
unsafe impl bytemuck::Zeroable for F8x4 {}

impl PackedWord for F8x4 {
    const ITEMS: usize = 4;
    const BIT_WIDTH: usize = 8;
    const IS_FLOAT: bool = true;
    type Array = [f32; 4];

    fn unpack_to_f32(self) -> [f32; 4] {
        let bytes = self.0.to_le_bytes();
        [
            decode_fp8_e4m3(bytes[0]),
            decode_fp8_e4m3(bytes[1]),
            decode_fp8_e4m3(bytes[2]),
            decode_fp8_e4m3(bytes[3]),
        ]
    }

    fn pack_from_f32(vals: [f32; 4]) -> Self {
        let mut word: u32 = 0;
        for i in 0..4 {
            word |= (encode_fp8_e4m3(vals[i]) as u32) << (i * 8);
        }
        F8x4(word)
    }

    fn wgsl_unpack_body() -> &'static str { ... }
    fn wgsl_return_type() -> &'static str { "vec4<f32>" }
}
```

**`decode_fp8_e4m3(byte: u8) -> f32` implementation:**
```rust
#[inline]
fn decode_fp8_e4m3(byte: u8) -> f32 {
    if byte & 0x7F == 0 {
        return 0.0;  // zero (+0 or -0)
    }
    if byte & 0x7F == 0x7F {
        return f32::NAN;  // NaN
    }
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0xF) as i32;  // 4-bit exponent
    let mant = (byte & 0x7) as i32;         // 3-bit mantissa
    if exp == 0 {
        // Subnormal: 0.mantissa * 2^(1-7)
        sign * (mant as f32) * 2.0f32.powi(1 - 7)  // *= 2^(-6)
    } else {
        // Normal: 1.mantissa * 2^(exp-7)
        let significand = 1.0 + (mant as f32) / 8.0;
        sign * significand * 2.0f32.powi(exp - 7)
    }
}
```

**`encode_fp8_e4m3(val: f32) -> u8` implementation:**
```rust
#[inline]
fn encode_fp8_e4m3(val: f32) -> u8 {
    let mut bits = val.to_bits();
    let sign = (bits >> 24) & 0x80;
    let abs_val = val.abs();
    
    if abs_val == 0.0 {
        return sign as u8;
    }
    if abs_val.is_nan() {
        return 0x7F | sign as u8;  // NaN with sign
    }
    if abs_val > 448.0 {
        return 0x7E | sign as u8;  // max normal (sat)
    }
    
    // FP32: S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM (8 exp + 23 mant)
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;  // unbias
    let mant = bits & 0x7FFFFF;
    
    // FP8: S EEEE MMM (4 exp + 3 mant)
    // Clamp exponent to FP8 range
    let fp8_exp = exp.max(-6).min(7);  // bias=7 → exp range [-6, 7]
    // The exponent field in FP8 is fp8_exp + 7 (rebias)
    let exp_field = (fp8_exp + 7) as u32;
    
    if fp8_exp <= -7 {
        // Subnormal: encode as 0.mantissa * 2^(-6)
        // Shift mantissa to fit 3 bits
        let shift = (-6 - fp8_exp) as u32;
        // The mantissa value is 0.(mantissa bits)
        // For subnormals, we need to extract the leading bits
        // Similar to nearest rounding
        let mant_field = if shift < 24 {
            ((mant >> (23 - shift)) as u32) & 0x7
        } else { 0 };
        return (sign as u8) | (exp_field as u8) | (mant_field as u8);
    } else {
        // Normal: 1.mantissa * 2^(exp)
        // Round mantissa from 23 bits to 3 bits
        let mant_field = ((mant >> 20) as u32) & 0x7;  // top 3 bits
        let round_bit = (mant >> 19) & 1;
        let mant_field = mant_field + round_bit;
        let (mant_field, carry) = if mant_field >= 8 { (0, 1) } else { (mant_field, 0) };
        let exp_field = exp_field + carry;
        return (sign as u8) | ((exp_field as u32) as u8) << 3 | (mant_field as u8);
    }
}
```

#### src/dtypes/f8x4r.rs (NEW)
Same structure as f8x4.rs but with E5M2:
- `decode_fp8_e5m2`: exp bias=15, mantissa bits=2
- `encode_fp8_e5m2`: exp range [-14, 15], mantissa 2 bits, saturate at ±57344
- NaN: 0x7E, 0x7F, 0xFE, 0xFF (exp=0x1F, mant != 0)
- Inf: 0x7C, 0xFC (exp=0x1F, mant = 0)

#### src/dtypes/mod.rs
```rust
pub mod f8x4;
pub mod f8x4r;
pub use f8x4::F8x4;
pub use f8x4r::F8x4R;
```

#### src/ir/node.rs
Add IrDType variants:
```rust
enum IrDType {
    // ... existing variants plus:
    
    I4 { scales: Vec<f32>, zero_points: Vec<f32> },
    I8 { scales: Vec<f32>, zero_points: Vec<f32> },
    
    // NEW:
    F8 { scales: Vec<f32> },                       // E4M3, no zero_point
    F8R { scales: Vec<f32> },                      // E5M2, no zero_point
    F4 { scales: Vec<f32>, block_scales: Vec<u8>, block_size: usize },
}
```

In `impl_ir_dtype_props!`:
```rust
(Self::I4 { .. }, 1, "i4", 4, 8),
(Self::I8 { .. }, 1, "i8", 8, 4),  // merges old I8 and U8
(Self::F8 { .. }, 1, "f8", 8, 4),
(Self::F8R { .. }, 1, "f8r", 8, 4),
(Self::F4 { .. }, 1, "f4", 4, 8),
```

In `packed_byte_size()`:
```rust
IrDType::F8 { .. } | IrDType::F8R { .. } => {
    let words = numel.div_ceil(4) + 16;
    words * 4
}
IrDType::F4 { .. } => {
    // F4 has block scales appended: +ceil(numel / block_size) bytes for E8M0 scales
    let words = numel.div_ceil(8) + 16;
    words * 4 + numel.div_ceil(16)  // +1 byte per 16 elements for block scales
}
```

In `byte_size_with_env()`:
```rust
IrDType::F8 { .. } | IrDType::F8R { .. } => {
    // Same as U8 but without zero-point header
    let inner_dim = ...; // last shape dim or 4
    let rows = ...;
    let words = rows * inner_dim.div_ceil(4) + 16;
    words * 4
}
IrDType::F4 { .. } => {
    let inner_dim = ...; // last shape dim or 8
    let rows = ...;
    let words = rows * inner_dim.div_ceil(8) + 16;
    words * 4 + numel.div_ceil(16)  // block scales
}
```

#### src/packed_tensor.rs
Add to PackedTensor struct:
```rust
pub struct PackedTensor<T: PackedWord> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) scales: Vec<f32>,
    pub(crate) zeros: Vec<f32>,
    pub(crate) block_size: usize,
    pub(crate) group_size: usize,
    // NEW: For F4x8 block scaling
    pub(crate) block_scales: Option<Vec<u8>>,  // E8M0 per block
}
```

Methods to add:
```rust
impl<T: PackedWord> PackedTensor<T> {
    /// Create from F8 data (no zero-point, 2-term dequant)
    pub fn from_f32_f8(data: &[f32], shape: &[usize]) -> Self { ... }
    
    /// Create F4x8 with NVFP4 block scaling
    pub fn from_f32_f4_blocked(
        data: &[f32], shape: &[usize], block_size: usize
    ) -> Self { ... }
}
```

#### src/backend/cpu/mod.rs
Add kernel dispatch for F8/F4:
```rust
// Quantized: F8 activation + F8 weight
(_, true)
    if input_dtypes.first().is_some_and(|d| matches!(d, IrDType::F8 { .. }))
        && input_dtypes.iter().any(|d| matches!(d, IrDType::F8 { .. })) =>
{
    "matmul_f8"
}
// ... similar for F4
```

In the dispatch match:
```rust
"matmul_f8" => {
    quantized_matmul_dispatch_fp::<F8x4>(
        input_slices, arena, params, param_dims, shape_env,
        weight_meta, out_start, out_end, 8, "matmul_f8",
    )?;
}
"matmul_f4" => {
    quantized_matmul_dispatch_f4::<F4x8>(
        input_slices, arena, params, param_dims, shape_env,
        weight_meta, out_start, out_end, 4, "matmul_f4",
    )?;
}
```

The `matmul_f8` kernel uses `packed_gemm_fp.rs` with 2-term dequant.
The `matmul_f4` kernel uses LUT-based approach with block scale factoring.

#### src/backend/cpu/packed_gemm_fp.rs (NEW)

**F8x4 GEMM (2-term dequant)**:
```rust
/// F8x4 GEMM: C = A × Bᵀ (2-term dequant, no zero-point)
pub fn gemm_packed_f8x4(
    a_packed: &PackedTensor<F8x4>,
    b_packed: &PackedTensor<F8x4>,
    c: &mut [f32],
) {
    let m = a_packed.shape()[0];
    let k = a_packed.shape()[1];
    let n = b_packed.shape()[0];
    let k_packed = k.div_ceil(4);
    
    let a_data = a_packed.as_packed();
    let b_data = b_packed.as_packed();
    
    for row in 0..m {
        let a_row = &a_data[row * k_packed..(row + 1) * k_packed];
        let a_scale = a_packed.scale_for_row(row);
        
        for col in 0..n {
            let b_row = &b_data[col * k_packed..(col + 1) * k_packed];
            let b_scale = b_packed.scale_for_row(col);
            
            let mut acc = 0.0f32;
            for kk in 0..k_packed {
                // Decode both packed words to f32, multiply elementwise, accumulate
                let a_f32 = a_row[kk].unpack_to_f32();
                let b_f32 = b_row[kk].unpack_to_f32();
                for lane in 0..4 {
                    acc += a_f32[lane] * b_f32[lane];
                }
            }
            
            // 2-term dequant: result = acc * scale_A * scale_B
            c[row * n + col] = acc * a_scale * b_scale;
        }
    }
}
```

**F8x4 GEMM with LNS approximate multiply** (faster, approximate):
```rust
/// F8x4 GEMM using LNS approximate multiply
/// Each element: approx_product = byte_a + byte_b - 0x38
/// This is approximate (faithful rounding for E4M3 with carry-in correction)
pub fn gemm_packed_f8x4_lns(
    a_packed: &PackedTensor<F8x4>,
    b_packed: &PackedTensor<F8x4>,
    c: &mut [f32],
) {
    const B: u32 = 0x38; // (7 << 3) — E4M3 bias
    // (B replicated 4 times in u32: 0x38383838)
    const B_REP: u32 = 0x38383838;
    
    // ... same outer loops ...
    for kk in 0..k_packed {
        // LNS approximate multiply: add byte patterns, subtract B
        let prod_approx = a_row[kk].0.wrapping_add(b_row[kk].0).wrapping_sub(B_REP);
        // Decode LNS products from packed bytes to f32
        let bytes = prod_approx.to_le_bytes();
        for lane in 0..4 {
            acc += decode_fp8_e4m3(bytes[lane]);
        }
    }
    // ... 2-term dequant ...
}
```

**F4x8 GEMM (LUT-based with block scaling)**:
```rust
/// F4x8 GEMM with NVFP4 block scaling
/// Block size = 16 elements
/// Each block has 1 E8M0 scale byte
pub fn gemm_packed_f4x8(
    a_packed: &PackedTensor<F4x8>,
    b_packed: &PackedTensor<F4x8>,
    c: &mut [f32],
) {
    let m = a_packed.shape()[0];
    let k = a_packed.shape()[1];
    let n = b_packed.shape()[0];
    let block_size = 16;  // NVFP4 block size
    let k_packed = k.div_ceil(8);
    let num_blocks = k.div_ceil(block_size);
    let words_per_block = block_size / 8;  // 2 packed words per block
    
    let a_data = a_packed.as_packed();
    let b_data = b_packed.as_packed();
    let a_block_scales = a_packed.block_scales.as_ref().unwrap();
    let b_block_scales = b_packed.block_scales.as_ref().unwrap();
    
    for row in 0..m {
        for col in 0..n {
            let mut acc_i32: i32 = 0;
            
            for blk in 0..num_blocks {
                let blk_start = blk * words_per_block;
                let a_blk = &a_data[row * k_packed + blk_start..];
                let b_blk = &b_data[col * k_packed + blk_start..];
                
                // Compute block dot product via LUT
                let mut block_sum: i32 = 0;
                for w in 0..words_per_block {
                    block_sum += f4x8_dot_lut(a_blk[w].0, b_blk[w].0);
                }
                
                // Apply block scales
                let a_scale = decode_e8m0(a_block_scales[row * num_blocks + blk]);
                let b_scale = decode_e8m0(b_block_scales[col * num_blocks + blk]);
                acc_i32 += (block_sum as f32 * a_scale * b_scale) as i32;
            }
            
            // Final dequant: divide by 4 (from LUT scaling) + per-channel scale
            c[row * n + col] = (acc_i32 as f32) / 4.0;
        }
    }
}
```

**`decode_e8m0(byte: u8) -> f32`**: E8M0 is a pure-exponent format (OCP MX).  
Value = `2^(byte - bias)` where bias is typically 0 or related to the block.
```rust
#[inline]
fn decode_e8m0(byte: u8) -> f32 {
    if byte == 0 { return 0.0; }
    f32::from_bits(((byte as u32) << 23) - (127u32 << 23))  // 2^(byte-127)
    // Simpler: 2.0_f32.powi((byte as i32) - 127)
}
```

Actually for NVFP4, the block scale is stored as FP8 E4M3, not E8M0.
```rust
#[inline]
fn decode_nvfp4_scale(byte: u8) -> f32 {
    decode_fp8_e4m3(byte)
}
```

#### src/backend/cpu/matmul.rs — New dispatch functions
```rust
/// F8x4 MatMul dispatch (FP32 activations × F8 weights)
pub(super) fn quantized_matmul_dispatch_fp<T: PackedWord>(
    input_slices: &[BufferSlice],
    arena: &CpuBuffer,
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    weight_meta: &Option<Arc<QuantizedWeightMeta>>,
    out_start: usize,
    out_end: usize,
    bit_width: usize,
    kernel_name: &str,
) -> Result<(), BackendError> {
    // Similar to quantized_matmul_dispatch but:
    // - No zero_point (zeros become empty or default)
    // - Calls gemm_packed_f8x4 instead of gemm_packed_i8x4
    // - Activations stay as f32 (not packed)
}

/// F4x8 MatMul dispatch with block scaling
pub(super) fn quantized_matmul_dispatch_f4<T: PackedWord>(
    // ... same signature ...
) -> Result<(), BackendError> {
    // Extracts block_scales from PackedTensor
    // Calls gemm_packed_f4x8
}
```

### Phase 3: SWAR ops for all types

#### src/swar/ops_i4.rs (NEW)
I4x8 SWAR operations using byte-expansion:
```rust
const I4_SIGN_MASK: u32 = 0x88888888;
const I4_LOW_MASK: u32 = 0x0F0F0F0F;

/// Expand low nibbles to bytes
#[inline]
fn expand_lo(v: u32) -> u32 { v & I4_LOW_MASK }

/// Expand high nibbles to bytes
#[inline]
fn expand_hi(v: u32) -> u32 { (v >> 4) & I4_LOW_MASK }

/// Pack byte values back to nibbles
#[inline]
fn pack_nibbles(lo: u32, hi: u32) -> u32 { (lo & I4_LOW_MASK) | ((hi & I4_LOW_MASK) << 4) }

pub fn i4x8_add(a: u32, b: u32) -> u32 {
    // Byte-level SWAR add (reusing ops_i8 approach with 4-bit masks)
    let a_lo = expand_lo(a);
    let a_hi = expand_hi(a);
    let b_lo = expand_lo(b);
    let b_hi = expand_hi(b);
    // ... use SWAR add on bytes with carry blocking, then pack
    // For 4-bit values in bytes: max value 15, sum max 30, < 128 so no carry overflow
    // Simpler approach: just add and mask (since 15+15=30 < 128, no carry to bit 7)
    let sum_lo = (a_lo + b_lo) & 0x0F0F0F0F;
    let sum_hi = (a_hi + b_hi) & 0x0F0F0F0F;
    pack_nibbles(sum_lo, sum_hi)
}

pub fn i4x8_relu(v: u32) -> u32 {
    // Zero out nibbles with sign=1 (bit 3 of each nibble)
    // byte-expansion approach
    let lo = expand_lo(v);
    let hi = expand_hi(v);
    // sign bits at byte level: 0x08 per byte
    let mask_lo = ((lo & 0x08080808) >> 3).wrapping_mul(0xFF);
    let mask_hi = ((hi & 0x08080808) >> 3).wrapping_mul(0xFF);
    let relu_lo = lo & !mask_lo;
    let relu_hi = hi & !mask_hi;
    pack_nibbles(relu_lo, relu_hi)
}

pub fn i4x8_max(a: u32, b: u32) -> u32 {
    // Expand to bytes, convert signed 4-bit values to unsigned bias for comparison
    // I4: sign=0x8 per byte → bias to unsigned: val ^ 0x08 per byte
    // Then use standard SWAR unsigned max (like i8x4 but with 0x08 instead of 0x80)
    let a_lo = expand_lo(a) ^ 0x08080808;
    let a_hi = expand_hi(a) ^ 0x08080808;
    let b_lo = expand_lo(b) ^ 0x08080808;
    let b_hi = expand_hi(b) ^ 0x08080808;
    // ... unsigned max at byte level, then ^ 0x08 to restore sign, pack
    let max_lo = byte_unsigned_max(a_lo, b_lo) ^ 0x08080808;
    let max_hi = byte_unsigned_max(a_hi, b_hi) ^ 0x08080808;
    pack_nibbles(max_lo, max_hi)
}
```

#### src/swar/ops_fp.rs (NEW)
```rust
// Total order transform for FP8 sign-magnitude values
const F8_SIGN: u32 = 0x80808080;

#[inline]
pub fn fp8_total_order(packed: u32) -> u32 {
    let sign = packed & F8_SIGN;
    let fill = (sign >> 7).wrapping_mul(0xFF);  // 0xFF per neg byte, 0x00 per pos byte
    let xor_mask = fill ^ F8_SIGN;  // 0x7F per neg byte, 0x80 per pos byte
    packed ^ xor_mask
}

#[inline]
pub fn fp8_relu(packed: u32) -> u32 {
    let sign = packed & F8_SIGN;
    let fill = (sign >> 7).wrapping_mul(0xFF);
    packed & !fill
}

#[inline]
pub fn fp8_max(a: u32, b: u32) -> u32 {
    let a_ord = fp8_total_order(a);
    let b_ord = fp8_total_order(b);
    let lt = a_ord.wrapping_sub(b_ord) & F8_SIGN;
    let mask = (lt >> 7).wrapping_mul(0xFF);
    let result_ord = a_ord ^ ((a_ord ^ b_ord) & mask);
    // Inverse transform (same as forward)
    fp8_total_order(result_ord)
}

// Same for F4x8 but with nibble expansion
const F4N_SIGN: u32 = 0x08080808;  // sign at byte level for expanded nibbles

pub fn f4x8_relu(packed: u32) -> u32 {
    // Expand nibbles to bytes, apply byte-level relu, repack
    let lo = packed & 0x0F0F0F0F;
    let hi = (packed >> 4) & 0x0F0F0F0F;
    let sign_lo = lo & F4N_SIGN;
    let sign_hi = hi & F4N_SIGN;
    let fill_lo = (sign_lo >> 3).wrapping_mul(0xFF);
    let fill_hi = (sign_hi >> 3).wrapping_mul(0xFF);
    let relu_lo = lo & !fill_lo;
    let relu_hi = hi & !fill_hi;
    (relu_lo & 0x0F0F0F0F) | ((relu_hi & 0x0F0F0F0F) << 4)
}
```

#### src/swar/mod.rs
```rust
pub mod ops_i8;
pub mod ops_i4;
pub mod ops_fp;
```

### Phase 4: Kernel dispatch

#### Kernel dispatch architecture
Current: string-based dispatch in `src/backend/cpu/mod.rs`
New: Add new kernel strings for F8 and F4 types.

The dispatch logic at lines 267-295 of mod.rs needs extended patterns:

```rust
// Quantized: F8 activation + F8 weight
(_, true)
    if input_dtypes.first().is_some_and(|d| matches!(d, IrDType::F8 { .. }))
        && input_dtypes.iter().any(|d| matches!(d, IrDType::F8 { .. })) =>
{
    "matmul_f8"
}
// Quantized: F4 weight (any activation)
(_, true)
    if input_dtypes.iter().any(|d| matches!(d, IrDType::F4 { .. })) =>
{
    "matmul_f4"
}
// Quantized: F8R weight
(_, true)
    if input_dtypes.iter().any(|d| matches!(d, IrDType::F8R { .. })) =>
{
    "matmul_f8r"
}
```

Priority order should be: I4 > I8 > F8 > F4 > F8R (from most specific to least).

#### Weight loading
Add `QuantizedWeightMeta` support for F8/F4:
```rust
// src/backend/prepared.rs or mod.rs
pub struct QuantizedWeightMeta {
    pub block_scales: Option<Vec<u8>>,  // F4 only
    pub block_size: Option<usize>,       // F4 only
    // ... existing fields
}
```

#### Conv2d dispatch
Similar patterns for conv2d:
```rust
"conv2d_f8" | "conv2d_f8_relu" | "conv2d_f8_gelu" | "conv2d_f8_silu" => {
    // im2col + gemm_packed_f8x4
}
"conv2d_f4" | ... => {
    // im2col + gemm_packed_f4x8
}
```

### Phase 5: Training support

F8x4R (E5M2) serves as the gradient type for all quantized training:
- Wider range (±57344) prevents gradient underflow
- Lower precision (2-bit mantissa) is acceptable for gradients
- STE bypasses quantization in backward pass
- Stochastic rounding for forward quant (optional)

In `wrap_quantized_optimizer()` (compiler pass), gradient types default to IrDType::F8R.

### Phase 6: Python API

```python
# fastnn/precision.py
class Precision(IntEnum):
    F32 = 0
    F16 = 1
    I8 = 2
    I4 = 3
    F8 = 4   # NEW
    F8R = 5  # NEW
    F4 = 6   # NEW
    
    @property
    def bit_width(self) -> int:
        return {F32: 32, F16: 16, I8: 8, I4: 4, F8: 8, F8R: 8, F4: 4}[self]
    
    @property
    def is_float(self) -> bool:
        return self in (F32, F16, F8, F8R, F4)
    
    @property
    def is_quantized(self) -> bool:
        return self in (I8, I4, F8, F8R, F4)
    
    @staticmethod
    def from_string(s: str) -> "Precision":
        mapping = {
            "f32": F32, "float32": F32,
            "f16": F16, "float16": F16,
            "i8": I8, "int8": I8,
            "i4": I4, "int4": I4,
            "f8": F8, "fp8": F8,
            "f8r": F8R, "fp8r": F8R, "e5m2": F8R,
            "f4": F4, "fp4": F4, "nvfp4": F4, "e2m1": F4,
        }
        ...
```

Quantizer needs block_scales support for F4:
```python
class Quantizer:
    BLOCK_SCALED = "block_scaled"  # NEW scheme for NVFP4
    
    def __init__(self, precision, scheme="per_channel", block_size=None, ...):
        ...
        if scheme == "block_scaled":
            self.block_size = block_size or 16
    
    def quantize(self, f32_weights):
        if self.precision == Precision.I4:
            # Use from_f32_per_channel (existing)
        elif self.precision == Precision.F4:
            # NVFP4 block scaling
            return self._quantize_nvfp4(f32_weights)
        ...
```

### Phase 7: Serialization

In `src/onnx/export.rs` and `.fnn` format:
- Type tag mapping: F8=4, F8R=5, F4=6
- Serialize F4 block_scales array alongside packed data
- `save_fnn` / `load_fnn` updated for new IrDType variants

### Phase 8: Tests

For each new type, add tests:
1. **Pack/unpack roundtrip**: Verify `pack_from_f32; unpack_to_f32` preserves values within quantization error
2. **GEMM correctness**: Verify `gemm_packed_f8x4` produces same output as F32 GEMM within tolerance
3. **F4 LUT correctness**: Verify LUT-based dot product matches direct FP4 computation
4. **Block scaling**: Verify block-scale dequantization matches NVFP4 specification
5. **SWAR operations**: Verify `relu/max/min` produce same results as f32 comparison
6. **Edge cases**: NaN propagation in F8, saturation in F8R (57344 max), zero handling
7. **Integration**: End-to-end forward pass with new types

Existing test files to update:
- `tests/quantized_pipeline.rs` — add F8/F4 tests
- `tests/cpu_reference_oracle.rs` — add oracle tests
- `benches/quantized_vs_pytorch.rs` — add F8/F4 benchmarks

### Phase 9: WGSL/GPU shaders

Update `src/backend/wgpu/quantized.rs`:
- Add `wgsl_unpack_body()` for F8x4, F8x4R, F4x8 in their respective PackedWord impls
- F8x4 WGSL: inline E4M3 decode (4 lines of bit manipulation per element)
- F4x8 WGSL: inline E2M1 decode + block scale decode

## Implementation Checklist

### Phase 1: Rename U→I
- [ ] Rename `src/dtypes/u4x8.rs` → `i4x8.rs` (content: I4x8→I4x8)
- [ ] Rename `src/dtypes/u8x4.rs` → `i8x4.rs` (content: I8x4→I8x4)
- [ ] Rename `src/swar/ops_8bit.rs` → `ops_i8.rs`
- [ ] Update `src/dtypes/mod.rs` (module decls, re-exports)
- [ ] Update `src/swar/mod.rs`
- [ ] Update `src/ir/node.rs` (IrDType variants, merge I8 activation with fields)
- [ ] Update `src/packed_tensor.rs`
- [ ] Update `src/storage_quantized.rs`
- [ ] Update `src/backend/cpu/mod.rs` (kernel strings, IrDType matches, type params)
- [ ] Update `src/backend/cpu/matmul.rs`
- [ ] Update `src/backend/cpu/swar.rs`
- [ ] Update `src/backend/cpu/packed_gemm.rs`
- [ ] Update `src/backend/cpu/packed_conv.rs`
- [ ] Update `src/backend/cpu/microkernels/gemm.rs`
- [ ] Update `src/backend/prepared.rs`
- [ ] Update `src/backend/wgpu/quantized.rs`
- [ ] Update `src/compiler/passes/*.rs`
- [ ] Update `src/storage.rs`
- [ ] Update `src/lib.rs`
- [ ] Update `src/onnx/*.rs`
- [ ] Update `fastnn/precision.py`
- [ ] Update `fastnn/io/*.py`
- [ ] Update `tests/*.rs`
- [ ] Update `benches/*.rs`
- [ ] Run `cargo build` to verify

### Phase 2: F8x4 + F8x4R types
- [ ] Create `src/dtypes/f8x4.rs` with decode_fp8_e4m3, encode_fp8_e4m3, PackedWord impl
- [ ] Create `src/dtypes/f8x4r.rs` with decode_fp8_e5m2, encode_fp8_e5m2, PackedWord impl
- [ ] Update `src/dtypes/mod.rs`
- [ ] Add IrDType::F8 and IrDType::F8R to `src/ir/node.rs`
- [ ] Update packed_byte_size, byte_size_with_env, impl_ir_dtype_props
- [ ] Add from_f32_f8 constructor to `src/packed_tensor.rs`
- [ ] Create `src/backend/cpu/packed_gemm_fp.rs` with gemm_packed_f8x4
- [ ] Add matmul dispatch functions to `src/backend/cpu/matmul.rs`
- [ ] Update kernel dispatch in `src/backend/cpu/mod.rs`
- [ ] Add conv2d_f8 dispatch
- [ ] Add tests: pack/unpack roundtrip, GEMM correctness
- [ ] Add WGSL unpack bodies

### Phase 3: F4x8 type
- [ ] Create `src/dtypes/f4x8.rs` — F4x8 struct, LUT definition, PackedWord impl
- [ ] Define `FP4_PROD_LUT: [i16; 256]` — precomputed products
- [ ] Add `f4x8_dot_lut(a: u32, b: u32) -> i32` — LUT-based dot product
- [ ] Update `src/dtypes/mod.rs`
- [ ] Add IrDType::F4 to `src/ir/node.rs`
- [ ] Add block_scales field to `PackedTensor`
- [ ] Add from_f32_f4_blocked constructor
- [ ] Add gemm_packed_f4x8 with block scaling to `packed_gemm_fp.rs`
- [ ] Add matmul dispatch for F4
- [ ] Add conv2d_f4 dispatch
- [ ] Add tests

### Phase 4: SWAR ops
- [ ] Create `src/swar/ops_i4.rs`
- [ ] Create `src/swar/ops_fp.rs`
- [ ] Update `src/swar/mod.rs`
- [ ] Test SWAR max/min/relu vs scalar versions

### Phase 5: Training support
- [ ] F8x4R gradient type in wrap_quantized_optimizer
- [ ] STE bypass for backward pass
- [ ] Stochastic rounding (optional, future)

### Phase 6: Python + serialization
- [ ] Update `fastnn/precision.py`
- [ ] Update Quantizer for F8/F4
- [ ] Update serialize/deserialize for new IrDType variants
- [ ] Update ONNX export
- [ ] Update .fnn format

## F4x8 LUT Definition

```rust
// Pre-computed LUT for FP4 E2M1 products
// LUT[a][b] = round(4.0 * real_val(a) * real_val(b))
// where real_val(code) = magnitude(code) * (1 - 2*((code>>3)&1))
// magnitude = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

const FP4_REAL: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn build_fp4_lut() -> [i16; 256] {
    let mut lut = [0i16; 256];
    for a_code in 0..16u8 {
        let a_mag = FP4_REAL[(a_code & 0x7) as usize];
        let a_sign = if (a_code >> 3) & 1 != 0 { -1.0 } else { 1.0 };
        let a_val = a_mag * a_sign;
        for b_code in 0..16u8 {
            let b_mag = FP4_REAL[(b_code & 0x7) as usize];
            let b_sign = if (b_code >> 3) & 1 != 0 { -1.0 } else { 1.0 };
            let b_val = b_mag * b_sign;
            let product = a_val * b_val;
            let scaled = (product * 4.0).round() as i16;
            lut[(a_code as usize) * 16 + (b_code as usize)] = scaled.clamp(-32768, 32767);
        }
    }
    lut
}

// Use either compile-time or lazy_static
static FP4_LUT: Lazy<[i16; 256]> = Lazy::new(build_fp4_lut);

#[inline]
pub fn f4x8_dot_lut(a: u32, b: u32) -> i32 {
    let mut sum: i32 = 0;
    let lut = &*FP4_LUT;
    for i in 0..8 {
        let nib_a = ((a >> (i * 4)) & 0xF) as usize;
        let nib_b = ((b >> (i * 4)) & 0xF) as usize;
        sum += lut[nib_a * 16 + nib_b] as i32;
    }
    sum
}
```

## FP4 Block Scaling Format

NVFP4 stores scales as FP8 E4M3 per block:

```
Block of 16 FP4 elements:
  Bytes 0-7:   16 FP4 nibbles packed into 8 bytes
  Byte 8:      FP8 E4M3 scale (shared across all 16 elements)
  
Total: 9 bytes per 16 elements = 4.5 bits per element
```

To compute the scale:
```rust
// For a block of 16 f32 values:
let max_abs = values.iter().map(|v| v.abs()).fold(0.0, f32::max);
let scale = if max_abs == 0.0 { 0x00 } else { encode_fp8_e4m3(max_abs / 6.0) };
// Stored as FP8 E4M3 byte
```

Dequantize: `real_val[i] = decode_fp8_e4m3(scale_byte) * fp4_real[code[i]]`

In the GEMM kernel, the block scale factors out:
```
block_result = Σ real(a[i]) * real(b[i])
             = scale_A * scale_B * Σ fp4_code(a[i]) * fp4_code(b[i])
             = scale_A * scale_B * (LUT_sum / 4.0)
```
So: `block_result = LUT_sum * scale_A * scale_B / 4.0`

## Edge Cases

### F8x4 (E4M3)
- **Zero**: both +0 (0x00) and -0 (0x80) are valid. Zero is preserved through pack/unpack.
- **NaN**: 0x7F or 0xFF. During pack, NaN inputs → 0x7F. During LNS multiply, NaN * anything = NaN.
- **Saturation**: values outside ±448 saturate to 0x7E (max +) or 0xFE (max -).
- **Subnormals**: exponent=0, mantissa!=0. Value = ±0.mant * 2^(-6). Minimum positive subnormal = 2^(-9) ≈ 0.00195.

### F8x4R (E5M2)
- **Inf**: 0x7C, 0xFC. Propagation: Inf * 0 = NaN, Inf * finite = Inf with sign.
- **NaN**: 0x7E, 0x7F, 0xFE, 0xFF.
- **Saturation**: values outside ±57344 saturate to 0x7B or 0xFB (max normal).
- **Subnormals**: minimum positive = 2^(-24) ≈ 5.96e-8.

### F4x8 (E2M1)
- **No NaN/Inf** — all 16 bit patterns are valid values.
- **-0**: 0x8 (sign=1, magnitude=0). Should be treated as 0.0 in arithmetic.
- **Saturation**: none needed (max values are 6.0 and -6.0, both exact).

## Compiler Pass Updates

### `src/compiler/passes/quantization.rs`
The `quantize_weights()` function currently matches IrDType::U4/U8. Extend to match I4/I8/F8/F8R/F4.

### `src/compiler/passes/auto_cast.rs`
Entry point for `model.to("f8")` / `model.to("f4")`. The existing pattern for "u4" should be extended.

### `src/compiler/passes/calibration.rs`
Scale computation for F8/F4 uses max-abs (no KL divergence since no zero-point).
For F4, also compute block scales (E8M0 per block of 16).

### `src/compiler/passes/activation_quantization.rs`
Activation quantization to F8x4 (I8 activation already exists). Uses `QuantizeActivations` with FP8 format.

## Cargo.toml Changes

No additional dependencies needed. `half` crate already provides f16 types. The FP8/FP4 encoding uses bit manipulation from standard library (`f32::to_bits`, `f32::from_bits`). The LUT for F4x8 is compile-time generated (or `lazy_static`/`OnceLock`).

## Verification Steps

For each phase:
1. `cargo build` — must compile without errors
2. `cargo test` — existing tests must pass
3. `cargo test <new_test>` — new tests pass
4. `cargo bench` — no regressions in existing benchmarks

For phase 1 specifically:
- After rename, `cargo build` should pass with zero warnings about deprecated names
- All tests should pass (rename doesn't change behavior)
- Python `maturin build` should produce working wheel

## SWAR Dot Product Summary

| Type | Dot Product Method | Accumulator | Dequant Terms |
|------|-------------------|-------------|---------------|
| I4x8 | per-nibble loop (8 iterations, sign-extend, multiply) | i32 | 4 (scale + zp) |
| I8x4 | per-byte extract (4 bytes, sign-extend, multiply) | i32 | 4 (scale + zp) |
| F8x4 | LNS int add: byte_a + byte_b - B (approx) then decode to f32 | f32 | 2 (scale only) |
| F8x4R | LNS int add: byte_a + byte_b - B (approx) then decode to f32 | f32 | 2 (scale only) |
| F4x8 | 256-entry i16 LUT per nibble pair | i32 → f32 at output | Block-scaled + /4 |

## Key Implementation Notes

1. **No zero-point cross-term for F8/F4**: This simplifies the GEMM inner loop significantly. The dequant is just `val * scale`, not `val * scale + zp`.

2. **F4 LUT is exact, not approximate**: The LUT stores `round(4 * real_product)` for each of the 16×16 FP4 code pairs. This is mathematically exact (the only error is the i16 rounding, which at ±32768/4 = ±8192 max product, is 0.0 error since all products are multiples of 0.25).

3. **Block scales factor out of the inner loop**: For F4x8 GEMM, the block scale multiplication is per-block, not per-element. For block_size=16 and 8 nibbles per word, each block covers 2 packed words. The LUT sum across those 2 words is multiplied by both block scales once.

4. **F8x4 LNS approximation**: The formula `byte_a + byte_b - B` gives an approximate product. For E4M3 (B=0x38), faithful rounding requires a carry-in correction (boolean expression per element). For performance, the approximate version may be acceptable (similar to how fastnn uses symmetric quantization without KL-divergence for some paths).

5. **WGSL shaders**: For GPU dispatch, both F8 and F4 need inline decode functions in WGSL. The `wgsl_unpack_body()` method in each PackedWord impl provides this, injected at compile time into the shader source.
