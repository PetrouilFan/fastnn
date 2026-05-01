//! Q4_K GGUF blockwise quantization — 4-bit, 256-elem super-blocks.
//!
//! Layout per super-block (`QK4_K = 256`):
//! ```text
//! ┌────────┬────────┬──────────┬──────────┐
//! │ d: f16 │dmin:f16│scales:u8×12│ qs:u8×128 │
//! │ 2 B    │ 2 B    │ 12 B       │ 128 B    │
//! └────────┴────────┴──────────┴──────────┘
//! total = 144 bytes / 256 weights ≈ 4.5 bits/weight
//! ```
//!
//! Each super-block contains 8 sub-blocks of 32 weights each.
//!
//! - `d`     : f16 scale applied to all quantized weights.
//! - `dmin`  : f16 scale applied to all quantized minimums.
//! - `scales`: 12 bytes packing 8×6-bit weight scales + 8×6-bit min scales.
//! - `qs`    : 128 bytes = 256 nibbles (4-bit weights).
//!
//! Dequantization of weight `i` in sub-block `s`:
//! ```text
//! w[i] = d * (scale_s * (qs[i] - 8) + min_s)
//! ```
//! where `qs[i]` is the 4-bit nibble (0..15), centered by subtracting 8.

const QK4_K: usize = 256;
const K_SCALE: usize = 32; // elements per sub-block
const NUM_SUB_BLOCKS: usize = QK4_K / K_SCALE; // 8
const BLOCK_SIZE_BYTES: usize = 2 + 2 + 12 + 128; // 144

/// Raw data for one Q4_K super-block.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ4_K {
    /// Global scale for this super-block.
    pub d: half::f16,
    /// Global minimum scale for this super-block.
    pub dmin: half::f16,
    /// 12 bytes packing 8×6-bit scales + 8×6-bit mins.
    pub scales: [u8; 12],
    /// 128 bytes = 256 4-bit weights.
    pub qs: [u8; QK4_K / 2],
}

unsafe impl bytemuck::Pod for BlockQ4_K {}
unsafe impl bytemuck::Zeroable for BlockQ4_K {}

impl BlockQ4_K {
    /// Unpack the 8 sub-block weight-scales and 8 sub-block minimum-scales.
    ///
    /// Returns `(weight_scales[8], min_scales[8])` as `u8` values (0..63).
    ///
    /// Layout matches GGML/llama.cpp Q4_K spec (4×6-bit packed into 3 bytes):
    /// ```text
    /// bytes 0..2 : weight scales 0-3   (or min scales 0-3)
    /// bytes 6..8 : weight scales 4-7   (or min scales 4-7)
    ///
    /// byte0 = scale0[5:0] | (scale1[1:0] << 6)
    /// byte1 = scale1[5:2] | (scale2[3:0] << 4)
    /// byte2 = scale2[5:4] | (scale3[5:0] << 2)
    /// ```
    #[inline]
    fn unpack_scales(
        &self,
    ) -> ([u8; NUM_SUB_BLOCKS], [u8; NUM_SUB_BLOCKS]) {
        let mut w_scales = [0u8; NUM_SUB_BLOCKS];
        let mut m_scales = [0u8; NUM_SUB_BLOCKS];

        // Unpack first 4 weight scales (bytes 0,1,2)
        w_scales[0] = self.scales[0] & 0x3F;
        w_scales[1] = ((self.scales[0] >> 6) & 0x03) | ((self.scales[1] & 0x0F) << 2);
        w_scales[2] = ((self.scales[1] >> 4) & 0x0F) | ((self.scales[2] & 0x03) << 4);
        w_scales[3] = (self.scales[2] >> 2) & 0x3F;

        // Unpack first 4 min scales (bytes 3,4,5)
        m_scales[0] = self.scales[3] & 0x3F;
        m_scales[1] = ((self.scales[3] >> 6) & 0x03) | ((self.scales[4] & 0x0F) << 2);
        m_scales[2] = ((self.scales[4] >> 4) & 0x0F) | ((self.scales[5] & 0x03) << 4);
        m_scales[3] = (self.scales[5] >> 2) & 0x3F;

        // Unpack second 4 weight scales (bytes 6,7,8)
        w_scales[4] = self.scales[6] & 0x3F;
        w_scales[5] = ((self.scales[6] >> 6) & 0x03) | ((self.scales[7] & 0x0F) << 2);
        w_scales[6] = ((self.scales[7] >> 4) & 0x0F) | ((self.scales[8] & 0x03) << 4);
        w_scales[7] = (self.scales[8] >> 2) & 0x3F;

        // Unpack second 4 min scales (bytes 9,10,11)
        m_scales[4] = self.scales[9] & 0x3F;
        m_scales[5] = ((self.scales[9] >> 6) & 0x03) | ((self.scales[10] & 0x0F) << 2);
        m_scales[6] = ((self.scales[10] >> 4) & 0x0F) | ((self.scales[11] & 0x03) << 4);
        m_scales[7] = (self.scales[11] >> 2) & 0x3F;

        (w_scales, m_scales)
    }

    /// Dequantize all 256 weights from this block into a f32 buffer.
    #[inline]
    fn dequantize(&self, out: &mut [f32; QK4_K]) {
        let d = self.d.to_f32();
        let dmin = self.dmin.to_f32();
        let (scales, mins) = self.unpack_scales();

        for sub in 0..NUM_SUB_BLOCKS {
            let scale = d * (scales[sub] as f32);
            let min = dmin * (mins[sub] as f32);
            let qs_offset = sub * (K_SCALE / 2); // 16 bytes per sub-block

            for byte_idx in 0..(K_SCALE / 2) {
                let packed = self.qs[qs_offset + byte_idx];
                let base = sub * K_SCALE + byte_idx * 2;

                let lo_nibble = (packed & 0x0F) as i32 - 8;
                let hi_nibble = ((packed >> 4) & 0x0F) as i32 - 8;

                out[base] = scale * (lo_nibble as f32) + min;
                out[base + 1] = scale * (hi_nibble as f32) + min;
            }
        }
    }
}

/// Native Q4_K tensor.
pub struct Q4_K {
    shape: [usize; 2], // [out, in]
    blocks: Vec<BlockQ4_K>,
}

impl Q4_K {
    /// Create from raw bytes.
    pub fn from_bytes(data: &[u8], shape: [usize; 2]) -> Self {
        let numel = shape[0] * shape[1];
        let num_blocks = numel.div_ceil(QK4_K);
        assert_eq!(
            data.len(),
            num_blocks * BLOCK_SIZE_BYTES,
            "Q4_K: data length {} does not match expected {} bytes",
            data.len(),
            num_blocks * BLOCK_SIZE_BYTES
        );
        let blocks: Vec<BlockQ4_K> = bytemuck::cast_slice(data).to_vec();
        Q4_K { shape, blocks }
    }

    fn blocks_per_row(&self) -> usize {
        self.shape[1].div_ceil(QK4_K)
    }

    pub fn row(&self, row_idx: usize) -> Vec<f32> {
        let blocks_per_row = self.blocks_per_row();
        let block_start = row_idx * blocks_per_row;
        let mut result = Vec::with_capacity(self.shape[1]);

        for block_idx in 0..blocks_per_row {
            let mut scratch = [0.0f32; QK4_K];
            self.blocks[block_start + block_idx].dequantize(&mut scratch);
            result.extend_from_slice(&scratch[..]);
        }

        result.truncate(self.shape[1]);
        result
    }
}

impl crate::quants::QuantizedGemm for Q4_K {
    fn gemv(&self, activation: &[f32], output: &mut [f32]) {
        assert_eq!(activation.len(), self.shape[1]);
        assert_eq!(output.len(), self.shape[0]);

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { gemv_q4_k_avx2(self, activation, output) };
                return;
            }
        }

        gemv_q4_k_scalar(self, activation, output);
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> crate::quants::QuantizedDType {
        crate::quants::QuantizedDType::Q4_K
    }

    fn memory_bytes(&self) -> usize {
        self.blocks.len() * BLOCK_SIZE_BYTES
    }

    fn row(&self, row_idx: usize) -> Vec<f32> {
        Q4_K::row(self, row_idx)
    }
}

// ---------------------------------------------------------------------------
// Scalar reference
// ---------------------------------------------------------------------------
pub fn gemv_q4_k_scalar(q: &Q4_K, activation: &[f32], output: &mut [f32]) {
    let blocks_per_row = q.blocks_per_row();
    let in_features = q.shape[1];

    for out_idx in 0..q.shape[0] {
        let mut accum = 0.0f32;
        let row_offset = out_idx * blocks_per_row;

        for block_idx in 0..blocks_per_row {
            let block = &q.blocks[row_offset + block_idx];
            let d = block.d.to_f32();
            let dmin = block.dmin.to_f32();
            let (scales, mins) = block.unpack_scales();
            let elem_base = block_idx * QK4_K;
            let k = (elem_base + QK4_K).min(in_features) - elem_base;

            for sub in 0..NUM_SUB_BLOCKS {
                let scale = d * (scales[sub] as f32);
                let min = dmin * (mins[sub] as f32);
                let qs_offset = sub * (K_SCALE / 2);

                for byte_idx in 0..(K_SCALE / 2) {
                    let packed = block.qs[qs_offset + byte_idx];
                    let a_idx_base = elem_base + sub * K_SCALE + byte_idx * 2;

                    let lo = (packed & 0x0F) as i32 - 8;
                    let hi = ((packed >> 4) & 0x0F) as i32 - 8;

                    if a_idx_base < in_features {
                        accum += (scale * (lo as f32) + min) * activation[a_idx_base];
                    }
                    if a_idx_base + 1 < in_features {
                        accum += (scale * (hi as f32) + min) * activation[a_idx_base + 1];
                    }
                }
            }
        }

        output[out_idx] = accum;
    }
}

// ---------------------------------------------------------------------------
// AVX2 kernel — dequantize each block to scratch, then FMA
// ---------------------------------------------------------------------------
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
unsafe fn gemv_q4_k_avx2(q: &Q4_K, activation: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let blocks_per_row = q.blocks_per_row();
    let in_features = q.shape[1];
    let mut scratch: [f32; QK4_K] = [0.0; QK4_K];

    for out_idx in 0..q.shape[0] {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let row_offset = out_idx * blocks_per_row;

        for block_idx in 0..blocks_per_row {
            let block = &q.blocks[row_offset + block_idx];
            let elem_base = block_idx * QK4_K;
            let k = (elem_base + QK4_K).min(in_features) - elem_base;

            // Dequantize full block into scratch
            block.dequantize(&mut scratch);

            // FMA dot-product scratch[0..k] · activation[elem_base..]
            let mut j = 0usize;
            while j + 32 <= k {
                let w0 = _mm256_loadu_ps(scratch.as_ptr().add(j));
                let a0 = _mm256_loadu_ps(activation.as_ptr().add(elem_base + j));
                acc0 = _mm256_fmadd_ps(w0, a0, acc0);

                let w1 = _mm256_loadu_ps(scratch.as_ptr().add(j + 8));
                let a1 = _mm256_loadu_ps(activation.as_ptr().add(elem_base + j + 8));
                acc1 = _mm256_fmadd_ps(w1, a1, acc1);

                let w2 = _mm256_loadu_ps(scratch.as_ptr().add(j + 16));
                let a2 = _mm256_loadu_ps(activation.as_ptr().add(elem_base + j + 16));
                acc2 = _mm256_fmadd_ps(w2, a2, acc2);

                let w3 = _mm256_loadu_ps(scratch.as_ptr().add(j + 24));
                let a3 = _mm256_loadu_ps(activation.as_ptr().add(elem_base + j + 24));
                acc3 = _mm256_fmadd_ps(w3, a3, acc3);

                j += 32;
            }
            while j + 8 <= k {
                let w = _mm256_loadu_ps(scratch.as_ptr().add(j));
                let a = _mm256_loadu_ps(activation.as_ptr().add(elem_base + j));
                acc0 = _mm256_fmadd_ps(w, a, acc0);
                j += 8;
            }
            while j < k {
                let w = scratch[j];
                let a = activation[elem_base + j];
                // Extract lane 0, add, store back
                let mut arr: [f32; 8] = [0.0; 8];
                std::ptr::copy_nonoverlapping(
                    &acc0 as *const __m256 as *const f32,
                    arr.as_mut_ptr(),
                    8,
                );
                arr[0] += w * a;
                acc0 = std::ptr::read_unaligned(arr.as_ptr() as *const __m256);
                j += 1;
            }
        }

        // Horizontal sum
        let mut total = 0.0f32;
        for acc in [acc0, acc1, acc2, acc3] {
            let mut arr: [f32; 8] = [0.0; 8];
            std::ptr::copy_nonoverlapping(
                &acc as *const __m256 as *const f32,
                arr.as_mut_ptr(),
                8,
            );
            for v in arr {
                total += v;
            }
        }
        output[out_idx] = total;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::quants::QuantizedGemm;

    #[test]
    fn test_q4_k_block_unpack_scales() {
        let block = BlockQ4_K {
            d: half::f16::from_f32_const(1.0),
            dmin: half::f16::from_f32_const(0.5),
            scales: [
                // First 4 weight scales (scale0=1, scale1=2, scale2=3, scale3=4)
                // byte0 = scale0[5:0] | (scale1[1:0] << 6) = 1 | (0 << 6) = 1... 
                // Wait, scale1=2=0b10, so scale1[1:0]=2
                // byte0 = 1 | (2 << 6) = 1 | 128 = 0x81
                0x81,
                // byte1 = scale1[5:2] | (scale2[3:0] << 4) = 0 | (3 << 4) = 0x30
                0x30,
                // byte2 = scale2[5:4] | (scale3[5:0] << 2) = 0 | (4 << 2) = 0x10
                0x10,
                // First 4 min scales (min0=5, min1=6, min2=7, min3=8)
                // min0=5=0b000101, min1=6=0b000110
                // byte3 = 5 | (6 << 6) = 5 | 384 = 389 = 0x185... wait, 6<<6 = 384, that's > 255
                // 6 in binary = 0b000110, bits[1:0] = 2 (0b10)
                // byte3 = 5 | (2 << 6) = 5 | 128 = 133 = 0x85
                0x85,
                // min1=6, bits[5:2] = 1 (0b0001)
                // min2=7=0b000111, bits[3:0] = 7
                // byte4 = 1 | (7 << 4) = 1 | 112 = 113 = 0x71
                0x71,
                // min2=7, bits[5:4] = 0 (0b00)
                // min3=8=0b001000, bits[5:0] = 8
                // byte5 = 0 | (8 << 2) = 32 = 0x20
                0x20,
                // Second 4 weight scales (scale4=9, scale5=10, scale6=11, scale7=12)
                // scale4=9=0b001001, scale5=10=0b001010, scale5[1:0]=2
                // byte6 = 9 | (2 << 6) = 9 | 128 = 137 = 0x89
                0x89,
                // scale5=10, bits[5:2]=2 (0b0010)
                // scale6=11=0b001011, bits[3:0]=11
                // byte7 = 2 | (11 << 4) = 2 | 176 = 178 = 0xB2
                0xB2,
                // scale6=11, bits[5:4]=0 (0b00)
                // scale7=12=0b001100, bits[5:0]=12
                // byte8 = 0 | (12 << 2) = 0 | 48 = 48 = 0x30
                0x30,
                // Second 4 min scales (min4=13, min5=14, min6=15, min7=16)
                // min4=13=0b001101, min5=14=0b001110, min5[1:0]=2
                // byte9 = 13 | (2 << 6) = 13 | 128 = 141 = 0x8D
                0x8D,
                // min5=14, bits[5:2]=3 (0b0011)
                // min6=15=0b001111, bits[3:0]=15
                // byte10 = 3 | (15 << 4) = 3 | 240 = 243 = 0xF3
                0xF3,
                // min6=15, bits[5:4]=3 (0b11)
                // min7=16=0b010000, bits[5:0]=16
                // byte11 = 0 | (16 << 2) = 0 | 64 = 64 = 0x40
                0x40,
            ],
            qs: [0u8; 128],
        };

        let (w_scales, m_scales) = block.unpack_scales();
        assert_eq!(w_scales[0], 1);
        assert_eq!(w_scales[1], 2);
        assert_eq!(w_scales[2], 3);
        assert_eq!(w_scales[3], 4);
        assert_eq!(m_scales[0], 5);
        assert_eq!(m_scales[1], 6);
        assert_eq!(m_scales[2], 7);
        assert_eq!(m_scales[3], 8);
        assert_eq!(w_scales[4], 9);
        assert_eq!(w_scales[5], 10);
        assert_eq!(w_scales[6], 11);
        assert_eq!(w_scales[7], 12);
        assert_eq!(m_scales[4], 13);
        assert_eq!(m_scales[5], 14);
        assert_eq!(m_scales[6], 15);
        assert_eq!(m_scales[7], 16);
    }

    #[test]
    fn test_q4_k_dequantize_zeros() {
        // All-zero quantized values → nibbles = 0 → centered = -8
        // scales = 1, mins = 0, d = 1.0, dmin = 0.0
        // So all dequantized values = 1.0 * (1 * (-8) + 0) = -8
        let mut block = BlockQ4_K {
            d: half::f16::from_f32_const(1.0),
            dmin: half::f16::from_f32_const(0.0),
            scales: [0u8; 12],
            qs: [0u8; 128],
        };
        // Set all scales to 1 (6-bit value 1)
        // scales[0] = 1 (scale0=1), rest = 0
        block.scales[0] = 0x01;
        // scales[3] = 1 (min0=1)... wait, the first 4 mins are in bytes 3,4,5
        // Actually, we want all 8 scales to be 1 and all 8 mins to be 0.
        // scales bytes 0,1,2 encode scale0..scale3
        // scale0=1: byte0 bit[5:0] = 1
        // scale1=1: byte0 bit[7:6]=0, byte1 bit[3:0]=1 → byte1 = 1
        // scale2=1: byte1 bit[7:4]=0, byte2 bit[1:0]=1 → byte2 = 1
        // scale3=1: byte2 bit[7:2]=0 → byte2 already = 1
        // Hmm, let me set the scales array more carefully.
        // For simplicity, set all scales to 0 except scale0=1, and verify only first sub-block.
        // Actually, let me set all scales to 1 and all mins to 0.
        // 
        // For 6-bit values packed across bytes:
        // 8 scales × 6 bits = 48 bits = 6 bytes (bytes 0..5)
        // 8 mins   × 6 bits = 48 bits = 6 bytes (bytes 6..11)
        //
        // To encode scales = [1,1,1,1,1,1,1,1]:
        // bits: 1,1,1,1,1,1,1,1  each is 6-bit = 0b000001
        // Byte layout (LSB first):
        // byte0: bits[5:0] of scale0 → 1
        // byte0: bits[7:6] of scale0 → 0
        // byte1: bits[3:0] = bits[9:6] of (scale1|scale0) ... this is getting confusing.
        //
        // Let me just manually set the bytes for scales=[1,1,1,1,1,1,1,1], mins=[0,0,0,0,0,0,0,0]
        // Each 6-bit value = 1 = 0b000001
        // Pack 4 values into 3 bytes:
        // val0=1: bits[5:0] = 1
        // val1=1: bits[5:0] = 1
        // val2=1: bits[5:0] = 1
        // val3=1: bits[5:0] = 1
        //
        // byte0 = val0[5:0] | (val1[1:0] << 6) = 1 | (1 << 6) = 0x41
        // byte1 = val1[5:2] | (val2[3:0] << 4) = 0 | (1 << 4) = 0x10
        // byte2 = val2[5:4] | (val3[5:0] << 2) = 0 | (1 << 2) = 0x04
        //
        // So bytes [0x41, 0x10, 0x04] encode [1,1,1,1]
        block.scales[0] = 0x41;
        block.scales[1] = 0x10;
        block.scales[2] = 0x04;
        // mins [0,0,0,0]: all zero
        block.scales[3] = 0;
        block.scales[4] = 0;
        block.scales[5] = 0;
        // scales [1,1,1,1] for sub-blocks 4..7
        block.scales[6] = 0x41;
        block.scales[7] = 0x10;
        block.scales[8] = 0x04;
        // mins [0,0,0,0] for sub-blocks 4..7
        block.scales[9] = 0;
        block.scales[10] = 0;
        block.scales[11] = 0;

        let mut out = [0.0f32; QK4_K];
        block.dequantize(&mut out);

        // All values should be -8 (nibble 0 → -8, scale 1, min 0, d=1)
        for (&val, i) in out.iter().zip(0..QK4_K) {
            let diff = (val - (-8.0)).abs();
            assert!(
                diff < 1e-4,
                "index {}: expected -8, got {} (diff={})",
                i, val, diff
            );
        }
    }

    #[test]
    fn test_q4_k_simple_gemv() {
        // 1 output row, 256 input features = 1 block.
        // All weights = -8 (nibble=0, scale=1, min=0, d=1)
        // Activation = all 1.0
        // Expected = 256 * (-8) = -2048
        let mut block = BlockQ4_K {
            d: half::f16::from_f32_const(1.0),
            dmin: half::f16::from_f32_const(0.0),
            scales: [0u8; 12],
            qs: [0u8; 128],
        };
        // Set all scales to 1 (same encoding as above)
        // Group 0 (bytes 0,1,2): scales 0-3 = 1
        block.scales[0] = 0x41;
        block.scales[1] = 0x10;
        block.scales[2] = 0x04;
        // Group 1 (bytes 3,4,5): mins 0-3 = 0 (already zero)
        // Group 2 (bytes 6,7,8): scales 4-7 = 1
        block.scales[6] = 0x41;
        block.scales[7] = 0x10;
        block.scales[8] = 0x04;
        // Group 3 (bytes 9,10,11): mins 4-7 = 0 (already zero)

        let q4 = Q4_K {
            shape: [1, 256],
            blocks: vec![block],
        };

        let act = vec![1.0f32; 256];
        let mut out = vec![0.0f32; 1];
        q4.gemv(&act, &mut out);

        let expected = -2048.0f32;
        let diff = (out[0] - expected).abs();
        assert!(
            diff < 1e-2,
            "expected {}, got {} (diff={})",
            expected, out[0], diff
        );
    }

    #[test]
    fn test_q4_k_scalar_vs_avx2() {
        let shape: [usize; 2] = [8, 512]; // 8 rows, 512 cols → 2 blocks/row
        let num_blocks: usize = (8usize * 512usize).div_ceil(QK4_K);
        let mut data = vec![0u8; num_blocks * BLOCK_SIZE_BYTES];

        // Fill with deterministic data
        for i in 0..num_blocks {
            let offset = i * BLOCK_SIZE_BYTES;
            let d = half::f16::from_f32_const(1.0 + (i as f32) * 0.001);
            let d_bytes: [u8; 2] = d.to_bits().to_le_bytes();
            data[offset] = d_bytes[0];
            data[offset + 1] = d_bytes[1];
            // dmin = 0.5
            let dmin = half::f16::from_f32_const(0.5);
            let dmin_bytes: [u8; 2] = dmin.to_bits().to_le_bytes();
            data[offset + 2] = dmin_bytes[0];
            data[offset + 3] = dmin_bytes[1];
            // scales: mix of values
            for s in 0..12 {
                data[offset + 4 + s] = ((i * 12 + s) % 64) as u8;
            }
            // qs: deterministic pattern
            for q in 0..128 {
                data[offset + 16 + q] = ((i * 128 + q) % 256) as u8;
            }
        }

        let q4 = Q4_K::from_bytes(&data, shape);
        let act: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();

        let mut out_scalar = vec![0.0f32; 8];
        gemv_q4_k_scalar(&q4, &act, &mut out_scalar);

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") {
            let mut out_avx2 = vec![0.0f32; 8];
            unsafe { gemv_q4_k_avx2(&q4, &act, &mut out_avx2) };
            for (i, (&scalar, &avx)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                let diff = (scalar - avx).abs();
                assert!(
                    diff < 1e-2,
                    "row {}: scalar={}, avx={}, diff={}",
                    i, scalar, avx, diff
                );
            }
        }
    }
}
