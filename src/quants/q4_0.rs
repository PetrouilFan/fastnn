//! Q4_0 GGUF blockwise quantization — 4-bit, 32-elem blocks.
//!
//! Layout per block (`QK4_0 = 32`):
//! ```text
//! ┌────────────────┬──────────────────────────────────┐
//! │ scale: f16×1   │ qs: u8×16  (32 nibbles packed)   │
//! │ 2 bytes        │ 16 bytes                         │
//! └────────────────┴──────────────────────────────────┘
//! total = 18 bytes / 32 weights ≈ 4.5 bits/weight
//! ```
//!
//! Dequantization: `weight = scale * (nibble - 8)` where nibble ∈ [0,15].

const QK4_0: usize = 32;
const BLOCK_SIZE_BYTES: usize = 2 + QK4_0 / 2; // 18

/// Raw data for one Q4_0 block.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ4_0 {
    /// Scale stored as IEEE-754 half.
    pub d: half::f16,
    /// 16 bytes = 32 nibbles.
    pub qs: [u8; QK4_0 / 2],
}

unsafe impl bytemuck::Pod for BlockQ4_0 {}
unsafe impl bytemuck::Zeroable for BlockQ4_0 {}

/// Native Q4_0 tensor.
pub struct Q4_0 {
    shape: [usize; 2],                 // [out, in]
    blocks: Vec<BlockQ4_0>,           // row-major, contiguous
}

impl Q4_0 {
    /// Create from raw half-precision scale + packed nibble bytes.
    ///
    /// `data` is a flat byte slice where every 18 bytes form one block.
    /// `shape` is `[out_features, in_features]`.
    pub fn from_bytes(data: &[u8], shape: [usize; 2]) -> Self {
        let numel = shape[0] * shape[1];
        let num_blocks = numel.div_ceil(QK4_0);
        assert_eq!(
            data.len(),
            num_blocks * BLOCK_SIZE_BYTES,
            "Q4_0: data length {} does not match num_blocks {} × block_bytes {}",
            data.len(),
            num_blocks,
            BLOCK_SIZE_BYTES
        );

        let blocks: Vec<BlockQ4_0> = bytemuck::cast_slice(data).to_vec();

        Q4_0 { shape, blocks }
    }

    /// Zero-copy view from a mmap'd or other contiguous region.
    ///
    /// # Safety
    /// `data` must be aligned to at least 2 bytes and valid for the lifetime
    /// of the returned struct.
    pub unsafe fn from_bytes_unchecked(data: &[u8], shape: [usize; 2]) -> Self {
        let numel = shape[0] * shape[1];
        let num_blocks = numel.div_ceil(QK4_0);
        assert_eq!(data.len(), num_blocks * BLOCK_SIZE_BYTES);
        let blocks = bytemuck::cast_slice(data).to_vec();
        Q4_0 { shape, blocks }
    }

    /// Number of blocks per row.
    fn blocks_per_row(&self) -> usize {
        self.shape[1].div_ceil(QK4_0)
    }
}

impl crate::quants::QuantizedGemm for Q4_0 {
    fn gemv(&self, activation: &[f32], output: &mut [f32]) {
        assert_eq!(activation.len(), self.shape[1]);
        assert_eq!(output.len(), self.shape[0]);

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { gemv_q4_0_avx2(self, activation, output) };
                return;
            }
        }

        gemv_q4_0_scalar(self, activation, output);
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> crate::quants::QuantizedDType {
        crate::quants::QuantizedDType::Q4_0
    }

    fn memory_bytes(&self) -> usize {
        self.blocks.len() * BLOCK_SIZE_BYTES
    }
}

// ---------------------------------------------------------------------------
// Scalar reference implementation (always correct, used for testing)
// ---------------------------------------------------------------------------
pub fn gemv_q4_0_scalar(q: &Q4_0, activation: &[f32], output: &mut [f32]) {
    let blocks_per_row = q.blocks_per_row();
    let in_features = q.shape[1];

    for out_idx in 0..q.shape[0] {
        let mut accum = 0.0f32;
        let row_offset = out_idx * blocks_per_row;

        for block_idx in 0..blocks_per_row {
            let block = &q.blocks[row_offset + block_idx];
            let scale = block.d.to_f32();
            let elem_base = block_idx * QK4_0;

            for byte_idx in 0..(QK4_0 / 2) {
                let packed = block.qs[byte_idx];
                let lo = ((packed & 0x0F) as i32 - 8) as f32;
                let hi = (((packed >> 4) & 0x0F) as i32 - 8) as f32;

                let a_idx0 = elem_base + byte_idx * 2;
                let a_idx1 = a_idx0 + 1;

                if a_idx0 < in_features {
                    accum += scale * lo * activation[a_idx0];
                }
                if a_idx1 < in_features {
                    accum += scale * hi * activation[a_idx1];
                }
            }
        }

        output[out_idx] = accum;
    }
}

// ---------------------------------------------------------------------------
// AVX2 kernel
//
// Strategy: pre-extract all 32 nibbles from one block into an aligned
// temporary [f32; 32], then use the standard AVX2 FMA dot-product routine
// against the activation slice.  This keeps the nibble-extraction code
// simple and correct while still getting full AVX2 throughput on the
// actual dot product.
// ---------------------------------------------------------------------------

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
unsafe fn gemv_q4_0_avx2(q: &Q4_0, activation: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let blocks_per_row = q.blocks_per_row();
    let in_features = q.shape[1];
    // Aligned scratch for unpacking one block at a time
    let mut scratch: [f32; QK4_0] = [0.0; QK4_0];

    for out_idx in 0..q.shape[0] {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let row_offset = out_idx * blocks_per_row;

        for block_idx in 0..blocks_per_row {
            let block = &q.blocks[row_offset + block_idx];
            let scale = block.d.to_f32();
            let elem_base = block_idx * QK4_0;
            let k = (elem_base + QK4_0).min(in_features) - elem_base;

            // Expand nibbles into scratch buffer
            for byte_idx in 0..(QK4_0 / 2) {
                let packed = block.qs[byte_idx];
                scratch[byte_idx * 2] =
                    scale * (((packed & 0x0F) as i32 - 8) as f32);
                scratch[byte_idx * 2 + 1] =
                    scale * (((packed >> 4) as i32 - 8) as f32);
            }

            // Dot product scratch[0..k] · activation[elem_base..elem_base+k]
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

            // Scalar tail (only for trailing elements, not full blocks)
            while j < k {
                let mut tail_acc = [0.0f32; 8];
                std::ptr::copy_nonoverlapping(
                    &acc0 as *const __m256 as *const f32,
                    tail_acc.as_mut_ptr(),
                    8,
                );
                tail_acc[0] += scratch[j] * activation[elem_base + j];
                acc0 = std::ptr::read_unaligned(tail_acc.as_ptr() as *const __m256);
                j += 1;
            }
        }

        // Horizontal sum of all accumulators
        let mut total = 0.0f32;
        let accumulators = [acc0, acc1, acc2, acc3];
        for acc in accumulators {
            // Sum 8 lanes
            let mut lanes: [f32; 8] = [0.0; 8];
            std::ptr::copy_nonoverlapping(
                &acc as *const __m256 as *const f32,
                lanes.as_mut_ptr(),
                8,
            );
            for lane in lanes {
                total += lane;
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
    fn test_q4_0_simple_gemv() {
        // 1 output row, 32 input features = 1 block.
        let scale = half::f16::from_f32_const(1.0);
        let mut qs = [0u8; 16];
        for i in 0..8 {
            qs[i] = i as u8 | (((i + 8) as u8) << 4);
        }

        let block = BlockQ4_0 { d: scale, qs };
        let q4 = Q4_0 {
            shape: [1, 32],
            blocks: vec![block],
        };

        let act = vec![1.0f32; 32];
        let mut out = vec![0.0f32; 1];
        q4.gemv(&act, &mut out);

        // Elements 0..15: [0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15]
        // After centering (-8): [-8,0,-7,1,-6,2,-5,3,-4,4,-3,5,-2,6,-1,7]
        // Sum of elements 0..15:
        //   Pairs: (-8,0)=-8, (-7,1)=-6, (-6,2)=-4, (-5,3)=-2, (-4,4)=0,
        //          (-3,5)=2, (-2,6)=4, (-1,7)=6
        //   Total = -8 + -6 + -4 + -2 + 0 + 2 + 4 + 6 = -8
        // Elements 16..31: all 0→-8, 16 * (-8) = -128
        // Total with activation=1.0: -8 + (-128) = -136
        let expected = -136.0f32;
        let diff = (out[0] - expected).abs();
        assert!(
            diff < 1e-3,
            "expected {}, got {} (diff={})",
            expected, out[0], diff
        );
    }

    #[test]
    fn test_q4_0_shape_memory() {
        let bs = crate::quants::QuantizedDType::Q4_0.block_bytes().unwrap();
        let shape = [64usize, 128];
        let num_blocks: usize = (64 * 128usize).div_ceil(32);
        let data = vec![0u8; num_blocks * bs];
        let q4 = Q4_0::from_bytes(&data, shape);

        assert_eq!(q4.out_features(), 64);
        assert_eq!(q4.in_features(), 128);
        assert_eq!(q4.memory_bytes(), num_blocks * bs);
    }

    #[test]
    fn test_q4_0_multirow_gemv_scalar_vs_avx2() {
        let shape: [usize; 2] = [16, 64]; // 16 rows, 64 cols → 2 blocks/row
        let num_blocks: usize = (16usize * 64usize).div_ceil(32);
        let mut data = vec![0u8; num_blocks * BLOCK_SIZE_BYTES];

        for i in 0..num_blocks {
            let offset = i * BLOCK_SIZE_BYTES;
            let scale = half::f16::from_f32_const(1.0 + (i as f32) * 0.01);
            let scale_bytes: [u8; 2] = scale.to_bits().to_le_bytes();
            data[offset] = scale_bytes[0];
            data[offset + 1] = scale_bytes[1];
            for b in 0..16 {
                data[offset + 2 + b] = ((i * 16 + b) % 256) as u8;
            }
        }

        let q4 = Q4_0::from_bytes(&data, shape);
        let act: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin()).collect();

        let mut out_scalar = vec![0.0f32; 16];
        gemv_q4_0_scalar(&q4, &act, &mut out_scalar);

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") {
            let mut out_avx2 = vec![0.0f32; 16];
            unsafe { gemv_q4_0_avx2(&q4, &act, &mut out_avx2) };
            for (i, (&scalar, &avx)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                let diff = (scalar - avx).abs();
                assert!(
                    diff < 1e-3,
                    "row {}: scalar={}, avx={}, diff={}",
                    i, scalar, avx, diff
                );
            }
        }
    }
}
