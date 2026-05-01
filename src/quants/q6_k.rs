//! Q6_K GGUF blockwise quantization — 6-bit, 256-elem super-blocks.
//!
//! Layout per super-block (`QK6_K = 256`):
//! ```text
//! ┌────────────────┬────────────┬──────────────┬───────┐
//! │ ql: u8×128     │ qh: u8×64  │scales:i8×16 │ d:f16 │
//! │ 128 B          │ 64 B       │ 16 B         │ 2 B   │
//! └────────────────┴────────────┴──────────────┴───────┘
//! total = 210 bytes / 256 weights ≈ 6.5625 bits/weight
//! ```
//!
//! Each super-block contains 128-element halves processed in two passes.
//! The 256 weights are split into two 128-element groups.
//!
//! - `d`      : f16 super-scale applied to all weights.
//! - `scales` : 16 int8 scale factors, 8 per 128-element half.
//! - `ql`     : 128 bytes = lower 4 bits of each 6-bit quant.
//! - `qh`     : 64 bytes = upper 2 bits of each 6-bit quant.
//!
//! Dequantization for element `l` in 128-element half `h`:
//! ```text
//! is = l / 16          (group index within the half)
//! q  = (ql bits | qh bits) - 32    (6-bit signed, centered at 0)
//! w  = d * scales[h*8 + is + offset] * q
//! ```

const QK6_K: usize = 256;
const BLOCK_SIZE_BYTES: usize = 128 + 64 + 16 + 2;

const _: () = assert!(std::mem::size_of::<BlockQ6K>() == BLOCK_SIZE_BYTES);

#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ6K {
    pub ql: [u8; QK6_K / 2],
    pub qh: [u8; QK6_K / 4],
    pub scales: [i8; QK6_K / 16],
    pub d: half::f16,
}

unsafe impl bytemuck::Pod for BlockQ6K {}
unsafe impl bytemuck::Zeroable for BlockQ6K {}

impl BlockQ6K {
    #[inline]
    fn dequantize(&self, out: &mut [f32; QK6_K]) {
        let d = self.d.to_f32();
        let ql = &self.ql;
        let qh = &self.qh;
        let sc = &self.scales;

        let mut out_idx = 0usize;
        let mut ql_offset = 0usize;
        let mut qh_offset = 0usize;
        let mut sc_offset = 0usize;

        for _half in 0..2 {
            for l in 0..32usize {
                let is = l / 16;
                let q1 = (((ql[ql_offset + l] & 0x0F) as i32)
                    | (((qh[qh_offset + l] as i32) & 3) << 4))
                    - 32;
                let q2 = (((ql[ql_offset + l + 32] & 0x0F) as i32)
                    | (((qh[qh_offset + l] as i32 >> 2) & 3) << 4))
                    - 32;
                let q3 = (((ql[ql_offset + l] >> 4) as i32)
                    | (((qh[qh_offset + l] as i32 >> 4) & 3) << 4))
                    - 32;
                let q4 = (((ql[ql_offset + l + 32] >> 4) as i32)
                    | (((qh[qh_offset + l] as i32 >> 6) & 3) << 4))
                    - 32;

                out[out_idx + l] = d * (sc[sc_offset + is + 0] as f32) * (q1 as f32);
                out[out_idx + l + 32] = d * (sc[sc_offset + is + 2] as f32) * (q2 as f32);
                out[out_idx + l + 64] = d * (sc[sc_offset + is + 4] as f32) * (q3 as f32);
                out[out_idx + l + 96] = d * (sc[sc_offset + is + 6] as f32) * (q4 as f32);
            }
            out_idx += 128;
            ql_offset += 64;
            qh_offset += 32;
            sc_offset += 8;
        }
    }
}

pub struct Q6K {
    shape: [usize; 2],
    blocks: Vec<BlockQ6K>,
}

impl Q6K {
    pub fn from_bytes(data: &[u8], shape: [usize; 2]) -> Self {
        let numel = shape[0] * shape[1];
        let num_blocks = numel.div_ceil(QK6_K);
        assert_eq!(
            data.len(),
            num_blocks * BLOCK_SIZE_BYTES,
            "Q6_K: data length {} does not match expected {} bytes",
            data.len(),
            num_blocks * BLOCK_SIZE_BYTES
        );
        let blocks: Vec<BlockQ6K> = bytemuck::cast_slice(data).to_vec();
        Q6K { shape, blocks }
    }

    fn blocks_per_row(&self) -> usize {
        self.shape[1].div_ceil(QK6_K)
    }

    pub fn row(&self, row_idx: usize) -> Vec<f32> {
        let blocks_per_row = self.blocks_per_row();
        let block_start = row_idx * blocks_per_row;
        let mut result = Vec::with_capacity(self.shape[1]);

        for block_idx in 0..blocks_per_row {
            let mut scratch = [0.0f32; QK6_K];
            self.blocks[block_start + block_idx].dequantize(&mut scratch);
            result.extend_from_slice(&scratch[..]);
        }

        result.truncate(self.shape[1]);
        result
    }
}

impl crate::quants::QuantizedGemm for Q6K {
    fn gemv(&self, activation: &[f32], output: &mut [f32]) {
        assert_eq!(activation.len(), self.shape[1]);
        assert_eq!(output.len(), self.shape[0]);

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { gemv_q6_k_avx2(self, activation, output) };
                return;
            }
        }

        gemv_q6_k_scalar(self, activation, output);
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> crate::quants::QuantizedDType {
        crate::quants::QuantizedDType::Q6_K
    }

    fn memory_bytes(&self) -> usize {
        self.blocks.len() * BLOCK_SIZE_BYTES
    }

    fn row(&self, row_idx: usize) -> Vec<f32> {
        Q6K::row(self, row_idx)
    }
}

pub fn gemv_q6_k_scalar(q: &Q6K, activation: &[f32], output: &mut [f32]) {
    let blocks_per_row = q.blocks_per_row();
    let in_features = q.shape[1];

    for out_idx in 0..q.shape[0] {
        let mut accum = 0.0f32;
        let row_offset = out_idx * blocks_per_row;

        for block_idx in 0..blocks_per_row {
            let block = &q.blocks[row_offset + block_idx];
            let d = block.d.to_f32();
            let ql = &block.ql;
            let qh = &block.qh;
            let sc = &block.scales;
            let elem_base = block_idx * QK6_K;

            let mut ql_offset = 0usize;
            let mut qh_offset = 0usize;
            let mut sc_offset = 0usize;

            for _half in 0..2 {
                for l in 0..32usize {
                    let is = l / 16;
                    let q1 = (((ql[ql_offset + l] & 0x0F) as i32)
                        | (((qh[qh_offset + l] as i32) & 3) << 4))
                        - 32;
                    let q2 = (((ql[ql_offset + l + 32] & 0x0F) as i32)
                        | (((qh[qh_offset + l] as i32 >> 2) & 3) << 4))
                        - 32;
                    let q3 = (((ql[ql_offset + l] >> 4) as i32)
                        | (((qh[qh_offset + l] as i32 >> 4) & 3) << 4))
                        - 32;
                    let q4 = (((ql[ql_offset + l + 32] >> 4) as i32)
                        | (((qh[qh_offset + l] as i32 >> 6) & 3) << 4))
                        - 32;

                    let idx_base = elem_base + _half * 128 + l;
                    let s1 = d * (sc[sc_offset + is + 0] as f32);
                    let s2 = d * (sc[sc_offset + is + 2] as f32);
                    let s3 = d * (sc[sc_offset + is + 4] as f32);
                    let s4 = d * (sc[sc_offset + is + 6] as f32);

                    if idx_base < in_features {
                        accum += s1 * (q1 as f32) * activation[idx_base];
                    }
                    if idx_base + 32 < in_features {
                        accum += s2 * (q2 as f32) * activation[idx_base + 32];
                    }
                    if idx_base + 64 < in_features {
                        accum += s3 * (q3 as f32) * activation[idx_base + 64];
                    }
                    if idx_base + 96 < in_features {
                        accum += s4 * (q4 as f32) * activation[idx_base + 96];
                    }
                }
                ql_offset += 64;
                qh_offset += 32;
                sc_offset += 8;
            }
        }

        output[out_idx] = accum;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
unsafe fn gemv_q6_k_avx2(q: &Q6K, activation: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let blocks_per_row = q.blocks_per_row();
    let in_features = q.shape[1];
    let mut scratch: [f32; QK6_K] = [0.0; QK6_K];

    for out_idx in 0..q.shape[0] {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let row_offset = out_idx * blocks_per_row;

        for block_idx in 0..blocks_per_row {
            let block = &q.blocks[row_offset + block_idx];
            let elem_base = block_idx * QK6_K;
            let k = (elem_base + QK6_K).min(in_features) - elem_base;

            block.dequantize(&mut scratch);

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
                let mut arr: [f32; 8] = [0.0; 8];
                std::ptr::copy_nonoverlapping(
                    &acc0 as *const __m256 as *const f32,
                    arr.as_mut_ptr(),
                    8,
                );
                arr[0] += scratch[j] * activation[elem_base + j];
                acc0 = std::ptr::read_unaligned(arr.as_ptr() as *const __m256);
                j += 1;
            }
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quants::QuantizedGemm;

    fn make_simple_block() -> BlockQ6K {
        let mut block = BlockQ6K {
            ql: [0u8; 128],
            qh: [0u8; 64],
            scales: [0i8; 16],
            d: half::f16::from_f32_const(1.0),
        };

        for i in 0..16 {
            block.scales[i] = 1;
        }

        block
    }

    #[test]
    fn test_q6_k_all_zeros() {
        let block = make_simple_block();
        let mut out = [0.0f32; QK6_K];
        block.dequantize(&mut out);

        for (i, &val) in out.iter().enumerate() {
            let expected = 1.0 * 1.0 * (-32.0);
            let diff = (val - expected).abs();
            assert!(
                diff < 1e-4,
                "index {}: expected {}, got {} (diff={})",
                i,
                expected,
                val,
                diff
            );
        }
    }

    #[test]
    fn test_q6_known_values() {
        let mut block = BlockQ6K {
            ql: [0u8; 128],
            qh: [0u8; 64],
            scales: [0i8; 16],
            d: half::f16::from_f32_const(2.0),
        };

        for i in 0..16 {
            block.scales[i] = 1;
        }

        block.ql[0] = 0x23;
        block.qh[0] = 0x01;

        let q1_low = (0x23u32 & 0x0F) | ((0x01u32 & 3) << 4);
        let expected_q1 = q1_low as i32 - 32;
        let expected_val = 2.0f32 * 1.0f32 * (expected_q1 as f32);

        let mut out = [0.0f32; QK6_K];
        block.dequantize(&mut out);

        let diff = (out[0] - expected_val).abs();
        assert!(
            diff < 1e-4,
            "element 0: expected {}, got {} (diff={})",
            expected_val,
            out[0],
            diff
        );
    }

    #[test]
    fn test_q6_k_simple_gemv() {
        let block = make_simple_block();
        let q6 = Q6K {
            shape: [1, 256],
            blocks: vec![block],
        };

        let act = vec![1.0f32; 256];
        let mut out = vec![0.0f32; 1];
        q6.gemv(&act, &mut out);

        let mut expected_sum = 0.0f32;
        let mut scratch = [0.0f32; QK6_K];
        block.dequantize(&mut scratch);
        for v in &scratch[..256] {
            expected_sum += v;
        }

        let diff = (out[0] - expected_sum).abs();
        assert!(
            diff < 1e-4,
            "expected {}, got {} (diff={})",
            expected_sum,
            out[0],
            diff
        );
    }

    #[test]
    fn test_q6_k_scalar_vs_avx2() {
        let shape: [usize; 2] = [4, 512];
        let num_blocks: usize = (4usize * 512usize).div_ceil(QK6_K);
        let mut data = vec![0u8; num_blocks * BLOCK_SIZE_BYTES];

        for i in 0..num_blocks {
            let offset = i * BLOCK_SIZE_BYTES;
            let d = half::f16::from_f32_const(0.5 + (i as f32) * 0.01);
            let d_bytes: [u8; 2] = d.to_bits().to_le_bytes();
            data[offset] = d_bytes[0];
            data[offset + 1] = d_bytes[1];
            for s in 0..16 {
                data[offset + 128 + 64 + s] = ((i * 16 + s) % 64) as u8;
            }
            for q in 0..128 {
                data[offset + 2 + q] = ((i * 128 + q) % 256) as u8;
            }
            for q in 0..64 {
                data[offset + 2 + 128 + q] = ((i * 64 + q) % 256) as u8;
            }
        }

        let q6 = Q6K::from_bytes(&data, shape);
        let act: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();

        let mut out_scalar = vec![0.0f32; 4];
        gemv_q6_k_scalar(&q6, &act, &mut out_scalar);

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") {
            let mut out_avx2 = vec![0.0f32; 4];
            unsafe { gemv_q6_k_avx2(&q6, &act, &mut out_avx2) };
            for (i, (&scalar, &avx)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                let diff = (scalar - avx).abs();
                assert!(
                    diff < 1e-2,
                    "row {}: scalar={}, avx={}, diff={}",
                    i,
                    scalar,
                    avx,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_q6_k_row() {
        let shape: [usize; 2] = [2, 256];
        let num_blocks: usize = 2;
        let mut data = vec![0u8; num_blocks * BLOCK_SIZE_BYTES];

        for i in 0..num_blocks {
            let offset = i * BLOCK_SIZE_BYTES;
            let d = half::f16::from_f32_const(1.0);
            let d_bytes: [u8; 2] = d.to_bits().to_le_bytes();
            data[offset] = d_bytes[0];
            data[offset + 1] = d_bytes[1];
            for s in 0..16 {
                data[offset + 128 + 64 + s] = 1i8 as u8;
            }
        }

        let q6 = Q6K::from_bytes(&data, shape);
        let row0 = q6.row(0);
        let row1 = q6.row(1);

        assert_eq!(row0.len(), 256);
        assert_eq!(row1.len(), 256);
    }

    #[test]
    fn test_q6_k_memory_bytes() {
        let shape: [usize; 2] = [8, 512];
        let num_blocks: usize = (8usize * 512usize).div_ceil(QK6_K);
        let data = vec![0u8; num_blocks * BLOCK_SIZE_BYTES];
        let q6 = Q6K::from_bytes(&data, shape);

        assert_eq!(q6.memory_bytes(), num_blocks * BLOCK_SIZE_BYTES);
        assert_eq!(q6.out_features(), 8);
        assert_eq!(q6.in_features(), 512);
    }
}