use crate::quants::{QuantizedDType, QuantizedGemm};
use half::f16;

const QK5_K: usize = 256;
const BLOCK_SIZE_BYTES: usize = 2 + 2 + 12 + 32 + 128; // 176 bytes

#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ5K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; 12],
    pub qh: [u8; 32],
    pub qs: [u8; QK5_K / 2],
}

unsafe impl bytemuck::Pod for BlockQ5K {}
unsafe impl bytemuck::Zeroable for BlockQ5K {}

impl BlockQ5K {
    #[inline]
    fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
        if j < 4 {
            let sc = q[j] & 63;
            let m = q[j + 4] & 63;
            (sc, m)
        } else {
            let sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
            let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
            (sc, m)
        }
    }

    fn dequantize_block(&self) -> [f32; QK5_K] {
        let d = self.d.to_f32();
        let dmin = self.dmin.to_f32();
        let ql = &self.qs;
        let qh = &self.qh;

        let mut result = [0.0f32; QK5_K];
        let mut is = 0usize;
        let mut y_idx = 0usize;
        let mut ql_offset = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for _j in (0..QK5_K).step_by(64) {
            let (sc1, m1) = Self::get_scale_min_k4(is, &self.scales);
            let d1 = d * sc1 as f32;
            let m1_val = dmin * m1 as f32;

            let (sc2, m2) = Self::get_scale_min_k4(is + 1, &self.scales);
            let d2 = d * sc2 as f32;
            let m2_val = dmin * m2 as f32;

            for l in 0..32 {
                let lo = (ql[ql_offset + l] & 0x0F) as f32;
                let h_bit = if qh[l] & u1 != 0 { 16.0f32 } else { 0.0f32 };
                result[y_idx] = d1 * (lo + h_bit) - m1_val;
                y_idx += 1;
            }
            for l in 0..32 {
                let hi = ((ql[ql_offset + l] >> 4) & 0x0F) as f32;
                let h_bit = if qh[l] & u2 != 0 { 16.0f32 } else { 0.0f32 };
                result[y_idx] = d2 * (hi + h_bit) - m2_val;
                y_idx += 1;
            }

            ql_offset += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }

        result
    }
}

pub struct Q5K {
    data: Vec<u8>,
    shape: [usize; 2],
}

impl Q5K {
    pub fn from_bytes(data: &[u8], shape: [usize; 2]) -> Self {
        Q5K {
            data: data.to_vec(),
            shape,
        }
    }
}

impl QuantizedGemm for Q5K {
    fn gemv(&self, activation: &[f32], output: &mut [f32]) {
        let out_feat = self.shape[0];
        let in_feat = self.shape[1];
        let n_blocks = in_feat / QK5_K;
        let blocks: &[BlockQ5K] = bytemuck::cast_slice(&self.data);

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { gemv_q5_k_avx2(self, activation, output) };
                return;
            }
        }

        // Scalar fallback
        for row in 0..out_feat {
            let block_offset = row * n_blocks;
            let mut sum = 0.0f32;
            for b in 0..n_blocks {
                let block = &blocks[block_offset + b];
                let weights = block.dequantize_block();
                let act_offset = b * QK5_K;
                for j in 0..QK5_K {
                    sum += weights[j] * activation[act_offset + j];
                }
            }
            output[row] = sum;
        }
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> QuantizedDType {
        QuantizedDType::Q5_K
    }

    fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    fn row(&self, row_idx: usize) -> Vec<f32> {
        let in_feat = self.shape[1];
        let n_blocks = in_feat / QK5_K;
        let blocks: &[BlockQ5K] = bytemuck::cast_slice(&self.data);
        let block_offset = row_idx * n_blocks;

        let mut result = Vec::with_capacity(in_feat);
        for b in 0..n_blocks {
            let block = &blocks[block_offset + b];
            let weights = block.dequantize_block();
            result.extend_from_slice(&weights);
        }
        result
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
unsafe fn gemv_q5_k_avx2(q: &Q5K, activation: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let in_features = q.shape[1];
    let n_blocks = in_features / QK5_K;
    let blocks: &[BlockQ5K] = bytemuck::cast_slice(&q.data);

    for out_idx in 0..q.shape[0] {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let row_offset = out_idx * n_blocks;

        for block_idx in 0..n_blocks {
            let block = &blocks[row_offset + block_idx];
            let elem_base = block_idx * QK5_K;
            let k = (elem_base + QK5_K).min(in_features) - elem_base;

            let weights = block.dequantize_block();
            let mut j = 0usize;
            while j + 32 <= k {
                let w0 = _mm256_loadu_ps(weights.as_ptr().add(j));
                let a0 = _mm256_loadu_ps(activation.as_ptr().add(elem_base + j));
                acc0 = _mm256_fmadd_ps(w0, a0, acc0);

                let w1 = _mm256_loadu_ps(weights.as_ptr().add(j + 8));
                let a1 = _mm256_loadu_ps(activation.as_ptr().add(elem_base + j + 8));
                acc1 = _mm256_fmadd_ps(w1, a1, acc1);

                let w2 = _mm256_loadu_ps(weights.as_ptr().add(j + 16));
                let a2 = _mm256_loadu_ps(activation.as_ptr().add(elem_base + j + 16));
                acc2 = _mm256_fmadd_ps(w2, a2, acc2);

                let w3 = _mm256_loadu_ps(weights.as_ptr().add(j + 24));
                let a3 = _mm256_loadu_ps(activation.as_ptr().add(elem_base + j + 24));
                acc3 = _mm256_fmadd_ps(w3, a3, acc3);

                j += 32;
            }
            while j + 8 <= k {
                let w = _mm256_loadu_ps(weights.as_ptr().add(j));
                let a = _mm256_loadu_ps(activation.as_ptr().add(elem_base + j));
                acc0 = _mm256_fmadd_ps(w, a, acc0);
                j += 8;
            }
            if j < k {
                // Handle remaining elements with scalar
                let mut arr: [f32; 8] = [0.0; 8];
                _mm256_storeu_ps(arr.as_mut_ptr(), acc0);
                for idx in 0..(k - j) {
                    arr[idx] += weights[j + idx] * activation[elem_base + j + idx];
                }
                acc0 = _mm256_loadu_ps(arr.as_ptr());
            }
        }

        let mut total = 0.0f32;
        for acc in [acc0, acc1, acc2, acc3] {
            let mut arr: [f32; 8] = [0.0; 8];
            _mm256_storeu_ps(arr.as_mut_ptr(), acc);
            for v in arr {
                total += v;
            }
        }
        output[out_idx] = total;
    }
}
