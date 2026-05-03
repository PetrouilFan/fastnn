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
            output[row] += sum;
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
