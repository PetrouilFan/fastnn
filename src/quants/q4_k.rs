//! Q4_K GGML blockwise quantization — 4-bit, 256-elem super-blocks.
//!
//! Layout per super-block (`QK4_K = 256`):
//! ```text
//! ┌────────┬────────┬──────────┬──────────┐
//! │ d: f16 │dmin:f16│scales:u8×12│ qs:u8×128 │
//! │ 2 B │ 2 B │ 12 B │ 128 B │
//! └────────┴────────┴──────────┴──────────┘
//! total = 144 bytes / 256 weights ≈ 4.5 bits/weight
//! ```
//!
//! Dequantization processes 64 elements per group:
//! - First 32:  `d * sc[is] * (qs & 0xF)  - dmin * m[is]`
//! - Next  32:  `d * sc[is+1] * (qs >> 4)  - dmin * m[is+1]`
//!
//! Scales are packed via `get_scale_min_k4` into 12 bytes.

const QK4_K: usize = 256;
const K_SCALE: usize = 32;
const NUM_SUB_BLOCKS: usize = QK4_K / K_SCALE;
const BLOCK_SIZE_BYTES: usize = 2 + 2 + 12 + 128;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ4_K {
    pub d: half::f16,
    pub dmin: half::f16,
    pub scales: [u8; 12],
    pub qs: [u8; QK4_K / 2],
}

unsafe impl bytemuck::Pod for BlockQ4_K {}
unsafe impl bytemuck::Zeroable for BlockQ4_K {}

impl BlockQ4_K {
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

    #[inline]
    fn dequantize(&self, out: &mut [f32; QK4_K]) {
        let d = self.d.to_f32();
        let dmin = self.dmin.to_f32();
        let q = &self.qs;
        let mut is = 0usize;
        let mut y_idx = 0usize;
        let mut ql_offset = 0usize;

        for _j in (0..QK4_K).step_by(64) {
            let (sc1, m1) = Self::get_scale_min_k4(is, &self.scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;

            let (sc2, m2) = Self::get_scale_min_k4(is + 1, &self.scales);
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            for l in 0..32 {
                out[y_idx] = d1 * (q[ql_offset + l] & 0x0F) as f32 - m1;
                y_idx += 1;
            }
            for l in 0..32 {
                out[y_idx] = d2 * ((q[ql_offset + l] >> 4) & 0x0F) as f32 - m2;
                y_idx += 1;
            }

            ql_offset += 32;
            is += 2;
        }
    }
}

pub struct Q4_K {
    shape: [usize; 2],
    blocks: Vec<BlockQ4_K>,
}

impl Q4_K {
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
            let qs = &block.qs;
            let elem_base = block_idx * QK4_K;
            let k = (elem_base + QK4_K).min(in_features) - elem_base;

            let mut is = 0usize;
            let mut w_idx = 0usize;
            let mut ql_offset = 0usize;

            for _j in (0..QK4_K).step_by(64) {
                let (sc1, m1) = BlockQ4_K::get_scale_min_k4(is, &block.scales);
                let d1 = d * sc1 as f32;
                let m1_val = dmin * m1 as f32;

                let (sc2, m2) = BlockQ4_K::get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc2 as f32;
                let m2_val = dmin * m2 as f32;

                for l in 0..32 {
                    let a_idx = elem_base + w_idx;
                    if a_idx < in_features {
                        let w = d1 * (qs[ql_offset + l] & 0x0F) as f32 - m1_val;
                        accum += w * activation[a_idx];
                    }
                    w_idx += 1;
                }
                for l in 0..32 {
                    let a_idx = elem_base + w_idx;
                    if a_idx < in_features {
                        let w = d2 * ((qs[ql_offset + l] >> 4) & 0x0F) as f32 - m2_val;
                        accum += w * activation[a_idx];
                    }
                    w_idx += 1;
                }

                ql_offset += 32;
                is += 2;
            }
        }

        output[out_idx] = accum;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
unsafe fn gemv_q4_k_avx2(q: &Q4_K, activation: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let blocks_per_row = q.blocks_per_row();
    let in_features = q.shape[1];

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

            let d = _mm256_set1_ps(block.d.to_f32());
            let dmin = _mm256_set1_ps(block.dmin.to_f32());

            // Load scales: 12 bytes packed as described in get_scale_min_k4
            // We need to expand them into vectors for each group of 32 weights
            let scales = block.scales;

            let mut j = 0usize;
            let mut is = 0usize;
            let mut ql_offset = 0usize;

            // Process 64 elements at a time (2 groups of 32)
            while j + 64 <= k {
                // Get scales for this group of 64
                let (sc1, m1) = BlockQ4_K::get_scale_min_k4(is, &scales);
                let (sc2, m2) = BlockQ4_K::get_scale_min_k4(is + 1, &scales);
                let d1 = _mm256_mul_ps(d, _mm256_set1_ps(sc1 as f32));
                let dmin1 = _mm256_mul_ps(dmin, _mm256_set1_ps(m1 as f32));
                let d2 = _mm256_mul_ps(d, _mm256_set1_ps(sc2 as f32));
                let dmin2 = _mm256_mul_ps(dmin, _mm256_set1_ps(m2 as f32));

                // Load 32 bytes of quantized weights (64 4-bit values)
                let qs = &block.qs[ql_offset..ql_offset + 32];

                // Load 32 bytes of activations as 8 floats each = 256 bytes total
                // We'll process in 4 chunks of 8 floats
                for chunk in 0..4 {
                    let a0 = _mm256_loadu_ps(activation.as_ptr().add(elem_base + j + chunk * 8));

                    // Extract low and high 4-bit values from 8 bytes
                    let q8 = _mm_loadl_epi64(qs.as_ptr().add(chunk * 8) as *const __m128i);
                    let q8_32 = _mm256_cvtepu8_epi32(q8);

                    // Low 4 bits: q8_32 & 0x0F
                    let q_low = _mm256_and_si256(q8_32, _mm256_set1_epi32(0x0F));
                    // High 4 bits: (q8_32 >> 4) & 0x0F
                    let q_high = _mm256_and_si256(_mm256_srli_epi32(q8_32, 4), _mm256_set1_epi32(0x0F));

                    // Convert to f32 vectors
                    let q_low_f = _mm256_cvtepi32_ps(q_low);
                    let q_high_f = _mm256_cvtepi32_ps(q_high);

                    // First group (low nibbles): d1 * q - dmin1
                    let w1 = _mm256_fmadd_ps(d1, q_low_f, _mm256_sub_ps(_mm256_setzero_ps(), dmin1));
                    acc0 = _mm256_fmadd_ps(w1, a0, acc0);

                    // Second group (high nibbles): d2 * q - dmin2  
                    let w2 = _mm256_fmadd_ps(d2, q_high_f, _mm256_sub_ps(_mm256_setzero_ps(), dmin2));
                    acc1 = _mm256_fmadd_ps(w2, a0, acc1);
                }

                j += 64;
                is += 2;
                ql_offset += 32;
            }

            // Handle remaining elements with scalar fallback
            while j < k {
                let (sc, m) = BlockQ4_K::get_scale_min_k4(is, &scales);
                let mut d_arr: [f32; 8] = [0.0; 8];
                let mut dmin_arr: [f32; 8] = [0.0; 8];
                _mm256_storeu_ps(d_arr.as_mut_ptr(), d);
                _mm256_storeu_ps(dmin_arr.as_mut_ptr(), dmin);
                let d_s = d_arr[0] * sc as f32;
                let m_s = dmin_arr[0] * m as f32;

                let q_byte = block.qs[ql_offset + j / 2];
                let q_val = if j % 2 == 0 {
                    (q_byte & 0x0F) as f32
                } else {
                    (q_byte >> 4) as f32
                };
                let w = d_s * q_val - m_s;
                let a = activation[elem_base + j];

                let mut arr: [f32; 8] = [0.0; 8];
                _mm256_storeu_ps(arr.as_mut_ptr(), acc2);
                arr[0] += w * a;
                acc2 = _mm256_loadu_ps(arr.as_ptr());

                j += 1;
                if j % 64 == 0 { is += 2; ql_offset += 32; }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quants::QuantizedGemm;

    fn encode_scales_k4(sc: &[u8; 8], m: &[u8; 8]) -> [u8; 12] {
        let mut s = [0u8; 12];
        for j in 0..4 {
            s[j] = (sc[j] & 63) | ((sc[j + 4] >> 4) << 6);
            s[j + 4] = (m[j] & 63) | ((m[j + 4] >> 4) << 6);
        }
        s[8] = (sc[4] & 0x0F) | ((m[4] & 0x0F) << 4);
        s[9] = (sc[5] & 0x0F) | ((m[5] & 0x0F) << 4);
        s[10] = (sc[6] & 0x0F) | ((m[6] & 0x0F) << 4);
        s[11] = (sc[7] & 0x0F) | ((m[7] & 0x0F) << 4);
        s
    }

    #[test]
    fn test_q4_k_get_scale_min_k4() {
        let sc = [1u8, 2, 3, 4, 9, 10, 11, 12];
        let m = [5u8, 6, 7, 8, 13, 14, 15, 16];
        let scales = encode_scales_k4(&sc, &m);
        for j in 0..8 {
            let (s, mi) = BlockQ4_K::get_scale_min_k4(j, &scales);
            assert_eq!(s, sc[j], "scale[{}] expected {}, got {}", j, sc[j], s);
            assert_eq!(mi, m[j], "min[{}] expected {}, got {}", j, m[j], mi);
        }
    }

    #[test]
    fn test_q4_k_dequantize_zeros() {
        let sc = [1u8; 8];
        let m = [0u8; 8];
        let scales = encode_scales_k4(&sc, &m);
        let block = BlockQ4_K {
            d: half::f16::from_f32_const(1.0),
            dmin: half::f16::from_f32_const(0.0),
            scales,
            qs: [0u8; 128],
        };
        let mut out = [0.0f32; QK4_K];
        block.dequantize(&mut out);
        for (i, &val) in out.iter().enumerate() {
            assert!((val - 0.0).abs() < 1e-4, "index {}: expected 0, got {}", i, val);
        }
    }

    #[test]
    fn test_q4_k_dequantize_nibble5() {
        let sc = [1u8; 8];
        let m = [1u8; 8];
        let scales = encode_scales_k4(&sc, &m);
        let block = BlockQ4_K {
            d: half::f16::from_f32_const(2.0),
            dmin: half::f16::from_f32_const(1.0),
            scales,
            qs: [0x55u8; 128],
        };
        let mut out = [0.0f32; QK4_K];
        block.dequantize(&mut out);
        for (i, &val) in out.iter().enumerate() {
            let expected = 2.0 * 1.0 * 5.0 - 1.0 * 1.0;
            assert!((val - expected).abs() < 1e-4, "index {}: expected {}, got {}", i, expected, val);
        }
    }

    #[test]
    fn test_q4_k_simple_gemv() {
        let sc = [1u8; 8];
        let m = [1u8; 8];
        let scales = encode_scales_k4(&sc, &m);
        let block = BlockQ4_K {
            d: half::f16::from_f32_const(1.0),
            dmin: half::f16::from_f32_const(1.0),
            scales,
            qs: [0u8; 128],
        };
        let q4 = Q4_K {
            shape: [1, 256],
            blocks: vec![block],
        };
        let act = vec![1.0f32; 256];
        let mut out = vec![0.0f32; 1];
        q4.gemv(&act, &mut out);
        let expected = -256.0f32;
        assert!((out[0] - expected).abs() < 1e-2, "expected {}, got {}", expected, out[0]);
    }

    #[test]
    fn test_q4_k_scalar_vs_avx2() {
        let shape: [usize; 2] = [8, 512];
        let num_blocks: usize = (8usize * 512usize).div_ceil(QK4_K);
        let mut data = vec![0u8; num_blocks * BLOCK_SIZE_BYTES];

        for i in 0..num_blocks {
            let offset = i * BLOCK_SIZE_BYTES;
            let d = half::f16::from_f32_const(1.0 + (i as f32) * 0.001);
            let d_bytes: [u8; 2] = d.to_bits().to_le_bytes();
            data[offset] = d_bytes[0];
            data[offset + 1] = d_bytes[1];
            let dmin = half::f16::from_f32_const(0.5);
            let dmin_bytes: [u8; 2] = dmin.to_bits().to_le_bytes();
            data[offset + 2] = dmin_bytes[0];
            data[offset + 3] = dmin_bytes[1];
            for s in 0..12 {
                data[offset + 4 + s] = ((i * 12 + s) % 64) as u8;
            }
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
                assert!(diff < 0.1, "row {}: scalar={}, avx={}, diff={}", i, scalar, avx, diff);
            }
        }
    }
}
