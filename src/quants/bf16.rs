use crate::quants::{QuantizedDType, QuantizedGemm};

#[derive(Clone)]
pub struct BF16 {
    data: Vec<u16>,
    shape: [usize; 2],
}

impl BF16 {
    pub fn from_bytes(raw: &[u8], shape: [usize; 2]) -> Self {
        let n_elements = shape[0] * shape[1];
        let mut data = Vec::with_capacity(n_elements);
        for i in 0..n_elements {
            let offset = i * 2;
            if offset + 2 <= raw.len() {
                data.push(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
            } else {
                data.push(0);
            }
        }
        BF16 { data, shape }
    }

    fn bf16_to_f32(bf16: u16) -> f32 {
        let f32_bits = (bf16 as u32) << 16;
        f32::from_bits(f32_bits)
    }

    fn f32_to_bf16(f: f32) -> u16 {
        let f32_bits = f.to_bits();
        (f32_bits >> 16) as u16
    }
}

impl QuantizedGemm for BF16 {
    fn gemv(&self, activation: &[f32], output: &mut [f32]) {
        let out_feat = self.shape[0];
        let in_feat = self.shape[1];

        for row in 0..out_feat {
            let row_offset = row * in_feat;
            let mut sum = 0.0f32;
            for col in 0..in_feat {
                let w = Self::bf16_to_f32(self.data[row_offset + col]);
                sum += w * activation[col];
            }
            output[row] += sum;
        }
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> QuantizedDType {
        QuantizedDType::Bf16
    }

    fn memory_bytes(&self) -> usize {
        self.data.len() * 2
    }

    fn row(&self, row_idx: usize) -> Vec<f32> {
        let in_feat = self.shape[1];
        let offset = row_idx * in_feat;
        self.data[offset..offset + in_feat]
            .iter()
            .map(|&b| Self::bf16_to_f32(b))
            .collect()
    }
}