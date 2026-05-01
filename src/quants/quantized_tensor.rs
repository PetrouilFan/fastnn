use crate::quants::{QuantizedDType, QuantizedGemm};

/// Generic container for a GGML-style quantized tensor.
#[derive(Clone)]
pub struct GgmlQuantizedTensor {
    dtype: QuantizedDType,
    shape: [usize; 2],
    data: Vec<u8>,
}

impl GgmlQuantizedTensor {
    /// Create a tensor from a raw GGUF tensor blob.
    pub fn from_gguf_bytes(dtype: QuantizedDType, shape: [usize; 2], data: Vec<u8>) -> Self {
        GgmlQuantizedTensor { dtype, shape, data }
    }

    /// Return a boxed `QuantizedGemm` impl for this tensor.
    pub fn as_gemm(&self) -> Box<dyn QuantizedGemm> {
        use crate::quants::{Q4_0, Q4_K, Q6K, QuantizedDType as DT};
        match self.dtype {
            DT::Q4_0 => Box::new(Q4_0::from_bytes(&self.data, self.shape)),
            DT::Q4_K => Box::new(Q4_K::from_bytes(&self.data, self.shape)),
            DT::Q6_K => Box::new(Q6K::from_bytes(&self.data, self.shape)),
            _ => unimplemented!("{:?} not yet implemented", self.dtype),
        }
    }

    /// Convenience wrapper: GEMV through the owned dispatch.
    pub fn gemv(&self, activation: &[f32], output: &mut [f32]) {
        if self.dtype == QuantizedDType::F32 {
            let data: &[f32] = bytemuck::cast_slice(&self.data);
            let in_feat = self.shape[1];
            let out_feat = self.shape[0];
            for row in 0..out_feat {
                let mut sum = 0.0f32;
                let row_offset = row * in_feat;
                for col in 0..in_feat {
                    sum += data[row_offset + col] * activation[col];
                }
                output[row] = sum;
            }
        } else {
            self.as_gemm().gemv(activation, output);
        }
    }

    pub fn dtype(&self) -> QuantizedDType { self.dtype }
    pub fn shape(&self) -> [usize; 2] { self.shape }
    pub fn out_features(&self) -> usize { self.shape[0] }
    pub fn in_features(&self) -> usize { self.shape[1] }
    pub fn memory_bytes(&self) -> usize { self.data.len() }
    pub fn data(&self) -> &[u8] { &self.data }

    pub fn row(&self, row_idx: usize) -> Vec<f32> {
        if self.dtype == QuantizedDType::F32 {
            let in_feat = self.shape[1];
            let start = row_idx * in_feat;
            let data: &[f32] = bytemuck::cast_slice(&self.data);
            if start + in_feat <= data.len() {
                data[start..start + in_feat].to_vec()
            } else {
                vec![1.0; in_feat]
            }
        } else {
            self.as_gemm().row(row_idx)
        }
    }
}
