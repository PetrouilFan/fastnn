use crate::storage_quantized::QuantizedTensor;

/// Block-wise quantized tensor (for KV cache, activations).
/// Uses U4x8 (4-bit) by default for maximum compression.
#[pyclass]
pub struct PyQuantizedTensor {
    inner: QuantizedTensor<U4x8>,
}

#[pymethods]
impl PyQuantizedTensor {
    /// Create from f32 data with block-wise quantization.
    #[staticmethod]
    #[pyo3(signature = (data, shape, block_size))]
    fn from_f32_blockwise(data: Vec<f64>, shape: Vec<usize>, block_size: usize) -> Self {
        let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        let inner = QuantizedTensor::from_f32_blockwise(&data_f32, &shape, block_size);
        PyQuantizedTensor { inner }
    }

    /// Convert to a PackedTensor (with single global scale)
    fn to_packed(&self) -> PyPackedTensor4 {
        let packed = self.inner.to_packed();
        PyPackedTensor4 { inner: packed }
    }

    /// Dequantize to f32 vector
    fn to_f32_vec(&self) -> Vec<f64> {
        self.inner.to_f32_vec().iter().map(|&x| x as f64).collect()
    }

    /// Memory usage in bytes
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    fn block_size(&self) -> usize {
        self.inner.block_size
    }
}
