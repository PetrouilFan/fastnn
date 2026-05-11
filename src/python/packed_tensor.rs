use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;

// ---- PyPackedTensor4 (4-bit, U4x8) ----

#[pyclass]
pub struct PyPackedTensor4 {
    inner: PackedTensor<U4x8>,
}

#[pymethods]
impl PyPackedTensor4 {
    #[new]
    #[pyo3(signature = (data, shape, scale = 1.0, zero = 0.0))]
    fn new(data: Vec<f64>, shape: Vec<usize>, scale: f64, zero: f64) -> Self {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let inner = PackedTensor::from_f32_slice(&data_f32, &shape, scale as f32, zero as f32);
        PyPackedTensor4 { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape))]
    fn from_f32_auto(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let inner = PackedTensor::from_f32_auto(&data_f32, &shape);
        PyPackedTensor4 { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape))]
    fn from_f32_per_channel(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let inner = PackedTensor::from_f32_per_channel(&data_f32, &shape);
        PyPackedTensor4 { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (data_bytes, shape, scales, zeros))]
    fn from_bytes(data_bytes: Vec<u8>, shape: Vec<usize>, scales: Vec<f64>, zeros: Vec<f64>) -> Self {
        let numel: usize = shape.iter().product();
        let packed_len = numel.div_ceil(<U4x8 as PackedWord>::ITEMS);
        let scales_f32: Vec<f32> = scales.into_iter().map(|x| x as f32).collect();
        let zeros_f32: Vec<f32> = zeros.into_iter().map(|x| x as f32).collect();
        assert_eq!(data_bytes.len(), packed_len * 4, "from_bytes: data length mismatch");
        let data: Vec<U4x8> = bytemuck::cast_slice(&data_bytes).to_vec();
        let inner = PackedTensor::from_raw(data, shape, scales_f32, zeros_f32);
        PyPackedTensor4 { inner }
    }

    fn to_f32_vec(&self) -> Vec<f64> {
        self.inner.to_f32_vec().into_iter().map(|x| x as f64).collect()
    }

    fn to_tensor(&self) -> PyTensor {
        let data = self.inner.to_f32_vec();
        let shape: Vec<i64> = self.inner.shape().iter().map(|&s| s as i64).collect();
        let t = Tensor::from_vec(data, shape);
        PyTensor::from_tensor(t)
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.inner.as_bytes().to_vec()
    }

    fn scales(&self) -> Vec<f64> {
        self.inner.scales.iter().map(|&s| s as f64).collect()
    }

    fn zeros(&self) -> Vec<f64> {
        self.inner.zeros.iter().map(|&z| z as f64).collect()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> { self.inner.shape().to_vec() }

    #[getter]
    fn numel(&self) -> usize { self.inner.numel() }

    #[getter]
    fn scale(&self) -> f64 { self.inner.scale() as f64 }

    #[getter]
    fn zero(&self) -> f64 { self.inner.zero() as f64 }

    #[getter]
    fn is_per_channel(&self) -> bool { self.inner.is_per_channel() }

    fn get(&self, idx: usize) -> f64 { self.inner.get(idx) as f64 }

    fn set(&mut self, idx: usize, val: f64) { self.inner.set(idx, val as f32); }
}

// ---- PyPackedTensor8 (8-bit, U8x4) ----

#[pyclass]
pub struct PyPackedTensor8 {
    inner: PackedTensor<U8x4>,
}

#[pymethods]
impl PyPackedTensor8 {
    #[new]
    #[pyo3(signature = (data, shape, scale = 1.0, zero = 0.0))]
    fn new(data: Vec<f64>, shape: Vec<usize>, scale: f64, zero: f64) -> Self {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let inner = PackedTensor::from_f32_slice(&data_f32, &shape, scale as f32, zero as f32);
        PyPackedTensor8 { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape))]
    fn from_f32_auto(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let inner = PackedTensor::from_f32_auto(&data_f32, &shape);
        PyPackedTensor8 { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape))]
    fn from_f32_per_channel(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let inner = PackedTensor::from_f32_per_channel(&data_f32, &shape);
        PyPackedTensor8 { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (data_bytes, shape, scales, zeros))]
    fn from_bytes(data_bytes: Vec<u8>, shape: Vec<usize>, scales: Vec<f64>, zeros: Vec<f64>) -> Self {
        let numel: usize = shape.iter().product();
        let packed_len = numel.div_ceil(<U8x4 as PackedWord>::ITEMS);
        let scales_f32: Vec<f32> = scales.into_iter().map(|x| x as f32).collect();
        let zeros_f32: Vec<f32> = zeros.into_iter().map(|x| x as f32).collect();
        assert_eq!(data_bytes.len(), packed_len * 4, "from_bytes: data length mismatch");
        let data: Vec<U8x4> = bytemuck::cast_slice(&data_bytes).to_vec();
        let inner = PackedTensor::from_raw(data, shape, scales_f32, zeros_f32);
        PyPackedTensor8 { inner }
    }

    fn to_f32_vec(&self) -> Vec<f64> {
        self.inner.to_f32_vec().into_iter().map(|x| x as f64).collect()
    }

    fn to_tensor(&self) -> PyTensor {
        let data = self.inner.to_f32_vec();
        let shape: Vec<i64> = self.inner.shape().iter().map(|&s| s as i64).collect();
        let t = Tensor::from_vec(data, shape);
        PyTensor::from_tensor(t)
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.inner.as_bytes().to_vec()
    }

    fn scales(&self) -> Vec<f64> {
        self.inner.scales.iter().map(|&s| s as f64).collect()
    }

    fn zeros(&self) -> Vec<f64> {
        self.inner.zeros.iter().map(|&z| z as f64).collect()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> { self.inner.shape().to_vec() }

    #[getter]
    fn numel(&self) -> usize { self.inner.numel() }

    #[getter]
    fn scale(&self) -> f64 { self.inner.scale() as f64 }

    #[getter]
    fn zero(&self) -> f64 { self.inner.zero() as f64 }

    #[getter]
    fn is_per_channel(&self) -> bool { self.inner.is_per_channel() }

    fn get(&self, idx: usize) -> f64 { self.inner.get(idx) as f64 }

    fn set(&mut self, idx: usize, val: f64) { self.inner.set(idx, val as f32); }
}

// ---- PyPackedTensor16 (16-bit, F16x2) ----

#[pyclass]
pub struct PyPackedTensor16 {
    inner: PackedTensor<F16x2>,
}

#[pymethods]
impl PyPackedTensor16 {
    #[new]
    #[pyo3(signature = (data, shape, scale = 1.0, zero = 0.0))]
    fn new(data: Vec<f64>, shape: Vec<usize>, scale: f64, zero: f64) -> Self {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let inner = PackedTensor::from_f32_slice(&data_f32, &shape, scale as f32, zero as f32);
        PyPackedTensor16 { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape))]
    fn from_f32_auto(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let inner = PackedTensor::from_f32_auto(&data_f32, &shape);
        PyPackedTensor16 { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape))]
    fn from_f32_per_channel(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let inner = PackedTensor::from_f32_per_channel(&data_f32, &shape);
        PyPackedTensor16 { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (data_bytes, shape, scales, zeros))]
    fn from_bytes(data_bytes: Vec<u8>, shape: Vec<usize>, scales: Vec<f64>, zeros: Vec<f64>) -> Self {
        let numel: usize = shape.iter().product();
        let packed_len = numel.div_ceil(<F16x2 as PackedWord>::ITEMS);
        let scales_f32: Vec<f32> = scales.into_iter().map(|x| x as f32).collect();
        let zeros_f32: Vec<f32> = zeros.into_iter().map(|x| x as f32).collect();
        assert_eq!(data_bytes.len(), packed_len * 4, "from_bytes: data length mismatch");
        let data: Vec<F16x2> = bytemuck::cast_slice(&data_bytes).to_vec();
        let inner = PackedTensor::from_raw(data, shape, scales_f32, zeros_f32);
        PyPackedTensor16 { inner }
    }

    fn to_f32_vec(&self) -> Vec<f64> {
        self.inner.to_f32_vec().into_iter().map(|x| x as f64).collect()
    }

    fn to_tensor(&self) -> PyTensor {
        let data = self.inner.to_f32_vec();
        let shape: Vec<i64> = self.inner.shape().iter().map(|&s| s as i64).collect();
        let t = Tensor::from_vec(data, shape);
        PyTensor::from_tensor(t)
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.inner.as_bytes().to_vec()
    }

    fn scales(&self) -> Vec<f64> {
        self.inner.scales.iter().map(|&s| s as f64).collect()
    }

    fn zeros(&self) -> Vec<f64> {
        self.inner.zeros.iter().map(|&z| z as f64).collect()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> { self.inner.shape().to_vec() }

    #[getter]
    fn numel(&self) -> usize { self.inner.numel() }

    #[getter]
    fn scale(&self) -> f64 { self.inner.scale() as f64 }

    #[getter]
    fn zero(&self) -> f64 { self.inner.zero() as f64 }

    #[getter]
    fn is_per_channel(&self) -> bool { self.inner.is_per_channel() }

    fn get(&self, idx: usize) -> f64 { self.inner.get(idx) as f64 }

    fn set(&mut self, idx: usize, val: f64) { self.inner.set(idx, val as f32); }
}

// ---- PyPackedTensor32 (32-bit, F32x1) ----

#[pyclass]
pub struct PyPackedTensor32 {
    inner: PackedTensor<F32x1>,
}

#[pymethods]
impl PyPackedTensor32 {
    #[new]
    #[pyo3(signature = (data, shape, scale = 1.0, zero = 0.0))]
    fn new(data: Vec<f64>, shape: Vec<usize>, scale: f64, zero: f64) -> Self {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let inner = PackedTensor::from_f32_slice(&data_f32, &shape, scale as f32, zero as f32);
        PyPackedTensor32 { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape))]
    fn from_f32_auto(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let inner = PackedTensor::from_f32_auto(&data_f32, &shape);
        PyPackedTensor32 { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape))]
    fn from_f32_per_channel(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let inner = PackedTensor::from_f32_per_channel(&data_f32, &shape);
        PyPackedTensor32 { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (data_bytes, shape, scales, zeros))]
    fn from_bytes(data_bytes: Vec<u8>, shape: Vec<usize>, scales: Vec<f64>, zeros: Vec<f64>) -> Self {
        let numel: usize = shape.iter().product();
        let packed_len = numel.div_ceil(<F32x1 as PackedWord>::ITEMS);
        let scales_f32: Vec<f32> = scales.into_iter().map(|x| x as f32).collect();
        let zeros_f32: Vec<f32> = zeros.into_iter().map(|x| x as f32).collect();
        assert_eq!(data_bytes.len(), packed_len * 4, "from_bytes: data length mismatch");
        let data: Vec<F32x1> = bytemuck::cast_slice(&data_bytes).to_vec();
        let inner = PackedTensor::from_raw(data, shape, scales_f32, zeros_f32);
        PyPackedTensor32 { inner }
    }

    fn to_f32_vec(&self) -> Vec<f64> {
        self.inner.to_f32_vec().into_iter().map(|x| x as f64).collect()
    }

    fn to_tensor(&self) -> PyTensor {
        let data = self.inner.to_f32_vec();
        let shape: Vec<i64> = self.inner.shape().iter().map(|&s| s as i64).collect();
        let t = Tensor::from_vec(data, shape);
        PyTensor::from_tensor(t)
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.inner.as_bytes().to_vec()
    }

    fn scales(&self) -> Vec<f64> {
        self.inner.scales.iter().map(|&s| s as f64).collect()
    }

    fn zeros(&self) -> Vec<f64> {
        self.inner.zeros.iter().map(|&z| z as f64).collect()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> { self.inner.shape().to_vec() }

    #[getter]
    fn numel(&self) -> usize { self.inner.numel() }

    #[getter]
    fn scale(&self) -> f64 { self.inner.scale() as f64 }

    #[getter]
    fn zero(&self) -> f64 { self.inner.zero() as f64 }

    #[getter]
    fn is_per_channel(&self) -> bool { self.inner.is_per_channel() }

    fn get(&self, idx: usize) -> f64 { self.inner.get(idx) as f64 }

    fn set(&mut self, idx: usize, val: f64) { self.inner.set(idx, val as f32); }
}
