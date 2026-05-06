
#[pyclass(from_py_object)]
#[derive(Clone)]
struct PyTensor {
    inner: crate::tensor::Tensor,
}

/// Destructor for DLPack PyCapsule - called when the capsule is garbage collected
unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    use crate::io::dlpack::DLManagedTensor;

    let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, c"dltensor".as_ptr());
    if !ptr.is_null() {
        let managed = ptr as *mut DLManagedTensor;
        if let Some(deleter) = (*managed).deleter {
            deleter(managed);
        }
    }
}

impl PyTensor {
    fn from_tensor(inner: crate::tensor::Tensor) -> Self {
        PyTensor { inner }
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>, shape: Vec<i64>) -> Self {
        PyTensor::from_tensor(crate::tensor::Tensor::from_vec(data, shape))
    }

    /// Create a tensor from a Python buffer object (zero-copy where possible)
    #[staticmethod]
    fn from_buffer(py: Python<'_>, buf: &Bound<'_, PyAny>) -> PyResult<Self> {
        use pyo3::buffer::PyBuffer;
        let buffer = PyBuffer::<f32>::get(buf)?;
        let shape: Vec<i64> = buffer.shape().iter().map(|&d| d as i64).collect();
        let data_len: usize = shape.iter().product::<i64>() as usize;
        let mut data = vec![0.0f32; data_len];
        buffer.copy_to_slice(py, &mut data)?;
        Ok(PyTensor::from_tensor(crate::tensor::Tensor::from_vec(data, shape)))
    }

    #[getter]
    fn shape(&self) -> Vec<i64> {
        self.inner.shape()
    }

    #[getter]
    fn dtype(&self) -> String {
        self.inner.dtype().as_str().to_string()
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device().as_str().to_string()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn numel(&self) -> i64 {
        self.inner.numel()
    }

    #[getter]
    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    fn item(&self) -> f32 {
        // Move to CPU if on GPU
        let tensor = if matches!(
            self.inner.inner.storage.as_ref(),
            crate::storage::Storage::Wgpu(_)
        ) {
            self.inner.to_cpu()
        } else {
            self.inner.clone()
        };
        tensor.item()
    }

    fn numpy(&self) -> Vec<f32> {
        self.inner.to_numpy()
    }

    /// DLPack protocol for zero-copy array exchange with NumPy, PyTorch, etc.
    /// Returns a PyCapsule wrapping a DLManagedTensor.
    fn __dlpack__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use crate::io::dlpack::to_dlpack;
        let ptr = to_dlpack(&self.inner);
        // Create PyCapsule with the DLPack tensor
        // The capsule name must be "dltensor" per DLPack spec
        let capsule = unsafe {
            pyo3::ffi::PyCapsule_New(
                ptr as *mut std::ffi::c_void,
                c"dltensor".as_ptr(),
                Some(dlpack_capsule_destructor),
            )
        };
        if capsule.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create DLPack capsule",
            ));
        }
        Ok(unsafe { pyo3::Bound::from_owned_ptr(py, capsule).unbind() })
    }

    /// DLPack device query protocol
    fn __dlpack_device__(&self) -> (i32, i32) {
        // (device_type, device_id) per DLPack spec
        // 1 = kDLCPU
        match self.inner.device() {
            crate::storage::Device::Cpu => (1, 0),
            crate::storage::Device::Wgpu(_) => (1, 0), // Report as CPU for DLPack compatibility
        }
    }

    fn debug_strides(&self) -> Vec<i64> {
        self.inner.strides()
    }

    #[pyo3(signature = (requires_grad))]
    fn requires_grad_(&mut self, requires_grad: bool) -> PyTensor {
        // Use Arc::make_mut to ensure exclusive ownership before modifying
        let inner = Arc::make_mut(&mut self.inner.inner);
        inner.set_requires_grad(requires_grad);
        PyTensor {
            inner: self.inner.clone(),
        }
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        let g = self.inner.grad();
        g.map(PyTensor::from_tensor)
    }

    #[getter]
    fn grad_fn(&self) -> Option<String> {
        self.inner.grad_fn().map(|_| "grad_fn".to_string())
    }

    #[getter]
    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    #[pyo3(signature = (grad=None))]
    fn backward(&self, py: Python<'_>, grad: Option<PyTensor>) {
        let inner = self.inner.clone();
        let grad_tensor = grad.map(|g| g.inner);
        py.detach(move || crate::autograd::backward(&inner, grad_tensor));
    }

    #[pyo3(signature = (grad))]
    fn set_grad(&mut self, grad: Option<PyTensor>) {
        let grad_tensor = grad.map(|g| g.inner);
        crate::tensor::TensorImpl::set_grad_for_tensor(&self.inner, grad_tensor);
    }

    fn detach(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.detach())
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn sum(&self, dim: Option<i32>, keepdim: bool) -> PyTensor {
        let dim = dim.unwrap_or(0);
        PyTensor::from_tensor(self.inner.sum(dim, keepdim))
    }

    #[pyo3(signature = (dim = None, keepdim = false))]
    fn mean(&self, dim: Option<i32>, keepdim: bool) -> PyTensor {
        let dim = dim.unwrap_or(0);
        PyTensor::from_tensor(self.inner.mean(dim, keepdim))
    }

    fn view(&self, shape: Vec<i64>) -> PyTensor {
        PyTensor::from_tensor(self.inner.view(shape))
    }

    fn reshape(&self, shape: Vec<i64>) -> PyTensor {
        PyTensor::from_tensor(self.inner.reshape(shape))
    }

    fn transpose(&self, dim0: i64, dim1: i64) -> PyTensor {
        PyTensor::from_tensor(self.inner.transpose(dim0 as usize, dim1 as usize))
    }

    fn permute(&self, dims: Vec<i64>) -> PyTensor {
        PyTensor::from_tensor(self.inner.permute(dims))
    }

    fn unsqueeze(&self, dim: i64) -> PyTensor {
        PyTensor::from_tensor(self.inner.unsqueeze(dim as usize))
    }

    fn squeeze(&self, dim: Option<i64>) -> PyTensor {
        PyTensor::from_tensor(self.inner.squeeze(dim.map(|d| d as usize)))
    }

    fn flip(&self, dim: i32) -> PyTensor {
        PyTensor::from_tensor(self.inner.flip(dim))
    }

    fn maximum(&self, other: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.maximum(&other.inner))
    }

    fn minimum(&self, other: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.minimum(&other.inner))
    }

    fn log_softmax(&self, dim: i32) -> PyTensor {
        PyTensor::from_tensor(self.inner.log_softmax(dim))
    }

    fn repeat(&self, repeats: Vec<i64>) -> PyTensor {
        PyTensor::from_tensor(self.inner.repeat(&repeats))
    }

    fn where_tensor(&self, condition: &PyTensor, other: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.where_tensor(&condition.inner, &other.inner))
    }

    fn __add__(&self, other: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.add(&other.inner))
    }

    fn __sub__(&self, other: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.sub(&other.inner))
    }

    fn __mul__(&self, other: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.mul(&other.inner))
    }

    fn __rmul__(&self, other: f32) -> PyTensor {
        PyTensor::from_tensor(self.inner.mul(&Tensor::from_scalar(other)))
    }

    fn mul_scalar(&self, other: f32) -> PyTensor {
        PyTensor::from_tensor(self.inner.mul(&Tensor::from_scalar(other)))
    }

    fn __truediv__(&self, other: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.div(&other.inner))
    }

    fn __matmul__(&self, other: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.matmul(&other.inner))
    }

    fn __neg__(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.neg())
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, device={})",
            self.inner.shape(),
            self.inner.dtype().as_str(),
            self.inner.device().as_str()
        )
    }

    fn __getitem__(&self, idx: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        use pyo3::types::PySlice;

        // Check if idx is a slice
        if let Ok(slice) = idx.cast::<PySlice>() {
            let length: isize = self.inner.shape()[0] as isize;
            let indices = slice.indices(length)?;
            // For now, only support slicing along dimension 0
            let start = indices.start as i64;
            let stop = indices.stop as i64;
            let step = indices.step as i64;
            let sliced = self.inner.slice(0, start, stop, step);
            Ok(PyTensor::from_tensor(sliced))
        } else {
            // Assume it's an integer index
            let idx_val: usize = idx.extract()?;
            // For 2D tensor [N, D], t[idx] returns [D] (the row)
            // For 1D tensor [N], t[idx] returns scalar (0-dim)
            // Implementation: slice(0, idx, idx+1, 1).squeeze(0)
            let sliced = self.inner.slice(0, idx_val as i64, (idx_val + 1) as i64, 1);
            Ok(PyTensor::from_tensor(sliced.squeeze(Some(0))))
        }
    }

    fn cpu(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.to_cpu())
    }

    fn to_gpu(&self, device_id: usize) -> PyTensor {
        PyTensor::from_tensor(self.inner.to_gpu(device_id))
    }

    fn contiguous(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.contiguous())
    }
}

#[pyfunction]
pub fn tensor_from_buffer(py: Python<'_>, buf: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    use pyo3::buffer::PyBuffer;
    let buffer = PyBuffer::<f32>::get(buf)?;
    let shape: Vec<i64> = buffer.shape().iter().map(|&d| d as i64).collect();
    let data_len: usize = shape.iter().product::<i64>() as usize;
    let mut data = vec![0.0f32; data_len];
    buffer.copy_to_slice(py, &mut data)?;
    Ok(PyTensor::from_tensor(Tensor::from_vec(data, shape)))
}

