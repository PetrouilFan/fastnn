#![allow(clippy::needless_range_loop)]
mod autograd;
mod dispatcher;
mod io;
mod iterator;
mod kernels;
mod nn;
mod optim;
mod storage;
mod storage_pool;
mod tensor;
mod train;

// Native packed precision modules
pub mod dtypes;
pub mod swar;
pub mod packed_tensor;
pub mod backends;
pub mod packed_layer;
pub mod packed_train;

// Re-export packed precision public API
pub use dtypes::{PackedWord, U4x8, U8x4, F16x2, F32x1};
pub use packed_tensor::PackedTensor;
pub use packed_layer::{PackedLinear, Linear4, Linear8, Linear16, Linear32};
pub use packed_layer::{use_cpu, use_wgpu, is_wgpu};
pub use packed_train::MasterWeightOptimizer;

use autograd::{no_grad_enter, no_grad_exit};
use dispatcher::list_registered_ops as dispatcher_list_ops;
use nn::Module;
use optim::Optimizer;
use parking_lot::RwLock;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::wrap_pyfunction;
use pyo3::PyAny;
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use storage::{allocator_stats as storage_allocator_stats, DType, Device};
use tensor::Tensor;

// Custom exception hierarchy for fastnn
pyo3::create_exception!(fastnn, FastnnError, PyRuntimeError, "Base exception for fastnn operations.");
pyo3::create_exception!(fastnn, ShapeError, FastnnError, "Shape mismatch or invalid shape for operation.");
pyo3::create_exception!(fastnn, DtypeError, FastnnError, "Data type mismatch or unsupported dtype.");
pyo3::create_exception!(fastnn, DeviceError, FastnnError, "Device-related error (e.g., GPU unavailable).");
pyo3::create_exception!(fastnn, AutogradError, FastnnError, "Error during autograd/backward pass.");
pyo3::create_exception!(fastnn, OptimizerError, FastnnError, "Error in optimizer step.");
pyo3::create_exception!(fastnn, IoError, FastnnError, "Error during serialization/deserialization.");
pyo3::create_exception!(fastnn, CudaError, FastnnError, "CUDA/GPU computation error.");

// Thread-local default device storage
static DEFAULT_DEVICE: OnceLock<RwLock<Device>> = OnceLock::new();

// Global seeded RNG for deterministic initialization
static SEEDED_RNG: OnceLock<Mutex<rand::rngs::StdRng>> = OnceLock::new();

fn get_default_device() -> Device {
    let guard = DEFAULT_DEVICE
        .get_or_init(|| RwLock::new(Device::Cpu))
        .read();
    *guard
}

fn set_default_device_internal(device: Device) {
    let mut guard = DEFAULT_DEVICE
        .get_or_init(|| RwLock::new(Device::Cpu))
        .write();
    *guard = device;
}

#[pyclass(from_py_object)]
#[derive(Clone)]
struct PyTensor {
    inner: Tensor,
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
    fn from_tensor(inner: Tensor) -> Self {
        PyTensor { inner }
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>, shape: Vec<i64>) -> Self {
        PyTensor::from_tensor(Tensor::from_vec(data, shape))
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
#[pyo3(signature = (data, shape, device = None))]
fn tensor_from_data<'py>(
    data: &Bound<'py, PyAny>,
    shape: &Bound<'py, PyAny>,
    device: Option<String>,
) -> PyResult<PyTensor> {
    let data: Vec<f32> = data.extract()?;
    let shape: Vec<i64> = shape.extract()?;
    let device = device
        .as_ref()
        .and_then(|s| Device::from_str(s))
        .unwrap_or_else(get_default_device);
    Ok(PyTensor::from_tensor(Tensor::from_vec_with_device(
        data, shape, device,
    )))
}

#[pyfunction]
fn tensor_factory(data: Vec<f32>, shape: Vec<i64>) -> PyTensor {
    PyTensor::from_tensor(Tensor::from_vec(data, shape))
}

#[pyfunction]
fn tensor_from_list(data: Vec<f32>, shape: Vec<i64>) -> PyTensor {
    PyTensor::from_tensor(Tensor::from_vec(data, shape))
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None, device = None))]
fn zeros(shape: Vec<i64>, dtype: Option<String>, device: Option<String>) -> PyTensor {
    let dtype = dtype
        .and_then(|s| DType::from_str(&s))
        .unwrap_or(DType::F32);
    let device = device
        .as_ref()
        .and_then(|s| Device::from_str(s))
        .unwrap_or_else(get_default_device);
    PyTensor::from_tensor(Tensor::zeros(shape, dtype, device))
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None, device = None))]
fn ones(shape: Vec<i64>, dtype: Option<String>, device: Option<String>) -> PyTensor {
    let dtype = dtype
        .and_then(|s| DType::from_str(&s))
        .unwrap_or(DType::F32);
    let device = device
        .as_ref()
        .and_then(|s| Device::from_str(s))
        .unwrap_or_else(get_default_device);
    PyTensor::from_tensor(Tensor::ones(shape, dtype, device))
}

#[pyfunction]
#[pyo3(signature = (shape, value, dtype = None, device = None))]
fn full(shape: Vec<i64>, value: f32, dtype: Option<String>, device: Option<String>) -> PyTensor {
    let dtype = dtype
        .and_then(|s| DType::from_str(&s))
        .unwrap_or(DType::F32);
    let device = device
        .as_ref()
        .and_then(|s| Device::from_str(s))
        .unwrap_or_else(get_default_device);
    PyTensor::from_tensor(Tensor::full(shape, value, dtype, device))
}

#[pyfunction]
#[pyo3(signature = (start, end, step = None, device = None))]
fn arange(start: f32, end: f32, step: Option<f32>, device: Option<String>) -> PyTensor {
    let step = step.unwrap_or(1.0);
    let start_f64 = start as f64;
    let end_f64 = end as f64;
    let step_f64 = step as f64;
    let numel = ((end_f64 - start_f64) / step_f64).ceil() as usize;
    let values: Vec<f32> = (0..numel).map(|i| (start_f64 + i as f64 * step_f64) as f32).collect();
    let device = device
        .as_ref()
        .and_then(|s| Device::from_str(s))
        .unwrap_or_else(get_default_device);
    PyTensor::from_tensor(Tensor::from_vec_with_device(
        values,
        vec![numel as i64],
        device,
    ))
}

#[pyfunction]
#[pyo3(signature = (start, end, steps, device = None))]
fn linspace(start: f32, end: f32, steps: usize, device: Option<String>) -> PyTensor {
    if steps == 0 {
        return PyTensor::from_tensor(Tensor::from_vec(vec![], vec![0]));
    }
    if steps == 1 {
        return PyTensor::from_tensor(Tensor::from_vec(vec![start], vec![1]));
    }
    let values: Vec<f32> = (0..steps)
        .map(|i| {
            let t = i as f32 / (steps - 1) as f32;
            start * (1.0 - t) + end * t
        })
        .collect();
    let device = device
        .as_ref()
        .and_then(|s| Device::from_str(s))
        .unwrap_or_else(get_default_device);
    PyTensor::from_tensor(Tensor::from_vec_with_device(
        values,
        vec![steps as i64],
        device,
    ))
}

#[pyfunction]
#[pyo3(signature = (n, m=None, device=None))]
fn eye(n: i64, m: Option<i64>, device: Option<String>) -> PyTensor {
    let m = m.unwrap_or(n);
    let mut values = vec![0.0f32; (n * m) as usize];
    for i in 0..n.min(m) {
        values[(i * m + i) as usize] = 1.0;
    }
    let device = device
        .as_ref()
        .and_then(|s| Device::from_str(s))
        .unwrap_or_else(get_default_device);
    PyTensor::from_tensor(Tensor::from_vec_with_device(values, vec![n, m], device))
}

#[pyfunction]
#[pyo3(signature = (shape, device = None))]
fn randn(shape: Vec<i64>, device: Option<String>) -> PyTensor {
    let numel: i64 = shape.iter().product();
    let mut values = vec![0.0f32; numel as usize];
    let mut rng = rand::thread_rng();
    for v in &mut values {
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }
    let device = device
        .as_ref()
        .and_then(|s| Device::from_str(s))
        .unwrap_or_else(get_default_device);
    PyTensor::from_tensor(Tensor::from_vec_with_device(values, shape, device))
}

#[pyfunction]
#[pyo3(signature = (shape, device = None))]
fn rand_uniform(shape: Vec<i64>, device: Option<String>) -> PyTensor {
    let numel: i64 = shape.iter().product();
    let values: Vec<f32> = (0..numel as usize).map(|_| rand::random()).collect();
    let device = device
        .as_ref()
        .and_then(|s| Device::from_str(s))
        .unwrap_or_else(get_default_device);
    PyTensor::from_tensor(Tensor::from_vec_with_device(values, shape, device))
}

#[pyfunction]
#[pyo3(signature = (shape, low, high, device = None))]
fn randint(shape: Vec<i64>, low: i32, high: i32, device: Option<String>) -> PyTensor {
    let numel: i64 = shape.iter().product();
    let range = (high - low) as u32;
    let values: Vec<f32> = if range == 0 {
        vec![low as f32; numel as usize]
    } else {
        use rand::distributions::{Distribution, Uniform};
        let uniform = Uniform::new(0u32, range);
        let mut rng = rand::thread_rng();
        (0..numel as usize)
            .map(|_| (uniform.sample(&mut rng) as i32 + low) as f32)
            .collect()
    };
    let device = device
        .as_ref()
        .and_then(|s| Device::from_str(s))
        .unwrap_or_else(get_default_device);
    PyTensor::from_tensor(Tensor::from_vec_with_device(values, shape, device))
}

#[pyfunction]
fn zeros_like(tensor: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(Tensor::zeros(
        tensor.inner.shape(),
        tensor.inner.dtype(),
        tensor.inner.device(),
    ))
}

#[pyfunction]
fn ones_like(tensor: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(Tensor::ones(
        tensor.inner.shape(),
        tensor.inner.dtype(),
        tensor.inner.device(),
    ))
}

#[pyfunction]
fn full_like(tensor: &PyTensor, value: f32) -> PyTensor {
    PyTensor::from_tensor(Tensor::full(
        tensor.inner.shape(),
        value,
        tensor.inner.dtype(),
        tensor.inner.device(),
    ))
}

#[pyfunction]
fn add(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.add(&b.inner))
}

#[pyfunction]
fn sub(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.sub(&b.inner))
}

#[pyfunction]
fn mul(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.mul(&b.inner))
}

#[pyfunction]
fn div(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.div(&b.inner))
}

#[pyfunction]
fn matmul(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.matmul(&b.inner))
}

#[pyfunction]
fn batched_mlp_forward(
    input: &PyTensor,
    weights: Vec<PyTensor>,
    biases: Vec<PyTensor>,
    activations: Vec<String>,
) -> PyTensor {
    let mut x = input.inner.clone();

    for i in 0..weights.len() {
        let w = &weights[i].inner;
        let b = biases.get(i).map(|b| &b.inner);

        // PyTorch convention: weight is [out_features, in_features], so x @ w.T
        x = x.matmul(&w.transpose(0, 1));
        if let Some(bias) = b {
            x = x.add(bias);
        }

        if i < activations.len() {
            let act = &activations[i];
            match act.as_str() {
                "relu" => {
                    use dispatcher::dispatch;
                    let dispatch_key = dispatcher::device_to_dispatch_key(x.device());
                    let result = dispatch("relu", dispatch_key, &[&x]);
                    x = result[0].clone();
                }
                "sigmoid" => {
                    use dispatcher::dispatch;
                    let dispatch_key = dispatcher::device_to_dispatch_key(x.device());
                    let result = dispatch("sigmoid", dispatch_key, &[&x]);
                    x = result[0].clone();
                }
                "tanh" => {
                    use dispatcher::dispatch;
                    let dispatch_key = dispatcher::device_to_dispatch_key(x.device());
                    let result = dispatch("tanh", dispatch_key, &[&x]);
                    x = result[0].clone();
                }
                _ => {}
            }
        }
    }

    PyTensor::from_tensor(x)
}

#[pyfunction]
fn neg(a: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.neg())
}

#[pyfunction]
fn abs(a: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.abs())
}

#[pyfunction]
fn exp(a: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.exp())
}

#[pyfunction]
fn log(a: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.ln())
}

#[pyfunction]
fn sqrt(py: Python<'_>, a: &PyTensor) -> PyTensor {
    let a_inner = a.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.sqrt()))
}

#[pyfunction]
fn relu(py: Python<'_>, a: &PyTensor) -> PyTensor {
    let a_inner = a.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.relu()))
}

#[pyfunction]
fn fused_add_relu(a: &PyTensor, b: &PyTensor) -> PyTensor {
    use crate::dispatcher::{device_to_dispatch_key, dispatch};
    let dispatch_key = device_to_dispatch_key(a.inner.device());
    let result = dispatch("fused_add_relu", dispatch_key, &[&a.inner, &b.inner]);
    PyTensor::from_tensor(result.into_iter().next().unwrap())
}

#[pyfunction]
fn fused_linear_relu(x: &PyTensor, w: &PyTensor, bias: Option<&PyTensor>) -> PyTensor {
    use crate::dispatcher::{device_to_dispatch_key, dispatch};
    let dispatch_key = device_to_dispatch_key(x.inner.device());
    let args: Vec<_> = match bias {
        Some(b) => vec![&x.inner, &w.inner, &b.inner],
        None => vec![&x.inner, &w.inner],
    };
    let result = dispatch("fused_linear_relu", dispatch_key, &args);
    PyTensor::from_tensor(result.into_iter().next().unwrap())
}

#[pyfunction]
fn fused_linear_gelu(x: &PyTensor, w: &PyTensor, bias: Option<&PyTensor>) -> PyTensor {
    use crate::dispatcher::{device_to_dispatch_key, dispatch};
    let dispatch_key = device_to_dispatch_key(x.inner.device());
    let args: Vec<_> = match bias {
        Some(b) => vec![&x.inner, &w.inner, &b.inner],
        None => vec![&x.inner, &w.inner],
    };
    let result = dispatch("fused_linear_gelu", dispatch_key, &args);
    PyTensor::from_tensor(result.into_iter().next().unwrap())
}

#[pyfunction]
fn gelu(py: Python<'_>, a: &PyTensor) -> PyTensor {
    let a_inner = a.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.gelu()))
}

#[pyfunction]
fn sigmoid(py: Python<'_>, a: &PyTensor) -> PyTensor {
    let a_inner = a.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.sigmoid()))
}

#[pyfunction]
fn tanh(py: Python<'_>, a: &PyTensor) -> PyTensor {
    let a_inner = a.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.tanh()))
}

#[pyfunction]
fn silu(py: Python<'_>, a: &PyTensor) -> PyTensor {
    let a_inner = a.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.silu()))
}

#[pyfunction]
fn softmax(py: Python<'_>, a: &PyTensor, dim: i32) -> PyTensor {
    let a_inner = a.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.softmax(dim)))
}

#[pyfunction]
fn log_softmax(a: &PyTensor, dim: i32) -> PyTensor {
    use dispatcher::dispatch;
    let dispatch_key = dispatcher::device_to_dispatch_key(a.inner.device());
    let result = dispatch(
        "log_softmax",
        dispatch_key,
        &[&a.inner, &Tensor::from_scalar(dim as f32)],
    );
    PyTensor::from_tensor(result[0].clone())
}

#[pyfunction]
fn embedding(weight: &PyTensor, indices: &PyTensor) -> PyTensor {
    use dispatcher::dispatch;
    let dispatch_key = dispatcher::device_to_dispatch_key(weight.inner.device());
    let result = dispatch("embedding", dispatch_key, &[&weight.inner, &indices.inner]);
    PyTensor::from_tensor(result[0].clone())
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn sum(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyTensor {
    let dim = dim.unwrap_or(0);
    PyTensor::from_tensor(a.inner.sum(dim, keepdim))
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn mean(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyTensor {
    let dim = dim.unwrap_or(0);
    PyTensor::from_tensor(a.inner.mean(dim, keepdim))
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn max(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyTensor {
    let dim = dim.unwrap_or(0);
    PyTensor::from_tensor(a.inner.max(dim, keepdim))
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn min(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyTensor {
    let dim = dim.unwrap_or(0);
    use dispatcher::dispatch;
    let dispatch_key = dispatcher::device_to_dispatch_key(a.inner.device());
    let result = dispatch(
        "min",
        dispatch_key,
        &[
            &a.inner,
            &Tensor::from_scalar(dim as f32),
            &Tensor::from_scalar(if keepdim { 1.0 } else { 0.0 }),
        ],
    );
    PyTensor::from_tensor(result[0].clone())
}

#[pyfunction]
#[pyo3(signature = (a, dim = None))]
fn argmax(a: &PyTensor, dim: Option<i32>) -> PyTensor {
    let dim = dim.unwrap_or(0);
    use dispatcher::dispatch;
    let dispatch_key = dispatcher::device_to_dispatch_key(a.inner.device());
    let result = dispatch(
        "max",
        dispatch_key,
        &[
            &a.inner,
            &Tensor::from_scalar(dim as f32),
            &Tensor::from_scalar(1.0),
        ],
    );
    PyTensor::from_tensor(result[0].clone())
}

#[pyfunction]
#[pyo3(signature = (a, dim = None))]
fn argmin(a: &PyTensor, dim: Option<i32>) -> PyTensor {
    let dim = dim.unwrap_or(0);
    use dispatcher::dispatch;
    let dispatch_key = dispatcher::device_to_dispatch_key(a.inner.device());
    let result = dispatch(
        "min",
        dispatch_key,
        &[
            &a.inner,
            &Tensor::from_scalar(dim as f32),
            &Tensor::from_scalar(1.0),
        ],
    );
    PyTensor::from_tensor(result[0].clone())
}

#[pyfunction]
#[pyo3(signature = (pred, target, reduction = None))]
fn mse_loss(pred: &PyTensor, target: &PyTensor, reduction: Option<String>) -> PyTensor {
    use dispatcher::dispatch;
    let dispatch_key = dispatcher::device_to_dispatch_key(pred.inner.device());
    let reduction = reduction.unwrap_or_else(|| "mean".to_string());
    let reduction_code = match reduction.as_str() {
        "none" => 0.0,
        "mean" => 1.0,
        "sum" => 2.0,
        _ => 1.0,
    };
    let result = dispatch(
        "mse_loss",
        dispatch_key,
        &[
            &pred.inner,
            &target.inner,
            &Tensor::from_scalar(reduction_code),
        ],
    );
    let output = result[0].clone();

    // Set up autograd tracking
    if autograd::is_grad_enabled() && pred.inner.requires_grad() {
        let edges = autograd::make_edge(&pred.inner);
        let backward = autograd::MSELossBackward::new(
            pred.inner.clone(),
            target.inner.clone(),
            reduction,
            edges,
        );
        let mut meta = autograd::AutogradMeta::new_non_leaf(true);
        meta.grad_fn = Some(std::sync::Arc::new(backward));
        let mut output = output.clone();
        Arc::make_mut(&mut output.inner).autograd_meta =
            Some(Arc::new(std::sync::Mutex::new(meta)));
        PyTensor::from_tensor(output)
    } else {
        PyTensor::from_tensor(output)
    }
}

#[pyfunction]
#[pyo3(signature = (pred, target, reduction = None))]
fn cross_entropy_loss(pred: &PyTensor, target: &PyTensor, reduction: Option<String>) -> PyTensor {
    use autograd::AutogradMeta;
    use dispatcher::dispatch;
    use std::sync::Arc;
    let dispatch_key = dispatcher::device_to_dispatch_key(pred.inner.device());

    let reduction = reduction.unwrap_or_else(|| "mean".to_string());
    let reduction_code = match reduction.as_str() {
        "none" => 0.0,
        "mean" => 1.0,
        "sum" => 2.0,
        _ => 1.0,
    };
    let result = dispatch(
        "cross_entropy_loss",
        dispatch_key,
        &[
            &pred.inner,
            &target.inner,
            &Tensor::from_scalar(reduction_code),
        ],
    );
    let output = result[0].clone();

    if pred.inner.requires_grad() {
        let edges = autograd::make_edge(&pred.inner);
        let backward = autograd::CrossEntropyBackward::new(
            pred.inner.clone(),
            target.inner.clone(),
            reduction.clone(),
            edges,
        );
        let mut meta = AutogradMeta::new_non_leaf(true);
        meta.grad_fn = Some(std::sync::Arc::new(backward));
        let mut output = output.clone();
        Arc::make_mut(&mut output.inner).autograd_meta =
            Some(Arc::new(std::sync::Mutex::new(meta)));
        PyTensor::from_tensor(output)
    } else {
        PyTensor::from_tensor(output)
    }
}

#[pyfunction]
fn _no_grad_enter() {
    no_grad_enter();
}

#[pyfunction]
fn _no_grad_exit() {
    no_grad_exit();
}

#[pyfunction]
#[pyo3(signature = (fn_name, inputs))]
#[allow(unused_variables)]
fn checkpoint(fn_name: &str, inputs: Vec<PyTensor>) -> PyResult<Vec<PyTensor>> {
    // For now, checkpoint just returns the inputs as outputs
    // A full implementation would store the function and recompute during backward
    // This is a placeholder that demonstrates the API

    // Note: PyO3 doesn't easily support passing Python callables to Rust.
    // A full implementation would need to store the Python function and call it during backward.
    // For now, we just return the inputs as-is (identity function).

    // Just return the inputs as outputs (identity function)
    // In a real implementation, this would store the computation graph
    // for recomputation during the backward pass

    Ok(inputs)
}

#[pyfunction]
fn _set_seed(seed: u64) {
    use rand::SeedableRng;
    // Initialize or re-initialize the global seeded RNG with the given seed
    if let Some(rng_lock) = SEEDED_RNG.get() {
        // Re-seed the existing RNG
        if let Ok(mut rng) = rng_lock.lock() {
            *rng = rand::rngs::StdRng::seed_from_u64(seed);
        }
    } else {
        // Initialize the RNG for the first time
        SEEDED_RNG.get_or_init(|| Mutex::new(rand::rngs::StdRng::seed_from_u64(seed)));
    }
}

// Helper function to get a random f32 from the global seeded RNG
// Falls back to rand::random() if the RNG is not initialized
pub fn random_f32() -> f32 {
    if let Some(rng_lock) = SEEDED_RNG.get() {
        if let Ok(mut rng) = rng_lock.try_lock() {
            return rng.gen::<f32>();
        }
    }
    // Fallback to non-deterministic random if RNG is not initialized
    rand::random::<f32>()
}

#[cfg(feature = "parallel")]
use rayon::ThreadPoolBuilder;

#[pyfunction]
#[cfg(feature = "parallel")]
fn _set_num_threads(n: i32) {
    if n > 0 {
        ThreadPoolBuilder::new()
            .num_threads(n as usize)
            .build_global()
            .expect("Failed to set rayon thread pool");
    }
}

#[pyfunction]
#[cfg(feature = "parallel")]
fn _get_num_threads() -> usize {
    rayon::current_num_threads()
}

#[pyfunction]
#[cfg(not(feature = "parallel"))]
fn _set_num_threads(_n: i32) {}

#[pyfunction]
#[cfg(not(feature = "parallel"))]
fn _get_num_threads() -> usize {
    1
}

#[pyfunction]
fn _set_default_device(device: String) {
    if let Some(device) = Device::from_str(&device) {
        set_default_device_internal(device);
    }
}

#[pyfunction]
fn allocator_stats() -> String {
    storage_allocator_stats()
}

#[pyfunction]
fn list_registered_ops() -> Vec<String> {
    dispatcher_list_ops()
}

#[pyfunction]
fn clamp(a: &PyTensor, min_val: f32, max_val: f32) -> PyTensor {
    PyTensor::from_tensor(a.inner.clamp(min_val, max_val))
}

#[pyfunction]
fn pow(a: &PyTensor, exponent: f32) -> PyTensor {
    PyTensor::from_tensor(a.inner.pow(exponent))
}

#[pyclass]
struct Linear {
    inner: nn::linear::Linear,
}

#[pymethods]
impl Linear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias = true))]
    fn new(in_features: i64, out_features: i64, bias: bool) -> Self {
        Linear {
            inner: nn::linear::Linear::new(in_features, out_features, bias),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(PyTensor::from_tensor)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, PyTensor)> {
        self.inner
            .named_parameters()
            .into_iter()
            .map(|(n, t)| (n, PyTensor::from_tensor(t)))
            .collect()
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn train(&self) {
        self.inner.train_mode();
    }

    fn eval(&self) {
        self.inner.eval_mode();
    }

    fn is_training(&self) -> bool {
        self.inner.is_training()
    }

    #[pyo3(signature = (device_id))]
    #[allow(clippy::wrong_self_convention)]
    fn to_gpu(&mut self, device_id: usize) {
        self.inner.weight = self.inner.weight.to_gpu(device_id);
        self.inner.bias = self.inner.bias.as_ref().map(|b| b.to_gpu(device_id));
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = weight.inner;
    }

    fn set_bias(&mut self, bias: Option<PyTensor>) {
        self.inner.bias = bias.map(|t| t.inner);
    }

    #[classmethod]
    fn from_weights(_cls: &Bound<'_, PyType>, weight: PyTensor, bias: Option<PyTensor>) -> Self {
        let weight_shape = weight.inner.shape();
        let out_features = weight_shape[0];
        let in_features = weight_shape[1];
        let has_bias = bias.is_some();
        let mut inner = nn::linear::Linear::new(in_features, out_features, has_bias);
        inner.weight = weight.inner;
        if let Some(b) = bias {
            inner.bias = Some(b.inner);
        }
        Linear { inner }
    }
}

#[pyclass]
struct Conv2d {
    inner: nn::conv::Conv2d,
}

#[pymethods]
impl Conv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
        bias: bool,
    ) -> Self {
        Conv2d {
            inner: nn::conv::Conv2d::new(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            ),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(PyTensor::from_tensor)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, PyTensor)> {
        self.inner
            .named_parameters()
            .into_iter()
            .map(|(n, t)| (n, PyTensor::from_tensor(t)))
            .collect()
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn train(&self) {
        self.inner.train_mode();
    }

    fn eval(&self) {
        self.inner.eval_mode();
    }

    #[pyo3(signature = (device_id))]
    #[allow(clippy::wrong_self_convention)]
    fn to_gpu(&mut self, device_id: usize) {
        self.inner.weight = self.inner.weight.to_gpu(device_id);
        self.inner.bias = self.inner.bias.as_ref().map(|b| b.to_gpu(device_id));
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = weight.inner;
    }

    fn set_bias(&mut self, bias: Option<PyTensor>) {
        self.inner.bias = bias.map(|t| t.inner);
    }

    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        _cls: &Bound<'_, PyType>,
        weight: PyTensor,
        bias: Option<PyTensor>,
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
    ) -> Self {
        let mut inner = nn::conv::Conv2d::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias.is_some(),
        );
        inner.weight = weight.inner;
        if let Some(b) = bias {
            inner.bias = Some(b.inner);
        }
        Conv2d { inner }
    }
}

#[pyclass]
struct MaxPool2d {
    inner: nn::pooling::MaxPool2d,
}

#[pymethods]
impl MaxPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=2, padding=1, dilation=1))]
    fn new(kernel_size: i64, stride: i64, padding: i64, dilation: i64) -> Self {
        MaxPool2d {
            inner: nn::pooling::MaxPool2d::new(kernel_size, stride, padding, dilation),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(PyTensor::from_tensor)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, PyTensor)> {
        self.inner
            .named_parameters()
            .into_iter()
            .map(|(n, t)| (n, PyTensor::from_tensor(t)))
            .collect()
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn train(&self) {
        self.inner.train_mode();
    }

    fn eval(&self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct LayerNorm {
    inner: nn::norm::LayerNorm,
}

#[pymethods]
impl LayerNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps = 1e-5))]
    fn new(normalized_shape: i64, eps: f64) -> Self {
        LayerNorm {
            inner: nn::norm::LayerNorm::new(normalized_shape, eps),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(PyTensor::from_tensor)
            .collect()
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = Some(weight.inner);
    }

    fn set_bias(&mut self, bias: Option<PyTensor>) {
        self.inner.bias = bias.map(|t| t.inner);
    }

    #[classmethod]
    fn from_weights(
        _cls: &Bound<'_, PyType>,
        weight: PyTensor,
        bias: PyTensor,
        normalized_shape: i64,
        eps: f64,
    ) -> Self {
        let mut inner = nn::norm::LayerNorm::new(normalized_shape, eps);
        inner.weight = Some(weight.inner);
        inner.bias = Some(bias.inner);
        LayerNorm { inner }
    }
}

#[pyclass]
struct BatchNorm1d {
    inner: nn::norm::BatchNorm1d,
}

#[pymethods]
impl BatchNorm1d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1))]
    fn new(num_features: i64, eps: f64, momentum: f64) -> Self {
        BatchNorm1d {
            inner: nn::norm::BatchNorm1d::new(num_features, eps, momentum),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(PyTensor::from_tensor)
            .collect()
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn train(&self) {
        self.inner.train_mode();
    }

    fn eval(&self) {
        self.inner.eval_mode();
    }

    fn is_training(&self) -> bool {
        self.inner.is_training()
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = Some(weight.inner);
    }

    fn set_bias(&mut self, bias: Option<PyTensor>) {
        self.inner.bias = bias.map(|t| t.inner);
    }

    fn set_running_mean(&mut self, running_mean: PyTensor) {
        self.inner.running_mean = Arc::new(RwLock::new(running_mean.inner));
    }

    fn set_running_var(&mut self, running_var: PyTensor) {
        self.inner.running_var = Arc::new(RwLock::new(running_var.inner));
    }

    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        _cls: &Bound<'_, PyType>,
        weight: PyTensor,
        bias: PyTensor,
        running_mean: PyTensor,
        running_var: PyTensor,
        num_features: i64,
        eps: f64,
        momentum: f64,
    ) -> Self {
        let mut inner = nn::norm::BatchNorm1d::new(num_features, eps, momentum);
        inner.weight = Some(weight.inner);
        inner.bias = Some(bias.inner);
        inner.running_mean = Arc::new(RwLock::new(running_mean.inner));
        inner.running_var = Arc::new(RwLock::new(running_var.inner));
        BatchNorm1d { inner }
    }
}

#[pyclass]
struct Dropout {
    inner: nn::dropout::Dropout,
}

#[pymethods]
impl Dropout {
    #[new]
    fn new(p: f64) -> Self {
        Dropout {
            inner: nn::dropout::Dropout::new(p),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn train(&self) {
        self.inner.train_mode();
    }

    fn eval(&self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct Embedding {
    inner: nn::embedding::Embedding,
}

#[pymethods]
impl Embedding {
    #[new]
    fn new(num_embeddings: i64, embedding_dim: i64) -> Self {
        Embedding {
            inner: nn::embedding::Embedding::new(num_embeddings, embedding_dim),
        }
    }

    fn __call__(&self, indices: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&indices.inner))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(PyTensor::from_tensor)
            .collect()
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = weight.inner;
    }

    #[classmethod]
    fn from_weights(
        _cls: &Bound<'_, PyType>,
        weight: PyTensor,
        num_embeddings: i64,
        embedding_dim: i64,
    ) -> Self {
        let mut inner = nn::embedding::Embedding::new(num_embeddings, embedding_dim);
        inner.weight = weight.inner;
        Embedding { inner }
    }
}

#[pyclass]
struct RMSNorm {
    inner: nn::norm::RMSNorm,
}

#[pymethods]
impl RMSNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps = 1e-5))]
    fn new(normalized_shape: i64, eps: f32) -> Self {
        RMSNorm {
            inner: nn::norm::RMSNorm::new(normalized_shape, eps),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner.parameters().into_iter().map(PyTensor::from_tensor).collect()
    }

    fn named_parameters(&self) -> Vec<(String, PyTensor)> {
        self.inner.named_parameters().into_iter().map(|(n, t)| (n, PyTensor::from_tensor(t))).collect()
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn train(&mut self) {
        self.inner.train_mode();
    }

    fn eval(&mut self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct GroupNorm {
    inner: nn::norm::GroupNorm,
}

#[pymethods]
impl GroupNorm {
    #[new]
    #[pyo3(signature = (num_groups, num_channels, eps = 1e-5))]
    fn new(num_groups: i64, num_channels: i64, eps: f32) -> Self {
        GroupNorm {
            inner: nn::norm::GroupNorm::new(num_groups, num_channels, eps),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner.parameters().into_iter().map(PyTensor::from_tensor).collect()
    }

    fn named_parameters(&self) -> Vec<(String, PyTensor)> {
        self.inner.named_parameters().into_iter().map(|(n, t)| (n, PyTensor::from_tensor(t))).collect()
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn train(&mut self) {
        self.inner.train_mode();
    }

    fn eval(&mut self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct BatchNorm2d {
    inner: nn::norm::BatchNorm2d,
}

#[pymethods]
impl BatchNorm2d {
    #[new]
    #[pyo3(signature = (num_features, eps = 1e-5, momentum = 0.1))]
    fn new(num_features: i64, eps: f32, momentum: f32) -> Self {
        BatchNorm2d {
            inner: nn::norm::BatchNorm2d::new(num_features, eps, momentum),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner.parameters().into_iter().map(PyTensor::from_tensor).collect()
    }

    fn named_parameters(&self) -> Vec<(String, PyTensor)> {
        self.inner.named_parameters().into_iter().map(|(n, t)| (n, PyTensor::from_tensor(t))).collect()
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn train(&mut self) {
        self.inner.train_mode();
    }

    fn eval(&mut self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct ReLU;

#[pymethods]
impl ReLU {
    #[new]
    fn new() -> Self {
        ReLU
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.relu())
    }
}

#[pyclass]
struct Gelu;

#[pymethods]
impl Gelu {
    #[new]
    fn new() -> Self {
        Gelu
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.gelu())
    }
}

#[pyclass]
struct Sigmoid;

#[pymethods]
impl Sigmoid {
    #[new]
    fn new() -> Self {
        Sigmoid
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.sigmoid())
    }
}

#[pyclass]
struct Tanh;

#[pymethods]
impl Tanh {
    #[new]
    fn new() -> Self {
        Tanh
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.tanh())
    }
}

#[pyclass]
struct SiLU;

#[pymethods]
impl SiLU {
    #[new]
    fn new() -> Self {
        SiLU
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.silu())
    }
}

#[pyclass]
struct LeakyReLU {
    negative_slope: f64,
}

#[pymethods]
impl LeakyReLU {
    #[new]
    #[pyo3(signature = (negative_slope = 0.01))]
    fn new(negative_slope: f64) -> Self {
        LeakyReLU { negative_slope }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.leaky_relu(self.negative_slope as f32))
    }
}

#[pyclass]
struct Softplus {
    beta: f64,
    threshold: f64,
}

#[pymethods]
impl Softplus {
    #[new]
    #[pyo3(signature = (beta = 1.0, threshold = 20.0))]
    fn new(beta: f64, threshold: f64) -> Self {
        Softplus { beta, threshold }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.softplus(self.beta as f32, self.threshold as f32))
    }
}

#[pyclass]
struct Hardswish;

#[pymethods]
impl Hardswish {
    #[new]
    fn new() -> Self {
        Hardswish
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.hardswish())
    }
}

#[pyclass(name = "Sequential_")]
struct Sequential {
    layers: Vec<Py<PyAny>>,
}

#[pymethods]
impl Sequential {
    #[new]
    fn new(layers: Vec<Py<PyAny>>) -> Self {
        Sequential { layers }
    }

    fn __call__(&self, py: Python<'_>, x: PyTensor) -> PyResult<PyTensor> {
        let mut result = x;
        for layer in &self.layers {
            let forward_method: Py<PyAny> = layer.getattr(py, "forward")?;
            let new_result = forward_method.call1(py, (result,))?;
            result = new_result.extract::<PyTensor>(py)?;
        }
        Ok(result)
    }

    fn parameters(&self, py: Python<'_>) -> PyResult<Vec<PyTensor>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            let params_method: Py<PyAny> = layer.getattr(py, "parameters")?;
            let layer_params: Vec<PyTensor> = params_method.call0(py)?.extract(py)?;
            params.extend(layer_params);
        }
        Ok(params)
    }
}

#[pyclass]
struct ModuleList {
    modules: Vec<Py<PyAny>>,
}

#[pymethods]
impl ModuleList {
    #[new]
    fn new(modules: Vec<Py<PyAny>>) -> Self {
        ModuleList { modules }
    }

    fn __getitem__(&self, idx: usize, py: Python<'_>) -> Py<PyAny> {
        self.modules[idx].clone_ref(py)
    }
}

#[pyclass]
struct PySGD {
    inner: optim::sgd::SGD,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (params, lr = 0.01, momentum = 0.0, weight_decay = 0.0))]
    fn new(params: Vec<PyTensor>, lr: f64, momentum: f64, weight_decay: f64) -> Self {
        let tensors: Vec<tensor::Tensor> = params.into_iter().map(|p| p.inner).collect();
        PySGD {
            inner: optim::sgd::SGD::new(tensors, lr, momentum, 0.0, weight_decay, false),
        }
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyDict, PyList};
        let dict = PyDict::new(py);
        dict.set_item("lr", self.inner.lr)?;
        dict.set_item("momentum", self.inner.momentum)?;
        dict.set_item("weight_decay", self.inner.weight_decay)?;
        let v_list = PyList::new(py, self.inner.velocity.iter().map(|t| PyTensor::from_tensor(t.clone())))?;
        dict.set_item("velocity", v_list)?;
        Ok(dict.into())
    }

    fn load_state_dict(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.lr = state.get_item("lr")?.extract()?;
        if let Ok(m) = state.get_item("momentum")?.extract::<f64>() {
            self.inner.momentum = m;
        }
        if let Ok(wd) = state.get_item("weight_decay")?.extract::<f64>() {
            self.inner.weight_decay = wd;
        }
        if let Ok(v_list) = state.get_item("velocity")?.extract::<Vec<PyTensor>>() {
            self.inner.velocity = v_list.into_iter().map(|p| p.inner).collect();
        }
        Ok(())
    }
}

#[pyclass]
struct PyAdam {
    inner: optim::adam::Adam,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (params, lr = 0.001, betas = None, eps = 1e-8, weight_decay = 0.0))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        betas: Option<(f64, f64)>,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let tensors: Vec<tensor::Tensor> = params.into_iter().map(|p| p.inner).collect();
        let betas = betas.unwrap_or((0.9, 0.999));
        PyAdam {
            inner: optim::adam::Adam::new(tensors, lr, betas, eps, weight_decay, false),
        }
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyDict, PyList};
        let dict = PyDict::new(py);
        dict.set_item("lr", self.inner.lr)?;
        dict.set_item("betas", self.inner.betas)?;
        dict.set_item("eps", self.inner.eps)?;
        dict.set_item("weight_decay", self.inner.weight_decay)?;
        dict.set_item("amsgrad", self.inner.amsgrad)?;
        let steps = PyList::new(py, self.inner.step.iter().copied())?;
        dict.set_item("step", steps)?;
        let m_list = PyList::new(py, self.inner.m.iter().map(|t| PyTensor::from_tensor(t.clone())))?;
        dict.set_item("m", m_list)?;
        let v_list = PyList::new(py, self.inner.v.iter().map(|t| PyTensor::from_tensor(t.clone())))?;
        dict.set_item("v", v_list)?;
        let v_hat_list = PyList::new(py, self.inner.v_hat.iter().map(|t| PyTensor::from_tensor(t.clone())))?;
        dict.set_item("v_hat", v_hat_list)?;
        Ok(dict.into())
    }

    fn load_state_dict(&mut self, _py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        
        self.inner.lr = state.get_item("lr")?.extract()?;
        self.inner.betas = state.get_item("betas")?.extract()?;
        self.inner.eps = state.get_item("eps")?.extract()?;
        self.inner.weight_decay = state.get_item("weight_decay")?.extract()?;
        if let Ok(ams) = state.get_item("amsgrad")?.extract::<bool>() {
            self.inner.amsgrad = ams;
        }
        if let Ok(steps) = state.get_item("step")?.extract::<Vec<u64>>() {
            self.inner.step = steps;
        }
        if let Ok(m_list) = state.get_item("m")?.extract::<Vec<PyTensor>>() {
            self.inner.m = m_list.into_iter().map(|p| p.inner).collect();
        }
        if let Ok(v_list) = state.get_item("v")?.extract::<Vec<PyTensor>>() {
            self.inner.v = v_list.into_iter().map(|p| p.inner).collect();
        }
        if let Ok(v_hat_list) = state.get_item("v_hat")?.extract::<Vec<PyTensor>>() {
            self.inner.v_hat = v_hat_list.into_iter().map(|p| p.inner).collect();
        }
        Ok(())
    }
}

#[pyclass]
struct PyAdamW {
    inner: optim::adamw::AdamW,
}

#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (params, lr = 0.001, betas = None, eps = 1e-8, weight_decay = 0.0))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        betas: Option<(f64, f64)>,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let tensors: Vec<tensor::Tensor> = params.into_iter().map(|p| p.inner).collect();
        let betas = betas.unwrap_or((0.9, 0.999));
        PyAdamW {
            inner: optim::adamw::AdamW::new(tensors, lr, betas, eps, weight_decay, false),
        }
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyDict, PyList};
        let dict = PyDict::new(py);
        dict.set_item("lr", self.inner.lr)?;
        dict.set_item("betas", self.inner.betas)?;
        dict.set_item("eps", self.inner.eps)?;
        dict.set_item("weight_decay", self.inner.weight_decay)?;
        let steps = PyList::new(py, self.inner.step.iter().copied())?;
        dict.set_item("step", steps)?;
        let m_list = PyList::new(py, self.inner.m.iter().map(|t| PyTensor::from_tensor(t.clone())))?;
        dict.set_item("m", m_list)?;
        let v_list = PyList::new(py, self.inner.v.iter().map(|t| PyTensor::from_tensor(t.clone())))?;
        dict.set_item("v", v_list)?;
        Ok(dict.into())
    }

    fn load_state_dict(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.lr = state.get_item("lr")?.extract()?;
        self.inner.betas = state.get_item("betas")?.extract()?;
        self.inner.eps = state.get_item("eps")?.extract()?;
        self.inner.weight_decay = state.get_item("weight_decay")?.extract()?;
        if let Ok(steps) = state.get_item("step")?.extract::<Vec<u64>>() {
            self.inner.step = steps;
        }
        if let Ok(m_list) = state.get_item("m")?.extract::<Vec<PyTensor>>() {
            self.inner.m = m_list.into_iter().map(|p| p.inner).collect();
        }
        if let Ok(v_list) = state.get_item("v")?.extract::<Vec<PyTensor>>() {
            self.inner.v = v_list.into_iter().map(|p| p.inner).collect();
        }
        Ok(())
    }
}

#[pyclass]
struct PyMuon {
    inner: optim::muon::Muon,
}

#[pymethods]
impl PyMuon {
    #[new]
    #[pyo3(signature = (params, lr = 0.025, momentum = 0.95, weight_decay = 0.0, nesterov = true))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
        nesterov: bool,
    ) -> Self {
        let tensors: Vec<tensor::Tensor> = params.into_iter().map(|p| p.inner).collect();
        PyMuon {
            inner: optim::muon::Muon::new(tensors, lr, momentum, weight_decay, nesterov),
        }
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyDict, PyList};
        let dict = PyDict::new(py);
        dict.set_item("lr", self.inner.lr)?;
        dict.set_item("momentum", self.inner.momentum)?;
        dict.set_item("weight_decay", self.inner.weight_decay)?;
        dict.set_item("nesterov", self.inner.nesterov)?;
        let m_list = PyList::new(py, self.inner.m.iter().map(|t| PyTensor::from_tensor(t.clone())))?;
        dict.set_item("m", m_list)?;
        Ok(dict.into())
    }

    fn load_state_dict(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.lr = state.get_item("lr")?.extract()?;
        if let Ok(m) = state.get_item("momentum")?.extract::<f64>() {
            self.inner.momentum = m;
        }
        if let Ok(wd) = state.get_item("weight_decay")?.extract::<f64>() {
            self.inner.weight_decay = wd;
        }
        if let Ok(n) = state.get_item("nesterov")?.extract::<bool>() {
            self.inner.nesterov = n;
        }
        if let Ok(m_list) = state.get_item("m")?.extract::<Vec<PyTensor>>() {
            self.inner.m = m_list.into_iter().map(|p| p.inner).collect();
        }
        Ok(())
    }
}

#[pyclass]
struct PyTransformerEncoder {
    inner: nn::transformer::TransformerEncoder,
}

#[pymethods]
impl PyTransformerEncoder {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vocab_size: i64,
        max_seq_len: i64,
        d_model: i64,
        num_heads: i64,
        num_layers: i64,
        ff_dim: i64,
        num_classes: i64,
        dropout_p: f32,
    ) -> Self {
        PyTransformerEncoder {
            inner: nn::transformer::TransformerEncoder::new(
                vocab_size,
                max_seq_len,
                d_model,
                num_heads,
                num_layers,
                ff_dim,
                num_classes,
                dropout_p,
            ),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(PyTensor::from_tensor)
            .collect()
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn train(&mut self) {
        self.inner.train_mode();
    }

    fn eval(&mut self) {
        self.inner.eval_mode();
    }
}

#[pyfunction]
fn save_model(model: &PyTransformerEncoder, path: String) -> PyResult<()> {
    io::serialize::save_model(&model.inner, &path).map_err(|e| {
        IoError::new_err(format!("Failed to save model: {}", e))
    })
}

#[pyfunction]
fn load_model(path: String) -> PyResult<HashMap<String, PyTensor>> {
    let state_dict = io::serialize::load_model(&path, None).map_err(|e| {
        IoError::new_err(format!("Failed to load model: {}", e))
    })?;
    Ok(state_dict
        .into_iter()
        .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
        .collect())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    // Register exception hierarchy
    m.add("FastnnError", py.get_type::<FastnnError>())?;
    m.add("ShapeError", py.get_type::<ShapeError>())?;
    m.add("DtypeError", py.get_type::<DtypeError>())?;
    m.add("DeviceError", py.get_type::<DeviceError>())?;
    m.add("AutogradError", py.get_type::<AutogradError>())?;
    m.add("OptimizerError", py.get_type::<OptimizerError>())?;
    m.add("IoError", py.get_type::<IoError>())?;
    m.add("CudaError", py.get_type::<CudaError>())?;

    m.add_function(wrap_pyfunction!(tensor_from_data, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_factory, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_from_list, py)?)?;
    m.add_function(wrap_pyfunction!(zeros, py)?)?;
    m.add_function(wrap_pyfunction!(ones, py)?)?;
    m.add_function(wrap_pyfunction!(full, py)?)?;
    m.add_function(wrap_pyfunction!(arange, py)?)?;
    m.add_function(wrap_pyfunction!(linspace, py)?)?;
    m.add_function(wrap_pyfunction!(eye, py)?)?;
    m.add_function(wrap_pyfunction!(randn, py)?)?;
    m.add_function(wrap_pyfunction!(rand_uniform, py)?)?;
    m.add_function(wrap_pyfunction!(randint, py)?)?;
    m.add_function(wrap_pyfunction!(zeros_like, py)?)?;
    m.add_function(wrap_pyfunction!(ones_like, py)?)?;
    m.add_function(wrap_pyfunction!(full_like, py)?)?;
    m.add_function(wrap_pyfunction!(add, py)?)?;
    m.add_function(wrap_pyfunction!(sub, py)?)?;
    m.add_function(wrap_pyfunction!(mul, py)?)?;
    m.add_function(wrap_pyfunction!(div, py)?)?;
    m.add_function(wrap_pyfunction!(matmul, py)?)?;
    m.add_function(wrap_pyfunction!(batched_mlp_forward, py)?)?;
    m.add_function(wrap_pyfunction!(neg, py)?)?;
    m.add_function(wrap_pyfunction!(abs, py)?)?;
    m.add_function(wrap_pyfunction!(exp, py)?)?;
    m.add_function(wrap_pyfunction!(log, py)?)?;
    m.add_function(wrap_pyfunction!(sqrt, py)?)?;
    m.add_function(wrap_pyfunction!(clamp, py)?)?;
    m.add_function(wrap_pyfunction!(pow, py)?)?;
    m.add_function(wrap_pyfunction!(relu, py)?)?;
    m.add_function(wrap_pyfunction!(fused_add_relu, py)?)?;
    m.add_function(wrap_pyfunction!(fused_linear_relu, py)?)?;
    m.add_function(wrap_pyfunction!(fused_linear_gelu, py)?)?;
    m.add_function(wrap_pyfunction!(gelu, py)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, py)?)?;
    m.add_function(wrap_pyfunction!(tanh, py)?)?;
    m.add_function(wrap_pyfunction!(silu, py)?)?;
    m.add_function(wrap_pyfunction!(softmax, py)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, py)?)?;
    m.add_function(wrap_pyfunction!(embedding, py)?)?;
    m.add_function(wrap_pyfunction!(sum, py)?)?;
    m.add_function(wrap_pyfunction!(mean, py)?)?;
    m.add_function(wrap_pyfunction!(max, py)?)?;
    m.add_function(wrap_pyfunction!(min, py)?)?;
    m.add_function(wrap_pyfunction!(argmax, py)?)?;
    m.add_function(wrap_pyfunction!(argmin, py)?)?;
    m.add_function(wrap_pyfunction!(mse_loss, py)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_loss, py)?)?;
    m.add_function(wrap_pyfunction!(_no_grad_enter, py)?)?;
    m.add_function(wrap_pyfunction!(_no_grad_exit, py)?)?;
    m.add_function(wrap_pyfunction!(checkpoint, py)?)?;
    m.add_function(wrap_pyfunction!(_set_seed, py)?)?;
    m.add_function(wrap_pyfunction!(_set_num_threads, py)?)?;
    m.add_function(wrap_pyfunction!(_get_num_threads, py)?)?;
    m.add_function(wrap_pyfunction!(_set_default_device, py)?)?;
    m.add_function(wrap_pyfunction!(allocator_stats, py)?)?;
    m.add_function(wrap_pyfunction!(list_registered_ops, py)?)?;
    m.add_function(wrap_pyfunction!(save_model, py)?)?;
    m.add_function(wrap_pyfunction!(load_model, py)?)?;

    m.add_class::<Linear>()?;
    m.add_class::<Conv2d>()?;
    m.add_class::<MaxPool2d>()?;
    m.add_class::<LayerNorm>()?;
    m.add_class::<BatchNorm1d>()?;
    m.add_class::<RMSNorm>()?;
    m.add_class::<GroupNorm>()?;
    m.add_class::<BatchNorm2d>()?;
    m.add_class::<Dropout>()?;
    m.add_class::<Embedding>()?;
    m.add_class::<ReLU>()?;
    m.add_class::<Gelu>()?;
    m.add_class::<Sigmoid>()?;
    m.add_class::<Tanh>()?;
    m.add_class::<SiLU>()?;
    m.add_class::<LeakyReLU>()?;
    m.add_class::<Softplus>()?;
    m.add_class::<Hardswish>()?;
    m.add_class::<Sequential>()?;
    m.add_class::<ModuleList>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyAdamW>()?;
    m.add_class::<PyMuon>()?;
    m.add_class::<PyTransformerEncoder>()?;

    m.add_class::<PyTensor>()?;

    m.add_function(wrap_pyfunction!(bucket_allreduce, py)?)?;
    m.add_function(wrap_pyfunction!(cat, py)?)?;

    Ok(())
}

#[pyfunction]
#[allow(clippy::needless_range_loop)]
fn bucket_allreduce(mut param_groups: Vec<Vec<PyTensor>>) -> PyResult<()> {
    // Optimized implementation: average gradients across replicas
    // param_groups is a list of parameter lists, one per replica

    if param_groups.is_empty() {
        return Ok(());
    }

    let num_replicas = param_groups.len();
    let num_params = param_groups[0].len();

    // Check all replicas have the same number of parameters
    for group in &param_groups {
        if group.len() != num_params {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All replicas must have the same number of parameters",
            ));
        }
    }

    // Pre-allocate gradients vector to avoid repeated allocations
    let mut gradients = Vec::with_capacity(num_replicas);

    // For each parameter index, average gradients across replicas
    for param_idx in 0..num_params {
        // Clear gradients vector for reuse (keeps capacity)
        gradients.clear();

        // Collect gradients from all replicas
        let mut all_have_grad = true;
        for replica_idx in 0..num_replicas {
            let param = &param_groups[replica_idx][param_idx];
            if let Some(grad) = param.grad() {
                gradients.push(grad.inner.clone());
            } else {
                // If any replica has no gradient, skip this parameter
                all_have_grad = false;
                break;
            }
        }

        if !all_have_grad {
            continue;
        }

        // If we collected gradients from all replicas, average them
        if gradients.len() == num_replicas {
            // Compute average gradient: sum all gradients and divide by num_replicas
            // Start with first gradient as base, accumulate in-place
            let mut avg_grad = gradients[0].clone();

            // Add remaining gradients in-place (skip first since it's already in avg_grad)
            for i in 1..gradients.len() {
                avg_grad.add_(&gradients[i]);
            }

            // Divide by number of replicas in-place
            avg_grad.mul_scalar_(1.0 / num_replicas as f32);

            // Set the averaged gradient back to all parameters
            // Create the PyTensor once and reuse for all replicas
            let avg_grad_py = PyTensor::from_tensor(avg_grad.clone());
            for replica_idx in 0..num_replicas {
                let param = &mut param_groups[replica_idx][param_idx];
                param.set_grad(Some(avg_grad_py.clone()));
            }
        }
    }

    Ok(())
}

#[pyfunction]
fn cat(tensors: Vec<PyTensor>, dim: i32) -> PyTensor {
    let tensors: Vec<tensor::Tensor> = tensors.into_iter().map(|p| p.inner).collect();
    PyTensor::from_tensor(tensor::Tensor::cat(&tensors, dim))
}
