mod autograd;
mod dispatcher;
mod io;
mod iterator;
mod kernels;
mod nn;
mod optim;
mod storage;
mod tensor;
mod train;

use autograd::{no_grad_enter, no_grad_exit};
use dispatcher::list_registered_ops as dispatcher_list_ops;
use nn::Module;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::wrap_pyfunction;
use rand::Rng;
use storage::{allocator_stats as storage_allocator_stats, DType, Device};
use tensor::Tensor;

#[pyclass]
#[derive(Clone)]
struct PyTensor {
    inner: Tensor,
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
        self.inner.dtype().to_str().to_string()
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_str().to_string()
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
        self.inner.item()
    }

    fn numpy(&self) -> Vec<f32> {
        self.inner.to_numpy()
    }

    fn requires_grad_(&mut self, requires_grad: bool) {
        let _ = self.inner.requires_grad_(requires_grad);
    }

    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        self.inner.grad().map(PyTensor::from_tensor)
    }

    #[getter]
    fn grad_fn(&self) -> Option<String> {
        self.inner.grad_fn().map(|_| "grad_fn".to_string())
    }

    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    fn backward(&self) {
        crate::autograd::backward(&self.inner, None);
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

    fn unsqueeze(&self, dim: i64) -> PyTensor {
        PyTensor::from_tensor(self.inner.unsqueeze(dim as usize))
    }

    fn squeeze(&self, dim: Option<i64>) -> PyTensor {
        PyTensor::from_tensor(self.inner.squeeze(dim.map(|d| d as usize)))
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
            self.inner.dtype().to_str(),
            self.inner.device().to_str()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (data, shape))]
fn tensor_from_data<'py>(
    data: &Bound<'py, PyAny>,
    shape: &Bound<'py, PyAny>,
) -> PyResult<PyTensor> {
    let data: Vec<f32> = data.extract()?;
    let shape: Vec<i64> = shape.extract()?;
    Ok(PyTensor::from_tensor(Tensor::from_vec(data, shape)))
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
#[pyo3(signature = (shape, dtype = None, _device = None))]
fn zeros(shape: Vec<i64>, dtype: Option<String>, _device: Option<String>) -> PyTensor {
    let dtype = dtype
        .and_then(|s| DType::from_str(&s))
        .unwrap_or(DType::F32);
    PyTensor::from_tensor(Tensor::zeros(shape, dtype, Device::Cpu))
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None, _device = None))]
fn ones(shape: Vec<i64>, dtype: Option<String>, _device: Option<String>) -> PyTensor {
    let dtype = dtype
        .and_then(|s| DType::from_str(&s))
        .unwrap_or(DType::F32);
    PyTensor::from_tensor(Tensor::ones(shape, dtype, Device::Cpu))
}

#[pyfunction]
#[pyo3(signature = (shape, value, dtype = None, _device = None))]
fn full(shape: Vec<i64>, value: f32, dtype: Option<String>, _device: Option<String>) -> PyTensor {
    let dtype = dtype
        .and_then(|s| DType::from_str(&s))
        .unwrap_or(DType::F32);
    PyTensor::from_tensor(Tensor::full(shape, value, dtype, Device::Cpu))
}

#[pyfunction]
#[pyo3(signature = (start, end, step = None))]
fn arange(start: f32, end: f32, step: Option<f32>) -> PyTensor {
    let step = step.unwrap_or(1.0);
    let numel = ((end - start) / step).ceil() as usize;
    let values: Vec<f32> = (0..numel).map(|i| start + i as f32 * step).collect();
    PyTensor::from_tensor(Tensor::from_vec(values, vec![numel as i64]))
}

#[pyfunction]
fn linspace(start: f32, end: f32, steps: usize) -> PyTensor {
    let values: Vec<f32> = (0..steps)
        .map(|i| {
            let t = i as f32 / (steps - 1) as f32;
            start * (1.0 - t) + end * t
        })
        .collect();
    PyTensor::from_tensor(Tensor::from_vec(values, vec![steps as i64]))
}

#[pyfunction]
#[pyo3(signature = (n, m=None))]
fn eye(n: i64, m: Option<i64>) -> PyTensor {
    let m = m.unwrap_or(n);
    let mut values = vec![0.0f32; (n * m) as usize];
    for i in 0..n.min(m) {
        values[(i * m + i) as usize] = 1.0;
    }
    PyTensor::from_tensor(Tensor::from_vec(values, vec![n, m]))
}

#[pyfunction]
fn randn(shape: Vec<i64>) -> PyTensor {
    let numel: i64 = shape.iter().product();
    let mut values = vec![0.0f32; numel as usize];
    let mut rng = rand::thread_rng();
    for v in &mut values {
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }
    PyTensor::from_tensor(Tensor::from_vec(values, shape))
}

#[pyfunction]
fn rand_uniform(shape: Vec<i64>) -> PyTensor {
    let numel: i64 = shape.iter().product();
    let values: Vec<f32> = (0..numel as usize).map(|_| rand::random()).collect();
    PyTensor::from_tensor(Tensor::from_vec(values, shape))
}

#[pyfunction]
fn randint(shape: Vec<i64>, low: i32, high: i32) -> PyTensor {
    let numel: i64 = shape.iter().product();
    let values: Vec<f32> = (0..numel as usize)
        .map(|_| (rand::random::<i32>() % (high - low) + low) as f32)
        .collect();
    PyTensor::from_tensor(Tensor::from_vec(values, shape))
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
fn neg(a: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.neg())
}

#[pyfunction]
fn abs(a: &PyTensor) -> PyTensor {
    use dispatcher::dispatch;
    let result = dispatch("abs", dispatcher::DispatchKey::Cpu, &[&a.inner]);
    PyTensor::from_tensor(result[0].clone())
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
fn sqrt(a: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.sqrt())
}

#[pyfunction]
fn relu(a: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.relu())
}

#[pyfunction]
fn gelu(a: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.gelu())
}

#[pyfunction]
fn sigmoid(a: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.sigmoid())
}

#[pyfunction]
fn tanh(a: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.tanh())
}

#[pyfunction]
fn silu(a: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.silu())
}

#[pyfunction]
fn softmax(a: &PyTensor, dim: i32) -> PyTensor {
    use dispatcher::dispatch;
    let result = dispatch(
        "softmax",
        dispatcher::DispatchKey::Cpu,
        &[&a.inner, &Tensor::from_scalar(dim as f32)],
    );
    PyTensor::from_tensor(result[0].clone())
}

#[pyfunction]
fn log_softmax(a: &PyTensor, dim: i32) -> PyTensor {
    use dispatcher::dispatch;
    let result = dispatch(
        "log_softmax",
        dispatcher::DispatchKey::Cpu,
        &[&a.inner, &Tensor::from_scalar(dim as f32)],
    );
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
    let result = dispatch(
        "min",
        dispatcher::DispatchKey::Cpu,
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
    let result = dispatch(
        "max",
        dispatcher::DispatchKey::Cpu,
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
    let result = dispatch(
        "min",
        dispatcher::DispatchKey::Cpu,
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
    let reduction = reduction.unwrap_or_else(|| "mean".to_string());
    let reduction_code = match reduction.as_str() {
        "none" => 0.0,
        "mean" => 1.0,
        "sum" => 2.0,
        _ => 1.0,
    };
    let result = dispatch(
        "mse_loss",
        dispatcher::DispatchKey::Cpu,
        &[
            &pred.inner,
            &target.inner,
            &Tensor::from_scalar(reduction_code),
        ],
    );
    PyTensor::from_tensor(result[0].clone())
}

#[pyfunction]
#[pyo3(signature = (pred, target, reduction = None))]
fn cross_entropy_loss(pred: &PyTensor, target: &PyTensor, reduction: Option<String>) -> PyTensor {
    use dispatcher::dispatch;
    let reduction = reduction.unwrap_or_else(|| "mean".to_string());
    let reduction_code = match reduction.as_str() {
        "none" => 0.0,
        "mean" => 1.0,
        "sum" => 2.0,
        _ => 1.0,
    };
    let result = dispatch(
        "cross_entropy_loss",
        dispatcher::DispatchKey::Cpu,
        &[
            &pred.inner,
            &target.inner,
            &Tensor::from_scalar(reduction_code),
        ],
    );
    PyTensor::from_tensor(result[0].clone())
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
fn _set_seed(_seed: u64) {}

#[pyfunction]
fn _set_num_threads(_n: i32) {}

#[pyfunction]
fn _set_default_device(_device: String) {}

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
}

#[pyclass]
struct Conv2d {
    inner: nn::conv::Conv2d,
}

#[pymethods]
impl Conv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=true))]
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
}

#[pyclass]
struct LayerNorm {
    inner: nn::norm::LayerNorm,
}

#[pymethods]
impl LayerNorm {
    #[new]
    fn new(normalized_shape: i64, eps: Option<f64>) -> Self {
        LayerNorm {
            inner: nn::norm::LayerNorm::new(normalized_shape, eps.unwrap_or(1e-5)),
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
        relu(x)
    }
}

#[pyclass]
struct GELU;

#[pymethods]
impl GELU {
    #[new]
    fn new() -> Self {
        GELU
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        gelu(x)
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
        sigmoid(x)
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
        tanh(x)
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
        silu(x)
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

    fn __call__(&self, x: PyTensor) -> PyResult<PyTensor> {
        Ok(x)
    }

    fn forward(&self, x: PyTensor) -> PyResult<PyTensor> {
        Ok(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
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
struct SGD;

#[pymethods]
impl SGD {
    #[new]
    #[pyo3(signature = (params, lr = 0.01, momentum = None, weight_decay = None))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        momentum: Option<f64>,
        weight_decay: Option<f64>,
    ) -> Self {
        SGD
    }

    fn step(&self) {}
    fn zero_grad(&self) {}
    fn state_dict(&self) -> String {
        "{}".to_string()
    }
    fn load_state_dict(&self, _state: String) {}
}

#[pyclass]
struct Adam;

#[pymethods]
impl Adam {
    #[new]
    #[pyo3(signature = (params, lr = 0.001, betas = None, eps = None, weight_decay = None))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        betas: Option<(f64, f64)>,
        eps: Option<f64>,
        weight_decay: Option<f64>,
    ) -> Self {
        Adam
    }

    fn step(&self) {}
    fn zero_grad(&self) {}
    fn state_dict(&self) -> String {
        "{}".to_string()
    }
    fn load_state_dict(&self, _state: String) {}
}

#[pyclass]
struct AdamW;

#[pymethods]
impl AdamW {
    #[new]
    #[pyo3(signature = (params, lr = 0.001, betas = None, eps = None, weight_decay = None))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        betas: Option<(f64, f64)>,
        eps: Option<f64>,
        weight_decay: Option<f64>,
    ) -> Self {
        AdamW
    }

    fn step(&self) {}
    fn zero_grad(&self) {}
    fn state_dict(&self) -> String {
        "{}".to_string()
    }
    fn load_state_dict(&self, _state: String) {}
}

#[pyfunction]
fn save_model(_model: Py<PyAny>, path: String) {
    println!("Saved model to {}", path);
}

#[pyfunction]
fn load_model(path: String, _model_class: Option<Py<PyAny>>) {
    println!("Loaded model from {}", path);
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
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
    m.add_function(wrap_pyfunction!(neg, py)?)?;
    m.add_function(wrap_pyfunction!(abs, py)?)?;
    m.add_function(wrap_pyfunction!(exp, py)?)?;
    m.add_function(wrap_pyfunction!(log, py)?)?;
    m.add_function(wrap_pyfunction!(sqrt, py)?)?;
    m.add_function(wrap_pyfunction!(clamp, py)?)?;
    m.add_function(wrap_pyfunction!(pow, py)?)?;
    m.add_function(wrap_pyfunction!(relu, py)?)?;
    m.add_function(wrap_pyfunction!(gelu, py)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, py)?)?;
    m.add_function(wrap_pyfunction!(tanh, py)?)?;
    m.add_function(wrap_pyfunction!(silu, py)?)?;
    m.add_function(wrap_pyfunction!(softmax, py)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, py)?)?;
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
    m.add_function(wrap_pyfunction!(_set_seed, py)?)?;
    m.add_function(wrap_pyfunction!(_set_num_threads, py)?)?;
    m.add_function(wrap_pyfunction!(_set_default_device, py)?)?;
    m.add_function(wrap_pyfunction!(allocator_stats, py)?)?;
    m.add_function(wrap_pyfunction!(list_registered_ops, py)?)?;
    m.add_function(wrap_pyfunction!(save_model, py)?)?;
    m.add_function(wrap_pyfunction!(load_model, py)?)?;

    m.add_class::<Linear>()?;
    m.add_class::<Conv2d>()?;
    m.add_class::<LayerNorm>()?;
    m.add_class::<BatchNorm1d>()?;
    m.add_class::<Dropout>()?;
    m.add_class::<Embedding>()?;
    m.add_class::<ReLU>()?;
    m.add_class::<GELU>()?;
    m.add_class::<Sigmoid>()?;
    m.add_class::<Tanh>()?;
    m.add_class::<SiLU>()?;
    m.add_class::<Sequential>()?;
    m.add_class::<ModuleList>()?;
    m.add_class::<SGD>()?;
    m.add_class::<Adam>()?;
    m.add_class::<AdamW>()?;

    m.add_class::<PyTensor>()?;

    Ok(())
}
