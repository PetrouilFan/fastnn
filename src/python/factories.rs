use pyo3::exceptions::PyValueError;

// Helper to resolve device from optional string parameter
fn resolve_device(device: Option<String>) -> Device {
    device
        .as_ref()
        .and_then(|s| Device::from_str_label(s))
        .unwrap_or_else(get_default_device)
}

// Helper to resolve dtype from optional string parameter
fn resolve_dtype(dtype: Option<String>) -> DType {
    dtype
        .and_then(|s| DType::from_str_label(&s))
        .unwrap_or(DType::F32)
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
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::from_vec_with_device(
        data, shape, device,
    )))
}

#[pyfunction]
fn tensor_factory(data: Vec<f32>, shape: Vec<i64>) -> PyTensor {
    PyTensor::from_tensor(Tensor::from_vec(data, shape))
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None, device = None))]
fn zeros(shape: Vec<i64>, dtype: Option<String>, device: Option<String>) -> PyTensor {
    let dtype = resolve_dtype(dtype);
    let device = resolve_device(device);
    PyTensor::from_tensor(Tensor::zeros(shape, dtype, device))
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None, device = None))]
fn empty(shape: Vec<i64>, dtype: Option<String>, device: Option<String>) -> PyTensor {
    let dtype = resolve_dtype(dtype);
    let device = resolve_device(device);
    PyTensor::from_tensor(Tensor::empty(shape, dtype, device))
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None, device = None))]
fn ones(shape: Vec<i64>, dtype: Option<String>, device: Option<String>) -> PyTensor {
    let dtype = resolve_dtype(dtype);
    let device = resolve_device(device);
    PyTensor::from_tensor(Tensor::ones(shape, dtype, device))
}

#[pyfunction]
#[pyo3(signature = (shape, value, dtype = None, device = None))]
fn full(shape: Vec<i64>, value: f32, dtype: Option<String>, device: Option<String>) -> PyTensor {
    let dtype = resolve_dtype(dtype);
    let device = resolve_device(device);
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
    let values: Vec<f32> = (0..numel)
        .map(|i| (start_f64 + i as f64 * step_f64) as f32)
        .collect();
    let device = resolve_device(device);
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
    let device = resolve_device(device);
    PyTensor::from_tensor(Tensor::from_vec_with_device(
        values,
        vec![steps as i64],
        device,
    ))
}

#[pyfunction]
#[pyo3(signature = (n, m=None, device=None))]
fn eye(n: i64, m: Option<i64>, device: Option<String>) -> PyResult<PyTensor> {
    let m = m.unwrap_or(n);
    if n <= 0 || m <= 0 {
        return Err(PyValueError::new_err(format!(
            "eye(): n and m must be positive, got n={}, m={}", n, m
        )));
    }
    let mut values = vec![0.0f32; (n * m) as usize];
    for i in 0..n.min(m) {
        values[(i * m + i) as usize] = 1.0;
    }
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::from_vec_with_device(values, vec![n, m], device)))
}

#[pyfunction]
#[pyo3(signature = (shape, device = None))]
fn randn(shape: Vec<i64>, device: Option<String>) -> PyTensor {
    let numel: i64 = shape.iter().product();
    let values: Vec<f32> = (0..numel as usize)
        .map(|_| {
            let u1 = crate::random_f32();
            let u2 = crate::random_f32();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();
    let device = resolve_device(device);
    PyTensor::from_tensor(Tensor::from_vec_with_device(values, shape, device))
}

#[pyfunction]
#[pyo3(signature = (shape, device = None))]
fn rand_uniform(shape: Vec<i64>, device: Option<String>) -> PyTensor {
    let numel: i64 = shape.iter().product();
    let values: Vec<f32> = (0..numel as usize).map(|_| crate::random_f32()).collect();
    let device = resolve_device(device);
    PyTensor::from_tensor(Tensor::from_vec_with_device(values, shape, device))
}

#[pyfunction]
#[pyo3(signature = (shape, low, high, device = None))]
fn randint(shape: Vec<i64>, low: i32, high: i32, device: Option<String>) -> PyResult<PyTensor> {
    if high <= low {
        return Err(PyValueError::new_err(format!(
            "randint(): high must be greater than low, got low={}, high={}", low, high
        )));
    }
    let numel: i64 = shape.iter().product();
    let range = (high - low) as u32;
    let values: Vec<f32> = if range == 0 {
        vec![low as f32; numel as usize]
    } else {
        (0..numel as usize)
            .map(|_| ((crate::random_f32() * range as f32) as i32 + low) as f32)
            .collect()
    };
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::from_vec_with_device(values, shape, device)))
}

#[pyfunction]
#[pyo3(signature = (data, shape, dtype = None))]
fn tensor_from_bytes(data: Vec<u8>, shape: Vec<i64>, dtype: Option<String>) -> PyResult<PyTensor> {
    let n: usize = shape.iter().product::<i64>() as usize;
    let expected_bytes = n * 4;
    if data.len() != expected_bytes {
        return Err(PyValueError::new_err(format!(
            "tensor_from_bytes: expected {expected_bytes} bytes for shape {:?}, got {}",
            shape, data.len()
        )));
    }
    let _dtype = resolve_dtype(dtype);
    let floats: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    Ok(PyTensor::from_tensor(Tensor::from_vec(floats, shape)))
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

