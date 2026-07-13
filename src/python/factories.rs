use pyo3::exceptions::{PyMemoryError, PyValueError};
use smallvec::SmallVec;

fn try_f32_buffer(numel: usize, fill: f32, operation: &str) -> PyResult<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(numel).map_err(|error| {
        PyMemoryError::new_err(format!(
            "{operation}: unable to allocate {numel} F32 values: {error}"
        ))
    })?;
    values.resize(numel, fill);
    Ok(values)
}

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
    Ok(PyTensor::from_tensor(Tensor::try_from_vec_with_device(
        data, shape, device,
    )?))
}

#[pyfunction]
fn tensor_factory(data: Vec<f32>, shape: Vec<i64>) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(Tensor::try_from_vec(data, shape)?))
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None, device = None))]
fn zeros(shape: Vec<i64>, dtype: Option<String>, device: Option<String>) -> PyResult<PyTensor> {
    let dtype = resolve_dtype(dtype);
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::try_zeros(
        shape, dtype, device,
    )?))
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None, device = None))]
fn empty(shape: Vec<i64>, dtype: Option<String>, device: Option<String>) -> PyResult<PyTensor> {
    let dtype = resolve_dtype(dtype);
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::try_empty(
        shape, dtype, device,
    )?))
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None, device = None))]
fn ones(shape: Vec<i64>, dtype: Option<String>, device: Option<String>) -> PyResult<PyTensor> {
    let dtype = resolve_dtype(dtype);
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::try_ones(
        shape, dtype, device,
    )?))
}

#[pyfunction]
#[pyo3(signature = (shape, value, dtype = None, device = None))]
fn full(
    shape: Vec<i64>,
    value: f32,
    dtype: Option<String>,
    device: Option<String>,
) -> PyResult<PyTensor> {
    let dtype = resolve_dtype(dtype);
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::try_full(
        shape, value, dtype, device,
    )?))
}

#[pyfunction]
#[pyo3(signature = (start, end, step = None, device = None))]
fn arange(
    start: f32,
    end: f32,
    step: Option<f32>,
    device: Option<String>,
) -> PyResult<PyTensor> {
    let step = step.unwrap_or(1.0);
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(PyValueError::new_err(
            "arange(): start, end, and step must be finite",
        ));
    }
    if step == 0.0 {
        return Err(PyValueError::new_err("arange(): step must be nonzero"));
    }
    let start_f64 = start as f64;
    let end_f64 = end as f64;
    let step_f64 = step as f64;
    let span = (end_f64 - start_f64) / step_f64;
    let numel = if span <= 0.0 { 0 } else { span.ceil() as usize };
    crate::tensor::validate_tensor_shape(&[numel as i64], DType::F32)?;
    let mut values = try_f32_buffer(numel, 0.0, "arange")?;
    for (index, value) in values.iter_mut().enumerate() {
        *value = (start_f64 + index as f64 * step_f64) as f32;
    }
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::try_from_vec_with_device(
        values,
        vec![numel as i64],
        device,
    )?))
}

#[pyfunction]
#[pyo3(signature = (start, end, steps, device = None))]
fn linspace(
    start: f32,
    end: f32,
    steps: usize,
    device: Option<String>,
) -> PyResult<PyTensor> {
    if !start.is_finite() || !end.is_finite() {
        return Err(PyValueError::new_err(
            "linspace(): start and end must be finite",
        ));
    }
    if steps > i64::MAX as usize {
        return Err(PyValueError::new_err(
            "linspace(): steps exceeds the supported tensor dimension range",
        ));
    }
    crate::tensor::validate_tensor_shape(&[steps as i64], DType::F32)?;
    let mut values = try_f32_buffer(steps, 0.0, "linspace")?;
    match steps {
        0 => {}
        1 => values[0] = start,
        _ => {
            for (index, value) in values.iter_mut().enumerate() {
                let t = index as f32 / (steps - 1) as f32;
                *value = start * (1.0 - t) + end * t;
            }
        }
    }
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::try_from_vec_with_device(
        values,
        vec![steps as i64],
        device,
    )?))
}

#[pyfunction]
#[pyo3(signature = (n, m=None, device=None))]
fn eye(n: i64, m: Option<i64>, device: Option<String>) -> PyResult<PyTensor> {
    let m = m.unwrap_or(n);
    if n <= 0 || m <= 0 {
        return Err(PyValueError::new_err(format!(
            "eye(): n and m must be positive, got n={}, m={}",
            n, m
        )));
    }
    let (numel, _) = crate::tensor::validate_tensor_shape(&[n, m], DType::F32)?;
    let mut values = try_f32_buffer(numel, 0.0, "eye")?;
    for i in 0..n.min(m) {
        values[(i * m + i) as usize] = 1.0;
    }
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::from_vec_with_device(
        values,
        vec![n, m],
        device,
    )))
}

#[pyfunction]
#[pyo3(signature = (shape, device = None))]
fn randn(shape: Vec<i64>, device: Option<String>) -> PyResult<PyTensor> {
    let (numel, _) = crate::tensor::validate_tensor_shape(&shape, DType::F32)?;
    let mut values = try_f32_buffer(numel, 0.0, "randn")?;
    for value in &mut values {
        let u1 = crate::random_f32();
        let u2 = crate::random_f32();
        *value = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::try_from_vec_with_device(
        values, shape, device,
    )?))
}

#[pyfunction]
#[pyo3(signature = (shape, device = None))]
fn rand_uniform(shape: Vec<i64>, device: Option<String>) -> PyResult<PyTensor> {
    let (numel, _) = crate::tensor::validate_tensor_shape(&shape, DType::F32)?;
    let mut values = try_f32_buffer(numel, 0.0, "rand_uniform")?;
    for value in &mut values {
        *value = crate::random_f32();
    }
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::try_from_vec_with_device(
        values, shape, device,
    )?))
}

#[pyfunction]
#[pyo3(signature = (shape, low, high, device = None))]
fn randint(shape: Vec<i64>, low: i32, high: i32, device: Option<String>) -> PyResult<PyTensor> {
    if high <= low {
        return Err(PyValueError::new_err(format!(
            "randint(): high must be greater than low, got low={}, high={}",
            low, high
        )));
    }
    let (numel, _) = crate::tensor::validate_tensor_shape(&shape, DType::F32)?;
    let range = i64::from(high) - i64::from(low);
    let mut values = try_f32_buffer(numel, 0.0, "randint")?;
    for value in &mut values {
        let offset = (f64::from(crate::random_f32()) * range as f64) as i64;
        *value = (i64::from(low) + offset.min(range - 1)) as f32;
    }
    let device = resolve_device(device);
    Ok(PyTensor::from_tensor(Tensor::try_from_vec_with_device(
        values, shape, device,
    )?))
}

#[pyfunction]
#[pyo3(signature = (data, shape, dtype = None))]
fn tensor_from_bytes(data: Vec<u8>, shape: Vec<i64>, dtype: Option<String>) -> PyResult<PyTensor> {
    let dtype = resolve_dtype(dtype);
    let (_, expected_bytes) = crate::tensor::validate_tensor_shape(&shape, dtype)?;
    if data.len() != expected_bytes {
        return Err(PyValueError::new_err(format!(
            "tensor_from_bytes: expected {expected_bytes} bytes for shape {:?} and dtype {}, got {}",
            shape, dtype.as_str(), data.len()
        )));
    }
    let sizes: SmallVec<[i64; 8]> = shape.into();
    let storage = Arc::new(crate::storage::Storage::Cpu(
        crate::storage::CpuStorage::from_vec(data, expected_bytes),
    ));
    let tensor = Tensor::new(crate::tensor::TensorImpl::try_new(
        storage, sizes, dtype,
    )?);
    Ok(PyTensor::from_tensor(tensor))
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
