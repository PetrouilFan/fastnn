fn reduce_mean_all(t: &Tensor) -> Tensor {
    let ndim = t.ndim();
    let mut result = t.clone();
    for _ in 0..ndim {
        result = result.mean(0, false);
    }
    // Keep Python scalar reductions as rank-1 tensors for the public API.
    if result.shape().is_empty() {
        result.reshape(vec![1])
    } else {
        result
    }
}

fn reduce_sum_all(t: &Tensor) -> Tensor {
    let ndim = t.ndim();
    let mut result = t.clone();
    for _ in 0..ndim {
        result = result.sum(0, false);
    }
    // Keep Python scalar reductions as rank-1 tensors for the public API.
    if result.shape().is_empty() {
        result.reshape(vec![1])
    } else {
        result
    }
}

fn reduce_max_all(t: &Tensor) -> Tensor {
    let ndim = t.ndim();
    let mut result = t.clone();
    for _ in 0..ndim {
        result = result.max(0, false);
    }
    result
}

fn reduce_min_all(t: &Tensor) -> Tensor {
    // Decompose min(x) = -max(-x). This is correct for all finite f32 inputs.
    // For inputs near the negation boundary (~-3.4e38), negation may overflow to
    // +inf, producing an incorrect -inf result. A native ReduceMin op would avoid
    // this, but the IR currently only has ReduceMax.
    let ndim = t.ndim();
    let mut result = t.clone();
    for _ in 0..ndim {
        result = result.neg().max(0, false).neg();
    }
    result
}

/// Macro for unary operations: fn op(a) -> PyTensor
macro_rules! unary_op {
    ($name:ident, $method:ident) => {
        #[pyfunction]
        fn $name(a: &PyTensor) -> PyTensor {
            PyTensor::from_tensor(a.inner.$method())
        }
    };
}

/// Macro for recoverable binary operations.
macro_rules! binary_op {
    ($name:ident, $method:ident) => {
        #[pyfunction]
        fn $name(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
            Ok(PyTensor::from_tensor(
                a.inner
                    .$method(&b.inner)
                    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
            ))
        }
    };
}

/// Macro for argmax/argmin operations
macro_rules! arg_op {
    ($name:ident, $method:ident) => {
        #[pyfunction]
        #[pyo3(signature = (a, dim = None))]
        fn $name(a: &PyTensor, dim: Option<i32>) -> PyResult<PyTensor> {
            let rank = a.inner.ndim() as i64;
            let requested = i64::from(dim.unwrap_or(0));
            let normalized = if requested < 0 {
                rank + requested
            } else {
                requested
            };
            if normalized < 0 || normalized >= rank {
                return Err(crate::error::FastnnError::shape(format!(
                    "{} dimension {requested} is out of range for {rank} dimensions",
                    stringify!($name)
                ))
                .into());
            }
            Ok(PyTensor::from_tensor(
                a.inner.try_argmax(Some(normalized as usize))?,
            ))
        }
    };
}

#[pyfunction]
fn full_like(tensor: &PyTensor, value: f32) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(Tensor::try_full(
        tensor.inner.shape(),
        value,
        tensor.inner.dtype(),
        tensor.inner.device(),
    )?))
}

// Binary operations using macro
binary_op!(add, try_add);
binary_op!(sub, try_sub);
binary_op!(mul, try_mul);
binary_op!(div, try_div);

#[pyfunction]
fn fused_add_relu(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let added = a
        .inner
        .try_add(&b.inner)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let output = added
        .try_relu()
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyTensor::from_tensor(output))
}

#[pyfunction]
fn fused_linear_relu(
    x: &PyTensor,
    w: &PyTensor,
    bias: Option<&PyTensor>,
) -> PyResult<PyTensor> {
    let mut output = x
        .inner
        .try_matmul(&w.inner)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    if let Some(bias) = bias {
        output = output
            .try_add(&bias.inner)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    }
    output = output
        .try_relu()
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyTensor::from_tensor(output))
}

#[pyfunction]
fn fused_linear_gelu(
    x: &PyTensor,
    w: &PyTensor,
    bias: Option<&PyTensor>,
) -> PyResult<PyTensor> {
    let mut output = x
        .inner
        .try_matmul(&w.inner)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    if let Some(bias) = bias {
        output = output
            .try_add(&bias.inner)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    }
    output = output
        .try_gelu()
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyTensor::from_tensor(output))
}

#[pyfunction]
fn fused_conv_bn_silu(
    x: &PyTensor,
    w: &PyTensor,
    bias: Option<&PyTensor>,
    bn_weight: &PyTensor,
    bn_bias: &PyTensor,
    bn_running_mean: &PyTensor,
    bn_running_var: &PyTensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    eps: f64,
) -> PyResult<PyTensor> {
    let has_bias = bias.is_some();
    let mut inputs: Vec<&Tensor> = vec![&x.inner, &w.inner];
    if let Some(b) = bias {
        inputs.push(&b.inner);
    }
    inputs.push(&bn_weight.inner);
    inputs.push(&bn_bias.inner);
    inputs.push(&bn_running_mean.inner);
    inputs.push(&bn_running_var.inner);

    let outputs = Tensor::exec_aot(&inputs, |g, ins| {
        let mut offset = 2usize;
        let conv_out = g.conv2d_with_params(&ins[0], &ins[1], stride, padding, dilation, groups);
        let after_conv = if has_bias {
            let biased = g.add(&conv_out, &ins[offset]);
            offset += 1;
            biased
        } else {
            conv_out
        };
        let bn_out = g.batch_norm(
            &after_conv,
            &ins[offset],
            &ins[offset + 1],
            &ins[offset + 2],
            &ins[offset + 3],
            eps,
        );
        vec![g.silu(&bn_out)]
    })
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "fused_conv_bn_silu AOT execution failed: {}",
            e
        ))
    })?;

    Ok(PyTensor::from_tensor(outputs.into_iter().next().unwrap()))
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
    let x_inner = input.inner.clone();
    let weights_inner: Vec<_> = weights.iter().map(|p| p.inner.clone()).collect();
    let biases_inner: Vec<_> = biases.iter().map(|p| Some(p.inner.clone())).collect();
    let activations_clone = activations.clone();
    let mut x = x_inner;
    for i in 0..weights_inner.len() {
        let w = &weights_inner[i];
        let b = biases_inner.get(i).and_then(|b| b.as_ref());
        x = x.matmul(&w.transpose(0, 1));
        if let Some(bias) = b {
            x = x.add(bias);
        }
        if i < activations_clone.len() {
            let act = &activations_clone[i];
            match act.as_str() {
                "relu" => {
                    x = x.relu();
                }
                "sigmoid" => {
                    x = x.sigmoid();
                }
                "tanh" => {
                    x = x.tanh();
                }
                _ => {}
            }
        }
    }
    PyTensor::from_tensor(x)
}

// Unary operations using macro
unary_op!(neg, neg);
unary_op!(abs, abs);
unary_op!(exp, exp);
unary_op!(log, ln);
unary_op!(sqrt, sqrt);
unary_op!(relu, relu);
unary_op!(gelu, gelu);
unary_op!(sigmoid, sigmoid);
unary_op!(tanh, tanh);
unary_op!(silu, silu);

// Activation ops
#[pyfunction]
#[pyo3(signature = (a, negative_slope = 0.01))]
fn leaky_relu(a: &PyTensor, negative_slope: f32) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(
        a.inner
            .try_leaky_relu(negative_slope)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
    ))
}

#[pyfunction]
#[pyo3(signature = (a, alpha = 1.0))]
fn elu(a: &PyTensor, alpha: f32) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(
        a.inner
            .try_elu(alpha)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
    ))
}

#[pyfunction]
#[pyo3(signature = (a, beta = 1.0, threshold = 20.0))]
fn softplus(a: &PyTensor, beta: f32, threshold: f32) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(
        a.inner
            .try_softplus(beta, threshold)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
    ))
}

#[pyfunction]
fn hardswish(a: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.hardswish())
}

#[pyfunction]
fn softmax(a: &PyTensor, dim: i32) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(a.inner.try_softmax(dim).map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?))
}

#[pyfunction]
fn log_softmax(a: &PyTensor, dim: i32) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(a.inner.try_log_softmax(dim).map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?))
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn sum(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyResult<PyTensor> {
    Ok(match dim {
        Some(d) => PyTensor::from_tensor(a.inner.try_sum(d, keepdim)?),
        None => PyTensor::from_tensor(reduce_sum_all(&a.inner)),
    })
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn mean(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyResult<PyTensor> {
    Ok(match dim {
        Some(d) => PyTensor::from_tensor(a.inner.try_mean(d, keepdim)?),
        None => PyTensor::from_tensor(reduce_mean_all(&a.inner)),
    })
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn max(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyResult<PyTensor> {
    Ok(match dim {
        Some(d) => PyTensor::from_tensor(a.inner.try_max(d, keepdim)?),
        None => PyTensor::from_tensor(reduce_max_all(&a.inner)),
    })
}

#[pyfunction]
#[pyo3(signature = (a, other))]
fn maximum(a: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(a.inner.try_maximum(&other.inner)?))
}

#[pyfunction]
#[pyo3(signature = (a, other))]
fn minimum(a: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(a.inner.try_minimum(&other.inner)?))
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn min(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyResult<PyTensor> {
    Ok(match dim {
        Some(d) => PyTensor::from_tensor(a.inner.neg().try_max(d, keepdim)?.neg()),
        None => PyTensor::from_tensor(reduce_min_all(&a.inner)),
    })
}

// Argmax and argmin using macro
arg_op!(argmax, argmax);

// argmin is argmax of negated input (no Tensor::argmin method)
#[pyfunction]
#[pyo3(signature = (a, dim = None))]
fn argmin(a: &PyTensor, dim: Option<i32>) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let rank = a_inner.ndim() as i64;
    let requested = i64::from(dim.unwrap_or(0));
    let normalized = if requested < 0 { rank + requested } else { requested };
    if normalized < 0 || normalized >= rank {
        return Err(crate::error::FastnnError::shape(format!(
            "argmin dimension {requested} is out of range for {rank} dimensions"
        ))
        .into());
    }
    Ok(PyTensor::from_tensor(
        a_inner.neg().try_argmax(Some(normalized as usize))?,
    ))
}

#[pyfunction]
fn _set_seed(seed: u64) {
    set_seeded_rng(seed);
}

/// Choose how many workers the CPU backend runs on.
///
/// `n <= 0` means "as many as the machine offers, without pinning", which is
/// what the backend already does when nobody asks. This can only take effect
/// before the first parallel operation, because that is when Rayon builds its
/// global pool; afterwards the error says so instead of failing with Rayon's
/// own wording.
#[pyfunction]
#[cfg(feature = "parallel")]
fn _set_num_threads(n: i32) -> PyResult<()> {
    let threads = if n > 0 {
        n as usize
    } else {
        std::thread::available_parallelism().map_or(1, |count| count.get())
    };
    crate::backend::cpu::affinity::configure_threads(threads)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
#[cfg(feature = "parallel")]
fn _get_num_threads() -> usize {
    crate::backend::cpu::affinity::worker_count()
}

#[pyfunction]
#[cfg(not(feature = "parallel"))]
fn _set_num_threads(_n: i32) -> PyResult<()> {
    Ok(())
}

#[pyfunction]
#[cfg(not(feature = "parallel"))]
fn _get_num_threads() -> usize {
    1
}

#[pyfunction]
fn _set_default_device(device: String) {
    if let Some(device) = Device::from_str_label(&device) {
        set_default_device_internal(device);
    }
}

#[pyfunction]
fn allocator_stats() -> String {
    storage_allocator_stats()
}

#[pyfunction]
fn clear_storage_pool() -> String {
    use crate::storage_pool::get_storage_pool;
    get_storage_pool().clear();
    "pool_cleared".to_string()
}

#[pyfunction]
fn clamp(a: &PyTensor, min_val: f32, max_val: f32) -> PyTensor {
    let a_inner = a.inner.clone();
    PyTensor::from_tensor(a_inner.clamp(min_val, max_val))
}

#[pyfunction]
fn pow(a: &PyTensor, exponent: f32) -> PyTensor {
    PyTensor::from_tensor(a.inner.pow(exponent))
}

#[pyfunction]
fn cat(tensors: Vec<PyTensor>, dim: i32) -> PyResult<PyTensor> {
    let tensors_inner: Vec<core_tensor::Tensor> = tensors.into_iter().map(|p| p.inner).collect();
    Ok(PyTensor::from_tensor(core_tensor::Tensor::try_cat(
        &tensors_inner,
        dim,
    )?))
}

#[pyfunction]
fn stack(tensors: Vec<PyTensor>, dim: i32) -> PyResult<PyTensor> {
    let tensors_inner: Vec<core_tensor::Tensor> = tensors.into_iter().map(|p| p.inner).collect();
    Ok(PyTensor::from_tensor(core_tensor::Tensor::try_stack(
        &tensors_inner,
        dim,
    )?))
}

#[pyfunction]
fn where_(
    condition: &PyTensor,
    x: &PyTensor,
    y: &PyTensor,
) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(
        x.inner.try_where_tensor(&condition.inner, &y.inner)?,
    ))
}

#[pyfunction]
fn repeat(tensor: &PyTensor, repeats: Vec<i64>) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(tensor.inner.try_repeat(&repeats)?))
}

#[pyfunction]
fn expand(tensor: &PyTensor, shape: Vec<i64>) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(tensor.inner.try_expand(shape)?))
}

#[pyfunction]
#[pyo3(signature = (tensor, dim, start, end, step = 1))]
fn slice(
    tensor: &PyTensor,
    dim: usize,
    start: i64,
    end: i64,
    step: i64,
) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(
        tensor.inner.try_slice(dim, start, end, step)?,
    ))
}

#[pyfunction]
fn topk(tensor: &PyTensor, k: i64, dim: i64) -> PyResult<(PyTensor, PyTensor)> {
    let ndim = tensor.inner.ndim() as i64;
    let norm_dim = if dim < 0 { ndim + dim } else { dim };
    let outputs = Tensor::exec_aot(&[&tensor.inner], |g, ins| {
        let (values, indices) = g.topk(&ins[0], k as usize, norm_dim);
        vec![values, indices]
    })
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Tensor::topk: AOT execution failed: {}",
            e
        ))
    })?;
    Ok((
        PyTensor::from_tensor(outputs[0].clone()),
        PyTensor::from_tensor(outputs[1].clone()),
    ))
}

#[pyfunction]
#[pyo3(signature = (tensor, axis, indices))]
fn gather(tensor: &PyTensor, axis: i64, indices: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(
        tensor.inner.try_gather(axis, &indices.inner)?,
    ))
}

#[pyfunction]
fn einsum(equation: &str, tensors: Vec<PyTensor>) -> PyResult<PyTensor> {
    let tensors_inner: Vec<Tensor> = tensors.into_iter().map(|t| t.inner).collect();
    Ok(PyTensor::from_tensor(core_tensor::try_einsum(
        equation,
        &tensors_inner,
    )?))
}

#[pyfunction]
#[pyo3(signature = (x, kernel_size, stride=1, padding=0, dilation=1))]
fn im2col(
    x: &PyTensor,
    kernel_size: i64,
    stride: i64,
    padding: i64,
    dilation: i64,
) -> PyResult<PyTensor> {
    // Ensure the tensor is on CPU before accessing raw data pointer
    let x_cpu = x.inner.clone();
    let shape = x_cpu.shape();
    if shape.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input must be a 4D tensor (N, C, H, W)",
        ));
    }
    let _batch = shape[0] as usize;
    let _in_channels = shape[1] as usize;
    let in_height = shape[2] as usize;
    let in_width = shape[3] as usize;

    let kernel_h = kernel_size as usize;
    let kernel_w = kernel_size as usize;
    let stride_us = stride as usize;
    let padding_us = padding as usize;
    let dilation_us = dilation as usize;

    let out_height = (in_height + 2 * padding_us - kernel_h) / stride_us + 1;
    let out_width = (in_width + 2 * padding_us - kernel_w) / stride_us + 1;

    let batch = shape[0] as usize;
    let c = shape[1] as usize;
    let h = shape[2] as usize;
    let w = shape[3] as usize;
    let col_rows = batch * out_height * out_width;
    let col_cols = c * kernel_h * kernel_w;
    let mut col_data = vec![0.0f32; col_rows * col_cols];
    let x_data = unsafe { std::slice::from_raw_parts(x_cpu.data_ptr_f32(), batch * c * h * w) };
    for n in 0..batch {
        let input_off = n * (c * h * w);
        let col_off = n * out_height * out_width * col_cols;
        unsafe {
            crate::backend::cpu::im2col::im2col_kernel(
                &x_data[input_off..input_off + c * h * w],
                c,
                h,
                w,
                kernel_h,
                stride_us,
                padding_us,
                dilation_us,
                &mut col_data[col_off..col_off + out_height * out_width * col_cols],
            );
        }
    }
    let col_tensor = Tensor::from_vec(col_data, vec![col_rows as i64, col_cols as i64]);
    Ok(PyTensor::from_tensor(col_tensor))
}

#[allow(unused_variables)]
#[pyfunction]
#[pyo3(signature = (q, k, v, scale = None, causal = None))]
fn flash_attention(
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    scale: Option<f32>,
    causal: Option<bool>,
) -> PyTensor {
    PyTensor::from_tensor(crate::backend::cpu::flash_attn::flash_attention(
        &q.inner,
        &k.inner,
        &v.inner,
        scale,
        causal.unwrap_or(false),
    ))
}

#[pyfunction]
fn cumsum(
    tensor: &PyTensor,
    dim: i64,
    exclusive: bool,
    reverse: bool,
) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(
        tensor.inner.try_cumsum(dim, exclusive, reverse)?,
    ))
}

#[pyfunction]
fn erf(tensor: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(tensor.inner.erf())
}

