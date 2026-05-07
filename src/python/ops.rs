/// Helper to dispatch a unary operation
fn dispatch_unary(op: &str, tensor: &Tensor) -> Tensor {
    let dispatch_key = dispatcher::device_to_dispatch_key(tensor.device());
    let result = dispatcher::dispatch(op, dispatch_key, &[tensor]);
    result.into_iter().next().unwrap()
}

/// Helper to dispatch a binary operation
fn dispatch_binary(op: &str, a: &Tensor, b: &Tensor) -> Tensor {
    let dispatch_key = dispatcher::device_to_dispatch_key(a.device());
    let result = dispatcher::dispatch(op, dispatch_key, &[a, b]);
    result.into_iter().next().unwrap()
}

/// Helper to dispatch an operation with variable number of arguments
fn dispatch_op(op: &str, args: &[&Tensor]) -> Tensor {
    let dispatch_key = dispatcher::device_to_dispatch_key(args[0].device());
    let result = dispatcher::dispatch(op, dispatch_key, args);
    result.into_iter().next().unwrap()
}

/// Helper to wrap loss function output with autograd support
fn wrap_loss_with_autograd(output: Tensor, input: &Tensor, backward_fn: impl FnOnce() -> std::sync::Arc<dyn autograd::Node>) -> PyTensor {
    if autograd::is_grad_enabled() && input.requires_grad() {
        let edges = autograd::make_edge(input);
        let backward = backward_fn();
        let mut meta = autograd::AutogradMeta::new_non_leaf(true);
        meta.grad_fn = Some(backward);
        let mut output = output.clone();
        Arc::make_mut(&mut output.inner).autograd_meta =
            Some(std::sync::Arc::new(std::sync::Mutex::new(meta)));
        PyTensor::from_tensor(output)
    } else {
        PyTensor::from_tensor(output)
    }
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
                    x = dispatch_unary("relu", &x);
                }
                "sigmoid" => {
                    x = dispatch_unary("sigmoid", &x);
                }
                "tanh" => {
                    x = dispatch_unary("tanh", &x);
                }
                _ => {}
            }
        }
    }

    PyTensor::from_tensor(x)
}

#[pyfunction]
fn neg(py: Python<'_>, a: &PyTensor) -> PyTensor {
    let a_inner = a.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.neg()))
}

#[pyfunction]
fn abs(py: Python<'_>, a: &PyTensor) -> PyTensor {
    let a_inner = a.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.abs()))
}

#[pyfunction]
fn exp(py: Python<'_>, a: &PyTensor) -> PyTensor {
    let a_inner = a.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.exp()))
}

#[pyfunction]
fn log(py: Python<'_>, a: &PyTensor) -> PyTensor {
    let a_inner = a.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.ln()))
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
    PyTensor::from_tensor(dispatch_binary("fused_add_relu", &a.inner, &b.inner))
}

#[pyfunction]
fn fused_linear_relu(x: &PyTensor, w: &PyTensor, bias: Option<&PyTensor>) -> PyTensor {
    let args: Vec<_> = match bias {
        Some(b) => vec![&x.inner, &w.inner, &b.inner],
        None => vec![&x.inner, &w.inner],
    };
    PyTensor::from_tensor(dispatch_op("fused_linear_relu", &args))
}

#[pyfunction]
fn fused_linear_gelu(x: &PyTensor, w: &PyTensor, bias: Option<&PyTensor>) -> PyTensor {
    let args: Vec<_> = match bias {
        Some(b) => vec![&x.inner, &w.inner, &b.inner],
        None => vec![&x.inner, &w.inner],
    };
    PyTensor::from_tensor(dispatch_op("fused_linear_gelu", &args))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn fused_conv_bn_silu(
    x: &PyTensor,
    w: &PyTensor,
    bias: Option<&PyTensor>,
    bn_weight: &PyTensor,
    bn_bias: &PyTensor,
    bn_running_mean: &PyTensor,
    bn_running_var: &PyTensor,
    stride: &PyTensor,
    padding: &PyTensor,
    dilation: &PyTensor,
    groups: &PyTensor,
    eps: &PyTensor,
) -> PyTensor {
    let mut args: Vec<&Tensor> = Vec::new();
    args.push(&x.inner);
    args.push(&w.inner);
    if let Some(b) = bias {
        args.push(&b.inner);
    }
    args.push(&bn_weight.inner);
    args.push(&bn_bias.inner);
    args.push(&bn_running_mean.inner);
    args.push(&bn_running_var.inner);
    args.push(&stride.inner);
    args.push(&padding.inner);
    args.push(&dilation.inner);
    args.push(&groups.inner);
    args.push(&eps.inner);
    PyTensor::from_tensor(dispatch_op("fused_conv_bn_silu", &args))
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
    let args = [&a.inner, &Tensor::from_scalar(dim as f32)];
    PyTensor::from_tensor(dispatch_op("log_softmax", &args))
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
#[pyo3(signature = (a, other))]
fn maximum(a: &PyTensor, other: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.maximum(&other.inner))
}

#[pyfunction]
#[pyo3(signature = (a, other))]
fn minimum(a: &PyTensor, other: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(a.inner.minimum(&other.inner))
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn min(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyTensor {
    let dim = dim.unwrap_or(0);
    let args = [
        &a.inner,
        &Tensor::from_scalar(dim as f32),
        &Tensor::from_scalar(if keepdim { 1.0 } else { 0.0 }),
    ];
    PyTensor::from_tensor(dispatch_op("min", &args))
}

#[pyfunction]
#[pyo3(signature = (a, dim = None))]
fn argmax(a: &PyTensor, dim: Option<i32>) -> PyTensor {
    let dim = dim.unwrap_or(0);
    let args = [
        &a.inner,
        &Tensor::from_scalar(dim as f32),
        &Tensor::from_scalar(1.0),
    ];
    PyTensor::from_tensor(dispatch_op("max", &args))
}

#[pyfunction]
#[pyo3(signature = (a, dim = None))]
fn argmin(a: &PyTensor, dim: Option<i32>) -> PyTensor {
    let dim = dim.unwrap_or(0);
    let args = [
        &a.inner,
        &Tensor::from_scalar(dim as f32),
        &Tensor::from_scalar(1.0),
    ];
    PyTensor::from_tensor(dispatch_op("min", &args))
}

#[pyfunction]
#[pyo3(signature = (pred, target, reduction = None))]
fn mse_loss(pred: &PyTensor, target: &PyTensor, reduction: Option<String>) -> PyTensor {
    let reduction = reduction.unwrap_or_else(|| "mean".to_string());
    let reduction_code = match reduction.as_str() {
        "none" => 0.0,
        "mean" => 1.0,
        "sum" => 2.0,
        _ => 1.0,
    };
    let args = [&pred.inner, &target.inner, &Tensor::from_scalar(reduction_code)];
    let output = dispatch_op("mse_loss", &args);

    wrap_loss_with_autograd(output, &pred.inner, || {
        std::sync::Arc::new(autograd::MSELossBackward::new(
            pred.inner.clone(),
            target.inner.clone(),
            reduction,
            autograd::make_edge(&pred.inner),
        ))
    })
}

#[pyfunction]
#[pyo3(signature = (pred, target, reduction = None))]
fn cross_entropy_loss(pred: &PyTensor, target: &PyTensor, reduction: Option<String>) -> PyTensor {
    let reduction = reduction.unwrap_or_else(|| "mean".to_string());
    let reduction_code = match reduction.as_str() {
        "none" => 0.0,
        "mean" => 1.0,
        "sum" => 2.0,
        _ => 1.0,
    };
    let args = [&pred.inner, &target.inner, &Tensor::from_scalar(reduction_code)];
    let output = dispatch_op("cross_entropy_loss", &args);

    wrap_loss_with_autograd(output, &pred.inner, || {
        std::sync::Arc::new(autograd::CrossEntropyBackward::new(
            pred.inner.clone(),
            target.inner.clone(),
            reduction,
            autograd::make_edge(&pred.inner),
        ))
    })
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
    set_seeded_rng(seed);
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
    let tensors: Vec<core_tensor::Tensor> = tensors.into_iter().map(|p| p.inner).collect();
    PyTensor::from_tensor(core_tensor::Tensor::cat(&tensors, dim))
}

#[pyfunction]
fn stack(tensors: Vec<PyTensor>, dim: i32) -> PyTensor {
    let tensors: Vec<core_tensor::Tensor> = tensors.into_iter().map(|p| p.inner).collect();
    PyTensor::from_tensor(core_tensor::Tensor::stack(&tensors, dim))
}

#[pyfunction]
fn bce_with_logits(input: &PyTensor, target: &PyTensor) -> PyTensor {
    let args = [&input.inner, &target.inner];
    PyTensor::from_tensor(dispatch_op("bce_with_logits", &args))
}

#[pyfunction]
fn huber_loss(input: &PyTensor, target: &PyTensor, delta: f32) -> PyTensor {
    let delta_t = core_tensor::Tensor::from_scalar(delta);
    let args = [&input.inner, &target.inner, &delta_t];
    PyTensor::from_tensor(dispatch_op("huber_loss", &args))
}

#[pyfunction]
fn einsum(equation: &str, tensors: Vec<PyTensor>) -> PyTensor {
    let tensors: Vec<core_tensor::Tensor> = tensors.into_iter().map(|p| p.inner).collect();
    PyTensor::from_tensor(core_tensor::einsum(equation, &tensors))
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
    let x_inner = &x.inner;
    let shape = x_inner.shape();
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

    // Compute output dimensions
    let out_height = (in_height + 2 * padding_us - kernel_h) / stride_us + 1;
    let out_width = (in_width + 2 * padding_us - kernel_w) / stride_us + 1;

    // Call the internal im2col_kernel
    let col_tensor = unsafe {
        crate::kernels::cpu::im2col_kernel(
            x_inner,
            kernel_h,
            kernel_w,
            stride_us,
            padding_us,
            dilation_us,
            out_height,
            out_width,
        )
    };

    Ok(PyTensor::from_tensor(col_tensor))
}

#[pyfunction]
#[pyo3(signature = (q, k, v, scale = None, causal = None))]
fn flash_attention(
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    scale: Option<f32>,
    causal: Option<bool>,
) -> PyTensor {
    let scale = scale.unwrap_or((q.inner.shape()[3] as f32).sqrt().recip());
    let causal = if causal.unwrap_or(false) { 1.0 } else { 0.0 };
    let args = [
        &q.inner,
        &k.inner,
        &v.inner,
        &core_tensor::Tensor::from_scalar(scale),
        &core_tensor::Tensor::from_scalar(causal),
    ];
    PyTensor::from_tensor(dispatch_op("flash_attention", &args))
}

#[pyfunction]
fn clip_grad_norm_(tensors: Vec<PyTensor>, max_norm: f32, norm_type: f32) -> f32 {
    let tensors: Vec<core_tensor::Tensor> = tensors.into_iter().map(|p| p.inner).collect();
    core_tensor::clip_grad_norm_(&tensors, max_norm, norm_type)
}

#[pyfunction]
fn clip_grad_value_(tensors: Vec<PyTensor>, clip_value: f32) {
    let tensors: Vec<core_tensor::Tensor> = tensors.into_iter().map(|p| p.inner).collect();
    core_tensor::clip_grad_value_(&tensors, clip_value);
}
