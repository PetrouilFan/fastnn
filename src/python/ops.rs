/// Helper to dispatch a unary operation
fn dispatch_unary(op: &str, tensor: &Tensor) -> PyResult<Tensor> {
    let dispatch_key = dispatcher::device_to_dispatch_key(tensor.device());
    let result = dispatcher::try_dispatch(op, dispatch_key, &[tensor])?;
    result.into_iter().next().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "dispatch returned empty result for unary op '{}'", op
        ))
    })
}

/// Helper to dispatch a binary operation
fn dispatch_binary(op: &str, a: &Tensor, b: &Tensor) -> PyResult<Tensor> {
    let dispatch_key = dispatcher::device_to_dispatch_key(a.device());
    let result = dispatcher::try_dispatch(op, dispatch_key, &[a, b])?;
    result.into_iter().next().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "dispatch returned empty result for binary op '{}'", op
        ))
    })
}

/// Helper to dispatch an operation with variable number of arguments
fn dispatch_op(op: &str, args: &[&Tensor]) -> PyResult<Tensor> {
    let dispatch_key = dispatcher::device_to_dispatch_key(args[0].device());
    let result = dispatcher::try_dispatch(op, dispatch_key, args)?;
    result.into_iter().next().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "dispatch returned empty result for op '{}'", op
        ))
    })
}

/// Helper to wrap loss function output with autograd support
fn wrap_loss_with_autograd(output: Tensor, input: &Tensor, backward_fn: impl FnOnce() -> std::sync::Arc<dyn autograd::Node>) -> PyTensor {
    if autograd::is_grad_enabled() && input.requires_grad() {
        let _edges = autograd::make_edge(input);
        let backward = backward_fn();
        let mut meta = autograd::AutogradMeta::new_non_leaf(true);
        meta.grad_fn = Some(backward);
        let mut output = output;
        Arc::make_mut(&mut output.inner).autograd_meta =
            Some(std::sync::Arc::new(std::sync::Mutex::new(meta)));
        PyTensor::from_tensor(output)
    } else {
        PyTensor::from_tensor(output)
    }
}

/// Helper to convert dim to tensor (avoids repeated Tensor::from_scalar(dim as f32))
fn dim_to_tensor(dim: i32) -> Tensor {
    Tensor::from_scalar(dim as f32)
}

/// Macro for unary operations: fn op(py, a) -> PyTensor { PyTensor::from_tensor(a.inner.op()) }
macro_rules! unary_op {
    ($name:ident, $method:ident) => {
        #[pyfunction]
        fn $name(py: Python<'_>, a: &PyTensor) -> PyTensor {
            let a_inner = a.inner.clone();
            py.detach(move || PyTensor::from_tensor(a_inner.$method()))
        }
    };
}

/// Macro for binary operations: fn op(py, a, b) -> PyTensor { PyTensor::from_tensor(a.inner.op(&b.inner)) }
macro_rules! binary_op {
    ($name:ident, $method:ident) => {
        #[pyfunction]
        fn $name(py: Python<'_>, a: &PyTensor, b: &PyTensor) -> PyTensor {
            let a_inner = a.inner.clone();
            let b_inner = b.inner.clone();
            py.detach(move || PyTensor::from_tensor(a_inner.$method(&b_inner)))
        }
    };
}

/// Macro for argmax/argmin operations
macro_rules! arg_op {
    ($name:ident, $dispatch_name:ident) => {
        #[pyfunction]
        #[pyo3(signature = (a, dim = None))]
        fn $name(a: &PyTensor, dim: Option<i32>) -> PyResult<PyTensor> {
            let dim = dim.unwrap_or(0);
            let args = [&a.inner, &dim_to_tensor(dim), &Tensor::from_scalar(1.0)];
            Ok(PyTensor::from_tensor(dispatch_op(stringify!($dispatch_name), &args)?))
        }
    };
}

/// Macro for loss functions with common pattern
macro_rules! loss_fn {
    ($name:ident, $backward_type:ident) => {
        #[pyfunction]
        #[pyo3(signature = (pred, target, reduction = None))]
        fn $name(pred: &PyTensor, target: &PyTensor, reduction: Option<String>) -> PyResult<PyTensor> {
            let reduction = reduction.unwrap_or_else(|| "mean".to_string());
            let reduction_code = match reduction.as_str() {
                "none" => 0.0,
                "mean" => 1.0,
                "sum" => 2.0,
                _ => 1.0,
            };
            let args = [&pred.inner, &target.inner, &Tensor::from_scalar(reduction_code)];
            let output = dispatch_op(stringify!($name), &args)?;

            Ok(wrap_loss_with_autograd(output, &pred.inner, || {
                std::sync::Arc::new(autograd::$backward_type::new(
                    pred.inner.clone(),
                    target.inner.clone(),
                    reduction,
                    autograd::make_edge(&pred.inner),
                ))
            }))
        }
    };
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

// Binary operations using macro
binary_op!(add, add);
binary_op!(sub, sub);
binary_op!(mul, mul);
binary_op!(div, div);

#[pyfunction]
fn fused_add_relu(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(dispatch_binary("fused_add_relu", &a.inner, &b.inner)?))
}

#[pyfunction]
fn fused_linear_relu(x: &PyTensor, w: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    let args: Vec<_> = match bias {
        Some(b) => vec![&x.inner, &w.inner, &b.inner],
        None => vec![&x.inner, &w.inner],
    };
    Ok(PyTensor::from_tensor(dispatch_op("fused_linear_relu", &args)?))
}

#[pyfunction]
fn fused_linear_gelu(x: &PyTensor, w: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    let args: Vec<_> = match bias {
        Some(b) => vec![&x.inner, &w.inner, &b.inner],
        None => vec![&x.inner, &w.inner],
    };
    Ok(PyTensor::from_tensor(dispatch_op("fused_linear_gelu", &args)?))
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
) -> PyResult<PyTensor> {
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
    Ok(PyTensor::from_tensor(dispatch_op("fused_conv_bn_silu", &args)?))
}

#[pyfunction]
fn matmul(py: Python<'_>, a: &PyTensor, b: &PyTensor) -> PyTensor {
    let a_inner = a.inner.clone();
    let b_inner = b.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.matmul(&b_inner)))
}

#[pyfunction]
fn batched_mlp_forward(
    input: &PyTensor,
    weights: Vec<PyTensor>,
    biases: Vec<PyTensor>,
    activations: Vec<String>,
) -> PyResult<PyTensor> {
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
                    x = dispatch_unary("relu", &x)?;
                }
                "sigmoid" => {
                    x = dispatch_unary("sigmoid", &x)?;
                }
                "tanh" => {
                    x = dispatch_unary("tanh", &x)?;
                }
                _ => {}
            }
        }
    }

    Ok(PyTensor::from_tensor(x))
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
    let args = [&a.inner, &Tensor::from_scalar(negative_slope)];
    Ok(PyTensor::from_tensor(dispatch_op("leaky_relu", &args)?))
}

#[pyfunction]
#[pyo3(signature = (a, alpha = 1.0))]
fn elu(a: &PyTensor, alpha: f32) -> PyResult<PyTensor> {
    let args = [&a.inner, &Tensor::from_scalar(alpha)];
    Ok(PyTensor::from_tensor(dispatch_op("elu", &args)?))
}

#[pyfunction]
#[pyo3(signature = (a, beta = 1.0, threshold = 20.0))]
fn softplus(a: &PyTensor, beta: f32, threshold: f32) -> PyResult<PyTensor> {
    let args = [&a.inner, &Tensor::from_scalar(beta), &Tensor::from_scalar(threshold)];
    Ok(PyTensor::from_tensor(dispatch_op("softplus", &args)?))
}

#[pyfunction]
fn hardswish(a: &PyTensor) -> PyResult<PyTensor> {
    Ok(PyTensor::from_tensor(dispatch_unary("hardswish", &a.inner)?))
}

#[pyfunction]
fn softmax(py: Python<'_>, a: &PyTensor, dim: i32) -> PyTensor {
    let a_inner = a.inner.clone();
    py.detach(move || PyTensor::from_tensor(a_inner.softmax(dim)))
}

#[pyfunction]
fn log_softmax(a: &PyTensor, dim: i32) -> PyResult<PyTensor> {
    let args = [&a.inner, &dim_to_tensor(dim)];
    Ok(PyTensor::from_tensor(dispatch_op("log_softmax", &args)?))
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
fn min(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyResult<PyTensor> {
    let dim = dim.unwrap_or(0);
    let args = [
        &a.inner,
        &dim_to_tensor(dim),
        &Tensor::from_scalar(if keepdim { 1.0 } else { 0.0 }),
    ];
    Ok(PyTensor::from_tensor(dispatch_op("min", &args)?))
}

// Argmax and argmin using macro
arg_op!(argmax, max);
arg_op!(argmin, min);

// Loss functions using macro
loss_fn!(mse_loss, MSELossBackward);
loss_fn!(cross_entropy_loss, CrossEntropyBackward);

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
fn checkpoint(fn_name: &str, inputs: Vec<PyTensor>) -> PyResult<Vec<PyTensor>> {
    let inputs_inner: Vec<Tensor> = inputs.iter().map(|p| p.inner.clone()).collect();
    let requires_grad = inputs.iter().any(|p| p.inner.requires_grad());

    if !requires_grad || !autograd::is_grad_enabled() {
        return Ok(inputs);
    }

    let num_inputs = inputs_inner.len();
    let forward_fn = fn_name.to_string();

    let output = PyTensor::from_tensor(dispatch_op(&forward_fn, &inputs_inner.iter().collect::<Vec<_>>())?);

    let output_inner = output.inner.clone();

    if output_inner.requires_grad() {
        let mut meta = autograd::AutogradMeta::new_non_leaf(true);
        let edges = autograd::make_edge(&inputs[0].inner);
        meta.grad_fn = Some(std::sync::Arc::new(checkpoint_impl::CheckpointNode::new(
            forward_fn,
            inputs_inner,
            edges,
        )));
        let mut out = output_inner.clone();
        Arc::make_mut(&mut out.inner).autograd_meta =
            Some(std::sync::Arc::new(std::sync::Mutex::new(meta)));
        let mut result = vec![PyTensor::from_tensor(out)];
        for i in 1..num_inputs {
            result.push(inputs[i].clone());
        }
        Ok(result)
    } else {
        Ok(inputs)
    }
}

mod checkpoint_impl {
    use super::{dispatch_op, Tensor};
    use crate::autograd::{Edge, Node};

    pub struct CheckpointNode {
        pub fn_name: String,
        pub inputs: Vec<Tensor>,
        pub edges: Vec<Edge>,
    }

    impl CheckpointNode {
        pub fn new(fn_name: String, inputs: Vec<Tensor>, edges: Vec<Edge>) -> Self {
            CheckpointNode {
                fn_name,
                inputs,
                edges,
            }
        }
    }

    impl Node for CheckpointNode {
        fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
            let args: Vec<&Tensor> = self.inputs.iter().collect();
            let _ = dispatch_op(&self.fn_name, &args);

            grad_outputs.into_iter().map(|g| g).collect()
        }

        fn next_edges(&self) -> &[Edge] {
            &self.edges
        }

        fn num_inputs(&self) -> usize {
            self.inputs.len()
        }

        fn name(&self) -> &str {
            "CheckpointBackward"
        }

        fn inputs(&self) -> &[Tensor] {
            &self.inputs
        }
    }
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

    // For each parameter index, average gradients across replicas
    for param_idx in 0..num_params {
        // Collect gradients from all replicas
        let mut gradients: Vec<core_tensor::Tensor> = Vec::with_capacity(num_replicas);
        let mut all_have_grad = true;
        for replica_idx in 0..num_replicas {
            let param = &param_groups[replica_idx][param_idx];
            if let Some(grad) = param.grad() {
                gradients.push(grad.inner.clone());
            } else {
                all_have_grad = false;
                break;
            }
        }

        if !all_have_grad || gradients.len() != num_replicas {
            continue;
        }

        // Compute average gradient in-place
        let mut avg_grad = gradients[0].clone();
        for i in 1..gradients.len() {
            avg_grad.add_(&gradients[i]);
        }
        avg_grad.mul_scalar_(1.0 / num_replicas as f32);

        // Set the averaged gradient to all parameters (share the same tensor via Arc)
        let avg_grad_arc = avg_grad;
        for replica_idx in 0..num_replicas {
            let param = &mut param_groups[replica_idx][param_idx];
            param.set_grad(Some(PyTensor::from_tensor(avg_grad_arc.clone())));
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
fn bce_with_logits(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    let args = [&input.inner, &target.inner];
    let output = dispatch_op("bce_with_logits", &args)?;

    Ok(wrap_loss_with_autograd(output, &input.inner, || {
        std::sync::Arc::new(autograd::BCEWithLogitsBackward::new(
            input.inner.clone(),
            target.inner.clone(),
            autograd::make_edge(&input.inner),
        ))
    }))
}

#[pyfunction]
fn huber_loss(input: &PyTensor, target: &PyTensor, delta: f32) -> PyResult<PyTensor> {
    let delta_t = core_tensor::Tensor::from_scalar(delta);
    let args = [&input.inner, &target.inner, &delta_t];
    let output = dispatch_op("huber_loss", &args)?;

    Ok(wrap_loss_with_autograd(output, &input.inner, || {
        std::sync::Arc::new(autograd::HuberLossBackward::new(
            input.inner.clone(),
            target.inner.clone(),
            delta,
            autograd::make_edge(&input.inner),
        ))
    }))
}

// Tensor manipulation ops
#[pyfunction]
fn where_(condition: &PyTensor, x: &PyTensor, y: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(x.inner.where_tensor(&condition.inner, &y.inner))
}

#[pyfunction]
fn repeat(tensor: &PyTensor, repeats: Vec<i64>) -> PyTensor {
    PyTensor::from_tensor(tensor.inner.repeat(&repeats))
}

#[pyfunction]
fn expand(tensor: &PyTensor, shape: Vec<i64>) -> PyTensor {
    PyTensor::from_tensor(tensor.inner.expand(shape))
}

#[pyfunction]
#[pyo3(signature = (tensor, dim, start, end, step = 1))]
fn slice(tensor: &PyTensor, dim: usize, start: i64, end: i64, step: i64) -> PyTensor {
    PyTensor::from_tensor(tensor.inner.slice(dim, start, end, step))
}

#[pyfunction]
fn topk(tensor: &PyTensor, _k: i64, dim: i64) -> PyResult<(PyTensor, PyTensor)> {
    let values = tensor.inner.max(dim as i32, false);
    let args = [&tensor.inner, &dim_to_tensor(dim as i32), &Tensor::from_scalar(1.0)];
    let indices = dispatch_op("max", &args)?;
    Ok((PyTensor::from_tensor(values), PyTensor::from_tensor(indices)))
}

#[pyfunction]
#[pyo3(signature = (tensor, axis, indices))]
fn gather(tensor: &PyTensor, axis: i64, indices: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(tensor.inner.gather(axis, &indices.inner))
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
    // SAFETY: `x_inner` is a valid tensor reference; the function parameters
    // (kernel dimensions, strides, padding, dilation) were computed from the
    // tensor's shape above and are valid.
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
) -> PyResult<PyTensor> {
    let scale = scale.unwrap_or((q.inner.shape()[3] as f32).sqrt().recip());
    let causal = if causal.unwrap_or(false) { 1.0 } else { 0.0 };
    let args = [
        &q.inner,
        &k.inner,
        &v.inner,
        &core_tensor::Tensor::from_scalar(scale),
        &core_tensor::Tensor::from_scalar(causal),
    ];
    Ok(PyTensor::from_tensor(dispatch_op("flash_attention", &args)?))
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

#[pyfunction]
fn cumsum(tensor: &PyTensor, dim: i64, exclusive: bool, reverse: bool) -> PyTensor {
    PyTensor::from_tensor(tensor.inner.cumsum(dim, exclusive, reverse))
}

#[pyfunction]
fn erf(tensor: &PyTensor) -> PyTensor {
    PyTensor::from_tensor(tensor.inner.erf())
}

#[allow(dead_code)]
#[pyfunction]
fn nonzero(tensor: &PyTensor) -> Vec<Vec<i64>> {
    tensor.inner.nonzero()
}


