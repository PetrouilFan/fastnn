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

fn reduce_mean_all(t: &Tensor) -> Tensor {
    let ndim = t.ndim();
    let mut result = t.clone();
    for _ in 0..ndim {
        result = result.mean(0, false);
    }
    result
}

fn reduce_sum_all(t: &Tensor) -> Tensor {
    let ndim = t.ndim();
    let mut result = t.clone();
    for _ in 0..ndim {
        result = result.sum(0, false);
    }
    result
}

fn checkpoint_op(name: &str, inputs: &[&Tensor]) -> PyResult<Tensor> {
    match name {
        "add" => Ok(inputs[0].add(inputs[1])),
        "sub" => Ok(inputs[0].sub(inputs[1])),
        "mul" => Ok(inputs[0].mul(inputs[1])),
        "div" => Ok(inputs[0].div(inputs[1])),
        "matmul" => Ok(inputs[0].matmul(inputs[1])),
        "relu" => Ok(inputs[0].relu()),
        "sigmoid" => Ok(inputs[0].sigmoid()),
        "tanh" => Ok(inputs[0].tanh()),
        "gelu" => Ok(inputs[0].gelu()),
        "silu" => Ok(inputs[0].silu()),
        "neg" => Ok(inputs[0].neg()),
        "exp" => Ok(inputs[0].exp()),
        "sqrt" => Ok(inputs[0].sqrt()),
        "abs" => Ok(inputs[0].abs()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown checkpoint operation: {}", name)
        ))
    }
}

/// Macro for unary operations: fn op(a) -> PyResult<PyTensor>
macro_rules! unary_op {
    ($name:ident, $method:ident) => {
        #[pyfunction]
        fn $name(a: &PyTensor) -> PyResult<PyTensor> {
            let a_inner = a.inner.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                PyTensor::from_tensor(a_inner.$method())
            }));
            result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
                format!("{} operation failed", stringify!($name))
            ))
        }
    };
}

/// Macro for binary operations: fn op(a, b) -> PyResult<PyTensor>
macro_rules! binary_op {
    ($name:ident, $method:ident) => {
        #[pyfunction]
        fn $name(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
            let a_inner = a.inner.clone();
            let b_inner = b.inner.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                PyTensor::from_tensor(a_inner.$method(&b_inner))
            }));
            result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
                format!("{} operation failed", stringify!($name))
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
            let dim = dim.unwrap_or(0) as usize;
            let a_inner = a.inner.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                PyTensor::from_tensor(a_inner.$method(Some(dim)))
            }));
            result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
                format!("{} operation failed", stringify!($name))
            ))
        }
    };
}

#[pyfunction]
fn full_like(tensor: &PyTensor, value: f32) -> PyResult<PyTensor> {
    let t_inner = tensor.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(Tensor::full(
            t_inner.shape(),
            value,
            t_inner.dtype(),
            t_inner.device(),
        ))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "full_like operation failed"
    ))
}

// Binary operations using macro
binary_op!(add, add);
binary_op!(sub, sub);
binary_op!(mul, mul);
binary_op!(div, div);

#[pyfunction]
fn fused_add_relu(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let b_inner = b.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.add(&b_inner).relu())
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "fused_add_relu operation failed"
    ))
}

#[pyfunction]
fn fused_linear_relu(x: &PyTensor, w: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    let x_inner = x.inner.clone();
    let w_inner = w.inner.clone();
    let b_inner = bias.map(|b| b.inner.clone());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let out = x_inner.matmul(&w_inner);
        let out = match &b_inner {
            Some(b) => out.add(b),
            None => out,
        };
        PyTensor::from_tensor(out.relu())
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "fused_linear_relu operation failed"
    ))
}

#[pyfunction]
fn fused_linear_gelu(x: &PyTensor, w: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    let x_inner = x.inner.clone();
    let w_inner = w.inner.clone();
    let b_inner = bias.map(|b| b.inner.clone());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let out = x_inner.matmul(&w_inner);
        let out = match &b_inner {
            Some(b) => out.add(b),
            None => out,
        };
        PyTensor::from_tensor(out.gelu())
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "fused_linear_gelu operation failed"
    ))
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
    let stride_val = stride.inner.item() as usize;
    let padding_val = padding.inner.item() as usize;
    let dilation_val = dilation.inner.item() as usize;
    let groups_val = groups.inner.item() as usize;
    let eps_val = eps.inner.item() as f64;

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
        let conv_out = g.conv2d_with_params(&ins[0], &ins[1], stride_val, padding_val, dilation_val, groups_val);
        let after_conv = if has_bias {
            let biased = g.add(&conv_out, &ins[offset]);
            offset += 1;
            biased
        } else {
            conv_out
        };
        let bn_out = g.batch_norm(&after_conv, &ins[offset], &ins[offset + 1], &ins[offset + 2], &ins[offset + 3], eps_val);
        vec![g.silu(&bn_out)]
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
        format!("fused_conv_bn_silu AOT execution failed: {}", e)
    ))?;

    Ok(PyTensor::from_tensor(outputs.into_iter().next().unwrap()))
}

#[pyfunction]
fn matmul(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let b_inner = b.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.matmul(&b_inner))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "matmul operation failed"
    ))
}

#[pyfunction]
fn batched_mlp_forward(
    input: &PyTensor,
    weights: Vec<PyTensor>,
    biases: Vec<PyTensor>,
    activations: Vec<String>,
) -> PyResult<PyTensor> {
    let x_inner = input.inner.clone();
    let weights_inner: Vec<_> = weights.iter().map(|p| p.inner.clone()).collect();
    let biases_inner: Vec<_> = biases.iter().map(|p| Some(p.inner.clone())).collect();
    let activations_clone = activations.clone();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
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
                    "relu" => { x = x.relu(); }
                    "sigmoid" => { x = x.sigmoid(); }
                    "tanh" => { x = x.tanh(); }
                    _ => {}
                }
            }
        }
        PyTensor::from_tensor(x)
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "batched_mlp_forward operation failed"
    ))
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
    let a_inner = a.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.leaky_relu(negative_slope))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "leaky_relu operation failed"
    ))
}

#[pyfunction]
#[pyo3(signature = (a, alpha = 1.0))]
fn elu(a: &PyTensor, alpha: f32) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.elu(alpha))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "elu operation failed"
    ))
}

#[pyfunction]
#[pyo3(signature = (a, beta = 1.0, threshold = 20.0))]
fn softplus(a: &PyTensor, beta: f32, threshold: f32) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.softplus(beta, threshold))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "softplus operation failed"
    ))
}

#[pyfunction]
fn hardswish(a: &PyTensor) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.hardswish())
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "hardswish operation failed"
    ))
}

#[pyfunction]
fn softmax(a: &PyTensor, dim: i32) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.softmax(dim))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "softmax operation failed"
    ))
}

#[pyfunction]
fn log_softmax(a: &PyTensor, dim: i32) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.log_softmax(dim))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "log_softmax operation failed"
    ))
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn sum(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let d = dim.unwrap_or(0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.sum(d, keepdim))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("sum operation failed"))
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn mean(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let d = dim.unwrap_or(0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.mean(d, keepdim))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("mean operation failed"))
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn max(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let d = dim.unwrap_or(0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.max(d, keepdim))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("max operation failed"))
}

#[pyfunction]
#[pyo3(signature = (a, other))]
fn maximum(a: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let b_inner = other.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.maximum(&b_inner))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("maximum operation failed"))
}

#[pyfunction]
#[pyo3(signature = (a, other))]
fn minimum(a: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let b_inner = other.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.minimum(&b_inner))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("minimum operation failed"))
}

#[pyfunction]
#[pyo3(signature = (a, dim = None, keepdim = false))]
fn min(a: &PyTensor, dim: Option<i32>, keepdim: bool) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let d = dim.unwrap_or(0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.neg().max(d, keepdim).neg())
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("min operation failed"))
}

// Argmax and argmin using macro
arg_op!(argmax, argmax);

// argmin is argmax of negated input (no Tensor::argmin method)
#[pyfunction]
#[pyo3(signature = (a, dim = None))]
fn argmin(a: &PyTensor, dim: Option<i32>) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let d = dim.unwrap_or(0) as usize;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.neg().argmax(Some(d)))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("argmin operation failed"))
}

// Loss functions
#[pyfunction]
#[pyo3(signature = (pred, target, reduction = None))]
fn mse_loss(pred: &PyTensor, target: &PyTensor, reduction: Option<String>) -> PyResult<PyTensor> {
    let pred_inner = pred.inner.clone();
    let target_inner = target.inner.clone();
    let reduction_str = reduction.unwrap_or_else(|| "mean".to_string());

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let diff = &pred_inner - &target_inner;
        let squared = diff.pow(2.0);
        let output = match reduction_str.as_str() {
            "none" => squared,
            "mean" => reduce_mean_all(&squared),
            "sum" => reduce_sum_all(&squared),
            _ => reduce_mean_all(&squared),
        };
        output
    }));
    let output = result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "mse_loss computation failed"
    ))?;

    let edges = autograd::make_edges(&pred.inner, &target.inner);
    let inputs = vec![pred.inner.clone(), target.inner.clone()];
    Ok(wrap_loss_with_autograd(output, &pred.inner, move || {
        std::sync::Arc::new(autograd::MSELossBackward::new(edges, inputs))
    }))
}

#[pyfunction]
#[pyo3(signature = (pred, target, reduction = None))]
fn cross_entropy_loss(pred: &PyTensor, target: &PyTensor, reduction: Option<String>) -> PyResult<PyTensor> {
    let pred_inner = pred.inner.clone();
    let target_inner = target.inner.clone();
    let reduction_str = reduction.unwrap_or_else(|| "mean".to_string());

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let log_probs = pred_inner.log_softmax(-1);
        let target_flat = target_inner.reshape(vec![-1, 1]);
        let gathered = log_probs.gather(1, &target_flat);
        let nll = gathered.neg().reshape(vec![pred_inner.shape()[0]]);
        match reduction_str.as_str() {
            "none" => nll,
            "mean" => reduce_mean_all(&nll),
            "sum" => reduce_sum_all(&nll),
            _ => reduce_mean_all(&nll),
        }
    }));
    let output = result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
        "cross_entropy_loss computation failed"
    ))?;

    let edges = autograd::make_edges(&pred.inner, &target.inner);
    let inputs = vec![pred.inner.clone(), target.inner.clone()];
    Ok(wrap_loss_with_autograd(output, &pred.inner, move || {
        std::sync::Arc::new(autograd::CrossEntropyBackward::new(edges, inputs))
    }))
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
fn checkpoint(fn_name: &str, inputs: Vec<PyTensor>) -> PyResult<Vec<PyTensor>> {
    let inputs_inner: Vec<Tensor> = inputs.iter().map(|p| p.inner.clone()).collect();
    let requires_grad = inputs.iter().any(|p| p.inner.requires_grad());

    if !requires_grad || !autograd::is_grad_enabled() {
        return Ok(inputs);
    }

    let num_inputs = inputs_inner.len();
    let forward_fn = fn_name.to_string();

    let inputs_refs: Vec<&Tensor> = inputs_inner.iter().collect();
    let output_tensor = checkpoint_op(&forward_fn, &inputs_refs)?;
    let output = PyTensor::from_tensor(output_tensor);

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
    use super::{checkpoint_op, Tensor};
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
            let _ = checkpoint_op(&self.fn_name, &args);
            grad_outputs.into_iter().collect()
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

#[allow(dead_code)]
#[pyfunction]
fn list_registered_ops() -> Vec<String> {
    vec![]
}

#[pyfunction]
fn clamp(a: &PyTensor, min_val: f32, max_val: f32) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.clamp(min_val, max_val))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("clamp operation failed"))
}

#[pyfunction]
fn pow(a: &PyTensor, exponent: f32) -> PyResult<PyTensor> {
    let a_inner = a.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(a_inner.pow(exponent))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("pow operation failed"))
}

#[pyfunction]
#[allow(clippy::needless_range_loop)]
fn bucket_allreduce(mut param_groups: Vec<Vec<PyTensor>>) -> PyResult<()> {
    if param_groups.is_empty() {
        return Ok(());
    }

    let num_replicas = param_groups.len();
    let num_params = param_groups[0].len();

    for group in &param_groups {
        if group.len() != num_params {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All replicas must have the same number of parameters",
            ));
        }
    }

    for param_idx in 0..num_params {
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

        let mut avg_grad = gradients[0].clone();
        for i in 1..gradients.len() {
            avg_grad.add_(&gradients[i]);
        }
        avg_grad.mul_scalar_(1.0 / num_replicas as f32);

        let avg_grad_arc = avg_grad;
        for replica_idx in 0..num_replicas {
            let param = &mut param_groups[replica_idx][param_idx];
            param.set_grad(Some(PyTensor::from_tensor(avg_grad_arc.clone())));
        }
    }

    Ok(())
}

#[pyfunction]
fn cat(tensors: Vec<PyTensor>, dim: i32) -> PyResult<PyTensor> {
    let tensors_inner: Vec<core_tensor::Tensor> = tensors.into_iter().map(|p| p.inner).collect();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(core_tensor::Tensor::cat(&tensors_inner, dim))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("cat operation failed"))
}

#[pyfunction]
fn stack(tensors: Vec<PyTensor>, dim: i32) -> PyResult<PyTensor> {
    let tensors_inner: Vec<core_tensor::Tensor> = tensors.into_iter().map(|p| p.inner).collect();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(core_tensor::Tensor::stack(&tensors_inner, dim))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("stack operation failed"))
}

#[pyfunction]
fn bce_with_logits(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    let zero = Tensor::from_scalar(0.0);
    let one = Tensor::from_scalar(1.0);
    let term1 = input.inner.maximum(&zero);
    let term2 = input.inner.mul(&target.inner);
    let term3 = one.add(&input.inner.abs().neg().exp()).ln();
    let output = term1.sub(&term2).add(&term3);

    let edges = autograd::make_edges(&input.inner, &target.inner);
    let inputs = vec![input.inner.clone(), target.inner.clone()];
    Ok(wrap_loss_with_autograd(output, &input.inner, move || {
        std::sync::Arc::new(autograd::BCEWithLogitsBackward::new(edges, inputs))
    }))
}

#[pyfunction]
fn huber_loss(input: &PyTensor, target: &PyTensor, delta: f32) -> PyResult<PyTensor> {
    let diff = &input.inner - &target.inner;
    let abs_diff = diff.abs();
    let delta_t = Tensor::from_scalar(delta);
    let zero = Tensor::from_scalar(0.0);
    let half = Tensor::from_scalar(0.5);
    let quadratic = diff.pow(2.0) * half.clone();
    let excess = (abs_diff - delta_t).maximum(&zero);
    let correction = excess.pow(2.0) * half;
    let output = quadratic - correction;

    let edges = autograd::make_edges(&input.inner, &target.inner);
    let inputs = vec![input.inner.clone(), target.inner.clone()];
    Ok(wrap_loss_with_autograd(output, &input.inner, move || {
        std::sync::Arc::new(autograd::HuberLossBackward::new(edges, inputs))
    }))
}

// Tensor manipulation ops
#[pyfunction]
fn where_(condition: &PyTensor, x: &PyTensor, y: &PyTensor) -> PyResult<PyTensor> {
    let x_inner = x.inner.clone();
    let c_inner = condition.inner.clone();
    let y_inner = y.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(x_inner.where_tensor(&c_inner, &y_inner))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("where operation failed"))
}

#[pyfunction]
fn repeat(tensor: &PyTensor, repeats: Vec<i64>) -> PyResult<PyTensor> {
    let t_inner = tensor.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(t_inner.repeat(&repeats))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("repeat operation failed"))
}

#[pyfunction]
fn expand(tensor: &PyTensor, shape: Vec<i64>) -> PyResult<PyTensor> {
    let t_inner = tensor.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(t_inner.expand(shape))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("expand operation failed"))
}

#[pyfunction]
#[pyo3(signature = (tensor, dim, start, end, step = 1))]
fn slice(tensor: &PyTensor, dim: usize, start: i64, end: i64, step: i64) -> PyResult<PyTensor> {
    let t_inner = tensor.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(t_inner.slice(dim, start, end, step))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("slice operation failed"))
}

#[pyfunction]
fn topk(tensor: &PyTensor, _k: i64, dim: i64) -> PyResult<(PyTensor, PyTensor)> {
    let t_inner = tensor.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let values = t_inner.max(dim as i32, false);
        let indices = t_inner.argmax(Some(dim as usize));
        (PyTensor::from_tensor(values), PyTensor::from_tensor(indices))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("topk operation failed"))
}

#[pyfunction]
#[pyo3(signature = (tensor, axis, indices))]
fn gather(tensor: &PyTensor, axis: i64, indices: &PyTensor) -> PyResult<PyTensor> {
    let t_inner = tensor.inner.clone();
    let i_inner = indices.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(t_inner.gather(axis, &i_inner))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("gather operation failed"))
}

#[pyfunction]
fn einsum(equation: &str, tensors: Vec<PyTensor>) -> PyResult<PyTensor> {
    let tensors_inner: Vec<core_tensor::Tensor> = tensors.into_iter().map(|p| p.inner).collect();
    let eq = equation.to_string();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(core_tensor::einsum(&eq, &tensors_inner))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("einsum operation failed"))
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

    let out_height = (in_height + 2 * padding_us - kernel_h) / stride_us + 1;
    let out_width = (in_width + 2 * padding_us - kernel_w) / stride_us + 1;

    let col_tensor = unsafe {
        crate::backend::cpu::im2col::im2col_kernel(
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

#[allow(unused_variables)]
#[pyfunction]
#[pyo3(signature = (q, k, v, scale = None, causal = None))]
fn flash_attention(
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    scale: Option<f32>,
    causal: Option<bool>,
) -> PyResult<PyTensor> {
    let q_inner = q.inner.clone();
    let k_inner = k.inner.clone();
    let v_inner = v.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(
            crate::backend::cpu::flash_attn::flash_attention(
                &q_inner, &k_inner, &v_inner, scale, causal.unwrap_or(false),
            )
        )
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("flash_attention operation failed"))
}

#[pyfunction]
fn cumsum(tensor: &PyTensor, dim: i64, exclusive: bool, reverse: bool) -> PyResult<PyTensor> {
    let t_inner = tensor.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(t_inner.cumsum(dim, exclusive, reverse))
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("cumsum operation failed"))
}

#[pyfunction]
fn erf(tensor: &PyTensor) -> PyResult<PyTensor> {
    let t_inner = tensor.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        PyTensor::from_tensor(t_inner.erf())
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("erf operation failed"))
}

#[allow(dead_code)]
#[pyfunction]
fn nonzero(tensor: &PyTensor) -> PyResult<Vec<Vec<i64>>> {
    let t_inner = tensor.inner.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        t_inner.nonzero()
    }));
    result.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("nonzero operation failed"))
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
