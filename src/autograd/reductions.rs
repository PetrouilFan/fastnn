pub struct SumBackward {
    pub input: Tensor,
    pub dim: usize,
    pub keepdim: bool,
    pub edges: Vec<Edge>,
}

impl SumBackward {
    pub fn new(input: Tensor, dim: usize, keepdim: bool, edges: Vec<Edge>) -> Self {
        SumBackward {
            input,
            dim,
            keepdim,
            edges,
        }
    }
}

impl Node for SumBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let shape = self.input.shape_ref();
        let grad_shape = grad.shape_ref();

        if grad_shape.is_empty() {
            vec![Some(grad.expand(shape.to_vec()))]
        } else if self.keepdim {
            vec![Some(grad.expand(shape.to_vec()))]
        } else {
            let mut expanded_shape: Vec<i64> = grad_shape.to_vec();
            expanded_shape.insert(self.dim, 1);
            vec![Some(grad.reshape(expanded_shape))]
        }
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "SumBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

pub struct MeanBackward {
    pub input: Tensor,
    pub dim: usize,
    pub keepdim: bool,
    pub numel: i64,
    pub edges: Vec<Edge>,
}

impl MeanBackward {
    pub fn new(input: Tensor, dim: usize, keepdim: bool, numel: i64, edges: Vec<Edge>) -> Self {
        MeanBackward {
            input,
            dim,
            keepdim,
            numel,
            edges,
        }
    }
}

impl Node for MeanBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let scale = 1.0 / self.numel as f32;

        let shape = self.input.shape_ref();
        let grad_shape = grad.shape_ref();

        let result = if grad_shape.is_empty() {
            // Scalar gradient: just multiply scalar and expand
            let scaled = grad.mul_scalar(scale);
            scaled.expand(shape.to_vec())
        } else if self.keepdim {
            let scaled = grad.mul_scalar(scale);
            scaled.expand(shape.to_vec())
        } else {
            let mut new_shape = shape.to_vec();
            new_shape[self.dim] = 1;
            let reshaped = grad.reshape(new_shape.to_vec());
            let scaled = reshaped.mul_scalar(scale);
            scaled.expand(shape.to_vec())
        };

        vec![Some(result)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "MeanBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}
pub struct ExpBackward {
    pub input: Tensor,
    pub edges: Vec<Edge>,
    one: Tensor,
}

impl ExpBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        ExpBackward {
            input,
            edges,
            one: Tensor::from_scalar(1.0),
        }
    }
}

impl Node for ExpBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let exp_x = self.input.exp();
        vec![Some(grad.mul(&exp_x))]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "ExpBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}
pub struct LogBackward {
    pub input: Tensor,
    pub edges: Vec<Edge>,
    one: Tensor,
}

impl LogBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        LogBackward {
            input,
            edges,
            one: Tensor::from_scalar(1.0),
        }
    }
}

impl Node for LogBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        // Clamp input to prevent ±inf gradients on dead ReLU -> log chains
        // Use a small epsilon to avoid division by zero
        let eps = Tensor::from_scalar(1e-8);
        let input_clamped = self.input.add(&eps);
        let inv_x = self.one.div(&input_clamped);
        vec![Some(grad.mul(&inv_x))]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "LogBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

#[allow(dead_code)]
pub struct SqrtBackward {
    pub input: Tensor,
    pub edges: Vec<Edge>,
    half: Tensor,
}

impl SqrtBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        SqrtBackward {
            input,
            edges,
            half: Tensor::from_scalar(0.5),
        }
    }
}

impl Node for SqrtBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let inv_sqrt_x = self.half.div(&self.input.sqrt());
        vec![Some(grad.mul(&inv_sqrt_x))]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "SqrtBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

#[allow(dead_code)]
pub struct AbsBackward {
    pub input: Tensor,
    pub edges: Vec<Edge>,
}

impl AbsBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        AbsBackward { input, edges }
    }
}

impl Node for AbsBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let sign = self.input.sign();
        vec![Some(grad.mul(&sign))]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "AbsBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

pub struct GeluBackward {
    pub input: Tensor,
    pub edges: Vec<Edge>,
    // Pre-allocated scalar constants to avoid per-call allocation
    sqrt_2_over_pi: Tensor,
    coeff: Tensor,
    d_inner_coeff: Tensor,
    half: Tensor,
    one: Tensor,
}

impl GeluBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        GeluBackward {
            input,
            edges,
            sqrt_2_over_pi: Tensor::from_scalar((2.0_f32 / std::f32::consts::PI).sqrt()),
            coeff: Tensor::from_scalar(0.044715_f32),
            d_inner_coeff: Tensor::from_scalar(0.134145_f32),
            half: Tensor::from_scalar(0.5),
            one: Tensor::from_scalar(1.0),
        }
    }
}

impl Node for GeluBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let x = &self.input;

        let dispatch_key = crate::dispatcher::device_to_dispatch_key(x.device());
        let result = crate::dispatcher::dispatch(
            "gelu_backward",
            dispatch_key,
            &[&grad, x],
        )
        .expect("GeluBackward::apply: dispatch failed");
        vec![Some(result[0].clone())]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "GeluBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

pub struct SigmoidBackward {
    pub input: Tensor,
    pub sigmoid_output: Tensor,
    pub edges: Vec<Edge>,
    one: Tensor,
}

impl SigmoidBackward {
    pub fn new(input: Tensor, sigmoid_output: Tensor, edges: Vec<Edge>) -> Self {
        SigmoidBackward {
            input,
            sigmoid_output,
            edges,
            one: Tensor::from_scalar(1.0),
        }
    }
}

impl Node for SigmoidBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        // derivative = sigmoid(x) * (1 - sigmoid(x))
        let derivative = self.sigmoid_output.mul(
            &self.sigmoid_output.neg().add(&self.one)
        );
        vec![Some(grad.mul(&derivative))]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "SigmoidBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

pub struct TanhBackward {
    pub input: Tensor,
    pub tanh_output: Tensor,
    pub edges: Vec<Edge>,
    one: Tensor,
}

impl TanhBackward {
    pub fn new(input: Tensor, tanh_output: Tensor, edges: Vec<Edge>) -> Self {
        TanhBackward {
            input,
            tanh_output,
            edges,
            one: Tensor::from_scalar(1.0),
        }
    }
}

impl Node for TanhBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        // derivative = 1 - tanh(x)^2
        let tanh_sq = self.tanh_output.pow(2.0);
        let one_minus_tanh_sq = self.one.sub(&tanh_sq);
        vec![Some(grad.mul(&one_minus_tanh_sq))]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "TanhBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

pub struct SiLUBackward {
    pub input: Tensor,
    pub edges: Vec<Edge>,
}

impl SiLUBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        SiLUBackward { input, edges }
    }
}

impl Node for SiLUBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let s = self.input.sigmoid();

        let dispatch_key = crate::dispatcher::device_to_dispatch_key(self.input.device());
        let result = crate::dispatcher::dispatch(
            "silu_backward",
            dispatch_key,
            &[&self.input, &s, &grad],
        )
        .expect("SiLUBackward::apply: dispatch failed");
        vec![result.first().cloned()]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "SiLUBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

pub struct SoftmaxBackward {
    pub input: Tensor,
    pub dim: usize,
    pub edges: Vec<Edge>,
}

impl SoftmaxBackward {
    pub fn new(input: Tensor, dim: usize, edges: Vec<Edge>) -> Self {
        SoftmaxBackward { input, dim, edges }
    }
}

impl Node for SoftmaxBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        // Recompute softmax from stored input to avoid storing output (which creates a ref cycle)
        let x = &self.input;
        let max_val = x.max(-1, true);
        let shifted = x.sub(&max_val);
        let exp_vals = shifted.exp();
        let sum_exp = exp_vals.sum(-1, true);
        let s = exp_vals.div(&sum_exp);
        let dim_tensor = Tensor::from_scalar(self.dim as f32);
        let dispatch_key = crate::dispatcher::device_to_dispatch_key(x.device());
        let result = crate::dispatcher::dispatch(
            "softmax_backward",
            dispatch_key,
            &[&s, &grad, &dim_tensor],
        )
        .expect("SoftmaxBackward::apply: dispatch failed");
        vec![result.first().cloned()]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "SoftmaxBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

#[allow(dead_code)]
pub struct LogSoftmaxBackward {
    pub output: Tensor,
    pub dim: i32,
    pub edges: Vec<Edge>,
}

impl LogSoftmaxBackward {
    pub fn new(output: Tensor, dim: i32, edges: Vec<Edge>) -> Self {
        LogSoftmaxBackward { output, dim, edges }
    }
}

impl Node for LogSoftmaxBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let softmax = self.output.exp();
        let grad_sum = grad.sum(self.dim, true);
        let grad_input = grad.sub(&softmax.mul(&grad_sum));
        vec![Some(grad_input)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "LogSoftmaxBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.output)
    }
}

