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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = match grad_outputs.into_iter().next().flatten() {
            Some(g) => g,
            None => {
                return vec![None];
            }
        };
        let shape = self.input.shape();
        let grad_shape = grad.shape();

        if grad_shape.is_empty() {
            // Scalar gradient: multiply by ones to broadcast to input shape
            // (expand creates non-contiguous views that many kernels can't handle)
            let ones = Tensor::ones(shape.clone(), grad.dtype(), grad.device());
            vec![Some(grad.mul(&ones))]
        } else if self.keepdim {
            vec![Some(grad.expand(shape))]
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
        let scale = 1.0 / self.numel as f32;

        let shape = self.input.shape();
        let grad_shape = grad.shape();

        let result = if grad_shape.is_empty() {
            // Scalar gradient: create ones and multiply
            let ones = Tensor::ones(shape.clone(), grad.dtype(), grad.device());
            let mut scaled = grad;
            scaled.mul_scalar_(scale);
            scaled.mul(&ones)
        } else if self.keepdim {
            let mut scaled = grad;
            scaled.mul_scalar_(scale);
            scaled.expand(shape)
        } else {
            let mut new_shape = shape.clone();
            new_shape[self.dim] = 1;
            let reshaped = grad.reshape(new_shape);
            let mut scaled = reshaped;
            scaled.mul_scalar_(scale);
            scaled.expand(shape)
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
}

impl ExpBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        ExpBackward { input, edges }
    }
}

impl Node for ExpBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
        let x = &self.input;

        // Use fused kernel to eliminate ~9 intermediate tensor allocations
        let result = crate::dispatcher::dispatch(
            "gelu_backward",
            crate::dispatcher::DispatchKey::Cpu,
            &[x, &grad],
        );
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
    pub edges: Vec<Edge>,
}

impl SigmoidBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        SigmoidBackward { input, edges }
    }
}

impl Node for SigmoidBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
        let sigmoid_x = self.input.sigmoid();
        let derivative = sigmoid_x.mul_scalar(1.0).sub(&sigmoid_x.pow(2.0));
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
    pub edges: Vec<Edge>,
}

impl TanhBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        TanhBackward { input, edges }
    }
}

impl Node for TanhBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
        let tanh_x = self.input.tanh();
        let tanh_sq = tanh_x.pow(2.0);
        let one_minus_tanh_sq = tanh_sq.mul_scalar(-1.0).add_scalar(1.0);
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
        let s = self.input.sigmoid();
        let one_minus_s = s.mul_scalar(1.0).sub(&s);
        let derivative = s.mul(&self.input.mul(&one_minus_s).add_scalar(1.0));
        vec![Some(grad.mul(&derivative))]
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

#[allow(dead_code)]
pub struct SoftmaxBackward {
    pub output: Tensor,
    pub dim: usize,
    pub edges: Vec<Edge>,
}

impl SoftmaxBackward {
    pub fn new(output: Tensor, dim: usize, edges: Vec<Edge>) -> Self {
        SoftmaxBackward { output, dim, edges }
    }
}

impl Node for SoftmaxBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
        let s = &self.output;
        let dim_i32 = self.dim as i32;
        let dot = grad.mul(s).sum(dim_i32, true);
        let grad_input = s.mul(&grad.sub(&dot));
        vec![Some(grad_input)]
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
        std::slice::from_ref(&self.output)
    }
}

#[allow(dead_code)]
pub struct LogSoftmaxBackward {
    pub output: Tensor,
    pub dim: usize,
    pub edges: Vec<Edge>,
}

impl LogSoftmaxBackward {
    pub fn new(output: Tensor, dim: usize, edges: Vec<Edge>) -> Self {
        LogSoftmaxBackward { output, dim, edges }
    }
}

impl Node for LogSoftmaxBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
        let softmax = self.output.exp();
        let dim_i32 = self.dim as i32;
        let grad_sum = grad.sum(dim_i32, true);
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

