#![allow(dead_code)]
mod engine;
pub use engine::backward;

use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

thread_local! {
    static NO_GRAD: AtomicBool = const { AtomicBool::new(false) };
}

pub fn is_grad_enabled() -> bool {
    NO_GRAD.with(|g| !g.load(Ordering::SeqCst))
}

pub fn no_grad_enter() {
    NO_GRAD.with(|g| g.store(true, Ordering::SeqCst));
}

pub fn no_grad_exit() {
    NO_GRAD.with(|g| g.store(false, Ordering::SeqCst));
}

#[derive(Clone)]
pub struct AutogradMeta {
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
    pub grad_fn: Option<Arc<dyn Node>>,
    pub is_leaf: bool,
}

impl AutogradMeta {
    pub fn new(requires_grad: bool) -> Self {
        AutogradMeta {
            requires_grad,
            grad: None,
            grad_fn: None,
            is_leaf: true,
        }
    }

    pub fn new_non_leaf(requires_grad: bool) -> Self {
        AutogradMeta {
            requires_grad,
            grad: None,
            grad_fn: None,
            is_leaf: false,
        }
    }
}

#[derive(Clone)]
pub struct Edge(pub Arc<dyn Node>, pub usize);

pub fn make_edge(tensor: &Tensor) -> Vec<Edge> {
    tensor
        .grad_fn()
        .map(|node| Edge(node, 0))
        .map(|e| vec![e])
        .unwrap_or_else(Vec::new)
}

pub fn make_edges(tensor_a: &Tensor, tensor_b: &Tensor) -> Vec<Edge> {
    let mut edges = Vec::new();
    if let Some(node) = tensor_a.grad_fn() {
        edges.push(Edge(node, 0));
    }
    if let Some(node) = tensor_b.grad_fn() {
        edges.push(Edge(node, 1));
    }
    edges
}

#[allow(clippy::len_zero)]
pub trait Node: Send + Sync {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>>;
    fn next_edges(&self) -> &[Edge];
    fn num_inputs(&self) -> usize;
    fn name(&self) -> &str;
    fn inputs(&self) -> &[Tensor];
    fn id(&self) -> usize {
        let ptr = self as *const _ as *const ();
        ptr as usize
    }
}

#[allow(dead_code)]
pub struct AddBackward {
    pub inputs: Vec<Tensor>,
    pub edges: Vec<Edge>,
}

impl AddBackward {
    #[allow(dead_code)]
    pub fn new(inputs: Vec<Tensor>, edges: Vec<Edge>) -> Self {
        AddBackward { inputs, edges }
    }
}

impl Node for AddBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone();
        vec![grad.clone(), grad]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "AddBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

#[allow(dead_code)]
pub struct SubBackward {
    pub inputs: Vec<Tensor>,
    pub edges: Vec<Edge>,
}

impl SubBackward {
    pub fn new(inputs: Vec<Tensor>, edges: Vec<Edge>) -> Self {
        SubBackward { inputs, edges }
    }
}

impl Node for SubBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let neg_grad = grad.neg();
        vec![Some(grad), Some(neg_grad)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "SubBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

#[allow(dead_code)]
pub struct MulBackward {
    pub inputs: Vec<Tensor>,
    pub edges: Vec<Edge>,
}

impl MulBackward {
    pub fn new(inputs: Vec<Tensor>, edges: Vec<Edge>) -> Self {
        MulBackward { inputs, edges }
    }
}

impl Node for MulBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let a = &self.inputs[0];
        let b = &self.inputs[1];

        let grad_a = grad.clone().mul(b);
        let grad_b = grad.mul(a);

        vec![Some(grad_a), Some(grad_b)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "MulBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

#[allow(dead_code)]
pub struct DivBackward {
    pub inputs: Vec<Tensor>,
    pub edges: Vec<Edge>,
}

impl DivBackward {
    pub fn new(inputs: Vec<Tensor>, edges: Vec<Edge>) -> Self {
        DivBackward { inputs, edges }
    }
}

impl Node for DivBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let a = &self.inputs[0];
        let b = &self.inputs[1];

        let b_sq = b.mul(b);
        let grad_a = grad.clone().div(&b.clone());
        let grad_b = grad.mul(a).div(&b_sq).neg();

        vec![Some(grad_a), Some(grad_b)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "DivBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

#[allow(dead_code)]
pub struct NegBackward {
    pub input: Tensor,
    pub edges: Vec<Edge>,
}

impl NegBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        NegBackward { input, edges }
    }
}

impl Node for NegBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![Some(
            Tensor::from_scalar(0.0).sub(&grad_outputs[0].clone().unwrap()),
        )]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "NegBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

pub struct ReluBackward {
    pub input: Tensor,
    pub edges: Vec<Edge>,
}

impl ReluBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        ReluBackward { input, edges }
    }
}

impl Node for ReluBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let mask = self.input.gt_scalar(0.0);
        let result = grad.mul(&mask);
        vec![Some(result)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "ReluBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

pub struct MatmulBackward {
    pub inputs: Vec<Tensor>,
    pub edges: Vec<Edge>,
}

impl MatmulBackward {
    pub fn new(a: Tensor, b: Tensor, edges: Vec<Edge>) -> Self {
        MatmulBackward {
            inputs: vec![a, b],
            edges,
        }
    }
}

impl Node for MatmulBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let a = &self.inputs[0];
        let b = &self.inputs[1];

        let ndim_b = b.ndim();
        let ndim_a = a.ndim();
        let grad_a = grad.matmul(&b.transpose(ndim_b - 2, ndim_b - 1));
        let grad_b = a.transpose(ndim_a - 2, ndim_a - 1).matmul(&grad);

        vec![Some(grad_a), Some(grad_b)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "MatmulBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = match grad_outputs[0].clone() {
            Some(g) => g,
            None => {
                return vec![None];
            }
        };
        let shape = self.input.shape();
        let grad_shape = grad.shape();

        if grad_shape.is_empty() {
            let ones = Tensor::from_vec(
                vec![1.0; shape.iter().product::<i64>() as usize],
                shape.clone(),
            );
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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let scale = 1.0 / self.numel as f32;

        let shape = self.input.shape();
        let grad_shape = grad_outputs[0]
            .as_ref()
            .map(|g| g.shape())
            .unwrap_or_default();

        let result = if grad_shape.is_empty() {
            let ones = Tensor::from_vec(vec![1.0; self.numel as usize], shape.clone());
            grad.mul(&ones).mul(&Tensor::from_scalar(scale))
        } else if self.keepdim {
            let scaled_grad = grad.mul(&Tensor::from_scalar(scale));
            scaled_grad.expand(shape)
        } else {
            let mut new_shape = shape.clone();
            new_shape[self.dim] = 1;
            let reshaped = grad.reshape(new_shape);
            let scaled_grad = reshaped.mul(&Tensor::from_scalar(scale));
            scaled_grad.expand(shape)
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

#[allow(dead_code)]
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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
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

#[allow(dead_code)]
pub struct LogBackward {
    pub input: Tensor,
    pub edges: Vec<Edge>,
}

impl LogBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        LogBackward { input, edges }
    }
}

impl Node for LogBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let one = Tensor::from_scalar(1.0);
        let inv_x = one.div(&self.input);
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
}

impl SqrtBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        SqrtBackward { input, edges }
    }
}

impl Node for SqrtBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let half = Tensor::from_scalar(0.5);
        let inv_sqrt_x = half.div(&self.input.sqrt());
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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
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
}

impl GeluBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        GeluBackward { input, edges }
    }
}

impl Node for GeluBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let x = &self.input;
        let sqrt_2_over_pi = Tensor::from_scalar((2.0_f32 / std::f32::consts::PI).sqrt());
        let coeff = Tensor::from_scalar(0.044715_f32);
        let d_inner_coeff = Tensor::from_scalar(0.134145_f32);

        let x2 = x.mul(x);
        let x3 = x2.mul(x);
        let inner = sqrt_2_over_pi.mul(&x.add(&coeff.mul(&x3)));
        let t = inner.tanh();
        let t2 = t.mul(&t);
        let sech2 = Tensor::from_scalar(1.0).sub(&t2);
        let d_inner_dx = sqrt_2_over_pi.mul(&Tensor::from_scalar(1.0).add(&d_inner_coeff.mul(&x2)));
        let derivative = Tensor::from_scalar(0.5)
            .mul(&Tensor::from_scalar(1.0).add(&t))
            .add(&Tensor::from_scalar(0.5).mul(x).mul(&sech2).mul(&d_inner_dx));
        vec![Some(grad.mul(&derivative))]
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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let sigmoid_x = self.input.sigmoid();
        let derivative = sigmoid_x.mul(&Tensor::from_scalar(1.0).sub(&sigmoid_x));
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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let tanh_x = self.input.tanh();
        let derivative = Tensor::from_scalar(1.0).sub(&tanh_x.mul(&tanh_x));
        vec![Some(grad.mul(&derivative))]
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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let x = &self.input;
        let s = x.sigmoid();
        let one_minus_s = Tensor::from_scalar(1.0).sub(&s);
        let derivative = s.mul(&Tensor::from_scalar(1.0).add(&x.mul(&one_minus_s)));
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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
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

#[allow(dead_code)]
pub struct Conv2dBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub groups: i64,
    pub edges: Vec<Edge>,
}

impl Conv2dBackward {
    pub fn new(
        input: Tensor,
        weight: Tensor,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
        edges: Vec<Edge>,
    ) -> Self {
        Conv2dBackward {
            input,
            weight,
            stride,
            padding,
            dilation,
            groups,
            edges,
        }
    }
}

impl Node for Conv2dBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let grad_input = grad.clone();
        let grad_weight = grad.clone();
        vec![Some(grad_input), Some(grad_weight)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "Conv2dBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &[]
    }
}

#[allow(dead_code)]
pub struct LayerNormBackward {
    pub input: Tensor,
    pub normalized: Tensor,
    pub gamma: Option<Tensor>,
    pub mean: Tensor,
    pub variance: Tensor,
    pub eps: f32,
    pub edges: Vec<Edge>,
}

impl LayerNormBackward {
    pub fn new(
        input: Tensor,
        normalized: Tensor,
        gamma: Option<Tensor>,
        mean: Tensor,
        variance: Tensor,
        eps: f32,
        edges: Vec<Edge>,
    ) -> Self {
        LayerNormBackward {
            input,
            normalized,
            gamma,
            mean,
            variance,
            eps,
            edges,
        }
    }
}

impl Node for LayerNormBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();

        let std = self.variance.add(&Tensor::from_scalar(self.eps)).sqrt();
        let grad_x_hat = if let Some(ref g) = self.gamma {
            grad.mul(g)
        } else {
            grad.clone()
        };

        let mean_grad_x_hat = grad_x_hat
            .sum(1, true)
            .div(&Tensor::from_scalar(self.normalized.shape()[1] as f32));
        let mean_grad_x_hat_x_hat = grad_x_hat
            .mul(&self.normalized)
            .sum(1, true)
            .div(&Tensor::from_scalar(self.normalized.shape()[1] as f32));
        let grad_input = grad_x_hat
            .sub(&mean_grad_x_hat)
            .sub(&self.normalized.mul(&mean_grad_x_hat_x_hat))
            .div(&std);

        let grad_gamma = if let Some(ref g) = self.gamma {
            Some(grad.mul(&self.normalized).sum(0, false))
        } else {
            None
        };
        let grad_beta = Some(grad.sum(0, false));

        vec![
            Some(grad_input),
            grad_gamma.or_else(|| {
                Some(Tensor::zeros(
                    vec![1],
                    crate::storage::DType::F32,
                    crate::storage::Device::Cpu,
                ))
            }),
            grad_beta.or_else(|| {
                Some(Tensor::zeros(
                    vec![1],
                    crate::storage::DType::F32,
                    crate::storage::Device::Cpu,
                ))
            }),
        ]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "LayerNormBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

#[allow(dead_code)]
pub struct EmbeddingBackward {
    pub weight: Tensor,
    pub indices: Tensor,
    pub edges: Vec<Edge>,
}

impl EmbeddingBackward {
    pub fn new(weight: Tensor, indices: Tensor, edges: Vec<Edge>) -> Self {
        EmbeddingBackward {
            weight,
            indices,
            edges,
        }
    }
}

impl Node for EmbeddingBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grad_outputs[0].clone()]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "EmbeddingBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.weight)
    }
}

pub struct CrossEntropyBackward {
    pub logits: Tensor,
    pub targets: Tensor,
    pub reduction: String,
    pub edges: Vec<Edge>,
}

impl CrossEntropyBackward {
    pub fn new(logits: Tensor, targets: Tensor, reduction: String, edges: Vec<Edge>) -> Self {
        CrossEntropyBackward {
            logits,
            targets,
            reduction,
            edges,
        }
    }
}

impl Node for CrossEntropyBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let _grad = grad_outputs[0].clone().unwrap();

        let logits = &self.logits;
        let targets = &self.targets;
        let batch_size = logits.shape()[0] as usize;
        let num_classes = logits.shape()[1] as usize;

        let logits_data = logits.as_f32_slice();
        let targets_data = targets.as_f32_slice();

        let mut grad_logits_data = vec![0.0f32; batch_size * num_classes];

        for (b, target_val) in targets_data.iter().take(batch_size).enumerate() {
            let target_class = *target_val as usize;
            let base_idx = b * num_classes;

            let mut max_logit = f32::MIN;
            for c in 0..num_classes {
                let val = logits_data[base_idx + c];
                if val > max_logit {
                    max_logit = val;
                }
            }

            let mut sum_exp = 0.0f32;
            for c in 0..num_classes {
                sum_exp += (logits_data[base_idx + c] - max_logit).exp();
            }

            let target_prob = (logits_data[base_idx + target_class] - max_logit).exp() / sum_exp;

            grad_logits_data[base_idx + target_class] = target_prob - 1.0;

            for c in 0..num_classes {
                if c != target_class {
                    let prob = (logits_data[base_idx + c] - max_logit).exp() / sum_exp;
                    grad_logits_data[base_idx + c] = prob;
                }
            }
        }

        let scale = if self.reduction == "mean" {
            1.0 / batch_size as f32
        } else {
            1.0
        };

        for v in &mut grad_logits_data {
            *v *= scale;
        }

        let grad_logits = Tensor::from_vec(
            grad_logits_data,
            vec![batch_size as i64, num_classes as i64],
        );

        vec![Some(grad_logits)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "CrossEntropyBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.logits)
    }
}
