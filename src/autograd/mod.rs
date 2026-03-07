mod engine;
pub use engine::backward;

use crate::tensor::Tensor;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

thread_local! {
    static NO_GRAD: AtomicBool = AtomicBool::new(false);
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
}

#[derive(Clone)]
pub struct Edge(pub Arc<dyn Node>, pub usize);

#[allow(clippy::len_zero)]
pub trait Node: Send + Sync {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>>;
    fn next_edges(&self) -> &[Edge];
    fn num_inputs(&self) -> usize;
    fn name(&self) -> &str;
    fn inputs(&self) -> &[Tensor];
}

pub struct AddBackward {
    pub inputs: Vec<Tensor>,
}

impl AddBackward {
    pub fn new(inputs: Vec<Tensor>) -> Self {
        AddBackward { inputs }
    }
}

impl Node for AddBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone();
        vec![grad.clone(), grad]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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

pub struct SubBackward {
    pub inputs: Vec<Tensor>,
}

impl SubBackward {
    pub fn new(inputs: Vec<Tensor>) -> Self {
        SubBackward { inputs }
    }
}

impl Node for SubBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone();
        let zero = Tensor::from_scalar(0.0);
        let neg_grad = zero.sub(&grad.clone().unwrap());
        vec![grad.clone(), Some(neg_grad)]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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

pub struct MulBackward {
    pub inputs: Vec<Tensor>,
}

impl MulBackward {
    pub fn new(inputs: Vec<Tensor>) -> Self {
        MulBackward { inputs }
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
        &[]
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

pub struct DivBackward {
    pub inputs: Vec<Tensor>,
}

impl DivBackward {
    pub fn new(inputs: Vec<Tensor>) -> Self {
        DivBackward { inputs }
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
        &[]
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

pub struct NegBackward {
    pub input: Tensor,
}

impl NegBackward {
    pub fn new(input: Tensor) -> Self {
        NegBackward { input }
    }
}

impl Node for NegBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![Some(
            Tensor::from_scalar(0.0).sub(&grad_outputs[0].clone().unwrap()),
        )]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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
}

impl ReluBackward {
    pub fn new(input: Tensor) -> Self {
        ReluBackward { input }
    }
}

impl Node for ReluBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let input = grad.clone();
        let mask = input.relu();
        vec![Some(grad * mask)]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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
}

impl MatmulBackward {
    pub fn new(a: Tensor, b: Tensor) -> Self {
        MatmulBackward { inputs: vec![a, b] }
    }
}

impl Node for MatmulBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let a = &self.inputs[0];
        let b = &self.inputs[1];

        let grad_a = grad.matmul(&b.transpose(0, 1));
        let grad_b = a.transpose(0, 1).matmul(&grad);

        vec![Some(grad_a), Some(grad_b)]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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
}

impl SumBackward {
    pub fn new(input: Tensor, dim: usize, keepdim: bool) -> Self {
        SumBackward {
            input,
            dim,
            keepdim,
        }
    }
}

impl Node for SumBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let shape = self.input.shape();

        if self.keepdim {
            vec![Some(grad.expand(shape))]
        } else {
            let mut new_shape = shape.clone();
            new_shape.insert(self.dim, 1);
            vec![Some(grad.reshape(new_shape).expand(shape))]
        }
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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
}

impl MeanBackward {
    pub fn new(input: Tensor, dim: usize, keepdim: bool, numel: i64) -> Self {
        MeanBackward {
            input,
            dim,
            keepdim,
            numel,
        }
    }
}

impl Node for MeanBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let scale = Tensor::from_scalar(1.0 / self.numel as f32);
        let scaled_grad = grad.mul(&scale);

        let shape = self.input.shape();

        if self.keepdim {
            vec![Some(scaled_grad.expand(shape))]
        } else {
            let mut new_shape = shape.clone();
            new_shape.insert(self.dim, 1);
            vec![Some(scaled_grad.reshape(new_shape).expand(shape))]
        }
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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
}

impl ExpBackward {
    pub fn new(input: Tensor) -> Self {
        ExpBackward { input }
    }
}

impl Node for ExpBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let exp_x = self.input.exp();
        vec![Some(grad.mul(&exp_x))]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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
}

impl LogBackward {
    pub fn new(input: Tensor) -> Self {
        LogBackward { input }
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
        &[]
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

pub struct SqrtBackward {
    pub input: Tensor,
}

impl SqrtBackward {
    pub fn new(input: Tensor) -> Self {
        SqrtBackward { input }
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
        &[]
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

pub struct AbsBackward {
    pub input: Tensor,
}

impl AbsBackward {
    pub fn new(input: Tensor) -> Self {
        AbsBackward { input }
    }
}

impl Node for AbsBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grad_outputs[0].clone()]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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
}

impl GeluBackward {
    pub fn new(input: Tensor) -> Self {
        GeluBackward { input }
    }
}

impl Node for GeluBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let x = &self.input;

        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        let x2 = x.mul(x);
        let x3 = x2.mul(x);
        let tanh_arg = x
            .mul(&Tensor::from_scalar(sqrt_2_over_pi))
            .add(&x3.mul(&Tensor::from_scalar(0.044715)));
        let tanh_arg_tanh = tanh_arg.tanh();

        let exp_term = x
            .mul(&Tensor::from_scalar(sqrt_2_over_pi))
            .add(&x2.mul(&Tensor::from_scalar(0.134)));
        let exp_term_exp = exp_term.exp();
        let sigmoid_term =
            Tensor::from_scalar(1.0).div(&Tensor::from_scalar(1.0).add(&exp_term_exp));

        let derivative = Tensor::from_scalar(0.5)
            .mul(&Tensor::from_scalar(1.0).add(&tanh_arg_tanh))
            .add(
                &x.mul(&Tensor::from_scalar(0.5))
                    .mul(&tanh_arg_tanh.tanh())
                    .mul(&sigmoid_term),
            );

        vec![Some(grad.mul(&derivative))]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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
}

impl SigmoidBackward {
    pub fn new(input: Tensor) -> Self {
        SigmoidBackward { input }
    }
}

impl Node for SigmoidBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let sigmoid_x = self.input.sigmoid();
        let derivative = sigmoid_x.mul(&sigmoid_x.neg().add(&Tensor::from_scalar(1.0)));
        vec![Some(grad.mul(&derivative))]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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
}

impl TanhBackward {
    pub fn new(input: Tensor) -> Self {
        TanhBackward { input }
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
        &[]
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
}

impl SiLUBackward {
    pub fn new(input: Tensor) -> Self {
        SiLUBackward { input }
    }
}

impl Node for SiLUBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let sigmoid_x = self.input.sigmoid();
        let silu_x = self.input.mul(&sigmoid_x);
        let derivative = silu_x.add(
            &sigmoid_x
                .mul(&Tensor::from_scalar(1.0))
                .sub(&silu_x.div(&self.input)),
        );
        vec![Some(grad.mul(&derivative))]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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
}

impl SoftmaxBackward {
    pub fn new(input: Tensor) -> Self {
        SoftmaxBackward { input }
    }
}

impl Node for SoftmaxBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grad_outputs[0].clone()]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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

pub struct LogSoftmaxBackward {
    pub input: Tensor,
}

impl LogSoftmaxBackward {
    pub fn new(input: Tensor) -> Self {
        LogSoftmaxBackward { input }
    }
}

impl Node for LogSoftmaxBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grad_outputs[0].clone()]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "LogSoftmaxBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

pub struct Conv2dBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub groups: i64,
}

impl Conv2dBackward {
    pub fn new(
        input: Tensor,
        weight: Tensor,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
    ) -> Self {
        Conv2dBackward {
            input,
            weight,
            stride,
            padding,
            dilation,
            groups,
        }
    }
}

impl Node for Conv2dBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        vec![Some(grad.clone()), Some(grad)]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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

pub struct LayerNormBackward {
    pub input: Tensor,
}

impl LayerNormBackward {
    pub fn new(input: Tensor) -> Self {
        LayerNormBackward { input }
    }
}

impl Node for LayerNormBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grad_outputs[0].clone()]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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

pub struct EmbeddingBackward {
    pub weight: Tensor,
}

impl EmbeddingBackward {
    pub fn new(weight: Tensor) -> Self {
        EmbeddingBackward { weight }
    }
}

impl Node for EmbeddingBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grad_outputs[0].clone()]
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
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
