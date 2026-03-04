mod engine;

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
            is_leaf: !requires_grad,
        }
    }
}

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

        let grad_a = grad.clone().mul(&b.clone());
        let grad_b = grad.clone().mul(&a.clone()).sum(0, true);

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
}

pub struct ReluBackward;

impl ReluBackward {
    pub fn new() -> Self {
        ReluBackward
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
}

pub struct MatmulBackward {
    pub a: Tensor,
    pub b: Tensor,
}

impl MatmulBackward {
    pub fn new(a: Tensor, b: Tensor) -> Self {
        MatmulBackward { a, b }
    }
}

impl Node for MatmulBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();

        let grad_a = grad.matmul(&self.b.transpose(0, 1));
        let grad_b = self.a.transpose(0, 1).matmul(&grad);

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
}

pub struct ExpBackward;

impl ExpBackward {
    pub fn new() -> Self {
        ExpBackward
    }
}

impl Node for ExpBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        // Would need input for proper gradient
        vec![grad_outputs[0].clone()]
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
}

pub struct LogBackward;

impl LogBackward {
    pub fn new() -> Self {
        LogBackward
    }
}

impl Node for LogBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        // Would need input for proper gradient
        vec![grad_outputs[0].clone()]
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
}

pub struct SqrtBackward;

impl SqrtBackward {
    pub fn new() -> Self {
        SqrtBackward
    }
}

impl Node for SqrtBackward {
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
        "SqrtBackward"
    }
}

pub struct AbsBackward;

impl AbsBackward {
    pub fn new() -> Self {
        AbsBackward
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
}

pub struct GeluBackward;

impl GeluBackward {
    pub fn new() -> Self {
        GeluBackward
    }
}

impl Node for GeluBackward {
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
        "GeluBackward"
    }
}

pub struct SigmoidBackward;

impl SigmoidBackward {
    pub fn new() -> Self {
        SigmoidBackward
    }
}

impl Node for SigmoidBackward {
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
        "SigmoidBackward"
    }
}

pub struct TanhBackward;

impl TanhBackward {
    pub fn new() -> Self {
        TanhBackward
    }
}

impl Node for TanhBackward {
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
        "TanhBackward"
    }
}

pub struct SiLUBackward;

impl SiLUBackward {
    pub fn new() -> Self {
        SiLUBackward
    }
}

impl Node for SiLUBackward {
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
        "SiLUBackward"
    }
}

pub struct SoftmaxBackward;

impl SoftmaxBackward {
    pub fn new() -> Self {
        SoftmaxBackward
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
}

pub struct LogSoftmaxBackward;

impl LogSoftmaxBackward {
    pub fn new() -> Self {
        LogSoftmaxBackward
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
        vec![grad_outputs[0].clone(), grad_outputs[0].clone()]
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
}
