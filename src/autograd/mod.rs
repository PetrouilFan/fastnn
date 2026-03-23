#![allow(dead_code)]
mod engine;
pub use engine::backward;

use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

static NO_GRAD_GLOBAL: AtomicBool = AtomicBool::new(false);

pub fn is_grad_enabled() -> bool {
    !NO_GRAD_GLOBAL.load(Ordering::Relaxed)
}

pub fn no_grad_enter() {
    NO_GRAD_GLOBAL.store(true, Ordering::Release);
}

pub fn no_grad_exit() {
    NO_GRAD_GLOBAL.store(false, Ordering::Release);
}

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
        .unwrap_or_default()
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

/// Fused Linear backward operation for better performance
/// Computes gradients for weight, bias, and input in a single backward operation
pub struct LinearBackward {
    pub inputs: Vec<Tensor>,
    pub edges: Vec<Edge>,
}

impl LinearBackward {
    pub fn new(input: Tensor, weight: Tensor, bias: Option<Tensor>, edges: Vec<Edge>) -> Self {
        let mut inputs = vec![input, weight];
        if let Some(b) = bias {
            inputs.push(b);
        }
        LinearBackward { inputs, edges }
    }
}

impl Node for LinearBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad_output = grad_outputs[0].clone().unwrap();

        let input = &self.inputs[0];
        let weight = &self.inputs[1];
        let has_bias = self.inputs.len() == 3;

        // Compute gradients with potential kernel fusion
        // grad_input = grad_output @ weight^T
        let grad_input = grad_output.matmul(&weight.transpose(0, 1));

        // grad_weight = input^T @ grad_output
        let grad_weight = input.transpose(0, 1).matmul(&grad_output);

        // grad_bias = sum(grad_output, dim=0)
        let grad_bias = if has_bias {
            Some(grad_output.sum(0, false))
        } else {
            None
        };

        vec![Some(grad_input), Some(grad_weight), grad_bias]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    fn name(&self) -> &str {
        "LinearBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
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
        let grad = grad_outputs[0].clone().unwrap();
        let a = &self.inputs[0];
        let b = &self.inputs[1];

        // Handle broadcasting: if input shape doesn't match output shape,
        // sum over the extra dimensions. Avoid double-clone when both match.
        let a_matches = a.shape() == grad.shape();
        let b_matches = b.shape() == grad.shape();

        let grad_a = if a_matches {
            grad.clone()
        } else {
            let mut grad_a = grad.clone();
            let diff = grad.shape().len() as i32 - a.shape().len() as i32;
            for i in (0..grad.shape().len()).rev() {
                let a_dim = if i as i32 >= diff {
                    a.shape()[(i as i32 - diff) as usize]
                } else {
                    1
                };
                if a_dim != grad.shape()[i] {
                    grad_a = grad_a.sum(i as i32, false);
                }
            }
            grad_a
        };

        let grad_b = if b_matches {
            if a_matches {
                // Both match: reuse grad instead of cloning again
                grad
            } else {
                grad.clone()
            }
        } else {
            let mut grad_b = grad;
            let diff = grad_b.shape().len() as i32 - b.shape().len() as i32;
            for i in (0..grad_b.shape().len()).rev() {
                let b_dim = if i as i32 >= diff {
                    b.shape()[(i as i32 - diff) as usize]
                } else {
                    1
                };
                if b_dim != grad_b.shape()[i] {
                    grad_b = grad_b.sum(i as i32, false);
                }
            }
            grad_b
        };

        vec![Some(grad_a), Some(grad_b)]
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

pub struct UnsqueezeBackward {
    pub inputs: Vec<Tensor>,
    pub dim: usize,
    pub next_edges: Vec<Edge>,
}

impl UnsqueezeBackward {
    pub fn new(input: Tensor, dim: usize) -> Self {
        let mut next_edges = Vec::new();
        if let Some(grad_fn) = input.grad_fn() {
            next_edges.push(Edge(grad_fn, 0));
        }
        UnsqueezeBackward {
            inputs: vec![input],
            dim,
            next_edges,
        }
    }
}

impl Node for UnsqueezeBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let grad_squeezed = grad.squeeze(Some(self.dim));
        vec![Some(grad_squeezed)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "UnsqueezeBackward"
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

        // Compute gradient for a: grad * b
        let mut grad_a = grad.mul(b);
        // Sum over dimensions that were broadcasted in a
        if a.shape() != grad_a.shape() {
            let diff = grad_a.shape().len() as i32 - a.shape().len() as i32;
            for i in (0..grad_a.shape().len()).rev() {
                let a_dim = if i as i32 >= diff {
                    a.shape()[(i as i32 - diff) as usize]
                } else {
                    1
                };
                if a_dim != grad_a.shape()[i] {
                    grad_a = grad_a.sum(i as i32, false);
                }
            }
        }

        // Compute gradient for b: grad * a
        let mut grad_b = grad.mul(a);
        // Sum over dimensions that were broadcasted in b
        if b.shape() != grad_b.shape() {
            let diff = grad_b.shape().len() as i32 - b.shape().len() as i32;
            for i in (0..grad_b.shape().len()).rev() {
                let b_dim = if i as i32 >= diff {
                    b.shape()[(i as i32 - diff) as usize]
                } else {
                    1
                };
                if b_dim != grad_b.shape()[i] {
                    grad_b = grad_b.sum(i as i32, false);
                }
            }
        }

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

        // d/da (a/b) = 1/b, d/db (a/b) = -a/b^2
        let grad_a = grad.div(b);
        let b_sq = b.mul(b);
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
        vec![Some(grad_outputs[0].clone().unwrap().neg())]
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

        // If grad is a scalar (e.g., from sum()), we need to expand it to the output shape
        let grad_shape = grad.shape();
        let grad = if grad_shape.is_empty() {
            // Expand scalar gradient to match the output shape [a.shape[0], b.shape[1]]
            let output_shape = vec![a.shape()[0], b.shape()[1]];
            grad.expand(output_shape)
        } else {
            grad
        };

        let ndim_b = b.ndim();
        let ndim_a = a.ndim();
        let grad_a = grad.matmul(&b.transpose(ndim_b - 2, ndim_b - 1));
        let mut grad_b = a.transpose(ndim_a - 2, ndim_a - 1).matmul(&grad);

        // Handle broadcasting: if b has fewer dimensions than grad_b,
        // sum over the batch dimensions
        if b.ndim() < grad_b.ndim() {
            let diff = grad_b.ndim() as i32 - b.ndim() as i32;
            for _ in 0..diff as usize {
                grad_b = grad_b.sum(0, false);
            }
        }

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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let scale = 1.0 / self.numel as f32;

        let shape = self.input.shape();
        let grad_shape = grad_outputs[0]
            .as_ref()
            .map(|g| g.shape())
            .unwrap_or_default();

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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let inv_x = self.one.div(&self.input);
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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
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
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let x = &self.input;

        let x2 = x.mul(x);
        let x3 = x2.mul(x);
        let inner = self.sqrt_2_over_pi.mul(&x.add(&self.coeff.mul(&x3)));
        let t = inner.tanh();
        let t2 = t.mul(&t);
        let sech2 = self.one.sub(&t2);
        let d_inner_dx = self
            .sqrt_2_over_pi
            .mul(&self.one.add(&self.d_inner_coeff.mul(&x2)));
        let derivative = self
            .half
            .mul(&self.one.add(&t))
            .add(&self.half.mul(x).mul(&sech2).mul(&d_inner_dx));
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
        // derivative = sigmoid * (1 - sigmoid)
        // Avoid allocating scalar tensor for 1.0: compute directly
        let mut derivative = sigmoid_x.clone();
        // Ensure exclusive ownership before in-place modification
        let inner = Arc::make_mut(&mut derivative.inner);
        let storage = Arc::make_mut(&mut inner.storage);
        let crate::storage::Storage::Cpu(cpu_storage) = storage else {
            panic!("Expected CPU storage");
        };
        let data = Arc::make_mut(&mut cpu_storage.data);
        let ptr = data.as_mut_ptr() as *mut f32;
        let sigmoid_ptr = sigmoid_x.data_ptr_f32();
        let numel = sigmoid_x.inner.numel() as usize;
        for i in 0..numel {
            unsafe {
                let s = *sigmoid_ptr.add(i);
                *ptr.add(i) = s * (1.0 - s);
            }
        }
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
        // derivative = 1 - tanh^2
        // Avoid allocating scalar tensor for 1.0: compute directly
        let mut derivative = tanh_x.clone();
        // Ensure exclusive ownership before in-place modification
        let inner = Arc::make_mut(&mut derivative.inner);
        let storage = Arc::make_mut(&mut inner.storage);
        let crate::storage::Storage::Cpu(cpu_storage) = storage else {
            panic!("Expected CPU storage");
        };
        let data = Arc::make_mut(&mut cpu_storage.data);
        let ptr = data.as_mut_ptr() as *mut f32;
        let tanh_ptr = tanh_x.data_ptr_f32();
        let numel = tanh_x.inner.numel() as usize;
        for i in 0..numel {
            unsafe {
                let t = *tanh_ptr.add(i);
                *ptr.add(i) = 1.0 - t * t;
            }
        }
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
        // SiLU derivative = s * (1 + x * (1 - s))
        // Avoid scalar tensor allocations: compute in single pass on owned copy
        let mut derivative = s.clone();
        let inner = Arc::make_mut(&mut derivative.inner);
        let storage = Arc::make_mut(&mut inner.storage);
        let crate::storage::Storage::Cpu(cpu_storage) = storage else {
            panic!("Expected CPU storage");
        };
        let data = Arc::make_mut(&mut cpu_storage.data);
        let ptr = data.as_mut_ptr() as *mut f32;
        let numel = s.inner.numel() as usize;
        let x_ptr = x.data_ptr_f32();
        let s_ptr = s.data_ptr_f32();
        for i in 0..numel {
            unsafe {
                let si = *s_ptr.add(i);
                *ptr.add(i) = si * (1.0 + *x_ptr.add(i) * (1.0 - si));
            }
        }
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
    pub inputs: Vec<Tensor>,
    pub normalized: Tensor,
    pub mean: Tensor,
    pub variance: Tensor,
    pub eps: f32,
    pub edges: Vec<Edge>,
}

impl LayerNormBackward {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        normalized: Tensor,
        mean: Tensor,
        variance: Tensor,
        eps: f32,
        edges: Vec<Edge>,
    ) -> Self {
        LayerNormBackward {
            inputs: vec![input, weight, bias],
            normalized,
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
        let input = &self.inputs[0];
        let weight = &self.inputs[1];
        let _bias = &self.inputs[2];

        let shape = input.shape();
        let outer_size: usize = shape[..shape.len() - 1]
            .iter()
            .map(|&d| d as usize)
            .product();
        let norm_dim: usize = shape[shape.len() - 1] as usize;
        let total = outer_size * norm_dim;

        let grad_cpu = grad.to_cpu();
        let x_hat_cpu = self.normalized.to_cpu();
        let var_cpu = self.variance.to_cpu();

        let grad_data = grad_cpu.as_f32_slice();
        let x_hat_data = x_hat_cpu.as_f32_slice();
        let var_data = var_cpu.as_f32_slice();

        let weight_data = weight.as_f32_slice();

        let mut grad_input_data = vec![0.0f32; total];
        let mut grad_weight_data = vec![0.0f32; norm_dim];
        let mut grad_bias_data = vec![0.0f32; norm_dim];

        crate::kernels::cpu::layer_norm_backward_f32(
            grad_data,
            x_hat_data,
            Some(weight_data),
            outer_size,
            norm_dim,
            self.eps,
            var_data,
            &mut grad_input_data,
            &mut grad_weight_data,
            &mut grad_bias_data,
        );

        let grad_input = Tensor::from_vec(grad_input_data, shape.clone());
        let grad_weight = Tensor::from_vec(grad_weight_data, vec![norm_dim as i64]);
        let grad_bias = Tensor::from_vec(grad_bias_data, vec![norm_dim as i64]);

        vec![Some(grad_input), Some(grad_weight), Some(grad_bias)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        3
    }

    fn name(&self) -> &str {
        "LayerNormBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
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
        let grad_output_tensor = grad_outputs[0].clone().unwrap();
        let grad_out = grad_output_tensor.item();

        let logits = &self.logits;
        let targets = &self.targets;
        let batch_size = logits.shape()[0] as usize;
        let num_classes = logits.shape()[1] as usize;

        let logits_cpu = logits.to_cpu();
        let targets_cpu = targets.to_cpu();

        let logits_data = logits_cpu.as_f32_slice();
        let targets_data = targets_cpu.as_f32_slice();

        let mut grad_logits_data = vec![0.0f32; batch_size * num_classes];

        crate::kernels::cpu::cross_entropy_backward_f32(
            logits_data,
            targets_data,
            grad_out,
            batch_size,
            num_classes,
            &self.reduction,
            &mut grad_logits_data,
        );

        let grad_logits = Tensor::from_vec(
            grad_logits_data,
            vec![batch_size as i64, num_classes as i64],
        );

        let grad_logits = match logits.device() {
            crate::storage::Device::Wgpu(device_id) => grad_logits.to_gpu(device_id),
            _ => grad_logits,
        };

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

#[allow(dead_code)]
pub struct MSELossBackward {
    pub pred: Tensor,
    pub target: Tensor,
    pub reduction: String,
    pub edges: Vec<Edge>,
    two_scalar: Tensor,
}

impl MSELossBackward {
    pub fn new(pred: Tensor, target: Tensor, reduction: String, edges: Vec<Edge>) -> Self {
        MSELossBackward {
            pred,
            target,
            reduction,
            edges,
            two_scalar: Tensor::from_scalar(2.0),
        }
    }
}

impl Node for MSELossBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let diff = self.pred.sub(&self.target);

        let grad_loss = match self.reduction.as_str() {
            "mean" => {
                let n = diff.numel() as f32;
                let mut g = diff.mul(&self.two_scalar);
                g.mul_scalar_(1.0 / n);
                g
            }
            "sum" => diff.mul(&self.two_scalar),
            _ => diff.mul(&self.two_scalar),
        };

        vec![Some(grad.mul(&grad_loss))]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "MSELossBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.pred)
    }
}

#[allow(dead_code)]
pub struct ViewBackward {
    pub input: Tensor,
    pub edges: Vec<Edge>,
}

impl ViewBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        ViewBackward { input, edges }
    }
}

impl Node for ViewBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();
        let shape = self.input.shape();
        vec![Some(grad.reshape(shape))]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "ViewBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

#[allow(dead_code)]
pub struct SliceBackward {
    pub input: Tensor,
    pub dim: usize,
    pub start: i64,
    pub end: i64,
    pub step: i64,
    pub edges: Vec<Edge>,
}

impl SliceBackward {
    pub fn new(
        input: Tensor,
        dim: usize,
        start: i64,
        end: i64,
        step: i64,
        edges: Vec<Edge>,
    ) -> Self {
        SliceBackward {
            input,
            dim,
            start,
            end,
            step,
            edges,
        }
    }
}

impl Node for SliceBackward {
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let grad = grad_outputs[0].clone().unwrap();

        let input_shape = self.input.shape();
        let mut grad_input = Tensor::zeros(
            input_shape.clone(),
            crate::storage::DType::F32,
            self.input.device(),
        );

        use crate::storage::Storage;

        if let Storage::Cpu(cpu_grad) = grad.inner.storage.as_ref() {
            let grad_input_storage = &mut Arc::make_mut(&mut grad_input.inner).storage;
            if let Storage::Cpu(cpu_grad_input) = Arc::make_mut(grad_input_storage) {
                let grad_ptr = cpu_grad.data.as_ref().as_ptr() as *const f32;
                let grad_input_data = Arc::make_mut(&mut cpu_grad_input.data);
                let grad_input_ptr = grad_input_data.as_mut_ptr() as *mut f32;

                let grad_numel = grad.numel() as usize;
                let input_strides = &self.input.inner.strides;

                // Compute the sliced shape
                let sliced_size = (self.end - self.start + self.step - 1) / self.step;
                let mut grad_shape = input_shape.clone();
                grad_shape[self.dim] = sliced_size;

                let ndim = grad_shape.len();
                // Pre-allocate coordinate arrays once instead of per-element
                let mut grad_coords = vec![0usize; ndim];
                let mut input_coords = vec![0usize; ndim];
                let start = self.start as usize;
                let step = self.step as usize;

                // Pre-compute shape strides for index decomposition
                let mut shape_strides = vec![0usize; ndim];
                let mut stride = 1usize;
                for d in (0..ndim).rev() {
                    shape_strides[d] = stride;
                    stride *= grad_shape[d] as usize;
                }

                for i in 0..grad_numel {
                    // Decompose linear index to coordinates
                    let mut temp = i;
                    for d in (0..ndim).rev() {
                        grad_coords[d] = temp % grad_shape[d] as usize;
                        temp /= grad_shape[d] as usize;
                    }

                    // Map to input coordinates
                    input_coords.copy_from_slice(&grad_coords);
                    input_coords[self.dim] = start + (grad_coords[self.dim] * step);

                    // Compute linear index in input using strides
                    let mut input_idx = 0;
                    for d in 0..ndim {
                        input_idx += input_coords[d] * input_strides[d] as usize;
                    }

                    unsafe {
                        *grad_input_ptr.add(input_idx) += *grad_ptr.add(i);
                    }
                }
            }
        }

        vec![Some(grad_input)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "SliceBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

/// CheckpointNode for gradient checkpointing
/// Stores a function and its inputs to recompute during backward pass
#[allow(clippy::type_complexity)]
pub struct CheckpointNode {
    pub inputs: Vec<Tensor>,
    pub edges: Vec<Edge>,
    #[allow(clippy::type_complexity)]
    pub checkpoint_fn: Box<dyn Fn(&[Tensor]) -> Vec<Tensor> + Send + Sync>,
    pub output_ids: Vec<usize>,
}

impl CheckpointNode {
    #[allow(clippy::type_complexity)]
    pub fn new(
        inputs: Vec<Tensor>,
        edges: Vec<Edge>,
        checkpoint_fn: Box<dyn Fn(&[Tensor]) -> Vec<Tensor> + Send + Sync>,
        output_ids: Vec<usize>,
    ) -> Self {
        CheckpointNode {
            inputs,
            edges,
            checkpoint_fn,
            output_ids,
        }
    }
}

impl Node for CheckpointNode {
    #[allow(clippy::unused_async)]
    fn apply(&self, grad_outputs: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        // For checkpointing, we need to recompute the forward pass
        // to get the intermediate activations needed for backward

        // Recompute forward pass with gradients enabled
        // Note: The recomputed outputs would normally be used to compute gradients
        // For now, we use the grad_outputs directly
        let _outputs = (self.checkpoint_fn)(&self.inputs);

        // The grad_outputs contain gradients from the output side.
        // We need to propagate these gradients back through the recomputed graph.
        // For now, we'll return the grad_outputs directly as input gradients.
        // A more sophisticated implementation would compute actual gradients
        // based on the recomputed forward pass.

        // Return gradients for each input based on grad_outputs
        // Note: This is a simplified implementation. In a full implementation,
        // we would need to properly compute gradients through the recomputed graph.
        grad_outputs.to_vec()
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    fn name(&self) -> &str {
        "CheckpointNode"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

/// Checkpoint function that enables gradient checkpointing
/// This stores the function and inputs to recompute during backward pass
pub fn checkpoint<F>(f: F, inputs: &[Tensor]) -> Vec<Tensor>
where
    F: Fn(&[Tensor]) -> Vec<Tensor> + Send + Sync + 'static,
{
    // Check if gradient computation is enabled
    if !is_grad_enabled() {
        // If no gradient, just execute the function normally
        return f(inputs);
    }

    // Execute forward pass normally to get outputs
    let outputs = f(inputs);

    // Store checkpoint information for backward pass
    // This would be attached to the output tensors' autograd metadata
    // For now, we'll create a CheckpointNode and attach it to the outputs

    let output_ids = outputs.iter().map(|t| t.id()).collect();

    // Create edges from input tensors
    let mut edges = Vec::new();
    for input in inputs {
        if let Some(node) = input.grad_fn() {
            edges.push(Edge(node, 0));
        }
    }

    // Create checkpoint node
    let checkpoint_node = Arc::new(CheckpointNode::new(
        inputs.to_vec(),
        edges,
        Box::new(f),
        output_ids,
    ));

    // Attach the checkpoint node to each output
    for output in &outputs {
        if output.requires_grad() {
            // Clone the inner Arc first to avoid temporary value dropped while borrowed
            let mut inner_arc = output.inner.clone();
            // Use Arc::make_mut to ensure we have exclusive access to the TensorImpl
            // This ensures modifications are reflected in the actual tensor
            let inner = Arc::make_mut(&mut inner_arc);
            if let Some(meta) = &inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad_fn = Some(checkpoint_node.clone());
                }
            }
        }
    }

    outputs
}
