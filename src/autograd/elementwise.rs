/// Fused Linear backward operation for better performance.
/// Computes gradients for weight, bias, and input in a single backward operation.
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad_output = grad_outputs.into_iter().next().flatten().unwrap();

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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
        let a = &self.inputs[0];
        let b = &self.inputs[1];

        // Handle broadcasting: if input shape doesn't match output shape,
        // sum over the extra dimensions. Minimize clones by reusing grad when possible.
        let a_matches = a.shape() == grad.shape();
        let b_matches = b.shape() == grad.shape();

        let (grad_a, grad_b) = match (a_matches, b_matches) {
            (true, true) => {
                // Both match: clone once, reuse for the other
                (grad.clone(), grad)
            }
            (true, false) => {
                // a matches, b doesn't: reuse grad for a, clone and modify for b
                let mut grad_b = grad.clone();
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
                (grad, grad_b)
            }
            (false, true) => {
                // b matches, a doesn't: clone and modify for a, reuse grad for b
                let mut grad_a = grad.clone();
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
                (grad_a, grad)
            }
            (false, false) => {
                // Neither matches: clone for both and modify each
                let mut grad_a = grad.clone();
                let mut grad_b = grad.clone();

                let diff_a = grad_a.shape().len() as i32 - a.shape().len() as i32;
                for i in (0..grad_a.shape().len()).rev() {
                    let a_dim = if i as i32 >= diff_a {
                        a.shape()[(i as i32 - diff_a) as usize]
                    } else {
                        1
                    };
                    if a_dim != grad_a.shape()[i] {
                        grad_a = grad_a.sum(i as i32, false);
                    }
                }

                let diff_b = grad_b.shape().len() as i32 - b.shape().len() as i32;
                for i in (0..grad_b.shape().len()).rev() {
                    let b_dim = if i as i32 >= diff_b {
                        b.shape()[(i as i32 - diff_b) as usize]
                    } else {
                        1
                    };
                    if b_dim != grad_b.shape()[i] {
                        grad_b = grad_b.sum(i as i32, false);
                    }
                }
                (grad_a, grad_b)
            }
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
        let a = &self.inputs[0];
        let b = &self.inputs[1];

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
            grad.neg()
        } else {
            let mut grad_b = grad.neg();
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
        let a = &self.inputs[0];
        let b = &self.inputs[1];

        let b_sq = b.mul(b);

        // d/da (a/b) = 1/b
        let grad_a = grad.div(b);

        // d/db (a/b) = -a/b^2
        let grad_b = grad.mul(a).div(&b_sq).neg();

        // Handle broadcasting for both gradients
        let a_matches = a.shape() == grad_a.shape();
        let b_matches = b.shape() == grad_b.shape();

        let final_grad_a = if a_matches {
            grad_a.clone()
        } else {
            let mut g = grad_a;
            let diff = g.shape().len() as i32 - a.shape().len() as i32;
            for i in (0..g.shape().len()).rev() {
                let a_dim = if i as i32 >= diff {
                    a.shape()[(i as i32 - diff) as usize]
                } else {
                    1
                };
                if a_dim != g.shape()[i] {
                    g = g.sum(i as i32, false);
                }
            }
            g
        };

        let final_grad_b = if b_matches {
            grad_b.clone()
        } else {
            let mut g = grad_b;
            let diff = g.shape().len() as i32 - b.shape().len() as i32;
            for i in (0..g.shape().len()).rev() {
                let b_dim = if i as i32 >= diff {
                    b.shape()[(i as i32 - diff) as usize]
                } else {
                    1
                };
                if b_dim != g.shape()[i] {
                    g = g.sum(i as i32, false);
                }
            }
            g
        };

        vec![Some(final_grad_a), Some(final_grad_b)]
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(
            grad_outputs.into_iter().next().flatten().unwrap().neg(),
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
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

pub struct LeakyReLUBackward {
    pub input: Tensor,
    pub negative_slope: f32,
    pub edges: Vec<Edge>,
}

impl LeakyReLUBackward {
    pub fn new(input: Tensor, negative_slope: f32, edges: Vec<Edge>) -> Self {
        LeakyReLUBackward {
            input,
            negative_slope,
            edges,
        }
    }
}

impl Node for LeakyReLUBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad_output = grad_outputs.into_iter().next().flatten().unwrap();
        let mask = self.input.gt_scalar(0.0);
        let _neg_slope_t = Tensor::from_scalar(self.negative_slope);
        let grad_input = mask.mul(&grad_output).add(
            &mask
                .logical_not()
                .mul(&grad_output)
                .mul_scalar(self.negative_slope),
        );
        vec![Some(grad_input)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "LeakyReLUBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

pub struct SoftplusBackward {
    pub input: Tensor,
    pub beta: f32,
    pub threshold: f32,
    pub edges: Vec<Edge>,
}

impl SoftplusBackward {
    pub fn new(input: Tensor, beta: f32, threshold: f32, edges: Vec<Edge>) -> Self {
        SoftplusBackward {
            input,
            beta,
            threshold,
            edges,
        }
    }
}

impl Node for SoftplusBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad_output = grad_outputs.into_iter().next().flatten().unwrap();
        let bx = self.input.mul_scalar(self.beta);
        let mask = bx.gt_scalar(self.threshold);
        let exp_bx = bx.exp();
        let sigmoid_bx = exp_bx.div(&exp_bx.add_scalar(1.0));
        let grad_input = mask
            .mul(&grad_output)
            .add(&mask.logical_not().mul(&sigmoid_bx).mul(&grad_output));
        vec![Some(grad_input)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "SoftplusBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

pub struct HardswishBackward {
    pub input: Tensor,
    pub edges: Vec<Edge>,
}

impl HardswishBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        HardswishBackward { input, edges }
    }
}

impl Node for HardswishBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad_output = grad_outputs.into_iter().next().flatten().unwrap();
        let x_plus_3 = self.input.add_scalar(3.0);
        let relu6 = x_plus_3.clamp(0.0, 6.0);
        let lt_minus3 = self.input.lt_scalar(-3.0);
        let gt_3 = self.input.gt_scalar(3.0);
        let between = lt_minus3.logical_not().mul(&gt_3.logical_not());
        let grad = between
            .mul(&relu6.div_scalar(6.0).add(&self.input.div_scalar(6.0)))
            .add(&gt_3.mul_scalar(1.0));
        vec![Some(grad.mul(&grad_output))]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "HardswishBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

pub struct EluBackward {
    pub input: Tensor,
    pub alpha: f32,
    pub edges: Vec<Edge>,
}

impl EluBackward {
    pub fn new(input: Tensor, alpha: f32, edges: Vec<Edge>) -> Self {
        EluBackward {
            input,
            alpha,
            edges,
        }
    }
}

impl Node for EluBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad_output = grad_outputs.into_iter().next().flatten().unwrap();
        // d/dx elu(x) = 1 if x > 0 else alpha * exp(x)
        let mask = self.input.gt_scalar(0.0);
        let exp_x = self.input.exp();
        let grad_input = mask.mul(&grad_output).add(
            &mask
                .logical_not()
                .mul(&exp_x)
                .mul_scalar(self.alpha)
                .mul(&grad_output),
        );
        vec![Some(grad_input)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "EluBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

// Minimum/Maximum backward nodes
#[allow(dead_code)]
pub struct MinimumBackward {
    pub inputs: Vec<Tensor>,
    pub edges: Vec<Edge>,
}

impl MinimumBackward {
    pub fn new(inputs: Vec<Tensor>, edges: Vec<Edge>) -> Self {
        MinimumBackward { inputs, edges }
    }
}

impl Node for MinimumBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
        let a = &self.inputs[0];
        let b = &self.inputs[1];

        // grad_a = grad * (a <= b)
        // grad_b = grad * (a > b)
        let mask = a.le_tensor(b);
        let mask_not = mask.logical_not();

        let mut grad_a = grad.mul(&mask);
        let mut grad_b = grad.mul(&mask_not);

        // Handle broadcasting
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
        "MinimumBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

#[allow(dead_code)]
pub struct MaximumBackward {
    pub inputs: Vec<Tensor>,
    pub edges: Vec<Edge>,
}

impl MaximumBackward {
    pub fn new(inputs: Vec<Tensor>, edges: Vec<Edge>) -> Self {
        MaximumBackward { inputs, edges }
    }
}

impl Node for MaximumBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = grad_outputs.into_iter().next().flatten().unwrap();
        let a = &self.inputs[0];
        let b = &self.inputs[1];

        // grad_a = grad * (a >= b)
        // grad_b = grad * (a < b)
        let mask = a.ge_tensor(b);
        let mask_not = mask.logical_not();

        let mut grad_a = grad.mul(&mask);
        let mut grad_b = grad.mul(&mask_not);

        // Handle broadcasting
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

    fn name(&self) -> &str {
        "MaximumBackward"
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}
