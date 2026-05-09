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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None, None];
        };
        let a = &self.inputs[0];
        let b = &self.inputs[1];

        // If grad is a scalar (e.g., from sum()), we need to expand it to the output shape
        let grad_shape = grad.shape_ref();
        let grad = if grad_shape.is_empty() {
            // Expand scalar gradient to match the output shape [a.shape[0], b.shape[1]]
            let output_shape = vec![a.shape_ref()[0], b.shape_ref()[1]];
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

