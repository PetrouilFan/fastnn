pub struct DropoutBackward {
    pub mask: Tensor,
    pub edges: Vec<Edge>,
    pub inputs: Vec<Tensor>,
}

impl DropoutBackward {
    pub fn new(mask: Tensor, edges: Vec<Edge>, inputs: Vec<Tensor>) -> Self {
        DropoutBackward { mask, edges, inputs }
    }
}

impl Node for DropoutBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        vec![Some(grad.mul(&self.mask))]
    }
    fn next_edges(&self) -> &[Edge] { &self.edges }
    fn num_inputs(&self) -> usize { self.inputs.len() }
    fn name(&self) -> &str { "DropoutBackward" }
    fn inputs(&self) -> &[Tensor] { &self.inputs }
}

pub struct Dropout2dBackward {
    pub channel_mask: Tensor,
    pub edges: Vec<Edge>,
    pub inputs: Vec<Tensor>,
}

impl Dropout2dBackward {
    pub fn new(channel_mask: Tensor, edges: Vec<Edge>, inputs: Vec<Tensor>) -> Self {
        Dropout2dBackward { channel_mask, edges, inputs }
    }
}

impl Node for Dropout2dBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        vec![Some(grad.mul(&self.channel_mask))]
    }
    fn next_edges(&self) -> &[Edge] { &self.edges }
    fn num_inputs(&self) -> usize { self.inputs.len() }
    fn name(&self) -> &str { "Dropout2dBackward" }
    fn inputs(&self) -> &[Tensor] { &self.inputs }
}
