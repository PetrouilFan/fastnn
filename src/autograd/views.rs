use smallvec::SmallVec;

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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let shape = self.input.shape_ref();
        vec![Some(grad.reshape(shape.to_vec()))]
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };

        let input_shape = self.input.shape_ref();
        let mut grad_input = Tensor::zeros(
            input_shape.to_vec(),
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
                let mut grad_shape = input_shape.to_vec();
                grad_shape[self.dim] = sliced_size;

                let ndim = grad_shape.len();
                // Pre-allocate coordinate arrays once instead of per-element
                let mut grad_coords: SmallVec<[usize; 8]> = SmallVec::new();
                grad_coords.resize(ndim, 0);
                let mut input_coords: SmallVec<[usize; 8]> = SmallVec::new();
                input_coords.resize(ndim, 0);
                let start = self.start as usize;
                let step = self.step as usize;

                // Pre-compute shape strides for index decomposition
                let mut shape_strides: SmallVec<[usize; 8]> = SmallVec::new();
                shape_strides.resize(ndim, 0);
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
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, output_tensor_id: usize) -> Vec<Option<Tensor>> {
        // For gradient checkpointing, we need to recompute the forward pass
        // WITH gradient tracking to build a computation graph for backward.
        // Save current gradient state and ensure gradients are enabled.
        let prev_grad_enabled = is_grad_enabled();

        // Enable gradients for recomputation if they're currently disabled
        if !prev_grad_enabled {
            no_grad_exit(); // Decrement no_grad counter to enable gradients
        }

        // Recompute the forward pass, building computation graph
        let recomputed_outputs = (self.checkpoint_fn)(&self.inputs);

        // Restore previous gradient tracking state after recomputation
        if !prev_grad_enabled {
            no_grad_enter(); // Re-disable gradients if they were disabled before
        }

        // Find which output the gradient is for using output_ids
        let output_idx = self
            .output_ids
            .iter()
            .position(|&id| id == output_tensor_id)
            .expect("output_tensor_id not found in checkpoint output_ids");

        // Backpropagate through the specific output that triggered backward
        if let Some(grad) = grad_outputs.into_iter().next().flatten() {
            crate::autograd::backward(&recomputed_outputs[output_idx], Some(grad));
        }

        // Collect gradients for each input tensor
        let input_grads: Vec<Option<Tensor>> = self.inputs
            .iter()
            .map(|input| input.grad())
            .collect();

        input_grads
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

pub struct TransposeBackward {
    pub input: Tensor,
    pub dim0: usize,
    pub dim1: usize,
    pub edges: Vec<Edge>,
}

impl TransposeBackward {
    pub fn new(input: Tensor, dim0: usize, dim1: usize, edges: Vec<Edge>) -> Self {
        TransposeBackward { input, dim0, dim1, edges }
    }
}

impl Node for TransposeBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        vec![Some(grad.transpose(self.dim0, self.dim1))]
    }

    fn next_edges(&self) -> &[Edge] { &self.edges }
    fn num_inputs(&self) -> usize { 1 }
    fn name(&self) -> &str { "TransposeBackward" }
    fn inputs(&self) -> &[Tensor] { std::slice::from_ref(&self.input) }
}

pub struct PermuteBackward {
    pub input: Tensor,
    pub dims: Vec<i64>,
    pub edges: Vec<Edge>,
}

impl PermuteBackward {
    pub fn new(input: Tensor, dims: Vec<i64>, edges: Vec<Edge>) -> Self {
        PermuteBackward { input, dims, edges }
    }
}

impl Node for PermuteBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let ndim = self.dims.len();
        let mut inverse = vec![0i64; ndim];
        for (i, &d) in self.dims.iter().enumerate() {
            inverse[d as usize] = i as i64;
        }
        vec![Some(grad.permute(inverse))]
    }

    fn next_edges(&self) -> &[Edge] { &self.edges }
    fn num_inputs(&self) -> usize { 1 }
    fn name(&self) -> &str { "PermuteBackward" }
    fn inputs(&self) -> &[Tensor] { std::slice::from_ref(&self.input) }
}

pub struct ExpandBackward {
    pub input: Tensor,
    pub edges: Vec<Edge>,
}

impl ExpandBackward {
    pub fn new(input: Tensor, edges: Vec<Edge>) -> Self {
        ExpandBackward { input, edges }
    }
}

impl Node for ExpandBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        vec![Some(crate::autograd::sum_to_shape(grad, self.input.shape_ref()))]
    }

    fn next_edges(&self) -> &[Edge] { &self.edges }
    fn num_inputs(&self) -> usize { 1 }
    fn name(&self) -> &str { "ExpandBackward" }
    fn inputs(&self) -> &[Tensor] { std::slice::from_ref(&self.input) }
}

pub struct CatBackward {
    pub inputs: Vec<Tensor>,
    pub dim: usize,
    pub split_sizes: Vec<usize>,
    pub edges: Vec<Edge>,
}

impl CatBackward {
    pub fn new(inputs: Vec<Tensor>, dim: usize, split_sizes: Vec<usize>, edges: Vec<Edge>) -> Self {
        CatBackward { inputs, dim, split_sizes, edges }
    }
}

impl Node for CatBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None; self.inputs.len()];
        };
        let mut result = Vec::with_capacity(self.inputs.len());
        let mut offset = 0i64;
        for (i, input) in self.inputs.iter().enumerate() {
            let size = self.split_sizes[i] as i64;
            if input.requires_grad() {
                let chunk = grad.slice(self.dim, offset, offset + size, 1);
                result.push(Some(chunk));
            } else {
                result.push(None);
            }
            offset += size;
        }
        result
    }

    fn next_edges(&self) -> &[Edge] { &self.edges }
    fn num_inputs(&self) -> usize { self.inputs.len() }
    fn name(&self) -> &str { "CatBackward" }
    fn inputs(&self) -> &[Tensor] { &self.inputs }
}

pub struct RepeatBackward {
    pub input: Tensor,
    pub repeats: Vec<i64>,
    pub edges: Vec<Edge>,
}

impl RepeatBackward {
    pub fn new(input: Tensor, repeats: Vec<i64>, edges: Vec<Edge>) -> Self {
        RepeatBackward { input, repeats, edges }
    }
}

impl Node for RepeatBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let input_shape = self.input.shape_ref().to_vec();
        let ndim = self.input.ndim();
        let offset = self.repeats.len() - ndim;
        let mut grad = grad;
        for j in (0..self.repeats.len()).rev() {
            let r = self.repeats[j];
            if r > 1 {
                let orig_size = if j < offset { 1i64 } else { input_shape[j - offset] };
                let grad_shape = grad.shape();
                let mut new_shape: Vec<i64> = Vec::with_capacity(grad_shape.len() + 1);
                for (k, &s) in grad_shape.iter().enumerate() {
                    if k == j {
                        new_shape.push(orig_size);
                        new_shape.push(r);
                    } else {
                        new_shape.push(s);
                    }
                }
                grad = grad.reshape(new_shape);
                grad = grad.sum((j + 1) as i32, false);
            }
        }
        if offset > 0 {
            for d in (0..offset).rev() {
                grad = grad.squeeze(Some(d));
            }
        }
        vec![Some(grad)]
    }

    fn next_edges(&self) -> &[Edge] { &self.edges }
    fn num_inputs(&self) -> usize { 1 }
    fn name(&self) -> &str { "RepeatBackward" }
    fn inputs(&self) -> &[Tensor] { std::slice::from_ref(&self.input) }
}

pub struct WhereBackward {
    pub inputs: Vec<Tensor>,
    pub condition: Tensor,
    pub edges: Vec<Edge>,
}

impl WhereBackward {
    pub fn new(inputs: Vec<Tensor>, condition: Tensor, edges: Vec<Edge>) -> Self {
        WhereBackward { inputs, condition, edges }
    }
}

impl Node for WhereBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None, None];
        };
        let mask = self.condition.gt_scalar(0.0);
        let mask_not = mask.logical_not();
        let grad_self = grad.mul(&mask);
        let grad_other = grad.mul(&mask_not);
        vec![Some(grad_self), Some(grad_other)]
    }

    fn next_edges(&self) -> &[Edge] { &self.edges }
    fn num_inputs(&self) -> usize { 2 }
    fn name(&self) -> &str { "WhereBackward" }
    fn inputs(&self) -> &[Tensor] { &self.inputs }
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
    let mut edges = Vec::with_capacity(inputs.len());
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
