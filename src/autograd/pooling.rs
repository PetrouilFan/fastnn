pub struct MaxPool2dBackward {
    pub input: Tensor,
    pub kernel_size: i64,
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub edges: Vec<Edge>,
    pub inputs: Vec<Tensor>,
    pub argmax_indices: Vec<usize>,
}

impl MaxPool2dBackward {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Tensor,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
        edges: Vec<Edge>,
        argmax_indices: Vec<usize>,
    ) -> Self {
        let inputs = vec![input.clone()];
        MaxPool2dBackward {
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            edges,
            inputs,
            argmax_indices,
        }
    }
}

impl Node for MaxPool2dBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };

        let input = &self.input;
        let input_shape = input.shape_ref();

        let grad_shape = grad.shape_ref();
        let out_h = grad_shape[2] as usize;
        let out_w = grad_shape[3] as usize;

        // Create zero-initialized gradient input
        let mut grad_input = Tensor::zeros(
            input_shape.to_vec(),
            grad.dtype(),
            grad.device(),
        );

        let grad_cpu = crate::autograd::ensure_cpu(&grad);
        let grad_data = grad_cpu.as_f32_slice();
        let grad_input_data = grad_input.data_ptr_f32_mut();

        let b_size = input_shape[0] as usize;
        let c_size = input_shape[1] as usize;

        // Use cached argmax indices
        let mut idx = 0;
        for b in 0..b_size {
            for c_ in 0..c_size {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let grad_out_idx = ((b * c_size + c_) * out_h + oh) * out_w + ow;
                        let grad_val = grad_data[grad_out_idx];
                        let max_linear = self.argmax_indices[idx];
                        idx += 1;
                        // SAFETY: indices are computed from valid tensor shapes
                        unsafe { *grad_input_data.add(max_linear) += grad_val; }
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
        "MaxPool2dBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}
