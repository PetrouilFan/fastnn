pub struct MaxPool2dBackward {
    pub input: Tensor,
    pub kernel_size: i64,
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub edges: Vec<Edge>,
    pub inputs: Vec<Tensor>,
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
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        let grad_shape = grad.shape_ref();
        let out_h = grad_shape[2];
        let out_w = grad_shape[3];

        let kernel_size = self.kernel_size;
        let stride = self.stride;
        let padding = self.padding;
        let dilation = self.dilation;

        // Create zero-initialized gradient input
        let mut grad_input = Tensor::zeros(
            input_shape.to_vec(),
            grad.dtype(),
            grad.device(),
        );

        let input_cpu = crate::autograd::ensure_cpu(input);
        let grad_cpu = crate::autograd::ensure_cpu(&grad);
        let input_data = input_cpu.as_f32_slice();
        let grad_data = grad_cpu.as_f32_slice();
        let grad_input_data = grad_input.data_ptr_f32_mut();

        let b_size = batch_size as usize;
        let c_size = channels as usize;
        let in_h_size = in_h as usize;
        let in_w_size = in_w as usize;
        let out_h_size = out_h as usize;
        let out_w_size = out_w as usize;
        let k_size = kernel_size as usize;
        let s = stride as usize;
        let d = dilation as usize;
        let pad = padding;

        for b in 0..b_size {
            for c in 0..c_size {
                for oh in 0..out_h_size {
                    for ow in 0..out_w_size {
                        // Find which input element was the max in this window
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_h: i64 = 0;
                        let mut max_w: i64 = 0;

                        for kh in 0..k_size {
                            for kw in 0..k_size {
                                let ih = (oh * s + kh * d) as i64 - pad;
                                let iw = (ow * s + kw * d) as i64 - pad;

                                if ih >= 0 && ih < in_h && iw >= 0 && iw < in_w {
                                    let idx = ((b * c_size + c) * in_h_size + ih as usize) * in_w_size + iw as usize;
                                    let val = input_data[idx];
                                    if val > max_val {
                                        max_val = val;
                                        max_h = ih;
                                        max_w = iw;
                                    }
                                }
                            }
                        }

                        let grad_val = grad_data
                            [((b * c_size + c) * out_h_size + oh) * out_w_size + ow];

                        let grad_out_idx =
                            ((b * c_size + c) * in_h_size + max_h as usize) * in_w_size + max_w as usize;
                        // SAFETY: indices are bounds-checked by the loop ranges
                        unsafe {
                            *grad_input_data.add(grad_out_idx) += grad_val;
                        }
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
