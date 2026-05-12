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

pub struct AvgPool2dBackward {
    pub input: Tensor,
    pub kernel_size: i64,
    pub stride: i64,
    pub padding: i64,
    pub edges: Vec<Edge>,
    pub inputs: Vec<Tensor>,
}

impl AvgPool2dBackward {
    pub fn new(
        input: Tensor,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        edges: Vec<Edge>,
    ) -> Self {
        let inputs = vec![input.clone()];
        AvgPool2dBackward {
            input,
            kernel_size,
            stride,
            padding,
            edges,
            inputs,
        }
    }
}

impl Node for AvgPool2dBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };

        let input = &self.input;
        let input_shape = input.shape_ref();
        let grad_shape = grad.shape_ref();
        let batch = input_shape[0] as usize;
        let channels = input_shape[1] as usize;
        let in_h = input_shape[2] as usize;
        let in_w = input_shape[3] as usize;
        let out_h = grad_shape[2] as usize;
        let out_w = grad_shape[3] as usize;
        let k_h = self.kernel_size as usize;
        let k_w = self.kernel_size as usize;
        let stride = self.stride as usize;
        let padding = self.padding as usize;

        let mut grad_input = Tensor::zeros(input_shape.to_vec(), grad.dtype(), grad.device());
        let grad_cpu = crate::autograd::ensure_cpu(&grad);
        let grad_data = grad_cpu.as_f32_slice();
        let grad_input_data = grad_input.data_ptr_f32_mut();

        for b in 0..batch {
            for c_ in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let h_start = (oh * stride).saturating_sub(padding);
                        let h_end = std::cmp::min(in_h, (oh * stride + k_h).saturating_sub(padding));
                        let w_start = (ow * stride).saturating_sub(padding);
                        let w_end = std::cmp::min(in_w, (ow * stride + k_w).saturating_sub(padding));
                        let count = (h_end - h_start) * (w_end - w_start);
                        if count == 0 {
                            continue;
                        }
                        let grad_val =
                            grad_data[((b * channels + c_) * out_h + oh) * out_w + ow] / count as f32;
                        for ih in h_start..h_end {
                            for iw in w_start..w_end {
                                let input_idx = ((b * channels + c_) * in_h + ih) * in_w + iw;
                                unsafe { *grad_input_data.add(input_idx) += grad_val; }
                            }
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
        "AvgPool2dBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

pub struct AdaptiveAvgPool2dBackward {
    pub input: Tensor,
    pub output_size: Vec<i64>,
    pub edges: Vec<Edge>,
    pub inputs: Vec<Tensor>,
}

impl AdaptiveAvgPool2dBackward {
    pub fn new(
        input: Tensor,
        output_size: Vec<i64>,
        edges: Vec<Edge>,
    ) -> Self {
        let inputs = vec![input.clone()];
        AdaptiveAvgPool2dBackward {
            input,
            output_size,
            edges,
            inputs,
        }
    }
}

impl Node for AdaptiveAvgPool2dBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };

        let input = &self.input;
        let input_shape = input.shape_ref();
        let batch = input_shape[0] as usize;
        let channels = input_shape[1] as usize;
        let in_h = input_shape[2] as usize;
        let in_w = input_shape[3] as usize;
        let out_h = self.output_size[0] as usize;
        let out_w = self.output_size[1] as usize;

        let stride_h = in_h / out_h;
        let kernel_h = in_h - (out_h - 1) * stride_h;
        let stride_w = in_w / out_w;
        let kernel_w = in_w - (out_w - 1) * stride_w;
        let window_pixels = kernel_h * kernel_w;
        if window_pixels == 0 {
            let grad_input = Tensor::zeros(input_shape.to_vec(), grad.dtype(), grad.device());
            return vec![Some(grad_input)];
        }

        let mut grad_input = Tensor::zeros(input_shape.to_vec(), grad.dtype(), grad.device());
        let grad_cpu = crate::autograd::ensure_cpu(&grad);
        let grad_data = grad_cpu.as_f32_slice();
        let grad_input_data = grad_input.data_ptr_f32_mut();

        for b in 0..batch {
            for c_ in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let h_start = oh * stride_h;
                        let h_end = h_start + kernel_h;
                        let w_start = ow * stride_w;
                        let w_end = w_start + kernel_w;
                        let grad_val =
                            grad_data[((b * channels + c_) * out_h + oh) * out_w + ow] / window_pixels as f32;
                        for ih in h_start..h_end {
                            for iw in w_start..w_end {
                                let input_idx = ((b * channels + c_) * in_h + ih) * in_w + iw;
                                unsafe { *grad_input_data.add(input_idx) += grad_val; }
                            }
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
        "AdaptiveAvgPool2dBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}
