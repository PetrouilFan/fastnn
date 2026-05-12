pub struct UpsampleBackward {
    pub input: Tensor,
    pub scale_factor: f64,
    pub mode: String,
    pub edges: Vec<Edge>,
    pub inputs: Vec<Tensor>,
}

impl UpsampleBackward {
    pub fn new(input: Tensor, scale_factor: f64, mode: String, edges: Vec<Edge>, inputs: Vec<Tensor>) -> Self {
        UpsampleBackward { input, scale_factor, mode, edges, inputs }
    }
}

impl Node for UpsampleBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let input = &self.input;
        let input_shape = input.shape_ref();
        let grad_shape = grad.shape_ref();
        let ndim = input_shape.len();
        let in_h = input_shape[ndim - 2] as usize;
        let in_w = input_shape[ndim - 1] as usize;
        let out_h = grad_shape[ndim - 2] as usize;
        let out_w = grad_shape[ndim - 1] as usize;
        let spatial_in = in_h * in_w;
        let spatial_out = out_h * out_w;
        let batch_channels: usize = input_shape[..ndim - 2].iter().map(|&x| x as usize).product();
        let scale = self.scale_factor as f32;

        if self.mode == "nearest" {
            let mut grad_input = Tensor::zeros(input_shape.to_vec(), grad.dtype(), grad.device());
            let grad_cpu = crate::autograd::ensure_cpu(&grad);
            let grad_data = grad_cpu.as_f32_slice();
            let grad_input_data = grad_input.data_ptr_f32_mut();

            for bc in 0..batch_channels {
                for oh in 0..out_h {
                    let ih = (oh as f32 / scale).min((in_h - 1) as f32) as usize;
                    for ow in 0..out_w {
                        let iw = (ow as f32 / scale).min((in_w - 1) as f32) as usize;
                        let g = grad_data[bc * spatial_out + oh * out_w + ow];
                        unsafe {
                            *grad_input_data.add(bc * spatial_in + ih * in_w + iw) += g;
                        }
                    }
                }
            }
            vec![Some(grad_input)]
        } else {
            let mut grad_input = Tensor::zeros(input_shape.to_vec(), grad.dtype(), grad.device());
            let grad_cpu = crate::autograd::ensure_cpu(&grad);
            let grad_data = grad_cpu.as_f32_slice();
            let grad_input_data = grad_input.data_ptr_f32_mut();

            for bc in 0..batch_channels {
                for oh in 0..out_h {
                    let ih_f = oh as f32 / scale;
                    let ih0 = ih_f.min((in_h - 2) as f32) as usize;
                    let ih1 = (ih0 + 1).min(in_h - 1);
                    let dy = ih_f - ih0 as f32;
                    for ow in 0..out_w {
                        let iw_f = ow as f32 / scale;
                        let iw0 = iw_f.min((in_w - 2) as f32) as usize;
                        let iw1 = (iw0 + 1).min(in_w - 1);
                        let dx = iw_f - iw0 as f32;

                        let g = grad_data[bc * spatial_out + oh * out_w + ow];

                        let base = bc * spatial_in;
                        unsafe {
                            *grad_input_data.add(base + ih0 * in_w + iw0) += g * (1.0 - dy) * (1.0 - dx);
                            *grad_input_data.add(base + ih0 * in_w + iw1) += g * (1.0 - dy) * dx;
                            *grad_input_data.add(base + ih1 * in_w + iw0) += g * dy * (1.0 - dx);
                            *grad_input_data.add(base + ih1 * in_w + iw1) += g * dy * dx;
                        }
                    }
                }
            }
            vec![Some(grad_input)]
        }
    }

    fn next_edges(&self) -> &[Edge] { &self.edges }
    fn num_inputs(&self) -> usize { self.inputs.len() }
    fn name(&self) -> &str { "UpsampleBackward" }
    fn inputs(&self) -> &[Tensor] { &self.inputs }
}
