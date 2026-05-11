pub struct BatchNorm2dBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub bias: Tensor,
    pub batch_mean: Tensor,
    pub batch_var: Tensor,
    pub eps: f32,
    pub edges: Vec<Edge>,
    pub inputs: Vec<Tensor>,
}

impl BatchNorm2dBackward {
    pub fn new(
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        batch_mean: Tensor,
        batch_var: Tensor,
        eps: f32,
        edges: Vec<Edge>,
    ) -> Self {
        let inputs = vec![input.clone(), weight.clone(), bias.clone()];
        BatchNorm2dBackward {
            input,
            weight,
            bias,
            batch_mean,
            batch_var,
            eps,
            edges,
            inputs,
        }
    }
}

impl Node for BatchNorm2dBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None, None, None];
        };

        let input = &self.input;
        let weight = &self.weight;

        let input_shape = input.shape_ref();
        let batch = input_shape[0];
        let channels = input_shape[1];
        let in_h = input_shape[2];
        let in_w = input_shape[3];
        let spatial = in_h * in_w;
        let n = batch * spatial; // total elements per channel

        let mean = &self.batch_mean;
        let var = &self.batch_var;
        let eps = self.eps;

        let input_cpu = crate::autograd::ensure_cpu(input);
        let grad_cpu = crate::autograd::ensure_cpu(&grad);
        let weight_cpu = crate::autograd::ensure_cpu(weight);
        let mean_cpu = crate::autograd::ensure_cpu(mean);
        let var_cpu = crate::autograd::ensure_cpu(var);

        let input_data = input_cpu.as_f32_slice();
        let grad_data = grad_cpu.as_f32_slice();
        let weight_data = weight_cpu.as_f32_slice();
        let mean_data = mean_cpu.as_f32_slice();
        let var_data = var_cpu.as_f32_slice();

        let b = batch as usize;
        let c = channels as usize;
        let hw = spatial as usize;
        let n_total = n as f32;

        // Allocate grad_input
        let mut grad_input = Tensor::zeros(input_shape.to_vec(), grad.dtype(), grad.device());
        let grad_input_data = grad_input.data_ptr_f32_mut();

        // Allocate grad_weight and grad_bias
        let mut grad_weight_data = vec![0.0f32; c];
        let mut grad_bias_data = vec![0.0f32; c];

        for ch in 0..c {
            let gamma = weight_data[ch];
            let mean_ch = mean_data[ch];
            let sigma_ch = 1.0 / (var_data[ch] + eps).sqrt();

            // First pass: compute x_hat, dx_hat, S1, S2
            let mut s1 = 0.0f32; // sum(dx_hat) over batch+spatial
            let mut s2 = 0.0f32; // sum(dx_hat * x_hat) over batch+spatial

            let mut x_hat_vals = vec![0.0f32; b * hw]; // store x_hat for second pass

            for i in 0..b {
                for j in 0..hw {
                    let idx = (i * c + ch) * hw + j;
                    let x_val = input_data[idx];
                    let x_hat = (x_val - mean_ch) * sigma_ch;
                    let dx_hat = grad_data[idx] * gamma;
                    x_hat_vals[i * hw + j] = x_hat;
                    s1 += dx_hat;
                    s2 += dx_hat * x_hat;
                    grad_bias_data[ch] += grad_data[idx];
                    grad_weight_data[ch] += grad_data[idx] * x_hat;
                }
            }

            // Second pass: compute dx
            for i in 0..b {
                for j in 0..hw {
                    let idx = (i * c + ch) * hw + j;
                    let x_hat = x_hat_vals[i * hw + j];
                    let dx_hat = grad_data[idx] * gamma;
                    // dx = (sigma / N) * (N * dx_hat - S1 - x_hat * S2)
                    let dx = (sigma_ch / n_total) * (n_total * dx_hat - s1 - x_hat * s2);
                    // SAFETY: idx is bounds-checked
                    unsafe {
                        *grad_input_data.add(idx) = dx;
                    }
                }
            }
        }

        let grad_weight = Tensor::from_vec(grad_weight_data, vec![channels]);
        let grad_bias = Tensor::from_vec(grad_bias_data, vec![channels]);

        vec![Some(grad_input), Some(grad_weight), Some(grad_bias)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    fn name(&self) -> &str {
        "BatchNorm2dBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}
