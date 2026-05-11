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

pub struct BatchNorm1dBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub bias: Tensor,
    pub batch_mean: Tensor,
    pub batch_var: Tensor,
    pub eps: f32,
    pub edges: Vec<Edge>,
    pub inputs: Vec<Tensor>,
}

impl BatchNorm1dBackward {
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
        BatchNorm1dBackward {
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

impl Node for BatchNorm1dBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None, None, None];
        };

        let input = &self.input;
        let weight = &self.weight;

        let input_shape = input.shape_ref();
        let batch = input_shape[0];
        let channels = input_shape[1];
        let spatial: i64 = if input_shape.len() > 2 {
            input_shape[2..].iter().product()
        } else {
            1
        };
        let n = batch * spatial;

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

        let mut grad_input = Tensor::zeros(input_shape.to_vec(), grad.dtype(), grad.device());
        let grad_input_data = grad_input.data_ptr_f32_mut();

        let mut grad_weight_data = vec![0.0f32; c];
        let mut grad_bias_data = vec![0.0f32; c];

        for ch in 0..c {
            let gamma = weight_data[ch];
            let mean_ch = mean_data[ch];
            let sigma_ch = 1.0 / (var_data[ch] + eps).sqrt();

            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;

            let mut x_hat_vals = vec![0.0f32; b * hw];

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

            for i in 0..b {
                for j in 0..hw {
                    let idx = (i * c + ch) * hw + j;
                    let x_hat = x_hat_vals[i * hw + j];
                    let dx_hat = grad_data[idx] * gamma;
                    let dx = (sigma_ch / n_total) * (n_total * dx_hat - s1 - x_hat * s2);
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
        "BatchNorm1dBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

pub struct RMSNormBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub eps: f32,
    pub normalized_shape: i64,
    pub edges: Vec<Edge>,
    pub inputs: Vec<Tensor>,
}

impl RMSNormBackward {
    pub fn new(
        input: Tensor,
        weight: Tensor,
        eps: f32,
        normalized_shape: i64,
        edges: Vec<Edge>,
    ) -> Self {
        let inputs = vec![input.clone(), weight.clone()];
        RMSNormBackward {
            input,
            weight,
            eps,
            normalized_shape,
            edges,
            inputs,
        }
    }
}

impl Node for RMSNormBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None, None];
        };

        let input = &self.input;
        let weight = &self.weight;
        let eps = self.eps;

        let input_shape = input.shape_ref();
        let ndim = input_shape.len();
        let norm_dim = input_shape[ndim - 1] as usize;
        let outer_size: usize = input_shape[..ndim - 1].iter().map(|&d| d as usize).product();
        let total = outer_size * norm_dim;

        let input_cpu = crate::autograd::ensure_cpu(input);
        let grad_cpu = crate::autograd::ensure_cpu(&grad);
        let weight_cpu = crate::autograd::ensure_cpu(weight);

        let input_data = input_cpu.as_f32_slice();
        let grad_data = grad_cpu.as_f32_slice();
        let weight_data = weight_cpu.as_f32_slice();

        let mut grad_input_data = vec![0.0f32; total];
        let mut grad_weight_data = vec![0.0f32; norm_dim];

        let nd = norm_dim;

        for row in 0..outer_size {
            let base = row * nd;

            let mut sum_sq = 0.0f32;
            for j in 0..nd {
                let x_val = input_data[base + j];
                sum_sq += x_val * x_val;
            }
            let inv_rms = 1.0 / ((sum_sq / nd as f32) + eps).sqrt();

            let mut sum_gx = 0.0f32;
            for j in 0..nd {
                let xn = input_data[base + j] * inv_rms;
                sum_gx += grad_data[base + j] * weight_data[j] * xn;
            }
            let mean_gx = sum_gx / nd as f32;

            for j in 0..nd {
                let xn = input_data[base + j] * inv_rms;
                let g_val = grad_data[base + j] * weight_data[j];
                grad_input_data[base + j] = (g_val - xn * mean_gx) * inv_rms;
                grad_weight_data[j] += grad_data[base + j] * xn;
            }
        }

        let grad_input = Tensor::from_vec(grad_input_data, input_shape.to_vec());
        let grad_weight = Tensor::from_vec(grad_weight_data, vec![norm_dim as i64]);

        vec![Some(grad_input), Some(grad_weight)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    fn name(&self) -> &str {
        "RMSNormBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

pub struct GroupNormBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub bias: Tensor,
    pub num_groups: i64,
    pub eps: f32,
    pub edges: Vec<Edge>,
    pub inputs: Vec<Tensor>,
}

impl GroupNormBackward {
    pub fn new(
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        num_groups: i64,
        eps: f32,
        edges: Vec<Edge>,
    ) -> Self {
        let inputs = vec![input.clone(), weight.clone(), bias.clone()];
        GroupNormBackward {
            input,
            weight,
            bias,
            num_groups,
            eps,
            edges,
            inputs,
        }
    }
}

impl Node for GroupNormBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None, None, None];
        };

        let input = &self.input;
        let weight = &self.weight;
        let num_groups = self.num_groups as usize;
        let eps = self.eps;

        let input_shape = input.shape_ref();
        let n = input_shape[0] as usize;
        let c = input_shape[1] as usize;
        let spatial: usize = input_shape[2..].iter().map(|&d| d as usize).product();
        let s = c / num_groups;
        let d = s * spatial;

        let input_cpu = crate::autograd::ensure_cpu(input);
        let grad_cpu = crate::autograd::ensure_cpu(&grad);
        let weight_cpu = crate::autograd::ensure_cpu(weight);

        let input_data = input_cpu.as_f32_slice();
        let grad_data = grad_cpu.as_f32_slice();
        let weight_data = weight_cpu.as_f32_slice();

        let mut grad_input_data = vec![0.0f32; input_data.len()];
        let mut grad_weight_data = vec![0.0f32; c];
        let mut grad_bias_data = vec![0.0f32; c];

        for batch_idx in 0..n {
            for g in 0..num_groups {
                let group_base = (batch_idx * num_groups + g) * d;
                let ch_base = g * s;

                let mut mean = 0.0f32;
                for i in 0..d {
                    mean += input_data[group_base + i];
                }
                mean /= d as f32;

                let mut var = 0.0f32;
                for i in 0..d {
                    let diff = input_data[group_base + i] - mean;
                    var += diff * diff;
                }
                var /= d as f32;
                let inv_std = 1.0 / (var + eps).sqrt();

                let mut sum_gw = 0.0f32;
                let mut sum_gw_xh = 0.0f32;

                for i in 0..d {
                    let x_val = input_data[group_base + i];
                    let x_hat = (x_val - mean) * inv_std;
                    let chan = i / spatial;
                    let ch_idx = ch_base + chan;
                    let dy = grad_data[group_base + i];
                    let w = weight_data[ch_idx];
                    let dx_hat = dy * w;
                    sum_gw += dx_hat;
                    sum_gw_xh += dx_hat * x_hat;
                    grad_weight_data[ch_idx] += dy * x_hat;
                    grad_bias_data[ch_idx] += dy;
                }

                let mean_gw = sum_gw / d as f32;
                let mean_gw_xh = sum_gw_xh / d as f32;

                for i in 0..d {
                    let x_val = input_data[group_base + i];
                    let x_hat = (x_val - mean) * inv_std;
                    let chan = i / spatial;
                    let ch_idx = ch_base + chan;
                    let dy = grad_data[group_base + i];
                    let w = weight_data[ch_idx];
                    let dx_hat = dy * w;
                    grad_input_data[group_base + i] = (dx_hat - mean_gw - x_hat * mean_gw_xh) * inv_std;
                }
            }
        }

        let grad_input = Tensor::from_vec(grad_input_data, input_shape.to_vec());
        let grad_weight = Tensor::from_vec(grad_weight_data, vec![c as i64]);
        let grad_bias = Tensor::from_vec(grad_bias_data, vec![c as i64]);

        vec![Some(grad_input), Some(grad_weight), Some(grad_bias)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    fn name(&self) -> &str {
        "GroupNormBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}
