pub struct LayerNormBackward {
    pub inputs: Vec<Tensor>,
    pub normalized: Tensor,
    pub mean: Tensor,
    pub variance: Tensor,
    pub eps: f32,
    pub edges: Vec<Edge>,
}

impl LayerNormBackward {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        normalized: Tensor,
        mean: Tensor,
        variance: Tensor,
        eps: f32,
        edges: Vec<Edge>,
    ) -> Self {
        LayerNormBackward {
            inputs: vec![input, weight, bias],
            normalized,
            mean,
            variance,
            eps,
            edges,
        }
    }
}

impl Node for LayerNormBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None, None, None];
        };
        let input = &self.inputs[0];
        let weight = &self.inputs[1];
        let _bias = &self.inputs[2];

        let shape = input.shape_ref();
        let outer_size: usize = shape[..shape.len() - 1]
            .iter()
            .map(|&d| d as usize)
            .product();
        let norm_dim: usize = shape[shape.len() - 1] as usize;
        let total = outer_size * norm_dim;

        let grad_cpu = crate::autograd::ensure_cpu(&grad);
        let x_hat_cpu = self.normalized.to_cpu();
        let var_cpu = self.variance.to_cpu();

        let grad_data = grad_cpu.as_f32_slice();
        let x_hat_data = x_hat_cpu.as_f32_slice();
        let var_data = var_cpu.as_f32_slice();

        let weight_data = weight.as_f32_slice();

        let mut grad_input_data = vec![0.0f32; total];
        let mut grad_weight_data = vec![0.0f32; norm_dim];
        let mut grad_bias_data = vec![0.0f32; norm_dim];

        crate::kernels::cpu::layer_norm_backward_f32(
            grad_data,
            x_hat_data,
            Some(weight_data),
            outer_size,
            norm_dim,
            self.eps,
            var_data,
            &mut grad_input_data,
            &mut grad_weight_data,
            &mut grad_bias_data,
        );

        let grad_input = Tensor::from_vec(grad_input_data, shape.to_vec());
        let grad_weight = Tensor::from_vec(grad_weight_data, vec![norm_dim as i64]);
        let grad_bias = Tensor::from_vec(grad_bias_data, vec![norm_dim as i64]);

        vec![Some(grad_input), Some(grad_weight), Some(grad_bias)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        3
    }

    fn name(&self) -> &str {
        "LayerNormBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}
pub struct EmbeddingBackward {
    pub weight: Tensor,
    pub indices: Tensor,
    pub edges: Vec<Edge>,
}

impl EmbeddingBackward {
    pub fn new(weight: Tensor, indices: Tensor, edges: Vec<Edge>) -> Self {
        EmbeddingBackward {
            weight,
            indices,
            edges,
        }
    }
}

impl Node for EmbeddingBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };

        let weight_shape = self.weight.shape_ref().to_vec();
        let num_embeddings = weight_shape[0] as usize;
        let embedding_dim = weight_shape[1] as usize;

        let mut grad_weight_data = vec![0.0f32; num_embeddings * embedding_dim];

        let grad_data = grad.to_cpu().as_f32_slice().to_vec();
        let indices_data = self.indices.to_cpu().as_i64_slice();
        let grad_shape = grad.shape_ref().to_vec();

        if grad_shape.len() == 2 {
            for i in 0..indices_data.len() {
                let idx = indices_data[i] as usize;
                if idx < num_embeddings {
                    for j in 0..embedding_dim {
                        grad_weight_data[idx * embedding_dim + j] += grad_data[i * embedding_dim + j];
                    }
                }
            }
        } else if grad_shape.len() == 3 {
            let batch_size = grad_shape[0] as usize;
            let seq_len = grad_shape[1] as usize;
            for i in 0..batch_size {
                for k in 0..seq_len {
                    let idx = indices_data[i * seq_len + k] as usize;
                    if idx < num_embeddings {
                        for j in 0..embedding_dim {
                            grad_weight_data[idx * embedding_dim + j] +=
                                grad_data[(i * seq_len + k) * embedding_dim + j];
                        }
                    }
                }
            }
        }

        let grad_weight = Tensor::from_vec(grad_weight_data, weight_shape);
        vec![Some(grad_weight)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "EmbeddingBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.weight)
    }
}

pub struct CrossEntropyBackward {
    pub logits: Tensor,
    pub targets: Tensor,
    pub reduction: String,
    pub edges: Vec<Edge>,
}

impl CrossEntropyBackward {
    pub fn new(logits: Tensor, targets: Tensor, reduction: String, edges: Vec<Edge>) -> Self {
        CrossEntropyBackward {
            logits,
            targets,
            reduction,
            edges,
        }
    }
}

impl Node for CrossEntropyBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad_output_tensor) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let grad_out = grad_output_tensor.item();

        let logits = &self.logits;
        let targets = &self.targets;
        let batch_size = logits.shape_ref()[0] as usize;
        let num_classes = logits.shape_ref()[1] as usize;

        let logits_cpu = crate::autograd::ensure_cpu(logits);
        let logits_data = logits_cpu.as_f32_slice();

        // Convert targets to integer indices
        let targets_cpu = crate::autograd::ensure_cpu(targets);
        let targets_i64 = targets_cpu.as_i64_slice();

        // Convert to f32 representation for kernel (kernel expects f32 bit patterns of indices)
        let targets_data: Vec<f32> = targets_i64.iter().map(|&x| x as f32).collect();

        let mut grad_logits_data = vec![0.0f32; batch_size * num_classes];

        crate::kernels::cpu::cross_entropy_backward_f32(
            logits_data,
            &targets_data,
            grad_out,
            batch_size,
            num_classes,
            &self.reduction,
            &mut grad_logits_data,
        );

        let grad_logits = Tensor::from_vec(
            grad_logits_data,
            vec![batch_size as i64, num_classes as i64],
        );

        let grad_logits = match logits.device() {
            crate::storage::Device::Wgpu(device_id) => grad_logits.to_gpu(device_id),
            _ => grad_logits,
        };

        vec![Some(grad_logits)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "CrossEntropyBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.logits)
    }
}
pub struct MSELossBackward {
    pub pred: Tensor,
    pub target: Tensor,
    pub reduction: String,
    pub edges: Vec<Edge>,
    two_scalar: Tensor,
}

impl MSELossBackward {
    pub fn new(pred: Tensor, target: Tensor, reduction: String, edges: Vec<Edge>) -> Self {
        MSELossBackward {
            pred,
            target,
            reduction,
            edges,
            two_scalar: Tensor::from_scalar(2.0),
        }
    }
}

impl Node for MSELossBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
        let Some(grad) = crate::autograd::extract_first_grad(grad_outputs) else {
            return vec![None];
        };
        let diff = self.pred.sub(&self.target);

        let grad_loss = match self.reduction.as_str() {
            "mean" => {
                let n = diff.numel() as f32;
                let mut g = diff.mul(&self.two_scalar);
                g.mul_scalar_(1.0 / n);
                g
            }
            "sum" => diff.mul(&self.two_scalar),
            _ => diff.mul(&self.two_scalar),
        };

        vec![Some(grad.mul(&grad_loss))]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "MSELossBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.pred)
    }
}

