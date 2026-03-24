use crate::nn::Module;
use crate::tensor::Tensor;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    #[allow(dead_code)]
    pub in_features: i64,
    #[allow(dead_code)]
    pub out_features: i64,
    training: std::sync::atomic::AtomicBool,
}

impl Linear {
    pub fn new(in_features: i64, out_features: i64, bias: bool) -> Self {
        let scale = (2.0 / in_features as f32).sqrt();

        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| (crate::random_f32() - 0.5) * 2.0 * scale)
            .collect();
        let weight = Tensor::from_vec(weight_data, vec![in_features, out_features]);
        let weight = weight.requires_grad_(true);

        let bias = if bias {
            let bias_data: Vec<f32> = (0..out_features).map(|_| 0.0).collect();
            let b = Tensor::from_vec(bias_data, vec![out_features]);
            let b = b.requires_grad_(true);
            Some(b)
        } else {
            None
        };

        Linear {
            weight,
            bias,
            in_features,
            out_features,
            training: std::sync::atomic::AtomicBool::new(true),
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        let output = x.matmul(&self.weight);

        if let Some(b) = &self.bias {
            // Use unsqueeze to add bias with broadcasting
            // The unsqueeze creates a view with shape [1, out_features]
            // which broadcasts over the batch dimension
            let bias_broadcast = b.unsqueeze(0);
            output.add(&bias_broadcast)
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            params.push(("bias".to_string(), b.clone()));
        }
        params
    }

    fn zero_grad(&self) {
        // Clear gradient on weight
        if let Some(meta) = &self.weight.inner.autograd_meta {
            if let Ok(mut lock) = meta.lock() {
                lock.grad = None;
            }
        }

        // Clear gradient on bias
        if let Some(b) = &self.bias {
            if let Some(meta) = &b.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }

    fn train_mode(&self) {
        self.training
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn eval_mode(&self) {
        self.training
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(std::sync::atomic::Ordering::Relaxed)
    }
}
