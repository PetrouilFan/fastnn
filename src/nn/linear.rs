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
            .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let weight = Tensor::from_vec(weight_data, vec![in_features, out_features]);
        weight.requires_grad_(true);

        let bias = if bias {
            let bias_data: Vec<f32> = (0..out_features).map(|_| 0.0).collect();
            let b = Tensor::from_vec(bias_data, vec![out_features]);
            b.requires_grad_(true);
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
            output.add(b)
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
        let mut meta = self.weight.inner.autograd_meta.clone();
        if let Some(m) = &mut meta {
            m.grad = None;
        }

        // Clear gradient on bias
        if let Some(b) = &self.bias {
            let mut meta = b.inner.autograd_meta.clone();
            if let Some(m) = &mut meta {
                m.grad = None;
            }
        }
    }

    fn train_mode(&self) {
        self.training
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn eval_mode(&self) {
        self.training
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    fn is_training(&self) -> bool {
        self.training.load(std::sync::atomic::Ordering::SeqCst)
    }
}
