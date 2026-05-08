#![allow(clippy::too_many_arguments)]
use crate::nn::{Module, TrainingState};
use crate::tensor::Tensor;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    #[allow(dead_code)]
    pub in_features: i64,
    #[allow(dead_code)]
    pub out_features: i64,
    training: TrainingState,
}

impl Linear {
    pub fn new(in_features: i64, out_features: i64, bias: bool) -> Self {
        let scale = (2.0 / in_features as f32).sqrt();

        // Store weight as [in_features, out_features] for direct matmul
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
            training: TrainingState::new(),
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [batch, in_features], weight: [in_features, out_features]
        // x @ weight = [batch, out_features]
        let output = x.matmul(&self.weight);

        if let Some(b) = &self.bias {
            let bias_broadcast = b.unsqueeze(0);
            output.add(&bias_broadcast)
        } else {
            output
        }
    }

    crate::impl_nn_params!(weight, bias);
    crate::impl_nn_named_params!(weight, bias, "weight", "bias");
    crate::impl_zero_grad!(self, self.weight, self.bias);
    crate::impl_training_state!(self, self.training);
}
