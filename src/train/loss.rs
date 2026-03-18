#![allow(dead_code)]
use crate::tensor::Tensor;

#[allow(dead_code)]
pub trait LossFn {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Tensor;
}

#[allow(dead_code)]
pub struct CrossEntropyLoss {
    pub reduction: String,
}

impl CrossEntropyLoss {
    pub fn new(reduction: &str) -> Self {
        CrossEntropyLoss {
            reduction: reduction.to_string(),
        }
    }
}

impl LossFn for CrossEntropyLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        let pred_data = pred.as_f32_slice();
        let target_data = target.as_f32_slice();

        let batch_size = pred.shape()[0] as usize;
        let num_classes = pred.shape()[1] as usize;

        let mut total_loss = 0.0f32;
        let mut losses = vec![0.0f32; batch_size];

        for (b, target_val) in target_data.iter().take(batch_size).enumerate() {
            let base_idx = b * num_classes;

            let max_logit = pred_data[base_idx..base_idx + num_classes]
                .iter()
                .fold(f32::NEG_INFINITY, |max, &x| if x > max { x } else { max });

            let mut sum_exp = 0.0f32;
            for c in 0..num_classes {
                sum_exp += (pred_data[base_idx + c] - max_logit).exp();
            }
            let log_sum_exp = sum_exp.ln();

            let target_class = *target_val as usize;
            let class_logit = pred_data[base_idx + target_class];

            losses[b] = log_sum_exp - class_logit;
            total_loss += losses[b];
        }

        match self.reduction.as_str() {
            "none" => Tensor::from_vec(losses, vec![batch_size as i64]),
            "mean" => Tensor::from_scalar(total_loss / batch_size as f32),
            "sum" => Tensor::from_scalar(total_loss),
            _ => Tensor::zeros(pred.shape(), pred.dtype(), pred.device()),
        }
    }
}
