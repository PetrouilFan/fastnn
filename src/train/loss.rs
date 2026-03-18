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

            // Check for NaN in target_val
            if target_val.is_nan() {
                losses[b] = 0.0; // or some default value
                continue;
            }

            let target_class = *target_val as usize;
            if target_class >= num_classes {
                // Target class is out of bounds
                losses[b] = 0.0;
                continue;
            }

            let max_logit = pred_data[base_idx..base_idx + num_classes]
                .iter()
                .fold(f32::NEG_INFINITY, |max, &x| if x > max { x } else { max });

            // Check if max_logit is NaN or infinity
            if max_logit.is_nan() || max_logit.is_infinite() {
                losses[b] = 0.0;
                continue;
            }

            let mut sum_exp = 0.0f32;
            for c in 0..num_classes {
                let diff = pred_data[base_idx + c] - max_logit;
                // Check for NaN or infinity in diff
                if diff.is_nan() || diff.is_infinite() {
                    sum_exp = f32::INFINITY;
                    break;
                }
                sum_exp += diff.exp();
            }

            // Check if sum_exp is NaN, infinity, or zero
            if sum_exp.is_nan() || sum_exp.is_infinite() || sum_exp == 0.0 {
                losses[b] = 0.0;
                continue;
            }

            let log_sum_exp = sum_exp.ln();
            let class_logit = pred_data[base_idx + target_class];

            // Check for NaN in class_logit
            if class_logit.is_nan() {
                losses[b] = 0.0;
                continue;
            }

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
