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
        match self.reduction.as_str() {
            "none" => Tensor::zeros(pred.shape(), pred.dtype(), pred.device()),
            "mean" | "sum" => {
                let pred_data = pred.to_numpy();
                let target_data = target.to_numpy();

                let batch_size = pred.shape()[0] as usize;
                let num_classes = pred.shape()[1] as usize;

                let mut total_loss = 0.0f32;

                for b in 0..batch_size {
                    let max_logit = (0..num_classes)
                        .map(|c| pred_data[b * num_classes + c])
                        .fold(f32::MIN, f32::max);

                    let mut sum_exp = 0.0f32;
                    for c in 0..num_classes {
                        sum_exp += (pred_data[b * num_classes + c] - max_logit).exp();
                    }
                    let log_sum_exp = sum_exp.ln();

                    let target_class = target_data[b] as usize;
                    let class_logit = pred_data[b * num_classes + target_class];

                    total_loss += log_sum_exp - class_logit;
                }

                let loss = if self.reduction == "mean" {
                    total_loss / batch_size as f32
                } else {
                    total_loss
                };
                Tensor::from_scalar(loss)
            }
            _ => Tensor::zeros(pred.shape(), pred.dtype(), pred.device()),
        }
    }
}
