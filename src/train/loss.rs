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
        let batch_size = pred.shape()[0];
        let num_classes = pred.shape()[1];

        let log_probs = pred.log_softmax(1);

        let indices = target.clone();
        let indices_long = if indices.dtype() == crate::storage::DType::I64 {
            indices
        } else {
            indices.to_dtype(crate::storage::DType::I64)
        };

        let loss_per_sample = log_probs.mul(&indices_long.neg()).sum(1, false);

        match self.reduction.as_str() {
            "none" => loss_per_sample,
            "mean" => loss_per_sample.mean(0, false),
            "sum" => loss_per_sample.sum(0, false),
            _ => Tensor::zeros(vec![batch_size], crate::storage::DType::F32, pred.device()),
        }
    }
}
