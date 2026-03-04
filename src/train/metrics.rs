use crate::tensor::Tensor;

pub trait Metric: Send + Sync {
    fn update(&mut self, pred: &Tensor, target: &Tensor);
    fn compute(&self) -> f64;
    fn reset(&mut self);
}

pub struct Accuracy;

impl Metric for Accuracy {
    fn update(&mut self, pred: &Tensor, target: &Tensor) {
        // Simplified - would calculate accuracy
    }

    fn compute(&self) -> f64 {
        0.0
    }

    fn reset(&mut self) {}
}
