pub mod activations;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod linear;
pub mod norm;
pub mod sequential;

use crate::tensor::Tensor;

pub trait Module: Send + Sync {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn named_parameters(&self) -> Vec<(String, Tensor)>;
    fn zero_grad(&self);
    fn train_mode(&self);
    fn eval_mode(&self);
    fn is_training(&self) -> bool;
}
