pub mod activations;
pub mod attention;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod fused;
pub mod linear;
pub mod norm;
pub mod pooling;
pub mod sequential;
pub mod transformer;
pub mod upsample;

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
