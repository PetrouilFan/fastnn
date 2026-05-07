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

use std::sync::atomic::Ordering;

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

/// Helper to clear gradient from a tensor.
pub fn clear_grad(tensor: &Tensor) {
    if let Some(meta) = &tensor.inner.autograd_meta {
        if let Ok(mut lock) = meta.lock() {
            lock.grad = None;
        }
    }
}

/// Helper struct for managing training/eval mode with atomic bool.
pub struct TrainingState(std::sync::atomic::AtomicBool);

impl TrainingState {
    pub fn new() -> Self {
        Self(std::sync::atomic::AtomicBool::new(true))
    }

    pub fn train_mode(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn eval_mode(&self) {
        self.0.store(false, Ordering::Relaxed);
    }

    pub fn is_training(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Macro to implement zero_grad for modules with weight/bias parameters.
#[macro_export]
macro_rules! impl_zero_grad {
    ($self:ident, $weight:expr, $bias:expr) => {
        fn zero_grad(&$self) {
            $crate::nn::clear_grad(&$weight);
            if let Some(b) = &$bias {
                $crate::nn::clear_grad(b);
            }
        }
    };
    ($self:ident, $weight:expr) => {
        fn zero_grad(&$self) {
            $crate::nn::clear_grad(&$weight);
        }
    };
}

/// Macro to implement training state methods.
#[macro_export]
macro_rules! impl_training_state {
    ($self:ident, $field:expr) => {
        fn train_mode(&$self) {
            $field.train_mode();
        }

        fn eval_mode(&$self) {
            $field.eval_mode();
        }

        fn is_training(&$self) -> bool {
            $field.is_training()
        }
    };
}
