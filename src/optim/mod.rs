pub mod adam;
pub mod adamw;
pub mod lion;
pub mod muon;
pub mod rmsprop;
pub mod sgd;

pub use adam::Adam;
pub use adamw::AdamW;
pub use lion::Lion;
pub use muon::Muon;
pub use rmsprop::RMSprop;
pub use sgd::SGD;

use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

pub trait Optimizer: Send + Sync {
    fn step(&mut self);
    fn params_mut(&mut self) -> &mut Vec<Tensor>;
    fn zero_grad(&mut self) {
        for param in self.params_mut() {
            let inner = Arc::make_mut(&mut param.inner);
            if let Some(meta) = &mut inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }
    #[allow(dead_code)]
    fn add_param_group(&mut self, params: Vec<Tensor>);
    #[allow(dead_code)]
    fn state_dict(&self) -> OptimizerState;
    #[allow(dead_code)]
    fn load_state_dict(&mut self, state: OptimizerState);
}

/// Macro to implement duplicate `params_mut()` for all optimizers
#[macro_export]
macro_rules! impl_params_mut {
    () => {
        fn params_mut(&mut self) -> &mut Vec<Tensor> {
            &mut self.params
        }
    };
}

/// Weight decay strategy enum to standardize inconsistent implementations
#[derive(Debug, Clone, Copy)]
pub enum WeightDecayType {
    L2,        // Add weight_decay * param to gradient (SGD, Muon)
    Decoupled, // Scale param directly: param *= (1 - lr * weight_decay) (Adam, AdamW, etc.)
    None,
}

/// Apply weight decay consistently across optimizers
pub(crate) fn apply_weight_decay(
    param: &mut Tensor,
    grad: &Tensor,
    weight_decay: f32,
    lr: f32,
    wd_type: WeightDecayType,
) -> Tensor {
    match wd_type {
        WeightDecayType::L2 => {
            let mut g = grad.clone();
            g.add_(&param.mul_scalar(weight_decay));
            g
        }
        WeightDecayType::Decoupled => {
            param.mul_scalar_(1.0 - lr * weight_decay);
            grad.clone()
        }
        WeightDecayType::None => grad.clone(),
    }
}

/// Helper to retrieve gradient from a parameter, replacing duplicate retrieval patterns
pub(crate) fn get_grad(param: &Tensor) -> Option<Tensor> {
    let inner = &param.inner;
    if let Some(meta) = &inner.autograd_meta {
        if let Ok(lock) = meta.lock() {
            return lock.grad.clone();
        }
    }
    None
}

/// Macro to replace the duplicate gradient retrieval pattern in optimizer step() methods
#[macro_export]
macro_rules! get_grad_or_skip {
    ($param:expr) => {
        match $crate::optim::get_grad($param) {
            Some(g) => g,
            None => continue,
        }
    };
}

pub trait WeightDecayOptimizer {
    fn params(&self) -> &Vec<Tensor>;
    fn no_decay(&self) -> &Vec<bool>;
    fn no_decay_mut(&mut self) -> &mut Vec<bool>;

    fn add_no_decay(&mut self, indices: &[usize]) {
        for &idx in indices {
            if idx < self.no_decay().len() {
                self.no_decay_mut()[idx] = true;
            }
        }
    }

    fn mark_biases_no_decay(&mut self) {
        let indices: Vec<usize> = self
            .params()
            .iter()
            .enumerate()
            .filter_map(|(i, p)| if p.ndim() == 1 { Some(i) } else { None })
            .collect();
        for i in indices {
            self.no_decay_mut()[i] = true;
        }
    }
}

pub(crate) fn zeros_like(params: &[Tensor]) -> Vec<Tensor> {
    params
        .iter()
        .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
        .collect()
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OptimizerState {
    pub param_groups: Vec<ParamGroup>,
    pub state: HashMap<usize, ParamState>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ParamState {
    pub step: u64,
    pub m: Option<Tensor>,
    pub v: Option<Tensor>,
    pub v_hat: Option<Tensor>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ParamGroup {
    pub params: Vec<Tensor>,
}
