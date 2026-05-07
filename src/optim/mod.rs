pub mod adam;
pub mod adamw;
pub mod lion;
pub mod muon;
pub mod rmsprop;
pub mod sgd;

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
        let indices: Vec<usize> = self.params().iter().enumerate()
            .filter_map(|(i, p)| if p.ndim() == 1 { Some(i) } else { None })
            .collect();
        for i in indices {
            self.no_decay_mut()[i] = true;
        }
    }
}

pub(crate) fn zeros_like(params: &[Tensor]) -> Vec<Tensor> {
    params.iter().map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device())).collect()
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
