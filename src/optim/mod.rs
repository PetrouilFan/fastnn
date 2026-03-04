pub mod adam;
pub mod adamw;
pub mod sgd;

use crate::tensor::Tensor;

pub trait Optimizer: Send + Sync {
    fn step(&mut self);
    fn zero_grad(&mut self);
    fn add_param_group(&mut self, params: Vec<Tensor>);
    fn state_dict(&self) -> OptimizerState;
    fn load_state_dict(&mut self, state: OptimizerState);
}

#[derive(Debug, Clone)]
pub struct OptimizerState {
    pub param_groups: Vec<ParamGroup>,
}

#[derive(Debug, Clone)]
pub struct ParamGroup {
    pub params: Vec<Tensor>,
}
