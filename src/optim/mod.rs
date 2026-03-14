pub mod adam;
pub mod adamw;
pub mod sgd;

use crate::tensor::Tensor;

pub trait Optimizer: Send + Sync {
    fn step(&mut self);
    fn zero_grad(&mut self);
    #[allow(dead_code)]
    fn add_param_group(&mut self, params: Vec<Tensor>);
    #[allow(dead_code)]
    fn state_dict(&self) -> OptimizerState;
    #[allow(dead_code)]
    fn load_state_dict(&mut self, state: OptimizerState);
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OptimizerState {
    pub param_groups: Vec<ParamGroup>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ParamGroup {
    pub params: Vec<Tensor>,
}
