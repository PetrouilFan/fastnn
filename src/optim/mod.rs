pub mod adam;
pub mod adamw;
pub mod lion;
pub mod muon;
pub mod sgd;

use crate::tensor::Tensor;
use std::collections::HashMap;

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
