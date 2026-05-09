use crate::optim::{
    apply_weight_decay, zeros_like, Optimizer, OptimizerState, ParamGroup, ParamState,
    WeightDecayOptimizer, WeightDecayType,
};
use crate::tensor::Tensor;
use crate::{get_grad_or_skip, impl_params_mut};
use std::collections::HashMap;

#[allow(clippy::upper_case_acronyms)]
pub struct SGD {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub momentum: f64,
    #[allow(dead_code)]
    pub dampening: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
    pub velocity: Vec<Tensor>,
    pub no_decay: Vec<bool>,
}

impl SGD {
    pub fn new(
        params: Vec<Tensor>,
        lr: f64,
        momentum: f64,
        dampening: f64,
        weight_decay: f64,
        nesterov: bool,
    ) -> Self {
        let velocity = zeros_like(&params);
        let no_decay = vec![false; params.len()];

        SGD {
            params,
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            velocity,
            no_decay,
        }
    }
}

impl WeightDecayOptimizer for SGD {
    fn params(&self) -> &Vec<Tensor> {
        &self.params
    }
    fn no_decay(&self) -> &Vec<bool> {
        &self.no_decay
    }
    fn no_decay_mut(&mut self) -> &mut Vec<bool> {
        &mut self.no_decay
    }
}

impl Optimizer for SGD {
    impl_params_mut!();

    #[allow(clippy::needless_range_loop)]
    fn step(&mut self) {
        let lr = self.lr as f32;
        let momentum = self.momentum as f32;
        let weight_decay = self.weight_decay as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = get_grad_or_skip!(param);

            // Apply L2 weight decay (consistent with WeightDecayType::L2)
            let grad = if weight_decay != 0.0 {
                apply_weight_decay(param, &grad, weight_decay, lr, WeightDecayType::L2)
            } else {
                grad
            };

            if momentum != 0.0 {
                let velocity = &mut self.velocity[i];

                if self.nesterov {
                    // Nesterov: param -= lr * (grad + momentum * velocity)
                    let nesterov_grad = grad.clone().add(&velocity.mul_scalar(momentum));
                    velocity.mul_scalar_(momentum).add_(&grad);
                    param.sub_(&nesterov_grad.mul_scalar(lr));
                } else {
                    // Standard SGD: velocity = momentum * velocity + grad
                    velocity.mul_scalar_(momentum).add_(&grad);
                    param.sub_(&velocity.clone().mul_scalar(lr));
                }
            } else {
                // No momentum: simple update
                param.sub_(&grad.mul_scalar(lr));
            }
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let velocity = zeros_like(&params);
        self.velocity.extend(velocity);
        self.no_decay.extend(vec![false; params.len()]);
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerState {
        let mut state = HashMap::new();
        for (i, _) in self.params.iter().enumerate() {
            state.insert(
                i,
                ParamState {
                    step: 0,
                    m: Some(self.velocity[i].clone()),
                    v: None,
                    v_hat: None,
                },
            );
        }
        OptimizerState {
            param_groups: vec![ParamGroup {
                params: self.params.clone(),
            }],
            state,
        }
    }

    fn load_state_dict(&mut self, state: OptimizerState) {
        if let Some(group) = state.param_groups.first() {
            self.params = group.params.clone();
        }
        for (i, param_state) in state.state {
            if i < self.velocity.len() {
                if let Some(m) = param_state.m {
                    self.velocity[i] = m;
                }
            }
        }
    }
}
