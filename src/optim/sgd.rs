use crate::optim::{Optimizer, OptimizerState, ParamGroup, ParamState, zeros_like};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

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
    // Pre-allocated buffers
    pub temp_wd: Vec<Tensor>,
    pub temp_nesterov_grad: Vec<Tensor>,
    pub temp_mom_v: Vec<Tensor>,
    pub temp_vel: Vec<Tensor>,
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
        let temp_wd = zeros_like(&params);
        let temp_nesterov_grad = zeros_like(&params);
        let temp_mom_v = zeros_like(&params);
        let temp_vel = zeros_like(&params);

        SGD {
            params,
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            velocity,
            temp_wd,
            temp_nesterov_grad,
            temp_mom_v,
            temp_vel,
        }
    }
}

impl Optimizer for SGD {
    fn params_mut(&mut self) -> &mut Vec<Tensor> {
        &mut self.params
    }

    #[allow(clippy::needless_range_loop)]
    fn step(&mut self) {
        let lr = self.lr as f32;
        let momentum = self.momentum as f32;
        let weight_decay = self.weight_decay as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let mut grad = if let Some(g) = param.grad() {
                g
            } else {
                continue;
            };

            if weight_decay != 0.0 {
                // grad = grad + weight_decay * param
                self.temp_wd[i] = param.clone();
                self.temp_wd[i].mul_scalar_(weight_decay);
                grad.add_(&self.temp_wd[i]);
            }

            if momentum != 0.0 {
                let velocity = &mut self.velocity[i];

                if self.nesterov {
                    // Nesterov: param -= lr * (grad + momentum * velocity)
                    self.temp_nesterov_grad[i] = grad.clone();
                    self.temp_mom_v[i] = velocity.clone();
                    self.temp_mom_v[i].mul_scalar_(momentum);
                    self.temp_nesterov_grad[i].add_(&self.temp_mom_v[i]);

                    // Now update velocity: velocity = momentum * velocity + grad
                    velocity.mul_scalar_(momentum);
                    velocity.add_(&grad);

                    // param = param - lr * nesterov_grad
                    self.temp_nesterov_grad[i].mul_scalar_(lr);
                    param.sub_(&self.temp_nesterov_grad[i]);
                } else {
                    // Standard SGD: velocity = momentum * velocity + grad
                    velocity.mul_scalar_(momentum);
                    velocity.add_(&grad);

                    // param = param - lr * velocity
                    self.temp_vel[i] = velocity.clone();
                    self.temp_vel[i].mul_scalar_(lr);
                    param.sub_(&self.temp_vel[i]);
                }
            } else {
                // No momentum: simple update
                grad.mul_scalar_(lr);
                param.sub_(&grad);
            }
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let velocity = zeros_like(&params);
        let temp_wd = zeros_like(&params);
        let temp_nesterov_grad = zeros_like(&params);
        let temp_mom_v = zeros_like(&params);
        let temp_vel = zeros_like(&params);

        self.velocity.extend(velocity);
        self.temp_wd.extend(temp_wd);
        self.temp_nesterov_grad.extend(temp_nesterov_grad);
        self.temp_mom_v.extend(temp_mom_v);
        self.temp_vel.extend(temp_vel);
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
