use crate::optim::{Optimizer, OptimizerState, ParamGroup};
use crate::tensor::Tensor;
use std::sync::Arc;

pub struct SGD {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
    pub velocity: Vec<Tensor>,
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
        let velocity: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        SGD {
            params,
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            velocity,
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let mut grad = if let Some(g) = param.grad() {
                g
            } else {
                continue;
            };

            if self.weight_decay != 0.0 {
                let weight_decay_term = param.mul(&Tensor::from_scalar(self.weight_decay as f32));
                grad = grad.add(&weight_decay_term);
            }

            if self.momentum != 0.0 {
                let velocity = &mut self.velocity[i];
                *velocity = velocity
                    .clone()
                    .mul(&Tensor::from_scalar(self.momentum as f32))
                    .add(&grad.clone());

                if self.nesterov {
                    grad = grad.add(
                        &velocity
                            .clone()
                            .mul(&Tensor::from_scalar(self.momentum as f32)),
                    );
                } else {
                    grad = velocity.clone();
                }
            }

            let _update = grad.mul(&Tensor::from_scalar(self.lr as f32));
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            let mut inner = param.inner.clone();
            Arc::make_mut(&mut inner).autograd_meta = None;
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let velocity: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        self.velocity.extend(velocity);
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerState {
        OptimizerState {
            param_groups: vec![ParamGroup {
                params: self.params.clone(),
            }],
        }
    }

    fn load_state_dict(&mut self, state: OptimizerState) {
        if let Some(group) = state.param_groups.first() {
            self.params = group.params.clone();
        }
    }
}
