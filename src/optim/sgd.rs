use crate::optim::{Optimizer, OptimizerState, ParamGroup};
use crate::tensor::Tensor;
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
                let mut wd_term = param.clone();
                wd_term.mul_scalar_(weight_decay);
                grad.add_(&wd_term);
            }

            if momentum != 0.0 {
                let velocity = &mut self.velocity[i];
                // velocity = momentum * velocity + grad
                velocity.mul_scalar_(momentum);
                velocity.add_(&grad);

                if self.nesterov {
                    // grad = grad + momentum * velocity
                    let mut mom_v = velocity.clone();
                    mom_v.mul_scalar_(momentum);
                    grad.add_(&mom_v);
                } else {
                    // grad = velocity (move, not clone)
                    grad = velocity.clone();
                }
            }

            // param = param - lr * grad  (in-place)
            grad.mul_scalar_(lr);
            param.sub_(&grad);
        }
    }

    fn zero_grad(&mut self) {
        for param in &mut self.params.iter_mut() {
            let inner = Arc::make_mut(&mut param.inner);
            if let Some(meta) = &mut inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
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
