use crate::optim::{Optimizer, OptimizerState, ParamGroup, ParamState};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

pub struct Lion {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub betas: (f64, f64),
    pub weight_decay: f64,
    pub m: Vec<Tensor>,
    pub step: Vec<u64>,
    // Pre-allocated buffers
    pub temp_grad_scaled: Vec<Tensor>,
    pub temp_update: Vec<Tensor>,
    pub temp_grad_scaled2: Vec<Tensor>,
}

impl Lion {
    pub fn new(params: Vec<Tensor>, lr: f64, betas: (f64, f64), weight_decay: f64) -> Self {
        let n = params.len();
        let m: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let step: Vec<u64> = vec![0; n];
        let temp_grad_scaled: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_update: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_grad_scaled2: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        Lion {
            params,
            lr,
            betas,
            weight_decay,
            m,
            step,
            temp_grad_scaled,
            temp_update,
            temp_grad_scaled2,
        }
    }
}

impl Optimizer for Lion {
    fn step(&mut self) {
        let beta1 = self.betas.0 as f32;
        let beta2 = self.betas.1 as f32;
        let lr = self.lr as f32;
        let weight_decay = self.weight_decay as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = if let Some(g) = param.grad() {
                g
            } else {
                continue;
            };

            self.step[i] += 1;

            // Weight decay
            if weight_decay != 0.0 {
                param.mul_scalar_(1.0 - lr * weight_decay);
            }

            // Update momentum: m = beta1 * m + (1 - beta1) * grad
            let beta1_c = 1.0 - beta1;
            self.m[i].mul_scalar_(beta1);
            self.temp_grad_scaled[i].copy_from_(grad);
            self.temp_grad_scaled[i].mul_scalar_(beta1_c);
            self.m[i].add_(&self.temp_grad_scaled[i]);

            // Compute sign of (beta2 * m + (1 - beta2) * grad)
            self.temp_update[i].copy_from_(&self.m[i]);
            self.temp_update[i].mul_scalar_(beta2);
            self.temp_grad_scaled2[i].copy_from_(grad);
            self.temp_grad_scaled2[i].mul_scalar_(1.0 - beta2);
            self.temp_update[i].add_(&self.temp_grad_scaled2[i]);

            // sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
            {
                let numel = update.inner.numel() as usize;
                let ptr = update.data_ptr_f32_mut();
                for j in 0..numel {
                    unsafe {
                        let v = *ptr.add(j);
                        *ptr.add(j) = if v > 0.0 {
                            1.0
                        } else if v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                    }
                }
            }

            // param = param - lr * sign(update)
            self.temp_update[i].sign_();
            self.temp_update[i].mul_scalar_(lr);
            param.sub_(&self.temp_update[i]);
        }
    }

    fn zero_grad(&mut self) {
        for param in &mut self.params {
            let inner = Arc::make_mut(&mut param.inner);
            if let Some(meta) = &mut inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let m: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_grad_scaled: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_update: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_grad_scaled2: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        self.m.extend(m);
        self.step.extend(vec![0u64; params.len()]);
        self.temp_grad_scaled.extend(temp_grad_scaled);
        self.temp_update.extend(temp_update);
        self.temp_grad_scaled2.extend(temp_grad_scaled2);
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerState {
        let mut state = HashMap::new();
        for (i, _) in self.params.iter().enumerate() {
            state.insert(
                i,
                ParamState {
                    step: self.step[i],
                    m: Some(self.m[i].clone()),
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
            if i < self.m.len() {
                if let Some(m) = param_state.m {
                    self.m[i] = m;
                }
                if i < self.step.len() {
                    self.step[i] = param_state.step;
                }
            }
        }
    }
}
