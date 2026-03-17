use crate::optim::{Optimizer, OptimizerState, ParamGroup};
use crate::storage::Storage;
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
        for (i, param) in self.params.iter_mut().enumerate() {
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
                // Use in-place operations to avoid allocation
                velocity.mul_(&Tensor::from_scalar(self.momentum as f32));
                velocity.add_(&grad);

                if self.nesterov {
                    let mom = Tensor::from_scalar(self.momentum as f32);
                    grad = grad.add(&velocity.clone().mul(&mom));
                } else {
                    grad = velocity.clone();
                }
            }

            let update = grad.mul(&Tensor::from_scalar(self.lr as f32));

            let inner = Arc::make_mut(&mut param.inner);
            let storage = Arc::make_mut(&mut inner.storage);
            let Storage::Cpu(cpu_storage) = storage else {
                panic!("Optimizer only supports CPU tensors");
            };
            let ptr = cpu_storage.data.as_mut_ptr() as *mut f32;
            let numel = param.numel() as usize;

            let update_slice = update.as_f32_slice();
            for j in 0..numel {
                unsafe {
                    let param_val = *ptr.add(j);
                    *ptr.add(j) = param_val - update_slice[j];
                }
            }
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
