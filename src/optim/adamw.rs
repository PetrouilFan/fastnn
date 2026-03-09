use crate::optim::{Optimizer, OptimizerState, ParamGroup};
use crate::tensor::Tensor;
use std::sync::Arc;

pub struct AdamW {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    pub m: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub v_hat: Vec<Tensor>,
    pub step: Vec<u64>,
}

impl AdamW {
    pub fn new(
        params: Vec<Tensor>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> Self {
        let m: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        let v: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        let v_hat: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        let step: Vec<u64> = vec![0; params.len()];

        AdamW {
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            m,
            v,
            v_hat,
            step,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        let beta1 = self.betas.0;
        let beta2 = self.betas.1;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = if let Some(g) = param.grad() {
                g
            } else {
                continue;
            };

            self.step[i] += 1;
            let t = self.step[i] as f64;

            let m_update = grad.clone().mul(&Tensor::from_scalar(beta1 as f32));
            self.m[i] = self.m[i]
                .clone()
                .mul(&Tensor::from_scalar(beta1 as f32))
                .add(&m_update);

            let g_sq = grad.clone().mul(&grad.clone());
            let v_update = g_sq.mul(&Tensor::from_scalar(beta2 as f32));
            self.v[i] = self.v[i]
                .clone()
                .mul(&Tensor::from_scalar(beta2 as f32))
                .add(&v_update);

            let bias_correction1 = 1.0 - beta1.powf(t);
            let bias_correction2 = 1.0 - beta2.powf(t);

            let m_hat = self.m[i]
                .clone()
                .mul(&Tensor::from_scalar((1.0 / bias_correction1) as f32));

            let v_hat = if self.amsgrad {
                let max_v = self.v[i].clone().max(0, false);
                self.v_hat[i] = max_v;
                self.v_hat[i]
                    .clone()
                    .mul(&Tensor::from_scalar((1.0 / bias_correction2) as f32))
            } else {
                self.v[i]
                    .clone()
                    .mul(&Tensor::from_scalar((1.0 / bias_correction2) as f32))
            };

            let denom = v_hat.add(&Tensor::from_scalar(self.eps as f32)).sqrt();
            let mut update = m_hat.div(&denom);

            if self.weight_decay != 0.0 {
                let weight_decay_term = param.mul(&Tensor::from_scalar(self.weight_decay as f32));
                update = update.add(&weight_decay_term);
            }

            let lr = Tensor::from_scalar(self.lr as f32);
            let step_size = lr.mul(&update);

            // Apply the update to parameters in-place
            let inner = Arc::make_mut(&mut param.inner);
            let storage = Arc::make_mut(&mut inner.storage);
            let ptr = storage.data.as_mut_ptr() as *mut f32;
            let numel = param.numel() as usize;

            let update_slice = step_size.as_f32_slice();
            for (j, update_val) in update_slice.iter().take(numel).enumerate() {
                unsafe {
                    let param_val = *ptr.add(j);
                    *ptr.add(j) = param_val - update_val;
                }
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &mut self.params.iter_mut() {
            let inner = Arc::make_mut(&mut param.inner);
            if let Some(meta) = &mut inner.autograd_meta {
                meta.grad = None;
            }
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let m: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        let v: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        let v_hat: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        self.m.extend(m);
        self.v.extend(v);
        self.v_hat.extend(v_hat);
        self.step.extend(vec![0u64; params.len()]);
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
