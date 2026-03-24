use crate::optim::{Optimizer, OptimizerState, ParamGroup};
use crate::tensor::Tensor;
use std::sync::Arc;

pub struct Adam {
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

impl Adam {
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

        Adam {
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

impl Optimizer for Adam {
    fn step(&mut self) {
        let beta1 = self.betas.0 as f32;
        let beta2 = self.betas.1 as f32;
        let lr = self.lr as f32;
        let eps = self.eps as f32;
        let weight_decay = self.weight_decay as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = if let Some(g) = param.grad() {
                g
            } else {
                continue;
            };

            self.step[i] += 1;
            let t = self.step[i] as f32;

            // m = beta1 * m + (1 - beta1) * grad
            // Equivalent to: m = beta1 * m + grad * (1 - beta1)
            // Rewrite as: m = beta1 * m + grad * beta1_complement
            // Where beta1_complement = 1 - beta1
            let beta1_c = 1.0 - beta1;
            self.m[i].mul_scalar_(beta1);
            // Create temp for (1-beta1)*grad to avoid clobbering grad
            let mut grad_scaled = grad.clone();
            grad_scaled.mul_scalar_(beta1_c);
            self.m[i].add_(&grad_scaled);

            // v = beta2 * v + (1 - beta2) * grad^2
            let beta2_c = 1.0 - beta2;
            self.v[i].mul_scalar_(beta2);
            let mut grad_sq = grad.clone();
            // grad_sq = grad * grad (element-wise square)
            {
                let numel = grad_sq.inner.numel() as usize;
                let ptr = grad_sq.data_ptr_f32_mut();
                for j in 0..numel {
                    unsafe {
                        let val = *ptr.add(j);
                        *ptr.add(j) = val * val;
                    }
                }
            }
            grad_sq.mul_scalar_(beta2_c);
            self.v[i].add_(&grad_sq);

            let bias_correction1 = 1.0 - beta1.powf(t);
            let bias_correction2 = 1.0 - beta2.powf(t);

            // m_hat = m / bias_correction1
            let mut m_hat = self.m[i].clone();
            m_hat.mul_scalar_(1.0 / bias_correction1);

            // v_hat = v / bias_correction2 (with optional amsgrad)
            let mut v_hat = if self.amsgrad {
                self.v_hat[i] = self.v[i].clone();
                self.v_hat[i].clone()
            } else {
                self.v[i].clone()
            };
            v_hat.mul_scalar_(1.0 / bias_correction2);

            // update = m_hat / (sqrt(v_hat) + eps)
            // Compute sqrt in-place then add eps
            {
                let numel = v_hat.inner.numel() as usize;
                let ptr = v_hat.data_ptr_f32_mut();
                for j in 0..numel {
                    unsafe {
                        *ptr.add(j) = (*ptr.add(j)).sqrt() + eps;
                    }
                }
            }
            // update = m_hat / v_hat (where v_hat is now sqrt(v_hat) + eps)
            m_hat.div_(&v_hat);

            // param = param - lr * update
            m_hat.mul_scalar_(lr);
            param.sub_(&m_hat);

            // Weight decay: param = param - lr * weight_decay * param
            if weight_decay != 0.0 {
                param.mul_scalar_(1.0 - lr * weight_decay);
            }
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
