use crate::optim::{zeros_like, Optimizer, WeightDecayOptimizer};
use crate::tensor::Tensor;
use crate::{get_grad_or_skip, impl_optim_boilerplate, impl_params_mut, impl_weight_decay};

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
    // Track which parameters should skip weight decay (e.g., biases, LayerNorm)
    pub no_decay: Vec<bool>,
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
        let m = zeros_like(&params);
        let v = zeros_like(&params);
        let v_hat = zeros_like(&params);
        let no_decay = vec![false; params.len()];

        AdamW {
            step: vec![0u64; params.len()],
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            m,
            v,
            v_hat,
            no_decay,
        }
    }
}

impl WeightDecayOptimizer for AdamW {
    impl_weight_decay!();
}

impl Optimizer for AdamW {
    impl_params_mut!();

    fn step(&mut self) {
        let beta1 = self.betas.0 as f32;
        let beta2 = self.betas.1 as f32;
        let lr = self.lr as f32;
        let eps = self.eps as f32;
        let weight_decay = self.weight_decay as f32;

        for i in 0..self.params.len() {
            let grad = get_grad_or_skip!(&self.params[i]);

            self.step[i] += 1;
            let t = self.step[i];

            let no_decay = self.no_decay.get(i).copied().unwrap_or(false);
            let wd = if weight_decay != 0.0 && !no_decay { weight_decay } else { 0.0 };

            let mut results = self.params[i].adamw_update(
                &grad, &self.m[i], &self.v[i],
                lr, beta1, beta2, eps, t, wd,
            );
            self.v[i] = results.pop().unwrap();
            self.m[i] = results.pop().unwrap();
            self.params[i] = results.pop().unwrap();

            self.params[i].set_grad(None);
        }
    }

    impl_optim_boilerplate!(true, m, v, v_hat);
}
