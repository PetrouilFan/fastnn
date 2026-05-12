use crate::autograd::{
    self, AutogradMeta, BatchNorm1dBackward, BatchNorm2dBackward, GroupNormBackward,
    RMSNormBackward,
};
use crate::dispatcher::{DispatchKey, dispatch};
use crate::storage::Device;
use crate::tensor::Tensor;
use crate::{
    impl_training_state,
    nn::{clear_grad, Module, TrainingState},
};
use parking_lot::RwLock;
use std::sync::Arc;

pub struct LayerNorm {
    #[allow(dead_code)]
    pub normalized_shape: i64,
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
    training: TrainingState,
    eps_scalar: Tensor,
}

impl LayerNorm {
    pub fn new(normalized_shape: i64, eps: f64) -> Self {
        let weight_data: Vec<f32> = (0..normalized_shape).map(|_| 1.0).collect();
        let weight = Tensor::from_vec(weight_data, vec![normalized_shape]).requires_grad_(true);

        let bias_data: Vec<f32> = (0..normalized_shape).map(|_| 0.0).collect();
        let bias = Tensor::from_vec(bias_data, vec![normalized_shape]).requires_grad_(true);

        LayerNorm {
            weight,
            bias,
            normalized_shape,
            eps,
            training: TrainingState::new(),
            eps_scalar: Tensor::from_scalar(eps as f32),
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape_ref();
        let _ndim = shape.len();

        // Use references to avoid cloning weight/bias on every forward pass
        let weight = &self.weight;
        let bias = &self.bias;

        let output = if x.device() == Device::Cpu {
            Tensor::exec_aot(&[x, weight, bias], |g, ins| {
                vec![g.layer_norm(&ins[0], &ins[1], &ins[2], self.eps)]
            })
            .expect("LayerNorm::forward: AOT failed")
            .into_iter()
            .next()
            .unwrap()
        } else {
            let result = dispatch(
                "layer_norm",
                DispatchKey::Cpu,
                &[x, x, weight, bias, &self.eps_scalar],
            )
            .expect("LayerNorm::forward: dispatch failed");
            result[0].clone()
        };

        // Set up gradient tracking for layer norm
        if x.requires_grad() || weight.requires_grad() || bias.requires_grad() {
            let edges = {
                let mut edges = crate::autograd::make_edge(x);
                edges.extend(crate::autograd::make_edge(weight));
                edges.extend(crate::autograd::make_edge(bias));
                edges
            };
            let backward = crate::autograd::LayerNormBackward::new();
            let mut meta = crate::autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(std::sync::Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("weight".to_string(), self.weight.clone()),
            ("bias".to_string(), self.bias.clone()),
        ]
    }

    fn zero_grad(&self) {
        clear_grad(&self.weight);
        clear_grad(&self.bias);
    }

    impl_training_state!(self, self.training);
}

pub struct BatchNorm1d {
    #[allow(dead_code)]
    pub num_features: i64,
    #[allow(dead_code)]
    pub eps: f64,
    #[allow(dead_code)]
    pub momentum: f64,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub running_mean: Arc<RwLock<Tensor>>,
    pub running_var: Arc<RwLock<Tensor>>,
    training: TrainingState,
    #[allow(dead_code)]
    pub track_running_stats: bool,
    // Pre-allocated scalar tensors
    eps_scalar: Tensor,
    training_true_scalar: Tensor,
    training_false_scalar: Tensor,
}

impl BatchNorm1d {
    pub fn new(num_features: i64, eps: f64, momentum: f64) -> Self {
        let weight_data: Vec<f32> = (0..num_features).map(|_| 1.0).collect();
        let weight = Tensor::from_vec(weight_data, vec![num_features]).requires_grad_(true);

        let bias_data: Vec<f32> = (0..num_features).map(|_| 0.0).collect();
        let bias = Tensor::from_vec(bias_data, vec![num_features]).requires_grad_(true);

        let running_mean_data: Vec<f32> = (0..num_features).map(|_| 0.0).collect();
        let running_mean = Tensor::from_vec(running_mean_data, vec![num_features]);

        let running_var_data: Vec<f32> = (0..num_features).map(|_| 1.0).collect();
        let running_var = Tensor::from_vec(running_var_data, vec![num_features]);

        BatchNorm1d {
            weight: Some(weight),
            bias: Some(bias),
            num_features,
            eps,
            momentum,
            running_mean: Arc::new(RwLock::new(running_mean)),
            running_var: Arc::new(RwLock::new(running_var)),
            training: TrainingState::new(),
            track_running_stats: true,
            eps_scalar: Tensor::from_scalar(eps as f32),
            training_true_scalar: Tensor::from_scalar(1.0),
            training_false_scalar: Tensor::from_scalar(0.0),
        }
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Use references instead of cloning weight/bias
        let default_weight;
        let default_bias;
        let weight_ref = match &self.weight {
            Some(w) => w,
            None => {
                default_weight = Tensor::from_scalar(1.0);
                &default_weight
            }
        };
        let bias_ref = match &self.bias {
            Some(b) => b,
            None => {
                default_bias = Tensor::from_scalar(0.0);
                &default_bias
            }
        };

        let is_training = self.training.is_training();

        let training_flag = if is_training {
            &self.training_true_scalar
        } else {
            &self.training_false_scalar
        };

        // Read current running stats - clone tensors so guards are dropped before dispatch
        let running_mean = self.running_mean.read().clone();
        let running_var = self.running_var.read().clone();

        let grads_needed = x.requires_grad()
            || self.weight.as_ref().is_some_and(|w| w.requires_grad())
            || self.bias.as_ref().is_some_and(|b| b.requires_grad());

        // Compute batch stats for backward (before running stats update modifies them)
        let (batch_mean, batch_var) = if is_training {
            let x_shape = x.shape_ref();
            let batch_size = x_shape[0];
            let num_features = x_shape[1];
            let spatial_size: i64 = if x_shape.len() > 2 {
                x_shape[2..].iter().product()
            } else {
                1
            };

            let x_reshaped = x.reshape(vec![batch_size, num_features, spatial_size]);
            let b_mean = x_reshaped.mean(2, false).mean(0, false);
            let centered = x_reshaped.sub(&b_mean.reshape(vec![1, num_features, 1]));
            let b_var = centered.mul(&centered).mean(2, false).mean(0, false);
            (b_mean, b_var)
        } else {
            (running_mean.clone(), running_var.clone())
        };

        // TODO(migration): Replace dispatch with AOT when batch_norm graph op supports training mode
        let result = dispatch(
            "batch_norm",
            DispatchKey::Cpu,
            &[
                x,
                weight_ref,
                bias_ref,
                &running_mean,
                &running_var,
                training_flag,
                &self.eps_scalar,
            ],
        )
        .expect("BatchNorm1d::forward: dispatch failed");

        let mut output = result[0].clone();

        // In training mode, update the running stats
        if is_training {
            let x_shape = x.shape_ref();
            let batch_size = x_shape[0];
            let _num_features = x_shape[1];
            let spatial_size: i64 = if x_shape.len() > 2 {
                x_shape[2..].iter().product()
            } else {
                1
            };

            let n = (batch_size * spatial_size) as f32;
            let unbiased_var = if n > 1.0 {
                batch_var.clone().mul_scalar(n / (n - 1.0))
            } else {
                batch_var.clone()
            };

            let mom = self.momentum as f32;
            let inv_mom = 1.0 - mom;

            // Get mutable references and update
            let mut running_mean_lock = self.running_mean.write();
            let new_mean = running_mean_lock
                .mul_scalar(inv_mom)
                .add(&batch_mean.mul_scalar(mom));
            *running_mean_lock = new_mean;

            let mut running_var_lock = self.running_var.write();
            let new_var = running_var_lock
                .mul_scalar(inv_mom)
                .add(&unbiased_var.mul_scalar(mom));
            *running_var_lock = new_var;
        }

        // Attach autograd
        if grads_needed {
            let edges = {
                let mut edges = autograd::make_edge(x);
                if let Some(w) = &self.weight {
                    edges.extend(autograd::make_edge(w));
                }
                if let Some(b) = &self.bias {
                    edges.extend(autograd::make_edge(b));
                }
                edges
            };
            let backward = BatchNorm1dBackward::new();
            let mut meta = AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(Arc::new(backward));
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
        }

        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];
        if let Some(w) = &self.weight {
            params.push(w.clone());
        }
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = vec![];
        if let Some(w) = &self.weight {
            params.push(("weight".to_string(), w.clone()));
        }
        if let Some(b) = &self.bias {
            params.push(("bias".to_string(), b.clone()));
        }
        params
    }

    fn zero_grad(&self) {
        if let Some(w) = &self.weight {
            clear_grad(w);
        }
        if let Some(b) = &self.bias {
            clear_grad(b);
        }
    }

    impl_training_state!(self, self.training);
}

pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
    pub normalized_shape: i64,
    eps_scalar: Tensor,
}

impl RMSNorm {
    pub fn new(normalized_shape: i64, eps: f32) -> Self {
        let weight = Tensor::ones(
            vec![normalized_shape],
            crate::storage::DType::F32,
            crate::storage::Device::Cpu,
        );
        let w = weight.clone();
        let w = w.requires_grad_(true);
        RMSNorm {
            weight: w,
            eps,
            normalized_shape,
            eps_scalar: Tensor::from_scalar(eps),
        }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let result = if x.device() == Device::Cpu {
            Tensor::exec_aot(&[x, &self.weight], |g, ins| {
                vec![g.rms_norm(&ins[0], &ins[1], self.eps as f64)]
            })
            .expect("RMSNorm::forward: AOT failed")
        } else {
            let dispatch_key = crate::dispatcher::device_to_dispatch_key(x.device());
            dispatch(
                "rms_norm",
                dispatch_key,
                &[x, &self.weight, &self.eps_scalar],
            )
            .expect("RMSNorm::forward: dispatch failed")
        };
        let mut output = result.into_iter().next().unwrap();

        if x.requires_grad() || self.weight.requires_grad() {
            let edges = {
                let mut edges = autograd::make_edge(x);
                edges.extend(autograd::make_edge(&self.weight));
                edges
            };
            let backward = RMSNormBackward::new();
            let mut meta = AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(Arc::new(backward));
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
        }

        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }

    fn zero_grad(&self) {
        if let Some(meta) = &self.weight.inner.autograd_meta {
            if let Ok(mut lock) = meta.lock() {
                lock.grad = None;
            }
        }
    }

    fn train_mode(&self) {}

    fn eval_mode(&self) {}

    fn is_training(&self) -> bool {
        false
    }
}

pub struct GroupNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f32,
    pub num_groups: i64,
    pub num_channels: i64,
}

impl GroupNorm {
    pub fn new(num_groups: i64, num_channels: i64, eps: f32) -> Self {
        assert!(
            num_channels % num_groups == 0,
            "num_channels must be divisible by num_groups"
        );
        let weight = Tensor::ones(
            vec![num_channels],
            crate::storage::DType::F32,
            crate::storage::Device::Cpu,
        );
        let bias = Tensor::zeros(
            vec![num_channels],
            crate::storage::DType::F32,
            crate::storage::Device::Cpu,
        );
        let w = weight.clone();
        let b = bias.clone();
        let w = w.requires_grad_(true);
        let b = b.requires_grad_(true);
        GroupNorm {
            weight: w,
            bias: b,
            eps,
            num_groups,
            num_channels,
        }
    }
}

impl Module for GroupNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_shape = x.shape_ref();
        let batch = x_shape[0];
        let channels = x_shape[1];
        let spatial: i64 = x_shape[2..].iter().product();
        let group_size = channels / self.num_groups;

        // Compute forward in no_grad to avoid per-op graph, then attach fused backward
        use crate::autograd::NoGradGuard;
        let mut output = {
            let _guard = NoGradGuard::new();
            let x_reshaped = x.reshape(vec![batch, self.num_groups, group_size, spatial]);
            let mean = x_reshaped.mean(2, true).mean(3, true);
            let var = x_reshaped.sub(&mean).pow(2.0).mean(2, true).mean(3, true);
            let x_norm = x_reshaped.sub(&mean).div(&var.add_scalar(self.eps).sqrt());
            let x_norm = x_norm.reshape(x_shape.to_vec());

            let mut weight_shape: smallvec::SmallVec<[i64; 8]> = smallvec::SmallVec::new();
            weight_shape.push(1);
            weight_shape.push(self.num_channels);
            for _ in 2..x_shape.len() {
                weight_shape.push(1);
            }
            let w = self.weight.reshape(weight_shape.clone().into_vec());
            let b = self.bias.reshape(weight_shape.into_vec());
            x_norm.mul(&w).add(&b)
        };

        if x.requires_grad() || self.weight.requires_grad() || self.bias.requires_grad() {
            let edges = {
                let mut edges = autograd::make_edge(x);
                edges.extend(autograd::make_edge(&self.weight));
                edges.extend(autograd::make_edge(&self.bias));
                edges
            };
            let backward = GroupNormBackward::new();
            let mut meta = AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(Arc::new(backward));
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
        }

        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("weight".to_string(), self.weight.clone()),
            ("bias".to_string(), self.bias.clone()),
        ]
    }

    fn zero_grad(&self) {
        for t in [&self.weight, &self.bias] {
            if let Some(meta) = &t.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }

    fn train_mode(&self) {}

    fn eval_mode(&self) {}

    fn is_training(&self) -> bool {
        false
    }
}

pub struct BatchNorm2d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: parking_lot::RwLock<Tensor>,
    pub running_var: parking_lot::RwLock<Tensor>,
    pub eps: f32,
    pub momentum: f32,
    pub num_features: i64,
    training: TrainingState,
    eps_scalar: Tensor,
    training_true_scalar: Tensor,
    training_false_scalar: Tensor,
}

impl BatchNorm2d {
    pub fn new(num_features: i64, eps: f32, momentum: f32) -> Self {
        let weight = Tensor::ones(
            vec![num_features],
            crate::storage::DType::F32,
            crate::storage::Device::Cpu,
        );
        let bias = Tensor::zeros(
            vec![num_features],
            crate::storage::DType::F32,
            crate::storage::Device::Cpu,
        );
        let running_mean = Tensor::zeros(
            vec![num_features],
            crate::storage::DType::F32,
            crate::storage::Device::Cpu,
        );
        let running_var = Tensor::ones(
            vec![num_features],
            crate::storage::DType::F32,
            crate::storage::Device::Cpu,
        );
        let w = weight.clone();
        let b = bias.clone();
        let w = w.requires_grad_(true);
        let b = b.requires_grad_(true);
        BatchNorm2d {
            weight: w,
            bias: b,
            running_mean: parking_lot::RwLock::new(running_mean),
            running_var: parking_lot::RwLock::new(running_var),
            eps,
            momentum,
            num_features,
            training: TrainingState::new(),
            eps_scalar: Tensor::from_scalar(eps),
            training_true_scalar: Tensor::from_scalar(1.0),
            training_false_scalar: Tensor::from_scalar(0.0),
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let is_training = self.training.is_training();

        let training_flag = if is_training {
            &self.training_true_scalar
        } else {
            &self.training_false_scalar
        };

        // Read current running stats - clone tensors so guards are dropped before dispatch
        let running_mean = self.running_mean.read().clone();
        let running_var = self.running_var.read().clone();

        // Compute batch stats (used for backward in training mode, or empty tensors as placeholders)
        let (batch_mean, batch_var) =
            if is_training && (x.requires_grad() || self.weight.requires_grad()) {
                let x_shape = x.shape_ref();
                let batch = x_shape[0];
                let channels = x_shape[1];
                let spatial: i64 = x_shape[2..].iter().product();
                let x_reshaped = x.reshape(vec![batch, channels, spatial]);
                let b_mean = x_reshaped.mean(2, false).mean(0, false);
                let centered = x_reshaped.sub(&b_mean.reshape(vec![1, channels, 1]));
                let b_var = centered.mul(&centered).mean(2, false).mean(0, false);
                (b_mean, b_var)
            } else {
                (running_mean.clone(), running_var.clone())
            };

        let dispatch_key = crate::dispatcher::device_to_dispatch_key(x.device());
        // TODO(migration): Replace dispatch with AOT when batch_norm graph op supports training mode
        let result = crate::dispatcher::dispatch(
            "batch_norm",
            dispatch_key,
            &[
                x,
                &self.weight,
                &self.bias,
                &running_mean,
                &running_var,
                training_flag,
                &self.eps_scalar,
            ],
        )
        .expect("BatchNorm2d::forward: dispatch failed");

        let mut output = result[0].clone();

        // In training mode, update the running stats
        if is_training {
            let x_shape = x.shape_ref();
            let batch = x_shape[0];
            let _channels = x_shape[1];
            let spatial: i64 = x_shape[2..].iter().product();

            let mom = self.momentum;
            let inv_mom = 1.0 - mom;

            let mut running_mean_lock = self.running_mean.write();
            let new_mean = running_mean_lock
                .mul_scalar(inv_mom)
                .add(&batch_mean.mul_scalar(mom));
            *running_mean_lock = new_mean;

            let mut running_var_lock = self.running_var.write();
            let n = (batch * spatial) as f32;
            let unbiased_var = if n > 1.0 {
                batch_var.clone().mul_scalar(n / (n - 1.0))
            } else {
                batch_var.clone()
            };
            let new_var = running_var_lock
                .mul_scalar(inv_mom)
                .add(&unbiased_var.mul_scalar(mom));
            *running_var_lock = new_var;
        }

        // Attach autograd
        if x.requires_grad() || self.weight.requires_grad() || self.bias.requires_grad() {
            let edges = {
                let mut edges = autograd::make_edge(x);
                edges.extend(autograd::make_edge(&self.weight));
                edges.extend(autograd::make_edge(&self.bias));
                edges
            };
            let backward = BatchNorm2dBackward::new();
            let mut meta = AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(Arc::new(backward));
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
        }

        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("weight".to_string(), self.weight.clone()),
            ("bias".to_string(), self.bias.clone()),
        ]
    }

    fn zero_grad(&self) {
        clear_grad(&self.weight);
        clear_grad(&self.bias);
    }

    impl_training_state!(self, self.training);
}
