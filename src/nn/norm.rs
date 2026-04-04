use crate::dispatcher::{dispatch, DispatchKey};
use crate::nn::Module;
use crate::tensor::Tensor;
use parking_lot::RwLock;
use std::sync::Arc;

pub struct LayerNorm {
    #[allow(dead_code)]
    pub normalized_shape: i64,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub eps: f64,
    training: std::sync::atomic::AtomicBool,
    eps_scalar: Tensor,
}

impl LayerNorm {
    pub fn new(normalized_shape: i64, eps: f64) -> Self {
        let weight_data: Vec<f32> = (0..normalized_shape).map(|_| 1.0).collect();
        let weight = Tensor::from_vec(weight_data, vec![normalized_shape]).requires_grad_(true);

        let bias_data: Vec<f32> = (0..normalized_shape).map(|_| 0.0).collect();
        let bias = Tensor::from_vec(bias_data, vec![normalized_shape]).requires_grad_(true);

        LayerNorm {
            weight: Some(weight),
            bias: Some(bias),
            normalized_shape,
            eps,
            training: std::sync::atomic::AtomicBool::new(true),
            eps_scalar: Tensor::from_scalar(eps as f32),
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let _ndim = shape.len();

        // Use references to avoid cloning weight/bias on every forward pass
        let weight = self
            .weight
            .as_ref()
            .unwrap_or_else(|| panic!("LayerNorm weight is required but was None"));
        let bias = self
            .bias
            .as_ref()
            .unwrap_or_else(|| panic!("LayerNorm bias is required but was None"));

        let result = dispatch(
            "layer_norm",
            DispatchKey::Cpu,
            &[x, x, weight, bias, &self.eps_scalar],
        );

        let output = result[0].clone();
        let mean = result[1].clone();
        let variance = result[2].clone();
        let x_hat = result[3].clone();

        // Set up gradient tracking for layer norm
        if x.requires_grad() || weight.requires_grad() || bias.requires_grad() {
            let edges = {
                let mut edges = crate::autograd::make_edge(x);
                edges.extend(crate::autograd::make_edge(weight));
                edges.extend(crate::autograd::make_edge(bias));
                edges
            };
            let backward = crate::autograd::LayerNormBackward::new(
                x.clone(),
                weight.clone(),
                bias.clone(),
                x_hat,
                mean,
                variance,
                self.eps as f32,
                edges,
            );
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
            if let Some(meta) = &w.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
        if let Some(b) = &self.bias {
            if let Some(meta) = &b.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }

    fn train_mode(&self) {
        self.training
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn eval_mode(&self) {
        self.training
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(std::sync::atomic::Ordering::Relaxed)
    }
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
    pub training: std::sync::atomic::AtomicBool,
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
            training: std::sync::atomic::AtomicBool::new(true),
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

        let is_training = self.training.load(std::sync::atomic::Ordering::Relaxed);

        let training_flag = if is_training {
            &self.training_true_scalar
        } else {
            &self.training_false_scalar
        };

        // Read current running stats - clone tensors so guards are dropped before dispatch
        let running_mean = self.running_mean.read().clone();
        let running_var = self.running_var.read().clone();

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
        );

        let output = result[0].clone();

        // In training mode, update the running stats
        if is_training {
            // Compute batch statistics for updating running stats
            let x_shape = x.shape();
            let batch_size = x_shape[0];
            let num_features = x_shape[1];
            let spatial_size: i64 = if x_shape.len() > 2 {
                x_shape[2..].iter().product()
            } else {
                1
            };

            // Compute mean over batch and spatial dimensions
            let x_reshaped = x.reshape(vec![batch_size, num_features, spatial_size]);
            let batch_mean = x_reshaped.mean(2, false).mean(0, false);

            // Compute variance over batch and spatial dimensions
            let centered = x_reshaped.sub(&batch_mean.reshape(vec![1, num_features, 1]));
            let batch_var = centered.mul(&centered).mean(2, false).mean(0, false);

            // Update running stats: running = momentum * running + (1 - momentum) * batch
            let mom = self.momentum as f32;
            let inv_mom = 1.0 - mom;

            // Get mutable references and update
            let mut running_mean_lock = self.running_mean.write();
            let new_mean = running_mean_lock
                .mul_scalar(mom)
                .add(&batch_mean.mul_scalar(inv_mom));
            *running_mean_lock = new_mean;

            let mut running_var_lock = self.running_var.write();
            let new_var = running_var_lock
                .mul_scalar(mom)
                .add(&batch_var.mul_scalar(inv_mom));
            *running_var_lock = new_var;
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
            if let Some(meta) = &w.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
        if let Some(b) = &self.bias {
            if let Some(meta) = &b.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }

    fn train_mode(&self) {
        self.training
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn eval_mode(&self) {
        self.training
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(std::sync::atomic::Ordering::Relaxed)
    }
}
