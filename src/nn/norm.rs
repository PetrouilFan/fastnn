use crate::autograd::{
    self, AutogradMeta, BatchNorm1dBackward, BatchNorm2dBackward, GroupNormBackward,
    RMSNormBackward,
};
use crate::ir::node::{DimExpr, IrDType, TensorType};
use crate::tensor::Tensor;
use crate::{
    impl_training_state,
    nn::{clear_grad, Module, TrainingState},
};
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Clone)]
pub struct LayerNorm {
    #[allow(dead_code)]
    pub normalized_shape: i64,
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
    training: TrainingState,
    #[allow(dead_code)]
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

        let output = Tensor::exec_aot(&[x, weight, bias], |g, ins| {
            vec![g.layer_norm(&ins[0], &ins[1], &ins[2], self.eps)]
        })
        .expect("LayerNorm::forward: AOT failed")
        .into_iter()
        .next()
        .unwrap();

        // Set up gradient tracking for layer norm
        if x.requires_grad() || weight.requires_grad() || bias.requires_grad() {
            let edges = {
                let mut edges = crate::autograd::make_edge(x);
                edges.extend(crate::autograd::make_edge(weight));
                edges.extend(crate::autograd::make_edge(bias));
                edges
            };
            let inputs = vec![x.clone(), weight.clone(), bias.clone()];
            let backward = crate::autograd::LayerNormBackward::new(edges, inputs);
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

#[derive(Clone)]
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

        let grads_needed = x.requires_grad()
            || self.weight.as_ref().is_some_and(|w| w.requires_grad())
            || self.bias.as_ref().is_some_and(|b| b.requires_grad());

        // Compute batch statistics via fused AOT graph (mean + var in one pass)
        // In eval mode, read running stats from the stored values
        let (mean, var, batch_mean, batch_var) = if is_training {
            let x_shape = x.shape_ref();
            let batch_size = x_shape[0];
            let num_features = x_shape[1];
            let spatial_size: i64 = if x_shape.len() > 2 {
                x_shape[2..].iter().product()
            } else {
                1
            };

            // Fused AOT graph: compute batch mean and variance in one compilation
            let x_reshaped = x.reshape(vec![batch_size, num_features, spatial_size]);
            let stats = Tensor::exec_aot(&[&x_reshaped], |g, ins| {
                let b_mean = g.reduce_mean(&ins[0], 2, false);
                let b_mean_2d = g.reduce_mean(&b_mean, 0, false);
                // The centered needs a reshape for correct broadcasting
                let centered = g.sub(
                    &ins[0],
                    &g.reshape(
                        &b_mean_2d,
                        &[
                            DimExpr::Known(1),
                            DimExpr::Known(num_features as u64),
                            DimExpr::Known(1),
                        ],
                    ),
                );
                let sq = g.mul(&centered, &centered);
                let b_var = g.reduce_mean(&sq, 2, false);
                let b_var_1d = g.reduce_mean(&b_var, 0, false);
                vec![b_mean_2d, b_var_1d]
            })
            .expect("BatchNorm stats computation failed");

            let b_mean = stats[0].clone();
            let b_var = stats[1].clone();

            // Clone before releasing read lock if training (for running stats update)
            let rm = self.running_mean.read();
            let rv = self.running_var.read();
            (rm.clone(), rv.clone(), b_mean, b_var)
        } else {
            let rm = self.running_mean.read();
            let rv = self.running_var.read();
            (rm.clone(), rv.clone(), rm.clone(), rv.clone())
        };

        let (mean_ref, var_ref) = if is_training {
            (&batch_mean, &batch_var)
        } else {
            (&mean, &var)
        };

        let result = Tensor::exec_aot(&[x, weight_ref, bias_ref, mean_ref, var_ref], |g, ins| {
            vec![g.batch_norm(&ins[0], &ins[1], &ins[2], &ins[3], &ins[4], self.eps)]
        })
        .expect("BatchNorm1d::forward: AOT failed");

        #[allow(unused_mut)]
        let mut output = result.into_iter().next().unwrap();

        // Remove redundant output.clone() — output is uniquely owned
        #[allow(unused_mut)]
        let mut output = output;

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
            let mut inputs = vec![x.clone()];
            if let Some(ref w) = self.weight {
                inputs.push(w.clone());
            }
            if let Some(ref b) = self.bias {
                inputs.push(b.clone());
            }
            let backward = BatchNorm1dBackward::new(edges, inputs);
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

#[derive(Clone)]
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
    pub normalized_shape: i64,
    #[allow(dead_code)]
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
        let result = Tensor::exec_aot(&[x, &self.weight], |g, ins| {
            vec![g.rms_norm(&ins[0], &ins[1], self.eps as f64)]
        })
        .expect("RMSNorm::forward: AOT failed");
        let mut output = result.into_iter().next().unwrap();

        if x.requires_grad() || self.weight.requires_grad() {
            let edges = {
                let mut edges = autograd::make_edge(x);
                edges.extend(autograd::make_edge(&self.weight));
                edges
            };
            let inputs = vec![x.clone(), self.weight.clone()];
            let backward = RMSNormBackward::new(edges, inputs);
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

#[derive(Clone)]
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

        // Single fused AOT graph for GroupNorm forward
        // Reduces overhead by compiling one graph instead of per-op exec_aot calls
        let mut output = Tensor::exec_aot(&[x, &self.weight, &self.bias], |g, ins| {
            let x = &ins[0];
            let w = &ins[1];
            let b = &ins[2];

            // Reshape: [N, C, H, W] -> [N, G, C/G, H*W]
            let reshaped = g.reshape(
                x,
                &[
                    DimExpr::Known(batch as u64),
                    DimExpr::Known(self.num_groups as u64),
                    DimExpr::Known(group_size as u64),
                    DimExpr::Known(spatial as u64),
                ],
            );

            // Scalar constants reused below
            let scalar_tt = TensorType::new(vec![], IrDType::F32);
            let two_data = 2.0f32.to_le_bytes().to_vec();
            let _two_c = g.constant(&two_data, scalar_tt.clone());
            let eps_data = self.eps.to_le_bytes().to_vec();
            let eps_c = g.constant(&eps_data, scalar_tt);

            // Mean along dim 2 (group_size) and 3 (spatial)
            let mean = g.reduce_mean(&reshaped, 2, true);
            let mean = g.reduce_mean(&mean, 3, true);

            // Variance: mean((x - mean)^2)
            let centered = g.sub(&reshaped, &mean);
            let two_data = 2.0f32.to_le_bytes().to_vec();
            let two_tt = TensorType::new(vec![], IrDType::F32);
            let two_c = g.constant(&two_data, two_tt);
            let sq = g.pow(&centered, &two_c);
            let var = g.reduce_mean(&sq, 2, true);
            let var = g.reduce_mean(&var, 3, true);

            // Normalize: (x - mean) / sqrt(var + eps)
            let var_eps = g.add_scalar(&var, &eps_c);
            let std = g.sqrt(&var_eps);
            let x_norm = g.div(&centered, &std);

            // Reshape back to original
            let out_shape: Vec<DimExpr> =
                x_shape.iter().map(|&d| DimExpr::Known(d as u64)).collect();
            let x_norm = g.reshape(&x_norm, &out_shape);

            // Weight * norm + bias with broadcast shape [1, C, 1, ...]
            let mut weight_shape: Vec<DimExpr> =
                vec![DimExpr::Known(1), DimExpr::Known(channels as u64)];
            for _ in 2..x_shape.len() {
                weight_shape.push(DimExpr::Known(1));
            }
            let w_r = g.reshape(w, &weight_shape);
            let b_r = g.reshape(b, &weight_shape);
            let scaled = g.mul(&x_norm, &w_r);
            vec![g.add(&scaled, &b_r)]
        })
        .expect("GroupNorm::forward: AOT execution failed")
        .into_iter()
        .next()
        .unwrap();

        if x.requires_grad() || self.weight.requires_grad() || self.bias.requires_grad() {
            let edges = {
                let mut edges = autograd::make_edge(x);
                edges.extend(autograd::make_edge(&self.weight));
                edges.extend(autograd::make_edge(&self.bias));
                edges
            };
            let inputs = vec![x.clone(), self.weight.clone(), self.bias.clone()];
            let backward = GroupNormBackward::new(edges, inputs);
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
}

impl Clone for BatchNorm2d {
    fn clone(&self) -> Self {
        BatchNorm2d {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            running_mean: parking_lot::RwLock::new(self.running_mean.read().clone()),
            running_var: parking_lot::RwLock::new(self.running_var.read().clone()),
            eps: self.eps,
            momentum: self.momentum,
            num_features: self.num_features,
            training: self.training.clone(),
        }
    }
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
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let is_training = self.training.is_training();

        // Read current running stats - clone tensors so guards are dropped before dispatch
        let running_mean = self.running_mean.read().clone();
        let running_var = self.running_var.read().clone();

        // Compute batch stats in training mode
        let (batch_mean, batch_var) = if is_training {
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

        let (mean, var) = if is_training {
            (&batch_mean, &batch_var)
        } else {
            (&running_mean, &running_var)
        };

        let result = Tensor::exec_aot(&[x, &self.weight, &self.bias, mean, var], |g, ins| {
            vec![g.batch_norm(&ins[0], &ins[1], &ins[2], &ins[3], &ins[4], self.eps as f64)]
        })
        .expect("BatchNorm2d::forward: AOT failed");

        let mut output = result.into_iter().next().unwrap();

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
            let inputs = vec![x.clone(), self.weight.clone(), self.bias.clone()];
            let backward = BatchNorm2dBackward::new(edges, inputs);
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
