use crate::nn::Module;
use crate::tensor::Tensor;
use rand::Rng;
use std::sync::atomic::AtomicBool;

pub struct Dropout {
    pub p: f64,
    pub training: AtomicBool,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Dropout {
            p,
            training: AtomicBool::new(true),
        }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        if self.training.load(std::sync::atomic::Ordering::Relaxed) {
            let x_data = x.as_f32_slice();
            let scale = 1.0 / (1.0 - self.p) as f32;
            let keep_prob = 1.0 - self.p;
            // Use thread-local RNG for batch generation (much faster than rand::random per element)
            let mut rng = rand::thread_rng();
            let mask_data: Vec<f32> = x_data
                .iter()
                .map(|&v| {
                    if rng.gen::<f64>() < keep_prob {
                        v * scale
                    } else {
                        0.0
                    }
                })
                .collect();

            let shape = x.shape();
            let mut out = Tensor::from_vec(mask_data, shape);
            if x.requires_grad() {
                out = out.requires_grad_(true);
            }
            out
        } else {
            x.clone()
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    fn zero_grad(&self) {}

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

pub struct Dropout2d {
    pub p: f64,
    pub training: std::sync::atomic::AtomicBool,
}

impl Dropout2d {
    pub fn new(p: f64) -> Self {
        Dropout2d {
            p,
            training: std::sync::atomic::AtomicBool::new(true),
        }
    }
}

impl Module for Dropout2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        if self.training.load(std::sync::atomic::Ordering::Relaxed) {
            let x_shape = x.shape();
            let batch = x_shape[0] as usize;
            let channels = x_shape[1] as usize;
            let spatial: usize = x_shape[2..].iter().map(|&x| x as usize).product();
            let scale = 1.0 / (1.0 - self.p) as f32;
            let keep_prob = 1.0 - self.p;

            // Generate one mask value per channel per batch
            let mut rng = rand::thread_rng();
            let channel_mask: Vec<f32> = (0..batch * channels)
                .map(|_| {
                    if rng.gen::<f64>() < keep_prob {
                        scale
                    } else {
                        0.0
                    }
                })
                .collect();

            // Apply mask: each channel's spatial dimensions get the same mask value
            let x_data = x.as_f32_slice();
            let mut out_data = Vec::with_capacity(x_data.len());
            for b in 0..batch {
                for c in 0..channels {
                    let mask_val = channel_mask[b * channels + c];
                    for s in 0..spatial {
                        out_data.push(x_data[b * channels * spatial + c * spatial + s] * mask_val);
                    }
                }
            }

            let mut out = Tensor::from_vec(out_data, x_shape);
            if x.requires_grad() {
                out = out.requires_grad_(true);
            }
            out
        } else {
            x.clone()
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    fn zero_grad(&self) {}

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
