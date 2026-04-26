use crate::autograd::{self, AutogradMeta};
use crate::dispatcher::{dispatch, DispatchKey};
use crate::nn::Module;
use crate::tensor::Tensor;
use std::sync::Arc;

pub struct Embedding {
    pub weight: Tensor,
    #[allow(dead_code)]
    pub num_embeddings: i64,
    #[allow(dead_code)]
    pub embedding_dim: i64,
    training: std::sync::atomic::AtomicBool,
}

impl Embedding {
    pub fn new(num_embeddings: i64, embedding_dim: i64) -> Self {
        let scale = 0.05;
        let weight_data: Vec<f32> = (0..num_embeddings * embedding_dim)
            .map(|_| (crate::random_f32() - 0.5) * 2.0 * scale)
            .collect();
        let weight = Tensor::from_vec(weight_data, vec![num_embeddings, embedding_dim]);
        let weight = weight.requires_grad_(true);

        Embedding {
            weight,
            num_embeddings,
            embedding_dim,
            training: std::sync::atomic::AtomicBool::new(true),
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }
}

impl Module for Embedding {
    fn forward(&self, indices: &Tensor) -> Tensor {
        // Validate indices are within [0, num_embeddings)
        let indices_cpu = indices.to_cpu();
        let indices_data = indices_cpu.as_i64_slice();
        for &idx in indices_data {
            if idx < 0 || idx >= self.num_embeddings {
                panic!(
                    "Embedding index out of range: got {}, expected 0 <= idx < {}",
                    idx,
                    self.num_embeddings
                );
            }
        }

        let result = dispatch("embedding", DispatchKey::Cpu, &[&self.weight, indices]);
        let mut output = result[0].clone();

        if self.weight.requires_grad() {
            let edges = autograd::make_edge(&self.weight);
            let backward = Arc::new(autograd::EmbeddingBackward::new(
                self.weight.clone(),
                indices.clone(),
                edges,
            ));
            let mut meta = AutogradMeta::new_non_leaf(false);
            meta.grad_fn = Some(backward);
            Arc::make_mut(&mut output.inner).autograd_meta = Some(Arc::new(std::sync::Mutex::new(meta)));
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
