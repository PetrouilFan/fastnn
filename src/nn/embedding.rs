use crate::dispatcher::{dispatch, DispatchKey};
use crate::nn::Module;
use crate::tensor::Tensor;

pub struct Embedding {
    pub weight: Tensor,
    pub num_embeddings: i64,
    pub embedding_dim: i64,
    training: std::sync::atomic::AtomicBool,
}

impl Embedding {
    pub fn new(num_embeddings: i64, embedding_dim: i64) -> Self {
        let scale = 0.05;
        let weight_data: Vec<f32> = (0..num_embeddings * embedding_dim)
            .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
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
}

impl Module for Embedding {
    fn forward(&self, indices: &Tensor) -> Tensor {
        let result = dispatch("embedding", DispatchKey::Cpu, &[&self.weight, indices]);

        result[0].clone()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }

    fn zero_grad(&self) {
        let mut meta = self.weight.inner.autograd_meta.clone();
        if let Some(m) = &mut meta {
            m.grad = None;
        }
    }

    fn train_mode(&self) {
        self.training
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn eval_mode(&self) {
        self.training
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    fn is_training(&self) -> bool {
        self.training.load(std::sync::atomic::Ordering::SeqCst)
    }
}
