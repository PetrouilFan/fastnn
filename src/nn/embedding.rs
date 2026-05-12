use crate::tensor::Tensor;
use crate::{
    impl_training_state,
    nn::{clear_grad, Module, TrainingState},
};

pub struct Embedding {
    pub weight: Tensor,
    #[allow(dead_code)]
    pub num_embeddings: i64,
    #[allow(dead_code)]
    pub embedding_dim: i64,
    training: TrainingState,
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
            training: TrainingState::new(),
        }
    }
}

impl Module for Embedding {
    fn forward(&self, indices: &Tensor) -> Tensor {
        Tensor::exec_aot(&[&self.weight, indices], |g, ins| {
            vec![g.embedding(&ins[0], &ins[1])]
        })
        .expect("Embedding::forward: AOT failed")
        .into_iter()
        .next()
        .unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }

    fn zero_grad(&self) {
        clear_grad(&self.weight);
    }

    impl_training_state!(self, self.training);
}
