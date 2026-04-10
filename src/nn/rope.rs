use crate::nn::Module;
use crate::tensor::Tensor;

pub struct RotaryEmbedding {
    dim: i64,
    max_seq_len: i64,
    base: f32,
    pub seq_len: i64,
    inv_freq: Vec<f32>,
    cached_cos: Option<Tensor>,
    cached_sin: Option<Tensor>,
}

impl RotaryEmbedding {
    pub fn new(dim: i64, max_seq_len: i64, base: f32) -> Self {
        let inv_freq = Self::compute_inv_freq(dim, base);
        RotaryEmbedding {
            dim,
            max_seq_len,
            base,
            seq_len: 0,
            inv_freq,
            cached_cos: None,
            cached_sin: None,
        }
    }

    fn compute_inv_freq(dim: i64, base: f32) -> Vec<f32> {
        let mut inv_freq = Vec::with_capacity((dim / 2) as usize);
        for i in 0..(dim / 2) {
            inv_freq.push(1.0 / (base.powf(4.0 * i as f32 / dim as f32)));
        }
        inv_freq
    }

    pub fn forward(&mut self, x: &Tensor, seq_len: i64) -> (Tensor, Tensor) {
        let batch = x.shape()[0];

        // Recompute if sequence length increased
        if seq_len > self.seq_len {
            self.compute_cos_sin(seq_len);
        }

        let cos = self.cached_cos.as_ref().unwrap();
        let sin = self.cached_sin.as_ref().unwrap();

        // Get the relevant slice for this sequence length
        let cos_slice = cos.slice(1, 0, seq_len, 1);
        let sin_slice = sin.slice(1, 0, seq_len, 1);

        (cos_slice, sin_slice)
    }

    fn compute_cos_sin(&mut self, seq_len: i64) {
        let seq_len = seq_len.min(self.max_seq_len);
        self.seq_len = seq_len;

        // Compute frequencies
        // freqs = inv_freq * position (for each position)
        let mut cos_data = Vec::with_capacity((seq_len * self.dim / 2) as usize);
        let mut sin_data = Vec::with_capacity((seq_len * self.dim / 2) as usize);

        for pos in 0..seq_len {
            for i in 0..(self.dim / 2) {
                let freq = self.inv_freq[i as usize] * pos as f32;
                let cos_f = freq.cos();
                let sin_f = freq.sin();
                // Each frequency pair is duplicated for the two dimensions
                cos_data.push(cos_f);
                cos_data.push(cos_f);
                sin_data.push(sin_f);
                sin_data.push(-sin_f);
            }
        }

        let shape = vec![1, seq_len, self.dim];
        self.cached_cos = Some(Tensor::from_vec(cos_data.clone(), shape.clone()));
        self.cached_sin = Some(Tensor::from_vec(sin_data.clone(), shape));
    }

    pub fn apply_rotary(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        // x shape: [batch, num_heads, seq_len, head_dim]
        // Apply rotary: x_rot = x * cos + rotate_half(x) * sin
        let x_shape = x.shape();
        let batch = x_shape[0];
        let num_heads = x_shape[1];
        let seq_len = x_shape[2];
        let head_dim = x_shape[3];

        // Split into first half and second half
        let half_dim = head_dim / 2;

        // x1 = x[..., :half_dim], x2 = x[..., half_dim:]
        let x1 = x.slice(3, 0, half_dim, 1);
        let x2 = x.slice(3, half_dim, head_dim, 1);

        // rotate_half: [-x2, x1]
        let mut x2_neg = x2.mul_scalar(-1.0);

        // Reshape cos/sin for broadcasting: [1, 1, seq_len, half_dim * 2] -> [batch, num_heads, seq_len, head_dim]
        let cos_exp = cos.view(vec![batch, num_heads, seq_len, head_dim]);
        let sin_exp = sin.view(vec![batch, num_heads, seq_len, head_dim]);

        // x * cos + rotate_half(x) * sin
        let x_cos = x.mul(&cos_exp);
        let rotate_x = Tensor::cat(&vec![x2_neg, x1], 3);
        let rotate_sin = rotate_x.mul(&sin_exp);

        x_cos.add(&rotate_sin)
    }
}

impl Module for RotaryEmbedding {
    fn forward(&self, x: &Tensor) -> Tensor {
        let seq_len = x.shape()[2];
        let batch = x.shape()[0];

        // Get cos/sin for this sequence
        let (cos, sin) = self.forward(x, seq_len);

        self.apply_rotary(x, &cos, &sin)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    fn zero_grad(&self) {}

    fn train_mode(&self) {}

    fn eval_mode(&self) {}

    fn is_training(&self) -> bool {
        false
    }
}

pub struct LlamaRotaryEmbedding {
    dim: i64,
    max_seq_len: i64,
    base: f32,
    rope_ratio: f32,
    pub seq_len: i64,
    inv_freq: Vec<f32>,
    cached_cos: Option<Tensor>,
    cached_sin: Option<Tensor>,
}

impl LlamaRotaryEmbedding {
    pub fn new(dim: i64, max_seq_len: i64, base: f32, rope_ratio: f32) -> Self {
        let inv_freq = Self::compute_inv_freq(dim, base, rope_ratio);
        LlamaRotaryEmbedding {
            dim,
            max_seq_len,
            base,
            rope_ratio,
            seq_len: 0,
            inv_freq,
            cached_cos: None,
            cached_sin: None,
        }
    }

    fn compute_inv_freq(dim: i64, base: f32, rope_ratio: f32) -> Vec<f32> {
        let mut inv_freq = Vec::with_capacity((dim / 2) as usize);
        for i in 0..(dim / 2) {
            let freq = base.powf(4.0 * i as f32 / dim as f32);
            inv_freq.push(rope_ratio / freq);
        }
        inv_freq
    }

    pub fn compute_cos_sin(&mut self, seq_len: i64) {
        let seq_len = seq_len.min(self.max_seq_len);

        if seq_len <= self.seq_len {
            return;
        }

        self.seq_len = seq_len;

        let mut cos_data = Vec::with_capacity((seq_len * self.dim) as usize);
        let mut sin_data = Vec::with_capacity((seq_len * self.dim) as usize);

        for pos in 0..seq_len {
            for i in 0..(self.dim / 2) {
                let freq = self.inv_freq[i as usize] * pos as f32;
                let cos_f = freq.cos();
                let sin_f = freq.sin();

                // Interleave: [cos, cos] per dimension pair
                cos_data.push(cos_f);
                cos_data.push(cos_f);
                sin_data.push(sin_f);
                sin_data.push(-sin_f);
            }
        }

        let shape = vec![seq_len, self.dim];
        self.cached_cos = Some(Tensor::from_vec(cos_data, shape.clone()));
        self.cached_sin = Some(Tensor::from_vec(sin_data, shape));
    }

    pub fn forward(&mut self, seq_len: i64) -> (Tensor, Tensor) {
        self.compute_cos_sin(seq_len);

        let cos = self.cached_cos.as_ref().unwrap();
        let sin = self.cached_sin.as_ref().unwrap();

        let cos_s = cos.slice(1, 0, seq_len, 1);
        let sin_s = sin.slice(1, 0, seq_len, 1);

        (cos_s, sin_s)
    }

    pub fn apply(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let x_shape = x.shape();
        let batch = x_shape[0];
        let num_heads = x_shape[1];
        let seq_len = x_shape[2];
        let head_dim = x_shape[3];

        let half_dim = head_dim / 2;

        let x1 = x.slice(3, 0, half_dim, 1);
        let x2 = x.slice(3, half_dim, head_dim, 1);

        let x2_neg = x2.mul_scalar(-1.0);

        // cos and sin are [seq_len, head_dim], expand to [batch, num_heads, seq_len, head_dim]
        let cos_exp = cos.view(vec![batch, num_heads, seq_len, head_dim]);
        let sin_exp = sin.view(vec![batch, num_heads, seq_len, head_dim]);

        let x_cos = x.mul(&cos_exp);
        let rotate_x = Tensor::cat(&vec![x2_neg, x1], 3);
        let rotate_sin = rotate_x.mul(&sin_exp);

        x_cos.add(&rotate_sin)
    }
}
