use crate::nn::Module;
use crate::tensor::Tensor;

pub struct Upsample {
    pub scale_factor: f64,
    pub mode: String, // "nearest" or "bilinear"
}

impl Upsample {
    pub fn new(scale_factor: f64, mode: String) -> Self {
        Upsample { scale_factor, mode }
    }
}

impl Module for Upsample {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_shape = x.shape_ref();
        let ndim = x_shape.len();
        if ndim < 2 {
            return x.clone();
        }

        let scale = self.scale_factor;
        let out_shape: Vec<i64> = x_shape
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                if i >= ndim - 2 {
                    (s as f64 * scale) as i64
                } else {
                    s
                }
            })
            .collect();

        if self.mode == "nearest" {
            // Nearest neighbor upsampling
            let x_data = x.as_f32_slice();
            let in_h = x_shape[ndim - 2] as usize;
            let in_w = x_shape[ndim - 1] as usize;
            let out_h = out_shape[ndim - 2] as usize;
            let out_w = out_shape[ndim - 1] as usize;
            let spatial_in = in_h * in_w;
            let spatial_out = out_h * out_w;
            let batch_channels: usize = x_shape[..ndim - 2].iter().map(|&x| x as usize).product();
            let scale_f = self.scale_factor as f32;

            let mut out_data = vec![0.0f32; batch_channels * spatial_out];
            for bc in 0..batch_channels {
                for oh in 0..out_h {
                    let ih = (oh as f32 / scale_f).min((in_h - 1) as f32) as usize;
                    for ow in 0..out_w {
                        let iw = (ow as f32 / scale_f).min((in_w - 1) as f32) as usize;
                        out_data[bc * spatial_out + oh * out_w + ow] =
                            x_data[bc * spatial_in + ih * in_w + iw];
                    }
                }
            }
            Tensor::from_vec(out_data, out_shape)
        } else {
            // Bilinear upsampling
            let x_data = x.as_f32_slice();
            let in_h = x_shape[ndim - 2] as usize;
            let in_w = x_shape[ndim - 1] as usize;
            let out_h = out_shape[ndim - 2] as usize;
            let out_w = out_shape[ndim - 1] as usize;
            let spatial_in = in_h * in_w;
            let spatial_out = out_h * out_w;
            let batch_channels: usize = x_shape[..ndim - 2].iter().map(|&x| x as usize).product();

            let mut out_data = vec![0.0f32; batch_channels * spatial_out];
            let scale_f = self.scale_factor as f32;
            for bc in 0..batch_channels {
                for oh in 0..out_h {
                    let ih_f = oh as f32 / scale_f;
                    let ih0 = ih_f.min((in_h - 2) as f32) as usize;
                    let ih1 = (ih0 + 1).min(in_h - 1);
                    let dy = ih_f - ih0 as f32;
                    for ow in 0..out_w {
                        let iw_f = ow as f32 / scale_f;
                        let iw0 = iw_f.min((in_w - 2) as f32) as usize;
                        let iw1 = (iw0 + 1).min(in_w - 1);
                        let dx = iw_f - iw0 as f32;

                        let v00 = x_data[bc * spatial_in + ih0 * in_w + iw0];
                        let v01 = x_data[bc * spatial_in + ih0 * in_w + iw1];
                        let v10 = x_data[bc * spatial_in + ih1 * in_w + iw0];
                        let v11 = x_data[bc * spatial_in + ih1 * in_w + iw1];

                        let top = v00 * (1.0 - dx) + v01 * dx;
                        let bottom = v10 * (1.0 - dx) + v11 * dx;
                        out_data[bc * spatial_out + oh * out_w + ow] =
                            top * (1.0 - dy) + bottom * dy;
                    }
                }
            }
            Tensor::from_vec(out_data, out_shape)
        }
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
