use crate::nn::conv::Conv2d;
use crate::nn::norm::BatchNorm2d;
use crate::nn::Module;
use crate::tensor::Tensor;

pub struct ResidualBlock {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub conv2: Conv2d,
    pub bn2: BatchNorm2d,
    pub downsample: Option<(Conv2d, BatchNorm2d)>,
}

impl ResidualBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        conv1_in: i64,
        conv1_out: i64,
        conv1_kernel: i64,
        conv1_stride: i64,
        conv1_padding: i64,
        bn1_features: i64,
        conv2_in: i64,
        conv2_out: i64,
        conv2_kernel: i64,
        conv2_stride: i64,
        conv2_padding: i64,
        bn2_features: i64,
        downsample: Option<(i64, i64, i64, i64, i64, i64)>,
    ) -> Self {
        let conv1 = Conv2d::new(conv1_in, conv1_out, conv1_kernel, conv1_stride, conv1_padding);
        let bn1 = BatchNorm2d::new(bn1_features, 1e-5, 0.1);
        let conv2 = Conv2d::new(conv2_in, conv2_out, conv2_kernel, conv2_stride, conv2_padding);
        let bn2 = BatchNorm2d::new(bn2_features, 1e-5, 0.1);
        let downsample = downsample.map(|(ds_in, ds_out, ds_k, ds_s, ds_p, ds_bn)| {
            let ds_conv = Conv2d::new(ds_in, ds_out, ds_k, ds_s, ds_p);
            let ds_bn = BatchNorm2d::new(ds_bn, 1e-5, 0.1);
            (ds_conv, ds_bn)
        });
        ResidualBlock { conv1, bn1, conv2, bn2, downsample }
    }
}

impl Module for ResidualBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        let identity = if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            ds_bn.forward(&ds_conv.forward(x))
        } else {
            x.clone()
        };

        let out = self.conv1.forward(x);
        let out = self.bn1.forward(&out);
        let out = out.relu();
        let out = self.conv2.forward(&out);
        let out = self.bn2.forward(&out);

        out.add(&identity).relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.conv1.parameters();
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            params.extend(ds_conv.parameters());
            params.extend(ds_bn.parameters());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = self.conv1.named_parameters();
        params.extend(self.bn1.named_parameters());
        params.extend(self.conv2.named_parameters());
        params.extend(self.bn2.named_parameters());
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            params.extend(ds_conv.named_parameters());
            params.extend(ds_bn.named_parameters());
        }
        params
    }

    fn zero_grad(&self) {
        for t in self.conv1.parameters().iter()
            .chain(self.bn1.parameters().iter())
            .chain(self.conv2.parameters().iter())
            .chain(self.bn2.parameters().iter())
        {
            if let Some(meta) = &t.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            for t in ds_conv.parameters().iter().chain(ds_bn.parameters().iter()) {
                if let Some(meta) = &t.inner.autograd_meta {
                    if let Ok(mut lock) = meta.lock() {
                        lock.grad = None;
                    }
                }
            }
        }
    }

    fn train_mode(&self) {
        self.conv1.train_mode();
        self.bn1.train_mode();
        self.conv2.train_mode();
        self.bn2.train_mode();
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            ds_conv.train_mode();
            ds_bn.train_mode();
        }
    }

    fn eval_mode(&self) {
        self.conv1.eval_mode();
        self.bn1.eval_mode();
        self.conv2.eval_mode();
        self.bn2.eval_mode();
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            ds_conv.eval_mode();
            ds_bn.eval_mode();
        }
    }

    fn is_training(&self) -> bool {
        self.conv1.is_training()
    }
}
