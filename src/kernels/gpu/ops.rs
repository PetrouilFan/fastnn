use crate::dispatcher::{register, DispatchKey, KernelFn};
use crate::kernels::gpu;
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use std::sync::Arc;

fn add_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_add(a, b, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let b_data = b.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        output_data[i] = unsafe { *a_data.add(i) + *b_data.add(i) };
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn sub_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_sub(a, b, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let b_data = b.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        output_data[i] = unsafe { *a_data.add(i) - *b_data.add(i) };
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn mul_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_mul(a, b, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let b_data = b.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        output_data[i] = unsafe { *a_data.add(i) * *b_data.add(i) };
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn div_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_div(a, b, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let b_data = b.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        output_data[i] = unsafe { *a_data.add(i) / *b_data.add(i) };
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn neg_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_neg(a, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        output_data[i] = -unsafe { *a_data.add(i) };
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn abs_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_abs(a, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        output_data[i] = unsafe { (*a_data.add(i)).abs() };
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn exp_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_exp(a, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        output_data[i] = unsafe { (*a_data.add(i)).exp() };
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn log_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_log(a, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        output_data[i] = unsafe { (*a_data.add(i)).ln() };
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn sqrt_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_sqrt(a, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        output_data[i] = unsafe { (*a_data.add(i)).sqrt() };
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_relu(a, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        let val = unsafe { *a_data.add(i) };
        output_data[i] = val.max(0.0);
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn fused_add_relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_fused_add_relu(a, b, device_id);
    }

    let a_shape = a.shape();
    let output_shape = a_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let b_data = b.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        let sum = unsafe { *a_data.add(i) + *b_data.add(i) };
        output_data[i] = sum.max(0.0);
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn gelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_gelu(a, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        let x = unsafe { *a_data.add(i) };
        let x3 = x * x * x;
        let t = (0.7978846 * (x + 0.044715 * x3)).tanh();
        output_data[i] = 0.5 * x * (1.0 + t);
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn sigmoid_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_sigmoid(a, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        let x = unsafe { *a_data.add(i) };
        output_data[i] = 1.0 / (1.0 + (-x).exp());
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn tanh_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_tanh(a, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        let x = unsafe { *a_data.add(i) };
        output_data[i] = x.tanh();
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn silu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_silu(a, device_id);
    }

    let iter = crate::iterator::TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();
    let numel: i64 = output_shape.iter().product();

    let a_data = a.data_ptr_f32();
    let mut output_data = vec![0.0f32; numel as usize];

    for i in 0..numel as usize {
        let x = unsafe { *a_data.add(i) };
        output_data[i] = x / (1.0 + (-x).exp());
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

fn matmul_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_matmul(a, b, device_id);
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() < 2 || b_shape.len() < 2 {
        panic!("matmul: both tensors must have at least 2 dimensions");
    }

    let m = a_shape[a_shape.len() - 2] as i32;
    let k = a_shape[a_shape.len() - 1] as i32;
    let n = b_shape[b_shape.len() - 1] as i32;

    let batch_a = if a_shape.len() > 2 {
        a_shape[..a_shape.len() - 2].iter().product::<i64>() as usize
    } else {
        1
    };
    let batch_b = if b_shape.len() > 2 {
        b_shape[..b_shape.len() - 2].iter().product::<i64>() as usize
    } else {
        1
    };
    let batch = batch_a.max(batch_b);

    let mut output_shape: Vec<i64> = vec![];
    if a_shape.len() > 2 {
        for i in 0..a_shape.len() - 2 {
            output_shape.push(a_shape[i].max(b_shape[i]));
        }
    }
    output_shape.push(m as i64);
    output_shape.push(n as i64);

    let a_data = a.data_ptr_f32();
    let b_data = b.data_ptr_f32();

    let a_rows = a_shape[a_shape.len() - 2] as usize;
    let a_cols = a_shape[a_shape.len() - 1] as usize;
    let b_cols = b_shape[b_shape.len() - 1] as usize;

    let mut output_data = vec![0.0f32; (m as usize) * (n as usize)];

    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut sum = 0.0f32;
            for p in 0..k as usize {
                let a_idx = i * a_cols + p;
                let b_idx = p * b_cols + j;
                sum += unsafe { *a_data.add(a_idx) * *b_data.add(b_idx) };
            }
            output_data[i * n as usize + j] = sum;
        }
    }

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
    ));
    vec![output]
}

#[ctor::ctor]
fn register_kernels() {
    register("add", DispatchKey::Wgpu, add_kernel as KernelFn);
    register("sub", DispatchKey::Wgpu, sub_kernel as KernelFn);
    register("mul", DispatchKey::Wgpu, mul_kernel as KernelFn);
    register("div", DispatchKey::Wgpu, div_kernel as KernelFn);
    register("neg", DispatchKey::Wgpu, neg_kernel as KernelFn);
    register("abs", DispatchKey::Wgpu, abs_kernel as KernelFn);
    register("exp", DispatchKey::Wgpu, exp_kernel as KernelFn);
    register("log", DispatchKey::Wgpu, log_kernel as KernelFn);
    register("sqrt", DispatchKey::Wgpu, sqrt_kernel as KernelFn);
    register("relu", DispatchKey::Wgpu, relu_kernel as KernelFn);
    register(
        "fused_add_relu",
        DispatchKey::Wgpu,
        fused_add_relu_kernel as KernelFn,
    );
    register("gelu", DispatchKey::Wgpu, gelu_kernel as KernelFn);
    register("sigmoid", DispatchKey::Wgpu, sigmoid_kernel as KernelFn);
    register("tanh", DispatchKey::Wgpu, tanh_kernel as KernelFn);
    register("silu", DispatchKey::Wgpu, silu_kernel as KernelFn);
    register("matmul", DispatchKey::Wgpu, matmul_kernel as KernelFn);
}
