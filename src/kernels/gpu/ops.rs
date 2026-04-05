#![allow(clippy::needless_range_loop)]

use crate::dispatcher::{register, DispatchKey, KernelFn};
use crate::kernels::gpu;
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use std::sync::Arc;

fn add_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    // Check if either tensor is on GPU
    let device_id = match (a.device(), b.device()) {
        (Device::Wgpu(id), _) => Some(id),
        (_, Device::Wgpu(id)) => Some(id),
        _ => None,
    };

    if let Some(device_id) = device_id {
        // Move tensors to the target GPU device if needed
        let a_gpu = match a.device() {
            Device::Cpu => a.to_gpu(device_id),
            Device::Wgpu(id) if id != device_id => a.to_gpu(device_id),
            _ => a.clone(),
        };
        let b_gpu = match b.device() {
            Device::Cpu => b.to_gpu(device_id),
            Device::Wgpu(id) if id != device_id => b.to_gpu(device_id),
            _ => b.clone(),
        };
        return gpu::gpu_add(&a_gpu, &b_gpu, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("add", crate::dispatcher::DispatchKey::Cpu, args)
}

fn sub_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    // Check if either tensor is on GPU
    let device_id = match (a.device(), b.device()) {
        (Device::Wgpu(id), _) => Some(id),
        (_, Device::Wgpu(id)) => Some(id),
        _ => None,
    };

    if let Some(device_id) = device_id {
        // Move tensors to the target GPU device if needed
        let a_gpu = match a.device() {
            Device::Cpu => a.to_gpu(device_id),
            Device::Wgpu(id) if id != device_id => a.to_gpu(device_id),
            _ => a.clone(),
        };
        let b_gpu = match b.device() {
            Device::Cpu => b.to_gpu(device_id),
            Device::Wgpu(id) if id != device_id => b.to_gpu(device_id),
            _ => b.clone(),
        };
        return gpu::gpu_sub(&a_gpu, &b_gpu, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("sub", crate::dispatcher::DispatchKey::Cpu, args)
}

fn mul_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    // Check if either tensor is on GPU
    let device_id = match (a.device(), b.device()) {
        (Device::Wgpu(id), _) => Some(id),
        (_, Device::Wgpu(id)) => Some(id),
        _ => None,
    };

    if let Some(device_id) = device_id {
        // Move tensors to the target GPU device if needed
        let a_gpu = match a.device() {
            Device::Cpu => a.to_gpu(device_id),
            Device::Wgpu(id) if id != device_id => a.to_gpu(device_id),
            _ => a.clone(),
        };
        let b_gpu = match b.device() {
            Device::Cpu => b.to_gpu(device_id),
            Device::Wgpu(id) if id != device_id => b.to_gpu(device_id),
            _ => b.clone(),
        };
        return gpu::gpu_mul(&a_gpu, &b_gpu, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("mul", crate::dispatcher::DispatchKey::Cpu, args)
}

fn div_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    // Check if either tensor is on GPU
    let device_id = match (a.device(), b.device()) {
        (Device::Wgpu(id), _) => Some(id),
        (_, Device::Wgpu(id)) => Some(id),
        _ => None,
    };

    if let Some(device_id) = device_id {
        // Move tensors to the target GPU device if needed
        let a_gpu = match a.device() {
            Device::Cpu => a.to_gpu(device_id),
            Device::Wgpu(id) if id != device_id => a.to_gpu(device_id),
            _ => a.clone(),
        };
        let b_gpu = match b.device() {
            Device::Cpu => b.to_gpu(device_id),
            Device::Wgpu(id) if id != device_id => b.to_gpu(device_id),
            _ => b.clone(),
        };
        return gpu::gpu_div(&a_gpu, &b_gpu, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("div", crate::dispatcher::DispatchKey::Cpu, args)
}

fn neg_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_neg(a, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("neg", crate::dispatcher::DispatchKey::Cpu, args)
}

fn abs_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_abs(a, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("abs", crate::dispatcher::DispatchKey::Cpu, args)
}

fn exp_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_exp(a, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("exp", crate::dispatcher::DispatchKey::Cpu, args)
}

fn log_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_log(a, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("log", crate::dispatcher::DispatchKey::Cpu, args)
}

fn sqrt_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_sqrt(a, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("sqrt", crate::dispatcher::DispatchKey::Cpu, args)
}

fn relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_relu(a, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("relu", crate::dispatcher::DispatchKey::Cpu, args)
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
        DType::F32,
    ));
    vec![output]
}

fn gelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_gelu(a, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("gelu", crate::dispatcher::DispatchKey::Cpu, args)
}

fn sigmoid_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_sigmoid(a, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("sigmoid", crate::dispatcher::DispatchKey::Cpu, args)
}

fn tanh_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_tanh(a, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("tanh", crate::dispatcher::DispatchKey::Cpu, args)
}

fn silu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_silu(a, device_id);
    }

    // Both are CPU - delegate to optimized CPU kernel
    crate::dispatcher::dispatch("silu", crate::dispatcher::DispatchKey::Cpu, args)
}

fn matmul_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    // Check if either tensor is on GPU
    let device_id = match (a.device(), b.device()) {
        (Device::Wgpu(id), _) => Some(id),
        (_, Device::Wgpu(id)) => Some(id),
        _ => None,
    };

    if let Some(device_id) = device_id {
        // Move tensors to the target GPU device if needed
        let a_gpu = match a.device() {
            Device::Cpu => a.to_gpu(device_id),
            Device::Wgpu(id) if id != device_id => a.to_gpu(device_id),
            _ => a.clone(),
        };
        let b_gpu = match b.device() {
            Device::Cpu => b.to_gpu(device_id),
            Device::Wgpu(id) if id != device_id => b.to_gpu(device_id),
            _ => b.clone(),
        };
        return gpu::gpu_matmul(&a_gpu, &b_gpu, device_id);
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
    let _batch = batch_a.max(batch_b);

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

    // Use BLAS/SIMD optimized matmul instead of naive triple loop
    // This is a fallback for CPU tensors when GPU is not available
    use crate::kernels::blas::matmul_blas;

    let a_rows = a_shape[a_shape.len() - 2] as usize;
    let a_cols = a_shape[a_shape.len() - 1] as usize;
    let b_cols = b_shape[b_shape.len() - 1] as usize;

    let a_slice = unsafe { std::slice::from_raw_parts(a_data, a_rows * a_cols) };
    let b_slice = unsafe { std::slice::from_raw_parts(b_data, k as usize * b_cols) };

    let result = matmul_blas(a_slice, b_slice, m as usize, k as usize, n as usize);
    let output_data = result;

    let storage = Arc::new(Storage::from_vec(output_data, DType::F32, a.device()));
    let output = Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        output_shape.iter().copied().collect(),
        DType::F32,
    ));
    vec![output]
}

fn sum_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_sum(a, dim, keepdim, device_id);
    }

    // CPU fallback
    // This would call the CPU sum kernel, but for now we'll just panic
    // since the CPU implementation is already in cpu.rs
    panic!("CPU sum kernel not implemented in gpu/ops.rs - should use cpu.rs");
}

fn mean_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_mean(a, dim, keepdim, device_id);
    }

    panic!("CPU mean kernel not implemented in gpu/ops.rs");
}

fn max_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_max(a, dim, keepdim, device_id);
    }

    panic!("CPU max kernel not implemented in gpu/ops.rs");
}

fn min_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    if let Device::Wgpu(device_id) = a.device() {
        return gpu::gpu_min(a, dim, keepdim, device_id);
    }

    panic!("CPU min kernel not implemented in gpu/ops.rs");
}

fn gt_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let scalar = args[1].item();
    let device_id = match a.device() {
        Device::Wgpu(id) => id,
        _ => panic!("gt_scalar GPU kernel called with CPU tensor"),
    };
    gpu::gpu_gt_scalar(a, scalar, device_id)
}

fn lt_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let scalar = args[1].item();
    let device_id = match a.device() {
        Device::Wgpu(id) => id,
        _ => panic!("lt_scalar GPU kernel called with CPU tensor"),
    };
    gpu::gpu_lt_scalar(a, scalar, device_id)
}

fn logical_not_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let device_id = match a.device() {
        Device::Wgpu(id) => id,
        _ => panic!("logical_not GPU kernel called with CPU tensor"),
    };
    gpu::gpu_logical_not(a, device_id)
}

fn mul_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let scalar = args[1].item();
    let device_id = match a.device() {
        Device::Wgpu(id) => id,
        _ => panic!("mul_scalar GPU kernel called with CPU tensor"),
    };
    gpu::gpu_mul_scalar(a, scalar, device_id)
}

fn add_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let scalar = args[1].item();
    let device_id = match a.device() {
        Device::Wgpu(id) => id,
        _ => panic!("add_scalar GPU kernel called with CPU tensor"),
    };
    gpu::gpu_add_scalar(a, scalar, device_id)
}

fn sub_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let scalar = args[1].item();
    let device_id = match a.device() {
        Device::Wgpu(id) => id,
        _ => panic!("sub_scalar GPU kernel called with CPU tensor"),
    };
    gpu::gpu_sub_scalar(a, scalar, device_id)
}

fn div_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let scalar = args[1].item();
    let device_id = match a.device() {
        Device::Wgpu(id) => id,
        _ => panic!("div_scalar GPU kernel called with CPU tensor"),
    };
    gpu::gpu_div_scalar(a, scalar, device_id)
}

fn transpose_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim0 = args[1].item() as usize;
    let dim1 = args[2].item() as usize;
    vec![a.transpose(dim0, dim1)]
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
    register("sum", DispatchKey::Wgpu, sum_kernel as KernelFn);
    register("mean", DispatchKey::Wgpu, mean_kernel as KernelFn);
    register("max", DispatchKey::Wgpu, max_kernel as KernelFn);
    register("min", DispatchKey::Wgpu, min_kernel as KernelFn);
    register("gt_scalar", DispatchKey::Wgpu, gt_scalar_kernel as KernelFn);
    register("lt_scalar", DispatchKey::Wgpu, lt_scalar_kernel as KernelFn);
    register(
        "logical_not",
        DispatchKey::Wgpu,
        logical_not_kernel as KernelFn,
    );
    register(
        "mul_scalar",
        DispatchKey::Wgpu,
        mul_scalar_kernel as KernelFn,
    );
    register(
        "add_scalar",
        DispatchKey::Wgpu,
        add_scalar_kernel as KernelFn,
    );
    register(
        "sub_scalar",
        DispatchKey::Wgpu,
        sub_scalar_kernel as KernelFn,
    );
    register(
        "div_scalar",
        DispatchKey::Wgpu,
        div_scalar_kernel as KernelFn,
    );
    register("transpose", DispatchKey::Wgpu, transpose_kernel as KernelFn);
}
