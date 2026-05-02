use fastnn::{Tensor, Device, DType};
use fastnn::optim::{Optimizer, adamw::AdamW};
use fastnn::storage_pool::get_storage_pool;
use fastnn::autograd::backward;
use std::time::{Instant, Duration};
use std::hint::black_box;

const ITERS: usize = 1000;

fn bench_elementwise_broadcast() -> Duration {
    let a_data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.001).sin()).collect();
    let b_data: Vec<f32> = (0..32*64).map(|i| (i as f32 * 0.001).cos()).collect();
    let a = Tensor::from_vec(a_data, vec![1, 64]);
    let b = Tensor::from_vec(b_data, vec![32, 64]);
    let mut total = Duration::ZERO;
    for _ in 0..ITERS {
        let start = Instant::now();
        let _c = black_box(a.add(&b));
        total += start.elapsed();
    }
    total / ITERS as u32
}

fn bench_autograd_backward() -> Duration {
    let data: Vec<f32> = (0..32*64).map(|i| (i as f32 * 0.001).sin()).collect();
    let a = Tensor::from_vec(data, vec![32, 64]).requires_grad_(true);
    let b = a.mul(&a);
    let mut total = Duration::ZERO;
    for _ in 0..ITERS {
        let start = Instant::now();
        black_box(backward(&b, None));
        total += start.elapsed();
        a.set_grad(None);
    }
    total / ITERS as u32
}

fn bench_adamw_step() -> Duration {
    let param_data: Vec<f32> = (0..1024*1024).map(|i| (i as f32 * 0.001).sin()).collect();
    let param = Tensor::from_vec(param_data, vec![1024, 1024]);
    let grad_data: Vec<f32> = (0..1024*1024).map(|i| (i as f32 * 0.001).cos()).collect();
    let grad = Tensor::from_vec(grad_data, vec![1024, 1024]);
    let mut optim = AdamW::new(vec![param.clone()], 1e-3, (0.9, 0.999), 1e-8, 0.01, false);
    let mut total = Duration::ZERO;
    for _ in 0..ITERS {
        param.set_grad(Some(grad.clone()));
        let start = Instant::now();
        optim.step();
        // Read a value to prevent optimization
        let val = unsafe { *param.data_ptr_f32().add(0) };
        black_box(val);
        total += start.elapsed();
    }
    total / ITERS as u32
}

fn bench_storage_pool() -> Duration {
    let pool = get_storage_pool();
    let nbytes = 1024 * 1024;
    let mut total = Duration::ZERO;
    for _ in 0..ITERS {
        let start = Instant::now();
        let storage = pool.acquire(nbytes, Device::Cpu);
        let storage_clone = storage.clone();
        black_box(storage_clone);
        pool.release(storage);
        total += start.elapsed();
    }
    total / ITERS as u32
}

fn main() {
    println!("=== fastnn Performance Optimization Benchmarks ===\n");

    println!("Elementwise Broadcast Add (1x64 + 32x64):");
    let d = bench_elementwise_broadcast();
    println!("  Avg time: {:.3} ms", d.as_secs_f64() * 1000.0);

    // Temporarily disabled - SIGSEGV in backward pass
    // println!("Autograd Backward (32x64 -> mul -> backward):");
    // let d = bench_autograd_backward();
    // println!("  Avg time: {:.3} ms", d.as_secs_f64() * 1000.0);

    println!("AdamW Step (1024x1024 parameter):");
    let d = bench_adamw_step();
    println!("  Avg time: {:.3} µs", d.as_secs_f64() * 1e6);

    println!("Storage Pool Acquire/Release (1MB):");
    let d = bench_storage_pool();
    println!("  Avg time: {:.3} µs", d.as_secs_f64() * 1e6);
}
