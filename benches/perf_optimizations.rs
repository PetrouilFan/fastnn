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

fn bench_autograd_chain() -> Duration {
    let a_data: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
    let a = Tensor::from_vec(a_data, vec![128]).requires_grad_(true);
    let mut total = Duration::ZERO;
    for _ in 0..ITERS {
        let x = a.clone();
        let y = black_box(x.clone() + x.clone());
        let start = Instant::now();
        backward(&y, None);
        total += start.elapsed();
    }
    total / ITERS as u32
}

fn main() {
    println!("=== fastnn Performance Optimization Benchmarks ===\n");

    println!("Elementwise Broadcast Add (1x64 + 32x64):");
    let d = bench_elementwise_broadcast();
    println!("  Avg time: {:.3} ms", d.as_secs_f64() * 1000.0);

    println!("\nAutograd Chain (add.backward):");
    let d = bench_autograd_chain();
    println!("  Avg time: {:.3} ms", d.as_secs_f64() * 1000.0);
}