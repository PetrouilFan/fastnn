//! Benchmark: compiled training step time.
//!
//! Measures per-step latency for a small MLP (64->128->32->1) with AdamW
//! via the AOT compiled training pipeline (CompiledTrainingModel::train_step).
//!
//! Naive (eager) training comparison is TBD — the Tensor autograd backward
//! pass does not yet handle this multi-layer graph structure correctly.

use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::GraphExecutor;
use fastnn::compiler::passes::training::{OptimizerConfig, TrainConfig};
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::IrDType;
use std::hint::black_box;
use std::time::Instant;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

fn main() {
    println!("=== Compiled Training vs Naive Training Benchmarks ===\n");

    // Build a 3-layer MLP: input(64) -> linear(64,128) -> relu -> linear(128,32) -> linear(32,1)
    let g = GraphBuilder::new();
    let x = g.input(&[1, 64], IrDType::F32);
    let targets = g.input(&[1, 1], IrDType::F32);
    let w1 = g.parameter(&[64, 128], IrDType::F32);
    let b1 = g.parameter(&[128], IrDType::F32);
    let w2 = g.parameter(&[128, 32], IrDType::F32);
    let b2 = g.parameter(&[32], IrDType::F32);
    let w3 = g.parameter(&[32, 1], IrDType::F32);
    let b3 = g.parameter(&[1], IrDType::F32);

    let mm1 = g.matmul(&x, &w1);
    let h1_pre = g.add(&mm1, &b1);
    let h1 = g.relu(&h1_pre);

    let mm2 = g.matmul(&h1, &w2);
    let h2_pre = g.add(&mm2, &b2);
    let h2 = g.relu(&h2_pre);

    let mm3 = g.matmul(&h2, &w3);
    let logits = g.add(&mm3, &b3);

    // MSE loss: mean((logits - targets)^2)
    let diff = g.sub(&logits, &targets);
    let sq = g.mul(&diff, &diff);
    let loss_tmp = g.reduce_mean(&sq, 0, false);
    let loss = g.reduce_mean(&loss_tmp, 0, false);

    let graph = g.to_graph();
    let loss_id = loss.node_id();
    let param_ids = vec![
        w1.node_id(),
        b1.node_id(),
        w2.node_id(),
        b2.node_id(),
        w3.node_id(),
        b3.node_id(),
    ];
    let batch_input_ids = vec![x.node_id(), targets.node_id()];

    // Data
    let x_data = f32_bytes(&vec![0.5f32; 64]);
    let t_data = f32_bytes(&vec![1.0f32; 1]);

    // Parameter initial values (small random)
    let rng_data = |n: usize| -> Vec<u8> {
        f32_bytes(
            &(0..n)
                .map(|i| (i as f32 * 0.01).sin() * 0.1)
                .collect::<Vec<_>>(),
        )
    };
    let w1_data = rng_data(64 * 128);
    let b1_data = f32_bytes(&vec![0.0f32; 128]);
    let w2_data = rng_data(128 * 32);
    let b2_data = f32_bytes(&vec![0.0f32; 32]);
    let w3_data = rng_data(32 * 1);
    let b3_data = f32_bytes(&vec![0.0f32; 1]);

    // ═══════════════════════════════════════════════════════════════════════
    // Compiled Training (AOT)
    // ═══════════════════════════════════════════════════════════════════════
    println!("--- Compiled Training (AOT, AdamW) ---");

    let executor = GraphExecutor::new(CpuBackend);
    let mut model = executor
        .compile_train(
            &graph,
            loss_id,
            &param_ids,
            &[&w1_data, &b1_data, &w2_data, &b2_data, &w3_data, &b3_data],
            &batch_input_ids,
            None,
            &TrainConfig {
                optimizer: OptimizerConfig::AdamW {
                    lr: 0.001,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.01,
                },
                quantize: None,
            },
        )
        .expect("compile_train should succeed");

    let batch = vec![x_data.clone(), t_data.clone()];
    let batch_refs: Vec<&[u8]> = batch.iter().map(|v| &v[..]).collect();

    // Warmup
    for _ in 0..10 {
        model.train_step(&batch_refs).unwrap();
    }

    let iters = 200;
    let start = Instant::now();
    for _ in 0..iters {
        black_box(model.train_step(&batch_refs)).unwrap();
    }
    let compiled_ns = start.elapsed().as_nanos() as f64 / iters as f64;
    let compiled_ms = compiled_ns / 1_000_000.0;

    println!(
        "  Mean step time: {:>10.4} ms  ({:>8.2} steps/sec)",
        compiled_ms,
        1000.0 / compiled_ms
    );

    println!();
    println!("=== Comparison ===");
    println!(
        "  {:<30} {:>12} {:>10}",
        "Method", "Mean Step Time", "vs Naive"
    );
    println!(
        "  {:<30} {:>10.4} ms  {:>6.2}x",
        "Compiled (AOT)", compiled_ms, 1.0
    );
    println!("  Naive (eager)             TBD (autograd chain needs fix)");
    println!();
}
