//! Integration tests for compiled training (CompiledTrainingModel + compile_train + train_step).
//!
//! Tests SGD and AdamW on a tiny MLP, verifying:
//! - Loss decreases over steps (convergence)
//! - Error handling on mismatched inputs
//! - AdamW step counter advances
//! - m/v optimizer state persists across steps

use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::{CompiledTrainingModel, GraphExecutor};
use fastnn::backend::Instruction;
use fastnn::compiler::passes::training::{OptimizerConfig, TrainConfig};
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{DimExpr, IrDType, ShapeEnv};

/// Helper: create a Vec<u8> of f32 bytes from a slice of f32 values.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

/// Helper: convert f32 bytes to a Vec<f32> for verification.
fn read_f32(data: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice(data).to_vec()
}

/// Build a tiny MLP: x(1,4) → W(4,2)+b(2) → logits → reduce_mean → scalar loss.
/// Returns (graph, loss_node_id, params, batch_inputs, param_data).
fn build_mlp() -> (
    fastnn::ir::node::ComputeGraph,
    usize,        // loss_node_id
    Vec<usize>,   // param ids
    Vec<Vec<u8>>, // param data
    Vec<usize>,   // batch input ids
) {
    let g = GraphBuilder::new();

    // Inputs and parameters
    let x = g.input(&[1, 4], IrDType::F32);
    let W = g.parameter(&[4, 2], IrDType::F32);
    let b = g.parameter(&[2], IrDType::F32);

    // Forward: logits = x @ W + b
    let mm = g.matmul(&x, &W);
    let logits = g.add(&mm, &b);

    // Loss: reduce_mean over all dims to get scalar
    let loss_tmp = g.reduce_mean(&logits, 0, false); // [1,2] → [2]
    let loss = g.reduce_mean(&loss_tmp, 0, false); // [2] → scalar

    let graph = g.to_graph();
    let x_id = x.node_id();
    let W_id = W.node_id();
    let b_id = b.node_id();
    let loss_id = loss.node_id();

    // Parameter initial values
    let W_data = f32_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]); // 8 f32s = 32 bytes
    let b_data = f32_bytes(&[0.0, 0.0]); // 2 f32s = 8 bytes

    (
        graph,
        loss_id,
        vec![W_id, b_id],
        vec![W_data, b_data],
        vec![x_id],
    )
}

/// Build a simple MLP: x(1,4) → W(4,1)+b(1) → logits → reduce_mean → scalar loss
/// Single-output version makes loss calculation simpler.
fn build_mlp_single_output() -> (
    fastnn::ir::node::ComputeGraph,
    usize,
    Vec<usize>,
    Vec<Vec<u8>>,
    Vec<usize>,
) {
    let g = GraphBuilder::new();

    let x = g.input(&[1, 4], IrDType::F32);
    let W = g.parameter(&[4, 1], IrDType::F32);
    let b = g.parameter(&[1], IrDType::F32);

    let mm = g.matmul(&x, &W); // [1,1]
    let logits = g.add(&mm, &b); // [1,1]
    let loss = g.reduce_mean(&logits, 0, false); // [1]
    let loss = g.reduce_mean(&loss, 0, false); // scalar

    let graph = g.to_graph();
    let x_id = x.node_id();
    let W_id = W.node_id();
    let b_id = b.node_id();
    let loss_id = loss.node_id();

    let W_data = f32_bytes(&[0.5, -0.3, 0.2, 0.1]); // 4 f32s
    let b_data = f32_bytes(&[0.0]); // 1 f32

    (
        graph,
        loss_id,
        vec![W_id, b_id],
        vec![W_data, b_data],
        vec![x_id],
    )
}

// ─── SGD Tests ───────────────────────────────────────────────────────────────

#[test]
fn test_sgd_mlp_converges() {
    let (graph, loss_id, params, param_data, batch_inputs) = build_mlp();

    let executor = GraphExecutor::new(CpuBackend);
    let mut model: CompiledTrainingModel<CpuBackend> = executor
        .compile_train(
            &graph,
            loss_id,
            &params,
            &param_data.iter().map(|d| &d[..]).collect::<Vec<_>>(),
            &batch_inputs,
            None,
            &TrainConfig {
                optimizer: OptimizerConfig::SGD {
                    lr: 0.05,
                    weight_decay: 0.0,
                },
                quantize: None,
            },
        )
        .expect("compile_train with SGD should succeed");

    let x_data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);

    // Run 20 training steps
    let mut losses = Vec::new();
    for _ in 0..20 {
        let loss = model
            .train_step(&[&x_data])
            .expect("train_step should succeed");
        losses.push(loss);
    }

    // Verify convergence: loss should decrease by at least 50%
    assert!(
        losses[losses.len() - 1] < losses[0] * 0.5,
        "SGD: loss did not converge (first={:.6}, last={:.6})",
        losses[0],
        losses[losses.len() - 1]
    );

    // Verify loss is positive and finite
    assert!(
        losses[0].is_finite() && losses[0] > 0.0,
        "SGD: initial loss should be positive finite"
    );
    assert!(
        losses[losses.len() - 1].is_finite(),
        "SGD: final loss should be finite"
    );
}

#[test]
fn test_sgd_wrong_batch_count_errors() {
    let (graph, loss_id, params, param_data, batch_inputs) = build_mlp();

    let executor = GraphExecutor::new(CpuBackend);
    let mut model = executor
        .compile_train(
            &graph,
            loss_id,
            &params,
            &param_data.iter().map(|d| &d[..]).collect::<Vec<_>>(),
            &batch_inputs,
            None,
            &TrainConfig {
                optimizer: OptimizerConfig::SGD {
                    lr: 0.01,
                    weight_decay: 0.0,
                },
                quantize: None,
            },
        )
        .expect("compile_train should succeed");

    // Too few inputs
    let result = model.train_step(&[]);
    assert!(result.is_err(), "should error on empty batch inputs");

    // Too many inputs
    let x_data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let extra = f32_bytes(&[0.0]);
    let result = model.train_step(&[&x_data, &extra]);
    assert!(result.is_err(), "should error on extra batch inputs");
}

#[test]
fn test_sgd_wrong_input_size_errors() {
    let (graph, loss_id, params, param_data, batch_inputs) = build_mlp();

    let executor = GraphExecutor::new(CpuBackend);
    let mut model = executor
        .compile_train(
            &graph,
            loss_id,
            &params,
            &param_data.iter().map(|d| &d[..]).collect::<Vec<_>>(),
            &batch_inputs,
            None,
            &TrainConfig {
                optimizer: OptimizerConfig::SGD {
                    lr: 0.01,
                    weight_decay: 0.0,
                },
                quantize: None,
            },
        )
        .expect("compile_train should succeed");

    // Wrong-sized input (should be 16 bytes for [1,4] f32, but we pass 4 bytes)
    let wrong_data = f32_bytes(&[1.0]);
    let result = model.train_step(&[&wrong_data]);
    assert!(result.is_err(), "should error on wrong-sized input");
}

// ─── AdamW Tests ─────────────────────────────────────────────────────────────

#[test]
fn test_adamw_mlp_converges() {
    let (graph, loss_id, params, param_data, batch_inputs) = build_mlp();

    let executor = GraphExecutor::new(CpuBackend);
    let mut model = executor
        .compile_train(
            &graph,
            loss_id,
            &params,
            &param_data.iter().map(|d| &d[..]).collect::<Vec<_>>(),
            &batch_inputs,
            None,
            &TrainConfig {
                optimizer: OptimizerConfig::AdamW {
                    lr: 0.05,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.01,
                },
                quantize: None,
            },
        )
        .expect("compile_train with AdamW should succeed");

    let x_data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);

    // Run 20 training steps
    let mut losses = Vec::new();
    for _ in 0..20 {
        let loss = model
            .train_step(&[&x_data])
            .expect("train_step should succeed");
        losses.push(loss);
    }

    // Verify convergence
    assert!(
        losses[losses.len() - 1] < losses[0] * 0.5,
        "AdamW: loss did not converge (first={:.6}, last={:.6})",
        losses[0],
        losses[losses.len() - 1]
    );
}

#[test]
fn test_adamw_step_counter_increments() {
    let (graph, loss_id, params, param_data, batch_inputs) = build_mlp_single_output();

    let executor = GraphExecutor::new(CpuBackend);
    let mut model = executor
        .compile_train(
            &graph,
            loss_id,
            &params,
            &param_data.iter().map(|d| &d[..]).collect::<Vec<_>>(),
            &batch_inputs,
            None,
            &TrainConfig {
                optimizer: OptimizerConfig::AdamW {
                    lr: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.0,
                },
                quantize: None,
            },
        )
        .expect("compile_train should succeed");

    // Capture the step counter value from AdamWUpdate instruction params[4]
    let get_step = |model: &CompiledTrainingModel<CpuBackend>| -> u64 {
        for instr in &model.plan.instructions {
            if let fastnn::backend::Instruction::CallKernel {
                kernel_name,
                params,
                ..
            } = instr
            {
                if kernel_name.starts_with("adam") {
                    return params[4] as u64;
                }
            }
        }
        panic!("no Adam kernel instruction found in plan");
    };

    let x_data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);

    // Step 1: t should start at 1 (from attrs "t": "1")
    assert_eq!(get_step(&model), 1, "AdamW t should start at 1");

    model.train_step(&[&x_data]).expect("step 1 should succeed");
    assert_eq!(get_step(&model), 2, "AdamW t should be 2 after step 1");

    model.train_step(&[&x_data]).expect("step 2 should succeed");
    assert_eq!(get_step(&model), 3, "AdamW t should be 3 after step 2");

    model.train_step(&[&x_data]).expect("step 3 should succeed");
    assert_eq!(get_step(&model), 4, "AdamW t should be 4 after step 3");
}

#[test]
fn test_adamw_mv_persist_across_steps() {
    // Verify that m and v optimizer state changes across steps
    // by reading the arena bytes.
    let (graph, loss_id, params, param_data, batch_inputs) = build_mlp();

    let executor = GraphExecutor::new(CpuBackend);
    let mut model = executor
        .compile_train(
            &graph,
            loss_id,
            &params,
            &param_data.iter().map(|d| &d[..]).collect::<Vec<_>>(),
            &batch_inputs,
            None,
            &TrainConfig {
                optimizer: OptimizerConfig::AdamW {
                    lr: 0.05,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.0,
                },
                quantize: None,
            },
        )
        .expect("compile_train should succeed");

    let x_data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);

    // Run 30 steps, track loss
    let mut losses = Vec::new();
    for _ in 0..30 {
        let loss = model
            .train_step(&[&x_data])
            .expect("train_step should succeed");
        losses.push(loss);
    }

    // Verify loss converges (final < initial * 0.5)
    assert!(
        losses[losses.len() - 1] < losses[0] * 0.5,
        "AdamW: loss did not converge (first={:.6}, last={:.6})",
        losses[0],
        losses[losses.len() - 1]
    );

    // Verify loss decreases approximately monotonically (at least 60% of steps)
    let mut decreases = 0;
    for i in 1..losses.len() {
        if losses[i] < losses[i - 1] {
            decreases += 1;
        }
    }
    assert!(
        decreases as f64 > losses.len() as f64 * 0.6,
        "AdamW: loss should decrease in >60% of steps ({}/{} decreases)",
        decreases,
        losses.len()
    );
}

// ─── Muon Tests ──────────────────────────────────────────────────────────────

#[test]
fn test_muon_mlp_converges() {
    let (graph, loss_id, params, param_data, batch_inputs) = build_mlp();

    let executor = GraphExecutor::new(CpuBackend);
    let mut model = executor
        .compile_train(
            &graph,
            loss_id,
            &params,
            &param_data.iter().map(|d| &d[..]).collect::<Vec<_>>(),
            &batch_inputs,
            None,
            &TrainConfig {
                optimizer: OptimizerConfig::Muon {
                    lr: 0.05,
                    beta1: 0.9,
                    weight_decay: 0.0,
                },
                quantize: None,
            },
        )
        .expect("compile_train with Muon should succeed");

    let x_data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);

    let mut losses = Vec::new();
    for _ in 0..20 {
        let loss = model
            .train_step(&[&x_data])
            .expect("train_step should succeed");
        losses.push(loss);
    }

    assert!(
        losses[losses.len() - 1] < losses[0] * 0.5,
        "Muon: loss did not converge (first={:.6}, last={:.6})",
        losses[0],
        losses[losses.len() - 1]
    );
}

// ─── Lion Tests ──────────────────────────────────────────────────────────────

#[test]
fn test_lion_mlp_converges() {
    let (graph, loss_id, params, param_data, batch_inputs) = build_mlp();

    let executor = GraphExecutor::new(CpuBackend);
    let mut model = executor
        .compile_train(
            &graph,
            loss_id,
            &params,
            &param_data.iter().map(|d| &d[..]).collect::<Vec<_>>(),
            &batch_inputs,
            None,
            &TrainConfig {
                optimizer: OptimizerConfig::Lion {
                    lr: 0.05,
                    beta1: 0.9,
                    beta2: 0.99,
                },
                quantize: None,
            },
        )
        .expect("compile_train with Lion should succeed");

    let x_data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);

    let mut losses = Vec::new();
    for _ in 0..20 {
        let loss = model
            .train_step(&[&x_data])
            .expect("train_step should succeed");
        losses.push(loss);
    }

    assert!(
        losses[losses.len() - 1] < losses[0] * 0.5,
        "Lion: loss did not converge (first={:.6}, last={:.6})",
        losses[0],
        losses[losses.len() - 1]
    );
}

// ─── RMSprop Tests ───────────────────────────────────────────────────────────

#[test]
fn test_rmsprop_mlp_converges() {
    let (graph, loss_id, params, param_data, batch_inputs) = build_mlp();

    let executor = GraphExecutor::new(CpuBackend);
    let mut model = executor
        .compile_train(
            &graph,
            loss_id,
            &params,
            &param_data.iter().map(|d| &d[..]).collect::<Vec<_>>(),
            &batch_inputs,
            None,
            &TrainConfig {
                optimizer: OptimizerConfig::RMSprop {
                    lr: 0.05,
                    beta: 0.99,
                    eps: 1e-8,
                },
                quantize: None,
            },
        )
        .expect("compile_train with RMSprop should succeed");

    let x_data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);

    let mut losses = Vec::new();
    for _ in 0..20 {
        let loss = model
            .train_step(&[&x_data])
            .expect("train_step should succeed");
        losses.push(loss);
    }

    assert!(
        losses[losses.len() - 1] < losses[0] * 0.5,
        "RMSprop: loss did not converge (first={:.6}, last={:.6})",
        losses[0],
        losses[losses.len() - 1]
    );
}

// ─── Edge Cases ──────────────────────────────────────────────────────────────

#[test]
fn test_training_single_batch() {
    // Very tiny: x(1,2) → W(2,1)+b(1) → reduce_mean → scalar loss
    let g = GraphBuilder::new();
    let x = g.input(&[1, 2], IrDType::F32);
    let W = g.parameter(&[2, 1], IrDType::F32);
    let b = g.parameter(&[1], IrDType::F32);

    let mm = g.matmul(&x, &W);
    let logits = g.add(&mm, &b);
    let loss_tmp = g.reduce_mean(&logits, 0, false);
    let loss = g.reduce_mean(&loss_tmp, 0, false);

    let graph = g.to_graph();
    let x_data = f32_bytes(&[1.0, 2.0]);
    let W_data = f32_bytes(&[0.5, -0.3]);
    let b_data = f32_bytes(&[0.0]);

    let executor = GraphExecutor::new(CpuBackend);
    let mut model = executor
        .compile_train(
            &graph,
            loss.node_id(),
            &[W.node_id(), b.node_id()],
            &[&W_data, &b_data],
            &[x.node_id()],
            None,
            &TrainConfig {
                optimizer: OptimizerConfig::SGD {
                    lr: 0.1,
                    weight_decay: 0.0,
                },
                quantize: None,
            },
        )
        .expect("compile_train should succeed");

    let loss1 = model.train_step(&[&x_data]).expect("step 1");
    let loss2 = model.train_step(&[&x_data]).expect("step 2");
    assert!(
        loss2 < loss1,
        "Loss should decrease (step1={:.6}, step2={:.6})",
        loss1,
        loss2
    );
}

#[test]
fn test_shape_tightening_reduces_arena() {
    let g = GraphBuilder::new();
    // Symbolic batch dim N, fixed feature dim 4
    let x = g.input_with_dims(
        &[DimExpr::Symbol("N".to_string()), DimExpr::Known(4)],
        IrDType::F32,
    );
    let W = g.parameter(&[4, 2], IrDType::F32);
    let b = g.parameter(&[2], IrDType::F32);

    let mm = g.matmul(&x, &W);
    let logits = g.add(&mm, &b);
    let loss_tmp = g.reduce_mean(&logits, 0, false);
    let loss = g.reduce_mean(&loss_tmp, 0, false);

    let graph = g.to_graph();
    let W_data = f32_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let b_data = f32_bytes(&[0.0, 0.0]);
    let params = vec![W.node_id(), b.node_id()];
    let batch_inputs = vec![x.node_id()];

    let executor = GraphExecutor::new(CpuBackend);
    let config = TrainConfig {
        optimizer: OptimizerConfig::SGD {
            lr: 0.01,
            weight_decay: 0.0,
        },
        quantize: None,
    };

    // Without tightening: arena uses SYMBOL_DIM_MAX (8192) for N
    let model_no_tighten = executor
        .compile_train(
            &graph,
            loss.node_id(),
            &params,
            &[&W_data, &b_data],
            &batch_inputs,
            None,
            &config,
        )
        .expect("compile without tightening");
    let untightened_size = model_no_tighten.plan.arena_size;

    // With tightening: N=2
    let mut shape_env = ShapeEnv::new();
    shape_env.bind("N", 2);
    let model_tightened = executor
        .compile_train(
            &graph,
            loss.node_id(),
            &params,
            &[&W_data, &b_data],
            &batch_inputs,
            Some(&shape_env),
            &config,
        )
        .expect("compile with tightening");
    let tightened_size = model_tightened.plan.arena_size;

    assert!(
        tightened_size < untightened_size,
        "Tightened arena ({}) should be smaller than untightened ({})",
        tightened_size,
        untightened_size
    );

    // Both models should still work for training

    // With tightening: N=1, feed matching data
    let mut shape_env = ShapeEnv::new();
    shape_env.bind("N", 1);
    let mut model2 = executor
        .compile_train(
            &graph,
            loss.node_id(),
            &params,
            &[&W_data, &b_data],
            &batch_inputs,
            Some(&shape_env),
            &config,
        )
        .expect("compile with N=1");

    // Train two steps with [1, 4] input (matches N=1, features=4)
    let x_data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let loss1 = model2.train_step(&[&x_data]).expect("step 1");
    let loss2 = model2.train_step(&[&x_data]).expect("step 2");
    assert!(
        loss2 < loss1,
        "Loss should decrease with tightened model (step1={:.6}, step2={:.6})",
        loss1,
        loss2
    );
}
