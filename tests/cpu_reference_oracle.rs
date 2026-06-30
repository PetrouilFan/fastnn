use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::GraphExecutor;
use fastnn::backend::Instruction;
use fastnn::compiler::passes::activation_quantization;
use fastnn::compiler::passes::shape_inference;
use fastnn::dtypes::{F4x8, F8x4, F8x4R, PackedWord};
use fastnn::ir::builder::{GraphBuilder, GraphTensor};
use fastnn::ir::node::{ComputeGraph, DimExpr, IrDType, TensorType};
use fastnn::packed_tensor::PackedTensor;
use half::f16;

#[derive(Clone, Copy)]
struct Tolerance {
    abs: f32,
    rel: f32,
}

fn seeded_values(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    let mut values = Vec::with_capacity(len);
    for i in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let centered = ((state >> 32) as i32 % 2049 - 1024) as f32 / 256.0;
        let bias = match i % 5 {
            0 => -0.75,
            1 => -0.25,
            2 => 0.0,
            3 => 0.25,
            _ => 0.75,
        };
        values.push(centered + bias);
    }
    values
}

fn naive_matmul(
    activations: &[f32],
    weights: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; batch * m * n];
    for batch_idx in 0..batch {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for depth in 0..k {
                    let a_idx = batch_idx * m * k + row * k + depth;
                    let w_idx = depth * n + col;
                    acc += activations[a_idx] * weights[w_idx];
                }
                out[batch_idx * m * n + row * n + col] = acc;
            }
        }
    }
    out
}

fn naive_broadcast_add(
    lhs: &[f32],
    rhs: &[f32],
    batch: usize,
    channels: usize,
    width: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; lhs.len()];
    for b in 0..batch {
        for c in 0..channels {
            for x in 0..width {
                let idx = b * channels * width + c * width + x;
                out[idx] = lhs[idx] + rhs[x];
            }
        }
    }
    out
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tol: Tolerance) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        let allowed = tol.abs.max(want.abs() * tol.rel);
        let err = (got - want).abs();
        assert!(
            err <= allowed,
            "{label}[{idx}] mismatch: got {got}, expected {want}, err {err}, allowed {allowed}"
        );
    }
}

fn graph_from(builder: &GraphBuilder, output: &GraphTensor) -> ComputeGraph {
    let mut graph = builder.to_graph();
    graph.inputs = builder.recorded_input_ids();
    graph.outputs = vec![output.node_id()];
    graph
}

fn kernel_names(plan: &fastnn::backend::ExecutablePlan) -> Vec<String> {
    plan.instructions
        .iter()
        .filter_map(|instruction| match instruction {
            Instruction::CallKernel { kernel_name, .. } => Some(kernel_name.clone()),
            _ => None,
        })
        .collect()
}

fn run_single_output_f32(
    graph: ComputeGraph,
    inputs: &[&[u8]],
    quantize: Option<u8>,
) -> (Vec<f32>, Vec<String>) {
    let mut executor = GraphExecutor::new(CpuBackend);
    let (mut plan, memory_plan, compiled_graph) = executor
        .compile_with_plan_and_quantize(graph, quantize, None)
        .expect("graph should compile");
    let names = kernel_names(&plan);
    let outputs = executor
        .execute(&compiled_graph, &mut plan, &memory_plan, inputs)
        .expect("graph should execute");
    let values: Vec<f32> = bytemuck::cast_slice(&outputs[0]).to_vec();
    (values, names)
}

fn run_single_output_f32_with_i8_activations(
    mut graph: ComputeGraph,
    inputs: &[&[u8]],
    quantize: u8,
) -> (Vec<f32>, Vec<String>) {
    shape_inference::infer_shapes(&mut graph).expect("shape inference should succeed");
    activation_quantization::quantize_activations(&mut graph)
        .expect("activation quantization rewrite should succeed");
    run_single_output_f32(graph, inputs, Some(quantize))
}

#[test]
fn matmul_f32_reference_oracle_covers_batched_gemv_and_gemm() {
    let cases = [
        (7_u64, 1_usize, 1_usize, 7_usize, 5_usize),
        (11_u64, 3_usize, 2_usize, 5_usize, 4_usize),
    ];

    for (seed, batch, m, k, n) in cases {
        let builder = GraphBuilder::new();
        let input = builder.input_with_dims(
            &[
                DimExpr::Known(batch as u64),
                DimExpr::Known(m as u64),
                DimExpr::Known(k as u64),
            ],
            IrDType::F32,
        );
        let weights = seeded_values(seed + 1, k * n);
        let weight = builder.constant(
            bytemuck::cast_slice(&weights),
            TensorType::new(
                vec![DimExpr::Known(k as u64), DimExpr::Known(n as u64)],
                IrDType::F32,
            ),
        );
        let output = builder.matmul(&input, &weight);
        let graph = graph_from(&builder, &output);

        let activations = seeded_values(seed, batch * m * k);
        let input_bytes = bytemuck::cast_slice(&activations).to_vec();
        let (actual, kernels) = run_single_output_f32(graph, &[&input_bytes], None);
        let expected = naive_matmul(&activations, &weights, batch, m, k, n);

        assert!(
            kernels.iter().any(|name| name == "matmul"),
            "expected matmul kernel, got {kernels:?}"
        );
        assert_close(
            &format!("f32 matmul seed={seed} shape=[{batch},{m},{k}]x[{k},{n}]"),
            &actual,
            &expected,
            Tolerance {
                abs: 1e-5,
                rel: 1e-5,
            },
        );
    }
}

#[test]
fn matmul_u4_reference_oracle_hits_quantized_dispatch_on_tail_k() {
    let cases = [
        (23_u64, 2_usize, 7_usize, 5_usize),
        (29_u64, 3_usize, 9_usize, 4_usize),
    ];

    for (seed, m, k, n) in cases {
        let builder = GraphBuilder::new();
        let input = builder.input_with_dims(
            &[DimExpr::Known(m as u64), DimExpr::Known(k as u64)],
            IrDType::F32,
        );
        let weights = seeded_values(seed + 1, k * n);
        let weight = builder.constant(
            bytemuck::cast_slice(&weights),
            TensorType::new(
                vec![DimExpr::Known(k as u64), DimExpr::Known(n as u64)],
                IrDType::F32,
            ),
        );
        let output = builder.matmul(&input, &weight);
        let graph = graph_from(&builder, &output);

        let activations = seeded_values(seed, m * k);
        let input_bytes = bytemuck::cast_slice(&activations).to_vec();
        let (actual, kernels) = run_single_output_f32(graph, &[&input_bytes], Some(4));
        let expected = naive_matmul(&activations, &weights, 1, m, k, n);

        assert!(
            kernels.iter().any(|name| name == "matmul_i4"),
            "expected matmul_u4 kernel, got {kernels:?}"
        );
        assert_close(
            &format!("u4 matmul seed={seed} shape=[{m},{k}]x[{k},{n}]"),
            &actual,
            &expected,
            Tolerance {
                abs: 0.45,
                rel: 0.40,
            },
        );
    }
}

#[test]
fn matmul_u8_reference_oracle_hits_quantized_dispatch_on_tail_k() {
    let cases = [
        (41_u64, 2_usize, 7_usize, 5_usize),
        (43_u64, 4_usize, 9_usize, 3_usize),
    ];

    for (seed, m, k, n) in cases {
        let builder = GraphBuilder::new();
        let input = builder.input_with_dims(
            &[DimExpr::Known(m as u64), DimExpr::Known(k as u64)],
            IrDType::F32,
        );
        let weights = seeded_values(seed + 1, k * n);
        let weight = builder.constant(
            bytemuck::cast_slice(&weights),
            TensorType::new(
                vec![DimExpr::Known(k as u64), DimExpr::Known(n as u64)],
                IrDType::F32,
            ),
        );
        let output = builder.matmul(&input, &weight);
        let graph = graph_from(&builder, &output);

        let activations = seeded_values(seed, m * k);
        let input_bytes = bytemuck::cast_slice(&activations).to_vec();
        let (actual, kernels) = run_single_output_f32(graph, &[&input_bytes], Some(8));
        let expected = naive_matmul(&activations, &weights, 1, m, k, n);

        assert!(
            kernels.iter().any(|name| name == "matmul_i8"),
            "expected matmul_u8 kernel, got {kernels:?}"
        );
        assert_close(
            &format!("u8 matmul seed={seed} shape=[{m},{k}]x[{k},{n}]"),
            &actual,
            &expected,
            Tolerance {
                abs: 0.12,
                rel: 0.10,
            },
        );
    }
}

#[test]
fn matmul_i8_activation_path_matches_reference_oracle() {
    let seed = 59_u64;
    let m = 3_usize;
    let k = 7_usize;
    let n = 4_usize;

    let builder = GraphBuilder::new();
    let input = builder.input_with_dims(
        &[DimExpr::Known(m as u64), DimExpr::Known(k as u64)],
        IrDType::F32,
    );
    let zero_bias = builder.constant(
        bytemuck::cast_slice(&vec![0.0f32; m * k]),
        TensorType::new(
            vec![DimExpr::Known(m as u64), DimExpr::Known(k as u64)],
            IrDType::F32,
        ),
    );
    let shifted = builder.add(&input, &zero_bias);
    let weights: Vec<f32> = seeded_values(seed + 1, k * n)
        .into_iter()
        .map(|value| value * 0.2)
        .collect();
    let weight = builder.constant(
        bytemuck::cast_slice(&weights),
        TensorType::new(
            vec![DimExpr::Known(k as u64), DimExpr::Known(n as u64)],
            IrDType::F32,
        ),
    );
    let output = builder.matmul(&shifted, &weight);
    let graph = graph_from(&builder, &output);

    let activations: Vec<f32> = seeded_values(seed, m * k)
        .into_iter()
        .map(|value| value * 0.2)
        .collect();
    let input_bytes = bytemuck::cast_slice(&activations).to_vec();
    let (actual, kernels) = run_single_output_f32_with_i8_activations(graph, &[&input_bytes], 8);
    let expected = naive_matmul(&activations, &weights, 1, m, k, n);

    assert!(
        kernels.iter().any(|name| name == "quantize_activations"),
        "expected quantize_activations kernel, got {kernels:?}"
    );
    assert!(
        kernels.iter().any(|name| name == "matmul_i8_i8"),
        "expected matmul_u8_i8 kernel, got {kernels:?}"
    );
    assert_close(
        "i8 activation matmul",
        &actual,
        &expected,
        Tolerance {
            abs: 0.16,
            rel: 0.12,
        },
    );
}

#[test]
fn matmul_u4_i8_activation_path_matches_reference_oracle() {
    let seed = 97_u64;
    let m = 2_usize;
    let k = 9_usize;
    let n = 3_usize;

    let builder = GraphBuilder::new();
    let input = builder.input_with_dims(
        &[DimExpr::Known(m as u64), DimExpr::Known(k as u64)],
        IrDType::F32,
    );
    let zero_bias = builder.constant(
        bytemuck::cast_slice(&vec![0.0f32; m * k]),
        TensorType::new(
            vec![DimExpr::Known(m as u64), DimExpr::Known(k as u64)],
            IrDType::F32,
        ),
    );
    let shifted = builder.add(&input, &zero_bias);
    let weights: Vec<f32> = seeded_values(seed + 1, k * n)
        .into_iter()
        .map(|value| value * 0.11)
        .collect();
    let weight = builder.constant(
        bytemuck::cast_slice(&weights),
        TensorType::new(
            vec![DimExpr::Known(k as u64), DimExpr::Known(n as u64)],
            IrDType::F32,
        ),
    );
    let output = builder.matmul(&shifted, &weight);
    let graph = graph_from(&builder, &output);

    let activations: Vec<f32> = seeded_values(seed, m * k)
        .into_iter()
        .map(|value| value * 0.14 + 0.35)
        .collect();
    let input_bytes = bytemuck::cast_slice(&activations).to_vec();
    let (actual, kernels) = run_single_output_f32_with_i8_activations(graph, &[&input_bytes], 4);
    let expected = naive_matmul(&activations, &weights, 1, m, k, n);

    assert!(
        kernels.iter().any(|name| name == "quantize_activations"),
        "expected quantize_activations kernel, got {kernels:?}"
    );
    assert!(
        kernels.iter().any(|name| name == "matmul_i4_i8"),
        "expected matmul_u4_i8 kernel, got {kernels:?}"
    );
    assert_close(
        "u4 i8 activation matmul",
        &actual,
        &expected,
        Tolerance {
            abs: 0.14,
            rel: 0.12,
        },
    );
}

#[test]
fn f16_roundtrip_reference_oracle_uses_cpu_conversion_kernels() {
    let seed = 71_u64;
    let len = 19_usize;

    let builder = GraphBuilder::new();
    let input = builder.input_with_dims(&[DimExpr::Known(len as u64)], IrDType::F32);
    let output = builder.to_f32(&builder.to_f16(&input));
    let graph = graph_from(&builder, &output);

    let source = seeded_values(seed, len);
    let input_bytes = bytemuck::cast_slice(&source).to_vec();
    let (actual, kernels) = run_single_output_f32(graph, &[&input_bytes], None);
    let expected: Vec<f32> = source
        .iter()
        .map(|&value| f16::from_f32(value).to_f32())
        .collect();

    assert!(
        kernels.iter().any(|name| name == "to_f16"),
        "expected to_f16 kernel, got {kernels:?}"
    );
    assert!(
        kernels.iter().any(|name| name == "to_f32"),
        "expected to_f32 kernel, got {kernels:?}"
    );
    assert_eq!(
        actual, expected,
        "f16 roundtrip should match reference conversion exactly"
    );
}

#[test]
fn broadcast_add_reference_oracle_covers_negative_values() {
    let seed = 83_u64;
    let batch = 2_usize;
    let channels = 3_usize;
    let width = 5_usize;

    let builder = GraphBuilder::new();
    let lhs = builder.input_with_dims(
        &[
            DimExpr::Known(batch as u64),
            DimExpr::Known(channels as u64),
            DimExpr::Known(width as u64),
        ],
        IrDType::F32,
    );
    let rhs_values = seeded_values(seed + 1, width);
    let rhs = builder.constant(
        bytemuck::cast_slice(&rhs_values),
        TensorType::new(vec![DimExpr::Known(width as u64)], IrDType::F32),
    );
    let output = builder.add(&lhs, &rhs);
    let graph = graph_from(&builder, &output);

    let lhs_values = seeded_values(seed, batch * channels * width);
    let input_bytes = bytemuck::cast_slice(&lhs_values).to_vec();
    let (actual, kernels) = run_single_output_f32(graph, &[&input_bytes], None);
    let expected = naive_broadcast_add(&lhs_values, &rhs_values, batch, channels, width);

    assert!(
        kernels.iter().any(|name| name == "add_f32"),
        "expected add_f32 kernel, got {kernels:?}"
    );
    assert_close(
        "broadcast add",
        &actual,
        &expected,
        Tolerance {
            abs: 1e-6,
            rel: 1e-6,
        },
    );
}

fn run_fp_matmul_test<T, F>(
    seed: u64,
    m: usize,
    k: usize,
    n: usize,
    make_dtype: F,
    expected_kernel: &str,
    tol: Tolerance,
) where
    T: PackedWord + 'static,
    F: Fn(Vec<f32>) -> IrDType,
{
    let builder = GraphBuilder::new();
    let weights = seeded_values(seed + 1, k * n);

    let mut transposed = vec![0.0f32; k * n];
    for r in 0..k {
        for c in 0..n {
            transposed[c * k + r] = weights[r * n + c];
        }
    }
    let pt = PackedTensor::<T>::from_f32_per_channel(&transposed, &[n, k]);
    let scales: Vec<f32> = (0..n).map(|r| pt.scale_for_row(r)).collect();
    let weight = builder.constant(
        pt.as_bytes(),
        TensorType::new(
            vec![DimExpr::Known(k as u64), DimExpr::Known(n as u64)],
            make_dtype(scales),
        ),
    );
    let input = builder.input_with_dims(
        &[DimExpr::Known(m as u64), DimExpr::Known(k as u64)],
        IrDType::F32,
    );
    let output = builder.matmul(&input, &weight);
    let graph = graph_from(&builder, &output);

    let activations = seeded_values(seed, m * k);
    let input_bytes = bytemuck::cast_slice(&activations).to_vec();
    let (actual, kernels) = run_single_output_f32(graph, &[&input_bytes], None);
    let expected = naive_matmul(&activations, &weights, 1, m, k, n);

    assert!(
        kernels.iter().any(|name| name == expected_kernel),
        "expected {expected_kernel} kernel, got {kernels:?}"
    );
    assert_close(
        &format!("{expected_kernel} matmul seed={seed} shape=[{m},{k}]x[{k},{n}]"),
        &actual,
        &expected,
        tol,
    );
}

#[test]
fn matmul_f4_reference_oracle() {
    for (seed, m, k, n) in [(23u64, 2usize, 7, 5), (29u64, 3usize, 9, 4)] {
        run_fp_matmul_test::<F4x8, _>(
            seed,
            m,
            k,
            n,
            |scales| IrDType::F4 { scales },
            "matmul_f4",
            Tolerance {
                abs: 2.0,
                rel: 0.25,
            },
        );
    }
}

#[test]
fn matmul_f8_reference_oracle() {
    for (seed, m, k, n) in [(23u64, 2usize, 7, 5), (29u64, 3usize, 9, 4)] {
        run_fp_matmul_test::<F8x4, _>(
            seed,
            m,
            k,
            n,
            |scales| IrDType::F8 { scales },
            "matmul_f8",
            Tolerance {
                abs: 0.20,
                rel: 0.10,
            },
        );
    }
}

#[test]
fn matmul_f8r_reference_oracle() {
    for (seed, m, k, n) in [(23u64, 2usize, 7, 5), (29u64, 3usize, 9, 4)] {
        run_fp_matmul_test::<F8x4R, _>(
            seed,
            m,
            k,
            n,
            |scales| IrDType::F8R { scales },
            "matmul_f8r",
            Tolerance {
                abs: 0.50,
                rel: 0.15,
            },
        );
    }
}
