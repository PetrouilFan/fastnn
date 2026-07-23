use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fastnn::backend::cpu::microkernels::{
    gemm_cpu, gemm_cpu_flat, gemm_cpu_flat_i8_i4x8, gemv_cpu,
};
use fastnn::backend::cpu::telemetry::{cpu_telemetry_snapshot, reset_cpu_telemetry};
use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::GraphExecutor;
use fastnn::dtypes::{F32x1, I4x8, I8x4, PackedWord};
use fastnn::ir::builder::{GraphBuilder, GraphTensor};
use fastnn::ir::{ComputeGraph, DimExpr, IrDType, TensorType};
use fastnn::packed_tensor::PackedTensor;

#[derive(Clone, Copy)]
struct GemvCase {
    name: &'static str,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Copy)]
struct GemmCase {
    name: &'static str,
    rows: usize,
    cols: usize,
    batch: usize,
}

#[derive(Clone, Copy)]
struct VecCase {
    name: &'static str,
    len: usize,
}

#[derive(Clone, Copy)]
struct ArenaBroadcastCase {
    name: &'static str,
    batch: usize,
    hidden: usize,
}

#[derive(Clone, Copy)]
struct BroadcastCase {
    name: &'static str,
    batch: usize,
    hidden: usize,
}

#[derive(Clone, Copy)]
struct RowwiseCase {
    name: &'static str,
    batch: usize,
    hidden: usize,
}

#[derive(Clone, Copy)]
struct FusionCase {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

fn weight_data(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| (((i % 97) as f32) - 48.0) * 0.03125)
        .collect()
}

fn activation_data(cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|i| (((i % 29) as f32) - 14.0) * 0.0625)
        .collect()
}

fn vector_data(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((((i * 17 + salt) % 101) as f32) - 50.0) * 0.03125)
        .collect()
}

fn graph_from(builder: &GraphBuilder, output: &GraphTensor) -> ComputeGraph {
    let mut graph = builder.to_graph();
    graph.inputs = builder.recorded_input_ids();
    graph.outputs = vec![output.node_id()];
    graph
}

fn batch_inputs(batch: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..batch)
        .map(|batch_idx| {
            (0..cols)
                .map(|col_idx| ((((batch_idx * 13 + col_idx) % 41) as f32) - 20.0) * 0.05)
                .collect()
        })
        .collect()
}

fn i8_activation_payload(values: &[f32]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(8 + values.len());
    write_i8_activation_payload(values, &mut payload);
    payload
}

fn write_i8_activation_payload(values: &[f32], payload: &mut Vec<u8>) {
    let max_abs = values
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    payload.clear();
    payload.reserve(8 + values.len());
    payload.extend_from_slice(&scale.to_le_bytes());
    payload.extend_from_slice(&0.0f32.to_le_bytes());
    payload.extend(values.iter().map(|value| {
        (value / scale)
            .round()
            .clamp(i8::MIN as f32, i8::MAX as f32) as i8 as u8
    }));
}

fn bench_w4_activation_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("w4_activation_paths");
    for &(name, m, k, n) in &[
        ("decode_1x256x256", 1, 256, 256),
        ("prefill_8x256x256", 8, 256, 256),
    ] {
        let weights = weight_data(n, k);
        let f32_weights = PackedTensor::<F32x1>::from_f32_auto(&weights, &[n, k]);
        let packed = PackedTensor::<I4x8>::from_f32_per_channel_asymmetric(&weights, &[n, k]);
        let activations: Vec<f32> = (0..m)
            .flat_map(|batch| vector_data(k, batch * 13))
            .collect();
        let activation_payload = i8_activation_payload(&activations);
        let mut quantized_payload = Vec::with_capacity(activation_payload.len());
        let mut float_output = vec![0.0f32; m * n];
        let mut f32_output = vec![0.0f32; m * n];
        let mut integer_output = vec![0.0f32; m * n];
        group.throughput(Throughput::Elements((m * k * n) as u64));
        group.bench_with_input(BenchmarkId::new("f32_native", name), &(), |b, _| {
            b.iter(|| {
                gemm_cpu_flat(
                    black_box(&f32_weights),
                    black_box(&activations),
                    black_box(&mut f32_output),
                    m,
                    k,
                    n,
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("w4a32_unpack_float", name), &(), |b, _| {
            b.iter(|| {
                gemm_cpu_flat(
                    black_box(&packed),
                    black_box(&activations),
                    black_box(&mut float_output),
                    m,
                    k,
                    n,
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("w4a8_direct_i32", name), &(), |b, _| {
            b.iter(|| {
                gemm_cpu_flat_i8_i4x8(
                    black_box(&packed),
                    black_box(&activation_payload),
                    black_box(&mut integer_output),
                    m,
                    k,
                    n,
                )
            });
        });
        group.bench_with_input(
            BenchmarkId::new("w4a8_quantize_and_i32", name),
            &(),
            |b, _| {
                b.iter(|| {
                    write_i8_activation_payload(
                        black_box(&activations),
                        black_box(&mut quantized_payload),
                    );
                    gemm_cpu_flat_i8_i4x8(
                        black_box(&packed),
                        black_box(&quantized_payload),
                        black_box(&mut integer_output),
                        m,
                        k,
                        n,
                    )
                });
            },
        );
    }
    group.finish();
}

fn naive_gemv(weights: &[f32], activation: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    for row in 0..rows {
        let row_slice = &weights[row * cols..(row + 1) * cols];
        let mut acc = 0.0f32;
        for col in 0..cols {
            acc += row_slice[col] * activation[col];
        }
        output[row] = acc;
    }
}

fn naive_gemm(
    weights: &[f32],
    inputs: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
    rows: usize,
    cols: usize,
) {
    for (batch_idx, input) in inputs.iter().enumerate() {
        let output = &mut outputs[batch_idx];
        for row in 0..rows {
            let row_slice = &weights[row * cols..(row + 1) * cols];
            let mut acc = 0.0f32;
            for col in 0..cols {
                acc += row_slice[col] * input[col];
            }
            output[row] = acc;
        }
    }
}

fn elementwise_binary(a: &[f32], b: &[f32], out: &mut [f32], op: fn(f32, f32) -> f32) {
    for i in 0..out.len() {
        out[i] = op(a[i], b[i]);
    }
}

fn elementwise_scalar(a: &[f32], scalar: f32, out: &mut [f32], op: fn(f32, f32) -> f32) {
    for i in 0..out.len() {
        out[i] = op(a[i], scalar);
    }
}

fn broadcast_add(input: &[f32], bias: &[f32], out: &mut [f32], batch: usize, hidden: usize) {
    for row in 0..batch {
        let base = row * hidden;
        for col in 0..hidden {
            out[base + col] = input[base + col] + bias[col];
        }
    }
}

fn reduce_sum(input: &[f32]) -> f32 {
    input.iter().copied().sum()
}

fn reduce_mean(input: &[f32]) -> f32 {
    reduce_sum(input) / input.len() as f32
}

fn reduce_max(input: &[f32]) -> f32 {
    input.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn rowwise_sum(input: &[f32], out: &mut [f32], batch: usize, hidden: usize) {
    for (row, out_val) in out.iter_mut().enumerate().take(batch) {
        let start = row * hidden;
        *out_val = input[start..start + hidden].iter().copied().sum();
    }
}

fn rowwise_mean(input: &[f32], out: &mut [f32], batch: usize, hidden: usize) {
    rowwise_sum(input, out, batch, hidden);
    for value in out.iter_mut().take(batch) {
        *value /= hidden as f32;
    }
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

fn matmul_bias_activation(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    out: &mut [f32],
    case: FusionCase,
    activation: fn(f32) -> f32,
) {
    for row in 0..case.m {
        for col in 0..case.n {
            let mut acc = bias[col];
            for kk in 0..case.k {
                acc += a[row * case.k + kk] * b[kk * case.n + col];
            }
            out[row * case.n + col] = activation(acc);
        }
    }
}

fn residual_add_norm(
    x: &[f32],
    residual: &[f32],
    bias: &[f32],
    out: &mut [f32],
    batch: usize,
    hidden: usize,
) {
    for row in 0..batch {
        let start = row * hidden;
        let row_out = &mut out[start..start + hidden];
        for col in 0..hidden {
            row_out[col] = x[start + col] + residual[start + col] + bias[col];
        }
        let mean = row_out.iter().copied().sum::<f32>() / hidden as f32;
        let var = row_out
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / hidden as f32;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for v in row_out {
            *v = (*v - mean) * inv_std;
        }
    }
}

fn sgd_update(param: &mut [f32], grad: &[f32], lr: f32, weight_decay: f32) {
    for i in 0..param.len() {
        param[i] -= lr * (grad[i] + weight_decay * param[i]);
    }
}

fn adamw_update(param: &mut [f32], grad: &[f32], m: &mut [f32], v: &mut [f32], t: f32) {
    let lr: f32 = 1e-3;
    let beta1: f32 = 0.9;
    let beta2: f32 = 0.999;
    let eps: f32 = 1e-8;
    let weight_decay: f32 = 0.01;
    let bias_correction1 = 1.0 - beta1.powf(t);
    let bias_correction2 = 1.0 - beta2.powf(t);
    for i in 0..param.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        let m_hat = m[i] / bias_correction1;
        let v_hat = v[i] / bias_correction2;
        param[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * param[i]);
    }
}

fn bench_fastnn_gemv<T: PackedWord>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &'static str,
    case: GemvCase,
) {
    let weights = PackedTensor::<T>::from_f32_auto(
        &weight_data(case.rows, case.cols),
        &[case.rows, case.cols],
    );
    let activation = activation_data(case.cols);
    let mut output = vec![0.0f32; case.rows];

    group.bench_with_input(BenchmarkId::new(label, case.name), &case, |b, _| {
        b.iter(|| {
            gemv_cpu(
                &weights,
                black_box(&activation),
                black_box(output.as_mut_slice()),
            );
            black_box(&output);
        });
    });
}

fn bench_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_gemv");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(20);

    let cases = [
        GemvCase {
            name: "256x256",
            rows: 256,
            cols: 256,
        },
        GemvCase {
            name: "1024x1024",
            rows: 1024,
            cols: 1024,
        },
        GemvCase {
            name: "4096x4096",
            rows: 4096,
            cols: 4096,
        },
    ];

    for case in cases {
        group.throughput(Throughput::Elements((case.rows * case.cols) as u64));

        let weights = weight_data(case.rows, case.cols);
        let activation = activation_data(case.cols);
        let mut output = vec![0.0f32; case.rows];
        group.bench_with_input(
            BenchmarkId::new("baseline_scalar_f32", case.name),
            &case,
            |b, _| {
                b.iter(|| {
                    naive_gemv(
                        black_box(&weights),
                        black_box(&activation),
                        black_box(output.as_mut_slice()),
                        case.rows,
                        case.cols,
                    );
                    black_box(&output);
                });
            },
        );

        bench_fastnn_gemv::<F32x1>(&mut group, "fastnn_f32x1", case);
        bench_fastnn_gemv::<I8x4>(&mut group, "fastnn_i8x4", case);
        bench_fastnn_gemv::<I4x8>(&mut group, "fastnn_i4x8", case);
    }

    group.finish();
}

fn bench_fastnn_gemm<T: PackedWord>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &'static str,
    case: GemmCase,
) {
    let weights = PackedTensor::<T>::from_f32_auto(
        &weight_data(case.rows, case.cols),
        &[case.rows, case.cols],
    );
    let inputs = batch_inputs(case.batch, case.cols);
    let mut outputs = vec![vec![0.0f32; case.rows]; case.batch];

    group.bench_with_input(BenchmarkId::new(label, case.name), &case, |b, _| {
        b.iter(|| {
            gemm_cpu(
                &weights,
                black_box(&inputs),
                black_box(outputs.as_mut_slice()),
            );
            black_box(&outputs);
        });
    });
}

fn bench_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_gemm");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(20);

    let cases = [
        GemmCase {
            name: "batch8_256x256",
            rows: 256,
            cols: 256,
            batch: 8,
        },
        GemmCase {
            name: "batch16_512x512",
            rows: 512,
            cols: 512,
            batch: 16,
        },
    ];

    for case in cases {
        group.throughput(Throughput::Elements(
            (case.batch * case.rows * case.cols) as u64,
        ));

        let weights = weight_data(case.rows, case.cols);
        let inputs = batch_inputs(case.batch, case.cols);
        let mut outputs = vec![vec![0.0f32; case.rows]; case.batch];
        group.bench_with_input(
            BenchmarkId::new("baseline_scalar_f32", case.name),
            &case,
            |b, _| {
                b.iter(|| {
                    naive_gemm(
                        black_box(&weights),
                        black_box(&inputs),
                        black_box(outputs.as_mut_slice()),
                        case.rows,
                        case.cols,
                    );
                    black_box(&outputs);
                });
            },
        );

        bench_fastnn_gemm::<F32x1>(&mut group, "fastnn_f32x1", case);
        bench_fastnn_gemm::<I8x4>(&mut group, "fastnn_i8x4", case);
        bench_fastnn_gemm::<I4x8>(&mut group, "fastnn_i4x8", case);
    }

    group.finish();
}

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_elementwise");
    group.warm_up_time(std::time::Duration::from_millis(300));
    group.measurement_time(std::time::Duration::from_secs(1));
    group.sample_size(10);

    let cases = [
        VecCase {
            name: "1k",
            len: 1_024,
        },
        VecCase {
            name: "64k",
            len: 65_536,
        },
        VecCase {
            name: "1m",
            len: 1_048_576,
        },
    ];
    #[allow(clippy::type_complexity)]
    let binary_ops: [(&str, fn(f32, f32) -> f32); 3] = [
        ("add", |x, y| x + y),
        ("mul", |x, y| x * y),
        ("div", |x, y| x / (y.abs() + 1.0)),
    ];

    for case in cases {
        group.throughput(Throughput::Elements(case.len as u64));
        let a = vector_data(case.len, 3);
        let b = vector_data(case.len, 11);
        let mut out = vec![0.0f32; case.len];
        for (op_name, op) in binary_ops {
            group.bench_with_input(
                BenchmarkId::new(format!("same_shape_{op_name}"), case.name),
                &case,
                |bench, _| {
                    bench.iter(|| {
                        elementwise_binary(black_box(&a), black_box(&b), black_box(&mut out), op);
                        black_box(&out);
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("scalar_{op_name}"), case.name),
                &case,
                |bench, _| {
                    bench.iter(|| {
                        elementwise_scalar(
                            black_box(&a),
                            black_box(0.125),
                            black_box(&mut out),
                            op,
                        );
                        black_box(&out);
                    });
                },
            );
        }
    }

    let broadcast_cases = [
        BroadcastCase {
            name: "batch16_hidden768",
            batch: 16,
            hidden: 768,
        },
        BroadcastCase {
            name: "batch8_hidden4096",
            batch: 8,
            hidden: 4096,
        },
    ];
    for case in broadcast_cases {
        let len = case.batch * case.hidden;
        group.throughput(Throughput::Elements(len as u64));
        let input = vector_data(len, 7);
        let bias = vector_data(case.hidden, 19);
        let mut out = vec![0.0f32; len];
        group.bench_with_input(
            BenchmarkId::new("broadcast_add", case.name),
            &case,
            |bench, _| {
                bench.iter(|| {
                    broadcast_add(
                        black_box(&input),
                        black_box(&bias),
                        black_box(&mut out),
                        case.batch,
                        case.hidden,
                    );
                    black_box(&out);
                });
            },
        );
    }

    group.finish();
}

fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_reductions");
    group.warm_up_time(std::time::Duration::from_millis(300));
    group.measurement_time(std::time::Duration::from_secs(1));
    group.sample_size(10);

    let cases = [
        VecCase {
            name: "1k",
            len: 1_024,
        },
        VecCase {
            name: "64k",
            len: 65_536,
        },
        VecCase {
            name: "1m",
            len: 1_048_576,
        },
    ];
    for case in cases {
        group.throughput(Throughput::Elements(case.len as u64));
        let input = vector_data(case.len, 23);
        group.bench_with_input(BenchmarkId::new("sum_1d", case.name), &case, |bench, _| {
            bench.iter(|| black_box(reduce_sum(black_box(&input))));
        });
        group.bench_with_input(BenchmarkId::new("mean_1d", case.name), &case, |bench, _| {
            bench.iter(|| black_box(reduce_mean(black_box(&input))));
        });
        group.bench_with_input(BenchmarkId::new("max_1d", case.name), &case, |bench, _| {
            bench.iter(|| black_box(reduce_max(black_box(&input))));
        });
    }

    let rowwise_cases = [
        RowwiseCase {
            name: "batch32_hidden768",
            batch: 32,
            hidden: 768,
        },
        RowwiseCase {
            name: "batch32_hidden1024",
            batch: 32,
            hidden: 1024,
        },
        RowwiseCase {
            name: "batch16_hidden4096",
            batch: 16,
            hidden: 4096,
        },
    ];
    for case in rowwise_cases {
        let len = case.batch * case.hidden;
        group.throughput(Throughput::Elements(len as u64));
        let input = vector_data(len, 29);
        let mut out = vec![0.0f32; case.batch];
        group.bench_with_input(
            BenchmarkId::new("rowwise_sum", case.name),
            &case,
            |bench, _| {
                bench.iter(|| {
                    rowwise_sum(
                        black_box(&input),
                        black_box(&mut out),
                        case.batch,
                        case.hidden,
                    );
                    black_box(&out);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("rowwise_mean", case.name),
            &case,
            |bench, _| {
                bench.iter(|| {
                    rowwise_mean(
                        black_box(&input),
                        black_box(&mut out),
                        case.batch,
                        case.hidden,
                    );
                    black_box(&out);
                });
            },
        );
    }

    group.finish();
}

fn bench_fusions(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_fusions");
    group.warm_up_time(std::time::Duration::from_millis(300));
    group.measurement_time(std::time::Duration::from_secs(1));
    group.sample_size(10);

    let cases = [
        FusionCase {
            name: "m16_k256_n256",
            m: 16,
            k: 256,
            n: 256,
        },
        FusionCase {
            name: "m8_k768_n768",
            m: 8,
            k: 768,
            n: 768,
        },
    ];
    for case in cases {
        group.throughput(Throughput::Elements((case.m * case.k * case.n) as u64));
        let a = vector_data(case.m * case.k, 31);
        let b = vector_data(case.k * case.n, 37);
        let bias = vector_data(case.n, 41);
        let mut out = vec![0.0f32; case.m * case.n];
        group.bench_with_input(
            BenchmarkId::new("matmul_bias_relu", case.name),
            &case,
            |bench, _| {
                bench.iter(|| {
                    matmul_bias_activation(
                        black_box(&a),
                        black_box(&b),
                        black_box(&bias),
                        black_box(&mut out),
                        case,
                        relu,
                    );
                    black_box(&out);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("matmul_bias_gelu", case.name),
            &case,
            |bench, _| {
                bench.iter(|| {
                    matmul_bias_activation(
                        black_box(&a),
                        black_box(&b),
                        black_box(&bias),
                        black_box(&mut out),
                        case,
                        gelu,
                    );
                    black_box(&out);
                });
            },
        );
    }

    let norm_cases = [
        RowwiseCase {
            name: "batch16_hidden768",
            batch: 16,
            hidden: 768,
        },
        RowwiseCase {
            name: "batch8_hidden4096",
            batch: 8,
            hidden: 4096,
        },
    ];
    for case in norm_cases {
        let len = case.batch * case.hidden;
        group.throughput(Throughput::Elements(len as u64));
        let x = vector_data(len, 43);
        let residual = vector_data(len, 47);
        let bias = vector_data(case.hidden, 53);
        let mut out = vec![0.0f32; len];
        group.bench_with_input(
            BenchmarkId::new("residual_add_norm", case.name),
            &case,
            |bench, _| {
                bench.iter(|| {
                    residual_add_norm(
                        black_box(&x),
                        black_box(&residual),
                        black_box(&bias),
                        black_box(&mut out),
                        case.batch,
                        case.hidden,
                    );
                    black_box(&out);
                });
            },
        );
    }

    group.finish();
}

fn bench_training_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_training_updates");
    group.warm_up_time(std::time::Duration::from_millis(300));
    group.measurement_time(std::time::Duration::from_secs(1));
    group.sample_size(10);

    let cases = [
        VecCase {
            name: "64k",
            len: 65_536,
        },
        VecCase {
            name: "1m",
            len: 1_048_576,
        },
    ];
    for case in cases {
        group.throughput(Throughput::Elements(case.len as u64));
        let grad = vector_data(case.len, 61);
        group.bench_with_input(BenchmarkId::new("sgd", case.name), &case, |bench, _| {
            let mut param = vector_data(case.len, 59);
            bench.iter(|| {
                sgd_update(
                    black_box(&mut param),
                    black_box(&grad),
                    black_box(1e-3),
                    black_box(0.01),
                );
                black_box(&param);
            });
        });
        group.bench_with_input(BenchmarkId::new("adamw", case.name), &case, |bench, _| {
            let mut param = vector_data(case.len, 67);
            let mut m = vec![0.0f32; case.len];
            let mut v = vec![0.0f32; case.len];
            bench.iter(|| {
                adamw_update(
                    black_box(&mut param),
                    black_box(&grad),
                    black_box(&mut m),
                    black_box(&mut v),
                    black_box(1.0),
                );
                black_box((&param, &m, &v));
            });
        });
    }

    group.finish();
}

fn bench_cpu_arena_telemetry(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_arena_telemetry");
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.measurement_time(std::time::Duration::from_secs(1));
    group.sample_size(10);

    let cases = [
        VecCase {
            name: "1k",
            len: 1_024,
        },
        VecCase {
            name: "64k",
            len: 65_536,
        },
        VecCase {
            name: "1m",
            len: 1_048_576,
        },
    ];

    for case in cases {
        group.throughput(Throughput::Elements(case.len as u64));

        let input = vector_data(case.len, 71);
        let input_bytes = bytemuck::cast_slice(&input).to_vec();
        let builder = GraphBuilder::new();
        let x = builder.input_with_dims(&[DimExpr::Known(case.len as u64)], IrDType::F32);
        let out = builder.relu(&x);
        let graph = graph_from(&builder, &out);
        let mut executor = GraphExecutor::new(CpuBackend);
        let (mut plan, memory_plan, compiled_graph) = executor
            .compile_with_plan_and_quantize(graph, None, None)
            .expect("relu graph should compile");
        group.bench_with_input(
            BenchmarkId::new("relu_zero_copy", case.name),
            &case,
            |bench, _| {
                bench.iter(|| {
                    reset_cpu_telemetry();
                    let outputs = executor
                        .execute(
                            &compiled_graph,
                            &mut plan,
                            &memory_plan,
                            &[black_box(&input_bytes)],
                        )
                        .expect("relu graph should execute");
                    let snapshot = cpu_telemetry_snapshot();
                    assert_eq!(snapshot.arena_temp_copies, 0, "relu snapshot={snapshot:?}");
                    assert_eq!(snapshot.arena_temp_copy_bytes, 0);
                    black_box(outputs);
                });
            },
        );

        let input = vector_data(case.len, 73);
        let input_bytes = bytemuck::cast_slice(&input).to_vec();
        let scalar = 1.25f32;
        let builder = GraphBuilder::new();
        let x = builder.input_with_dims(&[DimExpr::Known(case.len as u64)], IrDType::F32);
        let scalar_node = builder.constant(
            bytemuck::cast_slice(&[scalar]),
            TensorType::new(vec![], IrDType::F32),
        );
        let out = builder.add_scalar(&x, &scalar_node);
        let graph = graph_from(&builder, &out);
        let mut executor = GraphExecutor::new(CpuBackend);
        let (mut plan, memory_plan, compiled_graph) = executor
            .compile_with_plan_and_quantize(graph, None, None)
            .expect("add_scalar graph should compile");
        group.bench_with_input(
            BenchmarkId::new("add_scalar_zero_copy", case.name),
            &case,
            |bench, _| {
                bench.iter(|| {
                    reset_cpu_telemetry();
                    let outputs = executor
                        .execute(
                            &compiled_graph,
                            &mut plan,
                            &memory_plan,
                            &[black_box(&input_bytes)],
                        )
                        .expect("add_scalar graph should execute");
                    let snapshot = cpu_telemetry_snapshot();
                    assert_eq!(
                        snapshot.arena_temp_copies, 0,
                        "add_scalar snapshot={snapshot:?}"
                    );
                    assert_eq!(snapshot.arena_temp_copy_bytes, 0);
                    black_box(outputs);
                });
            },
        );

        let lhs = vector_data(case.len, 79);
        let rhs = vector_data(case.len, 83);
        let lhs_bytes = bytemuck::cast_slice(&lhs).to_vec();
        let rhs_bytes = bytemuck::cast_slice(&rhs).to_vec();
        let builder = GraphBuilder::new();
        let lhs_node = builder.input_with_dims(&[DimExpr::Known(case.len as u64)], IrDType::F32);
        let rhs_node = builder.input_with_dims(&[DimExpr::Known(case.len as u64)], IrDType::F32);
        let out = builder.add(&lhs_node, &rhs_node);
        let graph = graph_from(&builder, &out);
        let mut executor = GraphExecutor::new(CpuBackend);
        let (mut plan, memory_plan, compiled_graph) = executor
            .compile_with_plan_and_quantize(graph, None, None)
            .expect("plain add graph should compile");
        group.bench_with_input(
            BenchmarkId::new("add_plain_zero_copy", case.name),
            &case,
            |bench, _| {
                bench.iter(|| {
                    reset_cpu_telemetry();
                    let outputs = executor
                        .execute(
                            &compiled_graph,
                            &mut plan,
                            &memory_plan,
                            &[black_box(&lhs_bytes), black_box(&rhs_bytes)],
                        )
                        .expect("plain add graph should execute");
                    let snapshot = cpu_telemetry_snapshot();
                    assert_eq!(
                        snapshot.arena_temp_copies, 0,
                        "plain add snapshot={snapshot:?}"
                    );
                    assert_eq!(snapshot.arena_temp_copy_bytes, 0);
                    black_box(outputs);
                });
            },
        );
    }

    let reduce_cases = [
        RowwiseCase {
            name: "batch32_hidden768",
            batch: 32,
            hidden: 768,
        },
        RowwiseCase {
            name: "batch16_hidden4096",
            batch: 16,
            hidden: 4096,
        },
    ];

    for case in reduce_cases {
        let len = case.batch * case.hidden;
        group.throughput(Throughput::Elements(len as u64));

        for op in ["sum", "mean"] {
            let input = vector_data(len, if op == "sum" { 101 } else { 103 });
            let input_bytes = bytemuck::cast_slice(&input).to_vec();
            let builder = GraphBuilder::new();
            let x = builder.input_with_dims(
                &[
                    DimExpr::Known(case.batch as u64),
                    DimExpr::Known(case.hidden as u64),
                ],
                IrDType::F32,
            );
            let out = match op {
                "sum" => builder.reduce_sum(&x, 1, false),
                "mean" => builder.reduce_mean(&x, 1, false),
                _ => unreachable!(),
            };
            let graph = graph_from(&builder, &out);
            let mut executor = GraphExecutor::new(CpuBackend);
            let (mut plan, memory_plan, compiled_graph) = executor
                .compile_with_plan_and_quantize(graph, None, None)
                .expect("reduce graph should compile");
            group.bench_with_input(
                BenchmarkId::new(format!("reduce_{op}_zero_copy"), case.name),
                &case,
                |bench, _| {
                    bench.iter(|| {
                        reset_cpu_telemetry();
                        let outputs = executor
                            .execute(
                                &compiled_graph,
                                &mut plan,
                                &memory_plan,
                                &[black_box(&input_bytes)],
                            )
                            .expect("reduce graph should execute");
                        let snapshot = cpu_telemetry_snapshot();
                        assert_eq!(
                            snapshot.arena_temp_copies, 0,
                            "reduce_{op} snapshot={snapshot:?}"
                        );
                        assert_eq!(snapshot.arena_temp_copy_bytes, 0);
                        black_box(outputs);
                    });
                },
            );
        }
    }

    let broadcast_cases = [
        ArenaBroadcastCase {
            name: "batch16_hidden768",
            batch: 16,
            hidden: 768,
        },
        ArenaBroadcastCase {
            name: "batch8_hidden4096",
            batch: 8,
            hidden: 4096,
        },
    ];

    for case in broadcast_cases {
        let len = case.batch * case.hidden;
        group.throughput(Throughput::Elements(len as u64));

        let lhs = vector_data(len, 89);
        let rhs = vector_data(case.hidden, 97);
        let lhs_bytes = bytemuck::cast_slice(&lhs).to_vec();
        let rhs_bytes = bytemuck::cast_slice(&rhs).to_vec();
        let builder = GraphBuilder::new();
        let lhs_node = builder.input_with_dims(
            &[
                DimExpr::Known(case.batch as u64),
                DimExpr::Known(case.hidden as u64),
            ],
            IrDType::F32,
        );
        let rhs_node = builder.input_with_dims(&[DimExpr::Known(case.hidden as u64)], IrDType::F32);
        let out = builder.add(&lhs_node, &rhs_node);
        let graph = graph_from(&builder, &out);
        let mut executor = GraphExecutor::new(CpuBackend);
        let (mut plan, memory_plan, compiled_graph) = executor
            .compile_with_plan_and_quantize(graph, None, None)
            .expect("broadcast add graph should compile");
        group.bench_with_input(
            BenchmarkId::new("add_broadcast_zero_copy", case.name),
            &case,
            |bench, _| {
                bench.iter(|| {
                    reset_cpu_telemetry();
                    let outputs = executor
                        .execute(
                            &compiled_graph,
                            &mut plan,
                            &memory_plan,
                            &[black_box(&lhs_bytes), black_box(&rhs_bytes)],
                        )
                        .expect("broadcast add graph should execute");
                    let snapshot = cpu_telemetry_snapshot();
                    assert_eq!(
                        snapshot.arena_temp_copies, 0,
                        "broadcast add snapshot={snapshot:?}"
                    );
                    assert_eq!(snapshot.arena_temp_copy_bytes, 0);
                    black_box(outputs);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    cpu_benchmarks,
    bench_gemv,
    bench_gemm,
    bench_w4_activation_paths,
    bench_elementwise,
    bench_reductions,
    bench_fusions,
    bench_training_updates,
    bench_cpu_arena_telemetry
);
criterion_main!(cpu_benchmarks);
