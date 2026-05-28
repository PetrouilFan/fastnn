use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fastnn::backend::cpu::microkernels::{gemm_cpu, gemv_cpu};
use fastnn::dtypes::{F32x1, PackedWord, U4x8, U8x4};
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

fn batch_inputs(batch: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..batch)
        .map(|batch_idx| {
            (0..cols)
                .map(|col_idx| ((((batch_idx * 13 + col_idx) % 41) as f32) - 20.0) * 0.05)
                .collect()
        })
        .collect()
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
        bench_fastnn_gemv::<U8x4>(&mut group, "fastnn_u8x4", case);
        bench_fastnn_gemv::<U4x8>(&mut group, "fastnn_u4x8", case);
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
        bench_fastnn_gemm::<U8x4>(&mut group, "fastnn_u8x4", case);
        bench_fastnn_gemm::<U4x8>(&mut group, "fastnn_u4x8", case);
    }

    group.finish();
}

criterion_group!(cpu_benchmarks, bench_gemv, bench_gemm);
criterion_main!(cpu_benchmarks);
