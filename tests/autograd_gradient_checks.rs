use fastnn::autograd;
use fastnn::tensor::Tensor;

const FINITE_DIFF_EPS: f32 = 1e-3;
const DEFAULT_ATOL: f32 = 2e-2;
const DEFAULT_RTOL: f32 = 2e-2;
const GELU_ATOL: f32 = 4e-2;
const GELU_RTOL: f32 = 4e-2;
const SILU_ATOL: f32 = 3e-2;
const SILU_RTOL: f32 = 3e-2;

fn reduce_all_to_scalar(mut tensor: Tensor) -> Tensor {
    while !tensor.shape().is_empty() {
        tensor = tensor.mean(0, false);
    }
    tensor
}

fn analytical_gradients<F>(inputs: &[(&[f32], &[i64])], build_loss: F) -> Vec<Vec<f32>>
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let tensors: Vec<Tensor> = inputs
        .iter()
        .map(|(data, shape)| Tensor::from_vec(data.to_vec(), shape.to_vec()).requires_grad_(true))
        .collect();
    let loss = build_loss(&tensors);
    autograd::backward(&loss, None).unwrap();
    tensors
        .iter()
        .map(|tensor| tensor.grad().expect("missing gradient").to_numpy().unwrap())
        .collect()
}

fn numerical_gradients<F>(inputs: &[(&[f32], &[i64])], build_loss: F) -> Vec<Vec<f32>>
where
    F: Fn(&[Tensor]) -> Tensor,
{
    (0..inputs.len())
        .map(|input_idx| {
            let (base_values, shape) = inputs[input_idx];
            let mut grad = vec![0.0; base_values.len()];
            for element_idx in 0..base_values.len() {
                let mut plus_inputs: Vec<Vec<f32>> =
                    inputs.iter().map(|(values, _)| values.to_vec()).collect();
                plus_inputs[input_idx][element_idx] += FINITE_DIFF_EPS;
                let plus_tensors: Vec<Tensor> = plus_inputs
                    .iter()
                    .zip(inputs.iter())
                    .map(|(values, (_, tensor_shape))| {
                        Tensor::from_vec(values.clone(), tensor_shape.to_vec())
                    })
                    .collect();
                let plus = build_loss(&plus_tensors).item().unwrap();

                let mut minus_inputs: Vec<Vec<f32>> =
                    inputs.iter().map(|(values, _)| values.to_vec()).collect();
                minus_inputs[input_idx][element_idx] -= FINITE_DIFF_EPS;
                let minus_tensors: Vec<Tensor> = minus_inputs
                    .iter()
                    .zip(inputs.iter())
                    .map(|(values, (_, tensor_shape))| {
                        Tensor::from_vec(values.clone(), tensor_shape.to_vec())
                    })
                    .collect();
                let minus = build_loss(&minus_tensors).item().unwrap();

                grad[element_idx] = (plus - minus) / (2.0 * FINITE_DIFF_EPS);
            }
            assert_eq!(
                grad.len(),
                base_values.len(),
                "gradient shape mismatch for {shape:?}"
            );
            grad
        })
        .collect()
}

fn assert_gradients_close(
    analytical: &[Vec<f32>],
    numerical: &[Vec<f32>],
    atol: f32,
    rtol: f32,
    op_name: &str,
) {
    assert_eq!(
        analytical.len(),
        numerical.len(),
        "input count mismatch for {op_name}"
    );
    for (input_idx, (analytical_grad, numerical_grad)) in
        analytical.iter().zip(numerical.iter()).enumerate()
    {
        assert_eq!(
            analytical_grad.len(),
            numerical_grad.len(),
            "element count mismatch for {op_name} input {input_idx}"
        );
        for (elem_idx, (&actual, &expected)) in analytical_grad
            .iter()
            .zip(numerical_grad.iter())
            .enumerate()
        {
            let tolerance = atol + rtol * expected.abs().max(actual.abs());
            let delta = (actual - expected).abs();
            assert!(
                delta <= tolerance,
                "{op_name} input {input_idx} element {elem_idx}: analytical={actual}, numerical={expected}, delta={delta}, tolerance={tolerance}"
            );
        }
    }
}

fn run_gradient_check<F>(
    op_name: &str,
    inputs: &[(&[f32], &[i64])],
    atol: f32,
    rtol: f32,
    build_loss: F,
) where
    F: Fn(&[Tensor]) -> Tensor + Copy,
{
    let analytical = analytical_gradients(inputs, build_loss);
    let numerical = numerical_gradients(inputs, build_loss);
    assert_gradients_close(&analytical, &numerical, atol, rtol, op_name);
}

#[test]
fn add_gradient_matches_finite_difference() {
    let a = [0.2, -0.5, 1.1, 2.0, -1.3, 0.7];
    let b = [1.7, 0.3, -0.9, 0.6, 2.4, -1.1];
    let shape = [2, 3];
    run_gradient_check(
        "add",
        &[(&a, &shape), (&b, &shape)],
        DEFAULT_ATOL,
        DEFAULT_RTOL,
        |tensors| reduce_all_to_scalar(tensors[0].add(&tensors[1])),
    );
}

#[test]
fn mul_gradient_matches_finite_difference() {
    let a = [0.2, -0.5, 1.1, 2.0, -1.3, 0.7];
    let b = [1.7, 0.3, -0.9, 0.6, 2.4, -1.1];
    let shape = [2, 3];
    run_gradient_check(
        "mul",
        &[(&a, &shape), (&b, &shape)],
        DEFAULT_ATOL,
        DEFAULT_RTOL,
        |tensors| reduce_all_to_scalar(tensors[0].mul(&tensors[1])),
    );
}

#[test]
fn matmul_gradient_matches_finite_difference() {
    let a = [0.2, -0.5, 1.1, 2.0, -1.3, 0.7];
    let b = [1.7, 0.3, -0.9, 0.6, 2.4, -1.1];
    let a_shape = [2, 3];
    let b_shape = [3, 2];
    run_gradient_check(
        "matmul",
        &[(&a, &a_shape), (&b, &b_shape)],
        DEFAULT_ATOL,
        DEFAULT_RTOL,
        |tensors| reduce_all_to_scalar(tensors[0].matmul(&tensors[1])),
    );
}

#[test]
fn relu_gradient_matches_finite_difference() {
    let x = [-1.3, -0.4, 0.7, 2.1];
    let shape = [4];
    run_gradient_check(
        "relu",
        &[(&x, &shape)],
        DEFAULT_ATOL,
        DEFAULT_RTOL,
        |tensors| reduce_all_to_scalar(tensors[0].relu()),
    );
}

#[test]
fn gelu_gradient_matches_finite_difference() {
    let x = [-1.3, -0.4, 0.7, 2.1];
    let shape = [4];
    run_gradient_check("gelu", &[(&x, &shape)], GELU_ATOL, GELU_RTOL, |tensors| {
        reduce_all_to_scalar(tensors[0].gelu())
    });
}

#[test]
fn silu_gradient_matches_finite_difference() {
    let x = [-1.3, -0.4, 0.7, 2.1];
    let shape = [4];
    run_gradient_check("silu", &[(&x, &shape)], SILU_ATOL, SILU_RTOL, |tensors| {
        reduce_all_to_scalar(tensors[0].silu())
    });
}

#[test]
fn sum_reduction_gradient_matches_finite_difference() {
    let x = [0.2, -0.5, 1.1, 2.0, -1.3, 0.7];
    let shape = [2, 3];
    run_gradient_check(
        "sum(dim=1, keepdim=false)",
        &[(&x, &shape)],
        DEFAULT_ATOL,
        DEFAULT_RTOL,
        |tensors| reduce_all_to_scalar(tensors[0].sum(1, false)),
    );
}

#[test]
fn mean_reduction_gradient_matches_finite_difference() {
    let x = [0.2, -0.5, 1.1, 2.0, -1.3, 0.7];
    let shape = [2, 3];
    run_gradient_check(
        "mean(dim=1, keepdim=false)",
        &[(&x, &shape)],
        DEFAULT_ATOL,
        DEFAULT_RTOL,
        |tensors| reduce_all_to_scalar(tensors[0].mean(1, false)),
    );
}
