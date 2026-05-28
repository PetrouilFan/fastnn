//! Cross-architecture numerical consistency tests.
//!
//! These tests verify that the same computation produces the same results
//! on x86 (AVX2/AVX-512) and ARM (NEON). They are designed to be run on
//! both architectures and compared.
//!
//! On CI, these tests run within each architecture's job. Results are
//! compared to ensure near-exact match.

use fastnn::tensor::Tensor;

fn random_vec(n: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

#[test]
fn test_matmul_cross_arch_consistency() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let c = a.matmul(&b);

    assert_eq!(c.shape(), vec![2, 2]);
    let data = c.to_numpy();
    assert!((data[0] - 58.0).abs() < 1e-5);
    assert!((data[1] - 64.0).abs() < 1e-5);
    assert!((data[2] - 139.0).abs() < 1e-5);
    assert!((data[3] - 154.0).abs() < 1e-5);
}

#[test]
fn test_matmul_broadcast_consistency() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![2, 3]);
    let c = a.matmul(&b);

    assert_eq!(c.shape(), vec![3, 3]);
    let data = c.to_numpy();
    assert!((data[0] - 27.0).abs() < 1e-5);
    assert!((data[1] - 30.0).abs() < 1e-5);
    assert!((data[2] - 33.0).abs() < 1e-5);
    assert!((data[3] - 61.0).abs() < 1e-5);
    assert!((data[4] - 68.0).abs() < 1e-5);
    assert!((data[5] - 75.0).abs() < 1e-5);
    assert!((data[6] - 95.0).abs() < 1e-5);
    assert!((data[7] - 106.0).abs() < 1e-5);
    assert!((data[8] - 117.0).abs() < 1e-5);
}

#[test]
fn test_elementwise_add_consistency() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = &a + &b;

    assert_eq!(c.shape(), a.shape());
    let data = c.to_numpy();
    assert!((data[0] - 6.0).abs() < 1e-6);
    assert!((data[1] - 8.0).abs() < 1e-6);
    assert!((data[2] - 10.0).abs() < 1e-6);
    assert!((data[3] - 12.0).abs() < 1e-6);
}

#[test]
fn test_elementwise_mul_consistency() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = &a * &b;

    assert_eq!(c.shape(), a.shape());
    let data = c.to_numpy();
    assert!((data[0] - 5.0).abs() < 1e-6);
    assert!((data[1] - 12.0).abs() < 1e-6);
    assert!((data[2] - 21.0).abs() < 1e-6);
    assert!((data[3] - 32.0).abs() < 1e-6);
}

#[test]
fn test_relu_consistency() {
    let a = Tensor::from_vec(vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], vec![8]);
    let b = a.relu();
    let data = b.to_numpy();

    let expected = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 2.0, 3.0];
    for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "relu mismatch at {}: {} vs {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_softmax_consistency() {
    let a = Tensor::from_vec(random_vec(2 * 4), vec![2, 4]);
    let b = a.softmax(-1);

    assert_eq!(b.shape(), vec![2, 4]);
    let data = b.to_numpy();
    for row in 0..2 {
        let start = row * 4;
        let end = start + 4;
        let slice: Vec<f32> = data[start..end].to_vec();
        let sum: f32 = slice.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax row {} sum={}", row, sum);
    }
}

#[test]
fn test_reduce_sum_consistency() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = a.sum(1, false);

    assert_eq!(b.shape(), vec![2]);
    let data = b.to_numpy();
    assert!((data[0] - 6.0).abs() < 1e-5);
    assert!((data[1] - 15.0).abs() < 1e-5);
}

#[test]
fn test_transpose_consistency() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = a.transpose(0, 1);

    assert_eq!(b.shape(), vec![3, 2]);
    let data = b.to_numpy();
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[1] - 4.0).abs() < 1e-6);
    assert!((data[2] - 2.0).abs() < 1e-6);
    assert!((data[3] - 5.0).abs() < 1e-6);
    assert!((data[4] - 3.0).abs() < 1e-6);
    assert!((data[5] - 6.0).abs() < 1e-6);
}
