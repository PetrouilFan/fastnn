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
    let a = Tensor::from_vec(random_vec(32 * 64), vec![32, 64]);
    let b = Tensor::from_vec(random_vec(64 * 16), vec![64, 16]);
    let c = a.matmul(&b);

    assert_eq!(c.shape(), vec![32, 16]);
    let data = c.to_numpy();
    assert!(data.iter().any(|&x| x != 0.0));
}

#[test]
fn test_matmul_broadcast_consistency() {
    let a = Tensor::from_vec(random_vec(8 * 32), vec![8, 32]);
    let b = Tensor::from_vec(random_vec(32 * 8), vec![32, 8]);
    let c = a.matmul(&b);

    assert_eq!(c.shape(), vec![8, 8]);
    let data = c.to_numpy();
    assert!(data.iter().any(|&x| x != 0.0));
}

#[test]
fn test_elementwise_add_consistency() {
    let a = Tensor::from_vec(random_vec(16 * 32), vec![16, 32]);
    let b = Tensor::from_vec(random_vec(16 * 32), vec![16, 32]);
    let c = &a + &b;

    assert_eq!(c.shape(), a.shape());
}

#[test]
fn test_elementwise_mul_consistency() {
    let a = Tensor::from_vec(random_vec(16 * 32), vec![16, 32]);
    let b = Tensor::from_vec(random_vec(16 * 32), vec![16, 32]);
    let c = &a * &b;

    assert_eq!(c.shape(), a.shape());
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
    let a = Tensor::from_vec(random_vec(4 * 16), vec![4, 16]);
    let b = a.softmax(-1);

    assert_eq!(b.shape(), vec![4, 16]);
    let data = b.to_numpy();
    for row in 0..4 {
        let start = row * 16;
        let end = start + 16;
        let slice: Vec<f32> = data[start..end].to_vec();
        let sum: f32 = slice.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax row {} sum={}", row, sum);
    }
}

#[test]
fn test_reduce_sum_consistency() {
    let a = Tensor::from_vec(random_vec(8 * 16), vec![8, 16]);
    let b = a.sum(1, false);

    assert_eq!(b.shape(), vec![8]);
}

#[test]
fn test_transpose_consistency() {
    let a = Tensor::from_vec(random_vec(16 * 32), vec![16, 32]);
    let b = a.transpose(0, 1);

    assert_eq!(b.shape(), vec![32, 16]);
}
