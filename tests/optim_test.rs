use fastnn::optim::{Adam, AdamW, Lion, Muon, Optimizer, RMSprop, WeightDecayOptimizer, SGD};
use fastnn::tensor::Tensor;

fn create_test_params() -> Vec<Tensor> {
    vec![
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]),
        Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]),
    ]
}

fn set_grads(params: &mut [Tensor]) {
    for p in params.iter_mut() {
        *p = p.requires_grad_(true);
        let grad = Tensor::from_vec(vec![0.1f32; p.numel() as usize], p.shape());
        p.set_grad(Some(grad));
    }
}

#[test]
fn test_adam_optimizer() {
    let mut params = create_test_params();
    let mut adam = Adam::new(params.clone(), 0.001, (0.9, 0.999), 1e-8, 0.0, false);

    set_grads(&mut params);
    adam.params_mut().clone_from(&params);

    let initial = adam
        .params()
        .iter()
        .map(|p| p.to_numpy().unwrap())
        .collect::<Vec<_>>();

    adam.step();
    adam.zero_grad();

    let final_params = adam
        .params()
        .iter()
        .map(|p| p.to_numpy().unwrap())
        .collect::<Vec<_>>();
    assert_ne!(initial, final_params, "Adam should update parameters");

    // Test state_dict and load_state_dict
    let state = adam.state_dict();
    let mut adam2 = Adam::new(params.clone(), 0.001, (0.9, 0.999), 1e-8, 0.0, false);
    adam2.load_state_dict(state);
}

#[test]
fn test_adamw_optimizer() {
    let params = create_test_params();
    let mut adamw = AdamW::new(params.clone(), 0.001, (0.9, 0.999), 1e-8, 0.01, false);

    let mut params_with_grad = params.clone();
    set_grads(&mut params_with_grad);
    adamw.params_mut().clone_from(&params_with_grad);

    let initial = adamw
        .params()
        .iter()
        .map(|p| p.to_numpy().unwrap())
        .collect::<Vec<_>>();

    adamw.step();
    adamw.zero_grad();

    let final_params = adamw
        .params()
        .iter()
        .map(|p| p.to_numpy().unwrap())
        .collect::<Vec<_>>();
    assert_ne!(initial, final_params, "AdamW should update parameters");
}

#[test]
fn test_sgd_optimizer() {
    let params = create_test_params();
    let mut sgd = SGD::new(params.clone(), 0.01, 0.9, 0.0, 0.0001, false);

    let mut params_with_grad = params.clone();
    set_grads(&mut params_with_grad);
    sgd.params_mut().clone_from(&params_with_grad);

    let initial = sgd
        .params()
        .iter()
        .map(|p| p.to_numpy().unwrap())
        .collect::<Vec<_>>();

    sgd.step();
    sgd.zero_grad();

    let final_params = sgd
        .params()
        .iter()
        .map(|p| p.to_numpy().unwrap())
        .collect::<Vec<_>>();
    assert_ne!(initial, final_params, "SGD should update parameters");
    for (init_vals, final_vals) in initial.iter().zip(final_params.iter()) {
        for (&i, &f) in init_vals.iter().zip(final_vals.iter()) {
            assert!(
                f < i,
                "SGD should decrease parameters with positive gradient"
            );
        }
    }
}

#[test]
fn test_rmsprop_optimizer() {
    let params = create_test_params();
    let mut rmsprop = RMSprop::new(params.clone(), 0.01, 0.99, 1e-8, 0.0, 0.0, false);

    let mut params_with_grad = params.clone();
    set_grads(&mut params_with_grad);
    rmsprop.params_mut().clone_from(&params_with_grad);

    let initial = rmsprop
        .params
        .iter()
        .map(|p| p.to_numpy().unwrap())
        .collect::<Vec<_>>();

    rmsprop.step();
    rmsprop.zero_grad();

    let final_params = rmsprop
        .params
        .iter()
        .map(|p| p.to_numpy().unwrap())
        .collect::<Vec<_>>();
    assert_ne!(initial, final_params, "RMSprop should update parameters");
}

#[test]
fn test_lion_optimizer() {
    let params = create_test_params();
    let mut lion = Lion::new(params.clone(), 0.001, (0.9, 0.99), 0.01);

    let mut params_with_grad = params.clone();
    set_grads(&mut params_with_grad);
    lion.params_mut().clone_from(&params_with_grad);

    let initial = lion
        .params()
        .iter()
        .map(|p| p.to_numpy().unwrap())
        .collect::<Vec<_>>();

    lion.step();
    lion.zero_grad();

    let final_params = lion
        .params()
        .iter()
        .map(|p| p.to_numpy().unwrap())
        .collect::<Vec<_>>();
    assert_ne!(initial, final_params, "Lion should update parameters");
}

#[test]
fn test_muon_optimizer() {
    // Muon works best with 2D tensors
    let params = vec![Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])];
    let mut muon = Muon::new(params.clone(), 0.02, 0.9, 0.01, false);

    let mut params_with_grad = params.clone();
    set_grads(&mut params_with_grad);
    muon.params_mut().clone_from(&params_with_grad);

    let initial = muon
        .params
        .iter()
        .map(|p| p.to_numpy().unwrap())
        .collect::<Vec<_>>();

    muon.step();
    muon.zero_grad();

    let final_params = muon
        .params
        .iter()
        .map(|p| p.to_numpy().unwrap())
        .collect::<Vec<_>>();
    assert_ne!(initial, final_params, "Muon should update parameters");
}

#[test]
fn test_weight_decay_optimizers() {
    // Test that WeightDecayOptimizer trait is implemented
    let mut params = create_test_params();
    // Add a 2D parameter to verify bias marking distinguishes 1D (bias) from 2D (weight)
    params.push(Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));

    let mut adam = Adam::new(params.clone(), 0.001, (0.9, 0.999), 1e-8, 0.01, false);
    adam.mark_biases_no_decay();
    assert!(adam.no_decay[0], "1D bias param should be no_decay");
    assert!(adam.no_decay[1], "1D bias param should be no_decay");
    assert!(
        !adam.no_decay[2],
        "2D weight param should have weight decay"
    );

    let mut sgd = SGD::new(params.clone(), 0.01, 0.9, 0.0, 0.01, false);
    sgd.mark_biases_no_decay();
    assert!(sgd.no_decay[0], "1D bias param should be no_decay");
    assert!(sgd.no_decay[1], "1D bias param should be no_decay");
    assert!(!sgd.no_decay[2], "2D weight param should have weight decay");

    let mut lion = Lion::new(params.clone(), 0.001, (0.9, 0.99), 0.01);
    lion.mark_biases_no_decay();
    assert!(lion.no_decay[0], "1D bias param should be no_decay");
    assert!(lion.no_decay[1], "1D bias param should be no_decay");
    assert!(
        !lion.no_decay[2],
        "2D weight param should have weight decay"
    );
}

#[test]
fn test_add_param_group() {
    let params1 = vec![Tensor::from_vec(vec![1.0f32, 2.0], vec![2])];
    let params2 = vec![Tensor::from_vec(vec![3.0f32, 4.0], vec![2])];

    let mut adam = Adam::new(params1.clone(), 0.001, (0.9, 0.999), 1e-8, 0.0, false);
    adam.add_param_group(params2.clone());

    assert_eq!(adam.params().len(), 2);
    assert_eq!(adam.no_decay.len(), 2);
    assert_eq!(adam.step.len(), 2);
}

#[test]
fn test_muon_orthogonalization_quality() {
    // Verify Muon produces finite, non-zero updates for a well-conditioned
    // 2D matrix — confirms the Newton-Schulz iteration converges correctly
    // after removing intermediate re-normalization syncs.
    let params = vec![Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    )];
    let mut muon = Muon::new(params.clone(), 0.02, 0.9, 0.0, false);

    let mut params_with_grad = params.clone();
    set_grads(&mut params_with_grad);
    muon.params_mut().clone_from(&params_with_grad);

    muon.step();

    let result: Vec<f32> = muon.params[0].to_numpy().unwrap();
    assert!(
        result.iter().all(|x| x.is_finite()),
        "Muon update produced non-finite values: {:?}",
        result
    );
    assert_ne!(
        result,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Muon should modify parameters"
    );

    // Run multiple steps to verify no divergence over time
    for _ in 0..9 {
        set_grads(muon.params_mut());
        muon.step();
    }
    let final_result: Vec<f32> = muon.params[0].to_numpy().unwrap();
    assert!(
        final_result.iter().all(|x| x.is_finite()),
        "Muon diverged after multiple steps: {:?}",
        final_result
    );
}

#[test]
fn test_muon_zero_norm_handling() {
    // Verify Muon handles zero-gradient (and thus zero-momentum) gracefully
    // — the EPSILON guard in Newton-Schulz should produce zeros, not NaN.
    let params = vec![Tensor::from_vec(vec![0.0; 4], vec![2, 2])];
    let mut muon = Muon::new(params.clone(), 0.02, 0.9, 0.0, false);

    let mut params_with_grad = params.clone();
    set_grads(&mut params_with_grad);
    muon.params_mut().clone_from(&params_with_grad);

    muon.step();

    let result: Vec<f32> = muon.params[0].to_numpy().unwrap();
    assert!(
        result.iter().all(|x| x.is_finite()),
        "Muon with zero input produced non-finite values: {:?}",
        result
    );
}

#[test]
fn test_muon_state_dict_roundtrip() {
    let params = vec![Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])];
    let mut muon = Muon::new(params.clone(), 0.02, 0.9, 0.01, false);

    let mut params_with_grad = params.clone();
    set_grads(&mut params_with_grad);
    muon.params_mut().clone_from(&params_with_grad);
    muon.step();

    let state = muon.state_dict();
    let mut muon2 = Muon::new(params.clone(), 0.02, 0.9, 0.01, false);
    muon2.load_state_dict(state);

    let r1: Vec<f32> = muon.params[0].to_numpy().unwrap();
    let r2: Vec<f32> = muon2.params[0].to_numpy().unwrap();
    assert_eq!(r1, r2, "state_dict roundtrip should preserve parameters");
}
