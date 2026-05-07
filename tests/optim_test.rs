use fastnn::tensor::Tensor;
use fastnn::optim::{Adam, AdamW, SGD, RMSprop, Lion, Muon};

fn create_test_params() -> Vec<Tensor> {
    vec![
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]),
        Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]),
    ]
}

fn set_grads(params: &mut [Tensor]) {
    for p in params {
        let grad = Tensor::from_vec(vec![0.1f32; p.numel()], p.shape());
        p.set_grad(Some(grad));
    }
}

#[test]
fn test_adam_optimizer() {
    let mut params = create_test_params();
    let mut adam = Adam::new(params.clone(), 0.001, (0.9, 0.999), 1e-8, 0.0, false);
    
    set_grads(&mut params);
    adam.params_mut().clone_from(&params);
    
    // Step should work without errors
    adam.step();
    adam.zero_grad();
    
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
    
    adamw.step();
    adamw.zero_grad();
}

#[test]
fn test_sgd_optimizer() {
    let params = create_test_params();
    let mut sgd = SGD::new(params.clone(), 0.01, 0.9, 0.0, 0.0001, false);
    
    let mut params_with_grad = params.clone();
    set_grads(&mut params_with_grad);
    sgd.params_mut().clone_from(&params_with_grad);
    
    sgd.step();
    sgd.zero_grad();
}

#[test]
fn test_rmsprop_optimizer() {
    let params = create_test_params();
    let mut rmsprop = RMSprop::new(params.clone(), 0.01, 0.99, 1e-8, 0.0, 0.0, false);
    
    let mut params_with_grad = params.clone();
    set_grads(&mut params_with_grad);
    rmsprop.params_mut().clone_from(&params_with_grad);
    
    rmsprop.step();
    rmsprop.zero_grad();
}

#[test]
fn test_lion_optimizer() {
    let params = create_test_params();
    let mut lion = Lion::new(params.clone(), 0.001, (0.9, 0.99), 0.01);
    
    let mut params_with_grad = params.clone();
    set_grads(&mut params_with_grad);
    lion.params_mut().clone_from(&params_with_grad);
    
    lion.step();
    lion.zero_grad();
}

#[test]
fn test_muon_optimizer() {
    // Muon works best with 2D tensors
    let params = vec![
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]),
    ];
    let mut muon = Muon::new(params.clone(), 0.02, 0.9, 0.01, false);
    
    let mut params_with_grad = params.clone();
    set_grads(&mut params_with_grad);
    muon.params_mut().clone_from(&params_with_grad);
    
    muon.step();
    muon.zero_grad();
}

#[test]
fn test_weight_decay_optimizers() {
    // Test that WeightDecayOptimizer trait is implemented
    let params = create_test_params();
    
    let mut adam = Adam::new(params.clone(), 0.001, (0.9, 0.999), 1e-8, 0.01, false);
    adam.mark_biases_no_decay();
    
    let mut sgd = SGD::new(params.clone(), 0.01, 0.9, 0.0, 0.01, false);
    sgd.mark_biases_no_decay();
    
    let mut lion = Lion::new(params.clone(), 0.001, (0.9, 0.99), 0.01);
    lion.mark_biases_no_decay();
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
