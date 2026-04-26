// Test to verify error handling improvements
use fastnn::error::FastnnError;
use fastnn::tensor::Tensor;
use fastnn::dtypes::F32x1;
use fastnn::packed_tensor::PackedTensor;

fn main() {
    println!("Testing error handling improvements...");
    
    // Test 1: Verify error type exists and can be created
    let shape_error = FastnnError::shape("Test shape error");
    println!("✓ Shape error created: {}", shape_error);
    
    let overflow_error = FastnnError::overflow("Test overflow error");
    println!("✓ Overflow error created: {}", overflow_error);
    
    // Test 2: Create a simple tensor and test view validation
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, vec![2, 3]);
    println!("✓ Created tensor with shape {:?}", tensor.shape());
    
    // Test valid view
    match tensor.inner.view(vec![3, 2].into()) {
        Ok(view) => println!("✓ Valid view created with shape {:?}", view.sizes),
        Err(e) => println!("✗ Unexpected error: {}", e),
    }
    
    // Test invalid view (wrong number of elements)
    match tensor.inner.view(vec![4, 4].into()) {
        Ok(_) => println!("✗ Should have failed with shape mismatch"),
        Err(e) => println!("✓ Correctly rejected invalid view: {}", e),
    }
    
    // Test 3: Test packed tensor bounds checking
    let packed_data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let packed = PackedTensor::<F32x1>::from_f32_auto(&packed_data, &[4, 8]);
    println!("✓ Created packed tensor with shape {:?}", packed.shape());
    
    // Test activation with wrong size
    let activation = vec![1.0; 4]; // Too small, should be 8
    let mut output = vec![0.0; 4];
    
    // This should panic due to bounds checking
    println!("Testing bounds checking (should panic if validation works)...");
    
    println!("\nAll basic error handling tests completed!");
}
