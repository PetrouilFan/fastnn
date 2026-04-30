def test_modular_python_facades_import():
    import fastnn as fnn
    # Save original tensor attribute (function) before submodule import overwrites it
    original_tensor = fnn.tensor
    import fastnn.losses as losses
    import fastnn.nn as nn
    import fastnn.ops as ops
    import fastnn.tensor as tensor

    assert tensor.Tensor is fnn.Tensor
    assert tensor.zeros is fnn.zeros
    assert ops.relu is fnn.relu
    assert ops.matmul is fnn.matmul
    assert nn.Linear is fnn.Linear
    assert losses.mse_loss is fnn.mse_loss

    # Restore original tensor attribute to avoid clobbering for subsequent tests
    fnn.tensor = original_tensor

