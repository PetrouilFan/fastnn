def test_modular_python_facades_import():
    import fastnn as fnn
    import fastnn.fn3 as fn3
    import fastnn.tensor as tensor

    # fastnn.tensor is a callable wrapper around the tensor module
    assert fnn.tensor is tensor  # both are the wrapper
    assert tensor.Tensor is fnn.Tensor
    # Use equality check - different function objects but same functionality
    assert tensor.zeros((2, 2)).numpy().tolist() == fnn.zeros((2, 2)).numpy().tolist()
    assert tensor.tensor is fnn.tensor.tensor  # function accessible at fastnn.tensor.tensor

    # The 3.0 inference surface is importable and is the same object the
    # extension registers.
    assert fnn.fn3 is fn3
    assert fn3.Session is fnn._core.fn3.Session
