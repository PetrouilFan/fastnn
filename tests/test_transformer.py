"""
Pytest test suite for Transformer encoder.
"""

import numpy as np
import fastnn as fnn
from tests.test_utils import make_transformer, make_tensor, assert_shape_equal


def test_transformer_forward():
    """Test basic transformer forward pass."""
    model = make_transformer()

    # Test forward pass with batch
    x = fnn.randint(low=0, high=100, shape=[32, 16])
    logits = model(x)
    assert_shape_equal(logits, [32, 2])


def test_transformer_parameters():
    """Test that transformer parameters are accessible and drive a forward pass."""
    model = make_transformer()

    params = list(model.parameters())
    assert len(params) > 0, "Transformer should have parameters"

    x = fnn.randint(low=0, high=100, shape=[4, 16])
    logits = model(x)
    assert_shape_equal(logits, [4, 2])
