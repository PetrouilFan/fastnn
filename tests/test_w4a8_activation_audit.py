import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "gpt2_w4a8_activation_audit.py"
SPEC = importlib.util.spec_from_file_location("gpt2_w4a8_activation_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_grouped_i4_dequantize_preserves_constant_and_partial_groups():
    weight = np.concatenate(
        [
            np.full((32, 2), [2.5, -3.0], dtype=np.float32),
            np.linspace(-1.0, 1.0, 26, dtype=np.float32).reshape(13, 2),
        ],
        axis=0,
    )
    actual = AUDIT.grouped_i4_dequantize(weight, 32)

    np.testing.assert_array_equal(actual[:32], weight[:32])
    assert actual.shape == weight.shape
    assert np.all(np.isfinite(actual))


def test_activation_weighting_distinguishes_equal_weight_error():
    reference = np.ones((2, 4), dtype=np.float32)
    error_on_active_channel = reference.copy()
    error_on_inactive_channel = reference.copy()
    error_on_active_channel[0, 0] += 1.0
    error_on_inactive_channel[1, 0] += 1.0
    activation = np.asarray([[10.0, 0.01]], dtype=np.float32)

    active_output = activation @ error_on_active_channel
    inactive_output = activation @ error_on_inactive_channel
    reference_output = activation @ reference
    active_error = AUDIT.metrics(active_output, reference_output)["rmse"]
    inactive_error = AUDIT.metrics(inactive_output, reference_output)["rmse"]

    assert active_error > inactive_error * 100.0
