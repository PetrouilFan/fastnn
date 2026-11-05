import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "gpt2_w4a8_activation_audit.py"
SPEC = importlib.util.spec_from_file_location("gpt2_w4a8_activation_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_parse_sequences_supports_multiple_independent_trajectories():
    assert AUDIT.parse_sequences(["42,43", "50, 51,52"], [1]) == [
        [42, 43],
        [50, 51, 52],
    ]
    assert AUDIT.parse_sequences([], [7, 8]) == [[7, 8]]


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


def test_activation_weighted_clipping_reduces_error_from_inactive_outlier():
    weight = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(32, 1)
    weight[0, 0] = 10.0
    importance = np.ones(32, dtype=np.float64)
    importance[0] = 0.0

    baseline = AUDIT.grouped_i4_dequantize(weight, 32)
    optimized, ratios = AUDIT.activation_weighted_grouped_i4_dequantize(
        weight, importance, 32
    )
    baseline_error = np.sum(importance[:, None] * (baseline - weight) ** 2)
    optimized_error = np.sum(importance[:, None] * (optimized - weight) ** 2)

    assert optimized_error < baseline_error * 0.9
    assert ratios.shape == (1, 1)
    assert 0.70 <= ratios[0, 0] <= 1.0


def test_full_activation_objective_beats_diagonal_approximation_on_correlated_inputs():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(12, 1))
    activation = np.concatenate(
        [latent + 0.02 * rng.normal(size=(12, 1)) for _ in range(32)], axis=1
    ).astype(np.float32)
    weight = rng.normal(scale=0.4, size=(32, 3)).astype(np.float32)
    weight[0] *= 8.0
    importance = np.mean(activation.astype(np.float64) ** 2, axis=0)

    diagonal, _ = AUDIT.activation_weighted_grouped_i4_dequantize(
        weight, importance, 32
    )
    covariance, ratios = AUDIT.activation_weighted_grouped_i4_dequantize(
        weight, activation, 32
    )
    diagonal_error = np.mean((activation @ (diagonal - weight)) ** 2)
    covariance_error = np.mean((activation @ (covariance - weight)) ** 2)

    assert covariance_error <= diagonal_error
    assert ratios.shape == (3, 1)


def test_gptq_error_feedback_reduces_correlated_output_error():
    rng = np.random.default_rng(19)
    latent = rng.normal(size=(64, 4))
    mixing = rng.normal(size=(4, 32))
    activation = (latent @ mixing + 0.01 * rng.normal(size=(64, 32))).astype(np.float32)
    weight = rng.normal(scale=0.5, size=(32, 8)).astype(np.float32)

    baseline = AUDIT.grouped_i4_dequantize(weight, 32)
    reconstructed = AUDIT.gptq_grouped_i4_dequantize(weight, activation, 32)
    baseline_error = np.mean((activation @ (baseline - weight)) ** 2)
    reconstructed_error = np.mean((activation @ (reconstructed - weight)) ** 2)

    assert reconstructed_error < baseline_error
    assert reconstructed.shape == weight.shape
    assert np.all(np.isfinite(reconstructed))


def test_gptq_rejects_invalid_contracts():
    weight = np.zeros((32, 2), dtype=np.float32)
    activation = np.zeros((4, 31), dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "K dimension"):
        AUDIT.gptq_grouped_i4_dequantize(weight, activation, 32)
    with np.testing.assert_raises_regex(ValueError, "damping"):
        AUDIT.gptq_grouped_i4_dequantize(weight, np.zeros((4, 32)), 32, 0.0)


def test_policy_summary_reports_holdout_regressions():
    def metric(value):
        return {"normalized_rmse": value}

    layers = [
        {
            "activation_weighted_output_error": metric(0.3),
            "optimized_activation_weighted_output_error": metric(0.2),
            "gptq_activation_weighted_output_error": metric(0.1),
            "validation_output_error": metric(0.3),
            "optimized_validation_output_error": metric(0.1),
            "gptq_validation_output_error": metric(0.2),
        }
    ]
    summary = AUDIT.policy_summary(layers)

    assert summary["calibration"]["gptq_beats_clipping"] == 1
    assert summary["validation"]["gptq_beats_clipping"] == 0
    assert summary["validation"]["projection_count"] == 1
