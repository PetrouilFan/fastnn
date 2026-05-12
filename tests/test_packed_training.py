"""Integration tests for packed linear layer training."""

import fastnn
from fastnn import Linear4, Linear8, Linear16, Linear32
import numpy as np
import pytest


class TestPackedTraining:
    """Test that packed linear layers can train and loss decreases."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        fastnn.set_seed(42)

    def test_train_step_reduces_loss_cross_entropy(self):
        """Loss should decrease over multiple train_step calls with cross_entropy."""
        layer = Linear4(8, 2)
        
        inp = fastnn.tensor([1.0, 0.5, -0.5, 1.0, 0.5, -0.5, 0.5, 1.0], [8])
        target_one_hot = fastnn.tensor([0.0, 1.0], [2])
        
        losses = []
        for _ in range(30):
            loss = layer.train_step(inp, target_one_hot, loss_fn="cross_entropy", lr=0.1)
            losses.append(float(loss.numpy()[0]))
        
        assert losses[-1] < losses[0] * 0.5, (
            f"Loss did not decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_train_step_reduces_loss_mse(self):
        """Loss should decrease over multiple train_step calls with MSE."""
        layer = Linear8(8, 2)
        
        inp = fastnn.tensor([1.0, 0.5, -0.5, 1.0, 0.5, -0.5, 0.5, 1.0], [8])
        target = fastnn.tensor([3.0, -2.0], [2])
        
        losses = []
        for _ in range(50):
            loss = layer.train_step(inp, target, loss_fn="mse", lr=0.05)
            losses.append(float(loss.numpy()[0]))
        
        assert losses[-1] < losses[0] * 0.5, (
            f"Loss did not decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_weights_change_after_training(self):
        """Master weights should change after training steps."""
        layer = Linear16(8, 2)
        initial_w = layer.master_weight.copy()
        
        inp = fastnn.tensor([1.0, 0.5, -0.5, 1.0, 0.5, -0.5, 0.5, 1.0], [8])
        target_one_hot = fastnn.tensor([0.0, 1.0], [2])
        
        for _ in range(10):
            layer.train_step(inp, target_one_hot, loss_fn="cross_entropy", lr=0.01)
        
        final_w = layer.master_weight
        assert initial_w != final_w, "Weights did not change after training"

    def test_train_eval_mode(self):
        """Train/eval mode switching should work."""
        layer = Linear32(4, 2)
        assert layer.is_training, "Layer should start in training mode"
        
        layer.eval()
        assert not layer.is_training, "Layer should not be in training mode after eval()"
        
        layer.train()
        assert layer.is_training, "Layer should be in training mode after train()"
