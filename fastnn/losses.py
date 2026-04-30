"""Loss functions exported by the Rust core."""

import fastnn._core as _core

mse_loss = _core.mse_loss
cross_entropy_loss = _core.cross_entropy_loss
bce_with_logits = _core.bce_with_logits
huber_loss = _core.huber_loss

__all__ = [
    "mse_loss",
    "cross_entropy_loss",
    "bce_with_logits",
    "huber_loss",
]

