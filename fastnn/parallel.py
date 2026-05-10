"""Distributed Data Parallel (DDP) implementation for fastnn."""

import fastnn._core as _core
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

__all__ = [
    "DataParallel",
]


class DataParallel:
    """Distributed Data Parallel wrapper for fastnn models.

    This class enables multi-GPU training by distributing data across
    multiple GPU devices and synchronizing gradients using bucketed AllReduce.

    Note: Due to fastnn's Rust-based architecture, this implementation
    requires creating model replicas manually rather than using deepcopy.

    Args:
        models: List of model replicas (one per GPU)
        device_ids: List of GPU device IDs (e.g., [0, 1])
        weights: Optional list of data weights per GPU. If None, uses
                 memory-based weighting (e.g., [0.7, 0.3] for 11GB + 4GB GPUs)

    Example:
        >>> # Create model replicas for each GPU
        >>> model_gpu0 = fnn.models.MLP(input_dim=784, hidden_dims=[256], output_dim=10)
        >>> model_gpu1 = fnn.models.MLP(input_dim=784, hidden_dims=[256], output_dim=10)
        >>>
        >>> # Initialize DataParallel
        >>> dp_model = DataParallel(
        ...     [model_gpu0, model_gpu1],
        ...     device_ids=[0, 1],
        ...     weights=[0.6, 0.4]
        ... )
        >>>
        >>> # Training loop
        >>> for x_batch, y_batch in dataloader:
        ...     loss = dp_model.forward_backward(x_batch, y_batch, fnn.cross_entropy_loss)
        ...     dp_model.sync_gradients()
        ...     for opt in optimizers:
        ...         opt.step()
        ...         opt.zero_grad()
    """

    def __init__(
        self, models: List, device_ids: List[int], weights: Optional[List[float]] = None
    ):
        if not isinstance(models, list) or len(models) != len(device_ids):
            raise ValueError("models must be a list with same length as device_ids")

        self.device_ids = device_ids
        self.replicas = models
        self.epoch_times = [0.0] * len(device_ids)  # Track performance per GPU

        # Default weights: equal split for unknown configurations
        if weights is None:
            self.weights = [1.0 / len(device_ids)] * len(device_ids)
        else:
            if len(weights) != len(device_ids):
                raise ValueError("weights must have same length as device_ids")
            self.weights = weights

        # Normalize weights to sum to 1.0
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Ensure all replicas have parameters on correct devices
        for i, (replica, d) in enumerate(zip(self.replicas, self.device_ids)):
            if hasattr(replica, "to_gpu"):
                replica.to_gpu(d)
            else:
                device_str = f"cuda:{d}" if isinstance(d, int) else str(d)
                if hasattr(replica, "_parameters"):
                    for name, param in replica._parameters.items():
                        if hasattr(param, "to_device"):
                            param.to_device(device_str)
                        elif hasattr(param, "data") and hasattr(param.data, "to_device"):
                            param.data.to_device(device_str)
                if hasattr(replica, "_buffers"):
                    for name, buf in replica._buffers.items():
                        if hasattr(buf, "to_device"):
                            buf.to_device(device_str)

        self.param_groups = [list(r.parameters()) for r in self.replicas]

        # Initialize performance tracking
        self.performance_history = []
        self.adjustment_interval = 1  # Adjust weights after every epoch

    def forward_backward(self, x, y, loss_fn):
        """Run forward and backward passes concurrently on all GPUs.

        Args:
            x: Input tensor (on CPU or any device)
            y: Target tensor (on CPU or any device)
            loss_fn: Loss function (e.g., cross_entropy_loss)

        Returns:
            Average loss across all devices
        """
        import time

        batch_size = x.shape[0]
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        losses = [0.0] * len(self.device_ids)
        gpu_times = [0.0] * len(self.device_ids)  # Track time per GPU

        # Precompute split indices using cumulative boundaries with proper rounding
        cumulative_weights = []
        cumsum = 0.0
        for w in self.weights[:-1]:
            cumsum += w
            cumulative_weights.append(round(cumsum * batch_size))
        cumulative_weights.append(batch_size)

        split_indices = [0] + cumulative_weights

        def worker(i):
            # Get split indices from precomputed list
            start = split_indices[i]
            end = split_indices[i + 1]

            # Measure time for this GPU's work
            gpu_start = time.time()

            # Move data chunk to device
            x_chunk = x[start:end].to_gpu(self.device_ids[i])
            y_chunk = y[start:end].to_gpu(self.device_ids[i])

            # Forward pass
            pred = self.replicas[i](x_chunk)
            loss = loss_fn(pred, y_chunk)

            # Backward pass (GIL is released by Rust)
            loss.backward()
            losses[i] = loss.item()

            gpu_times[i] = time.time() - gpu_start

        # Execute concurrently using thread pool
        with ThreadPoolExecutor(max_workers=len(self.device_ids)) as executor:
            futures = [executor.submit(worker, i) for i in range(len(self.device_ids))]
            for f in futures:
                f.result()

        # Update epoch times with measured GPU times
        self.epoch_times = gpu_times

        batch_sizes = [
            split_indices[i + 1] - split_indices[i] for i in range(len(self.device_ids))
        ]
        total_samples = sum(batch_sizes)
        avg_loss = (
            sum(loss * n for loss, n in zip(losses, batch_sizes)) / total_samples
        )
        return avg_loss

    def sync_gradients(self):
        """Synchronize gradients across all devices using bucketed AllReduce.

        This uses the fast Rust implementation that:
        1. Pulls gradients to CPU
        2. Averages them
        3. Pushes back to all devices
        """
        _core.bucket_allreduce(self.param_groups)

    def adjust_weights_based_on_performance(self):
        """Adjust workload weights based on measured GPU performance.
        
        This method uses historical performance data to adjust weights,
        favoring faster GPUs while ensuring all GPUs are utilized.
        """
        if len(self.epoch_times) < 2:
            return  # Not enough data to adjust

        # Ensure we have valid times
        if any(t <= 0 for t in self.epoch_times):
            return

        # Update performance history with exponential moving average
        if not hasattr(self, "_performance_ema"):
            self._performance_ema = self.epoch_times.copy()
        else:
            alpha_ema = 0.3  # Smoothing factor for EMA
            self._performance_ema = [
                alpha_ema * new + (1 - alpha_ema) * old
                for new, old in zip(self.epoch_times, self._performance_ema)
            ]

        ema_times = self._performance_ema
        alpha = 0.3  # Smoothing factor for weight adjustment

        # Calculate inverse times and total
        inv_times = [1.0 / t for t in ema_times]
        total_inv = sum(inv_times)

        if total_inv <= 0:
            return

        # Calculate target weights proportional to inverse time (faster GPU -> higher weight)
        target_weights = [inv / total_inv for inv in inv_times]
        old_weights = getattr(self, "_last_weights", self.weights)

        # Smooth target weights with current weights and apply min/max bounds
        new_weights = []
        for target, current in zip(target_weights, self.weights):
            w = alpha * target + (1 - alpha) * current
            w = max(w, 0.1)  # Avoid starvation
            w = min(w, 0.9)  # Avoid overloading
            new_weights.append(w)

        # Renormalize to sum to 1.0
        total_new = sum(new_weights)
        self.weights = [w / total_new for w in new_weights]

        # Check for significant change compared to previous weights
        significant_change = any(
            abs(w - old_w) > 0.05 for w, old_w in zip(self.weights, old_weights)
        )

        # Log adjustment for debugging
        if significant_change:
            print(
                f"Adjusted weights: {self.weights} (GPU times: {self.epoch_times}, EMA: {ema_times})"
            )
            self._last_weights = self.weights.copy()

    def get_current_weights(self):
        """Return current workload weights."""
        return self.weights.copy()

    def step_optimizers(self, optimizers):
        """Step all optimizers and zero gradients.

        Args:
            optimizers: List of optimizers (one per replica)
        """
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
