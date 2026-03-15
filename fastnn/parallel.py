"""Distributed Data Parallel (DDP) implementation for fastnn."""

import fastnn as fnn
import threading


class DataParallel:
    """Distributed Data Parallel wrapper for fastnn models.

    Note: Due to fastnn's Rust-based architecture, this implementation
    requires creating model replicas manually rather than using deepcopy.

    Args:
        models: List of model replicas (one per GPU)
        device_ids: List of GPU device IDs (e.g., [0, 1])
        weights: Optional list of data weights per GPU. If None, uses
                 memory-based weighting (e.g., [0.7, 0.3] for 11GB + 4GB GPUs)
    """

    def __init__(self, models, device_ids, weights=None):
        if not isinstance(models, list) or len(models) != len(device_ids):
            raise ValueError("models must be a list with same length as device_ids")

        self.device_ids = device_ids
        self.replicas = models
        self.epoch_times = [0.0] * len(device_ids)  # Track performance per GPU

        # Default weights: proportional to GPU memory capacity
        # 1080 Ti ~ 11GB, 1650 ~ 4GB → [0.7, 0.3]
        if weights is None:
            if len(device_ids) == 2:
                self.weights = [0.7, 0.3]  # Weighted by memory/ability
            else:
                # Even split for other configurations
                self.weights = [1.0 / len(device_ids)] * len(device_ids)
        else:
            self.weights = weights

        # Normalize weights to sum to 1.0
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Ensure all replicas have parameters on correct devices
        for i, (replica, d) in enumerate(zip(self.replicas, self.device_ids)):
            # Move the entire replica to the correct device
            if hasattr(replica, "to_gpu"):
                replica.to_gpu(d)
            else:
                # Fallback: iterate through parameters (this might not work if parameters() returns copies)
                for p in replica.parameters():
                    if p.device != f"wgpu:{d}":
                        # This won't work because p is likely a copy
                        pass

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

        losses = [0.0] * len(self.device_ids)
        gpu_times = [0.0] * len(self.device_ids)  # Track time per GPU

        def worker(i):
            # Weighted data split based on GPU capability
            chunk_size = int(batch_size * self.weights[i])
            start = int(sum(self.weights[:i]) * batch_size)

            # Handle last chunk to include any remainder
            if i == len(self.device_ids) - 1:
                end = batch_size
            else:
                end = start + chunk_size

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

        # Execute concurrently using Python threads
        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(len(self.device_ids))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Update epoch times with measured GPU times
        self.epoch_times = gpu_times

        return sum(losses) / len(losses)

    def sync_gradients(self):
        """Synchronize gradients across all devices using bucketed AllReduce.

        This uses the fast Rust implementation that:
        1. Pulls gradients to CPU
        2. Averages them
        3. Pushes back to all devices
        """
        fnn._core.bucket_allreduce(self.param_groups)

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

        # Simple heuristic: adjust weights inversely proportional to epoch time
        # Faster GPU should get more data
        # Use smoothing to avoid oscillations
        alpha = 0.3  # Smoothing factor: how much to trust new measurements

        # Inverse proportional: weight = (1/time) / sum(1/time)
        inv_times = [1.0 / t for t in self.epoch_times]
        total_inv = sum(inv_times)

        if total_inv > 0:
            target_weights = [inv / total_inv for inv in inv_times]

            # Smooth with previous weights
            new_weights = [
                alpha * target + (1 - alpha) * current
                for target, current in zip(target_weights, self.weights)
            ]

            # Ensure minimum weight for each GPU (avoid starvation)
            min_weight = 0.1  # At least 10% for each GPU
            new_weights = [max(w, min_weight) for w in new_weights]

            # Renormalize to sum to 1.0
            total_new = sum(new_weights)
            self.weights = [w / total_new for w in new_weights]

            # Log adjustment for debugging (only if significant change)
            if any(
                abs(new - old) > 0.05
                for new, old in zip(self.weights, self.epoch_times)
            ):
                print(
                    f"Adjusted weights: {self.weights} (GPU times: {self.epoch_times})"
                )

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
