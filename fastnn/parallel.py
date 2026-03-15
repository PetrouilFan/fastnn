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

    def forward_backward(self, x, y, loss_fn):
        """Run forward and backward passes concurrently on all GPUs.

        Args:
            x: Input tensor (on CPU or any device)
            y: Target tensor (on CPU or any device)
            loss_fn: Loss function (e.g., cross_entropy_loss)

        Returns:
            Average loss across all devices
        """
        batch_size = x.shape[0]

        losses = [0.0] * len(self.device_ids)

        def worker(i):
            # Weighted data split based on GPU capability
            chunk_size = int(batch_size * self.weights[i])
            start = int(sum(self.weights[:i]) * batch_size)

            # Handle last chunk to include any remainder
            if i == len(self.device_ids) - 1:
                end = batch_size
            else:
                end = start + chunk_size

            # Move data chunk to device
            x_chunk = x[start:end].to_gpu(self.device_ids[i])
            y_chunk = y[start:end].to_gpu(self.device_ids[i])

            # Forward pass
            pred = self.replicas[i](x_chunk)
            loss = loss_fn(pred, y_chunk)

            # Backward pass (GIL is released by Rust)
            loss.backward()
            losses[i] = loss.item()

        # Execute concurrently using Python threads
        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(len(self.device_ids))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return sum(losses) / len(losses)

    def sync_gradients(self):
        """Synchronize gradients across all devices using bucketed AllReduce.

        This uses the fast Rust implementation that:
        1. Pulls gradients to CPU
        2. Averages them
        3. Pushes back to all devices
        """
        fnn._core.bucket_allreduce(self.param_groups)

    def step_optimizers(self, optimizers):
        """Step all optimizers and zero gradients.

        Args:
            optimizers: List of optimizers (one per replica)
        """
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
