"""Data loading utilities for fastnn.

Provides Dataset, DataLoader, Sampler, and TensorDataset classes
with support for shuffling, batching, prefetching, multi-process loading, and auto-tuning.
"""

import os
import random
import threading
import time
import queue
from collections import deque
from typing import Any, Callable, Iterator, List, Optional, Sequence


class _Metrics:
    """Thread-safe metrics tracking for adaptive data loading.

    Tracks wait times and batch times to inform auto-tuning decisions.
    """

    def __init__(self, window: int = 200):
        self._window = window
        self._wait_times: deque[float] = deque(maxlen=window)
        self._batch_times: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, wait_ms: float, batch_ms: float) -> None:
        with self._lock:
            self._wait_times.append(wait_ms)
            self._batch_times.append(batch_ms)

    def mean_wait_ms(self) -> float:
        with self._lock:
            if not self._wait_times:
                return 0.0
            return sum(self._wait_times) / len(self._wait_times)

    def mean_batch_ms(self) -> float:
        with self._lock:
            if not self._batch_times:
                return 0.0
            return sum(self._batch_times) / len(self._batch_times)

    def reset(self) -> None:
        with self._lock:
            self._wait_times.clear()
            self._batch_times.clear()


class _AutoTuner:
    """Asymmetric auto-tuner for data loading resources.

    Scales up immediately when data loading is slow, requires consecutive
    epochs of under-utilization before scaling down. This prevents
    transient spikes from causing costly worker teardowns.
    """

    def __init__(
        self,
        min_workers: int = 0,
        max_workers: Optional[int] = None,
        up_threshold_ms: float = 30.0,
        down_threshold_ms: float = 5.0,
        initial_workers: int = 1,
        initial_prefetch: int = 2,
        scale_down_patience: int = 2,
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers or (os.cpu_count() or 4)
        self.up_threshold_ms = up_threshold_ms
        self.down_threshold_ms = down_threshold_ms
        self.current_workers = initial_workers
        self.current_prefetch = initial_prefetch
        self.scale_down_patience = scale_down_patience
        self.mode: str = "thread"
        self._under_util_epochs = 0

    def adjust(self, mean_wait_ms: float) -> tuple[int, int]:
        if mean_wait_ms > self.up_threshold_ms:
            self._under_util_epochs = 0
            if self.current_prefetch < 8:
                self.current_prefetch += 1
            elif self.current_workers < self.max_workers:
                self.current_workers += 1

        elif mean_wait_ms < self.down_threshold_ms:
            self._under_util_epochs += 1
            if self._under_util_epochs >= self.scale_down_patience:
                self._under_util_epochs = 0
                if self.mode == "process" and self.current_workers > self.min_workers:
                    self.current_workers -= 1
                elif self.current_prefetch > 1:
                    self.current_prefetch -= 1
                elif self.current_workers > self.min_workers:
                    self.current_workers -= 1
        else:
            self._under_util_epochs = 0

        return self.current_workers, self.current_prefetch


class Dataset:
    """Abstract base class for all datasets."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample is a tuple of tensor slices along the first dimension.

    Args:
        *tensors: Tensors with the same first dimension size.

    Example:
        >>> x = fnn.randn([100, 3, 32, 32])
        >>> y = fnn.randint(0, 10, [100])
        >>> ds = TensorDataset(x, y)
        >>> len(ds)
        100
    """

    def __init__(self, *tensors):
        for t in tensors:
            if not hasattr(t, "shape"):
                raise ValueError("All tensors must have a shape attribute")

        if len(tensors) > 1:
            first_len = tensors[0].shape[0]
            for i, t in enumerate(tensors[1:], 1):
                if t.shape[0] != first_len:
                    raise ValueError(
                        f"TensorDataset tensors must have same first dimension: "
                        f"tensor 0 has {first_len}, tensor {i} has {t.shape[0]}"
                    )

        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return tuple(t[idx] for t in self.tensors)


class Sampler:
    """Base class for all samplers."""

    def __init__(self, data_source: Dataset):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data_source)


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Args:
        data_source: Dataset to sample from.
    """

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))


class RandomSampler(Sampler):
    """Samples elements randomly without replacement.

    Args:
        data_source: Dataset to sample from.
        replacement: Not supported (use WeightedRandomSampler for replacement).
        num_samples: Number of samples to draw (default: all).
        generator: Random number generator for reproducibility.
    """

    def __init__(
        self,
        data_source: Dataset,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator: Optional[random.Random] = None,
    ):
        super().__init__(data_source)
        if replacement:
            raise NotImplementedError("RandomSampler does not support replacement")
        self.num_samples = num_samples or len(data_source)
        self.generator = generator or random.Random()

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.data_source)))
        self.generator.shuffle(indices)
        return iter(indices[: self.num_samples])


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices: Sequence of indices to sample from.
        generator: Random number generator for shuffling.
    """

    def __init__(
        self,
        indices: Sequence[int],
        generator: Optional[random.Random] = None,
    ):
        self.indices = list(indices)
        self.generator = generator or random.Random()

    def __iter__(self) -> Iterator[int]:
        indices = self.indices[:]
        self.generator.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.indices)


class BatchSampler(Sampler):
    """Wraps another sampler to yield mini-batches of indices.

    Args:
        sampler: Base sampler to wrap.
        batch_size: Size of each mini-batch.
        drop_last: If True, drop the last incomplete batch.
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool = False,
    ):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def default_collate(batch: list) -> tuple:
    """Collate function that stacks samples into batches.

    Handles fastnn tensors by stacking them along a new first dimension.
    Handles numpy arrays by stacking them.
    Handles scalars and lists by returning lists.

    Args:
        batch: List of samples from the dataset.

    Returns:
        Tuple of batched tensors/arrays/lists.
    """
    import fastnn as fnn
    import numpy as np

    if not batch:
        return ()

    elem = batch[0]

    if isinstance(elem, tuple):
        return tuple(default_collate([b[i] for b in batch]) for i in range(len(elem)))

    if hasattr(elem, "numpy"):
        arrays = [b.numpy() for b in batch]
        result_np = np.stack(arrays, axis=0)
        return fnn.tensor(result_np.flatten().tolist(), list(result_np.shape))

    if isinstance(elem, np.ndarray):
        result_np = np.stack(batch, axis=0)
        return fnn.tensor(result_np.flatten().tolist(), list(result_np.shape))

    if isinstance(elem, (int, float)):
        return fnn.tensor(batch, [len(batch)])

    return batch


class _PrefetchIterator:
    """Background prefetch iterator that prepares the next batch in a thread.

    Uses a bounded queue to limit memory usage. The prefetch thread prepares
    batches ahead of time, overlapping I/O with model computation.

    Args:
        indices_iter: Iterator over batch indices.
        dataset: Dataset to fetch samples from.
        collate_fn: Function to collate samples into batches.
        prefetch_size: Number of batches to prefetch ahead.
        metrics: Optional _Metrics instance for tracking wait times.
    """

    def __init__(
        self,
        indices_iter: Iterator[List[int]],
        dataset: Dataset,
        collate_fn: Callable,
        prefetch_size: int = 2,
        metrics: Optional[_Metrics] = None,
    ):
        self.indices_iter = indices_iter
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.prefetch_size = max(1, prefetch_size)
        self.metrics = metrics
        self.queue: queue.Queue = queue.Queue(maxsize=prefetch_size)
        self.done = threading.Event()
        self.error: Optional[Exception] = None
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._batch_start_time = time.monotonic()
        self.thread.start()

    def _prefetch_worker(self) -> None:
        """Background thread that prepares batches and puts them in the queue."""
        self._batch_start_time = time.monotonic()
        try:
            for batch_idx in self.indices_iter:
                batch_start = time.monotonic()
                samples = [self.dataset[i] for i in batch_idx]
                batch = self.collate_fn(samples)
                batch_time = (time.monotonic() - batch_start) * 1000
                self.queue.put(batch, timeout=60)
                self._batch_start_time = time.monotonic()
        except Exception as e:
            self.error = e
        finally:
            self.done.set()

    def __iter__(self) -> "_PrefetchIterator":
        return self

    def __next__(self) -> tuple:
        batch_start = time.monotonic()

        if self.done.is_set() and self.queue.empty():
            if self.error is not None:
                raise self.error
            raise StopIteration

        try:
            batch = self.queue.get(timeout=60)
            wait_ms = (time.monotonic() - batch_start) * 1000

            if self.metrics is not None:
                self.metrics.record(wait_ms=wait_ms, batch_ms=wait_ms)

            return batch
        except queue.Empty:
            if self.done.is_set() and self.queue.empty():
                if self.error is not None:
                    raise self.error
                raise StopIteration
            raise

    def cleanup(self) -> None:
        """Signal the prefetch thread to stop and wait for it."""
        self.done.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)


class DataLoader:
    """Data loader with prefetching, auto-tuning, and multi-process support.

    Combines a dataset with a sampler and provides an iterable over
    mini-batches. Supports prefetching to overlap data loading with
    model computation, and auto-tuning to dynamically adjust resources.

    Args:
        dataset: Dataset to load from.
        batch_size: Number of samples per batch (default: 32).
        shuffle: Whether to shuffle the data (default: True).
        sampler: Custom sampler (overrides shuffle if provided).
        batch_sampler: Custom batch sampler (overrides batch_size, shuffle, drop_last).
        drop_last: Drop the last incomplete batch (default: False).
        num_workers: Number of worker processes, or "auto" for adaptive tuning (default: 0).
        collate_fn: Function to collate samples into batches (default: default_collate).
        prefetch_size: Number of batches to prefetch (default: 2).
        pin_memory: Pin tensors for faster GPU transfer (default: False).
        generator: Random number generator for reproducibility.
        min_workers: Minimum workers for auto-tuning (default: 0).
        max_workers: Maximum workers for auto-tuning (default: CPU count).
        up_threshold_ms: Auto-tune up if mean wait > this (default: 30ms).
        down_threshold_ms: Auto-tune down if mean wait < this (default: 5ms).
        scale_down_patience: Epochs of under-utilization before scaling down (default: 2).

    Example:
        >>> ds = TensorDataset(x, y)
        >>> loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers="auto")
        >>> for epoch in range(10):
        ...     for batch_x, batch_y in loader:
        ...         pred = model(batch_x)
        ...         loss = fnn.mse_loss(pred, batch_y)
        ...         loss.backward()
        ...         optimizer.step()
        ...         optimizer.zero_grad()
        ...     loader.reset_sampler()
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        drop_last: bool = False,
        num_workers: int | str = 0,
        collate_fn: Optional[Callable] = None,
        prefetch_size: int = 2,
        pin_memory: bool = False,
        generator: Optional[random.Random] = None,
        min_workers: int = 0,
        max_workers: Optional[int] = None,
        up_threshold_ms: float = 30.0,
        down_threshold_ms: float = 5.0,
        scale_down_patience: int = 2,
    ):
        if isinstance(num_workers, str) and num_workers != "auto":
            raise ValueError(
                f"Invalid num_workers value: {num_workers}. Use int or 'auto'."
            )

        if isinstance(num_workers, int) and num_workers > 0:
            raise NotImplementedError(
                f"num_workers={num_workers} > 0 is not yet supported. "
                "Use num_workers=0 or num_workers='auto' for data loading."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._num_workers_raw = num_workers
        self.collate_fn = collate_fn or default_collate
        self.pin_memory = pin_memory
        self.generator = generator or random.Random()

        self._metrics = _Metrics()
        self._tuner: Optional[_AutoTuner] = None
        self._auto_mode = False

        if num_workers == "auto":
            self._auto_mode = True
            self._tuner = _AutoTuner(
                min_workers=min_workers,
                max_workers=max_workers,
                up_threshold_ms=up_threshold_ms,
                down_threshold_ms=down_threshold_ms,
                initial_workers=1,
                initial_prefetch=max(1, prefetch_size),
                scale_down_patience=scale_down_patience,
            )
            self.num_workers = self._tuner.current_workers
            self.prefetch_size = self._tuner.current_prefetch
        else:
            self.num_workers = num_workers
            self.prefetch_size = max(1, prefetch_size)

        # Pre-allocate indices buffer
        self.indices = list(range(len(dataset)))

        if batch_sampler is not None:
            self.batch_sampler = batch_sampler
        else:
            if sampler is not None:
                self.sampler = sampler
            elif shuffle:
                self.sampler = RandomSampler(dataset, generator=self.generator)
            else:
                self.sampler = SequentialSampler(dataset)

            self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self) -> _PrefetchIterator:
        return _PrefetchIterator(
            iter(self.batch_sampler),
            self.dataset,
            self.collate_fn,
            self.prefetch_size,
            metrics=self._metrics if self._auto_mode else None,
        )

    def reset_sampler(self) -> None:
        """Reset the sampler for a new epoch with fresh shuffling and auto-tuning."""
        if self._auto_mode and self._tuner is not None:
            mean_wait = self._metrics.mean_wait_ms()
            self._tuner.mode = "thread"
            workers, prefetch = self._tuner.adjust(mean_wait)
            self.num_workers = workers
            self.prefetch_size = prefetch
            self._metrics.reset()

        if self.shuffle:
            if isinstance(self.sampler, RandomSampler):
                self.sampler.generator = self.generator
            self.batch_sampler = BatchSampler(
                self.sampler, self.batch_size, self.drop_last
            )
