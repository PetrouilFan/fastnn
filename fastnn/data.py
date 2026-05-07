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
import numpy as np
import fastnn as fnn


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
        self._numpy_tensors = None  # Lazy conversion

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return tuple(t[idx] for t in self.tensors)

    def __getstate__(self):
        """Return state for pickling - use numpy arrays (convert lazily)."""
        if self._numpy_tensors is None:
            self._numpy_tensors = tuple(
                t.numpy() if hasattr(t, "numpy") else t for t in self.tensors
            )
        return {"numpy_tensors": self._numpy_tensors}

    def __setstate__(self, state):
        """Restore state from pickling - use numpy arrays."""
        self._numpy_tensors = state["numpy_tensors"]
        # Reconstruct tensors from numpy (this is a limitation - we'll use numpy in workers)
        self.tensors = self._numpy_tensors


class CachedDataset(Dataset):
    """Dataset wrapper that caches items in memory for faster access.

    Useful for small to medium-sized datasets or frequently accessed items.
    Items are cached on first access and retained up to cache_size.

    Args:
        dataset: The dataset to wrap.
        cache_size: Maximum number of items to cache (default: 1000).
    """

    def __init__(self, dataset: Dataset, cache_size: int = 1000):
        self.dataset = dataset
        self.cache: dict[int, Any] = {}
        self.cache_size = cache_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        if idx in self.cache:
            return self.cache[idx]
        item = self.dataset[idx]
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item
        return item


def _convert_to_numpy(tensor):
    """Convert fastnn tensor to numpy array for pickling."""
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return tensor


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
        self._indices_buffer = None

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self._indices_buffer is None or len(self._indices_buffer) != n:
            self._indices_buffer = np.arange(n, dtype=np.int64)
        else:
            np.arange(n, dtype=np.int64, out=self._indices_buffer)
        self.generator.shuffle(self._indices_buffer)
        return iter(self._indices_buffer[: self.num_samples].tolist())


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
    if not batch:
        return ()

    elem = batch[0]

    if isinstance(elem, tuple):
        return tuple(map(default_collate, zip(*batch)))

    if hasattr(elem, "numpy"):
        arrays = [b.numpy() for b in batch]
        result_np = np.stack(arrays, axis=0)
        # tensor() accepts numpy arrays directly - no need for .ravel().tolist()
        return fnn.tensor(result_np, result_np.shape)

    if isinstance(elem, np.ndarray):
        result_np = np.stack(batch, axis=0)
        return fnn.tensor(result_np, result_np.shape)

    if isinstance(elem, (int, float)):
        return fnn.tensor(batch, [len(batch)])

    return batch


class _Metrics:
    """Thread-safe metrics tracking for adaptive data loading.

    Tracks wait times and batch times to inform auto-tuning decisions.
    Uses running sums for O(1) mean computation.
    """

    def __init__(self, window: int = 200):
        self._window = window
        self._wait_times: deque[float] = deque(maxlen=window)
        self._batch_times: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._wait_sum = 0.0
        self._batch_sum = 0.0

    def record(self, wait_ms: float, batch_ms: float) -> None:
        with self._lock:
            if len(self._wait_times) == self._window:
                self._wait_sum -= self._wait_times[0]
                self._batch_sum -= self._batch_times[0]
            self._wait_times.append(wait_ms)
            self._batch_times.append(batch_ms)
            self._wait_sum += wait_ms
            self._batch_sum += batch_ms

    def mean_wait_ms(self) -> float:
        with self._lock:
            if not self._wait_times:
                return 0.0
            return self._wait_sum / len(self._wait_times)

    def mean_batch_ms(self) -> float:
        with self._lock:
            if not self._batch_times:
                return 0.0
            return self._batch_sum / len(self._batch_times)

    def reset(self) -> None:
        with self._lock:
            self._wait_times.clear()
            self._batch_times.clear()
            self._wait_sum = 0.0
            self._batch_sum = 0.0


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


class _BaseIterator:
    """Base class for data iterators with shared cleanup, context manager, and metrics logic."""

    def __init__(self, metrics: Optional[_Metrics] = None):
        self.metrics = metrics

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources. Subclasses must implement this."""
        raise NotImplementedError

    def _record_metrics(self, wait_ms: float) -> None:
        """Record wait and batch times if metrics are enabled."""
        if self.metrics is not None:
            self.metrics.record(wait_ms=wait_ms, batch_ms=wait_ms)


class _PrefetchIterator(_BaseIterator):
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
        super().__init__(metrics=metrics)
        self.indices_iter = indices_iter
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.prefetch_size = max(1, prefetch_size)
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
                samples = [self.dataset[i] for i in batch_idx]
                batch = self.collate_fn(samples)
                self.queue.put(batch, timeout=60)
                self._batch_start_time = time.monotonic()
        except Exception as e:
            self.error = e
        finally:
            self.done.set()

    def __next__(self) -> tuple:
        batch_start = time.monotonic()

        if self.done.is_set() and self.queue.empty():
            if self.error is not None:
                raise self.error
            raise StopIteration

        try:
            batch = self.queue.get(timeout=60)
            wait_ms = (time.monotonic() - batch_start) * 1000
            self._record_metrics(wait_ms)
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
            self.thread.join(timeout=30)


class _MultiProcessIterator(_BaseIterator):
    """Multi-threaded iterator for parallel data loading.

    Uses a thread pool to fetch batches in parallel, overlapping I/O with
    model computation. This is the working implementation that avoids
    the pickle issues with fastnn tensors in multiprocessing.

    Args:
        indices_iter: Iterator over batch indices.
        dataset: Dataset to fetch samples from.
        collate_fn: Function to collate samples into batches.
        num_workers: Number of worker threads.
        prefetch_size: Number of batches to prefetch ahead.
        metrics: Optional _Metrics instance for tracking wait times.
    """

    def __init__(
        self,
        indices_iter: Iterator[List[int]],
        dataset: Dataset,
        collate_fn: Callable,
        num_workers: int = 2,
        prefetch_size: int = 2,
        metrics: Optional[_Metrics] = None,
    ):
        super().__init__(metrics=metrics)
        from concurrent.futures import ThreadPoolExecutor

        self.indices_iter = indices_iter
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.prefetch_size = max(1, prefetch_size)

        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._result_queue: queue.Queue = queue.Queue(maxsize=prefetch_size)
        self._done = threading.Event()
        self._error: Optional[Exception] = None
        self._started = False
        self._batch_iter = None

    def _start_workers(self) -> None:
        """Start worker thread."""
        if self._executor is None:
            return

        self._batch_iter = iter(self.indices_iter)

        def worker_fn():
            try:
                for batch_indices in self._batch_iter:
                    result = _worker_fetch_batch(
                        self.dataset, batch_indices, self.collate_fn
                    )
                    self._result_queue.put(result, timeout=60)
                self._done.set()
            except Exception as e:
                self._error = e
                self._done.set()

        self._thread = threading.Thread(target=worker_fn, daemon=True)
        self._thread.start()
        self._started = True

    def __next__(self) -> tuple:
        if not self._started:
            self._start_workers()

        batch_start = time.monotonic()

        try:
            result = self._result_queue.get(timeout=60)
            wait_ms = (time.monotonic() - batch_start) * 1000
            self._record_metrics(wait_ms)
            return result
        except queue.Empty:
            if self._done.is_set() and self._result_queue.empty():
                if self._error is not None:
                    raise self._error
                raise StopIteration
            raise

    def cleanup(self) -> None:
        """Shutdown the executor."""
        self._done.set()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None


def _worker_fetch_batch(
    dataset: Dataset, batch_indices: List[int], collate_fn: Callable
) -> tuple:
    """Fetch and collate a batch from the dataset."""
    samples = []
    for i in batch_indices:
        sample = dataset[i]
        # Convert fastnn tensors to numpy for consistency with collate
        if isinstance(sample, tuple):
            converted = tuple(s.numpy() if hasattr(s, "numpy") else s for s in sample)
            samples.append(converted)
        elif hasattr(sample, "numpy"):
            samples.append(sample.numpy())
        else:
            samples.append(sample)
    return collate_fn(samples)


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
        >>> loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
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

    def __iter__(self):
        """Return the appropriate iterator based on num_workers."""
        if self.num_workers > 0:
            return _MultiProcessIterator(
                iter(self.batch_sampler),
                self.dataset,
                self.collate_fn,
                num_workers=self.num_workers,
                prefetch_size=self.prefetch_size,
                metrics=self._metrics if self._auto_mode else None,
            )
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
            self._tuner.mode = "process" if self.num_workers > 0 else "thread"
            workers, prefetch = self._tuner.adjust(mean_wait)
            self.num_workers = workers
            self.prefetch_size = prefetch
            self._metrics.reset()

        if self.shuffle:
            if isinstance(self.sampler, RandomSampler):
                self.sampler.generator = self.generator
            # Only recreate BatchSampler if the sampler has changed
            if self.batch_sampler.sampler is not self.sampler:
                self.batch_sampler = BatchSampler(
                    self.sampler, self.batch_size, self.drop_last
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
