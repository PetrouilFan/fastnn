"""
Shared benchmark utility for consistent timing across all fastnn benchmarks.

Provides a standardized interface for microbenchmarking with configurable
units, statistics, and warmup behavior.

Usage:
    from tests.benchmark_utils import BenchmarkTimer, TimerResult

    # Simple usage - returns mean time in microseconds
    timer = BenchmarkTimer(warmup=10, iterations=100, unit="us")
    result = timer.measure(lambda: my_operation())
    print(f"Mean: {result.mean:.1f}μs, Std: {result.std:.1f}μs")

    # With median (robust to outliers)
    result = timer.measure(lambda: my_operation(), statistic="median")

    # Millisecond output
    timer = BenchmarkTimer(unit="ms")
    result = timer.measure(lambda: my_operation())
    print(f"Mean: {result.mean:.3f}ms")

    # Just the value (backward compatible)
    mean_us = timer.value(lambda: my_operation())

    # Compare two implementations
    timer = CompareTimer()
    timer.add("fastnn", lambda: fnn_op(), color="green")
    timer.add("pytorch", lambda: torch_op(), color="yellow")
    timer.compare()
"""

import time
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List


# ANSI color codes for terminal output
_COLORS = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "white": "\033[0m",
    "bold": "\033[1m",
}
_RESET = "\033[0m"


@dataclass
class TimerResult:
    """Result of a benchmark measurement.

    Attributes:
        mean: Mean time across all iterations (in configured units).
        std: Standard deviation (in configured units), 0 if not applicable.
        median: Median time (in configured units).
        min: Minimum time (in configured units).
        max: Maximum time (in configured units).
        unit: Time unit ("us" for microseconds, "ms" for milliseconds).
        iterations: Number of iterations measured.
        times_raw: Raw times in seconds (for further analysis).
    """
    mean: float
    std: float
    median: float
    min: float
    max: float
    unit: str
    iterations: int
    times_raw: List[float]

    @property
    def times(self) -> List[float]:
        """Return times converted to the configured unit."""
        factor = 1000.0 if self.unit == "ms" else 1e6
        return [t * factor for t in self.times_raw]

    def __str__(self) -> str:
        if self.std > 0:
            return f"{self.mean:.1f} ± {self.std:.1f} {self.unit}"
        return f"{self.mean:.1f} {self.unit}"

    def format(self, statistic: str = "mean", precision: int = 1) -> str:
        """Format the result using a specific statistic.

        Args:
            statistic: One of "mean", "median", "min", "max".
            precision: Decimal places for formatting.

        Returns:
            Formatted string like "123.4μs".
        """
        value = getattr(self, statistic)
        return f"{value:.{precision}f}{self.unit}"


class BenchmarkTimer:
    """Configurable benchmark timer for consistent measurements.

    This class provides a standardized way to benchmark code snippets
    with configurable warmup, iteration count, time units, and statistics.

    Args:
        warmup: Number of warmup iterations (not measured). Default 10.
        iterations: Number of measured iterations. Default 100.
        unit: Time unit for results: "us" (microseconds) or "ms" (milliseconds).
              Default "us".
        gc: Whether to run garbage collection before each iteration.
            Default False (faster, but may have more variance).
    """

    def __init__(
        self,
        warmup: int = 10,
        iterations: int = 100,
        unit: str = "us",
        gc: bool = False,
    ):
        if unit not in ("us", "ms"):
            raise ValueError(f"unit must be 'us' or 'ms', got {unit!r}")
        self.warmup = warmup
        self.iterations = iterations
        self.unit = unit
        self.gc = gc

    def measure(
        self,
        fn: Callable[[], None],
        statistic: str = "mean",
    ) -> TimerResult:
        """Measure execution time of a function.

        Args:
            fn: Callable to benchmark. Should take no arguments.
            statistic: Not used (kept for backward compatibility).

        Returns:
            TimerResult with timing statistics.
        """
        # Warmup phase
        for _ in range(self.warmup):
            fn()

        # Measurement phase
        times: List[float] = []
        for _ in range(self.iterations):
            if self.gc:
                import gc
                gc.collect()
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            times.append(end - start)

        # Convert to configured unit
        factor = 1000.0 if self.unit == "ms" else 1e6
        times_scaled = [t * factor for t in times]

        return TimerResult(
            mean=statistics.mean(times_scaled),
            std=statistics.stdev(times_scaled) if len(times_scaled) > 1 else 0.0,
            median=statistics.median(times_scaled),
            min=min(times_scaled),
            max=max(times_scaled),
            unit=self.unit,
            iterations=self.iterations,
            times_raw=times,
        )

    def value(
        self,
        fn: Callable[[], None],
        statistic: str = "mean",
    ) -> float:
        """Convenience method returning just the mean time.

        This is for backward compatibility with existing benchmarks that
        expect a single float return value.

        Args:
            fn: Callable to benchmark.
            statistic: Not used (kept for backward compatibility).

        Returns:
            Mean execution time in configured units.
        """
        return self.measure(fn).mean

    def compare(
        self,
        implementations: Dict[str, Callable[[], None]],
        statistic: str = "mean",
    ) -> Dict[str, TimerResult]:
        """Benchmark multiple implementations and return all results.

        Args:
            implementations: Mapping from name to callable.
            statistic: Statistic to use (passed to measure).

        Returns:
            Dict mapping names to TimerResult objects.
        """
        results: Dict[str, TimerResult] = {}
        for name, fn in implementations.items():
            results[name] = self.measure(fn, statistic)
        return results


class CompareTimer:
    """Utility for comparing multiple implementations side-by-side.

    This provides a convenient way to benchmark and display comparisons
    between different implementations (e.g., fastnn vs PyTorch).

    Usage:
        timer = CompareTimer()
        timer.add("fastnn", lambda: fnn_op(), color="green")
        timer.add("pytorch", lambda: torch_op(), color="yellow")
        timer.compare()
    """

    def __init__(
        self,
        warmup: int = 10,
        iterations: int = 100,
        unit: str = "us",
    ):
        self.warmup = warmup
        self.iterations = iterations
        self.unit = unit
        self._impls: List[tuple[str, Callable[[], None], str]] = []

    def add(
        self,
        name: str,
        fn: Callable[[], None],
        color: str = "white",
    ) -> None:
        """Add an implementation to compare.

        Args:
            name: Display name for this implementation.
            fn: Callable to benchmark.
            color: Color name for terminal output (green, yellow, blue, etc.).
        """
        self._impls.append((name, fn, color))

    def compare(self) -> Dict[str, TimerResult]:
        """Run benchmarks and display comparison table.

        Returns:
            Dict mapping implementation names to TimerResult objects.
        """
        if not self._impls:
            print("No implementations to compare.")
            return {}

        timer = BenchmarkTimer(
            warmup=self.warmup,
            iterations=self.iterations,
            unit=self.unit,
        )

        results: Dict[str, TimerResult] = {}
        max_name_len = max(len(name) for name, _, _ in self._impls)

        print(f"\n{'=' * 70}")
        print(f"Benchmark Comparison ({self.unit})")
        print(f"{'=' * 70}")
        print(f"{'Implementation':<{max_name_len + 2}} {'Mean':>12} {'Std':>12} {'Median':>12} {'Min':>12} {'Max':>12}")
        print("-" * 70)

        for name, fn, color in self._impls:
            result = timer.measure(fn)
            results[name] = result
            color_code = _COLORS.get(color, "")
            print(
                f"{color_code}{name:<{max_name_len + 2}} "
                f"{result.mean:>12.1f} {result.std:>12.1f} "
                f"{result.median:>12.1f} {result.min:>12.1f} {result.max:>12.1f}{_RESET}"
            )

        # Show speedups if we have exactly 2 implementations
        if len(results) == 2:
            names = list(results.keys())
            base = results[names[0]].mean
            comp = results[names[1]].mean
            speedup = base / comp if comp > 0 else 0
            if speedup > 1:
                print(f"\n{names[1]} is {speedup:.2f}x faster than {names[0]}")
            elif speedup < 1:
                print(f"\n{names[0]} is {1/speedup:.2f}x faster than {names[1]}")
            else:
                print("\nBoth implementations have equal performance")

        print(f"{'=' * 70}\n")
        return results


# =============================================================================
# Backward-compatible standalone functions
# =============================================================================
# These functions match the signatures used in existing benchmark files.

def benchmark(
    name: str,
    fn: Callable[[], None],
    warmup: int = 10,
    iterations: int = 100,
) -> tuple[float, float]:
    """Legacy benchmark function matching bench_fastnn.py, bench_gpu.py, etc.

    Args:
        name: Ignored (kept for compatibility).
        fn: Callable to benchmark.
        warmup: Number of warmup iterations.
        iterations: Number of measured iterations.

    Returns:
        Tuple of (mean_time_us, std_time_us).
    """
    timer = BenchmarkTimer(warmup=warmup, iterations=iterations, unit="us")
    result = timer.measure(fn)
    return result.mean, result.std


def bench_fn(
    fn: Callable[[], None],
    warmup: int = 20,
    iters: int = 500,
) -> float:
    """Legacy benchmark function matching bench_compare.py.

    Returns the minimum of 3 median measurements in milliseconds.

    Args:
        fn: Callable to benchmark.
        warmup: Number of warmup iterations per trial.
        iters: Number of measured iterations per trial.

    Returns:
        Best median time in milliseconds.
    """
    import gc
    medians = []
    for _trial in range(3):
        gc.collect()
        timer = BenchmarkTimer(warmup=warmup, iterations=iters, unit="ms")
        result = timer.measure(fn)
        medians.append(result.median)
    return min(medians)


def bench_fn_simple(
    fn: Callable[[], None],
    warmup: int = 10,
    iters: int = 200,
) -> float:
    """Legacy benchmark function matching bench_pytorch.py.

    Returns the median time in milliseconds.

    Args:
        fn: Callable to benchmark.
        warmup: Number of warmup iterations.
        iters: Number of measured iterations.

    Returns:
        Median time in milliseconds.
    """
    timer = BenchmarkTimer(warmup=warmup, iterations=iters, unit="ms")
    result = timer.measure(fn)
    return result.median
