"""Tests for the shared benchmark utility module."""

import time
import pytest
from tests.benchmark_utils import (
    BenchmarkTimer,
    CompareTimer,
    TimerResult,
    benchmark,
    bench_fn,
    bench_fn_simple,
)


class TestTimerResult:
    """Test TimerResult dataclass."""

    def test_times_conversion_us(self):
        result = TimerResult(
            mean=100.0,
            std=10.0,
            median=95.0,
            min=90.0,
            max=110.0,
            unit="us",
            iterations=10,
            times_raw=[0.0001] * 10,  # 100μs each
        )
        assert len(result.times) == 10
        assert all(t == 100.0 for t in result.times)

    def test_times_conversion_ms(self):
        result = TimerResult(
            mean=1.0,
            std=0.1,
            median=0.95,
            min=0.9,
            max=1.1,
            unit="ms",
            iterations=10,
            times_raw=[0.001] * 10,  # 1ms each
        )
        assert len(result.times) == 10
        assert all(abs(t - 1.0) < 0.001 for t in result.times)

    def test_str_with_std(self):
        result = TimerResult(
            mean=100.0,
            std=10.0,
            median=95.0,
            min=90.0,
            max=110.0,
            unit="us",
            iterations=10,
            times_raw=[0.0001] * 10,
        )
        s = str(result)
        assert "100.0" in s
        assert "±" in s
        assert "us" in s

    def test_format_mean(self):
        result = TimerResult(
            mean=123.456,
            std=10.0,
            median=120.0,
            min=100.0,
            max=150.0,
            unit="us",
            iterations=10,
            times_raw=[0.0001] * 10,
        )
        assert result.format("mean", 1) == "123.5us"
        assert result.format("mean", 0) == "123us"

    def test_format_median(self):
        result = TimerResult(
            mean=100.0,
            std=10.0,
            median=95.555,
            min=90.0,
            max=110.0,
            unit="us",
            iterations=10,
            times_raw=[0.0001] * 10,
        )
        assert result.format("median", 1) == "95.6us"


class TestBenchmarkTimer:
    """Test BenchmarkTimer class."""

    def test_basic_measurement(self):
        timer = BenchmarkTimer(warmup=2, iterations=5, unit="us")
        result = timer.measure(lambda: time.sleep(0.001))
        assert result.mean > 0
        assert result.iterations == 5
        assert result.unit == "us"

    def test_measurement_ms(self):
        timer = BenchmarkTimer(warmup=2, iterations=5, unit="ms")
        result = timer.measure(lambda: time.sleep(0.001))
        assert result.mean > 0
        assert result.unit == "ms"

    def test_value_method(self):
        timer = BenchmarkTimer(warmup=2, iterations=5, unit="us")
        mean_us = timer.value(lambda: time.sleep(0.001))
        assert mean_us > 0

    def test_std_with_single_iteration(self):
        timer = BenchmarkTimer(warmup=0, iterations=1, unit="us")
        result = timer.measure(lambda: time.sleep(0.001))
        assert result.std == 0.0

    def test_std_with_multiple_iterations(self):
        timer = BenchmarkTimer(warmup=0, iterations=10, unit="us")
        result = timer.measure(lambda: time.sleep(0.001))
        assert result.std >= 0

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="unit must be"):
            BenchmarkTimer(unit="invalid")

    def test_compare_method(self):
        timer = BenchmarkTimer(warmup=2, iterations=5, unit="us")
        results = timer.compare(
            {
                "op1": lambda: time.sleep(0.001),
                "op2": lambda: time.sleep(0.002),
            }
        )
        assert "op1" in results
        assert "op2" in results
        assert results["op1"].mean < results["op2"].mean

    def test_result_statistics(self):
        timer = BenchmarkTimer(warmup=0, iterations=10, unit="us")
        result = timer.measure(lambda: 1 + 1)  # Very fast operation
        assert result.mean > 0
        assert result.median > 0
        assert result.min > 0
        assert result.max >= result.min
        assert result.times_raw is not None
        assert len(result.times_raw) == 10


class TestCompareTimer:
    """Test CompareTimer class."""

    def test_basic_comparison(self, capsys):
        timer = CompareTimer(warmup=2, iterations=5, unit="us")
        timer.add("fast", lambda: time.sleep(0.001), color="green")
        timer.add("slow", lambda: time.sleep(0.002), color="yellow")
        results = timer.compare()

        assert "fast" in results
        assert "slow" in results
        assert results["fast"].mean < results["slow"].mean

        captured = capsys.readouterr()
        assert "Benchmark Comparison" in captured.out
        assert "fast" in captured.out
        assert "slow" in captured.out

    def test_empty_comparison(self, capsys):
        timer = CompareTimer()
        results = timer.compare()
        assert results == {}
        captured = capsys.readouterr()
        assert "No implementations" in captured.out

    def test_single_implementation(self, capsys):
        timer = CompareTimer(warmup=1, iterations=3, unit="us")
        timer.add("only", lambda: time.sleep(0.001), color="blue")
        results = timer.compare()

        assert "only" in results
        captured = capsys.readouterr()
        assert "only" in captured.out


class TestLegacyFunctions:
    """Test backward-compatible legacy functions."""

    def test_benchmark_legacy(self):
        mean_us, std_us = benchmark("test", lambda: time.sleep(0.001), warmup=2, iterations=5)
        assert mean_us > 0
        assert std_us >= 0

    def test_bench_fn(self):
        result_ms = bench_fn(lambda: time.sleep(0.001), warmup=2, iters=5)
        assert result_ms > 0

    def test_bench_fn_simple(self):
        result_ms = bench_fn_simple(lambda: time.sleep(0.001), warmup=2, iters=5)
        assert result_ms > 0


class TestIntegration:
    """Integration tests with actual fastnn operations."""

    def test_fastnn_add_benchmark(self):
        import fastnn as fnn

        timer = BenchmarkTimer(warmup=5, iterations=10, unit="us")
        a = fnn.rand([100, 100])
        b = fnn.rand([100, 100])
        result = timer.measure(lambda: a + b)

        assert result.mean > 0
        assert result.unit == "us"

    def test_fastnn_matmul_benchmark(self):
        import fastnn as fnn

        timer = BenchmarkTimer(warmup=2, iterations=5, unit="us")
        a = fnn.rand([32, 64])
        b = fnn.rand([64, 32])
        result = timer.measure(lambda: a @ b)

        assert result.mean > 0

    def test_compare_timer_with_fastnn(self, capsys):
        import fastnn as fnn

        a = fnn.rand([100, 100])
        b = fnn.rand([100, 100])

        timer = CompareTimer(warmup=5, iterations=10, unit="us")
        timer.add("add", lambda: a + b, color="green")
        timer.add("mul", lambda: a * b, color="yellow")
        results = timer.compare()

        assert "add" in results
        assert "mul" in results
        captured = capsys.readouterr()
        assert "Benchmark Comparison" in captured.out
