"""Comparison framework for LLM models.

This module provides tools to compare outputs, performance, and memory usage
between different model implementations (e.g., fastnn vs transformers).
"""

import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np

import fastnn._core as _core


@dataclass
class ComparisonResult:
    """Results from a model comparison."""
    
    # Output similarity metrics
    logits_mse: float = 0.0
    logits_cosine_sim: float = 0.0
    top1_match: bool = False
    top5_overlap: float = 0.0
    top10_overlap: float = 0.0
    
    # Performance metrics
    load_time_diff: float = 0.0
    inference_time_diff: float = 0.0
    tokens_per_second: float = 0.0
    
    # Memory metrics
    memory_peak_mb: float = 0.0
    
    # Raw data
    fastnn_logits: np.ndarray = None
    transformers_logits: np.ndarray = None
    
    def print_report(self):
        """Print a formatted comparison report."""
        print("\n" + "=" * 60)
        print("          LLM MODEL COMPARISON REPORT")
        print("=" * 60)
        
        print("\n📊 OUTPUT SIMILARITY:")
        print(f"  Logits MSE:            {self.logits_mse:.6f}")
        print(f"  Cosine Similarity:     {self.logits_cosine_sim:.4f}")
        print(f"  Top-1 Match:          {'✓ YES' if self.top1_match else '✗ NO'}")
        print(f"  Top-5 Overlap:         {self.top5_overlap*100:.1f}%")
        print(f"  Top-10 Overlap:        {self.top10_overlap*100:.1f}%")
        
        print("\n⚡ PERFORMANCE:")
        print(f"  Load Time Diff:       {self.load_time_diff:+.2f}s")
        print(f"  Inference Diff:       {self.inference_time_diff:+.4f}s")
        print(f"  Tokens/Second:        {self.tokens_per_second:.1f}")
        
        print("\n💾 MEMORY:")
        print(f"  Peak Memory:          {self.memory_peak_mb:.1f} MB")
        
        print("\n" + "=" * 60)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "logits_mse": float(self.logits_mse),
            "logits_cosine_sim": float(self.logits_cosine_sim),
            "top1_match": self.top1_match,
            "top5_overlap": float(self.top5_overlap),
            "top10_overlap": float(self.top10_overlap),
            "load_time_diff": float(self.load_time_diff),
            "inference_time_diff": float(self.inference_time_diff),
            "tokens_per_second": float(self.tokens_per_second),
            "memory_peak_mb": float(self.memory_peak_mb),
        }


@dataclass
class ModelOutput:
    """Wrapper for model output with metadata."""
    
    logits: np.ndarray
    tokens: List[int]
    time_taken: float
    tokenizer: Any = None
    
    @property
    def last_logits(self) -> np.ndarray:
        """Get logits for last token."""
        return self.logits[0, -1, :]
    
    @property
    def top_tokens(self) -> np.ndarray:
        """Get top token IDs by probability."""
        return np.argsort(self.last_logits)[::-1]
    
    @property
    def top_probs(self) -> np.ndarray:
        """Get top token probabilities."""
        probs = np.exp(self.last_logits) / np.sum(np.exp(self.last_logits))
        return probs[self.top_tokens]


class ModelComparator:
    """Compare outputs and performance between different model implementations.
    
    Usage:
        comparator = ModelComparator({
            "fastnn": fastnn_model,
            "transformers": transformers_model
        })
        
        result = comparator.compare("Hello world")
        result.print_report()
    """
    
    def __init__(
        self,
        models: Dict[str, Any],
        tokenizer: Any = None,
        default_device: str = "cpu"
    ):
        """
        Args:
            models: Dict of {name: model} pairs
            tokenizer: Optional tokenizer for encoding prompts
            default_device: Default device for models
        """
        self.models = models
        self.tokenizer = tokenizer
        self.default_device = default_device
    
    def compare(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        check_correctness: bool = True
    ) -> ComparisonResult:
        """Run comparison on a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            check_correctness: Whether to check output similarity
            
        Returns:
            ComparisonResult
        """
        result = ComparisonResult()
        
        # Generate with each model
        outputs = {}
        
        for name, model in self.models.items():
            print(f"  Running {name}...")
            
            start = time.time()
            output = self._generate(
                model, 
                prompt, 
                max_new_tokens,
                temperature
            )
            elapsed = time.time() - start
            
            output.time_taken = elapsed
            outputs[name] = output
            
            print(f"    Generated in {elapsed:.2f}s")
        
        # Compare
        if check_correctness and len(outputs) == 2:
            fastnn_out = outputs.get("fastnn") or outputs.get("fastnn_loader")
            transformers_out = outputs.get("transformers") or outputs.get("transformers_loader")
            
            if fastnn_out and transformers_out:
                # Compute similarity metrics
                result.logits_mse = self._compute_mse(
                    fastnn_out.last_logits,
                    transformers_out.last_logits
                )
                result.logits_cosine_sim = self._compute_cosine(
                    fastnn_out.last_logits,
                    transformers_out.last_logits
                )
                result.top1_match = (
                    fastnn_out.top_tokens[0] == transformers_out.top_tokens[0]
                )
                result.top5_overlap = self._compute_overlap(
                    fastnn_out.top_tokens[:5],
                    transformers_out.top_tokens[:5]
                )
                result.top10_overlap = self._compute_overlap(
                    fastnn_out.top_tokens[:10],
                    transformers_out.top_tokens[:10]
                )
                
                result.fastnn_logits = fastnn_out.last_logits
                result.transformers_logits = transformers_out.last_logits
        
        # Performance comparison
        if len(outputs) >= 2:
            times = [o.time_taken for o in outputs.values()]
            result.inference_time_diff = max(times) - min(times)
            
            # Tokens per second (use first model as reference)
            first_output = list(outputs.values())[0]
            if first_output.time_taken > 0:
                result.tokens_per_second = len(first_output.tokens) / first_output.time_taken
        
        return result
    
    def compare_multiple(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[ComparisonResult]:
        """Run comparison on multiple prompts.
        
        Args:
            prompts: List of prompts
            **kwargs: Arguments for compare()
            
        Returns:
            List of ComparisonResult
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt[:50]}...")
            result = self.compare(prompt, **kwargs)
            results.append(result)
        
        return results
    
    def benchmark(
        self,
        prompt: str,
        num_runs: int = 10,
        warmup: int = 2
    ) -> Dict[str, Any]:
        """Benchmark model performance.
        
        Args:
            prompt: Test prompt
            num_runs: Number of runs
            warmup: Number of warmup runs
            
        Returns:
            Benchmark results
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"\nBenchmarking {name}...")
            
            # Warmup
            for _ in range(warmup):
                self._generate(model, prompt, 10, 1.0)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.time()
                self._generate(model, prompt, 10, 1.0)
                times.append(time.time() - start)
            
            results[name] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "times": times
            }
            
            print(f"  Mean: {np.mean(times)*1000:.1f}ms ± {np.std(times)*1000:.1f}ms")
        
        return results
    
    def _generate(
        self,
        model: Any,
        prompt: str,
        max_new_tokens: int,
        temperature: float
    ) -> ModelOutput:
        """Generate output from a model."""
        
        # Tokenize
        if self.tokenizer:
            input_ids = self.tokenizer.encode(prompt)
        else:
            # Simple tokenization
            input_ids = [ord(c) % 65536 for c in prompt[:32]]
            if not input_ids:
                input_ids = [1]
        
        input_ids = np.array([input_ids], dtype=np.int32)
        
        # Generate token by token
        generated = list(input_ids[0])
        
        for _ in range(max_new_tokens):
            # Forward
            input_tensor = _core.tensor_from_data(
                generated,
                [1, len(generated)]
            )
            
            if hasattr(model, 'forward'):
                logits = model.forward(input_tensor)
            else:
                # Transformers model
                logits = model(input_tensor).logits
            
            # Get last token logits
            if hasattr(logits, 'numpy'):
                logits_np = logits.numpy()
            else:
                logits_np = logits
            
            last_logits = logits_np[0, -1, :]
            
            # Sample
            if temperature > 0:
                last_logits = last_logits / temperature
                probs = np.exp(last_logits) / np.sum(np.exp(last_logits))
                next_token = int(np.random.choice(len(probs), p=probs))
            else:
                next_token = int(np.argmax(last_logits))
            
            generated.append(next_token)
            
            # Check for EOS
            if hasattr(model, 'config'):
                eos = getattr(model.config, 'eos_token_id', 7)
                if next_token == eos:
                    break
        
        # Get final logits for comparison
        final_input = _core.tensor_from_data(
            generated,
            [1, len(generated)]
        )
        
        if hasattr(model, 'forward'):
            final_logits = model.forward(final_input)
        else:
            final_logits = model(final_input).logits
        
        return ModelOutput(
            logits=final_logits.numpy() if hasattr(final_logits, 'numpy') else final_logits,
            tokens=generated,
            time_taken=0.0,
            tokenizer=self.tokenizer
        )
    
    def _compute_mse(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute mean squared error."""
        return float(np.mean((a - b) ** 2))
    
    def _compute_cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))
    
    def _compute_overlap(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute overlap ratio between two arrays."""
        a_set = set(a)
        b_set = set(b)
        intersection = len(a_set & b_set)
        return float(intersection) / float(min(len(a_set), len(b_set)))
    
    def get_layer_outputs(
        self,
        prompt: str,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Get intermediate layer outputs for debugging.
        
        This is useful for understanding where outputs diverge.
        
        Args:
            prompt: Test prompt
            layer_names: Optional list of layer names to intercept
            
        Returns:
            Dict of {model_name: {layer_name: output}}
        """
        # This would require model modifications to implement
        # Placeholder for future enhancement
        raise NotImplementedError("Layer interception not yet implemented")


def quick_compare(
    model_path: str,
    prompt: str = "Hello, how are you?",
    max_tokens: int = 20,
    model_type: Optional[str] = None
) -> ComparisonResult:
    """Quick comparison between fastnn and transformers.
    
    Convenience function for quick testing.
    
    Args:
        model_path: Path to model directory
        prompt: Test prompt
        max_tokens: Tokens to generate
        model_type: Optional model type
        
    Returns:
        ComparisonResult
    """
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load models
    print("Loading models...")
    
    from fastnn.models.llm.loaders import FastNNLoader, TransformersLoader
    
    fastnn_loader = FastNNLoader()
    fastnn_info = fastnn_loader.load(model_path, model_type)
    
    transformers_loader = TransformersLoader()
    transformers_info = transformers_loader.load(model_path, model_type)
    
    # Compare
    comparator = ModelComparator(
        {
            "fastnn": fastnn_info.model,
            "transformers": transformers_info.model
        },
        tokenizer=tokenizer
    )
    
    return comparator.compare(prompt, max_tokens)