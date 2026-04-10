"""LLM inference utilities for fastnn.

This module provides components for running LLM inference:
- RoPE (Rotary Position Embedding) computation
- LlamaModel wrapper
- Generation utilities
"""

import numpy as np
import fastnn._core as _core


def compute_rope_inv_freq(dim: int, base: float = 500000.0, rope_ratio: float = 1.0):
    """Compute inverse frequencies for RoPE.

    Args:
        dim: Dimension of the attention head
        base: Base for the frequency computation
        rope_ratio: RoPE scaling ratio

    Returns:
        Array of inverse frequencies
    """
    inv_freq = np.zeros(dim // 2, dtype=np.float32)
    for i in range(dim // 2):
        inv_freq[i] = rope_ratio / (base ** (4.0 * i / dim))
    return inv_freq


def apply_rope(q, k, cos, sin):
    """Apply rotary embeddings to query and key.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # q and k should already be in [batch, heads, seq_len, head_dim] format
    # Apply rope: x_rot = x * cos + rotate_half(x) * sin

    # Get shapes
    q_shape = q.shape
    k_shape = k.shape

    # For now, implement in numpy for simplicity
    # This will be slow - we'll optimize later
    q_np = q.numpy()
    k_np = k.numpy()
    cos_np = cos.numpy()
    sin_np = sin.numpy()

    # Rotate q
    half_dim = q_shape[-1] // 2
    q1 = q_np[..., :half_dim]
    q2 = q_np[..., half_dim:]
    q_np_new = q1 * cos_np + np.concatenate([-q2, q1], axis=-1) * sin_np

    # Rotate k
    k1 = k_np[..., :half_dim]
    k2 = k_np[..., half_dim:]
    k_np_new = k1 * cos_np + np.concatenate([-k2, k1], axis=-1) * sin_np

    # Convert back to fastnn
    q_rotated = _core.tensor_from_data(
        q_np_new.flatten().tolist(), list(q_np_new.shape)
    )
    k_rotated = _core.tensor_from_data(
        k_np_new.flatten().tolist(), list(k_np_new.shape)
    )

    return q_rotated, k_rotated


def compute_rope_positions(
    seq_len: int, head_dim: int, base: float = 500000.0, rope_ratio: float = 1.0
):
    """Compute cos and sin for RoPE positions.

    Args:
        seq_len: Sequence length
        head_dim: Head dimension
        base: Base frequency
        rope_ratio: RoPE scaling ratio

    Returns:
        Tuple of (cos, sin) arrays, both [seq_len, head_dim]
    """
    inv_freq = compute_rope_inv_freq(head_dim, base, rope_ratio)

    positions = np.arange(seq_len, dtype=np.float32)

    # Compute angles for each position and frequency
    angles = np.outer(positions, inv_freq)  # [seq_len, dim/2]

    # Compute cos and sin
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)

    # Interleave: [cos, cos] and [sin, -sin]
    cos_full = np.concatenate([cos, cos], axis=-1)
    sin_full = np.concatenate([sin, -sin], axis=-1)

    return cos_full, sin_full


def generate_causal_mask(seq_len: int):
    """Generate causal mask for autoregressive generation.

    Args:
        seq_len: Sequence length

    Returns:
        Mask tensor [seq_len, seq_len] with -inf above diagonal
    """
    mask = np.full((seq_len, seq_len), 0.0, dtype=np.float32)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mask[i, j] = float("-inf")
    return _core.tensor_from_data(mask.flatten().tolist(), [seq_len, seq_len])


def top_k_top_p(logits, top_k: int = 0, top_p: float = 1.0, temperature: float = 1.0):
    """Apply top-k and top-p filtering to logits.

    Args:
        logits: Logits tensor [vocab_size]
        top_k: Top-k to keep (0 = all)
        top_p: Cumulative probability threshold (1.0 = all)
        temperature: Temperature for sampling

    Returns:
        Processed logits tensor
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    vocab_size = logits.shape[-1]
    logits_np = logits.numpy().flatten()

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, vocab_size)
        indices = np.argpartition(logits_np, -top_k)[-top_k:]
        mask = np.ones(vocab_size, dtype=np.float32)
        mask[np.setdiff1d(np.arange(vocab_size), indices)] = float("-inf")
        logits_np = logits_np + mask

    # Top-p filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(logits_np)[::-1]
        cumsum = np.cumsum(np.exp(logits_np[sorted_indices]))
        cumsum = cumsum / cumsum[-1]  # normalize

        # Find cutoff
        cutoff_idx = np.searchsorted(cumsum, top_p)
        valid_indices = sorted_indices[: cutoff_idx + 1]

        mask = np.ones(vocab_size, dtype=np.float32)
        mask[np.setdiff1d(np.arange(vocab_size), valid_indices)] = float("-inf")
        logits_np = logits_np + mask

    return _core.tensor_from_data(logits_np.tolist(), [vocab_size])


def sample_token(logits):
    """Sample next token from logits using greedy decoding.

    Args:
        logits: Logits tensor [vocab_size]

    Returns:
        Sampled token ID
    """
    return int(np.argmax(logits.numpy()))


def sample_token_temperature(logits, temperature: float = 0.7):
    """Sample next token with temperature.

    Args:
        logits: Logits tensor [vocab_size]
        temperature: Temperature for sampling (lower = more deterministic)

    Returns:
        Sampled token ID
    """
    logits_np = logits.numpy().flatten()

    # Apply temperature
    if temperature > 0:
        logits_np = logits_np / temperature

    # Softmax
    exp_logits = np.exp(
        logits_np - np.max(logits_np)
    )  # subtract max for numerical stability
    probs = exp_logits / np.sum(exp_logits)

    # Sample
    return int(np.random.choice(len(probs), p=probs))


class LlamaConfig:
    """Configuration for Llama model."""

    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 128256)
        self.hidden_size = kwargs.get("hidden_size", 2048)
        self.intermediate_size = kwargs.get("intermediate_size", 8192)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 16)
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
        self.head_dim = kwargs.get("head_dim", 64)
        self.rope_theta = kwargs.get("rope_theta", 500000.0)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-5)

        # Derived
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 131072)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        self.rope_ratio = 1.0
        if self.rope_scaling:
            self.rope_ratio = self.rope_scaling.get("factor", 1.0)

        self.hidden_act = kwargs.get("hidden_act", "silu")

        # Token IDs
        self.bos_token_id = kwargs.get("bos_token_id", 128000)
        self.eos_token_id = kwargs.get("eos_token_id", 128009)


def load_llama_config(config_path: str) -> LlamaConfig:
    """Load Llama config from JSON.

    Args:
        config_path: Path to config.json

    Returns:
        LlamaConfig object
    """
    import json

    with open(config_path, "r") as f:
        config = json.load(f)
    return LlamaConfig(**config)


def load_hf_weights(model_path: str):
    """Load weights from HuggingFace model directory.

    Args:
        model_path: Path to model directory containing .safetensors file

    Returns:
        Dictionary mapping parameter names to tensors
    """
    from fastnn.io import load_safetensors

    import os

    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        return load_safetensors(safetensors_path)

    # Try loading any .safetensors file
    for f in os.listdir(model_path):
        if f.endswith(".safetensors"):
            return load_safetensors(os.path.join(model_path, f))

    raise FileNotFoundError(f"No safetensors file found in {model_path}")


def remap_hf_state_dict(state_dict: dict, config: LlamaConfig):
    """Remap HuggingFace state dict to fastnn parameter names.

    HF parameter naming convention:
    - model.embed_tokens.weight -> embedding.weight
    - model.layers.{i}.attn.q_proj.weight -> layers.{i}.self_attn.q_proj.weight
    - model.layers.{i}.attn.k_proj.weight -> layers.{i}.self_attn.k_proj.weight
    - model.layers.{i}.attn.v_proj.weight -> layers.{i}.self_attn.v_proj.weight
    - model.layers.{i}.attn.o_proj.weight -> layers.{i}.self_attn.out_proj.weight
    - model.layers.{i}.mlp.gate_proj.weight -> layers.{i}.ff1.weight
    - model.layers.{i}.mlp.up_proj.weight -> layers.{i}.ff2.weight
    - model.layers.{i}.mlp.down_proj.weight -> layers.{i}.ff2.weight (combined)
    - model.layers.{i}.input_layernorm.weight -> layers.{i}.norm1.weight
    - model.layers.{i}.post_attention_layernorm.weight -> layers.{i}.norm2.weight
    - model.norm.weight -> norm.weight
    - lm_head.weight -> classifier.weight

    Args:
        state_dict: HF state dict
        config: LlamaConfig

    Returns:
        Remapped state dict
    """
    result = {}

    for key, value in state_dict.items():
        if key == "model.embed_tokens.weight":
            result["embedding.weight"] = value
        elif key == "model.norm.weight":
            result["norm.weight"] = value
        elif key == "lm_head.weight":
            result["classifier.weight"] = value
        elif key.startswith("model.layers."):
            # Parse layer index
            parts = key.split(".")
            layer_idx = parts[2]
            param_name = ".".join(parts[3:])

            if param_name == "attn.q_proj.weight":
                new_name = f"layers.{layer_idx}.self_attn.q_proj.weight"
            elif param_name == "attn.k_proj.weight":
                new_name = f"layers.{layer_idx}.self_attn.k_proj.weight"
            elif param_name == "attn.v_proj.weight":
                new_name = f"layers.{layer_idx}.self_attn.v_proj.weight"
            elif param_name == "attn.o_proj.weight":
                new_name = f"layers.{layer_idx}.self_attn.out_proj.weight"
            elif param_name == "mlp.gate_proj.weight":
                new_name = f"layers.{layer_idx}.ff1.weight"
            elif param_name == "mlp.up_proj.weight":
                new_name = f"layers.{layer_idx}.ff2.weight"
            elif param_name == "mlp.down_proj.weight":
                # Llama uses gate_proj + up_proj + down_proj
                # We'll combine gate+up into one, down as another
                continue  # Skip for now
            elif param_name == "input_layernorm.weight":
                new_name = f"layers.{layer_idx}.norm1.weight"
            elif param_name == "post_attention_layernorm.weight":
                new_name = f"layers.{layer_idx}.norm2.weight"
            else:
                new_name = f"layers.{layer_idx}.{param_name}"

            result[new_name] = value

    return result


def load_tokenizer(tokenizer_dir: str):
    """Load tokenizer using transformers library.

    This is a temporary solution - we'll replace with custom tokenizer later.

    Args:
        tokenizer_dir: Path to tokenizer directory

    Returns:
        Tokenizer object
    """
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_dir)
    except ImportError:
        raise ImportError(
            "transformers required for tokenizer. Install with: pip install transformers"
        )


class LlamaTokenizer:
    """Simple tokenizer wrapper using HF tokenizer.

    This is a temporary wrapper - we'll replace with custom implementation.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str, add_special_tokens: bool = True):
        """Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids, skip_special_tokens: bool = True):
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, token_ids_list, skip_special_tokens: bool = True):
        """Decode batch of token IDs.

        Args:
            token_ids_list: List of token ID lists
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded strings
        """
        return self.tokenizer.batch_decode(
            token_ids_list, skip_special_tokens=skip_special_tokens
        )
