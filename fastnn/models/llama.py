"""LlamaModel inference for fastnn.

A complete Llama model implementation for running inference.
"""

import os
import numpy as np
import fastnn._core as _core
from fastnn import Linear, Embedding, RMSNorm, LayerNorm, silu, matmul
from fastnn._core import softmax


class LlamaConfig:
    """Configuration for Llama model."""

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=131072,
        hidden_act="silu",
        bos_token_id=128000,
        eos_token_id=128009,
        rope_scaling=None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # RoPE scaling
        self.rope_ratio = 1.0
        if rope_scaling:
            self.rope_ratio = rope_scaling.get("factor", 1.0)

    @classmethod
    def from_dict(cls, d):
        # Filter to only known config keys
        known_keys = {
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rope_theta",
            "rms_norm_eps",
            "max_position_embeddings",
            "hidden_act",
            "bos_token_id",
            "eos_token_id",
            "rope_scaling",
        }
        filtered = {k: v for k, v in d.items() if k in known_keys}
        return cls(**filtered)


class LlamaRMSNorm:
    """RMSNorm implementation."""

    def __init__(self, hidden_size, eps=1e-5):
        self.weight = _core.ones([hidden_size])
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x):
        x_sq = x * x
        mean_sq = _core.mean(x_sq, dim=2, keepdim=True)
        eps_tensor = _core.full_like(mean_sq, self.eps)
        rms = _core.sqrt(mean_sq + eps_tensor)
        return x / rms * self.weight

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [self.weight]

    def named_parameters(self):
        return [("weight", self.weight)]


class LlamaAttention:
    """Llama attention with GQA support."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Separate projections for Q, K, V, O
        self.q_proj = Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = Linear(hidden_size, hidden_size, bias=False)

        self.scale = 1.0 / (head_dim**0.5)

    def forward(self, x, position_ids=None, cos_cache=None, sin_cache=None):
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, num_heads, seq_len, head_dim] for Q
        # and [batch, num_kv_heads, seq_len, head_dim] for K, V
        q = q.reshape([batch, seq_len, self.num_heads, self.head_dim])
        q = q.permute([0, 2, 1, 3]).contiguous()

        k = k.reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
        k = k.permute([0, 2, 1, 3]).contiguous()

        v = v.reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
        v = v.permute([0, 2, 1, 3]).contiguous()

        # Apply RoPE if caches provided
        if cos_cache is not None and sin_cache is not None:
            q, k = self._apply_rope(q, k, cos_cache, sin_cache)

        # Repeat K, V for each Q head (GQA)
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = self._repeat(k, repeat_factor)
            v = self._repeat(v, repeat_factor)

        # Attention using numpy to avoid matmul issues
        q_np = q.numpy()
        k_np = k.numpy()
        v_np = v.numpy()

        # Compute attention scores: Q @ K^T
        attn_scores = np.einsum("bhqd,bhkd->bhqk", q_np, k_np) * self.scale

        # causal mask
        mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)
        attn_scores = attn_scores + mask

        # softmax using exp/sum
        attn_weights = np.exp(attn_scores) / np.sum(
            np.exp(attn_scores), axis=-1, keepdims=True
        )

        # Apply to V
        context_np = np.einsum("bhqk,bhkd->bhqd", attn_weights, v_np)

        # Convert back to fastnn
        context = _core.tensor_from_array(context_np)

        # Reshape back to [batch, seq_len, hidden_size]
        context = context.permute([0, 2, 1, 3]).contiguous()
        context = context.reshape([batch, seq_len, self.hidden_size])

        return self.o_proj(context)

    def __call__(self, x, position_ids=None, cos_cache=None, sin_cache=None):
        return self.forward(x, position_ids, cos_cache, sin_cache)

    def _apply_rope(self, q, k, cos, sin):
        """Apply rotary position embeddings using numpy.

        Args:
            q, k: [batch, heads, seq_len, head_dim]
            cos, sin: [seq_len, half_head_dim] - only computed for half head_dim

        RoPE only applies to the first half of head_dim - the second half (which would
        be the "negative frequencies") are not stored explicitly but computed implicitly
        by the rotation formula.
        """
        # Convert to numpy
        q_np = q.numpy()
        k_np = k.numpy()
        cos_np = cos.numpy()  # [seq_len, half_head_dim]
        sin_np = sin.numpy()  # [seq_len, half_head_dim]

        # Expand to full head_dim by interleaving: [seq_len, half_head_dim] -> [seq_len, head_dim]
        # Pattern: cos[0], cos[0], cos[1], cos[1], ...
        seq_len, half_head = cos_np.shape
        head_dim = half_head * 2

        cos_full = np.zeros((seq_len, head_dim), dtype=np.float32)
        sin_full = np.zeros((seq_len, head_dim), dtype=np.float32)
        cos_full[:, ::2] = cos_np
        cos_full[:, 1::2] = cos_np
        sin_full[:, ::2] = sin_np
        sin_full[:, 1::2] = sin_np

        # Now expand for broadcasting: [1, 1, seq_len, head_dim]
        cos_expanded = cos_full[np.newaxis, np.newaxis, :, :]
        sin_expanded = sin_full[np.newaxis, np.newaxis, :, :]

        # Apply RoPE to q
        q_np = self._apply_rope_single_np(q_np, cos_expanded, sin_expanded)

        # Apply RoPE to k
        k_np = self._apply_rope_single_np(k_np, cos_expanded, sin_expanded)

        # Convert back to fastnn
        q = _core.tensor_from_array(q_np)
        k = _core.tensor_from_array(k_np)

        return q, k

    def _apply_rope_single_np(self, x, cos, sin):
        """Apply RoPE to single numpy tensor.

        x: [batch, heads, seq_len, head_dim]
        cos, sin: [1, 1, seq_len, head_dim] (expanded)
        """
        # Split head dim in half
        head_dim = x.shape[-1]
        half_dim = head_dim // 2

        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

        # Rotate: [-x2, x1]
        x_rotated = np.concatenate([x2 * -1, x1], axis=-1)

        # Apply: x * cos + x_rotated * sin
        return x * cos + x_rotated * sin

    def _repeat(self, x, repeat_factor):
        """Repeat K/V heads for GQA using numpy."""
        # x: [batch, num_kv_heads, seq_len, head_dim]
        # -> [batch, num_kv_heads * repeat, seq_len, head_dim]
        x_np = x.numpy()
        x_repeated = np.repeat(x_np, repeat_factor, axis=1)
        return _core.tensor_from_data(
            x_repeated.flatten().tolist(), list(x_repeated.shape)
        )

    def _causal_mask(self, seq_len):
        """Create causal mask."""
        mask_data = [0.0] * (seq_len * seq_len)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                mask_data[i * seq_len + j] = float("-inf")
        mask = _core.tensor_from_data(mask_data, [seq_len, seq_len])
        mask = mask.reshape([1, 1, seq_len, seq_len])
        return mask

    def parameters(self):
        return (
            self.q_proj.parameters()
            + self.k_proj.parameters()
            + self.v_proj.parameters()
            + self.o_proj.parameters()
        )

    def named_parameters(self):
        params = []
        for name, p in self.q_proj.named_parameters():
            params.append((f"q_proj.{name}", p))
        for name, p in self.k_proj.named_parameters():
            params.append((f"k_proj.{name}", p))
        for name, p in self.v_proj.named_parameters():
            params.append((f"v_proj.{name}", p))
        for name, p in self.o_proj.named_parameters():
            params.append((f"o_proj.{name}", p))
        return params


class LlamaMLP:
    """Llama feed-forward network (SwiGLU)."""

    def __init__(self, hidden_size, intermediate_size):
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = silu(gate)
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return (
            self.gate_proj.parameters()
            + self.up_proj.parameters()
            + self.down_proj.parameters()
        )

    def named_parameters(self):
        params = []
        for name, p in self.gate_proj.named_parameters():
            params.append((f"gate_proj.{name}", p))
        for name, p in self.up_proj.named_parameters():
            params.append((f"up_proj.{name}", p))
        for name, p in self.down_proj.named_parameters():
            params.append((f"down_proj.{name}", p))
        return params


class LlamaBlock:
    """Single Llama transformer block."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = LlamaAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, x, position_ids=None, cos_cache=None, sin_cache=None):
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position_ids, cos_cache, sin_cache)
        x = x + residual

        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual

        return x

    def parameters(self):
        return (
            self.self_attn.parameters()
            + self.mlp.parameters()
            + self.input_layernorm.parameters()
            + self.post_attention_layernorm.parameters()
        )


class LlamaModel:
    """Complete Llama model for inference."""

    def __init__(self, config: LlamaConfig):
        self.config = config
        self.embedding = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LlamaBlock(config, i) for i in range(config.num_hidden_layers)]
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids=None):
        """Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Get seq_len
        if hasattr(input_ids, "shape"):
            seq_len = input_ids.shape[1]
        else:
            seq_len = len(input_ids[0]) if isinstance(input_ids, list) else 1

        # Compute RoPE caches
        cos, sin = self._compute_rope_cache(seq_len)

        x = self.embedding(input_ids)

        # Process layers
        for layer in self.layers:
            x = layer.forward(x, position_ids, cos_cache=cos, sin_cache=sin)

        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    def _compute_rope_cache(self, seq_len):
        """Compute RoPE cos/sin caches.

        Args:
            seq_len: sequence length

        Returns:
            cos, sin: [seq_len, head_dim]
        """
        # Build frequency positions
        head_dim = self.config.head_dim
        positions = np.arange(seq_len, dtype=np.float32)

        # Compute inv_freq: 1 / (rope_theta ^ (2i/d))
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        )

        # Compute angles: positions * inv_freq
        angles = positions[:, None] * inv_freq[None, :]

        # Compute cos/sin
        cos = np.cos(angles).astype(np.float32)
        sin = np.sin(angles).astype(np.float32)

        # Convert to fastnn tensors
        cos_tensor = _core.tensor_from_array(cos)
        sin_tensor = _core.tensor_from_array(sin)

        return cos_tensor, sin_tensor

    def parameters(self):
        params = self.embedding.parameters()
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.lm_head.parameters())
        return params

    def named_parameters(self):
        params = []
        for name, p in self.embedding.named_parameters():
            params.append((f"embedding.{name}", p))
        for i, layer in enumerate(self.layers):
            for name, p in layer.named_parameters():
                params.append((f"layers.{i}.{name}", p))
        for name, p in self.norm.named_parameters():
            params.append((f"norm.{name}", p))
        for name, p in self.lm_head.named_parameters():
            params.append((f"lm_head.{name}", p))
        return params

    @classmethod
    def load(cls, model_dir: str):
        """Load Llama model from HuggingFace directory.

        Args:
            model_dir: Path to model directory

        Returns:
            Loaded LlamaModel
        """
        # Load config
        import json

        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = LlamaConfig.from_dict(config_dict)

        # Create model
        model = cls(config)

        # Load weights
        from fastnn.io import load_safetensors

        safetensors_path = os.path.join(model_dir, "model.safetensors")

        if os.path.exists(safetensors_path):
            state_dict = load_safetensors(safetensors_path)
            model.load_weights(state_dict)

        model.eval()
        return model

    def load_weights(self, state_dict):
        """Load weights from state dict.

        Args:
            state_dict: Dictionary mapping names to tensors
        """
        # Build parameter map
        param_map = {}
        for name, param in self.named_parameters():
            param_map[name] = param

        # Map HF names to fastnn names
        hf_to_fastnn = self._build_name_mapping()

        # Load each weight
        for hf_name, fastnn_tensor in state_dict.items():
            if hf_name in hf_to_fastnn:
                target_name = hf_to_fastnn[hf_name]
                if target_name in param_map:
                    target = param_map[target_name]
                    # Copy data
                    np_data = fastnn_tensor.numpy()
                    fastnn_data = _core.tensor_from_data(
                        np_data.flatten().tolist(), list(np_data.shape)
                    )
                    target.copy_(fastnn_data)

    def _build_name_mapping(self):
        """Build mapping from HF names to fastnn names."""
        mapping = {}

        # Embedding
        mapping["model.embed_tokens.weight"] = "embedding.weight"

        # Layers
        for i in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{i}."

            mapping[f"{prefix}attn.q_proj.weight"] = (
                f"layers.{i}.self_attn.q_proj.weight"
            )
            mapping[f"{prefix}attn.k_proj.weight"] = (
                f"layers.{i}.self_attn.k_proj.weight"
            )
            mapping[f"{prefix}attn.v_proj.weight"] = (
                f"layers.{i}.self_attn.v_proj.weight"
            )
            mapping[f"{prefix}attn.o_proj.weight"] = (
                f"layers.{i}.self_attn.o_proj.weight"
            )

            mapping[f"{prefix}mlp.gate_proj.weight"] = (
                f"layers.{i}.mlp.gate_proj.weight"
            )
            mapping[f"{prefix}mlp.up_proj.weight"] = f"layers.{i}.mlp.up_proj.weight"
            mapping[f"{prefix}mlp.down_proj.weight"] = (
                f"layers.{i}.mlp.down_proj.weight"
            )

            mapping[f"{prefix}input_layernorm.weight"] = (
                f"layers.{i}.input_layernorm.weight"
            )
            mapping[f"{prefix}post_attention_layernorm.weight"] = (
                f"layers.{i}.post_attention_layernorm.weight"
            )

        # Output norm
        mapping["model.norm.weight"] = "norm.weight"

        # LM head
        mapping["lm_head.weight"] = "lm_head.weight"

        return mapping

    def eval(self):
        """Set to eval mode."""
        pass

    def generate(
        self, tokenizer, prompt: str, max_tokens: int = 100, temperature: float = 0.7
    ):
        """Generate text from prompt.

        Args:
            tokenizer: Tokenizer (HF AutoTokenizer)
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = np.array([input_ids], dtype=np.int32)

        generated = []
        for _ in range(max_tokens):
            # Forward pass
            input_tensor = _core.tensor_from_data(
                input_ids.flatten().tolist(), [1, len(input_ids)]
            )
            logits = self.forward(input_tensor)

            # Get last token logits
            last_logits = logits[0, -1, :]  # [vocab_size]

            # Apply temperature and sample
            if temperature > 0:
                last_logits = last_logits / temperature
                probs = last_logits.softmax(dim=-1)
                probs_np = probs.numpy().flatten()
                next_token = np.random.choice(len(probs_np), p=probs_np)
            else:
                next_token = int(np.argmax(last_logits.numpy()))

            generated.append(next_token)
            input_ids = np.append(input_ids, next_token)

            # Check for EOS
            if next_token == self.config.eos_token_id:
                break

        # Decode
        output = tokenizer.decode(generated, skip_special_tokens=True)
        return output


def load_llama_model(model_dir: str):
    """Load Llama model from directory.

    Args:
        model_dir: Path to model directory

    Returns:
        LlamaModel instance
    """
    return LlamaModel.load(model_dir)


def load_tokenizer(tokenizer_dir: str):
    """Load tokenizer.

    Args:
        tokenizer_dir: Path to tokenizer directory

    Returns:
        HF AutoTokenizer
    """
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_dir)
    except ImportError:
        raise ImportError("transformers required: pip install transformers")
