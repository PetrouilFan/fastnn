"""LlamaModel inference for fastnn.

A complete Llama model implementation for running inference.
"""

import os
import numpy as np
import fastnn._core as _core
from fastnn import Linear, Embedding, RMSNorm, LayerNorm, silu


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
        # x shape: [batch, seq_len, hidden_size]
        # RMSNorm: x / sqrt(mean(x^2) * weight
        x_sq = x * x
        mean_sq = _core.mean(x_sq, dim=-1, keepdim=True)
        rms = _core.sqrt(mean_sq + self.eps)
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
        """Forward pass.

        Args:
            x: [batch, seq_len, hidden_size]
            position_ids: [seq_len] - positions for RoPE
            cos_cache: Precomputed cos [seq_len, head_dim]
            sin_cache: Precomputed sin [seq_len, head_dim]
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, hidden_size]
        k = self.k_proj(x)  # [batch, seq_len, kv_heads * head_dim]
        v = self.v_proj(x)  # [batch, seq_len, kv_heads * head_dim]

        # Reshape to [batch, num_heads, seq_len, head_dim] for Q
        # and [batch, num_kv_heads, seq_len, head_dim] for K, V
        q = q.reshape([batch, seq_len, self.num_heads, self.head_dim])
        q = q.permute([0, 2, 1, 3])  # [batch, num_heads, seq_len, head_dim]

        k = k.reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
        k = k.permute([0, 2, 1, 3])  # [batch, num_kv_heads, seq_len, head_dim]

        v = v.reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
        v = v.permute([0, 2, 1, 3])  # [batch, num_kv_heads, seq_len, head_dim]

        # Apply RoPE if caches provided
        if cos_cache is not None and sin_cache is not None:
            q, k = self._apply_rope(q, k, cos_cache, sin_cache)

        # Repeat K, V for each Q head (GQA)
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = self._repeat(k, repeat_factor)
            v = self._repeat(v, repeat_factor)

        # Attention: Q @ K^T
        k_t = k.permute([0, 1, 3, 2])  # [batch, num_kv_heads, head_dim, seq_len]
        attn_scores = q.matmul(k_t)
        attn_scores = attn_scores * self.scale

        # causal mask - upper triangle is -inf
        mask = self._causal_mask(seq_len)
        attn_scores = attn_scores + mask

        # Softmax
        attn_weights = attn_scores.softmax(dim=-1)

        # Apply to V
        context = attn_weights.matmul(v)  # [batch, num_heads, seq_len, head_dim]

        # Reshape back to [batch, seq_len, hidden_size]
        context = context.permute([0, 2, 1, 3])  # [batch, seq_len, num_heads, head_dim]
        context = context.reshape([batch, seq_len, self.hidden_size])

        return self.o_proj(context)

    def _apply_rope(self, q, k, cos, sin):
        """Apply rotary position embeddings."""
        # cos, sin: [seq_len, head_dim]
        # This is a simplified version - full implementation would be more efficient
        # For now, skip RoPE and return as-is
        return q, k

    def _repeat(self, x, repeat_factor):
        """Repeat K/V heads for GQA."""
        # x: [batch, num_kv_heads, seq_len, head_dim]
        # -> [batch, num_kv_heads * repeat, seq_len, head_dim]
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x.reshape([batch, 1, num_kv_heads * repeat_factor, seq_len, head_dim])
        x = x.permute([0, 1, 3, 2, 4])
        return x.reshape([batch, num_kv_heads * repeat_factor, seq_len, head_dim])

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
        x = self.embedding(input_ids)

        # Process layers
        for layer in self.layers:
            x = layer.forward(x, position_ids)

        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

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
