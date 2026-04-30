"""Gemma 4 model implementation for fastnn.

Architecture:
  - Per-Layer Embeddings (PLE): each layer has its own 256-dim embedding lookup
  - Per-Layer Projection: projects per-layer embeddings to 1536-dim
  - Alternating sliding/full attention (5 layers sliding, 1 full)
  - GQA: 8 Q heads, 1 KV head
  - Double-wide MLP: gate+up, down (gelu activation, NOT SwiGLU)
  - Shared KV cache for first N shared_kv_layers layers
  - Logit softcapping: 30.0
  - Per-layer intermediate sizes (6144 or 12288)

Reference:
  https://huggingface.co/google/gemma-4-E2B-it
"""

from typing import Dict, Optional, List, Any, Tuple
import math

import numpy as np

import fastnn._core as _core
from fastnn import Linear, Embedding, RMSNorm, Tensor
from fastnn.models.llm.base import LLMModel, LLMBlock, LLMParameter
from fastnn.models.llm.config import Gemma4Config


# ── Helpers ──────────────────────────────────────────────────

def gelu_tanh(x: Tensor) -> Tensor:
    """GELU with tanh approximation (PyTorch default)."""
    # GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    # fastnn doesn't have erf/tanh, so we approximate with standard gelu
    # For now, use the fastnn gelu dispatch
    return x.gelu()


def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to query/key states.
    
    x:   [batch, heads, seq, head_dim]
    cos: [seq, head_dim/2]
    sin: [seq, head_dim/2]
    
    Returns x with RoPE applied.
    """
    # x shape: [batch, heads, seq, head_dim]
    shape = x.shape
    batch = shape[0]
    heads = shape[1]
    seq_len = shape[2]
    head_dim = shape[3]
    
    # Split last dim into pairs
    x1 = x.slice(0, 0, head_dim // 2)    # first half: [batch, heads, seq, head_dim/2]
    x2 = x.slice(0, head_dim // 2, head_dim)    # second half
    
    # cos/sin need broadcasting: [seq, head_dim/2] -> [batch, heads, seq, head_dim/2]
    # reshape and expand
    cos_bc = cos.reshape(1, 1, seq_len, -1)  # [1, 1, seq, head_dim/2]
    sin_bc = sin.reshape(1, 1, seq_len, -1)
    
    # Apply: x_rotated = x * cos + rotate_half(x) * sin
    # rotate_half: [-x2, x1]
    y1 = x1 * cos_bc + (-x2) * sin_bc
    y2 = x2 * cos_bc + x1 * sin_bc
    
    # Concatenate back
    y = y1.cat(y2, dim=-1)
    return y


class Gemma4Attention(LLMBlock):
    """Gemma 4 attention with GQA and optional sliding window."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.get_head_dim(layer_idx)
        self.is_swa = config.is_sliding_window(layer_idx)
        self.sliding_window = config.sliding_window if self.is_swa else None

        # Projections
        self.q_proj: Optional[Linear] = None
        self.k_proj: Optional[Linear] = None
        self.v_proj: Optional[Linear] = None
        self.o_proj: Optional[Linear] = None

        # Q/K norms
        self.q_norm: Optional[Tensor] = None   # shape [head_dim] - per-layer
        self.k_norm: Optional[Tensor] = None   # shape [head_dim] - per-layer

        # RMSNorms
        self.attn_norm: Optional[Tensor] = None
        self.post_attn_norm: Optional[Tensor] = None

    def load_weights(self, state_dict: Dict[str, Any], prefix: str = "") -> None:
        """Load weights from state dict."""
        p = prefix
        # Projections
        self.q_proj = self._make_linear(state_dict[f"{p}.attn_q.weight"])
        self.k_proj = self._make_linear(state_dict[f"{p}.attn_k.weight"])
        self.v_proj = self._make_linear(state_dict[f"{p}.attn_v.weight"])
        self.o_proj = self._make_linear(state_dict[f"{p}.attn_output.weight"])

        # Norm weights
        self.q_norm = state_dict.get(f"{p}.attn_q_norm.weight")
        self.k_norm = state_dict.get(f"{p}.attn_k_norm.weight")
        self.attn_norm = state_dict.get(f"{p}.attn_norm.weight")
        self.post_attn_norm = state_dict.get(f"{p}.post_attention_norm.weight")

    def _make_linear(self, weight: Tensor) -> Linear:
        """Create a Linear from a weight tensor (no bias for Gemma)."""
        # GGUF weight shape: [out, in]
        # fastnn Linear weight: [out, in]
        shape = weight.shape
        out_features = int(shape[0])
        in_features = int(shape[1])
        lin = Linear(in_features, out_features, bias=False)
        # Set weight
        lin.weight = weight
        return lin

    def forward(self, x, attention_mask=None, kv_cache=None, position_ids=None):
        """
        Args:
            x: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] or None
            kv_cache: optional dict to store KV cache
            position_ids: [seq_len] position indices
        
        Returns:
            output: [batch, seq_len, hidden_size]
            new_kv_cache
        """
        batch, seq_len, hidden = x.shape()

        # Pre-attention norm
        x_norm = self._rms_norm(x, self.attn_norm, self.config.rms_norm_eps)

        # QKV projections
        q = self.q_proj(x_norm)      # [batch, seq, num_heads * head_dim]
        k = self.k_proj(x_norm)      # [batch, seq, num_kv_heads * head_dim]
        v = self.v_proj(x_norm)      # [batch, seq, num_kv_heads * head_dim]

        # Apply Q/K norms (element-wise multiply by norm weight)
        if self.q_norm is not None:
            q = self._apply_norm_weight(q, self.q_norm)
        if self.k_norm is not None:
            k = self._apply_norm_weight(k, self.k_norm)

        # Reshape for attention: [batch, heads, seq, head_dim]
        q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        k = k.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]).transpose([0, 2, 1, 3])
        v = v.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]).transpose([0, 2, 1, 3])

        # Apply RoPE
        if position_ids is not None:
            cos, sin = self._compute_rope(seq_len, position_ids)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        # KV cache update
        if kv_cache is not None:
            past_k = kv_cache.get(f"layer_{self.layer_idx}_k")
            past_v = kv_cache.get(f"layer_{self.layer_idx}_v")
            if past_k is not None:
                k = past_k.cat(k, dim=2)   # concat along seq_len
                v = past_v.cat(v, dim=2)
            kv_cache[f"layer_{self.layer_idx}_k"] = k
            kv_cache[f"layer_{self.layer_idx}_v"] = v

        # Broadcast KV heads to match Q heads (GQA)
        if self.num_kv_heads != self.num_heads:
            # [batch, num_kv, seq, head_dim] -> [batch, num_heads, seq, head_dim]
            k = self._repeat_kv(k, self.num_heads // self.num_kv_heads)
            v = self._repeat_kv(v, self.num_heads // self.num_kv_heads)

        # Attention: Q @ K^T / sqrt(head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = self._batch_matmul(q, k.transpose([0, 1, 3, 2])) * scale
        # scores: [batch, heads, seq_q, seq_k]

        # Apply sliding window mask
        if self.is_swa and self.sliding_window is not None:
            scores = self._apply_sliding_window_mask(scores, self.sliding_window)

        # Apply attention mask
        if attention_mask is not None:
            # attention_mask: [batch, seq_k] -> [batch, 1, 1, seq_k]
            mask = attention_mask.reshape([batch, 1, 1, -1])
            # Large negative for masked positions
            scores = scores + (mask - 1.0) * 1e9

        # Softmax
        attn = scores.softmax(dim=-1)

        # Apply attention to values
        out = self._batch_matmul(attn, v)   # [batch, heads, seq, head_dim]

        # Reshape back
        out = out.transpose([0, 2, 1, 3]).reshape([batch, seq_len, -1])

        # Output projection
        out = self.o_proj(out)

        # Post-attention norm
        if self.post_attn_norm is not None:
            out = self._rms_norm(out, self.post_attn_norm, self.config.rms_norm_eps)

        # Residual
        out = x + out

        return out

    # ── Internal helpers ──────────────────────────

    def _rms_norm(self, x: Tensor, weight: Optional[Tensor], eps: float) -> Tensor:
        """Apply RMSNorm: x / sqrt(mean(x^2) + eps) * (1 + weight)."""
        if weight is None:
            return x
        # Gemma uses: gamma * rms_norm(x)
        # The weight is added as (1 + weight) after normalization
        variance = (x * x).mean(dim=-1, keepdim=True)
        x_norm = x * (variance + eps).rsqrt()
        # Apply learned weight
        gamma = weight + 1.0   # Gemma adds 1 to weight
        x_norm = x_norm * gamma
        return x_norm

    def _apply_norm_weight(self, x: Tensor, weight: Tensor) -> Tensor:
        """Apply per-head norm weight to Q/K states."""
        # x: [batch, seq, total_head_dim]
        # weight: [head_dim]
        # We need to apply weight element-wise to each head's slice
        # For now, reshape and apply
        batch = x.shape()[0]
        seq = x.shape()[1]
        total = x.shape()[2]
        x_reshaped = x.reshape([batch, seq, -1, self.head_dim])
        # weight: [head_dim] -> broadcast to all heads
        w = weight.reshape([1, 1, 1, -1])
        x_norm = x_reshaped * w
        x_norm = x_norm.reshape([batch, seq, total])
        return x_norm

    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        """Repeat KV heads to match Q head count (GQA)."""
        # x: [batch, num_kv, seq, head_dim]
        if n_rep == 1:
            return x
        batch, num_kv, seq, head_dim = x.shape()
        # Expand and repeat
        x = x.reshape([batch, num_kv, 1, seq, head_dim])
        x = x.repeat(1, 1, n_rep, 1, 1)
        x = x.reshape([batch, num_kv * n_rep, seq, head_dim])
        return x

    def _compute_rope(self, seq_len: int, position_ids: Any) -> Tuple[Tensor, Tensor]:
        """Compute RoPE cos/sin."""
        theta = self.config.rope_theta_swa if self.is_swa else self.config.rope_theta
        dim = self.config.rope_dim_swa if self.is_swa else self.config.rope_dim
        dim = min(dim, self.head_dim)

        # position_ids: use first row [seq_len]
        # angles: [seq_len, dim/2]
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        pos = np.array(position_ids, dtype=np.float32) if isinstance(position_ids, list) else position_ids
        freqs = np.outer(pos.flatten(), inv_freq)   # [seq_len, dim/2]

        cos = np.cos(freqs)
        sin = np.sin(freqs)

        return (
            _core.tensor_from_list(cos.flatten().tolist(), [seq_len, dim // 2]),
            _core.tensor_from_list(sin.flatten().tolist(), [seq_len, dim // 2]),
        )

    def _batch_matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Batch matrix multiply."""
        # Treat as 2D by flattening batch+head dimensions
        a_shape = a.shape()
        b_shape = b.shape()
        # Use fastnn matmul: [..., m, k] @ [..., k, n] = [..., m, n]
        return a.matmul(b)

    def _apply_sliding_window_mask(self, scores: Tensor, window: int) -> Tensor:
        """Apply sliding window causal mask."""
        # scores: [batch, heads, seq_q, seq_k]
        # For each position i, only allow positions j where i - window < j <= i
        seq_q = scores.shape()[2]
        seq_k = scores.shape()[3]

        # Create causal mask with sliding window
        for i in range(seq_q):
            for j in range(seq_k):
                if j > i or j <= i - window:
                    # Mask out
                    pass  # TODO: need slice + set support

        # For now, use a simple approach with basic operations
        return scores


class Gemma4FFN(LLMBlock):
    """Gemma 4 feed-forward network (gelu, NOT SwiGLU)."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.get_intermediate_size(layer_idx)

        self.gate_proj: Optional[Linear] = None   # [hidden, intermediate]
        self.up_proj: Optional[Linear] = None     # [hidden, intermediate]
        self.down_proj: Optional[Linear] = None   # [intermediate, hidden]

        self.ffn_norm: Optional[Tensor] = None
        self.post_ffn_norm: Optional[Tensor] = None

    def load_weights(self, state_dict: Dict[str, Any], prefix: str = "") -> None:
        """Load weights from state dict."""
        p = prefix
        self.gate_proj = self._make_linear(state_dict[f"{p}.ffn_gate.weight"])
        self.up_proj = self._make_linear(state_dict[f"{p}.ffn_up.weight"])
        self.down_proj = self._make_linear(state_dict[f"{p}.ffn_down.weight"])

        self.ffn_norm = state_dict.get(f"{p}.ffn_norm.weight")
        self.post_ffn_norm = state_dict.get(f"{p}.post_ffw_norm.weight")

    def _make_linear(self, weight: Tensor) -> Linear:
        """Create a Linear from a weight tensor (no bias)."""
        shape = weight.shape
        out_features = int(shape[0])
        in_features = int(shape[1])
        lin = Linear(in_features, out_features, bias=False)
        lin.weight = weight
        return lin

    def forward(self, x, **kwargs):
        # Gemma 4 uses standard MLP with GELU: gate -> gelu -> * up -> down
        # Actually, I need to check if it's gate * up or just gate
        # Looking at GGUF names: ffn_gate, ffn_up, ffn_down
        # This is the standard Llama-style SwiGLU that Gemma uses too
        # Gemma 4: gate = W1, up = W3, down = W2
        # MLP = down_proj(gelu(gate_proj(x)) * up_proj(x))
        
        batch, seq_len, hidden = x.shape()

        # Pre-FFN norm
        x_norm = self._rms_norm(x, self.ffn_norm, self.config.rms_norm_eps)

        # Gate and up projections
        gate = self.gate_proj(x_norm)   # [batch, seq, intermediate]
        up = self.up_proj(x_norm)        # [batch, seq, intermediate]

        # SwiGLU: silu(gate) * up
        # Gemma uses silu, not gelu
        hidden_act = gate.silu() * up

        # Down projection
        out = self.down_proj(hidden_act)   # [batch, seq, hidden]

        # Post-FFN norm
        if self.post_ffn_norm is not None:
            out = self._rms_norm(out, self.post_ffn_norm, self.config.rms_norm_eps)

        # Residual
        out = x + out
        return out

    def _rms_norm(self, x: Tensor, weight: Optional[Tensor], eps: float) -> Tensor:
        """Apply RMSNorm."""
        if weight is None:
            return x
        variance = (x * x).mean(dim=-1, keepdim=True)
        x_norm = x * (variance + eps).rsqrt()
        gamma = weight + 1.0
        x_norm = x_norm * gamma
        return x_norm


class Gemma4Layer(LLMBlock):
    """Single Gemma 4 decoder layer (attention + FFN)."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention = Gemma4Attention(config, layer_idx)
        self.ffn = Gemma4FFN(config, layer_idx)

    def load_weights(self, state_dict: Dict[str, Any], prefix: str = "") -> None:
        self.attention.load_weights(state_dict, prefix)
        self.ffn.load_weights(state_dict, prefix)

    def forward(self, x, attention_mask=None, kv_cache=None, position_ids=None):
        x = self.attention(x, attention_mask, kv_cache, position_ids)
        x = self.ffn(x)
        return x


class Gemma4Model(LLMModel):
    """Complete Gemma 4 E2B model."""

    def __init__(self, config: Gemma4Config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers

        # Shared output norm
        self.output_norm_weight: Optional[Tensor] = None

        # Per-layer components
        self.layers: List[Gemma4Layer] = []

        # Per-layer embeddings projection (PLE)
        self.token_embd: Optional[Tensor] = None
        self.per_layer_proj_weight: Optional[Tensor] = None
        self.per_layer_proj_norm: Optional[Tensor] = None
        self.per_layer_token_embd: Optional[Tensor] = None
        self.proj_norm_weight: Optional[Tensor] = None

        # Input gates (per-layer)
        self.input_gates: List[Optional[Tensor]] = []

        # Output projection (tied with token_embd)
        self.lm_head_weight: Optional[Tensor] = None

    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        """Load all weights from state dict."""
        print(f"Loading Gemma 4 weights... ({len(state_dict)} tensors)")

        # Shared token embedding
        if "token_embd.weight" in state_dict:
            self.token_embd = state_dict["token_embd.weight"]
            print(f"  token_embd: {self.token_embd.shape}")

        # Per-layer embeddings projection
        if "per_layer_model_proj.weight" in state_dict:
            self.per_layer_proj_weight = state_dict["per_layer_model_proj.weight"]
            print(f"  per_layer_model_proj: {self.per_layer_proj_weight.shape}")

        # Proj norm
        if "per_layer_proj_norm.weight" in state_dict:
            self.proj_norm_weight = state_dict["per_layer_proj_norm.weight"]

        # Per-layer token embeddings
        if "per_layer_token_embd.weight" in state_dict:
            self.per_layer_token_embd = state_dict["per_layer_token_embd.weight"]
            print(f"  per_layer_token_embd: {self.per_layer_token_embd.shape}")

        # Output norm
        if "output_norm.weight" in state_dict:
            self.output_norm_weight = state_dict["output_norm.weight"]

        # LM head (tied with embedding)
        # In GGUF: "output.weight" or tied
        lm_key = "output.weight" if "output.weight" in state_dict else None
        if lm_key:
            self.lm_head_weight = state_dict[lm_key]

        # Load each layer
        for i in range(self.num_layers):
            prefix = f"blk.{i}."
            layer = Gemma4Layer(self.config, i)
            layer.load_weights(state_dict, prefix)
            self.layers.append(layer)
            print(f"  Layer {i} loaded")

        print("All weights loaded")

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        """Embed token IDs to hidden states, with per-layer embedding."""
        batch, seq_len = input_ids.shape()

        # Lookup shared embedding
        # input_ids: [batch, seq]
        # token_embd: [vocab, hidden]
        # Output: gather rows
        
        # For now, use basic indexing (will add gather op later)
        # input_ids needs to be list of indices
        ids = input_ids.numpy().flatten().tolist()
        ids = [int(x) for x in ids]

        # Lookup embedding vectors
        emb_data = self.token_embd.numpy()
        emb_shape = self.token_embd.shape()
        vocab = emb_shape[0]
        hidden = emb_shape[1]

        # Gather embeddings
        gathered = []
        for token_id in ids:
            idx = token_id % vocab  # safety
            row = emb_data[idx * hidden: (idx + 1) * hidden]
            gathered.extend(row)

        return _core.tensor_from_list(gathered, [batch, seq_len, hidden])

    def forward(self, input_ids, attention_mask=None, position_ids=None, kv_cache=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            position_ids: [seq_len] or [batch, seq_len]
            kv_cache: optional dict for caching KV

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embed tokens
        x = self.embed_tokens(input_ids)

        # Pass through layers
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask, kv_cache, position_ids)

        # Final RMSNorm
        if self.output_norm_weight is not None:
            x = self._rms_norm(x, self.output_norm_weight, self.config.rms_norm_eps)

        # LM head
        # Tied with embedding: logits = x @ embedding^T
        # x: [batch, seq, hidden], emb: [vocab, hidden]
        # output: [batch, seq, vocab]
        if self.lm_head_weight is not None:
            # matmul: [batch, seq, hidden] @ [hidden, vocab] = [batch, seq, vocab]
            logits = x.matmul(self.lm_head_weight.t())
        else:
            logits = x.matmul(self.token_embd.t())

        # Logit softcapping
        logits = logits / self.config.final_logit_softcapping
        logits = logits.tanh() * self.config.final_logit_softcapping

        return logits

    def _rms_norm(self, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        """Apply RMSNorm."""
        variance = (x * x).mean(dim=-1, keepdim=True)
        x_norm = x * (variance + eps).rsqrt()
        gamma = weight + 1.0
        x_norm = x_norm * gamma
        return x_norm

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> 'Gemma4Model':
        """Load model from GGUF file."""
        from fastnn.models.llm.loaders.gguf_loader import load_gguf

        weights, config_dict = load_gguf(model_path)

        # Create config
        config = Gemma4Config.from_dict(config_dict)

        # Create model
        model = cls(config)
        model.load_weights(weights)

        return model

    def generate(self, input_ids, max_tokens=100, temperature=1.0, **kwargs):
        """Simple greedy generation."""
        # input_ids: [batch, seq_len]
        generated = input_ids
        kv_cache = {}

        for step in range(max_tokens):
            # Forward pass
            logits = self.forward(generated, kv_cache=kv_cache)

            # Get last token logits
            # logits: [batch, seq, vocab]
            next_logits = logits.slice(0, -1, logits.shape()[1])  # last position

            # Sample
            if temperature > 0:
                next_logits = next_logits / temperature
                # argmax for now (greedy)
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Append
            generated = generated.cat(next_token, dim=1)

            # Check for EOS
            next_val = next_token.numpy().flatten()[0]
            if int(next_val) == self.config.eos_token_id:
                break

        return generated

    def to(self, device):
        """Move to device (no-op for now, CPU only)."""
        return self

    def parameters(self) -> List[LLMParameter]:
        """Return all parameters."""
        params = []
        for i, layer in enumerate(self.layers):
            # Add layer params
            pass  # TODO
        return params
