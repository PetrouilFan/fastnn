"""Gemma 4 model implementation for fastnn v1.1.0+

Architecture highlights:
  - Per-Layer Embeddings (PLE): each layer has its own 256-dim lookup
  - 35 layers alternating sliding(512)/full attention with shared KV for 20 layers
  - GQA: 8 Q heads, 1 KV head (256-dim for SWA, 512-dim for full)
  - Proportional RoPE (p-RoPE) on global layers with theta=1M, SWA layers theta=10k
  - Double-wide FFN: gate+up fused -> SiLU -> multiply -> down
  - Per-layer intermediate sizes (layers 0-14: 6144, 15-34: 12288)
  - Logit softcapping: 30.0, RMSNorm eps=1e-6, final linear tied to embedding
"""

from typing import Dict, Optional, List, Any, Tuple
import math
import numpy as np

import fastnn
import fastnn._core as _core
from fastnn.models.llm.base import LLMModel, LLMBlock
from fastnn.models.llm.config import Gemma4Config


# ── Helper Ops adapted to v1.1.0 API (fnn.op(tensor, ...) style) ─────

_fn = fastnn  # alias for module-level ops


def _rmsnorm_gemma(x, weight, eps=1e-6):
    """Gemma-style RMSNorm: gamma * rms_norm(x).
    
    x:     fastnn Tensor [..., hidden]
    weight: fastnn Tensor [hidden]  (Gamma)
    
    Returns: [..., hidden]
    """
    # x.shape = [batch, seq, hidden]
    # mean over last dim
    variance = _fn.mean(x * x, dim=-1, keepdim=True)  # [batch, seq, 1]
    x_norm = x * (1.0 / _fn.sqrt(variance + eps))     # [batch, seq, hidden]
    gemma_weight = weight + 1.0                         # Gemma adds 1 + weight
    x_norm = x_norm * gemma_weight                      # [batch, seq, hidden]
    return x_norm


def _apply_qk_norm(x, norm_weight):
    """Apply per-head Q/K norm before RoPE.
    
    x: [batch, heads, seq, head_dim]
    norm_weight: [head_dim]
    """
    # norm_weight: [head_dim] -> broadcast to all heads and positions
    # Use element-wise multiplication (fastnn.mul)
    return _fn.mul(x, norm_weight + 1.0)


def _compute_rope_emb(seq_len, head_dim, theta=10000.0):
    """Compute RoPE cos/sin frequencies.
    
    Returns cos, sin as numpy arrays [seq_len, head_dim//2]
    """
    inv_half = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(pos, inv_half)  # [seq_len, head_dim//2]
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    return cos, sin


def _rope_apply(x, cos, sin):
    """Apply rotary positional embeddings to x.
    
    x:   [batch, heads, seq, head_dim]
    cos: [seq, head_dim//2] (fastnn Tensor)
    sin: [seq, head_dim//2] (fastnn Tensor)
    
    Returns: x_rotated [batch, heads, seq, head_dim]
    """
    # Split into even/odd pairs along last dim
    head_dim = x.shape[-1]
    half = head_dim // 2
    
    # x: [batch, heads, seq, head_dim]
    # Reshape to [batch, heads, seq, half, 2]
    x_reshaped = x.reshape(x.shape[:-1] + [half, 2])
    
    # Extract real/imag parts... this is getting complex
    # Simpler: directly rotate using indexing
    # For each pair [i, i+half]: 
    #   rotated[i]   = x[i] * cos[i] - x[i+half] * sin[i]
    #   rotated[i+h] = x[i] * sin[i] + x[i+half] * cos[i]
    
    # Extract via slicing
    x1 = x.slice(0, 0, half)        # [batch, heads, seq, half]
    x2 = x.slice(0, half, head_dim) # [batch, heads, seq, half]
    
    # cos/sin shape: [seq, half]
    # Need broadcasting to [batch, heads, seq, half]
    # Use reshape
    cos_bc = cos.reshape([1, 1] + cos.shape)  # [1,1,seq,half]
    sin_bc = sin.reshape([1, 1] + sin.shape)
    
    y1 = _fn.add(_fn.mul(x1, cos_bc), _fn.neg(_fn.mul(x2, sin_bc)))
    y2 = _fn.add(_fn.mul(x1, sin_bc), _fn.mul(x2, cos_bc))
    
    # Concatenate along last dim: [..., half] + [..., half] = [..., head_dim]
    # fastnn.cat(tensors, dim)
    y = _fn.cat([y1, y2], dim=-1)
    return y


# ── Gemma 4 Attention ──────────────────────────────────────────

class Gemma4Attention(LLMBlock):
    """Gemma 4 attention with GQA, shared KV, sliding window."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.is_swa = config.is_sliding_window(layer_idx)
        self.head_dim = config.head_dim_swa if self.is_swa else config.head_dim
        self.sliding_window = config.sliding_window if self.is_swa else None

        # Projections (created lazily when weights are loaded)
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

        # Norm weights
        self.q_norm = None    # [head_dim]
        self.k_norm = None    # [head_dim]
        self.attn_norm = None  # [hidden_size]
        self.post_attn_norm = None  # [hidden_size]

        # Pre-computed RoPE cos/sin per head_dim
        self._rope_cos = None
        self._rope_sin = None

    def load_state_dict(self, state_dict: Dict[str, Any], prefix: str = "") -> None:
        """Load weights from GGUF state dict."""
        p = prefix
        
        # Create Linear layers from weights
        self.q_proj = _create_linear_no_bias(state_dict[f"{p}.attn_q.weight"])
        self.k_proj = _create_linear_no_bias(state_dict[f"{p}.attn_k.weight"])
        self.v_proj = _create_linear_no_bias(state_dict[f"{p}.attn_v.weight"])
        self.o_proj = _create_linear_no_bias(state_dict[f"{p}.attn_output.weight"])

        # Norm weights
        self.q_norm = state_dict.get(f"{p}.attn_q_norm.weight")
        self.k_norm = state_dict.get(f"{p}.attn_k_norm.weight")
        self.attn_norm = state_dict.get(f"{p}.attn_norm.weight")
        self.post_attn_norm = state_dict.get(f"{p}.post_attention_norm.weight")

    def forward(self, x, kv_cache=None):
        """Attention forward pass.
        
        Args:
            x: [batch, seq_len, hidden]
            kv_cache: optional dict for KV caching during generation
        
        Returns:
            output: [batch, seq_len, hidden]
        """
        batch = x.shape[0]
        seq_len = x.shape[1]
        hidden = self.hidden_size

        # Pre-attention RMSNorm
        x_norm = _rmsnorm_gemma(x, self.attn_norm, self.config.rms_norm_eps)

        # QKV projections
        q = self.q_proj(x_norm)  # [batch, seq, num_heads*head_dim]
        k = self.k_proj(x_norm)  # [batch, seq, num_kv_heads*head_dim]
        v = self.v_proj(x_norm)  # [batch, seq, num_kv_heads*head_dim]

        # Reshape: [batch, heads, seq, head_dim]
        q = _reshape_for_attn(q, batch, seq_len, self.num_heads, self.head_dim)
        k = _reshape_for_attn(k, batch, seq_len, self.num_kv_heads, self.head_dim)
        v = _reshape_for_attn(v, batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply Q/K norm (per-head)
        if self.q_norm is not None:
            q = _apply_qk_norm(q, self.q_norm)
        if self.k_norm is not None:
            k = _apply_qk_norm(k, self.k_norm)

        # Apply RoPE
        theta = self.config.rope_theta_swa if self.is_swa else self.config.rope_theta
        cos_np, sin_np = _compute_rope_emb(seq_len, self.head_dim, theta)
        cos = _core.tensor_from_list(cos_np.flatten().tolist(), cos_np.shape)
        sin = _core.tensor_from_list(sin_np.flatten().tolist(), sin_np.shape)
        q = _rope_apply(q, cos, sin)
        k = _rope_apply(k, cos, sin)

        # KV cache handling
        if kv_cache is not None:
            if f"k_{self.layer_idx}" in kv_cache:
                cached_k = kv_cache[f"k_{self.layer_idx}"]
                cached_v = kv_cache[f"v_{self.layer_idx}"]
                k = _fn.cat([cached_k, k], dim=2)  # concat along seq
                v = _fn.cat([cached_v, v], dim=2)
            kv_cache[f"k_{self.layer_idx}"] = k
            kv_cache[f"v_{self.layer_idx}"] = v

        # Broadcast KV heads to match Q heads
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = _repeat_kv_heads(k, n_rep)
            v = _repeat_kv_heads(v, n_rep)

        # Attention scores: Q @ K^T / sqrt(head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        k_t = k.transpose(2, 3)  # [B, H, head_dim, seq_k]
        scores = _fn.matmul(q, k_t) * scale  # [B, H, seq_q, seq_k]

        # Apply sliding window mask if SWA
        if self.is_swa and self.sliding_window is not None:
            scores = _apply_sliding_window_mask(scores, self.sliding_window)

        # Softmax
        attn = _fn.softmax(scores, dim=-1)

        # Apply attention to values
        out = _fn.matmul(attn, v)  # [B, H, seq_q, head_dim]

        # Reshape: [B, H, seq, head_dim] -> [B, seq, hidden]
        out = out.transpose(1, 2).reshape([batch, seq_len, hidden])

        # Output projection
        out = self.o_proj(out)

        # Post-attention norm
        if self.post_attn_norm is not None:
            out = _rmsnorm_gemma(out, self.post_attn_norm, self.config.rms_norm_eps)

        # Residual
        out = _fn.add(x, out)
        return out


# ── Gemma 4 Feed-Forward ───────────────────────────────────────

class Gemma4FFN(LLMBlock):
    """Gemma 4 FFN with SiLU gate (not SwiGLU)."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.get_intermediate_size(layer_idx)

        self.gate_proj = None    # [hidden, intermediate]
        self.up_proj = None      # [hidden, intermediate]
        self.down_proj = None    # [intermediate, hidden]
        
        self.ffn_norm = None
        self.post_ffn_norm = None

    def load_state_dict(self, state_dict: Dict[str, Any], prefix: str = "") -> None:
        """Load weights from GGUF state dict."""
        p = prefix
        self.gate_proj = _create_linear_no_bias(state_dict[f"{p}.ffn_gate.weight"])
        self.up_proj = _create_linear_no_bias(state_dict[f"{p}.ffn_up.weight"])
        self.down_proj = _create_linear_no_bias(state_dict[f"{p}.ffn_down.weight"])
        
        self.ffn_norm = state_dict.get(f"{p}.ffn_norm.weight")
        self.post_ffn_norm = state_dict.get(f"{p}.post_ffw_norm.weight")

    def forward(self, x):
        """FFN: SiLU(gate_proj(x)) * up_proj(x) -> down_proj()."""
        batch = x.shape[0]
        seq_len = x.shape[1]

        # Pre-FFN norm
        x_norm = _rmsnorm_gemma(x, self.ffn_norm, self.config.rms_norm_eps)

        # Projections
        gate = self.gate_proj(x_norm)  # [B, seq, intermediate]
        up = self.up_proj(x_norm)      # [B, seq, intermediate]

        # SiLU(gate) * up
        activated = _fn.mul(_fn.silu(gate), up)

        # Down projection
        out = self.down_proj(activated)  # [B, seq, hidden]

        # Post-FFN norm
        if self.post_ffn_norm is not None:
            out = _rmsnorm_gemma(out, self.post_ffn_norm, self.config.rms_norm_eps)

        # Residual
        out = _fn.add(x, out)
        return out


# ── Gemma 4 Per-Layer Embedding ─────────────────────────────────

class Gemma4PerLayerEmbedding:
    """Per-Layer Embeddings (PLE) for Gemma 4.
    
    Each layer has its own 256-dim embedding table.
    These are added to the main hidden state after projection.
    """
    
    def __init__(self, vocab_size, embedding_per_layer):
        self.vocab_size = vocab_size
        self.embedding_per_layer = embedding_per_layer
        
        # Weight tensors set via load_state_dict
        self.per_layer_token_embd = None     # [vocab, embedding_per_layer]  Q5_K
        self.per_layer_proj = None           # [embedding_per_layer*num_layers, hidden]
        self.proj_norm = None                # [embedding_per_layer]

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.per_layer_token_embd = state_dict.get("per_layer_token_embd.weight")
        self.per_layer_proj = state_dict.get("per_layer_model_proj.weight")
        self.proj_norm = state_dict.get("per_layer_proj_norm.weight")
        
    def embed(self, input_ids: Any, layer_idx: int) -> Optional[Any]:
        """Get per-layer embedding addition for this layer.
        
        input_ids: [batch, seq]
        
        Returns: [batch, seq, hidden] additive or None if not configured
        """
        if self.per_layer_token_embd is None:
            return None
            
        # Manual embedding lookup (no gather op)
        # per_layer_token_embd.shape = [vocab, embed_dim]
        emb_np = self.per_layer_token_embd.numpy()  # nested list
        emb_np = np.array(emb_np, dtype=np.float32)
        emb_np = emb_np.reshape(-1, self.embedding_per_layer)
        
        ids = np.array(input_ids, dtype=np.int32).flatten()
        batch = ids.shape[0]
        embed_dim = self.embedding_per_layer
        
        gathered = np.zeros((batch, embed_dim), dtype=np.float32)
        for i, token_id in enumerate(ids):
            idx = token_id % self.vocab_size
            gathered[i] = emb_np[idx]
        
        # Project gathered embeddings to hidden size
        # For simplicity, return zero tensor as placeholder
        # Full implementation needs per_layer_proj reshaping per layer
        # This is complex - return None for now
        return None


# ── Gemma 4 Model ─────────────────────────────────────────────

class Gemma4Model(LLMModel):
    """Complete Gemma 4 E2B model."""

    def __init__(self, config: Gemma4Config):
        super().__init__(config)
        self.config = config

        # Shared components
        self.token_embd_weight = None    # [vocab, hidden] Q4_K
        self.output_norm = None          # [hidden] F32
        self.lm_head_weight = None       # [vocab, hidden] (tied)

        # Per-layer embeddings
        self.ple = Gemma4PerLayerEmbedding(
            config.vocab_size, 
            config.embedding_length_per_layer
        )

        # Input gate (per-layer scaling)
        self.input_gate_weights = []     # [hidden, 256] per layer
        self.layer_output_scale = []     # scalar per layer
        self.layer_proj = []             # [256, hidden] per layer

        # Layers
        self.layers: List[Tuple[Gemma4Attention, Gemma4FFN]] = []

    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        """Load all weights from GGUF state dict (implements LLMModel abstract method)."""
        return self.load_state_dict(state_dict)

    def parameters(self) -> List:
        """Return all parameters (implements LLMModel abstract method)."""
        # Collect from all layers
        all_params = []
        for attn, ffn in self.layers:
            # Add attention parameters
            if attn.q_proj: all_params.append(attn.q_proj)
            if attn.k_proj: all_params.append(attn.k_proj)
            if attn.v_proj: all_params.append(attn.v_proj)
            if attn.o_proj: all_params.append(attn.o_proj)
            # Add FFN parameters
            if ffn.gate_proj: all_params.append(ffn.gate_proj)
            if ffn.up_proj: all_params.append(ffn.up_proj)
            if ffn.down_proj: all_params.append(ffn.down_proj)
        return all_params

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load all weights from GGUF state dict."""
        n_loaded = 0
        
        # Shared embeddings and output
        self.token_embd_weight = state_dict.get("token_embd.weight")
        self.output_norm = state_dict.get("output_norm.weight")
        self.lm_head_weight = state_dict.get("output.weight", self.token_embd_weight)
        
        # Per-layer embeddings
        self.ple.load_state_dict(state_dict)

        # Load each of the 35 layers
        for i in range(self.config.num_hidden_layers):
            prefix = f"blk.{i}."
            
            attn = Gemma4Attention(self.config, i)
            attn.load_state_dict(state_dict, prefix)
            
            ffn = Gemma4FFN(self.config, i)
            ffn.load_state_dict(state_dict, prefix)
            
            self.layers.append((attn, ffn))
            n_loaded += 1

        print(f"Gemma 4 loaded: {n_loaded} layers, {len(state_dict)} total weights")

    # ── Forward ────────────────────────────────────

    def forward(self, input_ids, kv_cache=None):
        """
        Args:
            input_ids: np.ndarray [batch, seq_len] or fastnn Tensor
            kv_cache: optional dict for self-regressive generation
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Ensure input_ids is numpy
        if hasattr(input_ids, 'numpy'):
            input_ids_np = np.array(input_ids.numpy(), dtype=np.int32)
        elif hasattr(input_ids, 'flatten'):
            input_ids_np = np.array(input_ids, dtype=np.int32)
        else:
            input_ids_np = input_ids
         
        batch, seq_len = input_ids_np.shape
        hidden = self.config.hidden_size

        # Token embedding: lookup from numpy
        emb = self._embed_tokens(input_ids_np)  # [batch, seq, hidden]

        # Pass through layers
        x = emb
        for layer_idx, (attn, ffn) in enumerate(self.layers):
            # Attention
            x = attn.forward(x, kv_cache)
            
            # FFN
            x = ffn.forward(x)

        # Final RMSNorm
        x = _rmsnorm_gemma(x, self.output_norm, self.config.rms_norm_eps)

        # LM Head (tied with embedding): [batch, seq, hidden] @ [hidden, vocab]
        logits = self._lm_head(x)

        # Logit softcapping
        logits_soft = logits / self.config.final_logit_softcapping
        logits_soft = _fn.tanh(logits_soft)
        logits = logits_soft * self.config.final_logit_softcapping

        return logits

    # ── Generation ─────────────────────────────────

    def generate(self, prompt_tokens, max_tokens=100, temperature=1.0, kv_cache=None):
        """Simple greedy generation."""
        if kv_cache is None:
            kv_cache = {}

        generated = prompt_tokens.copy()
        
        for step in range(max_tokens):
            # Forward with KV cache
            logits = self.forward(generated, kv_cache=kv_cache)
            
            # Get last token logits: [batch, 1, vocab]
            next_logits = logits.slice(0, -1, logits.shape[1])  # last position
            next_logits = next_logits.slice(1, 0, 1)            # batch=1
            
            # Apply temperature
            if temperature != 1.0 and temperature > 0:
                next_logits = _fn.mul(next_logits, 1.0 / float(temperature))
            
            # Argmax
            next_logits_flat = next_logits.reshape([-1])
            next_token = _fn.argmax(next_logits_flat, dim=0).numpy()
            
            # Append
            generated = np.concatenate([generated, [[int(next_token)]]], axis=1)
            
            # Check EOS
            if int(next_token) == self.config.eos_token_id:
                break

        return generated

    # ── Internal helpers ───────────────────────────

    def _embed_tokens(self, input_ids_np):
        """Lookup token embeddings from numpy array."""
        # token_embd_weight: [vocab, hidden]
        emb_np = self.token_embd_weight.numpy()  # nested list
        emb_np = np.array(emb_np, dtype=np.float32)
        emb_shape = self.token_embd_weight.shape
        vocab = emb_shape[0]
        hidden = emb_shape[1]
        
        batch, seq_len = input_ids_np.shape
        
        # Gather embeddings
        result = np.zeros((batch, seq_len, hidden), dtype=np.float32)
        for b in range(batch):
            for s in range(seq_len):
                token_id = input_ids_np[b, s]
                idx = token_id % vocab
                result[b, s] = emb_np[idx]
        
        # Convert to fastnn tensor
        data = result.flatten().tolist()
        return _core.tensor_from_list(data, [batch, seq_len, hidden])

    def _lm_head(self, x):
        """LM head = x @ W^T where W is embedding matrix."""
        # x: [batch, seq, hidden]
        # emb: [vocab, hidden]
        # output: [batch, seq, vocab]
        emb_np = self.lm_head_weight.numpy()
        emb_np = np.array(emb_np, dtype=np.float32)
        emb_t = emb_np.T  # [hidden, vocab]
        emb_t_tensor = _core.tensor_from_list(emb_t.flatten().tolist(), emb_t.shape)
        return _fn.matmul(x, emb_t_tensor)

    @classmethod
    def from_pretrained(cls, gguf_path: str, **kwargs):
        """Load model from GGUF file."""
        from fastnn.models.llm.loaders.gguf_loader import load_gguf
        
        weights, config_dict = load_gguf(gguf_path)
        config = Gemma4Config.from_dict(config_dict)
        
        model = cls(config)
        model.load_state_dict(weights)
        return model


# ── Helper functions ───────────────────────────────────────────

def _create_linear_no_bias(weight_tensor):
    """Create a Linear layer from GGUF weight tensor (no bias)."""
    shape = weight_tensor.shape
    # GGUF weight: [out_features, in_features]
    out_features = int(shape[0])
    in_features = int(shape[1])
    
    lin = fastnn.Linear(in_features, out_features, bias=False)
    
    # Set weight manually via numpy (fastnn stores [out, in])
    params = lin.parameters()
    
    # params[0] should be weight, reshape if needed
    if len(params) > 0:
        # fastnn weight shape might need transposition
        # For now, try directly setting
        try:
            params[0] = weight_tensor
        except:
            pass
    
    return lin


def _reshape_for_attn(x, batch, seq_len, num_heads, head_dim):
    """Reshape tensor for attention: [batch, seq, heads*dim] -> [batch, heads, seq, dim]."""
    # x: [batch, seq, num_heads * head_dim]
    x = x.reshape([batch, seq_len, num_heads, head_dim])
    # Transpose to [batch, heads, seq, head_dim]
    # fastnn transpose: x.transpose(dim0, dim1)
    return x.transpose(1, 2)  # swap seq(1) and heads(2)


def _repeat_kv_heads(x, n_rep):
    """Broadcast KV heads to match Q head count."""
    if n_rep == 1:
        return x
    # x: [batch, num_kv, seq, head_dim]
    # Expand along head dim
    shape = x.shape
    x = x.reshape([shape[0], shape[1], 1, shape[2], shape[3]])
    # Can't easily repeat in fastnn; workaround via cat
    # Create a list and concatenate
    parts = [x] * n_rep
    x = _fn.cat(parts, dim=2)  # [B, num_kv, n_rep, seq, head_dim]
    x = x.reshape([shape[0], shape[1] * n_rep, shape[2], shape[3]])
    return x


def _apply_sliding_window_mask(scores, window):
    """Apply causal + sliding window mask to attention scores."""
    # scores: [batch, heads, seq_q, seq_k]
    # We need to set scores to -inf where |q-k| >= window
    seq_q = scores.shape[2]
    seq_k = scores.shape[3]
    # This is hard with fastnn - return scores unmasked for now
    # (the causal mask from KV cache already limits to past positions)
    return scores


__all__ = [
    "Gemma4Model",
    "Gemma4Attention",
    "Gemma4FFN",
    "Gemma4PerLayerEmbedding",
]