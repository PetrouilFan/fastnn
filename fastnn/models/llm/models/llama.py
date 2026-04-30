"""Llama model implementation for fastnn.

This module provides a complete Llama model implementation.
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import fastnn._core as _core
from fastnn import Linear, Embedding
from fastnn import tensor_from_array
from fastnn.models.llm.base import LLMConfig, LLMModel, ModelRegistry
from fastnn.models.llm.config import LlamaConfig
from fastnn.models.llm.architectures.attention.gqa import GQAAttention
from fastnn.models.llm.architectures.ffn.swiglu import SwiGLU
from fastnn.models.llm.architectures.norms.rms import RMSNorm
from fastnn.models.llm.utils.rope import create_rope_cache
from fastnn.io import load_safetensors


class LlamaBlock:
    """Single Llama transformer block.
    
    Contains:
    - Self-attention (GQA)
    - Feed-forward (SwiGLU)
    - Two RMSNorms (pre-attention, pre-FFN)
    """
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        
        # Attention
        self.self_attn = GQAAttention(config, layer_idx)
        self.input_layernorm = RMSNorm(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config, layer_idx)
        
        # FFN
        self.mlp = SwiGLU(config, layer_idx)
    
    def forward(self, x, rope_cache=None, **kwargs):
        """Forward pass through the block."""
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        
        cos_cache = rope_cache[0] if rope_cache else None
        sin_cache = rope_cache[1] if rope_cache else None
        
        x = self.self_attn(x, cos_cache=cos_cache, sin_cache=sin_cache)
        x = x + residual
        
        # FFN with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        
        return x
    
    def parameters(self):
        params = []
        params.extend(self.self_attn.parameters())
        params.extend(self.input_layernorm.parameters())
        params.extend(self.post_attention_layernorm.parameters())
        params.extend(self.mlp.parameters())
        return params
    
    def named_parameters(self):
        params = []
        prefix = f"layers.{self.layer_idx}."
        
        for name, p in self.self_attn.named_parameters():
            params.append((f"{prefix}self_attn.{name}", p))
        for name, p in self.input_layernorm.named_parameters():
            params.append((f"{prefix}input_layernorm.{name}", p))
        for name, p in self.post_attention_layernorm.named_parameters():
            params.append((f"{prefix}post_attention_layernorm.{name}", p))
        for name, p in self.mlp.named_parameters():
            params.append((f"{prefix}mlp.{name}", p))
        
        return params


class LlamaModel(LLMModel):
    """Complete Llama model for inference.
    
    Standard Llama architecture with:
    - Embedding
    - N transformer blocks (each with GQA + SwiGLU)
    - Final RMSNorm
    - LM head projection
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
        # Embedding
        self.embedding = Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.layers = [LlamaBlock(config, i) for i in range(config.num_hidden_layers)]
        
        # Final norm
        self.norm = RMSNorm(config, -1)
        
        # LM head
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # RoPE cache
        self._rope_cache = None
        
        self._built = True
    
    def forward(self, input_ids, **kwargs):
        """Forward pass."""
        # Embed
        x = self.embedding(input_ids)
        
        # Get RoPE cache
        seq_len = input_ids.shape[1] if hasattr(input_ids, 'shape') else len(input_ids[0])
        rope_cache = self._get_rope_cache(seq_len)
        
        # Process layers
        for layer in self.layers:
            x = layer.forward(x, rope_cache=rope_cache)
        
        # Final norm
        x = self.norm(x)
        
        # LM head
        logits = self.lm_head(x)
        
        return logits
    
    def _get_rope_cache(self, seq_len: int):
        """Get or create RoPE cache."""
        if self._rope_cache is None or seq_len > self._rope_cache._cached_seq_len:
            self._rope_cache = create_rope_cache(self.config, seq_len)
            self._rope_cache.compute(seq_len)
        return self._rope_cache
    
    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        """Load weights from state dict."""
        # Load embedding
        if "model.embed_tokens.weight" in state_dict:
            w = state_dict["model.embed_tokens.weight"]
            w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
            self.embedding.set_weight(tensor_from_array(w_np.astype(np.float32)))
        
        # Load output norm
        if "model.norm.weight" in state_dict:
            w = state_dict["model.norm.weight"]
            w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
            self.norm.weight.numpy()[:] = w_np.astype(np.float32)
        
        # Load LM head (or tie to embedding)
        if "lm_head.weight" in state_dict:
            w = state_dict["lm_head.weight"]
            w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
            self.lm_head.set_weight(tensor_from_array(w_np.astype(np.float32)))
        
        # Load each layer
        for idx, layer in enumerate(self.layers):
            prefix = f"model.layers.{idx}."
            
            # Attention weights
            weight_map = [
                ("self_attn.q_proj.weight", "q_proj"),
                ("self_attn.k_proj.weight", "k_proj"),
                ("self_attn.v_proj.weight", "v_proj"),
                ("self_attn.o_proj.weight", "o_proj"),
            ]
            
            for hf_name, fnn_name in weight_map:
                if f"{prefix}{hf_name}" in state_dict:
                    w = state_dict[f"{prefix}{hf_name}"]
                    w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                    target = getattr(layer.self_attn, fnn_name).parameters()[0]
                    target.tensor.copy_(tensor_from_array(w_np.astype(np.float32)))
            
            # MLP weights (need transpose for HF format)
            mlp_map = [
                ("mlp.gate_proj.weight", "gate_proj"),
                ("mlp.up_proj.weight", "up_proj"),
                ("mlp.down_proj.weight", "down_proj"),
            ]
            
            for hf_name, fnn_name in mlp_map:
                if f"{prefix}{hf_name}" in state_dict:
                    w = state_dict[f"{prefix}{hf_name}"]
                    w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                    w_np = w_np.T  # Transpose for Llama
                    target = getattr(layer.mlp, fnn_name).parameters()[0]
                    target.tensor.copy_(tensor_from_array(w_np.astype(np.float32)))
            
            # Layer norms
            if f"{prefix}input_layernorm.weight" in state_dict:
                w = state_dict[f"{prefix}input_layernorm.weight"]
                w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                layer.input_layernorm.weight.numpy()[:] = w_np.astype(np.float32)
            
            if f"{prefix}post_attention_layernorm.weight" in state_dict:
                w = state_dict[f"{prefix}post_attention_layernorm.weight"]
                w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                layer.post_attention_layernorm.weight.numpy()[:] = w_np.astype(np.float32)
    
    def parameters(self):
        params = []
        params.extend(self.embedding.parameters())
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.lm_head.parameters())
        return params
    
    def named_parameters(self):
        params = []
        
        for name, p in self.embedding.named_parameters():
            params.append((f"embedding.{name}", p))
        
        for idx, layer in enumerate(self.layers):
            params.extend(layer.named_parameters())
        
        for name, p in self.norm.named_parameters():
            params.append((f"norm.{name}", p))
        
        for name, p in self.lm_head.named_parameters():
            params.append((f"lm_head.{name}", p))
        
        return params
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> 'LlamaModel':
        """Load Llama model from pretrained files."""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = LlamaConfig.from_dict(config_dict)
        model = cls(config)
        
        weights = load_safetensors(model_path)
        model.load_weights(weights)
        
        model.eval()
        return model
    
    def generate(
        self,
        tokenizer,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        input_ids = tokenizer.encode(prompt)
        input_ids = np.array([input_ids], dtype=np.int32)
        
        generated = []
        
        for _ in range(max_tokens):
            input_tensor = _core.tensor_from_data(
                input_ids.flatten().tolist(),
                [1, len(input_ids)]
            )
            logits = self.forward(input_tensor)
            
            last_logits = logits.numpy()[0, -1, :]
            
            if temperature > 0:
                last_logits = last_logits / temperature
            
            if top_k is not None:
                top_indices = np.argpartition(last_logits, -top_k)[-top_k:]
                mask = np.full_like(last_logits, -np.inf)
                mask[top_indices] = last_logits[top_indices]
                last_logits = mask
            
            probs = np.exp(last_logits) / np.sum(np.exp(last_logits))
            next_token = int(np.random.choice(len(probs), p=probs))
            
            generated.append(next_token)
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
            
            if next_token == self.config.eos_token_id:
                break
        
        return tokenizer.decode(generated, skip_special_tokens=True)


# Register model
ModelRegistry.register_model("llama", LlamaModel)
ModelRegistry.register_config("llama", LlamaConfig)