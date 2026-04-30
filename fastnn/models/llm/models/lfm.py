"""LFM2.5 Model Implementation.

LFM (Large Foundation Model) architecture with:
- Layerwise Interleaved Variational (LIV) Conv blocks
- GQA Attention blocks
- RMSNorm
- RoPE
"""

import json
import numpy as np
from typing import Dict, Any
import fastnn._core as _core
from fastnn import Linear, Embedding, silu
from fastnn import tensor_from_array
from fastnn.models.llm.base import LLMConfig, LLMBlock, LLMModel
from fastnn.models.llm.architectures.attention.gqa import GQAAttention
from fastnn.models.llm.architectures.ffn.swiglu import SwiGLU
from fastnn.models.llm.architectures.norms.rms import RMSNorm
from fastnn.models.llm.utils.rope import create_rope_cache


class LFM2_5Block:
    """Single LFM2.5 transformer block.

    LFM2.5 has two block types:
    - "conv" (LIV Conv): Has Conv + FFN with residual connections
    - "attention": Has Attention + FFN with residual connections

    The block type is determined by config.layer_types[layer_idx].
    """

    def __init__(self, config: LLMConfig, layer_idx: int, block_type: str):
        self.config = config
        self.layer_idx = layer_idx
        self.block_type = block_type

        # All blocks have operator_norm and feed_forward
        self.operator_norm = RMSNorm(config, layer_idx)
        self.ffn_norm = RMSNorm(config, layer_idx)
        self.feed_forward = SwiGLU(config, layer_idx)

        if block_type in ("attention", "full_attention", "gqa"):
            # Attention block has attention + feed_forward
            self.attention = GQAAttention(config, layer_idx)
        elif block_type in ("conv", "liv_conv"):
            # LIV Conv block has conv + feed_forward
            self.conv = LIVConv(config, layer_idx)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def forward(self, x, rope_cache=None, **kwargs):
        """Forward pass through a single block.

        For attention blocks:
            residual = x
            x = x + attention(operator_norm(x))
            x = x + feed_forward(ffn_norm(x))

        For conv blocks:
            residual = x
            x = x + conv(operator_norm(x))
            x = x + feed_forward(ffn_norm(x))
        """
        if self.block_type in ("attention", "full_attention", "gqa"):
            # Attention block
            residual = x
            x = self.operator_norm(x)

            cos_cache = rope_cache[0] if rope_cache else None
            sin_cache = rope_cache[1] if rope_cache else None

            attn_out = self.attention(x, cos_cache=cos_cache, sin_cache=sin_cache)
            x = attn_out + residual

            # FFN
            residual = x
            x = self.ffn_norm(x)
            x = self.feed_forward(x) + residual

            return x

        elif self.block_type in ("conv", "liv_conv"):
            # Conv block
            residual = x
            x = self.operator_norm(x)
            x = self.conv(x) + residual

            # FFN
            residual = x
            x = self.ffn_norm(x)
            x = self.feed_forward(x) + residual

            return x

    def __call__(self, x, rope_cache=None, **kwargs):
        return self.forward(x, rope_cache, **kwargs)

    def parameters(self):
        params = []
        params.extend(self.operator_norm.parameters())
        params.extend(self.ffn_norm.parameters())
        params.extend(self.feed_forward.parameters())
        if hasattr(self, 'attention'):
            params.extend(self.attention.parameters())
        if hasattr(self, 'conv'):
            params.extend(self.conv.parameters())
        return params

    def named_parameters(self):
        params = []
        for name, p in self.operator_norm.named_parameters():
            params.append((f"operator_norm.{name}", p))
        for name, p in self.ffn_norm.named_parameters():
            params.append((f"ffn_norm.{name}", p))
        for name, p in self.feed_forward.named_parameters():
            params.append((f"feed_forward.{name}", p))
        if hasattr(self, 'attention'):
            for name, p in self.attention.named_parameters():
                params.append((f"attention.{name}", p))
        if hasattr(self, 'conv'):
            for name, p in self.conv.named_parameters():
                params.append((f"conv.{name}", p))
        return params


class LIVConv(LLMBlock):
    """LIV Convolution block.

    Implements the Lfm2ShortConv from HF transformers.
    Has Conv1d + in_proj + out_proj structure.

    Since fastnn Conv1d doesn't support groups parameter for depthwise conv,
    we implement a simplified version using linear projections.

    Structure (simplified):
        BCx = in_proj(x)  # [batch, seq, 3*hidden]
        B = BCx[:, :, 0:hidden]
        C = BCx[:, :, hidden:2*hidden]
        x_part = BCx[:, :, 2*hidden:3*hidden]
        # Gate: B * x_part
        gate = B * x_part
        # (Conv1d would go here - simplified to identity)
        # Output gate: C * gate
        out_gate = C * gate
        out = out_proj(out_gate)
    """

    def __init__(self, config: LLMConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        hidden = config.hidden_size
        self.hidden_size = hidden

        # in_proj: projects hidden -> 3*hidden
        self.in_proj = Linear(hidden, 3 * hidden, bias=False)
        # out_proj: projects hidden -> hidden
        self.out_proj = Linear(hidden, hidden, bias=False)

    def forward(self, x, **kwargs):
        """LIV Conv forward pass (simplified without Conv1d).

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            Output: [batch, seq_len, hidden_size]
        """
        batch, seq_len, hidden = x.shape

        # BCx = in_proj(x) -> [batch, seq, 3*hidden]
        BCx = self.in_proj(x)
        BCx_np = BCx.numpy()

        # Split into B, C, x_part
        # Each is [batch, seq, hidden]
        B_np = BCx_np[:, :, 0:hidden]
        C_np = BCx_np[:, :, hidden:2*hidden]
        x_part_np = BCx_np[:, :, 2*hidden:3*hidden]

        # Element-wise multiply: gate = B * x_part
        gate_np = B_np * x_part_np

        # TODO: Apply Conv1d here (depthwise conv with groups=hidden)
        # For now, just pass through (identity)
        conv_out_np = gate_np

        # Output gate: C * conv_out
        out_gate_np = C_np * conv_out_np

        # out = out_proj(out_gate)
        out_gate = tensor_from_array(out_gate_np.astype(np.float32))
        out = self.out_proj(out_gate)

        return out

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def parameters(self):
        return (
            self.in_proj.parameters() +
            self.out_proj.parameters()
        )

    def named_parameters(self):
        params = []
        for name, p in self.in_proj.named_parameters():
            params.append((f"in_proj.{name}", p))
        for name, p in self.out_proj.named_parameters():
            params.append((f"out_proj.{name}", p))
        return params


class LFM2_5Model:
    """Complete LFM2.5 Model."""

    def __init__(self, config: LLMConfig):
        self.config = config

        # Embedding
        self.embedding = Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.blocks = []
        for i in range(config.num_hidden_layers):
            block_type = config.layer_types[i] if hasattr(config, 'layer_types') and i < len(config.layer_types) else "attention"
            self.blocks.append(LFM2_5Block(config, i, block_type))

        # Final norm
        self.embedding_norm = RMSNorm(config, -1)

        # LM head
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

        # RoPE cache
        self._rope_cache = None

    def forward(self, input_ids):
        """Forward pass.

        Args:
            input_ids: fastnn.Tensor [batch, seq_len]

        Returns:
            logits: fastnn.Tensor [batch, seq_len, vocab_size]
        """
        batch, seq_len = input_ids.shape

        # Embedding
        x = self.embedding(input_ids)

        # Get or create RoPE cache
        rope_cache = self._get_rope_cache(seq_len)

        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x, rope_cache=rope_cache)

        x = self.embedding_norm(x)

        logits = self.lm_head(x)

        return logits

    def _get_rope_cache(self, seq_len: int):
        """Get or create RoPE cache."""
        if self._rope_cache is None or seq_len > self._rope_cache._cached_seq_len:
            self._rope_cache = create_rope_cache(self.config, seq_len)
        return self._rope_cache.compute(seq_len)

    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        """Load weights from HuggingFace state dict.

        Args:
            state_dict: Dictionary mapping weight names to tensors
        """
        # Build mapping from HF names to fastnn module references
        # We need to handle both transpose and non-transpose cases

        # Load embedding
        if "model.embed_tokens.weight" in state_dict:
            w = state_dict["model.embed_tokens.weight"]
            w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
            w_np = w_np.astype(np.float32)
            self.embedding.set_weight(tensor_from_array(w_np))

        # Load each block
        for i, block in enumerate(self.blocks):
            prefix = f"model.layers.{i}"

            if block.block_type in ("attention", "full_attention", "gqa"):
                # Attention block weights
                # Q, K, V projections
                q_key = f"{prefix}.self_attn.q_proj.weight"
                k_key = f"{prefix}.self_attn.k_proj.weight"
                v_key = f"{prefix}.self_attn.v_proj.weight"
                o_key = f"{prefix}.self_attn.o_proj.weight"

                if q_key in state_dict:
                    w = state_dict[q_key]
                    w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                    w_np = w_np.T.astype(np.float32)
                    block.attention.q_proj.set_weight(tensor_from_array(w_np))

                if k_key in state_dict:
                    w = state_dict[k_key]
                    w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                    w_np = w_np.T.astype(np.float32)
                    block.attention.k_proj.set_weight(tensor_from_array(w_np))

                if v_key in state_dict:
                    w = state_dict[v_key]
                    w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                    w_np = w_np.T.astype(np.float32)
                    block.attention.v_proj.set_weight(tensor_from_array(w_np))

                if o_key in state_dict:
                    w = state_dict[o_key]
                    w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                    w_np = w_np.T.astype(np.float32)
                    block.attention.o_proj.set_weight(tensor_from_array(w_np))

                # Q/K norms (if present)
                if hasattr(block.attention, 'q_norm') and hasattr(block.attention.q_norm, 'set_weight'):
                    q_norm_key = f"{prefix}.self_attn.q_layernorm.weight"
                    if q_norm_key in state_dict:
                        w = state_dict[q_norm_key]
                        w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                        w_np = w_np.astype(np.float32)
                        block.attention.q_norm.set_weight(tensor_from_array(w_np))

                if hasattr(block.attention, 'k_norm') and hasattr(block.attention.k_norm, 'set_weight'):
                    k_norm_key = f"{prefix}.self_attn.k_layernorm.weight"
                    if k_norm_key in state_dict:
                        w = state_dict[k_norm_key]
                        w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                        w_np = w_np.astype(np.float32)
                        block.attention.k_norm.set_weight(tensor_from_array(w_np))

            elif block.block_type in ("conv", "liv_conv"):
                # LIV Conv block weights
                # in_proj and out_proj
                in_proj_key = f"{prefix}.conv.in_proj.weight"
                out_proj_key = f"{prefix}.conv.out_proj.weight"

                if in_proj_key in state_dict:
                    w = state_dict[in_proj_key]
                    w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                    # HF shape: [3*hidden, hidden]
                    # FastNN expects: [hidden, 3*hidden]
                    w_np = w_np.T.astype(np.float32)
                    block.conv.in_proj.set_weight(tensor_from_array(w_np))

                if out_proj_key in state_dict:
                    w = state_dict[out_proj_key]
                    w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                    # HF shape: [hidden, hidden]
                    # FastNN expects: [hidden, hidden] (transposed)
                    w_np = w_np.T.astype(np.float32)
                    block.conv.out_proj.set_weight(tensor_from_array(w_np))

                # Conv1d weights (if present) - skip for now

            # Load norms (applies to all block types)
            # operator_norm (applied before conv/attention)
            op_norm_key = f"{prefix}.operator_norm.weight"
            if op_norm_key in state_dict:
                w = state_dict[op_norm_key]
                w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                block.operator_norm.weight = tensor_from_array(w_np.astype(np.float32))

            # ffn_norm (applied before feed_forward)
            ffn_norm_key = f"{prefix}.ffn_norm.weight"
            if ffn_norm_key in state_dict:
                w = state_dict[ffn_norm_key]
                w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                block.ffn_norm.weight = tensor_from_array(w_np.astype(np.float32))

            # Load feed_forward (SwiGLU) - applies to all block types
            # Map HF names to fastnn names
            # HF: feed_forward.w1, w2, w3
            # FastNN: feed_forward.gate_proj, up_proj, down_proj
            w1_key = f"{prefix}.feed_forward.w1.weight"
            w2_key = f"{prefix}.feed_forward.w2.weight"
            w3_key = f"{prefix}.feed_forward.w3.weight"

            if w1_key in state_dict:
                w = state_dict[w1_key]
                w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                block.feed_forward.gate_proj.set_weight(tensor_from_array(w_np.T.astype(np.float32)))

            if w2_key in state_dict:
                w = state_dict[w2_key]
                w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                block.feed_forward.down_proj.set_weight(tensor_from_array(w_np.T.astype(np.float32)))

            if w3_key in state_dict:
                w = state_dict[w3_key]
                w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
                block.feed_forward.up_proj.set_weight(tensor_from_array(w_np.T.astype(np.float32)))

        # Load final norm (outside the loop)
        if "model.embedding_norm.weight" in state_dict:
            w = state_dict["model.embedding_norm.weight"]
            w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
            self.embedding_norm.weight = tensor_from_array(w_np.astype(np.float32))

        # Load LM head (outside the loop)
        if "lm_head.weight" in state_dict:
            w = state_dict["lm_head.weight"]
            w_np = w.numpy() if hasattr(w, 'numpy') else np.array(w)
            self.lm_head.set_weight(tensor_from_array(w_np.T.astype(np.float32)))

    def __call__(self, input_ids):
        return self.forward(input_ids)

    def eval(self):
        """Set model to evaluation mode."""
        pass  # fastnn modules don't have eval mode

    def train(self):
        """Set model to training mode."""
        pass  # fastnn modules don't have train mode

    def parameters(self):
        params = []
        params.extend(self.embedding.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.embedding_norm.parameters())
        params.extend(self.lm_head.parameters())
        return params

    def named_parameters(self):
        params = []
        for name, p in self.embedding.named_parameters():
            params.append((f"embedding.{name}", p))
        for i, block in enumerate(self.blocks):
            for name, p in block.named_parameters():
                params.append((f"blocks.{i}.{name}", p))
        for name, p in self.embedding_norm.named_parameters():
            params.append((f"embedding_norm.{name}", p))
        for name, p in self.lm_head.named_parameters():
            params.append((f"lm_head.{name}", p))
        return params
