"""Feed-Forward Network (FFN) implementations for LLM models.

Provides:
- SwiGLU: Swish-Gated Linear Unit (Llama, LFM)
- SwiGLUConv: Convolution-augmented SwiGLU (LFM LIV blocks)
- GELUMLP: Standard GELU MLP (BERT-style)
"""

from abc import ABC, abstractmethod
import numpy as np
import fastnn._core as _core
from fastnn import Linear, silu
from fastnn.models.llm.base import LLMConfig, LLMBlock


class BaseFFN(LLMBlock):
    """Base class for feed-forward networks."""
    
    @abstractmethod
    def forward(self, x):
        """FFN forward pass."""
        pass


class SwiGLU(LLMBlock):
    """SwiGLU (Swish-Gated Linear Unit) feed-forward network.
    
    Used by Llama, LFM, and other modern LLMs.
    Formula: ff = W_down(silu(W_gate(x)) * W_up(x))
    """
    
    def __init__(self, config: LLMConfig, layer_idx: int, layer_type: str = "ffn"):
        super().__init__(config, layer_idx)
        
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        
        # SwiGLU has three projections: gate, up, down
        self.gate_proj = Linear(hidden, intermediate, bias=False)  # W_gate
        self.up_proj = Linear(hidden, intermediate, bias=False)  # W_up
        self.down_proj = Linear(intermediate, hidden, bias=False)  # W_down
        
        self.hidden_size = hidden
        self.intermediate_size = intermediate
    
    def forward(self, x, **kwargs):
        """SwiGLU forward pass.
        
        Args:
            x: [batch, seq_len, hidden_size]
            
        Returns:
            Output: [batch, seq_len, hidden_size]
        """
        gate = self.gate_proj(x)
        gate = silu(gate)
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)
    
    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)
    
    def parameters(self):
        return (
            self.gate_proj.parameters() +
            self.up_proj.parameters() +
            self.down_proj.parameters()
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


class SwiGLUConv(LLMBlock):
    """Convolution-augmented SwiGLU (LIV Conv block).
    
    This is the LFM-specific block with w1/w2/w3 weights.
    It's similar to SwiGLU but tied differently and used in LFM's
    "conv" (LIV) layers.
    
    Formula: output = w2(silu(w1(x)) * w3(x))
    """
    
    def __init__(self, config: LLMConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        
        # LIV Conv uses w1, w2, w3 (instead of gate, up, down)
        # w1 and w3 take hidden -> intermediate
        # w2 takes intermediate -> hidden
        self.w1 = Linear(hidden, intermediate, bias=False)
        self.w3 = Linear(hidden, intermediate, bias=False)
        self.w2 = Linear(intermediate, hidden, bias=False)
        
        self.hidden_size = hidden
        self.intermediate_size = intermediate
    
    def forward(self, x, **kwargs):
        """LIV Conv forward pass.
        
        Args:
            x: [batch, seq_len, hidden_size]
            
        Returns:
            Output: [batch, seq_len, hidden_size]
        """
        # w1(x) * w3(x) with SiLU activation on w1
        w1_out = silu(self.w1(x))  # SiLU(w1(x))
        w3_out = self.w3(x)          # w3(x)
        
        # Element-wise multiply and project down
        hidden = w1_out * w3_out
        return self.w2(hidden)
    
    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)
    
    def parameters(self):
        return (
            self.w1.parameters() +
            self.w3.parameters() +
            self.w2.parameters()
        )
    
    def named_parameters(self):
        params = []
        for name, p in self.w1.named_parameters():
            params.append((f"w1.{name}", p))
        for name, p in self.w3.named_parameters():
            params.append((f"w3.{name}", p))
        for name, p in self.w2.named_parameters():
            params.append((f"w2.{name}", p))
        return params


class GELUMLP(LLMBlock):
    """Standard GELU MLP (BERT-style).
    
    Uses GELU activation instead of SwiGLU.
    """
    
    def __init__(self, config: LLMConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        
        self.up_proj = Linear(hidden, intermediate, bias=False)
        self.down_proj = Linear(intermediate, hidden, bias=False)
        
        self.hidden_size = hidden
        self.intermediate_size = intermediate
    
    def forward(self, x, **kwargs):
        """GELU MLP forward.
        
        Args:
            x: [batch, seq_len, hidden_size]
            
        Returns:
            Output: [batch, seq_len, hidden_size]
        """
        hidden = self.up_proj(x)
        # Use GELU approximation
        hidden = hidden * 0.5 * (1.0 + np.tanh(0.797885 * hidden * (1.0 + 0.044715 * hidden ** 3)))
        return self.down_proj(hidden)
    
    def parameters(self):
        return self.up_proj.parameters() + self.down_proj.parameters()
    
    def named_parameters(self):
        params = []
        for name, p in self.up_proj.named_parameters():
            params.append((f"up_proj.{name}", p))
        for name, p in self.down_proj.named_parameters():
            params.append((f"down_proj.{name}", p))
        return params


# Register FFN types
from fastnn.models.llm.base import BlockRegistry
BlockRegistry.register("ffn", SwiGLU)
BlockRegistry.register("swiglu", SwiGLU)
BlockRegistry.register("mlp", GELUMLP)
BlockRegistry.register("gelu_mlp", GELUMLP)