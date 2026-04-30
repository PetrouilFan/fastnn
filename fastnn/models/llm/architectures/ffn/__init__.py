"""FFN (Feed-Forward Network) modules."""

from fastnn.models.llm.architectures.ffn.swiglu import SwiGLU, SwiGLUConv, GELUMLP

__all__ = [
    "SwiGLU",
    "SwiGLUConv",
    "GELUMLP",
]