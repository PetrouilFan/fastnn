import fastnn._core as _core
from fastnn.models.base import BaseModel


class Transformer(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        num_classes: int,
        dropout_p: float = 0.1,
    ):
        self._model = _core.TransformerEncoder(
            vocab_size,
            max_seq_len,
            d_model,
            num_heads,
            num_layers,
            ff_dim,
            num_classes,
            dropout_p,
        )

    def forward(self, x):
        return self._model(x)


