import fastnn.core as core


class Transformer:
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
        self._model = core.PyTransformerEncoder(
            vocab_size,
            max_seq_len,
            d_model,
            num_heads,
            num_layers,
            ff_dim,
            num_classes,
            dropout_p,
        )

    def __call__(self, x):
        return self._model.forward(x)

    def forward(self, x):
        return self._model.forward(x)

    def parameters(self):
        return self._model.parameters()

    def zero_grad(self):
        self._model.zero_grad()

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()
