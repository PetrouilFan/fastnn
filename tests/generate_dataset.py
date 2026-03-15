"""
Synthetic text classification dataset generator.
Task: Sentiment classification (positive=1, negative=0)
Method: Rule-based token sequences with clear patterns.
Vocab: 100 tokens (0=PAD, 1=CLS, 2-50=positive-leaning, 51-99=negative-leaning)
"""

import numpy as np

np.random.seed(42)

VOCAB_SIZE = 100
MAX_SEQ_LEN = 16
N_TRAIN = 2000
N_TEST = 400
N_CLASSES = 2

POS_TOKENS = list(range(2, 51))  # 49 tokens strongly associated with label=1
NEG_TOKENS = list(range(51, 100))  # 49 tokens strongly associated with label=0


def generate_sample(label, seq_len=MAX_SEQ_LEN):
    """Generate a token sequence with majority tokens from the label's pool."""
    seq = [1]  # CLS token at position 0
    if label == 1:
        # 70% from positive pool, 30% from negative pool (noise)
        n_signal = int((seq_len - 1) * 0.7)
        n_noise = (seq_len - 1) - n_signal
        tokens = list(np.random.choice(POS_TOKENS, n_signal)) + list(
            np.random.choice(NEG_TOKENS, n_noise)
        )
    else:
        n_signal = int((seq_len - 1) * 0.7)
        n_noise = (seq_len - 1) - n_signal
        tokens = list(np.random.choice(NEG_TOKENS, n_signal)) + list(
            np.random.choice(POS_TOKENS, n_noise)
        )
    np.random.shuffle(tokens)
    seq += tokens
    return seq[:seq_len]


X_train, y_train = [], []
for _ in range(N_TRAIN // 2):
    X_train.append(generate_sample(1))
    y_train.append(1)
    X_train.append(generate_sample(0))
    y_train.append(0)

X_test, y_test = [], []
for _ in range(N_TEST // 2):
    X_test.append(generate_sample(1))
    y_test.append(1)
    X_test.append(generate_sample(0))
    y_test.append(0)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int64)

np.save("tests/X_train.npy", X_train)
np.save("tests/y_train.npy", y_train)
np.save("tests/X_test.npy", X_test)
np.save("tests/y_test.npy", y_test)

print("Dataset generated:")
print(
    f"  Train: {X_train.shape}, labels: {y_train.shape}, "
    f"pos={y_train.sum()}, neg={(1 - y_train).sum()}"
)
print(f"  Test:  {X_test.shape},  labels: {y_test.shape}")
print(f"  Vocab size: {VOCAB_SIZE}, Seq len: {MAX_SEQ_LEN}")
