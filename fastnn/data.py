import math
import random


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        for t in tensors:
            if not hasattr(t, "shape"):
                raise ValueError("All tensors must have a shape attribute")

        if len(tensors) > 1:
            first_len = tensors[0].shape[0]
            for i, t in enumerate(tensors[1:], 1):
                if t.shape[0] != first_len:
                    raise ValueError(
                        f"TensorDataset tensors must have same first dimension: "
                        f"tensor 0 has {first_len}, tensor {i} has {t.shape[0]}"
                    )

        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.tensors)


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
    ):
        if num_workers > 0:
            raise NotImplementedError(
                f"num_workers={num_workers} > 0 is not yet supported. "
                "Use num_workers=0 for single-threaded data loading."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        # PERF-13: Pre-allocate indices to avoid re-allocation every epoch
        self.indices = list(range(len(dataset)))

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self):
        if len(self.indices) == 0:
            return
        if self.shuffle:
            random.shuffle(self.indices)
        for start in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            yield tuple(
                stack([self.dataset[i][j] for i in batch_idx])
                for j in range(len(self.dataset[0]))
            )


def stack(tensors, dim=0):
    import fastnn as fnn
    import numpy as np

    if not tensors:
        raise ValueError("stack requires at least one tensor")

    first = tensors[0]
    if hasattr(first, "numpy"):
        arrays = [t.numpy() for t in tensors]
        result_np = np.stack(arrays, axis=0)
        result = fnn.tensor(result_np.flatten().tolist(), list(result_np.shape))

        if dim != 0:
            result = result.transpose(dim, 0)

        return result
    else:
        arrays = [np.array(t) for t in tensors]
        result = np.stack(arrays, axis=dim)
        return fnn.tensor(result.flatten().tolist(), list(result.shape))
