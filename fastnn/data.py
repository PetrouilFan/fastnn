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
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            yield tuple(
                stack([self.dataset[i][j] for i in batch_idx])
                for j in range(len(self.dataset[0]))
            )


def stack(tensors, dim=0):
    import fastnn as fnn

    if not tensors:
        raise ValueError("stack requires at least one tensor")

    first = tensors[0]
    if hasattr(first, "numpy"):
        first_data = first.numpy()
        numel = len(first_data)

        total_tensors = len(tensors)
        result_data = [0.0] * (numel * total_tensors)

        for i, t in enumerate(tensors):
            if hasattr(t, "numpy"):
                t_data = t.numpy()
                for j in range(numel):
                    result_data[i * numel + j] = t_data[j]

        result_shape = [total_tensors] + list(first.shape())
        result = fnn.tensor(result_data, result_shape)

        if dim != 0:
            result = result.transpose(dim, 0)

        return result
    else:
        import numpy as np

        arrays = []
        for t in tensors:
            arrays.append(np.array(t))
        result = np.stack(arrays, axis=dim)
        return fnn.tensor(result.flatten().tolist(), list(result.shape))
