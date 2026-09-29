import numpy as np
import torch
from sklearn import datasets
from torch.utils.data import Dataset

from certifiable_learning_stability.threats import Threats

from .dset_type import DsetType
from .perturbed_dset import PerturbedDataset


class Blobs(Dataset):
    DEFAULT_CENTER = 1
    DEFAULT_STD = 0.5

    def __init__(
        self,
        dset_type: DsetType,
        num_features: int = 2,
        num_samples_train_test: tuple[int] = (2500, 500),
        cluster_pos: float = None,
        cluster_std: float = None,
    ):
        assert dset_type in [DsetType.TRAIN_FULL, DsetType.TEST], "No need for validation set for this simple dataset"
        num_samples = num_samples_train_test[0] if dset_type == DsetType.TRAIN_FULL else num_samples_train_test[1]

        cstd = cluster_std if cluster_std is not None else Blobs.DEFAULT_STD
        cpos = cluster_pos if cluster_pos is not None else Blobs.DEFAULT_CENTER
        stds = [cstd, cstd]
        centers = np.array([[cpos, cpos], [-cpos, -cpos]])
        self.data, self.labels = datasets.make_blobs(n_features=num_features, centers=centers, cluster_std=stds, n_samples=num_samples)
        self.data, self.labels = torch.tensor(self.data, dtype=torch.float32), torch.tensor(self.labels, dtype=torch.float32)
        # self.data = Halfmoons.__normalize(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
