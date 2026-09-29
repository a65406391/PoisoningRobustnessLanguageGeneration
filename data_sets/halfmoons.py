import torch
from sklearn import datasets
from torch.utils.data import Dataset

from certifiable_learning_stability.threats import Threats

from .dset_type import DsetType
from .perturbed_dset import PerturbedDataset


class Halfmoons(Dataset):
    @staticmethod
    def __normalize(data: torch.Tensor) -> torch.Tensor:
        mins, maxs = data.min(dim=0).values, data.max(dim=0).values
        delta = maxs - mins

        return (data - mins) / delta

    def __init__(self, dset_type: DsetType, num_samples_train_test: tuple[int] = (2500, 500), noise: float = 0.1):
        assert dset_type in [DsetType.TRAIN_FULL, DsetType.TEST], "No need for validation set for this simple dataset"
        num_samples = num_samples_train_test[0] if dset_type == DsetType.TRAIN_FULL else num_samples_train_test[1]
        self.data, self.labels = datasets.make_moons(n_samples=num_samples, noise=noise)
        self.data, self.labels = torch.tensor(self.data, dtype=torch.float32), torch.tensor(self.labels, dtype=torch.float32)
        self.data = Halfmoons.__normalize(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class PerturbedHalfmoons(PerturbedDataset):
    def __init__(self, dataset: Halfmoons, attack_type: Threats, kwargs: dict):
        self.attack_type = attack_type
        self.data, self.labels = dataset.data, dataset.labels
        self.perturbed_indices = self.get_perturbation_indices(attack_type, kwargs)
        self.num_perturbed_indices = self.perturbed_indices.sum().item()

    def __getitem__(self, index):
        curr_data, curr_label = self.data[index], self.labels[index]
        if self.perturbed_indices[index]:
            curr_data = self.perturb(self.attack_type, curr_data)

        return curr_data, curr_label


class DiscreteHalfmoons(PerturbedDataset):
    def __init__(self, dataset: Halfmoons, attack_type: Threats, kwargs: dict):
        self.attack_type = attack_type
        self.data, self.labels = dataset.data, dataset.labels
        self.perturbed_indices = self.get_perturbation_indices(attack_type, kwargs)
        self.num_perturbed_indices = self.perturbed_indices.sum().item()
        self.perturbation_params = kwargs

    def __getitem__(self, index):
        curr_data, curr_label = self.data[index], self.labels[index]
        if self.perturbed_indices[index]:
            curr_data = self.perturb(self.attack_type, curr_data, self.perturbation_params)

        return curr_data, curr_label
