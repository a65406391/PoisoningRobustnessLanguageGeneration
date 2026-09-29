import torch
import torchvision
from torch.utils.data import Dataset, random_split

from certifiable_learning_stability.threats import Threats

from .dset_type import DsetType
from .perturbed_dset import PerturbedDataset


class VanillaMNIST(Dataset):
    NORMALIZE_DATA = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    def __init__(self, dset_type: DsetType, split_seed: int, flatten: bool = False):
        curr_dir = __file__.rsplit("/", 1)[0]
        data_dir = curr_dir.rsplit("/", 1)[0] + "/data"

        dataset = None
        if dset_type in [DsetType.TRAIN, DsetType.VALID, DsetType.TRAIN_FULL]:
            dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=VanillaMNIST.NORMALIZE_DATA)
            if dset_type in [DsetType.VALID, DsetType.TRAIN]:
                # Do 85-15 split for train-valid
                train_size, valid_size = int(0.85 * len(dataset)), len(dataset) - int(0.85 * len(dataset))
                [train_subset, valid_subset] = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(split_seed))
                if dset_type == DsetType.VALID:
                    dataset.data = dataset.data[valid_subset.indices]
                    dataset.targets = dataset.targets[valid_subset.indices]
                else:
                    dataset.data = dataset.data[train_subset.indices]
                    dataset.targets = dataset.targets[train_subset.indices]
        else:
            dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=VanillaMNIST.NORMALIZE_DATA)
        self.data = dataset.data.to(dtype=torch.float32)
        self.targets = dataset.targets.to(dtype=torch.int64)
        if flatten:
            self.data = self.data.view(self.data.shape[0], -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# It's not ideal to make it a subclass of the VanillaMNIST class and call super().__init__ in the constructor
# because then it re-loads the dataset from disk every time we create a new instance of the PerturbedMNIST class.
# That is why a VanillaMNIST dataset is passed as an argument to the constructor of PerturbedMNIST.
class PerturbedMNIST(PerturbedDataset):
    def __init__(self, dataset: VanillaMNIST, attack_type: Threats, kwargs: dict):
        self.data, self.targets = dataset.data, dataset.targets
        self.attack_type = attack_type
        # % We want to sample the perturbed indices when initializing, and perturb the data only when getting an item.
        # % This is because we want lazy loading of the perturbed data and computational efficiency.
        self.perturbed_indices = self.get_perturbation_indices(attack_type, kwargs)
        self.num_perturbed_indices = self.perturbed_indices.sum().item()

    def __getitem__(self, idx):
        curr_data, curr_label = self.data[idx], self.targets[idx]
        if self.perturbed_indices[idx]:
            curr_data = self.perturb(self.attack_type, curr_data)

        return curr_data, curr_label
