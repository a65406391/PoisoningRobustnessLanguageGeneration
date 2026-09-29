from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

from certifiable_learning_stability.rs_sampler import RS_Sampler
from certifiable_learning_stability.threats import Threats


class PerturbedDataset(Dataset, ABC):
    def __len__(self):
        assert hasattr(self, "data"), "self.data attribute must be defined"
        return len(self.data)

    def get_perturbation_indices(self, attack_type: Threats, kwargs: dict) -> torch.Tensor:
        match attack_type:
            case Threats.L0:
                assert "bernoulli_prob" in kwargs, "The Bernoulli probability for sampling uniformly must be provided for (discrete) L0 perturbations"
                return RS_Sampler.l0_sampler(len(self), kwargs["bernoulli_prob"])
            # TODO: The other perturbations, if the idea works
            case _:
                raise NotImplementedError(f"Attack type {attack_type} not supported")

    def perturb(self, attack_type: Threats, sample: torch.Tensor, kwargs: dict = None) -> torch.Tensor:
        match attack_type:
            case Threats.L0:
                # % So the behaviour is: we have K bins, sample a bin for each feature. If it is the first bin,
                # % the perturbed feature is 0, if it is the last bin it is 1, otherwise it's the midpoint of the bin.
                assert "K" in kwargs, "The number of bins for sampling uniformly must be provided for (discrete) L0 perturbations"
                K = kwargs["K"]
                lb, ub = 0, 1
                if "lb" in kwargs and "ub" in kwargs:
                    # If for some reason the tensor values are outside the usual range
                    lb, ub = kwargs["lb"], kwargs["ub"]
                bins = torch.linspace(lb, ub, K + 1)
                bin_vals = (bins[:-1] + bins[1:]) / 2
                bin_vals[0] = lb
                bin_vals[-1] = ub
                indices_sampled = torch.randint(low=0, high=K, size=sample.shape)
                return bin_vals[indices_sampled]
            case _:
                raise NotImplementedError(f"Attack type {attack_type} not supported")

    @abstractmethod
    def __getitem__(self, idx):
        pass


class FeaturePerturbedDataset(Dataset, ABC):
    def __len__(self):
        assert hasattr(self, "data"), "self.data attribute must be defined"
        return len(self.data)

    @staticmethod
    def get_perturbation_indices(sample: torch.Tensor, attack_type: Threats, kwargs: dict) -> torch.Tensor:
        match attack_type:
            case Threats.L0:
                assert "bernoulli_prob" in kwargs, "The Bernoulli probability for sampling uniformly must be provided for (discrete) L0 perturbations"
                return torch.rand_like(sample, dtype=torch.float32) <= kwargs["bernoulli_prob"]
            case Threats.L2:
                assert "norm_bound" in kwargs, "The norm bound for sampling uniformly must be provided for (L2) perturbations"
                return torch.ones_like(sample, dtype=torch.float32)
            case _:
                raise NotImplementedError(f"Attack type {attack_type} not supported")

    @staticmethod
    def perturb(sample: torch.Tensor, feature_perturbed_idxs: torch.Tensor, attack_type: Threats, kwargs: dict = None) -> torch.Tensor:
        # Assert again just in case I am stupid and forget something
        match attack_type:
            case Threats.L0:
                sample[feature_perturbed_idxs] = torch.rand_like(sample[feature_perturbed_idxs])
                return sample
            case Threats.L2:
                assert "norm_bound" in kwargs, "The norm bound for sampling uniformly must be provided for (L2) perturbations"
                return sample + torch.randn_like(sample) / torch.norm(sample, p=2) * kwargs["norm_bound"]
            case _:
                raise NotImplementedError(f"Attack type {attack_type} not supported")

    @abstractmethod
    # Needed for further processing down the line
    def sample_perturbation(self, sample: torch.Tensor, attack_type: Threats, kwargs: dict) -> torch.Tensor:
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass
