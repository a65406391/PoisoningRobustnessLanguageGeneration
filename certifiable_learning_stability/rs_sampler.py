import torch


class RS_Sampler:

    @staticmethod
    def l0_sampler(num_dataset_samples: int, bernoulli_prob: float) -> dict[int, int]:
        perturbed_indices = torch.rand(num_dataset_samples) <= bernoulli_prob

        return perturbed_indices