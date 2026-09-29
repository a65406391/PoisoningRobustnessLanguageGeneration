import gc
import math
import os
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy import stats
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import trange

from certifiable_learning_stability.agt_certifier import AGTCertifier
from certifiable_learning_stability.agt_certifier import (
    BlobsCertifier as AGT_BlobsCertifier,
)
from certifiable_learning_stability.agt_certifier import (
    Cifar10Certifier as AGT_Cifar10Certifier,
)
from certifiable_learning_stability.agt_certifier import (
    HalfmoonsCertifier as AGT_HalfmoonsCertifier,
)
from certifiable_learning_stability.agt_certifier import (
    MnistCertifier as AGT_MnistCertifier,
)
from certifiable_learning_stability.rdp_certifier import (
    Cifar10Certifier as RDP_Cifar10Certifier,
)
from certifiable_learning_stability.rdp_certifier import (
    HalfmoonsCertifier as RDP_HalfmoonsCertifier,
)
from certifiable_learning_stability.rdp_certifier import (
    MnistCertifier as RDP_MnistCertifier,
)
from certifiable_learning_stability.rdp_certifier import (
    StabilityCertifierWithRDP,
    validate_and_fix_model,
)
from data_sets.blobs import Blobs
from data_sets.cifar import CIFAR
from data_sets.dset_type import DsetType
from data_sets.halfmoons import Halfmoons
from data_sets.hash import tensor_generic_hash
from data_sets.hh_anthropic import get_hh_rlhf_preference_dataset
from data_sets.mnist import VanillaMNIST
from experiments.reproducibility import set_seed
from experiments.save_utils import (
    get_logfile_path,
    get_result_dir_path,
    get_state_dict_from_file,
    save_model_state_dict,
    torchsave,
    write_results_to_file,
)

from .certification_methods import AggregationType, CertificationMethod, RobustnessSetup
from .inference import (
    accuracy,
    aggregate_predictions_batch,
    aggregate_robustness_radii_to_dict,
    get_prediction,
)
from .models.fcn import FCN
from .models.generic_nn import Generic_NN
from .models.resnet import Resnet18, Resnet18Finetune
from .solver import certify_batch_dpa, certify_batch_dpa_roe


class StabilityCertifierWithDPA(ABC):
    """
    Abstract base class for stability certifiers that use Deep Partition Aggregation (DPA).
    See [Deep Partition Aggregation: Provable Defenses against General Poisoning Attacks](https://arxiv.org/pdf/2006.14768)
    """

    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None) -> None:
        check_hyperparams_present(hyperparams, save_kwargs)
        self.device = device
        # % Set general hyperparameters
        self.hparams_ensemble = hyperparams
        self.num_partitions = hyperparams["num_partitions"]
        self.seed = hyperparams["seed"]
        self.hparams_sgd = hyperparams["hp_sgd"]
        self.hparams_hybrid_rdp = hyperparams["hp_hrdp"]
        self.hparams_pointwise_rdp = hyperparams["hp_prdp"]
        self.hparams_agt = hyperparams["hp_agt"]
        self.method_name = hyperparams["method_name"]
        self.flatten = hyperparams.get("flatten", False)
        self.test_batch_size = hyperparams.get("test_batch_size", 1000)
        # % Set up dataset and model
        self.model = self.make_model().to(self.device)
        self.dp_model = validate_and_fix_model(deepcopy(self.model), self.device)
        self.initial_params = self.model.trainable_params.clone().detach()
        self.train_set = self.get_original_dset(DsetType.TRAIN_FULL)
        self.test_set = self.get_original_dset(DsetType.TEST)
        self.test_loader = self.get_dataloader(self.test_set, self.test_batch_size, shuffle=False)
        # % Set up saving results to file
        self.setup_logging_and_saving(save_kwargs)
        # % Make the certifiers now to avoid repeatedly creating them. The logfile for each one of them depends on calling the function above.
        self.hybrid_rdp_certifier = self.rdp_certifier(self.hparams_hybrid_rdp, {})
        self.ptwise_rdp_certifier = self.rdp_certifier(self.hparams_pointwise_rdp, {})
        self.agt_certifier = self.agt_certifier(self.hparams_agt, {"save": save_kwargs["save"]})
        # % Make sure the partition certifiers have the same model as the main one
        self.hybrid_rdp_certifier.model = deepcopy(self.model)
        self.ptwise_rdp_certifier.model = deepcopy(self.model)
        try:
            self.agt_certifier.model = deepcopy(self.model.to_sequential())
        except NotImplementedError:
            logger.warning("AGT certifier model cannot be converted to sequential, using the original model instead.")
            self.agt_certifier.model = deepcopy(self.model)

    def train_dpa_partitions(
        self,
        cert_methods: tuple = (CertificationMethod.SGD, CertificationMethod.HYBRID_RDP, CertificationMethod.POINTWISE_RDP, CertificationMethod.AGT),
        partitioning_method: str = "shard",
    ) -> None:
        assert partitioning_method in [
            "shard",
            "hash",
            "bag",
        ], f"Partitioning method {partitioning_method} not supported. Use `shard', 'bag' or'hash'."

        idx_groups = None
        if partitioning_method == "shard":
            idx_groups = self._partition_disjoint_naive()
        elif partitioning_method == "hash":
            idx_groups = self._partition_with_hashing()
        else:
            idx_groups = self._partition_bagging_with_replacement()

        cert_methods = list(cert_methods)  # Mutability
        hparams_dict = {
            CertificationMethod.SGD: self.hparams_sgd,
            CertificationMethod.HYBRID_RDP: self.hparams_hybrid_rdp,
            CertificationMethod.POINTWISE_RDP: self.hparams_pointwise_rdp,
            CertificationMethod.AGT: self.hparams_agt,
        }
        hparams = [hparams_dict[cert_method] for cert_method in cert_methods]
        for partition_idx in range(self.num_partitions):
            for cert_method, hp in zip(cert_methods, hparams):
                # Dispatches training for a partition and saves the corresponding models
                self._handle_partition(cert_method, hp, partition_idx, idx_groups[partition_idx])

        # Reset the model
        self.model.trainable_params = deepcopy(self.initial_params.clone().detach())

    @torch.no_grad()
    def multi_sample_certification(
        self,
        method_sample: list[CertificationMethod] | CertificationMethod,
        agg_type: AggregationType,
        ks_poison: np.ndarray,
        test_set: Dataset = None,
    ) -> tuple[float, float, float]:
        if not isinstance(method_sample, list):
            assert isinstance(method_sample, CertificationMethod), "A certification method must be specified as a list or a single method."
            method_sample = [method_sample] * self.num_partitions
        assert len(method_sample) == self.num_partitions, "A certification method must be specified for each partition."
        for cert_method in method_sample:
            assert isinstance(cert_method, CertificationMethod), f"Expected CertificationMethod, got {type(cert_method)}"

        test_loader = self.test_loader
        if test_set is not None:
            test_loader = self.get_dataloader(test_set, self.test_batch_size, shuffle=False)
        partition_dset_size, num_batches = len(self.train_set) // self.num_partitions, len(test_loader)
        print(f"Partition dataset size: {partition_dset_size}")
        all_correct_votes, all_correct_cert_rob, wc_accs_per_k_poison = (
            torch.tensor([]).to(device=self.device),
            torch.tensor([]).to(device=self.device),
            torch.zeros((num_batches, len(ks_poison)), device=self.device, dtype=torch.float32),
        )
        for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
            logger.info(f"Processing batch {batch_idx + 1}/{num_batches}  with {len(batch_data)} samples")
            batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
            # Cast vote and compute intrinsic robustness
            votes_per_class, prediction_per_partition, logits_per_partition_and_class = self._cast_ensemble_vote(method_sample, batch_data)
            intrinsic_rob = self._partitions_intrinsic_robustness(
                batch_data, batch_labels, method_sample, ignore_label=(agg_type == AggregationType.ROE)
            )
            logger.info(f"Intrinsic robustness for each partition: {intrinsic_rob}")
            # Prepare shapes for optimization
            scores, preds = logits_per_partition_and_class.transpose(0, 1), prediction_per_partition.transpose(0, 1)
            correct_rob_radius = None
            match agg_type:
                case AggregationType.DPA:
                    _, correct_rob_radius = self._dpa_sample_aggregation_margin(votes_per_class, batch_data, batch_labels, label_match=True)
                    for k_poison_idx, k_poison in enumerate(ks_poison):
                        wc_acc, _ = certify_batch_dpa(
                            intrinsic_rob, preds, batch_labels, self.num_classes, partition_dset_size, k_poison, self.device
                        )
                        wc_accs_per_k_poison[batch_idx, k_poison_idx] = wc_acc
                case AggregationType.ROE:
                    _, correct_rob_radius = self._roe_sample_aggregation_margin(
                        votes_per_class, prediction_per_partition, logits_per_partition_and_class, batch_data, batch_labels, label_match=True
                    )
                    for k_poison_idx, k_poison in enumerate(ks_poison):
                        wc_acc = certify_batch_dpa_roe(
                            intrinsic_rob, preds, scores, batch_labels, self.num_classes, partition_dset_size, k_poison, self.device
                        )
                        wc_accs_per_k_poison[batch_idx, k_poison_idx] = wc_acc
                case _:
                    raise ValueError(f"Unknown aggregation type: {agg_type}")
            correct_votes_mask = votes_per_class.argmax(dim=1) == batch_labels
            all_correct_votes = torch.cat((all_correct_votes, correct_votes_mask), dim=0)
            all_correct_cert_rob = torch.cat((all_correct_cert_rob, correct_rob_radius), dim=0)

            print(f"Num correct votes: {correct_votes_mask.sum().item()} out of {len(batch_data)}")

        avg_acc = all_correct_votes.float().mean().item()
        avg_correct_cert_rob = all_correct_cert_rob[all_correct_cert_rob >= 0].mean().item()
        avg_wc_acc = wc_accs_per_k_poison.mean(dim=0).cpu().numpy()
        avg_wc_acc_per_poison_budget = {int(k_poison): float(avg_wc_acc[k_poison_idx]) for k_poison_idx, k_poison in enumerate(ks_poison)}

        if self.result_file is not None:
            category = (
                self.method_name
                + "_agg_type_"
                + str(agg_type).lower()
                + f"_partitions_{self.num_partitions}"
                + f"_msc_batch_size_{self.test_batch_size}"
            )
            write_results_to_file(
                self.result_file,
                {
                    "accuracy": avg_acc,
                    "avg_correct_cert_rob": avg_correct_cert_rob,
                    "certified_acc_per_poison_budget": deepcopy(avg_wc_acc_per_poison_budget),
                },
                category,
            )
            self._save_hparams(category)

        return avg_acc, avg_correct_cert_rob, avg_wc_acc_per_poison_budget

    @torch.no_grad()
    def get_metrics_inference(
        self, method_sample: list[CertificationMethod] | CertificationMethod, agg_type: AggregationType, test_set: Dataset = None
    ) -> tuple[float, float, float, float]:
        if not isinstance(method_sample, list):
            assert isinstance(method_sample, CertificationMethod), "A certification method must be specified as a list or a single method."
            method_sample = [method_sample] * self.num_partitions
        assert len(method_sample) == self.num_partitions, "A certification method must be specified for each partition."
        for cert_method in method_sample:
            assert isinstance(cert_method, CertificationMethod), f"Expected CertificationMethod, got {type(cert_method)}"

        test_loader = self.test_loader
        if test_set is not None:
            test_loader = self.get_dataloader(test_set, self.test_batch_size, shuffle=False)
        num_datapoints, num_batches = len(test_loader.dataset), len(test_loader)
        all_correct_votes, all_cert_rob, all_correct_cert_rob, all_enhanced_rob_radius, all_enhanced_correct_rob_radius = (
            torch.tensor([]).to(device=self.device),
            torch.tensor([]).to(device=self.device),
            torch.tensor([]).to(device=self.device),
            torch.tensor([]).to(device=self.device),
            torch.tensor([]).to(device=self.device),
        )
        for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
            logger.info(f"Processing batch {batch_idx + 1}/{num_batches}  with {len(batch_data)} samples")
            batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
            rob_radius, correct_rob_radius = None, None
            votes_per_class, prediction_per_partition, logits_per_partition_and_class = self._cast_ensemble_vote(method_sample, batch_data)
            match agg_type:
                case AggregationType.DPA:
                    rob_radius, correct_rob_radius = self._dpa_sample_aggregation_margin(votes_per_class, batch_data, batch_labels, label_match=True)
                case AggregationType.ROE:
                    rob_radius, correct_rob_radius = self._roe_sample_aggregation_margin(
                        votes_per_class, prediction_per_partition, logits_per_partition_and_class, batch_data, batch_labels, label_match=True
                    )
                case _:
                    raise ValueError(f"Unknown aggregation type: {agg_type}")

            partitions_intrinsic_robustness = self._partitions_intrinsic_robustness(batch_data, batch_labels, method_sample)
            # Enhance robustness radius using intrinsic robustness
            same_voting_partitions = prediction_per_partition == votes_per_class.argmax(dim=1).unsqueeze(0)  # Should broadcast
            enhanced_rob_radius = self._aggregation_margin_enhance_intrinsic_robustness(
                rob_radius, partitions_intrinsic_robustness, same_voting_partitions
            )
            correctly_voting_partitions = prediction_per_partition == batch_labels.unsqueeze(0)  # Should broadcast
            enhanced_correct_rob_radius = self._aggregation_margin_enhance_intrinsic_robustness(
                correct_rob_radius, partitions_intrinsic_robustness, correctly_voting_partitions
            )
            correct_votes_mask = votes_per_class.argmax(dim=1) == batch_labels
            # Accumulate
            all_correct_votes = torch.cat((all_correct_votes, correct_votes_mask), dim=0)
            all_cert_rob = torch.cat((all_cert_rob, rob_radius), dim=0)
            all_correct_cert_rob = torch.cat((all_correct_cert_rob, correct_rob_radius), dim=0)
            all_enhanced_rob_radius = torch.cat((all_enhanced_rob_radius, enhanced_rob_radius), dim=0)
            all_enhanced_correct_rob_radius = torch.cat((all_enhanced_correct_rob_radius, enhanced_correct_rob_radius), dim=0)

            logger.info(f"Intrinsic robustness for each partition: {partitions_intrinsic_robustness}")
            print(f"Num correct votes: {correct_votes_mask.sum().item()} out of {len(batch_data)}")

        avg_acc, avg_cert_rob, avg_correct_cert_rob, avg_cert_acc, avg_enhanced_rob, avg_enhanced_correct_rob = (
            all_correct_votes.float().mean().item(),
            all_cert_rob.mean().item(),
            all_correct_cert_rob[all_correct_cert_rob >= 0].mean().item(),
            all_correct_cert_rob[all_correct_cert_rob >= 0].numel() / num_datapoints,
            all_enhanced_rob_radius.mean().item(),
            all_enhanced_correct_rob_radius[all_enhanced_correct_rob_radius >= 0].mean().item(),
        )

        percentage_robust_at_radius = aggregate_robustness_radii_to_dict(all_cert_rob)
        percentage_correct_at_radius = aggregate_robustness_radii_to_dict(all_correct_cert_rob)
        percentage_enhanced_robust_at_radius = aggregate_robustness_radii_to_dict(all_enhanced_rob_radius)
        percentage_enhanced_correct_robust_at_radius = aggregate_robustness_radii_to_dict(all_enhanced_correct_rob_radius)

        if self.result_file is not None:
            category = self.method_name + "_agg_type_" + str(agg_type).lower() + f"_partitions_{self.num_partitions}"
            write_results_to_file(
                self.result_file,
                {
                    # We need to copy the dictionaries because: https://support.atlassian.com/bitbucket-cloud/docs/yaml-anchors/
                    "percentage_robust_at_radii [potentially incorrect]": deepcopy(percentage_robust_at_radius),
                    "certified_acc_per_rob_radius": deepcopy(percentage_correct_at_radius),
                    "enhanced_robust_at_radii": deepcopy(percentage_enhanced_robust_at_radius),
                    "enhanced_certified_acc_per_rob_radius": deepcopy(percentage_enhanced_correct_robust_at_radius),
                    "avg_cert_rob": avg_cert_rob,
                    "avg_correct_cert_rob": avg_correct_cert_rob,
                    "avg_cert_acc": avg_cert_acc,
                    "accuracy": avg_acc,
                    "avg_enhanced_rob": avg_enhanced_rob,
                    "avg_enhanced_correct_rob": avg_enhanced_correct_rob,
                },
                category,
            )
            self._save_hparams(category)

        return avg_acc, avg_cert_acc, avg_cert_rob, avg_correct_cert_rob

    def agt_bagging_guarantee(self, k_private: int, clip_gamma: float = 0.1, test_set: Dataset = None) -> tuple[float, dict]:
        test_loader = self.test_loader
        if test_set is not None:
            test_loader = self.get_dataloader(test_set, self.test_batch_size, shuffle=False)
        num_datapoints, num_batches = len(test_loader.dataset), len(test_loader)

        all_votes, all_intrinsic_robustness, all_labels = (
            torch.tensor([]).to(self.device, torch.int64),
            torch.tensor([]).to(self.device, torch.int64),
            torch.tensor([]).to(self.device, torch.int64),
        )
        for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
            batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
            # Vote and get robustness for each partition by dispatching to AGT
            batch_ir, batch_votes = [], []
            for partition_idx in range(self.num_partitions):
                partition_save_path = os.path.join(self.save_load_dir, f"partition_{partition_idx}")
                ir, votes = self.agt_certifier.vote_and_get_robustness(batch_data, batch_labels, k_private, clip_gamma, partition_save_path)
                batch_ir.append(ir)
                batch_votes.append(votes)
                # print(f"Batch {batch_idx} partition {partition_idx} out of {self.num_partitions} done.")
            batch_ir = torch.stack(batch_ir, dim=0)  # Shape (num_partitions, batch_size)
            batch_votes = torch.stack(batch_votes, dim=0)  # Shape (num_partitions, batch_size)
            all_intrinsic_robustness = torch.cat((all_intrinsic_robustness, batch_ir), dim=1)
            all_votes = torch.cat((all_votes, batch_votes), dim=1)
            all_labels = torch.cat((all_labels, batch_labels), dim=0)

        assert all_intrinsic_robustness.shape[1] == num_datapoints
        assert all_votes.shape[1] == num_datapoints
        assert all_intrinsic_robustness.shape[0] == self.num_partitions
        assert all_votes.shape[0] == self.num_partitions

        acc, cert_correct_rob = 0, []
        for sample_idx in range(num_datapoints):
            # Get vote counts and intrinsic robustness for this sample
            vote_counts = all_votes[:, sample_idx].cpu().numpy().flatten()
            k_values = all_intrinsic_robustness[:, sample_idx].cpu().numpy().flatten()

            # Compute τ(x)
            vote_counts = np.bincount(vote_counts, minlength=self.num_classes)
            sorted_votes = np.sort(vote_counts)[::-1]
            predicted_class = np.argmax(vote_counts)
            votes_pred, votes_sec = sorted_votes[0], sorted_votes[1] if len(sorted_votes) > 1 else 0
            tau = math.ceil((votes_pred - votes_sec) / 2)

            # Binary search for certified radius
            left, right, certified_r, samples_per_bag = 0, 5000, 0, len(self.train_set) // self.num_partitions
            while left <= right:
                r = (left + right) // 2
                prob_poison = r / len(self.train_set)

                q_ir_vals = [1 - stats.binom.cdf(k_values[i], samples_per_bag, prob_poison) for i in range(self.num_partitions)]

                poi_bin_mean = sum(q_ir_vals)
                poi_bin_var = sum(q * (1 - q) for q in q_ir_vals)
                if poi_bin_var > 0:
                    poi_bin_std = math.sqrt(poi_bin_var)
                    # Use CLT to approximate the Poisson Binomial with a Standard Gaussian
                    z_score = (tau - 0.5 - poi_bin_mean) / poi_bin_std
                    prob = 1 - stats.norm.cdf(z_score)
                else:
                    prob = float(poi_bin_mean >= tau)

                if prob <= 0.05:  # 95% confidence
                    certified_r = r
                    left = r + 1
                else:
                    right = r - 1

            if all_labels[sample_idx] == predicted_class:
                acc += 1
                cert_correct_rob.append(certified_r)
            else:
                cert_correct_rob.append(-1)

            print(f"Sample {sample_idx}: certified radius = {certified_r}")

        percentage_correct_robust_at_radius = aggregate_robustness_radii_to_dict(torch.tensor(cert_correct_rob))
        acc = float(acc / num_datapoints)

        if self.result_file is not None:
            category = self.method_name + f"_partitions_{self.num_partitions}_k_private_{k_private}_clip_gamma_{clip_gamma}"
            write_results_to_file(
                self.result_file,
                {
                    "percentage_correct_robust_at_radius": deepcopy(percentage_correct_robust_at_radius),
                    "accuracy": acc,
                },
                category,
            )
            self._save_hparams(category)

        return acc, percentage_correct_robust_at_radius

    def _aggregation_margin_enhance_intrinsic_robustness(
        self, rob_radii: torch.Tensor, intrinsic_robustness: torch.Tensor, correctly_voting_partitions: torch.Tensor
    ) -> torch.Tensor:
        # If the sum of the intrinsic robustness over partitions dim is strictly less than one, than we have no enhanced robustness
        enhanced_indices = torch.nonzero((intrinsic_robustness >= 1).any(dim=0) * (rob_radii > 0), as_tuple=True)[0]
        max_agg_margin = int(rob_radii.max())
        # I.e. there are no enhanced datapoints
        if len(enhanced_indices) == 0 or max_agg_margin <= 0:
            return rob_radii

        print(f"max agg margin: {max_agg_margin}, enhanced indices: {enhanced_indices}")
        print(f"Intrinsic robustness: {intrinsic_robustness}")
        print(f"Robustness radii: {rob_radii}")

        enhanced_robustness = torch.zeros_like(rob_radii, device=self.device, dtype=torch.int64)
        for enhanced_idx in enhanced_indices:
            eidx = int(enhanced_idx)
            rob_radius = int(rob_radii[eidx])
            # Get the partitions that voted correctly for this sample
            correct_partitions_indices = correctly_voting_partitions[:, eidx].nonzero(as_tuple=True)[0]
            correct_partitions_robustness = intrinsic_robustness[correct_partitions_indices, eidx]
            # Get their intrinsic robustness and sort them in ascending order
            num_correct_partitions = correct_partitions_robustness.shape[0]
            if rob_radius > num_correct_partitions:
                rob_radius = num_correct_partitions
            bottom_r = torch.topk(correct_partitions_robustness, rob_radius, largest=False).values
            # Assume the attacker knows the least robust partitions and can poison them and that it needs in the worst case agg_margin partitions to
            # poison in order to change the prediction. Hence, take the sum of the bottom rob_radii and add the original aggregation margin
            br_sum = bottom_r[bottom_r > 0].sum().to(dtype=torch.int64)
            enhanced_robustness[eidx] = br_sum + rob_radii[eidx]  # Add the original aggregation margin

        return enhanced_robustness

    def _dpa_sample_aggregation_margin(
        self, votes_per_class: torch.Tensor, batch_data: torch.Tensor, batch_labels: torch.Tensor, label_match: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the aggregation margin using DPA for a given ensemble of parameters and a batch of data.

        :param torch.Tensor votes_per_class: A tensor containing the votes per class for each sample in the batch.
        :param torch.Tensor batch_data: The data batch to certify.

        Returns:
                torch.Tensor: A tensor containing the aggregation margin for each sample in the batch.
        """
        top_2 = torch.topk(votes_per_class, 2, dim=1)
        top_2_classes, top_2_votes = top_2.indices, top_2.values
        votes_pred, votes_sec = top_2_votes[:, 0], top_2_votes[:, 1]
        classes_pred, classes_sec = top_2_classes[:, 0], top_2_classes[:, 1]
        votes_pred, votes_sec, classes_pred, classes_sec = self._handle_ties(votes_pred, votes_sec, classes_pred, classes_sec)
        rob_radius = torch.zeros(batch_data.shape[0], device=self.device, dtype=torch.int64)
        rob_radius = ((votes_pred - votes_sec) / 2).floor().to(dtype=torch.int64)

        if label_match:
            correct_pred_rob_radius = rob_radius.clone()
            correct_pred_rob_radius[classes_pred != batch_labels] = -1
            return rob_radius, correct_pred_rob_radius

        return rob_radius

    def _roe_sample_aggregation_margin(
        self,
        votes_per_class: torch.Tensor,
        prediction_per_partition: torch.Tensor,
        logits_per_partition_and_class: torch.Tensor,
        batch_data: torch.Tensor,
        batch_labels: torch.Tensor,
        label_match: bool = False,
    ) -> torch.Tensor:
        """
        Compute the aggregation margin using DPA with Run-Off Election for a given ensemble of parameters and a batch of data.

        :param torch.Tensor votes_per_class: A tensor containing the votes per class for each sample in the batch.
        :param torch.Tensor prediction_per_partition: A tensor containing the predictions for each partition.
        :param torch.Tensor logits_per_partition_and_class: A tensor containing the logits for each partition and class.
        :param torch.Tensor batch_data: The data batch to certify.

        Returns:
                torch.Tensor: A tensor containing the aggregation margin for each sample in the batch.
        """
        rob_radius = torch.zeros(batch_data.shape[0], device=self.device, dtype=torch.int64)
        classes_pred, classes_sec = None, None
        # % ROUND 1
        if self.num_classes == 2:
            top_2 = torch.topk(votes_per_class, 2, dim=1)
            top_2_classes, top_2_votes = top_2.indices, top_2.values
            votes_pred, votes_sec = top_2_votes[:, 0], top_2_votes[:, 1]
            classes_pred, classes_sec = top_2_classes[:, 0], top_2_classes[:, 1]
            votes_pred, votes_sec, classes_pred, classes_sec = self._handle_ties(votes_pred, votes_sec, classes_pred, classes_sec)
            delta = ((votes_pred - votes_sec + 1) / 2).ceil()
            round_1_cert = torch.max(0, delta)
        else:
            top_3 = torch.topk(votes_per_class, 3, dim=1)
            top_3_classes, top_3_votes = top_3.indices, top_3.values
            votes_pred, votes_c1, votes_c2 = top_3_votes[:, 0], top_3_votes[:, 1], top_3_votes[:, 2]
            classes_pred, classes_sec, classes_c2 = top_3_classes[:, 0], top_3_classes[:, 1], top_3_classes[:, 2]
            # Check: 2-3 and swap, then 1-2 and swap. If 1-2-3 all tied, then one extra 2-3 check needed
            votes_c1, votes_c2, classes_sec, classes_c2 = self._handle_ties(votes_c1, votes_c2, classes_sec, classes_c2)
            votes_pred, votes_c1, classes_pred, classes_sec = self._handle_ties(votes_pred, votes_c1, classes_pred, classes_sec)
            votes_c1, votes_c2, classes_sec, classes_c2 = self._handle_ties(votes_c1, votes_c2, classes_sec, classes_c2)
            delta_c1 = ((votes_pred - votes_c1 + 1) / 2).ceil()
            delta_c2 = ((votes_pred - votes_c2 + 1) / 2).ceil()
            # The second multiplicand ensures we only select positive deltas, otherwise 0 -- i.e. max
            round_1_cert = torch.maximum(torch.zeros_like(delta_c1, device=self.device, dtype=torch.int64), delta_c1 + delta_c2)
        # % ROUND 2
        round_2_cert = torch.zeros(batch_data.shape[0], device=self.device, dtype=torch.int64)
        for partition_idx in range(self.num_partitions):
            partition_pred = prediction_per_partition[partition_idx]
            partition_logits = logits_per_partition_and_class[partition_idx]
            # Check if the prediction is in the top 2 classes
            # indicator_top_2_mask = (partition_pred == classes_pred) | (partition_pred == classes_sec)
            # indicator_top_2_predictions = partition_pred * indicator_top_2_mask.to(dtype=torch.int64)
            # Check if the logits of the top class are greater than the runner-up class logits
            partition_logits_pred = [partition_logits[i][classes_pred[i]] for i in range(partition_logits.shape[0])]
            partition_logits_sec = [partition_logits[i][classes_sec[i]] for i in range(partition_logits.shape[0])]
            indicator_logits = torch.tensor(partition_logits_pred) > torch.tensor(partition_logits_sec)
            indicator_logits = indicator_logits.to(dtype=torch.int64, device=self.device)

            round_2_cert += indicator_logits  # * indicator_top_2_predictions
        round_2_cert = (round_2_cert / 2).ceil()

        # % Final certificate
        rob_radius = torch.minimum(round_1_cert, round_2_cert)
        if label_match:
            correct_pred_rob_radius = rob_radius.clone()
            correct_pred_rob_radius[classes_pred != batch_labels] = -1
            return rob_radius, correct_pred_rob_radius

        return rob_radius

    def _handle_ties(
        self, votes_1: torch.Tensor, votes_2: torch.Tensor, classes_1: torch.Tensor, classes_2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert votes_1.ndim == 1 and classes_1.ndim == 1, "Votes and classes must be 2D tensors with shapes [batch_size]"
        condition = (votes_1 == votes_2) & (classes_1 > classes_2)
        swap_indices = condition.nonzero(as_tuple=True)[0]
        classes_1[swap_indices], classes_2[swap_indices] = classes_2[swap_indices], classes_1[swap_indices]

        return votes_1, votes_2, classes_1, classes_2

    @torch.no_grad()
    def _partitions_intrinsic_robustness(
        self,
        batch_data: torch.Tensor,
        batch_labels: torch.Tensor,
        ensemble_methods: list[CertificationMethod],
        robustness_setup: RobustnessSetup = RobustnessSetup.MEDIUM,
        ignore_label: bool = False,
    ) -> torch.Tensor:
        """Compute intrinsic robustness for just one sample of ensemble methods."""
        batch_size = batch_data.shape[0]
        intrinsic_robustness = torch.zeros((self.num_partitions, batch_size), device=self.device, dtype=torch.int64)
        for partition_idx in range(self.num_partitions):
            cert_method = ensemble_methods[partition_idx]
            # Get the parameters for this sample -- cast the method index from tensor to int
            match cert_method:
                case CertificationMethod.SGD:
                    intrinsic_robustness[partition_idx] = torch.zeros((batch_size,), device=self.device, dtype=torch.int64)
                case CertificationMethod.POINTWISE_RDP:
                    # ensemble_state_dicts_partition = self._load_dpa_partition_for_cert_method(cert_method, partition_idx)
                    state_dict_load_func = lambda model_idx: self._load_single_partition_model(partition_idx, cert_method, model_idx)
                    num_models = self.hparams_pointwise_rdp["mechanism_samples"]
                    agg_multinomial, agg_softmax = aggregate_predictions_batch(
                        self.dp_model,
                        state_dict_load_func,
                        num_models,
                        batch_data,
                        batch_labels,
                        self.num_classes,
                        self.device,
                    )
                    intrinsic_robustness[partition_idx] = torch.from_numpy(
                        self.ptwise_rdp_certifier.ptwise_intrinsic_robustness(
                            "dp_bagging_softmax_prob", agg_multinomial, agg_softmax, ignore_label=ignore_label
                        )
                    )
                case CertificationMethod.HYBRID_RDP:
                    raise NotImplementedError("Hybrid RDP certification is not implemented (properly) yet.")
                case CertificationMethod.AGT:
                    raise NotImplementedError("AGT certification is not implemented (properly) yet.")
                case _:
                    raise ValueError(f"Unknown certification method: {cert_method}")

        return intrinsic_robustness

    def _handle_partition(
        self, cert_method: CertificationMethod, hparams: dict, partition_idx: int, idx_group: torch.Tensor
    ) -> tuple[dict, list[dict]]:
        set_seed(self.seed + partition_idx)
        self.model.trainable_params = deepcopy(self.initial_params.clone().detach())
        if idx_group.numel() == 0:
            return self._handle_empty_partition()

        logger.info(f"Training partition {partition_idx + 1}/{self.num_partitions} with method {cert_method}")
        train_data, train_labels = zip(*[self.train_set[i] for i in idx_group])
        train_data = torch.stack(list(train_data), dim=0).to(self.device)
        train_labels = torch.stack(list(train_labels), dim=0).to(self.device)
        train_subset = TensorDataset(train_data, train_labels)
        match cert_method:
            case CertificationMethod.SGD:
                state_dict = self._vanilla_train_loop(train_subset, hparams)
                self._save_single_partition_model(state_dict, partition_idx, cert_method, 0)
            case CertificationMethod.POINTWISE_RDP:
                self.ptwise_rdp_certifier.update_train_set(train_subset)
                save_func = lambda model_state_dict, model_idx: self._save_single_partition_model(
                    model_state_dict, partition_idx, cert_method, model_idx
                )
                self.ptwise_rdp_certifier.certify_points(external_save_func=save_func, logfile=self.logfile)
            case CertificationMethod.HYBRID_RDP:
                raise NotImplementedError("Hybrid RDP certification is not implemented (properly) yet.")
            case CertificationMethod.AGT:
                save_dir_agt_partition = os.path.join(self.save_load_dir, f"partition_{partition_idx}")
                self.agt_certifier.certify(
                    hparams["ks_private"], hparams["clip_gammas"], train_set=train_subset, save_load_dir=save_dir_agt_partition, logfile=self.logfile
                )
            case _:
                raise ValueError(f"Unknown certification method: {cert_method}")

    def _vanilla_train_loop(self, train_subset: Dataset, hparams: dict) -> dict:
        assert "epochs" in hparams, "Number of epochs must be specified in hyperparameters"
        assert "lr" in hparams, "Learning rate must be specified in hyperparameters"
        assert "batch_size" in hparams, "Batch size must be specified in hyperparameters"

        model = self.make_model().to(self.device)
        # Setup training parameters
        batch_size = hparams["batch_size"]
        epochs = hparams["epochs"]
        lr = hparams["lr"]
        weight_decay = hparams.get("weight_decay", 0.0)
        sgd = hparams.get("sgd", False)
        momentum = hparams.get("momentum", 0.9)
        nesterov = hparams.get("nesterov", False)
        loader = self.get_dataloader(train_subset, batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if sgd:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        criterion = torch.nn.CrossEntropyLoss() if self.num_classes > 2 else torch.nn.BCEWithLogitsLoss()

        # Train
        is_binary_classification = self.num_classes == 2
        batch_acc = lambda preds, labels: (preds == labels).float().mean().item()
        compute_preds = lambda out: torch.sigmoid(out).round().int() if is_binary_classification else torch.argmax(out, dim=1).int()
        progress_bar = trange(
            epochs,
            desc="Epoch",
        )

        for epoch in progress_bar:
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = compute_preds(outputs)
                acc = batch_acc(preds, labels.int())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix({"Epoch": epoch + 1, "Batch": i + 1, "Loss": loss.item(), "Accuracy": acc})

        accuracy_score = accuracy(self.model, self.test_loader, self.device)
        logger.info(f"Inference accuracy on whole clean test set after training: {accuracy_score:.4f}")
        state_dict = deepcopy(model.state_dict())
        del optimizer, criterion, loader, model
        gc.collect()
        torch.cuda.empty_cache()

        return state_dict

    @torch.no_grad()
    def _cast_ensemble_vote(
        self, ensemble_cert_methods: list[CertificationMethod], batch_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the ensemble vote based on the parameters of each partition of the ensemble.
        This is a simple majority voting mechanism.

        :param list[torch.Tensor] ensemble_params:
                A list of tensors containing the parameters of each partition of the ensemble.
                One model can have multiple parameters sets if the certification method does ensembling in turn, such as RDP-based methods.

        Returns:
                tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                        - votes_per_class: A tensor of shape (batch_size, num_classes) containing the votes for each class.
                        - prediction_per_partition: A tensor of shape (num_partitions, batch_size) containing the predicted class for each partition.
                        - logits_per_partition: A tensor of shape (num_partitions, batch_size, num_classes) containing the logits for each partition.
        """
        model = self.model
        votes_per_class = torch.zeros((batch_data.shape[0], self.num_classes), device=self.device)
        prediction_per_partition = torch.zeros((self.num_partitions, batch_data.shape[0]), device=self.device, dtype=torch.int64)
        logits_per_partition = torch.zeros((self.num_partitions, batch_data.shape[0], self.num_classes), device=self.device)
        for partition_idx, partition_cert_method in enumerate(ensemble_cert_methods):
            partition_state_dicts = self._load_dpa_partition_for_cert_method(partition_cert_method, partition_idx)
            average_partition_logit = torch.zeros((batch_data.shape[0], self.num_classes), device=self.device)
            votes_for_partition = torch.zeros((batch_data.shape[0], self.num_classes), device=self.device)
            # Handle a partition with multiple models
            for state_dict in partition_state_dicts:
                try:
                    model.load_state_dict(state_dict, strict=True)
                except RuntimeError as _:
                    model = self.dp_model
                    model.load_state_dict(state_dict, strict=True)
                preds, logits = get_prediction(model, batch_data, self.device, with_logits=True)
                assert logits.ndim == 2, "Logits should be a 2D tensor"
                assert preds.ndim == 1, "Predictions should be a 1D tensor"
                average_partition_logit += logits
                votes_for_partition += torch.nn.functional.one_hot(preds, num_classes=self.num_classes)
            # Accumulate
            logits_per_partition[partition_idx] = average_partition_logit / len(partition_state_dicts)
            predicted_partition_class = torch.argmax(votes_for_partition, dim=1).to(dtype=torch.int64)
            prediction_per_partition[partition_idx] = predicted_partition_class
            partition_vote = torch.nn.functional.one_hot(predicted_partition_class, num_classes=self.num_classes)
            votes_per_class += partition_vote
            del partition_state_dicts
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return votes_per_class.detach().clone(), prediction_per_partition.detach().clone(), logits_per_partition.detach().clone()

    def _partition_with_hashing(self) -> list[torch.Tensor]:
        train_data, train_labels = zip(*[(self.train_set[i][0], self.train_set[i][1]) for i in range(len(self.train_set))])
        train_data = torch.stack(list(train_data), dim=0)
        train_labels = torch.stack(list(train_labels), dim=0)
        # Prepare for sorting
        train_data = (train_data * 255).int()
        hashes = tensor_generic_hash(train_data, self.num_partitions)
        idx_groups = [(hashes == i).nonzero().squeeze() for i in range(self.num_partitions)]
        # ! For now we handle empty groups by just not training the empty partition. If we decide to remove, uncomment line below
        # idx_group = [idx for idx in idx_group if idx.numel() > 0]
        for i in range(len(idx_groups)):
            if idx_groups[i].numel() == 0:
                logger.warning(f"Partition {i} is empty, skipping.")
                continue
            idx_group_labels = train_labels[idx_groups[i]].unsqueeze(1).int()
            idx_group_data = train_data[idx_groups[i]].reshape(idx_groups[i].shape[0], -1)
            idx_groups[i] = idx_groups[i][np.lexsort(torch.cat((idx_group_labels, idx_group_data), dim=1).cpu().numpy().transpose())]
        # Reset
        train_data = train_data.float() / 255.0

        return idx_groups

    def _partition_disjoint_naive(self, shuffle: bool = True) -> list[torch.Tensor]:
        train_set_size = len(self.train_set)
        partition_size = train_set_size // self.num_partitions
        shuffled_indices = torch.arange(train_set_size)
        if shuffle:
            # Shuffle uniformly once
            shuffled_indices = torch.randperm(train_set_size)
        idx_groups = []
        for partition_idx in range(self.num_partitions):
            start_index = partition_idx * partition_size
            end_index = (start_index + partition_size) if partition_idx < self.num_partitions - 1 else train_set_size
            partition_indices = shuffled_indices[start_index:end_index]
            idx_groups.append(partition_indices.detach().clone())

        return idx_groups

    def _partition_bagging_with_replacement(self) -> list[torch.Tensor]:
        # Partition into self.num_partitions by sampling at each step with replacement
        train_set_size = len(self.train_set)
        partition_size = train_set_size // self.num_partitions
        idx_groups = []
        for partition_idx in range(self.num_partitions):
            # Sample uniformly with replacement
            partition_indices = torch.randint(0, train_set_size, (partition_size,), device=self.device, dtype=torch.int64)
            idx_groups.append(partition_indices)

        return idx_groups

    def _handle_empty_partition(self) -> tuple[dict, torch.Tensor]:
        """
        Handle the case where a partition is empty. This is a fallback method that returns the initial model parameters.
        """
        logger.warning("Encountered an empty partition, returning initial model parameters")
        acc_score = accuracy(self.model, self.test_loader, self.device)
        return {0: acc_score}, self.model.interval_params.unsqueeze(0)

    def _save_hparams(self, category: str):
        rf, ext = os.path.splitext(self.result_file)
        params_file = rf + "_params" + ext
        # We need to copy the dictionaries because: https://support.atlassian.com/bitbucket-cloud/docs/yaml-anchors/
        self.hparams_ensemble["hp_sgd"] = deepcopy(self.hparams_sgd)
        self.hparams_ensemble["hp_hrdp"] = deepcopy(self.hparams_hybrid_rdp)
        self.hparams_ensemble["hp_prdp"] = deepcopy(self.hparams_pointwise_rdp)
        self.hparams_ensemble["hp_agt"] = deepcopy(self.hparams_agt)
        write_results_to_file(
            params_file,
            {
                "hyperparams_ensemble": self.hparams_ensemble,
            },
            category,
        )

    def _save_dpa_model_for_cert_method(
        self, partition_idx: int, cert_method: CertificationMethod, partition_state_dicts_for_cert_method: torch.Tensor, cert_data: dict
    ) -> None:
        save_dir = os.path.join(self.save_load_dir, f"partition_{partition_idx}")
        # Save the params and metadata for each partition and certification method
        for state_dict_idx, state_dict in enumerate(partition_state_dicts_for_cert_method):
            save_model_state_dict(state_dict, str(cert_method) + f"_model_{state_dict_idx}.pt", save_dir)
        torchsave(cert_data, str(cert_method) + "_metadata.pt", save_dir)
        logger.info(f"Saved parameters for partition {partition_idx + 1} with method {cert_method} to {save_dir}")

    def _save_single_partition_model(self, state_dict: dict, partition_idx: int, cert_method: CertificationMethod, model_idx: int) -> None:
        save_dir = os.path.join(self.save_load_dir, f"partition_{partition_idx}", str(cert_method))
        save_model_state_dict(state_dict, f"model_{model_idx}.pt", save_dir)
        logger.info(f"Saved state dict for partition {partition_idx + 1} with method {cert_method} and model index {model_idx} to {save_dir}")

    def _load_single_partition_model(self, partition_idx: int, cert_method: CertificationMethod, model_idx: int) -> dict:
        load_dir = os.path.join(self.save_load_dir, f"partition_{partition_idx}", str(cert_method))
        state_dict = get_state_dict_from_file(f"model_{model_idx}.pt", self.device, load_dir)
        logger.info(f"Loaded state dict for partition {partition_idx + 1} with method {cert_method} and model index {model_idx} from {load_dir}")

        return state_dict

    def _load_dpa_partition_for_cert_method(self, cert_method: CertificationMethod, partition_idx: int) -> list[dict]:
        result_dir_path = get_result_dir_path()
        load_dir = os.path.join(self.save_load_dir, f"partition_{partition_idx}", str(cert_method))
        full_load_dir_path = os.path.join(result_dir_path, load_dir)
        num_state_dicts = len(os.listdir(full_load_dir_path))

        state_dicts = []
        for state_dict_idx in range(num_state_dicts):
            state_dict = get_state_dict_from_file(f"model_{state_dict_idx}.pt", self.device, load_dir)
            state_dicts.append(state_dict)

        return state_dicts

    def setup_logging_and_saving(self, save_kwargs: dict) -> None:
        """
        Set up logging and saving for the certifier.
        """
        self.save, self.load, self.logfile, self.result_file, self.category = False, False, None, None, None
        self.root_dir = f"{self.__str__()}"
        self.suffix = f"partitions_{self.num_partitions}"
        self.save_load_dir = os.path.join(self.root_dir, self.method_name, self.suffix)
        if "logfile_name" in save_kwargs and save_kwargs["logfile_name"]:
            self.logfile_name = save_kwargs["logfile_name"]
        if "write_to_file" in save_kwargs and save_kwargs["write_to_file"]:
            self.result_file = "framework_" + self.__str__() + ".yaml"
        # Log everytime -- either to temp or to a specific file
        self.logfile = get_logfile_path(os.path.join(self.root_dir, "dpa", "logs", self.logfile_name + ".log"))
        if not os.path.exists(os.path.dirname(self.logfile)):
            os.makedirs(os.path.dirname(self.logfile))
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
        logger.remove()
        logger.add(self.logfile, level="INFO", format=log_format, colorize=False, backtrace=True, diagnose=True)

    def get_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @abstractmethod
    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        pass

    @abstractmethod
    def make_model(self) -> Generic_NN:
        pass

    @abstractmethod
    def rdp_certifier(self, hyperparams: dict, save_kwargs: dict) -> StabilityCertifierWithRDP:
        pass

    @abstractmethod
    def agt_certifier(self, hyperparams: dict, save_kwargs: dict) -> AGTCertifier:
        pass


class HalfmoonsCertifier(StabilityCertifierWithDPA):
    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None):
        self.num_classes = 2
        super().__init__(hyperparams, device, save_kwargs=save_kwargs)

    def make_model(self) -> Generic_NN:
        input_dim, output_dim, hidden_dims = 2, 1, [10]
        return FCN(input_dim, output_dim, hidden_dims, with_bias=False).to(self.device)

    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        return Halfmoons(dset_type, num_samples_train_test=(50000, 5000), noise=0.1)

    def rdp_certifier(self, hyperparams: dict, save_kwargs: dict) -> RDP_HalfmoonsCertifier:
        return RDP_HalfmoonsCertifier(hyperparams, self.device, save_kwargs=save_kwargs)

    def agt_certifier(self, hyperparams: dict, save_kwargs: dict) -> AGT_HalfmoonsCertifier:
        return AGT_HalfmoonsCertifier(hyperparams, self.device, save_kwargs=save_kwargs)

    def __str__(self):
        return "halfmoons"


class BlobsCertifier(StabilityCertifierWithDPA):
    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None):
        self.num_classes = 2
        super().__init__(hyperparams, device, save_kwargs=save_kwargs)

    def make_model(self) -> Generic_NN:
        input_dim, output_dim, hidden_dims = 2, 1, [10]
        return FCN(input_dim, output_dim, hidden_dims, with_bias=False).to(self.device)

    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        return Blobs(dset_type, num_samples_train_test=(100000, 10000))

    def rdp_certifier(self, hyperparams: dict, save_kwargs: dict) -> RDP_HalfmoonsCertifier:
        # TODO if we decide to do experiments with RDP -- THIS SHOULD BE RDP_BlobsCertifier
        return RDP_HalfmoonsCertifier(hyperparams, self.device, save_kwargs=save_kwargs)

    def agt_certifier(self, hyperparams: dict, save_kwargs: dict) -> AGT_HalfmoonsCertifier:
        return AGT_BlobsCertifier(hyperparams, self.device, save_kwargs=save_kwargs)

    def __str__(self):
        return "blobs"


class MnistCertifier(StabilityCertifierWithDPA):
    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None):
        self.num_classes = 10
        super().__init__(hyperparams, device, save_kwargs=save_kwargs)

    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        return VanillaMNIST(dset_type, self.seed, flatten=self.flatten)

    def make_model(self) -> Generic_NN:
        input_dim, output_dim, hidden_dims = 784, 10, [32]
        return FCN(input_dim, output_dim, hidden_dims).to(self.device)

    def rdp_certifier(self, hyperparams: dict, save_kwargs: dict) -> RDP_HalfmoonsCertifier:
        return RDP_MnistCertifier(hyperparams, self.device, save_kwargs=save_kwargs)

    def agt_certifier(self, hyperparams: dict, save_kwargs: dict) -> AGT_HalfmoonsCertifier:
        return AGT_MnistCertifier(hyperparams, self.device, save_kwargs=save_kwargs)

    def __str__(self):
        return "mnist"


class Cifar10Certifier(StabilityCertifierWithDPA):
    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None, pre_trained: bool = False, freeze_pt_blocks: bool = False):
        self.num_classes = 10
        self.pre_trained = pre_trained
        self.freeze_resnet_block = freeze_pt_blocks
        super().__init__(hyperparams, device, save_kwargs=save_kwargs)

    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        return CIFAR(dset_type, self.seed)

    def make_model(self) -> Generic_NN:
        model = None
        if self.pre_trained:
            hidden_layer_sizes, output_dim = [], 10
            model = Resnet18Finetune(hidden_layer_sizes, output_dim)
            if self.freeze_resnet_block:
                model.freeze_pretrained_blocks()
        else:
            model = Resnet18(output_dim=10)

        return model

    def rdp_certifier(self, hyperparams: dict, save_kwargs: dict) -> RDP_Cifar10Certifier:
        return RDP_Cifar10Certifier(hyperparams, self.device, save_kwargs=save_kwargs)

    def agt_certifier(self, hyperparams: dict, save_kwargs: dict) -> AGT_Cifar10Certifier:
        return AGT_Cifar10Certifier(hyperparams, self.device, save_kwargs=save_kwargs)

    def __str__(self):
        return "cifar10"


def check_hyperparams_present(hyperparams: dict, save_kwargs: dict) -> bool:
    assert "num_partitions" in hyperparams, "num_partitions must be specified"
    assert "seed" in hyperparams, "seed must be specified"
    assert "method_name" in hyperparams, "method_name must be specified"
    assert "hp_sgd" in hyperparams, "Hyperparameters for SGD must be specified"
    assert "hp_hrdp" in hyperparams, "Hyperparameters for Hybrid RDP must be specified"
    assert "hp_prdp" in hyperparams, "Hyperparameters for Pointwise RDP must be specified"
    assert "hp_agt" in hyperparams, "Hyperparameters for AGT must be specified"
    assert "hit_ratios" in hyperparams["hp_hrdp"], "Hyperparameters for Hybrid RDP must contain 'hit_ratios' for dispatch"
    assert (
        "ks_private" in hyperparams["hp_agt"] and "clip_gammas" in hyperparams["hp_agt"]
    ), "Hyperparameters for AGT must contain 'ks_private' and 'clip_gammas' for dispatch"
    assert len(hyperparams["hp_agt"]["clip_gammas"]) == 1, "AGT hyperparameters must contain exactly the optimal clip gamma value"
    if "save" in save_kwargs:
        assert "load" not in save_kwargs, "Cannot specify both save and load"
    if "load" in hyperparams:
        assert "save" not in save_kwargs, "Cannot specify both save and load"
        assert "save" not in save_kwargs, "Cannot specify both save and load"
        assert "save" not in save_kwargs, "Cannot specify both save and load"
        assert "save" not in save_kwargs, "Cannot specify both save and load"
        assert "save" not in save_kwargs, "Cannot specify both save and load"
        assert "save" not in save_kwargs, "Cannot specify both save and load"
