import sys

import numpy as np
import torch

from certifiable_learning_stability.certification_methods import (
    AggregationType,
    CertificationMethod,
)
from certifiable_learning_stability.dpa_certifier import Cifar10Certifier
from certifiable_learning_stability.rdp_certifier import (
    Cifar10Certifier as Cifar10RDPCertifier,
)
from experiments.misc import (
    dummy_hparams_agt,
    dummy_hparams_hrdp,
    dummy_hparams_prdp,
    dummy_hparams_sgd,
)
from experiments.reproducibility import get_device, make_reproducible

SEED = 42
make_reproducible(SEED)
# --------------------------------------------------------- #
# Hybrid DPA - Finetuned Resnet18 (Non-Frozen Resnet Block) #
# --------------------------------------------------------- #


def rdp_bagging(device):
    hyperparams = {
        "epochs": 40,
        "lr": 0.001,
        "mechanism_samples": 250,
        "confidence": 0.98,
        "seed": SEED,
        "sigma": 0.3,
        "max_grad_norm": 25.0,
        "sample_rate": 128 / 10000,
        "sub_training_size": 10000,
    }
    kwargs = {"logfile_name": "baseline_cifar10", "write_to_file": True, "load": True}
    cifar10_rdp_certifier = Cifar10RDPCertifier(hyperparams, device, save_kwargs=kwargs, pretrained=False)

    test_set_subset = torch.utils.data.Subset(cifar10_rdp_certifier.test_set, indices=torch.randperm(len(cifar10_rdp_certifier.test_set))[:5000])
    new_loader = cifar10_rdp_certifier.get_dataloader(test_set_subset, batch_size=500, shuffle=False)

    cifar10_rdp_certifier.test_set = test_set_subset
    cifar10_rdp_certifier.test_loader = new_loader

    cert_dict = cifar10_rdp_certifier.certify_points("dp_bagging_softmax_prob")
    print(cert_dict)


def rdp_bagging_finetune(device):
    hyperparams = {
        "epochs": 12,
        "lr": 0.001,
        "mechanism_samples": 250,
        "confidence": 0.95,
        "seed": SEED,
        "sigma": 0.4,
        "max_grad_norm": 22.0,
        "sample_rate": 128 / 10000,
        "weight_decay": 0.0001,
        "sub_training_size": 10000,
    }
    kwargs = {"logfile_name": "baseline_cifar10", "write_to_file": True, "load": True}
    cifar10_rdp_certifier = Cifar10RDPCertifier(hyperparams, device, save_kwargs=kwargs, pretrained=True)

    test_set_subset = torch.utils.data.Subset(cifar10_rdp_certifier.test_set, indices=torch.randperm(len(cifar10_rdp_certifier.test_set))[:5000])
    new_loader = cifar10_rdp_certifier.get_dataloader(test_set_subset, batch_size=500, shuffle=False)

    cifar10_rdp_certifier.test_set = test_set_subset
    cifar10_rdp_certifier.test_loader = new_loader

    cert_dict = cifar10_rdp_certifier.certify_points("dp_bagging_softmax_prob")
    print(cert_dict)


def dpa_rdp_and_bagging(device):
    hp_hrdp_cifar = dummy_hparams_hrdp()
    hp_agt_cifar = dummy_hparams_agt()
    hp_sgd_cifar = dummy_hparams_sgd()
    hp_prdp_cifar = {
        "epochs": 20,
        "lr": 0.001,
        "mechanism_samples": 100,
        "confidence": 0.95,
        "seed": SEED,
        "sigma": 0.35,
        "max_grad_norm": 22.0,
        "sample_rate": 128 / 5000,
        "sub_training_size": 5000,
    }
    hyperparams_dpa_cifar = {
        "num_partitions": 5,
        "test_batch_size": 100,
        "seed": SEED,
        "method_name": "dpa_prdp_bagging_raw_resnet",
        "hp_sgd": hp_sgd_cifar,
        "hp_agt": hp_agt_cifar,
        "hp_hrdp": hp_hrdp_cifar,
        "hp_prdp": hp_prdp_cifar,
    }

    kwargs = {"logfile_name": "generalized_framework_inference_2", "write_to_file": True}
    cifar10_dpa_certifier = Cifar10Certifier(hyperparams_dpa_cifar, device, save_kwargs=kwargs, pre_trained=False)

    # test_set_subset = torch.utils.data.Subset(cifar10_dpa_certifier.test_set, indices=torch.randperm(len(cifar10_dpa_certifier.test_set))[:1200])
    # cifar10_dpa_certifier.get_metrics_inference(CertificationMethod.POINTWISE_RDP, agg_type=AggregationType.DPA, test_set=test_set_subset)
    # cifar10_dpa_certifier.get_metrics_inference(CertificationMethod.POINTWISE_RDP, agg_type=AggregationType.ROE, test_set=test_set_subset)

    test_set_subset = torch.utils.data.Subset(cifar10_dpa_certifier.test_set, indices=torch.randperm(len(cifar10_dpa_certifier.test_set))[:600])
    ks_poison = np.arange(1, 1000, 4)
    cifar10_dpa_certifier.multi_sample_certification(CertificationMethod.POINTWISE_RDP, AggregationType.DPA, ks_poison, test_set=test_set_subset)
    # cifar10_dpa_certifier.multi_sample_certification(CertificationMethod.POINTWISE_RDP, AggregationType.ROE, ks_poison, test_set=test_set_subset)


def dpa_rdp_and_bagging_finetune(device):
    hp_hrdp_cifar = dummy_hparams_hrdp()
    hp_agt_cifar = dummy_hparams_agt()
    hp_sgd_cifar = dummy_hparams_sgd()
    hp_prdp_cifar = {
        "epochs": 18,
        "lr": 0.0025,
        "mechanism_samples": 250,
        "confidence": 0.98,
        "seed": SEED,
        "sigma": 0.35,
        "max_grad_norm": 23.0,
        "sample_rate": 100 / 3500,
        "sub_training_size": 3500,
        "weight_decay": 0.0005,
    }
    hyperparams_dpa_cifar = {
        "num_partitions": 8,
        "test_batch_size": 100,
        "seed": SEED,
        "method_name": "dpa_prdp_bagging_finetune_resnet",
        "hp_sgd": hp_sgd_cifar,
        "hp_agt": hp_agt_cifar,
        "hp_hrdp": hp_hrdp_cifar,
        "hp_prdp": hp_prdp_cifar,
    }

    kwargs = {"logfile_name": "generalized_framework_inference", "write_to_file": True}
    cifar10_dpa_certifier = Cifar10Certifier(hyperparams_dpa_cifar, device, save_kwargs=kwargs, pre_trained=True)

    # test_set_subset = torch.utils.data.Subset(cifar10_dpa_certifier.test_set, indices=torch.randperm(len(cifar10_dpa_certifier.test_set))[:1000])
    # cifar10_dpa_certifier.get_metrics_inference(CertificationMethod.POINTWISE_RDP, agg_type=AggregationType.DPA, test_set=test_set_subset)
    # cifar10_dpa_certifier.get_metrics_inference(CertificationMethod.POINTWISE_RDP, agg_type=AggregationType.ROE, test_set=test_set_subset)

    test_set_subset = torch.utils.data.Subset(cifar10_dpa_certifier.test_set, indices=torch.randperm(len(cifar10_dpa_certifier.test_set))[:600])
    ks_poison = np.arange(1, 1000, 4)
    cifar10_dpa_certifier.multi_sample_certification(CertificationMethod.POINTWISE_RDP, AggregationType.DPA, ks_poison, test_set=test_set_subset)
    # cifar10_dpa_certifier.multi_sample_certification(CertificationMethod.POINTWISE_RDP, AggregationType.ROE, ks_poison, test_set=test_set_subset)


def dpa_rdp_and_bagging_finetune_partitions_5(device):
    hp_hrdp_cifar = dummy_hparams_hrdp()
    hp_agt_cifar = dummy_hparams_agt()
    hp_sgd_cifar = dummy_hparams_sgd()
    hp_prdp_cifar = {
        "epochs": 20,
        "lr": 0.0025,
        "mechanism_samples": 250,
        "confidence": 0.98,
        "seed": SEED,
        "sigma": 0.35,
        "max_grad_norm": 24.0,
        "sample_rate": 128 / 6500,
        "sub_training_size": 6250,
    }
    hyperparams_dpa_cifar = {
        "num_partitions": 5,
        "test_batch_size": 100,
        "seed": SEED,
        "method_name": "dpa_prdp_bagging_finetune_resnet",
        "hp_sgd": hp_sgd_cifar,
        "hp_agt": hp_agt_cifar,
        "hp_hrdp": hp_hrdp_cifar,
        "hp_prdp": hp_prdp_cifar,
    }

    kwargs = {"logfile_name": "generalized_framework_inference_3", "write_to_file": True}
    cifar10_dpa_certifier = Cifar10Certifier(hyperparams_dpa_cifar, device, save_kwargs=kwargs, pre_trained=True)

    # test_set_subset = torch.utils.data.Subset(cifar10_dpa_certifier.test_set, indices=torch.randperm(len(cifar10_dpa_certifier.test_set))[:1200])
    # cifar10_dpa_certifier.get_metrics_inference(CertificationMethod.POINTWISE_RDP, agg_type=AggregationType.DPA, test_set=test_set_subset)
    # cifar10_dpa_certifier.get_metrics_inference(CertificationMethod.POINTWISE_RDP, agg_type=AggregationType.ROE, test_set=test_set_subset)

    test_set_subset = torch.utils.data.Subset(cifar10_dpa_certifier.test_set, indices=torch.randperm(len(cifar10_dpa_certifier.test_set))[:600])
    ks_poison = np.arange(1, 1000, 4)
    cifar10_dpa_certifier.multi_sample_certification(CertificationMethod.POINTWISE_RDP, AggregationType.DPA, ks_poison, test_set=test_set_subset)
    # cifar10_dpa_certifier.multi_sample_certification(CertificationMethod.POINTWISE_RDP, AggregationType.ROE, ks_poison, test_set=test_set_subset)


if __name__ == "__main__":
    assert len(sys.argv) > 2, "Please provide two arguments. First argument - Method | Second argument - GPU index"
    first_arg = int(sys.argv[1])
    second_arg = int(sys.argv[2])
    device = get_device(index=second_arg)
    match first_arg:
        case 0:
            rdp_bagging(device)
        case 1:
            rdp_bagging_finetune(device)
        case 2:
            dpa_rdp_and_bagging(device)
        case 3:
            dpa_rdp_and_bagging_finetune(device)
        case 4:
            dpa_rdp_and_bagging_finetune_partitions_5(device)
        case _:
            raise ValueError("Invalid Argument.")
