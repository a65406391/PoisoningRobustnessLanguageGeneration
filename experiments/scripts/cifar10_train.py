import sys

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
device = get_device(index=0)


def rdp_bagging():
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
    kwargs = {"logfile_name": "baseline_cifar10", "write_to_file": True, "save": True}
    cifar10_rdp_certifier = Cifar10RDPCertifier(hyperparams, device, save_kwargs=kwargs, pretrained=False)

    cert_dict = cifar10_rdp_certifier.certify_points("dp_bagging_softmax_prob")
    print(cert_dict)


def rdp_bagging_finetune():
    hyperparams = {
        "epochs": 12,
        "lr": 0.001,
        "mechanism_samples": 250,
        "confidence": 0.98,
        "seed": SEED,
        "sigma": 0.4,
        "max_grad_norm": 22.0,
        "sample_rate": 128 / 10000,
        "weight_decay": 0.0001,
        "sub_training_size": 10000,
    }
    kwargs = {"logfile_name": "baseline_cifar10", "write_to_file": False, "save": True}
    cifar10_rdp_certifier = Cifar10RDPCertifier(hyperparams, device, save_kwargs=kwargs, pretrained=True)

    cert_dict = cifar10_rdp_certifier.certify_points("dp_bagging_softmax_prob")
    print(cert_dict)


def dpa_rdp_and_bagging():
    hp_hrdp_cifar = dummy_hparams_hrdp()
    hp_agt_cifar = dummy_hparams_agt()
    hp_sgd_cifar = dummy_hparams_sgd()
    hp_prdp_cifar = {
        "epochs": 30,
        "lr": 0.001,
        "mechanism_samples": 250,
        "confidence": 0.98,
        "seed": SEED,
        "sigma": 0.35,
        "max_grad_norm": 25.0,
        "sample_rate": 128 / 6000,
        "sub_training_size": 6000,
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

    kwargs = {"logfile_name": "generalized_framework", "write_to_file": True}
    cifar10_dpa_certifier = Cifar10Certifier(hyperparams_dpa_cifar, device, save_kwargs=kwargs)

    cifar10_dpa_certifier.train_dpa_partitions((CertificationMethod.POINTWISE_RDP,), partitioning_method="disjoint_bag")


def dpa_rdp_and_bagging_finetune():
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

    kwargs = {"logfile_name": "generalized_framework", "write_to_file": True}
    cifar10_dpa_certifier = Cifar10Certifier(hyperparams_dpa_cifar, device, save_kwargs=kwargs, pre_trained=True)

    cifar10_dpa_certifier.train_dpa_partitions((CertificationMethod.POINTWISE_RDP,), partitioning_method="disjoint_bag")


def dpa_rdp_and_bagging_finetune_partitions_5():
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

    kwargs = {"logfile_name": "generalized_framework", "write_to_file": True}
    cifar10_dpa_certifier = Cifar10Certifier(hyperparams_dpa_cifar, device, save_kwargs=kwargs, pre_trained=True)

    cifar10_dpa_certifier.train_dpa_partitions((CertificationMethod.POINTWISE_RDP,), partitioning_method="disjoint_bag")


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please provide the method idx as an argument."
    first_arg = int(sys.argv[1])
    match first_arg:
        case 0:
            rdp_bagging()
        case 1:
            rdp_bagging_finetune()
        case 2:
            dpa_rdp_and_bagging()
        case 3:
            dpa_rdp_and_bagging_finetune()
        case 4:
            dpa_rdp_and_bagging_finetune_partitions_5()
        case _:
            raise ValueError("Invalid Argument.")
