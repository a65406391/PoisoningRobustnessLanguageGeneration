def dummy_hparams_hrdp():
    return {
        "hit_ratios": [0],
        "epochs": 1,
        "sample_rate": 0.01,
        "lr": 0.1,
        "mechanism_samples": 3,
        "confidence": 0.01,
        "seed": 0,
        "sigma": 0.15,
        "max_grad_norm": 0.1,
        "sub_training_size": 1e5,
    }


def dummy_hparams_sgd():
    return {
        "epochs": 1,
        "batch_size": 1000,
        "lr": 0.1,
        "weight_decay": 1e-4,
    }


def dummy_hparams_agt():
    return {
        "ks_private": [0],
        "clip_gammas": [0.3],
        "epochs": 1,
        "batch_size": 100,
        "lr": 0.5,
        "lr_decay": 0.6,
        "lr_min": 1e-3,
        "weight_decay": 1e-4,
        "flatten": False,
    }


def dummy_hparams_prdp():
    return {
        "epochs": 1,
        "sample_rate": 0.01,
        "lr": 0.1,
        "mechanism_samples": 3,
        "confidence": 0.01,
        "seed": 0,
        "sigma": 0.15,
        "max_grad_norm": 0.1,
        "sub_training_size": 1e5,
    }
