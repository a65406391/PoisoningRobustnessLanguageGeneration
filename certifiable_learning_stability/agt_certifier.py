import os
from abc import ABC, abstractmethod
from copy import deepcopy

import abstract_gradient_training as agt
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from data_sets.blobs import Blobs
from data_sets.cifar import CIFAR
from data_sets.dset_type import DsetType
from data_sets.halfmoons import Halfmoons
from data_sets.mnist import VanillaMNIST
from experiments.save_utils import (
    get_logfile_path,
    get_result_dir_path,
    save_model_state_dict,
    write_results_to_file,
)

from .models.generic_nn import Generic_NN
from .models.resnet import Resnet18


class AGTCertifier(ABC):
    """
    Abstract base class for AGT certifiers. All subclasses must implement the following methods:
    - `get_original_dset`: Returns the original dataset for the given DsetType.
    - `make_model`: Returns the model (needs to be a torch.Sequential) to be trained and certified.
    This is based off https://github.com/psosnin/AbstractGradientTraining
    """

    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None) -> None:
        """
        Initialize the AGT certifier with hyperparameters, device, and saving/loading options.

        :param dict hyperparams: Dictionary containing hyperparameters for training.
            Hyperparams needed:
            - epochs: Number of training epochs. (required)
            - batch_size: Size of the training batches. (required)
            - lr: Learning rate for the optimizer. (required)
            - fragsize: Fragment size for training. (optional, default: 10000)
            - lr_decay: Learning rate decay factor. (optional, default: 0.0)
            - lr_min: Minimum learning rate. (optional, default: 0.0)
            - weight_decay: L2 regularization factor. (optional, default: 0.0)
        :param torch.device device: The device (CPU or GPU) on which the model will be trained.
        :param dict save_kwargs: Dictionary containing options for saving and loading the model.
            - save: Boolean indicating whether to save the model. (optional, default: False)
            - load: Boolean indicating whether to load a pre-trained model. (optional, default: False)
            Note: You cannot specify both `save` and `load` as True at the same time.
            - logfile_name: Name of the log file to write logs to. (optional, default: "temp")
            - write_to_file: Boolean indicating whether to write results to a file. (optional, default: False)
        """
        check_hyperparams_present(hyperparams, save_kwargs)
        # % Set general hyperparameters
        self.all_hyperparams = hyperparams  # Save all hyperparameters for future reference
        self.epochs = hyperparams["epochs"]
        self.batch_size = hyperparams["batch_size"]
        self.lr = hyperparams["lr"]
        self.fragsize = hyperparams.get("fragsize", 10000)
        self.lr_decay = hyperparams.get("lr_decay", 0.0)
        self.lr_min = hyperparams.get("lr_min", 0.0)
        self.l2_reg = hyperparams.get("weight_decay", 0.0)
        self.seed = hyperparams.get("seed", 42)  # Default seed for reproducibility
        self.flatten = hyperparams.get("flatten", False)  # Whether to flatten the input data
        # % Set up dataset, model and device
        self.device = device
        self.train_set = self.get_original_dset(DsetType.TRAIN_FULL)
        self.train_loader = self.get_dataloader(self.train_set, self.batch_size)
        self.test_set = self.get_original_dset(DsetType.TEST)
        self.test_data, self.test_labels = zip(*[(x, y) for x, y in self.test_set])
        self.test_data = torch.stack(list(self.test_data), dim=0).to(self.device)
        self.test_labels = torch.tensor(self.test_labels, dtype=torch.float32).to(self.device)
        self.model = self.make_model().to(self.device)
        # % Set up saving results to file
        self.setup_logging_and_saving(save_kwargs)

    def certify(
        self,
        ks_private: list[int],
        clip_gammas: list[float],
        train_set: Dataset = None,
        return_models: bool = False,
        save_load_dir: str = None,
        logfile: str = None,
    ) -> dict | tuple[dict, torch.Tensor]:
        """
        Train and certify the model using AGT with specified private k values and clipping gammas.

        :param list[int] ks_private: List of private k values for certification.
        :param list[float] clip_gammas: List of clipping gamma values for certification.
        :param bool, optional return_models: If True, also return the trained model parameters. This is only used for partition-based approaches,
        :param str, optional logfile:
            Path to the log file to write logs to. If None, the logfile will either be the save_kwargs["logfile_name"] or "temp". If it is not None,
            please provide the full path. For the latter, this will again be used by (partition) aggregation methods to log the results to the same file.
        when a certain partition is dispatched to be trained with AGT. If True, "clip_gammas" must be a singleton representing the optimal value.

        Returns:
            dict: A dictionary containing the certified accuracies for each combination of k_private and clip_gamma.
            If return_models is True, also returns a list of trained model parameters where index `i` of k_private corresponds to index `i` of trained_models_params.
        """
        assert isinstance(ks_private, list) and isinstance(clip_gammas, list), "ks_private and clip_gammas must be lists"
        if return_models:
            assert len(clip_gammas) == 1, "If return_models is True, clip_gammas must be a singleton list representing the optimal value"
        self._setup_save_load_dir(logfile=logfile, new_save_load_dir=save_load_dir)
        cert_accs, trained_models_params = {}, []
        train_loader = self.train_loader if train_set is None else self.get_dataloader(train_set, self.batch_size)
        for k_private in ks_private:
            assert isinstance(k_private, int), "k_poison must be an integer"
            cert_accs[k_private] = {}
            for clip_gamma in clip_gammas:
                assert isinstance(clip_gamma, float), "clip_gamma must be a float"
                print(f"model {self.model} with k_private={k_private} and clip_gamma={clip_gamma}")
                bounded_model = agt.bounded_models.IntervalBoundedModel(self.model)
                config = agt.AGTConfig(
                    n_epochs=self.epochs,
                    learning_rate=self.lr,
                    # For unbounded adversaries, we need to train with DP
                    k_private=k_private,
                    clip_gamma=clip_gamma,
                    lr_decay=self.lr_decay,
                    lr_min=self.lr_min,
                    l2_reg=self.l2_reg,
                    fragsize=self.fragsize,
                    loss="cross_entropy" if isinstance(self.criterion, torch.nn.CrossEntropyLoss) else "binary_cross_entropy",
                )
                # experiments/results
                rpath = get_result_dir_path()
                fname = f"k_{k_private}_clip_{clip_gamma}.pt"
                fpath = os.path.join(rpath, self.save_load_dir, fname)
                # logger.info(f"fpath is {fpath}, save: {self.save}")
                if self.load:
                    bounded_model.load_params(fpath)
                else:
                    bounded_model = agt.privacy_certified_training(bounded_model, config, train_loader)
                    if self.save:
                        # Do a dummy save in case the file and parent directories do not exist
                        save_model_state_dict(self.model, fname, self.save_load_dir)
                        # Once they are created, save the actual model
                        bounded_model.save_params(fpath)
                cert_accs[k_private][clip_gamma] = agt.test_metrics.test_accuracy(bounded_model, self.test_data, self.test_labels)
                if return_models:
                    curr_model_params = torch.cat([p.flatten() for p in bounded_model.param_n])
                    trained_models_params.append(curr_model_params)

        result_write_dict, worst_case_metadata = {}, {}
        for k_private, cert_accs_for_gamma in cert_accs.items():
            result_write_dict[f"k_{k_private}"], worst_case_metadata[f"k_{k_private}"] = {}, {}
            for clip_gamma, cert_acc in cert_accs_for_gamma.items():
                cert_acc = [round(float(cert_acc[0]), 5), round(float(cert_acc[1]), 5), round(float(cert_acc[2]), 5)]
                result_write_dict[f"k_{k_private}"][f"clip_{clip_gamma}"] = {"worst": cert_acc[0], "nominal": cert_acc[1], "best": cert_acc[2]}
                worst_case_metadata[f"k_{k_private}"][f"clip_{clip_gamma}"] = cert_acc[0]

        logger.info(f"AGT results for ks_private={ks_private} and clip_gammas={clip_gammas}: \n\t {worst_case_metadata}")

        if self.result_file is not None:
            rf, ext = os.path.splitext(self.result_file)
            results_file = rf + "_agt" + ext
            params_file = rf + "_agt_params" + ext
            write_results_to_file(
                results_file,
                {
                    "certified_accs": result_write_dict,
                },
                self.suffix,
            )
            params = deepcopy(self.all_hyperparams)
            params["fragsize"], params["lr_decay"], params["lr_min"], params["l2_reg"] = self.fragsize, self.lr_decay, self.lr_min, self.l2_reg
            write_results_to_file(params_file, params, self.suffix)

        if return_models:
            return worst_case_metadata, torch.stack(trained_models_params, dim=0)

        return cert_accs

    def vote_and_get_robustness(
        self, batch_data: torch.Tensor, batch_labels: torch.Tensor, k_private: int, clip_gamma: float, partition_save_path: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rpath = get_result_dir_path()
        fname = f"k_{k_private}_clip_{clip_gamma}.pt"
        fpath = os.path.join(rpath, partition_save_path, fname)

        bounded_model = agt.bounded_models.IntervalBoundedModel(deepcopy(self.model))
        bounded_model.load_params(fpath)
        bounded_model = bounded_model.to(self.device)
        intrinsic_robustness = torch.zeros_like(batch_labels)
        for data_idx, (datapoint, label) in enumerate(zip(batch_data, batch_labels)):
            datapoint, label = datapoint.unsqueeze(0), label.unsqueeze(0)
            is_cert_correct = agt.test_metrics.test_accuracy(bounded_model, datapoint, label)[0]
            intrinsic_robustness[data_idx] = int(is_cert_correct)
        # If I am certified to k_private points I have 1 in intrinsic_robustness, else 0. Hence multiply by k_private.
        intrinsic_robustness *= k_private
        logit_preds = bounded_model.forward(batch_data)
        preds = None
        if logit_preds.shape[1] == 1:  # binary classification
            preds = (torch.nn.functional.sigmoid(logit_preds) > 0.5).long()
        else:  # multi-class classification
            preds = torch.argmax(logit_preds, dim=1)
        # logger.info(f"preds[:10]: {preds[:10]}, labels[:10]: {batch_labels[:10]}")
        # logger.info(f"test acc certified_predictions[:10]: {certified_predictions[:10]}")
        # logger.info(f"preds shape: {logit_preds.shape}, labels shape: {batch_labels.shape}")

        return intrinsic_robustness.to(self.device, torch.int64), preds.to(self.device, torch.int64)

    def setup_logging_and_saving(self, save_kwargs: dict) -> None:
        """
        Set up logging and saving for the certifier.
        """
        self.save, self.load, self.logfile, self.result_file = False, False, None, None
        self.root_dir = f"{self.__str__()}"
        self.suffix = f"epochs_{self.epochs}_batch_size_{self.batch_size}_lr_{self.lr}"
        self.save = "save" in save_kwargs and save_kwargs["save"]
        self.load = "load" in save_kwargs and save_kwargs["load"]
        self.logfile_name = "temp"
        if "logfile_name" in save_kwargs and save_kwargs["logfile_name"]:
            self.logfile_name = save_kwargs["logfile_name"]
        if "write_to_file" in save_kwargs and save_kwargs["write_to_file"]:
            self.result_file = self.__str__() + ".yaml"
        self.save_load_dir = os.path.join(self.root_dir, "agt", self.suffix)

    def _setup_save_load_dir(self, logfile: str = None, new_save_load_dir: str = None) -> None:
        if new_save_load_dir is not None:
            self.save_load_dir = new_save_load_dir
        if logfile is not None:
            self.logfile = logfile
            return
        # Log everytime -- either to temp or to a specific file
        self.logfile = get_logfile_path(os.path.join(self.root_dir, "agt", "logs", self.logfile_name + ".log"))
        if not os.path.exists(os.path.dirname(self.logfile)):
            os.makedirs(os.path.dirname(self.logfile))
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
        logger.remove()
        logger.add(self.logfile, level="INFO", format=log_format, colorize=False, backtrace=True, diagnose=True)

    def get_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def update_train_set(self, train_set: Dataset, batch_size: int) -> None:
        """
        Update the training set and the corresponding DataLoader.
        """
        self.train_set = train_set
        self.train_loader = self.get_dataloader(self.train_set, batch_size)

    @abstractmethod
    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        pass

    @abstractmethod
    def make_model(self) -> Generic_NN:
        pass


class MnistCertifier(AGTCertifier):
    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None):
        self.num_classes = 10
        super().__init__(hyperparams, device, save_kwargs=save_kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        return VanillaMNIST(dset_type, self.seed, flatten=self.flatten)

    def make_model(self) -> Generic_NN:
        return torch.nn.Sequential(torch.nn.Linear(784, 32, bias=False), torch.nn.ReLU(), torch.nn.Linear(32, 10, bias=False))

    def __str__(self):
        return "mnist"


class HalfmoonsCertifier(AGTCertifier):
    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None):
        self.num_classes = 2
        super().__init__(hyperparams, device, save_kwargs=save_kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        return Halfmoons(dset_type, num_samples_train_test=(20000, 2500), noise=0.1)

    def make_model(self) -> Generic_NN:
        return torch.nn.Sequential(torch.nn.Linear(2, 32, bias=False), torch.nn.ReLU(), torch.nn.Linear(32, 1, bias=False))

    def __str__(self):
        return "halfmoons"


class BlobsCertifier(AGTCertifier):
    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None):
        self.num_classes = 2
        super().__init__(hyperparams, device, save_kwargs=save_kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        return Blobs(dset_type, num_samples_train_test=(20000, 2500))

    def make_model(self) -> Generic_NN:
        return torch.nn.Sequential(torch.nn.Linear(2, 10, bias=False), torch.nn.ReLU(), torch.nn.Linear(10, 1, bias=False))

    def __str__(self):
        return "blobs"


class Cifar10Certifier(AGTCertifier):
    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None):
        self.num_classes = 10
        super().__init__(hyperparams, device, save_kwargs=save_kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        return CIFAR(dset_type, self.seed)

    def make_model(self) -> Generic_NN:
        return Resnet18(output_dim=10)

    def __str__(self):
        return "cifar10"


def check_hyperparams_present(hyperparams: dict, save_kwargs: dict) -> bool:
    assert "epochs" in hyperparams, "Hyperparameters must contain epochs"
    assert "batch_size" in hyperparams, "Hyperparameters must contain batch_size"
    assert "lr" in hyperparams, "Hyperparameters must contain lr"
    if "save" in save_kwargs:
        assert "load" not in save_kwargs, "Cannot specify both save and load"
    if "load" in save_kwargs:
        assert "save" not in save_kwargs, "Cannot specify both save and load"
        assert "save" not in save_kwargs, "Cannot specify both save and load"
