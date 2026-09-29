import argparse
import gc
import os
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import torch
from loguru import logger
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.data_loader import UniformWithReplacementSampler
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from opacus.validators.errors import UnsupportedModuleError
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from data_sets.cifar import CIFAR
from data_sets.dset_type import DsetType
from data_sets.halfmoons import Halfmoons
from data_sets.mnist import VanillaMNIST
from experiments.save_utils import (
    get_logfile_path,
    load_model_state_dict,
    save_model_state_dict,
    write_results_to_file,
)

from .bounds import tightest_interval_for_hit_proportion
from .inference import (
    accuracy,
    aggregate_predictions,
    aggregate_robustness_radii_to_dict,
    get_certified_accuracy_for_given_bounds,
)
from .models.fcn import FCN
from .models.generic_nn import Generic_NN
from .models.resnet import Resnet18, Resnet18Finetune
from .rdp_certify_utils import (
    DEFAULT_ALPHAS,
    DEFAULT_DELTA,
    CertifyRadiusDP,
    CertifyRadiusDPBS,
    CertifyRadiusDPBS_softmax_prob,
    CertifyRadiusRDP,
    aggres_meta_info,
    confident_interval_multinomial,
    confident_interval_softmax,
    gen_sub_dataset,
)


class StabilityCertifierWithRDP(ABC):
    """
    Abstract base class for stability certifiers that use RDP (Renyi Differential Privacy). All subclasses must implement the following methods:

    - `get_original_dset`: Returns the original dataset for the given DsetType.
    - `make_model`: Returns the model (needs to be a torch.nn.Module) to be trained and certified.
    - `make_optimizer`: Returns the optimizer for the model.
    See [Improved Pointwise Certifications against Poisoning Attacks](https://arxiv.org/pdf/2308.07553)
    """

    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None) -> None:
        """
        Initialize the RDP stability certifier with hyperparameters, device, and saving/loading options.

        :param dict hyperparams: Dictionary containing hyperparameters for training.
            Hyperparams needed:
            - epochs: Number of training epochs. (required)
            - batch_size: Size of the training batches. (required)
            - lr: Learning rate for the optimizer. (required)
            - mechanism_samples: Number of samples for the RDP mechanism. (required)
            - confidence: Confidence level for the RDP mechanism. (required)
            - seed: Random seed for reproducibility. (required)
            - sigma: Noise standard deviation for the RDP mechanism. (required)
            - max_grad_norm: Maximum gradient norm for clipping. (required)
        :param torch.device device: The device (CPU or GPU) on which the model will be trained.
        :param dict save_kwargs: Dictionary containing options for saving and loading the model.
            - save: Boolean indicating whether to save the model. (optional, default: False)
            - load: Boolean indicating whether to load a pre-trained model. (optional, default: False)
            Note: You cannot specify both `save` and `load` as True at the same time.
            - logfile_name: Name of the log file to write logs to. (optional, default: "temp")
            - write_to_file: Boolean indicating whether to write results to a file. (optional, default: False)
        """
        check_hyperparams_present(hyperparams, save_kwargs)
        self.all_hyperparams = hyperparams  # Save all hyperparameters for future reference
        self.epochs = hyperparams["epochs"]
        self.rdp_sample_rate = hyperparams["sample_rate"]
        self.lr = hyperparams["lr"]
        self.mechanism_samples = hyperparams["mechanism_samples"]
        self.confidence = hyperparams["confidence"]
        self.seed = hyperparams["seed"]
        self.noise_std = hyperparams["sigma"]  # noise standard deviation
        self.max_grad_norm = hyperparams["max_grad_norm"]  # max clip magnitude
        self.sub_training_size = hyperparams["sub_training_size"]
        self.weight_decay = hyperparams.get("weight_decay", 0.0)
        # % Set up DP-specific hyperparameters
        self.target_dp_delta = DEFAULT_DELTA
        self.noise_multiplier = self.noise_std / self.max_grad_norm
        self.dp_epsilon, self.best_alpha = None, None
        # % Set up dataset, model and device
        self.device = device
        self.full_train_set = self.get_original_dset(DsetType.TRAIN_FULL)
        self.full_train_set_size = len(self.full_train_set)
        self.batch_size = int(self.rdp_sample_rate * self.sub_training_size)
        self.steps = self.epochs * (self.sub_training_size // self.batch_size)
        self.test_set = self.get_original_dset(DsetType.TEST)
        self.test_loader = self.get_dataloader(self.test_set, self.batch_size, shuffle=False)
        self.model = self.make_model().to(self.device)

        self.aggregate_result, self.aggregate_result_softmax = None, None

        self.setup_logging_and_saving(save_kwargs)

    def certify_params(self, hit_ratios: list[float], return_models: bool = False, logfile: str = None) -> dict | tuple[dict, torch.Tensor]:
        """
        Certify the model parameters using RDP and compute the certified accuracy for given hit ratios using IBP.

        :param list[float] hit_ratios:
            List of hit ratios, i.e. the percentage of parameters that fit in a given interval for which to compute the certified accuracy.
        :param bool, optional return_models:
            If True, return the trained models' parameters as well. Default is False. This will be used when called from some (partition) aggregation method.
        :param str, optional logfile:
            Path to the log file to write logs to. If None, the logfile will either be the save_kwargs["logfile_name"] or "temp". If it is not None,
            please provide the full path. For the latter, this will again be used by (partition) aggregation methods to log the results to the same file.

        :return:
            A dictionary containing the certified accuracies for the given hit ratios and the certified radii for the parameters.
            If `return_models` is True, also returns the trained models' parameters as a torch.Tensor.
        """
        assert isinstance(hit_ratios, list) or hit_ratios is None, "hit_ratios must be a list of floats or None"
        self.dp_epsilon, self.best_alpha = None, None
        self.aggregate_result = np.zeros((len(hit_ratios), 3), dtype=np.int32)
        self._setup_save_load_dir(logfile=logfile)
        self.model.train()
        self.model = validate_and_fix_model(self.model, self.device)

        param_samples, all_param_samples = [], []
        for m_sample in range(self.mechanism_samples):
            logger.info(f"DP training with sample {m_sample + 1}/{self.mechanism_samples}")
            trainable_params, interval_params = self._dp_train_loop(m_sample)
            param_samples.append(interval_params)
            if return_models:
                all_param_samples.append(trainable_params.cpu())
        param_samples = torch.stack(param_samples, dim=0)

        tightest_intervals = {}
        for hr_idx, hit_ratio in enumerate(hit_ratios):
            ti = tightest_interval_for_hit_proportion(param_samples, hit_ratio)
            tightest_intervals[hit_ratio] = ti
            inside_cnt = int(hit_ratio * self.mechanism_samples)
            self.aggregate_result[hr_idx, 0] = inside_cnt
            self.aggregate_result[hr_idx, 1] = self.mechanism_samples - inside_cnt

        certified_radii_for_intervals = self.hybrid_robustness(hit_ratios)

        # Compute the certified accuracy as well
        certified_accs = {
            k: get_certified_accuracy_for_given_bounds(self.model, self.test_loader, ti, self.device) for k, ti in tightest_intervals.items()
        }
        logger.info(f"Certified accs for hit ratios are: {certified_accs}")
        interval_sizes = {k: (ti[:, 1] - ti[:, 0]).cpu().numpy() for k, ti in tightest_intervals.items()}

        if self.result_file is not None:
            rf, ext = os.path.splitext(self.result_file)
            results_file = rf + "_rdp_hybrid" + ext
            params_file = rf + "_rdp_hybrid_params" + ext
            write_results_to_file(
                results_file,
                {
                    "certified_accs": {k: round(float(ca), 5) for k, ca in certified_accs.items()},
                    "min_certified_radius": {
                        k: cr if hit_ratios is not None else round(float(cr.min()), 7) for k, cr in certified_radii_for_intervals.items()
                    },
                    "max_certified_radius": {
                        k: cr if hit_ratios is not None else round(float(cr.max()), 7) for k, cr in certified_radii_for_intervals.items()
                    },
                    "min interval": {k: round(float(isize.min()), 5) for k, isize in interval_sizes.items()},
                    "max interval": {k: round(float(isize.max()), 5) for k, isize in interval_sizes.items()},
                },
                self.category,
            )
            params = deepcopy(self.all_hyperparams)
            write_results_to_file(params_file, params, self.category)

        if return_models:
            cert_data = {hit_ratio: (certified_accs[hit_ratio], certified_radii_for_intervals[hit_ratio]) for hit_ratio in hit_ratios}
            all_param_samples = torch.stack(all_param_samples, dim=0)
            return cert_data, all_param_samples
        return certified_radii_for_intervals

    def certify_points(self, method_name: str = "dp_bagging_softmax_prob", external_save_func: callable = None, logfile: str = None) -> dict | None:
        """
        Certify the poisoning resilience of the model using pointwise RDP certification.

        :param bool, optional return_models:
            If True, return the trained models' parameters as well. Default is False. This will be used when called from some (partition) aggregation method.
        :param str, optional logfile:
            Path to the log file to write logs to. If None, the logfile will either be the save_kwargs["logfile_name"] or "temp". If it is not None,
            please provide the full path. For the latter, this will again be used by (partition) aggregation methods to log the results to the same file.

        :return:
            A dictionary containing the certified accuracies for the given certified radii.
            If `return_models` is True, also returns the trained models' parameters as a torch.Tensor.
        """
        self._setup_save_load_dir(rdp_type="rdp_pointwise", logfile=logfile)
        self.dp_epsilon, self.best_alpha = None, None
        self.aggregate_result = np.zeros((len(self.test_set), self.num_classes + 1), dtype=np.float32)
        self.aggregate_result_softmax = np.zeros((self.mechanism_samples, len(self.test_set), self.num_classes + 1), dtype=np.float32)
        self.model.train()
        self.model = validate_and_fix_model(self.model, self.device)

        for m_sample in range(self.mechanism_samples):
            logger.info(f"DP training with sample {m_sample + 1}/{self.mechanism_samples}")
            m_sample_model_state_dict = self._dp_train_loop(m_sample)
            if external_save_func is not None:
                external_save_func(m_sample_model_state_dict, m_sample)
        correct_labels = next(iter(self.get_dataloader(self.test_set, len(self.test_set), shuffle=False)))[1].cpu().numpy()
        self.aggregate_result[np.arange(0, len(self.test_set)), -1] = correct_labels
        self.aggregate_result_softmax[:, np.arange(0, len(self.test_set)), -1] = correct_labels

        # If this is called by the DPA certifier, return early
        if external_save_func is not None:
            return

        clean_acc = self.ptwise_intrinsic_robustness(method_name, self.aggregate_result, self.aggregate_result_softmax, clean_acc_only=True)
        certified_radii = self.ptwise_intrinsic_robustness(method_name, self.aggregate_result, self.aggregate_result_softmax)
        cert_acc_per_radius = aggregate_robustness_radii_to_dict(torch.from_numpy(certified_radii))
        logger.info(f"Certified radii: {certified_radii}")
        logger.info(f"Certified accuracies per radius: {cert_acc_per_radius}")

        if self.result_file is not None:
            rf, ext = os.path.splitext(self.result_file)
            results_file = rf + "_rdp_pointwise" + ext
            params_file = rf + "_rdp_pointwise_params" + ext
            write_results_to_file(
                results_file,
                {
                    "clean_acc": round(float(clean_acc), 6),
                    "certified_accs": {k: round(float(ca), 5) for k, ca in cert_acc_per_radius.items()},
                    "min_certified_radius": int(certified_radii.min()),
                    "max_certified_radius": int(certified_radii.max()),
                },
                self.category,
            )
            params = deepcopy(self.all_hyperparams)
            write_results_to_file(params_file, params, self.category)

        return cert_acc_per_radius

    def get_args_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(
            alpha=(1 - self.confidence),
            sample_rate=self.rdp_sample_rate,
            epochs=self.epochs,
            n_runs=self.mechanism_samples,
            lr=self.lr,
            sigma=self.noise_std,
            max_per_sample_grad_norm=self.max_grad_norm,
            training_size=self.full_train_set_size,
            train_mode="Sub-DP",
            radius_range=55,
            sub_training_size=self.sub_training_size,
        )

    @torch.no_grad()
    def hybrid_robustness(self, hit_ratios: list[float]) -> dict[int, float]:
        args = self.get_args_namespace()
        certified_poisoning_size_array = {}

        for hr_idx in range(len(hit_ratios)):
            confidence_interval, labels = confident_interval_multinomial(self.aggregate_result.astype(np.int64), hr_idx, "best", float(args.alpha))
            logger.info(f"Confidence interval for hit ratio {hit_ratios[hr_idx]}: {confidence_interval}")
            # ! "BEST" setup
            best_radius_dp = CertifyRadiusDP(args, labels, confidence_interval, self.dp_epsilon, self.target_dp_delta)
            best_radius_rdp = CertifyRadiusRDP(args, labels, confidence_interval, self.steps, self.rdp_sample_rate, self.noise_std)
            logger.info(f"Best radius for DP: {best_radius_dp}, Best radius for RDP: {best_radius_rdp}")
            optimal_radius = max(best_radius_dp, best_radius_rdp)

            certified_poisoning_size_array[hr_idx] = optimal_radius

        return certified_poisoning_size_array

    @torch.no_grad()
    def ptwise_intrinsic_robustness(
        self,
        method_name: str,
        aggregate_result: np.ndarray,
        aggregate_result_softmax: np.ndarray,
        clean_acc_only: bool = False,
        ignore_label: bool = False,
    ) -> np.ndarray | float:
        args = self.get_args_namespace()
        assert (
            aggregate_result.ndim == 2 and aggregate_result_softmax.ndim == 3
        ), "aggregate_result must be 2D and aggregate_result_softmax must be 3D"
        num_data = len(aggregate_result)
        mean_agg, confident_func, votes_per_class = None, None, None
        if "softmax" not in method_name:
            mean_agg, votes_per_class = aggregate_result, aggregate_result[:, :-1]
            confident_func = lambda datapoint_idx: confident_interval_multinomial(
                aggregate_result.astype(np.int64), datapoint_idx, method_name, float(args.alpha)
            )
        else:
            mean_agg = aggregate_result_softmax.mean(axis=0)
            predicted_classes = torch.from_numpy(aggregate_result_softmax[:, :, :-1]).argmax(dim=2).to(dtype=torch.int64)
            predicted_classes_one_hot = torch.nn.functional.one_hot(predicted_classes, num_classes=self.num_classes)
            votes_per_class = predicted_classes_one_hot.sum(dim=0).cpu().numpy()
            assert votes_per_class.shape == (num_data, self.num_classes)
            confident_func = lambda datapoint_idx: confident_interval_softmax(
                aggregate_result_softmax, mean_agg, datapoint_idx, method_name, float(args.alpha)
            )[:2]

        gt, pred = aggres_meta_info(mean_agg)
        clean_acc = float((gt == pred).sum() / len(pred))
        logger.info(f"Clean accuracy: {clean_acc:.5f}")
        if clean_acc_only:
            return clean_acc
        certified_poisoning_size_array = np.zeros([num_data], dtype=np.int32)

        for datapoint_idx in range(num_data):
            confidence_interval, labels = confident_func(datapoint_idx)
            # logger.info(f"Confidence interval for datapoint {datapoint_idx}: {confidence_interval}")
            match method_name:
                case "best":
                    best_radius_dp = CertifyRadiusDP(args, labels, confidence_interval, self.dp_epsilon, self.target_dp_delta)
                    best_radius_rdp = CertifyRadiusRDP(args, labels, confidence_interval, self.steps, self.rdp_sample_rate, self.noise_std)
                    optimal_radius = max(best_radius_dp, best_radius_rdp)
                case "rdp_softmax":
                    optimal_radius = CertifyRadiusRDP(
                        args, labels, confidence_interval, self.steps, self.rdp_sample_rate, self.noise_std, softmax=True
                    )
                case "dp_bagging":
                    optimal_radius, _ = CertifyRadiusDPBS(
                        args,
                        labels,
                        confidence_interval,
                        args.sub_training_size,
                        args.training_size,
                        self.dp_epsilon,
                        self.target_dp_delta,
                        self.steps,
                        args.sample_rate,
                        args.sigma,
                    )
                case "dp_bagging_softmax":
                    optimal_radius, _ = CertifyRadiusDPBS(
                        args,
                        labels,
                        confidence_interval,
                        args.sub_training_size,
                        args.training_size,
                        self.dp_epsilon,
                        self.target_dp_delta,
                        self.steps,
                        args.sample_rate,
                        args.sigma,
                        softmax=True,
                    )
                case "dp_bagging_softmax_prob":
                    optimal_radius = CertifyRadiusDPBS_softmax_prob(
                        labels,
                        confidence_interval,
                        args.sub_training_size,
                        args.training_size,
                        self.target_dp_delta,
                        self.rdp_sample_rate,
                        args.sample_rate,
                        args.sigma,
                        upper=self.mechanism_samples / 2,
                        ignore_label=ignore_label,
                    )
                    if optimal_radius == self.mechanism_samples / 2:
                        # Set it to the aggregation margin
                        votes_per_class_curr = torch.from_numpy(votes_per_class[datapoint_idx])
                        # Get top 2 predictions
                        top_2 = torch.topk(votes_per_class_curr, 2)
                        top_2_classes, top_2_votes = top_2.indices, top_2.values
                        [votes_pred, votes_sec] = list(top_2_votes)
                        [classes_pred, classes_sec] = list(top_2_classes)
                        # % We do not need to check that the label matches c_pred because the RDP framework already does it
                        optimal_radius = (votes_pred - votes_sec) // 2 + (classes_sec > classes_pred)
                        logger.info(f"Setting optimal radius to {optimal_radius} for datapoint {datapoint_idx} because it was equal to the MAX.")
                case _:
                    raise ValueError(f"Unknown method name: {method_name}")

            logger.info(f"Optimal radius for datapoint {datapoint_idx}: {optimal_radius}")
            certified_poisoning_size_array[datapoint_idx] = optimal_radius

        return certified_poisoning_size_array

    def update_train_set(self, new_train_set: Dataset) -> None:
        """
        Update the training set and reinitialize the DataLoader.

        :param Dataset new_train_set: The new training dataset to use.
        :info This method is called from dpa_certifier.py and assumes that the correct sample rate with respect to the partition subset has been set
            at initialization.
        """
        self.full_train_set = new_train_set
        self.full_train_set_size = len(self.full_train_set)

    def _dp_train_loop(self, m_sample_idx: int) -> dict:
        """
        Wraps the original model using a PrivacyEngine, trains it with DP, and returns the model parameters after training.
        This method assumes that the model, optimizer, criterion, and other necessary parameters are already set up.
        It uses the Opacus library to handle differential privacy during training.

            :param int m_sample_idx: The index of the current sample for training. Used for saving/loading model state.

        Returns:
            torch.Tensor: The model parameters after training with DP in a flattened tensor format.
        """
        assert hasattr(self, "epochs"), "Number of epochs must be defined before training"
        assert hasattr(self, "model"), "Model must be defined before training"
        assert hasattr(self, "criterion"), "Criterion must be defined before training"
        assert hasattr(self, "num_classes"), "Number of classes must be defined before training"
        assert hasattr(self, "max_physical_batch_size"), "max_physical_batch_size must be defined before training"

        model = None
        if self.load:
            model = deepcopy(self.model)
            # Load the model state dict if specified
            model = load_model_state_dict(model, f"private_model_{m_sample_idx}.pt", self.device, self.save_load_dir)
            logger.info(f"Loaded model state dict from private_model_{m_sample_idx}.pt")
        else:
            # Set up DP training wrapper for Opacus
            train_set = gen_sub_dataset(self.full_train_set, self.sub_training_size, with_replacement=True)
            loader = DataLoader(
                train_set,
                batch_sampler=UniformWithReplacementSampler(
                    num_samples=self.sub_training_size, sample_rate=self.rdp_sample_rate, generator=torch.Generator().manual_seed(self.seed)
                ),
            )
            #! Unfortunately, for now we need to deepcopy to avoid the error given by attaching hooks multiple times
            #! This is fine, because by deepcopying, we do not change the state of the original model.
            model_copy = deepcopy(self.model)
            optimizer_copy = self.make_optimizer(model_copy)
            privacy_engine = PrivacyEngine(accountant=RDPAccountant.mechanism())
            model, optimizer, loader = privacy_engine.make_private(
                module=model_copy,
                optimizer=optimizer_copy,
                criterion=self.criterion,
                data_loader=loader,
                noise_multiplier=self.noise_multiplier,
                target_delta=self.target_dp_delta,
                max_grad_norm=self.max_grad_norm,
            )

            # Set up train utils
            is_binary_classification = not isinstance(self.criterion, torch.nn.CrossEntropyLoss)
            batch_acc = lambda preds, labels: (preds == labels).float().mean().item()
            compute_preds = lambda out: torch.sigmoid(out).round().int() if is_binary_classification else torch.argmax(out, dim=1).int()
            progress_bar = trange(
                self.epochs,
                desc="Epoch",
            )
            losses, top1_accs = [], []

            # DP train loop
            with BatchMemoryManager(data_loader=loader, max_physical_batch_size=self.batch_size, optimizer=optimizer) as memory_safe_data_loader:
                for epoch in progress_bar:
                    for i, (images, labels) in enumerate(memory_safe_data_loader):
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        loss = self.criterion(outputs, labels)
                        preds = compute_preds(outputs)
                        acc = batch_acc(preds, labels.int())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())
                        top1_accs.append(acc)
                        progress_bar.set_postfix({"Epoch": epoch + 1, "Batch": i + 1, "Loss": loss.item(), "Batch Accuracy": acc})
                if self.save:
                    save_model_state_dict(model._module, f"private_model_{m_sample_idx}.pt", self.save_load_dir)
            del loader, optimizer, train_set, model_copy, optimizer_copy

            if m_sample_idx == 0:
                self.dp_epsilon, self.best_alpha = privacy_engine.accountant.get_privacy_spent(delta=self.target_dp_delta, alphas=DEFAULT_ALPHAS)
            model = model._module  # Unwrap

        inference_accuracy_score = accuracy(model, self.test_loader, self.device, self.num_classes)
        self.aggregate_result, self.aggregate_result_softmax = aggregate_predictions(
            model, self.test_loader, self.device, self.aggregate_result, self.aggregate_result_softmax, m_sample_idx
        )
        logger.info(f"Inference accuracy after DP training: {inference_accuracy_score:.4f}")
        state_dict = deepcopy(model.state_dict())
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return state_dict

    def setup_logging_and_saving(self, save_kwargs: dict) -> None:
        """
        Set up logging and saving for the certifier.
        """
        self.save, self.load, self.logfile, self.result_file, self.category = False, False, None, None, None
        self.root_dir = f"{self.__str__()}"
        self.suffix = (
            f"sigma_{self.noise_std}_"
            + f"clip_{self.max_grad_norm}_"
            + f"q_{round(self.rdp_sample_rate, 4)}_"
            + f"n_{self.mechanism_samples}_"
            + f"batch_{self.batch_size}_"
            + f"epochs_{self.epochs}_"
            + f"sts_{self.sub_training_size}"
        )
        self.category = self.suffix
        self.save = "save" in save_kwargs and save_kwargs["save"]
        self.load = "load" in save_kwargs and save_kwargs["load"]
        self.logfile_name = "temp"
        if "logfile_name" in save_kwargs and save_kwargs["logfile_name"]:
            self.logfile_name = save_kwargs["logfile_name"]
        if "write_to_file" in save_kwargs and save_kwargs["write_to_file"]:
            self.result_file = self.__str__() + ".yaml"

    def _setup_save_load_dir(self, rdp_type: str = "rdp_hybrid", logfile: str = None) -> None:
        if logfile is not None:
            self.logfile = logfile
            return
        self.save_load_dir = os.path.join(self.root_dir, rdp_type, self.suffix)
        # Log everytime -- either to temp or to a specific file
        self.logfile = get_logfile_path(os.path.join(self.root_dir, "rdp", "logs", self.logfile_name + ".log"))
        if not os.path.exists(os.path.dirname(self.logfile)):
            os.makedirs(os.path.dirname(self.logfile))
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
        logger.remove()
        logger.add(self.logfile, level="INFO", format=log_format, colorize=False, backtrace=True, diagnose=True)

    def get_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def make_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.StepLR:
        assert hasattr(self, "lr_decay"), "Learning rate decay must be defined before creating the scheduler"
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay)

    @abstractmethod
    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        pass

    @abstractmethod
    def make_model(self) -> Generic_NN:
        pass

    @abstractmethod
    def make_optimizer(self, model: Generic_NN) -> torch.optim.Optimizer:
        pass


class HalfmoonsCertifier(StabilityCertifierWithRDP):
    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None):
        self.num_classes = 2
        super().__init__(hyperparams, device, save_kwargs=save_kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.max_physical_batch_size = 10000

    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        return Halfmoons(dset_type, num_samples_train_test=(4000, 800), noise=0.1)

    def make_model(self) -> Generic_NN:
        input_dim, output_dim, hidden_dims = 2, 1, [10]
        return FCN(input_dim, output_dim, hidden_dims, with_bias=False).to(self.device)

    def make_optimizer(self, model: Generic_NN) -> torch.optim.Optimizer:
        assert hasattr(self, "lr"), "Learning rate must be defined before creating the optimizer"
        return torch.optim.Adam(model.parameters(), self.lr, weight_decay=self.weight_decay)

    def __str__(self):
        return "halfmoons"


class MnistCertifier(StabilityCertifierWithRDP):
    def __init__(self, hyperparams: dict, device: torch.device, save_kwargs: dict = None):
        self.num_classes = 10
        super().__init__(hyperparams, device, save_kwargs=save_kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.max_physical_batch_size = 10000

    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        return VanillaMNIST(dset_type, self.seed)

    def make_model(self) -> Generic_NN:
        input_dim, output_dim, hidden_dims = 784, 10, [32]
        return FCN(input_dim, output_dim, hidden_dims).to(self.device)

    def make_optimizer(self, model: Generic_NN) -> torch.optim.Optimizer:
        assert hasattr(self, "lr"), "Learning rate must be defined before creating the optimizer"
        return torch.optim.Adam(model.parameters(), self.lr, weight_decay=self.weight_decay)

    def __str__(self):
        return "mnist"


class Cifar10Certifier(StabilityCertifierWithRDP):
    def __init__(
        self, hyperparams: dict, device: torch.device, save_kwargs: dict = None, pretrained: bool = False, freeze_resnet_blocks: bool = False
    ):
        self.num_classes = 10
        self.pre_trained = pretrained
        self.freeze_resnet_blocks = freeze_resnet_blocks
        super().__init__(hyperparams, device, save_kwargs=save_kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.max_physical_batch_size = 10000

    def get_original_dset(self, dset_type: DsetType) -> Dataset:
        return CIFAR(dset_type, self.seed)

    def make_model(self) -> Generic_NN:
        model = None
        if self.pre_trained:
            hidden_layer_sizes, output_dim = [], 10
            model = Resnet18Finetune(hidden_layer_sizes, output_dim)
            if self.freeze_resnet_blocks:
                model.freeze_pretrained_blocks()
        else:
            model = Resnet18(output_dim=10)

        return model

    def make_optimizer(self, model: Generic_NN) -> torch.optim.Optimizer:
        assert hasattr(self, "lr"), "Learning rate must be defined before creating the optimizer"
        if self.pre_trained:
            return torch.optim.Adam(model.parameters(), self.lr, weight_decay=self.weight_decay)
        else:
            return torch.optim.SGD(model.parameters(), self.lr, momentum=0.9, weight_decay=0)

    def __str__(self):
        return "cifar10"


def validate_and_fix_model(model: Generic_NN, device: torch.device) -> Generic_NN:
    # Validate whether the model layers' are compatible with Opacus
    try:
        ModuleValidator.validate(model, strict=True)
    except UnsupportedModuleError as e:
        logger.error(f"Warning: unsupported module detected: {e}. Automatic fixing enabled.")
        model = ModuleValidator.fix(model)
        model = model.to(device)
        ModuleValidator.validate(model, strict=True)

    return model


def check_hyperparams_present(hyperparams: dict, save_kwargs: dict) -> bool:
    assert "epochs" in hyperparams, "Hyperparameters must contain epochs"
    assert "sample_rate" in hyperparams, "Hyperparameters must contain sample_rate"
    assert "sub_training_size" in hyperparams, "Hyperparameters must contain sub_training_size"
    assert "lr" in hyperparams, "Hyperparameters must contain lr"
    assert "mechanism_samples" in hyperparams, "Hyperparameters must contain mechanism_samples"
    assert "confidence" in hyperparams, "Hyperparameters must contain confidence"
    assert "seed" in hyperparams, "Hyperparameters must contain seed"
    assert "max_grad_norm" in hyperparams, "Hyperparameters must contain max_grad_norm (max clip magnitude)"
    assert "sigma" in hyperparams, "Hyperparameters must contain sigma (noise standard deviation)"
    if "save" in save_kwargs:
        assert "load" not in save_kwargs, "Cannot specify both save and load"
    if "load" in save_kwargs:
        assert "save" not in save_kwargs, "Cannot specify both save and load"
        assert "save" not in save_kwargs, "Cannot specify both save and load"
