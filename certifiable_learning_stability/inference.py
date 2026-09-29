import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models.generic_nn import Generic_NN
from .threats import Threats


def accuracy(model: Generic_NN, loader: DataLoader, device: torch.device, num_classes: int = None) -> float:
    """
    Get the accuracy of the model on the data.
    """
    with torch.no_grad():
        out = torch.tensor([], dtype=torch.float32, device=device)
        labels = torch.tensor([], dtype=torch.float32, device=device)
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            out = torch.cat((out, model(images)), dim=0)
            labels = torch.cat((labels, targets), dim=0)
        assert (
            hasattr(model, "num_classes") or num_classes is not None
        ), "Either the model must have num_classes attribute or num_classes must be provided"
        multiclass = model.num_classes > 2 if num_classes is None else num_classes > 2
        if multiclass:
            _, predicted = torch.max(out, 1)
        else:
            predicted = (out > 0).float()
        correct = (predicted == labels).float()
        acc = correct.sum() / len(labels)

    return acc.item()


@torch.no_grad()
def get_prediction(
    model: Generic_NN, data: torch.Tensor, device: torch.device, with_logits: bool = False, with_softmax: bool = False
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    Get the prediction of the model on the data.
    """
    if with_softmax:
        assert with_logits is True, "with_softmax must be True if with_logits is True"
    model.eval()
    model = model.to(device)
    data = data.to(device)
    out = model(data)
    preds, logits, act = None, None, None
    if model.num_classes > 2:
        preds = torch.softmax(out, dim=1).argmax(dim=1)
        logits = out
        act = lambda x: torch.softmax(x, dim=1)
    else:
        preds = out > 0
        # The logit needs to be returned in the format: torch.Shape([batch_size, num_classes])
        # Hence the need to handle binary classification separately.
        if preds == 1:
            logits = torch.stack([-out, out], dim=1)
        else:
            logits = torch.stack([out, -out], dim=1)
        act = lambda x: torch.sigmoid(x)

    if with_logits:
        if with_softmax:
            logits = act(logits)
        return preds.to(dtype=torch.int64), logits.to(dtype=torch.float32)
    else:
        return preds.to(dtype=torch.int64)


@torch.no_grad()
def aggregate_predictions_batch(
    model: Generic_NN,
    state_dict_load_func: callable,
    num_models: int,
    batch_data: torch.Tensor,
    batch_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate predictions and softmax logits for RDP certification from the model over a batch of data.
    """
    model.eval()
    model = model.to(device)
    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
    batch_size = batch_data.shape[0]
    agg_multinomial = np.zeros((batch_size, num_classes + 1), dtype=np.int64)
    agg_softmax = np.zeros((num_models, batch_size, num_classes + 1), dtype=np.float32)

    dim_indices = np.arange(0, batch_size)

    for model_idx in range(num_models):
        curr_state_dict = state_dict_load_func(model_idx)
        model.load_state_dict(curr_state_dict, strict=True)
        preds, softmax_logits = get_prediction(model, batch_data, device, with_logits=True, with_softmax=True)
        agg_multinomial[dim_indices, preds.detach().clone().cpu().numpy()] += 1
        agg_softmax[model_idx, dim_indices, :-1] = softmax_logits.detach().clone().cpu().numpy()

    agg_multinomial[dim_indices, -1] = batch_labels.cpu().numpy()
    agg_softmax[:, dim_indices, -1] = batch_labels.cpu().numpy()

    return agg_multinomial, agg_softmax


@torch.no_grad()
def aggregate_predictions(
    model: Generic_NN, loader: DataLoader, device: torch.device, agg_pred: np.ndarray, agg_softmax: np.ndarray, model_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate predictions and softmax logits from the model over the entire dataset.
    Return a tensor containing the predictions and a tensor containing the softmax logits.
    """
    test_preds, softmax_logits = zip(*[get_prediction(model, data, device, with_logits=True, with_softmax=True) for data, _ in loader])
    test_preds, softmax_logits = torch.cat(list(test_preds), dim=0), torch.cat(list(softmax_logits), dim=0)

    agg_pred[np.arange(0, len(test_preds)), test_preds.detach().clone().cpu().numpy()] += 1
    agg_softmax[model_idx, np.arange(0, len(test_preds)), :-1] = softmax_logits.detach().clone().cpu().numpy()

    return agg_pred, agg_softmax


def get_certified_accuracy_for_given_bounds(model: Generic_NN, loader: DataLoader, bounds: torch.Tensor, device: torch.device) -> float:
    """
    Get the certified accuracy of the model on the data.
    """
    model.eval()
    bounds = bounds.to(device)
    certified_points, total_points = 0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits_l, logits_u = model.propagate_param_intervals_forward(images, bounds)
            out = model.certify_points(logits_l, logits_u, targets)
            certified_points += torch.sum(out)
            total_points += len(images)

    return certified_points.item() / total_points


def aggregate_robustness_radii_to_dict(all_rob_radii: torch.Tensor) -> dict:
    percentage_robustness_radii = {}
    num_datapoints = all_rob_radii.numel()
    for rob_radius in torch.unique(all_rob_radii):
        larger_rob = all_rob_radii >= rob_radius
        percentage_robustness_radii[int(rob_radius)] = all_rob_radii[larger_rob].numel() / num_datapoints

    return percentage_robustness_radii
