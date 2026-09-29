import numpy as np
import torch
from loguru import logger

from .models.generic_nn import Generic_NN
from .threats import Threats


def tightest_interval_for_hit_proportion(params: torch.Tensor, hit_proportion: float) -> torch.Tensor:
    """
    Compute the tightest interval for the hit proportion.

    :param torch.Tensor params: The parameters of the model. Shape: (num_samples, num_params)
    :param float hit_proportion: The desired hit proportion as a percentage (in [0, 1]).

    Returns:
        torch.Tensor: A tensor containing the tightest lower and upper bounds of each parameter. Shape: (num_params, 2)
    """
    device = params.device
    # Ensure the hit_proportion is within the valid range
    assert 0 <= hit_proportion <= 1, "Hit proportion must be between 0 and 1."

    # Calculate the number of parameters
    num_samples = params.size(0)
    num_params = params.size(1)

    # Calculate the number of parameters that should be outside the interval
    num_inside_interval = int(round(num_samples * hit_proportion))
    if num_inside_interval == num_samples:
        # If all samples are inside the interval, return the min and max for each parameter
        return torch.stack((params.min(dim=0).values, params.max(dim=0).values), dim=1)
    if num_inside_interval == 0:
        # This doesn't make sense, as it means no samples are inside the interval
        # Thus, throw an error
        raise ValueError("Hit proportion is too low, resulting in no samples inside the interval.")
    num_outside_interval = num_samples - num_inside_interval

    # Sort the parameters to find the tightest interval
    sorted_params, _ = torch.sort(params, dim=0)

    tightest_bound_interval_size = torch.tensor(float("inf")).repeat(num_params).to(device)
    tightest_lb, tightest_ub = torch.zeros(num_params).to(device), torch.zeros(num_params).to(device)
    tightest_left_idx, tightest_right_idx = torch.zeros(num_params).to(device), torch.zeros(num_params).to(device)
    for i in range(0, num_outside_interval):
        # Calculate the lower and upper bounds
        lower_bound = sorted_params[i]
        upper_bound = sorted_params[i + num_inside_interval - 1]

        # Sanity check
        if torch.any(lower_bound > upper_bound):
            break

        interval_sizes = upper_bound - lower_bound
        rule = interval_sizes < tightest_bound_interval_size
        tightest_bound_interval_size[rule] = interval_sizes[rule]
        tightest_lb[rule], tightest_ub[rule] = lower_bound[rule], upper_bound[rule]
        tightest_left_idx[rule], tightest_right_idx[rule] = i, i + num_inside_interval - 1

    print(f"Hit proportion: {hit_proportion}, num_samples: {num_samples}, num_params: {num_params}")
    logger.info(f"Tightest interval for hit proportion {hit_proportion} is [{tightest_lb}, {tightest_ub}]")
    logger.info(f"Indices for tightest_bounds are [{tightest_left_idx}, {tightest_right_idx}]")

    # If the lower and upper bounds are the same, subtract & add a small delta to avoid any numerical issues
    min_delta = ((tightest_ub - tightest_lb) == 0) * 1e-5
    return torch.stack((tightest_lb - min_delta, tightest_ub + min_delta), dim=1)
