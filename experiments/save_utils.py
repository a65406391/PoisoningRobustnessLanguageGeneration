import os

import torch
import yaml


def get_result_dir_path():
    curr_dir = __file__.rsplit("/", 1)[0]
    results_dir = os.path.join(curr_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


def write_results_to_file(fname: str, results: dict, category: str) -> None:
    """
    Write the results to a file in the results directory.

    :param fname: The filename (.yaml) to write to in the results directory.
    :param results: The results to write to the file.
    :param category: The dictionary key to write to the yaml file.
    :return: None
    """
    assert fname.endswith(".yaml"), "Only yaml files supported"
    assert isinstance(results, dict), "Results must be a dict"
    results_dir = get_result_dir_path()
    file = os.path.join(results_dir, fname)
    if not os.path.exists(file):
        # Create a new file and write the dict to it
        with open(file, "w", encoding="utf-8") as f:
            yaml.dump({category: results}, f)
    else:
        # Load the existing yaml file, append to it and write it back
        new_results = None
        with open(file, "r", encoding="utf-8") as f:
            new_results = yaml.load(f, Loader=yaml.Loader) or {}
            new_results[category] = results
        with open(file, "w", encoding="utf-8") as f:
            yaml.dump(new_results, f)


def load_params_or_results_from_file(fname: str, category: str) -> dict:
    """
    Load the parameters or results from a file in the results directory.

    :param fname: The filename (.yaml) to load from in the results directory.
    :param category: The dictionary key to load from the yaml file.
    :return: The parameters or results as a dict.
    """
    assert fname.endswith(".yaml"), "Only yaml files supported"
    results_dir = get_result_dir_path()
    file = os.path.join(results_dir, fname)
    assert os.path.exists(file), "File does not exist"
    results = None
    with open(file, "r", encoding="utf-8") as f:
        results = yaml.load(f, Loader=yaml.Loader)

    return results[category] if results is not None else None


def save_model_state_dict(model_or_dict: torch.nn.Module | dict, fname: str, dir: str = None) -> None:
    """
    Save the model state dict to a file in the results directory.

    :param model: The model to save
    :param fname: The filename (.pt) to save the model to.
    :param dir: The directory under `results/` to save the model to. If None, the `results/` directory will be used.
    :return: None
    """
    assert fname.endswith(".pt"), "Only pt files supported"
    save_dir = get_result_dir_path()
    if dir:
        save_dir = os.path.join(save_dir, dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    file = os.path.join(save_dir, fname)
    state_dict = model_or_dict if isinstance(model_or_dict, dict) else model_or_dict.state_dict()
    torch.save(state_dict, file)


def get_logfile_path(fname: str) -> str:
    """
    Get the path to the log file in the results directory.

    :param fname: The filename (.log) to load the model from.
    :return: The path to the log file.
    """
    assert fname.endswith(".log"), "Only log files supported"
    results_dir = get_result_dir_path()
    return os.path.join(results_dir, fname)


def load_model_state_dict(model: torch.nn.Module, fname: str, device: torch.device, dir: str = None, weights_only: bool = False) -> torch.nn.Module:
    """
    Load the model state dict from a file in the results directory.

    :param model: The model to load the state dict into.
    :param fname: The filename (.pt) to load the model from.
    :param dir: The directory under `results/` to load the model from. If None, the `results/` directory will be used.
    :param weights_only: If True, only load the state dict. If False, load the entire model. Use the `with_bias` property of Generic_NN to set this parameter.
    :return: None
    Note: If `dir` does not exist, this method will raise an exception.
    """
    assert fname.endswith(".pt"), "Only pt files supported"
    save_dir = get_result_dir_path()
    if dir:
        save_dir = os.path.join(save_dir, dir)
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Directory {save_dir} does not exist")
    file = os.path.join(save_dir, fname)
    assert os.path.exists(file), "File does not exist"
    model.load_state_dict(torch.load(file, weights_only=weights_only, map_location=device))

    return model


def get_state_dict_from_file(fname: str, device: torch.device, dir: str = None) -> dict:
    """
    Load the state dict from a file in the results directory.

    :param fname: The filename (.pt) to load the model from.
    :param dir: The directory under `results/` to load the model from. If None, the `results/` directory will be used.
    :return: The state dict as a dict.
    Note: If `dir` does not exist, this method will raise an exception.
    """
    assert fname.endswith(".pt"), "Only pt files supported"
    save_dir = get_result_dir_path()
    if dir:
        save_dir = os.path.join(save_dir, dir)
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Directory {save_dir} does not exist")
    file = os.path.join(save_dir, fname)
    assert os.path.exists(file), "File does not exist"
    return torch.load(file, map_location=device)


def torchsave(data: object, fname: str, dir: str = None) -> None:
    """
    Save the tensor to a file in the results directory.

    :param data: The tensor to save. It can be a torch.Tensor or any object that can be saved by torch.save.
    :param fname: The filename (.pt) to save the model to.
    :param dir: The directory under `results/` to save the model to. If None, the `results/` directory will be used.
    :return: None
    """
    assert fname.endswith(".pt"), "Only pt files supported"
    save_dir = get_result_dir_path()
    if dir:
        save_dir = os.path.join(save_dir, dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    file = os.path.join(save_dir, fname)
    torch.save(data, file)


def torchload(fname: str, dir: str = None) -> object:
    """
    Load the tensor from a file in the results directory and return it.

    :param fname: The filename (.pt) to load the model from.
    :param dir: The directory under `results/` to load the model from. If None, the `results/` directory will be used.
    :return: None
    Note: If `dir` does not exist, this method will raise an exception.
    """
    assert fname.endswith(".pt"), "Only pt files supported"
    save_dir = get_result_dir_path()
    if dir:
        save_dir = os.path.join(save_dir, dir)
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Directory {save_dir} does not exist")
    file = os.path.join(save_dir, fname)
    assert os.path.exists(file), "File does not exist"
    return torch.load(file)
