import torch


def tensor_generic_hash(data: torch.Tensor, buckets: int) -> torch.Tensor:
    """
    Hash a batched tensor into a bucket using a generic hash function based on the sum of the elements (pixels, word embeddings, etc.).

    Parameters:
    data (torch.Tensor): The tensor to hash.
    buckets (int): The number of buckets to hash into.

    Returns:
    int: The bucket index for the hash.
    """
    assert data.ndim == 2 or data.ndim == 3 or data.ndim == 4, "Data must be a 2D, 3D, or 4D tensor (1st dimension representing the batch)."
    summing_dims = None
    if data.ndim == 2:
        summing_dims = (1,)
    elif data.ndim == 3:
        summing_dims = (1, 2)
    else:
        summing_dims = (1, 2, 3)

    return torch.sum(data, dim=summing_dims) % buckets
