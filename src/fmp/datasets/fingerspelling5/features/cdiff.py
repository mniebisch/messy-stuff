import torch

__all__ = ["cdiff_combination"]


def cdiff_combination(spatial_coords: torch.Tensor) -> torch.Tensor:
    if spatial_coords.dim() == 2:
        num_values = len(spatial_coords)
    elif spatial_coords.dim() == 3:
        num_values = spatial_coords.shape[1]
    else:
        raise ValueError(
            "Input has invalid number of dimensions. "
            "Either inputs with shape [P x M] or shape[B x P x M] are allowed."
        )

    combination_indices = torch.combinations(torch.arange(num_values), r=2)
    left_indices = combination_indices[:, 0]
    right_indices = combination_indices[:, 1]

    if spatial_coords.dim() == 2:
        left_values = spatial_coords[left_indices]
        right_values = spatial_coords[right_indices]
    else:
        left_values = spatial_coords[:, left_indices]
        right_values = spatial_coords[:, right_indices]

    return left_values - right_values
