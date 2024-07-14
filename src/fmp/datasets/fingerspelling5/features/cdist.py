import torch

__all__ = ["cdist_combination"]


def cdist_combination(spatial_coords: torch.Tensor) -> torch.Tensor:
    num_values = len(spatial_coords)
    combination_indices = torch.combinations(torch.arange(num_values), r=2)
    left_values = spatial_coords[combination_indices[:, 0]]
    right_values = spatial_coords[combination_indices[:, 1]]
    return left_values - right_values
