import torch

from fmp.datasets.fingerspelling5.features import cdist


def test_two_points() -> None:
    input_coords = torch.Tensor([[1, 0, 3], [2, 4, -8]])
    expected = torch.Tensor([[-1, -4, 11]])
    output = cdist.cdist_combination(input_coords)
    torch.testing.assert_close(output, expected)


def test_three_points() -> None:
    input_coords = torch.Tensor([[1, 0, 3], [2, 4, -8], [1, 0, -1]])
    expected = torch.Tensor([[-1, -4, 11], [0, 0, 4], [1, 4, -7]])
    output = cdist.cdist_combination(input_coords)
    torch.testing.assert_close(output, expected)
