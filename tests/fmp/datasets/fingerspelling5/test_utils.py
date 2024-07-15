import torch
from torch_geometric import data as pyg_data

from fmp.datasets.fingerspelling5 import utils


def test_is_any_invalid_attribute_set_containing_invalid_attributes():
    pos = torch.zeros(4, 3, dtype=torch.float)
    edge_index = torch.Tensor([[0, 0, 1, 1], [1, 2, 3, 3]])
    data = pyg_data.Data(pos=pos, edge_index=edge_index)

    assert utils.is_any_invalid_attribute_set(data)


def test_is_any_invalid_attribute_set_valid_attributes():
    pos = torch.zeros(4, 3, dtype=torch.float)
    data = pyg_data.Data(pos=pos)

    assert not utils.is_any_invalid_attribute_set(data)
