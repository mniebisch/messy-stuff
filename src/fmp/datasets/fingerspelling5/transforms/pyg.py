from torch_geometric import data as pyg_data
from torch_geometric import transforms as pyg_transforms

from fmp.datasets.fingerspelling5 import utils
from fmp.datasets.fingerspelling5.features import cdiff

__all__ = ["NodeCDiff"]


class NodeCDiff(pyg_transforms.BaseTransform):
    def forward(self, data: pyg_data.Data) -> pyg_data.Data:
        if data.pos is None:
            raise ValueError("Invalid 'pos' input.")

        if utils.is_any_invalid_attribute_set(data):
            raise ValueError(
                "Transform will invalidate previous node structure. "
                "No attribute besides 'pos' is allowed to be set."
            )

        data.pos = cdiff.cdiff_combination(data.pos)
        return data
