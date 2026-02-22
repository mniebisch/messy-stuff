from typing import Any, Dict, List, Tuple

import torch

__all__ = ["collate_x_y_meta"]


def collate_x_y_meta(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]):
    xs, ys, metas = zip(*batch)
    x = torch.stack(xs, 0)
    y = torch.stack(ys, 0)

    # Stack tensor fields; keep string fields as list
    out_meta: Dict[str, Any] = {}
    keys = metas[0].keys()
    for k in keys:
        vals = [m[k] for m in metas]
        if torch.is_tensor(vals[0]):
            out_meta[k] = torch.stack(vals, 0)
        else:
            out_meta[k] = vals  # list[str], etc.
    return x, y, out_meta
