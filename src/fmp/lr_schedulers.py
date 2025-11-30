import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

__all__ = ["cosine_annealing_warmup"]


def cosine_annealing_warmup(
    optimizer: Optimizer,
    warmup_iters: int,
    warmup_factor: float,
    total_iters: int,
) -> LRScheduler:

    t_max = total_iters - warmup_iters

    if t_max < 1:

        raise ValueError

    warmup_scheduler = LinearLR(
        optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    milestones = np.cumsum([warmup_iters])

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, scheduler],
        milestones=milestones.tolist(),
    )
