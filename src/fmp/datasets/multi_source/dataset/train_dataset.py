from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    get_worker_info,
)

__all__ = ["CapConfig", "MultiSourceEpochIterable"]


# ----------------------------
# Train-time sampler dataset: sample source first, then uniform within source
# ----------------------------
@dataclass(frozen=True)
class CapConfig:
    """
    Cap behavior for oversampling per epoch.

    quota_s <= cap_multiplier * N_s
    and optionally also >= cap_min and <= cap_max

    Examples:
      cap_multiplier=10.0 means any source can appear at most 10x its size per epoch.
      cap_min=0 allows a source to get 0 samples if probabilities/total are small.
    """

    cap_multiplier: float = 10.0
    cap_min: int = 0
    cap_max: Optional[int] = None


class MultiSourceEpochIterable(IterableDataset):
    """
    IterableDataset that yields training samples by:
      - allocating per-source quotas each epoch
      - shuffling the per-source schedule
      - sampling indices uniformly within each chosen source

    Important: for correct per-epoch randomness with persistent workers, call set_epoch()
    (see DataModule note below).
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        source_ids: Sequence[int],
        num_samples_per_epoch: int,
        alpha: float,
        cap: CapConfig,
        base_seed: int = 42,
    ) -> None:
        super().__init__()
        assert len(datasets) == len(source_ids)
        self.datasets = list(datasets)
        self.source_ids = list(source_ids)
        self.num_samples_per_epoch = int(num_samples_per_epoch)
        self.alpha = float(alpha)
        self.cap = cap
        self.base_seed = int(base_seed)

        self._epoch: int = 0  # set via set_epoch()

        # Precompute sizes
        self.sizes = [len(ds) for ds in self.datasets]
        if any(n <= 0 for n in self.sizes):
            raise ValueError("All subdatasets must have length > 0 for training.")

        # For logging/inspection
        self.last_quota: Optional[List[int]] = None
        self.last_p: Optional[List[float]] = None
        self.last_caps: Optional[List[int]] = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1

        # Make epoch-dependent RNG. This is crucial when persistent_workers=True.
        seed = self.base_seed + 1_000_003 * self._epoch + 10_007 * worker_id
        g = torch.Generator()
        g.manual_seed(seed)

        # Build the full epoch schedule ONCE (same across workers), then shard by worker.
        # This ensures total quota math is exact at epoch level.
        schedule = self._make_source_schedule(g)

        # Shard the schedule deterministically among workers:
        # worker 0 takes schedule[0], schedule[num_workers], ...
        schedule_worker = schedule[worker_id::num_workers]

        for src_local_idx in schedule_worker:
            ds = self.datasets[src_local_idx]
            n = self.sizes[src_local_idx]
            # uniform within source
            idx = int(torch.randint(low=0, high=n, size=(1,), generator=g).item())
            yield ds[idx]

    def _make_source_schedule(self, g: torch.Generator) -> List[int]:
        """
        Returns a list of length num_samples_per_epoch with entries being local dataset indices.
        """
        sizes = self.sizes
        K = len(sizes)
        total = self.num_samples_per_epoch

        # Probabilities over sources: p_s ∝ N_s^alpha
        w = torch.tensor([float(n) ** self.alpha for n in sizes], dtype=torch.double)
        p = (w / w.sum()).cpu().numpy()

        # Desired (floating) allocations
        desired = [p[i] * total for i in range(K)]

        # Cap per source
        caps: List[int] = []
        for n in sizes:
            cap_val = int(math.floor(self.cap.cap_multiplier * n))
            cap_val = max(cap_val, int(self.cap.cap_min))
            if self.cap.cap_max is not None:
                cap_val = min(cap_val, int(self.cap.cap_max))
            caps.append(cap_val)

        # Initial allocation: floor(desired), then clip by cap
        quota = [min(int(math.floor(d)), caps[i]) for i, d in enumerate(desired)]
        assigned = sum(quota)
        remaining = total - assigned

        # Distribute remaining using largest fractional parts, respecting caps
        # (This keeps allocation close to desired but never exceeds caps.)
        frac = [desired[i] - math.floor(desired[i]) for i in range(K)]

        # While we still have remaining, give +1 to sources with slack, in order of frac.
        # If everything is capped and remaining > 0, we must relax: we’ll allow extra draws
        # by ignoring caps (rare, only if caps are too strict).
        if remaining > 0:
            # indices sorted by fractional part descending
            order = sorted(range(K), key=lambda i: frac[i], reverse=True)

            # First pass: respect caps
            for i in order:
                if remaining <= 0:
                    break
                slack = caps[i] - quota[i]
                if slack <= 0:
                    continue
                add = min(slack, remaining)
                quota[i] += add
                remaining -= add

        # If still remaining, caps were too tight to reach total; relax caps as fallback.
        # This keeps training running rather than silently shortening the epoch.
        if remaining > 0:
            order = sorted(range(K), key=lambda i: p[i], reverse=True)
            j = 0
            while remaining > 0:
                i = order[j % K]
                quota[i] += 1
                remaining -= 1
                j += 1

        # Store for logging
        self.last_quota = quota
        self.last_p = p
        self.last_caps = caps

        # Now build schedule list of local dataset indices
        schedule: List[int] = []
        for i, q in enumerate(quota):
            schedule.extend([i] * q)

        # Shuffle schedule
        perm = torch.randperm(len(schedule), generator=g).tolist()
        schedule = [schedule[i] for i in perm]

        return schedule
