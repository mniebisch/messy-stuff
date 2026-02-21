"""
Multi-source training/validation scaffold for PyTorch Lightning.

Implements:
- CSV-driven dataset construction: ImageRowDataset
- "sample dataset first" training strategy: MultiSourceEpochIterable
- size-tempered sampling with cap: alpha + CapConfig
- 2 validation loaders: (0) train-eval (deterministic) and (1) true validation
- per-source + overall validation metrics using per-sample losses (reduction='none')
- logging of sampling plan (quotas/probs/caps/repeat) as Lightning metrics

Main classes:
- CapConfig
- ImageRowDataset
- MultiSourceEpochIterable
- MultiSourceCsvDataModule
- LitMultiSource

Notes:
- Provide your own image_loader(path)->Tensor[C,H,W] and optionally target_col(s).
- Loss function must return per-sample losses (set reduction='none').
- For iterable per-epoch randomness with persistent_workers=True, LitMultiSource calls dm.set_epoch().
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    get_worker_info,
)


# ----------------------------
# Helper functions
# ----------------------------
def seed_worker(worker_id: int) -> None:
    """
    Called once per worker process. With persistent_workers=True, this will NOT
    be called each epoch. Epoch-dependent randomness should be handled via
    set_epoch() and epoch-seeded generators inside the IterableDataset.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reduce_to_per_sample(loss_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert loss output to shape [B]:
      - If already [B], return as-is.
      - If scalar [], raise (not usable for per-source breakdown).
      - If [B, ...], mean over remaining dims.
    """
    if loss_tensor.ndim == 0:
        raise ValueError(
            "Loss returned a scalar. For per-source validation, use a loss with "
            "reduction='none' or otherwise return per-sample losses."
        )
    if loss_tensor.ndim == 1:
        return loss_tensor
    return loss_tensor.view(loss_tensor.size(0), -1).mean(dim=1)


# ----------------------------
# Helper dataclass for oversampling caps
# ----------------------------
@dataclass(frozen=True)
class CapConfig:
    """
    Cap behavior for oversampling per epoch.

    quota_s <= cap_multiplier * N_s
    and optionally:
      quota_s >= cap_min
      quota_s <= cap_max

    Example:
      cap_multiplier=10.0 means any source can appear at most 10x its size per epoch.
    """

    cap_multiplier: float = 10.0
    cap_min: int = 0
    cap_max: Optional[int] = None


# ----------------------------
# Map-style dataset built from a dataframe slice
# ----------------------------
class ImageRowDataset(Dataset):
    """
    Map-style dataset from a dataframe slice.

    Expects df columns:
      - file_path (required)
      - optional target columns (target_col)

    Returns dict:
      image: Tensor
      target: Tensor or None
      source: str
      source_id: int
      file_path: str
    """

    def __init__(
        self,
        df: pd.DataFrame,
        source_name: str,
        source_id: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        image_loader: Optional[Callable[[str], torch.Tensor]] = None,
        target_col: Optional[Union[str, Sequence[str]]] = None,
        target_transform: Optional[Callable[[Any], torch.Tensor]] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.source_name = str(source_name)
        self.source_id = int(source_id)
        self.transform = transform
        self.image_loader = image_loader or self._default_image_loader
        self.target_col = target_col
        self.target_transform = target_transform

        if "file_path" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'file_path' column.")

        if self.target_col is not None:
            cols = (
                [self.target_col]
                if isinstance(self.target_col, str)
                else list(self.target_col)
            )
            missing = [c for c in cols if c not in self.df.columns]
            if missing:
                raise ValueError(f"Missing target columns in df: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        path = row["file_path"]

        image = self.image_loader(path)  # Tensor [C,H,W] recommended
        if self.transform is not None:
            image = self.transform(image)

        target = None
        if self.target_col is not None:
            if isinstance(self.target_col, str):
                raw_t = row[self.target_col]
            else:
                raw_t = row[list(self.target_col)].to_numpy()

            if self.target_transform is not None:
                target = self.target_transform(raw_t)
            else:
                target = torch.as_tensor(raw_t)

        return {
            "image": image,
            "target": target,
            "source": self.source_name,
            "source_id": self.source_id,
            "file_path": path,
        }

    @staticmethod
    def _default_image_loader(path: str) -> torch.Tensor:
        raise NotImplementedError(
            "Provide an image_loader(path)->Tensor[C,H,W] (e.g., torchvision.io.read_image, PIL->tensor, etc.)"
        )


# ----------------------------
# IterableDataset: sample dataset first, then uniform within dataset
# ----------------------------
class MultiSourceEpochIterable(IterableDataset):
    """
    Yields samples for one epoch by:
      1) allocating per-source quotas (p_s ∝ N_s^alpha)
      2) applying caps (CapConfig)
      3) building and shuffling a schedule of source IDs
      4) for each scheduled source, sampling uniformly within that source

    Multi-worker behavior:
      - The full schedule is built deterministically per epoch, per worker_id seed.
      - Each worker takes a strided shard schedule[worker_id::num_workers].
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        num_samples_per_epoch: int,
        alpha: float,
        cap: CapConfig,
        base_seed: int = 42,
    ) -> None:
        super().__init__()
        self.datasets = list(datasets)
        self.num_samples_per_epoch = int(num_samples_per_epoch)
        self.alpha = float(alpha)
        self.cap = cap
        self.base_seed = int(base_seed)

        self.sizes = [len(ds) for ds in self.datasets]
        if any(n <= 0 for n in self.sizes):
            raise ValueError("All subdatasets must have length > 0 for training.")

        self._epoch: int = 0  # set via set_epoch()

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

        # Epoch-dependent RNG (important with persistent workers)
        seed = self.base_seed + 1_000_003 * self._epoch + 10_007 * worker_id
        g = torch.Generator()
        g.manual_seed(seed)

        # Build schedule and shard per worker
        schedule = self._make_source_schedule(g)
        schedule_worker = schedule[worker_id::num_workers]

        for src_local_idx in schedule_worker:
            ds = self.datasets[src_local_idx]
            n = self.sizes[src_local_idx]
            idx = int(torch.randint(low=0, high=n, size=(1,), generator=g).item())
            yield ds[idx]

    def _make_source_schedule(self, g: torch.Generator) -> List[int]:
        sizes = self.sizes
        K = len(sizes)
        total = self.num_samples_per_epoch

        # p_s ∝ N_s^alpha
        w = torch.tensor([float(n) ** self.alpha for n in sizes], dtype=torch.double)
        p = (w / w.sum()).cpu().numpy().tolist()
        desired = [p[i] * total for i in range(K)]

        # caps
        caps: List[int] = []
        for n in sizes:
            cap_val = int(math.floor(self.cap.cap_multiplier * n))
            cap_val = max(cap_val, int(self.cap.cap_min))
            if self.cap.cap_max is not None:
                cap_val = min(cap_val, int(self.cap.cap_max))
            caps.append(cap_val)

        # Initial: floor(desired), clipped by caps
        quota = [min(int(math.floor(d)), caps[i]) for i, d in enumerate(desired)]
        assigned = sum(quota)
        remaining = total - assigned

        # Distribute remainder using largest fractional parts, respecting caps
        frac = [desired[i] - math.floor(desired[i]) for i in range(K)]
        if remaining > 0:
            order = sorted(range(K), key=lambda i: frac[i], reverse=True)
            for i in order:
                if remaining <= 0:
                    break
                slack = caps[i] - quota[i]
                if slack <= 0:
                    continue
                add = min(slack, remaining)
                quota[i] += add
                remaining -= add

        # If caps too strict, relax to hit total (fallback)
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

        # Build schedule and shuffle
        schedule: List[int] = []
        for i, q in enumerate(quota):
            schedule.extend([i] * q)

        perm = torch.randperm(len(schedule), generator=g).tolist()
        return [schedule[i] for i in perm]


# ----------------------------
# Lightning DataModule: CSV -> per-source datasets -> training iterable + 2 val loaders
# ----------------------------
class MultiSourceCsvDataModule(pl.LightningDataModule):
    """
    CSV columns:
      - file_path
      - source
      - split ('train'/'valid' by default)

    Training:
      - uses MultiSourceEpochIterable ("sample dataset first")
      - p_s ∝ N_s^alpha, capped by CapConfig
      - num_samples_per_epoch = steps_per_epoch * batch_size

    Validation:
      - returns two loaders:
          0) train-eval: deterministic pass over the training split (eval transforms)
          1) val: deterministic pass over the validation split
    """

    def __init__(
        self,
        csv_path: str,
        batch_size: int,
        steps_per_epoch: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = True,
        train_split_value: str = "train",
        valid_split_value: str = "valid",
        # transforms keyed by source name
        source_transforms_train: Optional[
            Dict[str, Callable[[torch.Tensor], torch.Tensor]]
        ] = None,
        source_transforms_valid: Optional[
            Dict[str, Callable[[torch.Tensor], torch.Tensor]]
        ] = None,
        # data specifics
        image_loader: Optional[Callable[[str], torch.Tensor]] = None,
        target_col: Optional[Union[str, Sequence[str]]] = None,
        target_transform: Optional[Callable[[Any], torch.Tensor]] = None,
        # sampling
        alpha: float = 0.5,
        cap: CapConfig = CapConfig(cap_multiplier=10.0, cap_min=0, cap_max=None),
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.num_workers = int(num_workers)
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.train_split_value = train_split_value
        self.valid_split_value = valid_split_value
        self.source_transforms_train = source_transforms_train or {}
        self.source_transforms_valid = source_transforms_valid or {}
        self.image_loader = image_loader
        self.target_col = target_col
        self.target_transform = target_transform
        self.alpha = float(alpha)
        self.cap = cap
        self.seed = int(seed)

        self._df: Optional[pd.DataFrame] = None
        self.source_to_id: Dict[str, int] = {}
        self.id_to_source: Dict[int, str] = {}

        # built in setup
        self.train_datasets: List[ImageRowDataset] = []
        self.train_iterable: Optional[MultiSourceEpochIterable] = None
        self.train_eval_concat: Optional[ConcatDataset] = None
        self.valid_concat: Optional[ConcatDataset] = None

        self._epoch: int = 0

    @property
    def num_sources(self) -> int:
        return len(self.id_to_source)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if self.train_iterable is not None:
            self.train_iterable.set_epoch(self._epoch)

    def prepare_data(self) -> None:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        if self._df is None:
            df = pd.read_csv(self.csv_path)
            required = {"file_path", "source", "split"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"CSV is missing columns: {sorted(missing)}")
            self._df = df

        df = self._df
        sources = sorted(df["source"].astype(str).unique().tolist())
        self.source_to_id = {s: i for i, s in enumerate(sources)}
        self.id_to_source = {i: s for s, i in self.source_to_id.items()}

        if stage in (None, "fit"):
            df_train = df[df["split"] == self.train_split_value].copy()
            df_valid = df[df["split"] == self.valid_split_value].copy()

            # Train map-style datasets WITH train transforms (for the iterable)
            self.train_datasets = self._build_per_source_datasets(
                df_train,
                transforms_by_source=self.source_transforms_train,
            )

            # Train-eval concat: training split, but eval transforms (no random aug)
            self.train_eval_concat = ConcatDataset(
                self._build_per_source_datasets(
                    df_train,
                    transforms_by_source=self.source_transforms_valid,
                )
            )

            # True validation concat: validation split, eval transforms
            self.valid_concat = ConcatDataset(
                self._build_per_source_datasets(
                    df_valid,
                    transforms_by_source=self.source_transforms_valid,
                )
            )

            num_samples_per_epoch = self.steps_per_epoch * self.batch_size
            self.train_iterable = MultiSourceEpochIterable(
                datasets=self.train_datasets,
                num_samples_per_epoch=num_samples_per_epoch,
                alpha=self.alpha,
                cap=self.cap,
                base_seed=self.seed,
            )
            self.train_iterable.set_epoch(self._epoch)

    def _build_per_source_datasets(
        self,
        df_split: pd.DataFrame,
        transforms_by_source: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
    ) -> List[ImageRowDataset]:
        datasets: List[ImageRowDataset] = []

        for source_name, df_src in df_split.groupby(
            df_split["source"].astype(str), sort=True
        ):
            src_id = self.source_to_id[source_name]
            transform = transforms_by_source.get(source_name, None)
            ds = ImageRowDataset(
                df=df_src,
                source_name=source_name,
                source_id=src_id,
                transform=transform,
                image_loader=self.image_loader,
                target_col=self.target_col,
                target_transform=self.target_transform,
            )
            datasets.append(ds)

        # deterministic ordering
        datasets.sort(key=lambda d: d.source_id)
        return datasets

    def train_dataloader(self) -> DataLoader:
        assert self.train_iterable is not None, "Call setup() first."
        return DataLoader(
            self.train_iterable,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=self.drop_last,
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self):
        assert (
            self.train_eval_concat is not None and self.valid_concat is not None
        ), "Call setup() first."

        train_eval_loader = DataLoader(
            self.train_eval_concat,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=False,
            worker_init_fn=seed_worker,
        )

        valid_loader = DataLoader(
            self.valid_concat,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=False,
            worker_init_fn=seed_worker,
        )

        return [train_eval_loader, valid_loader]


# ----------------------------
# LightningModule: per-sample loss -> overall + per-source val metrics; sampling-plan logging
# ----------------------------
class LitMultiSource(pl.LightningModule):
    """
    Requirements:
      - batch contains: image, target, source_id
      - loss_fn must return per-sample losses (use reduction='none')

    Validation:
      - expects 2 val loaders:
          0) train-eval
          1) val
      - logs overall and per-source metrics for both
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,  # MUST produce per-sample losses
        num_sources: int,
        lr: float = 1e-3,
        log_sampling_plan: bool = True,
        track_mae: bool = True,  # example metric; swap/extend for your task
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.num_sources = int(num_sources)
        self.lr = float(lr)
        self.log_sampling_plan = bool(log_sampling_plan)
        self.track_mae = bool(track_mae)

        # overall loss per loader
        self.val_loss = nn.ModuleList([torchmetrics.MeanMetric() for _ in range(2)])
        # per-source loss per loader
        self.val_loss_by_source = nn.ModuleList(
            [
                nn.ModuleList(
                    [torchmetrics.MeanMetric() for _ in range(self.num_sources)]
                )
                for _ in range(2)
            ]
        )

        if self.track_mae:
            self.val_mae = nn.ModuleList(
                [torchmetrics.MeanAbsoluteError() for _ in range(2)]
            )
            self.val_mae_by_source = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            torchmetrics.MeanAbsoluteError()
                            for _ in range(self.num_sources)
                        ]
                    )
                    for _ in range(2)
                ]
            )

        self._sampling_plan_logged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # -----------------------
    # Training
    # -----------------------
    def on_train_epoch_start(self) -> None:
        # important for iterable epoch randomness with persistent workers
        dm = self.trainer.datamodule
        if hasattr(dm, "set_epoch"):
            dm.set_epoch(self.current_epoch)
        self._sampling_plan_logged = False

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        # quotas exist after iterable builds schedule; that happens before first batch
        if (
            self.log_sampling_plan
            and (not self._sampling_plan_logged)
            and batch_idx == 0
        ):
            self._log_sampling_plan_metrics()
            self._sampling_plan_logged = True

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        x = batch["image"]
        y = batch.get("target", None)

        pred = self(x)

        if y is None:
            loss = pred.mean() * 0.0  # placeholder
        else:
            loss_ps = reduce_to_per_sample(self.loss_fn(pred, y))  # [B]
            loss = loss_ps.mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _log_sampling_plan_metrics(self) -> None:
        dm = self.trainer.datamodule
        it = getattr(dm, "train_iterable", None)

        if (
            it is None
            or it.last_quota is None
            or it.last_p is None
            or it.last_caps is None
        ):
            return

        # planned sampling distribution
        for s in range(self.num_sources):
            q = float(it.last_quota[s])
            p = float(it.last_p[s])
            cap = float(it.last_caps[s])
            n = float(it.sizes[s])
            rep = q / n if n > 0 else 0.0

            self.log(f"sampling/plan_p/source_{s}", p, on_epoch=True)
            self.log(f"sampling/plan_quota/source_{s}", q, on_epoch=True)
            self.log(f"sampling/plan_cap/source_{s}", cap, on_epoch=True)
            self.log(f"sampling/plan_repeat/source_{s}", rep, on_epoch=True)

    # -----------------------
    # Validation
    # -----------------------
    def on_validation_epoch_start(self) -> None:
        for li in range(2):
            self.val_loss[li].reset()
            for s in range(self.num_sources):
                self.val_loss_by_source[li][s].reset()

        if self.track_mae:
            for li in range(2):
                self.val_mae[li].reset()
                for s in range(self.num_sources):
                    self.val_mae_by_source[li][s].reset()

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        x = batch["image"]
        y = batch.get("target", None)
        sid = batch["source_id"]

        if y is None:
            return

        if not isinstance(sid, torch.Tensor):
            sid = torch.as_tensor(sid)
        sid = sid.to(torch.long)

        pred = self(x)

        # per-sample loss -> [B]
        loss_ps = reduce_to_per_sample(self.loss_fn(pred, y))

        # overall, sample-weighted
        self.val_loss[dataloader_idx].update(loss_ps.detach())

        # per-source
        for s in sid.unique().tolist():
            mask = sid == s
            self.val_loss_by_source[dataloader_idx][s].update(loss_ps.detach()[mask])

        # optional MAE
        if self.track_mae:
            self.val_mae[dataloader_idx].update(pred.detach(), y.detach())
            for s in sid.unique().tolist():
                mask = sid == s
                self.val_mae_by_source[dataloader_idx][s].update(
                    pred.detach()[mask], y.detach()[mask]
                )

    def on_validation_epoch_end(self) -> None:
        loader_prefix = {0: "val_train", 1: "val"}

        for li in range(2):
            prefix = loader_prefix.get(li, f"val_loader{li}")

            self.log(f"{prefix}/loss", self.val_loss[li].compute(), prog_bar=(li == 1))
            if self.track_mae:
                self.log(
                    f"{prefix}/mae", self.val_mae[li].compute(), prog_bar=(li == 1)
                )

            for s in range(self.num_sources):
                loss_s = self.val_loss_by_source[li][s].compute()
                if torch.isfinite(loss_s):
                    self.log(f"{prefix}/loss/source_{s}", loss_s)

                if self.track_mae:
                    mae_s = self.val_mae_by_source[li][s].compute()
                    if torch.isfinite(mae_s):
                        self.log(f"{prefix}/mae/source_{s}", mae_s)
