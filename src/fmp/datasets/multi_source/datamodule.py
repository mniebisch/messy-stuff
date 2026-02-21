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
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    get_worker_info,
)

from .dataset.train_dataset import CapConfig, MultiSourceEpochIterable
from .dataset.valid_dataset import ImageRowDataset

__all__ = ["MultiSourceCsvDataModule"]


# ----------------------------
# Worker seeding (multi-worker safe)
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
