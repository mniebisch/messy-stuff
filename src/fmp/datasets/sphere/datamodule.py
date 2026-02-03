from typing import Any, List, Optional, Tuple

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from fmp.datasets.sphere.dataset import (
    FixedSphereSliceDataset,
    OnTheFlySphereSliceDataset,
    SphereSliceRanges,
)

__all__ = ["SphereDataModule"]


class SphereDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ranges: SphereSliceRanges,
        valA_ranges: SphereSliceRanges,
        train_length: int = 20000,
        valA_num: int = 1024,
        crop_size: Tuple[int, int] = (128, 128),
        batch_size: int = 8,
        num_workers: int = 4,
        train_seed: int = 12345,
        val_seed: int = 999,
        clamp_image: bool = True,
        normalize: bool = True,
        normalize_radius: bool = False,
        add_center_maps: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_ranges", "valA_ranges"])
        self.train_ranges = train_ranges
        self.valA_ranges = valA_ranges

        self.train_ds = None
        self.valA_ds = None

        self.train_length = train_length
        self.valA_num = valA_num
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_seed = train_seed
        self.val_seed = val_seed
        self.clamp_image = clamp_image
        self.normalize = normalize
        self.normalize_radius = normalize_radius
        self.add_center_maps = add_center_maps

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_ds = OnTheFlySphereSliceDataset(
                ranges=self.train_ranges,
                length=self.train_length,
                crop_size=self.crop_size,
                base_seed=self.train_seed,
                clamp_image=self.clamp_image,
                normalize=self.normalize,
                normalize_radius=self.normalize_radius,
                add_center_maps=self.add_center_maps,
            )

            self.valA_ds = FixedSphereSliceDataset(
                ranges=self.valA_ranges,
                num_samples=self.valA_num,
                crop_size=self.crop_size,
                seed=self.val_seed,
                clamp_image=self.clamp_image,
                normalize=self.normalize,
                normalize_radius=self.normalize_radius,
                add_center_maps=self.add_center_maps,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            # collate_fn=collate_x_y_cfg,
        )

    def val_dataloader(self) -> DataLoader:
        # You can return a list here in Lightning for multiple val loaders later (ValB, ValC, ...)
        return DataLoader(
            self.valA_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            # collate_fn=collate_x_y_cfg,
        )

    def on_train_epoch_start(self):
        # Ensure on-the-fly dataset changes across epochs deterministically
        if self.train_ds is not None:
            self.train_ds.set_epoch(int(self.trainer.current_epoch))


def collate_x_y_cfg(batch: List[Tuple[torch.Tensor, torch.Tensor, Any]]):
    """
    Batch entries are (x, y, cfg).
    Returns:
      x: (B, C, H, W)
      y: (B, 1, H, W)
      cfgs: list of cfg objects length B
    """
    xs, ys, cfgs = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    return x, y, list(cfgs)


if __name__ == "__main__":
    dm = SphereDataModule(
        train_ranges=SphereSliceRanges(), valA_ranges=SphereSliceRanges()
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    print("Done")
