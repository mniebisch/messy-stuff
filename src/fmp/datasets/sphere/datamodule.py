from typing import Any, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from fmp.datasets.sphere.collate import collate_x_y_meta
from fmp.datasets.sphere.dataset import (
    FixedSphereSliceDataset,
    OnTheFlySphereSliceDataset,
    SphereSliceRanges,
)
from fmp.datasets.sphere.sampling import (
    MixtureSamplerConfig,
    SamplingConfig,
    SoftDzSamplerConfig,
    coerce_sampling,
)

__all__ = ["SphereDataModule"]


class SphereDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ranges: SphereSliceRanges,
        valA_ranges: SphereSliceRanges,
        train_length: int = 20000,
        valA_num: int = 1024,
        crop_size: Optional[Tuple[int, int]] = (128, 128),
        batch_size: int = 8,
        num_workers: int = 4,
        train_seed: int = 12345,
        val_seed: int = 999,
        clamp_image: bool = True,
        normalize: bool = True,
        normalize_radius: bool = False,
        add_center_maps: bool = False,
        sampling: Optional[Union[SamplingConfig, dict]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_ranges", "valA_ranges", "sampling"])
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
        self.sampling = coerce_sampling(sampling)

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
                sampling=self.sampling,
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
                sampling=self.sampling,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_x_y_meta,
        )

    def val_dataloader(self) -> DataLoader:
        # You can return a list here in Lightning for multiple val loaders later (ValB, ValC, ...)
        return DataLoader(
            self.valA_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_x_y_meta,
        )

    def on_train_epoch_start(self):
        # Ensure on-the-fly dataset changes across epochs deterministically
        if self.train_ds is not None:
            self.train_ds.set_epoch(int(self.trainer.current_epoch))


if __name__ == "__main__":
    import plotly.express as px
    from torchvision.utils import make_grid

    dm = SphereDataModule(
        train_ranges=SphereSliceRanges(
            height=(100, 100),
            width=(100, 100),
            depth=(16, 16),
            pixel_size=(1.0, 1.0),
            radius=(1.0, 50.0),
            z_index=0,
        ),
        valA_ranges=SphereSliceRanges(
            height=(100, 100),
            width=(100, 100),
            depth=(16, 16),
            pixel_size=(1.0, 1.0),
            radius=(1.0, 50.0),
            z_index=0,
        ),
        crop_size=None,
        batch_size=32,
        train_length=100000,
        valA_num=2048,
        num_workers=0,
        normalize=True,
        normalize_radius=False,
        add_center_maps=False,
        sampling={
            "mode": "mixture",
            "fixed_hw": True,
            "fixed_h": 100,
            "fixed_w": 100,
            "fixed_depth": 16,
            "fixed_pixel_size": 1.0,
            "mixture": {
                "p_strong": 0.7,
                "p_faint": 0.2,
                "p_zero": 0.1,
                "strong_alpha": 0.4,
                "faint_beta": 0.8,
                "zero_gamma": 0.1,
            },
        },
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    images = batch[1]  # (B, 1, H, W)
    grid = make_grid(images, nrow=8, normalize=False, scale_each=False, pad_value=1.0)

    fig = px.imshow(grid.permute(1, 2, 0).cpu().numpy(), binary_string=True)
    fig.show()

    print("Done")
