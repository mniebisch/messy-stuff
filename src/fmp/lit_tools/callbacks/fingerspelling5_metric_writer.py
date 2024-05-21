import pathlib
from typing import Any, Literal, List, Dict, Union

import numpy as np
import pandas as pd
import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
import torchvision.transforms.v2 as transforms
import torch_geometric.transforms as pyg_transforms

from fmp.datasets.fingerspelling5 import metrics, utils

__all__ = ["Fingerseplling5MetricWriter"]


class Fingerseplling5MetricWriter(BasePredictionWriter):
    # Method won't work for GNNS!!! curenntly only for 'flat hands'
    # pl_model needs to be 'identity' function/module
    def __init__(
        self,
        output_dir: Union[str, pathlib.Path],
        write_interval: Union[
            Literal["batch"], Literal["epoch"], Literal["batch_and_epoch"]
        ] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metric_computer = metrics.FingerspellingMetrics()
        self.result_collection: List[Dict[str, float]] = []
        self.reshaper = utils.ReshapeToTriple()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        num_outputs = outputs.shape[0]
        for hand_index in range(num_outputs):
            hand_flat = outputs[hand_index]
            hand = self.reshaper(hand_flat)
            hand_metrics = self.metric_computer(hand.numpy())
            self.result_collection.append(hand_metrics)

        return None

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pred_transforms = trainer.datamodule.predict_data._transforms

        validate_pred_transforms(pred_transforms)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # assumes dataloader shuffle = False
        batch_indices = np.arange(len(self.result_collection))
        results = pd.DataFrame(self.result_collection)
        results["batch_indices"] = batch_indices

        dataset_name = trainer.datamodule.extract_dataset_name()
        pred_transforms = trainer.datamodule.predict_data._transforms
        transform_type = get_pred_transform_name(pred_transforms)
        with open(
            self.output_dir / f"{dataset_name}_{transform_type}_metric_hparams.yaml",
            "w",
        ) as hparams_file:
            yaml.dump(pl_module.trainer.datamodule.hparams, hparams_file)

        output_filename = f"{dataset_name}_{transform_type}.csv"
        results.to_csv(self.output_dir / output_filename, index=False)


def validate_pred_transforms(pred_transforms: transforms.Transform) -> None:
    if not isinstance(pred_transforms, transforms.Compose):
        raise ValueError(
            "Invalid transform structure (no 'Compose') for fingerspelling dataset. "
        )
    pred_transforms = pred_transforms.transforms
    num_transforms = len(pred_transforms)
    no_added_transforms = num_transforms == 2

    if no_added_transforms:
        return None
    elif num_transforms == 3:
        custom_transforms = pred_transforms[1]
        validate_custom_pred_transforms(custom_transforms)
    else:
        raise ValueError(
            "Invalid number of transforms for fingerspelling dataset. "
            "At max 3 transform 'stages' expected. "
            "Expected format: (1. transform to pyg data, "
            "2. [optional] custom transforms, "
            "3. remove pyg data)."
        )


def validate_custom_pred_transforms(
    custom_transforms: transforms.Transform,
) -> None:
    if not isinstance(custom_transforms, pyg_transforms.NormalizeScale):
        raise ValueError(
            "Invalid transform applied. "
            "For metric computation only PyGs NormalizeScale is valid or "
            "no tranformation at all. "
        )


def get_pred_transform_name(pred_transforms: transforms.Transform) -> str:
    validate_pred_transforms(pred_transforms)
    pred_transforms = pred_transforms.transforms

    num_transforms = len(pred_transforms)
    no_added_transforms = num_transforms == 2

    return "unscaled" if no_added_transforms else "scaled"
