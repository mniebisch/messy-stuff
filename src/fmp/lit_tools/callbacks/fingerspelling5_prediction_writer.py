import pathlib
from typing import Any, Literal, Sequence, Union

import torch
import numpy as np
import pandas as pd
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

__all__ = ["Fingerspelling5PredictionWriter"]


class Fingerspelling5PredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Union[str, pathlib.Path],
        write_interval: Union[
            Literal["batch"], Literal["epoch"], Literal["batch_and_epoch"]
        ] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self.output_dir = pathlib.Path(output_dir)

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
    ) -> None:

        default_root_dir = pathlib.Path(trainer.default_root_dir)
        if trainer.ckpt_path is None:
            raise ValueError
        ckpt_path = pathlib.Path(trainer.ckpt_path)
        ckpt_name = ckpt_path.stem
        ckpt_version = ckpt_path.parts[-3]
        prediction_filename = f"prediction__{ckpt_version}__{ckpt_name}.csv"

        predictions = torch.concat(predictions)
        batch_indices = np.concatenate([bi for bi in batch_indices[0]], dtype=int)

        results = pd.DataFrame(
            dict(batch_indices=batch_indices, predictions=predictions)
        )
        results.to_csv(
            default_root_dir / self.output_dir / prediction_filename, index=False
        )
