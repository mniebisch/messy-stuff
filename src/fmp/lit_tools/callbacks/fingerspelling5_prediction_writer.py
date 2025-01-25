import pathlib
from typing import Any, Literal, Sequence, Union

import numpy as np
import pandas as pd
import torch
import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from fmp.datasets import fingerspelling5

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

        if trainer.ckpt_path is None:
            raise ValueError
        ckpt_path = pathlib.Path(trainer.ckpt_path)
        ckpt_name = ckpt_path.stem
        ckpt_version = ckpt_path.parts[-3]
        datamodule: fingerspelling5.Fingerspelling5LandmarkDataModule = (
            trainer.datamodule
        )
        dataset_name = pathlib.Path(datamodule.dataset_dir).name
        prediction_name = f"prediction__{dataset_name}__{ckpt_version}__{ckpt_name}"
        prediction_filename = prediction_name + ".csv"
        prediction_filepath = self.output_dir / prediction_filename
        prediction_hparams_filepath = prediction_filepath.with_suffix(".yaml")

        predictions = torch.concat(predictions)
        batch_indices = np.concatenate([bi for bi in batch_indices[0]], dtype=int)
        # TODO Attention! won't work for vision dataset
        img_files = trainer.datamodule.predict_data._landmark_data["img_file"]

        results = pd.DataFrame(
            dict(
                batch_indices=batch_indices, predictions=predictions, img_file=img_files
            )
        )

        results.to_csv(prediction_filepath, index=False)

        prediction_hparams = {
            "ckpt": str(ckpt_path),
            "prediction_output": str(prediction_filepath),
        }
        with open(prediction_hparams_filepath, "w") as hparams_file:
            yaml.dump(prediction_hparams, hparams_file)


def validate_pred_datamodule(
    datamodule: fingerspelling5.Fingerspelling5LandmarkDataModule,
) -> None:
    if datamodule.dataquality_file is not None:
        raise ValueError(
            "A dataquality file was provided. "
            "There is no need in running metric computation only on good or bad data."
        )

    if datamodule.datasplit_file is not None:
        raise ValueError(
            "A datasplit file was provided. "
            "There is no need in running metric computation only on good or bad data."
        )
