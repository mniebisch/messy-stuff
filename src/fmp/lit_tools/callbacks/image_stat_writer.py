import pathlib
from typing import Any, List, Literal, Union

import pandas as pd
import torch
import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

__all__ = ["ImageStatWriter"]


class ImageStatWriter(BasePredictionWriter):
    def __init__(
        self,
        write_interval: Union[
            Literal["batch"], Literal["epoch"], Literal["batch_and_epoch"]
        ] = "batch",
    ) -> None:
        super().__init__(write_interval)

        self.mean_values: List[torch.Tensor] = []
        self.std_values: List[torch.Tensor] = []

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if type(trainer.predict_dataloaders.sampler).__name__ != "SequentialSampler":
            raise ValueError("Sampler must be SequentialSampler")

        self.mean_values.append(outputs.mean(dim=(2, 3)))
        self.std_values.append(outputs.std(dim=(2, 3)))

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        mean_values = torch.cat(self.mean_values)
        std_values = torch.cat(self.std_values)

        if trainer.datamodule.datasplit_file is not None:
            split_file = pathlib.Path(trainer.datamodule.datasplit_file)
            split_data = pd.read_csv(split_file)
            population_samples = split_data["split"] == "train"
        else:
            population_samples = torch.ones(mean_values.shape[0], dtype=torch.bool)

        mean_value = mean_values[population_samples].mean(dim=0)
        std_value = std_values[population_samples].mean(dim=0)

        output_filename_base = "image_mean_std"
        if trainer.datamodule.datasplit_file is not None:
            split_file = pathlib.Path(trainer.datamodule.datasplit_file)
            split_filename = split_file.stem
            output_filename = f"{split_filename}_{output_filename_base}.yaml"
        else:
            dataset_name = trainer.datamodule.dataset_name
            output_filename = f"{dataset_name}_{output_filename_base}.yaml"

        dataset_dir = pathlib.Path(trainer.datamodule.dataset_dir)
        with open(dataset_dir / output_filename, "w") as f:
            output = {
                "mean": mean_value.detach().cpu().numpy().tolist(),
                "std": std_value.detach().cpu().numpy().tolist(),
            }
            yaml.dump(output, f, default_flow_style=False)
