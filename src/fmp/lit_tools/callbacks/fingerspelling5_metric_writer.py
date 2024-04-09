import pathlib
from typing import Any, Literal, List, Dict, Union

import numpy as np
import pandas as pd
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from fmp.datasets.fingerspelling5 import metrics, utils

__all__ = ["Fingerseplling5MetricWriter"]


class Fingerseplling5MetricWriter(BasePredictionWriter):
    # Method won't work for GNNS!!! curenntly only for 'flat hands'
    # pl_model needs to be 'identity' function/module
    def __init__(
        self,
        output_dir: Union[str, pathlib.Path],
        output_filename: str,
        write_interval: (
            Literal["batch"] | Literal["epoch"] | Literal["batch_and_epoch"]
        ) = "batch",
    ) -> None:
        super().__init__(write_interval)
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_filename = output_filename

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
            hand_metrics = self.metric_computer(hand)
            self.result_collection.append(hand_metrics)

        return None

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # assumes dataloader shuffle = False
        batch_indices = np.arange(len(self.result_collection))
        results = pd.DataFrame(self.result_collection)
        results["batch_indices"] = batch_indices
        results.to_csv(self.output_dir / self.output_filename, index=False)
