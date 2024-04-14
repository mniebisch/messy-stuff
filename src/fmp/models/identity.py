from typing import Tuple

import lightning as L
import torch


__all__ = ["Identity"]


class Identity(L.LightningModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        landmark_data, _ = batch
        return self(landmark_data)
