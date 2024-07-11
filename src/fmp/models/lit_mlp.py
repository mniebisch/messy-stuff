from typing import Tuple, List, Optional, Sequence, Union

import lightning as L
from torch.nn import functional as F

import torch

from fmp.models import mlp

__all__ = ["LitMLP", "SingleLayerMLP"]


# TODO rename class to LandmarkClassifier or LandmarkFrameBasesClassifier
class LitMLP(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        apply_batchnorm: bool = False,
        apply_dropout: bool = False,
        dropout_rate: float = 0.5,
        ignore: Optional[Union[str, Sequence[str]]] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=ignore)

        self.example_input_array = torch.rand(1, input_dim)

        self.mlp = mlp.MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            apply_batchnorm=apply_batchnorm,
            apply_dropout=apply_dropout,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        landmarks, label = batch
        predicted_label = self(landmarks)
        loss = F.cross_entropy(predicted_label, label)
        # TODO how to set ID for run (SummaryWriter)
        self.log("loss", loss)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        landmarks, labels = batch
        predictions = self(landmarks)
        predictions = torch.argmax(predictions, dim=1)
        labels = torch.argmax(labels, dim=1)
        total = labels.shape[0]
        correct = (predictions == labels).sum()
        acc = correct / total
        split = self.trainer.val_dataloaders[dataloader_idx].dataset.split
        self.log(f"acc/{split}", acc, add_dataloader_idx=False)

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        landmark_data, _ = batch
        predictions = self(landmark_data)
        return torch.argmax(predictions, dim=1)


class SingleLayerMLP(LitMLP):
    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__(hidden_dims=[hidden_dim], ignore="hidden_dims", **kwargs)
