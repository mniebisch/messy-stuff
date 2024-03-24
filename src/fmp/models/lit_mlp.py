from typing import Tuple, List

import lightning as L
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

import torch

__all__ = ["LitMLP"]


# TODO rename class to LandmarkClassifier or LandmarkFrameBasesClassifier
class LitMLP(L.LightningModule):
    def __init__(
        self, input_dim, hidden_dim, output_dim, learning_rate, scheduler_T_max
    ):
        super().__init__()

        self.save_hyperparameters()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x

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

    def configure_optimizers(
        self,
    ) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.scheduler_T_max
        )
        return [optimizer], [lr_scheduler]
