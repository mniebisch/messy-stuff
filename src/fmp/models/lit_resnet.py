from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

__all__ = ["ResNet18Classifier"]


class ResNet18Classifier(L.LightningModule):
    def __init__(self, num_classes: int):
        super(ResNet18Classifier, self).__init__()
        self.save_hyperparameters()

        self.example_input_array = torch.rand(1, 3, 224, 224)

        # Load a pre-trained ResNet-18 model
        self.model = models.resnet18(pretrained=True)

        # Modify the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.log("loss", loss)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        images, labels = batch
        predictions = self(images)
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
    ) -> torch.Tensor:
        images, _ = batch
        predictions = self(images)
        return torch.argmax(predictions, dim=1)
