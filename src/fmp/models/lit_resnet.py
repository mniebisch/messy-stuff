from typing import Any, Dict, List, Optional, Tuple, Union

import kornia as K
import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import functional as F

from fmp.models.resnet18 import ResNet18

KorniaAugmentations = Optional[List[K.augmentation.AugmentationBase2D]]


__all__ = ["ResNetClassifier"]


class ResNetClassifier(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        train_gpu_augmentations: KorniaAugmentations = None,
        train_gpu_kwargs: Optional[Dict[str, Any]] = None,
        valid_gpu_augmentations: KorniaAugmentations = None,
        valid_gpu_kwargs: Optional[Dict[str, Any]] = None,
        test_gpu_augmentations: KorniaAugmentations = None,
        test_gpu_kwargs: Optional[Dict[str, Any]] = None,
        predict_gpu_augmentations: KorniaAugmentations = None,
        predict_gpu_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super(ResNetClassifier, self).__init__()
        self.save_hyperparameters(ignore=["model"])

        self.example_input_array = torch.rand(1, 3, 224, 224)

        if not isinstance(model, ResNet18):
            raise ValueError("model must be an instance of ResNet18")

        self.resnet = model

        self.train_gpu_augmentations = _construct_transforms(
            train_gpu_augmentations, train_gpu_kwargs
        )
        self.valid_gpu_augmentations = _construct_transforms(
            valid_gpu_augmentations, valid_gpu_kwargs
        )
        self.test_gpu_augmentations = _construct_transforms(
            test_gpu_augmentations, test_gpu_kwargs
        )
        self.predict_gpu_augmentations = _construct_transforms(
            predict_gpu_augmentations, predict_gpu_kwargs
        )

        self.val_acc_overall = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.resnet.model.fc.out_features
        )
        self.val_acc_per_split = torch.nn.ModuleDict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

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

        split_name = self.trainer.val_dataloaders[dataloader_idx].dataset.split

        if split_name not in self.val_acc_per_split:
            if split_name in {"train", "valid"}:
                raise ValueError(
                    f"Dataset split name '{split_name}' is reserved by torch ModuleDict. "
                    "Please use different split names."
                )
            metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.resnet.model.fc.out_features
            )
            metric = metric.to(self.device)
            self.val_acc_per_split[split_name] = metric

        self.val_acc_per_split[split_name].update(predictions, labels)
        if split_name != "train_split":
            self.val_acc_overall.update(predictions, labels)

    def on_validation_epoch_end(self) -> None:
        for split_name, metric in self.val_acc_per_split.items():
            acc = metric.compute()
            self.log(f"acc/{split_name}", acc)
            metric.reset()

        overall_acc = self.val_acc_overall.compute()
        self.log("acc/valid_overall", overall_acc, prog_bar=True)
        self.val_acc_overall.reset()

    def test_step(
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
        self.log("acc/test", acc)

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        images, _ = batch
        predictions = self(images)
        return torch.argmax(predictions, dim=1)

    def on_after_batch_transfer(
        self, batch: Tuple[torch.Tensor, torch.Tensor], dataloader_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.train_gpu_augmentations is not None and self.trainer.training:
            images, labels = batch
            images = self.train_gpu_augmentations(images)
            batch = (images, labels)
        elif self.valid_gpu_augmentations is not None and self.trainer.validating:
            images, labels = batch
            images = self.valid_gpu_augmentations(images)
            batch = (images, labels)
        elif self.test_gpu_augmentations is not None and self.trainer.testing:
            images, labels = batch
            images = self.test_gpu_augmentations(images)
            batch = (images, labels)
        elif self.predict_gpu_augmentations is not None and self.trainer.predicting:
            images, labels = batch
            images = self.predict_gpu_augmentations(images)
            batch = (images, labels)
        else:
            raise ValueError("Unknown state in on_after_batch_transfer.")

        return batch


def _construct_transforms(
    transforms: KorniaAugmentations, transforms_kwargs: Optional[Dict[str, Any]]
) -> Union[K.augmentation.container.AugmentationSequential, None]:
    if transforms is None:
        return None

    transforms_kwargs = (transforms_kwargs or {}).copy()
    if "data_keys" not in transforms_kwargs:
        raise ValueError(
            "transforms_kwargs must contain 'data_keys' key specifying the type of input data"
        )
    transforms_kwargs["data_keys"] = ["image"]

    return K.augmentation.container.AugmentationSequential(
        *transforms, **transforms_kwargs
    )
