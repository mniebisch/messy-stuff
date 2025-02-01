from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

__all__ = ["ResNet18"]


class ResNet18(nn.Module):
    def __init__(
        self,
        num_classes: int,
        imagenet_ckpt_path: Optional[str] = None,
        pretrained_ckpt_path: Optional[str] = None,
    ) -> None:
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=None)

        if imagenet_ckpt_path is not None and pretrained_ckpt_path is not None:
            raise ValueError(
                "Only one of imagenet_ckpt_path or pretrained_ckpt_path can be provided."
            )

        if imagenet_ckpt_path:
            self.model.load_state_dict(
                torch.load(imagenet_ckpt_path, weights_only=True)
            )

        # Modify the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if pretrained_ckpt_path:
            self.load_from_checkpoint(pretrained_ckpt_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def load_from_checkpoint(self, ckpt_path: str) -> None:
        checkpoint = torch.load(
            ckpt_path, map_location=lambda storage, loc: storage, weights_only=True
        )
        self.load_state_dict(checkpoint["state_dict"])
