import kornia.augmentation as K
import lightning as L
import torch
import torch.nn as nn


class TTAModelWrapper(L.LightningModule):
    def __init__(self, model: L.LightningModule, num_aug: int = 10, alpha: float = 0.5):
        super().__init__()
        self.model = model
        self.num_aug = num_aug

        # Define TTA-specific transforms (GPU accelerated)
        # These should be "safe" augmentations for inference (e.g., no erasing)
        self.aug = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(
                degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), p=1.0
            ),
        )

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # 1. Unpack batch (handle your specific data structure)
        if isinstance(batch, (list, tuple)):
            x = batch[0]  # Assuming image is at index 0
        else:
            x = batch

        # 2. Standard Prediction
        logits = self.model(x)
        probs = [torch.softmax(logits, dim=1)]

        # 3. Augmented Predictions
        for _ in range(self.num_aug):
            # Apply Kornia transforms directly on the GPU tensor
            x_aug = self.aug(x)
            aug_logits = self.model(x_aug)
            probs.append(torch.softmax(aug_logits, dim=1))

        # 4. Average
        avg_probs = torch.stack(probs).mean(dim=0)
        return torch.argmax(avg_probs, dim=1)
