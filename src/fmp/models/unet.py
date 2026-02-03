# LightningModule for SMP UNet regression on your synthetic slice task.
# - Uses segmentation_models.pytorch (smp)
# - Works with your datasets that return (x, y, cfg/meta)
# - Uses the WeightedHuberWithGradLoss (Huber + gradient term) from earlier

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Loss: Huber + gradient term (as previously shared)
# -------------------------

__all__ = ["SphereSliceUNetModule"]


class WeightedHuberWithGradLoss(nn.Module):
    """
    Weighted pixel-wise Huber (SmoothL1) + gradient consistency loss.

    Supports multi-channel outputs (C>=1) with configurable weighting behavior.

    Inputs:
      pred, target: (N, C, H, W) or (N, H, W) or (N, 1, H, W)

    Loss:
      L = L_pix + grad_weight * (L_grad_x + L_grad_y)

    Weighting:
      - weight_mode="per_channel": weights computed per-channel using target>0 etc.  (old behavior)
      - weight_mode="any_channel": one weight per pixel, active if ANY channel is active (base = max over channels)
      - weight_mode="mean_abs": one weight per pixel based on mean(abs(target)) across channels
      - weight_mode="l2": one weight per pixel based on sqrt(mean(target^2)) across channels

    Notes:
      - If you train on normalized targets in [0,1], beta ~ 0.1 is a good start.
      - If your targets can be negative, "pos_weight" with target>0 may be less meaningful; consider value weighting.
    """

    def __init__(
        self,
        beta: float = 0.1,
        grad_weight: float = 0.1,
        # weighting
        weight_mode: str = "any_channel",  # "per_channel" | "any_channel" | "mean_abs" | "l2"
        pos_weight: float = 0.0,
        pos_threshold: float = 0.0,  # used for "active" detection: base > pos_threshold
        use_value_weight: bool = False,
        value_weight_alpha: float = 0.0,
        value_weight_gamma: float = 1.0,
        value_weight_clamp_min: float = 0.0,
        # reduction
        eps: float = 1e-12,
        reduction: str = "mean",  # "mean" (weighted mean) or "sum" (weighted sum)
    ):
        super().__init__()

        if beta < 0:
            raise ValueError("beta must be >= 0.")
        if grad_weight < 0:
            raise ValueError("grad_weight must be >= 0.")
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'.")
        if weight_mode not in ("per_channel", "any_channel", "mean_abs", "l2"):
            raise ValueError(
                "weight_mode must be one of: per_channel, any_channel, mean_abs, l2."
            )
        if value_weight_gamma <= 0:
            raise ValueError("value_weight_gamma must be > 0.")

        self.beta = float(beta)
        self.grad_weight = float(grad_weight)

        self.weight_mode = weight_mode
        self.pos_weight = float(pos_weight)
        self.pos_threshold = float(pos_threshold)

        self.use_value_weight = bool(use_value_weight)
        self.value_weight_alpha = float(value_weight_alpha)
        self.value_weight_gamma = float(value_weight_gamma)
        self.value_weight_clamp_min = float(value_weight_clamp_min)

        self.eps = float(eps)
        self.reduction = reduction

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # (N,H,W)
            return x.unsqueeze(1)
        if x.dim() == 4:  # (N,C,H,W)
            return x
        raise ValueError("pred/target must have shape (N,H,W) or (N,C,H,W).")

    @staticmethod
    def _forward_diffs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward differences (∂x, ∂y) with replicate padding to keep shape.
        Returns dx, dy each shaped (N,C,H,W).
        """
        dx = x[..., :, 1:] - x[..., :, :-1]
        dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")

        dy = x[..., 1:, :] - x[..., :-1, :]
        dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")

        return dx, dy

    def _base_map(self, target: torch.Tensor) -> torch.Tensor:
        """
        Returns a base map for weighting with shape (N,1,H,W) unless weight_mode="per_channel",
        in which case it returns (N,C,H,W).
        """
        if self.weight_mode == "per_channel":
            return target  # (N,C,H,W)

        # One scalar per pixel (N,1,H,W)
        if target.size(1) == 1:
            return target  # (N,1,H,W)

        if self.weight_mode == "any_channel":
            # Use max magnitude to decide if any channel is active
            return target.abs().amax(dim=1, keepdim=True)
        if self.weight_mode == "mean_abs":
            return target.abs().mean(dim=1, keepdim=True)
        if self.weight_mode == "l2":
            return torch.sqrt(
                torch.mean(target * target, dim=1, keepdim=True).clamp_min(self.eps)
            )

        raise RuntimeError("Unhandled weight_mode")

    def _make_weights(self, target: torch.Tensor) -> torch.Tensor:
        """
        Produces weights with shape (N,C,H,W), either per-channel or broadcasted across channels.
        """
        base = self._base_map(target)  # (N,C,H,W) or (N,1,H,W)

        w = torch.ones_like(base)

        if self.pos_weight > 0:
            active = (base > self.pos_threshold).to(base.dtype)
            w = w * (1.0 + self.pos_weight * active)

        if self.use_value_weight and self.value_weight_alpha > 0:
            val = torch.clamp(base, min=self.value_weight_clamp_min)
            w = w * (1.0 + self.value_weight_alpha * (val**self.value_weight_gamma))

        # If base is (N,1,H,W) and target is multi-channel, expand weights across channels
        if w.size(1) == 1 and target.size(1) > 1:
            w = w.expand_as(target)

        return w

    def _reduce(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Weighted reduction. For reduction="mean", returns sum(w*x)/sum(w).
        For reduction="sum", returns sum(w*x).
        """
        xw = x * w
        if self.reduction == "sum":
            return xw.sum()
        denom = w.sum().clamp_min(self.eps)
        return xw.sum() / denom

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._to_nchw(pred)
        target = self._to_nchw(target)

        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target shapes must match. Got {pred.shape} vs {target.shape}"
            )

        w = self._make_weights(target)

        # Pixel-wise Huber
        pix = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")
        pix_loss = self._reduce(pix, w)

        if self.grad_weight == 0.0:
            return pix_loss

        # Gradient consistency (Huber on ∂x and ∂y)
        pred_dx, pred_dy = self._forward_diffs(pred)
        tgt_dx, tgt_dy = self._forward_diffs(target)

        grad_x = F.smooth_l1_loss(pred_dx, tgt_dx, beta=self.beta, reduction="none")
        grad_y = F.smooth_l1_loss(pred_dy, tgt_dy, beta=self.beta, reduction="none")

        grad_loss = self._reduce(grad_x, w) + self._reduce(grad_y, w)

        return pix_loss + self.grad_weight * grad_loss


# -------------------------
# LightningModule
# -------------------------


class SphereSliceUNetModule(pl.LightningModule):
    """
    UNet regression LightningModule (segmentation_models.pytorch).

    Expects batches as either:
      - (x, y) or
      - (x, y, cfg_list) or
      - (x, y, meta_dict)

    where:
      x: (B, C, H, W)
      y: (B, out_channels, H, W)  target top-layer image
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        # SMP UNet params:
        encoder_name: str = "resnet18",
        encoder_weights: Optional[
            str
        ] = None,  # set "imagenet" if you want, but with in_channels!=3 it's less pure
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
        # output handling:
        output_activation: str = "none",  # "none" | "sigmoid"
        # loss params:
        huber_beta: float = 0.1,
        grad_weight: float = 0.1,
        pos_weight: float = 0.0,
        use_value_weight: bool = False,
        value_weight_alpha: float = 0.0,
        value_weight_gamma: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,  # regression output channel
            decoder_channels=decoder_channels,
            activation=None,  # we'll apply optional sigmoid ourselves
        )

        self.output_activation = output_activation.lower()
        if self.output_activation not in ("none", "sigmoid"):
            raise ValueError("output_activation must be 'none' or 'sigmoid'.")

        self.criterion = WeightedHuberWithGradLoss(
            beta=huber_beta,
            grad_weight=grad_weight,
            weight_mode="any_channel",
            pos_weight=pos_weight,
            pos_threshold=0.0,
            use_value_weight=use_value_weight,
            value_weight_alpha=value_weight_alpha,
            value_weight_gamma=value_weight_gamma,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        if self.output_activation == "sigmoid":
            y_hat = torch.sigmoid(y_hat)
        return y_hat

    @staticmethod
    def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
                return x, y, None
            if len(batch) == 3:
                x, y, meta = batch
                return x, y, meta
        raise ValueError("Batch must be (x,y) or (x,y,meta/cfg).")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y, meta = self._unpack_batch(batch)
        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)

        # extra metrics (useful to watch)
        with torch.no_grad():
            l1 = torch.mean(torch.abs(y_hat - y))
            mse = torch.mean((y_hat - y) ** 2)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/l1", l1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mse", mse, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y, meta = self._unpack_batch(batch)
        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)
        l1 = torch.mean(torch.abs(y_hat - y))
        mse = torch.mean((y_hat - y) ** 2)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/l1", l1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mse", mse, on_step=False, on_epoch=True, prog_bar=False)

        # Optional: log a couple stats from meta if provided as a dict (from collate_x_y_cfg_to_tensors)
        if isinstance(meta, dict) and "radius" in meta and batch_idx == 0:
            self.log(
                "val/radius_mean",
                meta["radius"].float().mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            if "pixel_size" in meta:
                self.log(
                    "val/pixel_size_mean",
                    meta["pixel_size"].float().mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        return {"val_loss": loss}
