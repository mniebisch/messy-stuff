from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


def reduce_to_per_sample(loss_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert loss output to shape [B].
    - If already [B], return as-is.
    - If scalar [], that's NOT usable for per-source breakdown.
    - If [B, ...], mean over remaining dims.
    """
    if loss_tensor.ndim == 0:
        raise ValueError(
            "Loss returned a scalar. For per-source validation, use a loss with reduction='none' "
            "or otherwise return per-sample losses."
        )
    if loss_tensor.ndim == 1:
        return loss_tensor
    # [B, ...] -> [B]
    return loss_tensor.view(loss_tensor.size(0), -1).mean(dim=1)


class LitMultiSource(pl.LightningModule):
    """
    Requirements:
      - batch contains: image, target, source_id
      - loss_fn must return per-sample losses (use reduction='none')

    Validation:
      - expects 2 val loaders:
          0) train-eval
          1) val
      - logs overall and per-source metrics for both
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,  # MUST produce per-sample loss (reduction='none')
        num_sources: int,
        lr: float = 1e-3,
        log_sampling_plan: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.num_sources = int(num_sources)
        self.lr = float(lr)
        self.log_sampling_plan = bool(log_sampling_plan)

        # Overall loss per loader
        self.val_loss = nn.ModuleList([torchmetrics.MeanMetric() for _ in range(2)])

        # Per-source loss per loader: [loader][source]
        self.val_loss_by_source = nn.ModuleList(
            [
                nn.ModuleList(
                    [torchmetrics.MeanMetric() for _ in range(self.num_sources)]
                )
                for _ in range(2)
            ]
        )

        # Optional example metric: MAE (regression). If you want accuracy, swap it.
        self.val_mae = nn.ModuleList(
            [torchmetrics.MeanAbsoluteError() for _ in range(2)]
        )
        self.val_mae_by_source = nn.ModuleList(
            [
                nn.ModuleList(
                    [torchmetrics.MeanAbsoluteError() for _ in range(self.num_sources)]
                )
                for _ in range(2)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # -----------------------
    # Train hooks (epoch seeding + optional sampling-plan logging)
    # -----------------------
    def on_train_epoch_start(self) -> None:
        dm = self.trainer.datamodule
        if hasattr(dm, "set_epoch"):
            dm.set_epoch(self.current_epoch)

        # The iterable only computes quotas when iterated; so we log sampling plan
        # after first batch (see on_train_batch_start).
        self._sampling_plan_logged = False

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        if (
            self.log_sampling_plan
            and (not getattr(self, "_sampling_plan_logged", False))
            and batch_idx == 0
        ):
            self._log_sampling_plan_metrics()
            self._sampling_plan_logged = True

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        x = batch["image"]
        y = batch.get("target", None)

        pred = self(x)

        # If you need train loss too, you can compute per-sample then mean
        if y is None:
            loss = pred.mean() * 0.0  # placeholder
        else:
            loss_ps = reduce_to_per_sample(self.loss_fn(pred, y))  # [B]
            loss = loss_ps.mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _log_sampling_plan_metrics(self) -> None:
        """
        Logs planned per-source sampling quotas/caps/probs from the DataModule's iterable.
        """
        dm = self.trainer.datamodule
        it = getattr(dm, "train_iterable", None)

        if (
            it is None
            or it.last_quota is None
            or it.last_p is None
            or it.last_caps is None
        ):
            return

        for s in range(self.num_sources):
            q = float(it.last_quota[s])
            p = float(it.last_p[s])
            cap = float(it.last_caps[s])
            n = float(it.sizes[s])
            rep = q / n if n > 0 else 0.0

            self.log(f"sampling/plan_p/source_{s}", p, on_epoch=True)
            self.log(f"sampling/plan_quota/source_{s}", q, on_epoch=True)
            self.log(f"sampling/plan_cap/source_{s}", cap, on_epoch=True)
            self.log(f"sampling/plan_repeat/source_{s}", rep, on_epoch=True)

    # -----------------------
    # Validation (2 loaders)
    # -----------------------
    def on_validation_epoch_start(self) -> None:
        for li in range(2):
            self.val_loss[li].reset()
            self.val_mae[li].reset()
            for s in range(self.num_sources):
                self.val_loss_by_source[li][s].reset()
                self.val_mae_by_source[li][s].reset()

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        x = batch["image"]
        y = batch.get("target", None)
        sid = batch["source_id"]

        if not isinstance(sid, torch.Tensor):
            sid = torch.as_tensor(sid)
        sid = sid.to(torch.long)

        pred = self(x)

        if y is None:
            # If you truly have no targets, per-source loss isn't meaningful.
            # But leaving a placeholder here keeps the template consistent.
            return

        # 1) compute per-sample loss
        loss_ps = reduce_to_per_sample(self.loss_fn(pred, y))  # [B]

        # 2) overall loss (sample-weighted)
        self.val_loss[dataloader_idx].update(loss_ps.detach())

        # 3) per-source loss
        for s in sid.unique().tolist():
            mask = sid == s
            self.val_loss_by_source[dataloader_idx][s].update(loss_ps.detach()[mask])

        # Optional metric (MAE): overall + per-source
        self.val_mae[dataloader_idx].update(pred.detach(), y.detach())
        for s in sid.unique().tolist():
            mask = sid == s
            self.val_mae_by_source[dataloader_idx][s].update(
                pred.detach()[mask], y.detach()[mask]
            )

    def on_validation_epoch_end(self) -> None:
        loader_prefix = {0: "val_train", 1: "val"}

        for li in range(2):
            prefix = loader_prefix.get(li, f"val_loader{li}")

            self.log(f"{prefix}/loss", self.val_loss[li].compute(), prog_bar=(li == 1))
            self.log(f"{prefix}/mae", self.val_mae[li].compute(), prog_bar=(li == 1))

            for s in range(self.num_sources):
                # This will be NaN if never updated; skip non-finite
                loss_s = self.val_loss_by_source[li][s].compute()
                if torch.isfinite(loss_s):
                    self.log(f"{prefix}/loss/source_{s}", loss_s)

                mae_s = self.val_mae_by_source[li][s].compute()
                if torch.isfinite(mae_s):
                    self.log(f"{prefix}/mae/source_{s}", mae_s)
