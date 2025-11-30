import pathlib

import lightning as L
from aim.pytorch_lightning import AimLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from . import utils

__all__ = ["AimModelCheckpoint"]


class AimModelCheckpoint(ModelCheckpoint):
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
        if not isinstance(trainer.logger, AimLogger):
            raise ValueError("Trainer logger must be an instance of AimLogger")
        run = trainer.logger.experiment

        if self.dirpath is None:
            raise ValueError("dirpath must be specified for AimModelCheckpoint")
        if not isinstance(self.dirpath, pathlib.Path):
            self.dirpath = pathlib.Path(self.dirpath)

        self.dirpath = str(self.dirpath / run.experiment / run.hash)

        utils.update_meta(run, {"checkpoint_dir": str(self.dirpath)})

        super().setup(trainer, pl_module, stage)
