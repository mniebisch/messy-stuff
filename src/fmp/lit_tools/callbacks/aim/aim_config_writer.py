import os
import pathlib

import lightning as L
from aim.pytorch_lightning import AimLogger
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.cli import SaveConfigCallback
from typing_extensions import override

from . import utils

__all__ = ["AimConfigWriter"]


class AimConfigWriter(SaveConfigCallback):
    def __init__(self, *args, logdir: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.logdir = pathlib.Path(logdir).resolve()

    @override
    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        if self.already_saved:
            return

        # if not trainer.training or not trainer.testing:
        #     return

        if self.save_to_log_dir:
            if not isinstance(trainer.logger, AimLogger):
                raise ValueError("Trainer logger must be an instance of AimLogger")
            run = trainer.logger.experiment
            log_dir = str(self.logdir / run.experiment / run.hash)
            assert log_dir is not None
            config_path = os.path.join(log_dir, self.config_filename)
            fs = get_filesystem(log_dir)

            if not self.overwrite:
                # check if the file exists on rank 0
                file_exists = (
                    fs.isfile(config_path) if trainer.is_global_zero else False
                )
                # broadcast whether to fail to all ranks
                file_exists = trainer.strategy.broadcast(file_exists)
                if file_exists:
                    raise RuntimeError(
                        f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                        " results of a previous run. You can delete the previous config file,"
                        " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                        ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                    )

            if trainer.is_global_zero:
                # save only on rank zero to avoid race conditions.
                # the `log_dir` needs to be created as we rely on the logger to do it usually
                # but it hasn't logged anything at this point
                fs.makedirs(log_dir, exist_ok=True)
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=False,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )

                utils.update_meta(run, {"config_filename": str(log_dir)})

        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)
