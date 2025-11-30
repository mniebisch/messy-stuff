from typing import Any, Mapping

import lightning as L
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from . import utils

__all__ = ["AimMetaWriter"]


def _merge_dict(target: dict, src: Mapping[str, Any]) -> dict:
    """Recursively merge src into target."""
    for key, value in src.items():
        if isinstance(value, Mapping):
            node = target.setdefault(key, {})
            _merge_dict(node, value)
        else:
            target[key] = value
    return target


def _is_jsonish(x: Any) -> bool:
    return (
        isinstance(x, (str, int, float, bool))
        or (isinstance(x, (list, tuple)) and all(_is_jsonish(i) for i in x))
        or (
            isinstance(x, dict)
            and all(isinstance(k, str) and _is_jsonish(v) for k, v in x.items())
        )
    )


class AimMetaWriter(L.Callback):
    """Attach arbitrary metadata to the Aim Run at test time."""

    def __init__(
        self,
        meta_kwargs: Mapping[str, Any],
        validate_jsonish: bool = False,
    ):
        self.meta = dict(meta_kwargs)
        self.validate_jsonish = validate_jsonish

    @rank_zero_only
    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule):

        if not isinstance(trainer.logger, AimLogger):
            raise ValueError("Trainer logger must be an instance of AimLogger")
        run = trainer.logger.experiment

        payload = dict(self.meta)

        # optionally filter non-JSON-serializable values
        if self.validate_jsonish:

            def _filter_jsonish(d):
                return {
                    k: _filter_jsonish(v) if isinstance(v, dict) else v
                    for k, v in d.items()
                    if _is_jsonish(v) or isinstance(v, dict)
                }

            payload = _filter_jsonish(payload)

        utils.init_meta(run, payload)
