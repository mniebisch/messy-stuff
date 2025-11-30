from typing import Any, Dict

import aim

__all__ = ["update_meta", "init_meta"]


def update_meta(run: aim.Run, update: Dict[str, Any]) -> None:
    meta = run.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("Run meta must be a dictionary")
    if any(key in meta for key in update.keys()):
        raise ValueError(f"Keys {update.keys()} already exist in run meta")
    meta.update(update)
    run["meta"] = meta


def init_meta(run: aim.Run, meta: Dict[str, Any]) -> None:
    # TODO don't like the naming. fix in future
    for key, value in meta.items():
        if run.get(key) is not None:
            raise ValueError(f"Key '{key}' already exists in run")
        run[key] = value
