import pathlib
import subprocess
from typing import List, Tuple

import click
import yaml
from prefect import flow, task


def get_latest_directory(base_path: pathlib.Path) -> pathlib.Path:
    """
    Find all directories in the provided directory and return the full path of the latest directory.

    Parameters:
        base_path (pathlib.Path): The base directory to search for subdirectories.

    Returns:
        Path: The full path of the latest directory.
    """

    if not base_path.is_dir():
        raise ValueError(f"The provided path '{str(base_path)}' is not a directory.")

    # Get all subdirectories
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]

    if not subdirs:
        raise ValueError(
            f"No subdirectories found in the provided directory '{str(base_path)}'."
        )

    latest_dir = max(subdirs, key=lambda d: d.stat().st_ctime)

    return latest_dir


def get_latest_ckpt_file(base_path: pathlib.Path) -> pathlib.Path:
    """
    Find all .ckpt files in the provided directory and return the full path of the latest file.

    Parameters:
        base_path (pathlib.Path): The base directory to search for .ckpt files.

    Returns:
        Path: The full path of the latest .ckpt file.
    """

    if not base_path.is_dir():
        raise ValueError(f"The provided path '{str(base_path)}' is not a directory.")

    # Get all .ckpt files
    ckpt_files = [f for f in base_path.glob("*.ckpt") if f.is_file()]

    if not ckpt_files:
        raise ValueError(
            f"No .ckpt files found in the provided directory '{base_path}'."
        )

    latest_file = max(ckpt_files, key=lambda f: f.stat().st_ctime)

    return latest_file


@task
def train(train_config: pathlib.Path) -> None:
    subprocess.run(
        [
            "python",
            "scripts/train_fingerspelling5_litcli.py",
            "fit",
            "--config",
            str(train_config),
        ]
    )


@task
def predict(
    predict_config: pathlib.Path,
    train_log_config: pathlib.Path,
    ckpt_path: pathlib.Path,
) -> None:
    subprocess.run(
        [
            "python",
            "scripts/train_fingerspelling5_litcli.py",
            "predict",
            "--config",
            str(train_log_config),
            "--config",
            str(predict_config),
            "--ckpt_path",
            str(ckpt_path),
        ]
    )


@flow(log_prints=True)
def pipeline(train_config: pathlib.Path, predict_config: List[pathlib.Path]) -> None:
    train(train_config)
    train_log_config, ckpt_path = extract_train_outputs(train_config)
    for pconfig in predict_config:
        predict(pconfig, train_log_config, ckpt_path)


@task
def extract_train_outputs(
    train_config: pathlib.Path,
) -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Extracts the paths to the train configuration log file and the latest checkpoint file.

    Args:
        train_config (pathlib.Path): The path to the train configuration file.

    Returns:
        Tuple[pathlib.Path, pathlib.Path]: A tuple containing the path to the train configuration log file
        and the path to the latest checkpoint file.
    """
    with open(train_config, "r") as train_config_file:
        logged_train_config = yaml.safe_load(train_config_file)

    log_dir = pathlib.Path(
        logged_train_config["trainer"]["logger"]["init_args"]["save_dir"]
    )
    log_name = logged_train_config["trainer"]["logger"]["init_args"]["name"]

    experiment_path = get_latest_directory(log_dir / log_name)

    train_config_log = experiment_path / "config.yaml"
    latest_ckpt = get_latest_ckpt_file(experiment_path / "checkpoints")

    return train_config_log, latest_ckpt


@click.command()
@click.option(
    "--train-config",
    required=True,
    type=click.Path(
        path_type=pathlib.Path,
        exists=True,
        resolve_path=True,
        dir_okay=False,
        file_okay=True,
    ),
)
@click.option(
    "--predict-config",
    required=True,
    type=click.Path(
        path_type=pathlib.Path,
        exists=True,
        resolve_path=True,
        dir_okay=False,
        file_okay=True,
    ),
    multiple=True,
)
def main(train_config: pathlib.Path, predict_config: List[pathlib.Path]) -> None:
    pipeline(train_config, predict_config)


if __name__ == "__main__":
    main()
