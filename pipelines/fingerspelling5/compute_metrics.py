import pathlib
import subprocess

import click
from prefect import flow, task


def get_workspace_path() -> pathlib.Path:
    file_path = pathlib.Path(__file__)
    workspace_path = file_path.parent.parent.parent
    return workspace_path


def get_metric_script_path() -> pathlib.Path:
    workspace_path = get_workspace_path()
    script_path = workspace_path / "scripts" / "compute_fingerspelling5_metrics.py"
    return script_path


def get_unscaled_config_path() -> pathlib.Path:
    workspace_path = get_workspace_path()
    config_path = (
        workspace_path
        / "configs"
        / "fingerspelling5_singlehands"
        / "metrics_unscaled.yaml"
    )
    return config_path


def get_scaled_config_path() -> pathlib.Path:
    workspace_path = get_workspace_path()
    config_path = (
        workspace_path
        / "configs"
        / "fingerspelling5_singlehands"
        / "metrics_scaled.yaml"
    )
    return config_path


@task
def compute_metrics(
    config_path: pathlib.Path,
    metrics_output_path: pathlib.Path,
    dataset_path: pathlib.Path,
) -> None:
    script_path = get_metric_script_path()

    subprocess.run(
        [
            "python",
            str(script_path),
            "predict",
            "--config",
            str(config_path),
            "--trainer.callbacks.init_args.output_dir",
            str(metrics_output_path),
            "--data.dataset_dir",
            str(dataset_path),
        ],
        check=True,
    )


@flow
def pipeline(dataset_path: pathlib.Path, output_path: pathlib.Path) -> None:
    compute_metrics(
        config_path=get_unscaled_config_path(),
        metrics_output_path=output_path,
        dataset_path=dataset_path,
    )
    compute_metrics(
        config_path=get_scaled_config_path(),
        metrics_output_path=output_path,
        dataset_path=dataset_path,
    )


@click.command()
@click.option(
    "--dataset-path",
    required=True,
    type=click.Path(
        path_type=pathlib.Path,
        exists=True,
        resolve_path=True,
        dir_okay=True,
        file_okay=False,
    ),
)
@click.option(
    "--output-path",
    required=True,
    type=click.Path(
        path_type=pathlib.Path,
        exists=True,
        resolve_path=True,
        dir_okay=True,
        file_okay=False,
    ),
)
def main(dataset_path: pathlib.Path, output_path: pathlib.Path) -> None:
    pipeline(dataset_path, output_path)


if __name__ == "__main__":
    main()
