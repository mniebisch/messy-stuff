import pathlib
import subprocess

from prefect import flow, task


def get_workspace_path() -> pathlib.Path:
    file_path = pathlib.Path(__file__)
    workspace_path = file_path.parent.parent.parent
    return workspace_path


@task
def create_dataset(
    data_path: pathlib.Path, dataset_name: str, num_persons: int, num_samples: int
) -> None:
    scripts_path = get_workspace_path() / "scripts"
    scripts_file = scripts_path / "create_fingerspelling5_dummy.py"
    subprocess.run(
        [
            "python",
            str(scripts_file),
            "--dir-dest",
            str(data_path),
            "--dataset-name",
            dataset_name,
            "--num-persons",
            str(num_persons),
            "--num-samples",
            str(num_samples),
        ]
    )


@task
def create_datasplits(data_path: pathlib.Path, dataset_name: str) -> None:
    scripts_path = get_workspace_path() / "scripts"
    scripts_file = scripts_path / "create_fingerspelling5_splits.py"

    dataset_path = data_path / dataset_name

    subprocess.run(
        [
            "python",
            str(scripts_file),
            "--dataset-dir",
            str(dataset_path),
        ]
    )


@task
def create_random_dataquality_file(data_path: pathlib.Path, dataset_name: str) -> None:
    scripts_path = get_workspace_path() / "scripts"
    scripts_file = scripts_path / "create_fingerspelling5_dummy_dataquality.py"

    dataset_path = data_path / dataset_name

    subprocess.run(
        [
            "python",
            str(scripts_file),
            "--dataset-dir",
            str(dataset_path),
        ]
    )


@task
def create_random_images(data_path: pathlib.Path, csv_file: pathlib.Path) -> None:
    scripts_path = get_workspace_path() / "scripts"
    scripts_file = scripts_path / "create_fingerspelling5_dummy_images.py"

    subprocess.run(
        [
            "python",
            str(scripts_file),
            "--data-directory",
            str(data_path),
            "--csv-path",
            str(csv_file),
        ]
    )


@flow
def pipeline():
    data_path = get_workspace_path() / "data" / "fingerspelling5"
    dataset_name = "fingerspelling5_dummy1"
    num_persons = 3
    num_samples = 4

    create_dataset(data_path, dataset_name, num_persons, num_samples)
    create_datasplits(data_path, dataset_name)
    create_random_dataquality_file(data_path, dataset_name)

    images_data_path = get_workspace_path().parent.parent / "data"
    # meh, coupled information to what happends in 'create_dataset'
    filename_csv = data_path / dataset_name / f"{dataset_name}.csv"
    create_random_images(images_data_path, filename_csv)


if __name__ == "__main__":
    pipeline()
