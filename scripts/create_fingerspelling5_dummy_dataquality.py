import pathlib

import click
import numpy as np

from fmp.datasets import fingerspelling5


@click.command()
@click.option(
    "--dataset-dir",
    required=True,
    type=click.Path(
        path_type=pathlib.Path, resolve_path=True, dir_okay=True, file_okay=False
    ),
    help="Directory where dataset is stored. "
    "Split files will be placed here. "
    "Dataset data will be read from here.",
)
def main(dataset_dir: pathlib.Path):
    dataset_name = dataset_dir.parts[-1]
    data_file = f"{dataset_name}.csv"

    landmark_data = fingerspelling5.utils.read_csv(
        dataset_dir / data_file, filter_nans=True
    )

    landmark_data = landmark_data[["person", "letter", "img_file"]]
    views = np.random.randint(1, 100, size=len(landmark_data))
    is_corrupted = np.random.choice(
        [False, True], size=len(landmark_data), p=[0.90, 0.10]
    )

    landmark_data["views"] = views
    landmark_data["is_corrupted"] = is_corrupted

    file_name = f"{dataset_name}__data_quality.csv"
    landmark_data.to_csv(dataset_dir / file_name, index=False)


if __name__ == "__main__":
    main()
