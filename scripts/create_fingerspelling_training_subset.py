import pathlib
from typing import List

import click
import pandas as pd


def subset_letters(data: pd.DataFrame, letters: List[str]) -> pd.DataFrame:
    return data.loc[data["letter"].isin(letters)]


def subset_img_files(data: pd.DataFrame, img_files: pd.Series) -> pd.DataFrame:
    return data.loc[data["img_file"].isin(img_files)]


def landmark_filename(dataset_name: str) -> str:
    return f"{dataset_name}.csv"


def data_quality_filename(dataset_name: str) -> str:
    return f"{dataset_name}__data_quality.csv"


@click.command()
@click.option(
    "--dataset-dir",
    type=click.Path(
        path_type=pathlib.Path,
        exists=True,
        resolve_path=True,
        dir_okay=True,
        file_okay=False,
    ),
    required=True,
)
@click.option(
    "--letters",
    type=str,
    multiple=True,
    required=True,
)
def main(dataset_dir: pathlib.Path, letters: List[str]) -> None:
    dataset_path = dataset_dir.parents[0]
    dataset_name = dataset_dir.name
    output_dataset_name = f"{dataset_name}__{''.join(sorted(letters))}"
    output_dataset_path = dataset_path / output_dataset_name
    output_dataset_path.mkdir(exist_ok=False, parents=False)

    # load image files
    image_files_filename = "image_files.csv"
    image_files = pd.read_csv(dataset_dir / image_files_filename)
    image_files_subset = subset_letters(image_files, letters)
    image_files_subset.to_csv(output_dataset_path / image_files_filename, index=False)

    # load landmarks
    landmarks = pd.read_csv(dataset_dir / landmark_filename(dataset_name))
    landmarks_subset = subset_letters(landmarks, letters)
    landmarks_subset.to_csv(
        output_dataset_path / landmark_filename(output_dataset_name), index=False
    )

    # load data quality [OPTIONAL]
    data_quality_filepath = dataset_dir / data_quality_filename(dataset_name)
    if data_quality_filepath.exists():
        data_quality = pd.read_csv(data_quality_filepath)
        data_quality_subset = subset_letters(data_quality, letters)
        data_quality_subset.to_csv(
            output_dataset_path / data_quality_filename(output_dataset_name),
            index=False,
        )

    # load splits [OPTIONAL]
    split_files = list(dataset_dir.glob(f"split_*_{dataset_name}.csv"))
    if split_files:
        splits = [pd.read_csv(split_file) for split_file in split_files]
        splits_subset = [subset_letters(split, letters) for split in splits]

        for split_subset, split_file in zip(splits_subset, split_files):
            split_subset.to_csv(output_dataset_path / split_file.name, index=False)

    # load image sizes [OPTIONAL]
    image_sizes_filepath = dataset_dir / "image_sizes.csv"
    if image_sizes_filepath.exists():
        image_sizes = pd.read_csv(image_sizes_filepath)
        image_sizes_subset = subset_img_files(
            image_sizes, image_files_subset["img_file"]
        )
        image_sizes_subset.to_csv(output_dataset_path / "image_sizes.csv", index=False)


if __name__ == "__main__":
    main()
