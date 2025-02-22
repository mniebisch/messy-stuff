import pathlib
from typing import List

import click
import pandas as pd


def subset_letters(data: pd.DataFrame, letters: List[str]) -> pd.DataFrame:
    return data.loc[data["letter"].isin(letters)]


def subset_img_files(data: pd.DataFrame, img_files: pd.Series) -> pd.DataFrame:
    return data.loc[data["image_file"].isin(img_files)]


def main():
    workspace_dir = pathlib.Path(__file__).resolve().parents[1]

    letters = ["c", "l", "y"]
    dataset_dir = (
        workspace_dir / "data/fingerspelling5/fingerspelling5_singlehands_sorted"
    )
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

    # load data quality [OPTIONAL]
    def data_quality_filename(dataset_name: str) -> str:
        return f"{dataset_name}__data_quality.csv"

    data_quality = pd.read_csv(dataset_dir / data_quality_filename(dataset_name))

    def landmark_filename(dataset_name: str) -> str:
        return f"{dataset_name}.csv"

    # load landmarks
    landmarks = pd.read_csv(dataset_dir / landmark_filename(dataset_name))

    # load splits [OPTIONAL]
    split_files = list(dataset_dir.glob(f"split_*_{dataset_name}.csv"))
    splits = [pd.read_csv(split_file) for split_file in split_files]

    # load image sizes [OPTIONAL]
    image_sizes = pd.read_csv(dataset_dir / "image_sizes.csv")

    print("Done")


if __name__ == "__main__":
    main()
