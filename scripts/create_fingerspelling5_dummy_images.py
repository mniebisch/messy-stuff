from pathlib import Path
from typing import List

import click
import cv2
import numpy as np
import pandas as pd


def generate_and_save_random_images(file_paths: List[Path], m: int, n: int) -> None:
    for file_path in file_paths:
        # Create the necessary directories if they don't exist

        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate a random image of size [m, n] with 3 color channels (BGR)
        random_image = np.random.randint(0, 256, (m, n, 3), dtype=np.uint8)

        # Save the image to the specified path
        cv2.imwrite(str(file_path), random_image)


def extract_img_file_column(csv_path: Path) -> List[str]:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Check if 'img_file' column exists
    if "img_file" not in df.columns:
        raise ValueError("The CSV file does not contain an 'img_file' column.")

    # Extract the 'img_file' column and return it as a list
    img_file_list = df["img_file"].tolist()

    return img_file_list


def combine_data_directory_with_files(
    data_directory: Path, file_paths: List[str]
) -> List[Path]:
    combined_paths = [data_directory / file_path for file_path in file_paths]
    return combined_paths


def store_image_files_overview(csv_path: Path) -> None:
    landmark_data = pd.read_csv(csv_path)
    image_files_overview = landmark_data.loc[
        :, ["letter", "person", "img_file"]
    ].reset_index(drop=True)
    dataset_dir = Path(csv_path).parent
    image_files_file = dataset_dir / "image_files.csv"
    image_files_overview.to_csv(image_files_file, index=False)


@click.command()
@click.option(
    "--data-directory",
    required=True,
    type=click.Path(path_type=Path, resolve_path=True, dir_okay=True, file_okay=False),
)
@click.option(
    "--csv-path",
    required=True,
    type=click.Path(path_type=Path, resolve_path=True, dir_okay=False, file_okay=True),
)
def main(data_directory: Path, csv_path: Path):
    # Extract the 'img_file' column from the CSV file
    img_file_paths = extract_img_file_column(csv_path)

    # Combine the data directory with the file paths
    combined_paths = combine_data_directory_with_files(data_directory, img_file_paths)

    # Generate and save random images to the specified paths
    generate_and_save_random_images(combined_paths, 64, 64)

    store_image_files_overview(csv_path)


if __name__ == "__main__":
    main()
