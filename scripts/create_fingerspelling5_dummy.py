import pathlib
import string

import click
import numpy as np
import pandas as pd

from fmp.datasets.fingerspelling5 import utils


@click.command()
@click.option(
    "--dir-dest",
    required=True,
    type=click.Path(
        path_type=pathlib.Path, resolve_path=True, dir_okay=True, file_okay=False
    ),
    help="Directory where new dataset is stored.",
)
@click.option(
    "--dataset-name",
    required=True,
    type=str,
    help="Name of created dataset directory and csv file.",
)
@click.option(
    "--num-persons",
    required=True,
    type=int,
    help="Number of persons which are created. For each person all datasets letters are created.",
)
@click.option(
    "--num-samples",
    required=True,
    type=int,
    help="Number of samples per letter per person.",
)
def main(dir_dest: pathlib.Path, dataset_name: str, num_persons: int, num_samples: int):
    file_name = f"{dataset_name}.csv"

    letters = utils.fingerspelling5.letters
    num_nodes = utils.mediapipe_hand_landmarks.num_nodes
    spatial_dims = utils.mediapipe_hand_landmarks.spatial_coords

    num_value_cols = num_nodes * len(spatial_dims)
    num_value_rows = num_persons * num_samples * len(letters)

    value_column_names = utils.generate_hand_landmark_columns()
    person_ids = [letter for letter in string.ascii_uppercase[:num_persons]]

    numeriacal_values = np.random.rand(num_value_rows, num_value_cols)
    letter_values = np.repeat(letters, num_value_rows // len(letters))
    person_id_values = np.repeat(person_ids, num_value_rows // len(person_ids))

    df = pd.DataFrame(numeriacal_values, columns=value_column_names)
    df["letter"] = letter_values
    df["person"] = person_id_values

    dataset_dir = dir_dest / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=False)

    df.to_csv(dataset_dir / file_name, index=False)
    click.echo(f"Dataset stored at '{str(dataset_dir / file_name)}.")


if __name__ == "__main__":
    main()
