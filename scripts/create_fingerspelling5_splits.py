import pathlib

import click
import numpy as np
import pandas as pd
from sklearn import model_selection


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

    group_colname = "person"
    groups = landmark_data["person"]
    n_splits = landmark_data[group_colname].nunique()
    value_columns = fingerspelling5.utils.generate_hand_landmark_columns()

    X = landmark_data[value_columns].values
    y = landmark_data["letter"].values

    group_kfold = model_selection.GroupKFold(n_splits=n_splits)

    for i, (_, valid_index) in enumerate(group_kfold.split(X, y, groups)):
        split_overview = landmark_data.loc[:, ["person", "letter"]]
        split_column = pd.Series(np.repeat("train", len(landmark_data)))
        split_column[valid_index] = "valid"
        split_overview["split"] = split_column

        split_filename = f"split_{i:02d}_{dataset_name}.csv"

        split_overview.to_csv(dataset_dir / split_filename, index=False)


if __name__ == "__main__":
    main()
