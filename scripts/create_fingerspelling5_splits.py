import pathlib

import numpy as np
import pandas as pd
from sklearn import model_selection


from fmp.datasets.fingerspelling5 import utils

if __name__ == "__main__":
    root_path = pathlib.Path(__file__).parent.parent
    data_path = root_path / "data"
    fingerspelling5_csv = (
        data_path
        / "fingerspelling5"
        / "fingerspelling5"
        / "fingerspelling5_singlehands.csv"
    )
    save_path = fingerspelling5_csv.parent
    landmark_data = pd.read_csv(fingerspelling5_csv)
    group_colname = "person"
    groups = landmark_data["person"]
    n_splits = landmark_data[group_colname].nunique()
    value_columns = utils.generate_hand_landmark_columns()
    X = landmark_data[value_columns].values
    y = landmark_data["letter"].values
    split_overview = landmark_data.loc[:, ["person", "letter"]]
    group_kfold = model_selection.GroupKFold(n_splits=n_splits)
    for i, (train_index, valid_index) in enumerate(group_kfold.split(X, y, groups)):
        split_column = pd.Series(np.repeat("train", len(landmark_data)))
        split_column[valid_index] = "valid"
        filename = f"split_{i:02d}_{fingerspelling5_csv.stem}.csv"
        landmark_data["split"] = split_column
        landmark_data.to_csv(save_path / filename, index=False)
