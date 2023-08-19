import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_stat(data, func) -> float:
    per_sample_stat = func(data, axis=1)
    return np.mean(per_sample_stat)


if __name__ == "__main__":
    # Define the training dataset and dataloader (modify as per your data)
    data_path = pathlib.Path(__file__).parent / "data"
    fingerspelling_landmark_csv = data_path / "fingerspelling5_singlehands.csv"

    landmark_data = pd.read_csv(fingerspelling_landmark_csv)

    split_file = "fingerspelling_data_split.json"
    with open(split_file, "r") as f:
        split_data = json.load(f)
    train_index = split_data["train_index"]
    val_index = split_data["valid_index"]

    train_data = landmark_data.loc[train_index]
    train_data = train_data.dropna()
    valid_data = landmark_data.loc[val_index]

    coord_columns = train_data.columns.values[:-2]
    num_rows = len(train_data)
    point_data_raw = train_data.iloc[:, :-2].values
    point_data = point_data_raw.reshape(num_rows, -1, 3)

    min_sample_x = compute_stat(point_data[:, :, 0], np.min)
    max_sample_x = compute_stat(point_data[:, :, 0], np.max)
    std_sample_x = compute_stat(point_data[:, :, 0], np.std)

    min_x = np.min(point_data[:, :, 0])
    max_x = np.max(point_data[:, :, 0])
    std_x = np.std(point_data[:, :, 0])
    mean_x = np.mean(point_data[:, :, 0])
    median_x = np.median(point_data[:, :, 0])

    # TODO visualize via plotly!!! I guess something interactive seems easier
    # TODO check out z axis. data here inidcates that depth is computed
    # if that's the case why not for "live" demo

    print("Done")
