import json
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px

if __name__ == "__main__":
    # Define the training dataset and dataloader (modify as per your data)
    data_path = pathlib.Path(__file__).parent / "data"
    fingerspelling_landmark_csv = data_path / "fingerspelling5_singlehands.csv"

    landmark_data = pd.read_csv(fingerspelling_landmark_csv)

    # Load datasplit
    split_file = "fingerspelling_data_split.json"
    with open(split_file, "r") as f:
        split_data = json.load(f)
    train_index = split_data["train_index"]
    val_index = split_data["valid_index"]

    train_data = landmark_data.loc[train_index]
    train_data = train_data.dropna()

    # Reshape coords
    coord_columns = train_data.columns.values[:-2]
    num_rows = len(train_data)
    point_data_raw = train_data.iloc[:, :-2].values
    point_data = point_data_raw.reshape(num_rows, -1, 3)

    # nan row check
    zero_z = np.isclose(point_data[:, :, 2], 0, atol=0.001)
    zero_z_rows = np.all(zero_z, axis=1)
    print("Num samples with all landmarks on z=0.0: ", zero_z_rows.sum())

    z_start_value = np.repeat(point_data[:, 0, 2].reshape(-1, 1), 21, axis=1)
    start_z = np.isclose(point_data[:, :, 2], z_start_value, atol=0.001)
    same_plane_z_rows = np.all(start_z, axis=1)
    print(
        "Num samples where all landmarks are in same z plane", same_plane_z_rows.sum()
    )

    # plot distro
    coord_df = pd.DataFrame(
        {
            coord: point_data[:, :, ind].flatten()
            for ind, coord in enumerate(["x", "y", "z"])
        }
    )
    coord_df = coord_df.melt(value_vars=["x", "y", "z"])

    fig = px.violin(coord_df, y="value", color="variable")
    fig.show()
