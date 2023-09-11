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

    train_data = train_data.loc[train_data["letter"].isin(["u", "v", "r"])]
    train_data = train_data.loc[
        :, ["z_hand_12", "z_hand_8", "x_hand_12", "x_hand_8", "letter"]
    ]
    train_data["z_diff"] = train_data["z_hand_12"] - train_data["z_hand_8"]
    train_data["x_diff"] = train_data["x_hand_12"] - train_data["x_hand_8"]

    fig = px.scatter(train_data, x="x_diff", y="z_diff", color="letter")
    fig.show()
