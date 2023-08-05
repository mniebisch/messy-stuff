import json
import pathlib

import pandas as pd


if __name__ == "__main__":
    data_basepath = pathlib.Path.home() / "data"
    data_path = data_basepath / "asl_fingerspelling"

    train_file = data_path / "train.csv"
    train_df = pd.read_csv(train_file)

    example_file = train_df.loc[0].path
    example_path = data_path / example_file

    example_data = pd.read_parquet(example_path)

    spatial_dims = ("x", "y", "z")
    body_parts = ("face", "left_hand", "pose", "right_hand")
    nodes_num = {"face": 468, "left_hand": 21, "right_hand": 21, "pose": 33}

    keys = []
    for dim in spatial_dims:
        for body_part in body_parts:
            for node_ind in range(nodes_num[body_part]):
                key = f"{dim}_{body_part}_{node_ind}"
                keys.append(key)

    column_keys = example_data.columns[1:].values.tolist()

    assert len(keys) == len(column_keys)
    assert all([key1 == key2 for key1, key2 in zip(column_keys, keys)])

    repo_path = pathlib.Path(__file__).parent
    repo_data_path = repo_path / "data"
    csv_file = "indexing_asl_fingerspelling"

    with open(repo_data_path / csv_file, "w", encoding='utf-8') as f:
        json.dump(keys, f)
    