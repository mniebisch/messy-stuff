import json
import pathlib

import pandas as pd

if __name__ == "__main__":
    data_basepath = pathlib.Path("~/data")
    data_path = data_basepath / "asl_signs"

    # load labels
    data_csv = "train.csv"
    train_df = pd.read_csv(data_path / data_csv)
    example_data = pd.read_parquet(data_path / train_df.loc[0].path)

    body_parts = ("face", "left_hand", "pose", "right_hand")
    num_nodes = {"face": 468, "left_hand": 21, "right_hand": 21, "pose": 33}

    body_type = []
    node_indices = []
    for body_part in body_parts:
        for num_ind in range(num_nodes[body_part]):
            body_type.append(body_part)
            node_indices.append(num_ind)

    example_frame = example_data[:543]
    example_type = example_frame.type.values.tolist()
    example_landmark_index = example_frame.landmark_index.values.tolist()

    assert len(example_type) == len(body_type)
    assert len(example_landmark_index) == len(node_indices)
    assert all(
        example == extracted for example, extracted in zip(example_type, body_type)
    )
    assert all(
        example == extracted
        for example, extracted in zip(example_landmark_index, node_indices)
    )

    output = list(zip(body_type, node_indices))
    repo_path = pathlib.Path(__file__).parent
    repo_data_path = repo_path / "data"
    csv_file = "indexing_asl_signs"

    with open(repo_data_path / csv_file, "w", encoding='utf-8') as f:
        json.dump(output, f)
