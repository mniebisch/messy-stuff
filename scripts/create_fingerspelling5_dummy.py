import pathlib
import string

import numpy as np
import pandas as pd

from fmp.datasets.fingerspelling5 import fingerspelling5, utils

if __name__ == "__main__":
    data_path = pathlib.Path(__file__).parent.parent / "data"
    file_name = "fingerspelling5_dummy_data.csv"

    num_persons = 5
    letters = fingerspelling5.FINGERSPELLING5["letters"]
    num_nodes = utils.MEDIAPIPE_HAND_LANDMARKS["num_nodes"]
    spatial_dims = utils.MEDIAPIPE_HAND_LANDMARKS["spatial_coords"]
    num_samples = 3

    num_value_cols = num_nodes * len(spatial_dims)
    num_value_rows = num_persons * num_samples * len(letters)

    value_column_names = utils.generate_hand_landmark_columns()
    person_ids = [l for l in string.ascii_uppercase[:num_persons]]

    numeriacal_values = np.random.rand(num_value_rows, num_value_cols)
    letter_values = np.repeat(letters, num_value_rows // len(letters))
    person_id_values = np.repeat(person_ids, num_value_rows // len(person_ids))

    df = pd.DataFrame(numeriacal_values, columns=value_column_names)
    df["letter"] = letter_values
    df["person"] = person_id_values

    df.to_csv(data_path / file_name, index=False)

    print("Done")
