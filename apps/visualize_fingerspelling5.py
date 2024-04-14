import pathlib

import pandas as pd

# Load data
root_path = pathlib.Path(__file__).parent.parent

metrics_path = root_path / "metrics"
data_path = root_path / "data"
predictions_path = root_path / "predictions"

metrics_scaled_file = metrics_path / "fingerspelling5_singlehands_scaled.csv"
metrics_unscaled_file = metrics_path / "fingerspelling5_singlehands.csv"
fingerspelling_file = data_path / "fingerspelling5_singlehands.csv"
predictions_file = None  # TODO


metrics_scaled = pd.read_csv(metrics_scaled_file)
metrics_unscaled = pd.read_csv(metrics_unscaled_file)

scale_col = "scaled"
metrics_scaled[scale_col] = True
metrics_unscaled[scale_col] = False

metrics = pd.concat([metrics_scaled, metrics_unscaled], axis=0).reset_index()

fingerspelling_data = pd.read_csv(fingerspelling_file)
fingerspelling_data = fingerspelling_data.loc[~fingerspelling_data.isnull().any(axis=1)]
fingerspelling_data = fingerspelling_data.reset_index()


print("Done")
