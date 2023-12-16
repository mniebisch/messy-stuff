import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import tqdm
from numpy import typing as npt

import hand_description
import pipeline_fingerspelling5


def reshape_hands(df: pd.DataFrame) -> npt.NDArray:
    cols = pipeline_fingerspelling5.generate_hand_landmark_columns()
    hand_vals = df[cols].values
    hand_vals = hand_vals.reshape((-1, 21, 3))
    return hand_vals.astype(np.float64)


# load dataeset -> labels and co contained?
data_path = pathlib.Path(__file__).parent / "data"
fingerspelling_landmark_csv = data_path / "fingerspelling5_singlehands.csv"
landmark_data = pd.read_csv(fingerspelling_landmark_csv)

# are there nans?
landmark_data = landmark_data.loc[~landmark_data.isnull().any(axis=1)]
landmark_data = landmark_data.reset_index()

# scaling of landmarks?

# transform
hands = reshape_hands(landmark_data)

# compute stats
extent_stats = [hand_description.compute_extend(hand) for hand in tqdm.tqdm(hands)]
extent_stats = pd.DataFrame(extent_stats, columns=["x_extent", "y_extent", "z_extent"])

stats = pd.concat([landmark_data[["letter"]], extent_stats], axis=1)

stats_long = pd.melt(stats, id_vars="letter")
# goal in the end? -> checkout predictions and their properties

fig = px.box(stats_long, x="value", y="letter", color="variable")
fig.show()

print("Done")
