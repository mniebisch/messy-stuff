import pathlib

from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go
import numpy as np
from numpy import typing as npt
from torch_geometric import transforms as pyg_transforms
from torchvision.transforms import v2
import pandas as pd
import plotly.express as px
from sklearn import mixture, neighbors


from fmp.datasets import fingerspelling5

connections = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 5),
    (0, 17),
    (5, 9),
    (9, 13),
    (13, 17),
]


def create_hand_dataframe(
    hand_flat: npt.NDArray, letter: str, person: str, hand_transform, frame_id: int
) -> pd.DataFrame:
    hand = hand_transform(hand_flat)
    hand = pd.DataFrame(hand, columns=["x", "y", "z"])
    hand["letter"] = letter
    hand["person"] = person
    hand["landmark_id"] = [
        str(i) for i in fingerspelling5.utils.mediapipe_hand_landmarks.parts.all
    ]
    hand["frame_id"] = frame_id
    return hand


def create_landmark_volume(
    landmark_data: npt.NDArray, landmark_name: str, colorscale: str
) -> go.Volume:
    gmm = mixture.GaussianMixture(n_components=1)
    gmm.fit(landmark_data)

    gmm_covs = np.diagonal(gmm.covariances_).flatten()
    gmm_means = gmm.means_.flatten()

    cov_mult = 100
    start_values = gmm_means - cov_mult * gmm_covs
    # start_values = np.min(
    #     gmm_data, axis=0
    # ).flatten()  #  buffer depending if negative or positive
    stop_values = gmm_means + cov_mult * gmm_covs
    # stop_values = np.max(
    #     gmm_data, axis=0
    # ).flatten  # buffer dependig if negative or positive

    grid_steps_n = 41

    x_steps = np.linspace(start_values[0], stop_values[0], grid_steps_n)
    y_steps = np.linspace(start_values[1], stop_values[1], grid_steps_n)
    z_steps = np.linspace(start_values[2], stop_values[2], grid_steps_n)

    x_grid, y_grid, z_grid = np.meshgrid(x_steps, y_steps, z_steps)
    x_grid = x_grid.reshape(-1, 1)
    y_grid = y_grid.reshape(-1, 1)
    z_grid = z_grid.reshape(-1, 1)

    pred_values = np.hstack([x_grid, y_grid, z_grid])

    log_likelihood = gmm.score_samples(pred_values)
    likelihood = np.exp(log_likelihood)

    volume = go.Volume(
        x=x_grid.flatten(),
        y=y_grid.flatten(),
        z=z_grid.flatten(),
        value=likelihood.flatten(),
        opacity=0.2,  # needs to be small to see through all surfaces
        colorscale=colorscale,
        name=landmark_name,
        surface_count=10,  # needs to be a large number for good volume rendering
    )

    return volume


if __name__ == "__main__":

    data = pathlib.Path(__file__).parent.parent
    data = (
        data
        / "data"
        / "fingerspelling5"
        / "fingerspelling5_singlehands"
        / "fingerspelling5_singlehands.csv"
    )

    raw_data = fingerspelling5.utils.read_csv(data, filter_nans=True)
    data_module = fingerspelling5.Fingerspelling5Landmark(raw_data)

    data_transforms = v2.Compose(
        [
            data_module._setup_landmark_pre_transforms(),
            pyg_transforms.NormalizeScale(),
            fingerspelling5.utils.PyGDataUnwrapper(),
        ],
    )

    landmark_data = data_module._landmark_data.loc[:, data_module._landmark_cols].values
    landmark_data = landmark_data.astype(np.float32)
    letter_data = data_module._landmark_data.loc[:, "letter"].values
    person_data = data_module._landmark_data.loc[:, "person"].values

    vis_data = [
        create_hand_dataframe(
            hand_flat=hand_flat,
            letter=letter,
            person=person,
            hand_transform=data_transforms,
            frame_id=frame_id,
        )
        for frame_id, (hand_flat, letter, person) in enumerate(
            zip(landmark_data, letter_data, person_data)
        )
        if letter == "f" and person == "B"
    ]
    vis_data = pd.concat(vis_data)

    # colors = ["Blues", "Greens", "Greys", "Reds", "Purples"]
    # volumes = []
    # for i, color in zip(
    #     fingerspelling5.utils.mediapipe_hand_landmarks.parts.all,
    #     itertools.cycle(colors),
    # ):
    #     row_selection = (
    #         (vis_data["letter"] == "l")
    #         & (vis_data["person"] == "A")
    #         & (vis_data["landmark_id"] == str(i))
    #     )
    #     letter_cols = data_module._landmark_cols
    #     scatter_data = vis_data.loc[row_selection]
    #     gmm_data = vis_data.loc[row_selection, ["x", "y", "z"]].values

    #     volume = create_landmark_volume(gmm_data, str(i), color)
    #     volumes.append(volume)

    # fig = go.Figure(data=volumes)
    # fig.show()

    # kd = neighbors.KernelDensity(bandwidth=0.01)
    # kd.fit(gmm_data)
    # log_likelihood_kd = kd.score_samples(pred_values)
    # likelihood_kd = np.exp(log_likelihood_kd)

    # volume2 = go.Volume(
    #     x=x_grid.flatten(),
    #     y=y_grid.flatten(),
    #     z=z_grid.flatten(),
    #     value=likelihood_kd.flatten(),<
    #     opacity=0.1,  # needs to be small to see through all surfaces
    #     colorscale="Oranges",
    #     surface_count=40,  # needs to be a large number for good volume rendering
    # )

    scatter_fig = px.scatter_3d(
        vis_data,
        x="x",
        y="y",
        z="z",
        color="landmark_id",
        symbol="person",
        opacity=0.25,
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_data="frame_id",
    )

    example_df = vis_data.loc[vis_data["frame_id"] == 51187]
    hand_fig = px.scatter_3d(
        example_df,
        x="x",
        y="y",
        z="z",
        color="landmark_id",
        symbol="person",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_data="frame_id",
    )
    scatter_fig.add_traces(hand_fig.data)

    for connection in connections:
        node_a_id, node_b_id = connection
        # assumption: "row" only contains one element
        node_a_row = example_df.loc[example_df["landmark_id"] == str(node_a_id)]
        node_b_row = example_df.loc[example_df["landmark_id"] == str(node_b_id)]
        x_vals = [node_a_row["x"].values[0], node_b_row["x"].values[0]]
        y_vals = [node_a_row["y"].values[0], node_b_row["y"].values[0]]
        z_vals = [node_a_row["z"].values[0], node_b_row["z"].values[0]]

        scatter_fig.add_trace(
            go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode="lines",
                line=dict(color="black"),
                name="Connection",
            )
        )

    scatter_fig.show()
    # scatter_fig.data[0]

    print("Done")
