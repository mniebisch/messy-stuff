import collections
import pathlib
from typing import DefaultDict, Dict, List

from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go
import numpy as np
from numpy import typing as npt
from torch_geometric import transforms as pyg_transforms
from torchvision.transforms import v2
import pandas as pd
import plotly.express as px
from sklearn import mixture, neighbors
from skimage import io


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


img_data_dir = pathlib.Path(__file__).parent.parent.parent.parent / "data"
data_dir = pathlib.Path(__file__).parent.parent / "data" / "fingerspelling5"
dataset_name = "fingerspelling5_singlehands_sorted"
vis_dir = data_dir / dataset_name / "vis_data"
filename = f"{dataset_name}_vis_data.csv"

vis_data = pd.read_csv(vis_dir / filename, dtype={"landmark_id": str})


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

# scatter_fig = px.scatter_3d(
#     vis_data,
#     x="x",
#     y="y",
#     z="z",
#     color="landmark_id",
#     symbol="person",
#     opacity=0.25,
#     color_discrete_sequence=px.colors.qualitative.Dark24,
#     hover_data="frame_id",
# )

# example_df = vis_data.loc[vis_data["frame_id"] == 51187]
# hand_fig = px.scatter_3d(
#     example_df,
#     x="x",
#     y="y",
#     z="z",
#     color="landmark_id",
#     symbol="person",
#     color_discrete_sequence=px.colors.qualitative.Dark24,
#     hover_data="frame_id",
# )
# scatter_fig.add_traces(hand_fig.data)

# for connection in connections:
#     node_a_id, node_b_id = connection
#     # assumption: "row" only contains one element
#     node_a_row = example_df.loc[example_df["landmark_id"] == str(node_a_id)]
#     node_b_row = example_df.loc[example_df["landmark_id"] == str(node_b_id)]
#     x_vals = [node_a_row["x"].values[0], node_b_row["x"].values[0]]
#     y_vals = [node_a_row["y"].values[0], node_b_row["y"].values[0]]
#     z_vals = [node_a_row["z"].values[0], node_b_row["z"].values[0]]

#     scatter_fig.add_trace(
#         go.Scatter3d(
#             x=x_vals,
#             y=y_vals,
#             z=z_vals,
#             mode="lines",
#             line=dict(color="black"),
#             name="Connection",
#         )
#     )

persons = sorted(vis_data["person"].unique().tolist())
letters = fingerspelling5.utils.fingerspelling5.letters

frame_ids: DefaultDict[str, Dict[str, List[int]]] = collections.defaultdict(dict)
for person in persons:
    for letter in letters:
        df = vis_data.loc[
            (vis_data["person"] == person) & (vis_data["letter"] == letter)
        ]
        frame_ids[person][letter] = df["frame_id"].unique().tolist()

app = Dash(__name__)
app.layout = html.Div(
    [
        dcc.Dropdown(options=letters, value=letters[0], id="letter_id"),
        dcc.Dropdown(options=persons, value=persons[0], id="person_id"),
        dcc.Dropdown(options=frame_ids[persons[0]], id="frame_id"),
        html.Div(
            [
                html.Div(
                    dcc.Graph(
                        id="3d_hand_image", style={"height": "100%", "width": "100%"}
                    ),
                    style={"display": "inline-block", "width": "19%", "height": "19%"},
                ),
                html.Div(
                    # dcc.Graph(id="3d_figure", style={"height": "80vh", "width": "95vw"}),
                    dcc.Graph(
                        id="3d_figure", style={"height": "100%", "width": "100%"}
                    ),
                    style={"display": "inline-block", "width": "79%", "height": "80vh"},
                ),
            ]
        ),
    ]
)


@app.callback(
    Output("frame_id", "options"),
    Output("frame_id", "value"),
    Input("person_id", "value"),
    Input("letter_id", "value"),
)
def update_frame_ids(person_id, letter_id):
    options = frame_ids[person_id][letter_id]
    value = options[0]
    return options, value


@app.callback(
    Output("3d_figure", "figure"),
    Output("3d_hand_image", "figure"),
    Input("letter_id", "value"),
    Input("person_id", "value"),
    Input("frame_id", "value"),
)
def create_3d_scatter(letter_id: str, person_id: str, frame_id: str):
    landmark_color_map = {
        str(landmark_index): color
        for landmark_index, color in zip(
            fingerspelling5.utils.mediapipe_hand_landmarks.parts.all,
            px.colors.qualitative.Dark24,
        )
    }
    scatter_data = vis_data.loc[
        (vis_data["person"] == person_id) & (vis_data["letter"] == letter_id)
    ]
    scatter_fig = px.scatter_3d(
        scatter_data,
        x="x",
        y="y",
        z="z",
        color="landmark_id",
        symbol="person",
        opacity=0.25,
        color_discrete_map=landmark_color_map,
        hover_data="frame_id",
    )
    hand_data = scatter_data.loc[scatter_data["frame_id"] == frame_id].reset_index()
    add_hand_to_figure(scatter_fig, hand_data, "violet", "square", landmark_color_map)
    img_file = hand_data["img_file"].unique()
    if len(img_file) != 1:
        raise ValueError("Only one image should be source for data.")
    img_file = img_file[0]
    image_path = img_data_dir.joinpath(pathlib.Path(img_file))

    image_fig = create_2d_hand_image(image_path, hand_data)

    median_hand = (
        scatter_data.groupby("landmark_id")[["x", "y", "z"]].median().reset_index()
    )
    add_hand_to_figure(scatter_fig, median_hand, "grey", "diamond", landmark_color_map)

    camera = dict(
        up=dict(x=0, y=-1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=-2),
    )
    scatter_fig.update_layout(scene_camera=camera)

    return scatter_fig, image_fig


def add_hand_to_figure(
    fig: go.Figure,
    hand: pd.DataFrame,
    color: str,
    symbol: str,
    landmark_color_map: Dict[str, str],
) -> None:
    # add landmarks nodes
    hover_data = "frame_id" if "frame_id" in hand.columns else None
    hand_fig = px.scatter_3d(
        hand,
        x="x",
        y="y",
        z="z",
        color="landmark_id",
        color_discrete_map=landmark_color_map,
        hover_data=hover_data,
        symbol_sequence=[symbol],
    )
    fig.add_traces(hand_fig.data)

    # add lines betweem landmarks
    for connection in connections:
        node_a_id, node_b_id = connection
        # assumption: "row" only contains one element
        node_a_row = hand.loc[hand["landmark_id"] == str(node_a_id)]
        node_b_row = hand.loc[hand["landmark_id"] == str(node_b_id)]
        x_vals = [node_a_row["x"].values[0], node_b_row["x"].values[0]]
        y_vals = [node_a_row["y"].values[0], node_b_row["y"].values[0]]
        z_vals = [node_a_row["z"].values[0], node_b_row["z"].values[0]]

        fig.add_trace(
            go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode="lines",
                line=dict(color=color),
                name="Connection",
            )
        )

    # add palm
    nodes = fingerspelling5.utils.mediapipe_hand_landmarks.nodes
    nodes_indices = [
        nodes.wrist,
        nodes.index_mcp,
        nodes.middle_mcp,
        nodes.ring_mcp,
        nodes.pinky_mcp,
    ]
    nodes_indices = [str(node) for node in nodes_indices]
    mcp_data = hand.loc[hand["landmark_id"].isin(nodes_indices)]
    palm_fig = go.Mesh3d(
        x=mcp_data["x"], y=mcp_data["y"], z=mcp_data["z"], color=color, opacity=0.1
    )
    fig.add_trace(palm_fig)


def create_2d_hand_image(
    image_path: pathlib.Path, hand_landmarks: pd.DataFrame
) -> go.Figure:
    image = io.imread(image_path)
    hand_landmarks = hand_landmarks.copy()

    hand_landmarks.loc[:, "x_raw"] = hand_landmarks.loc[:, "x_raw"] * image.shape[1]
    hand_landmarks.loc[:, "y_raw"] = hand_landmarks.loc[:, "y_raw"] * image.shape[0]

    img_fig = px.imshow(image)
    fig_hand = px.scatter(hand_landmarks, x="x_raw", y="y_raw")
    img_fig.add_traces(fig_hand.data)

    for connection in connections:
        node_a_id, node_b_id = connection
        # assumption: "row" only contains one element
        node_a_row = hand_landmarks.loc[hand_landmarks["landmark_id"] == str(node_a_id)]
        node_b_row = hand_landmarks.loc[hand_landmarks["landmark_id"] == str(node_b_id)]
        x_vals = [node_a_row["x_raw"].values[0], node_b_row["x_raw"].values[0]]
        y_vals = [node_a_row["y_raw"].values[0], node_b_row["y_raw"].values[0]]
        img_fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color="yellow"),
                showlegend=False,
            )
        )

    return img_fig


# scatter_fig.show()
# scatter_fig.data[0]
if __name__ == "__main__":
    app.run(debug=True)
    print("Done")
