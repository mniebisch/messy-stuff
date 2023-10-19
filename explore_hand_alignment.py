import numpy as np
import plotly.graph_objects as go
from numpy import typing as npt
from plotly.subplots import make_subplots


def add_hand_2d(
    figure: go.Figure, hand: npt.NDArray, x_axis: int, y_axis: int, color: str
) -> None:
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 5),
        (0, 9),
        (0, 13),
        (0, 17),
        (5, 9),
        (9, 13),
        (13, 17),
    ]

    figure.add_trace(
        go.Scatter(
            x=hand[:, x_axis],
            y=hand[:, y_axis],
            mode="markers",
            marker=dict(size=5, color=color),
            name="Hand Landmarks",
        )
    )

    for connection in connections:
        x_vals = [hand[connection[0], x_axis], hand[connection[1], x_axis]]
        y_vals = [hand[connection[0], y_axis], hand[connection[1], y_axis]]

        figure.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color=color),
                name="Connection",
            )
        )

    return None


def add_hand_3d(figure: go.Figure, hand: npt.NDArray, color: str) -> None:
    # List of connections (replace with your actual connections)
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 5),
        (0, 9),
        (0, 13),
        (0, 17),
        (5, 9),
        (9, 13),
        (13, 17),
    ]
    # Add scatter points for hand landmarks
    figure.add_trace(
        go.Scatter3d(
            x=hand[:, 0],
            y=hand[:, 1],
            z=hand[:, 2],
            mode="markers",
            marker=dict(size=5, color=color),
            name="Hand Landmarks",
        )
    )

    # Add lines connecting the landmarks
    for connection in connections:
        x_vals = [hand[connection[0], 0], hand[connection[1], 0]]
        y_vals = [hand[connection[0], 1], hand[connection[1], 1]]
        z_vals = [hand[connection[0], 2], hand[connection[1], 2]]

        figure.add_trace(
            go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode="lines",
                line=dict(color=color),
                name="Connection",
            )
        )


# TODO load actual landmarks
hand_landmarks = np.random.rand(21, 3) * 100  # Replace this with your actual data
# TODO describe 'desired' knuckle line
# TODO describe 'desired' palm normal
# TODO create rotation matrix
# TODO compute rotated landmarks
hand_landmarks2 = np.random.rand(21, 3) * 100  # Replace this with your actual data


scatter_3d = go.Figure()
add_hand_3d(scatter_3d, hand_landmarks, "red")
add_hand_3d(scatter_3d, hand_landmarks2, "blue")

scatter_xy = go.Figure()
add_hand_2d(scatter_xy, hand_landmarks, x_axis=0, y_axis=1, color="red")
add_hand_2d(scatter_xy, hand_landmarks2, x_axis=0, y_axis=1, color="blue")

scatter_yz = go.Figure()
add_hand_2d(scatter_yz, hand_landmarks, x_axis=2, y_axis=1, color="red")
add_hand_2d(scatter_yz, hand_landmarks2, x_axis=2, y_axis=1, color="blue")

scatter_xz = go.Figure()
add_hand_2d(scatter_xz, hand_landmarks, x_axis=0, y_axis=2, color="red")
add_hand_2d(scatter_xz, hand_landmarks2, x_axis=0, y_axis=2, color="blue")

# Create subplots
fig = make_subplots(
    rows=5,
    cols=3,
    specs=[
        [
            {"type": "xy", "rowspan": 2},
            {"type": "xy", "rowspan": 2},
            {"type": "xy", "rowspan": 2},
        ],
        [None, None, None],
        [{"type": "scene", "colspan": 3, "rowspan": 3}, None, None],
        [None, None, None],
        [None, None, None],
    ],
    subplot_titles=[f"xy", f"zy", f"xz", "xyz"],
)


for trace_ind in range(len(scatter_xy["data"])):
    fig.append_trace(scatter_xy["data"][trace_ind], row=1, col=1)
for trace_ind in range(len(scatter_yz["data"])):
    fig.append_trace(scatter_yz["data"][trace_ind], row=1, col=2)
for trace_ind in range(len(scatter_xz["data"])):
    fig.append_trace(scatter_xz["data"][trace_ind], row=1, col=3)
for trace_ind in range(len(scatter_3d["data"])):
    fig.append_trace(scatter_3d["data"][trace_ind], row=3, col=1)

# Update layout
fig.update_layout(title="Explore alignment", height=1500, width=1500)

for col_ind, x_title, y_title in zip([1, 2, 3], ["x", "x", "z"], ["y", "z", "y"]):
    fig.update_xaxes(range=[0, 100], row=1, col=col_ind, title_text=x_title)
    fig.update_yaxes(range=[0, 100], row=1, col=col_ind, title_text=y_title)

# Show the figure
fig.show()
