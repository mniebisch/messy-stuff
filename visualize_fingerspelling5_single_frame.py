import json
import pathlib

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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

    landmarks = point_data[0]

    # Landmark indices and edges
    edges = [
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

    # Create scatter plot for landmarks
    scatter = go.Scatter3d(
        x=landmarks[:, 0],
        y=landmarks[:, 1],
        z=landmarks[:, 2],
        mode="markers+text",
        marker=dict(size=6, color="blue"),
        text=[str(i) for i in range(21)],  # Label each landmark with its index
        textposition="top center",
    )

    # Create lines for edges
    lines = []
    for edge in edges:
        x_line = [landmarks[edge[0], 0], landmarks[edge[1], 0]]
        y_line = [landmarks[edge[0], 1], landmarks[edge[1], 1]]
        z_line = [landmarks[edge[0], 2], landmarks[edge[1], 2]]
        lines.append(
            go.Scatter3d(
                x=x_line, y=y_line, z=z_line, mode="lines", line=dict(color="red")
            )
        )

    # Create the figure
    fig = go.Figure(data=[scatter, *lines])

    # Set layout
    fig.update_layout(
        title="Hand Landmarks with Connections",
        scene=dict(aspectmode="data"),
        showlegend=False,
    )

    # Show the plot
    fig.show()

    # Create scatter plot for (x-y) pairing
    scatter_xy = go.Scatter(
        x=landmarks[:, 0],
        y=landmarks[:, 1],
        mode="markers+text",
        marker=dict(
            size=10, color=landmarks[:, 2], colorscale="Viridis", showscale=True
        ),
        text=[str(i) for i in range(21)],  # Label each landmark with its index
        textposition="top center",
    )

    # Create lines for edges in the (x-y) view
    lines = []
    for edge in edges:
        x_line = [landmarks[edge[0], 0], landmarks[edge[1], 0]]
        y_line = [landmarks[edge[0], 1], landmarks[edge[1], 1]]
        lines.append(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="red"),
                showlegend=False,
            )
        )

    # Create the figure for (x-y) view
    fig_xy = go.Figure(data=[scatter_xy, *lines])

    # Set layout for (x-y) view scatter plot
    scatter_layout = dict(
        title="Hand Landmarks (X-Y) with Connections",
        showlegend=False,
        xaxis_title="X",
        yaxis_title="Y",
    )

    fig_xy.update_layout(scatter_layout)

    # Show the scatter plot for (x-y) view
    fig_xy.show()
    print("Done")
