import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    values = landmarks[:, 2]
    cmin = min(values)
    cmax = max(values)

    # Function to interpolate points between two landmarks
    def interpolate_points(p1, p2, num_points):
        return np.linspace(p1, p2, num_points + 2)[1:-1]

    # Create scatter plot for (x-y) pairing
    plt.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        c=values,
        cmap="viridis",
        s=30,  # Increase the size of the scatter points
        vmin=cmin,
        vmax=cmax,
    )
    for i in range(21):
        plt.text(landmarks[i, 0], landmarks[i, 1], str(i), ha="center", va="bottom")

    # Create lines for edges in the (x-y) view
    num_interpolated_points = 8

    for edge in edges:
        x_line = [landmarks[edge[0], 0], landmarks[edge[1], 0]]
        y_line = [landmarks[edge[0], 1], landmarks[edge[1], 1]]
        plt.plot(x_line, y_line, color="black")

        x_points = [
            *interpolate_points(
                landmarks[edge[0], 0], landmarks[edge[1], 0], num_interpolated_points
            ),
        ]
        y_points = [
            *interpolate_points(
                landmarks[edge[0], 1], landmarks[edge[1], 1], num_interpolated_points
            ),
        ]
        interpolated_values = [
            *interpolate_points(
                values[edge[0]], values[edge[1]], num_interpolated_points
            ),
        ]

        plt.scatter(
            x_points,
            y_points,
            c=interpolated_values,
            cmap="viridis",
            s=30,  # Increase the size of the interpolated points
            vmin=cmin,
            vmax=cmax,
        )

    # Set layout for (x-y) view scatter plot
    plt.title("Hand Landmarks (Y-X) with Connections and Interpolated Markers")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # Reverse the y-axis

    plt.colorbar(label="Value")

    # Show the scatter plot for (x-y) view
    plt.show()
    print("Done")
