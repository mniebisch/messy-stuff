import json
import pathlib

import cv2
import numpy as np
import pandas as pd


def draw_hand(canvas, landmarks):
    height, width, _ = canvas.shape
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

    # Draw landmarks and edges
    for i in range(21):
        x, y = landmarks[i, :2]
        cv2.circle(
            canvas,
            (int(x * width), int(y * height)),
            10 + 2,
            (0, 255, 255),
            -1,
        )
        cv2.circle(canvas, (int(x * width), int(y * height)), 10, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            str(i),
            (int(x * width), int(y * height) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    for edge in edges:
        x1, y1 = landmarks[edge[0], :2]
        x2, y2 = landmarks[edge[1], :2]
        cv2.line(
            canvas,
            (int(x1 * width), int(y1 * height)),
            (int(x2 * width), int(y2 * height)),
            (0, 0, 0),
            2,
        )

        interpolated_x = interpolate_points(x1, x2, 8)
        interpolated_y = interpolate_points(y1, y2, 8)
        interpolated_values = interpolate_points(values[edge[0]], values[edge[1]], 8)

        for x, y, value in zip(interpolated_x, interpolated_y, interpolated_values):
            color_value = int(255 * (value - cmin) / (cmax - cmin))
            color_bgr = (color_value, 0, 0)  # BGR format
            cv2.circle(
                canvas,
                (int(x * width), int(y * height)),
                5 + 1,
                (0, 255, 255),
                -1,
            )
            cv2.circle(
                canvas,
                (int(x * width), int(y * height)),
                5,
                color_bgr,
                -1,
            )
    return canvas


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

    # Create a white canvas
    canvas_width = 800
    canvas_height = 400
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    canvas = draw_hand(canvas, landmarks)

    # Show the canvas
    cv2.imshow("Hand Landmarks", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Done")
