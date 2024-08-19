import pathlib
from typing import Tuple

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt
from skimage import io
from matplotlib.colors import Normalize

from fmp.datasets import fingerspelling5


def map_line_color(node_a: int, node_b: int) -> Tuple[int, int, int]:
    hand_parts  = fingerspelling5.utils.mediapipe_hand_landmarks.parts.__dict__
    node_mapping =  [(part, node_index) for part, node_indices in  hand_parts.items() for node_index in node_indices]
    node_sorted = sorted(node_mapping, key=lambda x: x[1])
    node_lookup = [part for part, _ in node_sorted if part != "all" and part != "palm"]

    node_a_part = node_lookup[node_a]
    node_b_part = node_lookup[node_b]

    is_nodes_equal = node_a_part == node_b_part
    is_nodes_palm = node_a_part == "palm" or node_b_part == "palm"

    if is_nodes_equal and not is_nodes_palm:
        color_mapping = {
            "thumb": (26, 255, 26), 
            "index_finger": (0, 97, 230),
            "middle_finger": (89, 17, 212), 
            "ring_finger": (209, 108, 0), 
            "pinky": (0, 79, 153),
        }
        line_color = color_mapping[node_a_part]
    else:
        line_color = (0, 255, 255)

    return line_color

def add_img_label_text_canvas(canvas: npt.NDArray, img_label: bool) -> npt.NDArray:
    canvas_width = canvas.shape[1]
    text_canvas = np.ones((50, canvas_width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 0) if not img_label else (0, 0, 255)
    font_size = 1
    cv2.putText(
        text_canvas,
        f"img anomaly: {img_label}",
        (10, 25),
        font,
        font_scale,
        font_color,
        font_size,
    )
    return np.concatenate([text_canvas, canvas], axis=0)

def draw_hand(canvas, landmarks, mult: int, hand_label: bool):
    height, width, _ = canvas.shape
    canvas = cv2.resize(
        canvas, (mult * width, mult * height), interpolation=cv2.INTER_LINEAR
    )
    height = mult * height
    width = mult * width
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

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
    if landmarks.shape != (21, 3):
        raise ValueError("Landmarks have incorrect shape.")
    values = landmarks[:, 2]

    # Function to interpolate points between two landmarks
    def interpolate_points(p1, p2, num_points):
        return np.linspace(p1, p2, num_points + 2)  # [1:-1]

    for edge in edges:
        x1, y1 = landmarks[edge[0], :2]
        x2, y2 = landmarks[edge[1], :2]
        line_color = map_line_color(node_a=edge[0], node_b=edge[1])
        cv2.line(
            canvas,
            (int(x1 * width), int(y1 * height)),
            (int(x2 * width), int(y2 * height)),
            line_color,
            2,
        )

        num_interpolation = 0
        interpolated_x = interpolate_points(x1, x2, num_interpolation)
        interpolated_y = interpolate_points(y1, y2, num_interpolation)
        interpolated_values = interpolate_points(
            values[edge[0]], values[edge[1]], num_interpolation
        )

        for x, y, value in zip(interpolated_x, interpolated_y, interpolated_values):
            # Get the 'viridis' colormap
            viridis_colormap = plt.get_cmap("bwr")

            # Set the minimum and maximum values for your colormap
            cmap_min = -0.5
            cmap_max = 0.5

            # Create a normalization object to map values to the colormap range
            norm = Normalize(vmin=cmap_min, vmax=cmap_max)

            def value_to_rgb(value):
                rgba = viridis_colormap(norm(value))
                return tuple((np.array(rgba[:3]) * 255).astype(int))

            color_bgr = value_to_rgb(value)
            cv2.circle(
                canvas,
                (int(x * width), int(y * height)),
                2 + 1,
                (0, 255, 255),
                -1,
            )
            cv2.circle(
                canvas,
                (int(x * width), int(y * height)),
                2,
                (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])),
                -1,
            )

    canvas = add_img_label_text_canvas(canvas, hand_label)
    return canvas


@click.command()
@click.option(
    "--img-data-dir",
    required=True,
    type=click.Path(
        path_type=pathlib.Path, resolve_path=True, dir_okay=True, file_okay=False
    ),
)
@click.option(
    "--dataset-dir",
    required=True,
    type=click.Path(
        path_type=pathlib.Path, resolve_path=True, dir_okay=True, file_okay=False
    ),
)
@click.option("--dataset-name", required=True, type=str)
@click.option("--person", required=True, type=str)
@click.option("--letter", required=True, type=str)
@click.option("--image-resize-factor", type=int, default=1)
def main(
    img_data_dir: pathlib.Path,
    dataset_dir: pathlib.Path,
    dataset_name: str,
    person: str,
    letter: str,
    image_resize_factor: int,
):
    """
    ```
    python apps/visualize_videos.py \
        --img-data-dir ../../data/ \ 
        --dataset-dir data/fingerspelling5 \
        --dataset-name fingerspelling5_singlehands_sorted \
        --person D \
        --letter r
    ```
    """
    data_dir = dataset_dir
    example_person = person
    example_letter = letter

    other_data_file = data_dir / dataset_name / f"{dataset_name}.csv"
    other_data = fingerspelling5.utils.read_csv(other_data_file, filter_nans=True)
    other_selection = other_data.loc[
        (other_data["person"] == example_person)
        & (other_data["letter"] == example_letter)
    ]
    data_columns = fingerspelling5.utils.generate_hand_landmark_columns()
    raw_data = other_selection.loc[:, data_columns].values
    # TODO dangerous path!!! stacked variant vs single row variant are kinda duplicats
    raw_values_stacked = raw_data.reshape(
        -1,
        fingerspelling5.utils.mediapipe_hand_landmarks.num_nodes,
        len(fingerspelling5.utils.mediapipe_hand_landmarks.spatial_coords),
    )
    example_files = other_selection["img_file"]

    frames = []
    for example_file, example_values in zip(example_files, raw_values_stacked):
        image_path = img_data_dir.joinpath(pathlib.Path(example_file))
        img = io.imread(image_path)
        frames.append((img, example_values))

    # create cv2 window
    cv2.namedWindow("slider")
    # TODO at the moment window position is magic number
    cv2.moveWindow("slider", 600, 0)

    # use dataclas instead?
    callback_data = {
        "current_frame": 0,
        "frame_img_labels": np.zeros(len(frames), dtype=bool)
    }

    def on_trackbar(val):
        # global current_frame
        callback_data["current_frame"] = val
        img, values = frames[val]
        img = draw_hand(
            img, 
            values, 
            image_resize_factor, 
            callback_data["frame_img_labels"][callback_data["current_frame"]],
        )

        cv2.imshow("landmarks", img)

    cv2.createTrackbar("frame", "slider", 0, len(frames) - 1, on_trackbar)

    playing = False
    while True:
        if playing:

            cv2.setTrackbarPos("frame", "slider", callback_data["current_frame"])
            callback_data["current_frame"] += 1

            if callback_data["current_frame"] > len(frames):
                callback_data["current_frame"] = 0

        key = cv2.waitKey(150)

        if key == ord("s"):
            playing = False
        elif key == ord("d"):
            playing = True
        elif key == ord("q"):
            break
        elif key == ord("x"):
            current_frame = callback_data["current_frame"]
            frame_label = callback_data["frame_img_labels"][current_frame]
            callback_data["frame_img_labels"][current_frame] = not frame_label
            on_trackbar(callback_data["current_frame"])

    print("Done")

    # TODO add hotkeys for image is "bad" -> should be written to csv or similar
    # TODO if "bad" labeling available show as text somewhere?

    # TODO add creation of csv file to log 'quality' if not existent
    # TODO add 'quality' and 'view' information to csv, 'quality' via keystroke, and 'view' automatically
    # TODO add vis of 'quality' in vid


if __name__ == "__main__":
    main()
