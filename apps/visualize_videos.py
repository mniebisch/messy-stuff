import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from matplotlib.colors import Normalize

from fmp.datasets import fingerspelling5


def draw_hand(canvas, landmarks):
    height, width, _ = canvas.shape
    mult = 5
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
        cv2.line(
            canvas,
            (int(x1 * width), int(y1 * height)),
            (int(x2 * width), int(y2 * height)),
            (0, 255, 255),
            1,
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
    return canvas


img_data_dir = pathlib.Path(__file__).parent.parent.parent.parent / "data"
data_dir = pathlib.Path(__file__).parent.parent / "data" / "fingerspelling5"

dataset_name = "fingerspelling5_singlehands_sorted"

# data interface
example_person = "D"
example_letter = "r"

other_data_file = data_dir / dataset_name / f"{dataset_name}.csv"
other_data = fingerspelling5.utils.read_csv(other_data_file, filter_nans=True)
other_selection = other_data.loc[
    (other_data["person"] == example_person) & (other_data["letter"] == example_letter)
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
cv2.namedWindow("muh")


def on_trackbar(val):
    global current_frame
    current_frame = val
    img, values = frames[val]
    img = draw_hand(img, values)

    cv2.imshow("blub", img)


cv2.createTrackbar("oi", "muh", 0, len(frames) - 1, on_trackbar)

playing = False
current_frame = 0
while True:
    if playing:

        cv2.setTrackbarPos("oi", "muh", current_frame)
        current_frame += 1

        if current_frame > len(frames):
            current_frame = 0

    key = cv2.waitKey(150)

    if key == ord("s"):
        playing = False
    elif key == ord("d"):
        playing = True
    elif key == ord("q"):
        break


print("Done")


# TODO add hotkeys for image is "bad" -> should be written to csv or similar
# TODO if "bad" labeling available show as text somewhere?

# TODO argparse/click/jsonargparse for command line handling
# TODO add creation of csv file to log 'quality' if not existent
# TODO add 'quality' and 'view' information to csv, 'quality' via keystroke, and 'view' automatically
# TODO add vis of 'quality' in vid
