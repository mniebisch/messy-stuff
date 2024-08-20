import pathlib

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as pyg_transforms
from numpy import typing as npt

from fingerspelling_to_pandas_singlehand_landmarks import (
    create_column_map,
    extract_hand_point_cloud,
)
from pipeline_fingerspelling5 import generate_hand_landmark_columns
from visualize_fingerspelling5_single_frame_cv2 import draw_hand as draw_hand_fancy
from visualize_fingerspelling5_single_frame_cv2_xz import draw_hand_xz
from visualize_fingerspelling5_single_frame_cv2_yz import draw_hand_yz

base_path = pathlib.Path(__file__).parent
data_path = base_path

train_csv = data_path / "output.csv"
ckpt_file = "encoder_weights.pth"
ckpt_path = base_path / ckpt_file


def number_to_letter(number: int) -> str:
    if number < 0:
        raise ValueError("Input number must be non-negative.")
    if number <= 8:
        return chr(ord("a") + number)
    elif number <= 24:
        return chr(ord("a") + number + 1)
    else:
        raise ValueError("Invalid input: number out of range.")


# Define the encoder model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x


def draw_hand(frame, landmarks):
    height, width, _ = frame.shape

    if landmarks.shape != (21, 3):
        raise ValueError("Landmarks have incorrect shape.")
    for i in range(21):
        x, y = int(landmarks[i, 0] * width), int(landmarks[i, 1] * height)
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
    return frame


def draw_xz(size: int, landmarks) -> npt.NDArray:
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    canvas = draw_hand_xz(canvas, landmarks)
    return canvas


def draw_yz(size: int) -> npt.NDArray:
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    return canvas


def create_canvas(size: int) -> npt.NDArray:
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    return canvas


if __name__ == "__main__":
    # Set the number of features and encoding dimension
    input_size = 3 * 21
    encoding_dim = 128

    # Create an instance of the encoder model
    input_dim = 63
    hidden_dim = 512
    output_dim = 24
    model = MLPClassifier(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))

    # Initialize the MediaPipe solutions
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )

    # Create a VideoCapture object to capture video from the webcam
    video_capture = cv2.VideoCapture(
        0
    )  # 0 indicates the default webcam, change it to the appropriate index if you have multiple webcams

    landmark_columns = generate_hand_landmark_columns()

    inference_transforms = pyg_transforms.Compose(
        [
            pyg_transforms.NormalizeScale(),
        ]
    )

    # Loop to continuously read frames from the webcam
    while True:
        # Read the current frame from the webcam
        ret, frame = video_capture.read()

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces, facial landmarks, gestures, and pose landmarks in the frame
        results = mp_hands.process(frame_rgb)
        hand_point_cloud = extract_hand_point_cloud(results)
        column_map = create_column_map(hand_point_cloud)

        point_cloud = np.array(
            [column_map[col_name] for col_name in landmark_columns], dtype=np.float32
        )
        # TODO add inference pipline
        point_cloud = np.nan_to_num(point_cloud)
        # normalization START
        point_cloud = np.reshape(point_cloud, (-1, 3))
        point_coords_raw = point_cloud
        point_mean = np.mean(point_cloud, axis=-2, keepdims=True)
        point_cloud = point_cloud - point_mean
        point_scale = (1 / np.max(np.abs(point_cloud))) * 0.999999
        point_cloud = point_cloud * point_scale
        point_coords = point_cloud
        point_cloud = np.reshape(point_cloud, (-1,))
        # normalization END
        point_cloud = point_cloud.reshape(1, -1)

        point_cloud = torch.tensor(point_cloud)

        model.eval()
        with torch.no_grad():
            prediction = model(point_cloud)

            # Number of elements you want to consider (n)
            n = 5

            # Apply softmax to get pseudo probabilities
            probabilities = F.softmax(prediction, dim=1)[0]
            # Convert tensor to NumPy array
            probabilities_np = probabilities.detach().numpy()
            prediction = probabilities.argmax()

            # Get indices of the largest n elements
            largest_indices = np.argsort(probabilities_np)[-n:][::-1]

        # show probs
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 255, 0)
        y_offset = 25

        for i, idx in enumerate(largest_indices):
            pseudo_prob = probabilities_np[idx]
            text = f"{number_to_letter(idx)}, {pseudo_prob:.4f}"
            font_size = 2 if i == 0 else 1

            cv2.putText(
                frame,
                text,
                (10, y_offset * (i + 1)),
                font,
                font_scale,
                font_color,
                font_size,
            )

        landmarks = np.full((21, 3), np.nan, dtype=np.float32)
        canvas_xz = create_canvas(frame.shape[0])
        canvas_yz = create_canvas(frame.shape[0])
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) > 1:
                raise ValueError(
                    "More than hand detected. Configure mp.Hands properly."
                )
            # Fill landnmarks with x,y,z values.
            for hand_landmarks in results.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    landmarks[i, :] = (landmark.x, landmark.y, landmark.z)

            draw_hand_fancy(frame, landmarks)
            canvas_xz = draw_hand_xz(canvas_xz, landmarks)
            canvas_yz = draw_hand_yz(canvas_yz, landmarks)
        output_frame = np.concatenate([canvas_xz, frame, canvas_yz], axis=1)

        # show additional information
        _, output_width, _ = output_frame.shape

        bottom_canvas_height = 100

        bottom_canvas = (
            np.ones((bottom_canvas_height, output_width, 3), dtype=np.uint8) * 255
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)
        y_offset = 25
        font_size = 1

        x_diff = point_coords[12, 0] - point_coords[8, 0]
        text = f"(x_12 - x_8): {x_diff:.8f}"
        cv2.putText(
            bottom_canvas,
            text,
            (100, 20),
            font,
            font_scale,
            font_color,
            font_size,
        )

        z_diff = point_coords[12, 2] - point_coords[8, 2]
        text = f"(z_12 - z_8): {z_diff:.8f}"
        cv2.putText(
            bottom_canvas,
            text,
            (100, 40),
            font,
            font_scale,
            font_color,
            font_size,
        )

        wrist = point_coords_raw[0]
        middle = point_coords[12]
        text = f"(wrist (z) {wrist[2]}, middle (z) {middle[2]}, middle - wrist {middle[2] - wrist[2]})"
        cv2.putText(
            bottom_canvas,
            text,
            (100, 60),
            font,
            font_scale,
            font_color,
            font_size,
        )

        output_frame = np.concatenate([output_frame, bottom_canvas], axis=0)

        # Display the frame in a window called "Webcam Feed"
        cv2.imshow("Webcam Feed", output_frame)

        # Wait for the 'q' key to be pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the VideoCapture object and close the window
    video_capture.release()
    cv2.destroyAllWindows()
