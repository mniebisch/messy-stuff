import pathlib

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from fingerspelling_to_pandas_singlehand_landmarks import (
    create_column_map,
    extract_hand_point_cloud,
)
from pipeline_fingerspelling5 import generate_hand_landmark_columns

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


def draw_hand(frame, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                height, width, _ = frame.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
    return frame


if __name__ == "__main__":
    # Set the number of features and encoding dimension
    input_size = 3 * 21
    encoding_dim = 128

    # Create an instance of the encoder model
    input_dim = 63
    hidden_dim = 128
    output_dim = 24
    model = MLPClassifier(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(ckpt_path))

    # Initialize the MediaPipe solutions
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )

    # Create a VideoCapture object to capture video from the webcam
    video_capture = cv2.VideoCapture(
        0
    )  # 0 indicates the default webcam, change it to the appropriate index if you have multiple webcams

    landmark_columns = generate_hand_landmark_columns()

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
        point_cloud = np.nan_to_num(point_cloud)
        point_cloud = point_cloud.reshape(1, -1)
        point_cloud = torch.tensor(point_cloud)

        model.eval()
        with torch.no_grad():
            prediction = model(point_cloud)
            prediction = prediction.numpy()
            prediction = prediction.argmax()

        # show sample point coordinate
        sample_point = point_cloud[0, 6:9]
        sample_point = sample_point.tolist()
        samp_x, samp_y, samp_z = sample_point
        sample_point_text = f"Sample: ({samp_x:.2f},{samp_y:.2f},{samp_z:.2f})"
        sample_point_position = (10, 50)
        cv2.putText(
            frame,
            sample_point_text,
            sample_point_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        # Display the predicted label alongside the webcam output
        predicted_label = number_to_letter(prediction)
        label_position = (10, 30)
        cv2.putText(
            frame,
            f"Predicted: {predicted_label}",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        draw_hand(frame, results)
        # Display the frame in a window called "Webcam Feed"
        cv2.imshow("Webcam Feed", frame)

        # Wait for the 'q' key to be pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the VideoCapture object and close the window
    video_capture.release()
    cv2.destroyAllWindows()
