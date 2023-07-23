import pathlib

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from sklearn.neighbors import KNeighborsClassifier

from pipeline_single_frame import load_pointclouds
from recorded_image_to_landmarks import extract_point_cloud


base_path = pathlib.Path(__file__).parent
data_path = base_path

train_csv = data_path / "output.csv"
ckpt_file = "encoder_weights.pth"
ckpt_path = base_path / ckpt_file

# Define the encoder model
class Encoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


# Set the number of features and encoding dimension
input_size = 3 * 543
encoding_dim = 128

# Create an instance of the encoder model
encoder = Encoder(input_size, encoding_dim)
encoder.load_state_dict(torch.load(ckpt_path))

pipeline = load_pointclouds(csv_file=train_csv, data_path=data_path, batch_size=20)

embedding_data = []
embedding_labels = []
encoder.eval()
with torch.no_grad():
    for pointclouds, labels in pipeline:
        embedding = encoder(pointclouds)
        embedding_data.append(embedding)
        embedding_labels.append(labels)

embedding_data = torch.cat(embedding_data).numpy()
embedding_labels = np.concatenate([np.array(label_batch) for label_batch in embedding_labels])

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(embedding_data, embedding_labels)

# Initialize the MediaPipe solutions
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

# Create a VideoCapture object to capture video from the webcam
video_capture = cv2.VideoCapture(0)  # 0 indicates the default webcam, change it to the appropriate index if you have multiple webcams

# Loop to continuously read frames from the webcam
while True:
    # Read the current frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces, facial landmarks, gestures, and pose landmarks in the frame
    results = holistic.process(frame_rgb)

    # Check if any face landmarks are detected
    if results.face_landmarks:
        # Iterate over the facial landmarks and draw them on the frame
        for landmark in results.face_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Check if any gesture landmarks are detected
    if results.pose_landmarks:
        # Iterate over the pose landmarks and draw them on the frame
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    # Check if any pose landmarks are detected
    if results.left_hand_landmarks or results.right_hand_landmarks:
        # Iterate over the gesture landmarks and draw them on the frame
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    point_cloud = extract_point_cloud(results)
    point_cloud = np.array(point_cloud, dtype=np.float32)
    point_cloud = np.nan_to_num(point_cloud)
    point_cloud = point_cloud.reshape(1, -1)
    point_cloud = torch.tensor(point_cloud)

    encoder.eval()
    with torch.no_grad():
        embedding = encoder(point_cloud)
        embedding = embedding.numpy()

    prediction = neigh.predict(embedding)

    # Display the predicted label alongside the webcam output
    predicted_label = prediction[0]
    label_position = (10, 30)
    cv2.putText(frame, f"Predicted: {predicted_label}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame in a window called "Webcam Feed"
    cv2.imshow("Webcam Feed", frame)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the window
video_capture.release()
cv2.destroyAllWindows()