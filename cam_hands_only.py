import pathlib
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solution_base import SolutionBase

def extract_hand_point_cloud(hand_result: SolutionBase) -> List[Tuple[float, float, float]]:
    hand_num_landmarks = 21
    hand_count = {"Left": 0, "Right": 0}
    hand_index = {"Left": 0, "Right": hand_num_landmarks}
    hand_point_cloud = [(np.nan, np.nan, np.nan)] * hand_num_landmarks * 2
    if hand_result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
            handedness_label = handedness.classification[0].label
            hand_count[handedness_label] += 1
            hand_start_index = hand_index[handedness_label]
            for landmark_index, landmark in enumerate(hand_landmarks.landmark):
                # height, width, _ = frame.shape
                point_coordinate = (landmark.x, landmark.y, landmark.z)
                # x, y = int(landmark.x * width), int(landmark.y * height)
                hand_point_cloud[hand_start_index + landmark_index] = point_coordinate

    if hand_count["Left"] > 1 or hand_count["Right"] > 1:
        raise ValueError("Each hand is only allowed to occur once.")
    return hand_point_cloud

base_path = pathlib.Path(__file__).parent
data_path = base_path

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Create a VideoCapture object to capture video from the webcam
video_capture = cv2.VideoCapture(0)  # 0 indicates the default webcam, change it to the appropriate index if you have multiple webcams

# Loop to continuously read frames from the webcam
while True:
    # Read the current frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # extra

    # Detect faces, facial landmarks, gestures, and pose landmarks in the frame
    # results = holistic.process(frame_rgb)
    results = mp_hands.process(frame_rgb)

    hand_point_cloud = extract_hand_point_cloud(results)

    # if results.multi_hand_landmarks:
    #         for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
    #             handedness_label = handedness.classification[0].label
    #             for landmark in hand_landmarks.landmark:
    #                 height, width, _ = frame.shape
    #                 x, y = int(landmark.x * width), int(landmark.y * height)
    #                 color = (0, 255, 0) if handedness_label == "Right" else (255, 0, 0)
    #                 cv2.circle(frame, (x, y), 2, color, -1)

    height, width, _ = frame.shape
    for landmark_index, (x, y, _) in enumerate(hand_point_cloud):
        if np.isnan(x):
            continue
        
        x_frame, y_frame = int(x * width), int(y * height)
        color = (0, 255, 0) if landmark_index < 21 else (255, 0, 0)
        cv2.circle(frame, (x_frame, y_frame), 2, color, -1)
            

    # Display the frame in a window called "Webcam Feed"
    cv2.imshow("Webcam Feed", frame)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the window
video_capture.release()
cv2.destroyAllWindows()