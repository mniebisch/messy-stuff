from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
from mediapipe.python.solution_base import SolutionBase

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList 

def extract_point_cloud(holistic_results: SolutionBase) -> List[Tuple[float, float, float]]:
    def extract_coords(landmarks: Optional[NormalizedLandmarkList], 
                       landmark_type: str) -> List[Tuple[float, float, float]]:
        num_landmarks_dict = {
            'face': 468,
            'pose': 33,
            'hand': 21
        }

        try:
            num_coords = num_landmarks_dict[landmark_type]
        except KeyError:
            raise ValueError(f"Invalid landmark_type: {landmark_type}. Expected 'face', 'pose', or 'hand'.")
        
        if landmarks is not None:
            if landmark_type == 'pose':
                return [(landmark.x, landmark.y, landmark.z) if landmark.visibility > 0.5
                        else (np.nan, np.nan, np.nan) for landmark in landmarks.landmark]
            else:
                return [(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]
        else:
            # If no landmarks available, return NaNs
            return [(np.nan, np.nan, np.nan)] * num_coords

    # Extract x, y, and z coordinates of face landmarks
    face_landmarks_coords = extract_coords(holistic_results.face_landmarks, 'face')

    # Extract x, y, and z coordinates of pose landmarks
    pose_landmarks_coords = extract_coords(holistic_results.pose_landmarks, 'pose')

    # Extract x, y, and z coordinates of left hand landmarks
    left_hand_landmarks_coords = extract_coords(holistic_results.left_hand_landmarks, 'hand')

    # Extract x, y, and z coordinates of right hand landmarks
    right_hand_landmarks_coords = extract_coords(holistic_results.right_hand_landmarks, 'hand')

    # Combine face, pose, and hand landmark coordinates
    point_cloud = face_landmarks_coords + pose_landmarks_coords + left_hand_landmarks_coords + right_hand_landmarks_coords

    return point_cloud


def process_images(csv_file, output_csv):
    # Create a directory to save Parquet files
    output_dir = Path('parquet_files')
    output_dir.mkdir(exist_ok=True)

    # Initialize MediaPipe Holistic model
    holistic = mp.solutions.holistic.Holistic(static_image_mode=True)

    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Initialize a list to store processed data
    processed_data = []

    for index, row in data.iterrows():
        image_path = row['image_path']
        label = row['label']

        # Load the image
        image = cv2.imread(image_path)

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract holistic landmarks
        results = holistic.process(image_rgb)
        point_cloud = extract_point_cloud(holistic_results=results)

        # Save point cloud as Parquet file
        file_name = Path(image_path).stem
        parquet_file_path = output_dir / (file_name + '.parquet')
        table = pd.DataFrame({'x': [coord[0] for coord in point_cloud],
                              'y': [coord[1] for coord in point_cloud],
                              'z': [coord[2] for coord in point_cloud]})
        table.to_parquet(parquet_file_path)

        # Add processed data to the list
        processed_data.append({'image_path': parquet_file_path, 'label': label})

    # Save processed data to a new CSV file
    processed_data_df = pd.DataFrame(processed_data)
    processed_data_df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    # Example usage
    csv_file_path = 'signing_alphabet.csv'
    output_csv_path = 'output.csv'
    process_images(csv_file_path, output_csv_path)