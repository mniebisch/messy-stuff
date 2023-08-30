import pathlib
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tqdm
from mediapipe.python.solution_base import SolutionBase


def extract_hand_point_cloud(
    hand_result: SolutionBase,
) -> List[Tuple[float, float, float]]:
    hand_num_landmarks = 21
    hand_index = {"Left": 0, "Right": hand_num_landmarks}

    # hand_point_cloud structure
    # columns are x, y, z coordinates
    # first 21 rows are left hand landmarks
    # following 21 rows are right hand landmarks.

    hand_point_cloud = [(np.nan, np.nan, np.nan)] * hand_num_landmarks * 2
    if hand_result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            hand_result.multi_hand_landmarks, hand_result.multi_handedness
        ):
            handedness_label = handedness.classification[0].label
            hand_start_index = hand_index[handedness_label]
            for landmark_index, landmark in enumerate(hand_landmarks.landmark):
                point_coordinate = (landmark.x, landmark.y, landmark.z)
                hand_point_cloud[hand_start_index + landmark_index] = point_coordinate

    return hand_point_cloud


def create_column_map(
    hand_point_cloud: List[Tuple[float, float, float]]
) -> Dict[str, float]:
    if len(hand_point_cloud) != 42:
        raise ValueError

    if not all(len(point) == 3 for point in hand_point_cloud):
        raise ValueError

    column_map = {}
    for point_ind, point in enumerate(hand_point_cloud):
        hand, shift = ("left_hand", 0) if point_ind < 21 else ("right_hand", 21)
        for coord_name, coord_value in zip(("x", "y", "z"), point):
            column_name = f"{coord_name}_{hand}_{point_ind - shift}"
            column_map[column_name] = np.float32(coord_value)

    return column_map


def process_dataset(dataset_path: pathlib.Path):
    data = []

    dataset_path = dataset_path
    for person_dir in tqdm.tqdm(dataset_path.iterdir(), desc="Persons"):
        if not person_dir.is_dir():
            continue
        person = person_dir.name

        for letter_dir in tqdm.tqdm(person_dir.iterdir(), desc="Letters"):
            if not letter_dir.is_dir():
                continue
            letter = letter_dir.name

            images = [cv2.imread(str(f)) for f in letter_dir.glob("color_*")]
            mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
            )

            for image in tqdm.tqdm(images, desc="Images"):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = mp_hands.process(image_rgb)
                hand_point_cloud = extract_hand_point_cloud(results)
                column_map = create_column_map(hand_point_cloud)

                column_map["person"] = person
                column_map["letter"] = letter

                data.append(column_map)

    return pd.DataFrame(data)


if __name__ == "__main__":
    data_basepath = pathlib.Path.home() / "data"
    dataset_path = data_basepath / "fingerspelling5"

    output_path = pathlib.Path(__file__).parent / "data"
    output_file = output_path / "fingerspelling5_hands.csv"

    df = process_dataset(dataset_path=dataset_path)
    df.to_csv(output_path / "fingerspelling5_hands.csv", index=False)
    print(f"Saved to {str(output_file)}")
