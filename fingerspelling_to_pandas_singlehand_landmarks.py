import pathlib
from typing import Any, Dict, List, Tuple

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

    # hand_point_cloud structure
    # columns are x, y, z coordinates
    # first 21 rows are left hand landmarks
    # following 21 rows are right hand landmarks.

    hand_point_cloud = [(np.nan, np.nan, np.nan)] * hand_num_landmarks
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            for landmark_index, landmark in enumerate(hand_landmarks.landmark):
                point_coordinate = (landmark.x, landmark.y, landmark.z)
                hand_point_cloud[landmark_index] = point_coordinate

    return hand_point_cloud


def create_column_map(
    hand_point_cloud: List[Tuple[float, float, float]]
) -> Dict[str, float]:
    if len(hand_point_cloud) != 21:
        raise ValueError

    if not all(len(point) == 3 for point in hand_point_cloud):
        raise ValueError

    column_map = {}
    for point_ind, point in enumerate(hand_point_cloud):
        for coord_name, coord_value in zip(("x", "y", "z"), point):
            column_name = f"{coord_name}_hand_{point_ind}"
            column_map[column_name] = np.float32(coord_value)

    return column_map


def process_dataset(dataset_path: pathlib.Path):
    data = []

    dataset_path = dataset_path
    for person_dir in tqdm.tqdm(
        list(dataset_path.iterdir()), desc="Persons", leave=False
    ):
        if not person_dir.is_dir():
            continue
        person = person_dir.name

        for letter_dir in tqdm.tqdm(
            list(person_dir.iterdir()), desc="Letters", leave=False
        ):
            if not letter_dir.is_dir():
                continue
            letter = letter_dir.name

            # images = [cv2.imread(str(f)) for f in letter_dir.glob("color_*")]
            # images = [cv2.imread(str(f)) for f in letter_dir.glob("frame_*")] # for self recorded data
            mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
            )

            file_paths = list(letter_dir.glob("color_*"))
            for file_path in tqdm.tqdm(file_paths, desc="Images", leave=False):
                image = cv2.imread(str(file_path))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = mp_hands.process(image_rgb)
                hand_point_cloud = extract_hand_point_cloud(results)
                column_map: Dict[str, Any] = create_column_map(hand_point_cloud)

                column_map["person"] = person
                column_map["letter"] = letter
                column_map["img_file"] = str(pathlib.Path(*file_path.parts[-4:]))

                data.append(column_map)

    return pd.DataFrame(data)


if __name__ == "__main__":
    data_basepath = pathlib.Path.home() / "data"
    dataset_path = data_basepath / "fingerspelling5"
    # dataset_path = data_basepath / "recorded" / "asl_alphabet" / "images" # self recorded data

    output_path = pathlib.Path(__file__).parent / "data"
    output_file = output_path / "fingerspelling5_singlehands_with_filepath.csv"
    # output_file = output_path / "recorded_asl_alphabet_singlehands.csv"

    df = process_dataset(dataset_path=dataset_path)
    df.to_csv(output_file, index=False)
    print(f"Saved to {str(output_file)}")
