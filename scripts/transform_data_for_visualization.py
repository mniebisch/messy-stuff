import pathlib

from torch_geometric import transforms as pyg_transforms
import numpy as np
from numpy import typing as npt
import pandas as pd
from torchvision.transforms import v2
import tqdm

from fmp.datasets import fingerspelling5


def create_hand_dataframe(
    hand_flat: npt.NDArray, letter: str, person: str, hand_transform, frame_id: int
) -> pd.DataFrame:
    hand = hand_transform(hand_flat)
    hand = pd.DataFrame(hand, columns=["x", "y", "z"])
    hand["letter"] = letter
    hand["person"] = person
    hand["landmark_id"] = [
        str(i) for i in fingerspelling5.utils.mediapipe_hand_landmarks.parts.all
    ]
    hand["frame_id"] = frame_id
    return hand


if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parent.parent / "data" / "fingerspelling5"
    dataset_name = "fingerspelling5_singlehands"
    dataset_dir = data_dir / dataset_name
    data_csv = dataset_dir / f"{dataset_name}.csv"

    raw_data = fingerspelling5.utils.read_csv(data_csv, filter_nans=True)
    data_module = fingerspelling5.Fingerspelling5Landmark(raw_data)

    data_transforms = v2.Compose(
        [
            data_module._setup_landmark_pre_transforms(),
            pyg_transforms.NormalizeScale(),
            fingerspelling5.utils.PyGDataUnwrapper(),
        ],
    )

    landmark_data = data_module._landmark_data.loc[:, data_module._landmark_cols].values
    landmark_data = landmark_data.astype(np.float32)
    letter_data = data_module._landmark_data.loc[:, "letter"].values
    person_data = data_module._landmark_data.loc[:, "person"].values

    vis_data = [
        create_hand_dataframe(
            hand_flat=hand_flat,
            letter=letter,
            person=person,
            hand_transform=data_transforms,
            frame_id=frame_id,
        )
        for frame_id, (hand_flat, letter, person) in enumerate(
            tqdm.tqdm(
                zip(landmark_data, letter_data, person_data), total=len(person_data)
            )
        )
    ]
    vis_data = pd.concat(vis_data)

    output_dir = dataset_dir / "vis_data"
    output_dir.mkdir(exist_ok=True)

    filename = f"{dataset_name}_vis_data.csv"
    vis_data.to_csv(output_dir / filename, index=False)
