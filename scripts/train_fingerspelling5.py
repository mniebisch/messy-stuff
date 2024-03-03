import pathlib

import pandas as pd
import torchvision
from sklearn import model_selection
from torch_geometric import transforms as pyg_transforms

torchvision.disable_beta_transforms_warning()

from fmp import datasets, models

if __name__ == "__main__":
    # Load Data
    data_path = pathlib.Path(__file__).parent.parent / "data"
    fingerspelling_landmark_csv = data_path / "fingerspelling5_singlehands.csv"
    landmark_data = pd.read_csv(fingerspelling_landmark_csv)

    # Create Random Group Split
    groups = landmark_data["person"]
    gss = model_selection.GroupShuffleSplit(n_splits=3, train_size=0.8)
    train_index, val_index = next(gss.split(None, None, groups))
    train_data = landmark_data.loc[train_index]
    valid_data = landmark_data.loc[val_index]

    # Setup Training Datset
    train_transforms = pyg_transforms.Compose(
        [
            pyg_transforms.NormalizeScale(),  # rethink order of transforms?
            pyg_transforms.RandomFlip(axis=0),
            pyg_transforms.RandomJitter(0.05),
            pyg_transforms.RandomRotate(degrees=20, axis=0),
            pyg_transforms.RandomRotate(degrees=20, axis=1),
            pyg_transforms.RandomRotate(degrees=20, axis=2),
        ]
    )
    train_dataset = datasets.fingerspelling5.Fingerspelling5Landmark(
        train_data, transforms=train_transforms, filter_nans=True
    )

    # Setup Model
    model = models.MLPClassifier(
        input_dim=train_dataset.num_features,
        hidden_dim=10,
        output_dim=train_dataset.num_letters,
    )
    print("Done")
