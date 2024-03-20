from datetime import datetime
import pathlib

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn import model_selection
from torch.utils import data as torch_data
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import transforms as pyg_transforms
from tqdm import tqdm

torchvision.disable_beta_transforms_warning()

from fmp import datasets, models


def current_timestamp_string():
    now = datetime.now()
    timestamp_string = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # Extract milliseconds and remove trailing zeros
    return timestamp_string


class RollingLoss:
    def __init__(self, weight: float = 0.9):
        if not 0 <= weight <= 1:
            raise ValueError
        self.rolling_loss = None
        self.weight = weight

    def __call__(self, current_loss: float) -> float:
        if self.rolling_loss is None:
            self.rolling_loss = current_loss
        else:
            self.rolling_loss = (
                self.weight * current_loss + 1 - self.weight * self.rolling_loss
            )
        return self.rolling_loss

    def __repr__(self) -> str:
        return str(self.rolling_loss)



def configure_optimizers(model, lr: float):
    return optim.AdamW(model.parameters(), lr=lr)


def configure_scheduler(optimizer, t_max):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)


def training_step(batch, batch_idx, model, criterion, device):
    # hmpfs, following line is coupled to dataset! -> wayne
    landmarks, label = batch
    landmarks = landmarks.to(device)
    label = label.to(device)
    predicted_label = model(landmarks)
    loss = criterion(predicted_label, label)
    return loss


def validation_step():
    raise NotImplementedError


def train_model(
        model, 
        optimizer, 
        scheduler, 
        num_epochs, 
        training_loader, 
        criterion, 
        device,
        writer
    ):
    model.train()
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        batch_iterator = tqdm(
            training_loader, desc=f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
        )
        rolling_loss = RollingLoss()
        for batch_idx, batch in enumerate(batch_iterator):
            loss = training_step(batch, batch_idx, model, criterion, device)
            batch_iterator.set_postfix({"loss": rolling_loss(loss.item())})
            step = len(train_dataloader) * epoch + batch_idx
            writer.add_scalar("loss/train", loss, step)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()


if __name__ == "__main__":
    lr = 0.001
    num_epochs = 10
    batch_size = 128

    log_path = pathlib.Path(__file__).parent.parent / "runs"

    run_id = current_timestamp_string()

    # Load Data
    data_path = pathlib.Path(__file__).parent.parent / "data"
    # fingerspelling_landmark_csv = data_path / "fingerspelling5_singlehands.csv"
    fingerspelling_landmark_csv = data_path / "fingerspelling5_dummy_data.csv"
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
    train_dataloader = torch_data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )

    # Setup Model
    model = models.MLPClassifier(
        input_dim=train_dataset.num_features,
        hidden_dim=10,
        output_dim=train_dataset.num_letters,
    )

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = configure_optimizers(model=model, lr=lr)
    scheduler = configure_scheduler(optimizer=optimizer, t_max=num_epochs)

    writer = SummaryWriter(log_path / run_id)

    train_model(
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        num_epochs=num_epochs, 
        training_loader=train_dataloader, 
        criterion=criterion, 
        device=device,
        writer=writer,
    )

    writer.close()
    print("Done")


# TODO create run ID
# TODO read config
# TODO save run information (paths, id, config)
# TODO save checkpoint
# TODO track loss and some metric