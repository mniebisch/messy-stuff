from datetime import datetime
from typing import Dict, Tuple
import pathlib

import yaml
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



def configure_optimizers(model, lr: float) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=lr)


def configure_scheduler(optimizer, t_max) -> optim.lr_scheduler.LRScheduler:
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)


def training_step(
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int, 
        model: nn.Module, 
        criterion, 
        device: torch.device,
    ):
    # hmpfs, following line is coupled to dataset! -> wayne
    landmarks, label = batch
    landmarks = landmarks.to(device)
    label = label.to(device)
    predicted_label = model(landmarks)
    loss = criterion(predicted_label, label)
    return loss


def validation_step(
        batch: Tuple[torch.Tensor, torch.Tensor], 
        model: nn.Module, 
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    landmarks, labels = batch
    landmarks = landmarks.to(device)
    predictions: torch.Tensor = model(landmarks)
    predictions = predictions.detach().cpu()

    return predictions, labels


def train_model(
        model: nn.Module, 
        optimizer: optim.Optimizer, 
        scheduler: optim.lr_scheduler.LRScheduler, 
        num_epochs: int, 
        training_loader: torch_data.DataLoader, 
        criterion: nn.Module, 
        device: torch.device,
        writer: SummaryWriter,
        validation_dataloaders: Dict[str, torch_data.DataLoader],
        validation_step_size: int,
    ):
    model.train()
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        batch_iterator = tqdm(
            training_loader, desc=f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
        )
        rolling_loss = RollingLoss()
        for batch_idx, batch in enumerate(batch_iterator):
            step = len(train_dataloader) * epoch + batch_idx
            loss = training_step(batch, batch_idx, model, criterion, device)
            batch_iterator.set_postfix({"loss": rolling_loss(loss.item())})
            writer.add_scalar("loss/train", loss, step)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        if epoch % validation_step_size == 0 and epoch != 1:
            for split, dataloader in validation_dataloaders.items():
                acc = validate_model(model, dataloader, device)
                writer.add_scalar(f"acc/{split}", acc, step)


def validate_model(
        model: nn.Module,
        data_loader: torch_data.DataLoader,
        device: torch.device,
) -> float:
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            predictions, labels = validation_step(batch, model, device)
            predictions = torch.argmax(predictions, dim=1)
            labels = torch.argmax(labels, dim=1)
            total += labels.shape[0]
            correct += (predictions == labels).sum().item()
    model.train()
    return correct / total


if __name__ == "__main__":
    config_filepath = pathlib.Path(__file__).parent.parent / "configs" / "example.yaml"
    with open(config_filepath, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    validation_step_size = config["validation_step_size"]

    lr = config["lr"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]

    log_path = pathlib.Path(config["paths"]["tensorboard_logs"])

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

    # Setup Validation
    # # Validation Transforms
    valid_transforms = pyg_transforms.Compose(
        [
            pyg_transforms.NormalizeScale(),
        ]
    )
    # # Training Data Validation
    train_valid_dataset = datasets.fingerspelling5.Fingerspelling5Landmark(
        train_data, transforms=valid_transforms, filter_nans=True
    )
    train_valid_dataloader = torch_data.DataLoader(
        train_valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )
    # # Validation Data Validation
    valid_dataset = datasets.fingerspelling5.Fingerspelling5Landmark(
        valid_data, transforms=valid_transforms, filter_nans=True
    )
    valid_dataloader = torch_data.DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )
    validation_dataloaders = {
        "train": train_valid_dataloader,
        "valid": valid_dataloader,
    }

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
        validation_dataloaders=validation_dataloaders,
        validation_step_size=validation_step_size,
    )

    writer.close()
    print("Done")


# TODO save run information (paths, id, config)
# TODO save checkpoint
# TODO track loss and some metric