import json
import pathlib

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from torchdata import dataloader2
from tqdm import tqdm

import wandb
from pipeline_fingerspelling5 import load_fingerspelling5


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


def eval(model: torch.nn.Module, pipline, device: str) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        batch_iterator = tqdm(pipline)
        for landmark_batch, label_batch in batch_iterator:
            landmark_batch = landmark_batch.to(device)
            prediction = model(landmark_batch)
            prediction = prediction.detach().cpu()
            prediction_label = torch.argmax(prediction, dim=1)
            label_batch = torch.argmax(label_batch, dim=1)  # hacky hacky
            total += label_batch.shape[0]
            correct += (prediction_label == label_batch).sum().item()
    return correct / total


if __name__ == "__main__":
    wandb.init(project="fingerspelling5")

    input_dim = 63
    hidden_dim = 128
    output_dim = 24
    model = MLPClassifier(input_dim, hidden_dim, output_dim)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the training dataset and dataloader (modify as per your data)
    data_path = pathlib.Path(__file__).parent / "data"
    fingerspelling_landmark_csv = data_path / "fingerspelling5_singlehands.csv"
    batch_size = 512

    landmark_data = pd.read_csv(fingerspelling_landmark_csv)

    groups = landmark_data["person"]

    gss = GroupShuffleSplit(n_splits=3, train_size=0.8)
    train_index, val_index = next(gss.split(None, None, groups))

    split_log = {"train_index": train_index.tolist(), "valid_index": val_index.tolist()}

    with open("fingerspelling_data_split.json", "w", encoding="utf-8") as f:
        json.dump(split_log, f, ensure_ascii=False, indent=4)

    train_data = landmark_data.loc[train_index]
    valid_data = landmark_data.loc[val_index]

    filter_nan = True

    train_pipe = load_fingerspelling5(
        train_data, batch_size=batch_size, drop_last=True, filter_nan=filter_nan
    )
    train_loader = dataloader2.DataLoader2(train_pipe)

    eval_train_pipe = load_fingerspelling5(
        train_data, batch_size=batch_size, drop_last=False, filter_nan=filter_nan
    )
    eval_train_loader = dataloader2.DataLoader2(
        eval_train_pipe, datapipe_adapter_fn=dataloader2.adapter.Shuffle(enable=False)
    )
    eval_valid_pipe = load_fingerspelling5(
        valid_data, batch_size=batch_size, drop_last=False, filter_nan=filter_nan
    )
    eval_valid_loader = dataloader2.DataLoader2(
        eval_valid_pipe, datapipe_adapter_fn=dataloader2.adapter.Shuffle(enable=False)
    )

    wandb.watch(model, log="all")

    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        batch_iterator = tqdm(
            train_loader, desc=f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
        )
        rolling_loss = None
        for landmark_batch, label_batch in batch_iterator:
            optimizer.zero_grad()
            # Move the input data to the GPU
            landmark_batch = landmark_batch.to(device)
            label_batch = label_batch.to(device)

            # Forward pass
            output = model(landmark_batch)
            loss = criterion(output, label_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

            if rolling_loss is None:
                rolling_loss = loss.item()
            else:
                rolling_loss = 0.9 * rolling_loss + 0.1 * loss.item()
            batch_iterator.set_postfix({"loss": rolling_loss})
        train_acc = eval(model, eval_train_loader, device)
        valid_acc = eval(model, eval_valid_loader, device)
        print(f"Train acc: {train_acc}, Valid acc: {valid_acc}")

    # Save the encoder weights
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, "encoder_weights.pth")
