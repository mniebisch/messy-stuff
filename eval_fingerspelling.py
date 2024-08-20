import json
import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from numpy import typing as npt
from sklearn import metrics
from torchdata import dataloader2
from torchvision import transforms
from tqdm import tqdm

from cam_mediapipe_singlehand import number_to_letter
from pipeline_fingerspelling5 import (FlattenTriple, ReshapeToTriple, Scale,
                                      load_fingerspelling5)


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


def eval(
    model: torch.nn.Module, pipline, device: str
) -> Tuple[npt.NDArray, npt.NDArray]:
    model.eval()
    labels = []
    predictions = []
    with torch.no_grad():
        batch_iterator = tqdm(pipline)
        for landmark_batch, label_batch in batch_iterator:
            landmark_batch = landmark_batch.to(device)
            prediction = model(landmark_batch)
            prediction = prediction.detach().cpu()
            prediction_label = torch.argmax(prediction, dim=1)
            label_batch = torch.argmax(label_batch, dim=1)  # hacky hacky
            labels.append(label_batch)
            predictions.append(prediction_label)
    labels = torch.concat(labels)
    predictions = torch.concat(predictions)
    return predictions.numpy(), labels.numpy()


def get_nan_letter(df: pd.DataFrame) -> pd.Series:
    nan_samples = df.isna().any(axis=1)
    return df.loc[nan_samples, "letter"]


if __name__ == "__main__":
    input_dim = 63
    hidden_dim = 512
    output_dim = 24
    model = MLPClassifier(input_dim, hidden_dim, output_dim)
    ckpt_file = "encoder_weights.pth"
    base_path = pathlib.Path(__file__).parent
    ckpt_path = base_path / ckpt_file
    model.load_state_dict(torch.load(ckpt_path))

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the training dataset and dataloader (modify as per your data)
    data_path = pathlib.Path(__file__).parent / "data"
    fingerspelling_landmark_csv = data_path / "fingerspelling5_singlehands.csv"
    batch_size = 512
    filter_nan = True

    landmark_data = pd.read_csv(fingerspelling_landmark_csv)

    split_file = "fingerspelling_data_split.json"
    with open(split_file, "r") as f:
        split_data = json.load(f)
    train_index = split_data["train_index"]
    val_index = split_data["valid_index"]

    train_data = landmark_data.loc[train_index]
    valid_data = landmark_data.loc[val_index]

    eval_transforms = transforms.Compose([ReshapeToTriple(), Scale(), FlattenTriple()])

    eval_train_pipe = load_fingerspelling5(
        train_data,
        batch_size=batch_size,
        drop_last=False,
        filter_nan=filter_nan,
        transform=eval_transforms,
    )
    eval_train_loader = dataloader2.DataLoader2(
        eval_train_pipe, datapipe_adapter_fn=dataloader2.adapter.Shuffle(enable=False)
    )
    eval_valid_pipe = load_fingerspelling5(
        valid_data,
        batch_size=batch_size,
        drop_last=False,
        filter_nan=filter_nan,
        transform=eval_transforms,
    )
    eval_valid_loader = dataloader2.DataLoader2(
        eval_valid_pipe, datapipe_adapter_fn=dataloader2.adapter.Shuffle(enable=False)
    )

    train_pred, train_label = eval(model, eval_train_loader, device)
    valid_pred, valid_label = eval(model, eval_valid_loader, device)

    nan_dummy = torch.zeros((1, input_dim), dtype=torch.float32)
    nan_dummy = nan_dummy.to(device)
    nan_pred = model(nan_dummy)
    nan_pred = nan_pred.detach().cpu().argmax()
    nan_pred = number_to_letter(nan_pred)
    print(f"Predicted NaN letter: {nan_pred}")

    # Vis accuracy
    train_acc = metrics.accuracy_score(train_label, train_pred)
    valid_acc = metrics.accuracy_score(valid_label, valid_pred)

    print(f"Train acc: {train_acc}, Valid acc: {valid_acc}")

    # Vis NaN influence
    train_nan_letters = get_nan_letter(landmark_data.loc[train_index])
    valid_nan_letters = get_nan_letter(landmark_data.loc[val_index])
    train_nan_hist = train_nan_letters.value_counts().sort_index()
    train_nan_hist.plot(kind="bar")
    plt.title("train letter nans")
    plt.show()
    valid_nan_hist = valid_nan_letters.value_counts().sort_index()
    valid_nan_hist.plot(kind="bar")
    plt.title("valid letter nans")
    plt.show()

    # Vis confusion matrix
    letter_map = [number_to_letter(number) for number in range(24)]
    train_confusion_matrix = metrics.confusion_matrix(train_label, train_pred)
    disp_train = metrics.ConfusionMatrixDisplay(
        confusion_matrix=train_confusion_matrix, display_labels=letter_map
    )
    disp_train.plot()
    plt.show()

    valid_confusion_matrix = metrics.confusion_matrix(valid_label, valid_pred)
    disp_valid = metrics.ConfusionMatrixDisplay(
        confusion_matrix=valid_confusion_matrix, display_labels=letter_map
    )
    disp_valid.plot()
    plt.show()

    # TODO vis influence of nans
    # TODO create pipeline and perform eval for different dataset.

    print("Done")
