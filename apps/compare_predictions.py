import pathlib
import re
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import yaml
from numpy import typing as npt
from sklearn import metrics as sk_metrics

from fmp.datasets.fingerspelling5 import utils

# https://plotly.com/python/parallel-categories-diagram/

# TODO adapt dataclass of predicttion writer


@dataclass
class PredictionsYAML:
    ckpt: pathlib.Path
    prediction_output: pathlib.Path


def load_predictions(
    predictions_yaml: PredictionsYAML, workspace_dir: pathlib.Path
) -> pd.DataFrame:
    return pd.read_csv(workspace_dir / predictions_yaml.prediction_output)


def load_predictions_yaml(
    predictions_hparams: pathlib.Path, workspace_dir: pathlib.Path
) -> PredictionsYAML:
    with open(predictions_hparams, "r") as hparams_file:
        hparams = yaml.safe_load(hparams_file)

    return PredictionsYAML(
        ckpt=workspace_dir / hparams["ckpt"],
        prediction_output=workspace_dir / hparams["prediction_output"],
    )


def load_training_config(
    predictions_yaml: PredictionsYAML, workspace_dir: pathlib.Path
) -> Dict:
    training_dir = predictions_yaml.ckpt.parent.parent
    training_config_file = training_dir / "config.yaml"
    training_config_file = workspace_dir / training_config_file

    with open(training_config_file, "r") as file:
        training_config = yaml.safe_load(file)

    return training_config


def load_training_datasplit(
    predictions_yaml: PredictionsYAML, workspace_dir: pathlib.Path
) -> pd.DataFrame:

    training_config = load_training_config(predictions_yaml, workspace_dir)
    if "datasplit_file" not in training_config["data"]["init_args"]:
        raise ValueError("Datasplit file not found in training config.")

    datasplit_file = training_config["data"]["init_args"]["datasplit_file"]

    return pd.read_csv(datasplit_file)


def load_training_hparams(
    predictions_yaml: PredictionsYAML, workspace_dir: pathlib.Path
) -> Dict:
    training_dir = predictions_yaml.ckpt.parent.parent
    hparams_file = training_dir / "hparams.yaml"
    hparams_file = workspace_dir / hparams_file

    with open(hparams_file, "r") as file:
        hparams = yaml.unsafe_load(file)

    return hparams


def extract_info_from_path(file_path: pathlib.Path) -> Tuple[str, str, str, int, int]:
    """
    Extract information from the given file path.

    Parameters:
        file_path (Path): The path to the file.

    Returns:
        Tuple[str, str, str, int, int]: A tuple containing experiment_name, dataset_name, training_id, epoch_num, and step_num.
    """
    pattern = re.compile(
        r".*/(?P<experiment_name>[^/]+)/prediction__(?P<dataset_name>[^_]+(?:_[^_]+)*)__"
        r"(?P<training_id>[^_]+(?:_[^_]+)*)__epoch=(?P<epoch_num>\d+)-step=(?P<step_num>\d+)\.yaml"
    )
    match = pattern.match(str(file_path))
    if not match:
        raise ValueError(f"File path '{file_path}' does not match the expected format.")

    experiment_name = match.group("experiment_name")
    dataset_name = match.group("dataset_name")
    training_id = match.group("training_id")
    epoch_num = int(match.group("epoch_num"))
    step_num = int(match.group("step_num"))

    return experiment_name, dataset_name, training_id, epoch_num, step_num


# TODO move to utils?!
def map_predictions_to_letters(predictions: pd.Series) -> pd.Series:
    return predictions.replace(
        {ind: letter for ind, letter in enumerate(utils.fingerspelling5.letters)}
    )


def calc_metrics_letterwise(y_true: npt.NDArray, y_pred: npt.NDArray) -> pd.DataFrame:
    labels = np.sort(np.unique(y_true))
    recall_label = sk_metrics.recall_score(y_true, y_pred, average=None, labels=labels)
    precision_label = sk_metrics.precision_score(
        y_true, y_pred, average=None, labels=labels
    )
    f1_label = sk_metrics.f1_score(y_true, y_pred, average=None, labels=labels)

    metrics_label = pd.DataFrame(
        {
            "recall": recall_label,
            "precision": precision_label,
            "f1": f1_label,
            "label": labels,
        }
    )
    metrics_label = pd.melt(metrics_label, id_vars=["label"])
    return metrics_label


def calc_metrics_agg(y_true: npt.NDArray, y_pred: npt.NDArray) -> pd.DataFrame:
    recall_micro = sk_metrics.recall_score(y_true, y_pred, average="micro")
    precision_micro = sk_metrics.precision_score(y_true, y_pred, average="micro")
    f1_micro = sk_metrics.f1_score(y_true, y_pred, average="micro")
    recall_macro = sk_metrics.recall_score(y_true, y_pred, average="macro")
    precision_macro = sk_metrics.precision_score(y_true, y_pred, average="macro")
    f1_macro = sk_metrics.f1_score(y_true, y_pred, average="macro")
    accuracy = sk_metrics.accuracy_score(y_true, y_pred)

    metrics_agg = pd.DataFrame(
        {
            "metric": [
                "recall_micro",
                "recall_macro",
                "precision_micro",
                "precision_macro",
                "f1_micro",
                "f1_macro",
                "accuracy",
            ],
            "value": [
                recall_micro,
                recall_macro,
                precision_micro,
                precision_macro,
                f1_micro,
                f1_macro,
                accuracy,
            ],
        }
    )

    return metrics_agg


def create_prediction_id(
    train_dir: str,
    dataset_name: str,
    training_id: str,
    ckpt_epoch_num: int,
    ckpt_step_num: int,
) -> str:
    return f"{train_dir}__{dataset_name}__{training_id}__epoch={ckpt_epoch_num}-step={ckpt_step_num}"


def prepare_predictions(
    prediction_yaml: pathlib.Path, workspace_dir: pathlib.Path
) -> pd.DataFrame:
    prediction_hparams = load_predictions_yaml(prediction_yaml, workspace_dir)
    train_hparams = load_training_hparams(prediction_hparams, workspace_dir)
    train_config = load_training_config(prediction_hparams, workspace_dir)
    datasplit = load_training_datasplit(prediction_hparams, workspace_dir)
    predictions = load_predictions(prediction_hparams, workspace_dir)

    prediction = pd.merge(datasplit, predictions, on="img_file", validate="1:1")
    train_dir, dataset_name, training_id, epoch_num, step_num = extract_info_from_path(
        prediction_yaml
    )

    prediction["train_dir"] = train_dir
    prediction["dataset_name"] = dataset_name
    prediction["training_id"] = training_id
    prediction["ckpt_epoch_num"] = epoch_num
    prediction["ckpt_step_num"] = step_num

    prediction["prediction_id"] = create_prediction_id(
        train_dir, dataset_name, training_id, epoch_num, step_num
    )

    prediction["predictions"] = map_predictions_to_letters(prediction["predictions"])

    return prediction


def compute_split_agg_metrics(prediction: pd.DataFrame) -> pd.DataFrame:
    split_agg_metrics = (
        prediction.groupby("split")
        .apply(lambda x: calc_metrics_agg(x["predictions"], x["letter"]))
        .reset_index(level=0)
    )

    copying_cols = prediction[["split", "prediction_id"]].drop_duplicates()
    split_agg_metrics = pd.merge(
        split_agg_metrics, copying_cols, on="split", how="left"
    )

    return split_agg_metrics


if __name__ == "__main__":
    workspace_dir = pathlib.Path(__file__).parent.parent
    prediction_yaml = [
        (
            pathlib.Path(__file__).parent.parent
            / "predictions"
            / "example"
            / "prediction__fingerspelling5_dummy__version_63__epoch=8-step=9.yaml"
        )
    ]

    prediction_data = [
        prepare_predictions(prediction, workspace_dir) for prediction in prediction_yaml
    ]
    prediction_split_agg_metrics = [
        compute_split_agg_metrics(prediction) for prediction in prediction_data
    ]
    prediction_split_agg_metrics = pd.concat(prediction_split_agg_metrics).reset_index(
        drop=True
    )

    fig = px.bar(
        prediction_split_agg_metrics,
        x="metric",
        y="value",
        color="split",
        facet_row="prediction_id",
        barmode="group",
    )
    fig.show()

    print("Done")
