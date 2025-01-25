import pathlib
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import yaml


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
    datasplit_file = training_config["data"]["datasplit_file"]

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


if __name__ == "__main__":
    workspace_dir = pathlib.Path(__file__).parent.parent
    prediction_yaml = (
        pathlib.Path(__file__).parent.parent
        / "predictions"
        / "example"
        / "prediction__fingerspelling5_dummy__version_8__epoch=8-step=9.yaml"
    )

    prediction_hparams = load_predictions_yaml(prediction_yaml, workspace_dir)
    train_hparams = load_training_hparams(prediction_hparams, workspace_dir)
    train_config = load_training_config(prediction_hparams, workspace_dir)
    datasplit = load_training_datasplit(prediction_hparams, workspace_dir)
    predictions = load_predictions(prediction_hparams, workspace_dir)

    print("Done")
