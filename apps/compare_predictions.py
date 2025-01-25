import pathlib
import re
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import yaml

from fmp.datasets.fingerspelling5 import utils


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

    prediction["predictions"] = map_predictions_to_letters(prediction["predictions"])

    return prediction


if __name__ == "__main__":
    workspace_dir = pathlib.Path(__file__).parent.parent
    prediction_yaml = (
        pathlib.Path(__file__).parent.parent
        / "predictions"
        / "example"
        / "prediction__fingerspelling5_dummy__version_63__epoch=8-step=9.yaml"
    )

    p = prepare_predictions(prediction_yaml, workspace_dir)
    print("Done")
