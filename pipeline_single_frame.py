import functools
import pathlib
from typing import List, Tuple, Union

import numpy as np
from numpy import typing as npt
import pandas as pd
import torchdata


def create_path(inputs: Tuple[str, str, str, str], data_path: Union[str, pathlib.Path]) -> Tuple[str, str]:
    file_name, label = inputs # TODO use label
    file_path = pathlib.Path(data_path) / file_name
    return str(file_path), label

def load_parquet(inputs: Tuple[str, str]) -> Tuple[npt.NDArray, str]:
    file_path, label = inputs
    stacked_sequence = pd.read_parquet(file_path, columns=['x', 'y', 'z']).values
    return stacked_sequence, label

def unstack_sequence(stacked_sequence):
    num_nodes = 543
    return stacked_sequence.reshape(-1, num_nodes, 3)

def listify_frames(x):
    return [x_row for x_row in x]

def flatten_point_cloud(inputs: Tuple[npt.NDArray, str]) -> Tuple[npt.NDArray, str]:
    x, label = inputs
    return x.reshape(-1), label

def cast_float32(inputs: Tuple[npt.NDArray, str]) -> Tuple[npt.NDArray, str]:
    x, label = inputs
    return x.astype(np.float32), label

def nan_to_num(inputs: Tuple[npt.NDArray, str]) -> Tuple[npt.NDArray, str]:
    x, label = inputs
    return np.nan_to_num(x), label

def load_pointclouds(csv_file: str, data_path: pathlib.Path, batch_size: int) -> List[Tuple[npt.NDArray, str]]:
    file_path_mapper = functools.partial(create_path, data_path=data_path)

    # all available files
    datapipe = torchdata.datapipes.iter.FileOpener([str(csv_file)])
    datapipe = datapipe.parse_csv(skip_lines=1)
    datapipe = datapipe.map(file_path_mapper)

    # restructure sequence into bunch of frames
    datapipe = datapipe.map(load_parquet)
    datapipe = datapipe.map(flatten_point_cloud)
    datapipe = datapipe.map(nan_to_num)
    datapipe = datapipe.map(cast_float32)
    datapipe = datapipe.batch(batch_size=batch_size)
    datapipe = datapipe.collate()

    return datapipe


if __name__ == "__main__":
    base_path = pathlib.Path(__file__).parent
    data_path = base_path

    train_csv = data_path / "output.csv"

    fu = load_pointclouds(train_csv, data_path=data_path, batch_size=10)
    oi = list(fu)
    print('oi')
