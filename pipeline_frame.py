import functools
import pathlib
from typing import Union, Tuple

import numpy as np
from numpy import typing as npt
import pandas as pd
import torchdata


def create_path(inputs: Tuple[str, str, str, str], data_path: Union[str, pathlib.Path]) -> str:
    file_name, _, _, _ = inputs
    file_path = pathlib.Path(data_path) / file_name
    return str(file_path)

def load_parquet(file_path) -> npt.NDArray:
    stacked_sequence = pd.read_parquet(file_path, columns=['x', 'y', 'z']).values
    return stacked_sequence

def unstack_sequence(stacked_sequence):
    num_nodes = 543
    return stacked_sequence.reshape(-1, num_nodes, 3)

def listify_frames(x):
    return [x_row for x_row in x]

def flatten_point_cloud(x):
    num_frames = x.shape[0]
    return x.reshape(num_frames, -1)

def cast_float32(x: npt.NDArray) -> npt.NDArray:
    return x.astype(np.float32)

def load_data_framewise(csv_file, data_path, batch_size):
    file_path_mapper = functools.partial(create_path, data_path=data_path)

    # all available files
    datapipe = torchdata.datapipes.iter.FileOpener([str(csv_file)])
    datapipe = datapipe.parse_csv(skip_lines=1)
    datapipe = datapipe.shuffle(buffer_size=1000)
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(file_path_mapper)

    # restructure sequence into bunch of frames
    datapipe = datapipe.map(load_parquet)
    # datapipe = datapipe.in_memory_cache(size=1)
    datapipe = datapipe.map(unstack_sequence)
    datapipe = datapipe.map(flatten_point_cloud)
    datapipe = datapipe.map(np.nan_to_num)
    datapipe = datapipe.map(cast_float32)
    datapipe = datapipe.map(listify_frames)
    datapipe = datapipe.unbatch()
    
    datapipe = datapipe.shuffle(buffer_size=10000)
    datapipe = datapipe.batch(batch_size=batch_size, drop_last=True)
    datapipe = datapipe.collate()

    return datapipe


if __name__ == "__main__":
    base_path = pathlib.Path(__file__).parent
    data_path = base_path.parent / "effective-octo-potato" / "data"

    train_csv = data_path / "train.csv"

    fu = load_data_framewise(train_csv, data_path=data_path, batch_size=64)
    for batch in fu:
        print('oi')
