import pathlib
from typing import Union, Tuple

import torch
import pandas as pd
import torchdata
from torchdata.datapipes.iter import IterableWrapper

def load_from_ram(pt_file: Union[str, pathlib.Path], batch_size, drop_last_batch: bool = True):
    datapipe = IterableWrapper([pt_file])
    datapipe = datapipe.map(torch.load)
    datapipe = datapipe.shuffle(buffer_size=500000)
    datapipe = datapipe.batch(batch_size=batch_size, drop_last=drop_last_batch)
    datapipe = datapipe.collate()

    return datapipe