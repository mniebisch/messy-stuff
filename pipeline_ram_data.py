import pathlib
import time

import torch
from torchdata.datapipes.iter import IterableWrapper


def load_from_ram(pt_data: torch.Tensor, batch_size, drop_last_batch: bool = True):
    datapipe = IterableWrapper(pt_data, deepcopy=False)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.batch(batch_size=batch_size, drop_last=drop_last_batch)
    datapipe = datapipe.collate()

    return datapipe

if __name__ == "__main__":
    batch_size = 1024
    base_path = pathlib.Path(__file__).parent
    pt_file = base_path / "data" / "frame_dataset.pt"
    pt_data = torch.load(pt_file)
    time_start = time.time()
    train_pipe = load_from_ram(pt_data=pt_data, batch_size=batch_size)
    fu = list(train_pipe)
    run_time = time.time() - time_start
    print(run_time)
    print("oi")