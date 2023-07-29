import pathlib

import math
import pandas as pd
import torch
import pyarrow.parquet as pq
from torchdata import dataloader2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from pipeline_frame import load_data_framewise


def process_file(row_data):
    file_path, num_frames_per_row = row_data
    file_meta = pq.read_metadata(file_path)
    return file_meta.num_rows // num_frames_per_row

def count_frames_in_csv_parallel(csv_file_path: str) -> int:
    """
    Count the total number of frames in the Parquet files specified in the CSV file using parallel processing.

    Args:
        csv_file_path (str or pathlib.Path): The path to the CSV file containing 'path' column with Parquet file paths.

    Returns:
        int: The total number of frames calculated from the Parquet files.
    """
    data_path = pathlib.Path(csv_file_path).parent

    # Read the CSV file
    train_df = pd.read_csv(csv_file_path)

    num_frames_per_row = 543

    num_workers = min(cpu_count(), len(train_df))
    with Pool(num_workers) as pool:
        row_data = [(data_path / row['path'], num_frames_per_row) for _, row in train_df.iterrows()]
        with tqdm(total=len(train_df), desc="Counting frames", unit="file") as pbar:
            results = list(tqdm(pool.imap_unordered(process_file, row_data), total=len(train_df), position=1, leave=False))

    return sum(results)


# Define the training dataset and dataloader (modify as per your data)
base_path = pathlib.Path(__file__).parent
data_path = base_path.parent / "effective-octo-potato" / "data"
train_csv = data_path / "train.csv"
batch_size = 1024
train_pipe = load_data_framewise(csv_file=train_csv, data_path=data_path, batch_size=batch_size, drop_last_batch=False)
multi_processor = dataloader2.MultiProcessingReadingService(num_workers=12)
train_loader = dataloader2.DataLoader2(
    train_pipe, 
    reading_service=multi_processor,
    datapipe_adapter_fn=dataloader2.adapter.Shuffle(enable=False)
)

# Number of frames in overall dataset (extracted from count_frames)
# num_frames = count_frames_in_csv(train_csv)
num_frames = count_frames_in_csv_parallel(train_csv)

frame_dataset = []
with tqdm(desc="Processing", unit="iter", position=0, leave=True, total=math.ceil(num_frames / batch_size)) as pbar:
    for data in train_loader:
        frame_dataset.append(data)
        pbar.update(1)

frame_dataset = torch.cat(frame_dataset)
frame_dataset_filename = "frame_dataset.pt"
torch.save(frame_dataset, base_path / "data" / frame_dataset_filename)
    