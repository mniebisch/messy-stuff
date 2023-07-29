import pathlib
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pyarrow.parquet as pq
from torchdata import dataloader2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from pipeline_frame import load_data_framewise

# Define the autoencoder model with separate encoder and decoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def filter_state_dict_by_prefix(state_dict: OrderedDict[str, torch.Tensor], prefix: str) -> OrderedDict[str, torch.Tensor]:
    """
    Filters the given PyTorch state_dict to only keep keys that start with the specified prefix.

    Args:
        state_dict (OrderedDict[str, torch.Tensor]): The state dict of the PyTorch model, where keys are strings and values are PyTorch tensors.
        prefix (str): The prefix to filter keys by.

    Returns:
        OrderedDict[str, torch.Tensor]: The filtered state dict containing only keys starting with the given prefix.
    """
    filtered_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            filtered_state_dict[key] = value

    return filtered_state_dict


def count_frames_in_csv(csv_file_path: str) -> int:
    """
    Count the total number of frames in the Parquet files specified in the CSV file.

    Args:
        csv_file_path (str or pathlib.Path): The path to the CSV file containing 'path' column with Parquet file paths.

    Returns:
        int: The total number of frames calculated from the Parquet files.
    """
    data_path = pathlib.Path(csv_file_path).parent

    # Read the CSV file
    train_df = pd.read_csv(csv_file_path)

    num_frames = 0
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        file_meta = pq.read_metadata(data_path / row['path'])
        num_frames += file_meta.num_rows // 543

    return num_frames

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
        with tqdm(total=len(train_df), desc="Processing", unit="file") as pbar:
            results = list(tqdm(pool.imap_unordered(process_file, row_data), total=len(train_df), position=1, leave=False))

    return sum(results)

# Set the number of features and encoding dimension
input_size = 3 * 543
encoding_dim = 128

# Create an instance of the autoencoder model
model = Autoencoder(input_size, encoding_dim)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the training dataset and dataloader (modify as per your data)
base_path = pathlib.Path(__file__).parent
data_path = base_path.parent / "effective-octo-potato" / "data"
train_csv = data_path / "train.csv"
batch_size = 1024
train_pipe = load_data_framewise(csv_file=train_csv, data_path=data_path, batch_size=batch_size)
multi_processor = dataloader2.MultiProcessingReadingService(num_workers=12)
train_loader = dataloader2.DataLoader2(train_pipe, reading_service=multi_processor)

# Number of frames in overall dataset (extracted from count_frames)
# num_frames = count_frames_in_csv(train_csv)
num_frames = count_frames_in_csv_parallel(train_csv)

# Train the autoencoder
num_epochs = 3
for epoch in range(num_epochs):
    with tqdm(desc="Processing", unit="iter", position=0, leave=True, total=num_frames // batch_size) as pbar:
        for data in train_loader:
            img = data
            
            # Move the input data to the GPU
            img = img.to(device)

            # Forward pass
            output = model(img)
            loss = criterion(output, img)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)

            
        # Print the loss after each epoch
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the encoder weights
model_state_dict = model.state_dict()
torch.save(filter_state_dict_by_prefix(model_state_dict, 'encoder.'), 'encoder_weights.pth')
