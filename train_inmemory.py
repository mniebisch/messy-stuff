import pathlib
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import wandb
import torchvision.transforms as transforms

# Define the autoencoder model with separate encoder and decoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, encoding_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(encoding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_size),
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


class InMemoryDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        sample = self.data_tensor[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

if __name__ == "__main__":
    wandb.init(project="ssl-signing")

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

    pt_file = base_path / "data" / "frame_dataset.pt"
    pt_data = torch.load(pt_file)

    transform = transforms.Compose([
        AddGaussianNoise(mean=0.0, std=0.2),
    ])

    dataset = InMemoryDataset(pt_data, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    wandb.watch(model, log="all")

    # Train the autoencoder``
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        batch_iterator = tqdm(data_loader, desc=f"Epoch: {epoch+1:03d}/{num_epochs:03d}")
        rolling_loss = None
        for data in batch_iterator:
            optimizer.zero_grad()
            # Move the input data to the GPU
            data = data.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

            if rolling_loss is None:
                rolling_loss = loss.item()
            else:
                rolling_loss = 0.9 * rolling_loss + 0.1 * loss.item()
            batch_iterator.set_postfix({"loss": rolling_loss})


    # Save the encoder weights
    model_state_dict = model.state_dict()
    torch.save(filter_state_dict_by_prefix(model_state_dict, 'encoder.'), 'encoder_weights.pth')
 