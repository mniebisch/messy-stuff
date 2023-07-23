import pathlib
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torchdata import dataloader2


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
train_loader = dataloader2.DataLoader2(train_pipe)

# Train the autoencoder
num_epochs = 0
for epoch in range(num_epochs):
    print(epoch)
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
        
    # Print the loss after each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the encoder weights
model_state_dict = model.state_dict()
torch.save(filter_state_dict_by_prefix(model_state_dict, 'encoder.'), 'encoder_weights.pth')
