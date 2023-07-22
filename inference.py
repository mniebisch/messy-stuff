import torch
import torch.nn as nn

# Define the encoder model
class Encoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

# Set the number of features and encoding dimension
input_size = 3 * 543
encoding_dim = 128

# Create an instance of the encoder model
encoder = Encoder(input_size, encoding_dim)

# Load the saved encoder weights
encoder.load_state_dict(torch.load('encoder_weights.pth'))

# Example inference
input_data = torch.randn(10, input_size)  # Example input data with batch size 10
encoded_data = encoder(input_data)

# Print the encoded data
print(encoded_data)