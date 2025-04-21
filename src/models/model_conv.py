import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # -> [16, 129, 512]
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # -> [32, 65, 256]
        self.enc3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # -> [64, 33, 128]

        # Decoder
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # -> [32, 65, 256]
        self.dec2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1) # -> [16, 129, 512]
        self.dec3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)  # -> [1, 257, 1024]

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)  # No activation: we're predicting magnitudes
        
        return x[:, :, :257, :]