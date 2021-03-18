import torch.nn as nn

from torch.nn.utils import spectral_norm
from models.self_attention import SelfAttention
from itertools import starmap

class Generator(nn.Module):
    def __init__(self, size=1, in_channels=3, hidden_dims=[512, 256, 128, 64, 32], block=None):
        super().__init__()
        
        # Custom block support        
        if block != None: self.block = block
        
        self.main = nn.Sequential(*starmap(self.block, zip(hidden_dims[:-1], hidden_dims[1:])))
        
        self.final_layer = nn.Sequential(
            SelfAttention(hidden_dims[-1]),
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def block(self, in_channels, out_channels):
        return nn.Sequential(
            # spectral_norm(nn.ConvTranspose2d(in_channels, out_channels=in_channels, kernel_size=1)),
            # nn.BatchNorm2d(in_channels),
            # nn.LeakyReLU(),

            # spectral_norm(nn.ConvTranspose2d(in_channels, out_channels=in_channels, kernel_size=1)),
            # nn.BatchNorm2d(in_channels),
            # nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, input):
        return self.final_layer(self.main(input))