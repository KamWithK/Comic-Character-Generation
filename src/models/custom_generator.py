import torch.nn as nn

from torch.nn.utils import spectral_norm
from models.self_attention import SelfAttention
from itertools import starmap

class Generator(nn.Module):
    def __init__(self, latent_dims=128, hidden_dims=[512, 256, 128, 64, 32], block=None):
        super().__init__()
        
        # Custom block support        
        if block != None: self.block = block

        in_dims = hidden_dims[:-1]
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dims, hidden_dims[0], kernel_size=4, stride=1),
            *starmap(self.block, zip(in_dims, hidden_dims[1:]))
        )
        
        self.final_layer = nn.Sequential(
            # SelfAttention(hidden_dims[-1]),
            nn.Tanh()
        )
        
    def block(self, in_channels, out_channels):
        if in_channels == 128:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        elif out_channels == 3:
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

    def forward(self, input):
        return self.final_layer(self.main(input))
        