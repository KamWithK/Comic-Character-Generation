import torch.nn as nn

from torch.nn.utils import spectral_norm
from models.self_attention import SelfAttention
from utils.find_size import shape_change_conv
from itertools import starmap

class Generator(nn.Module):
    def __init__(self, in_size=1, out_size=256, latent_dims=128, out_channels=3, hidden_dims=[512, 256, 128, 64, 32], block=None):
        super().__init__()
        
        # Custom block support        
        if block != None: self.block = block

        in_dims = [latent_dims] + hidden_dims[:-1]
        self.main = nn.Sequential(*starmap(self.block, zip(in_dims, hidden_dims)))

        # Correct output shape
        final_conv = shape_change_conv(self.main, in_size, out_size, latent_dims, out_channels)
        
        self.final_layer = nn.Sequential(
            # SelfAttention(hidden_dims[-1]),
            spectral_norm(final_conv),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Tanh()
        )
        
    def block(self, in_channels, out_channels):
        return nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, input):
        return self.final_layer(self.main(input))