import torch.nn as nn

from torch.nn.utils import spectral_norm
from models.self_attention import SelfAttention
from utils.find_size import shape_change_conv
from itertools import starmap

class Discriminator(nn.Module):
    def __init__(self, in_size=128, channels=3, hidden_dims=[32, 64, 128, 256, 512], block=None):
        super().__init__()
        
        # Custom block support        
        if block != None: self.block = block
        
        in_dims = [channels] + hidden_dims[:-1]
        self.main = nn.Sequential(*starmap(self.block, zip(in_dims, hidden_dims)))

        # Correct output shape
        final_conv = shape_change_conv(self.main, in_size, 1, channels, 1)
        
        self.final_layer = nn.Sequential(
            SelfAttention(hidden_dims[-1]),
            final_conv
        )
        
    def block(self, in_channels, out_channels):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input):
        return self.final_layer(self.main(input))
        