import torch.nn as nn

from torch.nn.utils import spectral_norm
from models.self_attention import SelfAttention
from itertools import starmap

class Discriminator(nn.Module):
    def __init__(self, channels=3, hidden_dims=[32, 64, 128, 256, 512], block=None):
        super().__init__()
        
        # Custom block support        
        if block != None: self.block = block
        
        in_dims = [channels] + hidden_dims[:-1]
        self.main = nn.Sequential(
            # SelfAttention(in_dims[0]),
            *starmap(self.block, zip(in_dims, hidden_dims))
        )
        
    def block(self, in_channels, out_channels):
        # Custom final layer values for better performance
        if out_channels != 1:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            return nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, input):
        return self.main(input)
        