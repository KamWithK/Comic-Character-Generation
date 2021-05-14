import torch.nn as nn

from torch.nn.utils import spectral_norm
from models.self_attention import SelfAttention
from itertools import starmap

class Discriminator(nn.Module):
    def __init__(self, in_size=128, channels=3, hidden_dims=[32, 64, 128, 256, 512], block=None):
        super().__init__()
        
        # Custom block support        
        if block != None: self.block = block
        
        in_dims = [channels] + hidden_dims[:-1]
        self.main = nn.Sequential(*starmap(self.block, zip(in_dims, hidden_dims)))
        
    def block(self, in_channels, out_channels):
        # Custom final layer values for better performance
        stride = 2 if out_channels != 1 else 1
        padding = 1 if out_channels != 1 else 0

        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding, bias=False)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input):
        return self.main(input)
        