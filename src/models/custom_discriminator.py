import torch.nn as nn

from torch.nn.utils import spectral_norm
from models.self_attention import SelfAttention
from itertools import starmap

class Discriminator(nn.Module):
    def __init__(self, hidden_dims=[32, 64, 128, 256, 512], final_conv_kernel=1, block=None):
        super().__init__()
        
        # Custom block support        
        if block != None: self.block = block
        
        self.main = nn.Sequential(*starmap(self.block, zip([3, *hidden_dims[:-1]], hidden_dims)))
        
        self.final_layer = nn.Sequential(
            SelfAttention(hidden_dims[-1]),
            nn.Conv2d(hidden_dims[-1], 1, final_conv_kernel),
            nn.Sigmoid()
        )
        
    def block(self, in_channels, out_channels):
        return nn.Sequential(
            # nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1),
            # nn.BatchNorm2d(in_channels),
            # nn.LeakyReLU(),

            # nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1),
            # nn.BatchNorm2d(in_channels),
            # nn.LeakyReLU(),
            
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, input):
        return self.final_layer(self.main(input))