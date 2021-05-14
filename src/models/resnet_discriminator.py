import torch.nn as nn

from torchvision import models
from torch.nn.utils import spectral_norm
from utils.find_size import shape_change_conv

class ResnetDiscriminator(nn.Module):
    def __init__(self, model=models.resnet18(pretrained=True), in_size=128, channels=3):
        super().__init__()
        
        self.resnet = nn.Sequential(*list(model.children()))[:-2]
        self.final_layer = nn.Sequential(
            shape_change_conv(self.resnet, in_size, 1, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.final_layer(self.resnet(input))
        