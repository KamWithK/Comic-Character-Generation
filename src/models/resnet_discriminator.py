import torch.nn as nn

from torchvision import models
from utils.find_size import shape_change_conv

class ResnetDiscriminator(nn.Module):
    def __init__(self, model=models.resnet18(pretrained=True), size=128, in_channels=3):
        super().__init__()
        
        self.resnet = model
        self.resnet.avgpool, self.resnet.fc = shape_change_conv(self.resnet, size, 1, in_channels, 1), nn.Sigmoid()

    def forward(self, input):
        return self.resnet(input)
        