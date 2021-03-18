import torch.nn as nn

from torchvision import models
from utils.find_size import model_output

class ResnetDiscriminator(nn.Module):
    def __init__(self, model=models.resnet18(pretrained=True), size=128):
        super().__init__()
        
        resnet_output_shape = model_output(nn.Sequential(*list(model.children())[:-2]), size=size)
        final_conv_kernel, final_layer_input = resnet_output_shape[0][-1], resnet_output_shape[0][1]
        
        self.resnet = model
        self.resnet.avgpool, self.resnet.fc = nn.Conv2d(final_layer_input, 1, final_conv_kernel), nn.Sigmoid()

    def forward(self, input):
        return self.resnet(input)