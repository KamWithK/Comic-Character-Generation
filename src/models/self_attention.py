import torch

import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dims):
        super().__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dims, out_channels=in_dims // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dims, out_channels=in_dims // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dims, out_channels=in_dims, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        batch_size, layers, width, height = input.shape

        proj_query = self.query_conv(input).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(input).view(batch_size, -1, width * height)

        attention = self.softmax(torch.bmm(proj_query, proj_key)) # distribution of matrix-matrix product
        proj_value = self.value_conv(input).view(batch_size, -1, width * height)

        return self.gamma * torch.bmm(proj_value, attention.permute(0, 2, 1)).view(batch_size, layers, width, height) + input