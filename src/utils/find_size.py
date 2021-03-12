import torch

import torch.nn as nn

def model_output(model, in_channels=3, size=128, batch_size=1):
    output_shape = model(torch.randn(1, in_channels, size, size)).shape
    output_size = output_shape.numel() * batch_size

    return output_shape, output_size * batch_size

def decoder_input(hidden_dims, in_channels=3, size=128, batch_size=1):
    encoder = nn.Sequential(*[
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        ) for in_channels, out_channels in zip([3, *hidden_dims[:-1]], hidden_dims)
    ])

    output_shape = encoder(torch.randn(1, in_channels, size, size)).shape
    output_size = output_shape.numel() * batch_size

    return output_shape, output_size * batch_size

def encoder_output(hidden_dims, in_channels=3, size=1):
    decoder = nn.Sequential(*[
        nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        ) for in_channels, out_channels in zip(hidden_dims[:-1], hidden_dims[1:])
    ])

    output_shape = decoder(torch.randn(1, in_channels, size, size)).shape
    output_size = output_shape.numel()

    return output_shape, output_size
