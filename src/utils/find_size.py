import torch

import torch.nn as nn

# Get models output shape/size
def model_output(model, in_channels=3, size=128, batch_size=1):
    output_shape = model(torch.randn(1, in_channels, size, size)).shape
    output_size = output_shape.numel() * batch_size

    return output_shape, output_size * batch_size

# Conv to change input to desired size
def shape_change_conv(model, in_size, final_size, latent_dims, out_channels):
    _, in_channels, output_size, _ = model_output(model, latent_dims, in_size)[0]
    corrective_stride = final_size // output_size if final_size >= output_size else output_size // final_size

    if output_size < final_size:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, output_padding=corrective_stride - 1, stride=corrective_stride)
    elif output_size >= final_size:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=corrective_stride)
