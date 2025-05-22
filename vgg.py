import torchvision
from typing import Tuple

import numpy as np
from torch import nn
import torch

def conv_block(channels: Tuple[int, int],
               size: Tuple[int, int],
               stride: Tuple[int, int]=(1, 1)):
    """
    Create a block with N convolutional layers with ReLU activation function.
    The first layer is IN x OUT, and all others - OUT x OUT.

    Args:
        channels: (IN, OUT) - no. of input and output channels
        size: kernel size (fixed for all convolution in a block)
        stride: stride (fixed for all convolution in a block)
        N: no. of convolutional layers

    Returns:
        A sequential container of N convolutional layers.
    """
    N= 1
    
    # a single convolution + batch normalization + ReLU block
    block = lambda in_channels: nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=channels[1],
                  kernel_size=size,
                  stride=stride,
                  bias=False,
                  padding=(size[0] // 2, size[1] // 2)),
        nn.BatchNorm2d(num_features=channels[1]),
        nn.ReLU()
    )
    # create and return a sequential container of convolutional layers
    # input size = channels[0] for first block and channels[1] for all others
    return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])

class VGG16_FCN(nn.Module):
    """
    VGG16-based Fully Convolutional Network for density map regression.
    Uses VGG16 features (up to layer 23) as the downsampling path.
    """

    def __init__(self, input_filters: int = 3, **kwargs):
        """
        Args:
            input_filters: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super(VGG16_FCN, self).__init__()
        vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg16.features[:28]
        if input_filters != 3:
            # Replace first conv layer for grayscale input
            features[0] = nn.Conv2d(input_filters, 64, kernel_size=3, padding=1)
        self.down = nn.Sequential(*features)

        # Decoder: upsampling path to match input size and output 1 channel
        self.up = nn.Sequential(
            conv_block(channels=(512, 256), size=(3, 3)),
            nn.Upsample(scale_factor=2),

            conv_block(channels=(256, 128), size=(3, 3)),
            nn.Upsample(scale_factor=2),

            conv_block(channels=(128, 64), size=(3, 3)),
            nn.Upsample(scale_factor=2),

            conv_block(channels=(64, 32), size=(3, 3)),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

    def forward(self, x):
        x = self.down(x)
        x = self.up(x)
        return x
    
if __name__ == "__main__":
    # Example usage
    model = VGG16_FCN(input_filters=3)
    print(model)
    
    # Dummy input
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should be (1, 1, 224, 224) for a density map