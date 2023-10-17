"""
models.py
Model architectures for semantic segmentation

TODO: Add more models
"""
import torch
from torch import nn


activations = {
    "relu": nn.ReLU(),
}


class Encode(nn.Module):
    "Spatial Contraction of input"
    def __init__(self, in_channels, out_channels, bn=True, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn = bn
        self.activation = activation
        self.layers = nn.ModuleList()
        for _ in range(2):
            self.layers.append(nn.Conv2d(self.in_channels,
                                         self.out_channels,
                                         kernel_size=(3, 3),
                                         padding="same"))
            if self.bn:
                self.layers.append(nn.BatchNorm2d(self.out_channels))
            self.layers.append(activations.get(self.activation))
            self.in_channels = self.out_channels
        self.block = nn.Sequential(*self.layers)

    def forward(self, x):
        "feed forward"
        return self.block(x)


class Decode(nn.Module):
    "Spatial Expansion of input"
    def __init__(self, in_channels, out_channels, bn=True, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn = bn
        self.activation = activation
        self.layers = nn.ModuleList()
        self.upscale = nn.ConvTranspose2d(self.in_channels,
                                          self.out_channels,
                                          kernel_size=(2, 2),
                                          stride=2)
        for _ in range(2):
            self.layers.append(nn.Conv2d(self.in_channels,
                                         self.out_channels,
                                         kernel_size=(3, 3),
                                         padding="same"))
            if self.bn:
                self.layers.append(nn.BatchNorm2d(self.out_channels))
            self.layers.append(activations.get(self.activation))
            self.in_channels = self.out_channels
        self.block = nn.Sequential(*self.layers)

    def forward(self, encoded, x):
        "feed forward"
        x = self.upscale(x)
        x = torch.cat((encoded, x), dim=1)
        x = self.block(x)
        return x


class Unet(nn.Module):
    """U-net architecture for semantic segmentation
       https://arxiv.org/abs/1505.04597v1"""
    def __init__(self, n_classes=20):
        super().__init__()
        self.enc_ch_levels = [3, 16, 32, 64, 128, 256, 512]
        self.mid_ch = 1024
        self.dec_ch_levels = [self.mid_ch] + self.enc_ch_levels[-1:0:-1]
        self.maxpool = nn.MaxPool2d((2, 2), stride=2)
        self.enc_layers = nn.ModuleList()
        for in_channels, out_channels in zip(self.enc_ch_levels,
                                             self.enc_ch_levels[1:]):
            self.enc_layers.append(Encode(in_channels, out_channels))

        self.mid_layer = Encode(self.enc_ch_levels[-1], self.mid_ch)
        self.dec_layers = nn.ModuleList()
        for in_channels, out_channels in zip(self.dec_ch_levels,
                                             self.dec_ch_levels[1:]):
            self.dec_layers.append(Decode(in_channels, out_channels))

        self.channel_pool = nn.Conv2d(self.dec_ch_levels[-1], n_classes,
                                      kernel_size=(1, 1))

    def forward(self, x):
        "feed forward"
        encoded_outputs = []
        for layer in self.enc_layers:
            x = layer(x)
            encoded_outputs.append(x)
            x = self.maxpool(x)
        x = self.mid_layer(x)
        for enc_output, layer in zip(encoded_outputs[::-1], self.dec_layers):
            x = layer(enc_output, x)

        x = self.channel_pool(x)
        return x
