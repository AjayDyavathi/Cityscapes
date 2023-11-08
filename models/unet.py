"""
unet.py
U-net implementation for semantic segmentation
"""
import torch
from torch import nn


activations = {
    "relu": nn.ReLU(),
}


class Encode(nn.Module):
    "Spatial Contraction of input"
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_rep=2,
                 bn=True,
                 activation="relu",
                 dropout=1.0):
        super().__init__()
        layers = nn.ModuleList()
        for _ in range(n_rep):
            layers.append(nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=(3, 3),
                                    padding="same"))
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activations.get(activation))
            in_channels = out_channels
        if dropout != 1.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        "feed forward"
        return self.block(x)


class Decode(nn.Module):
    "Spatial Expansion of input"
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_rep=2,
                 bn=True,
                 activation="relu"):
        super().__init__()
        layers = nn.ModuleList()
        self.upscale = nn.ConvTranspose2d(in_channels,
                                          out_channels,
                                          kernel_size=(2, 2),
                                          stride=2)
        for _ in range(n_rep):
            layers.append(nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=(3, 3),
                                    padding="same"))
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activations.get(activation))
            in_channels = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, encoded, x):
        "feed forward"
        x = self.upscale(x)
        x = torch.cat((encoded, x), dim=1)
        x = self.block(x)
        return x


class Unet(nn.Module):
    """U-net architecture for semantic segmentation
       https://arxiv.org/abs/1505.04597v1"""
    def __init__(self, input_channels=3, n_classes=20):
        super().__init__()
        # Encoding filter bank (takes every consecutive pair)
        enc_ch_levels = [input_channels, 16, 32, 64, 128, 256, 512]
        mid_ch = 1024
        # Decoding filter bank (reverse of encoding filter bank)
        dec_ch_levels = [mid_ch] + enc_ch_levels[-1:0:-1]
        self.maxpool = nn.MaxPool2d((2, 2), stride=2)
        self.enc_layers = nn.ModuleList()
        for in_channels, out_channels in zip(enc_ch_levels,
                                             enc_ch_levels[1:]):
            self.enc_layers.append(Encode(in_channels, out_channels))

        self.mid_layer = Encode(enc_ch_levels[-1], mid_ch)
        self.dec_layers = nn.ModuleList()
        for in_channels, out_channels in zip(dec_ch_levels,
                                             dec_ch_levels[1:]):
            self.dec_layers.append(Decode(in_channels, out_channels))

        self.channel_pool = nn.Conv2d(dec_ch_levels[-1], n_classes,
                                      kernel_size=(1, 1))

    def forward(self, x):
        "feed forward"
        # To store outputs before maxpooling
        encoded_outputs = []
        for layer in self.enc_layers:
            # Perform encoding
            x = layer(x)
            # Store encoded outputs
            encoded_outputs.append(x)
            # Perform subsampling
            x = self.maxpool(x)
        # Pass through bottleneck
        x = self.mid_layer(x)
        # Perform upsampling
        for enc_output, layer in zip(encoded_outputs[::-1], self.dec_layers):
            x = layer(enc_output, x)

        # Perform 1x1 channel pooling
        x = self.channel_pool(x)
        return x
