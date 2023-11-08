"""
segnet.py
Segnet implementation for semantic segmentation

This core trainable segmentation engine consists of an encoder network,
a corresponding decoder network followed by a pixel-wise classification
layer. The architecture of the encoder network is topologically identical
to the 13 convolutional layers in the VGG16 network. The role of the
decoder network is to map the low resolution encoder feature maps to
full input resolution feature maps for pixel-wise classification.
The novelty of SegNet lies is in the manner in which the decoder
upsamples its lower resolution input feature map(s). Specifically,
the decoder uses pooling indices computed in the max-pooling step of
the corresponding encoder to perform non-linear upsampling. This
eliminates the need for learning to upsample.
"""
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
        self.upscale = nn.MaxUnpool2d(kernel_size=(2, 2),
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

    def forward(self, x, mp_indices, output_size):
        "feed forward"
        # Using max pool indices to perform upsampling
        x = self.upscale(input=x, indices=mp_indices,
                         output_size=output_size)
        x = self.block(x)
        return x


class Segnet(nn.Module):
    """Segnet architecture for semantic segmentation
       https://arxiv.org/abs/1511.00561
    """
    def __init__(self, input_channels=3, n_classes=20):
        super().__init__()
        # Encoding filter bank (takes every consecutive pair)
        enc_ch_levels = [input_channels, 16, 32, 64, 128, 256, 512]
        # Decoding filter bank (reverse of encoding filter bank)
        dec_ch_levels = enc_ch_levels[-1::-1]
        dec_ch_levels[-1] = n_classes
        # Max pooling layer
        self.maxpool = nn.MaxPool2d((2, 2), stride=2, return_indices=True)
        self.enc_layers = nn.ModuleList()
        for in_channels, out_channels in zip(enc_ch_levels,
                                             enc_ch_levels[1:]):
            self.enc_layers.append(Encode(in_channels, out_channels))

        self.dec_layers = nn.ModuleList()
        for in_channels, out_channels in zip(dec_ch_levels,
                                             dec_ch_levels[1:]):
            self.dec_layers.append(Decode(in_channels, out_channels))

    def forward(self, x):
        "feed forward"
        # To store outputs before maxpooling
        maxpool_indices = []
        output_sizes = []
        for layer in self.enc_layers:
            # Perform encoding
            x = layer(x)
            output_sizes.append(x.shape)
            # Perform subsampling
            x, indices = self.maxpool(x)
            maxpool_indices.append(indices)
        # Perform upsampling
        for layer, mp_idx, op_shape in zip(self.dec_layers,
                                           maxpool_indices[::-1],
                                           output_sizes[::-1]
                                           ):
            x = layer(x, mp_idx, op_shape)

        return x
