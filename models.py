import torch
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self, n_classes=20):
        super(Unet, self).__init__()

        # Contraction
        # 256 x 512 x 3
        self.enc_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding="same")
        self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_conv1_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding="same")
        self.enc_bn1_1 = nn.BatchNorm2d(16)
        self.enc_pool1 = nn.MaxPool2d((2, 2), stride=2)
        # 128 x 256 x 16
        self.enc_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same")
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same")
        self.enc_bn2_1 = nn.BatchNorm2d(32)
        self.enc_pool2 = nn.MaxPool2d((2, 2), stride=2)
        # 64 x 128 x 32
        self.enc_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same")
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same")
        self.enc_bn3_1 = nn.BatchNorm2d(64)
        self.enc_pool3 = nn.MaxPool2d((2, 2), stride=2)
        # 32 x 64 x 64
        self.enc_conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same")
        self.enc_bn4 = nn.BatchNorm2d(128)
        self.enc_conv4_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same")
        self.enc_bn4_1 = nn.BatchNorm2d(128)
        self.enc_pool4 = nn.MaxPool2d((2, 2), stride=2)
        # 16 x 32 x 128
        self.enc_conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same")
        self.enc_bn5 = nn.BatchNorm2d(256)
        self.enc_conv5_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same")
        self.enc_bn5_1 = nn.BatchNorm2d(256)
        self.enc_pool5 = nn.MaxPool2d((2, 2), stride=2)
        # 8 x 16 x 256
        self.enc_conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding="same")
        self.enc_bn6 = nn.BatchNorm2d(512)
        self.enc_conv6_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same")
        self.enc_bn6_1 = nn.BatchNorm2d(512)
        self.enc_pool6 = nn.MaxPool2d((2, 2), stride=2)
        # 4 x 8 x 512

        # Bottleneck
        self.mid_conv1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding="same")
        self.mid_bn1 = nn.BatchNorm2d(1024)
        self.mid_conv1_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding="same")
        self.mid_bn1_1 = nn.BatchNorm2d(1024)
        # 2 x 4 x 1024

        # Expansion
        self.dec_Tconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2, 2), stride=2)
        # Skip connection -> Tconv1(512 maps) + conv4_1(512 maps) = concatenated(1024 maps)
        self.dec_conv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3, 3), padding="same")
        self.dec_bn1 = nn.BatchNorm2d(512)
        self.dec_conv1_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same")
        self.dec_bn1_1 = nn.BatchNorm2d(512)
        # 32 x 64 x 128
        self.dec_Tconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2)
        # Skip connection -> Tconv1(256 maps) + conv4_1(256 maps) = concatenated(512 maps)
        self.dec_conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding="same")
        self.dec_bn2 = nn.BatchNorm2d(256)
        self.dec_conv2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same")
        self.dec_bn2_1 = nn.BatchNorm2d(256)
        # 32 x 64 x 128
        self.dec_Tconv3= nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2)
        # Skip connection -> Tconv1(128 maps) + conv4_1(128 maps) = concatenated(256 maps)
        self.dec_conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding="same")
        self.dec_bn3 = nn.BatchNorm2d(128)
        self.dec_conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same")
        self.dec_bn3_1 = nn.BatchNorm2d(128)
        # 32 x 64 x 128
        self.dec_Tconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2)
        # Skip connection -> Tconv2(64 maps) + conv3_1(64 maps) = concatenated(128 maps)
        self.dec_conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding="same")
        self.dec_bn4 = nn.BatchNorm2d(64)
        self.dec_conv4_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same")
        self.dec_bn4_1 = nn.BatchNorm2d(64)
        # 64 x 128 x 64
        self.dec_Tconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=2)
        # Skip connection -> Tconv3(32 maps) + conv2_1(32 maps) = concatenated(64 maps)
        self.dec_conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding="same")
        self.dec_bn5 = nn.BatchNorm2d(32)
        self.dec_conv5_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same")
        self.dec_bn5_1 = nn.BatchNorm2d(32)
        # 128 x 256 x 32
        self.dec_Tconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=2)
        # Skip connection -> Tconv4(16 maps) + conv1_1(16 maps) = concatenated(32 maps)
        self.dec_conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding="same")
        self.dec_bn6 = nn.BatchNorm2d(16)
        self.dec_conv6_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding="same")
        self.dec_bn6_1 = nn.BatchNorm2d(16)
        # 256 x 512 x 16
        # Depth wise convolution using 1x1 kernel
        self.output_layer = nn.Conv2d(in_channels=16, out_channels=n_classes, kernel_size=(1, 1))

    def forward(self, x):
        # Contraction - Stage 1
        x = self.enc_conv1(x)
        x = self.enc_bn1(x)
        x = nn.ReLU()(x)
        x = self.enc_conv1_1(x)
        x1 = self.enc_bn1_1(x)
        x = nn.ReLU()(x)
        x = self.enc_pool1(x1)

        # Contraction - Stage 2
        x = self.enc_conv2(x)
        x = self.enc_bn2(x)
        x = nn.ReLU()(x)
        x = self.enc_conv2_1(x)
        x2 = self.enc_bn2_1(x)
        x = nn.ReLU()(x)
        x = self.enc_pool2(x2)

        # Contraction - Stage 3
        x = self.enc_conv3(x)
        x = self.enc_bn3(x)
        x = nn.ReLU()(x)
        x = self.enc_conv3_1(x)
        x3 = self.enc_bn3_1(x)
        x = nn.ReLU()(x)
        x = self.enc_pool3(x3)

        # Contraction - Stage 4
        x = self.enc_conv4(x)
        x = self.enc_bn4(x)
        x = nn.ReLU()(x)
        x = self.enc_conv4_1(x)
        x4 = self.enc_bn4_1(x)
        x = nn.ReLU()(x)
        x = self.enc_pool4(x4)

        # Contraction - Stage 5
        x = self.enc_conv5(x)
        x = self.enc_bn5(x)
        x = nn.ReLU()(x)
        x = self.enc_conv5_1(x)
        x5 = self.enc_bn5_1(x)
        x = nn.ReLU()(x)
        x = self.enc_pool5(x5)

        # Contraction - Stage 6
        x = self.enc_conv6(x)
        x = self.enc_bn6(x)
        x = nn.ReLU()(x)
        x = self.enc_conv6_1(x)
        x6 = self.enc_bn6_1(x)
        x = nn.ReLU()(x)
        x = self.enc_pool6(x6)

        # Bottleneck
        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = nn.ReLU()(x)
        x = self.mid_conv1_1(x)
        x = self.mid_bn1_1(x)
        x = nn.ReLU()(x)

        # Expansion - Stage 1
        x = self.dec_Tconv1(x)
        x = torch.cat((x, x6), dim=1)
        x = self.dec_conv1(x)
        x = nn.ReLU()(x)
        x = self.dec_conv1_1(x)
        x = nn.ReLU()(x)

        # Expansion - Stage 2
        x = self.dec_Tconv2(x)
        x = torch.cat((x, x5), dim=1)
        x = self.dec_conv2(x)
        x = nn.ReLU()(x)
        x = self.dec_conv2_1(x)
        x = nn.ReLU()(x)

        # Expansion - Stage 3
        x = self.dec_Tconv3(x)
        x = torch.cat((x, x4), dim=1)
        x = self.dec_conv3(x)
        x = nn.ReLU()(x)
        x = self.dec_conv3_1(x)
        x = nn.ReLU()(x)

        # Expansion - Stage 4
        x = self.dec_Tconv4(x)
        x = torch.cat((x, x3), dim=1)
        x = self.dec_conv4(x)
        x = nn.ReLU()(x)
        x = self.dec_conv4_1(x)
        x = nn.ReLU()(x)

        # Expansion - Stage 5
        x = self.dec_Tconv5(x)
        x = torch.cat((x, x2), dim=1)
        x = self.dec_conv5(x)
        x = nn.ReLU()(x)
        x = self.dec_conv5_1(x)
        x = nn.ReLU()(x)

        # Expansion - Stage 6
        x = self.dec_Tconv6(x)
        x = torch.cat((x, x1), dim=1)
        x = self.dec_conv6(x)
        x = nn.ReLU()(x)
        x = self.dec_conv6_1(x)
        x = nn.ReLU()(x)

        x = self.output_layer(x)

        return x
