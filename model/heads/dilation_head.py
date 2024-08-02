import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple
from model.heads.dilation_bottleneck import DilationBottleneck


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
        )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)
        out += residual
        
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class DilationHead(nn.Module):
    def __init__(self, in_channels, channel=128, scales=(1, 2, 3, 6)):
        super().__init__()
        # DilationBottleneck Module
        self.dilation_bottleneck = DilationBottleneck(in_channels[-1], channel, scales)
        # self.dilation_bottleneck = ConvModule(in_channels[-1], channel, 3)
        
        self.feature_in = nn.ModuleList()
        self.feature_out = nn.ModuleList()

        for in_ch in in_channels[:-1]:
            self.feature_in.append(ResBlock(in_ch, channel, 1))
            self.feature_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels) * channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channel, 1, 1)

    def forward(self, features_in: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.dilation_bottleneck(features_in[-1])
        features = [f]

        for i in reversed(range(len(features_in) - 1)):
            feature = self.feature_in[i](features_in[i])
            f = feature + F.interpolate(
                f, size=feature.shape[-2:], mode="bilinear", align_corners=False
            )
            features.append(self.feature_out[i](f))

        features.reverse()
        for i in range(1, len(features_in)):
            features[i] = F.interpolate(
                features[i],
                size=features[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        output = self.bottleneck(torch.cat(features, dim=1))
        output = self.conv_seg(self.dropout(output))
        return features, output
