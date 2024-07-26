import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor, nn
from torch.nn import init
import math
from model.modules.conv_layers import Conv


class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        feat = x * y.expand_as(x)

        return feat


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class self_attn(nn.Module):
    def __init__(self, in_channels, mode="hw"):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(
            in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0
        )
        self.key_conv = Conv(
            in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0
        )
        self.value_conv = Conv(
            in_channels, in_channels, kSize=(1, 1), stride=1, padding=0
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if "h" in self.mode:
            axis *= height
        if "w" in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out


class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv(in_channel, out_channel, kSize=1, stride=1, padding=0)
        self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3), stride=1, padding=1)
        self.Hattn = self_attn(out_channel, mode="h")
        self.Wattn = self_attn(out_channel, mode="w")

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx
