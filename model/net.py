import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import *
from .heads import *
from .modules.cbam import CBAM
from model.modules.cfp import CFPModule
from timm.layers import trunc_normal_
import math
from model.modules.attentions import AA_kernel, ECA
from model.modules.conv_layers import Conv


class CrossAttention(nn.Module):
    def __init__(self, lower_c, higher_c) -> None:
        super().__init__()
        self.aa_1 = ECA(lower_c)
        self.aa_2 = ECA(higher_c)
        self.conv = Conv(lower_c, higher_c, 3, 1, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.attention = nn.Sequential(
            nn.Conv2d(
                in_channels=higher_c,
                out_channels=higher_c,
                kernel_size=1,
                groups=higher_c,
            ),
            nn.BatchNorm2d(higher_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=higher_c,
                out_channels=higher_c,
                kernel_size=1,
                groups=higher_c,
            ),
        )

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = F.interpolate(x1, size=x2.size()[2:], mode="bilinear")
        x = x1 + x2

        avg_pool = F.avg_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )

        max_pool = F.max_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )

        out = avg_pool + max_pool
        out = self.attention(out)
        out = F.softmax(out)

        out = x1 * out + x2 * out

        return out


class SegmentNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = custom_res2net50_v1b(pretrained=False)
        self.params = self.backbone.channels

        # self.cross_1 = CrossAttention(256, 512)
        # self.cross_2 = CrossAttention(512, 1024)
        # self.cross_3 = CrossAttention(1024, 2048)

        self.decoder = UPerHead(self.params, 128, 1)

        self.output_0 = nn.Conv2d(128, 1, 1)
        self.output_1 = nn.Conv2d(128, 1, 1)
        self.output_2 = nn.Conv2d(128, 1, 1)
        self.output_3 = nn.Conv2d(128, 1, 1)

        self.final = nn.Conv2d(5, 1, 3, 1, 1)

    def forward(self, x, warm_flag):
        enc_out = self.backbone(x)
        x0, x1, x2, x3 = enc_out

        # x1 = self.cross_1(x0, x1)
        # x2 = self.cross_2(x1, x2)
        # x3 = self.cross_3(x2, x3)

        sub_masks, global_mask = self.decoder([x0, x1, x2, x3])

        if warm_flag:

            x_d1, x_d2, x_d3, x_d4 = sub_masks

            mask0 = self.output_0(x_d1)
            mask1 = self.output_1(x_d2)
            mask2 = self.output_2(x_d3)
            mask3 = self.output_3(x_d4)

            mask0 = F.interpolate(
                mask0, size=x.size()[2:], mode="bilinear", align_corners=False
            )
            mask1 = F.interpolate(
                mask1, size=x.size()[2:], mode="bilinear", align_corners=False
            )
            mask2 = F.interpolate(
                mask2, size=x.size()[2:], mode="bilinear", align_corners=False
            )
            mask3 = F.interpolate(
                mask3, size=x.size()[2:], mode="bilinear", align_corners=False
            )
            global_mask = F.interpolate(
                global_mask, size=x.size()[2:], mode="bilinear", align_corners=False
            )
            output = self.final(
                torch.cat([mask0, mask1, mask2, mask3, global_mask], dim=1)
            )
            return [mask0, mask1, mask2, mask3, global_mask], output

        else:
            global_mask = F.interpolate(
                global_mask, size=x.size()[2:], mode="bilinear", align_corners=False
            )
            return [], global_mask


if __name__ == "__main__":
    model = SegmentNet()
    x = torch.randn(1, 3, 256, 256)
    outs, out = model(x)
    print(out.shape)
